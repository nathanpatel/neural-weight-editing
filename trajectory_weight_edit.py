"""Prepare and evaluate one multi-state trajectory weight-edit experiment."""

from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch

from evaluate_baseline import evaluate, summarize
from gridworld import ACTIONS, GridWorld, rollout, scenario_from_seed, shortest_path_actions
from llm_weight_edit import (
    ACTION_NAMES,
    MODEL,
    PATCH_SCHEMA,
    REASONING_EFFORT,
    apply_patch,
    load_model,
    parse_response,
    validate_patch,
)
from same_budget_analytical_edit import allocate_magnitudes

TARGET_SEED = 30_000
REPAIR_STATE_INDICES = [5, 10, 11, 12]
MAX_CHANGES = 16
MAX_ABSOLUTE_DELTA = 0.25
MAX_L2_CHANGE = 0.61


def remaining_distance(scenario, position: tuple[int, int]) -> int:
    queue = deque([(position, 0)])
    visited = {position}
    while queue:
        current, distance = queue.popleft()
        if current == scenario.goal:
            return distance
        for _, (row_delta, column_delta) in ACTIONS:
            next_position = (current[0] + row_delta, current[1] + column_delta)
            if (
                0 <= next_position[0] < scenario.size
                and 0 <= next_position[1] < scenario.size
                and next_position not in scenario.obstacles
                and next_position not in visited
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))
    raise ValueError("position cannot reach the goal")


def action_outcome(scenario, position: tuple[int, int], action: int) -> tuple[tuple[int, int], bool, int]:
    row_delta, column_delta = ACTIONS[action][1]
    proposed = (position[0] + row_delta, position[1] + column_delta)
    invalid = (
        not (0 <= proposed[0] < scenario.size)
        or not (0 <= proposed[1] < scenario.size)
        or proposed in scenario.obstacles
    )
    actual = position if invalid else proposed
    return actual, invalid, remaining_distance(scenario, actual)


def collect_repair_states(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Collect every objectively worse baseline decision on the expert route."""
    scenario = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    environment = GridWorld(scenario)
    expert_actions = shortest_path_actions(scenario)
    states = []

    for state_index, expert_action in enumerate(expert_actions):
        observation = torch.from_numpy(environment.observation())
        with torch.no_grad():
            hidden = model.network[:-1](observation)
            scores = model(observation)
        chosen_action = int(scores.argmax().item())
        expert_next, expert_invalid, expert_distance = action_outcome(
            scenario, environment.position, expert_action
        )
        chosen_next, chosen_invalid, chosen_distance = action_outcome(
            scenario, environment.position, chosen_action
        )
        strictly_worse = chosen_action != expert_action and (
            chosen_invalid or chosen_distance > expert_distance
        )
        if strictly_worse:
            competing_actions = [action for action in range(4) if action != expert_action]
            strongest_competitor = max(competing_actions, key=lambda action: float(scores[action]))
            states.append({
                "index": state_index,
                "position": list(environment.position),
                "expert_action": expert_action,
                "expert_action_name": ACTION_NAMES[expert_action],
                "chosen_action": chosen_action,
                "chosen_action_name": ACTION_NAMES[chosen_action],
                "strongest_competitor": strongest_competitor,
                "strongest_competitor_name": ACTION_NAMES[strongest_competitor],
                "expert_next_position": list(expert_next),
                "chosen_next_position": list(chosen_next),
                "expert_invalid": expert_invalid,
                "chosen_invalid": chosen_invalid,
                "expert_remaining_distance": expert_distance,
                "chosen_remaining_distance": chosen_distance,
                "hidden": hidden.tolist(),
                "scores": scores.tolist(),
            })
        environment.step(expert_action)

    return states


def build_prompt(model: torch.nn.Module) -> str:
    scenario = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    final_layer = model.network[-1]
    with torch.no_grad():
        weights = final_layer.weight.tolist()
        bias = final_layer.bias.tolist()
    model_information = {
        "architecture": "3 x 7 x 7 input -> Linear(32) -> Tanh -> Linear(4)",
        "actions": dict(enumerate(ACTION_NAMES)),
        "editable_parameters": {
            "network.2.weight": {"shape": [4, 32], "values": weights},
            "network.2.bias": {"shape": [4], "values": bias},
        },
    }
    repair_information = {
        "seed": TARGET_SEED,
        "goal": list(scenario.goal),
        "obstacles": sorted([list(position) for position in scenario.obstacles]),
        "states": collect_repair_states(model),
    }
    instructions = (
        "Propose one direct final-layer weight patch intended to improve the expert action "
        "at several states along this one route. Choose the parameters and numerical changes "
        "yourself. Use at most 16 scalar changes, with no individual absolute change above "
        "0.25 and total L2 change no greater than 0.61. Do not write code or use parameters "
        "outside the permitted tensors. An empty patch is valid if no defensible shared edit "
        "can be proposed. The patch will be tested on the complete route and other layouts; "
        "a local improvement is not guaranteed to improve the route. Return only JSON with "
        "changes and reason."
    )
    return (
        "You are proposing one conservative, shared neural-network weight patch. "
        "Your explanation is recorded for audit, but it is not proof that the edit is correct.\n\n"
        f"MODEL INFORMATION\n{json.dumps(model_information, indent=2)}\n\n"
        f"REPAIR TRAJECTORY INFORMATION\n{json.dumps(repair_information, indent=2)}\n\n"
        f"INSTRUCTIONS\n{instructions}"
    )


def trajectory_influences(states: list[dict[str, Any]], model: torch.nn.Module) -> list[dict[str, Any]]:
    """Build linear influences for the fixed repair-deficit objective."""
    final_layer = model.network[-1]
    influences = {
        ("network.2.weight", row, column): 0.0
        for row in range(final_layer.weight.shape[0])
        for column in range(final_layer.weight.shape[1])
    }
    influences.update({("network.2.bias", row, 0): 0.0 for row in range(final_layer.bias.shape[0])})

    for state in states:
        if state["chosen_action"] == state["expert_action"]:
            continue
        expert = state["expert_action"]
        competitor = state["strongest_competitor"]
        hidden = state["hidden"]
        for column, value in enumerate(hidden):
            influences[("network.2.weight", expert, column)] += value
            influences[("network.2.weight", competitor, column)] -= value
        influences[("network.2.bias", expert, 0)] += 1.0
        influences[("network.2.bias", competitor, 0)] -= 1.0

    candidates = []
    for (parameter, row, column), influence in influences.items():
        if influence == 0.0:
            continue
        index = [row, column] if parameter.endswith("weight") else [row]
        candidates.append({"parameter": parameter, "index": index, "influence": influence})
    return candidates


def analytical_patch(model: torch.nn.Module, states: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = trajectory_influences(states, model)
    candidates.sort(key=lambda item: (-abs(item["influence"]), item["parameter"], item["index"]))
    selected = candidates[:MAX_CHANGES]
    magnitudes = allocate_magnitudes([abs(item["influence"]) for item in selected])
    changes = [
        {
            "parameter": item["parameter"],
            "index": item["index"],
            "delta": float(torch.sign(torch.tensor(item["influence"])).item() * magnitude),
        }
        for item, magnitude in zip(selected, magnitudes)
    ]
    patch = validate_patch({"changes": changes, "reason": "Maximum fixed repair-deficit improvement across the selected states."}, model)
    return {
        "patch": patch,
        "active_states": [state["index"] for state in states if state["chosen_action"] != state["expert_action"]],
        "objective_improvement": sum(
            change["delta"] * item["influence"]
            for change, item in zip(patch["changes"], selected)
        ),
    }


def selected_state_summary(model: torch.nn.Module, states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for state in states:
        hidden = torch.tensor(state["hidden"])
        with torch.no_grad():
            scores = model.network[-1](hidden)
        expert = state["expert_action"]
        competitor = state["strongest_competitor"]
        summaries.append({
            "index": state["index"],
            "position": state["position"],
            "expert_action": state["expert_action_name"],
            "chosen_action": ACTION_NAMES[int(scores.argmax().item())],
            "scores": scores.tolist(),
            "expert_margin_vs_original_competitor": float(scores[expert] - scores[competitor]),
        })
    return summaries


def compare_model(original: torch.nn.Module, edited: torch.nn.Module, states: list[dict[str, Any]]) -> dict[str, Any]:
    target = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    original_standard = evaluate(original, "standard", 20_000, 50)
    edited_standard = evaluate(edited, "standard", 20_000, 50)
    original_difficult = evaluate(original, "difficult", 30_000, 30)
    edited_difficult = evaluate(edited, "difficult", 30_000, 30)

    def changes(before, after):
        before_by_seed = {result["seed"]: result for result in before}
        after_by_seed = {result["seed"]: result for result in after}
        return {
            "new_successes": sorted(seed for seed in before_by_seed if not before_by_seed[seed]["success"] and after_by_seed[seed]["success"]),
            "regressions": sorted(seed for seed in before_by_seed if before_by_seed[seed]["success"] and not after_by_seed[seed]["success"]),
        }

    difficult_original = [result for result in original_difficult if result["seed"] != TARGET_SEED]
    difficult_edited = [result for result in edited_difficult if result["seed"] != TARGET_SEED]
    return {
        "repair_states": selected_state_summary(edited, states),
        "target_route": rollout(target, lambda observation: int(edited(torch.from_numpy(observation)).argmax().item())),
        "ordinary": {
            "summary": summarize(edited_standard),
            "changes_vs_original": changes(original_standard, edited_standard),
        },
        "difficult_excluding_target": {
            "summary": summarize(difficult_edited),
            "changes_vs_original": changes(difficult_original, difficult_edited),
        },
    }


def call_model(prompt: str) -> tuple[dict[str, Any] | None, str]:
    """Make exactly one structured-output request using OPENAI_API_KEY."""
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=[
            {"role": "system", "content": "Return only the requested structured weight patch."},
            {"role": "user", "content": prompt},
        ],
        text={"format": {"type": "json_schema", "name": "weight_patch", "strict": True, "schema": PATCH_SCHEMA}},
    )
    return parse_response(response)


def save_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show-prompt", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--request", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--llm-result", type=Path, default=Path("outputs/trajectory_llm_result.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/trajectory_weight_edit_comparison.json"))
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    states = collect_repair_states(model)
    if args.show_prompt:
        if [state["index"] for state in states] != REPAIR_STATE_INDICES:
            raise RuntimeError("original checkpoint produced an unexpected repair-state set")
        print(build_prompt(model))
        return

    if args.request:
        if [state["index"] for state in states] != REPAIR_STATE_INDICES:
            raise RuntimeError("original checkpoint produced an unexpected repair-state set")
        prompt = build_prompt(model)
        result: dict[str, Any] = {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "prompt": prompt,
        }
        try:
            patch, raw_response = call_model(prompt)
        except ValueError as error:
            result.update({"status": "invalid_response", "error": str(error), "raw_response": ""})
            save_result(args.llm_result, result)
            print(json.dumps(result, indent=2))
            return
        except Exception as error:
            result.update({
                "status": "request_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "raw_response": "",
            })
            save_result(args.llm_result, result)
            print(json.dumps(result, indent=2))
            return

        result["raw_response"] = raw_response
        if patch is None:
            result["status"] = "empty_response"
        else:
            try:
                validated = validate_patch(patch, model)
                edited, edit_metrics = apply_patch(model, validated)
                result["status"] = "accepted"
                result["patch"] = validated
                result["edit_metrics"] = edit_metrics
                result["evaluation"] = compare_model(model, edited, states)
                torch.save({"model_state": edited.state_dict(), "patch": validated}, "outputs/trajectory_llm_edit.pt")
            except ValueError as error:
                result["status"] = "rejected"
                result["validation_error"] = str(error)
        result["original_model_unchanged"] = all(
            torch.equal(value, model.state_dict()[name]) for name, value in original_state.items()
        )
        save_result(args.llm_result, result)
        print(json.dumps(result, indent=2))
        return

    analytical = analytical_patch(model, states)
    analytical_model, analytical_metrics = apply_patch(model, analytical["patch"])
    saved_llm = json.loads(args.llm_result.read_text(encoding="utf-8"))
    if saved_llm.get("status") == "accepted" and "patch" in saved_llm:
        llm_patch = validate_patch(saved_llm["patch"], model)
        llm_model, llm_metrics = apply_patch(model, llm_patch)
        llm_report: dict[str, Any] = {
            "status": "accepted",
            "patch": llm_patch,
            "edit_metrics": llm_metrics,
            "comparison": compare_model(model, llm_model, states),
        }
    else:
        llm_report = {
            "status": saved_llm.get("status", "unknown"),
            "error": saved_llm.get("error") or saved_llm.get("validation_error"),
            "comparison": None,
        }
    report = {
        "settings": {
            "target_seed": TARGET_SEED,
            "repair_state_indices": REPAIR_STATE_INDICES,
            "maximum_scalar_changes": MAX_CHANGES,
            "maximum_absolute_delta": MAX_ABSOLUTE_DELTA,
            "maximum_l2_change": MAX_L2_CHANGE,
            "objective": "Equal-weight sum of margin improvements only at initially incorrect selected states, against each state's strongest original competitor.",
        },
        "repair_states": states,
        "llm": llm_report,
        "analytical": {
            "patch": analytical["patch"],
            "edit_metrics": analytical_metrics,
            "active_states": analytical["active_states"],
            "objective_improvement": analytical["objective_improvement"],
            "comparison": compare_model(model, analytical_model, states),
        },
        "original_model_unchanged": all(torch.equal(value, model.state_dict()[name]) for name, value in original_state.items()),
    }
    save_result(args.report, report)
    torch.save({"model_state": analytical_model.state_dict(), "patch": analytical["patch"]}, "outputs/trajectory_analytical_edit.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
