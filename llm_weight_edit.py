"""Prepare, validate and evaluate one structured LLM weight-edit proposal."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch

from direct_weight_edit import evaluate_edit
from evaluate_baseline import evaluate, summarize
from gridworld import GridWorld, rollout, scenario_from_seed
from model import PolicyNetwork

MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "medium"
TARGET_SEED = 30_000
MAX_CHANGES = 16
MAX_ABSOLUTE_DELTA = 0.25
MAX_L2_CHANGE = 0.61
ALLOWED_PARAMETERS = {"network.2.weight", "network.2.bias"}
ACTION_NAMES = ["up", "down", "left", "right"]

PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "maxItems": MAX_CHANGES,
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": sorted(ALLOWED_PARAMETERS),
                    },
                    "index": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "delta": {"type": "number"},
                },
                "required": ["parameter", "index", "delta"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["changes", "reason"],
    "additionalProperties": False,
}


def load_model(checkpoint_path: Path) -> PolicyNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PolicyNetwork()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def target_observation(model: PolicyNetwork) -> tuple[torch.Tensor, dict[str, Any]]:
    scenario = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    environment = GridWorld(scenario)
    environment.position = (5, 0)
    observation = torch.from_numpy(environment.observation())
    with torch.no_grad():
        hidden = model.network[:-1](observation)
        scores = model(observation)
    return observation, {
        "scenario": scenario,
        "hidden": hidden.tolist(),
        "scores": scores.tolist(),
    }


def format_grid(scenario, position: tuple[int, int]) -> str:
    rows = []
    for row in range(scenario.size):
        cells = []
        for column in range(scenario.size):
            cell = (row, column)
            if cell == position:
                cells.append("R")
            elif cell == scenario.goal:
                cells.append("G")
            elif cell in scenario.obstacles:
                cells.append("#")
            else:
                cells.append(".")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def build_instructions() -> str:
    return (
        "You are proposing a single, conservative neural-network weight patch. "
        "The patch will be validated and tested by a separate program. "
        "Your explanation is recorded for audit, but it is not proof that the edit is correct.\n\n"
        "Task: Propose one conservative direct edit to the permitted final-layer parameters.\n"
        f"Maximum scalar changes: {MAX_CHANGES}.\n"
        f"Maximum absolute change per parameter: {MAX_ABSOLUTE_DELTA}.\n"
        f"Maximum total L2 change: {MAX_L2_CHANGE}.\n\n"
        "Choose parameter indices and deltas yourself from the information provided.\n"
        "Do not write code or propose changes outside the two permitted tensors.\n"
        "An empty changes list is valid if no defensible edit can be proposed.\n"
        "Return only an object with changes and reason."
    )


def build_prompt(model: PolicyNetwork) -> str:
    _, details = target_observation(model)
    scenario = details["scenario"]
    final_layer = model.network[-1]
    with torch.no_grad():
        weights = final_layer.weight.tolist()
        bias = final_layer.bias.tolist()

    model_summary = {
        "architecture": "3 x 7 x 7 input -> Linear(32) -> Tanh -> Linear(4)",
        "actions": dict(enumerate(ACTION_NAMES)),
        "editable_parameters": {
            "network.2.weight": {"shape": [4, 32], "values": weights},
            "network.2.bias": {"shape": [4], "values": bias},
        },
    }
    failure_summary = {
        "seed": TARGET_SEED,
        "layout": format_grid(scenario, (5, 0)),
        "current_position": [5, 0],
        "failure": "The policy reached row 5, column 0 and repeatedly chose down. The cell below is an obstacle.",
        "expert_action": "right",
        "hidden_representation": details["hidden"],
        "action_scores": dict(zip(ACTION_NAMES, details["scores"])),
    }
    return (
        f"MODEL INFORMATION\n{json.dumps(model_summary, indent=2)}\n\n"
        f"FAILURE INFORMATION\n{json.dumps(failure_summary, indent=2)}\n\n"
        f"INSTRUCTIONS\n{build_instructions()}"
    )


def validate_patch(patch: object, model: PolicyNetwork) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")
    changes = patch.get("changes")
    reason = patch.get("reason")
    if not isinstance(changes, list) or not isinstance(reason, str):
        raise ValueError("patch must contain changes and reason")
    if len(changes) > MAX_CHANGES:
        raise ValueError(f"patch contains more than {MAX_CHANGES} changes")

    final_layer = model.network[-1]
    seen: set[tuple[str, tuple[int, ...]]] = set()
    l2_squared = 0.0
    validated_changes = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("each change must be an object")
        parameter = change.get("parameter")
        index = change.get("index")
        delta = change.get("delta")
        if parameter not in ALLOWED_PARAMETERS:
            raise ValueError(f"parameter is not editable: {parameter}")
        if not isinstance(index, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in index):
            raise ValueError("index must be a list of integers")
        expected_dimensions = 2 if parameter.endswith("weight") else 1
        if len(index) != expected_dimensions:
            raise ValueError(f"{parameter} requires {expected_dimensions} indices")
        shape = tuple(final_layer.weight.shape) if parameter.endswith("weight") else tuple(final_layer.bias.shape)
        if any(value < 0 or value >= limit for value, limit in zip(index, shape)):
            raise ValueError(f"index is outside the shape of {parameter}: {index}")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError("delta must be a finite number")
        if delta == 0 or abs(delta) > MAX_ABSOLUTE_DELTA:
            raise ValueError("delta is zero or exceeds the absolute limit")
        key = (parameter, tuple(index))
        if key in seen:
            raise ValueError(f"duplicate change: {key}")
        seen.add(key)
        l2_squared += float(delta) ** 2
        validated_changes.append({"parameter": parameter, "index": index, "delta": float(delta)})
    if math.sqrt(l2_squared) > MAX_L2_CHANGE + 1e-9:
        raise ValueError("total L2 change exceeds the limit")
    return {"changes": validated_changes, "reason": reason}


def apply_patch(model: PolicyNetwork, patch: dict[str, Any]) -> tuple[PolicyNetwork, dict[str, float]]:
    edited = copy.deepcopy(model)
    with torch.no_grad():
        for change in patch["changes"]:
            tensor = edited.state_dict()[change["parameter"]]
            tensor[tuple(change["index"])] += change["delta"]
    deltas = [float(change["delta"]) for change in patch["changes"]]
    return edited, {
        "changed_parameters": len(deltas),
        "maximum_absolute_change": max((abs(delta) for delta in deltas), default=0.0),
        "l2_change": math.sqrt(sum(delta * delta for delta in deltas)),
    }


def call_model(prompt: str) -> tuple[dict[str, Any] | None, str]:
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
    raw_text = response.output_text
    return (json.loads(raw_text) if raw_text else None), raw_text


def evaluate_edited_model(original: PolicyNetwork, edited: PolicyNetwork) -> dict[str, object]:
    target = scenario_from_seed(TARGET_SEED, difficulty="difficult")

    def choose(model: PolicyNetwork):
        return lambda observation: int(model(torch.from_numpy(observation)).argmax().item())

    original_target = rollout(target, choose(original))
    edited_target = rollout(target, choose(edited))
    original_standard = evaluate(original, "standard", 20_000, 50)
    edited_standard = evaluate(edited, "standard", 20_000, 50)
    original_difficult = evaluate(original, "difficult", 30_000, 30)
    edited_difficult = evaluate(edited, "difficult", 30_000, 30)
    analytical = copy.deepcopy(original)
    analytical_report = evaluate_edit(original, analytical)
    return {
        "target": {"original": original_target, "edited": edited_target},
        "ordinary": {"original": summarize(original_standard), "edited": summarize(edited_standard)},
        "difficult_excluding_target": {
            "original": summarize([result for result in original_difficult if result["seed"] != TARGET_SEED]),
            "edited": summarize([result for result in edited_difficult if result["seed"] != TARGET_SEED]),
        },
        "analytical_baseline": analytical_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--show-prompt", action="store_true")
    mode.add_argument("--show-instructions", action="store_true")
    mode.add_argument("--request", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    prompt = build_prompt(model)
    if args.show_prompt:
        print(prompt)
        return
    if args.show_instructions:
        print("SYSTEM INSTRUCTION\nReturn only the requested structured weight patch.\n")
        print(f"USER INSTRUCTIONS\n{build_instructions()}")
        return

    patch, raw_response = call_model(prompt)
    result: dict[str, Any] = {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompt": prompt,
        "raw_response": raw_response,
    }
    if patch is None:
        result["status"] = "empty_response"
    else:
        try:
            validated = validate_patch(patch, model)
            edited, edit_metrics = apply_patch(model, validated)
            result["status"] = "accepted"
            result["patch"] = validated
            result["edit_metrics"] = edit_metrics
            result["evaluation"] = evaluate_edited_model(model, edited)
            torch.save({"model_state": edited.state_dict(), "patch": validated}, "outputs/llm_edit.pt")
        except ValueError as error:
            result["status"] = "rejected"
            result["validation_error"] = str(error)
    output_path = Path("outputs/llm_weight_edit_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
