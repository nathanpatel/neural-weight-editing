"""Find and evaluate the strongest same-budget final-layer edit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from evaluate_baseline import evaluate, summarize
from gridworld import GridWorld, rollout, scenario_from_seed
from llm_weight_edit import apply_patch, load_model, validate_patch
from model import PolicyNetwork

TARGET_SEED = 30_000
MAX_CHANGES = 16
MAX_ABSOLUTE_DELTA = 0.25
MAX_L2_CHANGE = 0.61
RIGHT_ACTION = 3
DOWN_ACTION = 1


def target_hidden(model: PolicyNetwork) -> torch.Tensor:
    scenario = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    environment = GridWorld(scenario)
    environment.position = (5, 0)
    observation = torch.from_numpy(environment.observation())
    with torch.no_grad():
        return model.network[:-1](observation)


def candidate_influences(hidden: torch.Tensor) -> list[dict[str, Any]]:
    """Return final-layer scalars and their influence on right minus down."""
    candidates = []
    for column, value in enumerate(hidden.tolist()):
        candidates.append({
            "parameter": "network.2.weight",
            "index": [RIGHT_ACTION, column],
            "influence": float(value),
        })
        candidates.append({
            "parameter": "network.2.weight",
            "index": [DOWN_ACTION, column],
            "influence": -float(value),
        })
    candidates.extend([
        {
            "parameter": "network.2.bias",
            "index": [RIGHT_ACTION],
            "influence": 1.0,
        },
        {
            "parameter": "network.2.bias",
            "index": [DOWN_ACTION],
            "influence": -1.0,
        },
    ])
    return candidates


def allocate_magnitudes(
    influences: list[float],
    *,
    maximum_absolute_change: float = MAX_ABSOLUTE_DELTA,
    maximum_l2_change: float = MAX_L2_CHANGE,
) -> list[float]:
    """Maximise a positive linear objective under box and L2 limits.

    The KKT conditions give y_i = min(a, lambda * c_i), where c_i are
    positive influence magnitudes and y_i are the change magnitudes.
    """
    if not influences:
        return []
    if any(influence <= 0 for influence in influences):
        raise ValueError("influences must be positive")

    limit_squared = maximum_l2_change**2
    if len(influences) * maximum_absolute_change**2 <= limit_squared:
        return [maximum_absolute_change] * len(influences)

    lower = 0.0
    upper = maximum_absolute_change / min(influences)
    for _ in range(100):
        multiplier = (lower + upper) / 2.0
        magnitudes = [min(maximum_absolute_change, multiplier * influence) for influence in influences]
        if sum(magnitude**2 for magnitude in magnitudes) < limit_squared:
            lower = multiplier
        else:
            upper = multiplier
    multiplier = (lower + upper) / 2.0
    return [min(maximum_absolute_change, multiplier * influence) for influence in influences]


def optimal_patch(model: PolicyNetwork) -> dict[str, Any]:
    """Find the globally optimal patch for the right-minus-down objective."""
    candidates = candidate_influences(target_hidden(model))
    candidates.sort(key=lambda item: (-abs(item["influence"]), item["parameter"], item["index"]))
    selected = candidates[:MAX_CHANGES]
    magnitudes = allocate_magnitudes([abs(item["influence"]) for item in selected])
    changes = [
        {
            "parameter": item["parameter"],
            "index": item["index"],
            "delta": math.copysign(magnitude, item["influence"]),
        }
        for item, magnitude in zip(selected, magnitudes)
    ]
    patch = validate_patch({"changes": changes, "reason": "Maximum right-minus-down improvement under the fixed budget."}, model)
    return {
        "patch": patch,
        "selected_influences": [item["influence"] for item in selected],
        "objective_improvement": sum(
            change["delta"] * item["influence"]
            for change, item in zip(patch["changes"], selected)
        ),
    }


def choose(model: PolicyNetwork):
    return lambda observation: int(model(torch.from_numpy(observation)).argmax().item())


def action_summary(model: PolicyNetwork) -> dict[str, Any]:
    scenario = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    environment = GridWorld(scenario)
    environment.position = (5, 0)
    observation = torch.from_numpy(environment.observation())
    with torch.no_grad():
        scores = model(observation)
    return {
        "scores": scores.tolist(),
        "preferred_action": int(scores.argmax().item()),
    }


def compare_model(original: PolicyNetwork, edited: PolicyNetwork) -> dict[str, Any]:
    target = scenario_from_seed(TARGET_SEED, difficulty="difficult")
    original_standard = evaluate(original, "standard", 20_000, 50)
    edited_standard = evaluate(edited, "standard", 20_000, 50)
    original_difficult = evaluate(original, "difficult", 30_000, 30)
    edited_difficult = evaluate(edited, "difficult", 30_000, 30)

    def changed_successes(before, after):
        before_by_seed = {result["seed"]: result for result in before}
        after_by_seed = {result["seed"]: result for result in after}
        helped = sorted(
            seed for seed in before_by_seed
            if not before_by_seed[seed]["success"] and after_by_seed[seed]["success"]
        )
        regressed = sorted(
            seed for seed in before_by_seed
            if before_by_seed[seed]["success"] and not after_by_seed[seed]["success"]
        )
        return {"new_successes": helped, "regressions": regressed}

    return {
        "local_action": action_summary(edited),
        "target": rollout(target, choose(edited)),
        "ordinary": {
            "summary": summarize(edited_standard),
            "changes_vs_original": changed_successes(original_standard, edited_standard),
        },
        "difficult_excluding_target": {
            "summary": summarize([result for result in edited_difficult if result["seed"] != TARGET_SEED]),
            "changes_vs_original": changed_successes(
                [result for result in original_difficult if result["seed"] != TARGET_SEED],
                [result for result in edited_difficult if result["seed"] != TARGET_SEED],
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--llm-result", type=Path, default=Path("outputs/llm_weight_edit_result.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/same_budget_analytical_result.json"))
    args = parser.parse_args()

    original = load_model(args.checkpoint)
    original_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    best = optimal_patch(original)
    analytical, analytical_metrics = apply_patch(original, best["patch"])
    original_scores = action_summary(original)["scores"]
    analytical_scores = action_summary(analytical)["scores"]
    original_gap = original_scores[RIGHT_ACTION] - original_scores[DOWN_ACTION]
    analytical_gap = analytical_scores[RIGHT_ACTION] - analytical_scores[DOWN_ACTION]

    saved_llm = json.loads(args.llm_result.read_text(encoding="utf-8"))
    llm_patch = validate_patch(saved_llm["patch"], original)
    llm, llm_metrics = apply_patch(original, llm_patch)

    report = {
        "settings": {
            "target_seed": TARGET_SEED,
            "maximum_scalar_changes": MAX_CHANGES,
            "maximum_absolute_delta": MAX_ABSOLUTE_DELTA,
            "maximum_l2_change": MAX_L2_CHANGE,
        },
        "original": {"local_action": action_summary(original), "target": compare_model(original, original)["target"]},
        "llm": {
            "patch": llm_patch,
            "edit_metrics": llm_metrics,
            "local_action": action_summary(llm),
            "comparison": compare_model(original, llm),
        },
        "same_budget_analytical": {
            "patch": best["patch"],
            "edit_metrics": analytical_metrics,
            "objective_improvement": best["objective_improvement"],
            "original_right_minus_down": original_gap,
            "maximum_right_minus_down": analytical_gap,
            "right_is_highest": analytical_scores.index(max(analytical_scores)) == RIGHT_ACTION,
            "selected_influences": best["selected_influences"],
            "local_action": action_summary(analytical),
            "comparison": compare_model(original, analytical),
        },
        "original_model_unchanged": all(
            torch.equal(value, original.state_dict()[name])
            for name, value in original_state.items()
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"model_state": analytical.state_dict(), "patch": best["patch"]}, "outputs/same_budget_analytical_edit.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
