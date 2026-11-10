"""Evaluate the baseline on ordinary and deliberately difficult layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gridworld import rollout, scenario_from_seed
from model import PolicyNetwork


def evaluate(model: PolicyNetwork, difficulty: str, start_seed: int, count: int) -> list[dict[str, object]]:
    scenarios = [
        scenario_from_seed(start_seed + index, difficulty=difficulty)
        for index in range(count)
    ]
    model.eval()
    with torch.no_grad():
        return [
            rollout(
                scenario,
                lambda observation: model(torch.from_numpy(observation)).argmax().item(),
            )
            for scenario in scenarios
        ]


def summarize(results: list[dict[str, object]]) -> dict[str, float]:
    return {
        "scenario_count": float(len(results)),
        "success_rate": sum(bool(result["success"]) for result in results) / len(results),
        "mean_steps": sum(int(result["steps"]) for result in results) / len(results),
        "mean_invalid_moves": sum(int(result["invalid_moves"]) for result in results)
        / len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_evaluation.json"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PolicyNetwork()
    model.load_state_dict(checkpoint["model_state"])
    standard = evaluate(model, "standard", start_seed=20_000, count=50)
    difficult = evaluate(model, "difficult", start_seed=30_000, count=30)
    report = {
        "checkpoint_settings": checkpoint["settings"],
        "standard": {"summary": summarize(standard), "scenarios": standard},
        "difficult": {"summary": summarize(difficult), "scenarios": difficult},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"standard": report["standard"]["summary"], "difficult": report["difficult"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
