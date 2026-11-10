"""Plot one successful and one failed baseline episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gridworld import GridWorld, Position, rollout, scenario_from_seed, shortest_path_actions
from model import PolicyNetwork


def policy_action(model: PolicyNetwork, observation: np.ndarray) -> int:
    with torch.no_grad():
        return int(model(torch.from_numpy(observation)).argmax().item())


def expert_positions(scenario) -> list[Position]:
    environment = GridWorld(scenario)
    positions = [environment.position]
    for action in shortest_path_actions(scenario):
        environment.step(action)
        positions.append(environment.position)
    return positions


def plot_episode(axis, scenario, result: dict[str, object], expert_path: list[Position]) -> None:
    size = scenario.size
    grid = np.zeros((size, size, 3), dtype=float)
    for row, column in scenario.obstacles:
        grid[row, column] = (0.15, 0.15, 0.15)
    axis.imshow(grid, extent=(-0.5, size - 0.5, size - 0.5, -0.5))

    policy_path = result["positions"]
    policy_rows = [position[0] for position in policy_path]
    policy_columns = [position[1] for position in policy_path]
    expert_rows = [position[0] for position in expert_path]
    expert_columns = [position[1] for position in expert_path]
    axis.plot(expert_columns, expert_rows, "--", color="tab:green", label="expert path")
    axis.plot(policy_columns, policy_rows, "-o", color="tab:blue", markersize=3, label="policy path")
    axis.scatter(*scenario.start[::-1], color="tab:orange", s=90, label="start", zorder=3)
    axis.scatter(*scenario.goal[::-1], color="tab:red", marker="*", s=150, label="goal", zorder=3)
    axis.set_title(
        f"{scenario.difficulty.title()} layout, seed {scenario.seed}\n"
        f"success={result['success']}, steps={result['steps']}, "
        f"invalid moves={result['invalid_moves']}"
    )
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.grid(color="lightgray", linewidth=0.8)
    axis.legend(fontsize=8, loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_episodes.png"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PolicyNetwork()
    model.load_state_dict(checkpoint["model_state"])

    scenarios = [
        scenario_from_seed(20_000, difficulty="standard"),
        scenario_from_seed(30_000, difficulty="difficult"),
    ]
    results = [
        rollout(scenario, lambda observation: policy_action(model, observation))
        for scenario in scenarios
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, scenario, result in zip(axes, scenarios, results):
        expert_path = expert_positions(scenario)
        plot_episode(axis, scenario, result, expert_path)
        print(
            f"seed={scenario.seed}: expert reaches goal in {len(expert_path) - 1} steps; "
            f"policy success={result['success']}, policy steps={result['steps']}, "
            f"invalid moves={result['invalid_moves']}"
        )
    figure.suptitle("Baseline policy and shortest-path expert")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
