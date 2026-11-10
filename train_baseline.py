"""Train and save the supervised baseline policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gridworld import collect_expert_examples, scenario_from_seed
from model import PolicyNetwork


def train_baseline(
    *,
    seed: int = 7,
    scenario_count: int = 100,
    epochs: int = 150,
) -> tuple[PolicyNetwork, dict[str, object]]:
    """Train on standard layouts and return the model and recorded settings."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    scenarios = [scenario_from_seed(10_000 + index) for index in range(scenario_count)]
    observations, actions = collect_expert_examples(scenarios)
    model = PolicyNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    input_tensor = torch.from_numpy(observations)
    target_tensor = torch.from_numpy(actions)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(input_tensor), target_tensor)
        loss.backward()
        optimizer.step()

    settings = {
        "seed": seed,
        "scenario_count": scenario_count,
        "epochs": epochs,
        "grid_size": 7,
        "hidden_size": 32,
        "training_layout_difficulty": "standard",
    }
    return model, settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.pt"))
    args = parser.parse_args()
    model, settings = train_baseline()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "settings": settings}, args.output)
    print(f"Saved baseline policy to {args.output}")


if __name__ == "__main__":
    main()
