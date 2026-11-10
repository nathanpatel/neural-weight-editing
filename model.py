"""The small policy network used by the first experiment."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """A deliberately small MLP whose parameters are easy to inspect."""

    def __init__(self, grid_size: int = 7, hidden_size: int = 32) -> None:
        super().__init__()
        input_size = 3 * grid_size * grid_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)
