from __future__ import annotations

import torch
from torch import nn


class ActivationMLP(nn.Module):
    """Small MLP used for the next-20-frame binary classification task."""

    def __init__(self, input_size: int = 44) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class LinearBaseline(nn.Module):
    """A multinomial logistic-regression baseline with no hidden layer."""

    def __init__(self, input_size: int = 44) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)
