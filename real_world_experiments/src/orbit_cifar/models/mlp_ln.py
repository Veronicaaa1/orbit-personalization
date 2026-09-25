"""Fully connected LayerNorm comparator for native dataset shapes."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class MLPLayerNorm(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int, hidden_dimensions: Sequence[int] = (256, 128)) -> None:
        super().__init__()
        shape = tuple(int(value) for value in input_shape)
        hidden = tuple(int(value) for value in hidden_dimensions)
        if len(shape) != 3 or any(value < 1 for value in shape) or hidden != (256, 128):
            raise ValueError("mlp_ln requires a positive CxHxW shape and hidden_dimensions=[256, 128]")
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], int(num_classes)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        return self.network(inputs)
