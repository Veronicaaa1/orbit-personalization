"""Independent optimizers and shared epoch-level warmup/cosine schedule."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def build_optimizers(
    models: Sequence[nn.Module],
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
) -> list[torch.optim.Optimizer]:
    """Create one plain SGD optimizer, with independent state, per node."""
    return [
        torch.optim.SGD(
            model.parameters(),
            lr=float(learning_rate),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        for model in models
    ]


class WarmupCosineFleetScheduler:
    """Apply one epoch schedule consistently to a fleet of independent optimizers."""

    def __init__(
        self,
        optimizers: Sequence[torch.optim.Optimizer],
        base_learning_rate: float,
        total_epochs: int,
        warmup_epochs: int,
        minimum_learning_rate: float,
    ) -> None:
        if total_epochs < 1 or warmup_epochs < 0 or warmup_epochs > total_epochs:
            raise ValueError("Invalid total_epochs/warmup_epochs schedule")
        self.optimizers = list(optimizers)
        self.base_learning_rate = float(base_learning_rate)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.epoch = 0
        self._apply(self.learning_rate_for_epoch(0))

    def learning_rate_for_epoch(self, epoch: int) -> float:
        """Calculate the learning rate used throughout a zero-based epoch."""
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.base_learning_rate * float(epoch + 1) / float(self.warmup_epochs)
        decay_epochs = max(self.total_epochs - self.warmup_epochs, 1)
        progress = min(max(epoch - self.warmup_epochs, 0) / decay_epochs, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_learning_rate + (self.base_learning_rate - self.minimum_learning_rate) * cosine

    def _apply(self, value: float) -> None:
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = float(value)

    def step(self) -> None:
        """Advance at an epoch boundary."""
        self.epoch += 1
        self._apply(self.learning_rate_for_epoch(self.epoch))

    def get_last_lr(self) -> list[float]:
        return [float(self.optimizers[0].param_groups[0]["lr"])] if self.optimizers else []

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        self._apply(self.learning_rate_for_epoch(self.epoch))
