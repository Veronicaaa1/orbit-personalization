"""Atomic checkpoints with complete optimizer, scheduler, and RNG state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from orbit_cifar.training.optim import WarmupCosineFleetScheduler
from orbit_cifar.utils.io import atomic_torch_save
from orbit_cifar.utils.reproducibility import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    models: Sequence[nn.Module],
    optimizers: Sequence[torch.optim.Optimizer],
    scheduler: WarmupCosineFleetScheduler,
    epoch: int,
    global_step: int,
    config_fingerprint: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save all state necessary to resume at the next epoch boundary."""
    payload = {
        "models": [model.state_dict() for model in models],
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng_state": capture_rng_state(),
        "config_fingerprint": config_fingerprint,
        "extra": extra or {},
    }
    atomic_torch_save(path, payload)


def load_checkpoint(
    path: str | Path,
    models: Sequence[nn.Module],
    optimizers: Sequence[torch.optim.Optimizer],
    scheduler: WarmupCosineFleetScheduler,
    expected_fingerprint: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint, rejecting architecture/config mismatches."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {target}")
    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(target, map_location=map_location)
    if len(payload["models"]) != len(models) or len(payload["optimizers"]) != len(optimizers):
        raise ValueError("Checkpoint node count differs from the active trainer")
    if expected_fingerprint is not None and payload["config_fingerprint"] != expected_fingerprint:
        raise ValueError("Checkpoint configuration fingerprint does not match this run")
    for model, state in zip(models, payload["models"]):
        model.load_state_dict(state)
    for optimizer, state in zip(optimizers, payload["optimizers"]):
        optimizer.load_state_dict(state)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "extra": payload.get("extra", {}),
    }

