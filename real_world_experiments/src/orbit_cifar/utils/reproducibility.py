"""Stable seed derivation and random-state capture."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch


def derive_seed(run_seed: int, component_name: str, *parts: object) -> int:
    """Derive a stable unsigned 32-bit seed without Python's randomized hash."""
    payload = "|".join([str(run_seed), component_name, *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch on CPU and all visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG state required for epoch-boundary resumption."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"RNG payload lacks required fields: {sorted(missing)}")
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
        raise TypeError("torch_cpu RNG state must be a one-dimensional uint8 tensor")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location may move checkpoint tensors to CUDA; torch.set_rng_state
    # explicitly requires a CPU uint8 tensor.
    torch.set_rng_state(cpu_state.cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        if not all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 for value in state["torch_cuda"]):
            raise TypeError("CUDA RNG states must be uint8 tensors")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
