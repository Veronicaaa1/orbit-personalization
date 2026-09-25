"""Runtime environment recording and device resolution."""

from __future__ import annotations

import platform
import sys
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable requested CUDA devices."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available")
    return device


def environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
