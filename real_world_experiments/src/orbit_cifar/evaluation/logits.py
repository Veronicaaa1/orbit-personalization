from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def center_logits(raw_logits: torch.Tensor, tolerance: float = 1e-10) -> torch.Tensor:
    """Subtract each image's mean over classes and verify the result."""
    if raw_logits.ndim != 3:
        raise ValueError(f"Expected raw logits shaped [node,image,class], found {tuple(raw_logits.shape)}")
    centered = raw_logits.to(dtype=torch.float64, device="cpu")
    centered = centered - centered.mean(dim=-1, keepdim=True)
    maximum_sum = float(centered.sum(dim=-1).abs().max()) if centered.numel() else 0.0
    if maximum_sum > tolerance:
        raise ValueError(f"Centered logits do not sum to zero across classes; max error={maximum_sum}")
    return centered


@torch.no_grad()
def collect_logits(
    models: Sequence[nn.Module],
    probe_loader: DataLoader[Any],
    device: torch.device | str,
    input_digest: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect a ``[N,M,C]`` tensor from the same deterministic probe batches."""
    if not models:
        raise ValueError("At least one model is required for probe evaluation")
    target_device = torch.device(device)
    for model in models:
        model.eval()
    per_node: list[list[torch.Tensor]] = [[] for _ in models]
    label_batches: list[torch.Tensor] = []
    for inputs, labels in probe_loader:
        if input_digest is not None:
            # Hash the existing ordered transformed probe, without another pass.
            input_digest.update(inputs.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        inputs = inputs.to(target_device, non_blocking=True)
        label_batches.append(labels.detach().to(device="cpu", dtype=torch.long))
        for node, model in enumerate(models):
            per_node[node].append(model(inputs).detach().to(device="cpu"))
    if not label_batches:
        raise ValueError("Probe loader produced no examples")
    raw = torch.stack([torch.cat(chunks, dim=0) for chunks in per_node], dim=0)
    return raw, torch.cat(label_batches, dim=0)
