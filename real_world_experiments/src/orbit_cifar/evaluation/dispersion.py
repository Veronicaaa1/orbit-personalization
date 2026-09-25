from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from orbit_cifar.constants import REFERENCE_ORBITS


def validate_partition(partition: Sequence[Sequence[int]], num_nodes: int) -> None:
    """Require nonempty groups that cover every node exactly once."""
    if not partition or any(len(group) == 0 for group in partition):
        raise ValueError("A partition must contain nonempty groups")
    flat = [int(node) for group in partition for node in group]
    if sorted(flat) != list(range(num_nodes)):
        raise ValueError(f"Partition must cover node indices 0 through {num_nodes - 1} exactly once")


def dispersion_metrics(
    centered_logits: torch.Tensor | np.ndarray,
    partition: Sequence[Sequence[int]] = REFERENCE_ORBITS,
    decomposition_tolerance: float = 1e-9,
    ratio_threshold: float = 1e-14,
) -> dict[str, Any]:
    """Compute the prescribed within, between, total, and alignment ratio."""
    values = torch.as_tensor(centered_logits, dtype=torch.float64, device="cpu")
    if values.ndim != 3:
        raise ValueError(f"Expected centered logits [N,M,C], found {tuple(values.shape)}")
    num_nodes, num_images, num_classes = values.shape
    if num_images == 0 or num_classes == 0:
        raise ValueError("Centered logits must include images and classes")
    validate_partition(partition, num_nodes)
    global_mean = values.mean(dim=0)
    within_sum = torch.zeros((), dtype=torch.float64)
    between_sum = torch.zeros((), dtype=torch.float64)
    for group in partition:
        indices = torch.as_tensor(list(group), dtype=torch.long)
        group_values = values.index_select(0, indices)
        group_mean = group_values.mean(dim=0)
        within_sum += (group_values - group_mean).square().sum()
        between_sum += len(group) * (group_mean - global_mean).square().sum()
    denominator = float(num_classes * num_images * num_nodes)
    within = float(within_sum / denominator)
    between = float(between_sum / denominator)
    total = float((values - global_mean).square().sum() / denominator)
    within = 0.0 if abs(within) < ratio_threshold else within
    between = 0.0 if abs(between) < ratio_threshold else between
    total = 0.0 if abs(total) < ratio_threshold else total
    error = abs(total - within - between)
    scale = max(1.0, abs(total), abs(within) + abs(between))
    if error > decomposition_tolerance * scale:
        raise ValueError(
            f"Dispersion decomposition failed: total={total}, within={within}, between={between}, error={error}"
        )
    ratio_denominator = within + between
    undefined = ratio_denominator < ratio_threshold
    ratio = math.nan if undefined else between / ratio_denominator
    return {
        "D_within": within,
        "D_between": between,
        "D_total": total,
        "R_orbit": ratio,
        "ratio_undefined": undefined,
        "decomposition_error": error,
    }


def pairwise_distance_matrix(centered_logits: torch.Tensor | np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Compute squared distances with a flattened Gram matrix."""
    values = torch.as_tensor(centered_logits, dtype=torch.float64, device="cpu")
    if values.ndim != 3:
        raise ValueError("centered_logits must have shape [node,image,class]")
    num_nodes, num_images, num_classes = values.shape
    flattened = values.reshape(num_nodes, -1)
    gram = flattened @ flattened.T
    squared_norms = torch.diag(gram)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram
    distances /= float(num_classes * num_images)
    if float(distances.min()) < -tolerance:
        raise ValueError(f"Pairwise squared distance is materially negative: {float(distances.min())}")
    distances.clamp_(min=0.0)
    distances.fill_diagonal_(0.0)
    array = distances.numpy()
    if not np.allclose(array, array.T, atol=tolerance, rtol=0):
        raise ValueError("Pairwise distance matrix is not symmetric")
    return array


def terminal_window(rows: Sequence[dict[str, Any]], count: int = 5) -> tuple[dict[str, float], bool]:
    """Average valid scalar checkpoints and indicate whether fewer than requested were used."""
    valid = [row for row in rows if all(math.isfinite(float(row[key])) for key in ("D_within", "D_between"))]
    if not valid:
        raise ValueError("No valid evaluation checkpoints are available")
    selected = valid[-count:]
    means: dict[str, float] = {}
    for key in ("D_within", "D_between", "R_orbit"):
        values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
        means[f"{key}_bar"] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
    return means, len(valid) < count
