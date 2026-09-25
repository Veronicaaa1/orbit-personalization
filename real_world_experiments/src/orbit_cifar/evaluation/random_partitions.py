"""Matched-size random-partition descriptive diagnostic."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from orbit_cifar.constants import ORBIT_SIZES, REFERENCE_ORBITS
from orbit_cifar.evaluation.dispersion import dispersion_metrics


def generate_random_partitions(
    count: int = 10000,
    seed: int = 481516,
    group_sizes: Sequence[int] = ORBIT_SIZES,
) -> np.ndarray:
    """Uniformly permute nodes and cut labeled groups of exact requested sizes."""
    if count < 1 or any(size < 1 for size in group_sizes):
        raise ValueError("Partition count and group sizes must be positive")
    num_nodes = int(sum(group_sizes))
    rng = np.random.default_rng(seed)
    assignments = np.empty((count, num_nodes), dtype=np.int16)
    boundaries = np.cumsum([0, *group_sizes])
    for sample in range(count):
        permutation = rng.permutation(num_nodes)
        for group, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            assignments[sample, permutation[start:stop]] = group
    return assignments


def assignments_to_partition(assignment: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Convert one group-label vector into explicit groups."""
    labels = np.asarray(assignment)
    return tuple(tuple(np.flatnonzero(labels == group).tolist()) for group in sorted(np.unique(labels).tolist()))


def random_partition_diagnostic(
    centered_logits: torch.Tensor | np.ndarray,
    assignments: np.ndarray,
    true_partition: Sequence[Sequence[int]] = REFERENCE_ORBITS,
    ratio_threshold: float = 1e-14,
) -> dict[str, Any]:
    """Compare the reference ratio to the shared random-partition distribution."""
    values = torch.as_tensor(centered_logits, dtype=torch.float64, device="cpu")
    if assignments.ndim != 2 or assignments.shape[1] != values.shape[0]:
        raise ValueError("Random assignments must have shape [B, num_nodes]")
    true = dispersion_metrics(values, true_partition, ratio_threshold=ratio_threshold)
    flattened = values.reshape(values.shape[0], -1)
    gram = (flattened @ flattened.T).numpy()
    normalizer = float(values.shape[0] * values.shape[1] * values.shape[2])
    global_correction = float(gram.sum() / values.shape[0])
    total_numerator = float(np.trace(gram) - global_correction)
    random_ratios = np.empty(assignments.shape[0], dtype=np.float64)
    random_within = np.empty_like(random_ratios)
    random_between = np.empty_like(random_ratios)
    for index, assignment in enumerate(assignments):
        group_energy = 0.0
        for group in np.unique(assignment):
            members = np.flatnonzero(assignment == group)
            group_energy += float(gram[np.ix_(members, members)].sum() / len(members))
        between = max((group_energy - global_correction) / normalizer, 0.0)
        within = max((total_numerator / normalizer) - between, 0.0)
        denominator = within + between
        random_ratios[index] = math.nan if denominator < ratio_threshold else between / denominator
        random_within[index] = within
        random_between[index] = between
    finite = np.isfinite(random_ratios)
    true_ratio = float(true["R_orbit"])
    if not math.isfinite(true_ratio):
        p_value = math.nan
    else:
        p_value = float((1 + np.count_nonzero(random_ratios[finite] >= true_ratio)) / (1 + np.count_nonzero(finite)))
    return {
        "R_true": true_ratio,
        "true_ratio_undefined": bool(true["ratio_undefined"]),
        "R_random": random_ratios,
        "D_within_random": random_within,
        "D_between_random": random_between,
        "random_ratio_defined": finite,
        "p_perm": p_value,
    }


def aggregate_random_partition_diagnostics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average the same random partition across seeds before comparison."""
    usable = [result for result in results if np.isfinite(result["R_true"])]
    if not usable:
        return {"R_true_mean": math.nan, "R_random_mean": np.array([]), "p_perm_aggregate": math.nan, "valid_seeds": 0}
    random_stack = np.stack([np.asarray(result["R_random"], dtype=np.float64) for result in usable])
    random_mean = np.nanmean(random_stack, axis=0)
    true_mean = float(np.mean([float(result["R_true"]) for result in usable]))
    finite = np.isfinite(random_mean)
    p_value = float((1 + np.count_nonzero(random_mean[finite] >= true_mean)) / (1 + np.count_nonzero(finite)))
    return {"R_true_mean": true_mean, "R_random_mean": random_mean, "p_perm_aggregate": p_value, "valid_seeds": len(usable)}
