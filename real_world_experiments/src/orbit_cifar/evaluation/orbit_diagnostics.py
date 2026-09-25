"""Pure float64, observation-only orbit geometry and prediction displacement.

Indices are zero based. All distances are squared Frobenius distances / (P*C).
No thresholds, ratios, RNG, model calls, or modification of input tensors.
"""

from __future__ import annotations

from itertools import combinations
from numbers import Integral

import torch

from orbit_cifar.evaluation.dispersion import validate_partition
from orbit_cifar.evaluation.logits import center_logits

SCHEMA_VERSION = 1
PROFILE = "orbit_prediction_v1"
RETAINED_EPOCHS = (175, 180, 185, 190, 195, 200)
ADJACENT_PAIRS = tuple(zip(RETAINED_EPOCHS[:-1], RETAINED_EPOCHS[1:]))


def _values(raw, groups):
    if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or min(raw.shape) <= 0:
        raise ValueError("Predictions must be a nonempty [N,P,C] tensor")
    if not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("Predictions must contain finite floating point logits")
    if any(not isinstance(i, Integral) or isinstance(i, bool) for group in groups for i in group):
        raise ValueError("Group indices must be integers")
    validate_partition(groups, raw.shape[0])
    values = center_logits(raw.detach())
    centers = torch.stack([values[list(group)].mean(0) for group in groups])
    return values, centers, values.mean(0)


def geometry(raw, groups, *, strict=False):
    """Return tidy metric components and independently reconstructed aggregates."""
    values, centers, _ = _values(raw, groups)
    n = len(values)
    within = [(values[list(group)] - centers[k]).square().mean().item() for k, group in enumerate(groups)]
    between = {(k, l): (centers[k] - centers[l]).square().mean().item()
               for k, l in combinations(range(len(groups)), 2)}
    rows = [{"group": k, "metric": "W_k", "value": value} for k, value in enumerate(within)]
    rows += [{"group": k, "pair_group": l, "metric": "B_kl", "value": value}
             for (k, l), value in between.items()]
    aggregates = {
        "D_within_reconstructed": sum(len(group) / n * within[k] for k, group in enumerate(groups)),
        "D_between_reconstructed": sum(len(groups[k]) * len(groups[l]) / n**2 * value
                                       for (k, l), value in between.items()),
        "min_center_distance_squared": min(between.values()) if between else None,
        "max_orbit_dispersion": max(within),
    }
    if strict:
        membership = {i: k for k, group in enumerate(groups) for i in group}
        same, different = [], []
        for i, j in combinations(range(n), 2):
            distance = (values[i] - values[j]).square().mean().item()
            (same if membership[i] == membership[j] else different).append(distance)
        maximum, minimum = max(same) if same else None, min(different) if different else None
        aggregates.update(max_within_node_distance_squared=maximum, min_between_node_distance_squared=minimum,
                          strict_pairwise_gap_squared=None if maximum is None or minimum is None else minimum - maximum)
    rows += [{"metric": name, "value": value} for name, value in aggregates.items()]
    return rows


def motion(previous, current, groups):
    """Absolute and relative displacements; endpoint and adjacent use the same formula."""
    if previous.shape != current.shape:
        raise ValueError("Temporal predictions have mismatched shapes")
    old, old_centers, old_global = _values(previous, groups)
    new, new_centers, new_global = _values(current, groups)
    node = (new - old).square().mean(dim=(1, 2))
    center = (new_centers - old_centers).square().mean(dim=(1, 2))
    relative = ((new_centers - new_global) - (old_centers - old_global)).square().mean(dim=(1, 2))
    rows = [{"node": i, "metric": "T_node_i", "value": value.item()} for i, value in enumerate(node)]
    rows += [{"group": k, "metric": name, "value": value.item()}
             for name, vector in (("T_center_k", center), ("T_center_relative_k", relative))
             for k, value in enumerate(vector)]
    rows += [{"metric": name, "value": value} for name, value in (
        ("T_node_max", node.max().item()), ("T_node_mean", node.mean().item()),
        ("T_global", (new_global - old_global).square().mean().item()))]
    return rows
