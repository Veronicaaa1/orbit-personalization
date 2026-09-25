"""Auditable common-random-number seed plan."""

from __future__ import annotations

from typing import Any

from orbit_cifar.constants import NUM_NODES
from orbit_cifar.utils.reproducibility import derive_seed


def build_seed_plan(run_seed: int, namespace: str = "legacy") -> dict[str, Any]:
    """Derive stable seeds, keeping bundle streams condition-independent."""
    if namespace not in {"legacy", "final", "pilot", "preflight"}:
        raise ValueError(f"Unknown RNG namespace: {namespace}")
    component_seed = int(run_seed) if namespace in {"legacy", "final"} else derive_seed(run_seed, f"{namespace}_namespace")
    return {
        "run_seed": int(run_seed),
        "namespace": namespace,
        "data_run_seed": component_seed,
        "base_initialization": derive_seed(component_seed, "base_initialization"),
        "orbit_perturbation": derive_seed(component_seed, "orbit_perturbation"),
        "bundle_sampler": [derive_seed(component_seed, "sampler", bundle) for bundle in range(NUM_NODES)],
        "bundle_augmentation": [derive_seed(component_seed, "augmentation", bundle) for bundle in range(NUM_NODES)],
        "diagnostic": derive_seed(component_seed, "diagnostic"),
    }
