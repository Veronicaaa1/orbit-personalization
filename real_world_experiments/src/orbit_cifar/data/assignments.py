from __future__ import annotations

from typing import Any, Sequence

from orbit_cifar.constants import (
    ALIGNED_NODE_TO_BUNDLE,
    ALIGNED_PROFILE_NAMES,
    FULLY_SYMMETRIC_PROFILE_NAME,
    GLOBAL_SYMMETRY_ORBITS,
    MISALIGNED_NODE_TO_BUNDLE,
    NUM_NODES,
    REFERENCE_ORBITS,
)


def node_to_bundle_mapping(name: str) -> tuple[int, ...]:
    """Return the immutable aligned or prescribed misaligned mapping."""
    if name == "aligned":
        return ALIGNED_NODE_TO_BUNDLE
    if name == "misaligned":
        return MISALIGNED_NODE_TO_BUNDLE
    if name == "fully_symmetric":
        return ALIGNED_NODE_TO_BUNDLE
    raise ValueError(f"Unknown data assignment: {name}")


def assignment_summary(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Describe node-to-bundle/profile assignment and orbit compatibility."""
    mapping = node_to_bundle_mapping(name)
    if sorted(mapping) != list(range(NUM_NODES)):
        raise ValueError(f"{name} mapping must be a permutation of concrete bundle ids")
    profiles = [str(manifest["bundles"][bundle_id]["profile_name"]) for bundle_id in mapping]
    compatible = all(len({profiles[node] for node in orbit}) == 1 for orbit in REFERENCE_ORBITS)
    expected_compatible = name in {"aligned", "fully_symmetric"}
    if compatible != expected_compatible:
        raise ValueError(f"Assignment {name} has unexpected reference-orbit profile compatibility={compatible}")
    summary = {
        "assignment": name,
        "node_to_bundle": list(mapping),
        "permutation_direction": "node_index_to_heterogeneous_bundle_id",
        "node_profiles": profiles,
        "orbit_profiles": [[profiles[node] for node in orbit] for orbit in REFERENCE_ORBITS],
        "reference_orbit_profiles_compatible": compatible,
    }
    if name == "fully_symmetric":
        if set(profiles) != {FULLY_SYMMETRIC_PROFILE_NAME}:
            raise ValueError("Fully symmetric assignment must use the balanced global profile at every node")
        summary.update(
            {
                "partition_semantics": "reference_partition_only",
                "reference_groups": [list(group) for group in REFERENCE_ORBITS],
                "true_automorphism_orbits": [list(group) for group in GLOBAL_SYMMETRY_ORBITS],
                "primary_disagreement_metric": "D_global=D_total",
            }
        )
    return summary


def expected_misaligned_profile_pattern() -> list[list[str]]:
    profiles = [ALIGNED_PROFILE_NAMES[bundle] for bundle in MISALIGNED_NODE_TO_BUNDLE]
    return [[profiles[node] for node in orbit] for orbit in REFERENCE_ORBITS]
