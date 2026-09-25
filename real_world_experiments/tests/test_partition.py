from __future__ import annotations

import numpy as np

from orbit_cifar.data.assignments import assignment_summary, expected_misaligned_profile_pattern, node_to_bundle_mapping
from orbit_cifar.data.partition import (
    build_fully_symmetric_manifest,
    build_split_manifest,
    validate_fully_symmetric_manifest,
    validate_split_manifest,
)


def mock_targets() -> np.ndarray:
    return np.repeat(np.arange(10), 5000)


def test_exact_disjoint_manifest_and_determinism() -> None:
    targets = mock_targets()
    first = build_split_manifest(targets, split_seed=2027)
    second = build_split_manifest(targets, split_seed=2027)
    different = build_split_manifest(targets, split_seed=2028)
    summary = validate_split_manifest(first, targets)
    assert summary["local_examples"] == 28000
    assert summary["validation_examples"] == 5000
    assert all(len(bundle["indices"]) == 2000 for bundle in first["bundles"])
    assert first == second
    assert first != different


def test_misaligned_is_fixed_permutation_with_expected_profiles() -> None:
    manifest = build_split_manifest(mock_targets())
    mapping = node_to_bundle_mapping("misaligned")
    assert sorted(mapping) == list(range(14))
    summary = assignment_summary("misaligned", manifest)
    assert not summary["reference_orbit_profiles_compatible"]
    assert summary["orbit_profiles"] == expected_misaligned_profile_pattern()
    assert summary["orbit_profiles"] == [
        ["profile_1", "profile_4"],
        ["profile_2", "profile_3", "profile_4"],
        ["profile_1", "profile_2", "profile_3", "profile_4"],
        ["profile_2", "profile_3", "profile_3", "profile_4", "profile_4"],
    ]


def test_fully_symmetric_balanced_manifest_is_disjoint_and_deterministic() -> None:
    targets = mock_targets()
    heterogeneous = build_split_manifest(targets, split_seed=2027)
    original = build_split_manifest(targets, split_seed=2027)
    first = build_fully_symmetric_manifest(
        targets,
        validation_indices=heterogeneous["validation_indices"],
        split_seed=heterogeneous["split_seed"],
    )
    second = build_fully_symmetric_manifest(
        targets,
        validation_indices=heterogeneous["validation_indices"],
        split_seed=heterogeneous["split_seed"],
    )
    summary = validate_fully_symmetric_manifest(first, targets)
    assert heterogeneous == original
    assert first == second
    assert summary["num_bundles"] == 14
    assert summary["local_examples"] == 28000
    expected_counts = [200] * 10
    local_sets = []
    for bundle in first["bundles"]:
        assert len(bundle["indices"]) == 2000
        assert bundle["class_counts"] == expected_counts
        local_sets.append(set(bundle["indices"]))
    assert all(local_sets[left].isdisjoint(local_sets[right]) for left in range(14) for right in range(left + 1, 14))
    validation = set(heterogeneous["validation_indices"])
    assert all(indices.isdisjoint(validation) for indices in local_sets)


def test_fully_symmetric_assignment_uses_one_global_orbit() -> None:
    heterogeneous = build_split_manifest(mock_targets())
    manifest = build_fully_symmetric_manifest(mock_targets(), heterogeneous["validation_indices"])
    summary = assignment_summary("fully_symmetric", manifest)
    assert summary["node_to_bundle"] == list(range(14))
    assert summary["node_profiles"] == ["balanced_global"] * 14
    assert summary["true_automorphism_orbits"] == [list(range(14))]
    assert summary["partition_semantics"] == "reference_partition_only"
