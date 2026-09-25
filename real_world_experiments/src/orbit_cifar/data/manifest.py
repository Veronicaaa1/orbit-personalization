from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from orbit_cifar.data.partition import (
    validate_dataset_manifest, validate_fully_symmetric_manifest, validate_split_manifest,
)
from orbit_cifar.utils.io import atomic_write_json, read_json


def validate_manifest(manifest: dict[str, Any], targets: Sequence[int] | None = None) -> dict[str, Any]:
    """Dispatch by schema and assignment, including historical balanced manifests."""
    if "schema_version" in manifest:
        return validate_dataset_manifest(manifest, targets)
    if str(manifest.get("manifest_type", "")).startswith("fully_symmetric"):
        return validate_fully_symmetric_manifest(manifest, targets)
    return validate_split_manifest(manifest, targets)


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Validate and atomically save a split manifest."""
    validate_manifest(manifest)
    atomic_write_json(path, manifest)


def load_manifest(path: str | Path, targets: Sequence[int] | None = None) -> dict[str, Any]:
    """Load and validate a split manifest."""
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise TypeError("Split manifest root must be a mapping")
    validate_manifest(manifest, targets=targets)
    return manifest


def validate_manifest_pair(
    heterogeneous: dict[str, Any], balanced: dict[str, Any], targets: Sequence[int],
    dataset_name: str | None = None,
) -> None:
    """Require both assignments to use one ordered validation probe and source.

    Older manifests omit source_split; their documented source is official train.
    Actual labels are supplied by the loaded training dataset, never inferred from
    a checksum or from the balanced manifest itself.
    """
    validate_manifest(heterogeneous, targets)
    validate_manifest(balanced, targets)
    if not str(balanced.get("manifest_type", "")).startswith("fully_symmetric"):
        raise ValueError("Balanced assignment requires a Fully symmetric manifest")
    if heterogeneous["validation_indices"] != balanced["validation_indices"]:
        raise ValueError("Cross-manifest validation indices must be exactly equal, including order")
    for manifest in (heterogeneous, balanced):
        if manifest.get("source_split", "train") != "train":
            raise ValueError("Assignment source split must be the official training split")
        if dataset_name is not None and manifest.get("dataset", "cifar10") != dataset_name:
            raise ValueError("Assignment dataset does not match the loaded source dataset")
        if int(manifest.get("num_source_examples", -1)) != len(targets):
            raise ValueError("Assignment source example count does not match the loaded source")
    if heterogeneous.get("dataset", "cifar10") != balanced.get("dataset", "cifar10"):
        raise ValueError("Cross-manifest source identity mismatch: dataset")
    for field in ("dataset_version", "num_source_examples", "split_seed"):
        if heterogeneous.get(field) != balanced.get(field):
            raise ValueError(f"Cross-manifest source identity mismatch: {field}")
    for field, source in (("source_indices", list(range(len(targets)))),
                          ("source_labels", [int(value) for value in targets])):
        if heterogeneous.get(field, source) != balanced.get(field, source):
            raise ValueError(f"Cross-manifest source identity mismatch: {field}")


def validate_manifest_source(
    manifest: dict[str, Any], dataset: Any, dataset_name: str,
    dataset_version: str, split_seed: int,
) -> None:
    """Bind persisted assignments to actual official training labels and metadata."""
    validate_manifest(manifest, dataset.targets)
    if getattr(dataset, "train", True) is not True:
        raise ValueError("Assignments require the official training dataset split")
    if manifest.get("dataset", "cifar10") != dataset_name or manifest.get("source_split", "train") != "train":
        raise ValueError("Manifest must identify the requested official training split")
    if str(manifest.get("dataset_version")) != dataset_version:
        raise ValueError("Manifest dataset version does not match the configured source")
    if int(manifest.get("split_seed", -1)) != split_seed:
        raise ValueError("Manifest split seed does not match the configured assignment")
