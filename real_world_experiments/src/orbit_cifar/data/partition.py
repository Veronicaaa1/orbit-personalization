"""Deterministic, mutually exclusive MNIST and CIFAR-10 bundle construction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from orbit_cifar.constants import (
    ALIGNED_PROFILE_NAMES,
    CIFAR10_CLASSES,
    CLASS_PROFILES,
    FULLY_SYMMETRIC_CLASS_COUNTS,
    FULLY_SYMMETRIC_PROFILE_NAME,
    NUM_CLASSES,
    NUM_NODES,
    ORBIT_SIZES,
)
from orbit_cifar.utils.io import config_fingerprint


def _validate_integer_indices(manifest: dict[str, Any]) -> None:
    """Reject lossy casts before checking any assignment or source indices."""
    from numbers import Integral

    sequences = [manifest.get(key, []) for key in ("validation_indices", "unused_indices", "source_indices")]
    sequences.extend(bundle.get("indices", []) for bundle in manifest.get("bundles", []))
    for sequence in sequences:
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in sequence):
            raise ValueError("Manifest indices must be integers, without fractional or boolean values")


def build_split_manifest(
    targets: Sequence[int],
    split_seed: int = 2027,
    validation_per_class: int = 500,
    dataset_version: str = "torchvision-cifar10",
) -> dict[str, Any]:
    """Allocate aligned local bundles, balanced validation, and unused indices."""
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("CIFAR targets must be a one-dimensional sequence")
    per_class: list[np.ndarray] = []
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_id)
        rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), class_id]))
        per_class.append(indices[rng.permutation(len(indices))])
    cursors = [0] * NUM_CLASSES
    bundles: list[dict[str, Any]] = []
    for bundle_id, profile_name in enumerate(ALIGNED_PROFILE_NAMES):
        counts = CLASS_PROFILES[profile_name]
        selected: list[int] = []
        for class_id, count in enumerate(counts):
            start, stop = cursors[class_id], cursors[class_id] + int(count)
            if stop > len(per_class[class_id]):
                raise ValueError(
                    f"Class {class_id} lacks examples for bundle {bundle_id}: requested through {stop}, available {len(per_class[class_id])}"
                )
            selected.extend(per_class[class_id][start:stop].tolist())
            cursors[class_id] = stop
        bundles.append(
            {
                "bundle_id": bundle_id,
                "profile_name": profile_name,
                "class_counts": list(counts),
                "indices": selected,
            }
        )
    validation_indices: list[int] = []
    unused_indices: list[int] = []
    for class_id in range(NUM_CLASSES):
        validation_stop = cursors[class_id] + int(validation_per_class)
        if validation_stop > len(per_class[class_id]):
            raise ValueError(
                f"Class {class_id} lacks {validation_per_class} remaining validation examples after local allocation"
            )
        validation_indices.extend(per_class[class_id][cursors[class_id]:validation_stop].tolist())
        unused_indices.extend(per_class[class_id][validation_stop:].tolist())
    manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "class_names": list(CIFAR10_CLASSES),
        "split_seed": int(split_seed),
        "num_source_examples": int(len(labels)),
        "bundles": bundles,
        "validation_indices": validation_indices,
        "validation_class_counts": [int(validation_per_class)] * NUM_CLASSES,
        "unused_indices": unused_indices,
    }
    validate_split_manifest(manifest, targets=labels)
    return manifest


def build_fully_symmetric_manifest(
    targets: Sequence[int],
    validation_indices: Sequence[int],
    split_seed: int = 2027,
    dataset_version: str = "torchvision-cifar10",
) -> dict[str, Any]:
    """Build 14 disjoint balanced bundles around an existing fixed validation probe."""
    labels = np.asarray(targets, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("CIFAR targets must be a one-dimensional sequence")
    validation = [int(index) for index in validation_indices]
    if len(validation) != len(set(validation)):
        raise ValueError("Validation indices must be unique")
    if validation and (min(validation) < 0 or max(validation) >= len(labels)):
        raise ValueError("Validation index lies outside the source dataset")
    validation_counts = np.bincount(labels[np.asarray(validation, dtype=np.int64)], minlength=NUM_CLASSES)
    if validation_counts.tolist() != [500] * NUM_CLASSES:
        raise ValueError("The fully symmetric baseline requires the fixed 500-per-class validation probe")

    validation_set = set(validation)
    per_class: list[np.ndarray] = []
    required_per_class = NUM_NODES * FULLY_SYMMETRIC_CLASS_COUNTS[0]
    for class_id in range(NUM_CLASSES):
        eligible = np.asarray(
            [index for index in np.flatnonzero(labels == class_id).tolist() if index not in validation_set],
            dtype=np.int64,
        )
        if len(eligible) < required_per_class:
            raise ValueError(
                f"Class {class_id} has {len(eligible)} non-validation examples; {required_per_class} are required"
            )
        # The fixed tag gives this new manifest an independent stream without changing
        # the existing heterogeneous split or its RNG construction.
        rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), class_id, 0xF011]))
        per_class.append(eligible[rng.permutation(len(eligible))])

    bundles: list[dict[str, Any]] = []
    for bundle_id in range(NUM_NODES):
        selected: list[int] = []
        for class_id, indices in enumerate(per_class):
            start = bundle_id * FULLY_SYMMETRIC_CLASS_COUNTS[class_id]
            stop = start + FULLY_SYMMETRIC_CLASS_COUNTS[class_id]
            selected.extend(indices[start:stop].tolist())
        bundles.append(
            {
                "bundle_id": bundle_id,
                "profile_name": FULLY_SYMMETRIC_PROFILE_NAME,
                "class_counts": list(FULLY_SYMMETRIC_CLASS_COUNTS),
                "indices": selected,
            }
        )

    unused_indices: list[int] = []
    for indices in per_class:
        unused_indices.extend(indices[required_per_class:].tolist())
    manifest: dict[str, Any] = {
        "manifest_type": "fully_symmetric_balanced",
        "dataset_version": dataset_version,
        "class_names": list(CIFAR10_CLASSES),
        "split_seed": int(split_seed),
        "num_source_examples": int(len(labels)),
        "bundles": bundles,
        "validation_indices": validation,
        "validation_class_counts": [500] * NUM_CLASSES,
        "unused_indices": unused_indices,
    }
    validate_fully_symmetric_manifest(manifest, targets=labels)
    return manifest


def validate_fully_symmetric_manifest(
    manifest: dict[str, Any], targets: Sequence[int] | None = None
) -> dict[str, Any]:
    """Validate the balanced baseline's exact counts and concrete disjointness."""
    _validate_integer_indices(manifest)
    bundles = manifest.get("bundles", [])
    if len(bundles) != NUM_NODES:
        raise ValueError(f"Fully symmetric manifest must contain {NUM_NODES} bundles")
    all_local: list[int] = []
    for bundle_id, bundle in enumerate(bundles):
        if int(bundle["bundle_id"]) != bundle_id:
            raise ValueError("Bundle ids must be ordered 0 through 13")
        if str(bundle["profile_name"]) != FULLY_SYMMETRIC_PROFILE_NAME:
            raise ValueError(f"Bundle {bundle_id} does not use the globally balanced profile")
        if list(bundle["class_counts"]) != list(FULLY_SYMMETRIC_CLASS_COUNTS):
            raise ValueError(f"Bundle {bundle_id} is not exactly balanced")
        indices = [int(index) for index in bundle["indices"]]
        if len(indices) != 2000 or len(set(indices)) != 2000:
            raise ValueError(f"Bundle {bundle_id} must contain 2000 unique examples")
        all_local.extend(indices)
    if len(all_local) != 28000 or len(set(all_local)) != 28000:
        raise ValueError("Fully symmetric node bundles must contain 28000 pairwise-disjoint examples")

    validation = [int(index) for index in manifest.get("validation_indices", [])]
    if len(validation) != 5000 or len(set(validation)) != 5000:
        raise ValueError("Validation must contain exactly 5000 unique examples")
    if set(all_local).intersection(validation):
        raise ValueError("Fully symmetric local bundles overlap the validation probe")
    unused = [int(index) for index in manifest.get("unused_indices", [])]
    if set(all_local).intersection(unused) or set(validation).intersection(unused):
        raise ValueError("Unused indices overlap an allocated split")

    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64)
        allocated = all_local + validation + unused
        if any(value < 0 or value >= len(labels) for value in allocated):
            raise ValueError("A manifest index lies outside the source dataset")
        if len(allocated) != len(labels) or len(set(allocated)) != len(labels):
            raise ValueError("Fully symmetric allocations do not partition the source dataset")
        for bundle in bundles:
            actual = np.bincount(labels[np.asarray(bundle["indices"], dtype=np.int64)], minlength=NUM_CLASSES)
            if actual.tolist() != list(FULLY_SYMMETRIC_CLASS_COUNTS):
                raise ValueError(f"Bundle {bundle['bundle_id']} indices are not exactly balanced")
        actual_validation = np.bincount(labels[np.asarray(validation, dtype=np.int64)], minlength=NUM_CLASSES)
        if actual_validation.tolist() != [500] * NUM_CLASSES:
            raise ValueError("Validation indices are not balanced as declared")
    return {
        "num_bundles": NUM_NODES,
        "local_examples": len(all_local),
        "validation_examples": len(validation),
        "unused_examples": len(unused),
        "all_allocations_disjoint": True,
    }


def validate_split_manifest(manifest: dict[str, Any], targets: Sequence[int] | None = None) -> dict[str, Any]:
    """Check exact counts, index exclusivity, and optional labels."""
    _validate_integer_indices(manifest)
    bundles = manifest.get("bundles", [])
    if len(bundles) != NUM_NODES:
        raise ValueError(f"Manifest must contain {NUM_NODES} bundles, found {len(bundles)}")
    all_local: list[int] = []
    for bundle_id, bundle in enumerate(bundles):
        if int(bundle["bundle_id"]) != bundle_id:
            raise ValueError("Bundle ids must be ordered 0 through 13")
        expected = list(CLASS_PROFILES[str(bundle["profile_name"])])
        if list(bundle["class_counts"]) != expected:
            raise ValueError(f"Bundle {bundle_id} class counts do not match {bundle['profile_name']}")
        indices = [int(index) for index in bundle["indices"]]
        if len(indices) != 2000 or len(set(indices)) != 2000:
            raise ValueError(f"Bundle {bundle_id} must contain 2000 unique examples")
        all_local.extend(indices)
    if len(all_local) != 28000 or len(set(all_local)) != 28000:
        raise ValueError("Local bundles must contain exactly 28000 mutually exclusive examples")
    validation = [int(index) for index in manifest.get("validation_indices", [])]
    if len(validation) != 5000 or len(set(validation)) != 5000:
        raise ValueError("Validation must contain exactly 5000 unique examples")
    if set(all_local).intersection(validation):
        raise ValueError("Local and validation indices overlap")
    unused = [int(index) for index in manifest.get("unused_indices", [])]
    if set(all_local).intersection(unused) or set(validation).intersection(unused):
        raise ValueError("Unused indices overlap an allocated split")
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64)
        allocated = all_local + validation + unused
        if any(value < 0 or value >= len(labels) for value in allocated):
            raise ValueError("A manifest index lies outside the source dataset")
        if len(allocated) != len(labels) or len(set(allocated)) != len(labels):
            raise ValueError("Local, validation, and unused indices must partition the source dataset")
        for bundle in bundles:
            actual = Counter(labels[np.asarray(bundle["indices"], dtype=np.int64)].tolist())
            counts = [actual.get(class_id, 0) for class_id in range(NUM_CLASSES)]
            if counts != list(bundle["class_counts"]):
                raise ValueError(f"Bundle {bundle['bundle_id']} indices do not have their declared class counts")
        validation_counts = np.bincount(labels[np.asarray(validation, dtype=np.int64)], minlength=NUM_CLASSES).tolist()
        if validation_counts != list(manifest["validation_class_counts"]):
            raise ValueError("Validation indices are not balanced as declared")
        combined = all_local + validation
        per_class_used = np.bincount(labels[np.asarray(combined, dtype=np.int64)], minlength=NUM_CLASSES)
        available = np.bincount(labels, minlength=NUM_CLASSES)
        if np.any(per_class_used > available):
            raise ValueError("A class uses more source examples than are available")
    return {
        "num_bundles": NUM_NODES,
        "local_examples": len(all_local),
        "validation_examples": len(validation),
        "unused_examples": len(unused),
        "all_allocations_disjoint": True,
    }




def _allocate_manifest(
    labels: np.ndarray,
    profiles: Sequence[Sequence[int]],
    profile_names: Sequence[str],
    class_names: Sequence[str],
    dataset_name: str,
    dataset_version: str,
    split_seed: int,
    validation_per_class: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate concrete indices after checking all per-class capacity constraints."""
    num_classes = len(class_names)
    if labels.ndim != 1 or sorted(set(labels.tolist())) != list(range(num_classes)):
        raise ValueError(f"{dataset_name} targets must contain class ids 0..{num_classes - 1}")
    if len(profile_names) != NUM_NODES:
        raise ValueError("Exactly fourteen profile names are required")
    expanded = [list(map(int, profiles[index])) for index in range(len(profiles))]
    bundle_profiles = [expanded[index] for index, size in enumerate(ORBIT_SIZES) for _ in range(size)]
    requested = np.sum(np.asarray(bundle_profiles, dtype=np.int64), axis=0) + int(validation_per_class)
    available = np.bincount(labels, minlength=num_classes)
    offending = [
        {"class_id": index, "requested": int(requested[index]), "available": int(available[index])}
        for index in range(num_classes) if requested[index] > available[index]
    ]
    if offending:
        raise ValueError(f"Insufficient source-label capacity: {offending}")
    shuffled: list[np.ndarray] = []
    for class_id in range(num_classes):
        indices = np.flatnonzero(labels == class_id)
        # This is deliberately identical to the legacy CIFAR-10 random stream.
        rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), class_id]))
        shuffled.append(indices[rng.permutation(len(indices))])
    cursors = np.zeros(num_classes, dtype=np.int64)
    bundles: list[dict[str, Any]] = []
    for bundle_id, (profile_name, counts) in enumerate(zip(profile_names, bundle_profiles)):
        selected: list[int] = []
        selected_labels: list[int] = []
        for class_id, count in enumerate(counts):
            start, stop = int(cursors[class_id]), int(cursors[class_id] + count)
            selected.extend(shuffled[class_id][start:stop].tolist())
            selected_labels.extend([class_id] * int(count))
            cursors[class_id] = stop
        bundles.append({
            "bundle_id": bundle_id, "profile_name": profile_name,
            "class_counts": counts, "indices": selected,
            "labels_checksum": config_fingerprint(selected_labels),
        })
    validation: list[int] = []
    unused: list[int] = []
    for class_id in range(num_classes):
        stop = int(cursors[class_id]) + int(validation_per_class)
        validation.extend(shuffled[class_id][int(cursors[class_id]):stop].tolist())
        unused.extend(shuffled[class_id][stop:].tolist())
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "source_split": "train",
        "manifest_type": "heterogeneous",
        "dataset": dataset_name,
        "dataset_version": dataset_version,
        "num_classes": num_classes,
        "class_names": list(class_names),
        "split_seed": int(split_seed),
        "num_source_examples": int(len(labels)),
        "source_indices": list(range(len(labels))),
        "source_labels": labels.astype(int).tolist(),
        "source_indices_checksum": config_fingerprint(list(range(len(labels)))),
        "source_labels_checksum": config_fingerprint(labels.tolist()),
        "bundles": bundles,
        "validation_indices": validation,
        "validation_class_counts": [int(validation_per_class)] * num_classes,
        "unused_indices": unused,
        "profile_definitions": {str(name): list(map(int, profile)) for name, profile in zip(
            ("profile_1", "profile_2", "profile_3", "profile_4"), expanded
        )},
    }
    if extra:
        manifest.update(extra)
    validate_dataset_manifest(manifest, labels)
    return manifest


def build_dataset_manifest(
    dataset_name: str,
    targets: Sequence[int],
    split_seed: int = 2027,
    validation_per_class: int | None = None,
    dataset_version: str | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    labels = np.asarray(targets, dtype=np.int64)
    if dataset_name in {"mnist", "cifar10"}:
        names = list(class_names or (list(map(str, range(10))) if dataset_name == "mnist" else CIFAR10_CLASSES))
        profiles = [list(CLASS_PROFILES[f"profile_{index}"]) for index in range(1, 5)]
        validation = 500 if validation_per_class is None else int(validation_per_class)
        return _allocate_manifest(
            labels, profiles, [f"profile_{index}" for index in range(1, 5) for _ in range(ORBIT_SIZES[index - 1])],
            names, dataset_name, dataset_version or f"torchvision-{dataset_name}", split_seed, validation,
        )
    raise ValueError(f"Unknown dataset name: {dataset_name}")


def build_balanced_manifest(
    targets: Sequence[int],
    validation_indices: Sequence[int],
    dataset_name: str,
    split_seed: int = 2027,
    dataset_version: str | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build fourteen disjoint, class-balanced bundles around the same probe."""
    _validate_integer_indices({"validation_indices": validation_indices})
    if dataset_name not in {"mnist", "cifar10"}:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    labels = np.asarray(targets, dtype=np.int64)
    num_classes = 10
    per_node = 200
    validation = [int(value) for value in validation_indices]
    if any(value < 0 or value >= len(labels) for value in validation):
        raise ValueError("Validation index lies outside the source dataset")
    if len(validation) != len(set(validation)):
        raise ValueError("Validation indices must be unique")
    validation_set = set(validation)
    per_class: list[np.ndarray] = []
    for class_id in range(num_classes):
        eligible = np.asarray([value for value in np.flatnonzero(labels == class_id) if int(value) not in validation_set])
        required = NUM_NODES * per_node
        if len(eligible) < required:
            raise ValueError(f"Class {class_id}: requested {required} balanced local images, available {len(eligible)}")
        rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), class_id, 0xF011]))
        per_class.append(eligible[rng.permutation(len(eligible))])
    bundles = []
    for bundle_id in range(NUM_NODES):
        indices = [
            int(value) for class_id in range(num_classes)
            for value in per_class[class_id][bundle_id * per_node:(bundle_id + 1) * per_node]
        ]
        bundles.append({
            "bundle_id": bundle_id, "profile_name": FULLY_SYMMETRIC_PROFILE_NAME,
            "class_counts": [per_node] * num_classes, "indices": indices,
        })
    allocated = {value for bundle in bundles for value in bundle["indices"]} | validation_set
    manifest = {
        "schema_version": 2, "source_split": "train", "manifest_type": "fully_symmetric_balanced", "dataset": dataset_name,
        "dataset_version": dataset_version or f"torchvision-{dataset_name}", "num_classes": num_classes,
        "class_names": list(class_names or [str(value) for value in range(num_classes)]),
        "split_seed": int(split_seed), "num_source_examples": len(labels), "bundles": bundles,
        "source_indices": list(range(len(labels))), "source_labels": labels.astype(int).tolist(),
        "validation_indices": validation,
        "validation_class_counts": np.bincount(labels[np.asarray(validation)], minlength=num_classes).astype(int).tolist(),
        "unused_indices": [value for value in range(len(labels)) if value not in allocated],
        "source_labels_checksum": config_fingerprint(labels.tolist()),
    }
    validate_dataset_manifest(manifest, labels)
    return manifest


def validate_dataset_manifest(manifest: dict[str, Any], targets: Sequence[int] | None = None) -> dict[str, Any]:
    """Validate dimensions, declarations, labels, bounds, and global disjointness."""
    _validate_integer_indices(manifest)
    num_classes = int(manifest.get("num_classes", 10))
    bundles = list(manifest.get("bundles", []))
    if len(bundles) != NUM_NODES:
        raise ValueError(f"Manifest must contain {NUM_NODES} bundles")
    manifest_type = str(manifest.get("manifest_type", ""))
    dataset_name = str(manifest.get("dataset", "cifar10"))
    if dataset_name not in {"mnist", "cifar10"} or num_classes != 10:
        raise ValueError("Manifest dataset and class dimensions must match a registered dataset")
    if manifest.get("source_split", "train") != "train":
        raise ValueError("Assignment source split must be the official training split")
    if manifest_type not in {"heterogeneous", "fully_symmetric_balanced"}:
        raise ValueError(f"Unknown assignment manifest type: {manifest_type}")
    definitions = manifest.get("profile_definitions", {})
    if manifest_type == "heterogeneous":
        expected_definitions = {f"profile_{index}": list(CLASS_PROFILES[f"profile_{index}"]) for index in range(1, 5)}
        if definitions != expected_definitions:
            raise ValueError("Heterogeneous profile definitions must match the canonical scientific profiles")
    expected_profile_names = [f"profile_{index}" for index in range(1, 5) for _ in range(ORBIT_SIZES[index - 1])]
    local: list[int] = []
    for bundle_id, bundle in enumerate(bundles):
        indices = list(map(int, bundle.get("indices", [])))
        counts = list(map(int, bundle.get("class_counts", [])))
        if int(bundle.get("bundle_id", -1)) != bundle_id or len(indices) != 2000 or len(set(indices)) != 2000:
            raise ValueError(f"Bundle {bundle_id} must contain 2000 unique indices and its canonical id")
        if len(counts) != num_classes or sum(counts) != 2000:
            raise ValueError(f"Bundle {bundle_id} has invalid {num_classes}-class counts")
        if manifest_type == "fully_symmetric_balanced":
            expected_balanced = [200] * num_classes
            if str(bundle.get("profile_name")) != FULLY_SYMMETRIC_PROFILE_NAME or counts != expected_balanced:
                raise ValueError(f"Balanced bundle {bundle_id} does not have its exact equal class profile")
        elif manifest_type == "heterogeneous":
            expected_name = expected_profile_names[bundle_id]
            if str(bundle.get("profile_name")) != expected_name or counts != list(definitions.get(expected_name, [])):
                raise ValueError(f"Heterogeneous bundle {bundle_id} disagrees with {expected_name}")
        local.extend(indices)
    validation = list(map(int, manifest.get("validation_indices", [])))
    unused = list(map(int, manifest.get("unused_indices", [])))
    expected_validation = 5000
    if len(local) != 28000 or len(set(local)) != 28000 or len(validation) != expected_validation or len(set(validation)) != expected_validation:
        raise ValueError("Manifest must have 28000 disjoint local and 5000 unique validation indices")
    if set(local) & set(validation) or set(local) & set(unused) or set(validation) & set(unused):
        raise ValueError("Local, validation, and unused allocations overlap")
    if targets is None and "source_labels" in manifest:
        targets = manifest["source_labels"]
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64)
        if int(manifest.get("num_source_examples", -1)) != len(labels):
            raise ValueError("Source example count does not match the dataset")
        if labels.ndim != 1 or sorted(set(labels.tolist())) != list(range(num_classes)):
            raise ValueError("Source labels do not cover exactly the registered classes")
        if manifest.get("source_indices") != list(range(len(labels))):
            raise ValueError("Persisted source indices do not match the actual dataset")
        if manifest.get("source_labels") != labels.astype(int).tolist():
            raise ValueError("Persisted source labels do not match the actual dataset")
        all_indices = local + validation + unused
        if any(value < 0 or value >= len(labels) for value in all_indices):
            raise ValueError("A manifest index lies outside the source dataset")
        if len(all_indices) != len(labels) or len(set(all_indices)) != len(labels):
            raise ValueError("Local, validation, and unused indices must partition the source dataset")
        for bundle in bundles:
            actual = np.bincount(labels[np.asarray(bundle["indices"], dtype=np.int64)], minlength=num_classes).tolist()
            if actual != list(bundle["class_counts"]):
                raise ValueError(f"Bundle {bundle['bundle_id']} labels disagree with declared counts")
        actual_validation = np.bincount(labels[np.asarray(validation, dtype=np.int64)], minlength=num_classes).tolist()
        if actual_validation != list(manifest["validation_class_counts"]):
            raise ValueError("Validation labels disagree with declared counts")
        expected_validation_per_class = 500
        if actual_validation != [expected_validation_per_class] * num_classes:
            raise ValueError(f"{dataset_name} validation must contain {expected_validation_per_class} examples per class")
        fingerprint = manifest.get("source_labels_checksum")
        if fingerprint and fingerprint != config_fingerprint(labels.tolist()):
            raise ValueError("Source-label checksum does not match the actual dataset")
        if manifest.get("source_indices_checksum") and manifest["source_indices_checksum"] != config_fingerprint(list(range(len(labels)))):
            raise ValueError("Source-index checksum does not match the actual dataset")
    return {"num_bundles": 14, "local_examples": 28000, "validation_examples": 5000,
            "unused_examples": len(unused), "all_allocations_disjoint": True, "num_classes": num_classes}
