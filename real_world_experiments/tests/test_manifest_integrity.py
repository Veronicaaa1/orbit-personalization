from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from orbit_cifar.config_schema import load_config
from orbit_cifar.data.manifest import save_manifest, validate_manifest_pair
from orbit_cifar.data.partition import build_balanced_manifest, build_dataset_manifest, build_split_manifest, validate_dataset_manifest
from orbit_cifar.experiments.inputs import DatasetInputs, _fully_symmetric_dataset_inputs


@pytest.fixture(scope="module")
def assignments():
    labels = np.repeat(np.arange(10), 5000)
    heterogeneous = build_dataset_manifest("cifar10", labels)
    balanced = build_balanced_manifest(labels, heterogeneous["validation_indices"], "cifar10",
                                       class_names=heterogeneous["class_names"])
    return labels, heterogeneous, balanced


def test_valid_assignments_share_one_source_probe(assignments):
    labels, heterogeneous, balanced = assignments
    validate_manifest_pair(heterogeneous, balanced, labels, "cifar10")


def test_different_valid_probe_cannot_leak_into_fully_symmetric_training(tmp_path: Path, assignments):
    labels, heterogeneous, _ = assignments
    other = build_dataset_manifest("cifar10", labels, split_seed=999)
    wrong = build_balanced_manifest(labels, other["validation_indices"], "cifar10", split_seed=999,
                                   class_names=heterogeneous["class_names"])
    assert validate_dataset_manifest(wrong, labels)["all_allocations_disjoint"]
    assert set(heterogeneous["validation_indices"]) & {index for bundle in wrong["bundles"] for index in bundle["indices"]}
    cfg = load_config(Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml")
    cfg.project_root = str(tmp_path)
    save_manifest(wrong, tmp_path / str(cfg.manifest_root) / "cifar10/balanced.json")
    training = TensorDataset(torch.zeros(1, 3, 32, 32), torch.zeros(1, dtype=torch.long))
    training.targets = labels
    inputs = DatasetInputs(training, DataLoader(training), heterogeneous, lambda image: image,
                           class_names=tuple(heterogeneous["class_names"]))
    with pytest.raises(ValueError, match="Cross-manifest validation"):
        _fully_symmetric_dataset_inputs(inputs, cfg)


@pytest.mark.parametrize("mutation", ["validation_order", "source_split", "source_version", "source_labels"])
def test_manifest_pair_rejects_source_and_probe_mismatch(assignments, mutation):
    labels, heterogeneous, original = assignments
    balanced = copy.deepcopy(original)
    if mutation == "validation_order":
        balanced["validation_indices"][:2] = balanced["validation_indices"][:2][::-1]
    elif mutation == "source_split":
        balanced["source_split"] = "test"
    elif mutation == "source_version":
        balanced["dataset_version"] = "unrelated-source"
    else:
        balanced["source_labels"][0] = 1
    with pytest.raises(ValueError):
        validate_manifest_pair(heterogeneous, balanced, labels, "cifar10")


@pytest.mark.parametrize("index", [0.5, True, "0"])
def test_fractional_boolean_and_string_indices_rejected(assignments, index):
    labels, original, _ = assignments
    manifest = copy.deepcopy(original)
    manifest["bundles"][0]["indices"][0] = index
    with pytest.raises(ValueError, match="indices must be integers"):
        validate_dataset_manifest(manifest, labels)


def test_self_declared_balanced_profiles_cannot_masquerade_as_heterogeneous(assignments):
    labels, _, original = assignments
    manifest = copy.deepcopy(original)
    manifest["manifest_type"] = "heterogeneous"
    manifest["profile_definitions"] = {f"profile_{index}": [200] * 10 for index in range(1, 5)}
    for bundle, profile in zip(manifest["bundles"], [1] * 2 + [2] * 3 + [3] * 4 + [4] * 5):
        bundle["profile_name"] = f"profile_{profile}"
    with pytest.raises(ValueError, match="canonical scientific profiles"):
        validate_dataset_manifest(manifest, labels)


def test_source_labels_are_checked_even_when_manifest_is_read_offline(assignments):
    _, original, _ = assignments
    manifest = copy.deepcopy(original)
    manifest["source_labels"][manifest["bundles"][0]["indices"][0]] = 9
    with pytest.raises(ValueError, match="labels|checksum"):
        validate_dataset_manifest(manifest)


def test_historical_cifar10_optional_metadata_normalizes_to_actual_source(assignments):
    labels, _, _ = assignments
    old = build_split_manifest(labels)
    balanced = build_balanced_manifest(labels, old["validation_indices"], "cifar10", class_names=old["class_names"])
    validate_manifest_pair(old, balanced, labels, "cifar10")


def test_manifest_verifier_cli_rejects_a_different_balanced_probe(assignments, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orbit_cifar.cli import verify_run

    labels, heterogeneous, _ = assignments
    other = build_dataset_manifest("cifar10", labels, split_seed=999)
    wrong = build_balanced_manifest(labels, other["validation_indices"], "cifar10", split_seed=999,
                                   class_names=heterogeneous["class_names"])
    cfg = load_config(Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml")
    cfg.project_root = str(tmp_path)
    directory = tmp_path / str(cfg.manifest_root) / "cifar10"
    save_manifest(heterogeneous, directory / "heterogeneous.json")
    save_manifest(wrong, directory / "balanced.json")
    monkeypatch.setattr(verify_run, "parse_config", lambda *args: (SimpleNamespace(paths=[], verify_manifest=True), cfg))
    monkeypatch.setattr(verify_run, "load_dataset", lambda *args, **kwargs: SimpleNamespace(targets=labels, train=True))
    with pytest.raises(ValueError, match="Cross-manifest validation"):
        verify_run.main([])
