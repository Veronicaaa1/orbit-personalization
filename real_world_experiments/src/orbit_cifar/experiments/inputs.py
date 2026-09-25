"""Shared data construction for every architecture and isolated smoke fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset

from orbit_cifar.constants import ALIGNED_PROFILE_NAMES, FULLY_SYMMETRIC_PROFILE_NAME, NUM_NODES
from orbit_cifar.data.datasets import build_transforms, load_dataset
from orbit_cifar.data.loaders import build_probe_loader
from orbit_cifar.data.manifest import load_manifest, validate_manifest_pair, validate_manifest_source
from orbit_cifar.data.partition import build_balanced_manifest, build_fully_symmetric_manifest

@dataclass(frozen=True)
class DatasetInputs:
    training_dataset: Dataset[Any]
    probe_loader: DataLoader[Any]
    manifest: dict[str, Any]
    train_transform: Callable[[Any], torch.Tensor]
    dataset_name: str = "cifar10"
    class_names: tuple[str, ...] = ()


def _project_path(cfg: DictConfig, value: str) -> Path:
    project_root = Path(str(cfg.project_root)).expanduser().resolve()
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _real_dataset_inputs(cfg: DictConfig, stage: str) -> DatasetInputs:
    data_root = _project_path(cfg, str(cfg.data.root))
    name = str(cfg.data.name)
    training_dataset = load_dataset(name, data_root, train=True, download=bool(cfg.data.download))
    if OmegaConf.select(cfg, "experiment.name"):
        manifest_path = _project_path(cfg, str(cfg.manifest_root)) / name / "heterogeneous.json"
    else:
        manifest_path = _project_path(cfg, str(cfg.manifest_path))
    manifest = load_manifest(manifest_path, targets=training_dataset.targets)
    validate_manifest_source(manifest, training_dataset, name, str(cfg.data.dataset_version), int(cfg.data.split_seed))
    if stage in {"pilot", "preflight"}:
        probe_dataset = training_dataset
        probe_indices = manifest["validation_indices"]
    elif stage == "final":
        probe_dataset = load_dataset(name, data_root, train=False, download=bool(cfg.data.download))
        probe_indices = None
    else:
        raise ValueError(f"Unknown run stage: {stage}")
    probe_loader = build_probe_loader(
        probe_dataset,
        probe_indices,
        build_transforms(train=False, dataset_name=name),
        batch_size=int(cfg.training.batch_size_per_node),
        num_workers=int(cfg.training.num_workers),
        persistent_workers=bool(cfg.training.persistent_workers),
    )
    return DatasetInputs(
        training_dataset, probe_loader, manifest, build_transforms(train=True, dataset_name=name),
        dataset_name=name, class_names=tuple(str(value) for value in manifest["class_names"]),
    )


def _fully_symmetric_dataset_inputs(inputs: DatasetInputs, cfg: DictConfig) -> DatasetInputs:
    """Replace only the local-bundle manifest while retaining the fixed probe and dataset."""
    if inputs.manifest.get("manifest_type") == "fully_symmetric_smoke":
        if OmegaConf.select(cfg, "experiment.name"):
            raise ValueError("Smoke assignments are confined to the isolated test execution path")
        return inputs
    targets = getattr(inputs.training_dataset, "targets", None)
    if targets is None:
        raise TypeError("Fully symmetric data construction requires a training dataset with CIFAR-style targets")
    if OmegaConf.select(cfg, "experiment.name"):
        balanced_path = _project_path(cfg, str(cfg.manifest_root)) / inputs.dataset_name / "balanced.json"
        if balanced_path.is_file():
            manifest = load_manifest(balanced_path, targets=targets)
        else:
            manifest = build_balanced_manifest(
                targets, inputs.manifest["validation_indices"], inputs.dataset_name,
                split_seed=int(inputs.manifest.get("split_seed", cfg.data.split_seed)),
                dataset_version=str(inputs.manifest.get("dataset_version", cfg.data.dataset_version)),
                class_names=inputs.manifest["class_names"],
            )
    else:
        manifest = build_fully_symmetric_manifest(
            targets,
            validation_indices=inputs.manifest["validation_indices"],
            split_seed=int(inputs.manifest.get("split_seed", cfg.data.split_seed)),
            dataset_version=str(inputs.manifest.get("dataset_version", cfg.data.dataset_version)),
        )
    validate_manifest_pair(inputs.manifest, manifest, targets, inputs.dataset_name)
    return DatasetInputs(inputs.training_dataset, inputs.probe_loader, manifest, inputs.train_transform,
                         inputs.dataset_name, inputs.class_names)


def build_smoke_inputs(
    seed: int = 17,
    samples_per_bundle: int = 4,
    probe_samples: int = 20,
    assignment: str = "aligned",
) -> DatasetInputs:
    """Create tiny deterministic tensors without downloading CIFAR-10."""
    generator = torch.Generator().manual_seed(seed)
    total = NUM_NODES * samples_per_bundle
    images = torch.randn((total, 3, 8, 8), generator=generator)
    labels = torch.arange(total, dtype=torch.long) % 10
    training = TensorDataset(images, labels)
    bundles = []
    for bundle_id in range(NUM_NODES):
        start = bundle_id * samples_per_bundle
        bundles.append(
            {
                "bundle_id": bundle_id,
                "profile_name": (
                    FULLY_SYMMETRIC_PROFILE_NAME
                    if assignment == "fully_symmetric"
                    else ALIGNED_PROFILE_NAMES[bundle_id]
                ),
                "class_counts": [0] * 10,
                "indices": list(range(start, start + samples_per_bundle)),
            }
        )
    manifest = {"bundles": bundles}
    if assignment == "fully_symmetric":
        manifest["manifest_type"] = "fully_symmetric_smoke"
    probe_images = torch.randn((probe_samples, 3, 8, 8), generator=generator)
    probe_labels = torch.arange(probe_samples, dtype=torch.long) % 10
    probe_loader = DataLoader(TensorDataset(probe_images, probe_labels), batch_size=10, shuffle=False)
    return DatasetInputs(training, probe_loader, manifest, lambda image: image)


def smoke_config(cfg: DictConfig, output_root: str | Path | None = None) -> DictConfig:
    """Return a non-mutating tiny CPU configuration suitable for CI."""
    updates = {
        "device": "cpu",
        "resume": False,
        "training": {
            "epochs": 1,
            "batch_size_per_node": 2,
            "learning_rate": 0.01,
            "warmup_epochs": 0,
            "minimum_learning_rate": 0.001,
            "eval_every_epochs": 1,
            "checkpoint_every_epochs": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "terminal_window_checkpoints": 2,
        },
        "model": {"name": "tiny_cnn", "initialization_rho": 0.01},
        "analysis": {"diagnostic_partitions": 20},
    }
    if output_root is not None:
        updates["output_root"] = str(output_root)
    result = OmegaConf.merge(cfg, OmegaConf.create(updates))
    for key in ("experiment", "legacy_reuse", "_composition"):
        result.pop(key, None)
    return result
