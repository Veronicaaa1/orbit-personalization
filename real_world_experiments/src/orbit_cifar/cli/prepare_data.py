"""Create and validate the fixed MNIST/CIFAR-10 split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from orbit_cifar.cli.common import base_parser, parse_config
from omegaconf import OmegaConf

from orbit_cifar.data.datasets import dataset_spec, load_cifar10, load_dataset
from orbit_cifar.data.manifest import save_manifest, validate_manifest_pair
from orbit_cifar.data.partition import (
    build_balanced_manifest,
    build_dataset_manifest,
    build_split_manifest,
    validate_dataset_manifest,
    validate_split_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Prepare the fixed mutually exclusive MNIST/CIFAR-10 split manifests")
    arguments, cfg = parse_config(parser, argv)
    project_root = Path(str(cfg.project_root)).expanduser().resolve()
    data_root = Path(str(cfg.data.root)).expanduser()
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    if OmegaConf.select(cfg, "experiment.name"):
        name = str(cfg.data.name)
        dataset = load_dataset(name, data_root, train=True, download=bool(cfg.data.download))
        spec = dataset_spec(name)
        manifest = build_dataset_manifest(
            name, dataset.targets, split_seed=int(cfg.data.split_seed),
            validation_per_class=int(cfg.data.validation_per_class), dataset_version=str(cfg.data.dataset_version),
            class_names=spec.class_names,
        )
        balanced = build_balanced_manifest(
            dataset.targets, manifest["validation_indices"], name, split_seed=int(cfg.data.split_seed),
            dataset_version=str(cfg.data.dataset_version), class_names=manifest["class_names"],
        )
        validate_manifest_pair(manifest, balanced, dataset.targets, name)
        target_root = Path(str(cfg.manifest_root)).expanduser()
        if not target_root.is_absolute():
            target_root = (project_root / target_root).resolve()
        target = target_root / name / "heterogeneous.json"
        balanced_target = target_root / name / "balanced.json"
        save_manifest(manifest, target)
        save_manifest(balanced, balanced_target)
        summary = validate_dataset_manifest(manifest, targets=dataset.targets)
        balanced_summary = validate_dataset_manifest(balanced, targets=dataset.targets)
        print(f"Saved validated heterogeneous manifest to {target}")
        print(f"Saved validated balanced manifest to {balanced_target}")
        print({"heterogeneous": summary, "balanced": balanced_summary})
    else:
        dataset = load_cifar10(data_root, train=True, download=bool(cfg.data.download))
        manifest = build_split_manifest(
            dataset.targets, split_seed=int(cfg.data.split_seed),
            validation_per_class=int(cfg.data.validation_per_class), dataset_version=str(cfg.data.dataset_version),
        )
        target = Path(str(cfg.manifest_path)).expanduser()
        if not target.is_absolute():
            target = (project_root / target).resolve()
        save_manifest(manifest, target)
        summary = validate_split_manifest(manifest, targets=dataset.targets)
        print(f"Saved validated split manifest to {target}")
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
