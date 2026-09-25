from __future__ import annotations

from functools import partial
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from orbit_cifar.data.assignments import node_to_bundle_mapping
from orbit_cifar.data.datasets import IndexedTransformDataset
from orbit_cifar.utils.reproducibility import derive_seed


class EpochRandomSampler(Sampler[tuple[int, int]]):
    """Random sampler whose stream is a stable function of run, bundle, and epoch."""

    def __init__(self, data_source: Dataset[Any], run_seed: int, bundle_id: int) -> None:
        self.data_source = data_source
        self.run_seed = int(run_seed)
        self.bundle_id = int(bundle_id)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(derive_seed(self.run_seed, "sampler", self.bundle_id, self.epoch))
        positions = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter((int(position), self.epoch) for position in positions)

    def __len__(self) -> int:
        return len(self.data_source)


def build_node_loaders(
    base_dataset: Dataset[Any],
    manifest: dict[str, Any],
    assignment: str,
    transform: Any,
    run_seed: int,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = True,
) -> list[DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    """Create equal-length node loaders while binding random streams to bundle ids."""
    loaders: list[DataLoader[tuple[torch.Tensor, torch.Tensor]]] = []
    for bundle_id in node_to_bundle_mapping(assignment):
        dataset = IndexedTransformDataset(
            base_dataset,
            manifest["bundles"][bundle_id]["indices"],
            transform,
            run_seed,
            bundle_id,
        )
        sampler = EpochRandomSampler(dataset, run_seed, bundle_id)
        generator = torch.Generator().manual_seed(derive_seed(run_seed, "loader", bundle_id))
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                worker_init_fn=partial(_initialize_worker, run_seed, bundle_id),
                generator=generator,
                drop_last=False,
                persistent_workers=bool(persistent_workers and num_workers > 0),
            )
        )
    lengths = {len(loader) for loader in loaders}
    if len(lengths) != 1:
        raise ValueError(f"All node loaders must have equal step counts, found {sorted(lengths)}")
    return loaders


def _initialize_worker(run_seed: int, bundle_id: int, worker_id: int) -> None:
    """Seed a spawned DataLoader worker with a picklable top-level callback."""
    seed = derive_seed(run_seed, "worker", bundle_id, worker_id)
    torch.manual_seed(seed)
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)


def set_loader_epoch(loaders: Sequence[DataLoader[Any]], epoch: int) -> None:
    """Advance both sampling and augmentation streams at an epoch boundary."""
    for loader in loaders:
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)


def build_probe_loader(
    base_dataset: Dataset[Any],
    indices: Sequence[int] | None,
    transform: Any,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = True,
) -> DataLoader[Any]:
    """Build a deterministic non-shuffled validation or test loader."""
    if indices is None:
        dataset: Dataset[Any] = _TransformDataset(base_dataset, transform)
    else:
        dataset = _TransformSubset(base_dataset, indices, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )


class _TransformDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, base: Dataset[Any], transform: Any) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[index]
        return self.transform(image), int(label)


class _TransformSubset(_TransformDataset):
    def __init__(self, base: Dataset[Any], indices: Sequence[int], transform: Any) -> None:
        super().__init__(base, transform)
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return super().__getitem__(self.indices[index])
