from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from orbit_cifar.constants import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    MNIST_CLASSES,
    MNIST_MEAN,
    MNIST_STD,
)
from orbit_cifar.utils.reproducibility import derive_seed


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_shape: tuple[int, int, int]
    num_classes: int
    train_examples: int
    test_examples: int
    validation_per_class: int
    class_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", (1, 28, 28), 10, 60000, 10000, 500, MNIST_CLASSES, MNIST_MEAN, MNIST_STD),
    "cifar10": DatasetSpec(
        "cifar10", (3, 32, 32), 10, 50000, 10000, 500,
        ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"),
        CIFAR10_MEAN, CIFAR10_STD,
    ),
}


def dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[str(name)]
    except KeyError as error:
        raise ValueError(f"Unknown dataset name: {name}") from error


def build_transforms(train: bool, dataset_name: str = "cifar10") -> Callable[[Any], torch.Tensor]:
    """Build dataset-correct train or deterministic probe transforms."""
    from torchvision import transforms

    spec = dataset_spec(dataset_name)
    operations: list[Callable[..., Any]] = []
    if train and dataset_name == "cifar10":
        operations.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5)])
    operations.extend([transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)])
    return transforms.Compose(operations)


def load_cifar10(root: str | Path, train: bool, download: bool) -> Any:
    from torchvision.datasets import CIFAR10

    return CIFAR10(root=Path(root), train=train, transform=None, download=download)


def load_dataset(name: str, root: str | Path, train: bool, download: bool) -> Any:
    """Load an official torchvision split without attaching transforms."""
    from torchvision.datasets import MNIST

    if name == "cifar10":
        result = load_cifar10(root, train=train, download=download)
    elif name == "mnist":
        result = MNIST(root=Path(root), train=train, transform=None, download=download)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    spec = dataset_spec(name)
    if len(result) != (spec.train_examples if train else spec.test_examples):
        raise ValueError(f"{name} split has {len(result)} examples; expected {spec.train_examples if train else spec.test_examples}")
    targets = getattr(result, "targets", None)
    if targets is None or len(targets) != len(result):
        raise ValueError(f"{name} did not expose one target per example")
    observed = sorted(set(int(value) for value in targets))
    if observed != list(range(spec.num_classes)):
        raise ValueError(f"{name} labels are {observed}; expected 0..{spec.num_classes - 1}")
    return result




class IndexedTransformDataset(Dataset[tuple[torch.Tensor, int]]):
    """A concrete index subset with bundle-bound deterministic augmentation draws."""

    def __init__(
        self,
        base_dataset: Dataset[Any],
        indices: Sequence[int],
        transform: Callable[[Any], torch.Tensor],
        run_seed: int,
        bundle_id: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = tuple(int(index) for index in indices)
        self.transform = transform
        self.run_seed = int(run_seed)
        self.bundle_id = int(bundle_id)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch-specific augmentation stream."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int | tuple[int, int]) -> tuple[torch.Tensor, int]:
        if isinstance(position, tuple):
            position, epoch = int(position[0]), int(position[1])
        else:
            position, epoch = int(position), self.epoch
        source_index = self.indices[position]
        image, target = self.base_dataset[source_index]
        seed = derive_seed(self.run_seed, "augmentation", self.bundle_id, epoch, source_index)
        python_state = random.getstate()
        try:
            random.seed(seed)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                transformed = self.transform(image)
        finally:
            random.setstate(python_state)
        return transformed, int(target)
