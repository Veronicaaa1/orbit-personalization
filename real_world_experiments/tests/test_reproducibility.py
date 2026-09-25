from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from orbit_cifar.data.datasets import IndexedTransformDataset
from orbit_cifar.data.loaders import EpochRandomSampler, set_loader_epoch
from orbit_cifar.training.seed_plan import build_seed_plan
from orbit_cifar.utils.reproducibility import derive_seed


class _RandomValueTransform:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return torch.rand(1)


def test_stable_component_seeds_are_distinct() -> None:
    assert derive_seed(7, "sampler", 3, 2) == derive_seed(7, "sampler", 3, 2)
    assert derive_seed(7, "sampler", 3, 2) != derive_seed(7, "sampler", 4, 2)
    plan = build_seed_plan(7)
    assert len(set(plan["bundle_sampler"])) == 14
    assert plan == build_seed_plan(7)
    assert plan != build_seed_plan(8)


def test_persistent_worker_receives_sampler_epoch() -> None:
    base = TensorDataset(torch.arange(8, dtype=torch.float32).unsqueeze(1), torch.arange(8) % 2)
    dataset = IndexedTransformDataset(base, range(8), _RandomValueTransform(), run_seed=4, bundle_id=2)
    sampler = EpochRandomSampler(dataset, run_seed=4, bundle_id=2)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=1, persistent_workers=True)

    def collect(epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        set_loader_epoch([loader], epoch)
        batches = list(loader)
        return torch.cat([batch[0] for batch in batches]), torch.cat([batch[1] for batch in batches])

    values_0, labels_0 = collect(0)
    values_1, labels_1 = collect(1)
    values_0_repeat, labels_0_repeat = collect(0)
    assert not torch.equal(values_0, values_1) or not torch.equal(labels_0, labels_1)
    assert torch.equal(values_0, values_0_repeat)
    assert torch.equal(labels_0, labels_0_repeat)
