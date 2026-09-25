"""Architecture-neutral data streams and runtime source metadata, without real data."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from orbit_cifar.config_schema import load_config
from orbit_cifar.data.datasets import build_transforms, dataset_spec
from orbit_cifar.data.loaders import build_node_loaders, set_loader_epoch
from orbit_cifar.data.manifest import save_manifest
from orbit_cifar.data.partition import build_dataset_manifest
from orbit_cifar.experiments import inputs
from orbit_cifar.experiments.families import family_config
from orbit_cifar.models.factory import create_model
from orbit_cifar.training.seed_plan import build_seed_plan

CONFIG = Path(__file__).resolve().parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml"


class _SyntheticImages(Dataset):
    def __init__(self, name: str, count: int = 98):
        self.spec = dataset_spec(name)
        self.targets = (np.arange(count) % self.spec.num_classes).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        channels, height, width = self.spec.input_shape
        shape = (height, width) if channels == 1 else (height, width, channels)
        array = np.random.default_rng(index).integers(0, 256, shape, dtype=np.uint8)
        return Image.fromarray(array), self.targets[index]


def _streams(loaders, epoch):
    set_loader_epoch(loaders, epoch)
    collected = {}
    for loader in loaders:
        positions = list(loader.sampler)
        batches = list(loader)
        collected[loader.dataset.bundle_id] = (
            tuple(positions), torch.cat([batch[0] for batch in batches]), torch.cat([batch[1] for batch in batches]))
    return collected


def _assert_same_streams(left, right):
    assert left.keys() == right.keys()
    for bundle in left:
        assert left[bundle][0] == right[bundle][0]
        assert torch.equal(left[bundle][1], right[bundle][1])
        assert torch.equal(left[bundle][2], right[bundle][2])


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
@pytest.mark.parametrize("seed", [1, 3])
def test_sampler_and_augmentation_do_not_depend_on_architecture(dataset, seed):
    cfg = load_config(CONFIG)
    base = _SyntheticImages(dataset)
    manifest = {"bundles": [{"indices": list(range(bundle * 7, (bundle + 1) * 7))} for bundle in range(14)]}
    observed = {}
    for architecture in ("resnet20_gn", "mlp_ln"):
        family = family_config(cfg, dataset, architecture)
        # Real model construction consumes different amounts of global RNG.
        torch.manual_seed(17)
        model = create_model(architecture, num_classes=int(family.model.num_classes),
                             input_channels=int(family.model.input_channels),
                             input_height=int(family.model.input_height), input_width=int(family.model.input_width))
        assert model is not None
        data_seed = build_seed_plan(seed, namespace="final")["data_run_seed"]
        loaders = build_node_loaders(base, manifest, "aligned", build_transforms(True, dataset),
                                     data_seed, batch_size=3, num_workers=0, persistent_workers=False)
        observed[architecture] = _streams(loaders, 7)
        _assert_same_streams(observed[architecture], _streams(loaders, 7))
        changed = _streams(loaders, 8)
        assert any(observed[architecture][bundle][0] != changed[bundle][0] for bundle in range(14))
        if dataset != "mnist":
            # Compare a fixed source image so augmentation differences are not sampler order.
            assert any(not torch.equal(loader.dataset[(0, 7)][0], loader.dataset[(0, 8)][0]) for loader in loaders)
    _assert_same_streams(observed["resnet20_gn"], observed["mlp_ln"])


def test_misalignment_moves_bundles_with_their_original_random_streams():
    base = _SyntheticImages("cifar10")
    manifest = {"bundles": [{"indices": list(range(bundle * 7, (bundle + 1) * 7))} for bundle in range(14)]}
    streams = []
    for assignment in ("aligned", "misaligned"):
        loaders = build_node_loaders(base, manifest, assignment, build_transforms(True, "cifar10"),
                                     3, batch_size=4, num_workers=0, persistent_workers=False)
        streams.append(_streams(loaders, 4))
    _assert_same_streams(*streams)








@pytest.mark.parametrize("mismatch", ["dataset", "dataset_version", "split_seed"])
def test_runtime_rejects_individually_valid_wrong_source_manifest(tmp_path, monkeypatch, mismatch):
    cfg = family_config(load_config(CONFIG), "mnist", "mlp_ln")
    cfg.project_root = str(tmp_path)
    source = _SyntheticImages("mnist", count=60000)
    source.targets = np.repeat(np.arange(10), 6000).tolist()
    manifest = build_dataset_manifest("mnist", source.targets)
    manifest = copy.deepcopy(manifest)
    manifest[mismatch] = {"dataset": "cifar10", "dataset_version": "unrelated-source", "split_seed": 1009}[mismatch]
    save_manifest(manifest, tmp_path / cfg.manifest_root / "mnist/heterogeneous.json")
    monkeypatch.setattr(inputs, "load_dataset", lambda *args, **kwargs: source)
    with pytest.raises(ValueError, match="requested official|version|split seed"):
        inputs._real_dataset_inputs(cfg, "pilot")




def test_runtime_rejects_training_manifest_against_actual_test_split(tmp_path, monkeypatch):
    cfg = family_config(load_config(CONFIG), "mnist", "mlp_ln")
    cfg.project_root = str(tmp_path)
    source = _SyntheticImages("mnist", count=60000)
    source.targets = np.repeat(np.arange(10), 6000).tolist()
    source.train = False
    manifest = build_dataset_manifest("mnist", source.targets)
    save_manifest(manifest, tmp_path / cfg.manifest_root / "mnist/heterogeneous.json")
    monkeypatch.setattr(inputs, "load_dataset", lambda *args, **kwargs: source)
    with pytest.raises(ValueError, match="official training dataset split"):
        inputs._real_dataset_inputs(cfg, "pilot")

