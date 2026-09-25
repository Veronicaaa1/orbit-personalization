from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from orbit_cifar.config_schema import load_config
from orbit_cifar.constants import ACTIVE_MODELS
from orbit_cifar.data.datasets import build_transforms
from orbit_cifar.data.partition import (
    build_balanced_manifest,
    build_dataset_manifest,
    validate_dataset_manifest,
)
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.matrix import ScientificKey, final_keys, pilot_families
from orbit_cifar.experiments.planner import build_execution_plan
from orbit_cifar.experiments.preflight import preflight_rows
from orbit_cifar.models.factory import create_model
from orbit_cifar.models.resnet_gn import BasicBlock
from orbit_cifar.experiments.runner import run_experiment
from orbit_cifar.utils.reproducibility import capture_rng_state, restore_rng_state
from orbit_cifar.analysis.active import _stats
import pandas as pd


CONFIG = Path(__file__).parents[1] / "configs" / "mnist_cifar10_resnet_mlp_3seed.yaml"


def active_config():
    return load_config(CONFIG)


def test_exact_matrix_allowlist_and_locally_blocked_legacy(tmp_path: Path) -> None:
    cfg = active_config()
    cfg.project_root = str(tmp_path)
    cfg.legacy_reuse.source_root = str(tmp_path / "unmounted_drive")
    keys = list(final_keys(cfg))
    assert len(keys) == 48
    assert len(pilot_families(cfg)) == 4
    assert {(key.dataset, key.model) for key in keys} == {
        (dataset, model) for dataset in ("mnist", "cifar10") for model in ACTIVE_MODELS
    }
    plan = build_execution_plan(cfg)
    assert plan.summary == {
        "expected_final_scientific_cells": 48,
        "validated_legacy_final_cells": 0,
        "already_complete_native_cells": 0,
        "resumable_native_cells": 0,
        "new_final_training_required": 36,
        "ready_cells": 0,
        "awaiting_data_cells": 36,
        "awaiting_manifest_cells": 0,
        "awaiting_selection_cells": 0,
        "blocked_cells": 12,
    }
    assert plan.pilot_summary["conceptual_pilot_candidate_runs"] == 16
    assert plan.pilot_summary["new_pilot_candidate_runs"] == 12
    assert len(preflight_rows(cfg)) == 16
    override_cfg = load_config(CONFIG, [
        "legacy_reuse.retrain_overrides=[cifar10/resnet20_gn/main/seed_001]"
    ])
    override_cfg.project_root = str(tmp_path)
    override_cfg.legacy_reuse.source_root = str(tmp_path / "unmounted_drive")
    override_plan = build_execution_plan(override_cfg)
    assert override_plan.summary["blocked_cells"] == 11
    assert override_plan.summary["new_final_training_required"] == 37


def test_name_selection_loads_real_defaults_and_rejects_conflicts() -> None:
    cfg = load_config(CONFIG, ["data.name=mnist", "model.name=mlp_ln"])
    assert (cfg.model.input_channels, cfg.model.input_height, cfg.model.input_width, cfg.model.num_classes) == (1, 28, 28, 10)
    assert list(cfg.model.hidden_dimensions) == [256, 128]
    with pytest.raises(ValueError, match="conflicts"):
        load_config(CONFIG, ["data.name=mnist", "model.num_classes=11"])
    with pytest.raises(ValueError, match="Unknown model"):
        load_config(CONFIG, ["model.name=third_architecture"])
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_config(CONFIG, ["data.name=unsupported_dataset"])
    with pytest.raises(ValueError, match="final.datasets must be exactly"):
        load_config(CONFIG, ["final.datasets=[mnist,cifar10,unsupported_dataset]"])


@pytest.mark.parametrize("dataset,count,unused", [("mnist", 6000, 27000), ("cifar10", 5000, 17000)])
def test_supported_dataset_manifests(dataset, count, unused):
    targets = np.repeat(np.arange(10), count)
    manifest = build_dataset_manifest(dataset, targets)
    assert validate_dataset_manifest(manifest, targets)["unused_examples"] == unused
    balanced = build_balanced_manifest(targets, manifest["validation_indices"], dataset)
    assert all(bundle["class_counts"] == [200] * 10 for bundle in balanced["bundles"])


def test_mnist_transform_and_formal_model_shapes() -> None:
    names = [type(operation).__name__ for operation in build_transforms(True, "mnist").transforms]
    assert names == ["ToTensor", "Normalize"]
    cases = [((1, 28, 28), 10), ((3, 32, 32), 10)]
    for shape, classes in cases:
        for name in ACTIVE_MODELS:
            model = create_model(name, num_classes=classes, input_channels=shape[0], input_height=shape[1], input_width=shape[2])
            assert model(torch.randn(2, *shape)).shape == (2, classes)
            assert not any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)) for module in model.modules())
            if name == "resnet20_gn":
                assert sum(isinstance(module, BasicBlock) for module in model.modules()) == 9
            else:
                assert not any(isinstance(module, nn.Conv2d) for module in model.modules())




def test_direct_training_cannot_bypass_legacy_block(tmp_path: Path) -> None:
    cfg = active_config(); cfg.project_root = str(tmp_path)
    with pytest.raises(RuntimeError, match="reserved for legacy reuse"):
        run_experiment(cfg, "no_communication", 1, stage="final", lambda_override=0.0)


def test_small_n_statistics() -> None:
    assert math.isnan(_stats(pd.Series([1.0]))["sample_sd"])
    assert math.isnan(_stats(pd.Series([1.0]))["se"])


def test_rng_payload_types_are_validated() -> None:
    state = capture_rng_state()
    restore_rng_state(state)
    broken = dict(state)
    broken["torch_cpu"] = state["torch_cpu"].to(torch.int64)
    with pytest.raises(TypeError, match="uint8"):
        restore_rng_state(broken)
