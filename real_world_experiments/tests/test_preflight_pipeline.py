from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from orbit_cifar.cli.common import default_config_path
from orbit_cifar.config_schema import load_config
from orbit_cifar.experiments import preflight, runner
from orbit_cifar.experiments.inputs import DatasetInputs


def test_short_preflight_reaches_entire_real_pipeline_with_synthetic_inputs(tmp_path: Path, monkeypatch):
    cfg = load_config(default_config_path(), ["device=cpu", "training.batch_size_per_node=2"])
    cfg.project_root = str(tmp_path)
    generator = torch.Generator().manual_seed(8)
    training = TensorDataset(torch.randn(28, 1, 28, 28, generator=generator), torch.arange(28) % 10)
    probe = TensorDataset(torch.randn(10, 1, 28, 28, generator=generator), torch.arange(10))
    manifest = {"bundles": [{"bundle_id": node, "profile_name": profile,
                             "class_counts": [0] * 10, "indices": [2 * node, 2 * node + 1]}
                            for node, profile in enumerate(["profile_1"] * 2 + ["profile_2"] * 3 + ["profile_3"] * 4 + ["profile_4"] * 5)]}
    inputs = DatasetInputs(training, DataLoader(probe, batch_size=10), manifest, lambda image: image,
                           dataset_name="mnist", class_names=tuple(map(str, range(10))))
    stages = []

    def execute(family, *args, **kwargs):
        assert family.training.epochs == 1 and family.training.warmup_epochs == 0
        stages.append("data")
        return runner.run_experiment(family, *args, **kwargs, datasets=inputs)

    monkeypatch.setattr(preflight, "run_experiment", execute)
    original_step = runner.SynchronousDecentralizedTrainer.train_step

    def train_step(self, *args, **kwargs):
        assert len(self.models) == 14 and len(self.optimizers) == 14
        stages.extend(["model", "optimizer", "training_step"])
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(runner.SynchronousDecentralizedTrainer, "train_step", train_step)
    results = preflight.run_preflights(cfg, dataset="mnist", model="mlp_ln", condition="main",
                                       output_directory=tmp_path / "preflight-results")
    assert results.status.tolist() == ["PASS"], results.to_dict("records")
    run = Path(results.iloc[0].run_directory)
    checkpoint = torch.load(run / "checkpoints/final.pt", weights_only=False, map_location="cpu")
    assert checkpoint["epoch"] == 1 and checkpoint["global_step"] == 1
    terminal = json.loads((run / "metrics/terminal_metrics.json").read_text())
    assert terminal["probe_split"] == "validation"
    assert (run / "artifacts/confusion_matrices.npz").is_file()
    assert stages == ["data", "model", "optimizer", "training_step"]


def test_invalid_schedule_fails_before_any_pipeline_output(tmp_path: Path):
    cfg = runner.smoke_config(load_config(default_config_path()), tmp_path / "output")
    cfg.training.warmup_epochs = 5
    with pytest.raises(ValueError, match="warmup"):
        runner.run_experiment(cfg, "main", 1, lambda_override=0.1, datasets=runner.build_smoke_inputs())
    assert not (tmp_path / "output").exists()


def test_model_initialization_failure_has_failed_status(tmp_path: Path, monkeypatch):
    cfg = runner.smoke_config(load_config(default_config_path()), tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic initialization failure")

    monkeypatch.setattr(runner, "create_model", fail)
    with pytest.raises(RuntimeError, match="initialization failure"):
        runner.run_experiment(cfg, "no_communication", 1, lambda_override=0., datasets=runner.build_smoke_inputs())
    status = json.loads((tmp_path / "final/no_communication/seed_001/run_status.json").read_text())
    assert status["status"] == "failed"


def test_missing_resume_history_cannot_reset_a_saved_checkpoint(tmp_path: Path):
    cfg = runner.smoke_config(load_config(default_config_path()), tmp_path)
    inputs = runner.build_smoke_inputs(samples_per_bundle=2, probe_samples=10)
    result = runner.run_experiment(cfg, "no_communication", 1, lambda_override=0., datasets=inputs)
    last = result.run_directory / "checkpoints/last.pt"
    before = last.read_bytes()
    (result.run_directory / "metrics/checkpoint_metrics.csv").unlink()
    cfg.resume = True
    with pytest.raises(ValueError, match="Resume requires checkpoint metrics history"):
        runner.run_experiment(cfg, "no_communication", 1, lambda_override=0., datasets=inputs)
    assert last.read_bytes() == before
