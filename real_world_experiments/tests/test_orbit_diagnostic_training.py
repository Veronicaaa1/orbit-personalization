"""Bounded deterministic CPU fixtures compare observation on/off and resume."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from orbit_cifar.config_schema import load_config
from orbit_cifar.data.loaders import EpochRandomSampler
from orbit_cifar.experiments.runner import build_smoke_inputs, run_experiment, smoke_config
import orbit_cifar.experiments.runner as runner
from orbit_cifar.experiments.orbit_diagnostics import GEOMETRY, MOTION, STATUS

CONFIG = Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml"


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def compare_native(first, second):
    for name in ("last.pt", "final.pt"):
        a = torch.load(first / "checkpoints" / name, map_location="cpu", weights_only=False)
        b = torch.load(second / "checkpoints" / name, map_location="cpu", weights_only=False)
        for field in ("models", "optimizers", "scheduler", "epoch", "global_step", "rng_state"):
            equal(a[field], b[field])
    for name in ("checkpoint_metrics.csv", "train_diagnostics.csv"):
        a, b = (pd.read_csv(root / "metrics" / name) for root in (first, second))
        columns = [c for c in a.columns if "wall_seconds" not in c]
        pd.testing.assert_frame_equal(a[columns], b[columns], check_exact=True)


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(root, enabled=True):
    cfg = smoke_config(load_config(CONFIG), output_root=root)
    cfg.training.epochs = 2
    cfg.orbit_diagnostics.enabled = enabled
    # Nonzero momentum exercises real optimizer buffers, not only empty state.
    cfg.training.momentum = .4
    return cfg


def execute(cfg):
    return run_experiment(cfg, "main", 1, stage="final", lambda_override=.03,
                          datasets=build_smoke_inputs(seed=57, samples_per_bundle=4, probe_samples=10))


def test_recording_on_off_equal_training_states_metrics_rng_and_order(tmp_path, monkeypatch, cpu_threads):
    original = EpochRandomSampler.__iter__
    trace = []

    def observed(sampler):
        order = list(original(sampler))
        trace.append((sampler.bundle_id, sampler.epoch, order))
        return iter(order)

    monkeypatch.setattr(EpochRandomSampler, "__iter__", observed)
    off = execute(config(tmp_path / "off", False)).run_directory
    off_trace = trace.copy()
    trace.clear()
    on = execute(config(tmp_path / "on", True)).run_directory
    assert trace == off_trace
    compare_native(off, on)
    assert not (off / GEOMETRY).exists()
    frame = pd.read_csv(on / GEOMETRY)
    assert sorted(frame.epoch.unique()) == [0, 1, 2]
    assert len(frame.query("metric == 'W_k'")) == 12
    assert len(frame.query("metric == 'B_kl'")) == 18


def test_interrupted_cpu_training_restores_diagnostic_generation(tmp_path, monkeypatch, cpu_threads):
    cfg = config(tmp_path / "resumed")
    real_save = runner.save_checkpoint

    def interrupted(path, *args, **kwargs):
        if Path(path).name == "last.pt" and args[3] == 2:
            raise KeyboardInterrupt("after diagnostics publication, before checkpoint commit")
        return real_save(path, *args, **kwargs)

    monkeypatch.setattr(runner, "save_checkpoint", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(cfg)
    monkeypatch.setattr(runner, "save_checkpoint", real_save)
    cfg.resume = True
    resumed = execute(cfg).run_directory
    full = execute(config(tmp_path / "full")).run_directory
    compare_native(full, resumed)
    for name in (GEOMETRY, MOTION):
        pd.testing.assert_frame_equal(pd.read_csv(full / name), pd.read_csv(resumed / name), check_exact=True)
    # Re-finalization never duplicates diagnostics, changes retained raw arrays,
    # or adds epochs to the original terminal-window scalar computation.
    before = {name: (resumed / name).read_bytes() for name in (GEOMETRY, MOTION, STATUS)}
    execute(cfg)
    assert before == {name: (resumed / name).read_bytes() for name in before}


def test_old_run_resume_does_not_require_or_fabricate_diagnostics(tmp_path, monkeypatch, cpu_threads):
    cfg = config(tmp_path, False)
    first = execute(cfg).run_directory
    cfg.resume = True
    cfg.orbit_diagnostics.enabled = True
    def forbidden_training(*args, **kwargs):
        raise AssertionError("Already trained epochs must not be repeated")
    monkeypatch.setattr(runner.SynchronousDecentralizedTrainer, "train_epoch", forbidden_training)
    execute(cfg)
    assert not (first / GEOMETRY).exists()
