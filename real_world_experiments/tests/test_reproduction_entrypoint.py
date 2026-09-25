from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/reproduce_realworld.py"
spec = importlib.util.spec_from_file_location("reproduction_entrypoint", SCRIPT)
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


@pytest.mark.parametrize("arguments", [[], ["--stage", "all"]])
def test_default_and_unconfirmed_stage_never_execute(tmp_path, monkeypatch, capsys, arguments):
    def forbidden(*args):
        raise AssertionError("An unconfirmed command executed a stage")

    monkeypatch.setattr(reproduction, "execute_stage", forbidden)
    work = tmp_path / "untouched"
    assert reproduction.main([*arguments, "--work-root", str(work)]) == 0
    assert not work.exists()
    assert '"final_cells": 48' in capsys.readouterr().out


@pytest.mark.parametrize("directory", [".", "real_world_experiments", "configs", "data", "numerical experiments", "real_world_experiments/results"])
def test_work_root_cannot_overlap_existing_project_trees(directory):
    with pytest.raises(ValueError, match="Work root"):
        reproduction.checked_work_root(reproduction.PROJECT / directory)


def test_download_cannot_modify_external_data(tmp_path):
    with pytest.raises(SystemExit):
        reproduction.main(["--stage", "data", "--download", "--execute",
                           "--work-root", str(tmp_path / "work"), "--data-root", str(tmp_path / "existing")])
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "existing").exists()


def test_manifest_copy_is_exact_idempotent_and_never_overwrites(tmp_path):
    source, target = tmp_path / "archive.json", tmp_path / "new/manifest.json"
    source.write_bytes(b'{"ordered_indices": [2, 0, 1]}\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    reproduction.copy_manifest(source, target, digest)
    reproduction.copy_manifest(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"different existing split")
    with pytest.raises(ValueError, match="Refusing to replace"):
        reproduction.copy_manifest(source, target, digest)
    assert target.read_bytes() == b"different existing split"
    with pytest.raises(ValueError, match="checksum"):
        reproduction.copy_manifest(source, tmp_path / "must-not-exist.json", "0" * 64)
    assert not (tmp_path / "must-not-exist.json").exists()


def test_family_settings_and_explicit_legacy_scope(tmp_path):
    for dataset in ("mnist", "cifar10"):
        for model in ("resnet20_gn", "mlp_ln"):
            cfg = reproduction.family_config(dataset, model, tmp_path, tmp_path / "data")
            assert cfg.experiment.name == "mnist_cifar10_reproduction_v1"
            assert list(cfg.final.datasets) == ["mnist", "cifar10"]
            assert cfg.training.epochs == 200 and cfg.artifacts.save_final_logits is True
            assert cfg.training.num_workers == (0 if (dataset, model) == ("cifar10", "mlp_ln") else 1)
            assert cfg.training.persistent_workers == (cfg.training.num_workers > 0)
            assert cfg.data.download is False
            assert cfg.legacy_reuse.rerun_pilot_family is True
            assert len(set(cfg.legacy_reuse.retrain_overrides)) == 12


def test_pilot_worker_override_preserves_final_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from orbit_cifar.experiments import pilot

    captured = []
    monkeypatch.setattr(pilot, "run_pilot", lambda cfg: captured.append(cfg))
    cfg = reproduction.family_config("cifar10", "mlp_ln", tmp_path, tmp_path / "data")
    reproduction.execute_stage("pilot", [cfg], SimpleNamespace(), tmp_path)
    assert len(captured) == 1
    assert captured[0].training.num_workers == 1
    assert captured[0].training.persistent_workers is True
    assert cfg.training.num_workers == 0
    assert cfg.training.persistent_workers is False


def test_all_stages_are_explicit_and_ordered_without_training(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(reproduction, "execute_stage", lambda stage, *args: calls.append(stage))
    assert reproduction.main(["--stage", "all", "--execute", "--work-root", str(tmp_path / "work")]) == 0
    assert calls == ["data", "pilot", "select", "final", "verify", "analyze"]
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("failure", ["blocked", "incomplete", "unstable"])
def test_final_stage_rejects_unsuccessful_execution(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    import pandas as pd
    from orbit_cifar.experiments import final

    rows = [{"action": "trained", "numerically_stable": True} for _ in range(4)]
    if failure == "blocked":
        rows[0]["action"] = "skipped_blocked"
    elif failure == "incomplete":
        rows.pop()
    else:
        rows[0]["numerically_stable"] = False
    monkeypatch.setattr(final, "run_active_cells", lambda *args: pd.DataFrame(rows))
    cfg = reproduction.family_config("mnist", "mlp_ln", tmp_path, tmp_path / "data")
    with pytest.raises(RuntimeError):
        reproduction.execute_stage("final", [cfg], SimpleNamespace(seed=1), tmp_path)
