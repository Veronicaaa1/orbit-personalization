from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from orbit_cifar.cli.verify_run import verify_run
from orbit_cifar.config_schema import config_to_dict, load_config
from orbit_cifar.constants import NUM_NODES, REFERENCE_ORBITS
from orbit_cifar.evaluation.classification import classification_tables
from orbit_cifar.experiments.completion import publish_completion, verify_completed_run, verify_resumable_run, REQUIRED_FILES
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.families import family_config
from orbit_cifar.experiments.matrix import ScientificKey, validate_active_run
from orbit_cifar.experiments.pilot import run_pilot
from orbit_cifar.experiments.planner import build_execution_plan, _status
from orbit_cifar.experiments.provenance import (
    build_scientific_identity, make_selection_record, selection_context, validate_selected,
)
from orbit_cifar.experiments.selection import select_from_csv, summarize_pilot_runs
from orbit_cifar.utils.io import atomic_write_json, atomic_write_yaml, config_fingerprint

CONFIG = Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml"


from native_fixtures import family, probe_manifest, native_fixture


def republish(root, key):
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    publication = publish_completion(root, key, status["scientific_fingerprint"], status["last_valid_epoch"], status["global_step"])
    atomic_write_json(root / "run_status.json", {**status, **publication})


def test_explicit_yaml_and_cli_override_family_defaults(tmp_path):
    directory = tmp_path / "configs"
    shutil.copytree(CONFIG.parent, directory)
    target = directory / CONFIG.name
    raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    raw["model"] = {"initialization_rho": 0.041}
    raw["data"] = {"root": "custom-data"}
    target.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(target)
    assert cfg.model.initialization_rho == 0.041
    for dataset in ("mnist", "cifar10"):
        composed = family_config(cfg, dataset, "mlp_ln")
        assert composed.model.initialization_rho == 0.041
        assert composed.data.root == "custom-data"
        assert composed.model.num_classes == 10
    overridden = load_config(target, ["model.initialization_rho=0.052"])
    assert family_config(overridden, "mnist", "mlp_ln").model.initialization_rho == 0.052
    assert "_composition" not in config_to_dict(overridden)
    overridden.model.initialization_rho = 0.07
    switched = family_config(family_config(overridden, "mnist", "mlp_ln"), "cifar10", "resnet20_gn")
    assert switched.model.initialization_rho == 0.07


@pytest.mark.parametrize("changes,condition,seed,stage", [
    ({"data.name": "imagenet"}, "main", 1, "final"),
    ({"model.name": "tiny_cnn"}, "main", 1, "final"),
    ({}, "complete_graph", 1, "final"), ({}, "large_communication", 1, "final"),
    ({}, "main", 4, "final"), ({}, "main", True, "final"), ({}, "main", 1.5, "final"),
])
def test_active_api_rejects_nonmatrix_runs(tmp_path, changes, condition, seed, stage):
    cfg = family(tmp_path)
    for path, value in changes.items():
        OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        validate_active_run(cfg, condition, seed, stage)


@pytest.mark.parametrize("field,value", [
    ("experiment_name", "another"), ("model", "resnet20_gn"), ("manifest_fingerprint", "a" * 64),
    ("validation_probe_fingerprint", "b" * 64), ("pilot_seeds", [9]), ("candidate_grid", [0, 0.1]),
    ("epoch_budget", 10), ("training_settings", {}), ("selection_rule_version", "obsolete"), ("selected_lambda", 0.3),
])
def test_selected_lambda_rejects_every_stale_binding(tmp_path, field, value):
    cfg = family(tmp_path)
    manifest = probe_manifest()
    selected = make_selection_record(cfg, {"lambda_main": 0.1, "selection_status": "primary"}, manifest)
    assert validate_selected(cfg, selected, manifest) == selected
    selected["provenance"][field] = value
    selected["provenance_fingerprint"] = config_fingerprint(selected["provenance"])
    with pytest.raises(ValueError):
        validate_selected(cfg, selected, manifest)


def test_pilot_stale_context_never_skips_or_runs(tmp_path, monkeypatch):
    cfg = family(tmp_path)
    monkeypatch.setattr("orbit_cifar.experiments.provenance.manifest_for_config", lambda cfg: probe_manifest())
    root = tmp_path / str(cfg.output_root) / "pilot/mnist/mlp_ln"
    context = selection_context(cfg)
    atomic_write_json(root / "pilot_provenance.json", {"context": context, "context_fingerprint": config_fingerprint(context)})
    cfg.training.learning_rate = 0.2
    monkeypatch.setattr("orbit_cifar.experiments.pilot.run_experiment", lambda *a, **k: pytest.fail("stale pilot must not execute"))
    with pytest.raises(ValueError, match="stale scientific provenance"):
        run_pilot(cfg)


def test_wrong_family_pilot_csv_rejected(tmp_path, monkeypatch):
    cfg = family(tmp_path)
    monkeypatch.setattr("orbit_cifar.experiments.provenance.manifest_for_config", lambda cfg: probe_manifest())
    context = selection_context(cfg)
    context["model"] = "resnet20_gn"
    atomic_write_json(tmp_path / "pilot_provenance.json", {"context": context, "context_fingerprint": config_fingerprint(context)})
    pd.DataFrame({"lambda": [0.0], "seed": [0]}).to_csv(tmp_path / "pilot_runs.csv", index=False)
    with pytest.raises(ValueError, match="Wrong-family"):
        select_from_csv(tmp_path / "pilot_runs.csv", tmp_path / "selection", cfg=cfg)
    assert not (tmp_path / "selection").exists()


def test_false_string_is_not_a_stable_pilot_observation():
    rows = pd.DataFrame([{"lambda": 0.1, "seed": 0, "numerically_stable": "False", "D_within_bar": 1,
                          "D_between_bar": 2, "R_orbit_bar": 0.5, "orbit_ensemble_accuracy": 0.7}])
    assert not bool(summarize_pilot_runs(rows).iloc[0].all_stable)


def test_missing_data_is_matrix_membership_without_execution_readiness(tmp_path):
    plan = build_execution_plan(family(tmp_path))
    assert plan.summary["new_final_training_required"] == 36
    assert plan.summary["awaiting_data_cells"] == 36
    assert plan.summary["blocked_cells"] == 12
    assert all(row["matrix_member"] and not row["execution_ready"] for row in plan.rows)
    assert not any(row["state"] == "pending" for row in plan.rows)


def test_completed_native_requires_content_evidence(tmp_path):
    cfg, key, root, _ = native_fixture(tmp_path)
    result = verify_completed_run(root, key)
    assert result["valid"], result
    assert _status(root, key)[0] == "complete_native"
    (root / "completion.json").unlink()
    assert _status(root, key)[0] == "blocked"


@pytest.mark.parametrize("replacement", [None, "node_5", "node_1"], ids=["missing", "unpadded", "one-based"])
def test_real_confusion_archive_requires_canonical_node_keys(tmp_path, replacement):
    _, key, root, _ = native_fixture(tmp_path)
    labels = torch.arange(20) % 10
    raw = torch.randn(NUM_NODES, len(labels), 10, generator=torch.Generator().manual_seed(8))
    node_table, orbit_table, confusions = classification_tables(raw, labels)
    archive_path = root / "artifacts/confusion_matrices.npz"
    np.savez_compressed(archive_path, **confusions)
    torch.save(labels, root / "artifacts/final_labels.pt")
    metrics = pd.read_csv(root / "metrics/checkpoint_metrics.csv")
    terminal = json.loads((root / "metrics/terminal_metrics.json").read_text())
    for kind, table, field in (("node", node_table, "mean_node_accuracy"),
                               ("orbit", orbit_table, "orbit_ensemble_accuracy")):
        table.to_csv(root / f"metrics/{kind}_classification.csv", index=False)
        accuracy = float(table.loc[table.metric == "overall_accuracy", "value"].mean())
        metrics[field] = terminal[field] = terminal[f"final_{field}"] = accuracy
    metrics.to_csv(root / "metrics/checkpoint_metrics.csv", index=False)
    atomic_write_json(root / "metrics/terminal_metrics.json", terminal)
    for name in ("last", "final"):
        path = root / f"checkpoints/{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["extra"]["terminal_metrics"] = terminal
        torch.save(checkpoint, path)
    republish(root, key)

    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {f"node_{i:02d}" for i in range(NUM_NODES)} | {
            f"orbit_{k}" for k in range(1, len(REFERENCE_ORBITS) + 1)}
    assert verify_run(root)["valid"]

    removed = confusions.pop("node_05")
    if replacement is not None:
        confusions[replacement] = removed
    np.savez_compressed(archive_path, **confusions)
    # Refresh checksums so rejection must come from archive content validation.
    republish(root, key)
    with pytest.raises(ValueError, match="node_05 is not a file in the archive"):
        verify_run(root)


@pytest.mark.parametrize("damage", ["epoch", "window", "decomposition", "ratio", "pairwise_nan", "pairwise_diagonal", "terminal", "checkpoint", "mixed_models", "nonfinite_models", "scheduler", "rng", "identity", "artifact"])
def test_completed_verification_rejects_corruption_even_after_republication(tmp_path, damage):
    _, key, root, _ = native_fixture(tmp_path)
    if damage in {"epoch", "checkpoint", "mixed_models", "nonfinite_models", "scheduler", "rng"}:
        checkpoint = torch.load(root / "checkpoints/final.pt", weights_only=False)
        if damage == "epoch":
            checkpoint["epoch"] = 1
        elif damage == "checkpoint":
            checkpoint["config_fingerprint"] = "a" * 64
        elif damage == "mixed_models":
            checkpoint["models"][0]["weight"] += 1
        elif damage == "nonfinite_models":
            checkpoint["models"][0]["weight"][:] = float("nan")
        elif damage == "scheduler":
            checkpoint["scheduler"]["epoch"] = 1
        else:
            checkpoint["rng_state"]["torch_cpu"] = torch.tensor([1.0])
        torch.save(checkpoint, root / "checkpoints/final.pt")
    elif damage in {"window", "decomposition", "ratio"}:
        frame = pd.read_csv(root / "metrics/checkpoint_metrics.csv")
        if damage == "window":
            frame = frame[frame.epoch != 185]
        elif damage == "decomposition":
            frame.loc[0, "D_total"] = 10
        else:
            frame.loc[0, "R_orbit"] = 0.25
        frame.to_csv(root / "metrics/checkpoint_metrics.csv", index=False)
    elif damage.startswith("pairwise"):
        array = np.zeros((14, 14))
        array[0, 0] = np.nan if damage == "pairwise_nan" else 1
        np.save(root / "artifacts/pairwise_distance.npy", array)
    elif damage == "terminal":
        terminal = json.loads((root / "metrics/terminal_metrics.json").read_text())
        terminal.pop("D_total_bar")
        atomic_write_json(root / "metrics/terminal_metrics.json", terminal)
    elif damage == "identity":
        resolved = yaml.safe_load((root / "resolved_config.yaml").read_text())
        resolved["scientific_identity"]["training"]["learning_rate"] = 0.9
        atomic_write_yaml(root / "resolved_config.yaml", resolved)
    else:
        (root / "artifacts/confusion_matrices.npz").unlink()
    republish(root, key)
    assert not verify_completed_run(root, key)["valid"]


def test_status_and_publication_must_have_same_generation(tmp_path):
    _, key, root, _ = native_fixture(tmp_path)
    status = json.loads((root / "run_status.json").read_text())
    status["completion_generation"] = "b" * 64
    atomic_write_json(root / "run_status.json", status)
    assert not verify_completed_run(root, key)["valid"]


def test_interrupted_final_checkpoint_can_finalize_but_requires_history(tmp_path):
    _, key, root, _ = native_fixture(tmp_path)
    atomic_write_json(root / "run_status.json", {"status": "interrupted", "last_valid_epoch": 200})
    result = verify_resumable_run(root, key)
    assert result["valid"] and result["requires_finalization"], result
    (root / "metrics/checkpoint_metrics.csv").unlink()
    assert not verify_resumable_run(root, key)["valid"]


def test_current_scientific_settings_are_checked_for_complete_and_resume(tmp_path, monkeypatch):
    cfg, key, root, manifest = native_fixture(tmp_path)
    monkeypatch.setattr("orbit_cifar.experiments.completion.manifest_for_config", lambda cfg, balanced=False: manifest)
    assert verify_completed_run(root, key, cfg)["valid"]
    cfg.training.learning_rate = 0.9
    assert not verify_completed_run(root, key, cfg)["valid"]
    atomic_write_json(root / "run_status.json", {"status": "interrupted", "last_valid_epoch": 200})
    assert not verify_resumable_run(root, key, cfg)["valid"]


def test_readonly_condition_metadata_cannot_silently_change_the_design():
    cfg = load_config(CONFIG, ["condition.name=no_communication"])
    assert cfg.condition.lambda_source == "zero"
    with pytest.raises(ValueError, match="read-only"):
        load_config(CONFIG, ["condition.assignment=misaligned"])
    with pytest.raises(ValueError, match="read-only"):
        load_config(CONFIG, ["graph.spectral_match_complete=false"])


def test_no_valid_selection_cli_returns_failure_without_losing_diagnostics(tmp_path, monkeypatch):
    from orbit_cifar.cli.select_lambda import main
    monkeypatch.setattr("orbit_cifar.cli.select_lambda.select_from_csv", lambda *a, **k: {"lambda_main": None, "selection_status": "no_valid_candidate"})
    assert main(["--config", str(CONFIG), "--output-directory", str(tmp_path)]) == 2


def test_active_selection_consumes_verified_candidate_artifacts(tmp_path):
    rows = []
    for value in (0.0, 0.03, 0.1, 0.3):
        cfg, _, directory, _ = native_fixture(tmp_path, stage="pilot", epochs=80, value=value)
        terminal = json.loads((directory / "metrics/terminal_metrics.json").read_text())
        context = selection_context(cfg)
        rows.append({"lambda": value, "seed": 0, "numerically_stable": True,
                     "D_within_bar": terminal["D_within_bar"], "D_between_bar": terminal["D_between_bar"],
                     "R_orbit_bar": terminal["R_orbit_bar"], "orbit_ensemble_accuracy": terminal["orbit_ensemble_accuracy"],
                     "run_directory": str(directory), "context_fingerprint": config_fingerprint(context)})
    output = tmp_path / str(cfg.output_root) / "pilot/mnist/mlp_ln"
    atomic_write_json(output / "pilot_provenance.json", {"context": context, "context_fingerprint": config_fingerprint(context)})
    pd.DataFrame(rows).to_csv(output / "pilot_runs.csv", index=False)
    selected = select_from_csv(output / "pilot_runs.csv", output, cfg=cfg)
    assert selected["lambda_main"] == 0.03
    assert validate_selected(cfg, selected) == selected
    rows[1]["D_within_bar"] = 0.001
    pd.DataFrame(rows).to_csv(output / "pilot_runs.csv", index=False)
    with pytest.raises(ValueError, match="CSV differs"):
        select_from_csv(output / "pilot_runs.csv", output, cfg=cfg)
    rows[1]["numerically_stable"] = False
    pd.DataFrame(rows).to_csv(output / "pilot_runs.csv", index=False)
    with pytest.raises(ValueError, match="divergence evidence"):
        select_from_csv(output / "pilot_runs.csv", output, cfg=cfg)
