"""Offline adversarial tests for exact frozen legacy provenance and read-only reuse."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

import orbit_cifar.legacy as legacy
from orbit_cifar.config_schema import load_config
from orbit_cifar.data.assignments import assignment_summary
from orbit_cifar.data.partition import build_fully_symmetric_manifest, build_split_manifest
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.matrix import ScientificKey, run_directory
from orbit_cifar.experiments.provenance import validate_selected
from orbit_cifar.training.seed_plan import build_seed_plan
from orbit_cifar.utils.io import atomic_write_json, config_fingerprint


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str = "main"):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml")
    cfg.project_root = str(tmp_path / "project")
    cfg.legacy_reuse.source_root = str(tmp_path / "source/outputs/final")
    key = ScientificKey("cifar10", "resnet20_gn", condition, 1)
    run = legacy.legacy_source_directory(cfg.legacy_reuse.source_root, key)
    labels = np.repeat(np.arange(10), 5000)
    source = SimpleNamespace(targets=labels, data=np.arange(50000, dtype=np.uint8)[:, None])
    monkeypatch.setattr(legacy, "load_cifar10", lambda *args, **kwargs: source)
    manifest = build_split_manifest(labels)
    atomic_write_json(tmp_path / "source/data/split_manifest.json", manifest)
    atomic_write_json(Path(cfg.project_root) / cfg.manifest_root / "cifar10/heterogeneous.json", manifest)
    concrete = (build_fully_symmetric_manifest(labels, manifest["validation_indices"])
                if condition == "fully_symmetric" else manifest)
    selected = {"lambda_main": 0.1}
    condition_spec, graph = resolve_condition(condition, selected)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    resolved.update({"seed": 1, "stage": "final", "manifest_path": "data/split_manifest.json",
                     "resolved_condition": condition_spec.to_dict()})
    if condition == "fully_symmetric":
        resolved["fully_symmetric_design"] = {"reference_partition_semantics": "reference_partition_only"}
    assignment = assignment_summary(condition_spec.assignment, concrete)
    assignment.update({"manifest_fingerprint": config_fingerprint(concrete),
                       "validation_indices_checksum": config_fingerprint(concrete["validation_indices"]),
                       "data_fingerprint": legacy.dataset_fingerprint(source)})
    if condition == "fully_symmetric":
        assignment["balanced_manifest"] = concrete
    pairwise = np.full((14, 14), 84.0 / 13)
    np.fill_diagonal(pairwise, 0)
    metrics = pd.DataFrame([{"epoch": epoch, "D_within": 1., "D_between": 2., "D_total": 3.,
                             "R_orbit": 2 / 3, "mean_node_accuracy": .5, "orbit_ensemble_accuracy": .6}
                            for epoch in (180, 185, 190, 195, 200)])
    terminal = {"condition": condition, "seed": 1, "communication_lambda": condition_spec.communication_lambda,
                "probe_split": "official_test", "numerically_stable": True, "terminal_window_short": False,
                "D_within_bar": 1., "D_between_bar": 2., "D_total_bar": 3., "R_orbit_bar": 2 / 3,
                "mean_node_accuracy": .5, "orbit_ensemble_accuracy": .6,
                "final_mean_node_accuracy": .5, "final_orbit_ensemble_accuracy": .6}
    if condition == "fully_symmetric":
        terminal.update(D_global_bar=3., final_D_global=3., partition_semantics="reference_partition_only")
    payloads = {"resolved_config.yaml": resolved, "environment.json": {"evidence": "synthetic fixture"},
                "seed_plan.json": build_seed_plan(1), "data_assignment.json": assignment,
                "run_status.json": {"status": "completed", "numerically_stable": True, "last_valid_epoch": 200},
                "graph/graph_summary.json": {"name": graph.name}, "metrics/terminal_metrics.json": terminal}
    matrices = {"graph/adjacency.npy": graph.adjacency, "graph/communication_weights.npy": graph.communication_weights,
                "graph/laplacian.npy": graph.laplacian, "artifacts/pairwise_distance.npy": pairwise}
    for relative in legacy.REPORT_FILES:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in payloads:
            path.write_text(yaml.safe_dump(payloads[relative]) if path.suffix == ".yaml" else json.dumps(payloads[relative]), encoding="utf-8")
        elif relative in matrices:
            np.save(path, matrices[relative])
        elif path.name == "checkpoint_metrics.csv":
            metrics.to_csv(path, index=False)
        elif path.name == "pairwise_distance.csv":
            pd.DataFrame(pairwise).to_csv(path, index_label="node_label")
        else:
            node = path.name == "node_classification.csv"
            entity = "node" if node else "orbit"
            rows = []
            for identity in (range(14) if node else range(1, 5)):
                rows.append({entity: identity, "class_id": -1, "metric": "overall_accuracy", "value": .5 if node else .6})
                for class_id in range(10):
                    rows.extend([{entity: identity, "class_id": class_id, "metric": "class_accuracy", "value": .5},
                                 {entity: identity, "class_id": class_id, "metric": "class_cross_entropy", "value": 1.}])
            pd.DataFrame(rows).to_csv(path, index=False)
    return cfg, key, run, manifest, source


def _mutate_json(path: Path, mutation):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    atomic_write_json(path, value)


@pytest.mark.parametrize("condition", ["main", "no_communication", "misaligned_data", "fully_symmetric"])
def test_exact_verified_import_is_allowlisted_idempotent_and_read_only(tmp_path, monkeypatch, condition):
    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch, condition)
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    report = legacy.compatibility_report(run, key, cfg)
    assert report["decision"] == "compatible", report
    assert legacy.import_legacy_run(cfg, key)["decision"] == "imported"
    assert legacy.import_legacy_run(cfg, key)["decision"] == "already_imported_idempotent"
    target = run_directory(Path(cfg.project_root) / cfg.output_root, key)
    assert legacy.validate_imported_run(target, key, cfg)["valid"]
    assert not (target / "checkpoints").exists()
    assert before == {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="outside"):
        legacy.import_legacy_run(cfg, ScientificKey("mnist", "resnet20_gn", "main", 1))


@pytest.mark.parametrize("mutation", ["reorder", "other_seed", "fingerprint", "pixels", "labels", "source_split"])
def test_legacy_concrete_data_identity_cannot_be_replaced_by_counts_or_hash_text(tmp_path, monkeypatch, mutation):
    cfg, key, run, manifest, source = _fixture(tmp_path, monkeypatch)
    source_manifest = tmp_path / "source/data/split_manifest.json"
    if mutation == "reorder":
        indices = manifest["bundles"][0]["indices"]
        indices[0], indices[1] = indices[1], indices[0]
        atomic_write_json(source_manifest, manifest)
        _mutate_json(run / "data_assignment.json", lambda value: value.update(manifest_fingerprint=config_fingerprint(manifest)))
    elif mutation == "other_seed":
        atomic_write_json(source_manifest, build_split_manifest(source.targets, split_seed=999))
    elif mutation == "fingerprint":
        _mutate_json(run / "data_assignment.json", lambda value: value.update(manifest_fingerprint="f" * 64))
    elif mutation == "pixels":
        source.data[0] += 1
    elif mutation == "labels":
        source.targets[0] = 9
    else:
        manifest["source_split"] = "test"
        atomic_write_json(source_manifest, manifest)
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_incompatible"


@pytest.mark.parametrize("mutation", ["duplicates", "validation_leak", "false_labels", "different_validation"])
def test_fully_symmetric_legacy_checks_actual_bundle_indices_and_probe(tmp_path, monkeypatch, mutation):
    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch, "fully_symmetric")
    def corrupt(value):
        balanced = value["balanced_manifest"]
        if mutation == "duplicates":
            balanced["bundles"][0]["indices"] = [0] * 2000
        elif mutation == "validation_leak":
            balanced["bundles"][0]["indices"][0] = balanced["validation_indices"][0]
        elif mutation == "false_labels":
            first, second = balanced["bundles"][0]["indices"], balanced["bundles"][1]["indices"]
            first[0], second[-1] = second[-1], first[0]
        else:
            balanced["validation_indices"][0], balanced["unused_indices"][0] = balanced["unused_indices"][0], balanced["validation_indices"][0]
        value["manifest_fingerprint"] = config_fingerprint(balanced)
    _mutate_json(run / "data_assignment.json", corrupt)
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_incompatible"


@pytest.mark.parametrize("field,value", [("data.train_augmentation", False), ("data.normalization_mean", [0., 0., 0.]),
                                       ("model.group_norm_max_groups", 4), ("model.initialization_rho", .03),
                                       ("training.gradient_clipping", 1.), ("training.warmup_epochs", 1)])
def test_legacy_behavior_changing_protocol_fields_block_reuse(tmp_path, monkeypatch, field, value):
    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch)
    resolved = OmegaConf.load(run / "resolved_config.yaml")
    OmegaConf.update(resolved, field, value)
    OmegaConf.save(resolved, run / "resolved_config.yaml")
    report = legacy.import_legacy_run(cfg, key)
    assert report["decision"] == "blocked_incompatible"
    assert field in report["failures"]


def test_rng_plan_mutation_and_terminal_shortening_block_reuse(tmp_path, monkeypatch):
    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch)
    _mutate_json(run / "seed_plan.json", lambda value: value.update(base_initialization=999))
    assert "seed_plan.base_initialization" in legacy.import_legacy_run(cfg, key)["failures"]
    frame = pd.read_csv(run / "metrics/checkpoint_metrics.csv")
    frame.iloc[:-1].to_csv(run / "metrics/checkpoint_metrics.csv", index=False)
    assert "terminal_metrics_and_pairwise" in legacy.import_legacy_run(cfg, key)["failures"]


def test_missing_evidence_requires_cell_and_field_reason_and_never_becomes_verified(tmp_path, monkeypatch):
    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch)
    _mutate_json(run / "data_assignment.json", lambda value: value.pop("data_fingerprint"))
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_unverified"
    with pytest.raises(TypeError):
        legacy.import_legacy_run(cfg, key, approve_unverified=True)
    with pytest.raises(ValueError, match="currently unverified"):
        legacy.import_legacy_run(cfg, key, approved_fields={"anything": "reason"})
    report = legacy.import_legacy_run(cfg, key, approved_fields={"data_fingerprint": "Archived pixel binding unavailable"})
    assert report["decision"] == "imported_unverified"
    target = run_directory(Path(cfg.project_root) / cfg.output_root, key)
    assert not legacy.validate_imported_run(target, key, cfg)["valid"]


def test_tampered_import_is_rejected_by_idempotence_and_shared_validator(tmp_path, monkeypatch):
    from orbit_cifar.cli.verify_run import verify_run
    from orbit_cifar.experiments.planner import _status

    cfg, key, run, _, _ = _fixture(tmp_path, monkeypatch)
    legacy.import_legacy_run(cfg, key)
    target = run_directory(Path(cfg.project_root) / cfg.output_root, key)
    frame = pd.read_csv(target / "metrics/checkpoint_metrics.csv")
    frame["mean_node_accuracy"] = .4
    frame.to_csv(target / "metrics/checkpoint_metrics.csv", index=False)
    assert not legacy.validate_imported_run(target, key, cfg)["valid"]
    assert _status(target, key, cfg)[0] == "blocked"
    with pytest.raises(ValueError, match="verification failed"):
        verify_run(target)
    with pytest.raises(FileExistsError):
        legacy.import_legacy_run(cfg, key)
    with pytest.raises(FileExistsError):
        legacy.import_legacy_run(cfg, key, dry_run=True)


def test_malformed_concrete_manifest_is_reported_as_blocked(tmp_path, monkeypatch):
    cfg, key, _, _, _ = _fixture(tmp_path, monkeypatch)
    atomic_write_json(tmp_path / "source/data/split_manifest.json", {"bundles": []})
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_incompatible"


def test_verified_copy_survives_source_unavailability_but_rejects_changed_active_assignment(tmp_path, monkeypatch):
    cfg, key, _, manifest, _ = _fixture(tmp_path, monkeypatch)
    legacy.import_legacy_run(cfg, key)
    target = run_directory(Path(cfg.project_root) / cfg.output_root, key)
    cfg.legacy_reuse.source_root = str(tmp_path / "unavailable")
    assert legacy.validate_imported_run(target, key, cfg)["valid"]
    assert legacy.import_legacy_run(cfg, key)["decision"] == "already_imported_idempotent"
    indices = manifest["bundles"][0]["indices"]
    indices[0], indices[1] = indices[1], indices[0]
    atomic_write_json(Path(cfg.project_root) / cfg.manifest_root / "cifar10/heterogeneous.json", manifest)
    assert "active assignment changed" in legacy.validate_imported_run(target, key, cfg)["errors"]


def test_empty_digest_inventory_is_never_imported_and_registry_retry_recovers(tmp_path, monkeypatch):
    cfg, key, _, _, _ = _fixture(tmp_path, monkeypatch)
    real_write = legacy.atomic_write_json
    def interrupted(path, value):
        if "import_registry" in str(path):
            raise OSError("registry publication interrupted")
        return real_write(path, value)
    monkeypatch.setattr(legacy, "atomic_write_json", interrupted)
    with pytest.raises(OSError, match="registry"):
        legacy.import_legacy_run(cfg, key)
    target = run_directory(Path(cfg.project_root) / cfg.output_root, key)
    assert legacy.validate_imported_run(target, key, cfg)["valid"]
    monkeypatch.setattr(legacy, "atomic_write_json", real_write)
    assert legacy.import_legacy_run(cfg, key)["decision"] == "already_imported_idempotent"
    _mutate_json(target / "legacy_reference.json", lambda value: value.update(source_fingerprints={}))
    assert not legacy.validate_imported_run(target, key, cfg)["valid"]


def test_frozen_selection_is_bound_and_never_claims_candidate_imports(tmp_path, monkeypatch):
    cfg, _, _, _, _ = _fixture(tmp_path, monkeypatch)
    selected = legacy.register_legacy_selection(cfg, "Explicit archived report approval")
    assert selected["source_type"] == "legacy_selected_value"
    assert selected["candidate_runs_imported"] == 0
    assert selected["candidate_runs_not_rerun"] == 4
    validate_selected(cfg, selected)
    wrong = copy.deepcopy(selected)
    wrong["provenance"]["model"] = "mlp_ln"
    wrong["provenance_fingerprint"] = config_fingerprint(wrong["provenance"])
    with pytest.raises(ValueError, match="Wrong-family"):
        validate_selected(cfg, wrong)
    condition, _ = resolve_condition("fully_symmetric", selected)
    assert np.isclose(condition.communication_lambda, .1410835232, atol=1e-8)


def test_missing_legacy_source_is_blocked_without_fallback(tmp_path, monkeypatch):
    cfg, key, _, _, _ = _fixture(tmp_path, monkeypatch)
    cfg.legacy_reuse.source_root = str(tmp_path / "missing")
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_missing"
    cfg.legacy_reuse.source_root = None
    assert legacy.import_legacy_run(cfg, key)["decision"] == "blocked_missing"
