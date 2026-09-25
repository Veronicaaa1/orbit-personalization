"""Strict read-only import of the twelve allowlisted legacy observations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.constants import ACTIVE_CONDITIONS, ACTIVE_SEEDS
from orbit_cifar.data.assignments import assignment_summary
from orbit_cifar.data.manifest import validate_manifest
from orbit_cifar.data.partition import build_split_manifest, build_fully_symmetric_manifest
from orbit_cifar.data.datasets import load_cifar10
from orbit_cifar.experiments.matrix import ScientificKey, legacy_allowlisted, run_directory
from orbit_cifar.experiments.families import family_config
from orbit_cifar.graph.builders import build_blown_up_path, build_complete_graph
from orbit_cifar.training.seed_plan import build_seed_plan
from orbit_cifar.utils.io import atomic_write_json, atomic_write_yaml, config_fingerprint

COMPATIBILITY_VERSION = "legacy-cifar10-resnet20-gn-v2"
REPORT_FILES = (
    "resolved_config.yaml", "environment.json", "seed_plan.json", "data_assignment.json", "run_status.json",
    "graph/graph_summary.json", "graph/adjacency.npy", "graph/communication_weights.npy", "graph/laplacian.npy",
    "metrics/checkpoint_metrics.csv", "metrics/terminal_metrics.json", "metrics/node_classification.csv",
    "metrics/orbit_classification.csv", "artifacts/pairwise_distance.npy", "artifacts/pairwise_distance.csv",
)
OPTIONAL_FILES = ("graph/edge_list.csv", "artifacts/confusion_matrices.npz", "artifacts/random_partition_diagnostic.npz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def legacy_source_directory(source_root: str | Path, key: ScientificKey) -> Path:
    """Map only the allowlisted legacy condition/seed layout."""
    return Path(source_root).expanduser() / key.condition / f"seed_{key.seed:03d}"


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an evidence mapping: {path}")
    return value


def dataset_fingerprint(dataset: Any) -> str:
    """Hash concrete official-source pixels and labels, including shape/dtype."""
    pixels = np.ascontiguousarray(dataset.data)
    labels = np.asarray(dataset.targets, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(config_fingerprint({"dataset": "cifar10", "split": "official_train",
                                      "shape": list(pixels.shape), "dtype": str(pixels.dtype)}).encode())
    digest.update(pixels.tobytes())
    digest.update(labels.tobytes())
    return digest.hexdigest()


def _manifest_semantics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare ordered scientific allocations independent of manifest schema."""
    return {"source_split": manifest.get("source_split", "train"), "dataset": manifest.get("dataset", "cifar10")} | {key: manifest.get(key) for key in (
        "dataset_version", "split_seed", "num_source_examples",
        "validation_indices", "validation_class_counts", "unused_indices"
    )} | {"bundles": [
        {key: bundle.get(key) for key in ("bundle_id", "profile_name", "class_counts", "indices")}
        for bundle in manifest.get("bundles", [])
    ]}


def _source_manifest(run: Path, resolved: dict[str, Any]) -> Path | None:
    declared = resolved.get("manifest_path") or _nested(resolved, "data.manifest_path")
    # A recorded absolute source path is authoritative if available. Relative
    # archived paths are resolved within the source ancestors, never project cwd.
    if declared and Path(str(declared)).is_absolute() and Path(str(declared)).is_file():
        return Path(str(declared))
    relative = Path(str(declared)) if declared and not Path(str(declared)).is_absolute() else Path("data/split_manifest.json")
    for parent in (run, *run.parents):
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    return None


def _terminal_evidence(run: Path, key: ScientificKey) -> bool:
    terminal = _read_mapping(run / "metrics/terminal_metrics.json")
    frame = pd.read_csv(run / "metrics/checkpoint_metrics.csv")
    epochs = pd.to_numeric(frame["epoch"], errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(epochs).all() or not np.equal(epochs, epochs.astype(int)).all()
            or len(set(epochs)) != len(epochs) or np.any(np.diff(epochs) <= 0)
            or epochs[-5:].tolist() != [180, 185, 190, 195, 200]):
        return False
    dispersions = frame[["D_within", "D_between", "D_total"]].to_numpy(dtype=float)
    if (not np.isfinite(dispersions).all() or np.any(dispersions < 0)
            or not np.allclose(dispersions[:, 2], dispersions[:, 0] + dispersions[:, 1], atol=1e-9, rtol=1e-9)):
        return False
    for column in ("D_within", "D_between", "D_total", "R_orbit"):
        expected = float(frame.tail(5)[column].mean())
        observed = float(terminal[column + "_bar"])
        if not (np.isclose(observed, expected, atol=1e-9, rtol=1e-9)
                or (np.isnan(observed) and np.isnan(expected))):
            return False
    if key.condition == "fully_symmetric" and (
        not np.isclose(float(terminal["D_global_bar"]), float(terminal["D_total_bar"]), atol=1e-9, rtol=1e-9)
        or not np.isclose(float(terminal["final_D_global"]), dispersions[-1, 2], atol=1e-9, rtol=1e-9)
        or terminal.get("partition_semantics") != "reference_partition_only"
    ):
        return False
    ratio = frame["R_orbit"].to_numpy(dtype=float)
    totals = dispersions[:, 0] + dispersions[:, 1]
    defined = totals >= 1e-14
    if not np.allclose(ratio[defined], dispersions[defined, 1] / totals[defined], atol=1e-9, rtol=1e-9):
        return False
    if np.any(~np.isnan(ratio[~defined])):
        return False
    for field in ("mean_node_accuracy", "orbit_ensemble_accuracy"):
        values = frame[field].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            return False
        if not np.isclose(float(terminal[field]), values[-5:].mean(), atol=1e-9, rtol=1e-9):
            return False
        if not np.isclose(float(terminal["final_" + field]), values[-1], atol=1e-9, rtol=1e-9):
            return False
    for filename, entity, ids, accuracy in (
        ("node_classification.csv", "node", range(14), terminal["final_mean_node_accuracy"]),
        ("orbit_classification.csv", "orbit", range(1, 5), terminal["final_orbit_ensemble_accuracy"]),
    ):
        table = pd.read_csv(run / "metrics" / filename)
        expected_rows = {(identity, -1, "overall_accuracy") for identity in ids} | {
            (identity, class_id, metric) for identity in ids for class_id in range(10)
            for metric in ("class_accuracy", "class_cross_entropy")
        }
        actual_rows = list(table[[entity, "class_id", "metric"]].itertuples(index=False, name=None))
        values = table["value"].to_numpy(dtype=float)
        if (set(actual_rows) != expected_rows or len(actual_rows) != len(expected_rows)
                or not np.isfinite(values).all() or np.any(values < 0)
                or np.any(table.loc[table.metric.str.endswith("accuracy"), "value"].to_numpy(dtype=float) > 1)):
            return False
        if not np.isclose(table.loc[table.metric == "overall_accuracy", "value"].mean(), float(accuracy), atol=1e-9, rtol=1e-9):
            return False
    pairwise = np.load(run / "artifacts/pairwise_distance.npy", allow_pickle=False)
    csv_pairwise = pd.read_csv(run / "artifacts/pairwise_distance.csv", index_col=0).to_numpy(dtype=float)
    if (pairwise.shape != (14, 14) or not np.isfinite(pairwise).all() or np.any(pairwise < -1e-10)
            or not np.allclose(pairwise, pairwise.T, atol=1e-10, rtol=0)
            or not np.allclose(np.diag(pairwise), 0, atol=1e-10, rtol=0)
            or csv_pairwise.shape != pairwise.shape or not np.allclose(csv_pairwise, pairwise, atol=1e-10, rtol=0)
            or not np.isclose(pairwise.mean() / 2, dispersions[-1, 2], atol=1e-9, rtol=1e-9)):
        return False
    return (terminal.get("condition") == key.condition and terminal.get("seed") == key.seed
            and terminal.get("numerically_stable") is True and terminal.get("terminal_window_short") is False)


def compatibility_report(source: str | Path, key: ScientificKey, cfg: DictConfig | None = None) -> dict[str, Any]:
    """Verify concrete source evidence against the frozen family and exact split."""
    run = Path(source).expanduser().resolve()
    checks: list[dict[str, Any]] = []
    evidence_fingerprints: dict[str, str] = {}

    def check(field: str, observed: Any, expected: Any, *, present: bool | None = None,
              approximate: float | None = None) -> None:
        available = observed is not None if present is None else present
        try:
            equal = (bool(np.isclose(float(observed), float(expected), atol=approximate, rtol=0))
                     if approximate is not None and available else observed == expected)
        except (ValueError, TypeError):
            equal = False
        checks.append({"field": field, "outcome": "unverified" if not available else ("pass" if equal else "fail"),
                       "observed": observed, "expected": expected, "critical": True})

    def read(relative: str) -> dict[str, Any]:
        path = run / relative
        if not path.is_file():
            return {}
        try:
            return _read_mapping(path)
        except (OSError, ValueError, yaml.YAMLError):
            check("parse:" + relative, False, True)
            return {}

    resolved = read("resolved_config.yaml")
    check("scientific_key", key.dataset == "cifar10" and key.model == "resnet20_gn"
          and key.condition in ACTIVE_CONDITIONS and key.seed in ACTIVE_SEEDS, True)
    terminal = read("metrics/terminal_metrics.json")
    status = read("run_status.json")
    assignment = read("data_assignment.json")
    graph = read("graph/graph_summary.json")
    expected_values = {
        "data.name": "cifar10", "data.dataset_version": "torchvision-cifar10",
        "data.split_seed": 2027, "data.validation_per_class": 500,
        "data.train_augmentation": True,
        "data.normalization_mean": [0.4914, 0.4822, 0.4465],
        "data.normalization_std": [0.2470, 0.2435, 0.2616],
        "model.name": "resnet20_gn", "model.input_channels": 3,
        "model.input_height": 32, "model.input_width": 32, "model.num_classes": 10,
        "model.group_norm_max_groups": 8, "model.initialization_mode": "relative_rms",
        "model.initialization_rho": 0.02, "model.absolute_scale_floor": 0.001,
        "deterministic": True, "training.epochs": 200, "training.learning_rate": 0.05,
        "training.batch_size_per_node": 128, "training.momentum": 0.0, "training.weight_decay": 0.0,
        "training.warmup_epochs": 5, "training.minimum_learning_rate": 0.0005,
        "training.communication_every_steps": 1, "training.eval_every_epochs": 5,
        "training.checkpoint_every_epochs": 5, "training.terminal_window_checkpoints": 5,
        "training.label_smoothing": 0.0, "training.gradient_clipping": None, "training.mixed_precision": False,
        "training.ratio_threshold": 1e-14, "training.decomposition_tolerance": 1e-9,
    }
    family = family_config(cfg, "cifar10", "resnet20_gn") if cfg is not None else None
    for field, expected in expected_values.items():
        if family is not None:
            active = OmegaConf.select(family, field, default=expected)
            if OmegaConf.is_config(active):
                active = OmegaConf.to_container(active, resolve=True)
            check("active_protocol:" + field, active, expected, present=True)
        parent, _, leaf = field.rpartition(".")
        container = _nested(resolved, parent) if parent else resolved
        check(field, _nested(resolved, field), expected,
              present=isinstance(container, dict) and leaf in container)
    check("seed", resolved.get("seed"), key.seed)
    check("condition", _nested(resolved, "resolved_condition.name"), key.condition)
    check("completed_epochs", status.get("last_valid_epoch"), 200)
    check("status", status.get("status"), "completed")
    check("numerically_stable", status.get("numerically_stable"), True)
    check("probe_split", terminal.get("probe_split"), "official_test")
    for field, expected in (("dataset", key.dataset), ("model", key.model)):
        if field in terminal:
            check("terminal." + field, terminal[field], expected)
    seed_plan = read("seed_plan.json")
    expected_seed_plan = build_seed_plan(key.seed)
    # Historical plans predate the namespace annotation; the actual RNG streams
    # are still required in full and must agree exactly.
    for field, expected in expected_seed_plan.items():
        if field not in {"namespace", "data_run_seed"}:
            check("seed_plan." + field, seed_plan.get(field), expected)
    if "namespace" in seed_plan:
        check("seed_plan.namespace", seed_plan["namespace"] in {"legacy", "final"}, True)
    if "data_run_seed" in seed_plan:
        check("seed_plan.data_run_seed", seed_plan["data_run_seed"], key.seed)
    expected_graph = build_complete_graph() if key.condition == "fully_symmetric" else build_blown_up_path()
    check("graph", graph.get("name"), expected_graph.name)
    try:
        graph_valid = all(
            (observed := np.load(run / "graph" / filename, allow_pickle=False)).shape == expected.shape
            and np.isfinite(observed).all() and np.allclose(observed, expected, atol=1e-12, rtol=0)
            for filename, expected in (("adjacency.npy", expected_graph.adjacency),
                                       ("communication_weights.npy", expected_graph.communication_weights),
                                       ("laplacian.npy", expected_graph.laplacian))
        )
        check("condition_graph_and_weights", bool(graph_valid), True)
    except (OSError, ValueError):
        check("condition_graph_and_weights", False, True)
    expected_lambda = (0.0 if key.condition == "no_communication" else
                       0.1 * build_blown_up_path().spectral_norm / expected_graph.spectral_norm)
    lambda_tolerance = 1e-8 if key.condition == "fully_symmetric" else 1e-12
    check("actual_lambda", terminal.get("communication_lambda"), expected_lambda, approximate=lambda_tolerance)
    check("resolved_lambda", _nested(resolved, "resolved_condition.communication_lambda"), expected_lambda, approximate=lambda_tolerance)

    source_manifest = _source_manifest(run, resolved)
    manifest = None
    labels = None
    try:
        if source_manifest is not None:
            manifest = _read_mapping(source_manifest)
            validate_manifest(manifest)
            evidence_fingerprints["source_manifest"] = _sha256(source_manifest)
            evidence_fingerprints["heterogeneous_assignment"] = config_fingerprint(_manifest_semantics(manifest))
            dataset = load_cifar10(source_manifest.parent, train=True, download=False)
            labels = np.asarray(dataset.targets, dtype=np.int64)
            validate_manifest(manifest, targets=labels)
            expected_manifest = build_split_manifest(labels, split_seed=2027)
            check("exact_node_indices_and_validation_exclusion", _manifest_semantics(manifest) == _manifest_semantics(expected_manifest), True)
            actual_data_fingerprint = dataset_fingerprint(dataset)
            evidence_fingerprints["official_train_data"] = actual_data_fingerprint
            check("data_fingerprint", assignment.get("data_fingerprint"), actual_data_fingerprint)
            check("actual_source_label_validation", True, True)
            if family is not None:
                active_path = Path(str(family.project_root)) / str(family.manifest_root) / "cifar10" / "heterogeneous.json"
                if active_path.is_file():
                    active_manifest = _read_mapping(active_path)
                    validate_manifest(active_manifest, targets=labels)
                    check("active_manifest_exact_indices", _manifest_semantics(active_manifest) == _manifest_semantics(expected_manifest), True)
    except (OSError, RuntimeError):
        check("actual_source_label_validation", None, True)
    except (ValueError, KeyError, IndexError, TypeError):
        check("exact_node_indices_and_validation_exclusion", False, True)
        labels = None
    if manifest is None:
        check("exact_node_indices_and_validation_exclusion", None, True)
        check("actual_source_label_validation", None, True)

    assignment_name = "fully_symmetric" if key.condition == "fully_symmetric" else ("misaligned" if key.condition == "misaligned_data" else "aligned")
    actual_manifest = assignment.get("balanced_manifest") if key.condition == "fully_symmetric" else manifest
    if key.condition == "fully_symmetric":
        try:
            if actual_manifest is None:
                check("fully_symmetric_balanced_assignment", None, True)
            else:
                validate_manifest(actual_manifest, targets=labels)
                check("fully_symmetric_balanced_assignment", True, True)
                if manifest is not None:
                    check("fully_symmetric_shared_validation", actual_manifest["validation_indices"] == manifest["validation_indices"], True)
                if labels is not None and manifest is not None:
                    expected_balanced = build_fully_symmetric_manifest(labels, manifest["validation_indices"], split_seed=2027)
                    check("fully_symmetric_exact_indices", _manifest_semantics(actual_manifest) == _manifest_semantics(expected_balanced), True)
        except (ValueError, KeyError, IndexError, TypeError):
            check("fully_symmetric_balanced_assignment", False, True)
        check("initialization_centering", _nested(resolved, "fully_symmetric_design.reference_partition_semantics"), "reference_partition_only")
    if isinstance(actual_manifest, dict):
        check("manifest_fingerprint", assignment.get("manifest_fingerprint"), config_fingerprint(actual_manifest))
        check("validation_indices_checksum", assignment.get("validation_indices_checksum"), config_fingerprint(actual_manifest.get("validation_indices")))
        try:
            expected_assignment = assignment_summary(assignment_name, actual_manifest)
            for field, expected in expected_assignment.items():
                check("assignment." + field, assignment.get(field), expected)
            if "node_indices" in assignment:
                expected_indices = [actual_manifest["bundles"][bundle]["indices"] for bundle in expected_assignment["node_to_bundle"]]
                check("assignment.node_indices", assignment["node_indices"] == expected_indices, True)
            if "validation_indices" in assignment:
                check("assignment.validation_indices", assignment["validation_indices"] == actual_manifest["validation_indices"], True)
        except (KeyError, ValueError, IndexError, TypeError):
            check("assignment_semantics", False, True)
    else:
        check("manifest_fingerprint", None, "computed concrete manifest fingerprint")
    try:
        check("terminal_metrics_and_pairwise", _terminal_evidence(run, key), True)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        check("terminal_metrics_and_pairwise", False, True)
    missing = [relative for relative in REPORT_FILES if not (run / relative).is_file()]
    check("report_required_artifacts", missing, [])
    failures = sorted({row["field"] for row in checks if row["outcome"] == "fail"})
    unverified = sorted({row["field"] for row in checks if row["outcome"] == "unverified"})
    return {
        "compatibility_version": COMPATIBILITY_VERSION, "scientific_key": key.to_dict(),
        "source": str(run), "checks": checks, "failures": failures, "unverified": unverified,
        "evidence_fingerprints": evidence_fingerprints,
        "decision": "compatible" if not failures and not unverified else ("incompatible" if failures else "unverified"),
    }


def validate_imported_run(directory: str | Path, key: ScientificKey, cfg: DictConfig | None = None) -> dict[str, Any]:
    """One immutable copied-artifact contract for importer, planner and analysis.

    Original external storage need not remain mounted after a verified copy.
    Exceptions remain unverified and are excluded from scientific aggregation.
    """
    run = Path(directory)
    errors: list[str] = []
    if key.dataset != "cifar10" or key.model != "resnet20_gn" or key.condition not in ACTIVE_CONDITIONS or key.seed not in ACTIVE_SEEDS:
        errors.append("scientific key is outside frozen legacy matrix")
    try:
        reference = _read_mapping(run / "legacy_reference.json")
        if reference.get("scientific_key") != key.to_dict():
            errors.append("scientific key mismatch")
        if reference.get("compatibility_version") != COMPATIBILITY_VERSION:
            errors.append("obsolete compatibility contract")
        if reference.get("import_status") != "validated" or reference.get("source_type") != "legacy_reused":
            errors.append("legacy evidence is not verified")
        if not reference.get("original_run_path") or not reference.get("evidence_fingerprints", {}).get("official_train_data"):
            errors.append("missing original source identity")
        inventory = reference.get("source_fingerprints", {})
        if (not isinstance(inventory, dict) or not set(REPORT_FILES).issubset(inventory)
                or not set(inventory).issubset({*REPORT_FILES, *OPTIONAL_FILES})):
            errors.append("incomplete required-artifact digest inventory")
        else:
            for relative, expected in inventory.items():
                if not isinstance(expected, str) or _sha256(run / relative) != expected:
                    errors.append("artifact digest mismatch: " + relative)
        checks = reference.get("field_level_result", [])
        fields = {row.get("field") for row in checks}
        required = {"scientific_key", "data.name", "model.name", "seed", "condition", "completed_epochs", "status",
                    "exact_node_indices_and_validation_exclusion", "actual_source_label_validation",
                    "manifest_fingerprint", "data_fingerprint", "assignment.node_to_bundle",
                    "seed_plan.base_initialization", "condition_graph_and_weights", "actual_lambda",
                    "terminal_metrics_and_pairwise", "report_required_artifacts"}
        if (not required.issubset(fields) or any(row.get("outcome") != "pass" for row in checks)):
            errors.append("missing or unverified field-level evidence")
        if key.condition == "fully_symmetric" and not {
            "fully_symmetric_balanced_assignment", "fully_symmetric_shared_validation", "fully_symmetric_exact_indices"
        }.issubset(fields):
            errors.append("missing balanced assignment evidence")
        if cfg is not None:
            if reference.get("new_experiment_id") != str(cfg.experiment.name):
                errors.append("wrong active experiment")
            family = family_config(cfg, key.dataset, key.model)
            current_manifest = Path(str(family.project_root)) / str(family.manifest_root) / key.dataset / "heterogeneous.json"
            if current_manifest.is_file():
                value = _read_mapping(current_manifest)
                validate_manifest(value)
                if config_fingerprint(_manifest_semantics(value)) != reference.get("evidence_fingerprints", {}).get("heterogeneous_assignment"):
                    errors.append("active assignment changed")
            for row in checks:
                if str(row.get("field", "")).startswith("active_protocol:"):
                    field = row["field"].split(":", 1)[1]
                    current = OmegaConf.select(family, field)
                    if OmegaConf.is_config(current):
                        current = OmegaConf.to_container(current, resolve=True)
                    if current != row["expected"]:
                        errors.append("active protocol changed: " + field)
        status = _read_mapping(run / "run_status.json")
        if status.get("status") != "completed" or status.get("last_valid_epoch") != 200 or status.get("numerically_stable") is not True:
            errors.append("legacy completion status is invalid")
        if not _terminal_evidence(run, key):
            errors.append("legacy terminal artifacts are invalid")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, yaml.YAMLError) as error:
        errors.append(f"invalid imported artifacts: {type(error).__name__}: {error}")
    return {"valid": not errors, "errors": errors, "scientific_key": key.to_dict()}


def import_legacy_run(
    cfg: DictConfig, key: ScientificKey, dry_run: bool = False,
    approved_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not legacy_allowlisted(cfg, key):
        raise ValueError(f"Scientific key is outside the strict legacy allowlist: {key}")
    project = Path(str(cfg.project_root)).expanduser().resolve()
    root = project / str(cfg.output_root)
    target = run_directory(root, key)
    missing_source = not cfg.legacy_reuse.source_root or not legacy_source_directory(cfg.legacy_reuse.source_root, key).is_dir()
    if missing_source and (target / "legacy_reference.json").is_file() and validate_imported_run(target, key, cfg)["valid"]:
        existing = _read_mapping(target / "legacy_reference.json")
        registry = root / "import_registry" / key.dataset / key.model / key.condition / f"seed_{key.seed:03d}.json"
        if not dry_run and not registry.is_file():
            atomic_write_json(registry, existing)
        return {"scientific_key": key.to_dict(), "decision": "already_imported_idempotent", "source": existing["original_run_path"]}
    if not cfg.legacy_reuse.source_root:
        return {"scientific_key": key.to_dict(), "decision": "blocked_missing", "source": None}
    source = legacy_source_directory(cfg.legacy_reuse.source_root, key)
    if not source.is_dir():
        return {"scientific_key": key.to_dict(), "decision": "blocked_missing", "source": str(source)}
    before_validation = {relative: _sha256(source / relative) for relative in (*REPORT_FILES, *OPTIONAL_FILES)
                         if (source / relative).is_file()}
    report = compatibility_report(source, key, cfg)
    approvals = dict(approved_fields or {})
    if set(approvals).difference(report["unverified"]) or any(not isinstance(reason, str) or not reason.strip() for reason in approvals.values()):
        raise ValueError("Approvals must name currently unverified fields for this exact cell and give a reason")
    report["approval"] = {"scientific_key": key.to_dict(), "fields": approvals,
                          "statement": "Approved missing evidence remains unverified and excluded from scientific analysis"}
    if report["failures"] or set(report["unverified"]).difference(approvals):
        report["decision"] = "blocked_" + report["decision"]
        return report
    report["decision"] = ("would_import" if dry_run else "imported") + ("_unverified" if report["unverified"] else "")
    if target.exists():
        reference = target / "legacy_reference.json"
        if reference.is_file():
            existing = json.loads(reference.read_text(encoding="utf-8"))
            verified = validate_imported_run(target, key, cfg)
            if verified["valid"] and existing.get("source_fingerprints") == before_validation:
                registry = root / "import_registry" / key.dataset / key.model / key.condition / f"seed_{key.seed:03d}.json"
                if not dry_run and not registry.is_file():
                    atomic_write_json(registry, existing)
                report["decision"] = "already_imported_idempotent"
                return report
        raise FileExistsError(f"Refusing to overwrite native or incompatible target: {target}")
    if dry_run:
        return report
    source_files = [relative for relative in (*REPORT_FILES, *OPTIONAL_FILES) if (source / relative).is_file()]
    before = {relative: _sha256(source / relative) for relative in source_files}
    if before != before_validation:
        raise RuntimeError("Legacy source changed during compatibility validation")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.import-", dir=target.parent))
    try:
        for relative in source_files:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination)
            if _sha256(destination) != before[relative]:
                raise IOError(f"Copied artifact checksum mismatch: {relative}")
        reference = {
            "new_experiment_id": str(cfg.experiment.name), "scientific_key": key.to_dict(),
            "source_type": "legacy_reused", "original_run_path": str(source.resolve()),
            "original_experiment_identity": _nested(yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8")), "experiment.name"),
            "source_fingerprints": before, "compatibility_version": COMPATIBILITY_VERSION,
            "evidence_fingerprints": report["evidence_fingerprints"],
            "field_level_result": report["checks"], "approval": report["approval"],
            "artifact_locations": source_files, "mode": "copy_analysis_artifacts",
            "software_provenance": (
                json.loads((source / "environment.json").read_text(encoding="utf-8"))
                if (source / "environment.json").is_file() else None
            ),
            "code_provenance": _nested(
                yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8")), "code_provenance"
            ),
            "import_timestamp": datetime.now(timezone.utc).isoformat(),
            "import_status": "approved_unverified" if report["unverified"] else "validated",
        }
        atomic_write_json(staging / "legacy_reference.json", reference)
        after = {relative: _sha256(source / relative) for relative in source_files}
        if after != before:
            raise RuntimeError("Legacy source changed during read-only import")
        os.replace(staging, target)
        registry = root / "import_registry" / key.dataset / key.model / key.condition / f"seed_{key.seed:03d}.json"
        atomic_write_json(registry, reference)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def register_legacy_selection(cfg: DictConfig, evidence: str, dry_run: bool = False) -> dict[str, Any]:
    """Record the documented frozen choice; never claim historical pilots copied."""
    from orbit_cifar.experiments.provenance import make_selection_record, validate_selected

    if not evidence.strip():
        raise ValueError("Documented evidence text is required for the frozen legacy selected value")
    family = family_config(cfg, "cifar10", "resnet20_gn")
    selected = make_selection_record(
        family, {"source_type": "legacy_selected_value", "lambda_main": 0.1,
                 "selection_status": "legacy_selected_value", "evidence": evidence,
                 "candidate_runs_imported": 0, "candidate_runs_not_rerun": 4},
        source="legacy_documented_frozen",
    )
    validate_selected(family, selected)
    if not dry_run:
        root = Path(str(cfg.project_root)).expanduser().resolve() / str(cfg.output_root) / "pilot" / "cifar10" / "resnet20_gn"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "selected_hparams.yaml"
        if path.exists() and _read_mapping(path) != selected:
            raise FileExistsError("Refusing to replace a different frozen selection record")
        atomic_write_yaml(path, selected)
        atomic_write_json(root / "selection_diagnostics.json", {
            **selected, "statement": "Documented legacy value reused; zero historical candidate runs imported"
        })
    return selected


def import_allowlisted(
    cfg: DictConfig, dry_run: bool = False,
    approvals: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Import strict allowlisted cells with optional cell/field-scoped exceptions."""
    cells = [ScientificKey("cifar10", "resnet20_gn", str(condition), int(seed))
             for seed in cfg.legacy_reuse.seeds for condition in cfg.legacy_reuse.conditions]
    tokens = {f"{key.dataset}/{key.model}/{key.condition}/seed_{key.seed:03d}" for key in cells}
    approvals = dict(approvals or {})
    if set(approvals).difference(tokens):
        raise ValueError("Approval record contains a cell outside the frozen legacy allowlist")
    return [import_legacy_run(
        cfg, key, dry_run=dry_run,
        approved_fields=approvals.get(f"{key.dataset}/{key.model}/{key.condition}/seed_{key.seed:03d}"),
    ) for key in cells]
