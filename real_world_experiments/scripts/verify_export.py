"""Read-only verification of the archived 48-run scientific export.

This deliberately does not relax orbit_cifar's native completion verifier.
Exit zero means the exported evidence is consistent, including its explicitly
reported original omissions; it never means the native runs are complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

# The default command must not create Python caches, even without python -B.
sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from orbit_cifar.constants import ACTIVE_CONDITIONS, ACTIVE_DATASETS, ACTIVE_MODELS, ACTIVE_SEEDS
from orbit_cifar.experiments.conditions import resolve_condition

EXPORT_NAME = "results"
EXPORT_RELATIVE = Path("real_world_experiments") / EXPORT_NAME
KEY_FIELDS = ("dataset", "model", "condition", "seed")
TERMINAL_EPOCHS = [180, 185, 190, 195, 200]
RETAINED_EPOCHS = (175, 180, 185, 190, 195, 200)
SUMMARY_METRICS = ("D_within", "D_between", "D_total", "R_orbit",
                   "mean_node_accuracy", "orbit_ensemble_accuracy")
REQUIRED_FINAL = (
    "resolved_config.yaml", "run_status.json", "completion.json", "seed_plan.json",
    "data_assignment.json", "environment.json", "checkpoints/final.pt",
    "metrics/checkpoint_metrics.csv", "metrics/terminal_metrics.json",
    "metrics/train_diagnostics.csv", "metrics/node_classification.csv",
    "metrics/orbit_classification.csv", "artifacts/pairwise_distance.npy",
    "artifacts/pairwise_distance.csv", "artifacts/confusion_matrices.npz",
    "artifacts/final_labels.pt", "artifacts/final_raw_logits.pt",
    "artifacts/final_centered_logits.pt", "artifacts/random_partition_diagnostic.npz",
    "graph/adjacency.npy", "graph/communication_weights.npy", "graph/laplacian.npy",
    "graph/edge_list.csv", "graph/graph_summary.json",
)


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fingerprint(value: Any) -> str:
    # Same serialization as orbit_cifar.utils.io.config_fingerprint, without
    # importing that module's torch dependency for the lightweight checks.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str).encode("utf-8")).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    _require(bool(relative) and "\\" not in relative and ":" not in relative
             and not parts.is_absolute() and ".." not in parts.parts,
             f"unsafe artifact path: {relative}")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"artifact escapes export: {relative}")
    return path


def verify_package_manifest(root: Path) -> dict[str, Any]:
    """Hash every listed file; never trust the summary's valid-run count."""
    root = Path(root).resolve()
    hashes: dict[str, str] = {}
    errors: list[str] = []
    missing: list[str] = []
    listed: set[str] = set()
    with (root / "SHA256_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(bool(rows), "empty SHA256 manifest")
    for row in rows:
        name = row["relative_path"]
        try:
            _require(name not in listed, f"duplicate manifest path: {name}")
            listed.add(name)
            path = _safe_path(root, name)
            _require(re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]), f"invalid SHA256: {name}")
            if not path.is_file():
                missing.append(name)
                raise ValueError(f"manifest file missing: {name}")
            hashes[name] = _digest(path)
            _require(path.stat().st_size == int(row["bytes"]), f"manifest size mismatch: {name}")
            _require(hashes[name] == row["sha256"].lower(), f"manifest SHA256 mismatch: {name}")
        except (OSError, ValueError, TypeError) as error:
            errors.append(str(error))
    unlisted = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and p.relative_to(root).as_posix() not in listed)
    unexpected = sorted(set(unlisted) - {"PACKAGE_SUMMARY.json", "SHA256_MANIFEST.csv"})
    errors.extend(f"file not covered by export manifest: {name}" for name in unexpected)
    return {"entries": len(rows), "hashed_files": len(hashes), "hashes": hashes,
            "unlisted": unlisted, "missing": missing, "errors": errors}


def _export_exclusions(root: Path) -> set[str]:
    result = set()
    with (root / "tables/SOURCE_EXPORT_AUDIT.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["action"] != "EXCLUDED":
                continue
            if row["scope"] == "final":
                prefix = f"runs/{row['dataset']}/{row['model']}/{row['condition']}/seed_{int(row['seed']):03d}"
            elif row["scope"] == "pilot":
                prefix = f"pilots/{row['dataset']}/{row['model']}"
            else:
                raise ValueError(f"unknown export audit scope: {row['scope']}")
            name = f"{prefix}/{row['relative_path']}"
            _safe_path(root, name)
            _require(name not in result, f"duplicate exclusion in export audit: {name}")
            result.add(name)
    return result


def classify_missing(run_relative: str, artifact: str, audit_exclusions: set[str]) -> str | None:
    """Only the fixed original omissions AND their audit entries are accepted."""
    name = f"{run_relative}/{artifact}"
    if name not in audit_exclusions:
        return None
    parts = PurePosixPath(run_relative).parts
    if len(parts) == 5 and parts[0] == "runs":
        if artifact == "checkpoints/last.pt":
            return "original export omitted the native resume checkpoint"
        if parts[3] == "main" and artifact in {
            f"artifacts/terminal_predictions/epoch_{epoch}.pt" for epoch in RETAINED_EPOCHS
        }:
            return "original export omitted retained temporal prediction tensors"
    if len(parts) == 5 and parts[0] == "pilots" and artifact in {
        "checkpoints/last.pt", "checkpoints/final.pt",
        "artifacts/final_raw_logits.pt", "artifacts/final_centered_logits.pt",
    }:
        return "original compact pilot export omitted checkpoints or logits"
    return None


def _publication(root: Path, run: Path, key: dict[str, Any], stage: str,
                 hashes: dict[str, str], exclusions: set[str], report: dict[str, Any]) -> tuple[dict, dict]:
    relative = run.relative_to(root).as_posix()
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    status = _json(run / "run_status.json")
    publication = _json(run / "completion.json")
    generation = publication.pop("generation")
    identity = cfg["scientific_identity"]
    fingerprint = _fingerprint(identity)
    observed = {"dataset": cfg["data"]["name"], "model": cfg["model"]["name"],
                "condition": cfg["resolved_condition"]["name"], "seed": cfg["seed"]}
    _require(type(observed["seed"]) is int and observed == key == publication["scientific_key"]
             == status["scientific_key"], "scientific keys disagree")
    _require(generation == _fingerprint(publication) == status["completion_generation"],
             "completion generation mismatch")
    _require(fingerprint == cfg["scientific_fingerprint"] == publication["scientific_fingerprint"]
             == status["scientific_fingerprint"], "scientific fingerprint mismatch")
    _require(cfg["stage"] == identity["stage"] == stage, "stage mismatch")
    _require(identity["experiment"] == cfg["experiment"]["name"]
             and identity["dataset"]["name"] == key["dataset"]
             and identity["model"]["name"] == key["model"] and identity["seed"] == key["seed"],
             "scientific identity disagrees with resolved key")
    for section in ("model", "training"):
        _require(identity[section] == {field: cfg[section].get(field) for field in identity[section]},
                 f"resolved {section} disagrees with scientific identity")
    _require(identity["condition"] == cfg["resolved_condition"], "condition identity mismatch")
    _require(_json(run / "seed_plan.json") == identity["seed_plan"], "seed plan mismatch")
    _require(_json(run / "data_assignment.json")["manifest_fingerprint"] == identity["assignment_fingerprint"],
             "data assignment identity mismatch")
    expected_epoch = 200 if stage == "final" else 80
    _require(status["status"] == "completed" and status["numerically_stable"] is True,
             "source status is not stably completed")
    _require(status["last_valid_epoch"] == publication["epoch"] == cfg["training"]["epochs"]
             == expected_epoch, "completion epoch mismatch")
    _require(status["global_step"] == publication["global_step"], "global step mismatch")
    for artifact, expected in publication["files"].items():
        path = _safe_path(run, artifact)
        name = path.relative_to(root).as_posix()
        if not path.is_file():
            reason = classify_missing(relative, artifact, exclusions)
            if reason:
                report["known_original_exclusions"].append(
                    {"path": name, "reason": reason, "expected_sha256": expected})
            else:
                report["unexpected_missing"].append(name)
                report["errors"].append(f"unexplained missing completion artifact: {name}")
        else:
            _require(name in hashes, f"completion artifact absent from package manifest: {name}")
            _require(hashes[name] == expected, f"completion checksum mismatch: {name}")
    if stage == "final":
        for artifact in REQUIRED_FINAL:
            _require((run / artifact).is_file(), f"required exported artifact missing: {artifact}")
            if artifact not in {"completion.json", "run_status.json"}:
                _require(artifact in publication["files"], f"required artifact absent from completion: {artifact}")
    return cfg, publication


def verify_metrics(run: Path, key: dict[str, Any], cfg: dict, stage: str = "final") -> dict:
    """Check full evaluation cadence, terminal averages and scientific keys."""
    metrics = pd.read_csv(run / "metrics/checkpoint_metrics.csv")
    terminal = _json(run / "metrics/terminal_metrics.json")
    epochs, count = (200, 5) if stage == "final" else (80, 3)
    _require(metrics.epoch.tolist() == list(range(0, epochs + 1, 5)), "evaluation cadence is incomplete")
    _require(cfg["training"]["terminal_window_checkpoints"] == count, "terminal window configuration mismatch")
    _require({field: terminal[field] for field in KEY_FIELDS} == key, "terminal scientific key mismatch")
    _require(terminal["numerically_stable"] is True and terminal["terminal_window_short"] is False,
             "terminal stability/window flag mismatch")
    _require(terminal["probe_split"] == ("official_test" if stage == "final" else "validation"), "wrong probe split")
    scalars = ["D_within", "D_between", "D_total", "mean_node_accuracy", "orbit_ensemble_accuracy",
               "raw_logit_rms", "centered_logit_rms"]
    _require(np.isfinite(metrics[scalars].to_numpy(dtype=float)).all(), "nonfinite checkpoint metrics")
    _require((metrics[["D_within", "D_between", "D_total"]].to_numpy() >= -1e-12).all(), "negative dispersion")
    _require(np.allclose(metrics.D_total, metrics.D_within + metrics.D_between, rtol=0, atol=1e-9),
             "dispersion decomposition mismatch")
    denominator = (metrics.D_within + metrics.D_between).to_numpy()
    ratio = np.divide(metrics.D_between.to_numpy(), denominator, out=np.full(len(metrics), np.nan),
                      where=denominator >= float(cfg["training"]["ratio_threshold"]))
    _require(np.allclose(metrics.R_orbit, ratio, rtol=1e-10, atol=1e-12, equal_nan=True), "R_orbit mismatch")
    for field in scalars + ["R_orbit"]:
        name = field + "_bar" if field.startswith("D_") or field == "R_orbit" else field
        _require(np.isclose(terminal[name], metrics[field].tail(count).mean(), rtol=1e-10,
                            atol=1e-12, equal_nan=True), f"terminal mean mismatch: {name}")
    for field in scalars[3:]:
        _require(np.isclose(terminal["final_" + field], metrics[field].iloc[-1], rtol=1e-10, atol=1e-12),
                 f"final epoch scalar mismatch: {field}")
    if key["condition"] == "fully_symmetric":
        _require(np.isclose(terminal["D_global_bar"], terminal["D_total_bar"], rtol=1e-10, atol=1e-12),
                 "D_global differs from D_total")
    return terminal


def _matrices_and_classification(run: Path, cfg: dict, terminal: dict) -> None:
    pair = np.load(run / "artifacts/pairwise_distance.npy", allow_pickle=False)
    table = pd.read_csv(run / "artifacts/pairwise_distance.csv", index_col=0).to_numpy(dtype=float)
    _require(pair.shape == (14, 14) and np.isfinite(pair).all() and (pair >= -1e-12).all()
             and np.allclose(pair, pair.T, rtol=0, atol=1e-10)
             and np.allclose(np.diag(pair), 0, rtol=0, atol=1e-10)
             and table.shape == pair.shape and np.allclose(table, pair, rtol=1e-10, atol=1e-12),
             "pairwise matrices fail symmetry/nonnegativity/CSV consistency")
    _, graph = resolve_condition(cfg["resolved_condition"]["name"],
                                 lambda_override=cfg["resolved_condition"]["communication_lambda"])
    for field in ("adjacency", "communication_weights", "laplacian"):
        _require(np.allclose(np.load(run / f"graph/{field}.npy", allow_pickle=False),
                            getattr(graph, field), rtol=0, atol=1e-12), f"wrong graph {field}")
    class_counts = None
    with np.load(run / "artifacts/confusion_matrices.npz", allow_pickle=False) as confusions:
        for kind, indices, scalar in (("node", range(14), "final_mean_node_accuracy"),
                                      ("orbit", range(1, 5), "final_orbit_ensemble_accuracy")):
            classification = pd.read_csv(run / f"metrics/{kind}_classification.csv")
            accuracies = []
            for index in indices:
                name = f"node_{index:02d}" if kind == "node" else f"orbit_{index}"
                matrix = confusions[name]
                _require(matrix.shape == (10, 10) and np.isfinite(matrix).all() and (matrix >= 0).all()
                         and np.equal(matrix, np.floor(matrix)).all() and matrix.sum() == 10000,
                         f"invalid confusion matrix: {name}")
                if class_counts is None:
                    class_counts = matrix.sum(axis=1)
                _require(np.array_equal(matrix.sum(axis=1), class_counts), "confusion probe label counts differ")
                accuracy = np.trace(matrix) / matrix.sum()
                row = classification[(classification[kind] == index) & (classification.class_id == -1)
                                     & (classification.metric == "overall_accuracy")]
                _require(len(row) == 1 and np.isclose(row.iloc[0].value, accuracy, atol=1e-12, rtol=1e-10),
                         f"classification table differs from confusion matrix: {name}")
                accuracies.append(accuracy)
            _require(np.isclose(terminal[scalar], np.mean(accuracies), atol=1e-12, rtol=1e-10),
                     f"terminal classification mismatch: {scalar}")


def _tables(root: Path, terminals: dict[tuple, dict], configs: dict[tuple, dict]) -> None:
    table = pd.read_csv(root / "tables/terminal_metrics_per_seed_condition.csv")
    _require(len(table) == 48 and not table.duplicated(list(KEY_FIELDS)).any(), "terminal table key coverage mismatch")
    _require(set(table[list(KEY_FIELDS)].itertuples(index=False, name=None)) == set(terminals),
             "terminal table differs from the 48-run matrix")
    for _, row in table.iterrows():
        key = tuple(row[field] for field in KEY_FIELDS)
        terminal = terminals[key]
        for field in SUMMARY_METRICS + ("D_global",):
            name = field + "_bar" if field.startswith("D_") or field == "R_orbit" else "final_" + field
            value = terminal.get(name, terminal["D_total_bar"]) if field == "D_global" else terminal[name]
            _require(np.isclose(row[field], value, rtol=1e-10, atol=1e-12, equal_nan=True),
                     f"export table scalar mismatch: {key} {field}")
        _require(np.isclose(row.communication_lambda, configs[key]["resolved_condition"]["communication_lambda"],
                            rtol=1e-10, atol=1e-12), f"table lambda mismatch: {key}")
    summary = pd.read_csv(root / "tables/family_condition_summary_mean_sd_se.csv")
    expected_summary_keys = set(itertools.product(ACTIVE_DATASETS, ACTIVE_MODELS, ACTIVE_CONDITIONS, SUMMARY_METRICS))
    _require(len(summary) == 96 and set(summary[["dataset", "model", "condition", "metric"]]
             .itertuples(index=False, name=None)) == expected_summary_keys, "summary coverage mismatch")
    for _, row in summary.iterrows():
        values = table[(table.dataset == row.dataset) & (table.model == row.model)
                       & (table.condition == row.condition)][row.metric]
        expected = [values.mean(), values.std(ddof=1), values.std(ddof=1) / math.sqrt(3), values.min(), values.max()]
        _require(row.n == 3 and np.allclose(row[["mean", "sd", "se", "min", "max"]].astype(float),
                 expected, rtol=1e-10, atol=1e-12, equal_nan=True), "mean/SD/SE summary mismatch")
    contrasts = pd.read_csv(root / "tables/ABCD_contrasts_per_seed.csv")
    _require(len(contrasts) == 48 and not contrasts.duplicated(["dataset", "model", "seed", "check"]).any(),
             "contrast coverage mismatch")
    for _, row in contrasts.iterrows():
        values = table[(table.dataset == row.dataset) & (table.model == row.model) & (table.seed == row.seed)].set_index("condition")
        main = values.loc["main"]
        expected = {"A": main.D_between - main.D_within,
                    "B": values.loc["no_communication"].D_within - main.D_within,
                    "C": values.loc["misaligned_data"].D_within - main.D_within,
                    "D": main.D_total - values.loc["fully_symmetric"].D_global}[row.check[0]]
        _require(np.isclose(row.difference, expected, rtol=1e-10, atol=1e-12), "A/B/C/D contrast mismatch")
    contrast_summary = pd.read_csv(root / "tables/ABCD_contrasts_3seed_summary.csv")
    _require(len(contrast_summary) == 16 and not contrast_summary.duplicated(["dataset", "model", "check"]).any(),
             "contrast summary coverage mismatch")
    for _, row in contrast_summary.iterrows():
        values = contrasts[(contrasts.dataset == row.dataset) & (contrasts.model == row.model)
                           & (contrasts.check == row.check)].difference
        _require(len(values) == row.n == 3 and np.allclose([row["mean"], row.sd, row.se],
                 [values.mean(), values.std(ddof=1), values.std(ddof=1) / math.sqrt(3)], rtol=1e-10, atol=1e-12),
                 "contrast mean/SD/SE mismatch")
    for dataset, model in itertools.product(ACTIVE_DATASETS, ACTIVE_MODELS):
        expected = np.mean([np.load(root / f"runs/{dataset}/{model}/main/seed_{seed:03d}/artifacts/pairwise_distance.npy",
                                    allow_pickle=False) for seed in ACTIVE_SEEDS], axis=0)
        stem = root / f"tables/{dataset}_{model}_main_pairwise_3seed_mean"
        _require(np.allclose(np.load(stem.with_suffix(".npy"), allow_pickle=False), expected, rtol=1e-10, atol=1e-12)
                 and np.allclose(pd.read_csv(stem.with_suffix(".csv"), index_col=0).to_numpy(dtype=float),
                                 expected, rtol=1e-10, atol=1e-12), "three-seed pairwise average mismatch")


def _deep_run(run: Path, cfg: dict, publication: dict, terminal: dict) -> None:
    # Called only after every package hash and lightweight check passed.
    import torch
    from orbit_cifar.constants import REFERENCE_ORBITS
    from orbit_cifar.evaluation.dispersion import dispersion_metrics, pairwise_distance_matrix
    from orbit_cifar.experiments.completion import _verify_checkpoint_state
    from orbit_cifar.experiments.orbit_diagnostics import tensor_hash

    checkpoint = torch.load(run / "checkpoints/final.pt", map_location="cpu", weights_only=False)
    _verify_checkpoint_state(checkpoint)
    _require(checkpoint["epoch"] == 200 and checkpoint["global_step"] == publication["global_step"]
             and checkpoint["config_fingerprint"] == cfg["scientific_fingerprint"], "checkpoint generation mismatch")
    _require(_fingerprint(checkpoint["extra"]["terminal_metrics"]) == _fingerprint(terminal), "checkpoint terminal mismatch")
    raw = torch.load(run / "artifacts/final_raw_logits.pt", map_location="cpu", weights_only=True)
    centered = torch.load(run / "artifacts/final_centered_logits.pt", map_location="cpu", weights_only=True)
    labels = torch.load(run / "artifacts/final_labels.pt", map_location="cpu", weights_only=True)
    _require(raw.shape == centered.shape == (14, 10000, 10) and labels.shape == (10000,)
             and labels.dtype == torch.int64 and bool(((labels >= 0) & (labels < 10)).all()), "invalid logits/labels")
    _require(torch.allclose(centered, raw.double() - raw.double().mean(-1, keepdim=True), rtol=1e-10, atol=1e-12),
             "saved centered logits differ from raw logits")
    _require(np.allclose(pairwise_distance_matrix(centered), np.load(run / "artifacts/pairwise_distance.npy", allow_pickle=False),
                        rtol=1e-10, atol=1e-12), "logits and pairwise distances differ")
    final_row = pd.read_csv(run / "metrics/checkpoint_metrics.csv").iloc[-1]
    for field, value in dispersion_metrics(centered).items():
        if field in ("D_within", "D_between", "D_total", "R_orbit"):
            _require(np.isclose(value, final_row[field], rtol=1e-10, atol=1e-12, equal_nan=True), "logit dispersion mismatch")
    with np.load(run / "artifacts/confusion_matrices.npz", allow_pickle=False) as confusions:
        predictions = {f"node_{node:02d}": raw[node].argmax(-1) for node in range(14)}
        predictions.update({f"orbit_{index}": raw[list(orbit)].double().mean(0).argmax(-1)
                            for index, orbit in enumerate(REFERENCE_ORBITS, 1)})
        for name, predicted in predictions.items():
            matrix = torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10).numpy()
            _require(np.array_equal(matrix, confusions[name]), "logits and confusion matrix differ")
    if cfg.get("orbit_diagnostics_profile"):
        snapshot = checkpoint["extra"]["orbit_diagnostics"]
        base = dict(snapshot)
        generation = base.pop("generation")
        _require(_fingerprint(base) == generation, "diagnostic snapshot generation mismatch")
        metadata = _json(run / "artifacts/terminal_predictions/probe_metadata.json")
        _require(metadata == snapshot["metadata"] and metadata["labels_sha256"] == tensor_hash(labels),
                 "diagnostic labels/metadata mismatch")
        for filename, field in (("orbit_geometry.csv", "geometry"), ("predictive_motion.csv", "motion")):
            # Round-trip in memory to normalize CSV empty fields, never write a file.
            expected = pd.read_csv(io.StringIO(pd.DataFrame(snapshot[field]).to_csv(index=False)))
            pd.testing.assert_frame_equal(pd.read_csv(run / "metrics" / filename), expected,
                                          check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)


def verify_export(export_root: Path | str = ROOT / EXPORT_RELATIVE, *, deep: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable report without creating or modifying files."""
    root = Path(export_root).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": 1, "validation_scope": "exported_scientific_evidence", "export_root": str(root),
        "valid": False, "deep_requested": deep, "deep_runs_verified": 0,
        "errors": [], "unexpected_missing": [], "known_original_exclusions": [],
        "known_original_omissions": [], "runs": [], "pilots": [],
        "native_validation": {"complete": False, "performed": False,
            "reason": "Export layout and omitted source artifacts do not satisfy native completion verification."},
        "training_reproduced": False,
        "accuracy_definition": "Export accuracy tables use epoch 200; dispersion/R_orbit tables use the five-checkpoint terminal mean.",
        "integrity_note": "Checksums establish consistency with the supplied manifest, not an external signature.",
    }
    try:
        manifest = verify_package_manifest(root)
        hashes = manifest.pop("hashes")
        report["package_manifest"] = manifest
        report["errors"].extend(manifest["errors"])
        report["unexpected_missing"].extend(manifest["missing"])
        exclusions = _export_exclusions(root)
        expected_runs = {f"runs/{d}/{m}/{c}/seed_{s:03d}": dict(zip(KEY_FIELDS, (d, m, c, s)))
                         for d, m, c, s in itertools.product(ACTIVE_DATASETS, ACTIVE_MODELS, ACTIVE_CONDITIONS, ACTIVE_SEEDS)}
        actual_runs = {p.relative_to(root).as_posix() for p in (root / "runs").glob("*/*/*/seed_*") if p.is_dir()}
        if actual_runs != set(expected_runs):
            report["errors"].append(f"48-run coverage mismatch: missing={sorted(set(expected_runs) - actual_runs)}, extra={sorted(actual_runs - set(expected_runs))}")
        terminals, configs, publications = {}, {}, {}
        for relative, key in expected_runs.items():
            run = root / relative
            try:
                cfg, publication = _publication(root, run, key, "final", hashes, exclusions, report)
                terminal = verify_metrics(run, key, cfg)
                _matrices_and_classification(run, cfg, terminal)
                tuple_key = tuple(key[field] for field in KEY_FIELDS)
                terminals[tuple_key], configs[tuple_key], publications[relative] = terminal, cfg, publication
                report["runs"].append({**key, "path": relative, "export_checks_passed": True,
                                       "scientific_fingerprint": cfg["scientific_fingerprint"], "evaluations": 41,
                                       "terminal_epochs": TERMINAL_EPOCHS})
            except (OSError, ValueError, TypeError, KeyError, IndexError, AssertionError) as error:
                report["errors"].append(f"{relative}: {error}")
        expected_pilots = {f"pilots/{d}/{m}/{candidate}/seed_000":
                           {"dataset": d, "model": m, "condition": "main", "seed": 0}
                           for d, m, candidate in itertools.product(ACTIVE_DATASETS, ACTIVE_MODELS,
                                                                    ("lambda_0", "lambda_0p03", "lambda_0p1", "lambda_0p3"))}
        actual_pilots = {p.relative_to(root).as_posix() for p in (root / "pilots").glob("*/*/lambda_*/seed_*") if p.is_dir()}
        _require(actual_pilots == set(expected_pilots), "16-candidate pilot coverage mismatch")
        for relative, key in expected_pilots.items():
            try:
                cfg, _ = _publication(root, root / relative, key, "pilot", hashes, exclusions, report)
                verify_metrics(root / relative, key, cfg, "pilot")
                report["pilots"].append({**key, "path": relative, "export_checks_passed": True})
            except (OSError, ValueError, TypeError, KeyError, IndexError, AssertionError) as error:
                report["errors"].append(f"{relative}: {error}")
        for dataset, model in itertools.product(ACTIVE_DATASETS, ACTIVE_MODELS):
            family = root / "pilots" / dataset / model
            selected = yaml.safe_load((family / "selected_hparams.yaml").read_text(encoding="utf-8"))
            _require(_fingerprint(selected["provenance"]) == selected["provenance_fingerprint"], "selection provenance mismatch")
            omitted = family / "pilot_provenance.json"
            if not omitted.is_file():
                report["known_original_omissions"].append({"path": omitted.relative_to(root).as_posix(),
                    "reason": "Original compact export lacks the context file required by native pilot selection validation."})
        if len(terminals) == 48:
            _tables(root, terminals, configs)
            report["summary_tables_verified"] = True
        if deep and not report["errors"]:
            for relative, key in expected_runs.items():
                tuple_key = tuple(key[field] for field in KEY_FIELDS)
                try:
                    _deep_run(root / relative, configs[tuple_key], publications[relative], terminals[tuple_key])
                    report["deep_runs_verified"] += 1
                except Exception as error:
                    report["errors"].append(f"deep {relative}: {type(error).__name__}: {error}")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AssertionError) as error:
        report["errors"].append(str(error))
    report["unexpected_missing"] = sorted(set(report["unexpected_missing"]))
    report["known_original_exclusion_count"] = len(report["known_original_exclusions"])
    report["valid"] = not report["errors"] and len(report["runs"]) == 48 and len(report["pilots"]) == 16
    report["status"] = "verified_export_with_declared_gaps" if report["valid"] else "export_verification_failed"
    return report


def validate_output_directory(target: Path, project_root: Path, export_root: Path) -> Path:
    """Explicit reports go to a new directory, never into scientific evidence."""
    target, project_root, export_root = target.resolve(), project_root.resolve(), export_root.resolve()
    _require(not target.exists(), f"output directory already exists: {target}")
    _require(target != project_root and not target.is_relative_to(export_root), "output would overwrite scientific evidence")
    protected = ("real_world_experiments", "src", "tests", "configs", "scripts", "data", "docs", "reports", "outputs",
                 "reports_mnist_cifar10_resnet_mlp_3seed", "outputs_mnist_cifar10_resnet_mlp_3seed",
                 "numerical experiments", "output_numerical experiments", EXPORT_RELATIVE,
                 ".git", ".tmp", ".tmp_compatibility_figure_check")
    _require(not any(target.is_relative_to((project_root / name).resolve()) for name in protected),
             "output directory lies inside protected source or evidence")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project/data root (parent of real_world_experiments by default)")
    parser.add_argument("--export", type=Path, help="Export directory; relative paths are relative to --root")
    parser.add_argument("--deep", action="store_true", help="Also inspect trusted local checkpoints and recompute final logits; native gaps remain")
    parser.add_argument("--output", type=Path, help="Explicitly save verification.json in a NEW directory outside source/evidence")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    export = args.export or EXPORT_RELATIVE
    export = export if export.is_absolute() else root / export
    try:
        target = validate_output_directory(args.output, root, export) if args.output else None
        report = verify_export(export, deep=args.deep)
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        if target is not None:
            target.mkdir(parents=True, exist_ok=False)
            (target / "verification.json").write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if report["valid"] else 1
    except (OSError, ValueError) as error:
        print(json.dumps({"valid": False, "errors": [str(error)]}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
