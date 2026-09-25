"""Content-verified native completion publication and restart validation."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig

from orbit_cifar.constants import (
    ACTIVE_CONDITIONS, ACTIVE_DATASETS, ACTIVE_MODELS, ACTIVE_SEEDS, NUM_NODES, REFERENCE_ORBITS,
)
from orbit_cifar.experiments.matrix import ScientificKey
from orbit_cifar.experiments.provenance import (
    build_scientific_identity, manifest_for_config, load_selected, MODEL_IDENTITY_FIELDS, TRAINING_IDENTITY_FIELDS,
)
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.utils.io import atomic_write_json, config_fingerprint
from orbit_cifar.experiments.orbit_diagnostics import verify_diagnostics

TERMINAL_EPOCHS = [180, 185, 190, 195, 200]
REQUIRED_FILES = (
    "resolved_config.yaml", "seed_plan.json", "data_assignment.json", "environment.json",
    "checkpoints/final.pt", "checkpoints/last.pt", "metrics/terminal_metrics.json",
    "metrics/checkpoint_metrics.csv", "metrics/train_diagnostics.csv",
    "metrics/node_classification.csv", "metrics/orbit_classification.csv",
    "artifacts/pairwise_distance.npy", "artifacts/pairwise_distance.csv", "artifacts/confusion_matrices.npz",
    "artifacts/final_labels.pt", "artifacts/random_partition_diagnostic.npz",
    "graph/adjacency.npy", "graph/laplacian.npy", "graph/communication_weights.npy",
    "graph/edge_list.csv", "graph/graph_summary.json",
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_completion(path: str | Path, key: ScientificKey, fingerprint: str, epoch: int, global_step: int) -> dict[str, Any]:
    """Commit one generation after all artifacts/checkpoints have been written.

    The caller writes the returned fields to completed run_status.json last.
    A crash before that final status publication never declares completion.
    """
    root = Path(path)
    files = {item.relative_to(root).as_posix(): file_digest(item) for item in sorted(root.rglob("*"))
             if item.is_file() and item.name not in {"completion.json", "run_status.json"}
             and not item.name.endswith(".tmp")}
    publication = {"schema_version": 1, "scientific_key": key.to_dict(),
                   "scientific_fingerprint": fingerprint, "epoch": int(epoch), "global_step": int(global_step),
                   "files": files}
    generation = config_fingerprint(publication)
    atomic_write_json(root / "completion.json", {**publication, "generation": generation})
    return {"completion_generation": generation, "scientific_fingerprint": fingerprint,
            "scientific_key": key.to_dict(), "global_step": int(global_step)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict), "checkpoint must contain a mapping")
    return value


def _verify_key_path(root: Path, key: ScientificKey, resolved: dict[str, Any]) -> None:
    stage = resolved.get("stage")
    _require(key.dataset in ACTIVE_DATASETS and key.model in ACTIVE_MODELS and key.condition in ACTIVE_CONDITIONS,
             "scientific key is outside the active matrix")
    _require(stage in {"final", "pilot", "preflight"}, "unrecognized native stage")
    if stage == "final":
        _require(type(key.seed) is int and key.seed in ACTIVE_SEEDS and root.parent.name == key.condition, "noncanonical final condition or seed")
    _require(root.name == f"seed_{key.seed:03d}" and root.parents[1].name == key.model
             and root.parents[2].name == key.dataset and root.parents[3].name == stage,
             "run directory does not match its scientific namespace")


def _verify_checkpoint_state(checkpoint: dict[str, Any]) -> None:
    _require(len(checkpoint.get("models", [])) == len(checkpoint.get("optimizers", [])) == 14,
             "incomplete native checkpoint fleet")
    _require(checkpoint.get("scheduler", {}).get("epoch") == checkpoint.get("epoch"), "checkpoint scheduler generation mismatch")
    for state in checkpoint["models"]:
        _require(isinstance(state, dict) and bool(state), "empty checkpoint model state")
        _require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()),
                 "nonfinite or invalid checkpoint model parameters")
    for optimizer in checkpoint["optimizers"]:
        _require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict)
                 and isinstance(optimizer.get("param_groups"), list) and bool(optimizer["param_groups"]),
                 "incomplete optimizer state")
    rng = checkpoint.get("rng_state", {})
    _require({"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng), "incomplete RNG state")
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    cpu = rng["torch_cpu"]
    _require(isinstance(cpu, torch.Tensor) and cpu.dtype == torch.uint8 and cpu.ndim == 1, "invalid torch CPU RNG state")
    torch.Generator(device="cpu").set_state(cpu.cpu())
    _require(isinstance(rng["torch_cuda"], list) and all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
             and value.ndim == 1 and value.numel() > 0 for value in rng["torch_cuda"]), "invalid CUDA RNG state")


def _verify_expected_identity(cfg: DictConfig, key: ScientificKey, resolved: dict[str, Any], fingerprint: str) -> None:
    manifest = manifest_for_config(cfg, balanced=key.condition == "fully_symmetric")
    stage = str(resolved["stage"])
    _require(str(cfg.stage) == stage, "run stage differs from requested execution")
    if stage == "final":
        selected = {} if key.condition == "no_communication" else load_selected(
            cfg, Path(str(cfg.project_root)) / str(cfg.output_root) / "pilot" / key.dataset / key.model / "selected_hparams.yaml")
        condition, _ = resolve_condition(key.condition, selected)
    else:
        condition, _ = resolve_condition(key.condition, lambda_override=float(resolved["resolved_condition"]["communication_lambda"]))
    expected = build_scientific_identity(cfg, condition, key.seed, manifest, manifest["class_names"])
    _require(config_fingerprint(expected) == fingerprint, "run is stale against requested scientific settings")


def verify_completed_run(
    path: str | Path, key: ScientificKey, cfg: DictConfig | None = None, expected_epochs: int = 200,
) -> dict[str, Any]:
    """Fail closed on stale, truncated, mixed-generation or malformed results."""
    root = Path(path)
    try:
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
        _verify_key_path(root, key, resolved)
        publication = json.loads((root / "completion.json").read_text(encoding="utf-8"))
        generation = publication.pop("generation")
        _require(generation == config_fingerprint(publication) == status.get("completion_generation"), "completion generation mismatch")
        _require(status.get("status") == "completed" and status.get("numerically_stable") is True, "run is not stably completed")
        _require(status.get("last_valid_epoch") == publication.get("epoch") == expected_epochs, "completion epoch mismatch")
        observed = {"dataset": resolved["data"]["name"], "model": resolved["model"]["name"],
                    "condition": resolved["resolved_condition"]["name"], "seed": resolved["seed"]}
        _require(observed == key.to_dict() == publication.get("scientific_key") == status.get("scientific_key"), "scientific key mismatch")
        identity = resolved["scientific_identity"]
        fingerprint = config_fingerprint(identity)
        _require(fingerprint == resolved.get("scientific_fingerprint") == publication.get("scientific_fingerprint")
                 == status.get("scientific_fingerprint"), "scientific fingerprint mismatch")
        _require(identity.get("dataset", {}).get("name") == key.dataset and identity.get("model", {}).get("name") == key.model
                 and identity.get("condition", {}).get("name") == key.condition and identity.get("seed") == key.seed,
                 "scientific identity does not match run key")
        _require(identity.get("model") == {field: resolved["model"].get(field) for field in MODEL_IDENTITY_FIELDS}
                 and identity.get("training") == {field: resolved["training"].get(field) for field in TRAINING_IDENTITY_FIELDS}
                 and identity.get("condition") == resolved["resolved_condition"]
                 and identity.get("experiment") == resolved.get("experiment", {}).get("name")
                 and identity.get("stage") == resolved.get("stage"), "resolved settings disagree with scientific identity")
        _require(int(resolved["training"]["epochs"]) == expected_epochs and identity["training"]["epochs"] == expected_epochs,
                 "configured epoch budget mismatch")
        required = list(REQUIRED_FILES)
        if resolved.get("artifacts", {}).get("save_final_logits", True):
            required += ["artifacts/final_raw_logits.pt", "artifacts/final_centered_logits.pt"]
        files = publication.get("files", {})
        _require(all(name in files for name in required), "required final artifacts missing from completion manifest")
        for name, digest in files.items():
            artifact = (root / name).resolve()
            _require(artifact.is_relative_to(root.resolve()), "completion artifact escapes run directory")
            _require(artifact.is_file() and file_digest(artifact) == digest, f"completion artifact checksum mismatch: {name}")
        assignment = json.loads((root / "data_assignment.json").read_text(encoding="utf-8"))
        seed_plan = json.loads((root / "seed_plan.json").read_text(encoding="utf-8"))
        _require(assignment.get("manifest_fingerprint") == identity["assignment_fingerprint"], "assignment artifact disagrees with scientific identity")
        _require(seed_plan == identity.get("seed_plan"), "seed plan artifact disagrees with scientific identity")
        if cfg is not None:
            _verify_expected_identity(cfg, key, resolved, fingerprint)
        terminal = json.loads((root / "metrics/terminal_metrics.json").read_text(encoding="utf-8"))
        _require(terminal.get("numerically_stable") is True, "terminal metrics are not numerically stable")
        for field, expected in key.to_dict().items():
            _require(terminal.get(field) == expected, f"terminal scientific key mismatch: {field}")
        stage = str(resolved["stage"])
        _require(terminal.get("probe_split") == ("official_test" if stage == "final" else "validation"), "terminal probe split mismatch")
        metrics = pd.read_csv(root / "metrics/checkpoint_metrics.csv")
        scalar_fields = ["D_within", "D_between", "D_total", "orbit_ensemble_accuracy", "mean_node_accuracy", "raw_logit_rms", "centered_logit_rms"]
        _require(set(["epoch", "R_orbit", *scalar_fields]).issubset(metrics.columns), "terminal metrics schema is incomplete")
        _require(not metrics.empty and metrics["epoch"].is_unique and metrics["epoch"].is_monotonic_increasing, "checkpoint epochs are missing, duplicated or unordered")
        _require(int(metrics.iloc[-1]["epoch"]) == expected_epochs, "last metric checkpoint does not reach completion epoch")
        _require(np.isfinite(metrics[scalar_fields].to_numpy(dtype=float)).all(), "nonfinite checkpoint scalars")
        _require((metrics[["D_within", "D_between", "D_total"]].to_numpy(dtype=float) >= -1e-12).all(), "negative dispersion")
        _require(np.allclose(metrics.D_total, metrics.D_within + metrics.D_between, atol=1e-9, rtol=0), "D_total decomposition mismatch")
        denominators = metrics.D_within.to_numpy(dtype=float) + metrics.D_between.to_numpy(dtype=float)
        threshold = float(resolved["training"]["ratio_threshold"])
        expected_ratios = np.divide(metrics.D_between.to_numpy(dtype=float), denominators,
                                    out=np.full(len(metrics), np.nan), where=denominators >= threshold)
        _require(np.allclose(metrics.R_orbit.to_numpy(dtype=float), expected_ratios, rtol=1e-10, atol=1e-12, equal_nan=True),
                 "checkpoint R_orbit disagrees with dispersion")
        count = int(resolved["training"]["terminal_window_checkpoints"])
        window = metrics.tail(count)
        if stage == "final":
            _require(expected_epochs == 200 and window.epoch.tolist() == TERMINAL_EPOCHS, "terminal evaluations must be exactly 180,185,190,195,200")
            _require(terminal.get("terminal_window_short") is False, "completed final terminal window is short")
        for field in scalar_fields:
            terminal_field = field + "_bar" if field.startswith("D_") else field
            _require(terminal_field in terminal and math.isclose(float(terminal[terminal_field]), float(window[field].mean()), rel_tol=1e-10, abs_tol=1e-12), f"terminal scalar mismatch: {terminal_field}")
        for field in ("orbit_ensemble_accuracy", "mean_node_accuracy", "raw_logit_rms", "centered_logit_rms"):
            _require(math.isclose(float(terminal[f"final_{field}"]), float(metrics.iloc[-1][field]), rel_tol=1e-10, abs_tol=1e-12),
                     f"final-checkpoint scalar mismatch: {field}")
        ratio = window.R_orbit.to_numpy(dtype=float)
        ratio_expected = float(np.nanmean(ratio)) if np.isfinite(ratio).any() else math.nan
        ratio_actual = float(terminal["R_orbit_bar"])
        _require((math.isnan(ratio_expected) and math.isnan(ratio_actual)) or math.isclose(ratio_actual, ratio_expected, rel_tol=1e-10, abs_tol=1e-12), "terminal R_orbit mismatch")
        pairwise = np.load(root / "artifacts/pairwise_distance.npy", allow_pickle=False)
        _require(pairwise.shape == (14, 14) and np.isfinite(pairwise).all() and (pairwise >= -1e-12).all()
                 and np.allclose(pairwise, pairwise.T, atol=1e-10, rtol=0)
                 and np.allclose(np.diag(pairwise), 0, atol=1e-10, rtol=0), "invalid pairwise distance matrix")
        pairwise_csv = pd.read_csv(root / "artifacts/pairwise_distance.csv", index_col=0).to_numpy(dtype=float)
        _require(pairwise_csv.shape == (14, 14) and np.allclose(pairwise_csv, pairwise, rtol=1e-10, atol=1e-12),
                 "pairwise CSV and NPY artifacts disagree")
        _, graph = resolve_condition(key.condition, lambda_override=float(resolved["resolved_condition"]["communication_lambda"]))
        for name, expected_matrix in (("adjacency", graph.adjacency), ("laplacian", graph.laplacian), ("communication_weights", graph.communication_weights)):
            observed_matrix = np.load(root / f"graph/{name}.npy", allow_pickle=False)
            _require(observed_matrix.shape == expected_matrix.shape and np.allclose(observed_matrix, expected_matrix, rtol=0, atol=1e-12), f"graph artifact mismatch: {name}")
        classes = int(resolved["model"]["num_classes"])
        labels = torch.load(root / "artifacts/final_labels.pt", map_location="cpu", weights_only=True)
        _require(isinstance(labels, torch.Tensor) and labels.ndim == 1 and labels.numel() > 0
                 and labels.dtype == torch.int64 and bool(((labels >= 0) & (labels < classes)).all()), "invalid final labels")
        label_counts = torch.bincount(labels, minlength=classes).numpy()
        with np.load(root / "artifacts/confusion_matrices.npz", allow_pickle=False) as confusions:
            for kind, indices, terminal_field in (
                ("node", range(NUM_NODES), "final_mean_node_accuracy"),
                ("orbit", range(1, len(REFERENCE_ORBITS) + 1), "final_orbit_ensemble_accuracy"),
            ):
                table = pd.read_csv(root / f"metrics/{kind}_classification.csv")
                _require({kind, "class_id", "metric", "value"}.issubset(table.columns), f"invalid {kind} classification schema")
                accuracies = []
                for index in indices:
                    confusion_key = f"node_{index:02d}" if kind == "node" else f"orbit_{index}"
                    matrix = confusions[confusion_key]
                    _require(matrix.shape == (classes, classes) and np.isfinite(matrix).all() and (matrix >= 0).all()
                             and np.equal(matrix, np.floor(matrix)).all() and np.array_equal(matrix.sum(axis=1), label_counts),
                             f"invalid confusion matrix: {confusion_key}")
                    accuracy = float(np.trace(matrix) / matrix.sum())
                    overall = table[(table[kind] == index) & (table.class_id == -1) & (table.metric == "overall_accuracy")]
                    _require(len(overall) == 1 and math.isclose(float(overall.iloc[0]["value"]), accuracy, abs_tol=1e-12), "classification table/confusion mismatch")
                    accuracies.append(accuracy)
                _require(math.isclose(float(terminal[terminal_field]), float(np.mean(accuracies)), rel_tol=1e-10, abs_tol=1e-12), f"final classification scalar mismatch: {terminal_field}")
        if key.condition == "fully_symmetric":
            _require(math.isclose(float(terminal["D_global_bar"]), float(terminal["D_total_bar"]), rel_tol=1e-10, abs_tol=1e-12), "D_global does not equal D_total")
        final = _checkpoint(root / "checkpoints/final.pt")
        last = _checkpoint(root / "checkpoints/last.pt")
        for checkpoint in (last, final):
            _require(checkpoint.get("epoch") == expected_epochs and checkpoint.get("global_step") == publication.get("global_step")
                     == status.get("global_step") and checkpoint.get("config_fingerprint") == fingerprint,
                     "checkpoint/status generation mismatch")
            _verify_checkpoint_state(checkpoint)
            diagnostic_coverage = verify_diagnostics(root, checkpoint, resolved, completed=True)
            if resolved.get("orbit_diagnostics_profile"):
                _require(all(name in files for name in checkpoint["extra"]["orbit_diagnostics"]["files"]),
                         "diagnostic dependencies missing from completion manifest")
        _require(config_fingerprint(final.get("extra", {}).get("terminal_metrics")) == config_fingerprint(terminal), "checkpoint terminal metrics mismatch")
        for final_state, last_state in zip(final["models"], last["models"]):
            _require(final_state.keys() == last_state.keys() and all(torch.equal(final_state[name], last_state[name]) for name in final_state), "final and last checkpoint model generations differ")
        return {"valid": True, "errors": [], "scientific_fingerprint": fingerprint,
                "orbit_diagnostics": diagnostic_coverage}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, EOFError, IndexError, OverflowError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        return {"valid": False, "errors": [str(error)]}


def verify_resumable_run(path: str | Path, key: ScientificKey, cfg: DictConfig | None = None) -> dict[str, Any]:
    try:
        root = Path(path)
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
        resolved = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
        _verify_key_path(root, key, resolved)
        checkpoint = _checkpoint(root / "checkpoints/last.pt")
        identity = resolved["scientific_identity"]
        observed = {"dataset": identity["dataset"]["name"], "model": identity["model"]["name"], "condition": identity["condition"]["name"], "seed": identity["seed"]}
        _require(observed == key.to_dict(), "resumable scientific key mismatch")
        _require(status.get("status") in {"running", "interrupted"}, "run is not resumable")
        _require(config_fingerprint(identity) == resolved["scientific_fingerprint"] == checkpoint["config_fingerprint"], "resumable fingerprint mismatch")
        _require(0 <= checkpoint["epoch"] <= int(resolved["training"]["epochs"]), "resumable checkpoint epoch is outside the training budget")
        _verify_checkpoint_state(checkpoint)
        verify_diagnostics(root, checkpoint, resolved, completed=False)
        metrics = pd.read_csv(root / "metrics/checkpoint_metrics.csv")
        _require({"epoch", "D_within", "D_between", "D_total"}.issubset(metrics.columns), "resumable evaluation history is incomplete")
        committed = metrics[metrics.epoch <= checkpoint["epoch"]]
        cadence = int(resolved["training"]["eval_every_epochs"])
        expected = list(range(0, int(checkpoint["epoch"]) + 1, cadence))
        if checkpoint["epoch"] == int(resolved["training"]["epochs"]) and checkpoint["epoch"] not in expected:
            expected.append(checkpoint["epoch"])
        _require(committed.epoch.tolist() == expected, "resumable evaluation history does not cover committed checkpoints")
        _require(np.isfinite(committed[["D_within", "D_between", "D_total"]].to_numpy(dtype=float)).all()
                 and np.allclose(committed.D_total, committed.D_within + committed.D_between, atol=1e-9, rtol=0), "invalid resumable evaluation history")
        if checkpoint["epoch"] > 0:
            history = pd.read_csv(root / "metrics/train_diagnostics.csv")
            _require({"epoch", "global_step"}.issubset(history.columns), "resumable training history schema is incomplete")
            history = history[history.epoch < checkpoint["epoch"]]
            _require(not history.empty and history.epoch.max() == checkpoint["epoch"] - 1
                     and history.global_step.max() == checkpoint["global_step"], "resumable training history does not reach committed checkpoint")
        if cfg is not None:
            _verify_expected_identity(cfg, key, resolved, checkpoint["config_fingerprint"])
        return {"valid": True, "errors": [], "requires_finalization": checkpoint["epoch"] == int(resolved["training"]["epochs"])}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        return {"valid": False, "errors": [str(error)]}
