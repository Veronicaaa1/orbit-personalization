"""Scientific identities and validation-only selection provenance.

Paths, worker counts and device placement do not affect these identities. Data
assignments, transforms, initialization, the update protocol and selection rule do.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.config_schema import config_to_dict
from orbit_cifar.data.manifest import load_manifest
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.training.seed_plan import build_seed_plan
from orbit_cifar.utils.io import config_fingerprint

SELECTION_RULE_VERSION = "balanced-validation-primary-fallback-v1"
TRAINING_IDENTITY_FIELDS = (
    "epochs", "batch_size_per_node", "learning_rate", "momentum", "weight_decay",
    "warmup_epochs", "minimum_learning_rate", "communication_every_steps", "label_smoothing",
    "gradient_clipping", "mixed_precision", "eval_every_epochs", "checkpoint_every_epochs",
    "terminal_window_checkpoints", "decomposition_tolerance", "ratio_threshold",
)
MODEL_IDENTITY_FIELDS = (
    "name", "input_channels", "input_height", "input_width", "num_classes", "hidden_dimensions",
    "group_norm_max_groups", "initialization_mode", "initialization_rho", "absolute_scale_floor",
)


def manifest_for_config(cfg: DictConfig, balanced: bool = False) -> dict[str, Any]:
    project = Path(str(cfg.project_root)).expanduser().resolve()
    if OmegaConf.select(cfg, "experiment.name"):
        path = project / str(cfg.manifest_root) / str(cfg.data.name) / ("balanced.json" if balanced else "heterogeneous.json")
    else:
        path = project / str(cfg.manifest_path)
    return load_manifest(path)


def pilot_training_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.merge(cfg, {"training": {
        "epochs": int(cfg.pilot.epochs), "eval_every_epochs": int(cfg.pilot.eval_every_epochs),
        "terminal_window_checkpoints": int(cfg.pilot.terminal_window_checkpoints),
    }, "stage": "pilot"})


def build_scientific_identity(
    cfg: DictConfig, condition: Any, seed: int, manifest: dict[str, Any], class_names: Any,
) -> dict[str, Any]:
    resolved = config_to_dict(cfg)
    condition_value = condition.to_dict() if hasattr(condition, "to_dict") else condition
    _, graph = resolve_condition(condition_value["name"], lambda_override=condition_value["communication_lambda"])
    return {
        "experiment": OmegaConf.select(cfg, "experiment.name") or "legacy_cifar10",
        "stage": str(cfg.stage),
        "dataset": {"name": str(cfg.data.name),
                    "input_shape": [int(cfg.model.input_channels), int(cfg.model.input_height), int(cfg.model.input_width)],
                    "num_classes": int(cfg.model.num_classes), "class_names": list(class_names)},
        "dataset_version": str(cfg.data.dataset_version),
        "assignment_fingerprint": config_fingerprint(manifest),
        "graph_fingerprint": config_fingerprint({"adjacency": graph.adjacency.tolist(), "weights": graph.communication_weights.tolist()}),
        "seed_plan": build_seed_plan(int(seed), namespace=str(cfg.stage)),
        "transforms": {"train_augmentation": bool(cfg.data.train_augmentation),
                       "mean": resolved["data"].get("normalization_mean"), "std": resolved["data"].get("normalization_std"),
                       "definition_version": "torchvision-to-tensor-normalize-cifar-random-crop4-flip-v1"},
        "model": {key: resolved["model"].get(key) for key in MODEL_IDENTITY_FIELDS},
        "condition": condition_value, "seed": int(seed), "deterministic": bool(cfg.deterministic),
        "training": {key: resolved["training"].get(key) for key in TRAINING_IDENTITY_FIELDS},
    }


def selection_context(cfg: DictConfig, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest if manifest is not None else manifest_for_config(cfg)
    dataset = str(cfg.data.name)
    if manifest.get("dataset", dataset) != dataset:
        raise ValueError("Selection manifest belongs to a different dataset")
    if manifest.get("source_split", "train") != "train":
        raise ValueError("Selection manifest must use the official training source split")
    if any(type(value) is not int for value in cfg.pilot.seeds):
        raise ValueError("Pilot seeds must be exact integers")
    pilot = config_to_dict(pilot_training_config(cfg))
    validation = manifest.get("validation_indices")
    if not isinstance(validation, list) or not validation:
        raise ValueError("Selection requires a nonempty validation probe")
    context = {
        "experiment_name": str(cfg.experiment.name), "dataset": dataset, "model": str(cfg.model.name),
        "manifest_fingerprint": config_fingerprint(manifest),
        "validation_probe_fingerprint": config_fingerprint({
            "dataset": dataset, "source_split": "official_train", "dataset_version": str(cfg.data.dataset_version),
            "source_labels_checksum": manifest.get("source_labels_checksum"), "indices": validation,
        }),
        "pilot_seeds": [int(value) for value in cfg.pilot.seeds],
        "candidate_grid": [float(value) for value in cfg.pilot.lambda_grid],
        "epoch_budget": int(cfg.pilot.epochs),
        "training_settings": {
            "training": {key: pilot["training"].get(key) for key in TRAINING_IDENTITY_FIELDS},
            "model": {key: pilot["model"].get(key) for key in MODEL_IDENTITY_FIELDS},
            "deterministic": bool(cfg.deterministic),
            "data": {key: pilot["data"].get(key) for key in (
                "dataset_version", "split_seed", "train_augmentation", "normalization_mean", "normalization_std")},
            "condition": "main", "graph": "blown_up_path", "probe_split": "validation",
        },
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_thresholds": {key: float(cfg.pilot[key]) for key in (
            "within_fraction_max", "between_fraction_min", "accuracy_drop_max")},
    }
    if len(set(context["pilot_seeds"])) != len(context["pilot_seeds"]) or not context["pilot_seeds"]:
        raise ValueError("Pilot seeds must be nonempty and unique")
    grid = context["candidate_grid"]
    if not grid or len(set(grid)) != len(grid) or 0.0 not in grid or not any(value > 0 for value in grid) or any(not math.isfinite(value) or value < 0 for value in grid):
        raise ValueError("Pilot candidate grid must contain unique finite nonnegative values and a zero baseline")
    return context


def make_selection_record(
    cfg: DictConfig, selected: dict[str, Any], manifest: dict[str, Any] | None = None,
    source: str = "balanced_validation_only",
) -> dict[str, Any]:
    context = selection_context(cfg, manifest)
    provenance = {**context, "selected_lambda": selected.get("lambda_main"), "selection_source": source}
    return {**selected, "selection_source": source, "provenance": provenance,
            "provenance_fingerprint": config_fingerprint(provenance)}


def validate_selected(
    cfg: DictConfig, record: dict[str, Any], manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, dict) or not isinstance(record.get("provenance"), dict):
        raise ValueError("Selected lambda lacks bound scientific provenance")
    provenance = record["provenance"]
    if record.get("provenance_fingerprint") != config_fingerprint(provenance):
        raise ValueError("Selected-lambda provenance fingerprint mismatch")
    expected = selection_context(cfg, manifest)
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if mismatches:
        raise ValueError(f"Wrong-family or stale selected-lambda provenance: {mismatches}")
    value = record.get("lambda_main")
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("No valid positive Main lambda was selected")
    if provenance.get("selected_lambda") != value or provenance.get("selection_source") != record.get("selection_source"):
        raise ValueError("Selected lambda disagrees with its provenance")
    source = record.get("selection_source")
    if source == "balanced_validation_only":
        if float(value) not in expected["candidate_grid"] or record.get("selection_status") not in {"primary", "fallback"}:
            raise ValueError("Selected lambda is not an accepted pilot candidate")
    elif source == "legacy_documented_frozen":
        if (expected["dataset"], expected["model"], float(value)) != ("cifar10", "resnet20_gn", 0.1):
            raise ValueError("Documented frozen selection applies only to CIFAR-10/ResNet lambda=0.1")
    else:
        raise ValueError(f"Unrecognized selection provenance source: {source}")
    return record


def load_selected(cfg: DictConfig, path: str | Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        record = yaml.safe_load(handle)
    return validate_selected(cfg, record, manifest)
