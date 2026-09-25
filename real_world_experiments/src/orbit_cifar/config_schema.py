from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

from orbit_cifar.constants import (
    ACTIVE_CONDITIONS,
    ACTIVE_DATASETS,
    ACTIVE_MODELS,
    ACTIVE_SEEDS,
    EXPERIMENT_NAME,
)

DATA_DEFAULT_FILES = {"mnist": "mnist_orbit", "cifar10": "cifar10_orbit"}
MODEL_DEFAULT_FILES = {"resnet20_gn": "resnet20_gn", "mlp_ln": "mlp_ln"}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size_per_node: int = 128
    learning_rate: float = 0.05
    momentum: float = 0.0
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    minimum_learning_rate: float = 0.0005
    communication_every_steps: int = 1
    diagnostics_every_steps: int = 100
    graph_penalty_every_epochs: int = 1
    label_smoothing: float = 0.0
    gradient_clipping: float | None = None
    mixed_precision: bool = False
    eval_every_epochs: int = 5
    checkpoint_every_epochs: int = 5
    persistent_workers: bool = True


def _merge_defaults(config_path: Path, raw: DictConfig) -> DictConfig:
    config_dir = config_path.parent
    merged = OmegaConf.create()
    for item in raw.get("defaults", []):
        if item == "_self_":
            continue
        if not isinstance(item, dict) and not OmegaConf.is_dict(item):
            raise ValueError(f"Unsupported defaults entry: {item!r}")
        group, name = next(iter(dict(item).items()))
        child = config_dir / str(group) / f"{name}.yaml"
        if not child.is_file():
            raise FileNotFoundError(f"Configuration group file does not exist: {child}")
        merged = OmegaConf.merge(merged, OmegaConf.load(child))
    own = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    own.pop("defaults", None)
    return OmegaConf.merge(merged, own)


def load_config(config_path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """Load the project YAML, compose its groups, and apply OmegaConf dot-list overrides."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    raw = OmegaConf.load(path)
    cfg = _merge_defaults(path, raw)
    override_values = list(overrides)
    override_cfg = OmegaConf.from_dotlist(override_values)
    # Shared defaults -> selected family defaults -> explicit YAML -> CLI.
    requested_data = OmegaConf.select(override_cfg, "data.name") or OmegaConf.select(cfg, "data.name")
    requested_model = OmegaConf.select(override_cfg, "model.name") or OmegaConf.select(cfg, "model.name")
    config_dir = path.parent
    if requested_data in DATA_DEFAULT_FILES:
        data_defaults = OmegaConf.load(config_dir / "data" / f"{DATA_DEFAULT_FILES[str(requested_data)]}.yaml")
        cfg = OmegaConf.merge(cfg, data_defaults)
    if requested_model in MODEL_DEFAULT_FILES:
        model_defaults = OmegaConf.load(config_dir / "model" / f"{MODEL_DEFAULT_FILES[str(requested_model)]}.yaml")
        cfg = OmegaConf.merge(cfg, model_defaults)
    requested_condition = OmegaConf.select(override_cfg, "condition.name") or OmegaConf.select(cfg, "condition.name")
    condition_path = config_dir / "condition" / f"{requested_condition}.yaml"
    if condition_path.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(condition_path))
    own = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    own.pop("defaults", None)
    cfg = OmegaConf.merge(cfg, own, override_cfg)
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    # Operational metadata preserves explicit values during family composition.
    cfg._composition = {
        "data_defaults": {name: OmegaConf.to_container(OmegaConf.load(config_dir / "data" / f"{file}.yaml"), resolve=True)["data"]
                          for name, file in DATA_DEFAULT_FILES.items()},
        "model_defaults": {name: OmegaConf.to_container(OmegaConf.load(config_dir / "model" / f"{file}.yaml"), resolve=True)["model"]
                           for name, file in MODEL_DEFAULT_FILES.items()},
        "explicit": OmegaConf.to_container(OmegaConf.merge(own, override_cfg), resolve=True),
        "resolved_data": OmegaConf.to_container(cfg.data, resolve=True),
        "resolved_model": OmegaConf.to_container(cfg.model, resolve=True),
    }
    return cfg


def validate_config(cfg: DictConfig) -> None:
    diagnostic_version = OmegaConf.select(cfg, "orbit_diagnostics.schema_version")
    if diagnostic_version is not None and diagnostic_version != 1:
        raise ValueError("Unsupported orbit diagnostic schema_version")
    if int(cfg.training.epochs) < 1:
        raise ValueError("training.epochs must be positive")
    if not 0 <= int(cfg.training.warmup_epochs) <= int(cfg.training.epochs):
        raise ValueError("training.warmup_epochs must be nonnegative and no greater than training.epochs")
    if int(cfg.training.communication_every_steps) < 1:
        raise ValueError("training.communication_every_steps must be at least one")
    if int(cfg.training.diagnostics_every_steps) < 1:
        raise ValueError("training.diagnostics_every_steps must be at least one")
    if int(cfg.training.graph_penalty_every_epochs) < 1:
        raise ValueError("training.graph_penalty_every_epochs must be at least one")
    if int(cfg.training.batch_size_per_node) < 1:
        raise ValueError("training.batch_size_per_node must be positive")
    if int(cfg.training.eval_every_epochs) < 1:
        raise ValueError("training.eval_every_epochs must be at least one")
    if int(cfg.training.checkpoint_every_epochs) < 1:
        raise ValueError("training.checkpoint_every_epochs must be at least one")
    if float(cfg.training.momentum) < 0:
        raise ValueError("training.momentum cannot be negative")
    dataset_specs = {
        "mnist": (1, 28, 28, 10),
        "cifar10": (3, 32, 32, 10),
    }
    data_name = str(cfg.data.name)
    if data_name not in dataset_specs:
        raise ValueError(f"Unknown dataset name: {data_name}")
    channels, height, width, classes = dataset_specs[data_name]
    for key, expected in (("input_channels", channels), ("input_height", height),
                          ("input_width", width), ("num_classes", classes)):
        configured = OmegaConf.select(cfg, f"data.{key}")
        if configured is not None and int(configured) != expected:
            raise ValueError(f"data.{key}={configured} conflicts with {data_name}'s required value {expected}")
    for key, expected in (
        ("input_channels", channels), ("input_height", height),
        ("input_width", width), ("num_classes", classes),
    ):
        configured = OmegaConf.select(cfg, f"model.{key}")
        if configured is not None and int(configured) != expected:
            raise ValueError(f"model.{key}={configured} conflicts with {data_name}'s required value {expected}")
        OmegaConf.update(cfg, f"model.{key}", expected, force_add=True)
    model_name = str(cfg.model.name)
    if model_name not in {*ACTIVE_MODELS, "tiny_cnn"}:
        raise ValueError(f"Unknown model architecture: {model_name}")
    if OmegaConf.select(cfg, "experiment.name") == EXPERIMENT_NAME:
        # Graph and condition metadata describe canonical implementations; they
        # cannot change the scientific graph/assignment through ignored fields.
        from orbit_cifar.experiments.conditions import resolve_condition
        condition_name = str(cfg.condition.name)
        if condition_name not in ACTIVE_CONDITIONS:
            raise ValueError(f"Inactive condition: {condition_name}")
        condition, _ = resolve_condition(condition_name)
        expected_metadata = {"graph": condition.graph_name, "assignment": condition.assignment,
                             "lambda_source": {"main": "selected_main", "no_communication": "zero",
                                               "misaligned_data": "selected_main", "fully_symmetric": "spectral_matched_family_main"}[condition_name]}
        for field, expected in expected_metadata.items():
            if str(cfg.condition.get(field)) != expected:
                raise ValueError(f"condition.{field} is read-only canonical metadata; expected {expected!r}")
        if str(cfg.graph.name) != "blown_up_path" or cfg.graph.get("spectral_match_complete") is not True:
            raise ValueError("graph metadata is read-only: blown_up_path with spectral_match_complete=true")
        required_paths = {
            "output_root": "outputs_mnist_cifar10_resnet_mlp_3seed",
            "report_root": "reports_mnist_cifar10_resnet_mlp_3seed",
        }
        for key, expected in required_paths.items():
            if str(OmegaConf.select(cfg, key)) != expected:
                raise ValueError(f"{key} must remain {expected!r} for the active experiment")
        expected_lists = {
            "final.datasets": list(ACTIVE_DATASETS),
            "final.models": list(ACTIVE_MODELS),
            "final.conditions": list(ACTIVE_CONDITIONS),
            "final.seeds": list(ACTIVE_SEEDS),
        }
        for key, expected in expected_lists.items():
            actual = list(OmegaConf.select(cfg, key) or [])
            if actual != expected:
                raise ValueError(f"{key} must be exactly {expected} for {EXPERIMENT_NAME}; found {actual}")
        if model_name not in ACTIVE_MODELS:
            raise ValueError(f"The active experiment allows only models {list(ACTIVE_MODELS)}")
        expected_augmentation = data_name != "mnist"
        if bool(cfg.data.train_augmentation) != expected_augmentation:
            raise ValueError(f"data.train_augmentation must be {expected_augmentation} for {data_name}")
        if bool(cfg.training.mixed_precision):
            raise ValueError("The active experiment requires deterministic FP32; mixed precision is unsupported")
        if str(cfg.model.initialization_mode) != "relative_rms":
            raise ValueError("The active experiment supports only relative_rms initialization")
        if model_name == "mlp_ln" and list(cfg.model.hidden_dimensions) != [256, 128]:
            raise ValueError("mlp_ln hidden_dimensions must be exactly [256, 128]")
        normalization = {
            "mnist": ([0.1307], [0.3081]),
            "cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        }[data_name]
        if list(cfg.data.normalization_mean) != normalization[0] or list(cfg.data.normalization_std) != normalization[1]:
            raise ValueError(f"Normalization constants conflict with the registered {data_name} defaults")
        allowed_legacy_tokens = {
            f"cifar10/resnet20_gn/{condition}/seed_{seed:03d}"
            for condition in ACTIVE_CONDITIONS for seed in ACTIVE_SEEDS
        }
        overrides = [str(value) for value in (OmegaConf.select(cfg, "legacy_reuse.retrain_overrides") or [])]
        invalid_overrides = sorted(set(overrides).difference(allowed_legacy_tokens))
        if invalid_overrides or len(overrides) != len(set(overrides)):
            raise ValueError(f"legacy_reuse.retrain_overrides must contain unique allowlisted cell keys; invalid={invalid_overrides}")
        for key, expected in (("mode", "copy_analysis_artifacts"), ("on_missing", "block"), ("on_incompatible", "block")):
            if str(OmegaConf.select(cfg, f"legacy_reuse.{key}")) != expected:
                raise ValueError(f"legacy_reuse.{key} must be {expected!r} for the active experiment")
        if bool(cfg.legacy_reuse.allow_retrain):
            raise ValueError("legacy_reuse.allow_retrain is deliberately broad and cannot authorize active cells; use retrain_overrides")
        if not bool(cfg.legacy_reuse.enabled):
            raise ValueError("legacy_reuse.enabled must remain true; use cell-scoped retrain_overrides for exceptions")
        legacy_expected = {
            "allowed_dataset": "cifar10", "allowed_model": "resnet20_gn",
            "conditions": list(ACTIVE_CONDITIONS), "seeds": list(ACTIVE_SEEDS),
        }
        for key, expected in legacy_expected.items():
            observed = OmegaConf.select(cfg, f"legacy_reuse.{key}")
            observed = list(observed) if isinstance(expected, list) else str(observed)
            if observed != expected:
                raise ValueError(f"legacy_reuse.{key} must be exactly {expected!r}; found {observed!r}")
    if str(cfg.device) not in {"auto", "cpu", "cuda"} and not str(cfg.device).startswith("cuda:"):
        raise ValueError(f"Unsupported device specification: {cfg.device}")


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Return a fully resolved plain mapping suitable for serialization."""
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("The root configuration must be a mapping")
    value.pop("_composition", None)
    return value
