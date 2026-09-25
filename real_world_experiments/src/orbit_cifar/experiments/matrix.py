"""Immutable scientific matrix and canonical path helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from omegaconf import DictConfig, OmegaConf

from orbit_cifar.constants import ACTIVE_CONDITIONS, ACTIVE_DATASETS, ACTIVE_MODELS, ACTIVE_SEEDS


@dataclass(frozen=True, order=True)
class ScientificKey:
    dataset: str
    model: str
    condition: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_matrix(cfg: DictConfig) -> None:
    expected = (list(ACTIVE_DATASETS), list(ACTIVE_MODELS), list(ACTIVE_CONDITIONS), list(ACTIVE_SEEDS))
    actual = (list(cfg.final.datasets), list(cfg.final.models), list(cfg.final.conditions), list(cfg.final.seeds))
    if actual != expected:
        raise ValueError(f"Active matrix must be exactly datasets/models/conditions/seeds={expected}; found {actual}")


def final_keys(cfg: DictConfig) -> Iterator[ScientificKey]:
    """Iterate in deterministic dataset -> model -> seed -> condition order."""
    validate_matrix(cfg)
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            for seed in cfg.final.seeds:
                for condition in cfg.final.conditions:
                    yield ScientificKey(str(dataset), str(model), str(condition), int(seed))


def validate_active_run(cfg: DictConfig, condition: str, seed: int, stage: str) -> None:
    """Guard the active API before any data load or output mutation."""
    validate_matrix(cfg)
    if str(cfg.data.name) not in ACTIVE_DATASETS or str(cfg.model.name) not in ACTIVE_MODELS:
        raise ValueError(f"Unsupported active family: {cfg.data.name}/{cfg.model.name}")
    if condition not in ACTIVE_CONDITIONS:
        raise ValueError(f"Inactive condition: {condition}")
    if stage == "final":
        allowed_seeds = ACTIVE_SEEDS
    elif stage == "pilot":
        if condition != "main":
            raise ValueError("Active pilot runs must use Main")
        allowed_seeds = tuple(int(value) for value in cfg.pilot.seeds)
    elif stage == "preflight":
        allowed_seeds = (int(OmegaConf.select(cfg, "preflight.seed") or 0),)
    else:
        raise ValueError(f"Unsupported active stage: {stage}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed_seeds:
        raise ValueError(f"Seed {seed!r} is outside the active {stage} seeds {list(allowed_seeds)}")


def pilot_families(cfg: DictConfig) -> list[tuple[str, str]]:
    validate_matrix(cfg)
    return [(str(dataset), str(model)) for dataset in cfg.final.datasets for model in cfg.final.models]


def run_directory(output_root: str | Path, key: ScientificKey) -> Path:
    return Path(output_root) / "final" / key.dataset / key.model / key.condition / f"seed_{key.seed:03d}"


def pilot_directory(output_root: str | Path, dataset: str, model: str) -> Path:
    return Path(output_root) / "pilot" / dataset / model


def legacy_allowlisted(cfg: DictConfig, key: ScientificKey) -> bool:
    reuse = cfg.legacy_reuse
    return (
        bool(reuse.enabled)
        and key.dataset == str(reuse.allowed_dataset)
        and key.model == str(reuse.allowed_model)
        and key.condition in {str(value) for value in reuse.conditions}
        and key.seed in {int(value) for value in reuse.seeds}
    )
