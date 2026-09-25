"""Isolated validation-only real-data preflight matrix."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.families import family_config
from orbit_cifar.experiments.runner import run_experiment


def preflight_config(cfg: DictConfig) -> DictConfig:
    """Apply a disclosed short validation-only schedule before runner setup."""
    settings = OmegaConf.select(cfg, "preflight.training") or {
        "epochs": 1, "warmup_epochs": 0, "eval_every_epochs": 1,
        "checkpoint_every_epochs": 1, "terminal_window_checkpoints": 1,
        "num_workers": 0, "persistent_workers": False,
    }
    result = OmegaConf.merge(cfg, {"training": settings, "stage": "preflight"})
    if not 1 <= int(result.training.epochs) <= 2:
        raise ValueError("Preflight epoch budget must be one or two")
    if not 0 <= int(result.training.warmup_epochs) <= int(result.training.epochs):
        raise ValueError("Invalid preflight warmup schedule")
    return result


def preflight_rows(cfg: DictConfig) -> list[dict[str, Any]]:
    preflight = preflight_config(cfg)
    rows = []
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            for condition in cfg.final.conditions:
                spec, _ = resolve_condition(str(condition), {"lambda_main": 0.1})
                rows.append({"dataset": str(dataset), "model": str(model), "condition": str(condition),
                             "seed": int(OmegaConf.select(cfg, "preflight.seed") or 0),
                             "validation_only": True, "epochs": int(preflight.training.epochs),
                             "communication_lambda": spec.communication_lambda})
    if len(rows) != 16:
        raise ValueError(f"Preflight matrix must contain 16 combinations, found {len(rows)}")
    return rows


def run_preflights(
    cfg: DictConfig, *, dataset: str | None = None, model: str | None = None,
    condition: str | None = None, output_directory: str | Path | None = None,
) -> pd.DataFrame:
    """Execute selected preflights; artifacts are temporary unless explicitly kept."""
    rows = [row for row in preflight_rows(cfg)
            if (dataset is None or row["dataset"] == dataset)
            and (model is None or row["model"] == model)
            and (condition is None or row["condition"] == condition)]
    if not rows:
        raise ValueError("No active preflight combination matches the requested selectors")
    if output_directory is None:
        with TemporaryDirectory(prefix="orbit-preflight-") as temporary:
            return _execute_preflights(cfg, rows, Path(temporary))
    return _execute_preflights(cfg, rows, Path(output_directory).expanduser().resolve())


def _execute_preflights(cfg: DictConfig, rows: list[dict[str, Any]], root: Path) -> pd.DataFrame:
    results = []
    for row in rows:
        family = preflight_config(family_config(cfg, row["dataset"], row["model"]))
        family.output_root = str(root)
        try:
            result = run_experiment(
                family, row["condition"], int(row["seed"]), stage="preflight",
                lambda_override=float(row["communication_lambda"]), run_name=row["condition"],
            )
            results.append({**row, "status": "PASS" if result.numerically_stable else "FAIL",
                            "run_directory": str(result.run_directory)})
        except Exception as error:
            results.append({**row, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
    frame = pd.DataFrame(results)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "preflight_results.csv", index=False)
    return frame
