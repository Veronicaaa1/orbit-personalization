"""Paired active final orchestration with explicit readiness gates."""

from __future__ import annotations

from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.experiments.runner import run_experiment
from orbit_cifar.experiments.families import family_config
from orbit_cifar.experiments.planner import build_execution_plan


def run_final(cfg: DictConfig) -> pd.DataFrame:
    """Run ready/resumable active cells, or the explicitly selected historical sweep."""
    if hasattr(cfg.final, "datasets"):
        return run_active_cells(cfg)
    rows: list[dict[str, Any]] = []
    for seed in cfg.final.seeds:
        for condition in cfg.final.conditions:
            result = run_experiment(cfg, str(condition), int(seed), stage="final")
            row = {
                "seed": int(seed),
                "condition": str(condition),
                "numerically_stable": result.numerically_stable,
                "run_directory": str(result.run_directory),
            }
            row.update(result.terminal_metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def run_active_cells(
    cfg: DictConfig,
    dataset: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Execute authorized cells only; imported/complete cells are verified skips."""
    cfg = OmegaConf.merge(cfg, {"stage": "final"})
    for label, value, allowed in (("dataset", dataset, list(cfg.final.datasets)),
                                  ("model", model, list(cfg.final.models)),
                                  ("seed", seed, [int(item) for item in cfg.final.seeds])):
        if value is not None and value not in allowed:
            raise ValueError(f"Unknown requested {label} {value!r}; allowed values are {allowed}")
    plan = build_execution_plan(cfg, write=True)
    selected = [row for row in plan.rows if
                (dataset is None or row["dataset"] == dataset) and
                (model is None or row["model"] == model) and
                (seed is None or row["seed"] == int(seed))]
    blocked = [row for row in selected if row["state"] not in {"ready", "resumable_native", "imported", "complete_native"}]
    if blocked and any(value is not None for value in (dataset, model, seed)):
        keys = [f"{row['dataset']}/{row['model']}/{row['condition']}/seed_{row['seed']:03d}" for row in blocked]
        raise RuntimeError(f"Required cells are blocked or awaiting prerequisites: {keys}")
    rows: list[dict[str, Any]] = [{**row, "action": "skipped_blocked"} for row in blocked]
    for row in [value for value in selected if value["state"] in {"ready", "resumable_native", "imported", "complete_native"}]:
        if row["state"] in {"imported", "complete_native"}:
            rows.append({**row, "action": "skipped"})
            continue
        family = family_config(cfg, row["dataset"], row["model"])
        family.resume = row["state"] == "resumable_native"
        result = run_experiment(family, row["condition"], row["seed"], stage="final")
        rows.append({**row, "action": "resumed" if family.resume else "trained",
                     "numerically_stable": result.numerically_stable})
    return pd.DataFrame(rows)
