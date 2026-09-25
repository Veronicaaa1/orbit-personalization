"""Pre-registered lambda selection using pilot validation summaries only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.utils.io import atomic_write_json, atomic_write_yaml, ensure_directory
from orbit_cifar.utils.io import config_fingerprint
from orbit_cifar.experiments.provenance import (
    selection_context, make_selection_record, pilot_training_config, build_scientific_identity, manifest_for_config,
)
from orbit_cifar.experiments.conditions import resolve_condition


REQUIRED_PILOT_COLUMNS = {
    "lambda",
    "seed",
    "numerically_stable",
    "D_within_bar",
    "D_between_bar",
    "R_orbit_bar",
    "orbit_ensemble_accuracy",
}


def summarize_pilot_runs(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level terminal validation results by lambda."""
    missing = REQUIRED_PILOT_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(f"Pilot rows lack required columns: {sorted(missing)}")
    summaries: list[dict[str, Any]] = []
    for value, group in rows.groupby("lambda", sort=True):
        stable_values = group["numerically_stable"].map(_stable_flag)
        stable_metrics = group.loc[stable_values, ["D_within_bar", "D_between_bar", "orbit_ensemble_accuracy"]].to_numpy(dtype=float)
        if not np.isfinite(stable_metrics).all():
            raise ValueError("Stable pilot rows must contain finite validation metrics")
        summaries.append(
            {
                "lambda": float(value),
                "num_runs": int(len(group)),
                "num_stable": int(stable_values.sum()),
                "all_stable": bool(stable_values.all()),
                "D_within": float(group.loc[stable_values, "D_within_bar"].mean()),
                "D_between": float(group.loc[stable_values, "D_between_bar"].mean()),
                "R_orbit": float(group.loc[stable_values, "R_orbit_bar"].mean()),
                "orbit_ensemble_accuracy": float(group.loc[stable_values, "orbit_ensemble_accuracy"].mean()),
            }
        )
    return pd.DataFrame(summaries).sort_values("lambda").reset_index(drop=True)


def _stable_flag(value: Any) -> bool:
    if value is True or value is False or isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Pilot stability must be an explicit boolean, found {value!r}")


def select_lambdas(
    summary: pd.DataFrame,
    within_fraction_max: float = 0.75,
    between_fraction_min: float = 0.50,
    accuracy_drop_max: float = 0.03,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the primary constraints, then the disclosed fallback rule if needed."""
    baseline_rows = summary[np.isclose(summary["lambda"], 0.0)]
    if len(baseline_rows) != 1:
        raise ValueError("Pilot summary must contain exactly one lambda=0 baseline row")
    baseline = baseline_rows.iloc[0]
    if not bool(baseline["all_stable"]):
        raise ValueError("The lambda=0 pilot baseline is not numerically stable")
    dw0 = float(baseline["D_within"])
    db0 = float(baseline["D_between"])
    acc0 = float(baseline["orbit_ensemble_accuracy"])
    baseline_negligible = abs(dw0) < 1e-14 or abs(db0) < 1e-14
    diagnostics: list[dict[str, Any]] = []
    positive = summary[summary["lambda"] > 0].sort_values("lambda")
    for _, row in positive.iterrows():
        constraints = {
            "all_stable": bool(row["all_stable"]),
            "within_reduction": float(row["D_within"]) <= within_fraction_max * dw0,
            "between_preserved": float(row["D_between"]) >= between_fraction_min * db0,
            "accuracy_preserved": float(row["orbit_ensemble_accuracy"]) >= acc0 - accuracy_drop_max,
        }
        diagnostics.append(
            {
                "lambda": float(row["lambda"]),
                "constraints": constraints,
                "failed_constraints": [name for name, satisfied in constraints.items() if not satisfied],
            }
        )
    qualified = [item for item in diagnostics if all(item["constraints"].values())]
    selection_status = "primary"
    selection_outcome = "primary_pass"
    if qualified:
        lambda_main = min(float(item["lambda"]) for item in qualified)
        fallback_scores: dict[str, float] = {}
    else:
        selection_status = "fallback"
        selection_outcome = "fallback"
        eligible = positive[
            positive["all_stable"].astype(bool)
            & (positive["orbit_ensemble_accuracy"] >= acc0 - accuracy_drop_max)
        ].copy()
        if eligible.empty:
            selected = {
                "lambda_main": None, "selection_status": "no_valid_candidate",
                "selection_outcome": "no_valid_candidate", "selection_source": "balanced_validation_only",
            }
            return selected, {
                "baseline": {"Dw0": dw0, "Db0": db0, "Acc0": acc0, "near_zero_denominator": baseline_negligible},
                "thresholds": {"within_fraction_max": within_fraction_max, "between_fraction_min": between_fraction_min,
                               "accuracy_drop_max": accuracy_drop_max},
                "candidates": diagnostics, "fallback_scores": {}, "selected": selected,
            }
        if db0 <= 0:
            preservation = np.where(eligible["D_between"].to_numpy() > 0, 1.0, 0.0)
        else:
            preservation = np.minimum(eligible["D_between"].to_numpy(dtype=float) / db0, 1.0)
        scores = eligible["R_orbit"].to_numpy(dtype=float) * preservation
        if not np.isfinite(scores).any():
            raise ValueError("All fallback scores are undefined")
        eligible["fallback_score"] = scores
        best = eligible.loc[eligible["fallback_score"].idxmax()]
        lambda_main = float(best["lambda"])
        fallback_scores = {str(float(row["lambda"])): float(row["fallback_score"]) for _, row in eligible.iterrows()}
    stable_larger = positive[(positive["all_stable"].astype(bool)) & (positive["lambda"] > lambda_main)]
    if stable_larger.empty:
        lambda_large = 10.0 * lambda_main
        extrapolated = True
    else:
        lambda_large = float(stable_larger["lambda"].max())
        extrapolated = False
    selected = {
        "lambda_main": lambda_main,
        "lambda_large": lambda_large,
        "extrapolated_large_lambda": extrapolated,
        "selection_status": selection_status,
        "selection_outcome": selection_outcome,
        "selection_source": "balanced_validation_only",
    }
    full_diagnostics = {
        "baseline": {"Dw0": dw0, "Db0": db0, "Acc0": acc0, "near_zero_denominator": baseline_negligible},
        "thresholds": {
            "within_fraction_max": within_fraction_max,
            "between_fraction_min": between_fraction_min,
            "accuracy_drop_max": accuracy_drop_max,
        },
        "candidates": diagnostics,
        "fallback_scores": fallback_scores,
        "selected": selected,
    }
    return selected, full_diagnostics


def select_from_csv(
    pilot_runs_path: str | Path,
    output_directory: str | Path,
    within_fraction_max: float = 0.75,
    between_fraction_min: float = 0.50,
    accuracy_drop_max: float = 0.03,
    cfg: DictConfig | None = None,
) -> dict[str, Any]:
    """Read seed-level pilot validation data and write all selection artifacts."""
    source = Path(pilot_runs_path)
    lowered_parts = {part.lower() for part in source.parts}
    if "final" in lowered_parts or "test" in lowered_parts:
        raise ValueError("Lambda selection accepts pilot validation artifacts only, never final/test paths")
    rows = pd.read_csv(source)
    if cfg is not None and OmegaConf.select(cfg, "experiment.name"):
        _validate_active_pilot_rows(cfg, source, rows)
    summary = summarize_pilot_runs(rows)
    selected, diagnostics = select_lambdas(
        summary,
        within_fraction_max=within_fraction_max,
        between_fraction_min=between_fraction_min,
        accuracy_drop_max=accuracy_drop_max,
    )
    if cfg is not None and OmegaConf.select(cfg, "experiment.name"):
        expected_thresholds = (float(cfg.pilot.within_fraction_max), float(cfg.pilot.between_fraction_min), float(cfg.pilot.accuracy_drop_max))
        if (within_fraction_max, between_fraction_min, accuracy_drop_max) != expected_thresholds:
            raise ValueError("Selection thresholds disagree with the bound active configuration")
        selected = make_selection_record(cfg, {key: value for key, value in selected.items()
                                              if key not in {"lambda_large", "extrapolated_large_lambda"}})
        diagnostics["selected"] = selected
    output = ensure_directory(output_directory)
    summary.to_csv(output / "pilot_summary.csv", index=False)
    atomic_write_yaml(output / "selected_hparams.yaml", selected)
    atomic_write_json(output / "selection_diagnostics.json", diagnostics)
    return selected


def _validate_active_pilot_rows(cfg: DictConfig, source: Path, rows: pd.DataFrame) -> None:
    from orbit_cifar.experiments.completion import verify_completed_run
    from orbit_cifar.experiments.matrix import ScientificKey
    context = selection_context(cfg)
    digest = config_fingerprint(context)
    stored = json.loads((source.parent / "pilot_provenance.json").read_text(encoding="utf-8"))
    if stored != {"context": context, "context_fingerprint": digest}:
        raise ValueError("Wrong-family or stale pilot provenance")
    if "context_fingerprint" not in rows or not rows.context_fingerprint.eq(digest).all():
        raise ValueError("Pilot rows do not belong to this provenance generation")
    expected = {(float(value), int(seed)) for value in cfg.pilot.lambda_grid for seed in cfg.pilot.seeds}
    seeds = pd.to_numeric(rows["seed"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("Pilot CSV seeds must be exact integers")
    observed = list(zip(rows["lambda"].astype(float), rows["seed"].astype(int)))
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Pilot rows must contain every candidate/seed exactly once")
    pilot_cfg = pilot_training_config(cfg)
    for row in rows.to_dict("records"):
        directory = Path(row["run_directory"])
        key = ScientificKey(str(cfg.data.name), str(cfg.model.name), "main", int(row["seed"]))
        if not _stable_flag(row["numerically_stable"]):
            # Only recorded numerical divergence can exclude a candidate.
            status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
            resolved = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
            if status.get("status") != "diverged" or status.get("numerically_stable") is not False:
                raise ValueError("Ineligible pilot row lacks numerical-divergence evidence")
            manifest = manifest_for_config(cfg)
            condition, _ = resolve_condition("main", lambda_override=float(row["lambda"]))
            expected_identity = build_scientific_identity(pilot_cfg, condition, key.seed, manifest, manifest["class_names"])
            if resolved.get("scientific_identity") != expected_identity or resolved.get("scientific_fingerprint") != config_fingerprint(expected_identity):
                raise ValueError("Diverged candidate has wrong-family or stale scientific provenance")
            continue
        result = verify_completed_run(directory, key, pilot_cfg, expected_epochs=int(cfg.pilot.epochs))
        if not result["valid"]:
            raise ValueError(f"Pilot candidate evidence failed verification: {result['errors']}")
        terminal = json.loads((directory / "metrics/terminal_metrics.json").read_text(encoding="utf-8"))
        if not math.isclose(float(terminal["communication_lambda"]), float(row["lambda"]), rel_tol=0, abs_tol=0):
            raise ValueError("Pilot lambda differs from completed candidate evidence")
        for field in ("D_within_bar", "D_between_bar", "R_orbit_bar", "orbit_ensemble_accuracy"):
            left, right = float(row[field]), float(terminal[field])
            if not ((math.isnan(left) and math.isnan(right)) or math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError(f"Pilot CSV differs from completed validation metrics: {field}")
