"""Seed-level finite-sample statistics and descriptive terminal stability."""

import math
from typing import Any
import numpy as np
import pandas as pd

def _stats(values: pd.Series, expected: int = 3) -> dict[str, float | int]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    clean = numeric[np.isfinite(numeric)]
    n = len(clean)
    sd = float(clean.std(ddof=1)) if n >= 2 else math.nan
    return {"mean": float(clean.mean()) if n else math.nan, "sample_sd": sd,
            "se": sd / math.sqrt(n) if n >= 2 else math.nan, "valid_n": n, "expected_n": expected}


def _terminal_value(frame: pd.DataFrame, dataset: str, model: str, condition: str, seed: int, name: str) -> float:
    rows = frame[(frame.dataset == dataset) & (frame.model == model) &
                 (frame.condition == condition) & (frame.seed == seed)]
    if rows.empty or name not in rows:
        return math.nan
    try:
        if len(rows) != 1:
            raise ValueError("Duplicate seed observations in active analysis")
        value = float(rows.iloc[0][name])
        return value if math.isfinite(value) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _direction(value: float, tolerance: float = 1e-14) -> str:
    if not math.isfinite(value):
        return "undefined"
    if value > tolerance:
        return "positive"
    if value < -tolerance:
        return "negative"
    return "tied"


def _safe_ratio(numerator: float, denominator: float, threshold: float) -> float:
    return math.nan if (not math.isfinite(numerator) or not math.isfinite(denominator)
                        or abs(denominator) < threshold) else numerator / denominator


def _stability(rows: pd.DataFrame, dataset: str, model: str, seed: int, metric: str, threshold: float,
               expected_epochs: tuple[int, ...] | None = None) -> dict[str, Any]:
    if rows.empty or metric not in rows or "epoch" not in rows:
        return {"dataset": dataset, "model": model, "seed": seed, "metric": metric, "status": "UNDEFINED",
                "mean": math.nan, "sample_sd": math.nan, "cv": math.nan, "slope": math.nan,
                "normalized_drift": math.nan, "relative_range": math.nan, "evaluations": 0}
    numeric = rows[["epoch", metric]].apply(pd.to_numeric, errors="coerce")
    window = numeric.sort_values("epoch").tail(5)
    values, epochs = window[metric], window.epoch
    valid = np.isfinite(values) & np.isfinite(epochs)
    if (len(values) != 5 or not valid.all() or epochs.nunique() != 5
            or (expected_epochs is not None and tuple(epochs) != expected_epochs)):
        return {"dataset": dataset, "model": model, "seed": seed, "metric": metric, "status": "UNDEFINED",
                "mean": math.nan, "sample_sd": math.nan, "cv": math.nan, "slope": math.nan,
                "normalized_drift": math.nan, "relative_range": math.nan,
                "evaluations": int(epochs[valid].nunique())}
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if abs(mean) < threshold:
        status, cv, drift, relative_range = "UNDEFINED", math.nan, math.nan, math.nan
        slope = float(np.polyfit(epochs.to_numpy(float), values.to_numpy(float), 1)[0])
    else:
        slope = float(np.polyfit(epochs.to_numpy(float), values.to_numpy(float), 1)[0])
        cv = sd / abs(mean)
        drift = abs(slope) * float(epochs.max() - epochs.min()) / abs(mean)
        relative_range = float(values.max() - values.min()) / abs(mean)
        status = "PASS" if cv < 0.05 and drift < 0.05 else "WARN"
    return {"dataset": dataset, "model": model, "seed": seed, "metric": metric, "status": status,
            "mean": mean, "sample_sd": sd, "cv": cv, "slope": slope,
            "normalized_drift": drift, "relative_range": relative_range, "evaluations": 5}


