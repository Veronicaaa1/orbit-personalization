"""Data-derived Markdown summary with scientifically bounded language."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from orbit_cifar.analysis.legacy_aggregate import AnalysisBundle


def _terminal_value(bundle: AnalysisBundle, condition: str, metric: str) -> float:
    if bundle.terminal_summary.empty or not {"condition", "metric", "mean"}.issubset(bundle.terminal_summary.columns):
        return math.nan
    rows = bundle.terminal_summary[
        (bundle.terminal_summary["condition"] == condition) & (bundle.terminal_summary["metric"] == metric)
    ]
    return float(rows.iloc[0]["mean"]) if len(rows) == 1 else math.nan


def _format(value: float) -> str:
    return "unavailable" if not math.isfinite(value) else f"{value:.6g}"


def generate_summary(bundle: AnalysisBundle, path: str | Path) -> None:
    """Summarize only observed artifacts; never hard-code experimental success."""
    main_checkpoints = bundle.checkpoint_rows[bundle.checkpoint_rows["condition"] == "main"] if not bundle.checkpoint_rows.empty else pd.DataFrame()
    if main_checkpoints.empty:
        initial_within = terminal_within = reduction = math.nan
    else:
        initial_within = float(main_checkpoints[main_checkpoints["epoch"] == main_checkpoints["epoch"].min()]["D_within"].mean())
        terminal_within = _terminal_value(bundle, "main", "D_within_bar")
        reduction = 1.0 - terminal_within / initial_within if initial_within > 0 and math.isfinite(terminal_within) else math.nan
    main_between = _terminal_value(bundle, "main", "D_between_bar")
    main_accuracy = _terminal_value(bundle, "main", "mean_node_accuracy")
    diagnostic = bundle.random_partition
    true_ratio = float(diagnostic.get("R_true_mean", math.nan))
    random_values = np.asarray(diagnostic.get("R_random_mean", []), dtype=float)
    random_mean = float(np.nanmean(random_values)) if random_values.size and np.isfinite(random_values).any() else math.nan
    p_value = float(diagnostic.get("p_perm_aggregate", math.nan))
    misaligned_within = _terminal_value(bundle, "misaligned_data", "D_within_bar")
    complete_between = _terminal_value(bundle, "complete_graph", "D_between_bar")
    large_between = _terminal_value(bundle, "large_communication", "D_between_bar")
    symmetric_global = _terminal_value(bundle, "fully_symmetric", "D_total_bar")
    missing = bundle.run_inventory[~bundle.run_inventory["valid"]] if not bundle.run_inventory.empty else pd.DataFrame()
    lines = [
        "# Orbit-wise alignment empirical summary",
        "",
        "This report is generated only from available run artifacts. The experiment empirically illustrates prediction-space behavior; it does not prove an SDE approximation, a Hurwitz criterion, or a Gaussian stationary limit.",
        "",
        "## Main condition",
        "",
        f"- Initial mean within-orbit dispersion: {_format(initial_within)}.",
        f"- Terminal-window within-orbit dispersion: {_format(terminal_within)}.",
        f"- Initial-to-terminal within reduction fraction: {_format(reduction)}.",
        f"- Terminal-window between-orbit dispersion: {_format(main_between)}.",
        f"- Mean node overall test accuracy: {_format(main_accuracy)}.",
        "",
        "## Matched-size random-partition diagnostic",
        "",
        f"- Across-seed reference-partition ratio: {_format(true_ratio)}.",
        f"- Mean ratio across matched-size random partitions: {_format(random_mean)}.",
        f"- Descriptive aggregate randomization p-value: {_format(p_value)}.",
        "",
        "This is a descriptive randomization diagnostic, not a formal test derived from an SDE.",
        "",
        "## Controls",
        "",
        f"- Misaligned-data terminal within dispersion: {_format(misaligned_within)} (main: {_format(terminal_within)}).",
        f"- Complete-graph terminal between dispersion: {_format(complete_between)} (main: {_format(main_between)}). The four groups are a reference partition under this control.",
        f"- Large-communication terminal between dispersion: {_format(large_between)} (main: {_format(main_between)}).",
        "",
        "## Fully symmetric qualitative control",
        "",
        f"- Terminal global predictive disagreement D_total (= D_global): {_format(symmetric_global)}.",
        "- The fully_symmetric condition uses K14 and globally balanced node populations. K14 has one true automorphism orbit containing all 14 nodes.",
        "- O1-O4 are reference groups only for fully_symmetric and complete_graph; their fixed-partition metrics are descriptive compatibility outputs, not automorphism-orbit metrics.",
        "- This is a qualitative empirical control for the absence of designed orbit-level specialization. Residual disagreement is expected from disjoint datasets and independent stochastic streams; exact parameter equality or exactly zero disagreement is not claimed.",
        "",
        "## Run completeness",
        "",
    ]
    if missing.empty:
        lines.append("All configured condition/seed artifacts were valid.")
    else:
        lines.append(f"{len(missing)} configured runs were missing, failed, diverged, or incomplete:")
        lines.append("")
        for _, row in missing.iterrows():
            lines.append(f"- {row['condition']} seed {int(row['seed'])}: {row['status']}.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
