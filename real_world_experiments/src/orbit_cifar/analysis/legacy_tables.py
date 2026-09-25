"""CSV and LaTeX exports for aggregated results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from orbit_cifar.analysis.legacy_aggregate import AnalysisBundle
from orbit_cifar.utils.io import ensure_directory
from orbit_cifar.analysis.serialization import _latex_escape, _to_latex


def _write_pair(frame: pd.DataFrame, directory: Path, name: str) -> None:
    frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.tex").write_text(_to_latex(frame), encoding="utf-8")


def generate_tables(bundle: AnalysisBundle, reports_directory: str | Path) -> None:
    reports = ensure_directory(reports_directory)
    _write_pair(bundle.terminal_summary, reports, "terminal_metrics")
    _write_pair(bundle.paired_differences, reports, "paired_control_differences")
    classification = bundle.orbit_classification
    if not classification.empty:
        group_columns = ["condition", "orbit", "class_id", "metric"]
        for optional in ("group_semantics", "group_label"):
            if optional in classification:
                group_columns.append(optional)
        classification = (
            classification.groupby(group_columns, as_index=False)
            .agg(mean=("value", "mean"), se=("value", lambda values: values.std(ddof=1) / (len(values) ** 0.5) if len(values) > 1 else float("nan")), valid_seeds=("value", "count"))
        )
    _write_pair(classification, reports, "class_wise_performance")
    symmetric_classification = bundle.node_classification
    if not symmetric_classification.empty:
        symmetric_classification = symmetric_classification[
            (symmetric_classification["condition"] == "fully_symmetric")
            & (symmetric_classification["class_id"] >= 0)
            & symmetric_classification["metric"].isin(["class_accuracy", "class_cross_entropy"])
        ]
        if not symmetric_classification.empty:
            per_seed = (
                symmetric_classification.groupby(["seed", "class_id", "metric"], as_index=False)["value"].mean()
            )
            symmetric_classification = (
                per_seed.groupby(["class_id", "metric"], as_index=False)
                .agg(
                    mean=("value", "mean"),
                    se=(
                        "value",
                        lambda values: values.std(ddof=1) / (len(values) ** 0.5)
                        if len(values) > 1
                        else float("nan"),
                    ),
                    valid_seeds=("value", "count"),
                )
            )
            symmetric_classification["aggregation"] = "node_average_within_seed"
    _write_pair(symmetric_classification, reports, "fully_symmetric_class_wise_performance")
    _write_pair(bundle.pilot_summary, reports, "pilot_summary")
    _write_pair(bundle.run_inventory, reports, "run_inventory")
    _write_pair(bundle.time_series, reports, "time_series")
