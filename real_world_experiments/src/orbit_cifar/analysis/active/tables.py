"""Machine-readable tables and matching LaTeX exports."""

from pathlib import Path
import pandas as pd
from orbit_cifar.utils.io import ensure_directory
from orbit_cifar.analysis.serialization import _to_latex
from .contracts import ActiveAnalysisBundle

def _write_tables(bundle: ActiveAnalysisBundle, reports: Path) -> None:
    tables = ensure_directory(reports / "tables")
    for name in (
        "run_inventory", "per_seed_terminal_metrics", "terminal_summary_by_dataset_model",
        "paired_control_differences", "paired_control_summary", "robustness_consistency_per_seed", "robustness_consistency_summary",
        "relative_robustness_metrics", "main_terminal_stability", "classwise_performance",
        "pilot_summary_and_selection_provenance",
        "legacy_import_report",
        "orbit_geometry_per_seed", "predictive_motion_per_seed", "endpoint_displacement_per_seed",
        "terminal_orbit_geometry", "temporal_coverage",
    ):
        frame = getattr(bundle, name)
        frame.to_csv(tables / f"{name}.csv", index=False)
        (tables / f"{name}.tex").write_text(_to_latex(frame), encoding="utf-8")


