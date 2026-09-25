"""Top-level aggregation, figure, table, and report command."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf

from orbit_cifar.analysis.legacy_aggregate import AnalysisBundle, aggregate_outputs
from orbit_cifar.analysis.legacy_figures import generate_figures
from orbit_cifar.analysis.legacy_report import generate_summary
from orbit_cifar.analysis.legacy_tables import generate_tables
from orbit_cifar.utils.io import ensure_directory

CONDITION_DISPLAY_ORDER = (
    "fully_symmetric",
    "complete_graph",
    "large_communication",
    "main",
    "misaligned_data",
    "no_communication",
)


def analysis_conditions(output_root: Path, configured: list[str]) -> list[str]:
    """Order configured conditions and discover the optional symmetric baseline."""
    available = list(dict.fromkeys(configured))
    symmetric_root = output_root / "final" / "fully_symmetric"
    if symmetric_root.is_dir() and "fully_symmetric" not in available:
        available.append("fully_symmetric")
    ordered = [condition for condition in CONDITION_DISPLAY_ORDER if condition in available]
    ordered.extend(condition for condition in available if condition not in ordered)
    return ordered


def analyze(cfg: DictConfig) -> AnalysisBundle:
    """Aggregate valid seeds, report missing runs, and generate all deliverables."""
    if OmegaConf.select(cfg, "experiment.name"):
        from orbit_cifar.analysis.active import analyze_active
        return analyze_active(cfg)  # type: ignore[return-value]
    project_root = Path(str(cfg.project_root)).expanduser().resolve()
    output_root = project_root / str(cfg.output_root)
    reports = ensure_directory(project_root / str(cfg.analysis.reports_dir))
    conditions = analysis_conditions(output_root, [str(value) for value in cfg.final.conditions])
    bundle = aggregate_outputs(output_root, conditions, [int(value) for value in cfg.final.seeds])
    generate_figures(bundle, reports, dpi=int(cfg.analysis.figure_dpi))
    generate_tables(bundle, reports)
    generate_summary(bundle, reports / "summary.md")
    return bundle
