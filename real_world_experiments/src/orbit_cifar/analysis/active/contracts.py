"""Data contract for active analysis; one terminal observation per scientific key."""

from dataclasses import dataclass, field
import pandas as pd

@dataclass
class ActiveAnalysisBundle:
    run_inventory: pd.DataFrame
    per_seed_terminal_metrics: pd.DataFrame
    terminal_summary_by_dataset_model: pd.DataFrame
    paired_control_differences: pd.DataFrame
    paired_control_summary: pd.DataFrame
    robustness_consistency_per_seed: pd.DataFrame
    robustness_consistency_summary: pd.DataFrame
    relative_robustness_metrics: pd.DataFrame
    main_terminal_stability: pd.DataFrame
    classwise_performance: pd.DataFrame
    pilot_summary_and_selection_provenance: pd.DataFrame
    legacy_import_report: pd.DataFrame
    checkpoint_rows: pd.DataFrame
    orbit_geometry_per_seed: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictive_motion_per_seed: pd.DataFrame = field(default_factory=pd.DataFrame)
    endpoint_displacement_per_seed: pd.DataFrame = field(default_factory=pd.DataFrame)
    terminal_orbit_geometry: pd.DataFrame = field(default_factory=pd.DataFrame)
    temporal_coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


