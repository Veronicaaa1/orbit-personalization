from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from orbit_cifar.analysis.run import CONDITION_DISPLAY_ORDER, analysis_conditions, analyze
from orbit_cifar.config_schema import load_config


OLD_CONDITIONS = ["main", "no_communication", "misaligned_data", "complete_graph", "large_communication"]


def _write_fully_symmetric_artifacts(root: Path) -> None:
    run = root / "outputs" / "final" / "fully_symmetric" / "seed_000"
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts").mkdir(parents=True)
    (run / "run_status.json").write_text(
        json.dumps({"status": "completed", "numerically_stable": True}), encoding="utf-8"
    )
    checkpoint = pd.DataFrame(
        [
            {
                "epoch": 0,
                "D_within": 0.02,
                "D_between": 0.01,
                "D_total": 0.03,
                "D_global": 0.03,
                "R_orbit": 1 / 3,
                "mean_node_accuracy": 0.1,
                "orbit_ensemble_accuracy": 0.1,
            },
            {
                "epoch": 5,
                "D_within": 0.002,
                "D_between": 0.001,
                "D_total": 0.003,
                "D_global": 0.003,
                "R_orbit": 1 / 3,
                "mean_node_accuracy": 0.2,
                "orbit_ensemble_accuracy": 0.2,
            },
        ]
    )
    checkpoint.to_csv(run / "metrics" / "checkpoint_metrics.csv", index=False)
    (run / "metrics" / "terminal_metrics.json").write_text(
        json.dumps(
            {
                "D_within_bar": 0.002,
                "D_between_bar": 0.001,
                "D_global_bar": 0.003,
                "R_orbit_bar": 1 / 3,
                "mean_node_accuracy": 0.2,
                "orbit_ensemble_accuracy": 0.2,
            }
        ),
        encoding="utf-8",
    )
    node_rows = []
    for node in range(14):
        for class_id in range(10):
            node_rows.extend(
                [
                    {"node": node, "node_label": node + 1, "class_id": class_id, "metric": "class_accuracy", "value": 0.2},
                    {"node": node, "node_label": node + 1, "class_id": class_id, "metric": "class_cross_entropy", "value": 2.0},
                ]
            )
    pd.DataFrame(node_rows).to_csv(run / "metrics" / "node_classification.csv", index=False)
    pd.DataFrame(
        [
            {
                "orbit": group,
                "class_id": class_id,
                "metric": "class_accuracy",
                "value": 0.2,
                "group_semantics": "reference_partition_only",
                "group_label": "reference_group",
            }
            for group in range(1, 5)
            for class_id in range(10)
        ]
    ).to_csv(run / "metrics" / "orbit_classification.csv", index=False)
    np.save(run / "artifacts" / "pairwise_distance.npy", np.full((14, 14), 0.003) - np.eye(14) * 0.003)


def _analysis_config(project_root: Path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    cfg.project_root = str(project_root)
    cfg.output_root = "outputs"
    cfg.analysis.reports_dir = "reports"
    cfg.analysis.figure_dpi = 30
    cfg.final.seeds = [0]
    cfg.final.conditions = OLD_CONDITIONS
    return cfg


def test_analysis_is_backward_compatible_when_baseline_is_absent(tmp_path: Path) -> None:
    cfg = _analysis_config(tmp_path)
    conditions = analysis_conditions(tmp_path / "outputs", OLD_CONDITIONS)
    assert "fully_symmetric" not in conditions
    assert conditions == [name for name in CONDITION_DISPLAY_ORDER if name in OLD_CONDITIONS]
    bundle = analyze(cfg)
    assert set(bundle.run_inventory["condition"]) == set(OLD_CONDITIONS)


def test_analysis_discovers_and_labels_fully_symmetric_baseline(tmp_path: Path) -> None:
    _write_fully_symmetric_artifacts(tmp_path)
    cfg = _analysis_config(tmp_path)
    bundle = analyze(cfg)
    assert bundle.run_inventory.iloc[0]["condition"] == "fully_symmetric"
    assert bundle.fully_symmetric_pairwise_mean is not None
    reports = tmp_path / "reports"
    for name in (
        "global_disagreement_baseline.png",
        "fully_symmetric_pairwise_heatmap.png",
        "fully_symmetric_class_accuracy.png",
        "fully_symmetric_class_wise_performance.csv",
    ):
        assert (reports / name).is_file()
    summary = (reports / "summary.md").read_text(encoding="utf-8")
    assert "one true automorphism orbit containing all 14 nodes" in summary
    assert "O1-O4 are reference groups only for fully_symmetric and complete_graph" in summary
    classification = pd.read_csv(reports / "class_wise_performance.csv")
    symmetric = classification[classification["condition"] == "fully_symmetric"]
    assert set(symmetric["group_semantics"]) == {"reference_partition_only"}
    assert set(symmetric["group_label"]) == {"reference_group"}
