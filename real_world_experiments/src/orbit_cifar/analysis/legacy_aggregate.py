"""Robust aggregation that retains explicit failed and missing seeds."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from orbit_cifar.evaluation.random_partitions import aggregate_random_partition_diagnostics


@dataclass
class AnalysisBundle:
    run_inventory: pd.DataFrame
    checkpoint_rows: pd.DataFrame
    time_series: pd.DataFrame
    terminal_rows: pd.DataFrame
    terminal_summary: pd.DataFrame
    paired_differences: pd.DataFrame
    orbit_classification: pd.DataFrame
    node_classification: pd.DataFrame
    pairwise_mean: np.ndarray | None
    fully_symmetric_pairwise_mean: np.ndarray | None
    random_partition: dict[str, Any]
    confusion_means: dict[str, np.ndarray]
    pilot_summary: pd.DataFrame


def standard_error(values: pd.Series) -> float:
    """Compute seed-level sample standard error, undefined for one seed."""
    clean = values.dropna().astype(float)
    return float(clean.std(ddof=1) / math.sqrt(len(clean))) if len(clean) >= 2 else math.nan


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON mapping in {path}")
    return value


def aggregate_outputs(
    output_root: str | Path,
    conditions: Sequence[str],
    expected_seeds: Sequence[int],
) -> AnalysisBundle:
    """Discover valid runs while explicitly inventorying failures and missing seeds."""
    root = Path(output_root).expanduser().resolve()
    inventory: list[dict[str, Any]] = []
    checkpoints: list[pd.DataFrame] = []
    terminals: list[dict[str, Any]] = []
    orbit_tables: list[pd.DataFrame] = []
    node_tables: list[pd.DataFrame] = []
    pairwise: list[np.ndarray] = []
    fully_symmetric_pairwise: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    confusions: dict[str, list[np.ndarray]] = {}
    for condition in conditions:
        for seed in expected_seeds:
            run = root / "final" / condition / f"seed_{int(seed):03d}"
            status_path = run / "run_status.json"
            if not status_path.is_file():
                inventory.append({"condition": condition, "seed": int(seed), "status": "missing", "valid": False, "run_directory": str(run)})
                continue
            status = _read_json(status_path)
            valid = status.get("status") == "completed" and bool(status.get("numerically_stable", False))
            inventory.append({"condition": condition, "seed": int(seed), "status": status.get("status", "unknown"), "valid": valid, "run_directory": str(run)})
            if not valid:
                continue
            checkpoint_path = run / "metrics" / "checkpoint_metrics.csv"
            terminal_path = run / "metrics" / "terminal_metrics.json"
            if not checkpoint_path.is_file() or not terminal_path.is_file():
                inventory[-1]["valid"] = False
                inventory[-1]["status"] = "incomplete_artifacts"
                continue
            checkpoint = pd.read_csv(checkpoint_path)
            checkpoint["condition"] = condition
            checkpoint["seed"] = int(seed)
            checkpoints.append(checkpoint)
            terminal = _read_json(terminal_path)
            terminal.update({"condition": condition, "seed": int(seed)})
            if "D_total_bar" not in terminal and {"D_within_bar", "D_between_bar"}.issubset(terminal):
                terminal["D_total_bar"] = float(terminal["D_within_bar"]) + float(terminal["D_between_bar"])
            if condition == "fully_symmetric" and "D_global_bar" not in terminal and "D_total_bar" in terminal:
                terminal["D_global_bar"] = float(terminal["D_total_bar"])
            terminals.append(terminal)
            for filename, target in (("orbit_classification.csv", orbit_tables), ("node_classification.csv", node_tables)):
                path = run / "metrics" / filename
                if path.is_file():
                    frame = pd.read_csv(path)
                    if filename == "orbit_classification.csv" and "group_semantics" not in frame:
                        reference_only = condition in {"complete_graph", "fully_symmetric"}
                        frame["group_semantics"] = (
                            "reference_partition_only" if reference_only else "graph_automorphism_orbit"
                        )
                        frame["group_label"] = "reference_group" if reference_only else "orbit"
                    frame["condition"] = condition
                    frame["seed"] = int(seed)
                    target.append(frame)
            pairwise_path = run / "artifacts" / "pairwise_distance.npy"
            if condition == "main" and pairwise_path.is_file():
                pairwise.append(np.load(pairwise_path))
            if condition == "fully_symmetric" and pairwise_path.is_file():
                fully_symmetric_pairwise.append(np.load(pairwise_path))
            diagnostic_path = run / "artifacts" / "random_partition_diagnostic.npz"
            if condition == "main" and diagnostic_path.is_file():
                with np.load(diagnostic_path) as archive:
                    if "R_random" in archive:
                        diagnostics.append({"R_true": float(archive["R_true"]), "R_random": archive["R_random"].copy()})
            confusion_path = run / "artifacts" / "confusion_matrices.npz"
            if condition == "main" and confusion_path.is_file():
                with np.load(confusion_path) as archive:
                    for key in archive.files:
                        confusions.setdefault(key, []).append(archive[key].copy())
    inventory_frame = pd.DataFrame(inventory)
    checkpoint_frame = pd.concat(checkpoints, ignore_index=True) if checkpoints else pd.DataFrame()
    terminal_frame = pd.DataFrame(terminals)
    orbit_frame = pd.concat(orbit_tables, ignore_index=True) if orbit_tables else pd.DataFrame()
    node_frame = pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame()
    time_rows: list[dict[str, Any]] = []
    if not checkpoint_frame.empty:
        for (condition, epoch), group in checkpoint_frame.groupby(["condition", "epoch"], sort=False):
            for metric in (
                "D_within",
                "D_between",
                "D_total",
                "D_global",
                "R_orbit",
                "mean_node_accuracy",
                "orbit_ensemble_accuracy",
            ):
                if metric in group:
                    time_rows.append(
                        {
                            "condition": condition,
                            "epoch": int(epoch),
                            "metric": metric,
                            "mean": float(group[metric].mean()),
                            "se": standard_error(group[metric]),
                            "valid_seeds": int(group[metric].notna().sum()),
                        }
                    )
    time_series = pd.DataFrame(time_rows)
    terminal_summary_rows: list[dict[str, Any]] = []
    if not terminal_frame.empty:
        for condition, group in terminal_frame.groupby("condition", sort=False):
            for metric in (
                "D_within_bar",
                "D_between_bar",
                "D_total_bar",
                "D_global_bar",
                "R_orbit_bar",
                "mean_node_accuracy",
                "orbit_ensemble_accuracy",
            ):
                if metric in group:
                    terminal_summary_rows.append(
                        {
                            "condition": condition,
                            "metric": metric,
                            "mean": float(group[metric].mean()),
                            "se": standard_error(group[metric]),
                            "valid_seeds": int(group[metric].notna().sum()),
                        }
                    )
    terminal_summary = pd.DataFrame(terminal_summary_rows)
    paired_rows: list[dict[str, Any]] = []
    if not terminal_frame.empty and "main" in set(terminal_frame["condition"]):
        main = terminal_frame[terminal_frame["condition"] == "main"].set_index("seed")
        for condition in conditions:
            if condition == "main":
                continue
            control = terminal_frame[terminal_frame["condition"] == condition].set_index("seed")
            common = main.index.intersection(control.index)
            for seed in common:
                for metric in ("D_within_bar", "D_between_bar", "R_orbit_bar", "mean_node_accuracy"):
                    if metric in main and metric in control:
                        paired_rows.append(
                            {
                                "condition": condition,
                                "seed": int(seed),
                                "metric": metric,
                                "control_minus_main": float(control.loc[seed, metric] - main.loc[seed, metric]),
                            }
                        )
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        summaries = []
        for (condition, metric), group in paired.groupby(["condition", "metric"]):
            summaries.append(
                {
                    "condition": condition,
                    "seed": -1,
                    "metric": metric,
                    "control_minus_main": float(group["control_minus_main"].mean()),
                    "paired_difference_se": standard_error(group["control_minus_main"]),
                }
            )
        paired = pd.concat([paired, pd.DataFrame(summaries)], ignore_index=True)
    pairwise_mean = np.mean(np.stack(pairwise), axis=0) if pairwise else None
    fully_symmetric_pairwise_mean = (
        np.mean(np.stack(fully_symmetric_pairwise), axis=0) if fully_symmetric_pairwise else None
    )
    aggregate_diagnostic = aggregate_random_partition_diagnostics(diagnostics)
    confusion_means = {key: np.mean(np.stack(values), axis=0) for key, values in confusions.items()}
    pilot_path = root / "pilot" / "pilot_summary.csv"
    pilot_summary = pd.read_csv(pilot_path) if pilot_path.is_file() else pd.DataFrame()
    return AnalysisBundle(
        inventory_frame,
        checkpoint_frame,
        time_series,
        terminal_frame,
        terminal_summary,
        paired,
        orbit_frame,
        node_frame,
        pairwise_mean,
        fully_symmetric_pairwise_mean,
        aggregate_diagnostic,
        confusion_means,
        pilot_summary,
    )
