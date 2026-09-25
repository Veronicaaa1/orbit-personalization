"""Read canonical active observations and derive seed-level analysis tables."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from orbit_cifar.experiments.planner import build_execution_plan
from orbit_cifar.data.datasets import dataset_spec
from .contracts import ActiveAnalysisBundle
from .statistics import _stats, _terminal_value, _direction, _safe_ratio, _stability
from .orbit_diagnostics import diagnostic_tables

def aggregate_active(cfg: DictConfig) -> ActiveAnalysisBundle:
    project = Path(str(cfg.project_root)).expanduser().resolve()
    root = project / str(cfg.output_root)
    plan = build_execution_plan(cfg, write=False)
    inventory = pd.DataFrame(plan.rows)
    key_columns = ["dataset", "model", "condition", "seed"]
    if inventory.duplicated(key_columns).any():
        raise ValueError("Duplicate scientific key in active analysis inventory")
    inventory["source_type"] = inventory["state"].map({"imported": "legacy_reused", "complete_native": "native",
                                                        "resumable_native": "native"}).fillna("none")
    inventory["valid"] = inventory["state"].isin(["imported", "complete_native"])
    terminals: list[dict[str, Any]] = []
    checkpoints: list[pd.DataFrame] = []
    classes: list[pd.DataFrame] = []
    imports: list[dict[str, Any]] = []
    for row in inventory.to_dict("records"):
        run = Path(row["run_directory"])
        if not row["valid"]:
            terminals.append({key: row[key] for key in (*key_columns, "state", "source_type", "run_directory")})
            continue
        terminal_path = run / "metrics" / "terminal_metrics.json"
        checkpoint_path = run / "metrics" / "checkpoint_metrics.csv"
        if not terminal_path.is_file() or not checkpoint_path.is_file():
            inventory.loc[inventory.run_directory == str(run), ["valid", "state"]] = [False, "incomplete_artifacts"]
            terminals.append({**{key: row[key] for key in (*key_columns, "source_type", "run_directory")},
                              "state": "incomplete_artifacts"})
            continue
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal.update({key: row[key] for key in (*key_columns, "source_type", "run_directory")})
        terminal["state"] = row["state"]
        terminal["metric_scope"] = "terminal_window"
        terminal.setdefault("D_total_bar", float(terminal.get("D_within_bar", math.nan)) + float(terminal.get("D_between_bar", math.nan)))
        if row["condition"] == "fully_symmetric":
            terminal.setdefault("D_global_bar", terminal.get("D_total_bar"))
        terminals.append(terminal)
        checkpoint = pd.read_csv(checkpoint_path)
        final_epoch = int(pd.to_numeric(checkpoint.epoch, errors="raise").max())
        terminal["final_checkpoint_epoch"] = final_epoch
        for key in ("dataset", "model", "condition", "seed", "source_type"):
            checkpoint[key] = row[key]
        checkpoints.append(checkpoint)
        orbit_path = run / "metrics" / "orbit_classification.csv"
        if orbit_path.is_file():
            frame = pd.read_csv(orbit_path)
            if row["condition"] == "fully_symmetric":
                node_path = run / "metrics" / "node_classification.csv"
                if node_path.is_file():
                    nodes = pd.read_csv(node_path)
                    frame = nodes.groupby(["class_id", "metric"], as_index=False)["value"].mean()
                    frame["orbit"] = pd.NA
                    frame["aggregation"] = "node_average_within_seed_global_diagnostic"
            for key in ("dataset", "model", "condition", "seed", "source_type"):
                frame[key] = row[key]
            frame["metric_scope"] = "final_checkpoint"
            frame["checkpoint_epoch"] = final_epoch
            manifest_path = project / str(cfg.manifest_root) / row["dataset"] / "heterogeneous.json"
            if manifest_path.is_file():
                names = json.loads(manifest_path.read_text(encoding="utf-8")).get("class_names", [])
            else:
                names = list(dataset_spec(row["dataset"]).class_names)
            frame["class_name"] = frame["class_id"].map(
                lambda value: "overall" if int(value) < 0 else (names[int(value)] if int(value) < len(names) else f"class_{int(value)}")
            )
            if row["condition"] == "fully_symmetric":
                frame["designed_orbit"] = pd.NA
                frame["designed_emphasis_group"] = False
            else:
                designed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 8: 4, 9: 4}
                frame["designed_orbit"] = frame["class_id"].map(designed)
                frame["designed_emphasis_group"] = frame.apply(
                    lambda value: value.get("orbit") == value.get("designed_orbit") if pd.notna(value.get("orbit")) else False, axis=1
                )
            classes.append(frame)
        reference = run / "legacy_reference.json"
        if reference.is_file():
            value = json.loads(reference.read_text(encoding="utf-8"))
            imports.append({**{key: row[key] for key in ("dataset", "model", "condition", "seed")},
                            "source_type": value.get("source_type"), "original_run_path": value.get("original_run_path"),
                            "compatibility_version": value.get("compatibility_version"),
                            "import_timestamp": value.get("import_timestamp"), "import_status": value.get("import_status")})
    terminal_frame = pd.DataFrame(terminals)
    inventory["expected_n"] = 1
    inventory["valid_n"] = inventory["valid"].astype(int)
    lambda_rows = terminal_frame[["dataset", "model", "condition", "seed"]].copy()
    lambda_rows["actual_lambda"] = (
        pd.to_numeric(terminal_frame["communication_lambda"], errors="coerce")
        if "communication_lambda" in terminal_frame else math.nan
    )
    inventory = inventory.merge(lambda_rows, on=["dataset", "model", "condition", "seed"], how="left")
    checkpoint_frame = pd.concat(checkpoints, ignore_index=True) if checkpoints else pd.DataFrame()
    class_frame = pd.concat(classes, ignore_index=True) if classes else pd.DataFrame(columns=[
        "orbit", "class_id", "class_name", "metric", "value", "dataset", "model", "condition", "seed", "source_type"
    ])
    metric_names = ("D_within_bar", "D_between_bar", "D_total_bar", "D_global_bar", "R_orbit_bar",
                    "mean_node_accuracy", "orbit_ensemble_accuracy", "raw_logit_rms", "centered_logit_rms")
    summaries: list[dict[str, Any]] = []
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            for condition in cfg.final.conditions:
                subset = terminal_frame[(terminal_frame.dataset == dataset) & (terminal_frame.model == model) &
                                        (terminal_frame.condition == condition)]
                for metric in metric_names:
                    seed_values = {
                        f"seed_{int(seed):03d}": _terminal_value(
                            terminal_frame, str(dataset), str(model), str(condition), int(seed), metric
                        ) for seed in cfg.final.seeds
                    }
                    summaries.append({"dataset": str(dataset), "model": str(model), "condition": str(condition),
                                      "metric": metric, **seed_values,
                                      **_stats(subset[metric] if metric in subset else pd.Series(dtype=float))})
    terminal_summary = pd.DataFrame(summaries)
    paired: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    relative: list[dict[str, Any]] = []
    threshold = float(cfg.training.ratio_threshold)
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            for seed in cfg.final.seeds:
                dw_main = _terminal_value(terminal_frame, dataset, model, "main", seed, "D_within_bar")
                db_main = _terminal_value(terminal_frame, dataset, model, "main", seed, "D_between_bar")
                dt_main = _terminal_value(terminal_frame, dataset, model, "main", seed, "D_total_bar")
                dw_no = _terminal_value(terminal_frame, dataset, model, "no_communication", seed, "D_within_bar")
                dw_misaligned = _terminal_value(terminal_frame, dataset, model, "misaligned_data", seed, "D_within_bar")
                dg_symmetric = _terminal_value(terminal_frame, dataset, model, "fully_symmetric", seed, "D_global_bar")
                values = {
                    "A_between_minus_within_main": db_main - dw_main,
                    "B_no_communication_within_minus_main": dw_no - dw_main,
                    "C_misaligned_within_minus_main": dw_misaligned - dw_main,
                    "D_main_total_minus_fully_symmetric_global": dt_main - dg_symmetric,
                }
                for check, value in values.items():
                    paired.append({"dataset": dataset, "model": model, "seed": int(seed), "check": check,
                                   "value": value, "direction": _direction(value)})
                    consistency.append({"dataset": dataset, "model": model, "seed": int(seed), "check": check,
                                        "direction": _direction(value),
                                        "positive": _direction(value) == "positive" if math.isfinite(value) else pd.NA})
                ratios = {
                    "separation_ratio": _safe_ratio(db_main, dw_main, threshold),
                    "communication_reduction": 1.0 - _safe_ratio(dw_main, dw_no, threshold),
                    "compatibility_reduction": 1.0 - _safe_ratio(dw_main, dw_misaligned, threshold),
                    "global_symmetry_ratio": _safe_ratio(dg_symmetric, dt_main, threshold),
                    "R_orbit_main": _safe_ratio(db_main, dt_main, threshold),
                }
                for metric, value in ratios.items():
                    relative.append({"dataset": dataset, "model": model, "seed": int(seed), "metric": metric,
                                     "value": value, "aggregation_note": "seed_level_ratio"})
    paired_frame = pd.DataFrame(paired)
    paired_summary = []
    for (dataset, model, check), group in paired_frame.groupby(["dataset", "model", "check"], sort=False):
        statistics = _stats(group.value, expected=len(cfg.final.seeds))
        paired_summary.append({"dataset": dataset, "model": model, "check": check,
                               **statistics, "paired_n": statistics["valid_n"]})
    consistency_frame = pd.DataFrame(consistency)
    consistency_summary: list[dict[str, Any]] = []
    for (dataset, model, check), group in consistency_frame.groupby(["dataset", "model", "check"], sort=False):
        counts = group.direction.value_counts()
        consistency_summary.append({"dataset": dataset, "model": model, "check": check,
                                    **{name: int(counts.get(name, 0)) for name in ("positive", "negative", "tied", "undefined")},
                                    "valid_n": int(group.direction.ne("undefined").sum()),
                                    "paired_n": int(group.direction.ne("undefined").sum()),
                                    "expected_n": len(cfg.final.seeds)})
    stability: list[dict[str, Any]] = []
    main = checkpoint_frame[checkpoint_frame.condition == "main"] if not checkpoint_frame.empty else checkpoint_frame
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            for seed in cfg.final.seeds:
                group = main[(main.dataset == dataset) & (main.model == model) & (main.seed == seed)] if not main.empty else main
                for metric in ("D_within", "D_between", "R_orbit"):
                    stability.append(_stability(group, str(dataset), str(model), int(seed), metric, threshold,
                                                expected_epochs=(180, 185, 190, 195, 200)))
    pilots: list[dict[str, Any]] = []
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            family = root / "pilot" / str(dataset) / str(model)
            selection = family / "selected_hparams.yaml"
            source_type, status = "missing", "missing"
            lambda_main = math.nan
            if selection.is_file():
                import yaml
                value = yaml.safe_load(selection.read_text(encoding="utf-8")) or {}
                source_type, status = value.get("source_type", "native_pilot"), value.get("selection_status", "unknown")
                lambda_main = value.get("lambda_main", math.nan)
            summary_path = family / "pilot_summary.csv"
            candidate_rows = pd.read_csv(summary_path).to_dict("records") if summary_path.is_file() else [{}]
            for candidate in candidate_rows:
                pilots.append({"dataset": dataset, "model": model, "source_type": source_type,
                               "selection_status": status, "lambda_main": lambda_main,
                               "conceptual_candidates": 4,
                               "candidate_runs_imported": 0 if source_type == "legacy_selected_value" else math.nan,
                               **candidate})
    return ActiveAnalysisBundle(
        inventory, terminal_frame, terminal_summary, paired_frame, pd.DataFrame(paired_summary), consistency_frame,
        pd.DataFrame(consistency_summary), pd.DataFrame(relative),
        pd.DataFrame(stability, columns=["dataset", "model", "seed", "metric", "status", "mean", "sample_sd",
                                          "cv", "slope", "normalized_drift", "relative_range", "evaluations"]), class_frame,
        pd.DataFrame(pilots),
        pd.DataFrame(imports, columns=["dataset", "model", "condition", "seed", "source_type", "original_run_path",
                                       "compatibility_version", "import_timestamp", "import_status"]), checkpoint_frame,
        **diagnostic_tables(inventory),
    )


