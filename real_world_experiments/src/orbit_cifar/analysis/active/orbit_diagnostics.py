"""Seed-level finite-time geometry/motion; legacy derivations are report-only."""

import json
from pathlib import Path

import pandas as pd
import torch

from orbit_cifar.constants import REFERENCE_ORBITS
from orbit_cifar.evaluation.orbit_diagnostics import PROFILE, SCHEMA_VERSION, geometry
from orbit_cifar.experiments.orbit_diagnostics import COLUMNS, GEOMETRY, MOTION, STATUS, coverage, file_hash
from orbit_cifar.utils.io import ensure_directory

PROVENANCE = ["source_type", "source_path", "source_sha256"]
COVERAGE_COLUMNS = ["schema_version", "dataset", "model", "condition", "seed", "previous_epoch", "epoch",
                    "scope", "status", "temporal_coverage", "missing_epochs", "source_type"]


def diagnostic_tables(inventory):
    geometries, movements, terminal, temporal = [], [], [], []
    for run in inventory.to_dict("records"):
        root = Path(run["run_directory"])
        key = {name: run[name] for name in ("dataset", "model", "condition", "seed")}
        observed = coverage({})
        if run["valid"]:
            status_path = root / STATUS
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("schema_version") != SCHEMA_VERSION or status.get("profile") != PROFILE:
                    raise ValueError("Unsupported report diagnostic schema")
                observed = status
                for name, destination in ((GEOMETRY, geometries), (MOTION, movements)):
                    frame = pd.read_csv(root / name)
                    frame["source_type"] = "native_recorded"
                    frame["source_path"], frame["source_sha256"] = str(root / name), file_hash(root / name)
                    destination.append(frame)
                    if name == GEOMETRY:
                        terminal.append(frame[frame.scope == "final_checkpoint"].copy())
            else:
                # Never write to imported/native source files. No historical
                # temporal predictions can be inferred from this final array.
                raw_path = root / "artifacts/final_raw_logits.pt"
                if raw_path.is_file():
                    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
                    records = geometry(raw, REFERENCE_ORBITS, strict=True)
                    final_epoch = int(pd.read_csv(root / "metrics/checkpoint_metrics.csv").epoch.max())
                    frame = pd.DataFrame([{**dict.fromkeys(COLUMNS), **key, "schema_version": SCHEMA_VERSION,
                                           "epoch": final_epoch, "scope": "posthoc_reference_partition_final_checkpoint",
                                           **item} for item in records])
                    frame["source_type"] = "posthoc_" + run["source_type"]
                    frame["source_path"], frame["source_sha256"] = str(raw_path), file_hash(raw_path)
                    terminal.append(frame)
        if run["condition"] == "main":
            for pair in observed["pairs"] + [{**observed["endpoint"], "endpoint": True}]:
                temporal.append({**key, "schema_version": SCHEMA_VERSION, "previous_epoch": pair["previous_epoch"],
                                 "epoch": pair["epoch"], "scope": "endpoint_displacement" if pair.get("endpoint") else "adjacent_transition",
                                 "status": pair["status"] if run["valid"] else "run_unavailable",
                                 "temporal_coverage": observed["temporal_coverage"],
                                 "missing_epochs": json.dumps(pair.get("missing_epochs", [e for e in (175, 200)
                                                                  if e not in observed["retained_epochs"]])),
                                 "source_type": run["source_type"]})
    def combine(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + PROVENANCE)
    movement = combine(movements)
    return {"orbit_geometry_per_seed": combine(geometries), "predictive_motion_per_seed": movement,
            "endpoint_displacement_per_seed": movement[movement.scope == "endpoint_displacement"].copy(),
            "terminal_orbit_geometry": combine(terminal),
            "temporal_coverage": pd.DataFrame(temporal, columns=COVERAGE_COLUMNS)}


def diagnostic_figures(bundle, cfg, reports):
    from .figures import plt, _save

    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            fig, axes = plt.subplots(3, 5, figsize=(22, 11), squeeze=False)
            for row, seed in enumerate(cfg.final.seeds):
                for col, (table, metrics, title) in enumerate((
                    (bundle.orbit_geometry_per_seed, ["W_k"], "Orbit radius"),
                    (bundle.orbit_geometry_per_seed, ["B_kl"], "Center-pair separation"),
                    (bundle.predictive_motion_per_seed, ["T_center_k"], "Center movement"),
                    (bundle.predictive_motion_per_seed, ["T_center_relative_k"], "Relative center movement"),
                    (bundle.predictive_motion_per_seed, ["T_node_max", "T_node_mean", "T_global"], "Node/global movement"),
                )):
                    axis = axes[row, col]
                    values = table[(table.dataset == dataset) & (table.model == model) & (table.condition == "main")
                                   & (table.seed == seed) & table.metric.isin(metrics)
                                   & (table.scope != "endpoint_displacement")]
                    axis.set_title(f"Seed {seed}: {title}")
                    axis.set_ylabel("squared distance / (P C)")
                    if values.empty:
                        axis.text(.5, .5, "Not recorded / unavailable", transform=axis.transAxes, ha="center")
                        continue
                    for (metric, group, pair), points in values.groupby(["metric", "group", "pair_group"], dropna=False):
                        label = metric
                        if pd.notna(group):
                            label += f" {int(group)}"
                        if pd.notna(pair):
                            label += f",{int(pair)}"
                        # NaN at absent scheduled transitions prevents silently
                        # bridging a missing exact predecessor in the plot.
                        epochs = [180, 185, 190, 195, 200] if col >= 2 else list(range(0, 201, 5))
                        curve = points.set_index("epoch").value.reindex(epochs)
                        axis.plot(epochs, curve, marker=".", label=label)
                    if col >= 2:
                        axis.set_xticks([180, 185, 190, 195, 200], ["175-180", "180-185", "185-190", "190-195", "195-200"], rotation=45)
                    axis.set_xlabel("epoch pair" if col >= 2 else "epoch")
                    axis.legend(fontsize=6)
            fig.suptitle(f"{dataset}/{model}: Main finite-time geometry and observed late movement")
            fig.tight_layout()
            directory = ensure_directory(reports / "figures" / str(dataset) / str(model))
            _save(fig, directory, "orbit_geometry_and_motion_per_seed", int(cfg.analysis.figure_dpi))
