from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbit_cifar.analysis.legacy_aggregate import AnalysisBundle
from orbit_cifar.constants import CIFAR10_CLASSES
from orbit_cifar.utils.io import ensure_directory


def _save(figure: plt.Figure, reports: Path, name: str, dpi: int) -> None:
    figure.savefig(reports / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(reports / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def _empty_figure(title: str, message: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return figure


def _curve(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    frame = bundle.time_series
    main = frame[frame["condition"] == "main"] if not frame.empty else frame
    if main.empty:
        _save(_empty_figure("Main dispersion", "No valid main runs"), reports, "dispersion_curves", dpi)
        _save(_empty_figure("Orbit alignment ratio", "No valid main runs"), reports, "orbit_ratio_curve", dpi)
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for metric, label in (("D_within", "Within-orbit dispersion"), ("D_between", "Between-orbit dispersion")):
        rows = main[main["metric"] == metric].sort_values("epoch")
        x, mean = rows["epoch"].to_numpy(), rows["mean"].to_numpy()
        se = rows["se"].fillna(0).to_numpy()
        axis.plot(x, mean, label=label)
        axis.fill_between(x, mean - se, mean + se, alpha=0.2)
    valid = int(main["valid_seeds"].max())
    axis.set(xlabel="Epoch", ylabel="Mean squared centered-logit dispersion", title=f"Main condition (valid seeds: {valid})")
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, reports, "dispersion_curves", dpi)
    ratio = main[main["metric"] == "R_orbit"].sort_values("epoch")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x, mean = ratio["epoch"].to_numpy(), ratio["mean"].to_numpy()
    se = ratio["se"].fillna(0).to_numpy()
    axis.plot(x, mean, color="tab:purple", label="R_orbit")
    axis.fill_between(x, mean - se, mean + se, color="tab:purple", alpha=0.2)
    axis.set(xlabel="Epoch", ylabel="Orbit alignment ratio", title=f"Main condition (valid seeds: {valid}); interpret with raw dispersions")
    axis.set_ylim(0, 1.02)
    axis.legend()
    axis.grid(alpha=0.25)
    _save(figure, reports, "orbit_ratio_curve", dpi)


def _pairwise(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    if bundle.pairwise_mean is None:
        figure = _empty_figure("Pairwise centered-logit distance", "No valid main pairwise matrices")
    else:
        figure, axis = plt.subplots(figsize=(6.2, 5.4))
        image = axis.imshow(bundle.pairwise_mean, cmap="viridis", origin="upper")
        figure.colorbar(image, ax=axis, label="Squared centered-logit distance")
        axis.set_xticks(range(14), labels=range(1, 15))
        axis.set_yticks(range(14), labels=range(1, 15))
        valid = int(bundle.run_inventory[(bundle.run_inventory["condition"] == "main") & bundle.run_inventory["valid"]].shape[0])
        axis.set(xlabel="Node label", ylabel="Node label", title=f"Main: across-seed mean final pairwise distance (n={valid})")
        for boundary in (1.5, 4.5, 8.5):
            axis.axhline(boundary, color="white", linewidth=1.2)
            axis.axvline(boundary, color="white", linewidth=1.2)
    _save(figure, reports, "pairwise_distance_heatmap", dpi)


def _fully_symmetric_pairwise(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    matrix = bundle.fully_symmetric_pairwise_mean
    if matrix is None:
        figure = _empty_figure(
            "Fully symmetric pairwise centered-logit distance",
            "No valid fully_symmetric pairwise matrices",
        )
    else:
        figure, axis = plt.subplots(figsize=(6.2, 5.4))
        image = axis.imshow(matrix, cmap="viridis", origin="upper")
        figure.colorbar(image, ax=axis, label="Squared centered-logit distance")
        axis.set_xticks(range(14), labels=range(1, 15))
        axis.set_yticks(range(14), labels=range(1, 15))
        valid = int(
            bundle.run_inventory[
                (bundle.run_inventory["condition"] == "fully_symmetric") & bundle.run_inventory["valid"]
            ].shape[0]
        )
        axis.set(
            xlabel="Node label",
            ylabel="Node label",
            title=f"Fully symmetric K14: final pairwise distance (n={valid}); one global orbit",
        )
        # Deliberately omit O1--O4 boundaries: those groups are a reference
        # partition under K14, not automorphism orbits.
    _save(figure, reports, "fully_symmetric_pairwise_heatmap", dpi)


def _global_disagreement(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    frame = bundle.terminal_summary
    desired = ["fully_symmetric", "complete_graph", "main", "no_communication"]
    rows = frame[frame["metric"] == "D_total_bar"] if not frame.empty else frame
    if not rows.empty:
        rows = rows.set_index("condition").reindex(desired).dropna(subset=["mean"])
    if rows.empty:
        _save(
            _empty_figure("Global predictive disagreement", "No valid terminal D_total metrics"),
            reports,
            "global_disagreement_baseline",
            dpi,
        )
        return
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(range(len(rows)), rows["mean"], yerr=rows["se"].fillna(0), capsize=3)
    axis.set_xticks(range(len(rows)), labels=rows.index, rotation=30, ha="right")
    axis.set(
        ylabel="Terminal D_total (= D_global for fully_symmetric)",
        title="Global predictive disagreement control; seed-level mean ±1 SE",
    )
    axis.grid(axis="y", alpha=0.25)
    _save(figure, reports, "global_disagreement_baseline", dpi)


def _fully_symmetric_class_profile(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    frame = bundle.node_classification
    rows = (
        frame[
            (frame["condition"] == "fully_symmetric")
            & (frame["metric"] == "class_accuracy")
            & (frame["class_id"] >= 0)
        ]
        if not frame.empty
        else frame
    )
    if rows.empty:
        _save(
            _empty_figure("Fully symmetric class accuracy", "No valid fully_symmetric node classification data"),
            reports,
            "fully_symmetric_class_accuracy",
            dpi,
        )
        return
    per_seed = rows.groupby(["seed", "class_id"], as_index=False)["value"].mean()
    summary = per_seed.groupby("class_id")["value"].agg(["mean", "count", "std"]).reindex(range(10))
    errors = (summary["std"] / np.sqrt(summary["count"])).fillna(0)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.errorbar(range(10), summary["mean"], yerr=errors, marker="o", capsize=3)
    axis.set_xticks(range(10), labels=CIFAR10_CLASSES, rotation=35, ha="right")
    axis.set(
        xlabel="CIFAR-10 class",
        ylabel="Node-averaged accuracy",
        title="Fully symmetric baseline: class profile (no designed reference-group specialization)",
    )
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    _save(figure, reports, "fully_symmetric_class_accuracy", dpi)


def _controls(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    frame = bundle.terminal_summary
    if frame.empty:
        _save(_empty_figure("Terminal controls", "No valid terminal metrics"), reports, "control_terminal_metrics", dpi)
        return
    conditions = frame["condition"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, metric, title in zip(
        axes,
        ("D_within_bar", "D_between_bar", "R_orbit_bar"),
        (
            "Fixed-partition within dispersion",
            "Fixed-partition between dispersion",
            "Fixed-partition ratio",
        ),
    ):
        rows = frame[frame["metric"] == metric].set_index("condition").reindex(conditions)
        axis.bar(range(len(rows)), rows["mean"], yerr=rows["se"].fillna(0), capsize=3)
        axis.set_xticks(range(len(rows)), labels=conditions, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Terminal-window mean")
        axis.grid(axis="y", alpha=0.25)
    counts = bundle.run_inventory[bundle.run_inventory["valid"]].groupby("condition").size().to_dict()
    count_label = ", ".join(f"{name}:n={count}" for name, count in counts.items())
    figure.suptitle(f"Condition comparison; seed-level ±1 SE ({count_label})")
    _save(figure, reports, "control_terminal_metrics", dpi)


def _class_heatmap(bundle: AnalysisBundle, reports: Path, dpi: int, metric: str, name: str, label: str) -> None:
    frame = bundle.orbit_classification
    main = frame[(frame["condition"] == "main") & (frame["metric"] == metric) & (frame["class_id"] >= 0)] if not frame.empty else frame
    if main.empty:
        _save(_empty_figure(label, "No valid main orbit classification data"), reports, name, dpi)
        return
    table = main.groupby(["orbit", "class_id"])["value"].mean().unstack("class_id").reindex(index=range(1, 5), columns=range(10))
    figure, axis = plt.subplots(figsize=(10, 3.6))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="magma" if metric == "class_cross_entropy" else "viridis")
    figure.colorbar(image, ax=axis, label=label)
    axis.set_xticks(range(10), labels=CIFAR10_CLASSES, rotation=35, ha="right")
    axis.set_yticks(range(4), labels=["O1", "O2", "O3", "O4"])
    valid = int(main["seed"].nunique())
    axis.set(xlabel="CIFAR-10 class", ylabel="Reference orbit", title=f"Main orbit logit ensembles: {label} (n={valid} seeds)")
    _save(figure, reports, name, dpi)


def _lambda(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    frame = bundle.pilot_summary
    if frame.empty:
        _save(_empty_figure("Lambda validation sweep", "Pilot summary is unavailable"), reports, "lambda_sweep", dpi)
        return
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, label in zip(
        axes,
        ("D_within", "D_between", "orbit_ensemble_accuracy"),
        ("Within dispersion", "Between dispersion", "Orbit ensemble validation accuracy"),
    ):
        axis.plot(frame["lambda"], frame[metric], marker="o")
        axis.set_xscale("symlog", linthresh=0.005)
        axis.set(xlabel="Communication lambda", ylabel=label)
        axis.grid(alpha=0.25)
    figure.suptitle("Balanced-validation pilot only")
    _save(figure, reports, "lambda_sweep", dpi)


def _random_partition(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    diagnostic = bundle.random_partition
    values = np.asarray(diagnostic.get("R_random_mean", []), dtype=float)
    if values.size == 0:
        _save(_empty_figure("Random-partition diagnostic", "No valid main diagnostic artifacts"), reports, "random_partition_diagnostic", dpi)
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(values[np.isfinite(values)], bins=50, alpha=0.75, label="Matched-size random partitions")
    true = float(diagnostic["R_true_mean"])
    axis.axvline(true, color="tab:red", linewidth=2, label=f"Reference partition R={true:.3g}")
    axis.set(
        xlabel="Orbit alignment ratio",
        ylabel="Random partition count",
        title=f"Descriptive diagnostic; aggregate p={diagnostic['p_perm_aggregate']:.3g}, valid seeds={diagnostic['valid_seeds']}",
    )
    axis.legend()
    _save(figure, reports, "random_partition_diagnostic", dpi)


def _appendix(bundle: AnalysisBundle, reports: Path, dpi: int) -> None:
    appendix = ensure_directory(reports / "appendix")
    frame = bundle.checkpoint_rows
    if not frame.empty:
        figure, axis = plt.subplots(figsize=(8, 5))
        for (condition, seed), group in frame.groupby(["condition", "seed"]):
            axis.plot(group["epoch"], group["D_within"], alpha=0.45, label=f"{condition}/s{seed}")
        axis.set(xlabel="Epoch", ylabel="D_within", title="Per-seed within-orbit curves")
        if frame[["condition", "seed"]].drop_duplicates().shape[0] <= 15:
            axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        _save(figure, appendix, "per_seed_curves", dpi)
    terminal = bundle.terminal_rows
    if not terminal.empty and "mean_node_accuracy" in terminal:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for index, (condition, group) in enumerate(terminal.groupby("condition", sort=False)):
            axis.scatter(np.full(len(group), index), group["mean_node_accuracy"], alpha=0.75)
        labels = list(terminal.groupby("condition", sort=False).groups)
        axis.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
        axis.set(ylabel="Mean node test accuracy", title="Per-seed overall accuracy")
        _save(figure, appendix, "overall_accuracy", dpi)
    orbit_confusions = [(key, value) for key, value in sorted(bundle.confusion_means.items()) if key.startswith("orbit_")]
    if orbit_confusions:
        figure, axes = plt.subplots(2, 2, figsize=(9, 8), squeeze=False)
        for axis, (key, matrix) in zip(axes.flat, orbit_confusions):
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
            image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            axis.set(title=key.replace("_", " ").title(), xlabel="Predicted class", ylabel="True class")
        figure.colorbar(image, ax=axes.ravel().tolist(), label="Mean row-normalized count", shrink=0.75)
        figure.suptitle("Main orbit-logit-ensemble confusion matrices")
        _save(figure, appendix, "orbit_confusion_matrices", dpi)
    node_confusions = [(key, value) for key, value in sorted(bundle.confusion_means.items()) if key.startswith("node_")]
    if node_confusions:
        figure, axes = plt.subplots(3, 5, figsize=(13, 8), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (key, matrix) in zip(axes.flat, node_confusions):
            axis.axis("on")
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
            axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            axis.set_title(key.replace("_", " ").title(), fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle("Main per-node confusion matrices (row normalized)")
        _save(figure, appendix, "node_confusion_matrices", dpi)


def generate_figures(bundle: AnalysisBundle, reports_directory: str | Path, dpi: int = 180) -> None:
    """Generate every required figure, annotating absent data rather than inventing runs."""
    reports = ensure_directory(reports_directory)
    _curve(bundle, reports, dpi)
    _pairwise(bundle, reports, dpi)
    _fully_symmetric_pairwise(bundle, reports, dpi)
    _global_disagreement(bundle, reports, dpi)
    _fully_symmetric_class_profile(bundle, reports, dpi)
    _controls(bundle, reports, dpi)
    _class_heatmap(bundle, reports, dpi, "class_accuracy", "orbit_class_accuracy_heatmap", "Class accuracy")
    _class_heatmap(bundle, reports, dpi, "class_cross_entropy", "orbit_class_ce_heatmap", "Cross-entropy")
    _lambda(bundle, reports, dpi)
    _random_partition(bundle, reports, dpi)
    _appendix(bundle, reports, dpi)
