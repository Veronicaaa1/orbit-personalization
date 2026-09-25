"""Active family figures; unavailable evidence is never rendered as zero."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from orbit_cifar.utils.io import ensure_directory
from orbit_cifar.data.datasets import dataset_spec
from .contracts import ActiveAnalysisBundle

def _save(fig: plt.Figure, directory: Path, name: str, dpi: int) -> None:
    fig.savefig(directory / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _empty(title: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(.5, .5, "Unavailable: no valid observations", ha="center", va="center")
    return fig


def _directional_data(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep unobserved comparisons masked and distinguish 0/3 from 0/0."""
    positives = summary.pivot(index=["dataset", "model"], columns="check", values="positive")
    counts = summary.pivot(index=["dataset", "model"], columns="check", values="valid_n")
    return positives.where(counts > 0), counts


def _family_figures(bundle: ActiveAnalysisBundle, cfg: DictConfig, reports: Path, dpi: int) -> None:
    figure_root = ensure_directory(reports / "figures")
    per_seed = ensure_directory(reports / "per_seed")
    for dataset in cfg.final.datasets:
        for model in cfg.final.models:
            directory = ensure_directory(figure_root / str(dataset) / str(model))
            terminal = bundle.per_seed_terminal_metrics
            family = terminal[(terminal.dataset == dataset) & (terminal.model == model)]
            valid = family[family.state.isin(["imported", "complete_native"])]
            if valid.empty:
                for name in ("terminal_within_between", "main_dynamics", "main_pairwise_heatmap",
                             "main_vs_fully_symmetric_global", "class_conditioned_accuracy_ce"):
                    _save(_empty(f"{dataset}/{model}: {name}"), directory, name, dpi)
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for axis, metric in zip(axes, ("D_within_bar", "D_between_bar")):
                for index, condition in enumerate(("main", "no_communication", "misaligned_data")):
                    rows = valid[valid.condition == condition]
                    if metric in rows:
                        axis.scatter(np.full(len(rows), index), rows[metric], label=condition if metric == "D_within_bar" else None)
                axis.set_xticks(range(3), ["Main", "No comm.", "Misaligned"], rotation=25)
                axis.set_title(metric)
            axes[0].legend(fontsize=8)
            _save(fig, directory, "terminal_within_between", dpi)
            checkpoints = bundle.checkpoint_rows
            main = checkpoints[(checkpoints.dataset == dataset) & (checkpoints.model == model) & (checkpoints.condition == "main")] if not checkpoints.empty else checkpoints
            if main.empty:
                _save(_empty(f"{dataset}/{model}: Main dynamics"), directory, "main_dynamics", dpi)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for seed, rows in main.groupby("seed"):
                    axes[0].plot(rows.epoch, rows.D_within, alpha=.7, label=f"seed {seed}")
                    axes[1].plot(rows.epoch, rows.D_between, alpha=.7)
                mean_curve = main.groupby("epoch", as_index=False)[["D_within", "D_between"]].mean()
                axes[0].plot(mean_curve.epoch, mean_curve.D_within, color="black", linewidth=2, label="seed mean")
                axes[1].plot(mean_curve.epoch, mean_curve.D_between, color="black", linewidth=2)
                axes[0].set_title("D_within: individual trajectories")
                axes[1].set_title("D_between: individual trajectories")
                axes[0].legend(fontsize=8)
                _save(fig, directory, "main_dynamics", dpi)
            matrices = []
            for row in valid[valid.condition == "main"].to_dict("records"):
                path = Path(row["run_directory"]) / "artifacts" / "pairwise_distance.npy"
                if path.is_file():
                    matrix = np.load(path)
                    matrices.append(matrix)
                    fig, axis = plt.subplots(figsize=(5, 4.5)); image = axis.imshow(matrix, cmap="viridis")
                    fig.colorbar(image, ax=axis); axis.set_title(f"{dataset}/{model} Main seed {row['seed']} final checkpoint")
                    _save(fig, per_seed, f"{dataset}_{model}_main_seed_{int(row['seed']):03d}_pairwise", dpi)
            if matrices:
                fig, axis = plt.subplots(figsize=(5, 4.5)); image = axis.imshow(np.mean(matrices, axis=0), cmap="viridis")
                fig.colorbar(image, ax=axis); axis.set_title(f"{dataset}/{model}: mean final Main pairwise")
                _save(fig, directory, "main_pairwise_heatmap", dpi)
            else:
                _save(_empty(f"{dataset}/{model}: Main pairwise"), directory, "main_pairwise_heatmap", dpi)
            fig, axis = plt.subplots(figsize=(6, 4))
            plotted = False
            for index, (condition, metric) in enumerate((("main", "D_total_bar"), ("fully_symmetric", "D_global_bar"))):
                rows = valid[valid.condition == condition]
                if metric in rows:
                    axis.scatter(np.full(len(rows), index), rows[metric]); plotted = plotted or rows[metric].notna().any()
            if plotted:
                axis.set_xticks([0, 1], ["Main D_total", "Fully symmetric D_global"])
                axis.set_title("Per-seed terminal-window values")
                _save(fig, directory, "main_vs_fully_symmetric_global", dpi)
            else:
                plt.close(fig); _save(_empty(f"{dataset}/{model}: global comparison"), directory, "main_vs_fully_symmetric_global", dpi)
            classes = bundle.classwise_performance
            rows = classes[(classes.dataset == dataset) & (classes.model == model) & (classes.condition == "main")] if not classes.empty else classes
            if rows.empty:
                _save(_empty(f"{dataset}/{model}: class-conditioned accuracy/CE"), directory, "class_conditioned_accuracy_ce", dpi)
            else:
                max_class = dataset_spec(str(dataset)).num_classes
                rows = rows[rows.class_id >= 0]
                table = rows.groupby(["orbit", "class_id", "metric"])["value"].mean().reset_index()
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                axis_names = list(dataset_spec(str(dataset)).class_names)
                metrics = ("class_accuracy", "class_cross_entropy")
                for axis, metric in zip(axes, metrics):
                    pivot = table[table.metric == metric].pivot(index="orbit", columns="class_id", values="value").reindex(index=range(1, 5), columns=range(max_class))
                    axis.imshow(np.ma.masked_invalid(pivot.to_numpy(float)), aspect="auto", cmap="viridis")
                    axis.set_title(f"{metric}: final checkpoint (available-seed mean)")
                    axis.set_xticks(range(max_class), axis_names, rotation=45, ha="right", fontsize=7)
                    axis.set_yticks(range(4), ["O1", "O2", "O3", "O4"])
                _save(fig, directory, "class_conditioned_accuracy_ce", dpi)
    summary = bundle.robustness_consistency_summary
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if summary.empty or not summary.valid_n.gt(0).any():
        plt.close(fig); fig = _empty("4x4 directional consistency")
    else:
        pivot, counts = _directional_data(summary)
        image = axis.imshow(np.ma.masked_invalid(pivot.to_numpy(float)), vmin=0, vmax=3, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=axis, label="positive seeds; labels show positive/valid (expected 3)")
        for row_index in range(len(pivot)):
            for column_index in range(len(pivot.columns)):
                valid_n = int(counts.iloc[row_index, column_index])
                label = f"{int(pivot.iloc[row_index, column_index])}/{valid_n}" if valid_n else "N/A"
                axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
        axis.set_yticks(range(len(pivot)), [f"{a}/{b}" for a, b in pivot.index]); axis.set_xticks(range(len(pivot.columns)), ["A", "B", "C", "D"])
        axis.set_title("4x4 exploratory directional consistency")
    _save(fig, figure_root, "directional_consistency_4x4", dpi)
    relative = bundle.relative_robustness_metrics
    fig, axis = plt.subplots(figsize=(10, 4.5))
    if relative.empty or not np.isfinite(pd.to_numeric(relative.value, errors="coerce")).any():
        plt.close(fig); fig = _empty("Relative contrasts across four families")
    else:
        labels = []
        for index, ((dataset, model), rows) in enumerate(relative.groupby(["dataset", "model"], sort=False)):
            subset = rows[rows.metric == "separation_ratio"]
            axis.scatter(np.full(len(subset), index), subset.value); labels.append(f"{dataset}\n{model}")
        axis.set_xticks(range(len(labels)), labels, rotation=25); axis.set_title("Seed-level separation ratios; no cross-family pooling")
    _save(fig, figure_root, "relative_contrasts_across_families", dpi)


