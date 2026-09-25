"""Synthetic analysis contracts; all fixture and report files live in tmp_path."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from orbit_cifar.analysis.active import aggregate_active, analyze_active
from orbit_cifar.analysis.active.figures import _directional_data, _empty, _family_figures
from orbit_cifar.analysis.active.statistics import _safe_ratio, _stability, _stats
from orbit_cifar.analysis.serialization import _latex_escape
from orbit_cifar.config_schema import load_config
from orbit_cifar.experiments.matrix import final_keys


def _config(tmp_path: Path):
    config_path = Path(__file__).resolve().parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml"
    cfg = load_config(config_path)
    cfg.project_root = str(tmp_path)
    cfg.output_root = "observations"
    cfg.report_root = "reports"
    cfg.manifest_root = "manifests"
    cfg.analysis.figure_dpi = 24
    return cfg


def _observations(cfg, present, monkeypatch):
    """Model the verified planner boundary with production-shaped report artifacts.

    Native/checkpoint integrity verification has separate planner tests. These
    tests isolate reporting so they never fabricate a scientific checkpoint or
    launch training to create a fixture.
    """
    rows = []
    for key in final_keys(cfg):
        run = Path(cfg.project_root) / cfg.output_root / key.dataset / key.model / key.condition / f"seed_{key.seed:03d}"
        available = present is None or (key.dataset, key.model, key.condition, key.seed) in present
        rows.append({**key.to_dict(), "run_directory": str(run),
                     "state": "complete_native" if available else "awaiting_data", "reason": "synthetic test"})
        if not available:
            continue
        (run / "metrics").mkdir(parents=True)
        (run / "artifacts").mkdir()
        scale = {"main": 1., "no_communication": 2., "misaligned_data": 3., "fully_symmetric": .1}[key.condition]
        within = scale * key.seed
        checkpoints = pd.DataFrame([
            {"epoch": epoch, "D_within": within * factor, "D_between": 2 * key.seed * factor,
             "D_total": (within + 2 * key.seed) * factor, "R_orbit": 2 / (scale + 2),
             "mean_node_accuracy": .4 + .01 * j, "orbit_ensemble_accuracy": .5 + .01 * j}
            for j, (epoch, factor) in enumerate(zip((180, 185, 190, 195, 200), (.98, .99, 1., 1.01, 1.02)))
        ])
        checkpoints.to_csv(run / "metrics/checkpoint_metrics.csv", index=False)
        terminal = {f"{metric}_bar": float(checkpoints[metric].mean())
                    for metric in ("D_within", "D_between", "D_total", "R_orbit")}
        terminal.update(mean_node_accuracy=.42, orbit_ensemble_accuracy=.52,
                        final_mean_node_accuracy=.44, final_orbit_ensemble_accuracy=.54,
                        communication_lambda=0. if key.condition == "no_communication" else .1,
                        raw_logit_rms=1., centered_logit_rms=.8)
        if key.condition == "fully_symmetric":
            terminal["D_global_bar"] = terminal["D_total_bar"]
            terminal["true_automorphism_orbits"] = [list(range(14))]
            terminal["partition_semantics"] = "reference_partition_only"
        (run / "metrics/terminal_metrics.json").write_text(json.dumps(terminal), encoding="utf-8")
        # Pairwise artifact intentionally differs from terminal scalars.
        matrix = np.abs(np.subtract.outer(np.arange(14), np.arange(14))).astype(float) * key.seed / 100
        np.save(run / "artifacts/pairwise_distance.npy", matrix)
        class_count = 10
        class_rows = []
        for orbit in range(1, 5):
            # Insertion order used to select overall_accuracy and omit class CE.
            class_rows.append({"orbit": orbit, "class_id": -1, "metric": "overall_accuracy", "value": .54})
            for class_id in range(class_count):
                for metric, value in (("class_accuracy", .4 + key.seed / 100 + orbit / 100),
                                      ("class_cross_entropy", 1.5 + class_id / 100)):
                    class_rows.append({"orbit": orbit, "class_id": class_id, "metric": metric, "value": value})
        pd.DataFrame(class_rows).to_csv(run / "metrics/orbit_classification.csv", index=False)
        if key.condition == "fully_symmetric":
            node_rows = [{"node": node, "class_id": class_id, "metric": metric, "value": value}
                         for node in range(14) for class_id in range(class_count)
                         for metric, value in (("class_accuracy", .44), ("class_cross_entropy", 1.5))]
            pd.DataFrame(node_rows).to_csv(run / "metrics/node_classification.csv", index=False)
    aggregate_module = importlib.import_module("orbit_cifar.analysis.active.aggregate")
    monkeypatch.setattr(aggregate_module, "build_execution_plan", lambda cfg, write=False: SimpleNamespace(rows=rows))
    return rows


@pytest.mark.parametrize("complete", [False, True], ids=["one-main-partial", "full-48-cells"])
def test_nonempty_active_reporting(tmp_path, monkeypatch, complete):
    cfg = _config(tmp_path)
    present = None if complete else {("mnist", "mlp_ln", "main", 1)}
    _observations(cfg, present, monkeypatch)
    bundle = analyze_active(cfg)
    assert len(bundle.run_inventory) == 48
    assert bundle.run_inventory.valid.sum() == (48 if complete else 1)
    valid = bundle.per_seed_terminal_metrics.query("state == 'complete_native'")
    assert valid.run_directory.map(lambda value: Path(value).is_dir()).all()
    assert valid.metric_scope.eq("terminal_window").all()
    assert valid.final_checkpoint_epoch.eq(200).all()
    assert bundle.classwise_performance.metric_scope.eq("final_checkpoint").all()
    assert bundle.classwise_performance.checkpoint_epoch.eq(200).all()
    assert valid.mean_node_accuracy.eq(.42).all()
    assert valid.final_mean_node_accuracy.eq(.44).all()
    reports = tmp_path / cfg.report_root
    assert len(list((reports / "tables").glob("*.csv"))) == 17
    assert len(list((reports / "tables").glob("*.tex"))) == 17
    assert len(list((reports / "figures").rglob("*.png"))) == 26
    assert len(list((reports / "figures").rglob("*.pdf"))) == 26
    assert len(list((reports / "per_seed").glob("*.png"))) == (12 if complete else 1)
    assert (reports / "summary.md").is_file()
    counts = bundle.terminal_summary_by_dataset_model
    if complete:
        assert counts.query("metric == 'D_within_bar'").valid_n.eq(3).all()
        assert bundle.paired_control_summary.paired_n.eq(3).all()
        assert bundle.main_terminal_stability.status.eq("PASS").all()
    else:
        unavailable = counts.query("dataset == 'cifar10'")
        assert unavailable.valid_n.eq(0).all()
        assert unavailable["mean"].isna().all()
        contrasts = bundle.paired_control_summary
        assert contrasts.query("check == 'B_no_communication_within_minus_main'").paired_n.eq(0).all()


def test_class_panels_choose_accuracy_and_ce_and_label_endpoint(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _observations(cfg, {(dataset, model, "main", 1) for dataset in cfg.final.datasets for model in cfg.final.models}, monkeypatch)
    bundle = aggregate_active(cfg)
    snapshots = {}
    def capture(fig, directory, name, dpi):
        snapshots[str(directory), name] = [axis.get_title() for axis in fig.axes]
        plt.close(fig)
    figures = importlib.import_module("orbit_cifar.analysis.active.figures")
    monkeypatch.setattr(figures, "_save", capture)
    _family_figures(bundle, cfg, tmp_path / "figures", 24)
    for (directory, name), titles in snapshots.items():
        if name == "class_conditioned_accuracy_ce":
            expected = ("class_accuracy", "class_cross_entropy")
            assert all(metric in title and "final checkpoint" in title for metric, title in zip(expected, titles))


@pytest.mark.parametrize("values, count, mean, sd", [([], 0, math.nan, math.nan), ([2.], 1, 2., math.nan),
    ([1., 3.], 2, 2., math.sqrt(2)), ([1., 2., 3.], 3, 2., 1.), ([2., 2., 2.], 3, 2., 0.),
    ([1., math.nan, math.inf, -math.inf, 3.], 2, 2., math.sqrt(2))])
def test_statistics_only_count_finite_seed_observations(values, count, mean, sd):
    result = _stats(pd.Series(values, dtype=float))
    assert result["valid_n"] == count and result["expected_n"] == 3
    assert math.isnan(result["mean"]) if math.isnan(mean) else result["mean"] == pytest.approx(mean)
    if count < 2:
        assert math.isnan(result["sample_sd"]) and math.isnan(result["se"])
    else:
        assert result["sample_sd"] == pytest.approx(sd)
        assert result["se"] == pytest.approx(sd / math.sqrt(count))


@pytest.mark.parametrize("epochs, values", [([200] * 5, [1.] * 5),
    ([180, 185, 190, 195, 200], [1., 1., math.inf, 1., 1.]),
    ([180, 185, 190, 195, "invalid"], [1.] * 5), ([180, 185, 190, 195], [1.] * 4)])
def test_invalid_stability_evidence_is_undefined(epochs, values):
    result = _stability(pd.DataFrame({"epoch": epochs, "D_within": values}), "mnist", "mlp_ln", 1, "D_within", 1e-14)
    assert result["status"] == "UNDEFINED"


def test_stability_uses_actual_spacing_and_active_terminal_epoch_contract():
    epochs = np.array([171, 177, 188, 197, 200])
    rows = pd.DataFrame({"epoch": epochs, "D_within": 1 + epochs / 10000})
    result = _stability(rows, "mnist", "mlp_ln", 1, "D_within", 1e-14)
    assert result["slope"] == pytest.approx(.0001)
    assert result["normalized_drift"] == pytest.approx(.0001 * 29 / rows.D_within.mean())
    assert _stability(rows, "mnist", "mlp_ln", 1, "D_within", 1e-14,
                      expected_epochs=(180, 185, 190, 195, 200))["status"] == "UNDEFINED"
    assert math.isnan(_safe_ratio(math.inf, 1., 1e-14))
    assert 1 - _safe_ratio(4., 2., 1e-14) == -1


def test_missing_comparisons_are_masked_and_empty_report_makes_no_claims(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _observations(cfg, set(), monkeypatch)
    bundle = aggregate_active(cfg)
    values, counts = _directional_data(bundle.robustness_consistency_summary)
    assert values.isna().all().all() and counts.eq(0).all().all()
    summary = bundle.robustness_consistency_summary.copy()
    summary.loc[0, ["valid_n", "positive", "negative"]] = [3, 0, 3]
    values, counts = _directional_data(summary)
    assert values.notna().sum().sum() == 1
    assert values.stack().dropna().iloc[0] == 0  # Actual three negative seeds remain a valid zero count.
    figure = _empty("Empty test")
    assert "Unavailable" in figure.axes[0].texts[0].get_text()
    plt.close(figure)
    report = importlib.import_module("orbit_cifar.analysis.active.report")
    report._summary(bundle, cfg, tmp_path / "summary.md")
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "no empirical conclusions" in text
    assert "supported in every available seed" not in text
    assert "illustrates behavior consistent with orbit structure" not in text


def test_paired_counts_use_matched_finite_seeds_only(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _observations(cfg, {("mnist", "mlp_ln", "main", 1), ("mnist", "mlp_ln", "main", 2),
                        ("mnist", "mlp_ln", "no_communication", 2), ("mnist", "mlp_ln", "no_communication", 3)}, monkeypatch)
    bundle = aggregate_active(cfg)
    row = bundle.paired_control_summary.query("dataset == 'mnist' and model == 'mlp_ln' and check == 'B_no_communication_within_minus_main'").iloc[0]
    assert row.paired_n == row.valid_n == 1
    assert row.expected_n == 3 and row["mean"] == pytest.approx(2.)
    assert math.isnan(row.se)


def test_duplicate_scientific_observations_rejected(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    rows = _observations(cfg, set(), monkeypatch)
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="Duplicate scientific key"):
        aggregate_active(cfg)


@pytest.mark.parametrize("paired_n", [0, 1, 2, 3])
def test_paired_summary_reports_actual_sample_size(tmp_path, monkeypatch, paired_n):
    cfg = _config(tmp_path)
    present = {("mnist", "mlp_ln", condition, seed) for seed in range(1, paired_n + 1)
               for condition in ("main", "no_communication")}
    _observations(cfg, present, monkeypatch)
    row = aggregate_active(cfg).paired_control_summary.query(
        "dataset == 'mnist' and model == 'mlp_ln' and check == 'B_no_communication_within_minus_main'").iloc[0]
    assert row.paired_n == row.valid_n == paired_n and row.expected_n == 3
    if paired_n == 0:
        assert math.isnan(row["mean"])
    else:
        assert row["mean"] == pytest.approx(sum(range(1, paired_n + 1)) / paired_n)
    assert math.isnan(row.se) if paired_n < 2 else math.isfinite(row.se)


def test_relative_contrasts_are_seed_level_and_do_not_replace_mean_checkpoint_ratio(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    present = {("mnist", "mlp_ln", condition, seed) for seed in (1, 2, 3)
               for condition in ("main", "no_communication")}
    rows = _observations(cfg, present, monkeypatch)
    for row in rows:
        if row["state"] != "complete_native":
            continue
        path = Path(row["run_directory"]) / "metrics/terminal_metrics.json"
        terminal = json.loads(path.read_text(encoding="utf-8"))
        terminal["D_within_bar"] = {"main": [2., 4., 3.], "no_communication": [1., 2., 24.]}[row["condition"]][row["seed"] - 1]
        terminal["D_total_bar"] = terminal["D_within_bar"] + terminal["D_between_bar"]
        terminal["R_orbit_bar"] = .123  # Mean checkpoint ratios is a distinct supplied quantity.
        path.write_text(json.dumps(terminal), encoding="utf-8")
    bundle = aggregate_active(cfg)
    ratios = bundle.relative_robustness_metrics.query(
        "dataset == 'mnist' and model == 'mlp_ln' and metric == 'communication_reduction'")
    assert ratios.value.tolist() == pytest.approx([-1., -1., .875])
    assert ratios.value.mean() == pytest.approx(-.375)
    assert bundle.per_seed_terminal_metrics.query("state == 'complete_native'").R_orbit_bar.eq(.123).all()


def test_all_latex_special_characters_are_escaped_once():
    assert _latex_escape("$x^{2}$ ~ \\_%&#") == r"\$x\textasciicircum{}\{2\}\$ \textasciitilde{} \textbackslash{}\_\%\&\#"


def test_latex_serializes_structured_planner_prerequisites():
    assert _latex_escape([]) == "[]"
    assert _latex_escape(["missing_data", "missing_selection"]) == r'["missing\_data", "missing\_selection"]'
    assert _latex_escape({"reason": "missing_manifest"}) == r'\{"reason": "missing\_manifest"\}'
