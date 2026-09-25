"""Independent numerical and transactional regressions; no scientific training."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from orbit_cifar.constants import NUM_NODES, REFERENCE_ORBITS as GROUPS
from orbit_cifar.evaluation.dispersion import dispersion_metrics
from orbit_cifar.evaluation.logits import center_logits
from orbit_cifar.evaluation.orbit_diagnostics import geometry, motion, PROFILE, RETAINED_EPOCHS
from orbit_cifar.experiments.orbit_diagnostics import (
    OrbitDiagnosticRecorder, probe_contract, coverage, verify_diagnostics, GEOMETRY, MOTION, STATUS,
)
from orbit_cifar.experiments.completion import verify_completed_run


def values(rows, metric):
    return [row["value"] for row in rows if row["metric"] == metric]


def logits(classes=10):
    return torch.randn(14, 3, classes, generator=torch.Generator().manual_seed(82))


@pytest.mark.parametrize("classes", [10, 100])
def test_unequal_groups_reconstruct_native_and_do_not_mutate(classes):
    raw = logits(classes)
    original = raw.clone()
    rows = geometry(raw, GROUPS)
    native = dispersion_metrics(center_logits(raw))
    # Independently use flattened NumPy predictors, not the implementation helper.
    f = raw.numpy().astype(np.float64)
    f -= f.mean(-1, keepdims=True)
    flat = f.reshape(14, -1)
    means = [flat[list(group)].mean(0) for group in GROUPS]
    w = [np.mean((flat[list(group)] - means[k])**2) for k, group in enumerate(GROUPS)]
    b = [np.mean((means[k] - means[l])**2) for k in range(4) for l in range(k + 1, 4)]
    np.testing.assert_allclose(values(rows, "W_k"), w, rtol=1e-12)
    np.testing.assert_allclose(values(rows, "B_kl"), b, rtol=1e-12)
    for metric in ("D_within", "D_between"):
        assert values(rows, metric + "_reconstructed")[0] == pytest.approx(native[metric], rel=1e-12)
    assert torch.equal(raw, original)


def test_constant_distinct_and_coincident_centers():
    direction = torch.tensor([1., -1.] + [0.] * 8, dtype=torch.float64)
    raw = torch.zeros(14, 2, 10, dtype=torch.float64)
    for k, group in enumerate(GROUPS):
        raw[list(group)] = k * direction
    rows = geometry(raw, GROUPS)
    assert values(rows, "W_k") == [0.] * 4
    assert values(rows, "B_kl") == pytest.approx([.2, .8, 1.8, .2, .8, .2])
    raw[list(GROUPS[1])] = raw[0].clone()
    rows = geometry(raw, GROUPS)
    assert values(rows, "min_center_distance_squared") == [0.]
    assert values(rows, "D_between_reconstructed")[0] > 0


def test_common_drift_and_rigid_transform_preserve_geometry_but_move_predictors():
    raw = logits().double()
    drift = torch.tensor([1., -1.] + [0.] * 8)
    translated = raw + drift
    for changed in (translated, -raw):
        original, moved = geometry(raw, GROUPS), geometry(changed, GROUPS)
        np.testing.assert_allclose([r["value"] for r in original], [r["value"] for r in moved], atol=1e-12)
        observed = motion(raw, changed, GROUPS)
        assert min(values(observed, "T_center_k")) > 0
        assert min(values(observed, "T_node_i")) > 0
        assert values(observed, "T_global")[0] > 0
    np.testing.assert_allclose(values(motion(raw, translated, GROUPS), "T_center_relative_k"), 0, atol=1e-28)
    assert all(row["value"] == 0 for row in motion(raw, raw.clone(), GROUPS))


def test_negative_strict_gap_is_a_valid_observation():
    raw = torch.zeros(14, 2, 10)
    raw[0, :, 0], raw[0, :, 1] = 10, -10
    rows = geometry(raw, GROUPS, strict=True)
    assert values(rows, "strict_pairwise_gap_squared")[0] < 0
    with pytest.raises(ValueError, match="Partition"):
        geometry(raw, [[0], [0]], strict=True)
    with pytest.raises(ValueError, match="integers"):
        geometry(raw, [[float(i) for i in range(14)]])


def recorder(root, *, classes=10):
    images = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2)
    labels = torch.arange(3)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    contract = probe_contract(loader, "mnist", list(map(str, range(classes))), "fixture_v1")
    identity = {"dataset": "mnist", "model": "mlp_ln", "condition": "main", "seed": 1,
                "stage": "final", "scientific_fingerprint": "fixture_scientific_identity"}
    return OrbitDiagnosticRecorder(root, identity, contract), loader


def record(rec, epoch, raw=None, digest="ordered_images_v1"):
    raw = logits() if raw is None else raw
    rec.record(raw, torch.arange(3), digest, epoch, epoch * 2, dispersion_metrics(center_logits(raw)))


@pytest.mark.parametrize("mismatch", ["image_order", "dataset", "class_order", "group", "class_count", "run", "dtype"])
def test_wrong_probe_family_class_order_partition_and_precision_rejected(tmp_path, mismatch):
    rec, loader = recorder(tmp_path)
    record(rec, 0)
    snapshot = rec.publish(0, 0)
    if mismatch == "image_order":
        with pytest.raises(ValueError, match="probe identity"):
            record(rec, 5, digest="same_labels_swapped_images")
    elif mismatch == "class_count":
        with pytest.raises(ValueError, match="shape/class count"):
            record(rec, 5, raw=logits(100))
    elif mismatch == "dtype":
        with pytest.raises(ValueError, match="prediction dtype"):
            record(rec, 5, raw=logits().double())
    else:
        other, _ = recorder(tmp_path)
        if mismatch == "dataset":
            other.contract["dataset"] = "cifar10"
        elif mismatch == "class_order":
            other.contract["class_order"].reverse()
        elif mismatch == "group":
            other.groups[0][0], other.groups[1][0] = other.groups[1][0], other.groups[0][0]
        else:
            other.run_identity["seed"] = 2
        with pytest.raises(ValueError, match="identity mismatch"):
            other.restore(snapshot, 0, 0)
    with pytest.raises(ValueError, match="sequential"):
        probe_contract(DataLoader(loader.dataset, shuffle=True), "mnist", list(map(str, range(10))), "v1")


def test_missing_exact_pair_is_not_bridged_and_endpoint_is_not_path_sum(tmp_path):
    rec, _ = recorder(tmp_path)
    record(rec, 175)
    record(rec, 185)
    assert rec.motion_rows == []
    observed = coverage(rec.predictions)
    assert observed["temporal_coverage"] == "partial_history"
    assert observed["pairs"][1]["missing_epochs"] == [180]
    other, _ = recorder(tmp_path / "complete")
    direction = torch.tensor([1., -1.] + [0.] * 8, dtype=torch.float64)
    for k, epoch in enumerate(RETAINED_EPOCHS):
        record(other, epoch, logits().double() + k * direction)
    path_sum = sum(r["value"] for r in other.motion_rows if r["metric"] == "T_global" and r["scope"] == "adjacent_transition")
    endpoint = [r["value"] for r in other.motion_rows if r["metric"] == "T_global" and r["scope"] == "endpoint_displacement"][0]
    assert path_sum == pytest.approx(1.)
    assert endpoint == pytest.approx(5.)
    assert coverage(other.predictions)["temporal_coverage"] == "complete"


def test_transactional_restore_discards_uncommitted_arrays_and_rows(tmp_path):
    rec, _ = recorder(tmp_path / "resumed")
    for epoch in range(0, 176, 5):
        record(rec, epoch, logits() + epoch * .01 * torch.arange(10))
    committed = rec.publish(175, 350)
    record(rec, 180)
    rec.publish(180, 360)  # interruption before publishing the model checkpoint
    fresh, _ = recorder(rec.root)
    fresh.restore(committed, 175, 350)
    assert not (fresh.root / "artifacts/terminal_predictions/epoch_180.pt").exists()
    assert pd.read_csv(fresh.root / GEOMETRY).epoch.max() == 175
    uninterrupted, _ = recorder(tmp_path / "uninterrupted")
    for epoch in range(0, 201, 5):
        raw = logits() + epoch * .01 * torch.arange(10)
        record(uninterrupted, epoch, raw)
        if epoch > 175:
            record(fresh, epoch, raw)
    first = fresh.publish(200, 400)
    record(fresh, 200, logits() + 2. * torch.arange(10))
    second = fresh.publish(200, 400)
    assert first == second
    uninterrupted.publish(200, 400)
    for name in (GEOMETRY, MOTION, STATUS):
        assert (fresh.root / name).read_bytes() == (uninterrupted.root / name).read_bytes()
    for epoch in RETAINED_EPOCHS:
        assert torch.equal(fresh._load(epoch)["raw_logits"], uninterrupted._load(epoch)["raw_logits"])


def enhanced_checkpoint(rec):
    snapshot = rec.publish(200, 400)
    torch.save(torch.arange(3), rec.root / "artifacts/final_labels.pt")
    checkpoint = {"epoch": 200, "global_step": 400, "config_fingerprint": rec.run_identity["scientific_fingerprint"],
                  "extra": {"orbit_diagnostics": snapshot}}
    resolved = {"orbit_diagnostics_profile": PROFILE, "data": {"name": "mnist"}, "model": {"name": "mlp_ln", "num_classes": 10},
                "resolved_condition": {"name": "main"}, "seed": 1, "stage": "final",
                "training": {"epochs": 200, "eval_every_epochs": 5}}
    return checkpoint, resolved


@pytest.mark.parametrize("damage", ["truncated_array", "wrong_epoch", "mixed_table", "missing_table", "missing_history", "unknown_schema"])
def test_enhanced_completion_detects_corruption(tmp_path, damage):
    rec, _ = recorder(tmp_path)
    for epoch in range(0, 201, 5):
        record(rec, epoch)
    checkpoint, resolved = enhanced_checkpoint(rec)
    assert verify_diagnostics(tmp_path, checkpoint, resolved, completed=True)["temporal_coverage"] == "complete"
    if damage == "truncated_array":
        (tmp_path / "artifacts/terminal_predictions/epoch_185.pt").write_bytes(b"truncated")
    elif damage == "wrong_epoch":
        checkpoint["epoch"] = 195
    elif damage == "mixed_table":
        frame = pd.read_csv(tmp_path / GEOMETRY)
        frame.loc[0, "seed"] = 2
        frame.to_csv(tmp_path / GEOMETRY, index=False)
    elif damage == "missing_table":
        (tmp_path / MOTION).unlink()
    elif damage == "missing_history":
        checkpoint["extra"].clear()
    else:
        resolved["orbit_diagnostics_profile"] = "future_unknown_v99"
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_diagnostics(tmp_path, checkpoint, resolved, completed=True)


def test_old_completed_run_remains_valid_without_diagnostics(native_final_cell):
    _, key, root = native_final_cell
    result = verify_completed_run(root, key)
    assert result["valid"], result
    assert result["orbit_diagnostics"]["temporal_coverage"] == "not_recorded"


def test_reports_preserve_seed_coverage_and_do_not_mutate_legacy(tmp_path):
    from orbit_cifar.analysis.active.orbit_diagnostics import diagnostic_tables
    rec, _ = recorder(tmp_path / "new")
    for epoch in range(0, 201, 5):
        record(rec, epoch)
    rec.publish(200, 400)
    legacy = tmp_path / "old"
    (legacy / "artifacts").mkdir(parents=True)
    (legacy / "metrics").mkdir()
    torch.save(logits(), legacy / "artifacts/final_raw_logits.pt")
    pd.DataFrame({"epoch": [200]}).to_csv(legacy / "metrics/checkpoint_metrics.csv", index=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.rglob("*") if p.is_file()}
    rows = [{"dataset": "mnist", "model": "mlp_ln", "condition": "main", "seed": seed,
             "valid": seed != 3, "run_directory": str(rec.root if seed == 1 else legacy),
             "source_type": "native" if seed == 1 else "legacy_reused"} for seed in (1, 2, 3)]
    tables = diagnostic_tables(pd.DataFrame(rows))
    assert set(tables["temporal_coverage"].seed) == {1, 2, 3}
    assert tables["temporal_coverage"].query("seed == 2").temporal_coverage.unique().tolist() == ["not_recorded"]
    assert len(tables["endpoint_displacement_per_seed"]) == 25
    posthoc = tables["terminal_orbit_geometry"].query("seed == 2")
    assert len(posthoc.query("metric == 'B_kl'")) == 6
    assert posthoc.source_sha256.notna().all()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in legacy.rglob("*") if p.is_file()}


def test_actual_same_label_image_swap_rejected_without_an_extra_probe_pass(tmp_path):
    from orbit_cifar.experiments.evaluation import _evaluate
    from orbit_cifar.config_schema import load_config
    rec, loader = recorder(tmp_path)
    loader.dataset.tensors[1][:] = 0  # swapping images cannot change the label vector
    calls = []
    class Counter(torch.nn.Module):
        def forward(self, images):
            calls.append(len(images))
            return images.flatten(1).mean(1, keepdim=True).expand(-1, 10)
    models = [Counter() for _ in range(14)]
    cfg = load_config(Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml")
    _evaluate(models, loader, torch.device("cpu"), cfg, 0, diagnostics=rec)
    assert len(calls) == 14 * 2
    images = loader.dataset.tensors[0]
    images[[0, 1]] = images[[1, 0]].clone()
    with pytest.raises(ValueError, match="probe identity"):
        _evaluate(models, loader, torch.device("cpu"), cfg, 5, diagnostics=rec)
    assert len(calls) == 14 * 4


@pytest.mark.parametrize("classes", [10, 100])
def test_saved_raw_precision_and_recalculation(classes, tmp_path):
    rec, _ = recorder(tmp_path, classes=classes)
    raw = logits(classes).double() + 1e-10
    record(rec, 175, raw)
    saved = rec._load(175)["raw_logits"]
    assert saved.dtype == raw.dtype
    assert torch.equal(saved, raw)
    assert geometry(saved, GROUPS) == geometry(raw, GROUPS)


@pytest.mark.parametrize("field", ["probe_id", "run_identity", "shape", "epoch"])
def test_semantically_wrong_predecessor_rejected_even_with_updated_file_hash(tmp_path, field):
    from orbit_cifar.experiments.orbit_diagnostics import file_hash
    rec, _ = recorder(tmp_path)
    record(rec, 175)
    path = tmp_path / "artifacts/terminal_predictions/epoch_175.pt"
    payload = torch.load(path, weights_only=True)
    if field == "probe_id":
        payload[field] = "wrong-probe"
    elif field == "run_identity":
        payload[field]["model"] = "resnet20_gn"
    elif field == "shape":
        payload[field] = [14, 3, 100]
    else:
        payload[field] = 170
    torch.save(payload, path)
    rec.predictions["175"]["sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="identity/shape/precision"):
        record(rec, 180)


def test_native_completion_enforces_enhanced_dependencies_after_republication(native_final_cell):
    """Upgrade only this synthetic fixture to Main; no old source results are edited."""
    import yaml
    from orbit_cifar.experiments.conditions import resolve_condition
    from orbit_cifar.experiments.matrix import ScientificKey
    from orbit_cifar.experiments.completion import publish_completion, verify_resumable_run
    from orbit_cifar.utils.io import atomic_write_json, atomic_write_yaml, config_fingerprint
    _, _, old = native_final_cell
    root = old.parent.parent / "main" / old.name
    root.parent.mkdir()
    old.rename(root)
    key = ScientificKey("mnist", "mlp_ln", "main", 1)
    resolved = yaml.safe_load((root / "resolved_config.yaml").read_text())
    condition, _ = resolve_condition("main", lambda_override=0.)
    resolved["resolved_condition"] = condition.to_dict()
    resolved["scientific_identity"]["condition"] = condition.to_dict()
    fingerprint = config_fingerprint(resolved["scientific_identity"])
    resolved["scientific_fingerprint"] = fingerprint
    resolved["orbit_diagnostics_profile"] = PROFILE
    rec, _ = recorder(root)
    rec.run_identity["scientific_fingerprint"] = fingerprint
    rec.contract["class_order"] = resolved["scientific_identity"]["dataset"]["class_names"]
    rec.contract["dataset_version"] = resolved["scientific_identity"]["dataset_version"]
    rec.contract["split"] = "test"
    rec.contract["probe_size"] = 10
    rec.contract["ordered_indices"] = list(range(10))
    labels = torch.arange(10)
    torch.save(labels, root / "artifacts/final_labels.pt")
    confusions = {}
    for kind, indices, correct in (("node", range(NUM_NODES), 6), ("orbit", range(1, len(GROUPS) + 1), 7)):
        matrix = np.zeros((10, 10), dtype=np.int64)
        for label in range(10):
            matrix[label, label if label < correct else (label + 1) % 10] = 1
        for index in indices:
            confusions[f"node_{index:02d}" if kind == "node" else f"orbit_{index}"] = matrix
    np.savez_compressed(root / "artifacts/confusion_matrices.npz", **confusions)
    native = pd.read_csv(root / "metrics/checkpoint_metrics.csv")
    # These arrays and their metadata are synthetic verification fixtures only.
    for epoch in range(0, 201, 5):
        raw = logits().repeat(1, 4, 1)[:, :10]
        metrics = dispersion_metrics(center_logits(raw))
        rec.record(raw, labels, "fixture_images", epoch, epoch, metrics)
        for metric in ("D_within", "D_between", "D_total", "R_orbit"):
            native.loc[native.epoch == epoch, metric] = metrics[metric]
    native.to_csv(root / "metrics/checkpoint_metrics.csv", index=False)
    terminal = json.loads((root / "metrics/terminal_metrics.json").read_text())
    terminal["condition"] = "main"
    for metric in ("D_within", "D_between", "D_total", "R_orbit"):
        terminal[metric + "_bar"] = float(native.tail(5)[metric].mean())
    atomic_write_json(root / "metrics/terminal_metrics.json", terminal)
    atomic_write_yaml(root / "resolved_config.yaml", resolved)
    snapshot = rec.publish(200, 200)
    for name in ("last", "final"):
        path = root / f"checkpoints/{name}.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["config_fingerprint"] = fingerprint
        checkpoint["extra"] = {"terminal_metrics": terminal, "orbit_diagnostics": snapshot}
        torch.save(checkpoint, path)
    def complete():
        publication = publish_completion(root, key, fingerprint, 200, 200)
        atomic_write_json(root / "run_status.json", {"status": "completed", "numerically_stable": True,
                                                    "last_valid_epoch": 200, **publication})
    complete()
    result = verify_completed_run(root, key)
    assert result["valid"], result
    assert result["orbit_diagnostics"]["temporal_coverage"] == "complete"
    # Publishing a new outer completion manifest cannot bless files from a
    # different diagnostic checkpoint generation.
    table = pd.read_csv(root / GEOMETRY)
    table.loc[0, "value"] += 1
    table.to_csv(root / GEOMETRY, index=False)
    complete()
    result = verify_completed_run(root, key)
    assert not result["valid"] and "Diagnostic completion generation" in result["errors"][0]
    atomic_write_json(root / "run_status.json", {"status": "interrupted"})
    assert verify_resumable_run(root, key)["valid"]  # restore can repair the views
    rec.restore(snapshot, 200, 200)
    complete()
    assert verify_completed_run(root, key)["valid"]


def test_new_figures_show_each_orbit_pair_and_seed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orbit_cifar.analysis.active.orbit_diagnostics import diagnostic_tables, diagnostic_figures
    from orbit_cifar.analysis.active import figures
    rec, _ = recorder(tmp_path)
    for epoch in range(0, 201, 5):
        record(rec, epoch)
    rec.publish(200, 400)
    inventory = pd.DataFrame([{"dataset": "mnist", "model": "mlp_ln", "condition": "main", "seed": seed,
                               "valid": seed == 1, "source_type": "native", "run_directory": str(tmp_path)}
                              for seed in (1, 2, 3)])
    bundle = SimpleNamespace(**diagnostic_tables(inventory))
    cfg = SimpleNamespace(final=SimpleNamespace(datasets=["mnist"], models=["mlp_ln"], seeds=[1, 2, 3]),
                          analysis=SimpleNamespace(figure_dpi=24))
    observed = []
    def inspect(fig, directory, name, dpi):
        observed.append(name)
        assert len(fig.axes) == 15
        assert [len(axis.lines) for axis in fig.axes[:5]] == [4, 6, 4, 4, 3]
        assert "175-180" in [tick.get_text() for tick in fig.axes[2].get_xticklabels()]
        assert all("Seed 2" in axis.get_title() for axis in fig.axes[5:10])
        assert all("Seed 3" in axis.get_title() for axis in fig.axes[10:15])
        assert all(len(axis.lines) == 0 for axis in fig.axes[5:])
        figures.plt.close(fig)
    monkeypatch.setattr(figures, "_save", inspect)
    diagnostic_figures(bundle, cfg, tmp_path / "reports")
    assert observed == ["orbit_geometry_and_motion_per_seed"]
