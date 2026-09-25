import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from orbit_cifar.analysis.active import aggregate_active, analyze_active
from orbit_cifar.experiments import planner


def test_active_analysis_uses_verified_planner_cells(native_final_cell, monkeypatch):
    cfg, key, run = native_final_cell
    assert key.condition == "no_communication"
    manifest_path = Path(cfg.project_root) / cfg.manifest_root / key.dataset / "heterogeneous.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def dataset(name, root, train, download):
        assert download is False
        if name != key.dataset:
            raise FileNotFoundError("Only this synthetic family exists")
        return SimpleNamespace(targets=manifest["source_labels"], train=bool(train))

    # Replace only absent real dataset access. The planner, status reader,
    # manifest checks, completion verifier, and report generation remain real.
    monkeypatch.setattr(planner, "load_dataset", dataset)
    cfg.analysis.figure_dpi = 24
    bundle = analyze_active(cfg)
    assert len(bundle.run_inventory) == 48
    assert bundle.run_inventory.valid.sum() == 1
    cell = bundle.run_inventory[bundle.run_inventory.valid].iloc[0]
    assert cell.state == "complete_native"
    assert Path(cell.run_directory) == run
    valid = bundle.per_seed_terminal_metrics.query("state == 'complete_native'").iloc[0]
    assert valid.final_checkpoint_epoch == 200 and valid.metric_scope == "terminal_window"
    reports = Path(cfg.project_root) / cfg.report_root
    assert (reports / "summary.md").is_file()
    assert len(list((reports / "tables").glob("*.csv"))) == 17

    # Changing a signed artifact must make the exact same discovered cell
    # unavailable; a stale file cannot become a zero-valued scientific result.
    artifact = run / "artifacts/pairwise_distance.npy"
    pairwise = np.load(artifact, allow_pickle=False)
    pairwise[0, 1] += 1
    np.save(artifact, pairwise)
    invalid = aggregate_active(cfg)
    assert invalid.run_inventory.valid.sum() == 0
    stale_cell = invalid.run_inventory[invalid.run_inventory.run_directory == str(run)].iloc[0]
    assert stale_cell.state == "blocked" and "checksum" in stale_cell.reason
    summary = invalid.terminal_summary_by_dataset_model
    assert summary.valid_n.eq(0).all() and summary["mean"].isna().all()
