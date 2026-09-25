"""Contracts unique to the read-only export verifier, separate from native runs."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_export", Path(__file__).resolve().parents[1] / "scripts/verify_export.py")
assert _SPEC is not None and _SPEC.loader is not None
verifier = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verifier)


def _manifest(root: Path, names: list[str]) -> None:
    with (root / "SHA256_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        for name in names:
            payload = (root / name).read_bytes()
            writer.writerow({"relative_path": name, "bytes": len(payload),
                             "sha256": hashlib.sha256(payload).hexdigest()})


def test_manifest_detects_content_changes_and_missing_evidence_without_writing(tmp_path):
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("value\n1\n", encoding="utf-8")
    _manifest(tmp_path, ["evidence.csv"])
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = verifier.verify_package_manifest(tmp_path)
    assert result["errors"] == [] and result["hashed_files"] == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    evidence.write_text("value\n2\n", encoding="utf-8")
    assert any("SHA256 mismatch" in error for error in verifier.verify_package_manifest(tmp_path)["errors"])
    evidence.unlink()
    result = verifier.verify_package_manifest(tmp_path)
    assert result["missing"] == ["evidence.csv"] and result["errors"]


def test_manifest_rejects_duplicate_entries_unlisted_files_and_path_escape(tmp_path):
    (tmp_path / "evidence.csv").write_text("1", encoding="utf-8")
    _manifest(tmp_path, ["evidence.csv", "evidence.csv"])
    (tmp_path / "unlisted.csv").write_text("2", encoding="utf-8")
    errors = verifier.verify_package_manifest(tmp_path)["errors"]
    assert any("duplicate" in error for error in errors)
    assert any("not covered" in error for error in errors)
    for name in ("../outside", "C:/outside", "nested/../../outside", "nested\\outside"):
        with pytest.raises(ValueError, match="unsafe"):
            verifier._safe_path(tmp_path, name)


def test_original_omissions_require_both_fixed_policy_and_original_audit_record():
    run = "runs/mnist/mlp_ln/main/seed_001"
    last = "checkpoints/last.pt"
    retained = "artifacts/terminal_predictions/epoch_175.pt"
    audit = {f"{run}/{last}", f"{run}/{retained}", f"{run}/checkpoints/final.pt"}
    assert verifier.classify_missing(run, last, audit)
    assert verifier.classify_missing(run, retained, audit)
    assert verifier.classify_missing(run, last, set()) is None
    assert verifier.classify_missing(run, "checkpoints/final.pt", audit) is None
    other = run.replace("/main/", "/misaligned_data/")
    assert verifier.classify_missing(other, retained, {f"{other}/{retained}"}) is None
    pilot = "pilots/mnist/mlp_ln/lambda_0/seed_000"
    assert verifier.classify_missing(pilot, "checkpoints/final.pt", {f"{pilot}/checkpoints/final.pt"})


def _metric_fixture(root: Path) -> tuple[dict, dict]:
    (root / "metrics").mkdir()
    key = {"dataset": "mnist", "model": "mlp_ln", "condition": "main", "seed": 1}
    cfg = {"training": {"terminal_window_checkpoints": 5, "ratio_threshold": 1e-14}}
    rows = [{"epoch": epoch, "D_within": 1., "D_between": 2., "D_total": 3., "R_orbit": 2 / 3,
             "mean_node_accuracy": .5, "orbit_ensemble_accuracy": .6,
             "raw_logit_rms": 1., "centered_logit_rms": .9} for epoch in range(0, 201, 5)]
    pd.DataFrame(rows).to_csv(root / "metrics/checkpoint_metrics.csv", index=False)
    terminal = {**key, "numerically_stable": True, "terminal_window_short": False,
                "probe_split": "official_test", "D_within_bar": 1., "D_between_bar": 2.,
                "D_total_bar": 3., "R_orbit_bar": 2 / 3}
    for field in ("mean_node_accuracy", "orbit_ensemble_accuracy", "raw_logit_rms", "centered_logit_rms"):
        terminal[field] = terminal["final_" + field] = rows[-1][field]
    (root / "metrics/terminal_metrics.json").write_text(json.dumps(terminal), encoding="utf-8")
    return key, cfg


@pytest.mark.parametrize("corruption", ["missing_epoch", "terminal_average", "ratio"])
def test_completed_flags_do_not_hide_incomplete_or_inconsistent_metrics(tmp_path, corruption):
    key, cfg = _metric_fixture(tmp_path)
    assert verifier.verify_metrics(tmp_path, key, cfg)["D_total_bar"] == 3.
    path = tmp_path / "metrics/checkpoint_metrics.csv"
    if corruption == "terminal_average":
        path = tmp_path / "metrics/terminal_metrics.json"
        terminal = json.loads(path.read_text())
        terminal["D_within_bar"] = 2.
        path.write_text(json.dumps(terminal), encoding="utf-8")
    else:
        frame = pd.read_csv(path)
        if corruption == "missing_epoch":
            frame = frame[frame.epoch != 185]
        else:
            frame.loc[frame.epoch == 0, "R_orbit"] = .1
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cadence|terminal mean|R_orbit"):
        verifier.verify_metrics(tmp_path, key, cfg)


def test_report_output_requires_new_directory_outside_evidence(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    export = project / verifier.EXPORT_RELATIVE
    for relative in ("src/new", "numerical experiments/new", "output_numerical experiments/new",
                     f"{verifier.EXPORT_RELATIVE}/new", "data/new", "real_world_experiments/scripts/new"):
        with pytest.raises(ValueError, match="evidence|protected"):
            verifier.validate_output_directory(project / relative, project, export)
    with pytest.raises(ValueError, match="already exists"):
        verifier.validate_output_directory(project, project, export)
    target = project / "reproduced/verification"
    assert verifier.validate_output_directory(target, project, export) == target
    assert not target.exists()


def test_missing_export_fails_with_json_report_and_no_files(tmp_path, capsys):
    before = list(tmp_path.iterdir())
    code = verifier.main(["--root", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert code == 1 and result["valid"] is False and result["errors"]
    assert result["native_validation"]["complete"] is False
    assert list(tmp_path.iterdir()) == before
