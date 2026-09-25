from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce_paper.py"
_SPEC = importlib.util.spec_from_file_location("paper_reproduction_test_module", _SCRIPT)
paper = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paper)


def test_omitted_manuscript_has_no_reference_metadata(tmp_path):
    assert paper.reference_metadata(None) is None
    paper.write_readme(tmp_path, {}, 0)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "reference: not provided" in readme
    assert "No manuscript file or hash was recorded" in readme


def test_provided_reference_records_real_bytes_without_absolute_path(tmp_path):
    reference = tmp_path / "reference.txt"
    content = b"Optional reference fixture, not experimental evidence.\n"
    reference.write_bytes(content)
    metadata = paper.reference_metadata(reference)
    assert metadata == {"name": reference.name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()}
    paper.write_readme(tmp_path, {}, 0, metadata)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "`reference.txt`" in readme
    assert "not provided" not in readme
    assert str(tmp_path) not in readme
    assert str(tmp_path) not in str(metadata)


def test_explicit_missing_reference_fails_before_output_creation(tmp_path):
    output = tmp_path / "new-output"
    with pytest.raises(FileNotFoundError):
        paper.reproduce(output, project=tmp_path, paper=Path("missing-reference.pdf"))
    assert not output.exists()


@pytest.mark.parametrize("reference", [None, "optional-reference.pdf"])
def test_cli_forwards_optional_reference(monkeypatch, tmp_path, reference):
    received = {}

    def reproduce(output, *, paper):
        received.update(output=output, paper=paper)
        return {"test": True}

    monkeypatch.setattr(paper, "reproduce", reproduce)
    output = tmp_path / "output"
    arguments = ["--output", str(output)]
    if reference is not None:
        arguments.extend(["--paper", reference])
    assert paper.main(arguments) == 0
    assert received == {"output": output, "paper": Path(reference) if reference else None}
    assert not output.exists()


def test_output_rejects_evidence_ancestors_and_nonempty_destinations(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for relative in (
        paper.EXPORT_RELATIVE, f"{paper.EXPORT_RELATIVE}/figures/new",
        "numerical experiments/new", "output_numerical experiments/new",
        "reports/real_world_appendix_figures/new", "data/new", "scripts/new",
        "real_world_experiments", "real_world_experiments/scripts/new",
        "real_world_experiments/src/new", "real_world_experiments/configs/new",
        "real_world_experiments/tests/new", "real_world_experiments/data/new",
        "real_world_experiments/paper_results_backup/new",
        "outputs/final/new", "outputs_mnist_cifar10_resnet_mlp_3seed/new",
    ):
        with pytest.raises(ValueError, match="protected"):
            paper.checked_output(root / relative, root)
    for ancestor in (root, tmp_path):
        with pytest.raises(ValueError, match="ancestor"):
            paper.checked_output(ancestor, root)
    existing = root / "reproduction/paper"
    existing.mkdir(parents=True)
    marker = existing / "do-not-overwrite.txt"
    marker.write_text("evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        paper.checked_output(existing, root)
    assert marker.read_text(encoding="utf-8") == "evidence"
    assert paper.checked_output(root / "reproduction/new", root) == root / "reproduction/new"
    assert not (root / "reproduction/new").exists()
    paper_root = root / "real_world_experiments/paper_results"
    for destination in (paper_root, paper_root / "redraw", paper_root / "redraw/another"):
        assert paper.checked_output(destination, root) == destination
        assert not destination.exists()
    paper_root.mkdir(parents=True)
    generated = paper_root / "README.md"
    generated.write_text("existing paper results", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        paper.checked_output(paper_root, root)
    assert generated.read_text(encoding="utf-8") == "existing paper results"
    assert paper.checked_output(paper_root / "redraw", root) == paper_root / "redraw"


@pytest.mark.parametrize("link_location", ["paper_root", "nested"])
@pytest.mark.parametrize("destination", ["scripts", "results", "external"])
def test_paper_output_rejects_symlink_escape(tmp_path, link_location, destination):
    root = tmp_path / "project"
    code = root / "real_world_experiments"
    code.mkdir(parents=True)
    targets = {
        "scripts": code / "scripts",
        "results": root / paper.EXPORT_RELATIVE / "figures",
        "external": tmp_path / "outside",
    }
    target = targets[destination]
    target.mkdir(parents=True)
    paper_root = code / "paper_results"
    link = paper_root if link_location == "paper_root" else paper_root / "redirect"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Directory symlinks are unavailable: {error}")
    try:
        with pytest.raises(ValueError, match="escapes|protected"):
            paper.checked_output(link / "redraw", root)
        assert not (target / "redraw").exists()
    finally:
        link.unlink()


def test_source_hash_failure_and_unlisted_inputs_are_rejected(tmp_path):
    source = tmp_path / "package"
    source.mkdir()
    artifact = source / "metric.csv"
    original = b"epoch,value\n200,1\n"
    artifact.write_bytes(original)
    with (source / "SHA256_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow({"relative_path": "metric.csv", "bytes": len(original),
                         "sha256": hashlib.sha256(original).hexdigest()})
    reader = paper.SourceReader(source, tmp_path)
    assert reader.csv(artifact) == [{"epoch": "200", "value": "1"}]
    artifact.write_bytes(original.replace(b"200,1", b"200,2"))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        reader.read(artifact)
    with pytest.raises(ValueError, match="not covered"):
        reader.read(source / "unlisted.csv")
    with pytest.raises(ValueError, match="changed"):
        reader.check_unchanged()


def test_source_manifest_rejects_escape_paths_and_duplicate_entries(tmp_path):
    source = tmp_path / "package"
    source.mkdir()
    manifest = source / "SHA256_MANIFEST.csv"
    header = "relative_path,bytes,sha256\n"
    for unsafe in ("../outside.csv", "/outside.csv", "C:/outside.csv", "..\\outside.csv"):
        manifest.write_text(header + f"{unsafe},0,abc\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsafe"):
            paper.SourceReader(source, tmp_path)
    manifest.write_text(header + "a.csv,0,abc\na.csv,0,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        paper.SourceReader(source, tmp_path)


def _rows(terminal_within):
    rows = []
    values = [1.] * 36 + list(terminal_within)
    for epoch, within in zip(range(0, 201, 5), values):
        rows.append({"epoch": epoch, "D_within": within, "D_between": 1.,
                     "D_total": within + 1., "R_orbit": 1. / (within + 1.)})
    return rows


def test_terminal_ratios_are_averaged_before_seed_aggregation():
    terminal, _ = paper.terminal_statistics(_rows([1, 2, 3, 4, 5]))
    assert terminal["R_orbit"] == pytest.approx((1/2 + 1/3 + 1/4 + 1/5 + 1/6) / 5)
    assert terminal["R_orbit"] != pytest.approx(terminal["D_between"] / terminal["D_total"])
    mean, sd = paper.mean_sd([1., 2., 3.])
    assert mean == 2.
    assert sd == 1.  # sample SD, not sqrt(2/3)
    with pytest.raises(ValueError, match="three training seeds"):
        paper.mean_sd([1., 2.])


def test_fitted_drift_is_ols_not_endpoint_change():
    _, statistics = paper.terminal_statistics(_rows([1, 1, 1, 5, 1]))
    within = next(r for r in statistics if r["metric"] == "D_within")
    assert within["mean"] == pytest.approx(1.8)
    assert within["OLS_slope_per_epoch"] == pytest.approx(.08)
    assert within["fitted_drift_percent"] == pytest.approx(100 * 1.6 / 1.8)
    assert within["fitted_drift_percent"] > 0  # both endpoint values are 1
    assert within["sample_sd"] == pytest.approx(np.sqrt(3.2))
    with pytest.raises(ValueError, match="exactly epochs"):
        paper.terminal_statistics(_rows([1, 1, 1, 5, 1])[:-1])
    broken = _rows([1, 1, 1, 5, 1])
    broken[-1]["R_orbit"] = .1
    with pytest.raises(ValueError, match="R_orbit"):
        paper.terminal_statistics(broken)


def test_pairwise_means_exclude_diagonal_and_use_reference_groups():
    group = np.repeat(np.arange(4), [2, 3, 4, 5])
    matrix = np.where(group[:, None] == group[None, :], 2., 8.)
    np.fill_diagonal(matrix, 0)
    assert paper.pairwise_means(matrix) == (2., 8., 4.)
    matrix[0, 1] = matrix[1, 0] = 22.
    # Exactly 20 unordered within-orbit pairs; the one changed pair adds 20/20.
    assert paper.pairwise_means(matrix) == (3., 8., 8/3)


def test_centroid_displacement_uses_square_root_and_final_minimum_separation():
    assert paper.rho_percent([1, 4, 9, 16], [100, 200, 300, 400, 500, 600]) == 40.
    with pytest.raises(ValueError, match="Undefined"):
        paper.rho_percent([1, 4, 9, 16], [0, 200, 300, 400, 500, 600])
    with pytest.raises(ValueError, match="four centroid"):
        paper.rho_percent([1, 4, 9], [100, 200, 300, 400, 500, 600])
