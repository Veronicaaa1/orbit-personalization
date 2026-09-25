"""Regenerate current-paper figures and tables from the preserved FINAL export.

This command verifies exported evidence, not native training completeness. It
never trains, repairs source artifacts, or overwrites an existing report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT = CODE_ROOT.parent
EXPORT_NAME = "results"
EXPORT_RELATIVE = Path("real_world_experiments") / EXPORT_NAME
FAMILIES = (
    ("mnist", "mlp_ln", "MNIST / MLP-LN"),
    ("mnist", "resnet20_gn", "MNIST / ResNet-20-GN"),
    ("cifar10", "mlp_ln", "CIFAR-10 / MLP-LN"),
    ("cifar10", "resnet20_gn", "CIFAR-10 / ResNet-20-GN"),
)
CONDITIONS = ("main", "no_communication", "misaligned_data", "fully_symmetric")
SEEDS = (1, 2, 3)
METRICS = ("D_within", "D_between", "D_total", "R_orbit")
WINDOW = np.array([180, 185, 190, 195, 200])
PLOT_SCRIPT = Path("real_world_experiments/scripts/plot_appendix_figures.py")


def checked_output(path: Path, project: Path = PROJECT) -> Path:
    """Resolve symlinks before excluding every preserved evidence/code tree."""
    requested = path.expanduser().absolute()
    target, project = requested.resolve(), project.resolve()
    paper_root = project / "real_world_experiments" / "paper_results"
    paper_output = target == paper_root or paper_root in target.parents
    if (requested == paper_root or paper_root in requested.parents) and not paper_output:
        raise ValueError("Output escapes the designated paper-results tree after resolving symlinks")
    protected = (
        EXPORT_RELATIVE, "numerical experiments", "output_numerical experiments",
        "reports", "reports_mnist_cifar10_resnet_mlp_3seed", "data", "src",
        "configs", "tests", "docs", "scripts", ".git", ".vscode", ".codex",
        ".agents", "real_world_experiments", "outputs/final", "outputs/pilot",
        "outputs_mnist_cifar10_resnet_mlp_3seed",
    )
    if target == project or target in project.parents:
        raise ValueError("Output must not be the project root or one of its ancestors")
    for name in protected:
        if name == "real_world_experiments" and paper_output:
            continue
        evidence = (project / name).resolve()
        if target == evidence or evidence in target.parents or target in evidence.parents:
            raise ValueError(f"Output overlaps a protected tree: {name}")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Output must be a new or empty directory; existing results are preserved")
    return target


class SourceReader:
    """Read only package-manifest-listed inputs and retain their source hashes."""

    def __init__(self, source: Path, project: Path):
        self.source, self.project = source.resolve(), project.resolve()
        self.records: dict[str, dict] = {}
        manifest = self.source / "SHA256_MANIFEST.csv"
        raw = manifest.read_bytes()
        self.expected = {}
        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
            relative = PurePosixPath(row["relative_path"])
            if (relative.is_absolute() or ".." in relative.parts or
                    "\\" in row["relative_path"] or ":" in row["relative_path"]):
                raise ValueError("Unsafe path in the source manifest")
            key = relative.as_posix()
            if key in self.expected:
                raise ValueError(f"Duplicate source-manifest entry: {key}")
            self.expected[key] = row
        self._record(manifest, raw)

    def _record(self, path: Path, raw: bytes) -> None:
        key = path.resolve().relative_to(self.project).as_posix()
        self.records[key] = {"path": key, "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()}

    def read(self, path: Path) -> bytes:
        resolved = path.resolve()
        key = resolved.relative_to(self.source).as_posix()
        row = self.expected.get(key)
        if row is None:
            raise ValueError(f"Required input is not covered by the original manifest: {key}")
        raw = resolved.read_bytes()
        if len(raw) != int(row["bytes"]) or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Source hash/size mismatch: {key}")
        self._record(resolved, raw)
        return raw

    def track(self, path: Path) -> Path:
        self.read(path)
        return path

    def csv(self, path: Path) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self.read(path).decode("utf-8-sig"))))

    def json(self, path: Path):
        return json.loads(self.read(path).decode("utf-8-sig"))

    def reference(self, path: Path) -> None:
        """Record local code references separately from package inputs."""
        self._record(path, path.read_bytes())

    def check_unchanged(self) -> None:
        for relative, record in self.records.items():
            raw = (self.project / relative).read_bytes()
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Source changed while preparing reproduction: {relative}")


def terminal_statistics(rows: list[dict]) -> tuple[dict, list[dict]]:
    epochs = np.array([int(row["epoch"]) for row in rows])
    if not np.array_equal(epochs, np.arange(0, 201, 5)):
        raise ValueError("A final run must contain exactly epochs 0,5,...,200")
    values = {key: np.array([float(row[key]) for row in rows]) for key in METRICS}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Non-finite checkpoint metric")
    if not np.allclose(values["D_total"], values["D_within"] + values["D_between"], atol=1e-9, rtol=0):
        raise ValueError("Dispersion decomposition does not hold")
    if np.any(values["D_total"] < 1e-14) or not np.allclose(
            values["R_orbit"], values["D_between"] / values["D_total"], atol=1e-9, rtol=0):
        raise ValueError("Recorded R_orbit disagrees with checkpoint dispersions")
    chosen = np.isin(epochs, WINDOW)
    terminal = {key: float(v[chosen].mean()) for key, v in values.items()}
    stability = []
    for metric in ("D_within", "D_between", "R_orbit"):
        sample = values[metric][chosen]
        mean = float(sample.mean())
        if abs(mean) < 1e-14:
            raise ValueError("Undefined terminal stability denominator")
        slope = float(np.polyfit(WINDOW, sample, 1)[0])
        stability.append({"metric": metric, "mean": mean,
                          "sample_sd": float(sample.std(ddof=1)),
                          "CV_percent": 100 * float(sample.std(ddof=1)) / abs(mean),
                          "OLS_slope_per_epoch": slope,
                          "fitted_drift_percent": 100 * 20 * abs(slope) / abs(mean)})
    return terminal, stability


def pairwise_means(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.shape != (14, 14) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 14-by-14 final pairwise matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0) or not np.allclose(np.diag(matrix), 0):
        raise ValueError("Pairwise matrix is not symmetric with zero diagonal")
    i, j = np.triu_indices(14, 1)
    group = np.repeat(np.arange(4), [2, 3, 4, 5])
    within = group[i] == group[j]
    pw, pc = float(matrix[i[within], j[within]].mean()), float(matrix[i[~within], j[~within]].mean())
    if pw <= 0:
        raise ValueError("Undefined cross/within distance ratio")
    return pw, pc, pc / pw


def rho_percent(displacements, separations) -> float:
    t, b = np.asarray(displacements, dtype=float), np.asarray(separations, dtype=float)
    if t.shape != (4,) or b.shape != (6,) or not np.isfinite(t).all() or not np.isfinite(b).all():
        raise ValueError("Expected four centroid displacements and six centroid separations")
    if np.min(t) < 0 or np.min(b) <= 0:
        raise ValueError("Undefined normalized centroid displacement")
    return float(100 * np.sqrt(t.max() / b.min()))


def mean_sd(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Each paper statistic requires all three training seeds")
    return float(values.mean()), float(values.std(ddof=1))


def load_plot_module(reader: SourceReader, output: Path):
    source = reader.project / PLOT_SCRIPT
    reader.reference(source)
    spec = importlib.util.spec_from_file_location("paper_source_figures", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load preserved figure source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.SOURCE, module.OUT = reader.project, reader.source, output / "data"
    module.read_csv, module.track = reader.csv, reader.track
    collected = {}
    module.write_csv = lambda name, rows: collected.__setitem__(name, rows)
    # collect() now reads and calculates only: its CSV writes are captured in memory.
    trajectories, motion, comparisons = module.collect()
    return module, collected, trajectories, motion, comparisons


def build_tables(reader: SourceReader):
    source = reader.source
    tables, terminal, stability, matrices, endpoints, lambdas = {}, {}, [], {}, [], {}
    profiles = None
    for dataset in ("mnist", "cifar10"):
        manifests = source / "experiment_provenance/manifests" / dataset
        hetero, balanced = (reader.json(manifests / f"{name}.json") for name in ("heterogeneous", "balanced"))
        current = hetero["profile_definitions"]
        if profiles is not None and profiles != current:
            raise ValueError("The two datasets use different class-count profiles")
        profiles = current
        if hetero["validation_indices"] != balanced["validation_indices"]:
            raise ValueError("Balanced and heterogeneous validation splits differ")
        for manifest in (hetero, balanced):
            if manifest["split_seed"] != 2027 or manifest["validation_class_counts"] != [500] * 10:
                raise ValueError("The manifest does not match the paper's split protocol")
            if len(manifest["bundles"]) != 14 or any(len(b["indices"]) != 2000 for b in manifest["bundles"]):
                raise ValueError("The manifest does not contain fourteen 2000-image bundles")
        if any(b["class_counts"] != [200] * 10 for b in balanced["bundles"]):
            raise ValueError("Fully symmetric allocation is not balanced")
    tables["Table2_class_profiles.csv"] = [
        {"orbit": f"O{i}", **{f"class_{j}": count for j, count in enumerate(profiles[f"profile_{i}"])}}
        for i in range(1, 5)
    ]
    for dataset, model, _ in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run = source / "runs" / dataset / model / condition / f"seed_{seed:03d}"
                key = dataset, model, condition, seed
                terminal[key], details = terminal_statistics(reader.csv(run / "metrics/checkpoint_metrics.csv"))
                saved = reader.json(run / "metrics/terminal_metrics.json")
                if any(saved.get(k) != v for k, v in zip(("dataset", "model", "condition", "seed"), key)):
                    raise ValueError(f"Scientific identity mismatch: {key}")
                for metric, value in terminal[key].items():
                    if not np.isclose(float(saved[f"{metric}_bar"]), value, atol=1e-10, rtol=1e-10):
                        raise ValueError(f"Saved terminal metric mismatch: {key}, {metric}")
                lambdas[key] = float(saved["communication_lambda"])
                if condition != "main":
                    continue
                stability.extend({"dataset": dataset, "model": model, "seed": seed, **r} for r in details)
                matrix_path = run / "artifacts/pairwise_distance.npy"
                matrix = np.load(io.BytesIO(reader.read(matrix_path)), allow_pickle=False)
                pairwise_means(matrix)
                matrices[dataset, model, seed] = matrix
                geo = reader.csv(run / "metrics/orbit_geometry.csv")
                movement = reader.csv(run / "metrics/predictive_motion.csv")
                b = [float(r["value"]) for r in geo if r["metric"] == "B_kl" and int(r["epoch"]) == 200 and r["scope"] == "final_checkpoint"]
                row = {"dataset": dataset, "model": model, "seed": seed}
                for start in (175, 195):
                    t = [float(r["value"]) for r in movement if r["metric"] == "T_center_k" and int(r["previous_epoch"]) == start and int(r["epoch"]) == 200]
                    row[f"rho_{start}_200_percent"] = rho_percent(t, b)
                endpoints.append(row)
    selections, table4a, table4b, table5, table6 = [], [], [], [], []
    seed_geometry, seed_contrasts = [], []
    for dataset, model, _ in FAMILIES:
        identity = {"dataset": dataset, "model": model}
        selected = yaml.safe_load(reader.read(source / "pilots" / dataset / model / "selected_hparams.yaml"))
        chosen = float(selected["lambda_main"])
        if selected["selection_source"] != "balanced_validation_only" or selected["selection_outcome"] != "primary_pass":
            raise ValueError("The preserved selection does not match the paper's primary-rule statement")
        symmetric = [lambdas[dataset, model, "fully_symmetric", s] for s in SEEDS]
        for seed in SEEDS:
            if lambdas[dataset, model, "no_communication", seed] != 0:
                raise ValueError("No-communication run has nonzero lambda")
            if any(lambdas[dataset, model, c, seed] != chosen for c in ("main", "misaligned_data")):
                raise ValueError("Final communication lambda disagrees with the pilot selection")
        if len(set(symmetric)) != 1:
            raise ValueError("Fully symmetric lambda differs across seeds")
        selections.append({**identity, "lambda_main": chosen, "lambda_fully_symmetric": symmetric[0],
                           "pilot_seed": 0, "selection_source": selected["selection_source"],
                           "selection_outcome": selected["selection_outcome"]})
        arow, brow, crow = identity.copy(), identity.copy(), identity.copy()
        for metric in ("D_within", "D_between", "R_orbit"):
            arow[f"{metric}_mean"], arow[f"{metric}_sample_sd"] = mean_sd(
                [terminal[dataset, model, "main", s][metric] for s in SEEDS])
        geometry_rows, contrast_rows = [], []
        for seed in SEEDS:
            pw, pc, q = pairwise_means(matrices[dataset, model, seed])
            geometry_rows.append({**identity, "seed": seed, "P_within": pw, "P_cross": pc, "Q": q})
            a, b, c, d = [terminal[dataset, model, condition, seed] for condition in CONDITIONS]
            if min(b["D_within"], c["D_within"], a["D_total"]) <= 0:
                raise ValueError("Undefined paired relative reduction")
            contrast_rows.append({**identity, "seed": seed,
                                  "eta_B_percent": 100 * (1 - a["D_within"] / b["D_within"]),
                                  "eta_C_percent": 100 * (1 - a["D_within"] / c["D_within"]),
                                  "eta_D_percent": 100 * (1 - d["D_total"] / a["D_total"])})
        seed_geometry.extend(geometry_rows)
        seed_contrasts.extend(contrast_rows)
        for metric in ("P_within", "P_cross", "Q"):
            brow[f"{metric}_mean"], brow[f"{metric}_sample_sd"] = mean_sd([r[metric] for r in geometry_rows])
        for metric in ("eta_B_percent", "eta_C_percent", "eta_D_percent"):
            crow[f"{metric}_mean"], crow[f"{metric}_sample_sd"] = mean_sd([r[metric] for r in contrast_rows])
        st = [r for r in stability if r["dataset"] == dataset and r["model"] == model]
        ep = [r["rho_175_200_percent"] for r in endpoints if r["dataset"] == dataset and r["model"] == model]
        table6.append({**identity, "max_CV_percent": max(r["CV_percent"] for r in st),
                       "max_fitted_drift_percent": max(r["fitted_drift_percent"] for r in st),
                       "rho_175_200_min_percent": min(ep), "rho_175_200_max_percent": max(ep)})
        table4a.append(arow)
        table4b.append(brow)
        table5.append(crow)
    tables.update({"Table3_communication_coefficients.csv": selections,
                   "Table4A_terminal_geometry.csv": table4a, "Table4B_pairwise_geometry.csv": table4b,
                   "Table5_relative_reductions.csv": table5, "Table6_terminal_stability.csv": table6,
                   "main_stability_per_seed.csv": stability, "main_endpoint_motion_per_seed.csv": endpoints,
                   "main_pairwise_geometry_per_seed.csv": seed_geometry,
                   "relative_reductions_per_seed.csv": seed_contrasts,
                   "terminal_metrics_per_seed_condition.csv": [
                       dict(zip(("dataset", "model", "condition", "seed"), key), **value)
                       for key, value in terminal.items()]})
    return tables, matrices


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, folder: Path, stem: str) -> None:
    import matplotlib.pyplot as plt
    figure.savefig(folder / f"{stem}.pdf", bbox_inches="tight", pad_inches=.07,
                   metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.07)
    plt.close(figure)


def draw_figure2(matrices, folder: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    for index, ((dataset, model, title), axis) in enumerate(zip(FAMILIES, axes)):
        matrix = np.stack([matrices[dataset, model, s] for s in SEEDS]).mean(axis=0)
        image = axis.imshow(matrix, cmap="viridis", vmin=0)
        for boundary in (1.5, 4.5, 8.5):
            axis.axhline(boundary, color="#3670A0", linewidth=.8)
            axis.axvline(boundary, color="#3670A0", linewidth=.8)
        axis.set_title(f"({chr(97 + index)}) {title}", fontsize=10)
        axis.set(xticks=[0, 4, 9, 13], xticklabels=[1, 5, 10, 14],
                 yticks=[0, 4, 9, 13], yticklabels=[1, 5, 10, 14], xlabel="Agent i", ylabel="Agent j")
        fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
    fig.suptitle("Main: epoch 200 pairwise centered-logit squared distance; mean over training seeds 1, 2, 3")
    fig.tight_layout()
    save_figure(fig, folder, "Figure2_main_pairwise_distances_redrawn")


def reference_metadata(path: Path | None) -> dict | None:
    """Hash an optional reference without publishing a personal absolute path."""
    if path is None:
        return None
    raw = path.read_bytes()
    return {"name": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_readme(output: Path, tables, verified_count: int, manuscript: dict | None = None) -> None:
    links = "\n".join(f"- [{name}](tables/{name})" for name in tables)
    reference_note = (
        f"Optional manuscript reference: `{manuscript['name']}`. Its actual size and SHA256\n"
        "are recorded separately in `validation_summary.json`; its contents are not\n"
        "used for scientific calculations or compared with the generated figures."
        if manuscript is not None else
        "Optional manuscript reference: not provided. No manuscript file or hash was recorded."
    )
    text = f"""# Current-paper reproduction

Source: `{EXPORT_RELATIVE.as_posix()}`.
{reference_note}

Regeneration command, from the project root (choose a new or empty directory):

    python -B real_world_experiments/scripts/reproduce_paper.py --output real_world_experiments/paper_results/redraw

The command verified the export and checked hashes of every consumed package input.
`source_manifest.csv` records {verified_count} experiment-input/code files using project-relative paths.
This validates regeneration from exported evidence. It does not establish native
checkpoint completeness, rerun training, or restore omitted diagnostic snapshots.

## Figures

- [Figure 2 redraw](figures/Figure2_main_pairwise_distances_redrawn.pdf): four Main
  matrices, epoch 200, distances averaged over training seeds 1, 2, 3. Color scales
  are separate. This is a redraw, not a byte-identical copy of the manuscript's
  cropped/resized embedded images. Per-family mean matrices are in `data/`.
- [Figure 8](figures/Figure8_training_dynamics.pdf), manuscript page 44: the
  preserved S1 plotting functions, now writing only to this new directory.
- [Figure 9](figures/Figure9_late_centroid_motion.pdf), manuscript page 45: the
  preserved S2 plotting functions. Each adjacent interval (a,b) uses
  `100*sqrt(max_k T_k(a,b)/min_(k<l) B_kl(200))`: the denominator stays at epoch
  200. These are net endpoint displacements, not cumulative path lengths.

PNG companions are included. S3 is not a figure in the current manuscript.

## Tables

Tables 2–6 use the paper's definitions (pages 40, 41, 43, 44, 45). Table 4A averages
epochs 180,185,190,195,200 separately within each seed, including the checkpoint
ratios R_orbit. Table 4B uses 20 unordered within-orbit and 71 cross-orbit pairs,
with Q computed before averaging across seeds. All SDs use ddof=1.
Table 5 computes matched-seed relative reductions before averaging.
Table 6 takes the largest CV and fitted OLS drift across the three metrics and
three seeds, plus the three-seed range for the direct 175-to-200 displacement.
Drift is not the endpoint-change percentage in the old exported stability table.
Nodes and node pairs are not independent training replications.

{links}

The remaining CSVs retain seed-level calculations. Figure data and the fixed
epoch-200 denominator for Figure 9 are in `data/`. All source evidence remains
unchanged. This command does not download data, train models, or load checkpoints
for inference. The complete supplied export, including its checkpoints, must
remain available for file-hash verification.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def reproduce(output: Path, project: Path = PROJECT, *, paper: Path | None = None) -> dict:
    output = checked_output(output, project)
    paper_path = (project / paper.expanduser()).resolve() if paper is not None else None
    manuscript = reference_metadata(paper_path)
    from verify_export import verify_export
    source = project / EXPORT_RELATIVE
    verification = verify_export(source, deep=False)
    if not verification["valid"]:
        raise ValueError("Export verification failed; run real_world_experiments/scripts/verify_export.py for details")
    reader = SourceReader(source, project)
    reader.reference(Path(__file__).resolve())
    reader.reference(project / "real_world_experiments/scripts/verify_export.py")

    def check_inputs_unchanged() -> None:
        reader.check_unchanged()
        if reference_metadata(paper_path) != manuscript:
            raise ValueError("Optional manuscript reference changed during reproduction")

    tables, matrices = build_tables(reader)
    previous_mpl = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="orbit-paper-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            module, data, trajectories, motion, comparisons = load_plot_module(reader, output)
            check_inputs_unchanged()
            # Check again immediately before the first output mutation.
            output = checked_output(output, project)
            output.mkdir(parents=True, exist_ok=True)
            for name in ("tables", "data", "figures"):
                (output / name).mkdir()
            for name, rows in tables.items():
                write_csv(output / "tables" / name, rows)
            for name, rows in data.items():
                write_csv(output / "data" / name, rows)
            for dataset, model, _ in FAMILIES:
                matrix = np.stack([matrices[dataset, model, s] for s in SEEDS]).mean(axis=0)
                stem = f"{dataset}_{model}_main_pairwise_3seed_mean"
                np.save(output / "data" / f"{stem}.npy", matrix, allow_pickle=False)
                np.savetxt(output / "data" / f"{stem}.csv", matrix, delimiter=",")
            names = {"S1_training_dynamics": "Figure8_training_dynamics",
                     "S2_late_centroid_motion": "Figure9_late_centroid_motion"}
            module.save = lambda fig, stem: save_figure(fig, output / "figures", names[stem])
            with module.plt.rc_context({
                "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                "legend.fontsize": 7.5, "mathtext.fontset": "dejavusans",
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.linewidth": .65, "lines.linewidth": 1.45,
                "pdf.fonttype": 42, "ps.fonttype": 42,
                "savefig.facecolor": "white", "figure.facecolor": "white",
            }):
                draw_figure2(matrices, output / "figures")
                module.dynamics_figure(trajectories)
                module.motion_figure(motion)
            check_inputs_unchanged()
            write_csv(output / "source_manifest.csv", [reader.records[k] for k in sorted(reader.records)])
            report = {"source": EXPORT_RELATIVE.as_posix(),
                      "manuscript": manuscript["name"] if manuscript is not None else None,
                      "manuscript_status": "provided" if manuscript is not None else "not_provided",
                      "manuscript_reference": manuscript,
                      "validation_scope": "regeneration_from_exported_scientific_evidence",
                      "export_valid": True, "native_validation_complete": False,
                      "training_rerun": False, "source_files": len(reader.records),
                      "final_runs": len(trajectories), "training_seeds": list(SEEDS),
                      "epochs_per_run": 41, "main_motion_runs": len(motion),
                      "within_pairs": 20, "cross_pairs": 71,
                      "all_control_reductions_positive": all(r[k] > 0 for r in comparisons
                          for k in ("eta_B_percent", "eta_C_percent", "eta_D_percent")),
                      "all_Q_above_one": all(r["Q_epoch200"] > 1 for r in comparisons),
                      "all_terminal_stability_below_5_percent": all(
                          r["CV_percent"] < 5 and r["fitted_drift_percent"] < 5
                          for r in tables["main_stability_per_seed.csv"]),
                      "numpy_version": np.__version__, "matplotlib_version": module.matplotlib.__version__}
            (output / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            write_readme(output, tables, len(reader.records), manuscript)
            return report
        finally:
            if previous_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mpl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory; protected evidence paths are rejected")
    parser.add_argument("--paper", type=Path, help="Optional manuscript reference; records its name, size and SHA256 only. Relative paths use the project root.")
    args = parser.parse_args(argv)
    try:
        report = reproduce(args.output, paper=args.paper)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(1, f"Reproduction failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
