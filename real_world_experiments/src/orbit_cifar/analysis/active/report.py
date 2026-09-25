"""Publish active tables, figures, and descriptive report prose."""

from pathlib import Path
from omegaconf import DictConfig
from orbit_cifar.utils.io import ensure_directory
from .contracts import ActiveAnalysisBundle
from .aggregate import aggregate_active
from .tables import _write_tables
from .figures import _family_figures
from .orbit_diagnostics import diagnostic_figures

def _summary(bundle: ActiveAnalysisBundle, cfg: DictConfig, path: Path) -> None:
    valid = int(bundle.run_inventory.valid.sum())
    imported = int((bundle.run_inventory.source_type == "legacy_reused").sum())
    lines = [
        "# MNIST/CIFAR-10 × ResNet/MLP empirical summary", "",
        "This report is generated only from available artifacts; empty figures are explicitly placeholders.", "",
        f"The fixed matrix has 48 cells (2 datasets × 2 model families × 4 conditions × 3 seeds); {valid} are currently valid and {48-valid} are missing, pending, blocked, or incomplete.", "",
        f"Registered CIFAR-10/ResNet legacy observations: {imported}. These are reused observations, not newly trained or independent replications of the earlier ten-seed study.", "",
        "Seeds 1–3 are fixed for computational reuse, not selected for favorable performance. Seed variability is conditional on fixed assignments and the official test set.", "",
        "ResNet versus MLP is a comparison of two model families that differ in several architectural and parameterization dimensions; it is not a one-factor causal ablation.", "",
        "Sample SD, standard error, and directional counts describe seed variability; no confidence intervals or multiplicity-adjusted inference are produced. Undefined comparisons remain explicit.", "",
        "Fully symmetric changes graph, balanced data assignment, and initialization centering together. Its primary quantity is D_global=D_total; O1–O4 are reference groups only.", "",
        "Function-space decomposition is a descriptive diagnostic and cannot establish parameter consensus, zero dispersion, an affine SDE, Hurwitz stability, or a Gaussian stationary limit.", "",
        "Terminal-window scalars average epochs 180, 185, 190, 195, and 200. Pairwise matrices and class accuracy/CE describe the final checkpoint, and are labeled separately. Raw dispersion must not be pooled across dataset/model families.",
    ]
    if valid == 0:
        lines.extend(["", "No valid observations are available; no empirical conclusions can be drawn."])
    if valid > 0 and not bundle.robustness_consistency_summary.empty:
        lines.extend(["", "## Exploratory directional patterns", ""])
        for row in bundle.robustness_consistency_summary.to_dict("records"):
            valid_directions = int(row["positive"] + row["negative"] + row["tied"])
            if valid_directions == 0:
                label = "undefined"
            elif row["positive"] == valid_directions:
                label = "supported in every available seed"
            elif row["positive"] == 0:
                label = "absent in available seeds"
            else:
                label = "mixed"
            lines.append(
                f"- {row['dataset']}/{row['model']} {row['check']}: {label}; "
                f"positive={row['positive']}, negative={row['negative']}, tied={row['tied']}, undefined={row['undefined']}."
            )
    warn = int((bundle.main_terminal_stability.status == "WARN").sum()) if not bundle.main_terminal_stability.empty else 0
    lines.extend(["", "## Interpretation notes", "",
                  f"Main terminal stability warnings: {warn}. WARN is descriptive and never excludes or replaces a seed.",
                  "Class heatmaps display available-seed means and retain seed-level values in tables; they do not imply seed-by-seed consistency.",
                  "Dispersion contrasts should be read alongside the accuracy, cross-entropy, and recorded logit-scale diagnostics. Cross-condition ratios are dimensionless but are not invariant to separate logit rescalings across conditions.",
                  "Positive aggregate D_between does not establish separation of every orbit-center pair, and D_between>D_within does not order every cross-orbit distance above every within-orbit distance."])
    lines.append("Dataset contrasts are descriptive: image content, shape, normalization, and augmentation change together; both datasets use ten classes and the same prescribed class-count profiles.")
    lines.extend(["", "## Orbit geometry and observed late movement", "",
                  "The new per-seed tables retain each of the four orbit radii and six center-pair squared distances. They describe finite-time alignment and center separation; small dispersion alone does not locate a predictor over time.",
                  "Motion tables label the exact adjacent epoch pairs from 175-180 through 195-200. Endpoint displacement is the direct 175-to-200 squared displacement, not the sum of adjacent squared movements. Relative center movement removes the common global displacement.",
                  "Temporal coverage is explicit for every Main seed. Old observations without retained predictions remain not_recorded; no seed is replaced. Post hoc reference-partition geometry is derived only from available final raw arrays and records source paths and hashes in the report tables.",
                  "A positive strict final node-pair gap is finite-sample separation evidence. A nonpositive gap is a valid result. None of these diagnostics establishes asymptotic convergence or changes training, selection, or seed inclusion."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_active(cfg: DictConfig) -> ActiveAnalysisBundle:
    project = Path(str(cfg.project_root)).expanduser().resolve()
    reports = ensure_directory(project / str(cfg.report_root))
    bundle = aggregate_active(cfg)
    _write_tables(bundle, reports)
    _family_figures(bundle, cfg, reports, int(cfg.analysis.figure_dpi))
    diagnostic_figures(bundle, cfg, reports)
    _summary(bundle, cfg, reports / "summary.md")
    return bundle
