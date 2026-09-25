from pathlib import Path
import argparse
import csv, hashlib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "real_world_experiments/paper_results/appendix"
SOURCE = ROOT / "real_world_experiments/results"
FAMILIES = [
    ("mnist", "mlp_ln", "MNIST / MLP-LN", "MNIST\nMLP-LN"),
    ("mnist", "resnet20_gn", "MNIST / ResNet-20-GN", "MNIST\nResNet-20-GN"),
    ("cifar10", "mlp_ln", "CIFAR-10 / MLP-LN", "CIFAR-10\nMLP-LN"),
    ("cifar10", "resnet20_gn", "CIFAR-10 / ResNet-20-GN", "CIFAR-10\nResNet-20-GN"),
]
SEEDS = (1, 2, 3)
CONDITIONS = ("main", "no_communication", "misaligned_data", "fully_symmetric")
EPOCHS = np.arange(0, 201, 5)
WINDOW = [180, 185, 190, 195, 200]
INTERVALS = [(175, 180), (180, 185), (185, 190), (190, 195), (195, 200)]
COLORS = ("#0072B2", "#D55E00", "#8B559D")
MARKERS = ("o", "s", "^")
INPUTS = {}

def path(ds, model, condition, seed, suffix):
    return SOURCE / "runs" / ds / model / condition / f"seed_{seed:03d}" / suffix

def track(p):
    INPUTS[p.relative_to(ROOT).as_posix()] = dict(
        bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return p

def read_csv(p):
    with track(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (OUT/name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def collect():
    trajectories, terminal, motion, comparisons = {}, {}, {}, []
    trajectory_rows, motion_rows = [], []
    membership = np.repeat(np.arange(4), [2, 3, 4, 5])
    i, j = np.triu_indices(14, 1)
    same = membership[i] == membership[j]
    assert same.sum() == 20 and (~same).sum() == 71
    for ds, model, _, _ in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                rows = read_csv(path(ds, model, condition, seed, "metrics/checkpoint_metrics.csv"))
                epochs = np.array([int(r["epoch"]) for r in rows])
                assert np.array_equal(epochs, EPOCHS)
                vals = {m: np.array([float(r[m]) for r in rows])
                        for m in ("D_within", "D_between", "D_total", "R_orbit")}
                assert all(np.isfinite(v).all() for v in vals.values())
                assert np.allclose(vals["D_total"], vals["D_within"]+vals["D_between"])
                key = ds, model, condition, seed
                trajectories[key] = vals
                mask = np.isin(epochs, WINDOW)
                assert mask.sum() == 5
                terminal[key] = {m: v[mask].mean() for m, v in vals.items()}
                for k, epoch in enumerate(epochs):
                    trajectory_rows.append(dict(dataset=ds, model=model, condition=condition,
                        seed=seed, epoch=int(epoch), **{m: v[k] for m, v in vals.items()}))
        for seed in SEEDS:
            geo = read_csv(path(ds, model, "main", seed, "metrics/orbit_geometry.csv"))
            centers = [r for r in geo if r["metric"] == "B_kl" and int(r["epoch"]) == 200
                       and r["scope"] == "final_checkpoint"]
            assert len(centers) == 6
            min_b = min(float(r["value"]) for r in centers)
            assert min_b > 0
            displacements = read_csv(path(ds, model, "main", seed, "metrics/predictive_motion.csv"))
            values = []
            for start, end in INTERVALS:
                selected = [r for r in displacements if r["metric"] == "T_center_k"
                    and r["scope"] == "adjacent_transition"
                    and int(r["previous_epoch"]) == start and int(r["epoch"]) == end]
                assert len(selected) == 4
                max_t = max(float(r["value"]) for r in selected)
                value = 100*np.sqrt(max_t/min_b)
                values.append(value)
                motion_rows.append(dict(dataset=ds, model=model, seed=seed, previous_epoch=start,
                    epoch=end, max_center_displacement_squared=max_t,
                    min_center_separation_squared_epoch200=min_b,
                    normalized_displacement_percent=value))
            motion[ds, model, seed] = np.array(values)
            matrix = np.load(track(path(ds, model, "main", seed, "artifacts/pairwise_distance.npy")))
            assert matrix.shape == (14, 14) and np.allclose(matrix, matrix.T)
            pw, pc = float(matrix[i[same], j[same]].mean()), float(matrix[i[~same], j[~same]].mean())
            a, b, c, d = [terminal[ds, model, cond, seed] for cond in CONDITIONS]
            comparisons.append(dict(dataset=ds, model=model, seed=seed,
                eta_B_percent=100*(1-a["D_within"]/b["D_within"]),
                eta_C_percent=100*(1-a["D_within"]/c["D_within"]),
                eta_D_percent=100*(1-d["D_total"]/a["D_total"]),
                P_within_epoch200=pw, P_cross_epoch200=pc, Q_epoch200=pc/pw))
    write_csv("training_dynamics_data.csv", trajectory_rows)
    write_csv("centroid_motion_data.csv", motion_rows)
    write_csv("terminal_comparisons_data.csv", comparisons)
    return trajectories, motion, comparisons

def decorate(ax, title):
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="y", color="#DDE1E5", linewidth=.5)
    ax.tick_params(direction="out", length=3, width=.6)
    ax.set_axisbelow(True)

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT/f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)

def dynamics_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.45))
    lines = [
        ("main", "D_within", COLORS[0], "-", r"Main: $D_{\rm within}$"),
        ("main", "D_between", COLORS[1], "-", r"Main: $D_{\rm between}$"),
        ("no_communication", "D_within", "#555B62", "--", r"No communication: $D_{\rm within}$"),
        ("misaligned_data", "D_within", COLORS[2], ":", r"Misaligned data: $D_{\rm within}$"),
    ]
    for k, ((ds, model, title, _), ax) in enumerate(zip(FAMILIES, axes.flat)):
        decorate(ax, f"({chr(97+k)}) {title}")
        ax.axvspan(180, 200, color="#CDD1D5", alpha=.3, linewidth=0)
        for cond, metric, color, style, label in lines:
            vals = np.stack([data[ds, model, cond, seed][metric] for seed in SEEDS])
            mean, sd = vals.mean(0), vals.std(0, ddof=1)
            if cond == "main":
                ax.fill_between(EPOCHS, np.maximum(0, mean-sd), mean+sd, color=color, alpha=.13, linewidth=0)
            ax.plot(EPOCHS, mean, color=color, linestyle=style, label=label)
        ax.set(xlim=(0, 200), ylim=(0, None), xticks=[0, 50, 100, 150, 200])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if k//2 == 1: ax.set_xlabel("Epoch")
        if k%2 == 0: ax.set_ylabel("Centered-logit dispersion")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, .026), columnspacing=2.3, handlelength=2.8)
    fig.text(.5, .005, "Main bands: sample SD across 3 seeds. Gray strip: epochs 180–200.",
             ha="center", fontsize=7)
    fig.subplots_adjust(left=.09, right=.99, bottom=.20, top=.945, hspace=.38, wspace=.28)
    save(fig, "S1_training_dynamics")

def motion_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.35), sharey=True)
    for k, ((ds, model, title, _), ax) in enumerate(zip(FAMILIES, axes.flat)):
        decorate(ax, f"({chr(97+k)}) {title}")
        for seed, color, marker in zip(SEEDS, COLORS, MARKERS):
            vals = data[ds, model, seed]
            assert (vals > 0).all()
            ax.plot(range(5), vals, color=color, marker=marker, markersize=4,
                    linewidth=1.2, label=f"Seed {seed}")
        ax.set_yscale("log")
        ax.set(ylim=(.035, 16), xlim=(-.15, 4.15))
        ax.set_xticks(range(5), [f"{a}–{b}" for a, b in INTERVALS], rotation=20, ha="right")
        ax.yaxis.set_major_locator(FixedLocator([.05, .1, .5, 1, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        if k//2 == 1: ax.set_xlabel("Epoch interval")
        if k%2 == 0: ax.set_ylabel("Centroid displacement / final\ncentroid separation (%, log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, .005))
    fig.subplots_adjust(left=.115, right=.99, bottom=.15, top=.945, hspace=.43, wspace=.20)
    save(fig, "S2_late_centroid_motion")

def terminal_figure(rows):
    fig, axes = plt.subplots(2, 2, figsize=(7.55, 5.65))
    panels = [
        ("eta_B_percent", "(a) Within-orbit reduction: Main vs. NoComm"),
        ("eta_C_percent", "(b) Within-orbit reduction: Main vs. Misaligned"),
        ("eta_D_percent", "(c) Global reduction: FullySym vs. Main"),
        ("Q_epoch200", "(d) Cross/within distance ratio: Main"),
    ]
    for ax, (metric, title) in zip(axes.flat, panels):
        decorate(ax, title)
        for k, (ds, model, _, _) in enumerate(FAMILIES):
            vals = np.array([r[metric] for r in sorted(rows, key=lambda r:r["seed"])
                             if r["dataset"] == ds and r["model"] == model])
            assert len(vals) == 3
            for offset, value, color, marker in zip([-.20, -.07, .06], vals, COLORS, MARKERS):
                ax.scatter(k+offset, value, c=color, marker=marker, s=23, zorder=4,
                           linewidths=.35, edgecolors="white")
            ax.errorbar(k+.23, vals.mean(), yerr=vals.std(ddof=1), color="#17191C", fmt="D",
                        markersize=3.7, markerfacecolor="white", capsize=3, linewidth=.9, zorder=5)
        ax.set_xlim(-.48, 3.48)
        ax.set_xticks(range(4), [f[3] for f in FAMILIES])
        ax.tick_params(axis="x", length=0, pad=5, labelsize=6.8)
        if metric.startswith("eta"):
            ax.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], ylabel="Reduction (%)")
        else:
            ax.axhline(1, color="#555B62", linestyle="--", linewidth=.9)
            ax.set(ylim=(.8, 3.3), yticks=[1, 1.5, 2, 2.5, 3], ylabel="Mean squared-distance ratio")
    handles = [Line2D([], [], c=c, marker=m, linestyle="None", markersize=4, label=f"Seed {s}")
               for s, c, m in zip(SEEDS, COLORS, MARKERS)]
    handles += [Line2D([], [], c="#17191C", marker="D", markerfacecolor="white", linestyle="-",
                       markersize=4, label="Mean ± sample SD")]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, .008), columnspacing=1.5)
    fig.subplots_adjust(left=.085, right=.99, bottom=.14, top=.95, hspace=.47, wspace=.28)
    save(fig, "S3_terminal_controls_and_separation")

def main(argv=None):
    from reproduce_paper import checked_output

    global OUT
    parser = argparse.ArgumentParser(description="Regenerate appendix displays S1-S3 from the supplied results.")
    parser.add_argument("--output", type=Path, default=OUT, help="New or empty output directory")
    args = parser.parse_args(argv)
    assert SOURCE.is_dir()
    OUT = checked_output(args.output, ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family":"DejaVu Sans", "font.size":8, "axes.titlesize":8.5, "axes.labelsize":8,
        "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7.5,
        "mathtext.fontset":"dejavusans", "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":.65, "lines.linewidth":1.45, "pdf.fonttype":42, "ps.fonttype":42,
        "savefig.facecolor":"white", "figure.facecolor":"white",
    })
    trajectories, motion, comparisons = collect()
    dynamics_figure(trajectories)
    motion_figure(motion)
    terminal_figure(comparisons)
    write_csv("source_manifest.csv", [{"path": k, **v} for k, v in sorted(INPUTS.items())])
    report = dict(source=str(SOURCE), input_files=len(INPUTS), final_runs=len(trajectories),
        epochs_per_run=len(EPOCHS), main_motion_runs=len(motion), motion_intervals_per_run=5,
        within_pairs=20, cross_pairs=71,
        motion_all_final_less_than_first=all(v[-1]<v[0] for v in motion.values()),
        pairwise_ratio_all_above_one=all(r["Q_epoch200"]>1 for r in comparisons),
        control_reductions_all_positive=all(r[k]>0 for r in comparisons
            for k in ("eta_B_percent","eta_C_percent","eta_D_percent")),
        matplotlib_version=matplotlib.__version__, numpy_version=np.__version__)
    (OUT/"validation_summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote 3 PDF/PNG figure pairs, plot-data CSVs and input provenance.")

if __name__ == "__main__":
    main()
