"""Opt-in orchestration of an independent MNIST/CIFAR-10 reproduction.

The default only describes a plan. The archived FINAL directory is read-only.
Use the existing training, selection and completion implementations unchanged.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))

from omegaconf import OmegaConf

from orbit_cifar.config_schema import load_config
from orbit_cifar.experiments.matrix import final_keys, pilot_families

CONFIG = CODE_ROOT / "configs/reproduce_final.yaml"
ARCHIVE = CODE_ROOT / "results"
STAGES = ("data", "pilot", "select", "final", "verify", "analyze")


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def checked_work_root(value: str | Path) -> Path:
    root = project_path(value)
    allowed = (PROJECT / "reproduction").resolve()
    if PROJECT.is_relative_to(root) or (root.is_relative_to(PROJECT) and not root.is_relative_to(allowed)):
        raise ValueError("Work root must be under reproduction/ or outside the project, never an existing project tree")
    for name in ("outputs", "reports", "manifests", "data", "matplotlib-cache"):
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"Managed path escapes the independent work root: {name}")
    return root


def family_config(dataset: str, model: str, work_root: Path, data_root: Path, device: str = "auto"):
    cfg = load_config(CONFIG, [f"data.name={dataset}", f"model.name={model}", f"device={device}"])
    cfg.project_root = str(PROJECT)
    cfg.output_root = str(work_root / "outputs")
    cfg.report_root = str(work_root / "reports")
    cfg.manifest_root = str(work_root / "manifests")
    cfg.data.root = str(data_root)
    cfg.data.download = False
    overrides = cfg.reproduction.family_training_overrides.get(f"{dataset}/{model}", {})
    cfg.training = OmegaConf.merge(cfg.training, overrides)
    return cfg


def copy_manifest(source: Path, target: Path, expected_sha256: str) -> None:
    """Copy exact archived bytes; never replace an existing different manifest."""
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Archived manifest checksum mismatch: {source}")
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"Refusing to replace an existing different manifest: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)


def prepare_data(configs, download: bool, work_root: Path) -> None:
    from orbit_cifar.data.datasets import load_dataset
    from orbit_cifar.data.manifest import load_manifest, validate_manifest_pair, validate_manifest_source

    for dataset in dict.fromkeys(str(cfg.data.name) for cfg in configs):
        cfg = next(cfg for cfg in configs if str(cfg.data.name) == dataset)
        data_root = Path(str(cfg.data.root)).resolve()
        if download and not data_root.is_relative_to(work_root / "data"):
            raise ValueError("Downloads may only write work-root/data; an external --data-root is read-only")
        archived = ARCHIVE / "experiment_provenance/manifests" / dataset
        manifests = {kind: load_manifest(archived / f"{kind}.json") for kind in ("heterogeneous", "balanced")}
        # Reject modified archived bytes before downloading anything.
        for kind in manifests:
            name = f"{dataset}/{kind}.json"
            actual = hashlib.sha256((archived / f"{kind}.json").read_bytes()).hexdigest()
            if actual != cfg.reproduction.archived_manifest_sha256[name]:
                raise ValueError(f"Archived manifest checksum mismatch: {name}")
        training = load_dataset(dataset, data_root, train=True, download=download)
        load_dataset(dataset, data_root, train=False, download=download)
        for manifest in manifests.values():
            validate_manifest_source(manifest, training, dataset, str(cfg.data.dataset_version), int(cfg.data.split_seed))
        validate_manifest_pair(manifests["heterogeneous"], manifests["balanced"], training.targets, dataset)
        for kind in manifests:
            target = Path(str(cfg.manifest_root)) / dataset / f"{kind}.json"
            if not target.resolve().is_relative_to(work_root / "manifests"):
                raise ValueError("Manifest destination escapes the independent work root")
            copy_manifest(archived / f"{kind}.json", target, cfg.reproduction.archived_manifest_sha256[f"{dataset}/{kind}.json"])
        print(f"Validated official train/test files and retained archived split indices: {dataset}")


def verify_family(cfg, seed: int | None) -> int:
    import pandas as pd

    from orbit_cifar.experiments.completion import verify_completed_run
    from orbit_cifar.experiments.matrix import run_directory
    from orbit_cifar.experiments.provenance import load_selected
    from orbit_cifar.experiments.selection import _validate_active_pilot_rows

    pilot = Path(str(cfg.output_root)) / "pilot" / str(cfg.data.name) / str(cfg.model.name)
    # This also decodes completed pilot checkpoints and checks CSV/provenance.
    _validate_active_pilot_rows(cfg, pilot / "pilot_runs.csv", pd.read_csv(pilot / "pilot_runs.csv"))
    load_selected(cfg, pilot / "selected_hparams.yaml")
    count = 0
    for key in final_keys(cfg):
        if (key.dataset, key.model) != (str(cfg.data.name), str(cfg.model.name)) or (seed is not None and key.seed != seed):
            continue
        result = verify_completed_run(run_directory(cfg.output_root, key), key, cfg, expected_epochs=200)
        if not result["valid"]:
            raise ValueError(f"Native completion failed for {key}: {result['errors']}")
        count += 1
    return count


def execute_stage(stage: str, configs, args, work_root: Path) -> None:
    if stage == "data":
        prepare_data(configs, args.download, work_root)
    elif stage == "pilot":
        from orbit_cifar.experiments.pilot import run_pilot

        for cfg in configs:
            # Archive Pilot settings differ from CIFAR-10/MLP final settings.
            # Merge into a copy so --stage all retains the final-stage workers.
            pilot_cfg = OmegaConf.merge(cfg, {"training": cfg.reproduction.pilot_training_overrides})
            run_pilot(pilot_cfg)
    elif stage == "select":
        from orbit_cifar.experiments.selection import select_from_csv

        for cfg in configs:
            root = Path(str(cfg.output_root)) / "pilot" / str(cfg.data.name) / str(cfg.model.name)
            selected = select_from_csv(root / "pilot_runs.csv", root, cfg=cfg)
            if selected.get("lambda_main") is None:
                raise ValueError(f"No valid pilot candidate for {cfg.data.name}/{cfg.model.name}")
    elif stage == "final":
        from orbit_cifar.experiments.final import run_active_cells

        for cfg in configs:
            result = run_active_cells(cfg, str(cfg.data.name), str(cfg.model.name), args.seed)
            expected = len(cfg.final.conditions) * (1 if args.seed is not None else len(cfg.final.seeds))
            if len(result) != expected or "action" not in result or not result.action.isin(["trained", "resumed", "skipped"]).all():
                raise RuntimeError("Final execution returned blocked or incomplete cells; run verify before analysis")
            trained = result[result.action.isin(["trained", "resumed"])]
            if not trained.empty and ("numerically_stable" not in trained or not trained.numerically_stable.eq(True).all()):
                raise RuntimeError("Final training reported numerical instability; completion has not been verified")
    elif stage == "verify":
        count = sum(verify_family(cfg, args.seed) for cfg in configs)
        print(f"Verified {count} new native final cells and their pilot/selection evidence")
    elif stage == "analyze":
        # Keep Matplotlib's cache inside the independent work tree.
        os.environ["MPLCONFIGDIR"] = str(work_root / "matplotlib-cache")
        from orbit_cifar.experiments.planner import build_execution_plan

        cfg = configs[0]
        plan = build_execution_plan(cfg, write=False)
        if plan.summary["already_complete_native_cells"] != 48:
            raise ValueError("Analysis requires all 48 verified native reproduction cells; run verify first")
        from orbit_cifar.analysis.run import analyze

        analyze(cfg)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", *STAGES, "all"), default="plan")
    parser.add_argument("--execute", action="store_true", help="Actually execute the selected stage")
    parser.add_argument("--download", action="store_true", help="Allow official downloads during data/all only")
    parser.add_argument("--work-root", type=Path, default=Path("reproduction/real_world"))
    parser.add_argument("--data-root", type=Path, help="Read existing official data from this directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=("mnist", "cifar10"))
    parser.add_argument("--model", choices=("resnet20_gn", "mlp_ln"))
    parser.add_argument("--seed", type=int, choices=(1, 2, 3))
    args = parser.parse_args(argv)
    if args.download and args.stage not in ("data", "all"):
        parser.error("--download only applies to --stage data or all")
    if args.stage in ("all", "analyze") and any(value is not None for value in (args.dataset, args.model, args.seed)):
        parser.error("all/analyze uses the complete 48-cell matrix; use individual stages for subsets")
    if args.seed is not None and args.stage not in ("plan", "final", "verify"):
        parser.error("--seed only applies to plan/final/verify")
    work_root = checked_work_root(args.work_root)
    data_root = project_path(args.data_root) if args.data_root else work_root / "data"
    if args.download and not data_root.is_relative_to(work_root / "data"):
        parser.error("--download cannot write an existing/external --data-root; use work-root/data")
    base = load_config(CONFIG)
    families = [(dataset, model) for dataset, model in pilot_families(base)
                if (args.dataset is None or dataset == args.dataset) and (args.model is None or model == args.model)]
    configs = [family_config(dataset, model, work_root, data_root, args.device) for dataset, model in families]
    print(json.dumps({
        "scope": str(base.experiment.name), "archived_scope": str(base.reproduction.archived_scope),
        "historical_fingerprints_reused": False, "stage": args.stage,
        "execute": args.execute and args.stage != "plan", "work_root": str(work_root),
        "archive_root": str(ARCHIVE),
        "data_root": str(data_root), "download": args.download,
        "stages": list(STAGES) if args.stage in ("plan", "all") else [args.stage],
        "families": [{"dataset": str(cfg.data.name), "model": str(cfg.model.name),
                      "num_workers": int(cfg.training.num_workers), "persistent_workers": bool(cfg.training.persistent_workers),
                      "pilot_num_workers": int(cfg.reproduction.pilot_training_overrides.num_workers),
                      "pilot_persistent_workers": bool(cfg.reproduction.pilot_training_overrides.persistent_workers)} for cfg in configs],
        "pilot_candidates": len(configs) * len(base.pilot.lambda_grid),
        "final_cells": len(configs) * len(base.final.conditions) * (1 if args.seed else len(base.final.seeds)),
    }, indent=2))
    if not args.execute or args.stage == "plan":
        print("Plan only: no files created, no downloads or training. Add --execute to a named stage.")
        return 0
    for stage in STAGES if args.stage == "all" else (args.stage,):
        execute_stage(stage, configs, args, work_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
