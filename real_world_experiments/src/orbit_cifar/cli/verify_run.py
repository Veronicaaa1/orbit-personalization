"""Verify required artifacts and numerical invariants for one or many runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.config_schema import load_config
from orbit_cifar.data.manifest import load_manifest
from orbit_cifar.data.datasets import load_dataset

REQUIRED_RUN_FILES = (
    "resolved_config.yaml",
    "run_status.json",
    "environment.json",
    "seed_plan.json",
    "data_assignment.json",
    "graph/adjacency.npy",
    "graph/communication_weights.npy",
    "graph/laplacian.npy",
    "graph/edge_list.csv",
    "graph/graph_summary.json",
    "checkpoints/last.pt",
    "checkpoints/final.pt",
    "metrics/checkpoint_metrics.csv",
    "metrics/train_diagnostics.csv",
    "metrics/node_classification.csv",
    "metrics/orbit_classification.csv",
    "metrics/terminal_metrics.json",
    "artifacts/final_raw_logits.pt",
    "artifacts/final_centered_logits.pt",
    "artifacts/final_labels.pt",
    "artifacts/pairwise_distance.npy",
    "artifacts/pairwise_distance.csv",
    "artifacts/random_partition_diagnostic.npz",
)

IMPORTED_REQUIRED_FILES = (
    "legacy_reference.json", "resolved_config.yaml", "run_status.json", "data_assignment.json",
    "graph/adjacency.npy", "graph/communication_weights.npy", "graph/laplacian.npy",
    "metrics/checkpoint_metrics.csv", "metrics/terminal_metrics.json", "metrics/node_classification.csv",
    "metrics/orbit_classification.csv", "artifacts/pairwise_distance.npy", "artifacts/pairwise_distance.csv",
)


def verify_run(directory: str | Path, tolerance: float = 1e-8) -> dict[str, Any]:
    """Verify structure, graph matrices, pairwise matrix, and checkpoint decompositions."""
    run = Path(directory).expanduser().resolve()
    imported = (run / "legacy_reference.json").is_file()
    resolved_path = run / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
    if imported or resolved.get("experiment"):
        from orbit_cifar.experiments.matrix import ScientificKey
        key = ScientificKey(str(resolved["data"]["name"]), str(resolved["model"]["name"]),
                            str(resolved["resolved_condition"]["name"]), int(resolved["seed"]))
        if imported:
            from orbit_cifar.legacy import validate_imported_run
            result = validate_imported_run(run, key)
        else:
            from orbit_cifar.experiments.completion import verify_completed_run
            result = verify_completed_run(run, key, expected_epochs=int(resolved["training"]["epochs"]))
        if not result["valid"]:
            raise ValueError(f"Run verification failed: {result['errors']}")
        return {"run": str(run), "valid": True, "source_type": "legacy_reused" if imported else "native",
                "evaluation_checkpoints": len(pd.read_csv(run / "metrics/checkpoint_metrics.csv"))}
    required = IMPORTED_REQUIRED_FILES if imported else REQUIRED_RUN_FILES
    if not imported and (run / "resolved_config.yaml").is_file():
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        if resolved.get("artifacts", {}).get("save_final_logits") is False:
            required = tuple(value for value in required if value not in {
                "artifacts/final_raw_logits.pt", "artifacts/final_centered_logits.pt"})
        if resolved.get("experiment"):
            required = (*required, "artifacts/confusion_matrices.npz")
    missing = [relative for relative in required if not (run / relative).is_file()]
    if missing:
        raise ValueError(f"Run {run} lacks required artifacts: {missing}")
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"Run is not completed: {status}")
    adjacency = np.load(run / "graph" / "adjacency.npy")
    weights = np.load(run / "graph" / "communication_weights.npy")
    laplacian = np.load(run / "graph" / "laplacian.npy")
    for name, matrix in (("adjacency", adjacency), ("weights", weights), ("laplacian", laplacian)):
        if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
            raise ValueError(f"{name} is not symmetric")
    if not np.allclose(laplacian.sum(axis=1), 0.0, atol=tolerance, rtol=0):
        raise ValueError("Laplacian rows do not sum to zero")
    metrics = pd.read_csv(run / "metrics" / "checkpoint_metrics.csv")
    error = (metrics["D_total"] - metrics["D_within"] - metrics["D_between"]).abs().max()
    if float(error) > tolerance:
        raise ValueError(f"Dispersion decomposition error exceeds tolerance: {error}")
    pairwise = np.load(run / "artifacts" / "pairwise_distance.npy")
    if not np.allclose(pairwise, pairwise.T, atol=tolerance, rtol=0) or not np.allclose(np.diag(pairwise), 0.0, atol=tolerance):
        raise ValueError("Pairwise distance matrix fails symmetry or diagonal checks")
    if float(pairwise.min()) < -tolerance:
        raise ValueError("Pairwise distance matrix contains a negative value")
    if imported:
        reference = json.loads((run / "legacy_reference.json").read_text(encoding="utf-8"))
        if reference.get("source_type") != "legacy_reused" or reference.get("import_status") != "validated":
            raise ValueError("Imported run lacks a validated legacy lineage record")
    elif (run / "resolved_config.yaml").is_file():
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        if resolved.get("experiment") and int(resolved.get("training", {}).get("epochs", 0)) == 200:
            terminal_epochs = metrics[np.isfinite(metrics["D_within"]) & np.isfinite(metrics["D_between"])]["epoch"].tail(5).astype(int).tolist()
            if terminal_epochs != [180, 185, 190, 195, 200]:
                raise ValueError(f"Final scheduled terminal evaluations are incomplete: {terminal_epochs}")
    return {"run": str(run), "valid": True, "source_type": "legacy_reused" if imported else "native",
            "evaluation_checkpoints": int(len(metrics))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Verify run artifacts and numerical invariants")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--verify-manifest", action="store_true")
    arguments, cfg = parse_config(parser, argv)
    positional_overrides = [str(value) for value in arguments.paths if "=" in str(value)]
    if positional_overrides:
        cfg = load_config(arguments.config, positional_overrides)
        arguments.paths = [value for value in arguments.paths if "=" not in str(value)]
    project_root = Path(str(cfg.project_root)).expanduser().resolve()
    if arguments.verify_manifest:
        if OmegaConf.select(cfg, "experiment.name"):
            data_root = Path(str(cfg.data.root)).expanduser()
            if not data_root.is_absolute():
                data_root = project_root / data_root
            dataset = load_dataset(str(cfg.data.name), data_root, train=True, download=bool(cfg.data.download))
            heterogeneous = load_manifest(
                project_root / str(cfg.manifest_root) / str(cfg.data.name) / "heterogeneous.json", targets=dataset.targets
            )
            balanced = load_manifest(project_root / str(cfg.manifest_root) / str(cfg.data.name) / "balanced.json", targets=dataset.targets)
            from orbit_cifar.data.manifest import validate_manifest_pair, validate_manifest_source

            validate_manifest_source(heterogeneous, dataset, str(cfg.data.name), str(cfg.data.dataset_version), int(cfg.data.split_seed))
            validate_manifest_pair(heterogeneous, balanced, dataset.targets, str(cfg.data.name))
        else:
            load_manifest(project_root / str(cfg.manifest_path))
        print("Split manifest structure and exclusivity verified")
    paths = arguments.paths
    if not paths:
        output_root = project_root / str(cfg.output_root) / "final"
        pattern = "*/*/*/seed_*/run_status.json" if OmegaConf.select(cfg, "experiment.name") else "*/seed_*/run_status.json"
        paths = sorted(path.parent for path in output_root.glob(pattern))
    if not paths and not arguments.verify_manifest:
        raise FileNotFoundError("No run directories were supplied or discovered")
    for path in paths:
        print(verify_run(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
