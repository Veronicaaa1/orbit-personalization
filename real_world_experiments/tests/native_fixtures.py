from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from orbit_cifar.config_schema import config_to_dict, load_config
from orbit_cifar.constants import NUM_NODES, REFERENCE_ORBITS
from orbit_cifar.experiments.completion import publish_completion, REQUIRED_FILES
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.families import family_config
from orbit_cifar.experiments.matrix import ScientificKey
from orbit_cifar.experiments.provenance import build_scientific_identity
from orbit_cifar.utils.io import atomic_write_json, atomic_write_yaml, config_fingerprint
from orbit_cifar.utils.reproducibility import capture_rng_state
from orbit_cifar.data.partition import build_dataset_manifest
from orbit_cifar.data.manifest import save_manifest
from orbit_cifar.training.seed_plan import build_seed_plan

CONFIG = Path(__file__).parents[1] / "configs/mnist_cifar10_resnet_mlp_3seed.yaml"


def family(tmp_path: Path):
    cfg = family_config(load_config(CONFIG), "mnist", "mlp_ln")
    cfg.project_root = str(tmp_path)
    return cfg


def probe_manifest():
    return {"dataset": "mnist", "class_names": list(map(str, range(10))), "validation_indices": list(range(20)),
            "source_labels_checksum": config_fingerprint(list(range(10)))}


def native_fixture(tmp_path: Path, stage="final", epochs=200, value=0.0):
    cfg = family(tmp_path)
    cfg.stage = stage
    cfg.training.epochs = epochs
    cfg.training.terminal_window_checkpoints = 5 if stage == "final" else 3
    key = ScientificKey("mnist", "mlp_ln", "no_communication" if stage == "final" else "main", 1 if stage == "final" else 0)
    run_name = key.condition if stage == "final" else f"lambda_{value}"
    root = tmp_path / str(cfg.output_root) / stage / key.dataset / key.model / run_name / f"seed_{key.seed:03d}"
    manifest = build_dataset_manifest("mnist", np.tile(np.arange(10), 6000))
    save_manifest(manifest, tmp_path / str(cfg.manifest_root) / "mnist/heterogeneous.json")
    condition, graph = resolve_condition(key.condition, lambda_override=value)
    identity = build_scientific_identity(cfg, condition, key.seed, manifest, manifest["class_names"])
    fingerprint = config_fingerprint(identity)
    for relative in REQUIRED_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    resolved = {**config_to_dict(cfg), "seed": key.seed, "resolved_condition": condition.to_dict(),
                "scientific_identity": identity, "scientific_fingerprint": fingerprint}
    atomic_write_yaml(root / "resolved_config.yaml", resolved)
    metrics = pd.DataFrame([{"epoch": epoch, "D_within": 1.0, "D_between": 2.0, "D_total": 3.0,
                             "R_orbit": 2 / 3, "orbit_ensemble_accuracy": 0.7, "mean_node_accuracy": 0.6,
                             "raw_logit_rms": 0.5, "centered_logit_rms": 0.3}
                            for epoch in range(0, epochs + 1, 5)])
    metrics.to_csv(root / "metrics/checkpoint_metrics.csv", index=False)
    pd.DataFrame({"epoch": range(epochs), "global_step": range(1, epochs + 1)}).to_csv(root / "metrics/train_diagnostics.csv", index=False)
    terminal = {**key.to_dict(), "D_within_bar": 1.0, "D_between_bar": 2.0, "D_total_bar": 3.0,
                "R_orbit_bar": 2 / 3, "orbit_ensemble_accuracy": 0.7, "mean_node_accuracy": 0.6,
                "raw_logit_rms": 0.5, "centered_logit_rms": 0.3, "numerically_stable": True,
                "final_mean_node_accuracy": 0.6, "final_orbit_ensemble_accuracy": 0.7,
                "final_raw_logit_rms": 0.5, "final_centered_logit_rms": 0.3,
                "terminal_window_short": False, "probe_split": "official_test" if stage == "final" else "validation",
                "communication_lambda": value}
    atomic_write_json(root / "metrics/terminal_metrics.json", terminal)
    atomic_write_json(root / "seed_plan.json", build_seed_plan(key.seed, namespace=stage))
    atomic_write_json(root / "data_assignment.json", {"dataset": key.dataset, "manifest_fingerprint": config_fingerprint(manifest)})
    for kind, indices, accuracy in (("node", range(NUM_NODES), 0.6),
                                    ("orbit", range(1, len(REFERENCE_ORBITS) + 1), 0.7)):
        rows = []
        for index in indices:
            rows.append({kind: index, "class_id": -1, "metric": "overall_accuracy", "value": accuracy})
            for label in range(10):
                rows.extend([{kind: index, "class_id": label, "metric": "class_accuracy", "value": accuracy},
                             {kind: index, "class_id": label, "metric": "class_cross_entropy", "value": 1.0}])
        pd.DataFrame(rows).to_csv(root / f"metrics/{kind}_classification.csv", index=False)
    confusions = {(f"node_{index:02d}" if kind == "node" else f"orbit_{index}"):
                  np.eye(10, dtype=np.int64) * correct + np.roll(np.eye(10, dtype=np.int64), 1, axis=1) * (1000 - correct)
                  for kind, indices, correct in (("node", range(NUM_NODES), 600),
                                                 ("orbit", range(1, len(REFERENCE_ORBITS) + 1), 700)) for index in indices}
    np.savez_compressed(root / "artifacts/confusion_matrices.npz", **confusions)
    torch.save(torch.arange(10000) % 10, root / "artifacts/final_labels.pt")
    np.savez_compressed(root / "artifacts/random_partition_diagnostic.npz", R_true=2 / 3,
                        R_random=np.linspace(0.1, 0.9, 20), p_perm=0.3)
    np.save(root / "artifacts/pairwise_distance.npy", np.zeros((14, 14)))
    pd.DataFrame(np.zeros((14, 14)), index=range(1, 15), columns=range(1, 15)).to_csv(root / "artifacts/pairwise_distance.csv", index_label="node_label")
    np.save(root / "graph/adjacency.npy", graph.adjacency)
    np.save(root / "graph/laplacian.npy", graph.laplacian)
    np.save(root / "graph/communication_weights.npy", graph.communication_weights)
    checkpoint = {"epoch": epochs, "global_step": epochs, "config_fingerprint": fingerprint,
                  "models": [{"weight": torch.tensor([float(index)])} for index in range(14)],
                  "optimizers": [{"state": {}, "param_groups": [{"lr": 0.0005, "params": [0]}]} for _ in range(14)],
                  "scheduler": {"epoch": epochs}, "rng_state": capture_rng_state(),
                  "extra": {"terminal_metrics": terminal}}
    torch.save(checkpoint, root / "checkpoints/last.pt")
    torch.save(checkpoint, root / "checkpoints/final.pt")
    publication = publish_completion(root, key, fingerprint, epochs, epochs)
    atomic_write_json(root / "run_status.json", {"status": "completed", "numerically_stable": True,
                                               "last_valid_epoch": epochs, **publication})
    return cfg, key, root, manifest


