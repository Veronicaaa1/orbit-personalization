"""Versioned diagnostic publication bound to recoverable native checkpoints.

Tables are materialized views of the small snapshot in checkpoint.extra. Retained
arrays are immutable, hashed dependencies. A crash before checkpoint publication
can leave newer views; resume restores the committed snapshot without training.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, SequentialSampler, TensorDataset, default_collate

from orbit_cifar.constants import REFERENCE_ORBITS
from orbit_cifar.data.datasets import build_transforms
from orbit_cifar.data.loaders import _TransformDataset, _TransformSubset
from orbit_cifar.evaluation.orbit_diagnostics import (
    ADJACENT_PAIRS, PROFILE, RETAINED_EPOCHS, SCHEMA_VERSION, geometry, motion,
)
from orbit_cifar.evaluation.dispersion import validate_partition
from orbit_cifar.utils.io import atomic_torch_save, atomic_write_csv, atomic_write_json, config_fingerprint, read_json

LOGGER = logging.getLogger(__name__)
PREDICTIONS = "artifacts/terminal_predictions"
GEOMETRY = "metrics/orbit_geometry.csv"
MOTION = "metrics/predictive_motion.csv"
STATUS = "metrics/orbit_diagnostics_status.json"
METADATA = f"{PREDICTIONS}/probe_metadata.json"
LABELS = f"{PREDICTIONS}/labels.pt"
COLUMNS = ["schema_version", "dataset", "model", "condition", "seed", "epoch", "previous_epoch",
           "global_step", "previous_global_step", "group", "pair_group", "node", "metric", "value", "scope"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def probe_contract(loader, dataset, class_names, dataset_version):
    """Inspect ordering without iterating a loader or touching training minibatches.

Actual ordered, transformed image bytes are additionally hashed during the native
evaluation pass. This detects same-label image swaps and transform/source changes.
"""
    if (type(loader.sampler) is not SequentialSampler or type(loader.batch_sampler) is not BatchSampler
            or loader.drop_last or loader.collate_fn is not default_collate or not getattr(loader, "in_order", True)):
        raise ValueError("Diagnostics require the native sequential, complete probe ordering")
    source = loader.dataset
    if type(source) in {_TransformDataset, _TransformSubset}:
        indices = list(getattr(source, "indices", range(len(source))))
        transform = repr(source.transform)
        if transform != repr(build_transforms(False, dataset)):
            raise ValueError("Probe transforms differ from the deterministic native transforms")
        base = source.base
        source_type = f"{type(base).__module__}.{type(base).__qualname__}"
        split = "train" if getattr(base, "train", True) else "test"
    elif type(source) is TensorDataset:
        indices, transform, source_type, split = list(range(len(source))), "identity_tensor", "TensorDataset", "fixture"
    else:
        raise ValueError("Unknown probe dataset ordering contract")
    if len(indices) != len(set(indices)) or len(indices) != len(source):
        raise ValueError("Probe indices must be unique and complete")
    return {"dataset": dataset, "dataset_version": dataset_version, "source_type": source_type, "split": split,
            "ordered_indices": indices, "probe_size": len(source), "class_order": list(class_names),
            "transform": transform, "sampler": "SequentialSampler", "input_hash": "sha256_ordered_transformed_bytes_v1"}


def coverage(predictions):
    epochs = sorted(int(epoch) for epoch in predictions)
    pairs = [{"previous_epoch": previous, "epoch": epoch,
              "status": "recorded" if previous in epochs and epoch in epochs else "missing",
              "missing_epochs": [e for e in (previous, epoch) if e not in epochs]}
             for previous, epoch in ADJACENT_PAIRS]
    return {"temporal_coverage": "complete" if epochs == list(RETAINED_EPOCHS) else
            "partial_history" if epochs else "not_recorded", "retained_epochs": epochs, "pairs": pairs,
            "endpoint": {"previous_epoch": 175, "epoch": 200,
                         "status": "recorded" if 175 in epochs and 200 in epochs else "missing"}}


class OrbitDiagnosticRecorder:
    def __init__(self, root, run_identity, contract, groups=REFERENCE_ORBITS, *, final_epoch=200, cadence=5):
        self.root = Path(root)
        self.run_identity = copy.deepcopy(run_identity)
        self.contract = copy.deepcopy(contract)
        self.groups = [list(group) for group in groups]
        validate_partition(self.groups, sum(map(len, self.groups)))
        self.final_epoch, self.cadence = int(final_epoch), int(cadence)
        self.metadata = None
        self.geometry_rows, self.motion_rows, self.predictions = [], [], {}

    def validate_loader(self, loader):
        observed = probe_contract(loader, self.contract["dataset"], self.contract["class_order"],
                                  self.contract["dataset_version"])
        if observed != self.contract:
            raise ValueError("Probe ordering, source, classes, or transforms changed")

    def _rows(self, components, epoch, step, scope, previous=None, previous_step=None):
        key = {field: self.run_identity[field] for field in ("dataset", "model", "condition", "seed")}
        return [{**dict.fromkeys(COLUMNS), "schema_version": SCHEMA_VERSION, **key, "epoch": int(epoch),
                 "global_step": int(step), "previous_epoch": previous, "previous_global_step": previous_step,
                 "scope": scope, **item} for item in components]

    def record(self, raw, labels, input_digest, epoch, global_step, native_metrics):
        expected = [sum(map(len, self.groups)), self.contract["probe_size"], len(self.contract["class_order"])]
        if list(raw.shape) != expected or list(labels.shape) != [expected[1]]:
            raise ValueError("Prediction shape/class count differs from the probe contract")
        metadata = {"schema_version": SCHEMA_VERSION, "profile": PROFILE, "run_identity": self.run_identity,
                    "probe": self.contract, "groups": self.groups, "shape": expected, "dtype": str(raw.dtype),
                    "ordered_input_sha256": input_digest, "labels_sha256": tensor_hash(labels)}
        metadata["probe_id"] = config_fingerprint(metadata)
        if self.metadata is not None and metadata != self.metadata:
            raise ValueError("Wrong probe identity, ordered images, run, partition, or prediction dtype")
        if self.metadata is None:
            self.metadata = metadata
            atomic_torch_save(self.root / LABELS, labels.detach().cpu())
            atomic_write_json(self.root / METADATA, metadata)
            LOGGER.info("Orbit predictions: shape=%s dtype=%s; six raw arrays=%d bytes (plus serialization); float64 calculation=%d bytes",
                        expected, raw.dtype, raw.numel() * raw.element_size() * 6, raw.numel() * 8)
        components = geometry(raw, self.groups, strict=epoch == self.final_epoch)
        values = {row["metric"]: row["value"] for row in components}
        for name in ("D_within", "D_between"):
            if not np.isclose(values[f"{name}_reconstructed"], native_metrics[name], rtol=1e-9, atol=1e-12):
                raise ValueError(f"Independent {name} reconstruction disagrees with native evaluation")
        scope = "final_checkpoint" if epoch == self.final_epoch else "scheduled_evaluation"
        rows = self._rows(components, epoch, global_step, scope)
        existing = [row for row in self.geometry_rows if row["epoch"] == epoch]
        if existing and existing != rows:
            raise ValueError("Repeated evaluation changed committed geometry")
        self.geometry_rows = [row for row in self.geometry_rows if row["epoch"] != epoch] + rows
        if epoch not in RETAINED_EPOCHS:
            return
        name = f"{PREDICTIONS}/epoch_{epoch}.pt"
        payload = {"schema_version": SCHEMA_VERSION, "profile": PROFILE, "epoch": int(epoch),
                   "global_step": int(global_step), "run_identity": self.run_identity,
                   "probe_id": metadata["probe_id"], "shape": list(raw.shape), "dtype": str(raw.dtype),
                   "raw_sha256": tensor_hash(raw), "raw_logits": raw.detach().cpu()}
        if str(epoch) in self.predictions:
            old = self._load(epoch)
            if old["raw_sha256"] != payload["raw_sha256"] or old["global_step"] != global_step:
                raise ValueError("Repeated evaluation changed retained predictions")
        else:
            atomic_torch_save(self.root / name, payload)
            self.predictions[str(epoch)] = {"path": name, "sha256": file_hash(self.root / name),
                                            "global_step": int(global_step), "raw_sha256": payload["raw_sha256"]}
        self.motion_rows = [row for row in self.motion_rows if row["epoch"] != epoch]
        pairs = [(previous, "adjacent_transition") for previous, current in ADJACENT_PAIRS if current == epoch]
        if epoch == 200:
            pairs.append((175, "endpoint_displacement"))
        for previous, motion_scope in pairs:
            if str(previous) in self.predictions:
                old = self._load(previous)
                self.motion_rows += self._rows(motion(old["raw_logits"], raw, self.groups), epoch, global_step,
                                               motion_scope, previous, old["global_step"])

    def _load(self, epoch):
        item = self.predictions[str(epoch)]
        expected_path = f"{PREDICTIONS}/epoch_{epoch}.pt"
        if item["path"] != expected_path or file_hash(self.root / expected_path) != item["sha256"]:
            raise ValueError("Retained prediction file integrity mismatch")
        value = torch.load(self.root / expected_path, map_location="cpu", weights_only=True)
        raw = value["raw_logits"]
        if (value.get("schema_version") != SCHEMA_VERSION or value.get("profile") != PROFILE
                or value.get("run_identity") != self.run_identity or value.get("probe_id") != self.metadata["probe_id"]
                or value.get("epoch") != epoch or value.get("global_step") != item["global_step"]
                or value.get("shape") != self.metadata["shape"] or list(raw.shape) != value["shape"]
                or value.get("dtype") != self.metadata["dtype"] or str(raw.dtype) != value["dtype"]
                or tensor_hash(raw) != value.get("raw_sha256") or value["raw_sha256"] != item["raw_sha256"]):
            raise ValueError("Retained prediction identity/shape/precision mismatch")
        return value

    def publish(self, epoch, global_step):
        """Write dependencies first; caller atomically embeds return value in checkpoint."""
        if self.metadata is None:
            raise ValueError("No evaluated probe metadata exists")
        self.geometry_rows.sort(key=lambda row: row["epoch"])
        self.motion_rows.sort(key=lambda row: (row["epoch"], row["scope"]))
        status = {"schema_version": SCHEMA_VERSION, "profile": PROFILE, "run_identity": self.run_identity,
                  "probe_id": self.metadata["probe_id"], "epoch": int(epoch), "global_step": int(global_step),
                  "geometry_epochs": sorted({row["epoch"] for row in self.geometry_rows}), **coverage(self.predictions)}
        atomic_write_csv(self.root / GEOMETRY, pd.DataFrame(self.geometry_rows, columns=COLUMNS))
        atomic_write_csv(self.root / MOTION, pd.DataFrame(self.motion_rows, columns=COLUMNS))
        atomic_write_json(self.root / STATUS, status)
        files = {name: file_hash(self.root / name) for name in (GEOMETRY, MOTION, STATUS, METADATA, LABELS)}
        files.update({item["path"]: item["sha256"] for item in self.predictions.values()})
        snapshot = {"schema_version": SCHEMA_VERSION, "profile": PROFILE, "epoch": int(epoch),
                    "global_step": int(global_step), "metadata": self.metadata, "geometry": self.geometry_rows,
                    "motion": self.motion_rows, "predictions": self.predictions, "files": files}
        return copy.deepcopy({**snapshot, "generation": config_fingerprint(snapshot)})

    def restore(self, snapshot, epoch, global_step, *, repair=True):
        """Recover only committed diagnostic state; never change training epoch/state."""
        value = copy.deepcopy(snapshot)
        generation = value.pop("generation")
        if (generation != config_fingerprint(value) or value["schema_version"] != SCHEMA_VERSION
                or value["profile"] != PROFILE or value["epoch"] != epoch or value["global_step"] != global_step):
            raise ValueError("Diagnostic checkpoint generation mismatch")
        metadata = value["metadata"]
        meta_base = {key: item for key, item in metadata.items() if key != "probe_id"}
        if (metadata["probe_id"] != config_fingerprint(meta_base) or metadata["run_identity"] != self.run_identity
                or metadata["probe"] != self.contract or metadata["groups"] != self.groups):
            raise ValueError("Diagnostic probe/run/partition identity mismatch")
        self.metadata = metadata
        for name in (METADATA, LABELS):
            if file_hash(self.root / name) != value["files"][name]:
                raise ValueError("Committed probe metadata/labels integrity mismatch")
        if read_json(self.root / METADATA) != metadata:
            raise ValueError("Published probe metadata differs from checkpoint")
        labels = torch.load(self.root / LABELS, map_location="cpu", weights_only=True)
        if tensor_hash(labels) != metadata["labels_sha256"]:
            raise ValueError("Saved labels differ from probe metadata")
        self.geometry_rows, self.motion_rows, self.predictions = value["geometry"], value["motion"], value["predictions"]
        required = {GEOMETRY, MOTION, STATUS, METADATA, LABELS}
        required.update(f"{PREDICTIONS}/epoch_{e}.pt" for e in self.predictions)
        if set(value["files"]) != required:
            raise ValueError("Diagnostic checkpoint dependency manifest is incomplete")
        if any(int(e) > epoch or int(e) not in RETAINED_EPOCHS for e in self.predictions):
            raise ValueError("Uncommitted retained prediction in checkpoint")
        for retained in self.predictions:
            self._load(int(retained))
        expected_epochs = list(range(0, epoch + 1, self.cadence))
        if epoch == self.final_epoch and epoch not in expected_epochs:
            expected_epochs.append(epoch)
        if sorted({row["epoch"] for row in self.geometry_rows}) != expected_epochs:
            raise ValueError("Committed geometry history is incomplete")
        key = {field: self.run_identity[field] for field in ("dataset", "model", "condition", "seed")}
        for rows in (self.geometry_rows, self.motion_rows):
            keys = set()
            for row in rows:
                row_key = tuple(row.get(field) for field in ("epoch", "previous_epoch", "group", "pair_group", "node", "metric", "scope"))
                if (set(row) != set(COLUMNS) or row["schema_version"] != SCHEMA_VERSION
                        or any(row[field] != item for field, item in key.items())
                        or row["epoch"] > epoch or row["global_step"] > global_step or row_key in keys):
                    raise ValueError("Invalid or duplicate diagnostic table row")
                keys.add(row_key)
        for evaluated in expected_epochs:
            rows = [row for row in self.geometry_rows if row["epoch"] == evaluated]
            expected_count = len(self.groups) + len(self.groups) * (len(self.groups) - 1) // 2 + 4
            if len(rows) != expected_count + (3 if evaluated == self.final_epoch else 0):
                raise ValueError("Missing orbit/pair components in committed geometry")
        for previous, current in (*ADJACENT_PAIRS, (175, 200)):
            scope = "endpoint_displacement" if (previous, current) == (175, 200) else "adjacent_transition"
            rows = [row for row in self.motion_rows if row["previous_epoch"] == previous and row["epoch"] == current
                    and row["scope"] == scope]
            expected_count = sum(map(len, self.groups)) + 2 * len(self.groups) + 3
            if len(rows) != (expected_count if str(previous) in self.predictions and str(current) in self.predictions else 0):
                raise ValueError("Missing or bridged temporal pair in committed diagnostics")
        if repair:
            for retained in RETAINED_EPOCHS:
                path = self.root / PREDICTIONS / f"epoch_{retained}.pt"
                if str(retained) not in self.predictions and path.is_file():
                    path.unlink()  # exact uncommitted diagnostic artifact, never a model checkpoint
            restored = self.publish(epoch, global_step)
            if restored != snapshot:
                raise ValueError("Restored diagnostic publication differs from committed generation")


def verify_diagnostics(root, checkpoint, resolved, *, completed):
    """Explicit opt-in profile: old native/imported completion remains unchanged."""
    profile = resolved.get("orbit_diagnostics_profile")
    snapshot = checkpoint.get("extra", {}).get("orbit_diagnostics")
    if profile is None:
        if snapshot is not None:
            raise ValueError("Undeclared diagnostic checkpoint profile")
        return {"temporal_coverage": "not_recorded"}
    if profile != PROFILE or snapshot is None:
        raise ValueError("Unsupported or missing diagnostic checkpoint profile")
    identity = {"dataset": resolved["data"]["name"], "model": resolved["model"]["name"],
                "condition": resolved["resolved_condition"]["name"], "seed": resolved["seed"],
                "stage": resolved["stage"], "scientific_fingerprint": checkpoint["config_fingerprint"]}
    metadata = snapshot["metadata"]
    if identity["condition"] != "main" or identity["stage"] != "final":
        raise ValueError("Enhanced diagnostic profile is reserved for Main/final")
    probe = metadata["probe"]
    if (probe["dataset"] != identity["dataset"] or len(probe["class_order"]) != int(resolved["model"]["num_classes"])
            or metadata["shape"] != [sum(map(len, REFERENCE_ORBITS)), probe["probe_size"], len(probe["class_order"])]):
        raise ValueError("Diagnostic probe differs from resolved dataset/class shape")
    scientific = resolved.get("scientific_identity")
    if scientific is not None and (probe["class_order"] != scientific["dataset"]["class_names"]
                                   or probe["dataset_version"] != scientific["dataset_version"]
                                   or probe["split"] != "test"):
        raise ValueError("Diagnostic probe differs from native scientific source/class identity")
    recorder = OrbitDiagnosticRecorder(root, identity, metadata["probe"], final_epoch=int(resolved["training"]["epochs"]),
                                       cadence=int(resolved["training"]["eval_every_epochs"]))
    recorder.restore(snapshot, checkpoint["epoch"], checkpoint["global_step"], repair=False)
    expected_retained = [e for e in RETAINED_EPOCHS if e <= checkpoint["epoch"]]
    if sorted(map(int, recorder.predictions)) != expected_retained:
        raise ValueError("Enhanced profile is missing required terminal predictions")
    if completed:
        native_labels = torch.load(Path(root) / "artifacts/final_labels.pt", map_location="cpu", weights_only=True)
        if tensor_hash(native_labels) != metadata["labels_sha256"]:
            raise ValueError("Diagnostic probe labels differ from the native final probe")
        for name, digest in snapshot["files"].items():
            path = (Path(root) / name).resolve()
            if not path.is_relative_to(Path(root).resolve()) or file_hash(path) != digest:
                raise ValueError(f"Diagnostic completion generation mismatch: {name}")
    return coverage(recorder.predictions)
