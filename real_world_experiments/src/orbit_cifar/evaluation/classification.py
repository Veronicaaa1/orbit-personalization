"""Per-node and orbit-logit-ensemble classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from orbit_cifar.constants import NUM_CLASSES, REFERENCE_ORBITS


def overall_classification_accuracies(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    partition: Sequence[Sequence[int]] = REFERENCE_ORBITS,
) -> tuple[float, float]:
    """Return mean node and mean orbit-logit-ensemble accuracy without full tables."""
    if raw_logits.ndim != 3 or raw_logits.shape[1] != labels.shape[0]:
        raise ValueError("raw_logits/labels have incompatible shapes")
    values = raw_logits.detach().to(device="cpu")
    targets = labels.detach().to(device="cpu", dtype=torch.long)
    node_accuracy = (values.argmax(dim=2) == targets.unsqueeze(0)).to(torch.float64).mean(dim=1)
    orbit_accuracies = []
    for orbit in partition:
        indices = torch.as_tensor(list(orbit), dtype=torch.long)
        ensemble = values.index_select(0, indices).mean(dim=0)
        orbit_accuracies.append((ensemble.argmax(dim=1) == targets).to(torch.float64).mean())
    return float(node_accuracy.mean()), float(torch.stack(orbit_accuracies).mean())


def predictor_classification(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> dict[str, Any]:
    """Compute accuracy, per-class CE/accuracy, and a row=true confusion matrix."""
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    targets = labels.detach().to(device="cpu", dtype=torch.long)
    if values.ndim != 2 or values.shape[0] != targets.shape[0] or values.shape[1] != num_classes:
        raise ValueError("Classification logits/labels have incompatible shapes")
    predictions = values.argmax(dim=1)
    losses = functional.cross_entropy(values, targets, reduction="none")
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    indices = targets * num_classes + predictions
    confusion.view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.int64))
    class_accuracy: list[float] = []
    class_ce: list[float] = []
    for class_id in range(num_classes):
        mask = targets == class_id
        if not bool(mask.any()):
            class_accuracy.append(float("nan"))
            class_ce.append(float("nan"))
        else:
            class_accuracy.append(float((predictions[mask] == targets[mask]).to(torch.float64).mean()))
            class_ce.append(float(losses[mask].mean()))
    return {
        "overall_accuracy": float((predictions == targets).to(torch.float64).mean()),
        "class_accuracy": class_accuracy,
        "class_cross_entropy": class_ce,
        "confusion_matrix": confusion.numpy(),
    }


def classification_tables(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    partition: Sequence[Sequence[int]] = REFERENCE_ORBITS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Create long-form tables for nodes and orbit-average logit predictors."""
    if raw_logits.ndim != 3:
        raise ValueError("raw_logits must have shape [node,image,class]")
    num_classes = int(raw_logits.shape[2])
    node_rows: list[dict[str, Any]] = []
    orbit_rows: list[dict[str, Any]] = []
    confusions: dict[str, np.ndarray] = {}
    for node in range(raw_logits.shape[0]):
        result = predictor_classification(raw_logits[node], labels, num_classes=num_classes)
        confusions[f"node_{node:02d}"] = result["confusion_matrix"]
        node_rows.append({"node": node, "node_label": node + 1, "class_id": -1, "metric": "overall_accuracy", "value": result["overall_accuracy"]})
        for class_id in range(num_classes):
            node_rows.append({"node": node, "node_label": node + 1, "class_id": class_id, "metric": "class_accuracy", "value": result["class_accuracy"][class_id]})
            node_rows.append({"node": node, "node_label": node + 1, "class_id": class_id, "metric": "class_cross_entropy", "value": result["class_cross_entropy"][class_id]})
    for orbit_id, orbit in enumerate(partition, start=1):
        ensemble_logits = raw_logits[torch.as_tensor(list(orbit), dtype=torch.long)].mean(dim=0)
        result = predictor_classification(ensemble_logits, labels, num_classes=num_classes)
        confusions[f"orbit_{orbit_id}"] = result["confusion_matrix"]
        orbit_rows.append({"orbit": orbit_id, "class_id": -1, "metric": "overall_accuracy", "value": result["overall_accuracy"]})
        for class_id in range(num_classes):
            orbit_rows.append({"orbit": orbit_id, "class_id": class_id, "metric": "class_accuracy", "value": result["class_accuracy"][class_id]})
            orbit_rows.append({"orbit": orbit_id, "class_id": class_id, "metric": "class_cross_entropy", "value": result["class_cross_entropy"][class_id]})
    return pd.DataFrame(node_rows), pd.DataFrame(orbit_rows), confusions




def mean_orbit_ensemble_accuracy(orbit_table: pd.DataFrame) -> float:
    """Average the four overall orbit-logit-ensemble accuracies."""
    rows = orbit_table[(orbit_table["metric"] == "overall_accuracy") & (orbit_table["class_id"] == -1)]
    if rows.empty:
        raise ValueError("Orbit table has no overall accuracy rows")
    return float(rows["value"].mean())
