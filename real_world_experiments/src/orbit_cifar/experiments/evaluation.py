"""Evaluation orchestration; all metric mathematics lives in evaluation/."""

from __future__ import annotations

import logging
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from orbit_cifar.evaluation.classification import classification_tables, overall_classification_accuracies
from orbit_cifar.evaluation.dispersion import dispersion_metrics
from orbit_cifar.evaluation.logits import center_logits, collect_logits

LOGGER = logging.getLogger(__name__)

def _evaluate(
    models: list[torch.nn.Module],
    probe_loader: DataLoader[Any],
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
    full_classification: bool = False,
    reference_partition_only: bool = False,
    diagnostics: Any = None,
    global_step: int = 0,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    evaluation_start = time.perf_counter()
    input_digest = None
    if diagnostics is not None:
        diagnostics.validate_loader(probe_loader)
        input_digest = hashlib.sha256()
    raw, labels = collect_logits(models, probe_loader, device, input_digest=input_digest)
    expected_raw_bytes = int(raw.numel() * raw.element_size())
    expected_centered_bytes = int(raw.numel() * 8)
    LOGGER.info(
        "Evaluation logits shape=%s; raw=%d bytes; float64 centered=%d bytes",
        tuple(raw.shape), expected_raw_bytes, expected_centered_bytes,
    )
    centered = center_logits(raw)
    metrics = dispersion_metrics(
        centered,
        decomposition_tolerance=float(cfg.training.decomposition_tolerance),
        ratio_threshold=float(cfg.training.ratio_threshold),
    )
    mean_node_accuracy, orbit_ensemble_accuracy = overall_classification_accuracies(raw, labels)
    if full_classification:
        node_table, orbit_table, confusions = classification_tables(raw, labels)
    else:
        node_table, orbit_table, confusions = pd.DataFrame(), pd.DataFrame(), {}
    metrics.update(
        {
            "epoch": int(epoch),
            "orbit_ensemble_accuracy": orbit_ensemble_accuracy,
            "mean_node_accuracy": mean_node_accuracy,
            "evaluation_raw_logit_bytes": expected_raw_bytes,
            "evaluation_centered_logit_bytes": expected_centered_bytes,
            "evaluation_wall_seconds": float(time.perf_counter() - evaluation_start),
            "raw_logit_rms": float(raw.square().mean().sqrt()),
            "centered_logit_rms": float(centered.square().mean().sqrt()),
        }
    )
    if reference_partition_only:
        metrics.update(
            {
                "D_global": metrics["D_total"],
                "partition_semantics": "reference_partition_only",
                "primary_disagreement_metric": "D_global=D_total",
            }
        )
        if not orbit_table.empty:
            orbit_table["group_semantics"] = "reference_partition_only"
            orbit_table["group_label"] = "reference_group"
    if diagnostics is not None:
        diagnostics.record(raw, labels, input_digest.hexdigest(), epoch, global_step, metrics)
    return metrics, raw, centered, labels, node_table, orbit_table, confusions


