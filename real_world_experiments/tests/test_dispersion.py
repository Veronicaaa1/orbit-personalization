from __future__ import annotations

import math

import numpy as np
import torch

from orbit_cifar.constants import REFERENCE_ORBITS
from orbit_cifar.evaluation.classification import classification_tables, overall_classification_accuracies
from orbit_cifar.evaluation.dispersion import dispersion_metrics, pairwise_distance_matrix
from orbit_cifar.evaluation.logits import center_logits


def test_centering_decomposition_and_pairwise_properties() -> None:
    torch.manual_seed(9)
    raw = torch.randn(14, 11, 10)
    centered = center_logits(raw)
    assert torch.allclose(centered.sum(-1), torch.zeros(14, 11, dtype=torch.float64), atol=1e-10)
    metrics = dispersion_metrics(centered)
    assert np.isclose(metrics["D_total"], metrics["D_within"] + metrics["D_between"])
    pairwise = pairwise_distance_matrix(centered)
    assert np.allclose(pairwise, pairwise.T)
    assert np.allclose(np.diag(pairwise), 0)
    assert pairwise.min() >= 0


def test_zero_within_and_undefined_consensus_ratio() -> None:
    torch.manual_seed(1)
    orbit_values = torch.randn(4, 5, 10, dtype=torch.float64)
    values = torch.empty(14, 5, 10, dtype=torch.float64)
    for orbit_id, orbit in enumerate(REFERENCE_ORBITS):
        values[list(orbit)] = orbit_values[orbit_id]
    assert dispersion_metrics(values)["D_within"] < 1e-14
    consensus = values[0].expand(14, -1, -1).clone()
    metrics = dispersion_metrics(consensus)
    assert metrics["D_within"] == 0
    assert metrics["D_between"] == 0
    assert metrics["ratio_undefined"]
    assert math.isnan(metrics["R_orbit"])


def test_changing_partition_changes_ratio() -> None:
    values = torch.zeros(14, 2, 10, dtype=torch.float64)
    for orbit_id, orbit in enumerate(REFERENCE_ORBITS):
        values[list(orbit), :, orbit_id] = float(orbit_id + 1)
    true_ratio = dispersion_metrics(values)["R_orbit"]
    alternative = ((0, 2), (1, 3, 5), (4, 6, 9, 10), (7, 8, 11, 12, 13))
    assert dispersion_metrics(values, alternative)["R_orbit"] != true_ratio


def test_lightweight_accuracy_matches_full_classification_tables() -> None:
    torch.manual_seed(17)
    raw = torch.randn(14, 23, 10)
    labels = torch.arange(23) % 10
    mean_node, mean_orbit = overall_classification_accuracies(raw, labels)
    node_table, orbit_table, _ = classification_tables(raw, labels)
    expected_node = node_table[(node_table["metric"] == "overall_accuracy") & (node_table["class_id"] == -1)]["value"].mean()
    expected_orbit = orbit_table[(orbit_table["metric"] == "overall_accuracy") & (orbit_table["class_id"] == -1)]["value"].mean()
    assert np.isclose(mean_node, expected_node)
    assert np.isclose(mean_orbit, expected_orbit)
