from __future__ import annotations

import numpy as np
import torch

from orbit_cifar.evaluation.random_partitions import (
    aggregate_random_partition_diagnostics,
    assignments_to_partition,
    generate_random_partitions,
    random_partition_diagnostic,
)


def test_random_partition_exact_sizes_and_determinism() -> None:
    first = generate_random_partitions(50, seed=22)
    second = generate_random_partitions(50, seed=22)
    assert np.array_equal(first, second)
    for assignment in first:
        assert [len(group) for group in assignments_to_partition(assignment)] == [2, 3, 4, 5]


def test_random_partition_diagnostic_and_aggregate() -> None:
    torch.manual_seed(5)
    logits = torch.randn(14, 7, 10)
    assignments = generate_random_partitions(30, seed=2)
    result = random_partition_diagnostic(logits, assignments)
    assert result["R_random"].shape == (30,)
    assert 0 < result["p_perm"] <= 1
    aggregate = aggregate_random_partition_diagnostics([result, result])
    assert aggregate["valid_seeds"] == 2
    assert aggregate["R_random_mean"].shape == (30,)

