"""Structural and symmetry checks for communication graphs."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np

from orbit_cifar.constants import REFERENCE_ORBITS
from orbit_cifar.graph.builders import GraphData


def _swap_permutation(size: int, left: int, right: int) -> np.ndarray:
    permutation = np.eye(size, dtype=np.float64)
    permutation[[left, right]] = permutation[[right, left]]
    return permutation


def validate_graph(
    graph: GraphData,
    require_reference_orbit_invariance: bool | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Validate symmetry, PSD, row sums, and optional orbit transpositions."""
    adjacency = graph.adjacency
    weights = graph.communication_weights
    laplacian = graph.laplacian
    n = adjacency.shape[0]
    for label, matrix in (("adjacency", adjacency), ("weights", weights), ("laplacian", laplacian)):
        if matrix.shape != (n, n):
            raise ValueError(f"{label} has inconsistent shape {matrix.shape}")
        if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
            raise ValueError(f"{label} is not symmetric")
    if not np.allclose(np.diag(adjacency), 0.0, atol=tolerance):
        raise ValueError("adjacency contains self-loops")
    row_sum_error = float(np.max(np.abs(laplacian.sum(axis=1))))
    if row_sum_error > tolerance:
        raise ValueError(f"Laplacian row sums are not zero; max error={row_sum_error}")
    eigenvalues = np.linalg.eigvalsh(laplacian)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(f"Laplacian is not positive semidefinite; min eigenvalue={eigenvalues.min()}")
    check_orbits = graph.name == "blown_up_path" if require_reference_orbit_invariance is None else require_reference_orbit_invariance
    if check_orbits:
        for orbit in REFERENCE_ORBITS:
            for left, right in combinations(orbit, 2):
                permutation = _swap_permutation(n, left, right)
                if not np.allclose(permutation @ adjacency @ permutation.T, adjacency, atol=tolerance, rtol=0):
                    raise ValueError(f"Adjacency is not invariant under swap ({left}, {right})")
                if not np.allclose(permutation @ weights @ permutation.T, weights, atol=tolerance, rtol=0):
                    raise ValueError(f"Weights are not invariant under swap ({left}, {right})")
    return {
        "name": graph.name,
        "num_nodes": n,
        "num_edges": len(graph.edges),
        "degrees": graph.degrees.tolist(),
        "spectral_norm": graph.spectral_norm,
        "laplacian_min_eigenvalue": float(eigenvalues.min()),
        "row_sum_max_error": row_sum_error,
        "reference_orbit_invariance_checked": bool(check_orbits),
    }


def validate_expected_main_graph(graph: GraphData) -> dict[str, Any]:
    """Apply the exact prescribed blown-up path checks."""
    summary = validate_graph(graph, require_reference_orbit_invariance=True)
    expected_degrees = [3, 3, 6, 6, 6, 8, 8, 8, 8, 4, 4, 4, 4, 4]
    if summary["num_edges"] != 38 or summary["degrees"] != expected_degrees:
        raise ValueError("The blown-up path does not have the prescribed edges and degrees")
    if not np.isclose(graph.communication_weights[0, 2], 1 / 7):
        raise ValueError("O1-O2 Metropolis weight must be 1/7")
    if not np.isclose(graph.communication_weights[2, 5], 1 / 9):
        raise ValueError("O2-O3 Metropolis weight must be 1/9")
    if not np.isclose(graph.communication_weights[5, 9], 1 / 9):
        raise ValueError("O3-O4 Metropolis weight must be 1/9")
    return summary

