"""Builders for the prescribed undirected communication graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from orbit_cifar.constants import NUM_NODES, REFERENCE_ORBITS
from orbit_cifar.graph.weights import laplacian_from_weights, metropolis_weights


@dataclass(frozen=True)
class GraphData:
    name: str
    adjacency: np.ndarray
    communication_weights: np.ndarray
    laplacian: np.ndarray
    edges: tuple[tuple[int, int, float], ...]

    @property
    def degrees(self) -> np.ndarray:
        """Return unweighted node degrees."""
        return self.adjacency.sum(axis=1).astype(np.int64)

    @property
    def spectral_norm(self) -> float:
        """Return the largest eigenvalue of the symmetric Laplacian."""
        return float(np.linalg.eigvalsh(self.laplacian).max())


def _from_adjacency(name: str, adjacency: np.ndarray) -> GraphData:
    weights = metropolis_weights(adjacency)
    laplacian = laplacian_from_weights(weights)
    edges = tuple(
        (i, j, float(weights[i, j]))
        for i in range(adjacency.shape[0])
        for j in range(i + 1, adjacency.shape[0])
        if adjacency[i, j] != 0
    )
    return GraphData(name, adjacency, weights, laplacian, edges)


def build_blown_up_path(
    orbits: Sequence[Sequence[int]] = REFERENCE_ORBITS,
    num_nodes: int = NUM_NODES,
) -> GraphData:
    """Build complete bipartite blocks between adjacent reference orbits."""
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for left, right in zip(orbits[:-1], orbits[1:]):
        for i in left:
            for j in right:
                adjacency[i, j] = adjacency[j, i] = 1.0
    return _from_adjacency("blown_up_path", adjacency)


def build_complete_graph(num_nodes: int = NUM_NODES) -> GraphData:
    """Build the loop-free complete graph on ``num_nodes`` vertices."""
    if num_nodes < 2:
        raise ValueError("A complete communication graph needs at least two nodes")
    adjacency = np.ones((num_nodes, num_nodes), dtype=np.float64) - np.eye(num_nodes, dtype=np.float64)
    return _from_adjacency("complete", adjacency)


def build_graph(name: str) -> GraphData:
    if name == "blown_up_path":
        return build_blown_up_path()
    if name == "complete":
        return build_complete_graph()
    raise ValueError(f"Unknown graph name: {name}")
