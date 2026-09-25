"""Persistence of graph artifacts required in each run directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orbit_cifar.graph.builders import GraphData
from orbit_cifar.graph.validation import validate_graph
from orbit_cifar.utils.io import atomic_write_json, ensure_directory


def save_graph_artifacts(graph: GraphData, directory: str | Path, extra_summary: dict[str, object] | None = None) -> None:
    """Validate and persist matrices, weighted edge list, and a JSON summary."""
    target = ensure_directory(directory)
    np.save(target / "adjacency.npy", graph.adjacency)
    np.save(target / "communication_weights.npy", graph.communication_weights)
    np.save(target / "laplacian.npy", graph.laplacian)
    pd.DataFrame(graph.edges, columns=["node_i", "node_j", "weight"]).to_csv(target / "edge_list.csv", index=False)
    summary = validate_graph(graph)
    summary.update(extra_summary or {})
    atomic_write_json(target / "graph_summary.json", summary)

