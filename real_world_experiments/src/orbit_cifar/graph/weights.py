"""Communication weights and spectral matching."""

from __future__ import annotations

import numpy as np


def metropolis_weights(adjacency: np.ndarray) -> np.ndarray:
    """Apply symmetric Metropolis weights to a loop-free adjacency matrix."""
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    if not np.allclose(np.diag(adjacency), 0.0):
        raise ValueError("adjacency must not contain self-loops")
    degrees = (adjacency != 0).sum(axis=1)
    weights = np.zeros_like(adjacency)
    rows, columns = np.nonzero(np.triu(adjacency, k=1))
    for i, j in zip(rows.tolist(), columns.tolist()):
        weight = 1.0 / (1.0 + max(int(degrees[i]), int(degrees[j])))
        weights[i, j] = weights[j, i] = weight
    return weights


def laplacian_from_weights(weights: np.ndarray) -> np.ndarray:
    """Construct ``diag(C 1) - C``."""
    weights = np.asarray(weights, dtype=np.float64)
    return np.diag(weights.sum(axis=1)) - weights


def spectral_match_lambda(lambda_main: float, main_laplacian: np.ndarray, target_laplacian: np.ndarray) -> float:
    """Match largest eigenvalues of two symmetric communication operators."""
    main_norm = float(np.linalg.eigvalsh(np.asarray(main_laplacian, dtype=np.float64)).max())
    target_norm = float(np.linalg.eigvalsh(np.asarray(target_laplacian, dtype=np.float64)).max())
    if target_norm <= 0:
        raise ValueError("Target Laplacian must have a positive spectral norm")
    return float(lambda_main) * main_norm / target_norm

