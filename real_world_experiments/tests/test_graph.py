from __future__ import annotations

import numpy as np

from orbit_cifar.constants import ORBIT_SIZES, REFERENCE_ORBITS
from orbit_cifar.graph.builders import build_blown_up_path, build_complete_graph
from orbit_cifar.graph.validation import validate_expected_main_graph, validate_graph
from orbit_cifar.graph.weights import spectral_match_lambda


def test_blown_up_path_exact_structure_and_weights() -> None:
    graph = build_blown_up_path()
    summary = validate_expected_main_graph(graph)
    assert tuple(map(len, REFERENCE_ORBITS)) == ORBIT_SIZES == (2, 3, 4, 5)
    assert summary["num_edges"] == 38
    assert summary["degrees"] == [3, 3, 6, 6, 6, 8, 8, 8, 8, 4, 4, 4, 4, 4]
    assert np.isclose(graph.communication_weights[0, 2], 1 / 7)
    assert np.isclose(graph.communication_weights[2, 5], 1 / 9)
    assert np.isclose(graph.communication_weights[5, 9], 1 / 9)


def test_complete_graph_and_spectral_matching() -> None:
    main = build_blown_up_path()
    complete = build_complete_graph()
    validate_graph(complete, require_reference_orbit_invariance=False)
    assert len(complete.edges) == 91
    off_diagonal = complete.communication_weights[~np.eye(14, dtype=bool)]
    assert np.allclose(off_diagonal, 1 / 14)
    value = spectral_match_lambda(0.3, main.laplacian, complete.laplacian)
    expected = 0.3 * np.linalg.eigvalsh(main.laplacian).max() / np.linalg.eigvalsh(complete.laplacian).max()
    assert np.isclose(value, expected)


def test_complete_graph_is_invariant_under_arbitrary_node_permutations() -> None:
    adjacency = build_complete_graph().adjacency
    permutations = [
        np.array([1, 0, *range(2, 14)]),
        np.arange(13, -1, -1),
        np.array([3, 7, 11, 1, 9, 13, 5, 0, 12, 4, 8, 2, 10, 6]),
    ]
    for permutation in permutations:
        assert np.array_equal(adjacency[np.ix_(permutation, permutation)], adjacency)
