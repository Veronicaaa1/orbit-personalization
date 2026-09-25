from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from orbit_cifar.graph.builders import GraphData, build_blown_up_path, build_complete_graph
from orbit_cifar.graph.weights import spectral_match_lambda

FULLY_SYMMETRIC_LAMBDA_MAIN = 0.1


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    graph_name: str
    assignment: str
    communication_lambda: float
    extrapolated_large_lambda: bool = False
    lambda_main: float | None = None
    main_laplacian_spectral_norm: float | None = None
    active_laplacian_spectral_norm: float | None = None
    spectral_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_condition(
    name: str,
    selected_hparams: dict[str, Any] | None = None,
    lambda_override: float | None = None,
) -> tuple[ConditionSpec, GraphData]:
    """Resolve graph, assignment, and actual lambda without observing test metrics."""
    selected = selected_hparams or {}
    lambda_main = float(
        selected.get("lambda_main", FULLY_SYMMETRIC_LAMBDA_MAIN if name == "fully_symmetric" else 0.0)
    )
    lambda_large = float(selected.get("lambda_large", 10.0 * lambda_main))
    extrapolated = bool(selected.get("extrapolated_large_lambda", "lambda_large" not in selected))
    main_graph = build_blown_up_path()
    if name == "main":
        graph, assignment, value = main_graph, "aligned", lambda_main
    elif name == "no_communication":
        graph, assignment, value = main_graph, "aligned", 0.0
    elif name == "misaligned_data":
        graph, assignment, value = main_graph, "misaligned", lambda_main
    elif name in {"complete_graph", "fully_symmetric"}:
        graph = build_complete_graph()
        assignment = "aligned" if name == "complete_graph" else "fully_symmetric"
        value = spectral_match_lambda(lambda_main, main_graph.laplacian, graph.laplacian)
    elif name == "large_communication":
        graph, assignment, value = main_graph, "aligned", lambda_large
    else:
        raise ValueError(f"Unknown experimental condition: {name}")
    if lambda_override is not None:
        value = float(lambda_override)
    spec = ConditionSpec(
        name=name,
        graph_name=graph.name,
        assignment=assignment,
        communication_lambda=float(value),
        extrapolated_large_lambda=extrapolated if name == "large_communication" else False,
        lambda_main=lambda_main,
        main_laplacian_spectral_norm=main_graph.spectral_norm,
        active_laplacian_spectral_norm=graph.spectral_norm,
        spectral_ratio=(main_graph.spectral_norm / graph.spectral_norm if name in {"complete_graph", "fully_symmetric"} else 1.0),
    )
    return spec, graph
