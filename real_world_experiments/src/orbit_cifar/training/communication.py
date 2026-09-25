"""Graph-Laplacian gradient construction over all trainable parameters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import torch
from torch import nn

from orbit_cifar.models.orbit_initialization import validate_model_compatibility

WeightedEdge = tuple[int, int, float]


def communication_gradients(
    models: Sequence[nn.Module],
    edges: Iterable[WeightedEdge],
    communication_lambda: float,
) -> dict[str, list[torch.Tensor]]:
    """Build all simultaneous ``lambda L_C w`` tensors without mutating parameters."""
    names = validate_model_compatibility(models)
    parameter_maps = [dict(model.named_parameters()) for model in models]
    gradients = {
        name: [torch.zeros_like(parameter_maps[node][name]) for node in range(len(models))]
        for name in names
    }
    if communication_lambda == 0:
        return gradients
    with torch.no_grad():
        for node_i, node_j, weight in tuple(edges):
            if node_i == node_j or not (0 <= node_i < len(models) and 0 <= node_j < len(models)):
                raise ValueError(f"Invalid communication edge ({node_i}, {node_j})")
            coefficient = float(communication_lambda) * float(weight)
            for name in names:
                difference = parameter_maps[node_i][name].detach() - parameter_maps[node_j][name].detach()
                gradients[name][node_i].add_(difference, alpha=coefficient)
                gradients[name][node_j].add_(difference, alpha=-coefficient)
    return gradients


def add_communication_gradients(
    models: Sequence[nn.Module],
    edges: Iterable[WeightedEdge],
    communication_lambda: float,
    compute_norm: bool = True,
) -> float:
    """Add prebuilt simultaneous communication tensors to local gradients."""
    if communication_lambda == 0:
        return 0.0
    additions = communication_gradients(models, edges, communication_lambda)
    parameter_maps = [dict(model.named_parameters()) for model in models]
    squared_norm: torch.Tensor | None = None
    with torch.no_grad():
        for name, per_node in additions.items():
            for node, addition in enumerate(per_node):
                parameter = parameter_maps[node][name]
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad.add_(addition)
                if compute_norm:
                    contribution = addition.detach().to(torch.float64).square().sum()
                    squared_norm = contribution if squared_norm is None else squared_norm + contribution
    if squared_norm is None:
        return 0.0 if compute_norm else float("nan")
    return float(squared_norm.sqrt().item())


def graph_penalty(models: Sequence[nn.Module], edges: Iterable[WeightedEdge]) -> float:
    """Return ``0.5 * sum_(i,j) C_ij ||w_i-w_j||^2`` for diagnostics."""
    names = validate_model_compatibility(models)
    parameter_maps = [dict(model.named_parameters()) for model in models]
    edge_values = tuple(edges)
    if not edge_values:
        return 0.0
    device = next(models[0].parameters()).device
    left = torch.as_tensor([edge[0] for edge in edge_values], dtype=torch.long, device=device)
    right = torch.as_tensor([edge[1] for edge in edge_values], dtype=torch.long, device=device)
    weights = torch.as_tensor([edge[2] for edge in edge_values], dtype=torch.float64, device=device)
    value = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for name in names:
            stacked = torch.stack([parameters[name].detach() for parameters in parameter_maps], dim=0)
            differences = stacked.index_select(0, left) - stacked.index_select(0, right)
            per_edge = differences.to(torch.float64).reshape(len(edge_values), -1).square().sum(dim=1)
            value.add_(torch.dot(weights, per_edge), alpha=0.5)
    return float(value.item())


def trainable_parameter_names(models: Sequence[nn.Module]) -> tuple[str, ...]:
    """Expose the exact set included in communication for validation and tests."""
    return validate_model_compatibility(models)
