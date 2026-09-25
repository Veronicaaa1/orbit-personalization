"""Independent model cloning and orbit-centered parameter perturbations."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from orbit_cifar.constants import NUM_NODES, REFERENCE_ORBITS


def clone_models(base_model: nn.Module, num_nodes: int = NUM_NODES) -> list[nn.Module]:
    """Deep-copy a base model and verify that parameter storage is not shared."""
    models = [copy.deepcopy(base_model) for _ in range(num_nodes)]
    pointers: dict[str, list[int]] = {}
    for model in models:
        for name, parameter in model.named_parameters():
            pointers.setdefault(name, []).append(parameter.data_ptr())
    shared = [name for name, values in pointers.items() if len(set(values)) != len(values)]
    if shared:
        raise RuntimeError(f"Deep-copied models unexpectedly share parameter storage: {shared}")
    return models


def validate_model_compatibility(models: Sequence[nn.Module]) -> tuple[str, ...]:
    """Require identical ordered trainable parameter names, shapes, and dtypes."""
    if not models:
        raise ValueError("At least one model is required")
    reference = [(name, tuple(parameter.shape), parameter.dtype) for name, parameter in models[0].named_parameters() if parameter.requires_grad]
    if not reference:
        raise ValueError("Models contain no trainable parameters")
    for node, model in enumerate(models[1:], start=1):
        current = [(name, tuple(parameter.shape), parameter.dtype) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if current != reference:
            raise ValueError(f"Model {node} trainable parameter names, shapes, or dtypes differ from model 0")
    return tuple(item[0] for item in reference)


def orbit_centered_initialize(
    models: Sequence[nn.Module],
    base_model: nn.Module,
    rho: float = 0.02,
    absolute_scale_floor: float = 1e-3,
    seed: int = 0,
    orbits: Sequence[Sequence[int]] = REFERENCE_ORBITS,
) -> dict[str, Any]:
    """Set ``w_i = w0 + rho*delta_i`` with tensor-wise orbit-zero-mean noise."""
    if rho < 0:
        raise ValueError("rho cannot be negative")
    if len(models) != NUM_NODES:
        raise ValueError(f"Orbit-centered initialization requires {NUM_NODES} models")
    names = validate_model_compatibility(models)
    base_parameters = dict(base_model.named_parameters())
    node_parameters = [dict(model.named_parameters()) for model in models]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    maximum_orbit_mean_error = 0.0
    with torch.no_grad():
        for name in names:
            base = base_parameters[name].detach()
            noises = torch.randn((NUM_NODES, *base.shape), generator=generator, dtype=torch.float64)
            for orbit in orbits:
                index = torch.as_tensor(list(orbit), dtype=torch.long)
                noises[index] -= noises[index].mean(dim=0, keepdim=True)
            common_rms = noises.square().mean().sqrt()
            if float(common_rms) == 0.0:
                raise RuntimeError(f"Degenerate initialization noise for parameter {name}")
            noises /= common_rms
            base_rms = float(base.detach().to(dtype=torch.float64, device="cpu").square().mean().sqrt())
            scale = max(base_rms, float(absolute_scale_floor))
            perturbations = noises * (float(rho) * scale)
            for orbit in orbits:
                index = torch.as_tensor(list(orbit), dtype=torch.long)
                perturbations[index] -= perturbations[index].mean(dim=0, keepdim=True)
                maximum_orbit_mean_error = max(
                    maximum_orbit_mean_error,
                    float(perturbations[index].mean(dim=0).abs().max()),
                )
            for node in range(NUM_NODES):
                value = base + perturbations[node].to(device=base.device, dtype=base.dtype)
                node_parameters[node][name].copy_(value)
    validation = validate_orbit_centered_initialization(models, base_model, orbits=orbits)
    validation.update({"rho": float(rho), "seed": int(seed), "noise_mean_error": maximum_orbit_mean_error})
    return validation


def validate_orbit_centered_initialization(
    models: Sequence[nn.Module],
    base_model: nn.Module,
    orbits: Sequence[Sequence[int]] = REFERENCE_ORBITS,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    """Check that every orbit parameter average equals the shared base state."""
    validate_model_compatibility(models)
    base_parameters = dict(base_model.named_parameters())
    node_parameters = [dict(model.named_parameters()) for model in models]
    maximum_error = 0.0
    any_difference = False
    for name, base in base_parameters.items():
        if not base.requires_grad:
            continue
        for orbit in orbits:
            average = torch.stack([node_parameters[node][name].detach().to(torch.float64) for node in orbit]).mean(dim=0)
            error = float((average - base.detach().to(torch.float64)).abs().max())
            maximum_error = max(maximum_error, error)
            if error > tolerance:
                raise ValueError(f"Orbit average for parameter {name} differs from base by {error}")
        any_difference = any_difference or not torch.equal(node_parameters[0][name], node_parameters[-1][name])
    return {"maximum_orbit_average_error": maximum_error, "models_differ": any_difference}

