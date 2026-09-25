from __future__ import annotations

import torch
from torch import nn

from orbit_cifar.constants import GLOBAL_SYMMETRY_ORBITS, REFERENCE_ORBITS
from orbit_cifar.models.orbit_initialization import clone_models, orbit_centered_initialize


def base_model() -> nn.Module:
    torch.manual_seed(4)
    return nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))


def test_clone_storage_and_orbit_centered_determinism() -> None:
    base_a = base_model()
    models_a = clone_models(base_a)
    result = orbit_centered_initialize(models_a, base_a, rho=0.02, seed=99)
    assert result["models_differ"]
    for parameters in zip(*(model.parameters() for model in models_a)):
        assert len({parameter.data_ptr() for parameter in parameters}) == 14
    base_b = base_model()
    models_b = clone_models(base_b)
    orbit_centered_initialize(models_b, base_b, rho=0.02, seed=99)
    for model_a, model_b in zip(models_a, models_b):
        for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(parameter_a, parameter_b)
    for base_parameter, *node_parameters in zip(base_a.parameters(), *(model.parameters() for model in models_a)):
        for orbit in REFERENCE_ORBITS:
            average = torch.stack([node_parameters[node].double() for node in orbit]).mean(0)
            assert torch.allclose(average, base_parameter.double(), atol=2e-6, rtol=0)


def test_fully_symmetric_initialization_is_centered_across_all_nodes() -> None:
    base = base_model()
    models = clone_models(base)
    orbit_centered_initialize(models, base, rho=0.02, seed=101, orbits=GLOBAL_SYMMETRY_ORBITS)
    for name, base_parameter in base.named_parameters():
        perturbations = torch.stack(
            [dict(model.named_parameters())[name].detach().double() - base_parameter.detach().double() for model in models]
        )
        assert torch.allclose(perturbations.mean(dim=0), torch.zeros_like(perturbations[0]), atol=2e-6, rtol=0)
