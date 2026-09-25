from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from orbit_cifar.training.communication import communication_gradients, graph_penalty, trainable_parameter_names
from orbit_cifar.training.engine import SynchronousDecentralizedTrainer


def scalar_models(values: list[float]) -> list[nn.Module]:
    models = [nn.Linear(1, 1, bias=False) for _ in values]
    with torch.no_grad():
        for model, value in zip(models, values):
            model.weight.fill_(value)
    return models


def test_scalar_laplacian_and_edge_order_invariance() -> None:
    values = np.array([1.0, 2.0, 4.0])
    edges = [(0, 1, 0.25), (1, 2, 0.5)]
    models = scalar_models(values.tolist())
    gradients = communication_gradients(models, edges, 0.7)["weight"]
    laplacian = np.array([[0.25, -0.25, 0.0], [-0.25, 0.75, -0.5], [0.0, -0.5, 0.5]])
    expected = 0.7 * laplacian @ values
    assert np.allclose([float(value) for value in gradients], expected)
    reversed_gradients = communication_gradients(models, list(reversed(edges)), 0.7)["weight"]
    assert all(torch.allclose(left, right) for left, right in zip(gradients, reversed_gradients))


def test_full_one_step_matches_closed_form_and_no_early_step() -> None:
    torch.manual_seed(3)
    models = [nn.Linear(2, 2, bias=False) for _ in range(3)]
    with torch.no_grad():
        for node, model in enumerate(models):
            model.weight.copy_(torch.tensor([[1.0 + node, -0.5], [0.25, 0.5 + node]]))
    initial = torch.stack([model.weight.detach().clone() for model in models])
    batches = [
        (torch.tensor([[1.0, -1.0], [0.5, 0.25]]), torch.tensor([0, 1]))
        for _ in models
    ]
    local_gradients = []
    for model, (inputs, targets) in zip(models, batches):
        model.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), targets).backward()
        local_gradients.append(model.weight.grad.detach().clone())
    for model in models:
        model.zero_grad(set_to_none=True)
    edges = [(0, 1, 0.5), (1, 2, 0.25)]
    communication_lambda = 0.4
    learning_rate = 0.1
    laplacian = torch.tensor([[0.5, -0.5, 0.0], [-0.5, 0.75, -0.25], [0.0, -0.25, 0.25]])
    expected = initial - learning_rate * (torch.stack(local_gradients) + communication_lambda * torch.einsum("ij,jab->iab", laplacian, initial))
    all_models = models

    class CheckingSGD(torch.optim.SGD):
        def step(self, closure=None):
            assert all(parameter.grad is not None for model in all_models for parameter in model.parameters())
            return super().step(closure)

    optimizers = [CheckingSGD(model.parameters(), lr=learning_rate) for model in models]
    trainer = SynchronousDecentralizedTrainer(models, optimizers, edges, communication_lambda)
    trainer.train_step(batches)
    actual = torch.stack([model.weight.detach() for model in models])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0)
    assert trainable_parameter_names(models) == ("weight",)


def test_lambda_zero_equals_independent_sgd() -> None:
    torch.manual_seed(12)
    models = [nn.Linear(2, 2) for _ in range(3)]
    independent = [copy.deepcopy(model) for model in models]
    batches = [(torch.randn(4, 2), torch.tensor([0, 1, 0, 1])) for _ in models]
    trainer = SynchronousDecentralizedTrainer(
        models,
        [torch.optim.SGD(model.parameters(), lr=0.03) for model in models],
        [(0, 1, 0.5), (1, 2, 0.5)],
        0.0,
    )
    trainer.train_step(batches)
    for model, batch in zip(independent, batches):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(batch[0]), batch[1]).backward()
        optimizer.step()
    for actual, expected in zip(models, independent):
        for left, right in zip(actual.parameters(), expected.parameters()):
            assert torch.allclose(left, right)


def test_sparse_diagnostics_and_vectorized_graph_penalty() -> None:
    models = scalar_models([1.0, 2.0, 4.0])
    edges = [(0, 1, 0.25), (1, 2, 0.5)]
    expected_penalty = 0.5 * (0.25 * (1.0 - 2.0) ** 2 + 0.5 * (2.0 - 4.0) ** 2)
    assert np.isclose(graph_penalty(models, edges), expected_penalty)
    batches = [(torch.tensor([[1.0]]), torch.tensor([0])) for _ in models]
    trainer = SynchronousDecentralizedTrainer(
        models,
        [torch.optim.SGD(model.parameters(), lr=0.01) for model in models],
        edges,
        communication_lambda=0.2,
        diagnostics_every_steps=100,
    )
    first = trainer.train_step(batches)
    second = trainer.train_step(batches)
    assert math.isfinite(first.local_gradient_norm)
    assert math.isfinite(first.communication_gradient_norm)
    assert math.isnan(second.local_gradient_norm)
    assert math.isnan(second.communication_gradient_norm)
