"""Exact synchronous decentralized train-step orchestration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from orbit_cifar.data.loaders import set_loader_epoch
from orbit_cifar.models.orbit_initialization import validate_model_compatibility
from orbit_cifar.training.communication import add_communication_gradients, graph_penalty


@dataclass(frozen=True)
class StepDiagnostics:
    mean_local_loss: float
    local_gradient_norm: float
    communication_gradient_norm: float
    parameter_norm: float
    graph_penalty: float
    learning_rate: float


def _gradient_norm(models: Sequence[nn.Module]) -> float:
    total: torch.Tensor | None = None
    for model in models:
        for parameter in model.parameters():
            if parameter.grad is not None:
                contribution = parameter.grad.detach().to(torch.float64).square().sum()
                total = contribution if total is None else total + contribution
    return 0.0 if total is None else float(total.sqrt().item())


def _parameter_norm(models: Sequence[nn.Module]) -> float:
    total: torch.Tensor | None = None
    for model in models:
        for parameter in model.parameters():
            contribution = parameter.detach().to(torch.float64).square().sum()
            total = contribution if total is None else total + contribution
    return 0.0 if total is None else float(total.sqrt().item())


def _all_finite(tensors: Sequence[torch.Tensor]) -> bool:
    """Check a tensor collection with one host synchronization."""
    if not tensors:
        return True
    checks = torch.stack([torch.isfinite(tensor).all() for tensor in tensors])
    return bool(checks.all().item())


class SynchronousDecentralizedTrainer:
    """Single-process simulator preserving simultaneous graph-gradient semantics."""

    def __init__(
        self,
        models: Sequence[nn.Module],
        optimizers: Sequence[torch.optim.Optimizer],
        edges: Sequence[tuple[int, int, float]],
        communication_lambda: float,
        device: torch.device | str = "cpu",
        communication_every_steps: int = 1,
        diagnostics_every_steps: int = 100,
        graph_penalty_every_epochs: int = 1,
        label_smoothing: float = 0.0,
        gradient_clipping: float | None = None,
    ) -> None:
        if len(models) != len(optimizers):
            raise ValueError("Each model must have its own optimizer")
        if len({id(optimizer) for optimizer in optimizers}) != len(optimizers):
            raise ValueError("Optimizer instances must not be shared across nodes")
        validate_model_compatibility(models)
        if communication_every_steps < 1:
            raise ValueError("communication_every_steps must be at least one")
        if diagnostics_every_steps < 1 or graph_penalty_every_epochs < 1:
            raise ValueError("Diagnostic frequencies must be at least one")
        self.models = list(models)
        self.optimizers = list(optimizers)
        self.edges = tuple(edges)
        self.communication_lambda = float(communication_lambda)
        self.device = torch.device(device)
        self.communication_every_steps = int(communication_every_steps)
        self.diagnostics_every_steps = int(diagnostics_every_steps)
        self.graph_penalty_every_epochs = int(graph_penalty_every_epochs)
        self.gradient_clipping = gradient_clipping
        self.loss_function = nn.CrossEntropyLoss(reduction="mean", label_smoothing=float(label_smoothing))
        self.global_step = 0
        for model in self.models:
            model.to(self.device)

    def train_step(
        self,
        batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        force_diagnostics: bool = False,
    ) -> StepDiagnostics:
        """Compute every local and communication gradient before any optimizer step."""
        if len(batches) != len(self.models):
            raise ValueError(f"Expected {len(self.models)} node batches, received {len(batches)}")
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for model, (inputs, targets) in zip(self.models, batches):
            model.train()
            outputs = model(inputs.to(self.device, non_blocking=True))
            loss = self.loss_function(outputs, targets.to(self.device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local loss encountered before the global step")
            loss.backward()
            losses.append(loss.detach())
        collect_diagnostics = force_diagnostics or self.global_step % self.diagnostics_every_steps == 0
        local_norm = _gradient_norm(self.models) if collect_diagnostics else math.nan
        should_communicate = self.global_step % self.communication_every_steps == 0
        communication_norm = add_communication_gradients(
            self.models,
            self.edges,
            self.communication_lambda if should_communicate else 0.0,
            compute_norm=collect_diagnostics,
        )
        if self.gradient_clipping is not None:
            for model in self.models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.gradient_clipping))
        gradients = [parameter.grad for model in self.models for parameter in model.parameters()]
        if any(gradient is None for gradient in gradients):
            raise FloatingPointError("Missing combined gradient before optimizer step")
        if not _all_finite([gradient for gradient in gradients if gradient is not None]):
            raise FloatingPointError("Non-finite combined gradient before optimizer step")
        parameter_norm = _parameter_norm(self.models) if collect_diagnostics else math.nan
        learning_rate = float(self.optimizers[0].param_groups[0]["lr"])
        for optimizer in self.optimizers:
            optimizer.step()
        if not _all_finite([parameter for model in self.models for parameter in model.parameters()]):
            raise FloatingPointError("Non-finite parameter encountered after optimizer step")
        self.global_step += 1
        return StepDiagnostics(
            mean_local_loss=float(torch.stack(losses).mean().item()),
            local_gradient_norm=local_norm,
            communication_gradient_norm=communication_norm,
            parameter_norm=parameter_norm,
            graph_penalty=math.nan,
            learning_rate=learning_rate,
        )

    def train_epoch(self, loaders: Sequence[DataLoader[Any]], epoch: int) -> list[dict[str, float | int]]:
        """Run an epoch after asserting all node loaders have identical lengths."""
        if len(loaders) != len(self.models):
            raise ValueError("The number of loaders must equal the number of graph nodes")
        lengths = [len(loader) for loader in loaders]
        if len(set(lengths)) != 1:
            raise ValueError(f"Node loader lengths differ: {lengths}")
        set_loader_epoch(loaders, epoch)
        iterators = [iter(loader) for loader in loaders]
        rows: list[dict[str, float | int]] = []
        for step in range(lengths[0]):
            diagnostics = self.train_step(
                [next(iterator) for iterator in iterators],
                force_diagnostics=step == lengths[0] - 1,
            )
            row = {"epoch": int(epoch), "step_in_epoch": step, "global_step": self.global_step}
            row.update(diagnostics.__dict__)
            rows.append(row)
        if rows and (epoch + 1) % self.graph_penalty_every_epochs == 0:
            rows[-1]["graph_penalty"] = graph_penalty(self.models, self.edges)
        return rows
