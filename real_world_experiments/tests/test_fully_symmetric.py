from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from orbit_cifar.cli.common import default_config_path
from orbit_cifar.cli.verify_run import verify_run
from orbit_cifar.config_schema import load_config
from orbit_cifar.experiments.conditions import resolve_condition
from orbit_cifar.experiments.runner import build_smoke_inputs, run_experiment, smoke_config


def test_condition_uses_complete_graph_spectral_rule_and_balanced_assignment() -> None:
    selected = {"lambda_main": 0.1, "lambda_large": 0.3}
    complete, complete_graph = resolve_condition("complete_graph", selected)
    symmetric, symmetric_graph = resolve_condition("fully_symmetric")
    assert symmetric.assignment == "fully_symmetric"
    assert symmetric.graph_name == "complete"
    assert symmetric.lambda_main == 0.1
    assert symmetric.communication_lambda == complete.communication_lambda
    assert np.array_equal(symmetric_graph.adjacency, complete_graph.adjacency)


def test_fully_symmetric_smoke_run_and_resume(tmp_path: Path) -> None:
    cfg = smoke_config(load_config(default_config_path()), output_root=tmp_path)
    cfg.training.epochs = 2
    inputs = build_smoke_inputs(
        seed=23,
        samples_per_bundle=2,
        probe_samples=10,
        assignment="fully_symmetric",
    )
    result = run_experiment(cfg, "fully_symmetric", seed=0, stage="final", datasets=inputs)
    expected = tmp_path / "final" / "fully_symmetric" / "seed_000"
    assert result.run_directory == expected.resolve()
    assert result.numerically_stable
    assert verify_run(expected)["valid"]

    checkpoint_metrics = pd.read_csv(expected / "metrics" / "checkpoint_metrics.csv")
    finite_columns = ["D_total", "D_global", "D_within", "D_between", "decomposition_error"]
    assert np.isfinite(checkpoint_metrics[finite_columns].to_numpy()).all()
    assert np.allclose(
        checkpoint_metrics["D_total"],
        checkpoint_metrics["D_within"] + checkpoint_metrics["D_between"],
        atol=1e-9,
        rtol=0,
    )
    terminal = json.loads((expected / "metrics" / "terminal_metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(terminal["D_global_bar"])
    assert terminal["primary_disagreement_metric"] == "D_global=D_total"
    assignment = json.loads((expected / "data_assignment.json").read_text(encoding="utf-8"))
    assert assignment["partition_semantics"] == "reference_partition_only"
    assert assignment["true_automorphism_orbits"] == [list(range(14))]

    checkpoint = torch.load(expected / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
    assert checkpoint["extra"]["initialization"]["maximum_orbit_average_error"] < 2e-6

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_experiment(cfg, "fully_symmetric", seed=0, stage="final", datasets=inputs)

    cfg.resume = True
    resumed = run_experiment(cfg, "fully_symmetric", seed=0, stage="final", datasets=inputs)
    assert resumed.numerically_stable
    assert resumed.run_directory == expected.resolve()
