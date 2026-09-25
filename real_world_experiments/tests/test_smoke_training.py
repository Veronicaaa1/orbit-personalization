from __future__ import annotations

from pathlib import Path

from orbit_cifar.analysis.aggregate import aggregate_outputs
from orbit_cifar.cli.common import default_config_path
from orbit_cifar.cli.verify_run import verify_run
from orbit_cifar.config_schema import load_config
from orbit_cifar.experiments.runner import build_smoke_inputs, run_experiment, smoke_config
import orbit_cifar.experiments.runner as runner_module


def test_cpu_smoke_checkpoint_resume_and_aggregation(tmp_path: Path) -> None:
    cfg = smoke_config(load_config(default_config_path()), output_root=tmp_path)
    inputs = build_smoke_inputs(seed=8, samples_per_bundle=4, probe_samples=20)
    first = run_experiment(
        cfg,
        "no_communication",
        seed=8,
        stage="final",
        lambda_override=0.0,
        run_name="smoke",
        datasets=inputs,
    )
    assert first.numerically_stable
    assert verify_run(first.run_directory)["valid"]
    cfg.resume = True
    resumed = run_experiment(
        cfg,
        "no_communication",
        seed=8,
        stage="final",
        lambda_override=0.0,
        run_name="smoke",
        datasets=inputs,
    )
    assert resumed.numerically_stable
    bundle = aggregate_outputs(tmp_path, ["smoke", "missing_control"], [8])
    assert int(bundle.run_inventory["valid"].sum()) == 1
    assert set(bundle.run_inventory["status"]) == {"completed", "missing"}


def test_checkpoint_interval_and_final_evaluation_reuse(tmp_path: Path, monkeypatch) -> None:
    cfg = smoke_config(load_config(default_config_path()), output_root=tmp_path)
    cfg.training.epochs = 2
    cfg.training.eval_every_epochs = 2
    cfg.training.checkpoint_every_epochs = 5
    inputs = build_smoke_inputs(seed=13, samples_per_bundle=2, probe_samples=10)
    checkpoint_names: list[str] = []
    evaluation_epochs: list[int] = []
    real_save = runner_module.save_checkpoint
    real_evaluate = runner_module._evaluate

    def recording_save(path, *args, **kwargs):
        checkpoint_names.append(Path(path).name)
        return real_save(path, *args, **kwargs)

    def recording_evaluate(*args, **kwargs):
        evaluation_epochs.append(int(kwargs["epoch"]))
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner_module, "save_checkpoint", recording_save)
    monkeypatch.setattr(runner_module, "_evaluate", recording_evaluate)
    result = run_experiment(
        cfg,
        "no_communication",
        seed=13,
        stage="final",
        lambda_override=0.0,
        run_name="checkpoint_interval",
        datasets=inputs,
    )
    assert result.numerically_stable
    assert checkpoint_names.count("last.pt") == 2
    assert checkpoint_names.count("final.pt") == 1
    assert evaluation_epochs == [0, 2]
