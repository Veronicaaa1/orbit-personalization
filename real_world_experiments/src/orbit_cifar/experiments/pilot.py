"""Validation-only lambda pilot orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from orbit_cifar.config_schema import validate_config

from orbit_cifar.experiments.runner import run_experiment
from orbit_cifar.experiments.completion import verify_completed_run, verify_resumable_run
from orbit_cifar.experiments.matrix import ScientificKey, validate_active_run
from orbit_cifar.experiments.provenance import pilot_training_config, selection_context
from orbit_cifar.utils.io import ensure_directory, atomic_write_json, config_fingerprint


def lambda_directory_name(value: float) -> str:
    """Create a stable filesystem label without changing the numeric value."""
    return "lambda_" + format(float(value), ".12g").replace("-", "minus").replace(".", "p")


def run_pilot(cfg: DictConfig) -> pd.DataFrame:
    """Run all aligned main-graph validation pilots and save seed-level rows."""
    active = bool(OmegaConf.select(cfg, "experiment.name"))
    if active:
        for pilot_seed in cfg.pilot.seeds:
            validate_active_run(cfg, "main", int(pilot_seed), "pilot")
    if (OmegaConf.select(cfg, "experiment.name") and str(cfg.data.name) == "cifar10"
            and str(cfg.model.name) == "resnet20_gn" and not bool(cfg.legacy_reuse.rerun_pilot_family)):
        raise RuntimeError("CIFAR-10/ResNet pilot is frozen at the accepted legacy selection; explicit authorization is required to rerun it")
    pilot_cfg = pilot_training_config(cfg)
    validate_config(pilot_cfg)
    family_root = Path(str(cfg.project_root)).expanduser().resolve() / str(cfg.output_root) / "pilot"
    context_fingerprint = None
    if active:
        family_root = family_root / str(cfg.data.name) / str(cfg.model.name)
        context = selection_context(cfg)
        context_fingerprint = config_fingerprint(context)
        provenance_path = family_root / "pilot_provenance.json"
        if provenance_path.is_file():
            stored = json.loads(provenance_path.read_text(encoding="utf-8"))
            if stored != {"context": context, "context_fingerprint": context_fingerprint}:
                raise ValueError("Existing pilot family has stale scientific provenance")
        elif family_root.exists() and any(family_root.iterdir()):
            raise ValueError("Existing pilot outputs lack scientific provenance; cannot reuse or overwrite")
        atomic_write_json(provenance_path, {"context": context, "context_fingerprint": context_fingerprint})
    rows: list[dict[str, Any]] = []
    for value in cfg.pilot.lambda_grid:
        lambda_value = float(value)
        for seed in cfg.pilot.seeds:
            candidate_directory = family_root / lambda_directory_name(lambda_value) / f"seed_{int(seed):03d}"
            status_path = candidate_directory / "run_status.json"
            terminal_path = candidate_directory / "metrics" / "terminal_metrics.json"
            if status_path.is_file() and terminal_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if status.get("status") == "completed" and bool(status.get("numerically_stable", False)):
                    if active:
                        checked = verify_completed_run(candidate_directory, ScientificKey(str(cfg.data.name), str(cfg.model.name), "main", int(seed)),
                                                       pilot_cfg, expected_epochs=int(cfg.pilot.epochs))
                        if not checked["valid"]:
                            raise ValueError(f"Existing pilot candidate is stale or invalid: {checked['errors']}")
                    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
                    if active and float(terminal.get("communication_lambda", float("nan"))) != lambda_value:
                        raise ValueError("Existing pilot candidate lambda disagrees with its grid directory")
                    rows.append({"lambda": lambda_value, "seed": int(seed), "numerically_stable": True,
                                 "D_within_bar": terminal.get("D_within_bar", float("nan")),
                                 "D_between_bar": terminal.get("D_between_bar", float("nan")),
                                 "R_orbit_bar": terminal.get("R_orbit_bar", float("nan")),
                                 "orbit_ensemble_accuracy": terminal.get("orbit_ensemble_accuracy", float("nan")),
                                 "run_directory": str(candidate_directory), "action": "skipped_complete",
                                 "context_fingerprint": context_fingerprint})
                    continue
            candidate_cfg = pilot_cfg
            if (candidate_directory / "checkpoints" / "last.pt").is_file():
                if active:
                    checked = verify_resumable_run(candidate_directory, ScientificKey(str(cfg.data.name), str(cfg.model.name), "main", int(seed)), pilot_cfg)
                    if not checked["valid"]:
                        raise ValueError(f"Pilot candidate cannot resume: {checked['errors']}")
                candidate_cfg = OmegaConf.merge(pilot_cfg, OmegaConf.create({"resume": True}))
            result = run_experiment(
                candidate_cfg,
                condition_name="main",
                seed=int(seed),
                stage="pilot",
                lambda_override=lambda_value,
                run_name=lambda_directory_name(lambda_value),
            )
            terminal = result.terminal_metrics
            rows.append(
                {
                    "lambda": lambda_value,
                    "seed": int(seed),
                    "numerically_stable": result.numerically_stable,
                    "D_within_bar": terminal.get("D_within_bar", float("nan")),
                    "D_between_bar": terminal.get("D_between_bar", float("nan")),
                    "R_orbit_bar": terminal.get("R_orbit_bar", float("nan")),
                    "orbit_ensemble_accuracy": terminal.get("orbit_ensemble_accuracy", float("nan")),
                    "run_directory": str(result.run_directory),
                    "action": "resumed" if bool(candidate_cfg.resume) else "trained",
                    "context_fingerprint": context_fingerprint,
                }
            )
    frame = pd.DataFrame(rows)
    output_root = Path(str(pilot_cfg.project_root)).expanduser().resolve() / str(pilot_cfg.output_root) / "pilot"
    if OmegaConf.select(cfg, "experiment.name"):
        output_root = output_root / str(cfg.data.name) / str(cfg.model.name)
    ensure_directory(output_root)
    frame.to_csv(output_root / "pilot_runs.csv", index=False)
    return frame
