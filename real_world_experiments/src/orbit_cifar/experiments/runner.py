from __future__ import annotations

import logging
import math
import json
import warnings
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from orbit_cifar.config_schema import config_to_dict
from orbit_cifar.constants import (
    GLOBAL_SYMMETRY_ORBITS,
    NUM_NODES,
    REFERENCE_ORBITS,
)
from orbit_cifar.data.assignments import assignment_summary
from orbit_cifar.data.loaders import build_node_loaders
from orbit_cifar.evaluation.dispersion import pairwise_distance_matrix, terminal_window
from orbit_cifar.evaluation.random_partitions import generate_random_partitions, random_partition_diagnostic
from orbit_cifar.experiments.conditions import ConditionSpec, resolve_condition
from orbit_cifar.graph.artifacts import save_graph_artifacts
from orbit_cifar.models.factory import create_model
from orbit_cifar.models.orbit_initialization import clone_models, orbit_centered_initialize
from orbit_cifar.training.checkpoint import load_checkpoint, save_checkpoint
from orbit_cifar.training.engine import SynchronousDecentralizedTrainer
from orbit_cifar.training.optim import WarmupCosineFleetScheduler, build_optimizers
from orbit_cifar.training.seed_plan import build_seed_plan
from orbit_cifar.utils.environment import environment_summary, resolve_device
from orbit_cifar.utils.io import atomic_torch_save, atomic_write_csv, atomic_write_json, atomic_write_yaml, config_fingerprint, ensure_directory
from orbit_cifar.utils.reproducibility import seed_everything
from orbit_cifar.experiments.inputs import (
    DatasetInputs, _project_path, _real_dataset_inputs, _fully_symmetric_dataset_inputs,
    build_smoke_inputs, smoke_config,
)
from orbit_cifar.experiments.evaluation import _evaluate
from orbit_cifar.experiments.matrix import ScientificKey, validate_active_run
from orbit_cifar.experiments.provenance import build_scientific_identity, load_selected
from orbit_cifar.experiments.completion import publish_completion
from orbit_cifar.evaluation.orbit_diagnostics import PROFILE, RETAINED_EPOCHS
from orbit_cifar.experiments.orbit_diagnostics import OrbitDiagnosticRecorder, probe_contract

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    run_directory: Path
    numerically_stable: bool
    terminal_metrics: dict[str, Any]


def _load_selected(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Selected hyperparameters do not exist: {path}. Run orbit-select-lambda or provide an explicit lambda override."
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("selected_hparams.yaml must contain a mapping")
    if value.get("lambda_main") is None:
        raise RuntimeError(f"No valid Main lambda was selected for this family: {path}")
    return value


def run_experiment(
    cfg: DictConfig,
    condition_name: str,
    seed: int,
    stage: str = "final",
    lambda_override: float | None = None,
    run_name: str | None = None,
    datasets: DatasetInputs | None = None,
) -> RunResult:
    """Execute one complete, resumable, artifact-producing paired run."""
    cfg = OmegaConf.merge(cfg, {"stage": stage})
    active = bool(OmegaConf.select(cfg, "experiment.name"))
    if active:
        validate_active_run(cfg, condition_name, seed, stage)
    if int(cfg.training.epochs) < 1 or not 0 <= int(cfg.training.warmup_epochs) <= int(cfg.training.epochs):
        raise ValueError("Invalid total_epochs/warmup_epochs schedule")
    if active and stage == "final" and int(cfg.training.epochs) != 200:
        raise ValueError("Active final cells require 200 epochs; use the preflight stage for a short schedule")
    if active and datasets is not None and stage != "preflight":
        raise ValueError("Injected synthetic inputs are permitted only in preflight or the isolated smoke path")
    if (active and stage == "final" and bool(cfg.legacy_reuse.enabled)
            and str(cfg.data.name) == str(cfg.legacy_reuse.allowed_dataset)
            and str(cfg.model.name) == str(cfg.legacy_reuse.allowed_model)
            and condition_name in {str(value) for value in cfg.legacy_reuse.conditions}
            and int(seed) in {int(value) for value in cfg.legacy_reuse.seeds}):
        token = f"{cfg.data.name}/{cfg.model.name}/{condition_name}/seed_{int(seed):03d}"
        if token not in {str(value) for value in cfg.legacy_reuse.retrain_overrides}:
            raise RuntimeError(
                f"{token} is reserved for legacy reuse; register it first or add that exact token to "
                "legacy_reuse.retrain_overrides after explicit authorization"
            )
    if active and condition_name == "no_communication" and lambda_override not in {None, 0.0}:
        raise ValueError("no_communication has exact lambda zero; a nonzero generic override is forbidden")
    if active and stage == "final" and condition_name != "no_communication" and lambda_override is not None:
        raise ValueError("Active final lambda must come from the validated family selection")
    if run_name is not None and (Path(run_name).name != run_name or run_name in {".", ".."}):
        raise ValueError("run_name must be one directory component")
    if active and stage == "final" and run_name not in {None, condition_name}:
        raise ValueError("Active final runs must use the canonical scientific cell directory")
    selected: dict[str, Any] = {}
    if lambda_override is None and condition_name != "no_communication":
        if active:
            selected_path = (
                _project_path(cfg, str(cfg.output_root)) / "pilot" / str(cfg.data.name) /
                str(cfg.model.name) / "selected_hparams.yaml"
            )
            selected = load_selected(cfg, selected_path)
        elif condition_name != "fully_symmetric":
            selected = _load_selected(_project_path(cfg, str(cfg.selected_hparams_path)))
    condition, graph = resolve_condition(condition_name, selected, lambda_override=lambda_override)
    fully_symmetric = condition.name == "fully_symmetric"
    output_root = _project_path(cfg, str(cfg.output_root))
    condition_directory = run_name or condition_name
    if active:
        run_path = (output_root / stage / str(cfg.data.name) / str(cfg.model.name) / condition_directory / f"seed_{int(seed):03d}").resolve()
    else:
        run_path = (output_root / stage / condition_directory / f"seed_{int(seed):03d}").resolve()
    if run_path.exists() and not bool(cfg.resume) and (active or fully_symmetric):
        raise FileExistsError(
            f"Refusing to overwrite existing run directory without --resume: {run_path}"
        )
    if active and bool(cfg.resume) and (run_path / "legacy_reference.json").is_file():
        raise ValueError("Imported legacy observations are external completed results and can never be resumed")
    if active and bool(cfg.resume) and (run_path / "run_status.json").is_file():
        existing_status = json.loads((run_path / "run_status.json").read_text(encoding="utf-8"))
        if existing_status.get("status") == "completed":
            raise FileExistsError("The native run is already completed; use the planner's verified skip")
    data_inputs = datasets or _real_dataset_inputs(cfg, stage)
    if fully_symmetric:
        data_inputs = _fully_symmetric_dataset_inputs(data_inputs, cfg)
    run_directory = run_path
    resolved = config_to_dict(cfg)
    resolved["seed"] = int(seed)
    resolved["stage"] = stage
    resolved["resume"] = False
    resolved["resolved_condition"] = condition.to_dict()
    diagnostic_profile = None
    if bool(OmegaConf.select(cfg, "orbit_diagnostics.enabled")) and condition_name == "main" and stage == "final":
        if OmegaConf.select(cfg, "orbit_diagnostics.schema_version") != 1:
            raise ValueError("Unsupported orbit diagnostic schema_version")
        diagnostic_profile = PROFILE
    # The recorded profile is part of the publication contract, not the
    # scientific fingerprint. An old run is never upgraded implicitly on resume.
    if bool(cfg.resume) and (run_directory / "resolved_config.yaml").is_file():
        previous_resolved = yaml.safe_load((run_directory / "resolved_config.yaml").read_text(encoding="utf-8")) or {}
        diagnostic_profile = previous_resolved.get("orbit_diagnostics_profile")
    if diagnostic_profile is not None:
        if diagnostic_profile != PROFILE:
            raise ValueError("Unsupported saved orbit diagnostic profile")
        if active and any(epoch % int(cfg.training.eval_every_epochs) for epoch in RETAINED_EPOCHS):
            raise ValueError("Diagnostic profile requires existing scheduled terminal evaluations")
        resolved["orbit_diagnostics_profile"] = diagnostic_profile
    if active:
        resolved["dataset_metadata"] = {
            "name": str(cfg.data.name), "input_shape": [int(cfg.model.input_channels), int(cfg.model.input_height), int(cfg.model.input_width)],
            "num_classes": int(cfg.model.num_classes), "class_names": list(data_inputs.class_names),
        }
    if fully_symmetric:
        resolved["fully_symmetric_design"] = {
            "manifest_fingerprint": config_fingerprint(data_inputs.manifest),
            "true_automorphism_orbits": [list(group) for group in GLOBAL_SYMMETRY_ORBITS],
            "reference_partition_semantics": "reference_partition_only",
            "primary_disagreement_metric": "D_global=D_total",
        }
    scientific_identity = build_scientific_identity(
        cfg, condition, seed, data_inputs.manifest, data_inputs.class_names,
    )
    if active:
        fingerprint = config_fingerprint(scientific_identity)
        resolved["scientific_fingerprint"] = fingerprint
        resolved["scientific_identity"] = scientific_identity
    else:
        # Reconstruct the pre-extension fingerprint so existing native legacy
        # checkpoints remain resumable despite newly documented data metadata.
        legacy_identity = copy.deepcopy(resolved)
        legacy_identity.pop("orbit_diagnostics", None)
        legacy_identity.pop("orbit_diagnostics_profile", None)
        for key in ("input_channels", "input_height", "input_width", "class_names_source", "normalization_mean", "normalization_std"):
            legacy_identity.get("data", {}).pop(key, None)
        for key in ("input_channels", "input_height", "input_width", "hidden_dimensions"):
            legacy_identity.get("model", {}).pop(key, None)
        fingerprint = config_fingerprint(legacy_identity)
    if active and bool(cfg.resume):
        existing_resolved_path = run_directory / "resolved_config.yaml"
        existing_last_path = run_directory / "checkpoints" / "last.pt"
        if not existing_resolved_path.is_file() or not existing_last_path.is_file():
            raise FileNotFoundError("Active resume requires both resolved_config.yaml and checkpoints/last.pt")
        existing_resolved = yaml.safe_load(existing_resolved_path.read_text(encoding="utf-8")) or {}
        if existing_resolved.get("scientific_fingerprint") != fingerprint:
            raise ValueError("Existing native run scientific fingerprint does not match the requested resume")
    checkpoint_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    last_saved_epoch = 0
    ensure_directory(run_directory)
    for child in ("graph", "checkpoints", "metrics", "artifacts"):
        ensure_directory(run_directory / child)
    try:
        atomic_write_yaml(run_directory / "resolved_config.yaml", resolved)
        device = resolve_device(str(cfg.device))
        atomic_write_json(run_directory / "environment.json", environment_summary(device))
        seed_plan = build_seed_plan(seed, namespace=stage if active else "legacy")
        atomic_write_json(run_directory / "seed_plan.json", seed_plan)
        save_graph_artifacts(
            graph,
            run_directory / "graph",
            {
                "communication_lambda": condition.communication_lambda,
                "lambda_main": condition.lambda_main,
                "main_laplacian_spectral_norm": condition.main_laplacian_spectral_norm,
                "active_laplacian_spectral_norm": condition.active_laplacian_spectral_norm,
            },
        )
        assignment = assignment_summary(condition.assignment, data_inputs.manifest)
        assignment["manifest_fingerprint"] = config_fingerprint(data_inputs.manifest)
        assignment["dataset"] = str(cfg.data.name)
        assignment["validation_indices_checksum"] = config_fingerprint(data_inputs.manifest.get("validation_indices", []))
        if fully_symmetric:
            assignment["balanced_manifest"] = data_inputs.manifest
        atomic_write_json(run_directory / "data_assignment.json", assignment)
        atomic_write_json(
            run_directory / "run_status.json",
            {"status": "running", "numerically_stable": True, "last_valid_epoch": 0},
        )
        seed_everything(seed_plan["base_initialization"], deterministic=bool(cfg.deterministic))
        base_model = create_model(
            str(cfg.model.name),
            num_classes=int(cfg.model.num_classes),
            group_norm_max_groups=int(cfg.model.group_norm_max_groups),
            input_channels=int(cfg.model.input_channels),
            input_height=int(cfg.model.input_height),
            input_width=int(cfg.model.input_width),
            hidden_dimensions=list(OmegaConf.select(cfg, "model.hidden_dimensions") or [256, 128]),
        )
        resolved["model_parameter_count"] = int(sum(parameter.numel() for parameter in base_model.parameters()))
        atomic_write_yaml(run_directory / "resolved_config.yaml", resolved)
        models = clone_models(base_model)
        initialization = orbit_centered_initialize(
            models,
            base_model,
            rho=float(cfg.model.initialization_rho),
            absolute_scale_floor=float(cfg.model.absolute_scale_floor),
            seed=seed_plan["orbit_perturbation"],
            orbits=GLOBAL_SYMMETRY_ORBITS if fully_symmetric else REFERENCE_ORBITS,
        )
        optimizers = build_optimizers(
            models,
            learning_rate=float(cfg.training.learning_rate),
            momentum=float(cfg.training.momentum),
            weight_decay=float(cfg.training.weight_decay),
        )
        scheduler = WarmupCosineFleetScheduler(
            optimizers,
            base_learning_rate=float(cfg.training.learning_rate),
            total_epochs=int(cfg.training.epochs),
            warmup_epochs=int(cfg.training.warmup_epochs),
            minimum_learning_rate=float(cfg.training.minimum_learning_rate),
        )
        trainer = SynchronousDecentralizedTrainer(
            models,
            optimizers,
            graph.edges,
            condition.communication_lambda,
            device=device,
            communication_every_steps=int(cfg.training.communication_every_steps),
            diagnostics_every_steps=int(cfg.training.diagnostics_every_steps),
            graph_penalty_every_epochs=int(cfg.training.graph_penalty_every_epochs),
            label_smoothing=float(cfg.training.label_smoothing),
            gradient_clipping=cfg.training.gradient_clipping,
        )
        loaders = build_node_loaders(
            data_inputs.training_dataset,
            data_inputs.manifest,
            condition.assignment,
            data_inputs.train_transform,
            run_seed=int(seed_plan["data_run_seed"]),
            batch_size=int(cfg.training.batch_size_per_node),
            num_workers=int(cfg.training.num_workers),
            persistent_workers=bool(cfg.training.persistent_workers),
        )
        checkpoint_rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        final_evaluation_payload: tuple[
            dict[str, Any],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            pd.DataFrame,
            pd.DataFrame,
            dict[str, np.ndarray],
        ] | None = None
        start_epoch = 0
        last_saved_epoch = 0
        last_path = run_directory / "checkpoints" / "last.pt"
        diagnostics = None
        if diagnostic_profile is not None:
            classes = list(data_inputs.class_names) or list(map(str, range(int(cfg.model.num_classes))))
            contract = probe_contract(data_inputs.probe_loader, str(cfg.data.name), classes, str(cfg.data.dataset_version))
            diagnostics = OrbitDiagnosticRecorder(
                run_directory, {"dataset": str(cfg.data.name), "model": str(cfg.model.name),
                                "condition": condition_name, "seed": int(seed), "stage": stage,
                                "scientific_fingerprint": fingerprint}, contract,
                final_epoch=int(cfg.training.epochs), cadence=int(cfg.training.eval_every_epochs),
            )

        def checkpoint_extra(epoch, *, terminal=None):
            extra = {"initialization": initialization}
            if terminal is not None:
                extra["terminal_metrics"] = terminal
            if diagnostics is not None:
                extra["orbit_diagnostics"] = diagnostics.publish(epoch, trainer.global_step)
            return extra

        if bool(cfg.resume) and last_path.is_file():
            restored = load_checkpoint(last_path, models, optimizers, scheduler, fingerprint, map_location=device)
            start_epoch = int(restored["epoch"])
            last_saved_epoch = start_epoch
            trainer.global_step = int(restored["global_step"])
            if diagnostics is not None:
                saved_diagnostics = restored["extra"].get("orbit_diagnostics")
                if saved_diagnostics is None:
                    raise ValueError("Enhanced checkpoint is missing its diagnostic generation")
                diagnostics.restore(saved_diagnostics, start_epoch, trainer.global_step)
            metrics_path = run_directory / "metrics" / "checkpoint_metrics.csv"
            diagnostics_path = run_directory / "metrics" / "train_diagnostics.csv"
            if metrics_path.is_file():
                checkpoint_rows = pd.read_csv(metrics_path).query("epoch <= @start_epoch").to_dict("records")
            if diagnostics_path.is_file():
                training_rows = pd.read_csv(diagnostics_path).query("epoch < @start_epoch").to_dict("records")
            if not checkpoint_rows:
                raise ValueError("Resume requires checkpoint metrics history; refusing to reset a saved checkpoint to epoch zero")
            epochs = [int(row["epoch"]) for row in checkpoint_rows]
            expected = [0] + list(range(int(cfg.training.eval_every_epochs), start_epoch + 1, int(cfg.training.eval_every_epochs)))
            if start_epoch == int(cfg.training.epochs) and start_epoch not in expected:
                expected.append(start_epoch)
            if epochs != expected:
                raise ValueError("Resume metrics history is missing or duplicates scheduled evaluations")
        elif bool(cfg.resume) and not active:
            LOGGER.warning("Resume requested, but no last checkpoint exists at %s; starting from epoch 0", last_path)
        run_started = time.perf_counter()
        if not checkpoint_rows:
            initial, *_ = _evaluate(
                models,
                data_inputs.probe_loader,
                device,
                cfg,
                epoch=0,
                reference_partition_only=fully_symmetric,
                diagnostics=diagnostics,
                global_step=trainer.global_step,
            )
            checkpoint_rows.append(initial)
            atomic_write_csv(run_directory / "metrics" / "checkpoint_metrics.csv", pd.DataFrame(checkpoint_rows))
            save_checkpoint(last_path, models, optimizers, scheduler, 0, trainer.global_step, fingerprint, checkpoint_extra(0))
            last_saved_epoch = 0
        for epoch in range(start_epoch, int(cfg.training.epochs)):
            epoch_started = time.perf_counter()
            epoch_rows = trainer.train_epoch(loaders, epoch)
            if epoch_rows:
                epoch_rows[-1]["epoch_wall_seconds"] = float(time.perf_counter() - epoch_started)
            training_rows.extend(epoch_rows)
            scheduler.step()
            completed_epoch = epoch + 1
            if completed_epoch % int(cfg.training.eval_every_epochs) == 0 or completed_epoch == int(cfg.training.epochs):
                evaluation_payload = _evaluate(
                    models,
                    data_inputs.probe_loader,
                    device,
                    cfg,
                    epoch=completed_epoch,
                    full_classification=completed_epoch == int(cfg.training.epochs),
                    reference_partition_only=fully_symmetric,
                    diagnostics=diagnostics,
                    global_step=trainer.global_step,
                )
                evaluation = evaluation_payload[0]
                checkpoint_rows.append(evaluation)
                if completed_epoch == int(cfg.training.epochs):
                    final_evaluation_payload = evaluation_payload
            checkpoint_due = (
                completed_epoch % int(cfg.training.checkpoint_every_epochs) == 0
                or completed_epoch == int(cfg.training.epochs)
            )
            if checkpoint_due:
                atomic_write_csv(run_directory / "metrics" / "train_diagnostics.csv", pd.DataFrame(training_rows))
                atomic_write_csv(run_directory / "metrics" / "checkpoint_metrics.csv", pd.DataFrame(checkpoint_rows))
                save_checkpoint(
                    last_path,
                    models,
                    optimizers,
                    scheduler,
                    completed_epoch,
                    trainer.global_step,
                    fingerprint,
                    checkpoint_extra(completed_epoch),
                )
                last_saved_epoch = completed_epoch
                atomic_write_json(
                    run_directory / "run_status.json",
                    {"status": "running", "numerically_stable": True, "last_valid_epoch": completed_epoch},
                )
        if final_evaluation_payload is None:
            final_evaluation_payload = _evaluate(
                models,
                data_inputs.probe_loader,
                device,
                cfg,
                epoch=int(cfg.training.epochs),
                full_classification=True,
                reference_partition_only=fully_symmetric,
                diagnostics=diagnostics,
                global_step=trainer.global_step,
            )
        final_metrics, raw, centered, labels, node_table, orbit_table, confusions = final_evaluation_payload
        if not checkpoint_rows or int(checkpoint_rows[-1]["epoch"]) != int(cfg.training.epochs):
            checkpoint_rows.append(final_metrics)
        pd.DataFrame(checkpoint_rows).drop_duplicates(subset=["epoch"], keep="last").to_csv(
            run_directory / "metrics" / "checkpoint_metrics.csv", index=False
        )
        node_table.to_csv(run_directory / "metrics" / "node_classification.csv", index=False)
        orbit_table.to_csv(run_directory / "metrics" / "orbit_classification.csv", index=False)
        terminal, short_window = terminal_window(checkpoint_rows, count=int(cfg.training.terminal_window_checkpoints))
        if short_window:
            warnings.warn("Fewer valid evaluation checkpoints than the configured terminal window; using all available", stacklevel=2)
        valid_evaluations = [row for row in checkpoint_rows if math.isfinite(float(row["D_within"])) and math.isfinite(float(row["D_between"]))]
        terminal_evaluations = valid_evaluations[-int(cfg.training.terminal_window_checkpoints) :]
        if active and stage == "final" and int(cfg.training.epochs) == 200:
            expected_terminal_epochs = [180, 185, 190, 195, 200]
            actual_terminal_epochs = [int(row["epoch"]) for row in terminal_evaluations]
            if actual_terminal_epochs != expected_terminal_epochs:
                raise ValueError(
                    f"Complete final run requires scheduled terminal evaluations {expected_terminal_epochs}; found {actual_terminal_epochs}"
                )
        terminal.update(
            {
                "orbit_ensemble_accuracy": float(np.mean([float(row["orbit_ensemble_accuracy"]) for row in terminal_evaluations])),
                "mean_node_accuracy": float(np.mean([float(row["mean_node_accuracy"]) for row in terminal_evaluations])),
                "D_total_bar": float(np.mean([float(row["D_total"]) for row in terminal_evaluations])),
                "raw_logit_rms": float(np.mean([float(row["raw_logit_rms"]) for row in terminal_evaluations])),
                "centered_logit_rms": float(np.mean([float(row["centered_logit_rms"]) for row in terminal_evaluations])),
                "final_orbit_ensemble_accuracy": final_metrics["orbit_ensemble_accuracy"],
                "final_mean_node_accuracy": final_metrics["mean_node_accuracy"],
                "final_raw_logit_rms": final_metrics["raw_logit_rms"],
                "final_centered_logit_rms": final_metrics["centered_logit_rms"],
                "terminal_window_short": short_window,
                "condition": condition.name,
                "seed": int(seed),
                "communication_lambda": condition.communication_lambda,
                "numerically_stable": True,
                "probe_split": "validation" if stage in {"pilot", "preflight"} else "official_test",
                "dataset": str(cfg.data.name),
                "model": str(cfg.model.name),
                "source_type": "native",
                "run_wall_seconds": float(time.perf_counter() - run_started),
            }
        )
        if fully_symmetric:
            terminal.update(
                {
                    "D_global_bar": float(np.mean([float(row["D_total"]) for row in terminal_evaluations])),
                    "final_D_global": float(final_metrics["D_total"]),
                    "primary_disagreement_metric": "D_global=D_total",
                    "partition_semantics": "reference_partition_only",
                    "true_automorphism_orbits": [list(group) for group in GLOBAL_SYMMETRY_ORBITS],
                    "reference_group_ensemble_accuracy": terminal["orbit_ensemble_accuracy"],
                    "final_reference_group_ensemble_accuracy": terminal["final_orbit_ensemble_accuracy"],
                }
            )
        atomic_write_json(run_directory / "metrics" / "terminal_metrics.json", terminal)
        save_logits = bool(OmegaConf.select(cfg, "artifacts.save_final_logits") if OmegaConf.select(cfg, "artifacts.save_final_logits") is not None else True)
        if save_logits:
            atomic_torch_save(run_directory / "artifacts" / "final_raw_logits.pt", raw)
            atomic_torch_save(run_directory / "artifacts" / "final_centered_logits.pt", centered)
        atomic_torch_save(run_directory / "artifacts" / "final_labels.pt", labels)
        pairwise = pairwise_distance_matrix(centered)
        np.save(run_directory / "artifacts" / "pairwise_distance.npy", pairwise)
        pd.DataFrame(pairwise, index=np.arange(1, NUM_NODES + 1), columns=np.arange(1, NUM_NODES + 1)).to_csv(
            run_directory / "artifacts" / "pairwise_distance.csv", index_label="node_label"
        )
        np.savez_compressed(run_directory / "artifacts" / "confusion_matrices.npz", **confusions)
        if stage == "final" and not fully_symmetric:
            assignments = generate_random_partitions(
                count=int(cfg.analysis.diagnostic_partitions), seed=int(cfg.analysis.diagnostic_seed)
            )
            diagnostic = random_partition_diagnostic(
                centered,
                assignments,
                ratio_threshold=float(cfg.training.ratio_threshold),
            )
            np.savez_compressed(
                run_directory / "artifacts" / "random_partition_diagnostic.npz",
                assignments=assignments,
                R_true=diagnostic["R_true"],
                true_ratio_undefined=diagnostic["true_ratio_undefined"],
                R_random=diagnostic["R_random"],
                D_within_random=diagnostic["D_within_random"],
                D_between_random=diagnostic["D_between_random"],
                random_ratio_defined=diagnostic["random_ratio_defined"],
                p_perm=diagnostic["p_perm"],
            )
        else:
            np.savez_compressed(
                run_directory / "artifacts" / "random_partition_diagnostic.npz",
                skipped=True,
                reason=(
                    "reference_partition_is_not_the_true_complete_graph_orbit"
                    if fully_symmetric
                    else f"{stage}_validation_does_not_run_final_diagnostic"
                ),
            )
        save_checkpoint(
            run_directory / "checkpoints" / "final.pt",
            models,
            optimizers,
            scheduler,
            int(cfg.training.epochs),
            trainer.global_step,
            fingerprint,
            checkpoint_extra(int(cfg.training.epochs), terminal=terminal),
        )
        publication = publish_completion(
            run_directory, ScientificKey(str(cfg.data.name), str(cfg.model.name), condition_name, int(seed)),
            fingerprint, int(cfg.training.epochs), trainer.global_step,
        ) if active else {}
        atomic_write_json(
            run_directory / "run_status.json",
            {"status": "completed", "numerically_stable": True, "last_valid_epoch": int(cfg.training.epochs), **publication},
        )
        return RunResult(run_directory, True, terminal)
    except KeyboardInterrupt:
        atomic_write_json(
            run_directory / "run_status.json",
            {"status": "interrupted", "numerically_stable": True, "last_valid_epoch": int(last_saved_epoch),
             "recovery": "resume from the last confirmed epoch-boundary checkpoint"},
        )
        raise
    except FloatingPointError as error:
        pd.DataFrame(training_rows).to_csv(run_directory / "metrics" / "train_diagnostics.csv", index=False)
        pd.DataFrame(checkpoint_rows).to_csv(run_directory / "metrics" / "checkpoint_metrics.csv", index=False)
        last_valid_epoch = int(checkpoint_rows[-1]["epoch"]) if checkpoint_rows else 0
        failure = {
            "status": "diverged",
            "numerically_stable": False,
            "last_valid_epoch": last_valid_epoch,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_write_json(run_directory / "run_status.json", failure)
        terminal = {"numerically_stable": False, "condition": condition.name, "seed": int(seed), "error": str(error)}
        atomic_write_json(run_directory / "metrics" / "terminal_metrics.json", terminal)
        return RunResult(run_directory, False, terminal)
    except Exception as error:
        atomic_write_json(
            run_directory / "run_status.json",
            {
                "status": "failed",
                "numerically_stable": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


