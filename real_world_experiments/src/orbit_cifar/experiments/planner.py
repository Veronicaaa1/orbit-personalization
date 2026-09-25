"""Read-only matrix accounting and explicit execution prerequisites."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from orbit_cifar.data.datasets import load_dataset
from orbit_cifar.data.manifest import load_manifest, validate_manifest_pair, validate_manifest_source
from orbit_cifar.experiments.completion import verify_completed_run, verify_resumable_run
from orbit_cifar.experiments.families import family_config
from orbit_cifar.experiments.matrix import final_keys, legacy_allowlisted, pilot_directory, pilot_families, run_directory
from orbit_cifar.experiments.provenance import load_selected
from orbit_cifar.utils.io import atomic_write_json


@dataclass(frozen=True)
class ExecutionPlan:
    rows: list[dict[str, Any]]
    summary: dict[str, int]
    pilot_summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": self.rows, "summary": self.summary, "pilot_summary": self.pilot_summary}


def _status(path: Path, key: Any, cfg: DictConfig | None = None) -> tuple[str, str]:
    try:
        if (path / "legacy_reference.json").is_file():
            from orbit_cifar.legacy import validate_imported_run
            result = validate_imported_run(path, key, cfg)
            return ("imported", "validated immutable legacy reference") if result["valid"] else ("blocked", "; ".join(result["errors"]))
        if (path / "run_status.json").is_file():
            value = json.loads((path / "run_status.json").read_text(encoding="utf-8"))
            if value.get("status") == "completed":
                result = verify_completed_run(path, key, cfg)
                return ("complete_native", "verified native completion") if result["valid"] else ("blocked", "; ".join(result["errors"]))
            if value.get("status") in {"running", "interrupted"}:
                result = verify_resumable_run(path, key, cfg)
                return ("resumable_native", "verified partial native checkpoint") if result["valid"] else ("blocked", "; ".join(result["errors"]))
            return "blocked", f"existing native directory has status {value.get('status', 'unknown')}"
        if path.exists() and any(path.iterdir()):
            return "blocked", "nonempty unrecognized target directory"
        return "absent", "matrix member without native or imported output"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return "blocked", f"unreadable run evidence: {error}"


def _dataset_prerequisites(cfg: DictConfig) -> dict[str, str]:
    project = Path(str(cfg.project_root)).expanduser().resolve()
    failures: dict[str, str] = {}
    try:
        source = load_dataset(str(cfg.data.name), project / str(cfg.data.root), train=True, download=False)
        load_dataset(str(cfg.data.name), project / str(cfg.data.root), train=False, download=False)
    except (OSError, RuntimeError, ValueError) as error:
        return {"data": f"official local dataset unavailable: {error}"}
    root = project / str(cfg.manifest_root) / str(cfg.data.name)
    try:
        heterogeneous = load_manifest(root / "heterogeneous.json", targets=source.targets)
        validate_manifest_source(heterogeneous, source, str(cfg.data.name), str(cfg.data.dataset_version), int(cfg.data.split_seed))
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"manifest": f"heterogeneous manifest unavailable or invalid: {error}"}
    try:
        balanced = load_manifest(root / "balanced.json", targets=source.targets)
        validate_manifest_source(balanced, source, str(cfg.data.name), str(cfg.data.dataset_version), int(cfg.data.split_seed))
        validate_manifest_pair(heterogeneous, balanced, source.targets, str(cfg.data.name))
    except (OSError, ValueError, TypeError, KeyError) as error:
        failures["balanced_manifest"] = f"balanced manifest unavailable or inconsistent: {error}"
    return failures


def build_execution_plan(cfg: DictConfig, write: bool = False) -> ExecutionPlan:
    project = Path(str(cfg.project_root)).expanduser().resolve()
    root = project / str(cfg.output_root)
    families = {pair: family_config(cfg, *pair) for pair in pilot_families(cfg)}
    prerequisites = {dataset: _dataset_prerequisites(families[(dataset, str(cfg.final.models[0]))])
                     for dataset in map(str, cfg.final.datasets)}
    selections: dict[tuple[str, str], str | None] = {}
    legacy_selection_count = 0
    for pair, family in families.items():
        try:
            selected = load_selected(family, pilot_directory(root, *pair) / "selected_hparams.yaml")
            selections[pair] = None
            legacy_selection_count += int(selected.get("selection_source") == "legacy_documented_frozen")
        except (OSError, ValueError, TypeError, KeyError) as error:
            selections[pair] = str(error)
    rows: list[dict[str, Any]] = []
    for key in final_keys(cfg):
        target = run_directory(root, key)
        family = families[(key.dataset, key.model)]
        state, reason = _status(target, key, family)
        native_state = state
        needed: list[str] = []
        if state == "absent" and legacy_allowlisted(cfg, key):
            token = f"{key.dataset}/{key.model}/{key.condition}/seed_{key.seed:03d}"
            if token not in {str(value) for value in cfg.legacy_reuse.retrain_overrides}:
                state, reason = "blocked", "reserved legacy cell requires validated import; missing evidence never authorizes retraining"
        if state in {"absent", "resumable_native"}:
            failures = prerequisites[key.dataset]
            relevant = {name: message for name, message in failures.items()
                        if name != "balanced_manifest" or key.condition == "fully_symmetric"}
            if key.condition != "no_communication" and selections[(key.dataset, key.model)] is not None:
                relevant["selection"] = f"family selection unavailable or stale: {selections[(key.dataset, key.model)]}"
            needed = list(relevant.values())
            if relevant:
                state = "awaiting_data" if "data" in relevant else "awaiting_manifest" if any("manifest" in name for name in relevant) else "awaiting_selection"
                reason = "; ".join(needed)
            elif state == "absent":
                state, reason = "ready", "data, manifests and required selection validated"
        rows.append({**key.to_dict(), "matrix_member": True, "state": state, "native_state": native_state,
                     "execution_ready": state in {"ready", "resumable_native"}, "unmet_prerequisites": needed,
                     "reason": reason, "run_directory": str(target)})
    lineages: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row["state"] == "imported":
            lineage = json.loads((Path(row["run_directory"]) / "legacy_reference.json").read_text(encoding="utf-8")).get("original_run_path")
            if lineage:
                lineages.setdefault(str(lineage), []).append(index)
    for lineage, indices in lineages.items():
        if len(indices) > 1:
            for index in indices:
                rows[index].update(state="blocked", reason=f"duplicate legacy lineage: {lineage}", execution_ready=False)
    states = [row["state"] for row in rows]
    summary = {
        "expected_final_scientific_cells": len(rows), "validated_legacy_final_cells": states.count("imported"),
        "already_complete_native_cells": states.count("complete_native"), "resumable_native_cells": states.count("resumable_native"),
        "new_final_training_required": sum(row["native_state"] == "absent" and row["state"] != "blocked" for row in rows),
        "ready_cells": states.count("ready"), "awaiting_data_cells": states.count("awaiting_data"),
        "awaiting_manifest_cells": states.count("awaiting_manifest"), "awaiting_selection_cells": states.count("awaiting_selection"),
        "blocked_cells": states.count("blocked"),
    }
    reused_families = sum(pair == ("cifar10", "resnet20_gn") and not bool(cfg.legacy_reuse.rerun_pilot_family) for pair in families)
    candidate_count = len(cfg.pilot.lambda_grid) * len(cfg.pilot.seeds)
    pilot_summary = {
        "pilot_families": len(families), "conceptual_pilot_candidate_runs": len(families) * candidate_count,
        "families_using_legacy_selection": legacy_selection_count,
        "families_awaiting_selection": sum(error is not None for error in selections.values()),
        "new_pilot_candidate_runs": (len(families) - reused_families) * candidate_count,
    }
    plan = ExecutionPlan(rows, summary, pilot_summary)
    if write:
        atomic_write_json(root / "execution_plan.json", plan.to_dict())
        atomic_write_json(root / "experiment_manifest.json", {
            "experiment": str(cfg.experiment.name), "matrix": {"datasets": list(cfg.final.datasets), "models": list(cfg.final.models),
            "conditions": list(cfg.final.conditions), "seeds": list(cfg.final.seeds)},
            "expected_final_cells": len(rows), "pilot_families": len(families),
            "conceptual_pilot_cells": pilot_summary["conceptual_pilot_candidate_runs"],
            "expected_new_pilot_runs": pilot_summary["new_pilot_candidate_runs"],
        })
    return plan
