"""Dataset/model family composition without architecture-dependent data streams."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from orbit_cifar.config_schema import validate_config
from orbit_cifar.constants import ACTIVE_DATASETS, ACTIVE_MODELS


def family_config(cfg: DictConfig, dataset: str, model: str) -> DictConfig:
    if dataset not in ACTIVE_DATASETS or model not in ACTIVE_MODELS:
        raise ValueError(f"Unsupported active family: {dataset}/{model}")
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    metadata = OmegaConf.select(cfg, "_composition")
    if metadata is None:
        if str(cfg.data.name) != dataset or str(cfg.model.name) != model:
            raise ValueError("Changing families requires a config produced by load_config")
        validate_config(result)
        return result
    for group, name in (("data", dataset), ("model", model)):
        current = OmegaConf.to_container(cfg[group], resolve=True)
        baseline = OmegaConf.to_container(metadata[f"resolved_{group}"], resolve=True)
        explicit = OmegaConf.to_container(metadata.explicit.get(group, OmegaConf.create({})), resolve=True)
        # Programmatic edits have the same authority as CLI overrides.
        explicit.update({key: value for key, value in current.items() if value != baseline.get(key)})
        explicit.pop("name", None)
        composed = OmegaConf.merge(metadata[f"{group}_defaults"][name], explicit)
        composed.name = name
        result[group] = composed
        result._composition.explicit[group] = explicit
    validate_config(result)
    result._composition.resolved_data = OmegaConf.to_container(result.data, resolve=True)
    result._composition.resolved_model = OmegaConf.to_container(result.model, resolve=True)
    return result
