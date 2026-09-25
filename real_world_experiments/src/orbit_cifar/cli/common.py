"""Shared argument and OmegaConf loading helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from orbit_cifar.config_schema import load_config
from orbit_cifar.utils.logging import configure_logging


def default_config_path() -> Path:
    """Locate the source-tree config for editable and repository execution."""
    return Path(__file__).resolve().parents[3] / "configs" / "mnist_cifar10_resnet_mlp_3seed.yaml"


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_config(parser: argparse.ArgumentParser, argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, DictConfig]:
    """Parse command flags and treat remaining items as OmegaConf dot-list overrides."""
    arguments, overrides = parser.parse_known_args(argv)
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        parser.error(f"Unrecognized arguments (configuration overrides require key=value): {invalid}")
    configure_logging(bool(arguments.verbose))
    return arguments, load_config(arguments.config, overrides)
