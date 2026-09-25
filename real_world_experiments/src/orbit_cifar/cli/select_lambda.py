"""Apply the pre-registered validation-only lambda selection rule."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.selection import select_from_csv


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Select the family Main communication strength from pilot validation")
    parser.add_argument("--pilot-runs", type=Path)
    parser.add_argument("--output-directory", type=Path)
    arguments, cfg = parse_config(parser, argv)
    project_root = Path(str(cfg.project_root)).expanduser().resolve()
    family = Path(str(cfg.data.name)) / str(cfg.model.name) if OmegaConf.select(cfg, "experiment.name") else Path()
    source = arguments.pilot_runs or project_root / str(cfg.output_root) / "pilot" / family / "pilot_runs.csv"
    output = arguments.output_directory or project_root / str(cfg.output_root) / "pilot" / family
    selected = select_from_csv(
        source,
        output,
        within_fraction_max=float(cfg.pilot.within_fraction_max),
        between_fraction_min=float(cfg.pilot.between_fraction_min),
        accuracy_drop_max=float(cfg.pilot.accuracy_drop_max),
        cfg=cfg,
    )
    print(selected)
    return 2 if selected.get("lambda_main") is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
