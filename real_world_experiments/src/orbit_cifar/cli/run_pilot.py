"""Run the validation-only lambda sweep."""

from __future__ import annotations

from typing import Sequence

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.pilot import run_pilot


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Run the balanced-validation lambda pilot")
    _, cfg = parse_config(parser, argv)
    frame = run_pilot(cfg)
    print(f"Completed {len(frame)} pilot runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

