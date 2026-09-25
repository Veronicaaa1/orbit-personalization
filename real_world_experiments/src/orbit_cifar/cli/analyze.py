"""Aggregate runs and generate the complete reports directory."""

from __future__ import annotations

from typing import Sequence

from orbit_cifar.analysis.run import analyze
from orbit_cifar.cli.common import base_parser, parse_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Aggregate valid runs and generate figures, tables, and summary")
    _, cfg = parse_config(parser, argv)
    bundle = analyze(cfg)
    valid = int(bundle.run_inventory["valid"].sum()) if not bundle.run_inventory.empty else 0
    missing = int((~bundle.run_inventory["valid"]).sum()) if not bundle.run_inventory.empty else 0
    print(f"Analysis complete: valid runs={valid}, missing/failed/incomplete={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

