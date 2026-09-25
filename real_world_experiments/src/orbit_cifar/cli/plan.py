"""Print the active experiment's non-mutating execution plan."""

from __future__ import annotations

import json
from typing import Sequence

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.planner import build_execution_plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Print the 48-cell execution plan without training")
    parser.add_argument("--write", action="store_true", help="also write plan and experiment manifest")
    arguments, cfg = parse_config(parser, argv)
    plan = build_execution_plan(cfg, write=arguments.write)
    print(json.dumps({**plan.summary, **plan.pilot_summary}, indent=2))
    for row in plan.rows:
        print(f"{row['dataset']}/{row['model']}/{row['condition']}/seed_{row['seed']:03d}: {row['state']} - {row['reason']}")
    return 2 if plan.summary["blocked_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
