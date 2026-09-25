from __future__ import annotations

import json
from typing import Sequence

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.preflight import preflight_rows, run_preflights


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("List the 24 preflights; execute only with --execute")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--condition")
    parser.add_argument("--output-dir", help="Explicit directory for retained preflight artifacts; otherwise temporary")
    arguments, cfg = parse_config(parser, argv)
    if not arguments.execute:
        rows = preflight_rows(cfg)
        print(json.dumps({"status": "NOT RUN", "reason": "--execute was not supplied", "combinations": rows}, indent=2))
        return 0
    frame = run_preflights(cfg, dataset=arguments.dataset, model=arguments.model,
                          condition=arguments.condition, output_directory=arguments.output_dir)
    passed = int((frame.status == "PASS").sum())
    print(f"Preflights PASS={passed}, FAIL={len(frame)-passed}, expected={len(frame)}")
    return 0 if passed == len(frame) else 2


if __name__ == "__main__":
    raise SystemExit(main())
