"""Run the paired active final matrix."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.final import run_final


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Run the paired four-condition active final experiment")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int)
    arguments, cfg = parse_config(parser, argv)
    if hasattr(cfg.final, "datasets"):
        from orbit_cifar.experiments.final import run_active_cells
        frame = run_active_cells(cfg, arguments.dataset, arguments.model, arguments.seed)
    else:
        frame = run_final(cfg)
    trained = frame[frame["action"].isin(["trained", "resumed"])] if "action" in frame else frame
    stable = int(trained.get("numerically_stable", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    print(f"Planned cells: {len(frame)}; trained/resumed: {len(trained)}; numerically stable: {stable}")
    waiting = "action" in frame and frame["action"].eq("skipped_blocked").any()
    return 0 if stable == len(trained) and not waiting else 2


if __name__ == "__main__":
    raise SystemExit(main())
