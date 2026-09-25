"""Run one condition/seed or the tiny no-download smoke pipeline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from omegaconf import OmegaConf

from orbit_cifar.cli.common import base_parser, parse_config
from orbit_cifar.experiments.runner import build_smoke_inputs, run_experiment, smoke_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = base_parser("Train one synchronous decentralized run")
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stage", choices=["pilot", "preflight", "final"])
    parser.add_argument("--lambda-override", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments, cfg = parse_config(parser, argv)
    cfg.resume = bool(arguments.resume)
    seed = int(cfg.seed) if arguments.seed is None else arguments.seed
    if arguments.smoke:
        from orbit_cifar.cli.verify_run import verify_run

        with TemporaryDirectory(prefix="orbit-smoke-") as smoke_root:
            cfg = smoke_config(cfg, output_root=smoke_root)
            result = run_experiment(
                cfg, condition_name="no_communication", seed=seed, stage="final",
                lambda_override=0.0, run_name="smoke", datasets=build_smoke_inputs(seed=seed),
            )
            verified = verify_run(result.run_directory)
            print(f"Temporary smoke: {'PASS' if verified['valid'] and result.numerically_stable else 'FAIL'}")
            return 0 if verified["valid"] and result.numerically_stable else 2
    else:
        result = run_experiment(
            cfg,
            condition_name=arguments.condition or str(cfg.condition.name),
            seed=seed,
            stage=arguments.stage or str(cfg.stage),
            lambda_override=arguments.lambda_override,
        )
    print(f"Run directory: {result.run_directory}")
    print(f"Numerically stable: {result.numerically_stable}")
    return 0 if result.numerically_stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
