from __future__ import annotations

import pandas as pd
import numpy as np

from orbit_cifar.experiments.selection import select_lambdas, summarize_pilot_runs


def pilot_rows() -> pd.DataFrame:
    rows = []
    values = {
        0.0: (1.0, 1.0, 0.50, 0.80),
        0.1: (0.70, 0.70, 0.50, 0.79),
        0.3: (0.50, 0.40, 0.56, 0.78),
        1.0: (0.20, 0.10, 0.67, 0.60),
    }
    for value, (within, between, ratio, accuracy) in values.items():
        for seed in (0, 1, 2):
            rows.append(
                {
                    "lambda": value,
                    "seed": seed,
                    "numerically_stable": True,
                    "D_within_bar": within,
                    "D_between_bar": between,
                    "R_orbit_bar": ratio,
                    "orbit_ensemble_accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows)


def test_primary_selection_is_smallest_qualified_and_large_is_max_stable() -> None:
    summary = summarize_pilot_runs(pilot_rows())
    selected, diagnostics = select_lambdas(summary)
    assert selected["selection_status"] == "primary"
    assert selected["lambda_main"] == 0.1
    assert selected["lambda_large"] == 1.0
    assert not selected["extrapolated_large_lambda"]
    assert np.allclose(
        [diagnostics["baseline"][key] for key in ("Dw0", "Db0", "Acc0")],
        [1.0, 1.0, 0.8],
    )


def test_fallback_is_disclosed() -> None:
    rows = pilot_rows()
    rows.loc[rows["lambda"] > 0, "D_between_bar"] = 0.1
    summary = summarize_pilot_runs(rows)
    selected, diagnostics = select_lambdas(summary)
    assert selected["selection_status"] == "fallback"
    assert diagnostics["fallback_scores"]
