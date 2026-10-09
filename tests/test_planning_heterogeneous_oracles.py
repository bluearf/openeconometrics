"""Predeclared controlled diagnostics of actual Welch tests, not just nct fits."""

import math

import numpy as np
from scipy import stats

from openecon.econometrics.stats.planning_heterogeneous import power_welch


# Declared before execution. All finite trials stay in the denominator; no retry.
WELCH_PROTOCOL = {
    "seed": 529_202_610,
    "trials_per_cell": 40_000,
    "alpha": 0.05,
    "cells": [(8, 24, 1.0, 2.0), (24, 8, 2.0, 1.0), (40, 60, 1.0, 3.0), (80, 80, 1.0, 1.0)],
    "design_shifts": [0.0, 2.0],
    "absolute_approximation_tolerance": 0.025,
    "oracle": "scipy.stats.ttest_ind(equal_var=False), raw independent normal observations",
    "meaning": "bounded diagnostic; Satterthwaite/nct is still approximate, not universal coverage",
}


def welch_experiment():
    records = []
    p = WELCH_PROTOCOL
    for cell, (n1, n2, sd1, sd2) in enumerate(p["cells"]):
        se = math.sqrt(sd1**2 / n1 + sd2**2 / n2)
        for branch, shift in enumerate(p["design_shifts"]):
            rng = np.random.default_rng(p["seed"] + 10 * cell + branch)
            effect = shift * se
            first = rng.normal(0, sd1, (p["trials_per_cell"], n1))
            second = rng.normal(effect, sd2, (p["trials_per_cell"], n2))
            # This test estimates both sample variances and a new Welch df for every trial.
            result = stats.ttest_ind(second, first, axis=1, equal_var=False)
            valid = (
                np.isfinite(result.statistic) & np.isfinite(result.pvalue) & np.isfinite(result.df)
            )
            failures = int((~valid).sum())
            rejections = int((valid & (result.pvalue < p["alpha"])).sum())
            rate = rejections / p["trials_per_cell"]
            planned = power_welch(effect, sd1=sd1, sd2=sd2, n1=n1, n2=n2)["plan"].iloc[0]
            error = abs(rate - planned.power)
            records.append(
                dict(
                    n1=n1,
                    n2=n2,
                    sd1=sd1,
                    sd2=sd2,
                    effect=effect,
                    planned_df=planned.df,
                    design_shift=shift,
                    approximation_power=planned.power,
                    attempted=p["trials_per_cell"],
                    valid=int(valid.sum()),
                    failures=failures,
                    rejections=rejections,
                    actual_welch_rejection_rate=rate,
                    monte_carlo_se=math.sqrt(rate * (1 - rate) / p["trials_per_cell"]),
                    absolute_difference=error,
                    passed=bool(failures == 0 and error <= p["absolute_approximation_tolerance"]),
                )
            )
    return {"protocol": p, "cells": records, "all_passed": all(row["passed"] for row in records)}


def test_actual_welch_test_controlled_grid():
    receipt = welch_experiment()
    assert receipt["all_passed"], receipt
    assert len(receipt["cells"]) == 8
    assert sum(row["attempted"] for row in receipt["cells"]) == 320_000
    assert all(row["attempted"] == row["valid"] for row in receipt["cells"])
