"""Development-only NumPy oracle and bounded IID PLR calibration evidence.

Run with the source checkout on PYTHONPATH. No external estimator is needed.
This is a deliberately bounded Monte Carlo experiment, not uniform coverage
proof or Stata parity. Output is a reviewable scientific receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import openecon as oe


def ridge_reference(x, y, test, state):
    center = x.mean(0)
    scale = np.sqrt(np.mean((x - center) ** 2, axis=0))
    scale[scale == 0] = 1
    z = (x - center) / scale
    b = np.linalg.solve(
        z.T @ z / len(y) + state["selected_penalty"] * np.eye(x.shape[1]),
        z.T @ (y - y.mean()) / len(y),
    )
    return (test - center) / scale @ b + y.mean()


def independent_target(result, df, names):
    x, y, d = df[names].to_numpy(), df.y.to_numpy(), df.d.to_numpy()
    if result.spec.estimator == "postdouble":
        controls = np.column_stack(
            (np.ones(len(df)), df[result.extra["selected_controls"]].to_numpy())
        )
        projected = controls @ np.linalg.lstsq(controls, np.column_stack((y, d)), rcond=None)[0]
        lhat, mhat = projected[:, 0], projected[:, 1]
    elif result.spec.estimator == "dmlplr":
        lhat, mhat = np.empty(len(df)), np.empty(len(df))
        for fold in result.extra["fold_records"]:
            train, test = np.array(fold["train_positions"]), np.array(fold["test_positions"])
            assert not set(train) & set(test)
            lhat[test] = ridge_reference(x[train], y[train], x[test], fold["outcome_model"])
            mhat[test] = ridge_reference(x[train], d[train], x[test], fold["treatment_model"])
    else:
        state = result.extra["fold_records"][0]
        lhat = (
            x @ np.array(state["outcome_model"]["coefficients"])
            + state["outcome_model"]["constant"]
        )
        mhat = (
            x @ np.array(state["treatment_model"]["coefficients"])
            + state["treatment_model"]["constant"]
        )
    residual_d, residual_y = d - mhat, y - lhat
    theta = residual_d @ residual_y / (residual_d @ residual_d)
    influence = residual_d * (residual_y - theta * residual_d) / np.mean(residual_d**2)
    variance = np.mean(influence**2) / len(df)
    return theta, variance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=64)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/validation/market127_regularized.json")
    )
    args = parser.parse_args()
    if args.replications < 20:
        raise ValueError("At least 20 replications are needed for this bounded diagnostic.")
    started = time.perf_counter()
    theta_true, n, seed = 1.25, 320, 48197
    rng = np.random.default_rng(seed)
    records = {name: [] for name in ("postdouble", "partiallingout", "dmlplr")}
    max_target_error, max_variance_error = 0.0, 0.0
    for replication in range(args.replications):
        x = rng.normal(size=(n, 5))
        d = 0.5 * x[:, 0] + 0.35 * x[:, 1] + rng.normal(size=n)
        g = 0.5 * x[:, 0] - 0.2 * x[:, 2] + 0.1 * x[:, 4]
        y = theta_true * d + g + (0.7 + 0.3 * abs(x[:, 0])) * rng.normal(size=n)
        names = [f"x{j}" for j in range(5)]
        df = pd.DataFrame(x, columns=names)
        df["d"], df["y"] = d, y
        for name in records:
            options = dict(
                data=df, y="y", treatment="d", x=names, covariance="HC0", seed=seed + replication
            )
            if name == "dmlplr":
                options.update(nuisance="ridge", selection="fixed", penalty=0.001, folds=4)
            else:
                options.update(selection="plugin", plugin_iterations=50)
            result = getattr(oe, name)(**options)
            ref_theta, ref_var = independent_target(result, df, names)
            max_target_error = max(
                max_target_error, abs(ref_theta - result.coefficients[0].estimate)
            )
            max_variance_error = max(
                max_variance_error, abs(ref_var - result.covariance_matrix[0][0])
            )
            coefficient = result.coefficients[0]
            records[name].append(
                {
                    "estimate": coefficient.estimate,
                    "standard_error": coefficient.std_error,
                    "covered": coefficient.ci_low <= theta_true <= coefficient.ci_high,
                }
            )
    summary = {}
    for name, rows in records.items():
        estimates, ses = (
            np.array([row["estimate"] for row in rows]),
            np.array([row["standard_error"] for row in rows]),
        )
        empirical_sd = float(estimates.std(ddof=1))
        coverage = sum(row["covered"] for row in rows) / len(rows)
        # Binomial Monte Carlo uncertainty: Wilson interval, not a claim that
        # each individual confidence interval has finite-sample exact coverage.
        z, rep = 1.96, len(rows)
        midpoint = (coverage + z * z / (2 * rep)) / (1 + z * z / rep)
        half = (
            z
            * np.sqrt(coverage * (1 - coverage) / rep + z * z / (4 * rep * rep))
            / (1 + z * z / rep)
        )
        summary[name] = {
            "mean_bias": float(estimates.mean() - theta_true),
            "empirical_sd": empirical_sd,
            "mean_standard_error": float(ses.mean()),
            "se_to_empirical_sd": float(ses.mean() / empirical_sd),
            "coverage_95": coverage,
            "coverage_monte_carlo_wilson_95": [float(midpoint - half), float(midpoint + half)],
        }
    report = {
        "issue": "MARKET-127",
        "target": "single IID PLR effect",
        "oracle": "independent NumPy ridge/FWL projections and influence sandwich",
        "seed": seed,
        "nobs_per_replication": n,
        "replications": args.replications,
        "true_theta": theta_true,
        "dgp": "Independent normal controls/treatment innovations; sparse linear conditional means; heteroskedastic outcome noise .7+.3*abs(x0)",
        "methods": summary,
        "max_target_oracle_error": max_target_error,
        "max_variance_oracle_error": max_variance_error,
        "elapsed_seconds": time.perf_counter() - started,
        "stata_parity_validated": False,
        "limitations": [
            "One prespecified IID sparse-linear DGP and finite replication count; no uniform coverage claim.",
            "No cluster/panel/time-series, frozen desktop runtime, release or deployed UI validation.",
        ],
    }
    assert max_target_error < 1e-9 and max_variance_error < 1e-10
    for method in summary.values():
        assert abs(method["mean_bias"]) < 0.1
        assert 0.6 < method["se_to_empirical_sd"] < 1.5
        assert 0.80 <= method["coverage_95"] <= 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
