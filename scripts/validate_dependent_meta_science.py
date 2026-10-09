"""Predeclared correlated known-V normal coverage; all failures are misses.

Development-only independent NumPy generator and QR reference. This is not
random-component/plugin/CR1 coverage, licensed-vendor or packaged-runtime QA.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import linalg
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


SEEDS = (729629, 729632)
REPLICATIONS = 200
MARGINAL_GATE = (0.89, 0.995)
GRID = tuple(itertools.product((-0.25, 0.65), (0.2, 4.0), (False, True)))
BETA = np.asarray([0.4, -0.3])
NUMERICAL_RTOL = 2e-6
NUMERICAL_ATOL = 2e-7


def design(correlation, spread, weak):
    rng = np.random.default_rng(632635)
    sizes = (2, 3, 4, 2, 3, 4, 2, 3)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    n = len(groups)
    moderator = rng.normal(size=n)
    if weak:
        moderator = 8.0 + 1e-4 * moderator
    x = np.column_stack([np.ones(n), moderator])
    variance = 0.3 * np.exp(np.linspace(-spread, spread, n))
    s = np.zeros((n, n))
    start = 0
    for size in sizes:
        rows = slice(start, start + size)
        scales = np.sqrt(variance[rows])
        rho = (1 - correlation) * np.eye(size) + correlation * np.ones((size, size))
        s[rows, rows] = scales[:, None] * rho * scales[None, :]
        start += size
    # Declare the actual complete matrix before either generator or oracle
    # uses it. Production receives this exact matrix and does not repair it.
    s = (s + s.T) * 0.5
    labels = [f"effect:{i}" for i in range(n)]
    data = pd.DataFrame(
        {
            "effect": labels,
            "study": [f"study:{g}" for g in groups],
            "x": moderator,
            "yi": np.zeros(n),
        }
    )
    covariance = pd.DataFrame(s, index=labels, columns=labels)
    lower = linalg.cholesky(s, lower=True)
    q, r = np.linalg.qr(linalg.solve_triangular(lower, x, lower=True), mode="reduced")
    inverse_r = linalg.solve_triangular(r, np.eye(2))
    reference_covariance = inverse_r @ inverse_r.T
    return data, covariance, x, lower, q, r, reference_covariance


def source_receipt(root):
    paths = [
        "src/openecon/econometrics/meta/dependent.py",
        "src/openecon/econometrics/meta/dependent_kernels.py",
        "src/openecon/econometrics/meta/dependent_post.py",
        "tests/test_dependent_meta_oracles.py",
        "scripts/validate_dependent_meta_science.py",
    ]
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    }


def validate():
    torch.set_num_threads(1)
    cells = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for correlation, spread, weak in GRID:
            data, covariance, x, lower, q, r, reference_covariance = design(
                correlation, spread, weak
            )
            marginal = np.zeros(2, dtype=int)
            joint = failed = 0
            failure_codes = {}
            max_beta_error = max_covariance_error = 0.0
            for _ in range(REPLICATIONS):
                # Draw the raw effects independently from production code;
                # every planned study/effect and trial stays in denominators.
                y = x @ BETA + lower @ rng.normal(size=len(x))
                data["yi"] = y
                expected_beta = linalg.solve_triangular(
                    r, q.T @ linalg.solve_triangular(lower, y, lower=True)
                )
                try:
                    fit = oe.meta_dependent(
                        data=data,
                        covariance=covariance,
                        study="study",
                        moderators=["x"],
                        model="common",
                    )
                    beta = fit["coefficients"].estimate.to_numpy(dtype=float)
                    actual_covariance = fit["covariance"].to_numpy(dtype=float)
                    np.testing.assert_allclose(
                        beta, expected_beta, rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL
                    )
                    np.testing.assert_allclose(
                        actual_covariance,
                        reference_covariance,
                        rtol=NUMERICAL_RTOL,
                        atol=NUMERICAL_ATOL,
                    )
                    max_beta_error = max(max_beta_error, float(abs(beta - expected_beta).max()))
                    max_covariance_error = max(
                        max_covariance_error,
                        float(abs(actual_covariance - reference_covariance).max()),
                    )
                    rows = fit["coefficients"]
                    covered = (rows.ci_low.to_numpy() <= BETA) & (BETA <= rows.ci_high.to_numpy())
                    marginal += covered
                    joint += int(covered.all())
                except (AnalysisError, AssertionError) as error:
                    failed += 1
                    code = (
                        error.code
                        if isinstance(error, AnalysisError)
                        else "independent_numerical_mismatch"
                    )
                    failure_codes[code] = failure_codes.get(code, 0) + 1
            rates = marginal / REPLICATIONS
            cell = {
                "seed": seed,
                "correlation": correlation,
                "log_variance_spread": spread,
                "weak_moderator": weak,
                "n_effects": len(data),
                "n_studies": 8,
                "planned": REPLICATIONS,
                "completed": REPLICATIONS - failed,
                "failed": failed,
                "failure_codes": failure_codes,
                "marginal_coverage_including_failures_as_misses": rates.tolist(),
                "marginal_binomial_mcse": np.sqrt(rates * (1 - rates) / REPLICATIONS).tolist(),
                "joint_rectangle_coverage_diagnostic_only": joint / REPLICATIONS,
                "max_beta_absolute_error": max_beta_error,
                "max_full_covariance_absolute_error": max_covariance_error,
                "passed": failed == 0
                and bool(((rates >= MARGINAL_GATE[0]) & (rates <= MARGINAL_GATE[1])).all()),
            }
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    return {
        "status": "passed" if all(cell["passed"] for cell in cells) else "failed",
        "protocol": "correlated-known-V-normal-common-GLS-v1",
        "seeds_predeclared": SEEDS,
        "replications_per_cell_predeclared": REPLICATIONS,
        "marginal_coverage_gate_predeclared": MARGINAL_GATE,
        "numerical_rtol_predeclared": NUMERICAL_RTOL,
        "numerical_atol_predeclared": NUMERICAL_ATOL,
        "planned_total": len(SEEDS) * len(GRID) * REPLICATIONS,
        "level": 0.95,
        "beta_true": BETA.tolist(),
        "cells": cells,
        "scope": "Complete independent study blocks; correlated normal errors, known unequal sampling covariance; common GLS only.",
        "not_established": [
            "estimated-variance-component/plugin interval coverage",
            "CR0/CR1 universal small-study calibration",
            "selection/bias models",
            "licensed vendor execution",
            "frozen/native/public release",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    accepted_source = source_receipt(root)
    receipt = validate()
    receipt["source"] = accepted_source
    receipt["source_readback"] = source_receipt(root)
    receipt["source_files_unchanged_during_run"] = (
        accepted_source["files"] == receipt["source_readback"]["files"]
    )
    if not receipt["source_files_unchanged_during_run"]:
        receipt["status"] = "failed_source_changed_during_run"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    if receipt["status"] != "passed":
        raise SystemExit("Predeclared dependent-meta scientific validation failed")
