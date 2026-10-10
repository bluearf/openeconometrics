#!/usr/bin/env python3
"""Development-only independent LP/matrix audit of a complete IVQR state.

This program imports neither OpenEconometrics nor Torch. SciPy solves each
saved adjusted-outcome LP afresh; NumPy independently constructs the complete
Powell covariance, instrument Wald law and structural influence covariance.
Published-equation numerical comparison is not licensed Stata execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import chi2


def compare(actual, saved, name):
    actual, saved = np.asarray(actual, dtype=float), np.asarray(saved, dtype=float)
    if actual.shape != saved.shape or not np.isfinite(saved).all():
        raise AssertionError(f"{name}: invalid saved dimensions/finite values")
    np.testing.assert_allclose(actual, saved, rtol=2e-7, atol=2e-8, err_msg=name)
    return float(np.max(np.abs(actual - saved))) if actual.size else 0.0


def verify(state):
    if state.get("schema") == "openecon.summary.v1":
        state = state["attrs"]["state"]
    if state["schema"] != "openecon.ivquantile.v1":
        raise AssertionError("Invalid IVQR schema")
    raw = dict(state)
    sha = raw.pop("sha256")
    canonical = json.dumps(raw, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != sha:
        raise AssertionError("State integrity mismatch")
    columns = {v["name"]: np.array([np.nan if x is None else x for x in v["values"]], dtype=float)
               for v in state["source"]["columns"]}
    cfg, spec = state["config"], state["spec"]
    positions = state["sample_positions"]
    y, d = columns[spec["y"]][positions], columns[spec["endogenous"]][positions]
    xparts = ([np.ones(len(y))] if cfg["intercept"] else []) + [columns[v][positions] for v in spec["x"]]
    x = np.column_stack(xparts) if xparts else np.empty((len(y), 0))
    z = np.column_stack([columns[v][positions] for v in spec["instruments"]])
    w = np.c_[x, z]
    n, k = w.shape
    zr = z - x @ np.linalg.solve(x.T @ x, x.T @ z) if x.shape[1] else z
    criterion_weight = zr.T @ zr / n
    tau, kx = cfg["quantile"], x.shape[1]
    maxima, profile_checks = {}, []
    for record in state["evaluations"]:
        adjusted = y - d * record["alpha"]
        objective = np.r_[np.zeros(k), np.full(n, tau), np.full(n, 1 - tau)]
        fit = linprog(objective, A_eq=np.c_[w, np.eye(n), -np.eye(n)], b_eq=adjusted,
                      bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
        if not fit.success:
            raise AssertionError("Independent LP failed")
        beta = fit.x[:k]
        residual = adjusted - w @ beta
        if cfg["bandwidth"] is None:
            spread = min(np.std(residual, ddof=1), np.diff(np.quantile(residual, [.25, .75]))[0] / 1.349)
            h = 1.06 * spread * n**(-0.2)
        else:
            h = cfg["bandwidth"]
        weights = (np.abs(residual) <= h) / (2 * h)
        bread = w.T @ (weights[:, None] * w)
        inv = np.linalg.inv(bread)
        covariance = tau * (1 - tau) * inv @ (w.T @ w) @ inv.T
        gamma = beta[kx:]
        stat = gamma @ np.linalg.solve(covariance[kx:, kx:], gamma)
        expected = dict(coefficients=beta, qr_objective=fit.fun, bandwidth=h,
                        bread=bread, covariance=covariance, statistic=stat,
                        p_value=chi2.sf(stat, z.shape[1]),
                        criterion=gamma @ criterion_weight @ gamma)
        errors = {name: compare(value, record[name], name) for name, value in expected.items()}
        for name, error in errors.items():
            maxima[name] = max(maxima.get(name, 0), error)
        profile_checks.append(dict(alpha=record["alpha"], complete_lp_and_covariance=True,
                                   density_support=int(np.count_nonzero(weights))))
    selected = state["evaluations"][state["selected_evaluation"]]
    covariance_check = None
    if state["joint"] is not None:
        eta = np.asarray(selected["coefficients"])
        residual = y - d * selected["alpha"] - w @ eta
        weights = (np.abs(residual) <= selected["bandwidth"]) / (2 * selected["bandwidth"])
        bread = w.T @ (weights[:, None] * w)
        derivative = -np.linalg.solve(bread, w.T @ (weights * d))
        t = derivative[kx:]
        a = -(t @ criterion_weight) / (t @ criterion_weight @ t)
        influence = np.zeros((kx + 1, k))
        influence[0, kx:] = a
        influence[1:, :kx] = np.eye(kx)
        influence[1:] += derivative[:kx, None] * influence[0, None]
        inv = np.linalg.inv(bread)
        nuisance_c = tau * (1 - tau) * inv @ (w.T @ w) @ inv.T
        joint = influence @ nuisance_c @ influence.T
        covariance_check = {
            name: compare(value, state["joint"][name], name)
            for name, value in dict(covariance=joint, profile_derivative=derivative,
                                    influence_map=influence).items()
        }
    return dict(status="passed", input_state_sha256=sha, profiles_checked=len(profile_checks),
                complete_profile_checks=profile_checks, maximum_absolute_errors=maxima,
                complete_joint_covariance_check=covariance_check,
                oracle="SciPy HiGHS LP and independently assembled NumPy Powell/CH profile matrices",
                cached_state_comparison=True,
                input_runtime_provenance="requires separate source/frozen/native receipt",
                licensed_vendor_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(json.loads(args.state.read_text()))
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.report:
        args.report.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
