"""Independent natural-coordinate ML/OIM audit of the six-indicator QA fixture.

Development oracle only: imports neither OpenEconometrics nor Torch. Runtime
source, native UI and licensed vendor evidence are separate proof layers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def moments(v):
    lam = np.array([[1.0, 0.0], [v[0], 0.0], [v[1], 0.0], [0.0, 1.0], [0.0, v[2]], [0.0, v[3]]])
    phi = np.array([[v[4], v[5]], [v[5], v[6]]])
    return lam @ phi @ lam.T + np.diag(v[7:]), lam, phi


def objective(v, s, n):
    sigma, lam, phi = moments(v)
    if np.linalg.eigvalsh(phi).min() <= 0 or v[7:].min() <= 0:
        return 1e50, np.zeros(13)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        return 1e50, np.zeros(13)
    inv = np.linalg.inv(sigma)
    weight = inv - inv @ s @ inv
    derivative = []
    for j, f in ((1, 0), (2, 0), (4, 1), (5, 1)):
        dl = np.zeros((6, 2))
        dl[j, f] = 1
        derivative.append(dl @ phi @ lam.T + lam @ phi @ dl.T)
    for i, j in ((0, 0), (1, 0), (1, 1)):
        dp = np.zeros((2, 2))
        dp[i, j] = dp[j, i] = 1
        derivative.append(lam @ dp @ lam.T)
    for j in range(6):
        dt = np.zeros((6, 6))
        dt[j, j] = 1
        derivative.append(dt)
    return n / 2 * (6 * math.log(2 * math.pi) + logdet + np.trace(inv @ s)), n / 2 * np.array(
        [np.sum(weight * d) for d in derivative]
    )


def compare(actual, expected, name, unit=None):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise AssertionError(name + ": invalid numeric dimensions or finiteness")
    if unit is None:
        unit = np.maximum(np.abs(expected), np.finfo(float).tiny)
    normalized = np.abs(actual - expected) / unit
    if np.any(normalized > 5e-5):
        raise AssertionError(name + ": independently normalized numerical disagreement")
    return {
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "maximum_unit_normalized_error": float(np.max(normalized)),
    }


def verify(value):
    state = value["payload"] if set(value) == {"schema_version", "payload"} else value
    if state["schema"] != "openecon.cfa.v1" or state["spec"]["identification"] != "marker":
        raise AssertionError("The declared QA oracle requires marker-identified CFA state")
    spec, source, result = state["spec"], state["source"], state["results"]
    if (
        spec["columns"] != ["x1", "x2", "x3", "x4", "x5", "x6"]
        or spec["factor_names"] != ["z_factor", "a_factor"]
        or spec["anchors"] != ["x1", "x4"]
        or spec["free_loadings"] != [[1, 0], [2, 0], [4, 1], [5, 1]]
    ):
        raise AssertionError(
            "This independent fixture does not certify arbitrary latent specifications"
        )
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "digest"},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if digest != state["digest"]:
        raise AssertionError("CFA saved-state digest differs")
    if source["kind"] == "raw":
        y = np.asarray([[np.nan if v is None else v for v in row] for row in source["rows"]])
        positions = np.flatnonzero(np.isfinite(y).all(1))
        if positions.tolist() != source["positions"]:
            raise AssertionError("CFA physical complete-case positions differ")
        centered = y[positions] - y[positions].mean(0)
        s = centered.T @ centered / len(positions)
        n = len(positions)
    else:
        n = source["original_n"]
        s = np.asarray(source["covariance"])
        if source["divisor"] == "n-1":
            s = s * (n - 1) / n
    initial = np.r_[0.8, 0.7, 0.9, 0.6, 1.0, 0.3, 1.2, 0.4, 0.5, 0.6, 0.3, 0.4, 0.5]
    fit = minimize(
        lambda v: objective(v, s, n),
        initial,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    if np.max(np.abs(objective(fit.x, s, n)[1])) > 2e-5:
        raise AssertionError("Independent natural-parameter optimizer did not converge")
    hess = np.empty((13, 13))
    for j in range(13):
        step = 2e-5 * max(1.0, abs(fit.x[j]))
        delta = np.zeros(13)
        delta[j] = step
        hess[:, j] = (objective(fit.x + delta, s, n)[1] - objective(fit.x - delta, s, n)[1]) / (
            2 * step
        )
    covariance = np.linalg.inv((hess + hess.T) / 2)
    sigma, lam, phi = moments(fit.x)
    units = np.sqrt(np.diag(covariance))
    checks = {
        "parameters": compare(result["parameters"], fit.x, "parameters"),
        "full_covariance": compare(
            result["covariance"], covariance, "full covariance", units[:, None] * units[None, :]
        ),
        "implied_covariance": compare(
            result["implied_covariance"],
            sigma,
            "implied covariance",
            np.sqrt(np.diag(sigma))[:, None] * np.sqrt(np.diag(sigma))[None, :],
        ),
        "loadings": compare(result["loadings"], lam, "loadings", np.maximum(np.abs(lam), 1.0)),
        "factor_covariance": compare(result["factor_covariance"], phi, "latent covariance"),
        "log_likelihood": compare(result["log_likelihood"], -fit.fun, "normalized likelihood"),
        "sample_covariance_ml": compare(
            result["sample_covariance_ml"], s, "normal ML source covariance"
        ),
    }
    critical = norm.isf((1 - state["options"]["level"]) / 2)
    intervals = np.c_[fit.x - critical * units, fit.x + critical * units]
    return {
        "status": "passed",
        "state_sha256": digest,
        "n": n,
        "parameter_count": 13,
        "comparisons": checks,
        "independent_estimates": fit.x.tolist(),
        "independent_full_covariance": covariance.tolist(),
        "independent_intervals": intervals.tolist(),
        "reference": "independent NumPy natural covariance ML and finite differences of full observed score",
        "licensed_vendor_execution": False,
        "source_native_provenance": "requires separate receipts",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.state.read_text()))
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparisons": len(result["comparisons"]),
                "report": str(args.report),
            }
        )
    )


if __name__ == "__main__":
    main()
