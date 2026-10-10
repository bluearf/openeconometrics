"""Independent observation-space NIG integration and multivariate-t density audit.

Development/reference dependencies only: NumPy, SciPy and mpmath. This auditor
imports neither OpenEconometrics nor Torch and does not call production kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_t, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if len(raw) > 32 * 1024**2:
        raise ValueError("The independent audit input exceeds 32 MiB.")
    document = json.loads(raw)
    state = document.get("payload", document)
    if state["schema_version"] != "openecon.posterior_hypothesis_comparison.v1":
        raise ValueError("Unknown comparison schema.")
    alternative, null = state["alternative"], state["results"]["null_posterior"]
    source = alternative["state"]
    positions = source["sample_positions"]
    if not 1 <= len(positions) <= 200:
        raise ValueError(
            "The independent dense observation-space audit supports 1 to 200 sample rows."
        )
    block = np.asarray(
        [[column[i] for column in source["source_values"]] for i in positions], dtype=float
    )
    y, x = block[:, 0], block[:, 1:]
    if source["intercept"]:
        x = np.column_stack((np.ones(len(y)), x))
    prior = alternative["prior"]
    m, v = np.asarray(prior["mean"]), np.asarray(prior["scale_matrix"])
    c, d = np.asarray(state["constraints"]), np.asarray(state["values"])
    r, k = c.shape
    g = c @ v @ c.T
    delta = d - c @ m
    center = m + v @ c.T @ np.linalg.solve(g, delta)
    vc = v - v @ c.T @ np.linalg.solve(g, c @ v)
    if r == k:
        vc = np.zeros((k, k))
    ac, bc = prior["shape"] + r / 2, prior["scale"] + delta @ np.linalg.solve(g, delta) / 2
    omega = np.eye(len(y)) + x @ vc @ x.T
    e = y - x @ center
    an, bn = ac + len(y) / 2, bc + e @ np.linalg.solve(omega, e) / 2
    mean = center + vc @ x.T @ np.linalg.solve(omega, e)
    conditional = vc - vc @ x.T @ np.linalg.solve(omega, x @ vc)
    covariance = conditional * (bn / (prior["shape"] + (r + len(y) - 2) / 2))
    evidence = (
        -len(y) * math.log(2 * math.pi) / 2
        - np.linalg.slogdet(omega)[1] / 2
        + ac * math.log(bc)
        - an * math.log(bn)
        + gammaln(an)
        - gammaln(ac)
    )
    prior_density = multivariate_t.logpdf(
        d, loc=c @ m, shape=c @ v @ c.T * prior["scale"] / prior["shape"], df=2 * prior["shape"]
    )
    post_v = np.asarray(alternative["conditional_scale_matrix"])
    posterior_density = multivariate_t.logpdf(
        d,
        loc=c @ alternative["mean"],
        shape=c @ post_v @ c.T * alternative["scale"] / alternative["shape"],
        df=2 * alternative["shape"],
    )
    density_log_bf = posterior_density - prior_density
    # High precision integrates the original prior and null directly, without
    # using saved full-posterior fields or the production QR coordinate chart.
    with mp.workdps(90):
        xm, ym, mm, vm, cm, dm = map(
            mp.matrix, (x.tolist(), y.tolist(), m.tolist(), v.tolist(), c.tolist(), d.tolist())
        )
        aa, bb = mp.mpf(prior["shape"]), mp.mpf(prior["scale"])

        def mass(location, conditional_v, shape, scale):
            vv = mp.eye(len(y)) + xm * conditional_v * xm.T
            residual = ym - xm * location
            next_a = shape + mp.mpf(len(y)) / 2
            next_b = scale + (residual.T * (vv**-1) * residual)[0] / 2
            return (
                -len(y) * mp.log(2 * mp.pi) / 2
                - mp.log(mp.det(vv)) / 2
                + shape * mp.log(scale)
                - next_a * mp.log(next_b)
                + mp.loggamma(next_a)
                - mp.loggamma(shape)
            )

        gap, gram = dm - cm * mm, cm * vm * cm.T
        constrained_m = mm + vm * cm.T * (gram**-1) * gap
        constrained_v = vm - vm * cm.T * (gram**-1) * cm * vm
        constrained_a, constrained_b = aa + mp.mpf(r) / 2, bb + (gap.T * (gram**-1) * gap)[0] / 2
        mp_null, mp_full = (
            mass(constrained_m, constrained_v, constrained_a, constrained_b),
            mass(mm, vm, aa, bb),
        )
        high_precision_bf = float(mp_null - mp_full)
        high_precision_null = float(mp_null)
    checks = {}

    def check(name, actual, expected, *, tolerance=2e-10):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all():
            raise AssertionError(name + " availability/shape/finite mismatch")
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        scale = max(float(np.max(np.abs(b))) if b.size else 0.0, 1.0)
        if error > tolerance * scale:
            raise AssertionError(f"{name}: {error} exceeds {tolerance * scale}")
        checks[name] = {"max_absolute_error": error, "tolerance": tolerance * scale}

    for field, expected in (
        ("prior_mean", center),
        ("prior_conditional_scale_matrix", vc),
        ("mean", mean),
        ("conditional_scale_matrix", conditional),
        ("coefficient_covariance", covariance),
        ("shape", an),
        ("scale", bn),
    ):
        check(field, null[field], expected)
    if prior["shape"] < 1e5:
        check("observation_space_log_null", null["log_marginal_likelihood"], evidence)
        check(
            "scipy_original_constraint_density_ratio",
            state["results"]["log_bayes_factor_null_alternative"],
            density_log_bf,
        )
    check("90_digit_log_null", null["log_marginal_likelihood"], high_precision_null)
    check(
        "90_digit_log_bayes_factor",
        state["results"]["log_bayes_factor_null_alternative"],
        high_precision_bf,
    )
    check("constraint_support_mean", c @ np.asarray(null["mean"]), d)
    check(
        "constraint_support_covariance",
        c @ np.asarray(null["conditional_scale_matrix"]),
        np.zeros((r, k)),
    )
    if r < k:
        critical = t.isf(state["alpha"] / 2, 2 * an)
        diagonal = np.diag(conditional * bn / an)
        check(
            "credible_intervals",
            null["credible_intervals"],
            np.column_stack(
                (
                    mean - critical * np.sqrt(np.maximum(diagonal, 0)),
                    mean + critical * np.sqrt(np.maximum(diagonal, 0)),
                )
            ),
        )
    probability = state["results"]["posterior_probability_null"]
    if state["model_prior_odds_null_alternative"] is None:
        if probability is not None:
            raise AssertionError(
                "Posterior model probability available without declared model prior odds"
            )
    else:
        log_odds = high_precision_bf + math.log(state["model_prior_odds_null_alternative"])
        expected = (
            1 / (1 + math.exp(-log_odds))
            if log_odds >= 0
            else math.exp(log_odds) / (1 + math.exp(log_odds))
        )
        check("declared_model_odds_probability", probability, expected)
    receipt = {
        "passed": True,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "comparison_digest": state["digest"],
        "sample_rows": len(y),
        "constraint_rank": r,
        "free_dimension": k - r,
        "reference_target": "proper alternative joint NIG prior conditioned on C beta = d",
        "references": [
            "independent observation-space Gaussian/IG integration",
            "SciPy original-coordinate multivariate Student-t density ratio",
            "90-digit mpmath original-prior marginal integration",
        ],
        "checks": checks,
        "production_imported": False,
    }
    args.report.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "report": str(args.report), "checks": len(checks)}))


if __name__ == "__main__":
    main()
