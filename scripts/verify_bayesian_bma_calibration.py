"""Preregistered 256-case exact finite-mixture PIT/inclusion calibration, without replacement.

Source numerical checks and calibration are separate. All cases and failures
stay in the registered denominator; changed source/plan aborts acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from openecon.econometrics.bayesian import bma_kernels as kernels
from openecon.econometrics.bayesian.bma import bayes_bma
from openecon.econometrics.bayesian.bma_query import bayes_bma_predict
from openecon.econometrics.bayesian.posterior import NormalInverseGammaPrior
from verify_bayesian_bma_oracles import (
    audit,
    coefficient_cdf,
    query_cdf,
    reference,
    variance_cdf,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/econometrics/bayesian-bma-calibration-plan.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registration_matches(registration):
    failures = {
        name: {"expected": digest, "actual": sha(ROOT / name)}
        for name, digest in registration["file_sha256"].items()
        if sha(ROOT / name) != digest
    }
    if failures:
        raise RuntimeError("Preregistered source/plan changed: " + json.dumps(failures))


def fixed_setup(plan):
    n = plan["n"]
    rows = np.arange(n, dtype=float)
    data = pd.DataFrame(
        {"x1": np.linspace(-1, 1, n), "x2": np.sin(0.61 * rows), "x3": np.cos(0.43 * rows)},
        index=pd.RangeIndex(n, name="physical_row"),
    )
    full = np.c_[np.ones(n), data.to_numpy()]
    query = pd.DataFrame(
        [plan["query"]], columns=plan["optional"], index=pd.Index(["preregistered"], name="query")
    )
    Z = np.r_[1.0, query.to_numpy()[0]]
    m = np.array(plan["per_model_prior"]["mean_full"])
    V = np.array(plan["per_model_prior"]["scale_diagonal_full"])
    q = plan["model_prior_inclusion"]
    priors = []
    odds = []
    indices = []
    for mask in plan["model_masks"]:
        ix = [0, *[j + 1 for j in range(3) if mask & (1 << j)]]
        indices.append(ix)
        priors.append(
            NormalInverseGammaPrior(
                mean=m[ix].tolist(),
                scale_matrix=np.diag(V[ix]).tolist(),
                shape=3 + 0.25 * mask.bit_count(),
                scale=2 + 0.1 * mask.bit_count(),
            )
        )
        odds.append(
            float(np.prod([value if mask & (1 << j) else 1 - value for j, value in enumerate(q)]))
        )
    return data, full, query, Z, priors, np.array(odds), indices


def dkw(values):
    values = np.sort(np.array(values, dtype=float))
    n = len(values)
    return float(max(np.max(np.arange(1, n + 1) / n - values), np.max(values - np.arange(n) / n)))


def case(rep, plan, setup, folder):
    data, full, query, Z, priors, odds, indices = setup
    rng = np.random.Generator(np.random.PCG64(6500001 + 1009 * rep))
    jitter = np.random.Generator(np.random.PCG64(6900001 + 1009 * rep))
    mask = int(rng.choice(8, p=odds / odds.sum()))
    prior = priors[mask]
    sigma = prior.scale / float(rng.gamma(prior.shape))
    beta = np.zeros(4)
    beta[indices[mask]] = rng.multivariate_normal(
        np.array(prior.mean), sigma * np.array(prior.scale_matrix)
    )
    mean = full @ beta
    frame = data.copy()
    frame["y"] = mean + math.sqrt(sigma) * rng.standard_normal(len(frame))
    true_mean = float(Z @ beta)
    new_outcome = true_mean + math.sqrt(sigma) * float(rng.standard_normal())
    fitted = bayes_bma(
        data=frame, y="y", optional=plan["optional"], priors=priors, model_prior_odds=odds.tolist()
    )
    predicted = bayes_bma_predict(fitted, data=query)
    fitpath = folder / f"{rep:03d}-posterior.json"
    querypath = folder / f"{rep:03d}-prediction.json"
    fitpath.write_text(fitted.model_dump_json())
    querypath.write_text(predicted.model_dump_json())
    posterior = json.loads(fitpath.read_text())["payload"]
    prediction = json.loads(querypath.read_text())["payload"]
    exact = reference(posterior)
    raw_models = posterior["components"]
    w = posterior["results"]["model_weights"]
    qr = prediction["results"]
    checks = []
    ranks = []
    for j in range(1, 4):
        left = coefficient_cdf(exact, j, float(beta[j]), left=True)
        right = coefficient_cdf(exact, j, float(beta[j]))
        laws = kernels.marginal_laws(raw_models, j)
        checks.extend(
            [
                abs(kernels.cdf(float(beta[j]), laws, w, left=True) - left),
                abs(kernels.cdf(float(beta[j]), laws, w) - right),
            ]
        )
        ranks.append(left + float(jitter.uniform()) * (right - left))
    sigma_cdf = variance_cdf(exact, sigma)
    checks.append(
        abs(
            kernels.ig_cdf(
                sigma, [(v["posterior"]["shape"], v["posterior"]["scale"]) for v in raw_models], w
            )
            - sigma_cdf
        )
    )
    ranks.append(sigma_cdf)
    for target, truth in [("mean", true_mean), ("outcome", new_outcome)]:
        value = query_cdf(exact, Z[None, :], 0, truth, outcome=target == "outcome")
        laws = [
            (
                qr["model_mean"][i][0],
                qr["model_" + target + "_scale"][i][0],
                v["posterior"]["degrees_of_freedom"],
            )
            for i, v in enumerate(raw_models)
        ]
        checks.append(abs(kernels.cdf(truth, laws, w) - value))
        ranks.append(value)
    if max(checks) > plan["cdf_absolute_tolerance"]:
        raise AssertionError("Production true-value CDF differs from independent law")
    audit_post = audit(posterior, atol=plan["mixture_full_moment_tolerance"])
    audit_query = audit(prediction, atol=plan["mixture_full_moment_tolerance"])
    pip = np.array(posterior["results"]["posterior_inclusion"])
    indicator = np.array([bool(mask & (1 << j)) for j in range(3)], dtype=float)
    return {
        "replication": rep,
        "success": True,
        "mask": mask,
        "embedded_beta": beta.tolist(),
        "sigma_squared": sigma,
        "query_mean": true_mean,
        "query_new_outcome": new_outcome,
        "ranks": ranks,
        "pip": pip.tolist(),
        "inclusion_residual": (indicator - pip).tolist(),
        "negative_variance_rank": variance_cdf(exact, 4 * sigma),
        "negative_complement_inclusion_residual": float(indicator[0] - (1 - pip[0])),
        "cdf_max_error": max(checks),
        "posterior_audit": audit_post,
        "query_audit": audit_query,
        "state_sha256": {"posterior": sha(fitpath), "prediction": sha(querypath)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    registration_matches(registration)
    plan = json.loads(PLAN.read_text())
    if plan["replications"] != 256 or registration["plan_sha256"] != sha(PLAN):
        raise RuntimeError("Wrong prospective plan")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Use a fresh directory; registered cases cannot be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.output / "cases"
    folder.mkdir()
    setup = fixed_setup(plan)
    cases = []
    start = time.monotonic()
    for rep in range(256):
        try:
            result = case(rep, plan, setup, folder)
        except Exception as exc:
            result = {
                "replication": rep,
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        cases.append(result)
        (folder / f"{rep:03d}-case.json").write_text(json.dumps(result, indent=2) + "\n")
        if (rep + 1) % 16 == 0:
            registration_matches(registration)
            print(
                json.dumps(
                    {
                        "completed": rep + 1,
                        "denominator": 256,
                        "failures": sum(not v["success"] for v in cases),
                        "elapsed_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    registration_matches(registration)
    successful = [v for v in cases if v["success"]]
    failure_count = 256 - len(successful)
    stats = {}
    accept = False
    if failure_count == 0:
        ranks = np.array([v["ranks"] for v in cases])
        distances = [dkw(ranks[:, j]) for j in range(6)]
        residual = np.mean([v["inclusion_residual"] for v in cases], axis=0)
        wrong_variance = dkw([v["negative_variance_rank"] for v in cases])
        wrong_inclusion = abs(
            float(np.mean([v["negative_complement_inclusion_residual"] for v in cases]))
        )
        epsilon = plan["cdf_gate"]["epsilon"]
        bound = plan["inclusion_gate"]["maximum_absolute_mean_residual"]
        stats = {
            "cdf_targets": plan["targets"],
            "cdf_D": distances,
            "cdf_epsilon": epsilon,
            "cdf_family_passed": max(distances) <= epsilon,
            "inclusion_mean_residual": residual.tolist(),
            "inclusion_bound": bound,
            "inclusion_family_passed": float(np.max(np.abs(residual))) <= bound,
            "wrong_variance_D": wrong_variance,
            "wrong_variance_rejected": wrong_variance > epsilon,
            "complement_PIP_residual_abs": wrong_inclusion,
            "complement_PIP_rejected": wrong_inclusion > bound,
            "maximum_true_value_cdf_error": max(v["cdf_max_error"] for v in cases),
        }
        accept = (
            stats["cdf_family_passed"]
            and stats["inclusion_family_passed"]
            and stats["wrong_variance_rejected"]
            and stats["complement_PIP_rejected"]
        )
    receipt = {
        "schema_version": "openecon.bayesian_bma_calibration_receipt.v1",
        "registration_sha256": sha(args.registration),
        "plan_sha256": sha(PLAN),
        "source_pin": registration,
        "denominator": 256,
        "completed": len(cases),
        "failures": failure_count,
        "failure_cases": [v for v in cases if not v["success"]],
        "elapsed_seconds": time.monotonic() - start,
        "statistics": stats,
        "accepted": accept,
        "numerical_reference": "independent NumPy observation-space proper NIG + SciPy Student-t/IG mixture; no production posterior used as reference",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({"accepted": accept, "failures": failure_count, "statistics": stats}), flush=True
    )
    if not accept:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
