"""Development-only sampling check of all eight binary-mediation model domains.

The covariate grid is fixed across samples. Independent NumPy/SciPy truth
calculation evaluates the exact two-state response means. Native public fits
measure complete five-effect covariance and pointwise normal coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import expit, ndtr
import torch

import openecon as oe


def response(eta, model):
    return eta if model == "gaussian" else expit(eta) if model == "logit" else ndtr(eta) if model == "probit" else np.exp(eta)


def truth(c, ml, model):
    means = []
    for a, aprime in ((0, 0), (1, 0), (0, 1), (1, 1)):
        pm = response(-.35 + .6 * aprime + .3 * c, ml)
        y0 = response(.1 + .35 * a + .15 * c, model)
        y1 = response(.1 + .35 * a + .45 + .2 * a + .15 * c, model)
        means.append(np.mean((1 - pm) * y0 + pm * y1))
    u00, u10, u01, u11 = means
    return np.array([u10 - u00, u11 - u01, u01 - u00, u11 - u10, u11 - u00])


def summarize(estimates, covariances, coverage, target):
    estimates, covariances = np.asarray(estimates), np.asarray(covariances)
    reps = len(estimates)
    empirical = np.cov(estimates, rowvar=False, ddof=1)
    mean_covariance = covariances.mean(axis=0)
    scale = np.sqrt(np.outer(empirical.diagonal(), empirical.diagonal()))
    normalized = np.abs(mean_covariance - empirical) / scale
    # Sampling checks are diagnostic bounded asymptotic checks. Gates include
    # Monte Carlo error and do not certify exact finite-sample coverage.
    variance_gate = .10 + 4 * np.sqrt(2 / (reps - 1))
    coverage_gate = .025 + 4 * np.sqrt(.95 * .05 / reps)
    bias_z = np.abs(estimates.mean(axis=0) - target) / np.sqrt(empirical.diagonal() / reps)
    coverage_rate = np.asarray(coverage).mean(axis=0)
    assert normalized.max() < variance_gate, normalized
    assert np.all(np.abs(coverage_rate - .95) < coverage_gate), coverage_rate
    assert bias_z.max() < 5, bias_z
    return dict(status="passed", target=target.tolist(), empirical_covariance=empirical.tolist(),
                mean_covariance=mean_covariance.tolist(), coverage=coverage_rate.tolist(),
                maximum_absolute_bias_MC_z=float(bias_z.max()),
                maximum_normalized_full_covariance_error=float(normalized.max()),
                covariance_normalized_error_gate=float(variance_gate),
                coverage_deviation_gate=float(coverage_gate))


def validate(reps, n):
    root = Path(__file__).resolve().parents[1]
    paths = [root / "src/openecon/econometrics/decomposition" / name for name in ("binary.py", "binary_kernels.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    c = np.linspace(-1.5, 1.5, n)
    rng = np.random.default_rng(167441)
    results = {}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for ml in ("logit", "probit"):
            for model in ("gaussian", "logit", "probit", "poisson"):
                estimates, covariances, coverage = [], [], []
                target = truth(c, ml, model)
                for _ in range(reps):
                    a = rng.binomial(1, .5, n)
                    m = rng.binomial(1, response(-.35 + .6 * a + .3 * c, ml))
                    mu = response(.1 + .35 * a + .45 * m + .2 * a * m + .15 * c, model)
                    y = rng.normal(mu, .8) if model == "gaussian" else rng.poisson(mu) if model == "poisson" else rng.binomial(1, mu)
                    result = oe.mediation_binary(data={"y":y, "a":a, "m":m, "c":c}, y="y", treatment="a", mediator="m",
                                                 controls=["c"], mediator_link=ml, outcome_model=model,
                                                 interaction=True, covariance="HC0")
                    effect = result["effects"]
                    estimates.append(effect["estimate"].to_numpy())
                    names = ["PNDE", "TNDE", "PNIE", "TNIE", "TE"]
                    matrix = result["effects_covariance"]
                    assert effect["effect"].tolist() == matrix["parameter"].tolist() == names
                    numeric_covariance = matrix[names].to_numpy(dtype=float)
                    assert numeric_covariance.shape == (5, 5)
                    covariances.append(numeric_covariance)
                    coverage.append(((effect["ci_lower"].to_numpy() <= target) & (target <= effect["ci_upper"].to_numpy())).tolist())
                results[ml + "_" + model] = summarize(estimates, covariances, coverage, target)
                print(json.dumps({"model_domain":ml + "_" + model, "replications":reps, "status":"passed"}), flush=True)
    finally:
        torch.set_num_threads(previous)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return dict(status="passed", seed=167441, replications_per_domain=reps, full_public_API_calls=8 * reps,
                observations_per_sample=n, fixed_control_grid=c.tolist(), fixed_empirical_target=True,
                source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                source_modules_sha256=hashes, empirical_uncertainty=results,
                licensed_vendor_run=False, whole_product_parity=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=400)
    parser.add_argument("--observations", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 200 or not 200 <= args.observations <= 4096:
        parser.error("Require at least 200 replications per domain and 200..4096 observations.")
    result = validate(args.replications, args.observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:result[key] for key in ("status", "full_public_API_calls", "replications_per_domain")}))
