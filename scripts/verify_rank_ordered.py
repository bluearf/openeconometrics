"""Development-only independent Gumbel recovery/coverage receipt.

Runs every fixed-seed replicate, reports every failure, and uses no proprietary
software. NumPy/SciPy and the independent test oracle are development tools,
not dependencies of the public/native ranking estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.stats import binom, binomtest

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests/test_rank_ordered_oracle.py"
SOURCE_URLS = ["https://www.stata.com/manuals15/rrologit.pdf",
               "https://hturner.github.io/PlackettLuce/articles/Overview.html",
               "https://eml.berkeley.edu/books/choice2nd/Ch03_p34-75.pdf"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(*, trials=100, cases=240):
    if not 100 <= trials <= 200 or not 120 <= cases <= 512:
        raise ValueError("Use100–200 replicates per VCE and120–512 cases.")
    files = {"oracle": ORACLE, "verifier": Path(__file__),
             "native_fit": ROOT / "src/openecon/econometrics/discrete/rank_ordered.py",
             "native_postestimation": ROOT / "src/openecon/econometrics/discrete/rank_ordered_postestimation.py"}
    hashes_before = {name: digest(path) for name, path in files.items()}
    spec = importlib.util.spec_from_file_location("independent_rank_oracle", ORACLE)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    if not Path(oracle.oe.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("Run this source verifier with PYTHONPATH=src:packages/openecon-charts/src.")
    started = time.monotonic()
    configurations, failures = {}, []
    for vce_index, vce in enumerate(("oim", "hc0", "cr0")):
        estimates, covariances, covered = [], [], []
        for replicate in range(trials):
            seed = 5_450_000 + 10_000*vce_index + replicate
            frame = oracle.ranking_fixture(seed=seed, cases=cases, prefix=True, availability=True,
                                           repeated=vce == "cr0", common_shocks=vce == "cr0")
            try:
                result = oracle.fit_public(frame, vce)
                parameters = result["parameters"].set_index("parameter").loc[list(oracle.PARAMETERS)]
                estimate = parameters["estimate"].to_numpy(float)
                covariance = oracle.full_matrix(result, "covariance")
                estimates.append(estimate)
                covariances.append(covariance)
                covered.append((parameters.ci_lower.to_numpy() <= oracle.TRUE)
                               & (oracle.TRUE <= parameters.ci_upper.to_numpy()))
            except Exception as exc:
                failures.append({"vce": vce, "replicate": replicate, "seed": seed,
                                 "exception": type(exc).__name__, "message": str(exc)})
        if len(estimates) < 2:
            configurations[vce] = {"attempted": trials, "succeeded": len(estimates)}
            continue
        estimates, covariances, covered = map(np.asarray, (estimates, covariances, covered))
        empirical = np.cov(estimates, rowvar=False, ddof=1)
        mean_covariance = covariances.mean(axis=0)
        scale = np.sqrt(mean_covariance.diagonal())
        normalized_empirical = empirical/scale[:, None]/scale[None, :]
        normalized_mean = mean_covariance/scale[:, None]/scale[None, :]
        normalized_covariance_error = np.linalg.norm(normalized_empirical-normalized_mean, ord="fro")
        # Three Gaussian-Wishart RMS reference errors is a predeclared broad
        # Monte-Carlo allowance, not a covariance hypothesis-test p-value.
        covariance_allowance = 3*np.sqrt((np.trace(normalized_mean)**2+np.sum(normalized_mean**2))/(len(estimates)-1))
        bias = estimates.mean(axis=0)-oracle.TRUE
        coverage = covered.mean(axis=0)
        # Predeclared simultaneous finite-Monte-Carlo lower threshold: family
        # error <=.01 across15 coefficient/VCE comparisons under95% coverage.
        threshold = int(binom.ppf(.01/15, trials, .95))
        rows = {}
        for j, name in enumerate(oracle.PARAMETERS):
            interval = binomtest(int(covered[:, j].sum()), len(covered)).proportion_ci(confidence_level=.99)
            rows[name] = {"true": float(oracle.TRUE[j]), "mean": float(estimates[:, j].mean()),
                          "bias": float(bias[j]), "coverage": float(coverage[j]),
                          "coverage99_interval": [float(interval.low), float(interval.high)],
                          "covered_replicates": int(covered[:, j].sum()),
                          "empirical_variance": float(empirical[j, j]),
                          "mean_estimated_variance": float(mean_covariance[j, j]),
                          "variance_ratio": float(empirical[j, j]/mean_covariance[j, j])}
        configurations[vce] = {"attempted": trials, "succeeded": len(estimates), "parameters": rows,
                               "empirical_covariance": empirical.tolist(), "mean_estimated_covariance": mean_covariance.tolist(),
                               "coverage_minimum_count": threshold,
                               "coverage_gate_passed": bool((covered.sum(axis=0) >= threshold).all()),
                               "bias_gate_passed": bool((np.abs(bias) <= 5*np.sqrt(empirical.diagonal()/len(estimates))).all()),
                               "normalized_full_covariance_frobenius_error": float(normalized_covariance_error),
                               "normalized_full_covariance_error_allowance": float(covariance_allowance),
                               "full_covariance_gate_passed": bool(normalized_covariance_error <= covariance_allowance),
                               "covariance_allowance_definition": "3*sqrt((trace(R)^2+sum(R^2))/(replicates-1)); R=mean covariance scaled by its diagonal",
                               "generation": "iid unit-scale Gumbel across alternatives; same Gumbel vector shared by3 cases/respondent for CR0"}
    hashes_after = {name: digest(path) for name, path in files.items()}
    unchanged = hashes_before == hashes_after
    passed = unchanged and not failures and all(c.get("coverage_gate_passed") and c.get("bias_gate_passed")
                    and c.get("full_covariance_gate_passed") for c in configurations.values())
    return {"status": "passed" if passed else "failed", "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input_hashes_before": hashes_before, "input_hashes_after": hashes_after,
            "source_inputs_unchanged_during_run": unchanged,
            "primary_sources": SOURCE_URLS, "trials_per_vce": trials, "cases_per_replicate": cases,
            "uncensored_attempted_replicates": 3*trials, "failures": failures,
            "configurations": configurations, "duration_seconds": time.monotonic()-started,
            "strict_top_prefix_including_unranked_available_tail": True,
            "independent_random_utility_generation": True, "third_party_estimation_runtime": False,
            "proprietary_software_run": False, "native_window_verified": False,
            "public_release_delivered": False}


def verify(*, trials=100, cases=240):
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _verify(trials=trials, cases=cases)
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(trials=args.trials, cases=args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "uncensored_attempted_replicates", "duration_seconds")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
