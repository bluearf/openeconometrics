"""Complete portable NLSUR example and uncensored independent science checks.

Portable execution imports no SciPy/Statsmodels. The separate development
science mode uses hand-coded NumPy/SciPy likelihood and physical derivatives.
It does not claim licensed Stata parity, global optimality or public release.
The separately declared 512-row follow-up uses 128 genuinely independent
clusters of four dependent rows and retains the original 320-row receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/examples/nonlinear_sur.py"
PRIMARY_SOURCES = ["https://www.stata.com/manuals/rnlsur.pdf"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(directory, *, show=True):
    return runpy.run_path(str(EXAMPLE), init_globals={
        "NONLINEAR_SUR_RESULT_DIRECTORY": str(directory), "NONLINEAR_SUR_DISPLAY": show})


def scientific(*, trials=100, rows=320):
    import numpy as np
    from scipy.stats import binom, binomtest, chi2
    import torch

    import openecon as oe
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.nonlinear_sur_reference import Reference, fixture

    if not 100 <= trials <= 200 or not 256 <= rows <= 512:
        raise ValueError("Use 100..200 uncensored trials per covariance and 256..512 rows.")
    if not Path(oe.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("Run source science with PYTHONPATH=src:packages/openecon-charts/src.")
    inputs = {"verifier": Path(__file__), "oracle": ROOT / "scripts/nonlinear_sur_reference.py",
              "example": EXAMPLE,
              "native_fit": ROOT / "src/openecon/econometrics/systems/nonlinear_sur.py",
              "native_postestimation": ROOT / "src/openecon/econometrics/systems/nonlinear_sur_postestimation.py",
              "native_formula": ROOT / "src/openecon/econometrics/quantile/formula.py"}
    hashes_before = {name: digest(path) for name, path in inputs.items()}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    configurations, failures = {}, []
    # These broad finite Monte Carlo diagnostics are declared before seeing
    # estimates. They complement exact every-cell numerical reference checks.
    parameter_count = 7
    minimum = int(binom.ppf(.01 / (3 * (parameter_count + 1)), trials, .95))
    gates = dict(maximum_standardized_matrix_cell_error=2e-6,
                 bias_standard_errors=5., covariance_allowance_multiplier=3.,
                 minimum_coverage_count=minimum, nominal_coverage=.95,
                 coverage_tail_probability=.01 / (3 * (parameter_count + 1)),
                 coverage_family="7 physical coordinates and one joint ellipsoid for each of 3 covariance modes",
                 rows=rows, cluster_size=4, independent_clusters=(rows+3)//4)
    print(json.dumps({"predeclared_gates": gates}, sort_keys=True), flush=True)
    initial_frame, _ = fixture(rows=rows)
    initial_reference = Reference(initial_frame)
    initial_theta, initial_best, initial_starts = initial_reference.fit()
    oracle = dict(physical_parameters=initial_theta.tolist(), log_likelihood=-float(initial_best.fun),
                  profile_mean_score=(-initial_reference.profile(initial_theta[:4])[1]).tolist(),
                  information=initial_reference.moments(initial_theta)["information"].tolist(),
                  multistart_mean_parameters=[fit.x.tolist() for fit in initial_starts],
                  multistart_negative_log_likelihood=[float(fit.fun) for fit in initial_starts],
                  fixture_sha256=hashlib.sha256(initial_frame.to_json(orient="split", double_precision=15).encode()).hexdigest())
    try:
        for mode_index, covariance_type in enumerate(("oim", "hc0", "cr0")):
            estimates, covariances, covered, joint_covered, numerical_errors = [], [], [], [], []
            attempts = []
            truth = None
            for replicate in range(trials):
                seed = 6_720_000 + 10_000 * mode_index + replicate
                frame, truth = fixture(seed, rows, dependence=covariance_type == "cr0",
                                       heteroskedastic=covariance_type == "hc0")
                ref = Reference(frame)
                try:
                    result = oe.nlsur(frame, equations=ref.definitions, start=ref.start,
                                      covariance=covariance_type,
                                      cluster="cluster" if covariance_type == "cr0" else None)
                    fit = result.attrs["nonlinear_sur_state"]["fit"]
                    estimate = np.asarray(fit["params"])
                    expected = ref.covariance(estimate, covariance_type)
                    errors = {}
                    for name in ("information", "bread", "meat", "covariance"):
                        scale = np.sqrt(np.diag(expected[name]))
                        errors[name] = float(np.max(np.abs(np.asarray(fit[name]) - expected[name])
                                                    / scale[:, None] / scale[None, :]))
                    covariance = np.asarray(fit["covariance"])
                    difference = estimate - truth
                    joint_statistic = float(difference @ np.linalg.solve(covariance, difference))
                    intervals = result["parameters"]
                    coverage = (intervals.ci_lower.to_numpy() <= truth) & (truth <= intervals.ci_upper.to_numpy())
                    estimates.append(estimate)
                    covariances.append(covariance)
                    covered.append(coverage)
                    joint_covered.append(joint_statistic <= chi2.ppf(.95, parameter_count))
                    numerical_errors.append(errors)
                    attempts.append(dict(replicate=replicate, seed=seed, status="accepted",
                                         physical_parameters=estimate.tolist(), covariance=covariance.tolist(),
                                         covered=coverage.tolist(), joint_statistic=joint_statistic,
                                         numerical_errors=errors))
                except Exception as error:
                    failure = dict(covariance=covariance_type, replicate=replicate, seed=seed,
                                   exception=type(error).__name__, code=getattr(error, "code", None), message=str(error))
                    failures.append(failure)
                    attempts.append(dict(**failure, status="failed", covered=[False] * parameter_count,
                                         joint_covered=False))
                if (replicate + 1) % 20 == 0:
                    print(json.dumps(dict(covariance=covariance_type, attempted=replicate + 1,
                                          accepted=len(estimates)), sort_keys=True), flush=True)
            accepted = len(estimates)
            if accepted < 2:
                configurations[covariance_type] = dict(attempted=trials, accepted=accepted, attempts=attempts)
                continue
            estimates, covariances, covered = map(np.asarray, (estimates, covariances, covered))
            empirical = np.cov(estimates, rowvar=False, ddof=1)
            mean_covariance = covariances.mean(axis=0)
            sd = np.sqrt(np.diag(mean_covariance))
            normalized = mean_covariance / sd[:, None] / sd[None, :]
            covariance_error = float(np.linalg.norm((empirical - mean_covariance) / sd[:, None] / sd[None, :], ord="fro"))
            allowance = float(3 * np.sqrt((np.trace(normalized)**2 + np.sum(normalized**2)) / (accepted - 1)))
            bias = estimates.mean(axis=0) - truth
            bias_allowance = 5 * np.sqrt(np.diag(empirical) / accepted)
            names = ["b0", "b1", "amp", "rate", "cov__eq1__eq1", "cov__eq2__eq1", "cov__eq2__eq2"]
            parameters = {}
            for j, name in enumerate(names):
                count = int(covered[:, j].sum())
                interval = binomtest(count, trials).proportion_ci(confidence_level=.99)
                parameters[name] = dict(true=float(truth[j]), mean=float(estimates[:, j].mean()), bias=float(bias[j]),
                                        covered=count, conservative_coverage=count / trials,
                                        coverage99_interval=[float(interval.low), float(interval.high)],
                                        empirical_variance=float(empirical[j, j]),
                                        mean_estimated_variance=float(mean_covariance[j, j]))
            joint_count = int(np.count_nonzero(joint_covered))
            joint_interval = binomtest(joint_count, trials).proportion_ci(confidence_level=.99)
            maximum_errors = {name: max(value[name] for value in numerical_errors) for name in numerical_errors[0]}
            generation = ("Independent Gaussian rows with common marginal Sigma" if covariance_type == "oim" else
                          "iid random uniform design and conditional Gaussian errors; heteroskedastic scale .4..1.6 has theoretical E(scale)=1, with no realized normalization" if covariance_type == "hc0" else
                          "Independent clusters; 4 rows share sqrt(.4) Gaussian innovation plus sqrt(.6) independent innovation; total marginal Sigma unchanged")
            configurations[covariance_type] = dict(attempted=trials, accepted=accepted, attempts=attempts,
                parameters=parameters, empirical_covariance=empirical.tolist(), mean_estimated_covariance=mean_covariance.tolist(),
                normalized_full_covariance_error=covariance_error, covariance_error_allowance=allowance,
                covariance_gate_passed=covariance_error <= allowance,
                bias_gate_passed=bool((abs(bias) <= bias_allowance).all()),
                bias_allowance=bias_allowance.tolist(), coverage_minimum_count=minimum,
                coverage_gate_passed=bool((covered.sum(axis=0) >= minimum).all()),
                joint_coverage=dict(covered=joint_count, conservative_coverage=joint_count/trials,
                                    coverage99_interval=[float(joint_interval.low), float(joint_interval.high)],
                                    gate_passed=joint_count >= minimum),
                maximum_standardized_cell_error=maximum_errors,
                numerical_gate_passed=all(value < gates["maximum_standardized_matrix_cell_error"] for value in maximum_errors.values()),
                generation=generation)
    finally:
        torch.set_num_threads(previous_threads)
    hashes_after = {name: digest(path) for name, path in inputs.items()}
    unchanged = hashes_before == hashes_after
    passed = unchanged and not failures and all(value.get("coverage_gate_passed") and value.get("bias_gate_passed")
              and value.get("covariance_gate_passed") and value.get("numerical_gate_passed")
              and value.get("joint_coverage", {}).get("gate_passed") for value in configurations.values())
    return dict(status="passed" if passed else "failed",
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_hashes_before=hashes_before, input_hashes_after=hashes_after,
        source_inputs_unchanged_during_run=unchanged, primary_sources=PRIMARY_SOURCES,
        predeclared_gates=gates, trials_per_covariance=trials, rows_per_replicate=rows,
        uncensored_attempted_replicates=3*trials, failures=failures, configurations=configurations,
        fixture_multistart_oracle=oracle, duration_seconds=time.monotonic()-started,
        coverage_failure_policy="Every declared seed attempted once; failures count uncovered; no censoring, replacements or restarts of failed replicates",
        confidence_policy="Physical normal mean/off-diagonal covariance intervals; log-delta normal variance intervals; full 7-dimensional physical Wald ellipsoid",
        covariance_allowance_definition="3*sqrt((trace(R)^2+sum(R^2))/(accepted-1)); R is diagonal-standardized mean covariance; a broad Monte Carlo diagnostic, not a p-value",
        numerical_oracle="Hand-coded NumPy Gaussian score and observed physical Hessian, including off-diagonal Sigma multiplicity, nonlinear curvature and beta-Sigma crossblocks; every matrix cell",
        target_population="Bounded Gaussian-error systems: iid constant covariance for OIM, iid random-design conditionally Gaussian heteroskedastic rows for HC0, independent Gaussian clusters for CR0; HC0 target is unconditional Gaussian quasi-likelihood pseudo-true beta/Sigma; no universal finite-sample guarantee",
        third_party_estimation_runtime=False, proprietary_software_run=False, global_maximum_claimed=False,
        native_window_verified=False, public_release_delivered=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--directory", type=Path)
    mode.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.directory is not None:
        portable(args.directory, show=not args.no_display)
    else:
        if args.output.exists():
            raise FileExistsError("Retain historical scientific receipts; choose a fresh output filename.")
        result = scientific(trials=args.trials, rows=args.rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "uncensored_attempted_replicates", "duration_seconds")}))
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
