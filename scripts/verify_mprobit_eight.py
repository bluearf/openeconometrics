"""Portable MNP example replay and separately retained scientific calibration.

Portable mode uses the native example only; NumPy/SciPy reference dependencies
are imported exclusively in development science mode. The calibration design
and gates are declared before any fit, with every attempted seed retained.
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
EXAMPLE = ROOT / "docs/examples/mprobit.py"
PRIMARY_SOURCES = ["https://eml.berkeley.edu/books/choice2nd/Ch05_p97-133.pdf"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(directory, *, show=True):
    return runpy.run_path(str(EXAMPLE), init_globals={
        "MPROBIT_RESULT_DIRECTORY": str(directory), "MPROBIT_DISPLAY": show})


def scientific(*, trials=100, cases=512):
    import numpy as np
    from scipy.stats import binom, binomtest, chi2
    import torch

    import openecon as oe
    from openecon.econometrics.discrete.mprobit import mprobit
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.mprobit_reference import Reference, fixture

    if trials != 100 or cases != 512:
        raise ValueError("The retained predeclared design is 100 trials per VCE, 512 cases.")
    if not Path(oe.__file__).resolve().is_relative_to(ROOT / "src"):
        raise ValueError("Run source science with PYTHONPATH=src:packages/openecon-charts/src.")
    inputs = {"verifier": Path(__file__), "oracle": ROOT / "scripts/mprobit_reference.py",
              "example": EXAMPLE,
              "native_fit": ROOT / "src/openecon/econometrics/discrete/mprobit.py",
              "native_postestimation": ROOT / "src/openecon/econometrics/discrete/mprobit_postestimation.py",
              "native_gaussian_values": ROOT / "src/openecon/econometrics/discrete/bivariate.py",
              "native_validation_helpers": ROOT / "src/openecon/econometrics/discrete/nested_logit.py",
              "native_delta_helpers": ROOT / "src/openecon/econometrics/discrete/rank_ordered_postestimation.py"}
    hashes_before = {name: digest(path) for name, path in inputs.items()}
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    in_commit = {name: (subprocess.check_output(["git", "show", f"{source_revision}:{path.relative_to(ROOT)}"],
                                               cwd=ROOT) == path.read_bytes()) for name, path in inputs.items()}
    parameter_count = 4
    probability = .01/(3*(parameter_count+1))
    minimum = int(binom.ppf(probability, trials, .95))
    gates = dict(maximum_standardized_matrix_cell_error=5e-6,
                 maximum_failed_attempts=0, bias_standard_errors=5.,
                 covariance_allowance_multiplier=3., minimum_coverage_count=minimum,
                 nominal_coverage=.95, coverage_tail_probability=probability,
                 coverage_family="4 physical coordinates and one joint ellipsoid, each of 3 VCE modes",
                 cases=cases, independent_clusters=128, cases_per_cluster=4,
                 Gaussian_common_shock_variance_fraction=.35,
                 free_parameters=["x", "z", "sd3", "rho3"],
                 truth=[.55, -.4, 1.25, .2],
                 available_geometry="70% triples; 30% balanced binary subsets (mod10 residues7/8/9); finite512case counts follow that fixed period",
                 first_seed=6_990_000, mode_seed_spacing=10_000,
                 replacements=False, failures_count_as_uncovered=True,
                 covariance_diagnostics="full physical empirical covariance versus average estimated covariance")
    print(json.dumps({"predeclared_gates": gates}, sort_keys=True), flush=True)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    configurations, failures = {}, []
    try:
        for mode_index, vce in enumerate(("oim", "hc0", "cr0")):
            estimates, covariances, covered, joint_covered = [], [], [], []
            attempts, errors = [], []
            for replicate in range(trials):
                seed = 6_990_000 + 10_000*mode_index + replicate
                frame, truth = fixture(seed, cases, dependence=vce == "cr0")
                reference = Reference(frame)
                try:
                    result = mprobit(frame, "chosen", ["x", "z"], case="case", alternative="alternative",
                                     alternatives=["A", "B", "C"], available="available", vce=vce,
                                     cluster="cluster" if vce == "cr0" else None)
                    fit = result.attrs["mprobit_state"]["fit"]
                    estimate = np.asarray(fit["params"], dtype=float)
                    expected = reference.covariance(estimate, vce)
                    cell_errors = {}
                    for name in ("information", "bread", "meat", "covariance"):
                        scale = np.sqrt(np.diag(expected[name]))
                        cell_errors[name] = float(np.max(np.abs(np.asarray(fit[name])-expected[name])
                                                        /scale[:, None]/scale[None, :]))
                    covariance = np.asarray(fit["covariance"], dtype=float)
                    difference = estimate-truth
                    joint_statistic = float(difference@np.linalg.solve(covariance, difference))
                    parameters = result["parameters"]
                    coverage = ((parameters.ci_lower.to_numpy() <= truth)
                                & (truth <= parameters.ci_upper.to_numpy()))
                    estimates.append(estimate)
                    covariances.append(covariance)
                    covered.append(coverage)
                    joint_covered.append(bool(joint_statistic <= chi2.ppf(.95, parameter_count)))
                    errors.append(cell_errors)
                    attempts.append(dict(replicate=replicate, seed=seed, status="accepted",
                                         params=estimate.tolist(), covariance=covariance.tolist(),
                                         marginal_covered=coverage.tolist(), joint_statistic=joint_statistic,
                                         joint_covered=joint_covered[-1], matrix_cell_errors=cell_errors,
                                         log_likelihood=float(fit["log_likelihood"])))
                except Exception as exc:
                    failure = dict(vce=vce, replicate=replicate, seed=seed, status="failed",
                                   error_type=type(exc).__name__, message=str(exc))
                    failures.append(failure)
                    attempts.append(failure)
                if (replicate+1) % 10 == 0:
                    print(json.dumps(dict(vce=vce, attempted=replicate+1,
                                          accepted=len(estimates), failures=len(attempts)-len(estimates))), flush=True)
            if not estimates:
                configurations[vce] = dict(attempted=trials, accepted=0, failed=trials,
                                           gates_passed=False, attempts=attempts)
                continue
            estimates, covariances = np.asarray(estimates), np.asarray(covariances)
            empirical = np.cov(estimates, rowvar=False, ddof=1)
            average = np.mean(covariances, axis=0)
            bias = np.mean(estimates, axis=0)-truth
            bias_units = np.abs(bias)/np.sqrt(np.diag(empirical)/len(estimates))
            allowance = 3*np.sqrt((np.outer(np.diag(empirical), np.diag(empirical))+empirical**2)
                                  /max(1, len(estimates)-1))
            covariance_gate = bool(np.all(np.abs(empirical-average) <= allowance))
            marginal_counts = np.sum(covered, axis=0).astype(int)
            joint_count = int(sum(joint_covered))
            all_counts = marginal_counts.tolist()+[joint_count]
            uncertainty = [dict(covered=count, attempted=trials,
                                exact_binomial_99_low=float(binomtest(count, trials).proportion_ci(.99).low),
                                exact_binomial_99_high=float(binomtest(count, trials).proportion_ci(.99).high))
                           for count in all_counts]
            worst_error = max(max(value.values()) for value in errors)
            passed = (len(estimates) == trials and max(bias_units) <= gates["bias_standard_errors"]
                      and covariance_gate and min(all_counts) >= minimum
                      and worst_error <= gates["maximum_standardized_matrix_cell_error"])
            configurations[vce] = dict(attempted=trials, accepted=len(estimates), failed=trials-len(estimates),
                                       truth=truth.tolist(), mean_estimate=estimates.mean(0).tolist(),
                                       bias=bias.tolist(), bias_standard_error_units=bias_units.tolist(),
                                       empirical_full_covariance=empirical.tolist(),
                                       mean_estimated_full_covariance=average.tolist(),
                                       covariance_allowance=allowance.tolist(), covariance_gate_passed=covariance_gate,
                                       marginal_coverage_counts=marginal_counts.tolist(), joint_coverage_count=joint_count,
                                       binomial_uncertainty=uncertainty,
                                       maximum_standardized_matrix_cell_error=worst_error,
                                       gates_passed=bool(passed), attempts=attempts)
    finally:
        torch.set_num_threads(previous_threads)
    hashes_after = {name: digest(path) for name, path in inputs.items()}
    stable = hashes_before == hashes_after
    return dict(contract="independent_mprobit_calibration_v1", source_revision=source_revision,
                source_bytes_in_commit=in_commit, source_hashes_before=hashes_before,
                source_hashes_after=hashes_after, source_files_stable=stable,
                primary_sources=PRIMARY_SOURCES, predeclared_gates=gates,
                elapsed_seconds=time.monotonic()-started, attempted=3*trials,
                accepted=sum(value["accepted"] for value in configurations.values()), failures=failures,
                configurations=configurations,
                passed=bool(stable and all(in_commit.values()) and not failures
                            and all(value["gates_passed"] for value in configurations.values())),
                interpretation="Bounded finite Monte Carlo diagnostics, not proof of nominal coverage, global ML optimality, vendor parity or public release.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--directory", type=Path, default=ROOT/"docs/evidence/mprobit-eight-2026-10-09/portable")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cases", type=int, default=512)
    arguments = parser.parse_args()
    if arguments.science:
        output = scientific(trials=arguments.trials, cases=arguments.cases)
        if arguments.output is None:
            raise ValueError("Science requires a new retained receipt filename via --output.")
        if arguments.output.exists():
            raise FileExistsError("A retained scientific receipt is immutable; choose a fresh filename.")
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"receipt": str(arguments.output), "sha256": digest(arguments.output),
                          "passed": output["passed"], "attempted": output["attempted"],
                          "accepted": output["accepted"]}, sort_keys=True), flush=True)
        return 0 if output["passed"] else 1
    portable(arguments.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
