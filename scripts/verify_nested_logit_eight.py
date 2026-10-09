"""Nested-logit portable acceptance and independent Monte Carlo verification.

--directory executes the complete public example without SciPy imports.
--output runs uncensored development-only likelihood/covariance calibration.
Neither mode claims licensed vendor parity, a global maximum or public release.
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
EXAMPLE = ROOT/"docs/examples/nested_logit.py"
PRIMARY_SOURCES = ["https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(directory, *, show=True):
    return runpy.run_path(str(EXAMPLE), init_globals={"NESTED_LOGIT_RESULT_DIRECTORY": str(directory),
                                                   "NESTED_LOGIT_DISPLAY": show})


def monte_carlo_fixture(seed, cases=512, shared_uniform=False):
    """Sample nested probabilities with independent or respondent-shared uniforms.

A shared uniform supplies a fixed copula with the same correctly specified
choice probability in each marginal case and genuine within-person dependence.
No production fit or private mathematical helper participates in generation.
"""
    import numpy as np
    import pandas as pd
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.nested_logit_reference import Reference

    generator = np.random.default_rng(seed)
    rows = []
    for case in range(cases):
        design = generator.uniform(size=(6, 2))*[3., 2.]+.5
        for alternative in range(6):
            rows.append(dict(choice=case, alternative="ABCDEF"[alternative], nest="A" if alternative < 3 else "B",
                             chosen=int(alternative == 0), available=not (alternative == 2 and case % 5 == 0
                                     or alternative == 5 and case % 7 == 0), respondent=case//4,
                             cost=float(design[alternative, 0]), quality=float(design[alternative, 1])))
    frame = pd.DataFrame(rows)
    truth = np.array([-.45, .35, .35, .40])
    reference = Reference(frame)
    probabilities = reference.moments(truth)["probability"]
    uniforms = generator.random((cases+3)//4 if shared_uniform else cases)
    frame["chosen"] = 0
    for ci, group in enumerate(reference.groups):
        uniform = uniforms[ci//4 if shared_uniform else ci]
        selected = min(int(np.searchsorted(probabilities[group].cumsum(), uniform, side="right")), len(group)-1)
        frame.loc[group[selected], "chosen"] = 1
    return frame, truth


def scientific(*, trials=100, cases=512):
    import numpy as np
    from scipy.stats import binom, binomtest
    import torch

    import openecon as oe
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.nested_logit_reference import Reference

    if not 100 <= trials <= 200 or not 240 <= cases <= 512:
        raise ValueError("Use100..200 uncensored replicates per VCE and240..512 choice cases.")
    if not Path(oe.__file__).resolve().is_relative_to(ROOT/"src"):
        raise ValueError("Run source science with PYTHONPATH=src:packages/openecon-charts/src.")
    inputs = {"verifier": Path(__file__), "oracle": ROOT/"scripts/nested_logit_reference.py", "example": EXAMPLE,
              "native_fit": ROOT/"src/openecon/econometrics/discrete/nested_logit.py",
              "native_postestimation": ROOT/"src/openecon/econometrics/discrete/nested_logit_postestimation.py"}
    hashes_before = {name: digest(path) for name, path in inputs.items()}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.monotonic()
    configurations, failures = {}, []
    namespace = runpy.run_path(str(EXAMPLE), init_globals={"NESTED_LOGIT_LIBRARY_ONLY": True})
    fixture = namespace["synthetic_choices"]()
    fixture_reference = Reference(fixture)
    fixture_best, fixture_starts = fixture_reference.fit()
    fixture_oracle = dict(physical_parameters=fixture_best.x.tolist(), log_likelihood=-float(fixture_best.fun),
                          multistart_parameters=[value.x.tolist() for value in fixture_starts],
                          multistart_negative_loglikelihood=[float(value.fun) for value in fixture_starts],
                          physical_score=fixture_reference.objective(fixture_best.x)[1].tolist(),
                          information=fixture_reference.information(fixture_best.x).tolist(),
                          fixture_sha256=hashlib.sha256(fixture.to_json(orient="split", double_precision=15).encode()).hexdigest())
    try:
        for vce_index, vce in enumerate(("oim", "hc0", "cr0")):
            estimates, covariances, covered, numerical_errors = [], [], [], []
            for replicate in range(trials):
                seed = 6_080_000+10_000*vce_index+replicate
                frame, truth = monte_carlo_fixture(seed, cases, shared_uniform=vce == "cr0")
                try:
                    result = oe.nlogit(frame, "chosen", ["cost", "quality"], case="choice",
                                       alternative="alternative", nest="nest", available="available", vce=vce,
                                       cluster="respondent" if vce == "cr0" else None)
                    fit = result.attrs["nested_logit_state"]["fit"]
                    estimate = np.asarray(fit["params"])
                    reference = Reference(frame)
                    expected = reference.covariance(estimate, vce)
                    errors = {}
                    for name in ("information", "bread", "meat", "covariance"):
                        actual = np.asarray(fit[name])
                        scale = np.sqrt(np.diag(expected[name]))
                        errors[name] = float(np.max(np.abs(actual-expected[name])/scale[:, None]/scale[None, :]))
                    numerical_errors.append(errors)
                    covariance = np.asarray(fit["covariance"])
                    intervals = result["parameters"]
                    estimates.append(estimate)
                    covariances.append(covariance)
                    covered.append((intervals.ci_lower.to_numpy() <= truth) & (truth <= intervals.ci_upper.to_numpy()))
                except Exception as error:
                    failures.append(dict(vce=vce, replicate=replicate, seed=seed,
                                         exception=type(error).__name__, code=getattr(error, "code", None), message=str(error)))
                if (replicate+1) % 20 == 0:
                    print(json.dumps(dict(vce=vce, attempted=replicate+1, accepted=len(estimates)), sort_keys=True), flush=True)
            accepted = len(estimates)
            if accepted < 2:
                configurations[vce] = dict(attempted=trials, accepted=accepted)
                continue
            estimates, covariances, covered = map(np.asarray, (estimates, covariances, covered))
            empirical = np.cov(estimates, rowvar=False, ddof=1)
            mean_covariance = covariances.mean(axis=0)
            sd = np.sqrt(np.diag(mean_covariance))
            normalized_mean = mean_covariance/sd[:, None]/sd[None, :]
            error = float(np.linalg.norm((empirical-mean_covariance)/sd[:, None]/sd[None, :], ord="fro"))
            allowance = float(3*np.sqrt((np.trace(normalized_mean)**2+np.sum(normalized_mean**2))/(accepted-1)))
            minimum = int(binom.ppf(.01/12, trials, .95))
            bias = estimates.mean(axis=0)-truth
            bias_allowance = 5*np.sqrt(np.diag(empirical)/accepted)
            parameters = {}
            for j, name in enumerate(("cost", "quality", "lambda[A]", "lambda[B]")):
                success_count = int(covered[:, j].sum())
                interval = binomtest(success_count, trials).proportion_ci(confidence_level=.99)
                parameters[name] = dict(true=float(truth[j]), mean=float(estimates[:, j].mean()), bias=float(bias[j]),
                                        covered=success_count, conservative_coverage=success_count/trials,
                                        coverage99_interval=[float(interval.low), float(interval.high)],
                                        empirical_variance=float(empirical[j, j]), mean_estimated_variance=float(mean_covariance[j, j]))
            configurations[vce] = dict(attempted=trials, accepted=accepted, parameters=parameters,
                                       empirical_covariance=empirical.tolist(), mean_estimated_covariance=mean_covariance.tolist(),
                                       normalized_full_covariance_error=error, covariance_error_allowance=allowance,
                                       covariance_gate_passed=error <= allowance, bias_gate_passed=bool((abs(bias) <= bias_allowance).all()),
                                       coverage_minimum_count=minimum, coverage_gate_passed=bool((covered.sum(axis=0) >= minimum).all()),
                                       maximum_standardized_cell_error={name: max(row[name] for row in numerical_errors) for name in numerical_errors[0]},
                                       numerical_gate_passed=all(max(row.values()) < 2e-5 for row in numerical_errors),
                                       generation="same uniform shared across4 choice cases per respondent; independent respondents" if vce == "cr0" else "independent uniform per choice case")
    finally:
        torch.set_num_threads(previous_threads)
    hashes_after = {name: digest(path) for name, path in inputs.items()}
    passed = hashes_before == hashes_after and not failures and all(value.get("coverage_gate_passed")
                and value.get("bias_gate_passed") and value.get("covariance_gate_passed") and value.get("numerical_gate_passed")
                for value in configurations.values())
    return dict(status="passed" if passed else "failed", source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                input_hashes_before=hashes_before, input_hashes_after=hashes_after,
                source_inputs_unchanged_during_run=hashes_before == hashes_after, primary_sources=PRIMARY_SOURCES,
                trials_per_vce=trials, cases_per_replicate=cases, uncensored_attempted_replicates=3*trials,
                failures=failures, configurations=configurations, fixture_multistart_oracle=fixture_oracle, duration_seconds=time.monotonic()-started,
                coverage_failure_policy="every attempted replicate is retained; rejected boundary/unidentified fits count uncovered; no replacement seeds",
                confidence_policy="physical normal beta intervals; logistic transformed normal free-dissimilarity intervals",
                covariance_allowance_definition="3*sqrt((trace(R)^2+sum(R^2))/(accepted-1)); R=mean covariance scaled by its diagonal; broad Monte Carlo diagnostic, not a p-value",
                numerical_oracle="independent NumPy physical scores and five-point observed Hessian; every fullmatrix cell",
                third_party_estimation_runtime=False, proprietary_software_run=False, global_maximum_claimed=False,
                native_window_verified=False, public_release_delivered=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--directory", type=Path)
    group.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.directory is not None:
        portable(args.directory, show=not args.no_display)
    else:
        result = scientific(trials=args.trials, cases=args.cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "uncensored_attempted_replicates", "duration_seconds")}))
        if result["status"] != "passed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
