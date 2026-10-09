"""Reproducible independent MARKET-182 scientific validation (development only).

The runtime remains native Torch. NumPy/SciPy/statsmodels below are independent
development oracles and data generators, never numerical runtime dependencies.
Run --protocol-only first: the runner refuses to replace a different protocol.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests as reference
import torch

import openecon as oe

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/evidence/market-182"
METHODS = ("bonferroni", "sidak", "holm", "holm_sidak", "hochberg", "hommel", "bh", "by")
REF_NAMES = {
    "holm_sidak": "holm-sidak",
    "hochberg": "simes-hochberg",
    "bh": "fdr_bh",
    "by": "fdr_by",
}
DESIGNS = ("independent", "positive_prds", "mixed_sign_dependence")
FAMILIES, JOINT_DRAWS, ALPHAS = 5000, 4999, (0.01, 0.05, 0.10)


def digest(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def protocol():
    files = (
        "src/openecon/econometrics/postest/multiple.py",
        "scripts/validate_multiple_testing.py",
    )
    return {
        "schema": 1,
        "date": "2026-10-07",
        "families_per_cell": FAMILIES,
        "alphas": ALPHAS,
        "hypotheses": 12,
        "joint_hypotheses": 6,
        "joint_draws": JOINT_DRAWS,
        "observation_seed": 49172937,
        "null_seed": 83410991,
        "ci_seed": 17195739,
        "ci_observation_seed": 28576109,
        "ci_draws": 50000,
        "adjustment_designs": DESIGNS,
        "null_configurations": ["full_null", "half_false"],
        "positive_prds_methods": [m for m in METHODS if m not in ("sidak", "holm_sidak")],
        "mixed_sign_methods": ["bonferroni", "holm", "by"],
        "joint_designs": [*DESIGNS, "complete_signed_enumeration"],
        "ci_designs": [
            "independent",
            "positive_prds",
            "mixed_sign_dependence",
            "perfect_correlation",
        ],
        "signal_mean": 3.5,
        "normal_tests": "Known unit marginal variance, one-sided adjustments; two-sided joint tests",
        "joint_validity": "Known fixed covariance and location shifts imply subset pivotality; no estimated/model-specific null validity claim",
        "signed_validity": "All 64 equiprobable six-sign assignments, fixed comparable normalized design; null marginal probabilities computed from all assignments",
        "gates": {
            "rate_or_fdr_upper95_at_most_alpha_plus": 0.012,
            "coverage_lower95_at_least_nominal_minus": 0.012,
            "normal_analytic_cdf_error_at_most_mc_se_multiple": 4.0,
            "adjustment_reference_absolute_tolerance": 4e-15,
            "any_failure_fails": True,
        },
        "denominator": "All 5000 planned families; failed families are retained and never replaced",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        "scope": "Bounded source CPU-resident summary procedures; no installed desktop, CUDA, streaming or licensed vendor parity",
    }


def frozen_protocol(destination):
    current = json.loads(json.dumps(protocol()))
    path = destination / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise RuntimeError("Frozen protocol/source changed; do not overwrite prior outcomes.")
    else:
        path.write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    return current


def wilson(successes, n):
    z = 1.959963984540054
    p, scale = successes / n, 1 + z * z / n
    middle = (p + z * z / (2 * n)) / scale
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0.0, middle - radius), min(1.0, middle + radius)]


def covariance(design, n):
    if design == "independent":
        return np.eye(n)
    if design == "perfect_correlation":
        return np.ones((n, n))
    loading = np.ones(n) if design == "positive_prds" else np.array([(-1.0) ** j for j in range(n)])
    return 0.65 * np.outer(loading, loading) + 0.35 * np.eye(n)


def normal_draws(seed, design, n, rows):
    # Independent NumPy PCG64 and covariance square root; not runtime Torch RNG.
    rng = np.random.default_rng(seed)
    cov = covariance(design, n)
    eig, vectors = np.linalg.eigh(cov)
    return rng.normal(size=(rows, n)) @ (vectors * np.sqrt(np.maximum(eig, 0.0))).T


def summarize(adjusted, true_null, failures):
    summaries, counts = [], {}
    for alpha in ALPHAS:
        rejected = adjusted <= alpha
        false = rejected[:, true_null].sum(1)
        total = rejected.sum(1)
        fdp = false / np.maximum(total, 1)
        fwer = np.count_nonzero(false) / FAMILIES
        fdr = float(fdp.sum() / FAMILIES)
        fdr_se = float(fdp.std(ddof=1) / math.sqrt(FAMILIES))
        bounds = wilson(np.count_nonzero(false), FAMILIES)
        summaries.append(
            {
                "alpha": alpha,
                "false_rejection_families": int(np.count_nonzero(false)),
                "planned_families": FAMILIES,
                "failed_families": len(failures),
                "fwer": float(fwer),
                "fwer_wilson95": bounds,
                "fdr": fdr,
                "fdr_mc_se": fdr_se,
                "fdr_normal95": [max(0.0, fdr - 1.96 * fdr_se), min(1.0, fdr + 1.96 * fdr_se)],
            }
        )
        counts[str(alpha)] = {"false": false.tolist(), "total": total.tolist()}
    return summaries, counts


def adjustments(plan):
    records = []
    for index, design in enumerate(DESIGNS):
        allowed = (
            METHODS
            if index == 0
            else plan["positive_prds_methods"]
            if index == 1
            else plan["mixed_sign_methods"]
        )
        noise = normal_draws(plan["observation_seed"] + 100 * index, design, 12, FAMILIES)
        for partial in (False, True):
            mean = np.r_[np.full(6, 3.5), np.zeros(6)] if partial else np.zeros(12)
            true_null = mean == 0
            p = norm.sf(noise + mean)
            for method in allowed:
                adjusted = np.full_like(p, np.nan)
                failures, max_error = [], 0.0
                for row, values in enumerate(p):
                    try:
                        result = oe.multipletests(values, method=method)
                        adjusted[row] = result.adjusted_p_value
                        # Independent documented community reference, distributed across the full sample.
                        if row % 25 == 0:
                            expected = reference(values, method=REF_NAMES.get(method, method))[1]
                            max_error = max(
                                max_error, float(np.max(np.abs(expected - adjusted[row])))
                            )
                    except Exception as error:
                        failures.append(
                            {"row": row, "error": type(error).__name__, "message": str(error)}
                        )
                summary, counts = summarize(adjusted, true_null, failures)
                target = "fdr" if method in ("bh", "by") else "fwer"
                passed = (
                    not failures
                    and max_error <= plan["gates"]["adjustment_reference_absolute_tolerance"]
                )
                for cell in summary:
                    upper = cell["fdr_normal95"][1] if target == "fdr" else cell["fwer_wilson95"][1]
                    cell["applicable_target"] = target
                    cell["passed"] = bool(upper <= cell["alpha"] + 0.012)
                    passed = passed and cell["passed"]
                record = {
                    "design": design,
                    "configuration": "half_false" if partial else "full_null",
                    "method": method,
                    "seed": plan["observation_seed"] + 100 * index,
                    "true_null_positions": np.flatnonzero(true_null).tolist(),
                    "input_sha256": digest(p),
                    "adjusted_sha256": digest(adjusted),
                    "reference_rows": 200,
                    "reference_max_absolute_error": max_error,
                    "failures": failures,
                    "summary": summary,
                    "family_rejection_counts": counts,
                    "passed": bool(passed),
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "stage": "adjustments",
                            "design": design,
                            "partial": partial,
                            "method": method,
                            "passed": bool(passed),
                        }
                    ),
                    flush=True,
                )
    return records


def independent_stepdown(observed, null, min_p):
    # NumPy row-specific order, independent suffix extrema and monotone p-value formula.
    order = np.argsort(observed if min_p else -observed, axis=1, kind="stable")
    result = np.empty_like(observed)
    for row, ordering in enumerate(order):
        matrix = null[:, ordering]
        extremum = (
            np.minimum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
            if min_p
            else np.maximum.accumulate(matrix[:, ::-1], axis=1)[:, ::-1]
        )
        cutoff = observed[row, ordering]
        probability = (extremum <= cutoff).sum(0) if min_p else (extremum >= cutoff).sum(0)
        probability = (probability + 1) / (len(null) + 1)
        result[row, ordering] = np.maximum.accumulate(probability)
    return result


def joint_tests(plan):
    records = []
    for index, design in enumerate(plan["joint_designs"]):
        enumerated = design == "complete_signed_enumeration"
        if enumerated:
            rng = np.random.default_rng(plan["null_seed"] + 100 * index)
            matrix = rng.normal(size=(6, 6))
            matrix /= np.sqrt((matrix * matrix).sum(0))
            signs = np.array(list(product((-1.0, 1.0), repeat=6)))
            null = signs @ matrix
            observed_rng = np.random.default_rng(plan["observation_seed"] + 100 * index)
            noise = null[observed_rng.integers(0, len(null), size=FAMILIES)]
            null_p = (np.abs(null[:, None, :]) >= np.abs(null[None, :, :])).mean(0)
        else:
            null = normal_draws(plan["null_seed"] + 100 * index, design, 6, JOINT_DRAWS)
            noise = normal_draws(plan["observation_seed"] + 100 * index, design, 6, FAMILIES)
            null_p = 2 * norm.sf(np.abs(null))
        for partial in (False, True):
            mean = np.r_[np.full(3, 3.5), np.zeros(3)] if partial else np.zeros(6)
            true_null = mean == 0
            observed = noise + mean
            supplied_p = (
                (np.abs(null[:, None, :]) >= np.abs(observed[None, :, :])).mean(0)
                if enumerated
                else 2 * norm.sf(np.abs(observed))
            )
            for method in ("romano_wolf", "westfall_young"):
                min_p = method == "westfall_young"
                supplied = supplied_p if min_p else observed
                native_null = torch.as_tensor(null_p if min_p else null, dtype=torch.float64)
                adjusted, failures, max_error = np.full_like(supplied, np.nan), [], 0.0
                # Continuous normal draws have no ties; exact discrete tied-orbit reference is tested separately.
                oracle = (
                    None
                    if enumerated
                    else independent_stepdown(
                        supplied if min_p else np.abs(supplied),
                        null_p if min_p else np.abs(null),
                        min_p,
                    )
                )
                for row, values in enumerate(supplied):
                    try:
                        result = oe.stepdown(
                            values,
                            native_null,
                            method=method,
                            calibration="enumerated" if enumerated else "monte_carlo",
                            null_description=plan["signed_validity"]
                            if enumerated
                            else plan["joint_validity"],
                        )
                        adjusted[row] = result.adjusted_p_value
                        if oracle is not None:
                            max_error = max(
                                max_error, float(np.max(np.abs(oracle[row] - adjusted[row])))
                            )
                    except Exception as error:
                        failures.append(
                            {"row": row, "error": type(error).__name__, "message": str(error)}
                        )
                summary, counts = summarize(adjusted, true_null, failures)
                passed = not failures and max_error < 4e-15
                for cell in summary:
                    cell["applicable_target"] = "fwer"
                    cell["passed"] = bool(cell["fwer_wilson95"][1] <= cell["alpha"] + 0.012)
                    passed = passed and cell["passed"]
                records.append(
                    {
                        "design": design,
                        "configuration": "half_false" if partial else "full_null",
                        "method": method,
                        "observation_seed": plan["observation_seed"] + 100 * index,
                        "null_seed": plan["null_seed"] + 100 * index,
                        "draws": len(null),
                        "input_sha256": digest(supplied),
                        "null_sha256": digest(native_null.numpy()),
                        "adjusted_sha256": digest(adjusted),
                        "oracle_max_absolute_error": max_error,
                        "true_null_positions": np.flatnonzero(true_null).tolist(),
                        "failures": failures,
                        "summary": summary,
                        "family_rejection_counts": counts,
                        "passed": bool(passed),
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": "joint",
                            "design": design,
                            "partial": partial,
                            "method": method,
                            "passed": bool(passed),
                        }
                    ),
                    flush=True,
                )
    return records


def confidence_intervals(plan):
    records = []
    for index, design in enumerate(plan["ci_designs"]):
        cov = covariance(design, 6)
        noise = normal_draws(plan["ci_observation_seed"] + index, design, 6, FAMILIES)
        for alpha in ALPHAS:
            result = oe.simultaneous_ci(
                np.zeros(6),
                cov,
                alpha=alpha,
                draws=plan["ci_draws"],
                seed=plan["ci_seed"] + index,
                family_description="Known fixed covariance of compatible normal location estimates",
            )
            critical = result.attrs["critical_value"]
            # Shift equivariance: translating all saved endpoints by each native-input estimate
            # gives exactly the family intervals. A fresh critical simulation per translation is unnecessary.
            covered = np.all(
                (noise + result.ci_low.to_numpy() <= 0) & (noise + result.ci_high.to_numpy() >= 0),
                axis=1,
            )
            count = int(covered.sum())
            bounds = wilson(count, FAMILIES)
            exact_cdf = (
                float((2 * norm.cdf(critical) - 1) ** 6)
                if design == "independent"
                else float(2 * norm.cdf(critical) - 1)
                if design == "perfect_correlation"
                else None
            )
            passed = bounds[0] >= 1 - alpha - 0.012
            if exact_cdf is not None:
                passed = (
                    passed
                    and abs(exact_cdf - (1 - alpha))
                    <= 4 * result.attrs["cdf_monte_carlo_std_error"] + 2 / plan["ci_draws"]
                )
            records.append(
                {
                    "design": design,
                    "alpha": alpha,
                    "planned_families": FAMILIES,
                    "failed_families": 0,
                    "covered_families": count,
                    "coverage": count / FAMILIES,
                    "coverage_wilson95": bounds,
                    "critical_value": critical,
                    "analytic_critical_value": float(norm.ppf((1 + (1 - alpha) ** (1 / 6)) / 2))
                    if design == "independent"
                    else float(norm.ppf(1 - alpha / 2))
                    if design == "perfect_correlation"
                    else None,
                    "analytic_cdf_at_native_critical": exact_cdf,
                    "metadata": result.attrs,
                    "observation_seed": plan["ci_observation_seed"] + index,
                    "input_sha256": digest(noise),
                    "covered": covered.tolist(),
                    "passed": bool(passed),
                }
            )
            print(
                json.dumps(
                    {"stage": "ci", "design": design, "alpha": alpha, "passed": bool(passed)}
                ),
                flush=True,
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = frozen_protocol(args.output)
    print(
        json.dumps(
            {"protocol_sha256": digest(plan), "protocol": str(args.output / "protocol.json")}
        ),
        flush=True,
    )
    if args.protocol_only:
        return
    output = args.output / "scientific-validation.json.gz"
    if output.exists():
        raise RuntimeError(
            "Existing outcomes are immutable; inspect them rather than silently rerunning."
        )
    torch.set_num_threads(1)
    started = time.monotonic()
    records = {
        "adjustments": adjustments(plan),
        "joint_tests": joint_tests(plan),
        "confidence_intervals": confidence_intervals(plan),
    }
    passed = all(row["passed"] for group in records.values() for row in group)
    receipt = {
        "schema": 1,
        "protocol_sha256": digest(plan),
        "protocol": plan,
        **records,
        "all_planned_families_retained": True,
        "passed": passed,
        "elapsed_seconds": time.monotonic() - started,
        "source_cpu_verified": True,
        "installed_desktop_verified": False,
        "cuda_verified": False,
        "licensed_vendor_verified": False,
        "stata_parity_validated": False,
    }
    data = json.dumps(receipt, allow_nan=False, separators=(",", ":")).encode()
    with output.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
            compressed.write(data)
    print(
        json.dumps(
            {
                "passed": passed,
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "seconds": receipt["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
