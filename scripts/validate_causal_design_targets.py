"""A bounded, independently known-DGP interval diagnostic for causal targets.

Run source mode with this checkout on PYTHONPATH. Forty repetitions are a small
calibration diagnostic, not general method validation, an equivalence claim or
a coverage guarantee. The complete estimates/SEs/intervals/truths are retained.
No acceptance threshold is imposed on the realized diagnostic coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch

import openecon as oe


REPETITIONS = 40
ARM_N = 400
THRESHOLDS = [0.0, 1.0, 2.0]
QUANTILES = [0.25, 0.5, 0.75]
LEVEL = 0.95
BOOTSTRAP_REPS = 199
TAU = 2.5
MEANS = [1.0, 1.4]
SEED_BASE = 260_307
TARGETS = [f"cdf:{threshold}" for threshold in THRESHOLDS]
TARGETS += [f"quantile:{probability}" for probability in QUANTILES]
TARGETS += ["rmst:difference"]


def _cdf(value):
    """Independent analytic standard-normal CDF, with no estimator imports."""
    return 0.5 * math.erfc(-value / math.sqrt(2))


def _rmst(mean):
    """Integral of exponential survival exp(-t/mean) on [0,tau]."""
    return -mean * math.expm1(-TAU / mean)


def _wilson(successes, count):
    z = NormalDist().inv_cdf(0.975)
    observed = successes / count
    denominator = 1 + z * z / count
    center = (observed + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(observed * (1 - observed) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _numpy_global_digest():
    state = np.random.get_state()
    return hashlib.sha256(state[0].encode() + state[1].tobytes() + repr(state[2:]).encode()).hexdigest()


def _torch_global_digest():
    return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest()


def _data(seed):
    """Fixed arm counts, independent errors, independently permuted assignment."""
    rng = np.random.default_rng(seed)
    treatment = rng.permutation(np.repeat([0, 1], ARM_N))
    normal = pd.DataFrame({
        "outcome": rng.normal(size=2 * ARM_N) + treatment,
        "treated": treatment,
    })
    event_times = rng.exponential(scale=np.where(treatment == 1, MEANS[1], MEANS[0]))
    # Both random censoring and a fixed administrative limit are independent
    # of event times. No follow-up row is inserted or outcome-conditioned retry
    # performed to make the target pass its observed support domain.
    censor_times = np.minimum(rng.exponential(scale=4.0, size=2 * ARM_N), 3.0)
    survival = pd.DataFrame({
        "time": np.minimum(event_times, censor_times),
        "event": (event_times <= censor_times).astype(int),
        "treated": treatment,
    })
    return normal, survival


def _sample_digest(data):
    schema = json.dumps({name: str(dtype) for name, dtype in data.dtypes.items()}, sort_keys=True)
    rows = pd.util.hash_pandas_object(data, index=True).values.tobytes()
    return hashlib.sha256(schema.encode() + rows).hexdigest()


def _record(repetition, procedure, target, row, truth):
    record = {
        "repetition": repetition, "procedure": procedure, "target": target,
        "estimate": float(row["estimate"]), "std_error": float(row["std_error"]),
        "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
        "true_value": float(truth), "nominal_level": LEVEL,
    }
    if not all(math.isfinite(record[key]) for key in ("estimate", "std_error", "ci_low", "ci_high", "true_value")):
        raise ValueError("Diagnostic target contains a nonfinite scientific field.")
    if record["ci_low"] > record["ci_high"] or record["std_error"] < 0:
        raise ValueError("Diagnostic interval or standard error is invalid.")
    record["covered"] = record["ci_low"] <= record["true_value"] <= record["ci_high"]
    return record


def validate(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    before = {"numpy": _numpy_global_digest(), "torch": _torch_global_digest()}
    records, replications, failures = [], [], []
    for repetition in range(REPETITIONS):
        seed, bootstrap_seed = SEED_BASE + repetition, 1729 + repetition
        normal, survival = _data(seed)
        receipt = {
            "repetition": repetition, "dgp_seed": seed, "bootstrap_seed": bootstrap_seed,
            "sample_counts": [ARM_N, ARM_N], "normal_sample_sha256": _sample_digest(normal),
            "survival_sample_sha256": _sample_digest(survival),
            "survival_arm_max_followup": [
                float(survival.loc[survival.treated == arm, "time"].max()) for arm in (0, 1)
            ],
        }
        try:
            cdf = oe.treatment_cdf(normal, "outcome", "treated", design="randomized",
                                   thresholds=THRESHOLDS, level=LEVEL)
            quantile = oe.treatment_quantile(normal, "outcome", "treated", design="randomized",
                                             quantiles=QUANTILES, reps=BOOTSTRAP_REPS,
                                             seed=bootstrap_seed, level=LEVEL)
            rmst = oe.treatment_rmst(survival, "time", "event", "treated", design="randomized",
                                     tau=TAU, level=LEVEL)
            current = []
            for threshold, row in zip(THRESHOLDS, cdf["effects"].to_dict("records"), strict=True):
                truth = _cdf(threshold - 1.0) - _cdf(threshold)
                current.append(_record(repetition, "treatment_cdf", f"cdf:{threshold}", row, truth))
            for probability, row in zip(QUANTILES, quantile["effects"].to_dict("records"), strict=True):
                current.append(_record(repetition, "treatment_quantile", f"quantile:{probability}", row, 1.0))
            rmst_difference = rmst["effects"].loc[rmst["effects"].term == "difference"].iloc[0].to_dict()
            current.append(_record(repetition, "treatment_rmst", "rmst:difference", rmst_difference,
                                   _rmst(MEANS[1]) - _rmst(MEANS[0])))
            assert [row["target"] for row in current] == TARGETS
            receipt["artifact_sha256s"] = {
                name: oe.causal_design_save(fit)["sha256"] for name, fit in [
                    ("treatment_cdf", cdf), ("treatment_quantile", quantile), ("treatment_rmst", rmst)
                ]
            }
            receipt["status"] = "completed"
            records.extend(current)
            replications.append(receipt)
        except Exception as exc:
            # Preserve an actual failed draw. Never discard a failed support draw
            # and reroll to obtain a favorable completed diagnostic.
            failure = receipt | {"status": "failed", "exception": type(exc).__name__,
                                 "error_code": getattr(exc, "code", None), "message": str(exc)}
            failures.append(failure)
            replications.append(failure)
    after = {"numpy": _numpy_global_digest(), "torch": _torch_global_digest()}
    unchanged = before == after
    coverage = {}
    for target in TARGETS:
        rows = [row for row in records if row["target"] == target]
        covered = sum(row["covered"] for row in rows)
        count = len(rows)
        coverage[target] = {
            "completed": count, "covered": covered,
            "observed_coverage": covered / count if count else None,
            "wilson_95_interval": _wilson(covered, count) if count else None,
            "mean_estimate": float(np.mean([row["estimate"] for row in rows])) if count else None,
            "mean_std_error": float(np.mean([row["std_error"] for row in rows])) if count else None,
        }
    report = {
        "schema": "openecon.causal_design.bounded_interval_diagnostic.v1",
        "execution": "source CPU float64 only", "repetitions_requested": REPETITIONS,
        "repetitions_completed": REPETITIONS - len(failures), "targets": TARGETS,
        "nominal_level": LEVEL, "rows_per_arm": ARM_N, "bootstrap_reps": BOOTSTRAP_REPS,
        "predeclared_settings": {
            "thresholds": THRESHOLDS, "quantiles": QUANTILES, "tau": TAU,
            "dgp_seed_base": SEED_BASE, "bootstrap_seed_base": 1729,
            "normal_mean_shift": 1.0, "normal_sigma": 1.0,
            "survival_event_means": MEANS, "censor_mean": 4.0, "administrative_censor_time": 3.0,
        },
        "independent_truth": {
            "cdf": "Phi(threshold-1)-Phi(threshold) via math.erfc",
            "quantile": "Normal(1,1) marginal quantile minus Normal(0,1) marginal quantile = 1",
            "rmst": "mean1*(1-exp(-tau/mean1))-mean0*(1-exp(-tau/mean0)) via math.expm1",
            "rmst_control": _rmst(MEANS[0]), "rmst_treated": _rmst(MEANS[1]),
            "rmst_difference": _rmst(MEANS[1]) - _rmst(MEANS[0]),
        },
        "coverage": coverage, "records": records, "replications": replications, "failures": failures,
        "global_rng_unchanged": unchanged, "rng_before_sha256": before, "rng_after_sha256": after,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "torch": str(torch.__version__)},
        "scope": "Small finite-DGP interval diagnostic at declared targets. Wilson intervals describe "
                 "Monte Carlo uncertainty of observed coverage across independent repetitions. These "
                 "runs do not establish universal coverage, arbitrary distributions/designs, individual "
                 "counterfactual effects, clinical validity, simultaneous bands or Stata parity.",
        "coverage_acceptance_threshold": None, "stata_parity_validated": False,
        "status": "completed" if not failures and unchanged else "failed",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "repetitions_completed": report["repetitions_completed"],
                      "coverage": coverage, "global_rng_unchanged": unchanged}, sort_keys=True))
    if failures or not unchanged:
        raise RuntimeError("Bounded interval sweep did not complete all 40 draws with unchanged global RNG.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/causal-design-validation/bounded-coverage.json"))
    validate(parser.parse_args().output)
