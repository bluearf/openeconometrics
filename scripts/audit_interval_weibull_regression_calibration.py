"""Execute the immutable predeclared128-seed Weibull ML/delta coverage plan."""

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext.weibull_regression import (
    stinterval_weibull_regression,
    interval_weibull_regression_predict,
)


def main(output):
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "docs/econometrics/interval-weibull-covariate-calibration-plan.json"
    pin_path = Path("/tmp/market-689-preregistration.json")
    pin = json.loads(pin_path.read_text())
    plan = json.loads(plan_path.read_text())
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if plan_hash != pin["sha256"]:
        raise ValueError(
            "Preregistered scientific plan changed; preserve the original receipt and refuse"
        )
    source_paths = sorted(
        (root / "src/openecon/econometrics/survival_ext").glob("weibull_regression*.py")
    )
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    initial = {
        "fixed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_sha256": hashes,
        "plan_sha256": plan_hash,
        "preregistration": pin,
        "experiment_started": False,
        "generating_scripts_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "examples/interval_weibull_regression.py", Path(__file__))
        },
    }
    Path("/tmp/market-689-pre-calibration-source-pin.json").write_text(
        json.dumps(initial, indent=2) + "\n"
    )
    loader = importlib.util.spec_from_file_location(
        "weibull_calibration_example", root / "examples/interval_weibull_regression.py"
    )
    example = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(example)
    truth = np.r_[plan["aft_truth"], plan["log_sigma_truth"]]
    shape = np.exp(-truth[-1])
    phtruth = np.r_[-shape * truth[:-1], -truth[-1]]
    profiles = np.array(plan["query_profiles"])
    query = pd.DataFrame(profiles, columns=["x1", "x2"])
    times = np.array(plan["query_times"])
    eta = np.column_stack((np.ones(len(profiles)), profiles)) @ truth[:-1]
    trueh = np.zeros((len(profiles), len(times)))
    trueh[:, 1:] = np.exp(shape * (np.log(times[1:])[None, :] - eta[:, None]))
    trues = np.exp(-trueh)
    positive = np.tile(times > 0, len(profiles))
    z = norm.ppf((1 + plan["inference_level"]) / 2)
    counts = np.zeros(len(truth), dtype=int)
    phcounts = counts.copy()
    wrongcounts = counts.copy()
    scounts = np.zeros(sum(positive), dtype=int)
    hcounts = scounts.copy()
    joint = 0
    results = []
    means = []
    covariances = []
    for index, seed in enumerate(plan["seeds"]):
        entry = {"seed": seed, "accepted": False}
        try:
            data = example.case(seed, plan["n"])
            entry["reference_input_sha256"] = hashlib.sha256(
                data.to_csv(index=True, float_format="%.17g").encode()
            ).hexdigest()
            model = stinterval_weibull_regression(
                data,
                lower="lower",
                upper="upper",
                x=["x1", "x2"],
                level=plan["inference_level"],
                maxiter=plan["fit_maxiter"],
                max_work=plan["fit_max_work"],
            )
            pred = interval_weibull_regression_predict(
                model, query, times=plan["query_times"], level=plan["inference_level"]
            )
            raw = np.array(model.payload["result"]["aft_parameters"])
            cov = np.array(model.payload["result"]["aft_covariance"])
            pp = np.array(model.payload["result"]["ph_parameters"])
            pc = np.array(model.payload["result"]["ph_covariance"])
            se = np.sqrt(np.diag(cov))
            pse = np.sqrt(np.diag(pc))
            aft_hits = np.abs(raw - truth) <= z * se
            ph_hits = np.abs(pp - phtruth) <= z * pse
            wrong_hits = np.abs(raw - truth) <= z * se * np.sqrt(
                plan["thresholds"]["negative_control_covariance_multiplier"]
            )
            distance = float((raw - truth) @ np.linalg.solve(cov, raw - truth))
            joint_hit = distance <= chi2.ppf(plan["inference_level"], len(truth))
            rows = pred.payload["result"]["rows"]
            slo = np.array([row["survival_lower"] for row in rows])[positive]
            shi = np.array([row["survival_upper"] for row in rows])[positive]
            hlo = np.array([row["hazard_lower"] for row in rows])[positive]
            hhi = np.array([row["hazard_upper"] for row in rows])[positive]
            sv = trues.flatten()[positive]
            hv = trueh.flatten()[positive]
            survival_hits = (slo <= sv) & (sv <= shi)
            hazard_hits = (hlo <= hv) & (hv <= hhi)
            entry.update(
                accepted=True,
                aft=raw.tolist(),
                aft_covariance=cov.tolist(),
                ph=pp.tolist(),
                ph_covariance=pc.tolist(),
                joint_distance=distance,
                query_estimates=list(pred.payload["result"]["estimates"]),
                query_covariance=[list(row) for row in pred.payload["result"]["covariance"]],
                score_quadratic=model.payload["endpoint"]["score_quadratic"],
                state_sha256=model.payload["sha256"],
                query_sha256=pred.payload["sha256"],
            )
            # Commit all hits only after the entire seed's endpoint/query audit
            # succeeds. A late query error is a failure for every denominator.
            counts += aft_hits
            phcounts += ph_hits
            wrongcounts += wrong_hits
            joint += joint_hit
            scounts += survival_hits
            hcounts += hazard_hits
            means.append(raw)
            covariances.append(cov)
        except (AnalysisError, ValueError, RuntimeError) as exc:
            entry.update(error_code=getattr(exc, "code", type(exc).__name__), error=str(exc))
        results.append(entry)
        if (index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "fixed_denominator": plan["replications"],
                        "failures": sum(not r["accepted"] for r in results),
                    }
                ),
                flush=True,
            )
    total = plan["replications"]
    thresholds = plan["thresholds"]
    aggregation_available = len(means) > 1
    if aggregation_available:
        empirical = np.cov(np.array(means).T, ddof=1)
        average = np.mean(covariances, axis=0)
        chol = np.linalg.cholesky(average)
        whitened = np.linalg.solve(chol, empirical)
        whitened = np.linalg.solve(chol, whitened.T).T
        eigenvalues = np.linalg.eigvalsh((whitened + whitened.T) / 2)
        bias = np.mean(means, axis=0) - truth
        mcse = np.sqrt(np.diag(empirical) / len(means))
    else:
        eigenvalues = bias = mcse = None
    failures = sum(not row["accepted"] for row in results)
    gates = {
        "all_seeds_in_denominator": len(results) == total,
        "failures_within_declared_cap": failures <= thresholds["maximum_fit_or_admission_failures"],
        "aft_parameter_coverage": bool(
            np.all(counts / total >= thresholds["each_aft_and_ph_parameter_coverage_min"])
        ),
        "ph_parameter_coverage": bool(
            np.all(phcounts / total >= thresholds["each_aft_and_ph_parameter_coverage_min"])
        ),
        "joint_aft_coverage": bool(joint / total >= thresholds["joint_aft_ellipsoid_coverage_min"]),
        "survival_pointwise_coverage": bool(
            np.all(scounts / total >= thresholds["each_positive_survival_and_hazard_coverage_min"])
        ),
        "hazard_pointwise_coverage": bool(
            np.all(hcounts / total >= thresholds["each_positive_survival_and_hazard_coverage_min"])
        ),
        "bias_mcse": aggregation_available
        and bool(np.all(np.abs(bias) / mcse <= thresholds["aft_bias_mcse_multiple_max"])),
        "full_covariance_empirical_eigenvalues": aggregation_available
        and bool(
            eigenvalues[0] >= thresholds["full_covariance_whitened_empirical_eigenvalue_range"][0]
            and eigenvalues[-1]
            <= thresholds["full_covariance_whitened_empirical_eigenvalue_range"][1]
        ),
        "wrong_covariance_negative_control_detected": bool(
            np.any(
                wrongcounts / total < thresholds["negative_control_at_least_one_aft_coverage_below"]
            )
        ),
    }
    unchanged = all(
        hashlib.sha256((root / path).read_bytes()).hexdigest() == value
        for path, value in hashes.items()
    )
    gates["exact_source_and_plan_unchanged"] = (
        unchanged and hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_hash
    )
    receipt = {
        "issue": "MARKET-689",
        "source_pin": initial,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_unchanged": unchanged,
        "full_failure_denominator": total,
        "failures": failures,
        "aft_coverage": (counts / total).tolist(),
        "ph_coverage": (phcounts / total).tolist(),
        "joint_coverage": float(joint / total),
        "survival_coverage": (scounts / total).tolist(),
        "hazard_coverage": (hcounts / total).tolist(),
        "wrong_variance_coverage": (wrongcounts / total).tolist(),
        "aft_bias": None if bias is None else bias.tolist(),
        "bias_mcse": None if mcse is None else mcse.tolist(),
        "full_covariance_empirical_whitened_eigenvalues": None
        if eigenvalues is None
        else eigenvalues.tolist(),
        "gates": gates,
        "passed": all(gates.values()),
        "results": results,
    }
    Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key not in ("results", "source_pin")}
        )
    )
    return receipt["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.output) else 1)
