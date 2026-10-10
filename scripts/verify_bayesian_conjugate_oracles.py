#!/usr/bin/env python3
"""Independent analytic reference for a saved posterior or actual example receipt.

No OpenEconometrics/Torch imports. NumPy/SciPy are development oracles only.
Tolerances are declared here before inspecting the supplied receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import t

RTOL = 3e-10
UNIT_ATOL = 3e-12


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _compare(actual, reference, label):
    actual, reference = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        raise AssertionError(f"{label}: mismatched dimensions/nonfinite output")
    units = float(np.max(np.abs(reference))) if reference.size else 0.
    tolerance = RTOL*np.abs(reference) + UNIT_ATOL*max(units, np.finfo(float).tiny)
    if not np.all(np.abs(actual-reference) <= tolerance):
        raise AssertionError(f"{label}: independent analytic comparison failed")
    return {"shape": list(reference.shape), "max_absolute_error": float(np.max(np.abs(actual-reference))) if reference.size else 0.,
            "reference_unit": units}


def verify(receipt):
    if receipt.get("schema_version") == "openecon.posterior_contrast.v1":
        receipt = {"posterior": receipt["posterior"], "contrast": receipt}
    bundle = receipt.get("posterior", receipt)
    if bundle["schema_version"] != "openecon.posterior_bundle.v1":
        raise AssertionError("unsupported posterior schema")
    if _digest({k: v for k, v in bundle.items() if k != "integrity_sha256"}) != bundle["integrity_sha256"]:
        raise AssertionError("posterior digest mismatch")
    state, prior = bundle["state"], bundle["prior"]
    positions = state["sample_positions"]
    source = state["source_values"]
    complete = [i for i in range(len(source[0])) if all(column[i] is not None for column in source)]
    if positions != complete or (state["missing"] == "raise" and len(complete) != len(source[0])):
        raise AssertionError("saved sample is not the declared complete-case sample")
    y = np.array([source[0][i] for i in positions])
    x = np.array([[column[i] for column in source[1:]] for i in positions]).reshape(len(y), len(source)-1)
    if state["intercept"]:
        x = np.column_stack((np.ones(len(y)), x))
    m0, v0 = np.array(prior["mean"]), np.array(prior["scale_matrix"])
    p0 = np.linalg.inv(v0)
    precision = p0+x.T@x
    vn = np.linalg.inv(precision)
    mn = np.linalg.solve(precision, p0@m0+x.T@y)
    an = prior["shape"]+len(y)/2
    bn = prior["scale"]+(y@y+m0@p0@m0-mn@precision@mn)/2
    scale, covariance = vn*bn/an, None if an <= 1 else vn*bn/(an-1)
    evidence = (-len(y)*np.log(2*np.pi)/2+np.linalg.slogdet(vn)[1]/2-np.linalg.slogdet(v0)[1]/2
                + prior["shape"]*np.log(prior["scale"])-an*np.log(bn)
                + gammaln(an)-gammaln(prior["shape"]))
    critical = t.isf(bundle["alpha"]/2, 2*an)
    limits = np.column_stack((mn-critical*np.sqrt(np.diag(scale)), mn+critical*np.sqrt(np.diag(scale))))
    references = dict(mean=mn, conditional_scale_matrix=vn, shape=an, scale=bn,
                      degrees_of_freedom=2*an, coefficient_scale_matrix=scale,
                      log_marginal_likelihood=evidence, credible_intervals=limits)
    comparisons = {name: _compare(bundle[name], expected, name) for name, expected in references.items()}
    if covariance is None:
        if bundle["coefficient_covariance"] is not None or bundle["variance_mean"] is not None:
            raise AssertionError("nonexistent posterior covariance was fabricated")
    else:
        comparisons["coefficient_covariance"] = _compare(bundle["coefficient_covariance"], covariance, "full coefficient covariance")
        comparisons["variance_mean"] = _compare(bundle["variance_mean"], bn/(an-1), "variance_mean")
    if "contrast" in receipt:
        contrast = receipt["contrast"]
        if "posterior" in contrast:
            target = contrast["posterior"]
            if target != bundle or contrast["posterior_sha256"] != bundle["integrity_sha256"]:
                raise AssertionError("contrast complete target disagrees with the receipt posterior")
            if contrast["terms"] != bundle["terms"]:
                raise AssertionError("contrast weights have a different named coefficient order")
            if _digest({k: v for k, v in contrast.items() if k != "integrity_sha256"}) != contrast["integrity_sha256"]:
                raise AssertionError("contrast digest mismatch")
        w = np.array(contrast["weights"])
        mean, cs = w@mn, np.sqrt(w@scale@w)
        refs = dict(mean=mean, student_t_scale=cs, degrees_of_freedom=2*an, credible_lower=mean-t.isf(contrast["alpha"]/2, 2*an)*cs,
                    credible_upper=mean+t.isf(contrast["alpha"]/2, 2*an)*cs,
                    probability_above_threshold=float(mean>contrast["threshold"]) if cs == 0 else t.cdf((mean-contrast["threshold"])/cs, 2*an))
        comparisons.update({f"contrast.{name}": _compare(contrast[name], value, f"contrast {name}") for name, value in refs.items()})
        if covariance is not None:
            comparisons["contrast.posterior_variance"] = _compare(contrast["posterior_variance"], w@covariance@w, "contrast covariance")
        elif contrast["posterior_variance"] is not None:
            raise AssertionError("nonexistent contrast posterior variance was fabricated")
    if "query_design" in receipt:
        q = np.array(receipt["query_design"])
        prediction = receipt["prediction"]
        comparisons["prediction.degrees_of_freedom"] = _compare(prediction["attrs"]["degrees_of_freedom"], 2*an, "prediction df")
        means, joint = q@mn, q@scale@q.T
        prediction_critical = t.isf((1-prediction["attrs"]["credible_probability"])/2, 2*an)
        refs = dict(mean=means, mean_student_t_scale=np.sqrt(np.diag(joint)),
                    outcome_student_t_scale=np.sqrt(np.diag(joint)+bn/an),
                    mean_credible_lower=means-prediction_critical*np.sqrt(np.diag(joint)),
                    mean_credible_upper=means+prediction_critical*np.sqrt(np.diag(joint)),
                    outcome_predictive_lower=means-prediction_critical*np.sqrt(np.diag(joint)+bn/an),
                    outcome_predictive_upper=means+prediction_critical*np.sqrt(np.diag(joint)+bn/an))
        for name, value in refs.items():
            comparisons[f"prediction.{name}"] = _compare([row[name] for row in prediction["rows"]], value, name)
        factor = np.array(prediction["attrs"]["mean_scale_factor"])
        comparisons["prediction.full_joint_scale"] = _compare(factor@factor.T, joint, "full joint prediction scale")
        comparisons["prediction.outcome_independent_scale"] = _compare(prediction["attrs"]["outcome_independent_scale"], bn/an, "future observation scale")
        if covariance is not None:
            joint_covariance = q@covariance@q.T
            comparisons["prediction.mean_posterior_variance"] = _compare(
                [row["mean_posterior_variance"] for row in prediction["rows"]],
                np.diag(joint_covariance), "conditional mean posterior variance")
            covariance_factor = np.asarray(prediction["attrs"]["mean_covariance_factor"])
            comparisons["prediction.full_joint_covariance"] = _compare(
                covariance_factor@covariance_factor.T, joint_covariance,
                "full cross-query posterior covariance")
            comparisons["prediction.full_outcome_covariance"] = _compare(
                covariance_factor@covariance_factor.T
                + np.eye(len(q))*prediction["attrs"]["outcome_independent_variance"],
                joint_covariance + np.eye(len(q))*bn/(an-1),
                "full cross-query future-outcome covariance")
            if prediction["attrs"]["scale_to_covariance_status"] != "finite":
                raise AssertionError("representable prediction covariance multiplier was suppressed")
        elif (prediction["attrs"]["mean_covariance_factor"] is not None
              or prediction["attrs"]["outcome_independent_variance"] is not None):
            raise AssertionError("nonexistent full posterior predictive moments were fabricated")
    return {"schema_version": "openecon.bayesian_conjugate_oracle.v1", "status": "passed",
            "posterior_sha256": bundle["integrity_sha256"], "rows": len(y), "parameters": len(mn),
            "oracle": "independent NumPy sufficient statistics and SciPy t/gammaln",
            "tolerances": {"relative": RTOL, "reference_unit_absolute": UNIT_ATOL},
            "comparisons": comparisons,
            "claim": "Supplied saved numerical posterior/receipt comparison; source/native/vendor provenance is a separate gate."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(json.loads(args.posterior.read_text()))
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "comparisons": len(report["comparisons"]), "report": str(args.report)}))


if __name__ == "__main__":
    main()
