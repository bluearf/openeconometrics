#!/usr/bin/env python3
"""Development-only NumPy/SciPy oracle for eight native two-stage outputs.

No OpenEconometrics numerical implementation is imported. The reference uses
original physical rows, both declared stage populations, actual likelihood
scores/observed bread, and independent multivariate SRSWOR recursion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


CASES = ("mean", "total", "ratio", "proportion", "regress", "logit", "probit", "poisson")
TABLE_COLUMNS = ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]


def geometry(frame, roles):
    hname, pname = roles.get("strata"), roles["psu"]
    strata_labels = frame[hname].to_numpy() if hname else np.zeros(len(frame), dtype=int)
    weights = np.empty(len(frame), dtype=float)
    strata, saved_strata, saved_psus = [], [], []
    for h in pd.unique(strata_labels):
        hm = np.flatnonzero(strata_labels == h)
        psu_labels = pd.unique(frame[pname].iloc[hm])
        population = int(frame[roles["population_psu"]].iloc[hm[0]])
        f1 = len(psu_labels)/population
        psus, indices = [], []
        for p in psu_labels:
            rows = hm[frame[pname].iloc[hm].to_numpy() == p]
            M = int(frame[roles["population_ssu"]].iloc[rows[0]])
            f2 = len(rows)/M
            weights[rows] = 1/(f1*f2)
            indices.append(len(saved_psus))
            saved_psus.append({"stratum_index": len(strata), "n_ssu": len(rows),
                               "population_ssu": M, "row_positions": rows.tolist()})
            psus.append((rows, f2))
        saved_strata.append({"n_psu": len(psus), "population_psu": population,
                             "psu_indices": indices})
        strata.append((psus, f1))
    return weights, strata, saved_strata, saved_psus


def covariance_components(rows, strata):
    k = rows.shape[1]
    first, second = np.zeros((k, k)), np.zeros((k, k))
    for psus, f1 in strata:
        total = np.array([rows[positions].sum(axis=0) for positions, _ in psus])
        if f1 < 1:
            centered = total-total.mean(axis=0)
            first += (1-f1)*len(psus)/(len(psus)-1)*(centered.T @ centered)
        for positions, f2 in psus:
            if f2 < 1:
                centered = rows[positions]-rows[positions].mean(axis=0)
                second += f1*(1-f2)*len(positions)/(len(positions)-1)*(centered.T @ centered)
    return first, second


def likelihood(family, beta, x, y, w):
    eta = x @ beta
    if family == "regress":
        factor, curvature = y-eta, np.ones(len(y))
        objective = w @ (factor**2/2)
    elif family == "logit":
        p = special.expit(eta)
        factor, curvature = y-p, p*(1-p)
        objective = w @ (np.logaddexp(0, eta)-y*eta)
    elif family == "probit":
        sign = 2*y-1
        logp = special.log_ndtr(sign*eta)
        mills = np.exp(stats.norm.logpdf(eta)-logp)
        factor, curvature = sign*mills, mills*(mills+sign*eta)
        objective = -w @ logp
    else:
        mean = np.exp(eta)
        factor, curvature = y-mean, mean
        objective = w @ (mean-y*eta)
    return objective, (w*factor)[:, None]*x, x.T @ ((w*curvature)[:, None]*x)


def expected_case(frame, roles, kind, specification, shared):
    options = {**shared, **{key: value for key, value in specification.items() if key != "args"}}
    args = specification["args"]
    domain = options.get("domain")
    members = frame[domain].to_numpy(dtype=bool) if domain else np.ones(len(frame), dtype=bool)
    weights, strata, saved_strata, saved_psus = geometry(frame, roles)
    if kind in CASES[:4]:
        names = [args[0]] if isinstance(args[0], str) else list(args[0])
        denominators = list(args[1]) if kind == "ratio" else []
        selected = members & frame[names+denominators].notna().all(axis=1).to_numpy()
        if kind == "proportion":
            categories = options["categories"]
            values = np.column_stack([frame[names[0]].to_numpy() == level for level in categories]).astype(float)
            assert all(isinstance(level, str) for level in categories), "Native dictionary is string-valued."
            labels = [f"{names[0]}[{i}]=str:{level}" for i, level in enumerate(categories)]
        else:
            values = frame[names].fillna(0).to_numpy(dtype=float)
            labels = [f"{y}/{d}" for y, d in zip(names, denominators)] if kind == "ratio" else names
        values[~selected] = 0
        effective = weights*selected
        denominator = frame[denominators].fillna(0).to_numpy(dtype=float) if denominators else np.ones_like(values)
        if kind == "ratio":
            denominator[~selected] = 0
        numerator = effective @ values
        if kind == "total":
            estimate, z = numerator, values
        else:
            normalizer = effective @ denominator
            estimate = numerator/normalizer
            z = selected[:, None]*(values-denominator*estimate)/normalizer
        rows = weights[:, None]*z
        primitives = {"primitive_values": values, "primitive_denominators": denominator,
                      "row_influences": rows}
    else:
        outcome, regressors = args
        selected = members & frame[[outcome, *regressors]].notna().all(axis=1).to_numpy()
        X = frame[regressors].fillna(0).to_numpy(dtype=float)
        intercept = options.get("intercept", True)
        labels = (["_cons"] if intercept else [])+list(regressors)
        if intercept:
            X = np.column_stack((np.ones(len(frame)), X))
        x, y, w = X[selected], frame[outcome].to_numpy(dtype=float)[selected], weights[selected]
        if kind == "regress":
            estimate = np.linalg.solve(x.T @ (w[:, None]*x), x.T @ (w*y))
        else:
            def objective(beta):
                value, scores, _ = likelihood(kind, beta, x, y, w)
                return value/w.sum(), -scores.sum(axis=0)/w.sum()

            fit = optimize.minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS",
                                    options={"gtol": 1e-12, "maxiter": 1000})
            root = optimize.root(lambda beta: likelihood(kind, beta, x, y, w)[1].sum(axis=0),
                                 fit.x, jac=lambda beta: -likelihood(kind, beta, x, y, w)[2], tol=1e-12)
            estimate = root.x
            assert np.max(np.abs(objective(estimate)[1])) < 1e-10, "Independent score root did not converge."
        _, scores, bread = likelihood(kind, estimate, x, y, w)
        weighted_score_rows = np.zeros((len(frame), len(estimate)))
        weighted_score_rows[selected] = scores
        rows = np.linalg.solve(bread, weighted_score_rows.T).T
        # The fitted implementation normalizes active weights to mean one.
        normalization = selected.sum()/w.sum()
        primitives = {"bread": bread*normalization,
                      "normalized_row_scores": weighted_score_rows*normalization,
                      "primitive_X": x, "primitive_y": y}
    first, second = covariance_components(rows, strata)
    return {"estimates": estimate, "covariance": first+second,
            "stage1_covariance": first, "stage2_covariance": second,
            "selected": selected, "members": members, "weights": weights,
            "strata": saved_strata, "psus": saved_psus, "labels": labels,
            "df": len(saved_psus)-len(saved_strata), "alpha": options.get("alpha", .05),
            "null": np.broadcast_to(options.get("null", 0.), len(estimate)), "primitives": primitives}


def numerical(actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{name}: unexpected dimensions."
    assert np.isfinite(actual).all(), f"{name}: nonfinite numerical output."
    np.testing.assert_allclose(actual, expected, rtol=3e-8, atol=2e-10, err_msg=name)
    return float(np.max(np.abs(actual-expected))) if actual.size else 0.


def expected_table(expected):
    estimates, covariance = expected["estimates"], expected["covariance"]
    df, alpha, null = expected["df"], expected["alpha"], expected["null"]
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = []
    for i, value in enumerate(estimates):
        statistic = (value-null[i])/se[i] if df and se[i] else None
        p = 2*stats.t.sf(abs(statistic), df) if statistic is not None else None
        width = stats.t.ppf(1-alpha/2, df)*se[i] if df else (0. if se[i] == 0 else None)
        result.append([value, se[i], statistic, p,
                       None if width is None else value-width, None if width is None else value+width])
    return result


def verify_payload(payload):
    assert set(payload) >= {"states", "poststates", "oracle_inputs"}
    states, tables, inputs = (payload[name] for name in ("states", "poststates", "oracle_inputs"))
    assert set(states) == set(CASES) and set(tables) == set(CASES)
    assert set(inputs["cases"]) == set(CASES)
    frame = pd.DataFrame(inputs["frame"])
    roles, shared = inputs["design_roles"], inputs["options"]
    assert len(frame) == 48 and roles == {"psu": "p", "ssu": "j", "strata": "h",
                                         "population_psu": "N", "population_ssu": "M"}
    assert shared["domain"] == "domain" and shared["missing"] == "drop"
    checks = {}
    for kind in CASES:
        expected = expected_case(frame, roles, kind, inputs["cases"][kind], shared)
        state, table = states[kind], tables[kind]
        assert state["method"] == "two-stage-taylor"
        assert state["design"]["schema_version"] == "survey-two-stage-design-v1"
        assert state["design"]["stages"] == 2 and state["design"]["sampling"] == "sequential-srswor"
        assert all(state["design"][name] == value for name, value in roles.items())
        estimates = state["estimates" if kind in CASES[:4] else "coefficients"]
        assert state["labels"] == expected["labels"] and table["index"] == expected["labels"]
        assert state["df"] == expected["df"] and state["alpha"] == expected["alpha"]
        numerical(state["null"], expected["null"], kind+" null")
        estimate_error = numerical(estimates, expected["estimates"], kind+" estimates")
        covariance_error = numerical(state["covariance"], expected["covariance"], kind+" full covariance")
        validation, metadata = state["design"]["validation"], state["metadata"]
        assert validation["strata"] == expected["strata"] and validation["psus"] == expected["psus"]
        assert validation["nobs"] == len(frame) and validation["design_df"] == expected["df"]
        numerical(validation["sum_weights"], expected["weights"].sum(), kind+" derived weight sum")
        assert metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
        assert metadata["out_of_domain_positions"] == np.flatnonzero(~expected["members"]).tolist()
        assert metadata["outcome_exclusions"] == np.flatnonzero(expected["members"] & ~expected["selected"]).tolist()
        assert metadata["n_used"] == int(expected["selected"].sum())
        assert metadata["precision"] == "float64" and metadata["device"] == "cpu"
        assert metadata["multistage_support"] is True
        for key in ("stage1_covariance", "stage2_covariance"):
            numerical(metadata[key], expected[key], kind+" "+key)
        if kind in CASES[:4]:
            for key, values in expected["primitives"].items():
                numerical(metadata[key], values, kind+" "+key)
        else:
            for key in ("primitive_X", "primitive_y"):
                numerical(metadata[key], expected["primitives"][key], kind+" "+key)
            numerical(metadata["bread"], expected["primitives"]["bread"], kind+" observed bread")
            numerical(metadata["row_scores"], expected["primitives"]["normalized_row_scores"], kind+" row scores")
        assert table["columns"] == TABLE_COLUMNS
        reference_table = expected_table(expected)
        assert len(table["data"]) == len(reference_table)
        display_error = 0.
        for i, (actual_row, expected_row) in enumerate(zip(table["data"], reference_table)):
            assert len(actual_row) == len(TABLE_COLUMNS)
            for j, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
                name = f"{kind} physical row {i} {TABLE_COLUMNS[j]}"
                if expected_value is None:
                    assert actual_value is None, name+": undefined statistic must remain null."
                else:
                    display_error = max(display_error, numerical(actual_value, expected_value, name))
        attrs = table["attrs"]
        assert attrs["survey_state"] == state and attrs["df"] == expected["df"]
        numerical(attrs["covariance_matrix"], expected["covariance"], kind+" displayed covariance")
        assert attrs["alpha"] == expected["alpha"]
        checks[kind] = {"labels": expected["labels"], "n_used": int(expected["selected"].sum()),
                        "df": expected["df"], "estimate_max_abs_error": estimate_error,
                        "covariance_max_abs_error": covariance_error, "display_max_abs_error": display_error,
                        "both_stage_components_verified": True, "full_joint_covariance_verified": True,
                        "physical_display_rows_verified": True}
    return {"status": "passed", "independent_numpy_scipy": True, "production_numerical_imports": False,
            "case_count": len(checks), "display_table_count": len(checks), "physical_rows": len(frame),
            "sampling": "two sequential SRSWOR stages", "variance": "recursive two-stage Taylor",
            "df": "complete first-stage PSUs minus strata, reference convention only",
            "exact_nonlinear_variance_or_coverage_claim": False, "cases": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    raw = args.payload.read_bytes()
    receipt = verify_payload(json.loads(raw))
    receipt["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": receipt["status"], "cases": receipt["case_count"],
                      "tables": receipt["display_table_count"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
