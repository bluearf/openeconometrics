#!/usr/bin/env python3
"""Development-only independent PSU coefficient-replication acceptance oracle.

No OpenEcon statistical implementation is imported. NumPy normal equations and
SciPy logistic likelihood/score roots are evaluated from saved primitive inputs.
The optional execution JSON is the actual native displayed-table receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


METHODS = ("brr", "fay", "jackknife", "bootstrap")
CASES = tuple(f"{family}_{method}" for family in ("linear", "logit") for method in METHODS)
LABELS = ["_cons", "x", "z"]
SIGNS = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
BOOTSTRAP_FACTORS = np.array([
    [1.3, .7, 1.1, .9, .8, 1.2], [.7, 1.3, .9, 1.1, 1.2, .8],
    [1.15, .85, .75, 1.25, 1.05, .95], [.85, 1.15, 1.25, .75, .95, 1.05],
    [1.4, .6, .8, 1.2, 1.1, .9], [.6, 1.4, 1.2, .8, .9, 1.1],
    [1.05, .95, 1.3, .7, .7, 1.3], [.95, 1.05, .7, 1.3, 1.3, .7],
])
RSCALES = [1., 1., 1.2, 1.2, .8, .8, 1.1, 1.1]


def primitive_frame():
    rows = []
    xb = [-1., -.6, -.1, .4, .9, 1.4, -1.4, 1.8]
    zb = [.4, -.5, .8, -.3, .6, -.7, -.9, 1.1]
    noise = [-.6, .4, .8, -.2, .3, -.7, .5, -.4]
    patterns = [
        [0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1],
    ]
    for h in range(3):
        for p in range(2):
            for j in range(8):
                x, z = xb[j] + .12*h + .05*p, zb[j] + .15*p - .1*h
                rows.append({
                    "h": h, "p": p, "w": 1. + .2*h + .15*p + .07*j,
                    "x": x, "z": z,
                    "y": 1.2 + .65*x - .4*z + (h-1)*.25 + p*.3
                    + noise[j]*(.15+.05*h) + .1*h*x,
                    "b": patterns[2*h+p][j], "domain": 1,
                })
    return pd.DataFrame(rows)


def group_geometry(frame):
    pairs = list(dict.fromkeys(zip(frame.h.tolist(), frame.p.tolist())))
    groups = np.array([pairs.index(pair) for pair in zip(frame.h, frame.p)])
    strata = [np.array([i for i, pair in enumerate(pairs) if pair[0] == h])
              for h in dict.fromkeys(frame.h)]
    return groups, strata


def bootstrap_weights(frame):
    groups, _ = group_geometry(frame)
    return pd.DataFrame(
        (BOOTSTRAP_FACTORS[:, groups] * frame.w.to_numpy()).T,
        columns=[f"whole-psu-{i}" for i in range(8)],
    )


def default_options(method):
    if method == "bootstrap":
        return {"centering": "replicate_mean", "scale": 1/8, "rscales": RSCALES,
                "justification": "Declared whole-PSU synthetic perturbations; no row resampling."}
    if method == "jackknife":
        return {"centering": "stratum_mean"}
    if method == "fay":
        return {"rho": .5, "replicates": 4}
    return {"replicates": 4}


def replica_plan(frame, method, *, rho=.5, centering="original", supplied=None,
                 scale=None, rscales=None, df=None, **_):
    """Explicit Hadamard, whole-PSU delete and declared bootstrap construction."""
    groups, strata = group_geometry(frame)
    weights = frame.w.to_numpy(dtype=float)
    design_df = sum(len(group)-1 for group in strata)
    rstrata = None
    if method in {"brr", "fay"}:
        rho = 0. if method == "brr" else (.5 if rho is None else rho)
        factors = np.ones((4, len(strata)*2))
        for h, pair in enumerate(strata):
            factors[:, pair[0]] = 1 + (1-rho)*SIGNS[:, h]
            factors[:, pair[1]] = 1 - (1-rho)*SIGNS[:, h]
        rw = factors[:, groups] * weights
        multipliers = np.full(4, 1/(4*(1-rho)**2))
        ids = [f"sylvester-{i}" for i in range(4)]
    elif method == "jackknife":
        rw, ids, multipliers, rstrata = [], [], [], []
        for h, psus in enumerate(strata):
            m = len(psus)
            hm = np.isin(groups, psus)
            population = float(frame.loc[hm, "N"].iloc[0]) if "N" in frame else np.inf
            fpc = 1-m/population
            if not fpc:
                continue
            for psu in psus:
                factors = np.ones(len(frame))
                factors[hm] = m/(m-1)
                factors[groups == psu] = 0
                rw.append(factors*weights)
                ids.append(f"stratum-{h}-delete-psu-{psu}")
                multipliers.append(fpc*(m-1)/m)
                rstrata.append(h)
        rw, multipliers = np.asarray(rw), np.asarray(multipliers)
    else:
        table = bootstrap_weights(frame) if supplied is None else supplied
        rw = table.to_numpy(dtype=float).T
        ids = table.columns.tolist()
        multipliers = float(scale) * np.asarray(rscales, dtype=float)
    if supplied is not None and method != "bootstrap":
        rw, ids = supplied.to_numpy(dtype=float).T, supplied.columns.tolist()
    return rw, ids, multipliers, rstrata, (min(design_df, len(ids)-1) if df is None else df)


def weighted_fit(x, y, w, family):
    effective = w > 0
    x, y, w = x[effective], y[effective], w[effective]
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Oracle replica lacks full design rank.")
    if family == "linear":
        return np.linalg.solve(x.T @ (w[:, None]*x), x.T @ (w*y))
    def objective(beta):
        eta = x @ beta
        return np.dot(w, np.logaddexp(0, eta)-y*eta)/w.sum(), x.T @ (w*(special.expit(eta)-y))/w.sum()
    def curvature(beta):
        mu = special.expit(x @ beta)
        return x.T @ ((w*mu*(1-mu))[:, None]*x)
    minimized = optimize.minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS",
                                  options={"gtol": 2e-12, "maxiter": 500})
    polished = optimize.root(lambda beta: objective(beta)[1]*w.sum(), minimized.x,
                            jac=curvature, tol=1e-11)
    if np.linalg.norm(objective(polished.x)[1], np.inf) > 2e-10:
        raise AssertionError("Independent logistic score did not converge.")
    if np.linalg.eigvalsh(curvature(polished.x)).min() <= 1e-8:
        raise AssertionError("Independent logistic fit is not finite and identified.")
    return polished.x


def oracle(frame, family, method, *, intercept=True, domain=None, supplied=None, **options):
    columns = ["x", "z"]
    outcome = "y" if family == "linear" else "b"
    selected = frame[[outcome, *columns]].notna().all(axis=1).to_numpy()
    if domain is not None:
        selected &= frame[domain].to_numpy(dtype=bool)
    x = frame[columns].fillna(0).to_numpy(dtype=float)
    if intercept:
        x = np.column_stack((np.ones(len(frame)), x))
    y = frame[outcome].fillna(0).to_numpy(dtype=float)
    beta = weighted_fit(x[selected], y[selected], frame.w.to_numpy()[selected], family)
    rw, ids, multipliers, rstrata, df = replica_plan(frame, method, supplied=supplied, **options)
    replicas = np.array([weighted_fit(x[selected], y[selected], w[selected], family) for w in rw])
    center = options.get("centering", "original")
    if center == "original":
        centers = np.broadcast_to(beta, replicas.shape)
    elif center == "replicate_mean":
        centers = np.broadcast_to(replicas.mean(0), replicas.shape)
    else:
        centers = np.empty_like(replicas)
        for h in set(rstrata):
            membership = np.asarray(rstrata) == h
            centers[membership] = replicas[membership].mean(0)
    differences = replicas-centers
    covariance = np.einsum("r,ri,rj->ij", multipliers, differences, differences)
    return {"coefficients": beta, "covariance": covariance, "replicas": replicas,
            "ids": ids, "multipliers": multipliers, "df": df, "selected": selected,
            "weights": rw, "replicate_strata": rstrata}


def inference(beta, covariance, df, *, alpha=.05, null=0):
    beta, covariance = np.asarray(beta), np.asarray(covariance)
    se = np.sqrt(np.maximum(covariance.diagonal(), 0))
    statistic = (beta-null)/se
    critical = stats.t.ppf(1-alpha/2, df)
    return np.column_stack((beta, se, statistic, 2*stats.t.sf(abs(statistic), df),
                            beta-critical*se, beta+critical*se))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected, message):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, message+": shape differs")
    require(np.isfinite(actual).all(), message+": incomplete numeric state")
    np.testing.assert_allclose(actual, expected, atol=4e-9, rtol=5e-8, err_msg=message)
    return float(np.max(abs(actual-expected), initial=0))


def verify(states, inputs, poststates, execution=None):
    require(set(states) == set(CASES), "Eight complete model states are required.")
    require(set(poststates) == set(CASES), "Eight complete postestimation states are required.")
    frame = pd.DataFrame(inputs["frame"])
    require(len(frame) == 48 and inputs["regressors"] == ["x", "z"], "Primitive fit geometry differs.")
    supplied = pd.DataFrame(inputs["bootstrap_weights"]["data"], columns=inputs["bootstrap_weights"]["columns"])
    evaluation = pd.DataFrame(inputs["evaluation"]["data"], columns=["x", "z"])
    xe = np.column_stack((np.ones(len(evaluation)), evaluation.to_numpy()))
    differences = {}
    for case in CASES:
        family, method = case.split("_", 1)
        options = inputs["options"][method]
        reference = oracle(frame, family, method, supplied=supplied if method == "bootstrap" else None, **options)
        state, tables = states[case], poststates[case]
        beta, covariance, df = reference["coefficients"], reference["covariance"], reference["df"]
        require(state["labels"] == LABELS and state["family"] == family, case+": parameter order/family differs.")
        require(state["df"] == df and state["metadata"]["sample_positions"] == list(range(48)), case+": complete sample/df differs.")
        require(state["metadata"]["n_used"] == 48 and state["metadata"]["n_design"] == 48, case+": sample count differs.")
        require(state["metadata"]["n_domain"] == 48 and state["metadata"]["outcome_exclusions"] == []
                and state["metadata"]["out_of_domain_positions"] == [], case+": sample exclusions differ.")
        require(state["metadata"]["parameter_order"] == LABELS and state["regressors"] == ["x", "z"]
                and state["intercept"] is True, case+": saved coefficient roles differ.")
        require(state["metadata"]["device"] == "cpu" and state["metadata"]["precision"] == "float64", case+": numerical contract differs.")
        require(state["alpha"] == .05 and state["null"] == [0., 0., 0.] and state["metadata"]["stata_parity_validated"] is False,
                case+": saved inference declaration differs.")
        record = state["metadata"]["replication"]
        require(record["method"] == method, case+": saved replication method differs.")
        require(record["replicate_ids"] == reference["ids"] and record["failed_replicates"] == [], case+": incomplete replication IDs.")
        require(record["replicate_count"] == len(reference["ids"]), case+": replicate count differs.")
        require(record["centering"] == options.get("centering", "original"), case+": centering differs.")
        convergence = record["replicate_convergence"]
        require(len(convergence) == len(reference["ids"]), case+": incomplete fit diagnostics.")
        for diagnostic, identity, weights in zip(convergence, reference["ids"], reference["weights"]):
            positions = np.flatnonzero(weights > 0).tolist()
            require(diagnostic["replicate_id"] == identity and diagnostic["converged"] is True
                    and diagnostic["n_used"] == len(positions) and diagnostic["sample_positions"] == positions,
                    case+": replica support/diagnostics differ.")
        if method in {"brr", "fay"}:
            compare(record["balanced_signs"], SIGNS, case+" retained balanced signs")
            require(record["fay_rho"] == (0. if method == "brr" else options["rho"]), case+": rho differs.")
        for procedure in ("coefficients", "predict", "margins", "lincom", "test"):
            require(tables[procedure]["attrs"]["survey_regression_state"] == state,
                    case+": helper retained coefficient state differs.")
        differences[case+"/coefficients"] = compare(state["coefficients"], beta, case+" coefficients")
        differences[case+"/covariance"] = compare(state["covariance"], covariance, case+" covariance")
        differences[case+"/replicas"] = compare(record["replicate_estimates"], reference["replicas"], case+" replicas")
        compare(record["variance_multipliers"], reference["multipliers"], case+" variance multipliers")
        expected_table = inference(beta, covariance, df)
        require(tables["coefficients"]["index"] == LABELS, case+": coefficient row labels differ.")
        require(tables["coefficients"]["columns"] == ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"], case+": coefficient columns differ.")
        differences[case+"/display"] = compare(tables["coefficients"]["data"], expected_table, case+" displayed inference")
        eta = xe @ beta
        mu = eta if family == "linear" else special.expit(eta)
        jac = xe if family == "linear" else (mu*(1-mu))[:, None]*xe
        compare(tables["predict"]["data"], inference(mu, jac@covariance@jac.T, df), case+" saved prediction")
        compare(tables["predict"]["attrs"]["covariance_matrix"], jac@covariance@jac.T, case+" prediction full covariance")
        require(tables["predict"]["index"] == ["low", "middle", "high"], case+": prediction index differs.")
        require(tables["predict"]["attrs"]["physical_positions"] == [0, 1, 2], case+": prediction physical rows differ.")
        contrast = np.asarray(inputs["contrast"])
        estimate = np.array([contrast@beta])
        contrast_cov = np.array([[contrast@covariance@contrast]])
        compare(tables["lincom"]["data"], inference(estimate, contrast_cov, df), case+" saved contrast")
        compare(tables["lincom"]["attrs"]["covariance_matrix"], contrast_cov, case+" contrast covariance")
        restrictions = np.array(inputs["restrictions"])
        effect = restrictions@beta
        wald = float(effect@np.linalg.solve(restrictions@covariance@restrictions.T, effect))
        q = len(restrictions)
        f = (df-q+1)*wald/(q*df)
        joint = tables["test"]
        values = dict(zip(joint["columns"], joint["data"][0]))
        compare([values["statistic"], values["p_value"], values["df_num"], values["df_den"], values["wald_chi2"]],
                [f, stats.f.sf(f, q, df-q+1), q, df-q+1, wald], case+" saved joint adjusted F")
        # Average response and its exact coefficient gradient, with caller weights.
        xm = np.column_stack((np.ones(len(frame)), frame[["x", "z"]].to_numpy()))
        wm = frame.w.to_numpy()/frame.w.sum()
        response = xm@beta if family == "linear" else special.expit(xm@beta)
        gradient = wm@xm if family == "linear" else (wm*response*(1-response))@xm
        mcov = np.array([[gradient@covariance@gradient]])
        compare(tables["margins"]["data"], inference([wm@response], mcov, df), case+" average margin")
        compare(tables["margins"]["attrs"]["covariance_matrix"], mcov, case+" average margin covariance")
    if execution is not None:
        require(execution.get("status") == "ok" and execution.get("error") is None, "Native execution failed.")
        outputs = execution["outputs"]
        require(len(outputs) == 8, "Exactly eight displayed native coefficient tables are required.")
        for case, output in zip(CASES, outputs):
            require(output["type"] == "table", case+": native output is not a table.")
            data = output["data"]
            require(data["index"] == [[label] for label in LABELS], case+": native row labels differ.")
            require(data["columns"] == poststates[case]["coefficients"]["columns"], case+": native columns differ.")
            require(data["index_names"] == [None] and data["total_rows"] == 3 and data["total_columns"] == 6, case+": native table geometry differs.")
            require(bool(output.get("latex") or output.get("full_latex")), case+": missing LaTeX export.")
            differences[case+"/native"] = compare(data["rows"], poststates[case]["coefficients"]["data"], case+" actual native displayed table")
    return {"cases": 8, "rows": len(frame), "independent_oracle": "NumPy normal equations and SciPy logistic likelihood score root",
            "native_display_verified": execution is not None, "full_covariance_verified": True,
            "maximum_absolute_differences": differences, "stata_parity_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--poststates", required=True, type=Path)
    parser.add_argument("--native-execution", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = verify(json.loads(args.states.read_text()), json.loads(args.inputs.read_text()),
                     json.loads(args.poststates.read_text()),
                     json.loads(args.native_execution.read_text()) if args.native_execution else None)
    receipt["saved_states_sha256"] = hashlib.sha256(args.states.read_bytes()).hexdigest()
    rendered = json.dumps(receipt, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
