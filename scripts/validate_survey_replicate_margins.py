#!/usr/bin/env python3
"""Development-only complete empirical survey-margin replication oracle.

No OpenEcon statistical implementation is imported. SciPy log-Phi probit and exponential Poisson
likelihood/score roots are evaluated from saved primitive inputs.
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
CASES = tuple(f"{method}_{target}" for method in METHODS for target in ("mean", "ame"))
FAMILIES = {"brr_mean": "logit", "brr_ame": "probit", "fay_mean": "poisson", "fay_ame": "logit",
            "jackknife_mean": "probit", "jackknife_ame": "poisson", "bootstrap_mean": "logit", "bootstrap_ame": "poisson"}
PROFILES = {"low": {"x": -.8}, "middle": {"x": .3}, "high": {"x": 1.2}}
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
    patterns = [
        [0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1],
    ]
    counts = [
        [0, 2, 1, 0, 3, 1, 0, 2], [2, 0, 1, 3, 0, 2, 1, 0],
        [0, 1, 2, 1, 0, 3, 2, 1], [1, 2, 0, 2, 1, 0, 3, 1],
        [0, 2, 1, 3, 2, 0, 1, 2], [2, 1, 0, 1, 3, 2, 0, 1],
    ]
    for h in range(3):
        for p in range(2):
            for j in range(8):
                x, z = xb[j] + .12*h + .05*p, zb[j] + .15*p - .1*h
                rows.append({
                    "h": h, "p": p, "w": 1. + .2*h + .15*p + .07*j,
                    "x": x, "z": z,
                    "b": patterns[2*h+p][j], "c": counts[2*h+p][j], "domain": 1,
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


def likelihood_parts(beta, x, y, w, family):
    """Independent negative log likelihood, weighted score and observed bread."""
    eta = x@beta
    if family == "logit":
        probability = special.expit(eta)
        score_factor = y-probability
        curvature = probability*(1-probability)
        objective = w@(np.logaddexp(0, eta)-y*eta)
    elif family == "probit":
        sign = 2*y-1
        log_probability = special.log_ndtr(sign*eta)
        mills = np.exp(stats.norm.logpdf(eta)-log_probability)
        score_factor = sign*mills
        curvature = mills*(mills+sign*eta)
        objective = -w@log_probability
    else:
        mu = np.exp(eta)
        log_y = np.log(y, out=np.zeros_like(y, dtype=float), where=y > 0)
        objective = w@(mu-y+y*log_y-y*eta)
        score_factor, curvature = y-mu, mu
    score = x.T@(w*score_factor)
    bread = x.T@((w*curvature)[:, None]*x)
    return float(objective), score, bread


def weighted_fit(x, y, w, family):
    x, y, w = (np.asarray(value, dtype=float) for value in (x, y, w))
    effective = w > 0
    x, y, w = x[effective], y[effective], w[effective]
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Oracle replica lacks full design rank.")
    def objective(beta):
        value, score, _ = likelihood_parts(beta, x, y, w, family)
        return value/w.sum(), -score/w.sum()
    minimized = optimize.minimize(objective, np.zeros(x.shape[1]), jac=True, method="BFGS",
                                  options={"gtol": 2e-12, "maxiter": 500})
    polished = optimize.root(lambda beta: -likelihood_parts(beta, x, y, w, family)[1], minimized.x,
                            jac=lambda beta: likelihood_parts(beta, x, y, w, family)[2], tol=1e-11)
    if np.linalg.norm(objective(polished.x)[1], np.inf) > 2e-10:
        raise AssertionError("Independent weighted likelihood score did not converge.")
    if np.linalg.eigvalsh(likelihood_parts(polished.x, x, y, w, family)[2]).min() <= 1e-8:
        raise AssertionError("Independent likelihood is not finite and identified.")
    return polished.x


def response(family, eta):
    if family == "logit":
        return special.expit(eta)
    return special.ndtr(eta) if family == "probit" else np.exp(eta)


def response_derivative(family, eta):
    if family == "logit":
        p = special.expit(eta)
        return p*(1-p)
    return stats.norm.pdf(eta) if family == "probit" else np.exp(eta)


def response_second(family, eta):
    if family == "logit":
        p = special.expit(eta)
        return p*(1-p)*(1-2*p)
    return -eta*stats.norm.pdf(eta) if family == "probit" else np.exp(eta)


def evaluate(beta, x, w, family, target, *, regressors=("x", "z"), intercept=True,
             profiles=None, variables=None):
    """Evaluate the complete target under these actual weights, never fixed w0."""
    normalized = w/w.sum()
    if target == "mean":
        estimates, gradients = [], []
        for at in profiles.values():
            design = x.copy()
            for name, value in at.items():
                design[:, regressors.index(name)+int(intercept)] = value
            eta = design@beta
            estimates.append(normalized@response(family, eta))
            gradients.append((normalized*response_derivative(family, eta))@design)
        return np.asarray(estimates), np.asarray(gradients)
    eta = x@beta
    first = normalized@response_derivative(family, eta)
    second = (normalized*response_second(family, eta))@x
    positions = [regressors.index(name)+int(intercept) for name in variables]
    estimates = beta[positions]*first
    gradients = beta[positions, None]*second+first*np.eye(len(beta))[positions]
    return estimates, gradients


def target_covariance(values, original, multipliers, centering, strata=None):
    if centering == "original":
        difference = values-original
    elif centering == "replicate_mean":
        difference = values-values.mean(0)
    else:
        difference = np.empty_like(values)
        groups = np.asarray(strata)
        for group in set(groups):
            selected = groups == group
            difference[selected] = values[selected]-values[selected].mean(0)
    return np.einsum("r,ri,rj->ij", multipliers, difference, difference)


def oracle(frame, family, method, *, target="mean", profiles=None, variables=None,
           intercept=True, domain=None, supplied=None, **options):
    regressors = ["x", "z"]
    profiles = PROFILES if profiles is None and target == "mean" else profiles
    variables = ["x", "z"] if variables is None and target == "ame" else variables
    outcome = "c" if family == "poisson" else "b"
    selected = frame[[outcome, *regressors]].notna().all(axis=1).to_numpy()
    if domain is not None:
        selected &= frame[domain].to_numpy(dtype=bool)
    x = frame[regressors].fillna(0).to_numpy(dtype=float)
    if intercept:
        x = np.column_stack((np.ones(len(frame)), x))
    y = frame[outcome].to_numpy(dtype=float, na_value=0.)
    base_weights = frame.w.to_numpy(dtype=float)
    rw, ids, multipliers, strata, df = replica_plan(frame, method, supplied=supplied, **options)
    beta = weighted_fit(x[selected], y[selected], base_weights[selected], family)
    coefficient_replicas = np.array([
        weighted_fit(x[selected], y[selected], weights[selected], family) for weights in rw
    ])
    evaluation = {"profiles": profiles, "variables": variables, "intercept": intercept}
    estimates, gradient = evaluate(beta, x[selected], base_weights[selected], family, target, **evaluation)
    replicas = np.array([
        evaluate(coef, x[selected], weights[selected], family, target, **evaluation)[0]
        for coef, weights in zip(coefficient_replicas, rw)
    ])
    fixed_replicas = np.array([
        evaluate(coef, x[selected], base_weights[selected], family, target, **evaluation)[0]
        for coef in coefficient_replicas
    ])
    centering = options.get("centering", "original")
    covariance = target_covariance(replicas, estimates, multipliers, centering, strata)
    fixed_covariance = target_covariance(fixed_replicas, estimates, multipliers, centering, strata)
    coefficient_covariance = target_covariance(coefficient_replicas, beta, multipliers, centering, strata)
    return {"coefficients": beta, "coefficient_replicas": coefficient_replicas,
            "coefficient_covariance": coefficient_covariance,
            "estimates": estimates, "covariance": covariance, "replicas": replicas,
            "fixed_covariance": fixed_covariance, "delta_covariance": gradient@coefficient_covariance@gradient.T,
            "ids": ids, "multipliers": multipliers, "df": df, "selected": selected,
            "weights": rw, "replicate_strata": strata,
            "labels": list(profiles) if target == "mean" else ["AME:"+name for name in variables]}


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


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def fixture_design_sha(frame):
    """Independently encode this declared fixture's integer strata/PSU roles."""
    value = hashlib.sha256()
    value.update(json.dumps([(role, str(frame[role].dtype)) for role in ("w", "p", "h")],
                            ensure_ascii=False).encode())
    for row in frame.itertuples(index=False):
        encoded = [["int", int(row.h)], ["int", int(row.p)], float(row.w).hex(), None]
        value.update(json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode())
        value.update(b"\n")
    return value.hexdigest()


def verify_primitive(state, frame, reference, case, options, fit_options):
    """Bind the displayed saved record to independently declared physical inputs."""
    source, metadata = state["source_result"], state["source_result"]["metadata"]
    record, primitive = metadata["replication"], state["metadata"]["primitive"]
    groups, strata = group_geometry(frame)
    first = [int(np.flatnonzero(groups == group)[0]) for group in range(6)]
    x = np.column_stack([np.ones(48), frame[["x", "z"]].to_numpy(dtype=float)])
    y = frame["c" if source["family"] == "poisson" else "b"].to_numpy(dtype=float)
    base = frame.w.to_numpy(dtype=float)
    validation = source["design"]["validation"]
    require(source["regressors"] == ["x", "z"] and source["intercept"] is True
            and source["outcome"] == ("c" if source["family"] == "poisson" else "b"),
            case+": fitted response/regressor declaration differs.")
    require({key: source["design"][key] for key in ("weights", "psu", "strata", "fpc", "stages")}
            == {"weights": "w", "psu": "p", "strata": "h", "fpc": None, "stages": 1},
            case+": design role declaration differs.")
    require([validation[key] for key in ("nobs", "n_psu", "n_strata", "design_df")] == [48, 6, 3, 3]
            and validation["sum_weights"] == float(base.sum()), case+": complete design geometry differs.")
    require(validation["strata"] == [{"nobs": 16, "n_psu": 2, "population_psu": None, "certainty": False}]*3,
            case+": stratum/FPC geometry differs.")
    require(metadata["stratum_psu_indices"] == [group.tolist() for group in strata]
            and metadata["fpc_multipliers"] == [1., 1., 1.]
            and [metadata[key] for key in ("n_design", "n_domain", "n_used")] == [48, 48, 48]
            and metadata["outcome_exclusions"] == [] and metadata["out_of_domain_positions"] == []
            and metadata["parameter_order"] == LABELS, case+": sample/PSU/FPC alignment differs.")
    require(primitive["sample_positions"] == list(range(48)) and primitive["groups"] == groups.tolist(),
            case+": physical primitive support differs.")
    compare(primitive["X"], x, case+" primitive covariates")
    compare(primitive["y"], y, case+" primitive response")
    compare(primitive["base_weights"], base, case+" primitive base weights")
    compare(primitive["replicate_weights"], reference["weights"], case+" actual complete replica weights")
    compare(primitive["psu_factors"], reference["weights"][:, first]/base[first], case+" whole-PSU factors")
    require(record["replicate_count"] == len(reference["weights"])
            and record["centering"] == options.get("centering", "original")
            and record["df_explicit"] is False, case+": replication convention differs.")
    for fit in [metadata["convergence"], *record["replicate_convergence"]]:
        require(fit["converged"] is True and fit["tolerance"] == fit_options["tolerance"]
                and fit["max_iter"] == fit_options["max_iter"], case+": actual convergence declaration differs.")
    for fit, weight in zip(record["replicate_convergence"], reference["weights"]):
        positions = np.flatnonzero(weight > 0).tolist()
        require(fit["n_used"] == len(positions) and fit["sample_positions"] == positions,
                case+": actual replica weighted support differs.")
    require(validation["design_input_sha256"] == fixture_design_sha(frame), case+": physical design hash differs.")
    sample_hash = canonical_sha([validation["design_input_sha256"], [True]*48,
                                 np.column_stack([y, frame[["x", "z"]].to_numpy()]).tolist(),
                                 np.ones((48, 3)).tolist()])
    require(primitive["sample_input_sha256"] == metadata["sample_input_sha256"] == sample_hash,
            case+": admitted primitive sample hash differs.")
    require(primitive["replicate_weight_sha256"] == record["replicate_weight_sha256"]
            == canonical_sha(primitive["replicate_weights"]), case+": actual replica weight hash differs.")
    require(source["integrity_sha256"] == canonical_sha({key: value for key, value in source.items()
                                                         if key != "integrity_sha256"})
            and state["integrity_sha256"] == canonical_sha({key: value for key, value in state.items()
                                                           if key != "integrity_sha256"}),
            case+": saved source/target integrity digest differs.")


def verify(states, inputs, poststates, execution=None):
    require(set(states) == set(CASES), "Eight complete target states are required.")
    require(set(poststates) == set(CASES), "Eight complete target tables are required.")
    frame = pd.DataFrame(inputs["frame"])
    require(len(frame) == 48 and inputs["regressors"] == ["x", "z"], "Primitive fit geometry differs.")
    supplied = pd.DataFrame(inputs["bootstrap_weights"]["data"], columns=inputs["bootstrap_weights"]["columns"])
    differences = {}
    for case in CASES:
        spec = inputs["cases"][case]
        family, target, method = spec["family"], spec["target"], spec["method"]
        require(family == FAMILIES[case] and case == method+"_"+target, case+": method/target/family identity differs.")
        # JSON objects are unordered. The fixture's explicit declared target
        # sequence is low/middle/high, rather than the parsed profile key order.
        require(spec.get("profiles") == (PROFILES if target == "mean" else None)
                and spec.get("variables") == (["x", "z"] if target == "ame" else None),
                case+": declared focal profiles/AME variables differ.")
        options = inputs["options"][method]
        reference = oracle(frame, family, method, target=target,
                           profiles=PROFILES if target == "mean" else None, variables=spec.get("variables"),
                           supplied=supplied if method == "bootstrap" else None, **options)
        state, table = states[case], poststates[case]
        source = state["source_result"]
        require(source["family"] == family and source["labels"] == LABELS, case+": source coefficient roles differ.")
        require(state["target"] == target and state["labels"] == reference["labels"], case+": target labels differ.")
        require(state["profiles"] == (spec.get("profiles") or {})
                and state["variables"] == (spec.get("variables") or []), case+": target declaration differs.")
        require(source["df"] == reference["df"] and source["metadata"]["sample_positions"] == list(range(48)), case+": source df/sample differs.")
        record = source["metadata"]["replication"]
        verify_primitive(state, frame, reference, case, options, inputs["fit_options"])
        require(record["method"] == method and record["replicate_ids"] == reference["ids"]
                and record["failed_replicates"] == [], case+": complete replica plan differs.")
        compare(record["variance_multipliers"], reference["multipliers"], case+" variance multipliers")
        differences[case+"/source_coefficients"] = compare(source["coefficients"], reference["coefficients"], case+" source coefficients")
        differences[case+"/source_replica_coefficients"] = compare(record["replicate_estimates"], reference["coefficient_replicas"], case+" source replica coefficients")
        differences[case+"/source_covariance"] = compare(source["covariance"], reference["coefficient_covariance"], case+" source coefficient covariance")
        differences[case+"/estimates"] = compare(state["estimates"], reference["estimates"], case+" estimates")
        differences[case+"/replicas"] = compare(state["replicate_estimates"], reference["replicas"], case+" complete empirical replica targets")
        differences[case+"/covariance"] = compare(state["covariance"], reference["covariance"], case+" full target covariance")
        require(table["index"] == reference["labels"], case+": displayed target row labels differ.")
        require(table["columns"] == ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"], case+": target inference columns differ.")
        differences[case+"/display"] = compare(table["data"], inference(reference["estimates"], reference["covariance"], reference["df"]), case+" target design-t inference")
        compare(table["attrs"]["covariance_matrix"], reference["covariance"], case+" displayed full covariance")
        require(table["attrs"]["survey_replicate_margins_state"] == state
                and table["attrs"]["empirical_covariate_uncertainty"] is True
                and table["attrs"]["coefficient_uncertainty"] is True
                and table["attrs"]["target"] == target
                and table["attrs"]["method"] == f"single-stage {method} full empirical-target replication"
                and table["attrs"]["design_df"] == reference["df"] and table["attrs"]["alpha"] == .05
                and table["attrs"]["uncertainty"] == state["metadata"]["uncertainty"],
                case+": empirical uncertainty/source state differs.")
        require(state["alpha"] == .05 and state["null"] == [0.]*len(reference["labels"]), case+": target inference declaration differs.")
        require(source["metadata"]["device"] == "cpu" and source["metadata"]["precision"] == "float64"
                and source["metadata"]["stata_parity_validated"] is False, case+": numerical/scientific contract differs.")
    if execution is not None:
        require(execution.get("status") == "ok" and execution.get("error") is None, "Native execution failed.")
        require(len(execution["outputs"]) == 8, "Exactly eight displayed target tables are required.")
        for case, output in zip(CASES, execution["outputs"]):
            expected = poststates[case]
            require(output["type"] == "table", case+": native output is not a table.")
            data = output["data"]
            require(data["index"] == [[label] for label in expected["index"]]
                    and data["columns"] == expected["columns"] and data["index_names"] == [None], case+": native labels/columns differ.")
            require(data["total_rows"] == len(expected["index"]) and data["total_columns"] == 6, case+": native target geometry differs.")
            require(bool(output.get("latex") or output.get("full_latex")), case+": missing LaTeX export.")
            differences[case+"/native"] = compare(data["rows"], expected["data"], case+" actual native displayed target")
    return {"gates": 8, "rows": 48, "independent_oracle": "SciPy likelihood score roots and complete empirical target evaluation under each actual PSU replica",
            "native_display_verified": execution is not None, "full_covariance_verified": True,
            "population_distribution_uncertainty": True, "maximum_absolute_differences": differences,
            "stata_parity_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, help="JSON {states,poststates,oracle_inputs}")
    parser.add_argument("--states", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--poststates", type=Path)
    parser.add_argument("--native-execution", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.payload:
        payload = json.loads(args.payload.read_text())
    else:
        if not all((args.states, args.inputs, args.poststates)):
            parser.error("Supply --payload or all three --states/--inputs/--poststates files.")
        payload = {"states": json.loads(args.states.read_text()), "oracle_inputs": json.loads(args.inputs.read_text()),
                   "poststates": json.loads(args.poststates.read_text())}
    receipt = verify(payload["states"], payload["oracle_inputs"], payload["poststates"],
                     json.loads(args.native_execution.read_text()) if args.native_execution else None)
    receipt["saved_payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    rendered = json.dumps(receipt, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
