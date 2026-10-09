"""Independent NumPy/SciPy checks of eight two-stage control-function outcomes.

This development oracle imports neither OpenEconometrics nor Torch. It refits
the first-stage OLS and outcome criterion independently, differences the entire
stacked score (including the generated residual/design), constructs uncorrected
HC0 or whole-cluster CR0 sandwiches, and evaluates conditional prediction delta
uncertainty in the complete first-stage/outcome parameter coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import linalg, optimize, special, stats


KINDS = (
    "gaussian", "logit", "probit", "cloglog", "poisson", "gamma",
    "inverse_gaussian", "fractional_logit",
)
APIS = dict(zip(KINDS, (
    "cfregress", "cflogit", "cfprobit", "cfcloglog", "cfpoisson", "cfgamma",
    "cfinvgauss", "cffraclogit",
), strict=True))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, *, atol=2e-7, rtol=2e-7):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, "Independent comparison shape differs.")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "Independent comparison requires finite complete arrays.")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual - expected), initial=0.0))


def response(kind, eta):
    eta = np.asarray(eta, dtype=float)
    if kind == "gaussian":
        return eta
    if kind in {"logit", "fractional_logit"}:
        return special.expit(eta)
    if kind == "probit":
        return special.ndtr(eta)
    if kind == "cloglog":
        return -np.expm1(-np.exp(eta))
    return np.exp(eta)


def response_derivative(kind, eta):
    mean = response(kind, eta)
    if kind == "gaussian":
        return np.ones_like(mean)
    if kind in {"logit", "fractional_logit"}:
        return mean * (1 - mean)
    if kind == "probit":
        return stats.norm.pdf(eta)
    if kind == "cloglog":
        return np.exp(eta - np.exp(eta))
    return mean


def outcome_quantities(kind, y, eta):
    """Per-row working criterion, scalar score s, and observed derivative h.

    Gamma and inverse-Gaussian use fixed working dispersion one. Fractional
    logit is the Bernoulli quasi criterion, with no binomial choose constant.
    The observed h can depend on y; expected-Fisher replacements are not used.
    """
    y, eta = np.asarray(y, dtype=float), np.asarray(eta, dtype=float)
    require(kind in KINDS, "Unknown outcome kind.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if kind == "gaussian":
            score = y - eta
            return -0.5 * score**2, score, -np.ones_like(y)
        if kind in {"logit", "fractional_logit"}:
            mean = special.expit(eta)
            return y * eta - np.logaddexp(0, eta), y - mean, -mean * (1 - mean)
        if kind == "probit":
            pos = np.exp(stats.norm.logpdf(eta) - special.log_ndtr(eta))
            neg = np.exp(stats.norm.logpdf(eta) - special.log_ndtr(-eta))
            criterion = y * special.log_ndtr(eta) + (1 - y) * special.log_ndtr(-eta)
            score = y * pos - (1 - y) * neg
            derivative = -y * pos * (eta + pos) - (1 - y) * neg * (-eta + neg)
            return criterion, score, derivative
        if kind == "cloglog":
            rate = np.exp(eta)
            mean = -np.expm1(-rate)
            ratio = rate * np.exp(-rate) / mean
            criterion = y * np.log(mean) - (1 - y) * rate
            score = y * ratio - (1 - y) * rate
            derivative = y * ratio * (1 - rate - ratio) - (1 - y) * rate
            return criterion, score, derivative
        mean = np.exp(eta)
        if kind == "poisson":
            return y * eta - mean - special.gammaln(y + 1), y - mean, -mean
        if kind == "gamma":
            return -y / mean - eta, y / mean - 1, -y / mean
        return (-y / (2 * mean**2) + 1 / mean,
                (y - mean) / mean**2, (-2 * y + mean) / mean**2)


def central_jacobian(function, point, *, step=1e-5):
    """Whole-vector central difference with parameter-specific finite steps."""
    point = np.asarray(point, dtype=float)
    columns = []
    for j in range(len(point)):
        delta = step * max(1.0, abs(point[j]))
        upper, lower = point.copy(), point.copy()
        upper[j] += delta
        lower[j] -= delta
        columns.append((np.asarray(function(upper)) - np.asarray(function(lower))) / (2 * delta))
    return np.column_stack(columns)


def _codes(labels):
    """Keep scalar label types distinct; whole clusters are the score units."""
    unique, codes = {}, []
    for label in labels:
        if isinstance(label, np.generic):
            label = label.item()
        key = (type(label).__name__, json.dumps(label, sort_keys=True, allow_nan=False))
        if key not in unique:
            unique[key] = len(unique)
        codes.append(unique[key])
    return np.asarray(codes, dtype=int), len(unique)


def prepare_inputs(inputs, *, outcome=True):
    """Common physical sample, including instruments and declared cluster IDs."""
    data = inputs["data"]
    exog, instruments = list(inputs.get("x") or []), list(inputs["instruments"])
    endog = inputs["endogenous"]
    names = [*exog, *instruments, endog]
    if outcome:
        names.append(inputs["y"])
    cluster = inputs.get("cluster")
    if cluster:
        names.append(cluster)
    require(len({len(values) for values in data.values()}) == 1, "Input columns differ in length.")
    n = len(data[endog])
    keep = np.ones(n, dtype=bool)
    columns = {}
    for name in dict.fromkeys(names):
        require(name in data, "Missing independent input column: " + name)
        if name == cluster:
            values = list(data[name])
            valid = np.asarray([value is not None and not (
                isinstance(value, (float, np.floating)) and not np.isfinite(value)
            ) for value in values])
        else:
            values = np.asarray([np.nan if value is None else value for value in data[name]], dtype=float)
            valid = np.isfinite(values)
        columns[name] = values
        keep &= valid
    require(keep.all() or inputs.get("missing", "raise") == "drop", "Incomplete common sample.")
    positions = np.flatnonzero(keep)
    intercept = bool(inputs.get("intercept", True))
    constant = [np.ones(len(positions))] if intercept else []
    z = np.column_stack([*constant, *[columns[name][keep] for name in [*exog, *instruments]]])
    x = np.column_stack([*constant, *[columns[name][keep] for name in [*exog, endog]]])
    labels = [columns[cluster][i] for i in positions] if cluster else None
    return dict(
        z=z, x=x, d=columns[endog][keep],
        y=columns[inputs["y"]][keep] if outcome else None,
        cluster_labels=labels, positions=positions, n_original=n,
        z_terms=(["_cons"] if intercept else []) + exog + instruments,
        x_terms=(["_cons"] if intercept else []) + exog + [endog],
    )


def stacked_scores(theta, kind, z, x, d, y):
    first = z.shape[1]
    gamma, beta = theta[:first], theta[first:]
    residual = d - z @ gamma
    design = np.column_stack([x, residual])
    _, score, _ = outcome_quantities(kind, y, design @ beta)
    return np.column_stack([z * residual[:, None], design * score[:, None]])


def reference_at(kind, prepared, gamma, beta):
    """Analytic unsymmetrized negative Jacobian and manual score sandwich."""
    z, x, d, y = (prepared[key] for key in ("z", "x", "d", "y"))
    residual = d - z @ gamma
    q = np.column_stack([x, residual])
    eta = q @ beta
    criterion, score, h = outcome_quantities(kind, y, eta)
    first, second = z.shape[1], q.shape[1]
    bread = np.zeros((first + second, first + second))
    bread[:first, :first] = z.T @ z
    cross = beta[-1] * q.T @ (h[:, None] * z)
    cross[-1] += score @ z
    bread[first:, :first] = cross
    bread[first:, first:] = -q.T @ (h[:, None] * q)
    theta = np.r_[gamma, beta]
    row_scores = stacked_scores(theta, kind, z, x, d, y)
    cluster = prepared["cluster_labels"]
    if cluster is None:
        codes, count, score_units = None, None, row_scores
    else:
        codes, count = _codes(cluster)
        score_units = np.zeros((count, first + second))
        np.add.at(score_units, codes, row_scores)
    meat = score_units.T @ score_units
    inverse = linalg.solve(bread, np.eye(len(bread)))
    covariance = inverse @ meat @ inverse.T
    numeric_bread = -central_jacobian(
        lambda value: stacked_scores(value, kind, z, x, d, y).sum(0), theta,
    )
    return dict(
        **prepared, gamma=gamma, beta=beta, residual=residual, design=q, eta=eta,
        fitted=response(kind, eta), row_scores=row_scores, bread=bread, meat=meat,
        joint_covariance=covariance, scalar_score=score, scalar_derivative=h,
        criterion=criterion.sum(), cluster_codes=codes, cluster_count=count,
        numeric_bread=numeric_bread, score_units=score_units, theta=theta,
    )


def fit_oracle(kind, inputs):
    prepared = prepare_inputs(inputs)
    z, x, d, y = (prepared[key] for key in ("z", "x", "d", "y"))
    gamma = linalg.lstsq(z, d, lapack_driver="gelsd")[0]
    q = np.column_stack([x, d - z @ gamma])
    require(np.linalg.matrix_rank(z) == z.shape[1] and np.linalg.matrix_rank(q) == q.shape[1],
            "Independent fit requires full-rank declared designs.")
    if kind == "gaussian":
        initial = linalg.lstsq(q, y, lapack_driver="gelsd")[0]
    elif kind in {"gamma", "inverse_gaussian", "poisson"}:
        initial = linalg.lstsq(q, np.log(np.maximum(y, 0.2)), lapack_driver="gelsd")[0]
    else:
        initial = np.zeros(q.shape[1])
        if inputs.get("intercept", True):
            mean = y.mean()
            initial[0] = (special.ndtri(mean) if kind == "probit" else
                          np.log(-np.log1p(-mean)) if kind == "cloglog" else special.logit(mean))

    def criterion(beta):
        values = outcome_quantities(kind, y, q @ beta)[0]
        return -float(values.sum())

    def gradient(beta):
        return -q.T @ outcome_quantities(kind, y, q @ beta)[1]

    def hessian(beta):
        return -q.T @ (outcome_quantities(kind, y, q @ beta)[2][:, None] * q)

    fitted = optimize.minimize(criterion, initial, jac=gradient, hess=hessian,
                               method="trust-exact", options={"gtol": 2e-9, "maxiter": 500})
    point = fitted.x
    if np.max(np.abs(gradient(point))) > 1e-8:
        polished = optimize.root(gradient, point, jac=hessian, method="hybr", tol=1e-11)
        if np.isfinite(polished.x).all() and np.max(np.abs(gradient(polished.x))) < np.max(np.abs(gradient(point))):
            point = polished.x
    require(np.isfinite(criterion(point)) and np.max(np.abs(gradient(point))) < 3e-6,
            "Independent SciPy outcome criterion did not reach a stationary point: " + fitted.message)
    require(np.linalg.eigvalsh(hessian(point)).min() > 0,
            "Independent outcome information is not positive definite.")
    result = reference_at(kind, prepared, gamma, point)
    result["scipy_fit"] = dict(success=bool(fitted.success), message=str(fitted.message),
                               iterations=int(fitted.nit), max_abs_score=float(np.abs(gradient(point)).max()))
    return result


def prediction_oracle(kind, inputs, fitted):
    prepared = prepare_inputs(inputs, outcome=False)
    z, x, d = (prepared[key] for key in ("z", "x", "d"))
    gamma, beta, covariance = fitted["gamma"], fitted["beta"], fitted["joint_covariance"]
    residual = d - z @ gamma
    q = np.column_stack([x, residual])
    eta = q @ beta
    eta_jacobian = np.column_stack([-beta[-1] * z, q])
    mean = response(kind, eta)
    mean_jacobian = response_derivative(kind, eta)[:, None] * eta_jacobian
    eta_covariance = eta_jacobian @ covariance @ eta_jacobian.T
    mean_covariance = mean_jacobian @ covariance @ mean_jacobian.T
    eta_se = np.sqrt(np.diag(eta_covariance))
    mean_se = np.sqrt(np.diag(mean_covariance))
    critical = stats.norm.isf(inputs.get("alpha", 0.05) / 2)

    def full_mean(theta):
        a = z.shape[1]
        new_residual = d - z @ theta[:a]
        return response(kind, np.column_stack([x, new_residual]) @ theta[a:])

    numeric_mean = central_jacobian(full_mean, fitted["theta"])
    return dict(
        position=prepared["positions"], mean=mean, std_error=mean_se,
        ci_low=response(kind, eta - critical * eta_se),
        ci_high=response(kind, eta + critical * eta_se),
        eta=eta, eta_std_error=eta_se, control_residual=residual,
        mean_jacobian=mean_jacobian, eta_jacobian=eta_jacobian,
        numeric_mean_jacobian=numeric_mean, covariance_matrix=mean_covariance,
        eta_covariance_matrix=eta_covariance,
    )


def fixture(kind, rows=240, seed=20261007, *, just_identified=False):
    """Deterministic synthetic inputs independent of production estimator output."""
    require(kind in KINDS and rows >= 40, "Unknown fixture or too few rows.")
    rng = np.random.default_rng(seed)
    x, z1, z2 = rng.normal(size=(3, rows))
    latent = rng.normal(size=rows)
    d = 0.35 + 0.35*x + 0.8*z1 - 0.45*z2 + latent
    eta = -0.2 + 0.25*x + 0.35*d + 0.4*latent
    if kind == "gaussian":
        y = eta + rng.normal(scale=0.7 + 0.1*np.abs(x), size=rows)
    elif kind in {"logit", "probit", "cloglog"}:
        y = (rng.uniform(size=rows) < response(kind, eta)).astype(float)
    elif kind == "fractional_logit":
        mean = response(kind, eta)
        y = rng.beta(8*mean, 8*(1-mean))
        y[0], y[1] = 0, 1
    elif kind == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif kind == "gamma":
        y = rng.gamma(shape=3, scale=np.exp(eta)/3)
    else:
        y = rng.wald(mean=np.exp(eta), scale=4)
    data = dict(y=y.tolist(), d=d.tolist(), x=x.tolist(), z1=z1.tolist(), z2=z2.tolist(),
                cluster=(np.arange(rows)//4).tolist(), singleton=np.arange(rows).tolist())
    return dict(data=data, y="y", endogenous="d", x=["x"],
                instruments=["z1"] if just_identified else ["z1", "z2"],
                intercept=True, missing="raise", alpha=0.05)


def query_fixture():
    return dict(data=dict(x=[-0.6, 0.0, 0.4, 0.8], d=[0.3, -0.2, 0.8, 1.0],
                         z1=[0.5, -0.1, 0.2, -0.4], z2=[-0.3, 0.4, 0.1, 0.2]),
                y="y", endogenous="d", x=["x"], instruments=["z1", "z2"],
                intercept=True, missing="raise", alpha=0.05)


def two_stage_least_squares(inputs):
    """Independent 2SLS points and uncorrected score covariance for Gaussian checks."""
    sample = prepare_inputs(inputs)
    z, x, d, y = (sample[key] for key in ("z", "x", "d", "y"))
    structural = np.column_stack([x[:, :-1], d])
    fitted_design = z @ linalg.lstsq(z, structural, lapack_driver="gelsd")[0]
    bread = fitted_design.T @ structural
    beta = linalg.solve(bread, fitted_design.T @ y)
    scores = fitted_design * (y - structural @ beta)[:, None]
    if sample["cluster_labels"] is not None:
        codes, count = _codes(sample["cluster_labels"])
        units = np.zeros((count, scores.shape[1]))
        np.add.at(units, codes, scores)
        scores = units
    inverse = linalg.solve(bread, np.eye(len(bread)))
    return beta, inverse @ (scores.T @ scores) @ inverse.T


def _table_checks(table, expected, columns):
    names, rows = table["columns"], table["data"]
    require(len(names) == len(set(names)), "Receipt table repeats columns.")
    require(all(len(row) == len(names) for row in rows), "Receipt table has truncated rows.")
    result = {}
    for name in columns:
        require(name in names, "Receipt table lacks " + name)
        result[name] = compare([row[names.index(name)] for row in rows], expected[name],
                               atol=8e-6, rtol=8e-6)
    return result


def check_case(case):
    kind, inputs = case["kind"], dict(case["inputs"])
    independent = fit_oracle(kind, inputs)
    result = case["result"]
    state = result["extra"]["control_function_state"]
    require(state["kind"] == kind, "Receipt state kind differs.")
    require(state["schema"] == "openecon.control_function.v1", "Receipt state schema differs.")
    z_terms, x_terms = list(independent["z_terms"]), list(independent["x_terms"])
    if inputs.get("intercept", True):
        require(state["z_terms"][0] in {"_cons", "Intercept"}, "Unknown declared intercept label.")
        z_terms[0] = x_terms[0] = state["z_terms"][0]
    require(state["z_terms"] == z_terms and state["x_terms"] == x_terms,
            "Declared first-stage/outcome numeric term order differs.")
    order = ([dict(term="first_stage:"+term, equation="first_stage") for term in z_terms]
             + [dict(term="outcome:"+term, equation="outcome") for term in [*x_terms, "ControlResidual"]])
    require(state["parameter_order"] == order, "Saved joint parameter order differs.")
    require([dict(term=row["term"], equation=row["equation"]) for row in result["coefficients"]] == order,
            "Reported coefficient/equation order differs.")
    theta = np.asarray([row["estimate"] for row in result["coefficients"]])
    checks = {"independent_coefficients": compare(theta, independent["theta"], atol=8e-6, rtol=8e-6)}
    first = independent["z"].shape[1]
    at_saved = reference_at(kind, prepare_inputs(inputs), theta[:first], theta[first:])
    checks["full_unsymmetrized_numeric_bread"] = compare(
        at_saved["bread"], at_saved["numeric_bread"], atol=4e-6, rtol=2e-6,
    )
    require(np.max(np.abs(at_saved["row_scores"].sum(0))) < 2e-5,
            "Native coefficients do not solve both independently evaluated stages.")
    for name in ("z", "x", "d", "y", "gamma", "beta", "residual", "design", "fitted",
                 "row_scores", "bread", "meat", "joint_covariance", "scalar_score", "scalar_derivative"):
        checks[name] = compare(state[name], at_saved[name], atol=4e-6, rtol=2e-6)
    checks["working_criterion"] = compare(state["criterion"], at_saved["criterion"], atol=3e-6)
    checks["result_full_covariance"] = compare(result["covariance_matrix"], at_saved["joint_covariance"],
                                                atol=4e-6, rtol=2e-6)
    require(result["sample_positions"] == at_saved["positions"].tolist(), "Physical fit sample differs.")
    require(result["nobs"] == len(at_saved["y"]) and result["nobs_original"] == at_saved["n_original"],
            "Receipt sample counts differ.")
    if at_saved["cluster_count"] is not None:
        require(state["cluster_count"] == at_saved["cluster_count"], "Whole-cluster count differs.")
        # Numeric code labels can be permuted; compare induced equivalence classes.
        saved_codes = np.asarray(state["cluster_codes"])
        compare(saved_codes[:, None] == saved_codes[None, :],
                at_saved["cluster_codes"][:, None] == at_saved["cluster_codes"][None, :])
    se = np.sqrt(np.diag(at_saved["joint_covariance"]))
    alpha = inputs.get("alpha", 0.05)
    statistic = theta / se
    critical = stats.norm.isf(alpha / 2)
    expected = dict(estimate=theta, std_error=se, statistic=statistic,
                    p_value=2*stats.norm.sf(np.abs(statistic)),
                    ci_low=theta-critical*se, ci_high=theta+critical*se)
    table = dict(columns=list(expected), data=[[row[name] for name in expected] for row in result["coefficients"]])
    checks["coefficient_inference"] = _table_checks(table, expected, list(expected))
    control = result["tests"]["control_coefficient_zero"]
    control_statistic = float(theta[-1]**2 / at_saved["joint_covariance"][-1, -1])
    checks["control_coefficient_wald"] = compare(control["statistic"], control_statistic)
    checks["control_coefficient_wald_p"] = compare(control["p_value"], stats.chi2.sf(control_statistic, 1))
    require(control["df"] == 1 and control["distribution"] == "chi2", "Control coefficient test domain differs.")
    require(result["inference"]["small_sample_correction"] == 1.0 and result["inference"]["use_t"] is False,
            "Receipt silently adds a sampling correction or finite-sample t law.")
    require(not set(result["metrics"]).intersection({"aic", "bic", "log_likelihood"}),
            "Working-score result implies an unsupported estimated likelihood.")
    for index, prediction in enumerate(case.get("predictions", [])):
        query = {**inputs, **prediction["inputs"], "cluster": None}
        expected_prediction = prediction_oracle(kind, query, at_saved)
        saved = prediction["table"]
        pred_checks = _table_checks(saved, expected_prediction,
                                   ["position", "mean", "std_error", "ci_low", "ci_high", "eta",
                                    "eta_std_error", "control_residual"])
        attrs = saved["attrs"]
        for name in ("mean_jacobian", "eta_jacobian"):
            pred_checks[name] = compare(attrs[name], expected_prediction[name], atol=4e-6, rtol=2e-6)
        pred_checks["full_joint_prediction_parameter_covariance"] = compare(
            attrs["joint_parameter_covariance"], at_saved["joint_covariance"], atol=4e-6, rtol=2e-6,
        )
        saved_jac = np.asarray(attrs["mean_jacobian"])
        saved_cov = np.asarray(attrs["joint_parameter_covariance"])
        pred_checks["reconstructed_all_cross_row_covariance"] = compare(
            saved_jac @ saved_cov @ saved_jac.T, expected_prediction["covariance_matrix"],
            atol=4e-6, rtol=2e-6,
        )
        pred_checks["whole_conditional_mean_numeric_jacobian"] = compare(
            expected_prediction["mean_jacobian"], expected_prediction["numeric_mean_jacobian"],
            atol=2e-8, rtol=2e-7,
        )
        require(attrs["conditional_mean_only"] is True, "Prediction claims an unsupported target.")
        require(attrs["parameter_order"] == order, "Prediction delta parameter order differs.")
        require(attrs["source_state_sha256"] == state["integrity_sha256"], "Prediction source binding differs.")
        checks[f"prediction_{index}"] = pred_checks
    if kind == "gaussian":
        iv_beta, iv_covariance = two_stage_least_squares(inputs)
        checks["gaussian_point_2sls"] = compare(at_saved["beta"][:-1], iv_beta)
        if len(inputs["instruments"]) == 1:
            checks["just_identified_2sls_covariance"] = compare(
                at_saved["joint_covariance"][first:-1, first:-1], iv_covariance,
                atol=4e-6, rtol=2e-6,
            )
    return dict(checks=checks, scipy_fit=independent["scipy_fit"],
                numeric_bread_asymmetry=float(np.max(np.abs(at_saved["bread"]-at_saved["bread"].T))),
                first_outcome_cross_covariance=at_saved["joint_covariance"][:first, first:])


def check_receipt(receipt):
    cases = receipt["cases"]
    require(len(cases) == 16, "Acceptance needs all eight outcomes under HC0 and CR0.")
    expected = {kind+suffix for kind in KINDS for suffix in ("_hc0", "_cr0")}
    require({case["case_id"] for case in cases} == expected, "Acceptance case IDs are incomplete/duplicated.")
    methods = {}
    for case in cases:
        covariance = case["inputs"].get("covariance", case.get("covariance", "robust"))
        clustered = bool(case["inputs"].get("cluster"))
        require(clustered == case["case_id"].endswith("_cr0"), "Case ID/sample covariance unit differs.")
        require(covariance in {"robust", "cluster"}, "Acceptance requires declared HC0/CR0 domain.")
        methods[case["case_id"]] = check_case(case)
    provenance = None
    if "identity" in receipt:
        require(receipt["status"] in {"native-verified", "native-restarted"},
                "Native receipt has no completed readback.")
        execution = receipt["execution_record"]
        def canonical(value):
            return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        require(hashlib.sha256(canonical(execution)).hexdigest() == receipt["execution_sha256"],
                "Native execution record digest differs.")
        require(hashlib.sha256(execution["code"].encode()).hexdigest() == receipt["source_sha256"],
                "Native execution source differs from seeded exact script.")
        require(hashlib.sha256(canonical(receipt["complete_states"])).hexdigest() == receipt["states_sha256"],
                "Complete native state envelope digest differs.")
        require(receipt["complete_states"]["cases"] == cases, "Native receipt duplicates inconsistent cases.")
        require(hashlib.sha256(canonical(receipt["inputs"])).hexdigest() == receipt["inputs_sha256"],
                "Complete native original-input digest differs.")
        for case in cases:
            require(case["result"] == case["restored_result"], "Complete native saved/restored state differs.")
            require(case["inputs"]["data"] == receipt["inputs"][case["kind"]]["data"],
                    "Native fit inputs differ from preserved original inputs.")
        require(receipt["identity"]["identifier"] == "org.openecon.qa.control-function-eight-20261007",
                "Native receipt is not the dedicated synthetic app.")
        require(len(receipt["identity"]["source_ref"]) == 40 and receipt["identity"]["source_parity"],
                "Native receipt lacks explicit committed frozen-source parity.")
        require(len(receipt["outputs"]) == 8 and all(output.get("latex") for output in receipt["outputs"]),
                "Native receipt lacks eight exportable outputs.")
        for kind, output in zip(KINDS, receipt["outputs"], strict=True):
            require(output["type"] == "table", "Native result is not an actual coefficient table.")
            expected_case = next(case for case in cases if case["case_id"] == kind+"_hc0")
            coefficients = expected_case["result"]["coefficients"]
            table = output["data"]
            require(table["total_rows"] == len(coefficients) and len(table["rows"]) == len(coefficients),
                    "Native displayed result is truncated.")
            for row, coefficient in zip(table["rows"], coefficients, strict=True):
                for name, expected_value in coefficient.items():
                    require(name in table["columns"] and row[table["columns"].index(name)] == expected_value,
                            "Native displayed coefficient or inference differs.")
        provenance = dict(status=receipt["status"], source_ref=receipt["identity"]["source_ref"],
                          runtime_sha256=receipt["identity"]["runtime_sha256"],
                          execution_id=receipt["execution_id"], actual_output_tables=8,
                          restart_checked=receipt["status"] == "native-restarted")
    return dict(
        status="independent-control-function-comparison-passed", methods=methods,
        production_estimator_or_helper_imported=False, torch_imported=False,
        licensed_vendor_execution=False, versions=dict(numpy=np.__version__, scipy=scipy.__version__),
        uncertainty="Uncorrected observed stacked-score HC0/whole one-way CR0; conditional mean full gamma/beta delta; normal asymptotics",
        scope="Eight fixed-working-dispersion-one outcomes, one continuous OLS first stage; independent numerical comparison only",
        native_provenance=provenance,
    )


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unexpected oracle JSON value: " + type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-receipt", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.native_receipt.read_text())
    report = check_receipt(receipt)
    report["native_receipt_sha256"] = hashlib.sha256(args.native_receipt.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False, default=json_value)+"\n")
    print(json.dumps(dict(status=report["status"], cases=len(report["methods"]))))


if __name__ == "__main__":
    main()
