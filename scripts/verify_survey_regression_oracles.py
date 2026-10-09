"""Development-only independent acceptance of installed survey-regression output.

The twelve-row synthetic design admits closed-form coefficients for all four
families. NumPy computes scores, observed sensitivity, complete stratified PSU
covariance and postestimation Jacobians; SciPy supplies inverse links and t/F
reference distributions. No production estimator or helper is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import special, stats


FAMILIES = ("linear", "logit", "probit", "poisson")
HELPERS = ("predict", "margins", "lincom", "test")
ALPHA = 0.05
DF = 4


def fixture():
    """Synthetic fixture fixed before examining the implementation's output."""
    return {
        "y": np.array([2., 5., 3., 2., 1., 4., 7., 6., 3., 1., 4., 2.]),
        "b": np.array([0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0.]),
        "c": np.array([0., 2., 1., 0., 3., 1., 2., 4., 0., 1., 2., 0.]),
        "x": np.tile([0., 1.], 6),
        "w": np.array([1., 2., 3., 1., 2., 4., 4., 2., 1., 3., 2., 1.]),
        "p": np.repeat(np.arange(6), 2),
        "h": np.repeat([0, 1], 6),
        "N": np.repeat([12, 6], 6),
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def comparison(actual, expected, *, atol=2e-8, rtol=2e-8):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, "Oracle comparison shape differs.")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "Oracle comparison needs finite complete values.")
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual - expected), initial=0.))


def link(family, mean):
    if family == "linear":
        return mean
    if family == "logit":
        return special.logit(mean)
    if family == "probit":
        return special.ndtri(mean)
    return np.log(mean)


def response(family, eta):
    if family == "linear":
        return eta
    if family == "logit":
        return special.expit(eta)
    if family == "probit":
        return special.ndtr(eta)
    return np.exp(eta)


def first_derivative(family, eta):
    if family == "linear":
        return np.ones_like(eta)
    if family == "probit":
        return stats.norm.pdf(eta)
    value = response(family, eta)
    return value * (1 - value) if family == "logit" else value


def second_derivative(family, eta):
    if family == "linear":
        return np.zeros_like(eta)
    if family == "probit":
        return -eta * stats.norm.pdf(eta)
    value = response(family, eta)
    return value * (1 - value) * (1 - 2 * value) if family == "logit" else value


def design_covariance(frame, bread, row_scores):
    """Original PSU enumeration with stratum centering and first-stage FPC."""
    sums = np.array([row_scores[frame["p"] == p].sum(0) for p in range(6)])
    meat = np.zeros_like(bread)
    for h in range(2):
        psus = np.unique(frame["p"][frame["h"] == h])
        totals = sums[psus]
        centered = totals - totals.mean(0)
        count = len(psus)
        population = frame["N"][frame["h"] == h][0]
        meat += (1 - count / population) * count / (count - 1) * centered.T @ centered
    inverse = np.linalg.inv(bread)
    return inverse @ meat @ inverse.T, sums, meat


def fit_oracle(family, frame):
    y = frame[{"linear": "y", "logit": "b", "probit": "b", "poisson": "c"}[family]]
    weights = frame["w"]
    group_means = np.array([
        np.average(y[frame["x"] == g], weights=weights[frame["x"] == g])
        for g in (0, 1)
    ])
    transformed = link(family, group_means)
    beta = np.array([transformed[0], transformed[1] - transformed[0]])
    matrix = np.column_stack([np.ones(len(y)), frame["x"]])
    eta = matrix @ beta
    if family == "linear":
        score, curvature = y - eta, np.ones(len(y))
    elif family == "logit":
        mean = special.expit(eta)
        score, curvature = y - mean, mean * (1 - mean)
    elif family == "probit":
        signed = (2 * y - 1) * eta
        mills = stats.norm.pdf(signed) / special.ndtr(signed)
        score, curvature = (2 * y - 1) * mills, mills * (signed + mills)
    else:
        mean = np.exp(eta)
        score, curvature = y - mean, mean
    bread = matrix.T @ ((weights * curvature)[:, None] * matrix)
    row_scores = matrix * (weights * score)[:, None]
    comparison(row_scores.sum(0), np.zeros(2), atol=2e-13)
    covariance, sums, meat = design_covariance(frame, bread, row_scores)
    return {
        "coefficients": beta,
        "covariance": covariance,
        "sensitivity": bread,
        "psu_score_sums": sums,
        "meat": meat,
        "weighted_group_means": group_means,
    }


def inference(estimates, covariance, *, null=0., alpha=ALPHA):
    estimates = np.atleast_1d(estimates)
    null = np.broadcast_to(null, estimates.shape)
    se = np.sqrt(np.diag(covariance))
    require((se > 0).all(), "This acceptance fixture expects nondegenerate uncertainty.")
    statistic = (estimates - null) / se
    width = stats.t.isf(alpha / 2, DF) * se
    return {
        "estimate": estimates,
        "std_error": se,
        "statistic": statistic,
        "p_value": 2 * stats.t.sf(np.abs(statistic), DF),
        "ci_low": estimates - width,
        "ci_high": estimates + width,
    }


def postestimation_oracles(fits):
    fit = fits["probit"]
    matrix = np.column_stack([np.ones(3), [-1., 0., 1.]])
    eta = matrix @ fit["coefficients"]
    gradient = first_derivative("probit", eta)[:, None] * matrix
    prediction_covariance = gradient @ fit["covariance"] @ gradient.T
    predict = {
        "jacobian": gradient,
        "covariance": prediction_covariance,
        "table": inference(response("probit", eta), prediction_covariance),
    }

    fit = fits["logit"]
    beta = fit["coefficients"]
    # All observed rows are set to x=.25, so any positive averaging weights
    # yield the same fixed-covariate continuous marginal effect.
    fixed = np.array([1., .25])
    eta = np.array([fixed @ beta])
    d1, d2 = first_derivative("logit", eta)[0], second_derivative("logit", eta)[0]
    gradient = (d2 * beta[1] * fixed + d1 * np.array([0., 1.]))[None, :]
    margin_covariance = gradient @ fit["covariance"] @ gradient.T
    margins = {
        "jacobian": gradient,
        "covariance": margin_covariance,
        "table": inference([d1 * beta[1]], margin_covariance),
    }

    fit = fits["linear"]
    vector = np.array([[1., 2.]])
    contrast_covariance = vector @ fit["covariance"] @ vector.T
    lincom = {
        "jacobian": vector,
        "covariance": contrast_covariance,
        "table": inference(vector @ fit["coefficients"], contrast_covariance, null=3.),
    }

    fit = fits["poisson"]
    restrictions = np.eye(2)
    difference = restrictions @ fit["coefficients"]
    restriction_covariance = restrictions @ fit["covariance"] @ restrictions.T
    q = len(restrictions)
    wald = float(difference @ np.linalg.solve(restriction_covariance, difference))
    statistic = (DF - q + 1) * wald / (q * DF)
    test = {
        "jacobian": restrictions,
        "covariance": restriction_covariance,
        "table": {
            "wald_chi2": np.array([wald]),
            "statistic": np.array([statistic]),
            "df_num": np.array([q]),
            "df_den": np.array([DF - q + 1]),
            "p_value": np.array([stats.f.sf(statistic, q, DF - q + 1)]),
        },
    }
    return dict(predict=predict, margins=margins, lincom=lincom, test=test)


def table_checks(columns, rows, expected):
    require(len(columns) == len(set(columns)), "Displayed table columns must be unique.")
    require(all(len(row) == len(columns) for row in rows), "Displayed table has truncated rows.")
    require(len(rows) == len(next(iter(expected.values()))), "Displayed table row count differs.")
    result = {}
    for name, values in expected.items():
        require(name in columns, "Missing displayed inference column: " + name)
        actual = [row[columns.index(name)] for row in rows]
        result[name + "_max_abs_error"] = comparison(actual, values)
    return result


def saved_jacobian(attrs, name):
    for key in ("jacobian", "gradient", "contrast_coefficients", "restrictions"):
        if key in attrs:
            array = np.asarray(attrs[key], dtype=float)
            return array[None, :] if array.ndim == 1 else array
    raise ValueError("Saved helper lacks its actual Jacobian/restrictions: " + name)


def saved_metadata_checks(name, saved, states, frame):
    """Bind each restored helper to the actual fixed evaluation and fit state."""
    attrs = saved["attrs"]
    expected_index = {
        "predict": ["low", "mid", "high"],
        "margins": ["AME:x"],
        "lincom": ["linear contrast"],
        "test": ["joint Wald F"],
    }[name]
    family = {"predict": "probit", "margins": "logit", "lincom": "linear", "test": "poisson"}[name]
    require(saved["index"] == expected_index, "Saved helper labels differ: " + name)
    require(attrs["survey_regression_state"] == states[family],
            "Saved helper lost exact restored fit identity: " + name)
    require(type(attrs["design_df"]) is int and attrs["design_df"] == DF,
            "Saved helper lost design df: " + name)
    if name != "test":
        require(attrs["alpha"] == ALPHA, "Saved helper alpha differs: " + name)
    if name in ("predict", "margins"):
        count = 3 if name == "predict" else 12
        require(attrs["physical_positions"] == list(range(count))
                and attrs["covariate_exclusions"] == []
                and attrs["n_evaluation"] == count and attrs["missing"] == "raise"
                and attrs["conditional_fixed_covariates"] is True
                and attrs["population_distribution_uncertainty"] is False,
                "Saved evaluation sample/uncertainty differs: " + name)
        if name == "predict":
            require(attrs["kind"] == "response" and attrs["at"] == {}
                    and attrs["evaluation_weight_column"] is None,
                    "Saved prediction specification differs.")
            comparison(attrs["evaluation_weights"], np.full(3, 1 / 3))
        else:
            require(attrs["at"] == {"x": .25} and attrs["variables"] == ["x"]
                    and attrs["evaluation_weight_column"] == "w",
                    "Saved margins fixed covariates/AME/weight roles differ.")
            comparison(attrs["evaluation_weights"], frame["w"] / frame["w"].sum())
    if name == "test":
        comparison(attrs["restrictions"], np.eye(2))
        comparison(attrs["null"], np.zeros(2))
        comparison(attrs["restriction_row_scales"], np.ones(2))
        require(attrs["method"] == "survey-adjusted Wald F",
                "Saved joint test must retain its adjusted survey convention.")
    return {"labels": expected_index, "fit_family": family,
            "restored_fit_identity_compared": True,
            "physical_positions_compared": name in ("predict", "margins"),
            "conditional_sample_metadata_compared": name in ("predict", "margins"),
            "inference_specification_compared": True}


def native_label_checks(data, expected_index):
    # Console serialization represents even a simple index as one cell per
    # level; compare exact labels rather than inferring alignment from numbers.
    require(data.get("index") == [[value] for value in expected_index],
            "Actual native display lost physical/coefficient row labels.")
    require(data.get("index_names") == [None], "Actual native index structure differs.")
    require(data.get("total_rows") == len(expected_index)
            and data.get("total_columns") == len(data["columns"]),
            "Actual native display reports incomplete table dimensions.")
    return {"labels": expected_index, "physical_or_parameter_order_compared": True}


def check_states(states_path, execution_path, helpers_path):
    states = json.loads(states_path.read_text())
    execution = json.loads(execution_path.read_text())
    helpers = json.loads(helpers_path.read_text())
    require(set(states) == set(FAMILIES), "Persist exactly four full fit states.")
    require(set(helpers) == set(HELPERS), "Persist exactly four postestimation states.")
    require(execution.get("status") == "ok" and execution.get("error") is None,
            "Native acceptance must be an actual successful execution.")
    outputs = execution.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == 8, "Require eight actual native outputs.")
    frame = fixture()
    fits = {family: fit_oracle(family, frame) for family in FAMILIES}
    checks = {}
    for family in FAMILIES:
        state, expected = states[family], fits[family]
        metadata = state["metadata"]
        require(state["family"] == family and state["labels"] == ["_cons", "x"]
                and state["regressors"] == ["x"] and state["intercept"] is True,
                "Fit family/order differs from the declared acceptance fixture.")
        require(state["outcome"] == {"linear": "y", "logit": "b", "probit": "b", "poisson": "c"}[family],
                "Saved outcome differs from the declared fixture.")
        require(state["df"] == DF and state["alpha"] == ALPHA and state["null"] == [0., 0.],
                "Saved inference convention differs from the declared fixture.")
        require(metadata["n_design"] == 12 and metadata["n_used"] == 12
                and metadata["sample_positions"] == list(range(12))
                and metadata["stratum_psu_indices"] == [[0, 1, 2], [3, 4, 5]]
                and metadata["fpc_multipliers"] == [.75, .5],
                "Fit lost complete sample/PSU/FPC geometry.")
        require(state["design"]["weights"] == "w" and state["design"]["psu"] == "p"
                and state["design"]["strata"] == "h" and state["design"]["fpc"] == "N",
                "Saved declaration roles differ from the independent fixture.")
        require(metadata["convergence"]["converged"] is True,
                "Native fit did not record accepted convergence.")
        scaling = len(frame["w"]) / frame["w"].sum()
        checks[family] = {
            "coefficient_max_abs_error": comparison(state["coefficients"], expected["coefficients"]),
            "covariance_max_abs_error": comparison(state["covariance"], expected["covariance"]),
            "observed_sensitivity_max_abs_error": comparison(metadata["sensitivity"], scaling * expected["sensitivity"]),
            "complete_psu_scores_max_abs_error": comparison(metadata["psu_score_sums"], scaling * expected["psu_score_sums"]),
            "independent_coefficients": expected["coefficients"],
            "independent_covariance": expected["covariance"],
            "independent_table": inference(expected["coefficients"], expected["covariance"]),
            "df": DF,
            "complete_physical_positions": list(range(12)),
        }
    post = postestimation_oracles(fits)
    for name in HELPERS:
        saved, expected = helpers[name], post[name]
        attrs = saved["attrs"]
        checks[name] = {
            "saved_full_covariance_max_abs_error": comparison(attrs["covariance_matrix"], expected["covariance"]),
            "saved_jacobian_max_abs_error": comparison(saved_jacobian(attrs, name), expected["jacobian"]),
            "saved_displayed_inference": table_checks(saved["columns"], saved["data"], expected["table"]),
            "independent_covariance": expected["covariance"],
            "independent_jacobian": expected["jacobian"],
            "independent_table": expected["table"],
            "saved_metadata": saved_metadata_checks(name, saved, states, frame),
        }
    for name, output in zip((*FAMILIES, *HELPERS), outputs):
        require(output.get("type") == "table" and bool(output.get("latex")),
                "Require complete exportable native table for " + name)
        data = output["data"]
        checks[name]["actual_native_displayed_inference"] = table_checks(
            data["columns"], data["rows"], checks[name]["independent_table"])
        labels = ["_cons", "x"] if name in FAMILIES else helpers[name]["index"]
        checks[name]["actual_native_displayed_labels"] = native_label_checks(data, labels)
        checks[name]["actual_native_displayed_inference_compared"] = True
    serializable_fixture = {key: value.tolist() for key, value in frame.items()}
    return {
        "status": "independent-survey-regression-native-passed",
        "oracle": "Closed-form saturated group inverse links; NumPy observed-score bread, stratified PSU/FPC covariance and delta Jacobians; SciPy t/F",
        "production_helpers_used": False,
        "licensed_stata_execution": False,
        "sample_rows": 12,
        "design_df": DF,
        "fixture": serializable_fixture,
        "fixture_sha256": hashlib.sha256(json.dumps(serializable_fixture, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (states_path, execution_path, helpers_path)},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "methods": checks,
        "postestimation_uncertainty": "Saved survey coefficient uncertainty conditional on fixed evaluation covariates/weights; no sampled-covariate or future-outcome interval",
        "reference_formulas": [
            "https://www.stata.com/manuals/svyvarianceestimation.pdf",
            "https://www.stata.com/manuals/rtest.pdf",
            "https://www.stata.com/manuals/rmargins.pdf",
        ],
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unexpected oracle JSON value: " + type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--postestimation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution = args.execution or args.states.with_name("native-execution.json")
    postestimation = args.postestimation or args.states.with_name("postestimation-states.json")
    result = check_states(args.states, execution, postestimation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, default=json_value) + "\n")
    print(json.dumps({"status": result["status"], "methods": len(result["methods"])}))


if __name__ == "__main__":
    main()
