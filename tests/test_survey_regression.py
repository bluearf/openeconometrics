"""Independent weighted likelihood, PSU sandwich and saved inference checks."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


FAMILIES = ("regress", "logit", "probit", "poisson")


def fixture(*, fpc=False):
    """Several nonnested score directions, repeated PSU IDs and unequal weights."""
    rng = np.random.default_rng(19850717)
    n = 48
    x, z = rng.normal(size=(2, n))
    frame = pd.DataFrame(
        {
            "w": rng.uniform(0.7, 3.2, n),
            "h": np.repeat(["north", "south", "east", "west"], 12),
            "p": np.tile(np.repeat([1, 2, 3], 4), 4),
            "x": x,
            "z": z,
            "continuous": 1.4 + 0.6 * x - 0.35 * z + rng.normal(scale=0.7, size=n),
            "binary": rng.binomial(1, special.expit(0.1 + 0.35 * x - 0.2 * z)),
            "count": rng.poisson(np.exp(0.3 + 0.25 * x - 0.15 * z)),
            "domain": 1,
        },
        index=np.tile([37, 2, 37, -4], 12),
    )
    frame.loc[(frame.h == "west") & (frame.p == 3), "domain"] = 0
    if fpc:
        frame["N"] = np.repeat([6, 9, 12, 15], 12)
    design = oe.survey_design(
        frame, weights="w", psu="p", strata="h", fpc="N" if fpc else None
    )
    return frame, design


def outcome(family):
    return {"regress": "continuous", "logit": "binary", "probit": "binary", "poisson": "count"}[
        family
    ]


def matrix(frame, intercept=True):
    regressors = frame[["x", "z"]].to_numpy(dtype=float)
    return np.column_stack((np.ones(len(frame)), regressors)) if intercept else regressors


def response(family, eta):
    if family == "regress":
        return eta
    if family == "logit":
        return special.expit(eta)
    if family == "probit":
        return special.ndtr(eta)
    return np.exp(eta)


def response_derivative(family, eta):
    mu = response(family, eta)
    if family == "regress":
        return np.ones_like(eta)
    if family == "logit":
        return mu * (1 - mu)
    if family == "probit":
        return stats.norm.pdf(eta)
    return mu


def likelihood_parts(family, beta, x, y, w):
    """Analytic observed Hessian, independently expressed with NumPy/SciPy."""
    eta = x @ beta
    if family == "regress":
        residual = y - eta
        objective = np.sum(w * residual**2) / 2
        unit_score, curvature = residual, np.ones(len(y))
    elif family == "logit":
        p = special.expit(eta)
        objective = np.sum(w * (np.logaddexp(0, eta) - y * eta))
        unit_score, curvature = y - p, p * (1 - p)
    elif family == "probit":
        sign = 2 * y - 1
        log_probability = special.log_ndtr(sign * eta)
        objective = -np.sum(w * log_probability)
        # Inverse Mills ratio gives the observed, rather than expected, bread.
        mills = np.exp(stats.norm.logpdf(eta) - log_probability)
        unit_score = sign * mills
        curvature = mills * (mills + sign * eta)
    else:
        p = np.exp(eta)
        objective = np.sum(w * (p - y * eta))
        unit_score, curvature = y - p, p
    scores = (w * unit_score)[:, None] * x
    bread = x.T @ ((w * curvature)[:, None] * x)
    return objective, scores, bread


def oracle(frame, family, *, intercept=True, domain=False):
    """Weighted score root plus complete-design PSU Taylor covariance."""
    selected = np.ones(len(frame), dtype=bool)
    if domain:
        selected &= frame.domain.to_numpy(dtype=bool)
    selected &= frame[[outcome(family), "x", "z"]].notna().all(axis=1).to_numpy()
    x = matrix(frame.fillna(0), intercept)
    xs = x[selected]
    ys = frame[outcome(family)].to_numpy(dtype=float)[selected]
    ws = frame.w.to_numpy(dtype=float)[selected]
    if family == "regress":
        beta = np.linalg.solve(xs.T @ (ws[:, None] * xs), xs.T @ (ws * ys))
    else:
        def objective(beta):
            value, scores, _ = likelihood_parts(family, beta, xs, ys, ws)
            return value / ws.sum(), -scores.sum(0) / ws.sum()

        fit = optimize.minimize(
            objective, np.zeros(xs.shape[1]), jac=True, method="BFGS",
            options={"gtol": 1e-11, "maxiter": 500},
        )
        polished = optimize.root(
            lambda beta: likelihood_parts(family, beta, xs, ys, ws)[1].sum(0),
            fit.x,
            jac=lambda beta: -likelihood_parts(family, beta, xs, ys, ws)[2],
            tol=1e-11,
        )
        beta = polished.x
        assert np.linalg.norm(objective(beta)[1], ord=np.inf) < 1e-9
    _, scores, bread = likelihood_parts(family, beta, xs, ys, ws)
    inverse = np.linalg.inv(bread)
    row_scores = np.zeros((len(frame), len(beta)))
    row_scores[selected] = scores
    meat = np.zeros((len(beta), len(beta)))
    psu_scores = []
    for stratum in pd.unique(frame.h):
        hm = frame.h.to_numpy() == stratum
        psus = pd.unique(frame.loc[hm, "p"])
        total = np.array([
            row_scores[hm & (frame.p.to_numpy() == psu)].sum(0) for psu in psus
        ])
        psu_scores.extend(total)
        correction = 1 - len(psus) / float(frame.loc[hm, "N"].iloc[0]) if "N" in frame else 1
        if correction:
            centered = total - total.mean(0)
            meat += correction * len(psus) / (len(psus) - 1) * centered.T @ centered
    covariance = inverse @ meat @ inverse.T
    return beta, covariance, selected, np.asarray(psu_scores)


def assert_inference(table, estimates, covariance, df, *, alpha=0.05, null=0):
    estimates, covariance = np.asarray(estimates), np.asarray(covariance)
    np.testing.assert_allclose(table.estimate, estimates, atol=2e-9, rtol=2e-8)
    se = np.sqrt(np.maximum(covariance.diagonal(), 0))
    np.testing.assert_allclose(table.std_error, se, atol=2e-9, rtol=2e-8)
    if df:
        critical = stats.t.ppf(1 - alpha / 2, df)
        np.testing.assert_allclose(table.ci_low, estimates - critical * se, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(table.ci_high, estimates + critical * se, atol=2e-8, rtol=2e-8)
        positive = se > 1e-12
        np.testing.assert_allclose(
            table.loc[positive, "p_value"].to_numpy(dtype=float),
            2 * stats.t.sf(np.abs((estimates[positive] - np.asarray(null)) / se[positive]), df),
            atol=2e-9, rtol=2e-8,
        )
    else:
        assert table.p_value.isna().all()
        assert table.statistic.isna().all()


@pytest.mark.parametrize("family,beta,variance", [
    ("regress", 43 / 8, 25 / 64),
    ("logit", 0.0, 1 / 2),
    ("probit", 0.0, np.pi / 16),
    ("poisson", np.log(9 / 4), 5 / 81),
])
def test_intercept_only_hand_calculated_psu_sandwich(family, beta, variance):
    frame = pd.DataFrame({
        "w": [2.0] * 8, "h": ["a"] * 4 + ["b"] * 4,
        "p": [1, 1, 2, 2, 1, 1, 2, 2],
        "continuous": [1, 3, 2, 5, 6, 8, 7, 11],
        "binary": [0, 0, 0, 1, 1, 0, 1, 1],
        "count": [0, 1, 1, 2, 2, 3, 4, 5],
    })
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    actual = getattr(oe, "survey_" + family)(frame, design, outcome(family), [])
    assert actual.labels == ("_cons",)
    np.testing.assert_allclose(actual.coefficients, [beta], atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(actual.covariance, [[variance]], atol=1e-10, rtol=1e-9)
    assert_inference(actual.to_frame(), [beta], [[variance]], 2)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize("fpc", [False, True])
def test_weighted_likelihood_full_psu_covariance_and_t_inference(family, intercept, fpc):
    frame, design = fixture(fpc=fpc)
    result = getattr(oe, "survey_" + family)(
        frame, design, outcome(family), ["x", "z"], intercept=intercept, alpha=0.1,
        null=0.15,
    )
    beta, covariance, selected, _ = oracle(frame, family, intercept=intercept)
    np.testing.assert_allclose(result.coefficients, beta, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(result.covariance, covariance, atol=2e-9, rtol=2e-8)
    assert result.labels == (("_cons", "x", "z") if intercept else ("x", "z"))
    assert np.max(np.abs(covariance - np.diag(covariance.diagonal()))) > 1e-4
    assert result.df == 8
    assert result.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    assert result.metadata["n_design"] == 48
    assert result.metadata["n_used"] == 48
    assert_inference(result.to_frame(), beta, covariance, result.df, alpha=0.1, null=0.15)


@pytest.mark.parametrize("family", FAMILIES)
def test_domain_missing_physical_positions_and_zero_psu_retention(family):
    frame, design = fixture(fpc=True)
    # Duplicate/nonmonotone pandas labels do not define physical sample membership.
    frame.iloc[-4:, frame.columns.get_loc("x")] = np.nan
    frame.iloc[-4:, frame.columns.get_loc(outcome(family))] = np.nan
    frame.iloc[5, frame.columns.get_loc("z")] = np.nan
    fn = getattr(oe, "survey_" + family)
    with pytest.raises(AnalysisError):
        fn(frame, design, outcome(family), ["x", "z"], domain="domain")
    actual = fn(frame, design, outcome(family), ["x", "z"], domain="domain", missing="drop")
    beta, covariance, selected, psu_scores = oracle(frame, family, domain=True)
    np.testing.assert_allclose(actual.coefficients, beta, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(actual.covariance, covariance, atol=2e-9, rtol=2e-8)
    assert actual.df == 8
    assert actual.metadata["sample_positions"] == np.flatnonzero(selected).tolist()
    assert actual.metadata["outcome_exclusions"] == [5]
    assert actual.metadata["out_of_domain_positions"] == [44, 45, 46, 47]
    np.testing.assert_array_equal(psu_scores[-1], 0)
    np.testing.assert_array_equal(actual.metadata["psu_score_sums"][-1], [0, 0, 0])
    assert actual.metadata["n_design"] == 48 and actual.metadata["n_used"] == 43


@pytest.mark.parametrize("family", FAMILIES)
def test_global_sampling_weight_scale_cancels_from_coefficients_and_covariance(family):
    frame, design = fixture()
    fn = getattr(oe, "survey_" + family)
    before = fn(frame, design, outcome(family), ["x", "z"])
    scaled = frame.assign(w=frame.w * 17.3)
    new_design = oe.survey_design(scaled, weights="w", psu="p", strata="h")
    after = fn(scaled, new_design, outcome(family), ["x", "z"])
    np.testing.assert_allclose(after.coefficients, before.coefficients, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(after.covariance, before.covariance, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(after.to_frame(), before.to_frame(), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
def test_census_and_explicit_singleton_certainty_have_zero_design_covariance(family):
    frame, _ = fixture()
    fn = getattr(oe, "survey_" + family)
    census = frame.assign(N=3)
    d1 = oe.survey_design(census, weights="w", psu="p", strata="h", fpc="N")
    r1 = fn(census, d1, outcome(family), ["x", "z"])
    np.testing.assert_array_equal(r1.covariance, np.zeros((3, 3)))
    assert r1.to_frame().p_value.isna().all()
    certainty = frame.assign(p=1, N=1)
    d2 = oe.survey_design(
        certainty, weights="w", psu="p", strata="h", fpc="N", singleton="certainty"
    )
    r2 = fn(certainty, d2, outcome(family), ["x", "z"])
    assert r2.df == 0
    np.testing.assert_array_equal(r2.covariance, np.zeros((3, 3)))
    assert_inference(r2.to_frame(), r2.coefficients, r2.covariance, 0)
    np.testing.assert_allclose(r2.to_frame().ci_low, r2.coefficients)
    np.testing.assert_allclose(r2.to_frame().ci_high, r2.coefficients)


@pytest.mark.parametrize("family", FAMILIES)
def test_json_restore_without_original_sample_and_immutable_integrity(family):
    frame, design = fixture()
    fitted = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    cls = oe.SurveyRegressionResult
    restored = cls.model_validate_json(fitted.model_dump_json())
    assert restored == cls.model_validate(fitted.model_dump(mode="json"))
    pd.testing.assert_frame_equal(restored.to_frame(), fitted.to_frame())
    with pytest.raises((ValueError, TypeError, AttributeError)):
        restored.coefficients = (1.0, 2.0, 3.0)
    for field in ["coefficients", "covariance", "alpha", "df"]:
        corrupted = deepcopy(fitted.model_dump(mode="json"))
        if field == "coefficients":
            corrupted[field][0] += 0.01
        elif field == "covariance":
            corrupted[field][0][0] += 0.01
        elif field == "alpha":
            corrupted[field] = 0.2
        else:
            corrupted[field] += 1
        with pytest.raises(ValueError):
            cls.model_validate(corrupted)
    mutated = cls.model_validate_json(fitted.model_dump_json())
    mutated.metadata["sample_positions"][0] = 100000
    with pytest.raises(ValueError):
        mutated.to_frame()
    for helper in [
        lambda: oe.survey_predict(mutated, pd.DataFrame({"x": [0.2], "z": [0.1]})),
        lambda: oe.survey_margins(mutated, pd.DataFrame({"x": [0.2], "z": [0.1]})),
        lambda: oe.survey_lincom(mutated, [1, 0, 0]),
        lambda: oe.survey_test(mutated, [[1, 0, 0]]),
    ]:
        with pytest.raises((ValueError, AnalysisError)):
            helper()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["linear", "response"])
def test_prediction_of_new_rows_uses_full_saved_covariance(family, kind):
    frame, design = fixture()
    fitted = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    restored = oe.SurveyRegressionResult.model_validate_json(fitted.model_dump_json())
    new = pd.DataFrame({"x": [-1.1, 0.2, 1.5], "z": [0.8, -0.4, 0.1]}, index=[7, 7, -3])
    beta, covariance, _, _ = oracle(frame, family)
    x = matrix(new)
    eta = x @ beta
    gradient = x if kind == "linear" else response_derivative(family, eta)[:, None] * x
    estimates = eta if kind == "linear" else response(family, eta)
    expected_cov = gradient @ covariance @ gradient.T
    actual = oe.survey_predict(restored, new, kind=kind, alpha=0.1)
    assert actual.index.tolist() == [7, 7, -3]
    assert actual.attrs["physical_positions"] == [0, 1, 2]
    np.testing.assert_allclose(actual.attrs["covariance_matrix"], expected_cov, atol=2e-9, rtol=2e-8)
    assert actual.attrs["survey_regression_state"] == restored.model_dump(mode="json")
    assert_inference(actual, estimates, expected_cov, restored.df, alpha=0.1)


@pytest.mark.parametrize("family", FAMILIES)
def test_prediction_missing_rows_preserve_physical_positions(family):
    frame, design = fixture()
    fitted = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    new = pd.DataFrame({"x": [-1.0, np.nan, 0.5], "z": [0.2, 0.2, -0.3]}, index=[6, 6, 9])
    with pytest.raises(AnalysisError):
        oe.survey_predict(fitted, new)
    actual = oe.survey_predict(fitted, new, missing="drop")
    complete = oe.survey_predict(fitted, new.iloc[[0, 2]])
    assert actual.index.tolist() == [6, 9]
    assert actual.attrs["physical_positions"] == [0, 2]
    np.testing.assert_allclose(actual.to_numpy(dtype=float), complete.to_numpy(dtype=float))
    np.testing.assert_allclose(actual.attrs["covariance_matrix"], complete.attrs["covariance_matrix"])


def finite_difference(function, beta):
    """Symmetric coefficient perturbation independent of implementation gradients."""
    columns = []
    for j in range(len(beta)):
        step = np.zeros_like(beta)
        step[j] = 2e-5
        columns.append((np.asarray(function(beta + step)) - function(beta - step)) / (4e-5))
    return np.column_stack(columns)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("ame", [False, True])
def test_average_predictions_and_continuous_margins_delta_full_covariance(family, weighted, ame):
    frame, design = fixture()
    fitted = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    restored = oe.SurveyRegressionResult.model_validate_json(fitted.model_dump_json())
    new = pd.DataFrame({"x": [-1.2, -0.2, 0.7, 1.8], "z": [0.5, -0.1, 0.3, 1.0], "a": [1, 3, 2, 4]})
    at = {"z": 0.25}
    fixed = new.assign(**at)
    x = matrix(fixed)
    weight = new.a.to_numpy(dtype=float) if weighted else np.ones(len(new))
    weight /= weight.sum()
    beta, covariance, _, _ = oracle(frame, family)
    variables = ["x", "z"] if ame else None

    def estimand(coefficients):
        eta = x @ coefficients
        if not ame:
            return np.array([weight @ response(family, eta)])
        # SciPy link derivatives avoid cancellation from nested finite differences;
        # coefficient uncertainty still uses an independent numerical Jacobian.
        return (weight @ response_derivative(family, eta)) * coefficients[[1, 2]]

    estimates = estimand(beta)
    if ame:
        covariate_differences = []
        for j in [1, 2]:
            delta = np.zeros_like(x)
            delta[:, j] = 1e-5
            derivative = (
                response(family, (x + delta) @ beta) - response(family, (x - delta) @ beta)
            ) / (2e-5)
            covariate_differences.append(weight @ derivative)
        np.testing.assert_allclose(estimates, covariate_differences, atol=2e-10, rtol=2e-9)
    gradient = finite_difference(estimand, beta)
    expected_cov = gradient @ covariance @ gradient.T
    actual = oe.survey_margins(
        restored, new, variables=variables, at=at, weights="a" if weighted else None, alpha=0.1,
    )
    np.testing.assert_allclose(actual.attrs["covariance_matrix"], expected_cov, atol=2e-8, rtol=2e-6)
    assert actual.attrs["survey_regression_state"] == restored.model_dump(mode="json")
    assert_inference(actual, estimates, expected_cov, restored.df, alpha=0.1)


@pytest.mark.parametrize("family", FAMILIES)
def test_saved_linear_contrasts_and_survey_adjusted_joint_wald(family):
    frame, design = fixture()
    fit = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    restored = oe.SurveyRegressionResult.model_validate_json(fit.model_dump_json())
    beta, covariance, _, _ = oracle(frame, family)
    a = np.array([0.25, 1.0, -1.5])
    actual = oe.survey_lincom(restored, a.tolist(), null=0.2, alpha=0.1)
    mapped = oe.survey_lincom(restored, dict(zip(restored.labels, a)), null=0.2, alpha=0.1)
    pd.testing.assert_frame_equal(actual, mapped)
    variance = np.array([[a @ covariance @ a]])
    assert_inference(actual, [a @ beta], variance, 8, alpha=0.1, null=0.2)
    np.testing.assert_allclose(actual.attrs["covariance_matrix"], variance, atol=2e-9, rtol=2e-8)
    restrictions = np.array([[0, 1, 0], [0, 0, 1]])
    null = np.array([0.1, -0.1])
    difference = restrictions @ beta - null
    rcov = restrictions @ covariance @ restrictions.T
    wald = difference @ np.linalg.solve(rcov, difference)
    q, df = len(restrictions), restored.df
    statistic = (df - q + 1) * wald / (q * df)
    joint = oe.survey_test(restored, restrictions.tolist(), null=null.tolist()).iloc[0]
    assert joint.wald_chi2 == pytest.approx(wald, abs=2e-8, rel=2e-8)
    assert joint.statistic == pytest.approx(statistic, abs=2e-8, rel=2e-8)
    assert joint.df_num == q and joint.df_den == df - q + 1
    assert joint.p_value == pytest.approx(stats.f.sf(statistic, q, df - q + 1), abs=2e-9, rel=2e-8)
    one = oe.survey_test(restored, [a.tolist()], null=[0.2]).iloc[0]
    assert one.statistic == pytest.approx(actual.iloc[0].statistic**2, abs=2e-8, rel=2e-8)
    assert one.p_value == pytest.approx(actual.iloc[0].p_value, abs=2e-9, rel=2e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("options", [
    {"intercept": 1}, {"alpha": 0}, {"alpha": 1}, {"alpha": float("nan")},
    {"missing": "omit"}, {"max_iter": 0}, {"max_iter": True},
    {"tolerance": 0}, {"tolerance": float("inf")}, {"null": [0, 1]},
])
def test_explicit_option_admission(family, options):
    frame, design = fixture()
    with pytest.raises(AnalysisError):
        getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"], **options)


@pytest.mark.parametrize("family", FAMILIES)
def test_invalid_columns_rank_domains_and_changed_design(family):
    frame, design = fixture()
    fn = getattr(oe, "survey_" + family)
    for regressors in [["x", "x"], ["absent"], ["constant", "x"], ["x", "duplicate"]]:
        bad = frame.assign(constant=1.0, duplicate=frame.x * 2)
        with pytest.raises(AnalysisError):
            fn(bad, design, outcome(family), regressors)
    for values in [[0] * len(frame), [1] * 47 + [0.5], [1] * 47 + [np.nan]]:
        bad = frame.assign(domain=values)
        with pytest.raises(AnalysisError):
            fn(bad, design, outcome(family), ["x", "z"], domain="domain")
    changed = frame.copy()
    changed.iloc[0, changed.columns.get_loc("w")] *= 2
    with pytest.raises(AnalysisError) as exc:
        fn(changed, design, outcome(family), ["x", "z"])
    assert exc.value.code == "survey_design_changed"
    bad = frame.copy()
    bad.iloc[2, bad.columns.get_loc("x")] = np.inf
    with pytest.raises(AnalysisError):
        fn(bad, design, outcome(family), ["x", "z"])


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_binary_outcome_and_complete_separation_are_rejected(family):
    frame, design = fixture()
    fn = getattr(oe, "survey_" + family)
    bad = frame.assign(binary=0)
    with pytest.raises(AnalysisError):
        fn(bad, design, "binary", ["x", "z"])
    bad = frame.copy()
    bad.iloc[0, bad.columns.get_loc("binary")] = 2
    with pytest.raises(AnalysisError):
        fn(bad, design, "binary", ["x", "z"])
    separated = frame.assign(binary=(frame.x > 0).astype(int))
    with pytest.raises(AnalysisError):
        fn(separated, design, "binary", "x")


@pytest.mark.parametrize("family", ["logit", "probit"])
def test_quasi_separation_with_mixed_classes_on_the_boundary_is_rejected(family):
    frame = pd.DataFrame({
        "w": [1.0] * 6, "p": range(6), "x": [-2, -1, 0, 0, 1, 2],
        "y": [0, 0, 0, 1, 1, 1],
    })
    # Direction (intercept=0, slope=1) has nonnegative signed margins, some strict.
    signed_margins = (2 * frame.y.to_numpy() - 1) * frame.x.to_numpy()
    np.testing.assert_array_equal(signed_margins, [2, 1, 0, 0, 1, 2])
    design = oe.survey_design(frame, weights="w", psu="p")
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, "survey_" + family)(frame, design, "y", "x")
    assert exc.value.code == "survey_regression_separation"


def test_poisson_recession_direction_needs_a_combination_of_regressors():
    frame = pd.DataFrame({
        "w": [1.0] * 4, "p": range(4), "x": [1, 2, 1, 2],
        "z": [1, 2, 0, 0], "y": [1, 2, 0, 0],
    })
    # Neither column alone is the separating direction d=(-1,1).
    direction = frame[["x", "z"]].to_numpy() @ np.array([-1, 1])
    np.testing.assert_array_equal(direction, [0, 0, -1, -2])
    design = oe.survey_design(frame, weights="w", psu="p")
    with pytest.raises(AnalysisError) as exc:
        oe.survey_poisson(frame, design, "y", ["x", "z"], intercept=False)
    assert exc.value.code == "survey_regression_separation"


@pytest.mark.parametrize("family,variance", [
    ("logit", 8 / 15), ("probit", np.pi / 15), ("poisson", 2 / 15),
])
def test_no_intercept_all_zero_response_can_have_an_identified_finite_optimum(family, variance):
    frame = pd.DataFrame({"w": [1.0] * 4, "p": range(4), "x": [-2, -1, 1, 2], "y": 0})
    # At beta=0 the balanced score sums to zero and curvature is strictly positive.
    design = oe.survey_design(frame, weights="w", psu="p")
    fitted = getattr(oe, "survey_" + family)(frame, design, "y", "x", intercept=False)
    assert fitted.labels == ("x",)
    np.testing.assert_allclose(fitted.coefficients, [0.0], atol=1e-12)
    np.testing.assert_allclose(fitted.covariance, [[variance]], atol=1e-12, rtol=1e-11)
    assert fitted.df == 3
    assert_inference(fitted.to_frame(), [0], [[variance]], 3)


def test_poisson_invalid_response_boundary_and_convergence_failure():
    frame, design = fixture()
    for value in [-1.0, 0.5, float("inf"), float(2**53 + 2)]:
        bad = frame.copy()
        bad["count"] = bad["count"].astype(float)
        bad.iloc[0, bad.columns.get_loc("count")] = value
        with pytest.raises(AnalysisError):
            oe.survey_poisson(bad, design, "count", ["x", "z"])
    with pytest.raises(AnalysisError):
        oe.survey_poisson(frame.assign(count=0), design, "count", ["x", "z"])
    for family in ["logit", "probit", "poisson"]:
        with pytest.raises(AnalysisError):
            getattr(oe, "survey_" + family)(
                frame, design, outcome(family), ["x", "z"], max_iter=1,
            )


def test_original_integer_count_above_exact_float64_limit_is_rejected_before_rounding():
    frame, design = fixture()
    frame["count"] = [2**53 + 1, *frame["count"].tolist()[1:]]
    assert int(frame["count"].iloc[0]) == 2**53 + 1
    assert float(frame["count"].iloc[0]) == float(2**53)
    before = frame.copy(deep=True)
    with pytest.raises(AnalysisError) as exc:
        oe.survey_poisson(frame, design, "count", ["x", "z"])
    assert exc.value.code == "invalid_survey_response"
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_survey_apis_ignore_ambient_meta_default_device_and_restore_it(family):
    frame, design = fixture()
    original = frame.copy(deep=True)
    fn = getattr(oe, "survey_" + family)
    expected = fn(frame, design, outcome(family), ["x", "z"])
    new = pd.DataFrame({"x": [0.2, -0.5], "z": [0.1, 0.8]})
    expected_prediction = oe.survey_predict(expected, new)
    expected_margins = oe.survey_margins(expected, new, variables=["x", "z"])
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = fn(frame, design, outcome(family), ["x", "z"])
        assert torch.get_default_device().type == "meta"
        restored = oe.SurveyRegressionResult.model_validate_json(actual.model_dump_json())
        np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-12)
        np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-12)
        assert actual.metadata["device"] == "cpu"
        pd.testing.assert_frame_equal(restored.to_frame(), expected.to_frame())
        pd.testing.assert_frame_equal(oe.survey_predict(restored, new), expected_prediction)
        pd.testing.assert_frame_equal(
            oe.survey_margins(restored, new, variables=["x", "z"]), expected_margins
        )
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(previous)
    assert torch.get_default_device() == previous
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("family", FAMILIES)
def test_mapping_materializes_required_roles_only_and_preserves_original_values(family):
    class HostileUnrelatedColumn:
        def __len__(self):
            raise AssertionError("An unrelated column was sized during scoped admission.")

        def __iter__(self):
            raise AssertionError("An unrelated column was iterated during scoped admission.")

        def __array__(self, *args, **kwargs):
            raise AssertionError("An unrelated column was materialized during scoped admission.")

    frame, design = fixture()
    before = frame.copy(deep=True)
    required = frame.to_dict("list")
    snapshot = deepcopy(required)
    hostile = HostileUnrelatedColumn()
    mapping = {**required, "not_used": hostile}
    expected = getattr(oe, "survey_" + family)(frame, design, outcome(family), ["x", "z"])
    actual = getattr(oe, "survey_" + family)(mapping, design, outcome(family), ["x", "z"])
    np.testing.assert_allclose(actual.coefficients, expected.coefficients, atol=1e-12)
    np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-12)
    assert {key: mapping[key] for key in required} == snapshot
    assert mapping["not_used"] is hostile
    new_required = {"x": [0.2, -0.5], "z": [0.1, 0.8], "a": [1.0, 2.0]}
    new_snapshot = deepcopy(new_required)
    new_mapping = {**new_required, "not_used": hostile}
    new_frame = pd.DataFrame(new_required)
    pd.testing.assert_frame_equal(
        oe.survey_predict(actual, new_mapping), oe.survey_predict(expected, new_frame)
    )
    pd.testing.assert_frame_equal(
        oe.survey_margins(actual, new_mapping, variables=["x", "z"], weights="a"),
        oe.survey_margins(expected, new_frame, variables=["x", "z"], weights="a"),
    )
    assert {key: new_mapping[key] for key in new_required} == new_snapshot
    assert new_mapping["not_used"] is hostile
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("representation", ["columns", "rows"])
@pytest.mark.parametrize(
    "domain",
    [
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-mapping"),
        pytest.param(["domain"], id="column-list"),
        pytest.param({"column": "domain"}, id="column-mapping"),
        pytest.param("", id="empty-name"),
        pytest.param(False, id="boolean"),
    ],
)
def test_invalid_domain_roles_are_refused_before_mapping_or_row_projection(
    family, representation, domain,
):
    frame, design = fixture()
    data = frame.to_dict("list" if representation == "columns" else "records")
    with pytest.raises(AnalysisError) as exc:
        getattr(oe, "survey_" + family)(
            data, design, outcome(family), ["x", "z"], domain=domain,
        )
    assert exc.value.code == "invalid_survey_domain"
    assert str(exc.value) == "domain must name a complete Boolean/0-1 indicator column."


def test_fit_and_prediction_admission_limits_are_enforced_before_joint_allocation():
    frame, design = fixture()
    many = frame.assign(**{f"v{i}": frame.x + i * frame.z for i in range(33)})
    with pytest.raises(AnalysisError):
        oe.survey_regress(many, design, "continuous", [f"v{i}" for i in range(33)])
    fit = oe.survey_regress(frame, design, "continuous", ["x", "z"])
    with pytest.raises(AnalysisError):
        oe.survey_predict(fit, pd.DataFrame({"x": np.zeros(257), "z": np.zeros(257)}))
    with pytest.raises(AnalysisError):
        oe.survey_predict(fit, pd.DataFrame({"x": [np.nan], "z": [0]}), missing="drop")
    with pytest.raises(AnalysisError):
        oe.survey_predict(fit, pd.DataFrame({"x": [0], "z": [0]}), kind="unknown")
    big = pd.concat([frame] * 125, ignore_index=True)
    big_design = oe.survey_design(big, weights="w", psu="p", strata="h")
    with pytest.raises(AnalysisError) as exc:
        oe.survey_logit(big, big_design, "binary", ["x", "z"], max_iter=1000)
    assert exc.value.code == "survey_regression_budget"


def test_saved_helper_option_rank_and_weight_failures():
    frame, design = fixture()
    fit = oe.survey_regress(frame, design, "continuous", ["x", "z"])
    new = pd.DataFrame({"x": [0.2, 0.7], "z": [0.1, -0.2], "w": [1.0, -1.0]})
    for kwargs in [
        {"weights": "w"}, {"weights": "missing"}, {"variables": ["missing"]},
        {"variables": ["x", "x"]}, {"at": {"missing": 1}}, {"at": {"x": float("inf")}},
    ]:
        with pytest.raises(AnalysisError):
            oe.survey_margins(fit, new, **kwargs)
    for coefficients in [[1, 0], {"missing": 1}, [0, float("nan"), 0]]:
        with pytest.raises(AnalysisError):
            oe.survey_lincom(fit, coefficients)
    for restrictions in [
        [[0, 1, 0], [0, 2, 0]], [[0, 0, 0]], [[0, 1]], [[0, float("nan"), 0]],
    ]:
        with pytest.raises(AnalysisError):
            oe.survey_test(fit, restrictions)
    census = frame.assign(N=3)
    zero = oe.survey_regress(
        census, oe.survey_design(census, weights="w", psu="p", strata="h", fpc="N"),
        "continuous", ["x", "z"],
    )
    with pytest.raises(AnalysisError):
        oe.survey_test(zero, [[0, 1, 0]])
    few = frame.iloc[:12].copy()
    two_df = oe.survey_regress(
        few, oe.survey_design(few, weights="w", psu="p", strata="h"),
        "continuous", ["x", "z"],
    )
    assert two_df.df == 2
    with pytest.raises(AnalysisError):
        oe.survey_test(two_df, np.eye(3).tolist())


def test_rehashed_saved_state_still_requires_valid_geometry_and_psd_covariance():
    frame, design = fixture()
    fit = oe.survey_regress(frame, design, "continuous", ["x", "z"])
    for kind in [
        "df", "sample", "asymmetry", "negative_variance", "replay", "convergence",
        "score_replay", "sensitivity_replay",
    ]:
        payload = deepcopy(fit.model_dump(mode="json"))
        if kind == "df":
            payload["df"] += 1
        elif kind == "sample":
            payload["metadata"]["sample_positions"] = [1, 0, 2]
        elif kind == "asymmetry":
            payload["covariance"][0][1] += 0.3
        elif kind == "negative_variance":
            payload["covariance"][0][0] = -1.0
        elif kind == "replay":
            payload["covariance"][0][0] += 0.2
        elif kind == "convergence":
            payload["metadata"]["convergence"]["converged"] = False
        elif kind == "score_replay":
            payload["metadata"]["psu_score_sums"][0][0] += 1
        else:
            payload["metadata"]["sensitivity"][0][0] *= 2
        payload["integrity_sha256"] = hashlib.sha256(json.dumps(
            {key: value for key, value in payload.items() if key != "integrity_sha256"},
            sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ).encode()).hexdigest()
        with pytest.raises(ValueError):
            oe.SurveyRegressionResult.model_validate(payload)
