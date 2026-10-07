"""Independent reference checks for OLS postestimation, without application state."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.frame import DataFrame
from openecon.linear_ols.design import OLSDesign
from openecon.linear_ols.postestimation import OLSPostestimation, _chi2_sf, f_sf


class Model(OLSPostestimation):
    pass


def tensor(value):
    return torch.tensor(np.asarray(value), dtype=torch.float64)


def make_model(frame, *, covariance="nonrobust", weight_type=None, design=None, original=None, positions=None):
    frame = frame.copy()
    if design is None:
        design = OLSDesign(["x", "z"], [], intercept=True).prepare(frame)
    x = design.encode(frame)
    raw_weights = frame["w"].to_numpy() if weight_type else np.ones(len(frame))
    weights = raw_weights * len(frame) / raw_weights.sum() if weight_type == "aweight" else raw_weights
    groups = np.arange(len(frame)) % 11
    options = ({"cov_type": "cluster", "cov_kwds": {"groups": groups}, "use_t": True}
               if covariance == "cluster" else {"cov_type": covariance, "use_t": True})
    if weight_type == "fweight":
        repeated = np.repeat(np.arange(len(frame)), raw_weights.astype(int))
        reference = sm.OLS(frame.y.to_numpy()[repeated], x.numpy()[repeated]).fit(**options)
        nobs = int(raw_weights.sum())
    else:
        reference = sm.WLS(frame.y.to_numpy(), x.numpy(), weights=weights).fit(**options)
        nobs = len(frame)
    parameters = tensor(reference.params)
    bread = torch.linalg.inv(x.T @ (tensor(weights)[:, None] * x))
    fitted = x @ parameters
    residual = tensor(frame.y) - fitted
    model = Model()
    model.spec = SimpleNamespace(alpha=.05, outcome="y", predictors=design.required,
                                 covariance=covariance, intercept=True, weights="w" if weight_type else None,
                                 weight_type=weight_type, time=None)
    df_inference = 10 if covariance == "cluster" else nobs - x.shape[1]
    model.inference = {"use_t": True, "df_inference": df_inference, "df_resid": nobs - x.shape[1]}
    model.coefficients = [SimpleNamespace(term=term, estimate=float(value))
                          for term, value in zip(design.terms, parameters, strict=True)]
    model.covariance_matrix = np.asarray(reference.cov_params()).tolist()
    model.metrics = {"ss_resid": float(torch.dot(tensor(weights), residual.square()))}
    model._state = {"x": x, "y": tensor(frame.y), "params": parameters,
                    "covariance": tensor(reference.cov_params()), "bread": bread, "fitted": fitted,
                    "resid": residual, "weights": tensor(weights),
                    "leverage": (x @ bread * x).sum(dim=1) * tensor(weights),
                    "terms": design.terms, "frame": frame, "design": design, "encode": design.encode,
                    "predictor_columns": design.required, "nobs": nobs,
                    "df_resid": nobs - x.shape[1], "df_inference": df_inference,
                    "sigma2": model.metrics["ss_resid"] / (nobs - x.shape[1]),
                    "positions": torch.arange(len(frame)) if positions is None else torch.tensor(positions),
                    "original_rows": len(frame) if original is None else len(original),
                    "original_frame": frame if original is None else original.copy()}

    def refit(augmented_x):
        result = sm.WLS(frame.y.to_numpy(), augmented_x.numpy(), weights=weights).fit(**options)
        return {"params": tensor(result.params), "covariance": tensor(result.cov_params()),
                "df_inference": 10 if covariance == "cluster" else nobs - augmented_x.shape[1]}

    model._state["refit"] = refit
    return model, reference


@pytest.fixture
def frame():
    rng = np.random.default_rng(20261003)
    n = 173
    values = rng.normal(size=(n, 2))
    return pd.DataFrame({"x": values[:, 0], "z": values[:, 1],
                         "y": .4 + .7 * values[:, 0] - .3 * values[:, 1] + rng.normal(size=n),
                         "w": rng.uniform(.3, 3, n), "g": np.resize(["A", "B", "C"], n)})


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
def test_single_and_joint_wald_use_the_fitted_covariance_and_degrees(frame, covariance):
    model, reference = make_model(frame, covariance=covariance)
    for restriction, matrix, value in [
        ("x", [[0, 1, 0]], 0), (["x", "z"], [[0, 1, 0], [0, 0, 1]], [0, 0]),
        ({"x": 1, "z": -1}, [[0, 1, -1]], .2),
        ([{"x": 1}, {"z": 2}], [[0, 1, 0], [0, 0, 2]], [0, .1]),
    ]:
        result = model.test(restriction, value)
        oracle = reference.f_test((np.asarray(matrix), np.broadcast_to(value, len(matrix))))
        assert result["distribution"] == "F"
        assert result["statistic"] == pytest.approx(float(oracle.fvalue), rel=3e-12)
        assert result["p_value"] == pytest.approx(float(oracle.pvalue), rel=3e-11)
        assert result["df_denom"] == (10 if covariance == "cluster" else 170)
    explicit = model.test(tensor([[0, 1, -1]]), .2)
    assert explicit["statistic"] == model.test({"x": 1, "z": -1}, .2)["statistic"]


def test_reset_infinite_resampling_df_uses_independent_chi_squared_oracle(frame):
    model, _ = make_model(frame)
    model.spec.covariance = "bootstrap"
    model.inference["use_t"] = False
    recorded = {}

    def resampling_refit(x):
        parameter = torch.linalg.lstsq(x, tensor(frame.y)).solution
        residual = tensor(frame.y) - x @ parameter
        bread = torch.linalg.inv(x.T @ x)
        covariance = bread @ (x.T @ (residual.square()[:, None] * x)) @ bread
        recorded.update(params=parameter, covariance=covariance)
        return {"params": parameter, "covariance": covariance, "df_inference": float("inf")}

    model._state["refit"] = resampling_refit
    result = model.reset_test(powers=2)
    expected = float(recorded["params"][-1].square() / recorded["covariance"][-1, -1])
    assert result["distribution"] == "chi2"
    assert result["statistic"] == pytest.approx(expected, rel=1e-12)
    assert result["p_value"] == pytest.approx(stats.chi2.sf(expected, 1), rel=1e-12)


@pytest.mark.parametrize("failure", ["singular_covariance", "bad_covariance_shape", "bad_estimate_shape", "bad_basis_shape"])
def test_stable_contrast_callbacks_preserve_structured_validation(frame, failure):
    model, _ = make_model(frame)
    if failure == "singular_covariance":
        model._state["contrast_covariance"] = lambda rows: torch.ones((len(rows), len(rows)), dtype=torch.float64)
    elif failure == "bad_covariance_shape":
        model._state["contrast_covariance"] = lambda rows: torch.ones((1, 1), dtype=torch.float64)
    elif failure == "bad_estimate_shape":
        model._state["prediction_fitted"] = lambda rows: torch.ones(1, dtype=torch.float64)
    else:
        model._state["contrast_basis"] = lambda rows: torch.ones((len(rows), 1), dtype=torch.float64)
    with pytest.raises(AnalysisError) as invalid:
        model.test(["x", "z"])
    assert invalid.value.code == ("nonestimable_restriction" if failure == "singular_covariance" else "invalid_postestimation")


def test_lincom_nlcom_delta_method_and_serialized_coefficients(frame):
    model, reference = make_model(frame)
    result = model.lincom({"x": 2, "z": -1}, constant=.1)
    gradient = np.array([0, 2, -1])
    estimate = gradient @ reference.params + .1
    se = np.sqrt(gradient @ reference.cov_params() @ gradient)
    assert result["estimate"] == pytest.approx(estimate)
    assert result["std_error"] == pytest.approx(se)
    assert result["p_value"] == pytest.approx(2 * stats.t.sf(abs(estimate / se), 170), rel=2e-12)
    nonlinear = model.nlcom(lambda b: b["x"] / b["z"], null=1)
    derivative = np.array([0, 1 / reference.params[2], -reference.params[1] / reference.params[2] ** 2])
    assert nonlinear["estimate"] == pytest.approx(reference.params[1] / reference.params[2])
    assert nonlinear["std_error"] == pytest.approx(np.sqrt(derivative @ reference.cov_params() @ derivative))
    del model._state
    assert model.lincom({"x": 2, "z": -1}, .1) == result
    assert model.nlcom(lambda b: b["x"] / b["z"], null=1) == nonlinear
    with pytest.raises(AnalysisError, match="retained estimation") as error:
        model.predict()
    assert error.value.code == "estimation_state_unavailable"


def test_normal_inference_uses_z_and_chi_squared(frame):
    model, reference = make_model(frame)
    model.inference["use_t"] = False
    result = model.lincom({"x": 1})
    assert result["distribution"] == "normal" and result["df"] is None
    assert result["p_value"] == pytest.approx(2 * stats.norm.sf(abs(result["statistic"])))
    joint = model.test(["x", "z"])
    oracle = reference.wald_test([[0, 1, 0], [0, 0, 1]], use_f=False, scalar=True)
    assert joint["distribution"] == "chi2"
    assert joint["statistic"] == pytest.approx(float(oracle.statistic))
    assert joint["p_value"] == pytest.approx(float(oracle.pvalue), rel=2e-12)


def test_contrast_specific_df_and_scale_apply_to_single_tests_and_delta_intervals(frame):
    model, _ = make_model(frame)
    gradients = []
    def adjustment(gradient):
        gradients.append(gradient.clone())
        return {"df": 7.25, "scale": 1.3}
    model._state["contrast_inference"] = adjustment
    result = model.lincom({"x": 1, "z": -1})
    assert result["df"] == 7.25 and result["statistic_scale"] == 1.3
    assert result["statistic"] == pytest.approx(1.3 * result["estimate"] / result["std_error"])
    assert result["ci_high"] == pytest.approx(result["estimate"] + stats.t.ppf(.975, 7.25) * result["std_error"] / 1.3)
    single = model.test({"x": 1, "z": -1})
    assert single["statistic"] == pytest.approx(result["statistic"] ** 2)
    assert single["p_value"] == pytest.approx(result["p_value"])
    assert single["df_denom"] == 7.25
    joint = model.test(["x", "z"])
    assert joint["df_denom"] == 170 and "conventional" in joint["df_adjustment_note"]
    assert len(gradients) == 2


@pytest.mark.parametrize("restriction", [[], {}, ["x", "x"], "absent", {"absent": 1},
                                           [[0, 1, 0], [0, 2, 0]], [[1, 0]], {"x": float("nan")}])
def test_invalid_or_nonestimable_restrictions_are_actionable(frame, restriction):
    model, _ = make_model(frame)
    with pytest.raises(AnalysisError):
        model.test(restriction)


@pytest.mark.parametrize("expression", [lambda b: float(b["x"]), lambda b: torch.sqrt(-b["x"].abs()),
                                          lambda b: b["unknown"], lambda b: torch.tensor(1.)])
def test_nonlinear_expression_requires_finite_torch_derivatives(frame, expression):
    model, _ = make_model(frame)
    with pytest.raises(AnalysisError) as error:
        model.nlcom(expression)
    assert error.value.code == "invalid_nlcom"


def test_all_classical_influence_values_match_independent_deletion_oracle(frame):
    model, reference = make_model(frame)
    influence = sm.OLS(frame.y, model._state["x"].numpy()).fit().get_influence()
    expected = {"xb": reference.fittedvalues, "residuals": reference.resid,
                "leverage": influence.hat_matrix_diag, "rstandard": influence.resid_studentized_internal,
                "rstudent": influence.resid_studentized_external, "cook": influence.cooks_distance[0],
                "covratio": influence.cov_ratio, "dfits": influence.dffits[0],
                "welsch": influence.dffits[0] * np.sqrt((len(frame) - 1) / (1 - influence.hat_matrix_diag)),
                "stdp": np.sqrt(np.sum(model._state["x"].numpy() * (model._state["x"].numpy() @ reference.cov_params()), axis=1)),
                "stdf": np.sqrt(reference.scale * (1 + influence.hat_matrix_diag)),
                "stdr": np.sqrt(reference.scale * (1 - influence.hat_matrix_diag))}
    for kind, oracle in expected.items():
        result = model.predict(kind=kind)
        assert isinstance(result, DataFrame) and len(result) == len(frame)
        np.testing.assert_allclose(result[kind], oracle, rtol=4e-11, atol=4e-11)
    dfbeta = model.predict(kind="dfbeta")
    np.testing.assert_allclose(dfbeta, influence.dfbetas, rtol=4e-11, atol=4e-11)
    assert list(model.predict(kind="dfbeta", term="x")) == ["dfbeta[x]"]


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3", "cluster"])
def test_new_data_mean_intervals_use_fitted_vce_and_student_t(frame, covariance):
    model, reference = make_model(frame, covariance=covariance)
    new = frame.iloc[:7].copy()
    new.index = pd.Index(["a", "a", "b", "c", "d", "e", "f"])
    result = model.predict(new, kind="mean", alpha=.1)
    oracle = reference.get_prediction(model._state["encode"](new).numpy()).summary_frame(alpha=.1)
    if covariance == "cluster":
        # statsmodels get_prediction uses residual df even though its fitted
        # coefficient tests use G-1. Apply the actual clustered t reference.
        critical = stats.t.ppf(.95, 10)
        oracle["mean_ci_lower"] = oracle["mean"] - critical * oracle["mean_se"]
        oracle["mean_ci_upper"] = oracle["mean"] + critical * oracle["mean_se"]
    assert result.index.equals(new.index)
    np.testing.assert_allclose(result.xb, oracle["mean"], rtol=2e-12)
    np.testing.assert_allclose(result.ci_low, oracle["mean_ci_lower"], rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(result.ci_high, oracle["mean_ci_upper"], rtol=3e-11, atol=3e-11)
    if covariance == "nonrobust":
        observation = model.predict(new, kind="obs", alpha=.1)
        np.testing.assert_allclose(observation.ci_low, oracle["obs_ci_lower"], rtol=3e-11)
        np.testing.assert_allclose(observation.ci_high, oracle["obs_ci_upper"], rtol=3e-11)
    else:
        with pytest.raises(AnalysisError) as error:
            model.predict(new, kind="obs")
        assert error.value.code == "unsupported_prediction_vce"


def test_original_missing_rows_and_duplicate_index_labels_are_preserved(frame):
    original = frame.iloc[:30].copy()
    original.index = pd.Index([f"row{i // 2}" for i in range(30)])
    original.iloc[2, original.columns.get_loc("y")] = np.nan
    original.iloc[5, original.columns.get_loc("x")] = np.nan
    keep = original[["y", "x", "z"]].notna().all(axis=1)
    model, _ = make_model(original.loc[keep], original=original, positions=np.flatnonzero(keep))
    predicted = model.predict()
    assert len(predicted) == len(original) and predicted.index.equals(original.index)
    assert pd.notna(predicted.iloc[2].xb) and pd.isna(predicted.iloc[5].xb)
    residual = model.predict(kind="residuals")
    assert pd.isna(residual.iloc[2].residuals) and pd.isna(residual.iloc[5].residuals)
    dfits = model.predict(kind="dfits")
    assert pd.isna(dfits.iloc[2].dfits) and pd.isna(dfits.iloc[5].dfits)


@pytest.mark.parametrize("kind", ["leverage", "rstandard", "rstudent", "cook", "stdf", "stdr",
                                     "covratio", "dfits", "welsch", "dfbeta"])
def test_stata_nonclassical_vce_prediction_restrictions(frame, kind):
    model, _ = make_model(frame, covariance="HC3")
    with pytest.raises(AnalysisError) as error:
        model.predict(kind=kind)
    assert error.value.code == "unsupported_prediction_vce"


def test_analytic_weights_allow_hat_but_reject_classical_deletion_and_forecast(frame):
    model, _ = make_model(frame, weight_type="aweight")
    expected = (model._state["x"] @ model._state["bread"] * model._state["x"]).sum(dim=1) * model._state["weights"]
    np.testing.assert_allclose(model.predict(kind="hat").leverage, expected, rtol=3e-12)
    for kind in ("stdf", "stdr", "rstudent", "cook", "dfbeta"):
        with pytest.raises(AnalysisError) as error:
            model.predict(kind=kind)
        assert error.value.code == "unsupported_prediction_weight"


def test_frequency_weight_predictions_match_explicit_replication(frame):
    frame = frame.iloc[:35].copy()
    frame["w"] = np.resize([1, 2, 3], len(frame))
    model, reference = make_model(frame, weight_type="fweight")
    influence = reference.get_influence()
    first_duplicate = np.r_[0, np.cumsum(frame.w)[:-1]].astype(int)
    for kind, expected in [("leverage", influence.hat_matrix_diag),
                           ("rstudent", influence.resid_studentized_external),
                           ("dfits", influence.dffits[0]), ("covratio", influence.cov_ratio)]:
        np.testing.assert_allclose(model.predict(kind=kind)[kind], expected[first_duplicate], rtol=5e-11, atol=5e-11)


def test_specification_tests_match_independent_reference(frame):
    model, reference = make_model(frame)
    for method, robust in [("normal", False), ("iid", True)]:
        expected = het_breuschpagan(reference.resid, reference.model.exog, robust=robust)
        actual = model.hettest(rhs=True, method=method)
        assert actual["statistic"] == pytest.approx(expected[0], rel=3e-11)
        assert actual["p_value"] == pytest.approx(expected[1], rel=3e-11)
    fitted_z = np.column_stack((np.ones(len(frame)), reference.fittedvalues))
    expected_bp = het_breuschpagan(reference.resid, fitted_z, robust=False)
    assert model.hettest()["statistic"] == pytest.approx(expected_bp[0], rel=3e-11)
    expected_white = het_white(reference.resid, reference.model.exog)
    assert model.white_test()["statistic"] == pytest.approx(expected_white[0], rel=3e-11)
    assert model.white_test()["p_value"] == pytest.approx(expected_white[1], rel=3e-11)
    expected_reset = linear_reset(reference, power=4, use_f=True)
    assert model.reset_test()["statistic"] == pytest.approx(float(expected_reset.fvalue), rel=5e-10)
    expected_bg = acorr_breusch_godfrey(reference, nlags=2)
    actual_bg = model.breusch_godfrey(lags=2)
    assert actual_bg["statistic"] == pytest.approx(expected_bg[0], rel=3e-11)
    assert actual_bg["p_value"] == pytest.approx(expected_bg[1], rel=3e-11)
    assert actual_bg["f_statistic"] == pytest.approx(expected_bg[2], rel=3e-11)
    actual_vif = model.vif()
    for row in actual_vif.itertuples():
        assert row.vif == pytest.approx(variance_inflation_factor(reference.model.exog, model._state["terms"].index(row.term)), rel=3e-11)
    result = model.diagnostics(lags=2)
    assert result["durbin_watson"]["statistic"] == pytest.approx(durbin_watson(reference.resid))
    expected_jb = jarque_bera(reference.resid)
    assert result["jarque_bera"]["statistic"] == pytest.approx(expected_jb[0], rel=3e-11)
    assert result["jarque_bera"]["p_value"] == pytest.approx(expected_jb[1], rel=3e-11)
    assert isinstance(result["influence"], DataFrame)


@pytest.mark.parametrize("covariance", ["HC3", "cluster"])
def test_reset_refits_with_same_robust_or_cluster_vce(frame, covariance):
    model, reference = make_model(frame, covariance=covariance)
    fitted = np.asarray(reference.fittedvalues)
    normalized = (fitted - fitted.min()) / (fitted.max() - fitted.min())
    x = np.column_stack([reference.model.exog, normalized ** 2, normalized ** 3, normalized ** 4])
    settings = ({"cov_type": "cluster", "cov_kwds": {"groups": np.arange(len(frame)) % 11}, "use_t": True}
                if covariance == "cluster" else {"cov_type": covariance, "use_t": True})
    oracle = sm.OLS(frame.y, x).fit(**settings).f_test(np.eye(6)[3:])
    actual = model.reset_test()
    assert actual["statistic"] == pytest.approx(float(oracle.fvalue), rel=8e-10)
    assert actual["p_value"] == pytest.approx(float(oracle.pvalue), rel=8e-10)


def test_marginal_effects_reencode_categories_interactions_and_polynomials(frame):
    design = OLSDesign(["x*g", "I(x**3)"], ["g"], intercept=True).prepare(frame)
    model, _ = make_model(frame, design=design)
    terms = design.terms
    result = model.margins(["x", "g"], _return_gradients=True)
    gradient = torch.zeros(len(terms), dtype=torch.float64)
    gradient[terms.index("x")] = 1
    gradient[terms.index("I(x**3)")] = 3 * float((frame.x ** 2).mean())
    for level in ("B", "C"):
        gradient[terms.index(f"x:g[{level}]")] = float((frame.g == level).mean())
    expected = model._estimate_contrast(float(gradient @ model._state["params"]), gradient)
    np.testing.assert_allclose(result.iloc[0][["estimate", "std_error"]].astype(float),
                               [expected["estimate"], expected["std_error"]], rtol=3e-10)
    assert len(result.attrs["delta_gradients"]) == len(result)
    np.testing.assert_allclose(result.attrs["delta_gradients"][0], gradient, rtol=3e-10, atol=3e-10)
    at = model.margins("x", at={"x": [0., 1.], "g": "B"})
    for setting, estimate in zip(at["at[x]"], at.estimate, strict=True):
        derivative = (model._state["params"][terms.index("x")]
                      + model._state["params"][terms.index("x:g[B]")]
                      + 3 * setting ** 2 * model._state["params"][terms.index("I(x**3)")])
        assert estimate == pytest.approx(float(derivative), rel=5e-10)
    mem = model.margins("x", method="mem")
    gradient[terms.index("I(x**3)")] = 3 * float(frame.x.mean()) ** 2
    assert mem.iloc[0].estimate == pytest.approx(float(gradient @ model._state["params"]), rel=5e-10)
    assert model.testparm("g")["df_num"] == 2
    assert model.testparm("x:g*")["df_num"] == 2


@pytest.mark.parametrize("df", [1, 2, 5, 30, 384, 2048])
@pytest.mark.parametrize("statistic", [0, 1e-8, 1, 20, 300, 2000])
def test_native_chi_squared_tail_matches_independent_scipy(df, statistic):
    assert _chi2_sf(statistic, df) == pytest.approx(stats.chi2.sf(statistic, df), rel=2e-11, abs=3e-14)


@pytest.mark.parametrize("df1", [1, 2, 3, 4, 9, 50])
@pytest.mark.parametrize("df2", [2, 13, 170, 1_000_000, 100_000_000_000])
@pytest.mark.parametrize("statistic", [.1, 1, 4, 20])
def test_native_f_tail_matches_independent_scipy_including_large_sample(df1, df2, statistic):
    assert f_sf(statistic, df1, df2) == pytest.approx(stats.f.sf(statistic, df1, df2), rel=2e-9, abs=3e-13)


def test_streamed_state_predicts_new_rows_and_observation_intervals_without_full_rows(frame):
    model, reference = make_model(frame)
    for key in ("x", "y", "resid", "fitted", "frame", "original_frame"):
        model._state.pop(key)
    new = frame.iloc[:3].copy()
    result = model.predict(new, interval="obs")
    expected = reference.get_prediction(model._state["encode"](new).numpy()).summary_frame()
    np.testing.assert_allclose(result.ci_low, expected.obs_ci_lower, rtol=3e-11)
    with pytest.raises(AnalysisError, match="iter_predict") as error:
        model.predict()
    assert error.value.code == "estimation_state_unavailable"


def test_log_margins_at_small_positive_covariates_respect_expression_domain(frame):
    frame = frame.copy()
    frame["x"] = np.geomspace(1e-10, 2., len(frame))
    design = OLSDesign(["log(x)", "z"], [], intercept=True).prepare(frame)
    model, _ = make_model(frame, design=design)
    result = model.margins("x")
    expected = float(model._state["params"][design.terms.index("log(x)")]) * float((1 / frame.x).mean())
    assert result.iloc[0].estimate == pytest.approx(expected, rel=5e-10)


@pytest.mark.parametrize("at", [{"absent": 1}, {"x": []}, {"x": float("inf")}, {"g": "absent"},
                                    {"x": list(range(1001))}])
def test_invalid_margins_grids_reject_before_evaluation(frame, at):
    design = OLSDesign(["x*g", "z"], ["g"], intercept=True).prepare(frame)
    model, _ = make_model(frame, design=design)
    with pytest.raises(AnalysisError):
        model.margins("g", at=at)


def test_complex_restrictions_are_not_silently_cast_to_real(frame):
    model, _ = make_model(frame)
    for restriction in ({"x": 1 + 2j}, np.array([[0, 1j, 0]]), torch.tensor([[0, 1j, 0]])):
        with pytest.raises(AnalysisError) as error:
            model.test(restriction)
        assert error.value.code == "invalid_postestimation"


def test_prediction_and_diagnostics_do_not_mutate_fitted_data_or_parameters(frame):
    model, _ = make_model(frame)
    original = model._state["frame"].copy(deep=True)
    parameters = model._state["params"].clone()
    model.margins("x", at={"x": 1.})
    model.predict(kind="dfbeta")
    model.diagnostics()
    pd.testing.assert_frame_equal(original, model._state["frame"])
    torch.testing.assert_close(parameters, model._state["params"], rtol=0, atol=0)


def test_adjusted_serialized_single_term_is_available_but_new_contrast_needs_refit(frame):
    model, _ = make_model(frame)
    model.inference.update(dfadjust=True, hansen=True,
                           coefficient_df=[5., 6., 7.], coefficient_scale=[1., 1.2, 1.3])
    del model._state
    result = model.lincom({"x": 2})
    assert result["df"] == 6. and result["statistic_scale"] == 1.2
    with pytest.raises(AnalysisError) as error:
        model.lincom({"x": 1, "z": 1})
    assert error.value.code == "estimation_state_unavailable"
