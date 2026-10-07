"""Numerical and failure-contract tests for the first estimator adapters."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats

from openecon.analysis import AnalysisError, capabilities, fit
from openecon.models import ModelSpec


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(90210)
    x = rng.normal(size=120)
    z = rng.normal(size=120)
    return pd.DataFrame({
        "y": 2 + 0.8 * x - 0.3 * z + rng.normal(size=120) * (1 + x**2),
        "x": x, "z": z, "group": np.repeat(np.arange(12), 10),
    })


def test_ols_matches_analytic_solution_and_student_t():
    data = pd.DataFrame({"x": [0., 1., 2., 3., 4.], "y": [1., 2., 1., 4., 5.]})
    result = fit(ModelSpec(outcome="y", predictors=["x"], covariance="nonrobust"), data=data)
    x = np.column_stack([np.ones(5), data.x])
    beta = np.linalg.solve(x.T @ x, x.T @ data.y)
    residuals = data.y.to_numpy() - x @ beta
    covariance = np.linalg.inv(x.T @ x) * (residuals @ residuals) / 3
    errors = np.sqrt(np.diag(covariance))
    assert_allclose([term.estimate for term in result.coefficients], beta, rtol=1e-12)
    assert_allclose(result.covariance_matrix, covariance, rtol=1e-12)
    assert_allclose([term.p_value for term in result.coefficients], 2 * stats.t.sf(abs(beta / errors), 3))
    assert result.inference["use_t"] is True
    assert result.inference["df_inference"] == 3
    assert result.sample_positions == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("covariance", ["HC0", "HC1", "HC2", "HC3", "nonrobust", "cluster"])
def test_ols_covariances_match_explicit_statsmodels(linear_data, covariance):
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance,
                     cluster="group" if covariance == "cluster" else None)
    actual = fit(spec, data=linear_data)
    kwargs = {"groups": linear_data.group, "use_correction": True, "df_correction": True} if covariance == "cluster" else {}
    expected = sm.OLS(linear_data.y, sm.add_constant(linear_data[["x", "z"]])).fit(
        cov_type=covariance, cov_kwds=kwargs, use_t=True
    )
    assert_allclose(actual.covariance_matrix, expected.cov_params(), rtol=1e-12, atol=1e-12)
    assert_allclose([c.p_value for c in actual.coefficients], expected.pvalues, rtol=1e-12)
    assert_allclose([[c.ci_low, c.ci_high] for c in actual.coefficients], expected.conf_int(), rtol=1e-12)
    assert actual.inference["df_inference"] == (11 if covariance == "cluster" else 117)


def test_cluster_matches_independent_sandwich_and_cluster_df(linear_data):
    result = fit(ModelSpec(outcome="y", predictors=["x", "z"], covariance="cluster", cluster="group"), data=linear_data)
    x = np.column_stack([np.ones(120), linear_data.x, linear_data.z])
    beta = np.linalg.lstsq(x, linear_data.y, rcond=None)[0]
    residuals = linear_data.y.to_numpy() - x @ beta
    bread = np.linalg.inv(x.T @ x)
    scores = np.array([x[linear_data.group == g].T @ residuals[linear_data.group == g] for g in range(12)])
    correction = 12 / 11 * 119 / 117
    covariance = correction * bread @ (scores.T @ scores) @ bread
    assert_allclose(result.covariance_matrix, covariance, rtol=1e-12)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.t.sf(abs(beta / np.sqrt(np.diag(covariance))), df=11))
    assert result.inference["small_sample_correction"] == pytest.approx(correction)


def test_missing_policy_records_exact_positional_sample(linear_data):
    linear_data.index = ["duplicate"] * len(linear_data)
    linear_data.iloc[4, linear_data.columns.get_loc("x")] = np.nan
    linear_data.iloc[8, linear_data.columns.get_loc("group")] = np.nan
    spec = ModelSpec(outcome="y", predictors=["x"], covariance="cluster", cluster="group")
    with pytest.raises(AnalysisError) as caught:
        fit(spec, data=linear_data)
    assert caught.value.code == "missing_values"
    result = fit(spec.model_copy(update={"missing": "drop"}), data=linear_data)
    assert result.nobs == 118
    assert result.dropped_rows == 2
    assert 4 not in result.sample_positions and 8 not in result.sample_positions
    assert result.predictions[4]["row"] == 5
    assert result.provenance["data_hash"] != result.provenance["sample_hash"]


def test_irrelevant_missing_column_does_not_drop_rows(linear_data):
    linear_data["unused"] = np.nan
    result = fit(ModelSpec(outcome="y", predictors=["x"]), data=linear_data)
    assert result.nobs == len(linear_data)


def test_categorical_encoding_preserves_explicit_reference(linear_data):
    linear_data["sector"] = pd.Categorical(np.tile(["b", "a", "c"], 40), categories=["c", "b", "a"])
    result = fit(ModelSpec(outcome="y", predictors=["x", "sector"], categorical=["sector"]), data=linear_data)
    assert [term.term for term in result.coefficients] == ["Intercept", "x", "sector[b]", "sector[a]"]
    assert result.provenance["categorical_encoding"]["sector"]["reference"] == "c"
    expected_design = pd.DataFrame({"const": 1., "x": linear_data.x,
                                   "b": (linear_data.sector == "b").astype(float), "a": (linear_data.sector == "a").astype(float)})
    expected = sm.OLS(linear_data.y, expected_design).fit(cov_type="nonrobust", use_t=True)
    assert_allclose([term.estimate for term in result.coefficients], expected.params)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-12, atol=1e-12)


def test_categorical_reference_not_silently_changed_by_missing_drop():
    data = pd.DataFrame({"y": [np.nan, 1, 3, 2, 6, 4], "sector": ["a", "b", "b", "c", "c", "c"]})
    result = fit(ModelSpec(outcome="y", predictors=["sector"], categorical=["sector"], missing="drop"), data=data)
    assert result.provenance["categorical_encoding"]["sector"]["reference"] == "a"
    assert result.provenance["omitted_terms"] == ["sector[c]"]
    assert [term.term for term in result.coefficients] == ["Intercept", "sector[b]"]
    assert result.sample_positions == [1, 2, 3, 4, 5]
    retained = data.iloc[1:]
    design = pd.DataFrame({"const": 1., "b": (retained.sector == "b").astype(float)})
    expected = sm.OLS(retained.y, design).fit(use_t=True)
    assert_allclose([term.estimate for term in result.coefficients], expected.params, rtol=1e-12)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-12)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("covariance", ["nonrobust", "cluster"])
def test_binary_models_match_statsmodels(estimator, covariance):
    rng = np.random.default_rng(176)
    x = rng.normal(size=500)
    probability = stats.norm.cdf(-0.2 + 0.8 * x)
    data = pd.DataFrame({"y": rng.binomial(1, probability).astype(bool), "x": x, "group": np.repeat(np.arange(50), 10)})
    spec = ModelSpec(estimator=estimator, outcome="y", predictors=["x"], covariance=covariance,
                     cluster="group" if covariance == "cluster" else None)
    result = fit(spec, data=data)
    model = sm.Logit if estimator == "logit" else sm.Probit
    kwargs = {"groups": data.group, "use_correction": True, "df_correction": True} if covariance == "cluster" else {}
    covariance_options = {"cov_kwds": kwargs} if kwargs else {}
    expected = model(data.y.astype(float), sm.add_constant(data[["x"]])).fit(disp=False, maxiter=100,
        tol=1e-8, cov_type=covariance, use_t=False, **covariance_options)
    assert_allclose([c.estimate for c in result.coefficients], expected.params, rtol=1e-10)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-10)
    assert_allclose([c.p_value for c in result.coefficients], expected.pvalues, rtol=1e-10)
    assert result.inference["use_t"] is False
    assert result.inference["df_inference"] is None
    assert len(result.predictions) == 400
    assert result.predictions[-1]["row"] == 499
    assert all(0 <= point["fitted"] <= 1 for point in result.predictions)


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("x,y", [([-3,-2,-1,1,2,3], [0,0,0,1,1,1]), ([-2,-1,0,0,1,2], [0,0,0,1,1,1])])
def test_complete_and_quasi_separation_are_rejected(estimator, x, y):
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(estimator=estimator, outcome="y", predictors=["x"], covariance="nonrobust"), data=pd.DataFrame({"x": x, "y": y}))
    assert caught.value.code == "separation_detected"


def test_nonconvergence_is_an_error(monkeypatch):
    import openecon.analysis as analysis
    monkeypatch.setattr(analysis, "_MAX_BINARY_ITERATIONS", 1)
    rng = np.random.default_rng(19)
    x = rng.normal(size=200)
    data = pd.DataFrame({"x": x, "y": rng.binomial(1, stats.norm.cdf(0.6 * x))})
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(estimator="logit", outcome="y", predictors=["x"], covariance="nonrobust"), data=data)
    assert caught.value.code == "nonconvergence"


@pytest.mark.parametrize("estimator", ["logit", "probit"])
@pytest.mark.parametrize("covariance", ["HC1", "HC3"])
def test_unsupported_binary_covariance_is_rejected(linear_data, estimator, covariance):
    with pytest.raises(ValidationError, match="Binary models support nonrobust or cluster"):
        ModelSpec(estimator=estimator, outcome="y", predictors=["x"], covariance=covariance)
    spec = ModelSpec(estimator=estimator, outcome="y", predictors=["x"], covariance="nonrobust")
    with pytest.raises(ValidationError, match="Binary models support nonrobust or cluster"):
        fit(spec.model_copy(update={"covariance": covariance}), data=linear_data)


@pytest.mark.parametrize("case,code", [
    ("constant_y", "constant_outcome"), ("infinity", "non_finite_values"),
    ("text", "non_numeric_column"), ("single_cluster", "insufficient_clusters"),
])
def test_invalid_analysis_fails_explicitly(linear_data, case, code):
    spec = ModelSpec(outcome="y", predictors=["x", "z"])
    if case == "constant_y":
        linear_data.y = 1
    elif case == "infinity":
        linear_data.loc[0, "x"] = np.inf
    elif case == "text":
        linear_data.x = linear_data.x.astype(str)
    elif case == "single_cluster":
        linear_data.group = 1
        spec = ModelSpec(outcome="y", predictors=["x"], covariance="cluster", cluster="group")
    with pytest.raises(AnalysisError) as caught:
        fit(spec, data=linear_data)
    assert caught.value.code == code


@pytest.mark.parametrize("case,omitted,retained", [("singular", "z", "x"), ("constant_x", "x", "z")])
def test_ols_omits_collinear_predictors_with_reduced_model_parity(linear_data, case, omitted, retained):
    if case == "singular":
        linear_data.z = 2 * linear_data.x
    else:
        linear_data.x = 1
    result = fit(ModelSpec(outcome="y", predictors=["x", "z"]), data=linear_data)
    expected = sm.OLS(linear_data.y, sm.add_constant(linear_data[[retained]])).fit(use_t=True)
    assert result.provenance["omitted_terms"] == [omitted]
    assert [term.term for term in result.coefficients] == ["Intercept", retained]
    assert_allclose([term.estimate for term in result.coefficients], expected.params, rtol=1e-12)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-12)
    assert_allclose([term.p_value for term in result.coefficients], expected.pvalues, rtol=1e-12)


def test_fractional_binary_outcome_rejected(linear_data):
    linear_data.y = np.linspace(0, 1, len(linear_data))
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(estimator="logit", outcome="y", predictors=["x"], covariance="nonrobust"), data=linear_data)
    assert caught.value.code == "invalid_binary_outcome"


@pytest.mark.parametrize("update", [
    {"predictors": ["x", "x"]}, {"predictors": ["y"]}, {"predictors": [], "intercept": False},
    {"categorical": ["z"]}, {"categorical": ["x", "x"]}, {"outcome": " "},
    {"covariance": "cluster"}, {"covariance": "nonrobust", "cluster": "group"},
    {"covariance": "cluster", "cluster": "y"}, {"alpha": 0}, {"alpha": 1},
    {"weights": "w"},
])
def test_spec_rejects_invalid_or_unsupported_fields(update):
    values = {"outcome": "y", "predictors": ["x"], **update}
    with pytest.raises(ValidationError):
        ModelSpec(**values)


def test_intercept_only_model_matches_classical_reference(linear_data):
    result = fit(ModelSpec(outcome="y", predictors=[]), data=linear_data)
    expected = sm.OLS(linear_data.y, np.ones((len(linear_data), 1))).fit(use_t=True)
    assert result.spec.covariance == "nonrobust"
    assert [term.term for term in result.coefficients] == ["Intercept"]
    assert result.provenance["omitted_terms"] == []
    assert result.metrics["df_model"] == 0
    assert result.inference["df_inference"] == len(linear_data) - 1
    assert_allclose([term.estimate for term in result.coefficients], expected.params, rtol=1e-12)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-12)
    assert_allclose([term.p_value for term in result.coefficients], expected.pvalues, rtol=1e-12)


def test_spec_cluster_column_selects_cluster_covariance(linear_data):
    spec = ModelSpec(outcome="y", predictors=["x", "z"], cluster="group")
    result = fit(spec, data=linear_data)
    explicit = fit(spec.model_copy(update={"covariance": "cluster"}), data=linear_data)
    assert spec.covariance == "cluster"
    assert_allclose(result.covariance_matrix, explicit.covariance_matrix, rtol=1e-12, atol=1e-12)


def test_no_intercept_and_json_roundtrip(linear_data):
    spec = ModelSpec(outcome="y", predictors=["x", "z"], intercept=False)
    result = fit(spec, data=linear_data)
    expected = sm.OLS(linear_data.y, linear_data[["x", "z"]], hasconst=False).fit(cov_type="nonrobust", use_t=True)
    assert_allclose([c.estimate for c in result.coefficients], expected.params)
    assert_allclose(result.covariance_matrix, expected.cov_params(), rtol=1e-12, atol=1e-12)
    saved = result.__class__.model_validate_json(result.model_dump_json())
    assert saved.model_dump(mode="json") == result.model_dump(mode="json")
    repeat = fit(spec, data=linear_data)
    assert result.provenance["data_hash"] == repeat.provenance["data_hash"]
    assert result.provenance["sample_hash"] == repeat.provenance["sample_hash"]
    assert result.id != repeat.id


def test_capability_contract_is_honest():
    supported = capabilities()
    assert supported["stata_parity_validated"] is False
    assert supported["out_of_core_estimation"] is True
    assert supported["streaming"]["row_limit"] is None
    assert "hundred_billion_rows_hardware_validated" not in supported["streaming"]
    assert supported["estimators"]["ols"]["covariances"] == [
        "nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "cluster_hc2", "cluster_hc3",
        "hac", "bootstrap", "jackknife",
    ]
    assert "HC3" not in supported["estimators"]["probit"]["covariances"]


def test_dummy_memory_guard_precedes_expansion(monkeypatch):
    import openecon.analysis as analysis
    monkeypatch.setattr(analysis, "_MAX_DESIGN_BYTES", 1000)
    data = pd.DataFrame({"y": np.arange(50), "category": [f"group-{i}" for i in range(50)]})
    monkeypatch.setattr(pd, "get_dummies", lambda *a, **kw: pytest.fail("Expansion happened before the memory guard."))
    with pytest.raises(AnalysisError) as caught:
        fit(ModelSpec(outcome="y", predictors=["category"], categorical=["category"]), data=data)
    assert caught.value.code == "insufficient_observations"
