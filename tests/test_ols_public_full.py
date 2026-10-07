"""Independent public-contract tests for the comprehensive OLS family."""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.linear_ols.design import OLSDesign, parse_formula


@pytest.fixture
def frame():
    rng = np.random.default_rng(8641)
    data = pd.DataFrame({"x": rng.normal(size=160), "z": rng.normal(size=160),
                         "sector": np.tile(["a", "b", "c", "d"], 40),
                         "firm": np.repeat(np.arange(16), 10), "t": np.arange(160)})
    data["y"] = 2 + data.x - 0.3 * data.z + rng.normal(size=160)
    data["w"] = rng.uniform(0.1, 2, size=160)
    return data


def test_formula_compiler_matches_explicit_factorial_design(frame):
    outcome, terms, intercept = parse_formula("y ~ x * C(sector) + I(z**2)")
    design = OLSDesign(terms, [], intercept=intercept).prepare(frame)
    assert outcome == "y"
    assert design.required == ["x", "sector", "z"]
    x = design.encode(frame).numpy()
    categories = [(frame.sector == level).astype(float).to_numpy() for level in ["b", "c", "d"]]
    expected = np.column_stack([np.ones(160), frame.x, *categories,
                               *(frame.x * indicator for indicator in categories), frame.z**2])
    assert_allclose(x, expected)
    assert design.terms == ["Intercept", "x", "sector[b]", "sector[c]", "sector[d]",
                            "x:sector[b]", "x:sector[c]", "x:sector[d]", "I(z**2)"]


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "x[0]", "open(x)", "[x]", "C(x, 1)"])
def test_formula_never_evaluates_arbitrary_python(frame, expression):
    with pytest.raises(oe.AnalysisError):
        OLSDesign([expression], [], intercept=True).prepare(frame).encode(frame)


def test_formula_time_lags_respect_gaps_and_original_row_order():
    data = pd.DataFrame({"x": [5., 1., 2., 8.], "t": [5, 1, 2, 8]})
    design = OLSDesign(["L(x)", "D(x)"], [], intercept=False, time="t").prepare(data)
    actual = design.encode(data).numpy()
    assert_allclose(actual, [[np.nan, np.nan], [np.nan, np.nan], [1, 1], [np.nan, np.nan]], equal_nan=True)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "robust"])
def test_public_ols_matches_independent_estimator(frame, covariance):
    result = oe.ols(data=frame, y="y", x=["x", "z"], covariance=covariance)
    reference = sm.OLS(frame.y, sm.add_constant(frame[["x", "z"]])).fit(
        cov_type="HC1" if covariance == "robust" else covariance, use_t=True)
    assert_allclose([c.estimate for c in result.coefficients], reference.params, rtol=1e-11)
    assert_allclose(result.covariance_matrix, reference.cov_params(), rtol=1e-11)
    assert_allclose([c.p_value for c in result.coefficients], reference.pvalues, rtol=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(reference.scale), rel=1e-12)
    assert result.tests["model"]["statistic"] == pytest.approx(reference.fvalue, rel=1e-9)
    assert result.provenance["stata_parity_validated"] is False
    assert result.provenance["device"] == "cpu"


def test_stata_default_covariance_listwise_deletion_and_full_prediction(frame):
    frame.loc[3, "z"] = np.nan
    result = oe.ols(data=frame, y="y", x=["x", "z"])
    assert result.spec.covariance == "nonrobust"
    assert result.nobs == 159 and result.dropped_rows == 1
    prediction = result.predict()
    assert len(prediction) == 160
    assert prediction.iloc[3].isna().all()


def test_formula_fit_predictions_and_persistable_contract(frame):
    result = oe.ols(data=frame, formula="y ~ x * C(sector) + I(z**2)", covariance="HC3")
    design = result._state["x"].cpu().numpy()
    expected = sm.OLS(frame.y, design).fit(cov_type="HC3", use_t=True)
    assert_allclose([c.estimate for c in result.coefficients], expected.params, rtol=1e-11)
    new = frame.iloc[[11, 41, 59]].copy()
    predicted = result.predict(data=new)
    assert_allclose(predicted.iloc[:, 0], expected.predict(design[[11, 41, 59]]), rtol=1e-11)
    persisted = json.loads(result.model_dump_json())
    assert "_state" not in persisted and "frame" not in persisted
    restored = oe.ResultBundle.model_validate(persisted)
    assert restored.nobs == 160
    assert "\\toprule" in result.to_latex()


def test_collinear_predictors_are_omitted_without_losing_reference(frame):
    frame["duplicate"] = frame.x * 2
    result = oe.ols(data=frame, y="y", x=["x", "duplicate", "z"])
    assert result.provenance["omitted_terms"] == ["duplicate"]
    assert [c.term for c in result.coefficients] == ["Intercept", "x", "z"]
    reference = sm.OLS(frame.y, sm.add_constant(frame[["x", "z"]])).fit()
    assert_allclose([c.estimate for c in result.coefficients], reference.params, rtol=1e-11)


def test_fit_state_is_not_changed_by_mutating_caller_frame(frame):
    result = oe.ols(data=frame, y="y", x=["x", "z"])
    original = result.predict().copy()
    frame.loc[:, "x"] = 9000
    assert_allclose(result.predict(), original)


def test_prediction_rejects_unfitted_category(frame):
    result = oe.ols(data=frame, formula="y ~ x + C(sector)")
    new = frame.iloc[:1].copy()
    new["sector"] = "unseen"
    with pytest.raises(oe.AnalysisError, match="category"):
        result.predict(data=new)


def test_public_gpu_controls_are_explicit_and_mps_does_not_downcast(frame):
    if torch.backends.mps.is_available():
        cpu = oe.ols(data=frame, y="y", x=["x"], device="cpu")
        metal = oe.ols(data=frame, y="y", x=["x"], device="mps")
        assert metal.provenance["requested_device"] == "mps"
        assert metal.provenance["reporting_precision"] == "float64"
        assert metal.provenance["factor_devices"]["mps"] > 0
        assert_allclose(metal.covariance_matrix, cpu.covariance_matrix, rtol=1e-11, atol=1e-12)
    else:
        with pytest.raises(oe.AnalysisError):
            oe.ols(data=frame, y="y", x=["x"], device="mps")
    if not torch.cuda.is_available():
        with pytest.raises(oe.AnalysisError):
            oe.ols(data=frame, y="y", x=["x"], device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU on this verification host")
def test_public_cuda_matches_cpu_float64(frame):
    cpu = oe.ols(data=frame, y="y", x=["x", "z"], covariance="HC3", device="cpu")
    cuda = oe.ols(data=frame, y="y", x=["x", "z"], covariance="HC3", device="cuda")
    assert_allclose(cuda.covariance_matrix, cpu.covariance_matrix, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("options", [
    {"x": "x"}, {"weight_type": "fweight"}, {"weights": "w"},
    {"covariance": "hac"}, {"covariance": "cluster"}, {"lags": 2},
    {"covariance": "nonrobust", "weights": "w", "weight_type": "pweight"},
    {"covariance": "bootstrap", "reps": 1}, {"covariance": "bootstrap", "seed": -1},
])
def test_public_spec_validation_is_early_and_actionable(frame, options):
    arguments = {"data": frame, "y": "y", "x": ["x"]}
    arguments.update(options)
    with pytest.raises(oe.AnalysisError):
        oe.ols(**arguments)


def test_iter_predict_matches_full_prediction_without_collecting_output(frame):
    result = oe.ols(data=frame, formula="y ~ x * C(sector)")
    batches = list(result.iter_predict(batch_rows=23))
    assert len(batches) == 7 and max(len(part) for part in batches) <= 23
    assert_allclose(pd.concat(batches).to_numpy(), result.predict().to_numpy())
