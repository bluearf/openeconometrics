"""Publication disclosures and literal-column compatibility for full OLS."""
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.dataset import Dataset
from openecon.linear_ols.publication import model_notes
from openecon.output_latex import add_output_latex


@pytest.fixture
def frame():
    rng = np.random.default_rng(2041)
    data = pd.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120),
                         "t": np.arange(120), "firm": np.repeat(np.arange(12), 10),
                         "region": np.tile(np.arange(6), 20), "w": np.tile([1, 2, 3], 40)})
    data["y"] = 2 + .7 * data.x - .3 * data.z + rng.normal(size=120)
    return data


@pytest.mark.parametrize("covariance,label", [
    ("nonrobust", "conventional standard errors"),
    ("HC0", "HC0 heteroskedasticity-robust"), ("HC1", "HC1 heteroskedasticity-robust"),
    ("HC2", "HC2 heteroskedasticity-robust"), ("HC3", "HC3 heteroskedasticity-robust"),
    ("cluster", "cluster-robust standard errors"), ("cluster_hc2", "CR2 cluster leverage-adjusted"),
    ("cluster_hc3", "CR3 cluster leverage-adjusted"), ("hac", "autocorrelation consistent (HAC)"),
    ("bootstrap", "bootstrap standard errors"), ("jackknife", "jackknife standard errors"),
])
def test_every_covariance_has_publication_method_disclosure(frame, covariance, label):
    options = {"cluster": "firm"} if covariance.startswith("cluster") else {}
    if covariance == "hac":
        options.update(time="t", lags=2, kernel="parzen")
    if covariance == "bootstrap":
        options.update(reps=7, seed=124)
    result = oe.ols(data=frame, y="y", x=["x", "z"], covariance=covariance, **options)
    snapshot = result.model_dump_json()
    source = str(result.to_latex())
    assert label in source
    assert r"\toprule" in source and r"\bottomrule" in source
    assert result.model_dump_json() == snapshot
    if covariance.startswith("cluster"):
        assert "firm (12 clusters)" in source
    if covariance == "hac":
        for note in ("parzen kernel, lags = 2", "bandwidth = 3", "actual calendar gaps retained"):
            assert note in source
    if covariance == "bootstrap":
        assert "pairs resampling, 7 of 7 repetitions used" in source
        assert "seed = 124" in source and "normal z inference" in source
    if covariance == "jackknife":
        assert "delete-one resampling, 120 of 120 repetitions used" in source


@pytest.mark.parametrize("weight_type,description", [("aweight", "analytic"), ("fweight", "frequency"),
                                                     ("pweight", "sampling"), ("iweight", "importance")])
def test_weighted_effective_and_physical_samples_are_distinct(frame, weight_type, description):
    frame.loc[0, "w"] = 0
    result = oe.ols(data=frame, y="y", x=["x", "z"], weights="w", weight_type=weight_type)
    source = str(result.to_latex())
    assert f"{description} weights: w" in source
    assert f"effective observations = {result.nobs}" in source
    assert "physical estimation sample 119 of 120 observations" in source
    assert "1 observations excluded from estimation" in source
    assert "missing observations excluded" not in source
    diagnostic = str(result.to_latex(style="diagnostic"))
    assert "Excluded observations" in diagnostic
    assert "Excluded missing observations" not in diagnostic
    if weight_type in {"fweight", "iweight"}:
        assert f"estimation sample {result.nobs} of 120 observations" not in source


def test_multiway_cluster_notes_identify_each_dimension(frame):
    result = oe.ols(data=frame, y="y", x=["x", "z"], covariance="cluster", cluster=["firm", "region"])
    source = str(result.to_latex())
    assert "firm (12 clusters), region (6 clusters)" in source
    assert "CGM inclusion-exclusion" in source
    assert "Student t inference (df = 5)" in source


@pytest.mark.parametrize("covariance,hansen", [("HC2", False), ("HC3", True),
                                              ("cluster_hc2", False), ("cluster_hc3", True)])
def test_adjusted_df_hansen_scaling_and_model_f_use_actual_inference(frame, covariance, hansen):
    result = oe.ols(data=frame, y="y", x=["x"], covariance=covariance, dfadjust=True, hansen=hansen,
                    cluster="firm" if covariance.startswith("cluster") else None)
    source = str(result.to_latex())
    assert "coefficient-specific Bell-McCaffrey/Satterthwaite" in source
    assert f"Student t inference (df = {result.inference['df_inference']:g})" not in source
    for coefficient, degrees in zip(result.coefficients, result.inference["coefficient_df"], strict=True):
        assert f"{coefficient.term} = {degrees:g}" in source
    actual = result.tests["model"]
    assert f"model F(1, {actual['df2']:g})" in source
    assert actual["df2"] == pytest.approx(result.inference["coefficient_df"][1])
    if hansen:
        assert "Hansen (2025) scaling multiplies t statistics" in source
        assert "divides confidence interval widths" in source
        assert "printed standard errors are unchanged" in source
        for coefficient, scale in zip(result.coefficients, result.inference["coefficient_scale"], strict=True):
            assert f"{coefficient.term} = {scale:g}" in source
    diagnostic = str(result.to_latex(style="diagnostic"))
    assert "coefficient-specific Bell-McCaffrey/Satterthwaite" in diagnostic


def test_console_serialized_export_retains_adjustments_and_weight_context(frame):
    result = oe.ols(data=frame, y="y", x=["x"], covariance="HC3", weights="w", weight_type="fweight", hansen=True)
    payload = result.model_dump(mode="json", exclude={"sample_positions", "covariance_matrix"})
    payload["display_omitted"] = ["sample_positions", "covariance_matrix"]
    before = json.dumps(payload, sort_keys=True)
    item = add_output_latex({"type": "model", "data": payload})
    assert "Hansen (2025)" in item["latex"] and "frequency weights" in item["latex"]
    assert "physical estimation sample 120 of 120" in item["latex"]
    assert item["data"]["inference"]["coefficient_df"] == result.inference["coefficient_df"]
    assert json.dumps(payload, sort_keys=True) == before


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_nonols_exports_never_use_the_new_ols_helper(frame, estimator, monkeypatch):
    frame["binary"] = np.random.default_rng(31).binomial(1, .5, len(frame))
    result = getattr(oe, estimator)(data=frame, y="binary", x=["x"])
    monkeypatch.setattr("openecon.linear_ols.publication.model_notes", lambda *a, **k: pytest.fail("OLS helper used for a binary model"))
    assert f"{estimator.upper()}: conventional standard errors" in result.to_latex()
    assert "normal z inference" in result.to_latex()
    assert result.to_latex(style="diagnostic")


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_diagnostic_exclusion_label_remains_unchanged(frame, estimator):
    frame["binary"] = np.random.default_rng(31).binomial(1, .5, len(frame)).astype(float)
    frame.loc[0, "binary"] = np.nan
    result = getattr(oe, estimator)(data=frame, y="binary", x=["x"], missing="drop")
    assert "Excluded missing observations" in result.to_latex(style="diagnostic")


@pytest.mark.parametrize("name", ["annual income", "tax-rate", "return (%)", "a*b", "a:b",
                                  "I(x**2)", "Türkçe ücret", "__import__('os')"])
@pytest.mark.parametrize("streamed", [False, True])
def test_literal_names_survive_estimation_prediction_and_json_refit(frame, name, streamed):
    data = frame.rename(columns={"x": name})
    result = oe.ols(data=Dataset.from_frame(data) if streamed else data, y="y", x=[name, "z"], covariance="HC3")
    reference = oe.ols(data=frame, y="y", x=["x", "z"], covariance="HC3")
    assert [coefficient.term for coefficient in result.coefficients] == ["Intercept", name, "z"]
    assert_allclose(result.covariance_matrix, reference.covariance_matrix, rtol=2e-12, atol=1e-13)
    rebuilt = oe.fit(oe.ModelSpec.model_validate_json(result.spec.model_dump_json()), data=data)
    assert_allclose([coefficient.estimate for coefficient in rebuilt.coefficients],
                    [coefficient.estimate for coefficient in result.coefficients], rtol=2e-12, atol=1e-13)
    actual = pd.concat(list(result.iter_predict(data=Dataset.from_frame(data), batch_rows=7)))
    assert_allclose(actual.to_numpy(), reference.predict(data=frame).to_numpy(), rtol=2e-12, atol=1e-13)


def test_literal_categorical_label_is_not_split_as_an_interaction(frame):
    frame["sector:kind"] = np.tile(["a", "b", "c"], 40)
    result = oe.ols(data=frame, y="y", x=["sector:kind"], categorical=["sector:kind"])
    assert [coefficient.term for coefficient in result.coefficients] == ["Intercept", "sector:kind[b]", "sector:kind[c]"]
    rebuilt = oe.fit(oe.ModelSpec.model_validate_json(result.spec.model_dump_json()), data=frame)
    assert_allclose(rebuilt.covariance_matrix, result.covariance_matrix, rtol=2e-12, atol=1e-13)


@pytest.mark.parametrize("expression", ["__import__('os')", "x.__class__", "x[0]", "C(x, 1)"])
def test_formula_allowlist_remains_strict_for_literal_looking_expressions(frame, expression):
    with pytest.raises(oe.AnalysisError):
        oe.ols(data=frame, formula=f"y ~ {expression}")


def test_publication_helper_rejects_nonols_contract(frame):
    frame["binary"] = np.random.default_rng(11).binomial(1, .5, len(frame))
    result = oe.logit(data=frame, y="binary", x=["x"])
    with pytest.raises(ValueError, match="OLS"):
        model_notes(result)


@pytest.mark.parametrize("streamed", [False, True])
def test_lag_with_nullable_categorical_values_preserves_valid_history(frame, streamed):
    frame["sector"] = pd.Series(np.tile(["a", "b", "c"], 40), dtype="string")
    frame.loc[17, "x"] = np.nan
    frame.loc[30, "sector"] = pd.NA
    result = oe.ols(data=Dataset.from_frame(frame) if streamed else frame,
                    formula="y ~ L(x) + C(sector)", time="t")
    reference = frame.copy()
    reference["lag_x"] = reference.x.shift()
    expected = oe.ols(data=reference, y="y", x=["lag_x", "sector"], categorical=["sector"], time="t")
    assert result.nobs == expected.nobs == len(frame) - 3
    assert_allclose(result.covariance_matrix, expected.covariance_matrix, rtol=2e-12, atol=1e-13)
    predictions = pd.concat(list(result.iter_predict(data=Dataset.from_frame(frame), batch_rows=7)))
    assert np.isfinite(predictions.iloc[17, 0])
    assert np.isfinite(predictions.iloc[31, 0])
    assert np.isnan(predictions.iloc[18, 0])
    assert np.isnan(predictions.iloc[30, 0])
