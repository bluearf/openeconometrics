"""Saved-design, unsupported-family and malformed evaluation contracts."""

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from test_econ_postest_prediction import data, stored


@pytest.fixture(scope="module")
def model():
    return stored(oe.poisson(data=data(n=120), y="count", x=["x", "g"], categorical=["g"], missing="drop"))


def test_missing_rows_preserve_duplicate_index_positions(model):
    new = pd.DataFrame({"x": [-1., np.nan, 1., 0.], "g": ["A", "C", "B", None]}, index=[7, 7, 8, 8])
    predicted = oe.predict(model, new, interval="mean")
    assert predicted.index.tolist() == [7, 7, 8, 8]
    assert predicted.attrs["missing_row_positions"] == [1, 3]
    assert predicted.iloc[[1, 3]].isna().all().all()
    assert_allclose(predicted.iloc[[0, 2]], oe.predict(model, new.iloc[[0, 2]], interval="mean"))
    strict = model.model_copy(deep=True)
    strict.spec.missing = "raise"
    with pytest.raises(AnalysisError) as error:
        oe.predict(strict, new)
    assert error.value.code == "missing_values"


@pytest.mark.parametrize("new,code", [
    (pd.DataFrame({"x": [0.], "g": ["unseen"]}), "unknown_category"),
    (pd.DataFrame({"x": [0.]}), "missing_columns"),
    (pd.DataFrame({"x": [float("inf")], "g": ["A"]}), "non_finite_values"),
    (pd.DataFrame({"x": [1j], "g": ["A"]}), "complex_values"),
    (pd.DataFrame({"x": ["number"], "g": ["A"]}), "non_numeric_column"),
    (pd.DataFrame([[0, 0, "A"]], columns=["x", "x", "g"]), "duplicate_columns"),
])
def test_invalid_new_rows(model, new, code):
    with pytest.raises(AnalysisError) as error:
        oe.predict(model, new)
    assert error.value.code == code


@pytest.mark.parametrize("change,code", [
    (lambda m: m.provenance.update(categorical_encoding={}), "invalid_result"),
    (lambda m: m.provenance["categorical_encoding"]["g"].update(reference="C"), "invalid_result"),
    (lambda m: m.coefficients.pop(), "invalid_result"),
    (lambda m: m.extra.update(link="identity"), "invalid_result"),
    (lambda m: m.inference.update(use_t=True, distribution="t", df_inference=None), "invalid_inference"),
])
def test_malformed_saved_parameters_and_categories(model, change, code):
    changed = model.model_copy(deep=True)
    change(changed)
    with pytest.raises(AnalysisError) as error:
        oe.predict(changed, pd.DataFrame({"x": [0.], "g": ["A"]}))
    assert error.value.code == code


@pytest.mark.parametrize("options,code", [
    ({"kind": "classification"}, "unsupported_prediction_kind"),
    ({"interval": "obs"}, "unsupported_prediction_interval"),
    ({"kind": "derivative"}, "invalid_prediction_term"),
    ({"kind": "derivative", "term": "g"}, "invalid_prediction_term"),
    ({"kind": "response", "term": "x"}, "invalid_prediction_term"),
    ({"alpha": True}, "invalid_inference"),
])
def test_unsupported_options_are_explicit(model, options, code):
    with pytest.raises(AnalysisError) as error:
        oe.predict(model, pd.DataFrame({"x": [0.], "g": ["A"]}), **options)
    assert error.value.code == code


@pytest.mark.parametrize("options,code", [
    ({"variables": []}, "invalid_margins"), ({"variables": ["absent"]}, "invalid_margins"),
    ({"variables": ["x", "x"]}, "invalid_margins"), ({"at": {"absent": 1}}, "invalid_margins"),
    ({"at": {"x": []}}, "invalid_margins"), ({"at": {"x": float("inf")}}, "invalid_margins"),
    ({"at": {"g": "unseen"}}, "unknown_category"), ({"method": "wrong"}, "invalid_margins"),
    ({"kind": "classification"}, "invalid_margins"),
    ({"at": {"x": list(range(1001))}}, "margins_grid_limit"),
])
def test_invalid_margins_grids(model, options, code):
    with pytest.raises(AnalysisError) as error:
        oe.margins(model, data=data(n=10), **options)
    assert error.value.code == code


def test_unsupported_families_never_fall_back_to_linear_response(model):
    for estimator in ("nardl", "ardl", "biprobit", "frontier", "stcox"):
        changed = model.model_copy(deep=True)
        changed.spec.estimator = estimator
        for operation in (lambda: oe.predict(changed, data(n=5)), lambda: oe.margins(changed, data=data(n=5))):
            with pytest.raises(AnalysisError) as error:
                operation()
            assert error.value.code in {"unsupported_prediction","invalid_result","prediction_state_missing"}


def test_newly_supported_adapters_reject_relabelled_incompatible_saved_state(model):
    for estimator in ("areg", "reghdfe", "zip", "zinb", "tpoisson", "tnbreg", "hurdle", "mixed"):
        changed = model.model_copy(deep=True)
        changed.spec.estimator = estimator
        with pytest.raises(AnalysisError):
            oe.predict(changed, data(n=5))


def test_data_required_empty_eval_and_no_saved_chart_sample_substitution(model):
    with pytest.raises(AnalysisError) as error:
        oe.predict(model)
    assert error.value.code == "prediction_data_required"
    new = pd.DataFrame({"x": [np.nan], "g": ["A"]})
    assert oe.predict(model, new).isna().all().all()
    with pytest.raises(AnalysisError) as error:
        oe.margins(model, data=new)
    assert error.value.code == "empty_sample"


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_zero_weight_rows_are_excluded_before_category_numeric_and_link_evaluation(method):
    fitted = stored(oe.poisson(data=data(n=180), y="count", x=["x", "g"], categorical=["g"],
                              weights="w", weight_type="aweight"))
    fitted.coefficients[0].estimate, fitted.coefficients[1].estimate = .3, .5
    positive = pd.DataFrame({"x": [0.], "g": ["B"], "w": [1.]})
    extended = pd.DataFrame({"x": [0., 2000., float("inf")], "g": ["B", "unseen", "unseen"],
                             "w": [1., 0., 0.]})
    original = extended.copy(deep=True)
    expected = oe.margins(fitted, ["x", "g"], data=positive, method=method)
    actual = oe.margins(fitted, ["x", "g"], data=extended, method=method)
    assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"))
    assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"])
    assert actual.attrs["evaluation_rows"] == 1
    assert actual.attrs["input_evaluation_rows"] == 3
    assert actual.attrs["zero_weight_rows_excluded"] == 2
    pd.testing.assert_frame_equal(extended, original)
    with pytest.raises(AnalysisError) as error:
        oe.margins(fitted, "x", data=extended.assign(w=[1., -1., 0.]))
    assert error.value.code == "invalid_weights"


@pytest.mark.parametrize("link", ["cloglog", "loglog", "probit"])
def test_saturated_probability_tails_have_finite_exact_zero_effects_and_delta_gradients(link):
    fitted = stored(oe.glm(data=data(n=120), y="binary", x=["x"], family="binomial"))
    fitted.spec.options["link"] = link
    fitted.extra["link"] = link
    fitted.coefficients[0].estimate, fitted.coefficients[1].estimate = 0., 1.
    new = pd.DataFrame({"x": [-1e308, 1e308]})
    predicted = oe.predict(fitted, new, interval="mean")
    assert_allclose(predicted.response, [0., 1.])
    assert_allclose(predicted.std_error, 0.)
    derivative = oe.predict(fitted, new, kind="derivative", term="x", interval="mean")
    assert_allclose(derivative.select_dtypes("number"), 0.)
    effect = oe.margins(fitted, "x", data=new)
    assert_allclose(effect.estimate, 0.)
    assert_allclose(effect.std_error, 0.)
    assert_allclose(effect.attrs["delta_gradients"], 0.)


def test_invalid_exposure_and_binomial_trials_are_not_silent(model):
    changed = model.model_copy(deep=True)
    changed.spec.columns["exposure"] = "exposure"
    with pytest.raises(AnalysisError) as error:
        oe.predict(changed, pd.DataFrame({"x": [0.], "g": ["A"], "exposure": [0.]}))
    assert error.value.code == "invalid_exposure"


def test_link_domain_and_derivative_role_overlap_are_rejected():
    df = data(n=120)
    df["positive"] = np.exp(.5 + .15 * df.x) * np.random.default_rng(6).gamma(8, 1/8, len(df))
    model = stored(oe.glm(data=df, y="positive", x=["x"], family="gamma"))
    beta = np.array([c.estimate for c in model.coefficients])
    bad = -2 * beta[0] / beta[1]
    with pytest.raises(AnalysisError) as error:
        oe.predict(model, pd.DataFrame({"x": [bad]}))
    assert error.value.code == "prediction_domain"
    changed = model.model_copy(deep=True)
    changed.spec.columns["offset"] = "x"
    with pytest.raises(AnalysisError) as error:
        oe.margins(changed, "x", data=df)
    assert error.value.code == "unsupported_margins_transform"
