"""Bounded independent/exchangeable GEE parity and true group-sandwich oracle."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.mixed.gee import fit_xtgee
from openecon.econometrics.mixed.xt import fit_xtlogit, fit_xtprobit, fit_xtpoisson
from openecon.econometrics.streaming_gee import fit_pa, fit_streaming_gee
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def data(family="gaussian", seed=493515, n=311):
    rng = np.random.default_rng(seed)
    group = np.arange(n)%23
    x, e = rng.normal(size=(2, n))
    eta = .3+.15*x+.02*group
    if family == "gaussian":
        y = eta+e
    elif family == "binomial":
        y = rng.binomial(1, 1/(1+np.exp(-eta)))
    elif family in {"poisson", "nbinomial"}:
        y = rng.poisson(np.exp(eta))
    elif family in {"gamma", "igaussian"}:
        y = np.exp(eta+.3*e)
    frame = pd.DataFrame({"x": x, "y": y, "panel": group,
                          "offset": np.full(n, .1), "exposure": np.full(n, 1.2),
                          "cat": pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3])})
    return frame


def spec(family="gaussian", corr="exchangeable", covariance="nonrobust", **options):
    return ModelSpec(estimator="xtgee", outcome="y", predictors=["x"], panel="panel",
                     covariance=covariance, options={"family": family, "corr": corr, **options})


def parity(dense, actual):
    assert [c.term for c in dense.coefficients] == [c.term for c in actual.coefficients]
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in dense.coefficients], rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(actual.covariance_matrix, dense.covariance_matrix, rtol=4e-8, atol=3e-9)
    for key, value in dense.metrics.items():
        assert actual.metrics[key] == pytest.approx(value, rel=4e-8, abs=3e-9), key
    if "alpha" in dense.extra:
        np.testing.assert_allclose(actual.extra["alpha"], dense.extra["alpha"], atol=3e-9)
    assert actual.nobs == dense.nobs
    assert actual.sample_positions == []
    assert not actual.provenance["streaming"]["dense_observation_matrix"]
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 17
    ResultBundle.model_validate_json(actual.model_dump_json())


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson", "gamma", "nbinomial", "igaussian"])
@pytest.mark.parametrize("corr", ["independent", "exchangeable"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust"])
def test_current_families_correlations_covariances(family, corr, covariance):
    frame = data(family)
    # Positive links keep the source within their genuine parameter domain.
    s = spec(family, corr, covariance, **({"link": "log"} if family in {"gamma", "igaussian"} else {}))
    parity(fit_xtgee(s, frame), fit_streaming_gee(s, Dataset.from_frame(frame), batch_rows=17))


@pytest.mark.parametrize("scale", [None, "x2", "dev", 1.7])
@pytest.mark.parametrize("nmp", [False, True])
def test_gaussian_scale_nmp_with_offset_exposure_and_categories(scale, nmp):
    frame = data()
    s = spec(scale=scale, nmp=nmp).model_copy(update={"predictors": ["x", "cat"], "categorical": ["cat"],
                                                "columns": {"offset": "offset", "exposure": "exposure"}})
    parity(fit_xtgee(s, frame), fit_streaming_gee(s, Dataset.from_frame(frame), batch_rows=17))


@pytest.mark.parametrize("estimator,family,function", [("xtlogit", "binomial", fit_xtlogit),
                                ("xtprobit", "binomial", fit_xtprobit), ("xtpoisson", "poisson", fit_xtpoisson)])
def test_pa_private_hook_exact_existing_wrapper_contract(estimator, family, function):
    frame = data(family)
    s = ModelSpec(estimator=estimator, outcome="y", predictors=["x"], panel="panel", covariance="robust",
                  options={"model": "pa", "corr": "exchangeable"})
    parity(function(s, frame), fit_pa(s, Dataset.from_frame(frame), batch_rows=17))


def test_exchangeable_gaussian_numpy_gls_and_true_panel_sandwich_oracle():
    frame = data()
    result = fit_streaming_gee(spec(covariance="robust"), Dataset.from_frame(frame), batch_rows=17)
    alpha = result.extra["alpha"][0]
    x, y = np.column_stack([np.ones(len(frame)), frame.x]), frame.y.to_numpy()
    bread, score = np.zeros((2, 2)), np.zeros(2)
    for key in np.unique(frame.panel):
        select = frame.panel.to_numpy() == key
        ni = int(select.sum())
        inverse = (np.eye(ni)-alpha/(1+(ni-1)*alpha)*np.ones((ni, ni)))/(1-alpha)
        bread += x[select].T@inverse@x[select]
        score += x[select].T@inverse@y[select]
    b = np.linalg.solve(bread, score)
    meat = np.zeros((2, 2))
    for key in np.unique(frame.panel):
        select = frame.panel.to_numpy() == key
        ni = int(select.sum())
        inverse = (np.eye(ni)-alpha/(1+(ni-1)*alpha)*np.ones((ni, ni)))/(1-alpha)
        group_score = x[select].T@inverse@(y[select]-x[select]@b)
        meat += np.outer(group_score, group_score)
    cov = np.linalg.solve(bread, meat)@np.linalg.inv(bread)*23/22
    np.testing.assert_allclose([c.estimate for c in result.coefficients], b, atol=3e-9)
    np.testing.assert_allclose(result.covariance_matrix, cov, rtol=3e-9, atol=3e-11)


@pytest.mark.parametrize("corr", ["ar1", "stationary", "nonstationary", "unstructured"])
def test_ordered_correlations_require_time_before_source_read(monkeypatch, corr):
    source = Dataset.from_frame(data())
    def forbidden(*args, **kwargs):
        raise AssertionError("A timed GEE specification without time must fail before a source pass.")
    monkeypatch.setattr(source, "iter_batches", forbidden)
    with pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec(corr=corr), source)
    assert error.value.code == "invalid_spec"


def test_workspace_guard_has_no_total_row_ceiling():
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        fit_streaming_gee(spec(), Dataset.from_frame(data()), batch_rows=17)
    assert error.value.code == "workspace_limit"
