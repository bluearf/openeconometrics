"""Censored count ML with independent SciPy optimization and covariance oracles."""
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

import openecon as oe
from openecon.models import ModelSpec, ResultBundle
from test_econ_count_censored_kernels import finite_derivatives


def make_data(form="poisson", mode="mixed", n=650, seed=237):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    exposure = rng.uniform(.7, 1.4, n)
    mu = exposure * np.exp(.8 + .45 * x)
    if form == "poisson":
        latent = rng.poisson(mu)
    else:
        shape = 1 / .9 if form == "mean" else mu / .9
        latent = rng.negative_binomial(shape, shape / (shape + mu))
    y, lo, hi = latent.copy(), latent.astype(float), latent.astype(float)
    ll, ul = np.full(n, np.nan), np.full(n, np.nan)
    status = np.full(n, "exact", dtype=object)
    if mode == "auto":
        ll[:], ul[:] = 1, 5
        y = np.clip(latent, 1, 5)
        low, high = latent <= 1, latent >= 5
        lo[low], hi[low], status[low] = 0, 1, "left"
        lo[high], hi[high], status[high] = 5, np.inf, "right"
    elif mode == "mixed":
        left = (np.arange(n) % 4 == 1)
        right = (np.arange(n) % 4 == 2)
        interval = (np.arange(n) % 4 == 3)
        ll[left], ul[right] = 1, 5
        y[left], y[right] = np.maximum(latent[left], 1), np.minimum(latent[right], 5)
        low, high = left & (latent <= 1), right & (latent >= 5)
        lo[low], hi[low], status[low] = 0, 1, "left"
        lo[high], hi[high], status[high] = 5, np.inf, "right"
        lo[interval] = ll[interval] = (latent[interval] // 3) * 3
        hi[interval] = ul[interval] = lo[interval] + 2
        y[interval], status[interval] = lo[interval], "interval"
    return pd.DataFrame({"x": x, "y": y, "ll": ll, "ul": ul, "status": status,
                         "exposure": exposure, "offset": np.log(exposure),
                         "event_lo": lo, "event_hi": hi, "cluster": np.arange(n) % 30,
                         "w": rng.uniform(.5, 2, n)})


def rows(theta, data, form):
    mu = np.exp(theta[0] + theta[1] * data.x.to_numpy() + data.offset.to_numpy())
    if form == "poisson":
        dist = poisson(mu)
    else:
        alpha = np.exp(theta[2])
        shape = 1 / alpha if form == "mean" else mu / alpha
        dist = nbinom(shape, shape / (shape + mu))
    lo, hi = data.event_lo.to_numpy(), data.event_hi.to_numpy()
    exact, right, left = lo == hi, np.isinf(hi), (lo == 0) & (lo != hi)
    intervals = ~(exact | right | left)
    output = np.empty(len(data))
    output[exact] = dist.logpmf(lo)[exact]
    output[right] = dist.logsf(lo - 1)[right]
    output[left] = dist.logcdf(hi)[left]
    output[intervals] = np.log(dist.cdf(hi)[intervals] - dist.cdf(lo - 1)[intervals])
    return output


def estimate(data, form, mode="mixed", **options):
    function = oe.cpoisson if form == "poisson" else oe.cnbreg
    arguments = {"ll": "ll", "ul": "ul"} if mode != "none" else {}
    if mode == "mixed":
        arguments["censoring"] = "status"
    if form != "poisson":
        arguments["dispersion"] = form
    return function(data=data, y="y", x=["x"], exposure="exposure", **arguments, **options)


def coefficients(result):
    return np.array([part.estimate for part in result.coefficients])


@pytest.mark.parametrize("form", ["poisson", "mean", "constant"])
@pytest.mark.parametrize("mode", ["auto", "mixed", "none"])
def test_full_fit_matches_independent_scipy_optimizer_information_and_predictions(form, mode):
    data = make_data(form, mode)
    original = data.copy(deep=True)
    result = estimate(data, form, mode)
    start = np.array([.8, .4] + ([] if form == "poisson" else [np.log(.8)]))
    def function(theta):
        return rows(theta, data, form).sum()
    oracle = minimize(lambda theta: -function(theta), start, method="BFGS",
                      options={"gtol": 2e-6, "maxiter": 500})
    assert np.isfinite(oracle.fun)
    assert_allclose(coefficients(result), oracle.x, rtol=3e-5, atol=3e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-oracle.fun, rel=2e-10)
    _, hessian = finite_derivatives(function, oracle.x, step=4e-4)
    assert_allclose(result.covariance_matrix, np.linalg.inv(-hessian), rtol=3e-4, atol=2e-7)
    for point in result.predictions:
        row = data.iloc[point["row"]]
        expected = row.exposure * np.exp(coefficients(result)[0] + coefficients(result)[1] * row.x)
        assert point["fitted"] == pytest.approx(expected, rel=1e-11)
    assert result.extra["fitted_values"] == "latent unconditional mean E[Y|X]"
    assert result.extra["stata_parity_validated"] is False
    assert result.metrics["aic"] == pytest.approx(2 * len(result.coefficients) - 2 * result.metrics["log_likelihood"])
    assert json.loads(json.dumps(result.model_dump(), allow_nan=False)) == result.model_dump()
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.model_dump() == result.model_dump()
    assert "\\begin{tabular}" in restored.to_latex()
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("form", ["poisson", "mean", "constant"])
def test_no_censoring_reduces_to_existing_poisson_or_negative_binomial(form):
    data = make_data(form, "none")
    censored = estimate(data, form, "none")
    function = oe.poisson if form == "poisson" else oe.nbreg
    extra = {} if form == "poisson" else {"dispersion": form}
    ordinary = function(data=data, y="y", x=["x"], exposure="exposure", **extra)
    assert_allclose(coefficients(censored), coefficients(ordinary), rtol=2e-6, atol=2e-8)
    assert_allclose(censored.covariance_matrix, ordinary.covariance_matrix, rtol=3e-6, atol=2e-8)
    assert censored.metrics["log_likelihood"] == pytest.approx(ordinary.metrics["log_likelihood"], rel=1e-10)


@pytest.mark.parametrize("form", ["poisson", "mean"])
@pytest.mark.parametrize("covariance", ["opg", "robust", "cluster"])
def test_weighted_covariance_matches_independently_differenced_likelihood_scores(form, covariance):
    data = make_data(form, n=400)
    options = {"cluster": "cluster"} if covariance == "cluster" else {}
    result = estimate(data, form, weights="w", weight_type="iweight", covariance=covariance, **options)
    theta = coefficients(result)
    weights = data.w.to_numpy()
    def function(t):
        return weights @ rows(t, data, form)
    _, hessian = finite_derivatives(function, theta, step=4e-4)
    bread = np.linalg.inv(-hessian)
    scores = np.column_stack([(rows(theta + np.eye(len(theta))[j] * 1e-5, data, form)
                               - rows(theta - np.eye(len(theta))[j] * 1e-5, data, form)) / 2e-5
                              for j in range(len(theta))])
    if covariance == "opg":
        expected = np.linalg.inv(scores.T @ (weights[:, None] * scores))
    elif covariance == "robust":
        weighted = weights[:, None] * scores
        expected = bread @ (weighted.T @ weighted) @ bread * len(data) / (len(data) - 1)
    else:
        grouped = np.zeros((data.cluster.nunique(), len(theta)))
        np.add.at(grouped, data.cluster.to_numpy(), weights[:, None] * scores)
        expected = bread @ (grouped.T @ grouped) @ bread * len(grouped) / (len(grouped) - 1)
    assert_allclose(result.covariance_matrix, expected, rtol=4e-4, atol=3e-7)


@pytest.mark.parametrize("form", ["poisson", "mean"])
def test_frequency_weights_equal_actual_replication_and_probability_weights_are_scale_invariant(form):
    data = make_data(form, n=350)
    data["f"] = 1 + np.arange(len(data)) % 3
    expanded = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    weighted = estimate(data, form, weights="f", weight_type="fweight", covariance="robust")
    repeated = estimate(expanded, form, covariance="robust")
    assert weighted.nobs == len(expanded)
    assert_allclose(coefficients(weighted), coefficients(repeated), rtol=2e-6, atol=2e-8)
    assert_allclose(weighted.covariance_matrix, repeated.covariance_matrix, rtol=3e-6, atol=2e-8)
    probability = estimate(data, form, weights="w", weight_type="pweight")
    data["huge"] = data.w * 1e8
    scaled = estimate(data.rename(columns={"w": "old", "huge": "w"}), form,
                      weights="w", weight_type="pweight")
    assert probability.inference["covariance"] == "robust"
    assert_allclose(coefficients(probability), coefficients(scaled), rtol=2e-6, atol=2e-8)
    assert_allclose(probability.covariance_matrix, scaled.covariance_matrix, rtol=3e-6, atol=2e-8)


def test_direct_spec_fixed_limits_public_catalog_and_offset_are_consistent():
    data = make_data("poisson", "auto")
    arguments = dict(outcome="y", predictors=["x"], options={"ll": 1, "ul": 5}, columns={"offset": "offset"})
    spec = ModelSpec(estimator="cpoisson", **arguments)
    direct = oe.fit(spec, data=data)
    friendly = oe.cpoisson(data=data, y="y", x=["x"], ll=1, ul=5, exposure="exposure")
    assert_allclose(coefficients(direct), coefficients(friendly), rtol=1e-10)
    assert direct.extra["censoring_counts"] == friendly.extra["censoring_counts"]
    serialized = ModelSpec.model_validate_json(spec.model_dump_json())
    assert serialized.model_dump() == spec.model_dump()
    catalog = oe.capabilities()["estimators"]
    assert catalog["cpoisson"]["family"] == "count"
    assert catalog["cnbreg"]["function"] == "cnbreg"


def test_missing_unused_endpoint_values_do_not_drop_uncensored_rows():
    data = make_data("poisson", "mixed")
    result = estimate(data, "poisson")
    assert result.nobs == len(data) and result.dropped_rows == 0
    broken = data.copy()
    broken.loc[3, "x"] = np.nan
    with pytest.raises(oe.AnalysisError) as error:
        estimate(broken, "poisson")
    assert error.value.code == "missing_values"
    dropped = estimate(broken, "poisson", missing="drop")
    explicit = estimate(broken.drop(index=3).reset_index(drop=True), "poisson")
    assert dropped.dropped_rows == 1
    assert_allclose(coefficients(dropped), coefficients(explicit), rtol=1e-10)
