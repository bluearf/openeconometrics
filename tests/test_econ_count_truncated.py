"""tpoisson and tnbreg against independent oracles.

Oracles: the conditional likelihood ``logpmf(y) - logsf(ll)`` written with
scipy.stats and maximized by scipy.optimize, numerical scores and Hessians of
that likelihood for every covariance estimator, statsmodels
TruncatedLFPoisson / TruncatedLFNegativeBinomialP, duplicated rows for
frequency weights and mpmath for the far tail of the truncation probability.
"""

import warnings

import mpmath
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import optimize, stats
from statsmodels.discrete.truncated_model import (
    TruncatedLFNegativeBinomialP, TruncatedLFPoisson,
)

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.count.kernels import (
    NegBinDensity, PoissonDensity, TruncatedPieces,
)

X = ["x1", "x2"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(90210)
    n = 4000
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(100), 40), "state": rng.integers(0, 15, size=n),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 2.0, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "cut": rng.integers(0, 3, size=n).astype(float),
    })
    mu = frame.expo * np.exp(0.5 + 0.4 * frame.x1 - 0.3 * frame.x2)
    frame["y"] = rng.poisson(mu).astype(float)
    frame["ynb"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    frame["lnexpo"] = np.log(frame.expo)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def design(frame, columns=X):
    return sm.add_constant(frame[list(columns)]).to_numpy()


def truncated_loglik(theta, y, x, limit, offset=0.0, form=None):
    """Per-observation log likelihood of a count truncated to y > limit (scipy.stats)."""
    k = x.shape[1]
    mu = np.exp(x @ theta[:k] + offset)
    if form is None:
        return stats.poisson.logpmf(y, mu) - stats.poisson.logsf(limit, mu)
    dispersion = np.exp(theta[k])
    size = 1 / dispersion if form == "mean" else mu / dispersion
    p = size / (size + mu)
    return stats.nbinom.logpmf(y, size, p) - stats.nbinom.logsf(limit, size, p)


def numeric_scores(per_observation, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        step = np.zeros_like(theta)
        step[j] = h * max(1.0, abs(theta[j]))
        columns.append((per_observation(theta + step) - per_observation(theta - step))
                       / (2 * step[j]))
    return np.column_stack(columns)


def numeric_hessian(total, theta, h=1e-4):
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    steps = np.diag([h * max(1.0, abs(value)) for value in theta])
    hessian = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            value = (total(theta + steps[i] + steps[j]) - total(theta + steps[i] - steps[j])
                     - total(theta - steps[i] + steps[j]) + total(theta - steps[i] - steps[j]))
            hessian[i, j] = hessian[j, i] = value / (4 * steps[i, i] * steps[j, j])
    return hessian


def close(actual, desired, rtol=2e-5):
    desired = np.asarray(desired)
    assert_allclose(actual, desired, rtol=rtol, atol=rtol * np.abs(desired).max())


def cluster_meat(scores, labels):
    sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
    return sums.T @ sums


def brute_force(objective, start):
    best = optimize.minimize(lambda t: -objective(t), start, method="BFGS",
                             options={"gtol": 1e-9, "maxiter": 2000})
    best = optimize.minimize(lambda t: -objective(t), best.x, method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000,
                                      "maxfev": 20000})
    return best.x, -best.fun


# ---- tpoisson ------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0, 2])
def test_tpoisson_matches_statsmodels_and_brute_force(data, limit):
    sample = data[data.y > limit].reset_index(drop=True)
    result = oe.tpoisson(data=sample, y="y", x=X, ll=limit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = TruncatedLFPoisson(sample.y, sm.add_constant(sample[X]),
                                       truncation=limit).fit(method="newton", maxiter=200,
                                                             disp=0, tol=1e-12)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert_allclose(estimates(result), reference.params.to_numpy(), rtol=1e-8)
    assert_allclose(errors(result), reference.bse.to_numpy(), rtol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-11)
    y, x = sample.y.to_numpy(), design(sample)
    theta, value = brute_force(lambda t: truncated_loglik(t, y, x, limit).sum(), np.zeros(3))
    assert_allclose(estimates(result), theta, rtol=1e-4, atol=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-10)
    # Constant-only model, LR test and McFadden's pseudo R-squared.
    _, null = brute_force(lambda t: truncated_loglik(t, y, x[:, :1], limit).sum(), np.zeros(1))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-10)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - value / null, rel=1e-8)
    test = result.tests["model"]
    assert test["df"] == 2 and test["statistic"] == pytest.approx(2 * (value - null), rel=1e-7)
    assert result.metrics["aic"] == pytest.approx(-2 * value + 6, rel=1e-10)
    assert result.metrics["bic"] == pytest.approx(-2 * value + 3 * np.log(len(sample)),
                                                  rel=1e-10)
    assert result.extra["truncation_point"] == limit
    # The chart sample holds the conditional mean E[y | y > ll].
    mu = np.exp(x @ estimates(result))
    lower = sum(j * stats.poisson.pmf(j, mu) for j in range(limit + 1))
    mean = (mu - lower) / stats.poisson.sf(limit, mu)
    for row in result.predictions[:25]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-9)


def test_tpoisson_observation_specific_truncation_and_exposure(data):
    sample = data[data.y > data.cut].reset_index(drop=True)
    result = oe.tpoisson(data=sample, y="y", x=X, ll="cut", exposure="expo")
    y, x, limit = sample.y.to_numpy(), design(sample), sample.cut.to_numpy()
    offset = sample.lnexpo.to_numpy()

    def total(theta):
        return truncated_loglik(theta, y, x, limit, offset).sum()

    theta, value = brute_force(total, np.zeros(3))
    assert_allclose(estimates(result), theta, rtol=1e-4, atol=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-10)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))))
    assert result.spec.columns == {"truncation": "cut", "exposure": "expo"}
    assert result.extra["truncation_column"] == "cut"
    assert (result.extra["truncation_min"], result.extra["truncation_max"]) == (0, 2)
    by_offset = oe.tpoisson(data=sample, y="y", x=X, ll="cut", offset="lnexpo")
    assert_allclose(estimates(by_offset), estimates(result), rtol=1e-10)
    # A constant column of truncation points is the same model as the number.
    zero_cut = data[data.y > 1].assign(cut=1.0)
    assert_allclose(estimates(oe.tpoisson(data=zero_cut, y="y", x=X, ll="cut")),
                    estimates(oe.tpoisson(data=zero_cut, y="y", x=X, ll=1)), rtol=1e-10)


def test_tpoisson_covariances_and_weights(data):
    sample = data[data.y > 0].reset_index(drop=True)
    y, x = sample.y.to_numpy(), design(sample)
    base = oe.tpoisson(data=sample, y="y", x=X)
    theta = estimates(base)

    def per_observation(t):
        return truncated_loglik(t, y, x, 0)

    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(sample)
    close(covariance(base), bread)
    opg = oe.tpoisson(data=sample, y="y", x=X, covariance="opg")
    close(covariance(opg), np.linalg.inv(scores.T @ scores))
    robust = oe.tpoisson(data=sample, y="y", x=X, covariance="robust")
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1))
    assert robust.tests["model"]["label"].startswith("Wald chi2")
    wald = theta[1:] @ np.linalg.solve(covariance(robust)[1:, 1:], theta[1:])
    assert robust.tests["model"]["statistic"] == pytest.approx(wald, rel=1e-9)
    cluster = oe.tpoisson(data=sample, y="y", x=X, cluster="firm")
    groups = sample.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, sample.firm) @ bread * groups / (groups - 1))
    two_way = oe.tpoisson(data=sample, y="y", x=X, cluster=["firm", "state"])
    pair = sample.firm.astype(str) + "/" + sample.state.astype(str)
    meat = (cluster_meat(scores, sample.firm) + cluster_meat(scores, sample.state)
            - cluster_meat(scores, pair))
    smallest = min(groups, sample.state.nunique())
    close(covariance(two_way), bread @ meat @ bread * smallest / (smallest - 1))
    # Frequency weights replicate rows.
    repeated = sample.loc[sample.index.repeat(sample.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust"):
        weighted = oe.tpoisson(data=sample, y="y", x=X, weights="fw", weight_type="fweight",
                               covariance=kind)
        expanded = oe.tpoisson(data=repeated, y="y", x=X, covariance=kind)
        assert weighted.nobs == len(repeated)
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-9)
        assert_allclose(covariance(weighted), covariance(expanded), rtol=1e-8, atol=1e-14)
        assert weighted.tests["model"]["statistic"] == pytest.approx(
            expanded.tests["model"]["statistic"], rel=1e-8)
    # Analytic weights (rescaled to sum to N), importance and sampling weights.
    w = sample.aw.to_numpy() * n / sample.aw.sum()
    analytic = oe.tpoisson(data=sample, y="y", x=X, weights="aw", weight_type="aweight")
    best, value = brute_force(lambda t: (w * per_observation(t)).sum(), np.zeros(3))
    assert_allclose(estimates(analytic), best, rtol=1e-4, atol=1e-5)
    assert analytic.metrics["log_likelihood"] == pytest.approx(value, rel=1e-10)
    close(covariance(analytic),
          np.linalg.inv(-numeric_hessian(lambda t: (w * per_observation(t)).sum(),
                                         estimates(analytic))))
    importance = oe.tpoisson(data=sample, y="y", x=X, weights="aw", weight_type="iweight")
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-9)
    raw = sample.aw.to_numpy()
    close(covariance(importance),
          np.linalg.inv(-numeric_hessian(lambda t: (raw * per_observation(t)).sum(),
                                         estimates(importance))))
    sampling = oe.tpoisson(data=sample, y="y", x=X, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    weighted_scores = numeric_scores(per_observation, estimates(sampling)) * raw[:, None]
    weighted_bread = np.linalg.inv(-numeric_hessian(
        lambda t: (raw * per_observation(t)).sum(), estimates(sampling)))
    close(covariance(sampling),
          weighted_bread @ (weighted_scores.T @ weighted_scores) @ weighted_bread * n / (n - 1))
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample, y="y", x=X, weights="aw", weight_type="pweight",
                    covariance="opg")
    assert excinfo.value.code == "unsupported_covariance"


def test_tpoisson_failure_contract_collinearity_and_missing(data):
    sample = data[data.y > 0].reset_index(drop=True)
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=data, y="y", x=X)
    assert excinfo.value.code == "outcome_not_truncated" and "oe.hurdle" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample, y="y", x=X, ll=1)
    assert excinfo.value.code == "outcome_not_truncated"
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample.assign(y=sample.y + 0.5), y="y", x=X)
    assert excinfo.value.code == "invalid_count_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample.assign(y=1.0), y="y", x=X)
    assert excinfo.value.code == "constant_outcome"
    for bad in (-1, 1.5, True):
        with pytest.raises(AnalysisError) as excinfo:
            oe.tpoisson(data=sample, y="y", x=X, ll=bad)
        assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample.assign(cut=0.5), y="y", x=X, ll="cut")
    assert excinfo.value.code == "invalid_truncation"
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=sample, y="y", x="x1")
    assert excinfo.value.code == "invalid_spec"
    # Collinear regressors are omitted Stata-style; categorical terms are expanded.
    result = oe.tpoisson(data=sample.assign(twice=2 * sample.x1), y="y",
                         x=["x1", "twice", "sector"], categorical=["sector"])
    assert result.provenance["omitted_terms"] == ["twice"]
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "sector[b]", "sector[c]"]
    holes = sample.copy()
    holes.loc[[5, 9], "x2"] = np.nan
    with pytest.raises(AnalysisError) as excinfo:
        oe.tpoisson(data=holes, y="y", x=X)
    assert excinfo.value.code == "missing_values"
    dropped = oe.tpoisson(data=holes, y="y", x=X, missing="drop")
    assert dropped.nobs == len(sample) - 2
    assert_allclose(estimates(dropped),
                    estimates(oe.tpoisson(data=holes.dropna(subset=["x2"]), y="y", x=X)),
                    rtol=1e-10)
    # Without a constant: Wald test, no pseudo R-squared, a stationary point of the likelihood.
    no_constant = oe.tpoisson(data=sample, y="y", x=X, intercept=False)
    assert no_constant.tests["model"]["label"].startswith("Wald chi2")
    assert no_constant.metrics["pseudo_r_squared"] is None
    gradient = numeric_scores(lambda t: truncated_loglik(t, sample.y.to_numpy(),
                                                         design(sample)[:, 1:], 0),
                              estimates(no_constant)).sum(axis=0)
    assert np.abs(gradient).max() < 1e-4
    # x = [] is the constant-only model: the conditional mean equals the sample mean.
    constant = oe.tpoisson(data=sample, y="y", x=[])
    mu = np.exp(estimates(constant)[0])
    assert mu / -np.expm1(-mu) == pytest.approx(sample.y.mean(), rel=1e-9)
    assert constant.tests["model"]["df"] == 0


def test_tpoisson_json_summary_latex_and_public_access(data):
    sample = data[data.y > 0].reset_index(drop=True)
    result = oe.tpoisson(data=sample, y="y", x=X)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Truncated Poisson regression" in text and "pseudo_r_squared" in text
    assert "LR chi2 test" in text
    assert "x1" in result.to_latex()
    refit = oe.fit(result.spec, data=sample)
    assert_allclose(estimates(refit), estimates(result), rtol=1e-12)
    capability = oe.capabilities()["estimators"]["tpoisson"]
    assert capability["family"] == "count" and capability["stata"] == ["tpoisson"]
    assert "tpoisson" in oe.tpoisson.__doc__ and "Example" in oe.tpoisson.__doc__


# ---- tnbreg --------------------------------------------------------------------------------


@pytest.mark.parametrize(("form", "limit"), [("mean", 0), ("mean", 2), ("constant", 0),
                                             ("constant", 1)])
def test_tnbreg_matches_brute_force_and_information(data, form, limit):
    sample = data[data.ynb > limit].reset_index(drop=True)
    result = oe.tnbreg(data=sample, y="ynb", x=X, ll=limit, dispersion=form)
    y, x = sample.ynb.to_numpy(), design(sample)

    def total(theta):
        return truncated_loglik(theta, y, x, limit, form=form).sum()

    theta, value = brute_force(total, np.array([0.3, 0.0, 0.0, -0.3]))
    name = "alpha" if form == "mean" else "delta"
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", f"/ln{name}"]
    assert [c.equation for c in result.coefficients] == ["ynb"] * 3 + [None]
    assert_allclose(estimates(result), theta, rtol=5e-4, atol=5e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))), 1e-4)
    assert result.metrics[name] == pytest.approx(np.exp(estimates(result)[3]), rel=1e-12)
    record = result.extra["alpha"]
    assert record["parameter"] == name
    assert record["std_error"] == pytest.approx(result.metrics[name] * errors(result)[3],
                                                rel=1e-10)
    # LR test of alpha = 0 against the truncated Poisson model (boundary: chibar2(01)).
    poisson = oe.tpoisson(data=sample, y="ynb", x=X, ll=limit)
    statistic = 2 * (value - poisson.metrics["log_likelihood"])
    assert result.tests["alpha"]["statistic"] == pytest.approx(statistic, rel=1e-7)
    assert result.tests["alpha"]["p_value"] == pytest.approx(
        0.5 * stats.chi2.sf(statistic, 1), rel=1e-6, abs=1e-300)
    # Model test against the constant-only model with a free dispersion.
    _, null = brute_force(lambda t: truncated_loglik(t, y, x[:, :1], limit, form=form).sum(),
                          np.array([0.3, -0.3]))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-8)
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (value - null), rel=1e-5)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - value / null, rel=1e-6)
    assert result.metrics["aic"] == pytest.approx(-2 * value + 8, rel=1e-9)
    # Conditional mean in the chart sample.
    mu = np.exp(x @ estimates(result)[:3])
    dispersion = result.metrics[name]
    size = 1 / dispersion if form == "mean" else mu / dispersion
    p = size / (size + mu)
    lower = sum(j * stats.nbinom.pmf(j, size, p) for j in range(limit + 1))
    mean = (mu - lower) / stats.nbinom.sf(limit, size, p)
    for row in result.predictions[:25]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-8)


def test_tnbreg_matches_statsmodels(data):
    sample = data[data.ynb > 0].reset_index(drop=True)
    result = oe.tnbreg(data=sample, y="ynb", x=X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = TruncatedLFNegativeBinomialP(
            sample.ynb, sm.add_constant(sample[X]), truncation=0, p=2).fit(
                method="bfgs", maxiter=3000, disp=0, gtol=1e-10)
    assert_allclose(estimates(result)[:3], reference.params.to_numpy()[:3], rtol=1e-5)
    assert result.metrics["alpha"] == pytest.approx(reference.params.to_numpy()[3], rel=1e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10)
    # statsmodels reports alpha itself: its standard error is the delta-method one.
    assert result.extra["alpha"]["std_error"] == pytest.approx(reference.bse.to_numpy()[3],
                                                               rel=1e-3)
    assert_allclose(errors(result)[:3], reference.bse.to_numpy()[:3], rtol=1e-3)


def test_tnbreg_varying_truncation_covariances_and_weights(data):
    sample = data[data.ynb > data.cut].reset_index(drop=True)
    y, x, limit = sample.ynb.to_numpy(), design(sample), sample.cut.to_numpy()
    offset = sample.lnexpo.to_numpy()
    args = {"y": "ynb", "x": X, "ll": "cut", "exposure": "expo"}
    base = oe.tnbreg(data=sample, **args)
    theta = estimates(base)

    def per_observation(t):
        return truncated_loglik(t, y, x, limit, offset, form="mean")

    best, value = brute_force(lambda t: per_observation(t).sum(), np.array([0.3, 0, 0, -0.3]))
    assert_allclose(theta, best, rtol=5e-4, atol=5e-5)
    assert base.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(sample)
    close(covariance(base), bread, 1e-4)
    close(covariance(oe.tnbreg(data=sample, covariance="opg", **args)),
          np.linalg.inv(scores.T @ scores), 1e-4)
    robust = oe.tnbreg(data=sample, covariance="robust", **args)
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1), 1e-4)
    assert "alpha" not in robust.tests and "alpha_test_note" in robust.extra
    cluster = oe.tnbreg(data=sample, cluster="firm", **args)
    groups = sample.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, sample.firm) @ bread * groups / (groups - 1), 1e-4)
    repeated = sample.loc[sample.index.repeat(sample.fw.astype(int))].reset_index(drop=True)
    weighted = oe.tnbreg(data=sample, weights="fw", weight_type="fweight", **args)
    expanded = oe.tnbreg(data=repeated, **args)
    assert weighted.nobs == len(repeated)
    assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-7)
    assert_allclose(errors(weighted), errors(expanded), rtol=1e-6)
    for name in ("model", "alpha"):
        assert weighted.tests[name]["statistic"] == pytest.approx(
            expanded.tests[name]["statistic"], rel=1e-6)
    w = sample.aw.to_numpy() * n / sample.aw.sum()
    analytic = oe.tnbreg(data=sample, weights="aw", weight_type="aweight", **args)
    assert analytic.metrics["log_likelihood"] == pytest.approx(
        (w * per_observation(estimates(analytic))).sum(), rel=1e-10)
    gradient = (numeric_scores(per_observation, estimates(analytic)) * w[:, None]).sum(axis=0)
    assert np.abs(gradient).max() < 2e-4
    sampling = oe.tnbreg(data=sample, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-7)
    importance = oe.tnbreg(data=sample, weights="aw", weight_type="iweight", **args)
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-7)


def test_tnbreg_boundary_errors_and_reporting(data):
    # Truncated Poisson data are not overdispersed: alpha is estimated at zero.
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x1": rng.normal(size=3000)})
    frame["y"] = rng.binomial(10, 0.35 + 0.05 * np.tanh(frame.x1)).astype(float)
    frame = frame[frame.y > 0]
    with pytest.raises(AnalysisError) as excinfo:
        oe.tnbreg(data=frame, y="y", x=["x1"])
    assert excinfo.value.code == "boundary_solution" and "oe.tpoisson" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.tnbreg(data=data, y="ynb", x=X)
    assert excinfo.value.code == "outcome_not_truncated"
    with pytest.raises(AnalysisError) as excinfo:
        oe.tnbreg(data=data[data.ynb > 0], y="ynb", x=X, dispersion="quadratic")
    assert excinfo.value.code == "invalid_spec"
    sample = data[data.ynb > 0].reset_index(drop=True)
    result = oe.tnbreg(data=sample, y="ynb", x=X)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Truncated negative binomial regression" in text and "/lnalpha" in text
    assert "LR test of alpha = 0 against the truncated Poisson model" in text
    assert "lnalpha" in result.to_latex()
    assert oe.capabilities()["estimators"]["tnbreg"]["function"] == "tnbreg"


# ---- kernel: the truncation probability ------------------------------------------------------


def test_truncation_probability_is_accurate_in_both_tails():
    """ln Pr(Y > ll) keeps relative accuracy where 1 - cdf would cancel."""
    mpmath.mp.dps = 40
    mu = np.array([1e-8, 1e-4, 0.05, 1.0, 7.0, 40.0, 300.0])
    eta = torch.tensor(np.log(mu))
    for limit in (0, 1, 4, 25):
        pieces = TruncatedPieces(PoissonDensity(), torch.full((len(mu),), limit + 1.0,
                                                              dtype=torch.float64), limit)
        log_s = pieces.tail([eta], False)[0].numpy()
        exact = [float(mpmath.log1p(-mpmath.gammainc(limit + 1, m, mpmath.inf,
                                                      regularized=True)))
                 if m > limit + 1 else
                 float(mpmath.log(mpmath.gammainc(limit + 1, 0, m, regularized=True)))
                 for m in mu]
        assert_allclose(log_s, exact, rtol=1e-12, atol=1e-300)
    # Negative binomial with zero truncation: 1 - (1 + alpha mu)^(-1/alpha).
    tau = torch.tensor(np.log(0.7))
    pieces = TruncatedPieces(NegBinDensity("mean"), torch.ones(len(mu), dtype=torch.float64), 0)
    log_s = pieces.tail([eta, tau], False)[0].numpy()
    exact = [float(mpmath.log(1 - (1 + mpmath.mpf("0.7") * mpmath.mpf(float(m)))
                              ** (-1 / mpmath.mpf("0.7")))) for m in mu]
    assert_allclose(log_s, exact, rtol=1e-12)
    # A point where Pr(Y > ll) is not computable from the lower sum is rejected, not returned.
    far = TruncatedPieces(NegBinDensity("mean"), torch.full((1,), 31.0, dtype=torch.float64), 30)
    assert far.tail([torch.tensor([np.log(1e-4)]), tau], False) is None
