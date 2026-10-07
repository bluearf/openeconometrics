"""tobit, truncreg and intreg against independent oracles.

Every estimator is compared with a brute-force SciPy maximization of an
independently written NumPy likelihood in the REPORTED parameterization
(``sigma`` rather than ``ln sigma``, so the delta method is checked too). The
covariance estimators are rebuilt from complex-step scores and a numerical
Hessian of those likelihoods. The helpers at the top are shared by the other
``test_econ_limited_*`` files.
"""

import time

import mpmath
import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import optimize as sopt
from scipy import special, stats
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics.limited.kernels import CensoredObjective, TruncatedObjective
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec

KINDS = ("nonrobust", "opg", "robust", "cluster")
X = ["x1", "x2"]


# ---- shared oracle helpers ------------------------------------------------------------


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def terms(result):
    return [c.term for c in result.coefficients]


def design(frame, columns):
    return np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(dtype=float)])


def score_rows(loglik_obs, theta, *args):
    """Per-observation scores [n, k] by the complex-step method (exact to rounding)."""
    return approx_fprime_cs(np.asarray(theta, dtype=float), lambda t: loglik_obs(t, *args))


def hessian_of(loglik_obs, theta, weights, *args):
    """Hessian of the weighted log likelihood: central differences of the exact gradient."""
    def gradient(t):
        return (weights[:, None] * score_rows(loglik_obs, t, *args)).sum(axis=0)

    hessian = approx_fprime(np.asarray(theta, dtype=float), gradient, centered=True)
    return (hessian + hessian.T) / 2


def brute_force(loglik_obs, start, weights, *args):
    """Maximize ``sum w loglik_obs``: SciPy BFGS, then damped Newton on numerical derivatives."""
    def value(t):
        with np.errstate(all="ignore"):
            total = (weights * loglik_obs(t, *args)).sum()
        return total if np.isfinite(total) else -1e100

    def gradient(t):
        with np.errstate(all="ignore"):
            rows = (weights[:, None] * score_rows(loglik_obs, t, *args)).sum(axis=0)
        return np.where(np.isfinite(rows), rows, 0.0)

    with np.errstate(all="ignore"):
        solved = sopt.minimize(lambda t: -value(t), np.asarray(start, dtype=float),
                               jac=lambda t: -gradient(t), method="BFGS",
                               options={"gtol": 1e-6, "maxiter": 500})
    theta = solved.x if value(solved.x) > value(np.asarray(start, dtype=float)) \
        else np.asarray(start, dtype=float)
    for _ in range(100):
        slope = gradient(theta)
        values, vectors = np.linalg.eigh(-hessian_of(loglik_obs, theta, weights, *args))
        values = np.maximum(np.abs(values), 1e-10 * np.abs(values).max())
        step = vectors @ ((vectors.T @ slope) / values)
        size, current = 1.0, value(theta)
        while value(theta + size * step) < current - 1e-10 * abs(current) and size > 1e-10:
            size /= 2
        theta = theta + size * step
        if np.max(np.abs(size * step) / np.maximum(1.0, np.abs(theta))) < 1e-11:
            break
    assert np.max(np.abs(gradient(theta))) < 1e-5 * max(1.0, abs(value(theta)))
    return theta, value(theta)


def ml_covariance(loglik_obs, theta, weights, kind, *args, groups=None, frequency=None):
    """Stata's ml covariance conventions rebuilt from numerical derivatives.

    ``weights`` multiply the likelihood; ``frequency`` marks them as fweights
    (N = sum of weights and each row counts f times in the meat).
    """
    scores = score_rows(loglik_obs, theta, *args)
    bread = np.linalg.inv(-hessian_of(loglik_obs, theta, weights, *args))
    if kind == "nonrobust":
        return bread
    weighted = scores * weights[:, None]
    nobs = weights.sum() if frequency else len(weights)
    meat = weighted.T @ scores if frequency else weighted.T @ weighted
    if kind == "opg":
        # The OPG estimates the information of sum w l, which is linear in w for every
        # weight type (an integer iweight then agrees with the equal fweight).
        return np.linalg.inv(weighted.T @ scores)
    if kind == "robust":
        return nobs / (nobs - 1) * bread @ meat @ bread
    codes = pd.factorize(groups)[0]
    count = codes.max() + 1
    sums = np.zeros((count, scores.shape[1]))
    np.add.at(sums, codes, weighted)
    return count / (count - 1) * bread @ (sums.T @ sums) @ bread


def check_fit(result, loglik_obs, start, *args, weights=None, groups=None, frequency=False,
              rtol=2e-6, atol=1e-8, solution=None):
    """Estimates, log likelihood and covariance of ``result`` against the brute-force oracle.

    ``solution`` passes an oracle maximum computed earlier (``brute_force`` output).
    """
    n = len(args[0])
    weights = np.ones(n) if weights is None else weights
    theta, log_likelihood = solution or brute_force(loglik_obs, start, weights, *args)
    assert_allclose(estimates(result), theta, rtol=rtol, atol=1e-7)
    assert_allclose(result.metrics["log_likelihood"], log_likelihood, rtol=1e-10)
    expected = ml_covariance(loglik_obs, theta, weights, result.spec.covariance, *args,
                             groups=groups, frequency=frequency)
    assert_allclose(covariance(result), expected, rtol=2e-5, atol=atol)
    return theta, log_likelihood


def round_trip(result):
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert result.title in text and all(term in text for term in terms(result))
    latex = result.to_latex()
    assert ("tabular" in latex or "longtable" in latex) and "Observations" in latex
    return text


# ---- independent likelihoods ----------------------------------------------------------


def tobit_loglik(theta, x, y, ll, ul, offset=None):
    """Censored normal log likelihood per observation in (b, sigma)."""
    k = x.shape[1]
    mean = x @ theta[:k] + (0 if offset is None else offset)
    sigma = theta[k]
    left = np.zeros(len(y), bool) if ll is None else y <= ll
    right = np.zeros(len(y), bool) if ul is None else y >= ul
    density = -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    lower = np.log(special.ndtr(((ll if ll is not None else 0.0) - mean) / sigma))
    upper = np.log(special.ndtr((mean - (ul if ul is not None else 0.0)) / sigma))
    return np.where(left, lower, np.where(right, upper, density))


def truncreg_loglik(theta, x, y, ll, ul):
    """Truncated normal log likelihood per observation in (b, sigma)."""
    k = x.shape[1]
    mean, sigma = x @ theta[:k], theta[k]
    density = -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    top = 1.0 if ul is None else special.ndtr((ul - mean) / sigma)
    bottom = 0.0 if ll is None else special.ndtr((ll - mean) / sigma)
    return density - np.log(top - bottom)


def intreg_loglik(theta, x, low, high):
    """Interval-regression log likelihood per observation in (b, ln sigma)."""
    k = x.shape[1]
    mean, sigma = x @ theta[:k], np.exp(theta[k])
    open_low, open_high = np.isnan(low), np.isnan(high)
    point = ~open_low & ~open_high & (low == high)
    a = np.where(open_low, 0.0, low)
    b = np.where(open_high, 0.0, high)
    density = -0.5 * ((a - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    top = np.where(open_high, 1.0, special.ndtr((b - mean) / sigma))
    bottom = np.where(open_low, 0.0, special.ndtr((a - mean) / sigma))
    return np.where(point, density, np.log(np.where(point, 1.0, top - bottom)))


# ---- data -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261)
    n = 700
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "firm": np.repeat(np.arange(70), 10), "state": rng.integers(0, 9, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "off": rng.normal(scale=0.3, size=n),
    })
    shock = rng.normal(size=n) * (1 + 0.3 * np.repeat(rng.normal(size=70), 10))
    frame["latent"] = 0.4 + 0.9 * frame.x1 - 0.6 * frame.x2 + 1.3 * shock
    frame["y"] = frame.latent.clip(-0.5, 2.5)
    # Interval data: unit-wide brackets, open tails and a block of exact observations.
    frame["lo"] = np.floor(frame.latent)
    frame["hi"] = frame.lo + 1.0
    frame.loc[frame.latent < -1, ["lo", "hi"]] = [np.nan, -1.0]
    frame.loc[frame.latent > 3, ["lo", "hi"]] = [3.0, np.nan]
    exact = np.arange(n) % 5 == 0
    frame.loc[exact, "lo"] = frame.latent[exact]
    frame.loc[exact, "hi"] = frame.latent[exact]
    return frame


def tobit_start(frame, extra=0):
    return np.r_[np.zeros(3 + extra), 1.0]


# ---- tobit ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_tobit_matches_brute_force_likelihood(data, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, **options)
    x, y = design(data, X), data.y.to_numpy()
    theta, log_likelihood = check_fit(result, tobit_loglik, tobit_start(data), x, y, -0.5, 2.5,
                                      groups=data.firm)
    assert terms(result) == ["Intercept", "x1", "x2", "/sigma"]
    assert [c.equation for c in result.coefficients] == ["y", "y", "y", None]
    assert result.spec.covariance == kind and result.nobs == len(data)
    metrics = result.metrics
    assert metrics["n_left_censored"] == (y <= -0.5).sum()
    assert metrics["n_right_censored"] == (y >= 2.5).sum()
    assert metrics["n_uncensored"] == ((y > -0.5) & (y < 2.5)).sum()
    assert_allclose(metrics["sigma"], theta[3], rtol=1e-6)
    assert_allclose(metrics["aic"], -2 * log_likelihood + 2 * 4, rtol=1e-10)
    assert_allclose(metrics["bic"], -2 * log_likelihood + 4 * np.log(len(data)), rtol=1e-10)
    # Student t with N - df_model degrees of freedom (Stata's tobit).
    df = len(data) - 2
    assert metrics["df_model"] == 2 and metrics["df_resid"] == df
    assert result.inference["use_t"] and result.inference["df_inference"] == df
    statistic = estimates(result) / errors(result)
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.t.sf(np.abs(statistic), df), rtol=1e-8, atol=1e-300)
    half = stats.t.ppf(0.975, df) * errors(result)
    assert_allclose([c.ci_high for c in result.coefficients], estimates(result) + half, rtol=1e-9)
    # Constant-only model: LR test and McFadden's pseudo R-squared.
    null, null_ll = brute_force(tobit_loglik, [0.5, 1.5], np.ones(len(y)), x[:, :1], y, -0.5, 2.5)
    assert_allclose(result.extra["null_log_likelihood"], null_ll, rtol=1e-10)
    assert_allclose(metrics["pseudo_r_squared"], 1 - log_likelihood / null_ll, rtol=1e-9)
    test = result.tests["model"]
    if kind in {"nonrobust", "opg"}:
        assert test["distribution"] == "chi2" and test["df"] == 2
        assert_allclose(test["statistic"], 2 * (log_likelihood - null_ll), rtol=1e-8)
        assert_allclose(test["p_value"], stats.chi2.sf(test["statistic"], 2), rtol=1e-8)
    else:
        v = covariance(result)[1:3, 1:3]
        wald = theta[1:3] @ np.linalg.solve(v, theta[1:3]) / 2
        assert test["distribution"] == "F" and (test["df"], test["df2"]) == (2, df)
        assert_allclose(test["statistic"], wald, rtol=1e-5)
        assert_allclose(test["p_value"], stats.f.sf(wald, 2, df), rtol=1e-4, atol=1e-300)
    variance = result.extra["variance"]
    assert_allclose(variance["estimate"], theta[3] ** 2, rtol=1e-6)
    assert_allclose(variance["std_error"], 2 * theta[3] * errors(result)[3], rtol=1e-9)


def test_tobit_one_sided_limits_offset_and_spec_path(data):
    x, y = design(data, X), data.y.to_numpy()
    lower = oe.tobit(data=data, y="y", x=X, ll="min")
    check_fit(lower, tobit_loglik, tobit_start(data), x, y, y.min(), None)
    assert lower.extra["limits"] == {"lower": y.min(), "upper": None}
    assert lower.metrics["n_right_censored"] == 0
    upper = oe.tobit(data=data, y="y", x=X, ul="max", offset="off", covariance="robust")
    check_fit(upper, tobit_loglik, tobit_start(data), x, y, None, y.max(), data.off.to_numpy())
    assert upper.extra["limits"] == {"lower": None, "upper": y.max()}
    spec = ModelSpec(estimator="tobit", outcome="y", predictors=X, options={"ll_at_min": True})
    same = fit(spec, data=data)
    assert_allclose(estimates(same), estimates(lower), rtol=1e-12)
    assert spec.covariance == "nonrobust"
    # Categorical regressors are treatment coded.
    coded = oe.tobit(data=data, y="y", x=["x1", "sector"], categorical=["sector"], ll=-0.5)
    assert terms(coded) == ["Intercept", "x1", "sector[b]", "sector[c]", "/sigma"]
    dummies = np.column_stack([np.ones(len(data)), data.x1, data.sector == "b",
                               data.sector == "c"]).astype(float)
    check_fit(coded, tobit_loglik, np.r_[np.zeros(4), 1.0], dummies, y, -0.5, None)


def test_tobit_without_censoring_is_least_squares_with_ml_sigma(data):
    result = oe.tobit(data=data, y="latent", x=X)
    x, y = design(data, X), data.latent.to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sigma = np.sqrt(resid @ resid / len(y))
    assert_allclose(estimates(result), np.r_[beta, sigma], rtol=1e-10)
    assert_allclose(covariance(result)[:3, :3], sigma ** 2 * np.linalg.inv(x.T @ x), rtol=1e-8)
    assert_allclose(errors(result)[3], sigma / np.sqrt(2 * len(y)), rtol=1e-8)
    assert result.metrics["n_uncensored"] == len(y)
    log_likelihood = stats.norm.logpdf(resid, scale=sigma).sum()
    assert_allclose(result.metrics["log_likelihood"], log_likelihood, rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_tobit_frequency_weights_replicate_rows(data, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    weighted = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, weights="fw",
                        weight_type="fweight", **options)
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    replicated = oe.tobit(data=expanded, y="y", x=X, ll=-0.5, ul=2.5, **options)
    assert weighted.nobs == replicated.nobs == int(data.fw.sum())
    assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-9)
    assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-7, atol=1e-12)
    for name in ("log_likelihood", "pseudo_r_squared", "n_left_censored", "n_uncensored",
                 "n_right_censored", "df_resid"):
        assert_allclose(weighted.metrics[name], replicated.metrics[name], rtol=1e-9)
    assert_allclose(weighted.tests["model"]["statistic"], replicated.tests["model"]["statistic"],
                    rtol=1e-7)


def test_tobit_analytic_sampling_and_importance_weights(data):
    x, y = design(data, X), data.y.to_numpy()
    aw = data.aw.to_numpy()
    scaled = aw * len(aw) / aw.sum()
    analytic = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, weights="aw",
                        weight_type="aweight")
    check_fit(analytic, tobit_loglik, tobit_start(data), x, y, -0.5, 2.5, weights=scaled)
    importance = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, weights="aw",
                          weight_type="iweight", covariance="opg")
    check_fit(importance, tobit_loglik, tobit_start(data), x, y, -0.5, 2.5, weights=aw)
    sampling = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, weights="aw",
                        weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    check_fit(sampling, tobit_loglik, tobit_start(data), x, y, -0.5, 2.5, weights=aw)
    doubled = data.assign(aw=2 * data.aw)
    rescaled = oe.tobit(data=doubled, y="y", x=X, ll=-0.5, ul=2.5, weights="aw",
                        weight_type="pweight", cluster="firm")
    clustered = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, weights="aw",
                         weight_type="pweight", cluster="firm")
    check_fit(clustered, tobit_loglik, tobit_start(data), x, y, -0.5, 2.5, weights=aw,
              groups=data.firm)
    assert_allclose(covariance(rescaled), covariance(clustered), rtol=1e-8)
    with pytest.raises(AnalysisError) as error:
        oe.tobit(data=data, y="y", x=X, ll=-0.5, weights="aw", weight_type="pweight",
                 covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


def test_tobit_two_way_clustering(data):
    result = oe.tobit(data=data, y="y", x=X, ll=-0.5, cluster=["firm", "state"])
    x, y = design(data, X), data.y.to_numpy()
    theta, _ = brute_force(tobit_loglik, tobit_start(data), np.ones(len(y)), x, y, -0.5, None)
    scores = score_rows(tobit_loglik, theta, x, y, -0.5, None)
    bread = np.linalg.inv(-hessian_of(tobit_loglik, theta, np.ones(len(y)), x, y, -0.5, None))

    def meat(keys):
        codes = pd.factorize(keys)[0]
        sums = np.zeros((codes.max() + 1, scores.shape[1]))
        np.add.at(sums, codes, scores)
        return sums.T @ sums

    both = data.groupby(["firm", "state"]).ngroup().to_numpy()
    groups = min(data.firm.nunique(), data.state.nunique())
    total = meat(data.firm) + meat(data.state) - meat(both)
    expected = groups / (groups - 1) * bread @ total @ bread
    assert_allclose(covariance(result), expected, rtol=2e-5, atol=1e-9)
    assert result.inference["cluster_columns"] == ["firm", "state"]


# ---- truncreg -------------------------------------------------------------------------


@pytest.mark.parametrize("limits", [(-0.5, None), (None, 2.5), (-0.5, 2.5)])
@pytest.mark.parametrize("kind", KINDS)
def test_truncreg_matches_brute_force_likelihood(data, limits, kind):
    ll, ul = limits
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = oe.truncreg(data=data, y="latent", x=X, ll=ll, ul=ul, **options)
    keep = np.ones(len(data), bool)
    if ll is not None:
        keep &= data.latent.to_numpy() > ll
    if ul is not None:
        keep &= data.latent.to_numpy() < ul
    sample = data[keep]
    x, y = design(sample, X), sample.latent.to_numpy()
    theta, log_likelihood = check_fit(result, truncreg_loglik, tobit_start(data), x, y, ll, ul,
                                      groups=sample.firm)
    assert terms(result) == ["Intercept", "x1", "x2", "/sigma"]
    assert result.nobs == keep.sum() and result.metrics["n_truncated"] == (~keep).sum()
    assert result.dropped_rows == (~keep).sum()
    assert result.sample_positions == np.flatnonzero(keep).tolist()
    assert any("truncat" in warning for warning in result.warnings)
    assert not result.inference["use_t"]
    v = covariance(result)[1:3, 1:3]
    test = result.tests["model"]
    assert test["distribution"] == "chi2" and test["df"] == 2
    assert_allclose(test["statistic"], theta[1:3] @ np.linalg.solve(v, theta[1:3]), rtol=1e-5)
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.norm.sf(np.abs(estimates(result) / errors(result))), rtol=1e-8,
                    atol=1e-300)
    assert_allclose(result.metrics["aic"], -2 * log_likelihood + 8, rtol=1e-10)


def test_truncreg_with_limits_outside_the_data_is_least_squares(data):
    x, y = design(data, X), data.latent.to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    sigma = np.sqrt(((y - x @ beta) ** 2).sum() / len(y))
    free = oe.truncreg(data=data, y="latent", x=X)
    assert_allclose(estimates(free), np.r_[beta, sigma], rtol=1e-10)
    assert free.metrics["n_truncated"] == 0 and not free.warnings
    wide = oe.truncreg(data=data, y="latent", x=X, ll=-60.0, ul=60.0)
    assert_allclose(estimates(wide), np.r_[beta, sigma], rtol=1e-8)
    assert_allclose(covariance(wide), covariance(free), rtol=1e-6, atol=1e-12)


def test_truncreg_weights_and_offset(data):
    sample = data[data.latent > -0.5].reset_index(drop=True)
    weighted = oe.truncreg(data=data, y="latent", x=X, ll=-0.5, weights="fw",
                           weight_type="fweight", covariance="robust")
    expanded = sample.loc[sample.index.repeat(sample.fw.astype(int))].reset_index(drop=True)
    replicated = oe.truncreg(data=expanded, y="latent", x=X, ll=-0.5, covariance="robust")
    assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-9)
    assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-7)
    assert weighted.nobs == int(sample.fw.sum())
    assert weighted.metrics["n_truncated"] == int(data.fw[data.latent <= -0.5].sum())
    aw = sample.aw.to_numpy()
    analytic = oe.truncreg(data=data, y="latent", x=X, ll=-0.5, weights="aw",
                           weight_type="aweight")
    check_fit(analytic, truncreg_loglik, tobit_start(data), design(sample, X),
              sample.latent.to_numpy(), -0.5, None, weights=aw * len(aw) / aw.sum())
    shifted = data.assign(latent=data.latent + data.off)
    with_offset = oe.truncreg(data=shifted[shifted.latent > 0], y="latent", x=X, ll=0.0,
                              offset="off")
    kept = shifted[shifted.latent > 0]

    def offset_loglik(theta, x, y, off):
        return truncreg_loglik_offset(theta, x, y, off, 0.0)

    check_fit(with_offset, offset_loglik, tobit_start(data), design(kept, X),
              kept.latent.to_numpy(), kept.off.to_numpy())


def truncreg_loglik_offset(theta, x, y, offset, ll):
    k = x.shape[1]
    mean, sigma = x @ theta[:k] + offset, theta[k]
    density = -0.5 * ((y - mean) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    return density - np.log(1.0 - special.ndtr((ll - mean) / sigma))


# ---- intreg ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_intreg_matches_brute_force_likelihood(data, kind):
    options = {"cluster": "firm"} if kind == "cluster" else {"covariance": kind}
    result = oe.intreg(data=data, y_low="lo", y_high="hi", x=X, **options)
    x, low, high = design(data, X), data.lo.to_numpy(), data.hi.to_numpy()
    theta, log_likelihood = check_fit(result, intreg_loglik, np.r_[np.zeros(3), 0.0], x, low,
                                      high, groups=data.firm)
    assert terms(result) == ["Intercept", "x1", "x2", "/lnsigma"]
    metrics = result.metrics
    assert metrics["n_left_censored"] == np.isnan(low).sum()
    assert metrics["n_right_censored"] == np.isnan(high).sum()
    assert metrics["n_uncensored"] == (low == high).sum()
    assert metrics["n_interval"] == (low < high).sum()
    assert sum(metrics[name] for name in ("n_left_censored", "n_right_censored", "n_uncensored",
                                          "n_interval")) == len(data)
    sigma = result.extra["sigma"]
    assert_allclose(sigma["estimate"], np.exp(theta[3]), rtol=1e-6)
    assert_allclose(metrics["sigma"], sigma["estimate"])
    assert_allclose(sigma["std_error"], np.exp(theta[3]) * errors(result)[3], rtol=1e-9)
    assert_allclose([sigma["ci_low"], sigma["ci_high"]],
                    np.exp(theta[3] + np.array([-1, 1]) * stats.norm.ppf(0.975)
                           * errors(result)[3]), rtol=1e-8)
    null, null_ll = brute_force(intreg_loglik, [0.5, 0.3], np.ones(len(low)), x[:, :1], low, high)
    test = result.tests["model"]
    if kind in {"nonrobust", "opg"}:
        assert_allclose(test["statistic"], 2 * (log_likelihood - null_ll), rtol=1e-8)
        assert test["df"] == 2 and test["distribution"] == "chi2"
    else:
        v = covariance(result)[1:3, 1:3]
        assert_allclose(test["statistic"], theta[1:3] @ np.linalg.solve(v, theta[1:3]), rtol=1e-5)
    assert not result.inference["use_t"]
    assert len(result.predictions) == 400
    assert all(np.isfinite(list(row.values())).all() for row in result.predictions)


def test_intreg_reproduces_tobit_and_frequency_weights(data):
    coded = data.assign(
        low=np.where(data.y <= -0.5, np.nan, data.y.where(data.y < 2.5, 2.5)),
        high=np.where(data.y >= 2.5, np.nan, data.y.where(data.y > -0.5, -0.5)))
    interval = oe.intreg(data=coded, y_low="low", y_high="high", x=X)
    censored = oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5)
    assert_allclose(estimates(interval)[:3], estimates(censored)[:3], rtol=1e-9)
    assert_allclose(np.exp(estimates(interval)[3]), estimates(censored)[3], rtol=1e-9)
    assert_allclose(interval.metrics["log_likelihood"], censored.metrics["log_likelihood"],
                    rtol=1e-12)
    assert_allclose(interval.tests["model"]["statistic"], censored.tests["model"]["statistic"],
                    rtol=1e-8)
    weighted = oe.intreg(data=data, y_low="lo", y_high="hi", x=X, weights="fw",
                         weight_type="fweight", cluster="firm")
    expanded = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    replicated = oe.intreg(data=expanded, y_low="lo", y_high="hi", x=X, cluster="firm")
    assert_allclose(estimates(weighted), estimates(replicated), rtol=1e-9)
    assert_allclose(covariance(weighted), covariance(replicated), rtol=1e-7)
    assert weighted.metrics["n_interval"] == replicated.metrics["n_interval"]
    aw = data.aw.to_numpy()
    analytic = oe.intreg(data=data, y_low="lo", y_high="hi", x=X, weights="aw",
                         weight_type="aweight", offset="off")

    def offset_loglik(theta, x, low, high, off):
        return intreg_loglik(theta, x, low - off, high - off)

    check_fit(analytic, offset_loglik, np.r_[np.zeros(3), 0.0], design(data, X),
              data.lo.to_numpy(), data.hi.to_numpy(), data.off.to_numpy(),
              weights=aw * len(aw) / aw.sum())


# ---- derivatives ----------------------------------------------------------------------


def test_censored_and_truncated_kernels_have_exact_derivatives(data):
    x = torch.tensor(design(data, X))
    low = torch.tensor(data.lo.fillna(-np.inf).to_numpy())
    high = torch.tensor(data.hi.fillna(np.inf).to_numpy())
    weights = torch.tensor(data.aw.to_numpy())
    offset = torch.tensor(data.off.to_numpy())
    theta = torch.tensor([0.2, 0.7, -0.4, 0.35], dtype=torch.float64)
    objectives = [CensoredObjective(x, low, high, weights, offset)]
    y = torch.tensor(data.latent.to_numpy())
    for ll, ul in ((-4.0, None), (None, 5.0), (-4.0, 5.0), (None, None)):
        objectives.append(TruncatedObjective(x, y, weights, ll, ul, offset))
    for objective in objectives:
        report = check_derivatives(objective, theta)
        assert report["gradient_max_rel_error"] < 1e-8
        assert report["hessian_max_rel_error"] < 1e-7
        assert report["hessian_asymmetry"] < 1e-9
        value, gradient, _ = objective(theta)
        assert_allclose(float(objective.value(theta)), float(value), rtol=1e-13)
        rows = objective.score_rows(theta)
        assert_allclose((rows * weights[:, None]).sum(dim=0).numpy(), gradient.numpy(),
                        rtol=1e-9, atol=1e-9)


def test_censored_kernel_is_stable_far_in_the_tails():
    # Observations 30+ standard deviations beyond their limit keep finite, accurate terms.
    x = torch.ones((4, 1), dtype=torch.float64)
    low = torch.tensor([-np.inf, 35.0, 30.0, -38.0])
    high = torch.tensor([-35.0, np.inf, 31.0, -37.5])
    objective = CensoredObjective(x, low.double(), high.double(), torch.ones(4).double())
    theta = torch.tensor([0.0, 0.0], dtype=torch.float64)
    value, gradient, hessian = objective(theta)
    mpmath.mp.dps = 40
    expected = float(2 * mpmath.log(mpmath.ncdf(-35))
                     + mpmath.log(mpmath.ncdf(-30) - mpmath.ncdf(-31))
                     + mpmath.log(mpmath.ncdf(-37.5) - mpmath.ncdf(-38)))
    assert_allclose(float(value), expected, rtol=1e-12)
    assert bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(hessian).all())
    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-7 and report["hessian_max_rel_error"] < 1e-6


# ---- sample handling and failures -----------------------------------------------------


def test_collinearity_missing_values_and_restrictions(data):
    frame = data.assign(copy=data.x1 * 2, one=1.0)
    result = oe.tobit(data=frame, y="y", x=["x1", "copy", "x2", "one"], ll=-0.5)
    assert terms(result) == ["Intercept", "x1", "x2", "/sigma"]
    assert result.provenance["omitted_terms"] == ["copy", "one"]
    assert any("collinearity" in warning for warning in result.warnings)
    assert_allclose(estimates(result), estimates(oe.tobit(data=data, y="y", x=X, ll=-0.5)),
                    rtol=1e-8)
    for function, arguments in ((oe.truncreg, {"y": "latent", "ll": -0.5}),
                                (oe.intreg, {"y_low": "lo", "y_high": "hi"})):
        fitted = function(data=frame, x=["x1", "copy", "x2"], **arguments)
        assert fitted.provenance["omitted_terms"] == ["copy"]
    holes = data.copy()
    holes.loc[[3, 40], "x1"] = np.nan
    holes.loc[7, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.tobit(data=holes, y="y", x=X, ll=-0.5)
    assert error.value.code == "missing_values"
    dropped = oe.tobit(data=holes, y="y", x=X, ll=-0.5, missing="drop")
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert not {3, 7, 40} & set(dropped.sample_positions)
    complete = oe.tobit(data=holes.dropna(subset=["x1", "y"]), y="y", x=X, ll=-0.5)
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-12)
    # intreg: one missing bound is data, two missing bounds are a missing observation.
    gaps = data.copy()
    gaps.loc[[5, 6], ["lo", "hi"]] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.intreg(data=gaps, y_low="lo", y_high="hi", x=X)
    assert error.value.code == "missing_values"
    kept = oe.intreg(data=gaps, y_low="lo", y_high="hi", x=X, missing="drop")
    assert kept.nobs == len(data) - 2
    assert any("both interval bounds" in warning for warning in kept.warnings)
    reference = oe.intreg(data=gaps.drop(index=[5, 6]), y_low="lo", y_high="hi", x=X)
    assert_allclose(estimates(kept), estimates(reference), rtol=1e-12)
    zero = data.assign(fw=np.where(np.arange(len(data)) < 20, 0.0, data.fw))
    trimmed = oe.tobit(data=zero, y="y", x=X, ll=-0.5, weights="fw", weight_type="fweight")
    assert trimmed.nobs == int(zero.fw.sum()) and trimmed.dropped_rows == 20


def test_error_codes(data):
    def code(function, **arguments):
        with pytest.raises(AnalysisError) as error:
            function(data=arguments.pop("frame", data), **arguments)
        return error.value.code

    assert code(oe.tobit, y="y", x=X, ll=2.0, ul=1.0) == "invalid_limits"
    assert code(oe.tobit, y="y", x=X, ll="max") == "invalid_spec"
    assert code(oe.tobit, y="y", x=X, ll=float("nan")) == "invalid_spec"
    assert code(oe.tobit, y="y", x="x1", ll=0) == "invalid_spec"
    assert code(oe.tobit, y="y", x=X, ll=0, covariance="HC1") == "invalid_spec"
    assert code(oe.tobit, y="y", x=X, ll=0, weights="aw", weight_type="bogus") == "invalid_spec"
    assert code(oe.tobit, y="y", x=X, ll=100.0) == "no_uncensored_observations"
    assert code(oe.tobit, y="y", x=["x1", "nope"], ll=0) == "missing_columns"
    assert code(oe.tobit, y="y", x=X, ll=0, cluster="firm", covariance="robust") == "invalid_spec"
    assert code(oe.tobit, y="y", x=X, ll=0, weights="neg", weight_type="aweight",
                frame=data.assign(neg=-1.0)) == "negative_weights"
    assert code(oe.tobit, y="y", x=X, ll=0, weights="aw", weight_type="fweight") \
        == "noninteger_frequency_weights"
    assert code(oe.tobit, y="sector", x=X, ll=0) == "non_numeric_column"
    assert code(oe.tobit, y="y", x=X, ll=-0.5, frame=data.head(3)) == "insufficient_observations"
    assert code(oe.truncreg, y="latent", x=X, ll=50.0) == "empty_sample"
    assert code(oe.truncreg, y="latent", x=X, ll=3.0, ul=1.0) == "invalid_limits"
    assert code(oe.truncreg, y="latent", x=X, ll="min") == "invalid_spec"
    assert code(oe.intreg, y_low="hi", y_high="lo", x=X) == "invalid_interval"
    assert code(oe.intreg, y_low="lo", y_high="lo", x=X) == "invalid_spec"
    assert code(oe.intreg, y_low="lo", y_high="x1", x=X) == "invalid_spec"
    one_sided = data.assign(lo=np.nan, hi=1.0)
    assert code(oe.intreg, y_low="lo", y_high="hi", x=X, frame=one_sided) \
        == "no_uncensored_observations"
    infinite = data.assign(hi=data.hi.fillna(np.inf))
    assert code(oe.intreg, y_low="lo", y_high="hi", x=X, frame=infinite) == "non_finite_values"
    with pytest.raises(Exception) as error:
        ModelSpec(estimator="tobit", outcome="y", predictors=X, options={"limit": 0})
    assert "no option 'limit'" in str(error.value)
    # Censoring on both sides without a single uncensored outcome: the likelihood rises
    # monotonically in sigma, which is diagnosed before any iteration.
    separated = data.assign(y=np.where(data.x1 > 0, 1.0, 0.0))
    assert code(oe.tobit, y="y", x=["x1"], ll=0.0, ul=1.0, frame=separated) \
        == "no_uncensored_observations"
    # A monotone likelihood is reported, never returned as estimates.
    perfect = data.assign(flag=(data.y > -0.5) * 1.0)
    assert code(oe.tobit, y="y", x=["x1", "flag"], ll=-0.5, frame=perfect) == "nonconvergence"


def test_results_round_trip_render_and_are_exported(data):
    results = [
        oe.tobit(data=data, y="y", x=X, ll=-0.5, ul=2.5, cluster="firm"),
        oe.truncreg(data=data, y="latent", x=X, ll=-0.5),
        oe.intreg(data=data, y_low="lo", y_high="hi", x=X, covariance="robust"),
    ]
    for result, name in zip(results, ("tobit", "truncreg", "intreg"), strict=True):
        text = round_trip(result)
        assert result.spec.estimator == name and result.provenance["family"] == "limited"
        assert result.provenance["stata_parity_validated"] is False
        assert result.provenance["optimizer"]["converged"]
        assert "log_likelihood" in text
        assert name in oe.capabilities()["estimators"]
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
    assert "Student t" in results[0].inference["correction"]
    assert results[0].inference["cluster_count"] == 70


def test_censored_models_scale_linearly():
    rng = np.random.default_rng(5)
    n = 200_000
    frame = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"x{i}" for i in range(8)])
    latent = 0.3 + frame.to_numpy() @ np.linspace(-0.5, 0.5, 8) + rng.normal(size=n)
    frame["y"] = np.clip(latent, 0.0, None)
    frame["latent"] = latent
    frame["lo"] = np.floor(latent)
    frame["hi"] = frame.lo + 1
    columns = [f"x{i}" for i in range(8)]
    started = time.perf_counter()
    censored = oe.tobit(data=frame, y="y", x=columns, ll=0, covariance="robust")
    truncated = oe.truncreg(data=frame, y="latent", x=columns, ll=0)
    interval = oe.intreg(data=frame, y_low="lo", y_high="hi", x=columns)
    elapsed = time.perf_counter() - started
    assert elapsed < 20
    for result in (censored, truncated, interval):
        assert_allclose(estimates(result)[1:9], np.linspace(-0.5, 0.5, 8), atol=0.03)
