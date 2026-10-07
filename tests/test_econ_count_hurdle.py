"""churdle and hurdle against independent oracles.

Oracles: the two-part log likelihoods written with scipy.stats and maximized
by scipy.optimize; statsmodels Probit / Logit for the selection equation,
TruncatedLFPoisson and HurdleCountModel for the count hurdle; least squares of
ln(y) for the lognormal hurdle; numerical scores and Hessians of the joint
likelihood for every covariance estimator; duplicated rows for frequency weights.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize, special, stats
from statsmodels.discrete.truncated_model import HurdleCountModel, TruncatedLFPoisson

import openecon as oe
from openecon.analysis import AnalysisError

X = ["x1", "x2"]
Z = ["z1", "x1"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(424242)
    n = 2500
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n), "z1": rng.normal(size=n),
        "firm": np.repeat(np.arange(125), 20),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 2.0, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
    })
    frame["lnexpo"] = np.log(frame.expo)
    # Continuous outcomes with a mass at the limit (Cragg).
    takes_part = 0.2 + 0.7 * frame.z1 - 0.3 * frame.x1 + rng.normal(size=n) > 0
    frame["amount"] = np.where(
        takes_part, np.exp(0.8 + 0.5 * frame.x1 - 0.2 * frame.x2 + 0.6 * rng.normal(size=n)), 0.0)
    latent = 1.5 + 1.0 * frame.x1 - 0.5 * frame.x2 + 1.2 * rng.normal(size=n)
    frame["hours"] = np.where(takes_part & (latent > 0), latent, 0.0)
    shifted = 3.0 + 1.0 * frame.x1 + 1.2 * rng.normal(size=n)
    frame["floor2"] = np.where(takes_part & (shifted > 2), shifted, 2.0)
    # Counts with a separate participation process.
    mu = frame.expo * np.exp(0.5 + 0.4 * frame.x1 - 0.3 * frame.x2)
    positive = rng.poisson(mu)
    positive_nb = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    for _ in range(200):
        again = positive == 0
        positive[again] = rng.poisson(mu[again])
        again = positive_nb == 0
        positive_nb[again] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu[again]))
    visits = rng.uniform(size=n) < special.expit(0.3 + 0.6 * frame.z1 - 0.4 * frame.x1)
    frame["count"] = np.where(visits, positive, 0).astype(float)
    frame["count_nb"] = np.where(visits, positive_nb, 0).astype(float)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def design(frame, columns):
    return sm.add_constant(frame[list(columns)]).to_numpy()


def link_cdf(index, link):
    if link == "logit":
        return special.expit(index)
    if link == "probit":
        return stats.norm.cdf(index)
    return -np.expm1(-np.exp(index))


def churdle_loglik(theta, y, x, z, limit, kind, link="probit"):
    """Per-observation Cragg hurdle log likelihood in (b, g, ln sigma)."""
    k, q = x.shape[1], z.shape[1]
    xb, sigma = x @ theta[:k], np.exp(theta[k + q])
    probability = link_cdf(z @ theta[k:k + q], link)
    selected = y > limit
    safe = np.where(selected, y, limit + 1.0)
    if kind == "linear":
        density = stats.norm.logpdf(safe, xb, sigma) - stats.norm.logcdf((xb - limit) / sigma)
    else:
        density = stats.norm.logpdf(np.log(safe), xb, sigma) - np.log(safe)
        if limit > 0:
            density = density - stats.norm.logcdf((xb - np.log(limit)) / sigma)
    return np.where(selected, np.log(probability) + density, np.log1p(-probability))


def hurdle_loglik(theta, y, x, z, offset=0.0, link="logit", negbin=False):
    """Per-observation count hurdle log likelihood in (b, g[, ln alpha])."""
    k, q = x.shape[1], z.shape[1]
    mu = np.exp(x @ theta[:k] + offset)
    probability = link_cdf(z @ theta[k:k + q], link)
    safe = np.where(y > 0, y, 1.0)
    if negbin:
        size = np.exp(-theta[k + q])
        p = size / (size + mu)
        density = stats.nbinom.logpmf(safe, size, p) - stats.nbinom.logsf(0, size, p)
    else:
        density = stats.poisson.logpmf(safe, mu) - stats.poisson.logsf(0, mu)
    return np.where(y > 0, np.log(probability) + density, np.log1p(-probability))


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


def close(actual, desired, rtol=5e-5):
    desired = np.asarray(desired)
    assert_allclose(actual, desired, rtol=rtol, atol=rtol * np.abs(desired).max())


def cluster_meat(scores, labels):
    sums = pd.DataFrame(scores).groupby(np.asarray(labels), sort=False).sum().to_numpy()
    return sums.T @ sums


def brute_force(objective, start):
    # The oracle likelihood may overflow at far-away trial points of the search.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = optimize.minimize(lambda t: -objective(t), start, method="BFGS",
                                 options={"gtol": 1e-9, "maxiter": 3000})
        best = optimize.minimize(lambda t: -objective(t), best.x, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 30000,
                                          "maxfev": 30000})
    return best.x, -best.fun


# ---- churdle -------------------------------------------------------------------------------


def test_churdle_exponential_is_probit_plus_lognormal_regression(data):
    result = oe.churdle(data=data, y="amount", x=X, select_x=Z)
    assert result.spec.options == {"model": "exponential", "ll": 0.0, "select_link": "probit"}
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "x2", "select:Intercept", "select:z1", "select:x1", "/lnsigma"]
    assert [c.equation for c in result.coefficients] == ["amount"] * 3 + ["select"] * 3 + [None]
    probit = sm.Probit((data.amount > 0).astype(float), sm.add_constant(data[Z])).fit(
        disp=0, tol=1e-12)
    assert_allclose(estimates(result)[3:6], probit.params.to_numpy(), rtol=1e-7)
    assert_allclose(errors(result)[3:6], probit.bse.to_numpy(), rtol=1e-6)
    # With ll = 0 the outcome part is the normal regression of ln(y): OLS, sigma^2 = SSR/n.
    selected = data[data.amount > 0]
    ols = sm.OLS(np.log(selected.amount), sm.add_constant(selected[X])).fit()
    sigma = np.sqrt(ols.ssr / len(selected))
    assert_allclose(estimates(result)[:3], ols.params.to_numpy(), rtol=1e-9)
    assert result.metrics["sigma"] == pytest.approx(sigma, rel=1e-9)
    assert_allclose(errors(result)[:3], ols.bse.to_numpy() * np.sqrt(ols.df_resid / len(selected)),
                    rtol=1e-8)
    assert errors(result)[6] == pytest.approx(np.sqrt(1 / (2 * len(selected))), rel=1e-8)
    log_likelihood = (probit.llf + stats.norm.logpdf(ols.resid, scale=sigma).sum()
                      - np.log(selected.amount).sum())
    assert result.metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=1e-10)
    assert result.extra["selection_log_likelihood"] == pytest.approx(probit.llf, rel=1e-10)
    record = result.extra["sigma"]
    critical = stats.norm.ppf(0.975)
    assert record["std_error"] == pytest.approx(sigma * errors(result)[6], rel=1e-9)
    assert record["ci_low"] == pytest.approx(
        np.exp(estimates(result)[6] - critical * errors(result)[6]), rel=1e-9)
    assert result.extra["n_bounded_observations"] == int((data.amount == 0).sum())
    assert result.metrics["aic"] == pytest.approx(-2 * log_likelihood + 14, rel=1e-10)
    # Chart sample: E[y] = Pr(y > 0) exp(x'b + sigma^2 / 2).
    theta = estimates(result)
    mean = stats.norm.cdf(design(data, Z) @ theta[3:6]) * np.exp(
        design(data, X) @ theta[:3] + sigma ** 2 / 2)
    for row in result.predictions[:20]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-9)


@pytest.mark.parametrize(("outcome", "kind", "limit"), [
    ("hours", "linear", 0.0), ("floor2", "linear", 2.0), ("floor2", "exponential", 2.0)])
def test_churdle_matches_brute_force_and_information(data, outcome, kind, limit):
    result = oe.churdle(data=data, y=outcome, x=X, select_x=Z, model=kind, ll=limit)
    y, x, z = data[outcome].to_numpy(), design(data, X), design(data, Z)

    def total(theta):
        return churdle_loglik(theta, y, x, z, limit, kind).sum()

    start = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    theta, value = brute_force(total, start)
    assert_allclose(estimates(result), theta, rtol=5e-4, atol=5e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(total(estimates(result)), rel=1e-11)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))))
    # Model test and pseudo R-squared: constant-only outcome equation, same selection part.
    _, null = brute_force(lambda t: churdle_loglik(t, y, x[:, :1], z, limit, kind).sum(),
                          np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-8)
    assert result.tests["model"]["df"] == 2
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (value - null), rel=1e-5)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - value / null, rel=1e-6)
    assert result.extra["ll"] == limit and result.extra["model"] == kind
    # Chart sample: (1 - P) ll + P E[y | y > ll].
    estimate = estimates(result)
    xb, sigma = x @ estimate[:3], np.exp(estimate[6])
    probability = stats.norm.cdf(z @ estimate[3:6])
    if kind == "linear":
        shift = (xb - limit) / sigma
        conditional = xb + sigma * stats.norm.pdf(shift) / stats.norm.cdf(shift)
    else:
        shift = (xb - np.log(limit)) / sigma
        conditional = np.exp(xb + sigma ** 2 / 2) * stats.norm.cdf(shift + sigma) \
            / stats.norm.cdf(shift)
    mean = (1 - probability) * limit + probability * conditional
    for row in result.predictions[:20]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-8)


def test_churdle_covariances_weights_and_logit_selection(data):
    args = {"y": "hours", "x": X, "select_x": Z, "model": "linear"}
    y, x, z = data.hours.to_numpy(), design(data, X), design(data, Z)
    base = oe.churdle(data=data, **args)
    theta = estimates(base)

    def per_observation(t):
        return churdle_loglik(t, y, x, z, 0.0, "linear")

    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(data)
    close(covariance(oe.churdle(data=data, covariance="opg", **args)),
          np.linalg.inv(scores.T @ scores))
    robust = oe.churdle(data=data, covariance="robust", **args)
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1))
    assert robust.tests["model"]["label"].startswith("Wald chi2")
    cluster = oe.churdle(data=data, cluster="firm", **args)
    groups = data.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, data.firm) @ bread * groups / (groups - 1))
    # The joint cluster covariance links the two equations (it is not block diagonal).
    assert np.abs(covariance(cluster)[:3, 3:6]).max() > 0
    assert np.abs(covariance(base)[:3, 3:6]).max() == 0
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "robust"):
        weighted = oe.churdle(data=data, weights="fw", weight_type="fweight", covariance=kind,
                              **args)
        expanded = oe.churdle(data=repeated, covariance=kind, **args)
        assert weighted.nobs == len(repeated)
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(expanded), rtol=1e-7, atol=1e-13)
        assert weighted.extra["n_bounded_observations"] == \
            expanded.extra["n_bounded_observations"]
    w = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.churdle(data=data, weights="aw", weight_type="aweight", **args)
    assert analytic.metrics["log_likelihood"] == pytest.approx(
        (w * per_observation(estimates(analytic))).sum(), rel=1e-10)
    gradient = (numeric_scores(per_observation, estimates(analytic)) * w[:, None]).sum(axis=0)
    assert np.abs(gradient).max() < 2e-4
    close(covariance(analytic), np.linalg.inv(-numeric_hessian(
        lambda t: (w * per_observation(t)).sum(), estimates(analytic))))
    sampling = oe.churdle(data=data, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-7)
    raw = data.aw.to_numpy()
    weighted_scores = numeric_scores(per_observation, estimates(sampling)) * raw[:, None]
    weighted_bread = np.linalg.inv(-numeric_hessian(
        lambda t: (raw * per_observation(t)).sum(), estimates(sampling)))
    close(covariance(sampling),
          weighted_bread @ (weighted_scores.T @ weighted_scores) @ weighted_bread * n / (n - 1))
    # Logit selection (an OpenEcon extension) equals statsmodels Logit.
    logit = oe.churdle(data=data, select_link="logit", **args)
    reference = sm.Logit((data.hours > 0).astype(float), sm.add_constant(data[Z])).fit(
        disp=0, tol=1e-12)
    assert_allclose(estimates(logit)[3:6], reference.params.to_numpy(), rtol=1e-7)
    assert_allclose(estimates(logit)[:3], theta[:3], rtol=1e-9)


def test_churdle_failure_contract_and_reporting(data):
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=X)
    assert excinfo.value.code == "invalid_spec" and "select_x" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=X, select_x=Z, select=Z)
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=X, select_x=Z, ll=-1.0)
    assert excinfo.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data[data.amount > 0], y="amount", x=X, select_x=Z)
    assert excinfo.value.code == "no_selection_variation" and "truncreg" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=X, select_x=Z, ll=1e9)
    assert excinfo.value.code == "no_selection_variation"
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=X, select_x=Z, model="tobit")
    assert excinfo.value.code == "invalid_spec"
    # A selection regressor that separates the bounded observations.
    separated = data.assign(perfect=(data.amount > 0).astype(float))
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=separated, y="amount", x=X, select_x=["perfect"])
    assert excinfo.value.code == "separation_detected"
    # select is Stata's name for the selection regressors; categorical terms; collinearity.
    alias = oe.churdle(data=data, y="amount", x=X, select=Z)
    base = oe.churdle(data=data, y="amount", x=X, select_x=Z)
    assert_allclose(estimates(alias), estimates(base), rtol=1e-12)
    wide = oe.churdle(data=data.assign(twice=2 * data.x1), y="amount",
                      x=["x1", "twice", "sector"], select_x=["z1", "sector"],
                      categorical=["sector"])
    assert wide.provenance["omitted_terms"] == ["twice"]
    assert [c.term for c in wide.coefficients] == [
        "Intercept", "x1", "sector[b]", "sector[c]", "select:Intercept", "select:z1",
        "select:sector[b]", "select:sector[c]", "/lnsigma"]
    # Values below the limit are bounded observations, with a recorded note.
    below = oe.churdle(data=data.assign(amount=np.where(data.amount == 0, -1.0, data.amount)),
                       y="amount", x=X, select_x=Z)
    assert any("below the lower limit" in warning for warning in below.warnings)
    assert_allclose(estimates(below), estimates(base), rtol=1e-12)
    holes = data.copy()
    holes.loc[[1, 2, 3], "z1"] = np.nan
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=holes, y="amount", x=X, select_x=Z)
    assert excinfo.value.code == "missing_values"
    assert oe.churdle(data=holes, y="amount", x=X, select_x=Z, missing="drop").nobs \
        == len(data) - 3
    assert type(base).model_validate_json(base.model_dump_json()) == base
    text = base.summary()
    assert "Cragg hurdle regression" in text and "[select]" in text and "/lnsigma" in text
    assert "sigma:" in text and "pseudo_r_squared" in text
    assert "lnsigma" in base.to_latex()
    refit = oe.fit(base.spec, data=data)
    assert_allclose(estimates(refit), estimates(base), rtol=1e-12)
    capability = oe.capabilities()["estimators"]["churdle"]
    assert capability["stata"] == ["churdle linear", "churdle exponential"]
    assert "Cragg" in oe.churdle.__doc__


# ---- hurdle (counts) -------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_hurdle_poisson_is_binary_model_plus_truncated_poisson(data, link):
    result = oe.hurdle(data=data, y="count", x=X, select_x=Z, zero_link=link)
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "x2", "select:Intercept", "select:z1", "select:x1"]
    positive = (data["count"] > 0).astype(float)
    exog = sm.add_constant(data[Z])
    if link == "logit":
        binary = sm.Logit(positive, exog).fit(disp=0, tol=1e-12)
    elif link == "probit":
        binary = sm.Probit(positive, exog).fit(disp=0, tol=1e-12)
    else:
        binary = sm.GLM(positive, exog, family=sm.families.Binomial(
            sm.families.links.CLogLog())).fit(tol=1e-12)
    assert_allclose(estimates(result)[3:], binary.params.to_numpy(), rtol=1e-6)
    sample = data[data["count"] > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = TruncatedLFPoisson(sample["count"], sm.add_constant(sample[X]),
                                    truncation=0).fit(method="newton", disp=0, tol=1e-12)
    assert_allclose(estimates(result)[:3], counts.params.to_numpy(), rtol=1e-8)
    assert_allclose(errors(result)[:3], counts.bse.to_numpy(), rtol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(binary.llf + counts.llf, rel=1e-9)
    assert result.metrics["n_zero_observations"] == int((data["count"] == 0).sum())
    y, x, z = data["count"].to_numpy(), design(data, X), design(data, Z)

    def total(theta):
        return hurdle_loglik(theta, y, x, z, link=link).sum()

    assert result.metrics["log_likelihood"] == pytest.approx(total(estimates(result)), rel=1e-11)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))))
    assert result.extra["zero_link"] == link and result.extra["dist"] == "poisson"
    # Chart sample: Pr(y > 0) mu / (1 - exp(-mu)).
    theta = estimates(result)
    mu = np.exp(x @ theta[:3])
    mean = link_cdf(z @ theta[3:], link) * mu / -np.expm1(-mu)
    for row in result.predictions[:20]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-9)


def test_hurdle_matches_statsmodels_hurdle_count_model(data):
    # statsmodels' zero model is a Poisson censored at one: the cloglog participation equation.
    result = oe.hurdle(data=data, y="count", x=X, zero_link="cloglog")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = HurdleCountModel(data["count"], sm.add_constant(data[X]), dist="poisson",
                                     zerodist="poisson").fit(disp=0, maxiter=500)
    params = reference.params.to_numpy()                   # (zero model, count model)
    assert [c.term for c in result.coefficients][3:] == ["select:Intercept", "select:x1",
                                                         "select:x2"]
    assert_allclose(estimates(result), np.r_[params[3:], params[:3]], rtol=2e-4, atol=2e-5)
    assert_allclose(errors(result), np.r_[reference.bse.to_numpy()[3:],
                                          reference.bse.to_numpy()[:3]], rtol=2e-3)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-8)


def test_hurdle_negative_binomial_brute_force_tests_and_exposure(data):
    args = {"y": "count_nb", "x": X, "select_x": Z, "dist": "nbinomial", "exposure": "expo"}
    result = oe.hurdle(data=data, **args)
    y, x, z = data.count_nb.to_numpy(), design(data, X), design(data, Z)
    offset = data.lnexpo.to_numpy()

    def per_observation(theta):
        return hurdle_loglik(theta, y, x, z, offset, negbin=True)

    def total(theta):
        return per_observation(theta).sum()

    theta, value = brute_force(total, np.array([0.3, 0, 0, 0.2, 0, 0, -0.3]))
    assert [c.term for c in result.coefficients][-1] == "/lnalpha"
    assert_allclose(estimates(result), theta, rtol=5e-4, atol=5e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    close(covariance(result), np.linalg.inv(-numeric_hessian(total, estimates(result))), 1e-4)
    assert result.metrics["alpha"] == pytest.approx(np.exp(estimates(result)[6]), rel=1e-12)
    assert result.extra["alpha"]["std_error"] == pytest.approx(
        result.metrics["alpha"] * errors(result)[6], rel=1e-10)
    # LR test of alpha = 0 against the Poisson hurdle; LR model test of the count slopes.
    poisson = oe.hurdle(data=data, **{**args, "dist": "poisson"})
    statistic = 2 * (value - poisson.metrics["log_likelihood"])
    assert result.tests["alpha"]["statistic"] == pytest.approx(statistic, rel=1e-7)
    assert result.tests["alpha"]["p_value"] == pytest.approx(0.5 * stats.chi2.sf(statistic, 1),
                                                             rel=1e-6, abs=1e-300)
    _, null = brute_force(lambda t: hurdle_loglik(t, y, x[:, :1], z, offset, negbin=True).sum(),
                          np.array([0.3, 0.2, 0, 0, -0.3]))
    assert result.tests["model"]["df"] == 2
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (value - null), rel=1e-5)
    assert result.metrics["pseudo_r_squared"] == pytest.approx(1 - value / null, rel=1e-6)
    # Sandwich covariances of the joint estimator.
    scores = numeric_scores(per_observation, estimates(result))
    bread = np.linalg.inv(-numeric_hessian(total, estimates(result)))
    n = len(data)
    close(covariance(oe.hurdle(data=data, covariance="opg", **args)),
          np.linalg.inv(scores.T @ scores), 1e-4)
    robust = oe.hurdle(data=data, covariance="robust", **args)
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1), 1e-4)
    assert "alpha" not in robust.tests and "alpha_test_note" in robust.extra
    cluster = oe.hurdle(data=data, cluster="firm", **args)
    groups = data.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, data.firm) @ bread * groups / (groups - 1), 1e-4)
    by_offset = oe.hurdle(data=data, **{**args, "exposure": None, "offset": "lnexpo"})
    assert_allclose(estimates(by_offset), estimates(result), rtol=1e-9)


def test_hurdle_weights_defaults_and_failure_contract(data):
    args = {"y": "count", "x": X, "select_x": Z}
    y, x, z = data["count"].to_numpy(), design(data, X), design(data, Z)
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust"):
        weighted = oe.hurdle(data=data, weights="fw", weight_type="fweight", covariance=kind,
                             **args)
        expanded = oe.hurdle(data=repeated, covariance=kind, **args)
        assert weighted.nobs == len(repeated)
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(expanded), rtol=1e-7, atol=1e-13)
        assert weighted.metrics["n_zero_observations"] == \
            expanded.metrics["n_zero_observations"]
    n = len(data)
    w = data.aw.to_numpy() * n / data.aw.sum()
    analytic = oe.hurdle(data=data, weights="aw", weight_type="aweight", **args)
    assert analytic.metrics["log_likelihood"] == pytest.approx(
        (w * hurdle_loglik(estimates(analytic), y, x, z)).sum(), rel=1e-10)
    gradient = (numeric_scores(lambda t: hurdle_loglik(t, y, x, z), estimates(analytic))
                * w[:, None]).sum(axis=0)
    assert np.abs(gradient).max() < 2e-4
    sampling = oe.hurdle(data=data, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-7)
    importance = oe.hurdle(data=data, weights="aw", weight_type="iweight", **args)
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-7)
    # select_x defaults to the regressors of the count equation.
    default = oe.hurdle(data=data, y="count", x=X)
    assert [c.term for c in default.coefficients][3:] == ["select:Intercept", "select:x1",
                                                          "select:x2"]
    explicit = oe.hurdle(data=data, y="count", x=X, select_x=X)
    assert_allclose(estimates(default), estimates(explicit), rtol=1e-12)
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data[data["count"] > 0], **args)
    assert excinfo.value.code == "no_selection_variation" and "tpoisson" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data.assign(count=0.0), **args)
    assert excinfo.value.code == "no_selection_variation"
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data.assign(count=data["count"] + 0.5), **args)
    assert excinfo.value.code == "invalid_count_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data.assign(count=np.minimum(data["count"], 1.0)), **args)
    assert excinfo.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data, dist="geometric", **args)
    assert excinfo.value.code == "invalid_spec"
    # Poisson positive counts are not overdispersed: the NB hurdle is at its boundary.
    rng = np.random.default_rng(8)
    tight = np.where(data["count"] > 0, 1 + rng.binomial(6, 0.4, size=len(data)), 0.0)
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=data.assign(count=tight), dist="nbinomial", **args)
    assert excinfo.value.code == "boundary_solution" and "dist='poisson'" in str(excinfo.value)
    separated = data.assign(perfect=(data["count"] > 0).astype(float))
    with pytest.raises(AnalysisError) as excinfo:
        oe.hurdle(data=separated, y="count", x=X, select_x=["perfect"])
    assert excinfo.value.code == "separation_detected"
    wide = oe.hurdle(data=data.assign(twice=2 * data.x1), y="count", x=["x1", "twice"],
                     select_x=["z1", "sector"], categorical=["sector"])
    assert wide.provenance["omitted_terms"] == ["twice"]
    result = oe.hurdle(data=data, dist="nbinomial", y="count_nb", x=X, select_x=Z)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Hurdle count regression" in text and "[select]" in text and "/lnalpha" in text
    assert "LR test of alpha = 0 against the Poisson hurdle model" in text
    assert "select" in result.to_latex()
    refit = oe.fit(result.spec, data=data)
    assert_allclose(estimates(refit), estimates(result), rtol=1e-12)
    assert oe.capabilities()["estimators"]["hurdle"]["family"] == "count"
    assert "Mullahy" in oe.hurdle.__doc__


def test_two_part_models_constant_only_no_constant_and_degenerate_outcomes(data):
    # x = [] is the constant-only outcome equation: nothing to test, pseudo R-squared 0.
    constant = oe.churdle(data=data, y="amount", x=[], select_x=Z)
    selected = data[data.amount > 0]
    assert estimates(constant)[0] == pytest.approx(np.log(selected.amount).mean(), rel=1e-10)
    assert constant.metrics["sigma"] == pytest.approx(np.log(selected.amount).std(ddof=0),
                                                      rel=1e-9)
    assert constant.tests["model"]["df"] == 0
    assert constant.metrics["pseudo_r_squared"] == pytest.approx(0.0, abs=1e-12)
    counts = oe.hurdle(data=data, y="count", x=[])
    assert [c.term for c in counts.coefficients] == ["Intercept", "select:Intercept"]
    positive = data.loc[data["count"] > 0, "count"]
    mu = np.exp(estimates(counts)[0])
    assert mu / -np.expm1(-mu) == pytest.approx(positive.mean(), rel=1e-9)
    assert special.expit(estimates(counts)[1]) == pytest.approx((data["count"] > 0).mean(),
                                                               rel=1e-9)
    # Without a constant in the outcome equation: Wald test, no pseudo R-squared.
    no_constant = oe.churdle(data=data, y="hours", x=X, select_x=Z, model="linear",
                             intercept=False)
    assert [c.term for c in no_constant.coefficients][:2] == X
    assert no_constant.metrics["pseudo_r_squared"] is None
    assert no_constant.tests["model"]["label"].startswith("Wald chi2")
    y, x, z = data.hours.to_numpy(), design(data, X)[:, 1:], design(data, Z)
    gradient = numeric_scores(lambda t: churdle_loglik(t, y, x, z, 0.0, "linear"),
                              estimates(no_constant)).sum(axis=0)
    assert np.abs(gradient).max() < 2e-4
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=data, y="amount", x=[], select_x=Z, intercept=False)
    assert excinfo.value.code == "empty_design"
    # An outcome equation that fits exactly, or an outcome that does not vary above the limit.
    exact = data.assign(amount=np.where(data.amount > 0, np.exp(1.0 + 0.5 * data.x1), 0.0))
    with pytest.raises(AnalysisError) as excinfo:
        oe.churdle(data=exact, y="amount", x=["x1"], select_x=Z)
    assert excinfo.value.code == "perfect_fit"
    flat = data.assign(amount=np.where(data.amount > 0, 3.0, 0.0))
    for kind in ("exponential", "linear"):
        with pytest.raises(AnalysisError) as excinfo:
            oe.churdle(data=flat, y="amount", x=["x1"], select_x=Z, model=kind)
        assert excinfo.value.code == "constant_outcome"
