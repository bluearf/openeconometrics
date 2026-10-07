"""zip and zinb against independent oracles.

Oracles: scipy.stats pmfs with brute-force scipy.optimize maximization of an
independently written zero-inflated likelihood, numerical scores and Hessians
of that likelihood for every covariance estimator, statsmodels
ZeroInflatedPoisson / ZeroInflatedNegativeBinomialP, the Vuong statistic coded
from its definition, duplicated rows for frequency weights.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize, special, stats
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson,
)

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.models import ResultBundle

X = ["x1", "x2"]
Z = ["z1"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(20261)
    n = 1500
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n), "z1": rng.normal(size=n),
        "firm": np.repeat(np.arange(75), 20), "state": rng.integers(0, 12, size=n),
        "fw": rng.integers(1, 4, size=n).astype(float), "aw": rng.uniform(0.5, 2.5, size=n),
        "expo": rng.uniform(0.5, 2.0, size=n),
        "sector": pd.Categorical(rng.choice(["a", "b", "c"], size=n), categories=["a", "b", "c"]),
    })
    mu = frame.expo * np.exp(0.6 + 0.4 * frame.x1 - 0.3 * frame.x2)
    excess = rng.uniform(size=n) < special.expit(-0.6 + 0.8 * frame.z1)
    frame["y"] = np.where(excess, 0, rng.poisson(mu)).astype(float)
    frame["ynb"] = np.where(excess, 0, rng.negative_binomial(2.0, 2.0 / (2.0 + mu))).astype(float)
    frame["lnexpo"] = np.log(frame.expo)
    return frame


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def matrices(frame, x=X, z=Z):
    return (sm.add_constant(frame[list(x)]).to_numpy(), sm.add_constant(frame[list(z)]).to_numpy()
            if z else np.ones((len(frame), 1)))


def zi_loglik(theta, y, x, z, offset=0.0, link="logit", negbin=False):
    """Per-observation zero-inflated log likelihood written with scipy.stats pmfs."""
    k, q = x.shape[1], z.shape[1]
    mu = np.exp(x @ theta[:k] + offset)
    index = z @ theta[k:k + q]
    inflate = special.expit(index) if link == "logit" else stats.norm.cdf(index)
    if negbin:
        size = np.exp(-theta[k + q])
        log_f = stats.nbinom.logpmf(y, size, size / (size + mu))
    else:
        log_f = stats.poisson.logpmf(y, mu)
    return np.where(y == 0, np.log(inflate + (1 - inflate) * np.exp(log_f)),
                    np.log1p(-inflate) + log_f)


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
    """Matrix comparison whose absolute tolerance is relative to the largest entry."""
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


# ---- zip -----------------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_zip_matches_brute_force_and_information(data, link):
    result = oe.zip(data=data, y="y", x=X, inflate=Z, inflate_link=link, exposure="expo")
    y, (x, z), offset = data.y.to_numpy(), matrices(data), data.lnexpo.to_numpy()

    def per_observation(theta):
        return zi_loglik(theta, y, x, z, offset, link)

    def total(theta):
        return per_observation(theta).sum()

    theta, value = brute_force(total, np.array([0.3, 0.0, 0.0, -0.5, 0.0]))
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "x2", "inflate:Intercept", "inflate:z1"]
    assert [c.equation for c in result.coefficients] == ["y"] * 3 + ["inflate"] * 2
    assert_allclose(estimates(result), theta, rtol=2e-4, atol=2e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    assert result.metrics["log_likelihood"] == pytest.approx(total(estimates(result)), rel=1e-11)
    # Observed information from the numerical Hessian of the independent likelihood.
    hessian = numeric_hessian(total, estimates(result))
    assert_allclose(covariance(result), np.linalg.inv(-hessian), rtol=2e-5, atol=1e-9)
    assert result.metrics["aic"] == pytest.approx(-2 * value + 2 * 5, rel=1e-9)
    assert result.metrics["bic"] == pytest.approx(-2 * value + 5 * np.log(len(data)), rel=1e-9)
    assert result.metrics["n_zero_observations"] == int((y == 0).sum())
    assert result.inference["distribution"] == "normal"
    assert result.provenance["stata_parity_validated"] is False
    assert result.extra["inflate_link"] == link


def test_zip_matches_statsmodels(data):
    result = oe.zip(data=data, y="y", x=X, inflate=Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = ZeroInflatedPoisson(
            data.y, sm.add_constant(data[X]), exog_infl=sm.add_constant(data[Z]),
            inflation="logit").fit(method="newton", maxiter=200, disp=0, tol=1e-12)
    # statsmodels orders the parameters (inflation, count).
    assert_allclose(estimates(result), np.r_[reference.params.to_numpy()[2:],
                                             reference.params.to_numpy()[:2]], rtol=1e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10)
    assert result.metrics["aic"] == pytest.approx(reference.aic, rel=1e-10)


def test_zip_model_test_null_model_and_vuong(data):
    result = oe.zip(data=data, y="y", x=X, inflate=Z)
    y, (x, z) = data.y.to_numpy(), matrices(data)
    # Stata's comparison model: constant-only count equation, full inflation equation.
    ones = np.ones((len(data), 1))
    _, null = brute_force(lambda t: zi_loglik(t, y, ones, z).sum(), np.array([0.5, -0.5, 0.0]))
    assert result.extra["null_log_likelihood"] == pytest.approx(null, rel=1e-9)
    test = result.tests["model"]
    assert test["distribution"] == "chi2" and test["df"] == 2
    assert test["statistic"] == pytest.approx(2 * (result.metrics["log_likelihood"] - null),
                                              rel=1e-7)
    assert test["p_value"] == pytest.approx(stats.chi2.sf(test["statistic"], 2), rel=1e-8,
                                            abs=1e-300)
    # Vuong (1989) from the definition, against the Poisson maximum likelihood fit.
    poisson = sm.Poisson(data.y, sm.add_constant(data[X])).fit(disp=0, tol=1e-12)
    m = zi_loglik(estimates(result), y, x, z) - stats.poisson.logpmf(
        y, np.exp(x @ poisson.params.to_numpy()))
    n = len(m)
    v = np.sqrt(n) * m.mean() / m.std(ddof=1)
    vuong = result.tests["vuong"]
    assert vuong["statistic"] == pytest.approx(v, rel=1e-7)
    assert vuong["p_value"] == pytest.approx(stats.norm.sf(v), rel=1e-6)
    assert vuong["distribution"] == "normal"
    record = result.extra["vuong"]
    extra_parameters = z.shape[1]
    assert record["z_aic"] == pytest.approx(
        np.sqrt(n) * (m.mean() - extra_parameters / n) / m.std(ddof=1), rel=1e-7)
    assert record["z_bic"] == pytest.approx(
        np.sqrt(n) * (m.mean() - extra_parameters * np.log(n) / (2 * n)) / m.std(ddof=1),
        rel=1e-7)
    assert result.extra["poisson_log_likelihood"] == pytest.approx(poisson.llf, rel=1e-10)


def test_zip_covariances_against_numerical_scores(data):
    y, (x, z) = data.y.to_numpy(), matrices(data)
    base = oe.zip(data=data, y="y", x=X, inflate=Z)
    theta = estimates(base)

    def per_observation(t):
        return zi_loglik(t, y, x, z)

    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(data)
    opg = oe.zip(data=data, y="y", x=X, inflate=Z, covariance="opg")
    close(covariance(opg), np.linalg.inv(scores.T @ scores), 2e-5)
    robust = oe.zip(data=data, y="y", x=X, inflate=Z, covariance="robust")
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1), 2e-5)
    assert robust.inference["correction"].startswith("Huber-White sandwich")
    assert robust.tests["model"]["label"].startswith("Wald chi2")
    wald = theta[1:3] @ np.linalg.solve(covariance(robust)[1:3, 1:3], theta[1:3])
    assert robust.tests["model"]["statistic"] == pytest.approx(wald, rel=1e-9)
    assert "vuong" not in robust.tests and "vuong_note" in robust.extra
    cluster = oe.zip(data=data, y="y", x=X, inflate=Z, cluster="firm")
    groups = data.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, data.firm) @ bread * groups / (groups - 1), 2e-5)
    assert cluster.spec.covariance == "cluster" and cluster.inference["cluster_count"] == groups
    two_way = oe.zip(data=data, y="y", x=X, inflate=Z, cluster=["firm", "state"])
    pair = data.firm.astype(str) + "/" + data.state.astype(str)
    meat = (cluster_meat(scores, data.firm) + cluster_meat(scores, data.state)
            - cluster_meat(scores, pair))
    smallest = min(groups, data.state.nunique())
    close(covariance(two_way), bread @ meat @ bread * smallest / (smallest - 1), 2e-5)
    assert_allclose(estimates(two_way), theta, rtol=1e-12)


def test_zip_weights(data):
    args = {"y": "y", "x": X, "inflate": Z}
    y, (x, z) = data.y.to_numpy(), matrices(data)
    # Frequency weights replicate rows, for every covariance estimator.
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "opg", "robust"):
        weighted = oe.zip(data=data, weights="fw", weight_type="fweight", covariance=kind, **args)
        expanded = oe.zip(data=repeated, covariance=kind, **args)
        assert weighted.nobs == len(repeated) == expanded.nobs
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance(weighted), covariance(expanded), rtol=1e-7, atol=1e-12)
        assert weighted.metrics["log_likelihood"] == pytest.approx(
            expanded.metrics["log_likelihood"], rel=1e-10)
        assert weighted.metrics["n_zero_observations"] == \
            expanded.metrics["n_zero_observations"]
    weighted = oe.zip(data=data, weights="fw", weight_type="fweight", **args)
    expanded = oe.zip(data=repeated, **args)
    for name in ("model", "vuong"):
        assert weighted.tests[name]["statistic"] == pytest.approx(
            expanded.tests[name]["statistic"], rel=1e-7)
    cluster_w = oe.zip(data=data, weights="fw", weight_type="fweight", cluster="firm", **args)
    cluster_e = oe.zip(data=repeated, cluster="firm", **args)
    assert_allclose(covariance(cluster_w), covariance(cluster_e), rtol=1e-7)
    # Analytic weights are rescaled to sum to N: brute force of the weighted likelihood.
    w = data.aw.to_numpy() * len(data) / data.aw.sum()
    analytic = oe.zip(data=data, weights="aw", weight_type="aweight", **args)
    theta, value = brute_force(lambda t: (w * zi_loglik(t, y, x, z)).sum(),
                               np.array([0.3, 0.0, 0.0, -0.5, 0.0]))
    assert_allclose(estimates(analytic), theta, rtol=2e-4, atol=2e-5)
    assert analytic.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    hessian = numeric_hessian(lambda t: (w * zi_loglik(t, y, x, z)).sum(), estimates(analytic))
    assert_allclose(covariance(analytic), np.linalg.inv(-hessian), rtol=2e-5, atol=1e-9)
    scaled = oe.zip(data=data.assign(aw=data.aw * 1000), weights="aw", weight_type="aweight",
                    **args)
    assert_allclose(estimates(scaled), estimates(analytic), rtol=1e-9)
    assert_allclose(errors(scaled), errors(analytic), rtol=1e-8)
    assert "vuong" not in analytic.tests and "vuong_note" in analytic.extra
    # Importance weights enter as given; sampling weights give the weighted sandwich.
    importance = oe.zip(data=data, weights="aw", weight_type="iweight", **args)
    assert_allclose(estimates(importance), estimates(analytic), rtol=1e-8)
    raw = data.aw.to_numpy()
    assert importance.metrics["log_likelihood"] == pytest.approx(
        (raw * zi_loglik(estimates(importance), y, x, z)).sum(), rel=1e-10)
    sampling = oe.zip(data=data, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    scores = numeric_scores(lambda t: zi_loglik(t, y, x, z), estimates(sampling)) * raw[:, None]
    bread = np.linalg.inv(-numeric_hessian(lambda t: (raw * zi_loglik(t, y, x, z)).sum(),
                                           estimates(sampling)))
    n = len(data)
    close(covariance(sampling), bread @ (scores.T @ scores) @ bread * n / (n - 1), 5e-5)
    huge = oe.zip(data=data.assign(aw=data.aw * 1e6), weights="aw", weight_type="pweight", **args)
    assert_allclose(estimates(huge), estimates(sampling), rtol=1e-7)
    assert_allclose(errors(huge), errors(sampling), rtol=1e-6)
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, weights="aw", weight_type="pweight", covariance="nonrobust", **args)
    assert excinfo.value.code == "unsupported_covariance"


def test_zip_constant_inflation_offset_and_categoricals(data):
    # inflate=[] is Stata's inflate(_cons): one inflation probability.
    constant = oe.zip(data=data, y="y", x=X, inflate=[])
    y, (x, _) = data.y.to_numpy(), matrices(data)
    ones = np.ones((len(data), 1))
    theta, value = brute_force(lambda t: zi_loglik(t, y, x, ones).sum(),
                               np.array([0.3, 0.0, 0.0, -0.5]))
    assert [c.term for c in constant.coefficients][-1] == "inflate:Intercept"
    assert_allclose(estimates(constant), theta, rtol=2e-4, atol=2e-5)
    assert constant.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    assert constant.extra["mean_inflation_probability"] == pytest.approx(
        special.expit(estimates(constant)[-1]), rel=1e-10)
    # exposure equals an offset of ln(exposure); both cannot be given.
    by_exposure = oe.zip(data=data, y="y", x=X, inflate=Z, exposure="expo")
    by_offset = oe.zip(data=data, y="y", x=X, inflate=Z, offset="lnexpo")
    assert_allclose(estimates(by_exposure), estimates(by_offset), rtol=1e-10)
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, y="y", x=X, inflate=Z, exposure="expo", offset="lnexpo")
    assert excinfo.value.code == "invalid_spec"
    # Categorical regressors in both equations; no constant in the count equation.
    result = oe.zip(data=data, y="y", x=["x1", "sector"], inflate=["z1", "sector"],
                    categorical=["sector"])
    terms = [c.term for c in result.coefficients]
    assert terms == ["Intercept", "x1", "sector[b]", "sector[c]", "inflate:Intercept",
                     "inflate:z1", "inflate:sector[b]", "inflate:sector[c]"]
    dummies = pd.get_dummies(data.sector, drop_first=True).to_numpy(dtype=float)
    xs = np.column_stack([np.ones(len(data)), data.x1, dummies])
    zs = np.column_stack([np.ones(len(data)), data.z1, dummies])
    assert result.metrics["log_likelihood"] == pytest.approx(
        zi_loglik(estimates(result), y, xs, zs).sum(), rel=1e-11)
    gradient = numeric_scores(lambda t: zi_loglik(t, y, xs, zs), estimates(result)).sum(axis=0)
    assert np.abs(gradient).max() < 1e-4
    no_constant = oe.zip(data=data, y="y", x=X, inflate=Z, intercept=False)
    assert no_constant.tests["model"]["label"].startswith("Wald chi2")
    assert no_constant.extra["null_log_likelihood"] is None
    gradient = numeric_scores(lambda t: zi_loglik(t, y, x[:, 1:], matrices(data)[1]),
                              estimates(no_constant)).sum(axis=0)
    assert np.abs(gradient).max() < 1e-4


def test_zip_collinearity_missing_values_and_fitted(data):
    frame = data.assign(x1_copy=data.x1, z1_twice=2 * data.z1)
    result = oe.zip(data=frame, y="y", x=["x1", "x2", "x1_copy"], inflate=["z1", "z1_twice"])
    base = oe.zip(data=data, y="y", x=X, inflate=Z)
    assert result.provenance["omitted_terms"] == ["x1_copy", "inflate:z1_twice"]
    assert any("Omitted because of collinearity" in warning for warning in result.warnings)
    assert_allclose(estimates(result), estimates(base), rtol=1e-9)
    holes = data.copy()
    holes.loc[[3, 40], "z1"] = np.nan
    holes.loc[7, "y"] = np.nan
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=holes, y="y", x=X, inflate=Z)
    assert excinfo.value.code == "missing_values"
    dropped = oe.zip(data=holes, y="y", x=X, inflate=Z, missing="drop")
    complete = oe.zip(data=holes.dropna(subset=["y", "z1"]), y="y", x=X, inflate=Z)
    assert dropped.nobs == len(data) - 3 and dropped.dropped_rows == 3
    assert_allclose(estimates(dropped), estimates(complete), rtol=1e-10)
    # The chart sample holds E[y] = (1 - F) mu.
    theta = estimates(base)
    x, z = matrices(data)
    mean = (1 - special.expit(z @ theta[3:])) * np.exp(x @ theta[:3])
    for row in base.predictions[:20]:
        assert row["fitted"] == pytest.approx(mean[row["row"]], rel=1e-9)
        assert row["observed"] == data.y.iloc[row["row"]]


def test_zip_failure_contract(data):
    args = {"x": X, "inflate": Z}
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data.assign(y=data.y + 1), y="y", **args)
    assert excinfo.value.code == "no_zero_outcomes" and "oe.poisson" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data.assign(y=0.0), y="y", **args)
    assert excinfo.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data.assign(y=data.y - 1), y="y", **args)
    assert excinfo.value.code == "invalid_count_outcome"
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, y="y", x="x1", inflate=Z)
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, y="y", x=X, inflate=Z, inflate_link="cloglog")
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, y="y", x=X, inflate=Z, covariance="HC3")
    assert excinfo.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=data, y="y", x=X, inflate=["absent"])
    assert excinfo.value.code == "missing_columns"
    # Fewer zeros than Poisson predicts: the inflation probability is at its boundary.
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x1": rng.normal(size=4000), "z1": rng.normal(size=4000)})
    counts = rng.binomial(8, special.expit(-0.6 + 0.3 * frame.x1))       # underdispersed
    frame["y"] = counts.astype(float)
    assert (frame.y == 0).any()
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=frame, y="y", x=["x1"], inflate=["z1"])
    assert excinfo.value.code == "boundary_solution" and "oe.poisson" in str(excinfo.value)
    # A group without any zero: its inflation coefficient diverges.
    frame = data.assign(group=(np.arange(len(data)) % 5 == 0).astype(float))
    frame.loc[(frame.group == 1) & (frame.y == 0), "y"] = 3.0
    frame.loc[frame.group == 1, "y"] += 6.0
    with pytest.raises(AnalysisError) as excinfo:
        oe.zip(data=frame, y="y", x=X, inflate=["group"])
    assert excinfo.value.code in {"separation_detected", "boundary_solution"}
    # A non-integer outcome is fitted with a recorded note (Stata's rule for count models).
    noted = oe.zip(data=data.assign(y=np.where(data.y == 3, 2.5, data.y)), y="y", **args)
    assert any("non-integer" in warning for warning in noted.warnings)


def test_zip_reporting_json_and_public_access(data):
    result = oe.zip(data=data, y="y", x=X, inflate=Z)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert isinstance(result, ResultBundle)
    text = result.summary()
    assert "Zero-inflated Poisson regression" in text and "[inflate]" in text
    assert "inflate:z1" in text and "Vuong test" in text and "LR chi2" in text
    latex = result.to_latex()
    assert "inflate" in latex and "x1" in latex
    spec = result.spec
    assert spec.estimator == "zip" and spec.columns == {"inflate": ["z1"]}
    refit = oe.fit(spec, data=data)
    assert_allclose(estimates(refit), estimates(result), rtol=1e-12)
    capability = oe.capabilities()["estimators"]["zip"]
    assert capability["family"] == "count" and capability["function"] == "zip"
    assert "Zero-inflated Poisson regression" in oe.zip.__doc__ and "Stata" in oe.zip.__doc__


# ---- zinb ----------------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_zinb_matches_brute_force_and_information(data, link):
    result = oe.zinb(data=data, y="ynb", x=X, inflate=Z, inflate_link=link, offset="lnexpo")
    y, (x, z), offset = data.ynb.to_numpy(), matrices(data), data.lnexpo.to_numpy()

    def total(theta):
        return zi_loglik(theta, y, x, z, offset, link, negbin=True).sum()

    theta, value = brute_force(total, np.array([0.3, 0.0, 0.0, -0.5, 0.0, -0.5]))
    assert [c.term for c in result.coefficients] == [
        "Intercept", "x1", "x2", "inflate:Intercept", "inflate:z1", "/lnalpha"]
    assert [c.equation for c in result.coefficients] == ["ynb"] * 3 + ["inflate"] * 2 + [None]
    assert_allclose(estimates(result), theta, rtol=5e-4, atol=5e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(value, rel=1e-9)
    hessian = numeric_hessian(total, estimates(result))
    assert_allclose(covariance(result), np.linalg.inv(-hessian), rtol=5e-5, atol=1e-9)
    alpha = np.exp(estimates(result)[-1])
    assert result.metrics["alpha"] == pytest.approx(alpha, rel=1e-12)
    record = result.extra["alpha"]
    se = errors(result)[-1]
    critical = stats.norm.ppf(0.975)
    assert record["std_error"] == pytest.approx(alpha * se, rel=1e-10)
    assert record["ci_low"] == pytest.approx(np.exp(estimates(result)[-1] - critical * se),
                                             rel=1e-9)
    assert record["ci_high"] == pytest.approx(np.exp(estimates(result)[-1] + critical * se),
                                              rel=1e-9)
    assert result.metrics["aic"] == pytest.approx(-2 * value + 12, rel=1e-9)


def test_zinb_matches_statsmodels_and_reports_tests(data):
    result = oe.zinb(data=data, y="ynb", x=X, inflate=Z)
    y, (x, z) = data.ynb.to_numpy(), matrices(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = ZeroInflatedNegativeBinomialP(
            data.ynb, sm.add_constant(data[X]), exog_infl=sm.add_constant(data[Z]),
            inflation="logit", p=2).fit(method="bfgs", maxiter=3000, disp=0, gtol=1e-10)
    params = reference.params.to_numpy()          # (inflation, count, alpha)
    assert_allclose(estimates(result)[:5], np.r_[params[2:5], params[:2]], rtol=2e-4, atol=2e-5)
    assert result.metrics["alpha"] == pytest.approx(params[5], rel=2e-4)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-9)
    # LR test of alpha = 0 against zip: chibar2(01).
    zip_fit = oe.zip(data=data, y="ynb", x=X, inflate=Z)
    statistic = 2 * (result.metrics["log_likelihood"] - zip_fit.metrics["log_likelihood"])
    alpha_test = result.tests["alpha"]
    assert alpha_test["statistic"] == pytest.approx(statistic, rel=1e-9)
    assert alpha_test["p_value"] == pytest.approx(0.5 * stats.chi2.sf(statistic, 1), rel=1e-7,
                                                  abs=1e-300)
    assert alpha_test["distribution"] == "chibar2"
    assert result.extra["zip_log_likelihood"] == pytest.approx(
        zip_fit.metrics["log_likelihood"], rel=1e-10)
    # Model test: constant-only count equation, full inflation equation, free alpha.
    ones = np.ones((len(data), 1))
    _, null = brute_force(lambda t: zi_loglik(t, y, ones, z, negbin=True).sum(),
                          np.array([0.5, -0.5, 0.0, -0.5]))
    assert result.tests["model"]["df"] == 2
    assert result.tests["model"]["statistic"] == pytest.approx(
        2 * (result.metrics["log_likelihood"] - null), rel=1e-6)
    # Vuong against the negative binomial model without inflation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sm.NegativeBinomial(data.ynb, sm.add_constant(data[X])).fit(
            disp=0, maxiter=500, method="newton", tol=1e-12)
    size = 1 / plain.params.to_numpy()[-1]
    mu = np.exp(x @ plain.params.to_numpy()[:3])
    m = zi_loglik(estimates(result), y, x, z, negbin=True) - stats.nbinom.logpmf(
        y, size, size / (size + mu))
    v = np.sqrt(len(m)) * m.mean() / m.std(ddof=1)
    assert result.tests["vuong"]["statistic"] == pytest.approx(v, rel=1e-5)
    assert result.extra["nbreg_log_likelihood"] == pytest.approx(plain.llf, rel=1e-9)
    assert "zinb vs. standard negative binomial" in result.tests["vuong"]["label"]


def test_zinb_covariances_and_weights(data):
    args = {"y": "ynb", "x": X, "inflate": Z}
    y, (x, z) = data.ynb.to_numpy(), matrices(data)
    base = oe.zinb(data=data, **args)
    theta = estimates(base)

    def per_observation(t):
        return zi_loglik(t, y, x, z, negbin=True)

    scores = numeric_scores(per_observation, theta)
    bread = np.linalg.inv(-numeric_hessian(lambda t: per_observation(t).sum(), theta))
    n = len(data)
    opg = oe.zinb(data=data, covariance="opg", **args)
    close(covariance(opg), np.linalg.inv(scores.T @ scores), 5e-5)
    robust = oe.zinb(data=data, covariance="robust", **args)
    close(covariance(robust), bread @ (scores.T @ scores) @ bread * n / (n - 1), 5e-5)
    assert "alpha" not in robust.tests and "alpha_test_note" in robust.extra
    cluster = oe.zinb(data=data, cluster="firm", **args)
    groups = data.firm.nunique()
    close(covariance(cluster),
          bread @ cluster_meat(scores, data.firm) @ bread * groups / (groups - 1), 5e-5)
    repeated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    weighted = oe.zinb(data=data, weights="fw", weight_type="fweight", **args)
    expanded = oe.zinb(data=repeated, **args)
    assert weighted.nobs == len(repeated)
    assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-7, atol=1e-9)
    assert_allclose(errors(weighted), errors(expanded), rtol=1e-6)
    for name in ("model", "alpha", "vuong"):
        assert weighted.tests[name]["statistic"] == pytest.approx(
            expanded.tests[name]["statistic"], rel=1e-6)
    w = data.aw.to_numpy() * len(data) / data.aw.sum()
    analytic = oe.zinb(data=data, weights="aw", weight_type="aweight", **args)
    assert analytic.metrics["log_likelihood"] == pytest.approx(
        (w * zi_loglik(estimates(analytic), y, x, z, negbin=True)).sum(), rel=1e-10)
    gradient = (numeric_scores(per_observation, estimates(analytic)) * w[:, None]).sum(axis=0)
    assert np.abs(gradient).max() < 2e-4
    sampling = oe.zinb(data=data, weights="aw", weight_type="pweight", **args)
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(analytic), rtol=1e-7)


def test_zinb_boundaries_json_and_rendering(data):
    # Counts that are underdispersed beyond their excess zeros: alpha is estimated at zero.
    rng = np.random.default_rng(11)
    under = np.where(rng.uniform(size=len(data)) < special.expit(-0.6 + 0.8 * data.z1), 0,
                     rng.binomial(8, special.expit(-0.4 + 0.3 * data.x1)))
    with pytest.raises(AnalysisError) as excinfo:
        oe.zinb(data=data.assign(under=under.astype(float)), y="under", x=X, inflate=Z)
    assert excinfo.value.code == "boundary_solution" and "oe.zip" in str(excinfo.value)
    with pytest.raises(AnalysisError) as excinfo:
        oe.zinb(data=data.assign(ynb=data.ynb + 1), y="ynb", x=X, inflate=Z)
    assert excinfo.value.code == "no_zero_outcomes" and "oe.nbreg" in str(excinfo.value)
    result = oe.zinb(data=data, y="ynb", x=X, inflate=Z)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    text = result.summary()
    assert "Zero-inflated negative binomial regression" in text and "/lnalpha" in text
    assert "LR test of alpha = 0" in text and "alpha:" in text
    assert "lnalpha" in result.to_latex()
    assert oe.capabilities()["estimators"]["zinb"]["stata"] == ["zinb"]
