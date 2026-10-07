"""Independent oracles for oe.streg: brute-force SciPy maximization of likelihoods
written with scipy.stats densities and survivor functions, numerically
differentiated Hessians and per-observation scores (statsmodels numdiff)."""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival.parametric import ParametricObjective, SurvivalData
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle


def make_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "s": rng.integers(0, 2, n), "cl": rng.integers(0, 35, n),
                          "w": rng.integers(1, 4, n), "pw": rng.uniform(0.5, 2, n)})
    t = rng.weibull(1.4, n) * np.exp(-0.4 * frame.x + 0.2 * frame.z)
    c = rng.exponential(2.0, n)
    frame["t"] = np.minimum(t, c) + 1e-3
    frame["d"] = (t <= c).astype(float)
    frame["t0"] = np.where(rng.random(n) < 0.3, frame.t * rng.uniform(0.1, 0.9, n), 0.0)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def logdensity(dist, metric, t, mu, a):
    """(log f, log S) of each model, written with scipy.stats where it has the family."""
    if dist in ("exponential", "weibull"):
        p = 1.0 if a is None else np.exp(a)
        lam = np.exp(mu) if metric == "ph" else np.exp(-p * mu)
        log_s = -lam * t ** p
        return np.log(p * lam) + (p - 1) * np.log(t) + log_s, log_s
    if dist == "gompertz":
        log_s = -np.exp(mu) * np.expm1(a * t) / a
        return mu + a * t + log_s, log_s
    if dist == "lognormal":
        law = stats.lognorm(s=np.exp(a), scale=np.exp(mu))
        return law.logpdf(t), law.logsf(t)
    law = stats.fisk(c=np.exp(-a), scale=np.exp(mu))
    return law.logpdf(t), law.logsf(t)


def oracle_loglik(theta, dist, metric, frame, anc_cols=(), weights=None):
    x = np.column_stack([np.ones(len(frame)), frame.x])
    k = 2
    mu = x @ theta[:k]
    a = None
    if dist != "exponential":
        z = np.column_stack([np.ones(len(frame)), *(frame[c] for c in anc_cols)])
        a = z @ theta[k:]
    t, t0, d = frame.t.to_numpy(), frame.t0.to_numpy(), frame.d.to_numpy()
    log_f, log_s = logdensity(dist, metric, t, mu, a)
    contribution = d * log_f + (1 - d) * log_s
    late = t0 > 0
    if late.any():
        _, log_s0 = logdensity(dist, metric, t0[late], mu[late],
                               None if a is None else a[late])
        contribution[late] -= log_s0
    return contribution if weights is None else contribution * weights


CASES = [("exponential", "ph"), ("exponential", "aft"), ("weibull", "ph"), ("weibull", "aft"),
         ("gompertz", "ph"), ("lognormal", "aft"), ("loglogistic", "aft")]


@pytest.mark.parametrize("dist,metric", CASES)
def test_maximum_likelihood_matches_brute_force(data, dist, metric):
    result = oe.streg(data=data, time="t", failure="d", entry="t0", x=["x"],
                      distribution=dist, metric=metric)
    b = np.array([c.estimate for c in result.coefficients])
    se = np.array([c.std_error for c in result.coefficients])
    objective = lambda v: -oracle_loglik(v, dist, metric, data).sum()  # noqa: E731
    fitted = minimize(objective, b + 0.05, method="BFGS", options={"gtol": 1e-9})
    assert_allclose(b, fitted.x, atol=2e-5)
    # Stata reports the log likelihood of ln t: the time-scale value + sum d ln t
    # (verify-pass correction; checked against [ST] streg's kva example).
    log_time = (data.d * np.log(data.t)).sum()
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun + log_time, rel=1e-9)
    hessian = approx_hess(b, lambda v: oracle_loglik(v, dist, metric, data).sum())
    assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=1e-5)
    assert result.metrics["n_failures"] == data.d.sum()
    assert result.metrics["time_at_risk"] == pytest.approx((data.t - data.t0).sum())
    assert result.inference["distribution"] == "normal"
    # constant-only main equation for the LR test
    k = len(b)
    null = minimize(lambda v: -oracle_loglik(np.r_[v[0], 0.0, v[1:]], dist, metric, data).sum(),
                    np.r_[b[0], b[2:]], method="BFGS", options={"gtol": 1e-9})
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (null.fun - fitted.fun),
                                                               rel=1e-6, abs=1e-6)
    assert result.tests["model"]["df"] == 1 and k == (2 if dist == "exponential" else 3)


def test_metrics_relations_and_ancillary_transforms(data):
    ph = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="weibull")
    aft = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="weibull",
                   metric="aft")
    b_ph = {c.term: c.estimate for c in ph.coefficients}
    b_aft = {c.term: c.estimate for c in aft.coefficients}
    p = np.exp(b_ph["/ln_p"])
    assert b_aft["x"] == pytest.approx(-b_ph["x"] / p, rel=1e-8)
    assert ph.metrics["log_likelihood"] == pytest.approx(aft.metrics["log_likelihood"], rel=1e-10)
    se_lnp = next(c.std_error for c in ph.coefficients if c.term == "/ln_p")
    assert ph.extra["p"]["estimate"] == pytest.approx(p)
    assert ph.extra["p"]["std_error"] == pytest.approx(p * se_lnp)
    assert ph.extra["1/p"]["estimate"] == pytest.approx(1 / p)
    z = stats.norm.isf(0.025)
    assert ph.extra["p"]["ci_low"] == pytest.approx(np.exp(b_ph["/ln_p"] - z * se_lnp))
    assert ph.extra["hazard_ratios"]["x"]["ratio"] == pytest.approx(np.exp(b_ph["x"]))
    assert "time_ratios" in aft.extra
    expo = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="exponential")
    expo_aft = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="exponential",
                        metric="aft")
    assert_allclose([c.estimate for c in expo_aft.coefficients],
                    [-c.estimate for c in expo.coefficients], rtol=1e-9)
    ll = expo.metrics["log_likelihood"]
    assert expo.metrics["aic"] == pytest.approx(-2 * ll + 2 * 2)
    assert expo.metrics["bic"] == pytest.approx(-2 * ll + np.log(len(data)) * 2)
    lognormal = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="lognormal")
    lnsigma = {c.term: c.estimate for c in lognormal.coefficients}["/lnsigma"]
    assert lognormal.extra["sigma"]["estimate"] == pytest.approx(np.exp(lnsigma))


def test_ancillary_equation_and_strata(data):
    result = oe.streg(data=data, time="t", failure="d", entry="t0", x=["x"], ancillary=["z"],
                      distribution="weibull")
    terms = [c.term for c in result.coefficients]
    assert terms == ["Intercept", "x", "ln_p:Intercept", "ln_p:z"]
    assert [c.equation for c in result.coefficients] == ["t", "t", "ln_p", "ln_p"]
    b = np.array([c.estimate for c in result.coefficients])
    fitted = minimize(lambda v: -oracle_loglik(v, "weibull", "ph", data, ("z",)).sum(),
                      b + 0.05, method="BFGS", options={"gtol": 1e-9})
    assert_allclose(b, fitted.x, atol=2e-5)
    strata = oe.streg(data=data, time="t", failure="d", x=["x"], strata="s",
                      distribution="loglogistic")
    manual = oe.streg(data=data.assign(s1=(data.s == 1) * 1.0), time="t", failure="d",
                      x=["x", "s1"], ancillary=["s1"], distribution="loglogistic")
    assert_allclose([c.estimate for c in strata.coefficients],
                    [c.estimate for c in manual.coefficients], rtol=1e-8)
    assert [c.term for c in strata.coefficients][:3] == ["Intercept", "x", "s[1]"]
    # Stata tests the stratum indicators of the main equation too ([ST] streg example 8)
    assert strata.tests["model"]["df"] == 2
    assert strata.tests["model"]["statistic"] == pytest.approx(
        manual.tests["model"]["statistic"], rel=1e-7)


def test_covariances_and_weights(data):
    n = len(data)
    base = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="lognormal",
                    entry="t0")
    b = np.array([c.estimate for c in base.coefficients])
    scores = approx_fprime(b, lambda v: oracle_loglik(v, "lognormal", "aft", data),
                           centered=True)
    bread = np.linalg.inv(-approx_hess(b, lambda v: oracle_loglik(v, "lognormal", "aft",
                                                                  data).sum()))
    robust = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="lognormal",
                      entry="t0", covariance="robust")
    assert_allclose(np.array(robust.covariance_matrix),
                    n / (n - 1) * bread @ scores.T @ scores @ bread, rtol=1e-4)
    opg = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="lognormal",
                   entry="t0", covariance="opg")
    assert_allclose(np.array(opg.covariance_matrix), np.linalg.inv(scores.T @ scores),
                    rtol=1e-4)
    cluster = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="lognormal",
                       entry="t0", cluster="cl")
    sums = pd.DataFrame(scores).groupby(data.cl.to_numpy()).sum().to_numpy()
    g = data.cl.nunique()
    assert_allclose(np.array(cluster.covariance_matrix),
                    g / (g - 1) * bread @ sums.T @ sums @ bread, rtol=1e-4)
    expanded = data.loc[data.index.repeat(data.w)].reset_index(drop=True)
    fw = oe.streg(data=data, time="t", failure="d", x=["x"], weights="w", weight_type="fweight")
    plain = oe.streg(data=expanded, time="t", failure="d", x=["x"])
    assert_allclose([c.estimate for c in fw.coefficients],
                    [c.estimate for c in plain.coefficients], rtol=1e-8)
    assert_allclose([c.std_error for c in fw.coefficients],
                    [c.std_error for c in plain.coefficients], rtol=1e-8)
    iw = oe.streg(data=data, time="t", failure="d", entry="t0", x=["x"], weights="pw",
                  weight_type="iweight")
    b_iw = np.array([c.estimate for c in iw.coefficients])
    fitted = minimize(lambda v: -oracle_loglik(v, "weibull", "ph", data,
                                               weights=data.pw.to_numpy()).sum(),
                      b_iw + 0.05, method="BFGS", options={"gtol": 1e-9})
    assert_allclose(b_iw, fitted.x, atol=2e-5)
    pw = oe.streg(data=data, time="t", failure="d", entry="t0", x=["x"], weights="pw",
                  weight_type="pweight")
    assert pw.spec.covariance == "robust"
    assert_allclose([c.estimate for c in pw.coefficients], b_iw, rtol=1e-8)


def test_multiple_records_cluster_on_id(data):
    first = data.assign(stop=data.t / 2, event=0.0, start=0.0)
    second = data.assign(stop=data.t, event=data.d, start=data.t / 2)
    long = pd.concat([first, second]).sort_index(kind="stable").reset_index()
    single = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="weibull")
    split = oe.streg(data=long, time="stop", failure="event", entry="start", id="index",
                     x=["x"], distribution="weibull", covariance="robust")
    assert_allclose([c.estimate for c in split.coefficients],
                    [c.estimate for c in single.coefficients], rtol=1e-8)
    assert split.metrics["log_likelihood"] == pytest.approx(single.metrics["log_likelihood"])
    assert split.inference["cluster_column"] == "index"
    assert split.metrics["n_subjects"] == len(data)
    robust_single = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="weibull",
                             covariance="cluster", cluster="cl")
    assert robust_single.inference["cluster_count"] == data.cl.nunique()


def test_derivatives_failures_and_rendering(data):
    tensor = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    x = tensor(np.column_stack([np.ones(len(data)), data.x]))
    z = tensor(np.column_stack([np.ones(len(data)), data.z]))
    survival = SurvivalData(tensor(data.t), tensor(data.t0), tensor(data.d))
    for dist, metric in CASES:
        objective = ParametricObjective(dist, metric, x, None if dist == "exponential" else z,
                                        survival, tensor(data.pw), tensor(0.1 * data.z))
        theta = tensor([0.1, 0.2, 0.15, -0.1][:objective.size])
        report = check_derivatives(objective, theta)
        assert report["gradient_max_rel_error"] < 1e-8
        assert report["hessian_max_rel_error"] < 1e-8
    for kwargs, code in (({"distribution": "gompertz", "metric": "aft"}, "invalid_spec"),
                         ({"distribution": "lognormal", "metric": "ph"}, "invalid_spec"),
                         ({"distribution": "exponential", "ancillary": ["z"]}, "invalid_spec"),
                         ({"distribution": "cauchy"}, "invalid_spec")):
        with pytest.raises(AnalysisError) as error:
            oe.streg(data=data, time="t", failure="d", x=["x"], **kwargs)
        assert error.value.code == code
    with pytest.raises(AnalysisError) as error:
        oe.streg(data=data.assign(d=0.0), time="t", failure="d", x=["x"])
    assert error.value.code == "no_failures"
    collinear = oe.streg(data=data.assign(x2=2 * data.x), time="t", failure="d", x=["x", "x2"])
    assert "x2" in collinear.provenance["omitted_terms"]
    result = oe.streg(data=data, time="t", failure="d", x=["x"], distribution="gompertz")
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert isinstance(result, ResultBundle) and "/gamma" in result.summary()
    assert "tabular" in result.to_latex()


def gengamma_loglik(theta, frame):
    """Stata's generalized gamma written with scipy.stats.gengamma (c = kappa / sigma)."""
    x = np.column_stack([np.ones(len(frame)), frame.x])
    mu = x @ theta[:2]
    sigma, kappa = np.exp(theta[2]), theta[3]
    gamma = kappa ** -2
    c = kappa / sigma
    law = stats.gengamma(a=gamma, c=c, scale=np.exp(mu) * gamma ** (-1 / c))
    t, t0, d = frame.t.to_numpy(), frame.t0.to_numpy(), frame.d.to_numpy()
    value = d * law.logpdf(t) + (1 - d) * law.logsf(t)
    late = t0 > 0
    value[late] -= law.logsf(t0)[late]
    return value


def test_generalized_gamma_matches_scipy_gengamma():
    rng = np.random.default_rng(2)
    n = 400
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    t = rng.gamma(2.0, 1.0, n) * np.exp(0.3 * frame.x)
    c = rng.exponential(4, n)
    frame["t"], frame["d"] = np.minimum(t, c), (t <= c).astype(float)
    frame["t0"] = np.where(rng.random(n) < 0.3, frame.t * rng.uniform(0.1, 0.9, n), 0.0)
    result = oe.streg(data=frame, time="t", failure="d", entry="t0", x=["x"],
                      distribution="ggamma")
    assert [c.term for c in result.coefficients] == ["Intercept", "x", "/lnsigma", "/kappa"]
    b = np.array([c.estimate for c in result.coefficients])
    fitted = minimize(lambda v: -gengamma_loglik(v, frame).sum(), b + 0.03, method="BFGS",
                      options={"gtol": 1e-9})
    assert_allclose(b, fitted.x, atol=5e-5)
    log_time = (frame.d * np.log(frame.t)).sum()            # Stata's ln t scale
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun + log_time, rel=1e-9)
    hessian = approx_hess(b, lambda v: gengamma_loglik(v, frame).sum())
    assert_allclose([c.std_error for c in result.coefficients],
                    np.sqrt(np.diag(np.linalg.inv(-hessian))), rtol=1e-4)
    assert result.provenance["derivatives"].startswith("numerical")
    assert result.extra["sigma"]["estimate"] == pytest.approx(np.exp(b[2]))
    # kappa = 1 is the Weibull AFT model and kappa -> 0 the lognormal
    tensor = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    survival = SurvivalData(tensor(frame.t), tensor(frame.t0), tensor(frame.d))
    x = tensor(np.column_stack([np.ones(n), frame.x]))
    one = tensor(np.ones((n, 1)))
    gg = ParametricObjective("ggamma", "aft", x, one, survival, tensor(np.ones(n)), None)
    weibull = ParametricObjective("weibull", "aft", x, one, survival, tensor(np.ones(n)), None)
    lognormal = ParametricObjective("lognormal", "aft", x, one, survival, tensor(np.ones(n)),
                                    None)
    theta = tensor([0.4, 0.2, -0.3])
    assert float(gg.value(torch.cat([theta, tensor([1.0])]))) == pytest.approx(
        float(weibull.value(tensor([0.4, 0.2, 0.3]))), rel=1e-12)
    assert float(gg.value(torch.cat([theta, tensor([0.0])]))) == pytest.approx(
        float(lognormal.value(theta)), rel=1e-12)
    for kappa in (0.7, -0.4):
        report = check_derivatives(gg, torch.cat([theta, tensor([kappa])]))
        assert report["gradient_max_rel_error"] < 1e-8
        assert report["hessian_max_rel_error"] < 1e-6
