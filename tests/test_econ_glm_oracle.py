"""Independent verification of the glm family: Stata output and brute-force likelihoods.

Nothing here reuses a formula of ``openecon.econometrics.glm``:

1. REAL STATA OUTPUT. statsmodels ships the ``e()`` results of Stata's ``glm``,
   ``poisson`` and ``nbreg`` in its own test-suite (coefficients, standard errors,
   log likelihood, deviance, Pearson chi2, AIC, BIC, Wald/LR chi2, alpha, for
   several weight types and variance estimators). Those numbers pin the reporting
   conventions down directly.
2. BRUTE FORCE. Per-observation log likelihoods are written out below in NumPy,
   maximized with scipy's BFGS and polished by Newton steps on Richardson-extrapolated
   central differences. The Hessian, the score rows and therefore every covariance
   (OIM, OPG, robust, cluster) come from numerical differentiation, not from the
   analytic derivatives of the implementation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import optimize, special, stats

import openecon as oe

X = ["x1", "x2"]
STEP = 1e-3


# ---- oracle machinery --------------------------------------------------------------------


def num_jac(f, x, h=STEP):
    """Jacobian by central differences at h, h/2, h/4 with two Richardson extrapolations."""
    x = np.asarray(x, float)
    columns = []
    for j in range(len(x)):
        step = h * max(1.0, abs(x[j]))
        e = np.zeros_like(x)
        e[j] = step
        d = [(f(x + e / s) - f(x - e / s)) / (2 * step / s) for s in (1, 2, 4)]
        first, second = (4 * d[1] - d[0]) / 3, (4 * d[2] - d[1]) / 3
        columns.append((16 * second - first) / 15)
    return np.stack(columns, axis=-1)


def brute_force(ll_obs, start, w, h=STEP):
    """Maximize ``sum_i w_i ll_obs(theta)_i``; returns theta, Hessian, score rows, value."""
    def total(theta):
        return float(w @ ll_obs(theta))

    def gradient(theta):
        return num_jac(total, theta, h)

    def penalized(theta):
        value = -total(theta)
        return value if np.isfinite(value) else 1e300

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theta = optimize.minimize(penalized, start, method="BFGS", options={"gtol": 1e-7}).x
        previous = np.inf
        for _ in range(60):
            hessian = num_jac(gradient, theta, h)
            step = np.linalg.solve(-(hessian + hessian.T) / 2, gradient(theta))
            theta = theta + step
            size = np.abs(step).max() / max(1.0, np.abs(theta).max())
            # Converged, or at the noise floor of the numerical derivatives (far below 1e-7).
            if size <= 1e-12 or (size <= 1e-9 and size >= previous):
                break
            previous = size
        else:  # pragma: no cover - the oracle itself failed
            raise AssertionError("the brute-force oracle did not converge")
        hessian = num_jac(gradient, theta, h)
        return theta, (hessian + hessian.T) / 2, num_jac(ll_obs, theta, h), total(theta)


def ml_covariances(hessian, scores, w, n, *, frequency=False, groups=None):
    """Stata's -ml- variance estimators for the likelihood ``sum_i w_i l_i``.

    OIM ``(-H)^-1``; OPG ``(sum w_i s_i s_i')^-1``; robust
    ``N/(N-1) (-H)^-1 M (-H)^-1`` with ``M = sum f_i s_i s_i'`` for frequency weights
    (replicated rows) and ``sum (w_i s_i)(w_i s_i)'`` otherwise; cluster
    ``G/(G-1)`` with the weighted scores summed within clusters.
    """
    bread = np.linalg.inv(-hessian)
    weighted = scores * w[:, None]
    out = {"nonrobust": bread, "opg": np.linalg.inv(weighted.T @ scores)}
    meat = weighted.T @ scores if frequency else weighted.T @ weighted
    out["robust"] = n / (n - 1) * bread @ meat @ bread
    if groups is not None:
        ids = np.unique(groups)
        sums = np.stack([weighted[groups == g].sum(axis=0) for g in ids])
        out["cluster"] = len(ids) / (len(ids) - 1) * bread @ sums.T @ sums @ bread
    return out


def coefs(result):
    return np.array([c.estimate for c in result.coefficients])


def ses(result):
    return np.array([c.std_error for c in result.coefficients])


def cov(result):
    return np.asarray(result.covariance_matrix)


def check_inference(result, theta, covariance, *, alpha=0.05, rtol=2e-6):
    """Coefficients, full covariance, z statistics, p-values and confidence limits."""
    assert_allclose(coefs(result), theta, rtol=1e-7, atol=1e-8)
    # Every element of the covariance, relative to the standard errors of its row and column
    # (an off-diagonal element may be arbitrarily close to zero).
    units = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    assert_allclose(cov(result) / units, covariance / units, rtol=rtol, atol=rtol)
    assert_allclose(ses(result), np.sqrt(np.diag(covariance)), rtol=rtol)
    # Inference is normal: z = b/se, p = 2 Phi(-|z|), limits b -/+ z_(1-alpha/2) se.
    assert result.inference["use_t"] is False and result.inference["df_inference"] is None
    assert result.inference["distribution"] == "normal"
    z = coefs(result) / ses(result)
    crit = stats.norm.isf(alpha / 2)
    assert_allclose([c.statistic for c in result.coefficients], z, rtol=1e-12)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs(z)),
                    rtol=1e-9, atol=1e-300)
    assert_allclose([c.ci_low for c in result.coefficients], coefs(result) - crit * ses(result),
                    rtol=1e-9, atol=1e-12)
    assert_allclose([c.ci_high for c in result.coefficients],
                    coefs(result) + crit * ses(result), rtol=1e-9, atol=1e-12)


def check_test(entry, statistic, df, *, rtol=1e-6):
    assert entry["distribution"] == "chi2" and entry["df"] == df
    assert entry["statistic"] == pytest.approx(statistic, rel=rtol, abs=1e-8)
    # The independent finite-difference Hessian bounds the Wald statistic above.
    # A relative statistic error is amplified in extreme probability tails:
    # comparing two ~1e-30 probabilities at an unrelated fixed tolerance can
    # reject a statistic well inside its existing 1e-6 accuracy bound. Verify
    # the reported probability more tightly against SciPy at its actual
    # statistic, and against the independent statistic's transformed interval.
    assert entry["p_value"] == pytest.approx(
        stats.chi2.sf(entry["statistic"], df), rel=1e-9, abs=1e-300)
    radius = max(abs(statistic) * rtol, 1e-8)
    assert stats.chi2.sf(statistic + radius, df) - 1e-300 <= entry["p_value"]
    assert entry["p_value"] <= stats.chi2.sf(max(0.0, statistic - radius), df) + 1e-300


def wald(theta, covariance, index):
    index = list(index)
    block = covariance[np.ix_(index, index)]
    return float(theta[index] @ np.linalg.solve(block, theta[index]))


def weight_kwargs(kind):
    """Keyword arguments, the likelihood weights and N for one weight type of ``data``."""
    return {} if kind is None else {"weights": "fw" if kind == "fweight" else "aw",
                                    "weight_type": kind}


def likelihood_weights(frame, kind):
    n = len(frame)
    if kind is None:
        return np.ones(n), n
    if kind == "fweight":
        return frame.fw.to_numpy(), int(frame.fw.sum())
    raw = frame.aw.to_numpy()
    return (raw * n / raw.sum(), n) if kind == "aweight" else (raw, n)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(770)
    n = 360
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    sizes = rng.multinomial(n, rng.dirichlet(np.full(24, 2.0)))       # unbalanced clusters
    frame["g"] = np.repeat(np.arange(24), sizes)
    frame["h"] = rng.integers(0, 9, size=n)
    eta = 0.4 + 0.3 * frame.x1 - 0.25 * frame.x2
    mu = np.exp(eta)
    frame["gauss"] = 1 + 0.5 * frame.x1 - 0.3 * frame.x2 + rng.normal(size=n)
    frame["pos"] = rng.gamma(3.0, mu / 3.0)
    frame["cnt"] = rng.poisson(mu).astype(float)
    frame["over"] = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    p = special.expit(0.2 + 0.6 * frame.x1 - 0.4 * frame.x2)
    frame["bin"] = rng.binomial(1, p).astype(float)
    frame["m"] = rng.integers(1, 9, size=n).astype(float)
    frame["succ"] = rng.binomial(frame.m.astype(int), p).astype(float)
    frame["frac"] = np.where(rng.uniform(size=n) < 0.15, rng.integers(0, 2, size=n),
                             rng.beta(p * 6, (1 - p) * 6))
    frame["z"] = rng.normal(size=n)
    frame["beta"] = rng.beta(p * np.exp(1.8 + 0.4 * frame.z), (1 - p) * np.exp(1.8 + 0.4 * frame.z))
    frame["beta"] = frame.beta.clip(1e-6, 1 - 1e-6)
    frame["aw"] = rng.uniform(0.3, 3.0, size=n)
    frame["fw"] = rng.integers(1, 4, size=n).astype(float)
    frame["off"] = rng.normal(0, 0.3, size=n)
    frame["expo"] = rng.uniform(0.5, 4.0, size=n)
    frame["rate"] = rng.poisson(mu * frame.expo).astype(float)
    frame["cat"] = rng.choice(["a", "b", "c"], size=n, p=[0.5, 0.3, 0.2])
    # Inverse Gaussian with its canonical link: 1/mu^2 = eta stays well inside eta > 0.
    frame["wald"] = rng.wald((1.0 + 0.2 * frame.x1 + 0.1 * frame.x2) ** -0.5, 3.0)
    return frame


# ---- 1. real Stata output ---------------------------------------------------------------


def _cpunish():
    loaded = sm.datasets.cpunish.load()
    frame = pd.DataFrame(np.asarray(loaded.exog, float),
                         columns=["income", "perpoverty", "perblack", "vc", "south", "degree"])
    frame["lvc"] = np.log(frame.pop("vc"))
    frame["executions"] = np.asarray(loaded.endog, float)
    # The weights and cluster identifier of statsmodels' test_glm_weights (Stata do-file).
    frame["w"] = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3], float)
    frame["id"] = np.arange(1, 18) // 2
    return frame, ["income", "perpoverty", "perblack", "lvc", "south", "degree"]


@pytest.mark.parametrize("weight", ["none", "fweight", "aweight", "pweight"])
@pytest.mark.parametrize("vce", ["nonrobust", "hc1", "clu1"])
def test_glm_poisson_reproduces_stata_for_every_weight_type_and_vce(weight, vce):
    fixtures = pytest.importorskip("statsmodels.genmod.tests.results.results_glm_poisson_weights")
    stata = getattr(fixtures, f"results_poisson_{weight}_{vce}")
    frame, x = _cpunish()
    kwargs = {} if weight == "none" else {"weights": "w", "weight_type": weight}
    if vce == "clu1":
        kwargs["cluster"] = "id"
    elif vce == "hc1" or weight == "pweight":       # Stata: pweights imply vce(robust)
        kwargs["covariance"] = "robust"
    result = oe.glm(data=frame, y="executions", x=x, family="poisson", **kwargs)
    order = [*range(1, 7), 0]                        # Stata lists _cons last
    assert_allclose(coefs(result)[order], stata.params, rtol=5e-6)
    assert_allclose(ses(result)[order], stata.bse, rtol=5e-6)
    metrics = result.metrics
    assert result.nobs == stata.N and metrics["df_resid"] == stata.df
    assert metrics["log_likelihood"] == pytest.approx(stata.ll, rel=1e-7)
    assert metrics["deviance"] == pytest.approx(stata.deviance, rel=1e-6)
    assert metrics["pearson"] == pytest.approx(stata.deviance_p, rel=1e-6)
    assert metrics["dispersion_deviance"] == pytest.approx(stata.dispers, rel=1e-6)
    assert metrics["dispersion_pearson"] == pytest.approx(stata.dispers_p, rel=1e-6)
    assert metrics["scale"] == stata.phi == 1
    assert metrics["aic_glm"] == pytest.approx(stata.aic, rel=1e-7)
    assert metrics["bic_glm"] == pytest.approx(stata.bic, rel=1e-6)
    assert result.tests["model"]["statistic"] == pytest.approx(stata.chi2, rel=5e-6)
    assert result.tests["model"]["df"] == stata.df_m
    assert_allclose([c.p_value for c in result.coefficients][1:], stata.params_table[:6, 3],
                    rtol=2e-4, atol=1e-300)
    assert_allclose([c.ci_low for c in result.coefficients][1:], stata.params_table[:6, 4],
                    rtol=1e-5, atol=1e-9)


def _fixture_frame(endog, exog):
    exog = np.asarray(exog, float)
    exog = exog[:, [j for j in range(exog.shape[1]) if np.ptp(exog[:, j]) > 0]]   # drop _cons
    frame = pd.DataFrame(exog, columns=[f"v{j}" for j in range(exog.shape[1])])
    endog = np.asarray(endog, float)
    if endog.ndim == 2:                               # (successes, failures)
        frame["y"], frame["trials"] = endog[:, 0], endog.sum(axis=1)
    else:
        frame["y"] = endog
    return frame, [name for name in frame.columns if name.startswith("v")]


def _stata_glm_cases():
    results = pytest.importorskip("statsmodels.genmod.tests.results.results_glm")
    x = np.arange(100)
    square = np.c_[x, x ** 2]
    np.random.seed(54321)
    log_y = np.exp(-(-1.0 + 0.02 * x + 0.0001 * x ** 2)) + 0.001 * np.random.randn(100)
    np.random.seed(54321)
    np.random.randn(100)
    inverse_y = (1.0 + 0.02 * x + 0.001 * x ** 2) ** -1 + 0.001 * np.random.randn(100)
    committee = sm.datasets.committee.load()
    committee_x = np.asarray(committee.exog, float).copy()
    committee_x[:, 2] = np.log(committee_x[:, 2])
    committee_x = np.column_stack([committee_x, committee_x[:, 2] * committee_x[:, 1]])
    lbw, cancer, invgauss, medpar = (results.Lbw(), results.CancerLog(), results.InvGauss(),
                                     results.Medpar1())
    longley, star, scot = (sm.datasets.longley.load(), sm.datasets.star98.load(),
                           sm.datasets.scotland.load())
    return [
        ("gaussian", results.Longley(), longley.endog, longley.exog, {}),
        ("gaussian-log", results.GaussianLog(), log_y, square,
         {"family": "gaussian", "link": "log"}),
        ("gaussian-reciprocal", results.GaussianInverse(), inverse_y, square,
         {"family": "gaussian", "link": "reciprocal"}),
        ("binomial-trials", results.Star98(), star.endog, star.exog,
         {"family": "binomial", "trials": "trials"}),
        ("bernoulli", lbw, lbw.endog, lbw.exog, {"family": "binomial"}),
        ("gamma", results.Scotvote(), scot.endog, scot.exog, {"family": "gamma"}),
        ("gamma-log", cancer, cancer.endog, cancer.exog, {"family": "gamma", "link": "log"}),
        ("gamma-identity", results.CancerIdentity(), cancer.endog, cancer.exog,
         {"family": "gamma", "link": "identity"}),
        ("inverse-gaussian", invgauss, invgauss.endog, invgauss.exog,
         {"family": "inverse_gaussian"}),
        ("inverse-gaussian-log", results.InvGaussLog(), medpar.endog, medpar.exog,
         {"family": "inverse_gaussian", "link": "log"}),
        ("inverse-gaussian-identity", results.InvGaussIdentity(), medpar.endog, medpar.exog,
         {"family": "inverse_gaussian", "link": "identity"}),
        ("nbinomial", results.Committee(), committee.endog, committee_x,
         {"family": "nbinomial", "link": "log", "scale": "x2"}),
    ]


STATA_GLM = ["gaussian", "gaussian-log", "gaussian-reciprocal", "binomial-trials", "bernoulli",
             "gamma", "gamma-log", "gamma-identity", "inverse-gaussian", "inverse-gaussian-log",
             "inverse-gaussian-identity", "nbinomial"]


@pytest.mark.parametrize("case", STATA_GLM)
def test_glm_log_likelihood_aic_bic_and_scale_reproduce_stata(case):
    """Stata evaluates the gaussian likelihood at phi = deviance/N and the gamma and
    inverse Gaussian likelihoods at phi = 1, whatever scale multiplies the covariance."""
    name, stata, endog, exog, kwargs = next(c for c in _stata_glm_cases() if c[0] == case)
    frame, x = _fixture_frame(endog, exog)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = oe.glm(data=frame, y="y", x=x, **kwargs)
    metrics = result.metrics
    n, k = result.nobs, len(result.coefficients)
    log_likelihood = getattr(stata, "llf_Stata", stata.llf)
    # The fixtures carry 7-10 significant digits (some were typed in from Stata's display,
    # and the Star98 data differ from Stata's copy in the last digits).
    assert metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=5e-7)
    # Stata fitted the two simulated gaussian cases on a float copy of the data: its AIC
    # differs by 6e-6, while a likelihood at the Pearson scale would differ by 5e-4.
    assert metrics["aic_glm"] == pytest.approx(stata.aic_Stata, rel=5e-7, abs=2e-5)
    assert metrics["bic_glm"] == pytest.approx(stata.bic_Stata, rel=5e-7, abs=2e-3)
    assert metrics["deviance"] == pytest.approx(stata.deviance, rel=1e-6)
    assert metrics["scale"] == pytest.approx(stata.scale, rel=5e-5)
    assert metrics["df_resid"] == stata.df_resid == n - k
    assert metrics["aic"] == pytest.approx(-2 * metrics["log_likelihood"] + 2 * k, rel=1e-12)
    assert metrics["bic"] == pytest.approx(-2 * metrics["log_likelihood"] + k * np.log(n),
                                           rel=1e-12)
    if hasattr(stata, "pearson_chi2"):
        assert metrics["pearson"] == pytest.approx(stata.pearson_chi2, rel=2e-6)


def _ships():
    path = os.path.join(os.path.dirname(statsmodels.__file__), "discrete", "tests", "results",
                        "ships.csv")
    if not os.path.exists(path):
        pytest.skip("statsmodels was installed without its test data")
    return pd.read_csv(path, index_col=False).dropna()


SHIPS = [
    ("poisson", "results_poisson_clu", {"cluster": "ship"}),
    ("poisson", "results_poisson_hc1", {"covariance": "robust"}),
    ("poisson", "results_poisson_exposure_nonrobust", {"exposure": "service"}),
    ("poisson", "results_poisson_exposure_hc1", {"exposure": "service", "covariance": "robust"}),
    ("poisson", "results_poisson_exposure_clu", {"exposure": "service", "cluster": "ship"}),
    ("nbreg", "results_negbin_clu", {"cluster": "ship"}),
    ("nbreg", "results_negbin_hc1", {"covariance": "robust"}),
    ("nbreg", "results_negbin_exposure_nonrobust", {"exposure": "service"}),
    ("nbreg", "results_negbin_exposure_clu", {"exposure": "service", "cluster": "ship"}),
]


@pytest.mark.parametrize("command,fixture,kwargs", SHIPS, ids=[c[1] for c in SHIPS])
def test_poisson_and_nbreg_reproduce_stata(command, fixture, kwargs):
    fixtures = pytest.importorskip(
        "statsmodels.discrete.tests.results.results_count_robust_cluster")
    stata = getattr(fixtures, fixture)
    result = getattr(oe, command)(data=_ships(), y="accident", x=["yr_con", "op_75_79"],
                                  **kwargs)
    k = len(result.coefficients)
    order = [1, 2, 0, 3][:k]                         # Stata: slopes, _cons, /lnalpha
    # Stata stops at nrtolerance(1e-5); the flat nbreg likelihoods agree to about 1e-6.
    assert_allclose(coefs(result)[order], stata.params[:k], rtol=2e-6)
    assert_allclose(ses(result)[order], stata.bse[:k], rtol=2e-6)
    assert_allclose(cov(result)[np.ix_(order, order)], stata.cov, rtol=1e-5, atol=1e-12)
    assert result.nobs == stata.N
    metrics, model = result.metrics, result.tests["model"]
    assert metrics["log_likelihood"] == pytest.approx(stata.ll, rel=1e-10)
    assert result.extra["null_log_likelihood"] == pytest.approx(stata.ll_0, rel=1e-10)
    assert metrics["pseudo_r_squared"] == pytest.approx(1 - stata.ll / stata.ll_0, rel=1e-8)
    assert model["statistic"] == pytest.approx(stata.chi2, rel=1e-6)
    assert model["df"] == stata.df_m
    assert model["label"].startswith(stata.chi2type)          # "LR" or "Wald", as Stata prints
    assert model["p_value"] == pytest.approx(stata.p, rel=1e-5)
    assert metrics["aic"] == pytest.approx(-2 * stata.ll + 2 * stata.k, rel=1e-10)
    assert metrics["bic"] == pytest.approx(-2 * stata.ll + stata.k * np.log(stata.N), rel=1e-10)
    if command == "nbreg":
        assert metrics["alpha"] == pytest.approx(stata.alpha, rel=1e-6)
        alpha_row = stata.params_table[-1]             # alpha, se, ., ., lower, upper
        record = result.extra["alpha"]
        assert record["estimate"] == pytest.approx(alpha_row[0], rel=1e-6)
        assert record["std_error"] == pytest.approx(alpha_row[1], rel=1e-6)
        assert record["ci_low"] == pytest.approx(alpha_row[4], rel=1e-6)
        assert record["ci_high"] == pytest.approx(alpha_row[5], rel=1e-6)
        assert [c.term for c in result.coefficients][-1] == "/lnalpha"
        if hasattr(stata, "chi2_c"):                   # LR test of alpha = 0: chibar2(01)
            test = result.tests["alpha"]
            assert test["statistic"] == pytest.approx(stata.chi2_c, rel=1e-9)
            assert test["statistic"] == pytest.approx(2 * (stata.ll - stata.ll_c), rel=1e-9)
            assert test["p_value"] == pytest.approx(0.5 * stats.chi2.sf(stata.chi2_c, 1),
                                                    rel=1e-6)
            assert result.extra["poisson_log_likelihood"] == pytest.approx(stata.ll_c, rel=1e-10)
        else:
            assert "alpha" not in result.tests         # not shown with vce(robust|cluster)


# ---- 2. glm: brute force for every family, link, weight type and covariance ---------------

K_NB = 1.5      # fixed overdispersion of the nbinomial cases

INVERSE = {
    "identity": lambda eta: eta, "log": np.exp, "logit": special.expit, "probit": special.ndtr,
    "cloglog": lambda eta: -np.expm1(-np.exp(eta)), "loglog": lambda eta: np.exp(-np.exp(-eta)),
    "reciprocal": lambda eta: 1 / eta, "inverse_squared": lambda eta: eta ** -0.5,
    "sqrt": lambda eta: eta ** 2, "nbinomial": lambda eta: np.exp(eta) / (K_NB * -np.expm1(eta)),
}
FORWARD = {
    "identity": lambda mu: mu, "log": np.log, "logit": special.logit, "probit": special.ndtri,
    "cloglog": lambda mu: np.log(-np.log1p(-mu)), "loglog": lambda mu: -np.log(-np.log(mu)),
    "reciprocal": lambda mu: 1 / mu, "inverse_squared": lambda mu: mu ** -2.0,
    "sqrt": np.sqrt, "nbinomial": lambda mu: np.log(K_NB * mu / (1 + K_NB * mu)),
}
VARIANCE = {
    "gaussian": lambda mu: np.ones_like(mu), "binomial": lambda mu: mu * (1 - mu),
    "poisson": lambda mu: mu, "gamma": lambda mu: mu ** 2, "inverse_gaussian": lambda mu: mu ** 3,
    "nbinomial": lambda mu: mu + K_NB * mu ** 2,
}


def unit_ll(family, y, mu, trials=None):
    """Per-observation log likelihood at dispersion 1 (y a count of successes for binomial)."""
    if family == "gaussian":
        return -0.5 * (y - mu) ** 2 - 0.5 * np.log(2 * np.pi)
    if family == "binomial":
        m = np.ones_like(y) if trials is None else trials
        return (special.xlogy(y, mu) + special.xlogy(m - y, 1 - mu) + special.gammaln(m + 1)
                - special.gammaln(y + 1) - special.gammaln(m - y + 1))
    if family == "poisson":
        return special.xlogy(y, mu) - mu - special.gammaln(y + 1)
    if family == "gamma":
        return -y / mu - np.log(mu)
    if family == "inverse_gaussian":
        return -(y - mu) ** 2 / (2 * y * mu ** 2) - 0.5 * np.log(2 * np.pi * y ** 3)
    return (special.gammaln(y + 1 / K_NB) - special.gammaln(1 / K_NB) - special.gammaln(y + 1)
            + special.xlogy(y, K_NB * mu / (1 + K_NB * mu)) - np.log1p(K_NB * mu) / K_NB)


def saturated_mean(family, y, trials=None):
    """The mean that maximizes the unit likelihood of each observation (for the deviance)."""
    if family == "binomial":
        return y / (np.ones_like(y) if trials is None else trials)
    return y


def glm_oracle(frame, family, link, outcome, kind, *, trials=None, offset=None, x=X):
    """Brute-force GLM fit: everything Stata's glm reports, from the likelihood alone."""
    n = len(frame)
    design = np.column_stack([np.ones(n), *(frame[name].to_numpy(float) for name in x)])
    y = frame[outcome].to_numpy(float)
    m = None if trials is None else frame[trials].to_numpy(float)
    base = np.zeros(n) if offset is None else frame[offset].to_numpy(float)
    w, nobs = likelihood_weights(frame, kind)
    inverse = INVERSE[link]

    def ll_obs(theta):
        mu = inverse(design @ theta + base)
        return unit_ll(family, y, mu if m is None else m * 0 + mu, m)

    # Start: least squares of the link of a smoothed outcome (independent of the estimator).
    share = y if m is None else y / m
    smooth = (share + 0.5) / 2 if family == "binomial" else (y + y.mean()) / 2
    start = np.linalg.lstsq(design, FORWARD[link](smooth) - base, rcond=None)[0]
    theta, hessian, scores, value = brute_force(ll_obs, start, w)
    mu = inverse(design @ theta + base)
    prior = w if m is None else w * m
    with np.errstate(all="ignore"):
        saturated = unit_ll(family, y, saturated_mean(family, y, m), m)
    deviance = float(2 * (w @ (saturated - unit_ll(family, y, mu, m))))
    pearson = float((prior * (share - mu) ** 2 / VARIANCE[family](mu)).sum())
    k = design.shape[1]
    continuous = family in {"gaussian", "gamma", "inverse_gaussian"}
    phi = pearson / (nobs - k) if continuous else 1.0
    if family == "gaussian":        # Stata: phi = deviance / N concentrated out of the likelihood
        log_likelihood = float(w @ (-0.5 * ((y - mu) ** 2 / (deviance / nobs)
                                            + np.log(2 * np.pi * deviance / nobs))))
    else:
        log_likelihood = value
    covariances = ml_covariances(hessian, scores, w, nobs, frequency=kind == "fweight",
                                 groups=frame.g.to_numpy())
    covariances["nonrobust"] = covariances["nonrobust"] * phi
    covariances["opg"] = covariances["opg"] * phi ** 2
    return {"theta": theta, "cov": covariances, "deviance": deviance, "pearson": pearson,
            "phi": phi, "ll": log_likelihood, "nobs": nobs, "k": k, "mu": mu, "design": design,
            "w": w, "prior": prior}


GLM_CASES = [
    ("gaussian", "identity", "gauss"), ("gaussian", "log", "pos"),
    ("gaussian", "reciprocal", "pos"),
    ("binomial", "logit", "bin"), ("binomial", "probit", "bin"), ("binomial", "cloglog", "bin"),
    ("binomial", "loglog", "bin"), ("binomial", "logit", "succ"), ("binomial", "cloglog", "succ"),
    ("poisson", "log", "cnt"), ("poisson", "identity", "cnt"), ("poisson", "sqrt", "cnt"),
    ("gamma", "log", "pos"), ("gamma", "reciprocal", "pos"), ("gamma", "identity", "pos"),
    ("inverse_gaussian", "log", "pos"), ("inverse_gaussian", "inverse_squared", "wald"),
    ("nbinomial", "log", "over"), ("nbinomial", "nbinomial", "over"),
]


def glm_kwargs(family, link, outcome):
    kwargs = {"family": family, "link": link}
    if link == "sqrt":
        kwargs.update(link="power", power=0.5)
    if family == "nbinomial":
        kwargs["dispersion"] = K_NB
    if outcome == "succ":
        kwargs["trials"] = "m"
    return kwargs


@pytest.mark.parametrize("family,link,outcome", GLM_CASES,
                         ids=[f"{c[0]}-{c[1]}-{c[2]}" for c in GLM_CASES])
@pytest.mark.parametrize("kind", [None, "fweight", "aweight", "iweight", "pweight"])
def test_glm_matches_a_brute_force_likelihood(data, family, link, outcome, kind):
    oracle = glm_oracle(data, family, link, outcome, kind,
                        trials="m" if outcome == "succ" else None)
    kwargs = {**glm_kwargs(family, link, outcome), **weight_kwargs(kind)}
    theta, n, k = oracle["theta"], oracle["nobs"], oracle["k"]
    kinds = ["robust", "cluster"] if kind == "pweight" else ["nonrobust", "opg", "robust",
                                                             "cluster"]
    for covariance in kinds:
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.glm(data=data, y=outcome, x=X, **kwargs, **chosen)
        expected = oracle["cov"][covariance]
        check_inference(result, theta, expected, rtol=5e-6)
        check_test(result.tests["model"], wald(theta, expected, [1, 2]), 2, rtol=2e-5)
        metrics = result.metrics
        assert result.nobs == n and metrics["df_resid"] == n - k
        assert metrics["deviance"] == pytest.approx(oracle["deviance"], rel=1e-8, abs=1e-9)
        assert metrics["pearson"] == pytest.approx(oracle["pearson"], rel=1e-8)
        assert metrics["dispersion_deviance"] == pytest.approx(oracle["deviance"] / (n - k),
                                                               rel=1e-8)
        assert metrics["dispersion_pearson"] == pytest.approx(oracle["pearson"] / (n - k),
                                                              rel=1e-8)
        assert metrics["scale"] == pytest.approx(oracle["phi"], rel=1e-8)
        assert metrics["bic_glm"] == pytest.approx(oracle["deviance"] - (n - k) * np.log(n),
                                                   rel=1e-8)
        if family == "gaussian" and kind in {"iweight", "pweight"}:
            continue        # which N concentrates phi under unnormalized weights is not documented
        log_likelihood = oracle["ll"]
        assert metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=1e-9)
        assert metrics["aic"] == pytest.approx(-2 * log_likelihood + 2 * k, rel=1e-9)
        assert metrics["bic"] == pytest.approx(-2 * log_likelihood + k * np.log(n), rel=1e-9)
        assert metrics["aic_glm"] == pytest.approx((-2 * log_likelihood + 2 * k) / n, rel=1e-9)


def test_glm_irls_reports_the_expected_information(data):
    """optimizer='irls': the same coefficients, EIM = X' diag(w (dmu/deta)^2 / V) X."""
    for family, link, outcome in [("binomial", "probit", "bin"), ("binomial", "cloglog", "succ"),
                                  ("gamma", "log", "pos"), ("poisson", "sqrt", "cnt"),
                                  ("gaussian", "log", "pos"), ("nbinomial", "log", "over"),
                                  ("inverse_gaussian", "log", "pos")]:
        for kind in (None, "fweight", "aweight"):
            oracle = glm_oracle(data, family, link, outcome, kind,
                                trials="m" if outcome == "succ" else None)
            design, theta, n = oracle["design"], oracle["theta"], oracle["nobs"]
            eta = design @ theta
            slope = (INVERSE[link](eta + 1e-5) - INVERSE[link](eta - 1e-5)) / 2e-5   # dmu/deta
            weights = oracle["prior"] * slope ** 2 / VARIANCE[family](oracle["mu"])
            bread = np.linalg.inv(design.T @ (design * weights[:, None]))
            kwargs = {**glm_kwargs(family, link, outcome), **weight_kwargs(kind),
                      "optimizer": "irls"}
            result = oe.glm(data=data, y=outcome, x=X, **kwargs)
            assert result.extra["information"] == "expected"
            check_inference(result, theta, bread * oracle["phi"], rtol=2e-6)
            # The sandwich keeps its meat but takes the expected information as bread.
            unit_scores = ((data[outcome].to_numpy() / (data.m.to_numpy() if outcome == "succ"
                                                        else 1) - oracle["mu"])
                           / VARIANCE[family](oracle["mu"]) * slope)
            if outcome == "succ":
                unit_scores = unit_scores * data.m.to_numpy()
            scores = design * unit_scores[:, None]
            w = oracle["w"]
            meat = ((scores * w[:, None]).T @ scores if kind == "fweight"
                    else (scores * w[:, None]).T @ (scores * w[:, None]))
            robust = oe.glm(data=data, y=outcome, x=X, **kwargs, covariance="robust")
            check_inference(robust, theta, n / (n - 1) * bread @ meat @ bread, rtol=2e-6)


def test_glm_offset_exposure_categorical_missing_and_collinear_designs(data):
    # Offset: the linear predictor is x'b + offset, with every weight type.
    for family, link, outcome in [("poisson", "log", "cnt"), ("gamma", "log", "pos"),
                                  ("binomial", "logit", "succ")]:
        trials = "m" if outcome == "succ" else None
        for kind in (None, "aweight"):
            oracle = glm_oracle(data, family, link, outcome, kind, trials=trials, offset="off")
            result = oe.glm(data=data, y=outcome, x=X, offset="off",
                            **glm_kwargs(family, link, outcome), **weight_kwargs(kind))
            check_inference(result, oracle["theta"], oracle["cov"]["nonrobust"], rtol=5e-6)
            assert result.metrics["deviance"] == pytest.approx(oracle["deviance"], rel=1e-8)
            assert result.metrics["log_likelihood"] == pytest.approx(oracle["ll"], rel=1e-9)
    # Exposure is the offset ln(exposure).
    logged = data.assign(lexpo=np.log(data.expo))
    oracle = glm_oracle(logged, "poisson", "log", "rate", None, offset="lexpo")
    result = oe.glm(data=data, y="rate", x=X, family="poisson", exposure="expo")
    check_inference(result, oracle["theta"], oracle["cov"]["nonrobust"])
    # Categorical regressors are treatment-coded dummies (first level omitted).
    dummies = data.assign(cat_b=(data.cat == "b").astype(float), cat_c=(data.cat == "c") * 1.0)
    oracle = glm_oracle(dummies, "gamma", "log", "pos", None, x=["x1", "cat_b", "cat_c"])
    result = oe.glm(data=data, y="pos", x=["x1", "cat"], categorical=["cat"], family="gamma",
                    link="log", cluster="g")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "cat[b]", "cat[c]"]
    check_inference(result, oracle["theta"], oracle["cov"]["cluster"], rtol=5e-6)
    check_test(result.tests["model"], wald(oracle["theta"], oracle["cov"]["cluster"], [1, 2, 3]),
               3, rtol=2e-5)
    # missing='drop' is the fit on the complete rows; the default refuses missing values.
    holes = data.copy()
    holes.loc[[3, 50, 51, 200], "x1"] = np.nan
    holes.loc[[7, 300], "pos"] = np.nan
    holes.loc[[11], "aw"] = np.nan
    complete = holes.dropna(subset=["x1", "x2", "pos", "aw"]).reset_index(drop=True)
    oracle = glm_oracle(complete, "gamma", "log", "pos", "aweight")
    result = oe.glm(data=holes, y="pos", x=X, family="gamma", link="log", weights="aw",
                    weight_type="aweight", missing="drop", covariance="robust")
    assert result.nobs == len(complete) == 353 and result.dropped_rows == 7
    check_inference(result, oracle["theta"], oracle["cov"]["robust"], rtol=5e-6)
    with pytest.raises(oe.AnalysisError) as error:
        oe.glm(data=holes, y="pos", x=X, family="gamma", link="log")
    assert error.value.code == "missing_values"
    # A collinear regressor is omitted, recorded, and the rest is the smaller model.
    extra = data.assign(x3=2 * data.x1 - 3 * data.x2 + 1)
    oracle = glm_oracle(data, "poisson", "log", "cnt", None)
    result = oe.glm(data=extra, y="cnt", x=["x1", "x2", "x3"], family="poisson")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2"]
    assert result.provenance["omitted_terms"] == ["x3"]
    assert any("x3" in warning for warning in result.warnings)
    check_inference(result, oracle["theta"], oracle["cov"]["nonrobust"])
    assert result.metrics["df_resid"] == len(data) - 3


def test_glm_scale_rules_and_confidence_level(data):
    oracle = glm_oracle(data, "gamma", "log", "pos", None)
    n, k = oracle["nobs"], oracle["k"]
    unit = oracle["cov"]["nonrobust"] / oracle["phi"]
    for scale, phi in (("x2", oracle["pearson"] / (n - k)), ("dev", oracle["deviance"] / (n - k)),
                       (1, 1.0), (0.37, 0.37), (None, oracle["phi"])):
        result = oe.glm(data=data, y="pos", x=X, family="gamma", link="log", scale=scale,
                        alpha=0.1)
        assert result.metrics["scale"] == pytest.approx(phi, rel=1e-8)
        check_inference(result, oracle["theta"], unit * phi, alpha=0.1, rtol=5e-6)
        # The likelihood and its information criteria ignore the covariance scale.
        assert result.metrics["log_likelihood"] == pytest.approx(oracle["ll"], rel=1e-9)
    # Overdispersed Poisson: scale='x2' inflates the conventional covariance only.
    oracle = glm_oracle(data, "poisson", "log", "over", None)
    phi = oracle["pearson"] / (n - k)
    assert phi > 1.3
    result = oe.glm(data=data, y="over", x=X, family="poisson", scale="x2")
    check_inference(result, oracle["theta"], oracle["cov"]["nonrobust"] * phi, rtol=5e-6)
    robust = oe.glm(data=data, y="over", x=X, family="poisson", scale="x2", covariance="robust")
    check_inference(robust, oracle["theta"], oracle["cov"]["robust"], rtol=5e-6)


# ---- 3. poisson, cloglog, fracreg: single-index likelihoods ----------------------------------


def index_oracle(frame, ll_of_eta, outcome, kind, *, offset=None, x=X, intercept=True):
    """Brute force for ``l_i = ll_of_eta(y_i, x_i'b + offset_i)`` and its constant-only model."""
    n = len(frame)
    columns = [np.ones(n)] * intercept + [frame[name].to_numpy(float) for name in x]
    design = np.column_stack(columns)
    y = frame[outcome].to_numpy(float)
    base = np.zeros(n) if offset is None else frame[offset].to_numpy(float)
    w, nobs = likelihood_weights(frame, kind)
    theta, hessian, scores, value = brute_force(lambda t: ll_of_eta(y, design @ t + base),
                                                np.zeros(design.shape[1]), w)
    if intercept:
        null = brute_force(lambda t: ll_of_eta(y, t[0] + base), np.zeros(1), w)[3]
    else:
        null = float(w @ ll_of_eta(y, base))
    covariances = ml_covariances(hessian, scores, w, nobs, frequency=kind == "fweight",
                                 groups=frame.g.to_numpy())
    return {"theta": theta, "cov": covariances, "ll": value, "null": null, "nobs": nobs,
            "k": design.shape[1], "eta": design @ theta + base, "w": w, "y": y}


def poisson_ll(y, eta):
    return y * eta - np.exp(eta) - special.gammaln(y + 1)


def cloglog_ll(y, eta):
    return np.where(y == 1, np.log(-np.expm1(-np.exp(eta))), -np.exp(eta))


def fractional_ll(link):
    def ll(y, eta):
        if link == "logit":
            return y * -np.logaddexp(0, -eta) + (1 - y) * -np.logaddexp(0, eta)
        return y * special.log_ndtr(eta) + (1 - y) * special.log_ndtr(-eta)
    return ll


def check_likelihood_report(result, oracle, covariance, *, wald_only=False, parameters=None):
    """Stata's header: log likelihood, LR or Wald model test, pseudo R2, estat ic."""
    theta, k, n = oracle["theta"], oracle["k"], oracle["nobs"]
    parameters = k if parameters is None else parameters
    metrics = result.metrics
    assert result.nobs == n
    assert metrics["log_likelihood"] == pytest.approx(oracle["ll"], rel=1e-10)
    assert result.extra["null_log_likelihood"] == pytest.approx(oracle["null"], rel=1e-9)
    assert metrics["pseudo_r_squared"] == pytest.approx(1 - oracle["ll"] / oracle["null"],
                                                        rel=1e-7, abs=1e-10)
    assert metrics["aic"] == pytest.approx(-2 * oracle["ll"] + 2 * parameters, rel=1e-10)
    assert metrics["bic"] == pytest.approx(-2 * oracle["ll"] + parameters * np.log(n),
                                           rel=1e-10)
    slopes = [i for i, c in enumerate(result.coefficients)
              if c.term not in {"Intercept", "/lnalpha", "/lndelta"}]
    model = result.tests["model"]
    if covariance == "nonrobust" and not wald_only:
        assert model["label"].startswith("LR")
        check_test(model, 2 * (oracle["ll"] - oracle["null"]), len(slopes), rtol=1e-7)
    else:
        assert model["label"].startswith("Wald")
        check_test(model, wald(theta, oracle["cov"][covariance], slopes), len(slopes), rtol=2e-5)


def covariance_choices(kind):
    return ["robust", "cluster"] if kind == "pweight" else ["nonrobust", "opg", "robust",
                                                            "cluster"]


@pytest.mark.parametrize("kind", [None, "fweight", "aweight", "iweight", "pweight"])
def test_poisson_matches_a_brute_force_likelihood(data, kind):
    logged = data.assign(lexpo=np.log(data.expo))
    oracle = index_oracle(logged, poisson_ll, "rate", kind, offset="lexpo")
    theta, n, k = oracle["theta"], oracle["nobs"], oracle["k"]
    mu, y, w = np.exp(oracle["eta"]), oracle["y"], oracle["w"]
    deviance = float(2 * (w @ (special.xlogy(y, y / mu) - (y - mu))))
    pearson = float(w @ ((y - mu) ** 2 / mu))
    for covariance in covariance_choices(kind):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.poisson(data=data, y="rate", x=X, exposure="expo", **weight_kwargs(kind),
                            **chosen)
        check_inference(result, theta, oracle["cov"][covariance], rtol=5e-6)
        check_likelihood_report(result, oracle, covariance)
        assert result.metrics["deviance"] == pytest.approx(deviance, rel=1e-9)
        assert result.metrics["pearson"] == pytest.approx(pearson, rel=1e-9)
        assert result.metrics["df_resid"] == n - k
        # estat gof: both statistics are chi2(N - K) under a correctly specified Poisson.
        check_test(result.tests["gof_deviance"], deviance, n - k, rtol=1e-9)
        check_test(result.tests["gof_pearson"], pearson, n - k, rtol=1e-9)
    # The same model through an explicit offset, and through glm.
    offset = oe.poisson(data=logged, y="rate", x=X, offset="lexpo", **weight_kwargs(kind),
                        covariance="robust")
    assert_allclose(coefs(offset), theta, rtol=1e-8)
    assert_allclose(cov(offset), oracle["cov"]["robust"], rtol=5e-6)


def test_poisson_without_a_constant_compares_with_the_zero_model(data):
    oracle = index_oracle(data, poisson_ll, "cnt", None, intercept=False)
    result = oe.poisson(data=data, y="cnt", x=X, intercept=False)
    assert [c.term for c in result.coefficients] == X
    check_inference(result, oracle["theta"], oracle["cov"]["nonrobust"])
    # Without a constant the comparison model has every coefficient at zero (mu = 1).
    y = data.cnt.to_numpy()
    assert oracle["null"] == pytest.approx(float((-1 - special.gammaln(y + 1)).sum()))
    assert result.extra["null_log_likelihood"] == pytest.approx(oracle["null"], rel=1e-12)
    check_test(result.tests["model"], 2 * (oracle["ll"] - oracle["null"]), 2, rtol=1e-8)
    assert result.tests["model"]["label"] == ("LR chi2 test against the model with every "
                                              "coefficient zero")
    assert "no constant" in result.extra["null_model"]
    # nbreg without a constant: the comparison model keeps only the dispersion.
    y = data.over.to_numpy()
    design = data[X].to_numpy()
    ll = nb_ll("mean")
    full = brute_force(lambda t: ll(y, design @ t[:2], t[2]), np.array([0.1, -0.1, 0.0]),
                       np.ones(len(y)))
    null = brute_force(lambda t: ll(y, np.zeros(len(y)), t[0]), np.zeros(1), np.ones(len(y)))
    result = oe.nbreg(data=data, y="over", x=X, intercept=False)
    assert [c.term for c in result.coefficients] == [*X, "/lnalpha"]
    check_inference(result, full[0], np.linalg.inv(-full[1]), rtol=1e-5)
    assert result.extra["null_log_likelihood"] == pytest.approx(null[3], rel=1e-9)
    check_test(result.tests["model"], 2 * (full[3] - null[3]), 2, rtol=1e-7)
    assert "every coefficient zero" in result.tests["model"]["label"]
    assert "every coefficient zero" in result.extra["null_model"]


@pytest.mark.parametrize("kind", [None, "fweight", "iweight", "pweight"])
def test_cloglog_matches_a_brute_force_likelihood(data, kind):
    oracle = index_oracle(data, cloglog_ll, "bin", kind, offset="off")
    for covariance in covariance_choices(kind):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.cloglog(data=data, y="bin", x=X, offset="off", **weight_kwargs(kind),
                            **chosen)
        check_inference(result, oracle["theta"], oracle["cov"][covariance], rtol=5e-6)
        check_likelihood_report(result, oracle, covariance)
        w, y = oracle["w"], oracle["y"]
        assert result.extra["zero_outcomes"] == pytest.approx(float(w @ (y == 0)))
        assert result.extra["nonzero_outcomes"] == pytest.approx(float(w @ (y == 1)))


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("kind", [None, "fweight", "aweight", "iweight", "pweight"])
def test_fracreg_matches_a_brute_force_quasi_likelihood(data, link, kind):
    oracle = index_oracle(data, fractional_ll(link), "frac", kind)
    assert ((data.frac == 0).sum() > 5) and ((data.frac == 1).sum() > 5)     # corners included
    default = oe.fracreg(data=data, y="frac", x=X, link=link, **weight_kwargs(kind))
    assert default.spec.covariance == "robust"           # Stata's default for fracreg
    check_inference(default, oracle["theta"], oracle["cov"]["robust"], rtol=5e-6)
    for covariance in covariance_choices(kind):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.fracreg(data=data, y="frac", x=X, link=link, **weight_kwargs(kind), **chosen)
        check_inference(result, oracle["theta"], oracle["cov"][covariance], rtol=5e-6)
        check_likelihood_report(result, oracle, covariance, wald_only=True)   # always Wald
        assert result.extra["quasi_likelihood"] is True


# ---- 4. nbreg ---------------------------------------------------------------------------------


def nb_ll(form):
    def ll(y, eta, log_dispersion):
        mu, a = np.exp(eta), np.exp(log_dispersion)
        if form == "mean":                 # NB2: Var = mu + a mu^2, gamma shape 1/a
            shape, p = 1 / a, 1 / (1 + a * mu)
        else:                              # NB1: Var = mu (1 + a), gamma shape mu/a
            shape, p = mu / a, 1 / (1 + a)
        return stats.nbinom.logpmf(y, shape, p)
    return ll


def nb_oracle(frame, form, outcome, kind, *, offset=None):
    n = len(frame)
    design = np.column_stack([np.ones(n), *(frame[name].to_numpy(float) for name in X)])
    y = frame[outcome].to_numpy(float)
    base = np.zeros(n) if offset is None else frame[offset].to_numpy(float)
    w, nobs = likelihood_weights(frame, kind)
    ll = nb_ll(form)
    start = np.r_[np.log(y.mean()) - base.mean(), 0.0, 0.0, -0.5]
    theta, hessian, scores, value = brute_force(lambda t: ll(y, design @ t[:3] + base, t[3]),
                                                start, w)
    null = brute_force(lambda t: ll(y, t[0] + base, t[1]), start[[0, 3]], w)[3]
    poisson = brute_force(lambda t: poisson_ll(y, design @ t + base), start[:3], w)[3]
    covariances = ml_covariances(hessian, scores, w, nobs, frequency=kind == "fweight",
                                 groups=frame.g.to_numpy())
    return {"theta": theta, "cov": covariances, "ll": value, "null": null, "poisson": poisson,
            "nobs": nobs, "k": 3}


@pytest.mark.parametrize("form,name", [("mean", "alpha"), ("constant", "delta")])
@pytest.mark.parametrize("kind", [None, "fweight", "aweight", "iweight", "pweight"])
def test_nbreg_matches_a_brute_force_likelihood(data, form, name, kind):
    logged = data.assign(lexpo=np.log(data.expo))
    rng = np.random.default_rng(5)
    mu = data.expo * np.exp(0.3 + 0.4 * data.x1 - 0.3 * data.x2)
    logged["y"] = rng.negative_binomial(1.6, 1.6 / (1.6 + mu)).astype(float)
    oracle = nb_oracle(logged, form, "y", kind, offset="lexpo")
    theta, n = oracle["theta"], oracle["nobs"]
    for covariance in covariance_choices(kind):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.nbreg(data=logged, y="y", x=X, exposure="expo", dispersion=form,
                          **weight_kwargs(kind), **chosen, alpha=0.1)
        expected = oracle["cov"][covariance]
        assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", f"/ln{name}"]
        assert [c.equation for c in result.coefficients] == ["y", "y", "y", None]
        check_inference(result, theta, expected, alpha=0.1, rtol=1e-5)
        check_likelihood_report(result, oracle, covariance, parameters=4)
        assert result.metrics["df_resid"] == n - 4
        # alpha = exp(/lnalpha): delta-method standard error, interval exponentiated.
        dispersion, se_log = np.exp(theta[3]), np.sqrt(expected[3, 3])
        crit = stats.norm.isf(0.05)
        record = result.extra["alpha"]
        assert result.metrics[name] == pytest.approx(dispersion, rel=1e-7)
        assert record["estimate"] == pytest.approx(dispersion, rel=1e-7)
        assert record["std_error"] == pytest.approx(dispersion * se_log, rel=1e-5)
        assert record["ci_low"] == pytest.approx(dispersion * np.exp(-crit * se_log), rel=1e-5)
        assert record["ci_high"] == pytest.approx(dispersion * np.exp(crit * se_log), rel=1e-5)
        # LR test of alpha = 0: chibar2(01), shown only for likelihood-based covariances.
        if covariance in {"nonrobust", "opg"}:
            statistic = 2 * (oracle["ll"] - oracle["poisson"])
            test = result.tests["alpha"]
            assert test["statistic"] == pytest.approx(statistic, rel=1e-8)
            assert test["p_value"] == pytest.approx(0.5 * stats.chi2.sf(statistic, 1), rel=1e-6,
                                                    abs=1e-300)
            assert test["distribution"] == "chibar2" and test["df"] == 1
        else:
            assert "alpha" not in result.tests


def test_nbreg_agrees_with_statsmodels_negative_binomial(data):
    design = sm.add_constant(data[X])
    for form, method in (("mean", "nb2"), ("constant", "nb1")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sm.NegativeBinomial(data.over, design, loglike_method=method).fit(
                method="newton", maxiter=200, tol=1e-12, disp=0)
        result = oe.nbreg(data=data, y="over", x=X, dispersion=form)
        assert_allclose(coefs(result)[:3], ref.params.to_numpy()[:3], rtol=1e-6)
        assert result.metrics["alpha" if form == "mean" else "delta"] == pytest.approx(
            ref.params["alpha"], rel=1e-6)
        assert result.metrics["log_likelihood"] == pytest.approx(ref.llf, rel=1e-10)
        # statsmodels parameterizes alpha itself; the slope block of the OIM is the same.
        assert_allclose(cov(result)[:3, :3], ref.cov_params().to_numpy()[:3, :3], rtol=1e-5)
        # Fixing alpha at its estimate in glm reproduces the coefficients.
        if form == "mean":
            fixed = oe.glm(data=data, y="over", x=X, family="nbinomial",
                           dispersion=result.metrics["alpha"])
            assert_allclose(coefs(fixed), coefs(result)[:3], rtol=1e-7)
            assert fixed.metrics["log_likelihood"] == pytest.approx(
                result.metrics["log_likelihood"], rel=1e-10)


# ---- 5. betareg ---------------------------------------------------------------------------------

SCALE_INVERSE = {"log": np.exp, "identity": lambda eta: eta, "sqrt": lambda eta: eta ** 2}


def beta_oracle(frame, link, scale_link, kind, *, scale=("z",), outcome="beta"):
    n = len(frame)
    design = np.column_stack([np.ones(n), *(frame[name].to_numpy(float) for name in X)])
    scale_design = np.column_stack([np.ones(n), *(frame[name].to_numpy(float) for name in scale)])
    y = frame[outcome].to_numpy(float)
    w, nobs = likelihood_weights(frame, kind)
    k, q = design.shape[1], scale_design.shape[1]

    def ll_obs(theta):
        mu = INVERSE[link](design @ theta[:k])
        phi = SCALE_INVERSE[scale_link](scale_design @ theta[k:])
        return stats.beta.logpdf(y, mu * phi, (1 - mu) * phi)

    start = np.zeros(k + q)
    start[0] = FORWARD[link](y.mean())
    start[k] = {"log": np.log(8.0), "identity": 8.0, "sqrt": np.sqrt(8.0)}[scale_link]
    theta, hessian, scores, value = brute_force(ll_obs, start, w)
    covariances = ml_covariances(hessian, scores, w, nobs, frequency=kind == "fweight",
                                 groups=frame.g.to_numpy())
    eta, g = design @ theta[:k], FORWARD[link](y)
    centered_eta, centered_g = eta - np.average(eta, weights=w), g - np.average(g, weights=w)
    pseudo = (w @ (centered_eta * centered_g)) ** 2 / (
        (w @ centered_eta ** 2) * (w @ centered_g ** 2))
    return {"theta": theta, "cov": covariances, "ll": value, "nobs": nobs, "k": k, "q": q,
            "pseudo": float(pseudo)}


@pytest.mark.parametrize("link,scale_link", [("logit", "log"), ("probit", "log"),
                                             ("cloglog", "identity"), ("loglog", "sqrt"),
                                             ("logit", "sqrt"), ("logit", "identity")])
def test_betareg_matches_a_brute_force_likelihood(data, link, scale_link):
    oracle = beta_oracle(data, link, scale_link, None)
    theta, n, k, q = oracle["theta"], oracle["nobs"], oracle["k"], oracle["q"]
    for covariance in ("nonrobust", "opg", "robust", "cluster"):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.betareg(data=data, y="beta", x=X, scale=["z"], link=link,
                            scale_link=scale_link, **chosen)
        expected = oracle["cov"][covariance]
        assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2",
                                                         "scale:Intercept", "scale:z"]
        assert [c.equation for c in result.coefficients] == ["beta"] * 3 + ["scale"] * 2
        check_inference(result, theta, expected, rtol=1e-5)
        metrics = result.metrics
        assert metrics["log_likelihood"] == pytest.approx(oracle["ll"], rel=1e-10)
        assert metrics["aic"] == pytest.approx(-2 * oracle["ll"] + 2 * (k + q), rel=1e-10)
        assert metrics["bic"] == pytest.approx(-2 * oracle["ll"] + (k + q) * np.log(n),
                                               rel=1e-10)
        assert metrics["df_resid"] == n - k - q and result.nobs == n
        assert metrics["pseudo_r_squared"] == pytest.approx(oracle["pseudo"], rel=1e-7)
        check_test(result.tests["model"], wald(theta, expected, [1, 2]), 2, rtol=5e-5)


@pytest.mark.parametrize("kind", ["fweight", "iweight", "pweight"])
def test_betareg_weights_and_constant_precision(data, kind):
    oracle = beta_oracle(data, "logit", "log", kind, scale=())
    theta = oracle["theta"]
    for covariance in covariance_choices(kind):
        chosen = {"cluster": "g"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.betareg(data=data, y="beta", x=X, **weight_kwargs(kind), **chosen)
        expected = oracle["cov"][covariance]
        check_inference(result, theta, expected, rtol=1e-5)
        assert result.metrics["log_likelihood"] == pytest.approx(oracle["ll"], rel=1e-10)
        assert result.metrics["pseudo_r_squared"] == pytest.approx(oracle["pseudo"], rel=1e-7)
        # Constant precision: phi = exp(scale:Intercept) by the delta method.
        phi, se_log = np.exp(theta[3]), np.sqrt(expected[3, 3])
        record = result.extra["precision"]
        assert record["estimate"] == pytest.approx(phi, rel=1e-7)
        assert record["std_error"] == pytest.approx(phi * se_log, rel=1e-5)
        assert record["ci_low"] == pytest.approx(phi * np.exp(-stats.norm.isf(0.025) * se_log),
                                                 rel=1e-5)
        assert record["ci_high"] == pytest.approx(phi * np.exp(stats.norm.isf(0.025) * se_log),
                                                  rel=1e-5)


# ---- 6. ppmlhdfe: the explicit dummy-variable Poisson -----------------------------------------


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(4242)
    sizes = rng.integers(3, 16, size=30)                       # unbalanced firms
    firm = np.repeat(np.arange(30), sizes)
    n = len(firm)
    frame = pd.DataFrame({"firm": firm, "year": rng.integers(0, 7, size=n),
                          "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    frame["x1"] += 0.05 * frame.firm                          # correlated with the effects
    effect = rng.normal(0, 0.5, size=30)[firm] + 0.15 * frame.year
    frame["expo"] = rng.uniform(0.5, 3.0, size=n)
    frame["y"] = rng.poisson(frame.expo * np.exp(0.2 + 0.4 * frame.x1 - 0.3 * frame.x2
                                                 + effect)).astype(float)
    frame["region"] = frame.firm // 3                           # firms nested in regions
    frame["pair"] = rng.integers(0, 12, size=n)                 # crossed with everything
    frame["fw"] = rng.integers(1, 4, size=n).astype(float)
    frame["aw"] = rng.uniform(0.3, 3.0, size=n)
    return frame


def dummy_poisson(frame, absorb, *, w=None, offset=None, x=("x1", "x2")):
    """Poisson ML on explicit dummies by Newton steps (least-norm solves, so any rank)."""
    n = len(frame)
    slopes = np.column_stack([frame[name].to_numpy(float) for name in x])
    dummies = np.column_stack([pd.get_dummies(frame[name]).to_numpy(float) for name in absorb])
    z = np.column_stack([slopes, dummies])
    y = frame.y.to_numpy(float)
    w = np.ones(n) if w is None else w
    base = np.zeros(n) if offset is None else offset
    eta = np.log((y + np.average(y, weights=w)) / 2) - base
    for _ in range(200):
        mu = np.exp(eta + base)
        working = eta + (y - mu) / mu
        root = np.sqrt(w * mu)
        beta = np.linalg.lstsq(z * root[:, None], working * root, rcond=None)[0]
        change = np.abs(z @ beta - eta).max()
        eta = z @ beta
        if change < 1e-13:
            break
    mu = np.exp(eta + base)
    k = slopes.shape[1]
    information = np.linalg.pinv(z.T @ (z * (w * mu)[:, None]))
    rows = (information @ z.T)[:k]                 # influence of each score on the slopes
    return {"beta": beta[:k], "mu": mu, "rows": rows, "bread": information[:k, :k],
            "rank": np.linalg.matrix_rank(dummies), "score": w * (y - mu), "y": y, "w": w}


def ppml_covariances(fit, n, k_total, groups=None, *, frequency=None):
    """reghdfe's finite-sample factors on the Poisson sandwich of the slopes."""
    rows, score = fit["rows"], fit["score"]
    if frequency is None:
        white = (rows * score ** 2) @ rows.T
    else:                                          # replicated rows: f_i e_i^2
        white = (rows * (score ** 2 / frequency)) @ rows.T
    out = {"nonrobust": fit["bread"], "robust": n / (n - k_total) * white}
    if groups is not None:
        ids = np.unique(groups)
        sums = np.stack([(rows * score)[:, groups == g].sum(axis=1) for g in ids])
        factor = len(ids) / (len(ids) - 1) * (n - 1) / (n - k_total)
        out["cluster"] = factor * sums.T @ sums
    return out


def poisson_total(y, mu, w):
    return float(w @ (special.xlogy(y, mu) - mu - special.gammaln(y + 1)))


@pytest.mark.parametrize("absorb", [["firm"], ["firm", "year"]], ids=["oneway", "twoway"])
def test_ppmlhdfe_matches_the_dummy_variable_poisson(panel, absorb):
    n = len(panel)
    fit = dummy_poisson(panel, absorb)
    k_total = 2 + fit["rank"]
    expected = ppml_covariances(fit, n, k_total, panel.pair.to_numpy())
    y, mu, w = fit["y"], fit["mu"], fit["w"]
    log_likelihood = poisson_total(y, mu, w)
    null = poisson_total(y, np.full(n, y.mean()), w)
    for covariance in ("robust", "cluster", "nonrobust"):
        chosen = {"cluster": "pair"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=absorb, tolerance=1e-12, **chosen)
        assert [c.term for c in result.coefficients] == X          # the constant is absorbed
        check_inference(result, fit["beta"], expected[covariance], rtol=2e-6)
        check_test(result.tests["model"], wald(fit["beta"], expected[covariance], [0, 1]), 2,
                   rtol=1e-5)
        metrics = result.metrics
        assert result.nobs == n and metrics["df_absorbed"] == fit["rank"]
        assert metrics["df_resid"] == n - k_total
        assert metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=1e-10)
        assert metrics["deviance"] == pytest.approx(
            float(2 * (w @ (special.xlogy(y, y / mu) - (y - mu)))), rel=1e-9)
        assert metrics["pseudo_r_squared"] == pytest.approx(1 - log_likelihood / null, rel=1e-9)
        assert metrics["n_separated_dropped"] == 0 and metrics["n_singletons_dropped"] == 0
    assert oe.ppmlhdfe(data=panel, y="y", x=X, absorb=absorb).spec.covariance == "robust"
    # One absorbed dimension is the Poisson regression on its dummies.
    if absorb == ["firm"]:
        dummies = oe.poisson(data=panel, y="y", x=[*X, "firm"], categorical=["firm"])
        assert_allclose(coefs(dummies)[1:3], fit["beta"], rtol=1e-8)
        assert dummies.metrics["log_likelihood"] == pytest.approx(log_likelihood, rel=1e-10)


def test_ppmlhdfe_weights_exposure_and_nested_clusters(panel):
    n = len(panel)
    # Exposure enters as the offset ln(exposure); the null model keeps it.
    fit = dummy_poisson(panel, ["firm", "year"], offset=np.log(panel.expo.to_numpy()))
    k_total = 2 + fit["rank"]
    result = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm", "year"], exposure="expo",
                         tolerance=1e-12)
    check_inference(result, fit["beta"], ppml_covariances(fit, n, k_total)["robust"], rtol=2e-6)
    expo, y = panel.expo.to_numpy(), fit["y"]
    null = poisson_total(y, expo * y.sum() / expo.sum(), np.ones(n))
    assert result.metrics["pseudo_r_squared"] == pytest.approx(
        1 - poisson_total(y, fit["mu"], np.ones(n)) / null, rel=1e-9)
    # aweights are rescaled to sum to N; pweights give the same robust results at any scale.
    scaled = panel.aw.to_numpy() * n / panel.aw.sum()
    fit = dummy_poisson(panel, ["firm", "year"], w=scaled)
    expected = ppml_covariances(fit, n, k_total, panel.pair.to_numpy())
    for kind in ("aweight", "pweight"):
        for covariance in ("robust", "cluster"):
            chosen = {"cluster": "pair"} if covariance == "cluster" else {}
            result = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm", "year"], weights="aw",
                                 weight_type=kind, tolerance=1e-12, **chosen)
            check_inference(result, fit["beta"], expected[covariance], rtol=2e-6)
    conventional = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm", "year"], weights="aw",
                               weight_type="aweight", covariance="nonrobust", tolerance=1e-12)
    check_inference(conventional, fit["beta"], expected["nonrobust"], rtol=2e-6)
    # fweights replicate rows: N = sum f, and every result equals the expanded data set.
    f = panel.fw.to_numpy()
    total = int(f.sum())
    fit = dummy_poisson(panel, ["firm", "year"], w=f)
    expected = ppml_covariances(fit, total, k_total, panel.pair.to_numpy(), frequency=f)
    expanded = panel.loc[panel.index.repeat(panel.fw.astype(int))].reset_index(drop=True)
    for covariance in ("robust", "cluster", "nonrobust"):
        chosen = {"cluster": "pair"} if covariance == "cluster" else {"covariance": covariance}
        result = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm", "year"], weights="fw",
                             weight_type="fweight", tolerance=1e-12, **chosen)
        check_inference(result, fit["beta"], expected[covariance], rtol=2e-6)
        duplicated = oe.ppmlhdfe(data=expanded, y="y", x=X, absorb=["firm", "year"],
                                 tolerance=1e-12, **chosen)
        assert result.nobs == duplicated.nobs == total
        assert_allclose(coefs(result), coefs(duplicated), rtol=1e-9)
        assert_allclose(cov(result), cov(duplicated), rtol=1e-7)
        for name in ("log_likelihood", "deviance", "pseudo_r_squared", "df_resid", "df_absorbed"):
            assert result.metrics[name] == pytest.approx(duplicated.metrics[name], rel=1e-9)
    # Fixed effects nested in the cluster variable cost no degrees of freedom (xtreg, fe /
    # reghdfe): with firm nested in region only the constant is counted, K = k + 1.
    fit = dummy_poisson(panel, ["firm"])
    nested = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm"], cluster="region",
                         tolerance=1e-12)
    expected = ppml_covariances(fit, n, 2 + 1, panel.region.to_numpy())["cluster"]
    check_inference(nested, fit["beta"], expected, rtol=2e-6)
    assert nested.metrics["df_absorbed"] == 1 and nested.metrics["df_resid"] == n - 3
    assert nested.extra["absorbed"][0]["nested"] is True
    # ... while firm and year together: firm nested (free), year counted in full (7 levels).
    fit = dummy_poisson(panel, ["firm", "year"])
    mixed = oe.ppmlhdfe(data=panel, y="y", x=X, absorb=["firm", "year"], cluster="region",
                        tolerance=1e-12)
    assert mixed.metrics["df_absorbed"] == 7
    expected = ppml_covariances(fit, n, 2 + 7, panel.region.to_numpy())["cluster"]
    check_inference(mixed, fit["beta"], expected, rtol=2e-6)


def test_ppmlhdfe_drops_separated_levels_and_singletons_like_the_pruned_sample(panel):
    frame = panel.copy()
    frame.loc[frame.firm == 4, "y"] = 0.0             # a firm that never has a positive outcome
    lonely = pd.DataFrame({"firm": [900, 901], "year": [1, 2], "x1": [0.3, -0.2],
                           "x2": [0.1, 0.4], "expo": [1.0, 1.0], "y": [3.0, 1.0],
                           "region": [300, 301], "pair": [1, 2], "fw": [1.0, 1.0],
                           "aw": [1.0, 1.0]})
    frame = pd.concat([frame, lonely], ignore_index=True)
    pruned = frame[(frame.firm != 4) & (frame.firm < 900)].reset_index(drop=True)
    fit = dummy_poisson(pruned, ["firm", "year"])
    n = len(pruned)
    k_total = 2 + fit["rank"]
    result = oe.ppmlhdfe(data=frame, y="y", x=X, absorb=["firm", "year"], tolerance=1e-12)
    check_inference(result, fit["beta"], ppml_covariances(fit, n, k_total)["robust"], rtol=2e-6)
    assert result.nobs == n
    assert result.metrics["n_separated_dropped"] == int((frame.firm == 4).sum())
    assert result.metrics["n_singletons_dropped"] == 2
    assert result.dropped_rows == len(frame) - n
    assert sum("separated" in w or "singleton" in w for w in result.warnings) == 2
    assert result.sample_positions == np.flatnonzero(
        ((frame.firm != 4) & (frame.firm < 900)).to_numpy()).tolist()
    # Keeping singletons does not change the slopes (their effect fits them exactly) but
    # they count in N and in the absorbed degrees of freedom.
    kept = oe.ppmlhdfe(data=frame, y="y", x=X, absorb=["firm", "year"], drop_singletons=False,
                       tolerance=1e-12)
    assert kept.nobs == n + 2 and kept.metrics["n_singletons_dropped"] == 0
    assert_allclose(coefs(kept), fit["beta"], rtol=1e-8)
    assert kept.metrics["df_absorbed"] == fit["rank"] + 2


# ---- 7. regression tests for conventions that were wrong --------------------------------------


def test_opg_is_a_consistent_estimate_of_the_information():
    """With dispersion phi the OPG covariance is phi^2 (S'S)^-1, not phi (S'S)^-1.

    Under a correctly specified gamma model the OIM, OPG and robust standard errors
    estimate the same thing; phi (S'S)^-1 would be off by 1/sqrt(phi) = 2 here.
    """
    rng = np.random.default_rng(12)
    n = 60_000
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n)})
    mu = np.exp(0.4 + 0.3 * frame.x1 - 0.25 * frame.x2)
    frame["y"] = rng.gamma(4.0, mu / 4.0)                    # phi = 1/4
    fits = {kind: oe.glm(data=frame, y="y", x=X, family="gamma", link="log", covariance=kind)
            for kind in ("nonrobust", "opg", "robust")}
    assert fits["opg"].metrics["scale"] == pytest.approx(0.25, rel=0.03)
    assert_allclose(ses(fits["opg"]), ses(fits["nonrobust"]), rtol=0.03)
    assert_allclose(ses(fits["robust"]), ses(fits["nonrobust"]), rtol=0.03)
    assert "squared dispersion" in fits["opg"].inference["correction"]
    # The scale option moves OIM by phi and OPG by phi^2, and leaves the sandwich alone.
    fixed = {kind: oe.glm(data=frame, y="y", x=X, family="gamma", link="log", covariance=kind,
                          scale=1.0) for kind in ("nonrobust", "opg", "robust")}
    phi = fits["opg"].metrics["scale"]
    assert_allclose(cov(fits["nonrobust"]), cov(fixed["nonrobust"]) * phi, rtol=1e-10)
    assert_allclose(cov(fits["opg"]), cov(fixed["opg"]) * phi ** 2, rtol=1e-10)
    assert_allclose(cov(fits["robust"]), cov(fixed["robust"]), rtol=1e-12)


def test_opg_is_linear_in_the_weights_for_every_weight_type(data):
    """The likelihood is sum w l, so OIM and OPG treat integer iweights like fweights;
    only the sandwich distinguishes replication (sum f s s') from weighting (sum w^2 s s')."""
    for function, kwargs in ((oe.poisson, {"y": "cnt"}), (oe.nbreg, {"y": "over"}),
                             (oe.cloglog, {"y": "bin"}), (oe.betareg, {"y": "beta"}),
                             (oe.glm, {"y": "bin", "family": "binomial", "link": "probit"})):
        fits = {(kind, covariance): function(data=data, x=X, weights="fw", weight_type=kind,
                                             covariance=covariance, **kwargs)
                for kind in ("fweight", "iweight") for covariance in ("nonrobust", "opg",
                                                                      "robust")}
        for covariance in ("nonrobust", "opg"):
            assert_allclose(cov(fits["iweight", covariance]), cov(fits["fweight", covariance]),
                            rtol=1e-9)
        assert np.abs(cov(fits["iweight", "robust"]) / cov(fits["fweight", "robust"])
                      - 1).max() > 0.05
        # aweights are iweights rescaled to sum to N: the OPG scales by sum(w)/N.
        if function not in (oe.cloglog, oe.betareg):
            analytic = function(data=data, x=X, weights="fw", weight_type="aweight",
                                covariance="opg", **kwargs)
            assert_allclose(cov(analytic), cov(fits["iweight", "opg"]) * data.fw.sum()
                            / len(data), rtol=1e-8)


def test_gamma_and_inverse_gaussian_log_likelihoods_ignore_the_scale(data):
    """Stata evaluates both at phi = 1; scale only multiplies the covariance."""
    y = data.pos.to_numpy()
    for family, formula in (
            ("gamma", lambda mu: -(y / mu + np.log(mu))),
            ("inverse_gaussian", lambda mu: -(y - mu) ** 2 / (2 * y * mu ** 2)
             - 0.5 * np.log(2 * np.pi * y ** 3))):
        values = set()
        for scale in (None, "x2", "dev", 1, 3.5):
            result = oe.glm(data=data, y="pos", x=X, family=family, link="log", scale=scale)
            mu = np.exp(np.column_stack([np.ones(len(data)), data.x1, data.x2]) @ coefs(result))
            assert result.metrics["log_likelihood"] == pytest.approx(float(formula(mu).sum()),
                                                                     rel=1e-11)
            assert result.extra["log_likelihood_dispersion"] == 1.0
            values.add(round(result.metrics["log_likelihood"], 8))
        assert len(values) == 1
