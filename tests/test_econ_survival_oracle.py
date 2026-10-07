"""Independent verification oracles for the survival family (verify-and-repair pass).

Everything here is derived independently of the implementation:

* explicit O(n * T) NumPy loops over the failure times for the Cox partial
  likelihood (Breslow, Efron, exact partial by subset enumeration, tvc evaluated
  at each failure time), its efficient score residuals, Schoenfeld residuals,
  the Grambsch-Therneau tests (formulas of Stata [ST] stcox PH-assumption tests),
  Harrell's C over all pairs and the Breslow baseline hazard;
* brute-force ``scipy.optimize`` maximization of those likelihoods with
  numerically differentiated Hessians (statsmodels numdiff) for coefficients and
  covariance matrices;
* statsmodels ``PHReg`` (strata, entry, offset, ties) and ``survdiff``
  (log-rank family with strata and entry) where the identical model exists;
* parametric likelihoods written from Stata's [ST] streg parameterizations with
  ``scipy.special`` (regularized incomplete gamma for the generalized gamma);
* published Stata output: the ``kva`` generator experiment of [ST] stcox and
  [ST] streg (12 observations, reproduced to every printed digit);
* invariances: frequency weights == duplicated rows, row permutation, rescaling.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import special, stats
from scipy.optimize import minimize
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff
from statsmodels.tools.numdiff import approx_fprime, approx_hess

import openecon as oe

Z975 = stats.norm.isf(0.025)

# Stata's kva data ([ST] stcox example 1, [ST] streg example 1): time at risk 896.
KVA = pd.DataFrame({"failtime": [100, 140, 97, 122, 84, 100, 54, 52, 40, 55, 22, 30],
                    "load": [15, 15, 20, 20, 25, 25, 30, 30, 35, 35, 40, 40],
                    "bearings": [0, 1] * 6})


def make_data(seed=11, n=160, grid=3):
    """Ties (times on a grid), censoring, delayed entry, strata, clusters, weights."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    g = rng.integers(0, 3, n)
    eta = x @ [0.6, -0.4] + 0.5 * (g == 1)
    t = np.ceil(rng.exponential(np.exp(-eta)) * grid) / grid
    c = np.ceil(rng.exponential(1.4, n) * grid) / grid
    frame = pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c).astype(float),
                          "x1": x[:, 0], "x2": x[:, 1], "g": g,
                          "s": rng.integers(0, 2, n), "cl": rng.integers(0, 25, n),
                          "f": rng.integers(1, 4, n), "pw": rng.uniform(0.3, 3.0, n),
                          "off": 0.3 * rng.normal(size=n)})
    late = rng.random(n) < 0.25
    frame["t0"] = np.where(late, np.floor(frame.t * rng.uniform(0.0, 0.9, n) * grid) / grid, 0.0)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def params(result):
    return np.array([c.estimate for c in result.coefficients])


def cov(result):
    return np.array(result.covariance_matrix)


# ---- explicit Cox machinery (NumPy loops over failure times) ----------------------------


def failure_times(t, t0, d, s):
    """[(at-risk mask, failing mask)] per distinct failure time and stratum."""
    out = []
    for stratum in np.unique(s):
        member = s == stratum
        for tau in np.unique(t[member & (d > 0)]):
            out.append((member & (t0 < tau) & (t >= tau), member & (t == tau) & (d > 0)))
    return out


def cox_ll(beta, X, t, t0, d, s, w=None, offset=None, ties="breslow"):
    w = np.ones(len(t)) if w is None else w
    eta = X @ beta + (0 if offset is None else offset)
    total = 0.0
    for risk, dead in failure_times(t, t0, d, s):
        if ties == "breslow":
            total += (w[dead] * eta[dead]).sum() - w[dead].sum() * np.log(
                (w[risk] * np.exp(eta[risk])).sum())
        else:
            c = dead.sum()
            s0, d0 = np.exp(eta[risk]).sum(), np.exp(eta[dead]).sum()
            total += eta[dead].sum() - sum(np.log(s0 - r / c * d0) for r in range(c))
    return total


def cox_score_residuals(beta, X, t, t0, d, s, w=None, offset=None, ties="breslow"):
    """Efficient score residuals (Stata [ST] stcox postestimation; Therneau-Grambsch Efron)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    u = np.exp(X @ beta + (0 if offset is None else offset))
    W = np.zeros((n, k))
    for risk, dead in failure_times(t, t0, d, s):
        if ties == "breslow":
            s0 = (w * u)[risk].sum()
            mean = (w * u)[risk] @ X[risk] / s0
            dk = w[dead].sum()
            W[dead] += X[dead] - mean
            W[risk] -= u[risk, None] * (dk / s0) * (X[risk] - mean)
        else:
            c = dead.sum()
            s0, d0 = u[risk].sum(), u[dead].sum()
            s1, d1 = u[risk] @ X[risk], u[dead] @ X[dead]
            means = [(s1 - r / c * d1) / (s0 - r / c * d0) for r in range(c)]
            W[dead] += X[dead] - np.mean(means, axis=0)
            for r in range(c):
                phi = s0 - r / c * d0
                share = np.where(dead[risk], 1 - r / c, 1.0)
                W[risk] -= (u[risk] * share / phi)[:, None] * (X[risk] - means[r])
    return W


def cox_brute(X, t, t0, d, s, w=None, offset=None, ties="breslow", start=None):
    def objective(b):
        return -cox_ll(b, X, t, t0, d, s, w, offset, ties)
    b0 = np.zeros(X.shape[1]) if start is None else start
    fitted = minimize(objective, b0, method="BFGS", options={"gtol": 1e-10})
    hessian = approx_hess(fitted.x, lambda b: cox_ll(b, X, t, t0, d, s, w, offset, ties))
    return fitted.x, -fitted.fun, hessian


def arrays(frame, x=("x1", "x2")):
    return (frame[list(x)].to_numpy(float), frame.t.to_numpy(float), frame.t0.to_numpy(float),
            frame.d.to_numpy(float))


# ---- stcox ------------------------------------------------------------------------------


def test_stcox_reproduces_stata_kva_output():
    """[ST] stcox example 1: LR chi2(2) = 23.39, ll = -8.577853, b and se as printed."""
    result = oe.stcox(data=KVA, time="failtime", x=["load", "bearings"])
    assert_allclose(params(result), [0.4229578, -2.754461], atol=6e-7)
    assert_allclose(np.sqrt(np.diag(cov(result))), [0.1433485, 1.173115], atol=6e-7)
    assert result.metrics["log_likelihood"] == pytest.approx(-8.577853, abs=6e-7)
    assert result.tests["model"]["statistic"] == pytest.approx(23.39, abs=0.005)
    assert result.metrics["time_at_risk"] == 896 and result.metrics["n_failures"] == 12
    assert result.metrics["n_subjects"] == 12 and result.nobs == 12
    assert result.extra["hazard_ratios"]["load"]["hazard_ratio"] == pytest.approx(1.52647,
                                                                                    abs=6e-6)
    assert result.extra["hazard_ratios"]["bearings"]["std_error"] == pytest.approx(0.0746609,
                                                                                     abs=6e-7)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_stcox_matches_brute_force_with_strata_entry_offset(data, ties):
    X, t, t0, d = arrays(data)
    s, off = data.s.to_numpy(), data.off.to_numpy()
    b, ll, hessian = cox_brute(X, t, t0, d, s, offset=off, ties=ties)
    result = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s", offset="off",
                      x=["x1", "x2"], ties=ties)
    assert_allclose(params(result), b, atol=1e-6)
    assert_allclose(cov(result), np.linalg.inv(-hessian), rtol=2e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-10)
    null = cox_ll(np.zeros(2), X, t, t0, d, s, offset=off, ties=ties)
    assert result.metrics["log_likelihood_null"] == pytest.approx(null, rel=1e-12)
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (ll - null), rel=1e-7)
    assert result.tests["model"]["df"] == 2
    # Score test of b = 0: U(0)' I(0)^-1 U(0) with numerical derivatives of the loop.
    grad = approx_fprime(np.zeros(2), lambda v: cox_ll(v, X, t, t0, d, s, None, off, ties),
                         centered=True)
    info = -approx_hess(np.zeros(2), lambda v: cox_ll(v, X, t, t0, d, s, None, off, ties))
    assert result.tests["score"]["statistic"] == pytest.approx(
        grad @ np.linalg.solve(info, grad), rel=1e-5)
    # z inference and intervals
    se = np.sqrt(np.diag(cov(result)))
    for coef, beta, sd in zip(result.coefficients, b, se, strict=True):
        assert coef.p_value == pytest.approx(2 * stats.norm.sf(abs(beta / sd)), rel=1e-5)
        assert coef.ci_low == pytest.approx(coef.estimate - Z975 * coef.std_error, rel=1e-9)
    n = len(data)
    assert result.metrics["aic"] == pytest.approx(-2 * ll + 4, rel=1e-9)
    assert result.metrics["bic"] == pytest.approx(-2 * ll + 2 * math.log(n), rel=1e-9)
    assert result.metrics["time_at_risk"] == pytest.approx((data.t - data.t0).sum())
    assert result.metrics["n_failures"] == data.d.sum()


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_stcox_matches_statsmodels_phreg(data, ties):
    # PHReg counts entry == tau as at risk; Stata does not: shift entries off the grid.
    frame = data.assign(t0=np.where(data.t0 > 0, data.t0 + 1e-7, 0.0))
    X, t, t0, d = arrays(frame)
    reference = PHReg(t, X, status=d, entry=t0, strata=frame.s.to_numpy(),
                      offset=frame.off.to_numpy(), ties=ties).fit()
    result = oe.stcox(data=frame, time="t", failure="d", entry="t0", strata="s", offset="off",
                      x=["x1", "x2"], ties=ties)
    assert_allclose(params(result), reference.params, atol=1e-7)
    assert_allclose(cov(result), reference.cov_params(), rtol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(reference.llf, rel=1e-10)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_stcox_robust_and_cluster_sandwich_from_explicit_residuals(data, ties):
    X, t, t0, d = arrays(data)
    s = data.s.to_numpy()
    result = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s",
                      x=["x1", "x2"], ties=ties, covariance="robust")
    b = params(result)
    hessian = approx_hess(b, lambda v: cox_ll(v, X, t, t0, d, s, ties=ties))
    bread = np.linalg.inv(-hessian)
    W = cox_score_residuals(b, X, t, t0, d, s, ties=ties)
    assert_allclose(W.sum(0), 0, atol=1e-8)
    n = len(data)
    expected = n / (n - 1) * bread @ W.T @ W @ bread
    assert_allclose(cov(result), expected, rtol=2e-5, atol=1e-6 * expected.max())
    assert result.inference["small_sample_correction"] == pytest.approx(n / (n - 1))
    assert result.tests["model"]["label"].startswith("Wald")
    clustered = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s",
                         x=["x1", "x2"], ties=ties, covariance="cluster", cluster="cl")
    G = data.cl.nunique()
    sums = np.zeros((G, 2))
    np.add.at(sums, pd.factorize(data.cl)[0], W)
    expected = G / (G - 1) * bread @ sums.T @ sums @ bread
    assert_allclose(cov(clustered), expected, rtol=2e-5, atol=1e-6 * expected.max())


def test_stcox_weights_follow_stata(data):
    X, t, t0, d = arrays(data)
    s = data.s.to_numpy()
    # fweights == duplicated rows (coefficients, covariance, ll, counts, N for BIC).
    weighted = oe.stcox(data=data, time="t", failure="d", entry="t0", x=["x1", "x2"],
                        weights="f", weight_type="fweight")
    long = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    plain = oe.stcox(data=long, time="t", failure="d", entry="t0", x=["x1", "x2"])
    assert_allclose(params(weighted), params(plain), atol=1e-9)
    assert_allclose(cov(weighted), cov(plain), rtol=1e-8)
    for name in ("log_likelihood", "aic", "bic", "n_failures", "time_at_risk", "n_subjects"):
        assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-9)
    assert weighted.nobs == len(long)
    robust_f = oe.stcox(data=data, time="t", failure="d", entry="t0", x=["x1", "x2"],
                        weights="f", weight_type="fweight", covariance="robust")
    robust_long = oe.stcox(data=long, time="t", failure="d", entry="t0", x=["x1", "x2"],
                           covariance="robust")
    assert_allclose(cov(robust_f), cov(robust_long), rtol=1e-8)
    # iweights: the weighted Breslow likelihood of [ST] stcox Methods and formulas.
    pw = data.pw.to_numpy()
    b, ll, hessian = cox_brute(X, t, t0, d, s, w=pw)
    iw = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s", x=["x1", "x2"],
                  weights="pw", weight_type="iweight")
    assert_allclose(params(iw), b, atol=1e-6)
    assert_allclose(cov(iw), np.linalg.inv(-hessian), rtol=2e-5)
    assert iw.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-10)
    # pweights: same point estimates, robust sandwich with weighted residuals, invariant
    # to the scale of the weights.
    pwr = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s", x=["x1", "x2"],
                   weights="pw", weight_type="pweight")
    assert pwr.inference["covariance"] == "robust"
    assert_allclose(params(pwr), b, atol=1e-6)
    bread = np.linalg.inv(-approx_hess(params(pwr), lambda v: cox_ll(v, X, t, t0, d, s, pw)))
    W = cox_score_residuals(params(pwr), X, t, t0, d, s, w=pw) * pw[:, None]
    n = len(data)
    assert_allclose(cov(pwr), n / (n - 1) * bread @ W.T @ W @ bread, rtol=2e-5)
    scaled = oe.stcox(data=data.assign(pw=data.pw * 1e6), time="t", failure="d", entry="t0",
                      strata="s", x=["x1", "x2"], weights="pw", weight_type="pweight")
    assert_allclose(cov(scaled), cov(pwr), rtol=1e-7)


def test_stcox_invariances(data):
    base = oe.stcox(data=data, time="t", failure="d", entry="t0", x=["x1", "x2"], ties="efron")
    order = np.random.default_rng(5).permutation(len(data))
    shuffled = oe.stcox(data=data.iloc[order], time="t", failure="d", entry="t0",
                        x=["x1", "x2"], ties="efron")
    assert_allclose(params(shuffled), params(base), atol=1e-10)
    assert_allclose(cov(shuffled), cov(base), rtol=1e-9)
    # rescaling and shifting a covariate: b -> b / c, se -> se / c (no constant to absorb).
    # (1e4 + 1e-3 x keeps ~9 digits of x in float64; the fit must not lose more)
    moved = data.assign(x1=1e4 + 1e-3 * data.x1)
    scaled = oe.stcox(data=moved, time="t", failure="d", entry="t0", x=["x1", "x2"],
                      ties="efron")
    assert params(scaled)[0] * 1e-3 == pytest.approx(params(base)[0], rel=1e-7)
    assert math.sqrt(cov(scaled)[0, 0]) * 1e-3 == pytest.approx(math.sqrt(cov(base)[0, 0]),
                                                                rel=1e-6)
    # a huge coefficient (tiny covariate scale) must not overflow the hazard ratios
    tiny = oe.stcox(data=data.assign(x1=1e-4 * data.x1), time="t", failure="d", entry="t0",
                    x=["x1", "x2"], ties="efron")
    assert tiny.extra["hazard_ratios"]["x1"]["hazard_ratio"] is None   # inf -> missing
    assert params(tiny)[0] * 1e-4 == pytest.approx(params(base)[0], rel=1e-7)
    # time rescaling leaves the partial likelihood unchanged
    stretched = oe.stcox(data=data.assign(t=data.t * 1e5, t0=data.t0 * 1e5), time="t",
                         failure="d", entry="t0", x=["x1", "x2"], ties="efron")
    assert_allclose(params(stretched), params(base), atol=1e-10)


def test_stcox_exact_partial_likelihood_by_subset_enumeration():
    rng = np.random.default_rng(4)
    n = 16
    X = rng.normal(size=(n, 2))
    t = rng.integers(1, 5, n).astype(float)
    d = (rng.random(n) < 0.8).astype(float)
    frame = pd.DataFrame({"t": t, "d": d, "x1": X[:, 0], "x2": X[:, 1]})

    def loglik(b):
        eta = X @ b
        total = 0.0
        for risk, dead in failure_times(t, np.zeros(n), d, np.zeros(n)):
            members = np.flatnonzero(risk)
            c = int(dead.sum())
            sums = [np.exp(eta[list(subset)].sum())
                    for subset in itertools.combinations(members, c)]
            total += eta[dead].sum() - np.log(np.sum(sums))
        return total

    fitted = minimize(lambda b: -loglik(b), np.zeros(2), method="BFGS",
                      options={"gtol": 1e-10})
    result = oe.stcox(data=frame, time="t", failure="d", x=["x1", "x2"], ties="exactp")
    assert_allclose(params(result), fitted.x, atol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-10)
    assert_allclose(cov(result), np.linalg.inv(-approx_hess(fitted.x, loglik)), rtol=2e-5)
    assert result.metrics["log_likelihood_null"] == pytest.approx(loglik(np.zeros(2)),
                                                                  rel=1e-12)


@pytest.mark.parametrize("texp", ["identity", "log"])
def test_stcox_tvc_matches_likelihood_evaluated_at_each_failure_time(data, texp):
    X, t, t0, d = arrays(data)
    s = data.s.to_numpy()
    z = data.x2.to_numpy()
    g = (lambda v: v) if texp == "identity" else np.log

    def loglik(theta):
        total = 0.0
        for risk, dead in failure_times(t, t0, d, s):
            tau = t[dead][0]
            eta = X[:, 0] * theta[0] + X[:, 1] * theta[1] + z * g(tau) * theta[2]
            total += eta[dead].sum() - dead.sum() * np.log(np.exp(eta[risk]).sum())
        return total

    fitted = minimize(lambda v: -loglik(v), np.zeros(3), method="BFGS", options={"gtol": 1e-10})
    result = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s", x=["x1", "x2"],
                      tvc=["x2"], texp=texp)
    assert [c.term for c in result.coefficients] == ["x1", "x2", "tvc:x2"]
    assert_allclose(params(result), fitted.x, atol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-10)
    assert_allclose(cov(result), np.linalg.inv(-approx_hess(fitted.x, loglik)), rtol=2e-5)
    assert result.nobs == len(data)             # Stata: Number of obs = original records


def schoenfeld(beta, X, t, t0, d, s, ties="breslow"):
    """Schoenfeld residuals of the failing records (Stata's formulas) and their times."""
    u = np.exp(X @ beta)
    rows, times = [], []
    for risk, dead in failure_times(t, t0, d, s):
        if ties == "breslow":
            mean = u[risk] @ X[risk] / u[risk].sum()
        else:
            c = dead.sum()
            s0, d0, s1, d1 = u[risk].sum(), u[dead].sum(), u[risk] @ X[risk], u[dead] @ X[dead]
            mean = np.mean([(s1 - r / c * d1) / (s0 - r / c * d0) for r in range(c)], axis=0)
        for i in np.flatnonzero(dead):
            rows.append(X[i] - mean)
            times.append(t[i])
    return np.array(rows), np.array(times)


@pytest.mark.parametrize("transform,ties", [("identity", "breslow"), ("log", "efron"),
                                            ("km", "breslow"), ("rank", "efron")])
def test_stcox_ph_tests_follow_stata_formulas(data, transform, ties):
    X, t, t0, d = arrays(data)
    zero = np.zeros(len(t))
    result = oe.stcox(data=data, time="t", failure="d", x=["x1", "x2"], ties=ties,
                      phtest=transform)
    b, V = params(result), cov(result)
    r, times = schoenfeld(b, X, t, zero, d, zero, ties)
    if transform == "identity":
        g = times
    elif transform == "log":
        g = np.log(times)
    elif transform == "rank":
        g = stats.rankdata(times)
    else:
        km_times = np.unique(t[d > 0])
        surv = np.cumprod([1 - d[(t == tau)].sum() / (t >= tau).sum() for tau in km_times])
        g = 1 - surv[np.searchsorted(km_times, times)]
    nd = len(times)
    scaled = b + nd * r @ V                       # r*_i = b + d Var(b) r_i
    gc = g - g.mean()
    for p, term in enumerate(["x1", "x2"]):
        chi2 = (gc @ scaled[:, p]) ** 2 / (nd * V[p, p] * (gc @ gc))
        entry = result.tests[f"ph_{term}"]
        assert entry["statistic"] == pytest.approx(chi2, rel=1e-6)
        assert entry["p_value"] == pytest.approx(stats.chi2.sf(chi2, 1), rel=1e-5)
        assert entry["rho"] == pytest.approx(np.corrcoef(scaled[:, p], g)[0, 1], rel=1e-6)
    u = gc @ r
    glob = u @ (nd * V / (gc @ gc)) @ u
    assert result.tests["ph_global"]["statistic"] == pytest.approx(glob, rel=1e-6)
    assert result.tests["ph_global"]["df"] == 2


def test_stcox_harrell_c_over_all_pairs(data):
    X, t, _, d = arrays(data)
    s = data.s.to_numpy()
    result = oe.stcox(data=data, time="t", failure="d", strata="s", x=["x1", "x2"])
    eta = X @ params(result)
    usable = agree = tied = 0
    n = len(t)
    for i in range(n):
        for j in range(n):
            if s[i] != s[j] or i == j or d[i] == 0:
                continue
            # i fails first: t_i < t_j, or equal times and j censored
            if t[i] < t[j] or (t[i] == t[j] and d[j] == 0):
                usable += 1
                if eta[i] > eta[j]:
                    agree += 1
                elif eta[i] == eta[j]:
                    tied += 1
    detail = result.extra["concordance"]
    assert detail["pairs"] == usable and detail["concordant"] == agree
    assert detail["tied_predictions"] == tied
    assert result.metrics["concordance"] == pytest.approx((agree + tied / 2) / usable)
    assert detail["somers_d"] == pytest.approx(2 * (agree + tied / 2) / usable - 1)


def test_stcox_baseline_functions_and_stcurve(data):
    X, t, t0, d = arrays(data)
    zero = np.zeros(len(t))
    result = oe.stcox(data=data, time="t", failure="d", entry="t0", x=["x1", "x2"])
    b = params(result)
    u = np.exp(X @ b)
    hazard, times = [], []
    total = 0.0
    for risk, dead in failure_times(t, t0, d, zero):
        total += dead.sum() / u[risk].sum()
        hazard.append(total)
        times.append(t[dead][0])
    base = result.extra["baseline"]
    assert_allclose(base["time"], times)
    assert_allclose(base["cumulative_hazard"], hazard, rtol=1e-10)
    curve = oe.stcurve(result, data=data, at={"x1": 1.0, "x2": -0.5})
    expected = np.array(hazard) * np.exp(b @ [1.0, -0.5])
    assert_allclose(curve["cumulative_hazard"], expected, rtol=1e-10)
    assert_allclose(curve["survivor"], np.exp(-expected), rtol=1e-10)
    at_means = oe.stcurve(result)
    m = X.mean(0)
    assert_allclose(at_means["cumulative_hazard"], np.array(hazard) * np.exp(b @ m), rtol=1e-10)


def test_stcox_multiple_records_with_id(data):
    """Episode-split records give the single-record fit; robust clusters on id."""
    single = data[["t", "d", "x1", "x2"]].assign(id=np.arange(len(data)))
    cut = single.t / 2
    first = single.assign(t0=0.0, t=cut, d=0.0)
    second = single.assign(t0=cut)
    split = pd.concat([first, second]).sort_values("id").reset_index(drop=True)
    a = oe.stcox(data=single.assign(t0=0.0), time="t", failure="d", x=["x1", "x2"],
                 covariance="robust")
    b = oe.stcox(data=split, time="t", failure="d", entry="t0", id="id", x=["x1", "x2"],
                 covariance="robust")
    assert_allclose(params(b), params(a), atol=1e-9)
    G = len(single)
    # robust with id is cluster(id) with G/(G-1) = N/(N-1) of the single-record fit
    assert_allclose(cov(b), cov(a), rtol=1e-8)
    assert b.inference["cluster_count"] == G
    assert b.metrics["n_subjects"] == G and b.nobs == 2 * G
    assert b.metrics["time_at_risk"] == pytest.approx(single.t.sum())


# ---- streg -------------------------------------------------------------------------------


def test_streg_reproduces_stata_kva_output():
    """[ST] streg examples 1-3 (Weibull PH and AFT on the kva data)."""
    ph = oe.streg(data=KVA, time="failtime", x=["load", "bearings"], distribution="weibull")
    b = {c.term: (c.estimate, c.std_error) for c in ph.coefficients}
    assert b["load"] == pytest.approx((0.4695753, 0.1177884), abs=6e-7)
    assert b["bearings"] == pytest.approx((-1.667069, 0.6949745), abs=6e-7)
    assert b["Intercept"] == pytest.approx((-45.13191, 10.60663), abs=6e-5)
    assert b["/ln_p"] == pytest.approx((2.051552, 0.2317074), abs=6e-7)
    assert ph.metrics["log_likelihood"] == pytest.approx(5.6934189, abs=6e-8)
    assert ph.extra["null_log_likelihood"] == pytest.approx(-9.4408286, abs=6e-8)
    assert ph.tests["model"]["statistic"] == pytest.approx(30.27, abs=0.005)
    assert ph.extra["p"]["estimate"] == pytest.approx(7.779969, abs=6e-6)
    assert ph.extra["p"]["std_error"] == pytest.approx(1.802677, abs=6e-6)
    assert (ph.extra["p"]["ci_low"], ph.extra["p"]["ci_high"]) == pytest.approx(
        (4.940241, 12.25202), abs=6e-6)
    assert ph.extra["1/p"]["estimate"] == pytest.approx(0.1285352, abs=6e-7)
    assert ph.extra["1/p"]["std_error"] == pytest.approx(0.0297826, abs=6e-7)
    assert ph.extra["hazard_ratios"]["load"]["ratio"] == pytest.approx(1.599315, abs=6e-6)
    assert ph.extra["hazard_ratios"]["load"]["std_error"] == pytest.approx(0.1883807, abs=6e-7)
    aft = oe.streg(data=KVA, time="failtime", x=["load", "bearings"], distribution="weibull",
                   metric="aft")
    b = {c.term: (c.estimate, c.std_error) for c in aft.coefficients}
    assert b["load"] == pytest.approx((-0.060357, 0.0062214), abs=6e-7)
    assert b["bearings"] == pytest.approx((0.2142771, 0.0746451), abs=6e-7)
    assert b["Intercept"] == pytest.approx((5.80104, 0.1752301), abs=6e-7)
    assert aft.metrics["log_likelihood"] == pytest.approx(5.6934189, abs=6e-8)


def stata_streg_ll(theta, dist, metric, t, t0, d, X, Z, w=None):
    """Per-record log likelihood on Stata's scale (log-time density) from [ST] streg."""
    k, q = X.shape[1], (0 if Z is None else Z.shape[1])
    mu = X @ theta[:k]
    a = None if Z is None else Z @ theta[k:k + q]

    def log_fs(time):
        lt = np.log(time)
        if dist in ("exponential", "weibull"):
            p = 1.0 if a is None else np.exp(a)
            if metric == "ph":
                lam_t = np.exp(mu) * time ** p
            else:
                lam_t = (np.exp(-mu) * time) ** p
            return np.log(p) + np.log(lam_t) - lam_t, -lam_t   # ln(t f(t)), ln S
        if dist == "gompertz":
            big = np.exp(mu) * np.expm1(a * time) / a
            return mu + a * time + lt - big, -big
        sigma = np.exp(a)
        z = (lt - mu) / sigma
        if dist == "lognormal":
            return stats.norm.logpdf(z) - a, stats.norm.logsf(z)
        if dist == "loglogistic":
            return z - a - 2 * np.logaddexp(0, z), -np.logaddexp(0, z)
        kappa = theta[-1]
        if abs(kappa) < 1e-12:
            return stats.norm.logpdf(z) - a, stats.norm.logsf(z)
        gam = kappa ** -2
        zz = np.sign(kappa) * z
        u = gam * np.exp(abs(kappa) * zz)
        log_f = (gam * np.log(gam) - a - 0.5 * np.log(gam) - special.gammaln(gam)
                 + zz * np.sqrt(gam) - u)
        tail = special.gammaincc(gam, u) if kappa > 0 else special.gammainc(gam, u)
        return log_f, np.log(tail)

    log_f, log_s = log_fs(t)
    value = d * log_f + (1 - d) * log_s
    late = t0 > 0
    if late.any():
        keep_mu, keep_a = mu, a
        mu = mu[late]
        a = None if a is None else a[late]
        value[late] -= log_fs(t0[late])[1]
        mu, a = keep_mu, keep_a
    return value if w is None else value * w


CASES = [("exponential", "ph"), ("exponential", "aft"), ("weibull", "ph"), ("weibull", "aft"),
         ("gompertz", "ph"), ("lognormal", "aft"), ("loglogistic", "aft"), ("ggamma", "aft")]


def streg_data(seed=2, n=300):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.integers(0, 2, n),
                          "cl": rng.integers(0, 30, n), "f": rng.integers(1, 4, n)})
    t = rng.weibull(1.3, n) * np.exp(0.5 * frame.x - 0.3 * frame.z) * 3
    c = rng.exponential(5.0, n)
    frame["t"] = np.minimum(t, c) + 0.01
    frame["d"] = (t <= c).astype(float)
    frame["t0"] = np.where(rng.random(n) < 0.3, frame.t * rng.uniform(0.05, 0.9, n), 0.0)
    return frame


@pytest.fixture(scope="module")
def sdata():
    return streg_data()


@pytest.mark.parametrize("dist,metric", CASES)
def test_streg_matches_brute_force_on_stata_scale(sdata, dist, metric):
    t, t0, d = (sdata[c].to_numpy(float) for c in ("t", "t0", "d"))
    X = np.column_stack([np.ones(len(t)), sdata.x, sdata.z])
    Z = None if dist == "exponential" else np.ones((len(t), 1))
    result = oe.streg(data=sdata, time="t", failure="d", entry="t0", x=["x", "z"],
                      distribution=dist, metric=metric)
    theta = params(result)

    def ll(v):
        return stata_streg_ll(v, dist, metric, t, t0, d, X, Z).sum()

    fitted = minimize(lambda v: -ll(v), theta + 0.02, method="BFGS", options={"gtol": 1e-9})
    assert_allclose(theta, fitted.x, atol=3e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(ll(theta), rel=1e-10, abs=1e-9)
    assert -fitted.fun <= ll(theta) + 1e-8
    hessian = approx_hess(theta, ll)
    assert_allclose(cov(result), np.linalg.inv(-hessian), rtol=2e-4, atol=1e-9)
    size = len(theta)
    assert result.metrics["aic"] == pytest.approx(-2 * ll(theta) + 2 * size, rel=1e-9)
    # LR test against the constant-only main equation (ancillary as specified)
    tested = [1, 2]
    free = [i for i in range(size) if i not in tested]

    def null_ll(v):
        full = np.zeros(size)
        full[free] = v
        return ll(full)

    null = minimize(lambda v: -null_ll(v), theta[free], method="BFGS",
                    options={"gtol": 1e-9})
    assert result.extra["null_log_likelihood"] == pytest.approx(-null.fun, rel=1e-7, abs=1e-7)
    assert result.tests["model"]["df"] == 2
    assert result.tests["model"]["statistic"] == pytest.approx(
        2 * (ll(theta) + null.fun), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("kind", ["opg", "robust", "cluster"])
def test_streg_sandwich_covariances_from_numerical_scores(sdata, kind):
    t, t0, d = (sdata[c].to_numpy(float) for c in ("t", "t0", "d"))
    X = np.column_stack([np.ones(len(t)), sdata.x, sdata.z])
    Z = np.column_stack([np.ones(len(t)), sdata.z])
    options = {"cluster": "cl"} if kind == "cluster" else {}
    result = oe.streg(data=sdata, time="t", failure="d", entry="t0", x=["x", "z"],
                      ancillary=["z"], distribution="loglogistic", covariance=kind, **options)
    theta = params(result)

    def per_obs(v):
        return stata_streg_ll(v, "loglogistic", "aft", t, t0, d, X, Z)

    scores = approx_fprime(theta, per_obs, centered=True)
    bread = np.linalg.inv(-approx_hess(theta, lambda v: per_obs(v).sum()))
    n = len(t)
    if kind == "opg":
        expected = np.linalg.inv(scores.T @ scores)
    elif kind == "robust":
        expected = n / (n - 1) * bread @ scores.T @ scores @ bread
    else:
        codes = pd.factorize(sdata.cl)[0]
        sums = np.zeros((codes.max() + 1, len(theta)))
        np.add.at(sums, codes, scores)
        G = len(sums)
        expected = G / (G - 1) * bread @ sums.T @ sums @ bread
    assert_allclose(cov(result), expected, rtol=1e-4)
    assert result.tests["model"]["label"].startswith("Wald") if kind != "opg" else True


def test_streg_strata_model_test_includes_stratum_indicators(sdata):
    """[ST] streg example 8: strata(drug) with 3 levels reports LR chi2(3) for one covariate."""
    frame = sdata.assign(g=np.random.default_rng(1).integers(0, 3, len(sdata)))
    result = oe.streg(data=frame, time="t", failure="d", x=["x"], strata="g",
                      distribution="weibull")
    terms = [c.term for c in result.coefficients]
    assert terms == ["Intercept", "x", "g[1]", "g[2]", "ln_p:Intercept", "ln_p:g[1]",
                     "ln_p:g[2]"]
    assert result.tests["model"]["df"] == 3
    t, t0, d = (frame[c].to_numpy(float) for c in ("t", "t0", "d"))
    G = np.column_stack([(frame.g == 1), (frame.g == 2)]).astype(float)
    X = np.column_stack([np.ones(len(t)), frame.x, G])
    Z = np.column_stack([np.ones(len(t)), G])
    zero = np.zeros(len(t))

    def ll(v, Xm):
        return stata_streg_ll(v, "weibull", "ph", t, zero, d, Xm, Z).sum()

    full = minimize(lambda v: -ll(v, X), params(result), method="BFGS", options={"gtol": 1e-9})
    null = minimize(lambda v: -ll(v, X[:, :1]), params(result)[[0, 4, 5, 6]], method="BFGS",
                    options={"gtol": 1e-9})
    assert result.metrics["log_likelihood"] == pytest.approx(-full.fun, rel=1e-9)
    assert result.tests["model"]["statistic"] == pytest.approx(2 * (null.fun - full.fun),
                                                               rel=1e-6)


def test_streg_frequency_weights_equal_duplicated_rows(sdata):
    long = sdata.loc[sdata.index.repeat(sdata.f)].reset_index(drop=True)
    for dist in ("weibull", "lognormal"):
        weighted = oe.streg(data=sdata, time="t", failure="d", entry="t0", x=["x", "z"],
                            distribution=dist, weights="f", weight_type="fweight")
        plain = oe.streg(data=long, time="t", failure="d", entry="t0", x=["x", "z"],
                         distribution=dist)
        assert_allclose(params(weighted), params(plain), atol=1e-8)
        assert_allclose(cov(weighted), cov(plain), rtol=1e-7)
        for name in ("log_likelihood", "aic", "bic", "n_failures", "time_at_risk",
                     "n_subjects"):
            assert weighted.metrics[name] == pytest.approx(plain.metrics[name], rel=1e-9)
        assert weighted.tests["model"]["statistic"] == pytest.approx(
            plain.tests["model"]["statistic"], rel=1e-7)


# ---- sts ---------------------------------------------------------------------------------


def km_loop(t, t0, d, w):
    rows = []
    surv, green, logs, cum, cumvar = 1.0, 0.0, 0.0, 0.0, 0.0
    for tau in np.unique(t):
        n = w[(t0 < tau) & (t >= tau)].sum()
        e = w[(t == tau) & (d > 0)].sum()
        c = w[(t == tau) & (d == 0)].sum()
        if e > 0:
            surv *= 1 - e / n
            if n > e:
                green += e / (n * (n - e))
                logs += np.log((n - e) / n)
            cum += e / n
            cumvar += e / n ** 2
        rows.append((tau, n, e, c, surv, green, logs, cum, cumvar))
    return pd.DataFrame(rows, columns=["time", "n", "e", "c", "S", "green", "logs", "H", "VH"])


@pytest.mark.parametrize("conftype", ["loglog", "log", "plain"])
def test_sts_survival_table_from_explicit_loop(data, conftype):
    out = oe.sts(data, "t", failure="d", entry="t0", weights="f", conftype=conftype)
    table = out["survival"]
    ref = km_loop(*(data[c].to_numpy(float) for c in ("t", "t0", "d", "f")))
    assert_allclose(table["time"], ref.time)
    assert_allclose(table["n_risk"], ref.n)
    assert_allclose(table["n_event"], ref.e)
    assert_allclose(table["n_censored"], ref.c)
    assert_allclose(table["survivor"], ref.S, rtol=1e-12)
    assert_allclose(table["std_error"], ref.S * np.sqrt(ref.green), rtol=1e-10)
    inside = (ref.S < 1) & (ref.S > 0)
    if conftype == "loglog":
        sigma = np.sqrt(ref.green) / np.abs(ref.logs)
        low, high = ref.S ** np.exp(Z975 * sigma), ref.S ** np.exp(-Z975 * sigma)
    elif conftype == "log":
        low = ref.S * np.exp(-Z975 * np.sqrt(ref.green))
        high = np.minimum(ref.S * np.exp(Z975 * np.sqrt(ref.green)), 1)
    else:
        se = ref.S * np.sqrt(ref.green)
        low, high = np.clip(ref.S - Z975 * se, 0, 1), np.clip(ref.S + Z975 * se, 0, 1)
    assert_allclose(table["ci_low"][inside], low[inside], rtol=1e-10)
    assert_allclose(table["ci_high"][inside], high[inside], rtol=1e-10)
    assert_allclose(table["cumulative_hazard"], ref.H, rtol=1e-12)
    assert_allclose(table["cumulative_hazard_std_error"], np.sqrt(ref.VH), rtol=1e-10)
    positive = ref.H > 0
    phi = np.sqrt(ref.VH[positive]) / ref.H[positive]
    assert_allclose(table["cumulative_hazard_ci_low"][positive], ref.H[positive]
                    * np.exp(-Z975 * phi), rtol=1e-10)


def test_sts_summary_percentiles_and_restricted_mean(data):
    frame = data.assign(t0=0.0)
    out = oe.sts(frame, "t", failure="d", by="s")
    for row, (_, member) in zip(out["summary"].itertuples(), frame.groupby("s"), strict=True):
        t, d = member.t.to_numpy(), member.d.to_numpy()
        ref = km_loop(t, np.zeros(len(t)), d, np.ones(len(t)))
        fail = ref[ref.e > 0]
        assert row.median == pytest.approx(fail.time[fail.S <= 0.5].iloc[0])
        assert row.q25 == pytest.approx(fail.time[fail.S <= 0.75].iloc[0])
        sigma = np.sqrt(fail.green) / np.abs(fail.logs)
        upper = np.where(fail.S < 1, fail.S ** np.exp(-Z975 * sigma), 1.0)
        lower = np.where(fail.S < 1, fail.S ** np.exp(Z975 * sigma), 1.0)
        hits_low = fail.time[lower <= 0.5]
        hits_high = fail.time[upper <= 0.5]
        assert row.median_ci_low == pytest.approx(hits_low.iloc[0])
        if len(hits_high):
            assert row.median_ci_high == pytest.approx(hits_high.iloc[0])
        # restricted mean and its Klein-Moeschberger SE ([ST] stci)
        edges = np.r_[0.0, fail.time, t.max()]
        heights = np.r_[1.0, fail.S]
        area = np.sum(heights * np.diff(edges))
        assert row.restricted_mean == pytest.approx(area, rel=1e-12)
        tails = [np.sum(np.r_[fail.S.iloc[j:]] * np.diff(np.r_[fail.time.iloc[j:], t.max()]))
                 for j in range(len(fail))]
        var = np.sum(np.square(tails) * fail.e / (fail.n * (fail.n - fail.e)).where(
            fail.n > fail.e, np.inf))
        assert row.restricted_mean_std_error == pytest.approx(np.sqrt(var), rel=1e-10)
        assert row.events == d.sum() and row.n == len(t)


@pytest.mark.parametrize("test,weight_type,extra", [("logrank", None, {}), ("wilcoxon", "gb", {}),
                                                    ("tware", "tw", {}),
                                                    ("fh", "fh", {"fh_p": 1.0})])
def test_sts_rank_tests_match_statsmodels_survdiff(data, test, weight_type, extra):
    t, t0, d = (data[c].to_numpy(float) for c in ("t", "t0", "d"))
    g = data.g.to_numpy()
    # survdiff counts entry == tau at risk; Stata does not: shift entries off the grid.
    shifted = np.where(t0 > 0, t0 + 1e-7, 0.0)
    # statsmodels' survdiff fails with entry and strata together, and its FH weights are
    # NaN with delayed entry: test them separately (the explicit loop below covers FH/Peto
    # with delayed entry).
    cases = (("s", None),) if test == "fh" else ((None, shifted), ("s", None))
    for strata, entry in cases:
        options = {"fh_p": extra.get("fh_p", 0.0)} if test == "fh" else {}
        out = oe.sts(data.assign(t0=shifted if entry is not None else 0.0), "t", failure="d",
                     entry="t0", by="g", test=test, strata=strata, **options)
        ref_options = {"fh_p": extra["fh_p"]} if test == "fh" else {}
        chi2, p = survdiff(t, d, g, weight_type=weight_type, entry=entry,
                           strata=None if strata is None else data[strata].to_numpy(),
                           **ref_options)
        assert out.attrs["statistic"] == pytest.approx(chi2, rel=1e-8)
        assert out.attrs["p_value"] == pytest.approx(p, rel=1e-6)
        assert out.attrs["df"] == 2


def test_sts_peto_fh_and_trend_from_explicit_formulas(data):
    """[ST] sts test Methods and formulas, written as a loop over failure times."""
    t, t0, d, g = (data[c].to_numpy() for c in ("t", "t0", "d", "g"))
    groups = np.unique(g)
    taus = np.unique(t[d > 0])
    tilde, km = 1.0, 1.0
    stats_ = {"peto": [np.zeros(3), np.zeros((3, 3))], "fh": [np.zeros(3), np.zeros((3, 3))]}
    for tau in taus:
        risk = (t0 < tau) & (t >= tau)
        n, e = risk.sum(), ((t == tau) & (d > 0)).sum()
        n_g = np.array([(risk & (g == k)).sum() for k in groups])
        e_g = np.array([((t == tau) & (d > 0) & (g == k)).sum() for k in groups])
        tilde *= 1 - e / (n + 1)
        weights = {"peto": tilde, "fh": km ** 0.5 * (1 - km) ** 1.5}
        for name, weight in weights.items():
            stats_[name][0] += weight * (e_g - n_g * e / n)
            if n > 1:
                factor = weight ** 2 * e * (n - e) / (n * (n - 1))
                stats_[name][1] += factor * (np.diag(n_g) - np.outer(n_g, n_g) / n)
        km *= 1 - e / n
    for name, options in (("peto", {}), ("fh", {"fh_p": 0.5, "fh_q": 1.5})):
        u, V = stats_[name]
        chi2 = u[:2] @ np.linalg.solve(V[:2, :2], u[:2])
        out = oe.sts(data, "t", failure="d", entry="t0", by="g", test=name, trend=True,
                     **options)
        assert out.attrs["statistic"] == pytest.approx(chi2, rel=1e-9)
        a = groups.astype(float)
        trend = (a @ u) ** 2 / (a @ V @ a)
        assert out["tests"].loc[f"{name}_trend", "statistic"] == pytest.approx(trend, rel=1e-9)
        at_risk = [(t0 < tau) & (t >= tau) for tau in taus]
        expected = np.array([sum((risk & (g == k)).sum() * ((t == tau) & (d > 0)).sum()
                                 / risk.sum() for tau, risk in zip(taus, at_risk, strict=True))
                             for k in groups])
        assert_allclose(out["expected"]["expected"], expected, rtol=1e-10)


def test_sts_frequency_weights_equal_duplicated_rows(data):
    long = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    a = oe.sts(data, "t", failure="d", entry="t0", by="g", weights="f", strata="s")
    b = oe.sts(long, "t", failure="d", entry="t0", by="g", strata="s")
    for name in ("survival", "summary", "tests"):
        left, right = a[name], b[name]
        for column in left.columns:
            if left[column].dtype.kind == "f":
                assert_allclose(left[column], right[column], rtol=1e-10, equal_nan=True)


# ---- ltable ------------------------------------------------------------------------------


def test_ltable_from_explicit_actuarial_loop(data):
    t, d = data.t.to_numpy(), data.d.to_numpy()
    width = 0.5
    out = oe.ltable(data, "t", failure="d", intervals=width)
    table = out["table"]
    bins = np.floor(t / width).astype(int)
    alive = len(t)
    surv, green, logs, dens_terms = 1.0, 0.0, 0.0, 0.0
    rows = []
    for j in range(bins.max() + 1):
        deaths = ((bins == j) & (d > 0)).sum()
        lost = ((bins == j) & (d == 0)).sum()
        if deaths + lost == 0:
            continue
        n = alive - lost / 2
        q = deaths / n
        before = surv
        surv *= 1 - q
        if n > deaths:
            green += deaths / (n * (n - deaths))
            logs += np.log((n - deaths) / n)
        hazard = q / ((1 - q / 2) * width)
        hazard_se = hazard * np.sqrt((1 - (width * hazard / 2) ** 2) / deaths) if deaths else None
        density = before * q / width
        density_se = (density * np.sqrt(dens_terms + (1 - q) / (n * q))) if 0 < q < 1 else None
        if q < 1:
            dens_terms += q / (n * (1 - q))
        rows.append((j * width, alive, deaths, lost, n, q, surv, surv * np.sqrt(green), hazard,
                     hazard_se, density, density_se))
        alive -= deaths + lost
    ref = pd.DataFrame(rows, columns=["start", "entering", "deaths", "lost", "at_risk", "q",
                                      "S", "se", "hazard", "hazard_se", "density",
                                      "density_se"])
    assert_allclose(table["interval_start"], ref.start, atol=1e-12)
    assert_allclose(table["entering"], ref.entering)
    assert_allclose(table["deaths"], ref.deaths)
    assert_allclose(table["lost"], ref.lost)
    assert_allclose(table["at_risk"], ref.at_risk)
    assert_allclose(table["death_probability"], ref.q, rtol=1e-12)
    assert_allclose(table["survival"], ref.S, rtol=1e-12)
    assert_allclose(table["std_error"], ref.se, rtol=1e-10)
    assert_allclose(table["hazard"], ref.hazard, rtol=1e-12)
    mask = ref.hazard_se.notna().to_numpy()
    assert_allclose(table["hazard_std_error"][mask], ref.hazard_se[mask].astype(float),
                    rtol=1e-10)
    assert_allclose(table["density"], ref.density, rtol=1e-12)
    mask = ref.density_se.notna().to_numpy()
    assert_allclose(table["density_std_error"][mask], ref.density_se[mask].astype(float),
                    rtol=1e-10)


def test_ltable_homogeneity_tests(data):
    t, d, g = data.t.to_numpy(), data.d.to_numpy(), data.g.to_numpy()
    out = oe.ltable(data, "t", failure="d", by="g", intervals=1)
    D, T = d.sum(), t.sum()
    lr = 2 * (D * np.log(T / D) - sum(d[g == k].sum() * np.log(t[g == k].sum() / d[g == k].sum())
                                      for k in np.unique(g)))
    assert out["tests"].loc["likelihood_ratio", "statistic"] == pytest.approx(lr, rel=1e-10)
    assert out["tests"].loc["likelihood_ratio", "df"] == 2
    chi2, _ = survdiff(t, d, g)
    assert out["tests"].loc["logrank", "statistic"] == pytest.approx(chi2, rel=1e-9)


# ---- further variants ------------------------------------------------------------------


def test_stcox_exactp_with_strata_entry_and_offset():
    rng = np.random.default_rng(8)
    n = 18
    X = rng.normal(size=(n, 2))
    t = rng.integers(2, 6, n).astype(float)
    t0 = np.where(rng.random(n) < 0.4, rng.integers(0, 2, n), 0).astype(float)
    d = (rng.random(n) < 0.85).astype(float)
    s = rng.integers(0, 2, n)
    off = 0.5 * rng.normal(size=n)
    frame = pd.DataFrame({"t": t, "t0": t0, "d": d, "x1": X[:, 0], "x2": X[:, 1], "s": s,
                          "off": off})

    def loglik(b):
        eta = X @ b + off
        total = 0.0
        for risk, dead in failure_times(t, t0, d, s):
            c = int(dead.sum())
            sums = [np.exp(eta[list(sub)].sum())
                    for sub in itertools.combinations(np.flatnonzero(risk), c)]
            total += eta[dead].sum() - np.log(np.sum(sums))
        return total

    fitted = minimize(lambda b: -loglik(b), np.zeros(2), method="BFGS", options={"gtol": 1e-10})
    result = oe.stcox(data=frame, time="t", entry="t0", failure="d", strata="s", offset="off",
                      x=["x1", "x2"], ties="exactp")
    assert_allclose(params(result), fitted.x, atol=1e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-10)
    assert_allclose(cov(result), np.linalg.inv(-approx_hess(fitted.x, loglik)), rtol=2e-5)


def test_stcox_ph_tests_with_strata_and_frequency_weights(data):
    X, t, t0, d = arrays(data)
    zero = np.zeros(len(t))
    s = data.s.to_numpy()
    result = oe.stcox(data=data, time="t", failure="d", strata="s", x=["x1", "x2"])
    b, V = params(result), cov(result)
    r, times = schoenfeld(b, X, t, zero, d, s)
    nd = len(times)
    gc = times - times.mean()
    scaled = b + nd * r @ V
    chi2 = (gc @ scaled[:, 0]) ** 2 / (nd * V[0, 0] * (gc @ gc))
    assert result.tests["ph_x1"]["statistic"] == pytest.approx(chi2, rel=1e-6)
    long = data.loc[data.index.repeat(data.f)].reset_index(drop=True)
    weighted = oe.stcox(data=data, time="t", failure="d", strata="s", x=["x1", "x2"],
                        weights="f", weight_type="fweight", phtest="log")
    plain = oe.stcox(data=long, time="t", failure="d", strata="s", x=["x1", "x2"],
                     phtest="log")
    for name in ("ph_x1", "ph_x2", "ph_global"):
        assert weighted.tests[name]["statistic"] == pytest.approx(plain.tests[name]["statistic"],
                                                                  rel=1e-8)
    base_w, base_p = weighted.extra["baseline"], plain.extra["baseline"]
    assert_allclose(base_w["cumulative_hazard"], base_p["cumulative_hazard"], rtol=1e-10)
    assert_allclose(base_w["survivor_kp"], base_p["survivor_kp"], rtol=1e-10)


def test_stcox_concordance_with_tied_predictions():
    rng = np.random.default_rng(3)
    n = 90
    frame = pd.DataFrame({"g": rng.integers(0, 3, n), "t": rng.integers(1, 8, n).astype(float),
                          "d": (rng.random(n) < 0.7).astype(float)})
    result = oe.stcox(data=frame, time="t", failure="d", x=["g"], categorical=["g"])
    eta = np.array([0.0, *params(result)])[frame.g.to_numpy()]
    t, d = frame.t.to_numpy(), frame.d.to_numpy()
    usable = agree = tied = 0
    for i in range(n):
        for j in range(n):
            if i != j and d[i] > 0 and (t[i] < t[j] or (t[i] == t[j] and d[j] == 0)):
                usable += 1
                agree += eta[i] > eta[j]
                tied += eta[i] == eta[j]
    detail = result.extra["concordance"]
    assert (detail["pairs"], detail["concordant"], detail["tied_predictions"]) == (
        usable, agree, tied)


def test_streg_weights_and_offset_brute_force(sdata):
    t, t0, d = (sdata[c].to_numpy(float) for c in ("t", "t0", "d"))
    X = np.column_stack([np.ones(len(t)), sdata.x])
    Z = np.ones((len(t), 1))
    w = np.random.default_rng(6).uniform(0.2, 3.0, len(t))
    frame = sdata.assign(w=w, off=0.3 * sdata.z)

    def ll(v, weights=None):
        return stata_streg_ll(v, "weibull", "ph", t, t0, d, X, Z, weights)

    # iweights: sum_i w_i l_i as given
    result = oe.streg(data=frame, time="t", failure="d", entry="t0", x=["x"],
                      weights="w", weight_type="iweight")
    theta = params(result)
    fitted = minimize(lambda v: -ll(v, w).sum(), theta + 0.02, method="BFGS",
                      options={"gtol": 1e-9})
    assert_allclose(theta, fitted.x, atol=2e-5)
    assert result.metrics["log_likelihood"] == pytest.approx(ll(theta, w).sum(), rel=1e-10)
    bread = np.linalg.inv(-approx_hess(theta, lambda v: ll(v, w).sum()))
    assert_allclose(cov(result), bread, rtol=1e-4)
    # pweights: same estimates, N/(N-1) sandwich of the weighted scores
    pw = oe.streg(data=frame, time="t", failure="d", entry="t0", x=["x"], weights="w",
                  weight_type="pweight")
    assert pw.inference["covariance"] == "robust"
    assert_allclose(params(pw), theta, atol=1e-8)
    scores = approx_fprime(theta, lambda v: ll(v, w), centered=True)
    n = len(t)
    assert_allclose(cov(pw), n / (n - 1) * bread @ scores.T @ scores @ bread, rtol=1e-4)
    # offset: enters the main index with coefficient one
    with_offset = oe.streg(data=frame, time="t", failure="d", entry="t0", x=["x"],
                           offset="off")
    shifted = np.column_stack([X, frame.off])

    def ll_offset(v):
        return stata_streg_ll(np.r_[v[:2], 1.0, v[2:]], "weibull", "ph", t, t0, d, shifted,
                              Z).sum()

    fitted = minimize(lambda v: -ll_offset(v), params(with_offset) + 0.02, method="BFGS",
                      options={"gtol": 1e-9})
    assert_allclose(params(with_offset), fitted.x, atol=2e-5)
    assert with_offset.metrics["log_likelihood"] == pytest.approx(-fitted.fun, rel=1e-9)


def test_ltable_noadjust_and_cutpoints(data):
    t, d = data.t.to_numpy(), data.d.to_numpy()
    cuts = [0.5, 1.0, 2.0]
    out = oe.ltable(data, "t", failure="d", intervals=cuts, noadjust=True)
    table = out["table"]
    edges = [0.0, *cuts, np.inf]
    alive = len(t)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (t >= lo) & (t < hi)
        deaths, lost = (inside & (d > 0)).sum(), (inside & (d == 0)).sum()
        if deaths + lost:
            q = deaths / alive
            hazard = q / (hi - lo) if np.isfinite(hi) else None
            low = high = None
            if hazard is not None and deaths:
                low = hazard * stats.chi2.ppf(0.025, 2 * deaths) / (2 * deaths)
                high = hazard * stats.chi2.ppf(0.975, 2 * deaths) / (2 * deaths)
            rows.append((lo, alive, q, hazard, low, high))
        alive -= deaths + lost
    assert_allclose(table["interval_start"], [r[0] for r in rows])
    assert_allclose(table["at_risk"], [r[1] for r in rows])
    assert_allclose(table["death_probability"], [r[2] for r in rows], rtol=1e-12)
    finite = [r[3] is not None for r in rows]
    assert_allclose(table["hazard"][finite], [r[3] for r in rows if r[3] is not None],
                    rtol=1e-12)
    assert_allclose(table["hazard_ci_low"][finite], [r[4] for r in rows if r[3] is not None],
                    rtol=1e-8)
    assert_allclose(table["hazard_ci_high"][finite], [r[5] for r in rows if r[3] is not None],
                    rtol=1e-8)
    assert table["interval_end"].isna().iloc[-1]      # open last interval [2, inf)


def kp_alpha(u, w, dead, risk):
    """Kalbfleisch-Prentice factor: root of sum_D w u / (1 - a^u) = sum_R w u (brentq)."""
    from scipy.optimize import brentq
    target = (w * u)[risk].sum()
    ud, wd = u[dead], w[dead]
    if (wd * ud).sum() >= target * (1 - 1e-12):
        return 0.0

    def f(log_a):
        return (wd * ud / -np.expm1(ud * log_a)).sum() - target

    hi = -1e-300
    lo = -1.0
    while f(lo) > 0:
        lo *= 2
    return math.exp(brentq(f, lo, hi, xtol=1e-300, rtol=1e-15, maxiter=500))


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_stcox_baseline_kalbfleisch_prentice_and_efron_hazard(data, ties):
    """[ST] stcox postestimation: basesurv = prod alpha_j (Kalbfleisch-Prentice eq. 4.34);
    basechazard after efron uses Efron's averaged risk sets."""
    X, t, t0, d = arrays(data)
    s = data.s.to_numpy()
    result = oe.stcox(data=data, time="t", failure="d", entry="t0", strata="s",
                      x=["x1", "x2"], ties=ties)
    b = params(result)
    u = np.exp(X @ b)
    w = np.ones(len(t))
    base = result.extra["baseline"]
    hazard, survivor, times, strata = [], [], [], []
    for stratum in np.unique(s):
        total, product = 0.0, 1.0
        member = np.zeros(len(t), bool)
        member[s == stratum] = True
        for risk, dead in failure_times(t, t0, d, np.where(member, 0, 1)):
            if not (risk & member).any() or not (dead & member).any():
                continue
            risk, dead = risk & member, dead & member
            c = dead.sum()
            if ties == "breslow":
                total += c / u[risk].sum()
            else:
                total += sum(1 / (u[risk].sum() - r / c * u[dead].sum()) for r in range(c))
            product *= kp_alpha(u, w, dead, risk)
            hazard.append(total)
            survivor.append(product)
            times.append(t[dead][0])
            strata.append(stratum)
    assert_allclose(base["time"], times)
    assert base["stratum"] == strata
    assert_allclose(base["cumulative_hazard"], hazard, rtol=1e-10)
    assert_allclose(base["survivor"], np.exp(-np.array(hazard)), rtol=1e-10)
    assert_allclose(base["survivor_kp"], survivor, rtol=1e-9)
    curve = oe.stcurve(result, data=data, at={"x1": 0.5, "x2": 0.0})
    assert_allclose(curve["survivor_kp"], np.array(survivor) ** np.exp(0.5 * b[0]), rtol=1e-9)


def test_baseline_at_zero_coefficients_is_kaplan_meier_and_nelson_aalen(data):
    """'When estimated with no covariates, S0(t) is the Kaplan-Meier estimate' and H0
    the Nelson-Aalen estimate ([ST] stcox postestimation)."""
    import torch

    from openecon.econometrics.survival.cox_kernels import CoxObjective
    from openecon.econometrics.survival.data import RiskSets
    from openecon.econometrics.survival.diagnostics import baseline_table

    tensor = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    t, t0, d, f = (data[c].to_numpy(float) for c in ("t", "t0", "d", "f"))
    risk = RiskSets.build(tensor(t), tensor(t0), tensor(d))
    objective = CoxObjective(tensor(data[["x1"]]), risk, tensor(f), None, "breslow")
    increments, log_alpha = objective.baseline(torch.zeros(1, dtype=torch.float64), 0.0)
    table = baseline_table(risk, increments, log_alpha, limit=None)
    ref = km_loop(t, t0, d, f)
    ref = ref[ref.e > 0]
    assert_allclose(table["survivor_kp"], ref.S, rtol=1e-12)
    assert_allclose(table["cumulative_hazard"], ref.H, rtol=1e-12)


def test_stcox_exactp_predictions_use_breslow_formulas():
    """regression: after exactp the PH tests and baselines were omitted; Stata computes
    them with the Peto-Breslow formulas at the exact estimates."""
    rng = np.random.default_rng(12)
    n = 40
    X = rng.normal(size=(n, 2))
    t = rng.integers(1, 9, n).astype(float)
    d = (rng.random(n) < 0.8).astype(float)
    frame = pd.DataFrame({"t": t, "d": d, "x1": X[:, 0], "x2": X[:, 1]})
    result = oe.stcox(data=frame, time="t", failure="d", x=["x1", "x2"], ties="exactp")
    b, V = params(result), cov(result)
    zero = np.zeros(n)
    u = np.exp(X @ b)
    hazard = np.cumsum([dead.sum() / u[risk].sum()
                        for risk, dead in failure_times(t, zero, d, zero)])
    assert_allclose(result.extra["baseline"]["cumulative_hazard"], hazard, rtol=1e-10)
    r, times = schoenfeld(b, X, t, zero, d, zero, "breslow")
    gc = times - times.mean()
    u_ph = gc @ r
    glob = u_ph @ (len(times) * V / (gc @ gc)) @ u_ph
    assert result.tests["ph_global"]["statistic"] == pytest.approx(glob, rel=1e-6)
    curve = oe.stcurve(result, data=frame, at={"x1": 0.0, "x2": 0.0})
    assert_allclose(curve["cumulative_hazard"], hazard, rtol=1e-10)
