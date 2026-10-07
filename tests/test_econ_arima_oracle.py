"""Independent verification oracles for the arima family (written by the verify stage).

Nothing here reuses the family's algebra or the implementer's test code:

- The exact likelihood is an explicit Kalman filter in NumPy (Harvey's state-space form,
  stationary initial covariance from the Kronecker solve of the Lyapunov equation,
  one Python step per period). The package never runs such a filter: it evaluates a
  closed form, so agreement is a real check.
- Estimates come from a brute-force SciPy maximization of that likelihood started from
  OLS with white-noise errors, polished by Newton steps on numerical derivatives.
- Covariances are rebuilt from numerical Hessians and numerical per-observation scores
  of the oracle likelihood: OIM = (-H)^-1, OPG = (S'S)^-1, robust = N/(N-1) OIM S'S OIM.
- Forecasts and their standard errors come from the conditional distribution of a dense
  multivariate normal (autocovariances from the MA(infinity) weights), integrated to
  levels by an explicit matrix.
- statsmodels' SARIMAX filter is a second, unrelated implementation of the likelihood and
  of the OPG / Hessian covariances.
- Correlogram, portmanteau, Jarque-Bera and ARCH-LM statistics are recomputed from their
  textbook formulas; exponential smoothing from explicit recursions and a bounded
  brute-force search.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

import openecon as oe

# ---- the oracle: explicit Kalman filter and brute-force maximum likelihood -------------------


def polynomials(phi, theta, sphi=(), stheta=(), s=1):
    """Expanded AR polynomial 1 - phi*(L) and MA polynomial 1 + theta*(L), increasing powers."""
    ar, ma = np.r_[1.0, -np.asarray(phi, float)], np.r_[1.0, np.asarray(theta, float)]
    sar, sma = np.zeros(len(sphi) * s + 1), np.zeros(len(stheta) * s + 1)
    sar[0] = sma[0] = 1.0
    for j, value in enumerate(sphi, start=1):
        sar[j * s] = -value
    for j, value in enumerate(stheta, start=1):
        sma[j * s] = value
    return np.convolve(ar, sar), np.convolve(ma, sma)


def kalman(u, ar, ma, sigma):
    """Exact Gaussian log likelihood of a stationary ARMA series by the Kalman filter.

    Returns the per-observation log likelihood, the innovations v_t and the ratios
    F_t = Var(v_t) / sigma^2.
    """
    n, p, q = len(u), len(ar) - 1, len(ma) - 1
    r = max(p, q + 1)
    transition = np.zeros((r, r))
    transition[:p, 0] = -ar[1:]
    transition[:-1, 1:] = np.eye(r - 1)
    loading = np.zeros(r)
    loading[:q + 1] = ma
    noise = np.outer(loading, loading)
    cov = np.linalg.solve(np.eye(r * r) - np.kron(transition, transition),
                          noise.ravel()).reshape(r, r)
    state = np.zeros(r)
    ll, v, f = np.empty(n), np.empty(n), np.empty(n)
    for t in range(n):
        v[t], f[t] = u[t] - state[0], cov[0, 0]
        ll[t] = -0.5 * (math.log(2 * math.pi * sigma ** 2 * f[t]) + v[t] ** 2 / (sigma ** 2 * f[t]))
        gain = transition @ cov[:, 0] / f[t]
        state = transition @ state + gain * v[t]
        cov = transition @ cov @ transition.T + noise - np.outer(gain, gain) * f[t]
        cov = (cov + cov.T) / 2
    return ll, v, f


class Oracle:
    """Regression with (seasonal) ARMA errors on already differenced data.

    Parameters are ordered (b, phi, theta, Phi, Theta, sigma), as oe.arima reports them.
    ``exact=False`` is the conditional likelihood: presample u and e are zero, every
    observation has variance sigma^2.
    """

    def __init__(self, y, x, p=0, q=0, sp=0, sq=0, s=1, exact=True):
        self.y, self.x = np.asarray(y, float), np.asarray(x, float).reshape(len(y), -1)
        self.k, self.sizes, self.s, self.exact = self.x.shape[1], (p, q, sp, sq), s, exact

    def split(self, theta):
        k, (p, q, sp, sq) = self.k, self.sizes
        edges = np.cumsum([k, p, q, sp, sq])
        return theta[:k], *(theta[a:b] for a, b in zip(edges[:-1], edges[1:])), theta[-1]

    def parts(self, theta):
        b, phi, ma, sphi, sma, sigma = self.split(np.asarray(theta, float))
        ar, ma = polynomials(phi, ma, sphi, sma, self.s)
        return self.y - self.x @ b, ar, ma, sigma

    def loglike_obs(self, theta):
        u, ar, ma, sigma = self.parts(theta)
        if not sigma > 0.0:
            return np.full(len(u), -np.inf)
        if self.exact:
            # np.roots(ar) are the inverse roots of the AR polynomial (ar is 1, -phi_1, ...).
            if len(ar) > 1 and np.abs(np.roots(ar)).max() >= 1.0:
                return np.full(len(u), -np.inf)
            return kalman(u, ar, ma, sigma)[0]
        e = lfilter(ar, ma, u)
        return -0.5 * math.log(2 * math.pi) - math.log(sigma) - e ** 2 / (2 * sigma ** 2)

    def innovations(self, theta):
        u, ar, ma, sigma = self.parts(theta)
        return kalman(u, ar, ma, sigma)[1] if self.exact else lfilter(ar, ma, u)

    def loglike(self, theta):
        return float(self.loglike_obs(theta).sum())

    @staticmethod
    def _steps(theta, h):
        return h * np.maximum(1.0, np.abs(theta))

    def scores(self, theta, h=1e-5):
        """Per-observation scores by fourth-order central differences."""
        theta = np.asarray(theta, float)
        steps = self._steps(theta, h)
        out = np.empty((len(self.y), len(theta)))
        for i, step in enumerate(steps):
            e = np.zeros(len(theta))
            e[i] = step
            out[:, i] = (8 * (self.loglike_obs(theta + e) - self.loglike_obs(theta - e))
                         - self.loglike_obs(theta + 2 * e) + self.loglike_obs(theta - 2 * e)) \
                / (12 * step)
        return out

    def hessian(self, theta, h=1e-4):
        theta = np.asarray(theta, float)
        steps = self._steps(theta, h)
        size = len(theta)
        out = np.empty((size, size))
        for i in range(size):
            ei = np.zeros(size)
            ei[i] = steps[i]
            for j in range(i, size):
                ej = np.zeros(size)
                ej[j] = steps[j]
                out[i, j] = out[j, i] = (
                    self.loglike(theta + ei + ej) - self.loglike(theta + ei - ej)
                    - self.loglike(theta - ei + ej) + self.loglike(theta - ei - ej)
                ) / (4 * steps[i] * steps[j])
        return out

    def maximize(self):
        """OLS / white-noise start, SciPy BFGS on finite differences, Newton polish."""
        b = np.linalg.lstsq(self.x, self.y, rcond=None)[0] if self.k else np.zeros(0)
        resid = self.y - self.x @ b
        start = np.r_[b, np.zeros(sum(self.sizes)), resid.std()]

        def negative(theta):
            value = -self.loglike(theta) if theta[-1] > 0 else np.inf
            return value if np.isfinite(value) else 1e300

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theta = minimize(negative, start, method="BFGS",
                             options={"gtol": 1e-5, "maxiter": 1000}).x
        for _ in range(8):
            step = np.linalg.solve(self.hessian(theta), self.scores(theta).sum(axis=0))
            theta = theta - step
            if np.abs(step).max() < 1e-11:
                break
        return theta

    def covariances(self, theta):
        n = len(self.y)
        scores = self.scores(theta)
        opg = scores.T @ scores
        oim = np.linalg.inv(-self.hessian(theta))
        return {"nonrobust": oim, "opg": np.linalg.inv(opg),
                "robust": n / (n - 1) * oim @ opg @ oim}


def simulate(n, phi=(), theta=(), seed=0, sphi=(), stheta=(), s=1, burn=300):
    e = np.random.default_rng(seed).normal(size=n + burn)
    ar, ma = polynomials(phi, theta, sphi, stheta, s)
    return lfilter(ma, ar, e)[burn:]


def difference(values, d, big_d=0, s=1):
    values = np.asarray(values, float)
    for _ in range(big_d):
        values = values[s:] - values[:-s]
    for _ in range(d):
        values = values[1:] - values[:-1]
    return values


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def std_errors(result):
    return np.array([c.std_error for c in result.coefficients])


def scaled_difference(actual, expected):
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    return np.abs(np.asarray(actual) - expected).max() / 1.0 if scale.size == 0 else \
        (np.abs(np.asarray(actual) - expected) / scale).max()


# ---- designs -----------------------------------------------------------------------------------


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": np.arange(1, n + 1), "x1": rng.normal(size=n),
                         "x2": rng.uniform(0.0, 3.0, size=n),
                         "g": np.resize(["a", "b", "c"], n)})


def _designs():
    cases = {}
    frame = _frame(160, 42)
    frame["y"] = 2.0 + 0.7 * frame.x1 - 0.4 * frame.x2 + simulate(160, [0.5], [0.4], seed=1)
    cases["armax_ml"] = (frame, dict(y="y", x=["x1", "x2"], order=(1, 0, 1)))
    cases["armax_css"] = (frame, dict(y="y", x=["x1", "x2"], order=(1, 0, 1), method="css"))
    frame = _frame(150, 7)
    frame["y"] = 1.0 + simulate(150, [0.5, -0.3], seed=2)
    cases["ar2_ml"] = (frame, dict(y="y", order=(2, 0, 0)))
    frame = _frame(150, 8)
    frame["y"] = simulate(150, [], [0.5, 0.3], seed=3)
    cases["ma2_noconstant"] = (frame, dict(y="y", order=(0, 0, 2), constant=False))
    frame = _frame(140, 9)
    frame["y"] = np.cumsum(0.3 + simulate(140, [0.4], [0.3], seed=4)) + 0.5 * frame.x1
    cases["arima111_drift_x"] = (frame, dict(y="y", x=["x1"], order=(1, 1, 1)))
    frame = _frame(150, 10)
    shock = simulate(150, [], [-0.4], seed=5, stheta=[-0.5], s=12)
    frame["y"] = lfilter([1.0], np.convolve([1.0, -1.0], np.r_[1.0, np.zeros(11), -1.0]), shock)
    cases["airline"] = (frame, dict(y="y", order=(0, 1, 1), seasonal=(0, 1, 1), period=12,
                                    constant=False))
    frame = _frame(170, 11)
    frame["y"] = simulate(170, [0.3], [0.4], seed=7, sphi=[0.4], stheta=[0.3], s=4)
    cases["sarma_css"] = (frame, dict(y="y", order=(1, 0, 1), seasonal=(1, 0, 1), period=4,
                                      method="css"))
    frame = _frame(170, 12)
    frame["y"] = 3.0 + simulate(170, [0.4], seed=6, sphi=[0.5], s=4)
    cases["sar_ml"] = (frame, dict(y="y", order=(1, 0, 0), seasonal=(1, 0, 0), period=4))
    frame = _frame(26, 13)
    frame["y"] = 0.5 + simulate(26, [0.6], [0.3], seed=21)
    cases["tiny_arma11"] = (frame, dict(y="y", order=(1, 0, 1)))
    frame = _frame(120, 14)
    frame["x1"] *= 1e-4
    frame["y"] = 1e6 * (2.0 + 3e3 * frame.x1 + simulate(120, [0.6], seed=22))
    cases["badly_scaled_ar1"] = (frame, dict(y="y", x=["x1"], order=(1, 0, 0)))
    frame = _frame(132, 15)
    frame["y"] = 1.0 + 0.8 * (frame.g == "b") - 0.5 * (frame.g == "c") \
        + simulate(132, [], [0.5], seed=23)
    cases["categorical_ma1_css"] = (frame, dict(y="y", x=["g", "x1"], categorical=["g"],
                                                order=(0, 0, 1), method="css"))
    frame = _frame(128, 16)
    frame["y"] = lfilter([1.0], np.convolve([1.0, -1.0], [1.0, 0, 0, 0, -1.0]),
                         simulate(128, [], [-0.5], seed=24)) + 0.6 * frame.x2.cumsum()
    frame["x2"] = frame.x2.cumsum()
    cases["seasonal_difference_x"] = (frame, dict(y="y", x=["x2"], order=(0, 1, 1),
                                                  seasonal=(0, 1, 0), period=4, constant=False))
    return cases


DESIGNS = _designs()


def oracle_for(frame, options):
    """The oracle model of one design: differenced data, explicit dummies, the constant."""
    p, d, q = options["order"]
    sp, big_d, sq = options.get("seasonal") or (0, 0, 0)
    s = options.get("period") or 1
    columns, names = [], []
    for name in options.get("x") or []:
        if name in (options.get("categorical") or []):
            for level in sorted(frame[name].unique())[1:]:
                columns.append((frame[name] == level).to_numpy(float))
                names.append(f"{name}[{level}]")
        else:
            columns.append(frame[name].to_numpy(float))
            names.append(name)
    y = difference(frame[options["y"]], d, big_d, s)
    x = np.column_stack([difference(column, d, big_d, s) for column in columns]) \
        if columns else np.zeros((len(y), 0))
    if options.get("constant", True):
        x, names = np.column_stack([np.ones(len(y)), x]), ["Intercept", *names]
    exact = options.get("method", "ml") == "ml"
    return Oracle(y, x, p, q, sp, sq, s, exact), names, len(frame) - len(y)


@pytest.fixture(scope="module", params=list(DESIGNS))
def solved(request):
    frame, options = DESIGNS[request.param]
    oracle, names, lost = oracle_for(frame, options)
    theta = oracle.maximize()
    fits = {kind: oe.arima(data=frame, time="t", covariance=kind, **options)
            for kind in ("opg", "nonrobust", "robust")}
    return request.param, frame, options, oracle, names, lost, theta, fits


def test_estimates_and_likelihood_match_bruteforce_kalman_filter(solved):
    name, frame, options, oracle, names, lost, theta, fits = solved
    fit = fits["opg"]
    p, d, q = options["order"]
    sp, _, sq = options.get("seasonal") or (0, 0, 0)
    s = options.get("period") or 0
    terms = names + [f"ARMA:L{i}.ar" for i in range(1, p + 1)] \
        + [f"ARMA:L{i}.ma" for i in range(1, q + 1)] \
        + [f"ARMA{s}:L{i}.ar" for i in range(1, sp + 1)] \
        + [f"ARMA{s}:L{i}.ma" for i in range(1, sq + 1)] + ["/sigma"]
    assert [c.term for c in fit.coefficients] == terms
    n = len(oracle.y)
    assert fit.nobs == n and fit.nobs_original == len(frame) and fit.dropped_rows == lost
    assert fit.sample_positions == list(range(lost, len(frame)))
    assert_allclose(estimates(fit), theta, rtol=2e-6, atol=1e-8 * np.abs(theta).max())
    ll = oracle.loglike(theta)
    assert fit.metrics["log_likelihood"] == pytest.approx(ll, rel=1e-10, abs=1e-8)
    # The likelihood at the package's own estimates, through the independent filter.
    assert oracle.loglike(estimates(fit)) == pytest.approx(fit.metrics["log_likelihood"],
                                                           rel=1e-11, abs=1e-8)
    k = len(theta)
    assert fit.metrics["aic"] == pytest.approx(-2 * ll + 2 * k, rel=1e-10)
    assert fit.metrics["bic"] == pytest.approx(-2 * ll + k * math.log(n), rel=1e-10)
    assert fit.metrics["sigma"] == pytest.approx(theta[-1], rel=2e-6)
    assert fit.metrics["n_differenced"] == n
    # sigma is the ML estimate: root mean squared standardized innovation, divisor N.
    u, ar, ma, _ = oracle.parts(theta)
    if oracle.exact:
        _, v, f = kalman(u, ar, ma, 1.0)
        assert theta[-1] == pytest.approx(math.sqrt(np.mean(v ** 2 / f)), rel=1e-6)
    else:
        assert theta[-1] == pytest.approx(math.sqrt(np.mean(lfilter(ar, ma, u) ** 2)), rel=1e-6)
    assert fit.provenance["stata_parity_validated"] is False
    assert fit.inference["distribution"] == "normal" and fit.inference["use_t"] is False


def test_all_three_covariances_match_numerical_oim_opg_and_sandwich(solved):
    name, frame, options, oracle, names, lost, theta, fits = solved
    expected = oracle.covariances(theta)
    tolerance = {"opg": 2e-5, "nonrobust": 2e-5, "robust": 4e-5}
    if name == "tiny_arma11":
        tolerance = {key: 10 * value for key, value in tolerance.items()}
    for kind, fit in fits.items():
        assert fit.spec.covariance == kind and fit.inference["covariance"] == kind
        covariance = np.array(fit.covariance_matrix)
        assert scaled_difference(covariance, expected[kind]) < tolerance[kind], kind
        assert_allclose(std_errors(fit), np.sqrt(np.diag(expected[kind])), rtol=tolerance[kind])
        # Point estimates do not depend on the covariance estimator.
        assert_allclose(estimates(fit), estimates(fits["opg"]), rtol=1e-12, atol=1e-14)
    assert fits["opg"].spec.covariance == oe.arima(data=frame, time="t", **options).spec.covariance


def test_inference_rows_wald_test_and_sigma_conventions(solved):
    name, frame, options, oracle, names, lost, theta, fits = solved
    for kind, fit in fits.items():
        b, se = estimates(fit), std_errors(fit)
        covariance = np.array(fit.covariance_matrix)
        z = b / se
        p_values = 2 * stats.norm.sf(np.abs(z))
        p_values[-1] /= 2                                   # Stata: /sigma is tested one-sided
        critical = stats.norm.ppf(0.975)
        low, high = b - critical * se, b + critical * se
        low[-1] = max(low[-1], 0.0)                         # and its interval is truncated at 0
        assert_allclose([c.statistic for c in fit.coefficients], z, rtol=1e-12)
        assert_allclose([c.p_value for c in fit.coefficients], p_values, rtol=1e-9, atol=1e-300)
        assert_allclose([c.ci_low for c in fit.coefficients], low, rtol=1e-10, atol=1e-12)
        assert_allclose([c.ci_high for c in fit.coefficients], high, rtol=1e-10, atol=1e-12)
        tested = [i for i, term in enumerate(names) if term != "Intercept"] \
            + list(range(len(names), len(b) - 1))
        wald = b[tested] @ np.linalg.solve(covariance[np.ix_(tested, tested)], b[tested])
        model = fit.tests["model"]
        assert model["statistic"] == pytest.approx(wald, rel=1e-8)
        assert model["df"] == len(tested) and model["distribution"] == "chi2"
        assert model["p_value"] == pytest.approx(stats.chi2.sf(wald, len(tested)), rel=1e-7,
                                                 abs=1e-300)


def test_residual_tests_and_fitted_values_use_the_filter_innovations(solved):
    name, frame, options, oracle, names, lost, theta, fits = solved
    fit = fits["opg"]
    v = oracle.innovations(estimates(fit))
    n = len(v)
    rows = pd.DataFrame(fit.predictions)
    assert len(rows) == min(n, 400) and rows["row"].min() == lost
    levels = frame[options["y"]].to_numpy(float)
    index = rows["row"].to_numpy()
    scale = np.abs(levels).max()
    assert_allclose(rows["residual"], v[index - lost], atol=1e-9 * scale)
    assert_allclose(rows["fitted"], levels[index] - v[index - lost], atol=1e-9 * scale)
    assert_allclose(rows["observed"], levels[index], rtol=1e-14)
    # Ljung-Box at Stata's default lag min(floor(N/2) - 2, 40), df = lags.
    lags = min(n // 2 - 2, 40)
    centered = v - v.mean()
    r = np.array([centered[j:] @ centered[:n - j] for j in range(1, lags + 1)]) \
        / (centered @ centered)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    box = fit.tests["ljung_box"]
    assert box["lags"] == lags and box["df"] == lags
    assert box["statistic"] == pytest.approx(q, rel=1e-8)
    assert box["p_value"] == pytest.approx(stats.chi2.sf(q, lags), rel=1e-7)
    n_arma = len(theta) - len(names) - 1
    if lags > n_arma > 0:
        assert box["df_adjusted"] == lags - n_arma
        assert box["p_value_adjusted"] == pytest.approx(stats.chi2.sf(q, lags - n_arma), rel=1e-7)
    skew, kurt = stats.skew(v), stats.kurtosis(v, fisher=False)
    jb = n / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4)
    assert fit.tests["jarque_bera"]["statistic"] == pytest.approx(jb, rel=1e-8)
    assert fit.tests["jarque_bera"]["p_value"] == pytest.approx(stats.chi2.sf(jb, 2), rel=1e-7)
    squares = v ** 2
    design = np.column_stack([np.ones(n - 1), squares[:-1]])
    resid = squares[1:] - design @ np.linalg.lstsq(design, squares[1:], rcond=None)[0]
    r2 = 1 - resid @ resid / np.sum((squares[1:] - squares[1:].mean()) ** 2)
    assert fit.tests["arch_lm"]["statistic"] == pytest.approx((n - 1) * r2, rel=1e-7)
    assert fit.tests["arch_lm"]["p_value"] == pytest.approx(stats.chi2.sf((n - 1) * r2, 1),
                                                            rel=1e-6)
    # Inverse roots reported in extra are those of the estimated polynomials.
    p, _, q_order = options["order"]
    k = len(names)
    b = estimates(fit)
    if p:
        moduli = sorted(abs(root) for root in np.roots(np.r_[1.0, -b[k:k + p]]))
        assert_allclose(sorted(root["modulus"] for root in fit.extra["roots"]["ar"]), moduli,
                        rtol=1e-8)
    if q_order:
        moduli = sorted(abs(root) for root in np.roots(np.r_[1.0, b[k + p:k + p + q_order]]))
        assert_allclose(sorted(root["modulus"] for root in fit.extra["roots"]["ma"]), moduli,
                        rtol=1e-8)


# ---- forecasts against the conditional normal distribution -----------------------------------


def autocovariances(ar, ma, count, terms=20_000):
    impulse = np.zeros(terms)
    impulse[0] = 1.0
    psi = lfilter(ma, ar, impulse)
    return np.array([psi[:terms - k] @ psi[k:] for k in range(count)])


def dense_forecast(frame, options, theta, names, horizon, future):
    """Forecasts of the outcome in levels and their standard errors, by dense algebra."""
    p, d, q = options["order"]
    sp, big_d, sq = options.get("seasonal") or (0, 0, 0)
    s = options.get("period") or 1
    exact = options.get("method", "ml") == "ml"
    constant = "Intercept" in names
    k = len(names)
    slopes = theta[int(constant):k]
    drift = theta[0] if constant else 0.0
    arma, sigma = theta[k:-1], theta[-1]
    ar, ma = polynomials(arma[:p], arma[p:p + q], arma[p + q:p + q + sp], arma[p + q + sp:], s)
    regressors = [name for name in names if name != "Intercept"]
    net = frame[options["y"]].to_numpy(float) - frame[regressors].to_numpy(float) @ slopes
    delta = np.array([1.0])
    for _ in range(d):
        delta = np.convolve(delta, [1.0, -1.0])
    for _ in range(big_d):
        delta = np.convolve(delta, np.r_[1.0, np.zeros(s - 1), -1.0])
    lost = len(delta) - 1
    u = np.convolve(net, delta)[lost:len(net)] - drift
    n = len(u)
    if exact:
        gamma = autocovariances(ar, ma, n + horizon)
        full = gamma[np.abs(np.subtract.outer(np.arange(n + horizon), np.arange(n + horizon)))]
        solved = np.linalg.solve(full[:n, :n], full[:n, n:])
        mean, cov = solved.T @ u, full[n:, n:] - full[n:, :n] @ solved
    else:
        impulse = np.zeros(n + horizon)
        impulse[0] = 1.0
        psi = lfilter(ma, ar, impulse)
        weights = np.array([[psi[i - j] if i >= j else 0.0 for j in range(n + horizon)]
                            for i in range(n + horizon)])
        mean = weights[n:, :n] @ lfilter(ar, ma, u)
        cov = weights[n:, n:] @ weights[n:, n:].T
    impulse = np.zeros(horizon)
    impulse[0] = 1.0
    integrate = lfilter([1.0], delta, impulse)
    matrix = np.array([[integrate[i - j] if i >= j else 0.0 for j in range(horizon)]
                       for i in range(horizon)])
    levels = list(net)
    for h in range(horizon):
        levels.append(drift + mean[h] - sum(delta[j] * levels[-j] for j in range(1, lost + 1)))
    point = np.array(levels[len(net):])
    if regressors:
        point = point + future[regressors].to_numpy(float) @ slopes
    return point, sigma * np.sqrt(np.diag(matrix @ cov @ matrix.T))


@pytest.mark.parametrize("name", ["armax_ml", "armax_css", "ma2_noconstant", "arima111_drift_x",
                                  "airline", "sarma_css", "sar_ml", "seasonal_difference_x"])
def test_forecasts_match_the_conditional_normal_distribution(name):
    frame, options = DESIGNS[name]
    horizon = 14
    head, future = frame.iloc[:-horizon], frame.iloc[-horizon:]
    fit = oe.arima(data=head, time="t", **options)
    names = [c.term for c in fit.coefficients if c.equation not in {"ARMA", None}
             and not str(c.equation).startswith("ARMA")]
    exog = future[options["x"]] if options.get("x") else None
    table = oe.forecast(fit, horizon, exog=exog, alpha=0.1)
    point, error = dense_forecast(head, options, estimates(fit), names, horizon, future)
    assert_allclose(table["forecast"], point, rtol=1e-9, atol=1e-9 * np.abs(point).max())
    assert_allclose(table["std_error"], error, rtol=1e-8)
    critical = stats.norm.ppf(0.95)
    assert_allclose(table["ci_low"], point - critical * error, rtol=1e-7, atol=1e-7)
    assert_allclose(table["ci_high"], point + critical * error, rtol=1e-7, atol=1e-7)
    assert table["period"].tolist() == future["t"].tolist()
    assert table.attrs["alpha"] == 0.1 and table.attrs["steps"] == horizon
    # Forecasting from supplied data reproduces the stored state; a longer sample moves
    # the origin (parameters fixed) and matches the dense oracle on that sample.
    again = oe.forecast(fit, horizon, data=head, exog=exog, alpha=0.1)
    assert_allclose(again["forecast"], table["forecast"], rtol=1e-12)
    assert_allclose(again["std_error"], table["std_error"], rtol=1e-10)
    longer, later = frame.iloc[:-4], frame.iloc[-4:]
    moved = oe.forecast(fit, 4, data=longer, exog=later[options["x"]] if options.get("x") else None)
    point, error = dense_forecast(longer, options, estimates(fit), names, 4, later)
    assert_allclose(moved["forecast"], point, rtol=1e-9, atol=1e-9 * np.abs(point).max())
    assert_allclose(moved["std_error"], error, rtol=1e-8)
    assert moved["period"].tolist() == later["t"].tolist()


# ---- statsmodels: a second, unrelated filter ---------------------------------------------------


def _statsmodels(oracle, options, theta):
    p, _, q = options["order"]
    sp, _, sq = options.get("seasonal") or (0, 0, 0)
    s = options.get("period") or 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(oracle.y, exog=oracle.x if oracle.k else None, order=(p, 0, q),
                        seasonal_order=(sp, 0, sq, s), trend="n")
        params = np.r_[theta[:-1], theta[-1] ** 2]
        return {kind: model.smooth(params, cov_type=kind)
                for kind in ("opg", "approx", "robust_approx")}


@pytest.mark.parametrize("name", ["armax_ml", "ar2_ml", "arima111_drift_x", "airline", "sar_ml"])
def test_statsmodels_filter_and_covariances_agree_at_the_estimates(name):
    """SARIMAX on the differenced data (Stata's approach) with sigma^2 as its last parameter.

    The covariance in sigma follows from the one in sigma^2 by the delta method, which is
    exact for the OPG and sandwich forms (scores transform linearly).
    """
    frame, options = DESIGNS[name]
    oracle, names, lost = oracle_for(frame, options)
    fits = {kind: oe.arima(data=frame, time="t", covariance=kind, **options)
            for kind in ("opg", "nonrobust", "robust")}
    theta = estimates(fits["opg"])
    n = len(oracle.y)
    reference = _statsmodels(oracle, options, theta)
    assert fits["opg"].metrics["log_likelihood"] == pytest.approx(reference["opg"].llf, rel=1e-10)
    jacobian = np.diag(np.r_[np.ones(len(theta) - 1), 1 / (2 * theta[-1])])
    expected = {
        "opg": jacobian @ reference["opg"].cov_params() @ jacobian,
        "nonrobust": jacobian @ reference["approx"].cov_params() @ jacobian,
        "robust": n / (n - 1) * jacobian @ reference["robust_approx"].cov_params() @ jacobian,
    }
    assert scaled_difference(fits["opg"].covariance_matrix, expected["opg"]) < 1e-6
    assert scaled_difference(fits["nonrobust"].covariance_matrix, expected["nonrobust"]) < 5e-4
    assert scaled_difference(fits["robust"].covariance_matrix, expected["robust"]) < 1e-3
    # statsmodels' own optimum (started at ours) does not find a higher likelihood.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, _, q = options["order"]
        sp, _, sq = options.get("seasonal") or (0, 0, 0)
        refit = SARIMAX(oracle.y, exog=oracle.x if oracle.k else None, order=(p, 0, q),
                        seasonal_order=(sp, 0, sq, options.get("period") or 0), trend="n").fit(
            start_params=np.r_[theta[:-1], theta[-1] ** 2], disp=0, maxiter=200)
    assert refit.llf <= fits["opg"].metrics["log_likelihood"] + 1e-7


# ---- invariances and equivalences ---------------------------------------------------------------


def test_rescaling_the_outcome_and_the_regressors():
    frame, options = DESIGNS["armax_ml"]
    base = {kind: oe.arima(data=frame, time="t", covariance=kind, **options)
            for kind in ("opg", "nonrobust", "robust")}
    c, a = 1e5, 1e-3
    scaled = frame.assign(y=frame.y * c, x1=frame.x1 * a)
    n = len(frame)
    #            Intercept  x1     x2  ar   ma   sigma
    factor = np.array([c, c / a, c, 1.0, 1.0, c])
    for kind, fit in base.items():
        other = oe.arima(data=scaled, time="t", covariance=kind, **options)
        assert_allclose(estimates(other), estimates(fit) * factor, rtol=2e-6)
        assert_allclose(std_errors(other), std_errors(fit) * factor, rtol=2e-4)
        assert other.metrics["log_likelihood"] == pytest.approx(
            fit.metrics["log_likelihood"] - n * math.log(c), rel=1e-9)
        assert_allclose([c_.statistic for c_ in other.coefficients],
                        [c_.statistic for c_ in fit.coefficients], rtol=2e-4)
        assert other.tests["model"]["statistic"] == pytest.approx(fit.tests["model"]["statistic"],
                                                                  rel=5e-4)
        assert other.tests["ljung_box"]["statistic"] == pytest.approx(
            fit.tests["ljung_box"]["statistic"], rel=1e-5)
    # A large level shift of the outcome only moves the constant.
    shifted = oe.arima(data=frame.assign(y=frame.y + 1e7), time="t", **options)
    expected = estimates(base["opg"]) + np.array([1e7, 0, 0, 0, 0, 0])
    assert_allclose(estimates(shifted), expected, rtol=1e-6, atol=1e-6)
    assert_allclose(std_errors(shifted), std_errors(base["opg"]), rtol=1e-4)


def test_row_order_is_irrelevant_with_a_time_column_and_decisive_without():
    frame, options = DESIGNS["arima111_drift_x"]
    ordered = oe.arima(data=frame, time="t", **options)
    shuffled = frame.sample(frac=1.0, random_state=5)
    permuted = oe.arima(data=shuffled, time="t", **options)
    assert_allclose(estimates(permuted), estimates(ordered), rtol=1e-12)
    assert_allclose(permuted.covariance_matrix, ordered.covariance_matrix, rtol=1e-10)
    # Sample positions refer to the caller's rows: the first period is the one excluded.
    position_of_first = int(np.flatnonzero(shuffled["t"].to_numpy() == 1)[0])
    assert position_of_first not in permuted.sample_positions and permuted.nobs == len(frame) - 1
    assert_allclose(oe.forecast(permuted, 5, exog=frame[["x1"]].iloc[:5])["forecast"],
                    oe.forecast(ordered, 5, exog=frame[["x1"]].iloc[:5])["forecast"], rtol=1e-12)
    # Without a time column the row order is the time order.
    implicit = oe.arima(data=frame, **options)
    assert_allclose(estimates(implicit), estimates(ordered), rtol=1e-12)
    assert implicit.provenance["time_order"] == "input row order"
    assert abs(oe.arima(data=shuffled, **options).metrics["log_likelihood"]
               - ordered.metrics["log_likelihood"]) > 1.0
    # Datetime periods are taken as consecutive, with a recorded warning.
    dated = frame.assign(t=pd.date_range("2001-01-31", periods=len(frame), freq="ME"))
    by_date = oe.arima(data=dated.sample(frac=1.0, random_state=2), time="t", **options)
    assert_allclose(estimates(by_date), estimates(ordered), rtol=1e-12)
    assert any("consecutive periods" in warning for warning in by_date.warnings)


def test_differencing_inside_equals_differencing_by_hand():
    frame, options = DESIGNS["arima111_drift_x"]
    inside = oe.arima(data=frame, time="t", **options)
    by_hand = pd.DataFrame({"dy": np.diff(frame.y), "dx1": np.diff(frame.x1)})
    outside = oe.arima(data=by_hand, y="dy", x=["dx1"], order=(1, 0, 1))
    assert_allclose(estimates(inside), estimates(outside), rtol=1e-10)
    assert_allclose(inside.covariance_matrix, outside.covariance_matrix, rtol=1e-8)
    assert inside.metrics["log_likelihood"] == pytest.approx(outside.metrics["log_likelihood"],
                                                             rel=1e-12)
    assert inside.coefficients[0].equation == "D.y" and outside.coefficients[0].equation == "dy"
    frame, options = DESIGNS["seasonal_difference_x"]
    inside = oe.arima(data=frame, time="t", **options)
    dy, dx = difference(frame.y, 1, 1, 4), difference(frame.x2, 1, 1, 4)
    outside = oe.arima(data=pd.DataFrame({"dy": dy, "dx": dx}), y="dy", x=["dx"],
                       order=(0, 0, 1), constant=False)
    assert_allclose(estimates(inside), estimates(outside), rtol=1e-10)
    assert_allclose(std_errors(inside), std_errors(outside), rtol=1e-8)
    assert inside.coefficients[0].equation == "DS4.y" and inside.dropped_rows == 5
    # Two regular differences: the label is D2.y and two rows are lost.
    twice = oe.arima(data=frame, y="y", order=(0, 2, 1), time="t", constant=False)
    assert twice.dropped_rows == 2 and twice.metrics["n_differenced"] == len(frame) - 2
    assert twice.extra["model_order"]["d"] == 2


def test_equivalent_parameterizations_give_the_same_fit():
    frame, options = DESIGNS["sar_ml"]
    # A seasonal AR(1) with period 4 and nothing else is an AR(4) with three zero
    # coefficients; a pure seasonal MA likewise. Likelihoods at the seasonal estimates agree.
    seasonal = oe.arima(data=frame, y="y", order=(0, 0, 0), seasonal=(1, 0, 0), period=4, time="t")
    b = estimates(seasonal)
    oracle = Oracle(frame.y, np.ones((len(frame), 1)), p=4)
    assert oracle.loglike(np.r_[b[0], 0, 0, 0, b[1], b[2]]) == pytest.approx(
        seasonal.metrics["log_likelihood"], rel=1e-11)
    # seasonal=(0, 0, 0) with a period is the non-seasonal model.
    plain = oe.arima(data=frame, y="y", order=(1, 0, 0), time="t")
    trivial = oe.arima(data=frame, y="y", order=(1, 0, 0), seasonal=(0, 0, 0), period=4, time="t")
    assert_allclose(estimates(plain), estimates(trivial), rtol=1e-13)
    assert [c.term for c in trivial.coefficients] == ["Intercept", "ARMA:L1.ar", "/sigma"]
    # White noise with a constant: the sample mean and the ML standard deviation.
    y = frame.y.to_numpy()
    white = oe.arima(data=frame, y="y", order=(0, 0, 0), covariance="nonrobust")
    n = len(y)
    sigma = y.std()
    assert_allclose(estimates(white), [y.mean(), sigma], rtol=1e-12)
    assert_allclose(std_errors(white), [sigma / math.sqrt(n), sigma / math.sqrt(2 * n)], rtol=1e-6)
    assert white.metrics["log_likelihood"] == pytest.approx(stats.norm.logpdf(
        y, y.mean(), sigma).sum(), rel=1e-12)
    assert "model" not in white.tests
    # Regression with white-noise errors: OLS coefficients, ML sigma.
    frame, _ = DESIGNS["armax_ml"]
    regression = oe.arima(data=frame, y="y", x=["x1", "x2"], order=(0, 0, 0),
                          covariance="nonrobust")
    design = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    beta, ssr = np.linalg.lstsq(design, frame.y, rcond=None)[:2]
    assert_allclose(estimates(regression)[:3], beta, rtol=1e-10)
    s2 = float(ssr[0]) / len(frame)
    assert_allclose(np.array(regression.covariance_matrix)[:3, :3],
                    s2 * np.linalg.inv(design.T @ design), rtol=1e-6)


def test_missing_drop_collinear_and_categorical_designs_reduce_to_the_plain_fit():
    frame, options = DESIGNS["armax_ml"]
    base = oe.arima(data=frame, time="t", **options)
    # Leading and trailing incomplete rows are dropped; the fit is that of the complete rows.
    holes = frame.copy()
    holes.loc[[0, 1], "x1"] = np.nan
    holes.loc[len(frame) - 1, "y"] = np.nan
    dropped = oe.arima(data=holes, time="t", missing="drop", **options)
    inner = oe.arima(data=frame.iloc[2:-1], time="t", **options)
    assert_allclose(estimates(dropped), estimates(inner), rtol=1e-12)
    assert dropped.nobs == len(frame) - 3 and dropped.dropped_rows == 3
    assert dropped.sample_positions == list(range(2, len(frame) - 1))
    # An exactly collinear regressor is omitted (left to right) and recorded.
    collinear = oe.arima(data=frame.assign(x3=frame.x1 * 2.0 - 1.0), y="y",
                         x=["x1", "x2", "x3"], order=(1, 0, 1), time="t")
    assert_allclose(estimates(collinear), estimates(base), rtol=1e-9)
    assert collinear.provenance["omitted_terms"] == ["x3"]
    assert any("x3" in warning for warning in collinear.warnings)
    # A linear trend is a constant after differencing: omitted next to the drift.
    frame, options = DESIGNS["arima111_drift_x"]
    trend = oe.arima(data=frame.assign(trend=np.arange(len(frame), dtype=float)), y="y",
                     x=["x1", "trend"], order=(1, 1, 1), time="t")
    assert trend.provenance["omitted_terms"] == ["trend"]
    assert_allclose(estimates(trend), estimates(oe.arima(data=frame, time="t", **options)),
                    rtol=1e-9)
    # A categorical regressor equals its explicit indicators.
    frame, options = DESIGNS["categorical_ma1_css"]
    coded = oe.arima(data=frame, time="t", **options)
    dummies = frame.assign(gb=(frame.g == "b").astype(float), gc=(frame.g == "c").astype(float))
    explicit = oe.arima(data=dummies, y="y", x=["gb", "gc", "x1"], order=(0, 0, 1), method="css",
                        time="t")
    assert_allclose(estimates(coded), estimates(explicit), rtol=1e-10)
    assert [c.term for c in coded.coefficients][:4] == ["Intercept", "g[b]", "g[c]", "x1"]
    future = pd.DataFrame({"g": ["c", "a"], "x1": [0.5, -1.0]})
    assert_allclose(
        oe.forecast(coded, 2, exog=future)["forecast"],
        oe.forecast(explicit, 2, exog=pd.DataFrame({"gb": [0.0, 0.0], "gc": [1.0, 0.0],
                                                    "x1": [0.5, -1.0]}))["forecast"], rtol=1e-10)


def test_alpha_changes_only_the_intervals():
    frame, options = DESIGNS["ar2_ml"]
    wide = oe.arima(data=frame, time="t", alpha=0.01, **options)
    base = oe.arima(data=frame, time="t", **options)
    assert_allclose(estimates(wide), estimates(base), rtol=1e-13)
    critical = stats.norm.ppf(0.995)
    assert_allclose([c.ci_high for c in wide.coefficients],
                    estimates(wide) + critical * std_errors(wide), rtol=1e-10)
    assert wide.inference["confidence_level"] == pytest.approx(0.99)


# ---- series diagnostics from the textbook formulas ---------------------------------------------


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(77)
    n = 141
    values = simulate(n, [0.6, -0.2], [0.3], seed=31) * (1 + 0.5 * rng.normal(size=n) ** 2) + 4.0
    return pd.DataFrame({"when": np.arange(10, 10 + n), "z": values})


def test_corrgram_against_direct_linear_algebra(series):
    z = series.z.to_numpy()
    n = len(z)
    lags = 9
    centered = z - z.mean()
    r = np.array([centered[k:] @ centered[:n - k] for k in range(lags + 1)]) / (centered @ centered)
    # Partial autocorrelation k: last coefficient of the Yule-Walker system of order k.
    yule_walker = [np.linalg.solve(r[np.abs(np.subtract.outer(np.arange(k), np.arange(k)))],
                                   r[1:k + 1])[-1] for k in range(1, lags + 1)]
    q = n * (n + 2) * np.cumsum(r[1:] ** 2 / (n - np.arange(1, lags + 1)))
    table = oe.corrgram(series, "z", lags=lags, time="when")
    assert table["lag"].tolist() == list(range(1, lags + 1))
    assert_allclose(table["acf"], r[1:], rtol=1e-12)
    assert_allclose(table["pacf"], yule_walker, rtol=1e-9)
    assert_allclose(table["q"], q, rtol=1e-12)
    assert_allclose(table["p_value"], stats.chi2.sf(q, np.arange(1, lags + 1)), rtol=1e-8)
    assert table.attrs["nobs"] == n
    assert table.attrs["white_noise_band"] == pytest.approx(stats.norm.ppf(0.975) / math.sqrt(n))
    # Regression-based partial autocorrelations (Stata's default for corrgram / pac).
    regression = []
    for k in range(1, lags + 1):
        design = np.column_stack([np.ones(n - k)] + [z[k - j:n - j] for j in range(1, k + 1)])
        regression.append(np.linalg.lstsq(design, z[k:], rcond=None)[0][-1])
    other = oe.corrgram(series, "z", lags=lags, pacf="regression")
    assert_allclose(other["pacf"], regression, rtol=1e-8)
    assert_allclose(other["acf"], table["acf"], rtol=1e-14)
    # Default number of lags: min(floor(n/2) - 2, 40); the row order is irrelevant with time.
    assert len(oe.corrgram(series, "z")) == min(n // 2 - 2, 40)
    assert len(oe.corrgram(series.iloc[:30], "z")) == 13
    shuffled = oe.corrgram(series.sample(frac=1.0, random_state=1), "z", lags=lags, time="when")
    assert_allclose(shuffled["acf"], table["acf"], rtol=1e-13)
    # A location shift and a rescaling change nothing.
    moved = oe.corrgram(series.assign(z=series.z * 1e6 + 1e9), "z", lags=lags)
    assert_allclose(moved["acf"], table["acf"], rtol=1e-6)


def test_portmanteau_jarque_bera_and_arch_lm_against_their_formulas(series):
    z = series.z.to_numpy()
    n = len(z)
    centered = z - z.mean()
    for lags in (1, 7, None):
        m = min(n // 2 - 2, 40) if lags is None else lags
        r = np.array([centered[k:] @ centered[:n - k] for k in range(1, m + 1)]) \
            / (centered @ centered)
        q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, m + 1)))
        test = oe.wntestq(series, "z", lags=lags).attrs
        assert test["statistic"] == pytest.approx(q, rel=1e-12)
        assert test["df"] == m and test["lags"] == m
        assert test["p_value"] == pytest.approx(stats.chi2.sf(q, m), rel=1e-8, abs=1e-300)
    m2, m3, m4 = (np.mean(centered ** j) for j in (2, 3, 4))
    skewness, kurtosis = m3 / m2 ** 1.5, m4 / m2 ** 2
    jb = n / 6 * (skewness ** 2 + (kurtosis - 3) ** 2 / 4)
    test = oe.jarque_bera(series, "z").attrs
    assert test["statistic"] == pytest.approx(jb, rel=1e-12)
    assert test["skewness"] == pytest.approx(skewness, rel=1e-12)
    assert test["kurtosis"] == pytest.approx(kurtosis, rel=1e-12)
    assert test["p_value"] == pytest.approx(math.exp(-jb / 2), rel=1e-9) and test["df"] == 2
    assert test["statistic"] == pytest.approx(stats.jarque_bera(z).statistic, rel=1e-10)
    for demean in (False, True):
        u = centered if demean else z
        squares = u ** 2
        rows = []
        for p in (1, 3, 6):
            design = np.column_stack([np.ones(n - p)]
                                     + [squares[p - j:n - j] for j in range(1, p + 1)])
            target = squares[p:]
            resid = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
            r2 = 1 - resid @ resid / np.sum((target - target.mean()) ** 2)
            rows.append(((n - p) * r2, p, stats.chi2.sf((n - p) * r2, p), r2, n - p))
        table = oe.archlm(series, "z", lags=[1, 3, 6], demean=demean)
        assert_allclose(table["statistic"], [row[0] for row in rows], rtol=1e-8)
        assert table["df"].tolist() == [1, 3, 6] and table["nobs"].tolist() == [n - 1, n - 3, n - 6]
        assert_allclose(table["p_value"], [row[2] for row in rows], rtol=1e-7)
        assert_allclose(table["r_squared"], [row[3] for row in rows], rtol=1e-8)
        single = oe.archlm(series, "z", lags=3, demean=demean)
        assert single.attrs["statistic"] == pytest.approx(rows[1][0], rel=1e-8)
    # Multiplying the series by a constant leaves every statistic unchanged.
    scaled = series.assign(z=series.z * 1e-7)
    assert oe.archlm(scaled, "z", lags=2).attrs["statistic"] == pytest.approx(
        oe.archlm(series, "z", lags=2).attrs["statistic"], rel=1e-8)
    assert oe.jarque_bera(scaled, "z").attrs["statistic"] == pytest.approx(jb, rel=1e-10)


# ---- exponential smoothing from explicit recursions -------------------------------------------


def _line(x):
    half = max(2, len(x) // 2)
    slope, intercept = np.polyfit(np.arange(1, half + 1), x[:half], 1)
    return intercept, slope


def simple_smoothing(x, alpha):
    level = x[:max(1, len(x) // 2)].mean()
    forecast, smoothed = np.empty(len(x)), np.empty(len(x))
    for t, value in enumerate(x):
        forecast[t] = level
        level = alpha * value + (1 - alpha) * level
        smoothed[t] = level
    return forecast, smoothed, [level] * 3


def brown_smoothing(x, alpha):
    """Brown's double smoothing in its original two-smoother form (Stata's dexponential)."""
    intercept, slope = _line(x)
    s1 = intercept - (1 - alpha) / alpha * slope
    s2 = intercept - 2 * (1 - alpha) / alpha * slope
    forecast, smoothed = np.empty(len(x)), np.empty(len(x))
    ratio = alpha / (1 - alpha)
    for t, value in enumerate(x):
        forecast[t] = (2 + ratio) * s1 - (1 + ratio) * s2
        s1 = alpha * value + (1 - alpha) * s1
        s2 = alpha * s1 + (1 - alpha) * s2
        smoothed[t] = 2 * s1 - s2
    return forecast, smoothed, [(2 + h * ratio) * s1 - (1 + h * ratio) * s2 for h in (1, 2, 3)]


def holt_smoothing(x, alpha, beta):
    level, trend = _line(x)
    forecast, smoothed = np.empty(len(x)), np.empty(len(x))
    for t, value in enumerate(x):
        forecast[t] = level + trend
        new = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (new - level) + (1 - beta) * trend
        level = smoothed[t] = new
    return forecast, smoothed, [level + h * trend for h in (1, 2, 3)]


def winters_smoothing(x, m, alpha, beta, gamma, additive):
    n = len(x)
    seasons = max(2, min(n // m, (n // 2) // m))
    block = x[:seasons * m].reshape(seasons, m)
    means = block.mean(axis=1)
    trend = (means[-1] - means[0]) / ((seasons - 1) * m)
    level = means[0] - trend * (m + 1) / 2
    line = level + trend * np.arange(1, seasons * m + 1).reshape(seasons, m)
    if additive:
        seasonal = (block - line).mean(axis=0)
        seasonal = list(seasonal - seasonal.mean())
    else:
        seasonal = (block / line).mean(axis=0)
        seasonal = list(seasonal / seasonal.mean())
    forecast, smoothed = np.empty(n), np.empty(n)
    for t, value in enumerate(x):
        index = seasonal[t % m]
        base = level + trend
        forecast[t] = base + index if additive else base * index
        new = alpha * ((value - index) if additive else value / index) + (1 - alpha) * base
        trend = beta * (new - level) + (1 - beta) * trend
        level = new
        seasonal[t % m] = gamma * ((value - level) if additive else value / level) \
            + (1 - gamma) * index
        smoothed[t] = level + seasonal[t % m] if additive else level * seasonal[t % m]
    ahead = [(level + h * trend) + seasonal[(n + h - 1) % m] if additive
             else (level + h * trend) * seasonal[(n + h - 1) % m] for h in (1, 2, 3)]
    return forecast, smoothed, ahead


@pytest.fixture(scope="module")
def sales():
    rng = np.random.default_rng(3)
    t = np.arange(96)
    values = 20 + 0.3 * t + 4 * np.sin(2 * np.pi * t / 12) + rng.normal(size=96)
    return pd.DataFrame({"t": t, "x": values})


SMOOTHERS = [
    ("exponential", {"alpha": 0.3}, {}),
    ("exponential", {"alpha": 1.0}, {}),
    ("exponential", {"alpha": 0.0}, {}),
    ("dexponential", {"alpha": 0.35}, {}),
    ("hwinters", {"alpha": 0.3, "beta": 0.2}, {}),
    ("hwinters", {"alpha": 0.0, "beta": 0.5}, {}),
    ("hwinters", {"alpha": 1.0, "beta": 1.0}, {}),
    ("shwinters", {"alpha": 0.3, "beta": 0.2, "gamma": 0.4}, {"period": 12}),
    ("shwinters", {"alpha": 1.0, "beta": 0.3, "gamma": 0.5}, {"period": 12}),
    ("shwinters", {"alpha": 0.2, "beta": 0.0, "gamma": 0.0}, {"period": 12}),
    ("shwinters", {"alpha": 0.3, "beta": 0.2, "gamma": 0.4}, {"period": 12, "additive": False}),
    ("shwinters", {"alpha": 0.2, "beta": 0.1, "gamma": 1.0}, {"period": 4, "additive": False}),
]


def _explicit(method, x, parameters, options):
    if method == "exponential":
        return simple_smoothing(x, **parameters)
    if method == "dexponential":
        return brown_smoothing(x, **parameters)
    if method == "hwinters":
        return holt_smoothing(x, **parameters)
    return winters_smoothing(x, options["period"], parameters["alpha"], parameters["beta"],
                             parameters["gamma"], options.get("additive", True))


@pytest.mark.parametrize("method, parameters, options", SMOOTHERS)
def test_smoothing_recursions_with_fixed_parameters(sales, method, parameters, options):
    x = sales.x.to_numpy()
    n = len(x)
    forecast, smoothed, ahead = _explicit(method, x, parameters, options)
    table = oe.tssmooth(data=sales, y="x", method=method, forecast=3, time="t", **parameters,
                        **options)
    assert len(table) == n + 3 and list(table.columns) == ["period", "observed", "smoothed",
                                                           "forecast"]
    assert_allclose(table["forecast"][:n], forecast, rtol=1e-10, atol=1e-10)
    assert_allclose(table["smoothed"][:n], smoothed, rtol=1e-10, atol=1e-10)
    assert_allclose(table["forecast"][n:], ahead, rtol=1e-10)
    assert table["observed"][n:].isna().all() and table["smoothed"][n:].isna().all()
    assert table["period"].tolist() == list(range(n + 3))
    sse = float(np.sum((x - forecast) ** 2))
    assert table.attrs["sse"] == pytest.approx(sse, rel=1e-10)
    assert table.attrs["rmse"] == pytest.approx(math.sqrt(sse / n), rel=1e-10)
    assert table.attrs["parameters"] == parameters and table.attrs["estimated"] == []
    # Scale equivariance of the recursions.
    scaled = oe.tssmooth(data=sales.assign(x=sales.x * 1e6), y="x", method=method, forecast=3,
                         **parameters, **options)
    assert_allclose(scaled["forecast"], table["forecast"] * 1e6, rtol=1e-9)


@pytest.mark.parametrize("method, options, size", [
    ("exponential", {}, 1), ("dexponential", {}, 1), ("hwinters", {}, 2),
    ("shwinters", {"period": 12}, 3), ("shwinters", {"period": 12, "additive": False}, 3),
])
def test_estimated_smoothing_parameters_minimize_the_explicit_sse(sales, method, options, size):
    # A noisier series than the fixture keeps the optimum of most smoothers interior.
    rng = np.random.default_rng(11)
    t = np.arange(120)
    x = 50 + np.cumsum(rng.normal(0.2, 1.0, size=120)) + 5 * np.sin(2 * np.pi * t / 12) \
        + rng.normal(size=120)
    frame = pd.DataFrame({"x": x})
    names = ["alpha", "beta", "gamma"][:size]

    def sse(point):
        point = np.clip(point, 1e-10 if method == "dexponential" else 0.0,
                        1 - 1e-10 if method == "dexponential" else 1.0)
        forecast = _explicit(method, x, dict(zip(names, point)), options)[0]
        value = float(np.sum((x - forecast) ** 2))
        return value if np.isfinite(value) else 1e300

    best = None
    for start in np.array(np.meshgrid(*[[0.15, 0.5, 0.85]] * size)).T.reshape(-1, size):
        trial = minimize(sse, start, bounds=[(0.0, 1.0)] * size, method="L-BFGS-B",
                         options={"ftol": 1e-15, "gtol": 1e-10})
        if best is None or trial.fun < best.fun:
            best = trial
    table = oe.tssmooth(data=frame, y="x", method=method, **options)
    assert table.attrs["estimated"] == names
    assert table.attrs["sse"] <= best.fun * (1 + 1e-9)
    assert table.attrs["sse"] == pytest.approx(sse([table.attrs["parameters"][name]
                                                    for name in names]), rel=1e-9)
    identified = [name for name in names if name not in table.attrs["not_identified"]]
    if abs(table.attrs["sse"] - best.fun) <= 1e-8 * best.fun:
        assert_allclose([table.attrs["parameters"][name] for name in identified],
                        [best.x[names.index(name)] for name in identified], atol=2e-4)
    assert set(table.attrs["at_bounds"]) == {
        name for name in names if table.attrs["parameters"][name] in (0.0, 1.0)
        and name not in table.attrs["not_identified"]}
