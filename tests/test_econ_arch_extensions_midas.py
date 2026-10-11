"""GARCH-MIDAS (``oe.garch_midas``) against an explicit NumPy transcription of mfGARCH.

The R package mfGARCH 0.2.2 (Kleen; Conrad and Kleen 2020), the authors-adjacent
implementation of Engle, Ghysels and Sohn (2013), is not available in the oracle
environment, so its likelihood is transcribed here line by line from the CRAN source
(github.com/cran/mfGARCH: R/helper_functions.R ``calculate_phi``, ``calculate_tau_mf``,
``llh_mf``; src/sum_tau.cpp ``sum_tau``; src/calculate_g.cpp ``calculate_g``), including
the burn-in of the first K periods and the period merge. mfGARCH starts the short-run
recursion at ``g_zero = var(y)``; ``oe.garch_midas`` at the unconditional mean 1, so the
oracle is called with ``g_zero = 1``. The transcription shares no code with the Torch
kernel (loops over days, no filters).
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.arch.midas import MidasLayout, MidasLikelihood, beta_weights
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle


# ---- mfGARCH, transcribed --------------------------------------------------------------------


def calculate_phi(w1, w2, K):
    weights = np.array([(j / (K + 1)) ** (w1 - 1) * (1 - j / (K + 1)) ** (w2 - 1)
                        for j in range(1, K + 1)])
    return weights / weights.sum()


def sum_tau(m, theta, phivar, covariate, K):
    n = len(covariate) - K
    exponential = np.empty(n, dtype=complex)
    for i in range(1, n + 1):
        exponential[i - 1] = m
        for j in range(1, K + 1):
            exponential[i - 1] += theta * phivar[j - 1] * covariate[K + i - 1 - j]
    return exponential


def calculate_g(omega, alpha, beta, gamma, returns, g0):
    g = np.empty(len(returns), dtype=complex)
    g[0] = g0
    for i in range(1, len(returns)):
        if np.real(returns[i - 1]) >= 0:
            g[i] = omega + alpha * returns[i - 1] ** 2 + beta * g[i - 1]
        else:
            g[i] = omega + alpha * returns[i - 1] ** 2 + gamma * returns[i - 1] ** 2 \
                + beta * g[i - 1]
    return g


def llh_mf(df, x, y, low_freq, mu, omega, alpha, beta, gamma, m, theta, w1, w2, g_zero, K):
    """Negative per-day log likelihood of the days after the burn-in (mfGARCH llh_mf)."""
    phi = calculate_phi(w1, w2, K)
    tau_periods = np.r_[np.full(K, np.nan), np.exp(sum_tau(m, theta, phi, x, K))]
    periods = pd.unique(df[low_freq])
    lookup = dict(zip(periods, tau_periods, strict=True))
    tau = np.array([lookup[value] for value in df[low_freq]])
    keep = ~np.isnan(np.real(tau))
    ret, tau = np.asarray(y)[keep], tau[keep]
    g = calculate_g(omega, alpha, beta, gamma, (ret - mu) / np.sqrt(tau), g_zero)
    return 0.5 * np.log(2 * np.pi) + 0.5 * np.log(g * tau) + 0.5 * (ret - mu) ** 2 / (g * tau)


def oracle(df, params, *, K, weighting="beta_restricted", asymmetric=True, mean=True,
           covariate="macro"):
    names = MidasLayout(K, weighting, asymmetric, mean).names
    p = {"mu": 0.0, "gamma": 0.0, "w1": 1.0, "w2": 1.0, **dict(zip(names, params, strict=True))}
    if covariate is None:
        x = df.groupby("month", sort=False)["ret"].apply(lambda r: np.sum(r ** 2)).to_numpy()
    else:
        x = df.groupby("month", sort=False)[covariate].first().to_numpy()
    omega = 1 - p["alpha"] - p["beta"] - p["gamma"] / 2
    return -llh_mf(df, x, df["ret"].to_numpy(), "month", p["mu"], omega, p["alpha"], p["beta"],
                   p["gamma"], p["m"], p["theta"], p["w1"], p["w2"], 1.0, K)


# ---- data -----------------------------------------------------------------------------------


def simulate(n_periods=90, days=21, K=12, seed=5, mu=0.05, alpha=0.06, beta=0.88, gamma=0.08,
             m=0.0, theta=0.4, w2=4.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_periods)
    for s in range(1, n_periods):
        x[s] = 0.8 * x[s - 1] + rng.normal()
    phi = calculate_phi(1.0, w2, K)
    rows, g, u = [], 1.0, 0.0
    omega = 1 - alpha - beta - gamma / 2
    for s in range(n_periods):
        tau = math.exp(m + theta * sum(phi[k - 1] * x[s - k] for k in range(1, K + 1)
                                       if s - k >= 0))
        for _ in range(days):
            g = omega + (alpha + gamma * (u < 0)) * u ** 2 + beta * g
            r = mu + math.sqrt(tau * g) * rng.normal()
            u = (r - mu) / math.sqrt(tau)
            rows.append((s + 2000, x[s], r))
    return pd.DataFrame(rows, columns=["month", "macro", "ret"])


DATA = simulate()
TRUTH = {"Intercept": 0.05, "GARCH:alpha": 0.06, "GARCH:gamma": 0.08, "GARCH:beta": 0.88,
         "MIDAS:m": 0.0, "MIDAS:theta": 0.4, "MIDAS:w2": 4.0}


def _likelihood(df, layout, covariate="macro"):
    y = torch.tensor(df["ret"].to_numpy())
    codes = torch.tensor(pd.factorize(df["month"])[0], dtype=torch.int64)
    if covariate is None:
        x = torch.zeros(int(codes.max()) + 1, dtype=torch.float64).index_add_(0, codes, y ** 2)
    else:
        x = torch.tensor(df.groupby("month", sort=False)[covariate].first().to_numpy())
    return MidasLikelihood(y, codes, x, layout)


CASES = [
    dict(K=12, weighting="beta_restricted", asymmetric=True, mean=True, covariate="macro",
         params=[0.04, 0.05, 0.07, 0.86, 0.1, 0.3, 3.0]),
    dict(K=6, weighting="beta_unrestricted", asymmetric=False, mean=True, covariate=None,
         params=[0.04, 0.08, 0.85, -0.5, 0.05, 1.4, 2.5]),
    dict(K=1, weighting="beta_restricted", asymmetric=True, mean=False, covariate="macro",
         params=[0.05, 0.06, 0.87, 0.0, -0.2]),
    dict(K=3, weighting="beta_restricted", asymmetric=False, mean=False, covariate=None,
         params=[0.07, 0.9, -0.3, 0.02, 0.8]),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: f"K{case['K']}_{case['weighting']}")
def test_likelihood_matches_the_mfgarch_transcription(case):
    layout = MidasLayout(case["K"], case["weighting"], case["asymmetric"], case["mean"])
    like = _likelihood(DATA, layout, case["covariate"])
    theta = torch.tensor(case["params"], dtype=torch.float64)
    out = like.evaluate(theta, scores=True)
    expected = np.real(oracle(DATA, case["params"], K=case["K"], weighting=case["weighting"],
                              asymmetric=case["asymmetric"], mean=case["mean"],
                              covariate=case["covariate"]))
    assert like.n == len(expected) == len(DATA) - case["K"] * 21
    assert_allclose(out.loglik.numpy(), expected, rtol=0, atol=1e-11)

    def objective(point):
        evaluation = like.evaluate(point)
        if evaluation is None:
            return torch.tensor(math.nan), torch.full_like(point, math.nan)
        return evaluation.value, evaluation.gradient

    assert check_derivatives(objective, theta)["gradient_max_rel_error"] < 1e-6
    # Per-observation scores: complex-step derivatives of the transcription (exact).
    exact = np.empty((like.n, len(case["params"])))
    for k in range(len(case["params"])):
        shifted = np.array(case["params"], dtype=complex)
        shifted[k] += 1e-20j
        exact[:, k] = np.imag(oracle(DATA, shifted, K=case["K"], weighting=case["weighting"],
                                     asymmetric=case["asymmetric"], mean=case["mean"],
                                     covariate=case["covariate"])) / 1e-20
    assert_allclose(out.scores.numpy(), exact, rtol=1e-8, atol=1e-10)


def test_beta_weights_and_derivatives():
    for w1, w2, K in [(1.0, 4.0, 12), (1.7, 2.5, 24), (0.8, 1.2, 5)]:
        phi, d_phi = beta_weights(K, w1, w2, derivatives=True)
        assert_allclose(phi.numpy(), calculate_phi(w1, w2, K), rtol=1e-13)
        assert_allclose(d_phi[0].numpy(), approx_fprime(np.array([w1]), lambda v: calculate_phi(
            v[0], w2, K), centered=True)[:, 0], rtol=1e-6, atol=1e-10)
        assert_allclose(d_phi[1].numpy(), approx_fprime(np.array([w2]), lambda v: calculate_phi(
            w1, v[0], K), centered=True)[:, 0], rtol=1e-6, atol=1e-10)


@pytest.fixture(scope="module")
def fits():
    return {kind: oe.garch_midas(data=DATA, y="ret", period="month", covariate="macro", lags=12,
                                 covariance=kind) for kind in ("robust", "opg", "nonrobust")}


def test_maximum_likelihood_and_full_covariance_match_the_transcription(fits):
    def negative(params):
        if params[1] + params[3] + params[2] / 2 >= 1 or params[6] <= 0:
            return 1e10
        value = -np.real(oracle(DATA, params, K=12)).sum()
        return value if np.isfinite(value) else 1e10

    # mfGARCH's own starting values, then an independent optimizer.
    found = minimize(negative, [0.0, 0.02, 0.04, 0.85, 0.0, 0.0, 3.0], method="Nelder-Mead",
                     options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 40000, "maxfev": 40000})
    found = minimize(negative, found.x, method="BFGS", options={"gtol": 1e-7})
    fit = fits["nonrobust"]
    estimates = np.array([c.estimate for c in fit.coefficients])
    assert fit.metrics["log_likelihood"] >= -found.fun - 1e-8
    assert_allclose(estimates, found.x, rtol=2e-5, atol=2e-6)
    # Hessian: central differences of complex-step gradients of the transcription.
    k = len(estimates)

    def gradient(point):
        out = np.empty(k)
        for j in range(k):
            shifted = point.astype(complex)
            shifted[j] += 1e-20j
            out[j] = np.imag(oracle(DATA, shifted, K=12).sum()) / 1e-20
        return out

    hessian = np.empty((k, k))
    for j in range(k):
        step = 1e-5 * max(abs(estimates[j]), 1e-2)
        up, down = estimates.copy(), estimates.copy()
        up[j] += step
        down[j] -= step
        hessian[:, j] = (gradient(up) - gradient(down)) / (2 * step)
    hessian = 0.5 * (hessian + hessian.T)
    bread = np.linalg.inv(-hessian)
    assert_allclose(np.array(fit.covariance_matrix), bread, rtol=1e-5,
                    atol=1e-6 * np.abs(bread).max())
    scores = np.empty((fits["opg"].nobs, k))
    for j in range(k):
        shifted = estimates.astype(complex)
        shifted[j] += 1e-20j
        scores[:, j] = np.imag(oracle(DATA, shifted, K=12)) / 1e-20
    assert_allclose(np.array(fits["opg"].covariance_matrix), np.linalg.inv(scores.T @ scores),
                    rtol=1e-6)
    n = len(scores)
    sandwich = n / (n - 1) * bread @ scores.T @ scores @ bread
    assert_allclose(np.array(fits["robust"].covariance_matrix), sandwich, rtol=1e-5,
                    atol=1e-6 * np.abs(sandwich).max())
    # The seeded simulation is recovered within three standard errors.
    for coefficient in fit.coefficients:
        assert abs(coefficient.estimate - TRUTH[coefficient.term]) < 3 * coefficient.std_error


def test_reported_quantities_forecasts_and_round_trip(fits):
    fit = fits["robust"]
    assert fit.nobs == len(DATA) - 12 * 21
    model = fit.extra["model"]
    assert model["burn_in_observations"] == 12 * 21 and model["periods"] == 90
    est = {c.term: c.estimate for c in fit.coefficients}
    a, b, gamma = est["GARCH:alpha"], est["GARCH:beta"], est["GARCH:gamma"]
    assert fit.metrics["persistence"] == pytest.approx(a + b + gamma / 2)
    assert_allclose(fit.extra["weights"], calculate_phi(1.0, est["MIDAS:w2"], 12), rtol=1e-12)
    # mfGARCH's variance ratio from the fitted components, recomputed with pandas.
    x = DATA.groupby("month", sort=False)["macro"].first().to_numpy()
    tau_periods = np.exp(np.real(sum_tau(est["MIDAS:m"], est["MIDAS:theta"],
                                         calculate_phi(1, est["MIDAS:w2"], 12), x, 12)))
    frame = DATA.iloc[12 * 21:].copy()
    frame["tau"] = np.repeat(tau_periods, 21)
    g = np.real(calculate_g(1 - a - b - gamma / 2, a, b, gamma,
                            (frame["ret"].to_numpy() - est["Intercept"])
                            / np.sqrt(frame["tau"].to_numpy()), 1.0))
    frame["h"] = frame["tau"] * g
    grouped = frame.groupby("month")
    ratio = 100 * np.var(np.log(grouped["tau"].mean()), ddof=1) \
        / np.var(np.log(grouped["h"].mean()), ddof=1)
    assert fit.metrics["variance_ratio"] == pytest.approx(ratio, rel=1e-10)
    # Forecasts: tau of the next period from the last 12 covariate values; g by its
    # recursion, then geometric reversion to 1.
    tau_next = math.exp(est["MIDAS:m"] + est["MIDAS:theta"]
                        * calculate_phi(1, est["MIDAS:w2"], 12) @ x[::-1][:12])
    e = frame["ret"].iloc[-1] - est["Intercept"]
    u = e / math.sqrt(frame["tau"].iloc[-1])
    rho = a + b + gamma / 2
    first = (1 - rho) + (a + gamma * (u < 0)) * u * u + b * g[-1]
    expected = [tau_next * (1 + rho ** k * (first - 1)) for k in range(6)]
    table = oe.forecast(fit, 6)
    assert_allclose(table["variance_forecast"].to_numpy(), expected, rtol=1e-10)
    same = oe.forecast(fit, 2, new_period=False)
    assert same["long_run"].iloc[0] == pytest.approx(frame["tau"].iloc[-1], rel=1e-12)
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    assert_allclose(oe.forecast(restored, 6)["variance_forecast"].to_numpy(), expected,
                    rtol=1e-10)
    assert "MIDAS:theta" in fit.summary() and "MIDAS:w2" in fit.to_latex()
    assert fit.provenance["stata_parity_validated"] is False


def test_realized_variance_covariate_and_time_sorting():
    shuffled = DATA.assign(day=np.arange(len(DATA))).sample(frac=1.0, random_state=3)
    fit = oe.garch_midas(data=shuffled, y="ret", period="month", time="day", lags=6,
                         asymmetric=False)
    ordered = oe.garch_midas(data=DATA, y="ret", period="month", lags=6, asymmetric=False)
    assert_allclose([c.estimate for c in fit.coefficients],
                    [c.estimate for c in ordered.coefficients], rtol=1e-10)
    assert ordered.extra["model"]["covariate"].startswith("realized variance")
    params = [c.estimate for c in ordered.coefficients]
    assert ordered.metrics["log_likelihood"] == pytest.approx(
        np.real(oracle(DATA, params, K=6, asymmetric=False, covariate=None)).sum(), rel=1e-12)


def test_refusals():
    base = dict(data=DATA, y="ret", period="month", covariate="macro")
    cases = [
        (dict(lags=2, weighting="beta_unrestricted"), "unsupported_combination"),
        (dict(lags=89), "insufficient_observations"),
        (dict(lags=0), "invalid_spec"),
        (dict(weighting="almon"), "invalid_spec"),
    ]
    for keywords, code in cases:
        with pytest.raises(AnalysisError) as error:
            oe.garch_midas(**{**base, **keywords})
        assert error.value.code == code, keywords
    broken = DATA.copy()
    broken.loc[5, "macro"] += 1.0
    with pytest.raises(AnalysisError) as error:
        oe.garch_midas(data=broken, y="ret", period="month", covariate="macro")
    assert error.value.code == "invalid_covariate"
    for frame in (DATA.assign(month=DATA["month"] * 2), DATA.iloc[::-1]):
        with pytest.raises(AnalysisError) as error:
            oe.garch_midas(data=frame, y="ret", period="month", covariate="macro")
        assert error.value.code == "invalid_period"
    with pytest.raises(AnalysisError) as error:
        oe.garch_midas(data=DATA.assign(month=pd.to_datetime("2020-01-01")
                                        + pd.to_timedelta(DATA["month"], unit="D")),
                       y="ret", period="month")
    assert error.value.code == "invalid_period"
    holes = DATA.copy()
    holes.loc[700, "ret"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.garch_midas(data=holes, y="ret", period="month", missing="drop")
    assert error.value.code == "time_gaps"
    with pytest.raises(AnalysisError) as error:
        oe.garch_midas(data=Dataset.from_frame(DATA), y="ret", period="month")
    assert error.value.code == "streaming_unsupported"
    fit = oe.garch_midas(data=DATA, y="ret", period="month", lags=1)
    assert [c.term for c in fit.coefficients][-2:] == ["MIDAS:m", "MIDAS:theta"]
    with pytest.raises(AnalysisError) as error:
        oe.forecast(fit, 0)
    assert error.value.code == "invalid_steps"
    tampered = ResultBundle.model_validate_json(fit.model_dump_json())
    tampered.extra["state"]["covariate_tail"] = []
    with pytest.raises(AnalysisError) as error:
        oe.forecast(tampered, 3)
    assert error.value.code == "invalid_result"
