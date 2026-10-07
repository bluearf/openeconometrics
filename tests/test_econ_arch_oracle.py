"""Independent likelihood oracle for the ARCH family and kernel-level checks.

``oracle_loglike`` re-implements every model of ``oe.arch`` as an explicit loop over
time on Python floats, with SciPy's densities (``scipy.stats.norm``, ``t``,
``gennorm``). It shares no code with the tensor kernels (``inverse_filter``, the
parallel prefix scan, the analytic scores). The tests here compare the kernel's log
likelihood, per-observation contributions, analytic gradient and per-observation
scores with it at arbitrary parameter points, for every variance model, innovation
distribution, ARMA structure, ARCH-in-mean form and variance regressor.

The other ``test_econ_arch_*`` files import the oracle from this module.
"""

import math
import re

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import integrate, stats
from scipy.signal import lfilter
from scipy.special import gamma as gamma_function
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tsa.statespace.sarimax import SARIMAX

from openecon.econometrics.arch import densities
from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arch.recursions import (
    adjoint_filter, lagged, lead, solve_recurrence,
)
from openecon.econometrics.arima.filters import inverse_filter
from openecon.engines.optimize import check_derivatives

ABS_NORMAL = math.sqrt(2.0 / math.pi)


# ---- the independent oracle -----------------------------------------------------------------


def unpack(terms, values, model):
    """Parameters by NAME (the reported term names), independent of the kernel's layout."""
    p = {"beta": [], "psi": 0.0, "ar": {}, "ma": {}, "het": [], "het_const": None, "omega": None,
         "a": {}, "g": {}, "b": {}, "power": 2.0, "tau": None}
    for term, value in zip(terms, values, strict=True):
        value = float(value)
        lag = re.match(r"^(ARMA|ARCH):L(\d+)\.(\w+)$", term)
        if lag:
            number, name = int(lag.group(2)), lag.group(3)
            key = {"ar": "ar", "ma": "ma", "arch": "a", "earch": "a", "parch": "a",
                   "tarch": "g", "earch_a": "g", "garch": "b", "egarch": "b", "pgarch": "b"}[name]
            p[key][number] = value
        elif term.startswith("ARCHM:"):
            p["psi"] = value
        elif term == "HET:Intercept":
            p["het_const"] = value
        elif term.startswith("HET:"):
            p["het"].append(value)
        elif term == "ARCH:Intercept":
            p["omega"] = value
        elif term == "POWER:power":
            p["power"] = value
        elif term in {"/lndfm2", "/lnshape"}:
            p["tau"] = value
        else:
            p["beta"].append(value)
    return p


def oracle_recursion(p, y, x, z=None, *, model="garch", archm=None, signs=None):
    """(e, h, u) of one model by explicit recursion on Python floats; None when h <= 0.

    ``signs`` (EGARCH) replaces |z_t| by ``signs[t] * z_t``: the smooth piece of the
    likelihood selected by a sign pattern (the likelihood itself has kinks at z_t = 0).
    """
    n = len(y)
    kind = {"arch": "garch", "igarch": "garch", "tarch": "gjr"}.get(model, model)
    mean = [float(v) for v in (y - x @ np.asarray(p["beta"]) if x.shape[1] else y)]
    ar, ma = p["ar"], p["ma"]
    # Stage 1: residuals of the mean equation with ARMA terms, no ARCH-in-mean (for h0).
    plain = []
    for t in range(n):
        value = mean[t]
        value -= sum(c * mean[t - j] for j, c in ar.items() if t - j >= 0)
        value -= sum(c * plain[t - k] for k, c in ma.items() if t - k >= 0)
        plain.append(value)
    h0 = sum(v * v for v in plain) / n
    phi = p["power"]
    e, u, h = [], [], []
    for t in range(n):
        if z is not None and z.shape[1]:
            index = p["het_const"] + float(z[t] @ np.asarray(p["het"]))
            w = index if kind == "egarch" else math.exp(index)
        else:
            w = p["omega"]
        if kind == "egarch":
            log_h = w
            for i, c in p["a"].items():
                if t - i >= 0:
                    log_h += c * e[t - i] / math.sqrt(h[t - i])
            for i, c in p["g"].items():
                if t - i >= 0:
                    size = abs(e[t - i]) if signs is None else signs[t - i] * e[t - i]
                    log_h += c * (size / math.sqrt(h[t - i]) - ABS_NORMAL)
            for j, c in p["b"].items():
                log_h += c * math.log(h[t - j] if t - j >= 0 else h0)
            if not -700 < log_h < 700:
                return None
            ht = math.exp(log_h)
        elif kind == "parch":
            s = w
            for i, c in p["a"].items():
                s += c * (abs(e[t - i]) ** phi if t - i >= 0 else h0 ** (phi / 2))
            for j, c in p["b"].items():
                s += c * (h[t - j] if t - j >= 0 else h0) ** (phi / 2)
            if s <= 0:
                return None
            ht = s ** (2 / phi)
        else:
            ht = w
            for i, c in p["a"].items():
                ht += c * (e[t - i] ** 2 if t - i >= 0 else h0)
            for i, c in p["g"].items():
                ht += c * ((e[t - i] ** 2 if e[t - i] < 0 else 0.0) if t - i >= 0 else h0 / 2)
            for j, c in p["b"].items():
                ht += c * (h[t - j] if t - j >= 0 else h0)
            if ht <= 0:
                return None
        in_mean = {None: 0.0, "variance": ht, "sd": math.sqrt(ht), "log": math.log(ht)}[archm]
        ut = mean[t] - p["psi"] * in_mean
        et = ut - sum(c * u[t - j] for j, c in ar.items() if t - j >= 0) \
            - sum(c * e[t - k] for k, c in ma.items() if t - k >= 0)
        e.append(et)
        u.append(ut)
        h.append(ht)
    return np.asarray(e), np.asarray(h), np.asarray(u)


def oracle_loglike(p, y, x, z=None, *, model="garch", dist="normal", archm=None, signs=None):
    """Per-observation log likelihood: the explicit recursion with SciPy's densities."""
    path = oracle_recursion(p, y, x, z, model=model, archm=archm, signs=signs)
    if path is None:
        return None
    e, h, _ = path
    sd = np.sqrt(h)
    if dist == "normal":
        return stats.norm.logpdf(e, scale=sd)
    if dist == "t":
        nu = 2.0 + math.exp(p["tau"])
        scale = sd * math.sqrt((nu - 2.0) / nu)             # t_nu scaled to variance h
        return stats.t.logpdf(e / scale, nu) - np.log(scale)
    shape = math.exp(p["tau"])
    alpha = math.sqrt(gamma_function(1 / shape) / gamma_function(3 / shape))   # unit variance
    return stats.gennorm.logpdf(e / sd, shape, scale=alpha) - np.log(sd)


def terms_of(layout, k_x, k_z):
    return layout.terms("y", ["Intercept", *[f"x{i}" for i in range(1, k_x)]][:k_x],
                        [f"HET:z{i}" for i in range(k_z)])[0]


# ---- data and cases -------------------------------------------------------------------------


def series(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = np.c_[np.ones(n), rng.normal(size=(n, 2))]
    z = rng.normal(size=(n, 1))
    shock = rng.normal(size=n) * (1 + 0.5 * np.abs(rng.normal(size=n)))
    y = 0.3 + x[:, 1:] @ [0.5, -0.2] + shock
    return y, x, z


# (layout keywords, parameter vector in the layout's order)
CASES = {
    "garch11": (dict(model="garch", arch_lags=(1,), garch_lags=(1,), k_x=3),
                [0.3, 0.5, -0.2, 0.1, 0.8, 0.2]),
    "arch2": (dict(model="arch", arch_lags=(1, 2), garch_lags=(), k_x=3),
              [0.3, 0.5, -0.2, 0.2, 0.1, 0.9]),
    "garch_lags_t": (dict(model="garch", dist="t", arch_lags=(1, 2), garch_lags=(1, 3), k_x=3),
                     [0.3, 0.5, -0.2, 0.1, 0.05, 0.5, 0.2, 0.3, 1.2]),
    "gjr_ged": (dict(model="gjr", dist="ged", arch_lags=(1,), garch_lags=(1,), k_x=3),
                [0.3, 0.5, -0.2, 0.05, 0.1, 0.8, 0.2, 0.3]),
    "tarch_alias": (dict(model="tarch", arch_lags=(1, 2), garch_lags=(1,), k_x=3),
                    [0.3, 0.5, -0.2, 0.05, 0.03, 0.1, 0.06, 0.6, 0.2]),
    "garch_arma": (dict(model="garch", arch_lags=(1,), garch_lags=(1,), ar_lags=(1, 3),
                        ma_lags=(1, 2), k_x=3),
                   [0.3, 0.5, -0.2, 0.3, -0.1, 0.4, 0.2, 0.1, 0.8, 0.2]),
    "garch_het": (dict(model="garch", arch_lags=(1,), garch_lags=(1,), k_x=3, k_z=1),
                  [0.3, 0.5, -0.2, 0.3, -1.0, 0.1, 0.8]),
    "igarch_full": (dict(model="igarch", arch_lags=(1, 2), garch_lags=(1, 2), k_x=3),
                    [0.3, 0.5, -0.2, 0.1, 0.05, 0.5, 0.35, 0.02]),
    "parch": (dict(model="parch", arch_lags=(1,), garch_lags=(1,), k_x=3),
              [0.3, 0.5, -0.2, 0.1, 0.8, 0.2, 1.4]),
    "parch_all": (dict(model="parch", dist="t", arch_lags=(1, 2), garch_lags=(1,), ar_lags=(1,),
                       ma_lags=(1,), k_x=3, k_z=1),
                  [0.3, 0.5, -0.2, 0.2, 0.3, 0.3, -1.0, 0.1, 0.05, 0.7, 1.4, 1.0]),
    "egarch11": (dict(model="egarch", arch_lags=(1,), garch_lags=(1,), k_x=3),
                 [0.3, 0.5, -0.2, -0.1, 0.2, 0.9, 0.05]),
    "egarch_all": (dict(model="egarch", dist="t", arch_lags=(1, 2), garch_lags=(1, 2),
                        ar_lags=(1,), ma_lags=(2,), k_x=3, k_z=1),
                   [0.3, 0.5, -0.2, 0.2, 0.3, 0.3, 0.05, -0.1, 0.05, 0.2, 0.1, 0.5, 0.3, 1.0]),
    "garch_m_variance": (dict(model="garch", arch_lags=(1,), garch_lags=(1,), archm="variance",
                              k_x=3), [0.3, 0.5, -0.2, 0.2, 0.1, 0.8, 0.2]),
    "garch_m_sd_arma": (dict(model="garch", arch_lags=(1, 2), garch_lags=(1,), archm="sd",
                             ar_lags=(1, 2), ma_lags=(1,), k_x=3),
                        [0.3, 0.5, -0.2, 0.2, 0.3, -0.2, 0.3, 0.1, 0.05, 0.7, 0.2]),
    "egarch_m_log_ged": (dict(model="egarch", dist="ged", arch_lags=(1,), garch_lags=(1,),
                              archm="log", ar_lags=(1, 2), ma_lags=(1,), k_x=3),
                         [0.3, 0.5, -0.2, 0.2, 0.3, -0.2, 0.3, -0.1, 0.2, 0.9, 0.05, 0.6]),
    "parch_m_sd": (dict(model="parch", arch_lags=(1,), garch_lags=(1,), archm="sd", ar_lags=(2,),
                        k_x=3), [0.3, 0.5, -0.2, 0.2, 0.3, 0.1, 0.8, 0.2, 1.4]),
    "gjr_m_log_het": (dict(model="gjr", dist="t", arch_lags=(1, 2), garch_lags=(1,), archm="log",
                           ma_lags=(1,), k_x=3, k_z=1),
                      [0.3, 0.5, -0.2, 0.2, 0.3, 0.3, -1.0, 0.1, 0.05, 0.05, 0.1, 0.6, 1.0]),
    "no_regressors": (dict(model="garch", arch_lags=(1,), garch_lags=(1,), k_x=0),
                      [0.1, 0.8, 0.2]),
    # First-order models without ARMA terms take the scalar-state loop of run_path.
    "egarch_m_sd_het": (dict(model="egarch", dist="t", arch_lags=(1,), garch_lags=(1,),
                             archm="sd", k_x=3, k_z=1),
                        [0.3, 0.5, -0.2, 0.2, 0.3, 0.05, -0.1, 0.2, 0.9, 1.0]),
    "earch_only": (dict(model="egarch", arch_lags=(1,), garch_lags=(), k_x=3),
                   [0.3, 0.5, -0.2, -0.1, 0.3, 0.1]),
    "gjr_m_sd": (dict(model="gjr", arch_lags=(1,), garch_lags=(1,), archm="sd", k_x=3),
                 [0.3, 0.5, -0.2, 0.2, 0.05, 0.1, 0.8, 0.2]),
    "parch_m_log_ged": (dict(model="parch", dist="ged", arch_lags=(1,), garch_lags=(1,),
                             archm="log", k_x=3), [0.3, 0.5, -0.2, 0.2, 0.1, 0.8, 0.2, 1.4, 0.3]),
    "arch_m_variance": (dict(model="arch", arch_lags=(1,), garch_lags=(), archm="variance",
                             k_x=3), [0.3, 0.5, -0.2, 0.1, 0.2, 0.9]),
}


def build(name, n=240, seed=0):
    keywords, theta = CASES[name]
    keywords = {"dist": "normal", **keywords}
    layout = Layout(**keywords)
    y, x, z = series(n, seed)
    x, z = x[:, :layout.k_x], z[:, :layout.k_z]
    like = ArchLikelihood(torch.tensor(y), torch.tensor(x), torch.tensor(z), layout)
    assert len(theta) == layout.k
    return layout, like, torch.tensor(theta, dtype=torch.float64), (y, x, z)


def oracle_at(layout, theta, data):
    y, x, z = data
    terms = terms_of(layout, layout.k_x, layout.k_z)
    return oracle_loglike(unpack(terms, theta, layout.model), y, x, z, model=layout.model,
                          dist=layout.dist, archm=layout.archm)


# ---- kernel against the oracle ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_log_likelihood_matches_the_explicit_recursion(name):
    layout, like, theta, data = build(name)
    out = like.evaluate(theta, scores=True)
    reference = oracle_at(layout, theta.tolist(), data)
    assert out is not None and reference is not None
    assert_allclose(out.loglik.numpy(), reference, rtol=1e-10, atol=1e-10)
    assert out.value == pytest.approx(reference.sum(), rel=1e-12)
    assert_allclose(out.scores.sum(dim=0).numpy(), out.gradient.numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("name", sorted(CASES))
def test_analytic_gradient_matches_numerical_derivatives(name):
    _, like, theta, _ = build(name)

    def objective(point):
        out = like.evaluate(point)
        if out is None:
            return torch.tensor(math.nan), torch.full_like(point, math.nan)
        return out.value, out.gradient

    report = check_derivatives(objective, theta)
    assert report["gradient_max_rel_error"] < 1e-6


@pytest.mark.parametrize("name", ["garch11", "gjr_ged", "garch_arma", "egarch_all",
                                  "garch_m_sd_arma", "parch_all", "gjr_m_log_het"])
def test_per_observation_scores_match_numerical_scores_of_the_oracle(name):
    layout, like, theta, data = build(name, n=120)
    out = like.evaluate(theta, scores=True)
    numerical = approx_fprime(np.asarray(theta.tolist()),
                              lambda point: oracle_at(layout, point, data), centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("name", [name for name, (keywords, _) in CASES.items()
                                  if keywords["model"] != "egarch" and "archm" not in keywords])
def test_filter_and_scan_engines_agree(name):
    _, like, theta, _ = build(name)
    assert like.linear
    fast = like.evaluate(theta, scores=True, engine="filter")
    general = like.evaluate(theta, scores=True, engine="scan")
    assert fast.value == pytest.approx(general.value, rel=1e-12)
    assert_allclose(fast.variance.numpy(), general.variance.numpy(), rtol=1e-11)
    assert_allclose(fast.scores.numpy(), general.scores.numpy(), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("name", [name for name, (keywords, _) in CASES.items()
                                  if keywords["model"] != "egarch" and "archm" not in keywords])
def test_reverse_mode_gradient_equals_the_sum_of_the_forward_scores(name):
    # Without scores the filter engine computes the gradient in reverse mode (anti-causal
    # filters of the weights); with scores it carries every derivative path forward.
    _, like, theta, _ = build(name)
    reverse = like.evaluate(theta)
    forward = like.evaluate(theta, scores=True)
    assert reverse.scores is None and reverse.value == forward.value
    assert_allclose(reverse.gradient.numpy(), forward.scores.sum(dim=0).numpy(), rtol=1e-11,
                    atol=1e-11)


def test_nonlinear_models_refuse_the_filter_engine():
    _, like, theta, _ = build("egarch11")
    assert not like.linear
    with pytest.raises(ValueError):
        like.evaluate(theta, engine="filter")


def test_infeasible_points_return_none():
    layout, like, theta, _ = build("garch11")
    bad = theta.clone()
    bad[layout.index.c] = -5.0                      # negative variance
    assert like.evaluate(bad) is None
    layout, like, theta, _ = build("parch")
    bad = theta.clone()
    bad[layout.index.p[0]] = -1.0                   # power outside its domain
    assert like.evaluate(bad) is None
    layout, like, theta, _ = build("garch_lags_t")
    bad = theta.clone()
    bad[layout.index.d[0]] = 50.0                   # degrees of freedom beyond the bound
    assert like.evaluate(bad) is None
    assert like.evaluate(torch.full_like(theta, math.nan)) is None


def test_arma_residuals_match_scipy_lfilter():
    # With constant variance the innovations are the conditional (zero presample) ARMA
    # residuals: e = rho(L) (y - x b) / theta(L).
    layout, like, theta, (y, x, _) = build("garch_arma")
    ix = layout.index
    out = like.evaluate(theta)
    beta = np.asarray(theta[ix.x].tolist())
    rho, tma = theta[ix.ar].tolist(), theta[ix.ma].tolist()
    ar_polynomial = np.zeros(max(layout.ar_lags) + 1)
    ma_polynomial = np.zeros(max(layout.ma_lags) + 1)
    ar_polynomial[0] = ma_polynomial[0] = 1.0
    for value, lag in zip(rho, layout.ar_lags):
        ar_polynomial[lag] = -value
    for value, lag in zip(tma, layout.ma_lags):
        ma_polynomial[lag] = value
    reference = lfilter(ar_polynomial, ma_polynomial, y - x @ beta)
    assert_allclose(out.residual.numpy(), reference, rtol=1e-11, atol=1e-12)
    assert out.presample_variance == pytest.approx(np.mean(reference ** 2), rel=1e-12)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")        # raised inside statsmodels
@pytest.mark.parametrize("ar_lags,ma_lags", [((1,), (1,)), ((1, 2), ()), ((1, 3), (2,)),
                                             ((), (1, 2))])
def test_conditional_mean_likelihood_matches_statsmodels_at_a_fixed_variance(ar_lags, ma_lags):
    # With the ARCH coefficient at zero the variance is the constant omega and the model
    # is a regression with ARMA errors, conditional on zero presample disturbances. A
    # state-space ARMA model whose state before the sample is KNOWN to be zero (so that
    # the first predicted state is R e_1, with covariance R Q R') has that likelihood.
    layout = Layout(model="arch", dist="normal", arch_lags=(1,), garch_lags=(), ar_lags=ar_lags,
                    ma_lags=ma_lags, k_x=3)
    y, x, _ = series(300, 7)
    like = ArchLikelihood(torch.tensor(y), torch.tensor(x),
                          torch.empty((300, 0), dtype=torch.float64), layout)
    beta = [0.3, 0.5, -0.2]
    rho = [0.4, -0.25][:len(ar_lags)]
    tma = [0.35, 0.2][:len(ma_lags)]
    omega = 1.7
    theta = torch.tensor([*beta, *rho, *tma, 0.0, omega], dtype=torch.float64)
    out = like.evaluate(theta)

    def flags(lags):
        return [int(lag in lags) for lag in range(1, max(lags) + 1)] if lags else 0

    model = SARIMAX(y, exog=x, order=(flags(ar_lags), 0, flags(ma_lags)))

    def conditional(params, per_observation=False):
        model.update(params)
        selection = model.ssm["selection"]
        model.initialize_known(np.zeros(model.k_states),
                               selection @ model.ssm["state_cov"] @ selection.T)
        return model.loglikeobs(params) if per_observation else model.loglike(params)

    params = np.array([*beta, *rho, *tma, omega])
    assert_allclose(out.loglik.numpy(), conditional(params, True), rtol=1e-9, atol=1e-9)
    assert out.value == pytest.approx(conditional(params), rel=1e-11)
    assert_allclose(out.variance.numpy(), omega, rtol=1e-14)
    # The scores of the mean and ARMA parameters are those of the state-space model.
    numerical = approx_fprime(params, conditional, centered=True)
    keep = [*layout.index.x, *layout.index.ar, *layout.index.ma]
    assert_allclose(out.gradient[keep].numpy(), numerical[:len(keep)], rtol=1e-5, atol=1e-6)


# ---- building blocks ----------------------------------------------------------------------------


def test_recurrence_scan_matches_a_sequential_loop():
    rng = np.random.default_rng(3)
    n, m, k = 37, 3, 4
    maps = rng.normal(size=(n, m, m)) * 0.6
    forcing = rng.normal(size=(n, m, k))
    expected = np.zeros((n, m, k))
    state = np.zeros((m, k))
    for t in range(n):
        state = maps[t] @ state + forcing[t]
        expected[t] = state
    solved = solve_recurrence(torch.tensor(maps), torch.tensor(forcing))
    assert_allclose(solved.numpy(), expected, rtol=1e-11, atol=1e-12)
    one = solve_recurrence(torch.tensor(maps[:1]), torch.tensor(forcing[:1]))
    assert_allclose(one.numpy(), forcing[:1])


def test_lead_and_adjoint_filter_are_the_transposes_of_lag_and_filter():
    rng = np.random.default_rng(4)
    x, f = torch.tensor(rng.normal(size=50)), torch.tensor(rng.normal(size=50))
    assert lead(torch.arange(1.0, 6.0, dtype=torch.float64), 2).tolist() == [3.0, 4.0, 5.0, 0, 0]
    assert lead(x, 80).abs().max() == 0.0
    assert float(torch.dot(x, lagged(f, 3))) == pytest.approx(float(torch.dot(lead(x, 3), f)),
                                                              rel=1e-13)
    coefficients = [-0.5, 0.0, 0.2]
    assert float(torch.dot(x, inverse_filter(f, coefficients))) == pytest.approx(
        float(torch.dot(adjoint_filter(x, coefficients), f)), rel=1e-12)
    assert adjoint_filter(x, []) is x


def test_lagged_fills_the_presample():
    x = torch.arange(1.0, 6.0, dtype=torch.float64)
    assert lagged(x, 2, 9.0).tolist() == [9.0, 9.0, 1.0, 2.0, 3.0]
    assert lagged(x, 7, -1.0).tolist() == [-1.0] * 5
    block = torch.stack([x, 10 * x])
    fill = torch.tensor([[7.0], [8.0]], dtype=torch.float64)
    assert lagged(block, 1, fill).tolist() == [[7.0, 1.0, 2.0, 3.0, 4.0],
                                               [8.0, 10.0, 20.0, 30.0, 40.0]]


@pytest.mark.parametrize("dist,tau", [("normal", 0.0), ("t", 1.1), ("t", 3.0), ("ged", 0.2),
                                      ("ged", 0.9), ("ged", -0.3)])
def test_densities_are_standardized_and_match_scipy(dist, tau):
    if dist == "normal":
        frozen = stats.norm()
    elif dist == "t":
        nu = 2 + math.exp(tau)
        frozen = stats.t(nu, scale=math.sqrt((nu - 2) / nu))
    else:
        shape = math.exp(tau)
        frozen = stats.gennorm(shape, scale=math.sqrt(gamma_function(1 / shape)
                                                      / gamma_function(3 / shape)))
    assert frozen.var() == pytest.approx(1.0, rel=1e-10)
    e = torch.tensor([-2.5, -0.4, 0.0, 0.3, 1.7], dtype=torch.float64)
    h = torch.tensor([0.5, 1.0, 2.0, 0.7, 1.3], dtype=torch.float64)
    value, d_e, d_log_h, d_tau = densities.log_density(dist, e, h, tau)
    sd = np.sqrt(h.numpy())
    assert_allclose(value.numpy(), frozen.logpdf(e.numpy() / sd) - np.log(sd), rtol=1e-12)
    # Derivatives against central differences of the density itself.
    step = 1e-6
    up = densities.log_density(dist, e + step, h, tau, derivatives=False)[0]
    down = densities.log_density(dist, e - step, h, tau, derivatives=False)[0]
    keep = e != 0.0 if dist == "ged" else torch.ones_like(e, dtype=torch.bool)
    assert_allclose(d_e[keep].numpy(), ((up - down) / (2 * step))[keep].numpy(), rtol=1e-6,
                    atol=1e-8)
    up = densities.log_density(dist, e, h * math.exp(step), tau, derivatives=False)[0]
    down = densities.log_density(dist, e, h * math.exp(-step), tau, derivatives=False)[0]
    assert_allclose(d_log_h.numpy(), ((up - down) / (2 * step)).numpy(), rtol=1e-6, atol=1e-8)
    if dist != "normal":
        up = densities.log_density(dist, e, h, tau + step, derivatives=False)[0]
        down = densities.log_density(dist, e, h, tau - step, derivatives=False)[0]
        assert_allclose(d_tau.numpy(), ((up - down) / (2 * step)).numpy(), rtol=1e-6, atol=1e-8)
    # Absolute moments and two-sided critical values.
    for power in (1.0, 1.4, 2.0):
        expected = integrate.quad(lambda v: abs(v) ** power * frozen.pdf(v), -np.inf, np.inf)[0]
        assert densities.abs_moment(dist, power, tau) == pytest.approx(expected, rel=1e-7)
    assert densities.abs_moment(dist, 2.0, tau) == pytest.approx(1.0, rel=1e-10)
    assert densities.critical_value(dist, 0.05, tau) == pytest.approx(frozen.isf(0.025), rel=1e-8)


def test_t_absolute_moment_is_infinite_at_or_above_the_degrees_of_freedom():
    tau = math.log(1.0)                              # 3 degrees of freedom
    assert math.isinf(densities.abs_moment("t", 3.0, tau))
    assert math.isfinite(densities.abs_moment("t", 2.9, tau))


# ---- helpers shared with the other arch test files ------------------------------------------


def simulate(n=700, seed=0, *, model="garch", dist="normal", ar=0.0, ma=0.0, archm=None,
             psi=0.0, het=0.0, burn=200):
    """A regression with ARCH-family errors as a DataFrame (columns y, x, z, t)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    total = n + burn
    x, z = rng.normal(size=total), rng.normal(size=total)
    if dist == "t":
        shocks = rng.standard_t(7, size=total) * math.sqrt(5 / 7)
    elif dist == "ged":
        shocks = stats.gennorm.rvs(1.4, scale=math.sqrt(gamma_function(1 / 1.4)
                                                        / gamma_function(3 / 1.4)),
                                   size=total, random_state=rng)
    else:
        shocks = rng.normal(size=total)
    e, h, u = np.zeros(total), np.ones(total), np.zeros(total)
    for t in range(1, total):
        scale = math.exp(het * z[t])
        if model == "egarch":
            lagged_z = e[t - 1] / math.sqrt(h[t - 1])
            h[t] = math.exp(het * z[t] - 0.12 * lagged_z + 0.25 * (abs(lagged_z) - ABS_NORMAL)
                            + 0.85 * math.log(h[t - 1]))
        elif model == "gjr":
            h[t] = 0.1 * scale + (0.05 + 0.15 * (e[t - 1] < 0)) * e[t - 1] ** 2 + 0.75 * h[t - 1]
        elif model == "parch":
            h[t] = (0.1 * scale + 0.12 * abs(e[t - 1]) ** 1.3
                    + 0.8 * h[t - 1] ** 0.65) ** (2 / 1.3)
        elif model == "arch":
            h[t] = 0.5 * scale + 0.4 * e[t - 1] ** 2
        else:
            h[t] = 0.1 * scale + 0.12 * e[t - 1] ** 2 + 0.78 * h[t - 1]
        e[t] = math.sqrt(h[t]) * shocks[t]
        u[t] = ar * u[t - 1] + e[t] + ma * e[t - 1]
    in_mean = {None: 0.0, "variance": h, "sd": np.sqrt(h), "log": np.log(h)}[archm]
    y = 0.3 + 0.5 * x + psi * in_mean + u
    frame = pd.DataFrame({"y": y, "x": x, "z": z})[burn:].reset_index(drop=True)
    frame["t"] = np.arange(1, n + 1)
    return frame


def oracle_of(result, frame):
    """``(theta_hat, per_observation_loglike(theta))`` of a fitted result on its data.

    The function takes the FULL reported vector (for IGARCH the caller keeps the
    restriction). Regressors are rebuilt from the specification, not from the kernel.
    ``loglike(point, signs=...)`` evaluates a smooth piece of the EGARCH likelihood and
    ``loglike.path(point)`` returns the oracle's ``(e, h, u)``.
    """
    spec = result.spec
    terms = [c.term for c in result.coefficients]
    theta = np.array([c.estimate for c in result.coefficients])
    frame = frame.sort_values(spec.time) if spec.time else frame
    y = frame[spec.outcome].to_numpy(float)
    columns = [np.ones(len(frame))] if spec.intercept else []
    columns += [frame[name].to_numpy(float) for name in spec.predictors]
    x = np.column_stack(columns) if columns else np.empty((len(frame), 0))
    names = spec.columns.get("variance_x") or []
    z = np.column_stack([frame[name].to_numpy(float) for name in names]) if names else None
    options = spec.options

    def loglike(point, signs=None):
        return oracle_loglike(unpack(terms, point, options.get("model", "garch")), y, x, z,
                              model=options.get("model", "garch"),
                              dist=options.get("dist", "normal"), archm=options.get("archm"),
                              signs=signs)

    loglike.path = lambda point: oracle_recursion(
        unpack(terms, point, options.get("model", "garch")), y, x, z,
        model=options.get("model", "garch"), archm=options.get("archm"))
    return theta, loglike
