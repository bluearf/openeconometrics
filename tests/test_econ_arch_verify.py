"""Independent verification of ``oe.arch`` (verify-and-repair pass).

The oracle of this file was written from the model definitions alone and shares nothing
with the family package or with ``test_econ_arch_oracle``:

- GARCH-type models are evaluated with ``scipy.signal.lfilter`` (direct-form IIR
  filters); EGARCH and ARCH-in-mean models with an explicit loop on Python complex
  numbers;
- the densities are written out with ``scipy.special.loggamma``;
- every function accepts COMPLEX parameters, so gradients and per-observation scores are
  complex-step derivatives (exact to rounding, no step-size error) and the Hessian is a
  central difference of those gradients;
- the maximum is found by SciPy's BFGS from the data-generating values and polished by
  Newton steps on the oracle's own derivatives to machine precision.

Results of ``oe.arch`` are then compared with that maximum: estimates (to a small
fraction of a standard error and to 1e-6 relative), log likelihood, the OPG / OIM /
robust covariance matrices, z statistics, p-values, intervals, information criteria,
persistence, the Wald model test and the residual diagnostics.
"""

import cmath
import math
import re

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import loggamma
from test_econ_arch_oracle import simulate as kink_sample

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry

E_ABS_NORMAL = math.sqrt(2.0 / math.pi)
LAG_TERM = re.compile(r"^(ARMA|ARCH):L(\d+)\.(\w+)$")


# ---- the oracle -------------------------------------------------------------------------------


def split(terms, theta):
    """Parameters by reported term name (values may be complex)."""
    p = {"b": [], "psi": 0.0, "ar": [], "ma": [], "het": [], "het0": None, "omega": None,
         "arch": [], "asym": [], "garch": [], "power": 2.0, "tau": None}
    for term, value in zip(terms, theta, strict=True):
        found = LAG_TERM.match(term)
        if found:
            lag, name = int(found.group(2)), found.group(3)
            block = {"ar": "ar", "ma": "ma", "arch": "arch", "earch": "arch", "parch": "arch",
                     "tarch": "asym", "earch_a": "asym", "garch": "garch", "egarch": "garch",
                     "pgarch": "garch"}[name]
            p[block].append((lag, value))
        elif term.startswith("ARCHM:"):
            p["psi"] = value
        elif term == "HET:Intercept":
            p["het0"] = value
        elif term.startswith("HET:"):
            p["het"].append(value)
        elif term == "ARCH:Intercept":
            p["omega"] = value
        elif term == "POWER:power":
            p["power"] = value
        elif term in ("/lndfm2", "/lnshape"):
            p["tau"] = value
        else:
            p["b"].append(value)
    return p


def _poly(pairs, sign):
    """[1, sign*c_1, ..., sign*c_m] for (lag, value) pairs."""
    out = np.zeros(1 + max((lag for lag, _ in pairs), default=0), dtype=complex)
    out[0] = 1.0
    for lag, value in pairs:
        out[lag] = sign * value
    return out


def _shift(series, lag, fill):
    out = np.empty_like(series)
    out[:lag] = fill
    out[lag:] = series[:len(series) - lag]
    return out


def _mean_residual(p, y, x):
    """Residual of the mean equation with its ARMA terms and zero presample values."""
    m = y - x @ np.array(p["b"], dtype=complex) if x.shape[1] else y.astype(complex)
    return m, lfilter(_poly(p["ar"], -1.0), _poly(p["ma"], 1.0), m)


def paths(theta, terms, y, x, z, model, archm=None, signs=None):
    """(e, h) of the model; ``None`` if a variance is not positive."""
    p = split(terms, np.asarray(theta).tolist())          # Python scalars (complex or float)
    n = len(y)
    kind = {"arch": "garch", "igarch": "garch", "tarch": "gjr"}.get(model, model)
    m, e_plain = _mean_residual(p, y, x)
    h0 = (e_plain * e_plain).sum() / n
    if z is not None and z.shape[1]:
        index = p["het0"] + z @ np.array(p["het"], dtype=complex)
        w = index if kind == "egarch" else np.exp(index)
    else:
        w = np.full(n, p["omega"], dtype=complex)
    phi = p["power"]
    if kind != "egarch" and archm is None:
        e = e_plain
        if kind == "parch":
            news = np.exp(phi * np.log(e * np.sign(e.real)))
            start = np.exp(0.5 * phi * np.log(h0))
        else:
            news, start = e * e, h0
        forcing = w.copy()
        for lag, value in p["arch"]:
            forcing += value * _shift(news, lag, start)
        for lag, value in p["asym"]:
            forcing += value * _shift(news * (e.real < 0), lag, 0.5 * h0)
        for lag, value in p["garch"]:
            forcing[:lag] += value * start
        state = lfilter([1.0], _poly(p["garch"], -1.0), forcing)
        if not (state.real > 0).all():
            return None
        h = np.exp((2.0 / phi) * np.log(state)) if kind == "parch" else state
        return e, h
    try:
        return _loop(p, m.tolist(), w.tolist(), complex(h0), kind, archm, signs)
    except (OverflowError, ZeroDivisionError, ValueError):
        return None


def _loop(p, m, w, h0, kind, archm, signs):
    """EGARCH and ARCH-in-mean paths by an explicit loop on Python complex numbers."""
    phi = p["power"]
    e, u, h = [], [], []
    for t in range(len(m)):
        if kind == "egarch":
            v = w[t]
            for lag, value in p["arch"]:
                if t >= lag:
                    v += value * e[t - lag] / cmath.sqrt(h[t - lag])
            for lag, value in p["asym"]:
                if t >= lag:
                    zed = e[t - lag] / cmath.sqrt(h[t - lag])
                    side = (1.0 if zed.real >= 0 else -1.0) if signs is None else signs[t - lag]
                    v += value * (side * zed - E_ABS_NORMAL)
            for lag, value in p["garch"]:
                v += value * cmath.log(h[t - lag] if t >= lag else h0)
            if abs(v.real) > 600:
                return None
            ht = cmath.exp(v)
        else:
            power = 0.5 * phi if kind == "parch" else 1.0
            v = w[t]
            for lag, value in p["arch"]:
                if t < lag:
                    v += value * (cmath.exp(power * cmath.log(h0)) if kind == "parch" else h0)
                elif kind == "parch":
                    past = e[t - lag]
                    v += value * cmath.exp(phi * cmath.log(past if past.real >= 0 else -past))
                else:
                    v += value * e[t - lag] ** 2
            for lag, value in p["asym"]:
                if t < lag:
                    v += value * 0.5 * h0
                elif e[t - lag].real < 0:
                    v += value * e[t - lag] ** 2
            for lag, value in p["garch"]:
                past = h[t - lag] if t >= lag else h0
                v += value * (cmath.exp(power * cmath.log(past)) if kind == "parch" else past)
            if not v.real > 0:
                return None
            ht = cmath.exp(cmath.log(v) / power) if kind == "parch" else v
        in_mean = {None: 0.0, "variance": ht, "sd": cmath.sqrt(ht), "log": cmath.log(ht)}[archm]
        ut = m[t] - p["psi"] * in_mean
        et = ut
        for lag, value in p["ar"]:
            if t >= lag:
                et -= value * u[t - lag]
        for lag, value in p["ma"]:
            if t >= lag:
                et -= value * e[t - lag]
        e.append(et)
        u.append(ut)
        h.append(ht)
    return np.array(e), np.array(h)


def loglik(theta, terms, y, x, z, model, dist="normal", archm=None, signs=None, flat=None):
    """Per-observation log likelihood (complex for a complex ``theta``); ``None`` if undefined.

    ``signs`` (EGARCH) replaces |z_t| by ``signs[t] * z_t`` in the news; ``flat`` lists
    observations whose GED kernel is left out (the kinks of an EGARCH maximum).
    """
    theta = np.asarray(theta)
    out = paths(theta, terms, y, x, z, model, archm, signs)
    if out is None:
        return None
    e, h = out
    tau = split(terms, theta.tolist())["tau"]
    q = e * e / h
    if dist == "normal":
        return -0.5 * math.log(2.0 * math.pi) - 0.5 * np.log(h) - 0.5 * q
    if dist == "t":
        nu = 2.0 + np.exp(tau)
        return (loggamma(0.5 * (nu + 1.0)) - loggamma(0.5 * nu) - 0.5 * np.log(math.pi * (nu - 2.0))
                - 0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log(1.0 + q / (nu - 2.0)))
    s = np.exp(tau)
    log_lam = 0.5 * (-(2.0 / s) * math.log(2.0) + loggamma(1.0 / s) - loggamma(3.0 / s))
    size = e * np.sign(e.real) / np.sqrt(h)                     # |e| / sqrt(h)
    zero = size == 0                                            # the kernel vanishes at e = 0
    if flat:
        zero[flat] = True
    kernel = np.where(zero, 0.0, np.exp(s * (np.log(np.where(zero, 1.0, size)) - log_lam)))
    return (np.log(s) - log_lam - (1.0 + 1.0 / s) * math.log(2.0) - loggamma(1.0 / s)
            - 0.5 * np.log(h) - 0.5 * kernel)


def scores(function, theta):
    """Per-observation scores by complex steps: [n, k]."""
    theta = np.asarray(theta, dtype=float)
    columns = []
    for j in range(len(theta)):
        point = theta.astype(complex)
        point[j] += 1e-30j
        columns.append(function(point).imag / 1e-30)
    return np.column_stack(columns)


def hessian(function, theta):
    """Central differences of the complex-step gradient, symmetrized.

    The step is small (1e-7): the GED and power-ARCH likelihoods contain |e|^s with
    s < 2, whose third derivative is large at small residuals, and the complex-step
    gradient is exact to rounding, so a small step costs no accuracy. (A first version
    of this oracle used 1e-5 and was off by 0.3% for GED; the package was right.)
    """
    theta = np.asarray(theta, dtype=float)
    k = len(theta)
    out = np.empty((k, k))
    for j in range(k):
        step = 1e-7 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        out[:, j] = (scores(function, up).sum(axis=0) - scores(function, down).sum(axis=0)) \
            / (2.0 * step)
    return 0.5 * (out + out.T)


def maximize(function, start, newton=8):
    """BFGS on the complex-step gradient, then Newton steps: (theta, max |gradient|)."""
    start = np.asarray(start, dtype=float)
    scale = np.maximum(np.abs(start), 1e-3)

    def objective(t):
        value = function(t * scale)
        if value is None or not np.isfinite(value).all():
            return 1e12, np.zeros(len(t))
        return -value.real.sum(), -scores(function, t * scale).sum(axis=0) * scale

    found = minimize(objective, start / scale, jac=True, method="BFGS",
                     options={"gtol": 1e-7, "maxiter": 500})
    theta = found.x * scale
    for _ in range(newton):
        gradient = scores(function, theta).sum(axis=0)
        step = np.linalg.solve(hessian(function, theta), gradient)
        candidate = theta - step
        value = function(candidate)
        if value is None or not np.isfinite(value).all():
            break
        theta = candidate
        if np.abs(step).max() < 1e-13 * max(1.0, np.abs(theta).max()):
            break
    return theta, np.abs(scores(function, theta).sum(axis=0)).max()


# ---- data --------------------------------------------------------------------------------------


def draw(n, seed, *, model="garch", dist="normal", ar=0.0, ma=0.0, archm=None, psi=0.0,
         het=0.0, burn=300):
    """A simulated regression with ARCH-family errors: columns y, x1, x2, zv, t."""
    rng = np.random.default_rng(seed)
    total = n + burn
    x1, x2, zv = rng.normal(size=total), rng.uniform(-1, 1, size=total), rng.normal(size=total)
    if dist == "t":
        shock = rng.standard_t(8, size=total) / math.sqrt(8 / 6)
    elif dist == "ged":
        # GED(1.5) by its gamma representation: |z / lam|^s / 2 ~ Gamma(1/s).
        s = 1.5
        lam = math.sqrt(2 ** (-2 / s) * math.gamma(1 / s) / math.gamma(3 / s))
        shock = lam * (2 * rng.gamma(1 / s, size=total)) ** (1 / s) * rng.choice([-1, 1], total)
    else:
        shock = rng.normal(size=total)
    e, h, u = np.zeros(total), np.ones(total), np.zeros(total)
    for t in range(1, total):
        level = math.exp(het * zv[t])
        if model == "egarch":
            zl = e[t - 1] / math.sqrt(h[t - 1])
            h[t] = math.exp(-0.05 + het * zv[t] - 0.1 * zl + 0.3 * (abs(zl) - E_ABS_NORMAL)
                            + 0.8 * math.log(h[t - 1]))
        elif model == "gjr":
            h[t] = 0.15 * level + (0.06 + 0.16 * (e[t - 1] < 0)) * e[t - 1] ** 2 + 0.7 * h[t - 1]
        elif model == "parch":
            h[t] = (0.15 * level + 0.15 * abs(e[t - 1]) ** 1.5
                    + 0.7 * h[t - 1] ** 0.75) ** (2 / 1.5)
        elif model == "arch":
            h[t] = 0.6 * level + 0.35 * e[t - 1] ** 2
        else:
            h[t] = 0.15 * level + 0.15 * e[t - 1] ** 2 + 0.7 * h[t - 1]
        e[t] = math.sqrt(h[t]) * shock[t]
        u[t] = ar * u[t - 1] + e[t] + ma * e[t - 1]
    extra = {None: 0.0, "variance": h, "sd": np.sqrt(h), "log": np.log(h)}[archm]
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + psi * extra + u
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "zv": zv}).iloc[burn:].reset_index(drop=True)
    frame["t"] = np.arange(1, n + 1)
    return frame


def oracle_for(result, frame):
    """``f(theta) -> per-observation log likelihood`` of a result's model on its data."""
    spec = result.spec
    terms = [c.term for c in result.coefficients]
    y = frame[spec.outcome].to_numpy(float)
    columns = ([np.ones(len(frame))] if spec.intercept else []) \
        + [frame[name].to_numpy(float) for name in spec.predictors]
    x = np.column_stack(columns) if columns else np.empty((len(frame), 0))
    names = spec.columns.get("variance_x") or []
    z = np.column_stack([frame[name].to_numpy(float) for name in names]) if names else None
    options = spec.options

    def function(theta, signs=None, flat=None):
        return loglik(theta, terms, y, x, z, options.get("model", "garch"),
                      options.get("dist", "normal"), options.get("archm"), signs, flat)

    function.path = lambda theta, signs=None: paths(
        theta, terms, y, x, z, options.get("model", "garch"), options.get("archm"), signs)
    return function


def reported(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def truth(terms, model, generator):
    """The data-generating parameter values in the order of the reported terms."""
    kind = {"tarch": "gjr", "igarch": "garch"}.get(model, model)
    constant = {"egarch": -0.05, "arch": 0.6}.get(kind, 0.15)
    table = {
        "Intercept": 1.0, "x1": 0.6, "x2": -0.4,
        "ARMA:L1.ar": generator.get("ar", 0.0), "ARMA:L1.ma": generator.get("ma", 0.0),
        "HET:zv": generator.get("het", 0.0),
        "HET:Intercept": constant if kind == "egarch" else math.log(constant),
        "ARCH:Intercept": constant,
        "ARCH:L1.arch": {"gjr": 0.06, "arch": 0.35}.get(kind, 0.15), "ARCH:L2.arch": 0.02,
        "ARCH:L1.tarch": 0.16, "ARCH:L1.garch": 0.7, "ARCH:L2.garch": 0.05,
        "ARCH:L1.earch": -0.1, "ARCH:L1.earch_a": 0.3, "ARCH:L1.egarch": 0.8,
        "ARCH:L1.parch": 0.15, "ARCH:L1.pgarch": 0.7, "POWER:power": 1.5,
        "/lndfm2": math.log(6.0), "/lnshape": math.log(1.5),
    }
    return np.array([generator.get("psi", 0.0) if term.startswith("ARCHM:") else table[term]
                     for term in terms])


def scaled_error(actual, expected):
    """Largest covariance error in units of the expected standard errors."""
    sd = np.sqrt(np.diag(expected))
    return np.abs((np.asarray(actual) - expected) / np.outer(sd, sd)).max()


# ---- estimates and covariances against the independently maximized oracle -------------------

# name -> (oe.arch keywords, generator keywords, observations, seed)
MODELS = {
    "garch": (dict(), dict(), 700, 11),
    "garch_t": (dict(dist="t"), dict(dist="t"), 700, 12),
    "garch_ged": (dict(dist="ged"), dict(dist="ged"), 700, 13),
    "arch1": (dict(model="arch"), dict(model="arch"), 700, 14),
    "garch21": (dict(arch=2, garch=1), dict(), 700, 15),
    "gjr": (dict(model="gjr"), dict(model="gjr"), 700, 16),
    "gjr_t": (dict(model="tarch", dist="t"), dict(model="gjr", dist="t"), 700, 17),
    "parch": (dict(model="parch"), dict(model="parch"), 900, 18),
    "arma": (dict(ar=1, ma=1), dict(ar=0.5, ma=0.3), 700, 19),
    "het": (dict(variance_x=["zv"]), dict(het=0.4), 700, 20),
    "egarch": (dict(model="egarch"), dict(model="egarch"), 350, 21),
    "egarch_t_het": (dict(model="egarch", dist="t", variance_x=["zv"]),
                     dict(model="egarch", dist="t", het=0.3), 350, 22),
    "archm_variance": (dict(archm="variance"), dict(archm="variance", psi=0.3), 350, 23),
    "archm_sd_ma": (dict(archm="sd", ma=1), dict(archm="sd", psi=0.3, ma=0.3), 350, 24),
    "archm_log_gjr": (dict(model="gjr", archm="log"), dict(model="gjr", archm="log", psi=0.3),
                      350, 25),
}


@pytest.fixture(scope="module")
def solved():
    """Per model: the three fits, the data and the oracle's own maximum (computed once)."""
    cache = {}

    def get(name):
        if name not in cache:
            keywords, generator, n, seed = MODELS[name]
            frame = draw(n, seed, **generator)
            fits = {kind: oe.arch(data=frame, y="y", x=["x1", "x2"], time="t", covariance=kind,
                                  **keywords) for kind in ("opg", "nonrobust", "robust")}
            function = oracle_for(fits["opg"], frame)
            terms = [c.term for c in fits["opg"].coefficients]
            best, gradient = maximize(function, truth(terms, keywords.get("model", "garch"),
                                                      generator))
            cache[name] = (fits, frame, function, best, gradient)
        return cache[name]

    return get


@pytest.mark.parametrize("name", sorted(MODELS))
def test_estimates_equal_the_independent_maximum(name, solved):
    fits, frame, function, best, gradient = solved(name)
    result = fits["opg"]
    assert result.provenance["optimizer"].get("kink_observations") is None
    assert gradient < 1e-8                                     # the oracle is at its maximum
    theta, se = reported(result), errors(result)
    value = function(best).real.sum()
    assert result.metrics["log_likelihood"] == pytest.approx(function(theta).real.sum(),
                                                             rel=1e-11)
    assert result.metrics["log_likelihood"] == pytest.approx(value, abs=1e-8)
    assert np.abs((theta - best) / se).max() < 1e-4
    assert_allclose(theta, best, rtol=1e-6, atol=1e-7)
    for other in ("nonrobust", "robust"):
        assert_allclose(reported(fits[other]), theta, rtol=1e-12, atol=1e-14)
    k, n = len(theta), len(frame)
    assert result.nobs == n and result.dropped_rows == 0
    assert result.metrics["aic"] == pytest.approx(-2 * value + 2 * k, rel=1e-10)
    assert result.metrics["bic"] == pytest.approx(-2 * value + k * math.log(n), rel=1e-10)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_covariances_equal_those_of_the_oracle(name, solved):
    fits, frame, function, best, _ = solved(name)
    n = len(frame)
    second = hessian(function, best)
    assert np.linalg.eigvalsh(second).max() < 0
    gradients = scores(function, best)
    bread = np.linalg.inv(-second)
    outer = gradients.T @ gradients
    expected = {"opg": np.linalg.inv(outer), "nonrobust": bread,
                "robust": n / (n - 1) * bread @ outer @ bread}
    for kind, result in fits.items():
        covariance = np.array(result.covariance_matrix)
        assert scaled_error(covariance, expected[kind]) < 2e-5, kind
        assert_allclose(errors(result), np.sqrt(np.diag(expected[kind])), rtol=2e-5)
        assert result.spec.covariance == kind == result.inference["covariance"]
    assert fits["robust"].inference["small_sample_correction"] == pytest.approx(n / (n - 1))
    assert "N/(N-1)" in fits["robust"].inference["correction"]
    assert "observed information" in fits["nonrobust"].inference["correction"]
    assert "outer product" in fits["opg"].inference["correction"]


@pytest.mark.parametrize("name", ["garch", "gjr_t", "arma", "archm_sd_ma", "egarch_t_het"])
def test_inference_is_normal_theory(name, solved):
    fits, frame, _, _, _ = solved(name)
    for kind, result in fits.items():
        theta, se = reported(result), errors(result)
        z = np.array([c.statistic for c in result.coefficients])
        assert_allclose(z, theta / se, rtol=1e-12)
        assert_allclose([c.p_value for c in result.coefficients], 2 * stats.norm.sf(np.abs(z)),
                        rtol=1e-9, atol=1e-300)
        critical = stats.norm.ppf(0.975)
        assert_allclose([c.ci_low for c in result.coefficients], theta - critical * se,
                        rtol=1e-9, atol=1e-12)
        assert_allclose([c.ci_high for c in result.coefficients], theta + critical * se,
                        rtol=1e-9, atol=1e-12)
        record = result.inference
        assert record["use_t"] is False and record["distribution"] == "normal"
        assert record["df_inference"] is None and record["n_parameters"] == len(theta)
        # Wald chi2 of the mean equation: slopes, ARCH-in-mean and ARMA terms.
        terms = [c.term for c in result.coefficients]
        tested = [i for i, (term, c) in enumerate(zip(terms, result.coefficients))
                  if (c.equation == "y" and term != "Intercept") or c.equation in ("ARCHM", "ARMA")]
        block = np.array(result.covariance_matrix)[np.ix_(tested, tested)]
        wald = theta[tested] @ np.linalg.solve(block, theta[tested])
        test = result.tests["model"]
        assert test["statistic"] == pytest.approx(wald, rel=1e-8)
        assert test["df"] == len(tested) and test["distribution"] == "chi2"
        assert test["p_value"] == pytest.approx(stats.chi2.sf(wald, len(tested)), rel=1e-7,
                                                abs=1e-300)


def test_igarch_against_the_restricted_oracle():
    # The oracle is maximized over the FREE parameters; the reported covariance is the
    # delta-method image R V R' of the free one (singular by construction).
    frame = draw(800, 31)
    fits = {kind: oe.arch(data=frame, y="y", x=["x1", "x2"], model="igarch", covariance=kind)
            for kind in ("opg", "nonrobust", "robust")}
    result = fits["opg"]
    terms = [c.term for c in result.coefficients]
    assert terms == ["Intercept", "x1", "x2", "ARCH:L1.arch", "ARCH:L1.garch", "ARCH:Intercept"]
    function = oracle_for(result, frame)

    def restricted(free):
        free = np.asarray(free)
        return function(np.array([free[0], free[1], free[2], free[3], 1.0 - free[3], free[4]]))

    best, gradient = maximize(restricted, [1.0, 0.6, -0.4, 0.15, 0.1])
    assert gradient < 1e-8
    theta = reported(result)
    assert theta[3] + theta[4] == pytest.approx(1.0, abs=1e-13)
    assert_allclose(theta[[0, 1, 2, 3, 5]], best, rtol=1e-6, atol=1e-7)
    value = restricted(best).real.sum()
    assert result.metrics["log_likelihood"] == pytest.approx(value, abs=1e-8)
    assert result.metrics["aic"] == pytest.approx(-2 * value + 2 * 5, rel=1e-10)   # 5 free
    assert result.metrics["bic"] == pytest.approx(-2 * value + 5 * math.log(800), rel=1e-10)
    assert result.metrics["persistence"] == 1.0
    jacobian = np.zeros((6, 5))
    jacobian[[0, 1, 2, 3, 5], [0, 1, 2, 3, 4]] = 1.0
    jacobian[4, 3] = -1.0
    n = len(frame)
    gradients = scores(restricted, best)
    bread = np.linalg.inv(-hessian(restricted, best))
    outer = gradients.T @ gradients
    expected = {"opg": np.linalg.inv(outer), "nonrobust": bread,
                "robust": n / (n - 1) * bread @ outer @ bread}
    for kind, fit in fits.items():
        covariance = np.array(fit.covariance_matrix)
        assert scaled_error(covariance, jacobian @ expected[kind] @ jacobian.T) < 2e-5
        assert covariance[3, 4] == pytest.approx(-covariance[3, 3], rel=1e-10)


# ---- regression tests of the verify pass ----------------------------------------------------


def test_default_covariance_follows_stata():
    # Stata's arch prints "OPG std. err." unless vce(oim) or vce(robust) is requested.
    info = registry.get("arch")
    assert info.default_covariance == "opg" and set(info.covariances) == {"opg", "nonrobust",
                                                                         "robust"}
    result = oe.arch(data=draw(300, 2), y="y")
    assert result.spec.covariance == "opg"
    assert "Stata's default vce(opg)" in result.inference["correction"]


@pytest.mark.parametrize("name,seed,n", [("garch_ged", 113, 700), ("parch", 18, 900),
                                         ("gjr_t", 17, 700)])
def test_observed_information_is_accurate_when_a_residual_is_almost_zero(name, seed, n):
    # The gradients of the GED, power-ARCH and GJR likelihoods have kinks at zero
    # residuals. With the optimizer's default first difference step the numerical Hessian
    # of these samples was off by 12%, 0.2% and 0.01% of the covariance; the family now
    # differentiates with a small first step.
    keywords, generator, _, _ = MODELS[name]
    frame = draw(n, seed, **generator)
    result = oe.arch(data=frame, y="y", x=["x1", "x2"], covariance="nonrobust", **keywords)
    function = oracle_for(result, frame)
    theta = reported(result)
    expected = np.linalg.inv(-hessian(function, theta))
    assert scaled_error(np.array(result.covariance_matrix), expected) < 2e-5
    assert "1e-06" in result.provenance["optimizer"]["hessian"]


@pytest.mark.parametrize("kind", ["opg", "nonrobust", "robust"])
def test_regressors_with_huge_offsets_do_not_change_the_fit(kind):
    # Before regressors were centred during estimation, x + 1e6 gave a slope standard
    # error 50 times too small, x + 1e7 a different "maximum" and x + 1e8 no convergence.
    frame = draw(500, 1, het=0.3)
    base = oe.arch(data=frame, y="y", x=["x1"], variance_x=["zv"], covariance=kind)
    moved = frame.assign(x1=1e3 * frame["x1"] + 5e7, zv=frame["zv"] + 1e6)
    other = oe.arch(data=moved, y="y", x=["x1"], variance_x=["zv"], covariance=kind)
    b, o = {c.term: c for c in base.coefficients}, {c.term: c for c in other.coefficients}
    assert other.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                            abs=1e-6)
    assert o["x1"].estimate == pytest.approx(b["x1"].estimate / 1e3, rel=1e-6)
    assert o["x1"].std_error == pytest.approx(b["x1"].std_error / 1e3, rel=1e-5)
    assert o["Intercept"].estimate + 5e7 * o["x1"].estimate == pytest.approx(
        b["Intercept"].estimate, abs=1e-5)
    assert o["HET:Intercept"].estimate + 1e6 * o["HET:zv"].estimate == pytest.approx(
        b["HET:Intercept"].estimate, abs=1e-4)
    for term in ("HET:zv", "ARCH:L1.arch", "ARCH:L1.garch"):
        assert o[term].estimate == pytest.approx(b[term].estimate, rel=1e-6, abs=1e-8)
        assert o[term].std_error == pytest.approx(b[term].std_error, rel=1e-5)
    # The constants' standard errors follow from the exact back-mapping c = c* - mean(x) b.
    covariance = np.array(other.covariance_matrix)
    terms = [c.term for c in other.coefficients]
    i, j = terms.index("Intercept"), terms.index("x1")
    implied = covariance[i, i] + 2 * 5e7 * covariance[i, j] + 5e7 ** 2 * covariance[j, j]
    assert math.sqrt(implied) == pytest.approx(b["Intercept"].std_error, rel=1e-4)


def test_centred_estimation_reports_raw_coefficients():
    # The reported constant belongs to the raw regressors: the oracle, which never
    # centres anything, has its maximum at the reported vector.
    frame = draw(600, 5).assign(x1=lambda d: d["x1"] + 250.0, x2=lambda d: 40.0 * d["x2"] - 90.0)
    result = oe.arch(data=frame, y="y", x=["x1", "x2"], covariance="nonrobust")
    function = oracle_for(result, frame)
    theta = reported(result)
    assert np.abs(scores(function, theta).sum(axis=0) * errors(result)).max() < 1e-4
    expected = np.linalg.inv(-hessian(function, theta))
    assert scaled_error(np.array(result.covariance_matrix), expected) < 1e-4


def test_negative_variance_coefficients_are_reported():
    result = oe.arch(data=draw(500, 1).iloc[:12], y="y")
    omega = next(c for c in result.coefficients if c.term == "ARCH:Intercept")
    assert omega.estimate < 0
    assert any("negative coefficients (ARCH:Intercept)" in warning for warning in result.warnings)
    # GJR: a negative arch coefficient alone is flagged, and so is arch + tarch < 0.
    frame = draw(700, 16, model="gjr")
    clean = oe.arch(data=frame, y="y", x=["x1", "x2"], model="gjr")
    assert not any("negative coefficients" in warning for warning in clean.warnings)
    flipped = oe.arch(data=frame.assign(y=-frame["y"]), y="y", x=["x1", "x2"], model="gjr")
    estimates = by_term(flipped)
    assert estimates["ARCH:L1.tarch"] < 0 < estimates["ARCH:L1.arch"] + estimates["ARCH:L1.tarch"]
    assert not any("negative coefficients" in warning for warning in flipped.warnings)


# ---- invariances ------------------------------------------------------------------------------


def by_term(result):
    return {c.term: c.estimate for c in result.coefficients}


def test_gjr_on_the_negated_outcome_gives_statas_tarch_parameterization():
    # 1(e < 0) of -y is 1(e > 0) of y, Stata's tarch indicator: arch' = arch + tarch and
    # tarch' = -tarch, everything else unchanged (the documented mapping to Stata).
    frame = draw(700, 16, model="gjr")
    base = oe.arch(data=frame, y="y", x=["x1", "x2"], model="gjr")
    flipped = oe.arch(data=frame.assign(y=-frame["y"]), y="y", x=["x1", "x2"], model="gjr")
    b, f = by_term(base), by_term(flipped)
    assert flipped.metrics["log_likelihood"] == pytest.approx(base.metrics["log_likelihood"],
                                                              rel=1e-11)
    for term in ("Intercept", "x1", "x2"):
        assert f[term] == pytest.approx(-b[term], rel=1e-6)
    assert f["ARCH:L1.arch"] == pytest.approx(b["ARCH:L1.arch"] + b["ARCH:L1.tarch"], rel=1e-5)
    assert f["ARCH:L1.tarch"] == pytest.approx(-b["ARCH:L1.tarch"], rel=1e-5)
    assert f["ARCH:L1.garch"] == pytest.approx(b["ARCH:L1.garch"], rel=1e-6)
    assert f["ARCH:Intercept"] == pytest.approx(b["ARCH:Intercept"], rel=1e-5)
    assert flipped.metrics["persistence"] == pytest.approx(base.metrics["persistence"], rel=1e-6)


def test_egarch_on_the_negated_outcome_flips_the_leverage_term():
    frame = draw(500, 21, model="egarch")
    base = oe.arch(data=frame, y="y", x=["x1"], model="egarch")
    flipped = oe.arch(data=frame.assign(y=-frame["y"]), y="y", x=["x1"], model="egarch")
    assert base.provenance["optimizer"].get("kink_observations") is None
    b, f = by_term(base), by_term(flipped)
    assert f["ARCH:L1.earch"] == pytest.approx(-b["ARCH:L1.earch"], rel=1e-5)
    for term in ("ARCH:L1.earch_a", "ARCH:L1.egarch", "ARCH:Intercept"):
        assert f[term] == pytest.approx(b[term], rel=1e-5, abs=1e-8)


def test_affine_changes_of_units():
    frame = draw(600, 41, dist="t")
    keywords = dict(y="y", x=["x1", "x2"], dist="t", ar=1, covariance="robust")
    base = oe.arch(data=frame, **keywords)
    other = oe.arch(data=frame.assign(y=250.0 * frame["y"] - 30.0, x2=frame["x2"] / 8.0 + 3.0),
                    **keywords)
    b, o = {c.term: c for c in base.coefficients}, {c.term: c for c in other.coefficients}
    assert o["x1"].estimate == pytest.approx(250 * b["x1"].estimate, rel=1e-6)
    assert o["x2"].estimate == pytest.approx(250 * 8 * b["x2"].estimate, rel=1e-6)
    assert o["Intercept"].estimate == pytest.approx(
        250 * b["Intercept"].estimate - 30 - 3 * o["x2"].estimate, rel=1e-6)
    assert o["ARCH:Intercept"].estimate == pytest.approx(250 ** 2 * b["ARCH:Intercept"].estimate,
                                                         rel=1e-5)
    assert o["ARCH:Intercept"].std_error == pytest.approx(
        250 ** 2 * b["ARCH:Intercept"].std_error, rel=1e-4)
    for term in ("ARMA:L1.ar", "ARCH:L1.arch", "ARCH:L1.garch", "/lndfm2"):
        assert o[term].estimate == pytest.approx(b[term].estimate, rel=1e-5)
        assert o[term].std_error == pytest.approx(b[term].std_error, rel=1e-4)
        assert o[term].p_value == pytest.approx(b[term].p_value, rel=1e-3)
    assert other.metrics["log_likelihood"] == pytest.approx(
        base.metrics["log_likelihood"] - 600 * math.log(250.0), rel=1e-10)
    assert other.tests["model"]["statistic"] == pytest.approx(base.tests["model"]["statistic"],
                                                              rel=1e-4)
    for name in ("ljung_box", "ljung_box_squared", "arch_lm_residuals", "jarque_bera"):
        assert other.tests[name]["statistic"] == pytest.approx(base.tests[name]["statistic"],
                                                               rel=1e-5)


def test_row_order_and_unused_columns_do_not_matter():
    frame = draw(500, 42)
    base = oe.arch(data=frame, y="y", x=["x1"], time="t", model="gjr")
    shuffled = frame.iloc[np.random.default_rng(0).permutation(500)].assign(noise=1.0)
    other = oe.arch(data=shuffled, y="y", x=["x1"], time="t", model="gjr")
    assert_allclose(reported(other), reported(base), rtol=1e-12)
    assert_allclose(errors(other), errors(base), rtol=1e-12)
    assert other.sample_positions != base.sample_positions       # rows reordered by time
    assert sorted(other.sample_positions) == list(range(500))
    reversed_time = oe.arch(data=frame.iloc[::-1].reset_index(drop=True), y="y", x=["x1"],
                            time="t", model="gjr")
    assert_allclose(reported(reversed_time), reported(base), rtol=1e-12)


# ---- residual diagnostics by explicit algebra -------------------------------------------------


def standardized(result, frame):
    spec = result.spec
    terms = [c.term for c in result.coefficients]
    y = frame[spec.outcome].to_numpy(float)
    x = np.column_stack([np.ones(len(frame))] + [frame[name].to_numpy(float)
                                                 for name in spec.predictors])
    e, h = paths(reported(result), terms, y, x, None, spec.options.get("model", "garch"),
                 spec.options.get("archm"))
    return e.real, h.real


def portmanteau(series, lags):
    n = len(series)
    centred = series - series.mean()
    r = np.array([centred[k:] @ centred[:n - k] for k in range(1, lags + 1)]) / (centred @ centred)
    return n * (n + 2) * (r ** 2 / (n - np.arange(1, lags + 1))).sum()


@pytest.mark.parametrize("test_lags", [None, 7])
def test_diagnostics_of_the_standardized_residuals(test_lags, solved):
    _, frame, _, _, _ = solved("arma")
    result = oe.arch(data=frame, y="y", x=["x1", "x2"], time="t", ar=1, ma=1, test_lags=test_lags)
    e, h = standardized(result, frame)
    z = e / np.sqrt(h)
    n = len(z)
    lags = test_lags or min(n // 2 - 2, 40)
    box = result.tests["ljung_box"]
    q = portmanteau(z, lags)
    assert box["statistic"] == pytest.approx(q, rel=1e-9) and box["df"] == lags == box["lags"]
    assert box["p_value"] == pytest.approx(stats.chi2.sf(q, lags), rel=1e-7)
    # Two ARMA parameters were estimated: the Box-Pierce correction is reported alongside.
    assert box["df_adjusted"] == lags - 2
    assert box["p_value_adjusted"] == pytest.approx(stats.chi2.sf(q, lags - 2), rel=1e-7)
    squared = result.tests["ljung_box_squared"]
    q2 = portmanteau(z ** 2, lags)
    assert squared["statistic"] == pytest.approx(q2, rel=1e-9) and squared["df"] == lags
    assert squared["p_value"] == pytest.approx(stats.chi2.sf(q2, lags), rel=1e-7)
    # Engle's LM test: (n - p) R^2 of z^2 on a constant and p of its lags.
    p = test_lags or 5
    s = z ** 2
    design = np.column_stack([np.ones(n - p)] + [s[p - j:n - j] for j in range(1, p + 1)])
    target = s[p:]
    residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    r2 = 1 - residual @ residual / ((target - target.mean()) @ (target - target.mean()))
    lm = result.tests["arch_lm_residuals"]
    assert lm["statistic"] == pytest.approx((n - p) * r2, rel=1e-8) and lm["df"] == p
    assert lm["p_value"] == pytest.approx(stats.chi2.sf((n - p) * r2, p), rel=1e-7)
    # Jarque-Bera with the biased (divisor n) moments.
    centred = z - z.mean()
    m2, m3, m4 = (np.mean(centred ** k) for k in (2, 3, 4))
    jb = n / 6 * ((m3 / m2 ** 1.5) ** 2 + (m4 / m2 ** 2 - 3) ** 2 / 4)
    assert result.tests["jarque_bera"]["statistic"] == pytest.approx(jb, rel=1e-9)
    assert result.tests["jarque_bera"]["p_value"] == pytest.approx(stats.chi2.sf(jb, 2), rel=1e-7)
    # The chart sample is the one-step conditional mean and its innovation.
    first = result.predictions[5]
    row = first["row"]
    assert first["residual"] == pytest.approx(e[row], rel=1e-8, abs=1e-10)
    assert first["observed"] == pytest.approx(frame["y"].iloc[row], rel=1e-12)
    assert_allclose(result.extra["conditional_variance_tail"], h[-400:], rtol=1e-9)


def test_persistence_and_unconditional_variance_definitions(solved):
    garch = by_term(solved("garch")[0]["opg"])
    metrics = solved("garch")[0]["opg"].metrics
    total = garch["ARCH:L1.arch"] + garch["ARCH:L1.garch"]
    assert metrics["persistence"] == pytest.approx(total, rel=1e-12)
    assert metrics["unconditional_variance"] == pytest.approx(garch["ARCH:Intercept"] / (1 - total),
                                                              rel=1e-12)
    gjr = by_term(solved("gjr")[0]["opg"])
    metrics = solved("gjr")[0]["opg"].metrics
    total = gjr["ARCH:L1.arch"] + 0.5 * gjr["ARCH:L1.tarch"] + gjr["ARCH:L1.garch"]
    assert metrics["persistence"] == pytest.approx(total, rel=1e-12)
    assert metrics["unconditional_variance"] == pytest.approx(gjr["ARCH:Intercept"] / (1 - total),
                                                              rel=1e-12)
    egarch = by_term(solved("egarch")[0]["opg"])
    assert solved("egarch")[0]["opg"].metrics["persistence"] == pytest.approx(
        egarch["ARCH:L1.egarch"], rel=1e-12)
    assert solved("egarch")[0]["opg"].metrics["unconditional_variance"] is None
    parch = by_term(solved("parch")[0]["opg"])
    power = parch["POWER:power"]
    moment = 2 ** (power / 2) * math.gamma((power + 1) / 2) / math.sqrt(math.pi)   # E|z|^power
    assert moment == pytest.approx(stats.norm.expect(lambda v: abs(v) ** power), rel=1e-8)
    assert solved("parch")[0]["opg"].metrics["persistence"] == pytest.approx(
        moment * parch["ARCH:L1.parch"] + parch["ARCH:L1.pgarch"], rel=1e-10)


# ---- forecasts by closed forms ------------------------------------------------------------------


def test_forecast_of_arma_garch_with_student_errors_in_closed_form():
    frame = draw(700, 51, dist="t", ar=0.5, ma=0.3)
    result = oe.arch(data=frame, y="y", x=["x1", "x2"], time="t", ar=1, ma=1, dist="t")
    c = by_term(result)
    e, h = standardized(result, frame)
    u = frame["y"] - c["Intercept"] - c["x1"] * frame["x1"] - c["x2"] * frame["x2"]
    steps = 6
    future = pd.DataFrame({"x1": np.linspace(-1, 1, steps), "x2": np.linspace(0.5, -0.5, steps)})
    table = oe.forecast(result, steps, exog=future, alpha=0.1)
    a, b, omega = c["ARCH:L1.arch"], c["ARCH:L1.garch"], c["ARCH:Intercept"]
    rho, theta = c["ARMA:L1.ar"], c["ARMA:L1.ma"]
    variance = [omega + a * e[-1] ** 2 + b * h[-1]]
    disturbance = [rho * u.iloc[-1] + theta * e[-1]]
    for _ in range(steps - 1):
        variance.append(omega + (a + b) * variance[-1])
        disturbance.append(rho * disturbance[-1])
    variance, disturbance = np.array(variance), np.array(disturbance)
    # Closed form of the variance path: geometric reversion to omega / (1 - a - b).
    long_run = omega / (1 - a - b)
    assert_allclose(variance, long_run + (a + b) ** np.arange(steps) * (variance[0] - long_run),
                    rtol=1e-12)
    mean = c["Intercept"] + c["x1"] * future["x1"] + c["x2"] * future["x2"] + disturbance
    weights = np.array([1.0] + [rho ** (j - 1) * (rho + theta) for j in range(1, steps)])
    mse = np.array([sum(weights[j] ** 2 * variance[k - j] for j in range(k + 1))
                    for k in range(steps)])
    nu = 2 + math.exp(c["/lndfm2"])
    critical = stats.t.ppf(0.95, nu) * math.sqrt((nu - 2) / nu)
    assert list(table["period"]) == list(range(701, 701 + steps))
    assert_allclose(table["variance_forecast"], variance, rtol=1e-9)
    assert_allclose(table["mean_forecast"], mean, rtol=1e-9)
    assert_allclose(table["std_error"], np.sqrt(mse), rtol=1e-9)
    assert_allclose(table["ci_low"], mean - critical * np.sqrt(mse), rtol=1e-7)
    assert_allclose(table["ci_high"], mean + critical * np.sqrt(mse), rtol=1e-7)
    assert table.attrs["critical_value"] == pytest.approx(critical, rel=1e-8)


def test_forecast_of_egarch_with_variance_regressors_and_student_errors(solved):
    fits, frame, _, _, _ = solved("egarch_t_het")
    result = fits["opg"]
    c = by_term(result)
    terms = [k.term for k in result.coefficients]
    y = frame["y"].to_numpy(float)
    x = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    e, h = paths(reported(result), terms, y, x, frame[["zv"]].to_numpy(float), "egarch")
    e, h = e.real, h.real
    steps = 5
    future = pd.DataFrame({"x1": np.zeros(steps), "x2": np.ones(steps),
                           "zv": np.linspace(-1.0, 1.0, steps)})
    table = oe.arch_forecast(result, steps, exog=future)
    nu = 2 + math.exp(c["/lndfm2"])
    # E|z| of the unit-variance Student t.
    mean_abs = math.sqrt(nu - 2) * math.gamma((nu - 1) / 2) / (math.sqrt(math.pi)
                                                               * math.gamma(nu / 2))
    assert mean_abs == pytest.approx(
        stats.t(nu, scale=math.sqrt((nu - 2) / nu)).expect(abs), rel=1e-7)
    z_last = e[-1] / math.sqrt(h[-1])
    log_h = [c["HET:Intercept"] + c["HET:zv"] * future["zv"][0] + c["ARCH:L1.earch"] * z_last
             + c["ARCH:L1.earch_a"] * (abs(z_last) - E_ABS_NORMAL)
             + c["ARCH:L1.egarch"] * math.log(h[-1])]
    for k in range(1, steps):
        log_h.append(c["HET:Intercept"] + c["HET:zv"] * future["zv"][k]
                     + c["ARCH:L1.earch_a"] * (mean_abs - E_ABS_NORMAL)
                     + c["ARCH:L1.egarch"] * log_h[-1])
    assert_allclose(table["variance_forecast"], np.exp(log_h), rtol=1e-9)
    assert_allclose(table["mean_forecast"], c["Intercept"] + c["x2"], rtol=1e-10)
    assert_allclose(table["std_error"], np.exp(0.5 * np.array(log_h)), rtol=1e-9)
    assert list(table["period"]) == list(range(351, 356))


def test_forecast_needs_the_stored_state():
    result = oe.arch(data=draw(300, 2), y="y")
    stripped = result.model_copy(update={"extra": {name: value for name, value
                                                   in result.extra.items() if name != "state"}})
    state = dict(result.extra["state"])
    state["variances"] = [-1.0]
    corrupt = result.model_copy(update={"extra": {**result.extra, "state": state}})
    for broken in (stripped, corrupt):
        with pytest.raises(AnalysisError) as error:
            oe.arch_forecast(broken, 3)
        assert error.value.code == "invalid_result" and "state" in str(error.value)
    assert len(oe.arch_forecast(result, 3)) == 3


# ---- EGARCH maxima on kinks ---------------------------------------------------------------------


def kink_conditions(result, frame, *, mask=False):
    """Check the first-order conditions of a maximum on kinks with the oracle; return it."""
    record = result.provenance["optimizer"]
    active, weights = record["kink_observations"], record["kink_weights"]
    assert active and all(abs(weight) <= 1.0 for weight in weights)
    function = oracle_for(result, frame)
    theta, se = reported(result), errors(result)
    e, h = function.path(theta)
    assert np.abs(e.real[active] / np.sqrt(h.real[active])).max() < 1e-8
    signs = np.where(e.real < 0, -1.0, 1.0)
    signs[active] = weights
    signs, flat = signs.tolist(), (list(active) if mask else None)
    total = function(theta).real.sum()
    assert result.metrics["log_likelihood"] == pytest.approx(total, rel=1e-10)

    def piece(point):
        return function(point, signs=signs, flat=flat)

    # Zero is a convex combination of the one-sided gradients (weights at the kinks).
    assert np.abs(scores(piece, theta).sum(axis=0) * se).max() < 1e-4
    # No nearby point has a higher likelihood.
    rng = np.random.default_rng(7)
    for radius in (1e-1, 1e-2, 1e-3, 1e-4):
        for _ in range(25):
            value = function(theta + radius * se * rng.normal(size=len(theta)))
            assert value is None or value.real.sum() <= total + 1e-9
    return piece, theta


@pytest.mark.parametrize("seed,kinks", [(502, 2), (519, 1)])
def test_egarch_maximum_between_two_pieces_is_found(seed, kinks):
    # The smooth maxima of two adjacent pieces lay in each other's piece (three or four
    # residuals changed sign each time), so re-freezing the piece cycled for ever and the
    # fit failed with nonconvergence. Kinks are now entered one at a time, at the first
    # crossing. Seed 502 ends on a vertex of two kinks, seed 519 on one kink.
    frame = kink_sample(600, seed, model="egarch")
    result = oe.arch(data=frame, y="y", x=["x"], model="egarch", covariance="nonrobust")
    assert len(result.provenance["optimizer"]["kink_observations"]) == kinks
    piece, theta = kink_conditions(result, frame)
    second = hessian(piece, theta)
    assert np.linalg.eigvalsh(second).max() < 0
    assert scaled_error(np.array(result.covariance_matrix), np.linalg.inv(-second)) < 1e-4


@pytest.mark.parametrize("seed", [2016, 2001, 2002])
def test_egarch_with_ged_errors_on_a_kink_has_finite_standard_errors(seed):
    # At a zero residual the curvature of the GED density is unbounded (shape < 2). The
    # finite-difference Hessian there was dominated by that one observation: seed 2016
    # reported an observed-information standard error of 4e-6 for the constant (and
    # 4e-10 with the robust covariance); seeds 2001 and 2002 did not converge at all.
    frame = kink_sample(600, seed, model="egarch", dist="ged")
    fits = {kind: oe.arch(data=frame, y="y", x=["x"], model="egarch", dist="ged",
                          covariance=kind) for kind in ("opg", "nonrobust", "robust")}
    result = fits["nonrobust"]
    assert math.exp(result.coefficients[-1].estimate) < 2.0          # GED shape
    assert any("lies on a kink" in warning for warning in result.warnings)
    piece, theta = kink_conditions(result, frame, mask=True)
    n = len(frame)
    gradients = scores(piece, theta)
    bread = np.linalg.inv(-hessian(piece, theta))
    outer = gradients.T @ gradients
    expected = {"opg": np.linalg.inv(outer), "nonrobust": bread,
                "robust": n / (n - 1) * bread @ outer @ bread}
    for kind, fit in fits.items():
        assert scaled_error(np.array(fit.covariance_matrix), expected[kind]) < 1e-4, kind
        ratio = errors(fit) / errors(fits["opg"])
        assert 0.4 < ratio.min() and ratio.max() < 2.5


# ---- richer specifications: likelihood, stationarity and OPG covariance ----------------------

RICH = {
    "sparse_lags": (dict(arch=[1, 3], garch=[2], ar=[1, 3], ma=[2]), dict(ar=0.4), 800, 61),
    "gjr22_ged": (dict(model="gjr", arch=2, garch=2, dist="ged"), dict(model="gjr", dist="ged"),
                  1200, 62),
    "igarch_ma": (dict(model="igarch", arch=2, garch=1, ma=[1]), dict(ma=0.3), 800, 63),
    "parch_m_ar": (dict(model="parch", archm="sd", ar=[2]), dict(model="parch", archm="sd",
                                                                 psi=0.3), 500, 64),
    "tarch_m_arma_het_t": (dict(model="tarch", archm="variance", ar=1, ma=1, dist="t",
                                variance_x=["zv"]),
                           dict(model="gjr", archm="variance", psi=0.2, ar=0.4, ma=0.2,
                                dist="t", het=0.3), 500, 65),
    "egarch_m_log": (dict(model="egarch", archm="log"), dict(model="egarch", archm="log",
                                                             psi=0.3), 400, 66),
    "arch3_no_constant": (dict(model="arch", arch=3, constant=False), dict(model="arch"),
                          700, 67),
}


@pytest.mark.parametrize("name", sorted(RICH))
def test_rich_specifications_against_the_oracle(name):
    keywords, generator, n, seed = RICH[name]
    frame = draw(n, seed, **generator)
    result = oe.arch(data=frame, y="y", x=["x1", "x2"], time="t", **keywords)
    record = result.provenance["optimizer"]
    function = oracle_for(result, frame)
    theta, se = reported(result), errors(result)
    signs = None
    if record.get("kink_observations"):
        e, _ = function.path(theta)
        signs = np.where(e.real < 0, -1.0, 1.0)
        signs[record["kink_observations"]] = record["kink_weights"]
        signs = signs.tolist()

    def piece(point):
        return function(point, signs=signs)

    assert result.metrics["log_likelihood"] == pytest.approx(function(theta).real.sum(),
                                                             rel=1e-10)
    gradients = scores(piece, theta)
    if keywords.get("model") == "igarch":
        # The gradient vanishes along the restricted directions only.
        terms = [c.term for c in result.coefficients]
        last = max(i for i, term in enumerate(terms) if term.endswith(".garch"))
        linked = [i for i, term in enumerate(terms)
                  if term.endswith((".arch", ".garch")) and i != last]
        jacobian = np.eye(len(theta))[:, [i for i in range(len(theta)) if i != last]]
        for column, i in enumerate(i for i in range(len(theta)) if i != last):
            if i in linked:
                jacobian[last, column] = -1.0
        reduced = gradients @ jacobian
        assert np.abs(reduced.sum(axis=0)).max() < 1e-4 * n ** 0.5
        expected = jacobian @ np.linalg.inv(reduced.T @ reduced) @ jacobian.T
    else:
        assert np.abs(gradients.sum(axis=0) * se).max() < 1e-4
        expected = np.linalg.inv(gradients.T @ gradients)
    covariance = np.array(result.covariance_matrix)
    sd = np.sqrt(np.diag(expected))
    assert np.abs((covariance - expected) / np.outer(sd, sd)).max() < 1e-4
