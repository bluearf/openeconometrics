"""GARCH extensions of ``oe.arch`` against independent references (MARKET-161).

References, none of which shares code with the Torch kernels:

* ``tests/fixtures/garch_extensions/arch_skewt.json`` -- Python ``arch`` 8.0 (Sheppard):
  Hansen skewed-t densities, per-observation likelihoods of GARCH / GJR / EGARCH with
  skewed-t errors at fixed parameters, and zero-mean maximum likelihood fits.
* ``tests/fixtures/garch_extensions/rugarch_sged.json`` -- R ``fGarch`` / ``rugarch``:
  Fernandez-Steel skewed-GED densities, sGARCH-sged filter likelihoods and the maximum of
  that likelihood found by ``stats::optim`` with a Richardson Hessian (numDeriv).
* explicit NumPy/SciPy formulas in this file (densities written out with ``gammaln``,
  the explicit-loop recursion of ``test_econ_arch_oracle``, ``scipy.integrate.quad``).

Generators: ``scripts/generate_garch_extension_oracles.py`` and ``.R``.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import integrate
from scipy.special import gammaln
from statsmodels.tools.numdiff import approx_fprime

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import streaming_registry
from openecon.econometrics.arch import densities, skewed
from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.core import make_spec
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle
from test_econ_arch_oracle import CASES, oracle_recursion, series, terms_of, unpack

FIXTURES = Path(__file__).parent / "fixtures" / "garch_extensions"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


ARCH_SKEWT = _fixture("arch_skewt.json")
RUGARCH_SGED = _fixture("rugarch_sged.json")


# ---- explicit NumPy skewed densities (independent of the Torch kernels) ---------------------


def hansen_logpdf(z, eta, lam):
    """Hansen (1994) eq. (10)-(11), written out."""
    log_c = gammaln((eta + 1) / 2) - gammaln(eta / 2) - 0.5 * np.log(np.pi * (eta - 2))
    c = np.exp(log_c)
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    z = np.asarray(z, dtype=float)
    piece = np.where(z < -a / b, 1 - lam, 1 + lam)
    return np.log(b) + log_c - (eta + 1) / 2 * np.log1p(((b * z + a) / piece) ** 2 / (eta - 2))


def fs_sged_logpdf(z, nu, xi):
    """Fernandez-Steel skewed GED standardized to unit variance, written out."""
    lam = np.sqrt(2 ** (-2 / nu) * np.exp(gammaln(1 / nu) - gammaln(3 / nu)))
    m1 = 2 ** (1 / nu) * lam * np.exp(gammaln(2 / nu) - gammaln(1 / nu))
    mu = m1 * (xi - 1 / xi)
    sigma = np.sqrt((1 - m1 ** 2) * (xi ** 2 + xi ** -2) + 2 * m1 ** 2 - 1)
    y = np.asarray(z, dtype=float) * sigma + mu
    w = np.where(y < 0, y * xi, y / xi)
    log_ged = np.log(nu) - np.log(lam) - (1 + 1 / nu) * np.log(2) - gammaln(1 / nu) \
        - 0.5 * np.abs(w / lam) ** nu
    return np.log(sigma) + np.log(2 / (xi + 1 / xi)) + log_ged


def oracle_logpdf(dist, z, tau):
    if dist == "skewt":
        return hansen_logpdf(z, 2 + math.exp(tau[0]), math.tanh(tau[1]))
    return fs_sged_logpdf(z, math.exp(tau[0]), math.exp(tau[1]))


SKEW_TAUS = {"skewt": [(math.log(4.0), 0.4), (math.log(1.2), -0.7), (math.log(30.0), 0.05)],
             "sged": [(math.log(1.3), 0.3), (math.log(2.5), -0.5), (math.log(0.9), 0.6)]}


def _split_points(dist, tau):
    switch = skewed._skewt_switch(tau) if dist == "skewt" else skewed._sged_switch(tau)
    return sorted({0.0, switch})


def _quad(function, dist, tau, upper=math.inf):
    points = [p for p in _split_points(dist, tau) if p < upper]
    edges = [-math.inf, *points, upper]
    return sum(integrate.quad(function, a, b, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
               for a, b in zip(edges[:-1], edges[1:]) if a < b)


# ---- densities --------------------------------------------------------------------------------


def test_hansen_skewed_t_density_matches_the_arch_package():
    for row in ARCH_SKEWT["skewt_density"]:
        z = torch.tensor(row["z"], dtype=torch.float64)
        tau = (math.log(row["eta"] - 2), math.atanh(row["lambda"]))
        mine = densities.log_density("skewt", z, torch.ones_like(z), tau, False)[0].numpy()
        assert_allclose(mine, row["log_density"], rtol=0, atol=1e-12)
        assert_allclose(mine, hansen_logpdf(row["z"], row["eta"], row["lambda"]), atol=1e-12)


def test_skewed_ged_density_matches_fgarch_and_rugarch():
    for row in RUGARCH_SGED["density"]:
        z = torch.tensor(row["z"], dtype=torch.float64)
        tau = (math.log(row["shape"]), math.log(row["skew"]))
        mine = densities.log_density("sged", z, torch.ones_like(z), tau, False)[0].numpy()
        assert_allclose(mine, row["fgarch_log_density"], rtol=0, atol=1e-11)
        assert_allclose(np.exp(mine), row["rugarch_density"], rtol=1e-11)
        assert_allclose(mine, fs_sged_logpdf(row["z"], row["shape"], row["skew"]), atol=1e-12)


@pytest.mark.parametrize("dist", ["skewt", "sged"])
def test_skewed_densities_are_standardized_and_their_moments_match_quadrature(dist):
    for tau in SKEW_TAUS[dist]:
        def pdf(z):
            return math.exp(float(oracle_logpdf(dist, z, tau)))
        assert _quad(pdf, dist, tau) == pytest.approx(1.0, abs=1e-10)
        assert _quad(lambda z: z * pdf(z), dist, tau) == pytest.approx(0.0, abs=1e-10)
        assert _quad(lambda z: z * z * pdf(z), dist, tau) == pytest.approx(1.0, abs=1e-9)
        for power in (0.7, 1.0, 1.5, 2.0):
            if dist == "skewt" and power >= 2 + math.exp(tau[0]):
                assert densities.abs_moment(dist, power, tau) == math.inf
                continue
            expected = _quad(lambda z, p=power: abs(z) ** p * pdf(z), dist, tau)
            assert densities.abs_moment(dist, power, tau) == pytest.approx(expected, rel=1e-9)
        expected = _quad(lambda z: z * z * pdf(z), dist, tau, upper=0.0)
        assert densities.negative_share(dist, tau) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("dist", ["skewt", "sged"])
def test_skewed_quantiles_invert_the_distribution_function(dist):
    for tau in SKEW_TAUS[dist]:
        def pdf(z):
            return math.exp(float(oracle_logpdf(dist, z, tau)))
        for probability in (0.001, 0.025, 0.3, 0.5, 0.8, 0.975, 0.999):
            q = skewed.quantile(dist, probability, tau)
            assert _quad(pdf, dist, tau, upper=q) == pytest.approx(probability, abs=1e-10)
        low, high = densities.quantiles(dist, 0.05, tau)
        assert low == pytest.approx(skewed.quantile(dist, 0.025, tau))
        assert high == pytest.approx(skewed.quantile(dist, 0.975, tau))


def test_symmetric_limits_reduce_to_the_t_and_ged_densities():
    rng = np.random.default_rng(3)
    e = torch.tensor(rng.normal(size=200) * 2, dtype=torch.float64)
    h = torch.tensor(rng.uniform(0.3, 3.0, size=200), dtype=torch.float64)
    for dist, base in (("skewt", "t"), ("sged", "ged")):
        tau = 0.7
        skew, d_e, d_h, d_tau = densities.log_density(dist, e, h, (tau, 0.0))
        plain, p_e, p_h, p_tau = densities.log_density(base, e, h, tau)
        assert_allclose(skew.numpy(), plain.numpy(), rtol=1e-13, atol=1e-13)
        assert_allclose(d_e.numpy(), p_e.numpy(), rtol=1e-11, atol=1e-12)
        assert_allclose(d_h.numpy(), p_h.numpy(), rtol=1e-11, atol=1e-12)
        assert_allclose(d_tau[0].numpy(), p_tau.numpy(), rtol=1e-9, atol=1e-10)
        assert densities.negative_share(dist, (tau, 0.0)) == pytest.approx(0.5, abs=1e-12)
        assert densities.abs_moment(dist, 1.3, (tau, 0.0)) == pytest.approx(
            densities.abs_moment(base, 1.3, tau), rel=1e-10)
        assert densities.quantiles(dist, 0.1, (tau, 0.0))[1] == pytest.approx(
            densities.critical_value(base, 0.1, tau), rel=1e-10)


@pytest.mark.parametrize("dist", ["skewt", "sged"])
def test_skewed_density_derivatives_match_numerical_differentiation(dist):
    rng = np.random.default_rng(11)
    e = rng.normal(size=60) * 1.7
    h = rng.uniform(0.4, 3.0, size=60)
    for tau in SKEW_TAUS[dist]:
        value, d_e, d_log_h, d_tau = densities.log_density(
            dist, torch.tensor(e), torch.tensor(h), tau)

        def total(point):
            return oracle_logpdf(dist, point[0] / math.sqrt(point[1]), point[2:]) \
                - 0.5 * math.log(point[1])

        assert_allclose(value.numpy(), oracle_logpdf(dist, e / np.sqrt(h), tau) - 0.5 * np.log(h),
                        atol=1e-12)
        for t in range(0, 60, 7):
            numerical = approx_fprime(np.array([e[t], h[t], *tau]), total, centered=True)
            assert d_e[t] == pytest.approx(numerical[0], rel=1e-6, abs=1e-7)
            assert d_log_h[t] == pytest.approx(numerical[1] * h[t], rel=1e-6, abs=1e-7)
            assert d_tau[0, t] == pytest.approx(numerical[2], rel=1e-6, abs=1e-7)
            assert d_tau[1, t] == pytest.approx(numerical[3], rel=1e-6, abs=1e-7)


# ---- likelihoods of the variance models with skewed errors ---------------------------------


def _theta(layout, mu, values):
    ix = layout.index
    theta = torch.zeros(ix.k, dtype=torch.float64)
    if ix.x:
        theta[ix.x[0]] = mu
    for position, value in values.items():
        theta[position] = value
    return theta


def _likelihood(model, dist, y, mu):
    y = torch.tensor(y, dtype=torch.float64)
    layout = Layout(model=model, dist=dist, arch_lags=(1,), garch_lags=(1,), k_x=int(mu != 0))
    x = torch.ones((len(y), layout.k_x), dtype=torch.float64)
    return layout, ArchLikelihood(y, x, torch.empty((len(y), 0), dtype=torch.float64), layout)


@pytest.mark.parametrize("row", ARCH_SKEWT["skewt_likelihood"], ids=lambda row: row["model"])
def test_skewed_t_likelihood_matches_the_arch_package_at_common_parameters(row):
    layout, like = _likelihood(row["model"], "skewt", row["y"], row["mu"])
    ix = layout.index
    p = row["variance_params"]
    order = {"garch": lambda: [ix.c, ix.a[0], ix.b[0]],
             "gjr": lambda: [ix.c, ix.a[0], ix.g[0], ix.b[0]],
             "egarch": lambda: [ix.c, ix.g[0], ix.a[0], ix.b[0]]}[row["model"]]()
    values = dict(zip(order, p, strict=True))
    values.update({ix.d[0]: math.log(row["eta"] - 2), ix.d[1]: math.atanh(row["lambda"])})
    out = like.evaluate(_theta(layout, row["mu"], values), derivatives=False)
    assert out.presample_variance == pytest.approx(row["h0"], rel=1e-14)
    assert_allclose(out.variance[:5].numpy(), row["sigma2_head"], rtol=1e-13)
    assert float(out.variance[-1]) == pytest.approx(row["sigma2_last"], rel=1e-11)
    assert_allclose(out.loglik.numpy(), row["loglik"], rtol=0, atol=1e-11)
    assert out.value == pytest.approx(sum(row["loglik"]), rel=1e-12)


@pytest.mark.parametrize("row", RUGARCH_SGED["likelihood"], ids=lambda row: f"mu={row['mu']}")
def test_skewed_ged_likelihood_matches_the_rugarch_filter_at_common_parameters(row):
    layout, like = _likelihood("garch", "sged", row["y"], row["mu"])
    ix = layout.index
    values = {ix.c: row["omega"], ix.a[0]: row["alpha"], ix.b[0]: row["beta"],
              ix.d[0]: math.log(row["shape"]), ix.d[1]: math.log(row["skew"])}
    out = like.evaluate(_theta(layout, row["mu"], values), derivatives=False)
    assert out.presample_variance == pytest.approx(row["h0"], rel=1e-13)
    assert_allclose(out.loglik.numpy(), row["loglik"], rtol=0, atol=1e-11)
    assert out.value == pytest.approx(sum(row["loglik"]), rel=1e-12)


def _skew_case(name, dist):
    """An existing oracle case of test_econ_arch_oracle with skewed errors."""
    keywords, theta = CASES[name]
    keywords = dict(keywords)
    theta = list(theta)
    if keywords.get("dist", "normal") != "normal":
        theta = theta[:-1]
    keywords["dist"] = dist
    tau = (math.log(5.0), 0.3) if dist == "skewt" else (math.log(1.4), -0.25)
    layout = Layout(**keywords)
    y, x, z = series(240, 0)
    x, z = x[:, :layout.k_x], z[:, :layout.k_z]
    like = ArchLikelihood(torch.tensor(y), torch.tensor(x), torch.tensor(z), layout)
    full = torch.tensor([*theta, *tau], dtype=torch.float64)
    assert len(full) == layout.k
    return layout, like, full, (y, x, z)


def _skew_oracle(layout, theta, data):
    y, x, z = data
    terms = terms_of(layout, layout.k_x, layout.k_z)
    core = len(terms) - 2
    p = unpack(terms[:core], list(theta[:core]), layout.model)
    path = oracle_recursion(p, y, x, z, model=layout.model, archm=layout.archm)
    e, h, _ = path
    return oracle_logpdf(layout.dist, e / np.sqrt(h), theta[core:]) - 0.5 * np.log(h)


SKEW_CASES = ["garch11", "gjr_ged", "garch_arma", "garch_het", "igarch_full", "parch_all",
              "egarch11", "egarch_all", "garch_m_sd_arma", "gjr_m_log_het"]


@pytest.mark.parametrize("dist", ["skewt", "sged"])
@pytest.mark.parametrize("name", SKEW_CASES)
def test_skewed_errors_in_every_variance_model_match_the_explicit_recursion(name, dist):
    layout, like, theta, data = _skew_case(name, dist)
    out = like.evaluate(theta)
    assert_allclose(out.loglik.numpy(), _skew_oracle(layout, theta.numpy(), data),
                    rtol=1e-10, atol=1e-10)

    def objective(point):
        evaluation = like.evaluate(point)
        if evaluation is None:
            return torch.tensor(math.nan), torch.full_like(point, math.nan)
        return evaluation.value, evaluation.gradient

    assert check_derivatives(objective, theta)["gradient_max_rel_error"] < 1e-6
    scores = like.evaluate(theta, scores=True).scores.numpy()
    assert_allclose(scores.sum(axis=0), out.gradient.numpy(), rtol=1e-9, atol=1e-9)
    if layout.kind in {"garch", "gjr"} and layout.archm is None:
        # The filter engine's reverse-mode gradient against the forward scan engine.
        general = like.evaluate(theta, scores=True, engine="scan")
        assert_allclose(general.scores.numpy(), scores, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("dist", ["skewt", "sged"])
def test_per_observation_scores_match_numerical_scores_of_the_oracle(dist):
    layout, like, theta, data = _skew_case("gjr_m_log_het", dist)
    out = like.evaluate(theta, scores=True)
    numerical = approx_fprime(theta.numpy(), lambda point: _skew_oracle(layout, point, data),
                              centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=2e-5, atol=2e-6)


# ---- maximum likelihood against the packages ------------------------------------------------


_ARCH_ORDER = {"garch": ["ARCH:Intercept", "ARCH:L1.arch", "ARCH:L1.garch"],
               "gjr": ["ARCH:Intercept", "ARCH:L1.arch", "ARCH:L1.tarch", "ARCH:L1.garch"],
               "egarch": ["ARCH:Intercept", "ARCH:L1.earch_a", "ARCH:L1.earch",
                          "ARCH:L1.egarch"]}


def _natural(fit, names, transforms):
    """Estimates and delta-method standard errors of ``names`` with reported transforms."""
    terms = [c.term for c in fit.coefficients]
    theta = np.array([c.estimate for c in fit.coefficients])
    covariance = np.array(fit.covariance_matrix)
    index = [terms.index(name) for name in names]
    values, jacobian = theta[index].copy(), np.eye(len(index))
    for position, (value_map, slope) in transforms.items():
        jacobian[position, position] = slope(values[position])
        values[position] = value_map(values[position])
    return values, np.sqrt(np.diag(jacobian @ covariance[np.ix_(index, index)] @ jacobian.T))


@pytest.mark.parametrize("row", ARCH_SKEWT["skewt_fits"], ids=lambda row: row["model"])
def test_skewed_t_maximum_likelihood_matches_the_arch_package(row):
    # Zero mean: oe.arch's presample value mean(y^2) is fixed, so the likelihood is exactly
    # the one arch maximizes with backcast=mean(y^2). Estimates differ by the optimizers
    # (arch: SLSQP with ftol 1e-14), standard errors by the numerical Hessians: arch's
    # "classic" covariance differentiates the likelihood twice with its own step, ours
    # differentiates the analytic gradient with Ridders extrapolation. The GJR threshold
    # term has a second derivative that jumps at a zero residual, where arch's larger step
    # straddles the kink (up to 0.15% of a standard error).
    fit = oe.arch(data=pd.DataFrame({"y": row["y"]}), y="y", model=row["model"], dist="skewt",
                  constant=False, covariance="nonrobust")
    assert fit.metrics["log_likelihood"] >= row["loglik"] - 1e-9
    assert fit.metrics["log_likelihood"] == pytest.approx(row["loglik"], abs=1e-6)
    names = _ARCH_ORDER[row["model"]] + ["/lndfm2", "/atlambda"]
    k = len(names)
    values, errors = _natural(fit, names, {
        k - 2: (lambda t: 2 + math.exp(t), math.exp),
        k - 1: (math.tanh, lambda t: 1 - math.tanh(t) ** 2)})
    scale = np.maximum(np.abs(row["params"]), 0.05)
    assert np.max(np.abs(values - row["params"]) / scale) < 5e-5
    tolerance = 2e-3 if row["model"] == "gjr" else 1e-4
    assert_allclose(errors, row["std_errors"], rtol=tolerance)


def test_skewed_ged_maximum_likelihood_matches_optim_on_the_rugarch_likelihood():
    row = RUGARCH_SGED["mle"]
    fit = oe.arch(data=pd.DataFrame({"y": row["y"]}), y="y", dist="sged", constant=False,
                  covariance="nonrobust")
    assert fit.metrics["log_likelihood"] == pytest.approx(row["loglik"], abs=1e-7)
    values, errors = _natural(fit, ["ARCH:Intercept", "ARCH:L1.arch", "ARCH:L1.garch", "/lnxi",
                                    "/lnshape"],
                              {3: (math.exp, math.exp), 4: (math.exp, math.exp)})
    assert_allclose(values, row["params"], rtol=2e-6)
    # Numerical Hessians of a GED likelihood with shape below 2 depend on the step: the
    # exact Hessian has terms |e_t|^(shape - 2) that are large for the residuals nearest to
    # zero (docs/econometrics/arch.md, Limitations). OPG errors are compared exactly below.
    assert_allclose(errors, row["std_errors"], rtol=2e-3)
    # The package's own fit starts the recursion at h_1 = mean(e^2) (no presample step), a
    # different likelihood: estimates agree to O(1/n).
    assert_allclose(values, RUGARCH_SGED["package_fit"]["params"], rtol=2e-3)


@pytest.mark.parametrize("dist", ["skewt", "sged"])
def test_opg_and_robust_covariances_equal_the_numerical_score_formulas(dist):
    row = ARCH_SKEWT["skewt_fits"][1] if dist == "skewt" else RUGARCH_SGED["mle"]
    data = pd.DataFrame({"y": row["y"]})
    fits = {kind: oe.arch(data=data, y="y", model="gjr" if dist == "skewt" else "garch",
                          dist=dist, constant=False, covariance=kind)
            for kind in ("opg", "nonrobust", "robust")}
    theta = np.array([c.estimate for c in fits["opg"].coefficients])
    layout = Layout(model="gjr" if dist == "skewt" else "garch", dist=dist, arch_lags=(1,),
                    garch_lags=(1,))
    y = torch.tensor(row["y"], dtype=torch.float64)
    like = ArchLikelihood(y, torch.empty((len(y), 0), dtype=torch.float64),
                          torch.empty((len(y), 0), dtype=torch.float64), layout)

    def per_observation(point):
        return like.evaluate(torch.tensor(point), derivatives=False).loglik.numpy()

    scores = approx_fprime(theta, per_observation, centered=True)
    opg = np.linalg.inv(scores.T @ scores)
    assert_allclose(np.array(fits["opg"].covariance_matrix), opg, rtol=1e-4,
                    atol=1e-10 * np.abs(opg).max())
    information = np.linalg.inv(np.array(fits["nonrobust"].covariance_matrix))
    n = len(row["y"])
    bread = np.linalg.inv(information)
    sandwich = n / (n - 1) * bread @ (scores.T @ scores) @ bread
    assert_allclose(np.array(fits["robust"].covariance_matrix), sandwich, rtol=1e-4,
                    atol=1e-10 * np.abs(sandwich).max())


# ---- reported quantities, forecasts and persistence -----------------------------------------


@pytest.fixture(scope="module")
def skewed_gjr():
    row = ARCH_SKEWT["skewt_fits"][1]
    data = pd.DataFrame({"day": np.arange(1, len(row["y"]) + 1), "ret": np.array(row["y"]) + 0.1})
    return oe.arch(data=data, y="ret", time="day", model="gjr", dist="skewt")


def test_gjr_persistence_and_forecasts_use_the_skewed_negative_share(skewed_gjr):
    fit = skewed_gjr
    est = {c.term: c.estimate for c in fit.coefficients}
    tau = (est["/lndfm2"], est["/atlambda"])
    share = densities.negative_share("skewt", tau)
    assert share > 0.5            # lambda < 0: the left tail carries more variance
    persistence = est["ARCH:L1.arch"] + share * est["ARCH:L1.tarch"] + est["ARCH:L1.garch"]
    assert fit.metrics["persistence"] == pytest.approx(persistence, rel=1e-12)
    assert fit.metrics["unconditional_variance"] == pytest.approx(
        est["ARCH:Intercept"] / (1 - persistence), rel=1e-12)
    record = fit.extra["distribution"]
    assert record["df"] == pytest.approx(2 + math.exp(tau[0]))
    assert record["lambda"] == pytest.approx(math.tanh(tau[1]))
    assert record["negative_share"] == pytest.approx(share)
    table = oe.forecast(fit, 6)
    state = fit.extra["state"]
    e, h = state["residuals"][-1], state["variances"][-1]
    a, g, b = est["ARCH:L1.arch"], est["ARCH:L1.tarch"], est["ARCH:L1.garch"]
    expected = [est["ARCH:Intercept"] + (a + g * (e < 0)) * e * e + b * h]
    for _ in range(5):
        expected.append(est["ARCH:Intercept"] + (a + share * g + b) * expected[-1])
    assert_allclose(table["variance_forecast"].to_numpy(), expected, rtol=1e-12)
    low, high = densities.quantiles("skewt", 0.05, tau)
    assert low != pytest.approx(-high)                      # asymmetric interval
    sd = np.sqrt(expected[0])
    assert table["ci_low"].iloc[0] == pytest.approx(est["Intercept"] + low * sd, rel=1e-12)
    assert table["ci_high"].iloc[0] == pytest.approx(est["Intercept"] + high * sd, rel=1e-12)
    assert table.attrs["interval_quantiles"] == [low, high]


def test_egarch_and_power_arch_forecasts_use_moments_of_the_skewed_density():
    row = RUGARCH_SGED["mle"]
    data = pd.DataFrame({"y": row["y"]})
    egarch = oe.arch(data=data, y="y", model="egarch", dist="sged", constant=False)
    est = {c.term: c.estimate for c in egarch.coefficients}
    tau = (est["/lnshape"], est["/lnxi"])
    centred = densities.abs_moment("sged", 1.0, tau) - math.sqrt(2 / math.pi)
    state = egarch.extra["state"]
    z = state["residuals"][-1] / math.sqrt(state["variances"][-1])
    log_h = est["ARCH:Intercept"] + est["ARCH:L1.earch"] * z \
        + est["ARCH:L1.earch_a"] * (abs(z) - math.sqrt(2 / math.pi)) \
        + est["ARCH:L1.egarch"] * math.log(state["variances"][-1])
    path = [log_h]
    for _ in range(3):
        path.append(est["ARCH:Intercept"] + est["ARCH:L1.earch_a"] * centred
                    + est["ARCH:L1.egarch"] * path[-1])
    assert_allclose(oe.forecast(egarch, 4)["variance_forecast"].to_numpy(), np.exp(path),
                    rtol=1e-12)
    parch = oe.arch(data=data, y="y", model="parch", dist="sged", constant=False)
    est = {c.term: c.estimate for c in parch.coefficients}
    moment = densities.abs_moment("sged", est["POWER:power"], (est["/lnshape"], est["/lnxi"]))
    assert parch.metrics["persistence"] == pytest.approx(
        moment * est["ARCH:L1.parch"] + est["ARCH:L1.pgarch"], rel=1e-12)


def test_result_round_trips_and_renders(skewed_gjr):
    fit = skewed_gjr
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    assert [c.term for c in restored.coefficients][-2:] == ["/lndfm2", "/atlambda"]
    assert restored.extra["distribution"] == fit.extra["distribution"]
    assert_allclose(oe.forecast(restored, 3)["mean_forecast"].to_numpy(),
                    oe.forecast(fit, 3)["mean_forecast"].to_numpy(), rtol=0, atol=0)
    assert "Hansen skewed t" in fit.summary()
    assert "atlambda" in fit.to_latex()
    assert fit.provenance["stata_parity_validated"] is False


# ---- refusals --------------------------------------------------------------------------------


def test_unknown_distribution_and_dataset_route_are_refused():
    data = pd.DataFrame({"y": RUGARCH_SGED["mle"]["y"]})
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=data, y="y", dist="sstd")
    assert error.value.code == "invalid_spec"
    spec = make_spec("arch", outcome="y", options={"dist": "skewt"})
    assert streaming_registry.supports_spec(spec) is False
    plain = make_spec("arch", outcome="y", options={"dist": "t"})
    assert streaming_registry.supports_spec(plain) is True
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=Dataset.from_frame(data), y="y", dist="sged")
    assert error.value.code == "streaming_unsupported"


# ==== component GARCH (model="cgarch") =========================================================

RUGARCH_CSGARCH = _fixture("rugarch_csgarch.json")


def component_path(params, e, start):
    """Engle-Lee two-state recursion written out (rugarch's csgarchfilter, filters.c).

    ``start`` is (h_1, q_1); s_(t-1) = e_(t-1)^2 drives both components.
    """
    omega, alpha, beta, rho, phi = params
    h, q = start
    hs, qs = [h], [q]
    for t in range(1, len(e)):
        s = e[t - 1] ** 2
        q_next = omega + rho * q + phi * (s - h)
        h = q_next + alpha * (s - q) + beta * (h - q)
        q = q_next
        hs.append(h)
        qs.append(q)
    return np.array(hs), np.array(qs)


def complex_step_gradient(function, point, step=1e-20):
    point = np.asarray(point, dtype=complex)
    out = np.empty(len(point))
    for k in range(len(point)):
        shifted = point.copy()
        shifted[k] += 1j * step
        out[k] = np.imag(function(shifted)) / step
    return out


def complex_step_hessian(function, point, relative=1e-5):
    point = np.asarray(point, dtype=float)
    hessian = np.empty((len(point), len(point)))
    for j in range(len(point)):
        delta = relative * max(abs(point[j]), 1e-2)
        up, down = point.copy(), point.copy()
        up[j] += delta
        down[j] -= delta
        hessian[:, j] = (complex_step_gradient(function, up)
                         - complex_step_gradient(function, down)) / (2 * delta)
    return 0.5 * (hessian + hessian.T)


def component_loglik(params, e, start=None, dist="normal", tau=None):
    """Per-observation log likelihood; ``start=None`` is oe.arch's presample convention."""
    if start is None:
        h0 = float(np.mean(e ** 2))
        start = (params[0] + params[3] * h0,) * 2         # x_0 = (h0, h0), s_0 = h0
    h, _ = component_path(params, e, start)
    if np.any(np.real(h) <= 0):
        return np.full(len(e), -np.inf)
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(h) + e ** 2 / h)
    return oracle_logpdf(dist, e / np.sqrt(h), tau) - 0.5 * np.log(h)


@pytest.mark.parametrize("row", RUGARCH_CSGARCH["likelihood"], ids=lambda row: f"mu={row['mu']}")
def test_component_recursion_reproduces_the_rugarch_filter(row):
    e = np.array(row["y"]) - row["mu"]
    params = [row[name] for name in ("omega", "alpha", "beta", "rho", "phi")]
    permanent = row["omega"] / (1 - row["rho"])
    h, q = component_path(params, e, (float(np.mean(e ** 2)) + permanent, permanent))
    assert_allclose(h, row["h"], rtol=1e-13)
    assert_allclose(q, row["q"], rtol=1e-13)
    assert_allclose(-0.5 * (np.log(2 * np.pi) + np.log(h) + e ** 2 / h), row["loglik"],
                    atol=1e-12)


def _component(y, *, mean=True, ar=(), ma=(), dist="normal"):
    y = torch.tensor(y, dtype=torch.float64)
    layout = Layout(model="cgarch", dist=dist, arch_lags=(1,), garch_lags=(1,), ar_lags=ar,
                    ma_lags=ma, k_x=int(mean))
    from openecon.econometrics.arch.extended import likelihood
    like = likelihood(y, torch.ones((len(y), layout.k_x), dtype=torch.float64),
                      torch.empty((len(y), 0), dtype=torch.float64), layout)
    return layout, like


@pytest.mark.parametrize("row", RUGARCH_CSGARCH["likelihood"], ids=lambda row: f"mu={row['mu']}")
def test_component_likelihood_matches_the_explicit_recursion(row):
    layout, like = _component(row["y"], mean=row["mu"] != 0)
    ix = layout.index
    params = [row[name] for name in ("omega", "alpha", "beta", "rho", "phi")]
    theta = torch.zeros(ix.k, dtype=torch.float64)
    if ix.x:
        theta[ix.x[0]] = row["mu"]
    theta[[ix.c, ix.a[0], ix.b[0], ix.r[0], ix.r[1]]] = torch.tensor(params, dtype=torch.float64)
    out = like.evaluate(theta, scores=True)
    e = np.array(row["y"]) - row["mu"]
    assert_allclose(out.loglik.numpy(), component_loglik(params, e), rtol=0, atol=1e-11)
    h0 = float(np.mean(e ** 2))
    _, q = component_path(params, e, (params[0] + params[3] * h0,) * 2)
    assert out.extra_state["permanent"][0] == pytest.approx(q[-1], rel=1e-12)

    def oracle(point):
        mu = point[ix.x[0]] if ix.x else 0.0
        return component_loglik([point[ix.c], point[ix.a[0]], point[ix.b[0]], point[ix.r[0]],
                                 point[ix.r[1]]], np.array(row["y"]) - mu)

    numerical = approx_fprime(theta.numpy(), oracle, centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dist", ["normal", "t", "skewt"])
def test_component_gradient_with_arma_disturbances_and_other_densities(dist):
    y, _, _ = series(300, 4)
    layout, like = _component(y, ar=(1,), ma=(1,), dist=dist)
    ix = layout.index
    theta = torch.zeros(ix.k, dtype=torch.float64)
    theta[ix.x[0]], theta[ix.ar[0]], theta[ix.ma[0]] = 0.3, 0.4, -0.2
    theta[[ix.c, ix.a[0], ix.b[0], ix.r[0], ix.r[1]]] = torch.tensor(
        [0.05, 0.1, 0.6, 0.97, 0.04], dtype=torch.float64)
    if ix.d:
        theta[ix.d[0]] = 1.2
        if len(ix.d) > 1:
            theta[ix.d[1]] = -0.3

    def objective(point):
        evaluation = like.evaluate(point)
        if evaluation is None:
            return torch.tensor(math.nan), torch.full_like(point, math.nan)
        return evaluation.value, evaluation.gradient

    assert check_derivatives(objective, theta)["gradient_max_rel_error"] < 1e-6
    out = like.evaluate(theta, scores=True)

    def oracle(point):
        # ARMA residuals by explicit recursion, then the component recursion.
        m = y - point[ix.x[0]]
        e = np.zeros_like(m)
        for t in range(len(m)):
            e[t] = m[t] - (point[ix.ar[0]] * m[t - 1] if t else 0.0) \
                - (point[ix.ma[0]] * e[t - 1] if t else 0.0)
        params = [point[ix.c], point[ix.a[0]], point[ix.b[0]], point[ix.r[0]], point[ix.r[1]]]
        tau = tuple(point[ix.d]) if len(ix.d) > 1 else None
        if dist == "t":
            h0 = float(np.mean(e ** 2))
            h, _ = component_path(params, e, (params[0] + params[3] * h0,) * 2)
            nu = 2 + math.exp(point[ix.d[0]])
            scale = np.sqrt(h * (nu - 2) / nu)
            from scipy import stats
            return stats.t.logpdf(e / scale, nu) - np.log(scale)
        return component_loglik(params, e, dist=dist, tau=tau)

    assert_allclose(out.loglik.numpy(), oracle(theta.numpy()), atol=1e-11)
    numerical = approx_fprime(theta.numpy(), oracle, centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=2e-5, atol=2e-6)


@pytest.fixture(scope="module")
def component_fit():
    row = RUGARCH_CSGARCH["package_fit"]
    data = pd.DataFrame({"period": np.arange(1, len(row["y"]) + 1), "y": row["y"]})
    return {kind: oe.arch(data=data, y="y", time="period", model="cgarch", constant=False,
                          covariance=kind) for kind in ("opg", "nonrobust", "robust")}


_COMPONENT_TERMS = ["COMPONENT:Intercept", "ARCH:L1.arch", "ARCH:L1.garch", "COMPONENT:L1.rho",
                    "COMPONENT:L1.phi"]


def test_component_maximum_likelihood_matches_an_independent_maximization(component_fit):
    from scipy.optimize import minimize

    y = np.array(RUGARCH_CSGARCH["package_fit"]["y"])

    def negative(params):
        value = -component_loglik(params, y).sum()
        return value if np.isfinite(value) else 1e10

    start = [0.02, 0.07, 0.4, 0.97, 0.03]
    found = minimize(negative, start, method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 40000})
    found = minimize(negative, found.x, method="BFGS", options={"gtol": 1e-8})
    fit = component_fit["nonrobust"]
    est = {c.term: c.estimate for c in fit.coefficients}
    values = np.array([est[name] for name in _COMPONENT_TERMS])
    assert fit.metrics["log_likelihood"] >= -found.fun - 1e-9
    assert_allclose(values, found.x, rtol=5e-6, atol=1e-8)
    # Full covariance against derivatives of the explicit recursion: complex-step gradients
    # (exact to rounding) and central differences of them for the Hessian.
    terms = [c.term for c in fit.coefficients]
    order = [terms.index(name) for name in _COMPONENT_TERMS]
    hessian = complex_step_hessian(lambda p: component_loglik(p, y).sum(), values)
    assert_allclose(np.array(fit.covariance_matrix)[np.ix_(order, order)],
                    np.linalg.inv(-hessian), rtol=1e-5)
    scores = approx_fprime(values, lambda p: component_loglik(p, y), centered=True)
    opg = np.array(component_fit["opg"].covariance_matrix)[np.ix_(order, order)]
    assert_allclose(opg, np.linalg.inv(scores.T @ scores), rtol=1e-5)
    bread = np.linalg.inv(-hessian)
    n = len(y)
    robust = np.array(component_fit["robust"].covariance_matrix)[np.ix_(order, order)]
    assert_allclose(robust, n / (n - 1) * bread @ scores.T @ scores @ bread, rtol=5e-4)


def test_component_estimates_agree_with_rugarch_within_its_start_up_difference(component_fit):
    # rugarch starts at h_1 = mean(y^2) + omega/(1 - rho), q_1 = omega/(1 - rho) instead of
    # oe.arch's presample step: a different likelihood whose maximum differs by O(1/n),
    # small against the standard errors. rugarch's own standard errors come from a
    # numerical Hessian with relative step 0.1 that crosses rho = 1 and are not compared.
    row = RUGARCH_CSGARCH["package_fit"]
    fit = component_fit["nonrobust"]
    est = {c.term: (c.estimate, c.std_error) for c in fit.coefficients}
    for name, value in zip(_COMPONENT_TERMS, row["params"], strict=True):
        estimate, error = est[name]
        assert abs(estimate - value) < 0.25 * error, name
    assert fit.metrics["log_likelihood"] == pytest.approx(row["loglik"], abs=1.0)


def test_component_persistence_state_and_forecasts(component_fit):
    fit = component_fit["opg"]
    est = {c.term: c.estimate for c in fit.coefficients}
    omega, alpha, beta = est["COMPONENT:Intercept"], est["ARCH:L1.arch"], est["ARCH:L1.garch"]
    rho, phi = est["COMPONENT:L1.rho"], est["COMPONENT:L1.phi"]
    assert fit.metrics["persistence"] == pytest.approx(max(rho, alpha + beta))
    assert fit.metrics["unconditional_variance"] == pytest.approx(omega / (1 - rho))
    y = np.array(RUGARCH_CSGARCH["package_fit"]["y"])
    params = [omega, alpha, beta, rho, phi]
    h0 = float(np.mean(y ** 2))
    h, q = component_path(params, y, (omega + rho * h0,) * 2)
    state = fit.extra["state"]
    assert state["variances"][-1] == pytest.approx(h[-1], rel=1e-10)
    assert state["permanent"][0] == pytest.approx(q[-1], rel=1e-10)
    # Two-state recursion continued with E[e^2] = h beyond the sample.
    s, hh, qq, expected = y[-1] ** 2, h[-1], q[-1], []
    for _ in range(8):
        q_next = omega + rho * qq + phi * (s - hh)
        hh = q_next + alpha * (s - qq) + beta * (hh - qq)
        qq, s = q_next, hh
        expected.append(hh)
    table = oe.forecast(fit, 8)
    assert_allclose(table["variance_forecast"].to_numpy(), expected, rtol=1e-10)
    assert table["period"].tolist() == list(range(len(y) + 1, len(y) + 9))
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    assert_allclose(oe.forecast(restored, 8)["variance_forecast"].to_numpy(), expected,
                    rtol=1e-10)
    rebuilt = oe.forecast(fit, 3, data=pd.DataFrame({"period": np.arange(1, len(y) + 1),
                                                     "y": y}))
    assert_allclose(rebuilt["variance_forecast"].to_numpy(), expected[:3], rtol=1e-10)
    text = fit.summary()
    assert "CGARCH(1,1)" in text and "COMPONENT:L1.rho" in text
    assert "COMPONENT:L1.phi" in fit.to_latex()


def test_component_refusals():
    data = pd.DataFrame({"y": RUGARCH_CSGARCH["package_fit"]["y"][:400],
                         "z": np.linspace(0, 1, 400)})
    for keywords in ({"arch": 2}, {"garch": [1, 2]}, {"archm": "variance"},
                     {"variance_x": ["z"]}):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=data, y="y", model="cgarch", **keywords)
        assert error.value.code == "unsupported_combination", keywords
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=Dataset.from_frame(data), y="y", model="cgarch")
    assert error.value.code == "streaming_unsupported"
    broken = ResultBundle.model_validate_json(
        oe.arch(data=data, y="y", model="cgarch").model_dump_json())
    broken.extra["state"].pop("permanent")
    with pytest.raises(AnalysisError) as error:
        oe.forecast(broken, 2)
    assert error.value.code == "invalid_result"


# ==== FIGARCH (model="figarch") ================================================================

ARCH_FIGARCH = _fixture("arch_figarch.json")


def bbm_weights(phi, d, beta, count):
    """lambda_1..lambda_K by the published recursion (BBM 1996; Python arch figarch_weights)."""
    lam, delta = np.empty(count, dtype=complex), np.empty(count, dtype=complex)
    lam[0], delta[0] = phi - beta + d, d
    for i in range(1, count):
        delta[i] = (i - d) / (i + 1) * delta[i - 1]
        lam[i] = beta * lam[i - 1] + delta[i] - phi * delta[i - 1]
    return lam


def figarch_loglik(omega, phi, d, beta, e, count=1000, dist="normal", tau=None):
    """Explicit loop over time: presample squared innovations equal mean(e^2)."""
    lam = bbm_weights(phi, d, beta, count)
    s = e ** 2
    h0 = np.mean(s)
    h = np.empty(len(e), dtype=complex)
    for t in range(len(e)):
        lags = np.arange(1, count + 1)
        past = np.where(t - lags >= 0, s[np.clip(t - lags, 0, None)], h0)
        h[t] = omega / (1 - beta) + np.sum(lam * past)
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(h) + s / h)
    return oracle_logpdf(dist, np.real(e / np.sqrt(h)), tau) - 0.5 * np.log(np.real(h))


def _figarch(y, *, p=1, q=1, mean=True, ar=(), dist="normal", truncation=1000):
    from openecon.econometrics.arch.extended import likelihood
    y = torch.tensor(y, dtype=torch.float64)
    layout = Layout(model="figarch", dist=dist, arch_lags=(1,) if p else (),
                    garch_lags=(1,) if q else (), ar_lags=ar, k_x=int(mean),
                    truncation=truncation)
    return layout, likelihood(y, torch.ones((len(y), layout.k_x), dtype=torch.float64),
                              torch.empty((len(y), 0), dtype=torch.float64), layout)


@pytest.mark.parametrize("row", ARCH_FIGARCH["figarch_likelihood"],
                         ids=lambda row: f"p{row['p']}q{row['q']}")
def test_figarch_likelihood_matches_the_arch_package_at_common_parameters(row):
    layout, like = _figarch(row["y"], p=row["p"], q=row["q"], mean=row["mu"] != 0)
    ix = layout.index
    values = dict(zip(row["arch_parameter_names"], row["params"], strict=True))
    theta = torch.zeros(ix.k, dtype=torch.float64)
    if ix.x:
        theta[ix.x[0]] = row["mu"]
    theta[ix.c], theta[ix.r[0]] = values["omega"], values["d"]
    if ix.a:
        theta[ix.a[0]] = values["phi"]
    if ix.b:
        theta[ix.b[0]] = values["beta"]
    out = like.evaluate(theta, derivatives=False)
    assert out.presample_variance == pytest.approx(row["h0"], rel=1e-14)
    assert_allclose(out.variance[:5].numpy(), row["sigma2_head"], rtol=1e-13)
    assert float(out.variance[-1]) == pytest.approx(row["sigma2_last"], rel=1e-12)
    assert_allclose(out.loglik.numpy(), row["loglik"], rtol=0, atol=1e-12)
    assert out.value == pytest.approx(sum(row["loglik"]), rel=1e-13)


def test_figarch_weights_and_their_derivatives():
    from openecon.econometrics.arch.extended import figarch_weights
    for phi, d, beta in [(0.2, 0.45, 0.5), (0.0, 0.3, 0.6), (0.15, 0.7, 0.0)]:
        lam, dlam = figarch_weights(phi, d, beta, 400, derivatives=True)
        assert_allclose(lam.numpy(), np.real(bbm_weights(phi, d, beta, 400)), rtol=1e-12,
                        atol=1e-16)
        step = 1e-20
        for k in range(3):
            shifted = [complex(v) for v in (phi, d, beta)]
            shifted[k] += 1j * step
            exact = np.imag(bbm_weights(*shifted, 400)) / step
            assert_allclose(dlam[k].numpy(), exact, rtol=1e-10, atol=1e-15)
        # Untruncated weights sum to one whenever d > 0 (lambda(1) = 1) and the hyperbolic
        # tail lambda_i ~ i^(-1-d) leaves 1 - sum_(i<=K) lambda_i ~ K^(-d).
        long, _ = figarch_weights(phi, d, beta, 40_000)
        partial = torch.cumsum(long, 0)
        ratio = (1 - float(partial[-1])) / (1 - float(partial[9_999]))
        assert ratio == pytest.approx(4.0 ** -d, rel=0.02)


@pytest.mark.parametrize("dist", ["normal", "sged"])
def test_figarch_scores_match_the_explicit_loop(dist):
    y, _, _ = series(260, 7)
    layout, like = _figarch(y, ar=(1,), dist=dist, truncation=120)
    ix = layout.index
    theta = torch.zeros(ix.k, dtype=torch.float64)
    theta[ix.x[0]], theta[ix.ar[0]] = 0.3, 0.25
    theta[ix.c], theta[ix.a[0]], theta[ix.r[0]], theta[ix.b[0]] = 0.08, 0.2, 0.45, 0.5
    if ix.d:
        theta[ix.d[0]], theta[ix.d[1]] = 0.3, 0.2

    def oracle(point):
        m = y - point[ix.x[0]]
        e = m - point[ix.ar[0]] * np.r_[0.0, m[:-1]]
        tau = tuple(point[ix.d]) if ix.d else None
        return np.real(figarch_loglik(point[ix.c], point[ix.a[0]], point[ix.r[0]],
                                      point[ix.b[0]], e, 120, dist, tau))

    out = like.evaluate(theta, scores=True)
    assert_allclose(out.loglik.numpy(), oracle(theta.numpy()), atol=1e-11)
    numerical = approx_fprime(theta.numpy(), oracle, centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=2e-5, atol=2e-6)

    def objective(point):
        evaluation = like.evaluate(point)
        if evaluation is None:
            return torch.tensor(math.nan), torch.full_like(point, math.nan)
        return evaluation.value, evaluation.gradient

    assert check_derivatives(objective, theta)["gradient_max_rel_error"] < 1e-6


@pytest.mark.parametrize("number", [0, 1])
def test_figarch_maximum_likelihood_matches_the_arch_package(number):
    # Zero mean: the presample value mean(y^2) is fixed and arch's likelihood with
    # backcast=mean(y^2) is exactly the one maximized here; arch's "classic" standard
    # errors come from its own numerical Hessian.
    row = ARCH_FIGARCH["figarch_fits"][number]
    fit = oe.arch(data=pd.DataFrame({"y": row["y"]}), y="y", model="figarch", constant=False,
                  covariance="nonrobust")
    assert fit.metrics["log_likelihood"] >= row["loglik"] - 1e-9
    values, errors = _natural(fit, ["ARCH:Intercept", "ARCH:L1.phi", "ARCH:d", "ARCH:L1.beta"],
                              {})
    assert_allclose(values, row["params"], rtol=2e-5)
    assert_allclose(errors, row["std_errors"], rtol=1e-4)
    est = {c.term: c.estimate for c in fit.coefficients}
    weights = np.real(bbm_weights(est["ARCH:L1.phi"], est["ARCH:d"], est["ARCH:L1.beta"], 1000))
    assert fit.metrics["persistence"] == pytest.approx(weights.sum(), rel=1e-12)
    assert fit.metrics["unconditional_variance"] is None
    assert fit.extra["model"]["truncation"] == 1000


def test_figarch_forecasts_state_and_round_trip():
    row = ARCH_FIGARCH["figarch_fits"][1]
    y = np.array(row["y"]) + 0.2
    data = pd.DataFrame({"t": np.arange(1, len(y) + 1), "y": y})
    fit = oe.arch(data=data, y="y", time="t", model="figarch", truncation=300)
    est = {c.term: c.estimate for c in fit.coefficients}
    e = y - est["Intercept"]
    lam = np.real(bbm_weights(est["ARCH:L1.phi"], est["ARCH:d"], est["ARCH:L1.beta"], 300))
    level = est["ARCH:Intercept"] / (1 - est["ARCH:L1.beta"])
    history = list(e ** 2)
    expected = []
    for _ in range(6):
        value = level + sum(lam[i] * history[-1 - i] for i in range(300))
        expected.append(value)
        history.append(value)
    table = oe.forecast(fit, 6)
    assert_allclose(table["variance_forecast"].to_numpy(), expected, rtol=1e-11)
    assert len(fit.extra["state"]["squared_residuals"]) == 300
    restored = ResultBundle.model_validate_json(fit.model_dump_json())
    assert_allclose(oe.forecast(restored, 6)["variance_forecast"].to_numpy(), expected,
                    rtol=1e-11)
    assert_allclose(oe.forecast(fit, 6, data=data)["variance_forecast"].to_numpy(), expected,
                    rtol=1e-11)
    assert "FIGARCH(1,d,1)" in fit.summary() and "ARCH:d" in fit.to_latex()
    # A short sample (n < truncation) primes the missing lags with h_0.
    short = oe.arch(data=data.iloc[:200], y="y", time="t", model="figarch", truncation=300)
    stored = short.extra["state"]["squared_residuals"]
    assert len(stored) == 200
    est = {c.term: c.estimate for c in short.coefficients}
    lam = np.real(bbm_weights(est["ARCH:L1.phi"], est["ARCH:d"], est["ARCH:L1.beta"], 300))
    history = [short.extra["state"]["presample_variance"]] * 100 + stored
    value = est["ARCH:Intercept"] / (1 - est["ARCH:L1.beta"]) \
        + sum(lam[i] * history[-1 - i] for i in range(300))
    assert oe.forecast(short, 1)["variance_forecast"].iloc[0] == pytest.approx(value, rel=1e-11)
    broken = ResultBundle.model_validate_json(fit.model_dump_json())
    broken.extra["state"]["squared_residuals"] = []
    with pytest.raises(AnalysisError) as error:
        oe.forecast(broken, 2)
    assert error.value.code == "invalid_result"


def test_figarch_refusals_and_options():
    data = pd.DataFrame({"y": ARCH_FIGARCH["figarch_fits"][1]["y"][:500],
                         "z": np.linspace(0, 1, 500)})
    for keywords in ({"arch": 2}, {"garch": [1, 2]}, {"archm": "sd"}, {"variance_x": ["z"]}):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=data, y="y", model="figarch", **keywords)
        assert error.value.code == "unsupported_combination", keywords
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=data, y="y", model="garch", truncation=100)
    assert error.value.code == "invalid_option"
    for bad in (0, 6000):
        with pytest.raises(AnalysisError) as error:
            oe.arch(data=data, y="y", model="figarch", truncation=bad)
        assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.arch(data=Dataset.from_frame(data), y="y", model="figarch")
    assert error.value.code == "streaming_unsupported"
    plain = oe.arch(data=data, y="y", model="figarch", arch=0)
    assert [c.term for c in plain.coefficients] == ["Intercept", "ARCH:L1.beta", "ARCH:d",
                                                    "ARCH:Intercept"]
    assert plain.extra["model"]["label"] == "FIGARCH(0,d,1)"
