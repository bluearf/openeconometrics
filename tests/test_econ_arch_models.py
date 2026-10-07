"""Every variance model, mean structure and distribution of oe.arch against the oracle.

For each specification the fitted model must (i) report the log likelihood of the
independent explicit-recursion oracle at its estimates, (ii) be a strict local maximum
of that oracle (numerical gradient zero in standard-error units, numerical Hessian
negative definite) and (iii) report the inverse of the oracle's numerical observed
information as its ``nonrobust`` (OIM) covariance. IGARCH and power ARCH are additionally compared
with a brute-force SciPy maximization.

EGARCH: the likelihood has a kink wherever a standardized residual is zero, so the
numerical derivatives are taken on the smooth piece through the estimates (the oracle
with the signs of its own residuals held fixed), which is the likelihood itself in a
neighbourhood of the estimates unless the maximum lies on a kink (``test_econ_arch_kinks``).
"""

import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from test_econ_arch_oracle import oracle_of, simulate

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.arch import densities, postfit
from openecon.econometrics.arch.layout import Layout

# name -> (simulation keywords, oe.arch keywords[, seed]); the default seed is derived
# from the name.
SPECIFICATIONS = {
    "arch2": (dict(), dict(model="arch", arch=2)),
    "garch21": (dict(), dict(arch=2, garch=1)),
    "garch_ged": (dict(dist="ged"), dict(dist="ged")),
    "gjr_t": (dict(model="gjr", dist="t"), dict(model="gjr", dist="t")),
    "parch": (dict(model="parch"), dict(model="parch")),
    "arma11": (dict(ar=0.5, ma=0.3), dict(ar=1, ma=1)),
    "arma_sparse": (dict(ar=0.4), dict(ar=[1, 3], ma=[2])),
    "archm_variance": (dict(archm="variance", psi=0.3), dict(archm="variance")),
    "archm_sd": (dict(archm="sd", psi=0.4), dict(archm="sd")),
    "archm_log": (dict(archm="log", psi=0.3), dict(archm="log")),
    "archm_arma": (dict(archm="sd", psi=0.4, ar=0.5, ma=0.3), dict(archm="sd", ar=1, ma=1)),
    "het": (dict(het=0.5), dict(variance_x=["z"])),
    "egarch_het_t": (dict(model="egarch", dist="t", het=0.3),
                     dict(model="egarch", dist="t", variance_x=["z"])),
    "egarch_arma": (dict(model="egarch", ar=0.5), dict(model="egarch", ar=1)),
    "egarch_m": (dict(model="egarch", archm="sd", psi=0.4), dict(model="egarch", archm="sd")),
    "gjr_m_arma": (dict(model="gjr", archm="variance", psi=0.2, ma=0.3),
                   dict(model="gjr", archm="variance", ma=1)),
    "parch_m_t": (dict(model="parch", dist="t", archm="sd", psi=0.3),
                  dict(model="parch", dist="t", archm="sd"), 52),
}


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


@pytest.mark.parametrize("name", sorted(SPECIFICATIONS))
def test_fit_is_a_maximum_of_the_independent_likelihood(name):
    simulation, keywords, *seed = SPECIFICATIONS[name]
    frame = simulate(600, seed[0] if seed else sum(map(ord, name)), **simulation)
    result = oe.arch(data=frame, y="y", x=["x"], time="t", covariance="nonrobust", **keywords)
    theta, loglike = oracle_of(result, frame)
    total = loglike(theta).sum()
    assert result.metrics["log_likelihood"] == pytest.approx(total, rel=1e-10)
    signs = None
    if keywords.get("model") == "egarch":
        record = result.provenance["optimizer"]
        signs = np.where(loglike.path(theta)[0] < 0, -1.0, 1.0)
        signs[record.get("kink_observations", [])] = record.get("kink_weights", [])
        signs = signs.tolist()
        assert "smooth piece" in record["hessian"]
    standard_errors = np.array([c.std_error for c in result.coefficients])
    gradient = approx_fprime(theta, lambda point: loglike(point, signs).sum(), centered=True)
    assert np.abs(gradient * standard_errors).max() < 2e-4
    hessian = approx_hess(theta, lambda point: loglike(point, signs).sum())
    assert np.linalg.eigvalsh(hessian).max() < 0.0
    expected = np.linalg.inv(-hessian)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert_allclose(np.array(result.covariance_matrix) / scale, expected / scale, atol=5e-3)
    assert result.provenance["optimizer"]["engine"] == (
        "scan" if keywords.get("model") == "egarch" or "archm" in keywords else "filter")
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_term_names_follow_stata():
    frame = simulate(600, 4, model="egarch", dist="t", ar=0.4, archm="sd", psi=0.3, het=0.3)
    result = oe.arch(data=frame, y="y", x=["x"], model="egarch", dist="t", ar=1, ma=[2],
                     archm="sd", variance_x=["z"])
    assert [(c.equation, c.term) for c in result.coefficients] == [
        ("y", "Intercept"), ("y", "x"), ("ARCHM", "ARCHM:sigma"), ("ARMA", "ARMA:L1.ar"),
        ("ARMA", "ARMA:L2.ma"), ("HET", "HET:z"), ("HET", "HET:Intercept"),
        ("ARCH", "ARCH:L1.earch"), ("ARCH", "ARCH:L1.earch_a"), ("ARCH", "ARCH:L1.egarch"),
        (None, "/lndfm2")]
    assert result.title == ("EGARCH(1,1)-in-mean with ARMA disturbances regression, "
                            "Student t errors")
    assert result.tests["model"]["df"] == 4          # x, ARCH-in-mean, AR and MA terms
    power = oe.arch(data=simulate(600, 12, model="parch"), y="y", model="parch",
                    archm="variance")
    assert [c.term for c in power.coefficients] == [
        "Intercept", "ARCHM:sigma2", "ARCH:L1.parch", "ARCH:L1.pgarch", "ARCH:Intercept",
        "POWER:power"]
    assert power.coefficients[-1].equation == "POWER"
    log_form = oe.arch(data=simulate(600, 9, archm="log", psi=0.3), y="y", archm="log")
    assert log_form.coefficients[1].term == "ARCHM:lnsigma2"


def test_scan_engine_covariances_match_numerical_scores():
    frame = simulate(500, 21, model="egarch", archm="sd", psi=0.4)
    fits = {kind: oe.arch(data=frame, y="y", x=["x"], model="egarch", archm="sd",
                          covariance=kind) for kind in ("opg", "robust")}
    theta, loglike = oracle_of(fits["opg"], frame)
    assert not fits["opg"].provenance["optimizer"].get("kink_observations")
    n = len(frame)
    signs = np.where(loglike.path(theta)[0] < 0, -1.0, 1.0).tolist()
    scores = approx_fprime(theta, lambda point: loglike(point, signs), centered=True)
    bread = np.linalg.inv(-approx_hess(theta, lambda point: loglike(point, signs).sum()))
    expected = {"opg": np.linalg.inv(scores.T @ scores),
                "robust": n / (n - 1) * bread @ (scores.T @ scores) @ bread}
    for kind, result in fits.items():
        scale = np.sqrt(np.outer(np.diag(expected[kind]), np.diag(expected[kind])))
        assert_allclose(np.array(result.covariance_matrix) / scale, expected[kind] / scale,
                        atol=5e-3)


# ---- IGARCH -------------------------------------------------------------------------------------


def test_igarch_imposes_unit_persistence():
    frame = simulate(700, 31)
    result = oe.arch(data=frame, y="y", x=["x"], model="igarch", arch=1, garch=2,
                     covariance="nonrobust")
    by_term = {c.term: c for c in result.coefficients}
    total = sum(by_term[name].estimate for name in ("ARCH:L1.arch", "ARCH:L1.garch",
                                                    "ARCH:L2.garch"))
    assert total == pytest.approx(1.0, abs=1e-12)
    assert result.metrics["persistence"] == 1.0
    assert result.metrics["unconditional_variance"] is None
    assert not any("stationary" in warning for warning in result.warnings)
    assert "restriction" in result.inference
    theta, loglike = oracle_of(result, frame)
    assert result.metrics["log_likelihood"] == pytest.approx(loglike(theta).sum(), rel=1e-11)
    # The information criteria count the free parameters only.
    k_free = len(theta) - 1
    assert result.metrics["aic"] == pytest.approx(
        -2 * result.metrics["log_likelihood"] + 2 * k_free, rel=1e-12)

    # Brute force over the free parameters (the last GARCH coefficient is implied).
    def full(free):
        return np.r_[free[:4], 1.0 - free[2] - free[3], free[4]]

    def objective(free):
        value = loglike(full(free))
        return 1e10 if value is None else -value.sum()

    free_hat = np.r_[theta[:4], theta[5]]
    start = np.array([0.3, 0.5, 0.1, 0.5, 0.05])
    first = minimize(objective, start, method="BFGS", options={"gtol": 1e-6})
    best = minimize(objective, first.x, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 5000, "maxfev": 5000})
    assert result.metrics["log_likelihood"] >= -best.fun - 1e-7
    assert_allclose(free_hat, best.x, rtol=2e-3, atol=2e-4)
    # Covariance: delta method from the free parameters; the matrix is singular.
    hessian = approx_hess(free_hat, lambda free: -objective(free))
    free_covariance = np.linalg.inv(-hessian)
    jacobian = np.zeros((6, 5))
    jacobian[[0, 1, 2, 3, 5], [0, 1, 2, 3, 4]] = 1.0
    jacobian[4, [2, 3]] = -1.0
    expected = jacobian @ free_covariance @ jacobian.T
    covariance = np.array(result.covariance_matrix)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert_allclose(covariance / scale, expected / scale, atol=5e-3)
    assert np.linalg.matrix_rank(covariance / scale, tol=1e-8) == 5
    # Likelihood-ratio ordering: the unrestricted GARCH fits at least as well.
    unrestricted = oe.arch(data=frame, y="y", x=["x"], arch=1, garch=2)
    assert unrestricted.metrics["log_likelihood"] >= result.metrics["log_likelihood"] - 1e-8


def test_igarch11_standard_errors_of_arch_and_garch_coincide():
    result = oe.arch(data=simulate(700, 32), y="y", x=["x"], model="igarch")
    by_term = {c.term: c for c in result.coefficients}
    arch_term, garch_term = by_term["ARCH:L1.arch"], by_term["ARCH:L1.garch"]
    assert arch_term.estimate + garch_term.estimate == pytest.approx(1.0, abs=1e-12)
    assert arch_term.std_error == pytest.approx(garch_term.std_error, rel=1e-10)


# ---- power ARCH ---------------------------------------------------------------------------------


def test_parch_matches_brute_force_and_nests_garch():
    frame = simulate(700, 45, model="parch")
    result = oe.arch(data=frame, y="y", x=["x"], model="parch")
    theta, loglike = oracle_of(result, frame)

    def objective(point):
        value = loglike(point)
        return 1e10 if value is None or not np.isfinite(value).all() else -value.sum()

    first = minimize(objective, np.array([0.3, 0.5, 0.12, 0.8, 0.1, 1.3]), method="BFGS",
                     options={"gtol": 1e-6})
    best = minimize(objective, first.x, method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 6000, "maxfev": 6000})
    assert result.metrics["log_likelihood"] >= -best.fun - 1e-7
    assert result.metrics["log_likelihood"] == pytest.approx(-best.fun, abs=1e-5)
    assert_allclose(theta, best.x, rtol=5e-3, atol=5e-4)
    # At power 2 the power-ARCH likelihood is the GARCH likelihood.
    garch = oe.arch(data=frame, y="y", x=["x"])
    point = np.r_[estimates(garch), 2.0]
    assert loglike(point).sum() == pytest.approx(garch.metrics["log_likelihood"], rel=1e-11)
    assert result.metrics["log_likelihood"] >= garch.metrics["log_likelihood"] - 1e-8
    # Persistence uses E|z|^power of the fitted distribution.
    power = theta[-1]
    moment = 2 ** (power / 2) * math.gamma((power + 1) / 2) / math.sqrt(math.pi)
    assert result.metrics["persistence"] == pytest.approx(theta[2] * moment + theta[3], rel=1e-10)
    assert result.metrics["unconditional_variance"] is None


# ---- persistence and warnings -------------------------------------------------------------------


def test_persistence_definitions():
    theta = torch.tensor([0.1, 0.05, 0.2, 0.6, 0.3], dtype=torch.float64)
    assert postfit.persistence(Layout("garch", "normal", (1, 2), (1,)), theta[[0, 1, 3, 4]]) \
        == pytest.approx(0.75)
    gjr = Layout("gjr", "normal", (1,), (1,))
    values = torch.tensor([0.1, 0.2, 0.6, 0.3], dtype=torch.float64)       # a, g, b, omega
    assert postfit.persistence(gjr, values) == pytest.approx(0.1 + 0.1 + 0.6)
    assert postfit.unconditional_variance(gjr, values, 0.8) == pytest.approx(0.3 / 0.2)
    assert postfit.unconditional_variance(gjr, values, 1.0) is None
    egarch = Layout("egarch", "normal", (1,), (1, 2))
    values = torch.tensor([-0.1, 0.2, 0.5, 0.3, 0.0], dtype=torch.float64)
    assert postfit.persistence(egarch, values) == pytest.approx(0.8)
    assert postfit.unconditional_variance(egarch, values, 0.8) is None
    parch_t = Layout("parch", "t", (1,), (1,))
    values = torch.tensor([0.1, 0.8, 0.2, 1.5, math.log(4.0)], dtype=torch.float64)
    moment = densities.abs_moment("t", 1.5, math.log(4.0))
    assert postfit.persistence(parch_t, values) == pytest.approx(0.1 * moment + 0.8)
    heavy = torch.tensor([0.1, 0.8, 0.2, 3.5, math.log(1.0)], dtype=torch.float64)
    assert postfit.persistence(parch_t, heavy) is None       # E|z|^3.5 does not exist at 3 df
    igarch = Layout("igarch", "normal", (1,), (1,))
    assert postfit.persistence(igarch, torch.tensor([0.2, 0.8, 0.1], dtype=torch.float64)) == 1.0


def test_nonstationary_variance_is_reported_with_a_warning():
    # An integrated variance process: the unrestricted estimate lands above one.
    rng = np.random.default_rng(5)
    n = 400
    e, h = np.zeros(n), np.ones(n)
    for t in range(1, n):
        h[t] = 0.02 + 0.25 * e[t - 1] ** 2 + 0.8 * h[t - 1]
        e[t] = math.sqrt(h[t]) * rng.normal()
    result = oe.arch(data={"y": e}, y="y", constant=False)
    assert result.metrics["persistence"] >= 1.0
    assert result.metrics["unconditional_variance"] is None
    assert any("not covariance stationary" in warning for warning in result.warnings)


# ---- likelihoods without a regular maximum ------------------------------------------------------


def failure(**keywords):
    with pytest.raises(AnalysisError) as error:
        oe.arch(**keywords)
    assert error.value.code == "nonconvergence"
    return str(error.value)


def test_unbounded_likelihood_with_a_negative_arch_coefficient_is_reported():
    # ARCH(2) on ARCH(1) data: the profile likelihood rises as the second coefficient
    # falls, until the variance of one observation reaches zero (a spike, no maximum).
    frame = simulate(600, sum(map(ord, "arch2")), model="arch")
    message = failure(data=frame, y="y", x=["x"], model="arch", arch=2)
    assert "unbounded" in message and "approaches zero" in message
    assert "fewer ARCH/GARCH lags" in message
    # The supported model on the same data is fine.
    result = oe.arch(data=frame, y="y", x=["x"], model="arch", arch=1)
    assert result.coefficients[2].estimate > 0.0


def test_power_arch_cusp_is_reported():
    # The search drifts to a power below one, where the likelihood has a cusp at every
    # zero residual.
    message = failure(data=simulate(700, 41, model="parch"), y="y", x=["x"], model="parch")
    assert "power" in message and "cusp" in message and "model='garch'" in message


def test_search_outside_the_positive_coefficients_is_reported():
    # IGARCH forced on a sample with (almost) no ARCH effect at the restricted optimum.
    message = failure(data=simulate(600, 119), y="y", x=["x"], model="igarch")
    assert "non-negative variance coefficients" in message
