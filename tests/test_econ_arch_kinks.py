"""EGARCH: the kinks of the likelihood, smooth pieces and maxima that lie on a kink.

The EGARCH log likelihood is not differentiable where a standardized residual is zero.
``ArchLikelihood.evaluate(signs=...)`` evaluates the smooth piece selected by a sign
pattern; ``kinks.polish`` finds a maximum on a kink. Both are checked against the
independent explicit-recursion oracle of ``test_econ_arch_oracle``.
"""

import math

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_fprime, approx_hess
from test_econ_arch_oracle import build, oracle_loglike, oracle_of, simulate, terms_of, unpack

import openecon as oe
from openecon.econometrics.arch import kinks
from openecon.engines.optimize import check_derivatives

EGARCH_CASES = ["egarch11", "egarch_all", "egarch_m_log_ged"]


def piece_oracle(layout, data, signs):
    y, x, z = data
    terms = terms_of(layout, layout.k_x, layout.k_z)

    def loglike(point):
        return oracle_loglike(unpack(terms, point, layout.model), y, x, z, model=layout.model,
                              dist=layout.dist, archm=layout.archm, signs=signs)

    return loglike


@pytest.mark.parametrize("name", EGARCH_CASES)
def test_the_piece_through_a_point_is_the_likelihood_itself(name):
    _, like, theta, _ = build(name)
    plain = like.evaluate(theta, scores=True)
    frozen = like.evaluate(theta, scores=True, signs=kinks.piece(plain.residual))
    assert frozen.value == pytest.approx(plain.value, rel=1e-14)
    assert_allclose(frozen.scores.numpy(), plain.scores.numpy(), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", EGARCH_CASES)
def test_smooth_pieces_match_the_oracle_and_have_exact_gradients(name):
    layout, like, theta, data = build(name)
    rng = np.random.default_rng(8)
    # An arbitrary pattern, including fractional weights as on an active kink.
    pattern = rng.choice([-1.0, 1.0], size=like.n)
    pattern[[5, 17]] = [0.3, -0.6]
    signs = torch.tensor(pattern)
    out = like.evaluate(theta, scores=True, signs=signs)
    reference = piece_oracle(layout, data, pattern.tolist())
    assert_allclose(out.loglik.numpy(), reference(theta.tolist()), rtol=1e-10, atol=1e-10)

    def objective(point):
        value = like.evaluate(point, signs=signs)
        return value.value, value.gradient

    assert check_derivatives(objective, theta)["gradient_max_rel_error"] < 1e-6
    numerical = approx_fprime(np.asarray(theta.tolist()), reference, centered=True)
    assert_allclose(out.scores.numpy(), numerical, rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize("name", EGARCH_CASES)
def test_derivative_of_the_standardized_residual(name):
    _, like, theta, _ = build(name)
    active = [3, 40, 111]
    out = like.evaluate(theta, active=active)

    def standardized(point):
        value = like.evaluate(torch.tensor(point), derivatives=False)
        return (value.residual / value.variance.sqrt())[active].numpy()

    numerical = approx_fprime(np.asarray(theta.tolist()), standardized, centered=True)
    assert_allclose(out.d_standardized.numpy(), numerical, rtol=1e-6, atol=1e-8)
    assert like.evaluate(theta).d_standardized is None


def test_piece_hessian_is_the_derivative_of_the_piece_gradient():
    layout, like, theta, data = build("egarch11")
    k = layout.k
    identity, offset = torch.eye(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    hessian = kinks.piece_hessian(like, offset, identity, theta)
    signs = kinks.piece(like.evaluate(theta).residual).tolist()
    reference = piece_oracle(layout, data, signs)
    expected = approx_hess(np.asarray(theta.tolist()), lambda point: reference(point).sum())
    assert_allclose(hessian.numpy(), expected, rtol=2e-4, atol=2e-4 * np.abs(expected).max())


# ---- maxima on a kink ---------------------------------------------------------------------------

# Simulated samples whose EGARCH maximum lies on a kink (found by a seed search; a
# smooth optimizer stalls on them with a non-zero one-sided gradient).
KINKED = {
    "ar": (dict(model="egarch", ar=0.5), dict(model="egarch", ar=1), 1130),
    "in_mean": (dict(model="egarch", archm="sd", psi=0.4), dict(model="egarch", archm="sd"), 822),
}


@pytest.mark.parametrize("name", sorted(KINKED))
def test_maximum_on_a_kink_satisfies_the_nonsmooth_first_order_conditions(name):
    simulation, keywords, seed = KINKED[name]
    frame = simulate(600, seed, **simulation)
    result = oe.arch(data=frame, y="y", x=["x"], time="t", covariance="nonrobust", **keywords)
    record = result.provenance["optimizer"]
    active, weights = record["kink_observations"], record["kink_weights"]
    assert len(active) >= 1 and "kink" in record["message"]
    assert record["method"].endswith("active_set_newton")
    theta, loglike = oracle_of(result, frame)
    total = loglike(theta).sum()
    assert result.metrics["log_likelihood"] == pytest.approx(total, rel=1e-10)
    # (i) the kink observations have a zero standardized residual; the weights are convex.
    e, h, _ = loglike.path(theta)
    assert np.abs(e[active] / np.sqrt(h[active])).max() < 1e-8
    assert all(abs(weight) <= 1.0 for weight in weights)
    # (ii) the convex combination of the one-sided gradients vanishes: the gradient of the
    # piece with the weights at the kinks is zero, and that piece is strictly concave.
    signs = np.where(e < 0, -1.0, 1.0)
    signs[active] = weights
    signs = signs.tolist()
    standard_errors = np.array([c.std_error for c in result.coefficients])
    gradient = approx_fprime(theta, lambda point: loglike(point, signs).sum(), centered=True)
    assert np.abs(gradient * standard_errors).max() < 2e-4
    hessian = approx_hess(theta, lambda point: loglike(point, signs).sum())
    assert np.linalg.eigvalsh(hessian).max() < 0.0
    # (iii) the OIM covariance is the inverse information of that piece.
    expected = np.linalg.inv(-hessian)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert_allclose(np.array(result.covariance_matrix) / scale, expected / scale, atol=5e-3)
    # (iv) the one-sided gradients of the likelihood itself do not vanish ...
    side = approx_fprime(theta, lambda point: loglike(point).sum(), centered=False,
                         epsilon=1e-7 * standard_errors)
    assert np.abs(side * standard_errors).max() > 1e-3
    # (v) ... yet no nearby point has a higher likelihood.
    rng = np.random.default_rng(1)
    for radius in (1e-1, 1e-2, 1e-3, 1e-4):
        for _ in range(40):
            point = theta + radius * standard_errors * rng.normal(size=len(theta))
            value = loglike(point)
            assert value is None or value.sum() <= total + 1e-9
    simplex = minimize(lambda point: -loglike(point).sum(), theta, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 3000,
                                "initial_simplex": theta + 1e-3 * np.vstack(
                                    [np.zeros(len(theta)), np.diag(standard_errors)])})
    assert -simplex.fun <= total + 1e-7
    assert type(result).model_validate_json(result.model_dump_json()) == result
    # OPG and robust covariances use the scores of the same piece.
    n = len(frame)
    scores = approx_fprime(theta, lambda point: loglike(point, signs), centered=True)
    robust = oe.arch(data=frame, y="y", x=["x"], time="t", covariance="robust", **keywords)
    sandwich = n / (n - 1) * expected @ (scores.T @ scores) @ expected
    scale = np.sqrt(np.outer(np.diag(sandwich), np.diag(sandwich)))
    assert_allclose(np.array(robust.covariance_matrix) / scale, sandwich / scale, atol=5e-3)


def test_egarch_converges_on_every_sample_of_a_seed_sweep():
    # Without the kink treatment the fits of seeds 109, 117, 121 and 124 do not converge.
    kinked = 0
    for seed in range(109, 125):
        frame = simulate(600, seed, model="egarch", ar=0.5)
        result = oe.arch(data=frame, y="y", x=["x"], model="egarch", ar=1)
        record = result.provenance["optimizer"]
        assert record["converged"] is True
        kinked += bool(record.get("kink_observations"))
        assert math.isfinite(result.metrics["log_likelihood"])
        assert all(c.std_error > 0 for c in result.coefficients)
    assert kinked >= 3
