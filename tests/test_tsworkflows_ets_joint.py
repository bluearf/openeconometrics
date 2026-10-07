"""Full free-smoothing ETS information, independently differentiated in reporting units."""

import numpy as np
import pytest
from scipy.optimize import minimize
import openecon as oe
from test_tsworkflows_oracle import hessian


def scalar_path(z, initial, observations, model, period):
    trend = model not in {"ANN", "ANA"}
    seasonal = model.endswith("A")
    damped = "d" in model
    alpha = z[0]
    idx = 1
    beta = z[idx] if trend else 0.0
    idx += int(trend)
    gamma = z[idx] if seasonal else 0.0
    idx += int(seasonal)
    phi = z[idx] if damped else 1.0
    sigma = z[-1]
    level, b = initial[0], initial[1] if trend else 0.0
    seasons = list(initial[-period:]) if seasonal else [0.0] * period
    errors = []
    for t, obs in enumerate(observations):
        mean = level + phi * b + seasons[t % period]
        error = obs - mean
        errors.append(error)
        level = level + phi * b + alpha * error
        b = phi * b + beta * error if trend else 0.0
        if seasonal:
            seasons[t % period] += gamma * error
    return np.asarray(errors), sigma


@pytest.mark.parametrize("model", ["ANN", "AAN", "AdN", "ANA", "AAA", "AdA"])
def test_full_joint_ml_parameters_and_information(model):
    trend = model not in {"ANN", "ANA"}
    seasonal = model.endswith("A")
    damped = "d" in model
    initial = [3.0] + ([0.1] if trend else []) + ([-0.3, 0.1, 0.4, -0.2] if seasonal else [])
    level, b = initial[0], 0.1 if trend else 0
    season = list(initial[-4:]) if seasonal else [0.0] * 4
    alpha, beta, gamma, phi = 0.35, 0.08, 0.12, 0.92 if damped else 1.0
    y = []
    for t, error in enumerate(np.random.default_rng(44).normal(0, 0.4, 200)):
        y.append(level + phi * b + season[t % 4] + error)
        level = level + phi * b + alpha * error
        b = phi * b + beta * error if trend else 0.0
        if seasonal:
            season[t % 4] += gamma * error
    result = oe.ets(data={"y": y}, y="y", model=model, period=4, initial=initial)
    actual = np.array([c.estimate for c in result.coefficients])

    def objective(z):
        errors, sigma = scalar_path(z, initial, y, model, 4)
        return (
            0.5 * len(y) * np.log(2 * np.pi)
            + len(y) * np.log(sigma)
            + 0.5 * np.sum(errors**2) / sigma**2
        )

    bounds = (
        [(0.001, 0.999)]
        + ([(0.001, 0.25)] if trend else [])
        + ([(0.001, 0.25)] if seasonal else [])
        + ([(0.3, 0.999)] if damped else [])
        + [(0.001, 5)]
    )
    start = (
        [alpha]
        + ([beta] if trend else [])
        + ([gamma] if seasonal else [])
        + ([phi] if damped else [])
        + [0.4]
    )
    oracle = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 5000},
    )
    assert oracle.success
    np.testing.assert_allclose(actual, oracle.x, atol=2e-5)
    np.testing.assert_allclose(
        result.covariance_matrix,
        np.linalg.inv(hessian(objective, actual, h=1e-4)),
        rtol=4e-4,
        atol=3e-6,
    )
    assert result.provenance["optimizer"]["converged"]
