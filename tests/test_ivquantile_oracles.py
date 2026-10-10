"""Independent SciPy LP, NumPy matrix and seeded weak-ID law checks."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.stats import chi2, norm
import torch

from openecon.econometrics.ivquantile import ivqreg, ivqreg_predict
from openecon.econometrics.ivquantile.common import controls, data_state, geometry
from openecon.econometrics.ivquantile.kernels import accepted_runs, profile_record


@pytest.fixture(scope="module", autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def data(seed=627, n=241, strength=1.5, q=1):
    generator = np.random.default_rng(seed)
    x, z, u, e, z2 = generator.normal(size=(5, n))
    d = strength * z + 0.6 * x + u + (0.4 * z2 if q == 2 else 0)
    y = 0.7 + 1.1 * d - 0.4 * x + 0.5 * u + e
    return pd.DataFrame(dict(y=y, d=d, x=x, z=z, z2=z2))


def linear_program(w, target, tau):
    n, k = w.shape
    objective = np.r_[np.zeros(k), np.full(n, tau), np.full(n, 1 - tau)]
    equality = np.c_[w, np.eye(n), -np.eye(n)]
    fit = linprog(objective, A_eq=equality, b_eq=target,
                  bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
    assert fit.success
    return fit.x[:k], fit.fun, fit.eqlin.marginals


@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("q", [1, 2])
def test_all_profile_coefficients_covariance_and_weak_id_match_independent_lp(tau, q):
    frame = data(n=121, q=q)
    instruments = ["z"] if q == 1 else ["z", "z2"]
    fit = ivqreg(data=frame, y="y", endogenous="d", x=["x"], instruments=instruments,
                grid=[0.2, 0.8, 1.1, 1.5, 2.0], quantile=tau, bandwidth=0.6)
    w = np.column_stack([np.ones(len(frame)), frame["x"], *[frame[v] for v in instruments]])
    x = w[:, :2]
    z = w[:, 2:]
    zr = z - x @ np.linalg.solve(x.T @ x, x.T @ z)
    a = zr.T @ zr / len(frame)
    for record in fit.attrs["state"]["evaluations"]:
        target = frame["y"].to_numpy() - frame["d"].to_numpy() * record["alpha"]
        beta, objective, dual = linear_program(w, target, tau)
        np.testing.assert_allclose(beta, record["coefficients"], rtol=1e-7, atol=1e-8)
        assert objective == pytest.approx(record["qr_objective"], rel=1e-9)
        residual = target - w @ beta
        bread = w.T @ (w * ((np.abs(residual) <= 0.6) / 1.2)[:, None])
        inverse = np.linalg.inv(bread)
        covariance = tau * (1 - tau) * inverse @ (w.T @ w) @ inverse.T
        gamma = beta[2:]
        stat = gamma @ np.linalg.solve(covariance[2:, 2:], gamma)
        np.testing.assert_allclose(record["bread"], bread, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(record["covariance"], covariance, rtol=1e-8, atol=1e-9)
        assert record["criterion"] == pytest.approx(gamma @ a @ gamma, rel=1e-8, abs=1e-12)
        assert record["statistic"] == pytest.approx(stat, rel=1e-8)
        assert record["p_value"] == pytest.approx(chi2.sf(stat, q), rel=1e-8)
        np.testing.assert_allclose(w.T @ dual, np.zeros(w.shape[1]), atol=1e-7)


@pytest.mark.parametrize("q", [1, 2])
def test_identified_joint_influence_and_full_query_covariance_match_numpy(q):
    frame = data(q=q)
    fit = ivqreg(data=frame, y="y", endogenous="d", x=["x"],
                instruments=["z"] if q == 1 else ["z", "z2"],
                grid=np.linspace(-0.5, 2.5, 31).tolist(), inference="identified", bandwidth=0.6)
    state = fit.attrs["state"]
    selected = state["evaluations"][state["selected_evaluation"]]
    w = np.c_[np.ones(len(frame)), frame["x"], frame["z"]] if q == 1 \
        else np.c_[np.ones(len(frame)), frame["x"], frame["z"], frame["z2"]]
    x, z = w[:, :2], w[:, 2:]
    target = frame["y"].to_numpy() - frame["d"].to_numpy() * selected["alpha"]
    qr_beta, _, _ = linear_program(w, target, 0.5)
    residual = target - w @ qr_beta
    weights = (np.abs(residual) <= 0.6) / 1.2
    bread = w.T @ (weights[:, None] * w)
    derivative = -np.linalg.solve(bread, w.T @ (weights * frame["d"].to_numpy()))
    zr = z - x @ np.linalg.solve(x.T @ x, x.T @ z)
    a = zr.T @ zr / len(frame)
    t = derivative[2:]
    row = -(t @ a) / (t @ a @ t)
    influence = np.zeros((3, w.shape[1]))
    influence[0, 2:] = row
    influence[1:, :2] = np.eye(2)
    influence[1:] += derivative[:2, None] * influence[0, None]
    inverse = np.linalg.inv(bread)
    nuisance_c = 0.25 * inverse @ (w.T @ w) @ inverse.T
    covariance = influence @ nuisance_c @ influence.T
    np.testing.assert_allclose(state["joint"]["profile_derivative"], derivative, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(state["joint"]["influence_map"], influence, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(state["joint"]["covariance"], covariance, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(fit["coefficients"]["std_error"], np.sqrt(np.diag(covariance)), rtol=1e-8)
    params = np.r_[selected["alpha"], qr_beta[:2]]
    np.testing.assert_allclose(fit["coefficients"]["p_value"], 2 * norm.sf(abs(params / np.sqrt(np.diag(covariance)))), rtol=1e-7)
    query = frame.iloc[:8]
    result = ivqreg_predict(result=fit, data=query)
    gradient = np.c_[query["d"], np.ones(len(query)), query["x"]]
    np.testing.assert_allclose(result.attrs["target_covariance"], gradient @ covariance @ gradient.T, rtol=1e-8)
    if q == 1:
        # Just-identified CH J^-1 S J^-T is the same complete structural covariance.
        structural = np.c_[frame["d"], x]
        j = w.T @ (weights[:, None] * structural)
        direct = np.linalg.inv(j) @ (0.25 * w.T @ w) @ np.linalg.inv(j).T
        np.testing.assert_allclose(covariance, direct, rtol=1e-8)


def test_grid_topology_does_not_convexify_or_certify_untested_points():
    records = [dict(alpha=float(i), p_value=p) for i, p in enumerate([0.8, 0.01, 0.8, 0.9, 0.01, 0.8])]
    accepted, runs = accepted_runs(records, 0.95)
    assert accepted == [True, False, True, True, False, True]
    assert len(runs) == 3
    assert runs[0]["lower_grid_boundary"]
    assert runs[-1]["upper_grid_boundary"]
    assert all(r["between_points"] == "untested" and r["outer_tails"] == "unknown" for r in runs)
    assert accepted_runs([dict(alpha=0, p_value=0.01)], 0.95) == ([False], [])


@pytest.mark.parametrize("strength", [0.0, 1.5])
def test_null_size_with_weak_and_strong_instruments(strength):
    # Fixed protocol, MC acceptance includes sampling noise; not a finite-size guarantee.
    probabilities = []
    for seed in range(120):
        frame = data(seed=7000 + seed, n=321, strength=strength)
        config = controls(grid=[0.5, 1.1, 2.0], quantile=0.5, confidence=0.95, bandwidth=0.65,
                          inference="weak", intercept=True, missing="raise", device="cpu",
                          weights=None, cluster=None, max_work=10**12, max_evaluations=3,
                          refinement_tolerance=1e-6)
        _, spec, _ = data_state(frame, "y", ["x"], "d", ["z"], config)
        y, d, x, _, w, a, _ = geometry(frame, spec, config)
        record = profile_record(y, d, w, a, x.shape[1], 1.1, config)
        probabilities.append(record["p_value"])
    rejections = sum(p < 0.05 for p in probabilities)
    assert 0 <= rejections <= 14, (strength, rejections)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
