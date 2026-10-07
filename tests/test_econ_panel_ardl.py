"""NumPy/SciPy development oracles independent of the Torch ECM code."""

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def fixture(groups=8, columns=2, n=130):
    rng = np.random.default_rng(322)
    pieces = []
    theta = np.array([1.4, -0.6])[:columns]
    for unit in range(groups):
        length = n + unit * 3
        x = rng.normal(size=(length, columns)).cumsum(0)
        y = np.zeros(length)
        for t in range(1, length):
            y[t] = (
                y[t - 1]
                - (0.24 + 0.015 * unit) * (y[t - 1] - x[t - 1] @ theta)
                + 0.3 * (x[t] - x[t - 1]).sum()
                + 0.1 * unit
                + rng.normal(scale=1 + 0.08 * unit)
            )
        pieces.append(
            pd.DataFrame(
                {
                    "y": y,
                    **{f"x{j}": x[:, j] for j in range(columns)},
                    "unit": unit,
                    "t": np.arange(length),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def blocks(df, p, q, trend, columns):
    result = []
    for _, g in df.groupby("unit", sort=True):
        y, x = g.y.to_numpy(), g[columns].to_numpy()
        start = max(p, q)
        dy, dx = np.diff(y), np.diff(x, axis=0)
        terms = [dy[start - lag - 1 : len(y) - lag - 1, None] for lag in range(1, p)]
        terms += [dx[start - lag - 1 : len(y) - lag - 1] for lag in range(q)]
        if trend != "n":
            terms.append(np.ones((len(y) - start, 1)))
        if trend == "ct":
            terms.append(np.arange(start + 1, len(y) + 1)[:, None])
        sr = np.column_stack(terms)
        z = np.column_stack([y[start - 1 : -1], x[start - 1 : -1], sr])
        result.append((dy[start - 1 :], z, x[start - 1 : -1], sr))
    return result


def map_ecm(a, k):
    return np.r_[-a[1 : k + 1] / a[0], a[0], a[k + 1 :]]


@pytest.mark.parametrize("p,q,trend", [(1, 1, "n"), (1, 1, "c"), (2, 2, "c"), (2, 3, "ct")])
def test_mg_and_dfe_independent_matrix_oracles(p, q, trend):
    df = fixture()
    bs = blocks(df, p, q, trend, ["x0", "x1"])
    estimates = np.array([map_ecm(np.linalg.lstsq(z, y, rcond=None)[0], 2) for y, z, _, _ in bs])
    mg = oe.mg(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", p=p, q=q, trend=trend)
    np.testing.assert_allclose([c.estimate for c in mg.coefficients], estimates.mean(0), atol=2e-9)
    np.testing.assert_allclose(
        mg.covariance_matrix, np.cov(estimates.T, ddof=1) / len(bs), rtol=1e-8, atol=2e-10
    )
    nd = {"n": 0, "c": 1, "ct": 2}[trend]
    common = bs[0][1].shape[1] - nd
    designs = []
    for i, (_, zi, _, _) in enumerate(bs):
        design = np.zeros((len(zi), common + nd * len(bs)))
        design[:, :common] = zi[:, :common]
        if nd:
            design[:, common + i * nd : common + (i + 1) * nd] = zi[:, -nd:]
        designs.append(design)
    z, y = np.vstack(designs), np.concatenate([b[0] for b in bs])
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    residual = y - z @ beta
    jac = np.eye(len(beta))
    jac[:2] = 0
    jac[0, 0], jac[1, 0] = beta[1] / beta[0] ** 2, beta[2] / beta[0] ** 2
    jac[0, 1], jac[1, 2] = -1 / beta[0], -1 / beta[0]
    jac[2] = 0
    jac[2, 0] = 1
    expected = jac @ np.linalg.inv(z.T @ z) @ jac.T * (residual @ residual / len(y))
    dfe = oe.dfe(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", p=p, q=q, trend=trend)
    np.testing.assert_allclose([c.estimate for c in dfe.coefficients], map_ecm(beta, 2), atol=2e-8)
    np.testing.assert_allclose(dfe.covariance_matrix, expected, rtol=2e-7, atol=2e-8)
    assert dfe.sample_positions == mg.sample_positions
    assert dfe.nobs == len(df) - len(bs) * max(p, q)


@pytest.mark.parametrize("p,q,trend", [(1, 1, "n"), (1, 1, "c"), (2, 2, "c"), (2, 3, "ct")])
def test_pmg_profile_likelihood_and_full_observed_covariance(p, q, trend):
    scipy = pytest.importorskip("scipy.optimize")
    df = fixture(groups=5, columns=2)
    bs = blocks(df, p, q, trend, ["x0", "x1"])
    initial = np.mean(
        [map_ecm(np.linalg.lstsq(z, y, rcond=None)[0], 2)[:2] for y, z, _, _ in bs], axis=0
    )

    def objective(theta):
        ll, gradient = 0.0, np.zeros(2)
        for y, z, x, sr in bs:
            w = np.column_stack([z[:, 0] - x @ theta, sr])
            a = np.linalg.lstsq(w, y, rcond=None)[0]
            u = y - w @ a
            sigma = u @ u / len(u)
            ll += -0.5 * len(u) * (np.log(2 * np.pi) + 1 + np.log(sigma))
            gradient += -a[0] * (x.T @ u) / sigma
        return -ll, -gradient

    optimum = scipy.minimize(objective, initial, jac=True, method="BFGS", options={"gtol": 1e-7})
    result = oe.pmg(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", p=p, q=q, trend=trend)
    np.testing.assert_allclose([c.estimate for c in result.coefficients[:2]], optimum.x, atol=3e-7)
    assert abs(result.metrics["log_likelihood"] + optimum.fun) < 1e-7
    full = np.array(result.extra["full_parameter_vector"])
    hessian = np.zeros((len(full), len(full)))
    offset = 2
    for y, z, x, sr in bs:
        width = sr.shape[1] + 1
        a, variance = full[offset : offset + width], np.exp(full[offset + width])
        ect = z[:, 0] - x @ full[:2]
        u = y - a[0] * ect - sr @ a[1:]
        j = np.zeros((len(y), len(full)))
        j[:, :2], j[:, offset], j[:, offset + 1 : offset + width] = -a[0] * x, ect, sr
        hessian += j.T @ j / variance
        # Cross second derivative of the ECM mean, d²mu/dtheta/dphi=-x.
        hessian[:2, offset] += x.T @ u / variance
        hessian[offset, :2] += x.T @ u / variance
        cross = j.T @ u / variance
        hessian[:, offset + width] += cross
        hessian[offset + width, :] += cross
        hessian[offset + width, offset + width] += 0.5 * (u @ u) / variance
        offset += width + 1
    np.testing.assert_allclose(
        result.extra["full_covariance"], np.linalg.inv(hessian), rtol=3e-7, atol=1e-8
    )
    assert result.provenance["optimizer"]["converged"] is True
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra
    assert "\\toprule" in result.to_latex()
    mg = oe.mg(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", p=p, q=q, trend=trend)
    assert oe.panel_ardl_hausman(mg, result) == result.tests["long_run_homogeneity"]


def test_homogeneity_covariance_and_sample_contract():
    df = fixture()
    mg = oe.mg(data=df, y="y", x=["x0", "x1"], panel="unit", time="t")
    pmg = oe.pmg(data=df, y="y", x=["x0", "x1"], panel="unit", time="t")
    contrast = oe.panel_ardl_hausman(mg, pmg)
    delta = np.array([c.estimate for c in mg.coefficients[:2]]) - np.array(
        [c.estimate for c in pmg.coefficients[:2]]
    )
    v = np.array(mg.covariance_matrix)[:2, :2] - np.array(pmg.covariance_matrix)[:2, :2]
    if np.linalg.eigvalsh(v).min() > 0:
        assert contrast["statistic"] == pytest.approx(delta @ np.linalg.inv(v) @ delta)
    else:
        assert contrast["status"] == "undefined" and contrast["statistic"] is None
    changed = oe.dfe(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", p=2)
    with pytest.raises(AnalysisError, match="same data"):
        oe.panel_ardl_hausman(mg, changed)


@pytest.mark.parametrize("method", ["mg", "pmg", "dfe"])
@pytest.mark.parametrize(
    "bad,code",
    [
        ("gap", "time_gaps"),
        ("duplicate", "repeated_time_values"),
        ("rank", "singular_design"),
        ("missing", "time_gaps"),
    ],
)
def test_panel_ecm_calendar_rank_missing_errors(method, bad, code):
    df = fixture()
    if bad == "gap":
        df = df.drop(index=13)
    if bad == "duplicate":
        df.loc[13, "t"] = 12
    if bad == "rank":
        df.x1 = df.x0
    if bad == "missing":
        df.loc[13, "x1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        getattr(oe, method)(data=df, y="y", x=["x0", "x1"], panel="unit", time="t", missing="drop")
    assert error.value.code == code


def test_pmg_nonconvergence_is_not_a_successful_result():
    with pytest.raises(AnalysisError) as error:
        oe.pmg(data=fixture(), y="y", x=["x0", "x1"], panel="unit", time="t", max_iterations=1)
    assert error.value.code == "nonconvergence"
