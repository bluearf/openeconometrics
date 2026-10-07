"""Independent oracles for the general GMM estimator (oe.gmm).

Linear moments are checked against the iv family's ivregress gmm (which is
itself checked against explicit algebra) and against explicit NumPy GMM
algebra; nonlinear moments against a brute-force SciPy minimization of an
independently coded criterion, with the covariance built from a numerically
differentiated moment Jacobian.
"""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.quantile import formula as formulas
from openecon.econometrics.systems import gmm_kernels as gk

LINEAR = ["y - {b0} - {b1}*x1 - {b2}*x2"]
INSTR = ["x1", "z1", "z2"]


def make_iv(seed=3, n=400):
    rng = np.random.default_rng(seed)
    z1, z2, x1, v = rng.normal(size=(4, n))
    u = rng.normal(size=n) * (1 + 0.5 * np.abs(z1))
    x2 = 0.5 * z1 + 0.5 * z2 + 0.5 * u + v
    frame = pd.DataFrame({"y": 1 + 0.5 * x1 - 0.7 * x2 + u, "x1": x1, "x2": x2, "z1": z1,
                          "z2": z2, "c": rng.integers(0, 40, n), "t": np.arange(n),
                          "w": rng.integers(1, 4, n).astype(float)})
    frame["yc"] = rng.poisson(np.exp(0.2 + 0.3 * x1 + 0.2 * z1))
    return frame


@pytest.fixture(scope="module")
def data():
    return make_iv()


def est(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


def iv_gmm(frame, **options):
    return oe.ivregress(data=frame, y="y", x=["x1"], endog=["x2"], instruments=["z1", "z2"],
                        method="gmm", **options)


@pytest.mark.parametrize("options", [
    {}, {"cluster": "c"}, {"igmm": True}, {"center": True}, {"wmatrix": "unadjusted"},
    {"covariance": "hac", "lags": 2, "time": "t"},
    {"weights": "w", "weight_type": "aweight"}, {"weights": "w", "weight_type": "fweight"},
])
def test_linear_gmm_reproduces_ivregress(data, options):
    reference = iv_gmm(data, **options)
    result = oe.gmm(data=data, moments=LINEAR, instruments=INSTR, winitial="unadjusted",
                    **options)
    b, se = est(result)
    rb, rse = est(reference)
    assert_allclose(b, rb, rtol=1e-9, atol=1e-12)
    assert_allclose(se, rse, rtol=1e-8)
    if reference.metrics.get("j") is not None and options.get("wmatrix") != "unadjusted":
        assert_allclose(result.metrics["j"], reference.metrics["j"], rtol=1e-7, atol=1e-12)
    assert result.inference["use_t"] is False
    assert result.nobs == reference.nobs


def test_linear_gmm_explicit_algebra_identity_start(data):
    n = len(data)
    z = np.column_stack([np.ones(n), data[INSTR].to_numpy()])
    x = np.column_stack([np.ones(n), data[["x1", "x2"]].to_numpy()])
    y = data.y.to_numpy()
    # One-step with W = I: b = (X'Z Z'X)^-1 X'Z Z'y.
    a = z.T @ x
    b1 = np.linalg.solve(a.T @ a, a.T @ (z.T @ y))
    u1 = y - x @ b1
    s1 = (z * u1[:, None]).T @ (z * u1[:, None])
    w = np.linalg.inv(s1)
    b2 = np.linalg.solve(a.T @ w @ a, a.T @ w @ (z.T @ y))
    v2 = np.linalg.inv(a.T @ w @ a)
    g2 = z.T @ (y - x @ b2)
    result = oe.gmm(data=data, moments=LINEAR, instruments=INSTR)
    b, se = est(result)
    assert_allclose(b, b2, rtol=1e-9)
    assert_allclose(se, np.sqrt(np.diag(v2)), rtol=1e-8)
    assert_allclose(result.metrics["j"], g2 @ w @ g2, rtol=1e-8)
    assert_allclose(result.tests["hansen_j"]["p_value"], stats.chi2.sf(g2 @ w @ g2, 1), rtol=1e-7)
    assert_allclose(result.metrics["criterion"], g2 @ w @ g2 / n, rtol=1e-8)
    # One-step: sandwich with the robust S at the one-step estimates.
    onestep = oe.gmm(data=data, moments=LINEAR, instruments=INSTR, twostep=False)
    bread = np.linalg.inv(a.T @ a)
    v1 = bread @ a.T @ s1 @ a @ bread
    assert_allclose(est(onestep)[0], b1, rtol=1e-9)
    assert_allclose(est(onestep)[1], np.sqrt(np.diag(v1)), rtol=1e-8)
    assert "hansen_j" not in onestep.tests
    # Two-step with a covariance that differs from the weight matrix: sandwich at b2.
    unadj = oe.gmm(data=data, moments=LINEAR, instruments=INSTR, covariance="nonrobust")
    u2 = y - x @ b2
    s_u = (u2 @ u2 / n) * z.T @ z
    vs = v2 @ a.T @ w @ s_u @ w @ a @ v2
    assert_allclose(est(unadj)[1], np.sqrt(np.diag(vs)), rtol=1e-8)


def _criterion(theta, frame, weight):
    y, x1, z1 = frame.yc.to_numpy(), frame.x1.to_numpy(), frame.z1.to_numpy()
    z = np.column_stack([np.ones(len(y)), x1, z1, z1 ** 2])
    u = y - np.exp(theta[0] + theta[1] * x1 + theta[2] * z1)
    g = z.T @ u
    return g @ weight @ g, z, u


def _g(theta, frame):
    return _criterion(theta, frame, np.eye(4))[1].T @ _criterion(theta, frame, np.eye(4))[2]


def test_nonlinear_gmm_brute_force(data):
    frame = data.assign(z1sq=data.z1 ** 2)
    moments = ["yc - exp({a} + {b}*x1 + {c}*z1)"]
    result = oe.gmm(data=frame, moments=moments, instruments=["x1", "z1", "z1sq"])
    first = minimize(lambda t: _criterion(t, frame, np.eye(4))[0], np.zeros(3), method="BFGS",
                     options={"gtol": 1e-12, "maxiter": 10000})
    _, z, u = _criterion(first.x, frame, np.eye(4))
    m = z * u[:, None]
    weight = np.linalg.inv(m.T @ m)
    second = minimize(lambda t: _criterion(t, frame, weight)[0], first.x, method="BFGS",
                      options={"gtol": 1e-12, "maxiter": 10000})
    b, se = est(result)
    assert_allclose(b, second.x, rtol=1e-5, atol=1e-6)
    # Covariance from a central-difference Jacobian of g at the estimate.
    h = 1e-6
    jac = np.column_stack([(_g(b + h * e, frame) - _g(b - h * e, frame)) / (2 * h)
                           for e in np.eye(3)])
    v = np.linalg.inv(jac.T @ weight @ jac)
    assert_allclose(se, np.sqrt(np.diag(v)), rtol=1e-4)
    assert result.tests["hansen_j"]["df"] == 1


def test_analytic_moment_jacobian_matches_differences(data):
    parsed = [formulas.parse("yc - exp({a} + {b}*x1) * {c}^2"), formulas.parse("x2 - {a}*z1")]
    columns = {name: __import__("torch").tensor(data[name].to_numpy())
               for name in ("yc", "x1", "x2", "z1")}
    import torch

    z = torch.tensor(np.column_stack([np.ones(len(data)), data.z1, data.z2]))
    system = gk.MomentSystem(parsed, [[0, 1, 2], [0]], columns, [z, z[:, :2]], None,
                             len(data), 3)
    theta = torch.tensor([0.2, -0.1, 1.3], dtype=torch.float64)
    _, g, jac = system.moments(theta)
    h = 1e-6
    numeric = torch.stack([(system.moments(theta + h * e, False)[1]
                            - system.moments(theta - h * e, False)[1]) / (2 * h)
                           for e in torch.eye(3, dtype=torch.float64)], dim=1)
    assert_allclose(jac.numpy(), numeric.numpy(), rtol=1e-6, atol=1e-6)


def test_two_equations_with_shared_parameter(data):
    moments = {"demand": "y - {b0} - {b1}*x1 - {g}*x2", "aux": "x2 - {a0} - {g}*z1"}
    result = oe.gmm(data=data, moments=moments,
                    instruments=[["x1", "z1", "z2"], ["z1"]], winitial="unadjusted")
    n = len(data)
    z1 = np.column_stack([np.ones(n), data[INSTR].to_numpy()])
    z2 = np.column_stack([np.ones(n), data.z1])
    y, x1, x2, zz = data.y.values, data.x1.values, data.x2.values, data.z1.values

    def g(theta):
        b0, b1, gg, a0 = theta
        u1 = y - b0 - b1 * x1 - gg * x2
        u2 = x2 - a0 - gg * zz
        return np.concatenate([z1.T @ u1, z2.T @ u2]), u1, u2

    w0 = np.linalg.inv(np.block([[z1.T @ z1, np.zeros((4, 2))], [np.zeros((2, 4)), z2.T @ z2]]))
    first = minimize(lambda t: g(t)[0] @ w0 @ g(t)[0], np.zeros(4), method="BFGS",
                     options={"gtol": 1e-12})
    _, u1, u2 = g(first.x)
    rows = np.column_stack([z1 * u1[:, None], z2 * u2[:, None]])
    w = np.linalg.inv(rows.T @ rows)
    second = minimize(lambda t: g(t)[0] @ w @ g(t)[0], first.x, method="BFGS",
                      options={"gtol": 1e-12})
    names = [c.term for c in result.coefficients]
    assert names == ["b0", "b1", "g", "a0"]
    assert_allclose(est(result)[0], second.x, rtol=1e-6, atol=1e-7)
    assert result.metrics["n_moments"] == 6 and result.tests["hansen_j"]["df"] == 2
    assert set(result.extra["instruments"]) == {"demand", "aux"}


def test_gmm_errors(data):
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=["y - {b0} - {b1}*x1 - {b2}*x2"], instruments=[],
               instrument_constant=True)
    assert error.value.code == "underidentified"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=["y - {a} - {b}"], instruments=["z1"])
    assert error.value.code == "not_identified"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=["y = {a}"], instruments=["z1"])
    assert error.value.code == "invalid_formula"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=LINEAR, instruments=INSTR, start={"zz": 1})
    assert error.value.code == "invalid_start"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=LINEAR, instruments=INSTR, wmatrix="cluster")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=LINEAR, instruments=INSTR, wmatrix="hac")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=LINEAR, instruments=INSTR, weights="w", weight_type="pweight",
               covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=["y - ln({a} - x1)"], instruments=["z1"])
    assert error.value.code == "invalid_start"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data, moments=LINEAR, instruments="z1")
    assert error.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=data.assign(c=data.c % 2), moments=LINEAR, instruments=INSTR, cluster="c")
    assert error.value.code == "singular_weight_matrix"


def test_round_trip_and_rendering(data):
    result = oe.gmm(data=data, moments=LINEAR, instruments=INSTR)
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert json.loads(result.model_dump_json())["spec"]["estimator"] == "gmm"
    assert "Hansen" in result.summary()
    assert "b1" in result.to_latex()
    assert result.spec.outcome == "y" and set(result.spec.columns["system"]) == {
        "y", "x1", "x2", "z1", "z2"}
    refit = oe.fit(result.spec, data=data)
    assert_allclose(est(refit)[0], est(result)[0], rtol=1e-12)
    missing = data.copy()
    missing.loc[0, "z2"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.gmm(data=missing, moments=LINEAR, instruments=INSTR)
    assert error.value.code == "missing_values"
    dropped = oe.gmm(data=missing, moments=LINEAR, instruments=INSTR, missing="drop")
    assert dropped.nobs == len(data) - 1
    collinear = oe.gmm(data=data.assign(z3=2 * data.z1), moments=LINEAR,
                       instruments=[*INSTR, "z3"])
    assert collinear.provenance["omitted_terms"] == ["1:z3"]
    assert_allclose(est(collinear)[0], est(result)[0], rtol=1e-9)


def test_pweights_match_ivregress(data):
    reference = iv_gmm(data, weights="w", weight_type="pweight")
    result = oe.gmm(data=data, moments=LINEAR, instruments=INSTR, winitial="unadjusted",
                    weights="w", weight_type="pweight")
    assert_allclose(est(result)[0], est(reference)[0], rtol=1e-9)
    assert_allclose(est(result)[1], est(reference)[1], rtol=1e-8)
