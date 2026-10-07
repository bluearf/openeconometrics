"""Independent scale/optimization/score-covariance and contamination checks."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.optimize._numdiff import approx_derivative

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.robust import smm
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def np_rho(u, c):
    z = np.minimum((u / c) ** 2, 1)
    return 3 * z - 3 * z * z + z**3


def np_psi(u, c):
    return u * np.maximum(1 - (u / c) ** 2, 0) ** 2


def np_scale(r, c, b):
    top = max(abs(r).max(), 1e-8)
    return brentq(lambda s: np_rho(r / s, c).mean() - b, top * 1e-12, top * 10, xtol=1e-12)


def s_oracle(x, y, c, b):
    """Independent Nelder-Mead minimization of the implicitly defined M-scale."""
    rng = np.random.default_rng(872)
    fits = []
    for _ in range(30):
        rows = rng.choice(len(y), x.shape[1], replace=False)
        initial = np.linalg.solve(x[rows], y[rows])

        def objective(coefficient):
            return np_scale(y - x @ coefficient, c, b)

        fit = minimize(
            objective,
            initial,
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 1500},
        )
        if fit.success:
            fits.append(fit)
    return min(fits, key=lambda fit: fit.fun)


def sample(seed=22, n=95):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"x": x, "y": 1 + 2 * x + rng.normal(size=n) * 0.5, "group": np.arange(n) % 9}
    )


def test_bisquare_scale_calibration_and_gaussian_efficiency_independent_integrals():
    c = smm.calibrated_tune(0.5)

    def normal(z):
        return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

    expectation = quad(lambda z: float(np_rho(z, c)) * normal(z), -12, 12, epsabs=1e-11)[0]
    assert_allclose(c, 1.54764498, rtol=1e-7)
    assert_allclose(expectation, 0.5, atol=1e-8)
    for tuning, expected in [(c, 0.286826), (4.685061, 0.95)]:

        def derivative(z):
            return (1 - (z / tuning) ** 2) * (1 - 5 * (z / tuning) ** 2) if abs(z) < tuning else 0.0

        a = quad(lambda z: derivative(z) * normal(z), -12, 12, points=[-tuning, tuning])[0]
        b = quad(
            lambda z: float(np_psi(z, tuning) ** 2) * normal(z), -12, 12, points=[-tuning, tuning]
        )[0]
        assert_allclose(a * a / b, expected, atol=2e-6)
        assert_allclose(smm.gaussian_efficiency(tuning), a * a / b, atol=1e-6)
    r = np.array([-0.2, 0.3, 0.5, 1, 2, 3, 50.0])
    assert_allclose(smm.m_scale(torch.tensor(r), c, 0.5), np_scale(r, c, 0.5), rtol=1e-10)


def test_s_and_fixed_scale_mm_match_independent_nonlinear_optimization():
    df = sample(n=55)
    df.loc[:13, "y"] += 35
    x = np.c_[np.ones(len(df)), df.x]
    y = df.y.to_numpy()
    sr = oe.sreg(data=df, y="y", x=["x"], starts=150, tolerance=1e-10)
    c = sr.metrics["scale_tune"]
    oracle = s_oracle(x, y, c, 0.5)
    assert_allclose([c.estimate for c in sr.coefficients], oracle.x, atol=2e-6)
    assert_allclose(sr.metrics["scale"], oracle.fun, rtol=2e-8)
    mm = oe.mmreg(data=df, y="y", x=["x"], starts=150, tolerance=1e-10)
    scale = oracle.fun
    final = minimize(
        lambda coefficient: np_rho((y - x @ coefficient) / scale, 4.685061).sum(),
        oracle.x,
        jac=lambda coefficient: (
            -6 / (4.685061**2 * scale) * (x.T @ np_psi((y - x @ coefficient) / scale, 4.685061))
        ),
        method="BFGS",
        options={"gtol": 1e-10},
    )
    assert_allclose([c.estimate for c in mm.coefficients], final.x, atol=2e-6)
    assert_allclose(mm.metrics["scale"], sr.metrics["scale"], rtol=0, atol=0)
    diag = mm.provenance["solver_diagnostics"]
    assert diag["mm_final_objective"] <= diag["mm_initial_objective"]
    assert diag["scale_equation_error"] < 1e-10


@pytest.mark.parametrize("method", ["sreg", "mmreg"])
@pytest.mark.parametrize("kind", ["robust", "cluster"])
def test_joint_scale_aware_covariance_matches_numerical_jacobian(method, kind):
    df = sample()
    kwargs = {"cluster": "group"} if kind == "cluster" else {}
    result = getattr(oe, method)(
        data=df, y="y", x=["x"], starts=80, covariance=kind, tolerance=1e-10, **kwargs
    )
    x = np.c_[np.ones(len(df)), df.x]
    y = df.y.to_numpy()
    n, k = x.shape
    bs = np.array(result.extra["s_coefficients"])
    bm = np.array([c.estimate for c in result.coefficients])
    s = result.metrics["scale"]
    c0 = result.metrics["scale_tune"]
    c1 = result.metrics["efficiency_tune"]
    theta = np.r_[bm, bs, np.log(s)] if method == "mmreg" else np.r_[bs, np.log(s)]

    def scores(t):
        sigma = np.exp(t[-1])
        s_beta = t[k : 2 * k] if method == "mmreg" else t[:k]
        u0 = (y - x @ s_beta) / sigma
        block = np.c_[x * np_psi(u0, c0)[:, None], np_rho(u0, c0) - 0.5]
        if method == "mmreg":
            block = np.c_[x * np_psi((y - x @ t[:k]) / sigma, c1)[:, None], block]
        return block

    a = -approx_derivative(lambda t: scores(t).sum(axis=0), theta, method="3-point")
    phi = scores(theta) @ np.linalg.inv(a).T
    if kind == "cluster":
        group = df.group.to_numpy()
        G = group.max() + 1
        sums = np.array([phi[group == g].sum(axis=0) for g in range(G)])
        full = sums.T @ sums * G / (G - 1) * (n - 1) / (n - k)
        assert result.inference["df_inference"] == G - 1
    else:
        full = phi.T @ phi * n / (n - k)
    assert_allclose(result.covariance_matrix, full[:k, :k], rtol=3e-7, atol=1e-10)


@pytest.mark.parametrize("contamination", [0.2, 0.4])
@pytest.mark.parametrize("method", ["sreg", "mmreg"])
def test_vertical_and_bad_leverage_contamination_does_not_break_recovery(contamination, method):
    clean = sample(n=150)
    count = int(len(clean) * contamination)
    estimates = []
    for amplitude in [1e3, 1e6]:
        df = clean.copy()
        df.loc[: count - 1, "x"] = 50 + np.arange(count) / count
        df.loc[: count - 1, "y"] = amplitude - 70 * df.loc[: count - 1, "x"]
        result = getattr(oe, method)(data=df, y="y", x=["x"], starts=300, seed=19)
        estimates.append([c.estimate for c in result.coefficients])
        assert_allclose(estimates[-1], [1, 2], atol=0.22)
        assert max(result.extra["case_weights"][:count]) < 1e-10
    assert_allclose(estimates[0], estimates[1], atol=0.03)
    ols = np.linalg.lstsq(np.c_[np.ones(len(df)), df.x], df.y, rcond=None)[0]
    assert np.linalg.norm(ols - [1, 2]) > 1000


def test_seed_sample_collinearity_serialization_latex_and_public_spec():
    df = sample()
    df["duplicate"] = 2 * df.x
    df.loc[4, "x"] = np.nan
    original = df.copy(deep=True)
    torch.manual_seed(176)
    before = torch.random.get_rng_state().clone()
    result = oe.mmreg(data=df, y="y", x=["x", "duplicate"], missing="drop", starts=30, seed=64)
    assert torch.equal(before, torch.random.get_rng_state())
    repeat = oe.fit(
        ModelSpec(
            estimator="mmreg",
            outcome="y",
            predictors=["x", "duplicate"],
            missing="drop",
            options={"starts": 30, "seed": 64},
        ),
        data=df,
    )
    assert_allclose(result.covariance_matrix, repeat.covariance_matrix, rtol=0, atol=0)
    assert result.sample_positions == [i for i in range(len(df)) if i != 4]
    assert result.provenance["omitted_terms"] == ["duplicate"]
    assert result.extra["case_weight_sample_positions"] == result.sample_positions
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    assert "\\begin{table}" in result.to_latex()
    pd.testing.assert_frame_equal(original, df)


def test_regression_equivariance_and_dataset_input_is_never_collected():
    from openecon.dataset import Dataset

    df = sample(n=70)
    initial = oe.mmreg(data=df, y="y", x=["x"], starts=60, seed=48)
    changed = df.assign(x=-5 * df.x + 20, y=-2 * df.y + 1000)
    transformed = oe.mmreg(data=changed, y="y", x=["x"], starts=60, seed=48)
    b = np.array([c.estimate for c in initial.coefficients])
    expected = np.array([1000 - 2 * b[0] - 8 * b[1], 0.4 * b[1]])
    assert_allclose([c.estimate for c in transformed.coefficients], expected, atol=2e-7)
    assert_allclose(transformed.metrics["scale"], 2 * initial.metrics["scale"], rtol=1e-8)
    dataset = Dataset.from_frame(df)
    for method in [oe.sreg, oe.mmreg]:
        with pytest.raises(AnalysisError) as error:
            method(data=dataset, y="y", x=["x"])
        assert error.value.code == "streaming_unsupported"


def test_intercept_only_and_no_intercept_domains_are_estimable():
    df = sample(n=80)
    for method in [oe.sreg, oe.mmreg]:
        location = method(data=df, y="y", x=[], starts=60)
        assert [c.term for c in location.coefficients] == ["Intercept"]
        assert location.coefficients[0].std_error > 0
        no_constant = method(data=df, y="y", x=["x"], intercept=False, starts=60)
        assert [c.term for c in no_constant.coefficients] == ["x"]
        assert no_constant.coefficients[0].std_error > 0


def test_explicit_errors_and_resource_guard():
    df = sample(n=30)
    for kwargs, code in [
        ({"efficiency_tune": 1.0}, "invalid_option"),
        ({"max_iterations": 1}, "nonconvergence"),
    ]:
        with pytest.raises(AnalysisError) as error:
            oe.mmreg(data=df, y="y", x=["x"], starts=20, **kwargs)
        assert error.value.code == code
    exact = df.assign(y=1 + 2 * df.x)
    with pytest.raises(AnalysisError) as error:
        oe.sreg(data=exact, y="y", x=["x"])
    assert error.value.code == "perfect_fit"
    wide = pd.DataFrame(np.ones((200, 100)), columns=[f"x{i}" for i in range(100)]).assign(
        y=np.arange(200)
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.mmreg(data=wide, y="y", x=list(wide.columns[:-1]))
    assert error.value.code == "workspace_limit"


def test_manifest_and_public_exports_do_not_import_tensor_runtime():
    code = "import sys; import openecon; from openecon.econometrics.registry import public_exports; assert {'sreg','mmreg','panel_mmqr'} <= public_exports().keys(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
