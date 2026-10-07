"""Independent R published recursions and NumPy joint likelihood/information."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.econometrics.mgarch import kernels
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def simulated_data(kind="dcc", n=200, seed=97):
    rng = np.random.default_rng(seed)
    target = np.array([[1.0, 0.35], [0.35, 1.0]])
    h = np.array([1.0, 1.0])
    q = target.copy()
    covariance = np.eye(2)
    c = np.array([[0.3, 0], [0.07, 0.4]])
    a = np.array([[0.38, 0.17], [-0.08, 0.28]])
    b = np.array([[0.60, 0.10], [0.06, 0.58]])
    values = []
    burn = 200 if kind == "bekk" else 0
    for t in range(n + burn):
        corr = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        shock = rng.multivariate_normal([0.0, 0.0], covariance if kind == "bekk" else corr) * (
            1 if kind == "bekk" else np.sqrt(h)
        )
        if t >= burn:
            values.append(shock)
        if kind == "bekk":
            covariance = c @ c.T + a.T @ np.outer(shock, shock) @ a + b.T @ covariance @ b
        else:
            z = shock / np.sqrt(h)
            q = 0.1 * target + 0.2 * np.outer(z, z) + 0.7 * q
            h = np.array([0.1, 0.1]) + 0.25 * shock**2 + 0.65 * h
    return pd.DataFrame(values, columns=["x", "y"])


def independent_path(values, y, initial, kind, mean=False):
    """NumPy physical-unit oracle, never calls a production parameter map."""
    d = y.shape[1]
    index = d if mean else 0
    residuals = y - (values[:d] if mean else 0)
    if kind == "bekk":
        c = np.array([[values[index], 0], [values[index + 1], values[index + 2]]])
        a = values[index + 3 : index + 7].reshape(2, 2)
        b = values[index + 7 : index + 11].reshape(2, 2)
        h = initial.copy()
        path = [h]
        for shock in residuals[:-1]:
            h = c @ c.T + a.T @ np.outer(shock, shock) @ a + b.T @ h @ b
            path.append(h)
        return np.array(path), residuals
    garch = values[index : index + 3 * d].reshape(d, 3)
    omega, alpha, beta = garch[:, 0], garch[:, 1], garch[:, 2]
    correlation = np.eye(d)
    for rho, (i, j) in zip(
        values[index + 3 * d : index + 3 * d + d * (d - 1) // 2],
        [(i, j) for i in range(d) for j in range(i)],
        strict=True,
    ):
        correlation[i, j] = correlation[j, i] = rho
    q = correlation.copy()
    h = np.diag(initial).copy()
    path = []
    for t, shock in enumerate(residuals):
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        path.append(np.sqrt(np.outer(h, h)) * r)
        z = shock / np.sqrt(h)
        if kind == "dcc":
            aa, bb = values[-2:]
            q = (1 - aa - bb) * correlation + aa * np.outer(z, z) + bb * q
        h = omega + alpha * shock**2 + beta * h
    return np.array(path), residuals


def independent_ll(values, y, initial, kind, mean=False):
    path, residuals = independent_path(values, y, initial, kind, mean)
    total = 0.0
    for covariance, residual in zip(path, residuals, strict=True):
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign > 0
        total -= 0.5 * (
            len(residual) * np.log(2 * np.pi)
            + logdet
            + residual @ np.linalg.solve(covariance, residual)
        )
    return total


def numerical_hessian(function, point):
    def central(scale):
        h = scale * (1 + np.abs(point))
        value = function(point)
        out = np.zeros((len(point), len(point)))
        for i in range(len(point)):
            di = np.eye(len(point))[i] * h[i]
            out[i, i] = (function(point + di) - 2 * value + function(point - di)) / h[i] ** 2
            for j in range(i):
                dj = np.eye(len(point))[j] * h[j]
                out[i, j] = out[j, i] = (
                    function(point + di + dj)
                    - function(point + di - dj)
                    - function(point - di + dj)
                    + function(point - di - dj)
                ) / (4 * h[i] * h[j])
        return out

    return (4 * central(1e-4) - central(2e-4)) / 3


@pytest.mark.parametrize("kind", ["ccc", "dcc", "bekk"])
def test_actual_R_published_full_matrix_covariance_correlation_and_forecasts(kind):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/evidence/market-131-mgarch/published-reference.json"
        ).read_text()
    )
    reference = fixture[kind]
    m = {
        name: torch.tensor(value, dtype=torch.float64)
        for name, value in reference["parameters"].items()
    }
    m["mu"] = torch.tensor(fixture["mean"], dtype=torch.float64)
    initial = (
        torch.tensor(reference["initializer"], dtype=torch.float64)
        if kind == "bekk"
        else torch.diag(torch.tensor(reference["initializer_variances"], dtype=torch.float64))
    )
    y = torch.tensor(fixture["returns"], dtype=torch.float64)
    kernels.validate_matrices(m, kind)
    path, residuals, state = kernels.covariance_path(m, y, initial, kind)
    correlation = (
        path
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
    )
    np.testing.assert_allclose(path.numpy(), reference["covariance_path"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        correlation.numpy(), reference["correlation_path"], rtol=1e-12, atol=1e-13
    )
    forecast = kernels.forecast_path(m, state, kind, fixture["horizons"])
    np.testing.assert_allclose(
        forecast.numpy(), reference["covariance_forecast"], rtol=1e-12, atol=1e-13
    )
    fc_corr = (
        forecast
        / forecast.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
        / forecast.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
    )
    np.testing.assert_allclose(
        fc_corr.numpy(), reference["correlation_forecast"], rtol=1e-12, atol=1e-13
    )
    assert float(kernels.matrix_log_likelihood(m, y, initial, kind)) == pytest.approx(
        reference["gaussian_log_likelihood"], abs=1e-12
    )
    assert "actual r" in fixture["reference_runtime"].lower()
    if kind == "bekk":
        assert "H_t[[i+1]]" in fixture["literal_bekk_forecast_block"]


@pytest.mark.parametrize("kind", ["ccc", "dcc", "bekk"])
def test_complete_physical_score_and_hessian_away_from_maximum(kind):
    data = simulated_data(n=35)
    y = data.to_numpy()
    initial = y[:20].T @ y[:20] / 20
    point = (
        np.array([0.3, 0.07, 0.4, 0.3, 0.08, -0.06, 0.28, 0.6, 0.08, 0.03, 0.55])
        if kind == "bekk"
        else np.r_[0.1, 0.12, 0.75, 0.1, 0.16, 0.7, 0.35, [0.15, 0.65] if kind == "dcc" else []]
    )

    def fn(q):
        return independent_ll(q, y, initial, kind)

    native = kernels.gradient(
        lambda q: kernels.matrix_log_likelihood(
            kernels.physical_matrices(q, kind, 2, False),
            torch.tensor(y),
            torch.tensor(initial),
            kind,
        ),
        torch.tensor(point),
        True,
    )
    assert float(native[0]) == pytest.approx(fn(point), abs=1e-10)
    h = 1e-5 * (1 + np.abs(point))
    gradient = np.array(
        [
            (fn(point + np.eye(len(point))[i] * h[i]) - fn(point - np.eye(len(point))[i] * h[i]))
            / (2 * h[i])
            for i in range(len(point))
        ]
    )
    np.testing.assert_allclose(native[1].numpy(), gradient, rtol=2e-6, atol=3e-5)
    np.testing.assert_allclose(
        native[2].numpy(), numerical_hessian(fn, point), rtol=2e-4, atol=2e-4
    )
    raw = kernels.encode(torch.tensor(point), kind, 2, False)
    np.testing.assert_allclose(kernels.decode(raw, kind, 2, False)[0].numpy(), point, atol=1e-12)


@pytest.mark.parametrize("kind", ["ccc", "dcc", "bekk"])
def test_joint_fit_full_covariance_persisted_forecasts_and_nuisance_information(kind):
    data = simulated_data(kind="bekk", n=400, seed=18) if kind == "bekk" else simulated_data()
    result = getattr(oe, "mgarch_" + kind)(data, ["x", "y"], intercept=False, tolerance=1e-8)
    initial = np.array(result.extra["initialization"]["initial_covariance"])
    point = np.array([c.estimate for c in result.coefficients])

    def function(q):
        return independent_ll(q, data.to_numpy(), initial, kind)

    assert result.metrics["log_likelihood"] == pytest.approx(function(point), abs=1e-10)
    expected = np.linalg.inv(-numerical_hessian(function, point))
    np.testing.assert_allclose(result.covariance_matrix, expected, rtol=3e-4, atol=2e-6)
    assert np.abs(np.array(result.covariance_matrix)[:6, 6:]).max() > 1e-7
    if kind == "bekk":
        assert all(
            abs(result.extra["parameter_matrices"][matrix][i][j]) > 1e-4
            for matrix in ("A", "B")
            for i, j in ((0, 1), (1, 0))
        )
    restored = ResultBundle.model_validate_json(json.dumps(result.model_dump(), allow_nan=False))
    forecasts = oe.forecast(restored, 5)
    assert len(forecasts) == 5 and np.array(forecasts.attrs["conditional_covariance"]).shape == (
        5,
        2,
        2,
    )
    assert forecasts.attrs["exact_full_covariance_horizons"] == (
        list(range(1, 6)) if kind == "bekk" else [1]
    )
    if kind != "bekk":
        assert "approximation" in forecasts.attrs["forecast_method"]
    else:
        assert "exact" in forecasts.attrs["forecast_method"]
    publication = oe.to_latex(forecasts, index=False)
    assert "tabular" in result.to_latex() and "tabular" in publication
    assert "condition on fitted parameters" in publication
    assert "parameter uncertainty is excluded" in publication
    if kind != "bekk":
        assert "approximation" in publication
    from openecon.console_worker import _execute

    displayed = _execute("display(forecasts)", {"forecasts": forecasts}, "mgarch-publication")
    assert displayed["error"] is None
    persisted_publication = displayed["outputs"][0]["latex"]
    assert "condition on fitted parameters" in persisted_publication
    if kind != "bekk":
        assert "approximation" in persisted_publication
    assert result.extra["model"]["joint_estimation"]
    assert result.provenance["solver_diagnostics"]["converged"]


def test_guards_dimension_initialization_nonstationarity_time_missing_budget_and_dataset():
    data = simulated_data()
    with pytest.raises(AnalysisError, match="dimension"):
        oe.mgarch_bekk(data.assign(z=data.x * 0.5), ["x", "y", "z"])
    with pytest.raises(AnalysisError, match="positive definite"):
        oe.mgarch_ccc(data, ["x", "y"], initial_covariance=[[1, 2], [2, 1]])
    with pytest.raises(AnalysisError, match="max_n"):
        oe.mgarch_ccc(data, ["x", "y"], max_n=50)
    with pytest.raises(AnalysisError, match="did not converge"):
        oe.mgarch_ccc(data, ["x", "y"], max_iterations=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.mgarch_dcc(data, ["x", "y"])
    data["t"] = np.arange(len(data)) * 2
    with pytest.raises(AnalysisError, match="consecutive"):
        oe.mgarch_ccc(data, ["x", "y"], time="t")
    data["date"] = pd.date_range("2020-01-01", periods=len(data), freq="D")
    with pytest.raises(AnalysisError, match="integer periods"):
        oe.mgarch_ccc(data, ["x", "y"], time="date")
    data.loc[3, "y"] = np.nan
    with pytest.raises(AnalysisError, match="interior missing"):
        oe.mgarch_ccc(data, ["x", "y"], missing="drop")
    from openecon.dataset import Dataset

    with pytest.raises(AnalysisError, match="Dataset execution route"):
        oe.mgarch_ccc(Dataset.from_frame(data), ["x", "y"])
    m = {
        "mu": torch.zeros(2, dtype=torch.float64),
        "C": torch.eye(2, dtype=torch.float64),
        "A": torch.eye(2, dtype=torch.float64),
        "B": torch.eye(2, dtype=torch.float64) * 0.5,
    }
    with pytest.raises(KernelError, match="stationarity"):
        kernels.validate_matrices(m, "bekk")


def test_three_dimensional_correlation_parameter_map_and_invalid_forecast_state():
    values = torch.tensor(
        [0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8, 0.2, 0.3, 0.25, 0.05, 0.9],
        dtype=torch.float64,
    )
    raw = kernels.encode(values, "dcc", 3, False)
    physical, m = kernels.decode(raw, "dcc", 3, False)
    np.testing.assert_allclose(physical.numpy(), values.numpy(), atol=1e-12)
    kernels.validate_matrices(m, "dcc")
    result = oe.mgarch_ccc(simulated_data(), ["x", "y"], intercept=False, tolerance=1e-6)
    broken = result.model_copy(deep=True)
    broken.extra["state"]["H"] = [[1, 2], [2, 1]]
    with pytest.raises(AnalysisError, match="positive definite"):
        oe.forecast(broken, 3)
    broken = result.model_copy(deep=True)
    broken.extra["parameter_matrices"]["omega"][0] *= 2
    with pytest.raises(AnalysisError, match="persisted"):
        oe.forecast(broken, 3)
    broken = result.model_copy(deep=True)
    broken.extra["state"]["h"][0] *= 2
    with pytest.raises(AnalysisError, match="disagrees"):
        oe.forecast(broken, 3)


def test_full_bekk_stationarity_uses_radius_not_the_stronger_norm_condition():
    m = {
        "mu": torch.zeros(2, dtype=torch.float64),
        "C": torch.eye(2, dtype=torch.float64) * 0.2,
        "A": torch.tensor([[0.0, 1.1], [0.0, 0.0]], dtype=torch.float64),
        "B": torch.eye(2, dtype=torch.float64) * 0.3,
    }
    constraints = kernels.validate_matrices(m, "bekk")
    assert constraints["spectral_radius"] == pytest.approx(0.09)
    assert constraints["spectral_norm_contraction"] > 1
    values = torch.tensor([0.2, 0, 0.2, 0, 1.1, 0, 0, 0.3, 0, 0, 0.3], dtype=torch.float64)
    raw = kernels.encode(values, "bekk", 2, False)
    np.testing.assert_allclose(
        kernels.decode(raw, "bekk", 2, False)[0].numpy(), values.numpy(), atol=1e-12
    )


def test_ccc_multistep_offdiagonal_is_labeled_plugin_not_exact_expectation():
    """Independent Gaussian quadrature demonstrates the nonlinear expectation gap."""
    m = {
        "mu": torch.zeros(2, dtype=torch.float64),
        "omega": torch.tensor([0.1, 0.07], dtype=torch.float64),
        "alpha": torch.tensor([0.35, 0.15], dtype=torch.float64),
        "beta": torch.tensor([0.5, 0.7], dtype=torch.float64),
        "target": torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float64),
    }
    state = {
        "H": torch.tensor([[1.0, 0.6], [0.6, 1.0]], dtype=torch.float64),
        "h": torch.ones(2, dtype=torch.float64),
        "Q": m["target"],
        "last_residual": torch.tensor([0.7, -0.3], dtype=torch.float64),
    }
    path = kernels.forecast_path(m, state, "ccc", 2).numpy()
    points, weights = np.polynomial.hermite.hermgauss(30)
    z = np.array(np.meshgrid(points, points)).reshape(2, -1).T * np.sqrt(2)
    probability = np.outer(weights, weights).flatten() / np.pi
    shocks = z @ np.linalg.cholesky(path[0]).T
    random_h = (
        np.array(m["omega"])
        + np.array(m["alpha"]) * shocks**2
        + np.array(m["beta"]) * np.diag(path[0])
    )
    exact_offdiagonal = 0.6 * np.sum(probability * np.sqrt(np.prod(random_h, axis=1)))
    np.testing.assert_allclose(
        np.sum(random_h * probability[:, None], axis=0), np.diag(path[1]), atol=1e-12
    )
    assert abs(exact_offdiagonal - path[1, 0, 1]) > 1e-3


@pytest.mark.parametrize("kind", ["ccc", "dcc"])
def test_three_dimensional_joint_fit_and_constant_mean_covariance_blocks(kind):
    target = np.array([[1.0, 0.35, 0.2], [0.35, 1.0, 0.3], [0.2, 0.3, 1.0]])
    rng = np.random.default_rng(27)
    h = np.ones(3)
    q = target.copy()
    rows = []
    for t in range(300):
        r = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        shock = rng.multivariate_normal(np.zeros(3), r) * np.sqrt(h)
        rows.append(shock + np.array([0.2, -0.15, 0.1]))
        z = shock / np.sqrt(h)
        q = 0.1 * target + 0.2 * np.outer(z, z) + 0.7 * q
        h = 0.1 + 0.25 * shock**2 + 0.65 * h
    result = getattr(oe, "mgarch_" + kind)(
        pd.DataFrame(rows, columns=["x", "y", "z"]), ["x", "y", "z"], tolerance=1e-7
    )
    assert result.extra["model"]["dimension"] == 3
    assert [c.term for c in result.coefficients[:3]] == ["mean:x", "mean:y", "mean:z"]
    covariance = np.array(result.covariance_matrix)
    assert np.max(np.abs(covariance[:3, 3:])) > 1e-7
    assert np.array(oe.forecast(result, 3).attrs["conditional_covariance"]).shape == (3, 3, 3)
