"""Independent NumPy Gaussian-density and finite-difference scientific oracles."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.spatial import kernels
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture_data(n=24, seed=281):
    keys = [f"unit-{i}" for i in range(n)]
    edges = [
        (keys[i], keys[j], weight)
        for i in range(n)
        for j, weight in (((i - 1) % n, 1.0), ((i + 1) % n, 2.0), ((i + 5) % n, 0.3))
    ]
    weights = oe.spatial_weights(keys, edges)
    w = weights.dense().numpy()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = np.linalg.solve(
        np.eye(n) - 0.35 * w,
        1 + 0.8 * x - 0.4 * z + np.linalg.solve(np.eye(n) - 0.2 * w, rng.normal(size=n)),
    )
    return pd.DataFrame({"id": keys, "y": y, "x": x, "z": z}), weights, w


def covariance_density(params, y, x, w, model, we=None):
    """Independent reduced-form multivariate Gaussian, not innovation likelihood."""
    we = w if we is None else we
    k, identity = x.shape[1], np.eye(len(y))
    rho = params[k] if model != "sem" else 0.0
    lam = params[k] if model == "sem" else params[k + 1] if model == "sac" else 0.0
    inverse_a = np.linalg.inv(identity - rho * w)
    inverse_b = np.linalg.inv(identity - lam * (w if model == "sem" else we))
    mean = inverse_a @ x @ params[:k]
    root = inverse_a @ inverse_b
    covariance = np.exp(params[-1]) * root @ root.T
    deviation = y - mean
    sign, determinant = np.linalg.slogdet(covariance)
    assert sign > 0
    return -0.5 * (
        len(y) * np.log(2 * np.pi)
        + determinant
        + deviation @ np.linalg.solve(covariance, deviation)
    )


def profile_oracle(spatial, y, x, w, model, we=None):
    we = w if we is None else we
    identity = np.eye(len(y))
    rho = spatial[0] if model != "sem" else 0.0
    lam = spatial[0] if model == "sem" else spatial[1] if model == "sac" else 0.0
    a, b = identity - rho * w, identity - lam * (w if model == "sem" else we)
    beta = np.linalg.lstsq(b @ x, b @ a @ y, rcond=None)[0]
    innovation = b @ (a @ y - x @ beta)
    params = np.r_[beta, spatial, np.log(innovation @ innovation / len(y))]
    return covariance_density(params, y, x, w, model, we), params


def finite_hessian(fn, point):
    def central(scale):
        h = scale * (1 + np.abs(point))
        value = fn(point)
        out = np.empty((len(point), len(point)))
        for i in range(len(point)):
            di = np.eye(len(point))[i] * h[i]
            out[i, i] = (fn(point + di) - 2 * value + fn(point - di)) / h[i] ** 2
            for j in range(i):
                dj = np.eye(len(point))[j] * h[j]
                out[i, j] = out[j, i] = (
                    fn(point + di + dj)
                    - fn(point + di - dj)
                    - fn(point - di + dj)
                    + fn(point - di - dj)
                ) / (4 * h[i] * h[j])
        return out

    return (4 * central(1e-4) - central(2e-4)) / 3


@pytest.mark.parametrize("model", ["sar", "sem", "sac", "sdm"])
def test_complete_gaussian_likelihood_covariance_and_independent_optimizer(model):
    scipy = pytest.importorskip("scipy.optimize")
    data, weights, w = fixture_data()
    we = np.roll(w, 2, axis=1)
    np.fill_diagonal(we, 0)
    we /= we.sum(axis=1)[:, None]
    error_weights = oe.spatial_weights(
        weights.keys,
        [
            (weights.keys[i], weights.keys[j], we[i, j])
            for i, j in zip(*np.nonzero(we), strict=True)
        ],
    )
    result = getattr(oe, model)(
        data,
        "y",
        ["x", "z"],
        key="id",
        spatial_weights=weights,
        **({"error_weights": error_weights} if model == "sac" else {}),
    )
    x = np.column_stack((np.ones(len(data)), data.x, data.z))
    if model == "sdm":
        x = np.column_stack((x, w @ data[["x", "z"]].to_numpy()))
    y, params = data.y.to_numpy(), np.array([c.estimate for c in result.coefficients])

    def ll(q):
        return covariance_density(q, y, x, w, model, we)

    assert result.metrics["log_likelihood"] == pytest.approx(ll(params), abs=2e-10)
    expected_hessian = finite_hessian(ll, params)
    expected_covariance = np.linalg.inv(-expected_hessian)
    np.testing.assert_allclose(result.covariance_matrix, expected_covariance, rtol=2e-5, atol=1e-6)
    native = kernels.derivatives(
        lambda q: kernels.log_likelihood(
            q,
            torch.tensor(y),
            torch.tensor(x),
            torch.tensor(w),
            model=model,
            w_error=torch.tensor(we),
        ),
        torch.tensor(params),
    )
    np.testing.assert_allclose(native[2].numpy(), expected_hessian, rtol=3e-5, atol=8e-6)
    assert np.max(np.abs(native[1].numpy())) < 2e-5

    def objective(spatial):
        return -profile_oracle(np.atleast_1d(spatial), y, x, w, model, we)[0]

    if model == "sac":
        oracle = scipy.differential_evolution(
            objective, [(-0.99998, 0.99998)] * 2, seed=9, tol=1e-11, polish=True
        )
    else:
        # Dense grid brackets all scalar candidates; independent Brent searches.
        grid = np.linspace(-0.99998, 0.99998, 41)
        candidates = [
            scipy.minimize_scalar(
                objective, bounds=(a, b), method="bounded", options={"xatol": 1e-12}
            )
            for a, b in zip(grid[:-1], grid[1:], strict=True)
        ]
        oracle = min(candidates, key=lambda candidate: candidate.fun)
    oracle_value, oracle_params = profile_oracle(np.atleast_1d(oracle.x), y, x, w, model, we)
    np.testing.assert_allclose(params, oracle_params, rtol=2e-5, atol=4e-6)
    assert result.metrics["log_likelihood"] == pytest.approx(oracle_value, abs=1e-8)
    diagnostics = result.provenance["solver_diagnostics"]
    assert diagnostics["converged"] and diagnostics["global_maximum_guaranteed"] is False
    assert result.metrics["aic"] == pytest.approx(2 * len(params) - 2 * oracle_value)
    roundtrip = ResultBundle.model_validate_json(json.dumps(result.model_dump(), allow_nan=False))
    assert roundtrip.spec == result.spec and roundtrip.covariance_matrix == result.covariance_matrix
    latex = roundtrip.to_latex()
    assert "tabular" in latex and ("rho" in latex if model != "sem" else "lambda" in latex)
    replay = oe.fit(ModelSpec.model_validate_json(result.spec.model_dump_json()), data=data)
    np.testing.assert_allclose([c.estimate for c in replay.coefficients], params, atol=1e-11)


@pytest.mark.parametrize("model", ["sar", "sdm"])
def test_multiplier_effects_perturb_covariates_and_full_covariance_delta(model):
    data, weights, w = fixture_data()
    result = getattr(oe, model)(data, "y", ["x", "z"], key="id", spatial_weights=weights)
    params = np.array([c.estimate for c in result.coefficients])
    terms = [c.term for c in result.coefficients]
    covariance = np.array(result.covariance_matrix)
    x = data[["x", "z"]].to_numpy()

    def mean(q, regressors):
        base = np.column_stack((np.ones(len(x)), regressors))
        if model == "sdm":
            base = np.column_stack((base, w @ regressors))
        return np.linalg.solve(
            np.eye(len(x)) - q[terms.index("rho")] * w, base @ q[: base.shape[1]]
        )

    def impact(q, predictor_index):
        sensitivity = []
        for j in range(len(x)):
            plus, minus = x.copy(), x.copy()
            plus[j, predictor_index] += 1e-4
            minus[j, predictor_index] -= 1e-4
            sensitivity.append((mean(q, plus) - mean(q, minus)) / 2e-4)
        matrix = np.column_stack(sensitivity)
        direct, total = np.trace(matrix) / len(x), matrix.sum() / len(x)
        return np.array([direct, total - direct, total])

    for predictor_index, predictor in enumerate(("x", "z")):
        rows = [row for row in result.extra["impacts"]["rows"] if row["predictor"] == predictor]
        expected = impact(params, predictor_index)
        np.testing.assert_allclose([row["estimate"] for row in rows], expected, atol=1e-10)
        jacobian = np.column_stack(
            [
                (
                    impact(params + np.eye(len(params))[i] * 1e-3, predictor_index)
                    - impact(params - np.eye(len(params))[i] * 1e-3, predictor_index)
                )
                / 2e-3
                for i in range(len(params))
            ]
        )
        # Nested central differences intentionally use a larger step; Richardson
        # error in rho is O(h^2), independent from the analytic multiplier formula.
        np.testing.assert_allclose(
            [row["gradient"] for row in rows], jacobian, rtol=2e-5, atol=1e-7
        )
        expected_se = np.sqrt(np.diag(jacobian @ covariance @ jacobian.T))
        np.testing.assert_allclose(
            [row["std_error"] for row in rows], expected_se, rtol=2e-5, atol=1e-7
        )
    assert "tabular" in oe.spatial_impacts(result).to_latex(index=False)


def test_permutation_alignment_missing_subgraph_and_weight_hash():
    data, weights, _ = fixture_data()
    base = oe.sar(data, "y", ["x"], key="id", spatial_weights=weights)
    shuffled = data.sample(frac=1, random_state=6).reset_index(drop=True)
    result = oe.sar(shuffled, "y", ["x"], key="id", spatial_weights=weights)
    np.testing.assert_allclose(
        [c.estimate for c in base.coefficients],
        [c.estimate for c in result.coefficients],
        atol=1e-8,
    )
    altered = data.copy()
    altered.loc[[2, 6], "x"] = np.nan
    dropped = oe.sar(altered, "y", ["x"], key="id", spatial_weights=weights, missing="drop")
    assert dropped.sample_positions == [i for i in range(len(data)) if i not in {2, 6}]
    aligned = weights.align(altered.loc[altered.x.notna(), "id"].tolist(), subset=True)
    manual = oe.sar(
        altered.dropna().reset_index(drop=True), "y", ["x"], key="id", spatial_weights=aligned
    )
    np.testing.assert_allclose(
        [c.estimate for c in dropped.coefficients],
        [c.estimate for c in manual.coefficients],
        atol=1e-11,
    )
    assert dropped.extra["spatial_summary"]["sha256"] != base.extra["spatial_summary"]["sha256"]
    assert dropped.extra["spatial_weights"] == aligned.to_payload()


def test_guards_budget_missing_identity_rank_convergence_and_dataset():
    data, weights, w = fixture_data()
    with pytest.raises(AnalysisError, match="max_n"):
        oe.sar(data, "y", ["x"], key="id", spatial_weights=weights, max_n=10)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        big_data, big_weights, _ = fixture_data(48)
        oe.sar(big_data, "y", ["x"], key="id", spatial_weights=big_weights)
    with pytest.raises(AnalysisError, match="No interior"):
        oe.sar(data, "y", ["x"], key="id", spatial_weights=weights, max_iterations=1)
    bad = data.copy()
    bad.loc[0, "id"] = "unknown"
    with pytest.raises(AnalysisError, match="key domains"):
        oe.sar(bad, "y", ["x"], key="id", spatial_weights=weights)
    bad = data.copy()
    bad.loc[0, "x"] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        oe.sar(bad, "y", ["x"], key="id", spatial_weights=weights)
    data["duplicate"] = data.x
    with pytest.raises(AnalysisError, match="rank deficient"):
        oe.sar(data, "y", ["x", "duplicate"], key="id", spatial_weights=weights)
    empty = oe.spatial_weights(weights.keys, [], isolates="zero")
    with pytest.raises(AnalysisError, match="nonzero"):
        oe.sem(data, "y", ["x"], key="id", spatial_weights=empty)
    from openecon.dataset import Dataset

    with pytest.raises(AnalysisError, match="Dataset execution route"):
        oe.sar(Dataset.from_frame(data), "y", ["x"], key="id", spatial_weights=weights)


def test_manifest_and_runtime_import_boundary():
    code = "import sys; from openecon.econometrics import registry; assert all(s in registry.names() for s in ['sar','sem','sac','sdm']); registry.public_exports(); assert 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    import inspect
    from openecon.econometrics.spatial import estimators, moran, weights

    for module in (kernels, estimators, moran, weights):
        text = inspect.getsource(module)
        assert (
            "import scipy" not in text
            and "import statsmodels" not in text
            and "import linearmodels" not in text
        )


@pytest.mark.parametrize("model", ["sar", "sem", "sac", "sdm"])
def test_unprofiled_derivatives_away_from_optimum(model):
    data, weights, w = fixture_data(18)
    x = np.column_stack((np.ones(len(data)), data.x, data.z))
    if model == "sdm":
        x = np.column_stack((x, w @ data[["x", "z"]].to_numpy()))
    spatial = [0.15, -0.12] if model == "sac" else [-0.15] if model == "sem" else [0.15]
    point = np.r_[np.linspace(0.4, 0.8, x.shape[1]), spatial, 0.25]

    def independent(q):
        return covariance_density(q, data.y.to_numpy(), x, w, model)

    native = kernels.derivatives(
        lambda q: kernels.log_likelihood(
            q, torch.tensor(data.y.to_numpy()), torch.tensor(x), torch.tensor(w), model=model
        ),
        torch.tensor(point),
    )
    h = 1e-5 * (1 + np.abs(point))
    gradient = np.array(
        [
            (
                independent(point + np.eye(len(point))[i] * h[i])
                - independent(point - np.eye(len(point))[i] * h[i])
            )
            / (2 * h[i])
            for i in range(len(point))
        ]
    )
    np.testing.assert_allclose(native[1].numpy(), gradient, rtol=2e-8, atol=1e-7)
    np.testing.assert_allclose(
        native[2].numpy(), finite_hessian(independent, point), rtol=3e-5, atol=2e-5
    )


def test_exact_cycle_logdet_and_unnormalized_domain_with_isolates_no_intercept():
    n = 7
    cycle = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        cycle[i, (i + 1) % n] = 2.0
    assert kernels.stable_bound(cycle) == pytest.approx(0.4999995)
    for rho in (-0.3, 0.0, 0.2):
        assert float(
            kernels._logdet(torch.eye(n, dtype=torch.float64) - rho * cycle)
        ) == pytest.approx(np.log(1 - (2 * rho) ** n), abs=1e-14)
    data, weights, w = fixture_data()
    w[0, :] = 0
    raw = oe.spatial_weights(
        weights.keys,
        [
            (weights.keys[i], weights.keys[j], 3.3 * w[i, j])
            for i, j in zip(*np.nonzero(w), strict=True)
        ],
        normalization="none",
        isolates="zero",
    )
    result = oe.sdm(data, "y", ["x", "z"], key="id", spatial_weights=raw, intercept=False)
    bound = result.provenance["solver_diagnostics"]["parameter_bounds"]["rho"]
    assert bound == pytest.approx((1 - 1e-6) / 3.3)
    assert result.extra["spatial_summary"]["isolate_keys"] == [weights.keys[0]]
    assert result.extra["spatial_summary"]["spectral_radius"] <= 3.3
    x = np.column_stack(
        (data[["x", "z"]].to_numpy(), raw.dense().numpy() @ data[["x", "z"]].to_numpy())
    )
    params = np.array([c.estimate for c in result.coefficients])
    assert result.metrics["log_likelihood"] == pytest.approx(
        covariance_density(params, data.y.to_numpy(), x, raw.dense().numpy(), "sdm"), abs=1e-10
    )
    with pytest.raises(Exception, match="stable multiplier domain"):
        bad = torch.tensor(params)
        bad[-2] = bound * 1.01
        kernels.multiplier_impacts(
            bad,
            torch.tensor(result.covariance_matrix),
            raw.dense(),
            terms=[c.term for c in result.coefficients],
            predictors=["x", "z"],
            model="sdm",
        )
