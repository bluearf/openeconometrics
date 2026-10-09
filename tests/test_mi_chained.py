"""Independent posterior oracles and integrity checks for chained imputation."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi import chained
from openecon.econometrics.mi.common import MIResult
from openecon.resources import use_workspace_budget


def _normal_frame():
    generator = torch.Generator().manual_seed(9137)
    x = torch.linspace(-2, 2, 42, dtype=torch.float64)
    y = 0.7 + 1.3 * x + torch.randn(42, dtype=torch.float64, generator=generator)
    y[[2, 7, 9, 19, 30, 41]] = float("nan")
    return pd.DataFrame({"y": y.tolist(), "x": x.tolist()},
                        index=pd.Index([i // 2 for i in range(42)], name="source_row"))


def test_gaussian_posterior_matches_hand_regression_and_full_covariance_moments():
    # The reference is ordinary finite-dimensional matrix algebra; it does not
    # call another imputation package or reproduce the kernel's QR calculation.
    x1 = np.linspace(-2.2, 2.4, 20)
    x2 = 0.7 * x1 + np.cos(np.arange(20))
    design = np.column_stack([np.ones(20), x1, x2])
    y = 0.8 + 1.2 * x1 - 0.6 * x2 + np.sin(1.7 * np.arange(20))
    expected_beta = np.linalg.solve(design.T @ design, design.T @ y)
    residual = y - design @ expected_beta
    expected_sse = residual @ residual
    df = len(y) - design.shape[1]
    variance_mean = expected_sse / (df - 2)
    expected_covariance = variance_mean * np.linalg.inv(design.T @ design)
    fit = chained._normal_parameters(torch.tensor(design), torch.tensor(y))
    assert fit.df == df
    np.testing.assert_allclose(fit.beta_hat, expected_beta, atol=2e-14)
    assert float(fit.sse) == pytest.approx(expected_sse, rel=2e-14)
    generator = torch.Generator().manual_seed(3394)
    draws = [chained._gaussian_draw(fit, generator) for _ in range(12_000)]
    betas = torch.stack([value[0] for value in draws]).numpy()
    variances = np.array([float(value[1]) for value in draws])
    np.testing.assert_allclose(betas.mean(0), expected_beta, atol=0.012)
    np.testing.assert_allclose(np.cov(betas.T), expected_covariance, rtol=0.065, atol=0.001)
    assert variances.mean() == pytest.approx(variance_mean, rel=0.025)
    # Correlated regressors make off-diagonal uncertainty material.
    assert abs(expected_covariance[1, 2]) > 0.01


def test_gaussian_missing_predictive_joint_moments_include_beta_and_residual_uncertainty():
    x = torch.column_stack((torch.ones(18, dtype=torch.float64),
                            torch.linspace(-2, 2, 18, dtype=torch.float64)))
    y = 0.4 + 1.7 * x[:, 1] + torch.sin(torch.arange(18, dtype=torch.float64) * 1.4)
    fit = chained._normal_parameters(x, y)
    missing_design = torch.tensor([[1.0, 0.5], [1.0, 1.1]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(64197)
    predictions = []
    for _ in range(12_000):
        beta, sigma2 = chained._gaussian_draw(fit, generator)
        noise = torch.randn(2, dtype=torch.float64, generator=generator)
        predictions.append(missing_design @ beta + sigma2.sqrt() * noise)
    actual = torch.stack(predictions).numpy()
    xx = x.numpy()
    yy = y.numpy()
    beta = np.linalg.solve(xx.T @ xx, xx.T @ yy)
    sse = np.square(yy - xx @ beta).sum()
    expected_variance = sse / (len(yy) - xx.shape[1] - 2)
    xm = missing_design.numpy()
    expected_covariance = expected_variance * (np.eye(2) + xm @ np.linalg.inv(xx.T @ xx) @ xm.T)
    np.testing.assert_allclose(actual.mean(0), xm @ beta, atol=0.02)
    np.testing.assert_allclose(np.cov(actual.T), expected_covariance, rtol=0.11, atol=0.01)


def test_normal_fcs_preserves_all_observed_cells_index_replay_and_global_rng():
    frame = _normal_frame()
    before = frame.copy(deep=True)
    state = torch.random.get_rng_state().clone()
    result = chained.mi_chained(frame, ["y", "x"], methods={"y": "normal"},
                                m=4, seed=541, burn=3, iterations=4)
    assert torch.equal(state, torch.random.get_rng_state())
    pd.testing.assert_frame_equal(before, frame)
    for number in range(1, 5):
        completed = result.dataset(imputation=number)
        assert completed.index.equals(frame.index)
        assert completed.columns.tolist() == frame.columns.tolist()
        assert not completed.isna().any().any()
        for name in frame:
            observed = frame[name].notna()
            assert completed.loc[observed, name].tolist() == frame.loc[observed, name].tolist()
    replay = chained.mi_chained(frame, ["y", "x"], methods={"y": "normal"},
                                m=4, seed=541, burn=3, iterations=4)
    assert result.model_dump_json() == replay.model_dump_json()
    changed = chained.mi_chained(frame, ["y", "x"], methods={"y": "normal"},
                                 m=4, seed=542, burn=3, iterations=4)
    assert result.completed_matrices != changed.completed_matrices
    assert result.metadata["imputation_seeds"] == (541, 542, 543, 544)
    assert result.metadata["convergence_claim"] is False
    restored = MIResult.model_validate_json(result.model_dump_json())
    pd.testing.assert_frame_equal(restored.dataset(imputation=2), result.dataset(imputation=2))
    assert result.table is not None
    assert isinstance(result.latex, str) and result.latex


def test_pmm_observed_support_and_donor_positions_are_exact():
    frame = _normal_frame()
    result = chained.mi_chained(frame, ["y", "x"], methods={"y": "pmm"},
                                m=4, seed=98, burn=2, iterations=3, donors=4)
    missing_positions = np.flatnonzero(frame.y.isna())
    support = set(frame.y.dropna())
    for chain in result.metadata["chain_diagnostics"]:
        for trace in chain["trace"]:
            donor_positions = trace["variables"]["y"]["donor_row_positions"]
            assert len(donor_positions) == len(missing_positions)
            assert all(not pd.isna(frame.y.iloc[position]) for position in donor_positions)
        completed = result.dataset(imputation=chain["imputation"])
        assert set(completed.y.iloc[missing_positions]) <= support
        last = chain["trace"][-1]["variables"]["y"]["donor_row_positions"]
        assert completed.y.iloc[missing_positions].tolist() == frame.y.iloc[list(last)].tolist()


def test_pmm_type_one_metric_against_independent_donor_calculation():
    x = torch.tensor([[1., -2.], [1., -1.], [1., 0.], [1., 1.], [1., 2.]], dtype=torch.float64)
    y = torch.tensor([-1., 2., 0., 3., 4.], dtype=torch.float64)
    xm = torch.tensor([[1., -0.3], [1., 0.7], [1., 2.4]], dtype=torch.float64)
    fit = chained._normal_parameters(x, y)
    actual, beta, _, donors = chained._pmm_draw(
        x, xm, y, fit, torch.Generator().manual_seed(351), donors=1, ties="stable"
    )
    # Use normal-equation OLS on observed rows, not the kernel beta_hat.
    xx, yy = x.numpy(), y.numpy()
    reference_hat = np.linalg.solve(xx.T @ xx, xx.T @ yy)
    distances = abs((xx @ reference_hat)[None, :] - (xm.numpy() @ beta.numpy())[:, None])
    expected = np.argmin(distances, axis=1)
    np.testing.assert_array_equal(donors, expected)
    np.testing.assert_array_equal(actual, yy[expected])


def test_pmm_stable_ties_are_row_stable_and_random_ties_are_seeded():
    x = torch.ones((8, 1), dtype=torch.float64)
    y = torch.tensor([1., 8., 3., 6., 2., 7., 4., 5.], dtype=torch.float64)
    xm = torch.ones((80, 1), dtype=torch.float64)
    fit = chained._normal_parameters(x, y)
    stable = chained._pmm_draw(x, xm, y, fit, torch.Generator().manual_seed(42),
                               donors=3, ties="stable")
    assert set(stable[3].tolist()) == {0, 1, 2}
    randomized = chained._pmm_draw(x, xm, y, fit, torch.Generator().manual_seed(42),
                                   donors=3, ties="random")
    replay = chained._pmm_draw(x, xm, y, fit, torch.Generator().manual_seed(42),
                               donors=3, ties="random")
    assert set(randomized[3].tolist()) == set(range(8))
    assert torch.equal(randomized[3], replay[3])


def test_logistic_mh_posterior_against_one_dimensional_quadrature():
    # An intercept-only 3/10-success posterior is not an asymptotic normal
    # approximation. Integrate its actual Gaussian-prior density independently.
    x = torch.ones((10, 1), dtype=torch.float64)
    y = torch.tensor([1.] * 3 + [0.] * 7, dtype=torch.float64)
    prior_scale = 1.3

    def density(beta):
        return np.exp(3 * beta - 10 * np.logaddexp(0, beta) - beta * beta / (2 * prior_scale ** 2))

    normalizer = quad(density, -30, 30, epsabs=1e-12)[0]
    expected_mean = quad(lambda beta: beta * density(beta), -30, 30, epsabs=1e-12)[0] / normalizer
    expected_second = quad(lambda beta: beta ** 2 * density(beta), -30, 30, epsabs=1e-12)[0] / normalizer
    expected_probability = quad(lambda beta: density(beta) / (1 + np.exp(-beta)),
                                -30, 30, epsabs=1e-12)[0] / normalizer
    generator = torch.Generator().manual_seed(77814)
    beta, _ = chained._logit_draw(x, y, generator, prior_scale=prior_scale,
                                  proposal_scale=1.8, burn=1500, steps=1)
    draws = []
    accepted = proposals = 0
    for _ in range(6000):
        beta, diagnostic = chained._logit_draw(x, y, generator, prior_scale=prior_scale,
                                               proposal_scale=1.8, burn=0, steps=5, initial=beta)
        draws.append(float(beta[0]))
        accepted += diagnostic["accepted_sampling"]
        proposals += diagnostic["sampling_proposals"]
    actual = np.array(draws)
    assert actual.mean() == pytest.approx(expected_mean, abs=0.04)
    assert actual.var() == pytest.approx(expected_second - expected_mean ** 2, rel=0.07)
    assert (1 / (1 + np.exp(-actual))).mean() == pytest.approx(expected_probability, abs=0.008)
    assert 0.25 < accepted / proposals < 0.8


def test_logistic_fcs_handles_separation_with_stated_proper_prior_and_finite_diagnostics():
    x = np.linspace(-2, 2, 30)
    b = (x > 0).astype(float)
    b[[1, 5, 17, 22, 28]] = np.nan
    frame = pd.DataFrame({"b": b, "x": x})
    result = chained.mi_chained(frame, ["b", "x"], methods={"b": "logit"}, m=3,
                                seed=336, burn=2, iterations=3, logit_burn=70, logit_steps=80)
    for number in range(1, 4):
        completed = result.dataset(imputation=number)
        assert set(completed.b) == {0, 1}
        assert completed.b[frame.b.notna()].tolist() == frame.b.dropna().tolist()
    sampler = result.metadata["logit_sampler"]
    assert sampler["exact_target"] is True
    assert sampler["stationarity_claim"] is False
    assert result.metadata["logit_prior"]["scale"] == 2.5
    for chain in result.metadata["chain_diagnostics"]:
        totals = chain["logistic_acceptance"]["b"]
        assert totals["proposals"] == 5 * 150
        assert 0 < totals["accepted"] <= totals["proposals"]
        assert totals["acceptance_rate"] == totals["accepted"] / totals["proposals"]
        for trace in chain["trace"]:
            variable = trace["variables"]["b"]
            assert variable["burn_proposals"] == 70
            assert variable["sampling_proposals"] == 80
            assert 0 <= variable["mean_probability"] <= 1
    replay = chained.mi_chained(frame, ["b", "x"], methods={"b": "logit"}, m=3,
                                seed=336, burn=2, iterations=3, logit_burn=70, logit_steps=80)
    assert result.model_dump_json() == replay.model_dump_json()


def test_mixed_normal_pmm_binary_fcs_uses_latest_imputed_predictors():
    generator = torch.Generator().manual_seed(625)
    x = torch.linspace(-2, 2, 55, dtype=torch.float64)
    b = (torch.rand(55, generator=generator) < torch.sigmoid(x)).to(torch.float64)
    y = 1.3 + x + 0.5 * b + torch.randn(55, dtype=torch.float64, generator=generator)
    z = -0.2 + 0.6 * x + torch.randn(55, dtype=torch.float64, generator=generator)
    y[[3, 9, 11, 22, 36, 45]] = float("nan")
    z[[4, 9, 19, 30, 49]] = float("nan")
    b[[1, 9, 17, 29, 43, 51]] = float("nan")
    frame = pd.DataFrame({"y": y.tolist(), "z": z.tolist(), "b": b.tolist(), "x": x.tolist()})
    result = chained.mi_chained(frame, list(frame), methods={"b": "logit", "z": "pmm", "y": "normal"},
                                m=3, seed=889, burn=2, iterations=2, donors=3,
                                logit_burn=20, logit_steps=30)
    assert result.metadata["visit_order"] == ("y", "z", "b")
    assert result.metadata["cycles_per_chain"] == 4
    for number in range(1, 4):
        completed = result.dataset(imputation=number)
        assert not completed.isna().any().any()
        for name in frame:
            assert completed[name][frame[name].notna()].tolist() == frame[name].dropna().tolist()
        assert set(completed.z[frame.z.isna()]) <= set(frame.z.dropna())
        assert set(completed.b) <= {0, 1}


def test_explicit_predictor_mapping_supports_intercept_only_and_excluded_constant():
    frame = _normal_frame().assign(constant=7.0)
    with pytest.raises(AnalysisError, match="rank deficient"):
        chained.mi_chained(frame, list(frame), methods={"y": "normal"}, m=1, burn=0, iterations=1)
    result = chained.mi_chained(frame, list(frame), methods={"y": "normal"},
                                predictors={"y": []}, m=2, burn=0, iterations=1)
    assert result.metadata["predictors"]["y"] == ()
    assert result.dataset().constant.tolist() == frame.constant.tolist()
    assert len(result.metadata["chain_diagnostics"][0]["trace"][0]["variables"]["y"]["coefficient_draw"]) == 1


@pytest.mark.parametrize("options", [
    {"methods": {}}, {"methods": {"y": "normal", "x": "normal"}},
    {"methods": {"y": "probit"}}, {"methods": {"y": ["normal"]}},
    {"predictors": {}}, {"predictors": {"y": ["y"]}},
    {"predictors": {"y": ["x", "x"]}}, {"predictors": {"y": ["absent"]}},
    {"predictors": {"y": "x"}}, {"burn": -1}, {"iterations": 0}, {"m": True},
    {"seed": -1}, {"seed": True}, {"seed": 2 ** 63}, {"pmm_ties": "arbitrary"},
    {"logit_prior_scale": 0}, {"logit_prior_scale": float("inf")},
    {"logit_prior_scale": 1e-200},
    {"logit_proposal_scale": True}, {"logit_steps": 0}, {"logit_burn": -1},
])
def test_options_fail_closed(options):
    kwargs = {"methods": {"y": "normal"}, "m": 1, "burn": 0, "iterations": 1}
    kwargs.update(options)
    with pytest.raises(AnalysisError):
        chained.mi_chained(_normal_frame(), ["y", "x"], **kwargs)


@pytest.mark.parametrize("method, frame, code", [
    ("normal", pd.DataFrame({"y": [1., 2., np.nan], "x": [1., 2., 3.]}), "insufficient_observations"),
    ("normal", pd.DataFrame({"y": [1., 2., 4., 3., np.nan], "x": [1.] * 5}), "singular_design"),
    ("normal", pd.DataFrame({"y": [1., 3., 5., 7., np.nan], "x": [0., 1., 2., 3., 4.]}), "zero_residual_variance"),
    ("normal", pd.DataFrame({"y": [np.nan] * 5, "x": list(range(5))}), "all_missing"),
    ("pmm", pd.DataFrame({"y": [1., 2., 4., np.nan, np.nan], "x": list(range(5))}), "invalid_spec"),
    ("logit", pd.DataFrame({"y": [0., 0.5, 1., np.nan], "x": list(range(4))}), "invalid_binary_outcome"),
    ("logit", pd.DataFrame({"y": [1., 1., 1., np.nan], "x": list(range(4))}), "single_class"),
    ("logit", pd.DataFrame({"y": [0., 1., 1., np.nan], "x": [2.] * 4}), "singular_design"),
])
def test_degenerate_conditionals_do_not_drop_rows_add_ridge_or_shrink_donors(method, frame, code):
    with pytest.raises(AnalysisError) as failure:
        chained.mi_chained(frame, ["y", "x"], methods={"y": method}, m=1, burn=0, iterations=1)
    assert failure.value.code == code


def test_complete_input_rejects_missing_method_request():
    with pytest.raises(AnalysisError) as failure:
        chained.mi_chained(pd.DataFrame({"x": [1., 2., 3.]}), ["x"], methods={})
    assert failure.value.code == "no_missing_values"


def test_work_rejection_happens_before_any_conditional_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A conditional fit must not run after a rejected budget.")

    monkeypatch.setattr(chained, "_normal_parameters", forbidden)
    with pytest.raises(AnalysisError) as failure:
        chained.mi_chained(_normal_frame(), ["y", "x"], methods={"y": "pmm"},
                           donors=3, m=5, burn=2, iterations=4, max_work=5000)
    assert failure.value.code in {"resource_limit", "mi_work_limit"}
    with pytest.raises(AnalysisError) as failure:
        chained.mi_chained(_normal_frame(), ["y", "x"], methods={"y": "normal"},
                           burn=1000, iterations=1)
    assert failure.value.code == "resource_limit"


def test_mh_work_includes_all_conditional_burn_and_sampling_steps(monkeypatch):
    frame = pd.DataFrame({"b": [0., 1., 0., 1., np.nan], "x": [-2., -1., 1., 2., 0.]})

    def forbidden(*args, **kwargs):
        raise AssertionError("MH must not run after a rejected work budget.")

    monkeypatch.setattr(chained, "_logit_draw", forbidden)
    with pytest.raises(AnalysisError) as failure:
        chained.mi_chained(frame, ["b", "x"], methods={"b": "logit"}, m=2, burn=0,
                           iterations=1, logit_burn=1000, logit_steps=1000, max_work=5000)
    assert failure.value.code == "resource_limit"


def test_chain_trace_workspace_is_checked_before_any_conditional_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A conditional fit must not run after a rejected trace allocation.")

    monkeypatch.setattr(chained, "_normal_parameters", forbidden)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as failure:
            chained.mi_chained(_normal_frame(), ["y", "x"], methods={"y": "normal"},
                               m=10, burn=10, iterations=10)
    assert failure.value.code == "workspace_limit"
    assert "conditional traces" in str(failure.value)


def test_envelope_rejects_observed_cell_tampering():
    result = chained.mi_chained(_normal_frame(), ["y", "x"], methods={"y": "normal"},
                                m=1, burn=0, iterations=1)
    state = copy.deepcopy(result.model_dump())
    matrices = [[list(row) for row in matrix] for matrix in state["completed_matrices"]]
    matrices[0][0][0] += 1
    state["completed_matrices"] = matrices
    with pytest.raises(ValueError):
        MIResult.model_validate(state)


def test_estimation_module_has_no_scipy_numpy_statsmodels_kernel_dependency():
    tree = ast.parse(Path(chained.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.split(".")[0] in {"numpy", "scipy", "statsmodels"} for name in imports)
