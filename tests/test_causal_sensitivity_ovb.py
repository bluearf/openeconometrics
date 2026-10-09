"""Actual augmented OLS and FWL references for single-confounder sensitivity."""

import copy
import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.stats import t
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.ovb import ovb_sensitivity
from openecon.resources import use_workspace_budget


def fixture_data(n=80, sign=1, binary=False):
    rng = np.random.default_rng(304)
    x = rng.normal(size=(n, 2))
    z = 0.4 * x[:, 0] + rng.normal(size=n)
    d = 0.8 * z + 0.5 * x[:, 1] + rng.normal(size=n)
    if binary:
        d = (d > np.median(d)).astype(float)
    y = sign * (1.1 * d + 0.9 * z) + x @ np.array([0.4, -0.6]) + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x[:, 0], x2=x[:, 1], z=z))


def analyze(data=None, **kwargs):
    return ovb_sensitivity(fixture_data() if data is None else data, "y", "d", ["x1", "x2"], **kwargs)


def ols(x, y):
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficient
    df = len(y) - x.shape[1]
    covariance = np.linalg.inv(x.T @ x) * (residual @ residual / df)
    return coefficient, covariance, residual, df


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("binary", [False, True])
def test_actual_augmented_scalar_confounder_ols_fwl_oracle(sign, binary):
    data = fixture_data(sign=sign, binary=binary)
    controls = np.column_stack([np.ones(len(data)), data[["x1", "x2"]]])
    design = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2"]]])
    augmented = np.column_stack([design, data.z])
    base, covariance, base_residual, df = ols(design, data.y)
    full, full_covariance, full_residual, full_df = ols(augmented, data.y)
    _, _, d_residual, _ = ols(controls, data.d)
    _, _, dz_residual, _ = ols(np.column_stack([controls, data.z]), data.d)
    rd = 1 - dz_residual @ dz_residual / (d_residual @ d_residual)
    ry = 1 - full_residual @ full_residual / (base_residual @ base_residual)
    result = analyze(data, r2_treatment=[rd], r2_outcome=[ry])
    assert_allclose(result["coefficients"].estimate, base, rtol=2e-13, atol=1e-14)
    assert_allclose(result["covariance"], covariance, rtol=5e-13, atol=1e-15)
    rows = result["sensitivity"]
    exact = rows.loc[rows.direction == "reduce"].iloc[0]
    assert_allclose(exact.estimate, full[1], rtol=4e-13)
    assert_allclose(exact.variance, full_covariance[1, 1], rtol=4e-13)
    se = np.sqrt(full_covariance[1, 1])
    expected = [full[1], se, full[1] / se, full_df, 2 * t.sf(abs(full[1] / se), full_df),
                full[1] - t.ppf(.975, full_df) * se, full[1] + t.ppf(.975, full_df) * se]
    assert_allclose(exact[["estimate", "std_error", "t", "df", "p_value", "ci_lower", "ci_upper"]].astype(float), expected, rtol=1e-11, atol=1e-13)
    assert_allclose(result.attrs["state"]["fitted"], design @ base, rtol=2e-13, atol=1e-14)
    assert_allclose(result.attrs["state"]["residuals"], base_residual, rtol=2e-13, atol=1e-14)
    assert result.attrs["state"]["adjusted_df"] == df - 1
    assert result.attrs["state"]["adjusted_joint_covariance"] is None


def test_increasing_bias_orientation_matches_actual_negative_confounder_effect():
    data = fixture_data()
    # Replace Y with a negative omitted effect while keeping the target large
    # enough that the fitted coefficient is positive.
    data.y = data.y + 5 * data.d - 1.8 * data.z
    x = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2"]]])
    controls = np.column_stack([np.ones(len(data)), data[["x1", "x2"]]])
    base, _, residual, _ = ols(x, data.y)
    full, full_cov, residual_z, _ = ols(np.column_stack([x, data.z]), data.y)
    _, _, d_resid, _ = ols(controls, data.d)
    _, _, dz_resid, _ = ols(np.column_stack([controls, data.z]), data.d)
    rd = 1 - dz_resid @ dz_resid / (d_resid @ d_resid)
    ry = 1 - residual_z @ residual_z / (residual @ residual)
    result = analyze(data, r2_treatment=[rd], r2_outcome=[ry], direction="increase")
    row = result["sensitivity"].iloc[0]
    assert full[1] > base[1] > 0
    assert_allclose([row.estimate, row.variance], [full[1], full_cov[1, 1]], rtol=1e-12)


@pytest.mark.parametrize("sign", [1, -1])
def test_equal_strength_point_estimate_robustness_independent_root(sign):
    result = analyze(fixture_data(sign=sign))
    base = result["coefficients"].loc["d"]
    f = abs(base.t) / np.sqrt(base.df)
    root = brentq(lambda rho: rho / np.sqrt(1 - rho) - f, 0, 1 - 1e-12, xtol=1e-14)
    row = result["robustness"].iloc[0]
    assert_allclose([row.r2_treatment, row.r2_outcome], root, rtol=1e-13)
    assert_allclose(row.bias_magnitude, abs(base.estimate), rtol=3e-15)
    assert abs(row.tipping_estimate) < 1e-14
    tipping = analyze(fixture_data(sign=sign), r2_treatment=[root], r2_outcome=[root], direction="reduce")
    assert abs(tipping["sensitivity"].iloc[0].estimate) < 1e-13


def test_complete_grid_zero_strength_and_reduction_through_zero():
    result = analyze(r2_treatment=[0, .2, .95], r2_outcome=[0, .5, .95])
    assert len(result["sensitivity"]) == 18
    zero = result["sensitivity"].iloc[0]
    base = result["coefficients"].loc["d"]
    assert zero.bias_magnitude == 0 and zero.estimate == base.estimate
    assert_allclose(zero.std_error, base.std_error * np.sqrt(base.df / (base.df - 1)))
    rows = result["sensitivity"]
    assert rows[(rows.r2_treatment == .95) & (rows.r2_outcome == .95) & (rows.direction == "reduce")].iloc[0].estimate < 0


def test_no_controls_tied_binary_and_null_effect_are_admitted():
    data = pd.DataFrame(dict(d=[0, 0, 0, 0, 1, 1, 1, 1], y=[-1, 1, -2, 2, -2, 2, -1, 1]))
    result = ovb_sensitivity(data, "y", "d", [], r2_treatment=[0, .1], r2_outcome=[0, .1])
    assert result.attrs["state"]["terms"] == ["_cons", "d"]
    assert abs(result["coefficients"].loc["d", "estimate"]) < 1e-14
    assert result["robustness"].iloc[0].r2_treatment < 1e-14
    assert result.attrs["state"]["df"] == 6


def test_minimum_positive_augmented_df_and_t_interval():
    data = pd.DataFrame(dict(d=[0, 0, 1, 1], y=[-1., 1., 1., 4.]))
    result = ovb_sensitivity(data, "y", "d", [], r2_treatment=[0], r2_outcome=[0])
    row = result["sensitivity"].iloc[0]
    assert row.df == 1
    assert_allclose(row.ci_upper - row.estimate, t.ppf(.975, 1) * row.std_error, rtol=1e-11)
    with pytest.raises(AnalysisError) as error:
        ovb_sensitivity(data.iloc[:3], "y", "d", [])
    assert error.value.code == "insufficient_observations"


@pytest.mark.parametrize("feature_scale,outcome_scale", [(1e-150, 1), (1e149, 1), (1, 1e149), (1, 1e-150)])
def test_huge_and_tiny_units_keep_original_covariance(feature_scale, outcome_scale):
    data = fixture_data()
    reference = analyze(data)
    data[["d", "x1", "x2"]] *= feature_scale
    data.y *= outcome_scale
    result = analyze(data)
    transform = np.diag([outcome_scale, *([outcome_scale / feature_scale] * 3)])
    assert_allclose(result["coefficients"].estimate, transform @ reference["coefficients"].estimate, rtol=3e-13, atol=0)
    assert_allclose(result["covariance"], transform @ reference["covariance"].to_numpy() @ transform, rtol=2e-12, atol=0)
    assert_allclose(result["robustness"].r2_treatment, reference["robustness"].r2_treatment, rtol=3e-13)


def test_nonzero_bias_below_coefficient_resolution_is_refused():
    with pytest.raises(AnalysisError) as error:
        analyze(r2_treatment=[1e-200], r2_outcome=[1e-200])
    assert error.value.code == "numerical_failure"
    assert "bias adjustment" in str(error.value)


def test_zero_estimate_tiny_strengths_preserve_bias_without_product_underflow():
    from openecon.econometrics.causal_design.ovb import _scenario

    # At an exactly zero target, the representable bias survives even though
    # multiplying the two strengths before taking the root would become zero.
    row = _scenario(0.0, 1.0, 10, 1e-200, 1e-200, "reduce", 1, t.ppf(.975, 9))
    assert_allclose(row[3], np.sqrt(10) * 1e-200, rtol=3e-15, atol=0)
    assert row[5] == -row[3] and row[3] > 0


def test_high_signal_point_tipping_boundary_is_refused():
    # Noise is above the QR residual-variance rounding floor, but the
    # point-estimate tipping partial R2 is too close to one to represent.
    n = 20000
    data = pd.DataFrame(dict(
        d=np.tile([-1., -1., 1., 1.], n // 4),
        y=np.tile([-1e149 - 5e134, -1e149 + 5e134, 1e149 - 5e134, 1e149 + 5e134], n // 4),
    ))
    with pytest.raises(AnalysisError) as error:
        ovb_sensitivity(data, "y", "d", [], r2_treatment=[0], r2_outcome=[0])
    assert error.value.code == "numerical_failure"
    assert "tipping value" in str(error.value)


def test_positive_confidence_radius_lost_at_large_coefficient_scale_is_refused():
    from openecon.econometrics.causal_design.ovb import _infer

    with pytest.raises(AnalysisError) as error:
        _infer(1e150, 1.0, 30, t.ppf(.975, 30))
    assert error.value.code == "numerical_failure"
    assert "confidence radius" in str(error.value)


def test_nonrepresentable_original_variance_is_refused():
    data = fixture_data()
    data.y *= 1e-180
    with pytest.raises(AnalysisError) as error:
        analyze(data)
    assert error.value.code == "numerical_failure"


def test_underflowed_residual_variance_is_refused_even_when_model_covariance_is_positive():
    # A large centering transform can make coefficient covariance representable
    # even when RSS/df itself underflows. It must not persist as false zero noise.
    data = pd.DataFrame(dict(
        d=1 + np.tile([-1e-7, -1e-7, 1e-7, 1e-7], 20),
        y=np.tile([-1e-162, 1e-162, -1e-162, 1e-162], 20),
    ))
    with pytest.raises(AnalysisError) as error:
        ovb_sensitivity(data, "y", "d", [], r2_treatment=[0], r2_outcome=[0])
    assert error.value.code == "numerical_failure"
    assert "residual variance" in str(error.value)


def test_missing_positions_duplicate_labels_and_complete_artifact_restore(tmp_path):
    data = fixture_data()
    data.index = ["same"] * len(data)
    data.loc[:, "unused"] = np.nan
    data.iloc[3, data.columns.get_loc("x2")] = np.nan
    data.iloc[10, data.columns.get_loc("y")] = np.nan
    with pytest.raises(AnalysisError) as error:
        analyze(data)
    assert error.value.code == "missing_values"
    result = analyze(data, missing="drop", r2_treatment=[0, .1, .5], r2_outcome=[0, .1])
    assert result.attrs["positions"] == [i for i in range(80) if i not in (3, 10)]
    assert result.attrs["unit_labels"] == ["same"] * 78
    assert result.attrs["n_missing"] == 2
    path = tmp_path / "ovb.json"
    payload = causal_design_save(result, path)
    restored = causal_design_load(path)
    assert json.loads(path.read_text()) == payload
    assert restored.attrs == result.attrs
    assert list(restored) == list(result)
    for key in result:
        pd.testing.assert_frame_equal(restored[key], result[key], check_exact=True)
    assert causal_design_save(restored) == payload
    changed = copy.deepcopy(payload)
    changed["payload"]["attrs"]["state"]["covariance"][0][1] += .001
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(changed)


def test_native_path_private_rng_and_global_default_device():
    data = fixture_data()
    initial = torch.random.get_rng_state().clone()
    reference = analyze(data)
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = analyze(data)
    finally:
        torch.set_default_device(previous)
    assert torch.equal(initial, torch.random.get_rng_state())
    assert causal_design_save(actual) == causal_design_save(reference)


@pytest.mark.parametrize("options", [
    {"r2_treatment": []}, {"r2_outcome": [1]}, {"r2_treatment": [-.1]},
    {"r2_treatment": [.2, .1]}, {"r2_outcome": [.1, .1]},
    {"r2_outcome": [True]}, {"r2_outcome": [np.nan]}, {"r2_treatment": ".1"},
    {"r2_outcome": list(np.linspace(0, .9, 65))}, {"direction": "other"},
    {"level": 1}, {"level": 0}, {"level": True}, {"missing": "other"},
    {"device": "cuda"}, {"weights": "w"}, {"max_work": 0},
])
def test_invalid_options_are_refused(options):
    with pytest.raises(AnalysisError):
        analyze(**options)


@pytest.mark.parametrize("y,d,controls", [
    ("y", "y", []), ("y", "d", ["d"]), ("y", "d", ["x1", "x1"]),
    ("y", "d", "x1"), ("y", "d", None), ("y", "d", ["absent"]),
    ("y", "_cons", []),
])
def test_invalid_roles_are_refused(y, d, controls):
    with pytest.raises(AnalysisError):
        ovb_sensitivity(fixture_data(), y, d, controls)


@pytest.mark.parametrize("kind", ["collinear", "constant", "perfect_fit", "text", "boolean", "infinite"])
def test_invalid_designs_outcomes_and_roles_are_refused(kind):
    data = fixture_data()
    if kind == "collinear":
        data.x2 = 3 * data.x1 + 2 * data.d
    elif kind == "constant":
        data.d = 1
    elif kind == "perfect_fit":
        data.y = 2 + 3 * data.d - data.x1
    elif kind == "text":
        data.x1 = data.x1.astype(str)
    elif kind == "boolean":
        data.d = data.d > 0
    else:
        data.iloc[0, data.columns.get_loc("y")] = np.inf
    with pytest.raises(AnalysisError):
        analyze(data)


def test_work_and_workspace_refuse_before_qr(monkeypatch):
    import openecon.econometrics.causal_design.ovb as module

    monkeypatch.setattr(module, "least_squares", lambda *args, **kwargs: pytest.fail("QR should not execute"))
    with pytest.raises(AnalysisError) as error:
        analyze(max_work=100)
    assert error.value.code == "work_budget_exceeded"
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        analyze(fixture_data(n=2000))
    assert error.value.code == "workspace_limit"


def test_dataset_is_not_implicitly_collected():
    from openecon.dataset import Dataset

    data = object.__new__(Dataset)
    with pytest.raises(AnalysisError) as error:
        analyze(data)
    assert error.value.code == "unsupported_dataset"
