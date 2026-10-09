"""Independent nested OLS, FWL geometry and box minimization OVB oracles."""

import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import t
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.ovb import ovb_sensitivity
from openecon.econometrics.causal_design.ovb_postest import ovb_benchmark, ovb_robustness
from openecon.resources import use_workspace_budget


def fixture_data(n=90, effect=1.5, seed=871):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    d = 0.12 * x[:, 0] - 0.15 * x[:, 1] + rng.normal(size=n)
    y = effect * d + 0.25 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x[:, 0], x2=x[:, 1], x3=x[:, 2]))


def fit_base(data=None, **options):
    return ovb_sensitivity(
        fixture_data() if data is None else data, "y", "d", ["x1", "x2", "x3"], **options
    )


def benchmark(base=None, **options):
    return ovb_benchmark(
        fit_base() if base is None else base,
        groups=options.pop("groups", {"one": ["x1"], "joint": ["x1", "x2"]}),
        assumption=options.pop("assumption", "residualized_benchmark"),
        **options,
    )


def ols(x, y):
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ b
    df = len(y) - x.shape[1]
    cov = np.linalg.inv(x.T @ x) * (residual @ residual / df)
    return b, cov, residual, df


def residual(x, y):
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial_strength(y, z, controls):
    a, b = residual(controls, y), residual(controls, z)
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


def formula(a, b, kd, ky):
    d = kd * a / (1 - a)
    u = kd * a * a / ((1 - kd * a) * (1 - a))
    y = (np.sqrt(ky) + np.sqrt(u)) ** 2 / (1 - u) * b / (1 - b)
    return d, u, y


def test_observed_single_and_joint_groups_use_exact_same_sample_nested_ols():
    data = fixture_data()
    base = fit_base(data)
    before = causal_design_save(base)
    result = benchmark(base, kd=[0, 0.5, 1], ky=[0, 0.5, 1])
    controls = np.column_stack([np.ones(len(data)), data[["x1", "x2", "x3"]]])
    full = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2", "x3"]]])
    _, _, rd_full, _ = ols(controls, data.d)
    b, cov, ry_full, df = ols(full, data.y)
    for row in result["benchmarks"].itertuples():
        group = ["x1"] if row.group == "one" else ["x1", "x2"]
        others = [name for name in ["x1", "x2", "x3"] if name not in group]
        reduced_d = np.column_stack([np.ones(len(data)), data[others]])
        reduced_y = np.column_stack([np.ones(len(data)), data[["d", *others]]])
        _, _, rd_red, _ = ols(reduced_d, data.d)
        _, _, ry_red, _ = ols(reduced_y, data.y)
        a = 1 - rd_full @ rd_full / (rd_red @ rd_red)
        by = 1 - ry_full @ ry_full / (ry_red @ ry_red)
        assert_allclose(
            [row.r2_treatment_observed, row.r2_outcome_observed], [a, by], rtol=2e-12, atol=1e-15
        )
        for bounds in result["bounds"].loc[result["bounds"].group == row.group].itertuples():
            d, u, y = formula(a, by, bounds.kd, bounds.ky)
            bias = np.sqrt(cov[1, 1] * df * d * y / (1 - d))
            assert_allclose(
                [
                    bounds.r2_treatment_bound,
                    bounds.auxiliary_r2,
                    bounds.r2_outcome_bound,
                    bounds.bias_bound,
                    bounds.estimate_lower,
                    bounds.estimate_upper,
                ],
                [d, u, y, bias, b[1] - bias, b[1] + bias],
                rtol=2e-11,
                atol=1e-14,
            )
    assert len(result["bounds"]) == 18
    assert causal_design_save(base) == before
    assert result.attrs["state"]["base_artifact"] == before
    assert result.attrs["state"]["confidence_interval"] is None
    pd.testing.assert_frame_equal(result["covariance"], base["covariance"], check_exact=True)


@pytest.mark.parametrize("group", [["x1"], ["x1", "x2"]])
def test_formal_bound_contains_actual_residualized_confounder_under_declared_ratios(group):
    data = fixture_data()
    all_controls = np.column_stack([np.ones(len(data)), data[["x1", "x2", "x3"]]])
    rng = np.random.default_rng(881)
    z = residual(all_controls, rng.normal(size=len(data)))
    others = [name for name in ["x1", "x2", "x3"] if name not in group]
    reduced_controls = np.column_stack([np.ones(len(data)), data[others]])
    reduced_with_d = np.column_stack([reduced_controls, data.d])
    original_design = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2", "x3"]]])
    base = fit_base(data)
    observed = benchmark(base, groups={"g": group})["benchmarks"].iloc[0]
    kd = partial_strength(data.d.to_numpy(), z, reduced_controls) / observed.r2_treatment_observed
    ky = partial_strength(data.y.to_numpy(), z, reduced_with_d) / observed.r2_outcome_observed
    result = benchmark(base, groups={"g": group}, kd=[kd], ky=[ky])
    row = result["bounds"].iloc[0]
    actual_rd = partial_strength(data.d.to_numpy(), z, all_controls)
    actual_ry = partial_strength(data.y.to_numpy(), z, original_design)
    full, _, _, _ = ols(np.column_stack([original_design, z]), data.y)
    assert_allclose(row.r2_treatment_bound, actual_rd, rtol=5e-12)
    assert actual_ry <= row.r2_outcome_bound + 1e-13
    assert row.estimate_lower - 1e-13 <= full[1] <= row.estimate_upper + 1e-13
    assert_allclose(all_controls.T @ z, 0, atol=2e-14)


def author_rv(f, c):
    if f <= c:
        return 0.0, "already_nonrejected"
    h = f - c
    if c > 0 and f > 1 / c:
        return (f * f - c * c) / (1 + f * f), "interior"
    return (np.sqrt(h**4 + 4 * h * h) - h * h) / 2, "diagonal"


def box_minimum(f, rho):
    def objective(ry):
        return (f * np.sqrt(1 - rho) - np.sqrt(ry * rho)) / np.sqrt(1 - ry)

    if rho == 0:
        return f
    optimized = minimize_scalar(
        objective, bounds=(0, rho), method="bounded", options={"xatol": 1e-14}
    )
    return min(objective(0), objective(rho), optimized.fun)


@pytest.mark.parametrize("effect", [0.1, 1.5, 12.0, -1.5, -12.0])
def test_max_coordinate_rv_matches_author_piecewise_and_independent_box_optimization(effect):
    base = fit_base(fixture_data(effect=effect))
    result = ovb_robustness(base, q=[0, 0.5, 1, 1.5], alpha=[0.01, 0.05, 1])
    t0 = abs(base["coefficients"].loc["d", "t"])
    df = base.attrs["state"]["df"]
    for row in result["robustness"].itertuples():
        f = row.q * t0 / np.sqrt(df)
        c = abs(t.ppf(row.alpha / 2, df - 1)) / np.sqrt(df - 1)
        expected, regime = author_rv(f, c)
        assert_allclose(row.robustness_value, expected, rtol=3e-11, atol=2e-13)
        assert row.regime == regime
        if f <= c:
            assert row.robustness_value == 0
        else:
            root = brentq(lambda rho: box_minimum(f, rho) - c, 0, 1 - 1e-12, xtol=1e-12)
            assert_allclose(row.robustness_value, root, rtol=2e-9, atol=1e-11)
            assert_allclose(box_minimum(f, row.robustness_value), c, rtol=1e-7, atol=1e-9)
        assert row.witness_r2_treatment <= row.robustness_value
        assert row.witness_r2_outcome <= row.robustness_value
        if row.regime == "interior":
            assert row.witness_r2_outcome < row.robustness_value
            assert row.robustness_value < row.diagonal_value
    assert result.attrs["state"]["robustness_covariance"] is None


@pytest.mark.parametrize("effect", [1.5, -1.5, 12.0, -12.0])
def test_threshold_witness_constructs_actual_scalar_z_with_augmented_ols_tipping(effect):
    data = fixture_data(effect=effect)
    base = fit_base(data)
    result = ovb_robustness(base, q=[0.5, 1], alpha=[0.05, 1])
    control = np.column_stack([np.ones(len(data)), data[["x1", "x2", "x3"]]])
    design = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2", "x3"]]])
    du = residual(control, data.d.to_numpy())
    du /= np.linalg.norm(du)
    yu = residual(design, data.y.to_numpy())
    yu /= np.linalg.norm(yu)
    v = residual(np.column_stack([design, yu]), np.random.default_rng(17).normal(size=len(data)))
    v /= np.linalg.norm(v)
    sign = np.sign(base["coefficients"].loc["d", "estimate"])
    for row in result["robustness"].itertuples():
        rd, ry = row.witness_r2_treatment, row.witness_r2_outcome
        z = np.sqrt(rd) * du + np.sqrt(1 - rd) * (sign * np.sqrt(ry) * yu + np.sqrt(1 - ry) * v)
        full, cov, _, df = ols(np.column_stack([design, z]), data.y)
        se = np.sqrt(cov[1, 1])
        stat = (full[1] - row.reference_estimate) / se
        assert_allclose(partial_strength(data.d.to_numpy(), z, control), rd, rtol=1e-10, atol=1e-13)
        assert_allclose(partial_strength(data.y.to_numpy(), z, design), ry, rtol=1e-10, atol=1e-13)
        assert_allclose(
            [full[1], se, df, stat],
            [row.adjusted_estimate, row.adjusted_std_error, row.adjusted_df, row.t_to_reference],
            rtol=2e-10,
            atol=1e-11,
        )
        assert_allclose(abs(stat), row.critical_value, rtol=2e-10, atol=1e-11)


def test_point_tipping_at_q1_matches_original_and_restored_base_exactly():
    base = fit_base()
    result = ovb_robustness(base, q=[1], alpha=[1])
    assert (
        result["robustness"].iloc[0].robustness_value
        == base.attrs["state"]["equal_strength_point_estimate_robustness_value"]
    )
    restored = causal_design_load(causal_design_save(base))
    assert causal_design_save(result) == causal_design_save(
        ovb_robustness(restored, q=[1], alpha=[1])
    )
    assert causal_design_save(benchmark(base)) == causal_design_save(benchmark(restored))


@pytest.mark.parametrize("function", [benchmark, ovb_robustness])
def test_complete_original_missing_positions_labels_typed_tables_roundtrip(function, tmp_path):
    data = fixture_data()
    data.index = ["same"] * len(data)
    data.iloc[5, data.columns.get_loc("x2")] = np.nan
    data.iloc[13, data.columns.get_loc("y")] = np.nan
    base = fit_base(data, missing="drop")
    result = function(base)
    assert result.attrs["positions"] == base.attrs["positions"]
    assert result.attrs["unit_labels"] == base.attrs["unit_labels"]
    assert result.attrs["sample_sha256"] == base.attrs["sample_sha256"]
    assert result.attrs["n_missing"] == 2
    assert result.attrs["state"]["base_artifact"] == causal_design_save(base)
    path = tmp_path / "postest.json"
    artifact = causal_design_save(result, path)
    restored = causal_design_load(path)
    assert restored.attrs == result.attrs
    assert causal_design_save(restored) == artifact == json.loads(path.read_text())
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)
        assert restored[name].to_latex()


@pytest.mark.parametrize("function", [benchmark, ovb_robustness])
@pytest.mark.parametrize("kind", ["table", "sample", "state"])
def test_mutated_original_results_are_refused(function, kind):
    base = fit_base()
    if kind == "table":
        base["coefficients"].iloc[1, 0] += 0.2
    elif kind == "sample":
        base.attrs["positions"][0] = 100
    else:
        base.attrs["state"]["coefficients"][1] += 0.2
    with pytest.raises(AnalysisError) as error:
        function(base)
    assert error.value.code == "invalid_result"


@pytest.mark.parametrize("function", [benchmark, ovb_robustness])
@pytest.mark.parametrize("base", [None, {}, pd.DataFrame({"x": [1, 2]})])
def test_only_original_ovb_result_is_supported(function, base):
    # benchmark(None) is our fixture wrapper, so call the public function.
    call = (
        (lambda: ovb_benchmark(base, groups={"x": ["x"]}, assumption="residualized_benchmark"))
        if function == benchmark
        else lambda: ovb_robustness(base)
    )
    with pytest.raises(AnalysisError) as error:
        call()
    assert error.value.code == "invalid_result"


@pytest.mark.parametrize(
    "options",
    [
        {"groups": {}},
        {"groups": {"g": []}},
        {"groups": {"g": "x1"}},
        {"groups": {"g": ["x1", "x1"]}},
        {"groups": {"g": ["d"]}},
        {"groups": {"g": ["absent"]}},
        {"assumption": None},
        {"assumption": "observed_proof"},
        {"kd": []},
        {"kd": [-0.1]},
        {"kd": [True]},
        {"kd": [1, 1]},
        {"ky": [np.inf]},
        {"ky": [1, 0.5]},
        {"kd": [1e8]},
        {"ky": [1e8]},
    ],
)
def test_benchmark_invalid_or_uninformative_scopes_are_refused(options):
    with pytest.raises(AnalysisError):
        benchmark(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"q": []},
        {"q": [-1]},
        {"q": [True]},
        {"q": [1, 1]},
        {"q": [np.inf]},
        {"alpha": [0]},
        {"alpha": [1.01]},
        {"alpha": [True]},
        {"alpha": [1, 0.05]},
        {"q": list(np.arange(65))},
        {"alpha": [5e-324]},
    ],
)
def test_robustness_invalid_or_nonrepresentable_queries_are_refused(options):
    with pytest.raises(AnalysisError):
        ovb_robustness(fit_base(), **options)


@pytest.mark.parametrize("function", [benchmark, ovb_robustness])
@pytest.mark.parametrize(
    "options", [{"device": "cuda"}, {"device": "mps"}, {"weights": "w"}, {"max_work": 0}]
)
def test_unsupported_execution_contracts(function, options):
    with pytest.raises(AnalysisError):
        function(fit_base(), **options)


def test_reconstructed_bias_and_threshold_that_disappear_are_refused():
    with pytest.raises(AnalysisError) as error:
        benchmark(kd=[1e-200], ky=[1e-200])
    assert error.value.code == "numerical_failure"
    with pytest.raises(AnalysisError) as error:
        ovb_robustness(fit_base(), q=[1e-200])
    assert error.value.code == "numerical_failure"
    with pytest.raises(AnalysisError) as error:
        ovb_robustness(fit_base(), q=[1e150])
    assert error.value.code == "numerical_failure"


def test_resource_limits_admit_complete_original_and_queries_before_qr(monkeypatch):
    import openecon.econometrics.causal_design.ovb_postest as module

    base = fit_base()
    monkeypatch.setattr(module, "least_squares", lambda *a, **k: pytest.fail("QR must not execute"))
    with pytest.raises(AnalysisError) as error:
        benchmark(base, max_work=4000)
    assert error.value.code == "work_budget_exceeded"
    with pytest.raises(AnalysisError) as error:
        ovb_robustness(base, max_work=1)
    assert error.value.code == "work_budget_exceeded"
    large = fit_base(fixture_data(n=2000))
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        ovb_robustness(large)
    assert error.value.code == "workspace_limit"


def test_native_cpu_global_default_device_private_rng_and_base_immutability():
    base = fit_base()
    before = causal_design_save(base)
    initial = torch.random.get_rng_state().clone()
    expected = [causal_design_save(benchmark(base)), causal_design_save(ovb_robustness(base))]
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        actual = [causal_design_save(benchmark(base)), causal_design_save(ovb_robustness(base))]
    finally:
        torch.set_default_device(previous)
    assert actual == expected
    assert torch.equal(initial, torch.random.get_rng_state())
    assert causal_design_save(base) == before
