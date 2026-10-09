"""Independent inequality formulas, aligned identities and persistence checks.

Seeded coverage screens are empirical screens, separate from the mathematical
inequalities and their declared iid/fixed-support assumptions.
"""

import json
import math
import random

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest import bounded_joint as native
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget

METHODS = [native.hoeffding_mean_ci, native.empirical_bernstein_mean_ci]


def example():
    return pd.DataFrame({"x": [.1, .3, .7, .2, .9, .4, .6, .8],
                         "y": [-1.8, -.2, .3, 1.2, 2.7, .5, 1.9, -1.1]})


def run(method, data=None, columns=None, **options):
    return method(example() if data is None else data,
                  ["x", "y"] if columns is None else columns,
                  bounds=options.pop("bounds", [(0., 1.), (-2., 3.)]),
                  sampling_model=options.pop("sampling_model", "iid_bounded"), **options)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("alpha", [.01, .05, .25])
def test_independent_scalar_formula_and_pairwise_variance_oracle(method, alpha):
    data = example()
    result = run(method, data, alpha=alpha)
    m, n = 2, len(data)
    bernstein = method is native.empirical_bernstein_mean_ci
    log_factor = math.log((4 if bernstein else 2) * m / alpha)
    for j, (name, (a, b)) in enumerate(zip(["x", "y"], [(0., 1.), (-2., 3.)], strict=True)):
        values = list(data[name])
        mean = math.fsum(values) / n
        # Independent O(n²) U-stat definition, not the implementation kernel.
        variance = math.fsum((values[i] - values[k])**2
                             for i in range(n) for k in range(i + 1, n)) / (n * (n - 1))
        radius = (b - a) * math.sqrt(log_factor / (2 * n))
        if bernstein:
            radius = math.sqrt(2 * variance * log_factor / n) \
                + 7 * (b - a) * log_factor / (3 * (n - 1))
        row = result["intervals"].iloc[j]
        assert row.estimate == pytest.approx(mean, abs=1e-14)
        assert row.radius_unclipped == pytest.approx(radius, rel=2e-15)
        assert row.ci_low == pytest.approx(max(a, mean - radius), abs=1e-14)
        assert row.ci_high == pytest.approx(min(b, mean + radius), abs=1e-14)
        assert result["diagnostics"].sample_variance[j] == pytest.approx(variance, rel=2e-15)
    assert result.attrs["log_factor"] == pytest.approx(log_factor)
    assert result.attrs["one_sided_failure_budget"] == alpha / (2 * m)
    assert result.attrs["tail_count"] == 2 * m
    assert "exact nominal" in result.attrs["guarantee"]


@pytest.mark.parametrize("method", METHODS)
def test_translation_scale_and_family_permutation(method):
    original = example()
    source = run(method, original)
    scale, shift = np.array([3., .4]), np.array([5., -8.])
    moved = original * scale + shift
    target = run(method, moved, bounds=[(5., 8.), (-8.8, -6.8)])
    for column in ["estimate", "ci_low", "ci_high"]:
        np.testing.assert_allclose(target["intervals"][column],
                                   source["intervals"][column] * scale + shift, atol=5e-14)
    np.testing.assert_allclose(target["intervals"].radius_unclipped,
                               source["intervals"].radius_unclipped * scale, atol=2e-14)
    np.testing.assert_allclose(target["diagnostics"].sample_variance,
                               source["diagnostics"].sample_variance * scale**2, atol=2e-14)
    permuted = run(method, original, ["y", "x"], bounds={"x": (0., 1.), "y": (-2., 3.)})
    pd.testing.assert_frame_equal(source["intervals"].iloc[::-1].reset_index(drop=True),
                                  permuted["intervals"])
    assert permuted.attrs["family_members"] == ["y", "x"]


@pytest.mark.parametrize("method", METHODS)
def test_negative_scale_reflection_and_perfect_cross_coordinate_dependence(method):
    rng = np.random.default_rng(90031)
    x = rng.uniform(0, 1, 160)
    original = run(method, pd.DataFrame({"x": x}), ["x"], bounds=[(0, 1)])
    reflected = run(method, pd.DataFrame({"x": 3 - 2 * x}), ["x"], bounds=[(1, 3)])
    assert reflected["intervals"].ci_low[0] == pytest.approx(3 - 2 * original["intervals"].ci_high[0])
    assert reflected["intervals"].ci_high[0] == pytest.approx(3 - 2 * original["intervals"].ci_low[0])
    joint = run(method, pd.DataFrame({"x": x, "y": x}), bounds=[(0, 1), (0, 1)])
    np.testing.assert_array_equal(joint["intervals"].ci_low, [joint["intervals"].ci_low[0]] * 2)
    assert joint["intervals"].radius_unclipped[0] > original["intervals"].radius_unclipped[0]
    assert "arbitrary cross-coordinate" in joint.attrs["assumptions"]


@pytest.mark.parametrize("method", METHODS)
def test_constant_sample_radius_and_known_support_intersection(method):
    output = run(method, pd.DataFrame({"x": [.1] * 80}), ["x"], bounds=[(.1, 1.1)])
    row = output["intervals"].iloc[0]
    assert row.estimate == .1 and row.ci_low == .1
    assert .1 < row.ci_high <= 1.1 and row.radius_unclipped > 0
    assert output["diagnostics"].sample_variance[0] == 0
    if method is native.empirical_bernstein_mean_ci:
        assert row.radius_unclipped == pytest.approx(7 * math.log(4 / .05) / (3 * 79))


@pytest.mark.parametrize("method", METHODS)
def test_interior_constant_variance_exact_zero_and_narrower_bounds_are_not_inferred(method):
    data = pd.DataFrame({"x": [.3] * 83})
    narrow = run(method, data, ["x"], bounds=[(.1, 1.1)])
    broad = run(method, data, ["x"], bounds=[(-5, 5)])
    assert narrow["diagnostics"].sample_variance[0] == 0
    assert broad["diagnostics"].sample_variance[0] == 0
    assert broad["intervals"].radius_unclipped[0] \
        == pytest.approx(narrow["intervals"].radius_unclipped[0] * 10)
    assert narrow.attrs["bounds"] == [[.1, 1.1]] and broad.attrs["bounds"] == [[-5, 5]]


def test_hoeffding_one_row_and_bernstein_two_row_admission():
    output = run(native.hoeffding_mean_ci, pd.DataFrame({"x": [.4]}), ["x"], bounds=[(0, 1)])
    assert output.attrs["nobs"] == 1
    assert output["intervals"].ci_low[0] == 0 and output["intervals"].ci_high[0] == 1
    assert "Undefined" in output.attrs["variance_convention"]
    assert output["diagnostics"].sample_variance[0] is None
    with pytest.raises(AnalysisError) as e:
        run(native.empirical_bernstein_mean_ci, pd.DataFrame({"x": [.4]}), ["x"], bounds=[(0, 1)])
    assert e.value.code == "insufficient_observations"
    output = run(native.empirical_bernstein_mean_ci,
                 pd.DataFrame({"x": [0., 1.]}), ["x"], bounds=[(0, 1)])
    assert output["intervals"].ci_low[0] == 0 and output["intervals"].ci_high[0] == 1


@pytest.mark.parametrize("method", METHODS)
def test_complete_case_alignment_typed_labels_and_unchanged_caller(method):
    data = example()
    data.index = pd.Index([1, "1", ("a", 2), None, "duplicate", "duplicate", True, .5],
                          dtype=object, tupleize_cols=False)
    data.loc["1", "x"] = np.nan
    data.iloc[6, 1] = np.nan
    before = data.copy(deep=True)
    with pytest.raises(AnalysisError) as e:
        run(method, data)
    assert e.value.code == "missing_values"
    output = run(method, data, missing="drop")
    positions = [0, 2, 3, 4, 5, 7]
    assert output.attrs["sample_positions"] == positions
    assert [decode(v) for v in output.attrs["sample_labels_encoded"]] == [data.index[i] for i in positions]
    assert output.attrs["nobs"] == 6 and output.attrs["dropped_rows"] == 2
    assert output.attrs["sample_index_hash"] != output.attrs["raw_index_hash"]
    assert "retained rows iid" in output.attrs["missing_sample_contract"]
    pd.testing.assert_frame_equal(data, before)
    manual = run(method, data.iloc[positions])
    pd.testing.assert_frame_equal(output["intervals"], manual["intervals"])
    assert output.attrs["sample_hash"] == manual.attrs["sample_hash"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), -2.1, 3.1])
def test_invalid_available_values_not_hidden_by_another_columns_missingness(method, bad):
    data = example()
    data.loc[0, "x"], data.loc[0, "y"] = np.nan, bad
    with pytest.raises(AnalysisError) as e:
        run(method, data, missing="drop")
    assert e.value.code == ("nonfinite_values" if math.isinf(bad) else "outside_support")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype,values", [("object", [".2", ".3"]),
                                         ("bool", [True, False]),
                                         ("complex128", [.2 + 0j, .3 + 0j])])
def test_reject_strings_booleans_and_complex(method, dtype, values):
    with pytest.raises(AnalysisError) as e:
        run(method, pd.DataFrame({"x": pd.Series(values, dtype=dtype)}), ["x"], bounds=[(0, 1)])
    assert e.value.code == "invalid_values"


@pytest.mark.parametrize("method", METHODS)
def test_nullable_numeric_complete_sample(method):
    data = pd.DataFrame({"x": pd.Series([1, 2, pd.NA, 3], dtype="Int64")})
    output = run(method, data, ["x"], bounds=[(0, 4)], missing="drop")
    assert output.attrs["sample_positions"] == [0, 1, 3]
    assert output["intervals"].estimate[0] == 2


@pytest.mark.parametrize("bad", [None, (), (0, 1), [(0, 1), (-2,)],
                                  [(1, 0), (-2, 3)], [(0, 0), (-2, 3)],
                                  [(False, 1), (-2, 3)], [("0", 1), (-2, 3)],
                                  [(0, math.inf), (-2, 3)], {"x": (0, 1)},
                                  {"x": (0, 1), "y": (-2, 3), "extra": (0, 1)}])
def test_bounds_must_be_prespecified_resolved_exact_family(bad):
    with pytest.raises(AnalysisError):
        run(native.hoeffding_mean_ci, bounds=bad)


@pytest.mark.parametrize("bad", [None, "x", [], ["x", "x"], ["x", 1], ["absent"], [""]])
def test_unique_resident_column_contract(bad):
    with pytest.raises(AnalysisError) as e:
        native.hoeffding_mean_ci(example(), bad, bounds=[(0, 1)], sampling_model="iid_bounded")
    assert e.value.code == "invalid_columns"
    with pytest.raises(AnalysisError) as e:
        run(native.hoeffding_mean_ci, pd.DataFrame([[1, 2]], columns=["x", "x"]), ["x"], bounds=[(0, 3)])
    assert e.value.code == "invalid_columns"


@pytest.mark.parametrize("bad", [True, 0, -1, 1, 1e-13, math.nan, math.inf, "0.05", [0.05]])
def test_alpha_contract(bad):
    with pytest.raises(AnalysisError) as e:
        run(native.empirical_bernstein_mean_ci, alpha=bad)
    assert e.value.code == "invalid_option"


@pytest.mark.parametrize("bad", [None, "iid_normal", "iid", True, ["iid_bounded"]])
def test_sampling_model_declaration_required(bad):
    with pytest.raises(AnalysisError) as e:
        run(native.hoeffding_mean_ci, sampling_model=bad)
    assert e.value.code == "unsupported_sampling_model"


@pytest.mark.parametrize("method", METHODS)
def test_unsupported_routes_and_missing_policy(method):
    with pytest.raises(AnalysisError) as e:
        run(method, {"x": [.2, .3]}, ["x"], bounds=[(0, 1)])
    assert e.value.code == "unsupported_domain"
    with pytest.raises(TypeError):
        run(method, weights="w")
    with pytest.raises(TypeError):
        run(method, device="cuda")
    for missing in [None, False, "pairwise", ["drop"]]:
        with pytest.raises(AnalysisError) as e:
            run(method, missing=missing)
        assert e.value.code == "invalid_missing"
    with pytest.raises(AnalysisError) as e:
        run(method, example().iloc[:0])
    assert e.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as e:
        run(method, pd.DataFrame({"x": [math.nan, math.nan]}), ["x"], bounds=[(0, 1)], missing="drop")
    assert e.value.code == "insufficient_observations"


@pytest.mark.parametrize("method", METHODS)
def test_dimension_guards_precede_scans_and_numeric_allocation(monkeypatch, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dimension guard must precede any identity/numeric scan")
    monkeypatch.setattr(native, "_identity_scan", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_numpy", forbidden)
    with pytest.raises(AnalysisError) as e:
        run(method, pd.DataFrame({"x": np.zeros(native.MAX_ROWS + 1)}), ["x"], bounds=[(0, 1)])
    assert e.value.code == "work_budget"
    with pytest.raises(AnalysisError) as e:
        data = pd.DataFrame(np.zeros((3, native.MAX_COLUMNS + 1)))
        data.columns = [f"x{i}" for i in range(native.MAX_COLUMNS + 1)]
        run(method, data, list(data.columns), bounds=[(0, 1)] * len(data.columns))
    assert e.value.code == "work_budget"


@pytest.mark.parametrize("method", METHODS)
def test_workspace_guard_precedes_numeric_copy_and_reports_plan(monkeypatch, method):
    def forbidden(*args, **kwargs):
        raise AssertionError("Workspace guard must precede numerical copy")
    monkeypatch.setattr(pd.DataFrame, "to_numpy", forbidden)
    data = pd.DataFrame(np.full((4000, 8), .2), columns=[f"x{i}" for i in range(8)])
    with use_workspace_budget(1), pytest.raises(AnalysisError) as e:
        run(method, data, list(data.columns), bounds=[(0, 1)] * 8)
    assert e.value.code == "workspace_limit"
    assert e.value.resource_plan["estimated_workspace_bytes"] > 1024**2


def test_work_and_identity_budgets_precede_numeric_copy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Admission must precede numerical copy")
    monkeypatch.setattr(pd.DataFrame, "to_numpy", forbidden)
    monkeypatch.setattr(native, "MAX_WORK", 1)
    with pytest.raises(AnalysisError) as e:
        run(native.hoeffding_mean_ci)
    assert e.value.code == "work_budget"
    monkeypatch.setattr(native, "MAX_WORK", 100_000_000)
    data = example().rename(index={0: "x" * 5000})
    with pytest.raises(AnalysisError) as e:
        run(native.hoeffding_mean_ci, data)
    assert e.value.code == "work_budget"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bounds,values", [((-1e308, 1e308), [0., 1.]),
                                           ((0., 1e200), [0., 1e200]),
                                           ((0., 1e-200), [0., 1e-200])])
def test_unresolved_overflow_and_variance_underflow_explicit(method, bounds, values):
    with pytest.raises(AnalysisError) as e:
        run(method, pd.DataFrame({"x": values}), ["x"], bounds=[bounds])
    assert e.value.code in {"invalid_bounds", "numerical_range"}


@pytest.mark.parametrize("method", METHODS)
def test_integer_rounding_cannot_change_support_or_sample_identity(method):
    data = pd.DataFrame({"x": [2**53, 2**53 + 1]})
    with pytest.raises(AnalysisError) as e:
        run(method, data, ["x"], bounds=[(float(2**53 - 4), float(2**53 + 4))])
    assert e.value.code == "numerical_range"
    with pytest.raises(AnalysisError) as e:
        run(method, pd.DataFrame({"x": [.2, .3]}), ["x"], bounds=[(0, 2**53 + 1)])
    assert e.value.code == "numerical_range"


@pytest.mark.parametrize("method", METHODS)
def test_global_rng_dtype_device_isolation_and_observed_buffer_sizes(method):
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            output = run(method)
    finally:
        torch.set_default_dtype(old_dtype)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert python_state == random.getstate()
    assert output.attrs["stochastic_draws"] == 0
    assert output.attrs["device"] == "cpu" and output.attrs["precision"] == "float64"
    observed = output.attrs["workspace_observed"]
    assert observed["raw_float64_array_bytes"] == 8 * len(example()) * 2
    assert observed["normalized_tensor_bytes"] == observed["retained_tensor_bytes"]
    assert sum(v for v in observed.values() if isinstance(v, int)) \
        <= output.attrs["resource_plan"]["estimated_workspace_bytes"]


@pytest.mark.parametrize("method", METHODS)
def test_complete_summary_restoration_large_identities_settings_and_latex(method):
    data = pd.DataFrame({"x": np.linspace(0, 1, 401)})
    data.index = pd.Index([(i, f"row-{i}") for i in range(401)], tupleize_cols=False)
    output = run(method, data, ["x"], bounds=[(0, 1)])
    state = summary_state(output)
    restored = restore_summary(state)
    assert restored.title == output.title and restored.attrs == output.attrs
    assert set(restored) == {"intervals", "diagnostics", "sample", "settings"}
    for name in output:
        pd.testing.assert_frame_equal(restored[name], output[name])
    assert summary_state(restored) == state
    settings = dict(zip(restored["settings"].setting, restored["settings"].json, strict=True))
    assert "full_state" in json.loads(settings["sample_positions"])
    assert restored.attrs["sample_positions"] == list(range(401))
    assert [decode(v) for v in restored.attrs["sample_labels_encoded"]] == list(data.index)
    assert "\\begin{tabular}" in restored.to_latex()
    assert len(restored.attrs["sample_hash"]) == 64
    for value in restored["settings"].json:
        json.loads(value)


@pytest.mark.parametrize("method", METHODS)
def test_maximum_column_boundary_alpha_floor_and_small_scale(method):
    data = pd.DataFrame(np.tile(np.linspace(0, 1e-50, 37)[:, None], (1, native.MAX_COLUMNS)),
                        columns=[f"x{i}" for i in range(native.MAX_COLUMNS)])
    output = run(method, data, list(data.columns),
                 bounds=[(0, 1e-50)] * native.MAX_COLUMNS, alpha=1e-12)
    assert output.attrs["family_size"] == native.MAX_COLUMNS
    assert output["intervals"].shape == (native.MAX_COLUMNS, 7)
    assert output["settings"].shape[0] <= 50
    assert (output["diagnostics"].sample_variance > 0).all()
    assert summary_state(restore_summary(summary_state(output))) == summary_state(output)


def test_maximum_row_boundary_and_complete_identity_export():
    data = pd.DataFrame({"x": np.linspace(0, 1, native.MAX_ROWS)})
    output = run(native.empirical_bernstein_mean_ci, data, ["x"], bounds=[(0, 1)])
    assert output.attrs["nobs"] == native.MAX_ROWS
    assert output.attrs["sample_positions"][-1] == native.MAX_ROWS - 1
    assert len(output.attrs["sample_labels_encoded"]) == native.MAX_ROWS
    state = summary_state(output)
    assert len(state.encode()) < 32 * 1024**2
    restored = restore_summary(state)
    assert restored.attrs == output.attrs
    assert restored["intervals"].estimate[0] == pytest.approx(.5, abs=2e-15)


@pytest.mark.parametrize("method", METHODS)
def test_seeded_finite_sample_family_coverage_screen_separate_from_theorem(method):
    rng = np.random.default_rng(903241)
    failures = 0
    # Bernoulli endpoints, asymmetric low-variance marginal and dependent pairs.
    # Threshold prespecified: <=18/300 failures versus alpha=.10.
    for _ in range(300):
        uniform = rng.uniform(size=160)
        data = pd.DataFrame({"x": (uniform < .4).astype(float),
                             "y": 2 + 3 * (uniform < .04).astype(float)})
        result = run(method, data, bounds=[(0, 1), (2, 5)], alpha=.10)
        ci = result["intervals"]
        truth = np.array([.4, 2.12])
        failures += int(not np.all((ci.ci_low <= truth) & (truth <= ci.ci_high)))
    assert failures <= 18
