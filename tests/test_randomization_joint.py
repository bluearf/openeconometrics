"""Independent finite-orbit references, true-subvector error control and state."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest import randomization_joint as rj
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


SIGN_MODEL = "Independent row vectors; every true-null subvector is centrally symmetric about its declared center."
PERM_MODEL = "Independent rows; the joint true-null subvector is exchangeable across the two fixed-size groups."


def _tail(value, tail):
    return abs(value) if tail == "two-sided" else -value if tail == "less" else value


def _reference_stepdown(observed, draws, tail, *, monte_carlo=False):
    """Direct pure-Python intersection max counts, no production helper."""
    z = [_tail(v, tail) for v in observed]
    null = [[_tail(v, tail) for v in row] for row in draws]
    order = sorted(range(len(z)), key=lambda j: -z[j])
    raw, adjusted = [], [None] * len(z)
    correction = int(monte_carlo)
    for j in range(len(z)):
        count = sum(row[j] >= z[j] - 1e-10 for row in null)
        raw.append((count + correction) / (len(draws) + correction))
    previous, start = 0.0, 0
    while start < len(z):
        end = start + 1
        while end < len(z) and z[order[end]] == z[order[start]]:
            end += 1
        count = sum(max(row[j] for j in order[start:]) >= z[order[start]] - 1e-10 for row in null)
        value = max(previous, (count + correction) / (len(draws) + correction))
        for j in order[start:end]:
            adjusted[j] = value
        previous, start = value, end
    return raw, adjusted


def _sign_reference(rows, target=None):
    n, m = len(rows), len(rows[0])
    target = [0] * m if target is None else target
    centered = [[float(row[j] - target[j]) for j in range(m)] for row in rows]
    norms = [math.sqrt(math.fsum(row[j] ** 2 for row in centered)) for j in range(m)]
    observed = [math.fsum(row[j] for row in centered) / norms[j] for j in range(m)]
    draws = []
    for signs in itertools.product((-1, 1), repeat=n):
        draws.append(
            [math.fsum(s * row[j] for s, row in zip(signs, centered)) / norms[j] for j in range(m)]
        )
    return observed, draws


def _permutation_reference(rows, n1, first_indices):
    n, m = len(rows), len(rows[0])
    mean = [math.fsum(row[j] for row in rows) / n for j in range(m)]
    scale = [
        math.sqrt(math.fsum((row[j] - mean[j]) ** 2 for row in rows) / n)
        * math.sqrt(1 / n1 + 1 / (n - n1))
        for j in range(m)
    ]

    def stat(indices):
        chosen = set(indices)
        return [
            (
                math.fsum(row[j] for i, row in enumerate(rows) if i in chosen) / n1
                - math.fsum(row[j] for i, row in enumerate(rows) if i not in chosen) / (n - n1)
            )
            / scale[j]
            for j in range(m)
        ]

    return stat(first_indices), [stat(indices) for indices in itertools.combinations(range(n), n1)]


@pytest.mark.parametrize("tail", ["two-sided", "greater", "less"])
def test_exact_sign_reference_with_nonzero_null_centers(tail):
    frame = pd.DataFrame({"a": [2, 4, -3, 1, 5], "b": [8, 7, 1, 3, 6], "c": [1, -2, 4, 3, -1]})
    null = {"a": 1, "b": 2, "c": -1}
    observed, draws = _sign_reference(frame.to_numpy().tolist(), list(null.values()))
    expected_raw, expected_adjusted = _reference_stepdown(observed, draws, tail)
    out = rj.mean_sign_stepdown(
        frame, list(frame.columns), symmetry_model=SIGN_MODEL, null_values=null, tail=tail
    )
    assert_allclose(out["tests"].statistic, observed, atol=2e-15)
    assert_allclose(out["tests"].p_value, expected_raw, atol=0)
    assert_allclose(out["tests"].adjusted_p_value, expected_adjusted, atol=0)
    assert out.attrs["orbit_count"] == out.attrs["draws"] == 32
    for j in range(3):
        assert_allclose(
            sorted(row[j] for row in out.attrs["joint_null_statistics"]),
            sorted(row[j] for row in draws),
            atol=3e-15,
        )
    assert out.attrs["marginal_exceedances"] == [round(p * 32) for p in expected_raw]
    assert out.attrs["null_values"] == null
    assert "marginal symmetry alone" in out.attrs["assumptions"]


@pytest.mark.parametrize("tail", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("n1", [1, 2, 4])
def test_exact_permutation_all_fixed_size_assignments(tail, n1):
    frame = pd.DataFrame({"a": [0, 2, 7, -1, 4], "b": [3, 1, -2, 6, 4]})
    frame["arm"] = ["first"] * n1 + ["second"] * (5 - n1)
    observed, draws = _permutation_reference(
        frame[["a", "b"]].to_numpy().tolist(), n1, list(range(n1))
    )
    expected_raw, expected_adjusted = _reference_stepdown(observed, draws, tail)
    out = rj.mean_permutation_stepdown(
        frame, ["a", "b"], "arm", exchangeability_model=PERM_MODEL, tail=tail
    )
    assert_allclose(out["tests"].statistic, observed, atol=4e-15)
    assert_allclose(out["tests"].p_value, expected_raw, atol=0)
    assert_allclose(out["tests"].adjusted_p_value, expected_adjusted, atol=0)
    assert_allclose(out.attrs["joint_null_statistics"], draws, atol=4e-15)
    assert out.attrs["orbit_count"] == math.comb(5, n1)
    assert out.attrs["group_sizes"] == [n1, 5 - n1]
    assert out.attrs["group_labels"] == ["first", "second"]
    assert "Equality of means" in out.attrs["assumptions"]


@pytest.mark.parametrize("method", ["sign", "permutation"])
@pytest.mark.parametrize("tail", ["two-sided", "greater", "less"])
def test_scale_label_invariance_and_inclusive_numerical_ties(method, tail):
    frame = pd.DataFrame(
        {
            "a": [0.1, 0.2, 0.3, 0.4],
            "b": [0.2, 0.4, 0.6, 0.8],
            "c": [-1, 2, -1, 3],
            "g": [0, 0, 1, 1],
        }
    )
    function = rj.mean_sign_stepdown if method == "sign" else rj.mean_permutation_stepdown
    args = () if method == "sign" else ("g",)
    options = (
        {"symmetry_model": SIGN_MODEL}
        if method == "sign"
        else {"exchangeability_model": PERM_MODEL}
    )
    base = function(frame, ["a", "b", "c"], *args, tail=tail, **options)
    assert base["tests"].adjusted_p_value.iloc[0] == base["tests"].adjusted_p_value.iloc[1]
    assert any(group == [0, 1] for group in base["tests"].attrs["tie_groups"])
    frame["a"] *= 1e-180
    frame["b"] *= 1e180
    scaled = function(frame, ["c", "b", "a"], *args, tail=tail, **options)
    assert_allclose(
        scaled["tests"].adjusted_p_value, base["tests"].adjusted_p_value.iloc[[2, 1, 0]], atol=0
    )
    assert_allclose(scaled["tests"].p_value, base["tests"].p_value.iloc[[2, 1, 0]], atol=0)
    assert min(base["tests"].p_value) >= 1 / base.attrs["draws"]
    assert base.attrs["numerical_tail_guard"] > 0


@pytest.mark.parametrize("partial", [False, True])
def test_sign_strong_fwer_over_entire_true_subvector_orbit(partial):
    base = np.array([[1, 3], [2, -1], [4, 2], [3, 1], [2, 4], [1, -2]], dtype=float)
    false_column = np.full(len(base), 100.0)
    violations = 0
    for signs in itertools.product((-1, 1), repeat=len(base)):
        data = pd.DataFrame(np.asarray(signs)[:, None] * base, columns=["true0", "true1"])
        if partial:
            data["false"] = false_column
        out = rj.mean_sign_stepdown(data, list(data.columns), symmetry_model=SIGN_MODEL, alpha=0.25)
        violations += bool(out["tests"].reject.iloc[:2].any())
    assert violations <= 0.25 * 2 ** len(base)


@pytest.mark.parametrize("partial", [False, True])
def test_permutation_strong_fwer_over_true_subvector_label_orbit(partial):
    base = np.array([[1, 3], [2, -1], [4, 2], [3, 1], [2, 4], [1, -2]], dtype=float)
    violations = 0
    for assignment in itertools.combinations(range(6), 3):
        mask = np.array([i in assignment for i in range(6)])
        data = pd.DataFrame(base, columns=["true0", "true1"])
        columns = list(data.columns)
        if partial:
            data["false"] = np.where(mask, 100.0, -100.0)
            columns.append("false")
        data["group"] = np.where(mask, "A", "B")
        out = rj.mean_permutation_stepdown(
            data, columns, "group", exchangeability_model=PERM_MODEL, alpha=0.2
        )
        violations += bool(out["tests"].reject.iloc[:2].any())
    assert violations <= 0.2 * math.comb(6, 3)


@pytest.mark.parametrize("method", ["sign", "permutation"])
def test_monte_carlo_plus_one_local_generator_and_full_persistence(method):
    data = pd.DataFrame(
        {"a": [1.0, -2.0, 3.0, 5.0], "b": [4.0, 2.0, -1.0, 3.0], "group": ["A", "A", "B", "B"]}
    )
    function = rj.mean_sign_stepdown if method == "sign" else rj.mean_permutation_stepdown
    args = () if method == "sign" else ("group",)
    options = (
        {"symmetry_model": SIGN_MODEL}
        if method == "sign"
        else {"exchangeability_model": PERM_MODEL}
    )
    torch.manual_seed(823)
    before = torch.random.get_rng_state().clone()
    out = function(
        data, ["a", "b"], *args, calibration="monte_carlo", draws=131, seed=18, **options
    )
    assert torch.equal(before, torch.random.get_rng_state())
    same = function(
        data, ["a", "b"], *args, calibration="monte_carlo", draws=131, seed=18, **options
    )
    other = function(
        data, ["a", "b"], *args, calibration="monte_carlo", draws=131, seed=19, **options
    )
    assert out.attrs == same.attrs
    assert out.attrs["transformation_hash"] != other.attrs["transformation_hash"]
    expected_raw, expected_adjusted = _reference_stepdown(
        out.attrs["observed_statistics"],
        out.attrs["joint_null_statistics"],
        "two-sided",
        monte_carlo=True,
    )
    assert_allclose(out["tests"].p_value, expected_raw, atol=0)
    assert_allclose(out["tests"].adjusted_p_value, expected_adjusted, atol=0)
    assert_allclose(
        out["tests"].p_value, (np.array(out.attrs["marginal_exceedances"]) + 1) / 132, atol=0
    )
    assert min(out["tests"].p_value) >= 1 / 132
    payload = summary_state(out)
    restored = restore_summary(payload)
    assert restored.attrs == out.attrs
    for name in out:
        pd.testing.assert_frame_equal(
            restored[name], out[name], check_dtype=False, check_names=False
        )
    assert len(json.loads(payload)["attrs"]["joint_null_statistics"]) == 131
    assert (
        json.loads(out["settings"].set_index("setting").loc["joint_null_statistics", "json"])[
            "full_state"
        ]
        == "oe.summary_state(output)"
    )
    assert "\\end{tabular}" in out["tests"].to_latex()


@pytest.mark.parametrize("method", ["sign", "permutation"])
def test_joint_missing_selection_positional_hash_and_user_input_preservation(method):
    data = pd.DataFrame(
        {
            "a": [1.0, 2, np.nan, 4, 5],
            "b": [2.0, np.nan, 3, 4, 6],
            "arm": ["A", "A", "B", "B", "B"],
            "ignored": [np.nan] * 5,
        },
        index=[9, 9, 2, -1, 9],
    )
    before = data.copy(deep=True)
    function = rj.mean_sign_stepdown if method == "sign" else rj.mean_permutation_stepdown
    args = () if method == "sign" else ("arm",)
    options = (
        {"symmetry_model": SIGN_MODEL}
        if method == "sign"
        else {"exchangeability_model": PERM_MODEL}
    )
    with pytest.raises(AnalysisError, match="complete"):
        function(data, ["a", "b"], *args, **options)
    out = function(data, ["a", "b"], *args, missing="drop", **options)
    assert out.attrs["sample_positions"] == [0, 3, 4]
    assert out["sample"].position.tolist() == [0, 3, 4]
    assert out.attrs["excluded_rows"] == 2
    changed_index = data.set_axis(range(5))
    equal = function(changed_index, ["a", "b"], *args, missing="drop", **options)
    assert equal.attrs["input_hash"] == out.attrs["input_hash"]
    changed_values = data.copy()
    changed_values.loc[changed_values.index == -1, "a"] = 7
    different = function(changed_values, ["a", "b"], *args, missing="drop", **options)
    assert different.attrs["input_hash"] != out.attrs["input_hash"]
    pd.testing.assert_frame_equal(data, before)
    assert "selection must preserve" in out.attrs["assumptions"]


@pytest.mark.parametrize("bad", [None, "", " " * 3, False, "x" * 2001])
def test_explicit_invariance_model_required(bad):
    data = pd.DataFrame({"a": [1, 2], "g": [0, 1]})
    with pytest.raises(AnalysisError):
        rj.mean_sign_stepdown(data, ["a"], symmetry_model=bad)
    with pytest.raises(AnalysisError):
        rj.mean_permutation_stepdown(data, ["a"], "g", exchangeability_model=bad)


@pytest.mark.parametrize(
    "option,value",
    [
        ("alpha", True),
        ("alpha", np.nan),
        ("alpha", 1),
        ("tail", []),
        ("tail", "upper"),
        ("calibration", []),
        ("calibration", "exact"),
        ("draws", True),
        ("draws", 1),
        ("draws", 100001),
        ("seed", -1),
        ("seed", True),
        ("missing", []),
        ("missing", "ignore"),
    ],
)
def test_option_contract(option, value):
    with pytest.raises(AnalysisError):
        rj.mean_sign_stepdown(
            pd.DataFrame({"a": [1, 2]}), ["a"], symmetry_model=SIGN_MODEL, **{option: value}
        )


@pytest.mark.parametrize(
    "data,columns",
    [
        ({"a": [1, 2]}, ["a"]),
        (pd.DataFrame({"a": [1, 2]}), "a"),
        (pd.DataFrame({"a": [1, 2]}), []),
        (pd.DataFrame({"a": [1, 2]}), ["a", "a"]),
        (pd.DataFrame({"a": [1, 2]}), ["z"]),
        (pd.DataFrame({"a": [True, False]}), ["a"]),
        (pd.DataFrame({"a": [1 + 2j, 2 + 1j]}), ["a"]),
        (pd.DataFrame({"a": ["1", "2"]}), ["a"]),
        (pd.DataFrame({"a": [1, np.inf]}), ["a"]),
        (pd.DataFrame({"a": [0, 0]}), ["a"]),
        (pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]), ["a"]),
    ],
)
def test_reject_unsupported_input(data, columns):
    with pytest.raises(AnalysisError):
        rj.mean_sign_stepdown(data, columns, symmetry_model=SIGN_MODEL)


@pytest.mark.parametrize("null", [{}, {"a": 0, "extra": 1}, {"a": True}, {"a": np.inf}, [0]])
def test_null_center_contract(null):
    with pytest.raises(AnalysisError):
        rj.mean_sign_stepdown(
            pd.DataFrame({"a": [1, 2]}), ["a"], symmetry_model=SIGN_MODEL, null_values=null
        )


@pytest.mark.parametrize(
    "group_values", [[0, 0, 0], [0, 1, 2], [[1], [2], [1]], [0, np.inf, 1], ["", "A", "B"]]
)
def test_group_identity_contract(group_values):
    data = pd.DataFrame({"a": [1, 2, 3], "g": group_values})
    with pytest.raises(AnalysisError):
        rj.mean_permutation_stepdown(data, ["a"], "g", exchangeability_model=PERM_MODEL)


def test_zero_scale_finite_overflow_and_inf_not_hidden_by_missing():
    with pytest.raises(AnalysisError, match="RMS"):
        rj.mean_permutation_stepdown(
            pd.DataFrame({"a": [1, 1], "g": [0, 1]}), ["a"], "g", exchangeability_model=PERM_MODEL
        )
    with pytest.raises(AnalysisError, match="Subtracting"):
        rj.mean_sign_stepdown(
            pd.DataFrame({"a": [1e308, 1e308]}),
            ["a"],
            symmetry_model=SIGN_MODEL,
            null_values={"a": -1e308},
        )
    data = pd.DataFrame({"a": [1, np.inf, 3], "b": [1, np.nan, 3]})
    with pytest.raises(AnalysisError, match="infinite"):
        rj.mean_sign_stepdown(data, ["a", "b"], symmetry_model=SIGN_MODEL, missing="drop")


def test_native_numeric_uses_cpu_float64_without_changing_defaults():
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            out = rj.mean_sign_stepdown(
                pd.DataFrame({"a": [1, 2, 3]}), ["a"], symmetry_model=SIGN_MODEL
            )
        assert out.attrs["device"] == "cpu"
        assert out.attrs["precision"] == "float64"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(before)


def test_enumeration_and_work_guards_precede_numeric_and_draw_allocations(monkeypatch):
    def fail(_):
        raise AssertionError("Numeric tensor allocated before admission")

    monkeypatch.setattr(rj, "_numeric", fail)
    with pytest.raises(AnalysisError, match="orbit"):
        rj.mean_sign_stepdown(pd.DataFrame({"a": np.ones(17)}), ["a"], symmetry_model=SIGN_MODEL)
    with pytest.raises(AnalysisError, match="orbit"):
        rj.mean_permutation_stepdown(
            pd.DataFrame({"a": range(30), "g": [0] * 15 + [1] * 15}),
            ["a"],
            "g",
            exchangeability_model=PERM_MODEL,
        )
    with pytest.raises(AnalysisError, match="operations"):
        rj.mean_sign_stepdown(
            pd.DataFrame(np.ones((10000, 2)), columns=["a", "b"]),
            ["a", "b"],
            symmetry_model=SIGN_MODEL,
            calibration="monte_carlo",
            draws=10000,
        )
    with pytest.raises(AnalysisError, match="10000 rows"):
        rj.mean_sign_stepdown(pd.DataFrame({"a": np.ones(10001)}), ["a"], symmetry_model=SIGN_MODEL)


def test_configurable_workspace_guard_and_record():
    frame = pd.DataFrame({"a": range(15)})
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace") as caught:
        rj.mean_sign_stepdown(frame, ["a"], symmetry_model=SIGN_MODEL)
    assert caught.value.resource_plan["estimated_workspace_bytes"] > 1024**2
    out = rj.mean_sign_stepdown(frame.iloc[:5], ["a"], symmetry_model=SIGN_MODEL)
    plan = out.attrs["resource_plan"]
    assert plan["estimated_workspace_bytes"] == sum(plan["buffers"].values())
    assert plan["estimated_workspace_bytes"] <= plan["budget_bytes"]
    assert out.attrs["work"] == 32 * 5


def test_monte_carlo_large_orbit_records_capped_count_without_enumeration():
    out = rj.mean_sign_stepdown(
        pd.DataFrame({"a": range(30)}),
        ["a"],
        symmetry_model=SIGN_MODEL,
        calibration="monte_carlo",
        draws=49,
    )
    assert out.attrs["orbit_count"] is None
    assert out.attrs["orbit_count_exceeds_budget"] is True
    assert out.attrs["draws"] == 49
    perm = rj.mean_permutation_stepdown(
        pd.DataFrame({"a": range(30), "g": [0] * 15 + [1] * 15}),
        ["a"],
        "g",
        exchangeability_model=PERM_MODEL,
        calibration="monte_carlo",
        draws=49,
    )
    assert perm.attrs["orbit_count"] is None
    assert perm.attrs["orbit_count_exceeds_budget"] is True


def test_nullable_numbers_group_missing_and_group_orientation():
    data = pd.DataFrame(
        {"a": pd.Series([1, pd.NA, 4, 6], dtype="Float64"), "g": ["B", "A", "A", None]}
    )
    out = rj.mean_permutation_stepdown(
        data, ["a"], "g", exchangeability_model=PERM_MODEL, missing="drop", tail="less"
    )
    assert out.attrs["sample_positions"] == [0, 2]
    assert out.attrs["group_labels"] == ["B", "A"]
    assert out["tests"].statistic.iloc[0] < 0
    assert out.attrs["group_sizes"] == [1, 1]
    assert out["sample"].group.tolist() == ["B", "A"]


@pytest.mark.parametrize("method", ["sign", "permutation"])
def test_monte_carlo_samples_uniform_complete_transformations(method):
    """Independent orbit membership/frequencies detect biased/incomplete draws."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "g": [0, 0, 1, 1]})
    if method == "sign":
        observed, orbit = _sign_reference(frame[["a"]].to_numpy().tolist())
        out = rj.mean_sign_stepdown(
            frame, ["a"], symmetry_model=SIGN_MODEL, calibration="monte_carlo", draws=4096, seed=723
        )
    else:
        observed, orbit = _permutation_reference(frame[["a"]].to_numpy().tolist(), 2, [0, 1])
        out = rj.mean_permutation_stepdown(
            frame,
            ["a"],
            "g",
            exchangeability_model=PERM_MODEL,
            calibration="monte_carlo",
            draws=4096,
            seed=723,
        )
    allowed = [row[0] for row in orbit]
    assert len(set(allowed)) == len(allowed)
    frequencies = [0] * len(allowed)
    for row in out.attrs["joint_null_statistics"]:
        distances = [abs(row[0] - value) for value in allowed]
        choice = min(range(len(allowed)), key=distances.__getitem__)
        assert distances[choice] < 3e-14
        frequencies[choice] += 1
    expected = 4096 / len(allowed)
    assert max(abs(count - expected) for count in frequencies) < 6 * math.sqrt(expected)
    assert_allclose(out.attrs["observed_statistics"], observed, atol=3e-15)


def test_roundoff_guard_records_conservative_near_zero_tail_counts():
    frame = pd.DataFrame({"a": [1e100, -1e100, 3.0], "b": [1.0, 2.0, 4.0]})
    out = rj.mean_sign_stepdown(frame, ["a", "b"], symmetry_model=SIGN_MODEL, tail="greater")
    observed = out.attrs["observed_statistics"]
    raw_draws = out.attrs["joint_null_statistics"]
    exact_computed_counts = [sum(row[j] >= observed[j] for row in raw_draws) for j in range(2)]
    assert all(
        got >= unguarded
        for got, unguarded in zip(out.attrs["marginal_exceedances"], exact_computed_counts)
    )
    assert out.attrs["numerical_tail_rule"].startswith(
        "Move each transformed tail statistic outward"
    )


def test_permutation_nearly_constant_offsets_keep_centered_geometry_and_row_order():
    offsets = np.array([0, 1, 2, 4, 7, 8], dtype=float)
    frame = pd.DataFrame(
        {
            "a": 1.5 + offsets * math.ulp(1.5),
            "b": [2.0, 5.0, -1.0, 3.0, 4.0, 7.0],
            "g": [0, 0, 0, 1, 1, 1],
        }
    )
    # Independent oracle subtracts the raw minimum before summing, preserving
    # the exactly represented spacing even around a large common offset.
    raw = frame[["a", "b"]].to_numpy()
    centered_reference = raw - raw.min(axis=0)
    observed, orbit = _permutation_reference(centered_reference.tolist(), 3, [0, 1, 2])
    expected_raw, expected_adjusted = _reference_stepdown(observed, orbit, "two-sided")
    out = rj.mean_permutation_stepdown(frame, ["a", "b"], "g", exchangeability_model=PERM_MODEL)
    assert_allclose(out["tests"].statistic, observed, atol=3e-15)
    assert_allclose(out["tests"].p_value, expected_raw, atol=0)
    assert_allclose(out["tests"].adjusted_p_value, expected_adjusted, atol=0)
    assert_allclose(out.attrs["joint_null_statistics"], orbit, atol=3e-15)
    permuted = rj.mean_permutation_stepdown(
        frame.iloc[[2, 1, 0, 5, 4, 3]], ["a", "b"], "g", exchangeability_model=PERM_MODEL
    )
    assert_allclose(permuted["tests"].statistic, observed, atol=3e-15)
    assert_allclose(permuted["tests"].adjusted_p_value, expected_adjusted, atol=0)


def test_permutation_midpoint_anchor_handles_unrepresentable_raw_range():
    frame = pd.DataFrame({"a": [-1e308, -8e307, 7e307, 1e308], "g": [0, 0, 1, 1]})
    out = rj.mean_permutation_stepdown(frame, ["a"], "g", exchangeability_model=PERM_MODEL)
    observed, orbit = _permutation_reference((frame[["a"]] / 1e308).to_numpy().tolist(), 2, [0, 1])
    assert_allclose(out["tests"].statistic, observed, atol=3e-15)
    assert_allclose(out.attrs["joint_null_statistics"], orbit, atol=3e-15)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
@pytest.mark.parametrize("method", ["sign", "permutation"])
def test_unsafe_integer_observations_are_rejected_before_float64_cast(dtype, method, monkeypatch):
    # 2**53+1 and 2**53 are distinct observations but collapse to the same
    # double. A test on the cast table would therefore test different data.
    original = pd.Series([2**53, 2**53 + 1, 2**53 + 2], dtype=dtype)
    assert int(original.iloc[0]) != int(original.iloc[1])
    assert float(original.iloc[0]) == float(original.iloc[1])
    frame = pd.DataFrame({"a": original, "g": [0, 1, 1]})

    def fail(*_args, **_kwargs):
        raise AssertionError("Numeric float64 conversion preceded integer identity validation")

    monkeypatch.setattr(rj, "_numeric", fail)
    with pytest.raises(AnalysisError, match="exact float64 domain"):
        if method == "sign":
            rj.mean_sign_stepdown(frame, ["a"], symmetry_model=SIGN_MODEL)
        else:
            rj.mean_permutation_stepdown(frame, ["a"], "g", exchangeability_model=PERM_MODEL)


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_exact_integer_boundary_is_supported_but_unsafe_excluded_rows_are_refused(dtype):
    allowed = pd.DataFrame({"a": pd.Series([-(2**53), 2**53], dtype=dtype)})
    out = rj.mean_sign_stepdown(allowed, ["a"], symmetry_model=SIGN_MODEL)
    assert out["tests"].statistic.iloc[0] == 0
    bad = pd.DataFrame({"a": pd.Series([1, 2, -(2**53) - 1], dtype=dtype), "b": [1.0, 2.0, np.nan]})
    with pytest.raises(AnalysisError, match="exact float64 domain"):
        rj.mean_sign_stepdown(bad, ["a", "b"], symmetry_model=SIGN_MODEL, missing="drop")


def test_integer_null_identity_and_overflow_are_not_silently_converted():
    integer_center = 2**53 + 1
    cast_center = float(integer_center)
    assert cast_center != integer_center
    frame = pd.DataFrame({"a": [float(2**53), float(2**53)]})
    # The original null would put both centered observations at -1; the cast
    # null puts both at zero. Neither rounding nor the resulting zero-scale
    # route is an acceptable substitute for the caller's declared null.
    assert int(frame.a.iloc[0]) - integer_center == -1
    assert frame.a.iloc[0] - cast_center == 0
    with pytest.raises(AnalysisError, match="Integer null centers"):
        rj.mean_sign_stepdown(
            frame, ["a"], symmetry_model=SIGN_MODEL, null_values={"a": integer_center}
        )
    with pytest.raises(AnalysisError, match="representable finite"):
        rj.mean_sign_stepdown(
            pd.DataFrame({"a": [1.0, 2.0]}),
            ["a"],
            symmetry_model=SIGN_MODEL,
            null_values={"a": 10**1000},
        )
    # Exactly represented integer centers above the observation-safe range
    # remain valid scalars, because their declared numeric identity is intact.
    out = rj.mean_sign_stepdown(
        pd.DataFrame({"a": [1.0, 2.0]}), ["a"], symmetry_model=SIGN_MODEL, null_values={"a": 2**60}
    )
    assert out.attrs["null_values"]["a"] == 2**60


@pytest.mark.parametrize("method", ["sign", "permutation"])
def test_extended_precision_columns_and_null_scalars_require_explicit_conversion(method):
    dtype = np.dtype(np.longdouble)
    if dtype.itemsize <= 8:
        pytest.skip("Platform has no wider-than-float64 longdouble")
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    assert value != np.longdouble(1)
    assert float(value) == 1.0
    frame = pd.DataFrame({"a": np.array([np.longdouble(1), value], dtype=dtype), "g": [0, 1]})
    with pytest.raises(AnalysisError, match="wider observations"):
        if method == "sign":
            rj.mean_sign_stepdown(frame, ["a"], symmetry_model=SIGN_MODEL)
        else:
            rj.mean_permutation_stepdown(frame, ["a"], "g", exchangeability_model=PERM_MODEL)
    with pytest.raises(AnalysisError, match="wider scalars"):
        rj.mean_sign_stepdown(
            pd.DataFrame({"a": [1.0, 2.0]}),
            ["a"],
            symmetry_model=SIGN_MODEL,
            null_values={"a": value},
        )
