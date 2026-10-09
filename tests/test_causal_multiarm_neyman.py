"""Independent weighted assignment universes and complete covariance-bound laws."""

from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design import common as m
from openecon.econometrics.causal_design import multiarm_neyman as k
from openecon.resources import use_workspace_budget


L2 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
C3 = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 1.0]])
L3 = np.vstack([np.eye(3), C3])
CASES = ["bernoulli", "complete", "stratified"]


def fixture():
    return pd.DataFrame(
        {
            "y": [1.0, 3.0, 2.0, 6.0, 4.0, 9.0],
            "a": [0, 0, 1, 1, 2, 2],
            "s": ["one"] * 6,
            "d": [0, 0, 1, 1, 0, 1],
            "p": [0.2, 0.4, 0.6, 0.8, 0.3, 0.7],
        },
        index=["r", "r", 2**60 + 1, 2**60 + 2, False, 1.5],
    )


def invoke(case, data=None, **options):
    data = fixture() if data is None else data
    if case == "bernoulli":
        return k.bernoulli_neyman_ate(data, "y", "d", "p", design="bernoulli_randomized", **options)
    function = k.multiarm_neyman_ate if case == "complete" else k.stratified_multiarm_neyman_ate
    return function(
        data,
        "y",
        "a",
        *([] if case == "complete" else ["s"]),
        arms=[0, 1, 2],
        design="complete_randomized" if case == "complete" else "stratified_randomized",
        **options,
    )


def extract(result):
    return result["effects"]["estimate"].to_numpy(), result["covariance_bound"].to_numpy(
        dtype=float
    )


def assignments(counts):
    n = sum(counts)

    def walk(available, arm):
        if arm == len(counts) - 1:
            yield {i: arm for i in available}
        else:
            for chosen in combinations(available, counts[arm]):
                rest = [i for i in available if i not in chosen]
                for remaining in walk(rest, arm + 1):
                    yield {**{i: arm for i in chosen}, **remaining}

    for selected in walk(list(range(n)), 0):
        yield np.array([selected[i] for i in range(n)])


def observed_oracle(outcomes, codes, transform=L3):
    arms = [outcomes[codes == arm] for arm in range(transform.shape[1])]
    means = np.array([math.fsum(values) / len(values) for values in arms])
    variances = np.array([np.var(values, ddof=1) / len(values) for values in arms])
    return transform @ means, transform @ np.diag(variances) @ transform.T


def check_law(points, bounds, truth, gap, probabilities=None):
    points, bounds = np.asarray(points), np.asarray(bounds)
    probabilities = (
        np.full(len(points), 1 / len(points)) if probabilities is None else probabilities
    )
    mean = probabilities @ points
    true_covariance = np.einsum("i,ij,ik->jk", probabilities, points - mean, points - mean)
    expected_bound = np.einsum("i,ijk->jk", probabilities, bounds)
    np.testing.assert_allclose(mean, truth, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(expected_bound - true_covariance, gap, rtol=7e-13, atol=7e-13)
    assert np.linalg.eigvalsh(expected_bound - true_covariance).min() >= -3e-12
    return true_covariance


@pytest.mark.parametrize("counts", [[2, 2, 2], [2, 2, 3]])
def test_multiarm_full_90_210_assignment_laws_and_full_matrix_expectation_gap(counts):
    n = sum(counts)
    potential = np.array(
        [
            [-3.0, 2.0, 4.0],
            [1.0, -2.0, 7.0],
            [5.0, 6.0, -1.0],
            [-4.0, 3.0, 2.0],
            [8.0, 4.0, 9.0],
            [2.0, 11.0, -5.0],
            [7.0, -3.0, 6.0],
        ]
    )[:n]
    points, bounds = [], []
    for assigned in assignments(counts):
        observed = potential[np.arange(n), assigned]
        result = k.multiarm_neyman_ate(
            pd.DataFrame({"y": observed, "a": assigned}),
            "y",
            "a",
            arms=[0, 1, 2],
            design="complete_randomized",
        )
        point, bound = extract(result)
        reference, reference_bound = observed_oracle(observed, assigned)
        np.testing.assert_allclose(point, reference, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(bound, reference_bound, rtol=2e-13, atol=2e-13)
        for summary in result.attrs["state"]["group_summaries"]:
            np.testing.assert_allclose(summary["estimates"], reference, atol=2e-13)
        points.append(point)
        bounds.append(bound)
    assert len(points) == (90 if n == 6 else 210)
    gap = L3 @ np.cov(potential, rowvar=False, ddof=1) @ L3.T / n
    actual = check_law(points, bounds, L3 @ potential.mean(0), gap)
    assert abs(actual[0, 1]) > 0.01  # The unidentified actual arm covariance is not zero.
    assert any(abs(matrix[3, 4]) > 0.01 for matrix in bounds)  # Shared reference arm.


def test_two_strata_full_cartesian_8100_law_full_covariance_and_population_weights():
    first = np.array(
        [
            [-3.0, 2.0, 4.0],
            [1.0, -2.0, 7.0],
            [5.0, 6.0, -1.0],
            [-4.0, 3.0, 2.0],
            [8.0, 4.0, 9.0],
            [2.0, 11.0, -5.0],
        ]
    )
    second = np.array(
        [
            [9.0, -3.0, 1.0],
            [4.0, 8.0, -2.0],
            [-1.0, 2.0, 7.0],
            [6.0, 11.0, 5.0],
            [3.0, -4.0, 8.0],
            [-2.0, 5.0, 12.0],
        ]
    )
    universe = list(assignments([2, 2, 2]))
    points, bounds = [], []
    for left, right in product(universe, repeat=2):
        assigned = np.r_[left, right]
        observed = np.r_[first[np.arange(6), left], second[np.arange(6), right]]
        result = k.stratified_multiarm_neyman_ate(
            pd.DataFrame({"y": observed, "a": assigned, "s": ["left"] * 6 + ["right"] * 6}),
            "y",
            "a",
            "s",
            arms=[0, 1, 2],
            design="stratified_randomized",
        )
        point, bound = extract(result)
        p0, b0 = observed_oracle(observed[:6], left)
        p1, b1 = observed_oracle(observed[6:], right)
        np.testing.assert_allclose(point, (p0 + p1) / 2, atol=3e-13)
        np.testing.assert_allclose(bound, (b0 + b1) / 4, atol=3e-13)
        points.append(point)
        bounds.append(bound)
    assert len(points) == 8100
    gap = (
        L3 @ np.cov(first, rowvar=False, ddof=1) @ L3.T
        + L3 @ np.cov(second, rowvar=False, ddof=1) @ L3.T
    ) / 24
    check_law(points, bounds, L3 @ (first.mean(0) + second.mean(0)) / 2, gap)


def test_strata_different_sizes_and_counts_preserve_fixed_individual_weights():
    data = pd.DataFrame(
        {
            "y": [-3.0, 5.0, 2.0, 6.0, 4.0, 9.0, 1.0, 4.0, -2.0, 8.0, 3.0, 7.0, 11.0],
            "a": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2],
            "s": ["a"] * 6 + ["b"] * 7,
        }
    )
    result = invoke("stratified", data)
    left, lb = observed_oracle(data.y.to_numpy()[:6], data.a.to_numpy()[:6])
    right, rb = observed_oracle(data.y.to_numpy()[6:], data.a.to_numpy()[6:])
    point, bound = extract(result)
    np.testing.assert_allclose(point, 6 / 13 * left + 7 / 13 * right, atol=3e-14)
    np.testing.assert_allclose(bound, (6 / 13) ** 2 * lb + (7 / 13) ** 2 * rb, atol=3e-14)
    assert [g["weight"] for g in result.attrs["state"]["group_summaries"]] == [6 / 13, 7 / 13]


def test_heterogeneous_known_bernoulli_full_unconditional_weighted_law_and_psd_gap():
    p = np.array([0.15, 0.25, 0.55, 0.75, 0.85])
    potential = np.array([[-3.0, 4.0], [2.0, -1.0], [7.0, 3.0], [-4.0, 8.0], [5.0, -2.0]])
    points, bounds, masses = [], [], []
    n = len(p)
    for bits in product([0, 1], repeat=n):
        assigned = np.array(bits)
        observed = potential[np.arange(n), assigned]
        result = k.bernoulli_neyman_ate(
            pd.DataFrame({"y": observed, "d": assigned, "p": p}),
            "y",
            "d",
            "p",
            design="bernoulli_randomized",
        )
        scores = np.column_stack(
            [(1 - assigned) * observed / (1 - p) / n, assigned * observed / p / n]
        )
        reference = L2 @ np.array([math.fsum(scores[:, a]) for a in [0, 1]])
        reference_bound = L2 @ np.diag(np.square(scores).sum(0)) @ L2.T
        point, bound = extract(result)
        np.testing.assert_allclose(point, reference, rtol=3e-14, atol=3e-14)
        np.testing.assert_allclose(bound, reference_bound, rtol=3e-14, atol=3e-14)
        assert result.attrs["state"]["hc1_factor"] is None
        assert not result.attrs["state"]["covariance_identified"]
        points.append(point)
        bounds.append(bound)
        masses.append(np.prod(np.where(assigned, p, 1 - p)))
    np.testing.assert_allclose(math.fsum(masses), 1.0, atol=2e-16)
    gap = L2 @ (potential.T @ potential / n**2) @ L2.T
    actual = check_law(points, bounds, L2 @ potential.mean(0), gap, np.asarray(masses))
    assert actual[0, 1] != 0


@pytest.mark.parametrize("arm", [0, 1])
def test_empty_observed_bernoulli_arm_retained_without_fabricated_precision(arm):
    data = fixture()
    data["d"] = arm
    result = invoke("bernoulli", data)
    state = result.attrs["state"]
    assert state["observed_arm_counts"] == ([6, 0] if arm == 0 else [0, 6])
    missing = result["effects"].iloc[1 - arm]
    assert missing["estimate"] == missing["std_error"] == 0
    assert missing["degenerate_variance"]
    assert pd.isna(missing["ci_low"]) and pd.isna(missing["ci_high"])
    assert pd.isna(missing["z"]) and pd.isna(missing["p_value"])
    assert state["all_zero_all_one_allowed"]
    assert result.attrs["n_missing"] == 0


@pytest.mark.parametrize("value", [0.1, 1.0, 1e100])
def test_constant_bernoulli_outcome_keeps_true_nonzero_raw_ht_contrast(value):
    data = pd.DataFrame({"y": [value] * 5, "d": [0, 0, 1, 1, 1], "p": [0.5] * 5})
    result = invoke("bernoulli", data)
    point, _ = extract(result)
    np.testing.assert_allclose(point, np.array([0.8, 1.2, 0.4]) * value, rtol=3e-15)
    assert point[2] > 0
    state = result.attrs["state"]
    np.testing.assert_allclose(state["point_anchor_coefficients"], [0.8, 1.2, 0.4], rtol=2e-15)
    assert state["point_anchor"] == value
    assert result["effects"]["std_error"].iloc[2] > 0
    assert m.causal_design_load(m.causal_design_save(result)).attrs == result.attrs


@pytest.mark.parametrize("case", ["complete", "stratified"])
@pytest.mark.parametrize("value", [0.1, 1.0, 1e100])
def test_constant_multiarm_full_artifact_unequal_counts_exact_zero_contrasts(case, value):
    data = pd.DataFrame({"y": [value] * 7, "a": [0, 0, 1, 1, 2, 2, 2], "s": ["a"] * 7})
    if case == "stratified":
        data = pd.concat([data, data.assign(s="b")], ignore_index=True)
    result = invoke(case, data)
    for artifact in [result, m.causal_design_load(m.causal_design_save(result))]:
        expected = [value] * 3 + [0.0] * 3
        np.testing.assert_array_equal(artifact["effects"]["estimate"], expected)
        assert artifact.attrs["state"]["estimates"] == expected
        np.testing.assert_array_equal(artifact["covariance_bound"], np.zeros((6, 6)))
        assert artifact["effects"]["ci_low"].isna().all()
        for gi, summary in enumerate(artifact.attrs["state"]["group_summaries"]):
            assert summary["estimates"] == expected
            group = artifact["groups"].loc[artifact["groups"]["group_index"] == gi]
            np.testing.assert_array_equal(group["estimate"], expected)
            assert summary["point_anchor_coefficients"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_overlapping_custom_contrasts_nonzero_null_and_full_shared_covariance():
    contrasts = {"one": [-1.0, 1.0, 0.0], "two": [-1.0, 0.0, 1.0], "weighted": [-0.75, 0.5, 0.25]}
    nulls = {"weighted": 2.0, "one": -1.0, "two": 3.0}
    result = invoke("complete", contrasts=contrasts, null_values=nulls, level=0.9)
    transform = np.vstack([np.eye(3), list(contrasts.values())])
    point, bound = extract(result)
    reference, reference_bound = observed_oracle(
        fixture().y.to_numpy(), fixture().a.to_numpy(), transform
    )
    np.testing.assert_allclose(point, reference, atol=2e-15)
    np.testing.assert_allclose(bound, reference_bound, atol=2e-15)
    assert bound[3, 4] > 0
    effects = result["effects"]
    for i, name in enumerate(contrasts, 3):
        z = (point[i] - nulls[name]) / math.sqrt(bound[i, i])
        assert effects.iloc[i]["z"] == pytest.approx(z, rel=2e-14)
        assert effects.iloc[i]["p_value"] == pytest.approx(2 * norm.sf(abs(z)), rel=2e-12)
        radius = norm.isf(0.05) * math.sqrt(bound[i, i])
        assert effects.iloc[i]["ci_low"] == pytest.approx(point[i] - radius, rel=2e-12)
    assert effects["df"].isna().all() and effects["p_value"].iloc[:3].isna().all()
    assert not result.attrs["state"]["joint_tests"]


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_bernoulli_normal_tail_oracle_and_no_hc1(alternative):
    result = invoke("bernoulli", alternative=alternative, null_effect=1.0, level=0.9)
    point, bound = extract(result)
    z = (point[2] - 1) / math.sqrt(bound[2, 2])
    expected = (
        2 * norm.sf(abs(z))
        if alternative == "two-sided"
        else norm.sf(z if alternative == "greater" else -z)
    )
    assert result["effects"]["p_value"].iloc[2] == pytest.approx(expected, rel=2e-12)
    data = fixture()
    scores = np.column_stack(
        [(1 - data.d) * data.y / (1 - data.p) / 6, data.d * data.y / data.p / 6]
    )
    np.testing.assert_allclose(bound, L2 @ np.diag(np.square(scores).sum(0)) @ L2.T, atol=2e-14)
    assert not np.allclose(bound, L2 @ np.cov(scores, rowvar=False) * 6 @ L2.T)


@pytest.mark.parametrize("row", [[0.1, 0.2, -0.3], [1e100, -1e100, 1.0], [1.0, -1.0, 5e-324]])
def test_exact_dyadic_zero_sum_refuses_floating_or_tolerance_pretence(row):
    with pytest.raises(AnalysisError, match="exactly to zero"):
        invoke("complete", contrasts={"not_zero": row})


def test_dyadic_certificate_and_exact_original_contrast_hidden_by_rounded_arm_means():
    data = pd.DataFrame({"y": [1e16, 0.0, 1e16, 1.0, 0.0, 2.0], "a": [0, 0, 1, 1, 2, 2]})
    result = invoke("complete", data, contrasts={"first": [-1.0, 1.0, 0.0]})
    assert result["effects"]["estimate"].iloc[3] == 0.5
    assert result["effects"]["estimate"].iloc[0] == result["effects"]["estimate"].iloc[1]
    assert result.attrs["state"]["group_summaries"][0]["estimates"][3] == 0.5
    assert result["groups"]["estimate"].iloc[3] == 0.5
    certificate = result.attrs["state"]["contrast_zero_sum_certificates"][0]
    assert sum(int(value) for value in certificate["numerators"]) == 0


@pytest.mark.parametrize("case", ["complete", "stratified"])
def test_typed_arm_and_stratum_identity_large_integers_and_order_persist(case):
    labels = [True, 1, 1.0, "1", 2**60 + 1]
    data = pd.DataFrame(
        {
            "y": list(range(10)),
            "a": pd.Series([label for label in labels for _ in range(2)], dtype=object),
            "s": pd.Series([2**60 + 3] * 10, dtype=object),
        }
    )
    function = k.multiarm_neyman_ate if case == "complete" else k.stratified_multiarm_neyman_ate
    result = function(
        data,
        "y",
        "a",
        *([] if case == "complete" else ["s"]),
        arms=labels,
        design="complete_randomized" if case == "complete" else "stratified_randomized",
    )
    state = result.attrs["state"]
    assert state["arm_types"] == ["bool", "int", "float", "str", "int"]
    assert state["arm_order"][-1] == 2**60 + 1
    assert result["arms"]["identifier"].dtype == object
    assert result["arms"]["identifier"].iloc[-1] == 2**60 + 1
    restored = m.causal_design_load(m.causal_design_save(result))
    assert restored.attrs == result.attrs
    pd.testing.assert_frame_equal(restored["arms"], result["arms"], check_exact=True)


@pytest.mark.parametrize("case", CASES)
def test_full_path_json_persistence_typed_state_and_private_inputs(case, tmp_path):
    data = fixture()
    before = data.copy(deep=True)
    result = invoke(case, data)
    target = tmp_path / f"{case}.json"
    artifact = m.causal_design_save(result)
    m.causal_design_save(result, target)
    assert json.loads(target.read_text()) == artifact
    for restored in [m.causal_design_load(artifact), m.causal_design_load(target)]:
        assert list(restored) == list(result)
        assert restored.attrs == result.attrs
        for name in result:
            pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)
    pd.testing.assert_frame_equal(data, before)
    mutated = deepcopy(artifact)
    mutated["payload"]["attrs"]["state"]["estimates"][0] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        m.causal_design_load(mutated)


@pytest.mark.parametrize("case", CASES)
def test_cpu_float64_global_meta_and_rng_unchanged(case):
    expected = invoke(case)
    device, dtype, rng = (
        torch.get_default_device(),
        torch.get_default_dtype(),
        torch.get_rng_state().clone(),
    )
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        actual = invoke(case)
        assert (
            str(torch.get_default_device()) == "meta" and torch.get_default_dtype() == torch.float32
        )
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
    assert actual.attrs == expected.attrs
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "options",
    [
        {"missing": "drop"},
        {"device": "cuda"},
        {"weights": "w"},
        {"max_work": 0},
        {"level": 0},
        {"level": 1},
    ],
)
def test_unsupported_options_explicit(case, options):
    with pytest.raises(AnalysisError):
        invoke(case, **options)


@pytest.mark.parametrize("case", CASES)
def test_full_work_and_workspace_budget_before_sample_copy(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sample copied before complete admission")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        invoke(case, max_work=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        invoke(case, pd.concat([fixture()] * 16, ignore_index=True))


@pytest.mark.parametrize("case", CASES)
def test_dataset_and_long_labels_refused_before_selection(case, monkeypatch):
    dataset = object.__new__(Dataset)
    with pytest.raises(AnalysisError, match="Dataset"):
        invoke(case, dataset)
    data = fixture()
    data.index = ["x" * 1025] + list(range(5))

    def forbidden(*args, **kwargs):
        raise AssertionError("sample copied before identifier admission")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="1024"):
        invoke(case, data)


def test_escaped_stratum_label_bytes_preflight_actual_repeated_metadata_allowance():
    data = fixture()
    data["s"] = "\x00" * 1024
    result = invoke("stratified", data)
    byte_count = result.attrs["original_escaped_label_bytes"]
    assert byte_count >= 6 * 6144
    assert (
        result.attrs["computation_resource_plan"]["buffers"][
            "escaped_labels_full_metadata_tables_and_json"
        ]
        == 32 * byte_count
    )


@pytest.mark.parametrize("case", CASES)
def test_missing_outcomes_never_change_original_universe(case):
    data = fixture()
    data.iloc[0, data.columns.get_loc("y")] = math.nan
    with pytest.raises(AnalysisError, match="missing"):
        invoke(case, data)


def test_original_multiarm_topology_validated_before_missing_outcomes():
    data = fixture()
    data.iloc[0, data.columns.get_loc("y")] = math.nan
    data.iloc[0, data.columns.get_loc("a")] = 2
    with pytest.raises(AnalysisError, match="two observations"):
        invoke("complete", data)
    data = fixture()
    data["s"] = ["single"] + ["rest"] * 5
    with pytest.raises(AnalysisError, match="two observations"):
        invoke("stratified", data)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1, math.inf, math.nan])
def test_invalid_known_probabilities_not_clipped(p):
    data = fixture()
    data["p"] = p
    with pytest.raises(AnalysisError):
        invoke("bernoulli", data)


@pytest.mark.parametrize("arms", [None, [], [0, 1], [0, 1, 1], [0, 1, math.nan], list(range(9))])
def test_arms_explicit_finite_distinct_and_bounded(arms):
    with pytest.raises(AnalysisError):
        k.multiarm_neyman_ate(fixture(), "y", "a", arms=arms, design="complete_randomized")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contrasts": {}},
        {"contrasts": {"c": [1, -1]}},
        {"contrasts": {"c": [0, 0, 0]}},
        {"null_values": {"unknown": 0}},
    ],
)
def test_contrast_and_null_alignment_refused(kwargs):
    with pytest.raises(AnalysisError):
        invoke("complete", **kwargs)


@pytest.mark.parametrize("case", CASES)
def test_underflow_overflow_and_unrepresentable_ci_are_explicit(case):
    data = fixture()
    data["y"] *= 1e-200
    with pytest.raises(AnalysisError, match="underflows|rounded to zero"):
        invoke(case, data)
    data = fixture()
    data["y"] *= 1e200
    with pytest.raises(AnalysisError, match="finite float64|too large"):
        invoke(case, data)
    if case != "bernoulli":
        data = pd.DataFrame({"y": [1e100] * 48, "a": np.repeat([0, 1, 2], 16), "s": ["one"] * 48})
        data.loc[0, "y"] = math.nextafter(1e100, math.inf)
        with pytest.raises(AnalysisError, match="absorbed|confidence shift"):
            invoke(case, data, contrasts={"tiny": [-1.0, 1.0, 0.0]})


@pytest.mark.parametrize("case", CASES)
def test_unrepresented_original_integer_outcome_refused_before_copy(case, monkeypatch):
    data = fixture()
    data["y"] = [2**53 + 1, 1, 2, 3, 4, 5]

    def forbidden(*args, **kwargs):
        pytest.fail("integer outcome admission must precede selected-data copies")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="original integer outcome"):
        invoke(case, data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contrasts": {"c": [2**53 + 1, -(2**53 + 1), 0]}},
        {"null_values": {"difference[0,1]": 2**53 + 1, "difference[0,2]": 0, "difference[1,2]": 0}},
    ],
)
def test_unrepresented_integer_contrast_or_null_is_not_certified_after_rounding(kwargs):
    with pytest.raises(AnalysisError, match="original integer"):
        invoke("complete", **kwargs)
    with pytest.raises(AnalysisError, match="original integer null"):
        invoke("bernoulli", null_effect=2**53 + 1)


def test_nonbinary_real_controls_not_certified_after_float_conversion():
    third = Fraction(1, 3)
    with pytest.raises(AnalysisError, match="numeric contrast coefficient"):
        invoke("complete", contrasts={"third": [-third, third, 0]})
    with pytest.raises(AnalysisError, match="numeric contrast null"):
        invoke("complete", contrasts={"third": [-1, 1, 0]}, null_values={"third": third})
    with pytest.raises(AnalysisError, match="numeric null"):
        invoke("bernoulli", null_effect=third)
    accepted = invoke("complete", contrasts={"half": [-Fraction(1, 2), Fraction(1, 2), 0]})
    assert accepted.attrs["state"]["contrast_coefficients"] == [[-0.5, 0.5, 0.0]]


@pytest.mark.parametrize("case", ["complete", "stratified"])
def test_inexact_anchor_subtraction_preserves_original_contrast_and_all_group_state(case):
    data = pd.DataFrame(
        {"y": [1e16, 2.5, 2e16, 5.0, 2.5, 2.5], "a": [0, 0, 1, 1, 2, 2], "s": ["one"] * 6}
    )
    result = invoke(case, data, contrasts={"combined": [-2, 1, 1]})
    means = [
        sum((Fraction(float(v)) for v in data.y[data.a == arm]), Fraction()) / 2
        for arm in [0, 1, 2]
    ]
    reference = -2 * means[0] + means[1] + means[2]
    assert reference == Fraction(5, 2)
    assert result["effects"].estimate.iloc[-1] == float(reference)
    assert result["groups"].estimate.iloc[-1] == float(reference)
    state = result.attrs["state"]
    assert state["point_anchor"] == state["group_summaries"][0]["point_anchor"] == 0.0
    assert (
        state["estimates"][-1] == state["group_summaries"][0]["estimates"][-1] == float(reference)
    )
    restored = m.causal_design_load(m.causal_design_save(result))
    assert restored.attrs == result.attrs


def test_bernoulli_inexact_anchor_subtraction_preserves_small_original_ht_remainder():
    data = pd.DataFrame(
        {
            "y": [1e16, 2.5, 2e16, 5.0, 2.5, 2.5],
            "d": [0, 0, 1, 1, 0, 1],
            "p": [0.75, 0.75, 0.5, 0.5, 0.75, 0.5],
        }
    )
    reference = sum(
        (
            Fraction(float(y))
            / (Fraction(float(p)) if d else 1 - Fraction(float(p)))
            * (1 if d else -1)
            for y, d, p in zip(data.y, data.d, data.p, strict=True)
        ),
        Fraction(),
    ) / len(data)
    assert reference == -Fraction(5, 6)
    result = invoke("bernoulli", data)
    assert result["effects"].estimate.iloc[-1] == float(reference)
    assert result.attrs["state"]["point_anchor"] == 0.0
    assert m.causal_design_load(m.causal_design_save(result)).attrs == result.attrs


@pytest.mark.parametrize("case", CASES)
def test_compound_index_refused_before_generic_repr_and_copy(case, monkeypatch):
    data = fixture()
    data.index = pd.MultiIndex.from_tuples([("x" * 65536, i) for i in range(len(data))])
    monkeypatch.setattr(
        m,
        "sample",
        lambda *args, **kwargs: pytest.fail(
            "Compound index admission must precede selected copies."
        ),
    )
    with pytest.raises(AnalysisError, match="compound/custom labels"):
        invoke(case, data)


@pytest.mark.parametrize("case", CASES)
def test_wide_floating_dtype_domain_is_explicit_without_platform_skips(case, monkeypatch):
    data = fixture()
    data["y"] = data.y.astype(np.longdouble)
    if data.y.dtype.itemsize > 8:
        monkeypatch.setattr(
            m,
            "sample",
            lambda *args, **kwargs: pytest.fail("Wide input admission must precede copies."),
        )
        with pytest.raises(AnalysisError, match="wider than binary64"):
            invoke(case, data)
    else:
        # Some platforms alias longdouble to binary64. Its exact admission is
        # still exercised; an unavailable wider dtype is not a skipped test.
        np.testing.assert_array_equal(extract(invoke(case, data))[0], extract(invoke(case))[0])


def test_count_denominator_refusal_precedes_outcome_copy(monkeypatch):
    rows = []
    for group, counts in enumerate(
        [[2, 3, 5], [2, 7, 11], [2, 13, 17], [2, 19, 23], [2, 29, 31], [2, 37, 41]]
    ):
        for arm, count in enumerate(counts):
            rows.extend([(1.0, arm, group)] * count)
    data = pd.DataFrame(rows, columns=["y", "a", "s"])
    original_sample = m.sample

    def topology_only(data, columns, **options):
        assert "y" not in columns, (
            "oversized rational count certificate must be refused before outcome copies"
        )
        return original_sample(data, columns, **options)

    monkeypatch.setattr(m, "sample", topology_only)
    with pytest.raises(AnalysisError, match="common denominator"):
        invoke("stratified", data)


def test_ftz_daz_domain_refused_without_changing_global_mode_in_isolated_process():
    script = """
import pandas as pd
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.multiarm_neyman import bernoulli_neyman_ate
if not torch.set_flush_denormal(True):
    raise SystemExit(77)
tiny=torch.tensor(torch.finfo(torch.float64).tiny,dtype=torch.float64)
assert bool(tiny*.5 == 0)
try:
    bernoulli_neyman_ate(pd.DataFrame({'y':[1.,2.],'d':[0,1],'p':[.5,.5]}),
                        'y','d','p',design='bernoulli_randomized')
except AnalysisError as exc:
    assert 'gradual' in str(exc)
else:
    raise AssertionError('FTZ/DAZ arithmetic was accepted')
assert bool(tiny*.5 == 0), 'caller mode was changed'
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    if completed.returncode == 77:
        pytest.skip("platform does not expose flush-denormal control")
    assert completed.returncode == 0, completed.stderr
