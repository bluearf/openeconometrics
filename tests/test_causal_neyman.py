"""Independent assignment-universe and finite-population Neyman checks."""

from copy import deepcopy
from itertools import combinations, product
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design import common as m
from openecon.econometrics.causal_design import neyman as k
from openecon.resources import use_workspace_budget

TRANSFORM = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
CASES = [
    (k.neyman_ate, None, "complete_randomized"),
    (k.stratified_neyman_ate, "stratum", "stratified_randomized"),
    (k.cluster_neyman_ate, "cluster", "cluster_randomized"),
    (k.paired_neyman_ate, "pair", "paired_randomized"),
]


def fixture():
    return pd.DataFrame(
        {
            "y": [1.0, 4.0, 2.0, 7.0, 5.0, 8.0, 3.0, 9.0],
            "d": [0, 0, 1, 1, 0, 0, 1, 1],
            "stratum": ["a"] * 4 + ["b"] * 4,
            "pair": ["p0", "p1", "p0", "p1", "p2", "p3", "p2", "p3"],
            "cluster": ["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3"],
        },
        index=["r", "r", 2**60 + 1, 2**60 + 2, False, 1.5, "x", "y"],
    )


def invoke(case, data=None, **options):
    function, identity, design = case
    return function(
        fixture() if data is None else data,
        "y",
        "d",
        *([identity] if identity else []),
        **dict(design=design, **options),
    )


def complete_assignments(n, treated):
    for chosen in combinations(range(n), treated):
        assignment = np.zeros(n, dtype=int)
        assignment[list(chosen)] = 1
        yield assignment


def arm_oracle(outcomes, assignment):
    arms = [outcomes[assignment == arm] for arm in (0, 1)]
    means = np.array([math.fsum(arm) / len(arm) for arm in arms])
    bounds = np.array([np.var(arm, ddof=1) / len(arm) for arm in arms])
    return TRANSFORM @ means, TRANSFORM @ np.diag(bounds) @ TRANSFORM.T


def extract(result):
    return result["effects"]["estimate"].to_numpy(), result["covariance_bound"].to_numpy(
        dtype=float
    )


def check_law(estimates, bounds, truth, gap):
    estimates, bounds = np.asarray(estimates), np.asarray(bounds)
    np.testing.assert_allclose(estimates.mean(0), truth, rtol=2e-14, atol=2e-14)
    actual = np.cov(estimates, rowvar=False, ddof=0)
    observed_gap = bounds.mean(0) - actual
    np.testing.assert_allclose(observed_gap, gap, rtol=5e-13, atol=5e-13)
    assert np.linalg.eigvalsh(observed_gap).min() > -1e-12
    return actual


@pytest.mark.parametrize("n,n1", [(6, 2), (6, 3), (8, 3)])
@pytest.mark.parametrize("constant_effect", [False, True])
def test_complete_universe_unbiasedness_full_covariance_bound_and_heterogeneity_gap(
    n, n1, constant_effect
):
    y0 = np.array([-3.0, 2.0, 5.0, -1.0, 7.0, 4.0, 9.0, -6.0])[:n]
    effect = (
        np.full(n, 1.5)
        if constant_effect
        else np.array([2.0, -3.0, 1.0, 4.0, -2.0, 7.0, -4.0, 6.0])[:n]
    )
    y1 = y0 + effect
    estimates, bounds = [], []
    for assignment in complete_assignments(n, n1):
        outcome = np.where(assignment == 1, y1, y0)
        result = k.neyman_ate(
            pd.DataFrame({"y": outcome, "d": assignment}), "y", "d", design="complete_randomized"
        )
        estimate, bound = extract(result)
        independent_mean, independent_bound = arm_oracle(outcome, assignment)
        np.testing.assert_allclose(estimate, independent_mean, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(bound, independent_bound, rtol=2e-14, atol=2e-14)
        estimates.append(estimate)
        bounds.append(bound)
    potential = np.column_stack([y0, y1])
    actual = check_law(
        estimates,
        bounds,
        TRANSFORM @ potential.mean(0),
        TRANSFORM @ np.cov(potential, rowvar=False, ddof=1) @ TRANSFORM.T / n,
    )
    np.testing.assert_allclose(
        actual[2, 2],
        np.var(y0, ddof=1) / (n - n1) + np.var(y1, ddof=1) / n1 - np.var(effect, ddof=1) / n,
        rtol=2e-14,
    )
    if constant_effect:
        assert np.mean(bounds, axis=0)[2, 2] == pytest.approx(actual[2, 2])


def test_stratified_cartesian_universe_size_weights_and_full_bound():
    sizes, counts = [4, 5], [2, 2]
    y0 = np.array([-3.0, 4.0, 2.0, 7.0, -2.0, 5.0, 1.0, 8.0, -4.0])
    y1 = y0 + np.array([2.0, -1.0, 3.0, 4.0, -2.0, 6.0, 1.0, -3.0, 5.0])
    rows = [np.arange(4), np.arange(4, 9)]
    estimates, bounds = [], []
    for pieces in product(
        *[list(complete_assignments(n, n1)) for n, n1 in zip(sizes, counts, strict=True)]
    ):
        assignment = np.concatenate(pieces)
        outcome = np.where(assignment == 1, y1, y0)
        result = k.stratified_neyman_ate(
            pd.DataFrame({"y": outcome, "d": assignment, "s": ["a"] * 4 + ["b"] * 5}),
            "y",
            "d",
            "s",
            design="stratified_randomized",
        )
        estimate, bound = extract(result)
        means, matrices = zip(*[arm_oracle(outcome[r], assignment[r]) for r in rows], strict=True)
        np.testing.assert_allclose(
            estimate, sum(n / 9 * mean for n, mean in zip(sizes, means, strict=True)), atol=1e-14
        )
        np.testing.assert_allclose(
            bound,
            sum((n / 9) ** 2 * matrix for n, matrix in zip(sizes, matrices, strict=True)),
            atol=1e-14,
        )
        estimates.append(estimate)
        bounds.append(bound)
    potential = np.column_stack([y0, y1])
    gap = sum(
        (n / 9) ** 2 * TRANSFORM @ np.cov(potential[r], rowvar=False, ddof=1) @ TRANSFORM.T / n
        for n, r in zip(sizes, rows, strict=True)
    )
    check_law(estimates, bounds, TRANSFORM @ potential.mean(0), gap)


def test_unequal_cluster_size_ht_universe_targets_individual_average_not_cluster_average():
    sizes = [1, 2, 3, 4, 2]
    clusters = np.repeat(np.arange(len(sizes)), sizes)
    y0 = np.array([-3.0, 2.0, 4.0, 6.0, -1.0, 3.0, 5.0, 9.0, 2.0, 7.0, 1.0, 8.0])
    y1 = y0 + np.array([7.0, 1.0, 2.0, -4.0, 3.0, 5.0, 2.0, -1.0, 6.0, 4.0, -2.0, 3.0])
    scaled = (
        len(sizes)
        / len(y0)
        * np.array([[sum(y0[clusters == g]), sum(y1[clusters == g])] for g in range(len(sizes))])
    )
    estimates, bounds, naive = [], [], []
    for chosen in complete_assignments(len(sizes), 2):
        assignment = chosen[clusters]
        outcome = np.where(assignment == 1, y1, y0)
        result = k.cluster_neyman_ate(
            pd.DataFrame({"y": outcome, "d": assignment, "g": clusters}),
            "y",
            "d",
            "g",
            design="cluster_randomized",
        )
        estimate, bound = extract(result)
        np.testing.assert_allclose(
            estimate,
            arm_oracle(np.where(chosen == 1, scaled[:, 1], scaled[:, 0]), chosen)[0],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            bound,
            arm_oracle(np.where(chosen == 1, scaled[:, 1], scaled[:, 0]), chosen)[1],
            atol=1e-14,
        )
        assert result.attrs["state"]["n_clusters"] == 5
        np.testing.assert_array_equal(result["clusters"]["cluster_size"], sizes)
        estimates.append(estimate)
        bounds.append(bound)
        naive.append(outcome[assignment == 1].mean() - outcome[assignment == 0].mean())
    truth = TRANSFORM @ np.column_stack([y0, y1]).mean(0)
    check_law(
        estimates,
        bounds,
        truth,
        TRANSFORM @ np.cov(scaled, rowvar=False, ddof=1) @ TRANSFORM.T / len(sizes),
    )
    assert abs(np.mean(naive) - truth[2]) > 0.01


def test_fair_pair_exhaustive_matrix_expectation_gap_and_cross_arm_covariance():
    y0 = np.array([-3.0, 2.0, 4.0, 8.0, -1.0, 5.0, 9.0, 2.0])
    y1 = y0 + np.array([7.0, -1.0, 3.0, -4.0, 2.0, 6.0, -2.0, 5.0])
    pairs = np.repeat(np.arange(4), 2)
    estimates, bounds = [], []
    for bits in product((0, 1), repeat=4):
        assignment = np.array([value for bit in bits for value in (bit, 1 - bit)])
        outcome = np.where(assignment == 1, y1, y0)
        result = k.paired_neyman_ate(
            pd.DataFrame({"y": outcome, "d": assignment, "pair": pairs}),
            "y",
            "d",
            "pair",
            design="paired_randomized",
        )
        vectors = (
            np.array(
                [
                    [
                        outcome[(pairs == g) & (assignment == 0)][0],
                        outcome[(pairs == g) & (assignment == 1)][0],
                    ]
                    for g in range(4)
                ]
            )
            @ TRANSFORM.T
        )
        estimate, bound = extract(result)
        np.testing.assert_allclose(estimate, vectors.mean(0), atol=1e-14)
        np.testing.assert_allclose(bound, np.cov(vectors, rowvar=False, ddof=1) / 4, atol=1e-14)
        estimates.append(estimate)
        bounds.append(bound)
    pair_means = (
        np.array([[y0[pairs == g].mean(), y1[pairs == g].mean()] for g in range(4)]) @ TRANSFORM.T
    )
    actual = check_law(
        estimates,
        bounds,
        TRANSFORM @ np.column_stack([y0, y1]).mean(0),
        np.cov(pair_means, rowvar=False, ddof=1) / 4,
    )
    assert actual[0, 1] != 0
    assert any(abs(bound[0, 1]) > 0.01 for bound in bounds)


def test_conservative_is_not_sample_by_sample_or_exact_finite_sample_claim():
    y0 = np.array([0.0, 1.0, 2.0, 10.0, 15.0, 20.0])
    y1 = np.array([20.0, 15.0, 10.0, 2.0, 1.0, 0.0])
    estimates, bounds = [], []
    for assignment in complete_assignments(6, 3):
        result = k.neyman_ate(
            pd.DataFrame({"y": np.where(assignment == 1, y1, y0), "d": assignment}),
            "y",
            "d",
            design="complete_randomized",
        )
        estimate, bound = extract(result)
        estimates.append(estimate[2])
        bounds.append(bound[2, 2])
        assert result.attrs["state"]["true_randomization_covariance"] is None
        assert result.attrs["state"]["covariance_identified"] is False
    assert min(bounds) < np.var(estimates)
    assert np.mean(bounds) >= np.var(estimates)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_normal_inference_null_and_pointwise_intervals_independent_oracle(case, alternative):
    result = invoke(case, null_effect=0.25, alternative=alternative, level=0.9)
    effects = result["effects"]
    estimates, bound = extract(result)
    errors = np.sqrt(np.diag(bound))
    critical = norm.ppf(0.95)
    np.testing.assert_allclose(effects["std_error"], errors, rtol=2e-13)
    np.testing.assert_allclose(effects["ci_low"], estimates - critical * errors, rtol=2e-13)
    np.testing.assert_allclose(effects["ci_high"], estimates + critical * errors, rtol=2e-13)
    z = (estimates[2] - 0.25) / errors[2]
    expected = (
        2 * norm.sf(abs(z))
        if alternative == "two-sided"
        else norm.sf(z if alternative == "greater" else -z)
    )
    assert effects.iloc[2]["z"] == pytest.approx(z, rel=2e-13)
    assert effects.iloc[2]["p_value"] == pytest.approx(expected, rel=2e-12)
    assert pd.isna(effects.iloc[0]["p_value"]) and pd.isna(effects.iloc[1]["null_value"])
    assert effects["df"].isna().all()


@pytest.mark.parametrize("case", CASES)
def test_full_public_and_path_persistence_types_positions_and_private_inputs(case, tmp_path):
    source = fixture()
    before = source.copy(deep=True)
    result = invoke(case, source)
    function, identity, _ = case
    assert (
        getattr(oe, function.__name__)(
            source, "y", "d", *([identity] if identity else []), design=case[2]
        ).attrs["state"]
        == result.attrs["state"]
    )
    artifact = m.causal_design_save(result)
    target = tmp_path / (function.__name__ + ".json")
    m.causal_design_save(result, target)
    assert json.loads(target.read_text()) == artifact
    for restored in (m.causal_design_load(artifact), m.causal_design_load(target)):
        assert list(restored) == list(result)
        assert restored.attrs == result.attrs
        for key in result:
            pd.testing.assert_frame_equal(restored[key], result[key], check_exact=True)
    assert result.attrs["positions"] == list(range(8))
    assert result.attrs["unit_labels"][2] != result.attrs["unit_labels"][3]
    pd.testing.assert_frame_equal(source, before)
    bad = deepcopy(artifact)
    bad["payload"]["attrs"]["state"]["estimated_covariance_bound"][0][0] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        m.causal_design_load(bad)


@pytest.mark.parametrize("case", CASES)
def test_global_meta_default_and_rng_unchanged(case):
    expected = invoke(case)
    rng = torch.get_rng_state().clone()
    device = torch.get_default_device()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        actual = invoke(case)
        assert str(torch.get_default_device()) == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
    assert actual.attrs["state"] == expected.attrs["state"]
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cuda"},
        {"weights": "w"},
        {"missing": "drop"},
        {"alternative": "bad"},
        {"level": 0},
        {"level": 1},
        {"null_effect": math.inf},
        {"max_work": 0},
    ],
)
def test_unsupported_options(case, kwargs):
    with pytest.raises(AnalysisError):
        invoke(case, **kwargs)


@pytest.mark.parametrize("case", CASES)
def test_design_declaration_required_and_refuses_inferred_design(case):
    function, identity, _ = case
    args = [fixture(), "y", "d"] + ([identity] if identity else [])
    with pytest.raises((TypeError, AnalysisError)):
        function(*args)
    with pytest.raises(AnalysisError, match="declare"):
        function(*args, design="observational")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "column,value",
    [("y", math.nan), ("y", math.inf), ("y", "bad"), ("d", 2), ("d", math.nan), ("d", True)],
)
def test_invalid_original_numeric_sample(case, column, value):
    data = fixture()
    if isinstance(value, str) or isinstance(value, bool):
        data[column] = data[column].astype(object)
    data.iloc[0, data.columns.get_loc(column)] = value
    with pytest.raises(AnalysisError):
        invoke(case, data)


@pytest.mark.parametrize("case", CASES)
def test_complete_preflight_precedes_sample_allocation(case, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sample was allocated")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        invoke(case, max_work=1)


@pytest.mark.parametrize("case", CASES)
def test_complete_workspace_preflight_precedes_sample_allocation(case, monkeypatch):
    def no_sample(*args, **kwargs):
        raise AssertionError("sample was allocated")

    def refusal(*args, **kwargs):
        raise AnalysisError("workspace_budget_exceeded", "full workspace refused")

    monkeypatch.setattr(m, "sample", no_sample)
    monkeypatch.setattr(k, "plan_workspace", refusal)
    with pytest.raises(AnalysisError, match="full workspace"):
        invoke(case)


@pytest.mark.parametrize("case", CASES)
def test_actual_workspace_budget_and_dataset_refuse_before_collection(case, monkeypatch):
    function, identity, design = case
    dataset = object.__new__(Dataset)
    with pytest.raises(AnalysisError, match="Dataset"):
        function(dataset, "y", "d", *([identity] if identity else []), design=design)

    def no_sample(*args, **kwargs):
        raise AssertionError("sample was allocated")

    monkeypatch.setattr(m, "sample", no_sample)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        invoke(case, pd.concat([fixture()] * 40, ignore_index=True))


@pytest.mark.parametrize("case", CASES)
def test_sample_has_fixed_universe_and_no_deletion(case):
    data = fixture()
    data.loc[:, "y"] = 2.0
    result = invoke(case, data)
    assert result.attrs["n_missing"] == 0
    assert result["effects"]["degenerate_variance"].all()
    assert result["effects"]["p_value"].isna().all()
    np.testing.assert_array_equal(result["covariance_bound"], np.zeros((3, 3)))


@pytest.mark.parametrize("case", CASES[1:])
def test_identifier_role_required_and_distinct(case):
    function, _, design = case
    for bad in (None, "y", "d"):
        with pytest.raises(AnalysisError):
            function(fixture(), "y", "d", bad, design=design)


def test_topology_is_validated_before_missing_outcome():
    data = fixture()
    data.iloc[0, data.columns.get_loc("y")] = math.nan
    data.iloc[0, data.columns.get_loc("pair")] = "broken"
    with pytest.raises(AnalysisError, match="Every original pair"):
        invoke(CASES[3], data)
    data = fixture()
    data.iloc[0, data.columns.get_loc("y")] = math.nan
    data.iloc[0, data.columns.get_loc("d")] = 1
    with pytest.raises(AnalysisError, match="constant within"):
        invoke(CASES[2], data)


def test_arm_and_cluster_minima_and_stratum_singleton_refused():
    data = fixture()
    data["d"] = [1, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(AnalysisError, match="at least two units"):
        invoke(CASES[0], data)
    data = fixture()
    data["stratum"] = ["solo"] + ["rest"] * 7
    with pytest.raises(AnalysisError, match="at least two units"):
        invoke(CASES[1], data)
    data = fixture()
    data["cluster"] = ["one"] * 4 + ["two"] * 4
    data["d"] = [0] * 4 + [1] * 4
    with pytest.raises(AnalysisError, match="two original clusters"):
        invoke(CASES[2], data)


@pytest.mark.parametrize("case", CASES[1:])
def test_typed_group_identifiers_large_ints_preserved(case):
    function, identity, design = case
    data = fixture()
    labels = [True, 1, "1", 2**60 + 1]
    if identity == "stratum":
        data = pd.concat(
            [
                data.iloc[:4].assign(**{identity: pd.Series([label] * 4, dtype=object).to_numpy()})
                for label in labels
            ],
            ignore_index=True,
        )
    elif identity == "cluster":
        data[identity] = pd.Series(
            [label for label in labels for _ in range(2)], index=data.index, dtype=object
        )
    else:
        data[identity] = pd.Series(
            [True, 1, True, 1, "1", 2**60 + 1, "1", 2**60 + 1], index=data.index, dtype=object
        )
    result = function(data, "y", "d", identity, design=design)
    groups = result.attrs["state"]["design_groups"]
    assert len(groups) == 4
    assert [group["kind"] for group in groups] == ["bool", "int", "str", "int"]
    key = {"stratum": "groups", "cluster": "clusters", "pair": "pairs"}[identity]
    assert result[key]["identifier"].dtype == object
    assert result[key]["identifier"].iloc[-1] == 2**60 + 1
    pd.testing.assert_frame_equal(
        m.causal_design_load(m.causal_design_save(result))[key], result[key], check_exact=True
    )


def test_overflow_and_nonzero_variance_underflow_refused():
    data = fixture()
    data["y"] = 1e308
    with pytest.raises(AnalysisError, match="finite float64|too large"):
        invoke(CASES[0], data)
    data = fixture()
    data["y"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) * 1e-200
    with pytest.raises(AnalysisError, match="underflows|rounded to zero"):
        invoke(CASES[0], data)


def test_nested_cancellation_mean_is_not_falsely_zero():
    control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    treated = [1e70, 1e-70, 1e10, -1e70, -1e10, 0.0]
    result = k.neyman_ate(
        pd.DataFrame({"y": control + treated, "d": [0] * 6 + [1] * 6}),
        "y",
        "d",
        design="complete_randomized",
    )
    assert result["effects"]["estimate"].iloc[2] == math.fsum(treated) / 6
    assert result["effects"]["estimate"].iloc[2] != 0


def test_absorbed_pair_or_null_contribution_refused():
    data = fixture()
    data.iloc[0, data.columns.get_loc("y")] = 1e100
    with pytest.raises(AnalysisError, match="absorbed"):
        invoke(CASES[3], data)
    with pytest.raises(AnalysisError, match="absorbed"):
        invoke(CASES[0], null_effect=1e100)


def test_original_row_cluster_ht_expansion_preserves_remainder_lost_by_cluster_total_rounding():
    data = pd.DataFrame(
        {
            "y": [1e16, 1.0, -1e16, 0.0, 0.0, 0.0],
            "d": [1, 1, 1, 1, 0, 0],
            "g": ["a", "a", "b", "b", "c", "d"],
        }
    )
    result = k.cluster_neyman_ate(data, "y", "d", "g", design="cluster_randomized")
    assert result["effects"]["estimate"].iloc[1] == result["effects"]["estimate"].iloc[2] == 1 / 3
    contributions = result.attrs["state"]["original_row_mean_contributions"]
    assert math.fsum(row[2] for row in contributions) == 1 / 3
    assert result["effects"]["std_error"].iloc[2] > 0


def test_original_row_stratum_expansion_preserves_remainder_across_large_cell_means():
    data = pd.DataFrame(
        {
            "y": [0.0, 0.0, 1e16, 1.0, 0.0, 0.0, -1e16, 0.0],
            "d": [0, 0, 1, 1] * 2,
            "s": ["a"] * 4 + ["b"] * 4,
        }
    )
    result = k.stratified_neyman_ate(data, "y", "d", "s", design="stratified_randomized")
    assert result["effects"]["estimate"].iloc[1] == result["effects"]["estimate"].iloc[2] == 0.25
    assert result["effects"]["std_error"].iloc[2] > 0


@pytest.mark.parametrize("case", CASES)
def test_long_index_label_refused_before_sample_copy(case, monkeypatch):
    data = fixture()
    data.index = ["x" * 1025] + list(range(7))

    def forbidden(*args, **kwargs):
        raise AssertionError("sample was allocated")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="1024"):
        invoke(case, data)


@pytest.mark.parametrize("case", CASES[1:])
def test_long_group_label_refused_before_sample_copy(case, monkeypatch):
    data = fixture()
    data[case[1]] = "x" * 1025

    def forbidden(*args, **kwargs):
        raise AssertionError("sample was allocated")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="1024"):
        invoke(case, data)


def test_control_character_identifier_escaped_bytes_included_in_complete_preflight():
    data = fixture()
    data["stratum"] = ["\x00" * 1024] * 4 + ["\x01" * 1024] * 4
    result = invoke(CASES[1], data)
    metadata = result.attrs
    actual_bytes = (
        sum(len(m.canonical(m._label(value))) for value in data.index)
        + sum(len(m.canonical(value)) for value in data.stratum)
        + sum(len(m.canonical(value)) for value in ["y", "d", "stratum"])
    )
    assert metadata["original_escaped_label_bytes"] == actual_bytes
    assert (
        metadata["computation_resource_plan"]["buffers"][
            "escaped_labels_full_metadata_tables_and_json"
        ]
        == 16 * actual_bytes
    )


@pytest.mark.parametrize("value", [0.1, 1.0, 1e100])
@pytest.mark.parametrize("case", CASES)
def test_constant_outcomes_unequal_counts_do_not_invent_effect_or_variance(case, value):
    function, identity, design = case
    if identity is None:
        data = pd.DataFrame({"y": [value] * 5, "d": [0, 0, 1, 1, 1]})
    elif identity == "stratum":
        data = pd.DataFrame(
            {
                "y": [value] * 11,
                "d": [0, 0, 1, 1, 1] + [0, 0, 1, 1, 1, 1],
                "stratum": ["a"] * 5 + ["b"] * 6,
            }
        )
    elif identity == "cluster":
        data = pd.DataFrame(
            {
                "y": [value] * 10,
                "d": [0] * 4 + [1] * 6,
                "cluster": [str(i) for i in range(5) for _ in range(2)],
            }
        )
    else:
        data = pd.DataFrame(
            {
                "y": [value] * 6,
                "d": [0, 1] * 3,
                "pair": [str(i) for i in range(3) for _ in range(2)],
            }
        )
    result = function(data, "y", "d", *([identity] if identity else []), design=design)
    np.testing.assert_array_equal(result["effects"]["estimate"], [value, value, 0.0])
    np.testing.assert_array_equal(result["covariance_bound"], np.zeros((3, 3)))
    assert result["effects"]["degenerate_variance"].all()


def test_original_row_cluster_reference_differences_do_not_invent_zero_variance():
    data = pd.DataFrame(
        {
            "y": [1e16, 1.0, 1e16, 2.0, 0.0, 0.0],
            "d": [1, 1, 1, 1, 0, 0],
            "g": ["a", "a", "b", "b", "c", "d"],
        }
    )
    result = k.cluster_neyman_ate(data, "y", "d", "g", design="cluster_randomized")
    assert result["covariance_bound"].iloc[1, 1] == pytest.approx(1 / 9, rel=2e-14)
    assert result["covariance_bound"].iloc[2, 2] == pytest.approx(1 / 9, rel=2e-14)
    state = result.attrs["state"]
    np.testing.assert_allclose(
        state["reference_centered_scaled_cluster_totals"], [0.0, 2 / 3, 0.0, 0.0], atol=1e-16
    )


@pytest.mark.parametrize("value", [0.1, 1.0, 1e100])
@pytest.mark.parametrize("stratified", [False, True])
def test_full_artifact_constant_group_points_remain_exact_with_unequal_arm_counts(
    value, stratified
):
    data = pd.DataFrame({"y": [value] * 5, "d": [0, 0, 1, 1, 1]})
    if stratified:
        data = pd.concat([data.assign(s="a"), data.assign(s="b")], ignore_index=True)
        result = k.stratified_neyman_ate(data, "y", "d", "s", design="stratified_randomized")
    else:
        result = k.neyman_ate(data, "y", "d", design="complete_randomized")
    restored = m.causal_design_load(m.causal_design_save(result))
    for artifact in (result, restored):
        state = artifact.attrs["state"]
        np.testing.assert_array_equal(artifact["effects"]["estimate"], [value, value, 0.0])
        assert state["estimates"] == [value, value, 0.0]
        for row, summary in zip(
            artifact["groups"].itertuples(), state["group_summaries"], strict=True
        ):
            assert [row.mean_control, row.mean_treated, row.difference] == [value, value, 0.0]
            assert summary["estimates"] == [value, value, 0.0]
            assert summary["mean_anchor"] == value
            assert summary["mean_anchor_coefficients"] == [1.0, 1.0, 0.0]
            assert summary["mean_anchor_contributions"] == [value, value, 0.0]
            assert all(score == [0.0, 0.0, 0.0] for score in summary["mean_contributions"])
            np.testing.assert_array_equal(summary["covariance_bound"], np.zeros((3, 3)))
        assert artifact["effects"]["degenerate_variance"].all()
    assert restored.attrs == result.attrs
    for name in result:
        pd.testing.assert_frame_equal(restored[name], result[name], check_exact=True)


def test_full_artifact_group_contrast_retains_remainder_hidden_in_rounded_arm_means():
    # The arm means round to the same float64 value, but their original-row
    # contrast is representable and must remain .5 in every scientific layer.
    data = pd.DataFrame({"y": [1e16, 0.0, 1e16, 1.0], "d": [0, 0, 1, 1]})
    result = k.neyman_ate(data, "y", "d", design="complete_randomized")
    for artifact in (result, m.causal_design_load(m.causal_design_save(result))):
        assert artifact["effects"]["estimate"].iloc[2] == 0.5
        assert artifact["groups"]["difference"].iloc[0] == 0.5
        assert artifact.attrs["state"]["group_summaries"][0]["estimates"][2] == 0.5
        assert (
            artifact["groups"]["mean_control"].iloc[0] == artifact["groups"]["mean_treated"].iloc[0]
        )


@pytest.mark.parametrize("value", [0.1, 1.0, 1e100])
def test_full_artifact_constant_unequal_cluster_sizes_keep_true_nonzero_ht_point(value):
    sizes = [1, 2, 3, 4]
    data = pd.DataFrame(
        {
            "y": [value] * sum(sizes),
            "d": [0] * 3 + [1] * 7,
            "g": [i for i, size in enumerate(sizes) for _ in range(size)],
        }
    )
    result = k.cluster_neyman_ate(data, "y", "d", "g", design="cluster_randomized")
    expected = np.array([0.6, 1.4, 0.8]) * value
    # Fixed-N cluster HT means need not match under a constant outcome: the
    # observed arms contain different fractions of the original individuals.
    for artifact in (result, m.causal_design_load(m.causal_design_save(result))):
        state = artifact.attrs["state"]
        np.testing.assert_allclose(artifact["effects"]["estimate"], expected, rtol=2e-15)
        np.testing.assert_array_equal(state["estimates"], artifact["effects"]["estimate"])
        assert state["point_anchor_coefficients"] == [0.6, 1.4, 0.8]
        assert state["point_anchor"] == value
        assert state["estimates"][2] > 0
        assert artifact["effects"]["std_error"].iloc[2] > 0
        np.testing.assert_array_equal(
            artifact["clusters"]["observed_total"],
            [math.fsum([value] * size) for size in sizes],
        )
        np.testing.assert_array_equal(state["cluster_units"], artifact["clusters"].iloc[:, 4:])
