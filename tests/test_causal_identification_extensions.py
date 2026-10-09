"""Independent completion, discrete-law covariance and constrained-tail oracles."""

import copy
import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.stats import norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.causal_design import common as m
from openecon.econometrics.causal_design import identification_extensions as ext
from openecon.econometrics.causal_design.partial_identification import lee_bounds
from openecon.resources import use_workspace_budget


def manski(data=None, **options):
    if data is None:
        data = pd.DataFrame(dict(y=[0.0, 0.3, 0.7, 1.0, 0.2], d=[0, 0, 1, 1, 1]))
    return ext.manski_ate_inference(
        data, "y", "d", **(dict(lower=0.0, upper=1.0, sampling="iid") | options)
    )


def lee_data():
    a = pd.DataFrame(
        dict(
            y=[0.0, 3.0, np.nan, np.nan, 1.0, 2.0, 2.0, 2.0, 10.0, np.nan, np.nan],
            d=[0] * 4 + [1] * 7,
            s=[1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            g=["A"] * 11,
        )
    )
    b = pd.DataFrame(
        dict(
            y=[20.0, 21.0, 22.0, 23.0, np.nan, np.nan, np.nan, np.nan, 22.0, 24.0, 26.0, 28.0],
            d=[0] * 8 + [1] * 4,
            s=[1] * 4 + [0] * 4 + [1] * 4,
            g=["B"] * 12,
        )
    )
    data = pd.concat([a, b], ignore_index=True)
    data.index = pd.Index(["duplicate", 1, "1", True, *range(len(data) - 4)], dtype=object)
    return data


def lee(data=None, **options):
    return ext.stratified_lee_bounds(
        lee_data() if data is None else data,
        "y",
        "d",
        "s",
        **(
            dict(
                strata="g",
                levels=["A", "B"],
                design="randomized_within_strata",
                monotonicity="increasing",
            )
            | options
        ),
    )


def lp_oracle(data, levels, decreasing=False):
    high, low = (0, 1) if decreasing else (1, 0)
    bounds, masses = [], []
    for label in levels:
        cell = data[data.g == label]
        counts = [sum(cell.d == arm) for arm in (0, 1)]
        values = [cell.loc[(cell.d == arm) & (cell.s == 1), "y"].to_numpy() for arm in (0, 1)]
        rates = [len(values[arm]) / counts[arm] for arm in (0, 1)]
        retained = rates[low] / rates[high] * len(values[high])
        coefficients = values[high] / retained
        constraint = np.ones((1, len(coefficients)))
        lo = linprog(coefficients, A_eq=constraint, b_eq=[retained], bounds=(0, 1), method="highs")
        hi = linprog(-coefficients, A_eq=constraint, b_eq=[retained], bounds=(0, 1), method="highs")
        assert lo.success and hi.success
        fixed = math.fsum(values[low]) / len(values[low])
        bounds.append(
            [lo.fun - fixed, -hi.fun - fixed] if high == 1 else [fixed + hi.fun, fixed - lo.fun]
        )
        masses.append(len(cell) / len(data) * rates[low])
    weights = np.array(masses) / math.fsum(masses)
    return np.array(bounds), weights, weights @ np.array(bounds)


def test_manski_exhaustive_completion_and_independent_endpoint_covariance():
    data = pd.DataFrame(dict(y=[-0.2, 0.5, 1.2, 0.9, 0.1], d=[0, 0, 0, 1, 1]))
    result = manski(data, lower=-0.5, upper=1.5)
    values = []
    for counterfactual in itertools.product([-0.5, 1.5], repeat=len(data)):
        values.append(
            math.fsum(
                (y - missing if d else missing - y)
                for y, d, missing in zip(data.y, data.d, counterfactual)
            )
            / len(data)
        )
    expected = [min(values), max(values)]
    state = result.attrs["state"]
    np.testing.assert_allclose(state["ate_bounds"], expected, atol=2e-16)
    score = np.array(
        [
            [d * (y - 1.5) + (1 - d) * (-0.5 - y), d * (y + 0.5) + (1 - d) * (1.5 - y)]
            for y, d in zip(data.y, data.d)
        ]
    )
    covariance = np.cov(score, rowvar=False, ddof=1) / len(data)
    np.testing.assert_allclose(state["endpoint_covariance"], covariance, rtol=2e-15)
    assert np.linalg.matrix_rank(state["endpoint_covariance"]) == state["covariance_rank"] == 1
    assert state["endpoint_covariance"][0] == state["endpoint_covariance"][1]
    critical = norm.ppf(0.975)
    np.testing.assert_allclose(
        state["confidence_region"],
        np.array(expected) + np.array([-1, 1]) * critical * np.sqrt(covariance.diagonal()),
        rtol=2e-15,
    )
    assert state["endpoint_standard_errors"][0] == state["endpoint_standard_errors"][1]
    for kind, endpoint in [("lower", expected[0]), ("upper", expected[1])]:
        y1 = state["extremizers"][f"y1_{kind}"]
        y0 = state["extremizers"]["y0_upper" if kind == "lower" else "y0_lower"]
        assert math.fsum(a - b for a, b in zip(y1, y0)) / len(data) == pytest.approx(endpoint)


def test_manski_exact_discrete_iid_law_all_sample_counts_and_rank_one():
    # Enumerate the entire N=4 IID distribution for a bounded three-point law.
    outcomes = [(0, 0.2), (1, 0.7), (1, 0.9)]
    probabilities = [0.25, 0.5, 0.25]
    population_scores = np.array([d * (y - 1) + (1 - d) * (-y) for d, y in outcomes])
    population_lower = np.dot(probabilities, population_scores)
    population_variance = np.dot(probabilities, (population_scores - population_lower) ** 2)
    score_means, expected_sample_variance, mass = [], 0.0, 0.0
    for indices in itertools.product(range(3), repeat=4):
        probability = math.prod(probabilities[j] for j in indices)
        scores = population_scores[list(indices)]
        score_means.append((probability, scores.mean()))
        expected_sample_variance += probability * np.var(scores, ddof=1) / 4
        mass += probability
        if len({outcomes[j][0] for j in indices}) == 2:
            result = manski(
                pd.DataFrame(
                    dict(d=[outcomes[j][0] for j in indices], y=[outcomes[j][1] for j in indices])
                )
            )
            assert result.attrs["state"]["endpoint_covariance"][0][0] == pytest.approx(
                np.var(scores, ddof=1) / 4, abs=2e-17
            )
    assert mass == pytest.approx(1)
    assert math.fsum(p * mean for p, mean in score_means) == pytest.approx(population_lower)
    assert math.fsum(
        p * (mean - population_lower) ** 2 for p, mean in score_means
    ) == pytest.approx(population_variance / 4)
    assert expected_sample_variance == pytest.approx(population_variance / 4)


def test_manski_wide_symmetric_support_preserves_small_scores_and_refuses_absorbed_CI():
    data = pd.DataFrame(dict(y=[0.0, 0.0, 1.0, 2.0], d=[0, 0, 1, 1]))
    score, midpoint = ext._center_scores(
        torch.tensor(data.y.to_numpy(), dtype=torch.float64),
        torch.tensor(data.d.to_numpy(), dtype=torch.float64),
        -1e100,
        1e100,
    )
    assert score.tolist() == [0.0, 0.0, 1.0, 2.0] and midpoint == [0.0, 0.0]
    assert float(ext._mean(score)) == 0.75
    covariance = float(ext._sum((score - 0.75).square()) / (4 * 3))
    assert covariance == pytest.approx(11 / 48)
    with pytest.raises(AnalysisError, match="confidence expansion"):
        manski(data, lower=-1e100, upper=1e100)


def test_manski_midpoint_low_remainder_retained_against_independent_fsum():
    a, b = -1e100, 1e150
    y = torch.tensor([b / 2, b / 2, b / 2], dtype=torch.float64)
    d = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    score, components = ext._center_scores(y, d, a, b)
    expected = [
        (2 * arm - 1) * math.fsum([value, -a / 2, -b / 2])
        for value, arm in zip(y.tolist(), d.tolist())
    ]
    assert score.tolist() == expected
    assert components[1] == -5e99


@pytest.mark.parametrize(
    "shift,scale", [(3.0, 2.0), (1e140, 1e130), (-1e140, 1e130), (0.0, 1e-140)]
)
def test_manski_translation_scale_with_independent_transformed_oracle(shift, scale):
    data = pd.DataFrame(dict(y=np.array([0.125, 0.375, 0.625, 0.875]), d=[0, 0, 1, 1]))
    data.y = shift + scale * data.y
    a, b = shift, shift + scale
    out = manski(data, lower=a, upper=b).attrs["state"]
    # Compute the law of the actual representable inputs, not ideal decimal transforms.
    oracle = np.array([d * (y - b) + (1 - d) * (a - y) for y, d in zip(data.y, data.d)])
    assert out["ate_bounds"][0] == pytest.approx(math.fsum(oracle) / 4, rel=2e-15, abs=1e-300)
    assert out["endpoint_covariance"][0][0] == pytest.approx(
        np.var(oracle, ddof=1) / 4, rel=2e-14, abs=1e-300
    )


@pytest.mark.parametrize("width", [0.0, 1.0])
def test_manski_explicit_zero_variance_and_singleton_support(width):
    data = pd.DataFrame(dict(y=[0.5] * 4, d=[0, 0, 1, 1]))
    out = manski(data, lower=0.5 - width / 2, upper=0.5 + width / 2).attrs["state"]
    assert out["endpoint_covariance"] == [[0.0, 0.0], [0.0, 0.0]]
    assert out["confidence_region"] == out["ate_bounds"]
    assert out["covariance_rank"] == 0
    assert out["zero_variance_status"] == (
        "singleton_support_identifies_zero" if width == 0 else "zero_empirical_endpoint_variance"
    )


@pytest.mark.parametrize("decreasing", [False, True])
def test_stratified_lee_independent_linear_program_and_unequal_arm_allocation(decreasing):
    data = lee_data()
    if decreasing:
        data.d = 1 - data.d
    out = lee(data, monotonicity="decreasing" if decreasing else "increasing")
    cell_bounds, weights, expected = lp_oracle(data, ["A", "B"], decreasing)
    state = out.attrs["state"]
    np.testing.assert_allclose(state["cell_ate_bounds"], cell_bounds, atol=4e-15)
    np.testing.assert_allclose(state["stratum_population_weights"], weights, atol=1e-16)
    np.testing.assert_allclose(state["ate_bounds"], expected, atol=4e-15)
    np.testing.assert_allclose(weights, [11 / 23, 12 / 23])
    assert not np.allclose(weights, [1 / 3, 2 / 3])
    np.testing.assert_allclose(state["row_contribution_bound_reconstruction"], expected, atol=4e-15)
    weights_by_row = np.array(state["population_mean_weights"])
    for arm in (0, 1):
        np.testing.assert_allclose(weights_by_row[data.d == arm].sum(0), [1, 1], atol=2e-16)
    assert state["covariance"] is state["standard_error"] is state["confidence_interval"] is None


def test_stratified_lee_fractional_tie_symmetry_and_one_level_existing_reduction():
    data = lee_data().iloc[:11].copy()
    out = lee(data, levels=["A"])
    prior = lee_bounds(data, "y", "d", "s", design="randomized", monotonicity="increasing")
    np.testing.assert_allclose(
        out.attrs["state"]["ate_bounds"], prior.attrs["state"]["ate_bounds"], atol=2e-16
    )
    state = out.attrs["state"]
    for index in [5, 6, 7]:
        assert (
            state["retention_lower_bound"][index]
            == state["retention_upper_bound"][index]
            == pytest.approx(5 / 6)
        )
    assert state["strata"][0]["retained_selected_mass"] == dict(
        numerator=14, denominator=4, value=3.5
    )


def test_prespecified_covariates_tighten_bounds_on_exact_population_law_fixture():
    data = pd.DataFrame(
        dict(
            y=[
                0.0,
                0.0,
                np.nan,
                np.nan,
                1.0,
                1.0,
                1.0,
                1.0,
                10.0,
                10.0,
                np.nan,
                np.nan,
                11.0,
                11.0,
                11.0,
                11.0,
            ],
            d=[0] * 4 + [1] * 4 + [0] * 4 + [1] * 4,
            s=[1, 1, 0, 0] + [1] * 4 + [1, 1, 0, 0] + [1] * 4,
            g=["A"] * 8 + ["B"] * 8,
        )
    )
    out = lee(data).attrs["state"]
    pooled = lee_bounds(data, "y", "d", "s", design="randomized", monotonicity="increasing").attrs[
        "state"
    ]
    assert out["ate_bounds"] == [1.0, 1.0]
    assert pooled["ate_bounds"] == [-4.0, 6.0]
    assert out["bound_width"] < pooled["bound_width"]


def test_stratified_lee_no_trim_and_exhaustive_integral_subset_oracle():
    data = lee_data().iloc[11:].copy()
    data.g = "A"
    output = lee(data, levels=["A"])
    selected = data[(data.d == 1) & (data.s == 1)].y.tolist()
    possible = [
        math.fsum(values) / 2 - math.fsum(data.loc[(data.d == 0) & (data.s == 1), "y"]) / 4
        for values in itertools.combinations(selected, 2)
    ]
    assert output.attrs["state"]["ate_bounds"] == pytest.approx([min(possible), max(possible)])
    data.s = 1
    data.y = np.arange(len(data), dtype=float)
    out = lee(data, levels=["A"]).attrs["state"]
    assert out["ate_bounds"][0] == out["ate_bounds"][1]
    assert out["retention_lower_bound"] == [1.0] * len(data)


@pytest.mark.parametrize("labels", [[1, "1"], [True, "True"], [1, 1.0], [False, 0]])
def test_typed_prespecified_strata_do_not_merge_coincident_text_or_numeric_labels(labels):
    data = lee_data()
    data.g = pd.Series([labels[0]] * 11 + [labels[1]] * 12, index=data.index, dtype=object)
    out = lee(data, levels=labels).attrs["state"]
    assert out["stratum_codes"] == [0] * 11 + [1] * 12
    assert len({m.canonical(v) for v in out["typed_stratum_levels"]}) == 2


def test_strata_permutation_and_outcome_location_scale_equivariance():
    data = lee_data()
    base = lee(data).attrs["state"]
    moved = lee(data.iloc[::-1]).attrs["state"]
    np.testing.assert_allclose(moved["ate_bounds"], base["ate_bounds"], atol=3e-15)
    data.y = data.y * 2 + 1000
    shifted = lee(data).attrs["state"]
    np.testing.assert_allclose(shifted["ate_bounds"], 2 * np.array(base["ate_bounds"]), atol=1e-13)


@pytest.mark.parametrize("location", [1e140, -1e140])
def test_stratified_lee_large_common_location_is_removed_before_signed_sums(location):
    # Different selected counts give different rounded 1/N arm weights. A
    # constant common location must still cancel algebraically, not become an
    # invented effect from differently rounded weighted products.
    data = pd.DataFrame(dict(y=[location] * 5, d=[0, 0, 1, 1, 1], s=[1] * 5, g=["A"] * 5))
    state = lee(data, levels=["A"]).attrs["state"]
    assert state["ate_bounds"] == [0.0, 0.0]
    assert state["row_signed_bound_contributions"] == [[0.0, 0.0]] * 5
    data.y = [location + v * 1e130 for v in [0.0, 1.0, 2.0, 3.0, 4.0]]
    represented = np.array(data.y) - float(data.y.iloc[0])
    expected = math.fsum(represented[2:]) / 3 - math.fsum(represented[:2]) / 2
    np.testing.assert_allclose(
        lee(data, levels=["A"]).attrs["state"]["ate_bounds"], [expected, expected], rtol=2e-15
    )


def test_stratified_lee_tiny_nonzero_trim_contribution_refused():
    data = pd.DataFrame(
        dict(y=[0.0, 0.0, math.ulp(0.0), math.ulp(0.0)], d=[0, 0, 1, 1], s=[1] * 4, g=["A"] * 4)
    )
    with pytest.raises(AnalysisError, match="underflow"):
        lee(data, levels=["A"])


def test_native_expansion_nested_cancellation_oracle_and_capacity_refusal():
    values = [1e100, 1e-100, 1e16, -1e100, -1e16]
    for start in range(len(values)):
        rotated = values[start:] + values[:start]
        assert float(ext._sum(torch.tensor(rotated, dtype=torch.float64))) == math.fsum(rotated)
    spaced = [math.ldexp(1.0, -50 * i) for i in range(20)]
    # Exercise the declared hard capacity explicitly, independently of scientific inputs.
    prior = ext._CAPACITY
    try:
        ext._CAPACITY = 2
        with pytest.raises(AnalysisError, match="capacity"):
            ext._sum(torch.tensor(spaced, dtype=torch.float64))
    finally:
        ext._CAPACITY = prior


@pytest.mark.parametrize("method", [manski, lee])
def test_full_json_typed_roundtrip_state_identity_and_latex(method, tmp_path):
    result = method()
    artifact = m.causal_design_save(result, tmp_path / "full.json")
    restored = m.causal_design_load(tmp_path / "full.json")
    assert m.causal_design_save(restored) == artifact
    assert restored.attrs == result.attrs
    for name in result:
        pd.testing.assert_frame_equal(result[name], restored[name], check_exact=True)
        assert "\\begin{tabular}" in str(restored[name].to_latex())
    changed = copy.deepcopy(artifact)
    changed["payload"]["attrs"]["state"]["assignment"][0] = 1
    with pytest.raises(AnalysisError):
        m.causal_design_load(changed)
    if method is lee:
        assert result.attrs["positions"] == list(range(23))
        assert result.attrs["unit_labels"][:4] == ["duplicate", 1, "1", True]


def test_ignored_unselected_outcomes_do_not_change_full_scientific_identity():
    data = lee_data()
    base = lee(data)
    data.loc[data.s == 0, "y"] = np.inf
    ignored = lee(data)
    assert m.causal_design_save(ignored) == m.causal_design_save(base)
    data.iloc[0, data.columns.get_loc("y")] = 0.25
    changed = lee(data)
    assert changed.attrs["sample_sha256"] == base.attrs["sample_sha256"]
    assert (
        changed.attrs["full_scientific_sample_sha256"]
        != base.attrs["full_scientific_sample_sha256"]
    )


@pytest.mark.parametrize("method", [manski, lee])
def test_private_rng_default_dtype_device_and_early_budget_admission(method, monkeypatch):
    torch.manual_seed(139)
    before = torch.random.get_rng_state().clone()
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        out = method()
        assert out.attrs["dtype"] == "float64"
        assert torch.equal(before, torch.random.get_rng_state())
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)

    def forbidden(*args, **kwargs):
        raise AssertionError("sample allocation preceded preflight")

    monkeypatch.setattr(m, "sample", forbidden)
    with pytest.raises(AnalysisError, match="max_work"):
        method(max_work=1)
    large = (
        pd.concat([lee_data()] * 20, ignore_index=True)
        if method is lee
        else pd.DataFrame(dict(y=[0.5] * 500, d=[0, 1] * 250))
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        method(large)


def test_strata_actual_JSON_escaped_label_copies_admitted_before_sample(monkeypatch):
    labels = ["\0" * 1024, "B"]
    data = lee_data()
    data.g = [labels[0]] * 11 + [labels[1]] * 12
    assert len(labels[0].encode()) == 1024
    assert len(m.canonical(ext._level(labels[0])).encode()) > 6000

    def forbidden(*args, **kwargs):
        raise AssertionError("sample allocation preceded escaped-label budget")

    monkeypatch.setattr(m, "sample", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        lee(data, levels=labels)


@pytest.mark.parametrize("method", [manski, lee])
def test_original_index_escaped_bytes_admitted_before_sample_and_large_int_identity(
    method, monkeypatch
):
    data = (
        lee_data()
        if method is lee
        else pd.DataFrame(dict(y=[0.1, 0.2, 0.3, 0.4, 0.5] * 5, d=[0, 0, 1, 1, 1] * 5))
    )
    data.index = pd.Index([2**63 - 1, 2**64 - 1, *range(len(data) - 2)], dtype=object)
    result = method(data)
    assert result.attrs["unit_labels"][:2] == [2**63 - 1, 2**64 - 1]
    data.index = ["\0" * 1024] * len(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("index identities copied before budget")

    monkeypatch.setattr(m, "sample", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        method(data)


@pytest.mark.parametrize("method", [manski, lee])
def test_bounded_saved_original_index_label_size(method):
    data = lee_data() if method is lee else pd.DataFrame(dict(y=[0.1, 0.2, 0.3], d=[0, 1, 1]))
    data.index = ["x" * 8192] * len(data)
    with pytest.raises(AnalysisError, match="index label"):
        method(data)


@pytest.mark.parametrize(
    "options",
    [
        dict(sampling="finite_population"),
        dict(missing="drop"),
        dict(device="mps"),
        dict(weights="w"),
        dict(lower=2.0, upper=1.0),
        dict(lower=True),
        dict(level=1.0),
        dict(level=0.0),
        dict(max_work=True),
    ],
)
def test_manski_unsupported_contracts_refuse(options):
    with pytest.raises(AnalysisError):
        manski(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(design="randomized"),
        dict(monotonicity="auto"),
        dict(missing="drop"),
        dict(device="cuda"),
        dict(weights="w"),
        dict(levels=[]),
        dict(levels=["A", "A"]),
        dict(levels=["A", "B", "C"]),
        dict(levels=["A"]),
        dict(levels=["A", float("inf")]),
        dict(levels="A"),
        dict(strata="d"),
    ],
)
def test_stratified_lee_unsupported_contracts_refuse(options):
    with pytest.raises(AnalysisError):
        lee(**options)


@pytest.mark.parametrize(
    "kind",
    [
        "assignment_missing",
        "selection_missing",
        "stratum_missing",
        "selected_missing",
        "selection_boolean",
        "treatment_boolean",
        "selected_infinite",
        "unknown_level",
        "incompatible",
        "empty_arm",
    ],
)
def test_stratified_lee_adversarial_topology_fails_whole_calculation(kind):
    data = lee_data()
    if kind == "assignment_missing":
        data.iloc[0, data.columns.get_loc("d")] = np.nan
    if kind == "selection_missing":
        data.iloc[3, data.columns.get_loc("s")] = np.nan
    if kind == "stratum_missing":
        data.iloc[3, data.columns.get_loc("g")] = None
    if kind == "selected_missing":
        data.iloc[0, data.columns.get_loc("y")] = np.nan
    if kind == "selection_boolean":
        data.s = data.s.astype(bool)
    if kind == "treatment_boolean":
        data.d = data.d.astype(bool)
    if kind == "selected_infinite":
        data.iloc[0, data.columns.get_loc("y")] = np.inf
    if kind == "unknown_level":
        data.iloc[3, data.columns.get_loc("g")] = "C"
    if kind == "incompatible":
        data.loc[data.g == "B", "s"] = 1
        data.iloc[-1, data.columns.get_loc("s")] = 0
    if kind == "empty_arm":
        data.loc[data.g == "B", "d"] = 1
    with pytest.raises(AnalysisError):
        lee(data)


@pytest.mark.parametrize(
    "data,options",
    [
        (pd.DataFrame(dict(y=[0.0, 1e300, 0.0], d=[0, 1, 1])), dict(upper=1e300)),
        (pd.DataFrame(dict(y=[1e-200, 2e-200, 3e-200], d=[0, 1, 1])), dict(upper=4e-200)),
        (pd.DataFrame(dict(y=[0.0, 0.5, 1.0], d=[0, 1, 1])), dict(level=1e-300)),
        (pd.DataFrame(dict(y=[0.0, 2.0, 1.0], d=[0, 1, 1])), {}),
        (pd.DataFrame(dict(y=[True, False, True], d=[0, 1, 1])), {}),
        (pd.DataFrame(dict(y=[0.0, 1.0, 1.0], d=[1, 1, 1])), {}),
        (pd.DataFrame(dict(y=[0.0, np.nan, 1.0], d=[0, 1, 1])), {}),
    ],
)
def test_manski_unrepresentable_variance_interval_or_invalid_data_refused(data, options):
    with pytest.raises(AnalysisError):
        manski(data, **options)


@pytest.mark.parametrize("method", [manski, lee])
def test_streaming_dataset_rejected_without_collection(method):
    dataset = object.__new__(Dataset)
    with pytest.raises(AnalysisError, match="Dataset"):
        method(dataset)


def test_manski_prespecified_medium_iid_bounded_dgp_diagnostic():
    rng = np.random.default_rng(1776)
    y = rng.binomial(1, 0.4, 1200).astype(float)
    d = rng.binomial(1, 0.35, 1200)
    state = manski(pd.DataFrame(dict(y=y, d=d))).attrs["state"]
    # Population L=.35*(.4-1)+.65*(-.4)=-.47; U=.53.
    true = np.array([-0.47, 0.53])
    assert np.all(
        np.abs(np.array(state["ate_bounds"]) - true)
        < 3 * np.array(state["endpoint_standard_errors"])
    )
    assert state["confidence_region"][0] < true[0] < true[1] < state["confidence_region"][1]
    assert "asymptotic" in state["inference"] and "entire" in state["inference"]
