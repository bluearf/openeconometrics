"""Independent marginal-distribution oracles and complete-artifact contracts."""

import copy
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.distribution import treatment_cdf, treatment_quantile
from openecon.resources import use_workspace_budget


def sample():
    return pd.DataFrame(
        {
            "y": [-2.0, -1.0, 0.0, 1.0, 3.0, 6.0, -1.0, 2.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
            "d": [0] * 6 + [1] * 8,
        },
        index=[f"person-{i}" for i in range(14)],
    )


def run(data=None, quantile=False, **options):
    data = sample() if data is None else data
    method = treatment_quantile if quantile else treatment_cdf
    targets = (
        {"quantiles": [0.2, 0.5, 0.8], "reps": 99} if quantile else {"thresholds": [-1.0, 2.0, 5.0]}
    )
    return method(data, "y", "d", design="randomized", **(targets | options))


def test_cdf_independent_joint_covariance_and_pointwise_normal_oracle():
    data = sample()
    grid = [5.0, -1.0, 2.0, 100.0]
    output = run(data, thresholds=grid, level=0.9)
    groups = [data.loc[data.d == arm, "y"].to_numpy() for arm in (0, 1)]
    indicators = [(values[:, None] <= np.array(grid)).astype(float) for values in groups]
    cdfs = [values.mean(0) for values in indicators]
    covariance_arms = [np.cov(values, rowvar=False, ddof=1) / len(values) for values in indicators]
    covariance = sum(covariance_arms)
    effect = cdfs[1] - cdfs[0]
    se = np.sqrt(covariance.diagonal())
    table = output["effects"]
    np.testing.assert_allclose(table.cdf0, cdfs[0], atol=1e-15)
    np.testing.assert_allclose(table.cdf1, cdfs[1], atol=1e-15)
    np.testing.assert_allclose(table.estimate, effect, atol=1e-15)
    np.testing.assert_allclose(output["covariance"], covariance, atol=1e-15)
    np.testing.assert_allclose(
        output.attrs["state"]["arm_covariances"], covariance_arms, atol=1e-15
    )
    np.testing.assert_allclose(table.std_error, se, atol=1e-15)
    np.testing.assert_allclose(table.ci_low, effect - stats.norm.ppf(0.95) * se, atol=1e-15)
    np.testing.assert_allclose(table.ci_high, effect + stats.norm.ppf(0.95) * se, atol=1e-15)
    np.testing.assert_allclose(
        table.p_value.iloc[:3], 2 * stats.norm.sf(abs(effect[:3] / se[:3])), atol=1e-15
    )
    assert table.loc[3, "inference_status"] == "zero_empirical_variance"
    assert pd.isna(table.loc[3, "p_value"]) and pd.isna(table.loc[3, "z"])
    assert list(table.threshold) == grid
    assert output.attrs["state"]["conditioning"] == "observed arm counts"


def test_cdf_inclusive_ties_monotonic_marginal_cdfs_and_shift():
    data = sample()
    data.loc[data.d == 1, "y"] = [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    grid = [-100.0, -2.0, -1.0, 0.0, 2.0, 4.0, 100.0]
    output = run(data, thresholds=grid)
    assert (np.diff(output["effects"].cdf0) >= 0).all()
    assert (np.diff(output["effects"].cdf1) >= 0).all()
    assert output["effects"].loc[4, "cdf1"] == 3 / 8
    shifted = data.copy()
    shifted.y += 37
    shifted_output = run(shifted, thresholds=[v + 37 for v in grid])
    np.testing.assert_array_equal(output["effects"].estimate, shifted_output["effects"].estimate)
    np.testing.assert_array_equal(output["covariance"], shifted_output["covariance"])


def test_cdf_zero_variance_exposes_undefined_test_without_epsilon():
    data = pd.DataFrame({"y": [0.0] * 3 + [5.0] * 3, "d": [0] * 3 + [1] * 3})
    output = run(data, thresholds=[-1.0, 2.0, 8.0])
    table = output["effects"]
    assert list(table.estimate) == [0.0, -1.0, 0.0]
    assert list(table.std_error) == [0.0, 0.0, 0.0]
    assert table.z.isna().all() and table.p_value.isna().all()
    np.testing.assert_array_equal(table.ci_low, table.estimate)
    np.testing.assert_array_equal(table.ci_high, table.estimate)
    assert output.attrs["state"]["degenerate_grid_indices"] == [0, 1, 2]
    assert any("unsampled tail" in note for note in output.attrs["notes"])


def _independent_bootstrap_output(data, probabilities, draws):
    groups = [data.loc[data.d == arm, "y"].to_numpy() for arm in (0, 1)]
    points = [np.quantile(values, probabilities, method="inverted_cdf") for values in groups]
    replicates = [
        np.quantile(values[draw], probabilities, axis=1, method="inverted_cdf").T
        for values, draw in zip(groups, draws)
    ]
    effects = replicates[1] - replicates[0]
    covariance = np.cov(np.concatenate([*replicates, effects], axis=1), rowvar=False, ddof=1)
    return points, replicates, effects, covariance


def test_quantile_numpy_draws_independent_quantile_covariance_percentile_oracle(monkeypatch):
    data, probabilities, reps = sample(), [0.75, 0.1, 0.5], 303
    rng = np.random.default_rng(9824)
    draws = [rng.integers(0, size, size=(reps, size)) for size in (6, 8)]
    requested = []

    def independent_draw(high, size, *, generator, device):
        arm = len(requested)
        assert generator.initial_seed() == 41 and device == "cpu"
        assert high == (6, 8)[arm] and size == (reps, high)
        requested.append(arm)
        return torch.tensor(draws[arm], dtype=torch.int64)

    monkeypatch.setattr(torch, "randint", independent_draw)
    output = run(data, quantile=True, quantiles=probabilities, reps=reps, seed=41, level=0.8)
    points, replicates, effects, covariance = _independent_bootstrap_output(
        data, probabilities, draws
    )
    table, state = output["effects"], output.attrs["state"]
    np.testing.assert_array_equal(table.q0, points[0])
    np.testing.assert_array_equal(table.q1, points[1])
    np.testing.assert_array_equal(table.estimate, points[1] - points[0])
    np.testing.assert_array_equal(state["bootstrap_q0"], replicates[0])
    np.testing.assert_array_equal(state["bootstrap_q1"], replicates[1])
    np.testing.assert_array_equal(state["bootstrap_effects"], effects)
    np.testing.assert_allclose(output["joint_covariance"], covariance, atol=2e-14)
    np.testing.assert_allclose(output["covariance"], covariance[6:, 6:], atol=2e-14)
    np.testing.assert_allclose(table.std_error, np.std(effects, axis=0, ddof=1), atol=2e-14)
    intervals = np.quantile(effects, [0.1, 0.9], axis=0, method="linear")
    np.testing.assert_allclose(table.ci_low, intervals[0], atol=2e-14)
    np.testing.assert_allclose(table.ci_high, intervals[1], atol=2e-14)
    assert table.p_value.isna().all() and state["p_value_applicability"] == "not applicable"
    assert list(table["quantile"]) == probabilities
    assert requested == [0, 1]


def test_quantile_exhaustive_small_arm_bootstrap_fixture(monkeypatch):
    data = pd.DataFrame({"y": [0.0, 1.0, 2.0, 1.0, 4.0, 8.0], "d": [0] * 3 + [1] * 3})
    all_draws = np.array(list(itertools.product(range(3), repeat=3)))
    draws = [np.repeat(all_draws, len(all_draws), axis=0), np.tile(all_draws, (len(all_draws), 1))]
    called = []

    def enumerated(high, size, *, generator, device):
        assert high == 3 and size == (729, 3)
        draw = draws[len(called)]
        called.append(True)
        return torch.tensor(draw)

    monkeypatch.setattr(torch, "randint", enumerated)
    probabilities = [0.25, 0.5, 0.75]
    output = run(data, quantile=True, quantiles=probabilities, reps=729)
    _, _, effects, covariance = _independent_bootstrap_output(data, probabilities, draws)
    np.testing.assert_allclose(output["joint_covariance"], covariance, atol=2e-14)
    np.testing.assert_array_equal(output.attrs["state"]["bootstrap_effects"], effects)
    # Enumerating the product distribution independently verifies the arm
    # covariance blocks, including the zero cross-arm covariance in this fixture.
    np.testing.assert_allclose(covariance[:3, 3:6], 0, atol=1e-14)


def test_quantile_private_rng_reproducible_and_global_rng_unchanged():
    global_before = torch.random.get_rng_state().clone()
    first = run(quantile=True, seed=109)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    second = run(quantile=True, seed=109)
    assert causal_design_save(first) == causal_design_save(second)
    other = run(quantile=True, seed=110)
    assert first.attrs["state"]["bootstrap_effects"] != other.attrs["state"]["bootstrap_effects"]


def test_quantile_degenerate_tied_arm_is_not_given_false_variance():
    data = pd.DataFrame({"y": [2.0] * 4 + [5.0] * 5, "d": [0] * 4 + [1] * 5})
    output = run(data, quantile=True)
    table = output["effects"]
    assert list(table.estimate) == [3.0, 3.0, 3.0]
    assert list(table.std_error) == [0.0, 0.0, 0.0]
    np.testing.assert_array_equal(table.ci_low, table.estimate)
    np.testing.assert_array_equal(table.ci_high, table.estimate)
    assert table.p_value.isna().all()
    assert output.attrs["state"]["sample_contains_ties"] == [True, True]
    assert any("positive density" in note for note in output.attrs["notes"])


@pytest.mark.parametrize("quantile", [False, True])
def test_complete_positions_duplicate_labels_missing_and_json_roundtrip(tmp_path, quantile):
    data = sample()
    data.index = ["duplicate"] * len(data)
    data.iloc[1, 0] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        run(data, quantile=quantile)
    output = run(data, quantile=quantile, missing="drop")
    assert output.attrs["positions"] == [0, *range(2, 14)]
    assert output.attrs["unit_labels"] == ["duplicate"] * 13
    state = output.attrs["state"]
    assert state["sample"]["positions"] == output.attrs["positions"]
    assert len(state["outcomes"]) == 13 and state["arm_counts"] == [5, 8]
    path = tmp_path / "full-distribution.json"
    artifact = causal_design_save(output, path)
    restored = causal_design_load(path)
    assert causal_design_save(restored) == artifact
    assert json.loads(path.read_text()) == artifact
    tampered = copy.deepcopy(artifact)
    tampered["payload"]["attrs"]["state"]["outcomes"][0] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(tampered)
    assert output.attrs["stata_parity_validated"] is False


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda"},
        {"weights": "w"},
        {"level": 1.0},
        {"level": 0.0},
        {"level": True},
        {"missing": "ignore"},
    ],
)
def test_unsupported_and_invalid_options_fail_closed(quantile, options):
    with pytest.raises(AnalysisError):
        run(quantile=quantile, **options)


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"y": range(6), "d": [0, 0, 0, 1, 1, 2]}),
        pd.DataFrame({"y": range(6), "d": [0, 0, 0, 0, 0, 0]}),
        pd.DataFrame({"y": range(6), "d": [0, 0, 1, 1, 1, 1]}),
        pd.DataFrame({"y": [0.0, 1.0, 2.0, 3.0, np.inf, 5.0], "d": [0, 0, 0, 1, 1, 1]}),
        pd.DataFrame({"y": list("abcdef"), "d": [0, 0, 0, 1, 1, 1]}),
        pd.DataFrame({"y": range(6), "d": [False] * 3 + [True] * 3}),
    ],
)
def test_binary_numeric_arms_finite_outcome_minimum_arm_contract(data, quantile):
    with pytest.raises(AnalysisError):
        run(data, quantile=quantile)


@pytest.mark.parametrize("targets", [[], "fixed", [0, 0], [True], [np.nan], [np.inf], ["0"]])
def test_cdf_requires_predeclared_unique_finite_numeric_grid(targets):
    with pytest.raises(AnalysisError):
        run(thresholds=targets)


@pytest.mark.parametrize(
    "targets",
    [[], "fixed", [0.5, 0.5], [True], [np.nan], [np.inf], [".5"], [0.0], [1.0], [-0.1], [1.1]],
)
def test_quantile_requires_predeclared_unique_interior_probabilities(targets):
    with pytest.raises(AnalysisError):
        run(quantile=True, quantiles=targets)


@pytest.mark.parametrize(
    "options",
    [{"reps": True}, {"reps": 1}, {"reps": 2.5}, {"seed": True}, {"seed": -1}, {"seed": 2**63}],
)
def test_bootstrap_integer_rep_and_private_seed_contract(options):
    with pytest.raises(AnalysisError):
        run(quantile=True, **options)


@pytest.mark.parametrize(
    "method,targets",
    [
        (treatment_cdf, {"thresholds": [0.0]}),
        (treatment_quantile, {"quantiles": [0.5]}),
    ],
)
def test_randomization_declaration_is_required_and_cannot_be_inferred(method, targets):
    with pytest.raises(TypeError):
        method(sample(), "y", "d", **targets)
    for design in ["observational", None, True]:
        with pytest.raises(AnalysisError):
            method(sample(), "y", "d", design=design, **targets)


def test_cdf_full_covariance_work_budget_before_grid_allocation(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("No grid allocation is allowed after preflight rejects the work.")

    monkeypatch.setattr(torch, "tensor", refuse)
    with pytest.raises(AnalysisError, match="max_work"):
        run(thresholds=list(range(20)), max_work=100)


def test_quantile_draw_sort_joint_covariance_work_and_workspace_preflight(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("No bootstrap draws are allowed after failed preflight.")

    monkeypatch.setattr(torch, "randint", refuse)
    with pytest.raises(AnalysisError, match="max_work"):
        run(quantile=True, reps=1000, max_work=100)
    data = pd.DataFrame({"y": np.linspace(0, 1, 1000), "d": [0] * 500 + [1] * 500})
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        run(data, quantile=True, reps=200)


def test_cdf_grid_workspace_checked_before_full_indicator_allocation():
    data = pd.DataFrame({"y": np.linspace(0, 1, 1000), "d": [0] * 500 + [1] * 500})
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        run(data, thresholds=list(range(30)))


def test_quantile_empirical_definition_matches_numpy_shift_monotonicity():
    data = sample()
    probabilities = [0.0001, 0.1, 1 / 6, 0.5, 0.75, 0.9999]
    output = run(data, quantile=True, quantiles=probabilities)
    for arm in (0, 1):
        expected = np.quantile(data.loc[data.d == arm, "y"], probabilities, method="inverted_cdf")
        np.testing.assert_array_equal(output["effects"][f"q{arm}"], expected)
        assert (np.diff(output["effects"][f"q{arm}"]) >= 0).all()
    shifted = data.copy()
    shifted.loc[shifted.d == 1, "y"] += 12
    shifted_output = run(shifted, quantile=True, quantiles=probabilities)
    np.testing.assert_allclose(shifted_output["effects"].estimate, output["effects"].estimate + 12)
    np.testing.assert_allclose(shifted_output["effects"].std_error, output["effects"].std_error)
    np.testing.assert_allclose(shifted_output["covariance"], output["covariance"], atol=2e-14)


def test_cdf_near_one_level_is_finite_without_tail_rounding():
    output = run(level=math.nextafter(1.0, 0.0))
    assert np.isfinite(output["effects"].ci_low).all()
    assert np.isfinite(output["effects"].ci_high).all()
