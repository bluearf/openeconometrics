"""Independent full assignment laws and heterogeneous-odds signed-rank oracles."""

import copy
from itertools import combinations, product
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.randomization import (
    randomization_test, rosenbaum_rank_bounds, stratified_randomization,
)
from openecon.resources import use_workspace_budget


def complete_data():
    return pd.DataFrame({"y": [1., 2., 4., 7., 8., 3., -2.],
                         "t": [0, 0, 1, 0, 1, 1, 0]}, index=["duplicate"]*7)


def strata_data():
    return pd.DataFrame({"y": [1., 4., 9., 0., 5., 2., 8., 12.],
                         "t": [0, 1, 0, 1, 0, 1, 0, 1], "s": ["x"]*3+["z"]*5})


def pair_data(differences):
    rows = []
    for i, difference in enumerate(differences):
        rows.extend([[f"p{i}", 0, 0.], [f"p{i}", 1, difference]])
    return pd.DataFrame(rows, columns=["pair", "t", "y"])


def _fit(name, data=None, **options):
    if name == "complete":
        return randomization_test(complete_data() if data is None else data, "y", "t",
                                  design=options.pop("design", "complete_randomized"), **options)
    if name == "stratified":
        return stratified_randomization(strata_data() if data is None else data, "y", "t", "s",
                                        design=options.pop("design", "stratified_randomized"), **options)
    return rosenbaum_rank_bounds(pair_data([2., -1., 2., 0., -3.]) if data is None else data,
                                 "y", "t", "pair", **options)


def _statistic(values, chosen, groups):
    chosen = set(chosen)
    total = 0.
    for rows in groups:
        treated = [values[i] for i in rows if i in chosen]
        control = [values[i] for i in rows if i not in chosen]
        total += len(rows)/len(values)*(math.fsum(treated)/len(treated)-math.fsum(control)/len(control))
    return total


def _extreme(statistics, observed, alternative):
    tolerance = 1e-12
    if alternative == "greater":
        return np.asarray(statistics) >= observed-tolerance
    if alternative == "less":
        return np.asarray(statistics) <= observed+tolerance
    return np.abs(statistics) >= abs(observed)-tolerance


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("null_effect", [0., 1.5])
def test_complete_exact_all_combinations_and_additive_null_imputation(alternative, null_effect):
    data = complete_data()
    original = data.copy(deep=True)
    result = _fit("complete", data, alternative=alternative, null_effect=null_effect)
    groups = [list(range(len(data)))]
    values = data.y.to_numpy()-null_effect*data.t.to_numpy()
    support = list(combinations(range(len(data)), int(data.t.sum())))
    expected = [_statistic(values, chosen, groups) for chosen in support]
    observed = _statistic(values, np.flatnonzero(data.t), groups)
    state = result.attrs["state"]
    np.testing.assert_allclose(state["randomization_statistics"], expected, atol=2e-14, rtol=1e-13)
    assert state["observed_statistic"] == pytest.approx(observed)
    assert state["n_assignments"] == math.comb(7, 3)
    assert state["p_value"] == pytest.approx(_extreme(expected, observed, alternative).mean())
    actual = {tuple(i for i, char in enumerate(code) if char == "1") for code in state["assignment_bits"]}
    assert actual == set(support)
    assert all(code.count("1") == 3 for code in state["assignment_bits"])
    assert state["rng"] is None and state["monte_carlo_standard_error"] is None
    assert state["confidence_interval"] is state["standard_error"] is state["covariance"] is None
    assert result.attrs["unit_labels"] == ["duplicate"]*7
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_stratified_exact_cartesian_law_and_sample_size_weighted_statistic(alternative):
    data = strata_data()
    null = .75
    result = _fit("stratified", data, alternative=alternative, null_effect=null)
    groups = [list(range(3)), list(range(3, 8))]
    supports = [list(combinations(rows, int(data.t.iloc[rows].sum()))) for rows in groups]
    assignments = [tuple(i for chosen in pieces for i in chosen) for pieces in product(*supports)]
    values = data.y.to_numpy()-null*data.t.to_numpy()
    expected = [_statistic(values, chosen, groups) for chosen in assignments]
    observed = _statistic(values, np.flatnonzero(data.t), groups)
    state = result.attrs["state"]
    assert state["component_assignment_counts"] == [3, 10]
    assert state["n_assignments"] == 30
    np.testing.assert_allclose(state["randomization_statistics"], expected, atol=2e-14, rtol=1e-13)
    assert state["p_value"] == pytest.approx(_extreme(expected, observed, alternative).mean())
    assert result["design"].statistic_weight.tolist() == [3/8, 5/8]
    actual = {tuple(i for i, char in enumerate(code) if char == "1") for code in state["assignment_bits"]}
    assert actual == set(assignments)
    assert observed != pytest.approx(values[data.t == 1].mean()-values[data.t == 0].mean())


@pytest.mark.parametrize("name", ["complete", "stratified"])
def test_monte_carlo_fixed_counts_plus_one_simulation_se_private_rng_and_reproducibility(name):
    data = complete_data() if name == "complete" else strata_data()
    before = torch.get_rng_state().clone()
    result = _fit(name, data, method="monte_carlo", draws=301, seed=71, alternative="greater")
    assert torch.equal(torch.get_rng_state(), before)
    duplicate = _fit(name, data, method="monte_carlo", draws=301, seed=71, alternative="greater")
    assert causal_design_save(result) == causal_design_save(duplicate)
    state = result.attrs["state"]
    groups = [list(range(7))] if name == "complete" else [list(range(3)), list(range(3, 8))]
    values = data.y.to_numpy()
    expected = []
    for code in state["assignment_bits"]:
        chosen = [i for i, value in enumerate(code) if value == "1"]
        for rows in groups:
            assert sum(i in chosen for i in rows) == int(data.t.iloc[rows].sum())
        expected.append(_statistic(values, chosen, groups))
    np.testing.assert_allclose(state["randomization_statistics"], expected, atol=1e-13)
    flags = _extreme(expected, state["observed_statistic"], "greater")
    p = (int(flags.sum())+1)/302
    assert state["p_value"] == pytest.approx(p)
    assert state["monte_carlo_standard_error"] == pytest.approx(math.sqrt(p*(1-p)*301)/302)
    assert state["monte_carlo_maximum_standard_error"] == pytest.approx(math.sqrt(301/4)/302)
    assert _fit(name, data, method="monte_carlo", draws=301, seed=72).attrs["state"]["assignment_bits"] != state["assignment_bits"]


def test_complete_mc_uniform_support_sanity_on_fixed_seed():
    result = _fit("complete", method="monte_carlo", draws=3500, seed=999)
    counts = pd.Series(result.attrs["state"]["assignment_bits"]).value_counts()
    assert len(counts) == math.comb(7, 3)
    # A fixed deterministic sanity check, not a proof inferred from one sample.
    assert counts.min() > 50 and counts.max() < 150


def test_typed_strata_and_large_identifiers_remain_distinct_and_persist_exactly():
    big = 2**60+1
    data = pd.DataFrame({"y": range(8), "t": [0, 1]*4,
                         "s": pd.Series([True, True, 1, 1, big, big, 1.5, 1.5], dtype=object)})
    result = _fit("stratified", data)
    assert result.attrs["state"]["n_assignments"] == 16
    assert len(result["design"]) == 4
    assert result["design"].label_type.tolist() == ["bool", "int", "int", "float"]
    assert int(result["design"].label.iloc[2]) == big
    restored = causal_design_load(json.loads(json.dumps(causal_design_save(result))))
    pd.testing.assert_frame_equal(restored["design"], result["design"])


def _signed_rank_law(ranks, probabilities):
    total = int(sum(ranks))
    masses = np.zeros(total+1)
    for signs in product((0, 1), repeat=len(ranks)):
        score = sum(rank*sign for rank, sign in zip(ranks, signs))
        probability = math.prod(p if sign else 1-p for p, sign in zip(probabilities, signs))
        masses[score] += probability
    return masses, np.cumsum(masses[::-1])[::-1]


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.parametrize("differences", [[2., -1., 2., 0., -3.], [1., -2., 3., 4.], [-1., -1., 1.]])
def test_exact_signed_rank_dp_all_masses_tails_and_heterogeneous_odds_extrema(differences, alternative):
    result = _fit("rank", pair_data(differences), alternative=alternative, gammas=[1., 1.2, 2.])
    nonzero = np.asarray(differences)[np.asarray(differences) != 0]
    ranks = (2*rankdata(np.abs(nonzero), method="average")).astype(int)
    observed = int(ranks[nonzero > 0 if alternative == "greater" else nonzero < 0].sum())
    state = result.attrs["state"]
    assert state["doubled_ranks"] == list(ranks)
    assert state["doubled_statistic"] == observed
    assert state["zero_difference_pairs"] == len(differences)-len(nonzero)
    for distribution in state["distributions"]:
        gamma = distribution["gamma"]
        lower, upper = 1/(1+gamma), gamma/(1+gamma)
        low_mass, low_tail = _signed_rank_law(ranks, [lower]*len(ranks))
        high_mass, high_tail = _signed_rank_law(ranks, [upper]*len(ranks))
        np.testing.assert_allclose(distribution["mass_lower"], low_mass, atol=3e-15, rtol=3e-14)
        np.testing.assert_allclose(distribution["mass_upper"], high_mass, atol=3e-15, rtol=3e-14)
        np.testing.assert_allclose(distribution["tail_lower"], low_tail, atol=3e-15)
        np.testing.assert_allclose(distribution["tail_upper"], high_tail, atol=3e-15)
        # Enumerate every heterogeneous endpoint-odds vector independently;
        # the monotone tail extrema must equal the homogeneous endpoint laws.
        heterogeneous = [_signed_rank_law(ranks, p)[1][observed]
                         for p in product((lower, upper), repeat=len(ranks))]
        assert distribution["tail_lower"][observed] == pytest.approx(min(heterogeneous))
        assert distribution["tail_upper"][observed] == pytest.approx(max(heterogeneous))
        assert distribution["normalizer_lower"] == pytest.approx(1)
        assert distribution["normalizer_upper"] == pytest.approx(1)
    assert len(result["null_distributions"]) == 3*(int(ranks.sum())+1)
    assert state["confidence_interval"] is state["covariance"] is None


def test_signed_rank_zero_conditioning_shifted_null_and_subnormal_differences():
    all_zero = _fit("rank", pair_data([0., 0.]))
    assert all_zero["bounds"].p_lower.tolist() == [1.]*3
    assert all_zero["bounds"].p_upper.tolist() == [1.]*3
    assert all_zero.attrs["state"]["informative_pairs"] == 0
    shifted = _fit("rank", pair_data([1., 2., 3.]), null_effect=1.)
    unshifted = _fit("rank", pair_data([0., 1., 2.]))
    np.testing.assert_array_equal(shifted["bounds"], unshifted["bounds"])
    small = _fit("rank", pair_data([1e-320, -2e-320, 3e-320]))
    large = _fit("rank", pair_data([1., -2., 3.]))
    np.testing.assert_array_equal(small["bounds"], large["bounds"])


def test_signed_rank_positive_probability_underflow_is_not_a_truncated_exact_law():
    with pytest.raises(AnalysisError) as error:
        _fit("rank", pair_data([1., -2., 3., 4.]), gammas=[1e150])
    assert error.value.code == "numerical_failure"


@pytest.mark.parametrize("name", ["complete", "stratified", "rank"])
def test_all_three_full_artifact_roundtrip_order_dtypes_and_tamper_refusal(name, tmp_path):
    result = _fit(name)
    path = tmp_path/f"{name}.json"
    artifact = causal_design_save(result, path)
    restored = causal_design_load(path)
    assert list(result) == list(restored)
    assert causal_design_save(restored) == artifact
    for key in result:
        pd.testing.assert_frame_equal(restored[key], result[key])
    damaged = copy.deepcopy(artifact)
    damaged["payload"]["tables"][next(iter(result))]["data"][0][0] = 99
    with pytest.raises(AnalysisError):
        causal_design_load(damaged)
    assert result.attrs["stata_parity_validated"] is False


@pytest.mark.parametrize("name", ["complete", "stratified", "rank"])
def test_cpu_factory_context_is_explicit_and_not_mutated(name):
    expected = causal_design_save(_fit(name))
    with torch.device("meta"):
        observed = causal_design_save(_fit(name))
        assert torch.empty(0).device.type == "meta"
    assert observed == expected


@pytest.mark.parametrize("name", ["complete", "stratified"])
@pytest.mark.parametrize("design", [None, "observational", True, ["randomized"]])
def test_original_design_must_be_explicitly_declared(name, design):
    with pytest.raises(AnalysisError) as error:
        _fit(name, design=design)
    assert error.value.code == "unsupported_design"


@pytest.mark.parametrize("name", ["complete", "stratified"])
def test_missing_outcomes_cannot_shrink_fixed_original_assignment_universe(name):
    data = complete_data() if name == "complete" else strata_data()
    with pytest.raises(AnalysisError) as error:
        _fit(name, data, missing="drop")
    assert error.value.code == "unsupported_missing"
    data.iloc[0, data.columns.get_loc("y")] = np.nan
    with pytest.raises(AnalysisError) as error:
        _fit(name, data)
    assert error.value.code == "missing_values"
    # Invalid original assignment is detected before the missing outcome.
    data.iloc[1, data.columns.get_loc("t")] = 2
    with pytest.raises(AnalysisError) as error:
        _fit(name, data)
    assert error.value.code == "invalid_treatment"


def test_original_stratum_topology_refused_before_missing_outcome_filtering():
    data = strata_data()
    data.loc[:2, "t"] = 1
    data.loc[0, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        _fit("stratified", data)
    assert error.value.code == "invalid_stratum"


def test_signed_rank_whole_pair_missing_policy_does_not_repair_broken_topology():
    data = pair_data([1., 2., 3.])
    data.loc[:1, "y"] = np.nan
    result = _fit("rank", data, missing="drop")
    assert result.attrs["n_removed_pairs"] == 1
    assert result.attrs["positions"] == [2, 3, 4, 5]
    data = pair_data([1., 2., 3.])
    data.loc[0, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        _fit("rank", data, missing="drop")
    assert error.value.code == "invalid_pair"


@pytest.mark.parametrize("name", ["complete", "stratified", "rank"])
@pytest.mark.parametrize("options", [{"device": "cuda"}, {"weights": "w"}, {"max_work": 1}, {"null_effect": np.inf}])
def test_unsupported_options_and_budgets_fail_closed(name, options):
    with pytest.raises(AnalysisError):
        _fit(name, **options)


@pytest.mark.parametrize("name", ["complete", "stratified"])
@pytest.mark.parametrize("options", [{"method": "shuffle"}, {"draws": 0}, {"draws": True}, {"seed": -1}, {"seed": True}, {"alternative": "invalid"}])
def test_randomization_parameter_domain(name, options):
    with pytest.raises(AnalysisError):
        _fit(name, **options)


@pytest.mark.parametrize("options", [{"gammas": []}, {"gammas": [1, 1]}, {"gammas": [2, 1]},
                                    {"gammas": [True]}, {"gammas": [.9]}, {"alternative": "two-sided"}])
def test_signed_rank_parameter_domain(options):
    with pytest.raises(AnalysisError):
        _fit("rank", **options)


def test_complete_exact_support_and_mc_workspace_checked_before_assignment_allocation(monkeypatch):
    import openecon.econometrics.causal_design.randomization as module
    data = pd.DataFrame({"y": np.arange(20.), "t": [0]*10+[1]*10})
    with pytest.raises(AnalysisError) as error:
        _fit("complete", data, max_work=1000)
    assert error.value.code == "work_budget_exceeded"
    def refuse(*args, **kwargs):
        raise AssertionError("Assignments may not allocate after a failed preflight")
    monkeypatch.setattr(torch, "zeros", refuse)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        _fit("complete", method="monte_carlo", draws=5000)
    assert error.value.code == "workspace_limit"
    monkeypatch.undo()
    def refuse_dp(*args, **kwargs):
        raise AssertionError("DP must not run above its complete budget")
    monkeypatch.setattr(module, "_rank_distribution", refuse_dp)
    with pytest.raises(AnalysisError) as error:
        _fit("rank", pair_data(range(1, 30)), max_work=100)
    assert error.value.code == "work_budget_exceeded"


def test_numeric_underflow_and_dynamic_range_fail_instead_of_artificial_assignment_ties():
    data = pd.DataFrame({"y": [float.fromhex('0x0.0000000000001p-1022'), 0., 0., 0.],
                         "t": [1, 1, 0, 0]})
    with pytest.raises(AnalysisError) as error:
        _fit("complete", data)
    assert error.value.code == "numerical_failure"
    data = pd.DataFrame({"y": [1e150, 1e-250, 0., 1.], "t": [0, 0, 1, 1]})
    with pytest.raises(AnalysisError) as error:
        _fit("complete", data)
    assert error.value.code == "numerical_failure"


@pytest.mark.parametrize("name", ["complete", "stratified"])
def test_sharp_null_statistic_is_affine_equivariant_and_arm_swap_reverses_one_sided_test(name):
    data = complete_data() if name == "complete" else strata_data()
    reference = _fit(name, data, null_effect=.75, alternative="greater")
    scale, shift = 12., 123.
    transformed = _fit(name, data.assign(y=scale*data.y+shift), null_effect=scale*.75, alternative="greater")
    np.testing.assert_allclose(transformed["assignments"].statistic,
                               reference["assignments"].statistic*scale, atol=2e-12)
    assert transformed.attrs["state"]["p_value"] == reference.attrs["state"]["p_value"]
    swapped = _fit(name, data.assign(t=1-data.t), null_effect=-.75, alternative="less")
    assert swapped.attrs["state"]["observed_statistic"] == pytest.approx(-reference.attrs["state"]["observed_statistic"])
    assert swapped.attrs["state"]["p_value"] == reference.attrs["state"]["p_value"]


@pytest.mark.parametrize("strata", [None, "", 1])
def test_stratified_role_cannot_silently_select_the_complete_design(strata):
    with pytest.raises(AnalysisError) as error:
        stratified_randomization(strata_data(), "y", "t", strata, design="stratified_randomized")
    assert error.value.code == "invalid_spec"
