"""Independent full-law Fisher oracles, original topology and complete artifacts."""

from copy import deepcopy
from itertools import combinations
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.assignment_extensions import (
    bernoulli_randomization, cluster_randomization,
)
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.resources import use_workspace_budget


def cluster_data():
    return pd.DataFrame({
        "y": [1.25, -2.5, 7., 4.25, 8.5, -1., 9.75],
        "d": [0, 0, 1, 0, 0, 0, 1],
        "cluster": ["a", "a", "b", "c", "c", "c", "d"],
    }, index=["duplicate", 7, "duplicate", 7, 9, 9, 9])


def bernoulli_data():
    return pd.DataFrame({"y": [1.25, -2.5, 4.25, 8.5], "d": [0, 1, 0, 1],
                         "p": [.17, .31, .58, .83]}, index=[3, 3, "duplicate", "duplicate"])


def fit(kind, data=None, **options):
    if kind == "cluster":
        return cluster_randomization(cluster_data() if data is None else data, "y", "d", "cluster",
            design=options.pop("design", "cluster_randomized"), **options)
    return bernoulli_randomization(bernoulli_data() if data is None else data, "y", "d", "p",
        design=options.pop("design", "bernoulli_randomized"), **options)


def oracle_statistic(values, assignment, probabilities=None, clusters=None):
    n = len(values)
    if probabilities is not None:
        return math.fsum(float(y)*(int(d)/float(p)-(1-int(d))/(1-float(p)))/n
                         for y, d, p in zip(values, assignment, probabilities, strict=True))
    chosen = [i for i, value in enumerate(assignment) if value]
    control = [i for i, value in enumerate(assignment) if not value]
    totals = [math.fsum(float(values[i]) for i in rows) for rows in clusters]
    g = len(totals)
    return g/n*(math.fsum(totals[i] for i in chosen)/len(chosen)
                - math.fsum(totals[i] for i in control)/len(control))


def extreme(values, observed, alternative):
    if alternative == "greater":
        return np.asarray(values) >= observed-1e-12
    if alternative == "less":
        return np.asarray(values) <= observed+1e-12
    return np.abs(values) >= abs(observed)-1e-12


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("null", [0., .73])
def test_cluster_full_support_unequal_sizes_scaled_totals_and_unitwise_null(alternative, null):
    data = cluster_data()
    original = data.copy(deep=True)
    result = fit("cluster", data, alternative=alternative, null_effect=null)
    state = result.attrs["state"]
    clusters = [[0, 1], [2], [3, 4, 5], [6]]
    values = data.y.to_numpy()-null*data.d.to_numpy()
    assignments = [tuple(int(i in chosen) for i in range(4)) for chosen in combinations(range(4), 2)]
    expected = [oracle_statistic(values, a, clusters=clusters) for a in assignments]
    observed = oracle_statistic(values, [0, 1, 0, 1], clusters=clusters)
    np.testing.assert_allclose(state["randomization_statistics"], expected, rtol=3e-15, atol=3e-15)
    assert state["observed_statistic"] == pytest.approx(observed, rel=3e-15)
    assert state["assignment_bits"] == ["".join(map(str, a)) for a in assignments]
    assert state["n_assignments"] == state["assignment_universe_size"] == 6
    np.testing.assert_allclose(state["assignment_probabilities"], [1/6]*6, rtol=0, atol=0)
    assert state["probability_mass_sum"] == pytest.approx(1)
    assert state["p_value"] == pytest.approx(extreme(expected, observed, alternative).mean())
    for group_code, unit_code in zip(state["assignment_bits"], state["unit_assignment_bits"], strict=True):
        assert unit_code == "".join(group_code[group] for group in [0, 0, 1, 2, 2, 2, 3])
    assert state["cluster_null_imputed_totals"] == [math.fsum(values[i] for i in rows) for rows in clusters]
    # Unequal cluster sizes make received-arm unit means and cluster means different targets.
    assert observed != pytest.approx(values[data.d == 1].mean()-values[data.d == 0].mean())
    cluster_means = [np.mean(values[rows]) for rows in clusters]
    assert observed != pytest.approx(np.mean(np.asarray(cluster_means)[[1, 3]])-np.mean(np.asarray(cluster_means)[[0, 2]]))
    assert result.attrs["positions"] == list(range(7))
    assert result.attrs["unit_labels"] == ["duplicate", 7, "duplicate", 7, 9, 9, 9]
    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("null", [0., .73])
def test_bernoulli_all_binary_vectors_exact_unequal_mass_and_unconditional_tail(alternative, null):
    data = bernoulli_data()
    result = fit("bernoulli", data, alternative=alternative, null_effect=null)
    state = result.attrs["state"]
    values = data.y.to_numpy()-null*data.d.to_numpy()
    assignments = [tuple((i >> j) & 1 for j in range(4)) for i in range(16)]
    masses = [math.prod(float(p) if d else 1-float(p) for d, p in zip(a, data.p, strict=True)) for a in assignments]
    expected = [oracle_statistic(values, a, data.p) for a in assignments]
    observed = oracle_statistic(values, data.d, data.p)
    flags = extreme(expected, observed, alternative)
    np.testing.assert_allclose(state["randomization_statistics"], expected, rtol=3e-15, atol=1e-14)
    np.testing.assert_allclose(state["assignment_probabilities"], masses, rtol=0, atol=0)
    assert state["p_value"] == pytest.approx(math.fsum(p for p, flag in zip(masses, flags, strict=True) if flag), rel=3e-15)
    assert state["n_assignments"] == state["assignment_universe_size"] == 16
    assert "0000" in state["assignment_bits"] and "1111" in state["assignment_bits"]
    assert state["p_value"] != pytest.approx(float(flags.mean()))
    assert state["observed_assignment_probability"] == math.prod(p if d else 1-p for p, d in zip(data.p, data.d, strict=True))
    assert state["probability_mass_sum"] == pytest.approx(1)


@pytest.mark.parametrize("assignment", [[0, 0, 0, 0], [1, 1, 1, 1]])
def test_empty_observed_bernoulli_arm_is_part_of_the_declared_unconditional_law(assignment):
    data = bernoulli_data()
    data["d"] = assignment
    result = fit("bernoulli", data)
    state = result.attrs["state"]
    assert state["observed_statistic"] == pytest.approx(oracle_statistic(data.y, assignment, data.p))
    assert state["n_assignments"] == 16


def test_small_nonzero_bernoulli_exact_tail_and_every_mass_are_preserved():
    data = pd.DataFrame({"y": [1., 1., 1.], "d": [1, 1, 1], "p": [1e-5, 2e-5, 3e-5]})
    state = fit("bernoulli", data, alternative="greater").attrs["state"]
    assert state["p_value"] == pytest.approx(6e-15, rel=2e-15, abs=0)
    assert min(state["assignment_probabilities"]) > 0
    assert state["n_extreme"] == 1


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_private_monte_carlo_full_replay_plus_one_and_simulation_se(kind):
    before = torch.get_rng_state().clone()
    result = fit(kind, method="monte_carlo", draws=301, seed=89, alternative="greater", null_effect=.73)
    assert torch.equal(torch.get_rng_state(), before)
    assert causal_design_save(result) == causal_design_save(fit(kind, method="monte_carlo", draws=301, seed=89, alternative="greater", null_effect=.73))
    state = result.attrs["state"]
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    values = data.y.to_numpy()-.73*data.d.to_numpy()
    assignments = [tuple(map(int, code)) for code in state["assignment_bits"]]
    expected = [oracle_statistic(values, a, clusters=[[0, 1], [2], [3, 4, 5], [6]])
                if kind == "cluster" else oracle_statistic(values, a, data.p) for a in assignments]
    np.testing.assert_allclose(state["randomization_statistics"], expected, rtol=3e-15, atol=2e-14)
    if kind == "cluster":
        assert all(sum(a) == 2 for a in assignments)
        np.testing.assert_array_equal(state["assignment_probabilities"], [1/6]*301)
    else:
        masses = [math.prod(p if d else 1-p for p, d in zip(data.p, a, strict=True)) for a in assignments]
        np.testing.assert_allclose(state["assignment_probabilities"], masses, rtol=0, atol=0)
        assert len(set(sum(a) for a in assignments)) > 1
    p = (state["n_extreme"]+1)/302
    assert state["p_value"] == p
    assert state["monte_carlo_standard_error"] == math.sqrt(p*(1-p)*301)/302
    assert state["monte_carlo_maximum_standard_error"] == math.sqrt(301/4)/302
    assert state["probability_mass_sum"] is None
    assert state["rng"]["seed"] == 89
    assert state["assignment_bits"] != fit(kind, method="monte_carlo", draws=301, seed=90).attrs["state"]["assignment_bits"]
    for key in ["covariance", "standard_error", "confidence_interval", "df"]:
        assert state[key] is None


def test_monte_carlo_bernoulli_follows_unequal_marginals_without_conditioning():
    state = fit("bernoulli", method="monte_carlo", draws=5000, seed=71).attrs["state"]
    draws = np.asarray([list(map(int, code)) for code in state["assignment_bits"]])
    np.testing.assert_allclose(draws.mean(0), [.17, .31, .58, .83], rtol=0, atol=.025)
    assert (draws.sum(1) == 0).any() and (draws.sum(1) == 4).any()


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_full_typed_artifact_file_json_latex_and_tamper_refusal(kind, tmp_path):
    result = fit(kind)
    path = tmp_path/f"{kind}.json"
    artifact = causal_design_save(result, path)
    for restored in [causal_design_load(path), causal_design_load(json.loads(json.dumps(artifact)))]:
        assert causal_design_save(restored) == artifact
        assert restored.attrs == result.attrs
        assert list(restored) == list(result)
        for name in result:
            pd.testing.assert_frame_equal(restored[name], result[name])
    assert "tabular" in result.to_latex()
    damaged = deepcopy(artifact)
    damaged["payload"]["attrs"]["state"]["p_value"] = .123
    with pytest.raises(AnalysisError, match="artifact"):
        causal_design_load(damaged)
    result["assignments"].iloc[0, result["assignments"].columns.get_loc("probability")] = .123
    with pytest.raises(AnalysisError, match="intact"):
        causal_design_save(result)


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_cpu_global_device_and_default_dtype_do_not_change_any_artifact(kind):
    expected = causal_design_save(fit(kind))
    original = torch.get_default_dtype()
    try:
        for dtype in [torch.float32, torch.float64]:
            torch.set_default_dtype(dtype)
            with torch.device("meta"):
                actual = causal_design_save(fit(kind))
                assert torch.empty(0).device.type == "meta"
                assert torch.get_default_dtype() == dtype
            assert actual == expected
    finally:
        torch.set_default_dtype(original)


def test_typed_cluster_identities_large_integer_and_order_are_retained():
    big = 2**60+1
    data = pd.DataFrame({"y": [1., 2., 4., 5., 7., 9., 11., 13.], "d": [0, 0, 1, 1, 0, 0, 1, 1],
                         "cluster": pd.Series([True, True, 1, 1, big, big, 1.5, 1.5], dtype=object)})
    result = fit("cluster", data)
    assert result["design"].label_type.tolist() == ["bool", "int", "int", "float"]
    assert result["design"].label.iloc[2] == big
    restored = causal_design_load(causal_design_save(result))
    pd.testing.assert_frame_equal(restored["design"], result["design"])


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
@pytest.mark.parametrize("option,value,code", [
    ("design", None, "unsupported_design"), ("design", "randomized", "unsupported_design"),
    ("missing", "drop", "unsupported_missing"), ("device", "cuda", "unsupported_device"),
    ("weights", "w", "unsupported_weights"), ("method", "auto", "invalid_option"),
    ("alternative", "wrong", "invalid_option"), ("seed", True, "invalid_option"),
    ("draws", 0, "invalid_option"), ("null_effect", float("inf"), "invalid_option"),
    ("max_work", True, "invalid_option"),
])
def test_declarations_and_options_are_strict(kind, option, value, code):
    with pytest.raises(AnalysisError) as error:
        fit(kind, **{option: value})
    assert error.value.code == code


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
@pytest.mark.parametrize("bad", [True, 2, -.5])
def test_original_treatment_is_checked_before_missing_outcomes(kind, bad):
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    data["d"] = bad
    data.loc[data.index[0], "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit(kind, data)
    assert error.value.code == "invalid_treatment"


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_missing_outcomes_and_topology_are_never_dropped(kind):
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    data.iloc[0, data.columns.get_loc("y")] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit(kind, data)
    assert error.value.code == "missing_values"
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    role = "cluster" if kind == "cluster" else "p"
    data.iloc[0, data.columns.get_loc(role)] = None
    with pytest.raises(AnalysisError) as error:
        fit(kind, data)
    assert error.value.code == "missing_values"


def test_mixed_cluster_assignment_refused_before_missing_outcomes():
    data = cluster_data()
    data.iloc[0, data.columns.get_loc("d")] = 1
    data.iloc[0, data.columns.get_loc("y")] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit("cluster", data)
    assert error.value.code == "invalid_cluster"


@pytest.mark.parametrize("bad", [0., 1., -1., 2., True])
def test_assignment_probability_must_be_numeric_strictly_interior(bad):
    data = bernoulli_data()
    data["p"] = bad
    with pytest.raises(AnalysisError) as error:
        fit("bernoulli", data)
    assert error.value.code == "invalid_probability"


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_role_collisions_non_numeric_outcomes_and_generic_dataset_are_refused(kind):
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    method = cluster_randomization if kind == "cluster" else bernoulli_randomization
    declaration = f"{kind}_randomized"
    with pytest.raises(AnalysisError) as error:
        method(data, "y", "d", "y", design=declaration)
    assert error.value.code == "invalid_spec"
    data["y"] = "1"
    with pytest.raises(AnalysisError) as error:
        fit(kind, data)
    assert error.value.code == "non_numeric_column"
    from openecon.dataset import Dataset
    with pytest.raises(AnalysisError) as error:
        fit(kind, Dataset.from_frame(data))
    assert error.value.code == "unsupported_dataset"


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_complete_support_work_and_workspace_refuse_before_assignment_tensor_allocation(kind, monkeypatch):
    import openecon.econometrics.causal_design.assignment_extensions as module
    called = []
    original = module._statistics
    monkeypatch.setattr(module, "_statistics", lambda *a, **k: called.append(True) or original(*a, **k))
    with pytest.raises(AnalysisError) as error:
        fit(kind, max_work=5000)
    assert error.value.code == "work_budget_exceeded" and not called
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit(kind, method="monte_carlo", draws=2000)
    assert error.value.code == "workspace_limit" and not called


def test_exact_bernoulli_never_switches_to_monte_carlo_when_support_exceeds_budget():
    data = pd.DataFrame({"y": range(30), "d": [0, 1]*15, "p": [.5]*30})
    with pytest.raises(AnalysisError) as error:
        fit("bernoulli", data)
    assert error.value.code == "work_budget_exceeded"


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_tiny_normal_outcomes_large_rescaling_and_structural_zero_are_valid(kind):
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    reference = fit(kind, data)
    for multiplier in [1e-200, 1e140]:
        scaled = data.copy()
        scaled["y"] = scaled.y*multiplier
        result = fit(kind, scaled)
        assert result.attrs["state"]["p_value"] == reference.attrs["state"]["p_value"]
        np.testing.assert_allclose(np.asarray(result.attrs["state"]["randomization_statistics"])/multiplier,
                                   reference.attrs["state"]["randomization_statistics"], rtol=5e-15)
    data["y"] = 0.
    assert fit(kind, data).attrs["state"]["p_value"] == 1


def test_native_expansion_retains_nested_cancellation_that_plain_sum_can_lose():
    data = pd.DataFrame({"y": [1e100, 1., -1e100, 0.], "d": [1, 1, 1, 0], "p": [.5]*4})
    state = fit("bernoulli", data).attrs["state"]
    assert state["observed_statistic"] == .5


def test_distinct_assignment_comparisons_inside_roundoff_domain_are_refused():
    data = pd.DataFrame({"y": [1e100, 1., -1e100, 0.], "d": [1, 1, 1, 0], "p": [.5]*4})
    with pytest.raises(AnalysisError) as error:
        fit("bernoulli", data, alternative="greater")
    assert error.value.code == "numerical_failure"
    data["y"] = [1e4, 1., -1e4, 0.]
    state = fit("bernoulli", data, alternative="greater").attrs["state"]
    candidates = [tuple((i >> j) & 1 for j in range(4)) for i in range(16)]
    observed = oracle_statistic(data.y, data.d, data.p)
    expected = [oracle_statistic(data.y, a, data.p) for a in candidates]
    assert state["p_value"] == sum(value >= observed for value in expected)/16


def test_nonzero_subnormal_contributions_are_refused_before_a_zero_statistic_is_reported():
    data = cluster_data()
    data["y"] = np.nextafter(0., 1.)
    with pytest.raises(AnalysisError) as error:
        fit("cluster", data)
    assert error.value.code == "numerical_failure"
    data["y"] = 1e-200
    assert fit("cluster", data).attrs["state"]["n_assignments"] == 6


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
@pytest.mark.parametrize("label", ["x"*257, "\x00"*43, "\U0001f30d"*22])
def test_escaped_original_index_size_is_bounded_before_statistics(kind, label, monkeypatch):
    import openecon.econometrics.causal_design.assignment_extensions as module
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    data.index = [label]*len(data)
    monkeypatch.setattr(module, "_statistics", lambda *a, **k: pytest.fail("allocated assignment statistics"))
    with pytest.raises(AnalysisError) as error:
        fit(kind, data)
    assert error.value.code == "resource_limit"


def test_escaped_cluster_label_size_is_bounded_and_small_escaped_identifiers_persist():
    data = cluster_data()
    data["cluster"] = ["\x00"*43 if value == "a" else value for value in data.cluster]
    with pytest.raises(AnalysisError) as error:
        fit("cluster", data)
    assert error.value.code == "resource_limit"
    data["cluster"] = ["\x00"*42 if len(value) > 1 else value for value in data.cluster]
    state = fit("cluster", data).attrs["state"]
    assert state["clusters"][0]["label"] == "\x00"*42


@pytest.mark.parametrize("kind", ["cluster", "bernoulli"])
def test_absorbed_nonzero_null_shift_is_a_numerical_refusal(kind):
    data = cluster_data() if kind == "cluster" else bernoulli_data()
    data["y"] = 1e140
    with pytest.raises(AnalysisError) as error:
        fit(kind, data, null_effect=1.)
    assert error.value.code == "numerical_failure"


def test_probability_complement_and_positive_retained_mass_underflow_are_refused():
    data = bernoulli_data()
    data["p"] = 1e-30
    with pytest.raises(AnalysisError) as error:
        fit("bernoulli", data)
    assert error.value.code == "numerical_failure"
    data = pd.DataFrame({"y": np.ones(40), "d": np.ones(40), "p": np.full(40, 1e-10)})
    with pytest.raises(AnalysisError) as error:
        fit("bernoulli", data, method="monte_carlo", draws=1)
    assert error.value.code == "numerical_failure"
