"""Independent binomial/enumeration references for paired causal design APIs."""

import itertools
import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import binom
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.econometrics.causal_design.paired import paired_randomization, rosenbaum_bounds
from openecon.resources import use_workspace_budget


def data_for(differences):
    # Reversed pair rows and duplicate row labels exercise positional alignment.
    rows = []
    for i, difference in enumerate(differences):
        rows.extend([[f"pair-{i}", 0, 10.0 + i], [f"pair-{i}", 1, 10.0 + i + float(difference)]])
    return pd.DataFrame(rows, columns=["pair", "treatment", "y"], index=["row"] * len(rows))


def unshifted_data_for(differences):
    # Positive tiny inputs must not disappear while adding an ordinary baseline.
    data = data_for(differences)
    data["y"] = np.column_stack([np.zeros(len(differences)), differences]).reshape(-1)
    return data


def bounds(data, **options):
    return rosenbaum_bounds(data, "y", "treatment", "pair", **options)


def randomized(data, **options):
    return paired_randomization(
        data, "y", "treatment", "pair", design="paired_randomized", **options
    )


@pytest.mark.parametrize("alternative", ["greater", "less"])
@pytest.mark.parametrize(
    "differences", [[1, 3, 0, -1, 2, -2], [0, 0, 0], [-1], [2] * 80 + [-1] * 20]
)
def test_rosenbaum_exact_binomial_reference(alternative, differences):
    gamma = [1.0, 1.25, 2.0, 5.0]
    result = bounds(data_for(differences), gammas=gamma, alternative=alternative)
    informative = [value for value in differences if value != 0]
    k = sum(value > 0 for value in informative)
    if alternative == "less":
        k = len(informative) - k
    expected = [
        [
            g,
            binom.sf(k - 1, len(informative), 1 / (1 + g)),
            binom.sf(k - 1, len(informative), g / (1 + g)),
            1 / (1 + g),
            g / (1 + g),
        ]
        for g in gamma
    ]
    assert_allclose(result["bounds"].to_numpy(dtype=float), expected, rtol=2e-11, atol=2e-14)
    saved = result.attrs["state"]
    assert saved["zero_difference_pairs"] == differences.count(0)
    assert saved["informative_pairs"] == len(informative)
    assert saved["covariance"] is None and saved["df"] is None
    assert result.attrs["stata_parity_validated"] is False


def test_rosenbaum_binomial_bounds_cover_every_odds_bounded_assignment_law():
    differences = [1, 1, 1, -1]
    gamma = 2.4
    result = bounds(data_for(differences), gammas=[gamma])
    lower, upper = result["bounds"].iloc[0][["p_lower", "p_upper"]]
    # Deliberately heterogeneous allowed treatment-sign probabilities: the
    # reference integrates every assignment, independently of beta_inc.
    probabilities = [1 / (1 + gamma), 0.41, 0.55, gamma / (1 + gamma)]
    tail = 0.0
    for signs in itertools.product((0, 1), repeat=4):
        probability = np.prod([p if sign else 1 - p for sign, p in zip(signs, probabilities)])
        tail += probability * (sum(signs) >= 3)
    assert lower <= tail <= upper
    no_bias = bounds(data_for(differences), gammas=[1])
    assert no_bias["bounds"].iloc[0].p_lower == no_bias["bounds"].iloc[0].p_upper


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
@pytest.mark.parametrize("null_effect", [0, 0.75, -1.5])
def test_paired_fisher_exact_complete_assignment_reference(alternative, null_effect):
    observed = np.array([2.0, -0.75, 0.0, 4.5])
    result = randomized(data_for(observed), alternative=alternative, null_effect=null_effect)
    adjusted = observed - null_effect
    reference = np.array(
        [np.mean(np.array(signs) * adjusted) for signs in itertools.product((-1, 1), repeat=4)]
    )
    statistic = np.mean(adjusted)
    if alternative == "greater":
        extreme = reference >= statistic - 1e-12
    elif alternative == "less":
        extreme = reference <= statistic + 1e-12
    else:
        extreme = np.abs(reference) >= abs(statistic) - 1e-12
    state = result.attrs["state"]
    assert state["n_assignments"] == 16
    assert_allclose(sorted(state["randomization_statistics"]), sorted(reference), atol=1e-15)
    assert_allclose(state["p_value"], extreme.mean(), atol=0)
    assert_allclose(state["observed_statistic"], statistic, atol=0)
    assert state["monte_carlo_standard_error"] is None
    assert len(set(state["assignment_bits"])) == 16
    assert state["pair_rows"][0][1:3] == [1, 0]
    assert state["sample"]["unit_labels"] == ["row"] * 8


def test_paired_mc_plus_one_full_assignment_state_and_private_rng():
    data = data_for([1, 2, -1, 0.5, 3])
    original_rng = torch.random.get_rng_state().clone()
    first = randomized(data, method="monte_carlo", draws=2000, seed=701)
    second = randomized(data, method="monte_carlo", draws=2000, seed=701)
    assert torch.equal(torch.random.get_rng_state(), original_rng)
    assert first.attrs["state_sha256"] == second.attrs["state_sha256"]
    state = first.attrs["state"]
    adjusted = np.asarray(state["null_adjusted_differences"])
    reference = [
        np.mean(np.array([1 if bit == "1" else -1 for bit in bits]) * adjusted)
        for bits in state["assignment_bits"]
    ]
    assert_allclose(state["randomization_statistics"], reference, atol=1e-15)
    tolerance = state["statistic_tie_tolerance"]
    count = sum(abs(value) >= abs(state["observed_statistic"]) - tolerance for value in reference)
    assert state["n_extreme"] == count
    assert state["p_value"] == (count + 1) / 2001
    assert_allclose(
        state["monte_carlo_standard_error"],
        np.sqrt(state["p_value"] * (1 - state["p_value"]) * 2000) / 2001,
    )
    assert len(first["draws"]) == 2000
    assert (
        randomized(data, method="monte_carlo", draws=2000, seed=702).attrs["state_sha256"]
        != first.attrs["state_sha256"]
    )


def test_declared_cpu_path_ignores_global_torch_default_device():
    data = data_for([1, -2, 3])
    references = [bounds(data), randomized(data), randomized(data, method="monte_carlo", draws=20)]
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        results = [bounds(data), randomized(data), randomized(data, method="monte_carlo", draws=20)]
        assert [value.attrs["state_sha256"] for value in results] == [
            value.attrs["state_sha256"] for value in references
        ]
    finally:
        torch.set_default_device(previous)


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_all_null_adjusted_ties_have_p_one(method):
    result = randomized(data_for([1, 1, 1]), null_effect=1, method=method, draws=30)
    assert result.attrs["state"]["p_value"] == 1
    assert result.attrs["state"]["zero_difference_pairs"] == 3


@pytest.mark.parametrize("method", ["exact", "monte_carlo"])
def test_nonzero_subnormal_mean_contribution_cannot_become_an_assignment_tie(method):
    # This contrast exists in float64, but division by five erases it. The
    # greater-tail probability is1/2 after harmless positive rescaling.
    with pytest.raises(AnalysisError, match="underflows") as error:
        randomized(unshifted_data_for([5e-324, 0, 0, 0, 0]), alternative="greater", method=method)
    assert error.value.code == "numerical_failure"
    scaled = randomized(unshifted_data_for([1e-180, 0, 0, 0, 0]), alternative="greater")
    assert scaled.attrs["state"]["p_value"] == 0.5


@pytest.mark.parametrize("scale", [1e-180, 1e-300])
@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_balanced_representable_tiny_pair_contrasts_remain_valid(scale, alternative):
    # Genuine cancellation at zero is valid. A fixed-size epsilon floor would
    # reject this sample even though its assignment ranks are representable.
    values = [1, -1, 2, -2, 0]
    reference = randomized(data_for(values), alternative=alternative)
    tiny = randomized(unshifted_data_for([value*scale for value in values]), alternative=alternative)
    assert tiny.attrs["state"]["observed_statistic"] == 0
    assert tiny.attrs["state"]["p_value"] == reference.attrs["state"]["p_value"]


def test_pair_positions_preserved_across_row_shuffle_and_both_outcomes_missing():
    data = data_for([1, -2, 3, 4])
    data.iloc[2:4, data.columns.get_loc("y")] = np.nan
    shuffled = data.iloc[[7, 0, 2, 5, 1, 3, 4, 6]].copy()
    result = randomized(shuffled, missing="drop")
    assert result.attrs["n_input_pairs"] == 4
    assert result.attrs["n_removed_pairs"] == 1
    assert result.attrs["positions"] == [0, 1, 3, 4, 6, 7]
    assert_allclose(sorted(result.attrs["state"]["differences"]), [1, 3, 4])
    assert_allclose(
        result.attrs["state"]["p_value"], randomized(data_for([1, 3, 4])).attrs["state"]["p_value"]
    )


@pytest.mark.parametrize(
    "case",
    [
        "one_missing",
        "third_row_missing",
        "duplicate_treated",
        "one_member",
        "missing_identity",
        "missing_assignment",
    ],
)
def test_invalid_pair_topology_never_repaired_by_listwise_missing_drop(case):
    data = data_for([1, 2, 3])
    if case == "one_missing":
        data.iloc[0, data.columns.get_loc("y")] = np.nan
    elif case == "third_row_missing":
        data = pd.concat([data, pd.DataFrame([["pair-0", 0, np.nan]], columns=data.columns)])
    elif case == "duplicate_treated":
        data.iloc[0, data.columns.get_loc("treatment")] = 1
    elif case == "one_member":
        data = data.iloc[1:]
    elif case == "missing_identity":
        data.iloc[0, data.columns.get_loc("pair")] = None
    else:
        data.iloc[0, data.columns.get_loc("treatment")] = np.nan
    for procedure in (bounds, randomized):
        with pytest.raises(AnalysisError):
            procedure(data, missing="drop")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gammas=[]),
        dict(gammas=[0.9]),
        dict(gammas=[1, 1]),
        dict(gammas=[2, 1]),
        dict(gammas=[True]),
        dict(gammas=[float("inf")]),
        dict(gammas="1"),
        dict(alternative="two-sided"),
    ],
)
def test_sensitivity_rejects_invalid_options(kwargs):
    with pytest.raises(AnalysisError):
        bounds(data_for([1, -2]), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="permutation"),
        dict(alternative="invalid"),
        dict(null_effect=float("nan")),
        dict(null_effect=True),
        dict(draws=0),
        dict(draws=True),
        dict(seed=-1),
        dict(seed=0.5),
    ],
)
def test_randomization_rejects_invalid_options(kwargs):
    with pytest.raises(AnalysisError):
        randomized(data_for([1, -2]), **kwargs)


def test_design_declaration_required_and_observational_matching_not_a_design():
    data = data_for([1, -2])
    with pytest.raises(TypeError):
        paired_randomization(data, "y", "treatment", "pair")
    with pytest.raises(AnalysisError):
        paired_randomization(data, "y", "treatment", "pair", design="matched_observational")


@pytest.mark.parametrize("procedure", [bounds, randomized])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(device="cuda"),
        dict(device="mps"),
        dict(weights="w"),
        dict(max_work=1),
        dict(max_work=True),
    ],
)
def test_sample_device_weight_work_guards(procedure, kwargs):
    with pytest.raises(AnalysisError):
        procedure(data_for([1, -2]), **kwargs)


def test_exact_refuses_complete_work_and_simulation_refuses_workspace():
    with pytest.raises(AnalysisError, match="max_work"):
        randomized(data_for([1] * 20), max_work=1000)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        randomized(data_for([1] * 10), method="monte_carlo", draws=100000)


@pytest.mark.parametrize("procedure", [bounds, randomized])
def test_complete_table_state_json_roundtrip_and_tamper_refusal(procedure, tmp_path):
    result = procedure(data_for([1, -2, 0, 3]))
    path = tmp_path / "paired.json"
    artifact = causal_design_save(result, path)
    assert json.loads(path.read_text()) == artifact
    restored = causal_design_load(path)
    assert causal_design_save(restored) == artifact
    for name in result:
        assert list(restored[name].columns) == list(result[name].columns)
        assert list(restored[name].index) == list(result[name].index)
    json.dumps(restored.attrs, allow_nan=False)
    assert "tabular" in restored.to_latex()
    artifact["payload"]["attrs"]["state"]["differences"][0] += 1
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(artifact)
