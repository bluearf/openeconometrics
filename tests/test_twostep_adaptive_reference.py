"""Primary-equation, raw-row and persisted adaptive TwoStep acceptance.

IBM Statistics14 guide pp2,4–6; no licensed vendor output is represented.
Independent numerical oracles use Python/math/NumPy only in development.
"""
import copy
import json
import math

import numpy as np
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.twostep import kernel, public
from test_twostep_reference import reference_hierarchy, reference_ic, reference_loss


def fixture_criteria(values):
    return [[k, 0, 0, value, value] for k, value in enumerate(values, 1)]


def fixture_merges(losses):
    return [{"clusters_after": k-1, "distance": distance} for k, distance in sorted(losses.items(), reverse=True)]


def test_primary_two_stage_dominant_jump_differs_from_global_minimum():
    criteria = fixture_criteria([100., 0., -80., -140., -160., -159.])
    merges = fixture_merges({2: 21., 3: 10., 4: 5., 5: 2., 6: 1.})
    chosen, trace = public._select_two_stage(criteria, merges, "bic", 5)
    assert trace["initial_k"] == 5
    assert trace["changes"][-1]["ratio"] == -.01
    assert chosen == 4  # 5/2 dominates 21/10 by more than 1.15.
    assert min(criteria, key=lambda row: row[3])[0] == 5
    assert trace["reason"] == "dominant_jump"


def test_primary_top_two_near_tie_selects_larger_cluster_count():
    criteria = fixture_criteria([100., 0., -80., -140., -160., -159.])
    merges = fixture_merges({2: 21., 3: 10., 4: 5., 5: 2.5, 6: 1.25})
    chosen, trace = public._select_two_stage(criteria, merges, "bic", 5)
    assert chosen == 5  # largest2.1/second2 <1.15, larger k wins.
    assert trace["reason"] == "top_two_larger_count"


@pytest.mark.parametrize("values,cap,expected,reason", [
    ([10.], 5, 1, "one_feasible_cut"),
    ([10., 20.], 5, 1, "negative_initial_change"),
    ([10., 10.], 5, 1, "zero_initial_change_extension"),
    ([10., 9.], 5, 2, "two_cut_no_jump_extension"),
    ([10., 9., 8.], 1, 1, "declared_one_cluster_cap"),
])
def test_documented_terminal_extensions(values, cap, expected, reason):
    merges = fixture_merges({k: float(k) for k in range(2, len(values)+1)})
    selected, trace = public._select_two_stage(fixture_criteria(values), merges, "bic", cap)
    assert selected == expected and trace["reason"] == reason
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("losses,expected", [({2: 2., 3: 0., 4: 0.}, 2),
                                           ({2: 0., 3: 0., 4: 0.}, 3)])
def test_zero_jump_denominators_are_explicit_finite_json(losses, expected):
    selected, trace = public._select_two_stage(fixture_criteria([100., 0., -80., -140.]),
                                             fixture_merges(losses), "aic", 3)
    assert selected == expected
    assert any("extension" in entry["kind"] for entry in trace["jumps"])
    json.dumps(trace, allow_nan=False)


def independent_select(criteria, losses, cap):
    """Straight primary equations, for nondegenerate reference fixtures only."""
    if criteria[0]-criteria[1] < 0:
        return 1
    change = [criteria[j]-criteria[j+1] for j in range(len(criteria)-1)]
    initial = next((j+1 for j, value in enumerate(change[:cap]) if value/change[0] < .04), min(cap, len(change)))
    ratios = [(losses[k]/losses[k+1], k) for k in range(2, initial+1)]
    if len(ratios) == 1:
        return ratios[0][1]
    largest, second = sorted(ratios, reverse=True)[:2]
    return largest[1] if largest[0] > 1.15*second[0] else max(largest[1], second[1])


@pytest.mark.parametrize("criterion", ["bic", "aic"])
@pytest.mark.parametrize("seed", [11, 31, 59])
def test_actual_mixed_hierarchy_criteria_and_selection_match_raw_reference(criterion, seed):
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(18, 2))
    cats = np.arange(18) % 3
    result = oe.twostep(data={"x": numeric[:, 0], "y": numeric[:, 1], "c": cats},
                       continuous=["x", "y"], categorical=["c"],
                       selection="two_stage", criterion=criterion, max_clusters=8, max_preclusters=32)
    state = result.attrs["twostep_state"]
    # Independently standardize raw values; no production score/cut functions.
    X = (numeric-numeric.mean(0))/numeric.std(0, ddof=0)
    codes = cats[:, None]
    variance = np.var(X, axis=0)
    cuts, merges = reference_hierarchy(X, codes, (3,), variance,
                                       [cf["rows"] for cf in state["preclusters"]])
    expected = [reference_ic(X, codes, (3,), variance, cuts[k]) for k in range(1, 10)]
    column = 3 if criterion == "bic" else 2
    values = [row[column] for row in expected]
    np.testing.assert_allclose(np.array(state["criteria"])[:, 3 if criterion == "bic" else 4], values, atol=1e-11, rtol=1e-12)
    losses = {entry["clusters_after"]+1: entry["distance"] for entry in merges}
    assert state["selected_k"] == independent_select(values, losses, 8)
    restored = oe.twostep_load(oe.twostep_save(result))
    assert restored.attrs["twostep_state"] == state


def fit_noise(**options):
    data = {"x": [0.]*8+[1.]*8+[100., None], "c": ["a"]*18}
    return oe.twostep(data=data, continuous=["x"], categorical=["c"],
                      rebuild=True, noise="adaptive", missing="drop",
                      max_preclusters=2, **options)


def test_fullness_rebuild_sparse_reinsertion_and_every_physical_row_accounted():
    result = fit_noise(selection="two_stage")
    state = result.attrs["twostep_state"]
    assert state["noise_positions"] == [16] and state["missing_positions"] == [17]
    assert state["tree"]["rebuild_count"] >= 1
    assert any(entry["phase"] == "fullness_rebuild" for entry in state["tree"]["rebuild_trace"])
    assert state["noise_cf"]["rows"] == [16]
    kept = [row for cf in state["preclusters"] for row in cf["rows"]]
    assert sorted(kept+state["noise_positions"]+state["missing_positions"]) == list(range(18))
    assert state["tree"]["operation_cost"] <= state["resources"]["tree_work_bound"]
    assert oe.twostep_load(oe.twostep_save(result)).attrs["twostep_state"] == state


def test_log_volume_query_cutoff_independent_raw_score_and_strict_boundary():
    result = fit_noise()
    state = result.attrs["twostep_state"]
    training = np.asarray(state["training_numeric"])
    expected_cutoff = math.log(float(np.ptp(training[:, 0])))
    assert state["tree"]["noise_cutoff"] == pytest.approx(expected_cutoff, abs=1e-14)
    query = oe.twostep_assign(result, data={"x": [0., 10000., None], "c": ["a"]*3}, missing="drop")
    assert query["assignments"]["status"].tolist() == ["cluster", "noise", "missing"]
    transformed = np.vstack((training, [(10000.-state["scaling"][0]["mean"])/state["scaling"][0]["sd"]]))
    codes = np.vstack((np.asarray(state["training_codes"]), [0]))
    distances = [reference_loss(transformed, codes, (1,), np.asarray(state["global_variance"]),
                                [len(training)], cf["rows"]) for cf in state["cuts"][str(state["selected_k"])]]
    assert query["assignments"].iloc[1]["merge_loss"] == pytest.approx(min(distances), rel=1e-12)
    assert min(distances) > expected_cutoff


def test_final_sparse_leaf_pass_without_prior_fullness():
    result = oe.twostep(data={"x": [0.]*8+[5.]*8+[20.]}, continuous=["x"],
                       rebuild=True, noise="adaptive", max_preclusters=32)
    state = result.attrs["twostep_state"]
    assert state["tree"]["rebuild_count"] == 0
    assert [entry["phase"] for entry in state["tree"]["rebuild_trace"]] == ["final_noise"]
    assert state["noise_positions"] == [16]
    assert oe.twostep_load(oe.twostep_save(result)).attrs["state_sha256"] == result.attrs["state_sha256"]


def test_sparse_near_record_is_reinserted_without_growing_tree_and_far_record_remains_noise():
    result = oe.twostep(data={"x": [0.]*8+[1.]*8+[.01, 100.]}, continuous=["x"],
                       rebuild=True, noise="adaptive", max_preclusters=2)
    state = result.attrs["twostep_state"]
    assert any(16 in entry["reinserted_rows"] for entry in state["tree"]["rebuild_trace"])
    assert state["noise_positions"] == [17]
    assert 16 in [row for cf in state["preclusters"] for row in cf["rows"]]
    assert oe.twostep_load(oe.twostep_save(result)).attrs["twostep_state"] == state


@pytest.mark.parametrize("limits", [{"max_preclusters": 2}, {"branch_factor": 2, "max_nodes": 1}])
def test_lossless_adaptive_rebuild_recovers_both_tree_limits(limits):
    result = oe.twostep(data={"x": [0., .1, .2, 5., 5.1, 10., 10.1]}, continuous=["x"],
                       rebuild=True, **limits)
    state = result.attrs["twostep_state"]
    assert state["noise_positions"] == []
    assert sorted(row for cf in state["preclusters"] for row in cf["rows"]) == list(range(7))
    assert state["tree"]["threshold_final"] > state["tree"]["threshold_initial"]
    assert oe.twostep_load(oe.twostep_save(result)).attrs["twostep_state"] == state


def test_hard_retry_and_cumulative_work_limits_refuse_without_partial_model():
    with pytest.raises(AnalysisError, match="max_rebuilds"):
        oe.twostep(data={"x": [0., 1., 2.]}, continuous=["x"], rebuild=True,
                   max_rebuilds=0, max_preclusters=1)
    x = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
    with pytest.raises(AnalysisError, match="Cumulative"):
        kernel.build_tree(x, torch.empty((3, 0), dtype=torch.int64), (), x.var(0, correction=0),
                          torch.arange(3), threshold=0., branch_factor=2, max_preclusters=1,
                          max_nodes=1, rebuild=True, max_work=1)


@pytest.mark.parametrize("mutate", [
    lambda s: s["tree"].__setitem__("noise_cutoff", 100.),
    lambda s: s["tree"].__setitem__("rebuild_count", 0),
    lambda s: s["tree"]["rebuild_trace"][0].__setitem__("sparse_rows", []),
    lambda s: s["tree"]["rebuild_trace"][0].__setitem__("cost_after", 1),
    lambda s: s["tree"]["rebuild_trace"][0].__setitem__("threshold_after", 0.),
    lambda s: s["controls"].__setitem__("noise_fraction", .01),
    lambda s: s["noise_cf"]["mean"].__setitem__(0, 10.),
    lambda s: s.__setitem__("selection_trace", None),
    lambda s: s.__setitem__("version", 1),
])
def test_resealed_adaptive_semantic_tampering_is_rejected(mutate):
    result = fit_noise(selection="two_stage")
    bad = copy.deepcopy(result)
    mutate(bad.attrs["twostep_state"])
    bad.attrs["state_sha256"] = public._seal(bad.attrs["twostep_state"])
    with pytest.raises(AnalysisError):
        oe.twostep_assign(bad, data={"x": [0.], "c": ["a"]})


@pytest.mark.parametrize("options", [
    {"rebuild": 1}, {"selection": "automatic"}, {"noise": "trim"},
    {"noise": "adaptive"}, {"rebuild": True, "noise": "adaptive", "noise_fraction": 0},
    {"rebuild": True, "noise": "adaptive", "min_precluster_size": 2},
    {"rebuild": True, "max_rebuilds": True}, {"rebuild": True, "max_rebuilds": 17},
    {"noise_fraction": .5}, {"max_rebuilds": 4},
])
def test_conflicting_or_unsupported_modes_fail_explicitly(options):
    with pytest.raises(AnalysisError):
        oe.twostep(data={"x": [0., 1., 2.]}, continuous=["x"], **options)


def test_saved_fixed_cut_preserves_adaptive_history_and_overrides_selection():
    result = oe.twostep(data={"x": [0., .1, 5., 5.1, 10., 10.1]}, continuous=["x"],
                       selection="two_stage", rebuild=True, max_preclusters=4)
    fixed = oe.twostep_cut(result, 2)
    assert fixed.attrs["twostep_state"]["selection_trace"] is None
    assert fixed.attrs["twostep_state"]["tree"] == result.attrs["twostep_state"]["tree"]
    assert oe.twostep_load(oe.twostep_save(fixed)).attrs["twostep_state"]["selected_k"] == 2


@pytest.mark.parametrize('name', ['two_stage_mixed', 'lossless_rebuild', 'adaptive_noise', 'typed_seeded_missing'])
def test_saved_mac_examples_restore_without_fit_on_current_platform(name, monkeypatch):
    """Actual Mac files must remain usable on Linux and hosted Windows CPU."""
    from pathlib import Path
    import tarfile

    root = Path(__file__).resolve().parents[1]
    archive = root/'docs/evidence/twostep-adaptive-2026-10-09/mac-complete-proof.tar.gz'
    with tarfile.open(archive) as proof:
        raw = proof.extractfile('saved-project/twostep-adaptive-results/'+name+'.json').read().decode()

    def refused(*args, **kwargs):
        raise AssertionError('Saved platform transfer attempted a new fit')

    monkeypatch.setattr(oe, 'twostep', refused)
    monkeypatch.setattr(public, 'twostep', refused)
    restored = oe.twostep_load(raw)
    assert oe.twostep_save(restored) == raw
    assert 'twostep_state' in restored.attrs
    assert '\\begin{tabular}' in restored.to_latex()
    if name == 'adaptive_noise':
        assigned = oe.twostep_assign(restored, data={'x': [0., 10000.]})
        assert assigned['assignments']['status'].tolist() == ['cluster', 'noise']
