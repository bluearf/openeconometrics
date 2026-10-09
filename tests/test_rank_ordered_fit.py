"""Analytic ranking fixtures and adversarial complete-state acceptance gates."""
import copy
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete import rank_ordered as rankmod
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def binary_cases():
    return pd.DataFrame({
        "case": [g for g in range(100) for _ in range(2)],
        "alt": ["a", "b"]*100,
        "rank": [v for g in range(100) for v in ([1, 0] if g < 60 else [0, 1])],
        "x": [1., 0.]*100, "cluster": [g//5 for g in range(100) for _ in range(2)],
    })


def permutation_cases(counts=None, *, partial=False):
    orders = list(itertools.permutations(range(3)))
    counts = [8]*6 if counts is None else counts
    records = []
    g = 0
    for order, count in zip(orders, counts):
        for _ in range(count):
            for alt in range(3):
                position = order.index(alt)+1
                records.append([g, ["a", "b", "c"][alt], 0 if partial and position > 1 else position,
                                int(alt == 1), int(alt == 2), g % 4])
            g += 1
    return pd.DataFrame(records, columns=["case", "alt", "rank", "b", "c", "cluster"])


def fit(frame=None, **kwargs):
    frame = permutation_cases([12, 8, 2, 18, 16, 4]) if frame is None else frame
    return rankmod.rologit(frame, "rank", ["b", "c"], case="case", alternative="alt", **kwargs)


def sealed(result):
    return json.loads(json.dumps(result.attrs, sort_keys=True, ensure_ascii=False, allow_nan=False))


def reseal(attrs):
    attrs["rank_ordered_state"]["checksum"] = rankmod._checksum(attrs["rank_ordered_state"])
    attrs["settings"] = attrs["rank_ordered_state"]["settings"]
    return attrs


@pytest.mark.parametrize("vce,variance", [("oim", 1/24), ("hc0", 1/24), ("cr0", 5/24)])
def test_binomial_closed_form_full_covariance_and_case_cluster_scores(vce, variance):
    result = rankmod.rologit(binary_cases(), "rank", ["x"], case="case", alternative="alt",
                             vce=vce, cluster="cluster" if vce == "cr0" else None)
    state = result.attrs["rank_ordered_state"]["fit"]
    np.testing.assert_allclose(state["beta"], [math.log(1.5)], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(state["information"], [[24.]], rtol=1e-11)
    np.testing.assert_allclose(state["covariance"], [[variance]], rtol=1e-11)
    np.testing.assert_allclose(state["case_scores"], [[.4]]*60+[[-.6]]*40, rtol=1e-11)
    if vce == "cr0":
        np.testing.assert_allclose(state["cluster_scores"], [[2.]]*12+[[-3.]]*8, rtol=1e-11)
    row = result["parameters"].iloc[0]
    assert row["std_error"] == pytest.approx(math.sqrt(variance), rel=1e-11)
    assert row["z"] == pytest.approx(math.log(1.5)/math.sqrt(variance), rel=1e-10)
    assert row["p_value"] == pytest.approx(math.erfc(abs(row["z"])/math.sqrt(2)))
    assert state["log_likelihood"] == pytest.approx(60*math.log(.6)+40*math.log(.4), abs=1e-11)


@pytest.mark.parametrize("partial,probability", [(False, 1/6), (True, 1/3)])
def test_symmetric_permutations_have_known_remaining_set_probabilities(partial, probability):
    result = fit(permutation_cases(partial=partial))
    np.testing.assert_allclose(result["parameters"]["estimate"], [0, 0], atol=1e-14)
    assert np.allclose(result["case_likelihood"]["ranking_probability"], probability)
    for _, rows in result["probabilities"].groupby(["case", "stage"]):
        assert np.allclose(rows["probability"], 1/len(rows))
        assert rows["probability"].sum() == pytest.approx(1)
    if not partial:
        singleton = result["stages"][result["stages"]["riskset_size"] == 1]
        assert np.all(singleton["log_likelihood"] == 0)
        assert np.allclose(result["stage_scores"].iloc[2::3, 2:], 0, atol=0)


def test_full_ranking_is_not_a_first_choice_alias_and_robust_meat_uses_whole_case():
    counts = [12, 8, 2, 18, 16, 4]
    full = fit(permutation_cases(counts), vce="hc0")
    first = fit(permutation_cases(counts, partial=True), vce="hc0")
    np.testing.assert_allclose(first["parameters"]["estimate"], 0, atol=1e-14)
    assert np.linalg.norm(full["parameters"]["estimate"]) > .2
    saved = full.attrs["rank_ordered_state"]["fit"]
    case_scores = np.asarray(saved["case_scores"])
    stage_scores = np.asarray(saved["stage_scores"])
    np.testing.assert_allclose(saved["meat"], case_scores.T@case_scores, rtol=1e-13)
    assert np.linalg.norm(np.asarray(saved["meat"])-stage_scores.T@stage_scores) > 1
    info, bread = np.asarray(saved["information"]), np.asarray(saved["bread"])
    np.testing.assert_allclose(info@bread, np.eye(2), atol=2e-14)
    np.testing.assert_allclose(saved["covariance"], bread@case_scores.T@case_scores@bread, rtol=1e-13)


def test_full_ranking_probability_mass_and_partial_prefix_are_exact():
    frame = permutation_cases([1]*6)
    columns = rankmod._columns("rank", ["b", "c"], "case", "alt")
    p = rankmod._prepare(frame, columns, rankmod._options())
    theta = torch.tensor([.7, -.4], dtype=torch.float64, device="cpu")*p.scale
    moments = rankmod._moments(theta, p)
    masses = np.exp(np.asarray(moments["stage_loglikelihood"]).reshape(6, 3).sum(1))
    assert masses.sum() == pytest.approx(1, abs=4e-15)
    # The two completions with 'a' first exhaust exactly its first-choice mass.
    expected = 1/(1+math.exp(.7)+math.exp(-.4))
    assert masses[:2].sum() == pytest.approx(expected, abs=4e-15)


def test_large_common_case_attribute_levels_cancel_before_utility_evaluation():
    frame = permutation_cases([12, 8, 2, 18, 16, 4])
    original = fit(frame, vce="hc0")
    frame["b"] += 1e10
    frame["c"] += 5e9
    shifted = fit(frame, vce="hc0")
    np.testing.assert_allclose(shifted["parameters"]["estimate"], original["parameters"]["estimate"], rtol=1e-11)
    np.testing.assert_allclose(shifted.attrs["rank_ordered_state"]["fit"]["covariance"], original.attrs["rank_ordered_state"]["fit"]["covariance"], rtol=1e-11)
    assert rankmod.rologit_restore(sealed(shifted)).attrs == shifted.attrs


def test_unavailable_rows_are_saved_but_never_enter_risksets():
    frame = permutation_cases([4]*6)
    frame["available"] = True
    extra = pd.DataFrame([[g, "absent", 0, 999, -999, g % 4, False] for g in range(24)], columns=frame.columns)
    result = fit(pd.concat([frame, extra], ignore_index=True), available="available")
    assert len(result["inputs"]) == 96
    assert len(result["probabilities"]) == 24*6
    assert "absent" not in set(result["probabilities"]["alternative"])
    np.testing.assert_allclose(result["parameters"]["estimate"], 0, atol=1e-14)


@pytest.mark.parametrize("multipliers", [[1e8, 1e-8], [1e-8, 1e8], [3., -2.]])
@pytest.mark.parametrize("vce", ["oim", "hc0", "cr0"])
def test_original_units_preserve_all_parameter_covariance_and_scores(multipliers, vce):
    frame = permutation_cases([12, 8, 2, 18, 16, 4])
    kwargs = dict(vce=vce, cluster="cluster" if vce == "cr0" else None)
    original = fit(frame, **kwargs).attrs["rank_ordered_state"]["fit"]
    modified = frame.copy()
    modified[["b", "c"]] = modified[["b", "c"]]*multipliers
    transformed = fit(modified, **kwargs)
    saved = transformed.attrs["rank_ordered_state"]["fit"]
    factors = np.asarray(multipliers)
    np.testing.assert_allclose(np.asarray(saved["beta"])*factors, original["beta"], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(saved["covariance"])*factors[:, None]*factors[None, :], original["covariance"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(saved["information"])/factors[:, None]/factors[None, :], original["information"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(saved["case_scores"])/factors, original["case_scores"], rtol=1e-10, atol=1e-14)
    restored = rankmod.rologit_restore(sealed(transformed))
    assert all(restored[key].equals(transformed[key]) for key in transformed)


@pytest.mark.parametrize("vce", ["oim", "hc0", "cr0"])
def test_sorted_json_replay_is_complete_exact_and_has_no_optimizer(vce, monkeypatch):
    result = fit(vce=vce, cluster="cluster" if vce == "cr0" else None)
    monkeypatch.setattr(rankmod, "_fit", lambda *a, **k: pytest.fail("restore called optimizer"))
    restored = rankmod.rologit_restore(sealed(result))
    assert len(result) == 14
    assert restored.attrs == result.attrs
    assert all(result[key].equals(restored[key]) for key in result)
    assert restored.to_latex(index=False) == result.to_latex(index=False)
    changed = rankmod.rologit_restore(restored, level=.8)
    for key in result:
        if key != "parameters":
            assert changed[key].equals(result[key])
    for column in ("estimate", "std_error", "z", "p_value"):
        assert changed["parameters"][column].equals(result["parameters"][column])
    assert not changed["parameters"]["ci_lower"].equals(result["parameters"]["ci_lower"])


@pytest.mark.parametrize("field", ["beta", "case_scores", "stage_scores", "probabilities", "probability_jacobian", "information", "bread", "meat", "covariance"])
def test_resealed_scientific_array_tampering_is_rejected(field):
    attrs = sealed(fit(vce="hc0"))
    array = attrs["rank_ordered_state"]["fit"][field]
    if isinstance(array[0], list):
        array[0][0] += .02
    else:
        array[0] += .02
    with pytest.raises(AnalysisError):
        rankmod.rologit_restore(reseal(attrs))


def test_tiny_unit_covariance_inflation_is_rejected_without_absolute_floor():
    frame = permutation_cases([12, 8, 2, 18, 16, 4])
    frame[["b", "c"]] *= 1e8
    result = fit(frame, vce="hc0")
    assert result.attrs["rank_ordered_state"]["fit"]["covariance"][0][0] < 1e-16
    attrs = sealed(result)
    attrs["rank_ordered_state"]["fit"]["covariance"][0][0] += 1e-12
    with pytest.raises(AnalysisError, match="covariance"):
        rankmod.rologit_restore(reseal(attrs))


def test_resealed_undeclared_availability_cannot_misrepresent_saved_inputs():
    attrs = sealed(fit())
    attrs["rank_ordered_state"]["inputs"]["available"][0] = 0
    with pytest.raises(AnalysisError, match="inputs"):
        rankmod.rologit_restore(reseal(attrs))


def test_singular_cluster_covariance_stays_valid_but_resealed_indefinite_is_rejected():
    frame = permutation_cases([12, 8, 2, 18, 16, 4])
    frame["cluster"] = frame["case"] % 2
    result = fit(frame, vce="cr0", cluster="cluster")
    assert rankmod.rologit_restore(sealed(result)).attrs == result.attrs
    covariance = np.asarray(result.attrs["rank_ordered_state"]["fit"]["covariance"])
    assert np.linalg.eigvalsh(covariance)[0] < 1e-13*np.linalg.eigvalsh(covariance)[-1]
    attrs = sealed(result)
    alteration = 1e-9*math.sqrt(covariance[0, 0]*covariance[1, 1])
    sign = math.copysign(1, covariance[0, 1])
    attrs["rank_ordered_state"]["fit"]["covariance"][0][1] += sign*alteration
    attrs["rank_ordered_state"]["fit"]["covariance"][1][0] += sign*alteration
    with pytest.raises(AnalysisError, match="indefinite"):
        rankmod.rologit_restore(reseal(attrs))


@pytest.mark.parametrize("alter", [
    lambda f: f.assign(rank=f["rank"].where(f["rank"] != 2, 1)),
    lambda f: f.assign(rank=f["rank"].where(f["rank"] != 2, 4)),
    lambda f: f.assign(rank=f["rank"]*.5),
    lambda f: f.assign(b=True),
    lambda f: f.assign(b=float("nan")),
    lambda f: f.assign(b=1),
    lambda f: f.assign(c=f["b"]),
    lambda f: pd.concat([f, f.iloc[:1]], ignore_index=True),
])
def test_malformed_rank_sample_and_unidentified_attributes_refused(alter):
    with pytest.raises(AnalysisError):
        fit(alter(permutation_cases()))


@pytest.mark.parametrize("kwargs", [
    {"weights": "w"}, {"device": "cuda"}, {"vce": "robust"}, {"vce": "cluster"},
    {"cluster": "cluster"}, {"vce": "cr0"}, {"level": 1}, {"level": True},
    {"max_iterations": 0}, {"max_iterations": True}, {"tolerance": float("nan")},
    {"max_work": 1}, {"max_work": 2_000_000_001},
])
def test_unsupported_or_invalid_options_are_explicit(kwargs):
    with pytest.raises(AnalysisError):
        fit(**kwargs)


def test_case_cluster_must_be_constant_and_nontrivial():
    frame = permutation_cases()
    frame.loc[0, "cluster"] = 99
    with pytest.raises(AnalysisError, match="constant"):
        fit(frame, vce="cr0", cluster="cluster")
    frame["cluster"] = 1
    with pytest.raises(AnalysisError, match="at least two"):
        fit(frame, vce="cr0", cluster="cluster")


def test_separated_rankings_never_pass_a_small_score_only_gate():
    frame = binary_cases()
    frame["rank"] = [1, 0]*100
    with pytest.raises(AnalysisError):
        rankmod.rologit(frame, "rank", ["x"], case="case", alternative="alt")


def test_current_workspace_gate_still_applies_to_saved_historical_plan():
    frame = permutation_cases()
    large = pd.concat([frame.assign(case=frame["case"]+j*48) for j in range(10)], ignore_index=True)
    with pytest.raises(AnalysisError, match="workspace"):
        with use_workspace_budget(1):
            fit(large)
    attrs = sealed(fit(large))
    # A historical larger budget cannot suppress a current small workspace.
    with pytest.raises(AnalysisError, match="workspace"):
        with use_workspace_budget(1):
            rankmod.rologit_restore(attrs)


def test_explicit_cpu_float64_survives_global_meta_and_float32_defaults():
    frame = permutation_cases([12, 8, 2, 18, 16, 4])
    original = fit(frame, vce="hc0")
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        fitted = fit(frame, vce="hc0")
        restored = rankmod.rologit_restore(sealed(fitted))
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
    assert fitted.attrs == original.attrs
    assert restored.attrs == original.attrs


def test_query_parser_accepts_one_case_and_constant_query_attributes():
    state, _ = rankmod._validate_state(fit())
    frame = pd.DataFrame({"case": ["new"]*2, "alt": ["a", "b"], "b": [1, 1], "c": [0, 0]})
    query = rankmod._prepare(frame, state["columns"], state["options"], require_rank=False, for_fit=False)
    assert query.case_labels == ["new"]
    assert query.stages == []
    assert query.X.shape == (2, 2)


def test_typed_case_labels_and_colliding_parameter_headers_are_preserved():
    frame = binary_cases().iloc[:8].copy()
    frame["case"] = [1, 1, "1", "1", 2, 2, "2", "2"]
    frame["rank"] = [1, 0, 0, 1, 1, 0, 0, 1]
    frame = frame.rename(columns={"x": "parameter"})
    result = rankmod.rologit(frame, "rank", ["parameter"], case="case", alternative="alt")
    assert result["case_scores"].iloc[:, 0].tolist() == [1, "1", 2, "2"]
    assert all(table.columns.is_unique for table in result.values())
    assert rankmod.rologit_restore(sealed(result)).attrs == result.attrs


def test_table_and_outer_attrs_integrity_and_unknown_fields():
    result = fit()
    changed = copy.deepcopy(result)
    changed["parameters"].iloc[0, 1] += 1
    with pytest.raises(AnalysisError, match="tables"):
        rankmod.rologit_restore(changed)
    attrs = sealed(result)
    attrs["notes"].append("forged")
    with pytest.raises(AnalysisError, match="attrs"):
        rankmod.rologit_restore(attrs)
    attrs = sealed(result)
    attrs["rank_ordered_state"]["fit"]["unknown"] = 1
    with pytest.raises(AnalysisError, match="schema"):
        rankmod.rologit_restore(reseal(attrs))
