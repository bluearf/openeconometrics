"""Independent probability/Bayes oracles and guarded known-bank portability."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logsumexp
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.irt import calibrated as module
from openecon.econometrics.irt.calibrated import (
    _bank, _logp, _posterior, _posterior_input, _responses, _seal,
    irt_bank_binary, irt_bank_polytomous, irt_bank_restore, irt_posterior,
    irt_posterior_restore,
)
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget


def binary():
    return irt_bank_binary(["a", "b", "c"], [1.2, .8, 1.5], [-.5, .8, 0], guessing=[0, .2, .1], upper=[1, .9, .8])


def poly():
    return irt_bank_polytomous(["g", "p", "n"], family=["grm", "gpcm", "nrm"],
        thresholds=[[-1, .2, 1], [.8, -.5], None], discrimination=[1.3, .7, None],
        slopes=[None, None, [0, -.8, 1.2]], intercepts=[None, None, [0, .5, -.2]],
        scores=[None, None, [3, 0, 3]])


def numpy_probabilities(state, theta):
    result = []
    for d in state["definitions"]:
        if d["family"] == "binary":
            p = d["guessing"]+(d["upper"]-d["guessing"])*expit(d["discrimination"]*(theta-d["difficulty"]))
            result.append(np.column_stack([1-p, p]))
        elif d["family"] == "grm":
            cumulative = np.column_stack([np.ones(len(theta)), expit(d["discrimination"]*(theta[:, None]-np.array(d["thresholds"])[None, :])), np.zeros(len(theta))])
            result.append(cumulative[:, :-1]-cumulative[:, 1:])
        else:
            if d["family"] == "gpcm":
                logits = np.column_stack([np.zeros(len(theta)), np.cumsum(d["discrimination"]*(theta[:, None]-np.array(d["thresholds"])[None, :]), axis=1)])
            else:
                logits = theta[:, None]*np.array(d["slopes"])+np.array(d["intercepts"])
            result.append(np.exp(logits-logsumexp(logits, axis=1)[:, None]))
    return result


@pytest.mark.parametrize("factory", [binary, poly])
def test_known_bank_probabilities_match_independent_numpy(factory):
    bank = factory()
    state = _bank(bank)
    theta = np.linspace(-4, 4, 39)
    actual = _logp(state, torch.tensor(theta, dtype=torch.float64))
    for logp, expected in zip(actual, numpy_probabilities(state, theta)):
        np.testing.assert_allclose(logp.exp(), expected, rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(logp.exp().sum(1), 1, atol=3e-15)
        assert logp.dtype == torch.float64 and logp.device.type == "cpu"
    assert "known" in bank.attrs["calibration"]
    assert not any("std_error" in f.columns for f in bank.values())


def test_extreme_logits_and_arbitrarily_positive_grm_gap_are_finite():
    bank = irt_bank_binary(["x"], [5], [-8])
    logp = _logp(_bank(bank), torch.tensor([-12., 12.], dtype=torch.float64))[0]
    assert torch.isfinite(logp).all()
    assert float(logp[1, 0]) == pytest.approx(-100)
    graded = irt_bank_polytomous(["x"], family="grm", thresholds=[[0., 5e-324]], discrimination=[.1])
    logp = _logp(_bank(graded), torch.tensor([-8., 0., 8.], dtype=torch.float64))[0]
    assert torch.isfinite(logp).all()
    assert float(logp[1, 1]) == pytest.approx(np.log(5e-324)+np.log(.1)-np.log(4))


def test_category_code_and_gapped_duplicate_nominal_scores_are_distinct():
    bank = poly()
    state = _bank(bank)
    assert state["definitions"][2]["scores"] == [3, 0, 3]
    response, indices = _responses(pd.DataFrame({"g": [3], "p": [1], "n": [2]}, index=["person"]), state)
    assert response.tolist() == [[3, 1, 2]] and indices == ["person"]
    with pytest.raises(AnalysisError):
        _responses({"g": [3], "p": [1], "n": [3]}, state)


@pytest.mark.parametrize("factory", [binary, poly])
def test_exact_finite_bayes_independent_enumeration_partial_all_missing(factory):
    bank = factory()
    state = _bank(bank)
    items = state["items"]
    codes = [[0, 1, 0], [1, -1, 1], [-1, -1, -1]]
    frame = pd.DataFrame([[None if v < 0 else v for v in row] for row in codes], columns=items, index=["A", "B", "A"])
    support, masses = [-2., -.3, 1., 2.5], [.1, .2, .5, .2]
    result = irt_posterior(bank, data=frame, support=support, masses=masses)
    probability = numpy_probabilities(state, np.array(support))
    joint = np.tile(masses, (len(codes), 1))
    for i, row in enumerate(codes):
        for j, code in enumerate(row):
            if code >= 0:
                joint[i] *= probability[j][:, code]
    evidence = joint.sum(1)
    posterior = joint/evidence[:, None]
    saved = _posterior(result)
    np.testing.assert_allclose(saved["posterior"], posterior, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(saved["log_evidence"], np.log(evidence), atol=2e-15)
    mean = posterior@np.array(support)
    sd = np.sqrt(np.sum(posterior*(np.array(support)-mean[:, None])**2, axis=1))
    entropy = -np.sum(posterior*np.log(posterior), axis=1)
    np.testing.assert_allclose(result["summary"].posterior_mean, mean, atol=1e-14)
    np.testing.assert_allclose(result["summary"].posterior_sd, sd, atol=1e-14)
    np.testing.assert_allclose(result["summary"].entropy, entropy, atol=1e-14)
    for name, prob in (("q025", .025), ("q500", .5), ("q975", .975)):
        expected = [support[np.searchsorted(row.cumsum(), prob, side="left")] for row in posterior]
        np.testing.assert_array_equal(result["summary"][name], expected)
    assert saved["responses"] == codes and saved["indices"] == ["A", "B", "A"]
    np.testing.assert_allclose(saved["posterior"][2], masses, atol=2e-15)
    assert result.attrs["sample_positions"] == [0, 1, 2]


def test_one_point_prior_and_discrete_quantile_left_tie():
    bank = irt_bank_binary(["x"], [1], [0])
    result = irt_posterior(bank, data={"x": [None]}, support=[2.], masses=[1.])
    assert result["summary"].posterior_mean.iloc[0] == 2
    assert result["summary"].posterior_sd.iloc[0] == 0
    assert result["summary"].entropy.iloc[0] == 0
    result = irt_posterior(bank, data={"x": [None]}, support=[-1., 1.], masses=[.5, .5])
    assert result["summary"].q500.iloc[0] == -1


@pytest.mark.parametrize("factory", [binary, poly])
def test_full_bank_and_posterior_json_latex_restore(factory):
    bank = factory()
    data = dict.fromkeys(bank.attrs["bank_state"]["items"], [None, 0, 1])
    posterior = irt_posterior(bank, data=data, support=[-2, 0, 2], masses=[.2, .5, .3])
    for result, restore in ((bank, irt_bank_restore), (posterior, irt_posterior_restore)):
        encoded = summary_state(result)
        for restored in (restore(encoded), restore(restore_summary(encoded)), restore_summary(encoded)):
            assert summary_state(restored) == encoded
            assert restored.to_latex() == result.to_latex()
            for name in result:
                pd.testing.assert_frame_equal(restored[name], result[name], check_dtype=False)


@pytest.mark.parametrize("options", [
    {"discrimination": [0.]}, {"discrimination": [5.1]}, {"difficulty": [9.]},
    {"guessing": [.5]}, {"upper": [.5]}, {"discrimination": [True]},
    {"difficulty": [float("inf")]}, {"difficulty": [10**10000]},
    {"items": ["x"*257]}, {"discrimination": (1.,)}, {"items": []},
])
def test_invalid_binary_primitive_schema(options):
    args = {"items": ["x"], "discrimination": [1.], "difficulty": [0.]}
    args.update(options)
    with pytest.raises(AnalysisError):
        irt_bank_binary(**args)


@pytest.mark.parametrize("options", [
    {"family": "grm", "thresholds": [[0, 0]]},
    {"family": "gpcm", "thresholds": [[0]], "discrimination": [-1]},
    {"family": "nrm", "slopes": [[1, 2]], "intercepts": [[0, 0]], "scores": [[0, 1]]},
    {"family": "nrm", "slopes": [[0, 0]], "intercepts": [[0, 0]], "scores": [[0, 1]]},
    {"family": "nrm", "slopes": [[0, 1]], "intercepts": [[0, 0]]},
    {"family": "nrm", "slopes": [[0, 1]], "intercepts": [[0, 0]], "scores": [[0, 1.5]]},
    {"family": "nrm", "slopes": [[0, 1]], "intercepts": [[0, 0]], "scores": [[0, 11]]},
    {"family": "grm", "thresholds": [[0]], "slopes": [[0, 1]]},
    {"family": "unknown", "thresholds": [[0]]},
])
def test_invalid_polytomous_schema(options):
    with pytest.raises(AnalysisError):
        irt_bank_polytomous(["x"], **options)


@pytest.mark.parametrize("support,masses", [([0, 0], [.5, .5]), ([1, 0], [.5, .5]), ([-9, 0], [.5, .5]), ([0, 1], [0, 1]), ([0, 1], [.4, .5]), ([0, 1], [.4, True]), ((0, 1), [.5, .5])])
def test_invalid_finite_prior(support, masses):
    with pytest.raises(AnalysisError):
        irt_posterior(binary(), data={"a": [0], "b": [1], "c": [0]}, support=support, masses=masses)


@pytest.mark.parametrize("value", [-1, True, 1.2, float("inf"), "1"])
def test_observed_invalid_response_is_not_missing(value):
    bank = irt_bank_binary(["x"], [1], [0])
    with pytest.raises(AnalysisError):
        irt_posterior(bank, data={"x": [value]}, support=[0.], masses=[1.])


@pytest.mark.parametrize("field", ["posterior", "log_evidence", "responses", "support", "bank"])
def test_rehashed_numerically_incoherent_posterior_refused(field):
    bank = binary()
    result = irt_posterior(bank, data={"a": [0], "b": [1], "c": [1]}, support=[-2, 0, 2], masses=[.2, .5, .3])
    result = copy.deepcopy(result)
    state = result.attrs["posterior_state"]
    if field == "posterior":
        state[field][0][0] += .01
        state[field][0][1] -= .01
    elif field == "bank":
        state[field]["definitions"][0]["difficulty"] += .2
    elif field == "responses":
        state[field][0][0] = 1
    elif field == "support":
        state[field][0] -= .1
    else:
        state[field][0] -= .1
    result.attrs["state_sha256"] = _seal(state)
    with pytest.raises(AnalysisError) as error:
        irt_posterior_restore(result)
    assert error.value.code == "invalid_state"


def test_bank_table_and_metadata_tampering_refused():
    bank = binary()
    bank["parameters"].iloc[0, -1] += .1
    with pytest.raises(AnalysisError) as error:
        irt_bank_restore(bank)
    assert error.value.code == "invalid_state"
    bank = binary()
    bank.attrs["extra"] = "9"*100000
    with pytest.raises(AnalysisError) as error:
        _bank(bank)
    assert error.value.code == "invalid_state"


def test_dimension_guard_before_forged_frame_conversion(monkeypatch):
    bank = binary()
    bank["parameters"] = table([[1]], columns=["wrong"])
    def no_conversion(*args, **kwargs):
        raise AssertionError("Forged frame converted before dimension admission")
    monkeypatch.setattr(bank["parameters"], "to_numpy", no_conversion)
    with pytest.raises(AnalysisError) as error:
        irt_bank_restore(bank)
    assert error.value.code == "invalid_state"


def test_primitive_posterior_input_and_aggregate_budget_before_tensors(monkeypatch):
    result = irt_posterior(binary(), data={"a": [0], "b": [1], "c": [1]}, support=[-2, 0, 2], masses=[.2, .5, .3])
    def no_tensor(*args, **kwargs):
        raise AssertionError("Numerical allocation before aggregate admission")
    monkeypatch.setattr(torch, "tensor", no_tensor)
    assert len(_posterior_input(result)["support"]) == 3
    for options in ({"max_bytes": 1}, {"max_work": 1}, {"extra_work": 300_000_000}, {"extra_buffers": {"root": 10**9}}):
        with pytest.raises(AnalysisError) as error:
            _posterior(result, **options)
        assert error.value.code in ("resource_limit", "workspace_limit")


def test_invalid_bank_refused_before_hash_and_tensor(monkeypatch):
    bank = binary()
    bank.attrs["bank_state"]["definitions"][0]["difficulty"] = "9"*100000
    def no_hash(*args, **kwargs):
        raise AssertionError("Invalid primitives hashed before admission")
    monkeypatch.setattr(module, "_seal", no_hash)
    with pytest.raises(AnalysisError):
        _bank(bank)


def test_before_selection_rows_device_and_global_workspace_refusals():
    bank = binary()
    with pytest.raises(AnalysisError) as error:
        irt_posterior(bank, data={name: [0]*1001 for name in ["a", "b", "c"]}, support=[0], masses=[1])
    assert error.value.code == "resource_limit"
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            irt_posterior(bank, data={name: [0]*1000 for name in ["a", "b", "c"]}, support=np.linspace(-8, 8, 101).tolist(), masses=[1/101]*101)
        assert error.value.code == "workspace_limit"
    with pytest.raises(AnalysisError):
        irt_bank_binary(["x"], [1], [0], device="mps")
    with pytest.raises(AnalysisError):
        irt_posterior(bank, data={}, support=[0], masses=[1], device="cuda")


def test_default_meta_device_cannot_move_public_bank_posterior_or_restore():
    previous = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        bank = poly()
        result = irt_posterior(bank, data={name: [None, 0] for name in ["g", "p", "n"]}, support=[-2, 0, 2], masses=[.2, .5, .3])
        assert irt_posterior_restore(summary_state(result)).attrs["device"] == "cpu"
    finally:
        torch.set_default_device(previous)


def test_malformed_summary_json_and_fitted_route_refused():
    with pytest.raises(AnalysisError):
        irt_bank_restore('{"schema":"openecon.summary.v1","attrs":{},"tables":{},"title":"x"}')
    bank = binary()
    value = json.loads(summary_state(bank))
    value["tables"]["parameters"]["data"] = 2
    with pytest.raises(AnalysisError) as error:
        irt_bank_restore(json.dumps(value))
    assert error.value.code == "invalid_state"


def test_giant_response_integer_and_scalar_column_refused_safely():
    bank = irt_bank_binary(["x"], [1], [0])
    for data in ({"x": [10**10000]}, pd.DataFrame({"x": [10**10000]}, dtype=object), {"x": "0"}):
        with pytest.raises(AnalysisError):
            irt_posterior(bank, data=data, support=[0.], masses=[1.])


def test_tensor_cells_do_not_implicitly_transfer():
    bank = irt_bank_binary(["x"], [1], [0])
    with pytest.raises(AnalysisError) as error:
        irt_posterior(bank, data=pd.DataFrame({"x": [torch.tensor(0, device="meta")]}, dtype=object), support=[0.], masses=[1.])
    assert error.value.code == "unsupported_data"


def test_nonprimitive_state_identity_refused_before_hash(monkeypatch):
    class EqualFamily:
        def __eq__(self, other):
            return True
    bank = binary()
    def no_hash(*args, **kwargs):
        raise AssertionError("Nonprimitive state hashed")
    monkeypatch.setattr(module, "_seal", no_hash)
    for field in ("kind", "family"):
        forged = copy.deepcopy(bank)
        if field == "kind":
            forged.attrs["bank_state"]["kind"] = EqualFamily()
        else:
            forged.attrs["bank_state"]["definitions"][0]["family"] = EqualFamily()
        with pytest.raises(AnalysisError) as error:
            _bank(forged)
        assert error.value.code == "invalid_state"


def test_bounded_bank_metadata_refused_before_settings_serialization(monkeypatch):
    bank = binary()
    bank.attrs["sources"] = ["x"*2048]*1000
    def no_hash(*args, **kwargs):
        raise AssertionError("Oversized metadata reached serialization")
    monkeypatch.setattr(module, "_seal", no_hash)
    with pytest.raises(AnalysisError) as error:
        _bank(bank)
    assert error.value.code == "invalid_state"


def test_matching_shape_tensor_and_nested_json_cells_refused_before_expansion(monkeypatch):
    bank = irt_bank_binary(["x"], [1], [0])
    forged = copy.deepcopy(bank)
    forged["parameters"] = forged["parameters"].astype(object)
    forged["parameters"].iat[0, 2] = torch.arange(1000, dtype=torch.float64)
    def no_tensor_expansion(*args, **kwargs):
        raise AssertionError("Unplanned tensor cell expanded")
    monkeypatch.setattr(torch.Tensor, "tolist", no_tensor_expansion)
    with pytest.raises(AnalysisError) as error:
        irt_bank_restore(forged, max_bytes=70000)
    assert error.value.code == "invalid_state"
    payload = json.loads(summary_state(bank))
    payload["tables"]["parameters"]["data"][0][2] = [1]*1000
    with pytest.raises(AnalysisError) as error:
        irt_bank_restore(json.dumps(payload))
    assert error.value.code == "invalid_state"


def test_mapping_series_are_positional_and_numpy_missing_is_preserved():
    bank = irt_bank_binary(["a", "b"], [1, 1], [0, 0])
    data = {"a": pd.Series([0, 1], index=["x", "y"]), "b": pd.Series([1, 0], index=["z", "w"])}
    result = irt_posterior(bank, data=data, support=[0.], masses=[1.])
    state = _posterior(result)
    assert state["responses"] == [[0, 1], [1, 0]] and state["indices"] == [0, 1]
    response, _ = _responses({"a": np.array([np.nan]), "b": [None]}, _bank(bank))
    assert response.tolist() == [[-1, -1]]


def test_unicode_full_summary_escape_bound_before_numerical_allocation(monkeypatch):
    bank = irt_bank_binary(["x"], [1], [0])
    data = pd.DataFrame({"x": [None]*600}, index=["🌍"*256]*600)
    def no_tensor(*args, **kwargs):
        raise AssertionError("Unrestorable output admitted before tensor allocation")
    monkeypatch.setattr(torch, "tensor", no_tensor)
    with pytest.raises(AnalysisError) as error:
        irt_posterior(bank, data=data, support=[0.], masses=[1.])
    assert error.value.code == "resource_limit"
