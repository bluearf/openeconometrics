"""Independent recipe arithmetic, source preservation and scientific-state refusal."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import MIResult, _digest
from openecon.econometrics.mi.generation import mi_monotone
from openecon.econometrics.mi.passive import MIPassiveResult, _expand, mi_passive
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source():
    data = pd.DataFrame({"x": [0., 1., 2., 3., 4., 5., 6., 7.],
                         "y": [2., 1., 4., 3., 6., 4., None, None]},
                        index=pd.Index(["a", "b", "c", "a", "e", "f", "f", "h"], name="unit"))
    return mi_monotone(data, ["x", "y"], m=3, seed=57)


def recipes():
    return {
        "z": {"operation": "affine", "inputs": ["x", "y"], "coefficients": [2., -.5], "intercept": 3.},
        "xy": {"operation": "product", "inputs": ["x", "y"]},
        "z2": {"operation": "power", "inputs": ["z"], "exponent": 2},
    }


def large_source():
    data = pd.DataFrame({"x": [np.sin(i * .4) for i in range(64)],
                         "y": [1 + np.sin(i * .4) + np.cos(i * 1.3) for i in range(64)]})
    data.loc[52:, "y"] = np.nan
    return mi_monotone(data, ["x", "y"], m=3, seed=93)


def test_independent_full_matrix_oracle_and_original_null_geometry(source):
    before = source.model_dump_json()
    result = mi_passive(source, recipes())
    a = np.asarray(source.completed_matrices)
    z = 2 * a[..., 0] - .5 * a[..., 1] + 3
    expected = np.concatenate([a, z[..., None], (a[..., 0] * a[..., 1])[..., None], (z ** 2)[..., None]], axis=-1)
    np.testing.assert_array_equal(np.asarray(result.result.completed_matrices), expected)
    original = np.asarray(source.original, dtype=float)
    assert result.result.original[-1] == (*source.original[-1], None, None, None)
    assert result.result.original[0] == (0., 2., 2., 0., 4.)
    for i in range(1, 4):
        assert result.dataset(i).index.equals(source.dataset(i).index)
        pd.testing.assert_frame_equal(result.dataset(i).loc[:, ["x", "y"]], source.dataset(i))
    assert source.model_dump_json() == before
    assert np.isnan(original[-1, 1])
    assert result.metadata["new_stochastic_draws"] == 0
    assert result.metadata["fcs_feedback"] is False
    assert result.table["column"].tolist() == ["z", "xy", "z2"]
    assert "tabular" in result.latex


def test_full_json_restoration_and_detached_accessors(source):
    result = mi_passive(source, recipes())
    restored = MIPassiveResult.model_validate_json(result.model_dump_json())
    assert restored == result
    pd.testing.assert_frame_equal(restored.dataset(2), result.dataset(2))
    changed = restored.recipes
    changed["z"]["coefficients"][0] = 99
    assert restored.recipes["z"]["coefficients"][0] == 2
    frame = restored.dataset()
    frame.iloc[0, 0] = 999
    assert restored.dataset().iloc[0, 0] == 0


@pytest.mark.parametrize("bad", [
    {}, {"x": {"operation": "product", "inputs": ["x", "y"]}},
    {"a": {"operation": "power", "inputs": ["future"], "exponent": 2}},
    {"a": {"operation": "power", "inputs": ["a"], "exponent": 2}},
    {"a": {"operation": "eval", "inputs": ["x"], "expression": "x+1"}},
    {"a": {"operation": "power", "inputs": ["x"], "exponent": True}},
    {"a": {"operation": "power", "inputs": ["x"], "exponent": 0}},
    {"a": {"operation": "power", "inputs": ["x"], "exponent": 9}},
    {"a": {"operation": "affine", "inputs": ["x"], "coefficients": [True], "intercept": 0}},
    {"a": {"operation": "affine", "inputs": ["x"], "coefficients": [1j], "intercept": 0}},
    {"a": {"operation": "affine", "inputs": ["x"], "coefficients": [], "intercept": 0}},
    {"a": {"operation": "product", "inputs": ["x"]}},
    {"a": {"operation": "power", "inputs": ["x", "y"], "exponent": 2}},
    {"a": {"operation": [], "inputs": ["x"]}},
    {"a": {"operation": {}, "inputs": ["x"]}},
    {"a": {"operation": "affine", "inputs": ["x"], "coefficients": [10**1000], "intercept": 0}},
])
def test_explicit_recipe_refusals_before_numeric_allocation(source, bad, monkeypatch):
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated for invalid recipe"))
    with pytest.raises(AnalysisError):
        mi_passive(source, bad)


def test_workspace_and_work_gates_before_buffers(source, monkeypatch):
    large = large_source()
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated past gate"))
    with pytest.raises(AnalysisError, match="max_work"):
        mi_passive(source, recipes(), max_work=1)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        mi_passive(large, recipes())
    assert caught.value.code == "workspace_limit"


def test_replay_respects_current_budget_without_requiring_identical_old_budget(source):
    result = mi_passive(large_source(), recipes())
    with use_workspace_budget(100):
        assert MIPassiveResult.model_validate_json(result.model_dump_json()) == result
    with use_workspace_budget(1), pytest.raises(ValueError, match="workspace"):
        MIPassiveResult.model_validate_json(result.model_dump_json())


@pytest.mark.parametrize("field", ["derived", "source", "recipe", "feedback", "new_draws", "work"])
def test_recomputed_checksum_cannot_forge_scientific_state(source, field):
    result = mi_passive(source, recipes())
    state = copy.deepcopy(result.model_dump(mode="json"))
    if field == "derived":
        state["result"]["completed_matrices"][0][0][-1] += 1
        state["result"]["integrity_sha256"] = _digest({k: v for k, v in state["result"].items() if k != "integrity_sha256"})
    elif field == "source":
        state["source"]["completed_matrices"][0][-1][-1] += 1
        state["source"]["integrity_sha256"] = _digest({k: v for k, v in state["source"].items() if k != "integrity_sha256"})
    elif field == "recipe":
        formula = json.loads(state["recipes_json"])
        formula["z"]["coefficients"][0] = 7
        state["recipes_json"] = json.dumps(formula)
    else:
        meta = json.loads(state["metadata_json"])
        key, value = {"feedback": ("fcs_feedback", True), "new_draws": ("new_stochastic_draws", 1),
                      "work": ("work_estimate", 1)}[field]
        meta[key] = value
        state["metadata_json"] = json.dumps(meta)
    state["integrity_sha256"] = _digest({k: v for k, v in state.items() if k != "integrity_sha256"})
    with pytest.raises((ValueError, AnalysisError)):
        MIPassiveResult.model_validate(state)


def test_copy_cannot_skip_integrity_and_no_global_rng(source):
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    result = mi_passive(source, recipes())
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        result.model_copy(update={"recipes_json": "{}"})


def test_ambient_tensor_defaults_do_not_change_passive_arithmetic(source):
    expected = mi_passive(source, recipes()).result.completed_matrices
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    try:
        torch.set_default_device("meta")
        torch.set_default_dtype(torch.float32)
        assert mi_passive(source, recipes()).result.completed_matrices == expected
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("field,bad", [
    ("metadata_json", "[]"), ("metadata_json", "null"),
    ("metadata_json", "42"), ("metadata_json", '"not an object"'),
    ("recipes_json", "[]"), ("recipes_json", "null"),
])
def test_json_objects_are_required_before_numeric_buffers(source, field, bad, monkeypatch):
    state = mi_passive(source, recipes()).model_dump(mode="json")
    state[field] = bad
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated for a non-object envelope"))
    with pytest.raises(ValueError, match="JSON object"):
        MIPassiveResult.model_validate(state)


def test_nonmapping_resource_plan_and_unchecked_source_shapes_are_controlled(source, monkeypatch):
    state = mi_passive(source, recipes()).model_dump(mode="json")
    meta = json.loads(state["metadata_json"])
    meta["resource_plan"] = []
    state["metadata_json"] = json.dumps(meta)
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("allocated for invalid envelopes"))
    with pytest.raises(ValueError, match="resource plan"):
        MIPassiveResult.model_validate(state)
    for field, bad in (("columns", (["x"], "y")), ("metadata", []), ("original", None)):
        raw = source.model_dump()
        raw[field] = bad
        with pytest.raises((ValueError, AnalysisError)):
            mi_passive(MIResult.model_construct(**raw), recipes())


@pytest.mark.parametrize("operation,matrix,recipe,error", [
    ("product", [[1e-200, 1e-200], [1., 2.]], {"inputs": ["x", "y"]}, "underflow"),
    ("power", [[1e-140], [1.]], {"inputs": ["x"], "exponent": 3}, "underflow"),
    ("affine", [[1e-140], [1.]], {"inputs": ["x"], "coefficients": [1e-200], "intercept": 1.}, "underflow"),
    ("product", [[1e140, 1e140, 1e140]], {"inputs": ["x", "y", "z"]}, "overflow"),
    ("power", [[1e140]], {"inputs": ["x"], "exponent": 3}, "overflow"),
])
def test_nonzero_float64_underflow_and_overflow_are_refused(operation, matrix, recipe, error):
    values = torch.tensor(matrix, dtype=torch.float64, device="cpu")
    columns = ["x", "y", "z"][:values.shape[-1]]
    with pytest.raises(AnalysisError, match=error):
        _expand(values, columns, {"derived": {"operation": operation, **recipe}})


def test_true_zero_cancellation_subnormals_and_null_propagation_are_distinct():
    matrix = torch.tensor([[0., 2.], [-1., 1.], [1e-150, 1e-150], [float("nan"), 2.]],
                          dtype=torch.float64, device="cpu")
    derived = _expand(matrix, ["x", "y"], {
        "product": {"operation": "product", "inputs": ["x", "y"]},
        "sum": {"operation": "affine", "inputs": ["x", "y"], "coefficients": [1., 1.], "intercept": 0.},
        "power": {"operation": "power", "inputs": ["x"], "exponent": 2},
    }).numpy()
    assert derived[0, 2] == 0 and derived[0, 4] == 0
    assert derived[1, 3] == 0
    assert derived[2, 2] > 0 and derived[2, 4] > 0
    assert np.isnan(derived[-1, 2:]).all()


@pytest.mark.parametrize("extension", ["discrete", "delta"])
def test_nested_extension_restoration_keeps_semantic_validators(extension):
    from openecon.econometrics.mi.discrete import MIDiscreteResult, mi_discrete
    from openecon.econometrics.mi.sensitivity import MIDeltaResult, mi_delta
    data = pd.DataFrame({"x": np.linspace(-1, 1, 12),
                         "y": [0., 2., 1., 3., 2., 0., 4., 2., 1., 3., None, None]})
    if extension == "discrete":
        source = mi_discrete(data, ["x", "y"], methods={"y": "poisson"}, m=2,
                             seed=51, burn=0, iterations=1, mh_burn=1, mh_steps=3)
        expected_type = MIDiscreteResult
    else:
        source = mi_delta(data, ["x", "y"], target="y", kind="normal", delta=.75, m=2, seed=51)
        expected_type = MIDeltaResult
    result = mi_passive(source, {"y2": {"operation": "power", "inputs": ["y"], "exponent": 2}})
    restored = MIPassiveResult.model_validate_json(result.model_dump_json())
    assert isinstance(restored.source, expected_type)
    assert restored.metadata["source_result_class"] == expected_type.__name__
    assert restored.source.model_dump_json() == source.model_dump_json()
    state = restored.model_dump(mode="json")
    state["source"]["metadata"]["prior"]["family"] = "unsupported prior"
    state["source"]["integrity_sha256"] = _digest({k: v for k, v in state["source"].items() if k != "integrity_sha256"})
    with pytest.raises(ValueError, match="prior|Gaussian"):
        MIPassiveResult.model_validate(state)


def test_current_budget_also_guards_dataset_table_and_copy_replay():
    result = mi_passive(large_source(), recipes())
    with use_workspace_budget(1):
        for action in (result.dataset, lambda: result.table, result.model_copy):
            with pytest.raises(AnalysisError, match="workspace"):
                action()
