"""Lossless source, budget and adversarial semantic-replay acceptance."""

import copy
import inspect
import json

import numpy as np
import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.weakiv import (
    CLRConfidenceSet,
    CLRTestState,
    iv_clr_confidence_set,
    iv_clr_restore,
    iv_clr_test,
    iv_clr_test_restore,
)
from openecon.econometrics.weakiv import common, kernels
from openecon.econometrics.weakiv import api
from openecon.resources import use_workspace_budget
from test_weakiv_clr_math import data, fit


def rehash(state):
    state["sha256"] = common.digest({k: v for k, v in state.items() if k != "sha256"})
    return state


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(7, 167, 2, name="physical"),
        pd.Index([f"row{i // 2}" for i in range(80)], name="duplicate labels"),
        pd.date_range("2022-01-01", periods=80, freq="D", tz="Europe/Istanbul", name="date"),
        pd.timedelta_range("1 hour", periods=80, freq="h", name="elapsed"),
        pd.period_range("2020-01", periods=80, freq="M", name="month"),
        pd.CategoricalIndex(
            ["b", "a"] * 40, categories=["a", "b", "unused"], ordered=True, name="group"
        ),
        pd.MultiIndex.from_arrays(
            [pd.period_range("2020-01", periods=80, freq="M"), ["a", "b"] * 40],
            names=["month", "group"],
        ),
    ],
)
def test_typed_index_and_physical_missing_sample_roundtrip(index):
    f = data(4, 1)
    f.index = index
    f["z1"] = f.z1.astype("Float32")
    f["count"] = pd.Series(np.arange(80), index=index, dtype="Int32")
    f.iloc[[1, 24], f.columns.get_loc("z1")] = pd.NA
    out = iv_clr_confidence_set(
        f, "y", "d", ["count"], instruments=["z0", "z1", "z2"], missing="drop"
    )
    state = out.attrs["state"]
    assert state["positions"] == [i for i in range(80) if i not in (1, 24)]
    assert state["source"]["dtypes"] == [
        "float64",
        "float64",
        "Int32",
        "float64",
        "Float32",
        "float64",
    ]
    typed = CLRConfidenceSet(payload=state)
    restored = CLRConfidenceSet.model_validate_json(typed.model_dump_json()).to_tables()
    assert restored.attrs["state"] == state
    assert iv_clr_restore(json.dumps(state)).attrs["state"] == state
    query = iv_clr_test(typed, null=0.7)
    qtyped = CLRTestState(payload=query.attrs["state"])
    assert (
        CLRTestState.model_validate_json(qtyped.model_dump_json()).to_tables().attrs["state"]
        == query.attrs["state"]
    )


@pytest.mark.parametrize(
    "change",
    ["omega", "gram", "scales", "gap", "quadratic", "endpoint", "probability", "sample", "plan"],
)
def test_rehashed_source_state_cache_forgery_is_rejected(change):
    f = data(4, 1)
    f["y"] *= 1e-6
    state = copy.deepcopy(fit(f).attrs["state"])
    if change == "omega":
        state["geometry"]["reduced_form_covariance"][0][0] *= 1.2
    elif change == "gram":
        state["geometry"]["projected_gram"][0][0] *= 1.2
    elif change == "scales":
        state["geometry"]["response_scales"][0] *= 1.2
    elif change == "gap":
        state["geometry"]["eigen_gap"] *= 1.2
    elif change == "quadratic":
        state["solution"]["normalized_inequality"][0][0] *= 1.2
    elif change == "endpoint":
        state["solution"]["intervals"][0][0] *= 1.2
    elif change == "probability":
        state["root"]["trace"][0]["value"] *= 0.8
    elif change == "sample":
        state["positions"].pop()
    else:
        state["resource_plan"]["buffers"]["source copies and typed index"] += 1
    rehash(state)
    with pytest.raises(AnalysisError):
        iv_clr_restore(state)


@pytest.mark.parametrize("kind", ["copy", "construct"])
def test_forged_live_typed_state_cannot_bypass_any_reader_or_serializer(kind):
    out = fit(data(4, 1))
    typed = CLRConfidenceSet(payload=out.attrs["state"])
    state = copy.deepcopy(out.attrs["state"])
    state["geometry"]["reduced_form_covariance"][0][0] *= 4
    rehash(state)
    forged = (
        BaseModel.model_copy(typed, update={"payload": state})
        if kind == "copy"
        else CLRConfidenceSet.model_construct(payload=state)
    )
    for reader in [iv_clr_restore, lambda v: v.to_tables(), lambda v: iv_clr_test(v, null=0.7)]:
        with pytest.raises(AnalysisError):
            reader(forged)
    for serializer in [lambda v: v.model_dump(), lambda v: v.model_dump_json()]:
        with pytest.raises(Exception, match="numeric value differs"):
            serializer(forged)


@pytest.mark.parametrize("change", ["probability", "statistic", "decision", "null", "target"])
def test_complete_query_target_and_rehashed_query_caches_are_replayed(change):
    out = fit(data(4, 1))
    state = copy.deepcopy(iv_clr_test(out, null=0.7).attrs["state"])
    if change == "target":
        state["target"]["geometry"]["reduced_form_covariance"][1][1] *= 4
        rehash(state["target"])
    elif change == "decision":
        state["test"]["accepted"] = not state["test"]["accepted"]
    elif change == "null":
        state["test"]["null"] = 5.0
    else:
        state["test"]["conditional_p_value" if change == "probability" else "statistic"] *= 0.5
    rehash(state)
    with pytest.raises(AnalysisError):
        iv_clr_test_restore(state)
    forged = CLRTestState.model_construct(payload=state)
    with pytest.raises(AnalysisError):
        forged.to_tables()
    with pytest.raises(Exception):
        forged.model_dump_json()


@pytest.mark.parametrize(
    "change",
    [
        "float_to_int",
        "float32_rounding",
        "int_primitive",
        "index_extra",
        "source_extra",
        "shape",
        "integer_huge",
    ],
)
def test_source_or_cache_preadmission_refuses_dtype_coercion_and_shapes(change, monkeypatch):
    f = data(4, 1)
    state = copy.deepcopy(fit(f).attrs["state"])
    if change == "float_to_int":
        state["source"]["dtypes"][0] = "int64"
    elif change == "float32_rounding":
        state["source"]["dtypes"][0] = "float32"
    elif change == "int_primitive":
        state["source"]["values"][0][0] = 1
    elif change == "index_extra":
        state["source"]["index"]["extra"] = "unrecognized"
    elif change == "source_extra":
        state["source"]["extra"] = 0
    elif change == "shape":
        state["geometry"]["whitened_gram"] = [[1.0]]
    else:
        state["source"]["values"][0][0] = 1 << 4096
    if change != "integer_huge":
        rehash(state)

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted state reached source tensor reconstruction")

    monkeypatch.setattr(kernels, "tensor", forbidden)
    with pytest.raises(AnalysisError):
        iv_clr_restore(state)


@pytest.mark.parametrize("dtype", ["bool", "complex128", "object", "category"])
def test_unsupported_source_dtype_is_explicit(dtype):
    f = data()
    f["y"] = f.y.astype(dtype)
    with pytest.raises(AnalysisError) as exc:
        fit(f)
    assert exc.value.code == "unsupported_dtype"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(y="_cons"),
        dict(endog="y"),
        dict(instruments=[]),
        dict(x="z0"),
        dict(device="cuda"),
        dict(dtype="float32"),
        dict(missing="automatic"),
        dict(omega=[[1.0, 2.0], [2.0, 1.0]]),
        dict(max_order=129),
        dict(probability_tolerance=1e-5),
        dict(confidence=1.0),
    ],
)
def test_explicit_domain_options(kwargs):
    options = dict(y="y", endog="d", instruments=["z0", "z1", "z2"])
    options.update(kwargs)
    with pytest.raises(AnalysisError):
        iv_clr_confidence_set(data(), **options)


@pytest.mark.parametrize("kind", ["rows", "work", "workspace", "state_metadata"])
def test_resource_refusal_precedes_tensor_allocation(kind, monkeypatch):
    f = data()
    out = fit(f)

    def forbidden(*args, **kwargs):
        pytest.fail("Resource refusal happened after tensor allocation")

    monkeypatch.setattr(kernels, "tensor", forbidden)
    with pytest.raises(AnalysisError) as exc:
        if kind == "rows":
            fit(pd.concat([f] * 126, ignore_index=True))
        elif kind == "work":
            fit(f, max_work=1)
        elif kind == "workspace":
            with use_workspace_budget(1):
                fit(f)
        else:
            state = copy.deepcopy(out.attrs["state"])
            state["extra"] = ["x" * 10000] * 1000
            with use_workspace_budget(1):
                iv_clr_restore(state)
    assert exc.value.code in {
        "shape_limit",
        "invalid_argument",
        "work_limit",
        "workspace_limit",
        "state_limit",
    }


def test_default_device_dtype_and_ambient_rng_are_preserved():
    out = fit(data())
    before = torch.random.get_rng_state().clone()
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            restored = iv_clr_restore(out)
            assert restored.attrs["state"] == out.attrs["state"]
            iv_clr_test(out, null=0.7)
    finally:
        torch.set_default_dtype(default_dtype)
    assert torch.equal(before, torch.random.get_rng_state())


def test_missing_nonfinite_and_singular_admission():
    f = data()
    f.iloc[0, 0] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        fit(f)
    f.iloc[0, 0] = np.inf
    with pytest.raises(AnalysisError, match="finite"):
        fit(f, missing="drop")
    f = data()
    f["z1"] = f.z0
    with pytest.raises(AnalysisError) as exc:
        fit(f)
    assert exc.value.code == "singular_design"


def test_exact_topology_boundary_is_refused_instead_of_fake_tail_claim():
    g = dict(eigen_gap=4.0, coefficient_unit=1.0, eigen_directions=[[1.0, 0.0], [0.0, 1.0]])
    r = dict(lr=4.0, lr_bracket=[4.0, 4.0])
    with pytest.raises(AnalysisError) as exc:
        kernels.intervals(g, r)
    assert exc.value.code == "unresolved_topology"
    g["eigen_gap"] = 8.0
    g["eigen_directions"] = [[0.0, 1.0], [1.0, 0.0]]
    r.update(lr=4.0, lr_bracket=[3.9, 4.1])
    assert kernels.intervals(g, r)["topology"] == "two_unbounded_rays"


def test_state_hash_is_not_an_authority_for_arbitrary_inference():
    state = copy.deepcopy(fit(data()).attrs["state"])
    state["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="integrity"):
        CLRConfidenceSet(payload=state)
    with pytest.raises(ValidationError):
        CLRConfidenceSet(payload=fit(data()).attrs["state"], extra=True)


def test_dense_json_metadata_refusal_happens_before_decoding(monkeypatch):
    text = "[" + ",".join(["{}"] * 10000) + "]"

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted dense JSON reached the allocating decoder")

    monkeypatch.setattr(common.json, "loads", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        iv_clr_restore(text)
    assert exc.value.code == "workspace_limit"


def test_too_deep_json_is_refused_before_decoding(monkeypatch):
    text = "[" * 70 + "0" + "]" * 70

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted deep JSON reached the allocating decoder")

    monkeypatch.setattr(common.json, "loads", forbidden)
    with pytest.raises(AnalysisError) as exc:
        iv_clr_restore(text)
    assert exc.value.code == "state_limit"


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
@pytest.mark.parametrize("kind", ["dense", "deep"])
def test_typed_json_admission_precedes_any_allocating_decoder(cls, kind, monkeypatch):
    text = (
        '{"payload":{"dense":[' + ",".join(["{}"] * 50000) + "]}}"
        if kind == "dense"
        else '{"payload":' + "[" * 70 + "0" + "]" * 70 + "}"
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted typed JSON reached the allocating decoder")

    monkeypatch.setattr(common.json, "loads", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        cls.model_validate_json(text)
    assert exc.value.code == ("workspace_limit" if kind == "dense" else "state_limit")


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
@pytest.mark.parametrize("kind", ["model_copy", "deepcopy", "update"])
def test_deep_copy_and_update_admission_precedes_container_copy(cls, kind, monkeypatch):
    dense = {"dense": [{} for _ in range(50000)]}
    obj = cls.model_construct(payload=dense)
    if kind == "update":
        state = fit(data(), max_order=128).attrs["state"]
        if cls is CLRTestState:
            state = iv_clr_test(state, null=0.7).attrs["state"]
        obj = cls(payload=state)

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted deep copy reached copying allocation")

    monkeypatch.setattr(BaseModel, "__deepcopy__", forbidden)
    with use_workspace_budget(16 if kind == "update" else 1), pytest.raises(AnalysisError) as exc:
        if kind == "deepcopy":
            copy.deepcopy(obj)
        elif kind == "update":
            obj.model_copy(update={"payload": dense}, deep=True)
        else:
            obj.model_copy(deep=True)
    assert exc.value.code == "workspace_limit"


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
def test_complete_typed_transport_options_and_admitted_deep_copy(cls):
    state = fit(data()).attrs["state"]
    if cls is CLRTestState:
        state = iv_clr_test(state, null=0.7).attrs["state"]
    obj = cls(payload=state)
    for indent in [None, 0, 2, 8]:
        encoded = obj.model_dump_json(indent=indent, ensure_ascii=True)
        for text in [encoded, encoded.encode(), bytearray(encoded.encode())]:
            assert (
                cls.model_validate_json(
                    text, strict=True, context={"transport": "test"}, by_name=True
                )
                .to_tables()
                .attrs["state"]
                == state
            )
    for copied in [obj.model_copy(), obj.model_copy(deep=True), copy.deepcopy(obj)]:
        assert copied is not obj
        assert copied.to_tables().attrs["state"] == state
    with pytest.raises(ValidationError):
        cls.model_validate_json(obj.model_dump_json()[:-1] + ',"extra":true}')
    with pytest.raises(ValidationError):
        obj.model_copy(update={"extra": True}, deep=True)


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
@pytest.mark.parametrize("kind", ["size", "workspace"])
def test_json_output_format_admission_precedes_serializer(cls, kind, monkeypatch):
    state = fit(data(4, 1), max_order=128).attrs["state"]
    if cls is CLRTestState:
        state = iv_clr_test(state, null=0.7).attrs["state"]
    obj = cls(payload=state)

    def forbidden(*args, **kwargs):
        pytest.fail("Unadmitted formatting reached the allocating serializer")

    monkeypatch.setattr(BaseModel, "model_dump_json", forbidden)
    with use_workspace_budget(16), pytest.raises(AnalysisError) as exc:
        obj.model_dump_json(indent=10000 if kind == "size" else 1000)
    assert exc.value.code == ("state_limit" if kind == "size" else "workspace_limit")


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
@pytest.mark.parametrize(
    "projection",
    [
        {"exclude": {"payload"}},
        {"include": set()},
        {"include": {"payload": {"sha256"}}},
        {"exclude": {"payload": {"geometry", "target", "test"}}},
    ],
)
@pytest.mark.parametrize("kind", ["construct", "base_copy"])
def test_partial_exports_replay_complete_rehashed_state_before_serializer(
    cls, projection, kind, monkeypatch
):
    target = fit(data(4, 1), max_order=128).attrs["state"]
    state = target if cls is CLRConfidenceSet else iv_clr_test(target, null=0.7).attrs["state"]
    original = cls(payload=state)
    forged_state = copy.deepcopy(state)
    forged_target = forged_state if cls is CLRConfidenceSet else forged_state["target"]
    forged_target["geometry"]["reduced_form_covariance"][0][0] *= 4
    rehash(forged_target)
    if cls is CLRTestState:
        rehash(forged_state)
    forged = (
        cls.model_construct(payload=forged_state)
        if kind == "construct"
        else BaseModel.model_copy(original, update={"payload": forged_state})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Forged partial export reached an allocating Pydantic serializer")

    monkeypatch.setattr(BaseModel, "model_dump", forbidden)
    monkeypatch.setattr(BaseModel, "model_dump_json", forbidden)
    for method in [forged.model_dump, forged.model_dump_json]:
        with pytest.raises(AnalysisError, match="numeric value differs"):
            method(**projection)


@pytest.mark.parametrize("cls", [CLRConfidenceSet, CLRTestState])
def test_valid_partial_exports_preserve_signatures_and_do_not_fit(cls, monkeypatch):
    target = fit(data(4, 1), max_order=128).attrs["state"]
    state = target if cls is CLRConfidenceSet else iv_clr_test(target, null=0.7).attrs["state"]
    obj = cls(payload=state)

    def forbidden(*args, **kwargs):
        pytest.fail("Partial saved-state export reached the source estimator")

    monkeypatch.setattr(api, "iv_clr_confidence_set", forbidden)
    assert inspect.signature(cls.model_dump) == inspect.signature(BaseModel.model_dump)
    assert inspect.signature(cls.model_dump_json) == inspect.signature(BaseModel.model_dump_json)
    assert obj.model_dump(exclude={"payload"}, mode="json", round_trip=True) == {}
    assert obj.model_dump(include=set(), by_alias=True, exclude_unset=True) == {}
    assert json.loads(obj.model_dump_json(exclude={"payload"}, indent=2)) == {}
    # The scientific payload is an opaque complete custom-serialized field;
    # nested selectors retain that existing Pydantic serialization convention.
    expected = {"payload": state}
    assert obj.model_dump(include={"payload"}, warnings="error") == expected
    assert obj.model_dump(include={"payload": {"sha256"}}, warnings="error") == expected
    assert json.loads(obj.model_dump_json(include={"payload": {"sha256"}})) == expected
    assert obj.to_tables().attrs["state"] == state
