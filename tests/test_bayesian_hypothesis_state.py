"""Public typed-copy, primitive/shape/resource gates and complete portable replay."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic_core import PydanticSerializationError
from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian.commands import bayes_linear
from openecon.econometrics.bayesian.core import digest
from openecon.econometrics.bayesian.hypothesis import (
    PosteriorHypothesisComparison,
    PosteriorHypothesisDraws,
    bayes_hypothesis,
    bayes_hypothesis_draws,
    bayes_hypothesis_draws_restore,
    bayes_hypothesis_predict,
    bayes_hypothesis_restore,
)
from openecon.resources import use_workspace_budget

ERRORS = (AnalysisError, ValueError, PydanticSerializationError)


@pytest.fixture
def alternative():
    index = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("b", 2), ("c", 3), ("c", 4)], names=["group", "row"]
    )
    data = pd.DataFrame(
        {
            "y": pd.array([0.3, 0.9, None, 1.4, 0.2], dtype="Float64"),
            "x": pd.array([1, 2, 3, 4, 2], dtype="Int64"),
        },
        index=index,
    )
    return bayes_linear(
        data=data,
        y="y",
        x=["x"],
        missing="drop",
        prior=dict(mean=[0.1, -0.2], scale_matrix=[[1.2, 0.3], [0.3, 0.8]], shape=0.4, scale=0.7),
    )


@pytest.fixture
def comparison(alternative):
    return bayes_hypothesis(
        result=alternative,
        constraints=[[0.0, 1.0]],
        values=[0.2],
        labels=["slope equals 0.2"],
        model_prior_odds=3.0,
    )


def rehash(payload):
    payload["digest"] = digest({k: v for k, v in payload.items() if k != "digest"})
    return payload


def test_full_typed_source_index_dtype_and_sample_survive_serialization(comparison, alternative):
    restored = bayes_hypothesis_restore(saved=comparison.model_dump_json())
    assert restored.payload == comparison.payload
    state = restored.payload["alternative"]["state"]
    assert state["source_dtypes"] == ("Float64", "Int64")
    assert state["sample_positions"] == (0, 1, 3, 4)
    assert state["source_index"]["kind"] == "multi"
    assert restored.payload["alternative"]["integrity_sha256"] == alternative.integrity_sha256
    with pytest.raises(TypeError):
        restored.payload["alpha"] = 0.3
    with pytest.raises(TypeError):
        restored.payload["results"]["null_posterior"]["mean"][0] = 0.3


@pytest.mark.parametrize("constructor", ["copy", "construct"])
@pytest.mark.parametrize(
    "reader", ["restore", "summary", "tables", "predict", "draws", "dump", "json"]
)
def test_typed_copy_and_construct_cannot_render_or_serialize_rehashed_forged_covariance(
    comparison, constructor, reader
):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    null = payload["results"]["null_posterior"]
    null["coefficient_covariance"][0][0] *= 4
    rehash(payload)
    bad = (
        comparison.model_copy(update={"payload": payload})
        if constructor == "copy"
        else PosteriorHypothesisComparison.model_construct(payload=payload)
    )
    actions = {
        "restore": lambda: bayes_hypothesis_restore(saved=bad),
        "summary": bad.summary,
        "tables": bad.to_tables,
        "predict": lambda: bayes_hypothesis_predict(result=bad, data=pd.DataFrame({"x": [1.0]})),
        "draws": lambda: bayes_hypothesis_draws(result=bad, draws=2),
        "dump": bad.model_dump,
        "json": bad.model_dump_json,
    }
    with pytest.raises(ERRORS):
        actions[reader]()


@pytest.mark.parametrize(
    "field",
    [
        "mean",
        "prior_mean",
        "conditional_scale_matrix",
        "coefficient_scale_matrix",
        "coefficient_covariance",
        "basis",
        "free_conditional_scale_matrix",
        "free_mean",
        "credible_intervals",
    ],
)
def test_wrong_cached_dimensions_refuse_before_any_native_tensor(comparison, monkeypatch, field):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    payload["results"]["null_posterior"][field] = []
    rehash(payload)
    import torch

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: pytest.fail("wrong cached dimensions reached tensor allocation"),
    )
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 2**1024])
@pytest.mark.parametrize("field", ["mean", "shape", "free_dimension"])
def test_wrong_cached_primitives_refuse_before_tensor(comparison, monkeypatch, field, value):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    null = payload["results"]["null_posterior"]
    if field == "mean":
        null[field][0] = value
    else:
        null[field] = value
    # Nonfinite values need not be hashed: primitive admission precedes digest.
    import torch

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: pytest.fail("wrong cached primitive reached tensor allocation"),
    )
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


@pytest.mark.parametrize(
    "field",
    [
        "prior_shape",
        "prior_scale",
        "prior_scale_increment",
        "data_scale_increment",
        "shape",
        "scale",
        "log_marginal_likelihood",
    ],
)
def test_rehashed_null_prior_and_normalizer_tampering_is_detected(comparison, field):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    payload["results"]["null_posterior"][field] += 0.1
    rehash(payload)
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


@pytest.mark.parametrize(
    "field",
    [
        "log_bayes_factor_null_alternative",
        "bayes_factor_null_alternative",
        "constraint_prior_log_density",
        "constraint_posterior_log_density",
        "posterior_probability_null",
        "posterior_probability_alternative",
        "log_posterior_model_odds_null_alternative",
    ],
)
def test_rehashed_model_evidence_or_probability_tampering_is_detected(comparison, field):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    payload["results"][field] += 0.1
    rehash(payload)
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


@pytest.mark.parametrize(
    "c,d",
    [
        ([], []),
        ([[0.0, 0.0]], [0.0]),
        ([[1.0, 1.0], [2.0, 2.0]], [0.0, 0.0]),
        ([[1.0, 0.0], [1.0, 1e-9]], [0.0, 0.0]),
        ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, 0.0]),
        ([[True, 1.0]], [0.0]),
        ([[2**1024, 0.0]], [0.0]),
        ([[2**53 + 1, 0.0]], [0.0]),
        ([[1.0, 0.0]], [float("nan")]),
    ],
)
def test_constraint_full_rank_and_float64_domain_are_explicit(alternative, c, d):
    with pytest.raises(ERRORS):
        bayes_hypothesis(result=alternative, constraints=c, values=d)


@pytest.mark.parametrize("odds", [True, 0.0, -1.0, float("nan"), float("inf"), 2**1024])
def test_declared_model_prior_odds_must_be_proper_finite_positive(alternative, odds):
    with pytest.raises(ERRORS):
        bayes_hypothesis(
            result=alternative, constraints=[[0.0, 1.0]], values=[0.2], model_prior_odds=odds
        )


@pytest.mark.parametrize("schema", ["future", True, None])
def test_typed_schema_copy_cannot_bypass_public_readers(comparison, schema):
    bad = comparison.model_copy(update={"schema_version": schema})
    with pytest.raises(ERRORS):
        bad.summary()
    with pytest.raises(ERRORS):
        bad.model_dump()


def test_complete_query_binding_and_exact_local_seed_replay(comparison):
    query = pd.DataFrame(
        {"x": pd.array([1, None, 3, 3], dtype="Int64")},
        index=pd.Index(["a", "gap", "z", "z"], name="id"),
    )
    first = bayes_hypothesis_draws(result=comparison, draws=17, seed=19, data=query, missing="drop")
    second = bayes_hypothesis_draws(
        result=comparison, draws=17, seed=19, data=query, missing="drop"
    )
    assert first.payload == second.payload
    assert first.payload["query"]["sample_positions"] == (0, 2, 3)
    assert first.payload["query"]["dtypes"] == ("Int64",)
    assert bayes_hypothesis_draws_restore(saved=first.model_dump_json()).payload == first.payload


@pytest.mark.parametrize("constructor", ["copy", "construct"])
@pytest.mark.parametrize("reader", ["restore", "summary", "dump", "json"])
def test_draw_typed_bypass_refuses_rehashed_seed_cache_changes(comparison, constructor, reader):
    draws = bayes_hypothesis_draws(result=comparison, draws=7, seed=3)
    payload = copy.deepcopy(draws.model_dump()["payload"])
    payload["arrays"]["beta"][0][0] += 0.1
    rehash(payload)
    bad = (
        draws.model_copy(update={"payload": payload})
        if constructor == "copy"
        else PosteriorHypothesisDraws.model_construct(payload=payload)
    )
    actions = {
        "restore": lambda: bayes_hypothesis_draws_restore(saved=bad),
        "summary": bad.summary,
        "dump": bad.model_dump,
        "json": bad.model_dump_json,
    }
    with pytest.raises(ERRORS):
        actions[reader]()


def test_tiny_original_unit_covariance_tampering_cannot_hide_behind_absolute_tolerance():
    data = pd.DataFrame({"y": [1e-12, 2e-12, 3e-12], "x": [0.0, 1.0, 2.0]})
    alternative = bayes_linear(
        data=data,
        y="y",
        x=["x"],
        prior=dict(mean=[0.0, 0.0], scale_matrix=[[1.0, 0.0], [0.0, 1.0]], shape=2.0, scale=1e-24),
    )
    comparison = bayes_hypothesis(result=alternative, constraints=[[0.0, 1.0]], values=[0.0])
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    assert payload["results"]["null_posterior"]["coefficient_covariance"][0][0] < 1e-24
    payload["results"]["null_posterior"]["coefficient_covariance"][0][0] *= 4
    rehash(payload)
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


@pytest.mark.parametrize("fixed", ["coordinate", "all"])
@pytest.mark.parametrize(
    "field",
    [
        "prior_conditional_scale_matrix",
        "conditional_scale_matrix",
        "coefficient_scale_matrix",
        "coefficient_covariance",
    ],
)
@pytest.mark.parametrize("reader", ["restore", "tables", "dump", "json"])
def test_exact_fixed_support_rejects_rehashed_subnormal_uncertainty(
    alternative, fixed, field, reader
):
    constraints = [[1.0, 0.0]] if fixed == "coordinate" else [[1.0, 0.0], [0.0, 1.0]]
    comparison = bayes_hypothesis(
        result=alternative, constraints=constraints, values=[0.0] * len(constraints)
    )
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    assert payload["results"]["null_posterior"][field][0][0] == 0.0
    payload["results"]["null_posterior"][field][0][0] = 1e-320
    bad = comparison.model_copy(update={"payload": rehash(payload)})
    actions = {
        "restore": lambda: bayes_hypothesis_restore(saved=bad),
        "tables": bad.to_tables,
        "dump": bad.model_dump,
        "json": bad.model_dump_json,
    }
    with pytest.raises(ERRORS):
        actions[reader]()


def test_joint_query_and_work_limits_refuse_before_query_serialization(comparison, monkeypatch):
    query = pd.DataFrame({"x": np.ones(513)})
    import openecon.econometrics.bayesian.hypothesis as module

    monkeypatch.setattr(
        module, "source_values", lambda *a: pytest.fail("oversized query reached source copy")
    )
    with pytest.raises(ERRORS):
        bayes_hypothesis_predict(result=comparison, data=query)
    with pytest.raises(ERRORS):
        bayes_hypothesis_draws(result=comparison, draws=10001)


def test_saved_budget_refusal_precedes_tensor_allocation(comparison, monkeypatch):
    payload = copy.deepcopy(comparison.model_dump()["payload"])
    payload["alternative"]["max_bytes"] = 100
    import torch

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: pytest.fail("saved budget refusal reached tensor allocation"),
    )
    with pytest.raises(ERRORS):
        bayes_hypothesis_restore(saved=payload)


def test_saved_encoded_json_limit_precedes_decode():
    with pytest.raises(ERRORS):
        PosteriorHypothesisComparison.model_validate_json(" " * (32 * 1024 * 1024 + 1))
    with pytest.raises(ERRORS):
        PosteriorHypothesisDraws.model_validate_json(" " * (32 * 1024 * 1024 + 1))


@pytest.mark.parametrize("cls", [PosteriorHypothesisComparison, PosteriorHypothesisDraws])
@pytest.mark.parametrize("encoding", ["str", "bytes", "bytearray"])
def test_dense_typed_json_workspace_gate_precedes_pydantic_decoder(cls, encoding, monkeypatch):
    dense = '{"payload":{"dense":[' + ",".join(["{}"] * 50000) + "]}}"
    encoded = dense if encoding == "str" else dense.encode()
    if encoding == "bytearray":
        encoded = bytearray(encoded)

    class NoParser:
        def validate_json(self, *args, **kwargs):
            pytest.fail("unadmitted dense JSON reached Pydantic decoder")

    monkeypatch.setattr(cls, "__pydantic_validator__", NoParser())
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        cls.model_validate_json(encoded)


@pytest.mark.parametrize("restore", [bayes_hypothesis_restore, bayes_hypothesis_draws_restore])
def test_dense_public_json_workspace_gate_precedes_standard_decoder(restore, monkeypatch):
    encoded = '{"dense":[' + ",".join(["{}"] * 50000) + "]}"
    monkeypatch.setattr(
        json, "loads", lambda *a, **kw: pytest.fail("unadmitted JSON reached decoder")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        restore(saved=encoded)


@pytest.mark.parametrize("cls", [PosteriorHypothesisComparison, PosteriorHypothesisDraws])
@pytest.mark.parametrize("reader", ["model_copy", "deepcopy"])
def test_constructed_dense_typed_state_refuses_before_deep_copy(cls, reader, monkeypatch):
    model = cls.model_construct(payload={"dense": [{} for _ in range(50000)]})
    monkeypatch.setattr(
        BaseModel,
        "__deepcopy__",
        lambda *a, **kw: pytest.fail("unadmitted constructed state reached deep copy"),
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        model.model_copy(deep=True) if reader == "model_copy" else copy.deepcopy(model)


def test_complete_valid_models_support_semantically_checked_deep_copy(comparison):
    draws = bayes_hypothesis_draws(result=comparison, draws=2)
    for model in (comparison, draws):
        assert model.model_copy(deep=True).payload == model.payload
        assert copy.deepcopy(model).payload == model.payload


@pytest.mark.parametrize("kind", ["comparison", "draws"])
def test_large_indent_is_admitted_before_native_json_serializer(comparison, kind, monkeypatch):
    model = (
        comparison if kind == "comparison" else bayes_hypothesis_draws(result=comparison, draws=2)
    )

    class NoSerializer:
        def to_json(self, *args, **kwargs):
            pytest.fail("oversized indented output reached native serializer")

    monkeypatch.setattr(type(model), "__pydantic_serializer__", NoSerializer())
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        model.model_dump_json(indent=100000)


@pytest.mark.parametrize("indent", [True, -1, 2**1024])
def test_indentation_primitive_and_encoded_envelope_refusal(comparison, indent):
    with pytest.raises(AnalysisError):
        comparison.model_dump_json(indent=indent)


def test_serialization_admission_counts_both_container_indentation_lines():
    from openecon.econometrics.bayesian.hypothesis import _serialization_admission

    value = 1
    for _ in range(60):
        value = [value]
    model = SimpleNamespace(payload=value)
    raw = model.__dict__
    # Determine actual indentation geometry with a small reference output;
    # scaling padding predicts the oversized complete output without creating it.
    padding_per_unit = len(json.dumps(raw, indent=1)) - len(json.dumps(raw, indent=0))
    assert padding_per_unit * 10000 > 32 * 1024**2
    with pytest.raises(AnalysisError, match="32 MiB"):
        _serialization_admission(model, 10000)


def test_legitimate_indented_json_and_excluded_payload_validate_full_semantics(comparison):
    for model in (comparison, bayes_hypothesis_draws(result=comparison, draws=2)):
        encoded = model.model_dump_json(indent=2)
        assert type(model).model_validate_json(encoded).payload == model.payload
        assert model.model_dump(exclude={"payload"})["schema_version"] == model.schema_version
        bad = model.model_copy(update={"schema_version": "foreign"})
        with pytest.raises(AnalysisError):
            bad.model_dump(exclude={"payload"})
        with pytest.raises(AnalysisError):
            bad.model_dump_json(exclude={"payload"})


def test_resource_receipt_does_not_bind_replay_to_old_ambient_budget(comparison):
    # The original receipt remains immutable provenance; today's tighter live
    # budget is enforced independently before its source is replayed.
    with use_workspace_budget(4):
        assert bayes_hypothesis_restore(saved=comparison).payload == comparison.payload


def test_portable_external_payload_can_be_loaded_without_pydantic_class_name(comparison):
    serialized = json.dumps(comparison.model_dump()["payload"])
    assert bayes_hypothesis_restore(saved=serialized).payload == comparison.payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("values", [[True]]),
        ("values", [[2**1024]]),
        ("values", [[]]),
        ("dtypes", [True]),
        ("dtypes", ["object"]),
        ("columns", None),
        ("sample_positions", [True]),
        ("missing", []),
        ("rows_original", True),
    ],
)
def test_saved_query_primitive_shape_and_dtype_refusal_precedes_tensor(
    comparison, monkeypatch, field, value
):
    draws = bayes_hypothesis_draws(result=comparison, draws=2, data=pd.DataFrame({"x": [1.0]}))
    payload = copy.deepcopy(draws.model_dump()["payload"])
    payload["query"][field] = value
    import torch

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: pytest.fail("saved query intake reached tensor allocation"),
    )
    with pytest.raises(ERRORS):
        bayes_hypothesis_draws_restore(saved=payload)


def test_new_class_validation_always_revalidates_constructed_foreign_schema(comparison):
    bad = comparison.model_copy(update={"schema_version": "foreign"})
    with pytest.raises(ERRORS):
        PosteriorHypothesisComparison.model_validate(bad)


def test_tiny_shape_predictive_moment_overflow_refuses_finite_json_output():
    alternative = bayes_linear(
        data=pd.DataFrame({"y": [0.0], "x": [0.1]}),
        y="y",
        x=["x"],
        prior=dict(mean=[0.0, 0.0], scale_matrix=[[1.0, 0.0], [0.0, 1.0]], shape=1e-308, scale=1.0),
    )
    comparison = bayes_hypothesis(result=alternative, constraints=[[1.0, 0.0]], values=[0.0])
    with pytest.raises(ERRORS, match="representable"):
        bayes_hypothesis_predict(result=comparison, data=pd.DataFrame({"x": [100.0]}))


def test_unknown_control_types_are_typed_refusals_before_numerical_replay(comparison, monkeypatch):
    import torch

    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *a, **kw: pytest.fail("invalid public controls reached tensor allocation"),
    )
    for target in (["null"], None, True):
        with pytest.raises(ERRORS):
            bayes_hypothesis_predict(
                result=comparison, target=target, data=pd.DataFrame({"x": [1.0]})
            )
    for missing in ([], None, True):
        with pytest.raises(ERRORS):
            bayes_hypothesis_draws(result=comparison, draws=2, missing=missing)
    for count in (True, 0, 10001):
        with pytest.raises(ERRORS):
            bayes_hypothesis_draws(result=comparison, draws=count)
