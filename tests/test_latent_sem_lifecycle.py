"""Allocation-before-parser/copy/export gates for the complete SEM typed state."""

import copy
import inspect
import json
import tracemalloc

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.latent import sem
from openecon.resources import use_workspace_budget


def dense_json():
    return '{"payload":{"dense":[' + ",".join(["{}"] * 50000) + "]}}"


@pytest.fixture(scope="module")
def model():
    fitted = sem.latent_sem_covariance(
        [[1.0, 0.0], [0.0, 1.0]], columns=["x", "y"], n=303,
        divisor="n", meanstructure=False,
        residual_covariances={("x", "y"): 0.1}, tolerance=1e-8,
    )
    return sem.SEMState(payload=fitted.attrs["sem_state"])


@pytest.mark.parametrize("encoding", [str, bytes, bytearray])
def test_dense_json_refuses_before_pydantic_parser(encoding, monkeypatch):
    value = dense_json()
    if encoding is not str:
        value = encoding(value.encode())

    class NoParser:
        def validate_json(self, *args, **kwargs):
            pytest.fail("unadmitted JSON reached Pydantic")

    monkeypatch.setattr(sem.SEMState, "__pydantic_validator__", NoParser())
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        sem.SEMState.model_validate_json(value)


def test_public_restore_refuses_before_json_loads(monkeypatch):
    monkeypatch.setattr(sem.json, "loads", lambda *a, **kw: pytest.fail("parser reached"))
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        sem.latent_sem_restore(dense_json())


@pytest.mark.parametrize("operation", ["model_copy", "deepcopy", "update"])
def test_dense_copy_and_update_admitted_before_inherited_copy(operation, monkeypatch):
    dense = {"dense": [{} for _ in range(50000)]}
    value = sem.SEMState.model_construct(payload=dense)
    monkeypatch.setattr(BaseModel, "__deepcopy__", lambda *a, **kw: pytest.fail("copy reached"))
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        if operation == "model_copy":
            value.model_copy(deep=True)
        elif operation == "deepcopy":
            copy.deepcopy(value)
        else:
            sem.SEMState.model_construct(payload={}).model_copy(update={"payload": dense}, deep=True)


def test_dense_constructor_metadata_refuses_before_replay(monkeypatch):
    monkeypatch.setattr(sem, "_replay", lambda *a: pytest.fail("unadmitted metadata reached replay"))
    with use_workspace_budget(1), pytest.raises((AnalysisError, ValidationError)):
        sem.SEMState(payload={"dense": [{} for _ in range(50000)]})


def test_huge_indent_refuses_before_serializer(model, monkeypatch):
    class NoSerializer:
        def to_json(self, *args, **kwargs):
            pytest.fail("unadmitted formatted output reached serializer")

    monkeypatch.setattr(sem.SEMState, "__pydantic_serializer__", NoSerializer())
    with pytest.raises(AnalysisError):
        model.model_dump_json(indent=2**30)


@pytest.mark.parametrize("method", ["model_dump", "model_dump_json"])
@pytest.mark.parametrize("selection", [{"exclude": {"payload"}}, {"include": {"schema_version"}}])
@pytest.mark.parametrize("forgery", ["schema", "information"])
def test_partial_exports_still_validate_complete_semantic_payload(model, method, selection, forgery):
    if forgery == "schema":
        bad = model.model_copy(update={"schema_version": "foreign"})
    else:
        payload = copy.deepcopy(model.model_dump()["payload"])
        payload["results"]["observed_information_solver"][0][0] *= 4
        payload["digest"] = sem._digest({k: v for k, v in payload.items() if k != "digest"})
        bad = model.model_copy(update={"payload": payload})
    with pytest.raises(AnalysisError):
        getattr(bad, method)(**selection)


@pytest.mark.parametrize("reader", ["tables", "restore", "deepcopy"])
def test_shallow_unchecked_schema_update_rejected_by_public_readers(model, reader):
    bad = model.model_copy(update={"schema_version": "foreign"})
    assert bad.schema_version == "foreign"
    with pytest.raises(AnalysisError):
        if reader == "tables":
            bad.to_tables()
        elif reader == "restore":
            sem.latent_sem_restore(bad)
        else:
            copy.deepcopy(bad)


def test_valid_roundtrips_copies_partial_export_and_inherited_signature(model):
    saved = model.model_dump()
    for candidate in (copy.deepcopy(model), model.model_copy(deep=True),
                      sem.SEMState.model_validate_json(model.model_dump_json()),
                      sem.SEMState.model_validate_json(model.model_dump_json(), strict=False,
                                                      context={"origin": "lifecycle"})):
        assert candidate.model_dump() == saved
        pd.testing.assert_frame_equal(candidate.to_tables()["covariance"], model.to_tables()["covariance"])
    assert model.model_dump(exclude={"payload"}) == {"schema_version": sem.SCHEMA}
    assert json.loads(model.model_dump_json(include={"schema_version"})) == {"schema_version": sem.SCHEMA}
    for method in ("model_copy", "model_dump", "model_dump_json"):
        assert inspect.signature(getattr(sem.SEMState, method)) == inspect.signature(getattr(BaseModel, method))
        assert getattr(sem.SEMState, method).__doc__ == getattr(BaseModel, method).__doc__


def test_measured_dense_parser_and_copy_refuse_with_bounded_peak():
    value = dense_json()
    model = sem.SEMState.model_construct(payload={"dense": [{} for _ in range(50000)]})
    for call in (lambda: sem.SEMState.model_validate_json(value),
                 lambda: model.model_copy(deep=True), lambda: copy.deepcopy(model)):
        with use_workspace_budget(1):
            tracemalloc.start()
            try:
                with pytest.raises(AnalysisError):
                    call()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        assert peak < 100_000
