"""Pre-allocation parser/copy/export guards across the four new foundations."""

import copy
import datetime
import inspect
import json
import tracemalloc

import pandas as pd
import numpy as np
import pytest
import torch
from pydantic import BaseModel

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.bayesian.commands import bayes_contrast, bayes_draws, bayes_linear
from openecon.econometrics.bayesian.core import load_mapping
from openecon.econometrics.bayesian.posterior import (
    NormalInverseGammaPrior,
    PosteriorBundle,
    PosteriorContrast,
    PosteriorDraws,
)
from openecon.econometrics.ivquantile.api import _extract
from openecon.econometrics.ivquantile import ivqreg, ivqreg_restore
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.latent.cfa import CFAState, cfa_covariance, cfa_restore
from openecon.econometrics.state_lifecycle import _json_export_admission, _metadata_geometry
from openecon.econometrics.summary_state import restore_summary, saved_summary, summary_state
from openecon.econometrics.tsworkflows.ssdiffuse import DiffuseResult, _load_record, sspace_diffuse
from openecon.resources import use_workspace_budget

TYPED = [NormalInverseGammaPrior, PosteriorBundle, PosteriorContrast, PosteriorDraws, CFAState]


def dense_json(n=50000):
    return '{"payload":{"dense":[' + ",".join(["{}"] * n) + "]}}"


@pytest.fixture(scope="module")
def models():
    posterior = bayes_linear(
        data=pd.DataFrame({"y": [1.0, 3.0, 2.0, 5.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0, 4.0]}),
        y="y",
        x=["x"],
        prior={
            "mean": [0.2, -0.1],
            "scale_matrix": [[1.2, 0.1], [0.1, 0.8]],
            "shape": 1.2,
            "scale": 0.8,
        },
    )
    fitted_cfa = cfa_covariance(
        [[1.7, 0.8, 0.6], [0.8, 1.5, 0.6], [0.6, 0.6, 1.1]],
        factors={"factor": ["a", "b", "c"]},
        n=150,
        divisor="n",
    )
    return {
        NormalInverseGammaPrior: posterior.prior,
        PosteriorBundle: posterior,
        PosteriorContrast: bayes_contrast(result=posterior, weights=[1.0, 0.0]),
        PosteriorDraws: bayes_draws(result=posterior, draws=3, seed=4),
        CFAState: CFAState(payload=fitted_cfa.attrs["cfa_state"]),
    }


@pytest.mark.parametrize("cls", TYPED)
@pytest.mark.parametrize("encoding", ["str", "bytes", "bytearray"])
def test_dense_typed_json_refuses_before_pydantic_decoder(cls, encoding, monkeypatch):
    value = dense_json()
    if encoding != "str":
        value = value.encode()
    if encoding == "bytearray":
        value = bytearray(value)

    class NoParser:
        def validate_json(self, *args, **kwargs):
            pytest.fail("unadmitted dense state reached Pydantic's parser")

    monkeypatch.setattr(cls, "__pydantic_validator__", NoParser())
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        cls.model_validate_json(value)


@pytest.mark.parametrize("cls", TYPED)
@pytest.mark.parametrize("method", ["model_copy", "deepcopy"])
def test_constructed_dense_state_refuses_before_inherited_deep_copy(cls, method, monkeypatch):
    field = (
        "payload"
        if cls is CFAState
        else "beta"
        if cls is PosteriorDraws
        else "weights"
        if cls is PosteriorContrast
        else "mean"
    )
    model = cls.model_construct(**{field: [{} for _ in range(50000)]})
    monkeypatch.setattr(
        BaseModel,
        "__deepcopy__",
        lambda *a, **kw: pytest.fail("unadmitted state reached deep copy"),
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        model.model_copy(deep=True) if method == "model_copy" else copy.deepcopy(model)


@pytest.mark.parametrize("cls", TYPED)
def test_large_indent_refuses_before_native_serializer(models, cls, monkeypatch):
    class NoSerializer:
        def to_json(self, *args, **kwargs):
            pytest.fail("unadmitted indentation reached Pydantic's serializer")

    monkeypatch.setattr(cls, "__pydantic_serializer__", NoSerializer())
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        models[cls].model_dump_json(indent=100000)


@pytest.mark.parametrize("cls", TYPED)
def test_valid_json_deep_copy_and_complete_inherited_introspection(models, cls):
    model = models[cls]
    encoded = model.model_dump_json(indent=2)
    restored = cls.model_validate_json(encoded, strict=False, context={"source": "lifecycle-test"})
    assert restored.model_dump() == model.model_dump()
    assert model.model_copy(deep=True).model_dump() == model.model_dump()
    assert copy.deepcopy(model).model_dump() == model.model_dump()
    assert inspect.signature(cls.model_dump_json) == inspect.signature(BaseModel.model_dump_json)
    assert inspect.getdoc(cls.model_dump_json) == inspect.getdoc(BaseModel.model_dump_json)
    # Shallow copies remain the documented Pydantic unvalidated operation;
    # downstream public readers enforce the complete numerical contract.
    assert model.model_copy(update={"schema_version": "foreign"}).schema_version == "foreign"


@pytest.mark.parametrize("cls", TYPED)
def test_strict_json_keyword_behavior_matches_existing_pydantic_parser(models, cls):
    encoded = models[cls].model_dump_json()
    kwargs = {"strict": True, "context": {"preserve": "existing semantics"}}
    try:
        baseline = BaseModel.model_validate_json.__func__(cls, encoded, **kwargs)
    except ValueError as expected:
        with pytest.raises(type(expected)):
            cls.model_validate_json(encoded, **kwargs)
    else:
        assert cls.model_validate_json(encoded, **kwargs).model_dump() == baseline.model_dump()


@pytest.mark.parametrize(
    "parser", [load_mapping, cfa_restore, _extract, _load_record, restore_summary]
)
def test_public_string_reader_refuses_before_standard_json_parser(parser, monkeypatch):
    value = dense_json()
    monkeypatch.setattr(
        json, "loads", lambda *a, **kw: pytest.fail("unadmitted state reached json.loads")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        parser(value)


@pytest.mark.parametrize("cls", TYPED)
def test_actual_parser_guard_refuses_at_bounded_allocation_peak(cls):
    value = dense_json()
    tracemalloc.start()
    try:
        with use_workspace_budget(1), pytest.raises(AnalysisError):
            cls.model_validate_json(value)
        assert tracemalloc.get_traced_memory()[1] < 256 * 1024
    finally:
        tracemalloc.stop()


def test_diffuse_export_and_deep_copy_validate_before_large_allocation():
    system = {
        "a0": [0.3],
        "P_inf": [[1.0]],
        "P_star": [[0.0]],
        "Z": [[1.0]],
        "T": [[1.0]],
        "Q": [[0.25]],
        "H": [[0.6]],
        "c": [0.1],
        "d": [-0.2],
    }
    fitted = sspace_diffuse(pd.DataFrame({"y": [1.2, 1.8, 0.7, 2.1, 1.5]}), "y", system=system)
    assert copy.deepcopy(fitted).to_json() == fitted.to_json()
    bad = DiffuseResult({}, diffuse_result={"dense": [{} for _ in range(50000)]})
    for reader in (bad.to_json, lambda: copy.deepcopy(bad)):
        with use_workspace_budget(1), pytest.raises(AnalysisError):
            reader()


def test_nested_json_padding_counts_closing_lines_before_output_allocation():
    value = 1
    for _ in range(60):
        value = [value]
    size, depth_sum = _metadata_geometry(value)
    actual = json.dumps(value, indent=2)
    assert len(actual.encode()) <= size + 2 * 2 * depth_sum
    # Small-reference formatting establishes opening+closing line geometry;
    # multiplying padding predicts >32MiB without allocating that output.
    actual_padding_per_unit = len(json.dumps(value, indent=1)) - len(json.dumps(value, indent=0))
    assert actual_padding_per_unit * 10000 > 32 * 1024**2
    with pytest.raises(AnalysisError):
        _json_export_admission(value, indent=10000, limit=32 * 1024**2, operation="nested export")


def test_escaped_unicode_serialization_geometry_covers_surrogate_pairs():
    value = {"\U0001f30d" * 100: "\U0001f320" * 100}
    size, _ = _metadata_geometry(value)
    assert len(json.dumps(value, ensure_ascii=True).encode()) <= size


@pytest.mark.parametrize("exporter", [summary_state, saved_summary])
def test_ivqr_summary_export_admits_metadata_before_json_conversion(exporter, monkeypatch):
    output = TableSet(
        {}, state={"schema": "openecon.ivquantile.v1", "dense": [{} for _ in range(50000)]}
    )
    # Restore the process-wide serializer before pytest's report hooks run.
    # The forbidden production conversion remains patched throughout the call.
    with monkeypatch.context() as scoped:
        scoped.setattr(
            json, "dumps", lambda *a, **kw: pytest.fail("Unadmitted metadata was serialized")
        )
        with use_workspace_budget(1), pytest.raises(AnalysisError):
            exporter(output)


@pytest.mark.parametrize("exporter", [summary_state, saved_summary])
def test_ivqr_summary_table_dimensions_admitted_before_table_scan(exporter, monkeypatch):
    frame = table(np.zeros((10000, 2)), columns=["a", "b"])
    output = TableSet({"dense": frame}, state={"schema": "openecon.ivquantile.v1"})
    monkeypatch.setattr(
        type(frame),
        "memory_usage",
        lambda *a, **kw: pytest.fail("Unadmitted dimensions reached a scan"),
    )
    monkeypatch.setattr(
        type(frame), "to_numpy", lambda *a, **kw: pytest.fail("Unadmitted table was copied")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        exporter(output)


def test_ivqr_long_table_strings_admitted_before_portable_array_copy(monkeypatch):
    frame = table([["x" * 1000000]], columns=["value"])
    output = TableSet({"large": frame}, state={"schema": "openecon.ivquantile.v1"})
    monkeypatch.setattr(
        type(frame), "to_numpy", lambda *a, **kw: pytest.fail("Unadmitted string table was copied")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        summary_state(output)


def test_ivqr_long_axis_names_admitted_before_portable_array_copy(monkeypatch):
    frame = table([[1.0]], columns=["value"])
    frame.index.name = "x" * 1000000
    output = TableSet({"large": frame}, state={"schema": "openecon.ivquantile.v1"})
    monkeypatch.setattr(
        type(frame), "to_numpy", lambda *a, **kw: pytest.fail("Unadmitted axis name was copied")
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        summary_state(output)


def test_valid_complete_ivqr_summary_roundtrip_without_optimizer(monkeypatch):
    rng = np.random.default_rng(680)
    x, z, u, e = rng.normal(size=(4, 61))
    d = 1.5 * z + 0.6 * x + u
    data = pd.DataFrame({"y": 0.7 + 1.1 * d - 0.4 * x + 0.5 * u + e, "d": d, "x": x, "z": z})
    output = ivqreg(
        data=data, y="y", endogenous="d", instruments=["z"], x=["x"], grid=[-0.5, 1.0, 2.5]
    )
    assert output.attrs["state"]["schema"] == "openecon.ivquantile.v1"
    encoded = summary_state(output)
    restored = restore_summary(encoded)
    assert summary_state(restored) == encoded
    from openecon.econometrics.quantile import kernels

    monkeypatch.setattr(kernels, "solve", lambda *a, **kw: pytest.fail("Restoration refitted QR"))
    replayed = ivqreg_restore(encoded)
    assert replayed.attrs["state"] == output.attrs["state"]
    for name in output:
        pd.testing.assert_frame_equal(restored[name], output[name])
        pd.testing.assert_frame_equal(replayed[name], output[name])


def test_legacy_summary_tensor_numpy_tuple_date_metadata_keeps_permissive_conversion():
    output = TableSet({"values": table([[1.0]], columns=["value"])}, title="legacy")
    output.attrs.update(
        tensor=torch.tensor([1.0, 2.0], dtype=torch.float64),
        numpy=np.float64(3.0),
        tuple=(4, "five"),
        date=datetime.date(2026, 10, 9),
    )
    restored = restore_summary(summary_state(output))
    assert restored.attrs == {
        "tensor": [1.0, 2.0],
        "numpy": 3.0,
        "tuple": [4, "five"],
        "date": "2026-10-09",
    }
    pd.testing.assert_frame_equal(restored["values"], output["values"])


def test_summary_preparser_and_ivqr_export_actual_allocation_peaks_are_bounded():
    output = TableSet(
        {}, state={"schema": "openecon.ivquantile.v1", "dense": [{} for _ in range(50000)]}
    )
    encoded = json.dumps(
        {"schema": "openecon.summary.v1", "title": None, "attrs": output.attrs, "tables": {}}
    )
    for read in (lambda: summary_state(output), lambda: restore_summary(encoded)):
        tracemalloc.start()
        try:
            with use_workspace_budget(1), pytest.raises(AnalysisError):
                read()
            assert tracemalloc.get_traced_memory()[1] < 256 * 1024
        finally:
            tracemalloc.stop()
