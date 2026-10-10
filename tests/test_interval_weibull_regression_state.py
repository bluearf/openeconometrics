"""Semantic endpoint replay and strict pre-parser/source/query/lifecycle admission."""

import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survival_ext import weibull_regression as api
from openecon.econometrics.survival_ext import weibull_regression_common as c
from openecon.econometrics.survival_ext import weibull_regression_kernels as k
from openecon.resources import use_workspace_budget

spec = importlib.util.spec_from_file_location(
    "weibull_example",
    Path(__file__).resolve().parents[1] / "examples/interval_weibull_regression.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture(scope="module")
def fit():
    return api.stinterval_weibull_regression(
        example.case(223, n=90), lower="lower", upper="upper", x=["x1", "x2"]
    )


@pytest.fixture(scope="module")
def query(fit):
    return api.interval_weibull_regression_predict(
        fit, pd.DataFrame({"x1": [-0.5, 0.75], "x2": [0.25, -0.4]}), times=[0.0, 0.5, 1.5, 3.0]
    )


def payload(value):
    return value.model_dump()["payload"]


def rehash(value):
    value["sha256"] = c.digest({key: cell for key, cell in value.items() if key != "sha256"})
    return value


def forbidden(*args, **kwargs):
    raise AssertionError("inadmissible payload reached forbidden allocation/optimizer/parent work")


@pytest.mark.parametrize("kind", ["fit", "query"])
def test_all_readers_optimizer_free(fit, query, kind, monkeypatch):
    model = fit if kind == "fit" else query
    text = model.model_dump_json()
    monkeypatch.setattr(k, "fit", forbidden)
    monkeypatch.setattr(k, "maximize_bfgs", forbidden)
    cls = api.WeibullIntervalFit if kind == "fit" else api.WeibullIntervalPrediction
    restore = (
        api.restore_interval_weibull_regression
        if kind == "fit"
        else api.restore_interval_weibull_prediction
    )
    restored = restore(text)
    assert cls.model_validate_json(text).model_dump() == restored.model_dump()
    assert model.model_copy(deep=True).model_dump() == model.model_dump()
    assert copy.deepcopy(model).model_dump() == model.model_dump()
    assert model.model_dump(exclude={"payload"}) == {}
    assert model.model_dump_json(include=set()) == "{}"
    assert len(model.to_tables()) >= 2
    assert "tabular" in model.latex()
    if kind == "fit":
        assert restored.dataset().equals(example.case(223, n=90))


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(90, name="subject"),
        pd.Index(["same"] * 90, name="name"),
        pd.date_range("2023-01-01", periods=90, tz="Europe/Istanbul", name="date"),
        pd.timedelta_range("0d", periods=90, freq="1h", name="delay"),
        pd.MultiIndex.from_arrays([[1, 2] * 45, ["a", "b", "c"] * 30], names=["group", "id"]),
        pd.CategoricalIndex(
            ["a", "b"] * 45, categories=["unused", "b", "a"], ordered=True, name="category"
        ),
    ],
)
def test_complete_index_roundtrip(index):
    data = example.case(223, n=90)
    data.index = index
    model = api.stinterval_weibull_regression(data, lower="lower", upper="upper", x=["x1", "x2"])
    restored = api.restore_interval_weibull_regression(model.model_dump_json())
    assert restored.dataset().equals(data)
    assert restored.dataset().index.equals(index)


@pytest.mark.parametrize("field", ["information", "covariance"])
def test_rehashed_full_endpoint_cache_forgery_refuses(fit, field):
    forged = payload(fit)
    forged["endpoint"][field][0][0] *= 4
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_regression(rehash(forged))


@pytest.mark.parametrize(
    "field", ["aft_covariance", "ph_covariance", "ph_jacobian", "aft_parameters", "ph_parameters"]
)
def test_rehashed_original_and_ph_cache_forgery_refuses(fit, field):
    forged = payload(fit)
    if "parameters" in field:
        forged["result"][field][0] += 0.1
    else:
        forged["result"][field][0][0] *= 4
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_regression(rehash(forged))


@pytest.mark.parametrize(
    "field", ["covariance", "parameter_query_covariance", "jacobian", "normalized_jacobian"]
)
def test_rehashed_query_full_cross_cache_forgery_refuses(query, field):
    forged = payload(query)
    forged["result"][field][-1][-1] *= 4
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_prediction(rehash(forged))


def test_material_saved_endpoint_shift_fails_score_certificate(fit):
    forged = payload(fit)
    forged["theta"][1] += 0.5
    with pytest.raises(AnalysisError, match="score certificate"):
        api.restore_interval_weibull_regression(rehash(forged))


@pytest.mark.parametrize("kind", ["fit", "query"])
@pytest.mark.parametrize("path", ["dump", "json", "copy", "deepcopy", "tables"])
def test_construct_and_basecopy_forgery_cannot_bypass_excluded_exports(
    fit, query, kind, path, monkeypatch
):
    original = fit if kind == "fit" else query
    forged = payload(original)
    if kind == "fit":
        forged["result"]["aft_covariance"][0][0] *= 4
    else:
        forged["result"]["covariance"][-1][-1] *= 4
    obj = type(original).model_construct(payload=rehash(forged))
    if path in ("dump", "json"):
        monkeypatch.setattr(
            BaseModel, "model_dump" if path == "dump" else "model_dump_json", forbidden
        )
    with pytest.raises(AnalysisError):
        if path == "dump":
            obj.model_dump(exclude={"payload"})
        elif path == "json":
            obj.model_dump_json(include=set())
        elif path == "copy":
            obj.model_copy(deep=True)
        elif path == "deepcopy":
            copy.deepcopy(obj)
        else:
            obj.to_tables()


@pytest.mark.parametrize("kind", ["fit", "query"])
def test_json_and_dense_copy_preadmission(kind, monkeypatch):
    cls = api.WeibullIntervalFit if kind == "fit" else api.WeibullIntervalPrediction
    text = json.dumps({"payload": {"dense": [{} for _ in range(50000)]}})
    with use_workspace_budget(1):
        monkeypatch.setattr(c.json, "loads", forbidden)
        with pytest.raises(AnalysisError):
            cls.model_validate_json(text)
        obj = cls.model_construct(payload={"dense": [{} for _ in range(50000)]})
        with pytest.raises(AnalysisError):
            obj.model_copy(deep=True)
        with pytest.raises(AnalysisError):
            copy.deepcopy(obj)


@pytest.mark.parametrize("indent", [True, -1, 17, 10000, 1.5])
def test_indent_bounded_before_serializer(fit, indent, monkeypatch):
    monkeypatch.setattr(BaseModel, "model_dump_json", forbidden)
    monkeypatch.setattr(k, "evaluate", forbidden)
    with pytest.raises(AnalysisError):
        fit.model_dump_json(indent=indent)


def test_copy_update_header_before_existing_target_information(fit, monkeypatch):
    monkeypatch.setattr(k, "evaluate", forbidden)
    with pytest.raises(AnalysisError):
        fit.model_copy(update={"payload": {"bad": True}}, deep=True)
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit.model_copy(update={"payload": {"dense": [{} for _ in range(50000)]}}, deep=True)


@pytest.mark.parametrize("level", [1.0 - 2.0**-53, 5e-324])
def test_unresolved_confidence_level_refuses_before_target_or_fit(fit, level, monkeypatch):
    monkeypatch.setattr(k, "evaluate", forbidden)
    monkeypatch.setattr(k, "fit", forbidden)
    with pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(
            example.case(2, n=30), lower="lower", upper="upper", x=["x1", "x2"], level=level
        )
    with pytest.raises(AnalysisError):
        api.interval_weibull_regression_predict(
            fit, pd.DataFrame({"x1": [0.1], "x2": [0.2]}), times=[1.0], level=level
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("positions", False),
        ("dtype", "complex128"),
        ("dtype", "object"),
        ("integer", 2**53 + 1),
        ("source_bool", True),
        ("date", "definitely-not-a-date"),
    ],
)
def test_complete_source_refuses_before_tensor(fit, field, value, monkeypatch):
    forged = payload(fit)
    if field == "positions":
        forged["source"]["positions"][0] = value
    elif field == "dtype":
        forged["source"]["dtypes"][2] = value
    elif field == "integer":
        forged["source"]["dtypes"][2] = "int64"
        forged["source"]["values"][0][2] = value
    elif field == "source_bool":
        forged["source"]["values"][0][2] = value
    else:
        data = example.case(223, n=90)
        data.index = pd.date_range("2023-01-01", periods=90)
        forged["source"]["index"] = c.raw_index(data.index)
        forged["source"]["index"]["values"][0]["value"] = value
    rehash(forged)
    monkeypatch.setattr(k, "tensor", forbidden)
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_regression(forged)


@pytest.mark.parametrize(
    "field,value",
    [
        ("time", False),
        ("level", True),
        ("cell", True),
        ("dtype", "complex128"),
        ("dtype", "float32"),
        ("position", False),
    ],
)
def test_saved_query_admission_precedes_fitted_target_work(query, field, value, monkeypatch):
    forged = payload(query)
    if field == "time":
        forged["times"][0] = value
    elif field == "level":
        forged["level"] = value
    elif field == "cell":
        forged["source"]["values"][0][0] = value
    elif field == "position":
        forged["source"]["positions"][0] = value
    else:
        forged["source"]["dtypes"][0] = value
        if value == "float32":
            forged["source"]["values"][0][0] = 0.1
    rehash(forged)
    monkeypatch.setattr(api, "fit_replay", forbidden)
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_prediction(forged)


@pytest.mark.parametrize("cell", [True, 1 + 2j, 2**53 + 1, None, np.nan])
def test_public_bad_query_refuses_before_target_information(fit, cell, monkeypatch):
    frame = pd.DataFrame({"x1": [cell], "x2": [0.2]})
    monkeypatch.setattr(k, "evaluate", forbidden)
    with pytest.raises(AnalysisError):
        api.interval_weibull_regression_predict(fit, frame, times=[1.0])


@pytest.mark.parametrize("cell", [True, 1 + 2j, 2**53 + 1, np.nan])
def test_public_fit_bad_numeric_refuses_before_native(cell, monkeypatch):
    frame = example.case(2, n=30)
    frame["x1"] = pd.Series([cell] * 30, index=frame.index)
    monkeypatch.setattr(k, "prepare", forbidden)
    with pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(frame, lower="lower", upper="upper", x=["x1", "x2"])


def test_mapping_series_workspace_before_dataframe_or_indexcopy(monkeypatch):
    arrays = {
        "lower": pd.Series([1.0] * 1000),
        "upper": pd.Series([2.0] * 1000),
        **{f"x{i}": pd.Series(np.linspace(0.0, 1.0, 1000)) for i in range(8)},
    }
    monkeypatch.setattr(c, "raw_index", forbidden)
    with use_workspace_budget(2), pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(
            arrays, lower="lower", upper="upper", x=[f"x{i}" for i in range(8)]
        )


def test_non_bmp_index_refuses_before_encode_or_checksum(monkeypatch):
    data = example.case(2, n=30)
    data.index = pd.Index(["😀" * 16384] * 30)
    monkeypatch.setattr(c, "encode_index", forbidden)
    with use_workspace_budget(10), pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(data, lower="lower", upper="upper", x=["x1", "x2"])


def test_small_unit_covariance_has_no_absolute_replay_floor():
    model = api.stinterval_weibull_regression(
        example.case(22, n=60, x_units=(1e100, 1e100)), lower="lower", upper="upper", x=["x1", "x2"]
    )
    forged = payload(model)
    assert 0 < abs(forged["result"]["aft_covariance"][1][1]) < 1e-190
    forged["result"]["aft_covariance"][1][1] *= 4
    with pytest.raises(AnalysisError):
        api.restore_interval_weibull_regression(rehash(forged))


@pytest.mark.parametrize("kind", ["fit", "query"])
def test_structural_zero_subnormal_forgery_refuses(fit, query, kind):
    model = fit if kind == "fit" else query
    forged = payload(model)
    if kind == "fit":
        forged["result"]["ph_jacobian"][0][1] = 1e-320
    else:
        forged["result"]["covariance"][0][0] = 1e-320
    restore = (
        api.restore_interval_weibull_regression
        if kind == "fit"
        else api.restore_interval_weibull_prediction
    )
    with pytest.raises(AnalysisError):
        restore(rehash(forged))


def test_rng_and_default_dtype_restored():
    before = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        model = api.stinterval_weibull_regression(
            example.case(2, n=30), lower="lower", upper="upper", x=["x1", "x2"]
        )
        assert (
            api.restore_interval_weibull_regression(model).payload["theta"]
            == model.payload["theta"]
        )
        assert torch.equal(before, torch.random.get_rng_state())
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(dtype)


def test_supported_pydantic_keyword_signatures():
    for method in ("model_dump", "model_dump_json", "model_validate_json", "model_copy"):
        a = inspect.signature(getattr(api.WeibullIntervalFit, method))
        b = inspect.signature(getattr(BaseModel, method))
        assert str(a) == str(b)


@pytest.mark.parametrize(
    "option,value",
    [
        ("device", "cuda"),
        ("weights", [1]),
        ("entry", 0),
        ("cluster", "g"),
        ("frailty", "gamma"),
        ("missing", "drop"),
        ("level", True),
        ("maxiter", True),
        ("max_work", True),
    ],
)
def test_unsupported_controls_refuse_before_source(option, value, monkeypatch):
    monkeypatch.setattr(c, "source", forbidden)
    with pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(
            example.case(2, n=30), lower="lower", upper="upper", x=["x1", "x2"], **{option: value}
        )


def test_work_bound_precedes_source_indexcopy(monkeypatch):
    monkeypatch.setattr(c, "source", forbidden)
    with pytest.raises(AnalysisError):
        api.stinterval_weibull_regression(
            example.case(2, n=30), lower="lower", upper="upper", x=["x1", "x2"], max_work=1
        )
