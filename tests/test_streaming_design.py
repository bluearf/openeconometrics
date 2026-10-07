"""Bounded design preparation checked against the existing dense contract."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import torch

from openecon.analysis import _prepare_data
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec
from openecon.streaming_design import StreamingDesign, row_hash_bytes


def frame():
    return pd.DataFrame({"y": [float(i % 5) for i in range(24)],
                         "x": [float(i) / 7 for i in range(24)],
                         "category": ["ç", "β", "é"] * 8,
                         "group": [i % 4 for i in range(24)]})


def spec(**options):
    return ModelSpec(outcome="y", predictors=["x", "category"],
                     categorical=["category"], **options)


@pytest.mark.parametrize("rows", [1, 2, 7, 100])
@pytest.mark.parametrize("kind", ["unicode", "boolean", "numeric", "declared"])
def test_treatment_design_matches_dense(rows, kind):
    data = frame()
    if kind == "boolean":
        data["category"] = [True, False, True] * 8
    elif kind == "numeric":
        data["category"] = [5, 1, 2] * 8
    elif kind == "declared":
        data["category"] = pd.Categorical(data.category, categories=["β", "ç", "unused", "é"], ordered=True)
    model = spec()
    _, positions, y, x, terms, categories = _prepare_data(model, data)
    design = StreamingDesign(Dataset.from_frame(data), model, rows).prepare()
    actual_x, actual_y = design.encode(design.validate_batch(data))
    torch.testing.assert_close(actual_x, x, rtol=0, atol=0)
    torch.testing.assert_close(actual_y, y, rtol=0, atol=0)
    assert design.terms == terms
    assert design.categorical_encoding == categories
    assert design.sample_rows == positions
    assert design.baseline["original"] == design.baseline["used"] == len(data)
    assert actual_x.dtype == actual_y.dtype == torch.float64
    assert actual_x.device.type == actual_y.device.type == "cpu"


def test_pre_missing_levels_and_cluster_missing_rows_match_dense():
    data = frame()
    data.loc[0, "category"] = "a"
    data.loc[0, "y"] = float("nan")
    data.loc[7, "group"] = float("nan")
    model = spec(covariance="cluster", cluster="group", missing="drop")
    original, positions, y, x, terms, categories = _prepare_data(model, data)
    design = StreamingDesign(Dataset.from_frame(data), model, 3).prepare()
    retained = design.validate_batch(data)
    actual_x, actual_y = design.encode(retained)
    torch.testing.assert_close(actual_x, x, rtol=0, atol=0)
    torch.testing.assert_close(actual_y, y, rtol=0, atol=0)
    assert design.columns == ["y", "x", "category", "group"]
    assert design.terms == terms
    assert design.categorical_encoding == categories
    assert categories["category"]["reference"] == "a"
    assert design.sample_rows == positions
    assert design.baseline["original"] == len(original)
    assert design.baseline["used"] == len(retained) == 22


def test_discovery_digest_matches_fixed_existing_streaming_contract():
    data = frame()
    data.loc[2, "group"] = float("nan")
    model = spec(covariance="cluster", cluster="group", missing="drop")
    a = StreamingDesign(Dataset.from_frame(data), model, 1).prepare()
    b = StreamingDesign(Dataset.from_frame(data), model, 5).prepare()
    assert a.baseline == b.baseline
    digest = hashlib.sha256()
    digest.update(json.dumps({"columns": a.columns, "missing": model.missing},
                             sort_keys=True, separators=(",", ":")).encode())
    digest.update(row_hash_bytes(data.loc[:, a.columns]))
    positions = torch.tensor([i for i in range(24) if i != 2], dtype=torch.int64)
    positions_digest = hashlib.sha256(positions.numpy().astype("<i8", copy=False).tobytes())
    assert a.baseline["data_hash"] == digest.hexdigest()
    assert a.baseline["positions_hash"] == positions_digest.hexdigest()


def test_numeric_prepare_does_not_read_source():
    calls = []
    def factory():
        calls.append(1)
        yield frame()
    source = Dataset.from_batches(factory, list(frame().columns))
    model = ModelSpec(outcome="y", predictors=["x"], covariance="cluster", cluster="group")
    design = StreamingDesign(source, model, 4).prepare()
    assert calls == []
    assert design.baseline is None
    assert design.sample_rows == []
    assert design.columns == ["y", "x", "group"]
    assert design.terms == ["Intercept", "x"]
    assert design.categorical_encoding == {}


@pytest.mark.parametrize("case,code", [
    ("missing", "missing_values"), ("outcome_inf", "non_finite_values"),
    ("predictor_inf", "non_finite_values"), ("category_inf", "unsupported_category"),
    ("constant_outcome", "constant_outcome"), ("constant_predictor", "constant_predictor"),
    ("constant_category", "constant_predictor"), ("mixed_category", "ambiguous_categories"),
    ("non_scalar_category", "unsupported_category"),
])
def test_validation_is_global_across_tiny_chunks(case, code):
    data = frame()
    if case == "missing":
        data.loc[18, "x"] = float("nan")
    elif case.endswith("_inf"):
        column = {"outcome_inf": "y", "predictor_inf": "x", "category_inf": "category"}[case]
        data.loc[18, column] = float("inf")
    elif case == "constant_outcome":
        data["y"] = 1.
    elif case == "constant_predictor":
        data["x"] = 1.
    elif case == "constant_category":
        data["category"] = "a"
    elif case == "mixed_category":
        data.loc[18, "category"] = 1
    elif case == "non_scalar_category":
        data.at[18, "category"] = {"unexpected": 1}
    with pytest.raises(AnalysisError) as caught:
        StreamingDesign(Dataset.from_frame(data), spec(), 2).prepare()
    assert caught.value.code == code


@pytest.mark.parametrize("estimator", ["logit", "probit"])
def test_binary_outcome_validation(estimator):
    data = frame()
    data["y"] = [0, 1] * 12
    design = StreamingDesign(Dataset.from_frame(data), spec(estimator=estimator, covariance="nonrobust"), 1).prepare()
    assert design.baseline["used"] == 24
    data.loc[20, "y"] = 2
    with pytest.raises(AnalysisError) as caught:
        StreamingDesign(Dataset.from_frame(data), spec(estimator=estimator, covariance="nonrobust"), 1).prepare()
    assert caught.value.code == "invalid_binary_outcome"


def test_declared_unused_levels_checked_before_hash_or_dummy_allocation(monkeypatch):
    import openecon.streaming_design as module
    data = frame()
    data["category"] = pd.Categorical(data.category, categories=["ç", "β", "é", *map(str, range(10000))])
    source = Dataset.from_frame(data)
    monkeypatch.setattr(module, "row_hash_bytes", lambda *args: pytest.fail("Large dictionary hashed before width check"))
    monkeypatch.setattr(pd, "get_dummies", lambda *args, **kw: pytest.fail("Dummy allocation preceded width check"))
    with pytest.raises(AnalysisError) as caught:
        StreamingDesign(source, spec(), 1).prepare()
    assert caught.value.code == "model_too_wide"


def test_high_cardinality_stops_and_closes_factory_before_expansion(monkeypatch):
    closed = []
    data = frame()
    def factory():
        try:
            for i in range(1000000):
                part = data.iloc[:1].copy()
                part["category"] = f"level-{i}"
                yield part
        finally:
            closed.append(True)
    source = Dataset.from_batches(factory, list(data.columns))
    monkeypatch.setattr(pd, "get_dummies", lambda *args, **kw: pytest.fail("Premature dummy allocation"))
    with pytest.raises(AnalysisError) as caught:
        StreamingDesign(source, spec(), 1).prepare()
    assert caught.value.code == "model_too_wide"
    assert closed == [True]


def test_repeated_large_category_is_not_double_charged(monkeypatch):
    import openecon.streaming_design as module
    data = frame()
    data["category"] = ["a" * 500, "b" * 500] * 12
    monkeypatch.setattr(module, "MAX_CATEGORY_BYTES", 2600)
    design = StreamingDesign(Dataset.from_frame(data), spec(), 1).prepare()
    assert len(design.categorical_encoding["category"]["levels"]) == 2
    assert design._category_bytes < module.MAX_CATEGORY_BYTES


def test_category_byte_budget_is_explicit_before_encoding(monkeypatch):
    import openecon.streaming_design as module
    monkeypatch.setattr(module, "MAX_CATEGORY_BYTES", 512)
    data = frame()
    data["category"] = ["a" * 1000, "b"] * 12
    with pytest.raises(AnalysisError) as caught:
        StreamingDesign(Dataset.from_frame(data), spec(), 1).prepare()
    assert caught.value.code == "category_metadata_too_large"


def test_cluster_discovery_adapts_keys_budget_without_dropping_rows(monkeypatch):
    import openecon.streaming_design as module
    data = frame()
    data["group"] = ["long-label-a" * 20, "long-label-b" * 20] * 12
    monkeypatch.setattr(module, "MAX_CLUSTER_BYTES", 1024)
    design = StreamingDesign(Dataset.from_frame(data), spec(covariance="cluster", cluster="group"), 24).prepare()
    assert design.baseline["used"] == len(data)


def test_changed_declaration_rejected_even_for_entirely_dropped_batch():
    data = frame()
    data["category"] = pd.Categorical(data.category, categories=["β", "ç", "é"])
    design = StreamingDesign(Dataset.from_frame(data), spec(missing="drop"), 3).prepare()
    changed = data.iloc[:2].copy()
    changed["category"] = changed.category.cat.reorder_categories(["ç", "β", "é"])
    changed["y"] = float("nan")
    with pytest.raises(AnalysisError) as caught:
        design.validate_batch(changed)
    assert caught.value.code == "source_changed"


def test_new_category_after_discovery_is_rejected():
    data = frame()
    design = StreamingDesign(Dataset.from_frame(data), spec(), 3).prepare()
    changed = data.iloc[:3].copy()
    changed.loc[0, "category"] = "new"
    with pytest.raises(AnalysisError) as caught:
        design.encode(changed)
    assert caught.value.code == "source_changed"


def test_arrow_dictionary_uses_dense_sorted_observed_levels():
    import pyarrow as pa
    data = frame()
    raw = pa.DictionaryArray.from_arrays(pa.array([i % 3 for i in range(24)]), pa.array(["β", "unused", "ç", "é"]))
    data["category"] = pd.Series(raw, dtype=pd.ArrowDtype(raw.type))
    model = spec()
    _, _, expected_y, expected_x, terms, encoding = _prepare_data(model, data)
    design = StreamingDesign(Dataset.from_frame(data), model, 2).prepare()
    x, y = design.encode(data)
    torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    assert design.terms == terms
    assert design.categorical_encoding == encoding
    assert encoding["category"]["levels"] == ["unused", "ç", "β"]


def test_discovery_prediction_positions_are_bounded():
    data = pd.concat([frame()] * 40, ignore_index=True)
    design = StreamingDesign(Dataset.from_frame(data), spec(), 17).prepare()
    assert design.baseline["used"] == len(data)
    assert design.sample_rows == list(range(400))
    assert not any(isinstance(value, pd.DataFrame) for value in design.__dict__.values())


def test_unused_cluster_dictionary_is_not_hashed_in_full(monkeypatch):
    data = frame()
    data["group"] = pd.Categorical(data.group, categories=list(range(100000)))
    actual_hash = pd.util.hash_pandas_object
    calls = []
    def bounded_hash(value, **options):
        assert not isinstance(value["group"].dtype, pd.CategoricalDtype)
        calls.append(len(value))
        return actual_hash(value, **options)
    monkeypatch.setattr(pd.util, "hash_pandas_object", bounded_hash)
    digest = row_hash_bytes(data.iloc[:2])
    assert len(digest) == 16
    assert calls == [2]
