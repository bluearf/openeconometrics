"""Resident inputs use bounded projections, with independent numeric contracts."""
from bisect import bisect_right
import importlib
import math
import sys

import numpy as np
import pandas as pd
import pytest

import openecon_charts as charts
from openecon.dataset import Dataset


@pytest.mark.parametrize("helper,options", [(charts.hist, {"x": "x"}),
    (charts.scatter, {"x": "x", "y": "y"})])
def test_large_numeric_dataframe_never_materializes_python_columns(helper, options, monkeypatch):
    frame = pd.DataFrame({"x": np.arange(1_000_003, dtype=float), "y": np.ones(1_000_003)})
    implementation = importlib.import_module("openecon_charts.charts")
    monkeypatch.setattr(implementation, "_columns", lambda *a: pytest.fail("full Python columns allocated"))
    result = helper(data=frame, **options)
    assert result.total_n == len(frame)
    assert len(result.data) <= 2000


@pytest.mark.parametrize("container", ["dataframe", "mapping"])
def test_resident_line_refuses_before_copying_any_values(container, monkeypatch):
    values = {"x": range(10001), "y": range(10001)}
    data = pd.DataFrame(values) if container == "dataframe" else values
    implementation = importlib.import_module("openecon_charts.charts")
    monkeypatch.setattr(implementation, "_columns", lambda *a: pytest.fail("copied before line guard"))
    with pytest.raises(ValueError, match="10,000"):
        charts.line(data=data, x="x", y="y")


class GuardedArray(np.ndarray):
    def __new__(cls, values):
        result = np.asarray(values).view(cls)
        result.reads = []
        return result

    def __array_finalize__(self, other):
        self.reads = getattr(other, "reads", [])

    def __getitem__(self, selected):
        if isinstance(selected, slice):
            size = len(range(*selected.indices(len(self))))
            assert size <= 65536, "Copied a full resident column"
            self.reads.append(size)
        return super().__getitem__(selected)


@pytest.mark.parametrize("helper,options", [(charts.hist, {"x": "x"}),
    (charts.scatter, {"x": "x", "y": "y"})])
def test_typed_mapping_projects_only_bounded_native_arrays(helper, options, monkeypatch):
    values = {"x": GuardedArray(np.arange(131079, dtype=float)),
              "y": GuardedArray(np.ones(131079)), "unused": object()}
    implementation = importlib.import_module("openecon_charts.charts")
    monkeypatch.setattr(implementation, "_columns", lambda *a: pytest.fail("numeric arrays became Python lists"))
    result = helper(data=values, **options)
    assert result.total_n == 131079
    assert len(values["x"].reads) == 6 and max(values["x"].reads) == 65536
    assert result.config["processing"]["passes"] == 2


def test_dataframe_rows_are_sliced_before_column_projection(monkeypatch):
    original = pd.DataFrame.take
    def bounded_take(self, indices, axis=0, **options):
        if axis == 1:
            assert len(self) <= 65536, "Projection copied a complete input column"
        return original(self, indices, axis=axis, **options)
    monkeypatch.setattr(pd.DataFrame, "take", bounded_take)
    frame = pd.DataFrame({"unused": np.zeros(131079), "x": np.arange(131079, dtype=float)})
    assert charts.hist(data=frame, x="x").total_n == len(frame)


@pytest.mark.parametrize("dtype", ["float64", "float32", "Float64", "Int64", "UInt64", "boolean", "string", "object"])
def test_dataframe_mapping_and_dataset_match_independent_missing_and_numeric_contracts(dtype):
    values = np.arange(5007) % 103
    if dtype == "boolean":
        values = values % 2 == 0
    series = pd.Series(values, dtype=dtype)
    series.iloc[::17] = None
    frame = pd.DataFrame({"x": series, "y": np.sin(np.arange(len(series)))})
    # Different indices must not align column mappings into extra/missing rows.
    mapping = {"x": frame.x.set_axis(np.arange(len(frame)) + 10000),
               "y": frame.y.set_axis(np.arange(len(frame)) + 20000)}
    actual = []
    for data in (frame, mapping, Dataset.from_frame(frame)):
        scatter = charts.scatter(data=data, x="x", y="y")
        histogram = charts.hist(data=data, x="x", bins=17)
        finite = [(float(x), float(y)) for x, y in zip(frame.x, frame.y)
                  if x is not None and x is not pd.NA and math.isfinite(float(x)) and math.isfinite(float(y))]
        ranks = [round(i * (len(finite) - 1) / 1999) for i in range(2000)]
        assert scatter.data == [{"x": finite[rank][0], "y": finite[rank][1]} for rank in ranks]
        assert (scatter.sample_n, scatter.total_n, scatter.dropped_n) == (2000, len(finite), len(frame) - len(finite))
        edges = [row["x0"] for row in histogram.data] + [histogram.data[-1]["x1"]]
        counts = [0] * 17
        for x, _ in finite:
            counts[min(16, bisect_right(edges, x) - 1)] += 1
        assert [row["count"] for row in histogram.data] == counts
        assert (histogram.total_n, histogram.dropped_n) == (len(finite), len(frame) - len(finite))
        actual.append((scatter.model_dump(), histogram.model_dump()))
    assert actual[0] == actual[1] == actual[2]


@pytest.mark.parametrize("values,pattern", [([-1e308, 1e308], "numeric"),
    ([1e16, 1e16 + 2.], "collapse"), ([None, float("inf"), float("nan")], "at least one")])
def test_dataframe_and_dataset_extreme_histogram_errors_match(values, pattern):
    frame = pd.DataFrame({"x": values})
    for data in (frame, {"x": frame.x.to_numpy()}, Dataset.from_frame(frame)):
        with pytest.raises(ValueError, match=pattern):
            charts.hist(data=data, x="x", bins=20)


@pytest.mark.parametrize("data", [{"x": [2**53 + 1, None], "y": [None, 1]},
    {"x": np.array([2**64 - 1], dtype="uint64"), "y": np.array([math.nan])}])
def test_mapping_never_rounds_unsafe_integer_before_missing_pair_validation(data):
    for helper, options in ((charts.hist, {"x": "x"}), (charts.scatter, {"x": "x", "y": "y"})):
        with pytest.raises(ValueError, match="exact numeric range"):
            helper(data=data, **options)


def test_unsampled_extremes_and_log_validation_use_every_resident_row():
    frame = pd.DataFrame({"x": np.arange(1, 5006, dtype=float), "y": np.ones(5005)})
    frame.loc[1, "x"], frame.loc[2, "x"] = -100., 1e6
    result = charts.scatter(data=frame, x="x", y="y")
    assert min(row["x"] for row in result.data) > -100
    assert max(row["x"] for row in result.data) < 1e6
    assert result.config["processing"]["extents"]["x"] == [-100., 1e6]
    with pytest.raises(ValueError, match="strictly positive"):
        charts.scatter(data=frame, x="x", y="y", x_scale="log")


def test_missing_record_keys_can_first_appear_after_the_initial_block():
    records = [{"y": 1.}] * 65536 + [{"x": 2., "y": 3.}, {"x": 4., "y": 5.}]
    result = charts.scatter(data=records, x="x", y="y")
    assert result.data == [{"x": 2., "y": 3.}, {"x": 4., "y": 5.}]
    assert (result.total_n, result.dropped_n) == (2, 65536)


def test_framework_independent_sequences_and_one_shot_compatibility(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pandas", None)
    values = {"x": [0., 1., None, 3.], "y": [2., 3., 4., 5.]}
    result = charts.scatter(data=values, x="x", y="y")
    assert result.data == [{"x": 0., "y": 2.}, {"x": 1., "y": 3.}, {"x": 3., "y": 5.}]
    one_shot = charts.hist(data={"x": iter(values["x"])}, x="x", bins=2)
    assert [row["count"] for row in one_shot.data] == [2, 1]
    assert charts.line(data=values, x="x", y="y").dropped_n == 1


def test_unknown_length_line_stops_after_limit_plus_one_values():
    read = []
    def endless():
        value = 0
        while True:
            read.append(value)
            yield value
            value += 1
    with pytest.raises(ValueError, match="10,000"):
        charts.line(data={"x": endless(), "y": range(10001)}, x="x", y="y")
    assert len(read) == 10001


def test_resident_unsampled_mutation_is_rejected(monkeypatch):
    from openecon_charts import resident
    frame = pd.DataFrame({"x": np.arange(5005, dtype=float), "y": np.ones(5005)})
    original, calls = resident._Frame.iter_batches, []
    def changed(self, **options):
        calls.append(True)
        if len(calls) == 2:
            frame.loc[1, "x"] += .25
        yield from original(self, **options)
    monkeypatch.setattr(resident._Frame, "iter_batches", changed)
    with pytest.raises(ValueError, match="changed between passes"):
        charts.scatter(data=frame, x="x", y="y")


def test_line_keeps_irregular_spacing_gaps_and_input_identity():
    frame = pd.DataFrame({"x": [10., 1., 5., None, 25., 20.], "y": [3., 1., None, 4., 8., 6.]},
                         index=[5, 5, 2, 20, 10, 1])
    before = frame.copy(deep=True)
    result = charts.line(data=frame, x="x", y="y")
    assert result.data == [{"x": 1., "y": 1.}, {"x": 5., "y": None}, {"x": 10., "y": 3.},
                           {"x": None, "y": None}, {"x": 20., "y": 6.}, {"x": 25., "y": 8.}]
    other = charts.line(data=Dataset.from_frame(frame), x="x", y="y")
    assert result.data == other.data
    assert (result.total_n, result.dropped_n) == (other.total_n, other.dropped_n)
    pd.testing.assert_frame_equal(frame, before)
