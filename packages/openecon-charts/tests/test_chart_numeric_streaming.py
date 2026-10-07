"""Native numeric blocks preserve full validation, bin counts and finite ranks."""
from bisect import bisect_right
from decimal import Decimal
import importlib
import math
import sys

import pytest

import openecon_charts as charts

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


class Source:
    def __init__(self, frame, chunk=337, mutate=False, reverse=False):
        self.frame, self.columns = frame, list(frame)
        self.chunk, self.mutate, self.reverse = chunk, mutate, reverse
        self.calls, self.maximum_rows = 0, 0

    def iter_batches(self, *, columns, batch_rows):
        self.calls += 1
        size = min(self.chunk, batch_rows)
        for start in range(0, len(self.frame), size):
            value = self.frame.iloc[start:start + size][columns]
            if self.reverse:
                value = value.iloc[::-1]
            if self.mutate and self.calls > 1 and start == 0:
                value = value.copy()
                value.iloc[1, 0] += 0.25
            self.maximum_rows = max(self.maximum_rows, len(value))
            yield value


def no_python_numeric_loop(monkeypatch):
    implementation = importlib.import_module("openecon_charts.charts")
    monkeypatch.setattr(implementation, "_columns", lambda *args: pytest.fail("numeric batches must not materialize Python lists"))
    monkeypatch.setattr(implementation, "_number", lambda *args: pytest.fail("numeric batches must not validate each row in Python"))


def independent_bins(values, points):
    edges = [row["x0"] for row in points] + [points[-1]["x1"]]
    counts = [0] * len(points)
    for value in values:
        if math.isfinite(value):
            counts[min(len(points) - 1, bisect_right(edges, float(value)) - 1)] += 1
    return counts


@pytest.mark.parametrize("dtype", ["float64", "Float64", "float32", "Int64", "UInt64", "boolean"])
def test_histogram_native_numeric_blocks_match_all_row_bins_without_python_lists(dtype, monkeypatch):
    pytest.importorskip("torch")
    values = np.arange(23009, dtype=float) % 103
    if dtype == "boolean":
        values = values % 2 == 0
    series = pd.Series(values, dtype=dtype)
    if dtype in {"Float64", "Int64", "UInt64", "boolean"}:
        series.iloc[102:111] = pd.NA
    elif dtype in {"float64", "float32"}:
        series.iloc[102:111] = np.nan
        series.iloc[773] = np.inf
    frame = pd.DataFrame({"x": series, "unused": [object()] * len(series)})
    source = Source(frame)
    expected_values = series.to_numpy(dtype="float64", na_value=math.nan)
    no_python_numeric_loop(monkeypatch)
    result = charts.hist(data=source, x="x", bins=17)
    assert [row["count"] for row in result.data] == independent_bins(expected_values, result.data)
    assert result.total_n == int(np.isfinite(expected_values).sum())
    assert result.dropped_n == int((~np.isfinite(expected_values)).sum())
    assert source.calls == 2 and source.maximum_rows <= 337
    assert result.config["processing"]["projected_columns"] == ["x"]


def test_scatter_native_finite_rank_sampling_and_extents_independent_numpy(monkeypatch):
    pytest.importorskip("torch")
    n = 22007
    x = np.arange(n, dtype=float) / 13
    y = np.sin(x)
    x[::17], y[::31] = np.nan, np.inf
    valid = np.isfinite(x) & np.isfinite(y)
    finite_x, finite_y = x[valid], y[valid]
    ranks = [(2 * i * (len(finite_x) - 1) + 1999) // 3998 for i in range(2000)]
    expected = [{"x": float(finite_x[rank]), "y": float(finite_y[rank])} for rank in ranks]
    source = Source(pd.DataFrame({"x": pd.Series(x, dtype="Float64"), "y": y}))
    no_python_numeric_loop(monkeypatch)
    result = charts.scatter(data=source, x="x", y="y")
    assert result.data == expected
    assert (result.sample_n, result.total_n, result.dropped_n) == (2000, len(finite_x), n - len(finite_x))
    assert result.config["processing"]["extents"] == {
        "x": [float(finite_x.min()), float(finite_x.max())],
        "y": [float(finite_y.min()), float(finite_y.max())],
    }
    assert source.calls == 2


@pytest.mark.parametrize("dtype,value", [("int64", 2**53), ("UInt64", 2**64 - 1), ("Int64", -(2**53))])
@pytest.mark.parametrize("helper", ["hist", "scatter"])
def test_unsafe_integer_is_checked_before_float_conversion_even_when_pair_is_missing(dtype, value, helper):
    frame = pd.DataFrame({"x": pd.Series([1, value, 3], dtype=dtype), "y": [1., math.nan, 3.]})
    with pytest.raises(ValueError, match="outside the browser's exact numeric range"):
        if helper == "hist":
            charts.hist(data=Source(frame), x="x")
        else:
            charts.scatter(data=Source(frame), x="x", y="y")


@pytest.mark.parametrize("helper", ["hist", "scatter"])
def test_optional_torch_absence_preserves_framework_independent_numeric_pandas_fallback(helper, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    frame = pd.DataFrame({"x": pd.Series([0., 1., None, 3.], dtype="Float64"), "y": [2., 3., 4., 5.]})
    if helper == "hist":
        result = charts.hist(data=Source(frame), x="x", bins=2)
        assert [row["count"] for row in result.data] == [2, 1]
        assert result.config["processing"]["kernel"] == "stdlib bisect"
    else:
        result = charts.scatter(data=Source(frame), x="x", y="y")
        assert result.data == [{"x": 0., "y": 2.}, {"x": 1., "y": 3.}, {"x": 3., "y": 5.}]
    assert (result.total_n, result.dropped_n) == (3, 1)


@pytest.mark.parametrize("values,pattern", [(["1", "9007199254740992"], "exact numeric"),
    ([Decimal("1"), Decimal("9007199254740992")], "exact numeric"),
    (pd.date_range("2026-01-01", periods=2), "numeric or missing")])
def test_nonnumeric_dtype_fallback_keeps_string_decimal_date_and_complex_validation(values, pattern):
    with pytest.raises(ValueError, match=pattern):
        charts.hist(data=Source(pd.DataFrame({"x": values})), x="x")


def test_complex_dtype_keeps_existing_pandas_scalar_conversion_fallback():
    frame = pd.DataFrame({"x": [1 + 2j, 3 + 1j]})
    # pandas returns NumPy complex scalars; their existing float conversion
    # warns and keeps the real component. This optimization leaves it intact.
    with pytest.warns(Warning, match="Casting complex"):
        expected = charts.hist(data=frame, x="x", bins=2)
    with pytest.warns(Warning, match="Casting complex"):
        actual = charts.hist(data=Source(frame), x="x", bins=2)
    assert actual.data == expected.data


def test_numeric_source_integrity_detects_changed_unsampled_value():
    frame = pd.DataFrame({"x": np.arange(5005, dtype=float), "y": np.ones(5005)})
    with pytest.raises(ValueError, match="changed between passes"):
        charts.scatter(data=Source(frame, mutate=True), x="x", y="y")


def test_native_cpu_dtype_rng_scope_and_negative_strides_match_dense_order():
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame({"x": np.arange(11, dtype=float), "y": np.arange(11, dtype=float) ** 2})
    source = Source(frame, chunk=5, reverse=True)
    expected_rows = pd.concat([frame.iloc[j:j + 5].iloc[::-1] for j in range(0, len(frame), 5)])
    expected = [{"x": float(x), "y": float(y)} for x, y in zip(expected_rows.x, expected_rows.y)]
    rng = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    with torch.device("meta"):
        result = charts.scatter(data=source, x="x", y="y")
        histogram = charts.hist(data=Source(frame), x="x", bins=3)
        assert torch.empty(0).device.type == "meta"
    assert result.data == expected
    assert sum(row["count"] for row in histogram.data) == len(frame)
    assert torch.get_default_dtype() == dtype
    torch.testing.assert_close(torch.random.get_rng_state(), rng)


def test_actual_file_backed_parquet_uses_same_numeric_bins_and_complete_hash(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")
    from openecon.dataset import scan

    frame = pd.DataFrame({"x": np.arange(70009, dtype=float) % 113})
    frame.loc[::503, "x"] = math.nan
    path = tmp_path / "numbers.parquet"
    frame.to_parquet(path, index=False)
    result = charts.hist(data=scan(path), x="x", bins=13)
    assert [row["count"] for row in result.data] == independent_bins(frame.x, result.data)
    assert result.config["processing"]["source_rows"] == len(frame)
    assert result.config["processing"]["passes"] == 2
