"""Independent bounded source reductions, gap contracts and replay integrity."""
from bisect import bisect_right
import math
import sys

import pytest

import openecon_charts as charts


class Source:
    def __init__(self, values, chunk=113, mutate=False):
        self.values, self.columns = values, list(values)
        self.chunk, self.mutate, self.calls = chunk, mutate, 0
        self.projections = []

    def iter_batches(self, *, columns, batch_rows):
        assert batch_rows <= 65_536
        self.calls += 1
        self.projections.append(columns)
        for start in range(0, len(self.values[self.columns[0]]), self.chunk):
            block = {name: list(self.values[name][start:start + self.chunk]) for name in columns}
            if self.mutate and self.calls > 1 and start == 0:
                for name in columns:
                    if block[name] and isinstance(block[name][0], (int, float)):
                        block[name][0] += .125
                        break
            yield block


def test_scatter_uniform_finite_ranks_matches_dense_and_projects_only_requested_columns():
    values = {"x": list(range(5005)) + [None, 8],
              "y": [math.sin(i) for i in range(5005)] + [4, None], "unused": [object()] * 5007}
    source = Source(values)
    result = charts.scatter(data=source, x="x", y="y")
    expected = charts.scatter(data=values, x="x", y="y")
    assert result.data == expected.data
    assert (result.sample_n, result.total_n, result.dropped_n) == (2000, 5005, 2)
    assert source.calls == 2 and source.projections == [["x", "y"]] * 2
    assert len(result.model_dump_json()) < 180_000
    assert result.config["processing"]["sample_limit"] == 2000


@pytest.mark.parametrize("values,bins", [([-2, -1, 0, 1, 2, None, float("inf")], 4),
                                       ([3.] * 100, 7), ([1., 2., 2., 4.], 1),
                                       ([1e12, 1e12 + .01], 3)])
@pytest.mark.parametrize("stdlib", [False, True])
def test_exact_histogram_counts_match_independent_bin_edges(values, bins, stdlib, monkeypatch):
    if stdlib:
        monkeypatch.setitem(sys.modules, "torch", None)
    result = charts.hist(data=Source({"x": values}, chunk=2), x="x", bins=bins)
    edges = [item["x0"] for item in result.data] + [result.data[-1]["x1"]]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    expected = [0] * bins
    for value in finite:
        expected[min(bins - 1, bisect_right(edges, value) - 1)] += 1
    assert [item["count"] for item in result.data] == expected
    assert result.total_n == len(finite) and result.dropped_n == len(values) - len(finite)
    assert result.config["processing"]["passes"] == 2


def test_line_preserves_missing_runs_across_batches_and_retains_explicit_bound():
    values = {"x": [10, 1, 5, None, 25, 20], "y": [3, 1, None, 4, 8, 6]}
    result = charts.line(data=Source(values, chunk=2), x="x", y="y")
    assert result.data == charts.line(data=values, x="x", y="y").data
    with pytest.raises(ValueError, match="10,000"):
        charts.line(data=Source({"x": range(10001), "y": range(10001)}), x="x", y="y")


@pytest.mark.parametrize("helper", [charts.bar, charts.barh, charts.area, charts.stacked_bar])
@pytest.mark.parametrize("aggregate", ["sum", "mean", "count"])
def test_categorical_aggregation_is_explicit_and_full_source_counts_are_retained(helper, aggregate):
    values = {"group": ["B", "A", "B", "C", "A"], "y": [1., 2., 3., None, 4.]}
    if helper is charts.stacked_bar:
        values["y"][3] = 0.
    source = Source(values, chunk=2)
    if helper is charts.stacked_bar:
        result = helper(data=source, x="group", y="y", aggregate=aggregate)
    else:
        result = helper(data=source, x="group", y="y", aggregate=aggregate)
    expected = {"sum": [4., 6., None], "mean": [2., 3., None], "count": [2., 2., None]}[aggregate]
    if helper is charts.stacked_bar:
        expected[2] = 1 if aggregate == "count" else 0.
    assert result.config["categories"] == ["B", "A", "C"]
    assert result.config["series"][0]["values"] == expected
    assert result.total_n == sum(value is not None for value in values["y"])
    assert result.dropped_n == sum(value is None for value in values["y"])
    assert source.calls == 2
    assert result.config["processing"]["aggregation"] == aggregate


def test_dense_explicit_aggregation_uses_same_bounded_reduction_and_donut_contract():
    values = {"group": ["A", "B", "A"], "value": [1., 2., 3.]}
    plot = charts.donut(data=Source(values), labels="group", values="value", aggregate="sum")
    assert plot.config["categories"] == ["A", "B"]
    assert plot.config["series"][0]["values"] == [4., 2.]
    assert charts.bar(data=values, x="group", y="value", aggregate="sum").config["series"][0]["values"] == [4., 2.]
    with pytest.raises(ValueError, match="Aggregate"):
        charts.bar(data=Source(values), x="group", y="value")
    with pytest.raises(ValueError, match="1,000"):
        charts.bar(data=Source({"group": range(1001), "value": range(1001)}), x="group", y="value", aggregate="sum")
    with pytest.raises(ValueError, match="collide"):
        charts.bar(data=Source({"group": [1, "1"], "value": [2., 3.]}), x="group", y="value", aggregate="sum")


@pytest.mark.parametrize("helper,kwargs", [(charts.scatter, {"x": "x", "y": "y"}),
    (charts.hist, {"x": "x"}), (charts.line, {"x": "x", "y": "y"}),
    (charts.bar, {"x": "group", "y": "y", "aggregate": "sum"})])
def test_mutated_source_is_rejected_even_when_counts_are_unchanged(helper, kwargs):
    source = Source({"x": [1., 2., 3.], "y": [4., 5., 6.], "group": ["A", "B", "A"]}, mutate=True)
    with pytest.raises(ValueError, match="changed between passes"):
        helper(data=source, **kwargs)


def test_coefficient_dataset_is_bounded_and_retains_confidence_limits():
    source = Source({"term": ["x", "y"], "estimate": [1., 2.],
                     "ci_low": [.5, 1.5], "ci_high": [1.5, 2.5]})
    result = charts.coefficients(source)
    assert result.data == [{"term": "x", "estimate": 1., "ci_low": .5, "ci_high": 1.5},
                           {"term": "y", "estimate": 2., "ci_low": 1.5, "ci_high": 2.5}]
    assert source.calls == 2


def test_actual_dataset_protocol_and_cpu_scope_restore_under_meta():
    pd = pytest.importorskip("pandas")
    torch = pytest.importorskip("torch")
    from openecon.dataset import Dataset
    frame = pd.DataFrame({"x": [0., 1., 2., None]})
    with torch.device("meta"):
        plot = charts.hist(data=Dataset.from_frame(frame), x="x", bins=2)
        assert torch.empty(0).device.type == "meta"
    assert [item["count"] for item in plot.data] == [1, 2]
    assert (plot.total_n, plot.dropped_n) == (3, 1)


def test_log_scale_refuses_nonpositive_source_value_even_if_unsampled():
    values = {"x": [float(i + 1) for i in range(5005)], "y": [1.] * 5005}
    values["x"][1] = -1.
    with pytest.raises(ValueError, match="strictly positive"):
        charts.scatter(data=Source(values), x="x", y="y", x_scale="log")


def test_dataframe_style_batch_protocol_does_not_require_pandas_or_torch(monkeypatch):
    class Batch:
        columns = ["x"]
        dtypes = ["float"]
        def __getitem__(self, names):
            assert names == ["x"]
            return self
        def to_dict(self, orient):
            assert orient == "list"
            return {"x": [0., 1., 2.]}
    class IndependentSource:
        columns = ["x"]
        def iter_batches(self, *, columns, batch_rows):
            yield Batch()
    monkeypatch.setitem(sys.modules, "pandas", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    plot = charts.hist(data=IndependentSource(), x="x", bins=2)
    assert [row["count"] for row in plot.data] == [1, 2]
    assert plot.config["processing"]["kernel"] == "stdlib bisect"


def test_wide_chart_projection_adapts_reader_rows_instead_of_multiplying_batch_memory():
    from openecon_charts.streaming import Replay
    names = [f"series_{index}" for index in range(1000)]
    source = Source({name: [float(index), float(index + 1)] for index, name in enumerate(names)})
    replay = Replay(source, names)
    assert replay.batch_rows == (8 * 1024 * 1024) // (64 * len(names))
    assert replay.batch_rows < 1000
    assert sum(len(batch[names[0]]) for batch in replay.batches()) == 2
    replay.verify()
    assert replay.metadata()["reader_batch_rows"] == replay.batch_rows


def test_oversized_projected_dataframe_is_rejected_before_python_list_conversion(monkeypatch):
    import importlib
    import pandas as pd
    from openecon_charts.streaming import Replay

    class Oversized:
        columns = ["x"]

        def iter_batches(self, *, columns, batch_rows):
            yield pd.DataFrame({"x": ["x" * (17 * 1024 * 1024)]})

    implementation = importlib.import_module("openecon_charts.charts")
    monkeypatch.setattr(implementation, "_columns", lambda *args: pytest.fail("oversized raw batch must fail before list conversion"))
    with pytest.raises(ValueError, match="memory budget"):
        list(Replay(Oversized(), ["x"]).batches())
