"""Chart semantics and safe standalone output, independent of OpenEcon models."""
from dataclasses import dataclass
from decimal import Decimal
import html
import json
from pathlib import Path
import re

import pytest

import openecon_charts as charts


def test_mapping_and_record_adapters_have_identical_values():
    mapping = {"x": [1, 3, 10], "y": [4, 5, 6]}
    records = [dict(zip(mapping, values)) for values in zip(*mapping.values())]
    assert charts.scatter(data=mapping, x="x", y="y") == charts.scatter(data=records, x="x", y="y")
    with pytest.raises(ValueError, match="equal lengths"):
        charts.scatter(data={"x": [1, 2], "y": [3]}, x="x", y="y")
    with pytest.raises(ValueError, match="not found"):
        charts.scatter(data=mapping, x="absent", y="y")


def test_dataframe_protocol_projects_before_materializing():
    class Table:
        columns = ["x", "y", "unused"]
        selected = None

        def __getitem__(self, names):
            self.selected = names
            return self

        def to_dict(self, orient):
            assert orient == "list"
            assert self.selected == ["x", "y"]
            return {"x": [1, 2], "y": [3, 4]}

    assert charts.scatter(data=Table(), x="x", y="y").total_n == 2


def test_optional_pandas_adapter_and_missing_nullable_values():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"x": pd.Series([1, 2, 3], dtype="Int64"),
                          "y": pd.Series([4, None, 7], dtype="Int64")})
    result = charts.line(data=frame, x="x", y="y")
    assert result.data == [{"x": 1., "y": 4.}, {"x": 2., "y": None}, {"x": 3., "y": 7.}]
    assert result.dropped_n == 1


def test_dataframe_duplicate_requested_columns_raise_without_choosing_a_value():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame([[1, 100, 5], [2, 200, 6]], columns=["x", "x", "y"])
    with pytest.raises(ValueError, match="Column 'x' is duplicated"):
        charts.scatter(data=frame, x="x", y="y")
    unambiguous = pd.DataFrame([[1, 5, 9, 10], [2, 6, 11, 12]],
                             columns=["x", "y", "unused", "unused"])
    assert charts.scatter(data=unambiguous, x="x", y="y").data == [
        {"x": 1., "y": 5.}, {"x": 2., "y": 6.}]


def test_scatter_sampling_is_declared_and_keeps_endpoints():
    plot = charts.scatter(data={"x": list(range(5000)) + [None], "y": list(range(5000)) + [2]}, x="x", y="y")
    assert (plot.sample_n, plot.total_n, plot.dropped_n) == (2000, 5000, 1)
    assert plot.data[0] == {"x": 0., "y": 0.}
    assert plot.data[-1] == {"x": 4999., "y": 4999.}


def test_numeric_line_retains_irregular_spacing_and_both_kinds_of_gap():
    plot = charts.line(data={"x": [10, 1, 5, None, 25, 20], "y": [3, 1, None, 4, 8, 6]}, x="x", y="y")
    assert plot.data == [{"x": 1., "y": 1.}, {"x": 5., "y": None}, {"x": 10., "y": 3.},
                         {"x": None, "y": None}, {"x": 20., "y": 6.}, {"x": 25., "y": 8.}]
    assert (plot.total_n, plot.sample_n, plot.dropped_n) == (4, 4, 2)
    with pytest.raises(ValueError, match="10,000"):
        charts.line(data={"x": range(10001), "y": range(10001)}, x="x", y="y")


def test_histogram_edges_counts_missing_and_constants():
    plot = charts.hist(data={"x": [-2, -1, 0, 1, 2, None, float("inf")]}, x="x", bins=4)
    assert [item["count"] for item in plot.data] == [1, 1, 1, 2]
    assert plot.data[0]["x0"] == -2
    assert plot.data[-1]["x1"] == 2
    assert plot.total_n == 5 and plot.dropped_n == 2
    constant = charts.hist(data={"x": [3, 3, 3]}, x="x", bins=2)
    assert sum(item["count"] for item in constant.data) == 3
    assert constant.data[0]["x0"] < 3 < constant.data[-1]["x1"]
    with pytest.raises(ValueError, match="bins"):
        charts.hist(data={"x": [1]}, x="x", bins=True)


def test_histogram_rejects_collapsed_float_edges_before_serialization():
    values = [1e16, 1e16 + 2.]
    with pytest.raises(ValueError, match="bin edges collapse"):
        charts.hist(data={"x": values}, x="x", bins=20)
    one_bin = charts.hist(data={"x": values}, x="x", bins=1)
    assert one_bin.data == [{"x0": values[0], "x1": values[1], "count": 2}]
    centered = charts.hist(data={"x": [value - values[0] for value in values]}, x="x", bins=20)
    assert all(row["x0"] < row["x1"] for row in centered.data)
    assert sum(row["count"] for row in centered.data) == 2


def test_histogram_counts_agree_with_reported_float_edges():
    initial = charts.hist(data={"x": [1e12, 1e12 + .01]}, x="x", bins=3)
    edges = [row["x0"] for row in initial.data] + [initial.data[-1]["x1"]]
    reconstructed = charts.hist(data={"x": edges}, x="x", bins=3)
    assert [row["count"] for row in reconstructed.data] == [1, 1, 2]
    assert [row["x0"] for row in reconstructed.data] == edges[:-1]
    assert reconstructed.data[-1]["x1"] == edges[-1]


def test_signed_bar_area_and_categorical_years_are_not_reindexed():
    data = {"year": [2021, 2024], "a": [-3, None], "b": [2, 4]}
    for helper in [charts.bar, charts.barh, charts.area]:
        plot = helper(data=data, x="year", y=["a", "b"], palette=["#abc", "#123456"])
        assert plot.config["categories"] == ["2021", "2024"]
        assert plot.config["series"][0]["values"] == [-3., None]
        assert plot.config["series"][0]["color"] == "#aabbcc"
        assert plot.config["compositional"] is False
        assert plot.dropped_n == 1


@pytest.mark.parametrize("value", [-1, None, float("inf")])
def test_compositions_reject_negative_or_missing_components(value):
    data = {"label": ["A", "B"], "value": [2, value]}
    with pytest.raises(ValueError, match="nonnegative"):
        charts.stacked_bar(data=data, x="label", y="value")
    with pytest.raises(ValueError, match="nonnegative"):
        charts.donut(data=data, labels="label", values="value")


def test_compositions_positive_total_and_horizontal_contract():
    data = {"label": ["A", "B"], "value": [2, 3]}
    plot = charts.donut(data=data, labels="label", values="value", palette=["#abc", "#def"])
    assert plot.config["type"] == "donut" and plot.config["compositional"] is True
    assert plot.config["palette"] == ["#aabbcc", "#ddeeff"]
    assert charts.stacked_bar(data=data, x="label", y="value", horizontal=True).config["type"] == "stacked-horizontal"
    with pytest.raises(ValueError, match="positive total"):
        charts.donut(data={"label": ["A"], "value": [0]}, labels="label", values="value")


def test_duplicates_are_not_silently_aggregated():
    with pytest.raises(ValueError, match="Aggregate"):
        charts.bar(data={"x": ["A", "A"], "y": [1, 2]}, x="x", y="y")
    with pytest.raises(ValueError, match="Aggregate"):
        charts.bar(data={"x": [1, "1"], "y": [1, 2]}, x="x", y="y")


@pytest.mark.parametrize("palette", [["red"], ["#ggg"], ["#1234"], ["#12345678"], [], "#abcdef"])
def test_invalid_palette_is_an_error(palette):
    with pytest.raises(ValueError, match="palette"):
        charts.bar(data={"x": ["A"], "y": [1]}, x="x", y="y", palette=palette)


def test_categorical_bounds_are_explicit():
    with pytest.raises(ValueError, match="1,000"):
        charts.bar(data={"x": list(range(1001)), "y": list(range(1001))}, x="x", y="y")
    columns = {f"v{i}": [1] for i in range(25)}
    with pytest.raises(ValueError, match="24 series"):
        charts.area(data={"x": ["A"], **columns}, x="x", y=list(columns))


@pytest.mark.parametrize("large", [2**53, 2**53 + 1, str(2**53 + 1), Decimal(2**53 + 1)])
def test_unsafe_integers_never_silently_collapse(large):
    with pytest.raises(ValueError, match="exact numeric range"):
        charts.scatter(data={"x": [large], "y": [1]}, x="x", y="y")
    # Explicit floating values follow the caller's floating-point contract.
    assert charts.scatter(data={"x": [float(2**53)], "y": [1]}, x="x", y="y").data[0]["x"] == float(2**53)


def test_coefficients_use_duck_protocol_and_finite_ordered_intervals():
    @dataclass
    class Term:
        term: str = "education"
        estimate: float = 1.2
        ci_low: float = .8
        ci_high: float = 1.6

    @dataclass
    class Model:
        coefficients: tuple = (Term(),)

    plot = charts.coefficients(Model())
    assert plot.data[0]["estimate"] == 1.2
    with pytest.raises(ValueError, match="finite"):
        charts.coefficients({"coefficients": [{"term": "x", "estimate": float("nan"), "ci_low": 0, "ci_high": 1}]})
    with pytest.raises(ValueError, match="lower limits"):
        charts.coefficients({"coefficients": [{"term": "x", "estimate": 1, "ci_low": 3, "ci_high": 2}]})


def test_json_spec_rejects_nonfinite_and_mutated_payloads():
    with pytest.raises(ValueError, match="finite JSON"):
        charts.PlotSpec("scatter", "title", "x", "y", [{"x": float("inf"), "y": 1}], 1, 1)
    plot = charts.scatter(data={"x": [1], "y": [2]}, x="x", y="y")
    plot.data[0]["x"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        plot.model_dump()


def test_offline_html_escapes_labels_and_allows_isolated_exports(tmp_path):
    attack = '</script><script>alert("owned")</script>&\u2028\u2029'
    plot = charts.bar(data={"label": [attack], "value": [2]}, x="label", y="value", title=attack)
    document = plot.to_html(height=640)
    assert f"<title>{html.escape(attack)}</title>" in document
    assert "<h1>" not in document
    payload = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', document, re.S).group(1)
    assert "<" not in payload and ">" not in payload and "&" not in payload
    assert json.loads(payload)["config"]["categories"] == [attack]
    assert "\\u2028" in payload and "\\u2029" in payload
    assert "connect-src &#x27;none&#x27;" in document
    assert "img-src data: blob:" in document
    assert "data:font/woff2;base64," in document
    assert 'min-height:640px' in document
    assert not re.search(r"<script[^>]+src=", document)
    assert not re.search(r"url\(['\"]?Barlow\.woff2", document)
    iframe = plot._repr_html_()
    assert 'sandbox="allow-scripts allow-downloads"' in iframe
    assert "allow-same-origin" not in iframe
    path = plot.save_html(tmp_path / "chart.html")
    assert isinstance(path, Path) and path.read_text().startswith("<!doctype html>")


@pytest.mark.parametrize("helper", [charts.scatter, charts.line])
def test_finite_but_overflowing_numeric_axis_is_rejected(helper):
    with pytest.raises(ValueError, match="axis range"):
        helper(data={"x": [-1e308, 1e308], "y": [1., 2.]}, x="x", y="y")


def test_overflowing_composition_and_signed_span_are_rejected():
    with pytest.raises(ValueError, match="axis range"):
        charts.area(data={"label": ["A", "B"], "value": [-1e308, 1e308]}, x="label", y="value")
    with pytest.raises(ValueError, match="totals"):
        charts.donut(data={"label": ["A", "B"], "value": [1e308, 1e308]}, labels="label", values="value")
    with pytest.raises(ValueError, match="totals"):
        charts.stacked_bar(data={"label": ["A"], "a": [1e308], "b": [1e308]}, x="label", y=["a", "b"])



def test_raw_plot_spec_also_rejects_unsafe_integer_payloads():
    with pytest.raises(ValueError, match="exact numeric range"):
        charts.PlotSpec("scatter", "title", "x", "y", [{"x": 2**53 + 1, "y": 1}], 1, 1)
