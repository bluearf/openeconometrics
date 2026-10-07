"""Presentation changes preserve observations and export the same geometry."""
from copy import deepcopy
from io import StringIO
import json
import math
import re
import sys

import pytest

import openecon_charts as charts


def _plots(**options):
    numeric = {"x": [1, 10, 100], "y": [2, None, 8]}
    categories = {"category": ["A", "B"], "a": [2, 3], "b": [4, 5]}
    return [
        charts.scatter(data=numeric, x="x", y="y", **options),
        charts.line(data=numeric, x="x", y="y", **options),
        charts.hist(data=numeric, x="x", bins=3, **options),
        charts.coefficients({"coefficients": [{"term": "term", "estimate": 1, "ci_low": .5, "ci_high": 1.5}]}, **options),
        charts.bar(data=categories, x="category", y=["a", "b"], **options),
        charts.barh(data=categories, x="category", y=["a", "b"], **options),
        charts.area(data=categories, x="category", y=["a", "b"], **options),
        charts.stacked_bar(data=categories, x="category", y=["a", "b"], **options),
        charts.donut(data=categories, labels="category", values="a", **options),
    ]


def test_all_helpers_share_options_without_changing_any_data_or_metadata():
    for original, styled in zip(_plots(), _plots(width=960, height=640, color="#123", opacity=.6)):
        old, new = original.model_dump(), styled.model_dump()
        assert new["config"]["options"] == {"width": 960, "height": 640, "color": "#112233", "opacity": .6}
        options = new["config"].pop("options")
        assert options
        if old["config"] is None:
            new["config"] = None
        assert new == old
        assert styled.to_latex().startswith("% OpenEcon chart:")


def test_no_options_keep_legacy_wire_format_and_empty_fluent_call_is_independent():
    plot = charts.scatter(data={"x": [1], "y": [2]}, x="x", y="y")
    assert plot.model_dump() == {
        "kind": "scatter", "title": "y · x", "x_label": "x", "y_label": "y",
        "data": [{"x": 1., "y": 2.}], "sample_n": 1, "total_n": 1,
        "dropped_n": 0, "config": None,
    }
    copied = plot.with_options()
    assert copied == plot and copied is not plot and copied.data is not plot.data
    assert "height:500px" in plot._repr_html_()


def test_fluent_options_deep_copy_and_merge_without_mutating_the_original():
    original = charts.bar(data={"category": ["A", "B"], "a": [2, 3], "b": [4, 5]},
                         x="category", y=["a", "b"], palette=["#abc", "#def"])
    before = original.model_dump()
    palette = ["#123", "#456"]
    limits = [-1, 10]
    first = original.with_options(title="New title", palette=palette, ylim=limits, legend_position="top")
    second = first.with_options(opacity=.5, y_label="Physical axis")
    palette[0], limits[0] = "#fff", -100
    second.config["series"][0]["values"][0] = 99
    second.config["options"]["palette"][0] = "#ffffff"
    assert original.model_dump() == before
    assert first.config["series"][0]["values"][0] == 2
    assert first.config["options"]["palette"] == ["#112233", "#445566"]
    assert first.config["options"]["ylim"] == [-1., 10.]
    assert first.title == first.config["title"] == "New title"
    assert first.x_label == original.x_label and first.y_label == original.y_label
    assert "opacity" not in first.config["options"]


def test_viewport_and_labels_do_not_resample_rebin_aggregate_or_rewrite_source_columns():
    plot = charts.scatter(data={"x": range(5000), "y": range(5000)}, x="x", y="y")
    before = plot.model_dump()
    styled = plot.with_options(xlim=[10, 20], ylim=[100, 200], x_label="Horizontal", y_label="Vertical")
    assert styled.data == plot.data
    assert (styled.sample_n, styled.total_n, styled.dropped_n) == (2000, 5000, 0)
    assert (styled.x_label, styled.y_label) == ("x", "y")
    assert plot.model_dump() == before
    hist = charts.hist(data={"x": [1, 2, 2, 3, None]}, x="x", bins=3)
    assert hist.with_options(xlim=[1.5, 2.5], ylim=[0, 5]).data == hist.data


@pytest.mark.parametrize("options", [
    {"unknown": True}, {"__proto__": {}}, {"width": True}, {"width": 319}, {"height": 1601},
    {"height": 400.5}, {"opacity": float("nan")}, {"opacity": -1}, {"opacity": True},
    {"point_size": 0}, {"line_width": 13}, {"color": "url(javascript:alert(1))"},
    {"palette": ["red"]}, {"palette": None}, {"grid": 1}, {"xlim": [1, 1]},
    {"xlim": [2, 1]}, {"xlim": [1, float("inf")]}, {"xlim": [True, 2]},
    {"xlim": [-1e308, 1e308]}, {"x_scale": "symlog"}, {"x_format": "\\input{secret}"},
    {"x_format": []}, {"x_label": 1},
])
def test_invalid_options_fail_before_producing_html_or_latex(options):
    plot = charts.line(data={"x": [1, 2], "y": [2, 3]}, x="x", y="y")
    with pytest.raises((TypeError, ValueError)):
        plot.with_options(**options)


def test_applicability_rejects_unrepresented_axes_marks_legends_and_baseline_removal():
    scatter, _, hist, coeff, bar, barh, area, stacked, donut = _plots()
    cases = [
        (scatter, {"legend": False}), (scatter, {"line_width": 2}),
        (hist, {"point_size": 2}), (hist, {"y_scale": "log"}), (hist, {"ylim": [1, 5]}),
        (coeff, {"ylim": [0, 2]}), (coeff, {"x_scale": "log"}), (coeff, {"xlim": [1, 2]}),
        (bar, {"xlim": [0, 2]}), (bar, {"y_format": "date"}), (bar, {"ylim": [1, 5]}),
        (barh, {"ylim": [0, 2]}), (barh, {"y_format": "number"}),
        (area, {"y_scale": "log"}), (stacked, {"line_width": 2}),
        (donut, {"grid": False}), (donut, {"x_label": "Absent"}), (donut, {"x_scale": "linear"}),
        (donut, {"legend_position": "left"}), (donut, {"palette": ["#123"]}),
        (coeff, {"palette": ["#123"]}),
    ]
    for plot, options in cases:
        with pytest.raises((TypeError, ValueError)):
            plot.with_options(**options)
    assert barh.with_options(xlim=[-1, 10], x_format="percent").config["options"]["xlim"] == [-1., 10.]
    assert bar.with_options(ylim=[0, 10], y_scale="linear").config["options"]["y_scale"] == "linear"


def test_log_scales_require_positive_stored_values_and_positive_limits_without_dropping_rows():
    for helper in (charts.line, charts.scatter):
        for bad in (0, -1):
            with pytest.raises(ValueError, match="strictly positive"):
                helper(data={"x": [1, bad], "y": [2, 3]}, x="x", y="y", x_scale="log")
        with pytest.raises(ValueError, match="strictly positive"):
            helper(data={"x": [1, 10], "y": [2, 3]}, x="x", y="y", x_scale="log", xlim=[0, 10])
    plot = charts.line(data={"x": [1, 10, None, 100], "y": [2, None, 999, 8]}, x="x", y="y")
    styled = plot.with_options(x_scale="log", y_scale="log", x_format="percent", y_format="scientific")
    assert styled.data == plot.data and styled.dropped_n == 2
    with pytest.raises(ValueError, match="strictly positive"):
        styled.with_options(xlim=[-1, 10])


def test_mutated_options_and_log_data_are_revalidated_before_any_export_is_written():
    plot = charts.scatter(data={"x": [1, 10], "y": [2, 3]}, x="x", y="y", x_scale="log")
    plot.data[1]["x"] = 0
    output = StringIO()
    with pytest.raises(ValueError, match="strictly positive"):
        plot.to_latex(output)
    assert not output.getvalue()
    fresh = charts.scatter(data={"x": [1], "y": [2]}, x="x", y="y", color="#123")
    fresh.config["options"]["color"] = "red"
    with pytest.raises(ValueError, match="color"):
        fresh.to_html()


@pytest.mark.parametrize("kind", ["line", "stackedArea", "stacked-area"])
def test_legacy_categorical_line_and_stacked_area_keep_categorical_x_axis(kind):
    original = charts.area(data={"category": ["A", "B"], "value": [1, 2]}, x="category", y="value")
    config = deepcopy(original.config)
    config["type"] = kind
    config["compositional"] = kind != "line"
    plot = charts.PlotSpec("d3", original.title, original.x_label, original.y_label,
                           [], original.sample_n, original.total_n, config=config)
    styled = plot.with_options(point_size=4, line_width=2, ylim=[0, 5], y_format="number")
    assert styled.config["options"]["ylim"] == [0., 5.]
    for options in ({"xlim": [0, 2]}, {"x_scale": "linear"}, {"x_format": "number"}, {"y_scale": "log"}):
        with pytest.raises(ValueError):
            plot.with_options(**options)


def test_latex_physical_axes_styles_and_viewport_preserve_signed_bar_geometry():
    plot = charts.barh(data={"category": ["A", "B"], "a": [-3, 4], "b": [1, 2]},
                      x="category", y=["a", "b"], palette=["#abc", "#def"])
    before = deepcopy(plot.config)
    styled = plot.with_options(x_label="Value 50%", y_label="Group_1", xlim=[-5, 10],
                               width=960, height=480, palette=["#123", "#456"],
                               opacity=.5, grid=False, legend=False, x_format="integer")
    source = styled.to_latex()
    assert "width=10.0in" in source and "height=5.0in" in source
    assert "xlabel={Value 50\\%}" in source and "ylabel={Group\\_1}" in source
    assert "xmin=-5.0" in source and "xmax=10.0" in source and "grid=none" in source
    assert "precision=0" in source and "scaled x ticks=false" in source
    assert "opacity=0.5" in source and r"\addlegendentry" not in source
    assert r"\definecolor{oecolor0}{HTML}{112233}" in source
    assert r"\definecolor{oecolor1}{HTML}{445566}" in source
    assert re.findall(r"\(axis cs:.*? rectangle .*?;", source) == re.findall(
        r"\(axis cs:.*? rectangle .*?;", plot.to_latex())
    assert plot.config == before


def test_latex_log_formats_show_original_quantity_and_preserve_missing_gaps():
    plot = charts.line(data={"x": [1, 10, 100], "y": [2, None, 8]}, x="x", y="y",
                       x_scale="log", x_format="percent", y_scale="log", y_format="number",
                       line_width=2, point_size=4, color="#123", opacity=.5)
    source = plot.to_latex()
    assert "xmode=log" in source and "ymode=log" in source
    assert r"100*(pow(10,\tick))" in source  # 1,10,100 become 100%,1000%,10000%.
    assert r"\pgfmathparse{pow(10,\tick)}" in source  # Other log labels also show original values.
    assert r"\usepgflibrary{fpu}" in source and r"\pgfmathfloattofixed" in source
    assert "line width=1.505625pt" in source and "mark size=3.01125pt" in source
    assert re.findall(r"coordinates \{(.*?)\};", source) == ["(1.0,2.0)", "(100.0,8.0)"]
    assert "xmin=1.0" in source and "xmax=100.0" in source


@pytest.mark.parametrize("value", [sys.float_info.max, math.nextafter(0., 1.), 1e-200, .01])
def test_constant_positive_log_bounds_are_finite_increasing_even_at_float_extremes(value):
    plot = charts.scatter(data={"x": [value], "y": [2]}, x="x", y="y", x_scale="log")
    source = plot.to_latex()
    lower = float(re.search(r"xmin=([^,]+)", source).group(1))
    upper = float(re.search(r"xmax=([^,]+)", source).group(1))
    assert math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper
    assert lower <= value <= upper


def test_palette_and_uniform_color_precedence_match_legend_visibility_and_positions():
    _, _, _, coeff, bar, _, _, _, donut = _plots()
    source = coeff.with_options(palette=["#123", "#456"], line_width=2).to_latex()
    assert r"\definecolor{oecolor1}{HTML}{445566}" in source
    for plot in (bar, donut):
        source = plot.with_options(palette=["#123", "#456"], color="#abc", legend_position="top").to_latex()
        assert r"\definecolor{oecolor0}{HTML}{aabbcc}" in source
        assert r"\definecolor{oecolor1}{HTML}{aabbcc}" in source
    assert "anchor=south" in bar.with_options(legend_position="top").to_latex()
    assert "anchor=north" in bar.with_options(legend_position="bottom").to_latex()
    styled = donut.with_options(width=640, height=480, legend=False, opacity=.4)
    source = styled.to_latex()
    assert r"\begin{tikzpicture}[scale=" in source and "transform shape" in source
    assert "use as bounding box" in source and "xscale" not in source and "yscale" not in source
    assert r"A: \texttt" not in source and r"B: \texttt" not in source
    assert "start angle=90.0,end angle=-54.0,radius=2" in source


def test_options_are_offline_json_safe_and_notebook_height_contains_the_requested_chart():
    plot = charts.scatter(data={"x": [1, 2], "y": [3, 4]}, x="x", y="y", height=1000,
                          x_label='</script><script>alert("x")</script>', color="#123")
    document = plot.to_html()
    payload = re.search(r'<script type="application/json" id="chart-data">(.*?)</script>', document, re.S).group(1)
    assert "<" not in payload and ">" not in payload
    assert json.loads(payload)["config"]["options"]["height"] == 1000
    assert "height:1180px" in plot._repr_html_()
    assert "allow-same-origin" not in plot._repr_html_()
