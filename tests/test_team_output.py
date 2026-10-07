"""Worker-generated chart data is an untrusted browser input boundary."""
from copy import deepcopy
import json

import openecon_charts as charts
import pytest

from openecon.team_output import validate_plot


def numeric(kind="scatter", size=2):
    rows = [{"x": float(i), "y": float(i + 1)} for i in range(size)]
    if kind == "hist":
        rows = [{"x0": float(i), "x1": float(i + 1), "count": 1} for i in range(size)]
    elif kind == "coefficients":
        rows = [{"term": f"term{i}", "estimate": 0.0, "ci_low": -1.0, "ci_high": 1.0}
                for i in range(size)]
    return charts.PlotSpec(kind, "Test", "x", "y", rows, size, size).model_dump()


def categorical(categories=2, series=1):
    config = {"type": "bar", "categories": [str(i) for i in range(categories)],
              "series": [{"id": str(i), "name": f"Series {i}", "values": [1.0] * categories}
                         for i in range(series)], "compositional": False}
    return charts.PlotSpec("d3", "Test", "Category", "Value", [], categories * series,
                           categories * series, config=config).model_dump()


def helper_plots():
    data = {"x": [1, 2, 3, 4], "y": [3, None, -4, 6]}
    categorical_data = {"x": ["a", "b", "c"], "y": [2, 3, 5], "z": [1, 2, 4]}
    yield charts.scatter(data=data, x="x", y="y")
    yield charts.line(data={"x": [1, 2, None, 5, 7], "y": [2, None, 10, -1, 6]}, x="x", y="y")
    yield charts.hist(data=data, x="y", bins=5)
    # A bias-corrected confidence interval need not contain its estimate.
    yield charts.coefficients({"coefficients": [{"term": "Intercept", "estimate": 5,
                                                 "ci_low": 1, "ci_high": 4}]})
    for helper in (charts.bar, charts.barh, charts.area):
        yield helper(data={"x": ["a", "b"], "y": [None, -1], "z": [3, 4]},
                     x="x", y=["y", "z"], palette=["#123", "#fed"])
    for horizontal in (False, True):
        yield charts.stacked_bar(data=categorical_data, x="x", y=["y", "z"], horizontal=horizontal)
    yield charts.donut(data=categorical_data, labels="x", values="y",
                       palette=["#123456", "#abcdef", "#ABCDEF"])


@pytest.mark.parametrize("plot", list(helper_plots()))
def test_public_helpers_survive_json_boundary_without_semantic_changes(plot):
    payload = json.loads(plot.model_dump_json())
    result = validate_plot(payload)
    assert result == payload
    assert result is not payload and result["data"] is not payload["data"]
    if result["config"]:
        result["config"]["series"][0]["values"][0] = 99
        assert payload == plot.model_dump()


@pytest.mark.parametrize("kind,limit", [("scatter", 2000), ("line", 10000),
                                        ("hist", 200), ("coefficients", 1000)])
def test_numeric_limits_accept_boundary_and_reject_one_more(kind, limit):
    assert len(validate_plot(numeric(kind, limit))["data"]) == limit
    with pytest.raises(ValueError):
        validate_plot(numeric(kind, limit + 1))


@pytest.mark.parametrize("categories,series", [(1000, 10), (400, 24)])
def test_categorical_limits_accept_boundary(categories, series):
    assert validate_plot(categorical(categories, series))["sample_n"] == categories * series


@pytest.mark.parametrize("categories,series", [(1001, 1), (1, 25), (1000, 11)])
def test_categorical_limits_reject_excess(categories, series):
    with pytest.raises(ValueError):
        validate_plot(categorical(categories, series))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"),
                                    2**53, True, "1", {"x": 1}, [1], None])
def test_scatter_never_coerces_invalid_geometry(value):
    payload = numeric()
    payload["data"][0]["x"] = value
    with pytest.raises(ValueError):
        validate_plot(payload)


@pytest.mark.parametrize("field,value", [("title", {}), ("x_label", ["x"]),
                                         ("y_label", 1), ("sample_n", True),
                                         ("sample_n", 3), ("sample_n", 1),
                                         ("total_n", -1), ("dropped_n", 1.5),
                                         ("title", "x" * 4097)])
def test_envelope_and_display_count_checks(field, value):
    payload = numeric()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_sampled_scatter_retains_original_valid_count():
    payload = charts.scatter(data={"x": list(range(3000)), "y": list(range(3000))},
                             x="x", y="y").model_dump()
    result = validate_plot(payload)
    assert result["sample_n"] == 2000 and result["total_n"] == 3000


@pytest.mark.parametrize("plot", [
    charts.scatter(data={"x": [1, 10], "y": [2, 20]}, x="x", y="y"),
    charts.barh(data={"region": ["A", "B"], "value": [4, 8]}, x="region", y="value"),
])
def test_shared_charts_retain_validated_presentation_without_changing_geometry(plot):
    styled = plot.with_options(color="#123", x_label="Physical horizontal", height=500)
    payload = json.loads(styled.model_dump_json())
    baseline = deepcopy(payload)
    result = validate_plot(payload)
    assert result == baseline and payload == baseline
    assert result["config"]["options"]["color"] == "#112233"
    assert result["data"] == plot.model_dump()["data"]
    result["config"]["options"]["color"] = "#ffffff"
    assert payload == baseline


@pytest.mark.parametrize("options", [
    {"color": "url(javascript:alert(1))"},
    {"height": 100000},
    {"x_label": "x" * 4097},
    {"x_scale": "log"},  # Includes zero in the stored scatter geometry.
    {"on_click": "alert(1)"},
    {"palette": ["#123456"] * 1001},
])
def test_shared_chart_presentation_remains_a_bounded_untrusted_input(options):
    payload = numeric()
    payload["config"] = {"options": options}
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_numeric_presentation_cannot_smuggle_an_unrecognized_renderer_configuration():
    payload = numeric()
    payload["config"] = {"options": {"grid": False}, "javascript": "alert(1)"}
    with pytest.raises(ValueError):
        validate_plot(payload)


@pytest.mark.parametrize("plot", list(helper_plots()))
def test_annotations_survive_shared_output_without_mutating_source(plot):
    annotated = plot.annotate("Note <tag> & 50%", x=.2, y=.8, coords="axes")
    payload = json.loads(annotated.model_dump_json())
    source = deepcopy(payload)
    result = validate_plot(payload)
    assert result == source and payload == source
    result["config"]["options"]["annotations"][0]["text"] = "Changed"
    assert payload == source
    assert not (plot.config or {}).get("options", {}).get("annotations")


@pytest.mark.parametrize("annotation", [
    {"type": "text", "text": "x" * 501, "x": 1, "y": 2},
    {"type": "text", "text": "Note", "x": 1, "y": 2, "onclick": "alert(1)"},
    {"type": "text", "text": "Note", "x": 1, "y": 2, "color": "url(javascript:1)"},
    {"type": "arrow", "text": "Note", "x": float("nan"), "y": 2},
    {"type": "vspan", "x0": 2, "x1": 1},
    {"type": "html", "html": "<script>alert(1)</script>"},
])
def test_shared_annotation_payload_rejects_invalid_or_executable_fields(annotation):
    payload = numeric()
    payload["config"] = {"options": {"annotations": [annotation]}}
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_shared_annotations_require_coordinates_compatible_with_saved_view():
    annotated = charts.bar(data={"group": ["A", "B"], "value": [1, 2]},
                           x="group", y="value").annotate("A", x="A", y=1)
    payload = annotated.model_dump()
    payload["config"]["view"] = {"type": "horizontal"}
    with pytest.raises(ValueError):
        validate_plot(payload)
    payload["config"]["view"] = {"type": "area"}
    assert validate_plot(payload) == payload


def test_percent_view_never_reinterprets_data_annotation_units():
    stacked = charts.stacked_bar(data={"group": ["A", "B"], "value": [2, 4]},
                                 x="group", y="value")
    payload = stacked.hline(3, text="Threshold").model_dump()
    payload["config"]["view"] = {"percent": True}
    with pytest.raises(ValueError):
        validate_plot(payload)
    payload = stacked.annotate("Share", x=.1, y=.9, coords="axes").model_dump()
    payload["config"]["view"] = {"percent": True}
    assert validate_plot(payload) == payload


@pytest.mark.parametrize("kind,row", [
    ("line", {"x": None, "y": 1}),
    ("line", {"x": 1}),
    ("hist", {"x0": 1, "x1": 1, "count": 1}),
    ("hist", {"x0": 0, "x1": 1, "count": 0.5}),
    ("hist", {"x0": 0, "x1": 1, "count": -1}),
    ("coefficients", {"term": {}, "estimate": 1, "ci_low": 0, "ci_high": 2}),
    ("coefficients", {"term": "x", "estimate": 1, "ci_low": 2, "ci_high": 0}),
])
def test_invalid_numeric_shapes(kind, row):
    payload = numeric(kind)
    payload["data"][0] = row
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_overlapping_histogram_and_duplicate_coefficient_labels_rejected():
    payload = numeric("hist")
    payload["data"][1]["x0"] = 0.5
    with pytest.raises(ValueError):
        validate_plot(payload)
    payload = numeric("coefficients")
    payload["data"][1]["term"] = payload["data"][0]["term"]
    with pytest.raises(ValueError):
        validate_plot(payload)


@pytest.mark.parametrize("name,value", [
    ("height", 1e100), ("height", 0), ("height", True), ("height", "320"),
    ("title", {}), ("xLabel", []), ("unit", 5), ("compositional", "yes"),
    ("legend", []), ("compact", "true"), ("locale", "tr-TR-u"), ("locale", []),
    ("palette", ["url(https://attacker.invalid)"]), ("segmentColors", {}),
    ("max", "100"), ("suggestedMax", float("inf")), ("unknown", {}),
    ("view", {"hidden": "0"}), ("view", {"hidden": ["foreign"]}),
    ("view", {"hiddenSlices": ["2"]}), ("view", {"type": "donut"}),
    ("view", {"percent": "false"}), ("view", {"percent": True}),
])
def test_invalid_renderer_options_are_rejected(name, value):
    payload = categorical()
    payload["config"][name] = value
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_saved_renderer_options_are_retained():
    payload = categorical()
    payload["config"].update({"type": "stackedArea", "compositional": True, "height": 1200,
                               "locale": "zh-Hant-TW", "legend": False, "compact": True,
                               "palette": ["#abcdef"], "segmentColors": ["#123456"],
                               "suggestedMax": 12, "categoryLabel": "Year",
                               "view": {"hidden": ["0"], "hiddenSlices": ["1"],
                                        "type": "donut", "percent": True}})
    assert validate_plot(payload) == payload


@pytest.mark.parametrize("chart_type", ["line", "area", "bar", "horizontal", "stackedBar",
                                       "stacked_bar", "stacked", "stacked-horizontal",
                                       "stackedArea", "stacked-area", "horizontalBar", "donut"])
def test_categorical_chart_types_and_aliases(chart_type):
    payload = categorical()
    payload["config"].update(type=chart_type, compositional=True)
    assert validate_plot(payload)["config"]["type"] == chart_type


@pytest.mark.parametrize("mutation", ["label_object", "duplicate_category", "duplicate_series",
                                       "series_name_object", "series_bool", "short_series",
                                       "negative_composition", "missing_composition", "zero_donut"])
def test_categorical_geometry_and_labels(mutation):
    payload = categorical(series=2 if mutation == "duplicate_series" else 1)
    config = payload["config"]
    if mutation == "label_object":
        config["categories"][0] = {}
    elif mutation == "duplicate_category":
        config["categories"][0] = config["categories"][1]
    elif mutation == "duplicate_series":
        config["series"][1]["id"] = config["series"][0]["id"]
    elif mutation == "series_name_object":
        config["series"][0]["name"] = {}
    elif mutation == "series_bool":
        config["series"][0]["values"][0] = True
    elif mutation == "short_series":
        config["series"][0]["values"].pop()
    else:
        config.update(type="donut", compositional=True)
        config["series"][0]["values"] = {"negative_composition": [-1, 1],
                                            "missing_composition": [None, 1],
                                            "zero_donut": [0, 0]}[mutation]
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_gauge_uses_single_bounded_numeric_value():
    payload = categorical(categories=1)
    payload["config"].update(type="gauge", max=5)
    assert validate_plot(payload) == payload
    payload["config"]["max"] = 0.5
    with pytest.raises(ValueError):
        validate_plot(payload)


@pytest.mark.parametrize("kind", ["scatter", "line", "hist", "coefficients", "d3"])
def test_overflowing_axis_span_is_not_sent_to_d3(kind):
    payload = categorical() if kind == "d3" else numeric(kind)
    if kind == "d3":
        payload["config"]["series"][0]["values"] = [-1e308, 1e308]
    elif kind == "coefficients":
        payload["data"][0].update(ci_low=-1e308, ci_high=1e308)
    elif kind == "hist":
        payload["data"][0]["x0"] = -1e308
        payload["data"][1]["x1"] = 1e308
    else:
        payload["data"][0]["x"], payload["data"][1]["x"] = -1e308, 1e308
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_composition_overflow_rejected_even_when_individual_values_are_finite():
    payload = categorical(series=2)
    payload["config"].update(type="stackedBar", compositional=True)
    for series in payload["config"]["series"]:
        series["values"] = [1e308, 1e308]
    with pytest.raises(ValueError):
        validate_plot(payload)


def test_nested_objects_cycles_and_non_json_types_fail_as_value_errors():
    payload = numeric()
    for invalid in [None, [], {"bad": set()}, {1: "bad"}]:
        with pytest.raises(ValueError):
            validate_plot(invalid)
    payload["config"] = {}
    payload["config"]["cycle"] = payload
    with pytest.raises(ValueError):
        validate_plot(payload)
    payload = numeric()
    before = deepcopy(payload)
    assert validate_plot(payload) == before and payload == before


def test_numeric_unused_config_cannot_reach_renderer():
    payload = numeric()
    payload["config"] = {"height": 1e200, "view": {"type": "bar"}}
    assert validate_plot(payload)["config"] is None
