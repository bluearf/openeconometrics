"""Validate untrusted worker plots before storing them or rendering in a browser.

The chart package's PlotSpec checks the display envelope. This boundary also
checks the geometry and renderer options; it never executes worker objects.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re
from typing import Any

from openecon_charts import PlotSpec

_LIMITS = {"scatter": 2000, "line": 10000, "hist": 200, "coefficients": 1000}
_ALIASES = {"stackedBar": "stacked", "stacked_bar": "stacked",
            "stackedArea": "stacked-area", "horizontalBar": "horizontal"}
_BASE_TYPES = {"area", "line", "bar", "horizontal"}
_STACK_TYPES = {"stacked", "stacked-horizontal", "stacked-area"}
_CONFIG_KEYS = {
    "type", "categories", "series", "compositional", "title", "unit", "xLabel",
    "yLabel", "categoryLabel", "locale", "height", "legend", "compact", "palette",
    "segmentColors", "max", "suggestedMax", "view", "options",
}
_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")
_LOCALE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z]{4})?(?:-[A-Za-z]{2}|-[0-9]{3})?\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any, *, nonempty: bool = False) -> None:
    _require(isinstance(value, str) and len(value) <= 4096, "Invalid chart text.")
    _require(not nonempty or bool(value.strip()), "Chart labels cannot be empty.")


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _json_boundary(data: Any) -> None:
    # Check before dataclass/asdict recursion and duplication. These bounds also
    # apply when this helper is called independently of the HTTP body limit.
    pending = [(data, 0)]
    visited = 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        _require(depth <= 12 and visited <= 100000, "Chart structure is too large.")
        if type(value) is dict:
            _require(all(isinstance(key, str) for key in value), "JSON keys must be strings.")
            _require(len(value) <= 10000, "Chart object is too large.")
            pending.extend((item, depth + 1) for item in value.values())
        elif type(value) is list:
            _require(len(value) <= 10000, "Chart array is too large.")
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is float:
            _require(math.isfinite(value), "Chart values must be finite.")
        elif type(value) is int:
            _require(abs(value) <= 2**53 - 1, "Chart integer exceeds browser precision.")
        elif type(value) is str:
            _require(len(value) <= 4096, "Chart text is too large.")
        else:
            _require(value is None or type(value) is bool, "Chart data must be JSON values.")
    _require(len(json.dumps(data, ensure_ascii=True, allow_nan=False).encode()) <= 2 * 1024**2,
             "Chart payload is too large.")


def _extent(values: list, *, zero: bool = False) -> None:
    present = [value for value in values if value is not None]
    if zero:
        present.append(0)
    if present:
        _require(math.isfinite(max(present) - min(present)), "Chart axis range is too large.")


def _processing(plot: dict, metadata: Any) -> None:
    required = {"mode", "passes", "source_rows", "projected_columns", "reader_batch_rows",
                "projected_working_limit_bytes", "integrity"}
    optional = {"sampling", "sample_limit", "extents", "aggregation", "kernel"}
    _require(type(metadata) is dict and required <= set(metadata) <= required | optional,
             "Invalid numeric chart processing metadata.")
    _require(metadata["mode"] == "streaming", "Invalid chart processing mode.")
    for name in ("passes", "source_rows", "reader_batch_rows", "projected_working_limit_bytes"):
        _require(type(metadata[name]) is int and metadata[name] > 0,
                 "Invalid chart processing count or budget.")
    _require(metadata["reader_batch_rows"] <= 65536
             and metadata["projected_working_limit_bytes"] <= 32 * 1024**2,
             "Invalid chart processing budget.")
    _require(metadata["source_rows"] == plot["total_n"] + plot["dropped_n"],
             "Chart source rows do not match its finite and dropped counts.")
    columns = metadata["projected_columns"]
    _require(type(columns) is list and 1 <= len(columns) <= 30,
             "Invalid chart projected columns.")
    for column in columns:
        _text(column, nonempty=True)
    _require(len(set(columns)) == len(columns), "Chart projected columns must be unique.")
    for name in ("integrity", "sampling", "aggregation", "kernel"):
        if name in metadata:
            _text(metadata[name], nonempty=True)
    if "sample_limit" in metadata:
        _require(plot["kind"] == "scatter" and type(metadata["sample_limit"]) is int
                 and plot["sample_n"] <= metadata["sample_limit"] <= 2000,
                 "Invalid scatter sample limit.")
    if "extents" in metadata:
        extents = metadata["extents"]
        _require(plot["kind"] == "scatter" and type(extents) is dict and set(extents) == {"x", "y"},
                 "Invalid scatter source extents.")
        for axis, values in extents.items():
            _require(type(values) is list and len(values) == 2 and all(_finite(v) for v in values)
                     and values[0] <= values[1], "Invalid scatter source extent.")
            _extent(values)
            _require(all(values[0] <= point[axis] <= values[1] for point in plot["data"]),
                     "Scatter source extents exclude displayed points.")


def _numeric(plot: dict) -> None:
    kind, rows = plot["kind"], plot["data"]
    _require(1 <= len(rows) <= _LIMITS[kind], "Invalid number of chart rows.")
    count = 0
    if kind in {"scatter", "line"}:
        for row in rows:
            _require(set(row) == {"x", "y"}, "Invalid coordinate fields.")
            _require(all(_finite(row[key]) or (kind == "line" and row[key] is None)
                         for key in ("x", "y")), "Invalid chart coordinates.")
            _require(row["x"] is not None or row["y"] is None,
                     "A missing line x coordinate must be an explicit gap.")
            count += _finite(row["x"]) and _finite(row["y"])
        for key in ("x", "y"):
            _extent([row[key] for row in rows])
    elif kind == "hist":
        previous = None
        for row in rows:
            _require(set(row) == {"x0", "x1", "count"}, "Invalid histogram fields.")
            _require(_finite(row["x0"]) and _finite(row["x1"]) and row["x0"] < row["x1"],
                     "Invalid histogram edges.")
            _require(type(row["count"]) is int and row["count"] >= 0,
                     "Histogram counts must be nonnegative integers.")
            _require(previous is None or row["x0"] >= previous, "Histogram bins overlap.")
            previous = row["x1"]
            count += row["count"]
        _extent([rows[0]["x0"], rows[-1]["x1"]])
    else:
        terms = set()
        for row in rows:
            _require(set(row) == {"term", "estimate", "ci_low", "ci_high"},
                     "Invalid coefficient fields.")
            _text(row["term"], nonempty=True)
            _require(row["term"] not in terms, "Coefficient terms must be unique.")
            terms.add(row["term"])
            _require(all(_finite(row[key]) for key in ("estimate", "ci_low", "ci_high"))
                     and row["ci_low"] <= row["ci_high"], "Invalid confidence interval.")
        count = len(rows)
        _extent([row[key] for row in rows for key in ("estimate", "ci_low", "ci_high")],
                zero=True)
    _require(count > 0 and plot["sample_n"] == count, "Chart sample count does not match geometry.")
    # Older numeric envelopes carried unused renderer configuration. Retain that
    # compatibility, but preserve the new validated presentation contract.
    config = plot["config"]
    if config is None or not ({"options", "processing"} & set(config)):
        plot["config"] = None
    else:
        _require(set(config) <= {"options", "processing"}, "Unsupported numeric chart configuration.")
        if "processing" in config:
            _processing(plot, config["processing"])


def _colors(values: Any) -> None:
    _require(isinstance(values, list) and 1 <= len(values) <= 1000,
             "Invalid chart palette size.")
    _require(all(isinstance(value, str) and _COLOR.fullmatch(value) for value in values),
             "Chart colors must be six-digit hex values.")


def _categorical(plot: dict) -> None:
    config = plot["config"]
    _require(not plot["data"], "Categorical geometry belongs in config.")
    _require(set(config) <= _CONFIG_KEYS, "Unsupported chart options.")
    kind = config.get("type", "bar")
    _text(kind)
    kind = _ALIASES.get(kind, kind)
    _require(kind in _BASE_TYPES | _STACK_TYPES | {"donut", "gauge"}, "Unsupported chart type.")
    categories, series = config.get("categories"), config.get("series")
    _require(isinstance(categories, list) and 1 <= len(categories) <= 1000,
             "Invalid number of chart categories.")
    for value in categories:
        _text(value, nonempty=True)
    _require(len(set(categories)) == len(categories), "Chart categories must be unique.")
    _require(isinstance(series, list) and 1 <= len(series) <= 24
             and len(series) * len(categories) <= 10000, "Invalid number of chart values.")
    identifiers, values, count = set(), [], 0
    for index, item in enumerate(series):
        _require(isinstance(item, dict) and set(item) <= {"id", "name", "values", "color"},
                 "Invalid series fields.")
        item.setdefault("id", str(index))
        item.setdefault("name", item["id"])
        for key in ("id", "name"):
            _text(item[key], nonempty=True)
        _require(item["id"] not in identifiers, "Chart series IDs must be unique.")
        identifiers.add(item["id"])
        column = item.get("values")
        _require(isinstance(column, list) and len(column) == len(categories),
                 "Series values must match the categories.")
        _require(all(value is None or _finite(value) for value in column), "Invalid series values.")
        values.extend(column)
        count += sum(value is not None for value in column)
        if "color" in item:
            _colors([item["color"]])
    _require(count > 0 and plot["sample_n"] == count, "Chart sample count does not match geometry.")
    composition = config.get("compositional", False)
    _require(type(composition) is bool, "compositional must be a boolean.")
    if composition:
        _require(all(value is not None and value >= 0 for value in values),
                 "Compositions require complete nonnegative values.")
        totals = [math.fsum(item["values"][i] for item in series) for i in range(len(categories))]
        _require(all(math.isfinite(total) for total in totals), "Composition totals overflow.")
        if len(series) == 1:
            _require(math.isfinite(math.fsum(values)), "Composition total overflows.")
        _extent(totals, zero=True)
    _require(kind not in _STACK_TYPES | {"donut"} or composition,
             "Stacked and donut charts must be compositional.")
    if kind == "donut":
        _require(len(series) == 1 and any(value > 0 for value in values),
                 "A donut requires one series with a positive total.")
    for name in ("title", "unit", "xLabel", "yLabel", "categoryLabel"):
        if name in config:
            _text(config[name])
    for name in ("legend", "compact"):
        if name in config:
            _require(type(config[name]) is bool, f"{name} must be a boolean.")
    if "height" in config:
        _require(_finite(config["height"]) and 120 <= config["height"] <= 1200,
                 "Chart height must be between 120 and 1200 pixels.")
    if "locale" in config:
        _require(isinstance(config["locale"], str) and bool(_LOCALE.fullmatch(config["locale"])),
                 "Invalid chart locale.")
    for name in ("palette", "segmentColors"):
        if name in config:
            _colors(config[name])
    for name in ("max", "suggestedMax"):
        if name in config:
            _require(_finite(config[name]), f"{name} must be finite.")
    if "suggestedMax" in config:
        values.append(config["suggestedMax"])
    _extent(values, zero=True)
    if kind == "gauge":
        maximum = config.get("max", 100)
        _require(maximum > 0 and len(series) == len(categories) == 1
                 and 0 <= series[0]["values"][0] <= maximum, "Invalid gauge value or maximum.")
    if "view" in config:
        view = config["view"]
        _require(isinstance(view, dict) and set(view) <= {"hidden", "hiddenSlices", "type", "percent"},
                 "Invalid saved chart view.")
        for name, allowed in (("hidden", identifiers),
                              ("hiddenSlices", {str(i) for i in range(len(categories))})):
            if name in view:
                selected = view[name]
                _require(isinstance(selected, list) and len(selected) <= len(allowed)
                         and all(isinstance(value, str) and value in allowed for value in selected),
                         "Invalid hidden chart items.")
                _require(len(set(selected)) == len(selected), "Hidden chart items must be unique.")
        if "percent" in view:
            _require(type(view["percent"]) is bool, "percent must be a boolean.")
            _require(not view["percent"] or composition, "Percent view requires a composition.")
        if "type" in view:
            choices = {"gauge", "bar"} if kind == "gauge" else _BASE_TYPES.copy()
            if composition and kind != "gauge":
                choices |= _STACK_TYPES
                if len(series) == 1 and any(value > 0 for value in series[0]["values"]):
                    choices.add("donut")
            _require(isinstance(view["type"], str) and view["type"] in choices,
                     "Invalid saved chart type.")


def validate_plot(data: Any) -> dict[str, Any]:
    """Return a detached, bounded PlotSpec dictionary; reject invalid input with ValueError.

    Public plotting helpers and categorical renderer views are supported. Labels
    are plain strings of at most 4096 characters; custom JavaScript renderer
    options, unknown fields, and implicit coercion are intentionally unsupported.
    """
    try:
        _require(type(data) is dict, "A plot must be an object.")
        _json_boundary(data)
        from openecon_charts.timeline import unpack
        expanded = unpack(data)
        # Pooled records do not bypass the existing shared-result admission.
        if expanded is not data:
            _json_boundary(expanded)
        plot = PlotSpec(**expanded).model_dump()
        options = (plot["config"] or {}).get("options", {})
        # PlotSpec checks option names, types, applicability and coordinate
        # semantics. This shared-result boundary adds its own payload bounds.
        for name in ("x_label", "y_label"):
            if name in options:
                _text(options[name])
        if "palette" in options:
            _colors(options["palette"])
        if "color" in options:
            _colors([options["color"]])
        annotations = options.get("annotations", [])
        for annotation in annotations:
            if "text" in annotation:
                _text(annotation["text"])
            if "color" in annotation:
                _colors([annotation["color"]])
        if plot["kind"] == "network":
            from openecon_charts.network import validate_network
            _require(isinstance(plot["config"], dict)
                     and set(plot["config"]) <= {"network", "options"},
                     "Unsupported network chart configuration.")
            plot["config"]["network"] = validate_network(plot["config"]["network"])
        elif plot["kind"] == "d3":
            _categorical(plot)
            view = plot["config"].get("view", {})
            if annotations:
                _require(not view.get("percent") or all(
                    annotation.get("coords", "data") == "axes" for annotation in annotations
                ), "Percent view cannot reinterpret data-coordinate annotations.")
                if "type" in view:
                    viewed_plot = deepcopy(plot)
                    viewed_plot["config"]["type"] = view["type"]
                    # Physical coordinates must remain meaningful after a saved
                    # chart-type conversion, just as in the interactive renderer.
                    PlotSpec(**viewed_plot).model_dump()
        else:
            _numeric(plot)
        return plot
    except (TypeError, OverflowError, RecursionError, KeyError) as exc:
        raise ValueError("Invalid chart structure or geometry.") from exc
