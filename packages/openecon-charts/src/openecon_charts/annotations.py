"""Validated presentation annotations, without changing chart observations."""
from __future__ import annotations

from numbers import Integral, Real
import math
import re
import unicodedata

_COLOR = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\Z")
_TYPES = {"text", "arrow", "vline", "hline", "vspan", "hspan", "segment"}
_TEXT_STYLE = {"font_size", "align", "dx", "dy"}
_LINE_STYLE = {"line_width", "dash"}
_POSITIONS = {"text": {"x", "y"}, "arrow": {"x", "y"}, "vline": {"x"},
              "hline": {"y"}, "vspan": {"x0", "x1"}, "hspan": {"y0", "y1"},
              "segment": {"x0", "y0", "x1", "y1"}}


def _number(value, name: str, *, lower=None, upper=None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Annotation {name} must be a finite number.")
    if isinstance(value, Integral) and abs(value) > 2**53 - 1:
        raise ValueError(f"Annotation {name} exceeds finite, browser-safe numeric precision.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"Annotation {name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"Annotation {name} must be a finite number.")
    if (lower is not None and result < lower) or (upper is not None and result > upper):
        raise ValueError(f"Annotation {name} must be between {lower} and {upper}.")
    return result


def _text(value) -> str:
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError("Annotation text must be a string of at most 500 characters.")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
        raise ValueError("Annotation text cannot contain control characters or invalid Unicode.")
    return value


def _category(plot, axis: str) -> list[str]:
    if plot.kind == "coefficients" and axis == "y":
        return [row["term"] for row in plot.data]
    if plot.kind == "d3":
        return plot.config["categories"]
    return []


def normalize_annotations(plot, annotations, options: dict) -> list[dict]:
    """Return canonical strict JSON annotations for physical chart axes."""
    from .charts import _chart_type, _numeric_axes
    if not isinstance(annotations, list) or len(annotations) > 100:
        raise ValueError("annotations must be a list of at most 100 entries.")
    chart_type, numeric_axes = _chart_type(plot), _numeric_axes(plot)
    result = []
    for entry in annotations:
        if not isinstance(entry, dict):
            raise TypeError("Each annotation must be a dictionary.")
        kind = entry.get("type")
        if not isinstance(kind, str) or kind not in _TYPES:
            raise ValueError("Annotation type must be text, arrow, vline, hline, vspan, hspan, or segment.")
        has_text = "text" in entry
        if kind in {"text", "arrow"} and not has_text:
            raise ValueError("Text and arrow annotations require text.")
        allowed = {"type", "coords", "color", "opacity", "text", *_POSITIONS[kind]}
        if has_text:
            allowed |= _TEXT_STYLE
        if kind in {"arrow", "vline", "hline", "segment"}:
            allowed |= _LINE_STYLE
        if set(entry) - allowed:
            unknown = sorted(set(entry) - allowed, key=str)[0]
            raise ValueError(f"Unsupported field for {kind} annotation: {unknown}.")
        missing = _POSITIONS[kind] - entry.keys()
        if missing:
            raise ValueError(f"Annotation {kind} requires {sorted(missing)[0]}.")
        coords = entry.get("coords", "data")
        if not isinstance(coords, str) or coords not in {"data", "axes"}:
            raise ValueError("Annotation coords must be 'data' or 'axes'.")
        if kind not in {"text", "arrow"} and coords != "data":
            raise ValueError("Reference lines and spans require data coordinates.")
        if coords == "data" and chart_type in {"donut", "gauge"}:
            raise ValueError("Donut and gauge annotations require axes coordinates.")
        item = {"type": kind, "coords": coords}
        for name in sorted(_POSITIONS[kind]):
            axis, value = name[0], entry[name]
            if coords == "axes":
                item[name] = _number(value, name, lower=0, upper=1)
            elif axis in numeric_axes:
                item[name] = _number(value, name)
                if options.get(axis + "_scale") == "log" and item[name] <= 0:
                    raise ValueError("Annotation coordinates on a log axis must be strictly positive.")
            elif kind in {"vline", "hline", "vspan", "hspan", "segment"}:
                raise ValueError(f"Annotation {kind} requires a numeric {axis} axis.")
            elif not isinstance(value, str) or value not in _category(plot, axis):
                raise ValueError(f"Annotation {axis} must exactly match an existing category or coefficient term.")
            else:
                if any(unicodedata.category(char) in {"Cc", "Cs"} for char in value):
                    raise ValueError("Annotation categories cannot contain control characters or invalid Unicode.")
                item[name] = value
        if kind in {"vspan", "hspan"}:
            axis = "x" if kind == "vspan" else "y"
            if item[axis + "0"] >= item[axis + "1"]:
                raise ValueError("Annotation span limits must be strictly increasing.")
        color = entry.get("color", "#162d4a")
        if not isinstance(color, str) or not _COLOR.fullmatch(color):
            raise ValueError("Annotation color must be an opaque CSS hex color.")
        item["color"] = "#" + "".join(char * 2 for char in color[1:]) if len(color) == 4 else color
        item["opacity"] = _number(entry.get("opacity", .12 if kind in {"vspan", "hspan"} else 1),
                                  "opacity", lower=0, upper=1)
        if has_text:
            item["text"] = _text(entry["text"])
            item["font_size"] = _number(entry.get("font_size", 12), "font_size", lower=8, upper=36)
            align = entry.get("align", "left")
            if not isinstance(align, str) or align not in {"left", "center", "right"}:
                raise ValueError("Annotation align must be left, center, or right.")
            item["align"] = align
            for name, default in (("dx", 12 if kind == "arrow" else 0), ("dy", -12 if kind == "arrow" else 0)):
                item[name] = _number(entry.get(name, default), name, lower=-500, upper=500)
        if kind in {"arrow", "vline", "hline", "segment"}:
            item["line_width"] = _number(entry.get("line_width", 1.5), "line_width", lower=.25, upper=12)
            dash = entry.get("dash", "solid" if kind == "arrow" else "dashed")
            if not isinstance(dash, str) or dash not in {"solid", "dashed", "dotted"}:
                raise ValueError("Annotation dash must be solid, dashed, or dotted.")
            item["dash"] = dash
        result.append(item)
    return result
