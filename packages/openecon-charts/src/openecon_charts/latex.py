"""Dependency-free, data-faithful TikZ/PGFPlots exports.

Fragments require ``tikz`` and ``pgfplots`` in the host document. Standalone
documents use XeLaTeX or LuaLaTeX through ``fontspec`` for Unicode labels.
Only the chart specification's stored observations are exported; scatter
samples are never expanded back into observations that were not displayed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
import math
from pathlib import Path
import re
from typing import Any

_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
}
_COLORS = ("162D4A", "496582", "7892AD", "A8B8CB", "C9D3DF", "DEE5ED")
_LABEL = re.compile(r"[A-Za-z0-9:_.+\-]+\Z")
_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\Z")


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("LaTeX text must be a string.")
    result = []
    for char in value:
        if char.isspace():
            result.append(" ")
        elif ord(char) < 32 or ord(char) == 127:
            raise ValueError("LaTeX text cannot contain control characters.")
        else:
            result.append(_ESCAPES.get(char, char))
    return "".join(result)


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("LaTeX chart coordinates must be finite numbers.")
    # Python's shortest round-tripping representation preserves every stored
    # float; do not round estimates or confidence limits for this export.
    return repr(value)


def _bounds(values: Sequence[int | float], *, zero: bool = False, log: bool = False) -> tuple[str, str]:
    if not values:
        raise ValueError("The LaTeX chart needs at least one finite value.")
    for value in values:
        _number(value)
    lower, upper = min(values), max(values)
    if zero:
        lower, upper = min(lower, 0), max(upper, 0)
    if lower == upper:
        if log:
            lower, upper = lower * .8, upper * 1.2
            # At the ends of binary64, multiplicative padding can underflow,
            # overflow or round back to the same value. Use adjacent finite
            # positive values instead of manufacturing zero or infinity.
            if lower <= 0 or not math.isfinite(lower) or lower == upper:
                lower = math.nextafter(min(values), 0)
                if lower <= 0:
                    lower = min(values)
            if not math.isfinite(upper) or upper <= lower:
                upper = math.nextafter(max(values), math.inf)
                if not math.isfinite(upper):
                    upper = max(values)
        else:
            padding = max(abs(lower) * .08, .5)
            lower, upper = lower - padding, upper + padding
    if log and lower <= 0:
        raise ValueError("Log chart limits must stay strictly positive.")
    if log and lower >= upper:
        raise ValueError("Log chart limits collapse at this numeric precision; rescale the values.")
    if not math.isfinite(upper - lower):
        raise ValueError("The LaTeX chart axis range is too large. Rescale the data explicitly.")
    return _number(lower), _number(upper)


def _options(plot) -> dict:
    return (plot.config or {}).get("options", {})


def _style(plot, *, point=False, line=False) -> str:
    options = _options(plot)
    parts = []
    if "opacity" in options:
        parts.append("opacity=" + _number(options["opacity"]))
    if point and "point_size" in options:
        parts.append("mark size=" + _number(options["point_size"] * 72.27 / 96) + "pt")
    if line and "line_width" in options:
        parts.append("line width=" + _number(options["line_width"] * 72.27 / 96) + "pt")
    return "," + ",".join(parts) if parts else ""


def _annotation_node(item: dict, anchor: str, index: int, *, band=False) -> str:
    """CSS pixel offsets use screen y; TikZ uses the opposite vertical sign."""
    factor = 72.27 / 96
    size = _number(item["font_size"] * factor)
    leading = _number(item["font_size"] * factor * 1.2)
    alignment = {"left": "west", "center": "center", "right": "east"}[item["align"]]
    opacity = _number(1.0 if band else item["opacity"])
    return (f"\\node[anchor={alignment},text=oeann{index},opacity={opacity},inner sep=0pt,"
            f"font=\\fontsize{{{size}pt}}{{{leading}pt}}\\selectfont] at "
            f"([xshift={_number(item['dx'] * factor)}pt,yshift={_number(-item['dy'] * factor)}pt]{anchor}) "
            f"{{{_text(item['text'])}}};")


def _annotation_line_style(item: dict, index: int) -> str:
    return (f"color=oeann{index},opacity={_number(item['opacity'])},"
            f"line width={_number(item['line_width'] * 72.27 / 96)}pt,{item['dash']}")


def _axis_annotation_layers(plot, xlim, ylim) -> tuple[list[str], list[str]]:
    """Project into the fixed viewport, without adding observations or limits.

    Axis-description coordinates stay physical even when categorical y is
    reversed. Computing ratios before giving them to TeX also avoids huge
    dimension arithmetic for valid, far-off-viewport finite coordinates.
    """
    notes = _options(plot).get("annotations", [])
    if not notes:
        return [], []
    from .charts import _numeric_axes
    numeric = _numeric_axes(plot)
    limits = {"x": tuple(map(float, xlim)), "y": tuple(map(float, ylim))}
    reverse_y = plot.kind == "coefficients" or (plot.kind == "d3" and
                plot.config["type"] in {"horizontal", "horizontalBar", "stacked-horizontal"})

    def fraction(value, axis):
        if axis not in numeric:
            categories = ([row["term"] for row in plot.data] if plot.kind == "coefficients"
                          else plot.config["categories"])
            result = (categories.index(value) + .5) / len(categories)
        else:
            lower, upper = limits[axis]
            if _options(plot).get(axis + "_scale") == "log":
                def log_delta(a, b):
                    relative = (a - b) / b
                    return math.log1p(relative) if math.isfinite(relative) and relative > -1 else math.log(a) - math.log(b)
                result = log_delta(value, lower) / log_delta(upper, lower)
            else:
                extent = upper - lower
                result = (value - lower) / extent if math.isfinite(value - lower) else value / extent - lower / extent
        return 1 - result if axis == "y" and reverse_y else result

    def coord(x, y):
        return f"(axis description cs:{_number(x)},{_number(y)})"

    clip = r"\clip (axis description cs:0,0) rectangle (axis description cs:1,1);"
    behind, front = [], []
    for i, item in enumerate(notes):
        kind = item["type"]
        header = "% Annotation " + json.dumps(item, ensure_ascii=True, separators=(",", ":"))
        label_anchor = None
        if kind in {"vspan", "hspan"}:
            axis = "x" if kind == "vspan" else "y"
            first, last = fraction(item[axis + "0"], axis), fraction(item[axis + "1"], axis)
            first, last = sorted((max(0., min(1., first)), max(0., min(1., last))))
            if first == last:
                continue
            a, b = ((first, 0), (last, 1)) if axis == "x" else ((0, first), (1, last))
            behind.extend([header, r"\begin{scope}[overlay]", clip,
                           f"\\path[fill=oeann{i},fill opacity={_number(item['opacity'])},draw=none] "
                           + coord(*a) + " rectangle " + coord(*b) + ";", r"\end{scope}"])
            label_anchor = coord((first + last) / 2 if axis == "x" else .5,
                                 (first + last) / 2 if axis == "y" else .5)
        elif kind == "segment":
            a = (fraction(item["x0"], "x"), fraction(item["y0"], "y"))
            b = (fraction(item["x1"], "x"), fraction(item["y1"], "y"))
            front.extend([header, r"\begin{scope}[overlay]", clip,
                          f"\\draw[{_annotation_line_style(item, i)}] " + coord(*a) + " -- " + coord(*b) + ";",
                          r"\end{scope}"])
            continue
        elif kind in {"vline", "hline"}:
            axis = "x" if kind == "vline" else "y"
            position = fraction(item[axis], axis)
            if not 0 <= position <= 1:
                continue
            a, b = ((position, 0), (position, 1)) if axis == "x" else ((0, position), (1, position))
            front.extend([header, r"\begin{scope}[overlay]", clip,
                          f"\\draw[{_annotation_line_style(item, i)}] " + coord(*a) + " -- " + coord(*b) + ";"])
            if "text" in item:
                edge = coord(position, 1) if axis == "x" else coord(0, position)
                shift = (f"yshift={_number(-item['font_size'] * 72.27 / 96)}pt" if axis == "x"
                         else f"xshift={_number(12 * 72.27 / 96)}pt")
                front.append(f"\\coordinate (oeannanchor{i}) at ([{shift}]{edge[1:-1]});")
                front.append(_annotation_node(item, f"oeannanchor{i}", i))
            front.append(r"\end{scope}")
            continue
        else:
            x, y = ((item["x"], item["y"]) if item["coords"] == "axes"
                    else (fraction(item["x"], "x"), fraction(item["y"], "y")))
            # An anchor outside the fixed viewport is not displayed, even
            # when a pixel offset could move its text back into view.
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            label_anchor = coord(x, y)
        if label_anchor is not None and "text" in item:
            front.extend([header, r"\begin{scope}[overlay]", clip])
            if kind == "arrow":
                factor = 72.27 / 96
                front.append(f"\\coordinate (oeannanchor{i}) at {label_anchor};")
                front.append(f"\\draw[->,{_annotation_line_style(item, i)}] "
                             f"([xshift={_number(item['dx'] * factor)}pt,yshift={_number(-item['dy'] * factor)}pt]oeannanchor{i}) "
                             f"-- (oeannanchor{i});")
                label_anchor = f"oeannanchor{i}"
            front.append(_annotation_node(item, label_anchor[1:-1] if label_anchor.startswith("(") else label_anchor,
                                          i, band=kind in {"vspan", "hspan"}))
            front.append(r"\end{scope}")
    return behind, front


def _donut_annotations(plot) -> list[str]:
    """Relative notes use the full picture bounds inset by 20 CSS pixels."""
    notes = _options(plot).get("annotations", [])
    if not notes:
        return []
    inset = _number(20 * (72.27 / 96))
    result = [r"\begin{scope}[overlay,reset cm,transform shape=false]",
              f"\\coordinate (oeannbl) at ([xshift={inset}pt,yshift={inset}pt]current bounding box.south west);",
              f"\\coordinate (oeannbr) at ([xshift=-{inset}pt,yshift={inset}pt]current bounding box.south east);",
              f"\\coordinate (oeanntl) at ([xshift={inset}pt,yshift=-{inset}pt]current bounding box.north west);",
              f"\\coordinate (oeanntr) at ([xshift=-{inset}pt,yshift=-{inset}pt]current bounding box.north east);",
              r"\clip (oeannbl) rectangle (oeanntr);"]
    for i, item in enumerate(notes):
        x, y = _number(item["x"]), _number(item["y"])
        result.extend(["% Annotation " + json.dumps(item, ensure_ascii=True, separators=(",", ":")),
                       f"\\coordinate (oeannx{i}) at ($(oeannbl)!{x}!(oeannbr)$);",
                       f"\\coordinate (oeanny{i}) at ($(oeannbl)!{y}!(oeanntl)$);",
                       f"\\coordinate (oeannanchor{i}) at (oeannx{i} |- oeanny{i});"])
        if item["type"] == "arrow":
            factor = 72.27 / 96
            result.append(f"\\draw[->,{_annotation_line_style(item, i)}] "
                          f"([xshift={_number(item['dx'] * factor)}pt,yshift={_number(-item['dy'] * factor)}pt]oeannanchor{i}) "
                          f"-- (oeannanchor{i});")
        result.append(_annotation_node(item, f"oeannanchor{i}", i))
    return [*result, r"\end{scope}"]


def _axis(title: str, xlabel: str, ylabel: str, *, xlim, ylim, extra=(), plot=None) -> list[str]:
    presentation = _options(plot) if plot is not None else {}
    xlabel, ylabel = presentation.get("x_label", xlabel), presentation.get("y_label", ylabel)
    for axis in ("x", "y"):
        limit = presentation.get(axis + "lim")
        if limit is None and presentation.get(axis + "_scale") == "log":
            from .charts import _axis_values
            limit = _bounds(_axis_values(plot, axis), log=True)
        if limit is not None:
            if axis == "x":
                xlim = tuple(_number(value) if not isinstance(value, str) else value for value in limit)
            else:
                ylim = tuple(_number(value) if not isinstance(value, str) else value for value in limit)
    legend = r"legend style={draw=none,font=\small}"
    if "legend_position" in presentation:
        location = "at={(0.5,1.08)},anchor=south" if presentation["legend_position"] == "top" else "at={(0.5,-0.2)},anchor=north"
        legend = r"legend style={draw=none,font=\small," + location + "}"
    options = [
        "width=" + _number(presentation["width"] / 96) + "in" if "width" in presentation else r"width=0.9\linewidth",
        "height=" + _number(presentation["height"] / 96) + "in" if "height" in presentation else "height=7cm",
        "scale only axis",
        f"title={{{_text(title)}}}", f"xlabel={{{_text(xlabel)}}}",
        f"ylabel={{{_text(ylabel)}}}",
        f"xmin={xlim[0]}", f"xmax={xlim[1]}",
        f"ymin={ylim[0]}", f"ymax={ylim[1]}",
        "enlargelimits=false", "axis lines=left", "grid=major" if presentation.get("grid", True) else "grid=none",
        "grid style={gray!15}", "tick align=outside", "clip=true",
        legend, *extra,
    ]
    for axis in ("x", "y"):
        if presentation.get(axis + "_scale") == "log":
            options.extend([axis + "mode=log", "log basis " + axis + "=10"])
        formatting = presentation.get(axis + "_format", "auto")
        if formatting != "auto":
            options.append("scaled " + axis + " ticks=false")
            style = {"number": "fixed,precision=10", "integer": "fixed,precision=0",
                     "percent": "fixed,precision=10", "scientific": "sci,precision=3"}[formatting]
            # With an explicit log basis of 10, PGFPlots' \tick is the
            # exponent, not the original coordinate. Reconstruct that value
            # before formatting; never label log(10) as 1 or 100%.
            logarithmic = presentation.get(axis + "_scale") == "log"
            if logarithmic or formatting == "percent":
                expression = r"pow(10,\tick)" if logarithmic else r"\tick"
                if formatting == "percent":
                    expression = "100*(" + expression + ")"
                label = (r"\begingroup\pgfkeys{/pgf/fpu=true}\pgfmathparse{" + expression + "}"
                         r"\pgfmathfloattofixed{\pgfmathresult}\pgfmathprintnumber[" + style + r"]{\pgfmathresult}"
                         + (r"\%" if formatting == "percent" else "") + r"\endgroup")
                options.append(axis + "ticklabel={" + label + "}")
            else:
                options.append(axis + "ticklabel style={/pgf/number format/.cd," + style + "}")
    behind, front = _axis_annotation_layers(plot, xlim, ylim) if plot is not None else ([], [])
    if front:
        options.append("execute at end axis={\n" + "\n".join(front) + "\n}")
    return [r"\begin{axis}[", "  " + ",\n  ".join(options), "]", *behind]


def _coordinates(points) -> str:
    return " ".join(f"({_number(x)},{_number(y)})" for x, y in points)


def _segments(points) -> list[list[tuple[int | float, int | float]]]:
    result, current = [], []
    for x, y in points:
        if x is None or y is None:
            if current:
                result.append(current)
                current = []
        else:
            _number(x)
            _number(y)
            current.append((x, y))
    if current:
        result.append(current)
    return result


def _color(value: str | None, index: int) -> str:
    if value is None:
        return _COLORS[index % len(_COLORS)]
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError("LaTeX chart colors must be CSS hex colors.")
    digits = value[1:]
    return "".join(char * 2 for char in digits) if len(digits) == 3 else digits


def _define_colors(colors) -> list[str]:
    return [f"\\definecolor{{oecolor{i}}}{{HTML}}{{{value}}}" for i, value in enumerate(colors)]


def _numeric(plot) -> list[str]:
    points = [(row["x"], row["y"]) for row in plot.data]
    segments = _segments(points)
    output = _axis(plot.title, plot.x_label, plot.y_label,
                   xlim=_bounds([x for x, _ in points if x is not None], log=_options(plot).get("x_scale") == "log"),
                   ylim=_bounds([y for _, y in points if y is not None], log=_options(plot).get("y_scale") == "log"), plot=plot)
    if plot.kind == "scatter":
        output.append(r"\addplot[only marks,mark=*,mark size=1.5pt,color=oecolor0" + _style(plot, point=True) + "] coordinates {"
                      + _coordinates(point for segment in segments for point in segment) + "};")
    else:
        # Separate plots cannot connect across missing x/y observations.
        for segment in segments:
            output.append(r"\addplot[color=oecolor0,mark=*,mark size=1.5pt" + _style(plot, point=True, line=True) + "] coordinates {"
                          + _coordinates(segment) + "};")
    output.append(r"\end{axis}")
    return output


def _histogram(plot) -> list[str]:
    for row in plot.data:
        if row["x0"] >= row["x1"] or row["count"] < 0:
            raise ValueError("Histogram bins must have ordered edges and nonnegative counts.")
    output = _axis(plot.title, plot.x_label, plot.y_label,
                   xlim=_bounds([row[key] for row in plot.data for key in ("x0", "x1")]),
                   ylim=_bounds([row["count"] for row in plot.data], zero=True), plot=plot)
    # Draw the exact supplied edges instead of recomputing center/width pairs.
    for row in plot.data:
        output.append("\\draw[draw=white,fill=oecolor0" + _style(plot) + "] "
                      f"(axis cs:{_number(row['x0'])},0) rectangle "
                      f"(axis cs:{_number(row['x1'])},{_number(row['count'])});")
    output.append(r"\end{axis}")
    return output


def _coefficient_plot(plot) -> list[str]:
    limits = [0]
    labels = []
    for row in plot.data:
        for key in ("estimate", "ci_low", "ci_high"):
            _number(row[key])
            limits.append(row[key])
        if row["ci_low"] > row["ci_high"]:
            raise ValueError("Confidence interval lower limits cannot exceed upper limits.")
        labels.append(row["term"])
    n = len(labels)
    if not n:
        raise ValueError("The coefficient chart needs at least one term.")
    output = _axis(plot.title, plot.x_label, plot.y_label,
                   xlim=_bounds(limits), ylim=(_number(-.5), _number(n - .5)),
                   extra=["y dir=reverse", "ytick={" + ",".join(map(str, range(n))) + "}",
                          "yticklabels={" + ",".join("{" + _text(term) + "}" for term in labels) + "}"], plot=plot)
    output.append(f"\\draw[dashed,gray] (axis cs:0,-0.5) -- (axis cs:0,{_number(n - .5)});")
    # Draw interval endpoints directly. This also correctly represents valid
    # intervals which do not contain the point estimate, without negative
    # distances in PGFPlots' asymmetric-error-bar syntax.
    for index, row in enumerate(plot.data):
        low, high = _number(row["ci_low"]), _number(row["ci_high"])
        output.append(f"\\draw[color=oecolor1,thick{_style(plot, line=True)}] (axis cs:{low},{index}) -- (axis cs:{high},{index});")
        for endpoint in (low, high):
            output.append(f"\\draw[color=oecolor1{_style(plot, line=True)}] ([yshift=-2pt]axis cs:{endpoint},{index})"
                          f" -- ([yshift=2pt]axis cs:{endpoint},{index});")
    output.append(r"\addplot[only marks,mark=*,mark size=2pt,color=oecolor0" + _style(plot, point=True) + "] coordinates {"
                  + _coordinates((row["estimate"], i) for i, row in enumerate(plot.data)) + "};")
    output.append(r"\end{axis}")
    return output


def _categorical_data(plot):
    config = plot.config
    if not isinstance(config, Mapping):
        raise ValueError("Categorical LaTeX charts require a configuration.")
    categories, series = config.get("categories"), config.get("series")
    if (not isinstance(categories, list) or not categories or
            not all(isinstance(value, str) for value in categories)):
        raise ValueError("Categorical LaTeX charts require category labels.")
    if not isinstance(series, list) or not series:
        raise ValueError("Categorical LaTeX charts require at least one series.")
    for item in series:
        if (not isinstance(item, Mapping) or not isinstance(item.get("name"), str) or
                not isinstance(item.get("values"), list) or len(item["values"]) != len(categories)):
            raise ValueError("Categorical series must have a name and one value per category.")
        for value in item["values"]:
            if value is not None:
                _number(value)
    return config, categories, series


def _legend(series) -> list[str]:
    result = []
    for index, item in enumerate(series):
        result.extend([f"\\addlegendimage{{area legend,draw=oecolor{index},fill=oecolor{index}}}",
                       f"\\addlegendentry{{{_text(item['name'])}}}"])
    return result


def _bars(plot, config, categories, series) -> list[str]:
    kind = config["type"]
    horizontal = kind in {"horizontal", "stacked-horizontal"}
    stacked = kind in {"stackedBar", "stacked", "stacked_bar", "stacked-horizontal"}
    if stacked and any(value is None or value < 0 for item in series for value in item["values"]):
        raise ValueError("Stacked charts require complete, nonnegative values.")
    values = [value for item in series for value in item["values"] if value is not None]
    if stacked:
        values = [math.fsum(item["values"][i] for item in series) for i in range(len(categories))]
    value_lim = _bounds(values, zero=True)
    category_lim = (_number(-.5), _number(len(categories) - .5))
    ticks = "ytick" if horizontal else "xtick"
    ticklabels = "yticklabels" if horizontal else "xticklabels"
    extra = [ticks + "={" + ",".join(map(str, range(len(categories)))) + "}",
             ticklabels + "={" + ",".join("{" + _text(value) + "}" for value in categories) + "}"]
    if horizontal:
        extra.append("y dir=reverse")
    else:
        extra.append("x tick label style={rotate=35,anchor=east}")
    output = _axis(plot.title, plot.y_label if horizontal else plot.x_label,
                   plot.x_label if horizontal else plot.y_label,
                   xlim=value_lim if horizontal else category_lim,
                   ylim=category_lim if horizontal else value_lim, extra=extra, plot=plot)
    step = .78 if stacked else .78 / len(series)
    for si, item in enumerate(series):
        for ci, value in enumerate(item["values"]):
            if value is None:
                continue
            left = ci - .39 + (0 if stacked else si * step + step * .075)
            right = left + step * (1 if stacked else .85)
            base = math.fsum(previous["values"][ci] for previous in series[:si]) if stacked else 0
            top = math.fsum(previous["values"][ci] for previous in series[:si + 1]) if stacked else value
            first, second = ((base, left), (top, right)) if horizontal else ((left, base), (right, top))
            output.append(f"\\draw[draw=white,fill=oecolor{si}{_style(plot)}] "
                          f"(axis cs:{_number(first[0])},{_number(first[1])}) rectangle "
                          f"(axis cs:{_number(second[0])},{_number(second[1])});")
    if _options(plot).get("legend", True):
        output.extend(_legend(series))
    output.append(r"\end{axis}")
    return output


def _areas(plot, categories, series) -> list[str]:
    values = [value for item in series for value in item["values"] if value is not None]
    output = _axis(plot.title, plot.x_label, plot.y_label,
                   xlim=(_number(-.5), _number(len(categories) - .5)),
                   ylim=_bounds(values, zero=True),
                   extra=["xtick={" + ",".join(map(str, range(len(categories)))) + "}",
                          "xticklabels={" + ",".join("{" + _text(value) + "}" for value in categories) + "}",
                          "x tick label style={rotate=35,anchor=east}"], plot=plot)
    for index, item in enumerate(series):
        for segment in _segments(enumerate(item["values"])):
            if len(segment) > 1:
                output.append("% Area closure coordinates are the zero baseline, not observations.")
                closure = [*segment, (segment[-1][0], 0), (segment[0][0], 0)]
                output.append(f"\\addplot[draw=none,fill=oecolor{index},fill opacity=0.18,forget plot{_style(plot)}] coordinates {{"
                              + _coordinates(closure) + r"} \closedcycle;")
            output.append(f"\\addplot[color=oecolor{index},mark=*,mark size=1.5pt,forget plot{_style(plot, point=True, line=True)}] coordinates {{"
                          + _coordinates(segment) + "};")
    if _options(plot).get("legend", True):
        output.extend(_legend(series))
    output.append(r"\end{axis}")
    return output


def _donut(plot, config, categories, series) -> list[str]:
    if len(series) != 1:
        raise ValueError("A donut chart requires exactly one series.")
    values = series[0]["values"]
    if any(value is None or value < 0 for value in values):
        raise ValueError("Donut charts require complete, nonnegative values.")
    total = math.fsum(values)
    _number(total)
    if total <= 0:
        raise ValueError("A donut chart requires a positive total.")
    output = [f"\\node[anchor=south,font=\\bfseries] at (0,2.4) {{{_text(plot.title)}}};"]
    angle = 90.0
    for index, value in enumerate(values):
        end = angle - value / total * 360
        # A zero component has a legend entry but no artificial visible wedge.
        if value > 0:
            start_text, end_text = _number(angle), _number(end)
            output.append(f"\\path[fill=oecolor{index},draw=white{_style(plot)}] ({start_text}:2) "
                          f"arc[start angle={start_text},end angle={end_text},radius=2] "
                          f"-- ({end_text}:1.1) arc[start angle={end_text},end angle={start_text},radius=1.1] -- cycle;")
        angle = end
    output.append(f"\\node[align=center] at (0,0) {{Total\\\\\\texttt{{{_number(total)}}}" +
                  ("\\\\" + _text(config.get("unit", "")) if config.get("unit") else "") + "};")
    for index, (category, value) in enumerate(zip(categories, values)) if _options(plot).get("legend", True) else ():
        column, row = divmod(index, 16)
        x, y = 2.8 + column * 5.2, 2.1 - row * .34
        position = _options(plot).get("legend_position")
        if position is not None:
            x = -2 + column * 5.2
            rows = min(16, len(categories))
            y = 2.8 + (rows - 1 - row) * .34 if position == "top" else -2.7 - row * .34
        output.extend([
            f"\\fill[oecolor{index}] ({_number(x)},{_number(y - .08)}) rectangle "
            f"({_number(x + .18)},{_number(y + .08)});",
            f"\\node[anchor=west,text width=4.6cm,font=\\small] at ({_number(x + .26)},{_number(y)}) "
            f"{{{_text(category)}: \\texttt{{{_number(value)}}}}};",
        ])
    return output


def _donut_picture_options(plot, categories) -> tuple[str, list[str]]:
    """Fit a donut uniformly into explicit dimensions; never stretch wedges."""
    options = _options(plot)
    if not ({"width", "height"} & options.keys()):
        return "", []
    left, right, bottom, top = -2.2, 2.2, -2.2, 2.8
    if options.get("legend", True):
        columns, rows = math.ceil(len(categories) / 16), min(16, len(categories))
        position = options.get("legend_position")
        if position == "top":
            right, top = max(right, 3 + (columns - 1) * 5.2), 3 + (rows - 1) * .34
        elif position == "bottom":
            right, bottom = max(right, 3 + (columns - 1) * 5.2), -2.9 - (rows - 1) * .34
        else:
            right, bottom = 8 + (columns - 1) * 5.2, min(bottom, 1.9 - (rows - 1) * .34)
    natural_width, natural_height = right - left, top - bottom
    width = options.get("width")
    height = options.get("height")
    factors = ([width / 96 * 2.54 / natural_width] if width is not None else [])
    factors += ([height / 96 * 2.54 / natural_height] if height is not None else [])
    scale = min(factors)
    xpad = max(0, width / 96 * 2.54 / scale - natural_width) / 2 if width is not None else 0
    ypad = max(0, height / 96 * 2.54 / scale - natural_height) / 2 if height is not None else 0
    bounds = (f"\\path[use as bounding box] ({_number(left - xpad)},{_number(bottom - ypad)}) rectangle "
              f"({_number(right + xpad)},{_number(top + ypad)});")
    return "[scale=" + _number(scale) + ",transform shape]", [bounds]


def _network_summary(plot, metadata, caption, label, standalone) -> str:
    """Export original and shown counts as a publication summary table."""
    from .network import validate_network
    network = validate_network(plot.config["network"])
    display = {name: network[name] for name in ("directed", "node_count", "edge_count",
               "shown_node_count", "shown_edge_count", "sampled", "selection")}
    display["grouping"] = network.get("grouping", "Weak components")
    coverage = ("The displayed graph is a subset of the original graph."
                if network["sampled"] else "The complete graph is displayed.")
    notes = (f"{'Directed' if network['directed'] else 'Undirected'} graph. {coverage} "
             "Full counts describe the original graph; shown counts describe the display. "
             f"Groups: {display['grouping']}. "
             f"Selection: {network['selection']}.")
    output = ["% OpenEcon chart: " + json.dumps(metadata, ensure_ascii=True),
              "% Network display metadata: " + json.dumps(display, ensure_ascii=True),
              "% Requires booktabs. Unicode text requires XeLaTeX/LuaLaTeX.",
              r"\begin{table}[htbp]", r"\centering",
              f"\\caption{{{_text(plot.title if caption is None else caption)}}}"]
    if label is not None:
        output.append(f"\\label{{{label}}}")
    output.extend([r"\begin{tabular}{lrr}", r"\toprule", r" & Full graph & Shown \\",
                   r"\midrule",
                   f"Nodes & {network['node_count']:,} & {network['shown_node_count']:,} " + r"\\",
                   f"Edges & {network['edge_count']:,} & {network['shown_edge_count']:,} " + r"\\",
                   r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                   r"\begin{minipage}{0.9\linewidth}\footnotesize",
                   r"\emph{Notes:} " + _text(notes), r"\end{minipage}", r"\end{table}"])
    source = "\n".join(output) + "\n"
    if standalone:
        source = ("% Compile with XeLaTeX or LuaLaTeX.\n"
                  "\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n"
                  "\\usepackage{fontspec}\n\\usepackage{booktabs}\n"
                  "\\begin{document}\n" + source + "\\end{document}\n")
    return source


def to_latex(plot, buf=None, *, caption: str | None = None, label: str | None = None,
             standalone: bool = False) -> str:
    """Export a PlotSpec as a TikZ/PGFPlots figure or a complete TeX document.

    ``buf`` may be a UTF-8 file path or a text stream. The source string is
    returned in either case. Caption/title/category text is escaped; labels
    accept only letters, numbers, ``:_.+-``. Standalone documents must be
    compiled with XeLaTeX or LuaLaTeX for Unicode text, including Turkish.
    """
    if not isinstance(standalone, bool):
        raise TypeError("standalone must be a boolean.")
    if caption is not None:
        _text(caption)
    if label is not None and (not isinstance(label, str) or not _LABEL.fullmatch(label)):
        raise ValueError("label must contain only letters, numbers, ':', '_', '.', '+', or '-'.")
    canonical = plot.model_dump()  # Revalidate mutations and normalize default styles.
    if canonical["config"] != plot.config:
        plot = replace(plot, config=canonical["config"])
    presentation = _options(plot)
    metadata = {name: getattr(plot, name) for name in ("kind", "sample_n", "total_n", "dropped_n")}
    if plot.kind == "network":
        source = _network_summary(plot, metadata, caption, label, standalone)
        if buf is not None:
            if hasattr(buf, "write"):
                buf.write(source)
            else:
                Path(buf).expanduser().write_text(source, encoding="utf-8")
        return source
    output = ["% OpenEcon chart: " + json.dumps(metadata, ensure_ascii=True),
              "% Requires tikz and pgfplots (compat=1.18). Unicode labels require XeLaTeX/LuaLaTeX.",
              r"\begin{figure}[htbp]", r"\centering", r"\begingroup", r"\pgfplotsset{compat=1.18}"]
    if presentation.get("annotations"):
        output.append(r"\usetikzlibrary{calc}")
    if any(presentation.get(axis + "_format", "auto") != "auto" and
           (presentation.get(axis + "_scale") == "log" or presentation[axis + "_format"] == "percent")
           for axis in ("x", "y")):
        output.append(r"\usepgflibrary{fpu}")
    if plot.kind == "d3":
        config, categories, series = _categorical_data(plot)
        kind = config.get("type")
        if kind == "donut":
            palette = presentation.get("palette", config.get("palette", []))
            if not isinstance(palette, list) or palette and len(palette) < len(categories):
                raise ValueError("The donut palette must provide one color per category.")
            colors = [_color(presentation.get("color", palette[i] if palette else None), i) for i in range(len(categories))]
            body = _donut(plot, config, categories, series)
        else:
            palette = presentation.get("palette")
            colors = [_color(presentation.get("color", palette[i] if palette else item.get("color")), i)
                      for i, item in enumerate(series)]
            if kind == "area":
                body = _areas(plot, categories, series)
            elif kind in {"bar", "horizontal", "stackedBar", "stacked", "stacked_bar", "stacked-horizontal"}:
                body = _bars(plot, config, categories, series)
            else:
                raise ValueError(f"Unsupported categorical LaTeX chart type: {kind}.")
    else:
        palette = presentation.get("palette", [])
        colors = [_color(presentation.get("color", palette[i] if i < len(palette) else None), i) for i in range(2)]
        if plot.kind in {"scatter", "line"}:
            body = _numeric(plot)
        elif plot.kind == "hist":
            body = _histogram(plot)
        elif plot.kind == "coefficients":
            body = _coefficient_plot(plot)
        else:
            raise ValueError(f"Unsupported LaTeX chart kind: {plot.kind}.")
    output.extend(_define_colors(colors))
    output.extend(f"\\definecolor{{oeann{i}}}{{HTML}}{{{_color(item['color'], i)}}}"
                  for i, item in enumerate(presentation.get("annotations", [])))
    picture_options, bounding = _donut_picture_options(plot, categories) if plot.kind == "d3" and kind == "donut" else ("", [])
    if plot.kind == "d3" and kind == "donut":
        body.extend(_donut_annotations(plot))
    output.extend([r"\begin{tikzpicture}" + picture_options, *bounding, *body, r"\end{tikzpicture}", r"\endgroup"])
    # A supplied label needs a numbered caption for a meaningful \ref target.
    if caption is not None or label is not None:
        output.append(f"\\caption{{{_text(plot.title if caption is None else caption)}}}")
    if label is not None:
        output.append(f"\\label{{{label}}}")
    output.append(r"\end{figure}")
    source = "\n".join(output) + "\n"
    if standalone:
        source = ("% Compile with XeLaTeX or LuaLaTeX.\n"
                  "\\documentclass{article}\n\\usepackage[margin=2cm]{geometry}\n"
                  "\\usepackage{fontspec}\n\\usepackage{tikz}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
                  "\\begin{document}\n" + source + "\\end{document}\n")
    if buf is not None:
        if hasattr(buf, "write"):
            buf.write(source)
        else:
            Path(buf).expanduser().write_text(source, encoding="utf-8")
    return source
