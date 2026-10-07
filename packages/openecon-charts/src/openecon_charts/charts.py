"""Validated chart data, table adapters, and self-contained HTML export."""
from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
import html
from importlib import resources
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any

_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_KINDS = {"scatter", "line", "hist", "coefficients", "d3", "network"}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _json_valid(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False, ensure_ascii=True)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Chart data must contain only finite JSON-compatible values.") from exc
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**53 - 1:
            raise ValueError("Chart integers exceed the browser's exact numeric range. Rescale them or use string labels.")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)


@dataclass(frozen=True)
class PlotSpec:
    kind: str
    title: str
    x_label: str
    y_label: str
    data: list[dict[str, Any]]
    sample_n: int
    total_n: int
    dropped_n: int = 0
    config: dict[str, Any] | None = None

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"Unsupported chart kind: {self.kind}.")
        for name in ("title", "x_label", "y_label"):
            _text(getattr(self, name), name)
        for name in ("sample_n", "total_n", "dropped_n"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.sample_n > self.total_n:
            raise ValueError("sample_n cannot exceed total_n.")
        if not isinstance(self.data, list) or any(not isinstance(item, dict) for item in self.data):
            raise TypeError("data must be a list of dictionaries.")
        if self.kind == "d3" and not isinstance(self.config, dict):
            raise ValueError("A categorical chart requires a config dictionary.")
        if self.config is not None and not isinstance(self.config, dict):
            raise TypeError("config must be a dictionary or None.")
        self._validate_network()
        _json_valid(asdict(self))
        if self.config is not None and "options" in self.config:
            _validated_options(self, self.config["options"])

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Return finite JSON data; compatible with OpenEconometrics's display protocol."""
        if self.kind == "network":
            graph = self._validate_network()
            config = {"network": graph}
            if "options" in self.config:
                config["options"] = _validated_options(self, self.config["options"])
            # The network validator already returns owned, normalized records.
            # Avoid dataclasses.asdict copying a million edge dictionaries again.
            return {"kind": self.kind, "title": self.title, "x_label": "", "y_label": "",
                    "data": [], "sample_n": self.sample_n, "total_n": self.total_n,
                    "dropped_n": 0, "config": config}
        result = asdict(self)
        _json_valid(result)
        if self.config is not None and "options" in self.config:
            result["config"]["options"] = _validated_options(self, self.config["options"])
        return result

    def _validate_network(self):
        if self.kind != "network":
            return
        _text(self.title, "title")
        from .network import validate_network, validate_network_presentation, network_options
        if not isinstance(self.config, dict) or self.data:
            raise ValueError("Network charts require config.network and an empty data list.")
        graph = validate_network(self.config.get("network"))
        if (self.sample_n != graph["shown_node_count"] or self.total_n != graph["node_count"]
                or self.dropped_n != 0 or self.x_label or self.y_label):
            raise ValueError("Network chart envelope does not match its graph metadata.")
        if set(self.config) - {"network", "options"}:
            raise ValueError("Unsupported network chart configuration.")
        validate_network_presentation(graph, network_options(self.config.get("options", {})))
        return graph

    def with_options(self, **options) -> PlotSpec:
        """Return an independent chart with validated presentation options.

        Data, sampling, bin counts and confidence intervals stay unchanged.
        Explicit axis labels and limits refer to physical horizontal/vertical
        axes, including horizontal bars. Limits set a viewport, not a filter.
        """
        self.model_dump()  # Reject invalid mutations before copying.
        options = deepcopy(options)
        title = _text(options.pop("title"), "title") if "title" in options else self.title
        config = deepcopy(self.config)
        if options:
            if config is None:
                config = {}
            merged = {**config.get("options", {}), **options}
            config["options"] = _validated_options(self, merged)
        if config is not None and self.kind == "d3":
            config["title"] = title
        return replace(self, title=title, data=deepcopy(self.data), config=config)

    def _append_annotation(self, annotation: dict) -> PlotSpec:
        current = (self.model_dump()["config"] or {}).get("options", {}).get("annotations", [])
        return self.with_options(annotations=[*current, annotation])

    def annotate(self, text: str, *, x, y, coords: str = "data", arrow: bool = False,
                 dx=None, dy=None, **style) -> PlotSpec:
        """Append escaped plain text or an arrow whose head is at x/y.

        ``axes`` coordinates are fractions from left to right and bottom to
        top. Pixel offsets use screen coordinates: negative dy moves upward.
        """
        if not isinstance(arrow, bool):
            raise TypeError("arrow must be a boolean.")
        if "type" in style:
            raise ValueError("annotate chooses the annotation type; use arrow=True for an arrow.")
        if self.kind == "network":
            if coords != "data" or arrow or dx is not None or dy is not None:
                raise ValueError("Network annotations use layout coordinates; arrows and pixel offsets are not supported.")
            return self._append_annotation({**style, "text": text, "x": x, "y": y})
        item = {**style, "type": "arrow" if arrow else "text", "text": text,
                "x": x, "y": y, "coords": coords}
        if dx is not None:
            item["dx"] = dx
        if dy is not None:
            item["dy"] = dy
        return self._append_annotation(item)

    def annotate_node(self, text: str, *, node: str | int, color: str | None = None) -> PlotSpec:
        """Label a network node by its original, exactly typed identity.

        Integer 1 and string '1' remain different nodes. The label follows the
        node when its layout or saved position changes.
        """
        if self.kind != "network":
            raise ValueError("annotate_node requires a network chart.")
        if isinstance(node, bool) or not isinstance(node, (str, Integral)):
            raise TypeError("node must be a string or integer identity.")
        identity = {"type": "string" if isinstance(node, str) else "integer", "value": str(node)}
        graph = self._validate_network()
        found = next((item["id"] for item in graph["nodes"]
                      if item.get("identity", {"type": "integer", "value": str(item["id"])}) == identity), None)
        if found is None:
            raise ValueError("The node identity is absent from the displayed network.")
        annotation = {"text": text, "node_id": found}
        if color is not None:
            annotation["color"] = color
        return self._append_annotation(annotation)

    def vline(self, x, *, text=None, **style) -> PlotSpec:
        """Append a vertical reference line on a numeric physical x axis."""
        if "type" in style:
            raise ValueError("vline chooses the annotation type.")
        return self._append_annotation({**style, "type": "vline", "x": x,
                                        **({"text": text} if text is not None else {})})

    def hline(self, y, *, text=None, **style) -> PlotSpec:
        """Append a horizontal reference line on a numeric physical y axis."""
        if "type" in style:
            raise ValueError("hline chooses the annotation type.")
        return self._append_annotation({**style, "type": "hline", "y": y,
                                        **({"text": text} if text is not None else {})})

    def vspan(self, x0, x1, *, text=None, **style) -> PlotSpec:
        """Append a shaded interval on a numeric physical x axis."""
        if "type" in style:
            raise ValueError("vspan chooses the annotation type.")
        return self._append_annotation({**style, "type": "vspan", "x0": x0, "x1": x1,
                                        **({"text": text} if text is not None else {})})

    def hspan(self, y0, y1, *, text=None, **style) -> PlotSpec:
        """Append a shaded interval on a numeric physical y axis."""
        if "type" in style:
            raise ValueError("hspan chooses the annotation type.")
        return self._append_annotation({**style, "type": "hspan", "y0": y0, "y1": y1,
                                        **({"text": text} if text is not None else {})})

    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False, indent=indent)

    def save_view(self, path: str | Path) -> Path:
        """Save a validated network PlotSpec, including positions and code options.

        Browser-exported JSON can be reopened by ``PlotSpec.load_view``. The
        saved view contains only the explicit display graph, not omitted nodes.
        Calling this method does not execute a browser force-layout job.
        """
        if self.kind != "network":
            raise ValueError("save_view requires a network chart.")
        value = json.dumps(self.transport_dump(), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        from .network import MAX_PAYLOAD_BYTES
        if len(value.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise ValueError("The saved network view exceeds its payload budget.")
        target = Path(path)
        target.write_text(value + "\n", encoding="utf-8")
        return target

    @classmethod
    def load_view(cls, path: str | Path) -> PlotSpec:
        """Reopen network view JSON without running code or reading other files."""
        from .network import MAX_PAYLOAD_BYTES
        target = Path(path)
        if target.stat().st_size > MAX_PAYLOAD_BYTES + 1:
            raise ValueError("The saved network view exceeds its payload budget.")
        try:
            value = json.loads(target.read_text(encoding="utf-8"),
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON.")))
            from .timeline import unpack
            plot = cls(**unpack(value))
        except (TypeError, KeyError, RecursionError) as exc:
            raise ValueError("The saved network view is invalid.") from exc
        if plot.kind != "network":
            raise ValueError("load_view requires a network chart.")
        return plot

    def to_latex(self, buf=None, *, caption: str | None = None, label: str | None = None,
                 standalone: bool = False) -> str:
        """Return a data-faithful figure or network summary; optionally save source.

        Fragments need ``tikz`` and ``pgfplots`` packages. Standalone documents
        declare their packages and use XeLaTeX/LuaLaTeX for Unicode labels.
        Network charts export a booktabs table of full and displayed counts;
        their force-layout drawing is available through SVG or PNG export.
        """
        from .latex import to_latex
        return to_latex(self, buf, caption=caption, label=label, standalone=standalone)

    @property
    def latex(self) -> str:
        """The figure's escaped LaTeX source, equivalent to ``to_latex()``."""
        return self.to_latex()

    def to_html(self, *, height: int = 480) -> str:
        """Return a complete HTML document with embedded renderer, D3, and font.

        No server, CDN, or Python runtime is needed to reopen this document.
        Height sets a minimum chart workspace height; the renderer may expand
        for labels. Network requests are blocked with a Content Security Policy.
        """
        if isinstance(height, bool) or not isinstance(height, int) or not 240 <= height <= 4000:
            raise ValueError("height must be an integer between 240 and 4000 pixels.")
        assets = resources.files("openecon_charts").joinpath("assets")
        try:
            d3 = assets.joinpath("d3.min.js").read_text(encoding="utf-8")
            renderer = assets.joinpath("renderer.js").read_text(encoding="utf-8")
            stylesheet = assets.joinpath("charts.css").read_text(encoding="utf-8")
            network_renderer = assets.joinpath("network-renderer.js").read_text(encoding="utf-8") if self.kind == "network" else ""
            network_helpers = ""
            if self.kind == "network":
                stylesheet += "\n" + assets.joinpath("network.css").read_text(encoding="utf-8")
                network_helpers = "\n".join(assets.joinpath(name).read_text(encoding="utf-8")
                                             for name in ("network-webgl.js", "network-font.js", "network-pdf.js"))
                source = d3 + "\n" + assets.joinpath("network-worker.js").read_text(encoding="utf-8")
                network_helpers += "\nwindow.OpenEconNetworkWorkerSource=" + json.dumps(source, ensure_ascii=True) + ";"
            font = base64.b64encode(assets.joinpath("Barlow.woff2").read_bytes()).decode("ascii")
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise RuntimeError("The chart renderer assets are missing from this installation.") from exc
        # Replace the vendored font reference as well as setting a portable face.
        font_url = f"data:font/woff2;base64,{font}"
        stylesheet, font_references = re.subn(r"url\(\s*(['\"]?)[^)]*Barlow\.woff2\1\s*\)",
                                              lambda _: f"url('{font_url}')", stylesheet, flags=re.IGNORECASE)
        if not font_references:
            stylesheet = (f"@font-face{{font-family:Barlow;src:url('{font_url}') format('woff2');font-display:swap;}}\n"
                          + stylesheet)
        payload = json.dumps(self.transport_dump(), ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        # Trusted vendored JavaScript can still contain strings with HTML end tags.
        def safe_js(source):
            return re.sub(r"</script", r"<\\/script", source, flags=re.IGNORECASE)
        safe_css = re.sub(r"</style", r"<\\/style", stylesheet, flags=re.IGNORECASE)
        policy = "default-src 'none'; script-src 'unsafe-inline'; worker-src blob:; style-src 'unsafe-inline'; font-src data:; img-src data: blob:; connect-src 'none'; base-uri 'none'; form-action 'none'"
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="Content-Security-Policy" content="{html.escape(policy, quote=True)}">'
            f'<title>{html.escape(self.title)}</title><style>{safe_css}\n'
            f'html,body{{margin:0;background:#fff;font-family:Barlow,sans-serif}}#chart{{min-height:{height}px;width:100%;min-width:0;}}'
             '</style></head><body><main id="chart" aria-label="Chart"></main>'
            f'<script type="application/json" id="chart-data">{payload}</script>'
            f'<script>{safe_js(d3)}</script><script>{safe_js(network_helpers)}</script><script>{safe_js(network_renderer)}</script><script>{safe_js(renderer)}</script>'
            '<script>const chartHost=document.getElementById("chart");'
            'Promise.resolve().then(()=>OpenEconCharts.mount(chartHost,JSON.parse(document.getElementById("chart-data").textContent)))'
            '.catch(error=>{chartHost.setAttribute("role","alert");'
            'chartHost.textContent=String(error?.message||error||"The chart could not be rendered.");});</script>'
            '</body></html>'
        )

    def save_html(self, path: str | Path, *, height: int = 480) -> Path:
        """Save an offline HTML document to an explicitly chosen local path."""
        destination = Path(path).expanduser()
        destination.write_text(self.to_html(height=height), encoding="utf-8")
        return destination

    def transport_dump(self) -> dict:
        """Return checked lossless transport data, interning repeated timeline records.

        Aggregate decoded entry and workspace limits remain unchanged. Consumers
        must support timeline-pool-v1; use model_dump for the expanded protocol.
        """
        from .timeline import pack
        return pack(self.model_dump()) if self.kind == "network" else self.model_dump()

    def to_pdf(self, path: str | Path, **options) -> Path:
        """Export a vector network PDF in an owned offline Chromium session.

        See ``to_png`` for viewport, timeout, browser and overwrite options.
        Barlow must contain every displayed glyph. Vector exports refuse more
        than 200,000 displayed node/edge primitives; they never rasterize.
        """
        from .network_export import export_network
        return export_network(self, path, "pdf", **options)

    def to_svg(self, path: str | Path, **options) -> Path:
        """Export a vector network SVG with embedded Barlow and saved camera.

        See ``to_png`` for common options. Vector output never falls back to PNG.
        """
        from .network_export import export_network
        return export_network(self, path, "svg", **options)

    def to_png(self, path: str | Path, **options) -> Path:
        """Export a network PNG after font loading and layout completion.

        Options: ``width=960`` (320..2400 CSS pixels), ``height=600`` (240..1600
        network workspace pixels), ``scale=1`` (0.5..2, PNG only), ``timeout=60``
        (1..300 seconds), ``overwrite=False``, ``browser_executable=None``
        (installed Chrome, Chromium or Edge). No browser is downloaded and no
        user browser profile is opened. PNG is limited to 16 million pixels;
        output to 128 MiB. Layout/seed/filter/view choices come from PlotSpec.
        """
        from .network_export import export_network
        return export_network(self, path, "png", **options)

    def _repr_html_(self) -> str:
        """Use an opaque-origin iframe to isolate notebook output from its host."""
        document = html.escape(self.to_html(), quote=True)
        height = max(500, (self.config or {}).get("options", {}).get("height", 0) + 180)
        return f'<iframe title="{html.escape(self.title, quote=True)}" sandbox="allow-scripts allow-downloads" style="width:100%;height:{height}px;border:0" srcdoc="{document}"></iframe>'


def _columns(data: Any, names: Sequence[str]) -> dict[str, list]:
    """Accept column mappings, row mappings, or dataframe-style to_dict objects."""
    if not isinstance(data, Mapping) and hasattr(data, "to_dict"):
        if hasattr(data, "columns"):
            absent = [name for name in names if name not in data.columns]
            if absent:
                raise ValueError(f"Column '{absent[0]}' was not found.")
            column_names = list(data.columns)
            ambiguous = [name for name in dict.fromkeys(names) if column_names.count(name) > 1]
            if ambiguous:
                raise ValueError(f"Column '{ambiguous[0]}' is duplicated. Requested chart columns must be unique.")
            data = data[list(dict.fromkeys(names))]
        try:
            data = data.to_dict(orient="list")
        except TypeError:
            data = data.to_dict()
    if isinstance(data, Mapping):
        result = {}
        for name in names:
            if name not in data:
                raise ValueError(f"Column '{name}' was not found.")
            column = data[name]
            if isinstance(column, (str, bytes, Mapping)):
                raise TypeError(f"Column '{name}' must be a sequence of values.")
            try:
                result[name] = list(column)
            except TypeError as exc:
                raise TypeError(f"Column '{name}' must be a sequence of values.") from exc
        if len({len(column) for column in result.values()}) > 1:
            raise ValueError("Chart columns must have equal lengths.")
        return result
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be a column mapping, records, or a dataframe-style object.")
    try:
        records = list(data)
    except TypeError as exc:
        raise TypeError("data must be a column mapping, records, or a dataframe-style object.") from exc
    if any(not isinstance(row, Mapping) for row in records):
        raise TypeError("Rows must be mappings from column names to values.")
    for name in names:
        if records and not any(name in row for row in records):
            raise ValueError(f"Column '{name}' was not found.")
    return {name: [row.get(name) for row in records] for name in names}


def _number(value: Any, name: str) -> float | None:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, Integral) and abs(value) > 2**53 - 1:
        raise ValueError(f"Column '{name}' contains integers outside the browser's exact numeric range. Rescale them or use category labels.")
    if isinstance(value, (str, Decimal)):
        try:
            exact = Decimal(value)
            if exact.is_finite() and exact == exact.to_integral_value() and abs(exact) > 2**53 - 1:
                raise ValueError(f"Column '{name}' contains integers outside the browser's exact numeric range. Rescale them or use category labels.")
        except InvalidOperation:
            pass
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Column '{name}' must contain numeric or missing values.") from exc
    return number if math.isfinite(number) else None


def _title(value: str | None, default: str) -> str:
    return default if value is None else _text(value, "title")


def _check_extent(values, name: str, *, include_zero: bool = False) -> None:
    finite = [value for value in values if value is not None]
    if not finite:
        return
    lower, upper = min(finite), max(finite)
    if include_zero:
        lower, upper = min(0.0, lower), max(0.0, upper)
    if not math.isfinite(upper - lower):
        raise ValueError(f"The '{name}' axis range exceeds finite numeric precision. Rescale the values explicitly.")


def _finite_sum(values) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("Component totals exceed finite numeric precision. Rescale the values explicitly.") from exc
    if not math.isfinite(result):
        raise ValueError("Component totals exceed finite numeric precision. Rescale the values explicitly.")
    return result


def _processing_config(data, metadata):
    from .streaming import is_source
    # Keep small resident plots' existing wire format. Larger resident inputs
    # and Dataset sources retain full-sample processing and extent metadata.
    return {"processing": metadata} if is_source(data) or metadata["source_rows"] > 2000 else None


def scatter(*, data, x: str, y: str, title: str | None = None, **options) -> PlotSpec:
    """Plot finite numeric pairs, sampling at most 2,000 rows deterministically."""
    from . import resident, streaming
    points, count, dropped, metadata = streaming.scatter(resident.source(data, [x, y]), x, y)
    return _styled(PlotSpec("scatter", _title(title, f"{y} · {x}"), x, y,
        points, len(points), count, dropped, _processing_config(data, metadata)), options)


def line(*, data, x: str, y: str, title: str | None = None, **options) -> PlotSpec:
    """Keep numeric x distances and missing-value gaps; do not downsample lines.

    Finite-x runs are sorted by x. A row with missing x separates runs; a row
    with missing y remains at its x coordinate and breaks the rendered path.
    """
    from . import resident, streaming
    columns, metadata = streaming.bounded_columns(resident.source(data, [x, y], limit=10000), [x, y], 10000)
    points, run = [], []
    count = 0
    for a, b in zip(columns[x], columns[y]):
        xx, yy = _number(a, x), _number(b, y)
        if xx is None:
            points.extend(sorted(run, key=lambda item: item["x"]))
            run.clear()
            points.append({"x": None, "y": None})
        else:
            run.append({"x": xx, "y": yy})
            count += yy is not None
    points.extend(sorted(run, key=lambda item: item["x"]))
    if not count:
        raise ValueError("The line plot needs at least one finite numeric pair.")
    _check_extent((point["x"] for point in points), x)
    _check_extent((point["y"] for point in points), y)
    return _styled(PlotSpec("line", _title(title, f"{y} · {x}"), x, y, points,
                           count, count, len(columns[x]) - count, _processing_config(data, metadata)), options)


def hist(*, data, x: str, bins: int = 20, title: str | None = None, **options) -> PlotSpec:
    """Count every finite value in equal-width bins; the final edge is inclusive."""
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 200:
        raise ValueError("bins must be an integer between 1 and 200.")
    from . import resident, streaming
    points, count, dropped, metadata = streaming.histogram(resident.source(data, [x]), x, bins)
    return _styled(PlotSpec("hist", _title(title, f"{x} · histogram"), x, "Count",
        points, count, count, dropped, _processing_config(data, metadata)), options)


def coefficients(model, *, title: str | None = None, **options) -> PlotSpec:
    """Read a model's coefficients by protocol, without importing its framework.

    A model exposes ``coefficients`` or a mapping with that key. Each entry
    exposes term, estimate, ci_low, and ci_high as attributes or mapping keys.
    """
    from . import streaming
    if streaming.is_source(model):
        names = ["term", "estimate", "ci_low", "ci_high"]
        columns, metadata = streaming.bounded_columns(model, names, 1000)
        records = [dict(zip(names, row)) for row in zip(*(columns[name] for name in names))]
        plot = coefficients({"coefficients": records}, title=title, **options)
        return replace(plot, config={**(plot.config or {}), "processing": metadata})
    values = model.get("coefficients") if isinstance(model, Mapping) else getattr(model, "coefficients", None)
    if values is None:
        raise TypeError("The model must expose coefficient records.")
    points = []
    for value in values:
        def field(name):
            return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        term = field("term")
        if not isinstance(term, str) or not term:
            raise ValueError("Each coefficient must have a nonempty term name.")
        numbers = {name: _number(field(name), name) for name in ("estimate", "ci_low", "ci_high")}
        if any(number is None for number in numbers.values()):
            raise ValueError("Coefficient estimates and confidence limits must be finite.")
        if numbers["ci_low"] > numbers["ci_high"]:
            raise ValueError("Confidence interval lower limits cannot exceed upper limits.")
        points.append({"term": term, **numbers})
    if not points:
        raise ValueError("The model has no coefficients to plot.")
    if len({point["term"] for point in points}) != len(points):
        raise ValueError("Coefficient term names must be unique.")
    if len(points) > 1000:
        raise ValueError("Coefficient charts support at most 1,000 terms. Select terms explicitly.")
    _check_extent((point[name] for point in points for name in ("estimate", "ci_low", "ci_high")), "coefficients")
    return _styled(PlotSpec("coefficients", _title(title, "Coefficient estimates"), "Estimate", "Term",
                           points, len(points), len(points)), options)


def _palette(palette, needed: int) -> list[str] | None:
    if palette is None:
        return None
    if isinstance(palette, (str, bytes)) or not isinstance(palette, Sequence):
        raise ValueError("palette must be a sequence of CSS hex colors, for example ['#264653'].")
    colors = list(palette)
    if len(colors) < needed or any(not isinstance(color, str) or not _HEX_COLOR.fullmatch(color) for color in colors):
        raise ValueError(f"palette must provide at least {needed} valid CSS hex colors.")
    return ["#" + "".join(character * 2 for character in color[1:]) if len(color) == 4 else color
            for color in colors]


def _categories(values: list) -> list[str]:
    result = []
    for value in values:
        if value is None or type(value).__name__ in {"NAType", "NaTType"}:
            raise ValueError("Category labels cannot be missing.")
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError("Category labels cannot be non-finite.")
        label = str(value)
        if not label.strip():
            raise ValueError("Category labels cannot be empty.")
        result.append(label)
    if len(set(result)) != len(result):
        raise ValueError("Category labels must be unique. Aggregate repeated categories explicitly before plotting.")
    if not result:
        raise ValueError("The chart needs at least one category.")
    return result


def _categorical(kind: str, *, data, x: str, y: str | Sequence[str], title: str | None,
                 unit: str, palette, compositional: bool = False, aggregate=None) -> PlotSpec:
    names = [y] if isinstance(y, str) else list(y)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("y must be one column name or a sequence of unique column names.")
    if x in names:
        raise ValueError("Use distinct category and value columns.")
    if len(names) > 24:
        raise ValueError("Categorical charts support at most 24 series. Select series explicitly.")
    from . import streaming
    if streaming.is_source(data):
        columns, source_finite, source_missing, metadata = streaming.categorical(data, x, names, aggregate)
        plot = _categorical(kind, data=columns, x=x, y=names, title=title, unit=unit,
                            palette=palette, compositional=compositional)
        return replace(plot, total_n=source_finite if source_finite is not None else plot.total_n,
                       dropped_n=source_missing if source_missing is not None else plot.dropped_n,
                       config={**plot.config, "processing": metadata})
    if aggregate is not None:
        # Explicit reduction is available for dense inputs via the same protocol.
        columns = _columns(data, [x, *names])
        class Source:
            def __init__(self):
                self.columns = list(columns)
            def iter_batches(self, *, columns=None, batch_rows=65536):
                for start in range(0, len(next(iter(columns_data.values()))), batch_rows):
                    yield {name: columns_data[name][start:start + batch_rows] for name in columns}
        columns_data = columns
        return _categorical(kind, data=Source(), x=x, y=names, title=title, unit=unit,
                            palette=palette, compositional=compositional, aggregate=aggregate)
    columns = _columns(data, [x, *names])
    if len(columns[x]) > 1000 or len(columns[x]) * len(names) > 10000:
        raise ValueError("Categorical charts support at most 1,000 categories and 10,000 values. Filter or aggregate explicitly.")
    categories = _categories(columns[x])
    colors = _palette(palette, len(names))
    series = []
    finite = 0
    for index, name in enumerate(names):
        values = [_number(value, name) for value in columns[name]]
        if compositional and any(value is None or value < 0 for value in values):
            raise ValueError("Compositional charts require complete, finite, nonnegative values.")
        finite += sum(value is not None for value in values)
        item = {"id": name, "name": name, "values": values}
        if colors:
            item["color"] = colors[index]
        series.append(item)
    if not finite:
        raise ValueError("The chart needs at least one finite value.")
    _check_extent((value for item in series for value in item["values"]), "values", include_zero=True)
    if compositional and kind != "donut":
        totals = [_finite_sum(item["values"][index] for item in series) for index in range(len(categories))]
        _check_extent(totals, "component totals", include_zero=True)
    chart_title = _title(title, " · ".join(names))
    config = {"type": kind, "categories": categories, "series": series,
              "compositional": compositional, "title": chart_title, "xLabel": x,
              "yLabel": names[0] if len(names) == 1 else _text(unit, "unit"), "unit": _text(unit, "unit")}
    if colors:
        config["palette"] = colors
    total = len(categories) * len(names)
    return PlotSpec("d3", chart_title, x, config["yLabel"], [], finite, finite, total - finite, config)


def bar(*, data, x: str, y: str | Sequence[str], title: str | None = None,
        unit: str = "", palette=None, aggregate: str | None = None, **options) -> PlotSpec:
    """Plot signed series; aggregate=None preserves explicitly supplied rows."""
    return _styled(_categorical("bar", data=data, x=x, y=y, title=title, unit=unit, palette=palette, aggregate=aggregate), options)


def barh(*, data, x: str, y: str | Sequence[str], title: str | None = None,
         unit: str = "", palette=None, aggregate: str | None = None, **options) -> PlotSpec:
    """Plot horizontal bars with optional explicit sum/mean/count aggregation."""
    return _styled(_categorical("horizontal", data=data, x=x, y=y, title=title, unit=unit, palette=palette, aggregate=aggregate), options)


def area(*, data, x: str, y: str | Sequence[str], title: str | None = None,
         unit: str = "", palette=None, aggregate: str | None = None, **options) -> PlotSpec:
    """Plot signed categorical area series; all-missing groups remain gaps."""
    return _styled(_categorical("area", data=data, x=x, y=y, title=title, unit=unit, palette=palette, aggregate=aggregate), options)


def stacked_bar(*, data, x: str, y: str | Sequence[str], title: str | None = None,
                unit: str = "", palette=None, horizontal: bool = False,
                aggregate: str | None = None, **options) -> PlotSpec:
    """Stack explicitly provided, complete nonnegative components by category."""
    if not isinstance(horizontal, bool):
        raise TypeError("horizontal must be a boolean.")
    return _styled(_categorical("stacked-horizontal" if horizontal else "stackedBar", data=data, x=x, y=y,
                               title=title, unit=unit, palette=palette, compositional=True, aggregate=aggregate), options)


def donut(*, data, labels: str, values: str, title: str | None = None,
          unit: str = "", palette=None, aggregate: str | None = None, **options) -> PlotSpec:
    """Display a single nonnegative composition with an explicitly positive total."""
    plot = _categorical("donut", data=data, x=labels, y=values, title=title,
                        unit=unit, palette=None, compositional=True, aggregate=aggregate)
    if _finite_sum(plot.config["series"][0]["values"]) <= 0:
        raise ValueError("A donut composition needs a positive total.")
    colors = _palette(palette, len(plot.config["categories"]))
    if colors:
        plot.config["palette"] = colors
    return _styled(plot, options)


_OPTION_KEYS = {
    "width", "height", "color", "palette", "opacity", "point_size", "line_width",
    "grid", "legend", "legend_position", "x_label", "y_label", "xlim", "ylim",
    "x_scale", "y_scale", "x_format", "y_format", "annotations",
}
_TICK_FORMATS = {"auto", "number", "integer", "percent", "scientific"}


def _chart_type(plot: PlotSpec) -> str:
    if plot.kind != "d3":
        return plot.kind
    kind = plot.config.get("type", "bar")
    return {"stackedBar": "stacked", "stacked_bar": "stacked",
            "stackedArea": "stacked-area", "horizontalBar": "horizontal"}.get(kind, kind)


def _numeric_axes(plot: PlotSpec) -> set[str]:
    kind = _chart_type(plot)
    if plot.kind in {"scatter", "line", "hist"}:
        return {"x", "y"}
    if plot.kind == "coefficients" or kind in {"horizontal", "stacked-horizontal"}:
        return {"x"}
    if kind in {"bar", "area", "line", "stacked", "stacked-area"}:
        return {"y"}
    return set()


def _axis_values(plot: PlotSpec, axis: str) -> list:
    """Stored numeric-axis values; no filtering, fitting or transformations."""
    if plot.kind in {"scatter", "line"}:
        return [row[axis] for row in plot.data if row[axis] is not None]
    if plot.kind == "coefficients":
        return [0, *(row[name] for row in plot.data for name in ("estimate", "ci_low", "ci_high"))]
    if plot.kind == "hist":
        return ([row[name] for row in plot.data for name in ("x0", "x1")] if axis == "x"
                else [0, *(row["count"] for row in plot.data)])
    return [0, *(value for series in plot.config["series"] for value in series["values"] if value is not None)]


def _option_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    result = _number(value, name)
    if result is None:
        raise ValueError(f"{name} must be a finite number.")
    return result


def _validated_options(plot: PlotSpec, options) -> dict:
    if plot.kind == "network":
        from .network import network_options
        return network_options(options)
    if not isinstance(options, dict):
        raise TypeError("Chart options must be a dictionary.")
    unknown = set(options) - _OPTION_KEYS
    if unknown:
        raise TypeError(f"Unknown chart option: {sorted(unknown, key=str)[0]}.")
    result = deepcopy(options)
    kind = _chart_type(plot)
    axes = _numeric_axes(plot)
    for name, lower, upper in (("width", 320, 2400), ("height", 240, 1600)):
        if name in result:
            value = result[name]
            if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper} pixels.")
            result[name] = int(value)
    for name, lower, upper in (("opacity", 0, 1), ("point_size", 1, 24), ("line_width", .25, 12)):
        if name in result:
            value = _option_number(result[name], name)
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}.")
            result[name] = value
    point_marks = (plot.kind in {"scatter", "line", "coefficients"} or
                   plot.kind == "d3" and kind in {"area", "line", "stacked-area"})
    line_marks = (plot.kind in {"line", "coefficients"} or
                  plot.kind == "d3" and kind in {"area", "line", "stacked-area"})
    if "point_size" in result and not point_marks:
        raise ValueError(f"point_size is not supported for {kind} charts.")
    if "line_width" in result and not line_marks:
        raise ValueError(f"line_width is not supported for {kind} charts.")
    if "color" in result:
        value = result["color"]
        if not isinstance(value, str) or not _HEX_COLOR.fullmatch(value):
            raise ValueError("color must be an opaque CSS hex color.")
        result["color"] = _palette([value], 1)[0]
    if "palette" in result:
        needed = (len(plot.config["categories"]) if kind == "donut"
                  else len(plot.config["series"]) if plot.kind == "d3"
                  else 2 if kind == "coefficients" else 1)
        if result["palette"] is None:
            raise ValueError("palette must be a sequence of CSS hex colors.")
        result["palette"] = _palette(result["palette"], needed)
    for name in ("grid", "legend"):
        if name in result and not isinstance(result[name], bool):
            raise TypeError(f"{name} must be a boolean.")
    if "grid" in result and kind in {"donut", "gauge"}:
        raise ValueError(f"grid is not supported for {kind} charts.")
    if "legend_position" in result and (not isinstance(result["legend_position"], str) or
                                        result["legend_position"] not in {"top", "bottom"}):
        raise ValueError("legend_position must be 'top' or 'bottom'.")
    if plot.kind != "d3" and ({"legend", "legend_position"} & result.keys()):
        raise ValueError(f"{kind} charts do not have a named-series legend.")
    for axis in ("x", "y"):
        label, limit, scale, formatting = f"{axis}_label", f"{axis}lim", f"{axis}_scale", f"{axis}_format"
        if label in result:
            if kind in {"donut", "gauge"}:
                raise ValueError(f"Axis labels are not supported for {kind} charts.")
            result[label] = _text(result[label], label)
        if axis not in axes and any(name in result for name in (limit, scale, formatting)):
            raise ValueError(f"The {axis} axis is categorical or absent; numeric {axis}-axis options are unsupported.")
        if limit in result:
            values = result[limit]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 2:
                raise ValueError(f"{limit} must contain two finite increasing numbers.")
            values = [_option_number(value, limit) for value in values]
            if values[0] >= values[1] or not math.isfinite(values[1] - values[0]):
                raise ValueError(f"{limit} must contain two finite increasing numbers with a finite span.")
            baseline = (kind == "coefficients" or plot.kind == "d3" or kind == "hist" and axis == "y")
            if baseline and not values[0] <= 0 <= values[1]:
                raise ValueError(f"{limit} must include the zero baseline.")
            result[limit] = values
        if scale in result:
            if not isinstance(result[scale], str) or result[scale] not in {"linear", "log"}:
                raise ValueError(f"{scale} must be 'linear' or 'log'.")
            if result[scale] == "log":
                if plot.kind not in {"scatter", "line"}:
                    raise ValueError("Log scales are supported only for numeric scatter and line charts.")
                values = _axis_values(plot, axis)
                source_extent = (plot.config or {}).get("processing", {}).get("extents", {}).get(axis)
                if source_extent is not None:
                    values = [*values, *source_extent]
                if not values or any(_option_number(value, scale) <= 0 for value in values):
                    raise ValueError(f"{scale} requires strictly positive stored {axis} values; data is never omitted.")
                if limit in result and result[limit][0] <= 0:
                    raise ValueError(f"{limit} must be strictly positive for a log scale.")
        if formatting in result and (not isinstance(result[formatting], str) or result[formatting] not in _TICK_FORMATS):
            raise ValueError(f"{formatting} must be auto, number, integer, percent, or scientific.")
    if "annotations" in result:
        from .annotations import normalize_annotations
        result["annotations"] = normalize_annotations(plot, result["annotations"], result)
    return result


def _styled(plot: PlotSpec, options: dict) -> PlotSpec:
    return plot.with_options(**options) if options else plot
