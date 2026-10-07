"""Bounded network chart specifications through a dependency-free protocol.

Graph algorithms stay in the caller's library. This module validates only an
explicit display subset, its full-graph counts and safe presentation options.
It never imports Torch, OpenEconometrics or a graph-analysis dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Integral, Real
import re
from typing import Any


MAX_NODES = 100_000
MAX_EDGES = 1_000_000
MAX_LABEL_BYTES = 4_096
MAX_TOTAL_LABEL_BYTES = 16 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
_MAX_INTEGER = 2**53 - 1
_HEX = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\Z")
_GRAPH_KEYS = frozenset({"nodes", "edges", "directed", "node_count", "edge_count",
                         "shown_node_count", "shown_edge_count", "sampled", "selection"})
_GRAPH_OPTIONAL_KEYS = frozenset({"grouping", "frames"})
_NODE_KEYS = frozenset({"id", "label", "degree", "group"})
_EDGE_KEYS = frozenset({"source", "target", "weight"})
_NODE_OPTIONAL_KEYS = frozenset({"identity", "attrs", "x", "y", "fx", "fy", "fixed", "pinned", "longitude", "latitude"})
_EDGE_OPTIONAL_KEYS = frozenset({"attrs"})
_OPTION_KEYS = frozenset({"width", "height", "color", "palette", "point_size", "line_width", "opacity", "layout", "layout_options", "node_size", "node_color", "node_label", "edge_width", "edge_color", "filters", "annotations", "labels", "legend", "view", "timeline", "frame_index", "seed"})
_LAYOUTS = frozenset({"d3-force", "forceatlas2", "circular", "grid", "radial", "hierarchical", "geographic", "community", "fixed"})


def _integer(value: Any, name: str, maximum: int = _MAX_INTEGER, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return int(value)


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    if isinstance(value, Integral) and abs(value) > _MAX_INTEGER:
        raise ValueError(f"{name} exceeds the browser's exact integer range.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result) or nonnegative and result < 0:
        suffix = " nonnegative" if nonnegative else ""
        raise ValueError(f"{name} must be a finite{suffix} number.")
    return result


def _text(value: Any, name: str, maximum: int, *, nonempty: bool = False) -> str:
    # Check length before encoding an untrusted, potentially enormous string.
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if len(value) > maximum or nonempty and not value:
        raise ValueError(f"{name} must contain {'1 to ' if nonempty else 'at most '}{maximum} UTF-8 bytes.")
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        raise ValueError(f"{name} cannot contain control characters.")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode text.") from exc
    if size > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-8 bytes.")
    return value


def _keys(value: Any, expected: frozenset[str], name: str, *,
          optional: frozenset[str] = frozenset()) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a dictionary.")
    if len(value) > len(expected) + len(optional):
        raise ValueError(f"{name} contains unsupported fields.")
    actual = set(value)
    if not expected <= actual or not actual <= expected | optional:
        suffix = f"; optional: {', '.join(sorted(optional))}" if optional else ""
        raise ValueError(f"{name} must contain: {', '.join(sorted(expected))}{suffix}.")
    return value


def _scalar(value: Any, name: str):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        if abs(value) > _MAX_INTEGER:
            raise ValueError(f"{name} exceeds the browser's exact integer range; use a string explicitly.")
        return int(value)
    if isinstance(value, Real):
        return _finite(value, name)
    return _text(value, name, MAX_LABEL_BYTES)


def _attrs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(value) > 64:
        raise ValueError("Display attributes must be a dictionary with at most 64 scalar keys.")
    result = {}
    for key, item in value.items():
        key = _text(key, "attribute name", 200, nonempty=True)
        if key in {"__proto__", "prototype", "constructor"}:
            raise ValueError("Prototype names cannot be display attributes.")
        result[key] = _scalar(item, "attribute value")
    return result


def _identity(value: Any) -> dict[str, str]:
    item = _keys(value, frozenset({"type", "value"}), "Node identity")
    kind, label = item["type"], _text(item["value"], "identity value", MAX_LABEL_BYTES, nonempty=True)
    if kind == "integer":
        if not re.fullmatch(r"0|-?[1-9][0-9]{0,77}", label) or int(label).bit_length() > 256:
            raise ValueError("Integer identities must be canonical decimal integers of at most 256 bits.")
    elif kind != "string":
        raise ValueError("Identity type must be integer or string.")
    elif not label.strip():
        raise ValueError("String identities cannot consist only of whitespace.")
    return {"type": kind, "value": label}


def validate_network(value: Any, *, _nested: bool = False, _budget=None) -> dict[str, Any]:
    """Return an independent, finite network payload with consistent counts.

    The display is limited to 100,000 nodes and 1,000,000 edges. Every edge endpoint
    must be present, IDs must be unique browser-safe integers, and ``sampled``
    must describe whether any original nodes or edges were omitted. Full graph
    degree values are preserved; they need not equal degree in a display subset.
    """
    graph = _keys(value, _GRAPH_KEYS, "Network data", optional=_GRAPH_OPTIONAL_KEYS)
    if _nested and "frames" in graph:
        raise ValueError("Timeline frames cannot recursively contain frames.")
    budget = [0, 0, 0, 0] if _budget is None else _budget
    for name in ("directed", "sampled"):
        if not isinstance(graph[name], bool):
            raise TypeError(f"{name} must be a boolean.")
    nodes, edges = graph["nodes"], graph["edges"]
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TypeError("Network nodes and edges must be lists.")
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise ValueError(f"A network chart supports at most {MAX_NODES:,} nodes and {MAX_EDGES:,} edges.")
    budget[0] += len(nodes)
    budget[1] += len(edges)
    budget[2] += 192 * len(nodes) + 96 * len(edges) + 4096
    if budget[0] > MAX_NODES or budget[1] > MAX_EDGES:
        raise ValueError("Timeline base and frames together exceed the aggregate display entry budget.")
    counts = {name: _integer(graph[name], name) for name in
              ("node_count", "edge_count", "shown_node_count", "shown_edge_count")}
    if counts["shown_node_count"] != len(nodes) or counts["shown_edge_count"] != len(edges):
        raise ValueError("Shown network counts must match the stored nodes and edges.")
    if counts["node_count"] < len(nodes) or counts["edge_count"] < len(edges):
        raise ValueError("Full network counts cannot be smaller than the displayed subset.")
    sampled = counts["node_count"] != len(nodes) or counts["edge_count"] != len(edges)
    if graph["sampled"] != sampled:
        raise ValueError("sampled must agree with the full and shown network counts.")
    if counts["node_count"] == 0 and counts["edge_count"]:
        raise ValueError("A graph without nodes cannot contain edges.")
    selection = _text(graph["selection"], "selection", 1_000, nonempty=True)
    grouping = _text(graph["grouping"], "grouping", 1_000, nonempty=True) if "grouping" in graph else None
    normalized_nodes, ids, identities = [], set(), set()
    for item in nodes:
        node = _keys(item, _NODE_KEYS, "A network node", optional=_NODE_OPTIONAL_KEYS)
        node_id = _integer(node["id"], "node id")
        if node_id in ids:
            raise ValueError("Network node IDs must be unique.")
        ids.add(node_id)
        label = _text(node["label"], "node label", MAX_LABEL_BYTES)
        label_bytes = len(label.encode("utf-8"))
        budget[3] += label_bytes
        budget[2] += 2 * label_bytes
        if budget[3] > MAX_TOTAL_LABEL_BYTES:
            raise ValueError("Network node labels exceed the 16 MiB aggregate display budget.")
        normalized = {"id": node_id, "label": label,
                                 "degree": _finite(node["degree"], "node degree", nonnegative=True),
                                 "group": _integer(node["group"], "node group")}
        if "identity" in node:
            normalized["identity"] = _identity(node["identity"])
            identity_key = tuple(normalized["identity"].values())
            if identity_key in identities:
                raise ValueError("Typed node identities must be unique within a frame.")
            identities.add(identity_key)
            budget[2] += 2 * len(normalized["identity"]["value"].encode("utf-8")) + 64
        for name in ("fixed", "pinned"):
            if name in node:
                if not isinstance(node[name], bool):
                    raise TypeError(f"{name} must be a boolean.")
                normalized[name] = node[name]
                budget[2] += 16
        if "fixed" in node and "pinned" in node and node["fixed"] != node["pinned"]:
            raise ValueError("fixed and pinned cannot disagree.")
        for name, bound in (("x", 1e9), ("y", 1e9), ("fx", 1e9), ("fy", 1e9), ("longitude", 180), ("latitude", 90)):
            if name in node:
                number = _finite(node[name], name)
                if abs(number) > bound:
                    raise ValueError(f"{name} must be between {-bound} and {bound}.")
                normalized[name] = number
                budget[2] += 40
        if node.get("fixed", node.get("pinned", False)) and not (("x" in node or "fx" in node) and ("y" in node or "fy" in node)):
            raise ValueError("A fixed node needs finite coordinates on both axes.")
        if ("x" in node) != ("y" in node):
            raise ValueError("Initial node coordinates require both x and y.")
        if "attrs" in node:
            normalized["attrs"] = _attrs(node["attrs"])
            budget[2] += _attrs_bytes(normalized["attrs"])
        if budget[2] > MAX_PAYLOAD_BYTES:
            raise ValueError("Network payload exceeds the 128 MiB conservative display budget.")
        normalized_nodes.append(normalized)
    normalized_edges = []
    for item in edges:
        edge = _keys(item, _EDGE_KEYS, "A network edge", optional=_EDGE_OPTIONAL_KEYS)
        source, target = _integer(edge["source"], "edge source"), _integer(edge["target"], "edge target")
        if source not in ids or target not in ids:
            raise ValueError("Every network edge endpoint must reference a displayed node.")
        normalized = {"source": source, "target": target,
                      "weight": _finite(edge["weight"], "edge weight")}
        if "attrs" in edge:
            normalized["attrs"] = _attrs(edge["attrs"])
            budget[2] += _attrs_bytes(normalized["attrs"])
        if budget[2] > MAX_PAYLOAD_BYTES:
            raise ValueError("Network payload exceeds the 128 MiB conservative display budget.")
        normalized_edges.append(normalized)
    result = {"nodes": normalized_nodes, "edges": normalized_edges,
              "directed": graph["directed"], **counts, "sampled": sampled, "selection": selection}
    if grouping is not None:
        result["grouping"] = grouping
    if "frames" in graph:
        frames = graph["frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= 60:
            raise ValueError("Timeline frames must be a list of 1 to 60 frames.")
        result["frames"] = []
        union_identity = {node["id"]: node.get("identity") for node in normalized_nodes}
        for frame in frames:
            frame = _keys(frame, frozenset({"label", "network"}), "A timeline frame")
            label = _text(frame["label"], "frame label", MAX_LABEL_BYTES, nonempty=True)
            data = validate_network(frame["network"], _nested=True, _budget=budget)
            if data["directed"] != result["directed"]:
                raise ValueError("Every timeline frame must share graph directedness.")
            for node in data["nodes"]:
                if node["id"] in union_identity and union_identity[node["id"]] != node.get("identity"):
                    raise ValueError("Timeline node IDs must retain the same typed identity across frames.")
                union_identity[node["id"]] = node.get("identity")
            result["frames"].append({"label": label, "network": data})
    return result


def _attrs_bytes(values):
    return 128 + sum(128 + 6 * len(key) + (6 * len(value) if isinstance(value, str) else 32)
                     for key, value in values.items())


def _color(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{name} must be an opaque CSS hex color.")
    return "#" + ("".join(char * 2 for char in value[1:]) if len(value) == 4 else value[1:]).lower()


def _layout_options(value, layout):
    common = {"iterations": (1, 2000, True), "work_limit": (1000, 1_000_000_000, True),
              "time_limit_ms": (100, 120_000, True)}
    allowed = {
        "d3-force": {"charge": (-10000, 0), "link_distance": (1, 10000), "theta": (.2, 2), "collision": bool},
        "forceatlas2": {"scaling": (.01, 10000), "gravity": (0, 100), "strong_gravity": bool,
            "edge_weight_influence": (0, 4), "linlog": bool, "outbound_attraction_distribution": bool,
            "jitter_tolerance": (.01, 10), "slowdown": (.01, 100), "theta": (.2, 2)},
        "circular": {"radius": (1, 100000), "angle": (-360, 360)},
        "grid": {"spacing": (1, 10000), "columns": (1, 100000, True)},
        "radial": {"spacing": (1, 10000), "root": (0, _MAX_INTEGER, True), "angle": (-360, 360)},
        "hierarchical": {"spacing": (1, 10000), "direction": {"TB", "BT", "LR", "RL"}},
        "geographic": {"projection": {"equirectangular", "mercator"}, "scale": (.01, 10000)},
        "community": {"spacing": (1, 10000), "radius": (1, 100000)}, "fixed": {},
    }[layout]
    allowed = {**common, **allowed}
    if not isinstance(value, Mapping) or set(value) - set(allowed):
        raise ValueError(f"Unsupported layout options for {layout}.")
    result = {}
    for name, item in value.items():
        rule = allowed[name]
        if rule is bool:
            if not isinstance(item, bool):
                raise TypeError(f"{name} must be a boolean.")
            result[name] = item
        elif isinstance(rule, set):
            if not isinstance(item, str) or item not in rule:
                raise ValueError(f"{name} must be one of {sorted(rule)}.")
            result[name] = item
        else:
            number = _integer(item, name, rule[1], minimum=rule[0]) if len(rule) == 3 else _finite(item, name)
            if not rule[0] <= number <= rule[1]:
                raise ValueError(f"{name} must be between {rule[0]} and {rule[1]}.")
            result[name] = number
    return result


def _mapping(value, name):
    color = name.endswith("color")
    if not isinstance(value, Mapping):
        if color:
            return _color(value, name)
        if name == "node_label":
            return _text(value, name, 200, nonempty=True)
        number = _finite(value, name)
        lower, upper = (.1, 64) if name == "edge_width" else (1, 256)
        if not lower <= number <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}.")
        return number
    item = _keys(value, frozenset({"field"}), name,
                 optional=frozenset({"scale", "range", "domain", "missing"}))
    scale = item.get("scale", "categorical" if name == "node_label" else "linear")
    if not isinstance(scale, str) or scale not in {"linear", "sqrt", "log", "categorical"}:
        raise ValueError("Mapping scale must be linear, sqrt, log or categorical.")
    result = {"field": _text(item["field"], "mapping field", 200, nonempty=True), "scale": scale}
    if name == "node_label" and scale != "categorical":
        raise ValueError("Label mappings must use the categorical scale.")
    for key in ("range", "domain"):
        if key not in item:
            continue
        values = item[key]
        if not isinstance(values, list) or not 2 <= len(values) <= 64 or scale != "categorical" and len(values) != 2:
            raise ValueError(f"Mapping {key} must contain two bounds, or 2 to 64 categorical values.")
        if key == "range":
            if name == "node_label":
                result[key] = [_text(entry, "label range", MAX_LABEL_BYTES) for entry in values]
            elif color:
                result[key] = [_color(entry, "color range") for entry in values]
            else:
                result[key] = [_mapping(entry, name) for entry in values]
        else:
            result[key] = [_scalar(entry, "mapping domain") if scale == "categorical" else _finite(entry, "mapping domain") for entry in values]
            if scale != "categorical" and (result[key][0] >= result[key][1] or scale == "log" and result[key][0] <= 0):
                raise ValueError("Numeric mapping domain must be increasing; log bounds must be positive.")
            if scale == "categorical" and len({(type(entry).__name__ if isinstance(entry, (str, bool)) or entry is None else "number", entry) for entry in result[key]}) != len(result[key]):
                raise ValueError("Categorical mapping domains must contain distinct typed scalar values.")
    if "missing" in item:
        if isinstance(item["missing"], Mapping):
            raise ValueError("Mapping missing values must be scalar presentation values.")
        result["missing"] = _mapping(item["missing"], name)
    return result


def validate_network_presentation(graph, options):
    """Check node references and layout prerequisites against every saved frame."""
    frames = [graph, *(item["network"] for item in graph.get("frames", []))]
    all_ids = {node["id"] for frame in frames for node in frame["nodes"]}
    index = options.get("frame_index", 0)
    timeline = graph.get("frames")
    if timeline and index >= len(timeline) or not timeline and index != 0:
        raise ValueError("frame_index must identify an available timeline frame.")
    active = timeline[index]["network"] if timeline else graph
    active_ids = {node["id"] for node in active["nodes"]}
    view = options.get("view")
    saved = {item["id"] for item in view["positions"]} if view else set()
    if view and (not saved <= active_ids or view.get("selected") is not None and view["selected"] not in active_ids):
        raise ValueError("Saved view references a node outside the active saved frame.")
    if view:
        active_nodes = {node["id"]: node for node in active["nodes"]}
        for point in view["positions"]:
            node = active_nodes[point["id"]]
            identity = node.get("identity")
            bound = point.get("identity")
            if identity is not None and bound is None:
                raise ValueError("Saved positions require typed identity binding for native graph nodes.")
            expected = identity or {"type": "integer", "value": str(node["id"])}
            if bound is not None and bound != expected:
                raise ValueError("Saved position identity does not match the active graph node.")
    for annotation in options.get("annotations", []):
        if "node_id" in annotation and annotation["node_id"] not in all_ids:
            raise ValueError("Annotation node_id must reference a saved graph node.")
    layout, layout_options = options.get("layout", "d3-force"), options.get("layout_options", {})
    for frame in frames:
        ids = {node["id"] for node in frame["nodes"]}
        if layout == "radial" and "root" in layout_options and layout_options["root"] not in ids:
            raise ValueError("Radial layout root must be present in every saved frame.")
        if layout == "geographic" and any("longitude" not in node or "latitude" not in node for node in frame["nodes"]):
            raise ValueError("Geographic layout requires longitude and latitude on every saved node.")
        if layout == "fixed" and any(node["id"] not in saved and not (("x" in node or "fx" in node) and ("y" in node or "fy" in node)) for node in frame["nodes"]):
            raise ValueError("Fixed layout requires coordinates or saved positions for every node.")
        if layout == "forceatlas2" and any(edge["weight"] < 0 for edge in frame["edges"]):
            raise ValueError("ForceAtlas2 does not support negative edge weights.")


def _filters(value):
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError("filters must contain at most 100 declarative predicates.")
    result = []
    for item in value:
        item = _keys(item, frozenset({"scope", "field", "op", "value"}), "A filter")
        if item["scope"] not in {"nodes", "edges"} or item["op"] not in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"}:
            raise ValueError("Unsupported filter scope or comparison operator.")
        test = item["value"]
        if item["op"] in {"in", "not_in"}:
            if not isinstance(test, list) or len(test) > 10_000:
                raise ValueError("Membership filters require a list with at most 10,000 scalar values.")
            test = [_scalar(entry, "filter value") for entry in test]
        else:
            test = _scalar(test, "filter value")
            if item["op"] in {"gt", "gte", "lt", "lte"} and (isinstance(test, bool) or not isinstance(test, (int, float))):
                raise ValueError("Ordered filters require a finite numeric threshold.")
        result.append({"scope": item["scope"], "field": _text(item["field"], "filter field", 200, nonempty=True), "op": item["op"], "value": test})
    return result


def _view(value):
    item = _keys(value, frozenset({"version", "positions", "transform"}), "Saved view",
                 optional=frozenset({"selected", "filters"}))
    if item["version"] != 1 or isinstance(item["version"], bool):
        raise ValueError("Saved view version must be integer 1.")
    positions = item["positions"]
    if not isinstance(positions, list) or len(positions) > MAX_NODES:
        raise ValueError("Saved positions must be a bounded list.")
    copied, ids = [], set()
    for point in positions:
        point = _keys(point, frozenset({"id", "x", "y"}), "Saved position", optional=frozenset({"pinned", "identity"}))
        node_id = _integer(point["id"], "position id")
        pinned = point.get("pinned", False)
        if node_id in ids or not isinstance(pinned, bool):
            raise ValueError("Saved position IDs must be unique and pinned must be boolean.")
        ids.add(node_id)
        x, y = _finite(point["x"], "position x"), _finite(point["y"], "position y")
        if abs(x) > 1e9 or abs(y) > 1e9:
            raise ValueError("Saved positions must lie between -1e9 and 1e9.")
        normalized = {"id": node_id, "x": x, "y": y, "pinned": pinned}
        if "identity" in point:
            normalized["identity"] = _identity(point["identity"])
        copied.append(normalized)
    transform = _keys(item["transform"], frozenset({"k", "x", "y"}), "Saved transform")
    transformed = {key: _finite(transform[key], "transform " + key) for key in transform}
    if not .02 <= transformed["k"] <= 12 or abs(transformed["x"]) > 1e9 or abs(transformed["y"]) > 1e9:
        raise ValueError("Saved transform is outside supported zoom/position bounds.")
    result = {"version": 1, "positions": copied, "transform": transformed}
    if "selected" in item:
        result["selected"] = None if item["selected"] is None else _integer(item["selected"], "selected node")
    if "filters" in item:
        result["filters"] = _filters(item["filters"])
    return result


def network_options(options: Any) -> dict[str, Any]:
    """Validate serializable layouts, appearance, filters and saved views."""
    if not isinstance(options, dict):
        raise TypeError("Network chart options must be a dictionary.")
    unknown = set(options) - _OPTION_KEYS
    if unknown:
        raise TypeError(f"Unsupported network chart option: {sorted(unknown, key=str)[0]}.")
    result = {}
    for name, lower, upper in (("width", 320, 2400), ("height", 240, 1600)):
        if name in options:
            result[name] = _integer(options[name], name, upper, minimum=lower)
    for name, lower, upper in (("point_size", 1, 24), ("line_width", .25, 12), ("opacity", 0, 1)):
        if name in options:
            value = _finite(options[name], name)
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}.")
            result[name] = value
    if "color" in options:
        result["color"] = _color(options["color"], "color")
    if "palette" in options:
        values = options["palette"]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not 1 <= len(values) <= 64:
            raise ValueError("palette must contain between 1 and 64 opaque CSS hex colors.")
        result["palette"] = [_color(value, "palette color") for value in values]
    if "layout" in options or "layout_options" in options:
        layout = options.get("layout", "d3-force")
        layout = "d3-force" if layout == "force" else layout
        if not isinstance(layout, str) or layout not in _LAYOUTS:
            raise ValueError("Unknown network layout.")
        result["layout"] = layout
        if "layout_options" in options:
            result["layout_options"] = _layout_options(options["layout_options"], layout)
    for name in ("node_size", "node_color", "node_label", "edge_width", "edge_color"):
        if name in options:
            result[name] = _mapping(options[name], name)
    if "filters" in options:
        result["filters"] = _filters(options["filters"])
    for name in ("legend", "timeline"):
        if name in options:
            if not isinstance(options[name], bool):
                raise TypeError(f"{name} must be boolean.")
            result[name] = options[name]
    if "labels" in options:
        labels = options["labels"]
        if isinstance(labels, bool):
            result["labels"] = labels
        else:
            labels = _keys(labels, frozenset(), "Label options", optional=frozenset({"show", "min_zoom", "max_count"}))
            show = labels.get("show", True)
            if not isinstance(show, bool):
                raise TypeError("Label show must be boolean.")
            zoom = _finite(labels.get("min_zoom", .01), "label min_zoom")
            if not 0 <= zoom <= 100:
                raise ValueError("Label minimum zoom must be between 0 and 100.")
            result["labels"] = {"show": show, "min_zoom": zoom,
                                "max_count": _integer(labels.get("max_count", 80), "label max_count", 10000)}
    if "annotations" in options:
        annotations = options["annotations"]
        if not isinstance(annotations, list) or len(annotations) > 100:
            raise ValueError("Network annotations must be a list of at most 100 items.")
        result["annotations"] = []
        for annotation in annotations:
            annotation = _keys(annotation, frozenset({"text"}), "Annotation", optional=frozenset({"node_id", "x", "y", "color"}))
            node_anchor = "node_id" in annotation
            if node_anchor and ("x" in annotation or "y" in annotation) or not node_anchor and not {"x", "y"} <= set(annotation):
                raise ValueError("An annotation needs either node_id or both x and y.")
            copied = {"text": _text(annotation["text"], "annotation text", MAX_LABEL_BYTES, nonempty=True)}
            if node_anchor:
                copied["node_id"] = _integer(annotation["node_id"], "annotation node_id")
            else:
                for key in ("x", "y"):
                    copied[key] = _finite(annotation[key], "annotation " + key)
                    if abs(copied[key]) > 1e9:
                        raise ValueError("Annotation positions must lie between -1e9 and 1e9.")
            if "color" in annotation:
                copied["color"] = _color(annotation["color"], "annotation color")
            result["annotations"].append(copied)
    if "view" in options:
        result["view"] = _view(options["view"])
    if "frame_index" in options:
        result["frame_index"] = _integer(options["frame_index"], "frame_index", 59)
    if "seed" in options:
        result["seed"] = _integer(options["seed"], "seed", 2**32 - 1)
    return result


def network(graph: Any, *, title: str | None = None, max_nodes: int = 1_000,
            max_edges: int = 5_000, seed: int = 0, groups: Any = None, **options):
    """Create an efficient network chart from ``graph.to_plot_data(...)``.

    Limits bound the display, not graph analysis. Selection belongs to the
    supplied graph and is recorded verbatim with full and shown counts. HTML
    export is interactive and offline; LaTeX exports a compact summary table,
    not a force-layout drawing or metrics inferred from a displayed subset.
    Optional groups are a node mapping or community membership table accepted
    by the graph. Omitting groups preserves older to_plot_data protocols.
    """
    max_nodes = _integer(max_nodes, "max_nodes", MAX_NODES, minimum=1)
    max_edges = _integer(max_edges, "max_edges", MAX_EDGES, minimum=1)
    seed = _integer(seed, "seed", 2**32 - 1)
    heading = "Network" if title is None else _text(title, "title", MAX_LABEL_BYTES)
    presentation = network_options(options)
    if seed:
        presentation["seed"] = seed
    export = getattr(graph, "to_plot_data", None)
    if not callable(export):
        raise TypeError("network needs a graph with a callable to_plot_data method.")
    export_options = {"max_nodes": max_nodes, "max_edges": max_edges, "seed": seed}
    if groups is not None:
        export_options["groups"] = groups
    payload = validate_network(export(**export_options))
    frames = [payload, *(frame["network"] for frame in payload.get("frames", []))]
    if any(frame["shown_node_count"] > max_nodes or frame["shown_edge_count"] > max_edges for frame in frames):
        raise ValueError("The graph did not honor the requested network display limits.")
    validate_network_presentation(payload, presentation)
    from .charts import PlotSpec
    return PlotSpec("network", heading, "", "", [], payload["shown_node_count"],
                    payload["node_count"], 0, {"network": payload, "options": presentation})


__all__ = ["network"]
