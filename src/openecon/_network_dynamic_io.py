"""Bounded three-pass interval GEXF import and atomic identity-preserving export."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

from openecon.analysis_contracts import AnalysisError
from openecon.networks import _Budget, _error, _label, _weight
from openecon._network_io import (_Reader, _tag, _xml_text, _convert, _kind, _text, _attributes,
                                  _attribute_bytes, _IDENTITY, _EDGE_IDENTITY, _WEIGHTED, _TYPES)
from openecon._network_dynamic import DynamicNetwork, _BOUNDS

_PARSER = 262144
_CHILDREN = {
    "gexf": {"meta", "graph"}, "meta": {"creator", "description", "keywords"},
    "graph": {"attributes", "nodes", "edges"}, "attributes": {"attribute"},
    "attribute": {"default"}, "nodes": {"node"}, "edges": {"edge"},
    "node": {"attvalues", "spells"}, "edge": {"attvalues", "spells"},
    "attvalues": {"attvalue"}, "spells": {"spell"},
}
_XML_FIELDS = {
    "gexf": {"version", "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"},
    "meta": {"lastmodifieddate"}, "graph": {"mode", "defaultedgetype", "timeformat", "timerepresentation", "timezone", "idtype"} | _BOUNDS,
    "attributes": {"class", "mode"}, "attribute": {"id", "title", "type"},
    "nodes": {"count"}, "edges": {"count"}, "node": {"id", "label"} | _BOUNDS,
    "edge": {"id", "source", "target", "label", "kind", "type", "weight"} | _BOUNDS,
    "attvalue": {"for", "value"} | _BOUNDS, "spell": _BOUNDS,
}


def _records(path, maximum, hashes, guard):
    reader, stack, namespace = _Reader(path, maximum), [], None
    graph_count, record_events, workspace = 0, 0, 0
    try:
        for event, element in ET.iterparse(reader, events=("start", "end")):
            tag = _tag(element)
            if event == "start":
                if not stack:
                    if tag != "gexf":
                        _error("file_format", "Dynamic interchange requires a GEXF root.")
                    version = element.get("version", "1.3")
                    namespaces = {"1.2draft": {"http://gexf.net/1.2draft", "http://www.gexf.net/1.2draft"},
                                  "1.3": {"http://gexf.net/1.3"}}
                    uri = element.tag.split("}")[0][1:] if element.tag.startswith("{") else ""
                    if version not in namespaces or uri and uri not in namespaces[version]:
                        _error("file_format", "Dynamic GEXF supports matching 1.2draft/1.3 namespaces only.")
                    namespace = element.tag.split("}")[0] + "}" if uri else ""
                    yield "root", dict(element.attrib), None
                elif tag not in _CHILDREN.get(_tag(stack[-1]), set()) or element.tag != namespace + tag:
                    _error("file_feature", "Nested, mixed, timestamp, visualization and other GEXF extensions require explicit conversion.")
                if set(element.attrib) - _XML_FIELDS.get(tag, set()):
                    _error("file_feature", "Unsupported GEXF XML fields are not silently discarded.")
                for name, value in element.attrib.items():
                    if len(name.encode()) > 200 or len(value.encode()) > 16384:
                        _error("file_attribute", "XML names and scalar values exceed their bounded limits.")
                workspace += 512 + 16 * sum(len(value) for value in element.attrib.values())
                guard(workspace + _PARSER)
                if tag in {"node", "edge"}:
                    record_events = 0
                elif tag in {"spell", "attvalue"}:
                    record_events += 1
                    if record_events > 8192:
                        _error("event_budget", "One XML record supports at most 8192 children including static/native fields.")
                if len(stack) > 12 or len(element.attrib) > 16:
                    _error("file_feature", "Unsupported XML nesting or field count.")
                if tag == "graph":
                    graph_count += 1
                    if graph_count > 1 or element.get("mode") != "dynamic" or element.get("timerepresentation", "interval") != "interval":
                        _error("file_feature", "Use exactly one dynamic interval graph; static/slice/timestamp modes need their explicit APIs.")
                    yield "graph", dict(element.attrib), None
                elif tag == "attributes" and (element.get("class") not in {"node", "edge"}
                                               or element.get("mode", "static") not in {"static", "dynamic"}):
                    _error("file_feature", "Only static/dynamic node/edge scalar declarations are supported.")
                stack.append(element)
            else:
                if any(value and len(value.encode()) > 16384 for value in (element.text, element.tail)):
                    _error("file_attribute", "XML text exceeds the bounded scalar limit.")
                workspace += 16 * sum(len(value) for value in (element.text, element.tail) if value)
                guard(workspace + _PARSER)
                if tag in {"attribute", "node", "edge", "keywords"}:
                    context = dict(stack[-2].attrib) if len(stack) > 1 else {}
                    yield tag, element, context
                    stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                elif tag in {"attributes", "nodes", "edges", "meta"}:
                    stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                stack.pop()
        if graph_count != 1:
            _error("file_format", "The GEXF must contain exactly one dynamic interval graph.")
        hashes.append(reader.digest.hexdigest())
    except ET.ParseError:
        _error("file_format", "Malformed dynamic GEXF XML.")
    finally:
        reader.file.close()


def _entity(element, scope, definitions, idtype, directed):
    wire = element.get("id")
    if wire is None:
        _error("file_edge" if scope == "edge" else "file_format", "Identity-preserving dynamic import requires explicit node/edge XML IDs.")
    _label(wire)
    identity = int(wire) if idtype in {"integer", "long"} else wire
    attrs, dynamic, supplied, native = {}, {}, set(), None
    for (current, _), (name, kind, mode, default) in definitions.items():
        if current == scope and mode == "static" and default is not None:
            if name in {_IDENTITY, _EDGE_IDENTITY}:
                _error("file_attribute", "Native identity fields cannot have defaults.")
            attrs[name] = default
    for parent in element:
        if _tag(parent) != "attvalues":
            continue
        for value in parent:
            key = value.get("for")
            definition = definitions.get((scope, key))
            if definition is None or value.get("value") is None:
                _error("file_attribute", "Timed scalar values must reference a declared field and explicit value.")
            name, kind, mode, _ = definition
            scalar = _convert(value.get("value"), kind)
            bounds = {k: v for k, v in value.attrib.items() if k in _BOUNDS}
            if mode == "static":
                if key in supplied or bounds:
                    _error("file_attribute", "Static fields occur once without time bounds.")
                supplied.add(key)
                if name in {_IDENTITY, _EDGE_IDENTITY}:
                    native = _label(json.loads(scalar))
                else:
                    attrs[name] = scalar
            else:
                dynamic.setdefault(name, []).append({"value": scalar, **bounds})
    if native is not None:
        identity = native
    for name in ("label", "kind"):
        if element.get(name) is not None:
            if name in dynamic or name in attrs and attrs[name] != element.get(name):
                _error("file_attribute", "Conflicting XML core labels/kinds and scalar fields need explicit resolution.")
            attrs[name] = element.get(name)
    spells = [child for child in element if _tag(child) == "spells"]
    if len(spells) > 1:
        _error("dynamic_time", "One entity can contain only one spells container.")
    if sum(_tag(child) == "attvalues" for child in element) > 1:
        _error("file_attribute", "One entity can contain only one attvalues container.")
    raw_bounds = {k: v for k, v in element.attrib.items() if k in _BOUNDS}
    if spells and raw_bounds:
        _error("dynamic_time", "Use either direct time bounds or spell children, not both.")
    record = {"node" if scope == "node" else "edge_id": _label(identity), "attributes": _attributes(attrs),
              "dynamic_attributes": dynamic}
    if spells:
        record["spells"] = [dict(child.attrib) for child in spells[0]]
    elif raw_bounds:
        record.update(raw_bounds)
    if scope == "edge":
        direction = element.get("type", "directed" if directed else "undirected")
        if direction != ("directed" if directed else "undirected"):
            _error("file_feature", "Mixed edge directions require an explicit projection.")
        record.update(source=element.get("source"), target=element.get("target"))
        scalar = record["attributes"].pop("weight", None)
        if element.get("weight") is not None:
            if scalar is not None and float(element.get("weight")) != scalar:
                _error("file_attribute", "Conflicting core and declared static weights.")
            scalar = float(element.get("weight"))
        if scalar is not None:
            record["weight"] = _weight(scalar)
    return record, wire


def read_dynamic_network(path, *, max_memory_mb=256, max_file_mb=512, max_events=1_000_000, max_work=50_000_000):
    path = Path(path)
    if path.suffix.lower() != ".gexf":
        _error("file_feature", "Continuous dynamic interchange supports .gexf only.")
    budget, maximum = _Budget(max_memory_mb), _Budget(max_file_mb).limit
    if path.stat().st_size > maximum:
        _error("file_size", "The GEXF exceeds max_file_mb.")
    owned, definitions, fields, hashes, live, parse_live = _PARSER, {}, set(), [], [0], [0]
    root, settings, native_weighted = None, None, None
    try:
        for tag, item, context in _records(path, maximum, hashes, lambda size: budget.check(owned + size)):
            if tag == "root":
                root = item
            elif tag == "graph":
                settings = item
            elif tag == "keywords" and (item.text or "").startswith(_WEIGHTED + "="):
                if native_weighted is not None:
                    _error("file_attribute", "Duplicate native weighting metadata.")
                native_weighted = _convert(item.text.split("=", 1)[1], "boolean")
            elif tag == "attribute":
                scope, mode = context.get("class"), context.get("mode", "static")
                key, name, kind = item.get("id"), item.get("title"), item.get("type", "string")
                if (scope not in {"node", "edge"} or mode not in {"static", "dynamic"} or not key or not name
                        or kind not in _TYPES or (scope, key) in definitions or (scope, name) in fields):
                    _error("file_attribute", "Invalid, duplicate or conflicting GEXF scalar declarations.")
                native = name in {_IDENTITY, _EDGE_IDENTITY}
                if native:
                    if (scope, name) not in {("node", _IDENTITY), ("edge", _EDGE_IDENTITY)} or mode != "static" or kind != "string":
                        _error("file_attribute", "Native identities require matching static string fields.")
                else:
                    _attributes({name: 0})
                defaults = list(item)
                if len(defaults) > 1:
                    _error("file_attribute", "An attribute declaration has at most one default.")
                default = _convert(defaults[0].text or "", kind) if defaults else None
                owned += 2048 + (0 if default is None else _attribute_bytes({"value": default}))
                budget.check(owned + _PARSER)
                definitions[scope, key] = (name, kind, mode, default)
                fields.add((scope, name))
                if len(fields) > 258:
                    _error("file_attribute", "Dynamic GEXF admits 128 scalar fields per scope plus native IDs.")
        if not root or not settings:
            _error("file_format", "Missing dynamic graph declaration.")
        directed = settings.get("defaultedgetype", "undirected")
        if directed not in {"directed", "undirected"}:
            _error("file_feature", "Mixed graph direction is unsupported.")
        if settings.get("timezone", "UTC") not in {"UTC", "Z", "+00:00"}:
            _error("file_feature", "Graph-level timezone defaults support UTC only; use explicit ISO offsets on dateTime values.")
        idtype = settings.get("idtype", "string")
        if idtype not in {"string", "integer", "long"}:
            _error("file_feature", "Only string/integer GEXF identity types are supported.")
        defaults = {scope: {name: default for (current, _), (name, _, mode, default) in definitions.items()
                           if current == scope and mode == "dynamic" and default is not None}
                    for scope in ("node", "edge")}
        node_wires, edge_wires = {}, set()

        def parser_guard(size):
            parse_live[0] = size
            budget.check(owned + live[0] + size)

        def records(scope):
            nonlocal owned
            stream = _records(path, maximum, hashes, parser_guard)
            try:
                for tag, item, _ in stream:
                    if tag != scope:
                        continue
                    budget.check(owned + live[0] + 2 * parse_live[0] + 65536)
                    record, wire = _entity(item, scope, definitions, idtype, directed == "directed")
                    owned += 1024 + 8 * len(wire)
                    budget.check(owned + live[0] + parse_live[0])
                    if scope == "node":
                        if wire in node_wires:
                            _error("invalid_label", "Duplicate XML node ID.")
                        node_wires[wire] = record["node"]
                    else:
                        if wire in edge_wires:
                            _error("duplicate_edge_id", "Duplicate XML edge ID.")
                        edge_wires.add(wire)
                        for name in ("source", "target"):
                            if record[name] not in node_wires:
                                _error("unknown_node", "An edge endpoint is not a declared XML node ID.")
                            record[name] = node_wires[record[name]]
                    yield record
            finally:
                stream.close()
                parse_live[0] = 0

        budget.check(owned + _PARSER + 1_048_576)
        # Wire-ID sets grow during construction. The coupled guard includes
        # their current size and live graph/parser buffers on every record.
        graph = DynamicNetwork(records("node"), records("edge"), directed=directed == "directed",
            timeformat=settings.get("timeformat", "double"), version=root.get("version", "1.3"),
            interval={k: v for k, v in settings.items() if k in _BOUNDS},
            node_defaults=defaults["node"], edge_defaults=defaults["edge"],
            max_memory_mb=(budget.limit - owned - _PARSER) / 1024**2,
            max_events=max_events, max_work=max_work, _live_bytes=live,
            _external_guard=lambda size: budget.check(owned + size + parse_live[0]))
        if len(hashes) != 3 or len(set(hashes)) != 1:
            _error("file_changed", "The dynamic GEXF changed during its bounded three-pass read.")
        if native_weighted is not None:
            if not native_weighted and ("weight" in graph._defaults["edge"] or any(
                    record["weight"] != 1. or "weight" in record["dynamic_attributes"]
                    for record in graph._edges)):
                _error("file_attribute", "Native unweighted metadata conflicts with stored non-unit or timed weights.")
            graph.weighted = native_weighted
        graph._external_guard, graph._live = None, None
        graph._budget.peak = max(graph._budget.peak + owned + _PARSER, budget.peak)
        graph._budget.limit = budget.limit
        graph._guard(owned + _PARSER)
        graph._metadata.update(input_format="dynamic-gexf", file_sha256=hashes[0], file_passes=3,
            weighted=graph.weighted, max_memory_bytes=budget.limit, estimated_import_peak_bytes=graph._budget.peak)
        return graph
    except AnalysisError:
        raise
    except (TypeError, ValueError, OverflowError, UnicodeError):
        _error("file_format", "Malformed dynamic GEXF scalar value.")


def _declarations(graph):
    fields = {("node", _IDENTITY): ("static", "string"), ("edge", _EDGE_IDENTITY): ("static", "string")}
    def register(scope, name, mode, value):
        kind = _kind(value)
        previous = fields.get((scope, name))
        if previous is not None and previous != (mode, kind):
            _error("file_attribute", "XML export requires one scalar type/mode per named field; no implicit coercion.")
        fields[scope, name] = (mode, kind)
        graph._guard(4096 * len(fields))
    for scope in ("node", "edge"):
        for name, value in graph._defaults[scope].items():
            register(scope, name, "dynamic", value)
        for record in graph._nodes if scope == "node" else graph._edges:
            for name, value in record["attributes"].items():
                register(scope, name, "static", value)
            for name, values in record["dynamic_attributes"].items():
                for _, value in values:
                    register(scope, name, "dynamic", value)
    return {key: (f"a{i}", mode, kind) for i, (key, (mode, kind)) in enumerate(fields.items())}


def _write(graph, stream):
    declarations = _declarations(graph)
    stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    stream.write(f'<gexf xmlns="http://gexf.net/{graph.version}" version="{graph.version}"><meta><keywords>{_WEIGHTED}={str(graph.weighted).lower()}</keywords></meta>\n')
    attributes = {"mode": "dynamic", "defaultedgetype": "directed" if graph.directed else "undirected",
                  "timeformat": graph.timeformat, "idtype": "string", **{k: _text(v) for k, v in graph.interval.items()}}
    if graph.version == "1.3":
        attributes["timerepresentation"] = "interval"
    element = ET.Element("graph", attributes)
    stream.write(_xml_text(element).replace(" />", ">") + "\n")
    for scope in ("node", "edge"):
        for mode in ("static", "dynamic"):
            parent = ET.Element("attributes", {"class": scope, "mode": mode})
            for (current, name), (key, current_mode, kind) in declarations.items():
                if current == scope and mode == current_mode:
                    item = ET.SubElement(parent, "attribute", {"id": key, "title": name, "type": kind})
                    if mode == "dynamic" and name in graph._defaults[scope]:
                        ET.SubElement(item, "default").text = _text(graph._defaults[scope][name])
            if len(parent):
                stream.write(_xml_text(parent) + "\n")
    for scope in ("node", "edge"):
        stream.write("<" + scope + "s>\n")
        for i, record in enumerate(graph._nodes if scope == "node" else graph._edges):
            workspace = 4096 * len(declarations) + 4096 * len(record["spells"]) + 4 * _attribute_bytes(record["attributes"])
            workspace += sum(4096 + 4 * _attribute_bytes({name: value}) for name, events in record["dynamic_attributes"].items() for _, value in events)
            graph._guard(workspace + 262144)
            item = ET.Element(scope, {"id": ("n" if scope == "node" else "e") + str(i)})
            identifier = record["node" if scope == "node" else "edge_id"]
            if scope == "edge":
                item.attrib.update(source="n" + str(graph._node_index[record["source"]]),
                    target="n" + str(graph._node_index[record["target"]]), weight=_text(record["weight"]))
            parent = ET.SubElement(item, "spells")
            for spell in record["spells"]:
                ET.SubElement(parent, "spell", {k: _text(v) for k, v in spell.record().items()})
            parent = ET.SubElement(item, "attvalues")
            native = _IDENTITY if scope == "node" else _EDGE_IDENTITY
            ET.SubElement(parent, "attvalue", {"for": declarations[scope, native][0], "value": json.dumps(identifier, ensure_ascii=False)})
            for name, value in record["attributes"].items():
                ET.SubElement(parent, "attvalue", {"for": declarations[scope, name][0], "value": _text(value)})
            for name, events in record["dynamic_attributes"].items():
                for interval, value in events:
                    ET.SubElement(parent, "attvalue", {"for": declarations[scope, name][0], "value": _text(value),
                        **{k: _text(v) for k, v in interval.record().items()}})
            stream.write(_xml_text(item) + "\n")
        stream.write("</" + scope + "s>\n")
    stream.write("</graph></gexf>\n")


def write_dynamic_network(graph, path, *, overwrite=False):
    path = Path(path)
    if path.suffix.lower() != ".gexf":
        _error("file_feature", "Dynamic intervals require .gexf; use an explicit static slice for other formats.")
    if not isinstance(overwrite, bool):
        _error("invalid_option", "overwrite must be boolean.")
    graph._guard(262144)
    fd, temporary = tempfile.mkstemp(prefix=".openecon-dynamic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            _write(graph, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return path
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
