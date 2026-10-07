"""Streaming static GraphML, GEXF and Pajek interchange without graph libraries.

Mixed, dynamic, nested and hypergraphs are rejected instead of being silently
flattened. Scalar attributes and exact OpenEcon node identities are preserved.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
import xml.etree.ElementTree as ET
import re

from openecon.analysis_contracts import AnalysisError
from openecon.networks import _Budget, _error, _integer, _key, _label, _weight, network

_IDENTITY = "openecon.identity"
_WEIGHTED = "openecon.weighted"
_EDGE_IDENTITY = "openecon.edge_identity"
_FORMATS = {"graphml", "gexf", "pajek"}
_TYPES = {"boolean", "bool", "int", "integer", "long", "float", "double", "string"}
_PARSER_BYTES = 262144
_MARKUP_TOKENS = re.compile(br"['\">]")
_TEXT_TOKENS = re.compile(br"<")


def _format(path, format):
    if format is not None and (not isinstance(format, str) or not format):
        _error("file_format", "format must be graphml, gexf or pajek.")
    value = (format or Path(path).suffix.lstrip('.')).lower()
    value = "pajek" if value == "net" else value
    if value not in _FORMATS:
        _error("file_format", "Use GraphML (.graphml), GEXF (.gexf), or Pajek (.net).")
    return value


def _scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and -(2**63) <= value < 2**63:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 16_384:
        try:
            if len(value.encode('utf-8')) <= 16_384 and not any(
                    ord(c) < 32 and c not in '\t\n\r' or 127 <= ord(c) < 160 for c in value):
                return value
        except UnicodeError:
            pass
    _error("file_attribute", "Attributes must be bounded UTF-8 strings, booleans, int64 or finite floats.")


def _attributes(values):
    if not isinstance(values, dict) or len(values) > 128:
        _error("file_attribute", "Each item accepts at most 128 named scalar attributes.")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or len(name) > 200 or any(
                ord(c) < 32 or 127 <= ord(c) < 160 for c in name):
            _error("file_attribute", "Attribute names must be bounded nonempty UTF-8 strings.")
        try:
            if len(name.encode('utf-8')) > 200:
                raise UnicodeError()
        except UnicodeError:
            _error("file_attribute", "Attribute names must be bounded nonempty UTF-8 strings.")
        if name in {_IDENTITY, _WEIGHTED, _EDGE_IDENTITY}:
            _error("file_attribute", "OpenEcon identity and weighted keys are reserved.")
        result[name] = _scalar(value)
    return result


def _attribute_bytes(values):
    """Cheap conservative copy/ownership preflight, without encoded temporaries."""
    if not isinstance(values, dict) or len(values) > 128:
        _error("file_attribute", "Each item accepts at most 128 named scalar attributes.")
    result = 256
    for name, value in values.items():
        if not isinstance(name, str) or not name or len(name) > 200:
            _error("file_attribute", "Attribute names must be bounded nonempty UTF-8 strings.")
        if isinstance(value, str):
            if len(value) > 16384:
                _error("file_attribute", "String attributes exceed the bounded scalar limit.")
            size = 16 * len(value)  # Four UTF-8 bytes/character and copy/parse allowance.
        elif isinstance(value, (bool, int, float)):
            size = 128
        else:
            _error("file_attribute", "Attributes must be scalar strings, booleans, int64 or finite floats.")
        result += 256 + 16 * len(name) + size
    return result


def attach_attributes(graph, *, nodes=None, edges=None, graph_attributes=None):
    """Validate and own immutable scalar attributes, charging the graph budget."""
    node_attrs, edge_attrs = {}, {}
    valid_edges = None
    estimated = 0
    if any(value is not None and not isinstance(value, dict) for value in (nodes, edges, graph_attributes)):
        _error("file_attribute", "Node, edge and graph attributes must be dictionaries.")
    for label, attrs in (nodes or {}).items():
        label = _label(label)
        if label not in graph._index:
            _error("unknown_node", "Node attributes refer to a node outside the graph.")
        addition = _attribute_bytes(attrs)
        graph._guard(estimated + addition)
        item = _attributes(attrs)
        if not item:
            continue
        node_attrs[label] = item
        estimated += addition
        graph._guard(estimated)
    for pair, attrs in (edges or {}).items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            _error("file_attribute", "Edge attributes use (source, target) tuple keys.")
        u, v = map(_label, pair)
        if u not in graph._index or v not in graph._index:
            _error("unknown_node", "Edge attributes refer to a node outside the graph.")
        addition = 64 + _attribute_bytes(attrs)
        graph._guard(estimated + addition + 256 * graph.edge_count + 128 * min(graph.edge_count, 65536))
        if valid_edges is None:
            graph._guard(estimated + 256 * graph.edge_count + 128 * min(graph.edge_count, 65536))
            valid_edges = {(u, v) for u, v, _ in _edge_records(graph)}
        i, j = graph._index[u], graph._index[v]
        if not graph.directed and i > j:
            i, j = j, i
        if (i, j) not in valid_edges:
            _error("file_attribute", "Attributes refer to an edge outside the graph.")
        pair = (u, v) if graph.directed or _key(u) <= _key(v) else (v, u)
        item = _attributes(attrs)
        if 'weight' in item:
            _error("file_attribute", "Edge attribute 'weight' is reserved for the actual aggregate edge weight.")
        if not item:
            continue
        edge_attrs[pair] = item
        estimated += addition
        graph._guard(estimated + 192 * graph.edge_count)
    addition = _attribute_bytes(graph_attributes or {})
    graph._guard(estimated + addition + (256 * graph.edge_count if valid_edges is not None else 0))
    owned_graph = _attributes(graph_attributes or {})
    estimated += addition
    graph._guard(estimated)
    graph._node_attributes, graph._edge_attributes, graph._graph_attributes = node_attrs, edge_attrs, owned_graph
    graph._base_bytes += estimated
    graph._metadata.update(attribute_storage_bytes=estimated, estimated_owned_graph_bytes=graph._base_bytes)
    return graph


def _convert(text, kind):
    if kind not in _TYPES:
        _error("file_attribute", f"Unsupported scalar attribute type: {kind}.")
    try:
        if kind in {'boolean', 'bool'}:
            if text not in {'true', 'false', '0', '1'}:
                raise ValueError('boolean')
            return text in {'true', '1'}
        return _scalar(int(text) if kind in {'int', 'integer', 'long'} else
                       float(text) if kind in {'float', 'double'} else text)
    except AnalysisError:
        raise
    except (ValueError, OverflowError) as exc:
        _error("file_attribute", f"Malformed {kind} attribute: {str(exc)[:80]}.")


class _Reader:
    def __init__(self, path, maximum):
        self.file = open(path, 'rb')
        self.maximum, self.count, self.tail = maximum, 0, b''
        self.digest = hashlib.sha256()
        self.decoder = codecs.getincrementaldecoder('utf-8-sig')()
        self.markup, self.quote, self.segment = False, None, 0
        self.special, self.special_tail, self.prefix = None, b'', None

    def _bounded_tokens(self, data):
        # Bound parser-internal tokens before ElementTree can accumulate an
        # arbitrarily large quoted attribute or text node. Quoted '>' is data.
        at = 0
        while at < len(data):
            if self.prefix is not None:
                take = 1
                self.prefix += data[at:at + take]
                self.segment += take
                at += take
                if self.prefix.startswith(b'<![CDATA['):
                    self.special, self.prefix = b']]>', None
                elif self.prefix.startswith(b'<!--'):
                    self.special, self.prefix = b'-->', None
                elif any(marker.startswith(self.prefix) for marker in (b'<![CDATA[', b'<!--')):
                    continue
                else:
                    self.prefix = None
            if self.special is not None:
                block = self.special_tail + data[at:]
                found = block.find(self.special)
                stop = len(data) if found < 0 else at + max(0, found + len(self.special) - len(self.special_tail))
                self.segment += stop - at
                if self.segment > 131072:
                    _error("file_attribute", "XML text and markup tokens cannot exceed 128 KiB.")
                if found < 0:
                    self.special_tail = block[-2:]
                    break
                self.special, self.special_tail, self.markup, self.segment = None, b'', False, 0
                at = stop
                continue
            delimiter = self.quote if self.markup and self.quote is not None else None
            match = (data.find(bytes([delimiter]), at) if delimiter is not None else
                     (_MARKUP_TOKENS if self.markup else _TEXT_TOKENS).search(data, at))
            offset = (match if isinstance(match, int) else match.start() if match else -1)
            stop = len(data) if offset < 0 else offset + 1
            self.segment += stop - at
            if self.segment > 131072:
                _error("file_attribute", "XML text and markup tokens cannot exceed 128 KiB.")
            if offset < 0:
                break
            token = data[offset]
            if not self.markup:
                self.markup, self.segment = True, 1
                rest = data[offset:min(len(data), offset + 9)]
                if rest.startswith(b'<![CDATA['):
                    self.special, self.segment = b']]>', 9
                    stop = offset + 9
                elif rest.startswith(b'<!--'):
                    self.special, self.segment = b'-->', 4
                    stop = offset + 4
                elif any(marker.startswith(rest) for marker in (b'<![CDATA[', b'<!--')):
                    self.prefix, self.segment = rest, len(rest)
                    stop = len(data)
            elif self.quote is not None:
                self.quote = None
            elif token in (34, 39):
                self.quote = token
            else:
                self.markup, self.segment = False, 0
            at = stop

    def read(self, size=-1):
        data = self.file.read(min(16384, size) if size >= 0 else 16384)
        self.count += len(data)
        if self.count > self.maximum:
            _error("file_size", "Input exceeds max_file_mb.")
        scan = (self.tail + data).upper()
        if b'<!DOCTYPE' in scan or b'<!ENTITY' in scan or b'\x00' in data:
            _error("unsafe_xml", "DTD, entities and UTF-16 XML are not accepted; use UTF-8 static XML.")
        self.tail = scan[-16:]
        if self.count == len(data):
            declaration = re.search(br'encoding\s*=\s*[\'\"]([^\'\"]+)', data[:16384], re.I)
            if declaration and declaration[1].lower() not in {b'utf-8', b'utf8'}:
                _error("unsafe_xml", "Network XML must declare UTF-8 encoding.")
        self.decoder.decode(data, final=not data)
        self._bounded_tokens(data)
        self.digest.update(data)
        return data


def _tag(element):
    return element.tag.rsplit('}', 1)[-1]


def _xml_records(path, maximum, format, receipt, guard=None):
    reader, stack = _Reader(path, maximum), []
    graphs = 0
    workspace = 0
    namespace = None
    allowed_tags = ({'graphml', 'graph', 'node', 'edge', 'key', 'data', 'default', 'desc'}
                    if format == 'graphml' else
                    {'gexf', 'graph', 'nodes', 'edges', 'node', 'edge', 'attributes', 'attribute',
                     'default', 'attvalues', 'attvalue', 'meta', 'creator', 'description', 'keywords'})
    try:
        for event, element in ET.iterparse(reader, events=('start', 'end')):
            tag = _tag(element)
            if event == 'start':
                if not stack:
                    if tag != ('graphml' if format == 'graphml' else 'gexf'):
                        _error("file_format", "The XML root does not match its file format.")
                    allowed = ({'http://graphml.graphdrawing.org/xmlns'} if format == 'graphml' else
                               {'http://gexf.net/1.2draft', 'http://gexf.net/1.3', 'http://www.gexf.net/1.2draft'})
                    if element.tag.startswith('{') and element.tag.split('}')[0][1:] not in allowed:
                        _error("file_format", "Unsupported network XML namespace.")
                    namespace = element.tag.split('}')[0] + '}' if element.tag.startswith('{') else ''
                if tag not in allowed_tags or element.tag != namespace + tag:
                    _error("file_feature", "Only static topology and scalar XML attributes are supported; convert extensions explicitly.")
                if tag == 'node' and (any(_tag(parent) == 'node' for parent in stack) or element.get('pid') is not None):
                    _error("file_feature", "Hierarchical nodes require an explicit flat-graph projection.")
                if tag in {'node', 'edge'} and (not stack or _tag(stack[-1]) !=
                                               ('graph' if format == 'graphml' else tag + 's')):
                    _error("file_feature", "Node and edge records must be in the static graph container.")
                for name, value in element.attrib.items():
                    if len(name.encode('utf-8')) > 200 or len(value.encode('utf-8')) > 16384:
                        _error("file_attribute", "XML attribute names or values exceed their bounded scalar limits.")
                workspace += 512 + 4 * sum(len(value.encode('utf-8')) for value in element.attrib.values())
                if guard:
                    guard(workspace + _PARSER_BYTES)
                if tag in {'data', 'attvalue'}:
                    record = next((parent for parent in reversed(stack) if _tag(parent) in {'node', 'edge'}), None)
                    if record is not None and sum(1 for child in record.iter() if _tag(child) in {'data', 'attvalue'}) > 130:
                        _error("file_attribute", "Each network record accepts at most 128 scalar attributes plus native metadata.")
                stack.append(element)
                if len(stack) > 32 or len(element.attrib) > 128:
                    _error("file_attribute", "XML nesting or attribute count exceeds the static graph limit.")
                if tag in {'hyperedge', 'port', 'locator', 'spells', 'spell', 'slice', 'slices'}:
                    _error("file_feature", "Hypergraphs, ports, external references and dynamic graphs require an explicit conversion.")
                if tag == 'graph':
                    graphs += 1
                    if graphs > 1 or element.get('mode', 'static') != 'static':
                        _error("file_feature", "Only one static graph is supported per file.")
                    yield 'graph', element, None
                elif tag == 'attributes':
                    if element.get('mode', 'static') != 'static':
                        _error("file_feature", "Dynamic attribute declarations are unsupported.")
                if any(name in element.attrib for name in ('start', 'end', 'startopen', 'endopen')):
                    _error("file_feature", "Time-varying graph records require an explicit snapshot.")
            else:
                if tag in {'data', 'default', 'attvalue'} and len(element):
                    _error("file_feature", "Structured XML attribute values require explicit scalar conversion.")
                if any(text and len(text.encode('utf-8')) > 16384 for text in (element.text, element.tail)):
                    _error("file_attribute", "XML values exceed the scalar attribute limit.")
                workspace += 4 * sum(len(text.encode('utf-8')) for text in (element.text, element.tail) if text)
                if guard:
                    guard(workspace + _PARSER_BYTES)
                if tag in {'key', 'attribute', 'node', 'edge'}:
                    context = stack[-2].get('class') if len(stack) > 1 else None
                    yield tag, element, context
                    if len(stack) > 1:
                        stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                elif tag == 'data' and len(stack) > 1 and _tag(stack[-2]) == 'graph':
                    yield 'graph_data', element, None
                    stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                elif tag == 'keywords' and len(stack) > 1 and _tag(stack[-2]) == 'meta':
                    yield 'native_meta', element, None
                    stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                elif tag in {'nodes', 'edges', 'attributes', 'meta'}:
                    if len(stack) > 1:
                        stack[-2].remove(element)
                    element.clear()
                    workspace = 0
                stack.pop()
        if graphs != 1:
            _error("file_feature", "The file must contain exactly one static graph.")
        receipt.append(reader.digest.hexdigest())
    except ET.ParseError as exc:
        _error("file_format", f"Invalid network XML: {exc}.")
    finally:
        reader.file.close()


def _xml_values(element, definitions, scope, format):
    result = {name: default for (_, kind_scope), (name, _, default) in definitions.items()
              if kind_scope in {scope, 'all'} and default is not None}
    if format == 'graphml':
        values = [(_tag(child), child.get('key'), child.text or '') for child in element if _tag(child) == 'data']
    else:
        values = [('data', child.get('for'), child.get('value', '')) for child in element.iter()
                  if _tag(child) == 'attvalue']
    seen = set()
    for _, key, text in values:
        definition = definitions.get((key, scope)) or definitions.get((key, 'all'))
        if not definition or key in seen:
            _error("file_attribute", "Attributes must reference one declared key per item.")
        seen.add(key)
        name, kind, _ = definition
        result[name] = _convert(text, kind)
    return result


def _read_xml(path, format, maximum, batch_rows, max_memory_mb, *, multigraph=False):
    definitions, ids, nodes, edge_attrs, graph_attrs, receipt = {}, {}, {}, {}, {}, []
    direction, weighted, native_weighted = None, False, None
    budget, owned = _Budget(max_memory_mb), 0
    graph_seen = set()
    wire_edges = set()
    for tag, item, scope in _xml_records(path, maximum, format, receipt,
                                        lambda workspace: budget.check(owned + workspace)):
        if tag == 'native_meta':
            text = item.text or ''
            if text.startswith(_WEIGHTED + '='):
                if native_weighted is not None:
                    _error("file_attribute", "Native weighted metadata must occur at most once.")
                native_weighted = _convert(text[len(_WEIGHTED) + 1:], 'boolean')
        elif tag == 'graph':
            value = item.get('edgedefault' if format == 'graphml' else 'defaultedgetype',
                             None if format == 'graphml' else 'undirected')
            if value not in {'directed', 'undirected'}:
                _error("file_feature", "Mixed graph edge directions are unsupported.")
            direction = value == 'directed'
            for (_, current), (name, _, default) in definitions.items():
                if current in {'graph', 'all'} and default is not None:
                    if name == _WEIGHTED:
                        native_weighted = default
                    else:
                        graph_attrs[name] = default
            if format == 'gexf' and item.get(_WEIGHTED) is not None:
                if native_weighted is not None:
                    _error("file_attribute", "Native weighted metadata must occur at most once.")
                native_weighted = _convert(item.get(_WEIGHTED), 'boolean')
        elif tag in {'key', 'attribute'}:
            key = item.get('id')
            scope = item.get('for', 'all') if format == 'graphml' else scope
            name = item.get('attr.name', key) if format == 'graphml' else item.get('title', key)
            kind = item.get('attr.type', 'string') if format == 'graphml' else item.get('type', 'string')
            scopes = {'node', 'edge', 'graph', 'all'} if format == 'graphml' else {'node', 'edge'}
            if (not key or scope not in scopes or kind not in _TYPES or (key, scope) in definitions or
                    format == 'graphml' and any(previous == key for previous, _ in definitions)):
                _error("file_attribute", "Invalid or duplicate attribute declaration.")
            if name == _IDENTITY:
                if scope != 'node' or kind != 'string':
                    _error("file_attribute", "Native identities require a node string declaration.")
            elif name == _EDGE_IDENTITY:
                if not multigraph:
                    _error("file_feature", "This file preserves separate edge identities; use read_multigraph and explicitly project if needed.")
                if scope != 'edge' or kind != 'string':
                    _error("file_attribute", "Native edge identities require an edge string declaration.")
            elif name == _WEIGHTED:
                if scope != 'graph' or kind not in {'boolean', 'bool'}:
                    _error("file_attribute", "Native weighting requires a graph boolean declaration.")
            else:
                _attributes({name: ''})
            if any(previous_name == name and (current == scope or 'all' in {current, scope})
                   for (_, current), (previous_name, _, _) in definitions.items()):
                _error("file_attribute", "Attribute names must be unique within overlapping scopes.")
            default = next((child.text or '' for child in item if _tag(child) == 'default'), None)
            owned += 1024 + 4 * len((default or '').encode('utf-8'))
            budget.check(owned + _PARSER_BYTES)
            definitions[key, scope] = (name, kind, _convert(default, kind) if default is not None else None)
            if len(definitions) > (388 if multigraph else 387):
                _error("file_attribute", "Too many graph attribute declarations.")
        elif tag == 'graph_data':
            if item.get('key') in graph_seen:
                _error("file_attribute", "Graph data must reference each declared key at most once.")
            graph_seen.add(item.get('key'))
            definition = definitions.get((item.get('key'), 'graph')) or definitions.get((item.get('key'), 'all'))
            if not definition:
                _error("file_attribute", "Unknown graph data key.")
            name, kind, _ = definition
            value = _convert(item.text or '', kind)
            owned += 512 + len(json.dumps(value, ensure_ascii=False).encode('utf-8')) * 4
            budget.check(owned + _PARSER_BYTES)
            if name == _WEIGHTED:
                if not isinstance(value, bool):
                    _error("file_attribute", "OpenEcon weighted metadata must be boolean.")
                native_weighted = value
            else:
                graph_attrs[name] = value
        elif tag == 'node':
            external = item.get('id')
            if external is None or external in ids:
                _error("file_node", "Each node needs one unique ID.")
            attrs = _xml_values(item, definitions, 'node', format)
            identity = attrs.pop(_IDENTITY, None)
            try:
                label = _label(json.loads(identity) if identity is not None else external)
            except (ValueError, TypeError):
                _error("file_node", "Malformed native node identity.")
            if label in nodes:
                _error("file_node", "Two XML IDs map to the same node identity.")
            if format == 'gexf' and identity is None and item.get('label') is not None:
                attrs.setdefault('label', item.get('label'))
            attrs = _attributes(attrs)
            owned += 768 + len(external.encode('utf-8')) * 4 + len(json.dumps(attrs).encode()) * 4 + 256 * len(attrs)
            budget.check(owned + _PARSER_BYTES)
            ids[external], nodes[label] = label, attrs
        elif tag == 'edge':
            # The first pass discovers file weighting without retaining edges.
            weighted |= item.get('weight') is not None or any(
                name == 'weight' and s in {'edge', 'all'} for (_, s), (name, _, _) in definitions.items())
            if multigraph:
                identifier = item.get('id')
                if identifier is None or identifier in wire_edges:
                    _error("file_edge", "Multigraph import requires one unique XML edge ID per edge.")
                _label(identifier)
                owned += 1024 + 8 * len(identifier.encode('utf-8'))
                budget.check(owned + _PARSER_BYTES)
                wire_edges.add(identifier)
    if direction is None:
        _error("file_format", "Missing graph declaration.")

    def edge_value(item):
        u, v = item.get('source'), item.get('target')
        if u not in ids or v not in ids:
            _error("file_node", "Edge endpoint refers to an undeclared node.")
        edge_direction = item.get('directed') if format == 'graphml' else item.get('type')
        if edge_direction is not None:
            current = _convert(edge_direction, 'boolean') if format == 'graphml' else edge_direction == 'directed'
            if format == 'gexf' and edge_direction not in {'directed', 'undirected'} or current != direction:
                _error("file_feature", "Mixed edge directions are unsupported.")
        attrs = _xml_values(item, definitions, 'edge', format)
        identity = attrs.pop(_EDGE_IDENTITY, None)
        weight = attrs.pop('weight', 1.)
        if item.get('weight') is not None:
            weight = float(item.get('weight'))
        elif isinstance(weight, str):
            weight = float(weight)
        weight = _weight(weight)
        attrs = _attributes(attrs)
        pair = (ids[u], ids[v])
        if not direction and not multigraph and _key(pair[0]) > _key(pair[1]):
            pair = pair[::-1]
        identifier = None
        if multigraph:
            try:
                identifier = _label(json.loads(identity) if identity is not None else item.get('id'))
            except (ValueError, TypeError):
                _error("file_edge", "Malformed native edge identity.")
        return pair, weight, attrs, identifier

    if multigraph:
        from openecon._network_multi import _build_multigraph as build_multigraph
        live_bytes = [0]

        def multi_edges():
            for tag, item, _ in _xml_records(path, maximum, format, receipt,
                    lambda workspace: budget.check(owned + live_bytes[0] + workspace)):
                if tag == 'edge':
                    pair, weight, attrs, identifier = edge_value(item)
                    yield dict(edge_id=identifier, source=pair[0], target=pair[1], weight=weight, attributes=attrs)

        budget.check(owned + _PARSER_BYTES + 4096)
        graph = build_multigraph(multi_edges(), weight='weight', attributes='attributes', nodes=list(nodes),
            directed=direction, node_attributes=nodes, graph_attributes=_attributes(graph_attrs),
            batch_rows=batch_rows, max_memory_mb=(budget.limit - owned - _PARSER_BYTES) / 1024**2,
            _live_bytes=live_bytes)
        if len(set(receipt)) != 1:
            _error("file_changed", "The multigraph file changed during its bounded two-pass read.")
        graph._budget.limit = budget.limit
        graph._budget.peak = max(graph._budget.peak + owned + _PARSER_BYTES, budget.peak)
        graph.weighted = weighted if native_weighted is None else native_weighted
        graph._guard(owned + _PARSER_BYTES)
        graph._metadata.update(input_format=format, file_sha256=receipt[0], file_passes=2)
        graph._refresh()
        return graph

    def edges():
        for tag, item, _ in _xml_records(path, maximum, format, receipt,
                                        lambda workspace: budget.check(owned + workspace)):
            if tag != 'edge':
                continue
            pair, weight, _, _ = edge_value(item)
            yield {'source': pair[0], 'target': pair[1], 'weight': weight}

    # Reserve retained parser/attribute state from the graph importer budget.
    budget.check(owned + _PARSER_BYTES + 4096)
    available = (budget.limit - owned - _PARSER_BYTES) / 1024**2
    graph = network(edges(), weight='weight', directed=direction, nodes=list(nodes),
                    batch_rows=batch_rows, max_memory_mb=available)
    if receipt[0] != receipt[1]:
        _error("file_changed", "The input file changed during its two-pass read.")
    graph._budget.limit = budget.limit
    graph._budget.peak = max(graph._budget.peak + owned + _PARSER_BYTES, budget.peak)
    attribute_bytes = 0
    # Discover edge attributes after building topology: every growing retained
    # parser/attribute buffer is then charged against the complete graph too.
    for tag, item, _ in _xml_records(path, maximum, format, receipt,
                                    lambda workspace: graph._guard(owned + attribute_bytes + workspace)):
        if tag != 'edge':
            continue
        pair, weight, attrs, _ = edge_value(item)
        if attrs and weight > 0:
            if pair in edge_attrs and edge_attrs[pair] != attrs:
                _error("file_attribute", "Duplicate edges have conflicting attributes; resolve before import.")
            if pair not in edge_attrs:
                addition = 512 + len(json.dumps(attrs).encode()) * 4 + 256 * len(attrs)
                graph._guard(owned + attribute_bytes + addition + _PARSER_BYTES)
                edge_attrs[pair] = attrs
                attribute_bytes += addition
    if len(set(receipt)) != 1:
        _error("file_changed", "The input file changed during its bounded three-pass read.")
    graph.weighted = weighted if native_weighted is None else native_weighted
    graph_attrs = _attributes(graph_attrs)
    graph._metadata.update(weighted=graph.weighted, input_format=format, file_sha256=receipt[0],
                           max_memory_bytes=budget.limit, file_passes=3)
    graph._guard(2 * (owned + attribute_bytes) + 256 * graph.edge_count +
                 128 * min(graph.edge_count, 65536) + _PARSER_BYTES)
    attach_attributes(graph, nodes=nodes, edges=edge_attrs, graph_attributes=graph_attrs)
    graph._metadata['estimated_import_peak_bytes'] = graph._budget.peak
    return graph


def _pajek_lines(path, maximum, receipt):
    digest, count = hashlib.sha256(), 0
    with open(path, 'rb') as stream:
        while data := stream.readline(131073):
            count += len(data)
            if count > maximum or len(data) > 131072:
                _error("file_size", "Pajek input exceeds the file or line size limit.")
            digest.update(data)
            try:
                line = data.decode('utf-8-sig').strip()
            except UnicodeError:
                _error("file_format", "Pajek files must be UTF-8.")
            if line:
                yield line
    receipt.append(digest.hexdigest())


def _json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate native JSON field')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def _pajek_number(text):
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        _error("file_node", "Pajek vertex numbers must be positive decimal integers.")
    return str(int(text))


def _pajek_edge(parts, lists):
    if len(parts) < 2:
        _error("file_node", "A Pajek edge needs source and target vertex numbers.")
    source = _pajek_number(parts[0])
    targets = [_pajek_number(text) for text in (parts[1:] if lists else parts[1:2])]
    weight, visual, weighted = 1., '', False
    if not lists and len(parts) > 2:
        try:
            weight = float(parts[2])
        except ValueError:
            # Preserve common Pajek drawing properties as one opaque bounded
            # scalar. They do not become edge weights or rendering commands.
            if parts[2].lower() not in {'c', 'w', 'p', 'l', 's', 'a', 'ap', 'lc', 'lp', 'lr', 'la'}:
                _error("file_format", "Unsupported or malformed Pajek edge weight/visual property.")
            visual = ' '.join(parts[2:])
        else:
            weighted = True
            visual = ' '.join(parts[3:])
    return source, targets, _weight(weight), _scalar(visual), weighted


def _pajek_records(path, maximum, receipt, labels, direction, guard):
    section, pending = None, None
    for line in _pajek_lines(path, maximum, receipt):
        guard(_PARSER_BYTES + 4 * len(line.encode('utf-8')))
        if line.startswith('% openecon-edge '):
            if pending is not None or section not in {'*edges', '*arcs', '*edgeslist', '*arcslist'}:
                _error("file_attribute", "Native edge attributes must occur once immediately before an edge record.")
            pending = _attributes(_json(line[len('% openecon-edge '):]))
            if 'weight' in pending:
                _error("file_attribute", "Native edge attributes cannot replace the actual edge weight.")
        elif line.startswith('%'):
            continue
        elif line.startswith('*'):
            if pending is not None:
                _error("file_attribute", "Orphan native Pajek edge attributes.")
            section = line.lower().split()[0]
        elif section in {'*edges', '*arcs', '*edgeslist', '*arcslist'}:
            source, targets, weight, visual, _ = _pajek_edge(shlex.split(line), section.endswith('list'))
            if source not in labels or any(target not in labels for target in targets):
                _error("file_node", "Pajek edge endpoint is outside the declared vertex set.")
            attrs = dict(pending or {})
            if visual:
                if 'pajek_visual' in attrs and attrs['pajek_visual'] != visual:
                    _error("file_attribute", "Native and inline Pajek visual attributes disagree.")
                attrs['pajek_visual'] = visual
            attrs = _attributes(attrs)
            for target in targets:
                pair = labels[source], labels[target]
                if not direction and _key(pair[0]) > _key(pair[1]):
                    pair = pair[::-1]
                yield pair, weight, attrs
            pending = None
    if pending is not None:
        _error("file_attribute", "Orphan native Pajek edge attributes at end of file.")


def _read_pajek(path, maximum, batch_rows, max_memory_mb):
    nodes, ids, native, receipt, edge_attrs = {}, {}, {}, [], {}
    section, expected, direction, weighted, native_weighted = None, None, None, False, None
    budget, owned = _Budget(max_memory_mb), 0
    for line in _pajek_lines(path, maximum, receipt):
        budget.check(owned + _PARSER_BYTES + 4 * len(line.encode('utf-8')))
        if line.startswith('% openecon-node '):
            number, payload = line[len('% openecon-node '):].split(' ', 1)
            number = _pajek_number(number)
            extra = _json(payload)
            if (number in native or not isinstance(extra, dict) or 'identity' not in extra or
                    set(extra) - {'identity', 'attributes'}):
                _error("file_node", "Native Pajek identities must be unique bounded node records.")
            extra = {'identity': _label(extra['identity']), 'attributes': _attributes(extra.get('attributes', {}))}
            owned += 1024 + 4 * len(payload.encode('utf-8'))
            budget.check(owned + _PARSER_BYTES)
            native[number] = extra
        elif line.startswith('% openecon-weighted '):
            if native_weighted is not None or len(line.split()) != 3:
                _error("file_attribute", "Native Pajek weighted metadata must occur exactly once.")
            native_weighted = _convert(line.split()[-1], 'boolean')
        elif line.startswith('% openecon-edge '):
            _attributes(_json(line[len('% openecon-edge '):]))
        elif line.startswith('%'):
            continue
        elif line.startswith('*'):
            parts = line.lower().split()
            if parts[0] == '*vertices' and expected is None:
                if len(parts) != 2:
                    _error("file_feature", "Pajek two-mode vertex headers require explicit projection.")
                expected = _integer(int(parts[1]), 'vertices', zero=True)
                section = 'nodes'
            elif parts[0] in {'*edges', '*arcs', '*edgeslist', '*arcslist'} and expected is not None:
                current = parts[0] in {'*arcs', '*arcslist'}
                if direction is not None and direction != current or len(parts) != 1:
                    _error("file_feature", "Mixed or relation-indexed Pajek graphs require explicit conversion.")
                direction, section = current, parts[0]
            else:
                _error("file_feature", "Pajek supports static Vertices and Edges/Arcs (including lists), without matrices or partitions.")
        else:
            parts = shlex.split(line)
            if section == 'nodes':
                if len(parts) < 2:
                    _error("file_node", "Pajek vertices require a number and label.")
                number = _pajek_number(parts[0])
                if number in ids or int(number) > expected:
                    _error("file_node", "Invalid or duplicate Pajek vertex.")
                owned += 1024 + len(line.encode('utf-8')) * 4
                budget.check(owned + _PARSER_BYTES)
                ids[number] = parts[1], _scalar(' '.join(parts[2:]))
            elif section in {'*edges', '*arcs', '*edgeslist', '*arcslist'}:
                _, _, _, _, explicit = _pajek_edge(parts, section.endswith('list'))
                weighted |= explicit
            else:
                _error("file_format", "Pajek records must follow a section header.")
    if expected is None or len(ids) != expected:
        _error("file_node", "Vertex count does not match the Pajek header.")
    labels = {}
    for number, (label, visual) in ids.items():
        extra = native.get(number, {})
        identity = _label(extra.get('identity', number))
        if identity in nodes:
            _error("file_node", "Pajek labels are ambiguous; use unique labels or native identity metadata.")
        attrs = _attributes(extra.get('attributes', {}))
        if not extra:
            attrs['label'] = _scalar(label)
        if visual:
            attrs.setdefault('pajek_visual', visual)
        attrs = _attributes(attrs)
        owned += 1024 + 4 * len(json.dumps(attrs).encode('utf-8'))
        budget.check(owned + _PARSER_BYTES)
        labels[number], nodes[identity] = identity, attrs
    if set(native) - set(ids):
        _error("file_node", "Native Pajek metadata refers to an unknown vertex.")
    direction = bool(direction)

    def edges():
        for pair, weight, _ in _pajek_records(path, maximum, receipt, labels, direction,
                                             lambda workspace: budget.check(owned + workspace)):
            yield {'source': pair[0], 'target': pair[1], 'weight': weight}

    budget.check(owned + _PARSER_BYTES + 4096)
    graph = network(edges(), nodes=list(nodes), weight='weight', directed=direction,
                    max_memory_mb=(budget.limit - owned - _PARSER_BYTES) / 1024**2,
                    batch_rows=batch_rows)
    if receipt[0] != receipt[1]:
        _error("file_changed", "The Pajek file changed during reading.")
    graph._budget.limit = budget.limit
    graph._budget.peak = max(graph._budget.peak + owned + _PARSER_BYTES, budget.peak)
    attribute_bytes = 0
    for pair, weight, attrs in _pajek_records(path, maximum, receipt, labels, direction,
                                             lambda workspace: graph._guard(owned + attribute_bytes + workspace)):
        if attrs and weight > 0:
            if pair in edge_attrs and edge_attrs[pair] != attrs:
                _error("file_attribute", "Duplicate Pajek edges have conflicting attributes.")
            if pair not in edge_attrs:
                addition = 512 + len(json.dumps(attrs).encode()) * 4 + 256 * len(attrs)
                graph._guard(owned + attribute_bytes + addition + _PARSER_BYTES)
                edge_attrs[pair] = attrs
                attribute_bytes += addition
    if len(set(receipt)) != 1:
        _error("file_changed", "The Pajek file changed during its bounded three-pass read.")
    graph.weighted = weighted if native_weighted is None else native_weighted
    graph._metadata.update(weighted=graph.weighted, input_format='pajek', file_sha256=receipt[0],
                           max_memory_bytes=budget.limit, file_passes=3)
    graph._guard(2 * (owned + attribute_bytes) + 256 * graph.edge_count +
                 128 * min(graph.edge_count, 65536) + _PARSER_BYTES)
    attach_attributes(graph, nodes=nodes, edges=edge_attrs)
    graph._metadata['estimated_import_peak_bytes'] = graph._budget.peak
    return graph


def read_network(path, *, format=None, max_memory_mb=256, batch_rows=65536, max_file_mb=512):
    """Read one static graph, streaming records under explicit file/memory limits.

    XML requires UTF-8 without DTD/entities. Scalar attributes, isolates and
    typed native IDs round-trip. Unsupported dynamic/mixed graph features fail.
    """
    path = Path(path)
    format = _format(path, format)
    maximum = _Budget(max_file_mb).limit
    _integer(batch_rows, 'batch_rows')
    _Budget(max_memory_mb)
    if path.stat().st_size > maximum:
        _error("file_size", "Input exceeds max_file_mb.")
    try:
        return (_read_pajek(path, maximum, batch_rows, max_memory_mb) if format == 'pajek' else
                _read_xml(path, format, maximum, batch_rows, max_memory_mb))
    except AnalysisError:
        raise
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        _error("file_format", f"Malformed static graph file: {str(exc)[:100]}.")


def _kind(value):
    return 'boolean' if isinstance(value, bool) else 'long' if isinstance(value, int) else 'double' if isinstance(value, float) else 'string'


def _text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def _xml_text(element):
    # XML normalizes literal CR in text nodes. Numeric references preserve the
    # caller's scalar exactly, as ElementTree already does for XML attributes.
    return ET.tostring(element, encoding='unicode').replace('\r', '&#13;')


def _edge_records(graph):
    pairs, values = ((graph._endpoints, graph._weights) if getattr(graph, '_is_multigraph', False)
                     else (graph._edges.indices(), graph._edges.values()))
    for start in range(0, graph.edge_count, 65536):
        u, v = pairs[:, start:start + 65536].tolist()
        yield from zip(u, v, values[start:start + 65536].tolist())


def _declarations(graph):
    fields = {('node', _IDENTITY): 'string', ('edge', 'weight'): 'double', ('graph', _WEIGHTED): 'boolean'}
    multigraph = getattr(graph, '_is_multigraph', False)
    if multigraph:
        fields['edge', _EDGE_IDENTITY] = 'string'
    for scope, records in [('node', graph._node_attributes.values()), ('edge', graph._edge_attributes.values()),
                           ('graph', [graph._graph_attributes])]:
        for attrs in records:
            for name, value in attrs.items():
                kind = _kind(value)
                previous = fields.get((scope, name))
                if previous and previous != kind:
                    _error("file_attribute", "An attribute must have the same scalar type across items for XML export.")
                fields[scope, name] = kind
                if len(fields) > (388 if multigraph else 387):
                    _error("file_attribute", "XML export supports at most 387 unique scoped scalar fields.")
                graph._guard(2048 * len(fields))
    return {pair: (f'a{i}', kind) for i, (pair, kind) in enumerate(fields.items())}


def _write_xml(graph, stream, format):
    declarations = _declarations(graph)
    directed = 'directed' if graph.directed else 'undirected'
    stream.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    if format == 'graphml':
        stream.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for (scope, name), (key, kind) in declarations.items():
            stream.write(_xml_text(ET.Element('key', {'id': key, 'for': scope, 'attr.name': name, 'attr.type': kind})) + '\n')
        stream.write(f'<graph id="G" edgedefault="{directed}">\n')
    else:
        stream.write(f'<gexf xmlns="http://gexf.net/1.3" version="1.3"><meta><keywords>{_WEIGHTED}={str(graph.weighted).lower()}</keywords></meta><graph mode="static" defaultedgetype="{directed}">\n')
        if graph._graph_attributes:
            _error("file_feature", "GEXF has no scalar graph attribute declarations; use GraphML to preserve graph attributes.")
        for scope in ('node', 'edge'):
            parent = ET.Element('attributes', {'class': scope})
            for (current, name), (key, kind) in declarations.items():
                if current == scope and not (scope == 'edge' and name == 'weight'):
                    ET.SubElement(parent, 'attribute', {'id': key, 'title': name, 'type': kind})
            stream.write(_xml_text(parent) + '\n')

    def values(element, scope, attrs):
        parent = element if format == 'graphml' else ET.SubElement(element, 'attvalues')
        for name, value in attrs.items():
            key = declarations[scope, name][0]
            if format == 'graphml':
                ET.SubElement(parent, 'data', {'key': key}).text = _text(value)
            else:
                ET.SubElement(parent, 'attvalue', {'for': key, 'value': _text(value)})

    if format == 'graphml':
        graph._guard(2048 * len(declarations) + 2 * _attribute_bytes(graph._graph_attributes) + 131072)
        element = ET.Element('graph')
        values(element, 'graph', {_WEIGHTED: graph.weighted, **graph._graph_attributes})
        for child in element:
            stream.write(_xml_text(child) + '\n')
    else:
        stream.write('<nodes>\n')
    for i, label in enumerate(graph._labels):
        attrs = graph._node_attributes.get(label, {})
        graph._guard(2048 * len(declarations) + 128 * min(graph.edge_count, 65536) +
                     2 * _attribute_bytes(attrs) + 131072)
        element = ET.Element('node', {'id': f'n{i}'})
        if format == 'gexf':
            element.set('label', str(attrs.get('label', label)))
        values(element, 'node', {_IDENTITY: json.dumps(label, ensure_ascii=False), **attrs})
        stream.write(_xml_text(element) + '\n')
    if format == 'gexf':
        stream.write('</nodes><edges>\n')
    for k, (u, v, weight) in enumerate(_edge_records(graph)):
        label_u, label_v = graph._labels[u], graph._labels[v]
        pair = (label_u, label_v) if graph.directed or _key(label_u) <= _key(label_v) else (label_v, label_u)
        multigraph = getattr(graph, '_is_multigraph', False)
        attrs = graph._edge_attributes.get(graph._edge_ids[k] if multigraph else pair, {})
        if multigraph:
            attrs = {_EDGE_IDENTITY: json.dumps(graph._edge_ids[k], ensure_ascii=False), **attrs}
        graph._guard(2048 * len(declarations) + 128 * min(graph.edge_count, 65536) +
                     2 * _attribute_bytes(attrs) + 131072)
        element = ET.Element('edge', {'id': f'e{k}', 'source': f'n{u}', 'target': f'n{v}'})
        if format == 'gexf':
            element.set('weight', repr(weight))
            values(element, 'edge', attrs)
        else:
            values(element, 'edge', {'weight': weight, **attrs})
        stream.write(_xml_text(element) + '\n')
    stream.write('</graph></graphml>\n' if format == 'graphml' else '</edges></graph></gexf>\n')


def _write_pajek(graph, stream):
    if graph._graph_attributes:
        _error("file_feature", "Use GraphML to preserve scalar graph attributes.")
    graph._guard(128 * min(graph.edge_count, 65536) + 524288)

    def native_line(prefix, payload):
        pieces, count = [prefix], len(prefix.encode('utf-8')) + 1
        for piece in json.JSONEncoder(ensure_ascii=False).iterencode(payload):
            count += len(piece.encode('utf-8'))
            if count > 131072:
                _error("file_size", "Native Pajek metadata exceeds its readable 128 KiB line limit; use GraphML or GEXF.")
            pieces.append(piece)
        stream.write(''.join(pieces) + '\n')
    stream.write(f'% openecon-weighted {str(graph.weighted).lower()}\n*Vertices {graph.node_count}\n')
    for i, label in enumerate(graph._labels, 1):
        attrs = graph._node_attributes.get(label, {})
        payload = {'identity': label, 'attributes': attrs}
        native_line(f'% openecon-node {i} ', payload)
        stream.write(f'{i} {json.dumps(str(attrs.get("label", label)), ensure_ascii=False)}\n')
    stream.write('*Arcs\n' if graph.directed else '*Edges\n')
    for u, v, weight in _edge_records(graph):
        left, right = graph._labels[u], graph._labels[v]
        pair = (left, right) if graph.directed or _key(left) <= _key(right) else (right, left)
        attrs = graph._edge_attributes.get(pair)
        if attrs:
            native_line('% openecon-edge ', attrs)
        stream.write(f'{u+1} {v+1} {weight!r}\n')


def write_network(graph, path, *, format=None, overwrite=False):
    """Atomically export the complete graph; existing files require overwrite=True."""
    if not isinstance(overwrite, bool):
        _error("invalid_option", "overwrite must be boolean.")
    path = Path(path)
    format = _format(path, format)
    if getattr(graph, '_is_multigraph', False) and format == 'pajek':
        _error("file_feature", "Pajek edge-ID interchange is unavailable; use GraphML/GEXF or explicitly project a simple graph.")
    graph._guard(128 * min(graph.edge_count, 65536) + 32768)
    fd, temporary = tempfile.mkstemp(prefix='.openecon-network-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            if format == 'pajek':
                _write_pajek(graph, stream)
            else:
                _write_xml(graph, stream, format)
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
