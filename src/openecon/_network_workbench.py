"""Immutable, code-first sparse graph editing and attribute-table operations.

Edits build a new sparse snapshot, never a dense adjacency matrix. The source
and prospective snapshot both count toward the source's memory allowance.
Callable filters execute Python supplied by the caller; chart filters are a
separate, serializable comparison language and never evaluate code strings.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import math

import pandas as pd

from openecon.frame import as_frame
from openecon.networks import _Budget, _error, _key, _label, _weight, network


def _labels(values, name):
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        _error("invalid_option", f"{name} must be an iterable of exact node IDs.")
    return (_label(value) for value in values)


def _pair(graph, pair):
    if not isinstance(pair, tuple) or len(pair) != 2:
        _error("invalid_option", "Edge IDs must be (source, target) tuples.")
    u, v = map(_label, pair)
    if u not in graph._index or v not in graph._index:
        _error("unknown_node", "An edge operation refers to an unknown node.")
    return (u, v) if graph.directed or _key(u) <= _key(v) else (v, u)


def _records(graph):
    from openecon._network_io import _edge_records
    for u, v, weight in _edge_records(graph):
        pair = _pair(graph, (graph._labels[u], graph._labels[v]))
        yield pair, weight


def _build(graph, records, labels, nodes, edges, operation):
    # The bounded importer reserves its own temporaries out of the remaining
    # allowance, with the source still resident throughout construction.
    workspace = (1024 * len(labels) + 512 * len(nodes)
                 + 512 * max(len(edges), len(graph._edge_attributes)) + 4096)
    graph._budget.check(graph._base_bytes + workspace + 4096)
    remaining = graph._budget.limit - graph._base_bytes - workspace - 4096
    if remaining <= 4096:
        _error("memory_budget", "An immutable graph edit needs room for both source and result.")
    result = network(records, nodes=labels, weight="weight", directed=graph.directed,
                     max_memory_mb=remaining / 1024**2)
    graph._budget.check(graph._base_bytes + result._base_bytes + workspace + 4096)
    result._budget = _Budget(graph._budget.limit / 1024**2)
    from openecon._network_io import _attribute_bytes
    attributes = sum(_attribute_bytes(value) for value in nodes.values())
    attributes += sum(_attribute_bytes(value) + 64 for value in edges.values())
    attributes += _attribute_bytes(graph._graph_attributes)
    graph._budget.check(graph._base_bytes + result._base_bytes + workspace + attributes * 2 + 256 * result.edge_count + 8192)
    result = result.with_attributes(nodes=nodes, edges=edges, graph_attributes=graph._graph_attributes)
    result.weighted = graph.weighted
    result._metadata.update(weighted=result.weighted, max_memory_bytes=graph._budget.limit,
                            derived_from=operation, parent_node_count=graph.node_count,
                            parent_edge_count=graph.edge_count,
                            memory_scope="source plus new sparse graph and planned edit buffers; excludes caller input and process RSS")
    return result


def nodes_table(graph, *, attributes=True):
    """Exact IDs and full-graph degrees; attribute columns use ``attr.NAME``."""
    if not isinstance(attributes, bool):
        _error("invalid_option", "attributes must be boolean.")
    fields = set()
    if attributes:
        for values in graph._node_attributes.values():
            fields.update(values)
            graph._guard(512 * len(fields))
    graph._guard((512 + 128 * len(fields)) * graph.node_count + 4096)
    result = graph.degree()
    for field in sorted(fields):
        result["attr." + field] = pd.Series([graph._node_attributes.get(label, {}).get(field)
                                             for label in graph._labels], dtype=object)
    result.attrs.update(kind="network_nodes", network=graph.metadata)
    return result


def edges_table(graph, *, attributes=True):
    """Canonical aggregate edges; scalar attributes use ``attr.NAME`` columns."""
    if not isinstance(attributes, bool):
        _error("invalid_option", "attributes must be boolean.")
    fields = set()
    if attributes:
        for values in graph._edge_attributes.values():
            fields.update(values)
            graph._guard(512 * len(fields))
    graph._guard((768 + 128 * len(fields)) * graph.edge_count + 4096)
    columns = {"source": [], "target": [], "weight": [], **{"attr." + field: [] for field in sorted(fields)}}
    for pair, weight in _records(graph):
        columns["source"].append(pair[0])
        columns["target"].append(pair[1])
        columns["weight"].append(weight)
        attrs = graph._edge_attributes.get(pair, {})
        for field in fields:
            columns["attr." + field].append(attrs.get(field))
    result = as_frame(pd.DataFrame({key: pd.Series(value, dtype=object)
                                   if key != "weight" else pd.Series(value, dtype=float)
                                   for key, value in columns.items()}))
    result.attrs.update(kind="network_edges", network=graph.metadata)
    return result


def edit_nodes(graph, *, add=(), remove=(), rename=None):
    """Apply simultaneous ID renaming, removals and additions; retain isolates."""
    graph._guard(1024 * graph.node_count + 128 * min(graph.edge_count, 65536))
    removed = set()
    for label in _labels(remove, "remove"):
        if label not in graph._index:
            _error("unknown_node", "Node removal refers to an unknown node.")
        removed.add(label)
        graph._guard(1024 * (graph.node_count + len(removed)))
    if rename is not None and not isinstance(rename, Mapping):
        _error("invalid_option", "rename must map exact old IDs to exact new IDs.")
    renamed = {}
    for old, new in (rename or {}).items():
        old, new = _label(old), _label(new)
        if old not in graph._index or old in removed:
            _error("unknown_node", "Node renaming refers to an unknown or removed node.")
        renamed[old] = new
        graph._guard(1024 * (graph.node_count + len(removed) + len(renamed)))
    labels, unique = [], set()
    for old in graph._labels:
        if old in removed:
            continue
        new = renamed.get(old, old)
        if new in unique:
            _error("invalid_label", "Renaming cannot merge distinct node identities.")
        labels.append(new)
        unique.add(new)
    for label in _labels(add, "add"):
        if label in unique:
            _error("invalid_label", "Added node IDs must be distinct from retained and added nodes.")
        graph._guard(1024 * (graph.node_count + len(labels) + len(renamed)))
        labels.append(label)
        unique.add(label)
    node_attrs = {renamed.get(label, label): attrs for label, attrs in graph._node_attributes.items()
                  if label not in removed}
    edge_attrs = {}
    for (u, v), attrs in graph._edge_attributes.items():
        if u in removed or v in removed:
            continue
        u, v = renamed.get(u, u), renamed.get(v, v)
        edge_attrs[(u, v) if graph.directed or _key(u) <= _key(v) else (v, u)] = attrs
    def records():
        for (u, v), weight in _records(graph):
            if u not in removed and v not in removed:
                yield {"source": renamed.get(u, u), "target": renamed.get(v, v), "weight": weight}
    return _build(graph, records(), labels, node_attrs, edge_attrs, "immutable node edits")


def edit_edges(graph, *, add=(), remove=(), weights=None):
    """Add positive weights cumulatively, remove dyads or replace their weights.

    Additions are mapping records with source, target and optional weight/attrs.
    Endpoints must already exist. A zero replacement removes an existing edge.
    The graph remains a simple aggregate graph, not a parallel-edge multigraph.
    """
    from openecon._network_io import _attributes
    if not isinstance(add, Iterable) or isinstance(add, (str, bytes, Mapping)):
        _error("invalid_option", "add must be an iterable of edge records.")
    removed, additions, addition_attrs, replacements = set(), {}, {}, {}
    if weights is not None and not isinstance(weights, Mapping):
        _error("invalid_option", "weights must map existing dyad tuples to replacement weights.")
    if not isinstance(remove, Iterable) or isinstance(remove, (str, bytes, Mapping)):
        _error("invalid_option", "remove must be an iterable of dyad tuples.")
    for pair in remove:
        removed.add(_pair(graph, pair))
        graph._guard(1024 * len(removed))
    for pair, value in (weights or {}).items():
        pair = _pair(graph, pair)
        if pair in removed or pair in replacements:
            _error("invalid_option", "A dyad cannot have conflicting removal/replacement operations.")
        replacements[pair] = _weight(value)
        graph._guard(1024 * (len(removed) + len(replacements)))
    for row in add:
        if not isinstance(row, Mapping) or not {"source", "target"} <= set(row) or set(row) - {"source", "target", "weight", "attrs"}:
            _error("invalid_option", "Added edge records need source/target and optional weight/attrs.")
        pair = _pair(graph, (row["source"], row["target"]))
        if pair in removed or pair in replacements:
            _error("invalid_option", "A dyad cannot have conflicting addition/removal/replacement operations.")
        graph._guard(2048 * (len(removed) + len(replacements) + len(additions) + 1))
        total = additions.get(pair, 0.) + _weight(row.get("weight", 1.))
        if not math.isfinite(total):
            _error("precision", "Added aggregate weights overflow float64.")
        additions[pair] = total
        if "attrs" in row:
            attrs = _attributes(dict(row["attrs"])) if isinstance(row["attrs"], Mapping) else _attributes(row["attrs"])
            if "weight" in attrs:
                _error("file_attribute", "Edge attribute weight is reserved for the actual edge weight.")
            if pair in addition_attrs and addition_attrs[pair] != attrs:
                _error("file_attribute", "Duplicate added dyads cannot carry conflicting attributes.")
            addition_attrs[pair] = attrs
    needed = removed | set(replacements)
    present = set()
    for pair, _ in _records(graph):
        if pair in needed:
            present.add(pair)
    if present != needed:
        _error("unknown_edge", "Removal or replacement refers to an edge outside the graph.")
    graph._guard(2048 * (len(removed) + len(replacements) + len(additions)) + 512 * len(graph._edge_attributes))
    edge_attrs = {pair: attrs for pair, attrs in graph._edge_attributes.items()
                  if pair not in removed and replacements.get(pair, 1.) > 0}
    edge_attrs.update({pair: attrs for pair, attrs in addition_attrs.items() if additions[pair] > 0})
    def records():
        for (u, v), value in _records(graph):
            pair = (u, v)
            if pair not in removed:
                yield {"source": u, "target": v, "weight": replacements.get(pair, value)}
        for (u, v), value in additions.items():
            yield {"source": u, "target": v, "weight": value}
    result = _build(graph, records(), graph._labels, graph._node_attributes, edge_attrs, "immutable edge edits")
    if additions or replacements:
        result.weighted = True
        result._metadata["weighted"] = True
    return result


def update_attributes(graph, *, nodes=None, edges=None, graph_attributes=None):
    """Merge scalar updates into a fresh snapshot; never mutate existing values."""
    if any(value is not None and not isinstance(value, Mapping) for value in (nodes, edges, graph_attributes)):
        _error("invalid_option", "Attribute updates must be mappings.")
    graph._guard(2 * graph._metadata.get("attribute_storage_bytes", 0) + 4096)
    node_attrs = dict(graph._node_attributes) if nodes else graph._node_attributes
    edge_attrs = dict(graph._edge_attributes) if edges else graph._edge_attributes
    overall = graph._graph_attributes
    from openecon._network_io import _attributes, _attribute_bytes
    estimated = 2 * graph._metadata.get("attribute_storage_bytes", 0)
    for key, values in (nodes or {}).items():
        key = _label(key)
        if key not in graph._index or not isinstance(values, Mapping):
            _error("invalid_option", "Node updates need existing IDs and scalar dictionaries.")
        if len(values) > 128:
            _error("file_attribute", "Each node accepts at most 128 scalar attributes.")
        estimated += _attribute_bytes(dict(values)) + 512
        graph._guard(estimated)
        node_attrs[key] = _attributes({**node_attrs.get(key, {}), **values})
    for key, values in (edges or {}).items():
        key = _pair(graph, key)
        if not isinstance(values, Mapping):
            _error("invalid_option", "Edge updates need scalar dictionaries.")
        if len(values) > 128:
            _error("file_attribute", "Each edge accepts at most 128 scalar attributes.")
        estimated += _attribute_bytes(dict(values)) + 512
        graph._guard(estimated)
        edge_attrs[key] = _attributes({**edge_attrs.get(key, {}), **values})
    if graph_attributes:
        if len(graph_attributes) > 128:
            _error("file_attribute", "Graph accepts at most 128 scalar attributes.")
        graph._guard(estimated + _attribute_bytes(dict(graph_attributes)))
        overall = _attributes({**overall, **graph_attributes})
    return graph.with_attributes(nodes=node_attrs, edges=edge_attrs, graph_attributes=overall)


def attribute_columns(graph, *, scope, rename=None, drop=()):
    """Simultaneously rename or remove attribute columns, with collision errors."""
    if scope not in {"nodes", "edges", "graph"} or rename is not None and not isinstance(rename, Mapping):
        _error("invalid_option", "scope must be nodes/edges/graph; rename must be a mapping.")
    from openecon._network_io import _attributes
    if rename is not None:
        graph._guard(2048 * len(rename) + 4096)
    renamed = dict(rename or {})
    if isinstance(drop, (str, bytes)):
        dropped = {drop}
    elif isinstance(drop, Iterable) and not isinstance(drop, Mapping):
        dropped = set()
        for name in drop:
            _attributes({name: ""})
            graph._guard(2048 * (len(renamed) + len(dropped) + 1))
            dropped.add(name)
    else:
        _error("invalid_option", "drop must be an attribute name or an iterable of names.")
    for name in renamed:
        _attributes({name: ""})
        _attributes({renamed[name]: ""})
    for name in dropped:
        _attributes({name: ""})
    graph._guard(2 * graph._metadata.get("attribute_storage_bytes", 0) + 4096)
    records = (graph._node_attributes if scope == "nodes" else graph._edge_attributes
               if scope == "edges" else {None: graph._graph_attributes})
    available = set()
    for attrs in records.values():
        available.update(attrs)
        graph._guard(512 * len(available) + 4096)
    destinations = set()
    for name in available - dropped:
        new = renamed.get(name, name)
        if new in destinations:
            _error("file_attribute", "Attribute renaming cannot merge distinct columns.")
        destinations.add(new)
    result = {}
    for key, attrs in records.items():
        copied = {}
        for name, value in attrs.items():
            if name in dropped:
                continue
            new = renamed.get(name, name)
            if new in copied:
                _error("file_attribute", "Attribute renaming cannot merge distinct columns.")
            copied[new] = value
        result[key] = copied
    return graph.with_attributes(**({"graph_attributes": result.get(None, {})} if scope == "graph" else {scope: result}))


def _equal(left, right):
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    if isinstance(left, str) != isinstance(right, str):
        return False
    return left == right


def _field(row, field):
    if field.startswith("attrs.") or field.startswith("attr."):
        return row["attrs"].get(field.split(".", 1)[1])
    return row.get(field, row["attrs"].get(field))


def _predicate(value):
    if value is None:
        return lambda row: True
    if callable(value):
        def test(row):
            result = value(deepcopy(row))
            if not isinstance(result, bool):
                _error("invalid_option", "A Python filter must return a boolean for every row.")
            return result
        return test
    items = [value] if isinstance(value, Mapping) else value
    if not isinstance(items, (list, tuple)) or len(items) > 100:
        _error("invalid_option", "Filters need a callable or at most 100 field/op/value predicates.")
    predicates = []
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"field", "op", "value"} or not isinstance(item["field"], str) or not 1 <= len(item["field"]) <= 200:
            _error("invalid_option", "Each predicate requires field, op and value.")
        op, threshold = item["op"], item["value"]
        if op not in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"}:
            _error("invalid_option", "Unsupported filter comparison operator.")
        if op in {"in", "not_in"}:
            if not isinstance(threshold, (list, tuple, set, frozenset)) or len(threshold) > 10000:
                _error("invalid_option", "Membership predicates need at most 10,000 explicit scalar values.")
            threshold = {_token(candidate) for candidate in threshold}
        else:
            _token(threshold)
            if op in {"gt", "gte", "lt", "lte"} and (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)):
                _error("invalid_option", "Ordered predicates require a finite numeric threshold.")
        predicates.append((item["field"], op, deepcopy(threshold)))
    def test(row):
        for field, op, threshold in predicates:
            found = _field(row, field)
            if op in {"eq", "ne"}:
                passed = _equal(found, threshold)
                passed = passed if op == "eq" else not passed
            elif op in {"in", "not_in"}:
                passed = _token(found) in threshold
                passed = passed if op == "in" else not passed
            else:
                if not isinstance(found, (int, float)) or isinstance(found, bool) or not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                    passed = False
                else:
                    passed = {"gt": lambda: found > threshold, "gte": lambda: found >= threshold,
                              "lt": lambda: found < threshold, "lte": lambda: found <= threshold}[op]()
            if not passed:
                return False
        return True
    return test


def _token(value):
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int) and value.bit_length() <= 256:
        return ("number", value)
    if isinstance(value, float) and math.isfinite(value):
        return ("number", value)
    if isinstance(value, str) and len(value) <= 16_384:
        return ("string", value)
    _error("invalid_option", "Filter values must be bounded scalar strings, numbers, booleans or None.")


def filter_graph(graph, *, nodes=None, edges=None):
    """Return the induced node selection followed by an edge predicate.

    Fields are node/degree/strength and source/target/weight, or ``attrs.NAME``.
    Missing attributes compare as None; ordered comparisons exclude missing,
    string and boolean values. All predicates are combined with logical AND.
    """
    node_test, edge_test = _predicate(nodes), _predicate(edges)
    graph._guard(1024 * graph.node_count + 512 * len(graph._edge_attributes))
    out, incoming, out_strength, in_strength = graph._degrees()
    degree_values = memoryview((out + incoming).numpy())
    strength = out_strength + in_strength
    import torch
    if not bool(torch.isfinite(strength).all()):
        _error("precision", "A node strength exceeds float64 range.")
    strength_values = memoryview(strength.numpy())
    selected = set()
    for index, label in enumerate(graph._labels):
        if node_test({"node": label, "degree": degree_values[index],
                      "strength": strength_values[index],
                      "attrs": graph._node_attributes.get(label, {})}):
            selected.add(label)
    kept_attrs = {}
    def records():
        for (u, v), weight in _records(graph):
            if u in selected and v in selected and edge_test({"source": u, "target": v, "weight": weight,
                                                             "attrs": graph._edge_attributes.get((u, v), {})}):
                if (u, v) in graph._edge_attributes:
                    kept_attrs[(u, v)] = graph._edge_attributes[(u, v)]
                yield {"source": u, "target": v, "weight": weight}
    return _build(graph, records(), [label for label in graph._labels if label in selected],
                  {label: attrs for label, attrs in graph._node_attributes.items() if label in selected},
                  kept_attrs, "node and edge filters")


def with_positions(graph, positions, *, fixed=True):
    """Save finite layout coordinates as scalar attributes keyed by exact IDs."""
    if not isinstance(positions, Mapping) or not isinstance(fixed, bool):
        _error("invalid_option", "positions must be a mapping; fixed must be boolean.")
    updates = {}
    for label, item in positions.items():
        label = _label(label)
        if label not in graph._index:
            _error("unknown_node", "Position refers to an unknown exact node identity.")
        if isinstance(item, (list, tuple)) and len(item) == 2:
            item = {"x": item[0], "y": item[1], "fixed": fixed}
        if not isinstance(item, Mapping) or not {"x", "y"} <= set(item) or set(item) - {"x", "y", "fixed"}:
            _error("invalid_option", "Each position needs x/y and optional fixed boolean.")
        point = {}
        for key in ("x", "y"):
            number = item[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)) or abs(number) > 1e9:
                _error("invalid_option", "Position coordinates must be finite and lie between -1e9 and 1e9.")
            point[key] = float(number)
        point["fixed"] = item.get("fixed", fixed)
        if not isinstance(point["fixed"], bool):
            _error("invalid_option", "A position's fixed field must be boolean.")
        updates[label] = point
        graph._guard(2048 * len(updates))
    return update_attributes(graph, nodes=updates)


def from_plot_data(value, *, max_memory_mb=256):
    """Rebuild a displayed view using exact identities, retaining subset provenance.

    A sampled view contains only its saved subgraph. This does not reconstruct
    omitted nodes/edges, omitted snapshots or original import row counters.
    """
    from openecon_charts.network import validate_network
    budget = _Budget(max_memory_mb)
    if isinstance(value, Mapping) and isinstance(value.get("nodes"), list) and isinstance(value.get("edges"), list):
        budget.check(1024 * len(value["nodes"]) + 768 * len(value["edges"]) + 4096)
    data = validate_network(value)
    if "frames" in data:
        _error("invalid_option", "Select a saved timeline frame before importing a Network.")
    labels, unique, identities, node_attrs = [], set(), {}, {}
    for row in data["nodes"]:
        if "identity" not in row:
            _error("invalid_label", "Saved nodes require explicit typed identities; display labels are not IDs.")
        identity = row["identity"]
        label = _label(int(identity["value"]) if identity["type"] == "integer" else identity["value"])
        if label in unique:
            _error("invalid_label", "Saved typed identities must be unique.")
        labels.append(label)
        unique.add(label)
        identities[row["id"]] = label
        attrs = dict(row.get("attrs", {}))
        if row["label"] != str(label):
            attrs["label"] = row["label"]
        for field in ("x", "y", "fx", "fy", "fixed", "pinned", "longitude", "latitude"):
            if field in row:
                attrs[field] = row[field]
        node_attrs[label] = attrs
    edge_attrs = {}
    def records():
        for row in data["edges"]:
            u, v = identities[row["source"]], identities[row["target"]]
            pair = (u, v) if data["directed"] or _key(u) <= _key(v) else (v, u)
            if "attrs" in row:
                if pair in edge_attrs and edge_attrs[pair] != row["attrs"]:
                    _error("file_attribute", "Duplicate saved dyads have conflicting attributes.")
                edge_attrs[pair] = row["attrs"]
            yield {"source": u, "target": v, "weight": _weight(row["weight"])}
    graph = network(records(), nodes=labels, directed=data["directed"], weight="weight", max_memory_mb=max_memory_mb)
    graph = graph.with_attributes(nodes=node_attrs, edges=edge_attrs)
    graph._metadata.update(derived_from="saved network display subset", source_node_count=data["node_count"],
                           source_edge_count=data["edge_count"], source_sampled=data["sampled"],
                           source_selection=data["selection"])
    return graph
