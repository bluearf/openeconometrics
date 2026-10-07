"""Single-aspect coupled networks with explicit typed (node, layer) identities."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.frame import as_frame
from openecon.networks import _Budget, _error, _integer, _label, _missing, _rows, _weight
from openecon._network_io import _attribute_bytes, _attributes
from openecon._network_multi import _identity_bytes, multigraph


def _pair(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        _error("invalid_label", "Node-layer identities are (node, layer) pairs of exact string/int64 IDs.")
    return (_label(value[0]), _label(value[1]))


def multilayer_network(data, edge_id="edge_id", source="source", source_layer="source_layer",
                       target="target", target_layer="target_layer", weight=None,
                       directed=False, nodes=None, layers=None, *, attributes=None,
                       node_attributes=None, graph_attributes=None, missing="raise",
                       batch_rows=128, max_memory_mb=256):
    """Keep separate edges between typed node-layer pairs, including zero coupling.

    ``nodes`` enumerates (node, layer) pairs including isolates. An explicit
    ``layers`` list is an ordered allowlist, including empty layers. Otherwise
    layers are inferred in first-seen order. There is no Cartesian padding.
    Attributes are bounded scalar dictionaries; node attributes use pair keys.
    Resident import and operations charge conservative owned-workspace plans.
    """
    budget = _Budget(max_memory_mb)
    if not isinstance(directed, bool) or not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("invalid_option", "directed must be boolean; missing must be 'raise' or 'drop'.")
    columns = [edge_id, source, source_layer, target, target_layer]
    columns += [weight] if weight is not None else []
    columns += [attributes] if attributes is not None else []
    if (any(not isinstance(c, str) or not c.strip() or len(c) > 200 for c in columns)
            or len(set(columns)) != len(columns)):
        _error("invalid_columns", "Multilayer identity, endpoint, weight and attribute columns must be distinct.")
    requested = _integer(batch_rows, "batch_rows", 1_000_000)
    # One row bounds parsed attribute dictionaries and iterator lookahead.
    rows = 1
    pairs, index, ordered_layers, layer_set, records = [], {}, [], set(), []
    mapping_bytes = record_bytes = dropped = inputs = 0
    fixed = 524288
    budget.check(fixed)

    def guard(extra=0):
        budget.check(mapping_bytes + record_bytes + fixed + extra)

    def intern_layer(raw):
        nonlocal mapping_bytes
        value = _label(raw)
        if value not in layer_set:
            if layers is not None and not reading_layers:
                _error("unknown_layer", "An endpoint names a layer outside the explicit layers allowlist.")
            addition = _identity_bytes(value)
            guard(addition)
            layer_set.add(value)
            ordered_layers.append(value)
            mapping_bytes += addition
        return value

    def intern(raw):
        nonlocal mapping_bytes
        node, layer = _pair(raw)
        intern_layer(layer)
        pair = (node, layer)
        if pair not in index:
            addition = 512 + _identity_bytes(node) + _identity_bytes(layer)
            guard(addition)
            index[pair] = len(pairs)
            pairs.append(pair)
            mapping_bytes += addition
        return index[pair]

    reading_layers = True
    for values, name in ((layers, "layers"), (nodes, "nodes")):
        if values is not None and (isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable)):
            _error("invalid_option", f"{name} must be an iterable of exact identities.")
    for layer in layers if layers is not None else ():
        value = _label(layer)
        if value in layer_set:
            _error("invalid_label", "Explicit layers must be unique.")
        intern_layer(value)
    # Detect duplicates separately: interning otherwise deliberately reuses IDs.
    if layers is not None:
        reading_layers = False
    for pair in nodes if nodes is not None else ():
        intern(pair)
    iterator = _rows(data, columns, rows)
    try:
        for block, _ in iterator:
            for row in block:
                row = tuple(row)
                inputs += 1
                core = row[:5 + (weight is not None)]
                if any(_missing(v) for v in core):
                    if missing == "drop":
                        dropped += 1
                        continue
                    _error("missing", "Missing multilayer identity, endpoint or weight; use missing='drop' explicitly.")
                identifier = _label(row[0])
                attrs = row[-1] if attributes is not None else {}
                addition = 1024 + _identity_bytes(identifier) + 2 * _attribute_bytes(attrs)
                guard(addition)
                copied = _attributes(attrs)
                i, j = intern((row[1], row[2])), intern((row[3], row[4]))
                guard(addition)
                records.append(dict(edge_id=identifier, source=i, target=j,
                                    weight=_weight(row[5]) if weight is not None else 1., attributes=copied))
                record_bytes += addition
    finally:
        iterator.close()
    converted = {}
    if node_attributes is not None:
        if not isinstance(node_attributes, dict):
            _error("file_attribute", "Node attributes must map exact (node, layer) pairs to scalar dictionaries.")
        for raw, attrs in node_attributes.items():
            pair = _pair(raw)
            if pair not in index:
                _error("unknown_node", "Node attributes refer to an absent node-layer pair.")
            addition = 512 + 2 * _attribute_bytes(attrs)
            guard(addition)
            converted[index[pair]] = _attributes(attrs)
            record_bytes += addition
    graph_attrs = {} if graph_attributes is None else graph_attributes
    guard(2 * _attribute_bytes(graph_attrs))
    guard(4096)
    available = (budget.limit - mapping_bytes - record_bytes - fixed) / 1024**2
    graph = multigraph(records, weight="weight", directed=directed, nodes=range(len(pairs)),
                       attributes="attributes", node_attributes=converted,
                       graph_attributes=graph_attrs, batch_rows=1, max_memory_mb=available)
    budget.check(mapping_bytes + record_bytes + fixed + graph._budget.peak)
    graph.weighted = weight is not None
    graph._base_bytes += mapping_bytes
    graph._budget = budget
    graph._metadata.update(input_rows=inputs, missing_rows_dropped=dropped,
                           requested_batch_rows=requested, effective_batch_rows=rows,
                           node_layer_mapping_bytes=mapping_bytes)
    graph._refresh()
    return MultilayerNetwork(graph, tuple(pairs), index, tuple(ordered_layers))


class MultilayerNetwork:
    """Immutable edge snapshot with explicit intra/inter-layer topology on CPU."""
    def __init__(self, graph, pairs, index, layers):
        self._graph, self._pairs, self._index, self._layers = graph, pairs, index, layers
        self.directed, self.weighted = graph.directed, graph.weighted

    @property
    def node_count(self):
        return len(self._pairs)

    @property
    def edge_count(self):
        return self._graph.edge_count

    @property
    def layer_count(self):
        return len(self._layers)

    @property
    def metadata(self):
        values = self._graph.metadata
        values.update(representation="coupled single-aspect multilayer", layer_count=self.layer_count,
                      node_identity="exact (node, layer) pair; no Cartesian padding",
                      adjacency="A[source state, target state]; parallel weights sum; undirected loops once",
                      coupling="explicit inter-layer edge weights; no inferred coupling", dense_adjacency=False)
        return values

    def _frame(self, values, **metadata):
        result = as_frame(pd.DataFrame(values))
        result.attrs.update(network=self.metadata, exact=True, sampled=False, device="cpu", **metadata)
        return result

    def nodes(self, *, attributes=True):
        """Return the ordered supra_index -> exact node, layer map, including isolates."""
        workspace = 1536 * self.node_count
        self._graph._guard(workspace + 4096)
        columns = self._attribute_columns(range(self.node_count), self._graph._node_attributes,
                                          self.node_count, workspace, attributes)
        return self._frame(dict(supra_index=range(self.node_count),
            node=pd.Series([p[0] for p in self._pairs], dtype=object),
            layer=pd.Series([p[1] for p in self._pairs], dtype=object), **columns), kind="multilayer_nodes")

    def _attribute_columns(self, identities, records, count, workspace, enabled):
        if not isinstance(enabled, bool):
            _error("invalid_option", "attributes must be boolean.")
        if not enabled:
            return {}
        fields = set()
        for attrs in records.values():
            for name in attrs:
                self._graph._guard(workspace + 512 * (len(fields) + 1) + 256 * (len(fields) + 1) * count)
                fields.add(name)
        self._graph._guard(workspace + 512 * len(fields) + 256 * len(fields) * count)
        return {"attr." + name: [records.get(identifier, {}).get(name) for identifier in identities]
                for name in sorted(fields)}

    def layers(self):
        """Return declared/inferred layers in order, including explicit empty layers."""
        self._graph._guard(512 * self.layer_count)
        counts = dict.fromkeys(self._layers, 0)
        for _, layer in self._pairs:
            counts[layer] += 1
        return self._frame(dict(layer=pd.Series(self._layers, dtype=object), node_count=list(counts.values())),
                           kind="multilayer_layers")

    def _records(self):
        for row in self._graph._records():
            u, v = self._pairs[row.pop("source")], self._pairs[row.pop("target")]
            yield dict(edge_id=row["edge_id"], source=u[0], source_layer=u[1], target=v[0],
                       target_layer=v[1], weight=row["weight"], attributes=row["attributes"])

    def edges(self, *, attributes=True):
        """Return every raw edge identity and its two node-layer endpoints."""
        workspace = 2048 * self.edge_count
        self._graph._guard(workspace)
        columns = self._attribute_columns(self._graph._edge_ids, self._graph._edge_attributes,
                                          self.edge_count, workspace, attributes)
        u, v = self._graph._endpoints.tolist()
        return self._frame(dict(edge_id=pd.Series(self._graph._edge_ids, dtype=object),
            source=pd.Series([self._pairs[i][0] for i in u], dtype=object),
            source_layer=pd.Series([self._pairs[i][1] for i in u], dtype=object),
            target=pd.Series([self._pairs[i][0] for i in v], dtype=object),
            target_layer=pd.Series([self._pairs[i][1] for i in v], dtype=object),
            weight=self._graph._weights.numpy(),
            inter_layer=[self._pairs[i][1] != self._pairs[j][1] for i, j in zip(u, v)], **columns),
            kind="multilayer_edges")

    def _work(self, amount, maximum):
        maximum = _integer(maximum, "max_work", 2**63 - 1)
        if amount > maximum:
            _error("work_budget", "Multilayer operation exceeds max_work before numerical setup.")
        return maximum

    def matvec(self, vector, *, transpose=False, max_work=50_000_000):
        """Compute A @ x (or A.T @ x) using sparse edge reductions; no N² buffer.

        Input must map every exact (node, layer) pair to a finite real value.
        Output retains the complete node-layer map. Undirected adjacency loops
        contribute once, unlike the twice-counted graph-theoretic loop degree.
        """
        from openecon.networks import _real
        if not isinstance(transpose, bool) or not isinstance(vector, Mapping):
            _error("invalid_option", "vector must map all node-layer pairs; transpose must be boolean.")
        maximum = self._work(16 * self.node_count + 8 * self.edge_count, max_work)
        self._graph._guard(1024 * self.node_count + 128 * self.edge_count)
        with torch.device("cpu"), torch.no_grad():
            x = torch.zeros(self.node_count, dtype=torch.float64, device="cpu")
            seen = set()
            for raw, value in vector.items():
                pair = _pair(raw)
                if pair not in self._index:
                    _error("unknown_node", "Vector names an absent node-layer pair.")
                x[self._index[pair]] = _real(value, "invalid_option", "Vector values must be finite real numbers.")
                seen.add(pair)
            if len(seen) != self.node_count:
                _error("invalid_option", "Vector must include every node-layer pair, including isolates.")
            i, j = self._graph._endpoints
            if transpose:
                i, j = j, i
            y = torch.zeros_like(x)
            y.index_add_(0, i, self._graph._weights * x[j])
            if not self.directed:
                off = i != j
                y.index_add_(0, j[off], self._graph._weights[off] * x[i[off]])
            if not bool(torch.isfinite(y).all()):
                _error("precision", "Supra-matvec exceeds float64 range.")
            values = self.nodes(attributes=False)
            values["value"] = y.numpy()
            return self._frame(values, kind="supra_matvec", transpose=transpose, max_work=maximum)

    def supra_network(self, *, max_work=50_000_000):
        """Explicit sum-weight simple supra-graph; integer IDs map through nodes().

        Zero-weight pairs and edge attributes are explicitly dropped in this
        analytical projection. The original edge records remain unchanged.
        """
        result = self._graph.to_network(reducer="sum", zero="drop", attributes="drop", max_work=max_work)
        result._metadata["multilayer_supra"] = dict(node_mapping="nodes().supra_index -> (node, layer)",
            coupling="all explicit inter-layer edges included", node_layer_count=self.node_count,
            layer_count=self.layer_count, source_edges_unchanged=True)
        return result

    def pagerank(self, damping=.85, tol=1e-10, max_iter=200, personalization=None, *, max_work=1_000_000_000):
        """Weighted PageRank on the supra-graph; teleportation over node-layer pairs.

        Personalization keys are exact (node, layer) pairs. This is the ordinary
        random walk on the explicitly weighted supra-graph, not a layer-balanced
        or biased multiplex alternative. Dangling mass follows personalization.
        """
        _integer(max_iter, "max_iter", 1_000_000)
        self._work((32 * self.node_count + 16 * self.edge_count) * max_iter
                   + 32 * self.edge_count * max(1, self.edge_count.bit_length()), max_work)
        selected = None
        if personalization is not None:
            if not isinstance(personalization, Mapping):
                _error("invalid_option", "personalization must map node-layer pairs to nonnegative weights.")
            self._graph._guard(512 * self.node_count)
            selected = {}
            for raw, value in personalization.items():
                pair = _pair(raw)
                if pair not in self._index:
                    _error("unknown_node", "Personalization names an absent node-layer pair.")
                selected[self._index[pair]] = value
        graph = self.supra_network(max_work=max_work)
        # Charge the still-resident source while the derived graph is evaluated.
        graph._base_bytes += self._graph._base_bytes
        try:
            values = graph.pagerank(damping=damping, tol=tol, max_iter=max_iter, personalization=selected)
            self._graph._budget.check(graph._budget.peak)
            self._graph._guard(graph._base_bytes - self._graph._base_bytes + 1024 * self.node_count)
            result = self.nodes(attributes=False)
            result["pagerank"] = values["pagerank"].to_numpy()
            metadata = {k: deepcopy(v) for k, v in values.attrs.items() if k != "network"}
            metadata.update(kind="supra_pagerank", state_space="node-layer pairs", max_work=max_work)
            metadata.pop("device", None)
            metadata.pop("exact", None)
            metadata.pop("sampled", None)
            return self._frame(result, **metadata)
        finally:
            graph._base_bytes -= self._graph._base_bytes

    def layer(self, layer):
        """Extract an intra-layer MultiNetwork, preserving raw IDs, attrs and isolates."""
        layer = _label(layer)
        if layer not in self._layers:
            _error("unknown_layer", "Layer selection names an absent layer.")
        workspace = 1024 * (self.node_count + self.edge_count) + 65536
        self._graph._guard(workspace)
        nodes = [node for node, item in self._pairs if item == layer]
        records = (dict(edge_id=r["edge_id"], source=r["source"], target=r["target"],
                        weight=r["weight"], attributes=r["attributes"]) for r in self._records()
                   if r["source_layer"] == layer == r["target_layer"])
        attrs = {self._pairs[i][0]: a for i, a in self._graph._node_attributes.items() if self._pairs[i][1] == layer}
        available = (self._graph._budget.limit - self._graph._base_bytes - workspace - 4096) / 1024**2
        result = multigraph(records, weight="weight", attributes="attributes", nodes=nodes, node_attributes=attrs,
                           directed=self.directed, graph_attributes=self._graph._graph_attributes, max_memory_mb=available)
        result._metadata["multilayer_layer"] = dict(layer=layer, inter_layer="excluded", edge_ids_preserved=True)
        return result

    def project(self, *, reducer, inter_layer, zero="raise", attributes="raise", max_work=50_000_000):
        """Collapse exact physical node IDs with explicit reducer and inter-layer policy.

        ``inter_layer='drop'`` excludes coupling; ``'include'`` retains it,
        turning diagonal coupling into loops. Attributes are preserved only
        when all node replicas/parallel edges agree; choose 'drop' explicitly
        to discard them. This lossy operation never changes the source.
        """
        if (not isinstance(inter_layer, str) or inter_layer not in {"include", "drop"}
                or not isinstance(attributes, str) or attributes not in {"raise", "drop"}):
            _error("invalid_option", "inter_layer must be 'include' or 'drop'; attributes must be 'raise' or 'drop'.")
        self._work(32 * self.node_count + 32 * self.edge_count * max(1, self.edge_count.bit_length()), max_work)
        workspace = 1024 * (self.node_count + self.edge_count) + 65536
        self._graph._guard(workspace)
        nodes = list(dict.fromkeys(node for node, _ in self._pairs))
        node_attrs = {}
        if attributes == "raise":
            for i, (node, _) in enumerate(self._pairs):
                attrs = self._graph._node_attributes.get(i, {})
                if node in node_attrs and not _equal_attributes(node_attrs[node], attrs):
                    _error("projection_loss", "Node replicas have different attributes; choose attributes='drop' explicitly.")
                node_attrs[node] = attrs
        records = (dict(edge_id=r["edge_id"], source=r["source"], target=r["target"], weight=r["weight"],
                        attributes=r["attributes"] if attributes == "raise" else {}) for r in self._records()
                   if inter_layer == "include" or r["source_layer"] == r["target_layer"])
        available = (self._graph._budget.limit - self._graph._base_bytes - workspace - 4096) / 1024**2
        intermediate = multigraph(records, weight="weight", attributes="attributes", nodes=nodes,
                                  node_attributes=node_attrs, directed=self.directed,
                                  graph_attributes=self._graph._graph_attributes, max_memory_mb=available)
        result = intermediate.to_network(reducer=reducer, zero=zero, attributes=attributes, max_work=max_work)
        result._metadata["multilayer_projection"] = dict(reducer=reducer, inter_layer=inter_layer,
            node_mapping="exact typed node ID shared across layers; nodes() maps every replica",
            node_layer_count=self.node_count, projected_node_count=len(nodes),
            layers_preserved=False, edge_ids_preserved=False, attributes_policy=attributes)
        return result

    def edit_edges(self, *, add=(), remove=(), weights=None, attributes=None):
        """Immutably add/remove edges or update exact edge-ID weights/attributes.

        Added records require both source_layer and target_layer. Existing
        node-layer pairs/isolates remain; new pairs may use existing layers.
        """
        if isinstance(add, (str, bytes, Mapping)) or not isinstance(add, Iterable):
            _error("invalid_option", "add must be an iterable of complete multilayer edge records.")
        workspace = 2048 * (self.node_count + self.edge_count) + 65536
        self._graph._guard(workspace)
        removed = self._graph._select(remove, self._graph._edge_index, "unknown_edge")
        for option in (weights, attributes):
            if option is not None and not isinstance(option, dict):
                _error("invalid_option", "Updates must use exact edge-ID-keyed dictionaries.")
            for raw in option or {}:
                identifier = _label(raw)
                if identifier not in self._graph._edge_index or identifier in removed:
                    _error("unknown_edge", "Updates require a retained existing edge ID.")
        added_weighted = False
        def records():
            nonlocal added_weighted
            for row in self._records():
                identifier = row["edge_id"]
                if identifier not in removed:
                    row["weight"] = (weights or {}).get(identifier, row["weight"])
                    row["attributes"] = (attributes or {}).get(identifier, row["attributes"])
                    yield row
            iterator = iter(add)
            try:
                for row in iterator:
                    if not isinstance(row, Mapping):
                        _error("invalid_data", "Added multilayer edges must be records.")
                    added_weighted |= "weight" in row
                    yield {**row, "weight": row.get("weight", 1.), "attributes": row.get("attributes", {})}
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        available = (self._graph._budget.limit - self._graph._base_bytes - workspace - 4096) / 1024**2
        attrs = {self._pairs[i]: a for i, a in self._graph._node_attributes.items()}
        result = multilayer_network(records(), weight="weight", attributes="attributes", nodes=self._pairs,
            layers=self._layers, node_attributes=attrs, graph_attributes=self._graph._graph_attributes,
            directed=self.directed, max_memory_mb=available)
        result.weighted = result._graph.weighted = self.weighted or bool(weights) or added_weighted
        result._graph._refresh()
        result._graph._metadata["operation_semantics"] = "new multilayer snapshot; source unchanged"
        return result

    def write(self, path, *, overwrite=False):
        """Atomically save lossless versioned NDJSON; no implicit GraphML/GEXF flattening."""
        return _write(self, path, overwrite=overwrite)


def _equal_attributes(a, b):
    return a.keys() == b.keys() and all(type(v) is type(b[k]) and v == b[k] for k, v in a.items())


_SCHEMA = "openecon.coupled-multilayer/1"


def _write(graph, path, *, overwrite):
    if not isinstance(overwrite, bool):
        _error("invalid_option", "overwrite must be boolean.")
    path = Path(path)
    if path.exists() and not overwrite:
        _error("file_exists", "Destination exists; choose overwrite=True explicitly.")
    graph._graph._guard(524288)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".openecon-multilayer-", delete=False) as output:
            temporary = Path(output.name)
            def emit(row):
                planned = 1024 + _attribute_bytes(row.get("attributes", {}))
                planned += 16 * sum(len(v) for v in row.values() if isinstance(v, str))
                graph._graph._guard(2 * planned + 65536)
                text = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                graph._graph._guard(16 * len(text) + 65536)
                output.write(text + "\n")
            emit(dict(kind="header", schema=_SCHEMA, directed=graph.directed, weighted=graph.weighted,
                      attributes=graph._graph._graph_attributes))
            for layer in graph._layers:
                emit(dict(kind="layer", layer=layer))
            for i, (node, layer) in enumerate(graph._pairs):
                emit(dict(kind="node", node=node, layer=layer, attributes=graph._graph._node_attributes.get(i, {})))
            for row in graph._records():
                emit(dict(kind="edge", **row))
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        return path
    except OSError as exc:
        raise AnalysisError("network_file_io", "Multilayer export could not write the destination.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_multilayer_network(path, *, max_memory_mb=256, max_file_mb=512):
    """Read strict versioned NDJSON without losing typed pairs, edges or attributes."""
    budget, file_budget = _Budget(max_memory_mb), _Budget(max_file_mb)
    path = Path(path)
    nodes, attrs, layers, edges, seen_nodes, seen_layers = [], {}, [], [], set(), set()
    owned = consumed = phase = 0
    header = None
    budget.check(524288)
    allowed = {"header": {"kind", "schema", "directed", "weighted", "attributes"},
               "layer": {"kind", "layer"}, "node": {"kind", "node", "layer", "attributes"},
               "edge": {"kind", "edge_id", "source", "source_layer", "target", "target_layer", "weight", "attributes"}}
    def object_pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                _error("file_format", "Duplicate JSON fields are not accepted.")
            result[key] = value
        return result
    try:
        before = path.stat()
        if before.st_size > file_budget.limit:
            _error("file_budget", "Multilayer source exceeds max_file_mb.")
        with path.open("rb") as source:
            while True:
                maximum = min(16 * 1024 * 1024 + 1, (budget.limit - owned - 524288) // 16)
                budget.check(owned + 524288 + 16 * max(1, maximum))
                raw = source.readline(max(1, maximum))
                if not raw:
                    break
                consumed += len(raw)
                if consumed > file_budget.limit:
                    _error("file_budget", "Multilayer source exceeds max_file_mb.")
                budget.check(owned + 16 * len(raw) + 524288)
                if maximum < 16 * 1024 * 1024 + 1 and len(raw) == maximum and not raw.endswith(b"\n"):
                    _error("memory_budget", "A multilayer record exceeds the remaining owned parse workspace.")
                if len(raw) > 16 * 1024 * 1024 or not raw.endswith(b"\n"):
                    _error("file_format", "Each interchange record must be a newline-terminated bounded JSON object.")
                try:
                    row = json.loads(raw, object_pairs_hook=object_pairs,
                                     parse_constant=lambda value: _error("file_format", "Nonfinite JSON scalars are invalid."))
                except (ValueError, UnicodeError, RecursionError):
                    _error("file_format", "Invalid multilayer JSON record.")
                if (not isinstance(row, dict) or not isinstance(row.get("kind"), str)
                        or row["kind"] not in allowed or set(row) != allowed[row["kind"]]):
                    _error("file_feature", "Unknown/missing interchange fields are not silently discarded.")
                kind = row.pop("kind")
                order = {"header": 0, "layer": 1, "node": 2, "edge": 3}[kind]
                if order < phase or kind == "header" and header is not None or kind != "header" and header is None:
                    _error("file_format", "Interchange requires one header then layers, nodes and edges in order.")
                phase = order
                addition = 1024 + 16 * len(raw)
                budget.check(owned + 2 * addition + 524288)
                owned += addition
                if kind == "header":
                    if row["schema"] != _SCHEMA or not isinstance(row["directed"], bool) or not isinstance(row["weighted"], bool):
                        _error("file_format", "Unsupported coupled-multilayer schema or orientation.")
                    _attributes(row["attributes"])
                    header = row
                elif kind == "layer":
                    value = _label(row["layer"])
                    if value in seen_layers:
                        _error("file_format", "Duplicate layer declaration.")
                    layers.append(value)
                    seen_layers.add(value)
                elif kind == "node":
                    pair = _pair((row["node"], row["layer"]))
                    if pair in seen_nodes or pair[1] not in seen_layers:
                        _error("file_format", "Duplicate node-layer pair or undeclared layer.")
                    nodes.append(pair)
                    attrs[pair] = _attributes(row["attributes"])
                    seen_nodes.add(pair)
                else:
                    if (_pair((row["source"], row["source_layer"])) not in seen_nodes
                            or _pair((row["target"], row["target_layer"])) not in seen_nodes):
                        _error("file_format", "Edges must use declared node-layer pairs.")
                    if not header["weighted"] and _weight(row["weight"]) != 1.:
                        _error("file_format", "Unweighted interchange cannot carry non-unit weights.")
                    edges.append(row)
        after = path.stat()
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
            _error("source_changed", "The multilayer source changed during import.")
    except OSError as exc:
        raise AnalysisError("network_file_io", "Multilayer import could not read the source.") from exc
    if header is None:
        _error("file_format", "Missing multilayer header.")
    budget.check(owned + 524288 + 4096)
    available = (budget.limit - owned - 524288) / 1024**2
    result = multilayer_network(edges, weight="weight", attributes="attributes", nodes=nodes, layers=layers,
        node_attributes=attrs, directed=header["directed"], graph_attributes=header["attributes"], max_memory_mb=available)
    result.weighted = result._graph.weighted = header["weighted"]
    result._graph._metadata.update(interchange_schema=_SCHEMA, source_file_bytes=consumed)
    return result
