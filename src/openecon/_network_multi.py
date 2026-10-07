"""Resident typed-edge multigraphs; explicit projection, never implicit coalescing."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.analysis_contracts import AnalysisError
from openecon.networks import Network, _Budget, _error, _integer, _key, _label, _missing, _rows, _weight, network
from openecon._network_io import _attribute_bytes, _attributes


def _identity_bytes(value):
    return 512 + 2 * sys.getsizeof(value) + (len(value.encode("utf-8")) if isinstance(value, str) else 32)


def multigraph(data, edge_id="edge_id", source="source", target="target", weight=None,
               directed=False, nodes=None, *, attributes=None, node_attributes=None,
               edge_attributes=None, graph_attributes=None, missing="raise",
               batch_rows=65536, max_memory_mb=256):
    """Retain unique typed edge IDs, parallel edges and zero weights on CPU.

    ``attributes`` names an optional column of bounded scalar dictionaries.
    Endpoint orientation and input order are retained; projection is explicit.
    Owned graph/workspace is budgeted; caller input and total RSS are excluded.
    """
    return _build_multigraph(data, edge_id, source, target, weight, directed, nodes,
        attributes=attributes, node_attributes=node_attributes, edge_attributes=edge_attributes,
        graph_attributes=graph_attributes, missing=missing, batch_rows=batch_rows,
        max_memory_mb=max_memory_mb)


def _build_multigraph(data, edge_id="edge_id", source="source", target="target", weight=None,
               directed=False, nodes=None, *, attributes=None, node_attributes=None,
               edge_attributes=None, graph_attributes=None, missing="raise",
               batch_rows=65536, max_memory_mb=256, _live_bytes=None):
    """Keep every unique typed edge ID, including zero weights and parallel edges.

    ``attributes`` names an optional column of scalar dictionaries. Additional
    edge attributes use edge-ID keys. Input order and endpoint orientation stay
    unchanged even for undirected edges. Storage is resident, under an explicit
    owned-workspace estimate; whole-process RSS and caller input are excluded.
    """
    if not isinstance(directed, bool) or not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("invalid_option", "directed must be boolean; missing must be 'raise' or 'drop'.")
    columns = [edge_id, source, target] + ([weight] if weight is not None else []) + ([attributes] if attributes is not None else [])
    if (any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in columns)
            or len(set(columns)) != len(columns)):
        _error("invalid_columns", "Edge ID, endpoints, weight and attribute columns must have distinct valid names.")
    if nodes is not None and (not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes, Mapping))):
        _error("invalid_label", "nodes must be an iterable of exact typed node IDs.")
    requested = _integer(batch_rows, "batch_rows", 1_000_000)
    budget = _Budget(max_memory_mb)
    fixed = min(1024**2, budget.limit // 4) + 262144
    budget.check(fixed + 65536)
    rows = min(requested, 128, max(1, (budget.limit - fixed) // (8 * 65536)))
    if attributes is not None:
        rows = 1  # Parsed dictionaries can be much larger than numeric edge rows.
    labels, index, identifiers, edge_index, left, right, values, row_attrs = [], {}, [], {}, [], [], [], {}
    owned = inputs = dropped = peak = 0
    digest = hashlib.sha256()

    def publish_live():
        if _live_bytes is not None:
            _live_bytes[0] = owned + fixed + 65536 * rows

    def intern(raw):
        nonlocal owned
        value = _label(raw)
        if value not in index:
            addition = _identity_bytes(value)
            budget.check(owned + addition + fixed + 65536 * rows)
            index[value] = len(labels)
            labels.append(value)
            owned += addition
            publish_live()
        return index[value]

    for label in nodes if nodes is not None else ():
        intern(label)
    publish_live()
    iterator = _rows(data, columns, rows)
    try:
        for block, size in iterator:
            peak = max(peak, size)
            for row in block:
                row = tuple(row)
                inputs += 1
                raw = row[:3 + (weight is not None)]
                if any(_missing(value) for value in raw):
                    if missing == "drop":
                        dropped += 1
                        continue
                    _error("missing", "Missing multigraph identity, endpoint or weight; use missing='drop' explicitly.")
                identifier = _label(row[0])
                if identifier in edge_index:
                    _error("duplicate_edge_id", "Every multigraph edge needs a unique typed edge ID.")
                w = _weight(row[3]) if weight is not None else 1.
                attrs = row[-1] if attributes is not None else {}
                addition = _identity_bytes(identifier) + 512 + _attribute_bytes(attrs)
                budget.check(owned + addition + fixed + 65536 * rows)
                attrs = _attributes(attrs)
                if "weight" in attrs:
                    _error("file_attribute", "Edge attribute 'weight' is reserved for the actual edge weight.")
                i, j = intern(row[1]), intern(row[2])
                # Interning endpoints can consume the remaining row allowance.
                budget.check(owned + addition + fixed + 65536 * rows)
                edge_index[identifier] = len(identifiers)
                identifiers.append(identifier)
                left.append(i)
                right.append(j)
                values.append(w)
                if attrs:
                    row_attrs[identifier] = attrs
                owned += addition
                publish_live()
                digest.update(json.dumps([identifier, labels[i], labels[j], w, attrs],
                    ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    finally:
        iterator.close()
    budget.check(owned + 64 * len(values) + fixed)
    with torch.device("cpu"), torch.no_grad():
        endpoints = torch.tensor([left, right], dtype=torch.int64, device="cpu")
        weights = torch.tensor(values, dtype=torch.float64, device="cpu")
    result = MultiNetwork(tuple(labels), index, tuple(identifiers), edge_index, endpoints,
        weights, directed, weight is not None, budget, owned, dict(input_rows=inputs,
            missing_rows_dropped=dropped, zero_weight_rows_dropped=0,
            duplicate_edge_rows_aggregated=0, input_sha256=digest.hexdigest(),
            requested_batch_rows=requested, effective_batch_rows=rows, actual_peak_batch_rows=peak))
    if edge_attributes is not None:
        if not isinstance(edge_attributes, dict):
            _error("file_attribute", "Edge attributes must use an edge-ID-keyed dictionary.")
        result._guard(512 * len(row_attrs) + 2 * sum(_attribute_bytes(v) for v in row_attrs.values()))
        for identifier, attrs in edge_attributes.items():
            identifier = _label(identifier)
            if identifier not in edge_index:
                _error("unknown_edge", "Attributes name an edge outside the multigraph.")
            result._guard(512 * len(row_attrs) + _attribute_bytes(attrs))
            row_attrs[identifier] = _attributes(attrs)
    return result._attach({} if node_attributes is None else node_attributes, row_attrs,
                          {} if graph_attributes is None else graph_attributes)


class MultiNetwork:
    """Immutable-by-operation CPU edge snapshot with separate parallel IDs."""
    _is_multigraph = True

    def weighted_assignment(self, partition=None, *, objective="weight", max_matrix_entries=1_000_000,
                            max_work=50_000_000):
        """Choose actual parallel edge IDs in exact weighted assignment."""
        from openecon._network_matching import weighted_assignment
        return weighted_assignment(self, partition, objective=objective,
            max_matrix_entries=max_matrix_entries, max_work=max_work)

    def general_matching(self, *, objective="weight", max_component_nodes=24,
                         max_states=1_000_000, max_work=50_000_000):
        """Choose actual parallel edge IDs in bounded exact general matching."""
        from openecon._network_matching import general_matching
        return general_matching(self, objective=objective, max_component_nodes=max_component_nodes,
            max_states=max_states, max_work=max_work)

    def __init__(self, labels, index, identifiers, edge_index, endpoints, weights,
                 directed, weighted, budget, owned, metadata):
        self._labels, self._index = labels, index
        self._edge_ids, self._edge_index = identifiers, edge_index
        self._endpoints, self._weights = endpoints, weights
        self.directed, self.weighted = directed, weighted
        self._budget, self._base_bytes, self._metadata = budget, owned, metadata
        self._node_attributes, self._edge_attributes, self._graph_attributes = {}, {}, {}
        self._metadata.update(representation="typed-edge multigraph", storage="resident", device="cpu",
            dtype="float64/int64", exact=True, sampled=False,
            edge_identity_semantics="unique typed IDs; every parallel and zero-weight edge retained",
            memory_scope="planned owned graph and buffers; excludes caller input and total process RSS")
        self._refresh()

    @property
    def node_count(self):
        return len(self._labels)

    @property
    def edge_count(self):
        return len(self._edge_ids)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def node_attributes(self):
        self._guard(self._metadata.get("attribute_storage_bytes", 0))
        return deepcopy(self._node_attributes)

    @property
    def edge_attributes(self):
        self._guard(self._metadata.get("attribute_storage_bytes", 0))
        return deepcopy(self._edge_attributes)

    @property
    def graph_attributes(self):
        self._guard(_attribute_bytes(self._graph_attributes))
        return deepcopy(self._graph_attributes)

    def __repr__(self):
        return f"MultiNetwork(nodes={self.node_count}, edges={self.edge_count}, directed={self.directed})"

    def _guard(self, workspace):
        self._budget.check(self._base_bytes + int(workspace) + 4096)

    def _refresh(self):
        self._metadata.update(node_count=self.node_count, edge_count=self.edge_count,
            directed=self.directed, weighted=self.weighted, max_memory_bytes=self._budget.limit,
            estimated_owned_graph_bytes=self._base_bytes, estimated_import_peak_bytes=self._budget.peak)

    def _attach(self, nodes, edges, graph_attributes):
        total = 0
        owned = []
        for attrs, valid, code in ((nodes, self._index, "unknown_node"), (edges, self._edge_index, "unknown_edge")):
            if not isinstance(attrs, dict):
                _error("file_attribute", "Node/edge attributes must use exact-ID-keyed dictionaries.")
            target = {}
            for identifier, values in attrs.items():
                identifier = _label(identifier)
                if identifier not in valid:
                    _error(code, "Attributes name an identity outside the multigraph.")
                addition = 256 + _attribute_bytes(values)
                self._guard(total + 2 * addition)
                item = _attributes(values)
                if code == "unknown_edge" and "weight" in item:
                    _error("file_attribute", "Edge attribute 'weight' is reserved for the actual edge weight.")
                if not item:
                    continue
                total += addition
                target[identifier] = item
            owned.append(target)
        addition = _attribute_bytes(graph_attributes)
        self._guard(total + 2 * addition)
        self._node_attributes, self._edge_attributes = owned
        self._graph_attributes = _attributes(graph_attributes)
        self._base_bytes += total + addition
        self._metadata.update(attribute_storage_bytes=total + addition)
        self._refresh()
        return self

    def with_attributes(self, *, nodes=None, edges=None, graph_attributes=None):
        """Replace selected scalar-attribute dictionaries in a new edge-ID snapshot."""
        result = object.__new__(MultiNetwork)
        result.__dict__ = self.__dict__.copy()
        result._metadata = self.metadata
        reserve = self._metadata["attribute_storage_bytes"]
        result._budget = _Budget((self._budget.limit - reserve) / 1024**2)
        result._base_bytes -= reserve
        result._attach(self._node_attributes if nodes is None else nodes,
            self._edge_attributes if edges is None else edges,
            self._graph_attributes if graph_attributes is None else graph_attributes)
        result._budget.peak = max(self._budget.peak, result._budget.peak + reserve)
        result._budget.limit = self._budget.limit
        result._refresh()
        return result

    def _records(self):
        for start in range(0, self.edge_count, 128):
            self._guard(32768)
            u, v = self._endpoints[:, start:start + 128].tolist()
            weights = self._weights[start:start + 128].tolist()
            for offset, (i, j, w) in enumerate(zip(u, v, weights), start):
                identifier = self._edge_ids[offset]
                yield dict(edge_id=identifier, source=self._labels[i], target=self._labels[j],
                    weight=w, attributes=self._edge_attributes.get(identifier, {}))

    def _frame(self, values, **metadata):
        result = as_frame(pd.DataFrame(values))
        result.attrs.update(network=self.metadata, device="cpu", exact=True, sampled=False, **metadata)
        return result

    def _attribute_columns(self, identities, records, count):
        fields = set()
        for attrs in records.values():
            for name in attrs:
                self._guard(512 * len(fields) + (512 + 128 * (len(fields) + 1)) * count)
                fields.add(name)
        self._guard((512 + 128 * len(fields)) * count)
        return {"attr." + name: [records.get(identifier, {}).get(name) for identifier in identities]
                for name in sorted(fields)}

    def nodes(self, *, attributes=True):
        """Materialize ordered typed node IDs and optional scalar attribute columns."""
        if not isinstance(attributes, bool):
            _error("invalid_option", "attributes must be boolean.")
        self._guard(512 * self.node_count)
        columns = self._attribute_columns(self._labels, self._node_attributes, self.node_count) if attributes else {}
        return self._frame({"node": pd.Series(self._labels, dtype=object), **columns}, kind="multigraph_nodes")

    def edges(self, *, attributes=True):
        """Materialize every separate edge ID, raw endpoints, weight and optional attributes."""
        if not isinstance(attributes, bool):
            _error("invalid_option", "attributes must be boolean.")
        self._guard(512 * self.edge_count)
        columns = self._attribute_columns(self._edge_ids, self._edge_attributes, self.edge_count) if attributes else {}
        u, v = self._endpoints.tolist()
        return self._frame(dict(edge_id=pd.Series(self._edge_ids, dtype=object),
            source=pd.Series([self._labels[i] for i in u], dtype=object),
            target=pd.Series([self._labels[j] for j in v], dtype=object), weight=self._weights.numpy(), **columns),
            kind="multigraph_edges")

    def summary(self):
        """Tabulate separate edge counts, retained zero weights and self-loops."""
        self._guard(128 * self.edge_count + 32768)
        records = [("Nodes", self.node_count), ("Edges with separate IDs", self.edge_count),
            ("Directed", self.directed), ("Weighted input", self.weighted),
            ("Self-loops", int((self._endpoints[0] == self._endpoints[1]).sum())),
            ("Zero-weight edges retained", int((self._weights == 0).sum())),
            ("Input rows", self._metadata["input_rows"]),
            ("Missing rows dropped", self._metadata["missing_rows_dropped"])]
        return self._frame(pd.DataFrame(records, columns=["Metric", "Value"]), kind="multigraph_summary")

    def degree(self, *, max_work=50_000_000):
        """Count each edge including zeros; sum strengths with undirected loops counted twice."""
        limit = _integer(max_work, "max_work", 2**63 - 1)
        work = 16 * self.node_count + 4 * self.edge_count
        if work > limit:
            _error("work_budget", "Multigraph degree exceeds max_work before numerical setup.")
        self._guard(512 * self.node_count + 64 * self.edge_count)
        with torch.device("cpu"), torch.no_grad():
            i, j = self._endpoints
            outgoing, incoming = torch.bincount(i, minlength=self.node_count), torch.bincount(j, minlength=self.node_count)
            out_w, in_w = torch.zeros(self.node_count, dtype=torch.float64), torch.zeros(self.node_count, dtype=torch.float64)
            out_w.index_add_(0, i, self._weights)
            in_w.index_add_(0, j, self._weights)
            if not bool(torch.isfinite(out_w).all() & torch.isfinite(in_w).all()):
                _error("precision", "A multigraph strength exceeds float64 range.")
            if self.directed:
                values = dict(in_degree=incoming.numpy(), out_degree=outgoing.numpy(),
                    in_strength=in_w.numpy(), out_strength=out_w.numpy())
            else:
                strength = in_w + out_w
                if not bool(torch.isfinite(strength).all()):
                    _error("precision", "A multigraph strength exceeds float64 range.")
                values = dict(degree=(incoming + outgoing).numpy(), strength=strength.numpy())
            return self._frame({"node": pd.Series(self._labels, dtype=object), **values},
                kind="multigraph_degree", work_used=work, max_work=limit,
                edge_semantics="each separate edge counts, including zero; undirected loops count twice")

    def _select(self, values, valid, code):
        if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
            _error("invalid_option", "Selections must be iterables of exact typed IDs.")
        selected = set()
        for value in values:
            value = _label(value)
            if value not in valid:
                _error(code, "Selection names an identity outside the multigraph.")
            self._guard(512 * (len(selected) + 1))
            selected.add(value)
        return selected

    def _rebuild(self, records, nodes, node_attributes, *, weighted=None):
        workspace = 1024 * (self.node_count + self.edge_count) + 65536
        self._guard(workspace)
        available = (self._budget.limit - self._base_bytes - workspace - 4096) / 1024**2
        result = multigraph(records, weight="weight", attributes="attributes", directed=self.directed,
            nodes=nodes, node_attributes=node_attributes, graph_attributes=self._graph_attributes,
            batch_rows=32, max_memory_mb=available)
        result._budget.peak += self._base_bytes + workspace + 4096
        result._budget.limit = self._budget.limit
        result.weighted = self.weighted if weighted is None else weighted
        result._metadata["operation_semantics"] = "new snapshot; source identities and data unchanged"
        result._refresh()
        return result

    def filter(self, *, nodes=None, edge_ids=None):
        """Select exact node/edge IDs with AND, preserving retained order and isolates."""
        self._guard(512 * self.node_count)
        chosen_nodes = set(self._labels) if nodes is None else self._select(nodes, self._index, "unknown_node")
        chosen_edges = None if edge_ids is None else self._select(edge_ids, self._edge_index, "unknown_edge")
        self._guard(512 * (len(chosen_nodes) + (len(chosen_edges) if chosen_edges is not None else 0)))
        records = (record for record in self._records() if record["source"] in chosen_nodes
            and record["target"] in chosen_nodes and (chosen_edges is None or record["edge_id"] in chosen_edges))
        return self._rebuild(records, [node for node in self._labels if node in chosen_nodes],
            {node: attrs for node, attrs in self._node_attributes.items() if node in chosen_nodes})

    def edit_edges(self, *, add=(), remove=(), weights=None, attributes=None):
        """Add independent-ID records or remove/replace weights and attributes by exact edge ID."""
        if isinstance(add, (str, bytes, Mapping)) or not isinstance(add, Iterable):
            _error("invalid_option", "add must be an iterable of edge records.")
        removed = self._select(remove, self._edge_index, "unknown_edge")
        for option in (weights, attributes):
            if option is not None and not isinstance(option, dict):
                _error("invalid_option", "Weight and attribute updates use exact edge-ID-keyed dictionaries.")
        changed = {}
        for identifier, value in (weights or {}).items():
            identifier = _label(identifier)
            if identifier not in self._edge_index or identifier in removed:
                _error("unknown_edge", "Weight updates require an existing retained edge ID.")
            self._guard(512 * (len(changed) + len(removed) + 1))
            changed[identifier] = _weight(value)

        updates, seen_updates = {}, set()
        for identifier, values in (attributes or {}).items():
            identifier = _label(identifier)
            self._guard(1024 * (len(updates) + 1) + _attribute_bytes(values))
            updates[identifier] = values

        added_weighted = False
        def records():
            nonlocal added_weighted
            for record in self._records():
                if record["edge_id"] in removed:
                    continue
                if record["edge_id"] in changed:
                    record["weight"] = changed[record["edge_id"]]
                if record["edge_id"] in updates:
                    record["attributes"] = updates[record["edge_id"]]
                    seen_updates.add(record["edge_id"])
                yield record
            for record in add:
                if not isinstance(record, Mapping) or not {"edge_id", "source", "target"} <= set(record):
                    _error("invalid_data", "Added edges require edge_id, source and target.")
                identifier = _label(record["edge_id"])
                added_weighted |= "weight" in record
                attrs = record.get("attributes", {})
                if identifier in updates:
                    attrs = updates[identifier]
                    seen_updates.add(identifier)
                yield dict(edge_id=identifier, source=record["source"], target=record["target"],
                    weight=record.get("weight", 1.), attributes=attrs)
            if len(seen_updates) != len(updates):
                _error("unknown_edge", "Attribute updates require a retained or explicitly added edge ID.")

        result = self._rebuild(records(), self._labels, self._node_attributes,
            weighted=self.weighted or bool(changed))
        result.weighted |= added_weighted
        result._refresh()
        return result

    def edit_nodes(self, *, add=(), remove=(), rename=None):
        """Add isolates, remove nodes/incidents or rename simultaneously without merging IDs."""
        if isinstance(add, (str, bytes, Mapping)) or not isinstance(add, Iterable):
            _error("invalid_option", "add must be an iterable of exact typed node IDs.")
        removed = self._select(remove, self._index, "unknown_node")
        if rename is not None and not isinstance(rename, dict):
            _error("invalid_option", "rename must map existing node IDs to exact new IDs.")
        renamed = {}
        for node, value in (rename or {}).items():
            node, value = _label(node), _label(value)
            if node not in self._index or node in removed:
                _error("unknown_node", "Rename requires an existing retained node.")
            self._guard(1024 * (len(renamed) + len(removed) + 1) + _identity_bytes(value))
            renamed[node] = value
        nodes, seen = [], set()
        for node in self._labels:
            if node in removed:
                continue
            value = renamed.get(node, node)
            if value in seen:
                _error("invalid_label", "Simultaneous rename cannot merge distinct node identities.")
            self._guard(1024 * (len(nodes) + 1))
            nodes.append(value)
            seen.add(value)
        for value in add:
            value = _label(value)
            self._guard(1024 * (len(nodes) + 1) + _identity_bytes(value))
            if value not in seen:
                nodes.append(value)
                seen.add(value)

        def records():
            for record in self._records():
                if record["source"] in removed or record["target"] in removed:
                    continue
                record["source"] = renamed.get(record["source"], record["source"])
                record["target"] = renamed.get(record["target"], record["target"])
                yield record
        attrs = {renamed.get(node, node): attrs for node, attrs in self._node_attributes.items() if node not in removed}
        return self._rebuild(records(), nodes, attrs)

    def to_network(self, *, reducer, zero="raise", attributes="raise", max_work=50_000_000):
        """Explicit simple projection; edge identity loss and conflict/zero policy are recorded."""
        if not isinstance(reducer, str) or reducer not in {"sum", "min", "max", "mean", "count", "binary"}:
            _error("invalid_option", "Use reducer='sum', 'min', 'max', 'mean', 'count' or 'binary'.")
        if (not isinstance(zero, str) or zero not in {"raise", "drop"}
                or not isinstance(attributes, str) or attributes not in {"raise", "drop"}):
            _error("invalid_option", "zero and attributes must be 'raise' or explicit 'drop'.")
        limit = _integer(max_work, "max_work", 2**63 - 1)
        work = 32 * self.node_count + 32 * self.edge_count * max(1, self.edge_count.bit_length())
        if work > limit:
            _error("work_budget", "Multigraph projection exceeds max_work before reduction.")
        workspace = 1024 * self.edge_count + 512 * self.node_count + 65536
        self._guard(workspace)
        with torch.device("cpu"), torch.no_grad():
            pairs = self._endpoints.T
            if not self.directed:
                pairs = torch.sort(pairs, dim=1).values
            unique, inverse, counts = torch.unique(pairs, dim=0, sorted=True, return_inverse=True, return_counts=True)
            n = len(unique)
            if reducer in {"count", "binary"}:
                aggregate = counts.to(torch.float64) if reducer == "count" else torch.ones(n, dtype=torch.float64)
            elif reducer in {"min", "max"}:
                aggregate = torch.full((n,), float("inf") if reducer == "min" else 0., dtype=torch.float64)
                aggregate.scatter_reduce_(0, inverse, self._weights,
                    reduce="amin" if reducer == "min" else "amax", include_self=True)
            else:
                aggregate = torch.zeros(n, dtype=torch.float64)
                if reducer == "mean":
                    # Sum scaled values so a finite mean need not overflow.
                    maxima = torch.zeros(n, dtype=torch.float64)
                    maxima.scatter_reduce_(0, inverse, self._weights, reduce="amax", include_self=True)
                    scale = maxima[inverse]
                    scaled = torch.where(scale > 0, self._weights / scale, 0.)
                    aggregate.index_add_(0, inverse, scaled)
                    aggregate = (aggregate / counts) * maxima
                else:
                    aggregate.index_add_(0, inverse, self._weights)
            if not bool(torch.isfinite(aggregate).all()):
                _error("precision", "Projected aggregate weights exceed float64 range.")
            zeros = int((aggregate == 0).sum())
            if zeros and zero == "raise":
                _error("projection_loss", "Projection contains zero-weight pairs; choose zero='drop' or a topology-preserving count/binary reducer.")
            edge_attrs = {}
            if attributes == "raise":
                pairs_as_labels = []
                for u, v in unique.tolist():
                    a, b = self._labels[u], self._labels[v]
                    pairs_as_labels.append((a, b) if self.directed or _key(a) <= _key(b) else (b, a))
                grouped_attrs = {}
                for k, group in enumerate(inverse.tolist()):
                    attrs = self._edge_attributes.get(self._edge_ids[k], {})
                    if group in grouped_attrs:
                        previous = grouped_attrs[group]
                        same = previous.keys() == attrs.keys() and all(
                            type(value) is type(attrs[name]) and value == attrs[name]
                            for name, value in previous.items())
                        if not same:
                            _error("projection_loss", "Parallel edges have different attributes; choose attributes='drop' explicitly.")
                    grouped_attrs[group] = attrs
                edge_attrs = {pairs_as_labels[group]: attrs for group, attrs in grouped_attrs.items()
                              if attrs and float(aggregate[group]) > 0}
            available = (self._budget.limit - self._base_bytes - workspace - 4096) / 1024**2
            rows = (dict(source=self._labels[u], target=self._labels[v], weight=float(value))
                    for (u, v), value in zip(unique.tolist(), aggregate.tolist()))
            result = network(rows, weight="weight", directed=self.directed, nodes=self._labels,
                node_attributes=self._node_attributes, edge_attributes=edge_attrs, graph_attributes=self._graph_attributes,
                max_memory_mb=available, batch_rows=32)
            result._budget.limit = self._budget.limit
            result._budget.peak += self._base_bytes + workspace + 4096
            result._metadata.update(max_memory_bytes=self._budget.limit, estimated_import_peak_bytes=result._budget.peak,
                multigraph_projection=dict(reducer=reducer, source_edge_count=self.edge_count, zero_policy=zero,
                    zero_pairs_dropped=zeros, attributes_policy=attributes, edge_ids_preserved=False,
                    work_used=work, max_work=limit))
            return result

    def write(self, path, *, format=None, overwrite=False):
        """Atomically export edge-ID GraphML/GEXF; existing paths require overwrite=True."""
        from openecon._network_io import write_network
        return write_network(self, path, format=format, overwrite=overwrite)

    def __getattr__(self, name):
        if not name.startswith("_") and callable(getattr(Network, name, None)):
            def unsupported(*args, **kwargs):
                _error("multigraph_capacity", f"Multigraph {name} is unavailable; explicitly choose to_network(reducer=...) first.")
            return unsupported
        raise AttributeError(name)


def read_multigraph(path, *, format=None, max_memory_mb=256, batch_rows=65536, max_file_mb=512):
    """Read static GraphML/GEXF with required unique edge IDs; Pajek is explicit unsupported."""
    from openecon._network_io import _format, _read_xml
    path = Path(path)
    format = _format(path, format)
    if format == "pajek":
        _error("file_feature", "Pajek edge-ID interchange is unavailable; use GraphML/GEXF or explicitly project a simple graph.")
    maximum = _Budget(max_file_mb).limit
    _integer(batch_rows, "batch_rows", 1_000_000)
    _Budget(max_memory_mb)
    if path.stat().st_size > maximum:
        _error("file_size", "Input exceeds max_file_mb.")
    try:
        return _read_xml(path, format, maximum, batch_rows, max_memory_mb, multigraph=True)
    except AnalysisError:
        raise
    except (ValueError, TypeError, UnicodeError, OverflowError):
        _error("file_format", "The multigraph file contains malformed scalar values.")
