"""Sparse graph analysis with native Torch reductions and bounded table import.

Only O(V + E) sparse indices and values are retained. The memory limit estimates
owned graph/import/algorithm buffers; it is not a limit on whole-process RSS or
on a caller's resident dataframe. No SciPy or NetworkX is used.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import heapq
from itertools import islice
import math
from numbers import Integral, Real
import sys

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon._network_device import execution, metadata as device_metadata
from openecon.dataset import Dataset
from openecon.frame import as_frame


def _error(code, message):
    raise AnalysisError("network_" + code, message)


def _integer(value, name, maximum=None, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < (0 if zero else 1) or (maximum is not None and value > maximum)):
        _error("invalid_option", f"{name} must be a {'nonnegative' if zero else 'positive'} integer"
               + (f" no larger than {maximum:,}." if maximum is not None else "."))
    return int(value)


def _missing(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, Real) and not isinstance(value, Integral) and math.isnan(float(value))


def _real(value, code, message):
    if isinstance(value, bool) or not isinstance(value, Real):
        _error(code, message)
    try:
        value = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise AnalysisError("network_" + code, message) from exc
    if not math.isfinite(value):
        _error(code, message)
    return value


def _label(value):
    if isinstance(value, str):
        if not value.strip() or len(value) > 4096 or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value):
            _error("invalid_label", "Node strings must be nonempty, bounded and contain no control characters.")
        try:
            encoded = value.encode("utf-8")
        except UnicodeError as exc:
            raise AnalysisError("network_invalid_label", "Node strings must be valid UTF-8.") from exc
        if len(encoded) > 4096:
            _error("invalid_label", "Node strings cannot exceed 4,096 UTF-8 bytes.")
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        number = int(value)
        if number.bit_length() > 256:
            _error("invalid_label", "Integer node labels cannot exceed 256 bits.")
        return number
    _error("invalid_label", "Node labels must be strings or integers; booleans, floats and missing labels are invalid.")


def _weight(value):
    value = _real(value, "invalid_weight", "Edge weights must be finite nonnegative real numbers.")
    if value < 0:
        _error("invalid_weight", "Edge weights must be finite nonnegative real numbers.")
    return value


def _key(label):
    return (0, label) if isinstance(label, int) else (1, label)


class _Budget:
    def __init__(self, megabytes):
        megabytes = _real(megabytes, "invalid_option", "max_memory_mb must be positive and finite.")
        if not 0 < megabytes <= sys.maxsize / 1024**2:
            _error("invalid_option", "max_memory_mb must be positive and finite.")
        self.limit = int(float(megabytes) * 1024**2)
        self.peak = 0
        self.check(4096)

    def check(self, size):
        self.peak = max(self.peak, int(size))
        if size > self.limit:
            _error("memory_budget", "The graph or planned operation exceeds max_memory_mb; "
                   "use a smaller graph or a larger explicit memory budget.")


def _tensor_bytes(tensor):
    return tensor._nnz() * 24


def _coalesce(index, values, count, budget, live_bytes):
    # COO inputs, sortable keys/permutation, duplicate reduction and output.
    # Shape is symbolic: this operation never allocates count-by-count storage.
    budget.check(live_bytes + 96 * len(values) + 4096)
    result = torch.sparse_coo_tensor(index, values, (count, count), dtype=torch.float64,
                                     device="cpu", check_invariants=True).coalesce()
    if not bool(torch.isfinite(result.values()).all()):
        _error("precision", "Aggregated duplicate weights overflow float64.")
    return result


def _rows(data, columns, rows):
    if isinstance(data, Dataset):
        if any(data.columns.count(name) != 1 for name in columns):
            _error("invalid_columns", "Each edge column must exist exactly once.")
        data.assert_unchanged()
        iterator = data.iter_batches(columns=columns, batch_rows=rows)
        try:
            for block in iterator:
                yield block.itertuples(index=False, name=None), len(block)
        finally:
            iterator.close()
            data.assert_unchanged()
    elif isinstance(data, pd.DataFrame):
        if any(list(data.columns).count(name) != 1 for name in columns):
            _error("invalid_columns", "Each edge column must exist exactly once.")
        for start in range(0, len(data), rows):
            block = data.iloc[start:start + rows].loc[:, columns]
            yield block.itertuples(index=False, name=None), len(block)
    elif isinstance(data, Mapping):
        if any(name not in data for name in columns):
            _error("invalid_columns", "Every requested edge column is required.")
        try:
            sizes = [len(data[name]) for name in columns]
        except TypeError as exc:
            raise AnalysisError("network_invalid_data", "Column mappings need sized sequences.") from exc
        if len(set(sizes)) != 1 or any(isinstance(data[name], (str, bytes)) for name in columns):
            _error("invalid_data", "Edge columns must be equal-length sequences.")
        for start in range(0, sizes[0], rows):
            stop = min(sizes[0], start + rows)
            yield ((data[name].iloc[i] if isinstance(data[name], pd.Series) else data[name][i]
                    for name in columns) for i in range(start, stop)), stop - start
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        iterator = iter(data)
        try:
            while block := list(islice(iterator, rows)):
                if any(not isinstance(record, Mapping) or any(name not in record for name in columns)
                       for record in block):
                    _error("invalid_data", "Records must be mappings containing every requested edge column.")
                yield ((record[name] for name in columns) for record in block), len(block)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    else:
        _error("invalid_data", "Use an edge dataframe, column mapping, records or Dataset.")


def network(data, source="source", target="target", weight=None, directed=False,
            nodes=None, missing="raise", batch_rows=65536, max_memory_mb=256,
            store=None, max_disk_mb=4096, cancelled=None, node_attributes=None,
            edge_attributes=None, graph_attributes=None):
    """Build a sparse graph snapshot; zero-weight endpoints still retain nodes."""
    if store is not None or isinstance(data, Dataset) or cancelled is not None:
        from openecon._network_store import build_store
        result = build_store(data, store, source=source, target=target, weight=weight,
            directed=directed, nodes=nodes, missing=missing, batch_rows=batch_rows,
            max_memory_mb=max_memory_mb, max_disk_mb=max_disk_mb, cancelled=cancelled,
            node_attributes=node_attributes, edge_attributes=edge_attributes,
            graph_attributes=graph_attributes)
        if store is None and result.resident_import_bound <= result.metadata["max_memory_bytes"]:
            try:
                return result.materialize()
            finally:
                result.close()
        return result
    if not isinstance(directed, bool) or not isinstance(missing, str) or missing not in {"raise", "drop"}:
        _error("invalid_option", "directed must be boolean; missing must be 'raise' or 'drop'.")
    columns = [source, target] + ([weight] if weight is not None else [])
    if (any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in columns)
            or len(set(columns)) != len(columns)):
        _error("invalid_columns", "Source, target and optional weight must be distinct valid column names.")
    requested_rows = _integer(batch_rows, "batch_rows", 1_000_000)
    budget = _Budget(max_memory_mb)
    rows = min(requested_rows, max(1, (budget.limit - 4096) // 2048))
    labels, label_index = [], {}
    label_bytes = 4096
    chunks = []
    input_rows = missing_rows = zero_rows = positive_rows = 0
    peak_rows = 0
    digest = hashlib.sha256()

    def retained():
        return sum(_tensor_bytes(item) for item in chunks if item is not None)

    def intern(raw):
        nonlocal label_bytes
        value = _label(raw)
        if value not in label_index:
            # Includes hash-table growth, index integers, references and UTF-8
            # label storage. Reserve before inserting a new node.
            addition = 512 + 2 * sys.getsizeof(value) + (len(value.encode("utf-8")) if isinstance(value, str) else 32)
            budget.check(label_bytes + addition + retained() + 1024 * rows)
            label_index[value] = len(labels)
            labels.append(value)
            label_bytes += addition
        return label_index[value]

    if nodes is not None:
        if not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes, Mapping)):
            _error("invalid_label", "nodes must be an iterable of string/integer isolate labels.")
        for label in nodes:
            intern(label)
    with torch.device("cpu"), torch.no_grad():
        for block, size in _rows(data, columns, rows):
            budget.check(label_bytes + retained() + 1024 * size)
            left, right, weights = [], [], []
            peak_rows = max(peak_rows, size)
            for record in block:
                record = tuple(record)
                input_rows += 1
                if any(_missing(value) for value in record):
                    if missing == "raise":
                        _error("missing_data", "Edge endpoints/weights contain missing values; choose missing='drop' explicitly.")
                    missing_rows += 1
                    continue
                i, j = intern(record[0]), intern(record[1])
                value = _weight(record[2]) if weight is not None else 1.0
                digest.update(repr((_key(labels[i]), _key(labels[j]), value)).encode("utf-8") + b"\n")
                if value == 0:
                    zero_rows += 1
                    continue
                positive_rows += 1
                left.append(i if directed else min(i, j))
                right.append(j if directed else max(i, j))
                weights.append(value)
            if not weights:
                continue
            budget.check(label_bytes + retained() + 1024 * size + 120 * len(weights))
            index = torch.tensor([left, right], dtype=torch.int64)
            values = torch.tensor(weights, dtype=torch.float64)
            level = 0
            chunk = _coalesce(index, values, len(labels), budget, label_bytes + retained() + 1024 * size)
            del index, values, left, right, weights
            while level < len(chunks) and chunks[level] is not None:
                previous = chunks[level]
                total = previous._nnz() + chunk._nnz()
                budget.check(label_bytes + retained() + _tensor_bytes(chunk) + 120 * total)
                index = torch.cat([previous.indices(), chunk.indices()], dim=1)
                values = torch.cat([previous.values(), chunk.values()])
                chunk = _coalesce(index, values, len(labels), budget,
                                  label_bytes + retained() + _tensor_bytes(chunk))
                chunks[level] = None
                del index, values, previous
                level += 1
            if level == len(chunks):
                chunks.append(chunk)
            else:
                chunks[level] = chunk
        pieces = [part for part in chunks if part is not None]
        total = sum(part._nnz() for part in pieces)
        budget.check(label_bytes + retained() + 120 * total + 4096)
        if pieces:
            index = torch.cat([part.indices() for part in pieces], dim=1)
            values = torch.cat([part.values() for part in pieces])
        else:
            index = torch.empty((2, 0), dtype=torch.int64)
            values = torch.empty(0, dtype=torch.float64)
        edges = _coalesce(index, values, len(labels), budget, label_bytes + retained())
        del index, values, pieces
        chunks.clear()
        metadata = {"input_rows": input_rows, "missing_rows_dropped": missing_rows,
                    "zero_weight_rows_dropped": zero_rows, "positive_edge_rows": positive_rows,
                    "duplicate_edge_rows_aggregated": positive_rows - edges._nnz(),
                    "requested_batch_rows": requested_rows, "effective_batch_rows": rows,
                    "actual_peak_batch_rows": peak_rows, "input_sha256": digest.hexdigest(),
                    "import_algorithm": "bounded COO coalescing with binary-level chunk merges",
                    "memory_scope": "estimated owned graph and planned buffers; excludes caller input and total process RSS"}
        result = Network(tuple(labels), label_index, edges, directed, weight is not None,
                         budget, label_bytes, metadata)
        if any(value is not None for value in (node_attributes, edge_attributes, graph_attributes)):
            from openecon._network_io import attach_attributes
            result = attach_attributes(result, nodes=node_attributes, edges=edge_attributes,
                                       graph_attributes=graph_attributes)
        return result


def open_network(path, *, max_memory_mb=256, verify_source=True):
    """Reopen an atomically completed disk graph snapshot."""
    from openecon._network_store import open_store
    return open_store(path, max_memory_mb=max_memory_mb, verify_source=verify_source)


class Network:
    """A CPU sparse snapshot; analytical results always use the complete graph."""

    def __init__(self, labels, index, edges, directed, weighted, budget, label_bytes, metadata):
        self._labels, self._index = labels, index
        self._edges, self.directed, self.weighted = edges, directed, weighted
        self._budget, self._label_bytes, self._metadata = budget, label_bytes, metadata
        n, e = len(labels), edges._nnz()
        budget.check(label_bytes + 160 * e + 128 * n + 4096)
        with torch.device("cpu"), torch.no_grad():
            if directed:
                self._arcs = edges
            else:
                pairs, weights = edges.indices(), edges.values()
                off = pairs[0] != pairs[1]
                arcs = torch.cat([pairs, pairs[:, off].flip(0)], dim=1)
                values = torch.cat([weights, weights[off]])
                self._arcs = torch.sparse_coo_tensor(arcs, values, (n, n), dtype=torch.float64,
                                                     device="cpu", is_coalesced=False, check_invariants=True)
        self._node_attributes, self._edge_attributes, self._graph_attributes = {}, {}, {}
        self._base_bytes = label_bytes + _tensor_bytes(edges) + (0 if directed else _tensor_bytes(self._arcs))
        self._metadata.update(node_count=n, edge_count=e, directed=directed, weighted=weighted,
                              sparse_storage_bytes=self._base_bytes - label_bytes,
                              estimated_owned_graph_bytes=self._base_bytes,
                              max_memory_bytes=budget.limit,
                              estimated_import_peak_bytes=budget.peak)

    @property
    def node_count(self):
        return len(self._labels)

    @property
    def edge_count(self):
        return self._edges._nnz()

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def __repr__(self):
        return f"Network(nodes={self.node_count}, edges={self.edge_count}, directed={self.directed})"

    def _guard(self, workspace):
        self._budget.check(self._base_bytes + int(workspace) + 4096)

    def _frame(self, columns, **metadata):
        result = as_frame(pd.DataFrame({"node": pd.Series(self._labels, dtype=object), **columns}))
        result.attrs.update(network=self.metadata, **metadata)
        return result

    def _degrees(self, device="cpu"):
        workspace = 256 * self.node_count + 64 * self.edge_count
        with execution(self, device, workspace) as selected:
            pairs = self._edges.indices().to(selected)
            weights, n = self._edges.values().to(selected), self.node_count
            out = torch.bincount(pairs[0], minlength=n)
            incoming = torch.bincount(pairs[1], minlength=n)
            out_w, in_w = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
            out_w.index_add_(0, pairs[0], weights)
            in_w.index_add_(0, pairs[1], weights)
            if not bool(torch.isfinite(out_w).all() & torch.isfinite(in_w).all()):
                _error("precision", "A node strength exceeds float64 range.")
            return out, incoming, out_w, in_w

    def degree(self, *, device="cpu"):
        """Unique-edge degrees and aggregate strengths; undirected loops count twice."""
        workspace = 256 * self.node_count + 64 * self.edge_count
        with execution(self, device, workspace) as selected:
            out, incoming, out_w, in_w = self._degrees(str(selected))
            info = device_metadata(selected, workspace)
            if self.directed:
                return self._frame({"in_degree": incoming.cpu().numpy(), "out_degree": out.cpu().numpy(),
                                    "in_strength": in_w.cpu().numpy(), "out_strength": out_w.cpu().numpy()},
                                   kind="network_degree", **info)
            strength = out_w + in_w
            if not bool(torch.isfinite(strength).all()):
                _error("precision", "A node strength exceeds float64 range.")
            return self._frame({"degree": (out + incoming).cpu().numpy(), "strength": strength.cpu().numpy()},
                               kind="network_degree", **info)

    def pagerank(self, damping=.85, tol=1e-10, max_iter=200, personalization=None, device="cpu"):
        """Weighted PageRank; dangling mass follows the normalized personalization."""
        damping = _real(damping, "invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
        tol = _real(tol, "invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
        if not 0 <= damping < 1 or tol <= 0:
            _error("invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
        max_iter = _integer(max_iter, "max_iter", 1_000_000)
        n, a = self.node_count, self._arcs._nnz()
        workspace = 160 * n + 96 * a + 4096
        with execution(self, device, workspace) as selected:
            probability = torch.zeros(n, dtype=torch.float64, device="cpu")
            if personalization is None:
                if n:
                    probability.fill_(1 / n)
            else:
                if not isinstance(personalization, Mapping):
                    _error("invalid_option", "personalization must map known node labels to nonnegative weights.")
                for label, value in personalization.items():
                    label = _label(label)
                    if label not in self._index:
                        _error("unknown_node", "Personalization names a node outside the graph.")
                    probability[self._index[label]] = _weight(value)
                if not n or float(probability.max()) == 0:
                    _error("invalid_option", "Personalization must have positive total weight.")
                probability /= probability.max()
                probability /= probability.sum()
            if not n:
                return self._frame({"pagerank": []}, converged=True, iterations=0,
                                   l1_change=0., error_bound_estimate=0., algorithm="weighted sparse power iteration",
                                   **device_metadata(selected, workspace))
            pairs = self._arcs._indices().to(selected)
            weights = self._arcs._values().to(selected)
            probability = probability.to(selected)
            maximum = torch.zeros(n, dtype=torch.float64, device=selected)
            maximum.scatter_reduce_(0, pairs[0], weights, reduce="amax", include_self=True)
            scaled = weights / maximum[pairs[0]]
            totals = torch.zeros(n, dtype=torch.float64, device=selected)
            totals.index_add_(0, pairs[0], scaled)
            transition = scaled / totals[pairs[0]]
            dangling = totals == 0
            rank = probability.clone()
            delta = bound = math.inf
            for iteration in range(1, max_iter + 1):
                following = ((1 - damping) + damping * rank[dangling].sum()) * probability
                following.index_add_(0, pairs[1], damping * rank[pairs[0]] * transition)
                following /= following.sum()
                delta = float((following - rank).abs().sum())
                bound = float(damping / (1 - damping) * delta)
                rank = following
                if bound <= tol:
                    break
            else:
                _error("nonconvergence", f"PageRank did not reach tol within {max_iter} iterations; "
                       f"last L1 step={delta:.6g}, contraction error estimate={bound:.6g}.")
            return self._frame({"pagerank": rank.cpu().numpy()}, converged=True, iterations=iteration,
                               l1_change=delta, error_bound_estimate=bound, tol=float(tol), damping=float(damping),
                               algorithm="weighted sparse power iteration", dangling_distribution="personalization",
                               probability_sum=float(rank.sum()), **device_metadata(selected, workspace))

    def _components(self):
        self._guard(256 * self.node_count + 128 * min(self.edge_count, 65536))
        parent, size = list(range(self.node_count)), [1] * self.node_count

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pairs = self._edges.indices()
        for start in range(0, self.edge_count, 65536):
            left, right = pairs[:, start:start + 65536].tolist()
            for i, j in zip(left, right):
                i, j = find(i), find(j)
                if i != j:
                    if size[i] < size[j]:
                        i, j = j, i
                    parent[j] = i
                    size[i] += size[j]
        roots = [find(i) for i in range(self.node_count)]
        smallest = {}
        for i, root in enumerate(roots):
            key = _key(self._labels[i])
            if root not in smallest or key < smallest[root]:
                smallest[root] = key
        groups = {root: i for i, root in enumerate(sorted(smallest, key=smallest.get))}
        return [groups[root] for root in roots]

    def components(self, connectivity="weak"):
        """Weak or strong components, with deterministic IDs and every isolate."""
        if connectivity == "strong":
            from openecon._network_topology import components_strong
            return components_strong(self)
        if connectivity != "weak":
            _error("invalid_option", "connectivity must be 'weak' or 'strong'.")
        return self._frame({"component": self._components()}, kind="network_components", connectivity="weak")

    def shortest_paths(self, source):
        """One-source BFS/indexed Dijkstra; infinity means no reachable path."""
        source = _label(source)
        if source not in self._index:
            _error("unknown_node", "The shortest-path source is outside the graph.")
        from openecon._network_sparse import csr
        from openecon._network_centrality import _paths
        self._guard(256 * self.node_count + 64 * self._arcs._nnz())
        distance, _, _ = _paths(csr(self), self._index[source], self.weighted)
        return self._frame({"distance": distance}, kind="network_shortest_paths", source=source,
                           algorithm="Dijkstra" if self.weighted else "BFS", disconnected="infinity",
                           distance_semantics="sum of aggregate edge weights" if self.weighted else "edge count")

    def communities(self, method="leiden", resolution=1, seed=0, max_iter=100,
                    tol=1e-10, *, max_work=50_000_000):
        """Seeded Louvain or genuine Leiden refinement with weighted modularity."""
        from openecon._network_communities import communities
        return communities(self, method, resolution, seed, max_iter, tol, max_work=max_work)

    def modularity(self, membership, resolution=1, *, max_work=50_000_000, device="cpu"):
        """Weighted modularity of a complete node-to-community mapping or table."""
        from openecon._network_communities import modularity
        return modularity(self, membership, resolution, max_work=max_work, device=device)

    def betweenness(self, samples=None, normalized=True, endpoints=False, seed=0,
                    max_work=50_000_000):
        """Exact Brandes, or explicit unbiased source sampling via samples=k."""
        from openecon._network_centrality import betweenness
        return betweenness(self, samples, normalized, endpoints, seed, max_work)

    def closeness(self, direction="in", wf_improved=True, max_work=50_000_000):
        """Exact incoming/outgoing closeness with optional disconnected correction."""
        from openecon._network_centrality import closeness
        return closeness(self, direction, wf_improved, max_work)

    def harmonic(self, direction="in", max_work=50_000_000):
        """Exact incoming/outgoing harmonic centrality; unreachable terms are zero."""
        from openecon._network_centrality import harmonic
        return harmonic(self, direction, max_work)

    def eigenvector(self, max_iter=1000, tol=1e-10, *, device="cpu"):
        """Sparse strength-based incoming eigenvector, with a residual check."""
        from openecon._network_centrality import eigenvector
        return eigenvector(self, max_iter, tol, device=device)

    def hits(self, *, max_iter=1000, tol=1e-10, normalization="l1", max_work=50_000_000, device="cpu"):
        """Weighted hub/authority singular vectors, with checked sparse residuals."""
        from openecon._network_spectral import hits
        return hits(self, max_iter=max_iter, tol=tol, normalization=normalization, max_work=max_work, device=device)

    def katz(self, alpha=None, beta=1., *, direction="in", max_iter=1000, tol=1e-10,
             normalization="l2", max_work=50_000_000, device="cpu"):
        """Sparse Katz centrality with a sufficient contraction bound; alpha=None is safe auto attenuation."""
        from openecon._network_spectral import katz
        return katz(self, alpha, beta, direction=direction, max_iter=max_iter, tol=tol,
                    normalization=normalization, max_work=max_work, device=device)

    def shortest_path(self, source, target, *, max_work=50_000_000):
        """A deterministic source-to-target route, with cumulative distances and reachability."""
        from openecon._network_paths import shortest_path
        return shortest_path(self, source, target, max_work=max_work)




    def distances(self, sources=None, targets=None, *, direction="out",
                  max_work=50_000_000, max_pairs=1_000_000):
        """Selected ordered pair distances as a bounded long table, without dense V-by-V storage."""
        from openecon._network_paths import distances
        return distances(self, sources, targets, direction=direction, max_work=max_work, max_pairs=max_pairs)

    def eccentricity(self, *, direction="out", disconnected="infinite", max_work=50_000_000):
        """Exact node eccentricity with an explicit disconnected-graph convention."""
        from openecon._network_paths import eccentricity
        return eccentricity(self, direction=direction, disconnected=disconnected, max_work=max_work)

    def distance_summary(self, *, direction="out", disconnected="infinite", max_work=50_000_000):
        """Exact diameter, radius, average distance and efficiency, with unreachable pairs reported."""
        from openecon._network_paths import distance_summary
        return distance_summary(self, direction=direction, disconnected=disconnected, max_work=max_work)

    def bridges(self, *, max_work=50_000_000):
        """Edges whose removal disconnects the undirected simple/weak projection."""
        from openecon._network_paths import bridges
        return bridges(self, max_work=max_work)

    def k_shortest_paths(self, source, target, *, k=5, max_path_length=None,
                         max_output_nodes=100_000, max_frontier=100_000, max_work=50_000_000):
        """Exact cost-ordered simple paths with explicit hop, output, frontier and work budgets."""
        from openecon._network_bounded_paths import k_shortest_paths
        return k_shortest_paths(self, source, target, k=k, max_path_length=max_path_length,
                               max_output_nodes=max_output_nodes, max_frontier=max_frontier, max_work=max_work)

    def strong_bridges(self, *, max_work=50_000_000):
        """Oriented arcs whose deletion increases strongly connected component count."""
        from openecon._network_bounded_paths import strong_cuts
        return strong_cuts(self, vertices=False, max_work=max_work)

    def strong_articulation_points(self, *, max_work=50_000_000):
        """Directed vertex deletion flags under strong connectivity; no weak projection."""
        from openecon._network_bounded_paths import strong_cuts
        return strong_cuts(self, vertices=True, max_work=max_work)

    def articulation_points(self, *, max_work=50_000_000):
        """Cut-vertex flags from iterative Tarjan on the simple/weak projection."""
        from openecon._network_paths import articulation_points
        return articulation_points(self, max_work=max_work)

    def minimum_spanning_forest(self, *, max_work=50_000_000):
        """Minimum-cost forest of an undirected graph, retaining isolates and scalar attributes."""
        from openecon._network_paths import minimum_spanning_forest
        return minimum_spanning_forest(self, max_work=max_work)

    def max_flow(self, source, target, *, max_work=50_000_000):
        """Certified capacity flow and s-t cut; undirected flows use a signed shared capacity."""
        from openecon._network_flow import max_flow
        return max_flow(self, source, target, max_work)

    def min_cut(self, source, target, *, max_work=50_000_000):
        """Minimum-capacity s-t cut with its maximum-flow conservation certificate."""
        from openecon._network_flow import min_cut
        return min_cut(self, source, target, max_work)

    def global_min_cut(self, *, max_work=50_000_000):
        """Minimum-capacity global cut; directed graphs count only outgoing crossing arcs."""
        from openecon._network_cut import global_min_cut
        return global_min_cut(self, max_work=max_work)

    def min_cost_flow(self, demands, costs, *, capacities=None, domain="fractional",
                      max_work=50_000_000, max_entries=1_000_000, max_pivots=10000):
        """Minimum-cost directed flow; positive node demand is net inflow.

        Capacities default to graph weights. Costs map (source,target) pairs or
        follow coalesced edge order. Integral mode requires integral capacities
        and demands. Checked optimality, infeasibility and unboundedness certificates.
        """
        from openecon._network_cost_flow import cost_flow
        return cost_flow(self, {"flow": demands}, costs, capacities=capacities, domain=domain,
                         max_work=max_work, max_entries=max_entries, max_pivots=max_pivots)

    def multicommodity_flow(self, commodities, costs, *, capacities=None, domain="fractional",
                            max_work=50_000_000, max_entries=1_000_000, max_pivots=10000):
        """Fractional splittable flows share edge capacities; integer commodities are unsupported."""
        from openecon._network_cost_flow import cost_flow
        return cost_flow(self, commodities, costs, capacities=capacities, domain=domain,
                         max_work=max_work, max_entries=max_entries, max_pivots=max_pivots)

    def triad_census(self, *, max_work=50_000_000):
        """Exact induced three-node counts: 16 directed triads or four undirected motifs."""
        from openecon._network_motifs import triad_census
        return triad_census(self, max_work=max_work)

    def graphlets(self, size=4, *, connected=True, max_subgraphs=1_000_000,
                  max_output_rows=1_000_000, max_work=50_000_000):
        """Exact induced 4/5-node topology classes and automorphism orbits."""
        from openecon._network_graphlets import graphlets
        return graphlets(self, size, connected=connected, max_subgraphs=max_subgraphs,
                         max_output_rows=max_output_rows, max_work=max_work)

    def qap_correlation(self, other, *, values="weight", include_loops=False,
                        permutations=999, seed=0, alternative="two-sided", max_work=50_000_000):
        """Sparse dyad Pearson correlation and node-label QAP permutation test between aligned networks."""
        from openecon._network_qap import qap_correlation
        return qap_correlation(self, other, values=values, include_loops=include_loops,
                               permutations=permutations, seed=seed, alternative=alternative,
                               max_work=max_work)

    def qap_regression(self, predictors, *, values="weight", include_loops=False,
                       permutations=999, seed=0, alternative="two-sided",
                       intercept=True, method="freedman_lane", joint=None, adjustment="none", max_work=50_000_000):
        """Fit dyad OLS with coefficient-specific Freedman-Lane MRQAP tests.

        Fixed predictor networks and node-permuted reduced-model residuals
        define an approximate conditional test using partial correlations.
        Absent dyads count as zero; no dense node-by-node matrix is created.
        """
        from openecon._network_mrqap import qap_regression
        return qap_regression(self, predictors, values=values, include_loops=include_loops,
                              permutations=permutations, seed=seed, alternative=alternative,
                              intercept=intercept, method=method, joint=joint, adjustment=adjustment, max_work=max_work)

    def block_model(self, groups, *, values="binary", initial=None, seed=0,
                    starts=4, max_iter=100, tol=1e-9, max_work=50_000_000):
        """Fit a fixed-group hard-label Bernoulli stochastic block model.

        Coordinate ascent uses sparse edge and small block-count buffers.
        Positive edges define binary presence; loops are excluded. The best
        fitted local solution is returned with explicit convergence metadata.
        """
        from openecon._network_sbm import block_model
        return block_model(self, groups, values=values, initial=initial, seed=seed,
                           starts=starts, max_iter=max_iter, tol=tol, max_work=max_work)

    def poisson_block_model(self, groups, *, initial=None, seed=0, starts=4,
                            max_iter=100, tol=1e-9, max_work=50_000_000):
        """Fit fixed-group Poisson blocks to aggregate integer interaction counts.

        Off-diagonal dyads include unobserved zeros; loops are excluded.
        Sparse coordinate ascent returns a local fit with its complete
        Poisson likelihood, fitted mean counts and convergence metadata.
        Continuous edge strengths are not count observations.
        """
        from openecon._network_poisson import poisson_block_model
        return poisson_block_model(self, groups, initial=initial, seed=seed, starts=starts,
                                   max_iter=max_iter, tol=tol, max_work=max_work)

    def degree_corrected_block_model(self, groups, *, initial=None, seed=0, starts=4,
                                     max_iter=100, tol=1e-9, max_work=50_000_000):
        """Fit fixed-group degree-corrected Poisson blocks to integer counts.

        The full multigraph sample space includes zero and observed loops.
        Undirected raw loop counts use half-rate intensity; directed models
        estimate separate incoming/outgoing node parameters. Sparse local
        ascent reports convergence and explicitly unidentified zero-stub groups.
        """
        from openecon._network_dc_sbm import degree_corrected_block_model
        return degree_corrected_block_model(self, groups, initial=initial, seed=seed, starts=starts,
                                           max_iter=max_iter, tol=tol, max_work=max_work)

    def maximum_matching(self, partition=None, *, max_work=50_000_000):
        """Maximum-cardinality bipartite matching, with a minimum-vertex-cover certificate."""
        from openecon._network_flow import maximum_matching
        return maximum_matching(self, partition, max_work)

    def weighted_assignment(self, partition=None, *, objective="weight", max_matrix_entries=1_000_000,
                            max_work=50_000_000):
        """Exact weighted bipartite assignment with optional unmatched nodes."""
        from openecon._network_matching import weighted_assignment
        return weighted_assignment(self, partition, objective=objective,
            max_matrix_entries=max_matrix_entries, max_work=max_work)

    def general_matching(self, *, objective="weight", max_component_nodes=24,
                         max_states=1_000_000, max_work=50_000_000):
        """Bounded exact matching of undirected general graphs, including odd cycles."""
        from openecon._network_matching import general_matching
        return general_matching(self, objective=objective, max_component_nodes=max_component_nodes,
            max_states=max_states, max_work=max_work)

    def bipartite(self, partition=None):
        """Infer or validate exact binary partitions on the simple/weak projection."""
        from openecon._network_bipartite import bipartite
        return bipartite(self, partition)

    def bipartite_projection(self, partition=None, onto=0, weight="count", *,
                            max_work=50_000_000, max_edges=1_000_000):
        """Project one side using shared-neighbor count, binary presence or strength products."""
        from openecon._network_bipartite import bipartite_projection
        return bipartite_projection(self, partition, onto, weight, max_work=max_work, max_edges=max_edges)

    def link_prediction(self, pairs, method="jaccard", source="source", target="target", *,
                        max_pairs=100_000, max_work=50_000_000, batch_rows=65536):
        """Score explicit undirected candidate links with bounded common-neighbor similarities."""
        from openecon._network_bipartite import link_prediction
        return link_prediction(self, pairs, method, source, target, max_pairs=max_pairs,
                               max_work=max_work, batch_rows=batch_rows)

    def triangles(self, *, max_work=50_000_000):
        """Exact simple-topology triangle counts; loops and strengths are ignored."""
        from openecon._network_topology import triangles
        return triangles(self, max_work=max_work)

    def clustering(self, *, max_work=50_000_000):
        """Local binary clustering; directed graphs use Fagiolo's convention."""
        from openecon._network_topology import clustering
        return clustering(self, max_work=max_work)

    def core_numbers(self):
        """Linear bin-peeling core numbers; directed graphs use weak projection."""
        from openecon._network_topology import k_core
        return k_core(self)

    def k_core(self, k=None):
        """All core numbers, or the induced graph whose core number is at least k."""
        result = self.core_numbers()
        if k is None:
            return result
        k = _integer(k, "k", zero=True)
        return self.subgraph(result.loc[result.core_number >= k, "node"].tolist())

    def density(self):
        """Binary density excluding loops and duplicate connections."""
        from openecon._network_topology import density
        return density(self)

    def transitivity(self, *, max_work=50_000_000):
        """Global binary clustering with an explicit directed/undirected convention."""
        from openecon._network_topology import transitivity
        return transitivity(self, max_work=max_work)

    def assortativity(self):
        """Binary degree Pearson assortativity; undefined variance returns NaN."""
        from openecon._network_topology import assortativity
        return assortativity(self)

    def topology_summary(self, *, max_work=50_000_000):
        """Structural summary with clustering, triangles, cores and connectivity."""
        from openecon._network_topology import topology_summary
        return topology_summary(self, max_work=max_work)

    @property
    def node_attributes(self):
        """Independent scalar node attributes, keyed by exact typed node identity."""
        return deepcopy(self._node_attributes)

    @property
    def edge_attributes(self):
        """Independent scalar attributes on canonical aggregate edges."""
        return deepcopy(self._edge_attributes)

    @property
    def graph_attributes(self):
        return deepcopy(self._graph_attributes)

    def nodes(self, *, attributes=True):
        """Return exact typed IDs, full-graph degrees and ``attr.NAME`` columns."""
        from openecon._network_workbench import nodes_table
        return nodes_table(self, attributes=attributes)

    def edges(self, *, attributes=True):
        """Return canonical aggregate edges and ``attr.NAME`` scalar columns."""
        from openecon._network_workbench import edges_table
        return edges_table(self, attributes=attributes)

    def edit_nodes(self, *, add=(), remove=(), rename=None):
        """Return a new sparse snapshot after simultaneous exact-ID edits."""
        from openecon._network_workbench import edit_nodes
        return edit_nodes(self, add=add, remove=remove, rename=rename)

    def edit_edges(self, *, add=(), remove=(), weights=None):
        """Return a new snapshot after adding/removing dyads or setting weights."""
        from openecon._network_workbench import edit_edges
        return edit_edges(self, add=add, remove=remove, weights=weights)

    def update_attributes(self, *, nodes=None, edges=None, graph_attributes=None):
        """Merge scalar attribute updates without changing the source snapshot."""
        from openecon._network_workbench import update_attributes
        return update_attributes(self, nodes=nodes, edges=edges, graph_attributes=graph_attributes)

    def rename_attributes(self, mapping, *, scope="nodes"):
        """Rename attribute columns simultaneously, rejecting collisions."""
        from openecon._network_workbench import attribute_columns
        return attribute_columns(self, scope=scope, rename=mapping)

    def drop_attributes(self, names, *, scope="nodes"):
        """Remove selected scalar attribute columns from a new snapshot."""
        from openecon._network_workbench import attribute_columns
        return attribute_columns(self, scope=scope, drop=names)

    def filter(self, *, nodes=None, edges=None):
        """Apply Python callable or field/op/value filters to the complete graph."""
        from openecon._network_workbench import filter_graph
        return filter_graph(self, nodes=nodes, edges=edges)

    def with_positions(self, positions, *, fixed=True):
        """Save layout coordinates keyed by exact typed node IDs."""
        from openecon._network_workbench import with_positions
        return with_positions(self, positions, fixed=fixed)

    @classmethod
    def from_plot_data(cls, data, *, max_memory_mb=256):
        """Rebuild a saved displayed subgraph from explicit typed identities."""
        from openecon._network_workbench import from_plot_data
        return from_plot_data(data, max_memory_mb=max_memory_mb)

    def with_attributes(self, *, nodes=None, edges=None, graph_attributes=None):
        """Return a new snapshot sharing tensors, with validated scalar attributes."""
        from openecon._network_io import attach_attributes
        result = object.__new__(Network)
        result.__dict__ = self.__dict__.copy()
        result._metadata = self.metadata
        result._budget = _Budget(self._budget.limit / 1024**2)
        result._base_bytes -= self._metadata.get("attribute_storage_bytes", 0)
        return attach_attributes(result, nodes=self._node_attributes if nodes is None else nodes,
                                 edges=self._edge_attributes if edges is None else edges,
                                 graph_attributes=self._graph_attributes if graph_attributes is None else graph_attributes)

    def subgraph(self, nodes):
        """A complete induced graph for selected exact node IDs, retaining isolates."""
        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, Iterable):
            _error("invalid_option", "nodes must be an iterable of exact node IDs.")
        selected = set()
        for node in nodes:
            node = _label(node)
            if node not in self._index:
                _error("unknown_node", "Subgraph selection contains an unknown node.")
            selected.add(node)
            self._guard(256 * len(selected))
        chosen = [label for label in self._labels if label in selected]
        self._guard(256 * len(selected) + 128 * min(self.edge_count, 65536))
        from openecon._network_io import _edge_records
        def records():
            for u, v, value in _edge_records(self):
                left, right = self._labels[u], self._labels[v]
                if left in selected and right in selected:
                    yield {"source": left, "target": right, "weight": value}
        result = network(records(), nodes=chosen, weight="weight", directed=self.directed,
                         max_memory_mb=self._budget.limit / 1024**2)
        result.weighted = self.weighted
        result._metadata.update(weighted=self.weighted, derived_from="induced subgraph",
                                parent_node_count=self.node_count, parent_edge_count=self.edge_count)
        return result.with_attributes(nodes={label: attrs for label, attrs in self._node_attributes.items() if label in selected},
                                      edges={pair: attrs for pair, attrs in self._edge_attributes.items() if all(label in selected for label in pair)},
                                      graph_attributes=self._graph_attributes)

    def write(self, path, *, format=None, overwrite=False):
        """Atomically write a complete static GraphML, GEXF or Pajek graph."""
        from openecon._network_io import write_network
        return write_network(self, path, format=format, overwrite=overwrite)

    def to_graphml(self, path, *, overwrite=False):
        """Atomically export the complete static GraphML graph, with scalar attributes."""
        return self.write(path, format="graphml", overwrite=overwrite)

    def to_gexf(self, path, *, overwrite=False):
        """Atomically export a complete static GEXF graph, retaining typed native IDs."""
        return self.write(path, format="gexf", overwrite=overwrite)

    def to_pajek(self, path, *, overwrite=False):
        """Atomically export complete Pajek vertices and edges, including isolates."""
        return self.write(path, format="pajek", overwrite=overwrite)

    def summary(self):
        """Publication-ready summary; import counters describe physical source rows."""
        pairs = self._edges.indices()
        records = [("Nodes", self.node_count), ("Edges", self.edge_count), ("Directed", self.directed),
                   ("Weighted input", self.weighted), ("Self-loops", int((pairs[0] == pairs[1]).sum())),
                   ("Input rows", self._metadata["input_rows"]),
                   ("Missing rows dropped", self._metadata["missing_rows_dropped"]),
                   ("Zero-weight rows dropped", self._metadata["zero_weight_rows_dropped"]),
                   ("Duplicate rows aggregated", self._metadata["duplicate_edge_rows_aggregated"])]
        result = as_frame(pd.DataFrame(records, columns=["Metric", "Value"]))
        result.attrs.update(network=self.metadata, kind="network_summary")
        return result

    def to_plot_data(self, max_nodes=1000, max_edges=5000, seed=0, groups=None):
        """Deterministic capped highest-degree induced subgraph; no analysis sampling."""
        max_nodes = _integer(max_nodes, "max_nodes", 100_000)
        max_edges = _integer(max_edges, "max_edges", 1_000_000)
        _integer(seed, "seed", 2**32 - 1, zero=True)
        shown_nodes, shown_edges = min(max_nodes, self.node_count), min(max_edges, self.edge_count)
        self._guard(384 * self.node_count + 512 * shown_edges + 512 * shown_nodes
                    + 128 * min(self.edge_count, 65536))
        with torch.device("cpu"), torch.no_grad():
            pairs = self._edges.indices()
            degree = (torch.bincount(pairs[0], minlength=self.node_count)
                      + torch.bincount(pairs[1], minlength=self.node_count))
        degree_values = memoryview(degree.numpy())
        selected = heapq.nsmallest(max_nodes, range(self.node_count),
                                  key=lambda i: (-degree_values[i], _key(self._labels[i])))
        label_bytes = sum(len(self._labels[node].encode("utf-8")) if isinstance(self._labels[node], str)
                          else len(str(self._labels[node])) for node in selected)
        self._guard(384 * self.node_count + 512 * shown_edges + 512 * shown_nodes
                    + 128 * min(self.edge_count, 65536) + 2 * label_bytes)
        positions = {node: i for i, node in enumerate(selected)}
        grouping = "Weak components"
        if groups is None:
            groups = self._components()
        else:
            from openecon._network_communities import _membership
            method = getattr(groups, "attrs", {}).get("method")
            grouping = (f"{method.title()} communities" if method in {"louvain", "leiden"}
                        else {"bernoulli_sbm": "Bernoulli SBM blocks",
                              "poisson_sbm": "Poisson SBM blocks",
                              "degree_corrected_poisson_sbm": "Degree-corrected Poisson SBM blocks"}
                        .get(method, "User groups"))
            groups = _membership(self, groups)

        def induced():
            pairs, weights = self._edges.indices(), self._edges.values()
            for start in range(0, self.edge_count, 65536):
                left, right = pairs[:, start:start + 65536].tolist()
                for u, v, weight in zip(left, right, weights[start:start + 65536].tolist()):
                    if u in positions and v in positions:
                        if not self.directed and positions[u] > positions[v]:
                            u, v = v, u
                        yield positions[u], positions[v], u, v, weight

        # A complete edge export needs no heap or million-element tuple copy.
        # Preserve deterministic COO order; capped displays retain the previous
        # highest-degree-position ordering and deterministic edge selection.
        shown = induced() if max_edges >= self.edge_count else heapq.nsmallest(max_edges, induced())
        nodes = []
        for node in selected:
            label = self._labels[node]
            attrs = self._node_attributes.get(label, {})
            record = {"id": node, "label": str(attrs.get("label", label)), "degree": int(degree_values[node]),
                      "group": groups[node], "identity": {"type": "integer" if isinstance(label, int) else "string", "value": str(label)}}
            if attrs:
                record["attrs"] = deepcopy(attrs)
            for field in ("x", "y", "fx", "fy", "fixed", "pinned", "longitude", "latitude"):
                if field in attrs:
                    record[field] = attrs[field]
            nodes.append(record)
        edges = []
        for _, _, u, v, weight in shown:
            record = {"source": u, "target": v, "weight": weight}
            left, right = self._labels[u], self._labels[v]
            pair = (left, right) if self.directed or _key(left) <= _key(right) else (right, left)
            attrs = self._edge_attributes.get(pair)
            if attrs:
                record["attrs"] = deepcopy(attrs)
            edges.append(record)
        return {"nodes": nodes, "edges": edges,
                "directed": self.directed, "node_count": self.node_count, "edge_count": self.edge_count,
                "shown_node_count": len(selected), "shown_edge_count": len(edges),
                "sampled": len(selected) != self.node_count or len(edges) != self.edge_count,
                "selection": "highest-degree induced subgraph", "grouping": grouping}
