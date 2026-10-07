"""Exact CPU Torch graph kernels over bounded disk edge scans.

Edges stay on disk. O(V) numerical state and complete labelled outputs must be
admitted before tensor allocation. Logical scan bytes are not physical OS I/O.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import pandas as pd
import torch

from openecon.dataset import _file_identity
from openecon.frame import as_frame
from openecon.networks import _Budget, _error, _integer, _key, _real, _weight
from openecon._network_io import _attribute_bytes
from openecon._network_store import _decode, _identity


class _Work:
    def __init__(self, limit):
        self.limit = _integer(limit, "max_work", 2**63 - 1)
        self.used = 0

    def add(self, count):
        if self.used + count > self.limit:
            _error("work_budget", "Disk graph analysis exceeds max_work; no partial result is returned.")
        self.used += count


class _Scan:
    def __init__(self, graph, batch_rows, max_work, max_scan_bytes, *, passes, vector_passes=1):
        requested = _integer(batch_rows, "batch_rows", 1_000_000)
        self.graph = graph
        self.work = _Work(max_work)
        self.scan_limit = _integer(max_scan_bytes, "max_scan_bytes", 2**63 - 1)
        self.passes = self.peak_rows = self.bytes = 0
        self.fixed = min(1024 * 1024, graph._limit // 4) + 262144
        self.base = (self.fixed + graph._metadata["label_storage_bound"] + 1536 * graph.node_count
                     + 2 * _attribute_bytes(graph._metadata["graph_attributes"]))
        budget = _Budget(graph._limit / 1024**2)
        budget.check(self.base + 1024)
        self.rows = min(requested, 65536, max(1, (graph._limit - self.base) // 512))
        self.estimated_peak = self.base + 512 * self.rows
        budget.check(self.estimated_peak)
        if 24 * graph.edge_count * passes > self.scan_limit:
            _error("work_budget", "Required disk edge scans exceed max_scan_bytes; no result is returned.")
        if 16 * vector_passes * graph.node_count + 4 * graph.edge_count * passes > self.work.limit:
            _error("work_budget", "Required disk graph setup exceeds max_work; no result is returned.")
        self.work.add(16 * graph.node_count)

    def edges(self):
        cost = 24 * self.graph.edge_count
        if self.bytes + cost > self.scan_limit:
            _error("work_budget", "Disk edge scans exceed max_scan_bytes; no result is returned.")
        self.work.add(4 * self.graph.edge_count)
        self.bytes += cost
        self.passes += 1
        connection = self.graph._reader()
        try:
            cursor = connection.execute("SELECT src,dst,w FROM edges ORDER BY src,dst")
            while block := cursor.fetchmany(self.rows):
                self.peak_rows = max(self.peak_rows, len(block))
                yield block
            if _file_identity(self.graph.path) != self.graph._identity:
                _error("store_changed", "The disk graph changed during analysis.")
        finally:
            connection.close()

    def tensors(self, *, arcs=False):
        for block in self.edges():
            i = torch.tensor([row[0] for row in block], dtype=torch.int64, device="cpu")
            j = torch.tensor([row[1] for row in block], dtype=torch.int64, device="cpu")
            w = torch.tensor([row[2] for row in block], dtype=torch.float64, device="cpu")
            yield i, j, w
            if arcs and not self.graph.directed:
                off = i != j
                if bool(off.any()):
                    yield j[off], i[off], w[off]

    def labels(self):
        connection = self.graph._reader()
        labels = []
        try:
            cursor = connection.execute("SELECT token FROM nodes ORDER BY id")
            while block := cursor.fetchmany(self.rows):
                labels.extend(_decode(row[0]) for row in block)
            if _file_identity(self.graph.path) != self.graph._identity:
                _error("store_changed", "The disk graph changed during analysis.")
        finally:
            connection.close()
        return labels

    def frame(self, values, *, labels=None, **metadata):
        labels = self.labels() if labels is None else labels
        frame = as_frame(pd.DataFrame({"node": pd.Series(labels, dtype=object), **values}))
        frame.attrs.update(network=self.graph.metadata, device="cpu", dtype="float64/int64",
            exact=True, sampled=False, storage="disk", graph_materialized=False,
            resident_state_scope="O(V) Torch state plus complete labelled output, admitted before tensors",
            estimated_owned_peak_bytes=self.estimated_peak, max_memory_bytes=self.graph._limit,
            scan_passes=self.passes, edge_rows_read=self.passes * self.graph.edge_count,
            logical_edge_bytes_read=self.bytes, max_scan_bytes=self.scan_limit,
            scan_byte_scope="24 bytes per stored numeric edge per full pass; excludes B-tree/page overhead and cache effects",
            work_used=self.work.used, max_work=self.work.limit, actual_peak_batch_rows=self.peak_rows,
            **metadata)
        return frame


def degree(graph, *, batch_rows=65536, max_work=50_000_000, max_scan_bytes=1024**3):
    scan = _Scan(graph, batch_rows, max_work, max_scan_bytes, passes=1)
    with torch.device("cpu"), torch.no_grad():
        n = graph.node_count
        out, incoming = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)
        out_w, in_w = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
        for i, j, w in scan.tensors():
            out.index_add_(0, i, torch.ones_like(i))
            incoming.index_add_(0, j, torch.ones_like(j))
            out_w.index_add_(0, i, w)
            in_w.index_add_(0, j, w)
        if not bool(torch.isfinite(out_w).all() & torch.isfinite(in_w).all()):
            _error("precision", "A node strength exceeds float64 range.")
        if graph.directed:
            values = dict(in_degree=incoming.numpy(), out_degree=out.numpy(),
                          in_strength=in_w.numpy(), out_strength=out_w.numpy())
        else:
            strengths = out_w + in_w
            if not bool(torch.isfinite(strengths).all()):
                _error("precision", "A node strength exceeds float64 range.")
            values = dict(degree=(out + incoming).numpy(), strength=strengths.numpy())
        return scan.frame(values, kind="network_degree", algorithm="bounded disk edge Torch reductions")


def pagerank(graph, damping=.85, tol=1e-10, max_iter=200, personalization=None, device="cpu", *,
             batch_rows=65536, max_work=50_000_000, max_scan_bytes=1024**3):
    damping = _real(damping, "invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
    tol = _real(tol, "invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
    if not 0 <= damping < 1 or tol <= 0:
        _error("invalid_option", "Use 0 <= damping < 1 and a positive finite tol.")
    max_iter = _integer(max_iter, "max_iter", 1_000_000)
    if device != "cpu":
        _error("device", "Disk-scanned PageRank currently supports CPU float64 only; no device fallback is used.")
    n = graph.node_count
    scan = _Scan(graph, batch_rows, max_work, max_scan_bytes, passes=3 if n and damping else 0,
                 vector_passes=2 if n and damping else 1)
    with torch.device("cpu"), torch.no_grad():
        probability = torch.zeros(n, dtype=torch.float64)
        if personalization is None:
            if n:
                probability.fill_(1 / n)
        else:
            if not isinstance(personalization, Mapping):
                _error("invalid_option", "personalization must map known node labels to nonnegative weights.")
            connection = graph._reader()
            try:
                for label, value in personalization.items():
                    token, _ = _identity(label)
                    found = connection.execute("SELECT id FROM nodes WHERE token=?", (token,)).fetchone()
                    if found is None:
                        _error("unknown_node", "Personalization names a node outside the graph.")
                    probability[found[0]] = _weight(value)
            finally:
                connection.close()
            if not n or float(probability.max()) == 0:
                _error("invalid_option", "Personalization must have positive total weight.")
            probability /= probability.max()
            probability /= probability.sum()
        common = dict(converged=True, tol=tol, damping=damping,
                      algorithm="weighted power iteration over bounded disk edge scans",
                      dangling_distribution="personalization")
        if not n or damping == 0:
            return scan.frame(dict(pagerank=probability.numpy()), iterations=0 if not n else 1,
                l1_change=0., error_bound_estimate=0., probability_sum=float(probability.sum()), **common)
        maximum = torch.zeros(n, dtype=torch.float64)
        for i, _, w in scan.tensors(arcs=True):
            maximum.scatter_reduce_(0, i, w, reduce="amax", include_self=True)
        totals = torch.zeros(n, dtype=torch.float64)
        for i, _, w in scan.tensors(arcs=True):
            totals.index_add_(0, i, w / maximum[i])
        dangling = totals == 0
        rank = probability.clone()
        delta = bound = math.inf
        for iteration in range(1, max_iter + 1):
            scan.work.add(16 * n)
            following = ((1 - damping) + damping * rank[dangling].sum()) * probability
            for i, j, w in scan.tensors(arcs=True):
                transition = (w / maximum[i]) / totals[i]
                following.index_add_(0, j, damping * rank[i] * transition)
            following /= following.sum()
            delta = float((following - rank).abs().sum())
            bound = damping / (1 - damping) * delta
            rank = following
            if bound <= tol:
                break
        else:
            _error("nonconvergence", f"PageRank did not reach tol within {max_iter} iterations; "
                   f"last L1 step={delta:.6g}, contraction error estimate={bound:.6g}.")
        return scan.frame(dict(pagerank=rank.numpy()), iterations=iteration, l1_change=delta,
                          error_bound_estimate=bound, probability_sum=float(rank.sum()), **common)


def components(graph, connectivity="weak", *, batch_rows=65536, max_work=50_000_000,
               max_scan_bytes=1024**3):
    if connectivity == "strong":
        _error("capacity", "Disk-native strong components are unavailable; explicitly materialize an admitted small graph.")
    if connectivity != "weak":
        _error("invalid_option", "connectivity must be 'weak' or 'strong'.")
    scan = _Scan(graph, batch_rows, max_work, max_scan_bytes, passes=1)
    with torch.device("cpu"), torch.no_grad():
        n = graph.node_count
        parent_t, size_t = torch.arange(n, dtype=torch.int64), torch.ones(n, dtype=torch.int64)
        parent, size = memoryview(parent_t.numpy()), memoryview(size_t.numpy())

        def find(node):
            while parent[node] != node:
                scan.work.add(1)
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for block in scan.edges():
            for i, j, _ in block:
                a, b = find(i), find(j)
                if a != b:
                    if size[a] < size[b]:
                        a, b = b, a
                    parent[b], size[a] = a, size[a] + size[b]
        labels = scan.labels()
        smallest = {}
        for node in range(n):
            root, key = find(node), _key(labels[node])
            parent[node] = root
            if root not in smallest or key < smallest[root]:
                smallest[root] = key
        groups = {root: index for index, root in enumerate(sorted(smallest, key=smallest.get))}
        result = torch.tensor([groups[parent[node]] for node in range(n)], dtype=torch.int64)
        return scan.frame(dict(component=result.numpy() if n else []), labels=labels,
            kind="network_components", connectivity="weak", algorithm="Torch int64 union-find over disk edges")
