"""Exact bounded loopless paths and deletion-based strong directed cuts.

The path search orders simple prefixes by exact stored binary64 cost. Its
worst-case frontier is exponential; admission and actual work/output guards
fail without partial results. Strong cuts use iterative sparse Kosaraju per
deletion, O((V+E)^2) work, O(V+E) workspace, without dense closure or a solver.
"""
from __future__ import annotations

from fractions import Fraction
import heapq
import math

import pandas as pd
import torch

from openecon._network_flow import _Work, _endpoint
from openecon._network_sparse import csr
from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key


def _setup(graph, maximum, *, reverse=False):
    n, a = graph.node_count, graph._arcs._nnz()
    sorting = (reverse or not graph._arcs.is_coalesced())
    work = _Work(maximum, n + a + (2 * a * max(1, a.bit_length()) if sorting else 0))
    workspace = 1024 * n + 1024 * a + 4096
    graph._guard(workspace)
    return work, workspace


def k_shortest_paths(graph, source, target, *, k=5, max_path_length=None,
                     max_output_nodes=100_000, max_frontier=100_000, max_work=50_000_000):
    """Lowest-cost distinct simple paths, with deterministic typed-ID lexicographic ties.

    max_path_length restricts the requested domain in hops, not a hidden sample.
    Unreachable pairs return an empty table; source==target has one trivial path.
    Costs use aggregate stored edges (unweighted graphs use unique-edge hops).
    Heap/work/output/memory exhaustion raises an error, never a partial answer.
    """
    requested = _integer(k, "k", 2**63 - 1)
    output_max = _integer(max_output_nodes, "max_output_nodes", 2**63 - 1)
    frontier_max = _integer(max_frontier, "max_frontier", 2**63 - 1)
    s, t = _endpoint(graph, source, "source"), _endpoint(graph, target, "target")
    n = graph.node_count
    hops = n - 1 if max_path_length is None else _integer(max_path_length, "max_path_length", max(0, n - 1), zero=True)
    work, workspace = _setup(graph, max_work)
    outgoing = csr(graph, loops=False)
    ranks = {node: rank for rank, node in enumerate(sorted(range(n), key=lambda i: _key(graph._labels[i])))}
    # Each admitted prefix includes exact rational cost, tuple references and
    # heap keys; label objects belong to the input and are not duplicated.
    def size(path):
        return 1024 + 128 * len(path)
    live, retained = size((s,)), 0
    graph._guard(workspace + live)
    heap = [(Fraction(), (ranks[s],), (s,))]
    rows, count = [], 0
    while heap and len(rows) < requested:
        work.add(max(1, len(heap).bit_length()))
        exact, tie, path = heapq.heappop(heap)
        live -= size(path)
        u = path[-1]
        if u == t:
            if count + len(path) > output_max:
                _error("output_budget", "Simple path nodes exceed max_output_nodes; no partial result returned.")
            retained += size(path) + 512
            graph._guard(workspace + live + retained)
            try:
                cost = float(exact)
            except OverflowError:
                _error("precision", "Path cost exceeds float64 range.")
            if not math.isfinite(cost):
                _error("precision", "Path cost exceeds float64 range.")
            rows.append((len(rows) + 1, tuple(graph._labels[i] for i in path), cost,
                         len(path) - 1, exact.numerator, exact.denominator))
            count += len(path)
            continue
        if len(path) - 1 >= hops:
            continue
        for v, weight in outgoing.row(u):
            work.add(len(path) + max(1, (len(heap) + 1).bit_length()))
            if v in path:
                continue
            if len(heap) >= frontier_max:
                _error("output_budget", "Simple path search exceeds max_frontier; no partial result returned.")
            graph._guard(workspace + live + retained + size(path) + 128)
            following = path + (v,)
            live += size(following)
            heapq.heappush(heap, (exact + (Fraction(weight) if graph.weighted else 1),
                                 tie + (ranks[v],), following))
    graph._guard(workspace + live + retained + 512 * len(rows) + 128 * count)
    result = as_frame(pd.DataFrame(rows, columns=["rank", "path", "cost", "hops", "cost_numerator", "cost_denominator"]))
    for column, position in [("cost_numerator", 4), ("cost_denominator", 5)]:
        result[column] = pd.Series([row[position] for row in rows], dtype=object)
    result.attrs.update(kind="network_k_shortest_paths", network=graph.metadata, exact=True,
        algorithm="best-first simple-prefix search; exponential worst-case, explicitly bounded",
        device="cpu", weight_semantics="aggregate stored binary64 costs" if graph.weighted else "unique-edge hops",
        ties="exact rational cost, then lexicographic typed IDs (integers before strings)",
        source=graph._labels[s], target=graph._labels[t], k=requested, returned=len(rows),
        max_path_length=hops, max_output_nodes=output_max, output_nodes=count,
        max_frontier=frontier_max, max_work=work.maximum, work_used=work.used,
        partial=False, sampled=False)
    return result


def _scc_count(outgoing, incoming, work, *, removed_node=-1, removed_edge=None):
    """Iterative Kosaraju without constructing a graph per deletion."""
    n = outgoing.node_count
    work.add(2 * n)
    with torch.device("cpu"), torch.no_grad():
        visited = torch.zeros(n, dtype=torch.bool)
        seen = memoryview(visited.numpy())
        order = []
        for root in range(n):
            if root == removed_node or seen[root]:
                continue
            seen[root] = True
            stack = [(root, outgoing.offsets[root])]
            while stack:
                u, position = stack[-1]
                if position >= outgoing.offsets[u + 1]:
                    order.append(u)
                    stack.pop()
                    continue
                stack[-1] = (u, position + 1)
                v = outgoing.columns[position]
                work.add()
                if v == removed_node or (u, v) == removed_edge or seen[v]:
                    continue
                seen[v] = True
                stack.append((v, outgoing.offsets[v]))
        visited.zero_()
        count = 0
        for root in reversed(order):
            if seen[root]:
                continue
            count += 1
            seen[root] = True
            stack = [root]
            while stack:
                u = stack.pop()
                for v in incoming.neighbors(u):
                    work.add()
                    if v == removed_node or (v, u) == removed_edge or seen[v]:
                        continue
                    seen[v] = True
                    stack.append(v)
        return count


def strong_cuts(graph, *, vertices, max_work):
    if not graph.directed:
        _error("invalid_option", "Strong bridges/articulation points require a directed Network.")
    work, _ = _setup(graph, max_work, reverse=True)
    outgoing, incoming = csr(graph, loops=False), csr(graph, loops=False, reverse=True)
    baseline = _scc_count(outgoing, incoming, work)
    rows = []
    if vertices:
        for u in range(graph.node_count):
            after = _scc_count(outgoing, incoming, work, removed_node=u)
            rows.append((graph._labels[u], after > baseline, after))
        columns = ["node", "strong_articulation", "components_after"]
    else:
        for u in range(graph.node_count):
            for v in outgoing.neighbors(u):
                after = _scc_count(outgoing, incoming, work, removed_edge=(u, v))
                if after > baseline:
                    rows.append((graph._labels[u], graph._labels[v], after))
        columns = ["source", "target", "components_after"]
    graph._guard(1024 * graph.node_count + 768 * graph.edge_count + 512 * len(rows) + 4096)
    result = as_frame(pd.DataFrame(rows, columns=columns))
    for name in (["node"] if vertices else ["source", "target"]):
        result[name] = pd.Series([row[columns.index(name)] for row in rows], dtype=object)
    if vertices:
        result["strong_articulation"] = result["strong_articulation"].astype(bool)
    result.attrs.update(kind="network_strong_articulation_points" if vertices else "network_strong_bridges",
        network=graph.metadata, algorithm="iterative sparse Kosaraju per deletion",
        exact=True, device="cpu", projection="none; oriented aggregate simple arcs", baseline_components=baseline,
        criterion="deletion increases the total number of strongly connected components",
        max_work=work.maximum, work_used=work.used, sampled=False, partial=False)
    return result
