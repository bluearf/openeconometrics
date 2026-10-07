"""Exact sparse induced three-node motifs, with combinatorial open/null counts.

The Davis--Leinhardt triad nomenclature is verified against Table 1 of
Rawlings & Friedkin (2017), https://doi.org/10.1086/692757 . Sparse triad
census background: Batagelj & Mrvar (2001),
https://doi.org/10.1016/S0378-8733(01)00035-1 ; the authors' algorithm is
also discussed at https://arxiv.org/abs/1209.6308 .

This implementation counts wedges algebraically, then corrects only weak
triangles using degree-oriented forward intersections. Stars do not enumerate
their quadratic number of open triples. No dense matrix, triple list, external
graph solver or floating-point count arithmetic is used. Numeric adjacency and
degree storage belongs to CPU Torch; zero-copy memoryviews bridge traversal.
"""
from __future__ import annotations

from numbers import Integral

import pandas as pd
import torch

from openecon._network_sparse import csr
from openecon.analysis_contracts import AnalysisError
from openecon.frame import as_frame


DEFAULT_MAX_WORK = 50_000_000
DIRECTED_TRIADS = ("003", "012", "102", "021D", "021U", "021C", "111D", "111U",
                   "030T", "030C", "201", "120D", "120U", "120C", "210", "300")
UNDIRECTED_TRIADS = ("empty", "one_edge", "two_edge_path", "triangle")


def _error(code, message):
    raise AnalysisError("network_" + code, message)


def _limit(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        _error("invalid_option", "max_work must be a positive integer structural-work budget.")
    return int(value)


def _admit_work(work, maximum):
    if work > maximum:
        _error("work_budget", f"Exact triad census needs {work:,} structural work units, "
               f"exceeding max_work={maximum:,}; use a smaller graph or increase the explicit budget.")


def _choose2(n):
    return n * (n - 1) // 2


def _choose3(n):
    return n * (n - 1) * (n - 2) // 6 if n >= 3 else 0


def _view(tensor):
    return memoryview(tensor.numpy())


def _dyads(outgoing, incoming):
    """Merge sorted rows; 1=outward, 2=inward, 3=mutual relative to the node."""
    i = j = 0
    while i < len(outgoing) and j < len(incoming):
        a, b = outgoing[i], incoming[j]
        if a < b:
            yield a, 1
            i += 1
        elif b < a:
            yield b, 2
            j += 1
        else:
            yield a, 3
            i += 1
            j += 1
    while i < len(outgoing):
        yield outgoing[i], 1
        i += 1
    while j < len(incoming):
        yield incoming[j], 2
        j += 1


def _invert(code):
    return 3 if code == 3 else 3 - code


def _wedge_class(first, second):
    """Class of two dyads meeting at a center, without the closing dyad."""
    if first == second:
        return "201" if first == 3 else "021D" if first == 1 else "021U"
    if first == 3 or second == 3:
        asymmetric = second if first == 3 else first
        # 111D points into the mutual pair; 111U points out of that pair.
        return "111U" if asymmetric == 1 else "111D"
    return "021C"


def _triangle_class(ab, ac, bc):
    """Classify a complete weak triangle from three two-bit arc codes."""
    mutual = (ab == 3) + (ac == 3) + (bc == 3)
    if mutual == 3:
        return "300"
    if mutual == 2:
        return "210"
    if mutual == 1:
        if ab == 3:
            outside_out = bool(ac & 2) + bool(bc & 2)
        elif ac == 3:
            outside_out = bool(ab & 2) + bool(bc & 1)
        else:
            outside_out = bool(ab & 1) + bool(ac & 1)
        return "120D" if outside_out == 2 else "120U" if outside_out == 0 else "120C"
    out_a = bool(ab & 1) + bool(ac & 1)
    out_b = bool(ab & 2) + bool(bc & 1)
    out_c = bool(ac & 2) + bool(bc & 2)
    return "030C" if out_a == out_b == out_c == 1 else "030T"


class _Triads:
    def __init__(self, graph):
        self.graph, self.n = graph, graph.node_count
        # Include both live CSR directions, incoming sort temporaries, and the
        # weak/forward degree buffers before any new per-node allocation.
        graph._guard(192 * graph._arcs._nnz() + 256 * self.n + 8192)
        self.out = csr(graph, loops=False)
        self.incoming = csr(graph, reverse=True, loops=False) if graph.directed else self.out
        with torch.device("cpu"), torch.no_grad():
            self._degrees = torch.zeros(self.n, dtype=torch.int64)
            self._forward_degrees = torch.zeros(self.n, dtype=torch.int64)
        self.degree = _view(self._degrees)
        self.forward_degree = _view(self._forward_degrees)
        self.topology_edges = self.out.arc_count if graph.directed else self.out.arc_count // 2
        self.loops = graph.edge_count - self.topology_edges
        self.storage_bytes = (self.out.storage_bytes
            + (self.incoming.storage_bytes if graph.directed else 0) + 16 * self.n)

    def dyads(self, node):
        if self.graph.directed:
            return _dyads(self.out.neighbors(node), self.incoming.neighbors(node))
        return ((neighbor, 1) for neighbor in self.out.neighbors(node))

    def before(self, left, right):
        return (self.degree[left] < self.degree[right]
                or self.degree[left] == self.degree[right] and left < right)


def _forward_rows(topology):
    forward = [{} for _ in range(topology.n)]
    for node in range(topology.n):
        for neighbor, code in topology.dyads(node):
            if topology.before(node, neighbor):
                forward[node][neighbor] = code
    return forward


def _result(graph, counts, *, maximum, work, probes, topology_edges, weak_edges, loops, triangles):
    names = DIRECTED_TRIADS if graph.directed else UNDIRECTED_TRIADS
    # object is deliberate: all cells retain Python arbitrary-precision ints,
    # even when C(V,3) exceeds int64 or float64's exact integer range.
    result = as_frame(pd.DataFrame({"triad": names,
        "count": pd.Series([counts[name] for name in names], dtype=object)}))
    result.attrs.update(kind="network_triad_census", network=graph.metadata,
        directed=graph.directed, induced=True, exact=True,
        total_triads=_choose3(graph.node_count), count_storage="arbitrary-precision Python integers",
        weight_semantics="binary unique positive-edge topology", self_loops="excluded",
        self_loops_excluded=loops, parallel_edges="coalesced input edges count once",
        topology_edges=topology_edges, weak_projection_edges=weak_edges, weak_triangles=triangles,
        convention="Davis-Leinhardt 16 directed classes" if graph.directed else "four induced undirected classes",
        algorithm="algebraic wedges and degree-oriented weak-triangle corrections",
        max_work=maximum, planned_work=work, work_used=work,
        actual_work=work - 15 * (probes - triangles),
        membership_probes=probes,
        work_units="conservative linear/sort setup units plus 16 per forward membership probe",
        work_estimate="structural admission accounting, not measured CPU operations",
        computation_device="cpu")
    return result


def triad_census(graph, *, max_work=DEFAULT_MAX_WORK):
    """Count all induced triples exactly: 16 directed or four undirected types.

    Each unordered set of three distinct nodes counts once, including isolates.
    Positive unique edges define binary topology; weights and loops are ignored.
    Counts are arbitrary-precision integers. Sparse degree-oriented intersections
    enumerate only closed weak triangles, with algebraic counts for open, dyadic
    and empty triples. Workspace is O(V+E), never an adjacency matrix or V^3
    triple output. max_work admits conservative setup plus 16 units per exact
    forward membership probe before forward dictionaries are allocated. Dense
    triangle workloads may require an explicitly larger budget. This is CPU
    traversal and does not imply GPU execution or a streamed graph snapshot.
    """
    maximum = _limit(max_work)
    n, arcs = graph.node_count, graph._arcs._nnz()
    names = DIRECTED_TRIADS if graph.directed else UNDIRECTED_TRIADS
    counts = dict.fromkeys(names, 0)
    # Conservative CSR setup: reversed directed / mirrored undirected rows need
    # two stable sorts. The arithmetic estimate never constructs the sort input.
    setup = n + 12 * arcs + 2 * arcs * max(1, arcs.bit_length())
    _admit_work(setup, maximum)
    if n < 3 or not graph.edge_count:
        graph._guard(8192)
        pairs = graph._edges.indices()
        left, right = _view(pairs[0]), _view(pairs[1])
        loops = sum(a == b for a, b in zip(left, right))
        counts[names[0]] = _choose3(n)
        nonloops = graph.edge_count - loops
        # With <3 nodes the only possible weak dyad is counted once.
        weak = int(bool(nonloops)) if n < 3 else 0
        return _result(graph, counts, maximum=maximum, work=setup, probes=0,
                       topology_edges=nonloops, weak_edges=weak, loops=loops, triangles=0)

    topology = _Triads(graph)
    # Exact binomial degree arithmetic counts every (possibly closed) wedge
    # without allocating or iterating pairs of neighbors.
    for node in range(n):
        outward = inward = mutual = 0
        for _, code in topology.dyads(node):
            if code == 1:
                outward += 1
            elif code == 2:
                inward += 1
            else:
                mutual += 1
        topology.degree[node] = outward + inward + mutual
        if graph.directed:
            counts["021D"] += _choose2(outward)
            counts["021U"] += _choose2(inward)
            counts["021C"] += outward * inward
            counts["111U"] += outward * mutual
            counts["111D"] += inward * mutual
            counts["201"] += _choose2(mutual)
        else:
            counts["two_edge_path"] += _choose2(outward)

    weak_edges = 0
    for node in range(n):
        for neighbor, code in topology.dyads(node):
            if node >= neighbor:
                continue
            weak_edges += 1
            left = node if topology.before(node, neighbor) else neighbor
            topology.forward_degree[left] += 1
            # Count thirds adjacent to neither endpoint. Each common neighbor
            # is the missing +1, restored once for each weak triangle below.
            name = "102" if graph.directed and code == 3 else "012" if graph.directed else "one_edge"
            counts[name] += n - topology.degree[node] - topology.degree[neighbor]

    probes = 0
    for node in range(n):
        for neighbor, _ in topology.dyads(node):
            if node < neighbor:
                probes += min(topology.forward_degree[node], topology.forward_degree[neighbor])
    planned = setup + 16 * probes
    _admit_work(planned, maximum)
    # Python dict entry/key capacity plus both live CSRs and degree arrays are
    # reserved together, before constructing any forward row dictionaries.
    graph._guard(topology.storage_bytes + 256 * n + 256 * weak_edges + 8192)
    forward = _forward_rows(topology)

    triangles = 0
    for left, row in enumerate(forward):
        for right, ab in row.items():
            other = forward[right]
            small, large = (row, other) if len(row) <= len(other) else (other, row)
            for third in small:
                if third not in large:
                    continue
                triangles += 1
                if graph.directed:
                    ac, bc = row[third], other[third]
                    counts[_triangle_class(ab, ac, bc)] += 1
                    counts[_wedge_class(ab, ac)] -= 1
                    counts[_wedge_class(_invert(ab), bc)] -= 1
                    counts[_wedge_class(_invert(ac), _invert(bc))] -= 1
                    for code in (ab, ac, bc):
                        counts["102" if code == 3 else "012"] += 1
                else:
                    counts["triangle"] += 1
                    counts["two_edge_path"] -= 3
                    counts["one_edge"] += 3
    counts[names[0]] = _choose3(n) - sum(counts[name] for name in names[1:])
    if any(value < 0 for value in counts.values()):
        _error("invariant", "Exact induced triad counts cannot be negative.")
    return _result(graph, counts, maximum=maximum, work=planned, probes=probes,
                   topology_edges=topology.topology_edges, weak_edges=weak_edges,
                   loops=topology.loops, triangles=triangles)
