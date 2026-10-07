"""Exact sparse, loop-free topology for complete Network snapshots.

Positive stored edges define binary topology: duplicate input rows have
already been coalesced and zero-weight rows dropped. Weights never multiply
triangle counts or degree assortativity. Directed clustering follows binary
Fagiolo (2007), all orientations; directed cores use a weak simple projection.

SCC traversal and bin core peeling take O(V + E) after CSR creation. Triangle
enumeration uses degree-oriented forward dictionaries with O(V + E) storage.
The exact membership-probe bound is admitted before dictionaries and
intersections are allocated, rather than materializing wedges or V-by-V data.
"""
from __future__ import annotations

import math
from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.frame import as_frame
from openecon._network_sparse import csr

DEFAULT_MAX_WORK = 50_000_000
_FAGIOLO = "Fagiolo binary directed, all orientations"
_UNDIRECTED = "binary undirected"
_INT64_MAX = 2**63 - 1


def _error(code, message):
    raise AnalysisError("network_" + code, message)


def _work_limit(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        _error("invalid_option", "max_work must be a positive integer number of triangle membership probes.")
    return int(value)


def _key(label):
    return (0, label) if isinstance(label, int) else (1, label)


def _view(tensor):
    # NumPy is only a zero-copy buffer bridge. Numeric storage belongs to Torch.
    return memoryview(tensor.numpy())


def _union(left, right):
    """Merge unique sorted directed neighbor rows without allocating a set."""
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if a <= b:
            yield a
            i += 1
            if a == b:
                j += 1
        else:
            yield b
            j += 1
    while i < len(left):
        yield left[i]
        i += 1
    while j < len(right):
        yield right[j]
        j += 1


class _Topology:
    def __init__(self, graph):
        self.graph, self.n = graph, graph.node_count
        # Two live CSR directions, construction temporaries, counters, label
        # keys and linear traversal workspaces are reserved before allocating.
        graph._guard(96 * graph._arcs._nnz() + 768 * self.n)
        self.out = csr(graph, loops=False)
        self.incoming = csr(graph, reverse=True, loops=False) if graph.directed else self.out
        with torch.device("cpu"), torch.no_grad():
            offsets = torch.frombuffer(self.out.offsets, dtype=torch.int64)
            self.out_degree = offsets[1:] - offsets[:-1]
            if graph.directed:
                offsets = torch.frombuffer(self.incoming.offsets, dtype=torch.int64)
                self.in_degree = offsets[1:] - offsets[:-1]
            else:
                self.in_degree = self.out_degree
            self.reciprocal = torch.zeros(self.n, dtype=torch.int64)
            reciprocal = _view(self.reciprocal)
            if graph.directed:
                for node in range(self.n):
                    left, right = self.out.neighbors(node), self.incoming.neighbors(node)
                    i = j = count = 0
                    while i < len(left) and j < len(right):
                        a, b = left[i], right[j]
                        if a <= b:
                            i += 1
                            if a == b:
                                count += 1
                                j += 1
                        else:
                            j += 1
                    reciprocal[node] = count
                self.degree = self.out_degree + self.in_degree - self.reciprocal
            else:
                self.degree = self.out_degree
        self.degrees = _view(self.degree)
        self.out_degrees, self.in_degrees = _view(self.out_degree), _view(self.in_degree)
        self.reciprocals = reciprocal
        self.keys = [_key(label) for label in graph._labels]
        self.m = self.out.arc_count if graph.directed else self.out.arc_count // 2
        self.weak_m = sum(self.degrees) // 2
        self.loops = graph.edge_count - self.m
        tensors = {id(value): value for value in (self.degree, self.out_degree, self.in_degree, self.reciprocal)}
        self.storage_bytes = (self.out.storage_bytes
                              + (self.incoming.storage_bytes if graph.directed else 0)
                              + sum(value.numel() * value.element_size() for value in tensors.values()))

    def neighbors(self, node):
        if self.graph.directed:
            return _union(self.out.neighbors(node), self.incoming.neighbors(node))
        return iter(self.out.neighbors(node))

    def before(self, left, right):
        return (self.degrees[left] < self.degrees[right]
                or self.degrees[left] == self.degrees[right] and self.keys[left] < self.keys[right])

    def metadata(self):
        return {"weight_semantics": "unweighted positive-edge topology",
                "self_loops": "excluded", "self_loops_excluded": self.loops,
                "parallel_edges": "coalesced input edges count once",
                "topology_edges": self.m,
                "weak_projection_edges": self.weak_m,
                "clustering_convention": _FAGIOLO if self.graph.directed else _UNDIRECTED}


def _canonical_ids(topology, components):
    minima = {}
    for node, component in enumerate(components):
        key = topology.keys[node]
        if component not in minima or key < minima[component]:
            minima[component] = key
    names = {component: index for index, component in enumerate(sorted(minima, key=minima.get))}
    return [names[component] for component in components]


def _strong_ids(topology):
    """Iterative Kosaraju: no recursive call depends on path length."""
    visited, finish = bytearray(topology.n), []
    for root in range(topology.n):
        if visited[root]:
            continue
        visited[root] = 1
        stack = [(root, topology.out.offsets[root])]
        while stack:
            node, offset = stack[-1]
            if offset < topology.out.offsets[node + 1]:
                neighbor = topology.out.columns[offset]
                stack[-1] = node, offset + 1
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append((neighbor, topology.out.offsets[neighbor]))
            else:
                finish.append(node)
                stack.pop()
    components, count = [-1] * topology.n, 0
    for root in reversed(finish):
        if components[root] >= 0:
            continue
        components[root] = count
        stack = [root]
        while stack:
            for neighbor in topology.incoming.neighbors(stack.pop()):
                if components[neighbor] < 0:
                    components[neighbor] = count
                    stack.append(neighbor)
        count += 1
    return _canonical_ids(topology, components), count


def components_strong(graph):
    """Strong components including isolates, using iterative Kosaraju.

    Traversals are O(V + E); final stable component naming sorts the component
    minima by typed labels. Undirected strong connectivity equals connectivity.
    """
    topology = _Topology(graph)
    components, count = _strong_ids(topology)
    return graph._frame({"component": pd.Series(components, dtype="int64")},
                        kind="network_components", connectivity="strong",
                        component_count=count, algorithm="iterative Kosaraju",
                        component_identity="ascending minimum typed label",
                        **topology.metadata())


def _triangle_counts(topology, max_work):
    max_work = _work_limit(max_work)
    n = topology.n
    with torch.device("cpu"), torch.no_grad():
        planned = torch.zeros(n, dtype=torch.int64)
        forward_degree = _view(planned)
        for node in range(n):
            for neighbor in topology.neighbors(node):
                if topology.before(node, neighbor):
                    forward_degree[node] += 1
        work = 0
        for node in range(n):
            for neighbor in topology.neighbors(node):
                if node < neighbor:
                    work += min(forward_degree[node], forward_degree[neighbor])
        if work > max_work:
            _error("work_budget", f"Exact triangle intersections need {work:,} membership probes, "
                   f"exceeding max_work={max_work:,}; use a smaller graph or an explicit larger budget.")
        # Admit all Python forward dictionaries before constructing any of them.
        topology.graph._guard(topology.storage_bytes + 320 * n + 256 * topology.weak_m)
        forward = [{} for _ in range(n)]
        for node in range(n):
            for neighbor in topology.out.neighbors(node):
                if topology.before(node, neighbor):
                    left, right = node, neighbor
                elif topology.graph.directed:
                    left, right = neighbor, node
                else:
                    continue
                forward[left][right] = forward[left].get(right, 0) + 1
        counts = torch.zeros(n, dtype=torch.int64)
        result = _view(counts)
        for left, row in enumerate(forward):
            for right, multiplicity in row.items():
                other = forward[right]
                small, large = (row, other) if len(row) <= len(other) else (other, row)
                for third, first in small.items():
                    second = large.get(third, 0)
                    if not second:
                        continue
                    contribution = multiplicity * first * second
                    for node in (left, right, third):
                        value = result[node] + contribution
                        if value > _INT64_MAX:
                            _error("precision", "A triangle count exceeds the exact int64 result range.")
                        result[node] = value
    metadata = {"total_triangles": sum(result) // 3, "max_work": max_work,
                "work_units": "forward membership probes", "planned_work": work, "actual_work": work,
                "algorithm": "degree-oriented exact forward intersections",
                "triangle_convention": _FAGIOLO if topology.graph.directed else _UNDIRECTED,
                "reciprocal_triangle_multiplicity": "product of dyad arc counts (1 or 2)"
                                                  if topology.graph.directed else "one"}
    return counts, metadata


def triangles(graph, *, max_work=DEFAULT_MAX_WORK):
    """Exact per-node triangles; weights and loops are ignored.

    Directed Fagiolo triangles contribute the product of three dyad arc counts
    (1 or 2). A fully reciprocal triangle contributes eight to each node.
    max_work bounds the exact forward intersection membership probes.
    """
    max_work = _work_limit(max_work)
    topology = _Topology(graph)
    counts, metadata = _triangle_counts(topology, max_work)
    return graph._frame({"triangles": counts.numpy()}, kind="network_triangles",
                        **topology.metadata(), **metadata)


def _clustering_values(topology, counts):
    triangle_values = _view(counts)
    coefficients = torch.zeros(topology.n, dtype=torch.float64, device="cpu")
    result = _view(coefficients)
    denominator_sum = 0
    for node in range(topology.n):
        if topology.graph.directed:
            degree = topology.out_degrees[node] + topology.in_degrees[node]
            denominator = degree * (degree - 1) - 2 * topology.reciprocals[node]
        else:
            degree = topology.degrees[node]
            denominator = degree * (degree - 1) // 2
        denominator_sum += denominator
        result[node] = triangle_values[node] / denominator if denominator else 0.0
    pooled = sum(triangle_values) / denominator_sum if denominator_sum else 0.0
    return coefficients, pooled, denominator_sum


def clustering(graph, *, max_work=DEFAULT_MAX_WORK):
    """Binary local clustering, zero for nodes with no eligible triangle.

    Undirected C_i = 2*t_i/[k_i*(k_i-1)]. Directed Fagiolo C_i =
    t_i/[k_total*(k_total-1)-2*k_reciprocal]. Average includes isolates;
    transitivity pools actual/possible triangle orientations.
    """
    max_work = _work_limit(max_work)
    topology = _Topology(graph)
    counts, metadata = _triangle_counts(topology, max_work)
    coefficients, pooled, denominator = _clustering_values(topology, counts)
    return graph._frame({"clustering": coefficients.numpy(), "triangles": counts.numpy()},
                        kind="network_clustering", average_clustering=float(coefficients.mean()) if topology.n else 0.0,
                        transitivity=pooled, possible_triangles=denominator,
                        zero_denominator="zero", average_scope="all nodes including isolates",
                        **topology.metadata(), **metadata)


def _core_numbers(topology):
    # Batagelj-Zaversnik bin peeling: degree state changes in O(1) per arc.
    degree, n = list(topology.degrees), topology.n
    bins = [0] * (max(degree, default=0) + 1)
    for value in degree:
        bins[value] += 1
    start = 0
    for value in range(len(bins)):
        count = bins[value]
        bins[value] = start
        start += count
    position, vertices = [0] * n, [0] * n
    for node, value in enumerate(degree):
        position[node] = bins[value]
        vertices[position[node]] = node
        bins[value] += 1
    for value in range(len(bins) - 1, 0, -1):
        bins[value] = bins[value - 1]
    bins[0] = 0
    for node in vertices:
        for neighbor in topology.neighbors(node):
            if degree[neighbor] <= degree[node]:
                continue
            old_degree, old_position = degree[neighbor], position[neighbor]
            first_position = bins[old_degree]
            first = vertices[first_position]
            if first != neighbor:
                vertices[old_position], vertices[first_position] = first, neighbor
                position[first], position[neighbor] = old_position, first_position
            bins[old_degree] += 1
            degree[neighbor] -= 1
    return degree


def k_core(graph):
    """All core numbers and degeneracy; loops and weights are ignored.

    Directed graphs use the explicitly reported weak simple projection,
    counting reciprocal arcs as one neighbor. This is not an in- or out-core.
    Thresholding/induced graphs belong to the public Network wrapper.
    """
    topology = _Topology(graph)
    core = _core_numbers(topology)
    return graph._frame({"core_number": pd.Series(core, dtype="int64")},
                        kind="network_core_numbers", degeneracy=max(core, default=0),
                        algorithm="Batagelj-Zaversnik linear bin peeling",
                        core_convention="weak simple projection" if graph.directed else "simple undirected",
                        **topology.metadata())


def _density(topology):
    denominator = topology.n * (topology.n - 1)
    return topology.m * (1 if topology.graph.directed else 2) / denominator if denominator else 0.0


def _assortativity(topology):
    if not topology.m:
        return math.nan, "no loop-free edges"
    if topology.graph.directed:
        def pairs():
            for node in range(topology.n):
                for neighbor in topology.out.neighbors(node):
                    yield topology.out_degrees[node], topology.in_degrees[neighbor]
        x_mean = math.fsum(left for left, _ in pairs()) / topology.m
        y_mean = math.fsum(right for _, right in pairs()) / topology.m
        covariance = math.fsum((left - x_mean) * (right - y_mean) for left, right in pairs())
        x_variance = math.fsum((left - x_mean)**2 for left, _ in pairs())
        y_variance = math.fsum((right - y_mean)**2 for _, right in pairs())
    else:
        def pairs():
            for node in range(topology.n):
                for neighbor in topology.out.neighbors(node):
                    if node < neighbor:
                        yield topology.degrees[node], topology.degrees[neighbor]
        x_mean = y_mean = math.fsum(left + right for left, right in pairs()) / (2 * topology.m)
        covariance = math.fsum((left - x_mean) * (right - y_mean) for left, right in pairs())
        x_variance = y_variance = math.fsum((left - x_mean)**2 + (right - y_mean)**2
                                         for left, right in pairs()) / 2
    if not x_variance or not y_variance:
        return math.nan, "zero endpoint degree variance"
    return max(-1.0, min(1.0, covariance / math.sqrt(x_variance * y_variance))), None


def density(graph):
    """Loop-free binary density; zero when fewer than two nodes exist."""
    denominator = graph.node_count * (graph.node_count - 1)
    if not denominator:
        return 0.0
    # Density needs no CSR, neighbor table or vertex-sized workspace.
    graph._guard(9 * min(graph.edge_count, 65536))
    pairs, count = graph._edges.indices(), 0
    with torch.device("cpu"), torch.no_grad():
        for start in range(0, graph.edge_count, 65536):
            count += int(torch.count_nonzero(pairs[0, start:start + 65536]
                                            != pairs[1, start:start + 65536]))
    return count * (1 if graph.directed else 2) / denominator


def transitivity(graph, *, max_work=DEFAULT_MAX_WORK):
    """Pooled binary triangle ratio, with Fagiolo orientations when directed."""
    max_work = _work_limit(max_work)
    topology = _Topology(graph)
    counts, _ = _triangle_counts(topology, max_work)
    return _clustering_values(topology, counts)[1]


def assortativity(graph):
    """Endpoint degree Pearson correlation; NaN when undefined.

    Undirected endpoints are sampled symmetrically. Directed correlation is
    source out-degree against target in-degree. All degrees exclude loops.
    """
    return _assortativity(_Topology(graph))[0]


def topology_summary(graph, *, max_work=DEFAULT_MAX_WORK):
    """Exact publication-ready topology with conventions and work provenance."""
    max_work = _work_limit(max_work)
    topology = _Topology(graph)
    counts, metadata = _triangle_counts(topology, max_work)
    coefficients, pooled, denominator = _clustering_values(topology, counts)
    core = _core_numbers(topology)
    _, strong = _strong_ids(topology)
    weak = len(set(graph._components()))
    association, undefined = _assortativity(topology)
    records = [("Nodes", topology.n), ("Loop-free edges", topology.m),
               ("Self-loops excluded", topology.loops), ("Density", _density(topology)),
               ("Triangles", metadata["total_triangles"]),
               ("Average clustering", float(coefficients.mean()) if topology.n else 0.0),
               ("Transitivity", pooled), ("Degeneracy", max(core, default=0)),
               ("Weak components", weak), ("Strong components", strong),
               ("Degree assortativity", association)]
    result = as_frame(pd.DataFrame({"Metric": [name for name, _ in records],
                                   "Value": pd.Series([value for _, value in records], dtype=object)}))
    result.attrs.update(network=graph.metadata, kind="network_topology_summary",
                        possible_triangles=denominator, average_scope="all nodes including isolates",
                        core_convention="weak simple projection" if graph.directed else "simple undirected",
                        assortativity_convention="source out-degree to target in-degree"
                                                if graph.directed else "symmetric endpoint degree Pearson",
                        assortativity_defined=undefined is None, assortativity_undefined_reason=undefined,
                        **topology.metadata(), **metadata)
    return result


__all__ = ["components_strong", "triangles", "clustering", "k_core",
           "density", "transitivity", "assortativity", "topology_summary"]
