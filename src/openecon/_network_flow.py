"""Sparse flow and bipartite matching with certificates and explicit budgets.

Dinic's blocking-flow method is iterative, including the depth-first traversal:
https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=dan&paperid=35701&option_lang=eng
Hopcroft--Karp uses layered augmenting paths, also without recursive traversal:
https://epubs.siam.org/doi/10.1137/0202019

All numeric residual/queue/path storage belongs to CPU Torch. Memoryviews are
zero-copy traversal bridges only. Neither a dense adjacency matrix nor Python
objects per residual edge are constructed. No external graph solver is used.
"""
from __future__ import annotations

import math
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _label


def _view(tensor):
    return memoryview(tensor.numpy())


class _Work:
    """Actual structural traversal budget, never an elapsed-time promise."""
    def __init__(self, maximum, initial):
        self.maximum = _integer(maximum, "max_work", 2**63 - 1)
        self.used = 0
        self.add(initial)

    def add(self, amount=1):
        self.used += amount
        if self.used > self.maximum:
            _error("work_budget", "The exact network operation exceeds max_work; increase the "
                   "explicit traversal budget. No partial result is returned.")


def _endpoint(graph, raw, name):
    label = _label(raw)
    if label not in graph._index:
        _error("invalid_label", f"{name} is not a node in this graph.")
    return graph._index[label]


def _unscale(value, exponent):
    try:
        result = math.ldexp(value, exponent)
    except OverflowError:
        _error("precision", "The flow or cut capacity exceeds float64 range.")
    if not math.isfinite(result) or (value != 0 and result == 0):
        _error("precision", "The flow or cut capacity is not representable in float64.")
    return result


def _sum_add(values, corrections, node, term):
    """Compensated streaming sum with O(V) storage, including cancellation."""
    adjusted = term - corrections[node]
    following = values[node] + adjusted
    corrections[node] = (following - values[node]) - adjusted
    values[node] = following


def _tolerance(scale):
    return max(256 * sys.float_info.epsilon * scale, 8 * math.ulp(scale)) if scale else 0.


class NetworkFlowResult(dict):
    """Mapping result with a bounded publication-table summary.

    ``result['flows']`` and ``result['cut_edges']`` remain ordinary Frames for
    full exports. Display/LaTeX summary never serializes every flow or partition.
    """
    def summary(self):
        """Return eight bounded metrics; full edge/partition data stays in the mapping."""
        rows = [("Source", self["source"]), ("Target", self["target"]),
            ("Flow value", self["value"]), ("Cut capacity", self["metadata"]["cut_capacity"]),
            ("Source partition nodes", len(self["source_partition"])),
            ("Target partition nodes", len(self["target_partition"])),
            ("Cut edges", len(self["cut_edges"])),
            ("Certified", "Yes" if self["metadata"]["certified"] else "No")]
        result = as_frame(pd.DataFrame(rows, columns=["Metric", "Value"]))
        result.attrs.update(kind=self["metadata"]["kind"] + "_summary",
            algorithm=self["metadata"]["algorithm"], certified=self["metadata"]["certified"])
        return result

    def to_latex(self, buf=None, **kwargs):
        """Export the summary as a publication table, forwarding Frame export options."""
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def __repr__(self):
        return f"NetworkFlowResult(value={self['value']!r}, edges={len(self['flows'])}, " \
               f"certified={self['metadata']['certified']!r})"


def _flow(graph, source, target, max_work):
    s, t = _endpoint(graph, source, "source"), _endpoint(graph, target, "target")
    if s == t:
        _error("invalid_option", "source and target must be distinct nodes.")
    n, m = graph.node_count, graph.edge_count
    # Includes residual arrays, all simultaneous Torch construction buffers,
    # per-node traversal/certificate arrays, partitions and dataframe columns.
    # Admission happens before tensors, lists or output tables are allocated.
    work = _Work(max_work, n + 8 * m + 16)
    graph._guard(768 * n + 512 * m + 4096)
    with torch.device("cpu"), torch.no_grad():
        pairs, weights = graph._edges.indices(), graph._edges.values()
        starts, ends, capacities = _view(pairs[0]), _view(pairs[1]), _view(weights)
        # Loops cannot transmit s-t flow and must not determine numerical
        # scaling (an irrelevant huge loop may coexist with tiny capacities).
        largest = max((capacities[i] for i in range(m) if starts[i] != ends[i]), default=0.)
        exponent = math.frexp(largest)[1] if largest else 0
        heads = torch.full((n,), -1, dtype=torch.int64)
        next_edges = torch.full((2 * m,), -1, dtype=torch.int64)
        destinations = torch.empty(2 * m, dtype=torch.int64)
        residuals = torch.zeros(2 * m, dtype=torch.float64)
        scaled = torch.empty(m, dtype=torch.float64)
        level = torch.empty(n, dtype=torch.int64)
        current = torch.empty(n, dtype=torch.int64)
        queue = torch.empty(n, dtype=torch.int64)
        path_nodes = torch.empty(n, dtype=torch.int64)
        path_edges = torch.empty(n, dtype=torch.int64)
        h, following, dst, residual, cap = map(_view, (heads, next_edges, destinations, residuals, scaled))
        levels, cursor, q, nodes, edges = map(_view, (level, current, queue, path_nodes, path_edges))
        for i in range(m):
            u, v = starts[i], ends[i]
            dst[2 * i], dst[2 * i + 1] = v, u
            if u == v:
                cap[i] = 0.
                continue
            cap[i] = math.ldexp(capacities[i], -exponent)
            if (not math.isfinite(cap[i]) or cap[i] <= 0
                    or math.ldexp(cap[i], exponent) != capacities[i]):
                _error("precision", "Capacity rescaling loses a positive capacity; reduce the "
                       "capacity dynamic range before running a float64 flow analysis.")
            residual[2 * i] = cap[i]
            residual[2 * i + 1] = 0. if graph.directed else cap[i]
            following[2 * i], h[u] = h[u], 2 * i
            following[2 * i + 1], h[v] = h[v], 2 * i + 1
        value = correction = 0.
        phases = augmentations = 0
        while True:
            work.add(2 * n)
            level.fill_(-1)
            levels[s], q[0] = 0, s
            begin, finish = 0, 1
            while begin < finish:
                u = q[begin]
                begin += 1
                arc = h[u]
                while arc != -1:
                    work.add()
                    v = dst[arc]
                    if residual[arc] > 0 and levels[v] < 0:
                        levels[v], q[finish] = levels[u] + 1, v
                        finish += 1
                    arc = following[arc]
            if levels[t] < 0:
                break
            phases += 1
            work.add(n)
            current.copy_(heads)
            nodes[0], depth = s, 0
            while depth >= 0:
                u = nodes[depth]
                if u == t:
                    work.add(2 * depth)
                    delta = min(residual[edges[i]] for i in range(depth))
                    if delta <= 0 or not math.isfinite(delta):
                        _error("precision", "The augmenting flow is not positive and finite.")
                    first_saturated = depth
                    for i in range(depth):
                        arc = edges[i]
                        old, reverse = residual[arc], residual[arc ^ 1]
                        new, new_reverse = old - delta, reverse + delta
                        if (new == old or new_reverse == reverse or not math.isfinite(new_reverse)
                                or new < 0 or (new == 0 and old != delta)):
                            _error("precision", "A residual capacity update loses positive flow in "
                                   "float64; reduce the capacity dynamic range.")
                        residual[arc], residual[arc ^ 1] = new, new_reverse
                        if new == 0 and first_saturated == depth:
                            first_saturated = i
                    if first_saturated == depth:
                        _error("precision", "An augmentation did not saturate a residual edge.")
                    adjusted = delta - correction
                    updated = value + adjusted
                    correction, value = (updated - value) - adjusted, updated
                    augmentations += 1
                    depth = first_saturated
                    cursor[nodes[depth]] = following[edges[depth]]
                    continue
                arc = cursor[u]
                while arc != -1:
                    work.add()
                    v = dst[arc]
                    if residual[arc] > 0 and levels[v] == levels[u] + 1:
                        break
                    arc = following[arc]
                cursor[u] = arc
                if arc == -1:
                    levels[u] = -1
                    depth -= 1
                    if depth >= 0:
                        cursor[nodes[depth]] = following[edges[depth]]
                else:
                    edges[depth], nodes[depth + 1] = arc, dst[arc]
                    depth += 1
        # Final residual BFS is both the cut partition and an optimality
        # witness. Every original edge flow is checked for capacities and all
        # vertex balances are checked before an otherwise plausible answer is
        # returned. Compensated sums do not retain lists of incident flows.
        work.add(4 * m + 4 * n)
        balances_t = torch.zeros(n, dtype=torch.float64)
        balance_c_t = torch.zeros(n, dtype=torch.float64)
        mass_t = torch.zeros(n, dtype=torch.float64)
        mass_c_t = torch.zeros(n, dtype=torch.float64)
        flow_t = torch.zeros(m, dtype=torch.float64)
        balance, balance_c, mass, mass_c, flows = map(_view,
            (balances_t, balance_c_t, mass_t, mass_c_t, flow_t))
        cut_value = cut_correction = 0.
        cut_indices = []
        for i in range(m):
            u, v = starts[i], ends[i]
            f = 0. if u == v else cap[i] - residual[2 * i]
            if (f < (-cap[i] if not graph.directed else 0.) - _tolerance(cap[i])
                    or f > cap[i] + _tolerance(cap[i])):
                _error("precision", "The flow violates an original edge capacity.")
            flows[i] = _unscale(f, exponent)
            if u != v:
                _sum_add(balance, balance_c, u, f)
                _sum_add(balance, balance_c, v, -f)
                _sum_add(mass, mass_c, u, abs(f))
                _sum_add(mass, mass_c, v, abs(f))
                crossing = levels[u] >= 0 and levels[v] < 0 if graph.directed else (
                    (levels[u] >= 0) != (levels[v] >= 0))
                if crossing:
                    adjusted = cap[i] - cut_correction
                    updated = cut_value + adjusted
                    cut_correction, cut_value = (updated - cut_value) - adjusted, updated
                    cut_indices.append(i)
        normalized_error = 0.
        for node in range(n):
            expected = value if node == s else -value if node == t else 0.
            difference = abs(balance[node] - expected)
            allowed = _tolerance(max(mass[node], abs(expected)))
            normalized_error = max(normalized_error, difference)
            if difference > allowed:
                _error("precision", "The float64 flow failed vertex conservation; reduce "
                       "the capacity dynamic range.")
        if abs(value - cut_value) > _tolerance(max(value, cut_value)):
            _error("precision", "The float64 flow and cut certificate disagree.")
        source_nodes = tuple(graph._labels[i] for i in range(n) if levels[i] >= 0)
        target_nodes = tuple(graph._labels[i] for i in range(n) if levels[i] < 0)
        flow_frame = as_frame(pd.DataFrame({
            "source": pd.Series((graph._labels[starts[i]] for i in range(m)), dtype=object),
            "target": pd.Series((graph._labels[ends[i]] for i in range(m)), dtype=object),
            "capacity": weights.numpy(), "flow": flow_t.numpy()}))
        cut_frame = flow_frame.iloc[cut_indices].copy()
        flow_value, cut_capacity = _unscale(value, exponent), _unscale(cut_value, exponent)
        metadata = dict(kind="network_max_flow", algorithm="iterative Dinic",
            exact=True, approximate=False, capacities="aggregate edge weights",
            directed=graph.directed, undirected_flow="signed in stored source-to-target orientation",
            certificate="edge capacity, vertex conservation and max-flow/min-cut agreement",
            certified=True, cut_capacity=cut_capacity,
            maximum_conservation_error=_unscale(normalized_error, exponent),
            capacity_scale_power_of_two=exponent, phases=phases,
            augmentations=augmentations, work_used=work.used, max_work=work.maximum,
            work_unit="actual structural setup/traversal/certificate units; not elapsed time")
        flow_frame.attrs.update(metadata, network=graph.metadata)
        cut_frame.attrs.update(metadata, kind="network_min_cut", network=graph.metadata)
        return NetworkFlowResult(value=flow_value, source=graph._labels[s], target=graph._labels[t],
            source_partition=source_nodes, target_partition=target_nodes,
            flows=flow_frame, cut_edges=cut_frame, metadata=metadata)


def max_flow(graph, source, target, max_work=50_000_000):
    """Certified maximum capacity flow with a minimum-cut witness.

    Capacities always use aggregate edge weights, including duplicate unweighted
    records. Directed edges carry nonnegative flow. Undirected edges have a
    shared capacity and signed flow in the stored source/target orientation.
    Self-loops carry zero flow. This is not minimum-cost flow or multicommodity
    flow. ``exact`` means no sampling, not arbitrary-precision decimal arithmetic.
    """
    return _flow(graph, source, target, max_work)


def min_cut(graph, source, target, max_work=50_000_000):
    """Return an s-t minimum-capacity cut and its certified maximum flow."""
    result = _flow(graph, source, target, max_work)
    result["metadata"] = {**result["metadata"], "kind": "network_min_cut"}
    return result


def maximum_matching(graph, partition=None, max_work=50_000_000):
    """Maximum-cardinality bipartite matching plus a minimum-cover witness.

    Directed graphs use their weak binary projection; capacities and duplicate
    records do not change cardinality. Inferred/explicit partitions use the
    common bipartite validator. This is not weighted assignment or general
    non-bipartite matching.
    """
    from openecon._network_bipartite import _partition_ids
    from openecon._network_sparse import csr

    n, a = graph.node_count, graph._arcs._nnz()
    work = _Work(max_work, 8 * a + 4 * n + 16)
    graph._guard(1536 * n + 192 * a + 4096)
    topology = csr(graph, loops=False, undirected=graph.directed)
    sides = _partition_ids(graph, partition, _topology=topology)
    with torch.device("cpu"), torch.no_grad():
        pairs_t = torch.full((n,), -1, dtype=torch.int64)
        distances_t = torch.empty(n, dtype=torch.int64)
        cursors_t = torch.empty(n, dtype=torch.int64)
        queue_t = torch.empty(n, dtype=torch.int64)
        stack_t = torch.empty(n, dtype=torch.int64)
        selected_t = torch.empty(n, dtype=torch.int64)
        seen_t = torch.zeros(n, dtype=torch.bool)
        pair, distance, cursor, queue, stack, selected, seen = map(_view,
            (pairs_t, distances_t, cursors_t, queue_t, stack_t, selected_t, seen_t))
        offsets, columns = topology.offsets, topology.columns
        count = phases = 0
        while True:
            work.add(2 * n)
            distances_t.fill_(-1)
            first = last = 0
            for u in range(n):
                if sides[u] == 0 and pair[u] < 0:
                    distance[u], queue[last] = 0, u
                    last += 1
            shortest = n + 1
            while first < last:
                u = queue[first]
                first += 1
                if distance[u] >= shortest:
                    continue
                for at in range(offsets[u], offsets[u + 1]):
                    work.add()
                    mate = pair[columns[at]]
                    if mate < 0:
                        shortest = min(shortest, distance[u] + 1)
                    elif distance[mate] < 0 and distance[u] + 1 < shortest:
                        distance[mate], queue[last] = distance[u] + 1, mate
                        last += 1
            if shortest == n + 1:
                break
            phases += 1
            work.add(n)
            for u in range(n):
                cursor[u] = offsets[u]
            for root in range(n):
                if sides[root] != 0 or pair[root] >= 0 or distance[root] != 0:
                    continue
                stack[0], depth = root, 0
                augmented = False
                while depth >= 0:
                    u = stack[depth]
                    while cursor[u] < offsets[u + 1]:
                        work.add()
                        v = columns[cursor[u]]
                        cursor[u] += 1
                        mate = pair[v]
                        if mate < 0 and distance[u] + 1 == shortest:
                            selected[depth] = v
                            work.add(depth + 1)
                            for i in range(depth + 1):
                                left, right = stack[i], selected[i]
                                pair[left], pair[right] = right, left
                            count += 1
                            augmented = True
                            break
                        if mate >= 0 and distance[mate] == distance[u] + 1:
                            selected[depth], stack[depth + 1] = v, mate
                            depth += 1
                            break
                    else:
                        distance[u] = -1
                        depth -= 1
                        continue
                    if augmented:
                        break
        # Konig's theorem supplies an independent certificate: alternating
        # reachability from unmatched left nodes yields a cover of size |M|.
        work.add(4 * n)
        first = last = 0
        for u in range(n):
            if sides[u] == 0 and pair[u] < 0:
                seen[u], queue[last] = True, u
                last += 1
        while first < last:
            u = queue[first]
            first += 1
            for at in range(offsets[u], offsets[u + 1]):
                work.add()
                v = columns[at]
                if v == pair[u] or seen[v]:
                    continue
                seen[v] = True
                mate = pair[v]
                if mate >= 0 and not seen[mate]:
                    seen[mate], queue[last] = True, mate
                    last += 1
        cover = [i for i in range(n) if (sides[i] == 0 and not seen[i]) or (sides[i] == 1 and seen[i])]
        if len(cover) != count:
            _error("precision", "The matching and minimum-cover certificates disagree.")
        for u in range(n):
            if sides[u] == 0:
                for at in range(offsets[u], offsets[u + 1]):
                    work.add()
                    v = columns[at]
                    if seen[u] and not seen[v]:
                        _error("precision", "The matching certificate does not cover every edge.")
        matched = [i for i in range(n) if sides[i] == 0 and pair[i] >= 0]
        result = as_frame(pd.DataFrame({
            "source": pd.Series((graph._labels[i] for i in matched), dtype=object),
            "target": pd.Series((graph._labels[pair[i]] for i in matched), dtype=object)}))
        result.attrs.update(kind="network_maximum_matching", algorithm="iterative Hopcroft-Karp",
            exact=True, maximum=True, matched_pairs=count, minimum_cover_size=len(cover),
            phases=phases,
            minimum_vertex_cover=tuple(graph._labels[i] for i in cover),
            unmatched_nodes=tuple(graph._labels[i] for i in range(n) if pair[i] < 0),
            directed_projection="weak binary" if graph.directed else "binary undirected",
            certificate="matching cardinality equals a verified vertex-cover cardinality",
            certified=True, work_used=work.used, max_work=work.maximum,
            work_unit="actual structural setup/traversal/certificate units; not elapsed time",
            network=graph.metadata)
        return result
