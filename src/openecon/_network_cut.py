"""Exact stored-capacity global cuts with sparse Stoer--Wagner contractions.

Stoer and Wagner, A Simple Min-Cut Algorithm, JACM 44(4), 1997:
https://doi.org/10.1145/263867.263872

Binary64 input capacities are converted to exact common-power-of-two integer
units. Search comparisons and contraction sums therefore have no floating
near-tie ambiguity. Only the exported value rounds to binary64. This is exact
for the immutable snapshot, not for decimal values before their float import.
The mutable contracted topology is sparse O(V+E), with compacted dictionaries
and an indexed O(V) heap. No dense matrix or external graph solver is used.
"""
from __future__ import annotations

from fractions import Fraction
import math
import sys

import pandas as pd
import torch

from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key


def _view(tensor):
    return memoryview(tensor.numpy())


class _Work:
    def __init__(self, maximum, initial):
        self.maximum = _integer(maximum, "max_work", 2**63 - 1)
        self.used = 0
        self.add(initial)

    def add(self, count=1):
        self.used += count
        if self.used > self.maximum:
            _error("work_budget", "Global minimum cut exceeds max_work; raise the explicit "
                   "structural/word-operation budget. No incomplete result is returned.")

    def admit(self, minimum):
        if self.used + minimum > self.maximum:
            _error("work_budget", "Global minimum cut's remaining exact-search lower bound exceeds "
                   "max_work; no incomplete result is returned.")


def _parts(capacity):
    """Return odd positive mantissa and its exact binary exponent."""
    numerator, denominator = capacity.as_integer_ratio()
    trailing = (numerator & -numerator).bit_length() - 1
    return numerator >> trailing, trailing - (denominator.bit_length() - 1)


def _units(capacity, exponent):
    mantissa, power = _parts(capacity)
    return mantissa << (power - exponent)


def _export_value(units, exponent):
    exact = Fraction(units << exponent, 1) if exponent >= 0 else Fraction(units, 1 << -exponent)
    try:
        value = float(exact)
    except OverflowError:
        _error("precision", "The exact global cut capacity exceeds binary64 output range.")
    if not math.isfinite(value) or (units and value == 0.):
        _error("precision", "The exact global cut capacity is not representable as a finite binary64 value.")
    return value, exact


class NetworkCutResult(dict):
    """Global cut mapping with a bounded publication-table summary.

    Partitions have no distinguished source/sink. For undirected cuts the
    smallest exact typed label is on the source side; directed cuts retain the
    minimizing outgoing orientation. Full cut rows are accessible
    as ``result['cut_edges']`` without serializing them in display summaries.
    """
    def summary(self):
        """Return eight bounded cut metrics, excluding full edge/partition lists."""
        metadata = self["metadata"]
        result = as_frame(pd.DataFrame([
            ("Minimum cut capacity", self["value"]),
            ("Source partition nodes", len(self["source_partition"])),
            ("Target partition nodes", len(self["target_partition"])),
            ("Cut edges", len(self["cut_edges"])),
            ("Flow problems" if metadata["directed"] else "Contraction phases",
             metadata["flow_problems"] if metadata["directed"] else metadata["phases"]),
            ("Optimal for stored capacities", "Yes"),
            ("Capacity arithmetic", "Exact binary integer units"),
            ("Value export", "Rounded binary64")], columns=["Metric", "Value"]))
        result.attrs.update(kind="network_global_min_cut_summary",
            algorithm=metadata["algorithm"], exact_optimum_for_stored_weights=True)
        return result

    def to_latex(self, buf=None, **kwargs):
        """Export the bounded summary table, forwarding Frame export options."""
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def __repr__(self):
        return f"NetworkCutResult(value={self['value']!r}, edges={len(self['cut_edges'])}, optimal=True)"


class _Heap:
    """Indexed maximum-adjacency heap: one entry per active supernode."""
    def __init__(self, n, ranks, work, words):
        self.nodes, self.scores, self.ranks = [], [0] * n, ranks
        self._positions = torch.full((n,), -1, dtype=torch.int64, device="cpu")
        self.positions = _view(self._positions)
        self.work, self.words = work, words

    def _higher(self, left, right):
        self.work.add(self.words)
        return (self.scores[left] > self.scores[right] or
                (self.scores[left] == self.scores[right] and self.ranks[left] < self.ranks[right]))

    def _down(self, at, node):
        while 2 * at + 1 < len(self.nodes):
            child = 2 * at + 1
            if child + 1 < len(self.nodes) and self._higher(self.nodes[child + 1], self.nodes[child]):
                child += 1
            following = self.nodes[child]
            if not self._higher(following, node):
                break
            self.nodes[at], self.positions[following] = following, at
            at = child
        self.nodes[at], self.positions[node] = node, at

    def reset(self, active):
        self.work.add(len(active))
        self.nodes[:] = active
        for at, node in enumerate(active):
            self.scores[node], self.positions[node] = 0, at
        for at in range(len(self.nodes) // 2 - 1, -1, -1):
            self._down(at, self.nodes[at])

    def increase(self, node, capacity):
        self.work.add(self.words)
        self.scores[node] += capacity
        at = self.positions[node]
        while at:
            parent = (at - 1) // 2
            previous = self.nodes[parent]
            if not self._higher(node, previous):
                break
            self.nodes[at], self.positions[previous] = previous, at
            at = parent
        self.nodes[at], self.positions[node] = node, at

    def pop(self):
        self.work.add()
        result = self.nodes[0]
        last = self.nodes.pop()
        self.positions[result] = -1
        if self.nodes:
            self._down(0, last)
        return result


class _Adjacency:
    """Sparse symmetric contractions with admission before dict resizes.

    Compaction prevents historical tombstones/large former degrees retaining
    superlinear table storage as neighbors are contracted away. Integer bytes
    are bounded from binary64 exponents plus the input-edge sum bit bound.
    """
    def __init__(self, graph, fixed, integer_bytes, work):
        self.graph, self.fixed, self.integer_bytes, self.work = graph, fixed, integer_bytes, work
        self.rows = [{} for _ in range(graph.node_count)]
        self.table_bytes = sum(map(sys.getsizeof, self.rows))
        self.arcs = 0

    def _guard(self, extra=0):
        self.graph._guard(self.fixed + self.table_bytes + self.arcs * self.integer_bytes + extra)

    def put_pair(self, left, right, capacity):
        a, b = self.rows[left], self.rows[right]
        old_a, old_b = sys.getsizeof(a), sys.getsizeof(b)
        fresh = right not in a
        # Reserve both an old table and a conservative replacement allocation
        # before Python performs a resize; the subsequent exact sizes persist.
        extra = max(1024, 4 * old_a) + max(1024, 4 * old_b) + 3 * self.integer_bytes
        self._guard(extra)
        a[right] = b[left] = capacity
        self.table_bytes += sys.getsizeof(a) + sys.getsizeof(b) - old_a - old_b
        self.arcs += 2 if fresh else 0

    def remove_pair(self, left, right):
        self.rows[left].pop(right)
        self.rows[right].pop(left)
        self.arcs -= 2

    def compact(self, node):
        row = self.rows[node]
        size = sys.getsizeof(row)
        if size > max(224, 128 * len(row)):
            self.work.add(len(row))
            self._guard(max(1024, 4 * size))
            replacement = dict(row)
            self.rows[node] = replacement
            self.table_bytes += sys.getsizeof(replacement) - size

    def contract(self, keep, remove, words):
        row = self.rows[remove]
        # The source row stays intact until iteration is over. Neighbor rows
        # lose/remove keys; no new superedge increases occupied graph size.
        for other, capacity in row.items():
            self.work.add(words)
            if other == keep:
                continue
            value = self.rows[keep].get(other, 0) + capacity
            self.put_pair(keep, other, value)
            self.rows[other].pop(remove)
            self.arcs -= 1
            self.compact(other)
        if keep in row:
            self.rows[keep].pop(remove)
            self.arcs -= 1
        old_size, old_degree = sys.getsizeof(row), len(row)
        row.clear()
        self.table_bytes += sys.getsizeof(row) - old_size
        self.arcs -= old_degree
        self.compact(keep)
        self._guard()


def _result(graph, part, capacity, exponent, work, metadata):
    n, m = graph.node_count, graph.edge_count
    words = max(1, (metadata.get("maximum_integer_bits", 30) + 29) // 30)
    work.add(n * max(1, n.bit_length()) + 4 * m * words)
    graph._guard(512 * n + 384 * m + 4096)
    selected = [False] * n
    for node in part:
        selected[node] = True
    anchor = min(range(n), key=lambda i: _key(graph._labels[i]))
    if not graph.directed and not selected[anchor]:
        selected = [not side for side in selected]
    source = tuple(sorted((graph._labels[i] for i in range(n) if selected[i]), key=_key))
    target = tuple(sorted((graph._labels[i] for i in range(n) if not selected[i]), key=_key))
    pairs, values = graph._edges.indices(), graph._edges.values()
    left, right, weights = _view(pairs[0]), _view(pairs[1]), _view(values)
    records, readback = [], 0
    for index in range(m):
        crossing = (selected[left[index]] and not selected[right[index]]) if graph.directed else (
            selected[left[index]] != selected[right[index]])
        if crossing:
            a, b = graph._labels[left[index]], graph._labels[right[index]]
            records.append((a, b, weights[index]) if graph.directed or _key(a) <= _key(b)
                           else (b, a, weights[index]))
            readback += _units(weights[index], exponent)
    if readback != capacity or not source or not target:
        _error("precision", "The exact original-edge cut readback disagrees with the search result.")
    value, exact = _export_value(capacity, exponent)
    work.add(len(records) * max(1, len(records).bit_length()))
    records.sort(key=lambda row: (_key(row[0]), _key(row[1])))
    frame = as_frame(pd.DataFrame({
        "source": pd.Series((a for a, _, _ in records), dtype=object),
        "target": pd.Series((b for _, b, _ in records), dtype=object),
        "capacity": pd.Series((w for _, _, w in records), dtype="float64")}))
    metadata.update(kind="network_global_min_cut", exact=True, sampled=False,
        exact_optimum_for_stored_weights=True, certified=True,
        certificate=metadata.get("certificate", "Stoer-Wagner phase theorem plus exact original crossing-capacity readback"),
        capacity_semantics="aggregate stored positive binary64 edge weights, including duplicate unweighted records",
        capacity_arithmetic="exact common-power-of-two Python integer units; Torch source storage",
        exact_capacity_numerator=exact.numerator, exact_capacity_denominator=exact.denominator,
        value_rounding="correctly rounded binary64 export of exact stored-capacity optimum",
        partition_orientation="minimizing outgoing side; do not swap partitions" if graph.directed else
            "source_partition contains smallest exact typed label; no distinguished source or sink",
        cut_edge_order="original arc orientation and typed-label lexicographic order" if graph.directed else
            "typed-label canonical orientation and lexicographic order",
        self_loops="ignored", directed=graph.directed, device="cpu", work_used=work.used,
        max_work=work.maximum, work_unit="structural traversal and integer-word comparison/addition charges; not elapsed time")
    frame.attrs.update(metadata, network=graph.metadata)
    return NetworkCutResult(value=value, source_partition=source, target_partition=target,
        cut_edges=frame, metadata=metadata)


def global_min_cut(graph, *, max_work=50_000_000):
    """Exact global minimum capacity cut; directed cuts count only outgoing arcs.

    Disconnected graphs have a deterministic zero cut, including isolates.
    Positive stored binary64 capacities are summed/compared in exact integer
    units, avoiding numerical near-ties and overflow in intermediate degrees.
    Final output rounds to float64; an out-of-range optimum raises precision.
    Undirected graphs use sparse Stoer--Wagner contractions; directed graphs use
    exact rooted terminal flows or a reachability zero cut. Superlinear runtime
    is guarded and never replaced by an approximate cut.
    """
    if graph.directed:
        from openecon._network_directed_cut import directed_global_min_cut
        return directed_global_min_cut(graph, max_work=max_work)
    n, m = graph.node_count, graph.edge_count
    if n < 2:
        _error("invalid_option", "Global minimum cut requires at least two graph nodes.")
    work = _Work(max_work, n + 4 * m + 16)
    graph._guard(768 * n + 256 * m + 4096)
    pairs, weights = graph._edges.indices(), graph._edges.values()
    left, right, capacities = _view(pairs[0]), _view(pairs[1]), _view(weights)
    with torch.device("cpu"), torch.no_grad():
        parent_t, size_t = torch.arange(n, dtype=torch.int64), torch.ones(n, dtype=torch.int64)
    parent, sizes = _view(parent_t), _view(size_t)
    components = n

    def find(node, parent=parent):
        while parent[node] != node:
            work.add()
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index in range(m):
        work.add()
        a, b = find(left[index]), find(right[index])
        if a == b:
            continue
        if sizes[a] < sizes[b]:
            a, b = b, a
        parent[b], sizes[a] = a, sizes[a] + sizes[b]
        components -= 1
    if components > 1:
        anchor = min(range(n), key=lambda i: _key(graph._labels[i]))
        component = find(anchor)
        part = [i for i in range(n) if find(i) == component]
        del find, parent, sizes, parent_t, size_t
        return _result(graph, part, 0, 0, work,
            dict(algorithm="connected-component zero cut", phases=0, contractions=0,
                 component_count=components, capacity_scale_power_of_two=0, integer_word_bits=30))
    del find, parent, sizes, parent_t, size_t
    # Determine the minimal exact exponent and resulting integer bit bound
    # without retaining a second edge-value vector. Loops cannot affect a cut.
    common, top = math.inf, -math.inf
    for index in range(m):
        if left[index] == right[index]:
            continue
        mantissa, exponent = _parts(capacities[index])
        work.add(max(1, mantissa.bit_length() // 30 + 1))
        common, top = min(common, exponent), max(top, exponent + mantissa.bit_length())
    common = int(common)
    bound_bits = int(top) - common + max(1, m.bit_length())
    words = max(1, (bound_bits + 29) // 30)
    integer_bytes = 32 + 4 * words
    # In a connected phase, each vertex except the first needs a score update,
    # and heap construction adds at least one comparison per internal node.
    # This lower bound admits disconnected million-node zero cuts while
    # refusing an impossible connected request before adjacency dictionaries.
    work.admit((n * (n + 1) // 2 - 1) * words)
    fixed = (1024 + 6 * integer_bytes) * n + 4096
    graph._guard(fixed + (512 + 3 * integer_bytes) * m + 4096)
    node_ids = list(range(n))
    adjacency = _Adjacency(graph, fixed, integer_bytes, work)
    for index in range(m):
        if left[index] != right[index]:
            work.add(words)
            adjacency.put_pair(node_ids[left[index]], node_ids[right[index]],
                               _units(capacities[index], common))
    ordered = sorted(node_ids, key=lambda i: _key(graph._labels[i]))
    ranks = [0] * n
    for rank, node in enumerate(ordered):
        ranks[node] = rank
    work.add(n * max(1, n.bit_length()))
    active, members, added = node_ids.copy(), [[node] for node in node_ids], [False] * n
    heap = _Heap(n, ranks, work, words)
    best, best_part, phases = None, None, 0
    while len(active) > 1:
        heap.reset(active)
        for node in active:
            added[node] = False
        previous = None
        while heap.nodes:
            node = heap.pop()
            added[node] = True
            if not heap.nodes:
                candidate = heap.scores[node]
                if best is None or candidate < best:
                    work.add(len(members[node]))
                    best, best_part = candidate, members[node].copy()
                work.add(len(members[node]) + len(active))
                adjacency.contract(previous, node, words)
                members[previous].extend(members[node])
                members[node].clear()
                active.remove(node)
                phases += 1
                break
            for other, capacity in adjacency.rows[node].items():
                work.add()
                if not added[other]:
                    heap.increase(other, capacity)
            previous = node
    del adjacency, heap, members, added, active, node_ids, ranks, ordered
    return _result(graph, best_part, best, common, work,
        dict(algorithm="Stoer-Wagner sparse indexed maximum-adjacency", phases=phases,
             contractions=phases, component_count=1, capacity_scale_power_of_two=common,
             maximum_integer_bits=bound_bits, integer_word_bits=30))
