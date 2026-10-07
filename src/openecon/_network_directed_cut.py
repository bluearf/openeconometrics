"""Directed global cuts by 2(V-1) exact rooted terminal-flow problems.

The reduction is described in Goemans' MIT 18.433 flow/cut lecture notes:
https://math.mit.edu/~goemans/18433S09/flowscuts.pdf (Section 4.4).
Every proper outgoing cut separates a fixed anchor from some vertex in one of
the two directions. Exact iterative Dinic solves both directions for each
vertex. A disconnected reachability set supplies a zero-capacity optimum.

Torch stores the immutable sparse input and O(V+E) traversal indices. Residual
capacities use bounded Python integer units to preserve every binary64 bit,
as in undirected Stoer--Wagner. No dense matrix or external solver is used.
"""
from __future__ import annotations

import math

import torch

from openecon._network_cut import _Work, _parts, _result, _units, _view
from openecon.networks import _error, _key


class _Residual:
    """One sparse residual topology reused across all terminal problems."""
    def __init__(self, graph, exponent, words, work):
        n, m = graph.node_count, graph.edge_count
        self.n, self.m, self.words, self.work = n, m, words, work
        # References retain the owning Torch buffers for the memoryview lifetime.
        with torch.device("cpu"), torch.no_grad():
            self.buffers = [torch.full((n,), -1, dtype=torch.int64),
                torch.full((2 * m,), -1, dtype=torch.int64),
                torch.empty(2 * m, dtype=torch.int64),
                *[torch.empty(n, dtype=torch.int64) for _ in range(5)]]
        (self.head, self.next, self.dst, self.level, self.current,
         self.queue, self.path_nodes, self.path_edges) = map(_view, self.buffers)
        pairs, weights = graph._edges.indices(), graph._edges.values()
        self.starts, self.ends = _view(pairs[0]), _view(pairs[1])
        capacities = _view(weights)
        self.base, self.residual = [0] * (2 * m), [0] * (2 * m)
        work.add((n + 3 * m) * words)
        for i in range(m):
            a, b = self.starts[i], self.ends[i]
            self.dst[2 * i], self.dst[2 * i + 1] = b, a
            if a == b:
                continue
            self.base[2 * i] = _units(capacities[i], exponent)
            self.next[2 * i], self.head[a] = self.head[a], 2 * i
            self.next[2 * i + 1], self.head[b] = self.head[b], 2 * i + 1

    def reachability(self, anchor, *, reverse=False):
        """Original-arc closure or transpose closure, with loops excluded."""
        self.work.add(2 * self.n)
        seen = [False] * self.n
        seen[anchor], self.queue[0] = True, anchor
        begin, end = 0, 1
        while begin < end:
            node = self.queue[begin]
            begin += 1
            arc = self.head[node]
            while arc != -1:
                self.work.add(self.words)
                target = self.dst[arc]
                capacity = self.base[arc ^ 1] if reverse else self.base[arc]
                if capacity and not seen[target]:
                    seen[target], self.queue[end] = True, target
                    end += 1
                arc = self.next[arc]
        return seen

    def solve(self, source, target):
        """Return exact optimal units, the minimal source side and phase count.

        Every solved terminal problem is checked for exact edge feasibility,
        vertex conservation and equality with original directed cut capacity.
        """
        work, words, n, m = self.work, self.words, self.n, self.m
        work.add(2 * m)
        self.residual[:] = self.base
        residual, level, cursor, queue = self.residual, self.level, self.current, self.queue
        nodes, edges, phases, augmentations, value = self.path_nodes, self.path_edges, 0, 0, 0
        while True:
            work.add(2 * n)
            self.buffers[3].fill_(-1)
            level[source], queue[0] = 0, source
            begin, end = 0, 1
            while begin < end:
                node = queue[begin]
                begin += 1
                arc = self.head[node]
                while arc != -1:
                    work.add(words)
                    other = self.dst[arc]
                    if residual[arc] > 0 and level[other] < 0:
                        level[other], queue[end] = level[node] + 1, other
                        end += 1
                    arc = self.next[arc]
            if level[target] < 0:
                break
            phases += 1
            work.add(n)
            self.buffers[4].copy_(self.buffers[0])
            nodes[0], depth = source, 0
            while depth >= 0:
                node = nodes[depth]
                if node == target:
                    work.add((4 * depth + 1) * words)
                    delta = min(residual[edges[i]] for i in range(depth))
                    first_saturated = depth
                    for i in range(depth):
                        arc = edges[i]
                        residual[arc] -= delta
                        residual[arc ^ 1] += delta
                        if residual[arc] == 0 and first_saturated == depth:
                            first_saturated = i
                    if delta <= 0 or first_saturated == depth:
                        _error("precision", "Exact directed flow did not saturate a positive residual arc.")
                    value += delta
                    augmentations += 1
                    depth = first_saturated
                    cursor[nodes[depth]] = self.next[edges[depth]]
                    continue
                arc = cursor[node]
                while arc != -1:
                    work.add(words)
                    other = self.dst[arc]
                    if residual[arc] > 0 and level[other] == level[node] + 1:
                        break
                    arc = self.next[arc]
                cursor[node] = arc
                if arc == -1:
                    level[node] = -1
                    depth -= 1
                    if depth >= 0:
                        cursor[nodes[depth]] = self.next[edges[depth]]
                else:
                    edges[depth], nodes[depth + 1] = arc, self.dst[arc]
                    depth += 1
        work.add((n + 4 * m) * words)
        balance, cut = [0] * n, 0
        for i in range(m):
            a, b = self.starts[i], self.ends[i]
            capacity = self.base[2 * i]
            flow = capacity - residual[2 * i]
            if flow < 0 or flow > capacity:
                _error("precision", "Exact directed flow failed original edge capacity feasibility.")
            balance[a] += flow
            balance[b] -= flow
            if level[a] >= 0 and level[b] < 0:
                cut += capacity
        if value != cut or any(b != (value if i == source else -value if i == target else 0)
                               for i, b in enumerate(balance)):
            _error("precision", "Exact directed flow failed vertex conservation or cut equality.")
        return value, [i for i in range(n) if level[i] >= 0], phases, augmentations


def directed_global_min_cut(graph, *, max_work):
    n, m = graph.node_count, graph.edge_count
    if n < 2:
        _error("invalid_option", "Global minimum cut requires at least two graph nodes.")
    work = _Work(max_work, n + 4 * m + 16)
    graph._guard(768 * n + 256 * m + 4096)
    pairs, weights = graph._edges.indices(), graph._edges.values()
    starts, ends, capacities = _view(pairs[0]), _view(pairs[1]), _view(weights)
    common, top = math.inf, -math.inf
    for i in range(m):
        if starts[i] == ends[i]:
            continue
        mantissa, power = _parts(capacities[i])
        work.add(3)
        common, top = min(common, power), max(top, power + mantissa.bit_length())
    common = 0 if common == math.inf else int(common)
    bits = 1 if top == -math.inf else int(top) - common + max(1, m.bit_length())
    words, integer_bytes = max(1, (bits + 29) // 30), 32 + 4 * max(1, (bits + 29) // 30)
    # Includes base/residual integer arrays, max-flow balance/scratch integers,
    # all traversal buffers, sorted identities, retained best side and output.
    graph._guard((1024 + 6 * integer_bytes) * n + (512 + 6 * integer_bytes) * m + 8192)
    residual = _Residual(graph, common, words, work)
    work.add(n * max(1, n.bit_length()))
    ordered = sorted(range(n), key=lambda i: _key(graph._labels[i]))
    anchor = ordered[0]
    metadata = dict(algorithm="rooted exact iterative Dinic directed global cut",
        phases=0, contractions=0, flow_problems=0, flow_problem_limit=2 * (n - 1),
        certified_flow_problems=0, anchor=graph._labels[anchor],
        capacity_scale_power_of_two=common, maximum_integer_bits=bits, integer_word_bits=30,
        certificate="all rooted terminal-flow primal/dual equalities plus exact original outgoing-capacity readback",
        direction_semantics="outgoing arcs from source_partition to target_partition only")
    forward = residual.reachability(anchor)
    if not all(forward):
        part = [i for i in range(n) if forward[i]]
        metadata.update(algorithm="directed reachability zero cut",
                        certificate="proper original-arc forward closure has zero outgoing capacity")
        del residual, forward, ordered
        return _result(graph, part, 0, common, work, metadata)
    backward = residual.reachability(anchor, reverse=True)
    if not all(backward):
        part = [i for i in range(n) if not backward[i]]
        metadata.update(algorithm="directed transpose-reachability zero cut",
                        certificate="complement of proper transpose closure has zero outgoing capacity")
        del residual, forward, backward, ordered
        return _result(graph, part, 0, common, work, metadata)
    del forward, backward
    # Every remaining flow necessarily resets all arc references, initializes
    # vertex levels and reads every edge for its primal/dual certificate.
    work.admit(2 * (n - 1) * (2 * m + 2 * n + (n + 4 * m) * words))
    best, part = None, None
    for terminal in ordered[1:]:
        for source, target in ((anchor, terminal), (terminal, anchor)):
            candidate, selected, phases, augmentations = residual.solve(source, target)
            metadata["flow_problems"] += 1
            metadata["certified_flow_problems"] += 1
            metadata["phases"] += phases
            metadata["augmentations"] = metadata.get("augmentations", 0) + augmentations
            if best is None or candidate < best:
                best, part = candidate, selected
    del residual, ordered
    return _result(graph, part, best, common, work, metadata)
