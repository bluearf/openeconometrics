"""Native sparse modularity, Louvain and Leiden community detection.

Independent implementation of Traag--Waltman--van Eck (2019), Appendix A.1/A.2:
https://arxiv.org/html/1810.08473v3#A1.SS1 . Leiden uses singleton refinement,
well-connected merge eligibility and refined aggregation with the coarse
partition retained. It is not Louvain followed by connected-component splitting.
The objective is weighted configuration-model modularity; directed graphs use
the Leicht--Newman out/in null model (https://arxiv.org/abs/0709.4500).

Public wrappers live on Network. All arithmetic is CPU float64 Torch or scalar
stdlib arithmetic. No dense V-by-V matrix or external graph solver is used.
One stable iteration is reported, not global or asymptotic subset optimality.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import heapq
import math
import random

import pandas as pd
import torch

from openecon.networks import _error, _integer, _key, _label, _real


_THETA = .01
_BLOCK = 65536


class _Work:
    def __init__(self, maximum):
        self.maximum = _integer(maximum, "max_work", 2**63 - 1)
        self.used = 0
        self.trace_bytes = 0
        self.fixed_bytes = 0

    def add(self, count):
        count = int(count)
        if self.used + count > self.maximum:
            _error("work_budget", "Community analysis exceeds max_work; increase the explicit budget "
                   "or use a smaller graph. No incomplete partition is returned.")
        self.used += count

    def trace(self, graph, size):
        planned = self.trace_bytes + int(size)
        graph._guard(self.fixed_bytes + 768 * graph.node_count + 384 * graph.edge_count + planned)
        self.trace_bytes = planned


def _guard(graph, workspace, work):
    graph._guard(workspace + work.trace_bytes + work.fixed_bytes)


def _resolution(value):
    value = _real(value, "invalid_option", "resolution must be positive and finite.")
    if value <= 0:
        _error("invalid_option", "resolution must be positive and finite.")
    return value


def _encode(values):
    """Canonical integer IDs in first-encounter order, without sorting user IDs."""
    identifiers = {}
    encoded = []
    for value in values:
        value = _label(value)
        if value not in identifiers:
            identifiers[value] = len(identifiers)
        encoded.append(identifiers[value])
    return encoded


def _membership(graph, membership):
    n = graph.node_count
    graph._guard(384 * n + 4096)
    if isinstance(membership, pd.DataFrame):
        column = "community" if "community" in membership.columns else "block"
        if list(membership.columns).count("node") != 1 or list(membership.columns).count(column) != 1:
            _error("invalid_membership", "A membership table needs unique node and community or block columns.")
        if len(membership) != n:
            _error("invalid_membership", "Membership must cover every graph node exactly once.")
        mapping = {}
        for label, community in membership.loc[:, ["node", column]].itertuples(index=False, name=None):
            label = _label(label)
            if label not in graph._index or label in mapping:
                _error("invalid_membership", "Membership contains an unknown or duplicate node.")
            mapping[label] = community
        membership = mapping
    if isinstance(membership, Mapping):
        if len(membership) != n:
            _error("invalid_membership", "Membership must cover every graph node exactly once.")
        for label in membership:
            if _label(label) not in graph._index:
                _error("invalid_membership", "Membership contains an unknown node.")
        return _encode(membership[label] for label in graph._labels)
    if isinstance(membership, torch.Tensor):
        if (membership.ndim != 1 or membership.layout != torch.strided
                or membership.device.type == "meta" or membership.dtype == torch.bool
                or membership.dtype.is_floating_point or membership.dtype.is_complex):
            _error("invalid_membership", "Tensor membership must be a materialized one-dimensional integer vector.")
    sized = (isinstance(membership, (Sequence, pd.Series, torch.Tensor))
             or (hasattr(membership, "__len__") and hasattr(membership, "__getitem__")))
    if isinstance(membership, (str, bytes)) or not sized:
        _error("invalid_membership", "Use a node-to-community mapping, membership table or sized label sequence.")
    if len(membership) != n:
        _error("invalid_membership", "Membership must have one community label per graph node.")
    if isinstance(membership, torch.Tensor):
        membership = membership.cpu().tolist()
    return _encode(membership)


def _canonical(membership):
    return _encode(membership)


class _Level:
    """Normalized directed COO plus symmetric incident CSR for local moves.

    Undirected edges are represented in both directions; self-loop diagonal is
    twice its original weight. The diagonal is preserved in the objective, but
    excluded from move/refinement connections because it moves with the node.
    """
    def __init__(self, graph, n, pairs, values, work, live_bytes=0):
        a = len(values)
        _guard(graph, live_bytes + 192 * a + 768 * n, work)
        work.add(n + 3 * a)
        self.graph, self.n = graph, n
        self.pairs, self.values = pairs, values
        self.out = torch.zeros(n, dtype=torch.float64)
        self.incoming = torch.zeros(n, dtype=torch.float64)
        self.out.index_add_(0, pairs[0], values)
        self.incoming.index_add_(0, pairs[1], values)
        self.out = self.out.tolist()
        self.incoming = self.incoming.tolist()
        index = torch.cat([pairs, pairs.flip(0)], dim=1)
        weight = torch.cat([values, values])
        symmetric = torch.sparse_coo_tensor(index, weight, (n, n), dtype=torch.float64,
                                            device="cpu", check_invariants=True).coalesce()
        self.targets = symmetric.indices()[1]
        self.incident = symmetric.values()
        self.ptr = torch.cat([torch.zeros(1, dtype=torch.int64),
                              torch.bincount(symmetric.indices()[0], minlength=n).cumsum(0)]).tolist()
        # Tensor views retain the symmetric indices' complete 2-by-E storage.
        self.bytes = 24 * a + 24 * symmetric._nnz() + 192 * n + 4096
        self.live_bytes = live_bytes

    @classmethod
    def from_graph(cls, graph, work):
        e, n = graph.edge_count, graph.node_count
        # Original-row partition/mapping buffers remain live when coarse levels
        # have far fewer vertices. They must not vanish from later level plans.
        work.fixed_bytes = max(work.fixed_bytes, 256 * n)
        _guard(graph, 384 * e + 768 * n, work)
        work.add(e)
        pairs, raw = graph._edges.indices(), graph._edges.values()
        if not e:
            return cls(graph, n, pairs, raw, work)
        # Positive scaling avoids sum overflow. Underflow is refused explicitly:
        # silently deleting a tiny positive link can change the graph topology.
        scaled = raw / raw.max()
        if bool((scaled <= 0).any()):
            _error("precision", "The graph's weight range cannot be normalized in float64.")
        if not graph.directed:
            pairs = torch.cat([pairs, pairs.flip(0)], dim=1)
            scaled = torch.cat([scaled, scaled])
        scaled /= scaled.sum()
        if bool((scaled <= 0).any()):
            _error("precision", "The graph's normalized weights underflow float64.")
        sparse = torch.sparse_coo_tensor(pairs, scaled, (n, n), dtype=torch.float64,
                                         device="cpu", check_invariants=True).coalesce()
        return cls(graph, n, sparse.indices(), sparse.values(), work)

    def connections(self, node, membership, work, coarse=None):
        """Incident weights by community, excluding loops and optional outsiders."""
        start, stop = self.ptr[node], self.ptr[node + 1]
        work.add(1 + stop - start)
        result = {}
        for at in range(start, stop, _BLOCK):
            targets = self.targets[at:min(stop, at + _BLOCK)].tolist()
            values = self.incident[at:min(stop, at + _BLOCK)].tolist()
            for target, value in zip(targets, values):
                if target != node and (coarse is None or coarse[target] == coarse[node]):
                    group = membership[target]
                    result[group] = result.get(group, 0.) + value
        return result

    def neighbors(self, node, work):
        start, stop = self.ptr[node], self.ptr[node + 1]
        work.add(stop - start)
        for at in range(start, stop, _BLOCK):
            yield from self.targets[at:min(stop, at + _BLOCK)].tolist()

    def quality(self, membership, resolution, work):
        work.add(self.n + len(self.values))
        labels = torch.tensor(membership, dtype=torch.int64)
        size = max(membership, default=-1) + 1
        out, incoming = torch.zeros(size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64)
        out.index_add_(0, labels, torch.tensor(self.out, dtype=torch.float64))
        incoming.index_add_(0, labels, torch.tensor(self.incoming, dtype=torch.float64))
        total = 0.
        for start in range(0, len(self.values), _BLOCK):
            pairs = self.pairs[:, start:start + _BLOCK]
            total += float(self.values[start:start + _BLOCK][labels[pairs[0]] == labels[pairs[1]]].sum())
        result = total - resolution * float((out * incoming).sum())
        if not math.isfinite(result):
            _error("precision", "The modularity objective exceeds float64 range.")
        return result

    def aggregate(self, membership, work):
        membership = _canonical(membership)
        n = max(membership, default=-1) + 1
        a = len(self.values)
        _guard(self.graph, self.live_bytes + self.bytes + 192 * a + 768 * self.n, work)
        work.add(self.n + a)
        labels = torch.tensor(membership, dtype=torch.int64)
        pairs = labels[self.pairs]
        sparse = torch.sparse_coo_tensor(pairs, self.values, (n, n), dtype=torch.float64,
                                         device="cpu", check_invariants=True).coalesce()
        return _Level(self.graph, n, sparse.indices(), sparse.values(), work,
                      live_bytes=self.bytes + self.live_bytes), membership


def _group_strengths(level, membership):
    out, incoming = [0.] * level.n, [0.] * level.n
    counts = [0] * level.n
    for i, group in enumerate(membership):
        out[group] += level.out[i]
        incoming[group] += level.incoming[i]
        counts[group] += 1
    return out, incoming, counts


def _local_move(level, membership, resolution, tol, rng, work, fast):
    """Move only with positive objective gain; empty communities are candidates."""
    membership = list(membership)
    work.add(level.n)
    out, incoming, counts = _group_strengths(level, membership)
    empty = {i for i, count in enumerate(counts) if count == 0}
    free_heap = list(empty)
    heapq.heapify(free_heap)
    order = list(range(level.n))
    rng.shuffle(order)
    queue = deque(order)
    queued = [True] * level.n
    moves = 0

    def move(node):
        old = membership[node]
        connection = level.connections(node, membership, work)
        a, b = level.out[node], level.incoming[node]
        out[old] -= a
        incoming[old] -= b
        counts[old] -= 1
        if counts[old] == 0:
            out[old] = incoming[old] = 0.
            empty.add(old)
            heapq.heappush(free_heap, old)
        old_score = connection.get(old, 0.) - resolution * (a * incoming[old] + b * out[old])
        best, score = old, old_score
        candidates = sorted(connection)
        while free_heap and free_heap[0] not in empty:
            heapq.heappop(free_heap)
            work.add(1)
        if free_heap:
            candidates.append(free_heap[0])
        for candidate in candidates:
            value = connection.get(candidate, 0.) - resolution * (a * incoming[candidate] + b * out[candidate])
            if value > score + tol:
                best, score = candidate, value
        out[best] += a
        incoming[best] += b
        counts[best] += 1
        empty.discard(best)
        if len(free_heap) > 2 * max(1, level.n):
            # Lazy stale IDs cannot turn the heap into an unbounded move history.
            work.add(level.n)
            free_heap[:] = empty
            heapq.heapify(free_heap)
        membership[node] = best
        return best != old

    if fast:
        while queue:
            node = queue.popleft()
            queued[node] = False
            if move(node):
                moves += 1
                for neighbor in level.neighbors(node, work):
                    if membership[neighbor] != membership[node] and not queued[neighbor]:
                        queue.append(neighbor)
                        queued[neighbor] = True
    else:
        while True:
            changed = 0
            for node in order:
                changed += move(node)
            moves += changed
            if not changed:
                break
            rng.shuffle(order)
    return _canonical(membership), moves


def _refine(level, coarse, resolution, rng, work):
    """A.2 MergeNodesSubset: well-connected singleton and destination merges.

    For directed modularity, replace the scalar-degree null term by the symmetric
    out/in cross term. Refinement connectivity therefore means weak connectivity.
    """
    n = level.n
    work.add(n)
    membership = list(range(n))
    out, incoming = list(level.out), list(level.incoming)
    counts = [1] * n
    coarse_out, coarse_in, _ = _group_strengths(level, coarse)
    boundary = [0.] * n
    for node in range(n):
        boundary[node] = math.fsum(level.connections(node, membership, work, coarse).values())
    order = list(range(n))
    rng.shuffle(order)
    merged = 0

    def eligible(group, parent):
        expectation = resolution * (out[group] * max(0., coarse_in[parent] - incoming[group])
                                     + incoming[group] * max(0., coarse_out[parent] - out[group]))
        return boundary[group] >= expectation

    # Eligibility of individual vertices is evaluated before any refinement merge.
    allowed = [eligible(i, coarse[i]) for i in range(n)]
    for node in order:
        work.add(1)
        own, parent = membership[node], coarse[node]
        if counts[own] != 1 or not allowed[node]:
            continue
        connection = level.connections(node, membership, work, coarse)
        candidates = [(own, 0.)]
        for target in sorted(connection):
            if target == own or not eligible(target, parent):
                continue
            gain = connection[target] - resolution * (out[own] * incoming[target] + incoming[own] * out[target])
            if gain >= 0:
                candidates.append((target, gain))
        maximum = max(gain for _, gain in candidates)
        probabilities = [math.exp((gain - maximum) / _THETA) for _, gain in candidates]
        chosen = rng.random() * math.fsum(probabilities)
        target = candidates[-1][0]
        for (candidate, _), probability in zip(candidates, probabilities):
            chosen -= probability
            if chosen <= 0:
                target = candidate
                break
        if target != own:
            membership[node] = target
            boundary[target] += boundary[own] - 2 * connection[target]
            out[target] += out[own]
            incoming[target] += incoming[own]
            counts[target] += 1
            out[own] = incoming[own] = boundary[own] = 0.
            counts[own] = 0
            merged += 1
    return _canonical(membership), merged


def _one_iteration(graph, partition, method, resolution, tol, rng, work):
    level = _Level.from_graph(graph, work)
    original = torch.arange(graph.node_count, dtype=torch.int64)
    membership = list(partition)
    quality_before = level.quality(membership, resolution, work)
    history = []
    work.trace(graph, 2048)
    refinement_merges = local_moves = aggregations = refinement_attempts = 0
    while True:
        before = level.quality(membership, resolution, work)
        membership, moved = _local_move(level, membership, resolution, tol, rng, work, method == "leiden")
        after = level.quality(membership, resolution, work)
        if after + 1e-12 < before:
            _error("precision", "Local moving decreased modularity beyond floating-point tolerance.")
        local_moves += moved
        if max(membership, default=-1) + 1 == level.n:
            work.trace(graph, 64)
            history.append(after)
            result = torch.tensor(membership, dtype=torch.int64)[original].tolist()
            break
        if method == "leiden":
            refined, merges = _refine(level, membership, resolution, rng, work)
            refinement_attempts += 1
            refinement_merges += merges
            if not merges:
                # A randomized refinement can legitimately select only stays.
                # Retry with the same coarse state, under the shared work limit.
                continue
            coarse = torch.full((max(refined) + 1,), level.n, dtype=torch.int64)
            coarse.scatter_reduce_(0, torch.tensor(refined), torch.tensor(membership), reduce="amin", include_self=True)
            next_level, ids = level.aggregate(refined, work)
            membership = _canonical(coarse.tolist())
        else:
            next_level, ids = level.aggregate(membership, work)
            membership = list(range(next_level.n))
        work.trace(graph, 64)
        history.append(after)
        original = torch.tensor(ids, dtype=torch.int64)[original]
        # Previous levels are released, so retained hierarchy is O(V + E).
        level = next_level
        level.live_bytes = 0
        aggregations += 1
    if history and history[-1] + 1e-12 < quality_before:
        _error("precision", "Aggregation changed the modularity objective.")
    return _canonical(result), {"levels": len(history), "aggregations": aggregations,
                                "local_moves": local_moves, "refinement_merges": refinement_merges,
                                "refinement_attempts": refinement_attempts,
                                "level_objectives": history}


def modularity(graph, membership, resolution=1, *, max_work=50_000_000, device="cpu"):
    """Weighted modularity of a complete disjoint partition (edgeless graph: 0)."""
    resolution = _resolution(resolution)
    work = _Work(max_work)
    from openecon._network_device import execution
    n, e = graph.node_count, graph.edge_count
    arcs = e if graph.directed else 2 * e
    with execution(graph, device, 768 * n + 384 * arcs + 4096) as selected:
        work.add(graph.node_count)
        encoded = _membership(graph, membership)
        if selected.type == 'cpu':
            level = _Level.from_graph(graph, work)
            return level.quality(encoded, resolution, work)
        work.add(6 * n + 7 * arcs)
        return _modularity_reduction(graph, encoded, resolution, selected)


def _modularity_reduction(graph, encoded, resolution, selected):
    """Device-local modularity primitive; also independently tested on CPU."""
    e = graph.edge_count
    # The same normalized directed objective as _Level. Doubling each
    # undirected edge also doubles diagonal self-loops, preserving degree
    # and null-model semantics without a dense adjacency or incident CSR.
    labels = torch.tensor(encoded, dtype=torch.int64, device='cpu').to(selected)
    pairs = graph._edges.indices().to(selected)
    raw = graph._edges.values().to(selected)
    if not e:
        return 0.
    values = raw / raw.max()
    if bool((values <= 0).any()):
        _error('precision', "The graph's weight range cannot be normalized in float64.")
    if not graph.directed:
        pairs = torch.cat([pairs, pairs.flip(0)], dim=1)
        values = torch.cat([values, values])
    values /= values.sum()
    if bool((values <= 0).any()):
        _error('precision', "The graph's normalized weights underflow float64.")
    size = max(encoded, default=-1) + 1
    out, incoming = torch.zeros(size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64)
    left, right = labels[pairs[0]], labels[pairs[1]]
    out.index_add_(0, left, values)
    incoming.index_add_(0, right, values)
    result = float(values[left == right].sum() - resolution * (out * incoming).sum())
    if not math.isfinite(result):
        _error('precision', 'The modularity objective exceeds float64 range.')
    return result


def communities(graph, method="leiden", resolution=1, seed=0, max_iter=100,
                tol=1e-10, *, max_work=50_000_000):
    """Find a stable seeded heuristic modularity partition, with bounded work.

    max_iter counts complete iterations on the original graph. max_work bounds
    visited nodes/incident arcs and sparse reduction/aggregation elements over
    *all* phases and hierarchy levels. Limits raise, never return partial success.
    Leiden refinement uses theta=.01 on normalized objective gains. Stable means
    one unchanged iteration; no global optimum or asymptotic guarantee is made.
    """
    if not isinstance(method, str) or method not in {"leiden", "louvain"}:
        _error("invalid_option", "method must be 'leiden' or 'louvain'.")
    resolution = _resolution(resolution)
    seed = _integer(seed, "seed", zero=True)
    maximum = _integer(max_iter, "max_iter", 10000)
    tol = _real(tol, "invalid_option", "tol must be positive and finite.")
    if tol <= 0:
        _error("invalid_option", "tol must be positive and finite.")
    work, rng = _Work(max_work), random.Random(seed)
    graph._guard(768 * graph.node_count + 384 * graph.edge_count)
    with torch.device("cpu"), torch.no_grad():
        work.add(graph.node_count)
        partition = list(range(graph.node_count))
        objective_history = []
        levels = []
        if graph.edge_count:
            for iteration in range(1, maximum + 1):
                result, detail = _one_iteration(graph, partition, method, resolution, tol, rng, work)
                score = _Level.from_graph(graph, work).quality(result, resolution, work)
                if objective_history and score + 1e-12 < objective_history[-1]:
                    _error("precision", "An iteration decreased modularity beyond floating-point tolerance.")
                objective_history.append(score)
                levels.append(detail)
                if result == partition:
                    partition = result
                    break
                partition = result
            else:
                _error("nonconvergence", "Community detection did not reach a stable iteration within max_iter.")
        else:
            iteration = 0
            score = 0.
            work.add(graph.node_count)
        # Final public IDs follow the smallest typed original node label.
        work.add(2 * graph.node_count)
        smallest = {}
        for node, group in enumerate(partition):
            key = _key(graph._labels[node])
            if group not in smallest or key < smallest[group]:
                smallest[group] = key
        identifiers = {group: i for i, group in enumerate(sorted(smallest, key=smallest.get))}
        partition = [identifiers[group] for group in partition]
        return graph._frame({"community": partition}, kind="network_communities", method=method,
                            objective="directed configuration-model modularity" if graph.directed
                            else "undirected configuration-model modularity",
                            resolution=resolution, seed=seed, tol=tol, modularity=score, converged=True,
                            iterations=iteration, stable_iteration=True, asymptotic_optimality_claimed=False,
                            convergence_criterion="one unchanged original-graph partition; local gains exceed tol",
                            directed_connectivity="weak" if graph.directed else "undirected",
                            work_used=work.used, max_work=work.maximum, objective_history=objective_history,
                            hierarchy=levels, refinement_theta=_THETA if method == "leiden" else None)
