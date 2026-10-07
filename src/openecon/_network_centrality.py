"""Native sparse centralities, with complete-result work and memory budgets.

Brandes' dependency recursion (2001), without stored predecessor lists:
https://snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf
Positive-cost shortest-path arcs are revisited in reverse finalized order.
Log path counts and an indexed heap keep per-source storage O(V), including
graphs with exponentially many shortest paths. Only the CSR topology is O(E).
Eigenvector centrality is a checked, shifted left power iteration in Torch.
"""
from __future__ import annotations

import hashlib
import math
import random

import torch

from openecon._network_sparse import csr
from openecon.networks import _error, _integer, _real


def _boolean(value, name):
    if not isinstance(value, bool):
        _error("invalid_option", f"{name} must be boolean.")
    return value


def _direction(value):
    if not isinstance(value, str) or value not in {"in", "out"}:
        _error("invalid_option", "direction must be 'in' or 'out'.")
    return value


def _plan(graph, sources, max_work):
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    n, a = graph.node_count, graph._arcs._nnz()
    # Reversed and uncoalesced undirected arcs require two stable sorts;
    # forward canonical directed arcs can be reused without sorting.
    setup = n + 2 * a * max(1, a.bit_length())
    per_source = (n + a) * (2 * max(1, n.bit_length()) + 8 if graph.weighted else 5)
    planned = setup + sources * per_source
    if planned > maximum:
        _error("work_budget", f"Centrality requires {planned:,} planned structural work units, "
               f"above max_work={maximum:,}; raise the explicit budget or use betweenness "
               "source sampling. No incomplete result is returned.")
    # Covers CSR construction/retention, indexed heap, scalar traversal buffers,
    # dependency/output vectors and dataframe construction before allocating.
    graph._guard(512 * n + 96 * a + 4096)
    return dict(max_work=maximum, planned_work=planned,
                work_unit="conservative structural traversal/sort units; not elapsed time")


class _Heap:
    """At most one entry per vertex, with O(log V) decrease-key."""
    def __init__(self, distances):
        self.distances, self.nodes = distances, []
        self.positions = [-1] * len(distances)

    def _less(self, left, right):
        return (self.distances[left], left) < (self.distances[right], right)

    def decrease(self, node):
        at = self.positions[node]
        if at == -2:
            _error("precision", "A finalized shortest distance was improved in float64.")
        if at == -1:
            at = len(self.nodes)
            self.nodes.append(node)
        while at:
            parent = (at - 1) // 2
            previous = self.nodes[parent]
            if not self._less(node, previous):
                break
            self.nodes[at] = previous
            self.positions[previous] = at
            at = parent
        self.nodes[at] = node
        self.positions[node] = at

    def pop(self):
        result = self.nodes[0]
        last = self.nodes.pop()
        self.positions[result] = -2
        if self.nodes:
            at = 0
            while 2 * at + 1 < len(self.nodes):
                child = 2 * at + 1
                if child + 1 < len(self.nodes) and self._less(self.nodes[child + 1], self.nodes[child]):
                    child += 1
                following = self.nodes[child]
                if not self._less(following, last):
                    break
                self.nodes[at] = following
                self.positions[following] = at
                at = child
            self.nodes[at] = last
            self.positions[last] = at
        return result


def _logadd(left, right):
    if left == -math.inf:
        return right
    maximum, minimum = max(left, right), min(left, right)
    return maximum + math.log1p(math.exp(minimum - maximum))


def _paths(topology, source, weighted, *, count_paths=False, target=None, predecessors=None):
    """Shared paths; an optional target stops at its finalized distance.

    Targeted traversal returns partial distances for other nodes and does not
    count paths. Its precision checks concern the requested destination.
    An optional caller-owned predecessor list records a deterministic route,
    avoiding a second CSR direction solely for reconstruction.
    Existing all-source traversal and path-count semantics remain unchanged.
    """
    n = topology.node_count
    destination = target
    if target is not None:
        destination = _integer(target, "target index", n - 1, zero=True)
        if count_paths:
            _error("invalid_option", "Targeted shortest paths cannot count all shortest paths.")
    if predecessors is not None:
        if destination is None or not isinstance(predecessors, list) or len(predecessors) != n:
            _error("invalid_option", "Predecessors require a target and a caller-owned list of node_count length.")
        for index in range(n):
            predecessors[index] = -1
        predecessors[source] = source
    distance = [math.inf] * n
    distance[source] = 0.
    order = []
    sigma = [-math.inf] * n if count_paths else None
    if sigma is not None:
        sigma[source] = 0.
    if not weighted:
        order.append(source)
        at = 0
        while at < len(order):
            node = order[at]
            at += 1
            if node == destination:
                break
            following = distance[node] + 1
            for target in topology.neighbors(node):
                if distance[target] == math.inf:
                    distance[target] = following
                    order.append(target)
                    if predecessors is not None:
                        predecessors[target] = node
                elif predecessors is not None and distance[target] == following and node < predecessors[target]:
                    predecessors[target] = node
                if sigma is not None and distance[target] == following:
                    sigma[target] = _logadd(sigma[target], sigma[node])
        return distance, order, sigma
    queue = _Heap(distance)
    queue.decrease(source)
    overflow = False
    ambiguous = None
    while queue.nodes:
        node = queue.pop()
        if (destination is not None and distance[destination] != math.inf
                and distance[node] >= distance[destination]):
            # All edge costs are positive. A queued node at the destination's
            # tentative distance cannot improve it. Finalize the destination
            # before unrelated equal-distance branches lose tiny positive terms.
            order.append(destination)
            break
        order.append(node)
        for target, cost in topology.row(node):
            following = distance[node] + cost
            if not math.isfinite(following):
                overflow = True
                continue
            lost_term = following == distance[node] or (distance[node] > 0 and following == cost)
            if lost_term and following <= distance[target]:
                if destination is None:
                    _error("precision", "A positive shortest-path cost is lost at float64 precision; "
                           "rescale weights or use a graph with a smaller dynamic range.")
                # The true positive sum exceeds its rounded lower bound. Do
                # not relax an uncertifiable edge or build a route through it.
                # One deferred multi-source reachability check decides whether
                # such a branch could affect the requested destination at all.
                if following < distance[destination]:
                    if ambiguous is None:
                        ambiguous = [math.inf] * n
                    ambiguous[target] = min(ambiguous[target], following)
                continue
            if following < distance[target]:
                distance[target] = following
                queue.decrease(target)
                if predecessors is not None:
                    predecessors[target] = node
                if sigma is not None:
                    sigma[target] = sigma[node]
            elif following == distance[target]:
                if predecessors is not None and node < predecessors[target]:
                    predecessors[target] = node
                if sigma is not None:
                    sigma[target] = _logadd(sigma[target], sigma[node])
    if ambiguous is not None:
        seeds = [node for node, lower_bound in enumerate(ambiguous)
                 if lower_bound < distance[destination]]
        if seeds:
            seen = [False] * n
            for node in seeds:
                seen[node] = True
            for node in seeds:
                if node == destination:
                    _error("precision", "A destination-relevant positive shortest-path cost is lost at float64 "
                           "precision; rescale weights or use a smaller dynamic range.")
                for following in topology.neighbors(node):
                    if not seen[following]:
                        seen[following] = True
                        seeds.append(following)
    if overflow and (destination is None or distance[destination] == math.inf):
        # An overflowing, nonimproving walk is harmless. Refuse only when an
        # actually reachable vertex has no representable shortest distance.
        reachable, seen = [source], [False] * n
        seen[source] = True
        for node in reachable:
            if node == destination:
                break
            for target in topology.neighbors(node):
                if not seen[target]:
                    seen[target] = True
                    reachable.append(target)
        invalid = any(distance[node] == math.inf for node in reachable) if destination is None else seen[destination]
        if invalid:
            _error("precision", "A reachable shortest distance exceeds float64 range.")
    return distance, order, sigma


def betweenness(graph, samples=None, normalized=True, endpoints=False, seed=0,
                max_work=50_000_000):
    """Brandes centrality; uniform source sampling uses unbiased n/k scaling.

    With ``samples=None`` every source is used. Sampled values are estimators,
    not exact values or confidence intervals. Self loops contribute no paths.
    Weighted paths use aggregate edge weights as costs; unweighted paths use
    unique-edge hop distances, independent of duplicate record counts.
    """
    normalized = _boolean(normalized, "normalized")
    endpoints = _boolean(endpoints, "endpoints")
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    n = graph.node_count
    count = n if samples is None else _integer(samples, "samples", n)
    plan = _plan(graph, count, max_work)
    topology = csr(graph, loops=False)
    graph._guard(topology.storage_bytes + 512 * n + 4096)
    sources = range(n) if count == n else random.Random(seed).sample(range(n), count)
    output = [0.] * n
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).encode("ascii") + b"\n")
        distance, order, sigma = _paths(topology, source, graph.weighted, count_paths=True)
        dependency = [0.] * n
        if endpoints:
            output[source] += len(order) - 1
        for node in reversed(order):
            subtotal = 0.
            for target, cost in topology.row(node):
                cost = cost if graph.weighted else 1.
                if distance[target] > distance[node] and distance[target] == distance[node] + cost:
                    subtotal += math.exp(min(0., sigma[node] - sigma[target])) * (1 + dependency[target])
            dependency[node] = subtotal
            if node != source:
                output[node] += subtotal + float(endpoints)
    scale = n / count if count else 1.
    if normalized:
        denominator = n * (n - 1) if endpoints else (n - 1) * (n - 2)
        scale = scale / denominator if (n >= 2 if endpoints else n >= 3) else 0.
    elif not graph.directed:
        scale *= .5
    output = [value * scale for value in output]
    if any(not math.isfinite(value) for value in output):
        _error("precision", "Betweenness accumulation exceeds float64 range.")
    return graph._frame({"betweenness": output}, kind="network_betweenness", exact=count == n,
                        sampled=count < n, samples=count, seed=seed, normalized=normalized,
                        endpoints=endpoints, sampling="uniform sources without replacement",
                        seed_scope="snapshot node index order",
                        source_sum_estimator_scale=n / count if count else 1.,
                        source_indices_sha256=digest.hexdigest(),
                        algorithm="Brandes dependency recursion with log path counts",
                        shortest_paths="Dijkstra" if graph.weighted else "BFS",
                        weight_semantics="aggregate edge cost" if graph.weighted else "unique-edge hops",
                        distance_ties="exact binary64 equality", device="cpu", **plan)


def _distance_centrality(graph, direction, wf_improved, max_work, metric):
    direction = _direction(direction)
    if metric == "closeness":
        wf_improved = _boolean(wf_improved, "wf_improved")
    n = graph.node_count
    plan = _plan(graph, n, max_work)
    topology = csr(graph, reverse=graph.directed and direction == "in", loops=False)
    graph._guard(topology.storage_bytes + 512 * n + 4096)
    output = []
    for source in range(n):
        distance, order, _ = _paths(topology, source, graph.weighted)
        reachable = len(order) - 1
        if not reachable:
            output.append(0.)
            continue
        if metric == "harmonic":
            try:
                value = math.fsum(1 / distance[node] for node in order if node != source)
            except OverflowError:
                _error("precision", "Harmonic centrality exceeds float64 range.")
        else:
            maximum = max(distance[node] for node in order)
            total = math.fsum(distance[node] / maximum for node in order if node != source)
            value = (reachable / total) / maximum
            if wf_improved:
                value *= reachable / (n - 1)
        if not math.isfinite(value):
            _error("precision", f"{metric.title()} centrality exceeds float64 range.")
        output.append(value)
    metadata = dict(kind="network_" + metric, exact=True, direction=direction,
                    shortest_paths="Dijkstra" if graph.weighted else "BFS", device="cpu",
                    weight_semantics="aggregate edge cost" if graph.weighted else "unique-edge hops",
                    unreachable="zero contribution", distance_ties="exact binary64 equality", **plan)
    if metric == "closeness":
        metadata["wf_improved"] = wf_improved
    return graph._frame({metric: output}, **metadata)


def closeness(graph, direction="in", wf_improved=True, max_work=50_000_000):
    """Reciprocal mean reachable distance, optionally Wasserman--Faust scaled."""
    return _distance_centrality(graph, direction, wf_improved, max_work, "closeness")


def harmonic(graph, direction="in", max_work=50_000_000):
    """Sum reciprocal incoming/outgoing distances; unreachable nodes add zero."""
    return _distance_centrality(graph, direction, False, max_work, "harmonic")


def eigenvector(graph, max_iter=1000, tol=1e-10, *, device="cpu"):
    """Unit-L2 left eigenvector of aggregate strengths, checked by residual.

    A positive uniform start selects one dominant vector on reducible graphs;
    uniqueness is not assumed. A row-strength shift removes periodic cycling.
    The normalized adjacency and shift preserve the original eigenvectors.
    """
    maximum = _integer(max_iter, "max_iter", 1_000_000)
    tol = _real(tol, "invalid_option", "tol must be positive and finite.")
    if tol <= 0:
        _error("invalid_option", "tol must be positive and finite.")
    n, a = graph.node_count, graph._arcs._nnz()
    from openecon._network_device import execution, metadata
    workspace = 192 * n + 96 * a + 4096
    with execution(graph, device, workspace) as selected:
        pairs, raw = graph._arcs._indices().to(selected), graph._arcs._values().to(selected)
        rank = torch.full((n,), 1 / math.sqrt(n) if n else 0., dtype=torch.float64)
        weight_scale = float(raw.max()) if a else 1.
        scaled = raw / weight_scale
        if a and bool((scaled <= 0).any()):
            _error("precision", "Aggregate edge strengths cannot be scaled without float64 underflow.")
        totals = torch.zeros(n, dtype=torch.float64)
        totals.index_add_(0, pairs[0], scaled)
        shift = float(totals.max()) if n and a else 1.
        eigenvalue = residual = delta = 0.
        iteration = 0

        def multiply(vector):
            result = torch.zeros(n, dtype=torch.float64)
            result.index_add_(0, pairs[1], scaled * vector[pairs[0]])
            return result

        for iteration in range(1, maximum + 1) if n and a else []:
            following = multiply(rank) + shift * rank
            norm = float(torch.linalg.vector_norm(following))
            if not math.isfinite(norm) or norm <= 0:
                _error("precision", "Eigenvector iteration has no finite positive normalization.")
            following /= norm
            delta = float(torch.linalg.vector_norm(following - rank))
            rank = following
            if delta <= tol:
                product = multiply(rank)
                eigenvalue = float(torch.dot(rank, product))
                magnitude = float(torch.linalg.vector_norm(product))
                error = float(torch.linalg.vector_norm(product - eigenvalue * rank))
                residual = error / max(magnitude, abs(eigenvalue), torch.finfo(torch.float64).tiny)
                if residual <= tol:
                    break
        else:
            if n and a:
                _error("nonconvergence", f"Eigenvector centrality did not reach tol within {maximum} "
                       f"iterations; last L2 step={delta:.6g}. No unconverged vector is returned.")
        original = eigenvalue * weight_scale
        overflow = not math.isfinite(original)
        return graph._frame({"eigenvector": rank.cpu().numpy()}, kind="network_eigenvector", exact=False,
                            converged=True, iterations=iteration, tol=tol, l2_change=delta,
                            relative_eigen_residual=residual, eigenvalue=None if overflow else original,
                            eigenvalue_overflow=overflow, eigenvalue_scaled=eigenvalue,
                            weight_scale=weight_scale, shift_scaled=shift,
                            normalization="unit Euclidean norm", orientation="left (incoming adjacency)",
                            uniqueness="not assumed; positive-start dominant vector",
                            weight_semantics="aggregate edge strength",
                            algorithm="scaled sparse shifted power iteration", **metadata(selected, workspace))
