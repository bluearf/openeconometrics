"""Bounded sparse distances, graph cuts and minimum spanning forests.

Shortest paths share the indexed Dijkstra/BFS implementation with centrality.
Pair output is a long table with an explicit cardinality budget; no dense V²
matrix is allocated. Articulation points and bridges use iterative Tarjan on
the simple weak projection. Kruskal uses Torch sorting and O(V) union-find.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import math

import pandas as pd
import torch

from openecon._network_centrality import _direction, _paths
from openecon._network_sparse import csr
from openecon.frame import as_frame
from openecon.networks import Network, _Budget, _error, _integer, _key, _label


def _plan(graph, sources, max_work, *, pairs=0, reverse=False, targeted=False):
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    n, a = graph.node_count, graph._arcs._nnz()
    # Canonical directed outgoing COO needs no sort. Reversed/undirected
    # traversal needs two stable lexicographic sorts in the shared CSR builder.
    needs_sort = reverse or not graph._arcs.is_coalesced()
    setup = n + a + (2 * a * max(1, a.bit_length()) if needs_sort else 0)
    per_source = (n + a) * (2 * max(1, n.bit_length()) + 8 if graph.weighted else 5)
    # Targeted float64 paths may additionally need one union reachability pass
    # from deferred ambiguous endpoints and one overflow-reachability pass.
    planned = setup + sources * per_source + pairs + (4 * (n + a) if targeted else 0)
    if planned > maximum:
        _error("work_budget", f"Distances require {planned:,} planned structural work units, "
               f"above max_work={maximum:,}. Select fewer sources or raise the explicit budget; "
               "no incomplete result is returned.")
    # CSR retention/construction, traversal, ambiguous-bound/union-BFS buffers,
    # predecessor/route lists and dataframe output. Reserve
    # references to original immutable labels rather than encoding label copies.
    # In targeted mode ambiguous[n], union-BFS seeds/seen and a possible
    # overflow-reachability queue coexist with the route/predecessor buffers.
    graph._guard((832 if targeted else 640) * n + 96 * a + 256 * pairs + 4096)
    return {"max_work": maximum, "planned_work": planned,
            "work_unit": "conservative structural traversal/sort/output units; not elapsed time"}


def _metadata(graph, kind, **options):
    return dict(kind=kind, network=graph.metadata, exact=True, device="cpu",
                shortest_paths="Dijkstra" if graph.weighted else "BFS",
                weight_semantics="aggregate edge cost" if graph.weighted else "unique-edge hops",
                distance_ties="exact binary64 equality", **options)


def _node(graph, value, name):
    label = _label(value)
    if label not in graph._index:
        _error("unknown_node", f"The {name} is outside the graph.")
    return graph._index[label]


def _selection(graph, values, name):
    if values is None:
        return range(graph.node_count)
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        _error("invalid_option", f"{name} must be an iterable of exact node IDs, or None.")
    output, seen = [], set()
    for value in values:
        index = _node(graph, value, f"{name} selection")
        if index in seen:
            _error("invalid_option", f"{name} cannot contain repeated node IDs.")
        graph._guard(256 * (len(output) + 1))
        output.append(index)
        seen.add(index)
    return output


def shortest_path(graph, source, target, *, max_work=50_000_000):
    """One deterministic route; an unreachable target returns an empty table.

    Tied routes choose the lowest snapshot-index predecessor at each step.
    Route rows contain their cumulative source distance, rather than edge cost.
    """
    source_index, target_index = _node(graph, source, "source"), _node(graph, target, "target")
    plan = _plan(graph, 1, max_work, pairs=graph.node_count, targeted=True)
    outgoing = csr(graph, loops=False)
    predecessors = [-1] * graph.node_count
    distance, _, _ = _paths(outgoing, source_index, graph.weighted, target=target_index,
                           predecessors=predecessors)
    reachable = math.isfinite(distance[target_index])
    route = []
    if reachable:
        node = target_index
        route.append(node)
        while node != source_index:
            previous = predecessors[node]
            if previous < 0 or distance[previous] >= distance[node] or len(route) >= graph.node_count:
                _error("precision", "A finite shortest distance has no representable route.")
            route.append(previous)
            node = previous
        route.reverse()
    result = as_frame(pd.DataFrame({"step": pd.Series(range(len(route)), dtype="int64"),
                                   "node": pd.Series([graph._labels[i] for i in route], dtype=object),
                                   "distance": pd.Series([distance[i] for i in route], dtype="float64")}))
    result.attrs.update(_metadata(graph, "network_shortest_path", source=graph._labels[source_index],
                                 target=graph._labels[target_index], reachable=reachable,
                                 distance=distance[target_index], tie_break="lowest snapshot-index predecessor",
                                 disconnected="infinity; empty route", **plan))
    return result


def distances(graph, sources=None, targets=None, *, direction="out",
              max_work=50_000_000, max_pairs=1_000_000):
    """Selected ordered source/target distances including diagonal zero entries.

    ``direction='in'`` traverses reversed arcs in a directed graph. All node
    identities are exact. Infinity denotes an unreachable pair. No sample is
    substituted when the explicit output or traversal budget is exceeded.
    """
    direction = _direction(direction)
    maximum_pairs = _integer(max_pairs, "max_pairs", 2**63 - 1, zero=True)
    # Both explicit selections and their duplicate-detection sets can coexist
    # before the cardinality is known. Admit that peak before reading iterables.
    graph._guard(512 * graph.node_count + 4096)
    source_indices = _selection(graph, sources, "sources")
    target_indices = _selection(graph, targets, "targets")
    pairs = len(source_indices) * len(target_indices)
    if pairs > maximum_pairs:
        _error("pair_budget", f"Distance output requires {pairs:,} pairs, above "
               f"max_pairs={maximum_pairs:,}; select sources/targets explicitly.")
    plan = _plan(graph, len(source_indices) if pairs else 0, max_work, pairs=pairs,
                 reverse=graph.directed and direction == "in")
    left, right, values = [], [], []
    if pairs:
        topology = csr(graph, reverse=graph.directed and direction == "in", loops=False)
        for source in source_indices:
            distance, _, _ = _paths(topology, source, graph.weighted)
            for target in target_indices:
                left.append(graph._labels[source])
                right.append(graph._labels[target])
                values.append(distance[target])
    result = as_frame(pd.DataFrame({"source": pd.Series(left, dtype=object),
                                   "target": pd.Series(right, dtype=object),
                                   "distance": pd.Series(values, dtype="float64")}))
    result.attrs.update(_metadata(graph, "network_distances", direction=direction,
                                 source_count=len(source_indices), target_count=len(target_indices),
                                 pair_count=pairs, max_pairs=maximum_pairs,
                                 disconnected="infinity", **plan))
    return result


def _disconnected(value):
    if not isinstance(value, str) or value not in {"infinite", "reachable"}:
        _error("invalid_option", "disconnected must be 'infinite' or 'reachable'.")
    return value


def _eccentricity(distance, order, n, disconnected):
    return math.inf if disconnected == "infinite" and len(order) < n else max(
        (distance[i] for i in order), default=0.)


def eccentricity(graph, *, direction="out", disconnected="infinite", max_work=50_000_000):
    """Maximum distance from each node, with explicit disconnected convention.

    ``infinite`` requires reaching every node; ``reachable`` takes only finite
    distances. A reachable-only isolate has eccentricity zero.
    """
    direction, disconnected = _direction(direction), _disconnected(disconnected)
    n = graph.node_count
    plan = _plan(graph, n, max_work, pairs=n, reverse=graph.directed and direction == "in")
    topology = csr(graph, reverse=graph.directed and direction == "in", loops=False)
    values, reachable = [], []
    for source in range(n):
        distance, order, _ = _paths(topology, source, graph.weighted)
        values.append(_eccentricity(distance, order, n, disconnected))
        reachable.append(len(order) - 1)
    return graph._frame({"eccentricity": values, "reachable_nodes": reachable},
                        **{key: value for key, value in _metadata(
                            graph, "network_eccentricity", direction=direction,
                            disconnected=disconnected, **plan).items() if key != "network"})


def distance_summary(graph, *, direction="out", disconnected="infinite", max_work=50_000_000):
    """Exact ordered-pair averages, diameter/radius and global efficiency.

    Self pairs are excluded from averages. ``infinite`` makes average distance
    and diameter infinite if any ordered pair is unreachable; radius is the
    minimum full-graph eccentricity. ``reachable`` averages finite pairs and
    uses reachable-only eccentricities, including zero for isolates. Global
    efficiency always averages reciprocal finite distances over all ordered
    distinct pairs, assigning zero to unreachable pairs. Strength weights are
    positive *costs*, so weighted efficiency need not be bounded by one.
    """
    direction, disconnected = _direction(direction), _disconnected(disconnected)
    n = graph.node_count
    plan = _plan(graph, n, max_work, reverse=graph.directed and direction == "in")
    topology = csr(graph, reverse=graph.directed and direction == "in", loops=False)
    pair_count, reachable_count = n * (n - 1), 0
    maximum, scaled_distance_sum = 0., 0.
    minimum, scaled_reciprocal_sum = math.inf, 0.
    diameter, radius = 0., math.inf
    for source in range(n):
        distance, order, _ = _paths(topology, source, graph.weighted)
        eccentric = _eccentricity(distance, order, n, disconnected)
        diameter, radius = max(diameter, eccentric), min(radius, eccentric)
        reachable_count += len(order) - 1
        finite = [distance[i] for i in order if i != source]
        if not finite:
            continue
        high, low = max(finite), min(finite)
        if high > maximum:
            scaled_distance_sum *= maximum / high
            maximum = high
        scaled_distance_sum = math.fsum((scaled_distance_sum,
                                        math.fsum(value / maximum for value in finite)))
        if low < minimum:
            scaled_reciprocal_sum *= low / minimum
            minimum = low
        scaled_reciprocal_sum = math.fsum((scaled_reciprocal_sum,
                                          math.fsum(minimum / value for value in finite)))
    average = (scaled_distance_sum / reachable_count) * maximum if reachable_count else 0.
    if disconnected == "infinite" and reachable_count < pair_count:
        average = math.inf
    efficiency = (scaled_reciprocal_sum / pair_count) / minimum if pair_count and reachable_count else 0.
    if not math.isfinite(efficiency):
        _error("precision", "Global efficiency exceeds float64 range; rescale distance costs.")
    result = as_frame(pd.DataFrame([{ "node_count": n, "total_pairs": pair_count,
                                    "reachable_pairs": reachable_count,
                                    "unreachable_pairs": pair_count - reachable_count,
                                    "average_distance": average, "diameter": diameter,
                                    "radius": radius if n else 0., "global_efficiency": efficiency}]))
    result.attrs.update(_metadata(graph, "network_distance_summary", direction=direction,
                                 disconnected=disconnected, ordered_pairs=True,
                                 self_pairs_excluded=True, efficiency_unreachable_contribution=0.,
                                 reachable_isolate_eccentricity=0., **plan))
    return result


def _cut_plan(graph, max_work):
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    n, a = graph.node_count, graph._arcs._nnz()
    projection_arcs = a * (2 if graph.directed else 1)
    sorting = (projection_arcs * max(1, projection_arcs.bit_length()) if graph.directed else
               2 * a * max(1, a.bit_length()) if not graph._arcs.is_coalesced() else 0)
    planned = n * 12 + projection_arcs * 12 + sorting
    if planned > maximum:
        _error("work_budget", f"Weak graph cuts require {planned:,} planned work units, "
               f"above max_work={maximum:,}.")
    graph._guard(1024 * n + 160 * projection_arcs + 4096)
    return dict(max_work=maximum, planned_work=planned,
                work_unit="conservative structural traversal/sort/output units; not elapsed time")


def _cuts(graph, max_work):
    """Iterative Tarjan; each projected arc is examined exactly once."""
    plan = _cut_plan(graph, max_work)
    topology = csr(graph, loops=False, undirected=graph.directed)
    n = graph.node_count
    discovery, low, parent = [-1] * n, [0] * n, [-1] * n
    next_arc, children, articulation = [0] * n, [0] * n, [False] * n
    bridge_indices, clock = [], 0
    for root in range(n):
        if discovery[root] >= 0:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        next_arc[root] = topology.offsets[root]
        stack = [root]
        while stack:
            node = stack[-1]
            if next_arc[node] < topology.offsets[node + 1]:
                target = topology.columns[next_arc[node]]
                next_arc[node] += 1
                if target == parent[node]:
                    continue
                if discovery[target] < 0:
                    parent[target] = node
                    children[node] += 1
                    discovery[target] = low[target] = clock
                    clock += 1
                    next_arc[target] = topology.offsets[target]
                    stack.append(target)
                else:
                    low[node] = min(low[node], discovery[target])
            else:
                stack.pop()
                previous = parent[node]
                if previous < 0:
                    articulation[node] = children[node] > 1
                else:
                    low[previous] = min(low[previous], low[node])
                    if low[node] > discovery[previous]:
                        bridge_indices.append((previous, node))
                    if parent[previous] >= 0 and low[node] >= discovery[previous]:
                        articulation[previous] = True
    return articulation, bridge_indices, plan


def articulation_points(graph, *, max_work=50_000_000):
    """A boolean per node; directed graphs use the undirected weak projection.

    These are weak articulation points, not directed strong articulation.
    Self loops and duplicate edge records do not change connectivity.
    """
    points, _, plan = _cuts(graph, max_work)
    return graph._frame({"articulation": points}, kind="network_articulation_points",
                        exact=True, algorithm="iterative Tarjan", device="cpu",
                        connectivity="weak" if graph.directed else "undirected",
                        weight_semantics="binary simple topology", **plan)


def bridges(graph, *, max_work=50_000_000):
    """Bridge table of the simple graph or directed graph's weak projection.

    Each row denotes one *projected connection*: deleting all directed arcs
    and duplicates on that connection raises the weak component count.
    """
    _, indices, plan = _cuts(graph, max_work)
    records = []
    for left, right in indices:
        a, b = graph._labels[left], graph._labels[right]
        records.append((a, b) if _key(a) <= _key(b) else (b, a))
    records.sort(key=lambda pair: (_key(pair[0]), _key(pair[1])))
    result = as_frame(pd.DataFrame({"source": pd.Series([a for a, _ in records], dtype=object),
                                   "target": pd.Series([b for _, b in records], dtype=object)}))
    result.attrs.update(network=graph.metadata, kind="network_bridges", exact=True,
                        algorithm="iterative Tarjan", device="cpu",
                        connectivity="weak" if graph.directed else "undirected",
                        edge_scope="simple projected connections",
                        weight_semantics="binary simple topology", **plan)
    return result


def minimum_spanning_forest(graph, *, max_work=50_000_000):
    """Kruskal forest retaining all nodes, isolates and selected scalar attrs.

    Directed graphs are rejected. Weighted edge costs are aggregate weights;
    unweighted costs are one per unique edge, independent of duplicate rows.
    The returned graph retains original selected aggregate strengths and its
    original weighted flag. Loops never enter the forest. Cost ties follow
    the snapshot's coalesced edge order, making repeated calls deterministic.
    """
    if graph.directed:
        _error("invalid_option", "Minimum spanning forests require an undirected graph.")
    maximum = _integer(max_work, "max_work", 2**63 - 1)
    n, e = graph.node_count, graph.edge_count
    planned = 16 * n + e * (max(1, e.bit_length()) + 16)
    if planned > maximum:
        _error("work_budget", f"Kruskal requires {planned:,} planned work units, "
               f"above max_work={maximum:,}.")
    attribute_bytes = graph._metadata.get("attribute_storage_bytes", 0)
    # Source graph, sorted permutation, union-find, selected COO/coalescing,
    # child graph and attribute cloning coexist. Admit the combined peak first.
    graph._guard(1024 * n + 160 * e + graph._label_bytes + 2 * attribute_bytes + 4096)
    pairs, weights = graph._edges.indices(), graph._edges.values()
    with torch.device("cpu"), torch.no_grad():
        order = torch.argsort(weights, stable=True) if graph.weighted else torch.arange(e, dtype=torch.int64)
    left, right = memoryview(pairs[0].numpy()), memoryview(pairs[1].numpy())
    parent, size, selected = list(range(n)), [1] * n, []
    components = n

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for index in memoryview(order.numpy()):
        a, b = find(left[index]), find(right[index])
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
        selected.append(index)
        components -= 1
        if components == 1:
            break
    with torch.device("cpu"), torch.no_grad():
        chosen = torch.tensor(selected, dtype=torch.int64)
        edges = torch.sparse_coo_tensor(pairs[:, chosen], weights[chosen], (n, n),
                                        dtype=torch.float64, device="cpu", check_invariants=True).coalesce()
    # Selection does not depend on adding edge costs. Accumulate the public
    # positive total accurately only after the forest is selected.
    chosen_weights = memoryview(edges.values().numpy())
    if graph.weighted and len(chosen_weights):
        try:
            total_cost = math.fsum(chosen_weights)
        except OverflowError:
            total_cost = math.inf
        if not math.isfinite(total_cost):
            _error("precision", "Minimum spanning forest total cost exceeds float64 range.")
    else:
        total_cost = float(len(selected))
    metadata = deepcopy(graph._metadata)
    metadata.update(input_rows=len(selected), positive_edge_rows=len(selected),
                    missing_rows_dropped=0, zero_weight_rows_dropped=0,
                    duplicate_edge_rows_aggregated=0, derived_from="minimum spanning forest",
                    parent_node_count=n, parent_edge_count=e, algorithm="Kruskal union-find",
                    component_count=components, total_cost=total_cost,
                    cost_semantics="aggregate edge cost" if graph.weighted else "unique-edge unit cost",
                    retained_edge_strengths=True, exact=True, device="cpu",
                    max_work=maximum, planned_work=planned,
                    tie_break="stable coalesced snapshot edge order")
    # Remove source-only attributes accounting before Network recalculates the
    # structural child storage and attach_attributes accounts for its own copy.
    metadata.pop("attribute_storage_bytes", None)
    result = Network(graph._labels, graph._index.copy(), edges, False, graph.weighted,
                     _Budget(graph._budget.limit / 1024**2), graph._label_bytes, metadata)
    selected_pairs = set()
    for index in selected:
        a, b = graph._labels[left[index]], graph._labels[right[index]]
        selected_pairs.add((a, b) if _key(a) <= _key(b) else (b, a))
    return result.with_attributes(nodes=graph._node_attributes,
                                  edges={pair: attrs for pair, attrs in graph._edge_attributes.items()
                                         if pair in selected_pairs},
                                  graph_attributes=graph._graph_attributes)
