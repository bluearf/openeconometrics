"""Exact rational Hungarian assignment and bounded general matching."""

from __future__ import annotations

from fractions import Fraction
import math
from numbers import Integral

import pandas as pd

from openecon._network_flow import _Work
from openecon.frame import as_frame
from openecon.networks import _error, _integer, _key, _label


def _display(value):
    try:
        out = float(value)
    except OverflowError:
        _error("precision", "Matching objective exceeds finite float64 range.")
    if not math.isfinite(out):
        _error("precision", "Matching objective exceeds finite float64 range.")
    return out


class NetworkMatchingResult(dict):
    def summary(self):
        values = [
            ("Objective", self["metadata"]["objective"]),
            ("Cardinality", len(self["pairs"])),
            ("Total weight", self["weight"]),
            ("Unmatched nodes", len(self["unmatched"])),
            ("Certified", True),
            ("Algorithm", self["metadata"]["algorithm"]),
        ]
        return as_frame(pd.DataFrame(values, columns=["Metric", "Value"]))

    def to_latex(self, buf=None, **kwargs):
        kwargs.setdefault("index", False)
        return self.summary().to_latex(buf=buf, **kwargs)

    def __repr__(self):
        return f"NetworkMatchingResult(cardinality={len(self['pairs'])}, weight={self['weight']}, certified=True)"


def _edges(graph, objective, max_work):
    if objective not in ("weight", "cardinality", "cardinality_weight"):
        _error(
            "invalid_option", "objective must be 'weight', 'cardinality', or 'cardinality_weight'."
        )
    if graph.directed:
        _error(
            "invalid_option",
            "Matching requires an undirected graph; no weak projection is performed.",
        )
    graph._guard(2048 * (graph.node_count + graph.edge_count))
    work = _Work(max_work, graph.node_count + graph.edge_count)
    multi = getattr(graph, "_is_multigraph", False)
    if multi:
        pairs, values, identifiers = (
            graph._endpoints.T.tolist(),
            graph._weights.tolist(),
            graph._edge_ids,
        )
    else:
        pairs, values = graph._edges.indices().T.tolist(), graph._edges.values().tolist()
        identifiers = [tuple(graph._labels[u] for u in pair) for pair in pairs]
    best = {}
    for (u, v), weight, identifier in zip(pairs, values, identifiers):
        if u == v:
            _error("unsupported", "Self-loops are not matching edges; remove them explicitly.")
        if not math.isfinite(weight):
            _error("invalid_weight", "Matching weights must be finite.")
        pair, value = (min(u, v), max(u, v)), Fraction.from_float(weight)
        item, tie = best.get(pair), _key(identifier) if multi else pair
        # Actual parallel IDs remain alternatives, never an aggregate reward.
        if (
            item is None
            or (objective != "cardinality" and value > item[0])
            or ((objective == "cardinality" or value == item[0]) and tie < item[2])
        ):
            best[pair] = value, identifier, tie
    return best, work


def _result(graph, selected, edges, objective, work, **metadata):
    rows, used, total = [], set(), Fraction(0)
    for u, v in selected:
        u, v = min(u, v), max(u, v)
        value, identifier, _ = edges[u, v]
        total += value
        used.update((u, v))
        rows.append(
            dict(
                source=graph._labels[u],
                target=graph._labels[v],
                edge_id=identifier,
                weight=_display(value),
            )
        )
    pairs = as_frame(pd.DataFrame(rows, columns=["source", "target", "edge_id", "weight"]))
    for name in ("source", "target", "edge_id"):
        pairs[name] = pd.Series([row[name] for row in rows], dtype=object)
    unmatched = as_frame(
        pd.DataFrame(
            {
                "node": pd.Series(
                    [v for i, v in enumerate(graph._labels) if i not in used], dtype=object
                )
            }
        )
    )
    info = dict(
        kind="network_matching",
        network=graph.metadata,
        objective=objective,
        cardinality=len(rows),
        exact_weight=str(total),
        total_weight=_display(total),
        certified=True,
        exact=True,
        sampled=False,
        device="cpu",
        weight_semantics="finite signed reward; weight-only may leave zero/negative edges unmatched",
        parallel_edges="distinct alternatives; actual chosen edge ID retained, never summed",
        work_used=work.used,
        max_work=work.maximum,
        **metadata,
    )
    pairs.attrs.update(info)
    unmatched.attrs.update(info)
    return NetworkMatchingResult(
        pairs=pairs, unmatched=unmatched, weight=_display(total), metadata=info
    )


def weighted_assignment(
    graph, partition=None, *, objective="weight", max_matrix_entries=1_000_000, max_work=50_000_000
):
    """Exact rectangular Hungarian assignment with unmatched dummy columns.

    Missing edges are forbidden. Dense augmented geometry is explicitly capped.
    Finite signed binary64 rewards/potentials use exact rational arithmetic;
    the returned transformed-minimization primal/dual certificate is verified.
    ``cardinality_weight`` maximizes cardinality first, then total reward.
    """
    edges, work = _edges(graph, objective, max_work)
    n = graph.node_count
    if partition is None:
        colors, adj = [-1] * n, [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for root in sorted(range(n), key=lambda i: _key(graph._labels[i])):
            if colors[root] >= 0:
                continue
            colors[root], pending = 0, [root]
            while pending:
                u = pending.pop()
                for v in adj[u]:
                    work.add()
                    if colors[v] < 0:
                        colors[v] = 1 - colors[u]
                        pending.append(v)
                    elif colors[v] == colors[u]:
                        _error(
                            "invalid_partition",
                            "Graph is not bipartite; general matching is a separate API.",
                        )
    else:
        if not isinstance(partition, dict) or {_label(v) for v in partition} != set(graph._labels):
            _error("invalid_partition", "partition must map every exact node ID to integer 0 or 1.")
        colors = [partition[v] for v in graph._labels]
        if any(
            isinstance(c, bool) or not isinstance(c, Integral) or c not in (0, 1) for c in colors
        ):
            _error("invalid_partition", "Partition values must be integer 0 or 1.")
        if any(colors[u] == colors[v] for u, v in edges):
            _error("invalid_partition", "Every matching edge must cross the supplied partition.")
    left = sorted((u for u in range(n) if colors[u] == 0), key=lambda i: _key(graph._labels[i]))
    right = sorted((u for u in range(n) if colors[u] == 1), key=lambda i: _key(graph._labels[i]))
    nr, nc = len(left), len(right) + len(left)
    limit = _integer(max_matrix_entries, "max_matrix_entries", 10_000_000, zero=True)
    if nr * nc > limit:
        _error(
            "memory_budget",
            "Augmented assignment matrix exceeds max_matrix_entries before allocation.",
        )
    plan = nr * nr * (nc + 1) * 4 + 2 * nr * nc + work.used
    if plan > work.maximum:
        _error("work_budget", "Full Hungarian work plan exceeds max_work before allocation.")
    graph._guard(2048 * (n + graph.edge_count + nr * nc + nc + nr))
    bound = sum((abs(e[0]) for e in edges.values()), Fraction(0))
    bonus = 2 * bound + 1
    forbidden = (nr + 2) * bonus + 2 * bound + 1
    costs = []
    for u in left:
        row = []
        for v in right:
            item = edges.get((min(u, v), max(u, v)))
            reward = (
                (
                    item[0]
                    if objective == "weight"
                    else (Fraction(1) if objective == "cardinality" else bonus + item[0])
                )
                if item
                else None
            )
            row.append(-reward if reward is not None else forbidden)
        costs.append([*row, *([Fraction(0)] * nr)])
    pu, pv = [Fraction(0)] * (nr + 1), [Fraction(0)] * (nc + 1)
    assignment, way = [0] * (nc + 1), [0] * (nc + 1)
    for i in range(1, nr + 1):
        assignment[0], j0 = i, 0
        minimum, used = [None] * (nc + 1), [False] * (nc + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = assignment[j0], None, 0
            for j in range(1, nc + 1):
                if used[j]:
                    continue
                work.add()
                cur = costs[i0 - 1][j - 1] - pu[i0] - pv[j]
                if minimum[j] is None or cur < minimum[j]:
                    minimum[j], way[j] = cur, j0
                if delta is None or minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(nc + 1):
                work.add()
                if used[j]:
                    pu[assignment[j]] += delta
                    pv[j] -= delta
                elif minimum[j] is not None:
                    minimum[j] -= delta
            j0 = j1
            if assignment[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            assignment[j0], j0 = assignment[j1], j1
    selected, primal = [], Fraction(0)
    for j in range(1, nc + 1):
        if assignment[j] == 0:
            continue
        i = assignment[j] - 1
        primal += costs[i][j - 1]
        if j <= len(right):
            u, v = left[i], right[j - 1]
            if (min(u, v), max(u, v)) not in edges:
                _error("precision", "Forbidden assignment edge was selected.")
            selected.append((u, v))
    dual = sum(pu[1:]) + sum(pv[1:])
    if primal != dual or any(
        pu[i + 1] + pv[j + 1] > costs[i][j] for i in range(nr) for j in range(nc)
    ):
        _error("precision", "Exact Hungarian primal/dual certificate failed.")
    work.add(nr * nc)
    return _result(
        graph,
        selected,
        edges,
        objective,
        work,
        algorithm="exact-rational rectangular Hungarian assignment",
        tie_break="canonical typed row/column order; first equal reduced-cost column",
        certificate=dict(
            kind="augmented minimization primal/dual",
            primal_exact=str(primal),
            dual_exact=str(dual),
            row_potentials=[str(v) for v in pu[1:]],
            column_potentials=[str(v) for v in pv[1:]],
            dual_feasible=True,
            zero_duality_gap=True,
        ),
        matrix_entries=nr * nc,
        max_matrix_entries=limit,
        planned_work=plan,
    )


def general_matching(
    graph, *, objective="weight", max_component_nodes=24, max_states=1_000_000, max_work=50_000_000
):
    """Exact componentwise subset DP, including odd cycles; exponential and capped.

    Default connected-component limit is 24. There is no bipartite projection,
    approximation, or polynomial blossom scale claim. States/work/owned memory
    fail before a partial answer. The unmatched-or-paired recurrence exhausts
    every admissible matching; its per-component optimum certificate is exact.
    """
    edges, work = _edges(graph, objective, max_work)
    cap = _integer(max_component_nodes, "max_component_nodes", 32)
    state_limit = _integer(max_states, "max_states", 10_000_000)
    n, adj = graph.node_count, [[] for _ in range(graph.node_count)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    components, seen = [], set()
    for root in sorted(range(n), key=lambda i: _key(graph._labels[i])):
        if root in seen:
            continue
        group, pending = [], [root]
        seen.add(root)
        while pending:
            u = pending.pop()
            group.append(u)
            for v in adj[u]:
                work.add()
                if v not in seen:
                    seen.add(v)
                    pending.append(v)
        if len(group) > cap:
            _error(
                "work_budget",
                "General matching component exceeds max_component_nodes; no implicit approximation.",
            )
        components.append(sorted(group, key=lambda i: _key(graph._labels[i])))
    selected, total_states, certificates = [], 0, []
    for group in components:
        local, memo = {u: i for i, u in enumerate(group)}, {0: (0, Fraction(0), ())}

        def score(value):
            count, weight, _ = value
            return (
                (weight,)
                if objective == "weight"
                else ((count,) if objective == "cardinality" else (count, weight))
            )

        def solve(mask):
            if mask in memo:
                return memo[mask]
            work.add()
            i = (mask & -mask).bit_length() - 1
            u, remaining = group[i], mask & ~(1 << i)
            best = solve(remaining)
            for v in adj[u]:
                work.add()
                j = local[v]
                if remaining & (1 << j):
                    count, weight, pairs = solve(remaining & ~(1 << j))
                    pair = (min(u, v), max(u, v))
                    candidate = (count + 1, weight + edges[pair][0], tuple(sorted((*pairs, pair))))
                    if score(candidate) > score(best) or (
                        score(candidate) == score(best) and candidate[2] < best[2]
                    ):
                        best = candidate
            if total_states + len(memo) >= state_limit:
                _error(
                    "work_budget",
                    "General matching exceeds max_states; no partial optimum returned.",
                )
            graph._guard(2048 * (n + graph.edge_count) + 4096 * (len(memo) + 1))
            memo[mask] = best
            return best

        count, weight, pairs = solve((1 << len(group)) - 1)
        selected.extend(pairs)
        total_states += len(memo)
        certificates.append(
            dict(
                nodes=[graph._labels[u] for u in group],
                cardinality=count,
                weight_exact=str(weight),
                states=len(memo),
            )
        )
    return _result(
        graph,
        sorted(selected),
        edges,
        objective,
        work,
        algorithm="exact componentwise subset dynamic programming; exponential",
        tie_break="snapshot endpoint-index pair tuples",
        states=total_states,
        max_states=state_limit,
        max_component_nodes=cap,
        certificate=dict(
            kind="complete exact unmatched-or-paired recurrence",
            component_optima=certificates,
            exhaustive_recurrence=True,
        ),
    )
