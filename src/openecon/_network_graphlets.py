"""Exact bounded induced graphlet classes and automorphism orbits (4--5 nodes)."""

from __future__ import annotations

from itertools import combinations, permutations
import math

import pandas as pd

from openecon._network_flow import _Work
from openecon.frame import as_frame
from openecon.networks import _error, _integer


def _canonical(code, size, directed, work):
    pairs = [(u, v) for u in range(size) for v in range(size) if u != v and (directed or u < v)]
    present = {pair for at, pair in enumerate(pairs) if code & (1 << at)}
    if not directed:
        present |= {(v, u) for u, v in present}
    best, chosen = None, None
    for order in permutations(range(size)):
        work.add(len(pairs))
        value = sum(1 << at for at, (u, v) in enumerate(pairs) if (order[u], order[v]) in present)
        if best is None or value < best:
            best, chosen = value, order
    canonical_edges = {pair for at, pair in enumerate(pairs) if best & (1 << at)}
    if not directed:
        canonical_edges |= {(v, u) for u, v in canonical_edges}
    # Equivalence classes under all adjacency-preserving bijections.
    orbits = [set([u]) for u in range(size)]
    for order in permutations(range(size)):
        work.add(len(pairs))
        if all(
            ((u, v) in canonical_edges) == ((order[u], order[v]) in canonical_edges)
            for u, v in pairs
        ):
            for u, v in enumerate(order):
                orbits[u].add(v)
    masks = [sum(1 << u for u in orbit) for orbit in orbits]
    local_masks = [0] * size
    for canonical_node, local in enumerate(chosen):
        local_masks[local] = masks[canonical_node]
    return best, tuple(local_masks)


def graphlets(
    graph,
    size=4,
    *,
    connected=True,
    max_subgraphs=1_000_000,
    max_output_rows=1_000_000,
    max_work=50_000_000,
):
    """Count induced topology classes and sparse per-node orbit participation.

    ``connected`` means weakly connected for directed motifs. False includes
    every disconnected class, including the empty motif. Loops and multigraphs
    are rejected, never projected. Edge strength is ignored explicitly; there
    is no sampled or non-induced path. Absent orbit rows have zero count.
    Class IDs encode the minimum adjacency bit mask under all permutations;
    orbit IDs encode canonical-position automorphism sets, not another
    package's numerical orbit numbering.
    """
    size = _integer(size, "size", 5)
    if size < 4 or not isinstance(connected, bool):
        _error("invalid_option", "size must be 4 or 5; connected must be boolean.")
    if getattr(graph, "_is_multigraph", False):
        _error(
            "unsupported", "Graphlets require an explicit simple-graph projection of multiedges."
        )
    maximum = _integer(max_subgraphs, "max_subgraphs", 10_000_000, zero=True)
    output_limit = _integer(max_output_rows, "max_output_rows", 10_000_000, zero=True)
    total = math.comb(graph.node_count, size) if graph.node_count >= size else 0
    if total > maximum:
        _error(
            "work_budget",
            "Complete induced subset enumeration exceeds max_subgraphs before allocation.",
        )
    work = _Work(max_work, graph.node_count + graph.edge_count + total * size * size)
    graph._guard(512 * graph.edge_count + 512 * graph.node_count)
    indices = graph._edges.indices()
    edges = set(zip(indices[0].tolist(), indices[1].tolist()))
    if any(u == v for u, v in edges):
        _error(
            "unsupported",
            "Looped graphlet classes are unsupported; loops are not silently removed.",
        )
    if not graph.directed:
        edges |= {(v, u) for u, v in edges}
    pairs = [
        (u, v) for u in range(size) for v in range(size) if u != v and (graph.directed or u < v)
    ]
    counts, orbit_counts, cache = {}, {}, {}
    considered = 0
    for subset in combinations(range(graph.node_count), size):
        code = sum(1 << at for at, (u, v) in enumerate(pairs) if (subset[u], subset[v]) in edges)
        if connected:
            seen, pending = {0}, [0]
            while pending:
                u = pending.pop()
                for v in range(size):
                    if v not in seen and (
                        (subset[u], subset[v]) in edges or (subset[v], subset[u]) in edges
                    ):
                        seen.add(v)
                        pending.append(v)
            if len(seen) < size:
                continue
        if code not in cache:
            graph._guard(
                512 * graph.edge_count
                + 512 * graph.node_count
                + 1024 * (len(cache) + 1)
                + 512 * (len(orbit_counts) + len(counts))
            )
            cache[code] = _canonical(code, size, graph.directed, work)
        canonical, orbits = cache[code]
        class_id = f"{'d' if graph.directed else 'u'}{size}:{canonical:x}"
        new_rows = int(class_id not in counts) + sum(
            (u, class_id, orbit) not in orbit_counts for u, orbit in zip(subset, orbits)
        )
        if len(counts) + len(orbit_counts) + new_rows > output_limit:
            _error(
                "output_budget",
                "Complete class/orbit result exceeds max_output_rows; no partial counts returned.",
            )
        graph._guard(
            512 * graph.edge_count
            + 512 * graph.node_count
            + 1024 * len(cache)
            + 1024 * (len(orbit_counts) + len(counts) + new_rows)
        )
        counts[class_id] = counts.get(class_id, 0) + 1
        for u, orbit in zip(subset, orbits):
            key = (u, class_id, orbit)
            orbit_counts[key] = orbit_counts.get(key, 0) + 1
        considered += 1
    classes = as_frame(
        pd.DataFrame(
            [dict(class_id=c, count=n) for c, n in sorted(counts.items())],
            columns=["class_id", "count"],
        )
    )
    orbits = as_frame(
        pd.DataFrame(
            [
                dict(node=graph._labels[u], class_id=c, orbit=f"{c}/{o:x}", count=n)
                for (u, c, o), n in sorted(orbit_counts.items())
            ],
            columns=["node", "class_id", "orbit", "count"],
        )
    )
    orbits["node"] = pd.Series([graph._labels[u] for u, _, _ in sorted(orbit_counts)], dtype=object)
    metadata = dict(
        kind="network_graphlets",
        network=graph.metadata,
        size=size,
        directed=graph.directed,
        induced=True,
        connected=connected,
        connectivity="weak" if graph.directed else "undirected",
        exact=True,
        sampled=False,
        device="cpu",
        weight_semantics="binary topology; all stored positive edges present",
        class_encoding="lexicographically ordered off-diagonal adjacency bits; minimum integer over vertex permutations",
        orbit_encoding="automorphism set bitmask of canonical vertices; absent rows are zero",
        subsets_enumerated=total,
        included_subgraphs=considered,
        max_subgraphs=maximum,
        output_rows=len(classes) + len(orbits),
        max_output_rows=output_limit,
        work_used=work.used,
        max_work=work.maximum,
        algorithm="bounded exhaustive induced subsets and exact permutation canonicalization",
    )
    classes.attrs.update(metadata)
    orbits.attrs.update(metadata)
    return {"classes": classes, "orbits": orbits, "metadata": metadata}
