"""Validate native network results against pinned public data and scalar references.

Terminal: python docs/examples/network_reference_validation.py --output receipt.json
Code panel: paste this file, then call validate('/absolute/path/to/tests/fixtures/network_reference').
The fixture directory is explicit in the panel. No download or external solver runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import openecon as oe
import torch


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, *, tolerance=2e-11):
    check(math.isfinite(actual) and abs(actual - expected) <= tolerance,
          f"Reference mismatch: {actual!r} versus {expected!r} (absolute {tolerance})")


def load_fixture(directory, name):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    check(manifest["schema_version"] == 1, "Unsupported fixture manifest")
    record = manifest["datasets"][name]
    path = directory / record["file"]
    check(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
          f"The pinned {name} fixture changed")
    if name == "karate":
        with path.open() as handle:
            pairs = [(int(row["source"]), int(row["target"])) for row in csv.DictReader(handle)]
    else:
        with gzip.open(path, "rt", encoding="ascii") as handle:
            pairs = [tuple(map(int, line.split())) for line in handle
                     if line.strip() and not line.startswith("#")]
    physical_rows = len(pairs)
    # This projection is required for SNAP's reciprocal physical records. It
    # happens before library import; coalescing must not double every capacity.
    edges = sorted({tuple(sorted(pair)) for pair in pairs})
    nodes = sorted({node for edge in edges for node in edge})
    check(len(nodes) == record["expected_nodes"] and len(edges) == record["expected_edges"],
          f"Wrong public {name} topology")
    if "expected_physical_rows" in record:
        check(physical_rows == record["expected_physical_rows"], "Wrong physical input row count")
    return nodes, edges, record, physical_rows


def adjacency(nodes, edges):
    """Raw-file simple adjacency; loops do not participate in topology counts."""
    neighbors = {node: set() for node in nodes}
    for source, target in edges:
        if source != target:
            neighbors[source].add(target)
            neighbors[target].add(source)
    return neighbors


def components(neighbors):
    unseen, sizes = set(neighbors), []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, size = deque([start]), 0
        while queue:
            node = queue.popleft()
            size += 1
            for target in neighbors[node] & unseen:
                unseen.remove(target)
                queue.append(target)
        sizes.append(size)
    return sorted(sizes)


def stationary(nodes, edges, damping=.85, *, directed=False):
    """Dense scalar Gaussian elimination, independent of sparse power iteration.

    Used only on the 34-node reference. No Torch, graph solver or production
    adjacency helper is used to derive the expected values.
    """
    n, index = len(nodes), {node: i for i, node in enumerate(nodes)}
    weights = [[0.] * n for _ in nodes]
    for record in edges:
        source, target = record[:2]
        weight = record[2] if len(record) == 3 else 1.
        i, j = index[source], index[target]
        weights[i][j] += weight
        if not directed and i != j:
            weights[j][i] += weight
    transition = [[value / sum(row) for value in row] if sum(row) else [1 / n] * n
                  for row in weights]
    system = [[float(i == j) - damping * transition[j][i] for j in range(n)]
              + [(1 - damping) / n] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(system[i][col]))
        system[col], system[pivot] = system[pivot], system[col]
        scale = system[col][col]
        system[col] = [value / scale for value in system[col]]
        for row in range(n):
            if row != col:
                scale = system[row][col]
                system[row] = [a - scale * b for a, b in zip(system[row], system[col])]
    return dict(zip(nodes, (row[-1] for row in system)))


def distances(nodes, neighbors):
    """Dense min-plus closure on raw adjacency, not a production path query."""
    distance = {(a, b): 0 if a == b else 1 if b in neighbors[a] else math.inf
                for a in nodes for b in nodes}
    for k in nodes:
        for i in nodes:
            for j in nodes:
                distance[i, j] = min(distance[i, j], distance[i, k] + distance[k, j])
    return distance


def modularity(nodes, edges, membership):
    """Exact rational objective for a loopless unit-weight undirected graph."""
    check(all(a != b for a, b in edges), "This reference objective requires loopless data")
    degree = Counter(node for edge in edges for node in edge)
    adjacent = {tuple(sorted(edge)) for edge in edges}
    total = 2 * len(edges)
    return float(sum((Fraction(int(tuple(sorted((a, b))) in adjacent), total)
                      - Fraction(degree[a] * degree[b], total * total)
                      for a in nodes for b in nodes if membership[a] == membership[b]), Fraction()))


def flow_certificate(result, nodes, edges):
    """Check a feasible primal flow and an independently summed dual cut.

    Equal feasible primal and dual values prove optimality without running a
    second max-flow algorithm. All endpoints/capacities come from the raw file.
    """
    left, right = set(result["source_partition"]), set(result["target_partition"])
    check(left | right == set(nodes) and not left & right, "Invalid cut partition")
    check(result["source"] in left and result["target"] in right, "Invalid cut endpoints")
    cut = {edge for edge in edges if (edge[0] in left) != (edge[1] in left)}
    close(result["value"], len(cut))
    balances, seen = dict.fromkeys(nodes, 0.), set()
    for row in result["flows"].itertuples(index=False):
        edge = tuple(sorted((row.source, row.target)))
        check(edge in edges and edge not in seen, "Flow has missing or duplicate edge identity")
        seen.add(edge)
        close(row.capacity, 1.)
        check(abs(row.flow) <= 1 + 1e-12, "Flow exceeds raw-file capacity")
        balances[row.source] += row.flow
        balances[row.target] -= row.flow
    check(seen == set(edges), "Flow output omitted edges")
    for node, balance in balances.items():
        close(balance, result["value"] if node == result["source"] else
              -result["value"] if node == result["target"] else 0.)
    reported_cut = {tuple(sorted((row.source, row.target)))
                    for row in result["cut_edges"].itertuples(index=False)}
    check(reported_cut == cut, "Cut certificate omitted or added edges")


def validate(directory, *, names=("karate", "grqc"), threads=2):
    """Return a durable receipt; fail immediately on any numerical mismatch."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "running", "datasets": {},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "torch": str(torch.__version__), "sdk": oe.__version__,
                              "threads": threads, "device": "cpu", "dtype": "float64"},
              "scope": "Pinned public undirected graphs and independent scalar/certificate checks; "
                       "no CUDA, universal scale, temporal estimation or Stata-parity claim."}
    try:
        processor = platform.processor() or platform.machine()
        if sys.platform == "darwin":
            result = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, check=False)
            if result.returncode == 0:
                processor = result.stdout.strip()
        report["environment"]["processor"] = processor
        report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
        for name in names:
            before = time.perf_counter()
            nodes, edges, record, physical_rows = load_fixture(directory, name)
            neighbors = adjacency(nodes, edges)
            graph = oe.network({"source": [a for a, _ in edges], "target": [b for _, b in edges]},
                               nodes=list(reversed(nodes)), batch_rows=127, max_memory_mb=256)
            timing = {"input_and_graph": time.perf_counter() - before}
            check(graph.node_count == len(nodes) and graph.edge_count == len(edges), "Import changed topology")
            degree_table = graph.degree()
            degree = dict(zip(degree_table.node, degree_table.degree))
            expected_degree = Counter(node for edge in edges for node in edge)
            check(degree == expected_degree, "Degree/reference identity alignment failed")
            sizes = sorted(Counter(graph.components().component).values())
            check(sizes == components(neighbors), "Weak components differ from independent reachability")
            triangle_count = {node: sum(b in neighbors[a] for a in neighbors[node]
                                       for b in neighbors[node] if a < b) for node in nodes}
            triangle = graph.triangles()
            check(dict(zip(triangle.node, triangle.triangles)) == triangle_count,
                  "Triangles differ from raw-file unordered-neighbor enumeration")
            expected_clustering = {node: 2 * triangle_count[node] / (len(neighbors[node]) *
                                   (len(neighbors[node]) - 1)) if len(neighbors[node]) > 1 else 0.
                                   for node in nodes}
            clustering = graph.clustering()
            for row in clustering.itertuples(index=False):
                close(row.clustering, expected_clustering[row.node])
            details = {"physical_rows": physical_rows, "nodes": graph.node_count,
                       "edges": graph.edge_count, "loops": sum(a == b for a, b in edges),
                       "component_count": len(sizes), "largest_component_nodes": max(sizes),
                       "triangles": sum(triangle_count.values()) // 3,
                       "mean_clustering": sum(expected_clustering.values()) / len(nodes),
                       "fixture_sha256": record["sha256"], "source_url": record["source_url"],
                       "timing_seconds": timing, "checks": ["degree identities", "weak components",
                           "triangles", "nodewise clustering"]}
            if name == "grqc":
                check(details["loops"] == record["expected_loops"], "Published loop count changed")
                check(details["largest_component_nodes"] == record["expected_largest_component_nodes"],
                      "Published largest component differs")
                check(details["triangles"] == record["expected_triangles"], "Published triangle count differs")
                close(details["mean_clustering"], record["published_mean_clustering"],
                      tolerance=record["mean_clustering_absolute_tolerance"])
                details["checks"].append("SNAP published counts and rounded mean clustering")
            else:
                rank = graph.pagerank(tol=1e-12, max_iter=1000)
                expected_rank = stationary(nodes, edges)
                for row in rank.itertuples(index=False):
                    close(row.pagerank, expected_rank[row.node], tolerance=2e-12)
                raw_distance = distances(nodes, neighbors)
                for row in graph.distances(max_pairs=2000).itertuples(index=False):
                    close(row.distance, raw_distance[row.source, row.target], tolerance=0.)
                # Each internal vertex on a shortest path contributes one unit.
                # Sum across all vertices equals the ordered distance excess,
                # irrespective of which equal-length shortest path is chosen.
                expected_sum = sum(d - 1 for (a, b), d in raw_distance.items() if a != b) / (
                    (len(nodes) - 1) * (len(nodes) - 2))
                close(graph.betweenness().betweenness.sum(), expected_sum)
                community_checks = []
                for method in ("leiden", "louvain"):
                    for seed in (0, 11):
                        partition = graph.communities(method=method, seed=seed)
                        membership = dict(zip(partition.node, partition.community))
                        check(set(membership) == set(nodes), "Community membership lost identities")
                        expected = modularity(nodes, edges, membership)
                        close(graph.modularity(partition), expected)
                        close(partition.attrs["modularity"], expected)
                        community_checks.append({"method": method, "seed": seed, "modularity": expected})
                flow_certificate(graph.max_flow(1, 34), nodes, edges)
                details["community_objectives"] = community_checks
                details["checks"] += ["PageRank scalar Gaussian reference", "all-pairs min-plus closure",
                    "exact betweenness distance-sum identity", "rational community objectives",
                    "primal flow conservation/capacities and dual cut optimality"]
            timing["complete_case"] = time.perf_counter() - before
            report["datasets"][name] = details
        try:
            import resource
            raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = int(raw_rss * (1 if sys.platform == "darwin" else 1024)) if (
                sys.platform == "darwin" or sys.platform.startswith("linux")) else None
        except ImportError:
            rss = None
        report.update(status="passed", elapsed_seconds=time.perf_counter() - started,
                      peak_process_rss_bytes=rss,
                      rss_scope="Whole process high-water mark, including imports and reference buffers; "
                                "not the graph's owned-buffer estimate or memory-budget guarantee.",
                      full_graph_analysis=True, display_sampling=False, out_of_core_graph=False)
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tests/fixtures/network_reference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Choose a new receipt path; existing evidence is never replaced.")
    report = validate(args.fixtures)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__" and "display" not in globals():
    main()
