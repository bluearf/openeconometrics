"""Reproducible CPU sparse scale receipt; RSS and owned estimates stay separate."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import openecon as oe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    def measured(name, operation):
        started = time.perf_counter()
        result = operation()
        records.append({"operation": name, "seconds": time.perf_counter() - started,
                        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        return result
    h = measured("25000 nodes / 10000 hyperedges / 40000 memberships", lambda: oe.hypergraph(
        ({"id": i, "members": [(i * 7 + j * 137) % 25_000 for j in range(4)], "weight": i % 5}
         for i in range(10_000)), nodes=range(25_000), max_memory_mb=256))
    d = measured("full hyperdegree/strength", h.degree)
    assert d.hyperdegree.sum() == 40_000
    assert h.incidence()._nnz() == 40_000
    enormous = oe.hypergraph([{"id": "wide", "members": range(25_000)}], max_memory_mb=256)
    def rejected():
        try:
            enormous.clique_projection(max_edges=100_000)
        except oe.AnalysisError as exc:
            assert exc.code == "network_output_budget"
            return True
        raise AssertionError("312487500 candidate pairs must fail before expansion")
    measured("312487500 clique candidates rejected before expansion", rejected)
    g = oe.network(({"source": u, "target": v} for i in range(1000)
                    for u, v in [("s", i), (i, "t")]), directed=True, max_memory_mb=256)
    p = measured("1000 alternative routes / 2000 arcs / top 10", lambda: g.k_shortest_paths("s", "t", k=10))
    assert len(p) == 10 and p.hops.tolist() == [2] * 10
    cycle = oe.network(({"source": i, "target": (i + 1) % 400} for i in range(400)), directed=True)
    b = measured("400-node directed cycle: strong bridges", cycle.strong_bridges)
    a = measured("400-node directed cycle: strong articulation", cycle.strong_articulation_points)
    assert len(b) == 400 and a.strong_articulation.sum() == 400
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root / "src/openecon/_network_hypergraph.py", root / "src/openecon/_network_bounded_paths.py"]}
    receipt = {"status": "passed", "platform": platform.platform(), "architecture": platform.machine(),
               "device": "cpu", "cuda_verified": False, "source_sha256": hashes, "measurements": records,
               "hypergraph_owned_peak_estimate_bytes": h.metadata["owned_peak_estimate_bytes"],
               "path_work": p.attrs["work_used"], "bridge_work": b.attrs["work_used"],
               "articulation_work": a.attrs["work_used"], "dense_adjacency": False,
               "rss_scope": "whole-process high-water mark in bytes on macOS; not the memory admission estimate"}
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "passed", "measurements": records}))


if __name__ == "__main__":
    main()
