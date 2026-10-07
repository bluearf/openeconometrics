"""Measured exact directed cut on a sparse graph with independently known optimum.

A bidirected ring with capacity four has every outgoing cut >= 8. Extra arcs
stay within two halves, so the first-half outgoing cut remains exactly 8.
This proves the global optimum without calling another flow/cut algorithm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch
import openecon as oe
from openecon import _network_cut, _network_directed_cut


def fixture(n):
    if n < 12 or n % 2:
        raise ValueError("Use an even node count of at least 12")
    rows = []
    for node in range(n):
        following = (node + 1) % n
        rows.extend([(node, following, 4.), (following, node, 4.)])
        first, half = (node // (n // 2)) * (n // 2), n // 2
        for offset in (2, 3):
            other = first + (node - first + offset) % half
            rows.extend([(node, other, 1.), (other, node, 1.)])
    return rows


def run(n=500, threads=2, max_work=500_000_000):
    torch.set_num_threads(threads)
    fingerprints = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    for module in (_network_cut, _network_directed_cut)}
    start = time.perf_counter()
    records = fixture(n)
    graph = oe.network([{"source": a, "target": b, "capacity": capacity}
                        for a, b, capacity in records], directed=True, nodes=range(n),
                       weight="capacity", max_memory_mb=256)
    built = time.perf_counter()
    result = graph.global_min_cut(max_work=max_work)
    complete = time.perf_counter()
    assert result["value"] == 8 and result["metadata"]["certified"]
    assert result["metadata"]["exact_capacity_numerator"] == 8
    assert result["metadata"]["exact_capacity_denominator"] == 1
    left, right = set(result["source_partition"]), set(result["target_partition"])
    assert left and right and not left & right and left | right == set(range(n))
    assert sum(weight for a, b, weight in records if a in left and b in right) == 8
    assert result["cut_edges"].capacity.sum() == 8
    assert fingerprints == {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                            for module in (_network_cut, _network_directed_cut)}
    return {"status": "passed", "nodes": n, "edges": graph.edge_count,
            "physical_input_rows": len(records), "known_optimum": 8, "measured_optimum": result["value"],
            "dtype": "float64 stored capacities / exact integer search", "device": "cpu",
            "torch": str(torch.__version__), "python": platform.python_version(),
            "platform": platform.platform(), "threads": threads, "seed": None,
            "fixture": "deterministic bidirected ring + within-half arcs; no random draws",
            "graph_build_seconds": built - start, "global_cut_seconds": complete - built,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024),
            "graph_owned_bytes": graph.metadata["estimated_owned_graph_bytes"],
            "graph_owned_budget_mb": 256, "work_used": result["metadata"]["work_used"],
            "max_work": max_work, "flow_problems": result["metadata"]["flow_problems"],
            "source_sha256": fingerprints, "full_graph_analysis": True, "sampled": False,
            "scope": "Local sparse CPU topology and explicit work budget; lifetime process RSS "
                     "includes imports/input construction; no out-of-core or universal speed claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    result = run(args.nodes)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "nodes", "edges", "global_cut_seconds", "work_used")}))


if __name__ == "__main__":
    main()
