"""Reproducible full-network export benchmark; no browser or user data involved.

This is a native sparse graph -> chart validator -> bounded streaming JSON
measurement. It is not a rendered-frame benchmark or whole-process RAM limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import torch

from openecon.networks import network
from openecon_charts.network import MAX_PAYLOAD_BYTES, validate_network


def run(nodes=100_000, edges=1_000_000, max_memory_mb=1536):
    if not 2 <= nodes <= 100_000 or not 1 <= edges <= 1_000_000 or edges > min(nodes * 10, nodes * (nodes - 1) // 2):
        raise ValueError("Use 2..100,000 nodes and a distinct nonnegative ring graph with at most 10 forward offsets.")
    torch.set_num_threads(2)
    def records():
        for index in range(edges):
            source = index % nodes
            yield {"source": source, "target": (source + 1 + index // nodes) % nodes}
    start = perf_counter()
    graph = network(records(), nodes=range(nodes), batch_rows=8192, max_memory_mb=max_memory_mb)
    built = perf_counter()
    payload = graph.to_plot_data(max_nodes=nodes, max_edges=edges)
    exported = perf_counter()
    clean = validate_network(payload)
    validated = perf_counter()
    digest, byte_count = hashlib.sha256(), 0
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for chunk in encoder.iterencode(clean):
        raw = chunk.encode("utf-8")
        byte_count += len(raw)
        if byte_count > MAX_PAYLOAD_BYTES:
            raise ValueError("Full encoded output exceeds the explicit 128 MiB chart payload cap.")
        digest.update(raw)
    completed = perf_counter()
    assert graph.node_count == clean["shown_node_count"] == clean["node_count"] == nodes
    assert graph.edge_count == clean["shown_edge_count"] == clean["edge_count"] == edges
    assert not clean["sampled"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"schema": 1, "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "nodes": nodes, "stored_edges": edges, "shown_nodes": clean["shown_node_count"],
            "shown_edges": clean["shown_edge_count"], "sampled": clean["sampled"],
            "graph_max_memory_mb": max_memory_mb, "payload_max_bytes": MAX_PAYLOAD_BYTES,
            "encoded_json_bytes": byte_count, "encoded_json_sha256": digest.hexdigest(),
            "import_seconds": built - start, "native_export_seconds": exported - built,
            "strict_validation_seconds": validated - exported, "streaming_json_seconds": completed - validated,
            "total_seconds": completed - start,
            "process_peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
            "graph_estimated_import_peak_bytes": graph.metadata["estimated_import_peak_bytes"],
            "scope": "synthetic CPU sparse graph, complete native export, chart validation and streaming JSON; no browser rendering, GPU or human data"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--edges", type=int, default=1_000_000)
    parser.add_argument("--max-memory-mb", type=float, default=1536)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.nodes, args.edges, args.max_memory_mb)
    source = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(source)
    print(source, end="")


if __name__ == "__main__":
    main()
