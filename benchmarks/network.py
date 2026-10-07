"""Reproducible physical-Parquet graph benchmark; no universal scale promise.

Run from the repository: python benchmarks/network.py --output <owned report>.
Data creation/imports are excluded from the separate algorithm timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openecon.dataset import scan
from openecon.networks import network


def fingerprint(root):
    paths = ["src/openecon/networks.py", "src/openecon/dataset.py", "src/openecon/frame.py",
             "src/openecon/analysis_contracts.py", "benchmarks/network.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=716031)
    parser.add_argument("--max-memory-mb", type=float, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1 or args.nodes < 2:
        parser.error("Use positive rows and at least two nodes.")
    root = Path(__file__).resolve().parents[1]
    before = fingerprint(root)
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    timings = {}
    with tempfile.TemporaryDirectory(prefix="openecon-network-benchmark-") as owned:
        path = Path(owned) / "edges.parquet"
        writer = None
        try:
            for start in range(0, args.rows, 65536):
                rows = min(65536, args.rows - start)
                sources = torch.randint(args.nodes, (rows,), generator=generator)
                targets = torch.randint(args.nodes, (rows,), generator=generator)
                hotspots = torch.randint(max(2, args.nodes // 100), (rows,), generator=generator)
                targets = torch.where(torch.rand(rows, generator=generator) < .35, hotspots, targets)
                weights = .1 + 3 * torch.rand(rows, generator=generator, dtype=torch.float64)
                table = pa.Table.from_pandas(pd.DataFrame({"source": sources.numpy(), "target": targets.numpy(),
                                                           "weight": weights.numpy()}), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table, row_group_size=rows)
        finally:
            if writer is not None:
                writer.close()
        data = scan(path)
        start = time.perf_counter()
        graph = network(data=data, weight="weight", nodes=range(args.nodes), max_memory_mb=args.max_memory_mb)
        timings["import_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        degree = graph.degree()
        timings["degree_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        rank = graph.pagerank()
        timings["pagerank_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        components = graph.components()
        timings["components_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        plot = graph.to_plot_data()
        timings["plot_selection_seconds"] = time.perf_counter() - start
        assert data.row_count == graph.metadata["input_rows"] == args.rows
        assert graph.node_count == len(degree) == len(rank) == len(components) == args.nodes
        assert rank.attrs["converged"] and abs(float(rank.pagerank.sum()) - 1.) < 1e-12
        assert graph.metadata["actual_peak_batch_rows"] <= graph.metadata["effective_batch_rows"]
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"status": "passed", "source_input_sha256_before": before,
                  "source_input_sha256_after": fingerprint(root), "seed": args.seed, "threads": 2,
                  "precision": "float64", "device": "cpu", "physical_parquet_rows": args.rows,
                  "parquet_bytes": path.stat().st_size, "graph": graph.metadata, "timings": timings,
                  "pagerank_iterations": rank.attrs["iterations"], "pagerank_l1_change": rank.attrs["l1_change"],
                  "pagerank_error_bound_estimate": rank.attrs["error_bound_estimate"],
                  "component_count": int(components.component.nunique()),
                  "shown_node_count": plot["shown_node_count"], "shown_edge_count": plot["shown_edge_count"],
                  "peak_process_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                  "timing_scope": "Imports, Parquet creation and cleanup excluded; each listed complete operation measured separately.",
                  "memory_scope": "Process peak RSS includes imported Python/Torch, input creation and all phases; graph buffer budget is distinct.",
                  "scope": "One undirected positive-weight graph; not all networks/devices or a universal runtime guarantee."}
        assert result["source_input_sha256_after"] == before, "Source changed during measurement."
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "rows": args.rows, "nodes": args.nodes,
                          "edges": graph.edge_count, "timings": timings,
                          "peak_process_rss_bytes": result["peak_process_rss_bytes"]}))


if __name__ == "__main__":
    main()
