"""Measured sparse-network workloads, not a universal scale guarantee.

The large directed fixture is imported twice, sequentially: aggregate weights
are strengths for eigenvector centrality, whereas sampled Brandes uses unique
edge hop distances. One immutable physical Parquet file backs both snapshots.
The smaller undirected planted fixture exercises real Louvain and Leiden.
Imports and fixture creation are excluded from each complete API operation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import openecon as oe
from openecon.dataset import scan


SOURCE_PATHS = (
    "src/openecon/networks.py",
    "src/openecon/_network_sparse.py",
    "src/openecon/_network_centrality.py",
    "src/openecon/_network_topology.py",
    "src/openecon/_network_communities.py",
    "benchmarks/network_advanced.py",
)


def fingerprint(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def cpu_name():
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor()


def write_fixture(path, rows, nodes, seed, *, planted=False, group_size=50):
    """Create only bounded Arrow batches; never collect the physical source."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    writer = None
    try:
        for start in range(0, rows, 65_536):
            count = min(65_536, rows - start)
            sources = torch.randint(nodes, (count,), generator=generator)
            targets = torch.randint(nodes, (count,), generator=generator)
            if planted:
                own_group = torch.div(sources, group_size, rounding_mode="floor")
                local = own_group * group_size + torch.randint(group_size, (count,), generator=generator)
                targets = torch.where(torch.rand(count, generator=generator) < .90, local, targets)
                # A within-block ring ensures each planted node has an edge.
                physical = torch.arange(start, start + count)
                ring = physical < nodes
                sources[ring] = physical[ring]
                targets[ring] = (physical[ring] // group_size * group_size
                                + (physical[ring] + 1) % group_size)
            else:
                hotspots = torch.randint(max(2, nodes // 100), (count,), generator=generator)
                targets = torch.where(torch.rand(count, generator=generator) < .35, hotspots, targets)
            weights = .1 + 3 * torch.rand(count, generator=generator, dtype=torch.float64)
            table = pa.Table.from_pandas(pd.DataFrame({"source": sources.numpy(), "target": targets.numpy(),
                                                      "weight": weights.numpy()}), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table, row_group_size=count)
    finally:
        if writer is not None:
            writer.close()
    metadata = pq.ParquetFile(path).metadata
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size, "physical_rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "largest_row_group": max(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)),
            "seed": seed, "node_labels": "integer range [0, nodes)", "nodes": nodes,
            "topology": ("90% within 50-node planted blocks, 10% uniform; first N edges are within-block rings"
                         if planted else "uniform sources; targets 35% hotspot and 65% uniform"),
            "weights": "independent uniform [.1, 3.1), native float64"}


def numeric_summary(frame, column, *, nonnegative=True, upper=None):
    values = torch.as_tensor(frame[column].to_numpy(), dtype=torch.float64, device="cpu")
    assert bool(torch.isfinite(values).all()), f"Nonfinite {column}."
    if nonnegative:
        assert bool((values >= 0).all()), f"Negative {column}."
    if upper is not None:
        assert bool((values <= upper).all()), f"{column} exceeds {upper}."
    return {"rows": len(frame), "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "l2_norm": float(torch.linalg.vector_norm(values)),
            "metadata": frame.attrs}


def graph_summary(graph):
    return {"node_count": graph.node_count, "edge_count": graph.edge_count, "metadata": graph.metadata,
            "planned_lifetime_owned_peak_bytes": graph._budget.peak,
            "memory_limit_bytes": graph._budget.limit}


def measure(phases, name, options, callback, summarize):
    """Record a real failure instead of substituting a smaller result."""
    start = time.perf_counter()
    try:
        value = callback()
        seconds = time.perf_counter() - start
        record = {"status": "passed", "seconds": seconds, "options": options,
                  "result": summarize(value), "peak_process_rss_bytes_after": peak_rss()}
    except Exception as error:
        record = {"status": "failed", "seconds": time.perf_counter() - start, "options": options,
                  "error_type": type(error).__name__, "error_code": getattr(error, "code", None),
                  "error_message": str(error), "peak_process_rss_bytes_after": peak_rss()}
        value = None
    phases[name] = record
    print(json.dumps({"phase": name, "status": record["status"], "seconds": record["seconds"],
                      "error_code": record.get("error_code")}), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument("--community-rows", type=int, default=40_000)
    parser.add_argument("--community-nodes", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=716032)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-memory-mb", type=float, default=1024)
    parser.add_argument("--max-work", type=int, default=500_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.rows < 1 or args.nodes < 2 or not 1 <= args.samples <= args.nodes
            or args.community_nodes < 50 or args.community_nodes % 50
            or args.community_rows < args.community_nodes or args.max_work < 1
            or not math.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0):
        parser.error("Use valid positive dimensions, samples<=nodes, and community nodes divisible by 50.")
    if args.output.exists():
        parser.error("The output already exists; use a new owned receipt path.")
    root = Path(__file__).resolve().parents[1]
    before = fingerprint(root)
    torch.set_num_threads(2)
    phases, fixtures, graphs = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="openecon-network-advanced-") as owned:
        large_path, planted_path = Path(owned) / "large.parquet", Path(owned) / "planted.parquet"
        start = time.perf_counter()
        fixtures["large"] = write_fixture(large_path, args.rows, args.nodes, args.seed)
        fixtures["planted"] = write_fixture(planted_path, args.community_rows, args.community_nodes,
                                             args.seed + 1, planted=True)
        creation_seconds = time.perf_counter() - start
        data = scan(large_path)
        constructor = {"directed": True, "weight": "weight", "batch_rows": 65_536,
                       "max_memory_mb": args.max_memory_mb, "nodes": args.nodes}
        graph = measure(phases, "weighted_directed_import", constructor,
                        lambda data=data: oe.network(data, weight="weight", directed=True, nodes=range(args.nodes),
                                           max_memory_mb=args.max_memory_mb), graph_summary)
        if graph is not None:
            assert graph.node_count == args.nodes and graph.metadata["input_rows"] == args.rows
            result = measure(phases, "strong_components", {"connectivity": "strong"},
                             lambda graph=graph: graph.components(connectivity="strong"),
                             lambda value: {"rows": len(value), "component_count": int(value.component.nunique()),
                                            "metadata": value.attrs})
            if result is not None:
                assert len(result) == args.nodes
            del result
            result = measure(phases, "core_numbers", {}, graph.core_numbers,
                             lambda value: numeric_summary(value, "core_number"))
            del result
            result = measure(phases, "weighted_eigenvector", {"max_iter": 5000, "tol": 1e-9},
                             lambda graph=graph: graph.eigenvector(max_iter=5000, tol=1e-9),
                             lambda value: numeric_summary(value, "eigenvector"))
            if result is not None:
                assert len(result) == args.nodes and result.attrs["converged"]
                assert result.attrs["relative_eigen_residual"] <= 1e-9
                assert abs(phases["weighted_eigenvector"]["result"]["l2_norm"] - 1) < 1e-12
            del result
            result = measure(phases, "binary_directed_clustering", {"max_work": args.max_work},
                             lambda graph=graph: graph.clustering(max_work=args.max_work),
                             lambda value: numeric_summary(value, "clustering", upper=1 + 1e-12))
            if result is not None:
                assert len(result) == args.nodes and result.attrs["actual_work"] <= args.max_work
            del result
            graphs["weighted_directed"] = graph_summary(graph)
        del graph
        gc.collect()
        unweighted = {**constructor, "weight": None}
        graph = measure(phases, "unweighted_directed_reimport", unweighted,
                        lambda data=data: oe.network(data, directed=True, nodes=range(args.nodes),
                                           max_memory_mb=args.max_memory_mb), graph_summary)
        if graph is not None:
            result = measure(phases, "sampled_unweighted_betweenness",
                             {"samples": args.samples, "normalized": True, "endpoints": False,
                              "seed": args.seed, "max_work": args.max_work},
                             lambda graph=graph: graph.betweenness(samples=args.samples, seed=args.seed,
                                                        max_work=args.max_work),
                             lambda value: numeric_summary(value, "betweenness"))
            if result is not None:
                assert len(result) == args.nodes and result.attrs["samples"] == args.samples
                assert result.attrs["shortest_paths"] == "BFS"
                assert result.attrs["sampled"] == (args.samples < args.nodes)
            del result
            graphs["unweighted_directed"] = graph_summary(graph)
        del graph, data
        gc.collect()
        small_data = scan(planted_path)
        graph = measure(phases, "planted_undirected_import",
                        {"directed": False, "weight": "weight", "nodes": args.community_nodes,
                         "max_memory_mb": args.max_memory_mb},
                        lambda small_data=small_data: oe.network(small_data, weight="weight", nodes=range(args.community_nodes),
                                           max_memory_mb=args.max_memory_mb), graph_summary)
        if graph is not None:
            planted_q = graph.modularity({node: node // 50 for node in range(args.community_nodes)},
                                         max_work=args.max_work)
            graphs["planted_reference"] = {"group_size": 50, "group_count": args.community_nodes // 50,
                                           "modularity": planted_q,
                                           "scope": "Generator labels are a comparison, not a global optimum."}
            for method in ("louvain", "leiden"):
                result = measure(phases, "planted_" + method,
                                 {"method": method, "resolution": 1, "seed": args.seed, "max_iter": 100,
                                  "tol": 1e-10, "max_work": args.max_work},
                                 lambda method=method, graph=graph: graph.communities(method=method, seed=args.seed,
                                                                         max_work=args.max_work),
                                 lambda value: {"rows": len(value), "communities": int(value.community.nunique()),
                                                "metadata": value.attrs})
                if result is not None:
                    assert len(result) == args.community_nodes and result.attrs["converged"]
                    assert result.attrs["work_used"] <= args.max_work
                    start = time.perf_counter()
                    exact_q = graph.modularity(result, max_work=args.max_work)
                    phases["planted_" + method]["objective_readback_seconds"] = time.perf_counter() - start
                    phases["planted_" + method]["objective_readback"] = exact_q
                    assert abs(exact_q - result.attrs["modularity"]) < 1e-12
                del result
            graphs["planted_undirected"] = graph_summary(graph)
        del graph, small_data
        gc.collect()
        # Both analyses used the same unchanged physical large source.
        for name, path in (("large", large_path), ("planted", planted_path)):
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == fixtures[name]["sha256"]
    after = fingerprint(root)
    stable = before == after
    result = {"status": "passed" if stable and all(p["status"] == "passed" for p in phases.values()) else "failed",
              "source_sha256_before": before, "source_sha256_after": after, "source_bytes_unchanged": stable,
              "hardware": {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_name()},
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "pandas": pd.__version__, "pyarrow": pa.__version__, "openecon": oe.__version__},
              "torch_threads": torch.get_num_threads(), "device": "cpu", "precision": "float64",
              "fixtures": fixtures, "fixture_creation_seconds_excluded": creation_seconds,
              "graphs": graphs, "phases": phases, "peak_process_rss_bytes": peak_rss(),
              "temporary_sources_removed": True,
              "timing_scope": "Each full public API call measured; imports, Parquet creation, summary checks, "
                              "explicit objective readbacks and cleanup excluded from operation seconds.",
              "memory_scope": "RSS is the process lifetime peak, including Python/Torch, creation and all phases. "
                              "max_memory_mb controls conservative owned snapshot/workspace plans, not total RSS.",
              "scope": "One directed sparse fixture for SCC, weak cores, incoming weighted eigenvector and binary "
                       "Fagiolo clustering; sampled unit-hop Brandes on the same physical source; one planted "
                       "undirected weighted fixture for Louvain/Leiden. No weighted Brandes/all-pairs, GPU, "
                       "global optimality or universal runtime claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "peak_process_rss_bytes": result["peak_process_rss_bytes"],
                      "source_bytes_unchanged": stable}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
