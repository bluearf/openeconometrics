"""Physical sparse fixtures for triad census, global cut and bivariate QAP.

One measured call per method; fixture creation, batched imports and verification
are timed separately. CPU process RSS includes Python/Torch and every phase.
Sizes describe these fixtures, not maximum supported sizes or a speed promise.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import tempfile

import torch

import openecon as oe
from network_workflow_scale import cpu_name, peak_rss, write_records


SOURCE_PATHS = (
    "src/openecon/networks.py", "src/openecon/_network_sparse.py",
    "src/openecon/_network_motifs.py", "src/openecon/_network_cut.py",
    "src/openecon/_network_qap.py", "benchmarks/network_inference_scale.py",
    "benchmarks/network_workflow_scale.py",
)


def bands(nodes, degree, *, altered=False):
    for node in range(nodes):
        for step in range(1, degree + 1):
            distance = step + (altered and step == degree)
            yield node, (node + distance) % nodes, 1.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triad-nodes", type=int, default=100_000)
    parser.add_argument("--triad-degree", type=int, default=10)
    parser.add_argument("--cut-nodes", type=int, default=500)
    parser.add_argument("--qap-nodes", type=int, default=10_000)
    parser.add_argument("--qap-degree", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=39)
    parser.add_argument("--max-work", type=int, default=500_000_000)
    parser.add_argument("--max-memory-mb", type=float, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.triad_degree < 1 or args.triad_nodes <= 3 * args.triad_degree
            or args.cut_nodes < 9 or args.qap_degree < 1
            or args.qap_nodes <= 3 * (args.qap_degree + 1)
            or not 1 <= args.permutations <= 1_000_000 or args.max_work < 1
            or not math.isfinite(args.max_memory_mb) or args.max_memory_mb <= 0):
        parser.error("Use sparse positive band degrees and node counts larger than three band widths.")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    fingerprints = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in SOURCE_PATHS}
    phases = []
    receipts = {}

    def timed(name, function):
        start = time.perf_counter()
        result = function()
        phases.append({"name": name, "seconds": time.perf_counter() - start})
        return result

    def import_graph(path, directed):
        return oe.network(oe.scan(path), directed=directed, batch_rows=4096,
                          max_memory_mb=args.max_memory_mb)

    with tempfile.TemporaryDirectory(prefix="openecon-network-inference-") as owned:
        temporary = Path(owned)
        triads_path = temporary / "triads.parquet"
        physical = timed("create_triad_fixture", lambda: write_records(
            triads_path, bands(args.triad_nodes, args.triad_degree), weighted=False))
        graph = timed("import_triad_graph", lambda: import_graph(triads_path, True))
        census = timed("triad_census", lambda graph=graph: graph.triad_census(max_work=args.max_work))
        lookup = dict(zip(census.triad, census["count"]))
        assert sum(lookup.values()) == math.comb(args.triad_nodes, 3)
        assert lookup["030T"] == args.triad_nodes * math.comb(args.triad_degree, 2)
        assert lookup["021C"] == args.triad_nodes * args.triad_degree * (args.triad_degree + 1) // 2
        assert lookup["030C"] == 0
        receipts["triads"] = dict(physical=physical, nodes=graph.node_count,
            edges=graph.edge_count, exact_counts=lookup, work_used=census.attrs["work_used"],
            verification="analytical directed band counts for transitive triples/open paths; complete census sum")
        del graph, census
        gc.collect()

        cut_path = temporary / "cut.parquet"
        physical = timed("create_cut_fixture", lambda: write_records(
            cut_path, bands(args.cut_nodes, 4), weighted=False))
        graph = timed("import_cut_graph", lambda: import_graph(cut_path, False))
        result = timed("global_min_cut", lambda graph=graph: graph.global_min_cut(max_work=args.max_work))
        assert result["value"] == 8
        assert len(result["cut_edges"]) == 8
        assert len(result["source_partition"]) + len(result["target_partition"]) == args.cut_nodes
        receipts["global_cut"] = dict(physical=physical, nodes=graph.node_count,
            edges=graph.edge_count, value=result["value"], metadata=result["metadata"],
            partition_sizes=[len(result["source_partition"]), len(result["target_partition"])],
            verification="8-regular ring band, returned crossing edges and complete nontrivial partition")
        del graph, result
        gc.collect()

        paths = [temporary / "qap-a.parquet", temporary / "qap-b.parquet"]
        physical = [timed(f"create_qap_fixture_{i}", lambda i=i: write_records(
            paths[i], bands(args.qap_nodes, args.qap_degree, altered=bool(i)), weighted=False))
            for i in range(2)]
        graphs = [timed(f"import_qap_graph_{i}", lambda i=i: import_graph(paths[i], True))
                  for i in range(2)]
        result = timed("qap_correlation", lambda: graphs[0].qap_correlation(
            graphs[1], values="binary", permutations=args.permutations, seed=42,
            max_work=args.max_work))
        density = args.qap_degree / (args.qap_nodes - 1)
        expected = ((args.qap_degree - 1) / args.qap_degree - density) / (1 - density)
        assert abs(float(result.correlation.iloc[0]) - expected) < 1e-12
        assert float(result.pvalue.iloc[0]) == (1 + int(result.extreme_permutations.iloc[0])) / (args.permutations + 1)
        receipts["qap"] = dict(physical=physical, nodes=graphs[0].node_count,
            edges_per_snapshot=[g.edge_count for g in graphs],
            result=result.to_dict(orient="records"), metadata=result.attrs,
            independent_binary_correlation=expected,
            verification="analytical overlap/density Pearson correlation and plus-one Monte Carlo pvalue")
    assert fingerprints == {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in SOURCE_PATHS}, "Sources changed during benchmark."
    report = dict(status="passed", recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=cpu_name(), torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(), max_memory_mb_per_graph=args.max_memory_mb,
        max_work=args.max_work, warmups=0, measured_repetitions=1,
        timing_scope="separate fixture/import/method calls; library startup excluded",
        rss_scope="whole Python process, libraries, fixture creation, imports and all operations",
        process_peak_rss_bytes=peak_rss(), source_sha256=fingerprints,
        phases=phases, fixtures=receipts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "phases": phases,
                      "process_peak_rss_mib": report["process_peak_rss_bytes"] / 1024**2}))


if __name__ == "__main__":
    main()
