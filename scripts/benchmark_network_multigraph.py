"""Fresh-process typed parallel-edge preservation, degree and XML scale proof."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def fingerprint(graph):
    digest = hashlib.sha256()
    for record in graph._records():
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode() + b"\n")
    return digest.hexdigest()


def worker(edges, directory):
    import torch
    import openecon as oe
    torch.set_num_threads(1)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "src/openecon/_network_multi.py", "src/openecon/_network_io.py", "scripts/benchmark_network_multigraph.py")}
    assert edges % 10 == 0
    baseline = rss_bytes()
    start = time.perf_counter()
    graph = oe.multigraph((dict(edge_id=f"e:{i}" if i % 2 == 0 else i,
        source=1, target=1 if i % 10 == 0 else "1", weight=0. if i % 7 == 0 else 1. + (i % 3) / 2)
        for i in range(edges)), weight="weight", nodes=["isolated"], max_memory_mb=512,
        edge_attributes={"e:0": {"relation": "recorded zero loop"}, 1: {"relation": "parallel trade"}})
    build_seconds = time.perf_counter() - start
    original_hash = fingerprint(graph)
    start = time.perf_counter()
    degree = graph.degree()
    degree_seconds = time.perf_counter() - start
    expected = {"isolated": 0, 1: edges + edges // 10, "1": edges - edges // 10}
    assert degree.set_index("node").degree.to_dict() == expected
    start = time.perf_counter()
    projected = graph.to_network(reducer="count", attributes="drop", max_work=250_000_000)
    projection_seconds = time.perf_counter() - start
    assert projected.edge_count == 2
    assert projected.degree().set_index("node").strength.to_dict() == expected
    formats = {}
    for format in ("graphml", "gexf"):
        path = directory / ("owned." + format)
        start = time.perf_counter()
        graph.write(path)
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        reopened = oe.read_multigraph(path, max_memory_mb=512)
        read_seconds = time.perf_counter() - start
        assert reopened.edge_count == edges and fingerprint(reopened) == original_hash
        formats[format] = dict(file_bytes=path.stat().st_size, write_seconds=write_seconds,
            read_seconds=read_seconds, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            exact_ordered_record_fingerprint=original_hash, metadata=reopened.metadata,
            rss_peak_bytes=rss_bytes())
        del reopened
    assert fingerprint(graph) == original_hash
    return dict(edges=edges, nodes=graph.node_count, loops=edges // 10, zero_edges_retained=(edges - 1) // 7 + 1,
        build_seconds=build_seconds, degree_seconds=degree_seconds, projection_seconds=projection_seconds,
        metadata=graph.metadata, projection=projected.metadata["multigraph_projection"], formats=formats,
        rss_baseline_bytes=baseline, rss_peak_bytes=rss_bytes(), rss_increment_bytes=rss_bytes() - baseline,
        analytic_degree_and_count_projection_verified=True, source_unchanged=True, source_sha256=hashes,
        environment=dict(torch=torch.__version__, threads=1, device="cpu", machine=platform.machine()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edges", type=int, nargs="+", default=[25_000, 100_000])
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker, args.output_dir), allow_nan=False))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    for edges in args.edges:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(edges),
            "--output-dir", str(args.output_dir / str(edges))], capture_output=True, text=True, timeout=300, check=True)
        run = json.loads(process.stdout)
        runs.append(run)
        print(json.dumps({"edges": edges, "rss_peak_bytes": run["rss_peak_bytes"],
            "xml_seconds": {k: v["read_seconds"] for k, v in run["formats"].items()}}), flush=True)
    receipt = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=platform.python_version(), runs=runs,
        scope="owned 3-node undirected multigraphs: separate typed IDs, many parallel edges, zero weights, loops and conflicting sample attributes; fresh processes; CPU one Torch thread; declared 512 MiB per-operation owned workspace",
        limitations="Resident-only storage; no out-of-core claim. RSS includes runtime, parser, caller-held original graph and allocator caches and is not capped by the per-graph workspace budget. Exact fingerprints include typed IDs, orientation, weights, scalar attributes and order; two XML formats are checked. Pajek is explicitly unsupported. Shared-machine timings are observations, not arbitrary-topology performance guarantees.")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
