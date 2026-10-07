"""Fresh-process interval membership and exact ordered dynamic GEXF scale proof."""
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
    for value in (graph.interval, graph.defaults, dict(timeformat=graph.timeformat,
            version=graph.version, directed=graph.directed, weighted=graph.weighted)):
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
    for scope in ("node", "edge"):
        for record in graph._records(scope):
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
        "src/openecon/_network_dynamic.py", "src/openecon/_network_dynamic_io.py",
        "src/openecon/_network_multi.py", "scripts/benchmark_network_dynamic.py")}
    assert edges % 10 == 0
    baseline = rss_bytes()
    start = time.perf_counter()
    graph = oe.dynamic_network([dict(node=1), dict(node="1"), dict(node="isolated")],
        (dict(edge_id=f"e:{i}" if i % 2 == 0 else i, source=1 if i % 2 == 0 else "1",
            target=1 if i % 10 == 0 or i % 2 else "1", weight=0. if i % 7 == 0 else 1.,
            spells=[dict(start=i % 5, end=i % 5 + 2)],
            dynamic_attributes={"phase": [dict(value="early", end=2), dict(value="late", startopen=2)]})
         for i in range(edges)), version="1.2draft", interval=dict(start=0, end=10),
        max_memory_mb=2048, max_work=100_000_000)
    build_seconds = time.perf_counter() - start
    original_hash = fingerprint(graph)
    expected_ids = [f"e:{i}" if i % 2 == 0 else i for i in range(edges) if i % 5 <= 2]
    start = time.perf_counter()
    point = graph.at(2)
    assert point.edges().edge_id.tolist() == expected_ids
    expected = {1: 0, "1": 0, "isolated": 0}
    for i in range(edges):
        if i % 5 <= 2:
            u, v = (1 if i % 2 == 0 else "1"), (1 if i % 10 == 0 or i % 2 else "1")
            expected[u] += 1
            expected[v] += 1
    assert point.degree().set_index("node").degree.to_dict() == expected
    assert all(attrs["phase"] == "early" for attrs in point.edge_attributes.values())
    point_seconds = time.perf_counter() - start
    del point
    start = time.perf_counter()
    window = graph.window(1, 3, selection="cover", attributes="drop")
    assert window.edges().edge_id.tolist() == [f"e:{i}" if i % 2 == 0 else i for i in range(edges) if i % 5 == 1]
    assert len(window.metadata["dynamic_selection"]["dropped_dynamic_attributes"]) == edges // 5
    cover_seconds = time.perf_counter() - start
    del window
    path = directory / "owned.gexf"
    start = time.perf_counter()
    graph.write(path)
    write_seconds = time.perf_counter() - start
    start = time.perf_counter()
    reopened = oe.read_dynamic_network(path, max_memory_mb=2048, max_work=100_000_000)
    read_seconds = time.perf_counter() - start
    assert reopened.edge_count == edges and fingerprint(reopened) == original_hash
    assert fingerprint(graph) == original_hash
    with path.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(edges=edges, nodes=graph.node_count, point_edges=len(expected_ids), cover_edges=edges // 5,
        build_seconds=build_seconds, point_seconds=point_seconds, cover_seconds=cover_seconds,
        write_seconds=write_seconds, read_seconds=read_seconds, file_bytes=path.stat().st_size,
        file_sha256=file_hash, exact_ordered_record_fingerprint=original_hash,
        metadata=graph.metadata, reopened_metadata=reopened.metadata,
        rss_baseline_bytes=baseline, rss_peak_bytes=rss_bytes(), rss_increment_bytes=rss_bytes() - baseline,
        independent_membership_and_degree_verified=True, source_unchanged=True,
        source_sha256=hashes, environment=dict(torch=torch.__version__, threads=1, device="cpu", machine=platform.machine()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
            "--output-dir", str(args.output_dir / str(edges))], capture_output=True, text=True, timeout=600)
        if process.returncode:
            raise RuntimeError(f"Benchmark worker failed: {process.stderr[-4000:]}")
        run = json.loads(process.stdout)
        runs.append(run)
        print(json.dumps({"edges": edges, "rss_peak_bytes": run["rss_peak_bytes"],
            "point_seconds": run["point_seconds"], "read_seconds": run["read_seconds"]}), flush=True)
    receipt = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=platform.python_version(), runs=runs,
        scope="3-node undirected resident graphs, separate typed IDs, raw orientations, many parallel edges, loops/zeros, one spell and two timed values per edge; graph lifetime; independent point/cover oracle; exact full ordered GEXF fingerprints; fresh CPU processes, one Torch thread, 2048 MiB planned owned workspace",
        limitations="No out-of-core or arbitrary-topology performance claim. RSS includes runtime, simultaneous source/reopened graphs and allocator caches; the declared budget limits planned owned buffers, not whole-process RSS. Timings are local shared-machine observations.")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
