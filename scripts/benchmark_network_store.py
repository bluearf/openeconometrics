"""Separate-process disk graph RSS/size proof; never reads user datasets."""
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


def worker(rows, directory):
    import torch
    import sqlite3
    import openecon as oe
    from openecon._network_store import DiskNetwork
    from openecon.analysis_contracts import AnalysisError

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "src/openecon/_network_store.py", "src/openecon/networks.py", "scripts/benchmark_network_store.py")}
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "chain.csv"
    with source.open("x") as file:
        file.write("source,target,count\n")
        for node in range(rows):
            file.write(f"{node},{node+1},2\n")
    baseline = rss_bytes()
    start = time.perf_counter()
    graph = oe.network(oe.scan(source), weight="count", directed=True,
        store=directory / "graph.sqlite", max_memory_mb=8, batch_rows=64)
    build_seconds = time.perf_counter() - start
    assert isinstance(graph, DiskNetwork) and graph.node_count == rows + 1 and graph.edge_count == rows
    reopen_start = time.perf_counter()
    reopened = oe.open_network(graph.path, max_memory_mb=8)
    assert next(reopened.neighbors(rows - 1)) == [{"node": rows, "weight": 2.0}]
    assert next(reopened.neighbors(1, incoming=True)) == [{"node": 0, "weight": 2.0}]
    reopen_seconds = time.perf_counter() - reopen_start
    guarded = False
    try:
        reopened.materialize()
    except AnalysisError as exc:
        guarded = exc.code == "network_memory_budget"
    assert guarded
    return dict(rows=rows, nodes=graph.node_count, edges=graph.edge_count,
        source_bytes=source.stat().st_size, disk_bytes=graph.path.stat().st_size,
        rss_baseline_bytes=baseline, rss_peak_bytes=rss_bytes(),
        rss_increment_bytes=rss_bytes() - baseline, build_seconds=build_seconds,
        reopen_and_adjacency_seconds=reopen_seconds,
        estimated_resident_import_bytes=graph.resident_import_bound,
        resident_budget_guarded=guarded, metadata=graph.metadata, source_sha256=hashes,
        environment=dict(torch=torch.__version__, sqlite=sqlite3.sqlite_version,
                         torch_threads=1, device="cpu", machine=platform.machine()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[50_000, 250_000, 1_000_000])
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(worker(args.worker, args.output_dir), allow_nan=False))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runs = []
    for rows in args.rows:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
            "--worker", str(rows), "--output-dir", str(args.output_dir / str(rows))],
            capture_output=True, text=True, timeout=300, check=True)
        run = json.loads(process.stdout)
        runs.append(run)
        print(json.dumps({key: run[key] for key in ("rows", "disk_bytes", "rss_peak_bytes", "rss_increment_bytes", "build_seconds")}), flush=True)
    receipt = dict(scope="synthetic directed weighted chain; separate fresh processes; CPU, one Torch thread; 8 MiB declared graph workspace, 64-row batches; RSS includes runtime and CSV parsing",
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), python=platform.python_version(), runs=runs,
        limitations="Not a whole-process memory cap, arbitrary-topology guarantee or all-method disk-native analysis proof. SQLite DB quota reserves journal overhead; observed disk_bytes is the completed DB. Wall timings were observed on a shared development machine, not an isolated CPU benchmark.")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
