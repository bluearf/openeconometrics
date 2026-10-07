"""Fresh-process exact disk-kernel RSS and OS I/O proof on owned synthetic data."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def disk_io():
    """Kernel process disk-byte counters; cached reads may report zero bytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(block_input_operations=int(usage.ru_inblock), block_output_operations=int(usage.ru_oublock))
    if sys.platform == "darwin":
        # sys/resource.h rusage_info_v2: UUID, 16 uint64 counters, disk bytes.
        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16), ("counters", ctypes.c_uint64 * 18)]
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        library.proc_pid_rusage.restype = ctypes.c_int
        value = Usage()
        if library.proc_pid_rusage(os.getpid(), 2, ctypes.byref(value)) == 0:
            result.update(api="proc_pid_rusage/RUSAGE_INFO_V2", read_bytes=int(value.counters[16]),
                          write_bytes=int(value.counters[17]), current_resident_bytes=int(value.counters[6]))
        else:
            result.update(api="getrusage/block-operations", read_bytes=None, write_bytes=None)
    elif sys.platform.startswith("linux"):
        values = dict(line.split(": ", 1) for line in Path("/proc/self/io").read_text().splitlines())
        result.update(api="/proc/self/io", read_bytes=int(values["read_bytes"]), write_bytes=int(values["write_bytes"]))
    else:
        result.update(api="getrusage/block-operations", read_bytes=None, write_bytes=None)
    return result


def io_difference(before, after):
    return {"api": after["api"], **{name: (after[name] - before[name] if after[name] is not None else None)
            for name in ("read_bytes", "write_bytes", "block_input_operations", "block_output_operations")}}


def worker(rows, directory):
    import sqlite3
    import torch
    import openecon as oe
    from openecon.analysis_contracts import AnalysisError

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "src/openecon/_network_disk_algorithms.py", "src/openecon/_network_store.py",
        "src/openecon/networks.py", "scripts/benchmark_network_disk_algorithms.py")}
    n, offsets = 1000, rows // 1000
    assert rows % n == 0 and 2 <= offsets <= n
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "circulant.csv"
    with source.open("x") as file:
        file.write("source,target,count\n")
        for node in range(n):
            for offset in range(offsets):
                file.write(f"{node},{(node + offset) % n},2\n")
    baseline, io_before = rss_bytes(), disk_io()
    start = time.perf_counter()
    graph = oe.network(oe.scan(source), weight="count", directed=True,
        store=directory / "graph.sqlite", max_memory_mb=8, batch_rows=64)
    build_seconds = time.perf_counter() - start
    build_io = io_difference(io_before, disk_io())
    assert graph.node_count == n and graph.edge_count == rows
    graph = oe.open_network(graph.path, max_memory_mb=8)
    try:
        graph.materialize()
    except AnalysisError as exc:
        assert exc.code == "network_memory_budget"
    else:
        raise AssertionError("This graph must exceed the explicit resident import allowance")
    methods = {}
    for name in ("degree", "pagerank", "components"):
        before_peak, io_before = rss_bytes(), disk_io()
        start = time.perf_counter()
        result = getattr(graph, name)(batch_rows=8192)
        elapsed = time.perf_counter() - start
        io_after = disk_io()
        if name == "degree":
            assert result.in_degree.tolist() == result.out_degree.tolist() == [offsets] * n
            assert result.in_strength.tolist() == result.out_strength.tolist() == [2. * offsets] * n
        elif name == "pagerank":
            assert float((result.pagerank - 1 / n).abs().max()) < 1e-12
            assert abs(float(result.pagerank.sum()) - 1) < 1e-12
        else:
            assert result.component.tolist() == [0] * n
        methods[name] = dict(seconds=elapsed, rss_peak_bytes=rss_bytes(),
            rss_new_high_water_bytes=rss_bytes() - before_peak,
            current_resident_bytes=io_after.get("current_resident_bytes"),
            os_disk_io=io_difference(io_before, io_after), metadata=result.attrs,
            exact_closed_form_validated=True)
        del result
    return dict(nodes=n, edges=rows, offsets=offsets, source_bytes=source.stat().st_size,
        disk_bytes=graph.path.stat().st_size, rss_baseline_bytes=baseline, rss_peak_bytes=rss_bytes(),
        rss_increment_bytes=rss_bytes() - baseline, build_seconds=build_seconds, build_os_disk_io=build_io,
        estimated_resident_import_bytes=graph.resident_import_bound, resident_budget_guarded=True,
        methods=methods, source_sha256=hashes,
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
        print(json.dumps({"edges": rows, "rss_peak_bytes": run["rss_peak_bytes"],
            "methods_seconds": {name: value["seconds"] for name, value in run["methods"].items()}}), flush=True)
    receipt = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
        scope="separate fresh processes, 1000-node directed weighted circulants with loops; 8 MiB declared graph workspace; 64-row import and at most 8192-row analysis scans; CPU one Torch thread",
        platform=platform.platform(), python=platform.python_version(), runs=runs,
        limitations="RSS includes Python/Torch, parser and allocator caches and is not capped at 8 MiB. Resident admission is a conservative bound, not a measured resident allocation. OS disk bytes are kernel process counters; cached reads can be zero and are distinct from logical edge bytes. No cache flush or user-file access was performed. Uniform PageRank converges in one iteration; small asymmetric/dangling fixtures validate nonuniform iteration separately. Shared-machine timings do not certify arbitrary-topology performance or unsupported methods.")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
