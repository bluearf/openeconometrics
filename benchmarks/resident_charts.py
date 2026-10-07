"""Measure resident chart workspace on owned physical Parquet inputs.

Each size/input combination runs in a fresh process. Input creation, resident
loading and dependency warmup are excluded from chart measurements. Sampled
RSS and exact traced Python peaks are distinct from lifetime process RSS.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc


ROOT = Path(__file__).resolve().parents[1]


def current_rss():
    if sys.platform == "darwin":
        # macOS SDK mach/task_info.h, MACH_TASK_BASIC_INFO (20), 12 natural_t.
        class Info(ctypes.Structure):
            _fields_ = [("virtual", ctypes.c_uint64), ("resident", ctypes.c_uint64),
                ("maximum", ctypes.c_uint64), ("user_sec", ctypes.c_int32),
                ("user_usec", ctypes.c_int32), ("system_sec", ctypes.c_int32),
                ("system_usec", ctypes.c_int32), ("policy", ctypes.c_int32),
                ("suspended", ctypes.c_int32)]
        library = ctypes.CDLL(None)
        task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
        count, info = ctypes.c_uint32(ctypes.sizeof(Info) // 4), Info()
        library.task_info.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_uint32)]
        if library.task_info(task, 20, ctypes.byref(info), ctypes.byref(count)) != 0:
            raise RuntimeError("Cannot read owned process RSS")
        return info.resident
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    raise RuntimeError("This benchmark's RSS probe supports macOS/Linux")


def measure(operation):
    gc.collect()
    baseline = current_rss()
    peak, samples, stop = [baseline], [0], threading.Event()
    def monitor():
        while not stop.wait(.001):
            peak[0] = max(peak[0], current_rss())
            samples[0] += 1
    thread = threading.Thread(target=monitor, daemon=True)
    tracemalloc.start()
    thread.start()
    started = time.perf_counter()
    try:
        value = operation()
        seconds = time.perf_counter() - started
        peak[0] = max(peak[0], current_rss())
        python_peak = tracemalloc.get_traced_memory()[1]
    finally:
        stop.set()
        thread.join()
        tracemalloc.stop()
    return value, dict(seconds=seconds, rss_baseline_bytes=baseline,
        sampled_rss_peak_bytes=peak[0], sampled_rss_increase_bytes=peak[0] - baseline,
        python_traced_peak_bytes=python_peak, rss_poll_interval_seconds=.001,
        rss_samples=samples[0], instrumentation="RSS polling and tracemalloc enabled; timing includes both")


def worker(rows, kind):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import openecon_charts as charts
    from openecon_charts import resident, streaming

    torch.set_num_threads(2)
    # Warm dependencies before allocating the benchmark input.
    charts.hist(data={"x": np.array([0., 1.])}, x="x")
    charts.scatter(data={"x": np.array([0., 1.]), "y": np.array([0., 1.])}, x="x", y="y")
    modules = [charts.charts, resident, streaming]
    hashes = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for module in modules}
    with tempfile.TemporaryDirectory(prefix="openecon-resident-charts-") as directory:
        path = Path(directory) / "input.parquet"
        writer = None
        try:
            for start in range(0, rows, 65536):
                x = np.arange(start, min(rows, start + 65536), dtype="float64")
                table = pa.table({"x": x, "y": (x % 97) / 97, "unused": np.ones(len(x))})
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=65536)
        finally:
            if writer is not None:
                writer.close()
        del table, x
        source_hash = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        assert pq.ParquetFile(path).metadata.num_rows == rows
        frame = pd.read_parquet(path)
        data = frame if kind == "dataframe" else {name: frame[name].to_numpy(copy=False) for name in ("x", "y")}
        histogram, hist_memory = measure(lambda: charts.hist(data=data, x="x", bins=40))
        assert histogram.total_n == rows and histogram.dropped_n == 0
        expected = []
        for index, point in enumerate(histogram.data):
            lower = max(0, math.ceil(point["x0"]))
            upper = min(rows - 1, math.floor(point["x1"]) if index == 39 else math.ceil(point["x1"]) - 1)
            expected.append(max(0, upper - lower + 1))
        assert [point["count"] for point in histogram.data] == expected and sum(expected) == rows
        scatter, scatter_memory = measure(lambda: charts.scatter(data=data, x="x", y="y"))
        ranks = [round(index * (rows - 1) / 1999) for index in range(2000)]
        assert scatter.data == [{"x": float(rank), "y": (rank % 97) / 97} for rank in ranks]
        assert scatter.total_n == rows and scatter.sample_n == 2000 and scatter.dropped_n == 0
        assert scatter.config["processing"]["extents"] == {"x": [0., float(rows - 1)], "y": [0., 96 / 97]}
        def refused_line():
            try:
                charts.line(data=data, x="x", y="y")
            except ValueError as error:
                assert "10,000" in str(error)
                return "refused before copying"
            raise AssertionError("Oversized line returned success")
        refusal, line_memory = measure(refused_line)
        hist_memory["json_bytes"] = len(histogram.model_dump_json().encode())
        scatter_memory["json_bytes"] = len(scatter.model_dump_json().encode())
        assert hist_memory["json_bytes"] < 10000 and scatter_memory["json_bytes"] < 160000
        # Native buffers, transient Python conversions and output allocations
        # must remain bounded independently of the resident input size.
        for memory in (hist_memory, scatter_memory):
            assert memory["python_traced_peak_bytes"] < 32 * 1024**2
            assert memory["sampled_rss_increase_bytes"] < 64 * 1024**2
        assert source_hash == hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        assert hashes == {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in modules}
        result = dict(status="passed", rows=rows, input_kind=kind, physical_parquet=True,
            parquet_bytes=path.stat().st_size, source_sha256=source_hash,
            resident_frame_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
            source_module_sha256=hashes, hist=hist_memory, scatter=scatter_memory,
            line=dict(result=refusal, **line_memory),
            lifetime_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024),
            torch=str(torch.__version__), pandas=pd.__version__, python=platform.python_version(),
            platform=platform.platform(), threads=2, full_histogram=True, scatter_sample_n=2000,
            scope="Resident input and prior imports/creation are excluded from per-operation deltas. "
                  "RSS is sampled and may miss transients; tracemalloc covers Python allocations. "
                  "Lifetime RSS includes setup. No reader-allocation or out-of-core claim.")
    result["owned_source_removed"] = not path.exists()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--input-kind", choices=("dataframe", "mapping"), default="dataframe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 10001:
        parser.error("The resident benchmark requires at least 10,001 rows")
    if args.worker:
        print(json.dumps(worker(args.rows, args.input_kind), allow_nan=False))
        return
    if not args.output or args.output.exists():
        parser.error("Choose a new output receipt path")
    results = []
    for rows in (1_000_000, 5_000_000):
        for kind in ("dataframe", "mapping"):
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                "--rows", str(rows), "--input-kind", kind], cwd=ROOT, capture_output=True, text=True, check=True)
            result = json.loads(run.stdout)
            results.append(result)
            print(json.dumps({"rows": rows, "input_kind": kind, "status": result["status"],
                              "scatter_additional_rss": result["scatter"]["sampled_rss_increase_bytes"]}), flush=True)
    with args.output.open("x") as handle:
        json.dump({"status": "passed", "fresh_process_per_case": True, "cases": results}, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
