"""Seeded physical-Parquet benchmark of exact sparse cuts and spanning forests.

Run with an installed/editable OpenEconometrics SDK:
    python benchmarks/network_paths_scale.py --output measured-network-paths.json

No input fixture, private host name or absolute source path enters the receipt.
The disposable Parquet and graph are complete; this does not benchmark all-pairs
distances, infer universal speed guarantees, or claim an out-of-core graph.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import time

import pandas as pd
import torch

import openecon as oe
from openecon import _network_centrality, _network_io, _network_paths, _network_sparse, networks


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return value


def _positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Use a positive finite number.")
    return value


def _fingerprints():
    modules = (networks, _network_paths, _network_centrality, _network_sparse, _network_io)
    return {f"src/openecon/{module.__name__.rsplit('.', 1)[-1]}.py":
            hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
            for module in modules}


def _cpu():
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def run(options):
    rows, n, budget_mb = options.rows, options.nodes, options.budget_mb
    # This benchmark owns physical-input creation too. Reject a plainly
    # oversized fixture before generating tensors; total RSS remains separate.
    if rows * 128 + n * 64 > budget_mb * 1024**2:
        raise ValueError("Requested fixture creation exceeds the benchmark's explicit memory budget.")
    torch.set_num_threads(options.threads)
    before = _fingerprints()
    generator = torch.Generator().manual_seed(options.seed)
    source = torch.randint(n, (rows,), generator=generator)
    target = torch.randint(n, (rows,), generator=generator)
    weight = torch.randint(1, 100, (rows,), generator=generator).to(torch.float64)
    # numpy() is a zero-copy buffer bridge into pandas, not a numeric solver.
    frame = pd.DataFrame({"source": source.numpy(), "target": target.numpy(), "weight": weight.numpy()})
    report = {"status": "running", "cpu": _cpu(), "platform": platform.platform(),
              "python": platform.python_version(), "torch": str(torch.__version__),
              "pandas": pd.__version__, "sdk": oe.__version__,
              "physical_rows": rows, "node_count": n, "seed": options.seed,
              "torch_threads": options.threads, "max_memory_mb": budget_mb,
              "max_work": options.max_work, "batch_rows": options.batch_rows,
              "device": "cpu", "dtype": "float64", "directed": False, "weighted": True,
              "edge_costs": "aggregate random integer weights 1..99; duplicate rows add",
              "timing_seconds": {}}
    with tempfile.TemporaryDirectory(prefix="openeconometrics-network-paths-") as temporary:
        path = Path(temporary) / "physical.parquet"
        frame.to_parquet(path)
        del frame, source, target, weight
        dataset = oe.scan(path)
        started = time.perf_counter()
        graph = oe.network(dataset, nodes=range(n), weight="weight", batch_rows=options.batch_rows,
                           max_memory_mb=budget_mb)
        report["timing_seconds"]["import"] = time.perf_counter() - started
        report["edge_count"] = graph.edge_count
        report["physical_input_sha256"] = graph.metadata["input_sha256"]
        report["actual_peak_batch_rows"] = graph.metadata["actual_peak_batch_rows"]
        assert graph.node_count == n and graph.metadata["input_rows"] == rows
        for name in ("bridges", "articulation_points", "minimum_spanning_forest"):
            started = time.perf_counter()
            result = getattr(graph, name)(max_work=options.max_work)
            report["timing_seconds"][name] = time.perf_counter() - started
            if name == "bridges":
                report["bridge_count"] = len(result)
            elif name == "articulation_points":
                report["articulation_count"] = int(result.articulation.sum())
            else:
                report["forest_edge_count"] = result.edge_count
                report["forest_components"] = result.metadata["component_count"]
                report["forest_total_cost"] = result.metadata["total_cost"]
                assert result.node_count == n
                assert result.edge_count == n - result.metadata["component_count"]
            del result
        dataset.assert_unchanged()
    after = _fingerprints()
    if before != after:
        raise RuntimeError("Measured source bytes changed during the benchmark; rerun from stable source.")
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(status="passed", source_sha256_before=before, source_sha256_after=after,
                  source_bytes_unchanged=True, benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  peak_process_rss_bytes=maximum_rss * (1 if platform.system() == "Darwin" else 1024),
                  memory_scope="Owned graph/buffer budget is separate from process lifetime RSS, "
                               "which includes Python/Torch, physical-input creation and every phase.",
                  timing_scope="Library imports and Parquet creation excluded from operation timings.",
                  full_graph_analysis=True, out_of_core_graph=False, sampled=False,
                  universal_speed_guarantee=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=_positive_integer, default=100_000)
    parser.add_argument("--rows", type=_positive_integer, default=1_000_000)
    parser.add_argument("--seed", type=int, default=92834)
    parser.add_argument("--threads", type=_positive_integer, default=2)
    parser.add_argument("--budget-mb", type=_positive_float, default=1024)
    parser.add_argument("--max-work", type=_positive_integer, default=200_000_000)
    parser.add_argument("--batch-rows", type=_positive_integer, default=65536)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.output is not None and options.output.exists():
        parser.error("The output already exists; choose a new receipt path.")
    report = run(options)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        with options.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
