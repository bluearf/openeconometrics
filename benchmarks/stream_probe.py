"""Measured local streaming OLS, never an inferred 100-billion-row benchmark.

Run fresh child processes with one requested CPU thread. Generated inputs are
replayed deterministically on each pass; their generation is included in fit
time. Parquet and in-memory fixture preparation is recorded separately. No user
files, cloud resources or active desktop sessions are touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["src/openecon/analysis.py", "src/openecon/analysis_contracts.py",
           "src/openecon/streaming_analysis.py", "src/openecon/dataset.py",
           "src/openecon/engines/streaming_ols.py", "src/openecon/engines/inference.py",
           "src/openecon/models.py", "src/openecon/__init__.py", "benchmarks/stream_probe.py"]
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                  "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def worker(rows: int, predictors: int, mode: str):
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    import openecon as oe

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1), tempfile.TemporaryDirectory(prefix="openecon-stream-probe-") as folder:
        names = [f"x{i}" for i in range(predictors)]
        beta = torch.linspace(.1, .5, predictors, dtype=torch.float64)

        def batches():
            rng = torch.Generator().manual_seed(20261002)
            for first in range(0, rows, 65_536):
                size = min(65_536, rows - first)
                values = torch.randn(size, predictors, generator=rng, dtype=torch.float64)
                outcome = .25 + values @ beta + torch.randn(size, generator=rng, dtype=torch.float64)
                frame = pd.DataFrame(values.numpy(), columns=names)
                frame["y"] = outcome.numpy()
                yield frame

        start = time.perf_counter()
        file_bytes = None
        if mode == "generated":
            source = oe.Dataset.from_batches(batches, names + ["y"], row_count=rows)
        elif mode == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            target = Path(folder) / "numeric.parquet"
            writer = None
            try:
                for frame in batches():
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                    writer.write_table(table, row_group_size=65_536)
            finally:
                if writer:
                    writer.close()
            file_bytes = target.stat().st_size
            source = oe.scan(target)
        else:
            # One actual large DataFrame tests the automatic public routing.
            rng = torch.Generator().manual_seed(20261002)
            values = torch.randn(rows, predictors, generator=rng, dtype=torch.float64)
            outcome = .25 + values @ beta + torch.randn(rows, generator=rng, dtype=torch.float64)
            source = pd.DataFrame(values.numpy(), columns=names)
            source["y"] = outcome.numpy()
            del values, outcome
        preparation = time.perf_counter() - start
        start = time.perf_counter()
        model = oe.ols(data=source, y="y", x=names, covariance="HC3")
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        encoded = model.model_dump_json()
        json_seconds = time.perf_counter() - start
        truth = {"Intercept": .25, **dict(zip(names, beta.tolist(), strict=True))}
        distances = [abs(coef.estimate - truth[coef.term]) / coef.std_error
                     for coef in model.coefficients]
        assert model.nobs == rows and model.nobs_original == rows
        assert model.provenance["solver"] == "torch_tsqr"
        assert model.provenance["streaming"]["passes"] == 3
        assert model.sample_positions == [] and len(model.predictions) <= 400
        assert len(encoded.encode()) < 100_000
        assert max(distances) < 6
        assert all(coef.std_error > 0 for coef in model.coefficients)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"release": oe.__version__, "rows": rows, "numeric_predictors": predictors,
                "mode": mode, "covariance": "HC3", "device": "cpu",
                "fixture_preparation_s": preparation, "parquet_file_bytes": file_bytes,
                "fit_including_three_source_passes_s": fit_seconds,
                "full_result_json_s": json_seconds, "full_result_json_bytes": len(encoded.encode()),
                "peak_process_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
                "maximum_truth_distance_standard_errors": max(distances),
                "coefficients": [coef.model_dump() for coef in model.coefficients],
                "r_squared": model.metrics["r_squared"], "bounded_prediction_count": len(model.predictions),
                "full_sample_positions_retained": False,
                "streaming": model.provenance["streaming"],
                "solver_diagnostics": model.provenance["solver_diagnostics"],
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                "observed_threadpools": [{key: pool.get(key) for key in ("user_api", "internal_api", "num_threads")}
                                         for pool in threadpool_info()],
                "sanity_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--predictors", type=int, default=5)
    parser.add_argument("--mode", choices=["generated", "parquet", "frame"], default="generated")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/verification/streaming-local-current.json")
    options = parser.parse_args()
    if options.rows <= 0 or options.predictors <= 0:
        parser.error("rows and predictors must be positive")
    if options.worker:
        print(json.dumps(worker(options.rows, options.predictors, options.mode), allow_nan=False), flush=True)
        return
    before = hashes()
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"system": platform.system(), "release": platform.release(),
                              "machine": platform.machine(), "python": platform.python_version(),
                              "logical_cpus": os.cpu_count(), "requested_threads": THREADS},
              "limitations": ["One fresh measurement per case on one busy Mac; OS cache not purged",
                              "Generated source fit includes deterministic input generation on all three passes",
                              "Parquet fit includes local file reads; fixture generation/write is separate",
                              "Peak RSS includes imports, input preparation, fit and JSON; no cloud/UI/GPU timing",
                              "Synthetic well-conditioned numeric data; truth recovery is not Stata equivalence",
                              "100 billion physical rows have not been tested; no inferred speed claim"],
              "source_hashes_before": before, "cases": []}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for rows, mode in [(10_000_000, "frame"), (10_000_000, "parquet"), (100_000_000, "generated")]:
        print(json.dumps({"stage": "started", "rows": rows, "mode": mode}), flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--rows", str(rows),
                   "--predictors", "5", "--mode", mode]
        start = time.perf_counter()
        result = subprocess.run(command, env={**os.environ, **THREADS}, capture_output=True,
                                text=True, timeout=300, cwd=ROOT)
        if result.returncode:
            report["failed"] = {"rows": rows, "mode": mode, "exit_code": result.returncode,
                                "error_tail": result.stderr[-3000:]}
            options.output.write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit("Streaming probe failed; inspect recorded error.")
        case = json.loads(result.stdout)
        case["subprocess_wall_s"] = time.perf_counter() - start
        report["cases"].append(case)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)
    report["source_hashes_after"] = hashes()
    report["sources_unchanged"] = before == report["source_hashes_after"]
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["sources_unchanged"]:
        raise SystemExit("Source changed during measurement.")


if __name__ == "__main__":
    main()
