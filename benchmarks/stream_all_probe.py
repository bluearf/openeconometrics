"""Measure every shipped streaming estimator/covariance in fresh processes.

Each child writes a real Parquet fixture in bounded blocks, then fits through
the public Dataset API with categorical predictors and, where requested, 50,000
interleaved clusters. Preparation is separate from fitting. Each source replay
is timed without retaining its rows. Only owned temporary directories are used.
This measures one million physical rows, never an inferred larger workload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [("ols", covariance) for covariance in ("nonrobust", "HC1", "HC3", "cluster")]
CASES += [(model, covariance) for model in ("logit", "probit")
          for covariance in ("nonrobust", "cluster")]
SOURCES = ["src/openecon/__init__.py", "src/openecon/analysis.py",
           "src/openecon/analysis_contracts.py", "src/openecon/dataset.py",
           "src/openecon/models.py", "src/openecon/streaming_analysis.py",
           "src/openecon/streaming_design.py", "src/openecon/engines/contracts.py",
           "src/openecon/engines/streaming_ols.py", "src/openecon/engines/streaming_binary.py",
           "src/openecon/engines/streaming_groups.py", "src/openecon/engines/inference.py",
           "benchmarks/stream_all_probe.py", "pyproject.toml"]
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                  "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
TRUTH = {"Intercept": .2, "x": .35, "z": -.25, "category[B]": .3, "category[C]": -.2}
WRITE_BATCH_ROWS = 65_536
TIMEOUT_SECONDS = 120


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: file_hash(ROOT / name) for name in SOURCES}


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_fixture(path: Path, rows: int, groups: int) -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    rng = torch.Generator().manual_seed(20261003)
    writer = None
    written = blocks = maximum_frame_bytes = 0
    start = time.perf_counter()
    try:
        for first in range(0, rows, WRITE_BATCH_ROWS):
            size = min(WRITE_BATCH_ROWS, rows - first)
            position = torch.arange(first, first + size, dtype=torch.int64)
            values = torch.randn((size, 2), generator=rng, dtype=torch.float64)
            category = position.remainder(3)
            linear = (.2 + .35 * values[:, 0] - .25 * values[:, 1]
                      + .3 * (category == 1) - .2 * (category == 2))
            continuous = linear + torch.randn(size, generator=rng, dtype=torch.float64)
            logistic = (torch.rand(size, generator=rng, dtype=torch.float64)
                        < torch.sigmoid(linear)).to(torch.int64)
            probit = (linear + torch.randn(size, generator=rng, dtype=torch.float64) > 0).to(torch.int64)
            frame = pd.DataFrame({"x": values[:, 0].numpy(), "z": values[:, 1].numpy(),
                                  "category": pd.Categorical.from_codes(category.numpy(),
                                                                         ["A", "B", "C"]),
                                  "cluster": position.remainder(groups).numpy(),
                                  "y_ols": continuous.numpy(), "y_logit": logistic.numpy(),
                                  "y_probit": probit.numpy()})
            maximum_frame_bytes = max(maximum_frame_bytes, int(frame.memory_usage(deep=True).sum()))
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=WRITE_BATCH_ROWS)
            written += len(frame)
            blocks += 1
    finally:
        if writer is not None:
            writer.close()
    seconds = time.perf_counter() - start
    metadata = pq.read_metadata(path)
    if written != rows or metadata.num_rows != rows:
        raise AssertionError("The physical Parquet row count differs from the requested fixture.")
    return {"rows": written, "write_blocks": blocks, "maximum_write_batch_rows": min(rows, WRITE_BATCH_ROWS),
            "maximum_frame_bytes": maximum_frame_bytes, "parquet_row_groups": metadata.num_row_groups,
            "parquet_file_bytes": path.stat().st_size, "parquet_sha256": file_hash(path),
            "generation_and_write_s": seconds,
            "category_levels": ["A", "B", "C"], "distinct_cluster_count": min(groups, rows),
            "outcome_generation": {"ols": "linear mean plus independent standard normal noise",
                                   "logit": "Bernoulli with sigmoid(linear mean)",
                                   "probit": "one if linear mean plus standard normal noise is positive"},
            "truth": TRUTH, "seed": 20261003}


def worker(model_name: str, covariance: str, rows: int, groups: int, owned_directory: Path) -> dict:
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    import openecon as oe

    if not owned_directory.is_dir() or any(owned_directory.iterdir()):
        raise ValueError("The worker needs its parent's empty owned temporary directory.")
    imported_source = Path(oe.__file__).resolve()
    if not imported_source.is_relative_to(ROOT / "src"):
        raise RuntimeError("The imported package does not belong to this measured source snapshot.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before = source_hashes()
    with threadpool_limits(limits=1):
        fixture_path = owned_directory / "fixture.parquet"
        scratch = owned_directory / "scratch"
        scratch.mkdir(mode=0o700)
        previous_scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
        os.environ["OPENECON_SCRATCH_DIRECTORY"] = str(scratch)
        try:
            preparation_start = time.perf_counter()
            fixture = write_fixture(fixture_path, rows, groups)
            scan_start = time.perf_counter()
            source = oe.scan(fixture_path)
            scan_seconds = time.perf_counter() - scan_start
            preparation_seconds = time.perf_counter() - preparation_start
            assert source.row_count == rows
            pass_records = []
            original_iterator = source.iter_batches

            def timed_batches(*args, **kwargs):
                start = time.perf_counter()
                count = blocks = maximum_batch_rows = 0
                complete = False
                iterator = original_iterator(*args, **kwargs)
                try:
                    for frame in iterator:
                        count += len(frame)
                        blocks += 1
                        maximum_batch_rows = max(maximum_batch_rows, len(frame))
                        yield frame
                    complete = True
                finally:
                    iterator.close()
                    pass_records.append({"pass": len(pass_records) + 1, "rows_read": count,
                                         "batches_read": blocks, "maximum_batch_rows": maximum_batch_rows,
                                         "complete": complete,
                                         "elapsed_including_between_yields_s": time.perf_counter() - start})

            source.iter_batches = timed_batches
            rss_after_preparation = peak_rss_bytes()
            start = time.perf_counter()
            fitted = getattr(oe, model_name)(
                data=source, y=f"y_{model_name}", x=["x", "z", "category"],
                categorical=["category"], covariance=covariance,
                cluster="cluster" if covariance == "cluster" else None,
            )
            fit_seconds = time.perf_counter() - start
            scratch_cleaned = not any(scratch.iterdir())
            start = time.perf_counter()
            encoded = fitted.model_dump_json()
            serialization_seconds = time.perf_counter() - start
            distances = {coefficient.term: abs(coefficient.estimate - TRUTH[coefficient.term]) / coefficient.std_error
                         for coefficient in fitted.coefficients}
            diagnostics = fitted.provenance["solver_diagnostics"]
            expected_solver = ("torch_tsqr" if model_name == "ols"
                               else "torch_streaming_newton_cholesky_backtracking")
            assert fitted.nobs == rows and fitted.nobs_original == rows and fitted.dropped_rows == 0
            assert set(distances) == set(TRUTH) and max(distances.values()) < 6
            assert all(coefficient.std_error > 0 and math.isfinite(coefficient.std_error)
                       for coefficient in fitted.coefficients)
            assert fitted.sample_positions == [] and len(fitted.predictions) == min(400, rows)
            assert len(encoded.encode()) < 100_000
            assert fitted.provenance["solver"] == expected_solver
            assert fitted.provenance["streaming"]["passes"] == len(pass_records)
            assert all(record["complete"] and record["rows_read"] == rows for record in pass_records)
            assert fitted.provenance["categorical_encoding"]["category"]["reference"] == "A"
            assert scratch_cleaned
            if covariance == "cluster":
                assert fitted.inference["cluster_count"] == min(groups, rows)
                assert diagnostics["cluster_aggregation"] == "sqlite_spill"
                assert diagnostics["cluster_spilled_groups"] == min(groups, rows)
                assert diagnostics["cluster_spilled_groups"] > diagnostics["cluster_cache_capacity"]
                assert diagnostics["cluster_scratch_bytes"] > 0
            else:
                assert fitted.inference["cluster_count"] is None
            after = source_hashes()
            assert before == after
            return {"release": oe.__version__, "model": model_name, "covariance": covariance,
                    "rows": rows, "device": "cpu", "precision": "float64",
                    "fixture": fixture, "fixture_preparation_including_scan_and_hash_s": preparation_seconds,
                    "scan_metadata_s": scan_seconds, "fit_including_all_source_passes_s": fit_seconds,
                    "source_passes": pass_records, "actual_source_pass_count": len(pass_records),
                    "sum_source_pass_elapsed_s": sum(record["elapsed_including_between_yields_s"] for record in pass_records),
                    "peak_process_rss_bytes": peak_rss_bytes(),
                    "peak_process_rss_after_preparation_bytes": rss_after_preparation,
                    "full_result_json_s": serialization_seconds, "full_result_json_bytes": len(encoded.encode()),
                    "maximum_truth_distance_standard_errors": max(distances.values()),
                    "truth_distance_by_term_standard_errors": distances,
                    "coefficients": [coefficient.model_dump() for coefficient in fitted.coefficients],
                    "metrics": fitted.metrics, "group_count": fitted.inference["cluster_count"],
                    "bounded_prediction_count": len(fitted.predictions), "full_sample_positions_retained": False,
                    "solver": fitted.provenance["solver"], "streaming": fitted.provenance["streaming"],
                    "solver_diagnostics": diagnostics, "input_data_hash": fitted.provenance["data_hash"],
                    "selected_positions_hash": fitted.provenance["sample_positions_hash"],
                    "scratch_cleaned_after_fit": scratch_cleaned, "source_hashes_before": before,
                    "source_hashes_after": after, "sources_unchanged": True,
                    "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "observed_threadpools": [{key: pool.get(key) for key in ("user_api", "internal_api", "num_threads")}
                                             for pool in threadpool_info()],
                    "sanity_checks_passed": True}
        finally:
            if previous_scratch is None:
                os.environ.pop("OPENECON_SCRATCH_DIRECTORY", None)
            else:
                os.environ["OPENECON_SCRATCH_DIRECTORY"] = previous_scratch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=("ols", "logit", "probit"), default="ols")
    parser.add_argument("--covariance", choices=("nonrobust", "HC1", "HC3", "cluster"), default="nonrobust")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--groups", type=int, default=50_000)
    parser.add_argument("--owned-directory", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/verification/streaming-all-local-current.json")
    options = parser.parse_args()
    if options.rows < 10 or not 2 <= options.groups <= options.rows:
        parser.error("rows must be at least 10, and groups must be between 2 and rows")
    if options.worker:
        if options.owned_directory is None or (options.model, options.covariance) not in CASES:
            parser.error("a supported model/covariance pair and an owned directory are required")
        print(json.dumps(worker(options.model, options.covariance, options.rows, options.groups,
                                options.owned_directory), allow_nan=False), flush=True)
        return
    before = source_hashes()
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"system": platform.system(), "release": platform.release(),
                              "machine": platform.machine(), "python": platform.python_version(),
                              "logical_cpus": os.cpu_count(), "requested_threads": THREADS},
              "rows_per_case": options.rows, "distinct_clusters": options.groups,
              "timeout_seconds_per_case": TIMEOUT_SECONDS, "source_hashes_before": before, "cases": [],
              "limitations": ["One fresh process per model/covariance on one busy machine; OS cache is not purged",
                              "Each process generates and reads a real local Parquet fixture; no cloud, GUI or GPU timing",
                              "Peak RSS includes imports, fixture preparation, fitting and result serialization",
                              "Per-pass elapsed time includes model work between yielded batches",
                              "Synthetic well-conditioned data with three category levels and independent noise",
                              "Truth recovery is a sanity check, not Stata parity or a general numerical validation",
                              "One million physical rows by default; 100 billion rows have not been measured"]}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, **THREADS,
                   "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for model_name, covariance in CASES:
        print(json.dumps({"stage": "started", "model": model_name, "covariance": covariance,
                          "rows": options.rows}), flush=True)
        start = time.perf_counter()
        case = None
        failure = None
        with tempfile.TemporaryDirectory(prefix="openecon-stream-all-probe-") as folder:
            owned_path = Path(folder)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", model_name,
                       "--covariance", covariance, "--rows", str(options.rows), "--groups", str(options.groups),
                       "--owned-directory", str(owned_path)]
            try:
                result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                        timeout=TIMEOUT_SECONDS, cwd=ROOT)
                if result.returncode:
                    failure = {"model": model_name, "covariance": covariance, "exit_code": result.returncode,
                               "error_tail": result.stderr[-3000:]}
                else:
                    case = json.loads(result.stdout)
            except subprocess.TimeoutExpired:
                failure = {"model": model_name, "covariance": covariance, "timeout_seconds": TIMEOUT_SECONDS}
            except (json.JSONDecodeError, OSError) as exc:
                failure = {"model": model_name, "covariance": covariance, "error": str(exc)}
        cleanup_verified = not owned_path.exists()
        if failure is not None:
            report["failed"] = {**failure, "owned_fixture_and_scratch_cleanup_verified": cleanup_verified}
            report["source_hashes_after"] = source_hashes()
            options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            raise SystemExit("Streaming probe failed; inspect the recorded case and error.")
        case["subprocess_wall_s"] = time.perf_counter() - start
        case["owned_fixture_and_scratch_cleanup_verified"] = cleanup_verified
        assert cleanup_verified
        report["cases"].append(case)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"stage": "completed", "model": model_name, "covariance": covariance,
                          "fit_s": case["fit_including_all_source_passes_s"],
                          "peak_rss_bytes": case["peak_process_rss_bytes"],
                          "passes": case["actual_source_pass_count"]}), flush=True)
    report["source_hashes_after"] = source_hashes()
    report["sources_unchanged"] = before == report["source_hashes_after"]
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = report["sources_unchanged"] and len(report["cases"]) == len(CASES)
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("Source changed during measurement.")


if __name__ == "__main__":
    main()
