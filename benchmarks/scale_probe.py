"""Bounded current-core scaling probe, not a production or Stata comparison.

Each workload runs in a fresh process, using deterministic synthetic data and
one requested CPU thread. Timings include the public fit API, inference,
provenance, and result conversion. Imports, fixture generation, file I/O, HTTP,
the console broker, and UI are excluded. Peak RSS is the process high-water mark
including imports, data, warmup, and repeats; it is not an isolated kernel delta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}
SOURCES = [
    "benchmarks/scale_probe.py", "src/openecon/analysis.py",
    "src/openecon/engines/torch_engine.py", "src/openecon/engines/inference.py",
    "src/openecon/models.py", "src/openecon/data.py", "uv.lock",
]
CASES = [
    ("ols", "HC3", 10_000, 5), ("ols", "HC3", 100_000, 5),
    ("ols", "HC3", 1_000_000, 5), ("ols", "HC3", 100_000, 50),
    ("ols", "cluster", 100_000, 5), ("logit", "nonrobust", 100_000, 5),
]


def hashes() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def peak_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def worker(options) -> dict:
    import gc
    import math

    import pandas as pd
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    import openecon as oe

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1):
        baseline = peak_bytes()
        prep_start = time.perf_counter()
        rng = torch.Generator().manual_seed(20261002)
        values = torch.randn(options.rows, options.predictors, generator=rng, dtype=torch.float64)
        beta = torch.linspace(0.1, 0.5, options.predictors, dtype=torch.float64)
        linear = 0.25 + values @ beta
        if options.estimator == "ols":
            outcome = linear + torch.randn(options.rows, generator=rng, dtype=torch.float64)
        else:
            outcome = torch.bernoulli(torch.sigmoid(linear), generator=rng)
        names = [f"x{i}" for i in range(options.predictors)]
        frame = pd.DataFrame(values.numpy(), columns=names)
        frame["y"] = outcome.numpy()
        groups = None
        if options.covariance == "cluster":
            frame["group"] = (torch.arange(options.rows) % 1000).numpy()
            groups = "group"
        prep_s = time.perf_counter() - prep_start
        del values, outcome, linear

        def fit_and_convert():
            result = getattr(oe, options.estimator)(
                data=frame, y="y", x=names, covariance=options.covariance, cluster=groups,
            )
            payload = result.model_dump(mode="json")
            return result, payload

        start = time.perf_counter()
        result, payload = fit_and_convert()
        warmup_s = time.perf_counter() - start
        raw = []
        for _ in range(options.repeats):
            del result, payload
            gc.collect()
            start = time.perf_counter()
            result, payload = fit_and_convert()
            raw.append(time.perf_counter() - start)

        expected = {"Intercept": 0.25, **dict(zip(names, beta.tolist(), strict=True))}
        largest_error = max(abs(c.estimate - expected[c.term]) for c in result.coefficients)
        assert result.nobs == options.rows
        assert len(result.sample_positions) == options.rows
        assert result.sample_positions[0] == 0 and result.sample_positions[-1] == options.rows - 1
        assert len(result.predictions) == 400
        assert all(math.isfinite(c.estimate) and c.std_error > 0 for c in result.coefficients)
        # Recovery of a known simulation is a sanity check, not an equivalence oracle.
        assert largest_error < 0.1, largest_error
        start = time.perf_counter()
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        encoding_s = time.perf_counter() - start
        return {
            "estimator": options.estimator, "covariance": options.covariance,
            "rows": options.rows, "numeric_predictors": options.predictors,
            "design_columns_with_intercept": options.predictors + 1,
            "design_bytes_theoretical": options.rows * (options.predictors + 1) * 8,
            "frame_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
            "fixture_preparation_s": prep_s, "warmup_s": warmup_s,
            "fit_and_model_dump_raw_s": raw, "fit_and_model_dump_median_s": statistics.median(raw),
            "last_full_json_encoding_s": encoding_s, "full_result_json_bytes": len(encoded.encode()),
            "peak_process_rss_bytes": peak_bytes(), "import_baseline_peak_rss_bytes": baseline,
            "largest_coefficient_error_vs_simulated_truth": largest_error,
            "sanity_checks_passed": True, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "observed_threadpools": [
                {k: pool.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
                for pool in threadpool_info()
            ],
            "versions": result.provenance["versions"],
            "outside_import_row_limit": options.rows > 100_000,
            "solver_iterations": result.provenance.get("optimizer"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--estimator", default="ols")
    parser.add_argument("--covariance", default="HC3")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--predictors", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/verification/scale-local-current.json")
    options = parser.parse_args()
    if options.worker:
        print(json.dumps(worker(options), allow_nan=False))
        return
    before = hashes()
    report = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "openecon_release": "0.3.0a1", "environment": {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "requested_thread_environment": THREAD_ENV,
        },
        "scope": "Local warm public fit + inference + provenance + model_dump; fresh process per workload",
        "limitations": [
            "Synthetic well-conditioned numeric predictors; no missing values or categorical expansion",
            "No production, network, file loading, cold start, console sandbox, or UI timings",
            "No Stata or other estimator comparison; truth recovery is only a simulation sanity check",
            "Single desktop run; three repetitions do not establish tail latency or throughput",
            "One million rows exercise the direct core API beyond the 100,000-row file-import limit",
            "Peak RSS includes imports/data/warmup/repeats/encoding and is not cloud container memory",
            "Accelerate runtime thread count may not be observable",
        ],
        "cases": [], "source_hashes_before": before,
    }
    environment = {**os.environ, **THREAD_ENV}
    for estimator, covariance, rows, predictors in CASES:
        started = time.perf_counter()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--estimator", estimator,
                   "--covariance", covariance, "--rows", str(rows), "--predictors", str(predictors),
                   "--repeats", str(options.repeats)]
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True,
                                       timeout=60, check=True, cwd=ROOT)
            case = json.loads(completed.stdout)
            case["subprocess_wall_s_including_startup"] = time.perf_counter() - started
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            case = {"estimator": estimator, "covariance": covariance, "rows": rows,
                    "numeric_predictors": predictors, "failed": True,
                    "failure": type(exc).__name__, "elapsed_s": time.perf_counter() - started,
                    "detail": str(getattr(exc, "stderr", "") or "")[-3000:]}
        report["cases"].append(case)
        print(json.dumps(case, allow_nan=False), flush=True)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["source_hashes_after"] = hashes()
    report["sources_unchanged"] = before == report["source_hashes_after"]
    options.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["sources_unchanged"] or any(c.get("failed") for c in report["cases"]):
        raise SystemExit("Source changed or workload failed; inspect report before making speed claims.")


if __name__ == "__main__":
    main()
