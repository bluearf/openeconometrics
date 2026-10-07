"""Matched full-pipeline benchmark: former adapter, native NumPy and PyTorch.

Run: uv run --extra torch python scripts/benchmark.py
All output is JSON without hostnames, usernames or absolute machine paths.
This measures performance; tests/test_engine_equivalence.py tests correctness.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}
os.environ.update(_THREAD_ENVIRONMENT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import statsmodels  # noqa: E402
from scipy.special import expit  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402
from openecon.analysis import fit  # noqa: E402
from openecon.models import ModelSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260930
WARMUPS = 1
MEASUREMENTS = 5
WORKLOADS = [("ols", 10_000, 5), ("ols", 100_000, 5),
             ("ols", 10_000, 50), ("ols", 100_000, 50),
             ("logit", 10_000, 5), ("probit", 10_000, 5)]


def _cpu_description() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    return platform.processor() or "unreported"


def _blas_build(module: Any) -> dict[str, Any]:
    build = module.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    return {key: build[key] for key in ("name", "version", "openblas configuration", "has ilp64") if key in build}


def _source_hashes() -> dict[str, str]:
    files = [ROOT / "scripts/benchmark.py", ROOT / "benchmarks/reference/analysis_statsmodels.py", ROOT / "uv.lock"]
    files.extend(sorted((ROOT / "src/openecon").rglob("*.py")))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}


def _baseline():
    path = ROOT / "benchmarks/reference/analysis_statsmodels.py"
    spec = importlib.util.spec_from_file_location("openecon._statsmodels_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The preserved pre-native adapter is required for a matched comparison.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit


def _make_frame(estimator: str, rows: int, predictors: int):
    started = perf_counter()
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(rows, predictors)).astype(np.float64, copy=False)
    # Keep the first five coefficients identical to the historical workload.
    beta = np.resize(np.array([0.8, -0.4, 0.25, 0.1, -0.15]), predictors)
    beta[5:] /= np.sqrt(predictors)
    if estimator == "ols":
        y = 2.0 + x @ beta + (0.6 + 0.2 * np.abs(x[:, 0])) * rng.normal(size=rows)
    else:
        eta = -0.2 + x @ beta
        probability = expit(eta) if estimator == "logit" else scipy.special.ndtr(eta)
        y = rng.binomial(1, probability).astype(float)
    generated = perf_counter()
    names = [f"x{i}" for i in range(1, predictors + 1)]
    frame = pd.DataFrame({**{name: x[:, i] for i, name in enumerate(names)}, "y": y}, dtype=np.float64)
    return frame, names, {"synthetic_generation_seconds": generated - started,
                          "dataframe_construction_seconds": perf_counter() - generated}


def _difference(reference: dict, candidate: dict) -> dict[str, Any]:
    beta = np.array([c["estimate"] for c in candidate["coefficients"]])
    beta_ref = np.array([c["estimate"] for c in reference["coefficients"]])
    covariance = np.asarray(candidate["covariance_matrix"])
    covariance_ref = np.asarray(reference["covariance_matrix"])
    pvalues = np.array([c["p_value"] for c in candidate["coefficients"]])
    pvalues_ref = np.array([c["p_value"] for c in reference["coefficients"]])
    intervals = np.array([[c["ci_low"], c["ci_high"]] for c in candidate["coefficients"]])
    intervals_ref = np.array([[c["ci_low"], c["ci_high"]] for c in reference["coefficients"]])
    predictions = np.array([c["fitted"] for c in candidate["predictions"]])
    predictions_ref = np.array([c["fitted"] for c in reference["predictions"]])
    metrics = [name for name, value in reference["metrics"].items()
               if value is not None and name != "condition_number_scaled"]
    equivalent = bool(np.allclose(beta, beta_ref, rtol=2e-8, atol=2e-9)
                      and np.allclose(covariance, covariance_ref, rtol=2e-7, atol=2e-10)
                      and np.allclose(pvalues, pvalues_ref, rtol=2e-6, atol=2e-9)
                      and np.allclose(intervals, intervals_ref, rtol=2e-7, atol=2e-8)
                      and np.allclose(predictions, predictions_ref, rtol=2e-7, atol=2e-9)
                      and all(np.isclose(candidate["metrics"][name], reference["metrics"][name], rtol=2e-7, atol=2e-8) for name in metrics)
                      and candidate["sample_positions"] == reference["sample_positions"])
    return {"equivalent_on_workload": equivalent,
            "max_abs_coefficient_difference": float(np.max(np.abs(beta - beta_ref))),
            "max_abs_covariance_difference": float(np.max(np.abs(covariance - covariance_ref))),
            "max_abs_p_value_difference": float(np.max(np.abs(pvalues - pvalues_ref))),
            "max_abs_confidence_interval_difference": float(np.max(np.abs(intervals - intervals_ref))),
            "max_abs_chart_prediction_difference": float(np.max(np.abs(predictions - predictions_ref))),
            "absolute_metric_differences": {name: abs(candidate["metrics"][name] - reference["metrics"][name]) for name in metrics},
            "excluded_comparison_fields": ["id", "created_at", "backend/versions/optimizer provenance", "condition_number_scaled (different diagnostic scaling)"]}


def _measure(workload, engines, torch):
    estimator, rows, predictors = workload
    frame, names, preparation = _make_frame(estimator, rows, predictors)
    spec_data = {"estimator": estimator, "outcome": "y", "predictors": names,
                 "intercept": True, "covariance": "HC3" if estimator == "ols" else "nonrobust",
                 "missing": "raise", "alpha": 0.05}
    records = {label: {"warmup_seconds": [], "fit_seconds": []} for label in engines}
    outputs = {}
    labels = list(engines)
    # Rotate the engine order across repetitions to reduce monotonic warm/thermal bias.
    for iteration in range(WARMUPS + MEASUREMENTS):
        offset = iteration % len(labels)
        for label in labels[offset:] + labels[:offset]:
            function, backend, device = engines[label]
            gc.collect()
            if device == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            spec = ModelSpec.model_validate({**spec_data, "backend": backend, "device": device})
            result = function(spec, data=frame).model_dump(mode="json")
            if device == "cuda":
                torch.cuda.synchronize()
            duration = perf_counter() - started
            if result["nobs"] != rows or len(result["coefficients"]) != predictors + 1:
                raise RuntimeError("The fitted sample or design differs from the benchmark contract.")
            records[label]["warmup_seconds" if iteration < WARMUPS else "fit_seconds"].append(duration)
            if iteration == WARMUPS + MEASUREMENTS - 1:
                outputs[label] = result
            del result
    reference = outputs["statsmodels_adapter"]
    baseline_median = statistics.median(records["statsmodels_adapter"]["fit_seconds"])
    for label, record in records.items():
        durations = record["fit_seconds"]
        record["recorded_backend"] = outputs[label]["provenance"]["backend"]
        record["recorded_device"] = outputs[label]["provenance"].get("device", "cpu")
        record.update(median_seconds=statistics.median(durations), min_seconds=min(durations), max_seconds=max(durations))
        record["baseline_time_divided_by_engine_time"] = baseline_median / record["median_seconds"]
        record["vs_statsmodels_adapter"] = _difference(reference, outputs[label])
        if not record["vs_statsmodels_adapter"]["equivalent_on_workload"]:
            raise RuntimeError(f"Benchmark output differs from the oracle: {estimator}/{rows}/{predictors}/{label}.")
    design_bytes = rows * (predictors + 1) * 8
    return {"estimator": estimator, "rows": rows, "numeric_predictors": predictors, "dtype": "float64",
            "spec": spec_data, **preparation, "engines": records,
            "dense_design_theoretical_bytes": design_bytes, "dense_design_theoretical_mib": design_bytes / 1024**2,
            "peak_ram_measured": False}


def main():
    source_before = _source_hashes()
    engines = {"statsmodels_adapter": (_baseline(), "numpy", "cpu"), "native_numpy": (fit, "numpy", "cpu")}
    torch = None
    torch_environment = {"installed": False, "cuda_available": False, "cuda_measured": False}
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        engines["native_torch_cpu"] = (fit, "torch", "cpu")
        cuda = torch.cuda.is_available()
        torch_environment = {"installed": True, "version": torch.__version__, "num_threads": torch.get_num_threads(),
                             "num_interop_threads": torch.get_num_interop_threads(), "cuda_available": cuda, "cuda_measured": cuda,
                             "cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
                             "mps_measured": False, "dtype": "float64"}
        if cuda:
            engines["native_torch_cuda"] = (fit, "torch", "cuda")
    with threadpool_limits(limits=1):
        result = {"benchmark_version": 2, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                  "environment": {"os": platform.system(), "os_release": platform.release(),
                                  "macos_version": platform.mac_ver()[0] or None, "architecture": platform.machine(),
                                  "cpu": _cpu_description(), "logical_cpu_count": os.cpu_count(),
                                  "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                  "scipy": scipy.__version__, "statsmodels_oracle": statsmodels.__version__,
                                  "threadpoolctl": version("threadpoolctl"), "torch": torch_environment,
                                  "blas_build": {"numpy": _blas_build(np), "scipy": _blas_build(scipy)}},
                  "thread_control": {"requested_threads": 1, "environment_requests": _THREAD_ENVIRONMENT,
                                     "threadpoolctl_limit": 1,
                                     "observed_pools": [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()],
                                     "limitation": "Runtime counts are observable only for pools discovered by threadpoolctl; environment requests do not prove an undiscoverable backend's count."},
                  "method": {"seed": SEED, "warmups_per_workload_engine": WARMUPS,
                             "measurements_per_workload_engine": MEASUREMENTS, "engine_order": "rotated each repetition",
                             "timer": "time.perf_counter", "cuda_synchronization": "before and after every CUDA timing",
                             "timed": "ModelSpec validation + complete fit including input checks, binary separation LP, inference, provenance hashing and ResultBundle JSON-mode conversion; CPU/CUDA data transfers included",
                             "excluded": "Process/import startup, data generation, DataFrame construction, pre-run GC, JSON text encoding, filesystem/UI/MCP work",
                             "peak_ram_measured": False, "stata_comparison": False},
                  "source_sha256": source_before,
                  "results": [_measure(workload, engines, torch) for workload in WORKLOADS]}
    result["source_unchanged_during_run"] = _source_hashes() == source_before
    if not result["source_unchanged_during_run"]:
        raise RuntimeError("Source changed during the benchmark; rerun on a stable checkout.")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
