"""Reproducible native CPU timing for panel homogeneity and CADF/CIPS.

Default inputs have 1,000 units and 1,000 periods (one million rows).
Measurements time complete public API calls, including input validation,
preparation and kernels. Imports, data construction and small warm-ups are
excluded. These observations do not establish arbitrary-data throughput or
finite-sample inference, and CIPS inference is disabled outside table support.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.abc
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

_FORBIDDEN = ("statsmodels", "linearmodels", "sklearn", "scipy.optimize", "scipy.stats")
_ROOT = Path(__file__).resolve().parents[1]


def _forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN)


class _NativeOnly(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if _forbidden(fullname):
            self.attempts.append(fullname)
            raise ImportError(f"This benchmark requires native OpenEcon kernels: {fullname}")
        return None


def _sources() -> dict[str, str]:
    """Hash the actual imported repository sources, without inventing a commit."""
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        if name != "openecon" and not name.startswith("openecon."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if not path.is_relative_to(_ROOT) or path.suffix != ".py":
            raise RuntimeError("An OpenEcon module did not load from this checkout's Python source.")
        paths.add(path)
    return {path.relative_to(_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def _rss() -> dict[str, Any]:
    try:
        import resource
    except ImportError:
        return {"peak_process_rss_bytes": None, "scope": "Unavailable on this platform."}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB. The value is a process-wide
    # cumulative high-water mark, including libraries, data and all warm-ups.
    multiplier = 1 if platform.system() == "Darwin" else 1024
    return {"peak_process_rss_bytes": int(peak * multiplier),
            "scope": "Cumulative process high-water mark including imports, synthetic inputs, warm-ups and operations; not per-fit memory or total machine memory."}


def run(*, groups: int = 1000, periods: int = 1000, threads: int = 2,
        block_size: int = 128, memory_mb: int = 64, seed: int = 20261005) -> dict[str, Any]:
    if groups < 2 or periods < 8 or threads < 1:
        raise ValueError("Use at least two units, eight periods and one CPU thread.")
    previously_loaded = sorted(name for name in sys.modules if _forbidden(name))
    if previously_loaded:
        raise RuntimeError("Run this benchmark in a fresh process: a blocked estimation module is already loaded.")
    blocker = _NativeOnly()
    sys.meta_path.insert(0, blocker)
    import pandas as pd
    import torch
    import openecon as oe

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    k = 3
    with torch.no_grad(), torch.device("cpu"):
        slopes = torch.tensor([.5, -.3, .15], dtype=torch.float64)
        unit_sigma = torch.linspace(.6, 1.8, groups, dtype=torch.float64)
        unit_intercept = torch.randn(groups, dtype=torch.float64, generator=rng)
        x = torch.randn((groups, periods, k), dtype=torch.float64, generator=rng)
        y = x @ slopes + unit_intercept[:, None]
        y += unit_sigma[:, None] * torch.randn((groups, periods), dtype=torch.float64, generator=rng)
        ids = torch.arange(groups, dtype=torch.int64).repeat_interleave(periods)
        dates = torch.arange(periods, dtype=torch.int64).repeat(groups)
        homogeneity_data = pd.DataFrame({"id": ids.numpy(), "t": dates.numpy(), "y": y.flatten().numpy(),
                                        **{f"x{j}": x[..., j].flatten().numpy() for j in range(k)}})
        homogeneity = oe.xthst
        cips = oe.xtcips
        # Small public calls resolve lazy numerical imports and initialize the
        # CPU kernels. Their results/times are not included in the benchmark.
        warm_units = min(groups, 12)
        warm_periods = min(periods, 40)
        warm = homogeneity_data.loc[(homogeneity_data.id < warm_units)
                                    & (homogeneity_data.t < warm_periods)].copy()
        homogeneity(data=warm, y="y", x=["x0", "x1", "x2"], panel="id", time="t", method="all")
        cips(data=warm, y="y", panel="id", time="t", lags=1, inference="none")
        source_before = _sources()
        if any(_forbidden(name) for name in sys.modules):
            raise RuntimeError("A forbidden estimator module was loaded during initialization.")

        start = perf_counter()
        slope_test = homogeneity(data=homogeneity_data, y="y", x=["x0", "x1", "x2"],
                                 panel="id", time="t", method="all", block_size=block_size,
                                 memory_mb=memory_mb)
        slope_seconds = perf_counter() - start
        assert slope_test.attrs["nobs"] == groups * periods
        assert slope_test.attrs["n_groups"] == groups and slope_test.attrs["k"] == k
        assert slope_test.attrs["balanced"] is True and slope_test.attrs["n_dropped"] == 0
        assert slope_test.attrs["workspace"]["algorithm"] == "three_pass_batched_tsqr"
        assert slope_test.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= memory_mb * 1024**2
        assert slope_test.statistic.map(lambda number: float("-inf") < number < float("inf")).all()
        assert slope_test.p_value.between(0, 1).all()
        homogeneity_record = {
            "public_api": "openecon.xthst", "rows": len(homogeneity_data), "groups": groups,
            "periods": periods, "predictors": k, "seconds": slope_seconds,
            "dgp": {"model": "static common-slope Gaussian null with unit intercepts and independent innovations",
                    "slopes": slopes.tolist(), "sigma_min": float(unit_sigma.min()),
                    "sigma_max": float(unit_sigma.max()), "heteroskedasticity": "across units only"},
            "statistics": {name: {"statistic": float(slope_test.loc[name, "statistic"]),
                                  "p_value": float(slope_test.loc[name, "p_value"])} for name in slope_test.index},
            "dispersion": slope_test.attrs["dispersion"],
            "guards": {"minimum_complete_periods_required": max(6, k + 2), "actual_t_min": slope_test.attrs["t_min"],
                       "actual_t_max": slope_test.attrs["t_max"], "all_units_full_within_rank": True,
                       "positive_resolvable_pooled_fe_variances": True,
                       "max_unit_scaled_condition": slope_test.attrs["max_unit_scaled_condition"],
                       "p_values": "upper-tail asymptotic normal; not exact finite-sample calibration"},
            "variance_estimator": slope_test.attrs["variance_estimator"],
            "adjustment": slope_test.attrs["adjustment"],
            "workspace": slope_test.attrs["workspace"], "provenance": slope_test.attrs["provenance"],
            "rss_after_operation": _rss(),
        }
        del homogeneity_data, x, y, unit_sigma, unit_intercept, slope_test, warm
        gc.collect()

        # A one-factor unit-root null with nonzero mean loading. The first
        # supplied level is zero, with 999 potential differences by default.
        innovations = torch.randn((groups, periods), dtype=torch.float64, generator=rng)
        common = torch.randn(periods, dtype=torch.float64, generator=rng)
        loadings = torch.linspace(.8, 1.2, groups, dtype=torch.float64)
        innovations += loadings[:, None] * common
        innovations[:, 0] = 0
        levels = innovations.cumsum(dim=1)
        unit_root_data = pd.DataFrame({"id": ids.numpy(), "t": dates.numpy(), "y": levels.flatten().numpy()})
        start = perf_counter()
        panel_test = cips(data=unit_root_data, y="y", panel="id", time="t", lags=1,
                          trend="constant", inference="none", block_size=block_size, memory_mb=memory_mb)
        cips_seconds = perf_counter() - start
        assert panel_test.attrs["nobs"] == groups * periods
        assert panel_test.attrs["n_panels"] == groups and panel_test.attrs["periods"] == periods
        assert panel_test.attrs["balanced"] is True and panel_test.attrs["inference"] == "none"
        assert panel_test.attrs["critical_T"] == periods - 1
        assert panel_test.attrs["p_value"] is None
        assert panel_test.filter(regex="critical_|reject_").isna().all().all()
        assert panel_test.statistic.map(lambda number: float("-inf") < number < float("inf")).all()
        assert panel_test.attrs["workspace"]["algorithm"] == "batched_scaled_householder_qr"
        assert panel_test.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= memory_mb * 1024**2
        cips_record = {
            "public_api": "openecon.xtcips", "rows": len(unit_root_data), "groups": groups,
            "periods": periods, "lags": 1, "trend": "constant", "seconds": cips_seconds,
            "dgp": {"model": "one-factor unit-root null with iid Gaussian common and individual innovations",
                    "loading_min": float(loadings.min()), "loading_max": float(loadings.max()),
                    "initial_level": 0},
            "statistics": {name: float(panel_test.loc[name, "statistic"]) for name in panel_test.index},
            "inference": "none", "p_value": None, "critical_values": None, "rejection_decisions": None,
            "guards": {"complete_common_dates": True, "consecutive_integer_periods": True,
                       "full_rank_augmented_regressions": True, "actual_critical_T": periods - 1,
                       "regression_rows_per_unit": periods - 2, "parameters_per_unit": 6,
                       "published_table_range_N_and_T": [10, 200],
                       "outside_published_table_range": groups > 200 or periods - 1 > 200 or groups < 10 or periods - 1 < 10,
                       "no_extrapolated_inference": True},
            "truncated_panels": panel_test.attrs["truncated_panels"],
            "workspace": panel_test.attrs["workspace"], "provenance": panel_test.attrs["provenance"],
            "rss_after_operation": _rss(),
        }
    source_after = _sources()
    if source_before != source_after:
        raise RuntimeError("Loaded OpenEcon source changed during the run; these timings have no stable source snapshot.")
    forbidden_loaded = sorted(name for name in sys.modules if _forbidden(name))
    assert not forbidden_loaded
    return {
        "schema_version": 1, "status": "passed", "seed": seed,
        "environment": {"platform": platform.system(), "architecture": platform.machine(),
                        "python": platform.python_version(), "torch": str(torch.__version__), "pandas": pd.__version__,
                        "torch_cpu_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "device": "cpu", "dtype": "float64"},
        "native_runtime": {"blocked_import_prefixes": list(_FORBIDDEN), "blocked_import_attempts": blocker.attempts,
                           "forbidden_modules_loaded": forbidden_loaded},
        "source_snapshot": {"kind": "working-tree bytes of this benchmark and all loaded OpenEcon Python modules",
                            "claim": "No commit is assigned to uncommitted source. The source hashes matched before and after the operations.",
                            "file_count": len(source_before), "sha256": source_before,
                            "unchanged_after_measurement": True},
        "homogeneity": homogeneity_record, "cips": cips_record,
        "timing_scope": "Single warm complete public API call per procedure, including validation, input preparation and kernels. Imports, input generation, small warm-up calls and assertions are excluded.",
        "memory_scope": "RSS is a cumulative process high-water mark. Procedure memory_mb values bound estimates of additional tensor workspaces, excluding in-memory inputs, indexing, allocator and private library overhead.",
        "limits": "Local synthetic CPU observations; no universal speed or row-count guarantee, finite-sample calibration, Stata comparison, GPU, streaming or all-estimator fit claim. CIPS statistics outside published quantile support have no inference.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--periods", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--memory-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(groups=args.groups, periods=args.periods, threads=args.threads,
                 block_size=args.block_size, memory_mb=args.memory_mb, seed=args.seed)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps({"status": result["status"], "rows_per_procedure": args.groups * args.periods,
                          "homogeneity_seconds": result["homogeneity"]["seconds"],
                          "cips_seconds": result["cips"]["seconds"],
                          "peak_process_rss_bytes": result["cips"]["rss_after_operation"]["peak_process_rss_bytes"],
                          "source_files": result["source_snapshot"]["file_count"],
                          "source_unchanged": result["source_snapshot"]["unchanged_after_measurement"],
                          "blocked_import_attempts": result["native_runtime"]["blocked_import_attempts"]}, indent=2))
    else:
        print(serialized, end="")
