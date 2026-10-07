"""Bounded synthetic CPU timings for PANIC, Toda--Yamamoto and saved hetprobit.

One warmed full public API call is timed per operation, including validation
and table preparation. Imports, synthetic data generation, training and small
warm-ups are excluded. The observations do not calibrate inferential size or
establish arbitrary-data/GPU/out-of-core throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

from benchmarks.econometrics_wave4 import _FORBIDDEN, _NativeOnly, _forbidden, _rss, _sources


def run(*, rows: int = 100_000, groups: int = 200, periods: int = 300,
        threads: int = 2, seed: int = 20261005):
    if rows < 100 or groups < 10 or periods < 20 or threads < 1:
        raise ValueError("Use at least 100 evaluation rows, 10 panels, 20 periods and one thread.")
    if any(_forbidden(name) for name in sys.modules):
        raise RuntimeError("Run in a fresh process without external estimator modules.")
    blocker = _NativeOnly()
    sys.meta_path.insert(0, blocker)
    import pandas as pd
    import torch
    import openecon as oe
    from openecon.models import ResultBundle

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.device("cpu"), torch.no_grad():
        x = torch.randn((rows, 2), dtype=torch.float64, generator=generator)
        evaluated = pd.DataFrame({"x": x[:, 0].numpy(), "z": x[:, 1].numpy()})
        training = evaluated.iloc[:min(rows, 2000)].copy()
        noise = torch.randn(len(training), dtype=torch.float64, generator=generator)
        training["y"] = (.4 + .7 * x[:len(training), 0] + noise *
                         torch.exp(.2 * x[:len(training), 1] - .1 * x[:len(training), 0]) > 0).numpy().astype(int)
        fit = oe.hetprobit(data=training, y="y", x=["x"], het=["x", "z"])
        saved = ResultBundle.model_validate_json(fit.model_dump_json())
        before_saved = saved.model_dump_json()
        var_levels = torch.randn((rows, 3), dtype=torch.float64, generator=generator).cumsum(0)
        var_data = pd.DataFrame({f"v{j}": var_levels[:, j].numpy() for j in range(3)})
        increments = torch.randn((groups, periods), dtype=torch.float64, generator=generator)
        increments += torch.linspace(.8, 1.2, groups, dtype=torch.float64)[:, None] * \
            torch.randn(periods, dtype=torch.float64, generator=generator)
        levels = increments.cumsum(1)
        panel = pd.DataFrame({"id": torch.arange(groups).repeat_interleave(periods).numpy(),
                              "t": torch.arange(periods).repeat(groups).numpy(),
                              "y": levels.flatten().numpy()})
        oe.predict(saved, evaluated.iloc[:12], interval="mean")
        oe.margins(saved, ["x", "z"], data=evaluated.iloc[:12])
        oe.tycausality(data=var_data.iloc[:150], y=["v0", "v1", "v2"], lags=1, dmax=1)
        oe.xtpanic(panel.loc[(panel.id < 12) & (panel.t < 30)], "y", "id", "t", inference="none")
        source_before = _sources()
        own = Path(__file__).resolve()
        root = own.parents[1]
        source_before[own.relative_to(root).as_posix()] = hashlib.sha256(own.read_bytes()).hexdigest()
        operations = {}
        started = perf_counter()
        prediction = oe.predict(saved, evaluated, interval="mean")
        seconds = perf_counter() - started
        assert len(prediction) == rows and prediction.response.between(0, 1).all()
        assert prediction.std_error.ge(0).all() and prediction.notna().all().all()
        operations["hetprobit_prediction"] = {
            "public_api": "openecon.predict", "estimator": "hetprobit", "rows": rows,
            "seconds": seconds, "interval": "mean", "outcome_column_not_required": True,
            "covariance": "full saved mean/variance covariance", "rss_after_operation": _rss(),
        }
        started = perf_counter()
        margins = oe.margins(saved, ["x", "z"], data=evaluated)
        seconds = perf_counter() - started
        assert margins.variable.tolist() == ["x", "z"] and margins.std_error.ge(0).all()
        assert saved.model_dump_json() == before_saved
        operations["hetprobit_margins"] = {
            "public_api": "openecon.margins", "estimator": "hetprobit", "rows": rows,
            "seconds": seconds, "method": "ame", "mean_and_variance_overlap": True,
            "estimates": margins[["variable", "estimate", "std_error"]].to_dict("records"),
            "rss_after_operation": _rss(),
        }
        started = perf_counter()
        causality = oe.tycausality(data=var_data, y=["v0", "v1", "v2"], lags=1, dmax=1)
        seconds = perf_counter() - started
        assert len(causality["tests"]) == 9 and causality.attrs["observations"] == rows - 2
        assert causality["tests"].p_value.between(0, 1).all()
        operations["toda_yamamoto"] = {
            "public_api": "openecon.tycausality", "rows": rows, "variables": 3,
            "seconds": seconds, "base_lags": 1, "dmax": 1, "fitted_lags": 2,
            "first_base_lags_only": True, "covariance_divisor": causality.attrs["covariance_divisor"],
            "workspace_bytes_estimated": causality.attrs["estimated_workspace_bytes"],
            "test_results": causality["tests"].to_dict("records"), "rss_after_operation": _rss(),
        }
        started = perf_counter()
        panic = oe.xtpanic(panel, "y", "id", "t", factors=1, lags=1, inference="none")
        seconds = perf_counter() - started
        assert panic.attrs["nobs"] == groups * periods
        assert panic["idiosyncratic"].p_value.isna().all() and panic["common"].p_value.isna().all()
        assert panic.attrs["workspace"]["estimated_tensor_workspace_bytes"] <= 64 * 1024**2
        operations["panic"] = {
            "public_api": "openecon.xtpanic", "rows": groups * periods, "groups": groups,
            "periods": periods, "seconds": seconds, "factors": 1, "lags": 1,
            "inference": "none", "workspace": panic.attrs["workspace"],
            "rss_after_operation": _rss(),
        }
    source_after = _sources()
    source_after[own.relative_to(root).as_posix()] = hashlib.sha256(own.read_bytes()).hexdigest()
    assert source_before == source_after, "Source changed while measurements ran"
    assert not blocker.attempts and not any(_forbidden(name) for name in sys.modules)
    return {
        "schema_version": 1, "status": "passed", "seed": seed,
        "environment": {"platform": platform.system(), "architecture": platform.machine(),
                        "python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "dtype": "float64", "torch_threads": threads},
        "native_runtime": {"blocked_import_prefixes": list(_FORBIDDEN), "blocked_import_attempts": blocker.attempts},
        "source_snapshot": {"sha256": source_before, "unchanged_during_measurements": True,
                            "commit_assigned_to_uncommitted_source": False},
        "operations": operations,
        "timing_scope": "Single warm complete public API call per operation; includes validation/encoding/results, excludes imports/input construction/training/warm-ups.",
        "limitations": "Synthetic local CPU observations only. No finite-sample calibration, Stata parity, streaming/GPU or arbitrary-row-count guarantee. PANIC exact dense PCA has explicit memory and work limits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--groups", type=int, default=200)
    parser.add_argument("--periods", type=int, default=300)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = run(rows=args.rows, groups=args.groups, periods=args.periods, threads=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "operations": {
        name: {key: record[key] for key in ("rows", "seconds")}
        for name, record in result["operations"].items()}}))


if __name__ == "__main__":
    main()
