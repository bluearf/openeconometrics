"""Reproducible native-runtime timing for the third econometrics addition.

Warm timings exclude imports and synthetic data construction. These small local
benchmarks are not throughput or inferential guarantees for arbitrary datasets.
Estimation-library imports are blocked, rather than used as runtime backends.
"""
from __future__ import annotations

import argparse
import importlib.abc
import json
from pathlib import Path
import platform
import sys
from time import perf_counter


class _NativeOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in (
            "statsmodels", "linearmodels", "sklearn", "scipy.optimize", "scipy.stats",
        )):
            raise ImportError(f"Benchmark requires a native OpenEcon implementation: {fullname}")
        return None


def run(*, groups=1000, periods=1000, rows=5000, replications=999, threads=2):
    sys.meta_path.insert(0, _NativeOnly())
    import pandas as pd
    import torch
    import openecon as oe
    from openecon.models import ResultBundle

    torch.set_num_threads(threads)
    rng = torch.Generator().manual_seed(20261005)
    residuals = torch.randn((groups, periods), dtype=torch.float64, generator=rng)
    residual_data = pd.DataFrame({
        "id": torch.arange(groups).repeat_interleave(periods).numpy(),
        "t": torch.arange(periods).repeat(groups).numpy(), "e": residuals.flatten().numpy(),
    })
    # Resolve the lazy entry before measuring its numerical operation.
    dependence = oe.xtcd
    start = perf_counter()
    cd = dependence(data=residual_data, residual="e", panel="id", time="t")
    cd_seconds = perf_counter() - start
    assert cd.attrs["n_pairs"] == groups * (groups - 1) // 2
    assert cd.attrs["workspace"]["algorithm"] == "balanced_linear_cd"
    assert cd.p_value.between(0, 1).all()

    x = torch.randn(rows, dtype=torch.float64, generator=rng).cumsum(0)
    changes = torch.diff(x, prepend=x[:1])
    outcome = .4 * changes.clamp_min(0).cumsum(0) + .2 * changes.clamp_max(0).cumsum(0)
    outcome += .2 * torch.randn(rows, dtype=torch.float64, generator=rng)
    time_data = pd.DataFrame({"t": torch.arange(rows).numpy(), "x": x.numpy(), "y": outcome.numpy()})
    fit = oe.nardl(data=time_data, y="y", x=["x"], time="t", lags=[2, 1])
    saved = ResultBundle.model_validate_json(fit.model_dump_json())
    multiplier = oe.nardl_multipliers
    # Resolve its lazy numerical module too, before the warm operation timing.
    import openecon.econometrics.tsmodels.nardl_bootstrap as bootstrap_runtime
    compile_start = perf_counter()
    bootstrap_runtime._native_values(
        torch.zeros(3, dtype=torch.float64), torch.zeros((1, 1), dtype=torch.float64),
        torch.tensor([.1, .1], dtype=torch.float64), 2,
    )
    filter_initialization_seconds = perf_counter() - compile_start
    start = perf_counter()
    bands = multiplier(saved, data=time_data, method="wild", replications=replications,
                       seed=20261005, steps=40, batch_size=64)
    bootstrap_seconds = perf_counter() - start
    assert bands.attrs["bootstrap"]["completed"] == replications
    assert bands.positive_ci_low.le(bands.positive_ci_high).all()
    latex = str(bands.to_latex(notes=bands.attrs["notes"]))
    assert "percentile" in latex and "\\toprule" in latex

    from openecon.econometrics.registry import all_estimators, public_exports
    result = {
        "schema_version": 1, "seed": 20261005,
        "environment": {"platform": platform.system(), "architecture": platform.machine(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "threads": threads, "device": "cpu", "precision": "float64"},
        "registry": {"estimators": len(all_estimators()), "public_exports": len(public_exports())},
        "native_runtime": {"blocked": ["statsmodels", "linearmodels", "sklearn", "scipy.optimize", "scipy.stats"],
                           "estimation_library_loaded": any(name.startswith(("statsmodels", "linearmodels", "sklearn", "scipy.optimize", "scipy.stats")) for name in sys.modules)},
        "balanced_cd": {"rows": len(residual_data), "groups": groups, "periods": periods,
                        "pairs": cd.attrs["n_pairs"], "seconds": cd_seconds,
                        "algorithm": cd.attrs["workspace"]["algorithm"],
                        "workspace": cd.attrs["workspace"], "statistic": float(cd.statistic.iloc[0]),
                        "p_value": float(cd.p_value.iloc[0])},
        "nardl_bootstrap": {"rows": rows, "replications": replications, "horizons": 41,
                            "seconds": bootstrap_seconds, "batch_used": bands.attrs["bootstrap"]["batch_size_used"],
                            "filter_initialization_seconds": filter_initialization_seconds,
                            "outcome_generation": bands.attrs["bootstrap"]["outcome_generation"],
                            "solver": bands.attrs["bootstrap"]["solver"],
                            "saved_json_result": True, "publication_latex_with_notes": True},
        "timing_scope": "Warm operations only; imports, data generation, NARDL original fit and native filter initialization excluded. Filter initialization reported separately.",
        "limits": "Local synthetic CPU benchmark, in-memory inputs, fixed lags, pointwise fixed-x bootstrap; no Stata parity, general coverage, GPU or out-of-core claim.",
    }
    assert result["native_runtime"]["estimation_library_loaded"] is False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--periods", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--replications", type=int, default=999)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(groups=args.groups, periods=args.periods, rows=args.rows,
                  replications=args.replications, threads=args.threads)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
