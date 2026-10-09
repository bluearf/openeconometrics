"""Representative native dependency/persistence proof and bounded size diagnostic."""

import argparse
import builtins
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import tempfile
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--size", action="store_true")
parser.add_argument("--bias", action="store_true")
parser.add_argument("--installed-root", type=Path)
args = parser.parse_args()
blocked = {"scipy", "statsmodels", "sklearn", "linearmodels"}

assert not any(name.split(".")[0] in blocked for name in sys.modules)
original_import = builtins.__import__
attempts = []


def guard(name, *a, **kw):
    if name.split(".")[0] in blocked:
        attempts.append(name)
        raise AssertionError("External estimator imported: " + name)
    return original_import(name, *a, **kw)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            raise AssertionError("External estimator imported: " + fullname)
        return None


builtins.__import__ = guard
sys.meta_path.insert(0, Finder())
torch.set_num_threads(1)
import openecon as oe  # noqa: E402

if args.installed_root:
    assert Path(oe.__file__).resolve().is_relative_to(args.installed_root.resolve())

receipt = {
    "issues": [35, 39, 40, 46, 52, 61, 62, 68],
    "guard": "builtins import + MetaPathFinder before OpenEcon import",
    "blocked_dependencies": sorted(blocked),
    "scope": "Representative resident CPU native execution plus complete JSON persistence; not licensed vendor execution or installed desktop validation",
}
with tempfile.TemporaryDirectory(prefix="eight-capability-proof-") as directory:
    previous = os.getcwd()
    try:
        os.chdir(directory)
        outputs = runpy.run_path(str(ROOT / "docs/examples/eight_capability_closures.py"))
        receipt["example"] = outputs["proof"]
        receipt["example"]["file"] = "owned temporary workspace; readback before disposal"
        receipt["state_sha256"] = hashlib.sha256(outputs["state_file"].read_bytes()).hexdigest()
    finally:
        os.chdir(previous)
if args.size:
    # Fixed x, independent Gaussian cluster intercepts + observation noise,
    # G=8 unequal sizes 6..13, null beta_x=0; no outcome-selected restrictions.
    generator = torch.Generator().manual_seed(99487)
    groups = 8
    codes = torch.repeat_interleave(torch.arange(groups), torch.arange(groups) + 6)
    x = torch.randn(len(codes), generator=generator, dtype=torch.float64)
    rejections = 0
    for trial in range(200):
        errors = 0.7 * torch.randn(groups, generator=generator, dtype=torch.float64)[
            codes
        ] + torch.randn(len(codes), generator=generator, dtype=torch.float64)
        df = pd.DataFrame(dict(y=(1 + errors).tolist(), x=x.tolist(), g=codes.tolist()))
        fit = oe.ols(data=df, y="y", x=["x"])
        test = oe.wild_cluster_test(fit, data=df, cluster="g", null={"x": 0.0}, enumerate_all=True)
        rejections += test.attrs["p_value"] <= 0.05
    trials = 200
    rate = rejections / trials
    z = 1.959963984540054
    center = (rate + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * ((rate * (1 - rate) / trials + z * z / (4 * trials * trials)) ** 0.5)
        / (1 + z * z / trials)
    )
    receipt["wild_size_diagnostic"] = {
        "seed": 99487,
        "trials": trials,
        "rejections": rejections,
        "rejection_rate": rate,
        "wilson_95": [center - half, center + half],
        "alpha": 0.05,
        "clusters": 8,
        "cluster_sizes": list(range(6, 14)),
        "draws_per_trial": 256,
        "complete_conditional_rademacher_enumeration": True,
        "failed_trials": 0,
        "domain": "one fixed exogenous design; independent Gaussian cluster intercept + idiosyncratic noise; bounded diagnostic, no universal few-cluster calibration",
    }
receipt["forbidden_import_attempts"] = attempts
if args.bias:
    from openecon.econometrics.robust.panel_resampling import _fits
    from openecon.engines.distributions import normal_isf

    generator = torch.Generator().manual_seed(89271)
    taus = [0.25, 0.5, 0.75]
    truth = torch.tensor([1.2 - 0.08 * normal_isf(tau) for tau in taus], dtype=torch.float64)
    cells = []
    for periods in [12, 24, 48]:
        pairs = []
        failures = []
        for trial in range(100):
            ids = torch.arange(8).repeat_interleave(periods)
            x = 0.5 * (torch.rand(len(ids), generator=generator, dtype=torch.float64) - 0.5)
            y = (
                0.3 * ids
                + 1.2 * x
                + (1 + 0.04 * ids + 0.08 * x)
                * torch.randn(len(ids), generator=generator, dtype=torch.float64)
            )
            df = pd.DataFrame(
                dict(id=ids.tolist(), time=list(range(periods)) * 8, x=x.tolist(), y=y.tolist())
            )
            try:
                fit = oe.panel_mmqr(data=df, y="y", x=["x"], panel="id", time="time")
                corrected, states = _fits(fit.spec, df, True)
                pairs.append(
                    torch.stack(
                        (
                            torch.tensor(
                                [c.estimate for c in fit.coefficients], dtype=torch.float64
                            ),
                            corrected,
                        )
                    )
                )
            except oe.AnalysisError as exc:
                failures.append({"trial": trial, "code": exc.code})
        if not pairs:
            raise AssertionError("All panel bias fixtures failed")
        errors = torch.stack(pairs) - truth
        cells.append(
            {
                "periods": periods,
                "attempted": 100,
                "valid": len(pairs),
                "failed": failures,
                "raw_bias": errors[:, 0].mean(0).tolist(),
                "split_bias": errors[:, 1].mean(0).tolist(),
                "raw_bias_mc_se": (errors[:, 0].std(0) / len(pairs) ** 0.5).tolist(),
                "split_bias_mc_se": (errors[:, 1].std(0) / len(pairs) ** 0.5).tolist(),
            }
        )
    receipt["panel_bias_diagnostic"] = {
        "seed": 89271,
        "quantiles": taus,
        "true_slopes": truth.tolist(),
        "cells": cells,
        "domain": "balanced stationary strictly exogenous Gaussian location-scale fixture; any failures explicitly counted; bias means conditional on admitted fits",
        "interpretation": "bounded small-T diagnostic; no universal SPJ improvement, fixed-T consistency, 1/T-expansion proof or nominal CI coverage claim",
    }
assert not attempts
paths = [
    *ROOT.glob("src/openecon/econometrics/*/saved_weak.py"),
    *ROOT.glob("src/openecon/econometrics/*/wild_restricted.py"),
    *ROOT.glob("src/openecon/econometrics/*/panel_resampling.py"),
    *ROOT.glob("src/openecon/econometrics/*/planning_extended.py"),
    ROOT / "src/openecon/econometrics/stats/planning_scenarios.py",
    ROOT / "src/openecon/econometrics/spatial/iv.py",
    ROOT / "src/openecon/econometrics/unitroot/fisher_johansen.py",
    ROOT / "src/openecon/econometrics/summary_state.py",
    ROOT / "src/openecon/econometrics/resident_cpu.py",
    ROOT / "docs/examples/eight_capability_closures.py",
]
receipt["source_sha256"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
}
if args.installed_root:
    import importlib

    verified = {}
    for path in paths:
        if not path.is_relative_to(ROOT / "src"):
            continue
        name = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        module = importlib.import_module(name)
        installed = Path(module.__file__).resolve()
        assert installed.is_relative_to(args.installed_root.resolve())
        assert installed.read_bytes() == path.read_bytes(), name
        verified[name] = str(installed)
    receipt["installed_module_origins_equal_source"] = verified
output = (
    ROOT
    / "docs/evidence/eight-capability-closures-2026-10-07"
    / ("wheel-runtime.json" if args.installed_root else "runtime.json")
)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt.get("wild_size_diagnostic", {}), sort_keys=True))
print(str(output))
