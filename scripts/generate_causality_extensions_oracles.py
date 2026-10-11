"""Seeded, bounded null-size and power experiments for the bootstrap causality extensions.

Runs the public OpenEcon procedures on simulated data and records, for every
cell, the planned and completed number of datasets, failed calls, rejection
counts at 5% and Wilson 95% intervals. Nothing is dropped: a failed call is
counted and reported. Output: tests/fixtures/causality_extensions/size_power.json.

Run (about 3-6 minutes on one CPU thread):

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
        python scripts/generate_causality_extensions_oracles.py

This is descriptive Monte Carlo evidence for the declared designs only, not a
proof of finite-sample validity for every data-generating process.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.var.asymmetric import asymcausality
from openecon.econometrics.var.fourier_causality import fouriertycausality
from openecon.econometrics.var.panel_bootstrap_causality import ekcausality, konyacausality
from openecon.econometrics.var.rolling_causality import rollingcausality

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "causality_extensions" / "size_power.json"
ALPHA = 0.05
DRAWS = 199


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [centre - half, centre + half]


def panel(rng, units, periods, rho, causal, integrated=False):
    cov = rho * np.ones((units, units)) + (1 - rho) * np.eye(units)
    ch = np.linalg.cholesky(cov)
    burn = 50
    e = rng.normal(size=(periods + burn, units)) @ ch.T
    v = rng.normal(size=(periods + burn, units)) @ ch.T
    yv = np.zeros((periods + burn, units))
    xv = np.zeros((periods + burn, units))
    ar = 1.0 if integrated else .5
    for t in range(1, periods + burn):
        xv[t] = .2 * (1 - ar) + ar * xv[t - 1] + v[t]
        yv[t] = .3 + .4 * yv[t - 1] + causal * (np.arange(units) % 2) * xv[t - 1] + e[t]
    frame = pd.DataFrame({"y": yv[-periods:].T.ravel(), "x": xv[-periods:].T.ravel()})
    frame["unit"] = np.repeat(np.arange(units), periods)
    frame["t"] = np.tile(np.arange(periods), units)
    return frame


def fourier_series(rng, n, causal):
    v = np.zeros((n, 2))
    e = rng.normal(size=(n, 2))
    for t in range(1, n):
        v[t, 1] = v[t - 1, 1] + e[t, 1]
        v[t, 0] = 1.5 * math.sin(2 * math.pi * t / n) + .5 * v[t - 1, 0] + causal * v[t - 1, 1] + e[t, 0]
    return pd.DataFrame(v, columns=["y", "x"])


def stationary_pair(rng, n, causal):
    v = np.zeros((n + 50, 2))
    e = rng.normal(size=(n + 50, 2))
    for t in range(1, n + 50):
        v[t, 1] = .5 * v[t - 1, 1] + e[t, 1]
        v[t, 0] = .4 * v[t - 1, 0] + causal * v[t - 1, 1] + e[t, 0]
    return pd.DataFrame(v[50:], columns=["y", "x"])


def run_cell(name, replications, seed, make, test, extract):
    rng = np.random.default_rng(seed)
    counts, total, failures, values = 0, 0, [], []
    start = time.time()
    for r in range(replications):
        data = make(rng)
        try:
            result = test(data, r)
        except AnalysisError as exc:
            failures.append({"replication": r, "code": exc.code})
            continue
        decisions = extract(result)
        counts += int(sum(decisions))
        total += len(decisions)
        values.append(int(sum(decisions)))
    cell = {"cell": name, "planned_datasets": replications, "completed_datasets": replications - len(failures),
            "failed_calls": failures, "tests": total, "rejections": counts,
            "rate": counts / total if total else None,
            "wilson95": wilson(counts, total) if total else None, "seed": seed,
            "bootstrap": DRAWS, "alpha": ALPHA, "seconds": round(time.time() - start, 2)}
    print(json.dumps({k: cell[k] for k in ("cell", "tests", "rejections", "rate", "wilson95", "seconds")}))
    return cell


def main() -> None:
    torch.set_num_threads(1)
    cells = []
    for causal, label in ((0.0, "null"), (0.5, "power")):
        cells.append(run_cell(
            f"konya_{label}_N5_T50_rho0.5", 300, 7101 if causal == 0 else 7102,
            lambda rng, c=causal: panel(rng, 5, 50, .5, c),
            lambda data, r: konyacausality(data=data, y="y", x="x", panel="unit", time="t", lags=1,
                                           direction="x_to_y", bootstrap=DRAWS, seed=r),
            # Null: all units; power: causal (odd) units only.
            (lambda res: list(res["tests"].bootstrap_p_value <= ALPHA)) if causal == 0 else
            (lambda res: list(res["tests"].bootstrap_p_value.iloc[[1, 3]] <= ALPHA))))
    for causal, label in ((0.0, "null"), (0.3, "power")):
        cells.append(run_cell(
            f"ek_bootstrap_{label}_N5_T50_rho0.5_I1x", 400, 7201 if causal == 0 else 7202,
            lambda rng, c=causal: panel(rng, 5, 50, .5, c, integrated=True),
            lambda data, r: ekcausality(data=data, y="y", x="x", panel="unit", time="t", lags=1, dmax=1,
                                        direction="x_to_y", bootstrap=DRAWS, seed=r),
            lambda res: [bool(res["tests"].bootstrap_p_value.iloc[0] <= ALPHA)]))
    cells.append(run_cell(
        "ek_asymptotic_fisher_null_N5_T50_rho0.5_I1x", 400, 7201,
        lambda rng: panel(rng, 5, 50, .5, 0.0, integrated=True),
        lambda data, r: ekcausality(data=data, y="y", x="x", panel="unit", time="t", lags=1, dmax=1,
                                    direction="x_to_y", bootstrap=99, seed=r),
        lambda res: [bool(res["tests"].asymptotic_p_value.iloc[0] <= ALPHA)]))
    for causal, label in ((0.0, "null"), (0.3, "power")):
        cells.append(run_cell(
            f"fourier_ty_{label}_n100_selected", 400, 7301 if causal == 0 else 7302,
            lambda rng, c=causal: fourier_series(rng, 100, c),
            lambda data, r: fouriertycausality(data=data, y=["y", "x"], maxlag=4, kmax=3, dmax=1,
                                               effects=["y"], causes=["x"], bootstrap=DRAWS, seed=r),
            lambda res: [bool(res["tests"].bootstrap_p_value.iloc[0] <= ALPHA)]))
    for causal, label in ((0.0, "null"), (0.3, "power")):
        cells.append(run_cell(
            f"rolling_single_window_{label}_n80_p1", 400, 7401 if causal == 0 else 7402,
            lambda rng, c=causal: stationary_pair(rng, 80, c),
            lambda data, r: rollingcausality(data=data, y="y", x="x", window=80, lags=1,
                                             bootstrap=DRAWS, seed=r),
            lambda res: [bool(res["windows"].bootstrap_p_x_to_y.iloc[0] <= ALPHA)]))
    cells.append(run_cell(
        "asym_leverage_null_independent_random_walks_n100", 400, 7501,
        lambda rng: pd.DataFrame({"y": rng.normal(size=100).cumsum(), "x": rng.normal(size=100).cumsum()}),
        lambda data, r: asymcausality(data=data, y="y", x="x", lags=1, dmax=1, bootstrap=DRAWS, seed=r,
                                      resampling="leverage", max_work=10**12),
        lambda res: [bool(res["tests"].statistic.iloc[0] > res["tests"].cv_5.iloc[0])]))
    record = {"generated": dt.date.today().isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "numpy": np.__version__, "threads": 1,
              "note": "Descriptive bounded Monte Carlo for the declared designs; failures retained.",
              "cells": cells}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
