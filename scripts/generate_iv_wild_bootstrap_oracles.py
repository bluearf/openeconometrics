"""Seeded, bounded null-size and power cells for the IV WRE, multiway WCR and score bootstraps.

Runs the public OpenEcon procedures on simulated data and records planned and
completed datasets, failed calls, rejection counts at 5% and Wilson 95%
intervals, next to the conventional asymptotic decision on the same data.
Output: tests/fixtures/iv_wild_bootstrap/size_power.json.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
        python scripts/generate_iv_wild_bootstrap_oracles.py

Descriptive Monte Carlo evidence for the declared designs only.
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

import openecon as oe
from openecon.analysis_contracts import AnalysisError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "iv_wild_bootstrap" / "size_power.json"
ALPHA = 0.05
REPS = 199


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return [centre - half, centre + half]


def iv_frame(rng, groups=12, strength=.5, beta=1.0):
    sizes = np.round(np.geomspace(4, 60, groups)).astype(int)     # unequal clusters
    g = np.repeat(np.arange(groups), sizes)
    n = len(g)
    cg, cz = rng.normal(size=groups)[g], rng.normal(size=groups)[g]
    z1 = rng.normal(size=n) + cz
    z2, w = rng.normal(size=(2, n))
    v = rng.normal(size=n) * (1 + .5 * np.abs(z1)) + .7 * cg
    u = .8 * v + .6 * rng.normal(size=n) + .7 * cg
    y2 = strength * z1 + .5 * strength * z2 + .3 * w + v
    y1 = beta * y2 + .5 * w + u
    return pd.DataFrame(dict(y1=y1, y2=y2, z1=z1, z2=z2, w=w, g=g))


def multiway_frame(rng, beta=0.0):
    n = 400
    a, b = rng.integers(0, 12, n), rng.integers(0, 8, n)
    x = rng.normal(size=n) + rng.normal(size=12)[a] + rng.normal(size=8)[b]
    y = 1 + beta * x + rng.normal(size=12)[a] + rng.normal(size=8)[b] + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, x=x, w=rng.normal(size=n), a=a, b=b))


def binary_frame(rng, beta=0.0):
    groups = 15
    g = rng.integers(0, groups, 400)
    x1 = rng.normal(size=400) + rng.normal(size=groups)[g]
    x2 = rng.normal(size=400) + .5 * rng.normal(size=groups)[g]
    latent = .2 + .6 * x1 + beta * x2 + .8 * rng.normal(size=groups)[g] + rng.logistic(size=400)
    return pd.DataFrame(dict(d=(latent > 0).astype(float), x1=x1, x2=x2, g=g))


def run_cell(name, replications, seed, make, test):
    rng = np.random.default_rng(seed)
    boot = asym = 0
    failures = []
    start = time.time()
    for r in range(replications):
        data = make(rng)
        try:
            bootstrap_p, asymptotic_p = test(data, r)
        except AnalysisError as exc:
            failures.append({"replication": r, "code": exc.code})
            continue
        boot += bootstrap_p <= ALPHA
        asym += asymptotic_p <= ALPHA
    done = replications - len(failures)
    cell = {"cell": name, "planned_datasets": replications, "completed_datasets": done,
            "failed_calls": failures, "bootstrap_rejections": int(boot),
            "bootstrap_rate": boot / done, "bootstrap_wilson95": wilson(int(boot), done),
            "asymptotic_rejections": int(asym), "asymptotic_rate": asym / done,
            "asymptotic_wilson95": wilson(int(asym), done), "seed": seed, "reps": REPS,
            "alpha": ALPHA, "seconds": round(time.time() - start, 2)}
    print(json.dumps({k: cell[k] for k in ("cell", "bootstrap_rate", "asymptotic_rate", "seconds")}))
    return cell


def iv_test(method, null, cluster="g"):
    def run(data, r):
        row = oe.iv_wild_test(data=data, y="y1", endog="y2", instruments=["z1", "z2"], x=["w"],
                              null=null, method=method, cluster=cluster, reps=REPS, seed=r,
                              wild="webb")["test"].iloc[0]
        return row.bootstrap_p_value, row.asymptotic_p_value
    return run


def multiway_test(data, r):
    result = oe.ols(data=data, y="y", x=["x", "w"])
    row = oe.wild_multiway_test(result, data=data, null={"x": 0.0}, clusters=["a", "b"],
                                reps=REPS, seed=r, wild="webb")["test"].iloc[0]
    return row.bootstrap_p_value, row.asymptotic_p_value


def score_test(data, r):
    row = oe.score_bootstrap_test(data=data, y="d", x=["x1", "x2"], null={"x2": 0.0},
                                  cluster="g", reps=REPS, seed=r, wild="webb")["test"].iloc[0]
    return row.bootstrap_p_value, row.asymptotic_p_value


def main() -> None:
    torch.set_num_threads(1)
    cells = [
        run_cell("wre_wald_null_strong_G12_unequal", 400, 8101,
                 lambda rng: iv_frame(rng, strength=.6), iv_test("wald", 1.0)),
        run_cell("wre_wald_null_weak_G12_unequal", 400, 8102,
                 lambda rng: iv_frame(rng, strength=.12), iv_test("wald", 1.0)),
        run_cell("wre_ar_null_weak_G12_unequal", 400, 8103,
                 lambda rng: iv_frame(rng, strength=.12), iv_test("ar", 1.0)),
        run_cell("wre_wald_power_strong_G12_beta1.6", 400, 8104,
                 lambda rng: iv_frame(rng, strength=.6, beta=1.6), iv_test("wald", 1.0)),
        run_cell("multiway_wcr_null_G12_H8", 400, 8201, multiway_frame, multiway_test),
        run_cell("multiway_wcr_power_G12_H8_beta0.25", 400, 8202,
                 lambda rng: multiway_frame(rng, .25), multiway_test),
        run_cell("score_lm_logit_null_G15", 400, 8301, binary_frame, score_test),
        run_cell("score_lm_logit_power_G15_beta0.4", 400, 8302,
                 lambda rng: binary_frame(rng, .4), score_test),
    ]
    record = {"generated": dt.date.today().isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "numpy": np.__version__, "threads": 1,
              "note": "Descriptive bounded Monte Carlo for the declared designs; failures retained.",
              "cells": cells}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
