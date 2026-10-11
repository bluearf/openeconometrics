"""Seeded, bounded Monte Carlo critical values of the HEGY seasonal unit-root tests.

Why a simulation
----------------
OpenEconometrics found no HEGY critical-value source that replicates:

* Hylleberg, Engle, Granger and Yoo (1990, J. Econometrics 44, Table 1a/1b),
  as transcribed in R's uroot 2.1-3 (``.HEGY.orig.cv``), gives a 5% point of
  t(pi_1) with constant and seasonal dummies of -2.95 at T = 100 and -3.08 at
  T = 48. This simulation gives -2.83 and -2.79 for the same statistic and
  uroot's own ``hegy.test`` (R ``lm``, 3,000 replications) gives -2.85 and
  -2.76.
* The Diaz-Emparanza (2014, CSDA 76) response surfaces ported in uroot
  (``.HEGY.CM.tabs``, ``hegy.rs.pvalue``) reject 7.6% (F_3:4) and 7.7%
  (F_2:4) of 3,000 simulated null series at the nominal 5% level (T = 100,
  constant + seasonal dummies), and one surface serves every number of seasons
  although the limiting distributions of the joint F tests depend on it.

``scripts/generate_unitroot_hegy_oracles.R`` records those comparisons; the
tests repeat them from its fixture.

Design
------
x_t = x_{t-S} + e_t, e_t iid N(0, 1), x_t = 0 for t <= 0 (the seasonal random
walk null), regression on t = S+1..T without augmentation:

    Delta_S x_t = sum_i pi_i y_{i,t-1} + deterministic + error

with y_1 = (1 + L + ... + L^{S-1}) x, y_2 = -(1 - L + L^2 - ... - L^{S-1}) x
and, for each harmonic j = 1..S/2-1, two filters spanning frequency 2 pi j/S
(for S = 4 the explicit HEGY 1990 eq. (3.6) filters -(1 - L^2) x_{t-2},
-(1 - L^2) x_{t-1}). Statistics: t ratios of pi_1 and pi_2, the F test of each
harmonic pair, F of pi_2..pi_S and F of pi_1..pi_S. Deterministic cases:
none, constant, constant + trend, constant + S - 1 seasonal dummies, and
constant + seasonal dummies + trend. Pair-F quantiles of the monthly case pool
the five harmonic pairs (their null distributions coincide; the tests compare
them pair by pair).

N is the number of regression observations (T - S). Quantiles are reported
at 1%, 2.5%, 5%, 10% (lower tail for t, upper tail for F) with a
distribution-free Monte Carlo standard error (95% order-statistic interval /
3.92).

Run (single thread, about four minutes on an Apple M-series CPU):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/generate_unitroot_hegy_oracles.py \
        tests/fixtures/unitroot_hegy/simulated_critical_values.json \
        src/openecon/econometrics/unitroot/hegy_tables.py
"""

from __future__ import annotations

import datetime
import json
import platform
import pprint
import sys
import time

import numpy as np

SEED = 19900101
LOWER = (0.01, 0.025, 0.05, 0.10)
UPPER = (0.99, 0.975, 0.95, 0.90)
LEVELS = ("1%", "2.5%", "5%", "10%")
CASES = ("none", "constant", "trend", "seasonal", "seasonal_trend")
GRID = {4: {"nodes": (40, 80, 160, 320, 1280), "reps": 50_000, "chunk": 2_500},
        12: {"nodes": (120, 240, 480, 1200), "reps": 20_000, "chunk": 1_000}}


def filters(seasons: int) -> np.ndarray:
    """[S, S] coefficients on x_{t-1-j}, j = 0..S-1, of the lagged HEGY regressors."""
    if seasons == 4:
        return np.array([[1.0, 1.0, 1.0, 1.0],          # y1_{t-1}
                         [-1.0, 1.0, -1.0, 1.0],        # y2_{t-1}
                         [0.0, -1.0, 0.0, 1.0],         # y3_{t-2} = -(x_{t-2} - x_{t-4})
                         [-1.0, 0.0, 1.0, 0.0]])        # y3_{t-1} = -(x_{t-1} - x_{t-3})
    lag = np.arange(seasons)
    rows = [np.ones(seasons), (-1.0) ** (lag + 1)]
    for harmonic in range(1, seasons // 2):
        omega = 2.0 * np.pi * harmonic / seasons
        rows.append(np.cos((lag + 1) * omega))
        rows.append(np.sin((lag + 1) * omega))
    return np.array(rows)


def deterministic(n: int, seasons: int, case: str) -> np.ndarray:
    columns = []
    if case != "none":
        columns.append(np.ones(n))
    if case in ("trend", "seasonal_trend"):
        columns.append(np.arange(1, n + 1) / n)
    if case in ("seasonal", "seasonal_trend"):
        for season in range(1, seasons):
            columns.append((np.arange(n) % seasons == season).astype(float))
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def statistics(rng, seasons: int, n: int, case: str, reps: int, chunk: int) -> dict:
    total = n + seasons
    coef = filters(seasons)
    design = deterministic(n, seasons, case)
    q = np.linalg.qr(design)[0] if design.shape[1] else None
    df = n - design.shape[1] - seasons
    pairs = [[2 + 2 * j, 3 + 2 * j] for j in range(seasons // 2 - 1)]
    out = {"t_1": [], "t_2": [], "pair": [], "F_seasonal": [], "F_all": []}
    by_pair = {f"F_{a + 1}_{b + 1}": [] for a, b in pairs}
    for start in range(0, reps, chunk):
        size = min(chunk, reps - start)
        e = rng.standard_normal((size, total))
        x = np.empty_like(e)
        for season in range(seasons):
            x[:, season::seasons] = np.cumsum(e[:, season::seasons], axis=1)
        target = x[:, seasons:] - x[:, :-seasons]
        lags = np.stack([x[:, seasons - 1 - j:total - 1 - j] for j in range(seasons)], axis=2)
        h = lags @ coef.T
        if q is not None:
            target = target - (target @ q) @ q.T
            h = h - np.einsum("nk,rkj->rnj", q, np.einsum("nk,rnj->rkj", q, h))
        xtx = np.einsum("rni,rnj->rij", h, h)
        xty = np.einsum("rni,rn->ri", h, target)
        inverse = np.linalg.inv(xtx)
        beta = np.einsum("rij,rj->ri", inverse, xty)
        ssr = (target * target).sum(1) - (beta * xty).sum(1)
        s2 = ssr / df

        def wald(index):
            b = beta[:, index]
            v = inverse[:, index][:, :, index]
            return np.einsum("ri,rij,rj->r", b, np.linalg.inv(v), b) / len(index) / s2

        out["t_1"].append(beta[:, 0] / np.sqrt(s2 * inverse[:, 0, 0]))
        out["t_2"].append(beta[:, 1] / np.sqrt(s2 * inverse[:, 1, 1]))
        for (a, b) in pairs:
            value = wald([a, b])
            out["pair"].append(value)
            by_pair[f"F_{a + 1}_{b + 1}"].append(value)
        out["F_seasonal"].append(wald(list(range(1, seasons))))
        out["F_all"].append(wald(list(range(seasons))))
    result = {"df_resid": df}
    for name, values in out.items():
        result[name] = quantiles(np.concatenate(values), LOWER if name.startswith("t") else UPPER)
    if seasons > 4:
        result["pairs"] = {name: quantiles(np.concatenate(values), UPPER)
                           for name, values in by_pair.items()}
    return result


def quantiles(values: np.ndarray, probabilities) -> dict:
    values = np.sort(values)
    n = values.shape[0]
    out = {}
    for level, p in zip(LEVELS, probabilities, strict=True):
        half = 1.96 * np.sqrt(n * p * (1 - p))
        low = int(np.clip(np.floor(n * p - half), 0, n - 1))
        high = int(np.clip(np.ceil(n * p + half), 0, n - 1))
        out[level] = {"value": float(np.quantile(values, p)),
                      "mc_se": float((values[high] - values[low]) / (2 * 1.96))}
    return out


MODULE = '''"""Simulated critical values of the HEGY seasonal unit-root tests (generated file).

Generated by ``scripts/generate_unitroot_hegy_oracles.py`` (seed {seed}, {date},
NumPy {numpy}); the full quantiles with Monte Carlo standard errors are in
``tests/fixtures/unitroot_hegy/simulated_critical_values.json``. Do not edit by
hand: rerun the script. Design: seasonal random walk null with iid N(0, 1)
innovations and zero initial values, no augmentation; quarterly {reps4:,} and
monthly {reps12:,} replications per cell (monthly pair-F quantiles pool the five
harmonic pairs). The published HEGY (1990) table and the Diaz-Emparanza (2014)
response surfaces did not replicate (see the script docstring).

NODES[S] are the numbers of regression observations N = T - S - lags at which
the quantiles were simulated; CRITICAL[S][case][statistic] holds one row
(1%, 2.5%, 5%, 10%) per node. t statistics reject below, F statistics above.
"""

NODES = {nodes}
CRITICAL = {critical}
'''


def main(path: str, module: str | None) -> None:
    started = time.time()
    rng = np.random.default_rng(SEED)
    out = {"meta": {"script": "scripts/generate_unitroot_hegy_oracles.py", "seed": SEED,
                    "generated": datetime.date.today().isoformat(), "numpy": np.__version__,
                    "python": platform.python_version(),
                    "design": "x_t = x_{t-S} + e_t, iid N(0,1), zero initial values, no lags",
                    "mc_se": "half-width of the 95% order-statistic interval / 1.96"}}
    for seasons, grid in GRID.items():
        block = {"nodes": list(grid["nodes"]), "reps": grid["reps"], "cases": {}}
        for case in CASES:
            block["cases"][case] = {}
            for n in grid["nodes"]:
                block["cases"][case][str(n)] = statistics(rng, seasons, n, case, grid["reps"],
                                                          grid["chunk"])
                print(seasons, case, n, round(time.time() - started), flush=True)
        out[str(seasons)] = block
    out["meta"]["seconds"] = round(time.time() - started, 1)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")
    if module:
        names = ("t_1", "t_2", "pair", "F_seasonal", "F_all")
        critical = {seasons: {case: {name: tuple(
            tuple(round(out[str(seasons)]["cases"][case][str(n)][name][level]["value"], 3)
                  for level in LEVELS) for n in GRID[seasons]["nodes"]) for name in names}
            for case in CASES} for seasons in GRID}
        nodes = {seasons: tuple(float(n) for n in GRID[seasons]["nodes"]) for seasons in GRID}
        with open(module, "w", encoding="utf-8") as handle:
            handle.write(MODULE.format(seed=SEED, date=out["meta"]["generated"],
                                       numpy=np.__version__, reps4=GRID[4]["reps"],
                                       reps12=GRID[12]["reps"],
                                       nodes=pprint.pformat(nodes, width=96),
                                       critical=pprint.pformat(critical, width=96,
                                                               compact=True)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
