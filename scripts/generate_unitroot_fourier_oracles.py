"""Seeded, bounded Monte Carlo quantiles for the Fourier and ERS point-optimal tests.

Purpose
-------
1. Corroborate every transcribed critical-value table used by
   ``openecon.econometrics.unitroot.fourier`` and ``.ers`` against an
   independent NumPy simulation of the null distribution (``tests/
   test_econ_unitroot_fourier.py`` and ``test_econ_unitroot_ers.py`` compare
   each published entry with these quantiles within Monte Carlo tolerance).
2. Supply the entries that are not taken from a verified publication:
   Fourier KPSS critical values for frequencies k = 4, 5 and the supremum-F
   critical values of the Fourier KPSS linearity pretest (see the module
   docstring of ``unitroot/fourier_tables.py`` for why).

Designs (all under the null, iid N(0, 1) innovations, no augmentation)
----------------------------------------------------------------------
fourier_adf   y_t random walk (y_1 = e_1); Enders & Lee (2012, Econ. Letters)
              regression (3) with constant (or constant and trend), sin/cos of
              frequency k = 1..5; tau = t ratio of y_{t-1}; F(k) = test of the
              two Fourier terms; supF = max_k F(k).
fourier_kpss  y_t = e_t; Becker, Enders & Lee (2006) regression (3a)/(3b);
              tau(k) = sum S_t^2 / (T^2 sigma^2) with sigma^2 = SSR/T (the
              Bartlett estimator with zero lags, the iid design of the paper);
              F(k) of eq. (7) with SSR1/(T - q); supF = max over k = 1..5.
fourier_lm    y_t random walk; Enders & Lee (2012, OBES) eqs. (8)-(10): first
              differences regressed on [1, dsin, dcos], S~ detrended with these
              coefficients, tau_LM = t ratio of S~_{t-1} in Dy on
              [S~_{t-1}, 1, dsin, dcos]. F(k) as in the authors' published
              simulation program (time-series.net, table1critvalues.PRG):
              Dy on [S^L_{t-1}, 1, dsin, dcos] with S^L the linearly detrended
              LM series, testing dsin = dcos = 0. supF = max_k F(k).
ers_pt        y_t random walk; ERS P_T(0.5) with cbar = -7 (constant) and
              -13.5 (trend) and omega^2 = S(1)/T, exactly the design behind
              Table 1 of Elliott, Rothenberg & Stock (NBER TWP 130, 1992).

Monte Carlo standard errors are distribution free: half the width of the 95%
order-statistic interval of each quantile divided by 1.96.

Run (single thread, about one minute on an Apple M-series CPU):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/generate_unitroot_fourier_oracles.py \
        tests/fixtures/unitroot_fourier/simulated_quantiles.json
"""

from __future__ import annotations

import datetime
import json
import platform
import sys
import time

import numpy as np

SEED = 20261010
CHUNK = 5000
LOWER = (0.01, 0.025, 0.05, 0.10)
UPPER = (0.99, 0.975, 0.95, 0.90)
LEVELS = ("1%", "2.5%", "5%", "10%")


def projector(design: np.ndarray):
    q, _ = np.linalg.qr(design)
    return lambda a: a - (a @ q) @ q.T


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


def df_regression(dy, lag, design):
    """t ratio and SSR of lag in dy = design + rho lag (rows are replications)."""
    m = projector(design)
    a, b = m(dy), m(lag)
    bb = (b * b).sum(1)
    rho = (a * b).sum(1) / bb
    ssr = (a * a).sum(1) - rho ** 2 * bb
    df = dy.shape[1] - design.shape[1] - 1
    return rho / np.sqrt(ssr / df / bb), ssr, df


def fourier_adf(rng, total, case, reps):
    t = np.arange(2, total + 1, dtype=float)
    base = [np.ones(total - 1)] + ([t / total] if case == "trend" else [])
    taus = {k: [] for k in range(1, 6)}
    sup = []
    for start in range(0, reps, CHUNK):
        size = min(CHUNK, reps - start)
        y = np.cumsum(rng.standard_normal((size, total)), axis=1)
        dy, lag = np.diff(y, axis=1), y[:, :-1]
        _, ssr_r, _ = df_regression(dy, lag, np.column_stack(base))
        fstats = []
        for k in range(1, 6):
            design = np.column_stack(base + [np.sin(2 * np.pi * k * t / total),
                                             np.cos(2 * np.pi * k * t / total)])
            tau, ssr_u, df_u = df_regression(dy, lag, design)
            taus[k].append(tau)
            fstats.append(((ssr_r - ssr_u) / 2) / (ssr_u / df_u))
        sup.append(np.max(fstats, axis=0))
    return ({str(k): quantiles(np.concatenate(v), LOWER) for k, v in taus.items()},
            quantiles(np.concatenate(sup), UPPER))


def fourier_kpss(rng, total, case, reps):
    t = np.arange(1, total + 1, dtype=float)
    base = [np.ones(total)] + ([t / total] if case == "trend" else [])
    q = len(base) + 2
    taus = {k: [] for k in range(1, 6)}
    sup = []
    restricted = projector(np.column_stack(base))
    projections = {k: projector(np.column_stack(base + [np.sin(2 * np.pi * k * t / total),
                                                        np.cos(2 * np.pi * k * t / total)]))
                   for k in range(1, 6)}
    for start in range(0, reps, CHUNK):
        size = min(CHUNK, reps - start)
        y = rng.standard_normal((size, total))
        e0 = restricted(y)
        ssr0 = (e0 * e0).sum(1)
        fstats = []
        for k in range(1, 6):
            e = projections[k](y)
            ssr = (e * e).sum(1)
            partial = np.cumsum(e, axis=1)
            taus[k].append((partial * partial).sum(1) / total ** 2 / (ssr / total))
            fstats.append(((ssr0 - ssr) / 2) / (ssr / (total - q)))
        sup.append(np.max(fstats, axis=0))
    return ({str(k): quantiles(np.concatenate(v), UPPER) for k, v in taus.items()},
            quantiles(np.concatenate(sup), UPPER))


def fourier_lm(rng, total, reps):
    t = np.arange(1, total + 1, dtype=float)
    one = np.ones(total - 1)
    taus = {k: [] for k in range(1, 6)}
    fixed = {k: [] for k in range(1, 6)}
    sup = []
    for start in range(0, reps, CHUNK):
        size = min(CHUNK, reps - start)
        y = np.cumsum(rng.standard_normal((size, total)), axis=1)
        dy = np.diff(y, axis=1)
        linear = y - y[:, :1] - dy.mean(1)[:, None] * (t - 1)
        _, ssr_r, _ = df_regression(dy, linear[:, :-1], one[:, None])
        fstats = []
        for k in range(1, 6):
            sine, cosine = np.sin(2 * np.pi * k * t / total), np.cos(2 * np.pi * k * t / total)
            design = np.column_stack([one, np.diff(sine), np.diff(cosine)])
            coef = np.linalg.lstsq(design, dy.T, rcond=None)[0]
            detrended = y - (np.outer(coef[0], t) + np.outer(coef[1], sine)
                             + np.outer(coef[2], cosine))
            tau, _, _ = df_regression(dy, detrended[:, :-1], design)
            taus[k].append(tau)
            _, ssr_u, df_u = df_regression(dy, linear[:, :-1], design)
            stat = ((ssr_r - ssr_u) / 2) / (ssr_u / df_u)
            fixed[k].append(stat)
            fstats.append(stat)
        sup.append(np.max(fstats, axis=0))
    return ({str(k): quantiles(np.concatenate(v), LOWER) for k, v in taus.items()},
            {str(k): quantiles(np.concatenate(v), UPPER) for k, v in fixed.items()},
            quantiles(np.concatenate(sup), UPPER))


def ers_pt(rng, total, case, reps):
    t = np.arange(1, total + 1, dtype=float)
    z = np.ones((total, 1)) if case == "constant" else np.column_stack([np.ones(total), t / total])
    alpha = 1 + (-7.0 if case == "constant" else -13.5) / total

    def quasi(a):
        zq = z.copy()
        zq[1:] = z[1:] - a * z[:-1]
        return projector(zq)

    m_alpha, m_one = quasi(alpha), quasi(1.0)
    values = []
    for start in range(0, reps, CHUNK):
        size = min(CHUNK, reps - start)
        y = np.cumsum(rng.standard_normal((size, total)), axis=1)

        def ssr(m, a):
            yq = y.copy()
            yq[:, 1:] = y[:, 1:] - a * y[:, :-1]
            e = m(yq)
            return (e * e).sum(1)

        s_alpha, s_one = ssr(m_alpha, alpha), ssr(m_one, 1.0)
        values.append((s_alpha - alpha * s_one) / (s_one / total))
    return quantiles(np.concatenate(values), LOWER)


def main(path: str) -> None:
    started = time.time()
    rng = np.random.default_rng(SEED)
    out = {"meta": {"script": "scripts/generate_unitroot_fourier_oracles.py", "seed": SEED,
                    "generated": datetime.date.today().isoformat(), "numpy": np.__version__,
                    "python": platform.python_version(),
                    "innovations": "iid N(0,1); no augmentation",
                    "mc_se": "half-width of the 95% order-statistic interval / 1.96"},
           "fourier_adf": {}, "fourier_kpss": {}, "fourier_lm": {}, "ers_pt": {}}
    for case in ("constant", "trend"):
        for total in (100, 200, 500):
            taus, sup = fourier_adf(rng, total, case, 100_000)
            out["fourier_adf"][f"{case}/{total}"] = {"reps": 100_000, "tau": taus, "sup_f": sup}
            print("fourier_adf", case, total, round(time.time() - started), flush=True)
    for case in ("constant", "trend"):
        for total in (100, 500):
            taus, sup = fourier_kpss(rng, total, case, 200_000)
            out["fourier_kpss"][f"{case}/{total}"] = {"reps": 200_000, "tau": taus, "sup_f": sup}
            print("fourier_kpss", case, total, round(time.time() - started), flush=True)
    for total in (100, 200, 500):
        taus, fixed, sup = fourier_lm(rng, total, 100_000)
        out["fourier_lm"][str(total)] = {"reps": 100_000, "tau": taus, "f": fixed, "sup_f": sup}
        print("fourier_lm", total, round(time.time() - started), flush=True)
    for case in ("constant", "trend"):
        for total in (50, 100, 200, 500):
            out["ers_pt"][f"{case}/{total}"] = {"reps": 100_000,
                                                "p_t": ers_pt(rng, total, case, 100_000)}
            print("ers_pt", case, total, round(time.time() - started), flush=True)
    out["meta"]["seconds"] = round(time.time() - started, 1)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "simulated_quantiles.json")
