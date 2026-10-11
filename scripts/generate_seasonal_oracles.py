"""Independent NumPy references for the native seasonal / forecasting routines.

Development-only oracle (never imported at runtime). Run from the repository root:

    python scripts/generate_seasonal_oracles.py tests/fixtures/seasonal

It writes ``numpy_x11.json``: the classical X-11 method (Ladiray and Quenneville
2001; table logic as implemented in JDemetra+ 3, ``jdplus.x13.base.core.x11``),
written here as a plain loop-by-loop NumPy transcription that shares no code with
``openecon.econometrics.tsworkflows.x11`` (no tensors, no vectorized end handling,
explicit neighbour searches). Input series are generated from fixed seeds.
Further sections are appended by later stages of the same script.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

# ---- X-11 reference ----------------------------------------------------------------


def henderson(length):
    m = length // 2
    p = m + 2
    out = []
    for j in range(-m, m + 1):
        num = 315 * ((p - 1) ** 2 - j * j) * (p * p - j * j) * ((p + 1) ** 2 - j * j) * (3 * p * p - 16 - 11 * j * j)
        den = 8 * p * (p * p - 1) * (4 * p * p - 1) * (4 * p * p - 9) * (4 * p * p - 25)
        out.append(num / den)
    return out


def musgrave(h, ratio):
    """Musgrave (1964) end filters, classical X-11 formula; list indexed by missing points - 1."""
    n = len(h)
    d = 4.0 / (math.pi * ratio * ratio)
    filters = []
    for missing in range(1, n // 2 + 1):
        m = n - missing
        p1 = sum(h[j] for j in range(m, n)) / m
        p2 = sum(h[j] * ((j + 1) - (m + 1) / 2.0) for j in range(m, n))
        factor = d / (1 + m * (m - 1) * (m + 1) * d / 12.0)
        filters.append([h[i] + p1 + p2 * ((i + 1) - (m + 1) / 2.0) * factor for i in range(m)])
    return filters


def ratio_for(length, period):
    if period == 4:
        return 0.001 if length <= 5 else 4.5
    if length <= 9:
        return 1.0
    return 3.5 if length <= 13 else 4.5


SEASONAL = {
    "3x3": ([1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9],
            [[3 / 27, 7 / 27, 10 / 27, 7 / 27], [5 / 27, 11 / 27, 11 / 27]]),
    "3x5": ([1 / 15, 2 / 15, 3 / 15, 3 / 15, 3 / 15, 2 / 15, 1 / 15],
            [[4 / 60, 8 / 60, 13 / 60, 13 / 60, 13 / 60, 9 / 60],
             [4 / 60, 11 / 60, 15 / 60, 15 / 60, 15 / 60], [9 / 60, 17 / 60, 17 / 60, 17 / 60]]),
}


def apply_filter(x, sym, ends):
    """Symmetric filter inside; end filter k (k future points missing - 1) at both ends."""
    n = len(x)
    m = len(sym) // 2
    out = [math.nan] * n
    for t in range(m, n - m):
        out[t] = sum(sym[j] * x[t - m + j] for j in range(len(sym)))
    for k, w in enumerate(ends):
        t = n - m + k                      # right end: lags -m .. len(w)-m-1
        out[t] = sum(w[j] * x[t - m + j] for j in range(len(w)))
        t = m - 1 - k                      # left end: mirror image
        out[t] = sum(w[j] * x[t + m - j] for j in range(len(w)))
    return out


def ma_weights(period):
    if period % 2:
        return [1.0 / period] * period
    w = [1.0 / period] * (period + 1)
    w[0] = w[-1] = 0.5 / period
    return w


def centred_ma(x, period):
    w = ma_weights(period)
    return [sum(w[j] * x[t + j] for j in range(len(w))) for t in range(len(x) - len(w) + 1)]


def op(a, b, mul):
    return [ai / bi if mul else ai - bi for ai, bi in zip(a, b)]


def seasonal_filter(x, period, kind):
    sym, ends = SEASONAL[kind]
    out = [math.nan] * len(x)
    for j in range(period):
        positions = list(range(j, len(x), period))
        smooth = apply_filter([x[p] for p in positions], sym, ends)
        for p, value in zip(positions, smooth):
            out[p] = value
    return out


def normalize(s, extend, period, mul):
    w = ma_weights(period)
    drop = len(w) // 2
    centre = [math.nan] * len(s)
    inner = centred_ma(s, period)
    for i, value in enumerate(inner):
        centre[drop + i] = value
    for i in range(drop):
        centre[i] = centre[drop]
        centre[len(s) - 1 - i] = centre[len(s) - drop - 1]
    out = op(s, centre, mul)
    if extend:
        x = [math.nan] * extend + out + [math.nan] * extend
        for i in range(extend - 1, -1, -1):
            x[i] = x[i + period]
        for j in range(len(x) - extend, len(x)):
            x[j] = x[j - period]
        out = x
    return out


def positivity(x):
    x = list(x)
    n = len(x)
    for i in range(n):
        if x[i] <= 0:
            b = i - 1
            while b >= 0 and x[b] <= 0:
                b -= 1
            a = i + 1
            while a < n and x[a] <= 0:
                a += 1
            x[i] = (x[b] + x[a]) / 2 if b >= 0 and a < n else (x[b] if a >= n else x[a])
    return x


def henderson_trend(x, length, period, mul):
    h = henderson(length)
    out = apply_filter(x, h, musgrave(h, ratio_for(length, period)))
    return positivity(out) if mul else out


def abs_mean_variation(x, mul):
    total = 0.0
    for i in range(1, len(x)):
        d = abs(x[i] - x[i - 1])
        total += d / x[i - 1] if mul else d
    return total / (len(x) - 1)


def icr(s, period, mul):
    h = henderson(period + 1)
    m = len(h) // 2
    trend = [sum(h[j] * s[t - m + j] for j in range(len(h))) for t in range(m, len(s) - m)]
    irregular = [s[m + i] / trend[i] if mul else s[m + i] - trend[i] for i in range(len(trend))]
    ratio = abs_mean_variation(irregular, mul) / abs_mean_variation(trend, mul)
    return ratio * 3 if period == 4 else ratio


def select(ratio, period):
    if 1 <= ratio < 3.5:
        return period + 1
    if ratio < 1:
        return 9 if period == 12 else 5
    return 23 if period == 12 else 7


def single_sd(values, mean):
    kept = [v for v in values if not math.isnan(v)]
    return math.sqrt(sum((v - mean) ** 2 for v in kept) / len(kept))


def calc_sd(s, start, period, mean):
    n = len(s)
    ny = 1 + (start + n - 1) // period
    nfy = 1 + (start + n - 1) // period
    nbeg = 0
    if start != 0:
        nbeg = period - start
        nfy -= 1
    nend = (start + n) % period
    if nend != 0:
        nfy -= 1
    if nfy < 5:
        return [single_sd(s, mean)]
    ibeg = 2 + (1 if nbeg > 0 else 0)
    iend = ibeg + nfy - 5
    sd = [0.0] * ny
    pos, icur = nbeg, ibeg
    while icur <= iend:
        sd[icur] = single_sd(s[pos:pos + 5 * period], mean)
        icur += 1
        pos += period
    e = single_sd(s[:nbeg + 5 * period], mean) if nbeg > 0 else sd[ibeg]
    for i in range(ibeg):
        sd[i] = e
    if nend > 0:
        pos -= period
        e = single_sd(s[pos:], mean)
    else:
        e = sd[iend]
    for i in range(icur, ny):
        sd[i] = e
    return sd


def detect(s, sd, start, period, mean, lower, upper):
    n = len(s)
    w = [1.0] * n
    y, ibeg, iend = 0, 0, period - start
    while ibeg < n:
        sigma = sd[min(y, len(sd) - 1)]
        lv, uv = sigma * lower, sigma * upper
        if abs(sigma) >= 1e-5:
            for i in range(ibeg, iend):
                tt = abs(s[i] - mean)
                if tt - uv > 1e-15:
                    w[i] = 0.0
                elif tt - lv > 1e-15:
                    w[i] = (uv - tt) / (uv - lv)
        y += 1
        ibeg, iend = iend, min(iend + period, n)
    return w


def analyse(s, start, period, mul, lower, upper):
    mean = 1.0 if mul else 0.0
    w = detect(s, calc_sd(s, start, period, mean), start, period, mean, lower, upper)
    if any(v < 1 for v in w):
        trimmed = [math.nan if wi == 0 else si for si, wi in zip(s, w)]
        w = detect(s, calc_sd(trimmed, start, period, mean), start, period, mean, lower, upper)
    return w


def neighbours(i, w, period):
    found, lp, up = [], 0, 0
    lb = ub = i
    while lb >= period and lp != 2:
        lb -= period
        if w[lb] == 1.0:
            lp += 1
            found.append(lb)
    while ub < len(w) - period and up != 2:
        ub += period
        if w[ub] == 1.0:
            up += 1
            found.append(ub)
    if lp < 2:
        while ub < len(w) - period and len(found) < 4:
            ub += period
            if w[ub] == 1.0:
                found.append(ub)
    elif up < 2:
        while lb >= period and len(found) < 4:
            lb -= period
            if w[lb] == 1.0:
                found.append(lb)
    return found if len(found) >= 4 else None


def corrections(values, w, period):
    out = [math.nan] * len(values)
    for i, wi in enumerate(w):
        if wi == 1.0:
            continue
        found = neighbours(i, w, period)
        if found is None:
            same = values[i % period::period]
            out[i] = sum(same) / len(same)
        else:
            out[i] = (wi * values[i] + sum(values[k] for k in found)) / (4.0 + wi)
    return out


def apply(values, corr):
    return [c if not math.isnan(c) else v for v, c in zip(values, corr)]


def factors(s, w, mul):
    if mul:
        return [si / (1 + wi * (si - 1)) if wi < 1 else 1.0 for si, wi in zip(s, w)]
    return [si * (1 - wi) if wi < 1 else 0.0 for si, wi in zip(s, w)]


def x11_reference(y, period, mode="multiplicative", initial="3x3", final="3x5", fixed=None,
                  lower=1.5, upper=2.5, first_season=1):
    mul = mode == "multiplicative"
    start = first_season - 1
    n = len(y)
    drop = len(ma_weights(period)) // 2
    steps = {}

    def trend(name, series, first=False):
        if fixed is not None:
            length = fixed
        else:
            ratio = icr(series, period, mul)
            length = period + 1 if first and ratio >= 1 else select(ratio, period)
        steps[name] = length
        return henderson_trend(series, length, period, mul)

    b1 = list(y)
    b2 = centred_ma(b1, period)
    b3 = op(b1[drop:n - drop], b2, mul)
    b4a = normalize(seasonal_filter(b3, period, initial), 0, period, mul)
    w = analyse(op(b3, b4a, mul), (start + drop) % period, period, mul, lower, upper)
    b4 = corrections(b3, w, period)
    b5 = normalize(seasonal_filter(apply(b3, b4), period, initial), drop, period, mul)
    b6 = op(b1, b5, mul)
    b7 = trend("B7", b6, True)
    b8 = op(b1, b7, mul)
    b9c = normalize(seasonal_filter(b8, period, final), 0, period, mul)
    w = analyse(op(b8, b9c, mul), start, period, mul, lower, upper)
    b9 = corrections(b8, w, period)
    b10 = normalize(seasonal_filter(apply(b8, b9), period, final), 0, period, mul)
    b11 = op(b1, b10, mul)
    b13 = op(b11, b7, mul)
    b17 = analyse(b13, start, period, mul, lower, upper)
    b20 = factors(b13, b17, mul)
    c1 = op(b1, b20, mul)
    c4 = op(c1[drop:n - drop], centred_ma(c1, period), mul)
    c5 = normalize(seasonal_filter(c4, period, initial), drop, period, mul)
    c7 = trend("C7", op(c1, c5, mul))
    c10 = normalize(seasonal_filter(op(c1, c7, mul), period, final), 0, period, mul)
    c13 = op(op(b1, c10, mul), c7, mul)
    c17 = analyse(c13, start, period, mul, lower, upper)
    c20 = factors(c13, c17, mul)
    d1 = op(b1, c20, mul)
    d4 = op(d1[drop:n - drop], centred_ma(d1, period), mul)
    d5 = normalize(seasonal_filter(d4, period, initial), drop, period, mul)
    d7 = trend("D7", op(d1, d5, mul))
    d9bis = op(d1, d7, mul)
    d10 = normalize(seasonal_filter(d9bis, period, final), 0, period, mul)
    d11bis = op(d1, d10, mul)
    final_ratio = icr(d11bis, period, mul)
    d12 = trend("D12", d11bis)
    d11 = op(b1, d10, mul)
    d13 = op(d11, d12, mul)
    return {"b7": b7, "b13": b13, "b17": b17, "c7": c7, "c17": c17, "c20": c20, "d10": d10, "d11": d11, "d12": d12, "d13": d13,
            "henderson": steps, "final_ic_ratio": final_ratio}


def x11_cases():
    rng = np.random.default_rng(20261010)
    pattern12 = np.array([0.92, 0.95, 1.04, 1.08, 1.07, 1.02, 0.97, 0.93, 0.97, 1.01, 1.05, 0.99])
    pattern12 = pattern12 / pattern12.mean()
    cases = []
    t = np.arange(150)
    level = 200 + 0.6 * t + 6 * np.sin(2 * np.pi * t / 41)
    y = level * np.resize(pattern12, 150) * (1 + 0.004 * t / 150) * np.exp(rng.normal(0, 0.012, 150))
    y[[23, 70, 71, 128]] *= [1.22, 0.83, 1.12, 1.18]
    cases.append({"name": "monthly_multiplicative_auto", "y": y.tolist(), "period": 12, "options": {}})
    cases.append({"name": "monthly_multiplicative_start_april", "y": y[:131].tolist(), "period": 12,
                  "options": {"first_season": 4}})
    additive = level + 15 * (np.resize(pattern12, 150) - 1) * 10 + rng.normal(0, 2.0, 150)
    additive[[40, 97]] += [35, -28]
    cases.append({"name": "monthly_additive_h23_3x3", "y": additive.tolist(), "period": 12,
                  "options": {"mode": "additive", "henderson": 23, "seasonal_filter": "3x3"}})
    noisy = level * np.resize(pattern12, 150) * np.exp(rng.normal(0, 0.05, 150))
    cases.append({"name": "monthly_noisy_initial_3x5", "y": noisy[:110].tolist(), "period": 12,
                  "options": {"initial_seasonal_filter": "3x5", "sigma_limits": [1.8, 2.8]}})
    smooth = level * np.resize(pattern12, 150) * np.exp(rng.normal(0, 0.0005, 150))
    cases.append({"name": "monthly_smooth_low_ic", "y": smooth[:96].tolist(), "period": 12,
                  "options": {"seasonal_filter": "3x3"}})
    clustered = level[:84] * np.resize(pattern12, 84) * np.exp(rng.normal(0, 0.01, 84))
    clustered[[0, 12, 24, 48]] *= [1.3, 1.25, 0.75, 1.28]   # four of seven Januaries extreme
    cases.append({"name": "monthly_clustered_extremes_fallback", "y": clustered.tolist(),
                  "period": 12, "options": {}})
    tq = np.arange(58)
    quarterly = (80 + 1.1 * tq) * np.resize([1.06, 0.95, 0.91, 1.08], 58) * np.exp(rng.normal(0, 0.015, 58))
    quarterly[[13, 41]] *= [1.15, 0.88]
    cases.append({"name": "quarterly_multiplicative_auto", "y": quarterly.tolist(), "period": 4,
                  "options": {"first_season": 2}})
    cases.append({"name": "quarterly_additive_h7", "y": (quarterly - 100).tolist(), "period": 4,
                  "options": {"mode": "additive", "henderson": 7}})
    out = []
    for case in cases:
        opts = dict(case["options"])
        limits = opts.pop("sigma_limits", [1.5, 2.5])
        ref = x11_reference(case["y"], case["period"], mode=opts.get("mode", "multiplicative"),
                            initial=opts.get("initial_seasonal_filter", "3x3"),
                            final=opts.get("seasonal_filter", "3x5"), fixed=opts.get("henderson"),
                            lower=limits[0], upper=limits[1], first_season=opts.get("first_season", 1))
        out.append({**case, "reference": ref})
    return out


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(out_dir: str) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "generator": "scripts/generate_seasonal_oracles.py",
            "numpy_version": np.__version__,
            "seed": 20261010,
            "note": "plain-loop NumPy/Python X-11 method reference (independent of the Torch code)",
        },
        "x11": x11_cases(),
    }
    (target / "numpy_x11.json").write_text(json.dumps(_clean(payload), allow_nan=False))
    print("wrote", target / "numpy_x11.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/seasonal")
