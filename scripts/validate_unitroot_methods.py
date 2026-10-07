"""Reproduce published calibration protocols and a bounded size/power smoke grid.

NumPy is a QA-only independent numerical path. Paper simulation counts are
retained; the seed is declared, not the original authors' unavailable seed.
Finite-sample grid results are diagnostics, never universal validity claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.unitroot.advanced_series import NG_CRITICAL, KSS_CRITICAL

ROOT = Path(__file__).resolve().parents[1]


def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - radius), float(center + radius)]


def calibration():
    records = []
    for method, draws, steps, seed in [
        ("ngperron", 20_000, 5_000, 136001),
        ("kss", 100_000, 1_000, 137001),
    ]:
        rng = np.random.default_rng(seed)
        samples = {}
        for first in range(0, draws, 200):
            batch = min(200, draws - first)
            w = np.column_stack(
                (np.zeros(batch), np.cumsum(rng.normal(size=(batch, steps)), axis=1))
            ) / np.sqrt(steps)
            r = np.arange(steps + 1) / steps
            if method == "ngperron":
                for trend, c in [("constant", -7.0), ("trend", -13.5)]:
                    u = w
                    if trend == "trend":
                        lam = (1 - c) / (1 - c + c * c / 3)
                        slope = lam * w[:, -1] + 3 * (1 - lam) * np.trapezoid(
                            w * r, dx=1 / steps, axis=1
                        )
                        u = w - slope[:, None] * r
                    integral = np.trapezoid(u * u, dx=1 / steps, axis=1)
                    end = u[:, -1] ** 2
                    mza = (end - 1) / (2 * integral)
                    msb = np.sqrt(integral)
                    values = [
                        mza,
                        mza * msb,
                        msb,
                        c * c * integral + (-c if trend == "constant" else 1 - c) * end,
                    ]
                    for name, value in zip(["MZa", "MZt", "MSB", "MPT"], values, strict=True):
                        samples.setdefault((trend, name), []).append(value)
            else:
                # Original paper Table 1: iid normal random walks, T=1000;
                # vectorized direct dot-product cubic regression, not Torch QR.
                for trend in ["none", "constant", "trend"]:
                    u = w.copy()
                    if trend != "none":
                        u -= u.mean(axis=1)[:, None]
                    if trend == "trend":
                        centered = r - r.mean()
                        slope = (u * centered).sum(axis=1) / np.dot(centered, centered)
                        u -= slope[:, None] * centered
                    cubic = u[:, :-1] ** 3
                    delta = np.diff(u, axis=1)
                    ss = (cubic * cubic).sum(axis=1)
                    cross = (cubic * delta).sum(axis=1)
                    ssr = (delta * delta).sum(axis=1) - cross * cross / ss
                    statistic = cross / np.sqrt(ss * ssr / (steps - 1))
                    samples.setdefault((trend, "KSS"), []).append(statistic)
        for (trend, name), batches in samples.items():
            statistics = np.concatenate(batches)
            published = NG_CRITICAL[trend][name] if method == "ngperron" else KSS_CRITICAL[trend]
            checks = []
            for alpha, cv in zip([0.01, 0.05, 0.10], published, strict=True):
                count = int(np.sum(statistics < cv))
                rate = count / draws
                # Five Monte Carlo standard errors plus 0.0035 accommodate
                # the published simulation's unknown seed and rounded tables.
                tolerance = 0.0035 + 5 * np.sqrt(alpha * (1 - alpha) / draws)
                checks.append(
                    dict(
                        alpha=alpha,
                        published_critical=cv,
                        rejections=count,
                        rejection_rate=rate,
                        wilson_95=wilson(count, draws),
                        absolute_tolerance=float(tolerance),
                        passed=bool(abs(rate - alpha) < tolerance),
                    )
                )
            records.append(
                dict(
                    method=method,
                    trend=trend,
                    test=name,
                    replications=draws,
                    steps=steps,
                    seed=seed,
                    failures=0,
                    quantiles=np.quantile(statistics, [0.01, 0.05, 0.10]).tolist(),
                    checks=checks,
                )
            )
        print(
            json.dumps(dict(stage="published_calibration", method=method, replications=draws)),
            flush=True,
        )
    return records


def finite_grid():
    records = []
    repetitions = 250
    for n in [100, 250]:
        for dgp in ["iid_unit_root", "ar04_unit_root", "stationary_alternative"]:
            seed = (
                136137
                + n
                + (["iid_unit_root", "ar04_unit_root", "stationary_alternative"].index(dgp) * 1000)
            )
            rng = np.random.default_rng(seed)
            cells = {}
            for _ in range(repetitions):
                eps = rng.normal(size=n + 500)
                innovation = np.zeros(n + 500)
                for t in range(1, n + 500):
                    innovation[t] = (
                        0.4 * innovation[t - 1] if dgp == "ar04_unit_root" else 0
                    ) + eps[t]
                y = np.r_[0, np.cumsum(innovation[-(n - 1) :])]
                estar = np.zeros(n + 500)
                linear = np.zeros(n + 500)
                if dgp == "stationary_alternative":
                    for t in range(1, n + 500):
                        estar[t] = (
                            estar[t - 1]
                            - 0.7 * estar[t - 1] * (1 - np.exp(-0.1 * estar[t - 1] ** 2))
                            + eps[t]
                        )
                        linear[t] = 0.6 * linear[t - 1] + eps[t]
                for method, trends in [
                    ("ngperron", ["constant", "trend"]),
                    ("kss", ["none", "constant", "trend"]),
                ]:
                    data = pd.DataFrame(
                        {
                            "y": (linear[-n:] if method == "ngperron" else estar[-n:])
                            if dgp == "stationary_alternative"
                            else y
                        }
                    )
                    for trend in trends:
                        key = (method, trend)
                        cell = cells.setdefault(
                            key, dict(rejections={}, failures=0, error_codes={})
                        )
                        try:
                            options = (
                                {"maxlag": 8}
                                if method == "ngperron"
                                else {"lags": 4 if dgp == "ar04_unit_root" else 0}
                            )
                            result = getattr(oe, method)(data, "y", trend=trend, **options)
                            for row in result.to_dict("records"):
                                cell["rejections"][row["test"]] = cell["rejections"].get(
                                    row["test"], 0
                                ) + int(
                                    row["statistic"] < NG_CRITICAL[trend][row["test"]][1]
                                    if method == "ngperron"
                                    else row["reject_5pct"]
                                )
                        except Exception as e:
                            cell["failures"] += 1
                            code = getattr(e, "code", type(e).__name__)
                            cell["error_codes"][code] = cell["error_codes"].get(code, 0) + 1
            for (method, trend), cell in cells.items():
                records.append(
                    dict(
                        method=method,
                        trend=trend,
                        n_levels=n,
                        dgp=dgp,
                        seed=seed,
                        replications=repetitions,
                        failures=cell["failures"],
                        error_codes=cell["error_codes"],
                        options={"maxlag": 8}
                        if method == "ngperron"
                        else {"lags": 4 if dgp == "ar04_unit_root" else 0},
                        tests={
                            name: dict(
                                rejections=count,
                                unconditional_rate=count / repetitions,
                                wilson_95=wilson(count, repetitions),
                            )
                            for name, count in cell["rejections"].items()
                        },
                    )
                )
            print(json.dumps(dict(stage="finite_grid", n_levels=n, dgp=dgp)), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt path.")
    torch.set_num_threads(1)
    start = time.monotonic()
    calibration_records = calibration()
    grid = finite_grid()
    source = ROOT / "src/openecon/econometrics/unitroot/advanced_series.py"
    method_status = {}
    for method in ("ngperron", "kss"):
        complete = all(
            check["passed"]
            for item in calibration_records
            if item["method"] == method
            for check in item["checks"]
        )
        complete = complete and all(
            item["failures"] == 0 for item in grid if item["method"] == method
        )
        method_status[method] = "passed" if complete else "unresolved"
    passed = all(value == "passed" for value in method_status.values())
    record = dict(
        status="passed" if passed else "partial",
        method_status=method_status,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        elapsed_seconds=time.monotonic() - start,
        original_protocols={
            "ngperron": "Table I: 20000 Wiener paths, 5000 steps; theorem-1 null Brownian functionals",
            "kss": "Table 1: 100000 iid normal random walks, T=1000; raw, demeaned and OLS detrended cubic t ratios",
        },
        calibration=calibration_records,
        finite_sample_smoke=grid,
        limits="Declared seeds differ from unavailable authors' seeds. Quantiles are asymptotic. 250-draw grid is a bounded diagnostic, not validated universal finite-sample size/power; no negative-MA, heteroskedastic, break or arbitrary dependence claim.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"status": record["status"], "elapsed_seconds": record["elapsed_seconds"]}),
        flush=True,
    )
    if method_status["kss"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
