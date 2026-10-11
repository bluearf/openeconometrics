"""Regenerate the reference fixtures of the break/Fourier/threshold cointegration tests.

Development tool only (never imported by OpenEconometrics). It

1. builds seeded synthetic datasets (NumPy ``default_rng`` seeds below);
2. runs the external R oracles in ``generate_cointegration_breaks_oracles.R``
   (Bruce Hansen's published R procedures for Gregory-Hansen / Hatemi-J,
   CRAN makicoint 2.0.0, hatemicoint 1.1.0, cointests 1.1.0, urca);
3. re-encodes the Bayer-Hanck null distributions ``NullDistr.dta`` of the
   authors' SSC module ``bayerhanck`` into the runtime asset
   ``src/openecon/econometrics/unitroot/bayer_hanck_null.xz``;
4. writes ``tests/fixtures/cointegration_breaks/oracles.json``.

Every downloaded source is checked against the sha256 recorded below. Usage::

    python scripts/generate_cointegration_breaks_oracles.py --sources DIR --rlib RLIB

with DIR holding joe_96r.zip (https://users.ssc.wisc.edu/~bhansen/progs/joe_96r.zip),
NullDistr.dta (http://fmwww.bc.edu/RePEc/bocode/n/NullDistr.dta) and the CRAN
source tarballs combcoint_0.2.0, makicoint_2.0.0, hatemicoint_1.1.0 and
cointests_1.1.0 (the last three installed into RLIB).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/cointegration_breaks/oracles.json"
ASSET = ROOT / "src/openecon/econometrics/unitroot/bayer_hanck_null.xz"

SOURCES = {
    "joe_96r.zip": "970c7664119636b8948aa3ed38370ab4c5aba5096a513ae33ef264759c12d7f0",
    "NullDistr.dta": "4975161431432e5441a96b029879a10a0efb1d536a7d61b262f562deb5c4e6bc",
    "combcoint_0.2.0.tar.gz": "2d7986b6779f87e172f2bcb1f3bfd70116d24f4a0a3129d126062c729ed5753c",
}
QUANTUM = 1e-4          # resolution of the stored Bayer-Hanck null draws


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(directory: Path, name: str) -> Path:
    path = directory / name
    digest = sha256(path)
    if digest != SOURCES[name]:
        raise SystemExit(f"{name}: sha256 {digest} differs from the recorded {SOURCES[name]}")
    return path


def ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    shocks = rng.normal(size=n)
    out = np.zeros(n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + shocks[t]
    return out


def datasets() -> dict:
    out = {}
    rng = np.random.default_rng(20261010)
    n = 100
    x = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    y = 1 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + 2.0 * (np.arange(n) >= 50) + ar1(rng, n, 0.5)
    out["level_break"] = {"y": y.tolist(), "x": x.T.tolist()}
    rng = np.random.default_rng(20261011)
    n = 80
    walks = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    out["null_pair"] = {"y": walks[:, 0].tolist(), "x": [walks[:, 1].tolist()]}
    rng = np.random.default_rng(20261012)
    n = 120
    x = np.cumsum(rng.normal(size=n))
    slope = np.where(np.arange(n) >= 70, 1.5, 0.6)
    y = 2 + slope * x + 0.05 * np.arange(n) + ar1(rng, n, 0.3)
    out["regime_shift"] = {"y": y.tolist(), "x": [x.tolist()]}
    rng = np.random.default_rng(20261013)
    n = 150
    x = np.cumsum(rng.normal(size=(n, 3)), axis=0)
    y = 0.3 + x @ np.array([1.0, -0.5, 0.25]) + ar1(rng, n, 0.6)
    out["three_regressors"] = {"y": y.tolist(), "x": x.T.tolist()}
    return out


def cases() -> dict:
    gh = []
    for model in ("level", "trend", "regime", "regime_trend"):
        for rule, kmax in (("t", 6), ("aic", 4), ("bic", 4), ("fixed", 2)):
            gh.append({"data": "level_break", "model": model, "rule": rule, "kmax": kmax})
        gh.append({"data": "regime_shift", "model": model, "rule": "t", "kmax": 6})
        gh.append({"data": "null_pair", "model": model, "rule": "aic", "kmax": 3})
    gh.append({"data": "three_regressors", "model": "regime", "rule": "t", "kmax": 6})
    hj = [{"data": "null_pair", "rule": "t", "kmax": 4},
          {"data": "null_pair", "rule": "fixed", "kmax": 1},
          {"data": "level_break", "rule": "aic", "kmax": 3},
          {"data": "level_break", "rule": "fixed", "kmax": 0}]
    maki = []
    for model in ("level", "trend", "regime", "regime_trend"):
        maki.append({"data": "level_break", "model": model, "breaks": 3, "rule": "t",
                     "kmax": 12})
        maki.append({"data": "regime_shift", "model": model, "breaks": 2, "rule": "fixed",
                     "kmax": 1})
    maki.append({"data": "null_pair", "model": "level", "breaks": 5, "rule": "t", "kmax": 8})
    bh = [{"data": data, "trend": trend, "lags": lags}
          for data in ("three_regressors", "null_pair", "level_break")
          for trend in ("none", "constant", "trend") for lags in (0, 1, 2)]
    fourier = []
    for data in ("level_break", "regime_shift", "three_regressors"):
        for model in ("constant", "trend"):
            fourier.append({"test": "fadl", "data": data, "model": model, "max_frequency": 3,
                            "max_lag": 3, "criterion": "aic"})
            fourier.append({"test": "tsong", "data": data, "model": model, "max_frequency": 3,
                            "leads_lags": 2, "kernel": "bartlett", "bandwidth": 4})
    fourier.append({"test": "fadl", "data": "null_pair", "model": "constant",
                    "max_frequency": 5, "max_lag": 2, "criterion": "bic"})
    fourier.append({"test": "tsong", "data": "null_pair", "model": "trend", "max_frequency": 2,
                    "leads_lags": 1, "kernel": "iid", "bandwidth": 3})
    return {"gregory_hansen": gh, "hatemi_j": hj, "maki": maki, "bayer_hanck": bh,
            "fourier": fourier}


def _resid(design: np.ndarray, target: np.ndarray):
    gram = np.einsum("rti,rtj->rij", design, design)
    beta = np.linalg.solve(gram, np.einsum("rti,rt->ri", design, target)[..., None])[..., 0]
    return target - np.einsum("rti,ri->rt", design, beta), beta, gram


def fourier_check(seed: int = 20261010) -> dict:
    """Seeded simulation of the Fourier null distributions (bounded: about two minutes).

    Tsong: CI_f with OLS and the residual variance (exogenous x, iid errors; the same
    limit as the DOLS statistic), T = 1000, 10,000 replications per cell. FADL: t ratio
    with no lagged differences, T = 100 and 500 (5,000 replications) and 2500 (2,000).
    """
    from openecon.econometrics.unitroot.fourier_coint import FADL, TSONG  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    out = {"seed": seed, "tsong": [], "fadl": []}
    T = 1000
    t = np.arange(1, T + 1)
    for model in ("constant", "trend"):
        for k in (1, 2, 3):
            det = [np.ones(T)] + ([t / T] if model == "trend" else [])
            det += [np.sin(2 * np.pi * k * t / T), np.cos(2 * np.pi * k * t / T)]
            base = np.stack(det, 1)
            for m in (1, 2, 3, 4):
                stats = []
                for _ in range(10):
                    x = np.cumsum(rng.normal(size=(1000, T, m)), axis=1)
                    y = 0.5 * x.sum(-1) + rng.normal(size=(1000, T))
                    design = np.concatenate([np.broadcast_to(base, (1000, T, base.shape[1])), x], 2)
                    e = _resid(design, y)[0]
                    stats.append((np.cumsum(e, 1) ** 2).sum(1) / (T ** 2 * (e ** 2).mean(1)))
                q = np.quantile(np.concatenate(stats), [0.99, 0.95, 0.90])
                out["tsong"].append({"model": model, "k": k, "m": m, "simulated": q.tolist(),
                                     "table": [TSONG[model][k][m - 1][i] for i in (2, 1, 0)]})
    for T, reps in ((100, 5000), (500, 5000), (2500, 2000)):
        t = np.arange(2, T + 1)
        column = {100: 0, 500: 3, 2500: 6}[T]
        for model in ("constant", "trend"):
            for m in (1, 2, 3):
                for k in (1, 2, 3, 4, 5):
                    det = [np.ones(T - 1)] + ([t / T] if model == "trend" else [])
                    det += [np.sin(2 * np.pi * k * t / T), np.cos(2 * np.pi * k * t / T)]
                    base = np.stack(det, 1)
                    stats = []
                    for start in range(0, reps, 500):
                        z = np.cumsum(rng.normal(size=(500, T, m + 1)), axis=1)
                        dz = np.diff(z, axis=1)
                        design = np.concatenate([np.broadcast_to(base, (500, T - 1, base.shape[1])),
                                                 z[:, :-1], dz[:, :, 1:]], 2)
                        e, beta, gram = _resid(design, dz[:, :, 0])
                        s2 = (e ** 2).sum(1) / (design.shape[1] - design.shape[2])
                        j = base.shape[1]
                        stats.append(beta[:, j] / np.sqrt(s2 * np.linalg.inv(gram)[:, j, j]))
                    q = np.quantile(np.concatenate(stats), [0.01, 0.05, 0.10])
                    out["fadl"].append({"model": model, "k": k, "m": m, "T": T,
                                        "simulated": q.tolist(),
                                        "table": list(FADL[model][m][k - 1][column:column + 3])})
    return out


def _series(rng, n, m, kind):
    """Null (independent random walks) or cointegrated designs with AR(0.4) errors."""
    x = np.cumsum(rng.normal(size=(n, m)), axis=0)
    if kind == "null":
        y = np.cumsum(rng.normal(size=n))
    else:
        u = ar1(rng, n, 0.4)
        t = np.arange(n)
        shift = {"stable": 0.0, "level": 4.0 * (t >= n // 2),
                 "two_level": 4.0 * (t >= n // 3) - 4.0 * (t >= 2 * n // 3),
                 "regime": 1.5 * x[:, 0] * (t >= n // 2),
                 "two_regime": 1.5 * x[:, 0] * ((t >= n // 3) & (t < 2 * n // 3)),
                 "smooth": 3.0 * np.sin(2 * np.pi * t / n)}[kind]
        y = 1.0 + 0.7 * x.sum(axis=1) + shift + u
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(m)])
    frame.insert(0, "y", y)
    return frame, [f"x{i + 1}" for i in range(m)]


def size_power(reps: int = 500, seed: int = 20261014, n: int = 100) -> dict:
    """Seeded rejection frequencies at the published 5% points (n = 100, m = 1)."""
    import openecon as oe  # noqa: PLC0415

    rng = np.random.default_rng(seed)

    def below(out, name):
        return out.attrs["statistics"][name] < out.attrs["critical_values"][name]["5%"]

    designs = {
        "gregory_hansen_level": (lambda d, x: oe.gregory_hansen(d, "y", x, model="level"),
                                 ("ADF*", "Zt*", "Za*"), "level", below),
        "gregory_hansen_regime": (lambda d, x: oe.gregory_hansen(d, "y", x, model="regime"),
                                  ("ADF*", "Zt*", "Za*"), "regime", below),
        "hatemi_j": (lambda d, x: oe.hatemi_j(d, "y", x), ("ADF*", "Zt*"), "two_regime", below),
        "maki_level_m2": (lambda d, x: oe.maki_coint(d, "y", x, breaks=2, model="level"),
                          ("tau",), "two_level",
                          lambda out, _: out.attrs["statistic"] < out.attrs["critical_values"]["5%"]),
        "bayer_hanck": (lambda d, x: oe.bayer_hanck(d, "y", x, lags=1), ("EG-J", "EG-J-Ba-Bo"),
                        "stable", lambda out, name: out.attrs["statistics"][name]
                        > out.attrs["critical_values"][name]["5%"]),
        "fourier_adl": (lambda d, x: oe.fourier_adl(d, "y", x), ("t",), "smooth",
                        lambda out, _: out.attrs["statistic"] < out.attrs["critical_values"]["5%"]),
    }
    out = {"seed": seed, "reps": reps, "n": n, "regressors": 1, "level": "5%", "results": {}}
    for name, (call, statistics, alternative, reject) in designs.items():
        counts = {"size": dict.fromkeys(statistics, 0), "power": dict.fromkeys(statistics, 0)}
        for _ in range(reps):
            for key, kind in (("size", "null"), ("power", alternative)):
                frame, names = _series(rng, n, 1, kind)
                result = call(frame, names)
                for statistic in statistics:
                    counts[key][statistic] += bool(reject(result, statistic))
        out["results"][name] = {"alternative": alternative,
                                **{key: {s: v / reps for s, v in value.items()}
                                   for key, value in counts.items()}}
        print(name, out["results"][name], flush=True)
    # Tsong: H0 is cointegration (size under the stable design, power under no cointegration).
    counts = {"size": 0, "power": 0}
    for _ in range(reps):
        for key, kind in (("size", "stable"), ("power", "null")):
            frame, names = _series(rng, n, 1, kind)
            result = oe.tsong_fourier(frame, "y", names)
            critical = result.attrs["critical_values"]
            counts[key] += bool(critical is not None and critical["5%"] is not None
                                and result.attrs["statistic"] > critical["5%"])
    out["results"]["tsong_fourier"] = {"alternative": "no cointegration",
                                       "size": counts["size"] / reps,
                                       "power": counts["power"] / reps}
    # Enders-Siklos: no published critical values; record empirical null quantiles only.
    phi = []
    for _ in range(reps):
        frame, names = _series(rng, n, 1, "null")
        phi.append(oe.enders_siklos(frame, "y", names, lags=1).attrs["phi"])
    out["results"]["enders_siklos_tar_phi_null_quantiles"] = {
        "90%": float(np.quantile(phi, 0.90)), "95%": float(np.quantile(phi, 0.95)),
        "99%": float(np.quantile(phi, 0.99)),
        "note": "simulated for documentation only; not used as critical values"}
    return out


def encode_null(dta: Path) -> bytes:
    """Sorted columns, rounded to QUANTUM, int64 first value + int32 increments, LZMA."""
    import pandas as pd

    frame = pd.read_stata(dta)
    values = frame.to_numpy(np.float64).T                      # [132, 10000]
    if values.shape != (132, 10000) or not np.all(np.diff(values, axis=1) >= 0):
        raise SystemExit("NullDistr.dta is not 132 sorted columns of 10,000 draws.")
    scaled = np.round(values / QUANTUM).astype(np.int64)
    header = np.array([values.shape[0], values.shape[1]], dtype="<i4").tobytes()
    first = scaled[:, 0].astype("<i8").tobytes()
    steps = np.diff(scaled, axis=1)
    if steps.max() >= 2 ** 31:
        raise SystemExit("increment overflow")
    payload = header + first + steps.astype("<i4").tobytes()
    return lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--rlib", type=Path, required=True)
    parser.add_argument("--fourier-check", action="store_true",
                        help="also rerun the seeded simulation check of the Fourier tables")
    parser.add_argument("--size-power", action="store_true",
                        help="only rerun the seeded size/power simulation (needs openecon)")
    args = parser.parse_args()
    if args.size_power:
        target = FIXTURE.with_name("size_power.json")
        target.write_text(json.dumps(size_power(), indent=1, sort_keys=True) + "\n")
        print("wrote", target)
        return
    data = datasets()
    spec = {"datasets": data, **cases()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(checked(args.sources, "joe_96r.zip")) as archive:
            (tmp / "main.R").write_bytes(archive.read("main.R"))
        with tarfile.open(checked(args.sources, "combcoint_0.2.0.tar.gz")) as archive:
            member = archive.extractfile("combcoint/R/sysdata.rda")
            (tmp / "sysdata.rda").write_bytes(member.read())
        (tmp / "cases.json").write_text(json.dumps(spec))
        subprocess.run(["Rscript", "--vanilla", str(ROOT / "scripts/generate_cointegration_breaks_oracles.R"),
                        str(tmp / "cases.json"), str(tmp / "main.R"), str(tmp / "sysdata.rda"),
                        str(tmp / "out.json"), str(args.rlib)], check=True)
        oracles = json.loads((tmp / "out.json").read_text())
        combcoint = np.loadtxt(tmp / "combcoint_null.csv", delimiter=",", skiprows=1).T
    asset = encode_null(checked(args.sources, "NullDistr.dta"))
    ASSET.write_bytes(asset)
    import pandas as pd  # noqa: PLC0415

    original = pd.read_stata(checked(args.sources, "NullDistr.dta")).to_numpy(np.float64).T
    # The redistributed draws are those of MIT-licensed CRAN combcoint 0.2.0 (sysdata.rda).
    if np.max(np.abs(original - combcoint)) > 1e-6:
        raise SystemExit("NullDistr.dta and combcoint's null_dist differ")
    positions = [0, 99, 499, 4999, 9499, 9899, 9999]
    checkpoints = {"positions": positions, "values": original[:, positions].tolist()}
    fixture = {
        "generated_by": "scripts/generate_cointegration_breaks_oracles.py",
        "sources": SOURCES,
        "null_payload_sha256": hashlib.sha256(lzma.decompress(asset)).hexdigest(),
        "null_quantum": QUANTUM,
        "null_checkpoints": checkpoints,
        **spec,
        "oracles": oracles,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
    if args.fourier_check:
        check = FIXTURE.with_name("fourier_simulation.json")
        check.write_text(json.dumps(fourier_check(), indent=1, sort_keys=True) + "\n")
        print("wrote", check)
    print("wrote", FIXTURE, len(FIXTURE.read_bytes()), "bytes; asset", len(asset), "bytes")


if __name__ == "__main__":
    main()
