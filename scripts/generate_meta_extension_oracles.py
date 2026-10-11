#!/usr/bin/env python3
"""Reproduce the static meta-extension oracle fixtures; no OpenEconometrics import.

Seeded NumPy builds the synthetic inputs, then ``generate_meta_extension_oracles.R``
runs metafor (rma.mv, confint, anova, predict, trimfill, selmodel),
clubSandwich (CR2, Satterthwaite, HTZ), weightr and base-R lm. The published
Konstantopoulos (2011) and Hackshaw (1998) records are exported from the
metadat package (GPL >= 2) by the R process. Regular tests read the committed
JSON/CSV and need no R installation.

    python scripts/generate_meta_extension_oracles.py [--rscript Rscript] [--rlib DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests" / "fixtures" / "meta_extensions"
SEED = 20261010


def _write(frame: pd.DataFrame, path: Path) -> None:
    # Shortest round-trip decimal strings parse identically in Python and R.
    frame.to_csv(path, index=False)


def che() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlated within-study sampling errors; effect- and study-level moderators."""
    rng = np.random.default_rng(SEED)
    sizes = [1, 2, 3, 4, 5, 2, 3, 1, 4, 2, 3, 5, 2, 1, 3, 2]
    studies = np.repeat([f"s{j:02d}" for j in range(len(sizes))], sizes)
    n = len(studies)
    x1 = np.round(rng.normal(size=n), 4)
    x2 = np.repeat(np.round(rng.uniform(-1, 1, size=len(sizes)), 4), sizes)
    v = np.round(0.02 + 0.2 * rng.uniform(size=n), 6)
    s = np.zeros((n, n))
    start = 0
    for size in sizes:
        block = slice(start, start + size)
        root = np.sqrt(v[block])
        s[block, block] = 0.6 * np.outer(root, root)
        start += size
    s = np.round(s, 12)
    s[np.diag_indices(n)] = v
    s = np.round((s + s.T) / 2, 12)
    u = np.repeat(rng.normal(scale=0.28, size=len(sizes)), sizes)
    w = rng.normal(scale=0.17, size=n)
    e = np.linalg.cholesky(s) @ rng.normal(size=n)
    y = np.round(0.3 + 0.15 * x1 - 0.2 * x2 + u + w + e, 6)
    labels = [f"e{i:02d}" for i in range(n)]
    data = pd.DataFrame({"effect": labels, "study": studies, "yi": y, "vi": v, "x1": x1, "x2": x2})
    return data, pd.DataFrame(s, index=labels, columns=labels)


def boundary() -> pd.DataFrame:
    """No between-study component in the generating model."""
    rng = np.random.default_rng(SEED + 1)
    sizes = [2, 3, 2, 4, 3, 2, 3, 4, 2, 3, 2, 3]
    studies = np.repeat([f"b{j:02d}" for j in range(len(sizes))], sizes)
    n = len(studies)
    v = np.round(0.03 + 0.1 * rng.uniform(size=n), 6)
    y = np.round(0.2 + rng.normal(scale=0.25, size=n) + rng.normal(size=n) * np.sqrt(v), 6)
    return pd.DataFrame({"effect": [f"b{i:02d}" for i in range(n)], "study": studies, "yi": y, "vi": v})


def dependent() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Known correlated study blocks for one-component dependent fits."""
    rng = np.random.default_rng(SEED + 2)
    sizes = [2 + i % 3 for i in range(9)]
    ids = np.repeat([f"study:{i}" for i in range(len(sizes))], sizes)
    n = len(ids)
    x = np.round(rng.normal(size=n), 4)
    variance = np.round(np.exp(np.linspace(-1, 1, n)) * 0.3, 6)
    s = np.zeros((n, n))
    start = 0
    for size in sizes:
        block = slice(start, start + size)
        root = np.sqrt(variance[block])
        s[block, block] = 0.45 * np.outer(root, root)
        start += size
    s[np.diag_indices(n)] = variance
    s = np.round((s + s.T) / 2, 12)
    b = rng.normal(scale=np.sqrt(0.5), size=len(sizes))
    y = np.round(0.4 - 0.3 * x + np.repeat(b, sizes) + np.linalg.cholesky(s) @ rng.normal(size=n), 6)
    labels = [f"effect:{i}" for i in range(n)]
    data = pd.DataFrame({"effect": labels, "study": ids, "yi": y, "x": x})
    return data, pd.DataFrame(s, index=labels, columns=labels)


def asymmetric() -> pd.DataFrame:
    """Small-study suppression: non-significant studies retained with probability .35."""
    rng = np.random.default_rng(SEED + 3)
    rows = []
    while len(rows) < 30:
        size = int(rng.integers(16, 240))
        v = 4 / size
        y = 0.25 + rng.normal(scale=np.sqrt(0.02)) + rng.normal(scale=np.sqrt(v))
        if y / np.sqrt(v) > 1.6449 or rng.uniform() < 0.35:
            rows.append({"yi": round(y, 6), "vi": round(v, 8)})
    frame = pd.DataFrame(rows)
    frame.insert(0, "study", [f"a{i:02d}" for i in range(len(frame))])
    return frame


def selection_moderator() -> pd.DataFrame:
    """Step-function selection on one-sided p-values with a numeric moderator."""
    rng = np.random.default_rng(SEED + 4)
    rows = []
    while len(rows) < 60:
        x = round(float(rng.uniform()), 4)
        size = int(rng.integers(20, 300))
        v = 4 / size
        y = 0.1 + 0.4 * x + rng.normal(scale=np.sqrt(0.03)) + rng.normal(scale=np.sqrt(v))
        p = 0.5 * math.erfc(y / math.sqrt(v) / math.sqrt(2))
        keep = 1.0 if p <= 0.05 else 0.5 if p <= 0.5 else 0.2
        if rng.uniform() < keep:
            rows.append({"yi": round(y, 6), "vi": round(v, 8), "x": x})
    frame = pd.DataFrame(rows)
    frame.insert(0, "study", [f"m{i:02d}" for i in range(len(frame))])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--rlib", default=None, help="extra R library directory prepended to .libPaths()")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        inputs, outputs = work / "inputs", work / "outputs"
        inputs.mkdir()
        che_data, che_cov = che()
        dep_data, dep_cov = dependent()
        frames = {"che.csv": che_data, "boundary.csv": boundary(), "dependent.csv": dep_data,
                  "asymmetric.csv": asymmetric(), "selection_moderator.csv": selection_moderator(),
                  "konst_z.csv": pd.DataFrame({"z": np.round(np.random.default_rng(SEED + 5).normal(size=56), 3)})}
        for name, frame in frames.items():
            _write(frame, inputs / name)
            _write(frame, TARGET / name)
        for name, frame in (("che_covariance.csv", che_cov), ("dependent_covariance.csv", dep_cov)):
            _write(frame, inputs / name)
            _write(frame, TARGET / name)
        env = dict(os.environ)
        if args.rlib:
            env["R_LIBS"] = args.rlib
        subprocess.run([args.rscript, "--vanilla", str(ROOT / "scripts" / "generate_meta_extension_oracles.R"),
                        str(inputs), str(outputs)], check=True, env=env)
        for name in ("konstantopoulos2011.csv", "hackshaw1998.csv"):
            shutil.copyfile(outputs / name, TARGET / name)
        oracles = json.loads((outputs / "oracles.json").read_text())
        runtime = (outputs / "runtime.txt").read_text().splitlines()
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(TARGET.glob("*.csv"))}
    payload = {"generator": "scripts/generate_meta_extension_oracles.py", "seed": SEED, "runtime": runtime,
               "input_sha256": digests, "results": oracles}
    (TARGET / "oracles.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
