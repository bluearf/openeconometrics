#!/usr/bin/env python3
"""Reproduce the IV-extension reference fixtures; no OpenEcon import.

Regular tests read the checked-in JSON and need neither linearmodels nor R.
Run with an environment that has numpy, pandas, statsmodels and linearmodels
(fixtures were produced with linearmodels 7.0 and statsmodels 0.15) and,
optionally, ``--r`` to add R AER/sandwich results through
``scripts/generate_iv_extension_oracles.R``:

    python scripts/generate_iv_extension_oracles.py [--r] [--rscript Rscript]

Lewbel (2012) fixture: a seeded data set, the generated instruments computed
here with explicit NumPy algebra ((Z - mean Z) * residual of the endogenous
regressor on [1 x1 x2]) and linearmodels IV2SLS/IVGMM fits that treat those
instruments as data (ivreg2h's convention), plus statsmodels' Koenker
Breusch-Pagan statistic of the first-stage residuals.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "iv_extensions"


def lewbel_data() -> pd.DataFrame:
    rng = np.random.default_rng(20261010)
    n = 400
    x1, x2, zext, f = rng.normal(size=(4, n))
    e2 = np.exp(0.6 * x1 - 0.3 * x2) * rng.normal(size=n) + f
    p = 1 + 0.5 * x1 - 0.3 * x2 + 0.4 * zext + e2
    y = 1 + x1 - x2 + 0.5 * p + f + (1 + 0.3 * np.abs(x2)) * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "p": p, "zext": zext,
                         "g": np.repeat(np.arange(40), 10)})


def generated(frame: pd.DataFrame) -> np.ndarray:
    x = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    e2 = frame.p.to_numpy() - x @ np.linalg.solve(x.T @ x, x.T @ frame.p.to_numpy())
    z = frame[["x1", "x2"]].to_numpy()
    return (z - z.mean(0)) * e2[:, None], e2


def matrix(value) -> list[list[float]]:
    return np.asarray(value, dtype=float).tolist()


def lewbel_python(frame: pd.DataFrame) -> dict:
    from linearmodels.iv import IV2SLS, IVGMM
    import linearmodels
    import statsmodels
    from statsmodels.stats.diagnostic import het_breuschpagan

    gen, e2 = generated(frame)
    exog = pd.DataFrame({"Intercept": 1.0, "x1": frame.x1, "x2": frame.x2})
    endog = frame[["p"]]
    sets = {"generated": pd.DataFrame(gen, columns=["p_x1_g", "p_x2_g"]),
            "generated_external": pd.DataFrame(np.column_stack([frame.zext, gen]),
                                               columns=["zext", "p_x1_g", "p_x2_g"])}
    out: dict = {"versions": {"linearmodels": linearmodels.__version__,
                              "statsmodels": statsmodels.__version__,
                              "numpy": np.__version__}}
    for name, inst in sets.items():
        model = IV2SLS(frame.y, exog, endog, inst)
        entry = {}
        for label, kwargs in {
            "unadjusted": {"cov_type": "unadjusted", "debiased": False},
            "robust": {"cov_type": "robust", "debiased": False},
            "robust_small": {"cov_type": "robust", "debiased": True},
            "cluster_small": {"cov_type": "clustered", "clusters": frame.g, "debiased": True},
        }.items():
            fit = model.fit(**kwargs)
            entry[label] = {"params": fit.params.tolist(), "cov": matrix(fit.cov)}
        gmm = IVGMM(frame.y, exog, endog, inst, weight_type="robust").fit(
            cov_type="robust", iter_limit=2, debiased=False)
        entry["gmm_robust"] = {"params": gmm.params.tolist(), "cov": matrix(gmm.cov),
                               "j_stat": float(gmm.j_stat.stat)}
        out[name] = entry
    design = np.column_stack([np.ones(len(frame)), frame.x1, frame.x2])
    lm, lm_p, _, _ = het_breuschpagan(e2, design, robust=True)
    out["koenker_bp"] = {"statistic": float(lm), "p_value": float(lm_p), "df": 2}
    out["generated_instruments_head"] = matrix(gen[:5])
    return out


def lewbel_r(frame: pd.DataFrame, rscript: str) -> dict:
    gen, _ = generated(frame)
    with tempfile.TemporaryDirectory() as tmp:
        data = frame.assign(p_x1_g=gen[:, 0], p_x2_g=gen[:, 1])
        source = Path(tmp) / "lewbel.csv"
        target = Path(tmp) / "lewbel_aer.json"
        data.to_csv(source, index=False, float_format="%.17g")
        subprocess.run([rscript, "--vanilla", str(ROOT / "scripts" / "generate_iv_extension_oracles.R"),
                        str(source), str(target)], check=True)
        return json.loads(target.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r", action="store_true", help="also run the R AER/sandwich oracle")
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = lewbel_data()
    payload = {
        "source": "scripts/generate_iv_extension_oracles.py (seed 20261010)",
        "data": {name: frame[name].tolist() for name in frame.columns},
        "python": lewbel_python(frame),
    }
    if args.r:
        payload["r"] = lewbel_r(frame, args.rscript)
    (OUTPUT / "lewbel.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
