#!/usr/bin/env python3
"""Reproduce independent R fixtures for the mixed-model extensions; no OpenEcon import.

Stages (MARKET-152):
  kr       lme4 + lmerTest + pbkrtest Kenward-Roger / Satterthwaite contrasts
  lme      nlme::lme random effects with corAR1 / corCompSymm / varIdent residuals
  clmm     ordinal::clmm random-intercept cumulative logit/probit by adaptive GHQ

The censored (Tobit) GLMM has no R oracle here (glmmTMB is unavailable); its tests use
an explicit NumPy/SciPy quadrature likelihood instead. Regular tests read the checked-in
JSON and need no R installation. Data are seeded NumPy draws (seeds below) except the
published lme4 ``sleepstudy`` data, written by R itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT = ROOT / "scripts" / "generate_mixed_extension_oracles.R"
DESTINATION = ROOT / "tests" / "fixtures" / "mixed_extensions"
SEED = 152_001


def unbalanced_kr() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sizes = rng.integers(3, 10, size=18)
    g = np.repeat(np.arange(len(sizes)), sizes)
    n = len(g)
    x = rng.normal(0.3, 1.0, size=n)
    f = rng.integers(1, 4, size=n)
    u0 = rng.normal(0, 1.1, size=len(sizes))
    u1 = rng.normal(0, 0.6, size=len(sizes))
    y = 2.0 + 0.7 * x + 0.5 * (f == 2) - 0.4 * (f == 3) + u0[g] + u1[g] * x + rng.normal(0, 0.9, n)
    return pd.DataFrame({"g": g + 1, "f": f, "x": x, "y": y})


def nested_kr() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for school in range(10):
        a = rng.normal(0, 0.9)
        for klass in range(int(rng.integers(2, 5))):
            b = rng.normal(0, 0.6)
            for _ in range(int(rng.integers(3, 7))):
                x = rng.normal()
                rows.append({"school": school + 1, "klass": klass + 1, "x": x,
                             "y": 1.0 + 0.4 * x + a + b + rng.normal(0, 0.8)})
    return pd.DataFrame(rows)


def run(mode: str, spec: dict, root: Path, name: str, rscript: str) -> dict:
    path = root / f"{name}.json"
    output = root / f"{name}-out.json"
    path.write_text(json.dumps(spec))
    subprocess.run([rscript, "--vanilla", str(R_SCRIPT), mode, str(path), str(output)], check=True)
    return json.loads(output.read_text())


def kr_cases(root: Path, rscript: str) -> dict:
    sleep_csv = root / "sleepstudy.csv"
    subprocess.run([rscript, "--vanilla", str(R_SCRIPT), "sleepstudy", str(sleep_csv)], check=True)
    sleep = pd.read_csv(sleep_csv)
    unbalanced, nested = unbalanced_kr(), nested_kr()
    cases = {
        "sleepstudy_reml": (sleep, {"formula": "Reaction ~ Days + (Days | Subject)", "reml": True,
            "factors": [], "openecon": {"y": "Reaction", "x": ["Days"], "group": "Subject",
                                        "random": ["Days"], "covstructure": "unstructured",
                                        "method": "reml"},
            "contrasts": {"name": ["days", "both", "shifted"],
                          "L": [[[0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]],
                          "rhs": [[0], [0, 0], [250, 9]]}}),
        "unbalanced_reml": (unbalanced, {"formula": "y ~ x + f + (1 | g) + (0 + x | g)", "reml": True,
            "factors": ["f"], "openecon": {"y": "y", "x": ["x", "f"], "categorical": ["f"],
                                           "group": "g", "random": ["x"],
                                           "covstructure": "independent", "method": "reml"},
            "contrasts": {"name": ["x", "factor", "three"],
                          "L": [[[0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]],
                                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, -1]]],
                          "rhs": [[0.5], [0, 0], [0.5, 0.2, 0.6]]}}),
        "unbalanced_ml": (unbalanced, {"formula": "y ~ x + f + (1 | g) + (0 + x | g)", "reml": False,
            "factors": ["f"], "openecon": {"y": "y", "x": ["x", "f"], "categorical": ["f"],
                                           "group": "g", "random": ["x"],
                                           "covstructure": "independent", "method": "ml"},
            "contrasts": {"name": ["x", "factor"],
                          "L": [[[0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]],
                          "rhs": [[0], [0, 0]]}}),
        "nested_reml": (nested, {"formula": "y ~ x + (1 | school) + (1 | school:klass)", "reml": True,
            "factors": [], "openecon": {"y": "y", "x": ["x"], "group": ["school", "klass"],
                                        "method": "reml"},
            "contrasts": {"name": ["x", "both"], "L": [[[0, 1]], [[1, 0], [0, 1]]],
                          "rhs": [[0], [1, 0.4]]}}),
    }
    out = {}
    for name, (frame, spec) in cases.items():
        csv = root / f"{name}.csv"
        frame.to_csv(csv, index=False, float_format="%.17g")
        spec = {**spec, "data": str(csv)}
        out[name] = {"data": frame.to_dict("list"), "spec": spec, "oracle": run("lmer", spec, root, name, rscript)}
        out[name]["spec"].pop("data")
    return out


def ar1_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for subject in range(36):
        arm = "b" if subject % 3 == 0 else "a"
        u0, u1 = rng.normal(0, 1.4), rng.normal(0, 0.5)
        times = np.sort(rng.choice(8, size=int(rng.integers(3, 8)), replace=False))
        shocks = [rng.normal()]
        for _ in range(7):
            shocks.append(0.45 * shocks[-1] + np.sqrt(1 - 0.45 ** 2) * rng.normal())
        sd = 0.7 * (1.6 if arm == "b" else 1.0)
        for t in times:
            x = rng.normal()
            rows.append({"g": subject + 1, "t": int(t), "arm": arm, "x": x,
                         "y": 1 + 0.5 * x + 0.2 * t + u0 + u1 * x + sd * shocks[t]})
    return pd.DataFrame(rows)


def cs_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for school in range(14):
        a = rng.normal(0, 1.0)
        for klass in range(int(rng.integers(2, 5))):
            shared = rng.normal(0, np.sqrt(0.3) * 0.8)
            for _ in range(int(rng.integers(3, 7))):
                x = rng.normal()
                rows.append({"school": school + 1, "klass": klass + 1, "x": x,
                             "y": 1 + 0.4 * x + a + shared + rng.normal(0, np.sqrt(0.7) * 0.8)})
    return pd.DataFrame(rows)


def lme_cases(root: Path, rscript: str) -> dict:
    ar1, cs = ar1_data(), cs_data()
    ar1_rand = "~ 1 | g"
    cases = {
        "ar1_ri_varident_reml": (ar1, {"fixed": "y ~ x + t", "random": ar1_rand,
            "correlation": "corAR1(value = 0.2, form = ~ t | g)",
            "weights": "varIdent(form = ~ 1 | arm)", "method": "REML",
            "openecon": {"y": "y", "x": ["x", "t"], "group": "g", "residual": "ar1",
                         "occasion": "t", "stratum": "arm", "method": "reml"}}),
        "ar1_ri_varident_ml": (ar1, {"fixed": "y ~ x + t", "random": ar1_rand,
            "correlation": "corAR1(value = 0.2, form = ~ t | g)",
            "weights": "varIdent(form = ~ 1 | arm)", "method": "ML",
            "openecon": {"y": "y", "x": ["x", "t"], "group": "g", "residual": "ar1",
                         "occasion": "t", "stratum": "arm", "method": "ml"}}),
        "ar1_slope_unstructured_ml": (ar1, {"fixed": "y ~ x + t", "random": "~ x | g",
            "correlation": "corAR1(value = 0.2, form = ~ t | g)", "weights": "NULL",
            "method": "ML",
            "openecon": {"y": "y", "x": ["x", "t"], "group": "g", "random": ["x"],
                         "covstructure": "unstructured", "residual": "ar1", "occasion": "t",
                         "method": "ml"}}),
        "diag_slope_varident_reml": (ar1, {"fixed": "y ~ x + t", "random": "list(g = pdDiag(~ x))",
            "correlation": "NULL", "weights": "varIdent(form = ~ 1 | arm)", "method": "REML",
            "openecon": {"y": "y", "x": ["x", "t"], "group": "g", "random": ["x"],
                         "covstructure": "independent", "residual": "independent",
                         "stratum": "arm", "method": "reml"}}),
        "cs_nested_reml": (cs, {"fixed": "y ~ x", "random": "~ 1 | school",
            "correlation": "corCompSymm(value = 0.1, form = ~ 1 | school/klass)",
            "weights": "NULL", "method": "REML",
            "openecon": {"y": "y", "x": ["x"], "group": "school", "within": "klass",
                         "residual": "exchangeable", "method": "reml"}}),
    }
    out = {}
    for name, (frame, spec) in cases.items():
        csv = root / f"{name}.csv"
        frame.to_csv(csv, index=False, float_format="%.17g")
        spec = {**spec, "data": str(csv), "factors": []}
        out[name] = {"data": frame.to_dict("list"), "spec": spec, "oracle": run("lme", spec, root, name, rscript)}
        out[name]["spec"].pop("data")
    return out


def ordinal_data(link: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cuts = np.array([-1.0, 0.3, 1.5])
    for group in range(40):
        u = rng.normal(0, 1.0)
        for _ in range(int(rng.integers(4, 11))):
            x, z = rng.normal(), float(rng.random() < 0.4)
            noise = rng.logistic() if link == "logit" else rng.normal()
            latent = 0.8 * x - 0.5 * z + u + noise
            rows.append({"g": group + 1, "x": x, "z": z,
                         "y": int(np.searchsorted(cuts, latent)) + 1})
    return pd.DataFrame(rows)


def clmm_cases(root: Path, rscript: str) -> dict:
    cases = {
        "ologit_agq10": (ordinal_data("logit", SEED + 4), {"link": "logit", "nAGQ": 10,
            "openecon": {"y": "y", "x": ["x", "z"], "group": "g", "intpoints": 10}}),
        "oprobit_agq10": (ordinal_data("probit", SEED + 5), {"link": "probit", "nAGQ": 10,
            "openecon": {"y": "y", "x": ["x", "z"], "group": "g", "intpoints": 10}}),
        # Three points: the adaptive rule itself (mode, curvature, node placement) matters.
        # clmm's nAGQ < 0 places nodes at x_k (not sqrt(2) x_k), a different non-adaptive
        # rule from Stata's ghermite, so the non-adaptive option is checked in NumPy.
        "ologit_agq3": (ordinal_data("logit", SEED + 4), {"link": "logit", "nAGQ": 3,
            "openecon": {"y": "y", "x": ["x", "z"], "group": "g", "intpoints": 3}}),
    }
    out = {}
    for name, (frame, spec) in cases.items():
        csv = root / f"{name}.csv"
        frame.to_csv(csv, index=False, float_format="%.17g")
        spec = {**spec, "data": str(csv), "factors": [], "outcome": "y", "group": "g",
                "formula": "y ~ x + z + (1 | g)"}
        out[name] = {"data": frame.to_dict("list"), "spec": spec, "oracle": run("clmm", spec, root, name, rscript)}
        out[name]["spec"].pop("data")
    return out


STAGES = {"kr": kr_cases, "lme": lme_cases, "clmm": clmm_cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--stage", choices=sorted(STAGES), action="append")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    digest = {script.name: hashlib.sha256(script.read_bytes()).hexdigest()
              for script in (Path(__file__), R_SCRIPT)}
    with tempfile.TemporaryDirectory(prefix="mixed-extension-oracles-") as directory:
        for stage in args.stage or sorted(STAGES):
            cases = STAGES[stage](Path(directory), args.rscript)
            receipt = {"schema": "openecon.mixed-extension-oracles.v1", "stage": stage,
                       "seed_base": SEED, "script_sha256": digest, "cases": cases}
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"{stage}.json").write_text(
                json.dumps(receipt, sort_keys=True, indent=1, allow_nan=False) + "\n")
            print(json.dumps({"stage": stage, "cases": sorted(cases)}))


if __name__ == "__main__":
    main()
