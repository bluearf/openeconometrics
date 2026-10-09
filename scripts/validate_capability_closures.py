"""Replay five capability milestones without external estimation libraries.

Run with PYTHONPATH=src. --size adds a deliberately narrow Monte Carlo receipt;
it is a size diagnostic, not a calibration or vendor parity certificate.
"""

import argparse
import builtins
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/evidence/capability-closures-2026-10-07"
parser = argparse.ArgumentParser()
parser.add_argument("--size", action="store_true")
args = parser.parse_args()
blocked = {"scipy", "statsmodels", "sklearn", "linearmodels"}
assert not any(n.split(".")[0] in blocked for n in sys.modules)
attempts = []
original_import = builtins.__import__


def guarded_import(name, *a, **kw):
    if name.split(".")[0] in blocked:
        attempts.append(name)
        raise AssertionError(f"Forbidden dependency: {name}")
    return original_import(name, *a, **kw)


class BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            attempts.append(fullname)
            raise AssertionError(f"Forbidden dependency: {fullname}")
        return None


builtins.__import__ = guarded_import
sys.meta_path.insert(0, BlockFinder())
torch.set_num_threads(1)
import openecon as oe  # noqa: E402
from openecon.models import ModelSpec, ResultBundle  # noqa: E402

receipt = {
    "schema": "openecon.capability-closures.v1",
    "issues": [34, 54, 66, 71, 72],
    "blocked_libraries": sorted(blocked),
    "guard": "import hook and MetaPathFinder installed before importing OpenEcon",
    "fixture_dependencies": "NumPy RNG, pandas and Torch; no external estimator",
    "methods": [],
    "limits": "Representative resident CPU executions, not vendor execution, packaged-app validation or exhaustive domain proof.",
}


def run(name, call, post=None):
    started = time.perf_counter()
    result = call()
    record = {"method": name, "passed": True}
    if isinstance(result, ResultBundle):
        encoded = result.model_dump_json()
        restored = ResultBundle.model_validate_json(encoded)
        assert restored.covariance_matrix == result.covariance_matrix
        latex = str(restored.to_latex())
        assert latex
        record.update(
            nobs=result.nobs,
            full_covariance_dimension=len(result.covariance_matrix),
            result_json_roundtrip=True,
            latex_characters=len(latex),
        )
    else:
        encoded = json.dumps(
            {"tables": {k: v.to_dict("records") for k, v in result.items()}, "attrs": result.attrs},
            allow_nan=False,
        )
        assert json.loads(encoded)
        record["finite_json"] = True
        record["latex_characters"] = len(str(result.to_latex()))
    record["result_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    if post:
        record.update(post(result))
    record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    receipt["methods"].append(record)
    print(json.dumps(record), flush=True)


rng = np.random.default_rng(738)
n, g = 320, np.repeat(np.arange(40), 8)
z, c, v, e = rng.normal(size=(4, n))
d = z + 0.2 * c + v
y = 1 + 1.4 * d + 0.3 * c + 0.5 * v + e
frame = pd.DataFrame(dict(y=y, d=d, c=c, z=z, z2=z**2 - 1, g=g, t=np.arange(n)))
run("ivcue", lambda: oe.ivcue(data=frame, y="y", x=["c"], endog=["d"], instruments=["z", "z2"]))
run(
    "effective_f",
    lambda: oe.effective_f(data=frame, y="y", x=["c"], endog=["d"], instruments=["z", "z2"]),
)
panel = frame.assign(t=np.tile(np.arange(8), 40), z1=rng.normal(size=40)[g])
panel["z2"] = 0.5 * panel.z1 + rng.normal(size=40)[g]
panel["y"] = (
    1 + 0.8 * panel.c + 0.4 * panel.d + 0.7 * panel.z1 + 0.6 * panel.z2 + rng.normal(size=40)[g] + e
)


def saved_prediction(result):
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    rows = oe.panel_structural_predict(restored, data=panel.iloc[:3])
    return {"saved_prediction_rows": len(rows["predictions"])}


run("cre", lambda: oe.cre(data=panel, y="y", x=["c", "d"], panel="g", time="t"), saved_prediction)
run(
    "xthtaylor",
    lambda: oe.xthtaylor(
        data=panel, y="y", x1=["c"], x2=["d"], z1=["z1"], z2=["z2"], panel="g", time="t"
    ),
    saved_prediction,
)
med = frame.assign(m=0.7 * c + rng.normal(size=n))
med["y"] = 1 + med.c + 0.6 * med.m + 0.4 * med.c * med.m + e
run(
    "mediation_interaction",
    lambda: oe.mediation_interaction(data=med, y="y", x=["c"], mediator="m", controls=["z"]),
)
binary = frame.assign(group=np.where(np.arange(n) % 2, "A", "B"))
binary["y"] = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * c + z))))
run(
    "fairlie",
    lambda: oe.fairlie(
        data=binary,
        y="y",
        x=["c", "z"],
        group="group",
        groups=["A", "B"],
        reps=49,
        matching_reps=4,
        seed=738,
    ),
)
series = pd.DataFrame(
    dict(y=rng.normal(size=90).cumsum(), x=rng.normal(size=90).cumsum(), t=np.arange(90))
)
run(
    "asymcausality",
    lambda: oe.asymcausality(data=series, y="y", x="x", time="t", bootstrap=49, seed=738),
)
spec = ModelSpec(estimator="ols", outcome="y", predictors=["x"], time="t")
run("recursive_ols", lambda: oe.recursive_ols(spec, data=series, minimum=25, step=15))
run("reverse_rolling", lambda: oe.rolling(spec, data=series, window=25, step=15, reverse=True))
pspec = ModelSpec(
    estimator="xtreg", outcome="y", predictors=["c"], panel="g", time="t", options={"model": "fe"}
)
run("rolling_panel", lambda: oe.rolling_panel(pspec, data=panel, window=4, step=2))
aspec = ModelSpec(estimator="arima", outcome="x", time="t", options={"order": [0, 1, 0]})
run(
    "rolling_origin_evaluation",
    lambda: oe.rolling(
        aspec,
        data=series,
        window=45,
        step=30,
        forecast_steps=2,
        evaluate=True,
        selection={
            "d": 1,
            "constant": False,
            "max_p": 1,
            "max_q": 0,
            "max_P": 0,
            "max_Q": 0,
            "max_candidates": 2,
        },
    ),
)

if args.size:
    # Under this DGP the two raw random walks are independent. This checks one
    # finite-sample signed-increment null, not general cross-sectional dependence,
    # heteroskedasticity, leverage/wild bootstrap or structural causality.
    p_values, failures = [], []
    for replication in range(100):
        r = np.random.default_rng(97000 + replication)
        data = pd.DataFrame(
            dict(y=r.normal(size=160).cumsum(), x=r.normal(size=160).cumsum(), t=np.arange(160))
        )
        try:
            result = oe.asymcausality(
                data=data,
                y="y",
                x="x",
                time="t",
                lags=1,
                dmax=1,
                bootstrap=99,
                seed=43000 + replication,
            )
            p_values.append(float(result["tests"].bootstrap_p_value.iloc[0]))
        except Exception as exc:
            failures.append({"replication": replication, "error": str(exc)})
        if (replication + 1) % 10 == 0:
            print(json.dumps({"size_replications": replication + 1}), flush=True)
    rejected = sum(p <= 0.05 for p in p_values)
    m = len(p_values)
    phat, z95 = rejected / m, 1.959963984540054
    den = 1 + z95**2 / m
    center = (phat + z95**2 / (2 * m)) / den
    half = z95 * (phat * (1 - phat) / m + z95**2 / (4 * m**2)) ** 0.5 / den
    receipt["size_diagnostic"] = {
        "dgp": "two independent Gaussian random walks; positive-positive zero-initial partial sums; N=160; lags=1,dmax=1",
        "replications": 100,
        "bootstrap_draws": 99,
        "rejections_at_5pct": rejected,
        "rejection_rate": phat,
        "wilson_95pct_interval": [center - half, center + half],
        "p_values": p_values,
        "failures": failures,
        "interpretation": "Diagnostic only. Conditional iid partial-sum VAR resampling is not a calibrated Hatemi-J leverage/wild bootstrap; no vendor parity or unrestricted size claim.",
    }

receipt["forbidden_import_attempts"] = attempts
receipt["forbidden_loaded_modules"] = [n for n in sys.modules if n.split(".")[0] in blocked]
paths = [
    ROOT / "src/openecon/econometrics" / family
    for family in ("iv", "panel", "decomposition", "tsworkflows", "var")
]
receipt["source_sha256"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for directory in paths
    for p in sorted(directory.glob("*.py"))
}
receipt["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt["passed"] = not attempts and not receipt["forbidden_loaded_modules"]
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "native-runtime.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
assert receipt["passed"]
