"""Replay representative native fits with external estimator imports blocked.

Development dependencies: NumPy, pandas and Torch, plus the local OpenEcon source.
Run from the repository with PYTHONPATH=src:packages/openecon-charts/src. The
receipt names supplied-parameter DCC/BEKK recursions separately from CCC joint ML.
"""

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
OUTPUT = ROOT / "docs/evidence/market-124-127-129-131-132/runtime-dependencies.json"
BLOCKED = {"scipy", "statsmodels", "sklearn", "linearmodels"}
preloaded = [name for name in sys.modules if name.split(".")[0] in BLOCKED]
if preloaded:
    raise AssertionError(f"Forbidden libraries preloaded before guard: {preloaded}")
attempts = []
original_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in BLOCKED:
        attempts.append(name)
        raise AssertionError(f"Forbidden estimator dependency import: {name}")
    return original_import(name, *args, **kwargs)


class BlockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            attempts.append(fullname)
            raise AssertionError(f"Forbidden importlib dependency import: {fullname}")
        return None


builtins.__import__ = guarded_import
sys.meta_path.insert(0, BlockFinder())
torch.set_num_threads(1)
import openecon as oe  # noqa: E402 -- Import guard must precede runtime package imports.
from openecon.models import ResultBundle  # noqa: E402

receipt = {
    "schema": "openecon.runtime-dependency-check.v1",
    "blocked_libraries": sorted(BLOCKED),
    "guard": "builtins.__import__ plus importlib MetaPathFinder; forbidden libraries not preloaded",
    "fixture_dependencies": "NumPy RNG and pandas/Torch only; no external estimator/oracle libraries",
    "family_coverage": ["regularized", "robust", "spatial", "tsworkflows", "mgarch"],
    "methods": [],
    "failures": [],
    "limits": "Representative deterministic executions; neither external numerical parity nor exhaustive supported-domain proof.",
}


def run(name, family, call, post=None):
    started = time.perf_counter()
    try:
        result = call()
        record = {"method": name, "family": family, "passed": True}
        if isinstance(result, ResultBundle):
            encoded = result.model_dump_json()
            restored = ResultBundle.model_validate_json(encoded)
            latex = str(restored.to_latex())
            record.update(
                nobs=result.nobs,
                coefficient_count=len(result.coefficients),
                covariance_dimension=len(result.covariance_matrix),
                result_json_roundtrip=True,
                result_json_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
                latex_characters=len(latex),
                backend=result.provenance.get("backend"),
                precision=result.provenance.get("precision"),
            )
        else:
            json.dumps(result, allow_nan=False)
            record["finite_json"] = True
        if post:
            record.update(post(result))
        record["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        receipt["methods"].append(record)
        print(json.dumps({"method": name, "passed": True}), flush=True)
    except Exception as exc:
        record = {
            "method": name,
            "family": family,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
        receipt["methods"].append(record)
        receipt["failures"].append(record)
        print(json.dumps(record), flush=True)


rng = np.random.default_rng(110)
n = 360
g = np.repeat(np.arange(36), 10)
t = np.tile(np.arange(10), 36)
x, z, v, e = rng.normal(size=(4, n))
u = rng.normal(size=36)[g]
d = 0.8 * z + 0.3 * x + v
y = 1 + 0.7 * x + 0.5 * d + 0.7 * u + 0.5 * v + e
frame = pd.DataFrame({"y": y, "x": x, "z": z, "d": d, "id": g, "t": t})
run(
    "ridge",
    "regularized",
    lambda: oe.ridge(data=frame, y="y", x=["x", "z"], selection="cv", folds=3, n_lambdas=8),
)
run(
    "dmlplr",
    "regularized",
    lambda: oe.dmlplr(
        data=frame,
        y="y",
        treatment="d",
        x=["x", "z"],
        nuisance="ridge",
        selection="cv",
        folds=3,
        n_lambdas=8,
    ),
)
run(
    "localreg",
    "regularized",
    lambda: oe.localreg(
        data=frame, y="y", x=["x", "z"], selection="fixed", bandwidth=2.0, query=[[0.0, 0.0]]
    ),
    lambda r: {
        "saved_prediction": oe.regularized_predict(
            r, pd.DataFrame({"x": [0.0], "z": [0.0]})
        ).tolist()
    },
)
run("sreg", "robust", lambda: oe.sreg(data=frame, y="y", x=["x", "z"], starts=100, seed=110))
run("mmreg", "robust", lambda: oe.mmreg(data=frame, y="y", x=["x", "z"], starts=100, seed=110))
panel = pd.read_csv(ROOT / "tests/fixtures/robust/nlswork-mmqr.csv")
run(
    "panel_mmqr",
    "robust",
    lambda: oe.panel_mmqr(
        data=panel,
        y="ln_wage",
        x=["age", "ttl_exp", "tenure", "not_smsa", "south"],
        panel="idcode",
        time="year",
        quantiles=[0.1, 0.5, 0.9],
    ),
)
spatial = frame.iloc[:40].copy()
spatial["unit"] = np.arange(40)
weights = oe.spatial_weights(
    range(40), [(i, j, 1.0) for i in range(40) for j in ((i - 1) % 40, (i + 1) % 40)]
)
run(
    "sar",
    "spatial",
    lambda: oe.sar(spatial, "y", ["x", "z"], key="unit", spatial_weights=weights),
    lambda r: {
        "impacts_persisted": len(r.extra["impacts"]["rows"]),
        "impact_latex_characters": len(str(oe.spatial_impacts(r).to_latex())),
    },
)
run(
    "moran",
    "spatial",
    lambda: oe.moran(spatial, "y", key="unit", spatial_weights=weights, permutations=49, seed=110),
)
series = pd.DataFrame({"y": rng.normal(size=100), "t": np.arange(100)})
run(
    "ets",
    "tsworkflows",
    lambda: oe.ets(data=series, y="y", fixed={"alpha": 0.3}, initial=[0.0]),
    lambda r: {"forecast_rows": len(oe.forecast(r, 3))},
)
system = {"Z": [[1.0]], "T": [[0.7]], "Q": [[0.3]], "H": [[0.5]], "initialization": "stationary"}
run(
    "sspace",
    "tsworkflows",
    lambda: oe.sspace(data=series, y="y", time="t", system=system),
    lambda r: {"forecast_rows": len(oe.forecast(r, 3))},
)

h = np.ones(2)
q = np.array([[1.0, 0.35], [0.35, 1.0]])
target = q.copy()
values = []
vrng = np.random.default_rng(97)
for _ in range(200):
    correlation = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
    shock = vrng.multivariate_normal([0.0, 0.0], correlation) * np.sqrt(h)
    values.append(shock)
    standardized = shock / np.sqrt(h)
    q = 0.1 * target + 0.2 * np.outer(standardized, standardized) + 0.7 * q
    h = 0.1 + 0.25 * shock**2 + 0.65 * h
run(
    "mgarch_ccc",
    "mgarch",
    lambda: oe.mgarch_ccc(
        pd.DataFrame(values, columns=["a", "b"]), ["a", "b"], intercept=False, tolerance=1e-6
    ),
    lambda r: {"forecast_rows": len(oe.forecast(r, 3))},
)

from openecon.econometrics.mgarch import kernels as mgarch  # noqa: E402

reference = json.loads(
    (ROOT / "docs/evidence/market-131-mgarch/published-reference.json").read_text()
)


def supplied_matrix_path(kind):
    ref = reference[kind]
    matrices = {
        key: torch.tensor(value, dtype=torch.float64) for key, value in ref["parameters"].items()
    }
    matrices["mu"] = torch.tensor(reference["mean"], dtype=torch.float64)
    initial = (
        torch.tensor(ref["initializer"], dtype=torch.float64)
        if kind == "bekk"
        else torch.diag(torch.tensor(ref["initializer_variances"], dtype=torch.float64))
    )
    inputs = torch.tensor(reference["returns"], dtype=torch.float64)
    mgarch.validate_matrices(matrices, kind)
    path, residuals, state = mgarch.covariance_path(matrices, inputs, initial, kind)
    future = mgarch.forecast_path(matrices, state, kind, 5)
    assert bool(torch.isfinite(path).all()) and bool(torch.isfinite(future).all())
    return {
        "path": path.tolist(),
        "forecast": future.tolist(),
        "mode": "supplied-parameter native recursion; not fitted ML",
    }


run("mgarch_dcc_supplied_recursion", "mgarch", lambda: supplied_matrix_path("dcc"))
run("mgarch_bekk_supplied_recursion", "mgarch", lambda: supplied_matrix_path("bekk"))
receipt["forbidden_import_attempts"] = attempts
receipt["forbidden_loaded_modules"] = [
    name for name in sys.modules if name.split(".")[0] in BLOCKED
]
receipt["source_sha256"] = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for family in receipt["family_coverage"]
    for path in sorted((ROOT / "src/openecon/econometrics" / family).glob("*.py"))
}
receipt["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt["passed"] = (
    not receipt["failures"] and not attempts and not receipt["forbidden_loaded_modules"]
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            "passed": receipt["passed"],
            "methods": len(receipt["methods"]),
            "receipt": str(OUTPUT.relative_to(ROOT)),
        }
    ),
    flush=True,
)
if not receipt["passed"]:
    raise SystemExit(1)
