"""Editable examples for MARKET-103/113/123/128/133.

Uses repository public numeric fixtures and disposable synthetic data. Set
FIXTURE_DIRECTORY to tests/fixtures when running from the native code panel.
"""

import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

display = globals().get("display", print)
torch.set_num_threads(2)
base = Path(
    globals().get(
        "FIXTURE_DIRECTORY",
        Path(globals().get("__file__", ".")).resolve().parents[2] / "tests/fixtures",
    )
)
proof = {
    "issues": ["MARKET-103", "MARKET-113", "MARKET-123", "MARKET-128", "MARKET-133"],
    "frozen": bool(getattr(sys, "frozen", False)),
}


def restore(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


def collect(dataset):
    return pd.concat(list(dataset.iter_batches(batch_rows=7)))


# Published manufacturing productivity: original observed-row CCEMG sample.
macro = pd.read_csv(base / "common_factors/manu_prod_model.csv")
cce = oe.xtcce(
    data=macro,
    y="ly",
    x=["lk"],
    panel="nwbcode",
    time="year",
    model="ccemg",
    trend=True,
    missing="drop",
)
assert cce.nobs == 1194
assert math.isclose(cce.coefficients[0].estimate, 0.3124664, abs_tol=5e-7)
display(cce)
proof["published_ccemg"] = {"nobs": cce.nobs, "slope": cce.coefficients[0].estimate}

# Original Pedroni / Westerlund VR conventions, not Stata finite-sample parity.
rng = np.random.default_rng(8)
v = (rng.normal(size=(7, 42, 3)) + rng.normal(size=(1, 42, 3)) * 0.25).cumsum(axis=1)
panel = pd.DataFrame(
    {
        "id": np.repeat(np.arange(7), 42),
        "time": np.tile(np.arange(42), 7),
        "y": v[:, :, 0].ravel(),
        "x": v[:, :, 1].ravel(),
        "z": v[:, :, 2].ravel(),
    }
)
for method in ["kao", "pedroni", "westerlund"]:
    tested = oe.xtcointtest(
        panel,
        "y",
        ["x", "z"],
        "id",
        "time",
        test=method,
        lags=2,
        **({"bandwidth": "auto"} if method == "kao" else {"kernel_lags": 3}),
    )
    display(tested)
proof["cointegration"] = ["kao_auto", "pedroni_original", "westerlund_2005_vr"]

# Heterogeneous adoption and positive 2x2 weights reconstructing TWFE exactly.
rng = np.random.default_rng(556)
first = np.repeat([4, 7, 12], 6)
ids, time = np.repeat(np.arange(18), 12), np.tile(np.arange(12), 18)
d = (time >= first[ids]).astype(float)
y = (
    rng.normal(size=18)[ids]
    + time * 0.2
    + d * (first[ids] / 4 + (time - first[ids]) * 0.3)
    + rng.normal(size=len(ids))
)
adoption = pd.DataFrame({"id": ids, "time": time, "d": d, "y": y})
for model in ["bjs", "sunab"]:
    display(oe.heterodid(data=adoption, y="y", treatment="d", panel="id", time="time", model=model))
decomposition = oe.bacon(adoption, "y", "d", "id", "time")
display(decomposition)
california = pd.read_csv(base / "modern_did/california_prop99.csv", sep=";")
for model, target in [("sc", -19.5136298), ("sdid", -15.6053979)]:
    fitted = oe.synthcontrol(
        data=california,
        y="PacksPerCapita",
        treatment="treated",
        panel="State",
        time="Year",
        model=model,
    )
    assert math.isclose(fitted.coefficients[0].estimate, target, abs_tol=0.002)
    display(fitted)
proof["modern_causal"] = ["bjs", "sunab", "bacon", "sc", "sdid"]

# Independent public reference data for additional likelihood families.
air = pd.read_csv(base / "extended_mixed/airacc.csv")
nb = oe.xtnbreg(data=air, y="i_cnt", x=["inprog"], panel="airline", time="time", exposure="pmiles")
assert math.isclose(nb.metrics["log_likelihood"], -265.38202, abs_tol=5e-6)
display(nb)
frontier_data = pd.read_csv(base / "extended_mixed/xtfrontier1.csv")
frontier = oe.xtfrontier(
    data=frontier_data,
    y="lnwidgets",
    x=["lnmachines", "lnworkers"],
    panel="id",
    time="t",
    time_varying=True,
)
assert math.isclose(frontier.metrics["log_likelihood"], -1472.5289, abs_tol=5e-5)
display(frontier)
penicillin = pd.read_csv(base / "extended_mixed/penicillin.csv")
crossed = oe.mixedflex(data=penicillin, y="diameter", group=["plate", "sample"])
assert math.isclose(crossed.coefficients[0].estimate, 22.97222222222, abs_tol=1e-9)
display(crossed)
visits = pd.read_csv(base / "extended_mixed/drvisits.csv")
mixed_nb = oe.menbreg(
    data=visits, y="numvisit", x=["reform", "age", "educ", "married", "badh", "loginc"], group="id"
)
assert math.isclose(mixed_nb.metrics["log_likelihood"], -4513.1853, abs_tol=0.0002)
display(mixed_nb)
transport = pd.read_csv(base / "extended_mixed/transport.csv")
columns = ["trcost", "trtime"]
for alternative in ["Public", "Bicycle", "Walk"]:
    for column in ["age", "income"]:
        name = alternative + "_" + column
        transport[name] = (transport.alt == alternative) * transport[column]
        columns.append(name)
    name = alternative + "_const"
    transport[name] = (transport.alt == alternative).astype(float)
    columns.append(name)
choice = oe.mixedlogit(
    data=transport,
    y="choice",
    x=columns,
    group="id",
    case="t",
    alternative="alt",
    random=["trtime"],
    max_work=10_000_000_000,
)
assert math.isclose(choice.metrics["log_likelihood"], -1006.0052151, abs_tol=0.0001)
display(choice)
proof["additional_likelihoods"] = [
    "xtnbreg",
    "xtfrontier_tvd",
    "mixedflex_crossed",
    "menbreg",
    "mixedlogit",
]

# Dataset output must cover all 510 rows, beyond the 400-row chart preview.
rng = np.random.default_rng(140)
g = np.repeat(np.arange(85), 6)
x = rng.normal(size=len(g))
y = 1 + 0.7 * x + rng.normal(size=85)[g] + rng.normal(size=len(g)) * 0.6
grouped = pd.DataFrame({"g": g, "x": x, "y": y})
grouped.index = pd.Index(np.arange(len(g)) % 11, name="duplicate")
source = oe.Dataset.from_frame(grouped)
lmm = restore(oe.mixed(data=source, y="y", x=["x"], group="g"))
blups = collect(oe.mixed_predict(lmm, source, kind="fitted", batch_rows=5))
assert len(blups) == 510 and blups.index.equals(grouped.index)
display(blups.head(6))
fe = restore(oe.areg(data=source, y="y", x=["x"], absorb="g"))
conditional = collect(oe.predict(fe, source, interval="mean", batch_rows=5))
assert len(conditional) == 510 and (conditional.std_error > 0).all()
display(conditional.head(6))
binary = grouped.copy()
binary["chosen"] = np.tile([1, 0, 0, 1, 0, 0], 85)
logit = restore(oe.clogit(data=oe.Dataset.from_frame(binary), y="chosen", x=["x"], group="g"))
predicted = collect(oe.predict(logit, oe.Dataset.from_frame(binary), interval="mean", batch_rows=4))
assert len(predicted) == 510 and predicted.index.equals(binary.index)
assert np.allclose(predicted.response.to_numpy().reshape(85, 6).sum(1), 2)
display(predicted.head(6))
proof["complete_dataset_rows"] = 510
proof["index_and_success_counts"] = True
gc.collect()
if proof["frozen"]:
    import importlib.util

    assert importlib.util.find_spec("scipy") is None
    assert importlib.util.find_spec("statsmodels") is None
print("FIVE_PANEL_METHODS_OK " + json.dumps(proof, sort_keys=True))
