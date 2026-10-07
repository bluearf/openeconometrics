"""Paste into an OpenEconometrics code panel and Run. Synthetic data only.

Exercises MARKET-68/116/117/119/120, with deterministic seeds and saved-result
JSON round trips. No SciPy/statsmodels/R/Stata package is needed.
"""

import json
import numpy as np
import pandas as pd
import openecon as oe
from openecon.models import ResultBundle

rng = np.random.default_rng(710)
n = 180
x = rng.normal(size=n)
df = pd.DataFrame(
    dict(
        x=x,
        y=1 + 0.5 * x + rng.normal(size=n),
        nb=rng.negative_binomial(2, 2 / (2 + np.exp(0.2 + 0.3 * x))),
        f=rng.integers(1, 4, n),
        cluster=np.arange(n) // 15,
        t=np.arange(n),
    )
)

weighted_ols = oe.ols(data=df, y="y", x=["x"], weights="f", weight_type="fweight")
weighted_nb = oe.nbreg(data=df, y="nb", x=["x"], weights="f", weight_type="fweight")
joint = oe.suest(weighted_ols, weighted_nb, data=df, names=["ols", "count"], cluster="cluster")
assert np.isfinite(oe.test(joint, {"ols:x": 1, "count:x": -1})["statistic"])

cluster_fit = oe.ols(data=df, y="y", x=["x"], covariance="cluster", cluster="cluster")
wild = oe.bootstrap(
    cluster_fit,
    df,
    scheme="wild_cluster",
    cluster="cluster",
    reps=99,
    seed=19,
    wild="webb",
    null={"x": 0},
)
series = oe.ols(data=df, y="y", x=["x"], time="t", lags=3, covariance="hac")
block = oe.bootstrap(
    series, df, scheme="moving_block", block_length=8, ci="studentized", reps=99, seed=18
)

ids = np.repeat(np.arange(90), 6)
times = np.tile(np.arange(6), 90)
first = np.repeat(np.r_[np.full(30, np.nan), np.full(30, 3.0), np.full(30, 4.0)], 6)
panel = pd.DataFrame(
    dict(
        id=ids,
        t=times,
        first=first,
        y=ids / 90 + 0.2 * times + np.where(times >= first, 1.5, 0) + rng.normal(size=len(ids)),
    )
)
cs = oe.csdid(
    data=panel,
    y="y",
    group="id",
    time="t",
    treatment_time="first",
    bootstrap_reps=199,
    seed=11,
    uniform=True,
    anticipation=1,
)
assert cs.inference["families"]["dynamic"]["critical_value"] > 0

levels = rng.integers(0, 3, 360)
treatment = pd.DataFrame(dict(x=rng.normal(size=360), d=levels))
treatment["y"] = 1 + 0.4 * treatment.x + 1.2 * treatment.d + rng.normal(size=360)
atet = oe.teffects(
    data=treatment, y="y", treatment="d", x=["x"], method="aipw", estimand="atet", tlevel=2
)
match = pd.DataFrame(
    dict(
        x=np.tile([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 8),
        d=np.tile([0, 1], 24),
        cell=np.repeat(np.arange(8), 6),
        f=2,
    )
)
match["y"] = match.x + 2 * match.d + rng.normal(size=len(match))
matched = oe.teffects(
    data=match,
    y="y",
    treatment="d",
    x=["x"],
    ematch=["cell"],
    method="nnmatch",
    weights="f",
    weight_type="fweight",
    matching_vce="iid",
)
assert matched.extra["frequency_duplication"]["expanded_rows"] == 96

models = [joint, wild, block, cs, atet, matched]
for model in models:
    restored = ResultBundle.model_validate_json(model.model_dump_json())
    assert restored.covariance_matrix == model.covariance_matrix
    assert restored.sample_positions == model.sample_positions

network = oe.network(
    [
        {"source": "s", "target": "a", "capacity": 3},
        {"source": "a", "target": "t", "capacity": 3},
        {"source": "s", "target": "t", "capacity": 2},
    ],
    directed=True,
    weight="capacity",
)
flow = network.min_cost_flow(
    {"s": -4, "t": 4}, {("s", "a"): 1.0, ("a", "t"): 1.0, ("s", "t"): 5.0}, domain="integral"
)
assert flow["status"] == "optimal" and flow["objective"] == 11
multiple = network.multicommodity_flow(
    {"a": {"s": -2, "t": 2}, "b": {"s": -2, "t": 2}},
    {("s", "a"): 1.0, ("a", "t"): 1.0, ("s", "t"): 5.0},
)
assert multiple["objective"] == 11
render = globals().get("display")
if render:
    for model in models:
        render(model)
    render(flow)
    render(flow["flows"])
    render(pd.DataFrame(cs.extra["dynamic"]))
else:
    print(flow.summary().to_string(index=False))
print(
    "FIVE_ISSUES_RECEIPT:"
    + json.dumps(
        {
            "issues": [68, 116, 117, 119, 120],
            "json_round_trips": len(models),
            "models": [m.title for m in models],
            "flow_objective": flow["objective"],
            "multicommodity_objective": multiple["objective"],
            "flow_certified": flow["metadata"]["certified"],
            "uniform_families": list(cs.inference["families"]),
            "matching_effect": matched.coefficients[0].estimate,
            "source_runtime": "native Torch",
            "synthetic_only": True,
        }
    )
)
