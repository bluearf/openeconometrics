"""Synthetic integration example; source Python or the OpenEcon console.

Every state is written/read as complete JSON; displayed tables are previews.
No installed-app, public-release or licensed vendor-execution claim.
"""

import json
from pathlib import Path

import pandas as pd
import torch

import openecon as oe

rng = torch.Generator(device="cpu").manual_seed(4821)


def normal(*shape):
    return torch.randn(shape, generator=rng, dtype=torch.float64)


n = 96
group = torch.arange(8).repeat_interleave(12)
x, z, v, error = normal(4, n)
d = 0.8 * z + 0.2 * x + v
y = 1 + 0.4 * x + 1.2 * d + 0.4 * v + error
data = pd.DataFrame(
    dict(y=y.tolist(), x=x.tolist(), d=d.tolist(), z=z.tolist(), g=group.tolist(), t=list(range(n)))
)
iv = oe.ivregress(data=data, y="y", x=["x"], endog=["d"], instruments=["z"], covariance="nonrobust")
iv_restored = oe.ResultBundle.model_validate_json(iv.model_dump_json())
weak = oe.iv_saved_weak_test(iv_restored, data=data, null=1.2)
ar_set = oe.iv_saved_ar_confidence_set(iv_restored, data=data)
linear = oe.ols(data=data, y="y", x=["x", "d"])
wild = oe.wild_cluster_test(
    linear, data=data, cluster="g", null={"d": 1.2}, reps=49, seed=8, wild="webb"
)
wild_grid = oe.wild_cluster_confidence_set(
    linear, data=data, cluster="g", term="d", grid=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5], reps=49, seed=8
)
fisher = oe.fisher_johansen(
    data=pd.DataFrame(dict(unit=["a", "b", "c"], p=[0.02, 0.13, 0.4])),
    unit="unit",
    p_value="p",
    rank=0,
    calibration="Synthetic already-calibrated trace p-values; fixed constant; illustrative independent units",
    independent=True,
)

keys = [f"unit-{i}" for i in range(n)]
weights = oe.spatial_weights(
    keys, [(keys[i], keys[j], 1.0) for i in range(n) for j in [(i - 1) % n, (i + 1) % n]]
)
w = weights.dense()
sx = normal(n)
sy = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - 0.3 * w, 1 + 0.8 * sx + 0.3 * normal(n))
spatial_data = pd.DataFrame(
    dict(key=keys, x=sx.tolist(), z=(w @ sx).tolist(), z2=(w @ w @ sx).tolist(), y=sy.tolist())
)
spatial = oe.sar_iv(
    spatial_data,
    "y",
    ["x"],
    key="key",
    instruments=["z", "z2"],
    spatial_weights=weights,
    covariance="HC0",
)
spatial_restored = oe.ResultBundle.model_validate_json(spatial.model_dump_json())
prediction = oe.sar_iv_predict(spatial_restored, data=spatial_data[["key", "x"]])
impacts = oe.spatial_impacts(spatial_restored)

groups, periods = 5, 24
ids = torch.arange(groups).repeat_interleave(periods)
px = torch.rand(groups * periods, generator=rng, dtype=torch.float64) - 0.5
py = 0.3 * ids + 1.2 * px + (1 + 0.04 * ids + 0.05 * px) * normal(len(ids))
panel = pd.DataFrame(
    dict(id=ids.tolist(), time=list(range(periods)) * groups, x=px.tolist(), y=py.tolist())
)
mmqr = oe.panel_mmqr(data=panel, y="y", x=["x"], panel="id", time="time")
resampled = oe.panel_mmqr_bootstrap(mmqr, data=panel, reps=49, seed=71, split_panel=True)

plans = {
    "anova": oe.power_anova(0.25, groups=4, power=0.8),
    "regression": oe.power_regression(0.125, predictors=5, tested=2, power=0.8),
    "paired": oe.power_paired_mean(0.5, sd_before=1.0, sd_after=1.0, correlation=0.5, power=0.8),
    "cluster": oe.power_cluster_mean(0.5, sd=1.0, cluster_size=20, icc=0.05, power=0.8),
    "survival": oe.power_logrank(0.7, power=0.8, event_fraction=0.3),
}
scenario = oe.planning_scenarios(
    "power_regression",
    [dict(effect=0.125, predictors=5, tested=2, power=p) for p in (0.8, 0.85, 0.9)],
)
plot = oe.planning_plot(scenario)
states = {
    name: oe.summary_state(output)
    for name, output in dict(
        weak=weak,
        ar_set=ar_set,
        wild=wild,
        wild_grid=wild_grid,
        fisher=fisher,
        prediction=prediction,
        resampled=resampled,
        scenario=scenario,
        **plans,
    ).items()
}
state_file = Path.cwd() / "eight-capability-state.json"
state_file.write_text(json.dumps(states, allow_nan=False, sort_keys=True))
reopened = json.loads(state_file.read_text())
assert reopened == states
restored_resampled = oe.restore_summary(reopened["resampled"])
quantile_means = oe.panel_mmqr_resampled_predict(restored_resampled, data=panel)
assert oe.restore_summary(reopened["wild"]).attrs == wild.attrs
assert plans["regression"]["plan"].iloc[0].n == 81
proof = dict(
    status="passed",
    states=len(states),
    file=str(state_file),
    json_restored=True,
    synthetic=True,
    new_methods_desktop_validated=False,
    vendor_execution=False,
    spatial_mean_rows=len(prediction["network mean"]),
    mmqr_rows=len(quantile_means["conditional means"]),
)
print("EIGHT_CAPABILITY_OK " + json.dumps(proof, sort_keys=True))
show = globals().get("display")
if show:
    for output, name in [
        (weak, "test"),
        (wild, "test"),
        (wild_grid, "grid"),
        (fisher, "test"),
        (prediction, "network mean"),
        (resampled, "coefficients"),
        (scenario, "scenarios"),
    ]:
        show(output[name])
    show(plot)
