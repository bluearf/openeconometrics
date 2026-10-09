"""Eight declared observational causal targets and formal OVB options.

All rows are synthetic; logit treatment/CDF models match this generation model.
Survival treatment and censor nuisances are known from the generation parameters.
The complete eight artifacts retain every table, covariance, input and fit state.
"""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import openecon as oe

NAMEORDER = ["ovb_benchmark", "ovb_robustness", "treatment_cdf_ipw", "treatment_quantile_ipw",
             "treatment_cdf_aipw", "treatment_survival_ipcw", "treatment_rmst_ipcw", "treatment_rmst_aipw"]
REPRESENTATIVE_TABLES = {name: "effects" for name in NAMEORDER}
REPRESENTATIVE_TABLES.update(ovb_benchmark="bounds", ovb_robustness="robustness")
results = {}
rng = np.random.default_rng(20261007)
x = rng.normal(size=120)
d = .2*x + rng.normal(size=120)
y = 1. + 1.2*d + .15*x + rng.normal(size=120)
regression = pd.DataFrame({"x": x, "d": d, "y": y}, index=[f"ols-{i}" for i in range(120)])
base = oe.ovb_sensitivity(regression, "y", "d", ["x"], r2_treatment=[0.], r2_outcome=[0.])
results["ovb_benchmark"] = oe.ovb_benchmark(base, groups={"pretreatment": ["x"]}, kd=[1.,2.], ky=[1.,2.],
                                           assumption="residualized_benchmark")
results["ovb_robustness"] = oe.ovb_robustness(base, q=[.5,1.], alpha=[.05,1.])

# Logistic errors give exactly linear-logit conditional CDFs at every fixed threshold.
x = rng.normal(size=240)
p = 1/(1 + np.exp(-.3*x))
d = rng.binomial(1,p)
y = .4*d + .3*x + rng.logistic(size=len(x))
observed = pd.DataFrame({"x": x, "d": d, "y": y}, index=[f"observed-{i}" for i in range(len(x))])
torch_rng_before = torch.random.get_rng_state().clone()
results["treatment_cdf_ipw"] = oe.treatment_cdf_ipw(observed, "y", "d", x=["x"],
    design="unconfounded", thresholds=[-.25,.75])
results["treatment_quantile_ipw"] = oe.treatment_quantile_ipw(observed, "y", "d", x=["x"],
    design="unconfounded", quantiles=[.25,.5,.75], reps=99, seed=1729)
results["treatment_cdf_aipw"] = oe.treatment_cdf_aipw(observed, "y", "d", x=["x"],
    design="unconfounded", thresholds=[-.25,.75], folds=3, seed=1729)
assert torch.equal(torch.random.get_rng_state(), torch_rng_before)

x = rng.normal(size=160)
p = 1/(1 + np.exp(-.3*x))
d = rng.binomial(1,p)
r0 = .25*np.exp(.1*x)
r1 = .18*np.exp(.1*x)
event_rate = np.where(d==1,r1,r0)
event_time = rng.exponential(1/event_rate)
censor_rate = .08 + .03*d + .02/(1+np.exp(-x))
censor_rate[::7] = 0. # Exact infinite-censoring-support limit.
censor_time = np.full(len(x),np.inf)
positive = censor_rate>0
censor_time[positive] = rng.exponential(1/censor_rate[positive])
survival = pd.DataFrame({"time": np.minimum(event_time,censor_time), "event": (event_time<=censor_time).astype(int),
                         "d": d, "p": p, "rate": censor_rate}, index=[f"survival-{i}" for i in range(len(x))])
thresholds = [0.,.5,1.,2.,3.]
g_columns=[]
for j,t in enumerate(thresholds):
    name=f"known_g_{j}"
    g_columns.append(name)
    survival[name]=np.exp(-censor_rate*t)
results["treatment_survival_ipcw"] = oe.treatment_survival_ipcw(survival,"time","event","d","p",g_columns,
    design="unconfounded", nuisance="known", thresholds=thresholds)
results["treatment_rmst_ipcw"] = oe.treatment_rmst_ipcw(survival,"time","event","d","p","rate",
    design="unconfounded", nuisance="known", tau=3.)
complete = pd.DataFrame({"time": event_time, "event": np.ones(len(x),dtype=int), "d": d, "p": p,
                         "m0": -np.expm1(-3*r0)/r0, "m1": -np.expm1(-3*r1)/r1}, index=survival.index)
results["treatment_rmst_aipw"] = oe.treatment_rmst_aipw(complete,"time","event","d","p","m0","m1",
    design="unconfounded", nuisance="known", tau=3.)
assert list(results) == NAMEORDER

if "ARTIFACT_DIRECTORY" in globals():
    artifact_directory = Path(globals()["ARTIFACT_DIRECTORY"])
else:
    artifact_directory = Path(tempfile.mkdtemp(prefix="openecon-causal-targets-eight-"))
artifact_directory.mkdir(parents=True, exist_ok=True)

if "display" in globals():
    show = globals()["display"]
else:
    # Ordinary scripts still display the same eight bounded tables.
    def show(table):
        print(table.to_string(index=False))


artifact_sha256s = {}
table_counts = {}
for name in NAMEORDER:
    output = results[name]
    path = artifact_directory / f"{name}.json"
    artifact = oe.causal_design_save(output, path)
    restored = oe.causal_design_load(path)
    restored_artifact = oe.causal_design_save(restored)
    assert restored_artifact == artifact  # Every table, dtype and state field.
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert restored.attrs == output.attrs
    assert restored.keys() == output.keys()
    for key in output:
        pd.testing.assert_frame_equal(restored[key], output[key], check_exact=True)
    assert output.attrs["stata_parity_validated"] is False
    artifact_sha256s[name] = artifact["sha256"]
    table_counts[name] = len(output)
    show(output[REPRESENTATIVE_TABLES[name]].head(8))

print(
    "CAUSAL_TARGETS_EIGHT_OK "
    + json.dumps(
        {
            "nameorder": NAMEORDER,
            "artifact_sha256s": artifact_sha256s,
            "full_roundtrip_equal": True,
            "frozen": bool(getattr(sys, "frozen", False)),
            "table_counts": table_counts,
        },
        sort_keys=True,
    )
)
