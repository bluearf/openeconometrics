"""Eight prespecified causal sensitivity/assignment methods on synthetic data.

Run in the code panel or ordinary Python. The eight full JSON artifacts retain
all tables, inputs and assumptions and are restored exactly without refitting.
"""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import openecon as oe

NAMEORDER = ["ovb_sensitivity", "evalue", "manski_ate", "lee_bounds",
             "randomization_test", "stratified_randomization", "rosenbaum_rank_bounds", "bias_sensitivity"]
REPRESENTATIVE_TABLES = {"ovb_sensitivity": "sensitivity", "evalue": "evalues",
                        "manski_ate": "bounds", "lee_bounds": "bounds",
                        "randomization_test": "test", "stratified_randomization": "test",
                        "rosenbaum_rank_bounds": "bounds", "bias_sensitivity": "sensitivity"}
results = {}
rng = np.random.default_rng(72517)
x = rng.normal(size=100)
d = .3*x + rng.normal(size=100)
y = 1.0 + 1.5*d + .4*x + rng.normal(size=100)
regression = pd.DataFrame({"x": x, "d": d, "y": y}, index=[f"ols-unit-{i}" for i in range(100)])
results["ovb_sensitivity"] = oe.ovb_sensitivity(regression, "y", "d", ["x"],
    r2_treatment=[0., .1, .3], r2_outcome=[0., .1, .3], direction="both")
ratios = pd.DataFrame({"rr": [2., .5, 1.2], "lo": [1.2, .3, .8], "hi": [3., .8, 1.6]},
                      index=["harmful", "protective", "crosses-null"])
results["evalue"] = oe.evalue(ratios, "rr", lower="lo", upper="hi")
experiment = pd.DataFrame({"d": [0,1,0,1,0,1,0,1], "y": [1.,3.,2.,4.,3.,5.,4.,6.],
                           "stratum": ["A"]*4 + ["B"]*4}, index=[f"random-unit-{i}" for i in range(8)])
results["manski_ate"] = oe.manski_ate(experiment, "y", "d", lower=0., upper=10.)
selection = pd.DataFrame({"d": [0]*6 + [1]*6,
    "s": [1,1,1,0,0,0,1,1,1,1,1,0],
    "y": [1.,2.,3.,np.nan,np.nan,np.nan,2.,3.,4.,5.,6.,np.nan]})
results["lee_bounds"] = oe.lee_bounds(selection, "y", "d", "s",
                                      design="randomized", monotonicity="increasing")
torch_rng_before = torch.random.get_rng_state().clone()
results["randomization_test"] = oe.randomization_test(experiment, "y", "d",
    design="complete_randomized", null_effect=.5, method="exact")
results["stratified_randomization"] = oe.stratified_randomization(experiment, "y", "d", "stratum",
    design="stratified_randomized", null_effect=.5, method="monte_carlo", draws=199, seed=1729)
assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
paired = pd.DataFrame({"pair": np.repeat([f"pair-{i}" for i in range(6)],2),
                       "d": [0,1]*6, "y": [0.,1.,0.,-1.,0.,2.,0.,0.,0.,3.,0.,-2.]})
results["rosenbaum_rank_bounds"] = oe.rosenbaum_rank_bounds(paired, "y", "d", "pair", gammas=[1.,1.5,2.])
bias = pd.DataFrame({"estimate": [1.4,-.3,2.], "contrast": [.2,-1.,.5],
                     "lower": [-.1,-.5,0.], "upper": [.8,.1,.3]}, index=["post-1","post-2","post-3"])
results["bias_sensitivity"] = oe.bias_sensitivity(bias, "estimate", "contrast", "lower", "upper",
    covariance=[[.3,.05,-.02],[.05,.2,.01],[-.02,.01,.4]], restriction="fixed_external")
assert list(results) == NAMEORDER

if "ARTIFACT_DIRECTORY" in globals():
    artifact_directory = Path(globals()["ARTIFACT_DIRECTORY"])
else:
    artifact_directory = Path(tempfile.mkdtemp(prefix="openecon-causal-sensitivity-eight-"))
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
    "CAUSAL_SENSITIVITY_EIGHT_OK "
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
