"""Eight bounded causal-design procedures on synthetic data only.

Run in the OpenEconometrics code panel or as an ordinary Python script. Supply
ARTIFACT_DIRECTORY in the execution globals to choose an owned output folder;
otherwise a new temporary folder is created. Eight small representative tables
are displayed. The eight JSON files retain every table, dtype and scientific
state field and are read back without fitting again.
"""

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

import openecon as oe


# Every target, bin and horizon is fixed before the synthetic outcomes are made.
THRESHOLDS = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
QUANTILES = [0.25, 0.5, 0.75]
CUTPOINTS = {"baseline": [-0.5, 0.5], "x": [-0.5, 0.5]}
TAU = 2.5
NAMEORDER = [
    "ebalance",
    "cem",
    "balance",
    "rosenbaum_bounds",
    "paired_randomization",
    "treatment_cdf",
    "treatment_quantile",
    "treatment_rmst",
]
REPRESENTATIVE_TABLES = {
    "ebalance": "moments",
    "cem": "strata",
    "balance": "balance",
    "rosenbaum_bounds": "bounds",
    "paired_randomization": "test",
    "treatment_cdf": "effects",
    "treatment_quantile": "effects",
    "treatment_rmst": "effects",
}

rng = np.random.default_rng(42017)
n = 120
baseline = rng.normal(size=n)
x = rng.normal(size=n)
treated = rng.binomial(1, 0.5, size=n)
experiment = pd.DataFrame(
    {"baseline": baseline, "x": x, "treated": treated},
    index=[f"synthetic-unit-{i:03d}" for i in range(n)],
)
experiment["outcome"] = 1.0 + 1.4 * treated + 0.4 * baseline + rng.normal(size=n)

# The sample design is declared independently of balance diagnostics. Entropy
# and CEM are preprocessing methods; their weights do not certify identification.
results = {}
results["ebalance"] = oe.ebalance(experiment, "treated", ["baseline", "x"])
results["cem"] = oe.cem(
    experiment,
    "treated",
    ["baseline", "x"],
    cutpoints=CUTPOINTS,
)

# Join balancing weights by original physical positions, preserving row labels.
# This is only a fixed-weight balance diagnostic, not estimated-weight effect SE.
weighted = experiment.copy()
joined_weights = np.full(len(weighted), np.nan, dtype=float)
weight_table = results["ebalance"]["weights"]
joined_weights[weight_table.position.to_numpy(dtype=int)] = weight_table.weight.to_numpy(
    dtype=float
)
assert np.isfinite(joined_weights).all()
weighted["balance_weight"] = joined_weights
results["balance"] = oe.balance(
    weighted,
    "treated",
    ["baseline", "x"],
    balance_weights="balance_weight",
)

# Six independent synthetic pairs, each with equiprobable within-pair assignment.
# Exact randomization enumerates all 64 assignments, while Rosenbaum bounds use
# the paired sign statistic and do not provide an effect confidence interval.
pair_id = np.repeat([f"synthetic-pair-{i}" for i in range(6)], 2)
pair_treatment = np.column_stack([rng.binomial(1, 0.5, size=6), np.zeros(6, dtype=int)])
pair_treatment[:, 1] = 1 - pair_treatment[:, 0]
pair_treatment = pair_treatment.reshape(-1)
pair_baseline = np.repeat(rng.normal(size=6), 2)
paired = pd.DataFrame(
    {
        "pair": pair_id,
        "treated": pair_treatment,
        "outcome": pair_baseline + 0.8 * pair_treatment + rng.normal(scale=0.4, size=12),
    },
    index=[f"synthetic-paired-unit-{i:02d}" for i in range(12)],
)
results["rosenbaum_bounds"] = oe.rosenbaum_bounds(
    paired,
    "outcome",
    "treated",
    "pair",
    gammas=[1.0, 1.5, 2.0],
)
results["paired_randomization"] = oe.paired_randomization(
    paired,
    "outcome",
    "treated",
    "pair",
    design="paired_randomized",
    method="exact",
)
assert results["paired_randomization"].attrs["state"]["n_assignments"] == 64

results["treatment_cdf"] = oe.treatment_cdf(
    experiment,
    "outcome",
    "treated",
    design="randomized",
    thresholds=THRESHOLDS,
)
torch_rng_before = torch.random.get_rng_state().clone()
results["treatment_quantile"] = oe.treatment_quantile(
    experiment,
    "outcome",
    "treated",
    design="randomized",
    quantiles=QUANTILES,
    reps=199,
    seed=1729,
)
assert torch.equal(torch.random.get_rng_state(), torch_rng_before)

# Independent right censoring within each randomized arm. The fixed common
# horizon has observed support in both arms; no extrapolation is requested.
event_time = np.exp(1.0 + 0.25 * treated + rng.normal(scale=0.65, size=n))
censor_time = rng.uniform(1.5, 5.5, size=n)
survival = experiment.loc[:, ["treated"]].copy()
survival["time"] = np.minimum(event_time, censor_time)
survival["event"] = (event_time <= censor_time).astype(int)
assert all(survival.loc[survival.treated == arm, "time"].max() >= TAU for arm in (0, 1))
results["treatment_rmst"] = oe.treatment_rmst(
    survival,
    "time",
    "event",
    "treated",
    design="randomized",
    tau=TAU,
)
assert list(results) == NAMEORDER

if "ARTIFACT_DIRECTORY" in globals():
    artifact_directory = Path(globals()["ARTIFACT_DIRECTORY"])
else:
    artifact_directory = Path(tempfile.mkdtemp(prefix="openecon-causal-design-eight-"))
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
    "CAUSAL_DESIGN_EIGHT_OK "
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
