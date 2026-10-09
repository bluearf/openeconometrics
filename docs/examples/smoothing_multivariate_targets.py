"""Saved Gaussian local derivatives and scalar RM contrasts; synthetic inputs."""

import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd

import openecon as oe
from openecon.econometrics.core import TableSet


rng = random.Random(4473)
frame = pd.DataFrame(
    [(a := rng.uniform(-1, 1), b := rng.uniform(-1, 1),
      math.sin(2 * a) + b * b + rng.gauss(0, .1)) for _ in range(80)],
    columns=["a", "b", "y"],
)
original_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
query = frame.iloc[[5, 20, 50, 70]].drop(columns="y")
payload = {}
for method in ("kernelreg", "localreg"):
    fitted = getattr(oe, method)(data=frame, y="y", x=["a", "b"], kernel="gaussian",
                                 selection="fixed", bandwidth=[.7, .6])
    restored = oe.ResultBundle.model_validate_json(fitted.model_dump_json())
    derivatives = oe.local_derivatives(restored, data=query)
    averages = oe.local_average_derivatives(restored, data=query)
    targets = TableSet({"pointwise": derivatives, "average": averages},
                       **{**derivatives.attrs, "title": "Saved local derivative targets"})
    saved = oe.summary_state(targets)
    replay = oe.restore_summary(saved)
    assert replay["pointwise"].equals(derivatives)
    assert replay["average"].equals(averages)
    payload[method] = {"result": restored.model_dump(mode="json"), "targets": json.loads(saved)}

records = []
for subject in range(31):
    group = "A" if subject < 13 else "B"
    common = rng.gauss(0, 1)
    for time in range(3):
        outcome = .3 * time + (group == "A") * (.2 + .25 * time) + common + rng.gauss(0, .5 + .2 * time)
        records.append([subject, group, time, outcome])
long = pd.DataFrame(records, columns=["id", "group", "time", "y"])
rm = oe.rm_anova(oe.Dataset.from_frame(long), "y", "id", ["time"], between=["group"])
saved_rm = oe.summary_state(rm)
restored_rm = oe.restore_summary(saved_rm)
contrast = oe.rm_contrast(restored_rm, [-1, 0, 1], between_contrast=[0, 2])
saved_contrast = oe.summary_state(contrast)
assert oe.restore_summary(saved_contrast)["contrast"].to_numpy().tolist() == contrast["contrast"].to_numpy().tolist()
payload["repeated_measures"] = {"result": json.loads(saved_rm), "contrast": json.loads(saved_contrast)}
assert original_hash == hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
path = Path("smoothing_multivariate_targets.json")
path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
assert json.loads(path.read_text()) == payload
print("LOCAL_RM_TARGETS_QA " + json.dumps({
    "saved_gaussian_derivative_models": 2, "derivative_query_rows": 4,
    "whole_state_file_readback": True, "original_inputs_unchanged": True,
    "rm_subjects": 31, "rm_cells": 3, "source_state_sha256": contrast.attrs["source_state_sha256"],
    "derivative_inference_available": False, "rm_scalar_t_df": contrast["contrast"].df.iloc[0],
    "licensed_vendor_comparison": False, "new_desktop_validation": False,
}, default=int))
