"""One declared hypothesis family, supplied joint null, and compatible intervals.

The sign orbit below is a synthetic, complete equiprobable null enumeration.
Real analyses must justify their own null design and joint covariance.
"""

# ruff: noqa: F821 -- display is supplied by the OpenEconometrics workspace.

import itertools
import json

import openecon as oe

labels = ["Employment", "Income", "Productivity"]
p_values = [.01, .2, .03]
adjusted = oe.multipletests(p_values, labels=labels, method="holm")
assert adjusted["adjusted_p_value"].tolist() == [.03, .2, .06]
display(adjusted)

joint_null = [
    [sign[0] * 1.5, sign[1] * .4, sign[2] * 1.0]
    for sign in itertools.product([-1, 1], repeat=3)
]
joint = oe.stepdown(
    [1.5, .4, 1.0], joint_null,
    labels=labels, tail="greater", calibration="enumerated",
    null_description="Synthetic independent Rademacher signs conditional on fixed magnitudes; all eight equally likely assignments",
)
assert joint["adjusted_p_value"].tolist() == [.5, .5, .5]
display(joint)

intervals = oe.simultaneous_ci(
    [.3, .2, .1], [[.04, .01, 0], [.01, .09, .01], [0, .01, .01]],
    labels=labels, draws=20000, seed=1729,
    family_description="Synthetic compatible jointly normal estimates with the declared known full covariance",
)
assert intervals["ci_low"].lt(intervals["estimate"]).all()
assert intervals["ci_high"].gt(intervals["estimate"]).all()
display(intervals)

tables = [adjusted, joint, intervals]
states = [
    {"schema": 1, "table": json.loads(table.to_json(orient="table", double_precision=15)),
     "attrs": table.attrs}
    for table in tables
]
json.dumps(states, allow_nan=False)
print("MULTIPLE_TESTING_RECEIPT:" + json.dumps({
    "procedures": ["multipletests", "stepdown", "simultaneous_ci"],
    "family_labels": labels, "raw_p_values": p_values,
    "holm_adjusted_p_values": adjusted["adjusted_p_value"].tolist(),
    "full_covariance_saved": intervals.attrs["joint_covariance"],
    "normal_critical_value": intervals.attrs["critical_value"],
    "cdf_monte_carlo_std_error": intervals.attrs["cdf_monte_carlo_std_error"],
    "stata_parity_validated": False,
}))
