"""Eight single-stage survey methods, complete saved results and joint contrasts."""

from itertools import product
import json

import pandas as pd
import openecon as oe

frame = pd.DataFrame(
    {
        "w": [2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0],
        "p": [1, 1, 2, 2, 1, 1, 2, 2],
        "h": ["a"] * 4 + ["b"] * 4,
        "y": [1.0, 3.0, 2.0, 5.0, 6.0, 8.0, 7.0, 11.0],
        "x": [2.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 6.0],
        "domain": [1, 1, 1, 1, 0, 0, 0, 0],
        "category": ["a", "a", "a", "b", "b", "a", "b", "b"],
    }
)
design = oe.survey_design(frame, weights="w", psu="p", strata="h")
weights = {}
for i, choices in enumerate(product((1, 2), repeat=2)):
    weights[f"enumeration-{i}"] = [
        w * 2 if p == choices[0 if h == "a" else 1] else 0.0
        for w, p, h in zip(frame.w, frame.p, frame.h)
    ]
replicas = pd.DataFrame(weights)
results = {
    "mean": oe.survey_mean(frame, design, ["y", "x"]),
    "total": oe.survey_total(frame, design, ["y", "x"], domain="domain"),
    "ratio": oe.survey_ratio(frame, design, ["y", "x"], ["x", "y"]),
    "proportion": oe.survey_proportion(frame, design, "category", categories=["a", "b", "absent"]),
    "brr": oe.survey_brr(frame, design, ["y", "x"]),
    "fay": oe.survey_fay(frame, design, ["y", "x"], rho=0.3, domain="domain"),
    "jackknife": oe.survey_jackknife(frame, design, ["y", "x"], centering="stratum_mean"),
    "bootstrap": oe.survey_bootstrap(
        frame,
        design,
        ["y", "x"],
        replicas,
        justification="Exhaustive stratified two-PSU Rao-Wu rescaled bootstrap selections, one PSU per stratum",
        scale=0.25,
        centering="replicate_mean",
        df=2,
    ),
}
states = {name: value.model_dump(mode="json") for name, value in results.items()}
reopened = json.loads(json.dumps(states, allow_nan=False))
if "display" not in globals():
    display = print
for name, original in results.items():
    restored = oe.SurveyResult.model_validate(reopened[name])
    assert restored == original
    assert restored.metadata["stata_parity_validated"] is False
    table = restored.to_frame()
    pd.testing.assert_frame_equal(table, original.to_frame())
    coefficients = [1.0] + [-1.0] * (len(original.labels) - 1)
    pd.testing.assert_frame_equal(restored.contrast(coefficients), original.contrast(coefficients))
    assert oe.to_latex(table)
    display(table)
print(
    "SURVEY_METHOD_RECEIPT:"
    + json.dumps(
        {
            "stages": 8,
            "rows": len(frame),
            "full_covariance_saved": True,
            "restored_equal": True,
            "stata_parity_validated": False,
        }
    )
)
