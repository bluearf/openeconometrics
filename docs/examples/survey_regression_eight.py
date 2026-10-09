"""Four single-stage survey regressions and four saved coefficient procedures."""

import json

import pandas as pd
import openecon as oe

frame = pd.DataFrame({
    "w": [1., 2., 3., 1., 2., 4., 4., 2., 1., 3., 2., 1.],
    "p": [v for v in range(6) for _ in range(2)],
    "h": [0]*6 + [1]*6,
    "N": [12]*6 + [6]*6,
    "x": [0., 1.]*6,
    "y": [2., 5., 3., 2., 1., 4., 7., 6., 3., 1., 4., 2.],
    "b": [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    "c": [0, 2, 1, 0, 3, 1, 2, 4, 0, 1, 2, 0],
})
design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
models = {
    "linear": oe.survey_regress(frame, design, "y", ["x"]),
    "logit": oe.survey_logit(frame, design, "b", ["x"]),
    "probit": oe.survey_probit(frame, design, "b", ["x"]),
    "poisson": oe.survey_poisson(frame, design, "c", ["x"]),
}
states = {name: value.model_dump(mode="json") for name, value in models.items()}
restored = {
    name: oe.SurveyRegressionResult.model_validate_json(json.dumps(state, allow_nan=False))
    for name, state in states.items()
}
for name in models:
    assert restored[name] == models[name]
    pd.testing.assert_frame_equal(restored[name].to_frame(), models[name].to_frame())
    assert restored[name].metadata["stata_parity_validated"] is False
evaluation = pd.DataFrame({"x": [-1., 0., 1.]}, index=["low", "mid", "high"])
procedures = {
    "predict": oe.survey_predict(restored["probit"], evaluation),
    "margins": oe.survey_margins(restored["logit"], frame, variables=["x"], at={"x": .25}, weights="w"),
    "lincom": oe.survey_lincom(restored["linear"], [1., 2.], null=3.),
    "test": oe.survey_test(restored["poisson"], [[1., 0.], [0., 1.]], null=[0., 0.]),
}
poststates = {
    name: {"index": table.index.tolist(), "columns": table.columns.tolist(),
           "data": table.values.tolist(), "attrs": table.attrs}
    for name, table in procedures.items()
}
assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
if "display" not in globals():
    display = print
for table in [*(model.to_frame() for model in restored.values()), *procedures.values()]:
    assert oe.to_latex(table)
    display(table)
print("SURVEY_REGRESSION_METHOD_RECEIPT:" + json.dumps({
    "stages": 8, "rows": len(frame), "design_df": 4, "full_covariance_saved": True,
    "restored_equal": True, "conditional_margins": True, "stata_parity_validated": False,
}))
