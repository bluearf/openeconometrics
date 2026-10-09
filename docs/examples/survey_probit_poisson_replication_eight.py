"""Eight probit/Poisson single-stage PSU coefficient-replication workflows.

Deterministic synthetic data; every PSU has both binary values and zero/positive counts. This
example needs only the frozen runtime. The independent development oracle uses
the saved primitive inputs, never this runtime's statistical implementation.
"""

import json

import pandas as pd
import openecon as oe

rows = []
xb = [-1., -.6, -.1, .4, .9, 1.4, -1.4, 1.8]
zb = [.4, -.5, .8, -.3, .6, -.7, -.9, 1.1]
patterns = [
    [0, 1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1],
]
counts = [
    [0, 2, 1, 0, 3, 1, 0, 2], [2, 0, 1, 3, 0, 2, 1, 0],
    [0, 1, 2, 1, 0, 3, 2, 1], [1, 2, 0, 2, 1, 0, 3, 1],
    [0, 2, 1, 3, 2, 0, 1, 2], [2, 1, 0, 1, 3, 2, 0, 1],
]
for h in range(3):
    for p in range(2):
        for j in range(8):
            x, z = xb[j]+.12*h+.05*p, zb[j]+.15*p-.1*h
            rows.append({
                "h": h, "p": p, "w": 1.+.2*h+.15*p+.07*j,
                "x": x, "z": z,
                "b": patterns[2*h+p][j], "c": counts[2*h+p][j], "domain": 1,
            })
frame = pd.DataFrame(rows)
design = oe.survey_design(frame, weights="w", psu="p", strata="h")
bootstrap_factors = [
    [1.3, .7, 1.1, .9, .8, 1.2], [.7, 1.3, .9, 1.1, 1.2, .8],
    [1.15, .85, .75, 1.25, 1.05, .95], [.85, 1.15, 1.25, .75, .95, 1.05],
    [1.4, .6, .8, 1.2, 1.1, .9], [.6, 1.4, 1.2, .8, .9, 1.1],
    [1.05, .95, 1.3, .7, .7, 1.3], [.95, 1.05, .7, 1.3, 1.3, .7],
]
replicate_weights = pd.DataFrame({
    f"whole-psu-{r}": [row["w"]*factor[2*row["h"]+row["p"]] for row in rows]
    for r, factor in enumerate(bootstrap_factors)
})
options = {
    "brr": {"replicates": 4},
    "fay": {"replicates": 4, "rho": .5},
    "jackknife": {"centering": "stratum_mean"},
    "bootstrap": {"centering": "replicate_mean", "scale": 1/8,
                  "rscales": [1., 1., 1.2, 1.2, .8, .8, 1.1, 1.1],
                  "justification": "Declared whole-PSU synthetic perturbations; no row resampling."},
}
models = {}
for family, function, outcome in [
    ("probit", oe.survey_probit_replicate, "b"),
    ("poisson", oe.survey_poisson_replicate, "c"),
]:
    for method, method_options in options.items():
        models[f"{family}_{method}"] = function(
            frame, design, outcome, ["x", "z"], method=method,
            tolerance=1e-11,
            **method_options,
            **({"replicate_weights": replicate_weights} if method == "bootstrap" else {}),
        )
states = {case: model.model_dump(mode="json") for case, model in models.items()}
restored = {
    case: oe.SurveyRegressionResult.model_validate_json(json.dumps(state, allow_nan=False))
    for case, state in states.items()
}
evaluation = pd.DataFrame({"x": [-.8, .3, 1.2], "z": [.5, -.2, .9]},
                          index=["low", "middle", "high"])
contrast = [1., .4, -.3]
restrictions = [[0., 1., 0.], [0., 0., 1.]]


def table_state(table):
    return {"index": table.index.tolist(), "columns": table.columns.tolist(),
            "data": table.values.tolist(), "attrs": table.attrs}


poststates = {}
for case, model in restored.items():
    assert model == models[case]
    pd.testing.assert_frame_equal(model.to_frame(), models[case].to_frame())
    assert model.metadata["replication"]["failed_replicates"] == []
    assert model.metadata["stata_parity_validated"] is False
    procedures = {
        "coefficients": model.to_frame(),
        "predict": oe.survey_predict(model, evaluation),
        "margins": oe.survey_margins(model, frame, weights="w"),
        "lincom": oe.survey_lincom(model, contrast),
        "test": oe.survey_test(model, restrictions),
    }
    poststates[case] = {name: table_state(table) for name, table in procedures.items()}
oracle_inputs = {
    "frame": frame.to_dict(orient="list"), "regressors": ["x", "z"],
    "fit_options": {"tolerance": 1e-11, "max_iter": 100, "intercept": True, "alpha": .05, "null": 0.},
    "options": options,
    "bootstrap_weights": {"columns": replicate_weights.columns.tolist(),
                          "data": replicate_weights.values.tolist()},
    "evaluation": {"data": evaluation.values.tolist(), "index": evaluation.index.tolist()},
    "contrast": contrast, "restrictions": restrictions,
}
assert json.loads(json.dumps(states, allow_nan=False)) == states
assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
assert json.loads(json.dumps(oracle_inputs, allow_nan=False)) == oracle_inputs
if "display" not in globals():
    display = print
for model in restored.values():
    table = model.to_frame()
    assert oe.to_latex(table)
    display(table)
print("SURVEY_PROBIT_POISSON_REPLICATION_RECEIPT:"+json.dumps({
    "stages": 8, "rows": 48, "design_df": 3, "full_covariance_saved": True,
    "restored_equal": True, "whole_psu_replicates": True, "stata_parity_validated": False,
}))
