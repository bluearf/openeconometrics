"""Eight bounded two-stage SRSWOR estimates with full saved-state inference."""

import json
import pandas as pd
import openecon as oe

rows = []
xb = [-1.0, -0.6, -0.1, 0.4, 0.9, 1.4, -1.4, 1.8]
zb = [0.4, -0.5, 0.8, -0.3, 0.6, -0.7, -0.9, 1.1]
patterns = [
    [0, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 0, 1],
]
counts = [
    [0, 2, 1, 0, 3, 1, 0, 2],
    [2, 0, 1, 3, 0, 2, 1, 0],
    [0, 1, 2, 1, 0, 3, 2, 1],
    [1, 2, 0, 2, 1, 0, 3, 1],
    [0, 2, 1, 3, 2, 0, 1, 2],
    [2, 1, 0, 1, 3, 2, 0, 1],
]
for h in range(3):
    for p in range(2):
        for j in range(8):
            x, z = xb[j] + 0.12 * h + 0.05 * p, zb[j] + 0.15 * p - 0.1 * h
            rows.append(
                {
                    "h": h,
                    "p": p,
                    "j": j,
                    "N": 2 if h == 0 else 4 + h,
                    "M": 8 if h == 0 and p == 0 else 12 + 2 * h + p,
                    "x": x,
                    "z": z,
                    "b": patterns[2 * h + p][j],
                    "c": counts[2 * h + p][j],
                    "y": None
                    if h == 1 and p == 0 and j == 0
                    else 2.0 + 0.7 * x - 0.2 * z + 0.3 * counts[2 * h + p][j],
                    "d": 3.0 + 0.1 * j + 0.2 * h,
                    "cgroup": "a" if j % 2 else "b",
                    "domain": 0 if h == 2 and p == 1 else 1,
                }
            )
frame = pd.DataFrame(rows)
design_roles = dict(psu="p", ssu="j", strata="h", population_psu="N", population_ssu="M")
design = oe.survey_two_stage_design(frame, **design_roles)
cases = {
    "mean": {"args": [["y", "x"]]},
    "total": {"args": [["y", "x"]]},
    "ratio": {"args": [["y", "x"], ["d", "d2"]]},
    "proportion": {"args": ["cgroup"], "categories": ["a", "b", "absent"]},
    "regress": {"args": ["y", ["x", "z"]]},
    "logit": {"args": ["b", ["x", "z"]]},
    "probit": {"args": ["b", ["x", "z"]]},
    "poisson": {"args": ["c", ["x", "z"]]},
}
frame["d2"] = frame["d"] + 0.5
for row in rows:
    row["d2"] = row["d"] + 0.5
models = {}
for name, case in cases.items():
    options = {k: v for k, v in case.items() if k != "args"}
    if name in ("logit", "probit", "poisson"):
        options["tolerance"] = 1e-11
    models[name] = getattr(oe, "survey_two_stage_" + name)(
        frame, design, *case["args"], domain="domain", missing="drop", **options
    )
states = {name: model.model_dump(mode="json") for name, model in models.items()}
restored = {
    name: type(model).model_validate_json(json.dumps(states[name], allow_nan=False))
    for name, model in models.items()
}


def table_state(table):
    return {
        "index": table.index.tolist(),
        "columns": table.columns.tolist(),
        "data": table.astype(object).where(pd.notna(table), None).values.tolist(),
        "attrs": table.attrs,
    }


poststates = {}
for name, model in restored.items():
    assert model.model_dump(mode="json") == states[name]
    pd.testing.assert_frame_equal(model.to_frame(), models[name].to_frame())
    poststates[name] = table_state(model.to_frame())
    if name in ("mean", "total", "ratio", "proportion"):
        assert len(model.contrast([1.0] + [0.0] * (len(model.labels) - 1))) == 1
    else:
        assert len(model.lincom([0.0, 1.0, 0.0])) == 1
        assert len(model.predict(pd.DataFrame({"x": [-0.5, 0.5], "z": [0.2, -0.2]}))) == 2
oracle_inputs = {
    "frame": {key: [row[key] for row in rows] for key in rows[0]},
    "design_roles": design_roles,
    "cases": cases,
    "options": {
        "domain": "domain",
        "missing": "drop",
        "alpha": 0.05,
        "null": 0.0,
        "tolerance": 1e-11,
    },
}
assert json.loads(json.dumps(states, allow_nan=False)) == states
assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
if "display" not in globals():
    display = print
for model in restored.values():
    table = model.to_frame()
    assert oe.to_latex(table)
    display(table)
print(
    "SURVEY_TWO_STAGE_RECEIPT:"
    + json.dumps(
        {
            "stages": 8,
            "rows": 48,
            "design_df": 3,
            "full_covariance_saved": True,
            "restored_equal": True,
            "sampling_stages": 2,
            "both_fpc_terms": True,
            "stata_parity_validated": False,
        }
    )
)
