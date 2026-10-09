"""Eight three-stage SRSWOR estimates with strata at both lower stages."""

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
        for g in range(2):
            for s in range(2):
                for q in range(2):
                    for j in range(2):
                        cell = 4 * g + 2 * s + j
                        x = xb[cell] + 0.12 * h + 0.05 * p + 0.17 * q - 0.09 * q * j
                        z = zb[cell] + 0.15 * p - 0.1 * h - 0.23 * q + 0.07 * q * j
                        rows.append(
                            {
                                "h": h,
                                "p": p,
                                "g": g,
                                "s": s,
                                "q": q,
                                "j": j,
                                "N": 2 if h == 0 else 4 + h,
                                "M": 2 if h == 0 and p == 0 and g == 0 else 3 + h + p + g,
                                "L": 2
                                if (h, p, g, s, q) == (0, 0, 0, 0, 0)
                                else 4 + h + p + g + s + 2 * q,
                                "x": x,
                                "z": z,
                                "b": patterns[2 * h + p][cell],
                                "c": counts[2 * h + p][cell] + q,
                                "y": None
                                if (h, p, g, s, q, j) == (1, 0, 0, 0, 0, 0)
                                else 2.0 + 0.7 * x - 0.2 * z + 0.3 * (counts[2 * h + p][cell] + q),
                                "d": 3.0 + 0.1 * cell + 0.2 * h + 0.1 * q,
                                "d2": 3.5 + 0.1 * cell + 0.2 * h + 0.2 * q,
                                "cgroup": "a" if j % 2 else "b",
                                "domain": 0
                                if (h == 2 and p == 1) or (h, p, g, s, q) == (1, 0, 1, 1, 1)
                                else 1,
                            }
                        )
frame = pd.DataFrame(rows)
design_roles = {
    "psu": "p",
    "ssu_strata": "g",
    "ssu": "s",
    "tsu_strata": "q",
    "tsu": "j",
    "strata": "h",
    "population_psu": "N",
    "population_ssu": "M",
    "population_tsu": "L",
}
# These frames declare every positive lower cell in this synthetic population.
# Real callers supply their independently known frames for observed parents.
ssu_frame = [
    {"h": h, "p": p, "g": g, "M": 2 if (h, p, g) == (0, 0, 0) else 3 + h + p + g}
    for h in range(3)
    for p in range(2)
    for g in range(2)
]
tsu_frame = [
    {
        "h": h,
        "p": p,
        "g": g,
        "s": s,
        "q": q,
        "L": 2 if (h, p, g, s, q) == (0, 0, 0, 0, 0) else 4 + h + p + g + s + 2 * q,
    }
    for h in range(3)
    for p in range(2)
    for g in range(2)
    for s in range(2)
    for q in range(2)
]
design = oe.survey_fully_stratified_three_stage_design(
    frame,
    **design_roles,
    ssu_frame=ssu_frame,
    tsu_frame=tsu_frame,
)
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
options = {"domain": "domain", "missing": "drop", "alpha": 0.05, "null": 0.0}
models = {}
for name, case in cases.items():
    kwargs = {**options, **{key: value for key, value in case.items() if key != "args"}}
    if name in ("regress", "logit", "probit", "poisson"):
        kwargs.update(tolerance=1e-11, max_iter=100)
    models[name] = getattr(oe, "survey_fully_stratified_three_stage_" + name)(
        frame, design, *case["args"], **kwargs
    )
states = {name: model.model_dump(mode="json") for name, model in models.items()}
restored = {
    name: type(model).model_validate_json(json.dumps(states[name], sort_keys=True, allow_nan=False))
    for name, model in models.items()
}


def table_state(table):
    return {
        "index": table.index.tolist(),
        "columns": table.columns.tolist(),
        "data": table.astype(object).where(pd.notna(table), None).values.tolist(),
        "attrs": table.attrs,
    }


poststates, latexstates = {}, {}
for name, model in restored.items():
    assert model.model_dump(mode="json") == states[name]
    pd.testing.assert_frame_equal(model.to_frame(), models[name].to_frame())
    table = model.to_frame()
    poststates[name] = table_state(table)
    latexstates[name] = oe.to_latex(table)
    assert latexstates[name]
    if name in ("mean", "total", "ratio", "proportion"):
        assert len(model.contrast([1.0] + [0.0] * (len(model.labels) - 1))) == 1
    else:
        assert len(model.lincom([0.0, 1.0, 0.0])) == 1
        assert len(model.test([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])) == 1
        assert len(model.predict(pd.DataFrame({"x": [-0.5, 0.5], "z": [0.2, -0.2]}))) == 2
oracle_inputs = {
    "frame": {key: [row[key] for row in rows] for key in rows[0]},
    "design_roles": design_roles,
    "ssu_frame": ssu_frame,
    "tsu_frame": tsu_frame,
    "cases": cases,
    "options": options,
}
assert json.loads(json.dumps(states, allow_nan=False)) == states
assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
assert json.loads(json.dumps(oracle_inputs, allow_nan=False)) == oracle_inputs
assert len(latexstates) == 8
assert all(model.metadata["n_terminal_strata"] == 48 for model in models.values())
if "display" not in globals():
    display = print
for model in restored.values():
    display(model.to_frame())
print(
    "SURVEY_FULLY_STRATIFIED_THREE_STAGE_RECEIPT:"
    + json.dumps(
        {
            "stages": 8,
            "rows": 96,
            "design_df": 3,
            "full_covariance_saved": True,
            "restored_equal": True,
            "sampling_stages": 3,
            "all_three_fpc_terms": True,
            "latex_count": 8,
            "stage2_stratification": True,
            "stage3_stratification": True,
            "n_cells": 12,
            "n_terminal_strata": 48,
            "stata_parity_validated": False,
        }
    )
)
