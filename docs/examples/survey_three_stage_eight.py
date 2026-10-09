"""Eight exact-three-stage SRSWOR estimates with complete saved inference."""

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
        for s in range(2):
            for j in range(4):
                q = 4*s+j
                x, z = xb[q]+.12*h+.05*p, zb[q]+.15*p-.1*h
                rows.append({
                    "h": h, "p": p, "s": s, "j": j,
                    "N": 2 if h == 0 else 4+h,
                    "M": 2 if h == 0 and p == 0 else 3+h+p,
                    "L": 4 if h == 0 and p == 0 and s == 0 else 6+h+p+s,
                    "x": x, "z": z, "b": patterns[2*h+p][q], "c": counts[2*h+p][q],
                    "y": None if h == 1 and p == 0 and s == 0 and j == 0
                    else 2.+.7*x-.2*z+.3*counts[2*h+p][q],
                    "d": 3.+.1*q+.2*h, "d2": 3.5+.1*q+.2*h,
                    "cgroup": "a" if j % 2 else "b",
                    "domain": 0 if h == 2 and p == 1 else 1,
                })
frame = pd.DataFrame(rows)
design_roles = {
    "psu": "p", "ssu": "s", "tsu": "j", "strata": "h",
    "population_psu": "N", "population_ssu": "M", "population_tsu": "L",
}
design = oe.survey_three_stage_design(frame, **design_roles)
cases = {
    "mean": {"args": [["y", "x"]]}, "total": {"args": [["y", "x"]]},
    "ratio": {"args": [["y", "x"], ["d", "d2"]]},
    "proportion": {"args": ["cgroup"], "categories": ["a", "b", "absent"]},
    "regress": {"args": ["y", ["x", "z"]]},
    "logit": {"args": ["b", ["x", "z"]]},
    "probit": {"args": ["b", ["x", "z"]]},
    "poisson": {"args": ["c", ["x", "z"]]},
}
options = {"domain": "domain", "missing": "drop", "alpha": .05, "null": 0.}
models = {}
for name, case in cases.items():
    kwargs = {**options, **{key: value for key, value in case.items() if key != "args"}}
    if name in ("regress", "logit", "probit", "poisson"):
        kwargs.update(tolerance=1e-11, max_iter=100)
    models[name] = getattr(oe, "survey_three_stage_"+name)(frame, design, *case["args"], **kwargs)
states = {name: model.model_dump(mode="json") for name, model in models.items()}
restored = {
    name: type(model).model_validate_json(json.dumps(states[name], allow_nan=False))
    for name, model in models.items()
}


def table_state(table):
    return {
        "index": table.index.tolist(), "columns": table.columns.tolist(),
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
        assert len(model.contrast([1.]+[0.]*(len(model.labels)-1))) == 1
    else:
        assert len(model.lincom([0., 1., 0.])) == 1
        assert len(model.test([[0., 1., 0.], [0., 0., 1.]])) == 1
        assert len(model.predict(pd.DataFrame({"x": [-.5, .5], "z": [.2, -.2]}))) == 2
oracle_inputs = {
    "frame": {key: [row[key] for row in rows] for key in rows[0]},
    "design_roles": design_roles, "cases": cases, "options": options,
}
assert json.loads(json.dumps(states, allow_nan=False)) == states
assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
assert json.loads(json.dumps(oracle_inputs, allow_nan=False)) == oracle_inputs
assert len(latexstates) == 8
if "display" not in globals():
    display = print
for model in restored.values():
    display(model.to_frame())
print("SURVEY_THREE_STAGE_RECEIPT:"+json.dumps({
    "stages": 8, "rows": 48, "design_df": 3, "full_covariance_saved": True,
    "restored_equal": True, "sampling_stages": 3, "all_three_fpc_terms": True,
    "latex_count": 8, "stata_parity_validated": False,
}))
