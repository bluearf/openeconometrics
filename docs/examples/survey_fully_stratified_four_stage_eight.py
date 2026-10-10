"""Eight estimates with explicit strata and complete frames at all four selections."""

import json
from itertools import product
import pandas as pd
import openecon as oe


def fixture():
    rows = []
    for h, p, g, s, q, t, r, f in product(range(2), repeat=8):
        x = (-1.5, 0.5, 1.5, -0.5)[2 * r + f] + 0.1 * h + 0.04 * p + 0.03 * s + 0.02 * t
        z = (-0.4, 0.8, 0.3, -0.9)[2 * r + f] + 0.2 * p - 0.15 * g + 0.12 * s + 0.1 * q
        j = (2 * r + f + h + p + g + s + q + t) % 4
        c = (0, 2, 1, 3)[j]
        rows.append(
            dict(
                h=h,
                p=p,
                g=g,
                s=s,
                q=q,
                t=t,
                r=r,
                f=f,
                N=2 if h == 0 else 4,
                M=2 if (h, p, g) == (0, 0, 0) else 3 + g + p,
                L=2 if (h, p, g, s, q) == (0, 0, 0, 0, 0) else 4 + q + s,
                K=2 if (h, p, g, s, q, t, r) == (0, 0, 0, 0, 0, 0, 0) else 3 + r + t,
                x=x,
                z=z,
                b=(0, 1, 1, 0)[j],
                c=c,
                y=None
                if (h, p, g, s, q, t, r, f) == (1, 0, 0, 0, 0, 0, 0, 0)
                else 1.2 + 0.6 * x - 0.3 * z + 0.15 * c + 0.03 * p * g,
                d=3.0 + 0.1 * j + 0.2 * h,
                d2=3.5 + 0.1 * j + 0.2 * g,
                cgroup="a" if 2 * r + f < 1 + (h + p + g + s + q + t) % 3 else "b",
                domain=int(not (h == 1 and p == 1)),
            )
        )
    frame = pd.DataFrame(rows)
    return frame, roles(frame)


def roles(frame, *, census=()):
    columns = ["h", "p", "g", "s", "q", "t", "r"]
    names = ["stratum", "psu", "ssu_stratum", "ssu", "tsu_stratum", "tsu", "fsu_stratum"]
    result = dict(
        psu="p",
        ssu="s",
        tsu="t",
        fsu="f",
        strata="h",
        ssu_strata="g",
        tsu_strata="q",
        fsu_strata="r",
        population_psu="N",
    )
    for stage, population in enumerate(("M", "L", "K"), 1):
        selected = columns[: 2 * stage + 1]
        records = []
        for key, group in frame.groupby(selected, sort=False, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            key = tuple(v.item() if hasattr(v, "item") else v for v in key)
            count = (
                int(group[["s", "t", "f"][stage - 1]].nunique())
                if stage + 1 in census
                else int(group[population].iloc[0])
            )
            records.append(
                dict(zip(names[: len(selected)], key))
                | {("population_ssu", "population_tsu", "population_fsu")[stage - 1]: count}
            )
        result[("ssu_frame", "tsu_frame", "fsu_frame")[stage - 1]] = records
    return result


def table_state(table):
    return dict(
        index=table.index.tolist(),
        columns=table.columns.tolist(),
        data=table.astype(object).where(pd.notna(table), None).values.tolist(),
        attrs=table.attrs,
    )


def main():
    frame, design_roles = fixture()
    rows = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    design = oe.survey_fully_stratified_four_stage_design(frame, **design_roles)
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
        models[name] = getattr(oe, "survey_fully_stratified_four_stage_" + name)(
            frame, design, *case["args"], **kwargs
        )
    states = {name: model.model_dump(mode="json") for name, model in models.items()}
    restored = {
        name: type(model).model_validate_json(json.dumps(states[name], allow_nan=False))
        for name, model in models.items()
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
        "cases": cases,
        "options": options,
    }
    assert json.loads(json.dumps(states, allow_nan=False)) == states
    assert json.loads(json.dumps(poststates, allow_nan=False)) == poststates
    assert json.loads(json.dumps(oracle_inputs, allow_nan=False)) == oracle_inputs
    assert len(latexstates) == 8
    if "display" not in globals():
        display_fn = print
    else:
        display_fn = globals()["display"]
    for model in restored.values():
        display_fn(model.to_frame())
    print(
        "SURVEY_FULLY_STRATIFIED_FOUR_STAGE_RECEIPT:"
        + json.dumps(
            dict(
                stages=8,
                rows=256,
                design_df=2,
                sampling_stages=4,
                lower_stage_stratification="SSU/TSU/FSU",
                full_covariance_saved=True,
                restored_equal=True,
                all_four_fpc_terms=True,
                latex_count=8,
                stata_parity_validated=False,
            )
        )
    )
    return locals()


if __name__ == "__main__":
    globals().update(main())
