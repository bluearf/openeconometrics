"""Eight direct binary weight domains; synthetic reproducible installed acceptance."""

import json
import math
import random

import pandas as pd
import openecon as oe

rng = random.Random(736743)
rows = []
for i in range(240):
    x, z = rng.gauss(0, 1), rng.gauss(0, 1)
    rows.append(
        {
            "x": x,
            "z": z,
            "y": int(rng.random() < 1 / (1 + math.exp(-(0.2 + 0.5 * x - 0.3 * z)))),
            "w": float(1 + i % 4),
            "g": i % 24,
        }
    )
frame = pd.DataFrame(rows)


def table_state(table):
    return {
        "columns": list(table.columns),
        "index": list(table.index),
        "data": table.astype(object).where(pd.notna(table), None).values.tolist(),
        "attrs": table.attrs,
    }


states, queries, latex = {}, {}, {}
for model in ("logit", "probit"):
    for weight, kind in (
        ("fweight", "nonrobust"),
        ("aweight", "opg"),
        ("iweight", "robust"),
        ("pweight", "cluster"),
    ):
        name = model + "_" + weight
        result = getattr(oe, model)(
            data=frame,
            y="y",
            x=["x", "z"],
            weights="w",
            weight_type=weight,
            covariance=kind,
            cluster="g" if kind == "cluster" else None,
        )
        states[name] = result.model_dump(mode="json")
        restored = oe.ResultBundle.model_validate_json(json.dumps(states[name], allow_nan=False))
        assert restored == result
        queries[name] = {
            "predict": table_state(oe.predict(restored, data=frame.iloc[:7], interval="mean")),
            "margins": table_state(oe.margins(restored, variables=["x", "z"], data=frame)),
            "lincom": oe.lincom(restored, {"x": 1.0, "z": -0.5}),
            "test": oe.test(restored, [{"x": 1.0}, {"z": 1.0}]),
        }
        latex[name] = oe.to_latex(restored)
        assert latex[name]
        if "display" in globals():
            globals()["display"](restored)
oracle_inputs = frame.to_dict(orient="list")
