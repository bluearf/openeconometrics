"""Eight bounded MI extensions on synthetic resident CPU data.

Declared priors, category codes and fixed sensitivity offsets are retained.
Finite MH chains do not certify convergence; passive derivation occurs after MI.
"""

import json
import math

import openecon as oe
import pandas as pd

index = pd.Index([f"row-{i // 2}" for i in range(72)], name="participant")
x = [math.sin(i * 0.63) + i / 80 for i in range(72)]
normal_input = pd.DataFrame({"x": x, "y": [1.4 + 0.5 * x[i] + math.cos(i * 1.47) for i in range(72)]}, index=index)
poisson_input = pd.DataFrame({"x": x, "count": [float((i * 7) % 5) for i in range(72)]}, index=index)
ordinal_input = pd.DataFrame({"x": x, "ordered": [[10., 30., 20.][i % 3] for i in range(72)]}, index=index)
multinomial_input = pd.DataFrame({"x": x, "nominal": [[7., 8., 9.][(i * 5) % 3] for i in range(72)]}, index=index)
logit_input = pd.DataFrame({"x": x, "binary": [float((i * 7) % 11 >= 5) for i in range(72)]}, index=index)
for frame in [normal_input, poisson_input, ordinal_input, multinomial_input, logit_input]:
    frame.iloc[8:20, 1] = float("nan")

options = {"m": 6, "burn": 1, "iterations": 1, "mh_burn": 120, "mh_steps": 180}
poisson = oe.mi_discrete(poisson_input, ["x", "count"], methods={"count": "poisson"}, seed=431, **options)
ordinal = oe.mi_discrete(ordinal_input, ["x", "ordered"], methods={"ordered": "ordinal"}, categories={"ordered": [10, 30, 20]}, seed=432, **options)
multinomial = oe.mi_discrete(multinomial_input, ["x", "nominal"], methods={"nominal": "multinomial"}, categories={"nominal": [7, 8, 9]}, seed=433, **options)
normal_delta = oe.mi_delta(normal_input, ["x", "y"], target="y", kind="normal", delta=-0.4, m=6, seed=434)
logit_delta = oe.mi_delta(logit_input, ["x", "binary"], target="binary", kind="logit", delta=0.5, m=6, seed=435, mh_burn=120, mh_steps=180)
poisson_delta = oe.mi_delta(poisson_input, ["x", "count"], target="count", kind="poisson", delta=0.3, m=6, seed=436, mh_burn=120, mh_steps=180)
passive = oe.mi_passive(normal_delta, {
    "shifted_y": {"operation": "affine", "inputs": ["y"], "coefficients": [1.0], "intercept": 0.25},
    "x_squared": {"operation": "power", "inputs": ["x"], "exponent": 2},
    "interaction": {"operation": "product", "inputs": ["x", "shifted_y"]},
})
fits = [oe.ols(data=normal_delta.dataset(i), y="y", x=["x"], device="cpu") for i in range(1, 7)]
pool = oe.mi_pool(fits, imputation_description="Fixed normal delta -0.4; all 72 physical synthetic rows; fixed complete x predictor")
lincom = oe.mi_lincom(pool, [[1.0, 0.5], [0.0, 1.0]], values=[1.0, 0.5], names=["mean_at_x_half", "x_slope"])
results = {"poisson": poisson, "ordinal": ordinal, "multinomial": multinomial,
           "normal_delta": normal_delta, "logit_delta": logit_delta,
           "poisson_delta": poisson_delta, "passive": passive, "lincom": lincom}
states = {name: value.model_dump(mode="json") for name, value in results.items()}
for name, state in states.items():
    cls = (oe.MIDiscreteResult if name in {"poisson", "ordinal", "multinomial"}
           else oe.MIDeltaResult if name.endswith("_delta")
           else oe.MIPassiveResult if name == "passive" else oe.MILincomResult)
    restored = cls.model_validate_json(json.dumps(state, allow_nan=False))
    assert restored.model_dump(mode="json") == state
    assert restored.to_latex()
    if name != "lincom":
        assert restored.dataset(1).equals(results[name].dataset(1))
        assert restored.dataset(1).index.equals(index)
    rendered = restored.table
    if "display" in globals():
        globals()["display"](rendered)
    else:
        print(name, rendered)
print("MI_EXTENSIONS_EIGHT_OK:" + json.dumps({"stages": len(states), "restored_equal": True, "rows": 72}, sort_keys=True))
