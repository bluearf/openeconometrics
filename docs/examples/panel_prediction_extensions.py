"""Synthetic local example: saved PLS/CRE/HT and explicit window evaluation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openecon as oe

display = globals().get("display", print)
rng = np.random.default_rng(147148184)
groups, periods = 24, 9
ids = np.repeat(np.arange(groups), periods)
u, z, instrument = rng.normal(size=(3, groups))
x1 = instrument[ids] + rng.normal(size=len(ids))
x2 = 0.6 * u[ids] + rng.normal(size=len(ids))
z2 = 0.8 * instrument + 0.5 * u + rng.normal(size=groups)
data = pd.DataFrame(
    {
        "id": ids,
        "time": np.tile(np.arange(periods), groups),
        "x1": x1,
        "x2": x2,
        "z": z[ids],
        "z2": z2[ids],
    }
)
data["y"] = (
    1 + 0.7 * x1 - 0.4 * x2 + 0.2 * z[ids] + 0.6 * z2[ids] + u[ids] + rng.normal(size=len(ids))
)
before = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()

models = {
    "PLS": oe.pls(data=data, y="y", x=["x1", "x2", "z"], component_path=[1, 2, 3]),
    "CRE": oe.xtreg(
        data=data,
        y="y",
        x=["x1", "x2", "z"],
        panel="id",
        time="time",
        model="cre",
        covariance="robust",
    ),
    "HT": oe.htaylor_moment(
        data=data,
        y="y",
        panel="id",
        time="time",
        varying_exogenous=["x1"],
        varying_endogenous=["x2"],
        invariant_exogenous=["z"],
        invariant_endogenous=["z2"],
        covariance="robust",
    ),
}
restored = {
    name: oe.ResultBundle.model_validate_json(result.model_dump_json())
    for name, result in models.items()
}
for name in models:
    assert models[name].model_dump() == restored[name].model_dump()
display(oe.regularized_table(restored["PLS"]))
display(restored["CRE"])
display(oe.mundlak_test(restored["CRE"]))
display(restored["HT"])
query = data.iloc[:7].copy()
display(
    pd.DataFrame(
        {
            "PLS": oe.regularized_predict(restored["PLS"], query),
            "CRE": oe.cre_predict(restored["CRE"], query),
            "HT": oe.htaylor_predict(restored["HT"], query),
        }
    )
)
panel_spec = oe.ModelSpec(
    estimator="xtreg",
    outcome="y",
    predictors=["x1", "x2"],
    panel="id",
    time="time",
    options={"model": "fe"},
)
reverse = oe.rolling(panel_spec, data=data, window=5, step=2, reverse=True)
display(reverse["coefficients"])
prediction_spec = oe.ModelSpec(
    estimator="pls",
    outcome="y",
    predictors=["x1", "x2", "z"],
    options={"component_path": [1, 2, 3]},
)
evaluation = oe.rolling_predict(
    prediction_spec, data=data, time="time", panel="id", calendar=True, window=5, horizon=1, step=2
)
display(evaluation["predictions"])
json.dumps(reverse.attrs, allow_nan=False)
json.dumps(evaluation.attrs, allow_nan=False)
saved_state = {
    "models": {name: result.model_dump(mode="json") for name, result in restored.items()},
    "reverse": reverse.attrs,
    "predictive": evaluation.attrs,
}
proof_path = Path("panel_prediction_states.json")
proof_path.write_text(json.dumps(saved_state, allow_nan=False), encoding="utf-8")
assert json.loads(proof_path.read_text(encoding="utf-8")) == saved_state
display(pd.DataFrame([{"saved_workflows": str(proof_path), "readback_equal": True}]))
assert before == pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
assert not any(
    name == "scipy" or name.startswith("scipy.") or name == "sklearn" or name.startswith("sklearn.")
    for name in sys.modules
)
print(
    "PANEL_PREDICTION_QA:"
    + json.dumps(
        {
            "frozen": getattr(sys, "frozen", False),
            "sdk_from_bundle": hasattr(sys, "_MEIPASS")
            and str(oe.__file__).startswith(str(sys._MEIPASS)),
            "full_result_roundtrip": True,
            "native_only": True,
            "input_unchanged": True,
            "pls_components": models["PLS"].extra["pls_state"]["components"],
            "mundlak": models["CRE"].tests["mundlak"],
            "ht_inference": models["HT"].inference,
            "reverse_windows": len(reverse.attrs["origins"]),
            "predictive_windows": len(evaluation.attrs["origins"]),
            "displayed_outputs": 8,
            "scope": "synthetic bounded domains; no vendor parity or public release claim",
        }
    )
)
