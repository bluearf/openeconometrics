"""Scalar weighted/category Gaussian and partial PLS; save/replay without refit."""

import math

import pandas as pd

import openecon as oe
from openecon.models import ResultBundle

rows = range(60)
data = pd.DataFrame(
    {
        "x": [float(i) / 10 for i in rows],
        "control": [math.sin(i) for i in rows],
        "sector": ["red" if i % 2 else "blue" for i in rows],
        "outcome": [1.3 + 0.7 * i / 10 + 0.2 * math.sin(i) + 0.3 * (i % 2) for i in rows],
        "frequency": [1 + i % 3 for i in rows],
    }
)
results = {}
for name in ("ridge", "lasso", "elasticnet", "pls"):
    tuning = (
        {"component_path": [1, 2]}
        if name == "pls"
        else {"lambda_path": [0.3, 0.1, 0.03], "penalty_factors": {"x": 0.5, "sector": 1.5}}
    )
    result = getattr(oe, name)(
        data=data,
        y="outcome",
        x=["x", "control", "sector"],
        categorical=["sector"],
        weights="frequency",
        weight_type="fweight",
        forced_controls=["control"],
        selection="cv",
        folds=3,
        seed=173,
        **tuning,
    )
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored.extra == result.extra
    assert oe.regularized_predict(restored, data).equals(oe.regularized_predict(result, data))
    assert result.inference["available"] is False
    results[name] = result
    print(name, result.metrics)
    display = globals().get("display")
    if display is not None:
        display(oe.regularized_table(result))
print("REGULARIZED_ALL_FAMILY_STATE_REPLAY_OK")
