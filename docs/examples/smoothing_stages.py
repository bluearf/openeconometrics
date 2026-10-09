"""Synthetic eight-stage source/frozen/installed QA; no user data."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openecon as oe

display = globals().get("display", print)
rng = np.random.default_rng(240247)
x = np.linspace(0.25, 4, 90)
data = pd.DataFrame(
    {
        "x": x,
        "group": ["A", "B", "C"] * 30,
        "grade": ["low", "middle", "high"] * 30,
        "y": 1 + np.sin(x) + rng.normal(0, 0.12, len(x)),
    }
)
before = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
models = {
    "B-spline": oe.bspline_regress(data=data, y="y", x=["x"], knots={"x": [1.0, 2.0, 3.0]}),
    "Natural cubic": oe.rcs_regress(
        data=data, y="y", x=["x"], knots={"x": [0.25, 1.0, 2.0, 3.0, 4.0]}
    ),
    "Gaussian GAM": oe.gam_gaussian(
        data=data, y="y", x=["x"], selection="gcv", penalty_path=[0.0, 0.1, 1.0, 10.0]
    ),
    "LOESS": oe.loess(data=data, y="y", x=["x"], selection="loo", span_path=[0.4, 0.7, 1.0]),
    "Fractional polynomial": oe.fp_regress(data=data, y="y", x=["x"], powers=[0.0, 0.0]),
    "Closed-test MFP": oe.mfp_regress(data=data, y="y", x=["x"]),
    "MARS": oe.mars(data=data, y="y", x=["x"], max_terms=7, max_candidates=12),
    "Mixed kernel": oe.npreg_mixed(
        data=data,
        y="y",
        x=["x", "group", "grade"],
        variable_types={"x": "c", "group": "u", "grade": "o"},
        categories={"group": ["A", "B", "C"], "grade": ["low", "middle", "high"]},
        bandwidth=[0.5, 0.3, 0.4],
        selection="loo",
        bandwidth_path=[[0.3, 0.2, 0.3], [0.7, 0.5, 0.5]],
    ),
}
saved = {}
query = data.iloc[[10, 30, 50, 70]].copy()
for name, original in models.items():
    restored = oe.ResultBundle.model_validate_json(original.model_dump_json())
    assert original.model_dump() == restored.model_dump()
    interval = restored.spec.estimator not in ("loess", "mars", "npreg_mixed")
    prediction = oe.smoothing_predict(restored, data=query, interval=interval)
    assert prediction.row.tolist() == [0, 1, 2, 3]
    if original.coefficients:
        assert len(original.covariance_matrix) == len(original.coefficients)
    saved[name] = {
        "model": restored.model_dump(mode="json"),
        "prediction": prediction.to_dict(orient="records"),
        "prediction_attrs": prediction.attrs,
        "latex": original.to_latex(),
    }
    display(prediction)
assert before == pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
Path("smoothing_states.json").write_text(json.dumps(saved, allow_nan=False, indent=2) + "\n")
frozen = bool(getattr(sys, "frozen", False))
bundled = not frozen or Path(sys._MEIPASS) in Path(oe.__file__).parents
assert bundled and "scipy" not in sys.modules and "statsmodels" not in sys.modules
print(
    "SMOOTHING_EIGHT_QA:"
    + json.dumps(
        {
            "frozen": frozen,
            "sdk_from_bundle": bundled,
            "methods": len(models),
            "full_result_roundtrip": True,
            "native_only": True,
            "input_unchanged": True,
            "query_rows": 4,
            "full_covariance_and_selection_states_saved": True,
        }
    )
)
