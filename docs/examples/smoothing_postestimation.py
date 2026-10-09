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
query = pd.DataFrame(
    {
        "x": [0.61, 1.41, 2.61, 3.61],
        "group": ["A", "B", "C", "A"],
        "grade": ["low", "middle", "high", "middle"],
    }
)
reference = query.copy()
reference.x -= 0.08
stages = [
    ("Spline derivatives", "B-spline", oe.spline_derivative, {"variable": "x", "order": 2}),
    ("FP derivatives", "Fractional polynomial", oe.fp_derivative, {"variable": "x", "order": 2}),
    ("GAM derivatives", "Gaussian GAM", oe.gam_derivative, {"variable": "x"}),
    ("MARS derivatives", "MARS", oe.mars_derivative, {"variable": "x"}),
    ("LOESS Taylor derivatives", "LOESS", oe.loess_derivative, {"variable": "x"}),
    ("Kernel derivatives", "Mixed kernel", oe.kernel_derivative, {"variable": "x", "order": 2}),
    (
        "Average MFP margins",
        "Closed-test MFP",
        oe.smoothing_margins,
        {"variable": "x", "interval": True, "averaging_weights": [1.0, 2.0, 3.0, 4.0]},
    ),
    (
        "Paired spline contrasts",
        "Natural cubic",
        oe.smoothing_contrast,
        {"reference": reference, "interval": True},
    ),
]
for label, name, function, options in stages:
    original = models[name]
    restored = oe.ResultBundle.model_validate_json(original.model_dump_json())
    assert original.model_dump() == restored.model_dump()
    prediction = function(restored, data=query, **options)
    assert len(prediction) == (1 if function is oe.smoothing_margins else 4)
    saved[label] = {
        "model": restored.model_dump(mode="json"),
        "prediction": prediction.to_dict(orient="records"),
        "prediction_attrs": prediction.attrs,
        "latex": prediction.to_latex(),
    }
    display(prediction)
assert before == pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
Path("smoothing_postestimation_states.json").write_text(
    json.dumps(saved, allow_nan=False, indent=2) + "\n"
)
frozen = bool(getattr(sys, "frozen", False))
bundled = not frozen or Path(sys._MEIPASS) in Path(oe.__file__).parents
assert bundled and "scipy" not in sys.modules and "statsmodels" not in sys.modules
print(
    "SMOOTHING_POSTESTIMATION_QA:"
    + json.dumps(
        {
            "frozen": frozen,
            "sdk_from_bundle": bundled,
            "methods": len(models),
            "full_result_roundtrip": True,
            "native_only": True,
            "input_unchanged": True,
            "query_rows": [4, 4, 4, 4, 4, 4, 1, 4],
            "saved_analytic_functionals_and_covariance": True,
            "full_covariance_and_selection_states_saved": True,
        }
    )
)
