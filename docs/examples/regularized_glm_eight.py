"""Eight bounded penalized GLM acceptance stages; synthetic, no external runtime solver."""

import json
import math
import sys
from pathlib import Path
from random import Random

import openecon as oe


display = globals().get("display", print)
rng = Random(20261007)
rows = []
for i in range(120):
    x, z, group = rng.uniform(-1.5, 1.5), rng.uniform(-1, 1), "ABC"[i % 3]
    group_effect = {"A": -0.2, "B": 0.1, "C": 0.3}[group]
    probability = 1 / (1 + math.exp(-(-0.1 + 0.6*x - 0.25*z + group_effect)))
    rate = math.exp(0.35 + 0.3*x - 0.15*z + group_effect)
    product, count = 1.0, -1
    while product > math.exp(-rate):
        product *= rng.random()
        count += 1
    rows.append({"x": x, "z": z, "group": group, "binary": int(rng.random() < probability),
                 "count": count, "frequency": 1+i % 3, "importance": 0.5+(i % 7)/5})
data = oe.DataFrame(rows)
before = data.to_json()
common = dict(data=data, x=["x", "z", "group"], categorical=["group"],
              lambda_path=[0.3, 0.1, 0.02], selection="fixed", penalty=0.1)
specifications = [
    ("Binomial path", oe.elasticnet_logit, {"y": "binary"}),
    ("Poisson path", oe.elasticnet_poisson, {"y": "count"}),
    ("Binomial training-only CV", oe.elasticnet_logit,
     {"y": "binary", "selection": "cv", "penalty": None, "folds": 3}),
    ("Poisson training-only fraction CV", oe.elasticnet_poisson,
     {"y": "count", "selection": "cv", "penalty": None, "lambda_path": None,
      "n_lambdas": 3, "lambda_ratio": 0.1, "folds": 3}),
    ("Frequency-weighted prediction", oe.elasticnet_logit,
     {"y": "binary", "weights": "frequency", "weight_type": "fweight"}),
    ("Sampling-weighted empirical loss", oe.elasticnet_poisson,
     {"y": "count", "weights": "importance", "weight_type": "pweight"}),
    ("Saved category maps and forced controls", oe.elasticnet_logit,
     {"y": "binary", "forced_controls": ["group"]}),
    ("Literal feature penalty factors", oe.elasticnet_poisson,
     {"y": "count", "penalty_factors": {"x": 0.0, "z": 2.0, "group": 0.75}}),
]
complete = []
for label, function, options in specifications:
    result = function(**{**common, **options})
    restored = oe.ResultBundle.model_validate_json(result.model_dump_json())
    prediction = oe.regularized_glm_predict(restored, data.iloc[:5])
    link = oe.regularized_glm_predict(restored, data.iloc[:5], kind="link")
    path = oe.regularized_glm_path(restored)
    display(path)
    state = restored.extra["regularized_glm_state"]
    assert not restored.coefficients and restored.inference["available"] is False
    assert all(point["converged"] for point in state["paths"])
    complete.append({"label": label, "model": restored.model_dump(mode="json"),
                     "path": path.to_dict(orient="split"), "path_attrs": path.attrs,
                     "prediction": prediction.to_dict(orient="split"), "prediction_attrs": prediction.attrs,
                     "link": link.to_dict(orient="split"), "link_attrs": link.attrs})
    print(label, "rows=", len(path), "selected_penalty=", state["lambda_path"][state["selected_index"]])
assert data.to_json() == before
Path("regularized_glm_eight_states.json").write_text(json.dumps(complete, indent=2, allow_nan=False))
print("Eight penalized GLM stages complete; input unchanged; no coefficient/survey inference.")
frozen = bool(getattr(sys, "frozen", False))
bundled = not frozen or Path(sys._MEIPASS) in Path(oe.__file__).parents
assert bundled and "scipy" not in sys.modules and "statsmodels" not in sys.modules
print("REGULARIZED_GLM_EIGHT_QA:" + json.dumps({
    "frozen": frozen, "sdk_from_bundle": bundled, "methods": len(complete),
    "full_result_roundtrip": True, "native_only": True, "input_unchanged": True,
    "full_path_weight_category_cv_states_saved": True, "table_rows": [len(v["path"]["data"]) for v in complete],
}))
