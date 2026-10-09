"""Editable synthetic MARKET-629..636: eight complete dependent-meta artifacts."""

import hashlib
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet

display = globals().get("display", print)
torch.set_num_threads(2)
effect_ids = [f"effect-{i}" for i in range(14)]
study_ids = [f"study-{i//2}" for i in range(14)]
moderator = [-1.1, -.9, -.7, -.4, -.3, .0, .15, .4, .35, .65, .75, .9, 1., 1.25]
study_shift = [-.85, -.5, .2, .8, -.1, 1.1, -.6]
yi = [.4+.3*x+study_shift[i//2]+(.13 if i % 2 else -.08) for i, x in enumerate(moderator)]
data = pd.DataFrame({"effect": effect_ids, "study": study_ids, "yi": yi, "x": moderator},
                    index=[f"source-{i}" for i in range(14)])
sampling = torch.zeros((14, 14), dtype=torch.float64)
for group in range(7):
    start = group*2
    sampling[start, start] = .16
    sampling[start+1, start+1] = .25
    sampling[start, start+1] = sampling[start+1, start] = .056
known_covariance = pd.DataFrame(sampling.tolist(), index=effect_ids, columns=effect_ids)
settings = dict(data=data, yi="yi", effect="effect", study="study", covariance=known_covariance,
                moderators=["x"], method="REML")
common = oe.meta_dependent(**settings, model="common")
effect = oe.meta_dependent(**settings, model="effect")
study = oe.meta_dependent(**settings, model="study")
robust = oe.meta_dependent_robust(study, correction="CR1")
contrasts = pd.DataFrame([[1., 0.], [0., 1.]], columns=["Intercept", "x"],
                         index=["intercept", "slope"])
future = pd.DataFrame({"x": [-.5, .0, .8], "future_study": ["future-A", "future-A", "future-B"]},
                      index=["future-row-0", "future-row-1", "future-row-2"])
results = [common, effect, study, robust,
           oe.meta_dependent_contrast(robust, contrasts=contrasts, null=[.1, -.1]),
           oe.meta_dependent_predict(robust, data=future),
           oe.meta_dependent_predict_effect(robust, data=future, study="future_study"),
           oe.meta_dependent_diagnostics(study)]
names = ["common", "effect", "study", "robust", "contrast", "mean", "latent", "diagnostics"]
visible = ["coefficients"]*4+["contrasts", "prediction", "prediction", "summary"]
proof = {"frozen": bool(getattr(sys, "frozen", False)),
         "issues": [f"MARKET-{i}" for i in range(629, 637)],
         "results": names, "procedures": [], "artifact_sha256": {}, "table_counts": {},
         "complete_artifacts_equal": True, "all_latex": True}
destination = globals().get("DEPENDENT_META_RESULT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
context = nullcontext(destination) if destination is not None else tempfile.TemporaryDirectory(prefix="dependent-meta-eight-")


def payload(result):
    return {"title": result.title, "table_order": list(result),
            "tables": {key: frame.to_dict(orient="split") for key, frame in result.items()},
            "table_dtypes": {key: [str(dtype) for dtype in frame.dtypes] for key, frame in result.items()},
            "table_attrs": {key: frame.attrs for key, frame in result.items()},
            "attrs": result.attrs, "latex": result.to_latex()}


def encoded(result):
    return json.dumps(payload(result), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


with context as folder:
    for name, result, summary in zip(names, results, visible):
        raw = encoded(result)
        path = Path(folder)/f"{name}.json"
        path.write_bytes(raw)
        saved = json.loads(path.read_bytes())
        frames = {key: oe.DataFrame(**frame) for key, frame in saved["tables"].items()}
        # TableSet order is scientifically meaningful; sort_keys is storage-only.
        ordered_frames = {key: frames[key] for key in saved["table_order"]}
        for key in ordered_frames:
            for column, dtype in zip(ordered_frames[key].columns, saved["table_dtypes"][key]):
                ordered_frames[key][column] = ordered_frames[key][column].astype(dtype)
                if dtype == "object":
                    ordered_frames[key][column] = ordered_frames[key][column].where(ordered_frames[key][column].notna(), None)
            ordered_frames[key].attrs.update(saved["table_attrs"][key])
        restored = TableSet(ordered_frames, title=saved["title"], **saved["attrs"])
        assert encoded(restored) == raw
        assert "\\begin{tabular}" in restored.to_latex()
        proof["procedures"].append(result.attrs["procedure"])
        proof["artifact_sha256"][name] = hashlib.sha256(raw).hexdigest()
        proof["table_counts"][name] = {key: len(frame) for key, frame in result.items()}
        display(restored[summary])
    restored_study = json.loads((Path(folder)/"study.json").read_bytes())["attrs"]["prediction_state"]
    restored_robust = json.loads((Path(folder)/"robust.json").read_bytes())["attrs"]["prediction_state"]
    assert encoded(oe.meta_dependent_robust(restored_study, correction="CR1")) == encoded(robust)
    assert encoded(oe.meta_dependent_predict(restored_robust, data=future)) == encoded(results[5])
    assert encoded(oe.meta_dependent_predict_effect(restored_robust, data=future, study="future_study")) == encoded(results[6])
print("DEPENDENT_META_ACCEPTANCE_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
