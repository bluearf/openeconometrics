"""Editable synthetic MARKET-648..655: eight complete repeated GLS fits."""

import hashlib
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet

display = globals().get("display", print)
torch.set_num_threads(2)
structures = ("cs", "ar1", "diagonal", "unstructured")
methods = ("ML", "REML")
future = pd.DataFrame({"x": [-.5, .0, .8], "z": [.2, -.4, .1]},
                      index=["profile-a", "profile-b", "profile-c"])
targets = pd.DataFrame([[0., 1., 0.], [0., 0., 1.]],
                       columns=["Intercept", "x", "z"], index=["x-effect", "z-effect"])
null = [.1, -.1]


def synthetic_data(structure):
    """Original complete observations with missing occasion patterns, not missing values."""
    rng = np.random.default_rng(87421 + structures.index(structure))
    times = np.array([0, 1, 3, 4])
    if structure == "cs":
        shape = 1.24*np.eye(4) - .24*np.ones((4, 4))
    elif structure == "ar1":
        shape = np.power(-.62, np.abs(times[:, None] - times))
    elif structure == "diagonal":
        shape = np.diag(np.array([.45, .85, 1.3, 1.8])**2)
    else:
        factor = np.array([[.8, 0, 0, 0], [-.45, 1.25, 0, 0],
                           [.35, .25, .5, 0], [.1, -.35, .3, .9]])
        shape = factor@factor.T
    records = []
    for subject in range(32):
        keep = np.ones(4, dtype=bool) if subject < 6 else rng.random(4) > .24
        if keep.sum() < 2:
            keep[rng.choice(4, size=2, replace=False)] = True
        x, z = rng.normal(size=(2, 4))
        residual = np.linalg.cholesky(shape)@rng.normal(size=4)
        y = 1.7 + .8*x - .45*z + residual
        for occasion in np.flatnonzero(keep):
            records.append({"subject": f"s{subject:02d}", "occasion": int(times[occasion]),
                            "x": float(x[occasion]), "z": float(z[occasion]),
                            "y": float(y[occasion])})
    return pd.DataFrame(records, index=[f"source-{structure}-{i}" for i in range(len(records))])


def payload(result):
    """Retain all tables, labels, dtypes, attributes, state and exact LaTeX."""
    return {"title": result.title, "table_order": list(result),
            "tables": {key: frame.to_dict(orient="split") for key, frame in result.items()},
            "table_dtypes": {key: [str(dtype) for dtype in frame.dtypes] for key, frame in result.items()},
            "table_attrs": {key: frame.attrs for key, frame in result.items()},
            "attrs": result.attrs, "latex": result.to_latex()}


def encoded(result):
    return json.dumps(payload(result), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def restore_payload(saved):
    frames = {key: oe.DataFrame(**saved["tables"][key]) for key in saved["table_order"]}
    for key, frame in frames.items():
        for column, dtype in zip(frame.columns, saved["table_dtypes"][key]):
            frame[column] = frame[column].astype(dtype)
            if dtype == "object":
                frame[column] = frame[column].where(frame[column].notna(), None)
        frame.attrs.update(saved["table_attrs"][key])
    return TableSet(frames, title=saved["title"], **saved["attrs"])


names = [f"{structure}_{method.lower()}" for structure in structures for method in methods]
proof = {"frozen": bool(getattr(sys, "frozen", False)),
         "issues": [f"MARKET-{i}" for i in range(648, 656)],
         "results": names, "procedures": [], "artifact_sha256": {}, "table_counts": {},
         "post_artifact_sha256": {}, "complete_artifacts_equal": True,
         "state_restore_equal": True, "saved_operations_equal": True, "all_latex": True}
destination = globals().get("REPEATED_GLS_RESULT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
context = nullcontext(destination) if destination is not None else tempfile.TemporaryDirectory(prefix="repeated-gls-eight-")

with context as folder:
    for structure in structures:
        data = synthetic_data(structure)
        for method in methods:
            name = f"{structure}_{method.lower()}"
            result = oe.repeated_gls(data=data, y="y", x=["x", "z"], subject="subject",
                                     occasion="occasion", structure=structure, method=method)
            raw = encoded(result)
            path = Path(folder)/f"{name}.json"
            path.write_bytes(raw)
            saved = json.loads(path.read_bytes())
            restored = restore_payload(saved)
            assert encoded(restored) == raw
            assert "\\begin{tabular}" in restored.to_latex()
            state = saved["attrs"]["state"]
            assert state["schema"] == "openecon.repeated.gls.v1"
            assert encoded(oe.restore_repeated_gls(state)) == raw
            prediction = oe.repeated_gls_predict(result, data=future)
            contrast = oe.repeated_gls_contrast(result, contrast=targets, null=null)
            assert encoded(oe.repeated_gls_predict(state, data=future)) == encoded(prediction)
            assert encoded(oe.repeated_gls_contrast(state, contrast=targets, null=null)) == encoded(contrast)
            for operation, post_result in (("mean", prediction), ("contrast", contrast)):
                post_raw = encoded(post_result)
                post_name = f"{name}-{operation}"
                (Path(folder)/f"{post_name}.json").write_bytes(post_raw)
                assert encoded(restore_payload(json.loads(post_raw))) == post_raw
                proof["post_artifact_sha256"][post_name] = hashlib.sha256(post_raw).hexdigest()
            proof["procedures"].append(result.attrs["procedure"])
            proof["artifact_sha256"][name] = hashlib.sha256(raw).hexdigest()
            proof["table_counts"][name] = {key: len(frame) for key, frame in result.items()}
            display(restored["coefficients"])
print("REPEATED_GLS_ACCEPTANCE_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
