"""Editable MARKET-715..722: eight canonical Gaussian spending designs."""

import hashlib
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
import openecon as oe
from openecon.econometrics.core import TableSet

display = globals().get("display", print)
torch.set_num_threads(2)
laws = ("ldof", "ldpocock", "hsd", "power")
fractions = [.2, .4, .7, 1.]


def payload(result):
    """Complete tables, labelled axes, dtypes, attributes, scientific state and LaTeX."""
    return {"title": result.title, "table_order": list(result),
            "tables": {key: frame.to_dict(orient="split") for key, frame in result.items()},
            "table_dtypes": {key: [str(dtype) for dtype in frame.dtypes] for key, frame in result.items()},
            "table_axis_names": {key: {"index": list(frame.index.names),
                                       "columns": list(frame.columns.names)} for key, frame in result.items()},
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
        frame.index.names = saved["table_axis_names"][key]["index"]
        frame.columns.names = saved["table_axis_names"][key]["columns"]
        frame.attrs.update(saved["table_attrs"][key])
    return TableSet(frames, title=saved["title"], **saved["attrs"])


names = [f"{law}_{sides}" for law in laws for sides in (1, 2)]
proof = {"frozen": bool(getattr(sys, "frozen", False)),
         "issues": [f"MARKET-{i}" for i in range(715, 723)],
         "results": names, "procedures": [], "artifact_sha256": {}, "table_counts": {},
         "post_artifact_sha256": {}, "complete_artifacts_equal": True,
         "state_restore_equal": True, "saved_operations_equal": True,
         "saved_inverse_retained": True, "all_latex": True}
destination = globals().get("SEQUENTIAL_SPENDING_RESULT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
context = nullcontext(destination) if destination is not None else tempfile.TemporaryDirectory(prefix="sequential-spending-eight-")

with context as folder:
    for law in laws:
        for sides in (1, 2):
            name = f"{law}_{sides}"
            result = oe.sequential_design(fractions=fractions, spending=law, sides=sides,
                                          alpha=.05, effect=.3, information=100.)
            raw = encoded(result)
            path = Path(folder)/f"{name}.json"
            path.write_bytes(raw)
            saved = json.loads(path.read_bytes())
            restored = restore_payload(saved)
            assert encoded(restored) == raw
            assert "\\begin{tabular}" in restored.to_latex()
            state = saved["attrs"]["state"]
            assert state["schema"] == "openecon.sequential.design.v1"
            assert encoded(oe.restore_sequential_design(state)) == raw
            queried = oe.sequential_power(result, effect=.2, information=150.)
            sized = oe.sequential_information(result, effect=.3, power=.8)
            assert encoded(oe.sequential_power(state, effect=.2, information=150.)) == encoded(queried)
            for operation, post_result in (("power", queried), ("information", sized)):
                post_raw = encoded(post_result)
                post_name = f"{name}-{operation}"
                (Path(folder)/f"{post_name}.json").write_bytes(post_raw)
                assert encoded(restore_payload(json.loads(post_raw))) == post_raw
                assert encoded(oe.restore_sequential_design(post_result.attrs["state"])) == post_raw
                proof["post_artifact_sha256"][post_name] = hashlib.sha256(post_raw).hexdigest()
            assert sized.attrs["state"]["inverse"] is not None
            proof["procedures"].append(result.attrs["procedure"])
            proof["artifact_sha256"][name] = hashlib.sha256(raw).hexdigest()
            proof["table_counts"][name] = {key: len(frame) for key, frame in result.items()}
            display(restored["summary"])
print("SEQUENTIAL_SPENDING_ACCEPTANCE_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
