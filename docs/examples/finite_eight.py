"""Editable synthetic MARKET-385..392 acceptance: full Firth/exact/exhaustive LTS state."""

import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import openecon as oe

display = globals().get("display", print)
torch.set_num_threads(2)
bernoulli = pd.DataFrame(
    {"y": [0] * 10 + [1] * 9 + [0], "x": [0] * 10 + [1] * 10},
    index=[f"bernoulli-{i}" for i in range(20)],
)
new = pd.DataFrame({"x": [0.0, 0.5, 1.0]}, index=["new-a", "new-b", "new-c"])
fit = oe.firth_logit(bernoulli, "y", ["x"])
x = np.arange(12, dtype=float)
y = 1 + 2 * x + 0.05 * (-1.0) ** x + 0.009 * np.sin(1.7 * x)
y[9:] += 30
contaminated = pd.DataFrame({"x": x, "y": y}, index=[f"lts-{i}" for i in range(12)])
trimmed = oe.lts(contaminated, "y", ["x"], h=8)
results = [
    fit,
    oe.firth_predict(fit, new),
    oe.firth_profile(fit, ["x"], max_work=300_000_000),
    oe.firth_test(fit, [[0, 1]]),
    oe.exact_logistic([[[3, 4], [1, 5]], [[5, 2], [3, 6]]]),
    oe.exact_poisson_rate(84, 36),
    trimmed,
    oe.lts_predict(trimmed, new),
]
visible = [
    "coefficients",
    "predictions",
    "intervals",
    "test",
    "inference",
    "inference",
    "summary",
    "predictions",
]
proof = {
    "frozen": bool(getattr(sys, "frozen", False)),
    "issues": [f"MARKET-{i}" for i in range(385, 393)],
    "procedures": [],
    "artifact_sha256": {},
    "all_latex": True,
    "complete_artifacts_equal": True,
}
destination = globals().get("ARTIFACT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
folder_context = (
    nullcontext(destination)
    if destination is not None
    else tempfile.TemporaryDirectory(prefix="finite-eight-")
)
with folder_context as folder:
    for result, table in zip(results, visible):
        name = result.attrs["procedure"]
        saved = oe.finite_save(result, Path(folder) / f"{name}.json")
        restored = oe.finite_load(Path(folder) / f"{name}.json")
        assert oe.finite_save(restored) == saved
        assert "\\begin{tabular}" in restored.to_latex()
        proof["procedures"].append(name)
        proof["artifact_sha256"][name] = saved["sha256"]
        display(restored[table])
assert len(set(proof["procedures"])) == 8
print("FINITE_EIGHT_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
