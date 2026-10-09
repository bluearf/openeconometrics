"""Editable synthetic MARKET-489..496: eight complete conditional-regression artifacts."""

import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
import pandas as pd
import torch
import openecon as oe

display = globals().get("display", print)
torch.set_num_threads(2)
bernoulli = pd.DataFrame(
    {"y": [0, 1, 0, 1, 0, 1], "x": [-1, 0, 1, -1, 0, 1], "z": [0, 0, 0, 1, 1, 1]},
    index=[f"bernoulli-{i}" for i in range(6)],
)
counts = pd.DataFrame(
    {"y": [1, 0, 2, 1], "x": [-1, 0, 1, 2], "z": [0, 0, 1, 1], "e": [1.0, 2.0, 0.5, 3.0]},
    index=[f"count-{i}" for i in range(4)],
)
logit = oe.exact_logit_fit(bernoulli, "y", "x", nuisance=["z"])
poisson = oe.exact_poisson_fit(counts, "y", "x", nuisance=["z"], exposure="e")
results = [
    logit,
    oe.exact_logit_ci(logit),
    oe.exact_logit_test(logit, null=0.2),
    oe.exact_logit_moments(logit, coefficient=0.2),
    poisson,
    oe.exact_poisson_ci(poisson),
    oe.exact_poisson_test(poisson, null=0.2),
    oe.exact_poisson_moments(poisson, coefficient=0.2),
]
visible = ["coefficients", "intervals", "test", "response_moments"] * 2
proof = {
    "frozen": bool(getattr(sys, "frozen", False)),
    "issues": [f"MARKET-{i}" for i in range(489, 497)],
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
    else tempfile.TemporaryDirectory(prefix="conditional-eight-")
)
with folder_context as folder:
    for result, table in zip(results, visible):
        name = result.attrs["procedure"]
        saved = oe.conditional_save(result, Path(folder) / f"{name}.json")
        restored = oe.conditional_load(Path(folder) / f"{name}.json")
        assert oe.conditional_save(restored) == saved
        assert "\\begin{tabular}" in restored.to_latex()
        proof["procedures"].append(name)
        proof["artifact_sha256"][name] = saved["sha256"]
        display(restored[table])
assert len(set(proof["procedures"])) == 8
print("CONDITIONAL_EIGHT_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
