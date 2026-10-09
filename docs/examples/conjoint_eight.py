"""Editable synthetic MARKET-288..295 acceptance; complete local scored study."""

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
attributes = {"brand": ["a", "b"], "price": [10, 20, 30, 40],
              "speed": ["low", "high"], "quality": ["basic", "premium"]}
full = oe.conjoint_plan(attributes)
profiles = full["profiles"]
training = profiles.iloc[[i for i in range(len(profiles)) if i % 4 != 0]].copy()
held = profiles.iloc[::4].copy()
training_ids = set(training.profile_id)
rng = np.random.default_rng(288)
responses, holdout = [], []
for subject in ("person-z", "person-a", "person-m"):
    for card, brand, price, speed, quality in profiles.itertuples(index=False, name=None):
        score = 20+1.1*(brand == "a")-.02*(price-25)**2+.5*(speed == "high")+.2*(quality == "premium")
        row = [subject, card, score+rng.normal(scale=.2)]
        (responses if card in training_ids else holdout).append(row)
responses = pd.DataFrame(responses, columns=["subject", "profile_id", "score"])
holdout = pd.DataFrame(holdout, columns=responses.columns)
fraction = oe.conjoint_orthogonal({a: [-1,1] for a in "ABCD"}, list("ABC"), {"D": list("ABC")})
audit = oe.conjoint_diagnostics(fraction)
fit = oe.conjoint_fit(training, responses, attributes, factors={"price": "ideal"})
results = [full, fraction, audit, fit, oe.conjoint_predict(fit, held),
           oe.conjoint_holdout(fit, held, holdout), oe.conjoint_importance(fit),
           oe.conjoint_simulate(fit, held, method="logit", temperature=1.0)]
visible = ["profiles", "profiles", "summary", "coefficients", "predictions", "validation", "group", "group"]
proof = {"frozen": bool(getattr(sys, "frozen", False)), "issues": [f"MARKET-{i}" for i in range(288,296)],
         "procedures": [], "artifact_sha256": {}, "all_latex": True, "complete_artifacts_equal": True}
destination = globals().get("ARTIFACT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
folder_context = nullcontext(destination) if destination is not None else tempfile.TemporaryDirectory(prefix="conjoint-eight-")
with folder_context as folder:
    for result, table in zip(results, visible):
        name = result.attrs["procedure"]
        saved = oe.conjoint_save(result, Path(folder)/f"{name}.json")
        restored = oe.conjoint_load(Path(folder)/f"{name}.json")
        assert oe.conjoint_save(restored) == saved
        assert "\\begin{tabular}" in restored.to_latex()
        proof["procedures"].append(name)
        proof["artifact_sha256"][name] = saved["sha256"]
        display(restored[table])
assert len(set(proof["procedures"])) == 8
print("CONJOINT_EIGHT_OK "+json.dumps(proof, sort_keys=True, allow_nan=False))
