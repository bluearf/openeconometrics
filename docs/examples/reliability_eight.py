"""Editable synthetic acceptance of MARKET-264..271. No external data/dependencies."""

import json
import math
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
rng = np.random.default_rng(128)
latent = rng.normal(size=(40, 2))
latent[:, 1] = 0.5 * latent[:, 0] + math.sqrt(0.75) * latent[:, 1]
ordinal = pd.DataFrame(
    {
        "a": np.digitize(latent[:, 0], [-0.6, 0.5]),
        "b": np.digitize(latent[:, 1], [-0.6, 0.5]),
        "x": latent[:, 0],
    }
)
items = pd.DataFrame(
    rng.normal(size=(24, 1)) * np.array([[0.7, 0.8, 0.9, 0.75]]) + rng.normal(size=(24, 4)) * 0.8,
    columns=list("abcd"),
)
ratings = pd.DataFrame(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 2, 1],
        [2, 2, 2],
        [2, 1, 2],
        [2, 0, 1],
        [0, 2, 0],
        [1, 0, 2],
    ],
    columns=list("abc"),
    index=[f"unit-{i}" for i in range(10)],
)
sf = pd.DataFrame(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
    columns=list("abcd"),
)
results = [
    oe.polychoric(
        ordinal,
        ["a", "b"],
        categories=[[0, 1, 2], [0, 1, 2]],
        inference="jackknife",
        max_work=20_000_000_000,
    ),
    oe.polyserial(ordinal, "x", "b", categories=[0, 1, 2], inference="jackknife"),
    oe.omega_total(items, list("abcd"), inference="jackknife"),
    oe.icc(sf, list("abcd"), model="ICC2", inference="jackknife"),
    oe.cohen_kappa(
        ratings,
        ["a", "b"],
        categories=[0, 1, 2],
        agreement_weights="quadratic",
        inference="jackknife",
    ),
    oe.fleiss_kappa(ratings, list("abc"), categories=[0, 1, 2], inference="jackknife"),
    oe.krippendorff_alpha(
        ratings, list("abc"), categories=[0, 1, 2], metric="ordinal", inference="jackknife"
    ),
    oe.gwet_ac(
        ratings,
        list("abc"),
        categories=[0, 1, 2],
        agreement_weights="quadratic",
        inference="jackknife",
    ),
]
proof = dict(
    frozen=bool(getattr(sys, "frozen", False)),
    issues=[f"MARKET-{i}" for i in range(264, 272)],
    complete_artifacts_equal=True,
    all_latex=True,
    sample_sizes=[],
    fits=[],
    estimates=[],
    procedures=[],
    artifact_sha256={},
)
destination = globals().get("ARTIFACT_DIRECTORY")
if destination is not None:
    Path(destination).mkdir(parents=True, exist_ok=True)
artifact_folder = (
    nullcontext(destination)
    if destination is not None
    else tempfile.TemporaryDirectory(prefix="openecon-reliability-artifacts-")
)
with artifact_folder as folder:
    for result in results:
        name = result.attrs["procedure"]
        artifact = oe.reliability_save(result, Path(folder) / f"{name}.json")
        restored = oe.reliability_load(Path(folder) / f"{name}.json")
        assert artifact == oe.reliability_save(restored)
        assert "\\begin{tabular}" in restored.to_latex()
        assert restored.attrs["failed_fits"] == 0
        assert restored.attrs["completed_fits"] == restored.attrs["n"] + 1
        proof["sample_sizes"].append(restored.attrs["n"])
        proof["fits"].append(restored.attrs["completed_fits"])
        proof["estimates"].append(restored.attrs["estimate"])
        proof["procedures"].append(name)
        proof["artifact_sha256"][name] = artifact["sha256"]
        display(restored["estimate"])
assert len(set(proof["procedures"])) == 8
print("RELIABILITY_EIGHT_OK " + json.dumps(proof, sort_keys=True, allow_nan=False))
