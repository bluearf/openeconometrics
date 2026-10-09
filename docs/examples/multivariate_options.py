"""Eight bounded multivariate contracts; run in the desktop code panel.

Synthetic 1,203-row fixture; no customer data, external estimator or SciPy.
Set MULTIVARIATE_RESULT_DIRECTORY to retain complete JSON and LaTeX evidence.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch

import openecon as oe
from openecon.econometrics.core import TableSet

torch.set_num_threads(2)
rng = random.Random(20261007)
names = list("abcdef")
loadings = [[.8, .1], [.7, .05], [.75, .2], [.1, .8], [.05, .7], [.2, .75]]
records = []
for i in range(1203):
    f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
    records.append([(a*f1+b*f2+.55*rng.gauss(0, 1))*(j+1)+j+10
                    for j, (a, b) in enumerate(loadings)])
frame = pd.DataFrame(records, columns=names)
frame.index = pd.RangeIndex(1203, name="observation")
weighted = frame.assign(w=[i % 5 for i in range(1203)])
dataset = oe.Dataset.from_frame(weighted)

results = {}
results["frequency_pca"] = oe.pca(dataset, names, weights="w", components=2)
results["summary_pca"] = oe.pca_matrix(frame.cov(), n=1203, matrix="covariance", means=frame.mean(), components=2)
results["summary_factor"] = oe.factor_matrix(frame.corr(), n=1203, means=frame.mean(), sds=frame.std(), factors=2, rotate="varimax")
results["minres"] = oe.factor(oe.Dataset.from_frame(frame), names, method="minres", factors=2)
results["anderson_rubin"] = oe.factor(frame, names, factors=2, rotate="varimax", scores="anderson_rubin")
target = results["anderson_rubin"]["rotated_loadings"].to_numpy()
results["target"] = oe.factor(frame, names, factors=2, rotate="target", target=target)
results["geomin"] = oe.factor(frame, names, factors=2, rotate="geomin", geomin_oblique=True)
counts = [[35, 12, 3], [8, 31, 10], [3, 8, 26], [12, 5, 4]]
cells = pd.DataFrame([(f"r{i}", f"c{j}", value) for i, row in enumerate(counts)
                     for j, value in enumerate(row)], columns=["r", "c", "count"])
active_ca = oe.ca(cells, "r", "c", weights="count")
passive_rows = oe.ca_project(active_ca, {"c2": [3, 10], "c0": [35, 8], "c1": [12, 31]})
passive_cols = oe.ca_project(active_ca, {"r3": [12], "r1": [8], "r0": [35], "r2": [3]}, axis="column")
results["supplementary"] = TableSet({"rows": passive_rows, "columns": passive_cols}, procedure="supplementary", projection_state=active_ca.attrs["projection_state"])


def safe_table(table):
    return table.astype(object).where(table.notna(), None).to_dict(orient="split")


encoded = {}
scores = {}
for name, result in results.items():
    if name != "supplementary":
        scorer = oe.pca_scores if result.attrs["procedure"] == "pca" else oe.factor_scores
        score = scorer(result, frame)
        assert len(score) == 1203 and score.notna().all().all()
        scores[name] = safe_table(score)
    payload = {"attrs": result.attrs, "tables": {key: safe_table(value) for key, value in result.items()},
               "scores": scores.get(name), "latex": result.to_latex()}
    encoded[name] = json.dumps(payload, sort_keys=True, allow_nan=False)
    restored = TableSet({key: pd.DataFrame(**value) for key, value in json.loads(encoded[name])["tables"].items()}, **result.attrs)
    if name != "supplementary":
        again = scorer(restored, frame)
        assert float(torch.tensor((again.to_numpy()-score.to_numpy()).tolist()).abs().max()) < 1e-10

assert results["frequency_pca"].attrs["physical_rows"] == 962
assert results["frequency_pca"].attrs["n"] == 2403
score = torch.as_tensor(oe.factor_scores(results["anderson_rubin"], frame).to_numpy(), dtype=torch.float64)
assert torch.allclose(score.T @ score / 1202, torch.eye(2, dtype=torch.float64), atol=1e-10, rtol=0)
unrotated = torch.as_tensor(results["geomin"]["loadings"].to_numpy())
pattern = torch.as_tensor(results["geomin"]["rotated_loadings"].to_numpy())
phi = torch.as_tensor(results["geomin"]["factor_correlations"].to_numpy())
assert torch.allclose(pattern @ phi @ pattern.T, unrotated @ unrotated.T, atol=1e-10, rtol=0)
assert float(torch.tensor((passive_rows[["dim1", "dim2"]].to_numpy()-active_ca["rows"][["dim1", "dim2"]].iloc[:2].to_numpy()).tolist()).abs().max()) < 1e-10

if "MULTIVARIATE_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["MULTIVARIATE_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixture.json").write_text(frame.to_json(orient="split", double_precision=15))
    (directory/"active_ca.json").write_text(json.dumps({"attrs": active_ca.attrs, "tables": {key: safe_table(value) for key, value in active_ca.items()}}, allow_nan=False))

if "display" in globals():
    for name, result in results.items():
        primary = "rows" if name == "supplementary" else ("rotated_loadings" if "rotated_loadings" in result else "loadings")
        globals()["display"](result[primary])

print("MULTIVARIATE_OPTIONS_OK "+json.dumps({"cases": list(results), "physical_fixture_rows": 1203,
    "frozen": bool(getattr(sys, "frozen", False)), "weight_total": 2403, "complete_score_rows": {name: len(value["data"]) for name, value in scores.items()},
    "saved_table_counts": {name: {key: len(table) for key, table in result.items()} for name, result in results.items()},
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()}}))
