"""Eight new bounded multivariate contracts; run in the desktop code panel.

Synthetic 1,203-row, six-variable correlated two-factor fixture. The separate
bootstrap fixture has one regular iid factor. No customer data or external
estimator is used. Set MULTIVARIATE_EXTENSION_RESULT_DIRECTORY to retain full
JSON, saved score state, every bootstrap vector and complete LaTeX evidence.
Image extraction here is explicitly PCA of the linear-prediction image
covariance (SAS-style image-covariance analysis), not Kaiser image factoring.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import pandas as pd
import torch

import openecon as oe
from openecon.econometrics.core import TableSet

torch.set_num_threads(2)
rng = random.Random(20261007319)
names = list("abcdef")
declared_loadings = [[.8, .1], [.7, .05], [.75, .2], [.1, .8], [.05, .7], [.2, .75]]
records = []
for i in range(1203):
    f1 = rng.gauss(0, 1)
    f2 = .35*f1 + math.sqrt(1-.35**2)*rng.gauss(0, 1)
    records.append([(a*f1+b*f2+.55*rng.gauss(0, 1))*(j+1)+j+10
                    for j, (a, b) in enumerate(declared_loadings)])
frame = pd.DataFrame(records, columns=names)
frame.index = pd.RangeIndex(1203, name="observation")
weighted = frame.assign(w=[i % 5 for i in range(1203)])

# Targets and binary masks are declared before fitting. Each factor has a
# specified nonzero anchor; free cells stay explicitly outside the objective.
target = pd.DataFrame([[.82, .1], [.72, .06], [.75, .18],
                       [.1, .8], [.06, .7], [.2, .75]],
                      index=names, columns=["Factor1", "Factor2"])
mask = pd.DataFrame([[1, 0], [1, 1], [0, 1], [1, 0], [1, 1], [0, 1]],
                    index=names, columns=target.columns)
bootstrap_records = []
for i in range(1203):
    latent = rng.gauss(0, 1)
    bootstrap_records.append([(.65+.02*j)*latent+.6*rng.gauss(0, 1) for j in range(6)])
bootstrap_frame = pd.DataFrame(bootstrap_records, columns=names)

results = {}
results["frequency_factor"] = oe.factor(oe.Dataset.from_frame(weighted), names,
    weights="w", factors=2, rotate="varimax")
results["alpha"] = oe.factor(frame, names, method="alpha", factors=2,
    max_iterations=2000, tolerance=1e-7)
results["image_covariance"] = oe.factor(frame, names, method="image_covariance", factors=2)
results["cf_orthogonal"] = oe.factor(frame, names, factors=2, rotate="cf", cf_kappa=.37)
results["cf_oblique"] = oe.factor(frame, names, factors=2, rotate="cf", cf_kappa=.37, cf_oblique=True)
results["partial_target_orthogonal"] = oe.factor(frame, names, factors=2,
    rotate="partial_target", target=target, target_mask=mask)
results["partial_target_oblique"] = oe.factor(frame, names, factors=2,
    rotate="partial_target", target=target, target_mask=mask, target_oblique=True)
results["loading_bootstrap"] = oe.factor_bootstrap(bootstrap_frame, names,
    replications=199, confidence=.95, seed=7319, anchor="a")


def safe_table(table):
    return table.astype(object).where(table.notna(), None).to_dict(orient="split")


def restore(payload):
    return TableSet({key: pd.DataFrame(**value) for key, value in payload["tables"].items()},
                    **payload["attrs"])


encoded = {}
scores = {}
for name, result in results.items():
    if name != "loading_bootstrap":
        score = oe.factor_scores(result, frame)
        assert len(score) == 1203 and score.notna().all().all()
        scores[name] = safe_table(score)
    payload = {"attrs": result.attrs,
               "tables": {key: safe_table(value) for key, value in result.items()},
               "scores": scores.get(name), "latex": result.to_latex()}
    encoded[name] = json.dumps(payload, sort_keys=True, allow_nan=False)
    restored = restore(json.loads(encoded[name]))
    for key, original in result.items():
        saved = restored[key].to_numpy(dtype="float64")
        expected = original.to_numpy(dtype="float64")
        assert torch.allclose(torch.as_tensor(saved), torch.as_tensor(expected), atol=1e-12,
                              rtol=1e-12, equal_nan=True)
    if name != "loading_bootstrap":
        again = oe.factor_scores(restored, frame)
        assert torch.allclose(torch.as_tensor(again.to_numpy()), torch.as_tensor(score.to_numpy()),
                              atol=1e-10, rtol=0)

assert results["frequency_factor"].attrs["physical_rows"] == 962
assert results["frequency_factor"].attrs["n"] == 2403
assert results["frequency_factor"].attrs["covariance_divisor"] == 2402
assert results["alpha"].attrs["converged"]
assert results["image_covariance"].attrs["extraction_objective"] == "PCA_of_linear_prediction_image_covariance"
for name in ("cf_orthogonal", "cf_oblique", "partial_target_orthogonal", "partial_target_oblique"):
    result = results[name]
    unrotated = torch.as_tensor(result["loadings"].to_numpy())
    pattern = torch.as_tensor(result["rotated_loadings"].to_numpy())
    phi = (torch.as_tensor(result["factor_correlations"].to_numpy())
           if "factor_correlations" in result else torch.eye(2, dtype=torch.float64))
    assert torch.allclose(pattern @ phi @ pattern.T, unrotated @ unrotated.T, atol=1e-10, rtol=0)
    assert torch.allclose(phi.diagonal(), torch.ones(2, dtype=torch.float64), atol=1e-10, rtol=0)
    assert result.attrs["rotation_projected_gradient"] <= 1e-5
    if name.startswith("partial"):
        assert result.attrs["target_jacobian_rank"] == (2 if result.attrs["rotation_oblique"] else 1)
        assert result["rotation_target"].isna().sum().sum() == 4
        assert result["rotation_target_mask"].to_numpy().tolist() == mask.to_numpy().tolist()

bootstrap = results["loading_bootstrap"]
draws = torch.as_tensor(bootstrap["replicates"].to_numpy())
assert draws.shape == (199, 12) and bool(torch.isfinite(draws).all())
centred = draws-draws.mean(0)
covariance = centred.T @ centred/198
assert torch.allclose(covariance, torch.as_tensor(bootstrap["covariance"].to_numpy()), atol=1e-12, rtol=1e-12)
intervals = torch.quantile(draws, torch.tensor([.025, .975], dtype=torch.float64), dim=0)
assert torch.allclose(intervals.T, torch.as_tensor(bootstrap["estimates"][["ci_lower", "ci_upper"]].to_numpy()), atol=1e-12, rtol=1e-12)
assert bootstrap["estimates"][["p_value", "df"]].isna().all().all()
assert bootstrap.attrs["successful_replications"] == 199 and not bootstrap.attrs["failed_replications"]

if "MULTIVARIATE_EXTENSION_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["MULTIVARIATE_EXTENSION_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixture.json").write_text(json.dumps(safe_table(frame), allow_nan=False))
    (directory/"bootstrap_fixture.json").write_text(json.dumps(safe_table(bootstrap_frame), allow_nan=False))
    (directory/"declared_target.json").write_text(json.dumps({"target": safe_table(target), "mask": safe_table(mask)}, allow_nan=False))

if "display" in globals():
    for name, result in results.items():
        primary = "estimates" if name == "loading_bootstrap" else ("rotated_loadings" if "rotated_loadings" in result else "loadings")
        globals()["display"](result[primary])

print("MULTIVARIATE_EXTENSION_OK "+json.dumps({"cases": list(results), "physical_fixture_rows": 1203,
    "latent_correlation": .35, "frozen": bool(getattr(sys, "frozen", False)), "weight_total": 2403,
    "complete_score_rows": {name: len(value["data"]) for name, value in scores.items()},
    "saved_table_counts": {name: {key: len(table) for key, table in result.items()} for name, result in results.items()},
    "bootstrap_replications": 199, "bootstrap_covariance_dimension": 12,
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()}}))
