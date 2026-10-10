"""Eight fixed-family simultaneous query-score contrast acceptance domains.

Training fixtures contain missing and zero-frequency support rows. The three
query positions are fixed independently; one missing row remains aligned.
Complete training fits and joint results are saved separately from previews.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.postest.index_codec import encode, decode
from openecon.resources import use_workspace_budget

torch.set_num_threads(2)
names = [f"x{j+1}" for j in range(6)]
xs, ys = ["x1", "x2", "x3"], ["y1", "y2", "y3"]

def unit_counts(seed):
    rng = random.Random(seed)
    counts = [0] * 603
    for _ in range(1800):
        counts[rng.randrange(603)] += 1
    return counts


def portable_frame(frame):
    return {"columns": list(frame.columns),
            "data": frame.astype(object).where(pd.notna(frame), None).to_numpy().tolist(),
            "index_codes": [encode(value) for value in frame.index],
            "index_names": [encode(value) for value in frame.index.names],
            "column_axis_names": [encode(value) for value in frame.columns.names]}


def restored_frame(payload):
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    labels = [decode(value) for value in payload["index_codes"]]
    axis_names = [decode(value) for value in payload["index_names"]]
    if len(axis_names) > 1:
        frame.index = pd.MultiIndex.from_tuples(labels, names=axis_names)
    else:
        frame.index = pd.Index(labels, name=axis_names[0], tupleize_cols=False)
    frame.columns.name = decode(payload["column_axis_names"][0])
    return frame


if "CONTRAST_REPLAY_FIXTURES" in globals():
    fixture = globals()["CONTRAST_REPLAY_FIXTURES"]
    spectral, canonical = restored_frame(fixture["spectral"]), restored_frame(fixture["canonical"])
else:
    rng = random.Random(20261008568)
    spectral_rows = []
    for _ in range(603):
        f1, f2 = rng.gauss(0, 1.25), rng.gauss(0, .9)
        spectral_rows.append([
            .85*f1+.10*f2+rng.gauss(0, .65), .75*f1+.15*f2+rng.gauss(0, .70),
            .65*f1+.20*f2+rng.gauss(0, .75), .12*f1+.85*f2+rng.gauss(0, .65),
            .10*f1+.72*f2+rng.gauss(0, .70), .20*f1+.62*f2+rng.gauss(0, .75),
        ])
    spectral = pd.DataFrame(spectral_rows, columns=names)
    spectral["freq"] = unit_counts(314159)
    spectral.index = pd.MultiIndex.from_tuples([(i//100, f"atom-{i:04}") for i in range(603)], names=["block", "atom"])
    spectral.columns.name = "measurement"
    spectral.iloc[[10, 120, 230], 0] = float("nan")
    rng = random.Random(178)
    rows = []
    for _ in range(603):
        latent = [rng.gauss(0, 1) for _ in range(3)]
        errors = [rng.gauss(0, 1) for _ in range(6)]
        x = [latent[j]+.35*errors[j] for j in range(3)]
        y = [latent[j]*[.9, .5, .18][j]+errors[j+3]*[.35, .7, 1.0][j] for j in range(3)]
        rows.append([value*scale+shift for value, scale, shift in zip(x+y, [3, 1.7, 2.2, 4, .8, 1.4], [5, -9, 13, .5, 8, -1])])
    canonical = pd.DataFrame(rows, columns=xs+ys)
    canonical["freq"] = unit_counts(271828)
    labels = [True, "True", "duplicate", "duplicate", None, ("typed", 1)] + [f"atom-{i}" for i in range(6, 603)]
    canonical.index = pd.Index(labels, name="atom", tupleize_cols=False)
    canonical.columns.name = "joint_measurement"
    canonical.iloc[[12, 122, 232], 0] = float("nan")

target = [[.8, 0], [.75, .1], [.7, -.05], [0, .65], [.1, .6], [-.05, .55]]
options = dict(replications=199, confidence=.95)
query = pd.DataFrame([[.2, -.4, .6, .5, -.3, .7], [float("nan")]*6,
                      [1.1, -.8, .3, -.2, .9, -.7]], columns=names)
query.index = pd.Index([("fixed", 1), ("fixed", 1), pd.Timestamp("2026-01-04", tz="UTC")], tupleize_cols=False, name="query")
canonical_query = query.copy()
canonical_query.columns = xs + ys
with use_workspace_budget(512):
    fits = {
        "pca_covariance": oe.pca_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="covariance", anchors=["x1", "x4"], seed=71, **options),
        "pca_correlation": oe.pca_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="correlation", anchors=["x1", "x4"], seed=71, **options),
        "cca_iid": oe.canon_bootstrap(canonical, xs, ys, components=2, target="coefficients", anchors=["x1", "x2"], seed=19, **options),
        "cca_frequency": oe.canon_fweight_bootstrap(canonical, xs, ys, weights="freq", components=2, target="coefficients", anchors=["x1", "x2"], seed=19, **options),
        "pf_iid": oe.factor_multifactor_bootstrap(spectral, names, factors=2, anchors=["x1", "x4"], seed=53, **options),
        "target_iid": oe.factor_multifactor_bootstrap(spectral, names, factors=2, target=target, seed=53, **options),
        "pf_frequency": oe.factor_multifactor_fweight_bootstrap(spectral, names, weights="freq", factors=2, anchors=["x1", "x4"], seed=53, **options),
        "target_frequency": oe.factor_multifactor_fweight_bootstrap(spectral, names, weights="freq", factors=2, target=target, seed=53, **options),
    }
    results = {}
    for name, fit in fits.items():
        function = oe.pca_fweight_bootstrap_score_contrasts if name.startswith("pca") else oe.canon_bootstrap_score_contrasts if name.startswith("cca") else oe.factor_bootstrap_score_contrasts
        measurement = canonical_query if name.startswith("cca") else query
        axes = 4 if name.startswith("cca") else 2
        coefficients = [[0.0]*(3*axes) for _ in range(3)]
        coefficients[0][0], coefficients[0][2*axes] = 1.0, -1.0
        coefficients[1][1], coefficients[1][2*axes+1] = .75, 1.25
        coefficients[2][0], coefficients[2][axes-1] = 1.0, -.5
        results[name] = function(oe.restore_summary(oe.summary_state(fit)), measurement,
                                 coefficients, labels=["first query difference", "second axis blend", "within query contrast"])


def pack(result):
    return {"summary": oe.summary_state(result), "latex": result.to_latex(),
            "table_order": list(result),
            "table_dtypes": {name: [str(dtype) for dtype in frame.dtypes] for name, frame in result.items()}}


def restore_pack(state):
    result = oe.restore_summary(state["summary"])
    for name, dtypes in state["table_dtypes"].items():
        result[name] = result[name].astype(dict(zip(result[name].columns, dtypes, strict=True)))
    result = type(result)({name: result[name] for name in state["table_order"]}, title=result.title, **result.attrs)
    assert oe.summary_state(result) == state["summary"]
    assert result.to_latex() == state["latex"]
    return result


encoded = {}
for name, result in results.items():
    assert result.attrs["score_attrs"]["query_positions"] == [0, 2]
    assert result["score__point_scores"].iloc[1].isna().all()
    state = {"fit": pack(fits[name]), "post": pack(result)}
    restore_pack(state["fit"])
    restore_pack(state["post"])
    encoded[name] = json.dumps(state, sort_keys=True, allow_nan=False)
fixtures = {"spectral": portable_frame(spectral), "canonical": portable_frame(canonical),
            "query": portable_frame(query), "canonical_query": portable_frame(canonical_query)}
if "CONTRAST_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["CONTRAST_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixtures.json").write_text(json.dumps(fixtures, sort_keys=True, allow_nan=False))
if "display" in globals():
    for result in results.values():
        globals()["display"](result["estimates"])
print("MULTIVARIATE_SCORE_CONTRASTS_EIGHT_OK "+json.dumps({
    "cases": list(results), "frozen": bool(getattr(sys, "frozen", False)),
    "fixture_rows": len(spectral), "bootstrap_replications": 199,
    "full_saved_roundtrip": True, "query_positions": [0, 2], "contrasts_per_family": 3,
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()},
}))
