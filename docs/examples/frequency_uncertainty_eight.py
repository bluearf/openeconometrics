"""Eight compressed literal-IID-unit frequency uncertainty acceptance cases.

The support atoms are fixed before independent unit category draws determine
their integer frequencies. Complete-case deletion and complete zero-frequency
rows are explicit. Production methods never materialize repeated measurements.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.multivariate.frequency_bootstrap import validate_saved
from openecon.econometrics.postest.index_codec import encode, decode

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


if "FREQUENCY_REPLAY_FIXTURES" in globals():
    fixture = globals()["FREQUENCY_REPLAY_FIXTURES"]
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
fits = {
    "covariance_axes": oe.pca_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="covariance", anchors=["x1", "x4"], replications=199, seed=71),
    "correlation_axes": oe.pca_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="correlation", anchors=["x1", "x4"], replications=199, seed=71),
    "covariance_subspace": oe.pca_subspace_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="covariance", replications=199, seed=71),
    "correlation_subspace": oe.pca_subspace_fweight_bootstrap(spectral, names, weights="freq", components=2, matrix="correlation", replications=199, seed=71),
    "canonical_correlations": oe.canon_fweight_bootstrap(canonical, xs, ys, weights="freq", components=2, target="correlations", replications=199, seed=19),
    "canonical_coefficients": oe.canon_fweight_bootstrap(canonical, xs, ys, weights="freq", components=2, target="coefficients", anchors=["x1", "x2"], replications=199, seed=19),
    "unrotated_factor": oe.factor_multifactor_fweight_bootstrap(spectral, names, weights="freq", factors=2, anchors=["x1", "x4"], replications=199, seed=53),
    "target_factor": oe.factor_multifactor_fweight_bootstrap(spectral, names, weights="freq", factors=2, target=target, replications=199, seed=53),
}


def pack(result):
    return {"summary": oe.summary_state(result), "latex": result.to_latex(),
            "table_order": list(result),
            "table_dtypes": {name: [str(dtype) for dtype in frame.dtypes] for name, frame in result.items()}}


def restore_pack(state):
    result = oe.restore_summary(state["summary"])
    assert json.loads(oe.summary_state(result)) == json.loads(state["summary"])
    for name, dtypes in state["table_dtypes"].items():
        result[name] = result[name].astype(dict(zip(result[name].columns, dtypes)))
    result = type(result)({name: result[name] for name in state["table_order"]}, title=result.title, **result.attrs)
    assert result.to_latex() == state["latex"]
    assert oe.summary_state(result) == state["summary"]
    validate_saved(result)
    return result


encoded = {}
for name, result in fits.items():
    state = pack(result)
    restore_pack(state)
    encoded[name] = json.dumps({"fit": state, "post": None}, sort_keys=True, allow_nan=False)
fixtures = {"spectral": portable_frame(spectral), "canonical": portable_frame(canonical)}
if "SPECTRAL_GEOMETRY_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["SPECTRAL_GEOMETRY_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixtures.json").write_text(json.dumps(fixtures, sort_keys=True, allow_nan=False))
if "display" in globals():
    for result in fits.values():
        globals()["display"](result["estimates"])
print("FREQUENCY_UNCERTAINTY_EIGHT_OK "+json.dumps({
    "cases": list(fits), "frozen": bool(getattr(sys, "frozen", False)),
    "fixture_rows": len(spectral), "canonical_rows": len(canonical), "bootstrap_replications": 199,
    "full_saved_roundtrip": True, "fit_table_counts": {name: len(result) for name, result in fits.items()},
    "n_units": {name: result.attrs["n_units"] for name, result in fits.items()},
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()},
}))
