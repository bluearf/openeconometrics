"""Eight fixed spectral geometry/score contracts with complete saved payloads.

Source, frozen and actual native acceptance execute this same file. A caller
sets the Python global SPECTRAL_GEOMETRY_RESULT_DIRECTORY to save every table,
all draw vectors/settings, original LaTeX and declared rendering dtypes.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch
import openecon as oe

torch.set_num_threads(2)
rng = random.Random(20261008568)
names = [f"x{j+1}" for j in range(6)]
spectral_rows = []
for i in range(603):
    f1, f2 = rng.gauss(0, 1.25), rng.gauss(0, .9)
    spectral_rows.append([
        .85*f1+.10*f2+rng.gauss(0, .65), .75*f1+.15*f2+rng.gauss(0, .70),
        .65*f1+.20*f2+rng.gauss(0, .75), .12*f1+.85*f2+rng.gauss(0, .65),
        .10*f1+.72*f2+rng.gauss(0, .70), .20*f1+.62*f2+rng.gauss(0, .75),
    ])
spectral = pd.DataFrame(spectral_rows, columns=names)
spectral.index = pd.MultiIndex.from_tuples([(i//100, f"subject-{i:04}") for i in range(603)], names=["block", "subject"])
spectral.columns.name = "measurement"
spectral.iloc[[10, 120, 230], 0] = float("nan")

rng = random.Random(178)
canonical_rows = []
for i in range(603):
    latent = [rng.gauss(0, 1) for _ in range(3)]
    errors = [rng.gauss(0, 1) for _ in range(6)]
    x = [latent[j]+.35*errors[j] for j in range(3)]
    y = [latent[j]*[.9, .5, .18][j]+errors[j+3]*[.35, .7, 1.0][j] for j in range(3)]
    canonical_rows.append([value*scale+shift for value, scale, shift in zip(x+y, [3, 1.7, 2.2, 4, .8, 1.4], [5, -9, 13, .5, 8, -1])])
xs, ys = ["x1", "x2", "x3"], ["y1", "y2", "y3"]
canonical = pd.DataFrame(canonical_rows, columns=xs+ys)
counts = pd.DataFrame([[1300, 150, 120, 80], [100, 1000, 190, 110], [90, 140, 600, 400]],
                      index=pd.Index(["first", "second", "third"], name="row"),
                      columns=pd.Index(["A", "B", "C", "D"], name="column"))
query = pd.DataFrame([[.2, -.3, .5, .4, .1, -.2],
                      [1., 2., 1.5, -.5, -.3, .1],
                      [-1., -.8, -.9, .7, .6, .4],
                      [1.5, 1.2, .9, -.5, -.3, -.1]], columns=names)
query.iloc[1, 0] = float("nan")
query.index = pd.MultiIndex.from_tuples([(True, "fixed-A"), (False, "missing"), (True, "fixed-A"), (False, "fixed-B")], names=["flag", "query"])

fits = {
    "covariance_subspace": oe.pca_subspace_bootstrap(spectral, names, components=2, matrix="covariance", replications=199, seed=71),
    "correlation_subspace": oe.pca_subspace_bootstrap(spectral, names, components=2, matrix="correlation", replications=199, seed=71),
    "canonical_correlations": oe.canon_bootstrap(canonical, xs, ys, components=2, target="correlations", replications=199, seed=19),
    "canonical_coefficients": oe.canon_bootstrap(canonical, xs, ys, components=2, target="coefficients", anchors=["x1", "x2"], replications=199, seed=19),
    "multinomial_ca": oe.ca_bootstrap(counts, dimensions=2, sampling="multinomial", replications=199, seed=317),
    "row_multinomial_ca": oe.ca_bootstrap(counts, dimensions=2, sampling="row_multinomial", replications=199, seed=317),
}
for matrix in ("covariance", "correlation"):
    fit = oe.pca_bootstrap(spectral, names, components=2, matrix=matrix, anchors=["x1", "x4"], replications=199, seed=71)
    fits[f"{matrix}_scores"] = oe.pca_bootstrap_scores(oe.restore_summary(oe.summary_state(fit)), query)


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
    return result


encoded = {}
for name, result in fits.items():
    state = pack(result)
    restore_pack(state)
    encoded[name] = json.dumps({"fit": state, "post": None}, sort_keys=True, allow_nan=False)


def portable_frame(frame):
    values = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="split")
    values["index_codes"] = result_index_codes(frame.index)
    return values


def result_index_codes(index):
    from openecon.econometrics.postest.index_codec import encode
    return [encode(value) for value in index]


fixtures = {"spectral": portable_frame(spectral), "canonical": portable_frame(canonical),
            "counts": portable_frame(counts), "query": portable_frame(query)}
if "SPECTRAL_GEOMETRY_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["SPECTRAL_GEOMETRY_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixtures.json").write_text(json.dumps(fixtures, sort_keys=True, allow_nan=False))
if "display" in globals():
    for result in fits.values():
        globals()["display"](result["estimates"])
print("SPECTRAL_GEOMETRY_EIGHT_OK "+json.dumps({
    "cases": list(fits), "frozen": bool(getattr(sys, "frozen", False)),
    "fixture_rows": len(spectral), "canonical_rows": len(canonical), "bootstrap_replications": 199,
    "full_saved_roundtrip": True, "fit_table_counts": {name: len(result) for name, result in fits.items()},
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()},
}))
