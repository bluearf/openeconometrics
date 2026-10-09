"""Eight complete frequency/summary/RM cases, with full saved postestimation.

Native Run and frozen verification execute this same file. Set
MULTIVARIATE_WEIGHT_MATRIX_RESULT_DIRECTORY to retain full JSON/LaTeX/sample
state; display emits one primary table per scope. No reference estimator runs.
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
rng = random.Random(20261007376)
names = list("abcdef")
rows = []
for i in range(1203):
    f1, f2 = rng.gauss(0, 1), rng.gauss(0, 1)
    loading = [[.8, .1], [-.7, .1], [.75, .2], [.2, .8], [.15, .7], [.35, .75]]
    rows.append([(a*f1+b*f2+.6*rng.gauss(0, 1))*(j+1)+10+j for j, (a, b) in enumerate(loading)])
frame = pd.DataFrame(rows, columns=names)
weighted = frame.assign(w=[i%5 for i in range(1203)])
discriminant_rows = []
for i in range(1203):
    group = i%3
    common = rng.gauss(0, 1)
    discriminant_rows.append([group, *(group*(j+1)*.6+.3*common+(1+.12*group)*rng.gauss(0, 1) for j in range(6))])
discriminant = pd.DataFrame(discriminant_rows, columns=["group", *names]).assign(w=[i%5 for i in range(1203)])
rm_rows = []
for i in range(1203):
    subject = rng.gauss(0, 1)
    rm_rows.append([subject+.12*j+.55*rng.gauss(0, 1) for j in range(6)])
rm_frame = pd.DataFrame(rm_rows, columns=names)


def safe_table(table):
    return table.astype(object).where(table.notna(), None).to_dict(orient="split")


def pack(result):
    if isinstance(result, TableSet):
        return {"kind": "tableset", "attrs": result.attrs,
                "tables": {key: safe_table(value) for key, value in result.items()},
                "axis_names": {key: {"index": value.index.names, "columns": value.columns.names} for key, value in result.items()},
                "latex": result.to_latex()}
    return {"kind": "frame", "attrs": result.attrs, "table": safe_table(result),
            "axis_names": {"index": result.index.names, "columns": result.columns.names}, "latex": result.to_latex()}


def restore_frame(payload, axis_names):
    value = pd.DataFrame(**payload)
    value.index.names = axis_names["index"]
    value.columns.names = axis_names["columns"]
    return value


def restore(payload):
    if payload["kind"] == "tableset":
        return TableSet({key: restore_frame(value, payload["axis_names"][key]) for key, value in payload["tables"].items()}, **payload["attrs"])
    value = restore_frame(payload["table"], payload["axis_names"])
    value.attrs.update(payload["attrs"])
    return value


def full_equal(first, second):
    if isinstance(first, TableSet):
        assert set(first) == set(second)
        for key in first:
            pd.testing.assert_frame_equal(pd.DataFrame(first[key]), pd.DataFrame(second[key]), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    else:
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second), check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)


fits = {}
post = {}
fits["frequency_reliability"] = {"main": oe.alpha(oe.Dataset.from_frame(weighted), names, weights="w", model="guttman", reverse=["b"])}
fits["frequency_adequacy"] = {"main": oe.factortest(oe.Dataset.from_frame(weighted), names, weights="w")}
fits["frequency_discriminant"] = {method: oe.discrim(discriminant, "group", names, weights="w", method=method, loo=True) for method in ("lda", "qda")}
post["frequency_discriminant"] = {method: oe.discrim_predict(result, discriminant) for method, result in fits["frequency_discriminant"].items()}
means = discriminant.groupby("group")[names].mean()
counts = discriminant.groupby("group").size()
covariances = {group: value[names].cov() for group, value in discriminant.groupby("group")}
fits["summary_discriminant"] = {method: oe.discrim_summary(means, covariances, counts, method=method, priors="proportional") for method in ("lda", "qda")}
post["summary_discriminant"] = {method: oe.discrim_predict(result, discriminant) for method, result in fits["summary_discriminant"].items()}
fits["frequency_canonical"] = {"main": oe.canon(oe.Dataset.from_frame(weighted), names[:3], names[3:], weights="w")}
fits["summary_canonical"] = {"main": oe.canon_matrix(frame.cov(), n=len(frame), x=names[:3], y=names[3:], means=frame.mean())}
for key in ("frequency_canonical", "summary_canonical"):
    post[key] = {"main": oe.canon_scores(fits[key]["main"], frame)}
wide = oe.rm_anova_wide(rm_frame, names)
fits["wide_repeated"] = {"main": wide}
C = [[-1., 1., 0., 0., 0., 0.], [0., 0., 0., -1., 0., 1.]]
fits["repeated_contrasts"] = {"main": oe.rm_contrasts(wide, C, contrast_names=["early", "late"])}

encoded = {}
for name, cases in fits.items():
    payload = {"fits": {method: pack(result) for method, result in cases.items()},
               "post": {method: safe_table(value) for method, value in post.get(name, {}).items()},
               "post_axis_names": {method: {"index": value.index.names, "columns": value.columns.names} for method, value in post.get(name, {}).items()}}
    content = json.dumps(payload, sort_keys=True, allow_nan=False)
    again = json.loads(content)
    for method, result in cases.items():
        stored = restore(again["fits"][method])
        full_equal(result, stored)
        if name in ("frequency_discriminant", "summary_discriminant"):
            pd.testing.assert_frame_equal(pd.DataFrame(oe.discrim_predict(stored, discriminant)), pd.DataFrame(post[name][method]), check_dtype=False, atol=1e-12, rtol=1e-12)
        elif name in ("frequency_canonical", "summary_canonical"):
            pd.testing.assert_frame_equal(pd.DataFrame(oe.canon_scores(stored, frame)), pd.DataFrame(post[name][method]), check_dtype=False, atol=1e-12, rtol=1e-12)
        elif name in ("wide_repeated", "repeated_contrasts"):
            full_equal(result, oe.rm_restore(stored))
    encoded[name] = content

for name in ("frequency_reliability", "frequency_adequacy", "frequency_canonical"):
    result = fits[name]["main"]
    assert result.attrs["n"] == 2403 and result.attrs["physical_rows"] == 962
    assert result.attrs["covariance_divisor"] == 2402
for key in ("frequency_discriminant", "summary_discriminant"):
    for predicted in post[key].values():
        assert len(predicted) == 1203 and predicted.notna().all().all()
        posterior = torch.as_tensor(predicted.iloc[:, 1:].to_numpy(dtype="float64"))
        assert torch.allclose(posterior.sum(1), torch.ones(1203, dtype=torch.float64), atol=1e-12, rtol=0)
for key in ("frequency_canonical", "summary_canonical"):
    assert post[key]["main"].shape == (1203, 6) and post[key]["main"].notna().all().all()
assert fits["repeated_contrasts"]["main"]["subject_contrasts"].shape == (1203, 2)

if "MULTIVARIATE_WEIGHT_MATRIX_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["MULTIVARIATE_WEIGHT_MATRIX_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    fixtures = {"canonical": safe_table(frame), "discriminant": safe_table(discriminant), "repeated": safe_table(rm_frame), "C": C}
    (directory/"fixtures.json").write_text(json.dumps(fixtures, allow_nan=False))

if "display" in globals():
    primary = [(fits["frequency_reliability"]["main"]["items"]),
        fits["frequency_adequacy"]["main"], fits["frequency_discriminant"]["lda"]["classification_table_loo"],
        fits["summary_discriminant"]["lda"]["canonical_functions"], fits["frequency_canonical"]["main"]["correlations"],
        fits["summary_canonical"]["main"]["correlations"], wide["within"], fits["repeated_contrasts"]["main"]["estimates"]]
    for table in primary:
        globals()["display"](table)

print("MULTIVARIATE_WEIGHT_MATRIX_OK "+json.dumps({"cases": list(fits), "frozen": bool(getattr(sys, "frozen", False)),
    "fixture_rows": 1203, "weight_total": 2403,
    "full_post_rows": {key: {method: len(value) for method, value in values.items()} for key, values in post.items()},
    "full_subject_contrast_rows": 1203,
    "fit_table_counts": {key: {method: len(result) if isinstance(result, TableSet) else 1 for method, result in values.items()} for key, values in fits.items()},
    "hashes": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in encoded.items()}}))
