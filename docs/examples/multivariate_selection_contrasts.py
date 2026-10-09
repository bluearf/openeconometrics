"""Eight bounded selection/MANOVA/RM cases with complete saved follow-up.

Set MULTIVARIATE_SELECTION_RESULT_DIRECTORY to retain full JSON rather than
console previews. Source, frozen and native acceptance execute this same file.
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
rng = random.Random(20261007528)
names = [f"x{j+1}" for j in range(6)]
outcomes = [f"y{j+1}" for j in range(4)]
rows = []
rm_rows = []
for i in range(603):
    group = i % 3
    group_name = ["control", "early", "late"][group]
    a, b = rng.gauss(0, 1), rng.gauss(0, 1)
    values = [a + .7*group, .6*a + .8*group + rng.gauss(0, .7),
              b + .5*(group == 2), .3*b + rng.gauss(0, 1),
              rng.gauss(0, 1), .2*a + rng.gauss(0, 1)]
    ys = [3+j + .25*(j+1)*group + .3*a + .15*b + rng.gauss(0, 1)
          for j in range(4)]
    rows.append([group_name, *values, *ys, i % 5])
    subject = rng.gauss(0, 1)
    for time in range(2):
        for condition in range(2):
            rm_rows.append([f"subject-{i:04}", group_name, time, condition,
                            4 + subject + .25*group + .4*time + .15*condition
                            + .18*group*time + rng.gauss(0, .7)])
data = pd.DataFrame(rows, columns=["group", *names, *outcomes, "w"])
rm_data = pd.DataFrame(rm_rows, columns=["subject", "group", "time", "condition", "response"])
query = data.copy()
query.loc[7, "x1"] = float("nan")


def safe_table(value):
    payload = value.astype(object).where(value.notna(), None).to_dict(orient="split")
    if len(value.columns) == 0:
        payload["data"] = [[] for _ in value.index]
    return payload


def pack(value):
    if isinstance(value, TableSet):
        return {"kind": "tableset", "title": value.title, "attrs": value.attrs,
                "tables": {key: safe_table(frame) for key, frame in value.items()},
                "axis_names": {key: {"index": frame.index.names, "columns": frame.columns.names}
                               for key, frame in value.items()}, "latex": value.to_latex()}
    return {"kind": "frame", "attrs": value.attrs, "table": safe_table(value),
            "axis_names": {"index": value.index.names, "columns": value.columns.names},
            "latex": value.to_latex()}


def restore_frame(spec, axes):
    value = pd.DataFrame(**spec)
    value.index.names, value.columns.names = axes["index"], axes["columns"]
    return value


def restore(spec):
    if spec["kind"] == "tableset":
        return TableSet({key: restore_frame(value, spec["axis_names"][key])
                         for key, value in spec["tables"].items()}, title=spec["title"], **spec["attrs"])
    value = restore_frame(spec["table"], spec["axis_names"])
    value.attrs.update(spec["attrs"])
    return value


def same(first, second):
    if isinstance(first, TableSet):
        assert first.title == second.title and first.attrs == second.attrs
        assert set(first) == set(second)
        for key in first:
            same(first[key], second[key])
    else:
        pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(second),
                                      check_dtype=False, check_exact=False,
                                      check_index_type=False if first.index.empty and second.index.empty else "equiv",
                                      check_column_type=False if first.columns.empty and second.columns.empty else "equiv",
                                      rtol=1e-12, atol=1e-12)


fits, post = {}, {}
for method in ("forward", "backward", "stepwise"):
    key = method + "_selection"
    fits[key] = oe.discrim_stepwise(data, "group", names, method=method,
                                   include=["x1"], priors="proportional")
    post[key] = oe.discrim_stepwise_predict(fits[key], query)
fits["frequency_selection"] = oe.discrim_stepwise(
    data, "group", names, weights="w", include=["x1"], priors="proportional")
post["frequency_selection"] = oe.discrim_stepwise_predict(fits["frequency_selection"], query)
fits["frequency_manova"] = oe.manova_oneway(data, outcomes, "group", weights="w")

summary_means, summary_covariances, summary_counts = [], {}, []
group_order = sorted(data["group"].unique().tolist())
for label in group_order:
    block = data.loc[data.group == label]
    weights = torch.tensor(block.w.to_numpy(), dtype=torch.float64)
    values = torch.tensor(block[outcomes].to_numpy(), dtype=torch.float64)
    total = int(weights.sum())
    offset = values-values[0]
    location = (weights[:, None]*offset).sum(0)/total
    location += (weights[:, None]*(offset-location)).sum(0)/total
    residual = offset-location
    mean = values[0]+location
    covariance = residual.T @ (weights[:, None]*residual)/(total-1)
    summary_means.append(mean.tolist())
    summary_covariances[label] = pd.DataFrame(covariance.numpy(), index=outcomes, columns=outcomes)
    summary_counts.append(total)
means = pd.DataFrame(summary_means, index=group_order, columns=outcomes)
counts = pd.Series(summary_counts, index=group_order)
fits["summary_manova"] = oe.manova_summary(means, summary_covariances, counts)

M = [[1., 0.], [-1., 1.], [0., -1.], [0., 0.]]
L_oneway = [[-1., 1., 0.], [-1., 0., 1.]]
null = [[.1, 0.], [0., -.1]]
for key in ("frequency_manova", "summary_manova"):
    post[key] = oe.manova_contrast(fits[key], L_oneway, M=M, null=null,
                                  contrast_names=["early-control", "late-control"],
                                  transform_names=["first-gap", "second-gap"])

base_manova = oe.manova(data, outcomes, ["group"], covariates=["x1"])
base_dataset = oe.manova(oe.Dataset.from_frame(data), outcomes, ["group"], covariates=["x1"])
design_columns = base_manova.attrs["manova_contrast_state"]["design_columns"]
L = [[float(name == design_columns[2]) for name in design_columns],
     [float(name == "x1") for name in design_columns]]
fits["saved_manova_contrast"] = oe.manova_contrast(
    base_manova, L, M=M, null=null, contrast_names=["group-effect", "covariate"],
    transform_names=["first-gap", "second-gap"])
post["saved_manova_contrast"] = oe.manova_contrast(
    base_dataset, L, M=M, null=null, contrast_names=["group-effect", "covariate"],
    transform_names=["first-gap", "second-gap"])
# Different input provenance has distinct saved-state digests; compare every numerical table.
for key in fits["saved_manova_contrast"]:
    if key != "settings":
        same(fits["saved_manova_contrast"][key], post["saved_manova_contrast"][key])
base_rm = oe.rm_anova(rm_data, "response", "subject", ["time", "condition"], between=["group"])
L_rm = [[0., 1., 0.], [0., 0., 1.]]
fits["saved_rm_joint"] = oe.rm_mtest(
    base_rm, L_rm, M=M, null=null, contrast_names=["group-1", "group-2"],
    transform_names=["first-gap", "second-gap"])

# A valid empty screening model must retain its prior-only prediction.
balanced = pd.concat([pd.DataFrame({"group": label, "x": [-2., -1., 0., 1., 2.]})
                      for label in group_order], ignore_index=True)
empty = oe.discrim_stepwise(balanced, "group", ["x"], method="forward")
empty_predictions = oe.discrim_stepwise_predict(empty, pd.DataFrame(index=range(7)))
assert len(empty_predictions) == 7
same(empty, restore(pack(empty)))

encoded = {}
for name, result in fits.items():
    payload = {"fit": pack(result), "post": pack(post[name]) if name in post else None}
    content = json.dumps(payload, sort_keys=True, allow_nan=False)
    parsed = json.loads(content)
    stored = restore(parsed["fit"])
    same(result, stored)
    same(result, oe.restore_summary(oe.summary_state(result)))
    if name.endswith("selection"):
        prediction = oe.discrim_stepwise_predict(stored, query)
        same(prediction, post[name])
        assert len(prediction) == 603 and prediction.index.equals(query.index)
        assert prediction.iloc[7].isna().all()
        assert prediction.drop(index=7).notna().all().all()
    elif name in ("frequency_manova", "summary_manova"):
        same(post[name], oe.manova_contrast(stored, L_oneway, M=M, null=null,
             contrast_names=["early-control", "late-control"], transform_names=["first-gap", "second-gap"]))
    encoded[name] = content

base_models = {"manova": pack(base_manova), "manova_dataset": pack(base_dataset),
               "rm": pack(base_rm), "empty_selection": pack(empty),
               "empty_predictions": pack(empty_predictions)}
fixtures = {"data": safe_table(data), "query": safe_table(query), "rm_data": safe_table(rm_data),
            "M": M, "L": L, "L_oneway": L_oneway, "L_rm": L_rm, "null": null,
            "summary_means": safe_table(means), "summary_counts": summary_counts,
            "summary_covariances": {key: safe_table(value) for key, value in summary_covariances.items()}}
if "MULTIVARIATE_SELECTION_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["MULTIVARIATE_SELECTION_RESULT_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixtures.json").write_text(json.dumps(fixtures, sort_keys=True, allow_nan=False))
    (directory/"base-models.json").write_text(json.dumps(base_models, sort_keys=True, allow_nan=False))

if "display" in globals():
    for name, result in fits.items():
        globals()["display"](result["selection_history"] if name.endswith("selection") else result["multivariate"])

print("MULTIVARIATE_SELECTION_CONTRASTS_OK " + json.dumps({
    "cases": list(fits), "frozen": bool(getattr(sys, "frozen", False)),
    "fixture_rows": 603, "rm_long_rows": len(rm_data), "weight_total": int(data.w.sum()),
    "prediction_rows": {name: len(value) for name, value in post.items() if name.endswith("selection")},
    "empty_prediction_rows": len(empty_predictions),
    "fit_table_counts": {name: len(value) for name, value in fits.items()},
    "post_table_counts": {name: len(value) if isinstance(value, TableSet) else 1 for name, value in post.items()},
    "hashes": {name: hashlib.sha256(content.encode()).hexdigest() for name, content in encoded.items()}}))
