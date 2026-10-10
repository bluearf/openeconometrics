"""Development-only full CART comparison with native package-author rpart.

NumPy independently checks weighted objectives. R/rpart is never imported by
production. GPL package code/data remain external. --state compares the actual
saved native/source result, without fitting a substitute OpenEcon model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from openecon.econometrics.supervised import cart as module
from openecon.econometrics.supervised.cart import cart, cart_restore
from openecon.econometrics.supervised.split import prediction_split

RTOL = 2e-10
ATOL = 2e-12
ARCHIVE_SHA = "3183552d74f02749a70e2b989591c561ca0f7c146034f415748acc8480ca4050"

R_REFERENCE = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) > 2 && nzchar(args[3])) .libPaths(c(args[3], .libPaths()))
library(jsonlite)
library(rpart)
packet <- fromJSON(args[1], simplifyVector=TRUE)
data <- as.data.frame(packet$x)
names(data) <- paste0("x",seq_len(ncol(data)))
isclass <- packet$task == "classification"
data$y <- if(isclass) factor(packet$y,levels=packet$classes) else packet$y
data$w <- packet$w
train <- data[packet$train+1,,drop=FALSE]
control <- rpart.control(cp=0,xval=0,minbucket=packet$min_leaf,
 minsplit=packet$min_split,maxdepth=packet$max_depth,maxcompete=0,
 maxsurrogate=0,usesurrogate=0)
prior <- if(isclass) as.numeric(tapply(train$w,train$y,sum))/sum(train$w) else NULL
if(isclass) prior[length(prior)] <- 1-sum(prior[-length(prior)])
params <- if(isclass) list(split="gini",prior=prior) else NULL
fit <- rpart(y~.-w,data=train,weights=w,method=if(isclass) "class" else "anova",
 control=control,parms=params,model=TRUE,x=TRUE,y=TRUE)
frame <- fit$frame
ids <- as.integer(rownames(frame))
terminal <- ids[fit$where]
is_descendant <- function(leaf,parent) {
 while(leaf>parent) leaf <- leaf %/% 2
 leaf==parent
}
nodes <- vector("list",nrow(frame))
splitrow <- 1L
for(i in seq_len(nrow(frame))) {
 rows <- packet$train[vapply(terminal,is_descendant,logical(1),parent=ids[i])]
 internal <- frame$var[i]!="<leaf>"
 threshold <- if(internal) unname(fit$splits[splitrow,"index"]) else NULL
 if(internal) splitrow <- splitrow+1L
 probs <- if(isclass) unname(frame$yval2[i,(nlevels(train$y)+2):(2*nlevels(train$y)+1)]) else NULL
 nodes[[i]] <- list(rows=sort(rows),n=unname(frame$n[i]),weight=unname(frame$wt[i]),
 risk=unname(frame$dev[i]),prediction=if(isclass) NULL else unname(frame$yval[i]),
 probabilities=probs,feature=if(internal) as.integer(sub("x","",frame$var[i]))-1L else NULL,
 threshold=threshold)
}
selected_cp <- packet$requested_cp
if(packet$select=="validation") {
 row <- which(fit$cptable[,"nsplit"]==packet$selected_leaf_count-1L)
 if(length(row)!=1L) stop("Native full cptable has no matching selected stage")
 lower <- fit$cptable[row,"CP"]
 upper <- if(row==1L) lower+1 else fit$cptable[row-1L,"CP"]
 selected_cp <- (lower+upper)/2
}
selected <- prune(fit,cp=selected_cp)
prediction <- if(isclass) as.character(predict(selected,data,type="class")) else unname(predict(selected,data))
probabilities <- if(isclass) unname(predict(selected,data,type="prob")) else NULL
out <- list(R=R.version.string,rpart=as.character(packageVersion("rpart")),selected_cp=selected_cp,
 nodes=nodes,cptable=unname(fit$cptable),predictions=prediction,probabilities=probabilities)
write_json(out,args[2],auto_unbox=TRUE,digits=NA,null="null")
"""


def close(actual, expected, field, maxima):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        raise AssertionError(f"{field} shape {a.shape} != {b.shape}")
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=field)
    maxima[field] = max(maxima.get(field, 0.0), float(np.max(np.abs(a - b))) if a.size else 0.0)


def independent(state):
    """Direct centered weighted losses; no production split sufficient stats."""
    source, trans = state["source"], state["transform"]
    x = np.array([[np.nan if v is None else v for v in row] for row in source["x"]])
    for j in range(x.shape[1]):
        x[np.isnan(x[:, j]), j] = trans["imputation_means"][j]
        x[:, j] = (x[:, j] - trans["centers"][j]) / trans["scales"][j]
    y, w = np.asarray(source["y"]), np.asarray(source["weights"])
    classes = state["classes"]
    normalization = state["normalization"]
    total = normalization["weight_sum"]
    scale = normalization["outcome_scale"]
    maxima = {}

    def stats(rows):
        mass = w[rows].sum()
        if classes:
            counts = np.array([w[rows][y[rows] == value].sum() for value in classes])
            probabilities = counts / mass
            return (
                (mass - counts.max()) / total,
                (mass - (counts**2).sum() / mass) / total,
                probabilities,
            )
        mean = np.average(y[rows], weights=w[rows])
        risk = np.dot(w[rows], (y[rows] - mean) ** 2) / total / scale**2
        return risk, risk, mean

    for node in state["nodes"]:
        rows = node["positions"]
        risk, impurity, prediction = stats(rows)
        close(node["risk"], risk, "independent node risk", maxima)
        close(node["impurity"], impurity, "independent node impurity", maxima)
        close(
            node["probabilities"] if classes else node["prediction"],
            prediction,
            "independent node prediction",
            maxima,
        )
        if node["feature"] is None:
            continue
        candidates = []
        for j in range(x.shape[1]):
            unique = np.unique(x[rows, j])
            for low, high in zip(unique, unique[1:]):
                threshold = low / 2 + high / 2
                left = [i for i in rows if x[i, j] <= threshold]
                right = [i for i in rows if x[i, j] > threshold]
                settings = state["settings"]
                if (
                    min(len(left), len(right)) < settings["min_leaf"]
                    or min(w[left].sum(), w[right].sum()) / total < settings["min_weight_fraction"]
                ):
                    continue
                candidates.append((impurity - stats(left)[1] - stats(right)[1], j, threshold))
        best = max(candidates, key=lambda v: (v[0], -v[1], -v[2]))
        if best[1] != node["feature"]:
            raise AssertionError("Independent exhaustive candidate feature disagrees")
        close(node["threshold"], best[2], "independent split threshold", maxima)
        close(node["split_gain"], best[0], "independent split gain", maxima)
    return x, maxima


def compare(state, rscript, library):
    x, maxima = independent(state)
    settings, derived = state["settings"], state["derived"]
    if settings["min_weight_fraction"] != 0 or settings["max_depth"] == 0:
        raise ValueError(
            "Native rpart comparison requires zero minimum weight fraction and positive max_depth"
        )
    packet = {
        "x": x.tolist(),
        "y": state["source"]["y"],
        "w": state["source"]["weights"],
        "train": state["split"]["positions"]["train"],
        "task": settings["task"],
        "classes": [str(v) for v in state["classes"]],
        "min_leaf": settings["min_leaf"],
        "min_split": settings["min_split"],
        "max_depth": settings["max_depth"],
        "requested_cp": settings["cp"],
        "select": settings["select"],
        "selected_leaf_count": derived["path"][derived["selected_stage"]]["leaf_count"],
    }
    if packet["classes"]:
        packet["y"] = [str(v) for v in packet["y"]]
    with tempfile.TemporaryDirectory(prefix="cart-rpart-oracle-") as directory:
        tmp = Path(directory)
        (tmp / "input.json").write_text(json.dumps(packet))
        (tmp / "reference.R").write_text(R_REFERENCE)
        completed = subprocess.run(
            [
                rscript,
                str(tmp / "reference.R"),
                str(tmp / "input.json"),
                str(tmp / "output.json"),
                library or "",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        reference = json.loads((tmp / "output.json").read_text())
    if reference["rpart"] != "4.1.27":
        raise AssertionError("Native rpart version disagrees with pinned author version")
    actual = {tuple(node["positions"]): node for node in state["nodes"]}
    native = {
        tuple(node["rows"] if isinstance(node["rows"], list) else [node["rows"]]): node
        for node in reference["nodes"]
    }
    if set(actual) != set(native):
        raise AssertionError(
            f"Native maximal node memberships differ: {len(actual)} vs {len(native)}"
        )
    multiplier = state["normalization"]["weight_sum"] * (
        state["normalization"]["outcome_scale"] ** 2 if not state["classes"] else 1
    )
    for rows, node in actual.items():
        ref = native[rows]
        if node["n"] != ref["n"] or node["feature"] != ref["feature"]:
            raise AssertionError("Native node sample count or primary feature disagrees")
        close(node["weight_sum"], ref["weight"], "native node weights", maxima)
        close(node["risk"] * multiplier, ref["risk"], "native node physical risk", maxima)
        if node["feature"] is not None:
            close(node["threshold"], ref["threshold"], "native split thresholds", maxima)
        close(
            node["probabilities"] if state["classes"] else node["prediction"],
            ref["probabilities"] if state["classes"] else ref["prediction"],
            "native node prediction",
            maxima,
        )
    # rpart cptable runs root-to-maximal; our event convention runs maximal-to-root.
    native_path = np.asarray(reference["cptable"], dtype=float)
    path = list(reversed(derived["path"]))
    if len(path) != len(native_path):
        raise AssertionError(f"Native full path length differs {len(path)} vs {len(native_path)}")
    close([p["cp"] for p in path], native_path[:, 0], "native all pruning cp", maxima)
    close(
        [p["leaf_count"] - 1 for p in path], native_path[:, 1], "native all pruning splits", maxima
    )
    root_risk = state["nodes"][0]["risk"]
    close(
        [p["risk"] / root_risk for p in path],
        native_path[:, 2],
        "native all pruning relative risk",
        maxima,
    )
    if state["classes"]:
        if [str(v) for v in derived["prediction"]["predictions"]] != reference["predictions"]:
            raise AssertionError("Native selected class predictions disagree")
        close(
            derived["prediction"]["probabilities"],
            reference["probabilities"],
            "native selected probabilities",
            maxima,
        )
    else:
        close(
            derived["prediction"]["predictions"],
            reference["predictions"],
            "native selected predictions",
            maxima,
        )
    return {
        "native_R": reference["R"],
        "native_rpart": reference["rpart"],
        "max_absolute_errors": maxima,
        "nodes": len(actual),
        "full_pruning_stages": len(path),
        "source_digest": state["digest"],
        "native_reference": reference,
        "native_stage_selection": "requested fixed cp, or matching cptable leaf-count stage at its native interior cp for validation selection",
    }


def source_fixture(task, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(96, 3))
    data = pd.DataFrame(x, columns=["a", "b", "c"])
    data["y"] = (
        x[:, 0] + 0.4 * x[:, 1] + rng.normal(size=96) * 0.2
        if task == "regression"
        else np.where(x[:, 0] > 0.4, "C", np.where(x[:, 1] > 0, "B", "A"))
    )
    data["w"] = rng.uniform(0.3, 3, size=96)
    split = prediction_split(data, roles=["train"] * 72 + ["validation"] * 12 + ["test"] * 12)
    return cart(
        data,
        outcome="y",
        features=["a", "b", "c"],
        weights="w",
        split=split,
        task=task,
        min_leaf=3,
        max_depth=4,
        max_nodes=31,
        select="validation",
    )


def author_example(rscript, library):
    """Use all 81 package-author kyphosis rows externally; no vendored data."""
    code = '.libPaths(c(commandArgs(TRUE)[2],.libPaths()));library(rpart);library(jsonlite);write_json(kyphosis,commandArgs(TRUE)[1],dataframe="columns",digits=NA)'
    with tempfile.TemporaryDirectory(prefix="cart-original-author-") as directory:
        path = Path(directory) / "kyphosis.json"
        subprocess.run(
            [rscript, "-e", code, str(path), library or ""],
            check=True,
            timeout=30,
            capture_output=True,
        )
        raw = json.loads(path.read_text())
        data = pd.DataFrame(raw)
        data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    split = prediction_split(data, roles=["train"] * len(data))
    result = cart(
        data,
        outcome="Kyphosis",
        features=["Age", "Number", "Start"],
        split=split,
        task="classification",
        min_leaf=7,
        min_split=20,
        max_depth=20,
        max_nodes=127,
        cp=0.01,
        max_work=2_000_000_000,
    )
    return result, {
        "dataset": "full 81-row rpart package-author kyphosis",
        "external_source_sha256": data_hash,
        "vendor_data_copied": False,
        "published_example_controls": "default minsplit=20/minbucket=7/cp=.01; maximal cp0 path compared, saved cp.01 subtree",
        "license": "rpart GPL-2 | GPL-3, external development only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--author-example", action="store_true")
    parser.add_argument("--rscript", default=shutil.which("Rscript"))
    parser.add_argument("--r-library", default="/tmp/openecon-kfas-native-library")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.rscript:
        raise RuntimeError("Native Rscript is required for this development oracle")
    records, metadata = [], {}
    if args.state:
        raw = args.state.read_bytes()
        # Regression barrier: existing cached values, never substitute a fit.
        module._grow = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("cached native model regenerated")
        )
        records = [cart_restore(raw).attrs["state"]]
        metadata = {
            "actual_saved_path": str(args.state.resolve()),
            "actual_saved_sha256": hashlib.sha256(raw).hexdigest(),
            "no_fit_or_grow_called": True,
        }
    elif args.author_example:
        result, metadata = author_example(args.rscript, args.r_library)
        records = [result.attrs["state"]]
    else:
        records = [
            source_fixture(task, seed).attrs["state"]
            for task in ("regression", "classification")
            for seed in (8813, 19423)
        ]
        metadata["synthetic_cases"] = [
            {"task": task, "seed": seed, "rows": 96, "train": 72, "validation": 12, "test": 12}
            for task in ("regression", "classification")
            for seed in (8813, 19423)
        ]
    report = {
        "schema": "openecon.cart-oracle.v1",
        "author_archive_sha256": ARCHIVE_SHA,
        "primary_reference": "https://stat.ethz.ch/CRAN/web/packages/rpart/vignettes/longintro.pdf",
        "license": "GPL-2 | GPL-3 external author reference; independently derived production code",
        "tolerances_declared": {
            "relative": RTOL,
            "absolute": ATOL,
            "membership_feature_class_and_path_counts": "exact",
        },
        "boundary": "rpart package-author validation; proprietary Breiman CART/IBM/Stata not executed",
        "metadata": metadata,
        "cases": [compare(record, args.rscript, args.r_library) for record in records],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"passed": len(records), "receipt": str(args.output)}, ensure_ascii=False))


def fixed_original_source_fixture(task, seed):
    """Fit unchanged CART to complete immutable original author reference inputs.

    The legacy source_fixture generator remains unchanged and separate. Here
    the seed is the original four-case label, never a new random-input draw.
    No original fitted tree or reference output is passed into production.
    """
    import importlib.util
    loader_path = Path(__file__).resolve().parents[1] / "tests/reference/supervised_cart_fixed_inputs.py"
    spec = importlib.util.spec_from_file_location("cart_original_fixed_inputs", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    identity = loader.load_original_case(task, seed)
    source = identity["source"]
    index = module.c.restore_index(source["index"], identity["n"])
    columns = {}
    for position, feature in enumerate(identity["features"]):
        columns[feature] = pd.Series([row[position] for row in source["x"]],
                                    dtype=source["dtypes"]["features"][position], index=index)
    columns[identity["outcome"]] = pd.Series(source["y"], dtype=source["dtypes"]["outcome"], index=index)
    columns[identity["weight_name"]] = pd.Series(source["weights"], dtype=source["dtypes"]["weights"], index=index)
    frame = pd.DataFrame(columns, index=index)
    split_body = identity["split"]
    split = prediction_split(frame, roles=split_body["explicit_roles"], **split_body["settings"])
    if loader._canonical(split.payload) != loader._canonical(split_body):
        raise AssertionError("Reconstructed original CART full split/index/settings changed.")
    result = cart(frame, outcome=identity["outcome"], features=identity["features"],
                  weights=identity["weight_name"], split=split, **identity["settings"])
    actual = {key: result.attrs["state"][key] for key in loader.INPUT_FIELDS}
    if loader._canonical(actual) != loader._canonical(identity):
        raise AssertionError("Fresh estimator did not retain exact original CART physical inputs.")
    return result


if __name__ == "__main__":
    main()
