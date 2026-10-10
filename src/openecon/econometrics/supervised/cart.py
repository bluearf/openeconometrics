"""Continuous-feature weighted CART with full weakest-link pruning and replay.

Greedy splits exhaust all distinct-value boundaries. Gini selects classification
splits; weighted misclassification, rather than Gini, defines its pruning risk.
Adaptive tree uncertainty is unavailable. This is a prediction model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from openecon.econometrics.core import TableSet, table
from openecon.resources import plan_workspace
from . import common as c
from . import metrics
from .split import SplitState, prediction_split, split_restore, replay as split_replay

SCHEMA = "openecon.supervised.cart.v1"
RESULT_SCHEMA = "openecon.supervised.cart-result.v1"
NODE_KEYS = (
    "id",
    "depth",
    "positions",
    "n",
    "weight_sum",
    "prediction",
    "probabilities",
    "predicted_class",
    "risk",
    "impurity",
    "feature",
    "threshold",
    "split_gain",
    "left",
    "right",
    "stop_reason",
)


def _settings(
    *,
    task,
    min_leaf,
    min_split,
    min_weight_fraction,
    max_depth,
    max_nodes,
    impute,
    standardize,
    cp,
    select,
    max_work,
):
    if task not in ("regression", "classification") or type(task) is not str:
        c.fail("invalid_option", "task must be regression or classification.")
    c.integer(min_leaf, 1, c.MAX_ROWS, "min_leaf")
    if min_split is None:
        min_split = 2 * min_leaf
    c.integer(min_split, 2, c.MAX_ROWS * 2, "min_split")
    c.real(min_weight_fraction, 0, 0.49, "min_weight_fraction")
    c.integer(max_depth, 0, 20, "max_depth")
    c.integer(max_nodes, 1, 2047, "max_nodes")
    if max_nodes % 2 != 1:
        c.fail("invalid_option", "max_nodes must be odd for a complete binary tree.")
    if (
        impute not in (None, "mean")
        or standardize not in ("identity", "population")
        or select not in ("cp", "validation")
    ):
        c.fail(
            "invalid_option",
            "Use explicit mean imputation, identity/population scaling, and cp/validation selection.",
        )
    cp = c.real(cp, 0, 1e100, "cp")
    c.integer(max_work, 1, c.MAX_WORK, "max_work")
    return {
        "task": task,
        "min_leaf": min_leaf,
        "min_split": min_split,
        "min_weight_fraction": float(min_weight_fraction),
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "impute": impute,
        "standardize": standardize,
        "cp": cp,
        "select": select,
        "max_work": max_work,
    }


def _plan(n, p, k, settings, query=0):
    c.integer(n, 2, c.MAX_ROWS, "resident rows")
    c.integer(p, 1, 64, "features")
    c.integer(k, 0, 32, "classes")
    c.integer(query, 0, c.MAX_ROWS, "query rows")
    depth, nodes = settings["max_depth"], settings["max_nodes"]
    # Every row occurs at most depth+1 times. Full replay rechecks the exhaustive
    # candidate objective at each node; all path risks and held-out routing are
    # charged even if the eventual tree is small.
    work = (
        64 * n * p * (depth + 1) * (math.ceil(math.log2(max(2, n))) + max(1, k))
        + 64 * nodes**2 * (depth + 1 + math.ceil(math.log2(max(2, nodes))))
        + 64 * (n + query) * nodes * (depth + 1 + max(1, k) + math.ceil(math.log2(max(2, n))))
    )
    if work > settings["max_work"]:
        c.fail(
            "system_budget",
            f"Complete CART grow/replay/pruning/validation work {work} exceeds max_work={settings['max_work']}.",
        )
    storage = (
        n * (p + 12 + max(1, k)) * 128
        + n * (depth + 1) * 48
        + nodes * (max(1, k) + 28) * 128
        + nodes * 512
        + query * (p + k + 6) * 128
    )
    if storage > c.MAX_JSON:
        c.fail(
            "metadata_limit",
            "Complete source, tree/path and predictions exceed 64 MiB portable storage.",
        )
    plan = plan_workspace(
        "weighted CART, full pruning path and semantic replay",
        {
            "numeric source and sorted sufficient-statistic buffers": 64 * n * (p + k + 12),
            "node row partitions and tree summaries": 64 * n * (depth + 1) + nodes * (k + 32) * 128,
            "portable state, tables and complete JSON copies": storage * 4,
            "query and held-out probabilities": (n + query) * (k + 4) * 64,
        },
    ).record()
    return {"estimated_work": work, "estimated_json_bytes": storage, "workspace": plan}


def _outcome(series, task):
    if task == "regression":
        return c.numeric(series)
    _dtype(str(series.dtype), classification=True)
    raw = series.tolist()
    values = []
    for value in raw:
        if hasattr(value, "item") and not isinstance(value, str):
            value = value.item()
        if type(value) not in (int, str) or type(value) is str and (not value or len(value) > 1024):
            c.fail(
                "invalid_response",
                "Classification labels require complete homogeneous integers or bounded strings.",
            )
        values.append(value)
    if len({type(v) for v in values}) != 1:
        c.fail("invalid_response", "Classification labels must have one primitive type.")
    return values


def _dtype(dtype, *, classification=False):
    if type(dtype) is not str or len(dtype) > 128:
        c.fail("invalid_state", "Saved source dtype is invalid.")
    try:
        actual = pd.api.types.pandas_dtype(dtype)
    except (TypeError, ValueError):
        c.fail("invalid_state", "Saved source dtype is unknown.")
    if not classification and (
        not is_numeric_dtype(actual) or is_bool_dtype(actual) or is_complex_dtype(actual)
    ):
        c.fail("invalid_state", "Saved numerical dtype must be real numeric.")
    if classification and str(actual) not in (
        "object",
        "str",
        "string",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    ):
        c.fail("invalid_state", "Saved classification dtype does not represent declared labels.")
    return actual


def _representable(values, dtype, *, classification=False):
    actual = _dtype(dtype, classification=classification)
    try:
        converted = pd.Series(values, dtype=actual)
        expected = (
            _outcome(converted, "classification")
            if classification
            else c.numeric(converted, missing=True)
        )
        c.same(values, expected, "source dtype representability")
    except (ValueError, TypeError, OverflowError) as exc:
        if isinstance(exc, c.AnalysisError):
            raise
        c.fail("invalid_state", f"Saved values do not belong to their declared dtype: {exc}.")


def _transform(raw, weights, train, settings):
    p = len(raw[0])
    matrix = c.tensor([[float("nan") if value is None else value for value in row] for row in raw])
    rows = torch.tensor(train, dtype=torch.int64, device="cpu")
    means, centers, scales, constants = [], [], [], []
    for j in range(p):
        observed = ~torch.isnan(matrix[rows, j])
        if not bool(observed.any()):
            c.fail("invalid_design", "Each feature requires an observed training value.")
        values = matrix[rows, j][observed]
        w = weights[rows][observed]
        unit = float(values.abs().max()) or 1.0
        mean = float(((w / w.sum()) * (values / unit)).sum()) * unit
        means.append(mean)
        missing = torch.isnan(matrix[:, j])
        if bool(missing.any()):
            if settings["impute"] != "mean":
                c.fail(
                    "missing_values",
                    "Feature missingness requires explicit train-only mean imputation.",
                )
            matrix[missing, j] = mean
        training = matrix[rows, j]
        unit = float(training.abs().max()) or 1.0
        w = weights[rows] / weights[rows].sum()
        center = float((w * (training / unit)).sum()) * unit
        variance = float((w * (training / unit - center / unit).square()).sum())
        scale = math.sqrt(variance) * unit
        constant = scale == 0
        if settings["standardize"] == "population":
            centers.append(center)
            scales.append(scale if scale > 0 else 1.0)
            matrix[:, j] = (matrix[:, j] - center) / scales[-1]
        else:
            centers.append(0.0)
            scales.append(1.0)
        constants.append(constant)
    if not bool(torch.isfinite(matrix).all()):
        c.fail("numerical_domain", "Transformed features exceed finite float64 arithmetic.")
    return matrix, {
        "imputation_means": means,
        "centers": centers,
        "scales": scales,
        "training_constant": constants,
        "fit_positions": list(train),
        "scale_denominator": "sum of training case loss weights (population variance)",
    }


def _context(source, settings, positions):
    weights = c.tensor(source["weights"])
    if not bool((weights > 0).all()):
        c.fail("invalid_weights", "Case loss weights must be strictly positive.")
    train = positions["train"]
    if len(train) < 2:
        c.fail("insufficient_sample", "CART requires at least two training rows.")
    x, transform = _transform(source["x"], weights, train, settings)
    wtrain = weights[train]
    normalized_w = weights / wtrain.sum()
    if not bool(torch.isfinite(normalized_w).all()) or not bool((normalized_w > 0).all()):
        c.fail(
            "numerical_domain",
            "Case weight normalization loses a positive weight; rescale or reduce its dynamic range.",
        )
    if settings["task"] == "classification":
        classes = sorted({source["y"][i] for i in train})
        c.integer(len(classes), 2, 32, "training classes")
        mapping = {value: j for j, value in enumerate(classes)}
        y = torch.tensor(
            [mapping.get(value, -1) for value in source["y"]], dtype=torch.int64, device="cpu"
        )
        y_scale, y_center = 1.0, 0.0
    else:
        classes = []
        raw_y = c.tensor(source["y"])
        y_scale = float(raw_y[train].abs().max()) or 1.0
        y_center = float((normalized_w[train] * (raw_y[train] / y_scale)).sum())
        y = raw_y / y_scale - y_center
    return {
        "x": x,
        "y": y,
        "weights": weights,
        "w": normalized_w,
        "classes": classes,
        "k": len(classes),
        "y_scale": y_scale,
        "y_center": y_center,
        "raw_y": raw_y if not classes else None,
        "transform": transform,
        "settings": settings,
    }


def _stats(context, positions):
    y, w = context["y"][positions], context["w"][positions]
    mass = float(w.sum())
    if context["k"]:
        counts = torch.zeros(context["k"], dtype=c.FLOAT, device="cpu")
        counts.index_add_(0, y, w)
        probabilities = counts / counts.sum()
        predicted = int(counts.argmax())
        return {
            "prediction": None,
            "probabilities": probabilities.tolist(),
            "predicted_class": predicted,
            "risk": float(w[y != predicted].sum()),
            "impurity": mass * float((probabilities * (1 - probabilities)).sum()),
        }
    constant = bool((y == y[0]).all())
    mean = float(y[0]) if constant else float(((w / w.sum()) * y).sum())
    risk = 0.0 if constant else float((w * (y - mean).square()).sum())
    return {
        "prediction": (mean + context["y_center"]) * context["y_scale"],
        "probabilities": None,
        "predicted_class": None,
        "risk": risk,
        "impurity": risk,
    }


def _best_split(context, positions, stats):
    settings = context["settings"]
    n = len(positions)
    if n < 2 * settings["min_leaf"] or stats["impurity"] <= 0:
        return None
    index = torch.tensor(positions, dtype=torch.int64, device="cpu")
    best = None
    parent = stats["impurity"]
    for j in range(context["x"].shape[1]):
        order = torch.argsort(context["x"][index, j], stable=True)
        rows = index[order]
        x, w, y = context["x"][rows, j], context["w"][rows], context["y"][rows]
        boundaries = torch.nonzero(x[:-1] < x[1:], as_tuple=False).flatten()
        if not len(boundaries):
            continue
        left_mass = w.cumsum(0)[:-1]
        right_mass = float(w.sum()) - left_mass
        admissible = (
            (boundaries + 1 >= settings["min_leaf"])
            & (n - boundaries - 1 >= settings["min_leaf"])
            & (left_mass[boundaries] >= settings["min_weight_fraction"])
            & (right_mass[boundaries] >= settings["min_weight_fraction"])
            & (left_mass[boundaries] > 0)
            & (right_mass[boundaries] > 0)
        )
        boundaries = boundaries[admissible]
        if not len(boundaries):
            continue
        if context["k"]:
            counts = (
                torch.nn.functional.one_hot(y, num_classes=context["k"]).to(c.FLOAT) * w[:, None]
            )
            cumulative = counts.cumsum(0)
            a = cumulative[boundaries]
            b = cumulative[-1] - a
            loss = (
                left_mass[boundaries]
                - a.square().sum(1) / left_mass[boundaries]
                + right_mass[boundaries]
                - b.square().sum(1) / right_mass[boundaries]
            )
        else:
            first = (w * y).cumsum(0)
            second = (w * y.square()).cumsum(0)
            a = second[boundaries] - first[boundaries].square() / left_mass[boundaries]
            b = (
                second[-1]
                - second[boundaries]
                - (first[-1] - first[boundaries]).square() / right_mass[boundaries]
            )
            rounding = (
                256
                * c.EPS
                * (
                    second[-1].abs()
                    + second[boundaries].abs()
                    + first[-1].abs().square() / right_mass[boundaries]
                )
            )
            if bool((a < -rounding).any()) or bool((b < -rounding).any()):
                c.fail(
                    "numerical_failure",
                    "Split sufficient statistics exceed their rounding envelope.",
                )
            loss = a.clamp_min(0) + b.clamp_min(0)
        gain = parent - loss
        # Positive gain must exceed its arithmetic envelope. Within-envelope
        # comparisons are deterministic ties ordered by feature then threshold.
        maximum = float(gain.max())
        tolerance = 256 * c.EPS * max(abs(parent), abs(maximum))
        if maximum <= tolerance:
            continue
        candidates = torch.nonzero(gain >= maximum - tolerance, as_tuple=False).flatten()
        q = int(boundaries[int(candidates[0])])
        left, right = float(x[q]), float(x[q + 1])
        threshold = left / 2 + right / 2
        if not left <= threshold < right:
            threshold = left
        selected_gain = float(gain[int(candidates[0])])
        proposal = {
            "feature": j,
            "threshold": threshold,
            "split_gain": selected_gain,
            "left_positions": sorted(rows[: q + 1].tolist()),
            "right_positions": sorted(rows[q + 1 :].tolist()),
        }
        if best is None or selected_gain > best["split_gain"] + 256 * c.EPS * max(
            abs(selected_gain), abs(best["split_gain"])
        ):
            best = proposal
    return best


def _node(context, positions, node_id, depth):
    stats = _stats(context, positions)
    reason = (
        "max_depth"
        if depth >= context["settings"]["max_depth"]
        else "pure"
        if stats["impurity"] <= 0
        else "min_split"
        if len(positions) < context["settings"]["min_split"]
        else "min_leaf"
        if len(positions) < 2 * context["settings"]["min_leaf"]
        else None
    )
    split = None if reason else _best_split(context, positions, stats)
    if reason is None and split is None:
        reason = "no_positive_gain"
    node = {
        "id": node_id,
        "depth": depth,
        "positions": list(positions),
        "n": len(positions),
        "weight_sum": float(context["weights"][positions].sum()),
        **stats,
        "feature": split["feature"] if split else None,
        "threshold": split["threshold"] if split else None,
        "split_gain": split["split_gain"] if split else 0.0,
        "left": 2 * node_id if split else None,
        "right": 2 * node_id + 1 if split else None,
        "stop_reason": reason,
    }
    return node, split


def _grow(context, positions):
    pending = [(1, 0, list(positions))]
    nodes = []
    while pending:
        node_id, depth, rows = pending.pop()
        node, split = _node(context, rows, node_id, depth)
        if split is not None:
            if len(nodes) + len(pending) + 3 > context["settings"]["max_nodes"]:
                c.fail(
                    "tree_budget", "Maximal CART exceeds max_nodes; no truncated tree is returned."
                )
            pending.extend(
                (
                    (node["right"], depth + 1, split["right_positions"]),
                    (node["left"], depth + 1, split["left_positions"]),
                )
            )
        nodes.append(node)
    return sorted(nodes, key=lambda node: node["id"])


def _replay_nodes(context, positions, saved):
    """Recheck cached topology/candidates without calling the grow routine."""
    expected_positions = {1: (0, list(positions))}
    nodes = {node["id"]: node for node in saved}
    if len(nodes) != len(saved):
        c.fail("invalid_state", "Tree has duplicate node identities.")
    for node in sorted(saved, key=lambda node: node["id"]):
        node_id = node["id"]
        if node_id not in expected_positions:
            c.fail("state_replay", "Cached node has no admitted parent partition.")
        depth, rows = expected_positions[node_id]
        expected, split = _node(context, rows, node_id, depth)
        c.same(node, expected, f"tree node {node_id}")
        if split is not None:
            expected_positions[node_id * 2] = (depth + 1, split["left_positions"])
            expected_positions[node_id * 2 + 1] = (depth + 1, split["right_positions"])
    if set(nodes) != set(expected_positions):
        c.fail("state_replay", "Cached maximal tree omits required child partitions.")
    return nodes


def _leaves(nodes, pruned):
    leaves, active = [], []
    pending = [1]
    while pending:
        node_id = pending.pop()
        node = nodes[node_id]
        if node["left"] is None or node_id in pruned:
            leaves.append(node_id)
        else:
            active.append(node_id)
            pending.extend((node["right"], node["left"]))
    return sorted(leaves), sorted(active)


def _pruning(nodes):
    root_risk = nodes[1]["risk"]
    pruned = set()
    leaves, active = _leaves(nodes, pruned)
    path = [
        {
            "stage": 0,
            "alpha": 0.0,
            "cp": 0.0,
            "prune_nodes": [],
            "leaf_count": len(leaves),
            "risk": sum(nodes[j]["risk"] for j in leaves),
        }
    ]
    while active:
        summaries = {}
        leaf_set = set(leaves)
        for node_id in sorted((*leaves, *active), reverse=True):
            node = nodes[node_id]
            if node_id in leaf_set:
                summaries[node_id] = (node["risk"], 1)
            else:
                a, b = summaries[node["left"]], summaries[node["right"]]
                summaries[node_id] = (a[0] + b[0], a[1] + b[1])
        links = {}
        for j in active:
            difference = nodes[j]["risk"] - summaries[j][0]
            rounding = 256 * c.EPS * (abs(nodes[j]["risk"]) + abs(summaries[j][0]))
            if difference < -rounding:
                c.fail(
                    "numerical_failure",
                    "Subtree risk exceeds parent risk outside its rounding envelope.",
                )
            links[j] = max(0.0, difference) / (summaries[j][1] - 1)
        alpha = min(links.values())
        tied = {
            j
            for j in active
            if abs(links[j] - alpha) <= 256 * c.EPS * max(abs(alpha), abs(links[j]))
        }
        # Simultaneous weak links retain the ancestor when an ancestor/descendant
        # pair is tied; descendants become unreachable without separate edits.
        edits = []
        for j in sorted(tied):
            parent = j // 2
            covered = False
            while parent:
                if parent in tied:
                    covered = True
                    break
                parent //= 2
            if not covered:
                edits.append(j)
        pruned.update(edits)
        leaves, active = _leaves(nodes, pruned)
        path.append(
            {
                "stage": len(path),
                "alpha": alpha,
                "cp": alpha / root_risk if root_risk > 0 else 0.0,
                "prune_nodes": edits,
                "leaf_count": len(leaves),
                "risk": sum(nodes[j]["risk"] for j in leaves),
            }
        )
    return path


def _stage_pruned(path, stage):
    return {j for point in path[1 : stage + 1] for j in point["prune_nodes"]}


def _predict(context, nodes, path, stage, x):
    pruned = _stage_pruned(path, stage)
    leaves = []
    for row in x:
        node_id = 1
        while nodes[node_id]["left"] is not None and node_id not in pruned:
            node = nodes[node_id]
            node_id = (
                node["left"] if float(row[node["feature"]]) <= node["threshold"] else node["right"]
            )
        leaves.append(node_id)
    if context["k"]:
        probabilities = c.tensor([nodes[j]["probabilities"] for j in leaves])
        prediction = [context["classes"][int(j)] for j in probabilities.argmax(1).tolist()]
        return {
            "leaf_ids": leaves,
            "predictions": prediction,
            "probabilities": probabilities.tolist(),
        }
    return {
        "leaf_ids": leaves,
        "predictions": [nodes[j]["prediction"] for j in leaves],
        "probabilities": None,
    }


def _metric(context, positions, prediction, *, require_known=True):
    if not positions:
        return None
    y, weights = context["y"][positions], context["weights"][positions]
    if context["k"]:
        if bool((y < 0).any()):
            if require_known:
                c.fail(
                    "unknown_class",
                    "Evaluation contains a class absent from training; no class map is refitted.",
                )
            return {"available": False, "reason": "class absent from training", "n": len(positions)}
        return metrics.classification(
            y, c.tensor(prediction["probabilities"]), weights, context["k"]
        )
    raw_y = context["raw_y"][positions]
    return metrics.regression(raw_y, c.tensor(prediction["predictions"]), weights)


def _finish(context, nodes, positions):
    path = _pruning(nodes)
    validation = []
    for point in path:
        rows = positions["validation"]
        prediction = (
            _predict(context, nodes, path, point["stage"], context["x"][rows]) if rows else None
        )
        metric = (
            _metric(
                context,
                rows,
                prediction,
                require_known=context["settings"]["select"] == "validation",
            )
            if rows
            else None
        )
        if metric is None or metric.get("available") is False:
            loss = None
        elif context["k"]:
            loss = 1 - metric["accuracy"]
        else:
            loss = metric["rmse"]
        validation.append(
            {
                "stage": point["stage"],
                "loss": loss,
                "metric": "weighted misclassification" if context["k"] else "weighted RMSE",
            }
        )
    if context["settings"]["select"] == "validation":
        if not positions["validation"]:
            c.fail(
                "invalid_split", "Validation subtree selection requires nonempty validation rows."
            )
        best = min(v["loss"] for v in validation)
        tied = [
            v["stage"]
            for v in validation
            if abs(v["loss"] - best) <= 256 * c.EPS * max(abs(best), abs(v["loss"]))
        ]
        selected = max(tied, key=lambda i: (-path[i]["leaf_count"], i))
    else:
        selected = max(
            i for i, point in enumerate(path) if point["cp"] <= context["settings"]["cp"]
        )
    all_prediction = _predict(context, nodes, path, selected, context["x"])
    report = {}
    for role, rows in positions.items():
        prediction = {
            "leaf_ids": [all_prediction["leaf_ids"][i] for i in rows],
            "predictions": [all_prediction["predictions"][i] for i in rows],
            "probabilities": [all_prediction["probabilities"][i] for i in rows]
            if context["k"]
            else None,
        }
        report[role] = _metric(context, rows, prediction, require_known=False)
    importance = [0.0] * context["x"].shape[1]
    pruned = _stage_pruned(path, selected)
    _, active = _leaves(nodes, pruned)
    for node_id in active:
        node = nodes[node_id]
        # Recomputed child impurities avoid treating prefix rounding as importance.
        importance[node["feature"]] += max(
            0.0,
            node["impurity"] - nodes[node["left"]]["impurity"] - nodes[node["right"]]["impurity"],
        )
    total = sum(importance)
    return {
        "path": path,
        "validation": validation,
        "selected_stage": selected,
        "prediction": all_prediction,
        "metrics": report,
        "importance": {
            "raw": importance,
            "normalized": [v / total if total else 0.0 for v in importance],
            "estimand": "training weighted Gini reduction"
            if context["k"]
            else "training normalized weighted SSE reduction",
            "uncertainty_available": False,
        },
    }


def _assemble(record):
    derived = record["derived"]
    index = c.restore_index(record["source"]["index"], record["n"])
    prediction = derived["prediction"]
    tables = {
        "predictions": table(
            {
                "role": record["split"]["roles"],
                "leaf_id": prediction["leaf_ids"],
                "prediction": prediction["predictions"],
            },
            index=index,
        ),
        "nodes": table(
            [
                {k: v for k, v in node.items() if k not in ("positions", "probabilities")}
                for node in record["nodes"]
            ]
        ),
        "pruning": table(
            [{k: v for k, v in point.items() if k != "prune_nodes"} for point in derived["path"]]
        ),
        "validation": table(derived["validation"]),
        "importance": table(
            {
                "feature": record["features"],
                "importance": derived["importance"]["raw"],
                "normalized": derived["importance"]["normalized"],
            }
        ),
    }
    if prediction["probabilities"] is not None:
        tables["probabilities"] = table(
            prediction["probabilities"], columns=[str(v) for v in record["classes"]], index=index
        )
    for key in ("predictions", "probabilities"):
        if key in tables:
            tables[key].index = index
    tables.update(_metric_tables(derived["metrics"], record["classes"]))
    return CartResult(
        tables,
        title="Weighted CART prediction",
        schema=RESULT_SCHEMA,
        state=record,
        selected_stage=derived["selected_stage"],
        leaf_count=derived["path"][derived["selected_stage"]]["leaf_count"],
        metrics=derived["metrics"],
        sample={
            "n_original": record["n"],
            "n": record["n"],
            "dropped_rows": 0,
            "partition_counts": {
                role: len(rows) for role, rows in record["split"]["positions"].items()
            },
        },
        uncertainty={
            "available": False,
            "reason": "adaptive CART uncertainty has no implemented valid sampling procedure",
        },
    )


def _metric_tables(report, classes):
    scalar, confusion, classwise, calibration = [], [], [], []
    for role, metric in report.items():
        if metric is None:
            continue
        scalar.append(
            {
                "role": role,
                **{
                    key: value
                    for key, value in metric.items()
                    if not isinstance(value, (list, tuple, Mapping))
                },
            }
        )
        if classes and "confusion" in metric:
            for i, row in enumerate(metric["confusion"]):
                for j, mass in enumerate(row):
                    confusion.append(
                        {
                            "role": role,
                            "observed": classes[i],
                            "predicted": classes[j],
                            "weight": mass,
                        }
                    )
            for value in metric["classwise"]:
                classwise.append({"role": role, "class": classes[value["class_index"]], **value})
            calibration.extend({"role": role, **value} for value in metric["calibration"])
    tables = {"metrics": table(scalar)}
    for name, values in (
        ("confusion", confusion),
        ("classwise", classwise),
        ("calibration", calibration),
    ):
        if values:
            tables[name] = table(values)
    return tables


class CartState(c.Transport):
    @classmethod
    def replay(cls, payload):
        return replay(payload)


class CartResult(TableSet):
    @c.cpu_call
    def to_json(self, *, indent=None):
        restored = cart_restore(self)
        return c.json_output(restored.attrs, indent=indent)

    def __deepcopy__(self, memo=None):
        return cart_restore(self)

    def copy(self):
        return cart_restore(self)


class CartPrediction(TableSet):
    @c.cpu_call
    def to_json(self, *, indent=None):
        restored = cart_prediction_restore(self)
        return c.json_output(restored.attrs, indent=indent)

    def __deepcopy__(self, memo=None):
        return cart_prediction_restore(self)

    def copy(self):
        return cart_prediction_restore(self)


class CartQueryState(c.Transport):
    @classmethod
    def replay(cls, payload):
        return cart_prediction_restore(payload).attrs


def _cache_admission(record):
    n, p = record["n"], len(record["features"])
    classification = record["settings"]["task"] == "classification"
    if (classification and not 2 <= len(record["classes"]) <= 32) or (
        not classification and record["classes"]
    ):
        c.fail("invalid_state", "Saved class-map shape disagrees with the declared task.")
    c.keys(record["source"], ("index", "x", "y", "weights", "dtypes"), "saved source")
    source = record["source"]
    c.sequence(source["x"], n, "source features")
    for row in source["x"]:
        c.numeric_cache(row, p, missing=True)
    c.numeric_cache(source["weights"], n)
    c.sequence(source["y"], n, "source response")
    if record["settings"]["task"] == "regression":
        c.numeric_cache(source["y"], n)
    elif (
        any(type(v) not in (int, str) for v in source["y"])
        or len({type(v) for v in source["y"]}) != 1
    ):
        c.fail("invalid_state", "Saved classes have invalid primitive types.")
    c.keys(source["dtypes"], ("features", "outcome", "weights"), "source dtypes")
    c.sequence(source["dtypes"]["features"], p, "feature dtypes")
    for dtype in source["dtypes"]["features"]:
        _dtype(dtype)
    _dtype(
        source["dtypes"]["outcome"], classification=record["settings"]["task"] == "classification"
    )
    if record["weight_name"] is not None:
        _dtype(source["dtypes"]["weights"])
    elif source["dtypes"]["weights"] is not None:
        c.fail("invalid_state", "Unweighted source has an unexpected weight dtype.")
    if (
        not isinstance(record["nodes"], (list, tuple))
        or not 1 <= len(record["nodes"]) <= record["settings"]["max_nodes"]
    ):
        c.fail("invalid_state", "Saved tree exceeds its admitted node envelope.")
    for node in record["nodes"]:
        c.keys(node, NODE_KEYS, "tree node")
        c.integer(node["id"], 1, 2**21 - 1, "node id")
        c.integer(node["depth"], 0, record["settings"]["max_depth"], "node depth")
        c.integer(node["n"], 1, n, "node count")
        c.sequence(node["positions"], node["n"], "node positions")
        if any(type(v) is not int or not 0 <= v < n for v in node["positions"]):
            c.fail("invalid_state", "Node row positions have invalid primitives.")
        for key in ("weight_sum", "risk", "impurity", "split_gain"):
            if type(node[key]) is not float or not math.isfinite(node[key]) or node[key] < 0:
                c.fail("invalid_state", "Node loss/cache has invalid numerical primitives.")
        if node["feature"] is not None:
            c.integer(node["feature"], 0, p - 1, "node feature")
            if type(node["threshold"]) is not float or not math.isfinite(node["threshold"]):
                c.fail("invalid_state", "Node threshold is invalid.")
            for key in ("left", "right"):
                c.integer(node[key], 1, 2**21 - 1, key)
            if node["stop_reason"] is not None:
                c.fail("invalid_state", "An internal node cannot have a terminal stop reason.")
        else:
            if any(node[key] is not None for key in ("threshold", "left", "right")):
                c.fail("invalid_state", "A terminal node has no threshold or child identities.")
            if node["split_gain"] != 0.0:
                c.fail("invalid_state", "A terminal node has exact zero split gain.")
            if type(node["stop_reason"]) is not str or node["stop_reason"] not in (
                "max_depth",
                "pure",
                "min_split",
                "min_leaf",
                "no_positive_gain",
            ):
                c.fail(
                    "invalid_state", "A terminal node requires a declared primitive stop reason."
                )
        if classification:
            if node["prediction"] is not None:
                c.fail(
                    "invalid_state", "Classification nodes have no numerical response prediction."
                )
            c.numeric_cache(node["probabilities"], len(record["classes"]))
            if any(type(v) is not float or not 0 <= v <= 1 for v in node["probabilities"]):
                c.fail("invalid_state", "Class probabilities require finite float fractions.")
            c.integer(node["predicted_class"], 0, len(record["classes"]) - 1, "predicted class")
        else:
            if node["probabilities"] is not None or node["predicted_class"] is not None:
                c.fail(
                    "invalid_state", "Regression nodes have no class probability/prediction cache."
                )
            if type(node["prediction"]) is not float or not math.isfinite(node["prediction"]):
                c.fail("invalid_state", "Node response prediction is invalid.")
    c.keys(
        record["derived"],
        ("path", "validation", "selected_stage", "prediction", "metrics", "importance"),
        "derived tree cache",
    )
    derived = record["derived"]
    if not isinstance(derived["path"], (list, tuple)) or not 1 <= len(derived["path"]) <= len(
        record["nodes"]
    ):
        c.fail("invalid_state", "Pruning path has an invalid shape.")
    c.sequence(derived["validation"], len(derived["path"]), "validation path")
    c.integer(derived["selected_stage"], 0, len(derived["path"]) - 1, "selected stage")
    c.keys(
        derived["prediction"], ("leaf_ids", "predictions", "probabilities"), "cached predictions"
    )
    for key in ("leaf_ids", "predictions"):
        c.sequence(derived["prediction"][key], n, key)
    if record["classes"]:
        c.sequence(derived["prediction"]["probabilities"], n, "cached probabilities")
        for row in derived["prediction"]["probabilities"]:
            c.numeric_cache(row, len(record["classes"]))
    c.keys(
        derived["importance"],
        ("raw", "normalized", "estimand", "uncertainty_available"),
        "importance cache",
    )
    for key in ("raw", "normalized"):
        c.numeric_cache(derived["importance"][key], p)
    c.keys(
        record["transform"],
        (
            "imputation_means",
            "centers",
            "scales",
            "training_constant",
            "fit_positions",
            "scale_denominator",
        ),
        "transformation cache",
    )
    for key in ("imputation_means", "centers", "scales"):
        c.numeric_cache(record["transform"][key], p)
    c.sequence(record["transform"]["training_constant"], p, "training constant flags")
    if any(type(v) is not bool for v in record["transform"]["training_constant"]):
        c.fail("invalid_state", "Training constant flags require actual booleans.")
    rows = record["transform"]["fit_positions"]
    if (
        not isinstance(rows, (list, tuple))
        or not 2 <= len(rows) <= n
        or any(type(v) is not int or not 0 <= v < n for v in rows)
    ):
        c.fail("invalid_state", "Transformation source positions are malformed.")
    c.keys(
        record["normalization"],
        ("weight_sum", "outcome_scale", "outcome_center_scaled", "risk_convention"),
        "loss normalization",
    )
    for key in ("weight_sum", "outcome_scale", "outcome_center_scaled"):
        if type(record["normalization"][key]) is not float or not math.isfinite(
            record["normalization"][key]
        ):
            c.fail("invalid_state", "Loss normalization has invalid primitives.")
    for j, point in enumerate(derived["path"]):
        c.keys(
            point, ("stage", "alpha", "cp", "prune_nodes", "leaf_count", "risk"), "pruning point"
        )
        c.integer(point["stage"], 0, len(record["nodes"]), "pruning stage")
        c.integer(point["leaf_count"], 1, len(record["nodes"]), "pruning leaves")
        for key in ("alpha", "cp", "risk"):
            if type(point[key]) is not float or not math.isfinite(point[key]) or point[key] < 0:
                c.fail("invalid_state", "Pruning loss has invalid numerical primitives.")
        if (
            not isinstance(point["prune_nodes"], (list, tuple))
            or len(point["prune_nodes"]) > len(record["nodes"])
            or any(type(v) is not int for v in point["prune_nodes"])
        ):
            c.fail("invalid_state", "Pruning edits have malformed identities.")
        value = derived["validation"][j]
        c.keys(value, ("stage", "loss", "metric"), "validation point")
        c.integer(value["stage"], 0, len(record["nodes"]), "validation stage")
        if value["loss"] is not None and (
            type(value["loss"]) is not float or not math.isfinite(value["loss"])
        ):
            c.fail("invalid_state", "Validation loss is malformed.")
    if any(type(v) is not int for v in derived["prediction"]["leaf_ids"]):
        c.fail("invalid_state", "Prediction leaf identities require actual integers.")
    if not record["classes"]:
        c.numeric_cache(derived["prediction"]["predictions"], n)
        if derived["prediction"]["probabilities"] is not None:
            c.fail("invalid_state", "Regression has no class probabilities.")
    c.keys(derived["metrics"], ("train", "validation", "test"), "split metrics")
    for metric in derived["metrics"].values():
        metrics.admission(metric, len(record["classes"]))
        if metric is not None and not isinstance(metric, Mapping):
            c.fail("invalid_state", "Split metrics must be complete dictionaries or unavailable.")
        if record["classes"] and metric is not None and "confusion" in metric:
            c.sequence(metric["confusion"], len(record["classes"]), "confusion rows")
            for row in metric["confusion"]:
                c.numeric_cache(row, len(record["classes"]), bound=1e308)
            c.sequence(metric.get("classwise"), len(record["classes"]), "classwise metrics")
            c.sequence(metric.get("calibration"), 10, "reliability bins")


def _header(state):
    """Strict scalar/shape header before work plans or nested indexing."""
    c.keys(
        state,
        (
            "schema",
            "n",
            "features",
            "outcome",
            "weight_name",
            "settings",
            "split",
            "source",
            "classes",
            "transform",
            "normalization",
            "nodes",
            "derived",
            "digest",
        ),
        "CART state",
    )
    if state["schema"] != SCHEMA:
        c.fail("invalid_state", "Unsupported CART state schema.")
    c.columns(state["features"])
    c.columns([state["outcome"]])
    if state["outcome"] in state["features"]:
        c.fail("invalid_spec", "Outcome cannot be a predictor.")
    c.keys(
        state["settings"],
        (
            "task",
            "min_leaf",
            "min_split",
            "min_weight_fraction",
            "max_depth",
            "max_nodes",
            "impute",
            "standardize",
            "cp",
            "select",
            "max_work",
        ),
        "CART settings",
    )
    settings = _settings(**state["settings"])
    c.same(state["settings"], settings, "CART settings")
    if not isinstance(state["classes"], (list, tuple)):
        c.fail("invalid_state", "Saved training class map is invalid.")
    k = len(state["classes"])
    if (
        any(type(value) not in (str, int) for value in state["classes"])
        or k
        and len({type(value) for value in state["classes"]}) != 1
    ):
        c.fail("invalid_state", "Saved class map requires homogeneous primitive labels.")
    if state["weight_name"] is not None:
        c.columns([state["weight_name"]])
        if state["weight_name"] in (*state["features"], state["outcome"]):
            c.fail("invalid_state", "Saved loss weight conflicts with predictor/outcome roles.")
    return settings, k


@c.cpu_call
def replay(state):
    state = c.load(state)
    settings, k = _header(state)
    _plan(state["n"], len(state["features"]), k, settings)
    _cache_admission(state)
    c.unseal(state)
    source = state["source"]
    c.restore_index(source["index"], state["n"])
    split = split_replay(state["split"])
    c.same(split["n"], state["n"], "source/split rows")
    c.same(split["index"], source["index"], "source/split identity")
    for j, name in enumerate(state["features"]):
        _representable([row[j] for row in source["x"]], source["dtypes"]["features"][j])
    _representable(
        source["y"],
        source["dtypes"]["outcome"],
        classification=settings["task"] == "classification",
    )
    if state["weight_name"] is not None:
        c.columns([state["weight_name"]])
        _representable(source["weights"], source["dtypes"]["weights"])
    elif any(v != 1.0 for v in source["weights"]):
        c.fail("state_replay", "Unweighted source must retain exact unit weights.")
    context = _context(source, settings, split["positions"])
    c.same(state["classes"], context["classes"], "training class map")
    c.same(state["transform"], context["transform"], "train-only transformation")
    c.same(
        state["normalization"],
        {
            "weight_sum": float(context["weights"][split["positions"]["train"]].sum()),
            "outcome_scale": context["y_scale"],
            "outcome_center_scaled": context["y_center"],
            "risk_convention": "training weight-normalized misclassification"
            if k
            else "training weight-normalized SSE in centered response units",
        },
        "loss normalization",
    )
    nodes = _replay_nodes(context, split["positions"]["train"], state["nodes"])
    derived = _finish(context, nodes, split["positions"])
    c.same(state["derived"], derived, "pruning/selection/prediction/metrics")
    return state


@c.cpu_call
def cart(
    data,
    *,
    outcome,
    features,
    split=None,
    task="regression",
    weights=None,
    min_leaf=5,
    min_split=None,
    min_weight_fraction=0.0,
    max_depth=6,
    max_nodes=127,
    impute=None,
    standardize="identity",
    cp=0.0,
    select="cp",
    max_work=c.DEFAULT_WORK,
):
    """Fit exhaustive weighted continuous CART on the declared training rows.

    A fixed cp includes all pruning events with event_cp <= cp. Validation
    selection minimizes weighted RMSE/misclassification and ties choose fewer
    leaves. Test labels never choose or refit the tree. Unknown test classes are
    retained with unavailable scoring; training class maps remain fixed.
    """
    features = c.columns(features)
    c.columns([outcome])
    if outcome in features:
        c.fail("invalid_spec", "Outcome cannot be a predictor.")
    names = [*features, outcome]
    if weights is not None:
        c.columns([weights])
        if weights in names:
            c.fail("invalid_spec", "A case loss weight column cannot also be a feature/outcome.")
        names.append(weights)
    raw_split = c.load(split) if split is not None else None
    if raw_split is not None:
        for role_name in ("group_name", "time_name"):
            name = raw_split.get(role_name) if isinstance(raw_split, Mapping) else None
            if name is not None:
                c.columns([name])
                if name not in names:
                    names.append(name)
    n = c.probe(data, names)
    settings = _settings(
        task=task,
        min_leaf=min_leaf,
        min_split=min_split,
        min_weight_fraction=min_weight_fraction,
        max_depth=max_depth,
        max_nodes=max_nodes,
        impute=impute,
        standardize=standardize,
        cp=cp,
        select=select,
        max_work=max_work,
    )
    _plan(n, len(features), 32 if task == "classification" else 0, settings)
    frame, index = c.resident(data, names, n)
    split_state = (
        prediction_split(frame, max_work=max_work)
        if split is None
        else SplitState(payload=split_replay(split))
    )
    c.same(split_state.payload["index"], index, "split/source row identity")
    c.same(split_state.payload["n"], n, "split/source row count")
    for name_field, value_field in (("group_name", "groups"), ("time_name", "times")):
        name = split_state.payload[name_field]
        if name is not None:
            c.same(
                split_state.payload[value_field],
                [c.label(v) for v in frame[name]],
                "split/current source role identity",
            )
    feature_values = [c.numeric(frame[v], missing=True) for v in features]
    source = {
        "index": index,
        "x": [list(row) for row in zip(*feature_values)],
        "y": _outcome(frame[outcome], task),
        "weights": c.numeric(frame[weights]) if weights is not None else [1.0] * n,
        "dtypes": {
            "features": [str(frame[v].dtype) for v in features],
            "outcome": str(frame[outcome].dtype),
            "weights": str(frame[weights].dtype) if weights is not None else None,
        },
    }
    context = _context(source, settings, split_state.payload["positions"])
    _plan(n, len(features), context["k"], settings)
    nodes = _grow(context, split_state.payload["positions"]["train"])
    derived = _finish(
        context, {node["id"]: node for node in nodes}, split_state.payload["positions"]
    )
    record = c.seal(
        {
            "schema": SCHEMA,
            "n": n,
            "features": features,
            "outcome": outcome,
            "weight_name": weights,
            "settings": settings,
            "split": split_state.payload,
            "source": source,
            "classes": context["classes"],
            "transform": context["transform"],
            "normalization": {
                "weight_sum": float(
                    context["weights"][split_state.payload["positions"]["train"]].sum()
                ),
                "outcome_scale": context["y_scale"],
                "outcome_center_scaled": context["y_center"],
                "risk_convention": "training weight-normalized misclassification"
                if context["k"]
                else "training weight-normalized SSE in centered response units",
            },
            "nodes": nodes,
            "derived": derived,
        }
    )
    return _assemble(record)


@c.cpu_call
def cart_restore(saved):
    if isinstance(saved, CartResult):
        saved = saved.attrs
    saved = c.load(saved)
    if isinstance(saved, Mapping) and saved.get("schema") == RESULT_SCHEMA:
        c.keys(
            saved,
            ("schema", "state", "selected_stage", "leaf_count", "metrics", "sample", "uncertainty"),
            "CART result",
        )
        record = replay(saved["state"])
        result = _assemble(record)
        c.same(saved, result.attrs, "public CART result")
        return result
    return _assemble(replay(saved))


@c.cpu_call
def cart_predict(saved, data, *, outcome=None, weights=None):
    """Predict indexed resident rows; optional scoring uses the frozen class map."""
    # Probe and charge both full model replay and query before replay/numeric work.
    raw = c.load(saved.attrs if isinstance(saved, CartResult) else saved)
    record = raw.get("state", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(record, Mapping) or not isinstance(record.get("features"), (list, tuple)):
        c.fail("invalid_state", "A complete CART state is required.")
    settings, k = _header(record)
    names = c.columns(record["features"])
    if weights is not None and outcome is None:
        c.fail("invalid_spec", "Scoring weights require an explicit scoring outcome.")
    if outcome is not None and outcome == weights:
        c.fail("invalid_spec", "Scoring outcome and loss weight must be distinct.")
    for name in (outcome, weights):
        if name is not None:
            c.columns([name])
            if name in names:
                c.fail("invalid_spec", "Scoring response/weight cannot be a predictor.")
            names.append(name)
    query = c.probe(data, names)
    c.integer(query, 1, c.MAX_ROWS, "query rows")
    _plan(
        record["n"],
        len(record["features"]),
        k,
        settings,
        query=query,
    )
    restored = cart_restore(raw)
    record = restored.attrs["state"]
    frame, index = c.resident(data, names, query)
    transform = record["transform"]
    values = [c.numeric(frame[v], missing=True) for v in record["features"]]
    raw_x = [list(row) for row in zip(*values)]
    x = c.tensor([[float("nan") if v is None else v for v in row] for row in raw_x])
    for j in range(len(record["features"])):
        missing = torch.isnan(x[:, j])
        if bool(missing.any()):
            if record["settings"]["impute"] != "mean":
                c.fail("missing_values", "Saved model has no feature missing-value adapter.")
            x[missing, j] = transform["imputation_means"][j]
        x[:, j] = (x[:, j] - transform["centers"][j]) / transform["scales"][j]
    if not bool(torch.isfinite(x).all()):
        c.fail("numerical_domain", "Query transformation exceeds finite float64 arithmetic.")
    context = {"k": len(record["classes"]), "classes": record["classes"]}
    prediction = _predict(
        context,
        {node["id"]: node for node in record["nodes"]},
        record["derived"]["path"],
        record["derived"]["selected_stage"],
        x,
    )
    score = None
    y_values = weight_values = None
    if outcome is not None:
        y_values = _outcome(frame[outcome], record["settings"]["task"])
        weight_values = c.numeric(frame[weights]) if weights is not None else [1.0] * query
        w = c.tensor(weight_values)
        if not bool((w > 0).all()):
            c.fail("invalid_weights", "Scoring case loss weights must be strictly positive.")
        if context["k"]:
            mapping = {v: j for j, v in enumerate(context["classes"])}
            if any(v not in mapping for v in y_values):
                c.fail(
                    "unknown_class",
                    "Scoring class is absent from training; saved class map is unchanged.",
                )
            y = torch.tensor([mapping[v] for v in y_values], dtype=torch.int64, device="cpu")
            score = metrics.classification(
                y, c.tensor(prediction["probabilities"]), w, context["k"]
            )
        else:
            score = metrics.regression(c.tensor(y_values), c.tensor(prediction["predictions"]), w)
    tables = {
        "predictions": table(
            {"leaf_id": prediction["leaf_ids"], "prediction": prediction["predictions"]},
            index=c.restore_index(index, query),
        )
    }
    if context["k"]:
        tables["probabilities"] = table(
            prediction["probabilities"],
            columns=[str(v) for v in context["classes"]],
            index=c.restore_index(index, query),
        )
    query_index = c.restore_index(index, query)
    for key in tables:
        tables[key].index = query_index
    if score is not None:
        tables.update(_metric_tables({"query": score}, context["classes"]))
    attrs = c.seal(
        {
            "schema": "openecon.supervised.cart-query.v1",
            "model": record,
            "source_digest": record["digest"],
            "source": {
                "n": query,
                "index": index,
                "x": raw_x,
                "y": y_values,
                "weights": weight_values,
                "outcome_name": outcome,
                "weight_name": weights,
                "dtypes": {
                    "features": [str(frame[v].dtype) for v in record["features"]],
                    "outcome": str(frame[outcome].dtype) if outcome is not None else None,
                    "weights": str(frame[weights].dtype) if weights is not None else None,
                },
            },
            "prediction": prediction,
            "metrics": score,
            "sample": {
                "n": query,
                "dropped_rows": 0,
                "positions": list(range(query)),
                "role": "query",
            },
            "uncertainty": {"available": False},
        }
    )
    return CartPrediction(
        tables,
        title="CART saved prediction",
        **attrs,
    )


@c.cpu_call
def cart_prediction_restore(saved):
    """Replay all saved query rows and losses, without training a tree."""
    record = c.load(saved.attrs if isinstance(saved, CartPrediction) else saved)
    c.keys(
        record,
        (
            "schema",
            "model",
            "source_digest",
            "source",
            "prediction",
            "metrics",
            "sample",
            "uncertainty",
            "digest",
        ),
        "CART query result",
    )
    if record["schema"] != "openecon.supervised.cart-query.v1":
        c.fail("invalid_state", "Unsupported CART query schema.")
    c.keys(
        record["source"],
        ("n", "index", "x", "y", "weights", "outcome_name", "weight_name", "dtypes"),
        "complete query source",
    )
    source, model = record["source"], record["model"]
    if not isinstance(model, Mapping) or not isinstance(model.get("features"), (list, tuple)):
        c.fail("invalid_state", "Query requires its complete frozen model.")
    settings, k = _header(model)
    features = c.columns(model["features"])
    n, p = source["n"], len(features)
    c.integer(n, 1, c.MAX_ROWS, "query rows")
    _plan(model["n"], p, k, settings, query=n)
    c.sequence(source["x"], n, "query features")
    for row in source["x"]:
        c.numeric_cache(row, p, missing=True)
    c.keys(source["dtypes"], ("features", "outcome", "weights"), "query dtypes")
    c.sequence(source["dtypes"]["features"], p, "query feature dtypes")
    for dtype in source["dtypes"]["features"]:
        _dtype(dtype)
    if source["outcome_name"] is not None:
        c.columns([source["outcome_name"]])
        c.sequence(source["y"], n, "query outcomes")
        _dtype(source["dtypes"]["outcome"], classification=bool(model["classes"]))
        c.numeric_cache(source["weights"], n)
        if source["weight_name"] is not None:
            c.columns([source["weight_name"]])
            _dtype(source["dtypes"]["weights"])
        elif source["dtypes"]["weights"] is not None or any(v != 1.0 for v in source["weights"]):
            c.fail("invalid_state", "Unweighted query retains exact unit loss weights.")
    elif (
        any(source[key] is not None for key in ("y", "weights", "weight_name"))
        or source["dtypes"]["outcome"] is not None
        or source["dtypes"]["weights"] is not None
    ):
        c.fail("invalid_state", "Unscored query has unexpected response/weight source.")
    c.keys(record["prediction"], ("leaf_ids", "predictions", "probabilities"), "query predictions")
    for key in ("leaf_ids", "predictions"):
        c.sequence(record["prediction"][key], n, key)
    if model["classes"]:
        c.sequence(record["prediction"]["probabilities"], n, "query probabilities")
        for row in record["prediction"]["probabilities"]:
            c.numeric_cache(row, len(model["classes"]))
    else:
        c.numeric_cache(record["prediction"]["predictions"], n)
        if record["prediction"]["probabilities"] is not None:
            c.fail("invalid_state", "Regression query cannot have class probabilities.")
    metrics.admission(record["metrics"], k)
    c.unseal(record)
    c.same(record["source_digest"], model["digest"], "query model lineage")
    index = c.restore_index(source["index"], n)
    data = {}
    for j, name in enumerate(features):
        values = [row[j] for row in source["x"]]
        _representable(values, source["dtypes"]["features"][j])
        data[name] = pd.Series(values, dtype=source["dtypes"]["features"][j], index=index)
    for key, name_key, dtype_key in (
        ("y", "outcome_name", "outcome"),
        ("weights", "weight_name", "weights"),
    ):
        name = source[name_key]
        if name is not None:
            _representable(
                source[key],
                source["dtypes"][dtype_key],
                classification=key == "y" and bool(model["classes"]),
            )
            data[name] = pd.Series(source[key], dtype=source["dtypes"][dtype_key], index=index)
    result = cart_predict(
        model, pd.DataFrame(data), outcome=source["outcome_name"], weights=source["weight_name"]
    )
    c.same(record, result.attrs, "saved query predictions and losses")
    return result


EXPORTS = {
    "prediction_split": prediction_split,
    "split_restore": split_restore,
    "cart": cart,
    "cart_restore": cart_restore,
    "cart_predict": cart_predict,
    "cart_prediction_restore": cart_prediction_restore,
    "SplitState": SplitState,
    "CartState": CartState,
    "CartQueryState": CartQueryState,
}
