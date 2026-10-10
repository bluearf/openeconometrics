"""Weighted held-out prediction losses; no adaptive-tree inference claim."""

from __future__ import annotations

import math
import torch

from . import common as c


def admission(value, k):
    """Admit every retained metric primitive/shape before numerical replay."""
    if value is None:
        return
    if isinstance(value, dict) and "available" in value:
        c.keys(value, ("available", "reason", "n"), "unavailable score")
        if value["available"] is not False or type(value["reason"]) is not str:
            c.fail("invalid_state", "Unavailable score has invalid flags.")
        c.integer(value["n"], 1, c.MAX_ROWS, "score rows")
        return
    if k:
        expected = (
            "n",
            "weight_sum",
            "accuracy",
            "brier",
            "brier_convention",
            "log_loss",
            "log_loss_unbounded",
            "auc",
            "auc_defined",
            "auc_positive_class_index",
            "confusion",
            "classwise",
            "averages",
            "calibration",
            "calibration_target",
            "denominator",
            "uncertainty_available",
        )
        numeric = ("weight_sum", "accuracy", "brier", "log_loss", "auc")
        flags = ("log_loss_unbounded", "auc_defined", "uncertainty_available")
    else:
        expected = (
            "n",
            "weight_sum",
            "rmse",
            "mae",
            "r2",
            "r2_defined",
            "denominator",
            "uncertainty_available",
        )
        numeric = ("weight_sum", "rmse", "mae", "r2")
        flags = ("r2_defined", "uncertainty_available")
    c.keys(value, expected, "complete score cache")
    c.integer(value["n"], 1, c.MAX_ROWS, "score rows")
    for key in numeric:
        if value[key] is not None and (
            type(value[key]) is not float or not math.isfinite(value[key])
        ):
            c.fail(
                "invalid_state",
                "Scores require finite float primitives or explicit unavailable values.",
            )
    for key in flags:
        if type(value[key]) is not bool:
            c.fail("invalid_state", "Score flags require actual booleans.")
    if not k:
        return
    c.sequence(value["confusion"], k, "confusion matrix")
    for row in value["confusion"]:
        c.numeric_cache(row, k, bound=1e308)
    c.sequence(value["classwise"], k, "classwise score")
    for item in value["classwise"]:
        c.keys(
            item,
            ("class_index", "support_weight", "predicted_weight", "precision", "recall", "f1"),
            "classwise score",
        )
        c.integer(item["class_index"], 0, k - 1, "score class")
        for key in ("support_weight", "predicted_weight", "precision", "recall", "f1"):
            if item[key] is not None and (
                type(item[key]) is not float or not math.isfinite(item[key])
            ):
                c.fail("invalid_state", "Classwise score has malformed primitives.")
    c.keys(value["averages"], ("precision", "recall", "f1"), "score averages")
    for item in value["averages"].values():
        c.keys(item, ("macro", "macro_classes", "weighted", "weighted_support"), "score average")
        c.integer(item["macro_classes"], 0, k, "defined macro classes")
        for key in ("macro", "weighted", "weighted_support"):
            if item[key] is not None and (
                type(item[key]) is not float or not math.isfinite(item[key])
            ):
                c.fail("invalid_state", "Score averages have malformed primitives.")
    c.sequence(value["calibration"], 10, "reliability bins")
    for item in value["calibration"]:
        c.keys(
            item,
            (
                "bin",
                "lower",
                "upper",
                "upper_closed",
                "n",
                "weight_sum",
                "mean_confidence",
                "accuracy",
            ),
            "reliability bin",
        )
        c.integer(item["bin"], 0, 9, "reliability bin")
        c.integer(item["n"], 0, value["n"], "reliability count")
        if type(item["upper_closed"]) is not bool:
            c.fail("invalid_state", "Reliability endpoint requires an actual boolean.")
        for key in ("lower", "upper", "weight_sum", "mean_confidence", "accuracy"):
            if item[key] is not None and (
                type(item[key]) is not float or not math.isfinite(item[key])
            ):
                c.fail("invalid_state", "Reliability values have malformed primitives.")


def regression(y, prediction, weights):
    weight_sum = float(weights.sum())
    w = weights / weight_sum
    residual = y - prediction
    scale = max(float(y.abs().max()), float(prediction.abs().max()))
    if scale == 0:
        scale = 1.0
    mean = float((w * (y / scale)).sum())
    mse = float((w * (residual / scale).square()).sum())
    total = float((w * (y / scale - mean).square()).sum())
    return {
        "n": len(y),
        "weight_sum": weight_sum,
        "rmse": math.sqrt(max(0.0, mse)) * scale,
        "mae": float((w * residual.abs()).sum()),
        "r2": 1.0 - mse / total if total > 0 else None,
        "r2_defined": total > 0,
        "denominator": "sum of positive case loss weights",
        "uncertainty_available": False,
    }


def _auc(y, p, weights):
    positive = float(weights[y == 1].sum())
    negative = float(weights[y == 0].sum())
    if positive == 0 or negative == 0:
        return None
    order = torch.argsort(p, stable=True)
    ys, ps, ws = y[order], p[order], weights[order]
    concordance = below = 0.0
    start = 0
    for end in range(1, len(y) + 1):
        if end < len(y) and float(ps[end]) == float(ps[start]):
            continue
        labels, mass = ys[start:end], ws[start:end]
        pos = float(mass[labels == 1].sum())
        neg = float(mass[labels == 0].sum())
        concordance += pos * (below + 0.5 * neg)
        below += neg
        start = end
    return concordance / (positive * negative)


def classification(y, probabilities, weights, k):
    weight_sum = float(weights.sum())
    w = weights / weight_sum
    prediction = probabilities.argmax(1)
    confusion = torch.zeros((k, k), dtype=c.FLOAT, device="cpu")
    confusion.index_put_((y, prediction), weights, accumulate=True)
    support, predicted = confusion.sum(1), confusion.sum(0)
    diagonal = confusion.diagonal()
    classwise = []
    for j in range(k):
        precision = float(diagonal[j] / predicted[j]) if float(predicted[j]) > 0 else None
        recall = float(diagonal[j] / support[j]) if float(support[j]) > 0 else None
        f1 = (
            2 * float(diagonal[j]) / float(predicted[j] + support[j])
            if float(predicted[j] + support[j]) > 0
            else None
        )
        classwise.append(
            {
                "class_index": j,
                "support_weight": float(support[j]),
                "predicted_weight": float(predicted[j]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    averages = {}
    for metric in ("precision", "recall", "f1"):
        defined = [j for j in range(k) if classwise[j][metric] is not None]
        denominator = sum(float(support[j]) for j in defined)
        averages[metric] = {
            "macro": sum(classwise[j][metric] for j in defined) / len(defined) if defined else None,
            "macro_classes": len(defined),
            "weighted": sum(classwise[j][metric] * float(support[j]) for j in defined) / denominator
            if denominator > 0
            else None,
            "weighted_support": denominator,
        }
    true_probability = probabilities[torch.arange(len(y), device="cpu"), y]
    unbounded = bool((true_probability == 0).any())
    log_loss = None if unbounded else -float((w * true_probability.log()).sum())
    indicator = torch.nn.functional.one_hot(y, num_classes=k).to(c.FLOAT)
    confidence = probabilities.max(1).values
    calibration = []
    for j in range(10):
        mask = (confidence >= j / 10) & (confidence < (j + 1) / 10 if j < 9 else confidence <= 1)
        mass = float(weights[mask].sum())
        calibration.append(
            {
                "bin": j,
                "lower": j / 10,
                "upper": (j + 1) / 10,
                "upper_closed": j == 9,
                "n": int(mask.sum()),
                "weight_sum": mass,
                "mean_confidence": float((weights[mask] * confidence[mask]).sum()) / mass
                if mass
                else None,
                "accuracy": float((weights[mask] * (prediction[mask] == y[mask]).to(c.FLOAT)).sum())
                / mass
                if mass
                else None,
            }
        )
    auc = _auc(y, probabilities[:, 1], weights) if k == 2 else None
    return {
        "n": len(y),
        "weight_sum": weight_sum,
        "accuracy": float((w * (prediction == y).to(c.FLOAT)).sum()),
        "brier": float((w * (probabilities - indicator).square().sum(1)).sum()),
        "brier_convention": "sum across all classes",
        "log_loss": log_loss,
        "log_loss_unbounded": unbounded,
        "auc": auc,
        "auc_defined": auc is not None,
        "auc_positive_class_index": 1 if k == 2 else None,
        "confusion": confusion.tolist(),
        "classwise": classwise,
        "averages": averages,
        "calibration": calibration,
        "calibration_target": "descriptive top-label reliability; no fitted calibration",
        "denominator": "sum of positive case loss weights",
        "uncertainty_available": False,
    }
