"""Native float64 nuisance models and bounded CART forest kernels."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.regularized.kernels import fit_penalized
from openecon.econometrics.teffects.index import fit_index


class Work:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0

    def add(self, amount):
        self.used += int(amount)
        if self.used > self.maximum:
            raise AnalysisError(
                "work_limit",
                "Causal learning exceeds max_work; no observations, folds or trees were truncated.",
            )


def route(tree, x):
    leaf = torch.empty(len(x), dtype=torch.int64)
    stack = [(0, torch.arange(len(x)))]
    while stack:
        node, rows = stack.pop()
        item = tree[node]
        if "feature" not in item:
            leaf[rows] = node
        else:
            left = x[rows, item["feature"]] <= item["threshold"]
            stack.extend(((item["left"], rows[left]), (item["right"], rows[~left])))
    return leaf


def grow(x, y, o, generator, work, *, check_x=None):
    """Split using x/y only; optional check_x constrains support, never uses its outcomes."""
    n, p = x.shape
    if p == 0:
        return [{"value": float(y.mean()), "count": n}]
    mtry = o["mtry"] or max(1, math.ceil(math.sqrt(p)))
    if mtry > p:
        raise AnalysisError("invalid_mtry", "mtry must not exceed the number of controls.")
    tree = []

    def node(rows, depth, check_rows):
        index = len(tree)
        tree.append({"value": float(y[rows].mean()), "count": len(rows)})
        if depth >= o["max_depth"] or len(rows) < 2 * o["min_leaf"] or p == 0:
            return index
        features = torch.randperm(p, generator=generator)[:mtry].tolist()
        best = None
        for feature in features:
            work.add(len(rows) * (2 + int(math.ceil(math.log2(len(rows)))) + o["split_candidates"]))
            order = torch.argsort(x[rows, feature], stable=True)
            sorted_rows = rows[order]
            values, outcome = x[sorted_rows, feature], y[sorted_rows]
            candidates = (
                torch.linspace(
                    o["min_leaf"],
                    len(rows) - o["min_leaf"],
                    steps=min(o["split_candidates"], len(rows)),
                    dtype=torch.float64,
                )
                .long()
                .unique()
            )
            sums = outcome.cumsum(0)
            for at in candidates.tolist():
                if (
                    at < o["min_leaf"]
                    or at > len(rows) - o["min_leaf"]
                    or values[at - 1] == values[at]
                ):
                    continue
                threshold = float(values[at - 1] / 2 + values[at] / 2)
                if check_x is not None:
                    mask = check_x[check_rows, feature] <= threshold
                    work.add(len(check_rows))
                    if int(mask.sum()) < o["min_leaf"] or int((~mask).sum()) < o["min_leaf"]:
                        continue
                gain = float(
                    sums[at - 1].square() / at
                    + (sums[-1] - sums[at - 1]).square() / (len(rows) - at)
                    - sums[-1].square() / len(rows)
                )
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, feature, threshold, sorted_rows[:at], sorted_rows[at:])
        if best is not None:
            _, feature, threshold, left, right = best
            mask = None if check_x is None else check_x[check_rows, feature] <= threshold
            lcheck = None if mask is None else check_rows[mask]
            rcheck = None if mask is None else check_rows[~mask]
            tree[index] = {
                "feature": feature,
                "threshold": threshold,
                "left": node(left, depth + 1, lcheck),
                "right": node(right, depth + 1, rcheck),
            }
        return index

    node(torch.arange(n), 0, None if check_x is None else torch.arange(len(check_x)))
    return tree


def fit_model(x, y, binary, o, seed, work):
    n, p = x.shape
    if n < max(4, p + 2):
        raise AnalysisError(
            "insufficient_training_sample",
            "Each nuisance training stratum needs at least max(4,p+2) rows.",
        )
    if binary and (y.min() == y.max()):
        raise AnalysisError(
            "degenerate_training_stratum",
            "A binary nuisance training sample must contain both levels.",
        )
    learner = o["learner"]
    if learner == "forest":
        generator = torch.Generator().manual_seed(seed)
        trees = []
        for _ in range(o["trees"]):
            # Without-replacement subsampling has explicit training-only membership.
            size = max(2 * o["min_leaf"], int(math.ceil(0.8 * n)))
            size = min(n, size)
            rows = torch.randperm(n, generator=generator)[:size]
            tree = grow(x[rows], y[rows], o, generator, work)
            if binary:
                for item in tree:
                    if "value" in item:
                        item["value"] = (item["value"] * item["count"] + o["leaf_prior"]) / (
                            item["count"] + 2 * o["leaf_prior"]
                        )
            trees.append(tree)
        return {
            "kind": "forest",
            "binary": binary,
            "trees": trees,
            "seed": seed,
            "pseudocount": o["leaf_prior"] if binary else None,
            "training_n": n,
        }
    center = x.mean(0)
    scale = (x - center).square().mean(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    z = torch.cat((torch.ones((n, 1), dtype=torch.float64), (x - center) / scale), 1)
    if learner == "linear":
        work.add(n * (p + 1) ** 2 * (200 if binary else 2))
        beta = fit_index(
            "logit" if binary else "linear", z, y, torch.ones_like(y), what="cross-fitted nuisance"
        )
        kind = "logit" if binary else "linear"
    elif binary:
        work.add(n * (p + 1) ** 2 + (p + 1) ** 3)
        ratio = 1.0 if learner == "lasso" else 0.0
        beta = torch.zeros(p + 1, dtype=torch.float64)
        beta[0] = torch.logit(y.mean())
        # Proximal gradient with a global logistic Lipschitz bound.
        step = 1 / (
            float(torch.linalg.matrix_norm(z, ord=2).square()) / (4 * n)
            + o["penalty"] * (1 - ratio)
        )
        converged = False
        for iteration in range(1, o["max_iterations"] + 1):
            work.add(n * (p + 1) * 4)
            gradient = z.T @ (torch.sigmoid(z @ beta) - y) / n
            gradient[1:] += o["penalty"] * (1 - ratio) * beta[1:]
            candidate = beta - step * gradient
            candidate[1:] = candidate[1:].sign() * (
                candidate[1:].abs() - step * o["penalty"] * ratio
            ).clamp_min(0)
            beta = candidate
            gradient = z.T @ (torch.sigmoid(z @ beta) - y) / n
            gradient[1:] += o["penalty"] * (1 - ratio) * beta[1:]
            residual = torch.where(
                beta[1:] != 0,
                (gradient[1:] + o["penalty"] * ratio * beta[1:].sign()).abs(),
                (gradient[1:].abs() - o["penalty"] * ratio).clamp_min(0),
            )
            error = max(float(gradient[0].abs()), float(residual.max()) if p else 0)
            if error <= o["tolerance"]:
                converged = True
                break
        if not converged:
            raise AnalysisError(
                "nonconvergence", "Penalized logistic nuisance did not meet its KKT tolerance."
            )
        kind = "logit"
    else:
        work.add(n * (p + 1) * o["max_iterations"] if learner == "lasso" else n * (p + 1) ** 2 * 2)
        fitted = fit_penalized(
            x,
            y,
            ratio=float(learner == "lasso"),
            intercept=True,
            options={
                "selection": "fixed",
                "penalty": o["penalty"],
                "standardize": True,
                "max_iterations": o["max_iterations"],
                "tolerance": o["tolerance"],
                "max_work": o["max_work"],
            },
            seed=seed,
        )
        return {
            "kind": "linear_original",
            "coefficient": fitted["coefficient"].tolist(),
            "constant": float(fitted["constant"]),
            "solver": fitted["state"],
            "training_n": n,
        }
    return {
        "kind": kind,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficient": beta.tolist(),
        "training_n": n,
        "solver": {
            "converged": True,
            "penalty": None if learner == "linear" else o["penalty"],
            "kkt_max": None if learner == "linear" else error,
            "iterations": None if learner == "linear" else iteration,
        },
    }


def predict_model(model, x, work):
    if model["kind"] == "forest":
        output = torch.zeros(len(x), dtype=torch.float64)
        for tree in model["trees"]:
            work.add(len(x) * len(tree))
            leaves = route(tree, x)
            output += torch.tensor([tree[i]["value"] for i in leaves.tolist()], dtype=torch.float64)
        return output / len(model["trees"])
    work.add(len(x) * (x.shape[1] + 1))
    if model["kind"] == "linear_original":
        return x @ torch.tensor(model["coefficient"], dtype=torch.float64) + model["constant"]
    z = (x - torch.tensor(model["center"], dtype=torch.float64)) / torch.tensor(
        model["scale"], dtype=torch.float64
    )
    value = torch.cat((torch.ones((len(x), 1), dtype=torch.float64), z), 1) @ torch.tensor(
        model["coefficient"], dtype=torch.float64
    )
    return torch.sigmoid(value) if model["kind"] == "logit" else value
