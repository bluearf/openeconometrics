"""Direct LOESS and explicit finite-category product-kernel regression."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from .common import DT, call, fail, finite, prediction_result, prepare, solve


def loess(*, data, y, x, missing="raise", **options):
    """Direct one-dimensional Gaussian tricube LOESS; no pointwise confidence intervals."""
    return call("loess", data, y, x, missing, **options)


def npreg_mixed(*, data, y, x, missing="raise", **options):
    """Declared c/u/o product-kernel conditional mean, fixed or exact training LOO tuning."""
    return call("npreg_mixed", data, y, x, missing, **options)


def loess_coefficients(train, y, q, span, degree, exclude=None):
    mask = torch.ones(len(train), dtype=torch.bool)
    if exclude is not None:
        mask[exclude] = False
    x, target = train[mask], y[mask]
    distance = (x - q).abs()
    count = math.ceil(span * len(x))
    if count < degree + 2:
        fail(
            "local_rank_deficient",
            "Span selects too few local observations for the polynomial degree.",
        )
    radius = float(torch.kthvalue(distance, count).values)
    if radius <= 0 or not math.isfinite(radius):
        fail("local_rank_deficient", "The nearest-neighbor radius must be positive and finite.")
    w = (1 - (distance / radius).clamp(max=1).pow(3)).pow(3)
    z = (x - q) / radius
    a = torch.stack([z.pow(j) for j in range(degree + 1)], 1)
    try:
        b, _, _ = solve(a * w.sqrt()[:, None], target * w.sqrt())
    except AnalysisError as exc:
        fail("local_rank_deficient", str(exc))
    return finite(b), radius


def loess_point(train, y, q, span, degree, exclude=None):
    return loess_coefficients(train, y, q, span, degree, exclude)[0][0]


def loess_values(x, y, queries, span, degree, loo=False):
    if bool(((queries < x.min()) | (queries > x.max())).any()):
        fail("outside_support", "LOESS queries must lie within the saved training range.")
    return finite(
        torch.stack(
            [loess_point(x, y, q, span, degree, j if loo else None) for j, q in enumerate(queries)]
        )
    )


def _select(frame, path, predict):
    rows = []
    for candidate in path:
        try:
            fitted = predict(candidate)
            loss = float((frame.numeric(frame.spec.outcome) - fitted).square().mean())
            if not math.isfinite(loss):
                fail("numerical_failure", "Leave-one-out loss exceeds finite float64 arithmetic.")
            rows.append({"candidate": candidate, "loo_mse": loss, "failures": []})
        except AnalysisError as exc:
            rows.append(
                {
                    "candidate": candidate,
                    "loo_mse": None,
                    "failures": [{"code": exc.code, "message": str(exc)}],
                }
            )
    valid = [j for j, row in enumerate(rows) if row["loo_mse"] is not None]
    if not valid:
        fail(
            "no_valid_candidate", "Every leave-one-out candidate failed its support/rank contract."
        )
    chosen = min(valid, key=lambda j: (rows[j]["loo_mse"], j))
    return path[chosen], rows, chosen


def fit_loess(spec, data):
    if len(spec.predictors) != 1:
        fail("unsupported_dimensions", "LOESS currently accepts exactly one numeric predictor.")
    selection = spec.options.get("selection", "fixed")
    path = spec.options.get("span_path")
    if selection == "loo":
        if (
            not path
            or len(path) > 100
            or len(set(path)) != len(path)
            or any(not 0 < s <= 1 for s in path)
        ):
            fail("invalid_span", "LOO needs 1..100 unique span_path values in (0,1].")
        if "span" in spec.options:
            fail("invalid_span", "LOO selects span_path; a fixed span must not be supplied.")
    elif path is not None:
        fail("invalid_span", "span_path requires selection='loo'.")
    else:
        path = [spec.options.get("span", 0.75)]
    frame = prepare(spec, data, 4, len(path), quadratic=True)
    # Selection traverses n queries for each candidate, in addition to the final fit.
    work = (len(path) + 1) * frame.n * frame.n * 32
    if work > frame.option("max_work"):
        fail("work_limit", f"LOESS plans {work} scalar work units.")
    x, y = frame.numeric(spec.predictors[0]), frame.numeric(spec.outcome)
    degree = frame.option("degree")
    span, rows, selected = (
        _select(frame, path, lambda s: loess_values(x, y, x, s, degree, True))
        if selection == "loo"
        else (path[0], [], 0)
    )
    fitted = loess_values(x, y, x, span, degree)
    return prediction_result(
        frame,
        fitted,
        {
            "train_x": x.tolist(),
            "train_y": y.tolist(),
            "span": span,
            "degree": degree,
            "selection": selection,
            "candidates": rows,
            "selected": selected,
        },
        {"span_selection": rows},
    )


def category_key(value):
    if (
        type(value) not in (str, int, float, bool)
        or isinstance(value, float)
        and not math.isfinite(value)
    ):
        fail("invalid_category", "Categories must be finite JSON scalar strings/numbers/bools.")
    return (type(value).__name__, value)


def encode(data, columns, types, categories):
    from openecon.analysis import _numeric

    encoded = []
    for col in columns:
        kind = types[col]
        if kind == "c":
            encoded.append(_numeric(data[col], col))
        else:
            levels = categories[col]
            keys = [category_key(a) for a in levels]
            codes = {key: j for j, key in enumerate(keys)}
            values = [category_key(a) for a in data[col].tolist()]
            if any(a not in codes for a in values):
                fail(
                    "unknown_category",
                    f"{col} contains values outside the saved declared universe.",
                )
            encoded.append(torch.tensor([codes[a] for a in values], dtype=DT))
    return finite(torch.stack(encoded, 1))


def validate_bw(bw, columns, types, categories):
    if (
        not isinstance(bw, list)
        or len(bw) != len(columns)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in bw
        )
    ):
        fail("invalid_bandwidth", "One finite bandwidth is required per predictor.")
    for col, a in zip(columns, bw, strict=True):
        kind = types[col]
        if (
            kind == "c"
            and a <= 0
            or kind == "u"
            and not 0 <= a <= (len(categories[col]) - 1) / len(categories[col])
            or kind == "o"
            and not 0 <= a < 1
        ):
            fail(
                "invalid_bandwidth",
                "Numeric h>0; unordered lambda in [0,(K-1)/K]; ordered lambda in [0,1).",
            )


def mixed_values(x, y, q, bw, columns, types, categories, min_effective, loo=False):
    values = []
    for i, query in enumerate(q):
        logw = torch.zeros(len(x), dtype=DT)
        for j, col in enumerate(columns):
            h = bw[j]
            if types[col] == "c":
                if float(query[j]) < float(x[:, j].min()) or float(query[j]) > float(x[:, j].max()):
                    fail(
                        "outside_support", "Numeric queries must lie within saved training ranges."
                    )
                logw -= 0.5 * ((x[:, j] - query[j]) / h).square()
            elif types[col] == "u":
                weight = torch.where(
                    x[:, j] == query[j],
                    torch.tensor(1 - h, dtype=DT),
                    torch.tensor(h / (len(categories[col]) - 1), dtype=DT),
                )
                logw += weight.log()
            else:
                distance = (x[:, j] - query[j]).abs()
                weight = torch.where(
                    distance == 0, torch.tensor(1 - h, dtype=DT), 0.5 * (1 - h) * h**distance
                )
                logw += weight.log()
        if loo:
            logw[i] = -torch.inf
        if not bool(torch.isfinite(logw).any()):
            fail("no_local_support", "No training rows have positive product-kernel weight.")
        w = torch.exp(logw - logw.max())
        w /= w.sum()
        effective = float(1 / w.square().sum())
        if not math.isfinite(effective) or effective + 1e-12 < min_effective:
            fail("no_local_support", "Effective product-kernel sample is below min_effective.")
        values.append(w @ y)
    return finite(torch.stack(values))


def fit_npreg_mixed(spec, data):
    types, categories = spec.options["variable_types"], spec.options["categories"]
    if (
        not isinstance(types, dict)
        or set(types) != set(spec.predictors)
        or any(a not in ("c", "u", "o") for a in types.values())
    ):
        fail("invalid_category", "variable_types must declare exactly each predictor as c, u or o.")
    categorical = [c for c in spec.predictors if types[c] != "c"]
    if not categorical or not isinstance(categories, dict) or set(categories) != set(categorical):
        fail(
            "invalid_category",
            "Declare a complete categories map for every u/o predictor; use kernelreg for all-numeric data.",
        )
    for levels in categories.values():
        if (
            not isinstance(levels, list)
            or not 2 <= len(levels) <= 1000
            or len(set(map(category_key, levels))) != len(levels)
        ):
            fail(
                "invalid_category",
                "Each ordered/unordered universe needs 2..1000 unique typed levels.",
            )
    selection = spec.options.get("selection", "fixed")
    path = spec.options.get("bandwidth_path")
    if selection == "loo":
        if not isinstance(path, list) or not 1 <= len(path) <= 100:
            fail("invalid_bandwidth", "LOO needs 1..100 bandwidth_path candidates.")
        # required bandwidth is the declared starting/reference configuration, recorded separately.
    elif path is not None:
        fail("invalid_bandwidth", "bandwidth_path requires selection='loo'.")
    else:
        path = [spec.options["bandwidth"]]
    validate_bw(spec.options["bandwidth"], spec.predictors, types, categories)
    for bw in path:
        validate_bw(bw, spec.predictors, types, categories)
    frame = prepare(spec, data, len(spec.predictors) + 1, len(path), quadratic=True)
    work = (len(path) + 1) * frame.n * frame.n * (len(spec.predictors) + 2) * 8
    if work > frame.option("max_work"):
        fail("work_limit", f"Mixed kernel plans {work} scalar work units.")
    x = encode(frame.sample, spec.predictors, types, categories)
    y = frame.numeric(spec.outcome)
    arguments = (spec.predictors, types, categories, frame.option("min_effective"))
    bw, rows, selected = (
        _select(frame, path, lambda b: mixed_values(x, y, x, b, *arguments, loo=True))
        if selection == "loo"
        else (path[0], [], 0)
    )
    fitted = mixed_values(x, y, x, bw, *arguments)
    return prediction_result(
        frame,
        fitted,
        {
            "train_x": x.tolist(),
            "train_y": y.tolist(),
            "variable_types": types,
            "categories": categories,
            "bandwidth": bw,
            "reference_bandwidth": spec.options["bandwidth"],
            "min_effective": arguments[-1],
            "selection": selection,
            "candidates": rows,
            "selected": selected,
        },
        {"bandwidth_selection": rows},
    )
