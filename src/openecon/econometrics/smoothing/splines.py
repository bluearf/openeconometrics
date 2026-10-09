"""Cox-de Boor, natural truncated-power and penalized additive splines."""

from __future__ import annotations

import math

import torch

from openecon.econometrics.core import build_result
from .common import DT, call, fail, finite, prepare, result, seal


def bspline_regress(*, data, y, x, missing="raise", alpha=0.05, **options):
    """B-spline OLS with fixed/training knots; reject extrapolation."""
    return call("bspline_regress", data, y, x, missing, alpha, **options)


def rcs_regress(*, data, y, x, missing="raise", alpha=0.05, **options):
    """Restricted cubic OLS with persisted knots and exact linear tails."""
    return call("rcs_regress", data, y, x, missing, alpha, **options)


def gam_gaussian(*, data, y, x, missing="raise", alpha=0.05, **options):
    """Additive Gaussian P-splines; conditional frequentist covariance, training GCV."""
    return call("gam_gaussian", data, y, x, missing, alpha, **options)


def numbers(value, name, maximum=40):
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(isinstance(a, bool) or not isinstance(a, (int, float)) for a in value)
    ):
        fail("invalid_knots", f"{name} needs at most {maximum} finite numeric values.")
    try:
        converted = [float(a) for a in value]
    except (ValueError, OverflowError):
        fail("invalid_knots", f"{name} needs finite float64 values.")
    if any(not math.isfinite(a) for a in converted):
        fail("invalid_knots", f"{name} needs finite float64 values.")
    if any(b <= a for a, b in zip(converted, converted[1:])):
        fail("invalid_knots", f"{name} must be strictly increasing.")
    return converted


def knot_options(spec):
    raw = spec.options.get("knots")
    if raw is not None and not isinstance(raw, dict):
        fail("invalid_knots", "knots must map every predictor to its ordered knot list.")
    if raw is not None and set(raw) != set(spec.predictors):
        fail("invalid_knots", "Explicit knots must cover exactly the declared predictors.")
    if raw is not None:
        for value in raw.values():
            numbers(value, "knots")
    return raw


def bs_basis(x, knots, boundary, degree):
    lo, hi = boundary
    if bool(((x < lo) | (x > hi)).any()):
        fail("outside_support", "B-spline prediction requires points within the saved boundary.")
    t = [lo] * (degree + 1) + knots + [hi] * (degree + 1)
    n = len(t) - 1
    b = torch.stack([((x >= t[j]) & (x < t[j + 1])).to(DT) for j in range(n)], 1)
    for d in range(1, degree + 1):
        parts = []
        for j in range(n - d):
            a = (x - t[j]) / (t[j + d] - t[j]) * b[:, j] if t[j + d] > t[j] else torch.zeros_like(x)
            c = (
                (t[j + d + 1] - x) / (t[j + d + 1] - t[j + 1]) * b[:, j + 1]
                if t[j + d + 1] > t[j + 1]
                else torch.zeros_like(x)
            )
            parts.append(a + c)
        b = torch.stack(parts, 1)
    b[x == hi, -1] = 1.0
    return finite(b)


def rcs_basis(x, knots):
    lo, hi = knots[0], knots[-1]
    z = (x - lo) / (hi - lo)
    t = [(a - lo) / (hi - lo) for a in knots]
    last, penultimate = t[-1], t[-2]
    pieces = [z]
    for a in t[:-2]:
        pieces.append(
            (z - a).clamp_min(0).pow(3)
            - (z - penultimate).clamp_min(0).pow(3) * (last - a) / (last - penultimate)
            + (z - last).clamp_min(0).pow(3) * (penultimate - a) / (last - penultimate)
        )
    return finite(torch.stack(pieces, 1))


def _state(frame, raw, natural=False):
    records, blocks = [], [torch.ones((frame.n, 1), dtype=DT)]
    terms = ["_cons"]
    boundary = frame.spec.options.get("boundary")
    if boundary is not None and (
        not isinstance(boundary, dict) or set(boundary) != set(frame.spec.predictors)
    ):
        fail("invalid_knots", "boundary must map exactly every predictor to [lower, upper].")
    for col in frame.spec.predictors:
        x = frame.numeric(col)
        count = frame.option("n_knots")
        if natural:
            knots = (
                numbers(raw[col], "knots")
                if raw is not None
                else torch.quantile(x, torch.linspace(0.05, 0.95, count, dtype=DT)).tolist()
            )
            knots = numbers(knots, "knots")
            if len(knots) < 3:
                fail(
                    "invalid_knots",
                    "Restricted cubic splines need at least three distinct knots including boundaries.",
                )
            record = {"kind": "rcs", "column": col, "knots": knots}
            block = rcs_basis(x, knots)
        else:
            bounds = (
                numbers(boundary[col], "boundary", 2)
                if boundary is not None
                else [float(x.min()), float(x.max())]
            )
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                fail("invalid_knots", "Boundary requires two distinct ordered values.")
            knots = (
                numbers(raw[col], "knots")
                if raw is not None
                else torch.quantile(x, torch.linspace(0, 1, count + 2, dtype=DT)[1:-1]).tolist()
            )
            knots = numbers(knots, "knots")
            if any(a <= bounds[0] or a >= bounds[1] for a in knots):
                fail("invalid_knots", "Interior knots must lie strictly inside the boundary.")
            degree = 3 if frame.spec.estimator == "gam_gaussian" else frame.option("degree")
            record = {
                "kind": "bs",
                "column": col,
                "knots": knots,
                "boundary": bounds,
                "degree": degree,
            }
            # Drop first basis; intercept represents the shared constant.
            block = bs_basis(x, knots, bounds, degree)[:, 1:]
        blocks.append(block)
        terms.extend(f"{col}:basis{j + 1}" for j in range(block.shape[1]))
        records.append(record)
    return torch.cat(blocks, 1), {"transforms": records, "terms": terms}


def _fit(spec, data, natural=False):
    raw = knot_options(spec)
    maxknots = (
        max(map(len, raw.values())) if raw else spec.options.get("n_knots", 5 if natural else 3)
    )
    frame = prepare(spec, data, 1 + len(spec.predictors) * (maxknots + 4))
    x, state = _state(frame, raw, natural)
    return result(frame, x, state)


def fit_bspline_regress(spec, data):
    return _fit(spec, data)


def fit_rcs_regress(spec, data):
    return _fit(spec, data, True)


def fit_gam_gaussian(spec, data):
    raw = knot_options(spec)
    path = spec.options.get("penalty_path")
    selection = spec.options.get("selection", "fixed")
    if selection == "gcv":
        if not path or len(path) > 100 or any(a < 0 for a in path) or len(set(path)) != len(path):
            fail("invalid_penalty", "GCV needs 1..100 unique nonnegative penalty_path values.")
    elif path is not None:
        fail("invalid_penalty", "penalty_path is only accepted for selection='gcv'.")
    else:
        path = [spec.options.get("penalty", 1.0)]
    if selection == "gcv" and "penalty" in spec.options:
        fail(
            "invalid_penalty", "Supply penalty_path alone for GCV; fixed penalty would be ignored."
        )
    maxknots = max(map(len, raw.values())) if raw else spec.options.get("n_knots", 3)
    frame = prepare(spec, data, 1 + len(spec.predictors) * (maxknots + 4), len(path) + 1)
    x, state = _state(frame, raw)
    pmat = torch.zeros((x.shape[1], x.shape[1]), dtype=DT)
    offset = 1
    for record in state["transforms"]:
        size = len(record["knots"]) + 3
        means = x[:, offset : offset + size].mean(0)
        x[:, offset : offset + size] -= means
        record["center"] = means.tolist()
        d = torch.diff(torch.eye(size + 1, dtype=DT), n=2, dim=0)[:, 1:]
        pmat[offset : offset + size, offset : offset + size] = d.T @ d
        offset += size
    y = frame.numeric(spec.outcome)
    gram, rhs = x.T @ x, x.T @ y
    # Identification is required even if a penalty could mask confounding.
    from .common import solve

    solve(x, y)
    candidates = []
    for lam in path:
        inv = torch.linalg.inv(gram + lam * pmat)
        b = finite(inv @ rhs)
        rss = float((y - x @ b).square().sum())
        ig = inv @ gram
        edf = float(torch.trace(ig))
        residual_df = frame.n - 2 * edf + float(torch.trace(ig @ ig))
        if not math.isfinite(rss) or residual_df <= 0 or frame.n - edf <= 0:
            fail(
                "invalid_dispersion",
                "Penalized fit needs finite RSS and positive residual trace df.",
            )
        gcv = frame.n * rss / (frame.n - edf) ** 2
        candidates.append(
            {"penalty": lam, "gcv": gcv, "edf": edf, "residual_trace_df": residual_df, "rss": rss}
        )
    chosen = min(range(len(path)), key=lambda j: (candidates[j]["gcv"], j))
    # Keep only one candidate's dense solve live; replay the selected solve.
    inv = torch.linalg.inv(gram + path[chosen] * pmat)
    b, rdf = finite(inv @ rhs), candidates[chosen]["residual_trace_df"]
    cov = finite(candidates[chosen]["rss"] / rdf * inv @ gram @ inv.T)
    note = "Approximate conditional frequentist covariance around the penalized estimator expectation; smoothing bias and selected penalty uncertainty excluded. No coefficient tests."
    state = seal(
        {
            **state,
            "method": spec.estimator,
            "columns": spec.predictors,
            "coefficients": b.tolist(),
            "covariance": cov.tolist(),
            "penalty_matrix": pmat.tolist(),
            "candidates": candidates,
            "selection": selection,
            "selected": chosen,
            "df_resid": rdf,
            "sample_positions": frame.positions,
        }
    )
    return build_result(
        frame,
        terms=[],
        params=torch.empty(0, dtype=DT),
        covariance=torch.empty((0, 0), dtype=DT),
        fitted=x @ b,
        use_t=False,
        df_resid=rdf,
        metrics=candidates[chosen],
        inference={"available": False, "distribution": "none", "conditioning": note},
        extra={"smoothing_state": state, "notes": [note]},
        warnings=[note],
        solver="direct penalized Gaussian solve and training GCV",
    )
