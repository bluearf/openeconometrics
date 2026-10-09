"""Explicit FP1/FP2 and single-variable approximate closed F testing."""

from itertools import combinations_with_replacement

import torch

from openecon.engines.distributions import f_sf
from .common import DT, call, fail, finite, prepare, result, solve

POWERS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]


def fp_regress(*, data, y, x, missing="raise", alpha=0.05, **options):
    """Fixed FP1/FP2 first predictor plus linear adjustments; conditional t inference."""
    return call("fp_regress", data, y, x, missing, alpha, **options)


def mfp_regress(*, data, y, x, missing="raise", alpha=0.05, **options):
    """Single FP variable with fixed linear adjustments and approximate 4/3/2 df F tests."""
    return call("mfp_regress", data, y, x, missing, alpha, **options)


def fp_basis(x, powers, scale):
    z = x / scale
    if bool((z <= 0).any()):
        fail(
            "invalid_fp_domain",
            "Fractional polynomial x/scale must be strictly positive; no automatic shifts.",
        )
    log = z.log()
    columns = []
    for j, power in enumerate(powers):
        a = log if power == 0 else z.pow(power)
        if j and power == powers[j - 1]:
            a = a * log
        columns.append(a)
    return finite(torch.stack(columns, 1)) if columns else torch.empty((len(x), 0), dtype=DT)


def _state(frame, powers):
    col = frame.spec.predictors[0]
    scale = frame.option("scale")
    b = fp_basis(frame.numeric(col), powers, scale)
    adjustments = frame.spec.predictors[1:]
    x = torch.cat([torch.ones((frame.n, 1), dtype=DT), b, frame.matrix(adjustments)], 1)
    terms = ["_cons", *[f"{col}:fp{j + 1}({a:g})" for j, a in enumerate(powers)], *adjustments]
    return x, {
        "terms": terms,
        "transforms": [
            {"kind": "fp", "column": col, "powers": powers, "scale": scale},
            *[{"kind": "linear", "column": name} for name in adjustments],
        ],
    }


def fit_fp_regress(spec, data):
    powers = spec.options.get("powers", [1.0])
    if len(powers) not in (1, 2) or any(p not in POWERS for p in powers):
        fail("invalid_powers", "FP supports one/two powers from [-2,-1,-.5,0,.5,1,2,3].")
    powers = sorted(powers)
    frame = prepare(spec, data, len(spec.predictors) + 2)
    x, state = _state(frame, powers)
    return result(frame, x, state)


def fit_mfp_regress(spec, data):
    frame = prepare(spec, data, len(spec.predictors) + 2, 45)
    y = frame.numeric(spec.outcome)
    candidates = []
    for powers in [
        [],
        *[[p] for p in POWERS],
        *[list(p) for p in combinations_with_replacement(POWERS, 2)],
    ]:
        x, _ = _state(frame, powers)
        _, rss, _ = solve(x, y)
        candidates.append({"powers": powers, "rss": rss, "n_parameters": x.shape[1]})
    best1 = min(range(1, 9), key=lambda j: (candidates[j]["rss"], j))
    best2 = min(range(9, 45), key=lambda j: (candidates[j]["rss"], j))
    full = candidates[best2]
    denominator_df = frame.n - full["n_parameters"]
    if full["rss"] <= 0:
        fail(
            "invalid_dispersion",
            "Closed F comparisons require positive best-FP2 residual variance.",
        )
    linear = next(j for j, c in enumerate(candidates) if c["powers"] == [1.0])
    tests = []
    for reduced, dnum, name in [
        (0, 4, "inclusion"),
        (linear, 3, "nonlinearity"),
        (best1, 2, "simplification"),
    ]:
        f = max(0.0, (candidates[reduced]["rss"] - full["rss"]) / dnum) / (
            full["rss"] / denominator_df
        )
        tests.append(
            {
                "test": name,
                "statistic": f,
                "df_num": dnum,
                "df_den": denominator_df,
                "p_value": f_sf(f, dnum, denominator_df),
                "reduced_candidate": reduced,
                "full_candidate": best2,
                "reference": "approximate Gaussian closed-test F; adaptive power search is not exact nested-model testing",
            }
        )
    chosen = (
        0
        if tests[0]["p_value"] >= frame.option("select_alpha")
        else linear
        if tests[1]["p_value"] >= frame.option("form_alpha")
        else best1
        if tests[2]["p_value"] >= frame.option("form_alpha")
        else best2
    )
    x, state = _state(frame, candidates[chosen]["powers"])
    state.update(
        {
            "candidates": candidates,
            "selected": chosen,
            "closed_tests": tests,
            "select_alpha": frame.option("select_alpha"),
            "form_alpha": frame.option("form_alpha"),
            "selection_scope": "one nonlinear variable; other predictors always linear and retained",
        }
    )
    return result(
        frame,
        x,
        state,
        diagnostics={"closed_tests": tests, "selected_powers": candidates[chosen]["powers"]},
    )
