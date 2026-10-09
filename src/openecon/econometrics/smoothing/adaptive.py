"""Bounded deterministic paired-hinge forward and backward-GCV prediction."""

import math

import torch

from openecon.analysis_contracts import AnalysisError
from .common import DT, call, fail, finite, prepare, result, solve


def mars(*, data, y, x, missing="raise", **options):
    """Bounded paired-hinge MARS prediction; saved interaction basis, no coefficient inference."""
    return call("mars", data, y, x, missing, **options)


def hinge_basis(x, terms):
    parts = []
    for factors in terms:
        value = torch.ones(len(x), dtype=DT)
        for factor in factors:
            value *= ((x[:, factor["column_index"]] - factor["knot"]) * factor["sign"]).clamp_min(0)
        parts.append(value)
    return finite(torch.stack(parts, 1))


def fit_mars(spec, data):
    maxterms = spec.options.get("max_terms", 11)
    if maxterms % 2 != 1:
        fail("invalid_option", "max_terms must be odd: intercept plus paired hinges.")
    p = len(spec.predictors)
    frame = prepare(spec, data, maxterms)
    candidates = frame.option("max_candidates")
    work = frame.n * maxterms**4 * p * candidates * 8
    if work > frame.option("max_work"):
        fail("work_limit", f"MARS plans {work} scalar work units; reduce max_terms/candidates.")
    x, y = frame.matrix(spec.predictors), frame.numeric(spec.outcome)
    grids = []
    minspan = frame.option("min_span")
    for j in range(p):
        unique = torch.unique(x[:, j], sorted=True)
        knots = [
            float(k)
            for k in unique
            if int((x[:, j] < k).sum()) >= minspan and int((x[:, j] > k).sum()) >= minspan
        ]
        if len(knots) > candidates:
            indices = torch.linspace(0, len(knots) - 1, candidates).round().long().tolist()
            knots = [knots[i] for i in indices]
        grids.append(knots)
    if not any(grids):
        fail("no_valid_candidate", "No knot has min_span observations on both sides.")
    terms = [[]]
    _, rss, _ = solve(hinge_basis(x, terms), y)
    forward = [{"terms": 1, "rss": rss, "examined": 0, "rank_rejections": 0}]
    while len(terms) + 2 <= maxterms:
        best = None
        examined, rejected = 0, 0
        for parent, factors in enumerate(terms):
            if len(factors) >= frame.option("max_degree"):
                continue
            used = {a["column_index"] for a in factors}
            active = hinge_basis(x, [factors])[:, 0] != 0
            for j, knots in enumerate(grids):
                if j in used:
                    continue
                for knot in knots:
                    if (
                        int((active & (x[:, j] < knot)).sum()) < minspan
                        or int((active & (x[:, j] > knot)).sum()) < minspan
                    ):
                        continue
                    added = [
                        [*factors, {"column_index": j, "knot": knot, "sign": s}] for s in (1, -1)
                    ]
                    examined += 1
                    try:
                        _, loss, _ = solve(hinge_basis(x, terms + added), y)
                    except AnalysisError as exc:
                        if exc.code != "rank_deficient":
                            raise
                        rejected += 1
                        continue
                    if best is None or loss < best[0] - 1e-12 * max(1.0, rss):
                        best = (loss, added, parent, j, knot)
        if best is None or best[0] >= rss - 1e-12 * max(1.0, rss):
            forward.append(
                {
                    "terms": len(terms),
                    "rss": rss,
                    "examined": examined,
                    "rank_rejections": rejected,
                    "stop": "no_identified_improving_pair",
                }
            )
            break
        rss, added, parent, j, knot = best
        terms += added
        forward.append(
            {
                "terms": len(terms),
                "rss": rss,
                "examined": examined,
                "rank_rejections": rejected,
                "parent": parent,
                "column_index": j,
                "knot": knot,
            }
        )
    fullterms = terms
    retained = list(range(len(terms)))
    backward = []
    best = None
    while retained:
        selectedterms = [terms[j] for j in retained]
        _, rss, _ = solve(hinge_basis(x, selectedterms), y)
        count = len(retained)
        cost = count + frame.option("gcv_penalty") * (count - 1) / 2
        gcv = frame.n * rss / (frame.n - cost) ** 2 if cost < frame.n else None
        backward.append(
            {"retained": retained.copy(), "rss": rss, "effective_complexity": cost, "gcv": gcv}
        )
        if gcv is not None and (best is None or gcv < best[0]):
            best = (gcv, retained.copy())
        if count == 1:
            break
        removal = []
        for removed in retained[1:]:
            trial = [j for j in retained if j != removed]
            _, loss, _ = solve(hinge_basis(x, [terms[j] for j in trial]), y)
            removal.append((loss, removed))
        _, removed = min(removal)
        retained.remove(removed)
    if best is None or not math.isfinite(best[0]):
        fail("no_valid_candidate", "No backward model has valid finite GCV.")
    chosen = [fullterms[j] for j in best[1]]
    design = hinge_basis(x, chosen)
    state = {
        "hinges": chosen,
        "forward_terms": fullterms,
        "forward": forward,
        "backward": backward,
        "selected_indices": best[1],
        "candidate_knots": grids,
        "gcv_penalty": frame.option("gcv_penalty"),
        "terms": ["_cons", *[f"hinge{j}" for j in range(1, len(chosen))]],
        "transforms": [],
        "tie_policy": "input predictor order, parent order, ascending grid knots; backward removes smaller index on SSE tie",
    }
    return result(
        frame,
        design,
        state,
        predictive=True,
        diagnostics={"forward": forward, "backward": backward},
    )
