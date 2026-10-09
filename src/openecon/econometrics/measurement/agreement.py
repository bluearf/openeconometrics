"""Chance-corrected agreement on declared category universes, sampled unit rows."""

from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from . import common as m


def _counts(codes, q):
    present = codes >= 0
    counts = torch.zeros((len(codes), q), dtype=c.FLOAT)
    counts.scatter_add_(1, codes.clamp_min(0), present.to(c.FLOAT))
    return counts


def _complete(
    data, raters, categories, inference, level, max_fits, max_work, device, weights, missing
):
    m.options(inference, level, max_fits, max_work, device, weights)
    cats = m.levels(categories)
    selected, metadata = m.sample(
        data,
        raters,
        missing=missing,
        q=len(cats),
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
    )
    m.codes(selected, cats)
    return cats, selected, metadata


@c.procedure
def cohen_kappa(
    data: Any,
    raters: list[str],
    *,
    categories: list,
    agreement_weights: Any = "unweighted",
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Cohen kappa for exactly two raters; declared linear/quadratic/custom agreement weights.

    Independent units are rows; missing='drop' is listwise. Observed and chance
    agreement use the two separate rater marginals on the same category order.
    Optional inference='jackknife' retains all delete-one units and gives an
    approximate t interval, without clipping negative estimates or interval ends.
    CPU float64 resident input only. Complete result persists with reliability_save.
    """
    if len(c.name_list(raters, "raters", minimum=2)) != 2:
        raise AnalysisError("invalid_spec", "Cohen kappa requires exactly two rater columns.")
    cats, selected, metadata = _complete(
        data, raters, categories, inference, level, max_fits, max_work, device, weights, missing
    )
    w, q = m.agreement_weights(cats, agreement_weights), len(cats)

    def fit(frame):
        code = m.codes(frame, cats)
        observed = (
            torch.bincount(code[:, 0] * q + code[:, 1], minlength=q * q).reshape(q, q).to(c.FLOAT)
        )
        prob = observed / len(frame)
        expected = torch.outer(prob.sum(1), prob.sum(0))
        pa, pe = float((prob * w).sum()), float((expected * w).sum())
        return (
            m.ratio(pa - pe, 1 - pe),
            {
                "counts": c.frame(observed, index=cats, columns=cats),
                "chance_probabilities": c.frame(expected, index=cats, columns=cats),
                "agreement_weights": c.frame(w, index=cats, columns=cats),
            },
            dict(observed_agreement=pa, chance_agreement=pe),
        )

    return m.result(
        "cohen_kappa",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(categories=cats, agreement_weights=w.tolist()),
    )


@c.procedure
def fleiss_kappa(
    data: Any,
    raters: list[str],
    *,
    categories: list,
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Fleiss kappa for complete units with a common number of exchangeable raters.

    Unweighted observed pair agreement and pooled category-marginal chance
    agreement. Named rater order cannot change the estimate. No weighting of
    categories or observation weights. Optional unit jackknife as cohen_kappa.
    """
    cats, selected, metadata = _complete(
        data, raters, categories, inference, level, max_fits, max_work, device, weights, missing
    )
    q, k = len(cats), len(raters)

    def fit(frame):
        counts = _counts(m.codes(frame, cats), q)
        unit_agree = (counts.square().sum(1) - k) / (k * (k - 1))
        prob = counts.sum(0) / (len(frame) * k)
        pa, pe = float(unit_agree.mean()), float(prob.square().sum())
        return (
            m.ratio(pa - pe, 1 - pe),
            {
                "category_counts": c.frame(counts, columns=cats),
                "marginals": c.frame(prob[:, None], index=cats, columns=["probability"]),
                "unit_agreement": c.frame(unit_agree[:, None], columns=["agreement"]),
            },
            dict(observed_agreement=pa, chance_agreement=pe, raters_exchangeable=True),
        )

    return m.result(
        "fleiss_kappa",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(categories=cats, raters=k, weighting="unweighted"),
    )


@c.procedure
def gwet_ac(
    data: Any,
    raters: list[str],
    *,
    categories: list,
    agreement_weights: Any = "unweighted",
    inference: str = "none",
    level: float = 0.95,
    missing: str = "drop",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Gwet AC1 (identity weights) / AC2 (symmetric agreement weights), complete units.

    Uses the explicitly declared q-category universe, including absent levels.
    Chance agreement is sum(W)/(q(q-1)) sum_j p_j(1-p_j), with pooled marginals.
    Observed agreement averages all distinct-rater pairs within each unit.
    Optional subject jackknife approximates sampling uncertainty; no vendor SE claim.
    """
    cats, selected, metadata = _complete(
        data, raters, categories, inference, level, max_fits, max_work, device, weights, missing
    )
    w, q, k = m.agreement_weights(cats, agreement_weights), len(cats), len(raters)

    def fit(frame):
        counts = _counts(m.codes(frame, cats), q)
        unit_agree = ((counts @ w * counts).sum(1) - k) / (k * (k - 1))
        prob = counts.sum(0) / (len(frame) * k)
        pa = float(unit_agree.mean())
        pe = float(w.sum() * (prob * (1 - prob)).sum() / (q * (q - 1)))
        return (
            m.ratio(pa - pe, 1 - pe),
            {
                "category_counts": c.frame(counts, columns=cats),
                "marginals": c.frame(prob[:, None], index=cats, columns=["probability"]),
                "agreement_weights": c.frame(w, index=cats, columns=cats),
                "unit_agreement": c.frame(unit_agree[:, None], columns=["agreement"]),
            },
            dict(
                observed_agreement=pa,
                chance_agreement=pe,
                coefficient="AC1" if torch.equal(w, torch.eye(q, dtype=c.FLOAT)) else "AC2",
            ),
        )

    return m.result(
        "gwet_ac",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(categories=cats, agreement_weights=w.tolist(), raters=k),
    )


@c.procedure
def krippendorff_alpha(
    data: Any,
    raters: list[str],
    *,
    categories: list,
    metric: str = "nominal",
    inference: str = "none",
    level: float = 0.95,
    missing: str = "available",
    device: str = "cpu",
    weights: Any = None,
    max_fits: int = 1024,
    max_work: int = 100_000_000,
):
    """Krippendorff alpha using finite-sample coincidences, including incomplete units.

    Only units with >=2 observed judgments contribute. 'nominal' uses mismatch,
    'ordinal' uses squared empirical cumulative-marginal distances on the given
    category order; 'interval' uses squared numeric distances. Expected coincidences
    exclude self-pairs. Unit jackknife is distinct from the author's pair bootstrap.
    """
    m.options(inference, level, max_fits, max_work, device, weights)
    c.check_choice(metric, "metric", ("nominal", "ordinal", "interval"))
    cats = m.levels(categories)
    if metric == "interval" and (
        any(isinstance(x, (str, bool)) for x in cats) or max(abs(x) for x in cats) > 1e150
    ):
        raise AnalysisError(
            "invalid_categories",
            "Interval categories must be finite numeric values no larger than 1e150.",
        )
    selected, metadata = m.sample(
        data,
        raters,
        missing=missing,
        available=True,
        q=len(cats),
        inference=inference,
        max_fits=max_fits,
        max_work=max_work,
    )
    m.codes(selected, cats)
    q = len(cats)

    def fit(frame):
        counts = _counts(m.codes(frame, cats), q)
        size = counts.sum(1)
        # Off-diagonal ordered pair frequencies and diagonal c(c-1), weighted
        # by 1/(m_u-1): each pairable unit contributes m_u coincidences.
        coincidence = counts.T @ (counts / (size - 1)[:, None])
        coincidence.diagonal().sub_((counts / (size - 1)[:, None]).sum(0))
        marginals = coincidence.sum(1)
        total = float(marginals.sum())
        expected = torch.outer(marginals, marginals)
        expected.diagonal().sub_(marginals)
        expected /= total - 1
        scale = 1.0
        if metric == "nominal":
            delta = 1 - torch.eye(q, dtype=c.FLOAT)
        elif metric == "ordinal":
            centers = marginals.cumsum(0) - marginals / 2
            delta = (centers[:, None] - centers[None, :]).square()
        else:
            points = torch.tensor(cats, dtype=c.FLOAT)
            scale = float(points.max() - points.min())
            points = (points - points.min()) / scale
            delta = (points[:, None] - points[None, :]).square()
        do, de = float((coincidence * delta).sum() / total), float((expected * delta).sum() / total)
        return (
            1 - m.ratio(do, de),
            {
                "coincidences": c.frame(coincidence, index=cats, columns=cats),
                "expected_coincidences": c.frame(expected, index=cats, columns=cats),
                "squared_distances": c.frame(delta, index=cats, columns=cats),
                "unit_judgments": c.frame(size[:, None], columns=["observed_raters"]),
            },
            dict(
                observed_disagreement=do,
                expected_disagreement=de,
                coincidence_total=total,
                distance_scale=scale,
                pairs=int((size * (size - 1) / 2).sum()),
            ),
        )

    return m.result(
        "krippendorff_alpha",
        selected,
        metadata,
        fit,
        inference=inference,
        level=level,
        settings=dict(categories=cats, metric=metric, missing="available pairable units"),
    )
