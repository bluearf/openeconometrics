"""Scale reliability: ``oe.alpha`` (SPSS RELIABILITY, Stata alpha)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c

MODELS = ("alpha", "split", "guttman")


def cronbach(cov: Tensor) -> float:
    """alpha = k/(k-1) (1 - trace(C) / 1'C1) for an item covariance matrix C."""
    k = cov.shape[0]
    return k / (k - 1.0) * (1.0 - float(cov.diagonal().sum()) / float(cov.sum()))


def _inverse_diagonal(matrix: Tensor) -> Tensor | None:
    """diag(C^{-1}), or None when C is not positive definite to working precision."""
    try:
        factor = c.positive_definite(matrix, "item matrix", "")
    except AnalysisError:
        return None
    return torch.cholesky_inverse(factor).diagonal()


def _split_half(cov: Tensor) -> dict[str, float | None]:
    """SPSS split-half statistics: the first ceil(k/2) items against the rest."""
    k = cov.shape[0]
    k1 = (k + 1) // 2
    k2 = k - k1
    total = float(cov.sum())
    v1, v2 = float(cov[:k1, :k1].sum()), float(cov[k1:, k1:].sum())
    r = c.finite((total - v1 - v2) / 2.0 / math.sqrt(v1 * v2)) if v1 > 0 and v2 > 0 else None
    equal = unequal = None
    if r is not None and r > -1.0:
        equal = 2.0 * r / (1.0 + r)
        weight = k1 * k2 / (k * k)
        if abs(r) < 1.0:
            unequal = (-r * r + math.sqrt(r ** 4 + 4.0 * r * r * (1.0 - r * r) * weight)) \
                / (2.0 * (1.0 - r * r) * weight)
        else:
            unequal = 1.0                      # the limit of the formula as r^2 -> 1
    return {
        "alpha_part1": cronbach(cov[:k1, :k1]) if k1 > 1 and v1 > 0 else None,
        "n_items_part1": k1,
        "alpha_part2": cronbach(cov[k1:, k1:]) if k2 > 1 and v2 > 0 else None,
        "n_items_part2": k2,
        "correlation_between_forms": r,
        "spearman_brown_equal_length": equal,
        "spearman_brown_unequal_length": unequal,
        "guttman_split_half": 2.0 * (total - v1 - v2) / total,
    }


def _guttman(cov: Tensor) -> dict[str, float | None]:
    """Guttman's (1945) six lower bounds to reliability."""
    k = cov.shape[0]
    total = float(cov.sum())
    off = cov.square().sum(1) - cov.diagonal().square()        # sum_{i != j} c_ij^2 per item
    lambda1 = 1.0 - float(cov.diagonal().sum()) / total
    inverse = _inverse_diagonal(cov)
    return {
        "lambda1": lambda1,
        "lambda2": lambda1 + math.sqrt(k / (k - 1.0) * float(off.sum())) / total,
        "lambda3": k / (k - 1.0) * lambda1,
        "lambda4": _split_half(cov)["guttman_split_half"],
        "lambda5": lambda1 + 2.0 * math.sqrt(float(off.max())) / total,
        "lambda6": None if inverse is None else 1.0 - float((1.0 / inverse).sum()) / total,
    }


@c.procedure
def alpha(data: Any, columns: list[str], *, standardized: bool = False,
          reverse: list[str] | None = None, model: str = "alpha",
          missing: str = "drop", weights: str | None = None,
          weight_type: str = "fweight") -> TableSet:
    """Cronbach's alpha and item analysis of a summated scale.

    With k items, item covariance matrix C (divisor n - 1) and scale variance
    s_T^2 = 1'C1 (the variance of the sum of the items),

        alpha = k/(k-1) * (1 - sum_j c_jj / s_T^2),
        standardized alpha = k rbar / (1 + (k-1) rbar),

    rbar being the average inter-item correlation. For each item the table gives
    the corrected item-total (item-rest) correlation, the correlation between the
    item and the sum of the other items; the item-test correlation with the full
    scale; the squared multiple correlation of the item on the others; and the
    alpha of the scale without the item.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : two or more numeric item columns.
    standardized : analyse standardized items (C is the correlation matrix), as
        Stata's ``alpha, std``; ``alpha`` then equals the standardized alpha. Both
        coefficients are reported either way, as SPSS does.
    reverse : items whose sign is reversed (x -> -x) before the analysis, Stata's
        ``reverse()``. No item is reversed automatically (SPSS behaviour; Stata's
        ``asis``).
    model : ``"alpha"`` (default); ``"split"`` adds SPSS's split-half table (first
        ceil(k/2) items against the rest: correlation between forms, Spearman-Brown
        coefficients for equal and unequal lengths, Guttman split-half coefficient
        and the alpha of each half); ``"guttman"`` adds Guttman's lambda-1 to
        lambda-6 lower bounds.
    missing : ``"drop"`` (listwise deletion, as SPSS; Stata's ``casewise``) or
        ``"raise"``.
    weights : optional nonnegative integer frequency-count column. Weighted
        anchored moments use divisor sum(weights)-1 without expanding rows;
        zero counts are excluded, and missing values are deleted over items
        plus the count column. Counts and total must be <=2**53. Resident and
        bounded Dataset inputs are supported. Only weight_type="fweight";
        these descriptive coefficients have no sampling SE or CI.

    Returns
    -------
    TableSet with

    * ``scale`` (one ``value`` column): ``alpha``, ``standardized_alpha``, ``n_items``,
      ``average_interitem_covariance``, ``average_interitem_correlation``,
      ``scale_mean``, ``scale_variance``, ``scale_std_dev`` (of the sum of the items).
      All describe the analysed items: with ``standardized=True`` they are z-scores,
      so the average inter-item covariance is the average correlation;
    * ``items``: ``mean``, ``std_dev``, ``item_test_correlation``,
      ``item_rest_correlation``, ``scale_mean_if_deleted``, ``scale_variance_if_deleted``,
      ``squared_multiple_correlation``, ``alpha_if_deleted``;
    * ``split`` or ``guttman`` (one ``value`` column) when requested.

    ``attrs``: ``n``, ``n_missing``, ``alpha``, ``standardized_alpha``, ``n_items``,
    ``model``, ``standardized``, ``reversed``.

    Formulas of the optional tables. Split-half with part sums of variances s_1^2,
    s_2^2 and correlation r: Spearman-Brown 2r/(1+r) (equal length) and
    [-r^2 + sqrt(r^4 + 4 r^2 (1-r^2) k_1 k_2 / k^2)] / [2 (1-r^2) k_1 k_2 / k^2]
    (unequal length); Guttman split-half 2 (s_T^2 - s_1^2 - s_2^2) / s_T^2. Guttman:
    L1 = 1 - sum c_jj / s_T^2; L2 = L1 + sqrt(k/(k-1) sum_{i!=j} c_ij^2) / s_T^2;
    L3 = alpha; L4 = Guttman split-half; L5 = L1 + 2 sqrt(max_j sum_{i!=j} c_ij^2) / s_T^2;
    L6 = 1 - sum_j e_j^2 / s_T^2 with e_j^2 the residual variance of item j regressed
    on the other items.

    Equivalent commands: SPSS ``RELIABILITY /VARIABLES=q1 q2 q3 /MODEL=ALPHA
    /SUMMARY=TOTAL`` (``/MODEL=SPLIT``, ``/MODEL=GUTTMAN``); Stata
    ``alpha q1 q2 q3, item casewise asis`` (``std``).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"q1": [3, 4, 2, 5, 4, 1, 3, 5], "q2": [2, 4, 3, 5, 5, 1, 2, 4],
    ...         "q3": [3, 5, 2, 4, 4, 2, 3, 5]}
    >>> result = oe.alpha(data, ["q1", "q2", "q3"])
    >>> 0.8 < result.attrs["alpha"] < 1.0
    True
    """
    names = c.name_list(columns, "columns", minimum=2)
    c.check_flag(standardized, "standardized")
    c.check_choice(model, "model", MODELS)
    c.check_choice(weight_type, "weight_type", ("fweight",))
    flipped = c.name_list(reverse, "reverse", minimum=0)
    unknown = [name for name in flipped if name not in names]
    if unknown:
        raise AnalysisError("invalid_spec", f"reverse names column(s) that are not items: "
                            f"{', '.join(unknown)}.")
    diagnostics: dict[str, Any] = {}
    if weights is None:
        sample, _, dropped = c.select(data, names, missing=missing)
        x = c.matrix(sample, names)
        n, k = x.shape
        if n < 2:
            raise AnalysisError("insufficient_observations", "Reliability analysis needs at least "
                                "two complete observations.")
        if flipped:
            sign = torch.tensor([-1.0 if name in flipped else 1.0 for name in names], dtype=c.FLOAT)
            x = x * sign
        mean, sscp, _ = c.moments(x, names)
    else:
        from .weighted import frequency_moments
        mean, sscp, n, dropped, diagnostics = frequency_moments(data, names, weights, missing)
        k = len(names)
        if flipped:
            sign = torch.tensor([-1.0 if name in flipped else 1.0 for name in names], dtype=c.FLOAT)
            mean = mean * sign
            sscp = sscp * torch.outer(sign, sign)
        diagnostics["inference"] = "descriptive reliability coefficients; sampling SE and CI unavailable; no survey inference"
    covariance = sscp / (n - 1)
    corr = c.correlation(sscp)
    cov = corr if standardized else covariance
    if standardized:
        mean = torch.zeros_like(mean)
    total = float(cov.sum())
    if not total > 1e-12 * float(cov.diagonal().sum()):
        raise AnalysisError("degenerate_scale", "The sum of the items has zero variance, so "
                            "reliability coefficients are undefined. Check for items that "
                            "should be reverse-scored (reverse=[...]).")
    off = ~torch.eye(k, dtype=torch.bool)
    rbar = float(corr[off].mean())
    coefficient = cronbach(cov)
    standardized_alpha = c.finite(k * rbar / (1.0 + (k - 1.0) * rbar)) \
        if 1.0 + (k - 1.0) * rbar > 0 else None
    scale = [
        ["alpha", coefficient], ["standardized_alpha", standardized_alpha], ["n_items", k],
        # Of the analysed items: with standardized=True their covariances are correlations.
        ["average_interitem_covariance", float(cov[off].mean())],
        ["average_interitem_correlation", rbar], ["scale_mean", float(mean.sum())],
        ["scale_variance", total], ["scale_std_dev", math.sqrt(total)],
    ]

    diagonal = cov.diagonal()
    row = cov.sum(1)
    rest_variance = total + diagonal - 2.0 * row
    rest_ok = rest_variance > 1e-12 * total
    safe = torch.where(rest_ok, rest_variance, torch.ones_like(rest_variance))
    item_rest = torch.where(rest_ok, (row - diagonal) / (diagonal * safe).sqrt(),
                            torch.full_like(row, float("nan")))
    item_test = row / (diagonal * total).sqrt()
    if k > 2:
        deleted = (k - 1.0) / (k - 2.0) * (1.0 - (diagonal.sum() - diagonal) / safe)
        deleted = torch.where(rest_ok, deleted, torch.full_like(row, float("nan")))
    else:
        deleted = torch.full_like(row, float("nan"))
    inverse = _inverse_diagonal(corr)
    smc = torch.full_like(row, float("nan")) if inverse is None else 1.0 - 1.0 / inverse
    notes = []
    if inverse is None:
        notes.append("Squared multiple correlations are not available: the item correlation "
                     "matrix is singular.")
    if k == 2:
        notes.append("alpha_if_deleted is undefined for a two-item scale.")
    items = torch.stack([mean, diagonal.sqrt(), item_test, item_rest, mean.sum() - mean,
                         rest_variance, smc, deleted], dim=1)
    item_rows = [[c.finite(value) for value in line] for line in items.tolist()]
    tables = {
        "scale": c.frame([[value] for _, value in scale], columns=["value"],
                         index=[name for name, _ in scale]),
        "items": c.frame(item_rows, columns=[
            "mean", "std_dev", "item_test_correlation", "item_rest_correlation",
            "scale_mean_if_deleted", "scale_variance_if_deleted",
            "squared_multiple_correlation", "alpha_if_deleted"], index=names),
    }
    if model != "alpha":
        extra = _split_half(cov) if model == "split" else _guttman(cov)
        tables[model] = c.frame([[c.finite(value)] for value in extra.values()],
                                columns=["value"], index=list(extra))
    return TableSet(
        tables, title=f"Reliability analysis of {', '.join(names)}", procedure="alpha",
        n=n, n_missing=dropped, alpha=coefficient, standardized_alpha=standardized_alpha,
        n_items=k, model=model, standardized=standardized, reversed=flipped, notes=notes,
        missing="listwise", **diagnostics)
