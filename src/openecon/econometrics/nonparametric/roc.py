"""ROC analysis: roc (one classifier) and roccomp (correlated curves, DeLong test).

The area under the empirical ROC curve is the Mann-Whitney statistic
U / (n1 n0). Everything is computed from the tie groups of the pooled scores,
which one sort delivers: with p_g positives and m_g negatives at the g-th
distinct score, a positive in group g exceeds sum_{h<g} m_h negatives and ties
with m_g, so its "placement" (the share of negatives it beats, ties 1/2) is

    V10_g = (sum_{h<g} m_h + m_g / 2) / n0,

and a negative in group g has V01_g = (sum_{h>g} p_h + p_g / 2) / n1, the share
of positives above it. AUC = mean(V10) = mean(V01), and the placements are the
structural components of DeLong, DeLong and Clarke-Pearson (1988). One sort, no
n1-by-n0 comparison matrix.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import (
    FLOAT, binary_codes, check_alpha, column_names, normal_two_sided, numeric, procedure,
    select, tie_term,
)
from openecon.engines.distributions import chi2_sf, normal_isf
from openecon.engines.linalg import wald_statistic

CURVE_POINTS = 400


def score_groups(score: Tensor, positive: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Tie groups of the pooled scores.

    Returns (distinct scores ascending, positives per group, negatives per
    group, group index of every observation).
    """
    distinct, group, counts = torch.unique(score, sorted=True, return_inverse=True,
                                           return_counts=True)
    positives = torch.zeros(distinct.numel(), dtype=FLOAT).index_add_(0, group,
                                                                      positive.to(FLOAT))
    return distinct, positives, counts.to(FLOAT) - positives, group


def group_placements(positives: Tensor, negatives: Tensor) -> tuple[Tensor, Tensor, Tensor,
                                                                    Tensor]:
    """Per tie group: (V10, V01, negatives strictly below, positives strictly above)."""
    below = negatives.cumsum(0) - negatives
    above = positives.sum() - positives.cumsum(0)
    v10 = (below + negatives / 2.0) / negatives.sum()
    v01 = (above + positives / 2.0) / positives.sum()
    return v10, v01, below, above


def placements(score: Tensor, positive: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(V10 over positives, V01 over negatives, tie-group sizes of the pooled scores).

    The placements are in the row order of the positive (negative) cases, so
    that several scores of the same cases can be stacked column by column.
    """
    _, positives, negatives, group = score_groups(score, positive)
    v10, v01, _, _ = group_placements(positives, negatives)
    return v10[group[positive]], v01[group[~positive]], positives + negatives


def hanley_mcneil_q(positives: Tensor, negatives: Tensor) -> tuple[float, float]:
    """(Q1, Q2) of Hanley and McNeil (1982, Table II), estimated without a model.

    Q1 is the probability that two random positives both score above one
    random negative, Q2 that one positive scores above two random negatives.
    From the positives p_g and negatives m_g per distinct score, with
    P_g = sum_{h>g} p_h the positives above and M_g = sum_{h<g} m_h the
    negatives below,

        Q1 = sum_g m_g (P_g^2 + P_g p_g + p_g^2 / 3) / (n0 n1^2),
        Q2 = sum_g p_g (M_g^2 + M_g m_g + m_g^2 / 3) / (n1 n0^2);

    the 1/3 is the chance that both members of a tied pair win when ties are
    broken at random.
    """
    n1, n0 = float(positives.sum()), float(negatives.sum())
    _, _, below, above = group_placements(positives, negatives)
    q1 = float((negatives * (above * above + above * positives
                             + positives * positives / 3.0)).sum()) / (n0 * n1 * n1)
    q2 = float((positives * (below * below + below * negatives
                             + negatives * negatives / 3.0)).sum()) / (n1 * n0 * n0)
    return q1, q2


def _sample(data: Any, y: Any, scores: list[str], positive: Any,
            missing: str) -> tuple[Tensor, list[Tensor], Any, int]:
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of the binary outcome column.")
    frame, dropped = select(data, [y, *scores], missing=missing)
    codes, levels = binary_codes(frame, y, positive)
    is_positive = codes == 1
    n1 = int(is_positive.sum())
    if n1 == 0 or n1 == codes.numel():
        raise AnalysisError("invalid_groups", f"'{y}' must contain both positive and negative "
                            "cases for an ROC analysis.")
    return is_positive, [numeric(frame, name) for name in scores], levels[1], dropped


def _interval(value: float, se: float | None, critical: float) -> tuple[Any, Any]:
    if se is None:
        return None, None
    return max(0.0, value - critical * se), min(1.0, value + critical * se)


@procedure
def roc(data: Any, y: str, score: str, *, positive: Any = None, alpha: float = 0.05,
        missing: str = "drop"):
    """Empirical ROC curve and the area under it (SPSS ROC, Stata ``roctab``).

    ``y`` is the binary state; the positive state is the larger of its two
    values (1 for 0/1 data) or ``positive``. Larger ``score`` values indicate
    the positive state. A case is classified positive when score >= threshold.

    ``curve``: one row per cutoff: ``threshold``, ``sensitivity``,
    ``specificity`` and ``one_minus_specificity``. Cutoffs are, as in SPSS,
    the smallest score minus 1, the midpoints between consecutive distinct
    scores and the largest score plus 1. With more than 400 cutoffs the table
    is thinned to 400 evenly spaced ones; the area always uses all of them.

    ``auc``: the trapezoidal area, equal to the Mann-Whitney statistic
    U / (n1 n0), with two standard errors and normal 1 - ``alpha`` intervals
    (clipped to [0, 1]):

    * ``delong``: S10 / n1 + S01 / n0, the variances of the placements
      (DeLong, DeLong and Clarke-Pearson 1988; Stata's default).
    * ``hanley_mcneil``: [A (1 - A) + (n1 - 1)(Q1 - A^2) + (n0 - 1)(Q2 - A^2)]
      / (n1 n0) with Q1, Q2 estimated from the data as in Table II of Hanley
      and McNeil (1982), ties contributing n=^2 / 3 (see ``hanley_mcneil_q``).
      This is the standard error SPSS prints "under the nonparametric
      assumption". (Stata's ``roctab, hanley`` instead plugs in the
      approximations Q1 = A / (2 - A), Q2 = 2 A^2 / (1 + A), which is not
      reported here.)

    The test of H0: area = 0.5 uses the null (Mann-Whitney) standard error:
    z = (A - 0.5) / se_null with the tie-corrected variance
    [(N + 1) - sum (t^3 - t) / (N (N - 1))] / (12 n1 n0), so that z is the
    rank-sum z of ``oe.ranksum``. SPSS's "Asymptotic Sig." uses the variance
    without the tie term, (N + 1) / (12 n1 n0); that version is returned as
    ``se_null_uncorrected``, ``z_uncorrected`` and ``p_value_uncorrected``
    (identical when the scores have no ties).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : binary state column.
    score : numeric classifier column (larger = more positive).
    positive : the value of ``y`` that is the positive state (default: the
        larger value).
    alpha : 1 - confidence level of the intervals.
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``curve`` and ``auc`` (rows delong, hanley_mcneil; columns
    auc, std_error, ci_low, ci_high). ``attrs``: n, n_positive, n_negative,
    positive, auc, std_error (DeLong), se_hanley_mcneil, se_null, z, p_value,
    se_null_uncorrected, z_uncorrected, p_value_uncorrected, youden_index and
    youden_threshold (the cutoff maximizing sensitivity + specificity - 1),
    curve_points (number of distinct cutoffs).

    Stata: ``roctab y score [, detail]`` (DeLong standard error by default).
    SPSS: ``ROC score BY y(1) /PRINT=SE COORDINATES``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"sick": [0, 0, 0, 0, 1, 1, 1, 1], "marker": [1, 2, 3, 5, 4, 6, 7, 8]}
    >>> round(oe.roc(data, "sick", "marker").attrs["auc"], 4)
    0.9375
    """
    alpha = check_alpha(alpha)
    if not isinstance(score, str):
        raise AnalysisError("invalid_spec", "score must be the name of one numeric column.")
    is_positive, (s,), event, dropped = _sample(data, y, [score], positive, missing)
    n = s.numel()
    n1 = int(is_positive.sum())
    n0 = n - n1
    distinct, positives, negatives, _ = score_groups(s, is_positive)
    v10, v01, _, _ = group_placements(positives, negatives)
    auc = float((positives * v10).sum()) / n1
    critical = normal_isf(alpha / 2.0)
    se_delong = None
    if n1 > 1 and n0 > 1:
        # Variances (divisor n - 1) of the placements, summed over tie groups.
        s10 = float((positives * (v10 - auc) ** 2).sum()) / (n1 - 1.0)
        s01 = float((negatives * (v01 - auc) ** 2).sum()) / (n0 - 1.0)
        se_delong = math.sqrt(s10 / n1 + s01 / n0)
    q1, q2 = hanley_mcneil_q(positives, negatives)
    se_hanley = math.sqrt(max(0.0, auc * (1.0 - auc) + (n1 - 1.0) * (q1 - auc * auc)
                              + (n0 - 1.0) * (q2 - auc * auc)) / (n1 * n0))
    null_variance = ((n + 1.0) - tie_term(positives + negatives) / (n * (n - 1.0))) \
        / (12.0 * n1 * n0)
    attrs: dict[str, Any] = {"n": n, "n_positive": n1, "n_negative": n0, "positive": event,
                             "auc": auc, "std_error": se_delong, "se_hanley_mcneil": se_hanley}
    if null_variance > 0:
        z = (auc - 0.5) / math.sqrt(null_variance)
        untied = math.sqrt((n + 1.0) / (12.0 * n1 * n0))
        attrs.update({"se_null": math.sqrt(null_variance), "z": z,
                      "p_value": normal_two_sided(z), "se_null_uncorrected": untied,
                      "z_uncorrected": (auc - 0.5) / untied,
                      "p_value_uncorrected": normal_two_sided((auc - 0.5) / untied)})
    else:
        attrs["notes"] = ["The score is constant: the curve is the diagonal and no test is "
                          "computed."]
    # Curve: cumulative counts of positives / negatives up to each distinct score.
    zero = torch.zeros(1, dtype=FLOAT)
    sensitivity = 1.0 - torch.cat([zero, positives.cumsum(0)]) / n1
    false_positive = 1.0 - torch.cat([zero, negatives.cumsum(0)]) / n0
    threshold = torch.cat([distinct[:1] - 1.0, (distinct[:-1] + distinct[1:]) / 2.0,
                           distinct[-1:] + 1.0])
    youden = sensitivity - false_positive
    best = int(torch.argmax(youden))
    attrs.update({"youden_index": float(youden[best]), "youden_threshold": float(threshold[best]),
                  "curve_points": int(threshold.numel()), "alpha": alpha, "n_dropped": dropped,
                  "label": "ROC analysis"})
    if threshold.numel() > CURVE_POINTS:
        pick = torch.linspace(0, threshold.numel() - 1, CURVE_POINTS).round().to(torch.int64)
        pick = torch.unique(pick)
        threshold, sensitivity, false_positive = threshold[pick], sensitivity[pick], \
            false_positive[pick]
    curve = table({"threshold": threshold.tolist(), "sensitivity": sensitivity.tolist(),
                   "specificity": (1.0 - false_positive).tolist(),
                   "one_minus_specificity": false_positive.tolist()})
    area = table([[auc, se_delong, *_interval(auc, se_delong, critical)],
                  [auc, se_hanley, *_interval(auc, se_hanley, critical)]],
                 columns=["auc", "std_error", "ci_low", "ci_high"],
                 index=["delong", "hanley_mcneil"])
    return TableSet({"curve": curve, "auc": area}, title=f"ROC analysis: {score} for {y}", **attrs)


@procedure
def roccomp(data: Any, y: str, scores: list[str], *, positive: Any = None, alpha: float = 0.05,
            missing: str = "drop"):
    """Test of equality of the areas under correlated ROC curves (DeLong et al. 1988).

    ``scores`` are k >= 2 classifiers measured on the same cases. With V10
    ([n1, k]) and V01 ([n0, k]) the placements of each score, the vector of
    areas has covariance

        S = S10 / n1 + S01 / n0,     S10 = cov(V10), S01 = cov(V01)  (divisor n - 1).

    H0: all areas are equal is tested with chi2 = (L A)' (L S L')^- (L A),
    L contrasting every score with the first, on rank(L S L') (normally k - 1)
    degrees of freedom -- Stata's ``roccomp``. The ``pairwise`` table gives
    z = (A_i - A_j) / sqrt(S_ii + S_jj - 2 S_ij) for every pair (unadjusted
    two-sided p).

    Rows are deleted listwise over ``y`` and all ``scores``.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : binary state column.
    scores : list of 2 to 50 numeric classifier columns.
    positive : the value of ``y`` that is the positive state (default: the
        larger value).
    alpha : 1 - confidence level of the intervals of the areas.
    missing : "drop" (a row with a missing value in ``y`` or any score is
        deleted and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``auc`` (per score: auc, std_error, ci_low, ci_high),
    ``tests`` (row equality: statistic, df, p_value) and ``pairwise``
    (difference, std_error, z, p_value). ``attrs``: n, n_positive, n_negative,
    statistic, df, p_value, covariance (k x k, nested list).

    Stata: ``roccomp y score1 score2``. SPSS: ``ROC ANALYSIS score1 score2 BY
    y(1) /DESIGN PAIR=TRUE``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"sick": [0, 0, 0, 0, 1, 1, 1, 1], "a": [1, 2, 3, 5, 4, 6, 7, 8],
    ...         "b": [2, 1, 6, 4, 3, 5, 8, 7]}
    >>> round(oe.roccomp(data, "sick", ["a", "b"]).attrs["p_value"], 4)
    0.3865
    """
    alpha = check_alpha(alpha)
    names = column_names(scores, "scores", minimum=2)
    if len(names) > 50:
        raise AnalysisError("invalid_spec", "roccomp compares at most 50 scores.")
    is_positive, columns, event, dropped = _sample(data, y, names, positive, missing)
    n1 = int(is_positive.sum())
    n0 = is_positive.numel() - n1
    if n1 < 2 or n0 < 2:
        raise AnalysisError("too_few_observations", "Comparing ROC curves needs at least 2 "
                            "positive and 2 negative cases.")
    parts = [placements(column, is_positive) for column in columns]
    v10 = torch.stack([part[0] for part in parts], dim=1)
    v01 = torch.stack([part[1] for part in parts], dim=1)
    areas = v10.mean(0)
    c10, c01 = v10 - areas, v01 - areas
    covariance = c10.T @ c10 / ((n1 - 1.0) * n1) + c01.T @ c01 / ((n0 - 1.0) * n0)
    k = len(names)
    contrast = torch.cat([-torch.ones((k - 1, 1), dtype=FLOAT), torch.eye(k - 1, dtype=FLOAT)],
                         dim=1)
    difference = contrast @ areas
    spread = contrast @ covariance @ contrast.T
    if not bool((spread.diagonal() > 0).any()):
        raise AnalysisError("no_variation", "The scores order the cases identically, so their "
                            "ROC curves cannot be compared.")
    active = spread.diagonal() > 0
    statistic, df = wald_statistic(difference[active], spread[active][:, active])
    critical = normal_isf(alpha / 2.0)
    se = covariance.diagonal().clamp_min(0.0).sqrt()
    area_rows = [[float(a), float(e), *_interval(float(a), float(e), critical)]
                 for a, e in zip(areas, se)]
    first, second = torch.triu_indices(k, k, offset=1)
    pair_variance = (covariance.diagonal()[first] + covariance.diagonal()[second]
                     - 2.0 * covariance[first, second]).clamp_min(0.0)
    pair_rows, pair_index = [], []
    for i, j, variance in zip(first.tolist(), second.tolist(), pair_variance.tolist()):
        gap = float(areas[i] - areas[j])
        z = gap / math.sqrt(variance) if variance > 0 else None
        pair_rows.append([gap, math.sqrt(variance), z,
                          normal_two_sided(z) if z is not None else None])
        pair_index.append(f"{names[i]} - {names[j]}")
    attrs = {"n": is_positive.numel(), "n_positive": n1, "n_negative": n0, "positive": event,
             "statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df),
             "auc": {name: float(a) for name, a in zip(names, areas)},
             "covariance": covariance.tolist(), "distribution": "chi2", "alpha": alpha,
             "n_dropped": dropped, "label": "Test of equality of ROC areas (DeLong)"}
    return TableSet(
        {"auc": table(area_rows, columns=["auc", "std_error", "ci_low", "ci_high"], index=names),
         "tests": table([[statistic, df, attrs["p_value"]]],
                        columns=["statistic", "df", "p_value"], index=["equality"]),
         "pairwise": table(pair_rows, columns=["difference", "std_error", "z", "p_value"],
                           index=pair_index)},
        title=f"Comparison of ROC curves for {y}", **attrs)
