"""Tests for related samples: signrank, signtest, mcnemar, symmetry, friedman, cochran_q.

Paired tests work on the differences of two columns (or one column minus a
hypothesized median); the k-related-sample tests take the conditions as
columns of a wide table (rows are subjects) and rank within rows.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric import exact as exact_kernels
from openecon.econometrics.nonparametric.common import (
    FLOAT, adjust_p_values, binomial_lower, binomial_upper, check_choice, check_flag,
    check_number, column_names, group_codes, label, midranks, normal_two_sided, numeric,
    procedure, row_midranks, select, square_table,
)
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import wald_statistic

EXACT_SIGNRANK_N = 25


def _differences(data: Any, y: Any, paired_with: Any, mu: Any, missing: str) -> tuple[Tensor, int]:
    if not isinstance(y, str) or (paired_with is not None and not isinstance(paired_with, str)):
        raise AnalysisError("invalid_spec", "y and paired_with must each be the name of one "
                            "numeric column.")
    mu = check_number(mu, "mu")
    names = [y] if paired_with is None else [y, paired_with]
    frame, dropped = select(data, names, missing=missing)
    d = numeric(frame, y) - mu
    if paired_with is not None:
        d = d - numeric(frame, paired_with)
    return d, dropped


def _title(what: str, y: str, paired_with: str | None, mu: float) -> str:
    target = f"{y} - {paired_with}" if paired_with is not None else y
    return f"{what}: {target} = {mu:g}"


@procedure
def signrank(data: Any, y: str, paired_with: str | None = None, *, mu: float = 0.0,
             zero_method: str = "drop", exact: bool | None = None, missing: str = "drop"):
    """Wilcoxon matched-pairs signed-rank test.

    Tests H0: the differences d = y - paired_with - mu (or y - mu for one
    sample) are symmetric about zero. The |d| are ranked (midranks) and
    T+ / T- are the rank sums of the positive / negative differences.

    Zero differences (``zero_method``):

    * ``"drop"`` (Wilcoxon's rule, SPSS): zeros are removed before ranking.
    * ``"adjust"`` (Pratt's rule, Stata ``signrank``): zeros are ranked with
      the other |d| and then left out of both sums, which lowers the mean and
      variance of T+.

    In both cases, with r_j the ranks of the nonzero differences,

        E(T+) = sum_j r_j / 2,    Var(T+) = sum_j r_j^2 / 4,
        z = (T+ - E(T+)) / sqrt(Var(T+)).

    Without zeros and ties this is n(n+1)/4 and n(n+1)(2n+1)/24; the sum of
    squared midranks carries the tie correction sum (t^3 - t)/48. No continuity
    correction (SPSS and Stata). z has the sign of T+ - E(T+) (SPSS prints
    the z of the smaller rank sum, which is -|z|).

    Exact test: the distribution of T+ under random signs is computed by
    dynamic programming. ``exact=None`` uses it when there are at most 25
    nonzero differences and no ties among them; ``exact=True`` forces it (with
    ties: the exact conditional distribution given the midranks). The
    two-sided exact p is P(|T+ - E| >= |t - E|).

    Parameters
    ----------
    y : numeric column. paired_with : optional second numeric column.
    mu : hypothesized median of the difference (default 0).
    zero_method : "drop" or "adjust".  exact : None, True or False.
    missing : "drop" (pairwise-complete rows) or "raise".

    Returns
    -------
    TableSet with ``ranks`` (rows positive, negative, zero: n, rank_sum,
    mean_rank) and ``tests`` (rows z and, when computed, exact). ``attrs``:
    n, n_positive, n_negative, n_zero, t_plus, t_minus, statistic (the smaller
    rank sum, SPSS's T), z, p_value, exact, p_exact, p_exact_lower,
    p_exact_upper (tails of T+), effect_r = z / sqrt(n ranked).

    Stata: ``signrank y = x`` (zero_method="adjust"). SPSS: ``NPAR TESTS
    /WILCOXON=y WITH x (PAIRED)`` (zero_method="drop").

    Example
    -------
    >>> import openecon as oe
    >>> data = {"before": [10, 12, 9, 14, 11, 13], "after": [12, 15, 8, 18, 16, 19]}
    >>> round(oe.signrank(data, "after", "before").attrs["p_exact"], 4)
    0.0625
    """
    check_choice(zero_method, "zero_method", ("drop", "adjust"))
    exact = check_flag(exact, "exact", optional=True)
    d, dropped = _differences(data, y, paired_with, mu, missing)
    nonzero = d != 0
    n_zero = int((~nonzero).sum())
    if n_zero == d.numel():
        raise AnalysisError("no_variation", "Every difference is zero, so the signed-rank test "
                            "is undefined.")
    if zero_method == "drop":
        d = d[nonzero]
        nonzero = torch.ones(d.numel(), dtype=torch.bool)
    ranks, _ = midranks(d.abs())
    used = ranks[nonzero]
    signs = torch.sign(d[nonzero])
    positive = signs > 0
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    t_plus, t_minus = float(used[positive].sum()), float(used[~positive].sum())
    mean = float(used.sum()) / 2.0
    variance = float((used * used).sum()) / 4.0
    z = (t_plus - mean) / math.sqrt(variance)
    has_ties = bool(torch.unique(used).numel() < used.numel())
    n_ranked = d.numel()
    attrs: dict[str, Any] = {
        "n": n_positive + n_negative + n_zero, "n_positive": n_positive,
        "n_negative": n_negative, "n_zero": n_zero, "t_plus": t_plus, "t_minus": t_minus,
        "statistic": min(t_plus, t_minus), "expected": mean, "variance": variance, "z": z,
        "p_value": normal_two_sided(z), "zero_method": zero_method, "ties": has_ties,
        "effect_r": z / math.sqrt(n_ranked),
    }
    rows, index = [[z, attrs["p_value"]]], ["z"]
    use_exact = exact if exact is not None else (used.numel() <= EXACT_SIGNRANK_N
                                                and not has_ties)
    if use_exact:
        halves = bool((used != used.round()).any())
        scale = 2 if halves else 1
        scores = (used * scale).round().to(torch.int64)
        pmf = exact_kernels.signed_sum_distribution(scores)
        lower, upper, two_sided = exact_kernels.tail_probabilities(
            pmf, int(round(t_plus * scale)), mean * scale)
        attrs.update({"p_exact": two_sided, "p_exact_lower": lower, "p_exact_upper": upper})
        rows.append([attrs["statistic"], two_sided])
        index.append("exact")
    attrs.update({"exact": bool(use_exact), "n_dropped": dropped,
                  "label": "Wilcoxon signed-rank test"})
    rank_rows = [[n_positive, t_plus, t_plus / n_positive if n_positive else None],
                 [n_negative, t_minus, t_minus / n_negative if n_negative else None],
                 [n_zero, None, None]]
    return TableSet(
        {"ranks": table(rank_rows, columns=["n", "rank_sum", "mean_rank"],
                        index=["positive", "negative", "zero"]),
         "tests": table(rows, columns=["statistic", "p_value"], index=index)},
        title=_title("Wilcoxon signed-rank test", y, paired_with, float(mu)), **attrs)


@procedure
def signtest(data: Any, y: str, paired_with: str | None = None, *, mu: float = 0.0,
             missing: str = "drop"):
    """Sign test of the median of the differences (exact binomial).

    With n+ positive and n- negative differences d = y - paired_with - mu
    (zeros are discarded) and n = n+ + n-, the number of positive signs is
    Binomial(n, 1/2) under H0: median(d) = 0.

    * ``p_value`` (two-sided) = min(1, 2 P(X >= max(n+, n-))).
    * ``p_positive`` = P(X >= n+) for H1: median > 0;
      ``p_negative`` = P(X >= n-) for H1: median < 0 (Stata's one-sided lines).
    * ``z`` = (|n+ - n-| - 1) / sqrt(n) with the sign of n+ - n-, the
      continuity-corrected normal approximation SPSS prints for n > 25, with
      its two-sided p ``p_value_normal``.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column.
    paired_with : optional second numeric column (the test is on y - paired_with).
    mu : hypothesized median of the difference (default 0).
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``signs`` (n per sign: positive, negative, zero) and
    ``tests`` (rows two_sided, positive, negative, normal; columns statistic,
    p_value). ``attrs``: n (nonzero), n_positive, n_negative, n_zero, p_value,
    p_positive, p_negative, z, p_value_normal.

    Stata: ``signtest y = x``. SPSS: ``NPAR TESTS /SIGN=y WITH x (PAIRED)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"before": [10, 12, 9, 14, 11, 13], "after": [12, 15, 8, 18, 16, 19]}
    >>> round(oe.signtest(data, "after", "before").attrs["p_value"], 4)
    0.2188
    """
    d, dropped = _differences(data, y, paired_with, mu, missing)
    n_positive, n_negative = int((d > 0).sum()), int((d < 0).sum())
    n_zero = d.numel() - n_positive - n_negative
    n = n_positive + n_negative
    if n == 0:
        raise AnalysisError("no_variation", "Every difference is zero, so the sign test is "
                            "undefined.")
    p_positive = binomial_upper(n_positive, n, 0.5)
    p_negative = binomial_upper(n_negative, n, 0.5)
    p_two = min(1.0, 2.0 * binomial_upper(max(n_positive, n_negative), n, 0.5))
    gap = n_positive - n_negative
    z = math.copysign(max(0.0, abs(gap) - 1.0), gap) / math.sqrt(n)
    attrs = {"n": n, "n_positive": n_positive, "n_negative": n_negative, "n_zero": n_zero,
             "p_value": p_two, "p_positive": p_positive, "p_negative": p_negative, "z": z,
             "p_value_normal": normal_two_sided(z), "n_dropped": dropped, "label": "Sign test"}
    signs = table({"n": [n_positive, n_negative, n_zero]}, index=["positive", "negative", "zero"])
    tests = table([[max(n_positive, n_negative), p_two], [n_positive, p_positive],
                   [n_negative, p_negative], [z, attrs["p_value_normal"]]],
                  columns=["statistic", "p_value"],
                  index=["two_sided", "positive", "negative", "normal"])
    return TableSet({"signs": signs, "tests": tests},
                    title=_title("Sign test", y, paired_with, float(mu)), **attrs)


def _paired_table(data: Any, a: Any, b: Any, missing: str) -> tuple[Tensor, list[Any], int]:
    if not isinstance(a, str) or not isinstance(b, str):
        raise AnalysisError("invalid_spec", "a and b must each be the name of one column.")
    frame, dropped = select(data, [a, b], missing=missing)
    counts, levels = square_table(frame, a, b)
    return counts, levels, dropped


def _count_table(counts: Tensor, levels: list[Any], a: str, b: str) -> pd.DataFrame:
    names = [str(level) for level in levels]
    body = torch.cat([counts, counts.sum(1, keepdim=True)], dim=1)
    body = torch.cat([body, body.sum(0, keepdim=True)], dim=0).to(torch.int64)
    return table(body.tolist(), columns=[f"{b}={name}" for name in names] + ["total"],
                 index=[f"{a}={name}" for name in names] + ["total"])


@procedure
def mcnemar(data: Any, a: str, b: str, *, missing: str = "drop"):
    """McNemar's test of marginal homogeneity for two paired binary variables.

    From the 2x2 table of ``a`` (rows) by ``b`` (columns) only the discordant
    counts matter: n12 (a at the first level, b at the second) and n21.

        chi2 = (n12 - n21)^2 / (n12 + n21)                     (Stata ``mcc``)
        chi2_continuity = (|n12 - n21| - 1)^2 / (n12 + n21)    (SPSS, n12 + n21 > 25;
                                                                 0 when n12 = n21, as in R)
        exact p = min(1, 2 P(X <= min(n12, n21))),  X ~ Binomial(n12 + n21, 1/2)

    each chi2 on 1 degree of freedom. SPSS prints the exact binomial p when
    there are at most 25 discordant pairs; all three are returned here.
    Without discordant pairs the chi-squares are undefined and the exact p is 1.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    a, b : two columns classifying the same subjects into the same two
        categories (any scalar type; rows of the table are ``a``).
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``counts`` (the 2x2 table with margins) and ``tests`` (rows
    mcnemar, continuity, exact; columns statistic, df, p_value). ``attrs``: n,
    n12, n21, statistic, p_value (uncorrected chi2), statistic_continuity,
    p_value_continuity, p_exact, levels.

    Stata: ``mcc a b`` / ``symmetry a b``. SPSS: ``NPAR TESTS /MCNEMAR=a WITH
    b (PAIRED)`` or ``CROSSTABS /STATISTICS=MCNEMAR``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"pre": [1, 1, 0, 0, 1, 0, 1, 1], "post": [1, 0, 0, 0, 0, 0, 0, 1]}
    >>> round(oe.mcnemar(data, "pre", "post").attrs["p_exact"], 4)
    0.25
    """
    counts, levels, dropped = _paired_table(data, a, b, missing)
    if len(levels) != 2:
        raise AnalysisError("not_binary", f"McNemar's test needs two binary columns with the "
                            f"same two values; '{a}' and '{b}' take {len(levels)} distinct "
                            "value(s). Use oe.symmetry for more than two categories.")
    n12, n21 = float(counts[0, 1]), float(counts[1, 0])
    discordant = n12 + n21
    attrs: dict[str, Any] = {"n": int(counts.sum()), "n12": int(n12), "n21": int(n21),
                             "levels": levels, "df": 1, "distribution": "chi2"}
    if discordant > 0:
        chi2 = (n12 - n21) ** 2 / discordant
        corrected = max(0.0, abs(n12 - n21) - 1.0) ** 2 / discordant
        p_exact = min(1.0, 2.0 * binomial_lower(min(n12, n21), discordant, 0.5))
        attrs.update({"statistic": chi2, "p_value": chi2_sf(chi2, 1),
                      "statistic_continuity": corrected,
                      "p_value_continuity": chi2_sf(corrected, 1), "p_exact": p_exact})
    else:
        attrs.update({"statistic": None, "p_value": None, "statistic_continuity": None,
                      "p_value_continuity": None, "p_exact": 1.0,
                      "notes": ["There are no discordant pairs: the chi-square statistics are "
                                "undefined and the exact p-value is 1."]})
    attrs.update({"n_dropped": dropped, "label": "McNemar's test"})
    tests = table([[attrs["statistic"], 1, attrs["p_value"]],
                   [attrs["statistic_continuity"], 1, attrs["p_value_continuity"]],
                   [min(n12, n21), None, attrs["p_exact"]]],
                  columns=["statistic", "df", "p_value"],
                  index=["mcnemar", "continuity", "exact"])
    return TableSet({"counts": _count_table(counts, levels, a, b), "tests": tests},
                    title=f"McNemar's test: {a} vs {b}", **attrs)


@procedure
def symmetry(data: Any, a: str, b: str, *, missing: str = "drop"):
    """Symmetry (McNemar-Bowker) and marginal-homogeneity tests for a square K x K table.

    ``a`` and ``b`` are two classifications of the same subjects into the same
    K categories (the union of the values of both columns).

    * Bowker's test of symmetry: chi2 = sum_{i<j} (n_ij - n_ji)^2 / (n_ij + n_ji)
      over the pairs with n_ij + n_ji > 0; its df is the number of such pairs
      (Stata ``symmetry``, SPSS McNemar-Bowker). For K = 2 it is McNemar's
      uncorrected chi2.
    * Stuart-Maxwell test of marginal homogeneity: with d_i = n_i. - n_.i and
      V_ii = n_i. + n_.i - 2 n_ii, V_ij = -(n_ij + n_ji), chi2 = d' V^- d with
      df = rank(V) (K - 1 when every category has discordant pairs).
    * When both columns are numeric, the ordinal marginal-homogeneity z test
      of SPSS (``NPAR TESTS /MH``): z = sum_k (a_k - b_k) / sqrt(sum_k (a_k -
      b_k)^2), the standardized sum of the score differences over the
      discordant pairs.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    a, b : two columns classifying the same subjects (rows of the table are ``a``).
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``counts`` (K x K with margins) and ``tests`` (rows symmetry,
    marginal_homogeneity and, for numeric columns, marginal_homogeneity_ordinal;
    columns statistic, df, p_value). ``attrs``: n, k, statistic, df, p_value
    (Bowker), mh_statistic, mh_df, mh_p_value, mh_z, mh_z_p_value.

    Stata: ``symmetry a b``. SPSS: ``CROSSTABS /STATISTICS=MCNEMAR``, ``NPAR
    TESTS /MH``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"rater1": [1, 1, 2, 2, 3, 3, 1, 2, 3, 3], "rater2": [1, 2, 2, 3, 3, 3, 2, 3, 1, 3]}
    >>> round(oe.symmetry(data, "rater1", "rater2").attrs["statistic"], 3)
    5.0
    """
    counts, levels, dropped = _paired_table(data, a, b, missing)
    k = len(levels)
    if k < 2:
        raise AnalysisError("no_variation", f"'{a}' and '{b}' take a single value, so there is "
                            "nothing to test.")
    upper, lower = torch.triu_indices(k, k, offset=1)
    above, below = counts[upper, lower], counts[lower, upper]
    pair_total = above + below
    present = pair_total > 0
    df = int(present.sum())
    statistic = float(((above - below)[present] ** 2 / pair_total[present]).sum())
    notes: list[str] = []
    rows = [[statistic, df, chi2_sf(statistic, df) if df else None]]
    index = ["symmetry"]
    attrs: dict[str, Any] = {"n": int(counts.sum()), "k": k, "levels": levels,
                             "statistic": statistic, "df": df,
                             "p_value": chi2_sf(statistic, df) if df else None,
                             "distribution": "chi2"}
    if df == 0:
        notes.append("All observations lie on the diagonal: the table is exactly symmetric "
                     "and no test statistic can be computed.")
    else:
        d = counts.sum(1) - counts.sum(0)
        v = -(counts + counts.T)
        v.diagonal().copy_(counts.sum(1) + counts.sum(0) - 2.0 * counts.diagonal())
        active = v.diagonal() > 0
        mh, rank = wald_statistic(d[active], v[active][:, active])
        attrs.update({"mh_statistic": mh, "mh_df": rank, "mh_p_value": chi2_sf(mh, rank)})
        rows.append([mh, rank, attrs["mh_p_value"]])
        index.append("marginal_homogeneity")
        numeric_levels = all(isinstance(level, (int, float)) and not isinstance(level, bool)
                             for level in levels)
        if numeric_levels:
            scores = torch.tensor([float(level) for level in levels], dtype=FLOAT)
            gap = scores[:, None] - scores[None, :]
            total_gap, squares = float((counts * gap).sum()), float((counts * gap * gap).sum())
            z = total_gap / math.sqrt(squares)
            attrs.update({"mh_z": z, "mh_z_p_value": normal_two_sided(z)})
            rows.append([z, None, attrs["mh_z_p_value"]])
            index.append("marginal_homogeneity_ordinal")
    attrs.update({"n_dropped": dropped, "label": "Symmetry and marginal homogeneity tests"})
    if notes:
        attrs["notes"] = notes
    return TableSet({"counts": _count_table(counts, levels, a, b),
                     "tests": table(rows, columns=["statistic", "df", "p_value"], index=index)},
                    title=f"Symmetry tests: {a} vs {b}", **attrs)


def _wide(data: Any, columns: Any, missing: str, minimum: int) -> tuple[Any, list[str], int]:
    names = column_names(columns, "columns", minimum=minimum)
    frame, dropped = select(data, names, missing=missing)
    return frame, names, dropped


@procedure
def friedman(data: Any, columns: list[str], *, pairwise: bool = False,
             adjust: str = "bonferroni", missing: str = "drop"):
    """Friedman's two-way analysis of variance by ranks and Kendall's W.

    ``columns`` are k related measurements (conditions) of the same n subjects
    (rows). Values are ranked within each row (midranks); with R_j the rank
    sum of column j and T_i = sum (t^3 - t) over the tie groups of row i,

        chi2 = [12 / (n k (k + 1)) * sum_j R_j^2 - 3 n (k + 1)]
               / [1 - sum_i T_i / (n k (k^2 - 1))]            ~ chi2(k - 1),
        W = chi2 / (n (k - 1))                                 (Kendall's coefficient).

    ``pairwise=True`` adds the Dunn-type comparisons of SPSS: for columns i, j
    z = (Rbar_i - Rbar_j) / sqrt(k (k + 1) / (6 n)) with two-sided normal p
    and the adjustment ``adjust`` ("bonferroni", "holm" or "none").

    Missing values: rows are deleted listwise over ``columns``.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records (one row per subject).
    columns : list of k >= 2 numeric columns, one per condition.
    pairwise : add the ``pairwise`` table.
    adjust : "bonferroni", "holm" or "none".
    missing : "drop" (a row with a missing value in any of ``columns`` is
        deleted and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``ranks`` (rank_sum, mean_rank per column), ``tests`` (row
    friedman: statistic, df, p_value) and optionally ``pairwise``. ``attrs``:
    n, k, statistic, df, p_value, kendall_w.

    SPSS: ``NPAR TESTS /FRIEDMAN=v1 v2 v3`` and ``/KENDALL``. Stata: community
    command ``friedman`` (transposed layout); ``emh`` for the general case.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"a": [1, 2, 1, 3, 2], "b": [2, 3, 3, 4, 3], "c": [3, 1, 2, 5, 4]}
    >>> round(oe.friedman(data, ["a", "b", "c"]).attrs["statistic"], 3)
    4.8
    """
    check_flag(pairwise, "pairwise")
    check_choice(adjust, "adjust", ("bonferroni", "holm", "none"))
    frame, names, dropped = _wide(data, columns, missing, 2)
    x = torch.stack([numeric(frame, name) for name in names], dim=1)
    n, k = x.shape
    ranks, ties = row_midranks(x)
    sums = ranks.sum(0)
    correction = 1.0 - float(ties.sum()) / (n * k * (k * k - 1.0))
    if correction <= 0.0:
        raise AnalysisError("no_variation", "Every row is constant across the columns, so "
                            "Friedman's statistic is undefined.")
    raw = 12.0 / (n * k * (k + 1.0)) * float((sums * sums).sum()) - 3.0 * n * (k + 1.0)
    statistic = max(raw, 0.0) / correction
    df = k - 1
    attrs: dict[str, Any] = {"n": n, "k": k, "statistic": statistic, "df": df,
                             "p_value": chi2_sf(statistic, df),
                             "kendall_w": statistic / (n * (k - 1.0)), "distribution": "chi2",
                             "n_dropped": dropped, "label": "Friedman test"}
    mean_ranks = sums / n
    result = {"ranks": table({"rank_sum": sums.tolist(), "mean_rank": mean_ranks.tolist()},
                             index=names),
              "tests": table([[statistic, df, attrs["p_value"]]],
                             columns=["statistic", "df", "p_value"], index=["friedman"])}
    if pairwise:
        if k > 200:
            raise AnalysisError("too_many_groups", "Pairwise comparisons are limited to 200 "
                                "columns; use pairwise=False.")
        first, second = torch.triu_indices(k, k, offset=1)
        se = math.sqrt(k * (k + 1.0) / (6.0 * n))
        difference = mean_ranks[first] - mean_ranks[second]
        z = (difference / se).tolist()
        p = [normal_two_sided(value) for value in z]
        result["pairwise"] = table(
            {"column_1": [names[i] for i in first.tolist()],
             "column_2": [names[j] for j in second.tolist()],
             "mean_rank_diff": difference.tolist(), "std_error": [se] * len(z), "z": z,
             "p_value": p, "p_adjusted": adjust_p_values(p, adjust)},
            index=[f"{names[i]} - {names[j]}" for i, j in zip(first.tolist(), second.tolist())])
        attrs["adjust"] = adjust
    return TableSet(result, title=f"Friedman test: {', '.join(names)}", **attrs)


@procedure
def cochran_q(data: Any, columns: list[str], *, positive: Any = None, missing: str = "drop"):
    """Cochran's Q test for k related binary samples.

    ``columns`` are k binary measurements of the same n subjects. With C_j the
    number of successes in column j and L_i the number of successes of
    subject i,

        Q = (k - 1) [k sum_j C_j^2 - (sum_j C_j)^2] / [k sum_i L_i - sum_i L_i^2]

    is chi2(k - 1) under H0: the success probabilities are equal. For k = 2 it
    is McNemar's uncorrected chi2. The columns must share the same two values;
    the success is the larger one (1 for 0/1 data) unless ``positive`` names it.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records (one row per subject).
    columns : list of k >= 2 binary columns sharing the same two values.
    positive : the value counted as a success (default: the larger value).
    missing : "drop" (a row with a missing value in any of ``columns`` is
        deleted and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``frequencies`` (per column: success, failure, proportion)
    and ``tests`` (row cochran_q: statistic, df, p_value). ``attrs``: n, k,
    statistic, df, p_value, positive.

    SPSS: ``NPAR TESTS /COCHRAN=v1 v2 v3``. Stata: community command
    ``cochran``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"t1": [1, 1, 0, 1, 0, 1], "t2": [1, 0, 0, 1, 0, 0], "t3": [0, 0, 0, 1, 0, 0]}
    >>> round(oe.cochran_q(data, ["t1", "t2", "t3"]).attrs["statistic"], 3)
    4.667
    """
    frame, names, dropped = _wide(data, columns, missing, 2)
    stacked = pd.concat([frame[name] for name in names], ignore_index=True)
    codes, levels = group_codes(stacked)
    if positive is not None:
        positive = label(positive)
        if positive not in levels:
            raise AnalysisError("invalid_option", f"positive={positive!r} does not occur in the "
                                "columns.")
    if len(levels) > 2 or (len(levels) == 1 and positive is None
                           and levels[0] not in (0, 1, True, False)):
        raise AnalysisError("not_binary", "Cochran's Q needs binary columns sharing the same "
                            f"two values; found {len(levels)} distinct value(s).")
    if positive is None:
        positive = levels[-1] if len(levels) == 2 else (1 if levels[0] in (1, True) else None)
    n, k = len(frame), len(names)
    success = (codes == levels.index(positive)).to(FLOAT) if positive in levels \
        else torch.zeros(codes.numel(), dtype=FLOAT)
    x = success.reshape(k, n).T
    column_totals, row_totals = x.sum(0), x.sum(1)
    grand = float(column_totals.sum())
    denominator = k * grand - float((row_totals * row_totals).sum())
    if denominator <= 0.0:
        raise AnalysisError("no_variation", "Every subject has the same response in all "
                            "columns, so Cochran's Q is undefined.")
    statistic = (k - 1.0) * (k * float((column_totals ** 2).sum()) - grand * grand) / denominator
    df = k - 1
    attrs = {"n": n, "k": k, "statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df),
             "positive": positive, "distribution": "chi2", "n_dropped": dropped,
             "label": "Cochran's Q test"}
    frequencies = table({"success": column_totals.to(torch.int64).tolist(),
                         "failure": (n - column_totals).to(torch.int64).tolist(),
                         "proportion": (column_totals / n).tolist()}, index=names)
    tests = table([[statistic, df, attrs["p_value"]]], columns=["statistic", "df", "p_value"],
                  index=["cochran_q"])
    return TableSet({"frequencies": frequencies, "tests": tests},
                    title=f"Cochran's Q test: {', '.join(names)}", **attrs)
