"""Rank tests for independent samples: ranksum, kwallis, median_test, jonckheere.

All four pool the groups, rank once by sorting (midranks for ties) and work
on group rank sums obtained with ``index_add_``; nothing is O(n^2) except the
optional Hodges-Lehmann estimate, which is limited to 1e6 pairwise differences.
The Jonckheere-Terpstra pair counts come from the group-by-value table when
there are few groups and from an O(n log n) sort-based kernel otherwise.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric import exact as exact_kernels
from openecon.econometrics.nonparametric import tables as table_kernels
from openecon.econometrics.nonparametric.common import (
    FLOAT, adjust_p_values, check_alpha, check_choice, check_flag, group_codes, group_count,
    group_sum, label, midranks, normal_two_sided, numeric, procedure, select, tie_term,
    two_groups,
)
from openecon.engines.distributions import chi2_sf, normal_isf, normal_sf

HODGES_LEHMANN_PAIRS = 1_000_000
EXACT_PRODUCT = 400


def _pooled(data: Any, y: Any, by: Any, missing: str) -> tuple[Any, Tensor, int]:
    if not isinstance(y, str) or not isinstance(by, str):
        raise AnalysisError("invalid_spec", "y and by must each be the name of one column.")
    frame, dropped = select(data, [y, by], missing=missing)
    return frame, numeric(frame, y), dropped


def _hodges_lehmann(first: Tensor, second: Tensor, alpha: float, pmf: Tensor | None,
                    sd: float) -> dict[str, Any]:
    """Median of the pairwise differences and its distribution-free confidence interval.

    The interval is (D_(q), D_(n1 n2 + 1 - q)) on the ordered differences. q
    comes from the exact null distribution of U when it is available (the
    smallest q with P(U <= q) >= alpha/2, at least 1), otherwise from the
    normal approximation q = floor(n1 n2 / 2 + 1/2 - z sd).
    """
    pairs = first.numel() * second.numel()
    differences = torch.sort((first[:, None] - second[None, :]).reshape(-1)).values
    middle = pairs // 2
    estimate = float(differences[middle]) if pairs % 2 else \
        0.5 * float(differences[middle - 1] + differences[middle])
    if pmf is not None:
        cdf = torch.cumsum(pmf, 0)
        q = int(torch.searchsorted(cdf, torch.tensor(alpha / 2.0 * (1.0 - 1e-12), dtype=FLOAT)))
        method = "exact"
    else:
        q = int(math.floor(pairs / 2.0 + 0.5 - normal_isf(alpha / 2.0) * sd))
        method = "normal"
    q = min(max(q, 1), middle if middle else 1)
    return {"hl_estimate": estimate, "hl_ci_low": float(differences[q - 1]),
            "hl_ci_high": float(differences[pairs - q]), "hl_ci_method": method}


@procedure
def ranksum(data: Any, y: str, by: str, *, exact: bool | None = None, alpha: float = 0.05,
            missing: str = "drop"):
    """Wilcoxon rank-sum (Mann-Whitney U) test for two independent samples.

    Tests H0: the two groups of ``by`` have the same distribution of ``y``.
    The pooled sample is ranked (midranks for ties). With T1 the rank sum of
    the first group (levels of ``by`` in sorted order), n1, n2 the group sizes
    and N = n1 + n2:

        U1 = T1 - n1 (n1 + 1) / 2          (pairs with y1 > y2, ties counting 1/2)
        E(T1) = n1 (N + 1) / 2
        Var(T1) = n1 n2 / 12 * [ (N + 1) - sum_j (t_j^3 - t_j) / (N (N - 1)) ]
        z = (T1 - E(T1)) / sqrt(Var(T1))

    ``z`` carries no continuity correction, as in SPSS (NPAR TESTS M-W) and
    Stata (``ranksum``); ``z_continuity`` moves |T1 - E| by 1/2 towards zero
    (the default of R and SciPy). Both two-sided normal p-values are reported.

    Exact test. The permutation distribution of the rank sum is computed by
    dynamic programming over its support. ``exact=None`` computes it when
    n1 * n2 <= 400 and there are no ties (SPSS's rule); ``exact=True`` forces
    it, and with ties it is then the exact conditional distribution of the
    midrank sum (SPSS instead prints a p "not corrected for ties"). The
    two-sided exact p is P(|T - E| >= |t - E|).

    Effect sizes: ``effect_r = z / sqrt(N)`` and ``porder = U1 / (n1 n2)``, the
    probability that a value of the first group exceeds one of the second
    plus half the probability of a tie (Stata's ``porder``).

    Hodges-Lehmann: the median of the n1 * n2 differences (first - second
    group) with a distribution-free 1 - ``alpha`` confidence interval, when
    n1 * n2 <= 1,000,000.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome column.
    by : grouping column with exactly two distinct values.
    exact : None (rule above), True or False.
    alpha : 1 - confidence level of the Hodges-Lehmann interval.
    missing : "drop" (listwise deletion over y and by) or "raise".

    Returns
    -------
    TableSet with ``ranks`` (per group: n, rank_sum, mean_rank, expected) and
    ``tests`` (rows z, z_continuity and, when computed, exact; columns
    statistic, p_value). ``attrs``: n, n1, n2, groups, u, u1, u2, w, z, p_value
    (asymptotic, uncorrected), z_continuity, p_value_continuity, exact, p_exact,
    p_exact_lower, p_exact_upper (tails of T1), effect_r, porder, hl_estimate,
    hl_ci_low, hl_ci_high, ties, n_dropped.

    Stata: ``ranksum y, by(group) [exact] [porder]``. SPSS: ``NPAR TESTS /M-W=
    y BY group(a b)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"wage": [12, 15, 9, 20, 22, 18, 25, 30], "union": [0, 0, 0, 0, 1, 1, 1, 1]}
    >>> result = oe.ranksum(data, "wage", "union")
    >>> round(result.attrs["p_exact"], 4)
    0.0571
    """
    alpha = check_alpha(alpha)
    exact = check_flag(exact, "exact", optional=True)
    frame, x, dropped = _pooled(data, y, by, missing)
    codes, levels = two_groups(frame, by, "The rank-sum test")
    counts = group_count(codes, 2)
    n1, n2 = int(counts[0]), int(counts[1])
    total = n1 + n2
    ranks, tie_sizes = midranks(x)
    sums = group_sum(ranks, codes, 2)
    t1 = float(sums[0])
    ties = tie_term(tie_sizes)
    has_ties = ties > 0
    expected = n1 * (total + 1) / 2.0
    unadjusted = n1 * n2 * (total + 1) / 12.0
    variance = unadjusted - n1 * n2 * ties / (12.0 * total * (total - 1))
    if variance <= 0.0:
        raise AnalysisError("no_variation", f"All values of '{y}' are equal, so the rank-sum "
                            "statistic has no variance.")
    sd = math.sqrt(variance)
    z = (t1 - expected) / sd
    shift = abs(t1 - expected)
    z_cc = math.copysign(max(0.0, shift - 0.5), t1 - expected) / sd
    u1 = t1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    attrs: dict[str, Any] = {
        "n": total, "n1": n1, "n2": n2, "groups": levels, "u": min(u1, u2), "u1": u1, "u2": u2,
        "w": t1 if n1 <= n2 else float(sums[1]), "z": z, "p_value": normal_two_sided(z),
        "z_continuity": z_cc, "p_value_continuity": normal_two_sided(z_cc),
        "variance_unadjusted": unadjusted, "variance": variance, "ties": has_ties,
        "effect_r": z / math.sqrt(total), "porder": u1 / (n1 * n2),
    }
    rows = [[z, attrs["p_value"]], [z_cc, attrs["p_value_continuity"]]]
    index = ["z", "z_continuity"]
    use_exact = exact if exact is not None else (n1 * n2 <= EXACT_PRODUCT and not has_ties)
    pmf_u = None
    if use_exact:
        small = 0 if n1 <= n2 else 1
        m = min(n1, n2)
        scale = 2 if has_ties else 1
        scores = (ranks * scale).round().to(torch.int64)
        pmf = exact_kernels.rank_sum_distribution(scores, m)
        observed = int(round(float(sums[small]) * scale))
        lower, upper, two_sided = exact_kernels.tail_probabilities(
            pmf, observed, scale * m * (total + 1) / 2.0)
        if small == 1:
            lower, upper = upper, lower
        attrs.update({"p_exact": two_sided, "p_exact_lower": lower, "p_exact_upper": upper})
        rows.append([attrs["u"], two_sided])
        index.append("exact")
        if not has_ties:
            pmf_u = pmf[m * (m + 1) // 2:]
    attrs["exact"] = bool(use_exact)
    if n1 * n2 <= HODGES_LEHMANN_PAIRS:
        attrs.update(_hodges_lehmann(x[codes == 0], x[codes == 1], alpha, pmf_u, sd))
        attrs["alpha"] = alpha
    attrs["n_dropped"] = dropped
    attrs["label"] = "Wilcoxon rank-sum (Mann-Whitney) test"
    rank_table = table(
        [[n1, t1, t1 / n1, expected],
         [n2, float(sums[1]), float(sums[1]) / n2, n2 * (total + 1) / 2.0]],
        columns=["n", "rank_sum", "mean_rank", "expected"], index=[str(v) for v in levels])
    tests = table(rows, columns=["statistic", "p_value"], index=index)
    return TableSet({"ranks": rank_table, "tests": tests},
                    title=f"Two-sample Wilcoxon rank-sum (Mann-Whitney) test: {y} by {by}", **attrs)


def _groups(frame: Any, by: str, minimum: int, what: str) -> tuple[Tensor, list[Any], Tensor]:
    codes, levels = group_codes(frame[by])
    if len(levels) < minimum:
        raise AnalysisError("invalid_groups", f"{what} needs at least {minimum} groups, but "
                            f"column '{by}' has {len(levels)} distinct value(s).")
    return codes, levels, group_count(codes, len(levels))


@procedure
def kwallis(data: Any, y: str, by: str, *, pairwise: bool = False, adjust: str = "bonferroni",
            missing: str = "drop"):
    """Kruskal-Wallis equality-of-populations rank test for k independent samples.

    With R_j the sum of the pooled midranks in group j (size n_j) and
    N = sum n_j:

        H = 12 / (N (N + 1)) * sum_j R_j^2 / n_j - 3 (N + 1)
        H_ties = H / (1 - sum_g (t_g^3 - t_g) / (N^3 - N))

    Both are referred to chi2(k - 1); Stata prints both lines, SPSS prints the
    tie-corrected one, which is the ``statistic`` / ``p_value`` reported in
    ``attrs``. Effect sizes: ``epsilon_squared = H_ties / (N - 1)`` and
    ``eta_squared = (H_ties - k + 1) / (N - k)``.

    ``pairwise=True`` adds Dunn's (1964) post hoc comparisons as SPSS prints
    them: for groups i and j

        z = (Rbar_i - Rbar_j) / sqrt( [N (N + 1) / 12 - sum_g (t_g^3 - t_g) / (12 (N - 1))]
                                       * (1 / n_i + 1 / n_j) )

    with a two-sided normal p and the adjustment ``adjust``: "bonferroni" (p
    times the number of comparisons, SPSS), "holm" (step-down) or "none".

    Parameters
    ----------
    data, y, by : the table, the numeric outcome and the grouping column (k >= 2 groups).
    pairwise : add the ``pairwise`` table.
    adjust : "bonferroni", "holm" or "none".
    missing : "drop" (listwise over y and by) or "raise".

    Returns
    -------
    TableSet with ``ranks`` (n, rank_sum, mean_rank per group), ``tests``
    (rows chi2, chi2_ties; columns statistic, df, p_value) and optionally
    ``pairwise`` (group_1, group_2, mean_rank_diff, std_error, z, p_value,
    p_adjusted). ``attrs``: n, k, statistic, df, p_value (tie-corrected),
    statistic_unadjusted, p_value_unadjusted, epsilon_squared, eta_squared.

    Stata: ``kwallis y, by(group)`` (``dunntest`` is community-contributed).
    SPSS: ``NPAR TESTS /K-W=y BY group(1 k)``; pairwise comparisons from
    ``NPTESTS /INDEPENDENT ... KRUSKAL_WALLIS(COMPARE=PAIRWISE)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"score": [1, 3, 5, 2, 8, 9, 7, 12, 11], "arm": list("aaabbbccc")}
    >>> round(oe.kwallis(data, "score", "arm").attrs["statistic"], 3)
    4.356
    """
    check_flag(pairwise, "pairwise")
    check_choice(adjust, "adjust", ("bonferroni", "holm", "none"))
    frame, x, dropped = _pooled(data, y, by, missing)
    codes, levels, counts = _groups(frame, by, 2, "The Kruskal-Wallis test")
    k, total = len(levels), x.numel()
    ranks, tie_sizes = midranks(x)
    sums = group_sum(ranks, codes, k)
    ties = tie_term(tie_sizes)
    correction = 1.0 - ties / (float(total) ** 3 - total)
    if correction <= 0.0:
        raise AnalysisError("no_variation", f"All values of '{y}' are equal, so the "
                            "Kruskal-Wallis statistic is undefined.")
    h = 12.0 / (total * (total + 1.0)) * float((sums * sums / counts).sum()) - 3.0 * (total + 1.0)
    h = max(h, 0.0)
    h_ties = h / correction
    df = k - 1
    attrs: dict[str, Any] = {
        "n": total, "k": k, "groups": levels, "statistic": h_ties, "df": df,
        "p_value": chi2_sf(h_ties, df), "statistic_unadjusted": h,
        "p_value_unadjusted": chi2_sf(h, df), "epsilon_squared": h_ties / (total - 1.0),
        "eta_squared": (h_ties - k + 1.0) / (total - k) if total > k else None,
        "distribution": "chi2", "n_dropped": dropped,
        "label": "Kruskal-Wallis equality-of-populations rank test",
    }
    mean_ranks = sums / counts
    result = {
        "ranks": table({"n": counts.to(torch.int64).tolist(), "rank_sum": sums.tolist(),
                        "mean_rank": mean_ranks.tolist()}, index=[str(v) for v in levels]),
        "tests": table([[h, df, attrs["p_value_unadjusted"]], [h_ties, df, attrs["p_value"]]],
                       columns=["statistic", "df", "p_value"], index=["chi2", "chi2_ties"]),
    }
    if pairwise:
        if k > 200:
            raise AnalysisError("too_many_groups", "Pairwise comparisons are limited to 200 "
                                "groups; use pairwise=False.")
        sigma2 = total * (total + 1.0) / 12.0 - ties / (12.0 * (total - 1.0))
        first, second = torch.triu_indices(k, k, offset=1)
        difference = mean_ranks[first] - mean_ranks[second]
        se = torch.sqrt(sigma2 * (1.0 / counts[first] + 1.0 / counts[second]))
        z = (difference / se).tolist()
        p = [normal_two_sided(value) for value in z]
        result["pairwise"] = table(
            {"group_1": [levels[i] for i in first.tolist()],
             "group_2": [levels[j] for j in second.tolist()],
             "mean_rank_diff": difference.tolist(), "std_error": se.tolist(), "z": z,
             "p_value": p, "p_adjusted": adjust_p_values(p, adjust)},
            index=[f"{levels[i]} - {levels[j]}" for i, j in zip(first.tolist(), second.tolist())])
        attrs["adjust"] = adjust
    return TableSet(result, title=f"Kruskal-Wallis test: {y} by {by}", **attrs)


@procedure
def median_test(data: Any, y: str, by: str, *, ties: str = "below", missing: str = "drop"):
    """Mood's median test for k independent samples.

    The pooled median M splits every group into values above M and values not
    above M; the resulting 2 x k table is tested with Pearson's chi-square on
    k - 1 degrees of freedom. Values equal to M are counted as not above
    (``ties="below"``, the rule of SPSS and of Stata's ``median``), as above
    (``"above"``) or left out (``"drop"``).

    For two groups the continuity-corrected chi-square and Fisher's exact
    two-sided p are added (SPSS prints Fisher's p for small samples, Stata
    with ``exact``).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome column.
    by : grouping column with k >= 2 distinct values.
    ties : "below" (values equal to the median count as not above), "above" or
        "drop".
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``counts`` (per group: above, not_above, n) and ``tests``
    (rows pearson and, for two groups, continuity and fisher_exact; columns
    statistic, df, p_value). ``attrs``: n, k, median, statistic, df, p_value.

    Stata: ``median y, by(group) [exact medianties(below)]``. SPSS: ``NPAR
    TESTS /MEDIAN=y BY group(1 k)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [1, 2, 3, 4, 5, 6, 7, 8], "g": [0, 0, 0, 1, 0, 1, 1, 1]}
    >>> oe.median_test(data, "y", "g").attrs["statistic"]
    2.0
    """
    check_choice(ties, "ties", ("below", "above", "drop"))
    frame, x, dropped = _pooled(data, y, by, missing)
    codes, levels, _ = _groups(frame, by, 2, "The median test")
    k, total = len(levels), x.numel()
    ordered = torch.sort(x).values
    median = float(ordered[total // 2]) if total % 2 else \
        0.5 * float(ordered[total // 2 - 1] + ordered[total // 2])
    if ties == "drop":
        keep = x != median
        x, codes = x[keep], codes[keep]
    above = x >= median if ties == "above" else x > median
    upper = group_sum(above.to(FLOAT), codes, k)
    counts = torch.stack([upper, group_count(codes, k) - upper])
    if bool((counts.sum(1) == 0).any()):
        raise AnalysisError("no_variation", "No value lies on one side of the pooled median, so "
                            "the median test is undefined. Try another ties= rule.")
    trimmed, _, kept = table_kernels.trim(counts)
    if trimmed.shape[1] < 2:
        raise AnalysisError("invalid_groups", "Fewer than two groups remain after dropping the "
                            "values equal to the median.")
    statistic, df = table_kernels.pearson_chi2(trimmed)
    p_value = chi2_sf(statistic, df)
    rows, index = [[statistic, df, p_value]], ["pearson"]
    attrs: dict[str, Any] = {"n": int(counts.sum()), "k": k, "groups": levels, "median": median,
                             "statistic": statistic, "df": df, "p_value": p_value, "ties": ties,
                             "distribution": "chi2", "n_dropped": dropped,
                             "label": "Median test"}
    if trimmed.shape[1] == 2:
        corrected = table_kernels.yates_chi2(trimmed)
        fisher = table_kernels.fisher_2x2(trimmed)[0]
        rows += [[corrected, 1, chi2_sf(corrected, 1)], [None, None, fisher]]
        index += ["continuity", "fisher_exact"]
        attrs.update({"statistic_continuity": corrected,
                      "p_value_continuity": chi2_sf(corrected, 1), "p_exact": fisher})
    count_table = table({"above": counts[0].to(torch.int64).tolist(),
                         "not_above": counts[1].to(torch.int64).tolist(),
                         "n": counts.sum(0).to(torch.int64).tolist()},
                        index=[str(v) for v in levels])
    return TableSet({"counts": count_table,
                     "tests": table(rows, columns=["statistic", "df", "p_value"], index=index)},
                    title=f"Median test: {y} by {by}", **attrs)


# Largest (groups x distinct values) cross-classification scanned directly; beyond it the
# pair counts come from the sort-based kernel, whose cost does not depend on the group count.
PAIR_TABLE_CELLS = 20_000_000


def ascending_pairs(keys: Tensor) -> int:
    """Number of pairs i < j (in sequence order) with keys[i] < keys[j], in O(n log m).

    ``keys`` are int64 codes in [0, m). Two keys differ first at one binary
    digit; at that digit the earlier element has 0 and the later one 1, and
    both share all higher digits. The digits are processed from the highest
    down: elements with the same higher digits form a contiguous block that
    keeps the original order, the pairs "0 before 1" inside every block are
    counted with one cumulative sum, and the block is then split stably into
    its zeros and its ones. Every level is a handful of O(n) tensor
    operations; equal keys are never counted.
    """
    n = keys.numel()
    if n < 2:
        return 0
    start = torch.zeros(n, dtype=torch.int64)             # first index of the element's block
    last = torch.full((n,), n - 1, dtype=torch.int64)     # last index of the element's block
    position = torch.arange(n, dtype=torch.int64)
    total = 0
    for level in range(max(1, int(keys.max()).bit_length()) - 1, -1, -1):
        bit = (keys >> level) & 1
        zero = 1 - bit
        through = torch.cumsum(zero, 0)                   # zeros up to and including i
        before_block = (through - zero)[start]
        zeros_before = through - zero - before_block      # zeros of the block before i
        total += int((zeros_before * bit).sum())
        zeros_in_block = through[last] - before_block
        target = torch.where(bit == 0, start + zeros_before,
                             start + zeros_in_block + (position - start - zeros_before))
        split = start + zeros_in_block
        new_start = torch.where(bit == 0, start, split)
        new_last = torch.where(bit == 0, split - 1, last)
        keys = torch.empty_like(keys).scatter_(0, target, keys)
        start = torch.empty_like(start).scatter_(0, target, new_start)
        last = torch.empty_like(last).scatter_(0, target, new_last)
    return total


def _pair_counts_by_table(groups: Tensor, value_codes: Tensor, k: int, m: int) -> tuple[float,
                                                                                         float]:
    """(C, D) from the k x m cross-classification, scanned in blocks of value columns."""
    order = torch.argsort(value_codes, stable=True)
    value_codes, groups = value_codes[order], groups[order]
    boundaries = torch.searchsorted(value_codes, torch.arange(m + 1))
    block = max(1, 4_000_000 // k)
    before = torch.zeros(k, dtype=FLOAT)
    concordant = discordant = 0.0
    for low in range(0, m, block):
        high = min(m, low + block)
        rows = slice(int(boundaries[low]), int(boundaries[high]))
        width = high - low
        cell = groups[rows] * width + (value_codes[rows] - low)
        counts = torch.bincount(cell, minlength=k * width).reshape(k, width).to(FLOAT)
        smaller = before[:, None] + torch.cumsum(counts, 1) - counts     # same group, lower value
        lower_groups = torch.cumsum(smaller, 0) - smaller
        higher_groups = smaller.sum(0, keepdim=True) - lower_groups - smaller
        concordant += float((counts * lower_groups).sum())
        discordant += float((counts * higher_groups).sum())
        before += counts.sum(1)
    return concordant, discordant


def _pair_counts_by_sorting(groups: Tensor, value_codes: Tensor, k: int, m: int) -> tuple[float,
                                                                                           float]:
    """(C, D) in O(n log n), whatever the number of groups.

    Order the observations by group and, within a group, by decreasing value.
    An earlier observation with a smaller value then always belongs to a
    lower group, so C is the number of ascending pairs of the value sequence.
    D follows from the pairs in different groups, (N^2 - sum n_g^2) / 2, less
    C and less the pairs in different groups with equal values.
    """
    n = groups.numel()
    order = torch.argsort(groups * m + (m - 1 - value_codes))
    concordant = ascending_pairs(value_codes[order])

    def same(codes: Tensor) -> int:                      # pairs sharing a code
        counts = torch.unique(codes, return_counts=True)[1]
        return int((counts * (counts - 1)).sum()) // 2

    different_groups = n * (n - 1) // 2 - same(groups)
    tied_values = same(value_codes) - same(groups * m + value_codes)
    return float(concordant), float(different_groups - tied_values - concordant)


def ordered_pair_counts(groups: Tensor, values: Tensor, k: int) -> tuple[float, float]:
    """(C, D): pairs with a lower group and a lower value, and with a lower group and a
    higher value.

    With few groups the observations are cross-classified by group and
    distinct value and the counts are accumulated over blocks of value
    columns, O(k * m) for m distinct values. When that table would exceed
    ``PAIR_TABLE_CELLS`` cells (many groups) the counts come from a sort and
    the binary-digit kernel ``ascending_pairs``, O(n log n). Never O(n^2).
    """
    _, value_codes = torch.unique(values, sorted=True, return_inverse=True)
    m = int(value_codes.max()) + 1
    if k * m <= PAIR_TABLE_CELLS:
        return _pair_counts_by_table(groups, value_codes, k, m)
    return _pair_counts_by_sorting(groups, value_codes, k, m)


@procedure
def jonckheere(data: Any, y: str, by: str, *, order: list[Any] | None = None,
               missing: str = "drop"):
    """Jonckheere-Terpstra test for an ordered alternative across k independent samples.

    H0: the k distributions are equal; H1: they are stochastically ordered
    along the groups. With U_ij the Mann-Whitney count of pairs in which the
    value of the later group j exceeds that of the earlier group i (ties 1/2),

        J = sum_{i<j} U_ij,     E(J) = (N^2 - sum n_i^2) / 4,

        Var(J) = [N(N-1)(2N+5) - sum n_i(n_i-1)(2n_i+5) - sum t_g(t_g-1)(2t_g+5)] / 72
                 + [sum n_i(n_i-1)(n_i-2)] [sum t_g(t_g-1)(t_g-2)] / (36 N (N-1)(N-2))
                 + [sum n_i(n_i-1)] [sum t_g(t_g-1)] / (8 N (N-1)),

    t_g being the sizes of the tie groups of ``y`` (the tie-corrected variance
    SPSS uses). z = (J - E(J)) / sqrt(Var(J)) is standard normal under H0.
    J is computed from the group-by-value cross-classification or, with many
    groups, by an O(N log N) sort-based count; never from the O(N^2) pairs.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome column.
    by : grouping column with k >= 2 distinct values.
    order : the groups from lowest to highest; by default the sorted levels
        of ``by`` (the declared order for a categorical column).
    missing : "drop" (rows with a missing value in the columns used are deleted
        and counted in ``n_dropped``) or "raise".

    Returns
    -------
    TableSet with ``groups`` (n, mean_rank in the tested order) and ``tests``
    (one row: statistic J, mean, std_dev, z, p_value). ``attrs``: n, k,
    statistic, mean, std_dev, z, p_value (two-sided), p_increasing,
    p_decreasing.

    SPSS: ``NPAR TESTS /J-T=y BY group(1 k)``. Stata: community command
    ``jonter``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [1, 2, 3, 2, 4, 5, 4, 6, 7], "dose": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    >>> round(oe.jonckheere(data, "y", "dose").attrs["z"], 3)
    2.355
    """
    frame, x, dropped = _pooled(data, y, by, missing)
    codes, levels, counts = _groups(frame, by, 2, "The Jonckheere-Terpstra test")
    k, total = len(levels), x.numel()
    if order is not None:
        if not isinstance(order, (list, tuple)):
            raise AnalysisError("invalid_option", "order must list the groups from lowest to "
                                "highest.")
        wanted = [label(value) for value in order]
        if sorted(map(str, wanted)) != sorted(map(str, levels)) or len(set(map(str, wanted))) != k:
            raise AnalysisError("invalid_option", "order must contain every group of "
                                f"'{by}' exactly once: {', '.join(map(str, levels))}.")
        position = {str(level): index for index, level in enumerate(wanted)}
        remap = torch.tensor([position[str(level)] for level in levels], dtype=torch.int64)
        codes = remap[codes]
        counts = torch.zeros(k, dtype=FLOAT).index_copy_(0, remap, counts)
        levels = wanted
    if total < 3:
        raise AnalysisError("too_few_observations", "The Jonckheere-Terpstra test needs at "
                            "least 3 observations.")
    ranks, tie_sizes = midranks(x)
    concordant, discordant = ordered_pair_counts(codes, x, k)
    between = (total * total - float((counts * counts).sum())) / 2.0
    statistic = concordant + (between - concordant - discordant) / 2.0
    mean = between / 2.0
    t = tie_sizes.to(FLOAT)
    n = counts

    def moments(v: Tensor) -> tuple[float, float, float]:
        return (float((v * (v - 1) * (2 * v + 5)).sum()), float((v * (v - 1) * (v - 2)).sum()),
                float((v * (v - 1)).sum()))

    n_a, n_b, n_c = moments(n)
    t_a, t_b, t_c = moments(t)
    big = float(total)
    variance = (big * (big - 1) * (2 * big + 5) - n_a - t_a) / 72.0 \
        + n_b * t_b / (36.0 * big * (big - 1) * (big - 2)) + n_c * t_c / (8.0 * big * (big - 1))
    if variance <= 0.0:
        raise AnalysisError("no_variation", "The Jonckheere-Terpstra statistic has no variance: "
                            f"'{y}' is constant or every group but one is empty.")
    sd = math.sqrt(variance)
    z = (statistic - mean) / sd
    attrs = {"n": total, "k": k, "groups": levels, "statistic": statistic, "mean": mean,
             "std_dev": sd, "z": z, "p_value": normal_two_sided(z),
             "p_increasing": normal_sf(z), "p_decreasing": normal_sf(-z),
             "distribution": "normal", "n_dropped": dropped,
             "label": "Jonckheere-Terpstra test for ordered alternatives"}
    sums = group_sum(ranks, codes, k)
    groups = table({"n": counts.to(torch.int64).tolist(), "mean_rank": (sums / counts).tolist()},
                   index=[str(v) for v in levels])
    tests = table([[statistic, mean, sd, z, attrs["p_value"]]],
                  columns=["statistic", "mean", "std_dev", "z", "p_value"], index=["J-T"])
    return TableSet({"groups": groups, "tests": tests},
                    title=f"Jonckheere-Terpstra test: {y} by {by}", **attrs)
