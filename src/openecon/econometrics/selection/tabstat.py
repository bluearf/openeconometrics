"""Weighted summary statistics by group: ``oe.tabstat`` (Stata tabstat, SPSS MEANS).

Group statistics are O(n) scatter sums (``index_add_``) of weighted, centred
powers; percentiles, medians and grouped medians come from one sort by
(group, value) and a cumulative sum of weights. Nothing loops over
observations.

Weights follow Stata's ``summarize``. With weights ``w_i`` and ``W = sum w_i``
in a group:

- frequency weights (``fweight``): the row counts ``w_i`` times: n = W;
- analytic weights (``aweight``): the weights are rescaled to sum to the number
  of rows: n = number of rows, ``v_i = w_i n / W``.

Then mean = sum(w x) / W, variance = sum(v (x - mean)^2) / (n - 1), sum = sum(w x) =
W * mean (Stata's ``r(sum)``: with analytic weights it uses the weights as given), and
the central moments are m_r = sum(w (x - mean)^r) / W.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.selection import common as c

DEFAULT_STATS = ("n", "mean", "sd", "min", "max")
_ALIASES = {
    "count": "n", "N": "n", "sd": "std_dev", "semean": "std_error", "se": "std_error",
    "var": "variance", "skew": "skewness", "kurt": "kurtosis", "seskew": "se_skewness",
    "sekurt": "se_kurtosis", "gmedian": "grouped_median", "harmonic": "harmonic_mean",
    "hmean": "harmonic_mean", "geometric": "geometric_mean", "gmean": "geometric_mean",
}
_NAMES = ("n", "mean", "sum", "min", "max", "range", "std_dev", "variance", "cv", "std_error",
          "skewness", "kurtosis", "se_skewness", "se_kurtosis", "median", "iqr",
          "grouped_median", "harmonic_mean", "geometric_mean")
_SORTED = {"median", "iqr", "grouped_median"}


def _percent(name: str) -> float | None:
    if not name.startswith("p"):
        return None
    try:
        value = float(name[1:])
    except ValueError:
        return None
    return value if 0.0 < value < 100.0 else None


def _group_sum(x: Tensor, codes: Tensor, groups: int) -> Tensor:
    return torch.zeros(groups, dtype=torch.float64).index_add_(0, codes, x)


@torch.no_grad()
def weighted_percentile(ordered: Tensor, cumulative: Tensor, start: Tensor, rows: Tensor,
                        weight: Tensor, percent: float) -> Tensor:
    """Stata's percentile of every group from values sorted by (group, value).

    ``cumulative`` is the running sum of weights over the sorted rows (the row
    number for unweighted data). With ``P = W p / 100`` and ``W_(i)`` the
    cumulative weight inside the group, the percentile is ``x_(i)`` for the
    first ``i`` with ``W_(i) > P``, or the mean of ``x_(i-1)`` and ``x_(i)``
    when ``W_(i-1) = P`` exactly.
    """
    last_row = max(ordered.numel() - 1, 0)
    first = start.clamp(max=last_row)
    last = (start + rows - 1).clamp(min=0, max=last_row)
    base = torch.where(start > 0, cumulative[(first - 1).clamp_min(0)],
                       torch.zeros_like(weight))
    slack = 1e-9 * weight
    target = base + weight * (percent / 100.0)
    index = torch.searchsorted(cumulative, target - slack, right=True)
    index = torch.minimum(torch.maximum(index, first), last)
    hit = (cumulative[index] - target).abs() <= slack
    following = torch.minimum(index + 1, last)
    value = torch.where(hit, 0.5 * (ordered[index] + ordered[following]), ordered[index])
    return torch.where(rows > 0, value, torch.full_like(value, float("nan")))


@torch.no_grad()
def grouped_median(ordered: Tensor, w_ordered: Tensor, codes_ordered: Tensor, weight: Tensor,
                   groups: int) -> Tensor:
    """Median of grouped data, every distinct value standing for a class interval.

    The class of a value runs from the midpoint to the previous distinct value
    to the midpoint to the next one (the outermost classes are symmetric about
    their value). With ``L``, ``U`` the limits of the class in which the
    cumulative weight first reaches ``W / 2``, ``f`` its weight and ``F`` the
    cumulative weight below it:  ``L + (W/2 - F) / f * (U - L)``.
    """
    nan = torch.full((groups,), float("nan"), dtype=torch.float64)
    if not ordered.numel():
        return nan
    new = torch.ones_like(codes_ordered, dtype=torch.bool)
    new[1:] = (ordered[1:] != ordered[:-1]) | (codes_ordered[1:] != codes_ordered[:-1])
    run = torch.cumsum(new.to(torch.int64), 0) - 1
    value, group = ordered[new], codes_ordered[new]
    frequency = torch.zeros(value.numel(), dtype=torch.float64).index_add_(0, run, w_ordered)
    cumulative = torch.cumsum(frequency, 0)
    runs = torch.bincount(group, minlength=groups)
    first = torch.cumsum(runs, 0) - runs
    top = value.numel() - 1
    first_safe = first.clamp(max=top)
    last = (first + runs - 1).clamp(min=0, max=top)
    base = torch.where(first > 0, cumulative[(first_safe - 1).clamp_min(0)],
                       torch.zeros_like(weight))
    half = 0.5 * weight
    index = torch.searchsorted(cumulative, base + half - 1e-9 * weight, right=False)
    index = torch.minimum(torch.maximum(index, first_safe), last)
    below = torch.where(index > first_safe, cumulative[(index - 1).clamp_min(0)] - base,
                        torch.zeros_like(weight))
    here = value[index]
    previous = value[(index - 1).clamp_min(0)]
    following = value[(index + 1).clamp(max=top)]
    has_previous, has_next = index > first_safe, index < last
    gap_down = torch.where(has_previous, here - previous,
                           torch.where(has_next, following - here, torch.zeros_like(here)))
    gap_up = torch.where(has_next, following - here, gap_down)
    lower = here - 0.5 * gap_down
    width = 0.5 * (gap_down + gap_up)
    result = lower + (half - below) / frequency[index] * width
    return torch.where(runs > 0, result, nan)


@torch.no_grad()
def group_summaries(x: Tensor, codes: Tensor, groups: int, weights: Tensor | None, *,
                    frequency: bool, wanted: set[str], percents: dict[str, float],
                    moments: str, order: Tensor | None = None) -> dict[str, Tensor]:
    """Per-group statistics of one variable; every tensor has one entry per group.

    ``n`` is the sum of frequency weights or the number of rows; statistics
    that are undefined in a group (no rows, one row for a variance, zero spread
    for skewness) are NaN. ``order`` is an optional precomputed stable argsort
    of ``x`` (shared between the by-group and the overall table).
    """
    nan = torch.full((groups,), float("nan"), dtype=torch.float64)
    if not x.numel():
        empty = {name: nan for name in (*_NAMES, *percents, "ss")}
        empty.update({"n": torch.zeros(groups, dtype=torch.float64),
                      "weight": torch.zeros(groups, dtype=torch.float64)})
        return empty
    w = torch.ones_like(x) if weights is None else weights
    rows = torch.bincount(codes, minlength=groups)
    weight = _group_sum(w, codes, groups)
    n = weight if frequency else rows.to(torch.float64)
    filled = rows > 0
    safe_w = torch.where(filled, weight, torch.ones_like(weight))
    # Location-invariant sums are taken about the overall mean, then refined.
    shift = float((w * x).sum() / w.sum())
    xs = x - shift
    mean = _group_sum(w * xs, codes, groups) / safe_w
    mean = mean + _group_sum(w * (xs - mean[codes]), codes, groups) / safe_w
    deviation = xs - mean[codes]
    ss = _group_sum(w * deviation.square(), codes, groups)
    m2 = ss / safe_w
    variance = torch.where(n > 1, m2 * n / (n - 1).clamp_min(1e-300), nan)
    std_dev = variance.sqrt()
    out: dict[str, Tensor] = {
        "n": torch.where(filled, n, torch.zeros_like(n)),
        "mean": torch.where(filled, mean + shift, nan),
        "variance": variance, "std_dev": std_dev,
        "std_error": std_dev / n.clamp_min(1e-300).sqrt(),
        "ss": ss, "weight": weight,
    }
    out["sum"] = out["mean"] * weight            # sum(w x), Stata's r(sum)
    out["cv"] = torch.where(out["mean"] != 0, std_dev / out["mean"], nan)
    if wanted & {"min", "max", "range"}:
        low = torch.full((groups,), float("inf"), dtype=torch.float64)
        high = torch.full((groups,), float("-inf"), dtype=torch.float64)
        low = low.scatter_reduce(0, codes, x, reduce="amin")
        high = high.scatter_reduce(0, codes, x, reduce="amax")
        out["min"] = torch.where(filled, low, nan)
        out["max"] = torch.where(filled, high, nan)
        out["range"] = out["max"] - out["min"]
    if wanted & {"skewness", "kurtosis", "se_skewness", "se_kurtosis"}:
        spread = filled & (m2 > 0) & (ss > 1e-30 * safe_w * (mean + shift).square())
        root = torch.where(spread, m2.sqrt(), torch.ones_like(m2))
        z = deviation / root[codes]
        g1 = torch.where(spread, _group_sum(w * z ** 3, codes, groups) / safe_w, nan)
        g2 = torch.where(spread, _group_sum(w * z ** 4, codes, groups) / safe_w, nan)
        se_skewness = torch.where(
            n > 2, (6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)).clamp_min(1e-300)).sqrt(),
            nan)
        se_kurtosis = torch.where(
            n > 3, (4.0 * (n * n - 1) * se_skewness.square()
                    / ((n - 3) * (n + 5)).clamp_min(1e-300)).sqrt(), nan)
        if moments == "spss":
            g1 = torch.where(n > 2, g1 * (n * (n - 1)).clamp_min(0).sqrt()
                             / (n - 2).clamp_min(1e-300), nan)
            g2 = torch.where(n > 3, (n - 1) / ((n - 2) * (n - 3)).clamp_min(1e-300)
                             * ((n + 1) * g2 - 3.0 * (n - 1)), nan)
        out.update({"skewness": g1, "kurtosis": g2, "se_skewness": se_skewness,
                    "se_kurtosis": se_kurtosis})
    if "geometric_mean" in wanted or "harmonic_mean" in wanted:
        positive = x > 0
        defined = filled & (_group_sum((~positive).to(torch.float64), codes, groups) == 0)
        guarded = torch.where(positive, x, torch.ones_like(x))
        out["geometric_mean"] = torch.where(
            defined, (_group_sum(w * guarded.log(), codes, groups) / safe_w).exp(), nan)
        out["harmonic_mean"] = torch.where(
            defined, safe_w / _group_sum(w / guarded, codes, groups).clamp_min(1e-300), nan)
    if percents or wanted & _SORTED:
        if order is None:
            order = torch.argsort(x, stable=True)
        if groups > 1:
            order = order[torch.argsort(codes[order], stable=True)]
        ordered, w_ordered = x[order], w[order]
        start = torch.cumsum(rows, 0) - rows
        cumulative = torch.cumsum(w_ordered, 0)
        requested = dict(percents)
        if "median" in wanted:
            requested["median"] = 50.0
        if "iqr" in wanted:
            requested.update({"_p25": 25.0, "_p75": 75.0})
        for name, percent in requested.items():
            out[name] = weighted_percentile(ordered, cumulative, start, rows, weight, percent)
        if "iqr" in wanted:
            out["iqr"] = out["_p75"] - out["_p25"]
        if "grouped_median" in wanted:
            out["grouped_median"] = grouped_median(ordered, w_ordered, codes[order], weight,
                                                   groups)
    return out


def _parse_stats(stats: Any) -> tuple[list[str], dict[str, float]]:
    """Canonical statistic names in order (``q`` expanded) and the percentile requests."""
    given = c.name_list(list(DEFAULT_STATS) if stats is None else stats, "stats", minimum=1,
                        noun="statistic", example="mean")
    names: list[str] = []
    for name in given:
        names += ["p25", "p50", "p75"] if name == "q" else [_ALIASES.get(name, name)]
    percents: dict[str, float] = {}
    for name in names:
        percent = _percent(name)
        if percent is not None:
            percents[name] = percent
        elif name not in _NAMES:
            raise AnalysisError("invalid_option", f"Unknown statistic '{name}'. Available: "
                                f"{', '.join(_NAMES)}, q and percentiles such as p25.")
    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise AnalysisError("invalid_spec", "stats lists a statistic more than once (sd is "
                            "std_dev, semean is std_error, q is p25 p50 p75): "
                            f"{', '.join(repeated)}.")
    return names, percents


def tabstat(data: Any, columns: list[str], *, by: str | None = None,
            stats: list[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, moments: str = "stata", listwise: bool = False,
            total: bool = True, anova: bool = False) -> Any:
    """Table of summary statistics, optionally by group and weighted.

    Stata's ``tabstat`` and SPSS's ``MEANS``: one row per variable (and group)
    and one column per requested statistic.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : numeric columns to summarize.
    by : grouping column; groups are listed in ascending order, followed by a
        ``Total`` row per variable. Rows with a missing group are excluded.
    stats : statistics, in the order to report (default ``n, mean, sd, min,
        max``):

        - ``n`` (``count``): observations (sum of frequency weights);
        - ``mean``, ``sum``, ``min``, ``max``, ``range``;
        - ``sd`` (column ``std_dev``), ``variance`` (``var``), ``cv`` (sd / mean),
          ``semean`` (column ``std_error`` = sd / sqrt(n));
        - ``skewness``, ``kurtosis`` (see ``moments``), ``se_skewness``,
          ``se_kurtosis`` (SPSS's standard errors);
        - ``median``, ``p1`` ... ``p99`` (any percentile, also ``p2.5``), ``iqr``
          (p75 - p25), ``q`` (p25, p50 and p75);
        - the SPSS MEANS extras ``grouped_median`` (``gmedian``),
          ``harmonic_mean`` (``harmonic``) and ``geometric_mean``
          (``geometric``).
    weights, weight_type : weight column and its type, as in Stata's
        ``summarize``: ``"aweight"`` (default; analytic weights, rescaled to
        sum to the number of rows) or ``"fweight"`` (frequency weights: a row
        counts ``w`` times; this is how SPSS's WEIGHT command behaves). Rows
        with a zero or missing weight are excluded.
    moments : ``"stata"`` (default): skewness = m3 / m2^1.5 and kurtosis = m4 /
        m2^2 with m_r = sum(w (x - mean)^r) / sum(w), so a normal variable has
        kurtosis 3; ``"spss"``: the bias-corrected G1 = g1 sqrt(n(n-1)) / (n-2)
        and excess G2 = (n-1) ((n+1) g2 - 3(n-1)) / ((n-2)(n-3)).
    listwise : False (default) summarizes each variable on its own non-missing
        values (Stata tabstat, SPSS MEANS); True keeps only rows complete in
        all ``columns`` (Stata's ``casewise``).
    total : include the ``Total`` rows when ``by`` is given.
    anova : with ``by``, also return the one-way analysis-of-variance table of
        each variable on the groups and eta / eta squared (SPSS ``MEANS
        /STATISTICS ANOVA``).

    Formulas
    --------
    With weights ``w`` (1 when unweighted), ``W = sum w`` and ``n`` the number
    of observations (rows, or ``W`` for frequency weights):

    - mean = sum(w x) / W; variance = (n / W) sum(w (x - mean)^2) / (n - 1);
      sum = sum(w x) (Stata's ``r(sum)``; with analytic weights the weights as given,
      so unlike every other statistic it scales with them); standard error of the
      mean = sd / sqrt(n);
    - percentiles (Stata ``summarize, detail``): with P = W p / 100 and
      cumulative weights W_(i) of the sorted values, x_(i) for the first i with
      W_(i) > P, or (x_(i-1) + x_(i)) / 2 when W_(i-1) = P;
    - geometric mean = exp(sum(w ln x) / W) and harmonic mean = W / sum(w / x),
      defined when every value is positive;
    - grouped median: each distinct value is the midpoint of a class whose
      limits lie halfway to the neighbouring distinct values; the median is
      interpolated linearly inside the class where the cumulative weight
      reaches W / 2 (L + (W/2 - F) / f * h);
    - ANOVA: between = sum_g n_g (mean_g - mean)^2 with G - 1 degrees of
      freedom, within = sum of the groups' sums of squares with n - G, F =
      MS_between / MS_within, eta squared = between / (between + within).

    Returns
    -------
    A table. Without ``by``: index = variable names, one column per statistic.
    With ``by``: columns ``variable``, the ``by`` name, then the statistics;
    one row per variable and group and, with ``total``, a row labelled
    ``Total``. Undefined statistics (the standard deviation of one observation,
    the geometric mean of non-positive values) are missing.
    ``attrs``: ``by``, ``groups``, ``stats``, ``weights``, ``moments``,
    ``missing``, ``n`` (rows used).

    With ``anova=True`` a TableSet ``{"statistics", "anova", "association"}``:
    ``anova`` has one between / within / total block per variable (``ss``,
    ``df``, ``ms``, ``statistic``, ``p_value``) and ``association`` the ``eta``
    and ``eta_squared`` of each variable.

    Equivalent commands: Stata ``tabstat x1 x2 [aw=w], by(g) statistics(n mean
    sd min max)``; SPSS ``MEANS TABLES=x1 x2 BY g /CELLS=COUNT MEAN STDDEV
    MEDIAN GMEDIAN HARMONIC GEOMETRIC /STATISTICS ANOVA``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [2.0, 4.0, 6.0, 5.0, 7.0, 9.0], "g": ["a", "a", "a", "b", "b", "b"]}
    >>> table = oe.tabstat(data, ["x"], by="g", stats=["n", "mean", "median"])
    >>> table[["g", "n", "mean", "median"]].values.tolist()
    [['a', 3, 4.0, 4.0], ['b', 3, 7.0, 7.0], ['Total', 6, 5.5, 5.5]]
    """
    names = c.name_list(columns, "columns", minimum=1)
    if by is not None:
        c.check_name(by, "by")
    requested, percents = _parse_stats(stats)
    weights, weight_type = c.check_weights(weights, weight_type)
    c.check_choice(moments, "moments", ("stata", "spss"))
    for flag, label in ((listwise, "listwise"), (total, "total"), (anova, "anova")):
        c.check_flag(flag, label)
    if anova and by is None:
        raise AnalysisError("invalid_spec", "anova=True needs a grouping column: pass by=...")
    used = [*names, *([by] if by else []), *([weights] if weights else [])]
    frame = c.source(data, used, numeric=[*names, *([weights] if weights else [])])
    keep = torch.ones(len(frame), dtype=torch.bool)
    notes: list[str] = []
    if by is not None:
        keep &= c.present(frame, [by])
        if not bool(keep.any()):
            raise AnalysisError("empty_sample", f"Every value of by='{by}' is missing.")
    w_full = None
    if weights is not None:
        keep &= c.present(frame, [weights], [weights])
        if not bool(keep.any()):
            raise AnalysisError("empty_sample", f"Every value of weights='{weights}' is missing.")
        before = int(keep.sum())
        w_full = c.weight_column(frame, weights, weight_type, keep)
        if int(keep.sum()) < before:
            notes.append(f"Excluded {before - int(keep.sum())} observation(s) with zero weight.")
    if listwise:
        keep &= c.present(frame, names, names)
        if not bool(keep.any()):
            raise AnalysisError("empty_sample", "No observation is complete in all columns.")
    if by is not None:
        codes_full, labels = c.group_codes(frame[by], keep, by)
    else:
        codes_full, labels = torch.zeros(len(frame), dtype=torch.int64), [None]
    groups = len(labels)
    frequency = weight_type == "fweight"
    wanted = set(requested)
    options = {"frequency": frequency, "wanted": wanted, "percents": percents,
               "moments": moments}
    rows: list[list[Any]] = []
    anova_rows: list[list[Any]] = []
    association: list[list[Any]] = []

    def cells(table: dict[str, Tensor], g: int) -> list[Any]:
        count = int(round(float(table["n"][g])))
        row: list[Any] = []
        for name in requested:
            row.append(count if name == "n" else
                       c.finite(table[name][g]) if count > 0 else None)
        return row

    for variable in names:
        column = c.column(frame, variable, allow_missing=True)
        use = keep & ~torch.isnan(column)
        values, group = column[use], codes_full[use]
        w = None if w_full is None else w_full[use]
        order = torch.argsort(values, stable=True) if percents or wanted & _SORTED else None
        table = group_summaries(values, group, groups, w, order=order, **options) \
            if by is not None else None
        overall = group_summaries(values, torch.zeros_like(group), 1, w, order=order, **options)
        if by is None:
            rows.append(cells(overall, 0))
            continue
        for g, label in enumerate(labels):
            rows.append([variable, label, *cells(table, g)])
        if total:
            rows.append([variable, "Total", *cells(overall, 0)])
        if anova:
            block, measures = _anova(table, overall)
            anova_rows += [[variable, *line] for line in block]
            association.append(measures)
    attrs = {"by": by, "groups": labels if by is not None else None, "stats": requested,
             "weights": None if weights is None else {"column": weights, "type": weight_type},
             "moments": moments, "missing": "listwise" if listwise else "variable-wise",
             "n": int(keep.sum()), "notes": notes}
    if by is None:
        return c.result_table(rows, columns=requested, index=names, **attrs)
    statistics = c.result_table(rows, columns=["variable", by, *requested], **attrs)
    if not anova:
        return statistics
    anova_table = c.result_table(anova_rows, columns=["variable", "source", "ss", "df", "ms",
                                                      "statistic", "p_value"])
    measures_table = c.result_table(association, index=names, columns=["eta", "eta_squared"])
    return TableSet({"statistics": statistics, "anova": anova_table,
                     "association": measures_table},
                    title=f"Summary statistics by {by}", **attrs)


def _anova(table: dict[str, Tensor], overall: dict[str, Tensor]
           ) -> tuple[list[list[Any]], list[Any]]:
    """One-way ANOVA rows (between, within, total) and [eta, eta squared] of one variable."""
    n, weight = float(overall["n"][0]), float(overall["weight"][0])
    filled = table["n"] > 0
    k = int(filled.sum())
    if n <= 0 or weight <= 0:
        return [["between", None, None, None, None, None], ["within", None, None, None, None,
                                                            None],
                ["total", None, None, None, None, None]], [None, None]
    scale = n / weight                         # analytic weights are rescaled to sum to n
    grand = float(overall["mean"][0])
    means = torch.where(filled, table["mean"], torch.zeros_like(table["mean"]))
    between = float((table["weight"] * (means - grand).square())[filled].sum()) * scale
    within = float(table["ss"][filled].sum()) * scale
    df1, df2 = k - 1, n - k
    ms_between = between / df1 if df1 > 0 else None
    ms_within = within / df2 if df2 > 0 else None
    statistic = None
    if ms_between is not None and ms_within is not None and ms_within > 0 \
            and within > 1e-24 * (between + within):
        statistic = ms_between / ms_within
    eta_squared = between / (between + within) if between + within > 0 else None
    eta = None if eta_squared is None else eta_squared ** 0.5
    return [["between", between, df1, ms_between, statistic, c.f_upper(statistic, df1, df2)],
            ["within", within, df2, ms_within, None, None],
            ["total", between + within, n - 1, None, None, None]], [eta, eta_squared]
