"""Descriptive statistics by variable and group: ``oe.describe``.

SPSS DESCRIPTIVES / EXPLORE / FREQUENCIES and Stata ``summarize, detail`` /
``tabstat``. Moments come from O(n) group sums of centred powers and percentiles
from one sort by (group, value); nothing loops over observations.
"""

from __future__ import annotations

from typing import Any

import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import common as c

DEFAULT_STATS = ("n", "mean", "std_dev", "std_error", "ci", "min", "p25", "p50", "p75", "max",
                 "skewness", "kurtosis")
_ALIASES = {"sd": "std_dev", "se": "std_error", "median": "p50", "count": "n", "var": "variance"}
_NAMES = ("n", "missing", "mean", "std_dev", "std_error", "variance", "cv", "sum", "min", "max",
          "range", "ci", "iqr", "skewness", "se_skewness", "kurtosis", "se_kurtosis")


def _percent(name: str) -> float | None:
    if not name.startswith("p"):
        return None
    try:
        value = float(name[1:])
    except ValueError:
        return None
    return value if 0.0 < value < 100.0 else None


def percentiles(ordered: Tensor, start: Tensor, counts: Tensor, percent: float,
                method: str) -> Tensor:
    """Percentile of every group from values sorted by (group, value).

    ``"stata"`` (Stata's default, also SAS definition 5): with P = n p / 100, the
    mean of x_(P) and x_(P+1) when P is an integer, otherwise x_(floor(P)+1).
    ``"haverage"`` (SPSS HAVERAGE, Stata ``altdef``): the weighted average at
    position (n + 1) p / 100, (1 - g) x_(k) + g x_(k+1) with k its integer part and g
    its fraction, x_(1) below the first and x_(n) above the last position.
    """
    n = counts.to(torch.float64)
    last = (counts - 1).clamp_min(0)
    if method == "stata":
        position = n * percent / 100.0
        nearest = torch.round(position)
        whole = (position - nearest).abs() <= 1e-9 * n.clamp_min(1.0)
        low = torch.where(whole, nearest - 1.0, torch.floor(position)).to(torch.int64)
        high = torch.where(whole, low + 1, low)
        low, high = low.clamp(min=0), torch.minimum(high.clamp(min=0), last)
        low = torch.minimum(low, last)
        return 0.5 * (ordered[start + low] + ordered[start + high])
    position = (n + 1.0) * percent / 100.0
    k = torch.floor(position)
    g = position - k
    low = (k.to(torch.int64) - 1).clamp(min=0)
    low = torch.minimum(low, last)
    high = torch.minimum(k.to(torch.int64).clamp(min=0), last)
    g = torch.where((k < 1) | (k >= n), torch.zeros_like(g), g)
    low = torch.where(k >= n, last, low)
    return (1.0 - g) * ordered[start + low] + g * ordered[start + high]


def _moment_columns(x: Tensor, codes: Tensor, groups: int, moments: str) -> dict[str, Tensor]:
    """Per-group n, mean, variance, extremes, skewness and kurtosis with their SPSS errors."""
    low, high = c.group_extremes(x, codes, groups)
    x, shift = c.centre(x)
    base = c.group_moments(x, codes, groups)
    n = base.n
    nan = torch.full_like(n, float("nan"))
    sd = base.var.sqrt()
    spread = (n > 1) & (base.ss > 0)
    # Third and fourth powers of standardized deviations z = (x - mean) / s: powers of
    # the raw deviations overflow (or underflow) long before the data do.
    z = (x - base.mean[codes]) / torch.where(spread, sd, torch.ones_like(sd))[codes]
    third = c.group_sum(z ** 3, codes, groups)
    fourth = c.group_sum(z ** 4, codes, groups)
    if moments == "spss":
        # G1 = n M3 / ((n-1)(n-2) s^3)
        # G2 = n(n+1) M4 / ((n-1)(n-2)(n-3) s^4) - 3(n-1)^2 / ((n-2)(n-3))
        skew = torch.where(spread & (n > 2), n * third / ((n - 1) * (n - 2)), nan)
        kurt = torch.where(spread & (n > 3), n * (n + 1) * fourth
                           / ((n - 1) * (n - 2) * (n - 3))
                           - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)), nan)
        se_skew = torch.where(n > 2, torch.sqrt(6.0 * n * (n - 1)
                                                / ((n - 2) * (n + 1) * (n + 3))), nan)
        se_kurt = torch.where(n > 3, torch.sqrt(4.0 * (n * n - 1) * se_skew ** 2
                                                / ((n - 3) * (n + 5))), nan)
    else:
        # Stata: m3 / m2^1.5 and m4 / m2^2 with m_r = M_r / n (a normal has kurtosis 3);
        # m2 = s^2 (n - 1) / n, so the deviations in units of sqrt(m2) are z sqrt(n / (n - 1)).
        inflate = n / (n - 1.0).clamp_min(1.0)
        skew = torch.where(spread, third / n * inflate ** 1.5, nan)
        kurt = torch.where(spread, fourth / n * inflate ** 2, nan)
        se_skew = se_kurt = nan
    mean = base.mean + shift
    return {"n": n, "mean": mean, "variance": base.var, "std_dev": sd,
            "std_error": sd / n.sqrt(), "min": low, "max": high, "range": high - low,
            "sum": mean * n,
            "cv": torch.where(mean != 0, sd / mean, nan),
            "skewness": skew, "se_skewness": se_skew, "kurtosis": kurt, "se_kurtosis": se_kurt}


def describe(data: Any, columns: list[str] | None = None, *, by: str | None = None,
             stats: list[str] | None = None, percentile_method: str = "stata",
             moments: str = "spss", listwise: bool = False, alpha: float = 0.05) -> Any:
    """Descriptive statistics of numeric columns, optionally by group.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : numeric columns (default: every numeric column except ``by``).
    by : grouping column; one row per variable and group (rows with a missing
        group value are excluded).
    stats : statistics to report, in order. Available: ``n``, ``missing``, ``mean``,
        ``std_dev`` (``sd``), ``std_error`` (``se``), ``variance``, ``cv``, ``sum``, ``min``,
        ``max``, ``range``, ``ci`` (columns ci_low and ci_high of the mean, Student t),
        ``median``, ``iqr``, any percentile ``p1`` .. ``p99`` (also fractional, ``p2.5``),
        ``skewness``, ``se_skewness``, ``kurtosis``, ``se_kurtosis``. Default: n, mean,
        std_dev, std_error, ci, min, p25, p50, p75, max, skewness, kurtosis.
    percentile_method : ``"stata"`` (default; Stata ``summarize, detail``: average of
        x_(P) and x_(P+1) when P = n p/100 is an integer, else x_(floor(P)+1)) or
        ``"haverage"`` (SPSS's default HAVERAGE, the weighted average at (n+1) p/100;
        Stata ``centile``/``altdef``).
    moments : ``"spss"`` (default; bias-corrected G1 = n M3 / ((n-1)(n-2) s^3) and excess
        G2 = n(n+1) M4 / ((n-1)(n-2)(n-3) s^4) - 3(n-1)^2 / ((n-2)(n-3)), with standard
        errors sqrt(6n(n-1)/((n-2)(n+1)(n+3))) and sqrt(4(n^2-1) se_G1^2/((n-3)(n+5))))
        or ``"stata"`` (m3 / m2^1.5 and m4 / m2^2 with divisor n; a normal variable has
        kurtosis 3; no standard errors).
    listwise : False (default) uses all non-missing values of each variable (Stata
        ``summarize``, SPSS DESCRIPTIVES default); True keeps only rows complete in
        all ``columns``.
    alpha : 1 - confidence level of ``ci``.

    Returns
    -------
    Table with one row per variable (index = variable names) or, with ``by``, one
    row per variable and group (columns ``variable`` and the ``by`` name first).
    Statistics that are undefined for a group (for example the standard deviation
    of one observation) are missing. ``attrs``: ``percentile_method``, ``moments``,
    ``missing``, ``alpha``.

    Equivalent commands: SPSS ``DESCRIPTIVES``, ``EXAMINE``, ``FREQUENCIES
    /PERCENTILES``; Stata ``summarize x, detail``, ``tabstat x, by(g) stats(...)``,
    ``ci means x``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [2.0, 4.0, 4.0, 5.0, 7.0, 9.0], "g": ["a", "a", "a", "b", "b", "b"]}
    >>> table = oe.describe(data, ["x"], stats=["n", "mean", "median"])
    >>> table.loc["x"].tolist()
    [6.0, 5.166666666666667, 4.5]
    """
    alpha = c.check_alpha(alpha)
    c.check_choice(percentile_method, "percentile_method", ("stata", "haverage"))
    c.check_choice(moments, "moments", ("spss", "stata"))
    c.check_flag(listwise, "listwise")
    if by is not None:
        c.check_name(by, "by")
    from openecon.dataset import Dataset
    if columns is None and isinstance(data, Dataset):
        iterator = data.iter_batches(batch_rows=1)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise AnalysisError("empty_data", "The dataset contains no observations.") from exc
        finally:
            iterator.close()
        names = [name for name in first.columns if name != by
                 and is_numeric_dtype(first[name].dtype) and not is_bool_dtype(first[name].dtype)]
        if not names:
            raise AnalysisError("invalid_spec", "The data have no numeric columns to describe.")
    elif columns is None:
        source = _coerce_frame(data)
        names = [name for name in source.columns if isinstance(name, str) and name != by
                 and is_numeric_dtype(source[name].dtype) and not is_bool_dtype(source[name].dtype)]
        if not names:
            raise AnalysisError("invalid_spec", "The data have no numeric columns to describe.")
    else:
        names = c.name_list(columns, "columns")
    requested = [_ALIASES.get(name, name) for name in
                 c.name_list(stats if stats is not None else list(DEFAULT_STATS), "stats",
                             noun="statistic", example="mean")]
    for name in requested:
        if name not in _NAMES and _percent(name) is None:
            raise AnalysisError("invalid_option", f"Unknown statistic '{name}'. Available: "
                                f"{', '.join(_NAMES)}, median and percentiles such as p25.")
    if len(set(requested)) != len(requested):
        repeated = sorted({name for name in requested if requested.count(name) > 1})
        raise AnalysisError("invalid_spec", "stats lists a statistic more than once (sd is "
                            f"std_dev, se is std_error, median is p50): {', '.join(repeated)}.")
    if isinstance(data, Dataset):
        from .streaming_describe import describe as replay_describe
        return replay_describe(data, names, by, requested, percentile_method, moments, listwise, alpha)
    frame, _ = c.select(data, [*names, *([by] if by else [])], numeric=names, listwise=False)
    if by is not None:
        frame = frame.loc[frame[by].notna()].reset_index(drop=True)
        if not len(frame):
            raise AnalysisError("empty_sample", f"Every value of by='{by}' is missing.")
        codes, labels = c.group_codes(frame, by)
    else:
        codes, labels = torch.zeros(len(frame), dtype=torch.int64), [None]
    groups = len(labels)
    x = torch.stack([c.values(frame, name, allow_missing=True) for name in names], dim=1)
    present = ~torch.isnan(x)
    if listwise:
        keep = present.all(1)
        x, present, codes = x[keep], present[keep], codes[keep]
    available = torch.bincount(codes, minlength=groups)
    header: list[str] = []
    for name in requested:
        header += ["ci_low", "ci_high"] if name == "ci" else [name]
    critical: dict[int, float] = {}
    rows, index = [], []
    for j, variable in enumerate(names):
        values, group = x[present[:, j], j], codes[present[:, j]]
        table = _moment_columns(values, group, groups, moments)
        table["missing"] = available.to(torch.float64) - table["n"]
        ordered, start, counts = c.sorted_by_group(values, group, groups)
        start = torch.minimum(start, torch.tensor(max(values.numel() - 1, 0)))
        if "iqr" in requested:
            table["iqr"] = percentiles(ordered, start, counts, 75.0, percentile_method) \
                - percentiles(ordered, start, counts, 25.0, percentile_method)
        for name in requested:
            percent = _percent(name)
            if percent is not None and values.numel():
                table[name] = percentiles(ordered, start, counts, percent, percentile_method)
        cells = {name: table[name].tolist() for name in table}
        for g in range(groups):
            n = int(cells["n"][g])
            row: list[Any] = []
            for name in requested:
                if name == "ci":
                    if n > 1:
                        if n not in critical:
                            critical[n] = c.t_critical(alpha, n - 1)
                        half = critical[n] * cells["std_error"][g]
                        row += [cells["mean"][g] - half, cells["mean"][g] + half]
                    else:
                        row += [None, None]
                elif name in ("n", "missing"):
                    row.append(int(cells[name][g]))
                else:
                    value = cells[name][g] if n > 0 and name in cells else None
                    row.append(c.finite(value))
            rows.append(row if by is None else [variable, labels[g], *row])
            index.append(variable)
    attrs = {"percentile_method": percentile_method, "moments": moments, "alpha": alpha,
             "missing": "listwise" if listwise else "variable-wise", "n": int(x.shape[0])}
    if by is None:
        return c.frame(rows, columns=header, index=index, **attrs)
    return c.frame(rows, columns=["variable", by, *header], by=by, **attrs)
