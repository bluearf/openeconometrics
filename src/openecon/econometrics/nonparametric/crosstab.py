"""Cross-tabulation (crosstab) and one-way frequency tables (tabulate).

``crosstab`` cross-classifies two columns (optionally within the levels of a
layer column), with frequency weights, and reports what SPSS CROSSTABS and
Stata ``tabulate twoway`` report: counts with margins, expected counts,
percentages, residuals, the chi-square family of tests, Fisher's exact test,
measures of association with asymptotic standard errors, the 2x2 risk
estimates and, with a layer, the Cochran-Mantel-Haenszel analysis.

Counts are accumulated with one ``bincount`` / ``index_add_`` over the rows of
the data; every statistic is then a function of the small [R, C] table.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric import measures as measure_kernels
from openecon.econometrics.nonparametric import tables as table_kernels
from openecon.econometrics.nonparametric.common import (
    FLOAT, check_alpha, check_count, check_flag, group_codes, procedure, select, weight_column,
)
from openecon.engines.contracts import KernelError

MAX_CELLS = 100_000            # cells of one two-way table
MAX_LAYERS = 500
MAX_TOTAL_CELLS = 2_000_000    # cells over all layers (the printed tables hold every one)
PERCENTAGES = ("row", "column", "total")


def _scores(levels: list[Any]) -> Tensor:
    """Numeric category values as scores, or 1..K for non-numeric categories."""
    if all(isinstance(level, (int, float)) and not isinstance(level, bool) for level in levels):
        return torch.tensor([float(level) for level in levels], dtype=FLOAT)
    return torch.arange(1, len(levels) + 1, dtype=FLOAT)


def _cell(value: float, integer: bool) -> Any:
    return int(round(value)) if integer else value


def _with_margins(counts: Tensor, integer: bool) -> list[list[Any]]:
    body = torch.cat([counts, counts.sum(1, keepdim=True)], dim=1)
    body = torch.cat([body, body.sum(0, keepdim=True)], dim=0)
    return [[_cell(value, integer) for value in line] for line in body.tolist()]


def _percent(counts: Tensor, kind: str) -> list[list[Any]]:
    total = counts.sum()
    body = torch.cat([counts, counts.sum(1, keepdim=True)], dim=1)
    body = torch.cat([body, body.sum(0, keepdim=True)], dim=0)
    if kind == "row":
        base = body[:, -1:].expand_as(body)
    elif kind == "column":
        base = body[-1:, :].expand_as(body)
    else:
        base = total.expand_as(body)
    share = torch.where(base > 0, 100.0 * body / base.clamp_min(1e-300),
                        torch.full_like(body, float("nan")))
    return [[None if value != value else value for value in line] for line in share.tolist()]


def _residuals(counts: Tensor) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]]]:
    """(expected, standardized residuals, adjusted standardized residuals) as nested lists."""
    total = counts.sum()
    rows, columns = counts.sum(1), counts.sum(0)
    expected = torch.outer(rows, columns) / total
    positive = expected > 0
    safe = torch.where(positive, expected, torch.ones_like(expected))
    standardized = (counts - expected) / safe.sqrt()
    leverage = torch.outer(1.0 - rows / total, 1.0 - columns / total)
    defined = positive & (leverage > 0)
    adjusted = (counts - expected) / (safe * torch.where(defined, leverage,
                                                         torch.ones_like(leverage))).sqrt()

    def lists(values: Tensor, mask: Tensor) -> list[list[Any]]:
        return [[value if keep else None for value, keep in zip(line, flags)]
                for line, flags in zip(values.tolist(), mask.tolist())]

    return expected.tolist(), lists(standardized, positive), lists(adjusted, defined)


def _analyze(counts: Tensor, row_scores: Tensor, column_scores: Tensor, same_levels: bool,
             exact: bool | None, reps: int, seed: int, alpha: float) -> dict[str, Any]:
    """Tests, measures and risk estimates of one [R, C] table."""
    out: dict[str, Any] = {"tests": [], "measures": [], "risk": [], "notes": [], "attrs": {}}
    trimmed, keep_rows, keep_columns = table_kernels.trim(counts)
    r, c = trimmed.shape
    total = float(trimmed.sum())
    if r < 2 or c < 2:
        out["notes"].append("No statistics are computed because the row or the column variable "
                            "is constant in this table.")
        return out
    integer = table_kernels.is_integer_table(trimmed)
    chi2, df = table_kernels.pearson_chi2(trimmed)
    g2 = table_kernels.likelihood_ratio(trimmed)
    p_chi2 = table_kernels.chi2_p(chi2, df)
    out["tests"] += [("pearson", chi2, df, p_chi2),
                     ("likelihood_ratio", g2, df, table_kernels.chi2_p(g2, df))]
    out["attrs"].update({"statistic": chi2, "df": df, "p_value": p_chi2, "likelihood_ratio": g2,
                         "p_value_likelihood_ratio": table_kernels.chi2_p(g2, df)})
    two_by_two = r == 2 and c == 2
    if two_by_two:
        corrected = table_kernels.yates_chi2(trimmed)
        out["tests"].append(("continuity", corrected, 1, table_kernels.chi2_p(corrected, 1)))
    if exact is not False and not integer:
        if exact:
            out["notes"].append("Exact tests need integer counts; they are skipped for "
                                "non-integer weighted tables.")
    elif exact is not False and two_by_two:
        try:
            two_sided, lower, upper = table_kernels.fisher_2x2(trimmed)
        except KernelError:
            if exact:
                raise
            out["notes"].append("Fisher's exact test is skipped: the table total is too large "
                                "to tabulate the exact distribution.")
        else:
            out["tests"] += [("fisher_exact", None, None, two_sided),
                             ("fisher_exact_less", None, None, lower),
                             ("fisher_exact_greater", None, None, upper)]
            out["attrs"].update({"p_exact": two_sided, "p_exact_less": lower,
                                 "p_exact_greater": upper, "exact_method": "hypergeometric"})
    elif exact:
        p_exact = table_kernels.fisher_enumerate(trimmed)
        if p_exact is not None:
            out["attrs"].update({"p_exact": p_exact, "exact_method": "enumeration"})
        else:
            p_exact, low, high = table_kernels.fisher_monte_carlo(trimmed, reps, seed)
            out["attrs"].update({"p_exact": p_exact, "exact_method": "monte_carlo",
                                 "exact_reps": reps, "exact_seed": seed,
                                 "p_exact_ci_low": low, "p_exact_ci_high": high})
            out["notes"].append(f"Fisher's exact p is a Monte Carlo estimate from {reps} sampled "
                                f"tables (seed {seed}); 99% interval [{low:.4f}, {high:.4f}].")
        out["tests"].append(("fisher_exact", None, None, p_exact))
    x, y = row_scores[keep_rows], column_scores[keep_columns]
    linear = table_kernels.linear_by_linear(trimmed, x, y)
    if linear is not None:
        out["tests"].append(("linear_by_linear", linear, 1, table_kernels.chi2_p(linear, 1)))
    expected = table_kernels.expected_counts(trimmed)
    small = int((expected < 5).sum())
    out["attrs"].update({"min_expected": float(expected.min()), "cells_below_5": small})
    if small:
        out["notes"].append(f"{small} cell(s) ({100.0 * small / (r * c):.1f}%) have expected "
                            f"count less than 5. The minimum expected count is "
                            f"{float(expected.min()):.2f}.")
    rows: dict[str, Any] = {}
    rows.update(measure_kernels.nominal_measures(trimmed, chi2, g2))
    rows.update(measure_kernels.ordinal_measures(trimmed))
    rows.update(measure_kernels.correlations(trimmed, x, y))
    if same_levels and counts.shape[0] == counts.shape[1]:
        kinds = ("none",) if counts.shape[0] == 2 else ("none", "linear", "quadratic")
        for kind in kinds:
            value = measure_kernels.kappa(counts, kind)
            if value[0] is not None:
                rows["kappa" if kind == "none" else f"kappa_{kind}"] = value
    out["measures"] = [(name, *values) for name, values in rows.items()]
    out["attrs"]["cramers_v"] = rows["cramers_v"][0]
    if two_by_two:
        out["risk"] = [(name, *values)
                       for name, values in measure_kernels.risk_2x2(trimmed, alpha).items()]
    out["attrs"]["n"] = _cell(total, integer)
    return out


def _percent_kinds(percentages: Any) -> list[str]:
    if percentages is None:
        return []
    if percentages == "all":
        return list(PERCENTAGES)
    if isinstance(percentages, str):
        percentages = [percentages]
    if not isinstance(percentages, (list, tuple)) or any(kind not in PERCENTAGES
                                                         for kind in percentages):
        raise AnalysisError("invalid_option", "percentages must be None, 'all' or a list drawn "
                            "from 'row', 'column', 'total'.")
    return list(dict.fromkeys(percentages))


@procedure
def crosstab(data: Any, row: str, column: str, *, layer: str | None = None,
             weights: str | None = None, exact: bool | None = None, expected: bool = False,
             percentages: Any = None, residuals: bool = False, exact_reps: int = 10_000,
             seed: int = 2_000_003, alpha: float = 0.05, missing: str = "drop"):
    """Two-way cross-tabulation with tests of independence and measures of association.

    Counts the rows of ``data`` by the categories of ``row`` and ``column``
    (sorted values; declared order for categorical columns), optionally within
    each level of ``layer`` and with frequency ``weights``.

    Tests (table ``tests``; f_ij observed, E_ij = r_i c_j / N expected):

    * ``pearson``: sum (f - E)^2 / E, df (R-1)(C-1).
    * ``likelihood_ratio``: 2 sum f ln(f / E).
    * ``continuity`` (2x2): Yates' corrected chi-square.
    * ``fisher_exact`` (2x2, integer counts): two-sided p as the probability of
      all tables no more probable than the observed one, and the one-sided
      ``fisher_exact_less`` = P(n11 <= observed), ``fisher_exact_greater``.
      When a frequency-weighted table is so large that the hypergeometric
      distribution cannot be tabulated (totals of roughly 1e10 and more) the
      rows are omitted with a note; ``exact=True`` then raises
      ``exact_unavailable``.
      For larger tables ``exact=True`` gives the Fisher-Freeman-Halton p by
      complete enumeration when at most 3,000,000 partial tables arise, and
      otherwise a Monte Carlo estimate from ``exact_reps`` tables sampled with
      the observed margins (generator seed ``seed``; 99% interval reported),
      as the Monte Carlo option of SPSS Exact Tests.
    * ``linear_by_linear``: (N - 1) r^2, df 1, r the Pearson correlation of
      the category scores (the numeric category values, or 1..K for text).

    Empty rows and columns are removed before testing. A note reports how
    many cells have an expected count below 5, as SPSS does. A table may have
    at most 100,000 cells, 500 layers and 2,000,000 cells over all layers
    (``too_many_categories`` otherwise).

    Measures (table ``measures``: value, ase, t, p_value; see ``measures.py``):
    phi, Cramer's V, contingency coefficient; lambda, Goodman-Kruskal tau and
    the uncertainty coefficient (symmetric / row dependent / column dependent);
    gamma, Kendall's tau-b and tau-c, Somers' d; Pearson's r and Spearman's
    rho; Cohen's kappa (plus linear- and quadratic-weighted kappa for more
    than two categories) when rows and columns have the same categories.
    ``ase`` is the asymptotic standard error not assuming independence
    (SPSS's ASE1); ``t`` = value / ASE0 uses the error under independence and
    ``p_value`` is its normal (correlations: Student t) two-sided p; phi, V
    and the contingency coefficient carry the Pearson chi-square p, the
    uncertainty coefficient the likelihood-ratio p, and Goodman-Kruskal's tau
    its chi-square approximation.

    2x2 tables add ``risk``: the odds ratio and the relative risk of each
    column outcome (row 1 versus row 2) with 1 - ``alpha`` log-scale (Woolf)
    intervals.

    Layers. With ``layer`` every table is repeated per layer level and for the
    total, row labels being prefixed with the layer level. For 2x2 tables the
    ``cmh`` table holds the Cochran-Mantel-Haenszel tests of conditional
    independence (``cmh``, ``cmh_continuity``, ``cochran``) and the
    Breslow-Day and Tarone tests of homogeneity of the odds ratios, and
    ``common_odds_ratio`` the Mantel-Haenszel estimate with its
    Robins-Breslow-Greenland standard error (of the log), interval and the
    z test of ln(odds ratio) = 0.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    row, column : the two classification columns (any scalar type).
    layer : optional stratification column.
    weights : optional column of nonnegative frequency weights.
    exact : None (Fisher for 2x2 tables only), True (also r x c), False (none).
    expected : add the ``expected`` table.
    percentages : None, "all" or a list of "row", "column", "total".
    residuals : add ``residuals`` (standardized, (f - E)/sqrt(E)) and
        ``adjusted_residuals`` ((f - E)/sqrt(E (1 - r/N)(1 - c/N))).
    exact_reps, seed : Monte Carlo replications and seed.
    alpha : 1 - confidence level of the risk intervals.
    missing : "drop" (listwise over the columns used) or "raise".

    Returns
    -------
    TableSet with ``counts`` and, as requested or applicable, ``expected``,
    ``row_percent``, ``column_percent``, ``total_percent``, ``residuals``,
    ``adjusted_residuals``, ``tests``, ``measures``, ``risk``, ``cmh``,
    ``common_odds_ratio``. ``attrs`` (of the total table): n, statistic, df,
    p_value (Pearson), likelihood_ratio, p_exact, cramers_v, min_expected,
    cells_below_5, row_levels, column_levels, notes.

    Stata: ``tabulate row column, chi2 lrchi2 exact V gamma taub expected
    row column``; ``cc`` / ``cs`` / ``mhodds`` for risk and stratified
    analysis; ``kap``. SPSS: ``CROSSTABS /TABLES=row BY column BY layer
    /STATISTICS=CHISQ PHI CC LAMBDA UC GAMMA D BTAU CTAU CORR KAPPA RISK CMH``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"treated": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0], "cured": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]}
    >>> result = oe.crosstab(data, "treated", "cured")
    >>> round(result.attrs["p_exact"], 4)
    0.2063
    """
    for name, value in (("row", row), ("column", column)):
        if not isinstance(value, str):
            raise AnalysisError("invalid_spec", f"{name} must be the name of one column.")
    for name, value in (("layer", layer), ("weights", weights)):
        if value is not None and not isinstance(value, str):
            raise AnalysisError("invalid_spec", f"{name} must be the name of one column.")
    exact = check_flag(exact, "exact", optional=True)
    check_flag(expected, "expected")
    check_flag(residuals, "residuals")
    kinds = _percent_kinds(percentages)
    reps = check_count(exact_reps, "exact_reps", minimum=100)
    seed = check_count(seed, "seed", minimum=0)
    alpha = check_alpha(alpha)
    names = [row, column] + [name for name in (layer, weights) if name is not None]
    frame, dropped = select(data, names, missing=missing)
    row_codes, row_levels = group_codes(frame[row])
    column_codes, column_levels = group_codes(frame[column])
    r, c = len(row_levels), len(column_levels)
    if layer is not None:
        layer_codes, layer_levels = group_codes(frame[layer])
    else:
        layer_codes, layer_levels = torch.zeros(len(frame), dtype=torch.int64), [None]
    layers = len(layer_levels)
    if r * c > MAX_CELLS or layers > MAX_LAYERS or r * c * layers > MAX_TOTAL_CELLS:
        raise AnalysisError("too_many_categories", f"The table would have {r} x {c} cells in "
                            f"{layers} layer(s). Cross-tabulate categorical columns (group "
                            "continuous values into classes first).")
    w = weight_column(frame, weights)
    cell = (layer_codes * r + row_codes) * c + column_codes
    if w is None:
        cube = torch.bincount(cell, minlength=layers * r * c).to(FLOAT)
    else:
        cube = torch.zeros(layers * r * c, dtype=FLOAT).index_add_(0, cell, w)
    cube = cube.reshape(layers, r, c)
    integer = table_kernels.is_integer_table(cube)
    pooled = cube.sum(0)
    blocks = [("", pooled)] if layer is None else \
        [(f"{layer}={level} | ", cube[index]) for index, level in enumerate(layer_levels)] \
        + [("total | ", pooled)]
    row_names = [str(level) for level in row_levels]
    column_names = [str(level) for level in column_levels]
    row_scores, column_scores = _scores(row_levels), _scores(column_levels)
    same_levels = row_levels == column_levels

    parts: dict[str, tuple[list[Any], list[str]]] = {}

    def extend(name: str, lines: list[list[Any]], labels: list[str]) -> None:
        store = parts.setdefault(name, ([], []))
        store[0].extend(lines)
        store[1].extend(labels)

    notes: list[str] = []
    total_attrs: dict[str, Any] = {}
    for prefix, counts in blocks:
        margin_labels = [prefix + name for name in row_names + ["total"]]
        body_labels = [prefix + name for name in row_names]
        extend("counts", _with_margins(counts, integer), margin_labels)
        for kind in kinds:
            extend(f"{kind}_percent", _percent(counts, kind), margin_labels)
        if expected or residuals:
            fitted, standardized, adjusted = _residuals(counts) if float(counts.sum()) > 0 \
                else ([[None] * c] * r,) * 3
            if expected:
                extend("expected", fitted, body_labels)
            if residuals:
                extend("residuals", standardized, body_labels)
                extend("adjusted_residuals", adjusted, body_labels)
        analysis = _analyze(counts, row_scores, column_scores, same_levels, exact, reps, seed,
                            alpha) if float(counts.sum()) > 0 else \
            {"tests": [], "measures": [], "risk": [], "attrs": {},
             "notes": ["The table is empty."]}
        for key in ("tests", "measures", "risk"):
            extend(key, [list(line[1:]) for line in analysis[key]],
                   [prefix + line[0] for line in analysis[key]])
        notes += [prefix + note for note in analysis["notes"]]
        total_attrs = analysis["attrs"]                   # the last block is the total table
    columns_with_margin = column_names + ["total"]
    result: dict[str, Any] = {
        "counts": table(parts["counts"][0], columns=columns_with_margin,
                        index=parts["counts"][1])}
    for name in ("expected", "row_percent", "column_percent", "total_percent", "residuals",
                 "adjusted_residuals"):
        if name in parts:
            wide = columns_with_margin if name.endswith("percent") else column_names
            result[name] = table(parts[name][0], columns=wide, index=parts[name][1])
    layouts = {"tests": ["statistic", "df", "p_value"],
               "measures": ["value", "ase", "t", "p_value"],
               "risk": ["value", "ci_low", "ci_high"]}
    for name, header in layouts.items():
        if parts.get(name, ([], []))[0]:
            result[name] = table(parts[name][0], columns=header, index=parts[name][1])
    attrs: dict[str, Any] = {"n": _cell(float(pooled.sum()), integer), **total_attrs,
                             "row": row, "column": column, "row_levels": row_levels,
                             "column_levels": column_levels, "alpha": alpha,
                             "n_dropped": dropped, "distribution": "chi2"}
    if layer is not None:
        attrs.update({"layer": layer, "layer_levels": layer_levels})
        if r == 2 and c == 2:
            stratified = measure_kernels.mantel_haenszel(cube, alpha)
            attrs.update({"strata": stratified["strata"],
                          "strata_used": stratified["strata_used"]})
            if "tests" in stratified:
                tests = stratified["tests"]
                result["cmh"] = table([list(values) for values in tests.values()],
                                      columns=["statistic", "df", "p_value"], index=list(tests))
                attrs["cmh"], attrs["cmh_p_value"] = tests["cmh"][0], tests["cmh"][2]
            if "odds_ratio" in stratified:
                result["common_odds_ratio"] = table(
                    [list(stratified["odds_ratio"])], index=["mantel_haenszel"],
                    columns=["value", "std_error_log", "ci_low", "ci_high", "z", "p_value"])
                attrs["common_odds_ratio"] = stratified["odds_ratio"][0]
            if stratified["strata_used"] < stratified["strata"]:
                notes.append(f"{stratified['strata'] - stratified['strata_used']} layer(s) "
                             "with a zero margin are left out of the Mantel-Haenszel analysis.")
        else:
            notes.append("The Cochran-Mantel-Haenszel analysis is computed for 2x2 tables only.")
    if notes:
        attrs["notes"] = notes
    title = f"Cross-tabulation: {row} by {column}" + (f" by {layer}" if layer else "")
    return TableSet(result, title=title, **attrs)


@procedure
def tabulate(data: Any, column: str, *, weights: str | None = None, missing: str = "drop"):
    """One-way frequency table (SPSS FREQUENCIES, Stata ``tabulate oneway``).

    One row per distinct value of ``column`` (sorted; declared order for a
    categorical column), with

    * ``count``: the (weighted) frequency;
    * ``percent``: share of all rows, including missing ones;
    * ``valid_percent``: share of the non-missing rows;
    * ``cumulative_percent``: running sum of ``valid_percent``.

    Missing values of ``column`` are not an error here: they are counted in
    the final row "missing" (``missing="raise"`` refuses them instead). Rows
    with a missing weight are dropped.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    column : the column to tabulate (any scalar type).
    weights : optional column of nonnegative frequency weights.
    missing : "drop" (missing values of ``column`` get their own row; rows
        without a weight are dropped) or "raise" (both are refused).

    Returns
    -------
    One table indexed by the category labels. ``attrs``: n (valid), n_missing,
    n_total, k (number of categories), mode, n_dropped (rows without a weight).

    Stata: ``tabulate x [fw=w], missing``. SPSS: ``FREQUENCIES VARIABLES=x``.

    Example
    -------
    >>> import openecon as oe
    >>> table = oe.tabulate({"x": ["a", "b", "a", None, "a"]}, "x")
    >>> table["count"].tolist()
    [3, 1, 1]
    """
    if not isinstance(column, str) or (weights is not None and not isinstance(weights, str)):
        raise AnalysisError("invalid_spec", "column and weights must each be the name of one "
                            "column.")
    if missing not in ("drop", "raise"):
        raise AnalysisError("invalid_option", "missing must be one of: drop, raise.")
    frame, dropped = _tabulate_frame(data, column, weights, missing)
    absent = frame[column].isna()
    n_missing_rows = int(absent.sum())
    if n_missing_rows and missing == "raise":
        raise AnalysisError("missing_values", f"{n_missing_rows} observation(s) have a missing "
                            f"'{column}'. Pass missing='drop' to list them in a separate row.")
    valid = frame.loc[~absent].reset_index(drop=True)
    if not len(valid):
        raise AnalysisError("empty_sample", f"'{column}' has no non-missing values.")
    w_all = weight_column(frame, weights)
    codes, levels = group_codes(valid[column])
    if len(levels) > MAX_CELLS:
        raise AnalysisError("too_many_categories", f"'{column}' has {len(levels)} distinct "
                            "values. Tabulate a categorical column (group continuous values "
                            "into classes first).")
    keep = torch.as_tensor((~absent).to_numpy())
    w_valid = None if w_all is None else w_all[keep]
    counts = torch.zeros(len(levels), dtype=FLOAT).index_add_(
        0, codes, torch.ones(codes.numel(), dtype=FLOAT) if w_valid is None else w_valid)
    n_valid = float(counts.sum())
    n_missing = float(n_missing_rows) if w_all is None else float(w_all[~keep].sum())
    n_total = n_valid + n_missing
    integer = table_kernels.is_integer_table(counts) and n_missing == round(n_missing)
    percent = 100.0 * counts / n_total
    valid_percent = 100.0 * counts / n_valid
    rows = [[_cell(count, integer), share, valid_share, cumulative]
            for count, share, valid_share, cumulative in zip(
                counts.tolist(), percent.tolist(), valid_percent.tolist(),
                valid_percent.cumsum(0).clamp_max(100.0).tolist())]
    index = [str(level) for level in levels]
    if n_missing > 0:
        rows.append([_cell(n_missing, integer), 100.0 * n_missing / n_total, None, None])
        index.append("missing")
    attrs = {"n": _cell(n_valid, integer), "n_missing": _cell(n_missing, integer),
             "n_total": _cell(n_total, integer), "k": len(levels),
             "mode": levels[int(torch.argmax(counts))], "n_dropped": dropped,
             "label": f"Frequencies of {column}"}
    return table(rows, columns=["count", "percent", "valid_percent", "cumulative_percent"],
                 index=index, **attrs)


def _tabulate_frame(data: Any, column: str, weights: str | None,
                    missing: str) -> tuple[Any, int]:
    """The tabulated column (missing values kept) and the weights; rows without a weight
    are dropped (or refused with ``missing="raise"``) and counted."""
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    names = [column] + ([weights] if weights is not None else [])
    if weights == column:
        raise AnalysisError("invalid_spec", "A column is used in more than one role: "
                            f"{column}.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The dataset contains no observations.")
    frame = frame.loc[:, names].reset_index(drop=True)
    dropped = 0
    if weights is not None:
        unweighted = frame[weights].isna()
        dropped = int(unweighted.sum())
        if dropped and missing == "raise":
            raise AnalysisError("missing_values", f"{dropped} observation(s) have a missing "
                                f"weight in '{weights}'. Pass missing='drop' to exclude them.")
        frame = frame.loc[~unweighted].reset_index(drop=True)
        if not len(frame):
            raise AnalysisError("empty_sample", "No observation has a weight.")
    return frame, dropped
