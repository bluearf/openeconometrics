"""Student t tests and tests of equal variances: ``oe.ttest`` and ``oe.sdtest``.

Conventions follow SPSS T-TEST (both the pooled and the Welch row, Levene's test
based on the mean, effect sizes of SPSS 27+) and Stata ``ttest`` / ``sdtest`` /
``robvar`` (one-sided p-values, variance-ratio F test, W0 / W50 / W10).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import common as c
from openecon.engines import distributions as dist

_TEST_COLUMNS = ["statistic", "df", "p_value", "p_less", "p_greater", "mean_difference",
                 "std_error", "ci_low", "ci_high"]
_STAT_COLUMNS = ["n", "mean", "std_dev", "std_error", "ci_low", "ci_high"]


def _moments(x: Tensor) -> tuple[int, float, float]:
    """(n, mean, sample variance) of one sample."""
    n = x.shape[0]
    mean = float(x.mean())
    mean += float((x - mean).mean())
    variance = float((x - mean).square().sum()) / (n - 1) if n > 1 else float("nan")
    return n, mean, variance


def _stat_row(n: float, mean: float, variance: float, alpha: float) -> list[Any]:
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)
    crit = c.t_critical(alpha, n - 1)
    return [int(n), mean, sd, se, mean - crit * se, mean + crit * se]


def _test_row(difference: float, se: float, df: float, alpha: float) -> list[Any]:
    """t = difference / se against Student's t with ``df`` degrees of freedom."""
    statistic = difference / se
    if not math.isfinite(statistic):
        raise AnalysisError("invalid_option", "The t statistic overflows: mu is too far from "
                            "the data for double precision.")
    crit = c.t_critical(alpha, df)
    upper = c.t_upper(statistic, df)
    return [statistic, df, c.t_two_sided(statistic, df), c.t_upper(-statistic, df), upper,
            difference, se, difference - crit * se, difference + crit * se]


def hedges_factor(df: float) -> float | None:
    """Hedges' exact small-sample factor J = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2))."""
    if not df > 1:
        return None
    return math.exp(math.lgamma(0.5 * df) - math.lgamma(0.5 * (df - 1.0))) / math.sqrt(0.5 * df)


def _effect_table(difference: float, standardizers: dict[str, float | None]) -> Any:
    rows = [[value, c.ratio(difference, value) if value is not None else None]
            for value in standardizers.values()]
    return c.frame(rows, columns=["standardizer", "estimate"], index=list(standardizers))


def _headline(test: Any, row: str) -> dict[str, Any]:
    values = test.loc[row]
    one_sided = min(values["p_less"], values["p_greater"])
    return {"statistic": values["statistic"], "df": values["df"], "p_value": values["p_value"],
            "p_less": values["p_less"], "p_greater": values["p_greater"],
            "one_sided_p_value": one_sided, "distribution": "t"}


def ttest(data: Any, y: str, *, by: str | None = None, paired_with: str | None = None,
          mu: float = 0.0, alpha: float = 0.05, missing: str = "drop") -> TableSet:
    """Student's t test: one sample, paired samples or two independent samples.

    One sample (neither ``by`` nor ``paired_with``): H0 mean(y) = ``mu``,
    t = (ybar - mu) / (s / sqrt(n)) with n - 1 degrees of freedom.

    Paired samples (``paired_with``): the one-sample test of the differences
    d = y - paired_with against ``mu``, on the rows where both are observed.

    Independent samples (``by`` with exactly two values, compared as
    first - second in ascending order of the group value): both rows SPSS prints,

    * ``equal_variances``: pooled variance s_p^2 = [(n1-1) s1^2 + (n2-1) s2^2] / (n1+n2-2),
      se = s_p sqrt(1/n1 + 1/n2), df = n1 + n2 - 2 (Stata's default);
    * ``unequal_variances``: Welch, se = sqrt(s1^2/n1 + s2^2/n2) with Satterthwaite's
      df = (s1^2/n1 + s2^2/n2)^2 / [(s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)]
      (Stata: ``ttest y, by(g) unequal``),

    and Levene's test of equal variances based on the group means (as in SPSS).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric column to test.
    by : grouping column with two distinct values (independent samples).
    paired_with : second numeric column (paired samples).
    mu : hypothesized mean (one sample), mean difference (paired) or difference of
        means (independent). Default 0.
    alpha : 1 - confidence level of the intervals.
    missing : ``"drop"`` excludes rows with a missing value in the columns used
        (listwise; the count is in ``attrs["n_missing"]``), ``"raise"`` rejects them.

    Returns
    -------
    TableSet with tables

    * ``statistics``: n, mean, std_dev, std_error and the confidence interval of each
      mean (per variable, or per group plus ``combined``);
    * ``test``: statistic (t), df, p_value (two-sided), p_less = P(T < t),
      p_greater = P(T > t), mean_difference (minus ``mu``), std_error, ci_low, ci_high;
    * ``levene`` (independent samples): statistic (F), df1, df2, p_value;
    * ``effect_sizes``: Cohen's d, Hedges' g and Glass's delta with their standardizers.

    ``attrs``: ``kind``, ``statistic`` / ``df`` / ``p_value`` (the pooled row for
    independent samples), ``one_sided_p_value`` (the smaller tail, as SPSS prints),
    ``cohens_d``, ``hedges_g``, ``glass_delta``, ``n``, ``n_missing`` and, for paired
    samples, ``correlation`` and ``correlation_p_value``.

    Effect sizes: Cohen's d divides the mean difference by the pooled standard
    deviation (independent), the standard deviation of the differences (paired) or
    the sample standard deviation (one sample); Hedges' g multiplies d by the exact
    factor J(df) = Gamma(df/2) / (sqrt(df/2) Gamma((df-1)/2)); Glass's delta divides by
    the standard deviation of the second (control) group.

    Equivalent commands: SPSS ``T-TEST``; Stata ``ttest y == mu``, ``ttest y == x``,
    ``ttest y, by(g)`` and ``ttest y, by(g) unequal``; ``esize``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"score": [5.1, 4.9, 6.2, 5.8, 6.9, 7.4, 6.6, 7.1],
    ...         "group": ["a", "a", "a", "a", "b", "b", "b", "b"]}
    >>> result = oe.ttest(data, "score", by="group")
    >>> round(result.attrs["statistic"], 3)
    -4.33
    """
    c.check_name(y, "y")
    alpha = c.check_alpha(alpha)
    mu = c.check_number(mu, "mu")
    if by is not None and paired_with is not None:
        raise AnalysisError("invalid_spec", "Choose either by (independent samples) or "
                            "paired_with (paired samples), not both.")
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_tests import ttest as replay_ttest
        return replay_ttest(data, y, by=by, paired_with=paired_with, mu=mu,
                            alpha=alpha, missing=missing)
    if by is not None:
        return _independent(data, y, c.check_name(by, "by"), mu, alpha, missing)
    columns = [y] if paired_with is None else [y, c.check_name(paired_with, "paired_with")]
    frame, dropped = c.select(data, columns, numeric=columns, missing=missing)
    first = c.values(frame, y)
    n = first.shape[0]
    if n < 2:
        raise AnalysisError("insufficient_observations", "A t test needs at least two "
                            "complete observations.")
    kind = "one_sample" if paired_with is None else "paired"
    extra: dict[str, Any] = {}
    if paired_with is None:
        sample = first
        rows, labels = [], [y]
    else:
        second = c.values(frame, paired_with)
        sample = first - second
        rows = [_stat_row(*_moments(first), alpha), _stat_row(*_moments(second), alpha)]
        labels = [y, paired_with, "difference"]
        extra = _paired_correlation(first, second)
    # Deviations are taken from the sample mean (shift) and the tested difference is
    # (mean - shift) + (shift - mu): exact when mu is close to a large common level
    # (shift - mu is then exact) and when it is far from the data.
    centred, shift = c.centre(sample)
    _, offset, variance = _moments(centred)
    if c.negligible(variance * (n - 1), float(sample.square().sum()), 1e-30):
        raise AnalysisError("zero_variance", "The t statistic is undefined because the "
                            f"{'differences' if paired_with else 'values'} do not vary.")
    excess = offset + (shift - mu)
    rows.append(_stat_row(n, offset + shift, variance, alpha))
    sd = math.sqrt(variance)
    test = c.frame([_test_row(excess, sd / math.sqrt(n), n - 1, alpha)],
                   columns=_TEST_COLUMNS, index=[kind])
    factor = hedges_factor(n - 1)
    effects = _effect_table(excess, {
        "cohens_d": sd, "hedges_g": sd / factor if factor else None})
    return TableSet(
        {"statistics": c.frame(rows, columns=_STAT_COLUMNS, index=labels), "test": test,
         "effect_sizes": effects},
        title={"one_sample": f"One-sample t test of {y} = {mu:g}",
               "paired": f"Paired t test of {y} - {paired_with} = {mu:g}"}[kind],
        kind=kind, **_headline(test, kind), n=n, n_missing=dropped, mu=mu, alpha=alpha,
        cohens_d=effects.loc["cohens_d", "estimate"], hedges_g=effects.loc["hedges_g", "estimate"],
        glass_delta=None, missing="listwise", **extra)


def _paired_correlation(first: Tensor, second: Tensor) -> dict[str, Any]:
    a, b = first - first.mean(), second - second.mean()
    denominator = c.root_product(float(a.square().sum()), float(b.square().sum()))
    if denominator == 0.0:
        return {"correlation": None, "correlation_p_value": None}
    r = max(-1.0, min(1.0, float((a * b).sum()) / denominator))
    df = first.shape[0] - 2
    p_value = None
    if df > 0:
        p_value = 0.0 if abs(r) == 1.0 else c.t_two_sided(r * math.sqrt(df / (1 - r * r)), df)
    return {"correlation": r, "correlation_p_value": p_value}


def _independent(data: Any, y: str, by: str, mu: float, alpha: float, missing: str) -> TableSet:
    frame, dropped = c.select(data, [y, by], numeric=[y], missing=missing)
    codes, labels = c.group_codes(frame, by)
    if len(labels) != 2:
        raise AnalysisError("invalid_groups", f"by='{by}' must have exactly two distinct values "
                            f"for an independent-samples t test; it has {len(labels)}. Use "
                            "oe.oneway for more than two groups.")
    values, shift = c.centre(c.values(frame, y))
    moments = c.group_moments(values, codes, 2)
    n1, n2 = (int(v) for v in moments.n.tolist())
    if min(n1, n2) < 2:
        raise AnalysisError("insufficient_observations", "Each group needs at least two "
                            "observations for an independent-samples t test.")
    m1, m2 = moments.mean.tolist()
    v1, v2 = moments.var.tolist()
    difference = m1 - m2 - mu
    pooled = (moments.ss[0].item() + moments.ss[1].item()) / (n1 + n2 - 2)
    if c.negligible(pooled * (n1 + n2 - 2), float(values.square().sum()), 1e-30):
        raise AnalysisError("zero_variance", "The t statistic is undefined because the outcome "
                            "does not vary within either group.")
    rows = [_test_row(difference, math.sqrt(pooled * (1 / n1 + 1 / n2)), n1 + n2 - 2, alpha)]
    a, b = v1 / n1, v2 / n2
    welch_df = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
    rows.append(_test_row(difference, math.sqrt(a + b), welch_df, alpha))
    test = c.frame(rows, columns=_TEST_COLUMNS,
                   index=["equal_variances", "unequal_variances"])
    _, mean, variance = _moments(values)
    statistics = c.frame(
        [_stat_row(n1, m1 + shift, v1, alpha), _stat_row(n2, m2 + shift, v2, alpha),
         _stat_row(n1 + n2, mean + shift, variance, alpha)],
        columns=_STAT_COLUMNS, index=[str(labels[0]), str(labels[1]), "combined"])
    lev = c.levene(values, codes, 2, moments.mean)
    levene = c.frame([[lev["statistic"], lev["df1"], lev["df2"], lev["p_value"]]],
                     columns=["statistic", "df1", "df2", "p_value"], index=["levene_mean"])
    factor = hedges_factor(n1 + n2 - 2)
    sp = math.sqrt(pooled)
    effects = _effect_table(difference, {
        "cohens_d": sp, "hedges_g": sp / factor if factor else None,
        "glass_delta": math.sqrt(v2) if v2 > 0 else None})
    return TableSet(
        {"statistics": statistics, "test": test, "levene": levene, "effect_sizes": effects},
        title=f"Independent-samples t test of {y} by {by} ({labels[0]} - {labels[1]})",
        kind="independent", **_headline(test, "equal_variances"), n=n1 + n2, n_missing=dropped,
        mu=mu, alpha=alpha, groups=labels, cohens_d=effects.loc["cohens_d", "estimate"],
        hedges_g=effects.loc["hedges_g", "estimate"],
        glass_delta=effects.loc["glass_delta", "estimate"],
        glass_delta_first=c.ratio(difference, math.sqrt(v1)) if v1 > 0 else None,
        levene_statistic=lev["statistic"], levene_p_value=lev["p_value"], missing="listwise")


def trimmed_means(y: Tensor, codes: Tensor, groups: int, cut: float = 0.05) -> Tensor:
    """Group means after removing floor(cut * n_i) observations from each tail."""
    ordered, start, counts = c.sorted_by_group(y, codes, groups)
    group = torch.repeat_interleave(torch.arange(groups), counts)
    rank = torch.arange(y.shape[0]) - start[group]
    trim = torch.floor(cut * counts.to(torch.float64)).to(torch.int64)
    keep = (rank >= trim[group]) & (rank < (counts - trim)[group])
    total = c.group_sum(ordered[keep], group[keep], groups)
    return total / (counts - 2 * trim).to(torch.float64)


def sdtest(data: Any, y: str, *, by: str | None = None, sd: float | None = None,
           center: str = "mean", alpha: float = 0.05, missing: str = "drop") -> TableSet:
    """Tests of variances: variance-ratio F, chi-square, Levene and Brown-Forsythe.

    With ``by`` (two or more groups):

    * ``variance_ratio`` (two groups only): F = s1^2 / s2^2 with (n1 - 1, n2 - 1)
      degrees of freedom, groups in ascending order of their value; p_value is
      2 * min(P(F < f), P(F > f)) as in Stata's ``sdtest``. It assumes normality.
    * ``robust``: Levene's statistic W0 (absolute deviations from the group mean)
      and the Brown-Forsythe variants W50 (median) and W10 (10% trimmed mean: the
      mean after removing floor(0.05 n_i) observations from each tail), each the
      one-way ANOVA F of z_ij = |y_ij - center_i| with (k - 1, N - k) degrees of
      freedom (Stata's ``robvar``; SPSS's Levene test is W0). ``center`` chooses
      which of the three is reported in ``attrs`` (``"mean"``, ``"median"``,
      ``"trimmed"``).

    Without ``by``, ``sd`` is required: the one-sample chi-square test of
    H0 sd(y) = ``sd``, chi2 = (n - 1) s^2 / sd^2 with n - 1 degrees of freedom
    (Stata ``sdtest y == #``).

    Parameters
    ----------
    data, y : table and numeric column.
    by : grouping column (two or more distinct values).
    sd : hypothesized standard deviation of the one-sample test.
    center : headline version of the robust test.
    alpha : 1 - confidence level of the reported intervals of the means.
    missing : ``"drop"`` (listwise, default) or ``"raise"``.

    Returns
    -------
    TableSet with ``statistics`` (n, mean, std_dev, std_error, ci_low, ci_high per
    group), ``variance_ratio`` or ``chi2`` (statistic, df / df1, df2, p_value, p_less,
    p_greater) and ``robust`` (rows w0, w50, w10: statistic, df1, df2, p_value).
    ``attrs``: ``statistic``, ``df1``, ``df2``, ``p_value`` of the chosen robust test
    (or of the chi-square test), ``n``, ``n_missing``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"score": [5.1, 4.9, 6.2, 5.8, 6.9, 9.4, 4.6, 7.1],
    ...         "group": ["a", "a", "a", "a", "b", "b", "b", "b"]}
    >>> result = oe.sdtest(data, "score", by="group", center="median")
    >>> sorted(result)
    ['robust', 'statistics', 'variance_ratio']
    """
    c.check_name(y, "y")
    alpha = c.check_alpha(alpha)
    c.check_choice(center, "center", ("mean", "median", "trimmed"))
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_tests import sdtest as replay_sdtest
        return replay_sdtest(data, y, by=by, sd=sd, center=center, alpha=alpha,
                             missing=missing)
    if by is None:
        return _sd_one_sample(data, y, sd, alpha, missing)
    if sd is not None:
        raise AnalysisError("invalid_spec", "sd is the hypothesized value of the one-sample "
                            "test; omit it when by is given.")
    frame, dropped = c.select(data, [y, c.check_name(by, "by")], numeric=[y], missing=missing)
    codes, labels = c.group_codes(frame, by)
    groups = len(labels)
    if groups < 2:
        raise AnalysisError("invalid_groups", f"by='{by}' must have at least two distinct "
                            "values.")
    values, shift = c.centre(c.values(frame, y))
    moments = c.group_moments(values, codes, groups)
    if bool((moments.n < 2).any()):
        raise AnalysisError("insufficient_observations", "Each group needs at least two "
                            "observations to estimate its variance.")
    tables = {"statistics": c.frame(
        [_stat_row(n, mean + shift, var, alpha) for n, mean, var in
         zip(moments.n.tolist(), moments.mean.tolist(), moments.var.tolist(), strict=True)],
        columns=_STAT_COLUMNS, index=[str(label) for label in labels])}
    notes = []
    if groups == 2:
        v1, v2 = moments.var.tolist()
        df1, df2 = int(moments.n[0]) - 1, int(moments.n[1]) - 1
        if v1 > 0.0 and v2 > 0.0:
            f = v1 / v2
            lower, upper = dist.f_cdf(f, df1, df2), dist.f_sf(f, df1, df2)
            row = [f, df1, df2, min(1.0, 2.0 * min(lower, upper)), lower, upper]
        else:
            row = [None, df1, df2, None, None, None]
            notes.append("The variance ratio is undefined because a group has zero variance.")
        tables["variance_ratio"] = c.frame(
            [row], columns=["statistic", "df1", "df2", "p_value", "p_less", "p_greater"],
            index=["f"])
    else:
        notes.append("The variance-ratio F test compares two groups; with more groups only the "
                     "robust tests are reported.")
    centers = {"w0": moments.mean, "w50": c.group_medians(values, codes, groups),
               "w10": trimmed_means(values, codes, groups)}
    results = {name: c.levene(values, codes, groups, center_values)
               for name, center_values in centers.items()}
    tables["robust"] = c.frame(
        [[r["statistic"], r["df1"], r["df2"], r["p_value"]] for r in results.values()],
        columns=["statistic", "df1", "df2", "p_value"], index=list(results))
    chosen = results[{"mean": "w0", "median": "w50", "trimmed": "w10"}[center]]
    return TableSet(tables, title=f"Tests of equal variances of {y} by {by}",
                    statistic=chosen["statistic"], df1=chosen["df1"], df2=chosen["df2"],
                    p_value=chosen["p_value"], distribution="F", center=center,
                    n=int(moments.n.sum()), n_missing=dropped, groups=labels, alpha=alpha,
                    notes=notes, missing="listwise")


def _sd_one_sample(data: Any, y: str, sd: Any, alpha: float, missing: str) -> TableSet:
    if sd is None:
        raise AnalysisError("invalid_spec", "Give by (a grouping column) to compare variances "
                            "between groups, or sd (a hypothesized standard deviation) for the "
                            "one-sample test.")
    sd = c.check_number(sd, "sd")
    if not sd > 0.0:
        raise AnalysisError("invalid_option", "sd must be positive.")
    frame, dropped = c.select(data, [y], numeric=[y], missing=missing)
    values, shift = c.centre(c.values(frame, y))
    n, mean, variance = _moments(values)
    mean += shift
    if n < 2:
        raise AnalysisError("insufficient_observations", "A variance test needs at least two "
                            "observations.")
    relative = math.sqrt(variance) / sd
    statistic = (n - 1) * relative * relative
    if not math.isfinite(statistic):
        raise AnalysisError("invalid_option", f"sd={sd:g} is too small relative to the sample "
                            f"standard deviation ({math.sqrt(variance):g}): the chi-square "
                            "statistic overflows. Check the units of sd.")
    lower, upper = dist.chi2_cdf(statistic, n - 1), dist.chi2_sf(statistic, n - 1)
    p_value = min(1.0, 2.0 * min(lower, upper))
    return TableSet(
        {"statistics": c.frame([_stat_row(n, mean, variance, alpha)] if variance > 0 else
                               [[n, mean, 0.0, 0.0, mean, mean]], columns=_STAT_COLUMNS,
                               index=[y]),
         "chi2": c.frame([[statistic, n - 1, p_value, lower, upper]],
                         columns=["statistic", "df", "p_value", "p_less", "p_greater"],
                         index=["chi2"])},
        title=f"One-sample test of sd({y}) = {sd:g}", statistic=statistic, df=n - 1,
        p_value=p_value, distribution="chi2", n=n, n_missing=dropped, sd=sd, alpha=alpha,
        missing="listwise")
