"""One-way analysis of variance with post hoc comparisons: ``oe.oneway``.

Output follows SPSS ONEWAY (Descriptives, ANOVA, Tests of Homogeneity of
Variances, Robust Tests of Equality of Means, ANOVA Effect Sizes, Multiple
Comparisons) plus Bartlett's test, which Stata's ``oneway`` prints.
"""

from __future__ import annotations

import math
from typing import Any

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import common as c
from openecon.econometrics.stats import posthoc as ph


def welch_anova(moments: c.GroupMoments) -> dict[str, Any]:
    """Welch's (1951) heteroskedasticity-robust F test of equal means.

    w_i = n_i / s_i^2, ybar_w = sum w_i ybar_i / sum w_i,
    lam = sum (1 - w_i / W)^2 / (n_i - 1),
    F = [sum w_i (ybar_i - ybar_w)^2 / (k - 1)] / [1 + 2 (k - 2) lam / (k^2 - 1)],
    df1 = k - 1, df2 = (k^2 - 1) / (3 lam).
    """
    k = moments.n.shape[0]
    w = moments.n / moments.var
    total = float(w.sum())
    centre = float((w * moments.mean).sum()) / total
    lam = float(((1.0 - w / total).square() / (moments.n - 1.0)).sum())
    numerator = float((w * (moments.mean - centre).square()).sum()) / (k - 1)
    statistic = numerator / (1.0 + 2.0 * (k - 2) * lam / (k * k - 1.0))
    df2 = (k * k - 1.0) / (3.0 * lam)
    return {"statistic": statistic, "df1": k - 1, "df2": df2,
            "p_value": c.f_upper(statistic, k - 1, df2)}


def brown_forsythe_anova(moments: c.GroupMoments, between: float) -> dict[str, Any]:
    """Brown and Forsythe's (1974) F* test of equal means.

    F* = sum n_i (ybar_i - ybar)^2 / sum (1 - n_i / N) s_i^2, df1 = k - 1,
    1 / df2 = sum c_i^2 / (n_i - 1) with c_i = (1 - n_i/N) s_i^2 / sum (1 - n_j/N) s_j^2.
    """
    k = moments.n.shape[0]
    total = float(moments.n.sum())
    parts = (1.0 - moments.n / total) * moments.var
    denominator = float(parts.sum())
    statistic = between / denominator
    share = parts / denominator
    df2 = 1.0 / float((share.square() / (moments.n - 1.0)).sum())
    return {"statistic": statistic, "df1": k - 1, "df2": df2,
            "p_value": c.f_upper(statistic, k - 1, df2)}


def bartlett(moments: c.GroupMoments) -> dict[str, Any]:
    """Bartlett's (1937) chi-square test of equal variances (assumes normality).

    chi2 = [(N - k) ln s_p^2 - sum (n_i - 1) ln s_i^2] / C,
    C = 1 + [sum 1/(n_i - 1) - 1/(N - k)] / (3 (k - 1)),  df = k - 1.
    """
    k = moments.n.shape[0]
    df = moments.n - 1.0
    total = float(df.sum())
    pooled = float(moments.ss.sum()) / total
    numerator = total * math.log(pooled) - float((df * moments.var.log()).sum())
    correction = 1.0 + (float((1.0 / df).sum()) - 1.0 / total) / (3.0 * (k - 1))
    statistic = max(numerator / correction, 0.0)
    return {"statistic": statistic, "df": k - 1, "p_value": c.chi2_upper(statistic, k - 1)}


def _descriptives(labels: list[Any], moments: c.GroupMoments, grand: float, total_ss: float,
                  alpha: float, shift: float, extremes: tuple[Any, Any]) -> Any:
    rows = []
    low, high = extremes
    stats = list(zip(moments.n.tolist(), moments.mean.tolist(), moments.var.tolist(),
                     low.tolist(), high.tolist(), strict=True))
    count = float(moments.n.sum())
    stats.append((count, grand, total_ss / (count - 1), float(low.min()), float(high.max())))
    for n, mean, var, low, high in stats:
        mean = mean + shift
        if n > 1:
            sd = math.sqrt(var)
            se = sd / math.sqrt(n)
            crit = c.t_critical(alpha, n - 1)
            rows.append([int(n), mean, sd, se, mean - crit * se, mean + crit * se, low, high])
        else:
            rows.append([int(n), mean, None, None, None, None, low, high])
    return c.frame(rows, columns=["n", "mean", "std_dev", "std_error", "ci_low", "ci_high",
                                  "min", "max"],
                   index=[*(str(label) for label in labels), "Total"])


def oneway(data: Any, y: str, by: str, *, posthoc: list[str] | None = None, alpha: float = 0.05,
           welch: bool = True, control: Any = None, missing: str = "drop") -> TableSet:
    """One-way analysis of variance with homogeneity tests, robust tests and post hoc tests.

    Model: y_ij = mu_i + e_ij for groups i = 1..k of sizes n_i (N in total).

        SS_between = sum_i n_i (ybar_i - ybar)^2,          df = k - 1
        SS_within  = sum_ij (y_ij - ybar_i)^2,             df = N - k
        F = MS_between / MS_within  ~  F(k - 1, N - k) under equal means.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome column.
    by : grouping column (two or more distinct values; groups are ordered by value).
    posthoc : list of pairwise-comparison methods among ``"tukey"`` (Tukey HSD /
        Tukey-Kramer), ``"bonferroni"``, ``"sidak"``, ``"scheffe"``, ``"lsd"``,
        ``"games_howell"``, ``"dunnett"`` (two-sided, every group against ``control``)
        and ``"holm"``.
    alpha : 1 - confidence level of every interval.
    welch : also report the Welch and Brown-Forsythe robust tests of equal means.
    control : label of the control group for Dunnett's test (default: the first
        group in ascending order; SPSS's default is the last).
    missing : ``"drop"`` excludes rows with a missing ``y`` or ``by`` (listwise;
        counted in ``attrs["n_missing"]``), ``"raise"`` rejects them.

    Returns
    -------
    TableSet with tables

    * ``descriptives``: n, mean, std_dev, std_error, confidence interval of the mean,
      min and max for each group and in total;
    * ``anova``: rows ``between``, ``within``, ``total`` with ss, df, ms, statistic (F),
      p_value;
    * ``homogeneity``: Levene's test based on the mean, on the median (Brown-Forsythe)
      and on the median with Satterthwaite-adjusted denominator df, and Bartlett's
      chi-square test (statistic, df1, df2, p_value);
    * ``robust`` (``welch=True``): Welch's F and Brown-Forsythe's F* with df1, df2,
      p_value;
    * ``effect_sizes``: eta squared = SS_b / SS_t, epsilon squared
      = (SS_b - (k-1) MS_w) / SS_t and the fixed-effect omega squared
      = (SS_b - (k-1) MS_w) / (SS_t + MS_w) (negative estimates are not truncated);
    * ``posthoc_<method>``: group_i, group_j, mean_difference (i - j), std_error,
      statistic (t = difference / std_error), df, p_value (adjusted), ci_low, ci_high.

    ``attrs``: ``statistic``, ``df1``, ``df2``, ``p_value`` of the ANOVA F test,
    ``eta_squared``, ``epsilon_squared``, ``omega_squared``, ``rmse``, ``n``, ``groups``,
    ``n_missing``, ``notes``.

    Post hoc methods use the pooled within-group mean square with N - k degrees of
    freedom, except Games-Howell, which uses separate variances with Welch degrees
    of freedom. Tukey and Games-Howell p-values and intervals come from the
    studentized range distribution (q = |t| sqrt(2)); Scheffe's from
    F = t^2 / (k - 1); Dunnett's from the exact two-sided many-one t distribution.

    Equivalent commands: SPSS ``ONEWAY y BY g /STATISTICS DESCRIPTIVES HOMOGENEITY
    WELCH BROWNFORSYTHE /POSTHOC=TUKEY ...``; Stata ``oneway y g, tabulate bonferroni
    sidak scheffe``, ``pwmean y, over(g) mcompare(tukey) effects``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [4.1, 5.0, 4.6, 6.2, 6.8, 6.1, 8.0, 7.4, 7.9],
    ...         "g": ["a", "a", "a", "b", "b", "b", "c", "c", "c"]}
    >>> result = oe.oneway(data, "y", "g", posthoc=["tukey"])
    >>> round(result.attrs["statistic"], 2)
    51.47
    """
    c.check_name(y, "y")
    c.check_name(by, "by")
    alpha = c.check_alpha(alpha)
    c.check_flag(welch, "welch")
    methods = c.name_list(posthoc, "posthoc", minimum=0, noun="method", example="tukey")
    for method in methods:
        c.check_choice(method, "posthoc entries", ph.METHODS)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_tests import oneway as replay_oneway
        return replay_oneway(data, y, by, methods=methods, alpha=alpha, welch=welch,
                             control=control, missing=missing)
    frame, dropped = c.select(data, [y, by], numeric=[y], missing=missing)
    codes, labels = c.group_codes(frame, by)
    k = len(labels)
    raw = c.values(frame, y)
    values, shift = c.centre(raw)
    total = values.shape[0]
    if k < 2:
        raise AnalysisError("invalid_groups", f"by='{by}' must have at least two distinct "
                            "values.")
    if total - k < 1:
        raise AnalysisError("insufficient_observations", "One-way ANOVA needs more observations "
                            "than groups.")
    moments = c.group_moments(values, codes, k)
    grand = float((moments.n * moments.mean).sum()) / total
    between = float((moments.n * (moments.mean - grand).square()).sum())
    within = float(moments.ss.sum())
    if c.negligible(within, float(values.square().sum()), 1e-30):
        raise AnalysisError("zero_variance", "The outcome does not vary within any group, so the "
                            "F statistic is undefined.")
    df1, df2 = k - 1, total - k
    msb, msw = between / df1, within / df2
    f = msb / msw
    p_value = c.f_upper(f, df1, df2)
    anova = c.frame([[between, df1, msb, f, p_value], [within, df2, msw, None, None],
                     [between + within, total - 1, None, None, None]],
                    columns=["ss", "df", "ms", "statistic", "p_value"],
                    index=["between", "within", "total"])
    notes: list[str] = []
    tables = {"descriptives": _descriptives(labels, moments, grand, between + within, alpha,
                                            shift, c.group_extremes(raw, codes, k)),
              "anova": anova}
    by_mean = c.levene(values, codes, k, moments.mean)
    by_median = c.levene(values, codes, k, c.group_medians(values, codes, k))
    rows = [[by_mean["statistic"], df1, df2, by_mean["p_value"]],
            [by_median["statistic"], df1, df2, by_median["p_value"]],
            [by_median["statistic"], df1, by_median["adjusted_df2"],
             c.f_upper(by_median["statistic"], df1, by_median["adjusted_df2"] or 0.0)]]
    spread = bool((moments.n > 1).all()) and bool((moments.var > 0).all())
    if spread:
        chi2 = bartlett(moments)
        rows.append([chi2["statistic"], chi2["df"], None, chi2["p_value"]])
    else:
        rows.append([None, df1, None, None])
        notes.append("Bartlett's test and the robust tests of equal means need at least two "
                     "observations and a positive variance in every group.")
    tables["homogeneity"] = c.frame(rows, columns=["statistic", "df1", "df2", "p_value"],
                                    index=["levene_mean", "levene_median",
                                           "levene_median_adjusted_df", "bartlett"])
    if welch:
        robust = [welch_anova(moments), brown_forsythe_anova(moments, between)] if spread \
            else [{"statistic": None, "df1": df1, "df2": None, "p_value": None}] * 2
        tables["robust"] = c.frame([[r["statistic"], r["df1"], r["df2"], r["p_value"]]
                                    for r in robust],
                                   columns=["statistic", "df1", "df2", "p_value"],
                                   index=["welch", "brown_forsythe"])
    sst = between + within
    eta, epsilon = between / sst, (between - df1 * msw) / sst
    omega = (between - df1 * msw) / (sst + msw)
    tables["effect_sizes"] = c.frame([[eta], [epsilon], [omega]], columns=["estimate"],
                                     index=["eta_squared", "epsilon_squared", "omega_squared"])
    position = 0
    if "dunnett" in methods:
        if control is not None:
            matches = [i for i, label in enumerate(labels) if label == control
                       or str(label) == str(control)]
            if not matches:
                raise AnalysisError("invalid_option", f"control={control!r} is not a value of "
                                    f"'{by}'. Its values are: "
                                    f"{', '.join(map(str, labels[:20]))}.")
            position = matches[0]
    elif control is not None:
        raise AnalysisError("invalid_option", "control applies to posthoc=['dunnett'] only.")
    for method in methods:
        tables[f"posthoc_{method}"] = ph.compare(method, labels, moments, msw, df2, alpha,
                                                 position)
    return TableSet(tables, title=f"One-way ANOVA of {y} by {by}", statistic=f, df1=df1, df2=df2,
                    p_value=p_value, distribution="F", eta_squared=eta, epsilon_squared=epsilon,
                    omega_squared=omega, rmse=math.sqrt(msw), n=total, n_missing=dropped,
                    groups=labels, alpha=alpha, notes=notes, missing="listwise")
