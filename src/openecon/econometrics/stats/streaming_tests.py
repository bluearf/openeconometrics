"""Exact classical tests on replayable sources, including disk-backed group ranks."""
from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.engines import distributions as dist
from openecon.resources import plan_workspace
from . import common as c
from . import posthoc as ph
from .replay import Grouped, Replay, numeric_moments
from .ttest import _effect_table, _headline, _stat_row, _test_row, hedges_factor, _STAT_COLUMNS, _TEST_COLUMNS


def ttest(data: Any, y: str, *, by: str | None, paired_with: str | None, mu: float,
          alpha: float, missing: str) -> TableSet:
    if by is not None:
        return independent(data, y, c.check_name(by, "by"), mu, alpha, missing)
    names = [y] if paired_with is None else [y, c.check_name(paired_with, "paired_with")]
    sample = Replay(data, names, names, missing)
    state = numeric_moments(sample, names, difference=paired_with is not None)
    n = state.n
    if n < 2:
        raise AnalysisError("insufficient_observations", "A t test needs at least two complete observations.")
    kind = "one_sample" if paired_with is None else "paired"
    position = 0 if paired_with is None else 2
    variance = float(state.sscp[position, position]) / (n - 1)
    if c.negligible(variance * (n - 1), float(state.raw_ss[position]), 1e-30):
        raise AnalysisError("zero_variance", "The t statistic is undefined because the values do not vary.")
    difference = float(state.mean[position]) + (float(state.anchor[position]) - mu)
    mean = float(state.location[position])
    extra: dict[str, Any] = {}
    if paired_with is None:
        rows, labels = [], [y]
    else:
        rows = [_stat_row(n, float(state.location[i]), float(state.sscp[i, i]) / (n - 1), alpha) for i in range(2)]
        labels = [y, paired_with, "difference"]
        denominator = c.root_product(float(state.sscp[0, 0]), float(state.sscp[1, 1]))
        r = max(-1.0, min(1.0, float(state.sscp[0, 1]) / denominator)) if denominator else None
        p = None if r is None or n < 3 else 0.0 if abs(r) == 1 else c.t_two_sided(r * math.sqrt((n - 2) / (1 - r*r)), n - 2)
        extra = {"correlation": r, "correlation_p_value": p}
    rows.append(_stat_row(n, mean, variance, alpha))
    sd = math.sqrt(variance)
    test = c.frame([_test_row(difference, sd / math.sqrt(n), n - 1, alpha)], columns=_TEST_COLUMNS, index=[kind])
    factor = hedges_factor(n - 1)
    effects = _effect_table(difference, {"cohens_d": sd, "hedges_g": sd / factor if factor else None})
    return TableSet({"statistics": c.frame(rows, columns=_STAT_COLUMNS, index=labels), "test": test,
                     "effect_sizes": effects},
                    title=f"One-sample t test of {y} = {mu:g}" if paired_with is None else f"Paired t test of {y} - {paired_with} = {mu:g}",
                    kind=kind, **_headline(test, kind), n=n, n_missing=sample.dropped, mu=mu, alpha=alpha,
                    cohens_d=effects.loc["cohens_d", "estimate"], hedges_g=effects.loc["hedges_g", "estimate"],
                    glass_delta=None, missing="listwise", **extra, **sample.attrs())


def independent(data: Any, y: str, by: str, mu: float, alpha: float, missing: str) -> TableSet:
    sample = Replay(data, [y, by], [y], missing)
    grouped = Grouped(sample, y, by)
    try:
        moments, labels = grouped.moments, grouped.labels
        if len(labels) != 2:
            raise AnalysisError("invalid_groups", f"by='{by}' must have exactly two distinct values for an independent-samples t test.")
        n1, n2 = (int(v) for v in moments.n.tolist())
        if min(n1, n2) < 2:
            raise AnalysisError("insufficient_observations", "Each group needs at least two observations for an independent-samples t test.")
        m1, m2 = moments.mean.tolist()
        v1, v2 = moments.var.tolist()
        difference = m1 - m2 - mu
        pooled = float(moments.ss.sum()) / (n1+n2-2)
        if c.negligible(pooled * (n1+n2-2), float(grouped.overall.sscp[0, 0]), 1e-30):
            raise AnalysisError("zero_variance", "The outcome does not vary within either group.")
        a, b = v1/n1, v2/n2
        df = (a+b)**2 / (a*a/(n1-1)+b*b/(n2-1))
        test = c.frame([_test_row(difference, math.sqrt(pooled*(1/n1+1/n2)), n1+n2-2, alpha),
                        _test_row(difference, math.sqrt(a+b), df, alpha)], columns=_TEST_COLUMNS,
                       index=["equal_variances", "unequal_variances"])
        statistics = c.frame([_stat_row(n1, m1+grouped.shift, v1, alpha),
                               _stat_row(n2, m2+grouped.shift, v2, alpha),
                               _stat_row(n1+n2, float(grouped.overall.location[0]), float(grouped.overall.sscp[0,0])/(n1+n2-1), alpha)],
                              columns=_STAT_COLUMNS, index=[str(labels[0]), str(labels[1]), "combined"])
        lev = grouped.levene(moments.mean)
        levene = c.frame([[lev["statistic"], lev["df1"], lev["df2"], lev["p_value"]]],
                         columns=["statistic", "df1", "df2", "p_value"], index=["levene_mean"])
        factor = hedges_factor(n1+n2-2)
        sp = math.sqrt(pooled)
        effects = _effect_table(difference, {"cohens_d": sp, "hedges_g": sp/factor if factor else None,
                                             "glass_delta": math.sqrt(v2) if v2 > 0 else None})
        return TableSet({"statistics": statistics, "test": test, "levene": levene, "effect_sizes": effects},
                        title=f"Independent-samples t test of {y} by {by} ({labels[0]} - {labels[1]})",
                        kind="independent", **_headline(test, "equal_variances"), n=n1+n2, n_missing=sample.dropped,
                        mu=mu, alpha=alpha, groups=labels, cohens_d=effects.loc["cohens_d", "estimate"],
                        hedges_g=effects.loc["hedges_g", "estimate"], glass_delta=effects.loc["glass_delta", "estimate"],
                        glass_delta_first=c.ratio(difference, math.sqrt(v1)) if v1 > 0 else None,
                        levene_statistic=lev["statistic"], levene_p_value=lev["p_value"], missing="listwise", **sample.attrs())
    finally:
        grouped.close()


def oneway(data: Any, y: str, by: str, *, methods: list[str], alpha: float,
           welch: bool, control: Any, missing: str) -> TableSet:
    sample = Replay(data, [y, by], [y], missing)
    grouped = Grouped(sample, y, by, ordered=True)
    try:
        return _oneway_finish(grouped, y, by, methods, alpha, welch, control)
    finally:
        grouped.close()


def _oneway_finish(grouped: Grouped, y: str, by: str, methods: list[str],
                   alpha: float, welch: bool, control: Any) -> TableSet:
    from .oneway import _descriptives, bartlett, welch_anova, brown_forsythe_anova
    sample = grouped.sample
    dropped = sample.dropped
    moments, labels, shift = grouped.moments, grouped.labels, grouped.shift
    k, total = len(labels), sample.n
    if k < 2:
        raise AnalysisError("invalid_groups", f"by='{by}' must have at least two distinct values.")
    if total - k < 1:
        raise AnalysisError("insufficient_observations", "One-way ANOVA needs more observations than groups.")
    grand = float((moments.n * moments.mean).sum()) / total
    between = float((moments.n * (moments.mean - grand).square()).sum())
    within = float(moments.ss.sum())
    if c.negligible(within, float(grouped.overall.sscp[0, 0]), 1e-30):
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
                                            shift, (torch.tensor([float(grouped.states[label].low[0]) for label in labels], dtype=torch.float64),
                                                     torch.tensor([float(grouped.states[label].high[0]) for label in labels], dtype=torch.float64))),
              "anova": anova}
    by_mean = grouped.levene(moments.mean)
    by_median = grouped.levene(grouped.centres())
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
    if methods:
        pair_counts = [k - 1 if method == "dunnett" else k*(k-1)//2 for method in methods]
        plan_workspace("chunked oneway retained posthoc tables", {
            "all_numeric_posthoc_result_frames": 56 * sum(pair_counts),
            "largest_pair_distribution_buffers": 256 * max(pair_counts),
            "helper_source_and_group_buffers": sample.plan.estimated_bytes})
    for method in methods:
        tables[f"posthoc_{method}"] = ph.compare(method, labels, moments, msw, df2, alpha,
                                                 position)
    return TableSet(tables, title=f"One-way ANOVA of {y} by {by}", statistic=f, df1=df1, df2=df2,
                    p_value=p_value, distribution="F", eta_squared=eta, epsilon_squared=epsilon,
                    omega_squared=omega, rmse=math.sqrt(msw), n=total, n_missing=dropped,
                    groups=labels, alpha=alpha, notes=notes, missing="listwise", **sample.attrs())


def sdtest(data: Any, y: str, *, by: str | None, sd: float | None,
           center: str, alpha: float, missing: str) -> TableSet:
    if by is None:
        return sd_one_sample(data, y, sd, alpha, missing)
    if sd is not None:
        raise AnalysisError("invalid_spec", "sd applies to a one-sample variance test only.")
    by = c.check_name(by, "by")
    sample = Replay(data, [y, by], [y], missing)
    grouped = Grouped(sample, y, by, ordered=True)
    try:
        return _sd_finish(grouped, y, by, center, alpha)
    finally:
        grouped.close()


def _sd_finish(grouped: Grouped, y: str, by: str, center: str, alpha: float) -> TableSet:
    sample, labels, dropped = grouped.sample, grouped.labels, grouped.sample.dropped
    groups = len(labels)
    if groups < 2:
        raise AnalysisError("invalid_groups", f"by='{by}' must have at least two distinct "
                            "values.")
    moments, shift = grouped.moments, grouped.shift
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
    centers = {"w0": moments.mean, "w50": grouped.centres(),
               "w10": grouped.centres(trimmed=True)}
    results = {name: grouped.levene(center_values)
               for name, center_values in centers.items()}
    tables["robust"] = c.frame(
        [[r["statistic"], r["df1"], r["df2"], r["p_value"]] for r in results.values()],
        columns=["statistic", "df1", "df2", "p_value"], index=list(results))
    chosen = results[{"mean": "w0", "median": "w50", "trimmed": "w10"}[center]]
    return TableSet(tables, title=f"Tests of equal variances of {y} by {by}",
                    statistic=chosen["statistic"], df1=chosen["df1"], df2=chosen["df2"],
                    p_value=chosen["p_value"], distribution="F", center=center,
                    n=int(moments.n.sum()), n_missing=dropped, groups=labels, alpha=alpha,
                    notes=notes, missing="listwise", **sample.attrs())


def sd_one_sample(data: Any, y: str, sd: Any, alpha: float, missing: str) -> TableSet:
    if sd is None:
        raise AnalysisError("invalid_spec", "Give by or a hypothesized standard deviation sd.")
    sd = c.check_number(sd, "sd")
    if sd <= 0:
        raise AnalysisError("invalid_option", "sd must be positive.")
    sample = Replay(data, [y], [y], missing)
    state = numeric_moments(sample, [y])
    n, mean, dropped = state.n, float(state.location[0]), sample.dropped
    if n < 2:
        raise AnalysisError("insufficient_observations", "A variance test needs at least two observations.")
    variance = float(state.sscp[0,0]) / (n-1)
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
        missing="listwise", **sample.attrs())
