"""Factorial analysis of variance and covariance: ``oe.anova`` (SPSS UNIANOVA, Stata anova)."""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from openecon.econometrics.core import TableSet
from openecon.econometrics.stats import common as c
from openecon.econometrics.stats.glm import (
    INTERCEPT, LinearModel, Term, build_terms, check_sequential_order,
)
from openecon.econometrics.stats.posthoc import sidak
from openecon.engines import distributions as dist

ANOVA_COLUMNS = ["ss", "df", "ms", "statistic", "p_value", "partial_eta_squared"]


def prepare(data: Any, outcomes: list[str], factors: list[str], covariates: list[str],
            interactions: Any, slopes: bool, missing: str) -> tuple[LinearModel, int, Tensor]:
    """Listwise sample, factor codes and the sum-to-zero model of ``outcomes``."""
    if not factors and not covariates:
        raise AnalysisError("invalid_spec", "Name at least one factor or covariate.")
    frame, dropped = c.select(data, [*outcomes, *factors, *covariates],
                              numeric=[*outcomes, *covariates], missing=missing)
    codes, levels = {}, {}
    for name in factors:
        codes[name], levels[name] = c.group_codes(frame, name)
    numeric = {name: c.values(frame, name) for name in covariates}
    y = c.matrix(frame, outcomes)
    terms = build_terms(factors, covariates, interactions, slopes)
    shift = y.mean(0)
    return LinearModel(terms, codes, levels, numeric, y - shift, shift), dropped, y


def anova_rows(model: LinearModel, column: int, ss_type: int, y: Tensor | None, *,
               totals: tuple[float, float] | None = None) -> tuple[list, dict]:
    """Rows of the 'Tests of Between-Subjects Effects' table for outcome ``column``."""
    sse = float(model.error[column, column])
    df_error = model.df_error
    if totals is None:
        values = y[:, column]
        total = float(values.square().sum())
        values = values - model.shift[column]
        corrected_total = float((values - values.mean()).square().sum())
    else:
        total, corrected_total = totals
    if c.negligible(sse, corrected_total):
        raise AnalysisError("perfect_fit", "The model fits the outcome exactly, so the error "
                            "mean square is zero and F tests are undefined.")
    mse = sse / df_error

    def row(ss: float, df: int) -> list[Any]:
        f = ss / df / mse
        return [ss, df, ss / df, f, c.f_upper(f, df, df_error), ss / (ss + sse)]

    rows, labels = [], []
    if model.k > 1:
        rows.append(row(max(corrected_total - sse, 0.0), model.k - 1))
        labels.append("corrected_model")
    for term in model.terms:
        rows.append(row(max(float(model.hypothesis(term, ss_type)[column, column]), 0.0),
                        model.df(term)))
        labels.append(term.name)
    rows += [[sse, df_error, mse, None, None, None], [total, model.n, None, None, None, None],
             [corrected_total, model.n - 1, None, None, None, None]]
    labels += ["error", "total", "corrected_total"]
    r2 = 1.0 - sse / corrected_total if corrected_total > 0 else None
    summary = {"r_squared": r2, "rmse": math.sqrt(mse), "df_model": model.k - 1,
               "df_resid": df_error,
               "adjusted_r_squared": None if r2 is None else
               1.0 - mse / (corrected_total / (model.n - 1))}
    return [rows, labels], summary


def _adjusted(p: float, m: int, adjust: str) -> float:
    if adjust == "bonferroni":
        return min(1.0, m * p)
    return sidak(p, m) if adjust == "sidak" else p


def _level(alpha: float, m: int, adjust: str) -> float:
    if adjust == "bonferroni":
        return alpha / m
    return -math.expm1(math.log1p(-alpha) / m) if adjust == "sidak" else alpha


def marginal_means(model: LinearModel, entries: list[Any], factors: list[str], alpha: float,
                   adjust: str) -> dict[str, Any]:
    """Estimated marginal means (covariates at their means) and pairwise comparisons."""
    tables: dict[str, Any] = {}
    means = {name: float(values.mean()) for name, values in model.covariates.items()}
    mse = float(model.error[0, 0]) / model.df_error
    beta = model.beta[:, 0]
    crit = c.t_critical(alpha, model.df_error)
    for entry in entries:
        names = [entry] if isinstance(entry, str) else list(entry) \
            if isinstance(entry, (list, tuple)) else None
        if not names or any(name not in factors for name in names) \
                or len(set(names)) != len(names):
            raise AnalysisError("invalid_spec", f"emmeans entry {entry!r} must be a factor of "
                                "the model or a list of its factors.")
        count = math.prod(len(model.levels[name]) for name in names)
        pairs = count * (count - 1) // 2 if len(names) == 1 else 0
        plan_workspace("ANOVA estimated marginal means", {
            "fitted_model_geometry": 256 * (model.k + 1)**2,
            "marginal_design_vectors": 24 * count * model.k,
            "marginal_covariance_and_intervals": 16 * count * count + 128 * count,
            "pairwise_numeric_buffers_and_result": 312 * pairs,
            "already_retained_result_tables": sum(int(frame.memory_usage(index=True, deep=True).sum())
                                                    for frame in tables.values())})
        cells = list(itertools.product(*(range(len(model.levels[name])) for name in names)))
        vectors = torch.stack([model.marginal_vector(dict(zip(names, cell, strict=True)), means)
                               for cell in cells])
        estimate = vectors @ beta + model.shift[0]
        covariance = mse * (vectors @ model.full.xtx_inv @ vectors.T)
        se = covariance.diagonal().clamp_min(0.0).sqrt()
        rows = [[*(model.levels[name][i] for name, i in zip(names, cell, strict=True)), est, s,
                 model.df_error, est - crit * s, est + crit * s]
                for cell, est, s in zip(cells, estimate.tolist(), se.tolist(), strict=True)]
        key = "#".join(names)
        tables[f"emmeans_{key}"] = c.frame(
            rows, columns=[*names, "mean", "std_error", "df", "ci_low", "ci_high"])
        if len(names) > 1:
            continue
        first, second = torch.triu_indices(len(cells), len(cells), offset=1)
        m = first.shape[0]
        difference = (estimate[first] - estimate[second]).tolist()
        spread = (covariance.diagonal()[first] + covariance.diagonal()[second]
                  - 2.0 * covariance[first, second]).clamp_min(0.0).sqrt().tolist()
        bound = dist.t_isf(0.5 * _level(alpha, m, adjust), model.df_error)
        labels = model.levels[names[0]]
        pairs = []
        for a, b, d, s in zip(first.tolist(), second.tolist(), difference, spread, strict=True):
            t = d / s
            p = _adjusted(c.t_two_sided(t, model.df_error), m, adjust)
            pairs.append([labels[a], labels[b], d, s, t, model.df_error, p, d - bound * s,
                          d + bound * s])
        tables[f"pairwise_{key}"] = c.frame(
            pairs, columns=["level_i", "level_j", "mean_difference", "std_error", "statistic",
                            "df", "p_value", "ci_low", "ci_high"], adjustment=adjust)
    return tables


def anova(data: Any, y: str, factors: list[str] | None = None, *,
          covariates: list[str] | None = None, interactions: Any = "full", ss_type: int = 3,
          within: list[str] | None = None, subject: str | None = None,
          emmeans: list[Any] | None = None, adjust: str = "bonferroni",
          homogeneity_of_slopes: bool = False, alpha: float = 0.05,
          missing: str = "drop") -> TableSet:
    """Factorial analysis of variance / covariance through the general linear model.

    Model: y = mu + (main effects of the factors) + (interactions) + b'covariates + e,
    estimated by least squares on a sum-to-zero (effect-coded) design that is built
    internally: a factor with L levels has L - 1 columns, interactions are products
    of factor columns, covariates enter linearly.

    Sums of squares (``ss_type``):

    * 1 - sequential: each term adjusted for the terms before it, in the order
      covariates, main effects (as listed in ``factors``), interactions (lowest
      order first, or as listed; an interaction must follow the effects it contains);
    * 2 - each term adjusted for every term that does not contain it;
    * 3 - each term adjusted for all other terms (the SPSS and SAS default; equal to
      Stata's partial sums of squares). Valid for unbalanced data as long as every
      cell of each interaction is observed.

    Every F is MS_term / MS_error with (df_term, N - p) degrees of freedom, p the
    number of parameters; partial eta squared = SS_term / (SS_term + SS_error).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome.
    factors : categorical columns (any scalar values; levels are sorted).
    covariates : numeric columns entering linearly (ANCOVA).
    interactions : ``"full"`` (all interactions among the factors, default),
        ``"none"`` (main effects only) or a list such as ``[["a", "b"]]`` or
        ``["a#b"]``; a covariate may appear in an interaction. An interaction needs
        the effects it contains in the model (``a#b#c`` needs ``a#b``, ``a#c`` and
        ``b#c``; ``a#b#x`` needs ``a#x`` and ``b#x``), otherwise ``invalid_spec``.
    ss_type : 1, 2 or 3.
    within, subject : repeated-measures designs; the call is passed on to
        ``oe.rm_anova`` with ``factors`` as between-subject factors (Type III, all
        between-subject interactions; covariates, ``emmeans`` and other values of
        ``ss_type`` or ``interactions`` are rejected).
    emmeans : factors (or lists of factors) for which estimated marginal means are
        reported: predictions averaged with equal weights over the levels of the
        other factors, with covariates at their sample means. Single factors also
        get pairwise comparisons.
    adjust : ``"bonferroni"`` (default), ``"sidak"`` or ``"lsd"`` (no adjustment,
        SPSS's default) for the pairwise comparisons and their intervals.
    homogeneity_of_slopes : add every factor-by-covariate interaction, which
        tests the ANCOVA assumption of equal slopes.
    alpha : 1 - confidence level.
    missing : ``"drop"`` (listwise deletion, counted in ``attrs["n_missing"]``) or
        ``"raise"``.

    Returns
    -------
    TableSet with ``anova`` (rows ``corrected_model``, ``Intercept``, one per term
    named ``a``, ``a#b``, ..., ``error``, ``total``, ``corrected_total``; columns ss, df,
    ms, statistic (F), p_value, partial_eta_squared), and ``emmeans_<factor>`` /
    ``pairwise_<factor>`` when requested. ``attrs``: ``r_squared``,
    ``adjusted_r_squared``, ``rmse``, ``df_model``, ``df_resid``, ``n``, ``n_missing``,
    ``ss_type``, ``statistic`` and ``p_value`` of the corrected model.

    Errors: ``empty_cells`` when a cell of an interaction is unobserved (Type IV
    sums of squares are not implemented), ``collinear_design`` for confounded
    effects, ``no_residual_df`` for saturated models.

    Equivalent commands: SPSS ``UNIANOVA y BY a b WITH x /METHOD=SSTYPE(3)
    /EMMEANS=TABLES(a) COMPARE ADJ(BONFERRONI)``; Stata ``anova y a##b c.x``
    (partial SS; ``, sequential`` for Type I), ``margins a, asbalanced``,
    ``pwcompare a, mcompare(bonferroni)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [3.1, 4.0, 5.2, 6.1, 4.4, 5.1, 7.9, 8.4, 3.6, 5.5, 6.0, 8.8],
    ...         "a": ["u", "u", "v", "v", "u", "u", "v", "v", "u", "v", "u", "v"],
    ...         "b": ["p", "p", "p", "p", "q", "q", "q", "q", "p", "p", "q", "q"]}
    >>> result = oe.anova(data, "y", ["a", "b"])
    >>> list(result["anova"].index)[:5]
    ['corrected_model', 'Intercept', 'a', 'b', 'a#b']
    """
    c.check_name(y, "y")
    factors = c.name_list(factors, "factors", minimum=0)
    covariates = c.name_list(covariates, "covariates", minimum=0)
    alpha = c.check_alpha(alpha)
    if within is not None or subject is not None:
        if within is None or subject is None:
            raise AnalysisError("invalid_spec", "A repeated-measures design needs both within "
                                "(the within-subject factors) and subject.")
        if covariates or emmeans or homogeneity_of_slopes:
            raise AnalysisError("invalid_spec", "Covariates and emmeans are not available for "
                                "repeated-measures designs.")
        if ss_type != 3 or interactions != "full":
            raise AnalysisError("invalid_spec", "Repeated-measures designs use Type III sums of "
                                "squares with all interactions of the between-subject factors; "
                                "leave ss_type=3 and interactions='full'.")
        from openecon.econometrics.stats.rm_anova import rm_anova
        return rm_anova(data, y, subject, within, between=factors or None, alpha=alpha,
                        missing=missing)
    if not isinstance(ss_type, int) or isinstance(ss_type, bool) or ss_type not in (1, 2, 3):
        raise AnalysisError("invalid_option", "ss_type must be 1, 2 or 3 (Type IV sums of "
                            "squares are not implemented).")
    c.check_choice(adjust, "adjust", ("bonferroni", "sidak", "lsd"))
    c.check_flag(homogeneity_of_slopes, "homogeneity_of_slopes")
    if emmeans is not None and (isinstance(emmeans, str) or not isinstance(emmeans, (list, tuple))):
        raise AnalysisError("invalid_spec", "emmeans must be a list of factors, for example "
                            "emmeans=['a'].")
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_anova import anova as replay_anova
        return replay_anova(data, y, factors, covariates, interactions=interactions,
                            slopes=homogeneity_of_slopes, ss_type=ss_type, emmeans=emmeans,
                            adjust=adjust, alpha=alpha, missing=missing)
    model, dropped, values = prepare(data, [y], factors, covariates, interactions,
                                     homogeneity_of_slopes, missing)
    if ss_type == 1:
        check_sequential_order(model.terms)
    (rows, labels), summary = anova_rows(model, 0, ss_type, values)
    tables = {"anova": c.frame(rows, columns=ANOVA_COLUMNS, index=labels)}
    if emmeans:
        tables.update(marginal_means(model, list(emmeans), factors, alpha, adjust))
    headline = tables["anova"].iloc[0]
    return TableSet(
        tables, title=f"Analysis of variance of {y} (Type {'I' * ss_type} sums of squares)",
        statistic=headline["statistic"] if model.k > 1 else None,
        p_value=headline["p_value"] if model.k > 1 else None, distribution="F", **summary,
        n=model.n, n_missing=dropped, ss_type=ss_type, alpha=alpha,
        terms=[term.name for term in model.terms],
        levels={name: labels_ for name, labels_ in model.levels.items()},
        coding="sum-to-zero (effect) coding", missing="listwise")


__all__ = ["anova", "prepare", "anova_rows", "marginal_means", "INTERCEPT", "Term"]
