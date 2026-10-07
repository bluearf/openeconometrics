"""Pairwise multiple comparisons of group means after a one-way ANOVA.

Every method starts from the pairwise differences d_ij = mean_i - mean_j (i before j
in ascending order of the group value) and t_ij = d_ij / se_ij:

    lsd            se from the pooled MSE, Student t with N - k df, no adjustment
    bonferroni     p * m, m = k (k - 1) / 2; interval at alpha / m
    sidak          1 - (1 - p)^m; interval at 1 - (1 - alpha)^(1/m)
    holm           step-down Bonferroni (Holm 1979); no simultaneous interval
    scheffe        F = t^2 / (k - 1) against F(k - 1, N - k); interval sqrt((k-1) F_alpha)
    tukey          studentized range, q = |t| sqrt(2), k groups, N - k df
                   (Tukey-Kramer when the group sizes differ)
    games_howell   se_ij = sqrt(s_i^2/n_i + s_j^2/n_j), Welch-Satterthwaite df_ij,
                   studentized range with k groups and df_ij
    dunnett        each group against a control, two-sided, exact many-one t
                   distribution with corr = lambda_i lambda_j

The studentized range and Dunnett distributions are in ``srange.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from openecon.econometrics.core import kernel_call
from openecon.econometrics.stats import common as c
from openecon.econometrics.stats import srange
from openecon.engines import distributions as dist

METHODS = ("tukey", "bonferroni", "sidak", "scheffe", "lsd", "games_howell", "dunnett", "holm")
COLUMNS = ["group_i", "group_j", "mean_difference", "std_error", "statistic", "df", "p_value",
           "ci_low", "ci_high"]


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values: max over earlier ranks of (m - rank + 1) p."""
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted, running = [0.0] * m, 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def sidak(p: float, m: int) -> float:
    return -math.expm1(m * math.log1p(-p)) if p < 1.0 else 1.0


def _frame(labels: list[Any], first: Tensor, second: Tensor, difference: Tensor, se: Tensor,
           df: list[float], p_values: list[float], critical: list[float | None], method: str,
           **attrs: Any) -> Any:
    rows = []
    for a, b, d, s, nu, p, crit in zip(first.tolist(), second.tolist(), difference.tolist(),
                                       se.tolist(), df, p_values, critical, strict=True):
        low, high = (None, None) if crit is None else (d - crit * s, d + crit * s)
        rows.append([labels[a], labels[b], d, s, d / s, nu, p, low, high])
    return c.frame(rows, columns=COLUMNS, method=method, **attrs)


def _critical(quantile: Any, alpha: float, *args: Any) -> Tensor:
    """Upper-alpha critical value of a range-type distribution, with a usable message."""
    try:
        value = kernel_call(quantile, alpha, *args)
    except AnalysisError as exc:
        if exc.code == "dunnett_unbalanced":
            raise
        raise AnalysisError("invalid_option", f"The critical value for alpha={alpha:g} cannot be "
                            "computed (it is not finite in double precision); use a larger "
                            "alpha.") from exc
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("invalid_option", f"The critical value for alpha={alpha:g} is not "
                            "finite in double precision; use a larger alpha.")
    return value


def compare(method: str, labels: list[Any], moments: c.GroupMoments, mse: float, df_error: int,
            alpha: float, control: int) -> Any:
    """The pairwise-comparison table of one method (see the module docstring)."""
    k = len(labels)
    pairs = k - 1 if method == "dunnett" else k * (k - 1) // 2
    plan_workspace("posthoc multiple comparisons", {
        "pair_indices_and_distribution_buffers": 256 * pairs,
        "numeric_result_frame": 56 * pairs})
    n, mean, var = moments.n, moments.mean, moments.var
    if method == "dunnett":
        first = torch.tensor([g for g in range(k) if g != control], dtype=torch.int64)
        second = torch.full_like(first, control)
    else:
        first, second = torch.triu_indices(k, k, offset=1)
    m = first.shape[0]
    difference = mean[first] - mean[second]
    if method == "games_howell":
        if bool((n < 2).any()) or not bool((var > 0).all()):
            raise AnalysisError("zero_variance", "Games-Howell comparisons need at least two "
                                "observations and a positive variance in every group.")
        a, b = var[first] / n[first], var[second] / n[second]
        se = torch.sqrt(a + b)
        df_pair = (a + b).square() / (a.square() / (n[first] - 1) + b.square() / (n[second] - 1))
        q = (difference / se).abs() * math.sqrt(2.0)
        p_values = kernel_call(srange.ptukey_sf, q, k, df_pair).tolist()
        critical = (_critical(srange.qtukey_upper, alpha, k, df_pair) / math.sqrt(2.0)).tolist()
        return _frame(labels, first, second, difference, se, df_pair.tolist(), p_values, critical,
                      method, adjustment="studentized range with Welch degrees of freedom")
    se = torch.sqrt(mse * (1.0 / n[first] + 1.0 / n[second]))
    t_abs = (difference / se).abs()
    t = t_abs.tolist()
    df = [df_error] * m
    if method == "tukey":
        p_values = kernel_call(srange.ptukey_sf, t_abs * math.sqrt(2.0), k,
                               float(df_error)).tolist()
        crit = float(_critical(srange.qtukey_upper, alpha, k, float(df_error))[0])
        return _frame(labels, first, second, difference, se, df, p_values,
                      [crit / math.sqrt(2.0)] * m, method,
                      adjustment="studentized range (Tukey-Kramer for unequal group sizes)")
    if method == "dunnett":
        lambdas = torch.sqrt(n[first] / (n[first] + n[second]))
        try:
            p_values = kernel_call(srange.pdunnett_sf, t_abs, lambdas,
                                   float(df_error)).tolist()
            crit = float(_critical(srange.qdunnett_upper, alpha, lambdas, float(df_error))[0])
        except AnalysisError as exc:
            if exc.code != "dunnett_unbalanced":
                raise
            raise AnalysisError("dunnett_unbalanced", "Dunnett's test is not available when a "
                                "group is several hundred times larger than the control group; "
                                "use posthoc=['bonferroni'] or ['sidak'] instead.") from exc
        return _frame(labels, first, second, difference, se, df, p_values, [crit] * m, method,
                      adjustment="Dunnett two-sided many-one t", control=labels[control])
    if method == "scheffe":
        p_values = [dist.f_sf(value * value / (k - 1), k - 1, df_error) for value in t]
        crit = math.sqrt((k - 1) * dist.f_isf(alpha, k - 1, df_error))
        return _frame(labels, first, second, difference, se, df, p_values, [crit] * m, method,
                      adjustment="Scheffe F with k - 1 numerator degrees of freedom")
    raw = [min(1.0, 2.0 * dist.t_sf(value, df_error)) for value in t]
    if method == "lsd":
        p_values, level, note = raw, alpha, "none (Fisher's least significant difference)"
    elif method == "bonferroni":
        p_values, level = [min(1.0, m * p) for p in raw], alpha / m
        note = f"Bonferroni, {m} comparisons"
    elif method == "sidak":
        p_values, level = [sidak(p, m) for p in raw], -math.expm1(math.log1p(-alpha) / m)
        note = f"Sidak, {m} comparisons"
    else:
        p_values, level, note = holm_adjust(raw), None, f"Holm step-down, {m} comparisons"
    crit = None if level is None else dist.t_isf(0.5 * level, df_error)
    return _frame(labels, first, second, difference, se, df, p_values, [crit] * m, method,
                  adjustment=note)
