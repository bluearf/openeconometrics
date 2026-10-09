"""Eight prospective design laws. See docs/econometrics/power-designs.md.

These accept prespecified design assumptions, never samples or fitted models.
The t/F laws assume normal independent errors; Pearson and ICC design-effect
approximations are explicitly labelled. No observed power is computed.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import check_number, procedure
from openecon.engines.distributions import chi2_isf, f_isf
from .planning import _count, _normal_power, _number, _options, _result, _root, _sizes, _solve
from .power_distributions import MAX_NONCENTRALITY, chi_power, f_power, t_power

MAX_N = 20000
MAX_CELLS = 100


def _scenarios(row: Callable[[int], dict[str, Any]], n: int, minimum: int,
               maximum: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, omitted = [], []
    for size in _sizes(n, minimum, maximum):
        try:
            rows.append(row(size))
        except AnalysisError as error:
            if error.code != "resource_limit" or size == n:
                raise
            omitted.append(dict(n=size, code=error.code, reason=str(error)))
    return rows, omitted


def _search(function: Callable[[int], float], target: float, minimum: int, maximum: int) -> int:
    """Exponential bracket avoids evaluating huge noncentrality needlessly.

All laws in this module are monotone in integer n on the declared design path.
The returned size and its immediate predecessor are both checked.
"""
    low = high = minimum
    while function(high) < target:
        if high == maximum:
            raise AnalysisError("unattainable_design", "Target power is unattainable within max_n.")
        low, high = high + 1, min(maximum, max(high + 1, high * 2))
    while low < high:
        middle = (low + high) // 2
        if function(middle) >= target:
            high = middle
        else:
            low = middle + 1
    if function(low) < target or (low > minimum and function(low - 1) >= target):
        raise AnalysisError("numerical_failure", "Minimum integer sample-size gate failed.")
    return low


def _ratio(value: Any) -> float:
    value = check_number(value, "ratio")
    if not .05 <= value <= 20:
        raise AnalysisError("invalid_option", "ratio must be in [0.05,20].")
    return value


def _allocation(max_n: Any, ratio: float) -> tuple[int, int]:
    maximum = _count(max_n, "max_n", 2, MAX_N)
    minimum = max(2, math.floor(1 / ratio) + 1)
    limit = min(maximum, math.floor(maximum / ratio))
    while limit >= minimum and math.ceil(ratio * limit) > maximum:
        limit -= 1
    if limit < minimum:
        raise AnalysisError("resource_limit", "Both groups require at least two units within max_n.")
    return minimum, limit


def _direction(effect: float, alternative: str, sign: int) -> None:
    if effect == 0 or (alternative != "two-sided" and effect * sign <= 0):
        raise AnalysisError("unattainable_design", "Sample-size planning needs a nonzero effect "
                            "in the prespecified direction.")


def _t_design(method: str, effect: Any, n: Any, power: Any, sd: float, ratio: float | None,
              alpha: Any, alternative: Any, direction: Any, max_n: Any,
              **design: Any) -> TableSet:
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(effect, n, power, alpha)
    target = float(power) if power is not None else None
    if ratio is None:
        minimum, maximum = 2, _count(max_n, "max_n", 2, MAX_N)
    else:
        minimum, maximum = _allocation(max_n, ratio)
    n = _count(n, "n", minimum, maximum) if n is not None else None
    effect = _number(effect, "effect") if effect is not None else None
    def geometry(size: int) -> tuple[int | None, int, float]:
        n2 = math.ceil(ratio * size) if ratio is not None else None
        df = size + n2 - 2 if n2 is not None else size - 1
        se = sd * math.sqrt(1 / size + (1 / n2 if n2 is not None else 0))
        return n2, df, se
    def probability(size: int, difference: float) -> float:
        _, df, se = geometry(size)
        return t_power(difference / se, df, alpha, alternative)
    if solve == "effect":
        _, df, se = geometry(n)
        delta = _root(lambda shift: t_power(sign * shift, df, alpha, alternative), target, 64.)
        effect = _number(sign * delta * se, "detectable effect")
    if solve == "n":
        _direction(effect, alternative, sign)
        n = _search(lambda size: probability(size, effect), target, minimum, maximum)
    def row(size: int) -> dict[str, Any]:
        n2, df, se = geometry(size)
        return dict(n=size, n2=n2, total_n=size + (n2 or 0), df=df, effect=effect,
                    design_standard_error=se, noncentrality=effect / se,
                    power=probability(size, effect))
    scenarios, omitted = _scenarios(row, n, minimum, maximum)
    return _result(method, dict(solve_for=solve, **row(n), alpha=alpha,
                               alternative=alternative, target_power=target),
                   scenarios, omitted_adjacent_scenarios=omitted,
                   inference="exact noncentral-t rejection probability under declared normal design",
                   sd=sd, ratio=ratio, alpha=alpha, alternative=alternative,
                   direction="upper" if sign > 0 else "lower", max_n_per_group=int(max_n),
                   probability_budget="|delta|<=64, df<=40000; 16/32-node quadrature error gate 2e-11",
                   sd_role="prespecified population SD for design; sample SD estimated at analysis",
                   target_power=target, **design)


@procedure
def power_tmean(effect: float | None = None, *, sd: float, n: int | None = None,
                power: float | None = None, alpha: float = .05,
                alternative: str = "two-sided", direction: str | None = None,
                max_n: int = MAX_N) -> TableSet:
    """Normal one-sample Student-t power, minimum integer n, or signed MDE.

Exactly two of effect/n/power are required. Effect is mean minus null mean.
SD is prespecified population SD, estimated with sample SD at analysis.
"""
    return _t_design("power_tmean", effect, n, power, _number(sd, "sd", positive=True),
                     None, alpha, alternative, direction, max_n,
                     sampling="iid normal observations", effect_definition="population mean-null")


@procedure
def power_ttwomeans(effect: float | None = None, *, sd: float, n: int | None = None,
                    power: float | None = None, ratio: float = 1., alpha: float = .05,
                    alternative: str = "two-sided", direction: str | None = None,
                    max_n: int = MAX_N) -> TableSet:
    """Pooled equal-variance independent normal t design, n2=ceil(ratio*n).

Exactly two of effect/n/power; n is group-1 size, effect is mean2-mean1
minus null difference. Unequal variances/Welch are outside this contract.
"""
    return _t_design("power_ttwomeans", effect, n, power, _number(sd, "sd", positive=True),
                     _ratio(ratio), alpha, alternative, direction, max_n,
                     sampling="independent normal equal-variance groups; pooled analysis SD",
                     allocation="n2=ceil(ratio*n); at least 2 per group",
                     effect_definition="population mean2-mean1 minus null difference")


@procedure
def power_tpaired(effect: float | None = None, *, sd1: float, sd2: float, correlation: float,
                  n: int | None = None, power: float | None = None, alpha: float = .05,
                  alternative: str = "two-sided", direction: str | None = None,
                  max_n: int = MAX_N) -> TableSet:
    """Normal paired t design; n counts complete independent pairs.

Difference SD derives from prespecified marginal SDs and pair correlation.
Exactly two of effect/n/power; effect is mean(after-before) minus null.
"""
    sd1, sd2 = _number(sd1, "sd1", positive=True), _number(sd2, "sd2", positive=True)
    correlation = check_number(correlation, "correlation")
    if not -1 <= correlation <= 1:
        raise AnalysisError("invalid_option", "correlation must be in [-1,1].")
    sd = math.hypot(sd1 - sd2, math.sqrt(sd1) * math.sqrt(sd2) * math.sqrt(2 * (1 - correlation)))
    sd = _number(sd, "difference SD", positive=True)
    return _t_design("power_tpaired", effect, n, power, sd, None, alpha, alternative,
                     direction, max_n, sd1=sd1, sd2=sd2, correlation=correlation,
                     sampling="independent normal paired differences; n complete pairs",
                     effect_definition="population after-before difference minus null")


@lru_cache(maxsize=1024)
def _f_cut(alpha: float, df1: int, df2: int) -> float:
    with torch.device("cpu"):
        return f_isf(alpha, df1, df2)


def _f_design(method: str, effect: Any, n: Any, power: Any, alpha: Any, max_n: Any,
              minimum: int, geometry: Callable[[int], tuple[int, int, float]],
              squared: bool, **design: Any) -> TableSet:
    alpha, _, _ = _options(alpha, "upper", None)
    solve = _solve(effect, n, power, alpha)
    maximum = _count(max_n, "max_n", minimum, MAX_N)
    effect = _number(effect, "effect") if effect is not None else None
    if effect is not None and effect < 0:
        raise AnalysisError("invalid_option", "The standardized effect must be nonnegative.")
    target = float(power) if power is not None else None
    n = _count(n, "n", minimum, maximum) if n is not None else None
    def probability(size: int, value: float) -> float:
        df1, df2, scale = geometry(size)
        return f_power(_f_cut(alpha, df1, df2), df1, df2, scale * (value**2 if squared else value))
    if solve == "effect":
        _, _, scale = geometry(n)
        bound = MAX_NONCENTRALITY / scale
        upper = math.nextafter(math.sqrt(bound) if squared else bound, 0.)
        effect = _root(lambda value: probability(n, value), target, upper)
    if solve == "n":
        if effect == 0:
            raise AnalysisError("unattainable_design", "A nonzero standardized effect is required.")
        n = _search(lambda size: probability(size, effect), target, minimum, maximum)
    def row(size: int) -> dict[str, Any]:
        df1, df2, scale = geometry(size)
        nc = scale * (effect**2 if squared else effect)
        return dict(n=size, effect=effect, df1=df1, df2=df2, noncentrality=nc,
                    critical_value=_f_cut(alpha, df1, df2), power=probability(size, effect))
    scenarios, omitted = _scenarios(row, n, minimum, maximum)
    return _result(method, dict(solve_for=solve, **row(n), alpha=alpha, target_power=target),
                   scenarios, omitted_adjacent_scenarios=omitted,
                   inference="exact noncentral-F rejection probability under fixed Gaussian design",
                   alpha=alpha, target_power=target, max_n=int(max_n),
                   probability_budget="lambda<=4096; Poisson omitted mass<=2e-14; df<=40000",
                   **design)


@procedure
def power_anova(effect: float | None = None, *, groups: int, n: int | None = None,
                power: float | None = None, alpha: float = .05, max_n: int = MAX_N) -> TableSet:
    """Balanced fixed one-way normal ANOVA; n is size per group, effect is Cohen f.

f=sqrt(mean((mu_j-grand_mu)^2))/sigma, lambda=groups*n*f^2.
Exactly two of effect/n/power. Independent errors and common variance required.
"""
    groups = _count(groups, "groups", 2, 30)
    maximum = min(_count(max_n, "max_n", 2, MAX_N), MAX_N // groups)
    result = _f_design("power_anova", effect, n, power, alpha, maximum, 2,
                       lambda size: (groups - 1, groups * (size - 1), groups * size), True,
                       groups=groups, allocation="balanced, n per group; total_n=groups*n",
                       effect_definition="Cohen f; fixed population group means/common residual SD",
                       sampling="independent normal errors, equal variance; fixed group means",
                       max_total_n=MAX_N)
    for frame in (result["plan"], result["scenarios"]):
        frame["total_n"] = frame["n"] * groups
    return result


@procedure
def power_regression(effect: float | None = None, *, predictors: int, tested: int,
                     n: int | None = None, power: float | None = None, alpha: float = .05,
                     max_n: int = MAX_N) -> TableSet:
    """Fixed-design Gaussian partial F test; effect=f2=lambda/n, not observed R2.

predictors counts all full-model slopes excluding intercept; tested counts q
restrictions. Design must be full rank with signal scaling lambda=n*f2.
Exactly two of effect/n/power. Random-X and robust/cluster errors unsupported.
"""
    predictors = _count(predictors, "predictors", 1, 100)
    tested = _count(tested, "tested", 1, predictors)
    return _f_design("power_regression", effect, n, power, alpha, max_n, predictors + 2,
                     lambda size: (tested, size - predictors - 1, size), False,
                     predictors=predictors, tested=tested,
                     effect_definition="prespecified f2=normalized conditional signal/noise, lambda=n*f2",
                     sampling="full-rank fixed design/intercept, independent homoskedastic normal errors",
                     allocation="n total observations; fixed design scaling must hold across n")


@procedure
def power_cluster_mean(effect: float | None = None, *, sd: float, cluster_size: int, icc: float,
                       n: int | None = None, power: float | None = None, ratio: float = 1.,
                       alpha: float = .05, alternative: str = "two-sided",
                       direction: str | None = None, max_n: int = MAX_N) -> TableSet:
    """ICC design-effect normal two-arm planning; n counts clusters in arm 1.

Equal cluster sizes, common marginal SD/ICC, independent clusters and known
covariance Gaussian cluster means (or explicit asymptotic normal approximation).
Exactly two of effect/n/power. No estimated-ICC or small-cluster t correction.
"""
    sd = _number(sd, "sd", positive=True)
    cluster_size = _count(cluster_size, "cluster_size", 1, 10000)
    icc = check_number(icc, "icc")
    if not 0 <= icc <= 1:
        raise AnalysisError("invalid_option", "icc must be in [0,1].")
    ratio = _ratio(ratio)
    minimum, maximum = _allocation(max_n, ratio)
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(effect, n, power, alpha)
    target = float(power) if power is not None else None
    effect = _number(effect, "effect") if effect is not None else None
    n = _count(n, "n", minimum, maximum) if n is not None else None
    deff = 1 + (cluster_size - 1) * icc
    def se(size: int) -> float:
        return sd * math.sqrt(deff / cluster_size * (1 / size + 1 / math.ceil(ratio * size)))
    if solve == "effect":
        shift = _root(lambda value: _normal_power(sign * value, alpha, alternative), target, 64.)
        effect = _number(sign * shift * se(n), "detectable effect")
    if solve == "n":
        _direction(effect, alternative, sign)
        n = _search(lambda size: _normal_power(effect / se(size), alpha, alternative),
                    target, minimum, maximum)
    def row(size: int) -> dict[str, Any]:
        n2 = math.ceil(ratio * size)
        return dict(n=size, n2=n2, total_clusters=size + n2,
                    total_n=(size + n2) * cluster_size, effect=effect,
                    design_effect=deff, design_standard_error=se(size),
                    power=_normal_power(effect / se(size), alpha, alternative))
    return _result("power_cluster_mean", dict(solve_for=solve, **row(n), alpha=alpha,
                                              alternative=alternative, target_power=target),
                   [row(size) for size in _sizes(n, minimum, maximum)],
                   inference="normal ICC design-effect approximation; exact only for known-covariance Gaussian cluster means",
                   cluster_size=cluster_size, icc=icc, sd=sd, ratio=ratio,
                   alpha=alpha, alternative=alternative, target_power=target,
                   direction="upper" if sign > 0 else "lower", max_clusters_per_arm=int(max_n),
                   allocation="n2=ceil(ratio*n) clusters; common equal size in both arms",
                   effect_definition="population mean2-mean1 minus null difference",
                   exclusions="unequal cluster sizes, estimated ICC, small-cluster t, stepped-wedge or multi-stage designs")


def _probabilities(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= MAX_CELLS:
        raise AnalysisError("invalid_option", f"{name} must be a list/tuple of 2..100 probabilities.")
    output = [check_number(item, name) for item in value]
    if any(not 0 <= item <= 1 for item in output) or abs(math.fsum(output) - 1) > 1e-12:
        raise AnalysisError("invalid_option", f"{name} must be nonnegative and sum to one; no normalization.")
    return output


def _pearson(method: str, null: list[float], alternative: list[float], df: int,
             strength: Any, n: Any, power: Any, alpha: Any, max_n: Any,
             **design: Any) -> TableSet:
    if len(null) != len(alternative) or any(value <= 0 for value in null):
        raise AnalysisError("invalid_option", "All null probabilities must be positive, with matching cells.")
    if min(null) < 5 / MAX_N:
        raise AnalysisError("resource_limit", "The smallest null cell cannot reach expected count 5 within max_n.")
    minimum = max(2, math.ceil(5 / min(null)))
    # Floating point rounding at the integer count boundary is checked directly.
    while minimum * min(null) < 5:
        minimum += 1
    maximum = _count(max_n, "max_n", minimum, MAX_N)
    alpha, _, _ = _options(alpha, "upper", None)
    solve = _solve(strength, n, power, alpha)
    target = float(power) if power is not None else None
    strength = check_number(strength, "strength") if strength is not None else None
    if strength is not None and not 0 <= strength <= 1:
        raise AnalysisError("invalid_option", "strength must lie in [0,1]; extrapolation is unsupported.")
    n = _count(n, "n", minimum, maximum) if n is not None else None
    w2 = math.fsum((a - b)**2 / b for a, b in zip(alternative, null, strict=True))
    with torch.device("cpu"):
        cut = chi2_isf(alpha, df)
    def probability(size: int, scale: float) -> float:
        return chi_power(cut, df, size * w2 * scale**2)
    if solve == "effect":
        bound = min(1., math.sqrt(MAX_NONCENTRALITY / (n * w2))) if w2 else 1.
        strength = _root(lambda value: probability(n, value), target, math.nextafter(bound, 0.))
    if solve == "n":
        if w2 == 0 or strength == 0:
            raise AnalysisError("unattainable_design", "The proposed probabilities equal the null.")
        n = _search(lambda size: probability(size, strength), target, minimum, maximum)
    def row(size: int) -> dict[str, Any]:
        return dict(n=size, strength=strength, w=math.sqrt(w2) * strength, df=df,
                    noncentrality=size * w2 * strength**2, power=probability(size, strength),
                    min_expected_null_count=size * min(null))
    probabilities = [b + strength * (a - b) for a, b in zip(alternative, null, strict=True)]
    scenarios, omitted = _scenarios(row, n, minimum, maximum)
    result = _result(method, dict(solve_for="strength" if solve == "effect" else solve,
                                 **row(n), alpha=alpha, target_power=target),
                     scenarios, omitted_adjacent_scenarios=omitted,
                     inference="asymptotic noncentral chi-square Pearson design approximation; not finite-sample exact",
                     null_probabilities=null, proposed_probabilities=alternative,
                     planned_probabilities=probabilities, alpha=alpha, target_power=target,
                     strength_definition="convex path null+strength*(proposed-null), 0<=strength<=1",
                     expected_cell_gate="every null expected cell >=5; still an asymptotic approximation",
                     max_n=int(max_n), probability_budget="lambda<=4096; omitted Poisson mass<=2e-14",
                     **design)
    result["cells"] = table([dict(cell=j, null_probability=b, proposed_probability=a,
                                  planned_probability=p, expected_null=n * b)
                             for j, (b, a, p) in enumerate(zip(null, alternative, probabilities, strict=True))])
    return result


@procedure
def power_gof(null_probabilities: list[float], proposed_probabilities: list[float], *,
              strength: float | None = 1., n: int | None = None, power: float | None = None,
              alpha: float = .05, max_n: int = MAX_N) -> TableSet:
    """Multinomial fixed-null Pearson goodness-of-fit planning, df=cells-1.

Specify exactly two of strength/n/power; strength=None solves the smallest
detectable convex departure toward proposed probabilities at fixed n.
Null probabilities are fixed, not estimated; category merging is unsupported.
"""
    null = _probabilities(null_probabilities, "null_probabilities")
    proposed = _probabilities(proposed_probabilities, "proposed_probabilities")
    return _pearson("power_gof", null, proposed, len(null) - 1, strength, n, power,
                    alpha, max_n, sampling="iid multinomial with prespecified fixed categories/null",
                    fitted_null_parameters=0)


@procedure
def power_independence(joint_probabilities: list[list[float]], *, strength: float | None = 1.,
                       n: int | None = None, power: float | None = None, alpha: float = .05,
                       max_n: int = MAX_N) -> TableSet:
    """Pearson r-by-c independence design; convex strength preserves proposed margins.

Positive prespecified joint probabilities must sum to one; df=(r-1)*(c-1).
Specify exactly two of strength/n/power; n is total iid multinomial observations.
No structural zeros, weights, matched samples or continuity correction.
"""
    if not isinstance(joint_probabilities, (list, tuple)) or not 2 <= len(joint_probabilities) <= 50:
        raise AnalysisError("invalid_option", "joint_probabilities must have 2..50 rows.")
    rows = len(joint_probabilities)
    if any(not isinstance(row, (list, tuple)) for row in joint_probabilities):
        raise AnalysisError("invalid_option", "Each joint probability row must be a list/tuple.")
    columns = len(joint_probabilities[0])
    if columns < 2 or any(len(row) != columns for row in joint_probabilities) or rows * columns > MAX_CELLS:
        raise AnalysisError("invalid_option", "The joint matrix must be rectangular, 2+ columns, <=100 cells.")
    flat = _probabilities([item for row in joint_probabilities for item in row], "joint_probabilities")
    if any(item <= 0 for item in flat):
        raise AnalysisError("invalid_option", "Joint probabilities must be positive; structural zeros unsupported.")
    marg_rows = [math.fsum(flat[r * columns:(r + 1) * columns]) for r in range(rows)]
    marg_cols = [math.fsum(flat[r * columns + c] for r in range(rows)) for c in range(columns)]
    null = [a * b for a in marg_rows for b in marg_cols]
    return _pearson("power_independence", null, flat, (rows - 1) * (columns - 1),
                    strength, n, power, alpha, max_n, rows=rows, columns=columns,
                    row_marginals=marg_rows, column_marginals=marg_cols,
                    sampling="iid multinomial cells; estimated margins under independence null",
                    exclusions="structural zeros, weights, matching, continuity correction")
