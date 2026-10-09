"""Prospective design calculations; inputs are assumptions, never fitted data.

Normal means/CI widths are exact only for independent normal observations with
known population SDs. Correlation uses the explicit Fisher-z approximation.
Binomial rejection regions are nonrandomized equal-tail regions, summed on
native CPU float64 tensors. See docs/econometrics/planning.md for contracts.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from numbers import Integral
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import (
    check_choice, check_number, check_probability, procedure,
)
from openecon.engines.distributions import normal_cdf, normal_isf

MAX_N = 10_000_000
MAX_BINOMIAL_N = 1000
RHO_LIMIT = 1.0 - 1e-12


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    result = check_number(value, name)
    if abs(result) > 1e100 or (result != 0 and abs(result) < 1e-100) \
            or (positive and result <= 0):
        raise AnalysisError("invalid_option", f"{name} must have magnitude in [1e-100, 1e100]"
                            + (" and be positive." if positive else " (or be zero)."))
    return result


def _count(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise AnalysisError("invalid_option", f"{name} must be an integer >= {minimum}.")
    if value > maximum:
        raise AnalysisError("resource_limit", f"{name} exceeds the limit of {maximum}.")
    return int(value)


def _options(alpha: Any, alternative: Any, direction: Any) -> tuple[float, str, int]:
    alpha = check_probability(alpha, "alpha")
    if not 1e-8 <= alpha < .5:
        raise AnalysisError("invalid_option", "alpha must be in [1e-8, 0.5).")
    alternative = check_choice(alternative, "alternative", ("two-sided", "upper", "lower"))
    if direction is None:
        direction = "lower" if alternative == "lower" else "upper"
    direction = check_choice(direction, "direction", ("upper", "lower"))
    if alternative != "two-sided" and direction != alternative:
        raise AnalysisError("invalid_option", "direction must match the one-sided alternative.")
    return alpha, alternative, 1 if direction == "upper" else -1


def _solve(effect: Any, n: Any, power: Any, alpha: float) -> str:
    if sum(value is None for value in (effect, n, power)) != 1:
        raise AnalysisError("invalid_option", "Specify exactly two of effect/alternative, n, power.")
    if power is not None:
        target = check_probability(power, "power")
        if not alpha < target <= 1 - 1e-12:
            raise AnalysisError("invalid_option", "Target power must exceed alpha and be <= 1-1e-12.")
    return "effect" if effect is None else "n" if n is None else "power"


def _normal_power(shift: float, alpha: float, alternative: str) -> float:
    z = normal_isf(alpha / 2 if alternative == "two-sided" else alpha)
    if alternative == "two-sided":
        return min(1., normal_cdf(shift - z) + normal_cdf(-shift - z))
    return normal_cdf((shift if alternative == "upper" else -shift) - z)


def _minimum_n(function: Callable[[int], float], target: float, minimum: int,
               maximum: int, *, decreasing: bool = False) -> int:
    """Monotone integer search, followed by both boundary checks; <= 50 calls."""
    qualifies = (lambda value: value <= target) if decreasing else (lambda value: value >= target)
    if not qualifies(function(maximum)):
        raise AnalysisError("resource_limit", "Target is unattainable within max_n; increase the "
                            "budget or revise the prespecified design.")
    low, high = minimum, maximum
    while low < high:
        middle = (low + high) // 2
        if qualifies(function(middle)):
            high = middle
        else:
            low = middle + 1
    if not qualifies(function(low)) or (low > minimum and qualifies(function(low - 1))):
        raise AnalysisError("numerical_failure", "Integer sample-size boundary verification failed.")
    return low


def _root(function: Callable[[float], float], target: float, maximum: float) -> float:
    """Unique upper crossing above null size, maintaining the feasible upper bound."""
    if maximum <= 0 or function(maximum) < target:
        raise AnalysisError("unattainable_design", "Target power is unattainable in the supported "
                            "effect range at this sample size.")
    low, high = 0., maximum
    for _ in range(64):
        middle = low + (high - low) / 2
        if middle == low or middle == high:
            break
        if function(middle) >= target:
            high = middle
        else:
            low = middle
    if abs(function(high) - target) > 1e-9:
        raise AnalysisError("numerical_failure", "Detectable-effect inversion did not converge.")
    return high


def _design(sd: Any, sd2: Any, ratio: Any, max_n: Any) -> tuple[float, float | None, float, int]:
    sd = _number(sd, "sd", positive=True)
    sd2 = _number(sd2, "sd2", positive=True) if sd2 is not None else None
    ratio = check_number(ratio, "ratio")
    if not .01 <= ratio <= 100 or (sd2 is None and ratio != 1):
        raise AnalysisError("invalid_option", "ratio must be in [0.01, 100]; one sample needs ratio=1.")
    maximum = _count(max_n, "max_n", 1, MAX_N)
    # n2 = ceil(ratio*n1), and each group must stay within max_n.
    limit = min(maximum, math.floor(maximum / ratio)) if sd2 is not None else maximum
    while sd2 is not None and limit > 0 and math.ceil(ratio * limit) > maximum:
        limit -= 1
    if limit < 1:
        raise AnalysisError("resource_limit", "No allocation fits max_n for both groups.")
    return sd, sd2, ratio, limit


def _se(n: int, sd: float, sd2: float | None, ratio: float) -> float:
    return math.hypot(sd / math.sqrt(n), sd2 / math.sqrt(math.ceil(ratio * n))
                      if sd2 is not None else 0.)


def _result(method: str, plan: dict[str, Any], scenarios: list[dict[str, Any]],
            **metadata: Any) -> TableSet:
    metadata = dict(method=method, prospective=True, input="prespecified scalar design assumptions",
                    observations_used=False, sample_missing_weights="not applicable; no data input",
                    covariance_df_pvalues_likelihood="not applicable; no fitted model or observed test",
                    device="cpu", precision="float64", stochastic_draws=0,
                    solver="bounded integer search / bracketed effect inversion", **metadata)
    return TableSet({"plan": table([plan]), "scenarios": table(scenarios),
                     "settings": table([[key, json.dumps(value, allow_nan=False)]
                                        for key, value in metadata.items()], columns=["setting", "json"])},
                    title=method + " — prospective design", **metadata)


def _sizes(n: int, minimum: int, maximum: int) -> list[int]:
    return sorted({max(minimum, n - 1), n, min(maximum, n + 1)})


@procedure
def power_mean(effect: float | None = None, *, sd: float, n: int | None = None,
               power: float | None = None, sd2: float | None = None, ratio: float = 1.,
               alpha: float = .05, alternative: str = "two-sided", direction: str | None = None,
               max_n: int = MAX_N) -> TableSet:
    """Known-SD normal mean/fixed-null mean-difference power, integer n1, or MDE.

    Exactly two of ``effect``, ``n`` (first-group size), ``power`` are required.
    Providing ``sd2`` selects independent groups, n2=ceil(ratio*n1). Signed
    effect is population mean minus null, or population mean2-mean1 minus null
    difference. ``direction`` chooses the two-sided detectable-effect branch.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(effect, n, power, alpha)
    sd, sd2, ratio, maximum = _design(sd, sd2, ratio, max_n)
    target = float(power) if power is not None else None
    effect = _number(effect, "effect") if effect is not None else None
    if n is not None:
        n = _count(n, "n", 1, maximum)
    if solve == "effect":
        magnitude = _root(lambda shift: _normal_power(sign * shift, alpha, alternative), target, 64.)
        effect = _number(sign * magnitude * _se(n, sd, sd2, ratio), "detectable effect")
    if solve == "n":
        if effect == 0 or (alternative != "two-sided" and effect * sign <= 0):
            raise AnalysisError("unattainable_design", "Sample-size planning needs a nonzero effect "
                                "in the prespecified alternative direction.")
        n = _minimum_n(lambda size: _normal_power(effect / _se(size, sd, sd2, ratio),
                                                alpha, alternative), target, 1, maximum)
    def row(size: int) -> dict[str, Any]:
        n2 = math.ceil(ratio * size) if sd2 is not None else None
        se = _se(size, sd, sd2, ratio)
        return dict(n=size, n2=n2, total_n=size + (n2 or 0), effect=effect, standard_error=se,
                    power=_normal_power(effect / se, alpha, alternative))
    plan = dict(solve_for=solve, **row(n), sd=sd, sd2=sd2, ratio=ratio, alpha=alpha,
                alternative=alternative, target_power=target)
    return _result("power_mean", plan, [row(size) for size in _sizes(n, 1, maximum)],
                   inference="exact normal; known population SD; independent normal observations",
                   effect_definition="mean-null or (mean2-mean1)-null_difference",
                   allocation="n2=ceil(ratio*n); n denotes group 1", max_n_per_group=int(max_n),
                   sd=sd, sd2=sd2, ratio=ratio, alpha=alpha, alternative=alternative,
                   direction="upper" if sign > 0 else "lower", target_power=target,
                   source="Stata power onemean/twomeans, Methods and formulas (known SD)")


def _rho(value: Any, name: str) -> float:
    value = check_number(value, name)
    if abs(value) > RHO_LIMIT:
        raise AnalysisError("invalid_option", f"{name} must have |rho| <= 1-1e-12.")
    return value


@procedure
def power_correlation(rho0: float, rho: float | None = None, *, n: int | None = None,
                      power: float | None = None, alpha: float = .05,
                      alternative: str = "two-sided", direction: str | None = None,
                      max_n: int = MAX_N) -> TableSet:
    """Approximate Fisher-z Pearson-correlation power, integer n, or detectable rho.

    Specify exactly two of ``rho``, ``n``, ``power``. Assumes iid bivariate normal
    observations; uses N(atanh(rho), 1/(n-3)). This is not finite-sample exact.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(rho, n, power, alpha)
    maximum = _count(max_n, "max_n", 4, MAX_N)
    rho0 = _rho(rho0, "rho0")
    rho = _rho(rho, "rho") if rho is not None else None
    target = float(power) if power is not None else None
    if n is not None:
        n = _count(n, "n", 4, maximum)
    z0 = math.atanh(rho0)
    if solve == "effect":
        scale = math.sqrt(n - 3)
        available = (math.atanh(RHO_LIMIT) - sign * z0) * scale
        shift = _root(lambda distance: _normal_power(sign * distance, alpha, alternative),
                      target, available)
        rho = _rho(math.tanh(z0 + sign * shift / scale), "detectable rho")
    difference = math.atanh(rho) - z0
    if solve == "effect" and abs(_normal_power(difference * math.sqrt(n - 3),
                                               alpha, alternative) - target) > 1e-9:
        raise AnalysisError("numerical_failure", "Detectable correlation loses required power "
                            "accuracy when transformed back to the correlation scale.")
    if solve == "n":
        if difference == 0 or (alternative != "two-sided" and sign * difference <= 0):
            raise AnalysisError("unattainable_design", "rho must differ from rho0 in the "
                                "prespecified alternative direction.")
        n = _minimum_n(lambda size: _normal_power(difference * math.sqrt(size - 3),
                                                alpha, alternative), target, 4, maximum)
    def row(size: int) -> dict[str, Any]:
        return dict(n=size, rho0=rho0, rho=rho, difference=rho - rho0,
                    fisher_difference=difference, fisher_standard_error=1 / math.sqrt(size - 3),
                    power=_normal_power(difference * math.sqrt(size - 3), alpha, alternative))
    plan = dict(solve_for=solve, **row(n), alpha=alpha, alternative=alternative, target_power=target)
    return _result("power_correlation", plan, [row(size) for size in _sizes(n, 4, maximum)],
                   inference="Fisher-z normal approximation; iid bivariate normal observations",
                   variance="1/(n-3); no bias adjustment; n>=4; not exact finite-sample power",
                   rho0=rho0, alpha=alpha, alternative=alternative, max_n=maximum,
                   direction="upper" if sign > 0 else "lower", target_power=target,
                   source="Stata power onecorrelation, Methods and formulas")


def _binomial_pmf(n: int, probability: float) -> torch.Tensor:
    if probability in (0., 1.):
        pmf = torch.zeros(n + 1, dtype=torch.float64, device="cpu")
        pmf[0 if probability == 0 else n] = 1.
        return pmf
    k = torch.arange(n + 1, dtype=torch.float64, device="cpu")
    logs = (math.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
            + k * math.log(probability) + (n - k) * math.log1p(-probability))
    return torch.exp(logs - torch.logsumexp(logs, dim=0))


def _region(n: int, p0: float, alpha: float, alternative: str) -> tuple[int, int, float]:
    if p0 == .5:
        # Symmetric null probabilities are exact integer counts / 2**n. A
        # log-PMF roundoff must not discard equality at a dyadic alpha boundary.
        tail = alpha / 2 if alternative == "two-sided" else alpha
        numerator, denominator = tail.as_integer_ratio()
        outcomes = 1 << n
        threshold = numerator * outcomes
        cumulative, coefficient, accepted_mass, boundary = 1, 1, 0, -1
        for k in range(n // 2 + 1):
            if cumulative * denominator > threshold:
                break
            boundary, accepted_mass = k, cumulative
            coefficient = coefficient * (n - k) // (k + 1)
            cumulative += coefficient
        lo = boundary if alternative != "upper" else -1
        hi = n - boundary if alternative != "lower" and boundary >= 0 else n + 1
        active_tails = 2 if alternative == "two-sided" else 1
        return lo, hi, active_tails * accepted_mass / outcomes
    pmf = _binomial_pmf(n, p0)
    lower = torch.cumsum(pmf, dim=0)
    upper = torch.cumsum(pmf.flip(0), dim=0).flip(0)
    tail = alpha / 2 if alternative == "two-sided" else alpha
    lows = torch.nonzero(lower <= tail).flatten() if alternative != "upper" else []
    highs = torch.nonzero(upper <= tail).flatten() if alternative != "lower" else []
    lo = int(lows[-1]) if len(lows) else -1
    hi = int(highs[0]) if len(highs) else n + 1
    size = float(pmf[:lo + 1].sum() + pmf[hi:].sum())
    if size > alpha + 1e-12 or lo >= hi:
        raise AnalysisError("numerical_failure", "Binomial null rejection-region verification failed.")
    return lo, hi, size


def _binomial_power(n: int, p1: float, lo: int, hi: int) -> float:
    pmf = _binomial_pmf(n, p1)
    return min(1., float(pmf[:lo + 1].sum() + pmf[hi:].sum()))


@procedure
def power_proportion(p0: float, p1: float | None = None, *, n: int | None = None,
                     power: float | None = None, alpha: float = .05,
                     alternative: str = "two-sided", direction: str | None = None,
                     max_n: int = MAX_BINOMIAL_N) -> TableSet:
    """Exact binomial prospective power, smallest integer n, or detectable p1.

    Specify exactly two of ``p1``, ``n``, ``power``. Nonrandomized equal-tail
    rejection X<=lower_critical or X>=upper_critical; disabled tails use -1/n+1.
    n inversion scans every preceding integer (discrete power is not monotone).
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(p1, n, power, alpha)
    maximum = _count(max_n, "max_n", 1, MAX_BINOMIAL_N)
    p0 = check_probability(p0, "p0")
    p1 = check_probability(p1, "p1") if p1 is not None else None
    target = float(power) if power is not None else None
    if n is not None:
        n = _count(n, "n", 1, maximum)
    evaluations = 0
    if solve == "n":
        if p1 == p0 or (alternative != "two-sided" and sign * (p1 - p0) <= 0):
            raise AnalysisError("unattainable_design", "p1 must differ from p0 in the "
                                "prespecified alternative direction.")
        for size in range(1, maximum + 1):
            lo, hi, _ = _region(size, p0, alpha, alternative)
            evaluations += 1
            if _binomial_power(size, p1, lo, hi) >= target:
                n = size
                break
        if n is None:
            raise AnalysisError("resource_limit", "No sample size achieves target power within "
                                "max_n; no partial solution is returned.")
    lo, hi, null_size = _region(n, p0, alpha, alternative)
    if solve == "effect":
        # Acceptance probability of a contiguous binomial interval is unimodal
        # in p. Since null power<=alpha<target, each branch has at most one
        # target crossing even if power initially decreases away from p0.
        distance = _root(lambda d: _binomial_power(n, p0 + sign * d, lo, hi), target,
                         1 - p0 if sign > 0 else p0)
        p1 = p0 + sign * distance
        if not 0 < p1 < 1:
            raise AnalysisError("unattainable_design", "Detectable p1 rounds to a degenerate endpoint.")
    def row(size: int) -> dict[str, Any]:
        lower, upper, actual_alpha = _region(size, p0, alpha, alternative)
        return dict(n=size, p0=p0, p1=p1, difference=p1 - p0,
                    lower_critical=lower, upper_critical=upper, actual_alpha=actual_alpha,
                    power=_binomial_power(size, p1, lower, upper))
    plan = dict(solve_for=solve, **row(n), alpha=alpha, alternative=alternative, target_power=target)
    return _result("power_proportion", plan, [row(size) for size in _sizes(n, 1, maximum)],
                   inference="exact binomial rejection probability; iid Bernoulli fixed success probability",
                   rejection_rule="X<=lower_critical or X>=upper_critical; each active tail <= alpha/k",
                   inactive_tail="lower=-1 / upper=n+1; nonrandomized; no continuity correction",
                   actual_alpha=null_size, p0=p0, alpha=alpha, alternative=alternative,
                   direction="upper" if sign > 0 else "lower", max_n=maximum,
                   n_search="exhaustive from 1 to first qualifying n; power need not be monotone",
                   n_search_evaluations=evaluations, planned_max_pmf_cells=2 * (maximum + 1),
                   planned_max_search_cells=maximum * (maximum + 3),
                   source="Stata power oneproportion binomial rejection rule; independent exact inversion")


@procedure
def precision_mean(*, sd: float, width: float | None = None, n: int | None = None,
                   confidence: float = .95, sd2: float | None = None, ratio: float = 1.,
                   max_n: int = MAX_N) -> TableSet:
    """Known-SD two-sided mean/difference CI total width or minimum integer n1.

    Exactly one of ``width`` (requested maximum total width) and ``n`` is
    required. With ``sd2``: independent groups, n2=ceil(ratio*n1).
    The known-SD width is deterministic; it has no sampling-width probability.
    """
    if (width is None) == (n is None):
        raise AnalysisError("invalid_option", "Specify exactly one of width and n.")
    confidence = check_probability(confidence, "confidence")
    if not .5 < confidence <= 1 - 1e-8:
        raise AnalysisError("invalid_option", "confidence must be in (0.5, 1-1e-8].")
    sd, sd2, ratio, maximum = _design(sd, sd2, ratio, max_n)
    z = normal_isf((1 - confidence) / 2)
    target = _number(width, "width", positive=True) if width is not None else None
    def function(size: int) -> float:
        return 2 * z * _se(size, sd, sd2, ratio)
    solve = "n" if n is None else "width"
    n = _minimum_n(function, target, 1, maximum, decreasing=True) if n is None \
        else _count(n, "n", 1, maximum)
    def row(size: int) -> dict[str, Any]:
        n2 = math.ceil(ratio * size) if sd2 is not None else None
        return dict(n=size, n2=n2, total_n=size + (n2 or 0), width=function(size),
                    standard_error=_se(size, sd, sd2, ratio))
    plan = dict(solve_for=solve, **row(n), target_width=target, confidence=confidence,
                z_critical=z, sd=sd, sd2=sd2, ratio=ratio)
    return _result("precision_mean", plan, [row(size) for size in _sizes(n, 1, maximum)],
                   inference="exact normal two-sided CI width; known SD; independent normal observations",
                   width_definition="total width=2*z_(1-alpha/2)*standard_error; not half-width",
                   width_random=False, width_probability="not applicable; deterministic known-SD width",
                   confidence=confidence, sd=sd, sd2=sd2, ratio=ratio,
                   allocation="n2=ceil(ratio*n); n denotes group 1", max_n_per_group=int(max_n),
                   source="Stata ciwidth onemean/twomeans, Methods and formulas (known SD)")
