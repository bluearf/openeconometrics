"""Prospective random-CI-width laws with complete discrete mass accounting.

Population parameters are design assumptions. No observed data or fitted model
enters these calculations. Discrete first-n searches are exhaustive and bounded;
the Poisson law is never renormalized after its explicit tail cutoff.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import check_number, check_probability, procedure
from openecon.econometrics.nonparametric.distribution import clopper_pearson
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import chi2_cdf, chi2_isf, chi2_ppf, f_ppf, gamma_p, t_isf
from openecon.resources import plan_workspace

from .planning import _binomial_pmf, _count, _number, _result, _sizes

MAX_NORMAL_N = 10_000
MAX_BINOMIAL_N = 2_000
MAX_POISSON_N = 10_000
MAX_COUNT = 100_000
MAX_WIDTH_EVALUATIONS = 20_000
MAX_COUNT_CELLS = 1_000_000
EPSILON = torch.finfo(torch.float64).eps


def _settings(width: Any, n: Any, confidence: Any, assurance: Any):
    if (width is None) == (n is None):
        raise AnalysisError("invalid_option", "Specify exactly one of width and n.")
    confidence = check_probability(confidence, "confidence")
    assurance = check_probability(assurance, "assurance")
    if not 0.5 < confidence <= 1 - 1e-8 or not 1e-6 <= assurance <= 1 - 1e-6:
        raise AnalysisError(
            "invalid_option", "confidence must be in (0.5,1-1e-8], assurance in [1e-6,1-1e-6]."
        )
    target = _number(width, "width", positive=True) if width is not None else None
    return target, confidence, assurance


def _first_n(function, minimum, maximum, assurance):
    for size in range(minimum, maximum + 1):
        low, high = function(size)
        if low >= assurance:
            return size, size - minimum + 1
        if high >= assurance:
            raise AnalysisError(
                "unresolved_assurance",
                "The width probability straddles assurance within its numerical/tail bounds; "
                "tighten tail_tolerance or choose a distinguishable target. No n is skipped.",
            )
    raise AnalysisError("resource_limit", "No qualifying width-assurance design within max_n.")


def _probability_bounds(mass, missing_mass, roundoff):
    return max(0.0, mass - roundoff), min(1.0, mass + missing_mass + roundoff)


def _discrete_quantile(widths, masses, assurance, roundoff):
    # Group exact width ties before assessing the inclusive quantile. In the
    # binomial law complementary count widths share their canonical value.
    grouped: dict[float, list[float]] = {}
    for value, mass in zip(widths, masses, strict=True):
        grouped.setdefault(value, []).append(mass)
    cumulative = 0.0
    for value in sorted(grouped):
        cumulative = math.fsum((cumulative, math.fsum(grouped[value])))
        if cumulative - roundoff >= assurance:
            return value
    raise AnalysisError("unresolved_assurance", "Enumerated probability cannot resolve assurance.")


def _finish(method, plan, scenarios, distribution=None, **metadata):
    result = _result(method, plan, scenarios, **metadata)
    result.attrs["solver"] = (
        "fixed-n native width quantile or exhaustive first-n search; no effect inversion"
    )
    if distribution is not None:
        result["width_distribution"] = table(distribution)
    return saved_summary(result)


@procedure
def precision_twomeans_unknown(
    *,
    sd: float,
    n: int | None = None,
    width: float | None = None,
    ratio: float = 1.0,
    confidence: float = 0.95,
    assurance: float = 0.9,
    max_n: int = 10_000,
) -> TableSet:
    """Pooled-normal two-mean CI width quantile or exhaustive first n1.

    Exactly one of n/width. n2=ceil(ratio*n1), interpreting ratio's shortest
    decimal value exactly; both groups have >=2 observations
    and <=max_n. The common population SD is prespecified, while the pooled
    sample variance is estimated at analysis. Independent iid normal groups
    give df=n1+n2-2 and an exact chi-square random-width law. Unequal population
    variances, Welch intervals and width assurance=confidence coverage are not
    assumed. Searches admit at most 10000 first-group integer designs.
    """
    target, confidence, assurance = _settings(width, n, confidence, assurance)
    sd = _number(sd, "sd", positive=True)
    ratio = check_number(ratio, "ratio")
    if not 0.01 <= ratio <= 100:
        raise AnalysisError("invalid_option", "ratio must lie in [0.01,100].")
    decimal_ratio = Fraction(str(ratio))
    numerator, denominator = decimal_ratio.numerator, decimal_ratio.denominator

    def group_two(size):
        return (size * numerator + denominator - 1) // denominator

    maximum = _count(max_n, "max_n", 2, MAX_NORMAL_N)
    limit = min(maximum, maximum * denominator // numerator)
    minimum = max(2, denominator // numerator + 1)
    if minimum > limit:
        raise AnalysisError("resource_limit", "No two-group allocation fits max_n.")
    resource = plan_workspace("two-mean prospective width", {"scalar design workspace": 4096})

    def row(size):
        n2 = group_two(size)
        df = size + n2 - 2
        scale = 2 * t_isf((1 - confidence) / 2, df) * sd * math.sqrt(1 / size + 1 / n2)
        quantile = chi2_isf(1 - assurance, df) if assurance > 0.5 else chi2_ppf(assurance, df)
        assured_width = scale * math.sqrt(quantile / df)
        if not math.isfinite(assured_width) or assured_width <= 0:
            raise AnalysisError(
                "numerical_failure", "The future width is unresolved at this scale."
            )
        evaluated_width = assured_width if target is None else target
        log_argument = math.log(df) + 2 * (math.log(evaluated_width) - math.log(scale))
        probability = (
            1.0
            if log_argument > 709
            else 0.0
            if log_argument < -745
            else chi2_cdf(math.exp(log_argument), df)
        )
        return dict(
            n=size,
            n2=n2,
            total_n=size + n2,
            df=df,
            width=assured_width,
            target_width=target,
            evaluated_width=evaluated_width,
            width_probability=probability,
            confidence=confidence,
            assurance=assurance,
        )

    evaluations = 0
    if n is None:
        # The exact width quantile assesses the same probability crossing and
        # avoids treating a last-bit CDF inversion error as a failed boundary.
        for size in range(minimum, limit + 1):
            evaluations += 1
            if row(size)["width"] <= target:
                n = size
                break
        if n is None:
            raise AnalysisError(
                "resource_limit", "No pooled width-assurance solution within max_n."
            )
    else:
        n = _count(n, "n", minimum, limit)
    return _finish(
        "precision_twomeans_unknown",
        dict(
            solve_for="n" if target is not None else "width", **row(n), planning_sd=sd, ratio=ratio
        ),
        [row(k) for k in _sizes(n, minimum, limit)],
        inference="exact independent normal common-variance pooled-t CI random-width law; float64 numerical quantiles",
        interval="two-sided pooled Student-t mean-difference CI",
        sampling_distribution="df*pooled_sample_variance/planning_sd² ~ chi-square(df)",
        planning_sd=sd,
        ratio=ratio,
        confidence=confidence,
        assurance=assurance,
        allocation="n2=ceil(ratio*n1), ratio interpreted as its exact shortest decimal value; n means n1; both groups>=2 and<=max_n",
        allocation_ratio_numerator=numerator,
        allocation_ratio_denominator=denominator,
        max_n_per_group=maximum,
        n_search="exhaustive first qualifying n1; no monotonicity assumption",
        n_search_evaluations=evaluations,
        assurance_definition="Pr(future total CI width<=evaluated_width); not coverage",
        comparison="inclusive float64 width<=target; returned width solves the continuous chi-square quantile",
        width_random=True,
        width_definition="total width in mean-difference units",
        resource_plan=resource.record(),
        source="https://support.sas.com/documentation/cdl/en/statug/68162/HTML/default/statug_power_syntax84.htm",
    )


@procedure
def precision_binomial(
    *,
    p: float,
    n: int | None = None,
    width: float | None = None,
    confidence: float = 0.95,
    assurance: float = 0.9,
    max_n: int = 500,
) -> TableSet:
    """Exact Clopper-Pearson width law, discrete quantile or exhaustive first n.

    Exactly one of n/width. p is the prespecified iid Bernoulli population
    proportion, including endpoints. All n+1 counts and their probabilities
    enter the width event; complementary count widths have canonical equal
    values. No observed Wald/normal interval is substituted. <=2000 trials and
    <=20000 CP count-width evaluations across an exhaustive search apply.
    """
    target, confidence, assurance = _settings(width, n, confidence, assurance)
    p = check_number(p, "p")
    if not 0 <= p <= 1:
        raise AnalysisError("invalid_option", "planning p must lie in [0,1].")
    maximum = _count(max_n, "max_n", 1, MAX_BINOMIAL_N)
    calls, count_cells = 0, 0

    def law(size):
        nonlocal calls, count_cells
        evaluations = size // 2 + 1
        if calls + evaluations > MAX_WIDTH_EVALUATIONS or count_cells + size + 1 > MAX_COUNT_CELLS:
            raise AnalysisError(
                "resource_limit",
                "CP exhaustive search exceeds its width/count work budget; no partial design.",
            )
        resource = plan_workspace(
            "binomial prospective width", {"count law and interval buffers": 128 * (size + 1)}
        )
        calls += evaluations
        count_cells += size + 1
        intervals = [None] * (size + 1)
        widths = [0.0] * (size + 1)
        for k in range(evaluations):
            lo, hi = clopper_pearson(k, size, 1 - confidence)
            if k > 0:
                # Preserve the small lower tail directly. The general CP
                # helper uses isf(1-alpha/2), whose subtraction loses relative
                # tail precision at the upper end of our confidence domain.
                f = f_ppf((1 - confidence) / 2, 2 * k, 2 * (size - k + 1))
                lo = k * f / (size - k + 1 + k * f)
            if not 0 <= lo <= hi <= 1:
                raise AnalysisError("numerical_failure", "Native CP interval is unresolved.")
            intervals[k] = (lo, hi)
            intervals[size - k] = (1 - hi, 1 - lo)
            widths[k] = widths[size - k] = hi - lo
        exact_mass = p in (0.0, 1.0) or (p == 0.5 and size <= 53)
        if p == 0.5 and size <= 53:
            # Every combination count and its cumulative sum fits exactly in
            # float64; dividing by 2**n is exact. Preserve dyadic assurance
            # equalities instead of advancing the width by one outcome.
            masses = [math.comb(size, k) / (1 << size) for k in range(size + 1)]
        else:
            masses = _binomial_pmf(size, p).tolist()
        roundoff = 0.0 if exact_mass else 64 * EPSILON * (size + 1)
        assured_width = _discrete_quantile(widths, masses, assurance, roundoff)
        evaluated = assured_width if target is None else target
        represented = math.fsum(
            mass for value, mass in zip(widths, masses, strict=True) if value <= evaluated
        )
        low, high = _probability_bounds(represented, 0.0, roundoff)
        distribution = [
            dict(
                count=k,
                probability=mass,
                ci_low=intervals[k][0],
                ci_high=intervals[k][1],
                width=widths[k],
                qualifies=widths[k] <= evaluated,
            )
            for k, mass in enumerate(masses)
        ]
        return (
            dict(
                n=size,
                width=assured_width,
                target_width=target,
                evaluated_width=evaluated,
                width_probability=represented,
                probability_lower=low,
                probability_upper=high,
                confidence=confidence,
                assurance=assurance,
                enumerated_mass=math.fsum(masses),
                omitted_mass=0.0,
                roundoff_bound=roundoff,
            ),
            distribution,
            resource.record(),
        )

    evaluations = 0
    selected = None
    if n is None:

        def probability(size):
            nonlocal selected
            selected = law(size)
            return selected[0]["probability_lower"], selected[0]["probability_upper"]

        n, evaluations = _first_n(probability, 1, maximum, assurance)
    else:
        n = _count(n, "n", 1, maximum)
        selected = law(n)
    # Keep only selected full law. Adjacent scenario laws are transient; work
    # accounting includes them and no global memo survives the call.
    scenarios = []
    for size in _sizes(n, 1, maximum):
        scenarios.append(selected[0] if size == n else law(size)[0])
    return _finish(
        "precision_binomial",
        dict(solve_for="n" if target is not None else "width", **selected[0], planning_p=p),
        scenarios,
        selected[1],
        inference="exact iid binomial count law and conservative exact CP coverage; float64 numerical inversion",
        interval="two-sided equal-tail Clopper-Pearson",
        planning_p=p,
        confidence=confidence,
        assurance=assurance,
        width_definition="total proportion interval width",
        width_random=True,
        assurance_definition="Pr(future CP width<=evaluated_width); not confidence coverage",
        comparison="inclusive float64 width<=target; complementary counts share canonical width; exact ties grouped",
        quantile="smallest enumerated width whose represented CDF minus roundoff bound reaches assurance",
        n_search="exhaustive first integer; probability ambiguity refuses without skipping n",
        n_search_evaluations=evaluations,
        cp_width_evaluations=calls,
        count_cells_evaluated=count_cells,
        max_cp_width_evaluations=MAX_WIDTH_EVALUATIONS,
        max_count_cells=MAX_COUNT_CELLS,
        max_n=maximum,
        resource_plan=selected[2],
        source="https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html",
    )


@procedure
def precision_poisson(
    *,
    rate: float,
    exposure_per_unit: float = 1.0,
    n: int | None = None,
    width: float | None = None,
    confidence: float = 0.95,
    assurance: float = 0.9,
    max_n: int = 1000,
    tail_tolerance: float = 1e-12,
    max_count: int = 10_000,
) -> TableSet:
    """Garwood rate-CI width law with explicit unrenormalized Poisson tail.

    Exactly one of n/width. Independent homogeneous Poisson units each have
    the fixed exposure_per_unit. rate is the planning event rate per exposure
    unit. Counts 0..cutoff are included; native gamma tails certify omitted
    probability<=tail_tolerance. All probability lower/upper bounds retain
    that omitted mass plus explicit numerical summation tolerance. A first-n
    search refuses when bounds cannot decide; no ambiguous n is skipped.
    Zero planning rate is supported; overdispersion/random exposure is not.
    """
    target, confidence, assurance = _settings(width, n, confidence, assurance)
    rate = check_number(rate, "rate")
    if rate < 0 or rate > 1e100:
        raise AnalysisError("invalid_option", "rate must be finite in [0,1e100].")
    exposure_per_unit = _number(exposure_per_unit, "exposure_per_unit", positive=True)
    maximum = _count(max_n, "max_n", 1, MAX_POISSON_N)
    max_count = _count(max_count, "max_count", 1, MAX_COUNT)
    tail_tolerance = check_number(tail_tolerance, "tail_tolerance")
    if not 1e-14 <= tail_tolerance <= 1e-6:
        raise AnalysisError("invalid_option", "tail_tolerance must lie in [1e-14,1e-6].")
    calls, count_cells = 0, 0

    def law(size):
        nonlocal calls, count_cells
        exposure = size * exposure_per_unit
        mean = rate * exposure
        if rate > 0 and mean == 0:
            raise AnalysisError("numerical_failure", "Positive planning mean underflows float64.")
        if (
            not math.isfinite(mean)
            or not math.isfinite(exposure)
            or exposure <= 0
            or mean > max_count
        ):
            raise AnalysisError(
                "resource_limit", "Planning mean/exposure exceeds the bounded count domain."
            )
        if mean == 0:
            cutoff, tail = 0, 0.0
        else:
            if gamma_p(max_count + 1, mean) > tail_tolerance:
                raise AnalysisError(
                    "resource_limit", "max_count cannot certify the requested omitted-tail budget."
                )
            lo, hi = 0, max_count
            while lo < hi:
                mid = (lo + hi) // 2
                if gamma_p(mid + 1, mean) <= tail_tolerance:
                    hi = mid
                else:
                    lo = mid + 1
            cutoff, tail = lo, gamma_p(lo + 1, mean)
        if calls + cutoff + 1 > MAX_WIDTH_EVALUATIONS or count_cells + cutoff + 1 > MAX_COUNT_CELLS:
            raise AnalysisError(
                "resource_limit",
                "Poisson exhaustive search exceeds its interval/count work budget; no partial design.",
            )
        resource = plan_workspace(
            "Poisson prospective width", {"count law and interval buffers": 160 * (cutoff + 1)}
        )
        calls += cutoff + 1
        count_cells += cutoff + 1
        counts = torch.arange(cutoff + 1, dtype=torch.float64, device="cpu")
        masses = (
            [1.0]
            if mean == 0
            else torch.exp(counts * math.log(mean) - mean - torch.lgamma(counts + 1)).tolist()
        )
        intervals = []
        widths = []
        for k in range(cutoff + 1):
            low = 0.0 if k == 0 else chi2_ppf((1 - confidence) / 2, 2 * k) / (2 * exposure)
            high = chi2_isf((1 - confidence) / 2, 2 * (k + 1)) / (2 * exposure)
            if not math.isfinite(high) or not 0 <= low < high:
                raise AnalysisError(
                    "numerical_failure", "Garwood interval is unresolved at this exposure scale."
                )
            intervals.append((low, high))
            widths.append(high - low)
        enumerated_mass = math.fsum(masses)
        roundoff = 0.0 if mean == 0 else 64 * EPSILON * (cutoff + 1) * max(1.0, math.log1p(mean))
        if abs(enumerated_mass + tail - 1.0) > roundoff:
            raise AnalysisError(
                "numerical_failure", "Poisson probabilities fail complete mass accounting."
            )
        assured_width = _discrete_quantile(widths, masses, assurance, roundoff)
        evaluated = assured_width if target is None else target
        represented = math.fsum(
            mass for value, mass in zip(widths, masses, strict=True) if value <= evaluated
        )
        low, high = _probability_bounds(represented, tail, roundoff)
        distribution = [
            dict(
                count=k,
                probability=mass,
                ci_low=intervals[k][0],
                ci_high=intervals[k][1],
                width=widths[k],
                qualifies=widths[k] <= evaluated,
            )
            for k, mass in enumerate(masses)
        ]
        return (
            dict(
                n=size,
                total_exposure=exposure,
                expected_count=mean,
                width=assured_width,
                target_width=target,
                evaluated_width=evaluated,
                width_probability=represented,
                probability_lower=low,
                probability_upper=high,
                confidence=confidence,
                assurance=assurance,
                enumerated_mass=enumerated_mass,
                omitted_mass=tail,
                roundoff_bound=roundoff,
                count_cutoff=cutoff,
            ),
            distribution,
            resource.record(),
        )

    evaluations = 0
    selected = None
    if n is None:

        def probability(size):
            nonlocal selected
            selected = law(size)
            return selected[0]["probability_lower"], selected[0]["probability_upper"]

        n, evaluations = _first_n(probability, 1, maximum, assurance)
    else:
        n = _count(n, "n", 1, maximum)
        selected = law(n)
    scenarios = [selected[0] if size == n else law(size)[0] for size in _sizes(n, 1, maximum)]
    return _finish(
        "precision_poisson",
        dict(solve_for="n" if target is not None else "width", **selected[0], planning_rate=rate),
        scenarios,
        selected[1],
        inference="exact homogeneous Poisson law/Garwood coverage; bounded float64 tail accounting",
        interval="two-sided equal-tail Garwood rate interval; zero count lower endpoint=0",
        planning_rate=rate,
        exposure_per_unit=exposure_per_unit,
        confidence=confidence,
        assurance=assurance,
        width_definition="total rate-CI width; events per exposure unit",
        width_random=True,
        assurance_definition="Pr(future Garwood width<=evaluated_width); bounds include omitted tail and roundoff, not coverage",
        comparison="inclusive float64 width<=target; every enumerated count is retained",
        quantile="smallest enumerated width with represented CDF minus roundoff>=assurance; conservative within omitted mass budget",
        count_law="Poisson(rate*n*exposure_per_unit); no tail renormalization or replacement draws",
        omitted_tail="native gamma_p(cutoff+1, expected_count); all probability bounds retain tail",
        tail_tolerance=tail_tolerance,
        max_count=max_count,
        max_n=maximum,
        n_search="exhaustive first integer; ambiguous probability bounds refuse without skipping n",
        n_search_evaluations=evaluations,
        interval_width_evaluations=calls,
        count_cells_evaluated=count_cells,
        max_interval_width_evaluations=MAX_WIDTH_EVALUATIONS,
        max_count_cells=MAX_COUNT_CELLS,
        resource_plan=selected[2],
        source="https://www.statsmodels.org/dev/generated/statsmodels.stats.rates.confint_poisson.html",
    )
