"""Bounded prospective designs: known-SD laws and explicitly named approximations.

No observations, random draws, fitted models or third-party distribution kernels.
Count units and width assurance are stored in ordinary result tables.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.nonparametric.common import check_probability, procedure
from openecon.econometrics.stats.planning import (
    MAX_N,
    RHO_LIMIT,
    _count,
    _design,
    _minimum_n,
    _normal_power,
    _number,
    _options,
    _result,
    _rho,
    _root,
    _sizes,
    _solve,
    power_proportion,
)
from openecon.engines.distributions import chi2_cdf, chi2_isf, chi2_ppf, t_isf

MAX_PRECISION_N = 10_000
MAX_PROJECTED_COUNT = 1_000_000_000


def _normal_solution(
    effect: float | None,
    n: int | None,
    target: float | None,
    se: Callable[[int], float],
    minimum: int,
    maximum: int,
    alpha: float,
    alternative: str,
    sign: int,
    effect_limit: float = 1e100,
) -> tuple[str, float, int]:
    solve = _solve(effect, n, target, alpha)
    if n is not None:
        n = _count(n, "n", minimum, maximum)
    if solve == "effect":
        shift = _root(
            lambda x: _normal_power(sign * x, alpha, alternative),
            target,
            min(64.0, effect_limit / se(n)),
        )
        effect = _number(sign * shift * se(n), "detectable effect")
    if solve == "n":
        if effect == 0 or (alternative != "two-sided" and sign * effect <= 0):
            raise AnalysisError(
                "unattainable_design",
                "A nonzero effect in the alternative direction is required for count inversion.",
            )
        n = _minimum_n(
            lambda size: _normal_power(effect / se(size), alpha, alternative),
            target,
            minimum,
            maximum,
        )
    return solve, effect, n


def _power_settings(
    alpha: float, alternative: str, sign: int, target: float | None, maximum: int
) -> dict[str, Any]:
    return dict(
        alpha=alpha,
        alternative=alternative,
        direction="upper" if sign > 0 else "lower",
        target_power=target,
        max_n=maximum,
    )


@procedure
def power_paired_mean(
    effect: float | None = None,
    *,
    sd_before: float,
    sd_after: float,
    correlation: float,
    n: int | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_n: int = MAX_N,
) -> TableSet:
    """Exact known-SD normal paired-difference power, pair count or detectable effect.

    Exactly two of effect/n/power. Independent iid normal pairs; effect is
    (after-before)-null. Both marginal population SDs and correlation are known.
    n counts independent pairs, not measurements; no estimated-SD t power.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    before = _number(sd_before, "sd_before", positive=True)
    after = _number(sd_after, "sd_after", positive=True)
    rho = _rho(correlation, "correlation")
    maximum = _count(max_n, "max_n", 1, MAX_N)
    difference_sd = math.hypot(
        before - after, math.sqrt(before) * math.sqrt(after) * math.sqrt(2 * (1 - rho))
    )
    effect = _number(effect, "effect") if effect is not None else None
    solve, effect, n = _normal_solution(
        effect,
        n,
        power,
        lambda k: difference_sd / math.sqrt(k),
        1,
        maximum,
        alpha,
        alternative,
        sign,
    )

    def row(size: int) -> dict[str, Any]:
        se = difference_sd / math.sqrt(size)
        return dict(
            n=size,
            pairs=size,
            measurements=2 * size,
            effect=effect,
            difference_sd=difference_sd,
            standard_error=se,
            power=_normal_power(effect / se, alpha, alternative),
        )

    return _result(
        "power_paired_mean",
        dict(
            solve_for=solve,
            **row(n),
            sd_before=before,
            sd_after=after,
            correlation=rho,
            alpha=alpha,
            alternative=alternative,
            target_power=power,
        ),
        [row(k) for k in _sizes(n, 1, maximum)],
        inference="exact normal; known bivariate-normal population covariance; iid pairs",
        count_unit="independent pairs; measurements=2*n",
        effect_definition="(after-before)-null_difference",
        sd_before=before,
        sd_after=after,
        correlation=rho,
        difference_sd=difference_sd,
        **_power_settings(alpha, alternative, sign, power, maximum),
    )


def _allocation(ratio: float, maximum: int, minimum2: int) -> tuple[float, int, int]:
    _, _, ratio, limit = _design(1.0, 1.0, ratio, maximum)
    minimum = _minimum_n(lambda k: float(math.ceil(ratio * k) >= minimum2), 1.0, 1, limit)
    return ratio, minimum, limit


def _expected_minimum(probability: float) -> int:
    # Binary representation of .9 or .8 must not require a gratuitous extra trial.
    smallest = math.floor(10 / min(probability, 1 - probability))
    return smallest if smallest * min(probability, 1 - probability) >= 10 - 1e-12 else smallest + 1


def _fraction(value: Any, name: str) -> float:
    value = _number(value, name, positive=True)
    if value > 1:
        raise AnalysisError("invalid_option", f"{name} must be in (0,1].")
    return value


def _proportion(value: Any, name: str) -> float:
    value = check_probability(value, name)
    if not 0.01 <= value <= 0.99:
        raise AnalysisError("invalid_option", f"{name} must be in [0.01, 0.99].")
    return value


@procedure
def power_two_proportions(
    p1: float,
    p2: float | None = None,
    *,
    n: int | None = None,
    power: float | None = None,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_n: int = MAX_N,
) -> TableSet:
    """Cohen arcsine-h normal approximation: power, group-1 n or detectable p2.

    Exactly two of p2/n/power. Independent Bernoulli groups; n2=ceil(ratio*n).
    Probabilities [0.01,0.99], >=10 expected successes AND failures per group.
    Not score, Pearson, Wald, exact or continuity-corrected power.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(p2, n, power, alpha)
    p1 = _proportion(p1, "p1")
    p2 = _proportion(p2, "p2") if p2 is not None else None
    # An effect inversion starts from the null; require it in the admissible domain too.
    base2 = p1 if p2 is None else p2
    minimum1 = _expected_minimum(p1)
    minimum2 = _expected_minimum(base2)
    ratio, minimum, maximum = _allocation(ratio, max_n, minimum2)
    minimum = max(minimum, minimum1)
    if minimum > maximum:
        raise AnalysisError(
            "resource_limit",
            "No allocation has 10 expected successes/failures per group within max_n.",
        )

    def transform(p):
        return 2 * math.asin(math.sqrt(p))

    center = transform(p1)

    def se(k):
        return math.sqrt(1 / k + 1 / math.ceil(ratio * k))

    if solve == "effect":
        n = _count(n, "n", minimum, maximum)
        n2 = math.ceil(ratio * n)
        endpoint = min(0.99, 1 - 10 / n2) if sign > 0 else max(0.01, 10 / n2)
        available = sign * (transform(endpoint) - center)
        distance = _root(
            lambda d: _normal_power(sign * d / se(n), alpha, alternative), power, available
        )
        p2 = _proportion(math.sin((center + sign * distance) / 2) ** 2, "detectable p2")
    difference = transform(p2) - center
    _, _, n = _normal_solution(
        difference,
        n,
        power if solve == "n" else None,
        se,
        minimum,
        maximum,
        alpha,
        alternative,
        sign,
    )
    if (
        solve == "effect"
        and abs(_normal_power(difference / se(n), alpha, alternative) - power) > 1e-9
    ):
        raise AnalysisError("numerical_failure", "Detectable probability loses power accuracy.")

    def row(size: int) -> dict[str, Any]:
        n2 = math.ceil(ratio * size)
        return dict(
            n=size,
            n2=n2,
            total_n=size + n2,
            p1=p1,
            p2=p2,
            difference=p2 - p1,
            arcsine_h=difference,
            standard_error=se(size),
            power=_normal_power(difference / se(size), alpha, alternative),
        )

    valid_sizes = [
        k
        for k in _sizes(n, minimum, maximum)
        if math.ceil(ratio * k) * min(p2, 1 - p2) >= 10 - 1e-12
    ]
    if math.ceil(ratio * n) * min(p2, 1 - p2) < 10 - 1e-12:
        raise AnalysisError("unattainable_design", "Detectable p2 violates expected-count domain.")
    return _result(
        "power_two_proportions",
        dict(
            solve_for=solve,
            **row(n),
            ratio=ratio,
            alpha=alpha,
            alternative=alternative,
            target_power=power,
        ),
        [row(k) for k in valid_sizes],
        inference="Cohen arcsine-h normal approximation; independent Bernoulli groups",
        approximation="h=2*(asin(sqrt(p2))-asin(sqrt(p1))); variance=1/n+1/n2",
        domain="p1,p2 in [0.01,0.99]; >=10 expected successes and failures in each group",
        count_unit="n=group 1; n2=ceil(ratio*n)",
        p1=p1,
        ratio=ratio,
        max_n_per_group=int(max_n),
        **_power_settings(alpha, alternative, sign, power, maximum),
    )


@procedure
def power_two_correlations(
    rho1: float,
    rho2: float | None = None,
    *,
    n: int | None = None,
    power: float | None = None,
    ratio: float = 1.0,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_n: int = MAX_N,
) -> TableSet:
    """Independent Fisher-z correlation-difference approximation, n or detectable rho2.

    Exactly two of rho2/n/power. Two iid bivariate-normal samples, each n>=4.
    Variance=1/(n-3)+1/(n2-3); n2=ceil(ratio*n). No shared/overlapping samples.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    solve = _solve(rho2, n, power, alpha)
    rho1 = _rho(rho1, "rho1")
    rho2 = _rho(rho2, "rho2") if rho2 is not None else None
    maximum_input = _count(max_n, "max_n", 4, MAX_N)
    ratio, minimum, maximum = _allocation(ratio, maximum_input, 4)
    minimum = max(4, minimum)
    if minimum > maximum:
        raise AnalysisError("resource_limit", "Both correlation groups need n>=4 within max_n.")

    def se(k):
        return math.sqrt(1 / (k - 3) + 1 / (math.ceil(ratio * k) - 3))

    center = math.atanh(rho1)
    if solve == "effect":
        n = _count(n, "n", minimum, maximum)
        distance = _root(
            lambda d: _normal_power(sign * d / se(n), alpha, alternative),
            power,
            math.atanh(RHO_LIMIT) - sign * center,
        )
        rho2 = _rho(math.tanh(center + sign * distance), "detectable rho2")
    difference = math.atanh(rho2) - center
    _, _, n = _normal_solution(
        difference,
        n,
        power if solve == "n" else None,
        se,
        minimum,
        maximum,
        alpha,
        alternative,
        sign,
    )
    if (
        solve == "effect"
        and abs(_normal_power(difference / se(n), alpha, alternative) - power) > 1e-9
    ):
        raise AnalysisError("numerical_failure", "Detectable correlation loses power accuracy.")

    def row(size: int) -> dict[str, Any]:
        n2 = math.ceil(ratio * size)
        return dict(
            n=size,
            n2=n2,
            total_n=size + n2,
            rho1=rho1,
            rho2=rho2,
            difference=rho2 - rho1,
            fisher_difference=difference,
            standard_error=se(size),
            power=_normal_power(difference / se(size), alpha, alternative),
        )

    return _result(
        "power_two_correlations",
        dict(
            solve_for=solve,
            **row(n),
            ratio=ratio,
            alpha=alpha,
            alternative=alternative,
            target_power=power,
        ),
        [row(k) for k in _sizes(n, minimum, maximum)],
        inference="Fisher-z normal approximation; independent iid bivariate-normal groups",
        variance="1/(n-3)+1/(n2-3); both n>=4; no bias adjustment; not finite-sample exact",
        count_unit="n=group 1; n2=ceil(ratio*n)",
        rho1=rho1,
        ratio=ratio,
        max_n_per_group=maximum_input,
        **_power_settings(alpha, alternative, sign, power, maximum),
    )


@procedure
def power_slope(
    effect: float | None = None,
    *,
    error_sd: float,
    design_variance: float,
    n: int | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_n: int = MAX_N,
) -> TableSet:
    """Exact known-error-SD fixed-X simple-regression slope power, n or MDE.

    Exactly two of effect/n/power; effect=slope-null. Prespecify positive
    design_variance=sum((x-xbar)^2)/n, held fixed as n changes. Includes intercept,
    iid normal errors and n>=2; not random-X R-squared or unknown-SD t/F power.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    sd = _number(error_sd, "error_sd", positive=True)
    variance = _number(design_variance, "design_variance", positive=True)
    effect = _number(effect, "effect") if effect is not None else None
    maximum = _count(max_n, "max_n", 2, MAX_N)

    def se(k):
        return sd / (math.sqrt(k) * math.sqrt(variance))

    solve, effect, n = _normal_solution(effect, n, power, se, 2, maximum, alpha, alternative, sign)

    def row(size: int) -> dict[str, Any]:
        return dict(
            n=size,
            effect=effect,
            centered_gram=size * variance,
            standard_error=se(size),
            power=_normal_power(effect / se(size), alpha, alternative),
        )

    return _result(
        "power_slope",
        dict(
            solve_for=solve,
            **row(n),
            error_sd=sd,
            design_variance=variance,
            alpha=alpha,
            alternative=alternative,
            target_power=power,
        ),
        [row(k) for k in _sizes(n, 2, maximum)],
        inference="exact normal conditional on full-rank fixed X; known error SD; intercept+slope",
        design="sum((x-xbar)^2)=n*design_variance; same prespecified Gram per observation at each n",
        effect_definition="slope-null_slope",
        count_unit="independent observations at fixed X",
        error_sd=sd,
        design_variance=variance,
        **_power_settings(alpha, alternative, sign, power, maximum),
    )


def _projection(count: int, fraction: float | None, name: str) -> int | None:
    if fraction is None:
        return None
    projected = count / fraction
    if projected > MAX_PROJECTED_COUNT:
        raise AnalysisError("resource_limit", f"Expected {name} projection exceeds 1,000,000,000.")
    return math.ceil(projected)


@procedure
def power_logrank(
    hazard_ratio: float | None = None,
    *,
    events: int | None = None,
    power: float | None = None,
    information_fraction: float = 0.5,
    event_fraction: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_events: int = MAX_N,
) -> TableSet:
    """Schoenfeld PH/local-alternative approximation: power, events or detectable HR.

    Exactly two of hazard_ratio/events/power. information_fraction is the
    prespecified constant group-2 fraction in risk sets, not inferred enrollment.
    >=10 expected information events per group. Optional event_fraction projects
    expected enrollment only; no censoring/accrual model or event guarantee.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    q = check_probability(information_fraction, "information_fraction")
    if not 0.05 <= q <= 0.95:
        raise AnalysisError("invalid_option", "information_fraction must be in [0.05,0.95].")
    fraction = _fraction(event_fraction, "event_fraction") if event_fraction is not None else None
    maximum = _count(max_events, "max_events", 1, MAX_N)
    minimum = _expected_minimum(q)
    if minimum > maximum:
        raise AnalysisError(
            "resource_limit", "At least 10 expected information events per group are required."
        )
    effect = (
        math.log(_number(hazard_ratio, "hazard_ratio", positive=True))
        if hazard_ratio is not None
        else None
    )

    def se(k):
        return 1 / math.sqrt(k * q * (1 - q))

    solve, effect, events = _normal_solution(
        effect,
        events,
        power,
        se,
        minimum,
        maximum,
        alpha,
        alternative,
        sign,
        effect_limit=math.log(1e100),
    )
    hr = _number(math.exp(effect), "detectable hazard_ratio", positive=True)

    def row(size: int) -> dict[str, Any]:
        return dict(
            events=size,
            hazard_ratio=hr,
            log_hazard_ratio=effect,
            information=size * q * (1 - q),
            standard_error=se(size),
            power=_normal_power(effect / se(size), alpha, alternative),
            expected_enrollment=_projection(size, fraction, "enrollment"),
        )

    return _result(
        "power_logrank",
        dict(
            solve_for="events" if solve == "n" else solve,
            **row(events),
            information_fraction=q,
            event_fraction=fraction,
            alpha=alpha,
            alternative=alternative,
            target_power=power,
        ),
        [
            row(k)
            for k in _sizes(events, minimum, maximum)
            if fraction is None or k / fraction <= MAX_PROJECTED_COUNT
        ],
        inference="Schoenfeld PH/local-alternative normal approximation; distinct event times",
        information_assumption="constant prespecified group-2 risk-set fraction; not inferred enrollment ratio",
        count_unit="events; >=10 expected information events per group",
        projection="ceil(events/event_fraction); expectation only, no event/accrual/follow-up guarantee",
        information_fraction=q,
        event_fraction=fraction,
        max_events=maximum,
        **_power_settings(alpha, alternative, sign, power, maximum),
    )


@procedure
def power_mcnemar(
    probability: float | None = None,
    *,
    discordant_pairs: int | None = None,
    power: float | None = None,
    discordance_fraction: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_discordant_pairs: int = 1000,
) -> TableSet:
    """Conditional exact McNemar power, fixed discordant-pair count or detectable r.

    Exactly two of probability/discordant_pairs/power. Given fixed m=b+c,
    b~Binomial(m,r), H0:r=.5; equal-tail nonrandomized rejection. Optional
    discordance_fraction projects expected total pairs, not unconditional power.
    """
    fraction = (
        _fraction(discordance_fraction, "discordance_fraction")
        if discordance_fraction is not None
        else None
    )
    original = power_proportion(
        0.5,
        probability,
        n=discordant_pairs,
        power=power,
        alpha=alpha,
        alternative=alternative,
        direction=direction,
        max_n=max_discordant_pairs,
    )

    def rename(row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        count = row.pop("n")
        row.pop("p0")
        row["probability"] = row.pop("p1")
        row["discordant_pairs"] = count
        row["expected_total_pairs"] = _projection(count, fraction, "total pairs")
        if row.get("solve_for") == "n":
            row["solve_for"] = "discordant_pairs"
        return row

    metadata = {
        key: value
        for key, value in original.attrs.items()
        if key
        in (
            "alpha",
            "alternative",
            "direction",
            "target_power",
            "actual_alpha",
            "n_search",
            "n_search_evaluations",
            "planned_max_pmf_cells",
            "planned_max_search_cells",
        )
    }
    return _result(
        "power_mcnemar",
        dict(**rename(original["plan"].to_dict("records")[0]), discordance_fraction=fraction),
        [
            rename(row)
            for row in original["scenarios"].to_dict("records")
            if fraction is None or row["n"] / fraction <= MAX_PROJECTED_COUNT
        ],
        inference="conditional exact McNemar; given fixed discordant count m; b~Binomial(m,r), null r=.5",
        count_unit="fixed discordant pairs; no unconditional random-discordance power",
        rejection_rule="equal tails; inclusive b<=lower_critical or b>=upper_critical; nonrandomized",
        projection="ceil(m/discordance_fraction); expected total pairs only, never guaranteed m",
        discordance_fraction=fraction,
        max_discordant_pairs=int(max_discordant_pairs),
        **metadata,
    )


def _precision(
    method: str,
    parameter: float,
    width: Any,
    n: Any,
    confidence: Any,
    assurance: Any,
    max_n: Any,
    *,
    variance_interval: bool,
) -> TableSet:
    if (width is None) == (n is None):
        raise AnalysisError("invalid_option", "Specify exactly one of width and n.")
    confidence = check_probability(confidence, "confidence")
    assurance = check_probability(assurance, "assurance")
    if not 0.5 < confidence <= 1 - 1e-8 or not 1e-6 <= assurance <= 1 - 1e-6:
        raise AnalysisError(
            "invalid_option", "confidence in (0.5,1-1e-8]; assurance in [1e-6,1-1e-6]."
        )
    maximum = _count(max_n, "max_n", 2, MAX_PRECISION_N)
    target = _number(width, "width", positive=True) if width is not None else None
    # Local bounded memo: reused adjacent rows and probability checks, never global state.
    criticals: dict[int, tuple[float, float]] = {}

    def critical(size: int) -> tuple[float, float]:
        if size not in criticals:
            df = size - 1
            if variance_interval:
                lower = chi2_ppf((1 - confidence) / 2, df)
                upper = chi2_isf((1 - confidence) / 2, df)
                scale = parameter * (1 / lower - 1 / upper)
            else:
                scale = 2 * t_isf((1 - confidence) / 2, df) * parameter / math.sqrt(size * df)
            quantile = chi2_isf(1 - assurance, df) if assurance > 0.5 else chi2_ppf(assurance, df)
            assured_width = scale * (quantile if variance_interval else math.sqrt(quantile))
            if not math.isfinite(assured_width) or assured_width <= 0:
                raise AnalysisError(
                    "numerical_failure", "Invalid width quantile; revise planning scale."
                )
            criticals[size] = scale, assured_width
        return criticals[size]

    evaluations = 0
    if n is None:
        for size in range(2, maximum + 1):
            evaluations += 1
            if critical(size)[1] <= target:
                n = size
                break
        if n is None:
            raise AnalysisError(
                "resource_limit", "No width-assurance solution within max_n; no partial result."
            )
    else:
        n = _count(n, "n", 2, maximum)

    def row(size: int) -> dict[str, Any]:
        scale, assured_width = critical(size)
        test_width = assured_width if target is None else target
        # Log-scale inversion avoids overflow in squared mean-width ratios.
        argument_log = math.log(test_width) - math.log(scale)
        if not variance_interval:
            argument_log *= 2
        probability = (
            1.0
            if argument_log > 709
            else 0.0
            if argument_log < -745
            else chi2_cdf(math.exp(argument_log), size - 1)
        )
        return dict(
            n=size,
            df=size - 1,
            width=assured_width,
            target_width=target,
            width_probability=probability,
            confidence=confidence,
            assurance=assurance,
        )

    return _result(
        method,
        dict(
            solve_for="n" if target is not None else "width",
            **row(n),
            **{"planning_variance" if variance_interval else "planning_sd": parameter},
        ),
        [row(k) for k in _sizes(n, 2, maximum)],
        inference="exact iid normal chi-square sampling law; sample variance estimated at analysis",
        interval="equal-tail variance CI" if variance_interval else "two-sided Student-t mean CI",
        width_definition="total width in variance units"
        if variance_interval
        else "total width in mean units",
        width_random=True,
        sampling_distribution_df="n-1 for chi-square and Student-t widths; stored per scenario; no fitted-model df",
        assurance_definition="Pr(future total CI width <= requested width); distinct from coverage",
        confidence=confidence,
        assurance=assurance,
        planning_parameter=parameter,
        assumption="prespecified true population variance"
        if variance_interval
        else "prespecified true population SD",
        max_n=maximum,
        n_search="exhaustive from 2 to first qualifying integer; no monotonicity assumption",
        n_search_evaluations=evaluations,
        max_cached_integer_rows=maximum - 1,
    )


@procedure
def precision_mean_unknown(
    *,
    sd: float,
    width: float | None = None,
    n: int | None = None,
    confidence: float = 0.95,
    assurance: float = 0.9,
    max_n: int = MAX_PRECISION_N,
) -> TableSet:
    """Student-t mean CI width quantile or first n meeting width assurance.

    Exactly one of width/n. iid normal data; population SD is a planning
    assumption, sample SD will be estimated at analysis. Assurance is a future
    width probability, distinct from confidence coverage. Bounded exhaustive n.
    """
    return _precision(
        "precision_mean_unknown",
        _number(sd, "sd", positive=True),
        width,
        n,
        confidence,
        assurance,
        max_n,
        variance_interval=False,
    )


@procedure
def precision_variance(
    *,
    variance: float,
    width: float | None = None,
    n: int | None = None,
    confidence: float = 0.95,
    assurance: float = 0.9,
    max_n: int = MAX_PRECISION_N,
) -> TableSet:
    """Exact normal variance CI total-width quantile or first n meeting assurance.

    Exactly one of width/n. Equal-tail chi-square interval for population
    variance, not SD. variance is the prespecified population planning value;
    assurance is a width probability, distinct from confidence coverage.
    """
    return _precision(
        "precision_variance",
        _number(variance, "variance", positive=True),
        width,
        n,
        confidence,
        assurance,
        max_n,
        variance_interval=True,
    )
