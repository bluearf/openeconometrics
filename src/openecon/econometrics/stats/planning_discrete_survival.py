"""Prospective random-discordance power and expected survival-event accrual.

These scalar design calculations use native CPU float64 tensors. They do not
fit observations, infer censoring hazards, or guarantee observed event counts.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import check_number, procedure
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.stats.planning import (
    MAX_BINOMIAL_N,
    MAX_N,
    _binomial_pmf,
    _count,
    _number,
    _options,
    _region,
    _result,
    _sizes,
)
from openecon.resources import plan_workspace


def _unit_probability(value: Any, name: str) -> float:
    number = check_number(value, name)
    if not 0 <= number <= 1:
        raise AnalysisError("invalid_option", f"{name} must be in [0,1].")
    return number


def _solver_record(output: TableSet, description: str) -> TableSet:
    """Save accurate solver metadata and a stable complete-export display."""
    output.attrs["solver"] = description
    # JSON serializes mixed integer/float numeric row matrices as floats.
    # Match those display types while retaining integer identities in the
    # mixed-type plan and metadata, so complete LaTeX survives restoration.
    for name in ("scenarios", "conditional_rejection", "search", "arms"):
        if name in output and len(output[name]):
            output[name] = output[name].astype(float)
    output = saved_summary(output)
    return TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)


def _conditional_law(maximum: int, theta: float | None, alpha: float, alternative: str):
    powers = torch.zeros(maximum + 1, dtype=torch.float64, device="cpu")
    sizes = torch.zeros_like(powers)
    records = [
        {
            "discordant_pairs": 0,
            "lower_critical": -1,
            "upper_critical": 1,
            "conditional_power": 0.0,
            "conditional_null_size": 0.0,
        }
    ]
    if theta is None:
        return powers, sizes, records
    for count in range(1, maximum + 1):
        lower, upper, size = _region(count, 0.5, alpha, alternative)
        pmf = _binomial_pmf(count, theta)
        probability = float(pmf[: lower + 1].sum() + pmf[upper:].sum())
        if not math.isfinite(probability) or not 0 <= probability <= 1 + 1e-14:
            raise AnalysisError("numerical_failure", "Conditional McNemar probability failed.")
        probability = min(1.0, probability)
        powers[count], sizes[count] = probability, size
        records.append(
            {
                "discordant_pairs": count,
                "lower_critical": lower,
                "upper_critical": upper,
                "conditional_power": probability,
                "conditional_null_size": size,
            }
        )
    return powers, sizes, records


@procedure
def power_mcnemar_unconditional(
    p10: float,
    p01: float,
    *,
    n: int | None = None,
    power: float | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    direction: str | None = None,
    max_n: int = 1000,
) -> TableSet:
    """Exact finite-law power for N iid pairs with prespecified p10 and p01.

    Exactly one of ``n`` and target ``power``. D~Bin(N,p10+p01), then
    B|D~Bin(D,p10/(p10+p01)). Reuse the nonrandomized inclusive equal-tail
    conditional rule under B|D~Bin(D,.5), with D=0 never rejecting. Positive
    p10-p01 is the upper alternative. Every earlier N is evaluated for a
    minimum-N claim; neither discrete monotonicity nor an MDE is assumed.
    """
    alpha, alternative, sign = _options(alpha, alternative, direction)
    first, second = _unit_probability(p10, "p10"), _unit_probability(p01, "p01")
    discordance = first + second
    if discordance > 1:
        raise AnalysisError("invalid_option", "p10+p01 must not exceed one.")
    maximum = _count(max_n, "max_n", 1, MAX_BINOMIAL_N)
    if (n is None) == (power is None):
        raise AnalysisError("invalid_option", "Specify exactly one of n and target power.")
    target = None if power is None else check_number(power, "power")
    if target is not None and not alpha < target <= 1 - 1e-12:
        raise AnalysisError("invalid_option", "Target power must exceed alpha and be <=1-1e-12.")
    if n is not None:
        n = _count(n, "n", 1, maximum)
    elif (
        discordance == 0
        or first == second
        or (alternative != "two-sided" and sign * (first - second) <= 0)
    ):
        raise AnalysisError(
            "unattainable_design",
            "The probabilities must differ in the prespecified alternative direction.",
        )

    limit = maximum if n is None else min(maximum, n + 1)
    cells = (limit + 1) * (limit + 2) // 2
    workspace = plan_workspace(
        "unconditional McNemar design",
        {
            "conditional_laws_and_binomial_buffers": 8 * 12 * (limit + 1),
        },
    )
    theta = first / discordance if discordance else None
    conditional_power, conditional_size, regions = _conditional_law(
        limit, theta, alpha, alternative
    )
    evaluated: dict[int, dict[str, Any]] = {}

    def row(count: int) -> dict[str, Any]:
        if count not in evaluated:
            weights = _binomial_pmf(count, discordance)
            actual_power = float(weights @ conditional_power[: count + 1])
            null_size = float(weights @ conditional_size[: count + 1])
            if not (
                math.isfinite(actual_power)
                and 0 <= actual_power <= 1 + 1e-13
                and math.isfinite(null_size)
                and 0 <= null_size <= alpha + 1e-13
            ):
                raise AnalysisError(
                    "numerical_failure", "Unconditional McNemar probability verification failed."
                )
            evaluated[count] = dict(
                n=count,
                pairs=count,
                power=min(1.0, actual_power),
                actual_alpha=null_size,
                expected_discordant_pairs=count * discordance,
                probability_no_discordance=float(weights[0]),
            )
        return evaluated[count]

    search_records = []
    if n is None:
        for count in range(1, maximum + 1):
            candidate = row(count)
            search_records.append(candidate)
            if candidate["power"] >= target:
                n = count
                break
        else:
            raise AnalysisError(
                "resource_limit",
                "Target power is unattainable within max_n; no failed or earlier counts were omitted.",
            )
    scenarios = [row(count) for count in _sizes(n, 1, maximum)]
    output = _result(
        "power_mcnemar_unconditional",
        dict(
            solve_for="power" if target is None else "n",
            **row(n),
            p10=first,
            p01=second,
            discordance_fraction=discordance,
            conditional_probability=theta,
            alpha=alpha,
            alternative=alternative,
            target_power=target,
        ),
        scenarios,
        inference="finite iid paired multinomial law; nonrandomized conditional McNemar rejection mixed over random D",
        count_unit="total independent pairs, not conditioned discordant pairs",
        pair_orientation="p10=P(first=1,second=0); p01=P(first=0,second=1); upper tests p10>p01",
        p10=first,
        p01=second,
        discordance_fraction=discordance,
        conditional_probability=theta,
        null_law="p10=p01=(prespecified discordance fraction)/2; conditional validity holds for all discordance fractions",
        rejection_rule="D=0 never rejects; B<=lower_critical or B>=upper_critical; inclusive equal tails; no randomization",
        alternative=alternative,
        direction="upper" if sign == 1 else "lower",
        alpha=alpha,
        n_search="exhaustive increasing integers, no monotonicity assumption"
        if target is not None
        else None,
        n_search_evaluations=len(search_records),
        max_n=maximum,
        planned_conditional_cells=cells,
        planned_mixture_cells=cells,
        workspace=workspace.record(),
        exclusions=[
            "observed/estimated probabilities",
            "dependent pairs",
            "optional stopping",
            "detectable-effect inversion",
            "continuous/family inference",
            "Dataset/GPU",
        ],
        source="McNemar conditional binomial rule plus law of total probability for iid paired multinomial counts",
    )
    output["conditional_rejection"] = table(
        regions[: min(limit, n + 1) + 1], null_probability=0.5, orientation="B=count of10 pairs"
    )
    output["search"] = table(
        search_records,
        columns=[
            "n",
            "pairs",
            "power",
            "actual_alpha",
            "expected_discordant_pairs",
            "probability_no_discordance",
        ],
    )
    return _solver_record(
        output,
        "finite conditional-binomial mixture; exhaustive minimum-N search with all earlier counts retained",
    )


def _nonnegative(value: Any, name: str) -> float:
    number = _number(value, name)
    if number < 0:
        raise AnalysisError("invalid_option", f"{name} must be nonnegative.")
    return number


def _event_probabilities(
    rates: list[float], dropouts: list[float], accrual: float, followup: float
) -> torch.Tensor:
    event = torch.tensor(rates, dtype=torch.float64, device="cpu")
    total = event + torch.tensor(dropouts, dtype=torch.float64, device="cpu")
    x, y = total * accrual, total * followup
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    # 1-(1-exp(-x))/x: Taylor avoids loss of all significant digits near zero.
    small = torch.where(x < 1e-4, x, torch.zeros_like(x))
    series = small * (0.5 + small * (-1 / 6 + small * (1 / 24 + small * (-1 / 120 + small / 720))))
    one_minus_average = torch.where(x < 1e-4, series, 1 + torch.expm1(-x) / safe_x)
    observed_before_end = -torch.expm1(-y) + torch.exp(-y) * one_minus_average
    safe_total = torch.where(total == 0, torch.ones_like(total), total)
    probability = event / safe_total * observed_before_end
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise AnalysisError(
            "numerical_failure", "Observed event probability is outside finite float64 [0,1]."
        )
    return probability


@procedure
def survival_accrual(
    *,
    event_rate1: float,
    event_rate2: float,
    accrual: float,
    followup: float,
    dropout_rate1: float = 0.0,
    dropout_rate2: float = 0.0,
    allocation: float = 0.5,
    n: int | None = None,
    target_events: float | None = None,
    max_n: int = 10_000_000,
) -> TableSet:
    """Expected observed events or minimal enrollment for target expected events.

    Uniform independent entry on [0,A], administrative study end A+F,
    per-arm exponential event and independent dropout clocks. ``allocation``
    is the nominal arm-2 enrollment share, interpreted as its shortest decimal
    string. Set n2=floor(allocation*N), n1=N-n2 using exact integer arithmetic,
    with both arms nonempty. Exactly one of n/target_events; no
    event-count assurance, logrank power, or estimated risk-set allocation.
    """
    rates = [_nonnegative(event_rate1, "event_rate1"), _nonnegative(event_rate2, "event_rate2")]
    dropouts = [
        _nonnegative(dropout_rate1, "dropout_rate1"),
        _nonnegative(dropout_rate2, "dropout_rate2"),
    ]
    accrual, followup = _nonnegative(accrual, "accrual"), _nonnegative(followup, "followup")
    fraction = check_number(allocation, "allocation")
    if not 0.01 <= fraction <= 0.99:
        raise AnalysisError("invalid_option", "allocation must be in [0.01,0.99].")
    share = Fraction(str(fraction))
    maximum = _count(max_n, "max_n", 2, MAX_N)
    if (n is None) == (target_events is None):
        raise AnalysisError("invalid_option", "Specify exactly one of n and target_events.")
    target = (
        None if target_events is None else _number(target_events, "target_events", positive=True)
    )
    minimum = max(2, (share.denominator + share.numerator - 1) // share.numerator)
    if minimum > maximum:
        raise AnalysisError(
            "resource_limit", "Both integer enrollment arms must be nonempty within max_n."
        )
    if n is not None:
        n = _count(n, "n", minimum, maximum)
    workspace = plan_workspace("survival expected-event accrual", {"scalar_arm_tensors": 1024})
    probabilities = _event_probabilities(rates, dropouts, accrual, followup).tolist()
    evaluations = 0

    def row(count: int) -> dict[str, Any]:
        nonlocal evaluations
        evaluations += 1
        arm2 = (share.numerator * count) // share.denominator
        arm1 = count - arm2
        expected1, expected2 = arm1 * probabilities[0], arm2 * probabilities[1]
        return dict(
            n=count,
            n1=arm1,
            n2=arm2,
            achieved_allocation=arm2 / count,
            expected_events1=expected1,
            expected_events2=expected2,
            expected_events=expected1 + expected2,
            event_probability1=probabilities[0],
            event_probability2=probabilities[1],
        )

    if n is None:
        if row(maximum)["expected_events"] < target:
            raise AnalysisError(
                "unattainable_design",
                "Target expected events cannot be reached within max_n under the declared event/accrual/dropout law.",
            )
        low, high = minimum, maximum
        while low < high:
            middle = (low + high) // 2
            if row(middle)["expected_events"] >= target:
                high = middle
            else:
                low = middle + 1
        n = low
        if row(n)["expected_events"] < target or (
            n > minimum and row(n - 1)["expected_events"] >= target
        ):
            raise AnalysisError(
                "numerical_failure", "Minimum-enrollment boundary verification failed."
            )
    plan = row(n)
    output = _result(
        "survival_accrual",
        dict(
            solve_for="expected_events" if target is None else "n",
            **plan,
            target_events=target,
            allocation=fraction,
            accrual=accrual,
            followup=followup,
        ),
        [row(count) for count in _sizes(n, minimum, maximum)],
        inference="analytic expected observed event counts; no power or count assurance",
        rates={"event": rates, "dropout": dropouts},
        accrual=accrual,
        followup=followup,
        event_probabilities=probabilities,
        allocation=fraction,
        enrollment_rule="n2=floor(allocation*N); n1=N-n2; both arms nonempty; exact arithmetic for shortest decimal allocation string",
        allocation_rational={"numerator": share.numerator, "denominator": share.denominator},
        observation_law="iid per-arm exponential event times independent of exponential dropout and uniform entry; administrative end accrual+followup",
        formula="lambda/q * [1-exp(-q*F)*(-expm1(-q*A))/(q*A)], q=lambda+kappa; continuous A=0 and q=0 limits",
        time_unit="one common declared unit: rates per unit, accrual/followup in that unit",
        count_unit="integer enrolled subjects; expected events are real-valued",
        min_n=minimum,
        max_n=maximum,
        search="monotone expected counts with nondecreasing integer arm counts; bracketed integer search and adjacent boundary verification",
        search_evaluations=evaluations,
        workspace=workspace.record(),
        exclusions=[
            "logrank/Cox test power",
            "probability of attaining target events",
            "hazard/rate estimation",
            "nonuniform accrual",
            "dependent/informative dropout",
            "time-varying hazards",
            "risk-set allocation inference",
            "Dataset/GPU",
        ],
        source="uniform-entry expected failure probability; SAS SEQDESIGN uniform accrual/loss derivation after Lachin-Foulkes",
    )
    output["arms"] = table(
        [
            {
                "arm": j + 1,
                "event_rate": rates[j],
                "dropout_rate": dropouts[j],
                "event_probability": probabilities[j],
                "enrollment": plan["n1" if j == 0 else "n2"],
                "expected_events": plan["expected_events1" if j == 0 else "expected_events2"],
            }
            for j in range(2)
        ]
    )
    return _solver_record(
        output,
        "analytic per-arm event probability; monotone integer enrollment search with adjacent boundary checks",
    )
