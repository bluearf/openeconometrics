"""Complete fixed-margin common-odds distribution and fixed-exposure Garwood rate."""

from __future__ import annotations

import math
from numbers import Integral

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines import distributions as dist
from . import common as c


def count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 10000:
        raise AnalysisError(
            "invalid_count", "Counts must be nonnegative integers <=10000, not booleans."
        )
    return int(value)


def logchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def bisect(function, target, increasing):
    lo, hi = -40.0, 40.0
    a, b = function(lo) - target, function(hi) - target
    if not (a <= 0 <= b if increasing else b <= 0 <= a):
        raise AnalysisError(
            "unbracketed_interval",
            "Conditional root is outside the declared log-odds +/-40 domain.",
        )
    for _ in range(60):
        mid = (lo + hi) / 2
        if (function(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    result = (lo + hi) / 2
    if abs(function(result) - target) > 1e-9:
        raise AnalysisError(
            "nonconvergence", "Conditional exact root failed its tail/moment check."
        )
    return result


@resident_cpu
def exact_logistic(
    tables, *, null_odds=1.0, level=0.95, max_work=100_000_000, device="cpu", weights=None
):
    """Exact common exposure odds in independent 2x2 strata with fixed row/outcome margins.

    Each table is [[exposed events, exposed non-events], [unexposed events,
    unexposed non-events]]. At most 16 strata, 4096 total counts and 513 full
    support points. Complete log-space convolution; inclusive equal-tail CI
    and doubled smaller-tail p. CMLE zero/infinity boundaries stay explicit.
    """
    c.domain(device, weights)
    level = c.confidence(level)
    null_odds = c.c.check_number(
        null_odds, "null_odds", minimum=math.exp(-40), maximum=math.exp(40)
    )
    if not isinstance(tables, (list, tuple)) or not tables or len(tables) > 16:
        raise AnalysisError(
            "invalid_spec", "Declare 1..16 independent 2x2 integer strata as a list."
        )
    validated, total, offset, width, observed = [], 0, 0, 1, 0
    for item in tables:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in item)
        ):
            raise AnalysisError(
                "invalid_spec", "Each stratum must be a complete 2x2 integer table."
            )
        a, b, cc, d = [count(v) for row in item for v in row]
        r1, r0, events = a + b, cc + d, a + cc
        low, high = max(0, events - r0), min(r1, events)
        total += r1 + r0
        offset += low
        width += high - low
        observed += a
        validated.append(
            {
                "cells": [[a, b], [cc, d]],
                "exposed": r1,
                "unexposed": r0,
                "events": events,
                "support_low": low,
                "support_high": high,
            }
        )
    if total > 4096 or width > 513:
        raise AnalysisError(
            "resource_limit",
            "Exact common odds requires <=4096 total counts and <=513 complete support points.",
        )
    if width == 1:
        raise AnalysisError(
            "unidentified_design", "Fixed margins contain no common-odds information."
        )
    settings = c.guard(
        "exact_logistic",
        total,
        1,
        work=128 * len(validated) * width**2,
        max_work=max_work,
        records=width,
    )
    base = torch.zeros(1, dtype=torch.float64)
    for row in validated:
        low, high = row["support_low"], row["support_high"]
        stratum = torch.tensor(
            [
                logchoose(row["exposed"], k) + logchoose(row["unexposed"], row["events"] - k)
                for k in range(low, high + 1)
            ],
            dtype=torch.float64,
        )
        combined = torch.empty(len(base) + len(stratum) - 1, dtype=torch.float64)
        for k in range(len(combined)):
            left, right = max(0, k - len(stratum) + 1), min(k, len(base) - 1)
            idx = torch.arange(left, right + 1, dtype=torch.int64)
            combined[k] = torch.logsumexp(base[idx] + stratum[k - idx], 0)
        base = combined - torch.max(combined)
    support = torch.arange(offset, offset + len(base), dtype=torch.float64)
    index = observed - offset

    def probabilities(eta):
        return torch.softmax(base + eta * (support - support[0]), 0)

    def mean(eta):
        return float(probabilities(eta) @ support)

    null_prob = probabilities(math.log(null_odds))
    lower_tail, upper_tail = float(null_prob[: index + 1].sum()), float(null_prob[index:].sum())
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))
    if index == 0:
        estimate, log_odds, boundary = 0.0, None, "zero"
        fit_prob = torch.zeros_like(base)
        fit_prob[0] = 1
    elif index == len(base) - 1:
        estimate, log_odds, boundary = None, None, "positive_infinity"
        fit_prob = torch.zeros_like(base)
        fit_prob[-1] = 1
    else:
        log_odds = bisect(mean, observed, True)
        estimate, boundary = math.exp(log_odds), "interior"
        fit_prob = probabilities(log_odds)
    alpha = (1 - level) / 2
    low_log = (
        None
        if index == 0
        else bisect(lambda eta: float(probabilities(eta)[index:].sum()), alpha, True)
    )
    high_log = (
        None
        if index == len(base) - 1
        else bisect(lambda eta: float(probabilities(eta)[: index + 1].sum()), alpha, False)
    )
    ci_low = 0.0 if low_log is None else math.exp(low_log)
    ci_high = None if high_log is None else math.exp(high_log)
    state = {
        "strata": validated,
        "stratum_order": list(range(len(validated))),
        "support": support.tolist(),
        "log_base": base.tolist(),
        "observed_statistic": observed,
        "null_odds": null_odds,
        "null_probabilities": null_prob.tolist(),
        "fit_probabilities": fit_prob.tolist(),
        "log_odds_cmle": log_odds,
        "estimate_boundary": boundary,
        "ci_log_low": low_log,
        "ci_log_high": high_log,
        "ci_high_boundary": "positive_infinity" if high_log is None else "finite",
        "level": level,
        "settings": settings,
        "tail_convention": "inclusive doubled smaller tail; equal-tailed CI",
    }
    return c.seal(
        "exact_logistic",
        {
            "inference": table(
                [
                    [
                        estimate,
                        boundary,
                        ci_low,
                        ci_high,
                        state["ci_high_boundary"],
                        p_value,
                        lower_tail,
                        upper_tail,
                        observed,
                    ]
                ],
                columns=[
                    "odds_cmle",
                    "estimate_boundary",
                    "ci_low",
                    "ci_high",
                    "ci_high_boundary",
                    "p_value",
                    "lower_tail",
                    "upper_tail",
                    "statistic",
                ],
            ),
            "support": table(
                [
                    [int(s), float(b), float(p), float(f)]
                    for s, b, p, f in zip(support, base, null_prob, fit_prob)
                ],
                columns=["statistic", "log_base", "null_probability", "fit_probability"],
            ),
            "strata": table(
                [
                    [
                        i,
                        *[v for r in item["cells"] for v in r],
                        item["support_low"],
                        item["support_high"],
                    ]
                    for i, item in enumerate(validated)
                ],
                columns=["stratum", "a", "b", "c", "d", "support_low", "support_high"],
            ),
        },
        state,
        inference="conditional exact fixed-margin common odds; inclusive equal tails; no MUE substitution",
        **settings,
    )


@resident_cpu
def exact_poisson_rate(
    events, exposure, *, null_rate=1.0, level=0.95, max_work=100_000_000, device="cpu", weights=None
):
    """Garwood exact Poisson rate CI and inclusive central test for known fixed exposure.

    Integer events <=10000, positive exposure and null mean <=10000. Zero-event
    lower bound is zero; the central upper bound has one-sided (1+level)/2
    coverage. Plugin variance is descriptive, not a Wald interval.
    """
    c.domain(device, weights)
    level = c.confidence(level)
    events = count(events)
    exposure = c.c.check_number(exposure, "exposure", minimum=1e-12, maximum=1e12)
    null_rate = c.c.check_number(null_rate, "null_rate", minimum=1e-12, maximum=1e12)
    mean = null_rate * exposure
    if mean > 10000:
        raise AnalysisError("resource_limit", "Exact Poisson null mean must be <=10000.")
    settings = c.guard(
        "exact_poisson_rate", events + 1, 1, work=256 * (events + 1), max_work=max_work
    )
    tail = (1 - level) / 2
    low = 0.0 if events == 0 else 0.5 * dist.chi2_ppf(tail, 2 * events) / exposure
    high = 0.5 * dist.chi2_isf(tail, 2 * (events + 1)) / exposure
    lower_tail = dist.chi2_sf(2 * mean, 2 * (events + 1))
    upper_tail = 1.0 if events == 0 else dist.chi2_cdf(2 * mean, 2 * events)
    estimate, plugin_variance = events / exposure, events / exposure**2
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))
    state = {
        "events": events,
        "exposure": exposure,
        "null_rate": null_rate,
        "null_mean": mean,
        "level": level,
        "lower_tail": lower_tail,
        "upper_tail": upper_tail,
        "settings": settings,
        "variance_kind": "descriptive MLE plugin K/exposure^2; exact inference uses tails",
    }
    return c.seal(
        "exact_poisson_rate",
        {
            "inference": table(
                [
                    [
                        events,
                        exposure,
                        estimate,
                        plugin_variance,
                        low,
                        high,
                        p_value,
                        lower_tail,
                        upper_tail,
                    ]
                ],
                columns=[
                    "events",
                    "exposure",
                    "rate",
                    "plugin_variance",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "lower_tail",
                    "upper_tail",
                ],
            ),
        },
        state,
        inference="Garwood fixed-exposure exact central Poisson tails",
        **settings,
    )
