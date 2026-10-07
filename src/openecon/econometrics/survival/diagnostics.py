"""Post-estimation diagnostics of the Cox model: PH tests, Harrell's C, baseline functions.

Proportional-hazards test (Stata's ``estat phtest``; Grambsch and Therneau 1994)
------------------------------------------------------------------------------------
With Schoenfeld residuals ``r_i`` of the failing records, a time function
``g`` (``identity``: analysis time, Stata's default; ``log``; ``km``: 1 minus the
Kaplan-Meier estimate at the failure time; ``rank``: rank of the failure time),
``gbar`` its mean over failures, ``d`` the number of failures, ``V`` the
estimated covariance of b and ``u = sum_i (g_i - gbar) r_i``:

    covariate p:  chi2(1) = d (V u)_p^2 / (V_pp sum_i (g_i - gbar)^2)
    global:       chi2(k) = u' V u d / sum_i (g_i - gbar)^2

which are Stata's formulas written with the scaled Schoenfeld residuals
``r*_i = b + d V r_i``; ``rho`` is the correlation of ``r*_p`` with g.

Harrell's C (Stata's ``estat concordance``)
-------------------------------------------
A pair is usable when the shorter time ends in failure, or when both times are
equal and exactly one of them fails; it is concordant when the subject failing
first has the larger linear predictor. ``C = (E + T/2) / D`` with ties ``T`` in
the prediction, summed over strata. All pairs are counted without enumerating
them: for every failure, the later records with a smaller linear predictor are
counted bit by bit of the predictor's rank (at the highest bit where two ranks
differ, the smaller one has a 0 and the larger a 1), each level being one sort
and two ``searchsorted`` calls: O(n log^2 n) and no n-by-n object.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.econometrics.survival.data import FLOAT, RiskSets, kaplan_meier
from openecon.engines.distributions import chi2_sf

PH_TRANSFORMS = ("identity", "log", "km", "rank")
BASELINE_LIMIT = 400


def average_ranks(values: Tensor) -> Tensor:
    """Ranks 1..n with ties given the mean of the ranks they span."""
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    _, inverse, counts = torch.unique_consecutive(ordered, return_inverse=True,
                                                  return_counts=True)
    ends = torch.cumsum(counts, 0).to(FLOAT)
    mean_rank = ends - (counts.to(FLOAT) - 1) / 2
    ranks = torch.empty_like(values)
    ranks[order] = mean_rank[inverse]
    return ranks


def time_function(kind: str, times: Tensor, km_time: Tensor, km_entry: Tensor | None,
                  km_failure: Tensor, km_weights: Tensor) -> Tensor:
    """g(t) of the failing records for the PH test."""
    if kind == "identity":
        return times
    if kind == "log":
        return torch.log(times)
    if kind == "rank":
        return average_ranks(times)
    risk = RiskSets.build(km_time, km_entry, km_failure)
    survivor = kaplan_meier(risk.at_risk(km_weights), risk.at_event(km_weights * km_failure),
                            risk.event_stratum)
    position = torch.searchsorted(risk.event_time, times)
    return 1.0 - survivor[position]


def ph_tests(schoenfeld: Tensor, g: Tensor, weights: Tensor, covariance: Tensor, beta: Tensor,
             terms: list[str], transform: str) -> dict[str, Any]:
    """Grambsch-Therneau tests for every covariate and the global test."""
    d = float(weights.sum())
    centred = g - float((weights * g).sum()) / d
    spread = float((weights * centred.square()).sum())
    tests: dict[str, Any] = {}
    if spread <= 0:
        return {"ph_global": {"statistic": None, "df": len(terms), "p_value": None,
                              "distribution": "chi2", "label": "PH test: g(t) does not vary "
                              "over the failures", "time_function": transform}}
    u = (schoenfeld * (weights * centred)[:, None]).sum(0)
    vu = covariance @ u
    scaled = beta + d * schoenfeld @ covariance                       # scaled Schoenfeld
    for p, term in enumerate(terms):
        statistic = d * float(vu[p]) ** 2 / (float(covariance[p, p]) * spread)
        column = scaled[:, p]
        mean = float((weights * column).sum()) / d
        dev = column - mean
        denom = (float((weights * dev.square()).sum()) * spread) ** 0.5
        rho = float((weights * dev * centred).sum()) / denom if denom > 0 else None
        tests[f"ph_{term}"] = {"statistic": statistic, "df": 1, "p_value": chi2_sf(statistic, 1),
                               "distribution": "chi2", "rho": rho, "time_function": transform,
                               "label": f"PH test (scaled Schoenfeld residuals) for {term}"}
    statistic = float(u @ vu) * d / spread
    tests["ph_global"] = {"statistic": statistic, "df": len(terms),
                          "p_value": chi2_sf(statistic, len(terms)), "distribution": "chi2",
                          "time_function": transform,
                          "label": "Global test of proportional hazards (Grambsch-Therneau)"}
    return tests


def harrell_c(eta: Tensor, time: Tensor, failure: Tensor, strata: Tensor | None) -> dict[str, Any]:
    """Harrell's C over all usable pairs within strata (see the module notes)."""
    n = eta.numel()
    _, time_rank = torch.unique(time, sorted=True, return_inverse=True)
    key = 2 * time_rank + (failure <= 0).to(torch.int64)       # censored ties count as later
    span = 2 * int(time_rank.max()) + 4
    stratum = torch.zeros(n, dtype=torch.int64) if strata is None else strata
    position = stratum * span + key
    end = (stratum + 1) * span
    fail = failure > 0
    _, rank = torch.unique(eta, sorted=True, return_inverse=True)
    width = (int(stratum.max()) + 1) * span
    if width * (int(rank.max()) + 2) >= 2 ** 62:
        return {"concordance": None, "note": "too many distinct values for exact counting"}
    pos_f, end_f, rank_f = position[fail], end[fail], rank[fail]
    sorted_position = torch.sort(position).values
    usable = (torch.searchsorted(sorted_position, end_f)
              - torch.searchsorted(sorted_position, pos_f, right=True))
    tie_keys = torch.sort(rank * width + position).values
    ties = (torch.searchsorted(tie_keys, rank_f * width + end_f)
            - torch.searchsorted(tie_keys, rank_f * width + pos_f, right=True))
    concordant = torch.zeros((), dtype=torch.int64)
    for bit in range(max(1, int(rank.max()).bit_length())):
        prefix, flag = rank >> (bit + 1), (rank >> bit) & 1
        zero = flag == 0
        keys = torch.sort(prefix[zero] * width + position[zero]).values
        query = (flag[fail] == 1)
        head = prefix[fail][query] * width
        concordant += (torch.searchsorted(keys, head + end_f[query])
                       - torch.searchsorted(keys, head + pos_f[query], right=True)).sum()
    pairs, tied, agree = int(usable.sum()), int(ties.sum()), int(concordant)
    value = (agree + tied / 2) / pairs if pairs else None
    return {"concordance": value, "somers_d": None if value is None else 2 * value - 1,
            "pairs": pairs, "concordant": agree, "tied_predictions": tied,
            "discordant": pairs - agree - tied}


def baseline_table(risk: RiskSets, increments: Tensor, log_alpha: Tensor, *,
                   labels: list[Any] | None = None,
                   limit: int | None = BASELINE_LIMIT) -> dict[str, list[Any]]:
    """Baseline functions at the failure times (per stratum), at covariates and offset zero.

    ``increments`` are the cumulative-hazard jumps (Breslow ``d_k / S0_k``; the Efron
    analogue after ``ties='efron'``) and ``log_alpha`` the Kalbfleisch-Prentice factors
    (``CoxObjective.baseline``). Columns: ``cumulative_hazard`` ``H0(t)`` (Stata's
    ``predict basechazard``), ``survivor`` ``exp(-H0(t))`` and ``survivor_kp``
    ``prod_{tau_k <= t} alpha_k`` (Stata's ``predict basesurv``). ``limit`` thins the rows
    evenly; ``labels`` name the strata (codes are reported otherwise).
    """
    from openecon.econometrics.survival.data import segmented_cumsum

    hazard = segmented_cumsum(increments, risk.event_stratum)
    # -inf factors (everyone at risk fails) make the product 0 from there on
    dead = segmented_cumsum((log_alpha == -math.inf).to(FLOAT), risk.event_stratum) > 0
    finite = torch.where(log_alpha == -math.inf, torch.zeros_like(log_alpha), log_alpha)
    kp = torch.exp(segmented_cumsum(finite, risk.event_stratum))
    kp = torch.where(dead, torch.zeros_like(kp), kp)
    rows = torch.arange(risk.n_events)
    if limit is not None and risk.n_events > limit:
        rows = torch.linspace(0, risk.n_events - 1, limit, dtype=FLOAT).round().to(torch.int64)
    strata = risk.event_stratum[rows].tolist()
    if labels is not None:
        strata = [labels[code] for code in strata]
    return {"stratum": strata, "time": risk.event_time[rows].tolist(),
            "cumulative_hazard": hazard[rows].tolist(),
            "survivor": torch.exp(-hazard[rows]).tolist(),
            "survivor_kp": kp[rows].tolist(),
            "thinned": bool(limit is not None and risk.n_events > limit),
            "n_times": risk.n_events}
