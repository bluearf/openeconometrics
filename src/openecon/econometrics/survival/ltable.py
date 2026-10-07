"""Actuarial life tables: Stata's ``ltable``, SPSS ``SURVIVAL``.

Formulas (Stata [ST] ltable, Methods and formulas; SPSS SURVIVAL algorithms)
-----------------------------------------------------------------------------
Times ``tau_i`` are grouped into intervals ``[t_j, t_{j+1})``. With ``N_j``
alive at the start of interval j, ``d_j`` failures and ``m_j`` censored
(withdrawn) records in it, and width ``w_j = t_{j+1} - t_j``:

* effective number at risk ``n_j = N_j - m_j / 2`` (``noadjust``: ``n_j = N_j``);
* conditional probability of failure ``q_j = d_j / n_j``, ``p_j = 1 - q_j``;
* cumulative survival at the end of the interval ``S_j = prod_{k <= j} p_k`` with
  Greenwood's standard error ``S_j sqrt(sum_{k <= j} d_k / (n_k (n_k - d_k)))`` and
  the interval ``S_j^{exp(+-z s_j)}``,
  ``s_j = sqrt(sum d/(n(n-d))) / |sum ln((n-d)/n)|`` (the ln(-ln S) transform);
* hazard ``lambda_j = q_j / ((1 - q_j/2) w_j)`` with
  ``SE = lambda_j sqrt((1 - (w_j lambda_j / 2)^2) / d_j)`` and a normal interval
  (``noadjust``: ``lambda_j = q_j / w_j``, ``SE = lambda_j / sqrt(d_j)``, interval
  ``lambda_j chi2_{2 d_j, (alpha/2, 1-alpha/2)} / (2 d_j)``);
* probability density ``f_j = S_{j-1} q_j / w_j`` with
  ``SE = f_j sqrt(sum_{k < j} q_k / (n_k p_k) + p_j / (n_j q_j))`` (SPSS; Gehan 1969).

Tests of homogeneity across ``by`` groups (Stata's ``ltable, test``): the
likelihood-ratio statistic of Lawless (2003, p. 155) for exponential
distributions, ``2 {(sum d_g) ln(sum T_g / sum d_g) - sum d_g ln(T_g / d_g)}`` with
``T_g`` the total time of group g, on G - 1 degrees of freedom, and the log-rank
test of ``sts test`` on the individual times.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import check_alpha, procedure
from openecon.econometrics.survival.data import FLOAT
from openecon.econometrics.survival.km import (
    check_group_count, combined_codes, names_of, rank_tests,
)
from openecon.engines.distributions import chi2_ppf, chi2_sf, normal_isf

INTERVAL_LIMIT = 10_000


def _cutpoints(intervals: Any, largest: float) -> list[float]:
    """Left edges of the intervals (the last one may be open-ended)."""
    if intervals is None:
        intervals = 1.0
    if isinstance(intervals, (int, float)) and not isinstance(intervals, bool):
        width = float(intervals)
        if not math.isfinite(width) or width <= 0:
            raise AnalysisError("invalid_option", "intervals must be a positive width or an "
                                "increasing list of cutpoints.")
        count = int(math.floor(largest / width)) + 2
        if count > INTERVAL_LIMIT:
            raise AnalysisError("too_many_intervals", f"Width {width:g} creates {count:,} "
                                f"intervals (limit {INTERVAL_LIMIT:,}); choose a wider width.")
        return [j * width for j in range(count)]
    if not isinstance(intervals, (list, tuple)) or not intervals or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in intervals):
        raise AnalysisError("invalid_option", "intervals must be a positive width or an "
                            "increasing list of cutpoints.")
    edges = [float(v) for v in intervals]
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)) or edges[0] < 0:
        raise AnalysisError("invalid_option", "Interval cutpoints must be nonnegative and "
                            "strictly increasing.")
    if len(edges) > INTERVAL_LIMIT:
        raise AnalysisError("too_many_intervals", f"At most {INTERVAL_LIMIT:,} cutpoints.")
    return ([0.0] + edges) if edges[0] > 0 else edges


def _group_table(t: torch.Tensor, d: torch.Tensor, w: torch.Tensor, edges: torch.Tensor,
                 open_last: bool, adjust: bool, z: float, alpha: float) -> list[dict[str, Any]]:
    bins = torch.bucketize(t, edges, right=True) - 1           # interval index of each time
    count = edges.numel()
    deaths = torch.zeros(count, dtype=FLOAT).index_add_(0, bins, w * d)
    lost = torch.zeros(count, dtype=FLOAT).index_add_(0, bins, w * (1 - d))
    leaving = deaths + lost
    entering = float(w.sum()) - torch.cumsum(leaving, 0) + leaving
    at_risk = entering - lost / 2 if adjust else entering
    q = torch.where(at_risk > 0, deaths / at_risk.clamp_min(1e-300), torch.zeros_like(at_risk))
    p = 1 - q
    survival = torch.cumprod(p, 0)
    alive = at_risk - deaths
    green = torch.cumsum(torch.where(alive > 0, deaths / (at_risk * alive.clamp_min(1e-300)),
                                     torch.zeros_like(alive)), 0)
    logs = torch.cumsum(torch.where(alive > 0, torch.log(alive.clamp_min(1e-300)
                                                         / at_risk.clamp_min(1e-300)),
                                    torch.zeros_like(alive)), 0)
    widths = torch.empty(count, dtype=FLOAT)
    widths[:-1] = edges[1:] - edges[:-1]
    widths[-1] = float("inf") if open_last else 1.0
    before = torch.ones_like(survival)
    before[1:] = survival[:-1]
    density_terms = torch.cumsum(torch.where(p > 0, q / (at_risk.clamp_min(1e-300)
                                                         * p.clamp_min(1e-300)),
                                             torch.zeros_like(q)), 0)
    rows = []
    for j in range(count):
        if float(leaving[j]) == 0:
            continue
        s, n_j, d_j, w_j = float(survival[j]), float(at_risk[j]), float(deaths[j]), float(widths[j])
        se = s * math.sqrt(float(green[j])) if s > 0 else None
        low = high = None
        if 0 < s < 1 and float(logs[j]) < 0:
            sigma = math.sqrt(float(green[j])) / abs(float(logs[j]))
            low, high = s ** math.exp(z * sigma), s ** math.exp(-z * sigma)
        hazard = hazard_se = hazard_low = hazard_high = density = density_se = None
        last_open = open_last and j == count - 1
        if not last_open:
            q_j = float(q[j])
            hazard = q_j / ((1 - q_j / 2) * w_j) if adjust else q_j / w_j
            if d_j > 0:
                if adjust:
                    hazard_se = hazard * math.sqrt(max(1 - (w_j * hazard / 2) ** 2, 0.0) / d_j)
                    hazard_low, hazard_high = max(hazard - z * hazard_se, 0.0), \
                        hazard + z * hazard_se
                else:
                    hazard_se = hazard / math.sqrt(d_j)
                    hazard_low = hazard * chi2_ppf(alpha / 2, 2 * d_j) / (2 * d_j)
                    hazard_high = hazard * chi2_ppf(1 - alpha / 2, 2 * d_j) / (2 * d_j)
            density = float(before[j]) * q_j / w_j
            if 0 < q_j < 1:
                previous = float(density_terms[j - 1]) if j > 0 else 0.0
                density_se = density * math.sqrt(previous + float(p[j]) / (n_j * q_j))
        rows.append({
            "interval_start": float(edges[j]),
            "interval_end": None if last_open else float(edges[j]) + w_j,
            "entering": float(entering[j]), "deaths": d_j, "lost": float(lost[j]),
            "at_risk": n_j, "death_probability": float(q[j]), "survival": s, "std_error": se,
            "ci_low": low, "ci_high": high, "hazard": hazard, "hazard_std_error": hazard_se,
            "hazard_ci_low": hazard_low, "hazard_ci_high": hazard_high, "density": density,
            "density_std_error": density_se})
    return rows


@procedure
def ltable(data: Any, time: str, *, failure: str | None = None,
           by: str | Sequence[str] | None = None, intervals: Any = None,
           weights: str | None = None, alpha: float = 0.05, noadjust: bool = False,
           missing: str = "drop") -> TableSet:
    """Actuarial life table with survival, hazard and density estimates (Stata ``ltable``).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    time : failure or censoring time (>= 0; 0 is allowed, it falls in the first interval).
    failure : 0/1 failure indicator (default: every record fails).
    by : column(s) defining groups with separate tables and the homogeneity tests.
    intervals : None (unit-width intervals, Stata's default), a width, or a list of
        increasing cutpoints (0 is prepended; times beyond the last cutpoint form an
        open-ended interval whose hazard and density are not defined).
    weights : frequency weights (nonnegative integers).
    alpha : 1 - confidence level.
    noadjust : use the number entering instead of ``N - lost/2`` as the number at risk.
    missing : "drop" (listwise deletion) or "raise".

    Returns
    -------
    TableSet with ``table`` (per group and interval: interval_start, interval_end,
    entering, deaths, lost, at_risk (effective), death_probability, survival,
    std_error, ci_low, ci_high, hazard with std_error and interval, density with
    std_error; intervals without deaths or losses are omitted) and, with ``by``,
    ``tests`` (likelihood-ratio and log-rank tests of homogeneity: statistic, df,
    p_value). ``attrs``: n, n_dropped, intervals, adjust and the LR test.

    Stata: ``ltable t died, by(drug) intervals(5) hazard test``. SPSS: ``SURVIVAL TABLE=t
    BY drug(1 2) /INTERVAL=THRU 40 BY 5 /STATUS=died(1) /PRINT=TABLE /COMPARE``.

    Example
    -------
    >>> import openecon as oe
    >>> df = {"t": [1, 2, 2, 3, 5, 6, 8, 9, 12, 15], "d": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]}
    >>> out = oe.ltable(df, "t", failure="d", intervals=5)
    >>> list(out["table"].deaths)
    [3.0, 3.0, 0.0, 1.0]
    """
    from openecon.econometrics.survival.km import survival_sample

    alpha = check_alpha(alpha)
    if not isinstance(noadjust, bool):
        raise AnalysisError("invalid_option", "noadjust must be True or False.")
    by_columns = names_of(by, "by")
    notes: list[str] = []
    frame, tensors, dropped = survival_sample(data, time, failure, None, weights, by_columns,
                                              missing, notes, allow_zero=True)
    t, d, w = tensors["time"], tensors["failure"], tensors["weights"]
    edges = _cutpoints(intervals, float(t.max()))
    open_last = not (intervals is None or isinstance(intervals, (int, float)))
    edge_tensor = torch.tensor(edges, dtype=FLOAT)
    groups, labels = combined_codes(frame, by_columns)
    check_group_count(len(labels), by_columns)
    z = normal_isf(alpha / 2)
    rows = []
    for g, group_label in enumerate(labels):
        member = groups == g
        for row in _group_table(t[member], d[member], w[member], edge_tensor, open_last,
                                not noadjust, z, alpha):
            rows.append({"group": group_label, **row} if by_columns else row)
    tables = {"table": table(pd.DataFrame(rows))}
    attrs: dict[str, Any] = {"n": float(w.sum()), "n_dropped": dropped,
                             "intervals": edges if open_last else None,
                             "width": None if open_last else edges[1] - edges[0],
                             "adjust": not noadjust, "alpha": alpha, "notes": notes}
    if by_columns:
        if len(labels) < 2:
            raise AnalysisError("invalid_groups", "The homogeneity tests need at least two "
                                "groups in by.")
        deaths = torch.zeros(len(labels), dtype=FLOAT).index_add_(0, groups, w * d)
        total = torch.zeros(len(labels), dtype=FLOAT).index_add_(0, groups, w * t)
        pieces = torch.where(deaths > 0, deaths * torch.log(total / deaths.clamp_min(1e-300)),
                             torch.zeros_like(deaths))
        all_d, all_t = float(deaths.sum()), float(total.sum())
        if all_d == 0:
            raise AnalysisError("no_failures", "The homogeneity tests need failures.")
        lr = max(2 * (all_d * math.log(all_t / all_d) - float(pieces.sum())), 0.0)
        df = len(labels) - 1
        positive = t > 0
        logrank = rank_tests(t[positive], None, d[positive], w[positive], groups[positive],
                             len(labels), torch.zeros(int(positive.sum()), dtype=torch.int64),
                             ["logrank"], (0.0, 0.0), None)["logrank"]
        tables["tests"] = table(
            [["Likelihood-ratio (exponential homogeneity)", lr, df, chi2_sf(lr, df)],
             ["Log-rank", logrank["statistic"], logrank["df"], logrank["p_value"]]],
            columns=["test", "statistic", "df", "p_value"], index=["likelihood_ratio",
                                                                    "logrank"])
        attrs.update({"groups": labels, "statistic": lr, "df": df, "p_value": chi2_sf(lr, df),
                      "distribution": "chi2"})
    return TableSet(tables, title=f"Life table: {time}", **attrs)
