"""Kaplan-Meier and Nelson-Aalen estimates and the log-rank family: Stata's ``sts``.

``oe.sts`` combines ``sts list`` (survivor and cumulative hazard functions),
``stsum`` / ``stci`` (percentiles with confidence intervals, restricted mean)
and ``sts test`` (log-rank, Wilcoxon-Breslow-Gehan, Tarone-Ware,
Peto-Peto-Prentice and Fleming-Harrington tests, stratified versions and the
trend test). SPSS: ``KM time BY group /STATUS=failure(1) /TEST=LOGRANK BRESLOW
TARONE``.

Formulas (Stata [ST] sts, sts test, stci, Methods and formulas)
------------------------------------------------------------------
With ``n_j`` at risk just before the failure time ``t_j`` and ``d_j`` failures
at ``t_j`` (frequency-weighted counts):

* Kaplan-Meier ``S(t) = prod_{t_j <= t} (n_j - d_j) / n_j``; Greenwood
  ``Var S(t) = S(t)^2 sum_{t_j <= t} d_j / (n_j (n_j - d_j))``.
* Confidence intervals (``conftype``): ``loglog`` (Stata's default)
  ``S^{exp(+-z sigma)}`` with ``sigma^2 = sum d/(n(n-d)) / (sum ln((n-d)/n))^2``;
  ``log`` ``S exp(+-z sqrt(sum d/(n(n-d))))`` capped at 1; ``plain``
  ``S +- z se`` clipped to [0, 1] (SPSS's interval).
* Nelson-Aalen ``H(t) = sum d_j / n_j`` with ``Var H = sum d_j / n_j^2`` and the
  interval ``H exp(+-z sqrt(Var H) / H)``.
* Percentile p: the smallest t with ``S(t) <= 1 - p``; its confidence limits
  are the first times at which the lower / upper pointwise limit of S falls to
  ``1 - p`` or below (Brookmeyer-Crowley inversion, as ``stci`` does it).
* Restricted mean: the area under S from 0 to the largest observed time, with
  ``SE^2 = sum_j A_j^2 d_j / (n_j (n_j - d_j))``, ``A_j`` the area from ``t_j``
  to the largest time (``stci, rmean``).
* Tests: ``u_g = sum_j W_j (d_gj - n_gj d_j / n_j)`` and
  ``V_gl = sum_j W_j^2 n_gj d_j (n_j - d_j) / (n_j (n_j - 1)) (1{g=l} - n_lj / n_j)``,
  ``chi2 = u' V^- u`` on r - 1 degrees of freedom; weights ``W_j``: 1 (log-rank),
  ``n_j`` (Wilcoxon-Breslow-Gehan), ``sqrt(n_j)`` (Tarone-Ware),
  ``prod_{l <= j} (1 - d_l / (n_l + 1))`` (Peto-Peto-Prentice),
  ``S(t_{j-1})^p (1 - S(t_{j-1}))^q`` (Fleming-Harrington, pooled KM). Strata:
  u and V are summed over strata. Trend: ``(a'u)^2 / a'Va`` on 1 df with the
  group values as scores ``a``.

All risk-set counts come from one sort per call (``data.RiskSets``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.nonparametric.common import (
    check_alpha, check_choice, group_codes, numeric, procedure, select,
)
from openecon.econometrics.survival.data import (
    FLOAT, RiskSets, check_times, end_before_entry_note, failure_indicator, kaplan_meier,
    segmented_cumsum,
)
from openecon.engines.distributions import chi2_sf, normal_isf
from openecon.engines.linalg import wald_statistic

# Rows of the survival table per group before it is thinned evenly.
TABLE_LIMIT = 2000
# Most groups in ``by`` (one curve, summary row and test column each).
GROUP_LIMIT = 1000
TESTS = ("logrank", "wilcoxon", "tware", "peto", "fh")
SURVIVAL_COLUMNS = ("time", "n_risk", "n_event", "n_censored", "survivor", "std_error", "ci_low",
                    "ci_high", "cumulative_hazard", "cumulative_hazard_std_error",
                    "cumulative_hazard_ci_low", "cumulative_hazard_ci_high")
_TEST_LABELS = {
    "logrank": "Log-rank (Mantel-Haenszel)", "wilcoxon": "Wilcoxon-Breslow-Gehan",
    "tware": "Tarone-Ware", "peto": "Peto-Peto-Prentice", "fh": "Fleming-Harrington",
}


# ---- the sample -----------------------------------------------------------------------


def names_of(value: Any, what: str) -> list[str]:
    """One column name or a list of names (``by``, ``strata``)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    raise AnalysisError("invalid_spec", f"{what} must be a column name or a list of names.")


def survival_sample(data: Any, time: str, failure: str | None, entry: str | None,
                    weights: str | None, groups: Sequence[str], missing: str,
                    notes: list[str], allow_zero: bool = False
                    ) -> tuple[pd.DataFrame, dict[str, Tensor], int]:
    """Complete, usable rows and their survival tensors (time, entry, failure, weights).

    ``allow_zero`` keeps records with time 0 (life tables group them into the first
    interval; the st-style estimators exclude them like stset).
    """
    if not isinstance(time, str):
        raise AnalysisError("invalid_spec", "time must be a column name.")
    names = [time, *(n for n in (failure, entry, weights) if n is not None), *groups]
    frame, dropped = select(data, names, missing=missing)
    t = numeric(frame, time)
    t0 = numeric(frame, entry) if entry is not None else None
    delta = failure_indicator(numeric(frame, failure) if failure else None, len(frame), failure)
    w = torch.ones(len(frame), dtype=FLOAT)
    if weights is not None:
        w = numeric(frame, weights)
        if bool((w < 0).any()) or bool((w != w.round()).any()):
            raise AnalysisError("noninteger_frequency_weights",
                                f"weights '{weights}' are frequency weights and must be "
                                "nonnegative integers.")
    usable = check_times(t, t0, time, entry)
    if allow_zero and t0 is None:
        usable = t >= 0
    ended = int((~usable).sum())
    if ended:
        notes.append(end_before_entry_note(ended))
    if weights is not None:
        zero = (w == 0) & usable
        if bool(zero.any()):
            notes.append(f"Excluded {int(zero.sum())} observation(s) with zero weight.")
        usable &= ~zero
    if not bool(usable.any()):
        raise AnalysisError("empty_sample", "No usable survival records remain (every record "
                            "ends at time 0 or before its entry time).")
    if not bool(usable.all()):
        keep = usable.numpy()
        frame = frame.loc[keep].reset_index(drop=True)
        t, delta, w = t[usable], delta[usable], w[usable]
        t0 = t0[usable] if t0 is not None else None
    tensors = {"time": t, "entry": t0, "failure": delta, "weights": w}
    return frame, tensors, dropped


def check_group_count(count: int, columns: Sequence[str]) -> None:
    """Refuse a ``by`` with more than GROUP_LIMIT groups (a continuous variable, an id)."""
    if count > GROUP_LIMIT:
        raise AnalysisError("too_many_groups", f"by={list(columns)} defines {count:,} groups "
                            f"(limit {GROUP_LIMIT:,}): one curve and one test column per group. "
                            "Group a continuous variable into categories first.")


def combined_codes(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[Tensor, list[Any]]:
    """Codes of the distinct value combinations of ``columns`` and their labels."""
    if not columns:
        return torch.zeros(len(frame), dtype=torch.int64), [None]
    if len(columns) == 1:
        return group_codes(frame[columns[0]])
    parts = [group_codes(frame[name]) for name in columns]
    sizes = [len(levels) for _, levels in parts]
    key = torch.zeros(len(frame), dtype=torch.int64)
    for (codes, _), size in zip(parts, sizes, strict=True):
        key = key * size + codes
    unique, codes = torch.unique(key, sorted=True, return_inverse=True)
    labels = []
    for value in unique.tolist():
        pieces = []
        for (codes_, levels), size in zip(reversed(parts), reversed(sizes), strict=True):
            pieces.append(levels[value % size])
            value //= size
        labels.append(", ".join(f"{name}={level}" for name, level
                                in zip(columns, reversed(pieces), strict=True)))
    return codes, labels


# ---- Kaplan-Meier and Nelson-Aalen ---------------------------------------------------------


def confidence_band(survivor: Tensor, greenwood_sum: Tensor, log_sum: Tensor, z: float,
                    conftype: str) -> tuple[Tensor, Tensor]:
    """Pointwise limits of S (NaN where undefined: S = 1 before any failure, or S = 0)."""
    nan = torch.full_like(survivor, float("nan"))
    if conftype == "loglog":
        inside = (survivor > 0) & (survivor < 1) & (log_sum < 0)
        sigma = torch.where(inside, greenwood_sum.sqrt() / log_sum.abs().clamp_min(1e-300), nan)
        low = torch.where(inside, survivor.clamp_min(1e-300).pow(torch.exp(z * sigma)), nan)
        high = torch.where(inside, survivor.clamp_min(1e-300).pow(torch.exp(-z * sigma)), nan)
        return low, high
    spread = greenwood_sum.sqrt()
    if conftype == "log":
        inside = survivor > 0
        low = torch.where(inside, survivor * torch.exp(-z * spread), nan)
        high = torch.where(inside, (survivor * torch.exp(z * spread)).clamp(max=1.0), nan)
        return low, high
    se = survivor * spread
    finite = torch.isfinite(se)
    return (torch.where(finite, (survivor - z * se).clamp(0.0, 1.0), nan),
            torch.where(finite, (survivor + z * se).clamp(0.0, 1.0), nan))


def product_limit(t: Tensor, t0: Tensor | None, delta: Tensor, w: Tensor, groups: Tensor,
                  z: float, conftype: str) -> dict[str, Tensor]:
    """KM / Nelson-Aalen at every distinct exit time of every group."""
    risk = RiskSets.build(t, t0, torch.ones_like(t), groups)
    n_risk = risk.at_risk(w)
    events = risk.at_event(w * delta)
    censored = risk.at_event(w * (1.0 - delta))
    segments = risk.event_stratum
    survivor = kaplan_meier(n_risk, events, segments)
    alive = n_risk - events
    # Where everyone at risk fails S drops to 0 and its SE is undefined (reported as
    # missing); the increment is left out so that later strata's sums stay finite.
    increment = torch.where(alive > 0, events / (n_risk * alive.clamp_min(1e-300)),
                            torch.zeros_like(events))
    greenwood = segmented_cumsum(increment, segments)
    std_error = survivor * greenwood.sqrt()
    std_error = torch.where(survivor > 0, std_error, torch.full_like(std_error, float("nan")))
    logs = torch.where(alive > 0, torch.log(alive.clamp_min(1e-300) / n_risk),
                       torch.zeros_like(n_risk))
    log_sum = segmented_cumsum(logs, segments)
    low, high = confidence_band(survivor, greenwood, log_sum, z, conftype)
    hazard = segmented_cumsum(events / n_risk, segments)
    hazard_var = segmented_cumsum(events / n_risk.square(), segments)
    hazard_se = hazard_var.sqrt()
    positive = hazard > 0
    factor = torch.exp(z * hazard_se / hazard.clamp_min(1e-300))
    nan = torch.full_like(hazard, float("nan"))
    return {"time": risk.event_time, "group": segments, "n_risk": n_risk, "n_event": events,
            "n_censored": censored, "survivor": survivor, "std_error": std_error,
            "ci_low": low, "ci_high": high, "cumulative_hazard": hazard,
            "cumulative_hazard_std_error": hazard_se,
            "cumulative_hazard_ci_low": torch.where(positive, hazard / factor, nan),
            "cumulative_hazard_ci_high": torch.where(positive, hazard * factor, nan),
            "greenwood": greenwood}


def _first_time(times: Tensor, values: Tensor, level: float) -> float | None:
    hit = (values <= level + 1e-12).nonzero()
    return float(times[hit[0, 0]]) if hit.numel() else None


def group_summary(curve: dict[str, Tensor], rows: Tensor, t: Tensor, t0: Tensor | None,
                  delta: Tensor, w: Tensor, member: Tensor, z: float) -> dict[str, Any]:
    """stsum / stci statistics of one group (rows: its positions in the curve)."""
    times = curve["time"][rows]
    survivor = curve["survivor"][rows]
    low = torch.nan_to_num(curve["ci_low"][rows], nan=1.0)
    high = torch.nan_to_num(curve["ci_high"][rows], nan=1.0)
    low = torch.where(survivor <= 0, torch.zeros_like(low), low)
    high = torch.where(survivor <= 0, torch.zeros_like(high), high)
    failed = curve["n_event"][rows] > 0
    record: dict[str, Any] = {
        "n": float(w[member].sum()), "events": float((w * delta)[member].sum()),
        "time_at_risk": float((w * (t - (0.0 if t0 is None else t0)))[member].sum()),
    }
    for name, p in (("q25", 0.25), ("median", 0.5), ("q75", 0.75)):
        record[name] = _first_time(times[failed], survivor[failed], 1.0 - p)
    record["median_ci_low"] = _first_time(times[failed], low[failed], 0.5)
    record["median_ci_high"] = _first_time(times[failed], high[failed], 0.5)
    # Restricted mean: area under the step function from 0 to the largest exit time.
    largest = float(t[member].max())
    event_times, event_s = times[failed], survivor[failed]
    edges = torch.cat([event_times, torch.tensor([largest], dtype=FLOAT)])
    widths = edges[1:] - edges[:-1]
    first = float(event_times[0]) if event_times.numel() else largest
    pieces = event_s * widths
    record["restricted_mean"] = first + float(pieces.sum())
    tail = pieces.flip(0).cumsum(0).flip(0)                # area from t_j to the largest time
    n_risk, d = curve["n_risk"][rows][failed], curve["n_event"][rows][failed]
    alive = n_risk - d
    terms = torch.where(alive > 0, tail.square() * d / (n_risk * alive.clamp_min(1e-300)),
                        torch.zeros_like(tail))
    se = math.sqrt(float(terms.sum()))
    record.update({"restricted_mean_std_error": se,
                   "restricted_mean_ci_low": record["restricted_mean"] - z * se,
                   "restricted_mean_ci_high": record["restricted_mean"] + z * se,
                   "restricted_mean_limit": largest,
                   "largest_time_censored": bool(not bool(
                       (delta[member] > 0)[t[member] == largest].any()))})
    return record


# ---- log-rank family ----------------------------------------------------------------------


def rank_tests(t: Tensor, t0: Tensor | None, delta: Tensor, w: Tensor, groups: Tensor,
               n_groups: int, strata: Tensor, which: Sequence[str], fh: tuple[float, float],
               scores: Tensor | None) -> dict[str, Any]:
    """Weighted log-rank statistics (u, V) pooled over strata and their chi2 tests."""
    risk = RiskSets.build(t, t0, delta, strata)
    if risk.n_events == 0:
        raise AnalysisError("no_failures", "The tests need at least one failure.")
    member = torch.zeros((t.numel(), n_groups), dtype=FLOAT)
    member[torch.arange(t.numel()), groups] = w
    n_g = risk.at_risk(member)                       # [T, G]
    d_g = risk.at_event(member * delta[:, None])     # [T, G]
    n, d = n_g.sum(1), d_g.sum(1)
    segments = risk.event_stratum
    expected = n_g * (d / n)[:, None]
    ratio = torch.where(n > 1, d * (n - d) / (n * (n - 1.0)).clamp_min(1e-300),
                        torch.zeros_like(n))
    share = n_g / n[:, None]
    survivor = kaplan_meier(n, d, segments)
    before = torch.ones_like(survivor)               # pooled KM just before t_j, per stratum
    before[1:] = survivor[:-1]
    first = torch.searchsorted(segments, segments) == torch.arange(segments.numel())
    before[first] = 1.0
    p, q = fh
    weights = {
        "logrank": torch.ones_like(n), "wilcoxon": n, "tware": n.sqrt(),
        "peto": kaplan_meier(n + 1.0, d, segments),
        "fh": before.pow(p) * (1.0 - before).pow(q),
    }
    results: dict[str, Any] = {"observed": d_g.sum(0), "expected": expected.sum(0)}
    for name in which:
        weight = weights[name]
        u = (weight[:, None] * (d_g - expected)).sum(0)
        scale = (weight.square() * ratio)[:, None]
        v = torch.diag((scale * n_g).sum(0)) - (scale * share).T @ n_g
        v = (v + v.T) / 2
        try:
            statistic, rank = kernel_call(wald_statistic, u[:-1], v[:-1, :-1])
        except AnalysisError:
            statistic, rank = float("nan"), 0
        entry = {"statistic": statistic if rank else None, "df": rank,
                 "p_value": chi2_sf(statistic, rank) if rank else None,
                 "u": u.tolist()}
        if scores is not None:
            numerator = float(scores @ u) ** 2
            denominator = float(scores @ v @ scores)
            trend = numerator / denominator if denominator > 0 else None
            entry["trend"] = {"statistic": trend, "df": 1,
                              "p_value": chi2_sf(trend, 1) if trend is not None else None}
        results[name] = entry
    return results


# ---- public procedure ---------------------------------------------------------------------


def _thin(count: int) -> Tensor:
    if count <= TABLE_LIMIT:
        return torch.arange(count)
    return torch.linspace(0, count - 1, TABLE_LIMIT, dtype=FLOAT).round().to(torch.int64)


@procedure
def sts(data: Any, time: str, *, failure: str | None = None, entry: str | None = None,
        by: str | Sequence[str] | None = None, strata: str | Sequence[str] | None = None,
        weights: str | None = None, alpha: float = 0.05, conftype: str = "loglog",
        test: str | None = None, fh_p: float = 0.0, fh_q: float = 0.0, trend: bool = False,
        missing: str = "drop") -> TableSet:
    """Kaplan-Meier survivor and Nelson-Aalen cumulative hazard functions with tests.

    Stata's ``sts list`` / ``sts list, cumhaz`` / ``stsum`` / ``stci`` / ``sts test``;
    SPSS ``KM``. See ``docs/econometrics/survival.md`` and the module notes for every
    formula.

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    time : exit time (failure or censoring time), >= 0.
    failure : 0/1 failure indicator column (default: every record fails).
    entry : entry time for delayed entry / counting-process records (default 0). Records
        ending on or before their entry are excluded with a note.
    by : column(s) defining the groups whose curves are estimated and compared.
    strata : column(s) for stratified tests (u and V summed over strata).
    weights : frequency weights (Stata's fweight), nonnegative integers.
    alpha : 1 - confidence level of the pointwise intervals and percentile limits.
    conftype : "loglog" (Stata), "log" or "plain" (SPSS) pointwise interval for S(t).
    test : None (all five tests when ``by`` is given), or one of "logrank", "wilcoxon"
        (Wilcoxon-Breslow-Gehan), "tware" (Tarone-Ware), "peto" (Peto-Peto-Prentice),
        "fh" (Fleming-Harrington with ``fh_p``, ``fh_q``).
    fh_p, fh_q : Fleming-Harrington exponents (``fh(0 0)`` is the log-rank test).
    trend : add the trend test for ordered groups (one numeric ``by`` column, at least
        three groups; scores are the group values).
    missing : "drop" (listwise deletion over the columns used) or "raise".

    Returns
    -------
    TableSet with
      ``survival``: per group and distinct exit time: time, n_risk, n_event, n_censored,
      survivor (Kaplan-Meier), std_error (Greenwood), ci_low, ci_high, cumulative_hazard
      (Nelson-Aalen), cumulative_hazard_std_error and its interval. A group with more
      than 2000 distinct times is thinned evenly (noted in ``attrs['notes']``).
      ``summary``: per group n, events, time_at_risk, q25, median (with
      median_ci_low/high), q75, restricted_mean (with std_error and interval) and the
      time it is restricted to.
      ``tests`` (when ``by`` is given): statistic, df, p_value of every requested test
      (plus ``*_trend`` rows); ``expected``: observed and expected failures per group.
      ``attrs``: n, n_dropped, events, groups, conftype, alpha and the primary test's
      statistic, df, p_value.

    Example
    -------
    >>> import openecon as oe
    >>> df = {"t": [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32],
    ...       "d": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...       "arm": [0] * 9 + [1] * 9}
    >>> out = oe.sts(df, "t", failure="d", by="arm", test="logrank")
    >>> round(out.attrs["statistic"], 3) > 0
    True
    """
    alpha = check_alpha(alpha)
    check_choice(conftype, "conftype", ("loglog", "log", "plain"))
    if test is not None:
        check_choice(test, "test", TESTS)
    for value, name in ((fh_p, "fh_p"), (fh_q, "fh_q")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise AnalysisError("invalid_option", f"{name} must be a nonnegative number.")
    if not isinstance(trend, bool):
        raise AnalysisError("invalid_option", "trend must be True or False.")
    by_columns, strata_columns = names_of(by, "by"), names_of(strata, "strata")
    if (test is not None or trend or strata_columns) and not by_columns:
        raise AnalysisError("invalid_spec", "Tests compare groups: give by=... with test, "
                            "trend or strata.")
    notes: list[str] = []
    frame, tensors, dropped = survival_sample(data, time, failure, entry, weights,
                                              [*by_columns, *strata_columns], missing, notes)
    t, t0, delta, w = (tensors[k] for k in ("time", "entry", "failure", "weights"))
    groups, labels = combined_codes(frame, by_columns)
    check_group_count(len(labels), by_columns)
    z = normal_isf(alpha / 2)
    curve = product_limit(t, t0, delta, w, groups, z, conftype)
    survival_rows, summary_rows = [], []
    for g, group_label in enumerate(labels):
        rows = (curve["group"] == g).nonzero().flatten()
        member = groups == g
        summary_rows.append({"group": group_label,
                             **group_summary(curve, rows, t, t0, delta, w, member, z)})
        if summary_rows[-1]["largest_time_censored"]:
            notes.append(f"Group {group_label}: the largest observed time is censored, so the "
                         "restricted mean is restricted to it and underestimates the mean.")
        chosen = rows[_thin(rows.numel())]
        if rows.numel() > TABLE_LIMIT:
            notes.append(f"Group {group_label}: {rows.numel()} distinct times; the survival "
                         f"table shows {TABLE_LIMIT} evenly spaced ones.")
        columns = {"group": [group_label] * chosen.numel()}
        columns.update({key: curve[key][chosen].tolist() for key in SURVIVAL_COLUMNS})
        survival_rows.append(pd.DataFrame(columns))
    survival_table = pd.concat(survival_rows, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows).drop(columns=["largest_time_censored"])
    if not by_columns:
        survival_table = survival_table.drop(columns=["group"])
        summary_table = summary_table.drop(columns=["group"])
    tables = {"survival": table(survival_table), "summary": table(summary_table)}
    attrs: dict[str, Any] = {
        "n": float(w.sum()), "n_dropped": dropped, "events": float((w * delta).sum()),
        "groups": labels if by_columns else None, "conftype": conftype, "alpha": alpha,
        "time": time, "failure": failure, "entry": entry, "by": by_columns or None,
        "weights": weights,
    }
    if by_columns:
        if len(labels) < 2:
            raise AnalysisError("invalid_groups", "The tests need at least two groups in by.")
        stratum_codes, stratum_labels = combined_codes(frame, strata_columns)
        scores = None
        if trend:
            if len(by_columns) != 1 or len(labels) < 3 \
                    or not all(isinstance(v, (int, float)) for v in labels):
                raise AnalysisError("invalid_trend", "The trend test needs one numeric by "
                                    "column with at least three ordered groups.")
            scores = torch.tensor([float(v) for v in labels], dtype=FLOAT)
        which = list(TESTS) if test is None else [test]
        results = rank_tests(t, t0, delta, w, groups, len(labels), stratum_codes, which,
                             (float(fh_p), float(fh_q)), scores)
        test_rows, index = [], []
        for name in which:
            entry_ = results[name]
            title = _TEST_LABELS[name] + (f" ({fh_p:g}, {fh_q:g})" if name == "fh" else "")
            test_rows.append([title, entry_["statistic"], entry_["df"], entry_["p_value"]])
            index.append(name)
            if scores is not None:
                trend_ = entry_["trend"]
                test_rows.append([title + ": trend", trend_["statistic"], 1, trend_["p_value"]])
                index.append(f"{name}_trend")
        tables["tests"] = table(test_rows, columns=["test", "statistic", "df", "p_value"],
                                index=index)
        tables["expected"] = table({"group": labels,
                                    "observed": results["observed"].tolist(),
                                    "expected": results["expected"].tolist()})
        primary = results[test or "logrank"]
        attrs.update({"test": test or "logrank", "statistic": primary["statistic"],
                      "df": primary["df"], "p_value": primary["p_value"],
                      "distribution": "chi2",
                      "stratified": bool(strata_columns), "strata": stratum_labels
                      if strata_columns else None})
    attrs["notes"] = notes
    title = f"Survivor function: {time}" + (f" by {', '.join(by_columns)}" if by_columns else "")
    return TableSet(tables, title=title, **attrs)
