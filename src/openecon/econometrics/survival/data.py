"""Survival-time data and risk-set bookkeeping shared by the survival family.

Survival data are declared per call, as Stata's ``stset`` would declare them:

* ``time``: the exit time ``t_i`` of record i (failure or censoring time);
* ``failure``: the 0/1 event indicator ``delta_i`` (every record fails when it is
  not given);
* ``entry``: the entry time ``t0_i`` (delayed entry / left truncation, or the
  start of a ``(start, stop]`` record of counting-process data), 0 when not given;
* ``id``: the subject identifier when a subject contributes several records.

Record i is at risk at time ``s`` when ``t0_i < s <= t_i``. Records with
``t_i <= t0_i`` are never at risk and are excluded with a recorded note, as
``stset`` excludes them ("obs. end on or before enter()"). Negative times are
an error.

Risk sets without loops
-----------------------
Every risk-set quantity is a sum over the records at risk at the distinct
failure times ``tau_1 < ... < tau_T`` (per stratum). With all time values (exits
and entries) replaced by their dense rank ``r`` and a stratum code ``s``, the key
``s (R + 1) + r`` orders records stratum-major, so one sort of the exit keys and
one sort of the entry keys serve every stratum at once:

    sum_{i at risk at tau_k} v_i = sum_{s_i = s_k, t_i >= tau_k} v_i
                                   - sum_{s_i = s_k, t0_i >= tau_k} v_i

and both terms are differences of suffix sums over the sorted keys at positions
found by ``searchsorted``: O(n log n) once, O(n k) per evaluation. The
transposed operation, ``A_i = sum_{k: i at risk at tau_k} a_k`` for a vector
``a`` over failure times, is a difference of prefix sums of ``a`` at the
positions of the record's exit and entry keys. Together they give the Cox
score and Hessian without forming per-time k-by-k matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError

FLOAT = torch.float64


def safe_exp(value: float) -> float:
    """``exp`` that returns inf instead of raising; the result layer reports inf as missing."""
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def failure_indicator(values: Tensor | None, n: int, name: str | None) -> Tensor:
    """The 0/1 failure indicator (all ones when no failure column is declared)."""
    if values is None:
        return torch.ones(n, dtype=FLOAT)
    if bool(((values != 0) & (values != 1)).any()):
        raise AnalysisError(
            "invalid_failure_indicator",
            f"The failure column '{name}' must be coded 0 (censored) or 1 (failure); recode it "
            "explicitly (Stata treats every nonzero value as a failure, OpenEconometrics does not guess).")
    return values.to(FLOAT)


def check_times(time: Tensor, entry: Tensor | None, time_name: str,
                entry_name: str | None) -> Tensor:
    """Validate exit and entry times; return the mask of usable records (t > t0).

    Negative times are refused: analysis time starts at 0. Records ending on or
    before their entry are never at risk; the caller excludes them with a note.
    """
    if bool((time < 0).any()):
        raise AnalysisError("invalid_survival_time",
                            f"The time column '{time_name}' contains negative values; survival "
                            "times are measured from 0 (shift the time origin).")
    if entry is None:
        return time > 0
    if bool((entry < 0).any()):
        raise AnalysisError("invalid_survival_time",
                            f"The entry column '{entry_name}' contains negative values; entry "
                            "times are measured from 0 like the exit times.")
    return time > entry


def end_before_entry_note(count: int) -> str:
    return (f"Excluded {count} record(s) that end on or before their entry time (never at "
            "risk; Stata's stset reports them as 'obs. end on or before enter()').")


def check_overlap(time: Tensor, entry: Tensor, ids: Tensor) -> None:
    """Records of one subject must not overlap in time (Stata's stset check)."""
    order = torch.argsort(entry, stable=True)
    order = order[torch.argsort(ids[order], stable=True)]
    same = ids[order][1:] == ids[order][:-1]
    overlap = same & (entry[order][1:] < time[order][:-1])
    if bool(overlap.any()):
        raise AnalysisError(
            "overlapping_records",
            f"{int(overlap.sum())} record(s) overlap an earlier record of the same subject in "
            "time. Records of one id must describe disjoint (entry, exit] intervals.")


@dataclass
class RiskSets:
    """Risk-set structure of (entry, exit] records in strata (see the module notes).

    ``event_time``/``event_stratum`` describe the T distinct failure times in
    stratum-major order; ``event_of`` maps every failing record to its failure
    time (``-1`` for censored records).
    """

    n: int
    n_events: int
    event_time: Tensor          # [T] tau_k
    event_stratum: Tensor       # [T] stratum code of tau_k
    event_of: Tensor            # [n] index of the record's failure time, -1 if censored
    exit_order: Tensor          # [n] records sorted by exit key
    entry_order: Tensor | None  # [n] records sorted by entry key (None without delayed entry)
    exit_low: Tensor            # [T] first sorted-exit position with exit >= tau_k
    exit_high: Tensor           # [T] end of tau_k's stratum in the sorted exits
    entry_low: Tensor | None    # [T] first sorted-entry position with entry >= tau_k
    entry_high: Tensor | None
    exit_count: Tensor          # [n] failure times with key <= the record's exit key
    entry_count: Tensor         # [n] failure times with key <= the record's entry key

    @classmethod
    def build(cls, time: Tensor, entry: Tensor | None, failure: Tensor,
              strata: Tensor | None = None) -> RiskSets:
        n = time.numel()
        values = time if entry is None else torch.cat([time, entry])
        unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
        stride = unique.numel() + 1
        base = torch.zeros(n, dtype=torch.int64) if strata is None else strata * stride
        exit_key = base + inverse[:n]
        failed = failure > 0
        event_keys = torch.unique(exit_key[failed], sorted=True)
        n_events = event_keys.numel()
        event_time = unique[event_keys % stride]
        event_stratum = event_keys // stride
        event_of = torch.full((n,), -1, dtype=torch.int64)
        event_of[failed] = torch.searchsorted(event_keys, exit_key[failed])
        next_stratum = (event_stratum + 1) * stride
        sorted_exit, exit_order = torch.sort(exit_key, stable=True)
        exit_low = torch.searchsorted(sorted_exit, event_keys)
        exit_high = torch.searchsorted(sorted_exit, next_stratum)
        exit_count = torch.searchsorted(event_keys, exit_key, right=True)
        if entry is None:
            entry_order = entry_low = entry_high = None
            # Every record enters at 0 < tau: no failure time precedes its entry key.
            entry_count = torch.searchsorted(event_keys, base - 1, right=True)
        else:
            entry_key = base + inverse[n:]
            sorted_entry, entry_order = torch.sort(entry_key, stable=True)
            entry_low = torch.searchsorted(sorted_entry, event_keys)
            entry_high = torch.searchsorted(sorted_entry, next_stratum)
            entry_count = torch.searchsorted(event_keys, entry_key, right=True)
        return cls(n, n_events, event_time, event_stratum, event_of, exit_order, entry_order,
                   exit_low, exit_high, entry_low, entry_high, exit_count, entry_count)

    # ---- sums over risk sets ------------------------------------------------------

    @staticmethod
    def _suffix(values: Tensor, order: Tensor) -> Tensor:
        ordered = values[order]
        suffix = torch.zeros((ordered.shape[0] + 1, *ordered.shape[1:]), dtype=FLOAT)
        suffix[:-1] = ordered.flip(0).cumsum(0).flip(0)
        return suffix

    def at_risk(self, values: Tensor) -> Tensor:
        """``sum_{i in R(tau_k)} values_i`` for every failure time ([T] or [T, m])."""
        suffix = self._suffix(values, self.exit_order)
        total = suffix[self.exit_low] - suffix[self.exit_high]
        if self.entry_order is not None:
            late = self._suffix(values, self.entry_order)
            total = total - (late[self.entry_low] - late[self.entry_high])
        return total

    def accumulate(self, per_time: Tensor) -> Tensor:
        """``sum_{k: i in R(tau_k)} per_time_k`` for every record ([n] or [n, m])."""
        prefix = torch.zeros((self.n_events + 1, *per_time.shape[1:]), dtype=FLOAT)
        prefix[1:] = per_time.cumsum(0)
        return prefix[self.exit_count] - prefix[self.entry_count]

    def at_event(self, values: Tensor) -> Tensor:
        """``sum_{i fails at tau_k} values_i`` for every failure time ([T] or [T, m])."""
        rows = self.event_of >= 0
        out = torch.zeros((self.n_events, *values.shape[1:]), dtype=FLOAT)
        return out.index_add_(0, self.event_of[rows], values[rows])

    def within_stratum_cumsum(self, per_time: Tensor) -> Tensor:
        """Cumulative sums over the failure times of each stratum (restarting per stratum)."""
        return segmented_cumsum(per_time, self.event_stratum)


def segmented_cumsum(values: Tensor, segments: Tensor) -> Tensor:
    """Cumulative sums restarting at every change of the nondecreasing ``segments`` codes."""
    total = values.cumsum(0)
    if values.shape[0] == 0:
        return total
    first = torch.searchsorted(segments, segments)          # first index of each row's segment
    before = torch.zeros_like(total)
    before[1:] = total[:-1]
    return total - before[first]


def kaplan_meier(at_risk: Tensor, events: Tensor, segments: Tensor) -> Tensor:
    """Product-limit survivor values ``prod_{j <= k} (1 - d_j / n_j)`` per failure time.

    ``segments`` (nondecreasing stratum codes of the failure times) restarts the
    product in every stratum. The product is ``exp(cumsum(log(1 - d/n)))``; a
    factor of exactly zero (everyone at risk fails) makes every later value of
    that stratum zero.
    """
    factor = 1.0 - events / at_risk
    zero = factor <= 0
    logs = torch.log(torch.where(zero, torch.ones_like(factor), factor))
    survivor = torch.exp(segmented_cumsum(logs, segments))
    dead = segmented_cumsum(zero.to(FLOAT), segments) > 0
    return torch.where(dead, torch.zeros_like(survivor), survivor)
