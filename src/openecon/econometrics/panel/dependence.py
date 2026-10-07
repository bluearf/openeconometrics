"""Cross-section residual dependence with bounded float64 tensor workspaces.

Pesaran (2004), IZA DP 1240: equations (3), (6), (7), and (31), section 9.
Only supplied residuals are tested: no model is fitted or reconstructed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.engines.distributions import chi2_sf, normal_sf

_SOURCE = "https://docs.iza.org/dp1240.pdf"
_METHODS = ("cd", "lm", "scaled_lm", "all")
_EPS = torch.finfo(torch.float64).eps


class _Sum:
    """Compensated scalar accumulation without one stored value per pair tile."""

    def __init__(self):
        self.value = self.correction = 0.0

    def add(self, value: float) -> None:
        total = self.value + value
        if abs(self.value) >= abs(value):
            self.correction += (self.value - total) + value
        else:
            self.correction += (value - total) + self.value
        self.value = total

    def total(self) -> float:
        return self.value + self.correction


def _integer(value: Any, name: str, lower: int, upper: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < lower or (upper is not None and value > upper):
        suffix = f" through {upper}" if upper is not None else " or greater"
        raise AnalysisError("invalid_option", f"{name} must be an integer {lower}{suffix}.")
    return int(value)


@dataclass
class _Residuals:
    values: Tensor                 # grouped by unit, then exact date; unit L2 norm = 1
    times: Tensor                  # integer codes for exact observed dates, not ranks inside units
    offsets: Tensor
    n_groups: int
    n_periods: int
    nobs_input: int
    n_dropped: int
    balanced: bool

    def matrices(self, start: int, stop: int, first: int, last: int) -> tuple[Tensor, Tensor]:
        """One bounded unit/date tile; loops over units, never observations."""
        if self.balanced:
            values = self.values.view(self.n_groups, self.n_periods)[start:stop, first:last]
            return values, torch.ones_like(values)
        values = torch.zeros((stop - start, last - first), dtype=torch.float64)
        present = torch.zeros_like(values)
        for unit in range(start, stop):
            begin, end = int(self.offsets[unit]), int(self.offsets[unit + 1])
            times = self.times[begin:end]
            bounds = torch.searchsorted(times, torch.tensor([first, last], dtype=torch.int64))
            left, right = begin + int(bounds[0]), begin + int(bounds[1])
            positions = self.times[left:right] - first
            values[unit - start, positions] = self.values[left:right]
            present[unit - start, positions] = 1
        return values, present


def _prepare(data: Any, residual: str, panel: str, time: str, missing: str, minimum: int) -> _Residuals:
    names = [residual, panel, time]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != 3:
        raise AnalysisError("invalid_spec", "residual, panel and time must name three distinct columns.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The residual panel contains no observations.")
    selected = frame.loc[:, names].reset_index(drop=True)
    if bool(selected[[panel, time]].isna().any().any()):
        raise AnalysisError("missing_panel_time", "Missing unit identifiers or dates cannot be aligned; supply them explicitly.")
    dates = selected[time]
    if is_bool_dtype(dates.dtype) or not (is_integer_dtype(dates.dtype) or is_datetime64_any_dtype(dates.dtype)
                                         or is_numeric_dtype(dates.dtype)):
        raise AnalysisError("invalid_time", "time must contain exact integer periods or datetime values.")
    if is_numeric_dtype(dates.dtype) and not is_integer_dtype(dates.dtype):
        numeric_dates = _numeric(dates, time)
        if bool(((numeric_dates != numeric_dates.round()) | (numeric_dates.abs() > 2**53 - 1)).any()):
            raise AnalysisError("invalid_time", "Floating dates must be exact integers within 2**53-1; use an integer dtype for larger periods.")
    identifiers = selected[panel]
    if is_numeric_dtype(identifiers.dtype) and not is_integer_dtype(identifiers.dtype):
        _numeric(identifiers, panel)  # reject infinite/complex labels without rounding integer labels
    try:
        if bool(selected.duplicated([panel, time]).any()):
            raise AnalysisError("duplicate_panel_time", "Every unit/date key must occur exactly once, including rows with missing residuals.")
        input_units = selected[panel].nunique()
    except TypeError as exc:
        raise AnalysisError("invalid_panel", "Unit identifiers must be hashable scalar values.") from exc
    absent_residuals = selected[residual].isna()
    dropped = int(absent_residuals.sum())
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", f"{dropped} residual(s) are missing; use missing='drop' for pairwise date overlap.")
    selected = selected.loc[~absent_residuals].reset_index(drop=True)
    if not len(selected):
        raise AnalysisError("empty_sample", "No finite residual observations remain.")
    codes_array, units = pd.factorize(selected[panel], sort=False)
    time_array, dates = pd.factorize(selected[time], sort=True)
    groups, periods = len(units), len(dates)
    if groups != input_units:
        raise AnalysisError("insufficient_panel_observations", "At least one unit has no nonmissing residual observations.")
    if groups < 2:
        raise AnalysisError("insufficient_panels", "Cross-section dependence requires at least two units.")
    codes = torch.as_tensor(codes_array, dtype=torch.int64)
    times = torch.as_tensor(time_array, dtype=torch.int64)
    values = _numeric(selected[residual], residual)
    counts = torch.bincount(codes, minlength=groups)
    if bool((counts < minimum).any()):
        raise AnalysisError("insufficient_panel_observations", f"Every unit must supply at least {minimum} residual observations.")
    if groups * periods > 2**63 - 1:
        raise AnalysisError("panel_index_overflow", "Unit/date indexing exceeds int64; restrict the diagnostic sample.")
    order = torch.argsort(codes * periods + times)
    codes, times, values = codes[order], times[order], values[order]
    scales = torch.zeros(groups, dtype=torch.float64).scatter_reduce_(0, codes, values.abs(), reduce="amax", include_self=True)
    if bool((scales == 0).any()):
        raise AnalysisError("degenerate_residuals", "Every unit needs residual variation; one unit has only zeros.")
    # Divide before summing/subtracting to avoid overflow at large residual scales.
    values = values / scales[codes]
    means = torch.zeros(groups, dtype=torch.float64).index_add_(0, codes, values) / counts
    values = values - means[codes]
    norms = torch.zeros(groups, dtype=torch.float64).index_add_(0, codes, values.square()).sqrt()
    if bool((norms == 0).any()):
        raise AnalysisError("degenerate_residuals", "Every unit needs nonconstant residuals.")
    values = values / norms[codes]
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return _Residuals(values, times, offsets, groups, periods, len(frame), dropped,
                      bool((counts == periods).all()))


def _workspace(sample: _Residuals, block_size: int, memory_mb: int) -> tuple[int, int, int]:
    # Conservative allowance for simultaneously live pair buffers, masks, dense
    # tiles and expression temporaries. Input/table/index storage is separate.
    budget = memory_mb * 1024**2
    block = min(block_size, sample.n_groups, max(1, math.isqrt(budget // (2 * 48 * 8))))
    fixed = 48 * block * block * 8
    chunk = min(sample.n_periods, (budget - fixed) // ((24 * block + 8) * 8))
    if chunk < 1:
        raise AnalysisError("workspace_too_small", "The requested workspace cannot hold one unit/date tile.")
    estimated = fixed + (24 * block + 8) * chunk * 8
    return block, chunk, estimated


def _balanced_cd(sample: _Residuals, block: int, chunk: int) -> float:
    """Sum correlations through ||sum_i normalized(e_i)||², in linear work."""
    squares = _Sum()
    for first in range(0, sample.n_periods, chunk):
        last = min(sample.n_periods, first + chunk)
        total = torch.zeros(last - first, dtype=torch.float64)
        for start in range(0, sample.n_groups, block):
            values, _ = sample.matrices(start, min(sample.n_groups, start + block), first, last)
            total += values.sum(dim=0)
        squares.add(float(total.square().sum()))
    return (squares.total() - sample.n_groups) / 2


def _pair_statistics(sample: _Residuals, block: int, chunk: int, minimum: int) -> dict[str, float | int]:
    correlation_sums, weighted_sums, square_sums = _Sum(), _Sum(), _Sum()
    smallest, largest = sample.n_periods, 0
    for start in range(0, sample.n_groups, block):
        stop = min(sample.n_groups, start + block)
        for other in range(start, sample.n_groups, block):
            end = min(sample.n_groups, other + block)
            shape = (stop - start, end - other)
            cross = torch.zeros(shape, dtype=torch.float64)
            if not sample.balanced:
                count, sum_left, sum_right, sq_left, sq_right = [torch.zeros(shape, dtype=torch.float64) for _ in range(5)]
            for first in range(0, sample.n_periods, chunk):
                last = min(sample.n_periods, first + chunk)
                left, left_mask = sample.matrices(start, stop, first, last)
                right, right_mask = sample.matrices(other, end, first, last)
                cross += left @ right.T
                if not sample.balanced:
                    count += left_mask @ right_mask.T
                    sum_left += left @ right_mask.T
                    sum_right += left_mask @ right.T
                    sq_left += left.square() @ right_mask.T
                    sq_right += left_mask @ right.square().T
            mask = torch.ones(shape, dtype=torch.bool)
            if start == other:
                mask = torch.triu(mask, diagonal=1)
            if not bool(mask.any()):
                continue
            if sample.balanced:
                overlap = torch.full((int(mask.sum()),), sample.n_periods, dtype=torch.float64)
                rho = cross[mask]
            else:
                overlap = count[mask]
                if bool((overlap < minimum).any()):
                    raise AnalysisError("insufficient_pair_overlap", f"Every unit pair needs at least {minimum} common dates; no pair is silently omitted.")
                sl, sr = sum_left[mask], sum_right[mask]
                ql, qr = sq_left[mask], sq_right[mask]
                vl, vr = ql - sl.square() / overlap, qr - sr.square() / overlap
                # Never turn cancellation or a constant overlap into a correlation.
                if bool(((vl <= 64 * _EPS * ql) | (vr <= 64 * _EPS * qr)).any()):
                    raise AnalysisError("degenerate_pair_variance", "An overlap has constant residuals or its centered variance cannot be resolved reliably in float64; restrict/rescale the residual sample.")
                rho = (cross[mask] - sl * sr / overlap) / (vl.sqrt() * vr.sqrt())
            if not bool(torch.isfinite(rho).all()) or bool((rho.abs() > 1 + 1e-10).any()):
                raise AnalysisError("numerical_failure", "A pair correlation is nonfinite or outside its numerical domain.")
            # Project only tiny roundoff at the algebraic [-1,1] boundary.
            rho = rho.clamp(-1, 1)
            correlation_sums.add(float(rho.sum()))
            weighted_sums.add(float((overlap.sqrt() * rho).sum()))
            square_sums.add(float((overlap * rho.square()).sum()))
            smallest = min(smallest, int(overlap.min()))
            largest = max(largest, int(overlap.max()))
    return {"correlation_sum": correlation_sums.total(), "weighted_correlation_sum": weighted_sums.total(),
            "lm": square_sums.total(), "overlap_min": smallest, "overlap_max": largest}


def xtcd(*, data: Any, residual: str, panel: str, time: str, method: str = "cd",
         missing: str = "raise", min_overlap: int = 4, block_size: int = 128,
         memory_mb: int = 64):
    """Test supplied panel residuals for contemporaneous cross-section dependence.

    ``cd`` supports exact pairwise common-date centering on unbalanced panels.
    ``lm``, ``scaled_lm`` and ``all`` require a balanced common-date sample.
    Every pair needs at least ``min_overlap >= 4`` dates; missing unit/date keys,
    duplicate keys and undefined overlap variances are errors. ``missing='drop'``
    drops only missing residual rows. Integer periods (including large int64)
    and datetime dates retain exact identities; gaps do not create artificial
    matches. No model is refitted and no weights/common-factor correction is used.

    ``memory_mb`` bounds an estimate of tensor calculation workspace, separate
    from the in-memory input/index arrays, allocator and BLAS private workspaces.
    Panel/date tiles avoid unbounded N-by-N/N-by-T allocations. Balanced CD uses
    a linear aggregation identity; LM and unbalanced CD examine every unit pair.
    Normal/chi-square p-values are asymptotic, not finite-sample guarantees.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(_METHODS)}.")
    if not isinstance(missing, str) or missing not in ("raise", "drop"):
        raise AnalysisError("invalid_option", "missing must be 'raise' or 'drop'.")
    minimum = _integer(min_overlap, "min_overlap", 4)
    requested = _integer(block_size, "block_size", 1, 512)
    memory_mb = _integer(memory_mb, "memory_mb", 1, 1024)
    sample = _prepare(data, residual, panel, time, missing, minimum)
    if not sample.balanced and method != "cd":
        raise AnalysisError("unbalanced_lm_unsupported", "LM/scaled LM/all require every unit to share the same complete date set; only method='cd' supports unbalanced overlaps.")
    block, chunk, estimate = _workspace(sample, requested, memory_mb)
    pairs = sample.n_groups * (sample.n_groups - 1) // 2
    linear_cd = sample.balanced and method == "cd"
    if linear_cd:
        correlation = _balanced_cd(sample, block, chunk)
        metrics = {"correlation_sum": correlation, "weighted_correlation_sum": math.sqrt(sample.n_periods) * correlation,
                   "overlap_min": sample.n_periods, "overlap_max": sample.n_periods}
    else:
        metrics = _pair_statistics(sample, block, chunk, minimum)
    cd = float(metrics["weighted_correlation_sum"]) / math.sqrt(pairs)
    rows, index = [], []
    if method in ("cd", "all"):
        rows.append([cd, "normal", None, kernel_call(normal_sf, abs(cd)) * 2, "two-sided"])
        index.append("cd")
    if method in ("lm", "all"):
        lm = float(metrics["lm"])
        rows.append([lm, "chi2", pairs, kernel_call(chi2_sf, lm, pairs), "upper"])
        index.append("lm")
    if method in ("scaled_lm", "all"):
        scaled = (float(metrics["lm"]) - pairs) / math.sqrt(2 * pairs)
        rows.append([scaled, "normal", None, kernel_call(normal_sf, abs(scaled)) * 2, "two-sided"])
        index.append("scaled_lm")
    if not all(math.isfinite(row[0]) and math.isfinite(row[3]) for row in rows):
        raise AnalysisError("numerical_failure", "The requested dependence statistic is not finite.")
    notes = ["Supplied residuals must satisfy the chosen model/null assumptions; this function does not verify their fitted origin.",
             "CD uses centered common-date correlations and asymptotic normal inference; opposite-signed dependencies can cancel.",
             "Serially independent, symmetric innovations and an appropriate unit-specific regression design underpin Pesaran's CD null.",
             "No correction for estimated common/time effects, latent factors, serial correlation, weights or selected unit pairs is applied."]
    if method != "cd":
        notes.append("BP LM assumes fixed/small N and large T; scaled LM takes T large before N. Both can be badly sized for small T/large N; scaled LM is not bias-adjusted LM.")
    return table(rows, columns=["statistic", "distribution", "df", "p_value", "alternative"], index=index,
                 method=method, label="Cross-section residual dependence", null="Zero contemporaneous cross-section residual correlations",
                 nobs=sample.nobs_input - sample.n_dropped, nobs_input=sample.nobs_input, n_dropped=sample.n_dropped,
                 n_groups=sample.n_groups, n_periods=sample.n_periods, n_pairs=pairs, balanced=sample.balanced,
                 overlap_min=metrics["overlap_min"], overlap_max=metrics["overlap_max"], min_overlap=minimum,
                 mean_correlation=float(metrics["correlation_sum"]) / pairs,
                 weighted_correlation_sum=metrics["weighted_correlation_sum"], residual_column=residual,
                 panel_column=panel, time_column=time, notes=notes,
                 workspace={"algorithm": "balanced_linear_cd" if linear_cd else "pairwise_tiles",
                            "requested_block_size": requested, "panel_block_size": block, "time_block_size": chunk,
                            "estimated_tensor_workspace_bytes": estimate, "budget_bytes": memory_mb * 1024**2,
                            "input_storage": "in-memory O(nobs); excluded from calculation workspace estimate",
                            "includes_private_blas_workspace": False},
                 provenance={"method_source": _SOURCE, "equations": [3, 6, 7, 31], "dtype": "float64",
                             "device": "cpu", "stata_parity_validated": False, "refit": False,
                             "pairwise_centered": True, "common_effect_correction": False})
