"""Pesaran--Yamagata slope dispersion for static, independent-error panels.

The error variance is estimated from the *unweighted pooled FE* residuals,
not from the individual unrestricted fits.  Gaussian moment adjustment on an
unbalanced panel standardizes each unit separately (CESifo WP1438, remark 3.2).
Three bounded QR passes avoid normal equations and stored N-by-k-by-k factors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Iterator, Sequence

import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype,
)
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import column_list, table
from openecon.engines.distributions import normal_sf

if TYPE_CHECKING:
    import openecon

_SOURCE = "https://doi.org/10.1016/j.jeconom.2007.05.010"
_WORKING_PAPER = "https://www.ifo.de/DocDL/cesifo1_wp1438.pdf"
_METHODS = ("delta", "adjusted", "all")
# Consistent with the panel family's 1e-13 relative within sum-of-squares screen.
_RANK_TOL = math.sqrt(1e-13)
_RESIDUAL_TOL = 1e-14


class _Sum:
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


def _integer(value: Any, name: str, low: int, high: int | None = None) -> int:
    from numbers import Integral

    if isinstance(value, bool) or not isinstance(value, Integral) or value < low \
            or (high is not None and value > high):
        interval = f"{low} through {high}" if high is not None else f"at least {low}"
        raise AnalysisError("invalid_option", f"{name} must be an integer {interval}.")
    return int(value)


@dataclass
class _Sample:
    values: Tensor                    # unit-origin shifted and globally column scaled
    means: Tensor                     # mean of values per unit, subtracted inside tiles
    sizes: Tensor
    starts: Tensor
    n_groups: int
    n_periods: int
    balanced: bool                    # same exact dates, not only the same unit lengths
    nobs_input: int
    n_dropped: int
    data_hash: str


def _prepare(data: Any, y: str, x: Sequence[str], panel: str, time: str,
             missing: str) -> _Sample:
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "xthst requires an in-memory panel table; it does not stream datasets.")
    names = [y, *x, panel, time]
    if not x or not all(isinstance(name, str) and name for name in names) \
            or len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "y, x, panel and time must name distinct columns, with at least one slope in x.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    if not len(frame):
        raise AnalysisError("empty_data", "The panel contains no observations.")
    selected = frame.loc[:, names].reset_index(drop=True)
    if bool(selected[[panel, time]].isna().any().any()):
        raise AnalysisError("missing_panel_time", "Missing unit identifiers or dates are not dropped; supply them explicitly.")
    dates = selected[time]
    if is_bool_dtype(dates.dtype) or not (is_integer_dtype(dates.dtype)
                                         or is_datetime64_any_dtype(dates.dtype)
                                         or is_numeric_dtype(dates.dtype)):
        raise AnalysisError("invalid_time", "time must contain exact integer periods or datetime values.")
    if is_numeric_dtype(dates.dtype) and not is_integer_dtype(dates.dtype):
        numeric_dates = _numeric(dates, time).to(device="cpu")
        if bool(((numeric_dates != numeric_dates.round()) | (numeric_dates.abs() > 2**53 - 1)).any()):
            raise AnalysisError("invalid_time", "Floating dates must be exact integers within 2**53-1; use an integer dtype for larger dates.")
    identifiers = selected[panel]
    if is_numeric_dtype(identifiers.dtype) and not is_integer_dtype(identifiers.dtype):
        _numeric(identifiers, panel)
    try:
        if bool(selected.duplicated([panel, time]).any()):
            raise AnalysisError("duplicate_panel_time", "Every unit/date key must occur once, including rows with missing model inputs.")
        input_units = selected[panel].nunique()
    except TypeError as exc:
        raise AnalysisError("invalid_panel", "Unit identifiers must be hashable scalar values.") from exc
    missing_rows = selected[[y, *x]].isna().any(axis=1)
    dropped = int(missing_rows.sum())
    if dropped and missing == "raise":
        raise AnalysisError("missing_values", f"{dropped} observation(s) have missing y/x inputs; choose missing='drop' to exclude them.")
    selected = selected.loc[~missing_rows].reset_index(drop=True)
    if not len(selected):
        raise AnalysisError("empty_sample", "No complete panel observations remain.")
    units_array, units = pd.factorize(selected[panel], sort=False)
    dates_array, dates = pd.factorize(selected[time], sort=True)
    groups, periods = len(units), len(dates)
    if groups != input_units:
        raise AnalysisError("insufficient_panel_observations", "At least one unit has no complete observations; no unit is silently dropped.")
    if groups < 2:
        raise AnalysisError("insufficient_panels", "Slope homogeneity requires at least two units.")
    codes = torch.as_tensor(units_array, dtype=torch.int64, device="cpu")
    time_codes = torch.as_tensor(dates_array, dtype=torch.int64, device="cpu")
    sizes = torch.bincount(codes, minlength=groups)
    minimum = max(6, len(x) + 2)
    if bool((sizes < minimum).any()):
        raise AnalysisError("insufficient_panel_observations", f"Every unit needs at least {minimum} complete rows: T_i > 5 and T_i > k+1.")
    if groups * periods > 2**63 - 1:
        raise AnalysisError("panel_index_overflow", "Unit/date indexing exceeds int64; restrict the diagnostic sample.")
    order = torch.argsort(codes * periods + time_codes)
    codes = codes[order]
    starts = torch.cat((torch.zeros(1, dtype=torch.int64, device="cpu"), sizes.cumsum(0)[:-1]))
    values = torch.stack([_numeric(selected[name], name).to(device="cpu") for name in [*x, y]], dim=1)[order]
    # Subtracting a unit origin before scaling preserves small within variation
    # in columns with large additive offsets.  Fall back to scale-first only in
    # columns whose opposite-sign subtraction would overflow float64.
    origins = values[starts]
    shifted = values - origins[codes]
    overflow = ~torch.isfinite(shifted).all(dim=0)
    if bool(overflow.any()):
        scales = values[:, overflow].abs().amax(dim=0)
        shifted[:, overflow] = values[:, overflow] / scales - origins[:, overflow][codes] / scales
    scales = shifted.abs().amax(dim=0)
    if bool((scales[:-1] == 0).any()):
        raise AnalysisError("rank_deficient_panel", "Every slope must have within-unit variation in every unit; no slope is automatically omitted.")
    if float(scales[-1]) == 0:
        raise AnalysisError("degenerate_panel_variance", "The outcome has no within-unit variation; positive error variances are required.")
    values = shifted / scales
    means = torch.zeros((groups, len(x) + 1), dtype=torch.float64, device="cpu").index_add_(0, codes, values)
    means /= sizes[:, None]
    return _Sample(values, means, sizes, starts, groups, periods,
                   bool((sizes == periods).all()), len(frame), dropped,
                   _frame_hasher(selected).hexdigest())


def _workspace(sample: _Sample, block_size: int, memory_mb: int) -> dict[str, int | str]:
    width = sample.values.shape[1]
    budget = memory_mb * 1024**2
    # QR factor copies, stacking buffers, SVD/rank and residual temporaries,
    # plus dense row tiles and int64 indices.  LAPACK allocator overhead and
    # the materialized input / unit means / keys are outside this estimate.
    block = min(block_size, sample.n_groups, max(1, budget // (128 * width * width * 8)))
    fixed = 64 * block * width * width * 8
    rows = (budget - fixed) // (block * (12 * width + 4) * 8)
    if rows < width:
        raise AnalysisError("workspace_too_small", "The workspace cannot hold one QR tile with k+1 rows; increase memory_mb or reduce the slope count.")
    rows = min(rows, int(sample.sizes.max()))
    estimate = fixed + block * rows * (12 * width + 4) * 8
    return {"algorithm": "three_pass_batched_tsqr", "budget_bytes": budget,
            "panel_block_size": block, "row_block_size": rows,
            "estimated_tensor_workspace_bytes": estimate,
            "input_tensor_bytes": sample.values.numel() * 8,
            "unit_summary_tensor_bytes": sample.means.numel() * 8 + sample.sizes.numel() * 16,
            "passes": 3}


def _factors(sample: _Sample, workspace: dict[str, Any]) -> Iterator[tuple[Tensor, int, Tensor]]:
    """Yield unit joint [within-X, within-y] R factors in bounded unit/row tiles."""
    block, rows = workspace["panel_block_size"], workspace["row_block_size"]
    width = sample.values.shape[1]
    size_order = torch.argsort(sample.sizes)
    lengths, multiplicities = torch.unique_consecutive(sample.sizes[size_order], return_counts=True)
    cursor = 0
    for size_tensor, count_tensor in zip(lengths, multiplicities, strict=True):
        size = int(size_tensor)
        count = int(count_tensor)
        units = size_order[cursor:cursor + count]
        cursor += count
        for first in range(0, len(units), block):
            selected = units[first:first + block]
            r = None
            for begin in range(0, size, rows):
                end = min(begin + rows, size)
                positions = sample.starts[selected, None] + torch.arange(begin, end, dtype=torch.int64, device="cpu")
                tile = sample.values[positions] - sample.means[selected, None]
                if r is not None:
                    tile = torch.cat((r, tile), dim=1)
                _, r = torch.linalg.qr(tile, mode="r")
            # size >= width+1 and rows >= width, so the final factor is square.
            if r.shape[-2:] != (width, width) or not bool(torch.isfinite(r).all()):
                raise AnalysisError("numerical_failure", "The panel QR factor is nonfinite or has invalid dimensions.")
            yield selected, size, r


def _norm(values: Tensor, dim: int) -> Tensor:
    """Scaled Euclidean norm; do not square tiny/large columns before scaling."""
    peaks = values.abs().amax(dim=dim)
    divisor = torch.where(peaks == 0, torch.ones_like(peaks), peaks)
    return peaks * (values / divisor.unsqueeze(dim)).square().sum(dim=dim).sqrt()


def _rank(r: Tensor, what: str) -> float:
    norms = _norm(r, -2)
    if bool((norms == 0).any()):
        raise AnalysisError("rank_deficient_panel", f"{what} has a slope without within variation; no term is automatically omitted.")
    singular = torch.linalg.svdvals(r / norms.unsqueeze(-2))
    if not bool(torch.isfinite(singular).all()) or bool((singular[..., -1] <= _RANK_TOL * singular[..., 0]).any()):
        raise AnalysisError("rank_deficient_panel", f"{what} is rank deficient or its slopes are too nearly collinear in float64; supply a common full-rank slope set.")
    return float((singular[..., 0] / singular[..., -1]).max())


def _append(r: Tensor | None, block: Tensor) -> Tensor:
    width = block.shape[-1]
    compressed = block.reshape(-1, width)
    if r is not None:
        compressed = torch.cat((r, compressed), dim=0)
    _, result = torch.linalg.qr(compressed, mode="r")
    if not bool(torch.isfinite(result).all()):
        raise AnalysisError("numerical_failure", "The pooled panel QR factor is nonfinite; rescale the model inputs.")
    return result


def _pooled(r: Tensor, label: str) -> Tensor:
    k = r.shape[1] - 1
    _rank(r[:, :k], label)
    beta = torch.linalg.solve_triangular(r[:k, :k], r[:k, k, None], upper=True).flatten()
    if not bool(torch.isfinite(beta).all()):
        raise AnalysisError("numerical_failure", "The common panel slopes are nonfinite; rescale the model inputs.")
    return beta


def _sigma(r: Tensor, beta_fe: Tensor, size: int) -> Tensor:
    residual = r[..., -1] - (r[..., :-1] @ beta_fe)
    residual_norm = _norm(residual, -1)
    outcome_norm = _norm(r[..., -1], -1)
    if bool((residual_norm <= _RESIDUAL_TOL * outcome_norm).any()) or bool((residual_norm == 0).any()):
        raise AnalysisError("degenerate_panel_variance", "A pooled-FE unit residual variance is zero or rounding noise; every unit needs a positive resolvable error variance.")
    sigma = residual_norm / math.sqrt(size - 1)
    if not bool(torch.isfinite(sigma).all()) or bool((sigma == 0).any()):
        raise AnalysisError("numerical_failure", "A unit variance cannot be resolved in float64; rescale the model inputs.")
    return sigma


def _kernel(sample: _Sample, workspace: dict[str, Any]) -> tuple[float, float, float, float]:
    k = sample.values.shape[1] - 1
    factor_fe, condition = None, 0.0
    for _, _, r in _factors(sample, workspace):
        condition = max(condition, _rank(r[..., :k], "An individual panel"))
        factor_fe = _append(factor_fe, r)
    beta_fe = _pooled(factor_fe, "The pooled within design")
    factor_wfe = None
    for _, size, r in _factors(sample, workspace):
        sigma = _sigma(r, beta_fe, size)
        weighted = r / sigma[:, None, None]
        if not bool(torch.isfinite(weighted).all()):
            raise AnalysisError("numerical_failure", "Variance weighting exceeds float64 range; rescale the model inputs.")
        factor_wfe = _append(factor_wfe, weighted)
    beta_wfe = _pooled(factor_wfe, "The variance-weighted within design")
    dispersion, centered, adjusted = _Sum(), _Sum(), _Sum()
    for _, size, r in _factors(sample, workspace):
        sigma = _sigma(r, beta_fe, size)
        # Q'y[:k] - R_x[:k] beta_wfe = R_x[:k](beta_i-beta_wfe).
        # Evaluate this prediction difference without solving individual slopes.
        scores = (r[:, :k, -1] - r[:, :k, :k] @ beta_wfe) / sigma[:, None]
        d = scores.square().sum(dim=1)
        if not bool(torch.isfinite(d).all()):
            raise AnalysisError("numerical_failure", "The slope dispersion is nonfinite.")
        dispersion.add(float(d.sum()))
        centered.add(float((d - k).sum()))
        variance = 2 * k * (size - k - 1) / (size + 1)
        adjusted.add(float(((d - k) / math.sqrt(variance)).sum()))
    root_n = math.sqrt(sample.n_groups)
    return dispersion.total(), centered.total() / math.sqrt(2 * k * sample.n_groups), \
        adjusted.total() / root_n, condition


def xthst(*, data: Any, y: str, x: Sequence[str], panel: str, time: str,
          method: str = "all", missing: str = "raise", block_size: int = 128,
          memory_mb: int = 64) -> openecon.DataFrame:
    """Test a common set of numeric slopes with unit-specific intercepts.

    ``delta`` uses large-T scaling; ``adjusted`` applies Gaussian unit-specific
    finite-T moments; ``all`` returns both.  Balanced and unbalanced samples
    are supported. P-values use the upper standard-normal tail and are always
    asymptotic.  The inferential contract is a static model with strictly
    exogenous regressors and independent Gaussian errors of positive variance
    constant over time within each unit. Variances may differ across units.

    This helper does not infer exogeneity/normality, fit a dynamic model,
    reconstruct a saved fit, or provide HAC/CSD/weight/subset extensions.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise AnalysisError("invalid_option", "method must be 'delta', 'adjusted' or 'all'.")
    if not isinstance(missing, str) or missing not in ("raise", "drop"):
        raise AnalysisError("invalid_option", "missing must be 'raise' or 'drop'.")
    slopes = column_list(x, "x")
    block_size = _integer(block_size, "block_size", 1, 512)
    memory_mb = _integer(memory_mb, "memory_mb", 1)
    # _numeric's array bridge also follows Torch's default device.  Override it
    # locally before constructing tensors, without changing the caller's state.
    with torch.no_grad(), torch.device("cpu"):
        sample = _prepare(data, y, slopes, panel, time, missing)
        workspace = _workspace(sample, block_size, memory_mb)
        dispersion, delta, adjusted, condition = _kernel(sample, workspace)
        statistics = {"delta": delta, "adjusted": adjusted}
        methods = ["delta", "adjusted"] if method == "all" else [method]
        rows = [{"statistic": statistics[name],
                 "p_value": float(normal_sf(torch.tensor(statistics[name], dtype=torch.float64, device="cpu"))),
                 "n_groups": sample.n_groups, "nobs": len(sample.values), "k": len(slopes)} for name in methods]
    notes = ["Upper-tail asymptotic standard-normal p-values; neither statistic is an exact finite-sample test.",
             "Assumes a static, strictly exogenous model with independent Gaussian errors, constant variance over time within each unit; variances may differ across units.",
             "Inference requires large N and T_i with the paper's design/moment regularity conditions; the accepted minimum T_i is a domain guard, not a sample-size guarantee."]
    if "adjusted" in methods:
        notes.append("Adjusted uses Gaussian moments for each T_i separately; this does not remove finite-N estimation error or allow serial/cross-section dependence.")
    if sample.n_dropped:
        notes.append(f"Excluded {sample.n_dropped} row(s) with missing y/x inputs; no unit or slope was silently omitted.")
    return table(rows, index=methods, title="Pesaran--Yamagata slope homogeneity", procedure="xthst",
                 y=y, x=slopes, panel=panel, time=time, methods=methods, missing=missing,
                 n_groups=sample.n_groups, nobs=len(sample.values), nobs_input=sample.nobs_input,
                 n_dropped=sample.n_dropped, k=len(slopes), n_periods=sample.n_periods,
                 t_min=int(sample.sizes.min()), t_max=int(sample.sizes.max()),
                 balanced=sample.balanced, equal_unit_lengths=bool((sample.sizes == sample.sizes[0]).all()),
                 dispersion=dispersion, max_unit_scaled_condition=condition,
                 null="Every coefficient in x is the same across units; intercepts may differ.",
                 alternative="Slopes differ across a nonvanishing fraction of units.",
                 p_value_tail="upper", reference_distribution="asymptotic standard normal",
                 variance_estimator="unweighted pooled-FE residual SSR_i/(T_i-1)",
                 adjustment="unit-specific 2*k*(T_i-k-1)/(T_i+1)", notes=notes,
                 workspace=workspace, provenance={"method": "native_float64_torch_tsqr", "dtype": "float64",
                 "device": "cpu", "source": _SOURCE, "working_paper": _WORKING_PAPER,
                 "working_paper_equations": ["3.1", "3.2", "3.3", "3.17", "3.20", "3.21", "3.22", "3.23"],
                 "data_hash": sample.data_hash, "stata_parity_validated": False,
                 "finite_sample_p_values": False, "gaussian_inference_only": True})
