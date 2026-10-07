"""Native cross-sectionally augmented panel unit-root tests (Pesaran 2007)."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.unitroot.cips_tables import SOURCE, TRUNCATION, critical_values
from openecon.econometrics.unitroot.common import check_choice, check_count, check_flag, json_label
from openecon.econometrics.unitroot.panel_kernels import batch_ols

if TYPE_CHECKING:
    import openecon


def _cpu_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.device("cpu"):
            return function(*args, **kwargs)

    return wrapped


def _sample(data: Any, y: str, panel: str, time: str):
    names = [y, panel, time]
    if not all(isinstance(n, str) and n for n in names) or len(set(names)) != 3:
        raise AnalysisError(
            "invalid_spec", "y, panel and time must name distinct nonempty columns."
        )
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    if not len(frame):
        raise AnalysisError("empty_data", "The panel contains no observations.")
    absent = [n for n in names if n not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    chosen = frame.loc[:, names].reset_index(drop=True)
    if bool(chosen.isna().any().any()):
        raise AnalysisError(
            "missing_values",
            "CADF/CIPS needs complete common-date panels; no missing rows are silently dropped.",
        )
    dates = chosen[time]
    if is_bool_dtype(dates.dtype) or not (
        is_integer_dtype(dates.dtype)
        or is_datetime64_any_dtype(dates.dtype)
        or is_numeric_dtype(dates.dtype)
    ):
        raise AnalysisError(
            "invalid_time", "time must contain exact integer periods or regular datetime dates."
        )
    if is_numeric_dtype(dates.dtype) and not is_integer_dtype(dates.dtype):
        numeric = _numeric(dates, time)
        if bool(((numeric != numeric.round()) | (numeric.abs() > 2**53 - 1)).any()):
            raise AnalysisError(
                "invalid_time",
                "Floating periods must be exact integers within 2**53-1; use an integer dtype for larger periods.",
            )
    try:
        duplicated = bool(chosen.duplicated([panel, time]).any())
        codes, labels = pd.factorize(chosen[panel], sort=True)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            "invalid_panel", "Panel identifiers must be hashable scalar values."
        ) from exc
    if duplicated:
        raise AnalysisError("repeated_time_values", "Each panel/time key must occur exactly once.")
    if is_numeric_dtype(chosen[panel].dtype) and not is_integer_dtype(chosen[panel].dtype):
        _numeric(chosen[panel], panel)
    dates_codes, date_labels = pd.factorize(dates, sort=True)
    n, total = len(labels), len(date_labels)
    if n < 2:
        raise AnalysisError("insufficient_panels", "CADF/CIPS requires at least two panels.")
    if total < 5:
        raise AnalysisError(
            "insufficient_observations",
            "CADF/CIPS requires at least five levels per panel, and additional observations for lags/trends.",
        )
    if n * total > 2**63 - 1:
        raise AnalysisError("panel_index_overflow", "Unit/time indexing exceeds int64.")
    unit = torch.as_tensor(codes, dtype=torch.int64)
    date = torch.as_tensor(dates_codes, dtype=torch.int64)
    counts = torch.bincount(unit, minlength=n)
    if not bool((counts == total).all()):
        raise AnalysisError(
            "unbalanced_panel",
            "CADF/CIPS requires exactly the same complete date set in every panel.",
        )
    if is_datetime64_any_dtype(dates.dtype):
        if pd.infer_freq(pd.DatetimeIndex(date_labels)) is None:
            raise AnalysisError(
                "time_gaps",
                "Datetime observations must share a regular calendar frequency; fill or restrict gaps before testing.",
            )
    elif int(date_labels[-1]) - int(date_labels[0]) != total - 1:
        raise AnalysisError(
            "time_gaps",
            "Integer periods must be consecutive; missing common periods cannot be treated as one-step lags.",
        )
    order = torch.argsort(unit * total + date)
    values = _numeric(chosen[y], y)[order].reshape(n, total)
    return (
        values,
        [json_label(v) for v in labels],
        [json_label(date_labels[0]), json_label(date_labels[-1])],
    )


def _design(y: Tensor, mean: Tensor, order: int, trend: str, start: int):
    count, total = y.shape
    nobs = total - 1 - start
    k = 3 + 2 * order + {"none": 0, "constant": 1, "trend": 2}[trend]
    if nobs <= k:
        raise AnalysisError(
            "insufficient_observations",
            f"CADF has {nobs} observations for {k} parameters; use fewer lags or longer panels.",
        )
    dy, dm = y[:, 1:] - y[:, :-1], mean[1:] - mean[:-1]
    columns = [y[:, start:-1], mean[start:-1].expand(count, nobs), dm[start:].expand(count, nobs)]
    columns += [dm[start - j : total - 1 - j].expand(count, nobs) for j in range(1, order + 1)]
    columns += [dy[:, start - j : total - 1 - j] for j in range(1, order + 1)]
    if trend != "none":
        columns.append(torch.ones((count, nobs), dtype=torch.float64))
    if trend == "trend":
        # Centering/scaling a deterministic trend leaves the regression space unchanged.
        columns.append(torch.linspace(-1, 1, nobs, dtype=torch.float64).expand(count, nobs))
    return torch.stack(columns, dim=2), dy[:, start:]


def _fit(y: Tensor, mean: Tensor, order: int, trend: str, start: int):
    x, target = _design(y, mean, order, trend, start)
    scales = x.abs().amax(dim=1)
    target_scale = target.abs().amax(dim=1)
    if bool((scales == 0).any()):
        raise AnalysisError(
            "collinear_regressors",
            "A CADF column is zero; common means and every individual series must identify the full augmented regression.",
        )
    if bool((target_scale == 0).any()):
        raise AnalysisError("constant_series", "Every panel needs nonzero differences.")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(target).all()):
        raise AnalysisError(
            "numerical_failure",
            "Differences or levels overflow float64; rescale the series before testing.",
        )
    fit = batch_ols(
        x / scales[:, None, :],
        target / target_scale[:, None],
        "The cross-sectionally augmented Dickey-Fuller regression",
    )
    if bool((fit.ssr <= 1e-24 * (target / target_scale[:, None]).square().sum(dim=1)).any()):
        raise AnalysisError(
            "perfect_fit",
            "At least one CADF regression fits exactly or at float64 rounding level; its t statistic is undefined.",
        )
    stat = fit.beta[:, 0] / fit.se[:, 0]
    beta = fit.beta[:, 0] * (target_scale / scales[:, 0])
    se = fit.se[:, 0] * (target_scale / scales[:, 0])
    if (
        not bool(torch.isfinite(stat).all())
        or not bool(torch.isfinite(beta).all())
        or not bool(torch.isfinite(se).all())
    ):
        raise AnalysisError(
            "numerical_failure",
            "A CADF coefficient or statistic is not resolvable in float64; rescale the series.",
        )
    return fit, stat, beta, se, target_scale


def _workspace(n: int, total: int, order: int, trend: str, requested: int, memory_mb: int):
    k = 3 + 2 * order + {"none": 0, "constant": 1, "trend": 2}[trend]
    # Live design, scaled design, QR, projections, auxiliary differences, and temporaries.
    per_unit = 8 * (6 * total * k + 16 * total + 10 * k * k)
    budget = memory_mb * 1024**2
    block = min(n, requested, budget // per_unit)
    if block < 1:
        raise AnalysisError(
            "workspace_too_small",
            "The requested tensor workspace cannot hold one CADF regression; increase memory_mb or restrict the sample.",
        )
    return block, block * per_unit


@_cpu_scope
@torch.no_grad()
def xtcips(
    data: Any,
    y: str,
    panel: str,
    time: str,
    *,
    lags: int | str = 0,
    maxlag: int | None = None,
    trend: str = "constant",
    truncated: bool = True,
    individual: bool = False,
    inference: str = "table",
    block_size: int = 128,
    memory_mb: int = 64,
) -> openecon.DataFrame:
    """Pesaran CADF and CIPS tests with native batched QR and published quantiles.

    Complete common-date panels are required. ``trend`` is ``none``, ``constant``
    or ``trend``. Each regression includes the lagged individual level, lagged
    cross-section mean, current/lagged mean differences and individual lagged
    differences (Pesaran 2007, equation 54). ``lags`` is a common integer or
    ``aic``/``bic``/``hqic``: choose each unit's lag on the common maxlag sample,
    then refit at that lag. Default maxlag is min(8, floor((T_levels-d-5)/3)),
    where d is the number of deterministic columns.

    The default table contains CIPS and truncated CIPS*; ``individual=True``
    instead returns the unit CADF statistics. ``truncated`` selects the primary
    statistic and relevant individual quantiles. Inference uses Tables I/II,
    bilinear N,T interpolation on 10..200 with no extrapolation. Table T equals
    the number of potential one-step differences (levels minus one); the same
    asymptotic reference applies with lag augmentation. Only1%,5%,10% decisions
    are supplied, never fabricated exact p-values. ``inference='none'`` permits
    statistics outside the table range. This implements the one-factor CADF
    test; it is not PANIC or arbitrary multifactor calibration.

    ``memory_mb`` bounds estimated tensor fitting workspace; in-memory input,
    index arrays, per-unit outputs, mean series and private BLAS work are separate.
    """
    check_choice(trend, "trend", ("none", "constant", "trend"))
    check_choice(inference, "inference", ("table", "none"))
    check_flag(truncated, "truncated")
    check_flag(individual, "individual")
    requested = check_count(block_size, "block_size", minimum=1)
    memory_mb = check_count(memory_mb, "memory_mb", minimum=1)
    if requested > 512 or memory_mb > 1024:
        raise AnalysisError("invalid_option", "block_size must be <=512 and memory_mb <=1024.")
    automatic = isinstance(lags, str)
    if automatic:
        check_choice(lags, "lags", ("aic", "bic", "hqic"))
        if maxlag is not None:
            maxlag = check_count(maxlag, "maxlag")
    else:
        lags = check_count(lags, "lags")
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag applies only to automatic lag selection.")
    values, labels, date_range = _sample(data, y, panel, time)
    n, total = values.shape
    nominal_t = total - 1
    if inference == "table":
        critical_values("cips", trend, n, nominal_t, truncated)
    deterministics = {"none": 0, "constant": 1, "trend": 2}[trend]
    limit = (
        (min(8, max(0, (nominal_t - deterministics - 4) // 3)) if maxlag is None else maxlag)
        if automatic
        else lags
    )
    if nominal_t - limit <= 3 + 2 * limit + deterministics:
        raise AnalysisError(
            "invalid_lags",
            "The largest lag order leaves too few observations for the full CADF design.",
        )
    if trend != "none":
        values -= values[:, :1].clone()  # unit intercepts absorb this, including in the mean.
    if not bool(torch.isfinite(values).all()):
        raise AnalysisError(
            "numerical_failure", "Removing initial levels overflowed float64; rescale the series."
        )
    mean = (values / n).sum(dim=0)
    block, estimated = _workspace(n, total, limit, trend, requested, memory_mb)
    orders = torch.full((n,), limit, dtype=torch.int64)
    statistics = torch.empty(n, dtype=torch.float64)
    coefficients, standard_errors = torch.empty_like(statistics), torch.empty_like(statistics)
    for first in range(0, n, block):
        last = min(n, first + block)
        series = values[first:last]
        if automatic:
            best = torch.full((last - first,), float("inf"), dtype=torch.float64)
            for order in range(limit + 1):
                fit, _, _, _, target_scale = _fit(series, mean, order, trend, limit)
                penalty = {
                    "aic": 2.0,
                    "bic": float(torch.log(torch.tensor(fit.nobs, dtype=torch.float64))),
                    "hqic": float(
                        2 * torch.log(torch.log(torch.tensor(fit.nobs, dtype=torch.float64)))
                    ),
                }[lags]
                score = (
                    fit.nobs * (torch.log(fit.ssr / fit.nobs) + 2 * torch.log(target_scale))
                    + penalty * fit.k
                )
                better = score < best
                orders[first:last][better] = order
                best = torch.minimum(best, score)
        for order in torch.unique(orders[first:last]).tolist():
            local = (orders[first:last] == order).nonzero().flatten()
            _, stat, beta, se, _ = _fit(series[local], mean, order, trend, order)
            (
                statistics[first + local],
                coefficients[first + local],
                standard_errors[first + local],
            ) = stat, beta, se
    lower, upper = TRUNCATION[trend]
    cut = statistics.clamp(lower, upper)
    raw, clipped = float(statistics.mean()), float(cut.mean())
    selected = clipped if truncated else raw
    rows, index = [], []
    if individual:
        quantiles = (
            critical_values("cadf", trend, n, nominal_t, truncated) if inference == "table" else {}
        )
        for i in range(n):
            stat = float(cut[i] if truncated else statistics[i])
            rows.append(
                [
                    float(statistics[i]),
                    float(cut[i]),
                    float(coefficients[i]),
                    float(standard_errors[i]),
                    int(orders[i]),
                    nominal_t - int(orders[i]),
                    *[quantiles.get(level) for level in ("1%", "5%", "10%")],
                    *[
                        (stat < quantiles[level]) if level in quantiles else None
                        for level in ("1%", "5%", "10%")
                    ],
                ]
            )
        index = labels
        columns = [
            "cadf",
            "cadf_truncated",
            "level_coefficient",
            "std_error",
            "lags",
            "nobs",
            "critical_1pct",
            "critical_5pct",
            "critical_10pct",
            "reject_1pct",
            "reject_5pct",
            "reject_10pct",
        ]
    else:
        for name, stat, is_truncated in (("cips", raw, False), ("cips_truncated", clipped, True)):
            quantiles = (
                critical_values("cips", trend, n, nominal_t, is_truncated)
                if inference == "table"
                else {}
            )
            rows.append(
                [
                    stat,
                    *[quantiles.get(level) for level in ("1%", "5%", "10%")],
                    *[
                        (stat < quantiles[level]) if level in quantiles else None
                        for level in ("1%", "5%", "10%")
                    ],
                ]
            )
            index.append(name)
        columns = [
            "statistic",
            "critical_1pct",
            "critical_5pct",
            "critical_10pct",
            "reject_1pct",
            "reject_5pct",
            "reject_10pct",
        ]
    notes = [
        "One-factor residual structure and a nonzero average factor loading underpin the cross-section-mean augmentation; arbitrary multifactor dependence is not calibrated.",
        "Critical values are published nonstandard left-tail quantiles, not normal/t distributions or exact p-values; interpolated quantiles are approximate.",
        "The finite table uses potential differences T=levels-1. Lagged-difference extensions use the same asymptotic reference; no finite-T automatic-lag-selection adjustment is supplied.",
    ]
    if trend == "none":
        notes.append(
            "No-intercept finite-T tables use zero initial levels. Supplied levels are not centered; a nonzero initial cross-section mean can affect finite-T similarity. Choose deterministics and initial conditions consistent with the null model."
        )
    if inference == "none":
        notes.append("No critical values or rejection decisions were requested.")
    return table(
        rows,
        columns=columns,
        index=index,
        label="Cross-sectionally augmented panel unit-root test",
        series=y,
        test="cadf" if individual else "cips",
        statistic=selected,
        cips=raw,
        cips_truncated=clipped,
        p_value=None,
        distribution="Pesaran nonstandard",
        h0="All panels contain a unit root",
        ha="A nonzero fraction of panels is stationary",
        n_panels=n,
        periods=total,
        critical_T=nominal_t,
        nobs=n * total,
        balanced=True,
        dates=date_range,
        trend=trend,
        truncated=truncated,
        truncation=[lower, upper],
        truncated_panels=int((cut != statistics).sum()),
        inference=inference,
        lag_method=lags if automatic else "fixed",
        lags_by_panel=orders,
        cadf_by_panel=statistics,
        panel_labels=labels,
        notes=notes,
        workspace={
            "algorithm": "batched_scaled_householder_qr",
            "panel_block_size": block,
            "estimated_tensor_workspace_bytes": estimated,
            "budget_bytes": memory_mb * 1024**2,
            "excludes_in_memory_input_indices_outputs_and_private_blas": True,
        },
        provenance={
            "method_source": SOURCE,
            "equation": 54,
            "critical_value_tables": ["I(a-c)", "II(a-c)"],
            "dtype": "float64",
            "device": "cpu",
            "stata_parity_validated": False,
        },
    )
