"""Ng--Perron M and Kapetanios--Shin--Snell tests with published quantiles.

Only the discrete lower-tail critical values are calibrated. Neither procedure
has a Student-t p-value; the auxiliary regression's t ratio is nonstandard.
"""

from __future__ import annotations

import math
from typing import Any

from pandas.api.types import is_datetime64_any_dtype
import torch

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.unitroot.common import (
    adf_design,
    check_choice,
    check_count,
    load_series,
    require_variation,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.series import nested_adf
from openecon.resources import plan_workspace, workspace_budget_bytes

NG_SOURCE = "https://www.columbia.edu/~sn2294/pub/ecta01.pdf"
KSS_SOURCE = "https://www.econ.ed.ac.uk/papers/id69_esedps.pdf"
LEVELS = ("1%", "5%", "10%")
# Ng & Perron (2001), Econometrica 69, Table I, p. 1524.
NG_CRITICAL = {
    "constant": {
        "MZa": (-13.8, -8.1, -5.7),
        "MZt": (-2.58, -1.98, -1.62),
        "MSB": (0.174, 0.233, 0.275),
        "MPT": (1.78, 3.17, 4.45),
    },
    "trend": {
        "MZa": (-23.8, -17.3, -14.2),
        "MZt": (-3.42, -2.91, -2.62),
        "MSB": (0.143, 0.168, 0.185),
        "MPT": (4.03, 5.48, 6.67),
    },
}
# Shin & Snell (2000), Edinburgh discussion paper 69, Table 1, p. 5;
# also Kapetanios, Shin & Snell (2003), J. Econometrics 112, 359--379.
KSS_CRITICAL = {
    "none": (-2.82, -2.22, -1.92),
    "constant": (-3.48, -2.93, -2.66),
    "trend": (-3.93, -3.40, -3.13),
}


def _prepare(data, y, time, order, memory_mb, work, name):
    """Plan tensor/selected-input buffers before constructing them.

    Coercion to a resident DataFrame precedes this plan. Caller input, that
    coercion, result Python objects and private BLAS memory are not RSS-bounded.
    """
    if not isinstance(y, str) or (time is not None and not isinstance(time, str)):
        raise AnalysisError("invalid_spec", "y and time must name columns.")
    memory_mb = check_count(memory_mb, "max_memory_mb", minimum=1)
    work = check_count(work, "max_work", minimum=1)
    frame = _coerce_frame(data)
    n = len(frame)
    if n < 8:
        raise AnalysisError("insufficient_observations", f"{name} needs at least 8 levels.")
    if order is None:
        order = min(int(12 * ((n - 1) / 100) ** 0.25), (n - 3) // 2)
    order = check_count(order, "lags")
    if n - 1 - order <= order + 1:
        raise AnalysisError("insufficient_observations", "Use fewer lags or more observations.")
    estimate = 4 * n * (order + 2) ** 2 + 36 * n
    if estimate > work:
        raise AnalysisError(
            "work_limit",
            f"{name} needs {estimate:,} planned row-column products; increase max_work or reduce lags/sample.",
        )
    plan = plan_workspace(
        name,
        {
            "selected input and level transforms": 8 * n * 16,
            "QR, selected regression and lag design": 8 * n * (order + 3) * 8,
            "triangular fits and result buffers": 8 * (order + 3) ** 2 * 8,
        },
        budget_bytes=min(memory_mb * 1024**2, workspace_budget_bytes()),
    )
    if time is not None and time in frame and is_datetime64_any_dtype(frame[time].dtype):
        raise AnalysisError(
            "invalid_time",
            "Use consecutive integer periods for this test; datetime spacing is not inferred.",
        )
    values, labels = load_series(frame, [y], time)
    x = values[:, 0].contiguous()
    require_variation(x, 8, name)
    return x, labels, order, plan.record(), estimate


def _row(name, statistic, values):
    return {
        "test": name,
        "statistic": statistic,
        "p_value": None,
        **{
            f"critical_{level.rstrip('%')}pct": cv for level, cv in zip(LEVELS, values, strict=True)
        },
        **{
            f"reject_{level.rstrip('%')}pct": statistic < cv
            for level, cv in zip(LEVELS, values, strict=True)
        },
    }


def _resolved(fit, name):
    if fit.exact or not fit.s2 > 0 or not bool((fit.se > 0).all()):
        raise AnalysisError(
            "perfect_fit",
            f"{name} has unresolved residual variance; a deterministic series has no defined test statistic.",
        )


@torch.no_grad()
def ngperron(
    data: Any,
    y: str,
    *,
    time: str | None = None,
    trend: str = "constant",
    lags: int | None = None,
    maxlag: int | None = None,
    inference: str = "none",
    max_memory_mb: int = 256,
    max_work: int = 100_000_000,
):
    """Ng--Perron MZa/MZt/MSB/MPT statistics; calibrated inference pending.

    H0: a unit root; H1: level/trend stationarity. All four reject in the lower
    tail. trend='constant' uses GLS c=-7, 'trend' c=-13.5. n levels are y_0..y_T
    with T=n-1, GLS alpha=1+c/T, and the unweighted initial GLS observation.
    lags fixes the AR augmentation; otherwise MAIC compares 0..maxlag on one
    common T-maxlag sample, then refits T-selected_lags observations. Default
    maxlag=min(floor(12*(T/100)**.25), floor((T-2)/2)). The AR spectral estimate
    uses SSR/(T-lags)/(1-sum(phi))**2, including the lag-zero candidate.

    Missing/repeated/gapped integer calendars and singular/exact fits fail.
    Datetime calendars require explicit conversion to consecutive integers.
    inference='none' returns statistics only, p_value=None and no critical
    values or rejection decisions. Published-table calibration did not pass
    independent replication; inference='published' refuses until MARKET-136
    resolves that discrepancy. This procedure cannot currently decide H0.
    Resource limits plan complete resident tensor buffers and QR work, excluding
    caller input, initial DataFrame coercion, Python/BLAS/allocator overhead.
    """
    check_choice(trend, "trend", ("constant", "trend"))
    check_choice(inference, "inference", ("none", "published"))
    if inference == "published":
        raise AnalysisError(
            "critical_values_unverified",
            "Ng--Perron published-table calibration is unresolved (MARKET-136). Use inference='none' for statistics only.",
        )
    if lags is not None and maxlag is not None:
        raise AnalysisError("invalid_lags", "Supply fixed lags or MAIC maxlag, not both.")
    fixed = lags is not None
    if maxlag is not None:
        maxlag = check_count(maxlag, "maxlag")
    x, labels, upper, plan, work = _prepare(
        data, y, time, lags if fixed else maxlag, max_memory_mb, max_work, "Ng--Perron"
    )
    n, c = len(x), -7.0 if trend == "constant" else -13.5
    horizon = n - 1
    shifted = x - x[0]
    scale = float(shifted.abs().max())
    shifted = shifted / scale
    z = torch.ones((n, 1), dtype=torch.float64)
    if trend == "trend":
        z = torch.cat([z, torch.arange(n, dtype=torch.float64)[:, None] / horizon], dim=1)
    alpha = 1 + c / horizon
    quasi_z = torch.cat([z[:1], z[1:] - alpha * z[:-1]])
    quasi_y = torch.cat([shifted[:1], shifted[1:] - alpha * shifted[:-1]])
    detrending = ols(quasi_z, quasi_y, "Ng--Perron GLS detrending")
    u = shifted - z @ detrending.beta
    if float(u.square().sum()) <= 1e-22 * float(shifted.square().sum()):
        raise AnalysisError("perfect_fit", "GLS detrending leaves a deterministic series.")
    selection = []
    selected = upper
    if not fixed:
        for fit in nested_adf(u, upper, "n", "Ng--Perron MAIC"):
            if fit.exact or fit.ssr <= 0:
                raise AnalysisError("perfect_fit", "A MAIC candidate has unresolved variance.")
            variance = fit.ssr / fit.nobs
            tau = fit.coefficient**2 * fit.level_ss / variance
            maic = math.log(variance) + 2 * (tau + fit.lags) / fit.nobs
            selection.append({"lags": fit.lags, "maic": maic, "nobs": fit.nobs})
        selected = min(selection, key=lambda row: row["maic"])["lags"]
    design, target, _ = adf_design(u, selected, "n")
    ar = ols(design, target, "Ng--Perron AR spectral regression")
    _resolved(ar, "Ng--Perron AR spectral regression")
    lag_sum = float(ar.beta[1:].sum())
    if abs(1 - lag_sum) <= 1e-10 * max(1.0, float(ar.beta[1:].abs().sum())):
        raise AnalysisError(
            "invalid_covariance",
            "The AR spectral denominator is unresolved; reduce lags or inspect the series.",
        )
    spectral = ar.ssr / ar.nobs / (1 - lag_sum) ** 2
    integral = float(u[:-1].square().sum()) / horizon**2
    endpoint = float(u[-1]) ** 2 / horizon
    if not spectral > 0 or not integral > 0:
        raise AnalysisError(
            "invalid_covariance", "Ng--Perron needs positive resolved level and long-run variances."
        )
    mza = (endpoint - spectral) / (2 * integral)
    msb = math.sqrt(integral / spectral)
    mpt = (c**2 * integral + (-c if trend == "constant" else 1 - c) * endpoint) / spectral
    statistics = {"MZa": mza, "MZt": mza * msb, "MSB": msb, "MPT": mpt}
    if not all(math.isfinite(v) for v in statistics.values()):
        raise AnalysisError(
            "numerical_failure",
            "Ng--Perron statistics are numerically unresolved; rescale the input.",
        )
    return table(
        [{"test": name, "statistic": value, "p_value": None} for name, value in statistics.items()],
        test="ngperron",
        label="Ng--Perron GLS M tests",
        statistic=mza,
        statistics=statistics,
        p_value=None,
        critical_values=None,
        reject=None,
        inference=inference,
        calibration_status="unresolved_published_table_replication",
        distribution=None,
        null="Unit root",
        alternative="Level stationarity" if trend == "constant" else "Trend stationarity",
        trend=trend,
        lags=selected,
        lag_selection="fixed" if fixed else "maic",
        maxlag=upper if not fixed else None,
        lag_selection_results=selection,
        selection_nobs=horizon - upper if not fixed else None,
        nobs=ar.nobs,
        n_levels=n,
        horizon=horizon,
        gls_c=c,
        gls_alpha=alpha,
        ar_lag_sum=lag_sum,
        normalized_long_run_variance=spectral,
        normalization_scale=scale,
        time_start=labels.iloc[0].item()
        if hasattr(labels.iloc[0], "item")
        else int(labels.iloc[0]),
        time_end=labels.iloc[-1].item()
        if hasattr(labels.iloc[-1], "item")
        else int(labels.iloc[-1]),
        calibration_source=NG_SOURCE,
        calibration_table="Table I, p. 1524",
        dtype="float64",
        device="cpu",
        resource_plan=plan,
        estimated_work=work,
        notes=[
            "Statistics only: published critical-value calibration failed independent replication; no rejection decisions are supplied (MARKET-136).",
            "Short-memory innovations and adequate AR augmentation are required; underfitting can distort size.",
        ],
    )


@torch.no_grad()
def kss(
    data: Any,
    y: str,
    *,
    time: str | None = None,
    lags: int = 0,
    trend: str = "constant",
    max_memory_mb: int = 256,
    max_work: int = 100_000_000,
):
    """Kapetanios--Shin--Snell nonlinear unit-root test (fixed augmentation).

    H0: a unit root; H1: globally stationary ESTAR dynamics. trend='none' uses
    raw levels (zero-mean/vanishing initial condition), 'constant' OLS demeans,
    'trend' OLS demeans/detrends over all levels before lag construction.
    Regress Delta z_t on z_(t-1)**3 and lags lagged differences, with no added
    constant/trend; return the cubic coefficient's classical OLS t ratio.
    Its distribution is nonstandard: published lower-tail 1/5/10% critical
    values, p_value=None. It is not a general test of nonlinearity.

    Requires complete consecutive integer calendars; no row dropping or date
    frequency inference. Positive scaling before cubing avoids overflow without
    changing the statistic. Coefficient/standard error use these normalized units.
    Resource limits exclude caller input, initial coercion and private memory.
    """
    check_choice(trend, "trend", ("none", "constant", "trend"))
    lags = check_count(lags, "lags")
    x, _, lags, plan, work = _prepare(data, y, time, lags, max_memory_mb, max_work, "KSS")
    n = len(x)
    shifted = x if trend == "none" else x - x[0]
    scale = float(shifted.abs().max())
    u = shifted / scale
    if trend != "none":
        z = torch.ones((n, 1), dtype=torch.float64)
        if trend == "trend":
            z = torch.cat([z, torch.arange(n, dtype=torch.float64)[:, None] / (n - 1)], dim=1)
        detrending = ols(z, u, "KSS OLS detrending")
        detrended = u - z @ detrending.beta
        if float(detrended.square().sum()) <= 1e-22 * float(u.square().sum()):
            raise AnalysisError("perfect_fit", "OLS detrending leaves a deterministic series.")
        u = detrended
    design, target, _ = adf_design(u, lags, "n")
    design[:, 0] = design[:, 0].pow(3)
    fit = ols(design, target, "KSS cubic auxiliary regression")
    _resolved(fit, "KSS cubic auxiliary regression")
    statistic = float(fit.beta[0] / fit.se[0])
    if not math.isfinite(statistic):
        raise AnalysisError(
            "numerical_failure", "KSS statistic is numerically unresolved; rescale the input."
        )
    cvs = KSS_CRITICAL[trend]
    return table(
        [_row("KSS", statistic, cvs)],
        test="kss",
        label="KSS nonlinear unit-root test",
        statistic=statistic,
        p_value=None,
        critical_values=dict(zip(LEVELS, cvs, strict=True)),
        reject={level: statistic < cv for level, cv in zip(LEVELS, cvs, strict=True)},
        distribution="KSS asymptotic lower-tail distribution",
        null="Unit root",
        alternative="Globally stationary ESTAR dynamics",
        trend=trend,
        lags=lags,
        lag_selection="fixed",
        nobs=fit.nobs,
        n_levels=n,
        df_resid=fit.df,
        normalized_cubic_coefficient=float(fit.beta[0]),
        normalized_cubic_std_error=float(fit.se[0]),
        normalization_scale=scale,
        calibration_source=KSS_SOURCE,
        calibration_table="Table 1, p. 5 (Edinburgh discussion paper 69)",
        dtype="float64",
        device="cpu",
        resource_plan=plan,
        estimated_work=work,
        notes=[
            "Asymptotic critical values only; no Student-t p-value or finite-sample size guarantee.",
            "Use adequate augmentation for short-memory innovations; raw levels require the zero-mean initial-condition contract.",
        ],
    )
