"""Residual Phillips--Ouliaris Za/Zt with calibrated response surfaces."""

from __future__ import annotations

import math
from typing import Any
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot.common import (
    check_choice,
    check_count,
    column_names,
    deterministic_columns,
    load_series,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot import po_tables as calibration
from openecon.engines.distributions import normal_cdf

_CASES = {"none": "n", "constant": "c", "trend": "ct", "quadratic": "ctt"}


def phillips_ouliaris(
    data: Any,
    y: str,
    x: list[str],
    *,
    time: str | None = None,
    test: str = "Zt",
    trend: str = "constant",
    kernel: str = "bartlett",
    bandwidth: int | None = None,
    inference: str = "calibrated",
) -> TableSet:
    """Za or Zt test of H0: no cointegration; negative tails reject.

    Bartlett/Parzen/Quadratic Spectral long-run variance uses uncentered AR(1)
    innovations with denominator original T. Default bandwidth is the recorded
    Schwert/Newey-West rule, floor(4*(T/100)**(2/9)); supply it for replication.
    Calibration retains arch 8.0.0 response surfaces for 2..6 series and its
    supported minimum sample. Smaller samples require inference='none'.
    Complete regular calendars and full-rank cointegrating regressions are
    required. All computation is float64 Torch; arch is not a runtime dependency.
    """
    check_choice(test, "test", ("Za", "Zt"))
    check_choice(trend, "trend", tuple(_CASES))
    check_choice(kernel, "kernel", ("bartlett", "parzen", "quadratic_spectral"))
    check_choice(inference, "inference", ("calibrated", "none"))
    names = column_names(x, "x")
    if not 1 <= len(names) <= 5:
        raise AnalysisError(
            "critical_values_unavailable", "PO supports one to five regressors (2..6 series)."
        )
    values, _ = load_series(data, [y, *names], time)
    n = len(values)
    if n > 1_000_000:
        raise AnalysisError(
            "workspace_limit", "Resident PO calibration is limited to 1,000,000 observations."
        )
    case = _CASES[trend]
    key = (test, case, len(names) + 1)
    if inference == "calibrated" and n < calibration.CV_TAU_MIN[key]:
        raise AnalysisError(
            "critical_values_unavailable",
            "The sample is below this response surface's calibrated minimum; use inference='none'.",
        )
    if bandwidth is None:
        bandwidth = int(4 * (n / 100) ** (2 / 9))
    bandwidth = check_count(bandwidth, "bandwidth")
    if bandwidth >= n - 1:
        raise AnalysisError("invalid_lags", "bandwidth must be smaller than T-1.")
    covariance_lags = n - 2 if kernel == "quadratic_spectral" and bandwidth else bandwidth
    if n * covariance_lags > 100_000_000:
        raise AnalysisError(
            "work_limit",
            "PO kernel work exceeds 100,000,000 observation-lag products; reduce the sample/bandwidth or use a finite-support kernel.",
        )
    centered = values - values[0] if case != "n" else values
    deterministic, _ = deterministic_columns(n, case)
    design = torch.stack(
        [*(centered[:, j] for j in range(1, len(names) + 1)), *deterministic], dim=1
    )
    fit = ols(design, centered[:, 0], "PO cointegrating regression")
    u = fit.resid
    denominator = float(u[:-1].square().sum())
    if denominator <= 1e-22 * float(centered[:, 0].square().sum()):
        raise AnalysisError("perfect_fit", "The cointegrating residual variance is unresolved.")
    ar = float(u[:-1] @ u[1:]) / denominator
    innovation = u[1:] - ar * u[:-1]
    short = float(innovation.square().sum()) / n
    correction = 0.0
    last = n - 2 if kernel == "quadratic_spectral" and bandwidth else bandwidth
    for lag in range(1, last + 1):
        ratio = lag / (bandwidth + 1) if kernel != "quadratic_spectral" else lag / bandwidth
        if kernel == "bartlett":
            weight = 1 - ratio
        elif kernel == "parzen":
            weight = 1 - 6 * ratio**2 + 6 * ratio**3 if ratio <= 0.5 else 2 * (1 - ratio) ** 3
        else:
            z = 6 * math.pi * ratio / 5
            weight = 3 / z**2 * (math.sin(z) / z - math.cos(z))
        correction += weight * float(innovation[lag:] @ innovation[:-lag]) / n
    long_run = short + 2 * correction
    if not long_run > 0:
        raise AnalysisError("invalid_covariance", "PO long-run variance must be positive.")
    adjusted = ar - 1 - n * correction / denominator
    statistic = n * adjusted if test == "Za" else adjusted / math.sqrt(long_run / denominator)
    p_value, critical = None, None
    if inference == "calibrated":
        if statistic < calibration.PVAL_TAU_MIN[key]:
            p_value = 0.0
        elif statistic > calibration.PVAL_TAU_MAX[key]:
            p_value = 1.0
        else:
            coefficients = (
                calibration.PVAL_SMALL_P
                if statistic <= calibration.PVAL_TAU_STAR[key]
                else calibration.PVAL_LARGE_P
            )[key]
            p_value = normal_cdf(sum(c * statistic**j for j, c in enumerate(coefficients)))
        critical = {
            f"{level}%": sum(c / n**j for j, c in enumerate(coefficients))
            for level, coefficients in calibration.CV_PARAMETERS[key].items()
        }
    attrs = dict(
        test="phillips_ouliaris",
        statistic=statistic,
        p_value=p_value,
        critical_values=critical,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        nobs=n,
        ar_nobs=n - 1,
        n_series=len(names) + 1,
        long_run_variance=long_run,
        short_run_variance=short,
        one_sided_strict=correction,
        inference=inference,
        calibration_source="arch 8.0.0 Phillips--Ouliaris response surfaces (NCSA)",
        calibration_min_n=calibration.CV_TAU_MIN[key],
        dtype="float64",
        device="cpu",
        null="No cointegration",
        notes=["Asymptotic p-value; finite-sample response-surface critical values."],
    )
    return TableSet(
        {
            "test": table(
                [
                    {
                        "test": test,
                        "statistic": statistic,
                        "p_value": p_value,
                        **{f"critical_{k}": v for k, v in (critical or {}).items()},
                    }
                ]
            )
        },
        **attrs,
    )


def po_za(data: Any, y: str, x: list[str], **kwargs) -> TableSet:
    """Phillips--Ouliaris Z-alpha; see phillips_ouliaris."""
    return phillips_ouliaris(data, y, x, test="Za", **kwargs)


def po_zt(data: Any, y: str, x: list[str], **kwargs) -> TableSet:
    """Phillips--Ouliaris Z-t; see phillips_ouliaris."""
    return phillips_ouliaris(data, y, x, test="Zt", **kwargs)
