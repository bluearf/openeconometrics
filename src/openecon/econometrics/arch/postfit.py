"""Quantities derived from a fitted ARCH-family model: persistence, state, diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch import densities
from openecon.econometrics.arch.kernels import Evaluation
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arima.diagnostics import (
    archlm_test, default_lags, jarque_bera_test, ljung_box_test,
)

VARIANCE_TAIL = 400        # conditional variances kept in result.extra for charts
ARCH_LM_LAGS = 5           # default lag order of the ARCH-LM test of the standardized residuals


def persistence(layout: Layout, theta: Tensor) -> float | None:
    """How strongly a shock to the variance carries over to the next period.

    - garch / arch / igarch: ``sum a_i + sum b_j`` (exactly 1 for IGARCH);
    - gjr: ``sum a_i + sum g_i / 2 + sum b_j`` (a negative shock has probability 1/2);
    - egarch: ``sum b_j``, the autoregressive coefficient of ln h;
    - parch: ``E|z|^phi sum a_i + sum b_j``, with the absolute moment of the fitted
      innovation distribution (``None`` when that moment does not exist).
    """
    ix = layout.index
    a, b = float(theta[ix.a].sum()), float(theta[ix.b].sum())
    if layout.restricted:
        return 1.0
    if layout.kind == "garch":
        return a + b
    if layout.kind == "gjr":
        return a + 0.5 * float(theta[ix.g].sum()) + b
    if layout.kind == "egarch":
        return b
    tau = float(theta[ix.d[0]]) if ix.d else 0.0
    moment = densities.abs_moment(layout.dist, float(theta[ix.p[0]]), tau)
    return a * moment + b if math.isfinite(moment) else None


def unconditional_variance(layout: Layout, theta: Tensor, persist: float | None) -> float | None:
    """``omega / (1 - persistence)`` for covariance-stationary GARCH and GJR models.

    ``None`` for the other models (EGARCH and power ARCH have no closed form in the
    reported parameters), with variance regressors, and when it does not exist.
    """
    if layout.kind not in {"garch", "gjr"} or layout.restricted or layout.k_z or persist is None:
        return None
    omega = float(theta[layout.index.c])
    return omega / (1.0 - persist) if persist < 1.0 and omega > 0.0 else None


def state_record(layout: Layout, evaluation: Evaluation, last_period: int | None) -> dict[str, Any]:
    """What ``arch_forecast`` needs to continue the recursions after the last observation.

    The last ``max_lag`` innovations e_t, structural disturbances u_t and conditional
    variances h_t (oldest first), plus the presample variance h_0 used in the fit.
    """
    keep = min(layout.max_lag, evaluation.residual.shape[0])
    return {
        "residuals": evaluation.residual[-keep:].tolist(),
        "disturbances": evaluation.disturbance[-keep:].tolist(),
        "variances": evaluation.variance[-keep:].tolist(),
        "presample_variance": evaluation.presample_variance,
        "last_period": last_period,
    }


def diagnostics(evaluation: Evaluation, layout: Layout, test_lags: int | None) -> dict[str, Any]:
    """Tests on the standardized residuals z_t = e_t / sqrt(h_t) of a fitted model.

    ``ljung_box`` (remaining autocorrelation in z), ``ljung_box_squared`` (remaining
    autocorrelation in z^2, i.e. ARCH effects the variance equation missed),
    ``arch_lm_residuals`` (Engle's LM test on z) and ``jarque_bera`` (normality of z).
    Tests the sample is too short for are left out.
    """
    z = evaluation.residual / evaluation.variance.sqrt()
    n = int(z.shape[0])
    tests: dict[str, Any] = {}
    if not bool((z != z[0]).any()):
        return tests
    lags = min(test_lags, n - 1) if test_lags is not None else default_lags(n)
    if 1 <= lags < n:
        tests["ljung_box"] = ljung_box_test(
            z, lags, fitted_parameters=len(layout.ar_lags) + len(layout.ma_lags),
            label=f"Ljung-Box Q({lags}) test of the standardized residuals")
        squares = z.square()
        if bool((squares != squares[0]).any()):
            tests["ljung_box_squared"] = ljung_box_test(
                squares, lags, label=f"Ljung-Box Q({lags}) test of the squared standardized "
                                     "residuals")
    lm_lags = test_lags if test_lags is not None else ARCH_LM_LAGS
    if n >= 2 * lm_lags + 3:
        try:
            tests["arch_lm_residuals"] = archlm_test(
                z, lm_lags, label=f"ARCH-LM({lm_lags}) test of the standardized residuals")
        except AnalysisError:
            pass                 # a degenerate auxiliary regression is not a failure of the fit
    tests["jarque_bera"] = jarque_bera_test(
        z, label="Jarque-Bera normality test of the standardized residuals")
    return tests


def variance_tail(evaluation: Evaluation) -> list[float]:
    """The last ``VARIANCE_TAIL`` conditional variances, oldest first."""
    return evaluation.variance[-VARIANCE_TAIL:].tolist()


def distribution_record(layout: Layout, theta: Tensor, covariance: Tensor,
                        critical: float) -> dict[str, Any]:
    """The fitted innovation distribution with its natural parameter and interval.

    Student t: ``df = 2 + exp(/lndfm2)``; GED: ``shape = exp(/lnshape)``. The interval
    transforms the end points of the normal interval of the reported log parameter, as
    Stata does.
    """
    record: dict[str, Any] = {"name": layout.dist}
    ix = layout.index
    if not ix.d:
        return record
    position = ix.d[0]
    tau = float(theta[position])
    half = critical * math.sqrt(max(float(covariance[position, position]), 0.0))
    name = "df" if layout.dist == "t" else "shape"

    def natural(value: float) -> float | None:
        try:
            return densities.shape_value(layout.dist, value)
        except OverflowError:
            return None

    record.update({name: natural(tau), f"{name}_ci": [natural(tau - half), natural(tau + half)]})
    return record


def standardized_residuals(evaluation: Evaluation) -> Tensor:
    return evaluation.residual / torch.sqrt(evaluation.variance)
