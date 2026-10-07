"""Stata's rreg: robust regression by Huber and biweight iteratively reweighted least squares.

The procedure is the one of [R] rreg (Li 1985; Hamilton 1991), in the form that
reproduces Stata's printed results (iteration log, coefficients, standard errors and F
of the manual's ``rreg mpg weight foreign`` example on auto.dta) digit for digit:

1. Screening.  OLS on all observations; observations with Cook's distance
   D_i = e_i^2 h_i / (K s^2 (1 - h_i)^2) > 1 are excluded from everything that follows.
2. Huber iterations.  With the residuals e_i of the current fit and
   M = med |e_i - med(e)| (the median absolute deviation from the median residual),
       w_i = 1 if |e_i| <= 2 M,   2 M / |e_i| otherwise.
   The manual writes this as c_h = 1.345 on the scale s = M / 0.6745, i.e. 1.994 M,
   "about 2 M"; Stata's results correspond to exactly 2 M. Weighted least squares with
   these weights gives new residuals; this repeats until the largest change in any
   weight is below the Huber threshold max(0.05, tolerance).
3. Biweight iterations.  The same loop with Tukey's biweight on u_i = e_i / s,
   s = M / 0.6745,
       w_i = (1 - (u_i / c_b)^2)^2 if |u_i| < c_b,   0 otherwise,   c_b = 4.685 tune / 7,
   until the largest change in any weight is below ``tolerance`` (default 0.01).
   The reported coefficients are those of the last weighted least-squares fit.
4. Standard errors by the pseudovalues of Street, Carroll and Ruppert (1988).  With the
   final weights w_i, the scale s that produced them, the final residuals e_i,
   u_i = e_i / s and psi'(u) = (1 - (u/c_b)^2)(1 - 5 (u/c_b)^2) inside |u| < c_b (0 outside),
       m = mean_i psi'(u_i),   lambda = 1 + (K / (N - K)) (1 - m) / m,
       ytilde_i = yhat_i + (lambda / m) w_i e_i.
   OLS of ytilde on X returns the same coefficients (X'We = 0) and its classical
   covariance is the reported one,
       V = s~^2 (X'X)^-1,   s~^2 = (lambda / m)^2 sum_i (w_i e_i)^2 / (N - K),
   with the model F test and t tests on N - K degrees of freedom.

Every least-squares step is a Householder QR (``linalg.least_squares``); medians are
sort-based. N counts the observations that passed the screening.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.linear.common import check_observations
from openecon.engines import linalg
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle

HUBER_MAD_MULTIPLE = 2.0    # Huber weights start to fall at |e| = 2 M (1.349 scaled units)
BIWEIGHT_C = 4.685
MAD_CONSTANT = 0.6745
HUBER_TOLERANCE = 0.05      # the Huber stage stops at max(0.05, tolerance)
MAX_ITERATIONS = 1000
_LOG_LIMIT = 50
_ZERO_SCALE = 1e-12          # MAD scale / max |y| at or below which the fit is exact


def median(values: Tensor) -> Tensor:
    """Sample median (mean of the two middle order statistics for an even count)."""
    ordered = torch.sort(values).values
    n = ordered.numel()
    return (ordered[(n - 1) // 2] + ordered[n // 2]) / 2


def mad_scale(resid: Tensor) -> float:
    """s = med |e_i - med(e)| / 0.6745, consistent for sigma under normal errors."""
    return float(median((resid - median(resid)).abs())) / MAD_CONSTANT


def huber_weights(u: Tensor, c: float = HUBER_MAD_MULTIPLE * MAD_CONSTANT) -> Tensor:
    """min(1, c / |u|) for scaled residuals u = e / s (default c = 2 * 0.6745: |e| <= 2 M)."""
    return c / u.abs().clamp_min(c)


def biweight_weights(u: Tensor, c: float) -> Tensor:
    ratio = (u / c).square()
    return torch.where(ratio < 1, (1 - ratio).square(), torch.zeros_like(u))


@dataclass
class RobustFit:
    beta: Tensor                 # coefficients of the last weighted least-squares fit
    covariance: Tensor           # pseudovalue covariance s~^2 (X'X)^-1
    weights: Tensor              # final case weights (those of the last fit)
    fitted: Tensor               # x beta
    scale: float                 # MAD scale s that produced the final weights
    rmse: float                  # s~, root mean squared error of the pseudovalue regression
    mean_psi_prime: float        # m
    correction: float            # lambda
    huber_iterations: int
    biweight_iterations: int
    log: list[dict[str, Any]] = field(default_factory=list)


def _scaled(resid: Tensor, floor: float = 0.0) -> tuple[Tensor, float]:
    """(e / s, s) with s the MAD scale; a scale at the rounding level of the outcome is zero."""
    scale = mad_scale(resid)
    if not scale > floor:
        raise KernelError(
            "zero_scale",
            "The median absolute deviation of the residuals is zero (at least half of the "
            "observations are fitted exactly), so robust-regression weights are undefined.")
    return resid / scale, scale


@torch.no_grad()
def robust_regression(x: Tensor, y: Tensor, *, tune: float = 7.0, tolerance: float = 0.01,
                      huber_tolerance: float | None = None,
                      max_iterations: int = MAX_ITERATIONS) -> RobustFit:
    """Steps 2-4 of the module docstring on screened data (float64 ``x`` [n, k], ``y`` [n])."""
    n, k = x.shape
    if n <= k:
        raise KernelError("insufficient_observations", "Robust regression needs more "
                          "observations than parameters.")
    if not (tune > 0 and math.isfinite(tune)) or not (tolerance > 0 and math.isfinite(tolerance)):
        raise KernelError("invalid_option", "tune and tolerance must be positive numbers.")
    if huber_tolerance is None:
        huber_tolerance = max(HUBER_TOLERANCE, tolerance)
    c_biweight = BIWEIGHT_C * tune / 7.0
    ols = linalg.least_squares(x, y, drop_collinear=False)
    resid, beta = ols.resid, ols.beta
    weights = torch.ones_like(y)
    log: list[dict[str, Any]] = []
    counts = {"huber": 0, "biweight": 0}
    iteration, scale = 0, 0.0
    floor = _ZERO_SCALE * float(y.abs().max())
    for stage, threshold in (("huber", huber_tolerance), ("biweight", tolerance)):
        while True:
            if iteration == max_iterations:
                raise KernelError("nonconvergence", f"Robust regression did not converge in "
                                  f"{max_iterations} iterations; increase tolerance or check "
                                  "the data for a degenerate fit.")
            iteration += 1
            counts[stage] += 1
            u, scale = _scaled(resid, floor)
            new = huber_weights(u) if stage == "huber" else biweight_weights(u, c_biweight)
            change = float((new - weights).abs().max())
            weights = new
            try:
                fit = linalg.least_squares(x, y, weights, drop_collinear=False)
            except KernelError as exc:
                if exc.code != "singular_design":
                    raise
                raise KernelError(
                    "singular_design",
                    "So many observations received zero weight that the regressors became "
                    "collinear; increase tune or drop sparse indicator variables.") from exc
            resid, beta = fit.resid, fit.beta
            log.append({"stage": stage, "iteration": iteration, "max_weight_change": change})
            if change < threshold:
                break
    # Pseudovalues (Street, Carroll and Ruppert 1988): psi(u_i) s = w_i e_i with the final
    # weights, m from the biweight psi' at the final residuals on the scale of those weights.
    ratio = (resid / (scale * c_biweight)).square()
    slope = torch.where(ratio < 1, (1 - ratio) * (1 - 5 * ratio), torch.zeros_like(ratio))
    m = float(slope.mean())
    if not m > 0:
        raise KernelError(
            "undefined_pseudovalues",
            "The mean derivative of the biweight psi function is not positive (too many "
            "observations are heavily downweighted), so the pseudovalue standard errors are "
            "undefined; increase tune.")
    correction = 1 + (k / (n - k)) * (1 - m) / m
    variance = (correction / m) ** 2 * float((weights * resid).square().sum()) / (n - k)
    if not variance > 0:
        raise KernelError("perfect_fit", "The robust fit reproduces the outcome exactly, so "
                          "standard errors are undefined.")
    return RobustFit(
        beta=beta, covariance=ols.xtx_inv * variance, weights=weights, fitted=y - resid,
        scale=scale, rmse=math.sqrt(variance), mean_psi_prime=m, correction=correction,
        huber_iterations=counts["huber"], biweight_iterations=counts["biweight"], log=log,
    )


@torch.no_grad()
def cooks_distance(x: Tensor, y: Tensor) -> Tensor:
    """Cook's D_i = e_i^2 h_i / (K s^2 (1 - h_i)^2) of the OLS fit, s^2 = RSS / (N - K)."""
    n, k = x.shape
    fit = linalg.least_squares(x, y, drop_collinear=False, need_leverage=True)
    variance = float(fit.ssr) / (n - k)
    if not variance > 0:
        raise KernelError("perfect_fit", "The outcome is fitted exactly by least squares; robust "
                          "regression has nothing to downweight.")
    gap = 1 - fit.leverage
    distance = fit.resid.square() * fit.leverage / (k * variance * gap.square())
    # An observation with leverage one is fitted exactly (e = 0): its distance is undefined
    # and it is kept, as Stata keeps observations whose Cook's D is missing.
    return torch.where(gap > 1e-12, distance, torch.zeros_like(distance))


def fit_rreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``rreg``."""
    frame = ModelFrame(spec, data)
    tune, tolerance = float(frame.option("tune")), float(frame.option("tolerance"))
    if not tune > 0:
        raise AnalysisError("invalid_option", "tune must be positive (Stata's default is 7).")
    if not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tolerance must lie between 0 and 1 (Stata's "
                            "default is 0.01).")
    design = frame.drop_collinear(frame.design())
    if not design.terms:
        raise AnalysisError("no_regressors", "The model has no estimable term: keep the intercept "
                            "or add predictors.")
    check_observations(frame, len(design.terms))
    y = frame.numeric(spec.outcome)
    distance = kernel_call(cooks_distance, design.x, y)
    gross = distance > 1
    dropped = int(gross.sum())
    if dropped:
        frame.restrict(~gross, f"Excluded {dropped} observation(s) with Cook's distance above 1 "
                               "in the initial least-squares fit (rreg's screening step).")
        if frame.n <= len(design.terms):
            raise AnalysisError(
                "insufficient_observations",
                f"After excluding {dropped} observation(s) with Cook's distance above 1 only "
                f"{frame.n} remain for {len(design.terms)} coefficients; robust regression "
                "needs more observations than parameters.")
        # Rows are gone: rebuild on the screened sample and screen collinearity again.
        design = frame.drop_collinear(Design(design.x[~gross], design.terms, design.categories,
                                             design.intercept))
        y = y[~gross]
    x = design.x
    n, k = x.shape
    fit = kernel_call(robust_regression, x, y, tune=tune, tolerance=tolerance)
    slopes = list(range(int(design.intercept), k))
    tests = {}
    if slopes:
        tests["model"] = wald_test(fit.beta, fit.covariance, slopes, df_resid=n - k,
                                   label="F test of the slopes (pseudovalue regression)")
    iterations = fit.huber_iterations + fit.biweight_iterations
    metrics = {"rmse": fit.rmse, "scale": fit.scale, "iterations": iterations,
               "huber_iterations": fit.huber_iterations,
               "biweight_iterations": fit.biweight_iterations, "n_dropped_cooks": dropped,
               "df_model": len(slopes), "df_resid": n - k}
    weights = fit.weights
    extra = {
        "tune": tune, "tolerance": tolerance,
        "huber_cutoff_in_mad": HUBER_MAD_MULTIPLE,
        "huber_c": HUBER_MAD_MULTIPLE * MAD_CONSTANT,
        "biweight_c": BIWEIGHT_C * tune / 7.0,
        "huber_tolerance": max(HUBER_TOLERANCE, tolerance),
        "weights": {"min": float(weights.min()), "mean": float(weights.mean()),
                    "max": float(weights.max()), "n_zero": int((weights == 0).sum()),
                    "n_below_half": int((weights < 0.5).sum())},
        "pseudovalues": {"mean_psi_prime": fit.mean_psi_prime, "lambda": fit.correction},
        "iteration_log": fit.log[-_LOG_LIMIT:],
    }
    info = {"covariance": "nonrobust", "df_inference": n - k,
            "correction": "pseudovalues of Street, Carroll and Ruppert (1988): "
                          "(lambda/m)^2 sum (w e)^2/(N-K) (X'X)^-1, "
                          "lambda = 1 + K/(N-K) (1-m)/m"}
    return build_result(
        frame, terms=design.terms, params=fit.beta, covariance=fit.covariance,
        df_inference=n - k, df_resid=n - k, metrics=metrics, fitted=fit.fitted,
        solver="huber_biweight_irls",
        solver_diagnostics={"iterations": iterations, "converged": True,
                            "final_max_weight_change": fit.log[-1]["max_weight_change"]},
        inference=info, tests=tests, extra=extra, categories=design.categories,
    )


def rreg(*, data: Any, y: str, x: Sequence[str], tune: float = 7, tolerance: float = 0.01,
         covariance: str | None = None, categorical: Sequence[str] | None = None,
         intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Robust regression (Stata's ``rreg``).

    A regression that resists outliers in the outcome: observations with large
    residuals are downweighted, the worst ones to zero. The procedure is
    Stata's (Li 1985; Hamilton 1991):

    1. OLS on all observations; those with Cook's distance > 1 are excluded.
    2. Huber iterations: with M the median absolute deviation of the residuals
       from their median, w_i = min(1, 2M / |e_i|) (the manual's c_h = 1.345
       on the scale s = M / 0.6745, "about 2M"), weighted least squares,
       repeated until the largest change in a weight is below 0.05.
    3. Biweight iterations from that fit: u_i = e_i / s, s = M / 0.6745,
       w_i = (1 - (u_i / c)^2)^2 for |u_i| < c and 0 beyond,
       c = 4.685 * tune / 7 (with the default ``tune=7`` residuals beyond about
       7 MAD get weight zero), repeated until the largest change in a weight is
       below ``tolerance``. The coefficients of the last weighted fit are
       reported.
    4. Standard errors from the pseudovalues of Street, Carroll and Ruppert
       (1988): ytilde_i = yhat_i + (lambda / m) w_i e_i with m the mean of the
       biweight psi'(u_i) and lambda = 1 + (K / (N - K)) (1 - m) / m; the
       classical OLS covariance of ytilde on the regressors,
       (lambda/m)^2 sum (w_i e_i)^2 / (N - K) (X'X)^-1, is reported together
       with its F test.

    These steps reproduce the iteration log, coefficients, standard errors and
    F statistic that the Stata manual prints for ``rreg mpg weight foreign``
    (auto.dta). About 95% as efficient as OLS under normal errors. It does not
    protect against bad leverage points beyond the Cook's distance screen.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    y, x : outcome and regressors; ``categorical`` names regressors to
        treatment-code; ``intercept=False`` removes the constant.
    tune : biweight tuning constant (default 7). Smaller values downweight
        more aggressively and lose efficiency under normal errors.
    tolerance : convergence threshold of the biweight stage on the largest
        change in a case weight (default 0.01). The Huber stage stops at
        max(0.05, tolerance).
    covariance : only ``'nonrobust'`` (the pseudovalue covariance); rreg has no
        other vce and takes no weights, as in Stata.
    missing : ``'raise'`` (default) or ``'drop'``.  alpha : test size.

    Result
    ------
    Student t inference with N - K degrees of freedom, N the number of
    observations after the screening. ``metrics``: ``rmse`` (of the
    pseudovalue regression), ``scale`` (the MAD scale s behind the final
    weights), ``iterations``, ``huber_iterations``, ``biweight_iterations``,
    ``n_dropped_cooks``, ``df_model``, ``df_resid``. ``tests['model']`` is the
    F test of the slopes. ``extra``: a summary of the final weights (min,
    mean, max, number of zero weights), the tuning constants, m and lambda and
    the iteration log (largest weight change per iteration, as Stata prints).
    Stata prints no R-squared for rreg.

    Errors: ``zero_scale`` (MAD of the residuals is zero), ``perfect_fit``,
    ``undefined_pseudovalues``, ``nonconvergence``, ``singular_design``.

    Stata: ``rreg y x1 x2, tune(7)``.

    Example::

        import openecon as oe
        fit = oe.rreg(data=df, y="mpg", x=["weight", "foreign"])
        print(fit.summary())
        print(fit.extra["weights"])
    """
    from openecon.analysis import fit

    spec = make_spec(
        "rreg", outcome=y, predictors=column_list(x, "x"), covariance=covariance,
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        missing=missing, alpha=alpha,
        options={"tune": None if tune == 7 else tune,
                 "tolerance": None if tolerance == 0.01 else tolerance},
    )
    return fit(spec, data=data)
