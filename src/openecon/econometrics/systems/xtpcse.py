"""Linear regression with panel-corrected standard errors: Stata's ``xtpcse``.

Model and estimator
-------------------
``y_it = x_it'b + e_it`` estimated by OLS, or, with ``correlation='ar1'`` /
``'psar1'``, by Prais-Winsten regression after estimating a common or
panel-specific AR(1) coefficient from the OLS residuals (``panelgls``; the
common ``rho`` weights the panel coefficients by ``T_i - 1``, by ``T_i`` with
``np1``).

Panel-corrected covariance (Beck and Katz 1995; Methods and formulas of
[XT] xtpcse)
    ``V = (X'X)^-1 X'Omega X (X'X)^-1`` with ``Omega = Sigma (x) I_T`` estimated from
    the (transformed) OLS residuals: ``sigma_ij = e_i'e_j / T_ij``. For
    unbalanced panels ``X'Omega X = sum_t X_t' Sigma_t X_t`` over the panels
    observed in period ``t``, and ``Sigma`` uses either the periods common to
    every panel (``casewise``, the default) or, element by element, the
    periods that panels ``i`` and ``j`` share (``pairwise=True``).
    ``hetonly``: ``Sigma`` diagonal (``sigma_ii = e_i'e_i/T_i``, panels
    heteroskedastic but independent); ``independent``: ``Sigma = sigma^2 I`` with
    ``sigma^2 = e'e/N`` (the OLS covariance with divisor N).

The meat never forms ``Omega``: casewise it is ``(1/T_c) sum_{s,t} h_st h_st'`` with
``h_st = sum_i e_is x_it`` (``s`` over the common periods), pairwise
``sum_t X_t' Sigma X_t`` on the dense period x panel grid, whichever is
cheaper; hetonly and independent are weighted cross products.

Reported: z inference, Wald chi2 of the slopes, R-squared of the (transformed)
regression (``1 - RSS/TSS``, TSS centred), the panel structure and ``rho``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.systems.common import EXACT_FIT
from openecon.econometrics.systems.panelgls import (
    PanelLayout, dense, estimate_rho, observed, panel_covariance_plan, panel_layout, panel_summary,
    prais_winsten, require_consecutive,
)
from openecon.engines.covariance import group_sums
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_LIST_LIMIT = 200


def _common_periods(layout: PanelLayout) -> Tensor:
    counts = torch.bincount(layout.tcodes, minlength=layout.periods)
    return (counts == layout.m).nonzero().flatten()


def _check_dense(layout: PanelLayout, width: int) -> None:
    panel_covariance_plan(layout, max(width - 1, 1))


def pcse_meat(x: Tensor, resid: Tensor, layout: PanelLayout, *, hetonly: bool,
              independent: bool, pairwise: bool) -> tuple[Tensor, dict[str, Any]]:
    """``X'Omega X`` and a description of the Sigma estimate."""
    n = x.shape[0]
    panel_covariance_plan(layout, x.shape[1], correlated=not (independent or hetonly))
    if independent:
        sigma2 = float(resid.square().sum()) / n
        return sigma2 * (x.T @ x), {"sigma": "sigma^2 I", "sigma2": sigma2}
    if hetonly:
        if pairwise:
            sums = group_sums(resid.square()[:, None], layout.codes, layout.m)[:, 0]
            variances = sums / layout.sizes
        else:
            common = _common_periods(layout)
            if len(common) == 0:
                raise AnalysisError("no_common_periods", "No period is observed in every panel; "
                                    "use pairwise=True.")
            keep = torch.isin(layout.tcodes, common)
            sums = group_sums((resid.square() * keep)[:, None], layout.codes, layout.m)[:, 0]
            variances = sums / len(common)
        return (x * variances[layout.codes][:, None]).T @ x, {
            "sigma": "diagonal", "variances": variances}
    _check_dense(layout, x.shape[1] + 1)
    grid_x = dense(x, layout)                                 # [T, m, k]
    grid_e = dense(resid, layout)                             # [T, m]
    if not pairwise:
        common = _common_periods(layout)
        if len(common) == 0:
            raise AnalysisError("no_common_periods", "No period is observed in every panel, so "
                                "the casewise Sigma is undefined; use pairwise=True.")
        errors = grid_e[common]                               # [Tc, m]
        sigma = errors.T @ errors / len(common)
        if len(common) < layout.m:
            h = torch.einsum("si,tik->stk", errors, grid_x).reshape(-1, x.shape[1])
            meat = h.T @ h / len(common)
        else:
            meat = torch.einsum("tik,til->kl", grid_x, torch.einsum("ij,tjk->tik", sigma, grid_x))
        return (meat + meat.T) / 2, {"sigma": "casewise", "common_periods": len(common),
                                     "matrix": sigma}
    present = observed(layout)
    counts = present.T @ present
    sigma = torch.where(counts > 0, (grid_e.T @ grid_e) / counts.clamp_min(1),
                        torch.zeros_like(counts))
    meat = torch.einsum("tik,til->kl", grid_x, torch.einsum("ij,tjk->tik", sigma, grid_x))
    return (meat + meat.T) / 2, {"sigma": "pairwise", "unpaired": int((counts == 0).sum()) // 2,
                                 "matrix": sigma}


def fit_xtpcse(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``xtpcse``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    frame.sort_panel()
    layout = panel_layout(frame)
    correlation, rhotype = frame.option("correlation"), frame.option("rhotype")
    hetonly, independent = bool(frame.option("hetonly")), bool(frame.option("independent"))
    pairwise, np1 = bool(frame.option("pairwise")), bool(frame.option("np1"))
    if hetonly and independent:
        raise AnalysisError("invalid_spec", "hetonly and independent are alternatives; choose "
                            "one.")
    if correlation != "independent":
        require_consecutive(layout)
    plan = panel_covariance_plan(layout, frame.design_width(), correlated=not (independent or hetonly),
                                 resident_bytes=frame.resource_input_bytes,
                                 budget_bytes=frame.resource_budget_bytes)
    frame.resource_plans.append(plan.record())
    design = frame.drop_collinear(frame.design())
    x, y = design.x, frame.numeric(spec.outcome)
    k, nobs = x.shape[1], frame.n
    if k == 0:
        raise AnalysisError("empty_design", "Every regressor was omitted; nothing to estimate.")
    if nobs <= k:
        raise AnalysisError("insufficient_observations", f"The model has {k} coefficients but "
                            f"only {nobs} observations.")
    fit = kernel_call(least_squares, x, y, drop_collinear=True, tol=0.0)
    rho = None
    if correlation != "independent":
        rho = estimate_rho(frame, fit.resid, layout, correlation, rhotype, k, np1)
        xs, ys = prais_winsten(x, rho, layout), prais_winsten(y, rho, layout)
        fit = kernel_call(least_squares, xs, ys, drop_collinear=True, tol=0.0)
    else:
        xs, ys = x, y
    if fit.omitted:
        raise AnalysisError("singular_design", "The (transformed) design is rank deficient.")
    rss = float(fit.ssr)
    if rss <= EXACT_FIT * float(ys.square().sum()):
        raise AnalysisError("perfect_fit", "The regressors fit the outcome exactly; the "
                            "panel-corrected covariance is zero.")
    meat, record = pcse_meat(xs, fit.resid, layout, hetonly=hetonly, independent=independent,
                             pairwise=pairwise)
    covariance = fit.xtx_inv @ meat @ fit.xtx_inv
    covariance = (covariance + covariance.T) / 2
    if bool((covariance.diagonal() <= 0).any()):
        raise AnalysisError("invalid_covariance", "The panel-corrected covariance is not "
                            "positive definite (the pairwise Sigma is indefinite); use the "
                            "casewise Sigma or hetonly=True.")
    tss = float((ys - ys.mean()).square().sum())
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(fit.beta, covariance, slopes, label="Wald chi2 test of the slopes")}
    sigma_kind = "independent" if independent else "hetonly" if hetonly else (
        "pairwise" if pairwise else "casewise")
    counts = {"independent": 1, "hetonly": layout.m,
              "pairwise": layout.m * (layout.m + 1) // 2,
              "casewise": layout.m * (layout.m + 1) // 2}[sigma_kind]
    metrics: dict[str, Any] = {
        "r_squared": 1 - rss / tss if tss > 0 else None, **panel_summary(layout),
        "estimated_covariances": counts,
        "estimated_autocorrelations": {"independent": 0, "ar1": 1, "psar1": layout.m}[
            correlation], "estimated_coefficients": k, "df_model": len(slopes)}
    if correlation == "ar1":
        metrics["rho"] = float(rho[0])
    extra: dict[str, Any] = {"correlation": correlation, "rhotype": rhotype, "sigma": sigma_kind,
                             "np1": np1}
    if rho is not None and layout.m <= _LIST_LIMIT:
        extra["rho"] = rho.tolist()
    for key, value in record.items():
        if isinstance(value, Tensor):
            if layout.m <= _LIST_LIMIT:
                extra["sigma_" + key] = value.tolist()
        elif key != "sigma":
            extra[key] = value
    info = {"nobs": nobs, "df_inference": None,
            "correction": f"Beck-Katz panel-corrected (X'X)^-1 X'(Sigma kron I)X (X'X)^-1, "
                          f"Sigma {sigma_kind}"
                          + ("" if correlation == "independent" else
                             f", after Prais-Winsten ({correlation}, rhotype {rhotype})")}
    return build_result(
        frame, terms=design.terms, params=fit.beta, covariance=covariance,
        title="Linear regression, panel-corrected standard errors"
              + ("" if correlation == "independent" else " (Prais-Winsten)"),
        use_t=False, metrics=metrics, fitted=x @ fit.beta, nobs=nobs, inference=info,
        tests=tests, extra=extra, categories=design.categories,
        solver="householder_qr_panel_corrected",
        solver_diagnostics={"condition_number": fit.condition_number})


def xtpcse(*, data: Any, y: str, x: Sequence[str], panel: str, time: str,
           correlation: str = "independent", rhotype: str = "regress", hetonly: bool = False,
           independent: bool = False, pairwise: bool = False, np1: bool = False,
           intercept: bool = True, covariance: str | None = None,
           categorical: Sequence[str] | None = None, missing: str = "raise",
           alpha: float = 0.05) -> ResultBundle:
    """Linear regression with Beck-Katz panel-corrected standard errors (Stata ``xtpcse``).

    Model: ``y_it = x_it'b + e_it`` with errors heteroskedastic across panels
    and contemporaneously correlated (``E[e_it e_jt] = sigma_ij``), optionally
    AR(1) within panels.

    Estimator: OLS (``correlation='independent'``) or Prais-Winsten
    regression with a common (``'ar1'``) or panel-specific (``'psar1'``)
    autocorrelation from the OLS residuals (``rhotype`` as in ``oe.xtgls``;
    the common ``rho`` weights panels by ``T_i - 1``, ``np1=True`` by ``T_i``).
    Covariance ``V = (X'X)^-1 X'(Sigma kron I_T)X (X'X)^-1`` with
    ``sigma_ij = e_i'e_j / T_ij`` from the (transformed) residuals: over the
    periods common to all panels (default casewise) or the periods each pair
    shares (``pairwise=True``); ``hetonly=True``: diagonal Sigma;
    ``independent=True``: ``sigma^2 I`` with ``sigma^2 = e'e/N``.

    Parameters
    ----------
    data, y, x, panel, time : as for ``oe.xtgls``. correlation, rhotype,
        hetonly, independent, pairwise, np1 : see above. covariance :
        ``'robust'`` only (the panel-corrected family). intercept,
        categorical, missing, alpha : as usual.

    Returns
    -------
    z inference; ``metrics``: ``r_squared`` (of the transformed regression
    with AR(1)), ``n_groups``, ``n_periods``, observations per group,
    ``estimated_covariances``, ``estimated_autocorrelations``,
    ``estimated_coefficients``, ``rho`` (ar1); ``tests['model']``: Wald chi2
    of the slopes; ``extra``: ``rho`` per panel, the Sigma estimate.

    Stata: ``xtpcse y x1 x2, correlation(ar1) pairwise``.

    Example::

        import openecon as oe
        fit = oe.xtpcse(data=df, y="invest", x=["mvalue", "kstock"], panel="company",
                        time="year", correlation="ar1")
        print(fit.summary())
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtpcse", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        intercept=intercept, covariance=covariance,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        options={"correlation": None if correlation == "independent" else correlation,
                 "rhotype": None if rhotype == "regress" else rhotype,
                 "hetonly": hetonly or None, "independent": independent or None,
                 "pairwise": pairwise or None, "np1": np1 or None})
    return fit(spec, data=data)
