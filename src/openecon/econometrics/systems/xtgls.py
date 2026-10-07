"""Panel feasible GLS with heteroskedastic, correlated and autocorrelated panels: Stata's ``xtgls``.

Model
-----
``y_it = x_it'b + e_it`` for panels ``i = 1..m`` with ``E[e e'] = Omega``:

* ``panels='iid'``: ``Omega = sigma^2 I``;
* ``panels='heteroskedastic'``: ``Var(e_it) = sigma_i^2``, panels independent;
* ``panels='correlated'``: ``E[e_it e_jt] = sigma_ij`` (contemporaneous
  correlation, ``Omega = Sigma (x) I_T``; balanced panels required);
* ``corr='ar1'`` / ``'psar1'``: ``e_it = rho_i e_i,t-1 + u_it`` with a common or
  panel-specific ``rho``.

Estimator (Methods and formulas of [XT] xtgls)
----------------------------------------------
1. OLS; with AR(1), ``rho_i`` from the OLS residuals by ``rhotype`` (default
   ``regress``; ``panelgls``), averaged for ``ar1``, and the Prais-Winsten
   transformation of ``[y X]`` within panels, followed by OLS on the
   transformed data.
2. The variance structure from the (transformed) residuals: ``sigma^2 = e'e/N``,
   ``sigma_i^2 = e_i'e_i/T_i`` or ``sigma_ij = e_i'e_j/T``.
3. GLS on the transformed data: ``b = (X*'Omega^-1 X*)^-1 X*'Omega^-1 y*`` with
   ``V = (X*'Omega^-1 X*)^-1``, solved by QR after whitening (weights
   ``1/sigma_i^2``, or ``Sigma^-1/2`` applied across panels period by period).
4. ``igls`` iterates steps 1-3 from the GLS estimates (``rho`` re-estimated
   from the GLS residuals) until ``max_j |b_j - b_j,old| / (|b_j,old| + 1)``
   is below ``tolerance``; without autocorrelation the limit is the ML
   estimator.

Inference is z with a Wald chi2 test of the slopes. ``log_likelihood`` is the
Gaussian log likelihood at the final estimates with the variance structure of
their residuals, reported without autocorrelation only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.systems.common import EXACT_FIT
from openecon.econometrics.systems.kernels import cholesky_factor
from openecon.econometrics.systems.panelgls import (
    PanelLayout, estimate_rho, panel_layout, panel_summary, prais_winsten, require_consecutive,
)
from openecon.engines.covariance import group_sums
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_LIST_LIMIT = 200


def _solve(x: Tensor, y: Tensor, weights: Tensor | None = None):
    fit = kernel_call(least_squares, x, y, weights, drop_collinear=True, tol=0.0)
    if fit.omitted:
        raise AnalysisError("singular_design", "The transformed design is rank deficient.")
    return fit


def _panel_variances(resid: Tensor, layout: PanelLayout) -> Tensor:
    return group_sums(resid.square()[:, None], layout.codes, layout.m)[:, 0] / layout.sizes


def _correlated_sigma(resid: Tensor, layout: PanelLayout) -> Tensor:
    grid = resid.reshape(layout.m, layout.periods)
    return grid @ grid.T / layout.periods


def _whiten(values: Tensor, factor: Tensor, layout: PanelLayout) -> Tensor:
    """``(C^-1 (x) I_T)`` applied to panel-major rows: C^-1 across panels in every period."""
    block = values if values.ndim == 2 else values[:, None]
    grid = block.reshape(layout.m, layout.periods, -1)
    solved = torch.linalg.solve_triangular(
        factor, grid.reshape(layout.m, -1), upper=False).reshape(grid.shape)
    out = solved.reshape(-1, block.shape[1])
    return out if values.ndim == 2 else out[:, 0]


def _gls(x: Tensor, y: Tensor, resid: Tensor, layout: PanelLayout, panels: str, nobs: int
         ) -> tuple[Tensor, Tensor, Any]:
    """GLS under the variance structure estimated from ``resid``; returns (b, V, structure)."""
    if panels == "iid":
        sigma2 = float(resid.square().sum()) / nobs
        fit = _solve(x, y)
        return fit.beta, fit.xtx_inv * sigma2, sigma2
    if panels == "heteroskedastic":
        variances = _panel_variances(resid, layout)
        if bool((variances <= 0).any()):
            raise AnalysisError("perfect_fit", "Some panel has zero residual variance, so its "
                                "GLS weight is infinite. Drop that panel or use panels='iid'.")
        fit = _solve(x, y, 1 / variances[layout.codes])
        return fit.beta, fit.xtx_inv, variances
    sigma = _correlated_sigma(resid, layout)
    factor = kernel_call(cholesky_factor, sigma)
    fit = _solve(_whiten(x, factor, layout), _whiten(y, factor, layout))
    return fit.beta, fit.xtx_inv, sigma


def _log_likelihood(resid: Tensor, layout: PanelLayout, panels: str, nobs: int) -> float | None:
    two_pi = math.log(2 * math.pi)
    if panels == "iid":
        return -0.5 * nobs * (two_pi + math.log(float(resid.square().sum()) / nobs) + 1)
    if panels == "heteroskedastic":
        variances = _panel_variances(resid, layout)
        return float(-0.5 * (layout.sizes * (two_pi + variances.log() + 1)).sum())
    sign, logdet = torch.linalg.slogdet(_correlated_sigma(resid, layout))
    if float(sign) <= 0:
        return None
    return -0.5 * (nobs * (two_pi + 1) + layout.periods * float(logdet))


def fit_xtgls(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``xtgls``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    frame.sort_panel()
    layout = panel_layout(frame)
    panels, corr = frame.option("panels"), frame.option("corr")
    rhotype, igls = frame.option("rhotype"), bool(frame.option("igls"))
    if panels == "correlated":
        if not layout.balanced:
            raise AnalysisError("unbalanced_panel", "panels='correlated' needs balanced panels "
                                "(every panel observed in every period), as in Stata.")
        if layout.periods < layout.m:
            raise AnalysisError("insufficient_periods", f"panels='correlated' estimates an "
                                f"{layout.m} x {layout.m} covariance from {layout.periods} "
                                "periods; it needs at least as many periods as panels.")
    if corr != "independent":
        require_consecutive(layout)
    design = frame.drop_collinear(frame.design())
    x, y = design.x, frame.numeric(spec.outcome)
    k, nobs = x.shape[1], frame.n
    if k == 0:
        raise AnalysisError("empty_design", "Every regressor was omitted; nothing to estimate.")
    if nobs <= k:
        raise AnalysisError("insufficient_observations", f"The model has {k} coefficients but "
                            f"only {nobs} observations.")
    ols = _solve(x, y)
    if float(ols.ssr) <= EXACT_FIT * float(y.square().sum()):
        raise AnalysisError("perfect_fit", "The regressors fit the outcome exactly; the error "
                            "variance is zero.")
    beta, rho = ols.beta, None
    tolerance, limit = float(frame.option("tolerance")), int(frame.option("max_iterations"))
    iterations, converged = 0, not igls
    while True:
        if corr == "independent":
            xs, ys = x, y
        else:
            rho = estimate_rho(frame, y - x @ beta, layout, corr, rhotype, k)
            xs, ys = prais_winsten(x, rho, layout), prais_winsten(y, rho, layout)
        reference = beta if iterations or corr == "independent" else _solve(xs, ys).beta
        new_beta, covariance, structure = _gls(xs, ys, ys - xs @ reference, layout, panels, nobs)
        iterations += 1
        change = float(((new_beta - beta).abs() / (beta.abs() + 1)).max())
        beta = new_beta
        if not igls:
            break
        if iterations > 1 and change <= tolerance:
            converged = True
            break
        if iterations >= limit:
            break
    if not converged:
        raise AnalysisError("nonconvergence", f"Iterated GLS did not converge in {iterations} "
                            "iterations; raise max_iterations or set igls=False.")
    final_resid = ys - xs @ beta
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model": wald_test(beta, covariance, slopes, label="Wald chi2 test of the slopes")}
    counts = {"iid": 1, "heteroskedastic": layout.m,
              "correlated": layout.m * (layout.m + 1) // 2}[panels]
    autocorrelations = {"independent": 0, "ar1": 1, "psar1": layout.m}[corr]
    metrics: dict[str, Any] = {
        **panel_summary(layout), "estimated_covariances": counts,
        "estimated_autocorrelations": autocorrelations, "estimated_coefficients": k,
        "log_likelihood": _log_likelihood(final_resid, layout, panels, nobs)
        if corr == "independent" else None,
        "df_model": len(slopes)}
    if corr == "ar1":
        metrics["rho"] = float(rho[0])
    extra: dict[str, Any] = {"panels": panels, "corr": corr, "rhotype": rhotype, "igls": igls,
                             "iterations": iterations}
    if rho is not None and layout.m <= _LIST_LIMIT:
        extra["rho"] = rho.tolist()
    if panels == "iid":
        extra["sigma2"] = float(structure)
    elif layout.m <= _LIST_LIMIT:
        extra["sigma"] = structure.tolist()
    extra["panel_levels"] = frame.levels(spec.panel)[:_LIST_LIMIT]
    divisor = {"iid": "e'e/N", "heteroskedastic": "e_i'e_i/T_i", "correlated": "e_i'e_j/T"}
    info = {"nobs": nobs, "df_inference": None,
            "correction": f"GLS covariance (X*'Omega^-1 X*)^-1, Omega from {divisor[panels]}"
                          + ("" if corr == "independent" else
                             f" after Prais-Winsten ({corr}, rhotype {rhotype})")}
    return build_result(
        frame, terms=design.terms, params=beta, covariance=covariance,
        title="Cross-sectional time-series FGLS regression", use_t=False, metrics=metrics,
        fitted=x @ beta, nobs=nobs, inference=info, tests=tests, extra=extra,
        categories=design.categories, solver="feasible_gls_householder_qr",
        solver_diagnostics={"iterations": iterations, "converged": converged})


def xtgls(*, data: Any, y: str, x: Sequence[str], panel: str, time: str,
          panels: str = "iid", corr: str = "independent", rhotype: str = "regress",
          igls: bool = False, intercept: bool = True, covariance: str | None = None,
          categorical: Sequence[str] | None = None, tolerance: float = 1e-7,
          max_iterations: int = 100, missing: str = "raise", alpha: float = 0.05
          ) -> ResultBundle:
    """Panel-data feasible GLS (Stata ``xtgls``; EViews pooled/panel GLS weights).

    Model: ``y_it = x_it'b + e_it`` with a non-spherical error covariance:

    - ``panels='iid'`` (``Omega = sigma^2 I``), ``'heteroskedastic'`` (panel
      variances ``sigma_i^2``) or ``'correlated'`` (contemporaneous covariances
      ``sigma_ij``: ``Omega = Sigma kron I_T``; balanced panels with at least
      as many periods as panels);
    - ``corr='independent'``, ``'ar1'`` (common ``rho``) or ``'psar1'``
      (panel-specific ``rho_i``) within-panel AR(1) disturbances.

    Estimator (Stata's Methods and formulas): OLS residuals give ``rho``
    (``rhotype``: ``'regress'`` default, ``'freg'``, ``'tscorr'``, ``'dw'``,
    ``'theil'``, ``'nagar'``; the common ``rho`` is the ``T_i - 1``-weighted
    mean) and the Prais-Winsten transformation; the transformed OLS residuals
    give ``sigma^2 = e'e/N``, ``sigma_i^2 = e_i'e_i/T_i`` or
    ``sigma_ij = e_i'e_j/T``; then GLS ``b = (X*'Omega^-1 X*)^-1 X*'Omega^-1 y*``
    with ``V = (X*'Omega^-1 X*)^-1``. ``igls=True`` iterates to convergence
    (ML without autocorrelation).

    Parameters
    ----------
    data, y, x : outcome and regressors. panel, time : panel and integer
        time columns (the sample is sorted by them; AR(1) needs consecutive
        periods). panels, corr, rhotype, igls : see above. intercept,
        categorical, missing, alpha : as usual. covariance : ``'nonrobust'``
        only (the GLS covariance). tolerance, max_iterations : ``igls``.

    Returns
    -------
    z inference; ``metrics``: ``n_groups``, ``n_periods``, observations per
    group, ``estimated_covariances``, ``estimated_autocorrelations``,
    ``estimated_coefficients``, ``log_likelihood`` (without autocorrelation),
    ``rho`` (``ar1``); ``tests['model']``: Wald chi2 of the slopes; ``extra``:
    ``rho`` per panel, ``sigma2`` / panel variances / ``Sigma``.

    Stata: ``xtgls y x1 x2, panels(correlated) corr(psar1) rhotype(tscorr)``.

    Example::

        import openecon as oe
        fit = oe.xtgls(data=df, y="invest", x=["mvalue", "kstock"], panel="company",
                       time="year", panels="heteroskedastic", corr="ar1")
        print(fit.summary())
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtgls", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        intercept=intercept, covariance=covariance,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        options={"panels": None if panels == "iid" else panels,
                 "corr": None if corr == "independent" else corr,
                 "rhotype": None if rhotype == "regress" else rhotype, "igls": igls or None,
                 "tolerance": None if tolerance == 1e-7 else tolerance,
                 "max_iterations": None if max_iterations == 100 else max_iterations})
    return fit(spec, data=data)
