"""Stata's qreg: linear quantile regression with i.i.d., robust and cluster covariances.

Estimator.  beta(tau) minimizes sum_i w_i rho_tau(y_i - x_i'b) (``kernels.solve``:
Frisch-Newton interior point finished at the exact linear-programming vertex).
Reported with it, as in Stata's header:

    sum_adev   sum_i w_i rho_tau(y_i - x_i'beta)          "Min sum of deviations"
    sum_rdev   sum_i w_i rho_tau(y_i - q)                 "Raw sum of deviations", q the
               order statistic int(tau (N + 1)) of y (``raw_quantile``, the "about" value)
    pseudo_r_squared = 1 - sum_adev / sum_rdev

Covariance (Methods and formulas of [R] qreg; Koenker 2005, sections 3.4 and 4.10).
With N observations, K coefficients, the bandwidth h_N of ``kernels.bandwidth`` and the
density method ``density`` (Stata's denmethod):

nonrobust   vce(iid):  V = tau (1 - tau) s^2 (X'WX)^-1, s the sparsity 1/f(0) of the
            residuals:
              fitted    s = xbar'(beta(tau + h) - beta(tau - h)) / (2 h), two extra
                        quantile regressions evaluated at the regressor means (default);
              residual  s = (F^-1(tau + h) - F^-1(tau - h)) / (2 h) with F^-1 the empirical
                        quantile function of the residuals;
              kernel    f(0) = sum_i w_i K(r_i / c) / (N c),  s = 1 / f(0), with
                        c = min(sd(r), IQR(r) / 1.34) (Phi^-1(tau + h) - Phi^-1(tau - h)).
robust      vce(robust):  V = D^-1 M D^-1,   D = sum_i w_i f_i x_i x_i',
            M = tau (1 - tau) sum_i m_i x_i x_i',  m_i = w_i^2 (f_i for frequency weights,
            1 without weights), with observation-level densities
              fitted    f_i = 2 h / (x_i'beta(tau + h) - x_i'beta(tau - h)), zero where that
                        spacing is not positive (Hendricks and Koenker 1992; default);
              kernel    f_i = K(r_i / c) / c (Powell 1991).
cluster     Parente and Santos Silva (2016):  V = D^-1 (sum_g s_g s_g') D^-1 with the
            D of the robust covariance (kernel by default, as in qreg2) and
            s_g = sum_{i in g} w_i (tau - 1[r_i <= 0]) x_i.

No finite-sample factor is applied; coefficient tests use Student t with N - K degrees
of freedom, as Stata prints. Weights: aweights and pweights are rescaled to sum to N,
fweights replicate observations (N = their sum); pweights need robust or cluster.

Checked against Stata output: the default vce(iid) and vce(robust) standard errors of
the [R] qreg examples on auto.dta, and e(sparsity), e(kbwidth) and the standard errors of
vce(iid, kernel(k) bw) for every kernel and bandwidth rule on Engel's data (the Stata
results shipped with statsmodels' test suite).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, kernel_call, make_spec,
)
from openecon.econometrics.linear.common import (
    check_observations, estimation_weights, weighted_mean,
)
from openecon.econometrics.quantile import kernels
from openecon.engines import covariance as cov
from openecon.engines import linalg
from openecon.engines.distributions import normal_ppf
from openecon.models import ModelSpec, ResultBundle

_BANDWIDTH_ALPHA = 0.05     # z_alpha of the Hall-Sheather and Chamberlain rules


def check_quantile(value: Any, what: str = "quantile") -> float:
    """A quantile as a fraction strictly inside (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise AnalysisError("invalid_quantile", f"The {what} must be a fraction strictly between "
                            "0 and 1 (for example 0.5 for the median).")
    return float(value)


def quantile_title(tau: float) -> str:
    return "Median regression" if tau == 0.5 else f"{tau:g} Quantile regression"


def quantile_sample(frame: ModelFrame) -> tuple[Design, Tensor, Tensor | None, int]:
    """Design (collinear terms omitted), outcome, estimation weights and N."""
    weights, nobs = estimation_weights(frame)
    design = frame.drop_collinear(frame.design(), weights)
    if not design.terms:
        raise AnalysisError("no_regressors", "The model has no estimable term: keep the intercept "
                            "or add predictors.")
    check_observations(frame, len(design.terms))
    return design, frame.numeric(frame.spec.outcome), weights, nobs


def raw_deviations(y: Tensor, tau: float, weights: Tensor | None) -> tuple[float, float]:
    """(raw tau-th quantile of y, weighted check loss about it)."""
    quantile = kernels.raw_quantile(y, tau, weights)
    return quantile, float(kernels.check_loss(y - quantile, tau, weights))


def require_variation(fit: kernels.QuantileFit, raw: float) -> None:
    if fit.exact_fit or fit.objective <= 1e-14 * max(raw, 1e-300):
        raise AnalysisError("perfect_fit", "The outcome is fitted exactly by the regressors, so "
                            "quantile-regression standard errors are undefined. Check for an "
                            "outcome that is a linear function of the predictors.")


def _window(tau: float, h: float, rule: str) -> None:
    if not (tau - h > 0 and tau + h < 1):
        raise AnalysisError(
            "bandwidth_out_of_range",
            f"The {rule} bandwidth h = {h:.4g} reaches outside (0, 1) around the quantile "
            f"{tau:g}. Use more observations, a less extreme quantile, another bandwidth rule "
            "or a bootstrap covariance (oe.bsqreg).")


def _kernel_scale(resid: Tensor, weights: Tensor | None, nobs: int, tau: float, h: float) -> float:
    """c = min(sd(r), IQR(r) / 1.34) (Phi^-1(tau + h) - Phi^-1(tau - h))."""
    centre = weighted_mean(resid, weights)
    squares = (resid - centre).square()
    variance = float(squares.sum() if weights is None else (squares * weights).sum()) / (nobs - 1)
    iqr = (kernels.stata_quantile(resid, 0.75, weights)
           - kernels.stata_quantile(resid, 0.25, weights))
    spread = min(math.sqrt(variance), iqr / 1.34)
    return spread * (normal_ppf(tau + h) - normal_ppf(tau - h))


def _degenerate(what: str) -> AnalysisError:
    return AnalysisError(
        "degenerate_sparsity",
        f"The {what} is not positive, so the standard errors are undefined: the outcome is too "
        "discrete (or the sample too small) around this quantile. Try another density or "
        "bandwidth option, or a bootstrap covariance (oe.bsqreg).")


def _density_method(frame: ModelFrame) -> tuple[str, str]:
    """(density method, kernel) of the requested covariance, with Stata's defaults."""
    spec = frame.spec
    kind = spec.covariance
    method = spec.options.get("density")
    kernel = spec.options.get("kernel")
    if method is None:
        # kernel() selects the kernel method, as in vce(iid, kernel(parzen)).
        method = "kernel" if kernel is not None or kind == "cluster" else "fitted"
    if method != "kernel" and kernel is not None:
        raise AnalysisError("invalid_spec", "kernel applies to density='kernel' only, not to "
                            f"density='{method}'.")
    if method == "residual" and kind != "nonrobust":
        raise AnalysisError(
            "invalid_spec",
            "density='residual' estimates one sparsity for all observations and is available "
            f"for covariance='nonrobust' only; covariance='{kind}' needs density='fitted' or "
            "density='kernel'.")
    return method, kernel or "epanechnikov"


def _covariance(frame: ModelFrame, problem: kernels.QuantileProblem, fit: kernels.QuantileFit,
                x: Tensor, weights: Tensor | None, nobs: int, tau: float,
                ) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Covariance by ``spec.covariance``; returns (V, inference record, density record)."""
    spec = frame.spec
    kind = spec.covariance
    rule = frame.option("bandwidth")
    method, kernel = _density_method(frame)
    h = kernel_call(kernels.bandwidth, tau, nobs, rule, _BANDWIDTH_ALPHA)
    _window(tau, h, rule)
    resid = fit.resid
    record: dict[str, Any] = {"density_method": method, "bandwidth_method": rule, "bandwidth": h,
                              "kernel": kernel if method == "kernel" else None,
                              "kernel_bandwidth": None, "zero_density_observations": None}
    density_weights = None          # w_i f_i of the density-weighted design matrix D
    if method == "fitted":
        low = kernel_call(kernels.solve, problem, tau - h)
        high = kernel_call(kernels.solve, problem, tau + h)
        difference = high.beta - low.beta
        sparsity = float(weighted_mean(x, weights) @ difference) / (2 * h)
        if kind != "nonrobust":
            # Hendricks-Koenker: f_i = 2h / (x_i'b(tau + h) - x_i'b(tau - h)), zero where the
            # two fitted quantiles cross (a nonpositive spacing).
            spacing = x @ difference
            positive = spacing > 1e-10 * float(spacing.abs().max())
            density_weights = torch.where(positive, (2 * h) / spacing.clamp_min(1e-300),
                                          torch.zeros_like(spacing))
            record["zero_density_observations"] = int((~positive).sum())
    elif method == "residual":
        sparsity = (kernels.stata_quantile(resid, tau + h, weights)
                    - kernels.stata_quantile(resid, tau - h, weights)) / (2 * h)
    else:
        c = _kernel_scale(resid, weights, nobs, tau, h)
        if not c > 0:
            raise _degenerate("kernel bandwidth of the residual density")
        density_weights = kernel_call(kernels.kernel_values, resid / c, kernel) / c
        density = float(density_weights.sum() if weights is None
                        else (density_weights * weights).sum()) / nobs
        if not density > 0:
            raise _degenerate("estimated residual density at zero")
        sparsity = 1 / density
        record["kernel_bandwidth"] = c
    if not (sparsity > 0 and math.isfinite(sparsity)):
        raise _degenerate("estimated sparsity")
    record.update({"sparsity": sparsity, "density": 1 / sparsity})
    how = (f"{method} density" if method != "kernel" else f"{kernel} kernel density") \
        + f", {rule} bandwidth"
    info: dict[str, Any] = {"covariance": kind, "df_inference": nobs - x.shape[1]}
    if kind == "nonrobust":
        gram = kernel_call(linalg.weighted_crossprod, x, weights)
        bread = kernel_call(linalg.cholesky_inverse, gram, code="singular_design",
                            what="cross-product matrix of the regressors")
        info["correction"] = f"i.i.d. errors: tau (1 - tau) s^2 (X'X)^-1, sparsity s by {how}"
        return bread * (tau * (1 - tau) * sparsity ** 2), info, record
    if weights is not None:
        density_weights = density_weights * weights
    try:
        bread = kernel_call(linalg.cholesky_inverse,
                            kernel_call(linalg.weighted_crossprod, x, density_weights),
                            code="singular_density_matrix", what="density-weighted design")
    except AnalysisError as exc:
        if exc.code != "singular_density_matrix":
            raise
        advice = ("Use density='kernel' or a bootstrap covariance (oe.bsqreg)."
                  if method == "fitted" else
                  "Use bandwidth='bofinger', kernel='gaussian', density='fitted' or a bootstrap "
                  "covariance (oe.bsqreg).")
        raise AnalysisError(
            "singular_density_matrix",
            "Too few observations have a positive estimated density to form the "
            f"density-weighted design matrix. {advice}") from exc
    if kind == "robust":
        if weights is None:
            square = None
        else:
            square = weights if spec.weight_type == "fweight" else weights.square()
        meat = kernel_call(linalg.weighted_crossprod, x, square) * (tau * (1 - tau))
        name = "Hendricks-Koenker" if method == "fitted" else "Powell kernel"
        info["correction"] = f"{name} sandwich ({how}), no finite-sample factor"
        return kernel_call(cov.sandwich, bread, meat), info, record
    (codes, count), = frame.cluster_dimensions()
    psi = torch.full_like(resid, tau)
    psi[resid <= 0] = tau - 1.0
    if weights is not None:
        psi = psi * weights
    meat = kernel_call(cov.meat_cluster, x * psi[:, None], codes, count)
    info.update({"correction": f"Parente-Santos Silva cluster sandwich ({how}), no "
                               "finite-sample factor",
                 "cluster_count": count, "cluster_df": count - 1,
                 "cluster_columns": [spec.cluster], "cluster_column": spec.cluster})
    return kernel_call(cov.sandwich, bread, meat), info, record


def fit_qreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``qreg``."""
    frame = ModelFrame(spec, data)
    tau = check_quantile(frame.option("quantile"))
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance", "pweights are sampling weights and need "
                            "covariance='robust' (Stata's vce(robust)) or a cluster column.")
    design, y, weights, nobs = quantile_sample(frame)
    x = design.x
    k = x.shape[1]
    problem = kernel_call(kernels.prepare, x, y, weights)
    fit = kernel_call(kernels.solve, problem, tau)
    raw_quantile, sum_rdev = raw_deviations(y, tau, weights)
    require_variation(fit, sum_rdev)
    v, info, density = _covariance(frame, problem, fit, x, weights, nobs, tau)
    if not fit.unique:
        frame.warn("The quantile-regression solution is not unique (several coefficient vectors "
                   "attain the minimum); one optimal vertex is reported.")
    metrics = {
        "quantile": tau,
        "pseudo_r_squared": 1 - fit.objective / sum_rdev if sum_rdev > 0 else None,
        "sum_adev": fit.objective, "sum_rdev": sum_rdev, "raw_quantile": raw_quantile,
        "sparsity": density["sparsity"], "density": density["density"],
        "bandwidth": density["bandwidth"], "kernel_bandwidth": density["kernel_bandwidth"],
        "iterations": fit.iterations, "df_model": k - int(design.intercept), "df_resid": nobs - k,
    }
    extra = {"density_method": density["density_method"],
             "bandwidth_method": density["bandwidth_method"], "kernel": density["kernel"],
             "bandwidth_alpha": _BANDWIDTH_ALPHA, "unique_solution": fit.unique}
    if density["zero_density_observations"] is not None:
        extra["zero_density_observations"] = density["zero_density_observations"]
    return build_result(
        frame, terms=design.terms, params=fit.beta, covariance=v, title=quantile_title(tau),
        df_inference=nobs - k, df_resid=nobs - k, metrics=metrics, fitted=x @ fit.beta,
        solver="frisch_newton_interior_point", solver_diagnostics=solver_record(fit),
        inference=info, extra=extra, categories=design.categories, nobs=nobs,
    )


def solver_record(fit: kernels.QuantileFit) -> dict[str, Any]:
    return {"iterations": fit.iterations, "simplex_pivots": fit.pivots,
            "relative_duality_gap": fit.gap, "exact_vertex": True, "unique": fit.unique}


def qreg(*, data: Any, y: str, x: Sequence[str] | None = None, quantile: float = 0.5,
         covariance: str | None = None, cluster: str | None = None, weights: str | None = None,
         weight_type: str | None = None, density: str | None = None,
         bandwidth: str = "hsheather", kernel: str | None = None,
         categorical: Sequence[str] | None = None, intercept: bool = True,
         missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Quantile regression (Stata's ``qreg``; SPSS Quantile Regression; EViews ``qreg``).

    Model: the ``quantile``-th conditional quantile of ``y`` is linear in the
    regressors, Q_tau(y | x) = x'b(tau). The estimator minimizes the check loss

        sum_i w_i rho_tau(y_i - x_i'b),      rho_tau(u) = u (tau - 1[u < 0]),

    a linear program. It is solved by the Frisch-Newton interior-point method of
    Portnoy and Koenker (1997) and finished at the exact vertex, so the
    reported coefficients interpolate K observations exactly, as a simplex
    solver's would. tau = 0.5 is median (least absolute deviations) regression.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    y : outcome column.
    x : list of regressors (may be empty: the raw quantile); ``categorical``
        names those to treatment-code; ``intercept=False`` removes the constant.
    quantile : tau strictly inside (0, 1); default 0.5.
    covariance : ``'nonrobust'`` (default; Stata's ``vce(iid)``), ``'robust'``
        (``vce(robust)``) or ``'cluster'`` (with ``cluster``; implied when a
        cluster column is given).
    density : how the error density is estimated (Stata's ``denmethod``):
        ``'fitted'`` (default for nonrobust and robust), ``'residual'``
        (nonrobust only) or ``'kernel'`` (default for cluster).
    bandwidth : ``'hsheather'`` (default), ``'bofinger'`` or ``'chamberlain'``.
    kernel : kernel of ``density='kernel'``, one of Stata's ``kdensity``
        kernels: ``'epanechnikov'`` (default), ``'epan2'``, ``'biweight'``,
        ``'cosine'``, ``'gaussian'``, ``'parzen'``, ``'rectangle'``,
        ``'triangle'``. Giving a kernel selects ``density='kernel'`` (Stata's
        ``vce(iid, kernel(parzen))``).
    weights, weight_type : ``'aweight'``, ``'fweight'`` (replicated
        observations; N is their sum) or ``'pweight'`` (needs robust/cluster and
        selects robust by default).
    missing : ``'raise'`` (default) or ``'drop'``.  alpha : test size.

    Covariance conventions
    ----------------------
    With the bandwidth h (on the probability scale; the Hall-Sheather and
    Chamberlain rules use z_alpha at alpha = 0.05 whatever ``alpha`` is):

    - nonrobust: V = tau (1 - tau) s^2 (X'WX)^-1 with the sparsity s = 1/f(0)
      estimated by ``density``: ``fitted`` s = xbar'(b(tau + h) - b(tau - h))/(2h)
      from two extra quantile regressions; ``residual`` the same difference
      quotient of the empirical residual quantiles; ``kernel``
      f(0) = sum_i w_i K(r_i/c)/(N c), c = min(sd(r), IQR(r)/1.34)
      (Phi^-1(tau + h) - Phi^-1(tau - h)).
    - robust: D^-1 [tau (1 - tau) X'X] D^-1 with D = sum_i f_i x_i x_i' and the
      observation-level densities f_i = 2h / (x_i'b(tau + h) - x_i'b(tau - h))
      (``fitted``; zero where the spacing is not positive; Hendricks and
      Koenker 1992) or f_i = K(r_i/c)/c (``kernel``; Powell's sandwich).
    - cluster: the Parente-Santos Silva (2016) covariance (Stata's ``qreg2,
      cluster()``): D as for robust (kernel by default) with the meat
      sum_g s_g s_g', s_g = sum_{i in g} (tau - 1[r_i <= 0]) x_i.

    No finite-sample factor is applied. Tests and confidence intervals use
    Student t with N - K degrees of freedom. Collinear regressors are omitted
    Stata-style and listed in ``warnings``. The default nonrobust and robust
    standard errors reproduce the [R] qreg examples of the Stata manual.

    Result
    ------
    ``metrics``: ``quantile``, ``pseudo_r_squared`` (1 - sum_adev/sum_rdev),
    ``sum_adev`` (minimized sum of check losses), ``sum_rdev`` (the same sum
    about ``raw_quantile``, the order statistic int(tau (N + 1)) of y that
    Stata prints as "about"), ``sparsity``, ``density``, ``bandwidth``,
    ``kernel_bandwidth`` (kernel method), ``iterations``, ``df_model``,
    ``df_resid``. ``extra`` names the density method, bandwidth rule and
    kernel, says whether the minimizer is unique and, for the fitted robust
    covariance, counts the observations whose density was set to zero. Stata's
    qreg prints no model test, so ``tests`` is empty.

    Errors: ``invalid_quantile``, ``bandwidth_out_of_range`` (tau +- h leaves
    (0, 1)), ``degenerate_sparsity``, ``singular_density_matrix``,
    ``perfect_fit``, ``insufficient_observations``.

    Stata: ``qreg y x1 x2, quantile(.25) vce(robust)``.

    Example::

        import openecon as oe
        median = oe.qreg(data=df, y="price", x=["weight", "length", "foreign"])
        print(median.summary())
        q25 = oe.qreg(data=df, y="price", x=["weight", "length"], quantile=0.25,
                      covariance="robust", kernel="gaussian")
    """
    from openecon.analysis import fit

    if covariance is None and weight_type == "pweight" and not cluster:
        covariance = "robust"
    spec = make_spec(
        "qreg", outcome=y, predictors=column_list(x, "x"), covariance=covariance, cluster=cluster,
        categorical=column_list(categorical, "categorical"), intercept=intercept, weights=weights,
        weight_type=weight_type, missing=missing, alpha=alpha,
        options={"quantile": quantile, "density": density,
                 "bandwidth": None if bandwidth == "hsheather" else bandwidth, "kernel": kernel},
    )
    return fit(spec, data=data)
