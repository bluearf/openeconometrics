"""Linear regression with high-dimensional fixed effects: Correia's ``reghdfe``.

Model and estimator
-------------------
``y_i = x_i'b + sum_d alpha_d[codes_d(i)] + u_i`` for ``D`` absorbed
dimensions. The joint dummy space is projected out of ``y`` and ``X`` by the
method of alternating projections accelerated with conjugate gradients
(``engines.absorb.demean``: symmetric Kaczmarz sweeps, weighted when weights
are given; exact one-pass within transform for ``D = 1``). Least squares on
the demeaned block then reproduces the dummy-variable slopes
(Frisch-Waugh-Lovell). Regressors absorbed by the fixed effects are omitted
(``ModelFrame.drop_absorbed``, within versus mean-deviated sum of squares at
1e-13). No constant is reported: it is absorbed by the first fixed-effect
dimension.

Pipeline (reghdfe 2017)
-----------------------
1. ``drop_singletons`` (default): observations alone in a level of any
   dimension are removed iteratively (``engines.absorb.singleton_mask``); they
   carry no information about the slopes and bias clustered standard errors
   downward. With frequency weights a row stands for ``f_i`` observations, so
   a row with ``f_i >= 2`` is never a singleton and protects its levels (the
   level counts are weighted, as ftools' ``drop_singletons`` does with
   fweights), which keeps the result identical to the duplicated-row data set.
   The count is recorded as a warning.
2. Demean ``[y X]`` jointly to ``tolerance`` within ``max_iterations`` sweeps.
3. Omit absorbed and collinear regressors; QR least squares.
4. Degrees of freedom lost to the fixed effects
   (``engines.absorb.absorbed_degrees_of_freedom``): the observed levels of
   each dimension minus the redundant coefficients - zero for the first
   dimension, the number of connected components ("mobility groups") of the
   bipartite level graph for the second, the pairwise maximum for later ones.
   With a cluster covariance a dimension whose levels nest inside ANY cluster
   column is not counted at all (checked against every cluster column, so the
   order of the cluster columns cannot change the result); the remaining
   dimensions are ranked among themselves. When EVERY dimension is nested one
   degree of freedom is added back for the constant they absorb (reghdfe's
   nested adjustment, which makes ``absorb(id) cluster(id)`` agree with
   ``xtreg, fe vce(cluster id)``, where ``K = k + 1``).
   ``K = k_slopes + df_absorbed`` and ``df_resid = N - K``.
5. Exact fits are rejected (``perfect_fit``): the residual sum of squares is
   compared with the rounding floor of the outcome and, for two or more
   dimensions, with ``(32 tolerance)^2 TSS_within``, the error the iterative
   demeaning itself leaves in the residuals. A fit whose within R-squared
   exceeds ``1 - (32 tolerance)^2`` cannot be resolved at that tolerance.

Covariance
----------
``nonrobust`` = ``RSS/df_resid (X~'WX~)^-1``; ``HC1`` (``robust``) times
``N/(N-K)``; ``cluster`` = CR1 ``G/(G-1) (N-1)/(N-K)``; two cluster columns
use Cameron-Gelbach-Miller inclusion-exclusion with the smallest ``G``
(reghdfe's convention) and ``G_min - 1`` inference degrees of freedom; an
indefinite two-way meat has its negative eigenvalues set to zero
(``inference['psd_adjusted']``, with a warning), as reghdfe does.

Reported
--------
``r_squared`` (overall, with the fixed effects), ``adjusted_r_squared``
``1 - (1-R^2)(N-1)/df_resid``, ``r_squared_within`` and
``adjusted_r_squared_within`` ``1 - (RSS/df_resid)/(TSS_within/(N - df_absorbed))``,
``rmse``, ``df_model``, ``df_resid``, ``df_absorbed``, ``n_singletons_dropped``;
``extra['absorbed']`` per dimension (column, levels, redundant, nested) and the
demeaning diagnostics; ``tests['model']`` the F test of the slopes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec,
)
from openecon.econometrics.linear.common import (
    check_fit, check_pweights, covariance_kind, estimation_weights, model_test,
    resolve_covariance, solve_least_squares, solver_diagnostics, sum_of_squares,
)
from openecon.engines.absorb import absorbed_degrees_of_freedom, demean, singleton_mask
from openecon.models import ModelSpec, ResultBundle

# A within residual below this multiple of the demeaning tolerance (relative to the within
# variation of y) is demeaning error, not data: the fit is exact as far as it can be resolved.
_DEMEANING_NOISE = 32


def _controls(frame: ModelFrame) -> tuple[float, int]:
    tolerance, max_iterations = frame.option("tolerance"), frame.option("max_iterations")
    if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and 0 < tolerance < 1):
        raise AnalysisError("invalid_option", "tolerance must be a number in (0, 1), e.g. 1e-8.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations < 1:
        raise AnalysisError("invalid_option", "max_iterations must be a positive integer.")
    return float(tolerance), max_iterations


def _drop_singletons(frame: ModelFrame, absorb: list[str]) -> int:
    """Iterative singleton removal; with frequency weights a row with ``f_i >= 2`` stands
    for several observations and can never be a singleton."""
    dimensions = [frame.codes(name) for name in absorb]
    if frame.spec.weight_type == "fweight":
        repeated = frame.weights() >= 2
        if bool(repeated.any()):
            # One extra copy of the codes of every repeated row gives the kernel's level
            # counts the duplicated-row data set's values: such rows keep their levels
            # at a count of at least two, so neither they nor their copies ever drop.
            dimensions = [(torch.cat([codes, codes[repeated]]), count)
                          for codes, count in dimensions]
    keep = kernel_call(singleton_mask, dimensions)[: frame.n]
    dropped = frame.n - int(keep.sum())
    if dropped == frame.n:
        raise AnalysisError("empty_sample", "Every observation is a singleton in some absorbed "
                            "dimension; the fixed effects fit the data exactly. Absorb fewer "
                            "dimensions or set drop_singletons=False.")
    if dropped:
        frame.restrict(keep, f"Dropped {dropped} singleton observation(s) alone in a level of an "
                       "absorbed dimension (reghdfe default; drop_singletons=False keeps them).")
    return dropped


def _absorbed_dof(dimensions: list[tuple[Tensor, int]], clusters: list[tuple[Tensor, int]],
                  ) -> tuple[list[int], list[int], list[bool], int]:
    """reghdfe's (levels, redundant, nested, total) with nesting checked against every
    cluster column; non-nested dimensions are ranked among themselves by the kernel.

    The kernel takes one cluster dimension and already skips the dimensions nested in it
    when ranking the others. A dimension counts as nested here when ANY cluster column
    nests it, so the kernel runs once per cluster column; if one of those runs already
    carries the full set of nested flags its accounting is the answer, otherwise the
    non-nested dimensions are ranked once more on their own.
    """
    if not clusters:
        dof = kernel_call(absorbed_degrees_of_freedom, dimensions)
        return dof.levels, dof.redundant, dof.nested, dof.total
    runs = [kernel_call(absorbed_degrees_of_freedom, dimensions, cluster) for cluster in clusters]
    nested = [any(flags) for flags in zip(*(run.nested for run in runs), strict=True)]
    for run in runs:
        if run.nested == nested:
            return run.levels, run.redundant, nested, run.total
    free = [dimension for dimension, flag in zip(dimensions, nested, strict=True) if not flag]
    ranked = kernel_call(absorbed_degrees_of_freedom, free).redundant if free else []
    levels, redundant, position = runs[0].levels, [], 0
    for count, flag in zip(levels, nested, strict=True):
        if flag:
            redundant.append(count)
        else:
            redundant.append(ranked[position])
            position += 1
    return levels, redundant, nested, sum(levels) - sum(redundant)


def fit_reghdfe(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``reghdfe``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    tolerance, max_iterations = _controls(frame)
    absorb = frame.role("absorb")
    dropped = _drop_singletons(frame, absorb) if frame.option("drop_singletons") else 0
    dimensions = [frame.codes(name) for name in absorb]
    weights, nobs = estimation_weights(frame)
    kind = covariance_kind(frame)
    clustered = kind == "cluster"
    clusters = frame.cluster_dimensions() if clustered else []
    cluster_names = registry.cluster_columns(spec)
    y = frame.numeric(spec.outcome)
    design = frame.design(intercept=False)
    block = torch.cat([y[:, None], design.x], dim=1)
    absorbed = kernel_call(demean, block, dimensions, weights, tol=tolerance,
                           max_iter=max_iterations)
    y_within, x_within = absorbed.values[:, 0], absorbed.values[:, 1:]
    design, x_within = frame.drop_absorbed(design, x_within, weights)
    if not design.terms:
        raise AnalysisError("empty_design", "Every regressor is collinear with the absorbed fixed "
                            "effects; nothing is left to estimate.")
    k_slopes = len(design.terms)
    levels, redundant, nested, total = _absorbed_dof(dimensions, clusters)
    constant_added = bool(dimensions) and all(nested)
    df_absorbed = total + int(constant_added)
    k_total = k_slopes + df_absorbed
    df_resid = nobs - k_total
    fitted = solve_least_squares(x_within, y_within, weights)
    rss = float(fitted.ssr)
    tss = sum_of_squares(y, weights, centered=True)
    tss_within = sum_of_squares(y_within, weights, centered=False)
    # Iterative demeaning leaves an error of up to ~10 tolerance (relative) in y~ and in X~ b,
    # so an exact fit ends with RSS near (20 tolerance)^2 TSS_within rather than at rounding.
    resolution = 0.0 if absorbed.method == "within" \
        else (_DEMEANING_NOISE * tolerance) ** 2 * tss_within
    check_fit(rss, tss, df_resid, sum_of_squares(y, weights, centered=False), absorbed=True,
              resolution=resolution)
    covariance, info = linear_covariance(
        frame, x=x_within, resid=fitted.resid, bread=fitted.xtx_inv, n=nobs, k=k_total,
        df_resid=df_resid, weights=weights, ssr=rss, kind=kind, clusters=clusters or None,
        cluster_names=cluster_names if clustered else None)
    r_squared, r_squared_within = 1 - rss / tss, 1 - rss / tss_within
    metrics = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (nobs - 1) / df_resid,
        "r_squared_within": r_squared_within,
        "adjusted_r_squared_within": 1 - (rss / df_resid) / (tss_within / (nobs - df_absorbed)),
        "rmse": (rss / df_resid) ** 0.5, "df_model": k_slopes, "df_resid": df_resid,
        "df_absorbed": df_absorbed, "n_singletons_dropped": dropped,
    }
    test = model_test(frame, fitted.beta, covariance, list(range(k_slopes)),
                      df_inference=info["df_inference"],
                      classical=(tss_within - rss, k_slopes, rss, df_resid),
                      label="Model F test (slopes)")
    info.update({"nobs": nobs, "k_total": k_total, "absorbed_degrees_of_freedom": df_absorbed,
                 "degrees_of_freedom_convention": "reghdfe: observed levels minus redundant "
                 "(connected components); dimensions nested within any cluster column not "
                 "counted, one degree of freedom added for the constant when every dimension "
                 "is nested"})
    extra = {
        "absorbed": [{"column": name, "levels": count, "redundant": lost, "nested": flag}
                     for name, count, lost, flag
                     in zip(absorb, levels, redundant, nested, strict=True)],
        "constant_degree_of_freedom_added": constant_added,
        "iterations": absorbed.iterations, "converged": absorbed.converged,
        "method": absorbed.method, "max_update": absorbed.max_update, "tolerance": tolerance,
        "singletons_dropped": dropped,
    }
    return build_result(
        frame, terms=design.terms, params=fitted.beta, covariance=covariance, nobs=nobs,
        df_inference=info["df_inference"], df_resid=df_resid, metrics=metrics,
        fitted=y - fitted.resid, inference=info, tests={"model": test},
        categories=design.categories, solver="alternating_projections_householder_qr",
        solver_diagnostics=solver_diagnostics(fitted, demeaning_sweeps=absorbed.iterations,
                                              demeaning_method=absorbed.method),
        extra=extra)


def reghdfe(*, data: Any, y: str, x: Sequence[str], absorb: Sequence[str],
            covariance: str | None = None, cluster: str | Sequence[str] | None = None,
            weights: str | None = None, weight_type: str | None = None,
            categorical: Sequence[str] | None = None, drop_singletons: bool = True,
            tolerance: float = 1e-8, max_iterations: int = 10000, missing: str = "raise",
            alpha: float = 0.05) -> ResultBundle:
    """Linear regression absorbing several fixed-effect dimensions, Correia's ``reghdfe``.

    Model
        ``y = x'b + alpha_1[d1] + alpha_2[d2] + ... + u``. The fixed effects of every
        dimension in ``absorb`` are projected out of ``y`` and ``X`` by conjugate-
        gradient accelerated alternating projections (exact one-pass demeaning for a
        single dimension); the slopes are then least squares on the demeaned data,
        identical to the dummy-variable regression. No ``Intercept`` is reported: the
        constant is absorbed by the fixed effects (reghdfe reports none either).
        Regressors without within variation are omitted and recorded.

    Degrees of freedom (reghdfe convention)
        ``K = k + df_absorbed`` where ``df_absorbed`` sums, over dimensions, the observed
        levels minus the redundant coefficients: 0 for the first dimension (it also
        absorbs the constant), the number of connected components of the bipartite
        level graph for the second (Abowd-Creecy-Kramarz), the pairwise maximum for
        later ones (reghdfe's ``dofadjustments(pairwise)``). With a cluster covariance a
        dimension nested within ANY cluster column is not counted (so the order of the
        cluster columns is irrelevant); when every dimension is nested, one degree of
        freedom is added back for the constant (so ``absorb(id) cluster(id)`` matches
        ``xtreg, fe vce(cluster id)``). ``df_resid = N - K``; t tests use ``df_resid``
        (``G_min - 1`` with clusters).

    Parameters
        data: DataFrame, mapping of columns or list of row records.
        y: outcome column. x: list of regressor columns (a bare string is an error).
        absorb: list of columns to absorb (any scalar labels).
        covariance: ``'nonrobust'`` (default; ``RSS/df_resid (X~'WX~)^-1``), ``'HC1'`` /
            ``'robust'`` (``N/(N-K)``), ``'cluster'`` (CR1 ``G/(G-1)(N-1)/(N-K)``; two
            cluster columns use inclusion-exclusion with the smallest ``G``, made
            positive semidefinite when needed with ``inference['psd_adjusted']``).
        cluster: one cluster column, or two for two-way clustering.
        weights, weight_type: ``'aweight'``, ``'fweight'`` (replicated observations:
            also for the singleton rule) or ``'pweight'`` (needs HC1 or cluster; HC1 is
            their default).
        drop_singletons: iteratively drop observations alone in a level (default True,
            as reghdfe); the number dropped is recorded in ``warnings`` and metrics.
        tolerance, max_iterations: convergence tolerance (relative update, default 1e-8)
            and sweep limit of the demeaning.
        categorical: regressors expanded as ``name[level]`` dummies (first level omitted).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing model inputs.
        alpha: significance level of the confidence intervals.

    Result
        ``metrics``: r_squared (overall, with fixed effects), adjusted_r_squared,
        r_squared_within, adjusted_r_squared_within, rmse, df_model, df_resid,
        df_absorbed, n_singletons_dropped. ``tests['model']``: F test of the slopes
        (classical or Wald-robust). ``extra['absorbed']``: per dimension column, levels,
        redundant, nested; plus demeaning iterations, convergence, method and tolerance.

    Errors
        ``empty_sample`` (every row a singleton), ``empty_design`` (every regressor
        absorbed), ``absorption_nonconvergence``, ``invalid_option``,
        ``insufficient_observations``, ``perfect_fit``, ``constant_outcome``,
        ``unsupported_covariance`` (pweights with the conventional covariance),
        ``missing_values``, ``invalid_spec``.

    Stata
        ``reghdfe y x1 x2, absorb(firm year) vce(cluster firm)`` is
        ``oe.reghdfe(data=df, y='y', x=['x1', 'x2'], absorb=['firm', 'year'], cluster='firm')``.

    Example
        >>> import pandas as pd, numpy as np, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"firm": np.repeat(range(50), 6), "year": np.tile(range(6), 50)})
        >>> df["x"] = rng.normal(size=300)
        >>> df["y"] = 0.2 * df.firm + 0.5 * df.year + 1.5 * df.x + rng.normal(size=300)
        >>> print(oe.reghdfe(data=df, y="y", x=["x"], absorb=["firm", "year"]).summary())
    """
    spec = make_spec(
        "reghdfe", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"absorb": column_list(absorb, "absorb")},
        options={"drop_singletons": drop_singletons, "tolerance": tolerance,
                 "max_iterations": max_iterations})
    return fit(spec, data=data)
