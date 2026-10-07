"""IV regression with high-dimensional fixed effects: ``ivreghdfe`` (ivreg2 + reghdfe).

Model: ``y = X1 b1 + X2 b2 + sum_d alpha_d[codes_d] + u`` with endogenous
``X2`` and excluded instruments ``Z2``. By Frisch-Waugh-Lovell the absorbed
dummies can be projected out of ``y``, ``X1``, ``X2`` and ``Z2`` first; IV on
the residualized blocks reproduces the dummy-variable IV estimates.

Pipeline
    1. ``drop_singletons`` (default): observations alone in a level of any
       absorbed dimension are dropped iteratively (reghdfe). With frequency
       weights a row with ``f_i >= 2`` stands for several observations and is
       never a singleton, as in the duplicated-row data set.
    2. ``[y X1 X2 Z2]`` are demeaned jointly by conjugate-gradient accelerated
       alternating projections (``engines.absorb.demean``, weighted).
    3. Regressors and instruments without variation left after the absorption,
       or collinear, are omitted (``ModelFrame.drop_absorbed``).
    4. 2SLS, LIML or two-step GMM on the residualized blocks exactly as in
       ``ivregress``; GMM uses the weight matrix of the covariance type
       (ivreg2's ``gmm2s``).
    5. Degrees of freedom lost to the effects follow reghdfe
       (``engines.absorb.absorbed_degrees_of_freedom``): observed levels minus
       redundant coefficients; dimensions nested in ANY cluster column are not
       counted, and when every dimension is nested one degree of freedom is
       added back for the constant. ``K = k + df_absorbed``.
    6. Exact fits are rejected (``perfect_fit``); for two or more dimensions a
       residual sum of squares below ``(32 tolerance)^2 TSS_within`` is demeaning
       error and counts as exact (the reghdfe rule of the linear family).

Statistics
    ``small=True`` (ivreghdfe's default): t and F, ``s^2 = RSS/(N-K)``, robust
    ``N/(N-K)``, cluster ``G/(G-1) (N-1)/(N-K)`` with ``G - 1`` (``G_min - 1``)
    inference degrees of freedom. ``small=False`` (ivreg2 without small): z and
    chi2, ``RSS/N`` and no finite-sample factor at all. No constant is reported.
    Diagnostics (first stage, weak identification, overidentification,
    endogeneity) are those of ``ivregress`` on the residualized data with the
    absorbed degrees of freedom removed from every denominator.
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
    ModelFrame, build_result, column_list, kernel_call, make_spec,
)
from openecon.econometrics.iv.common import (
    Setting, check_fit, check_pweights, estimate, estimation_weights, role_designs, screen,
    sum_of_squares,
)
from openecon.econometrics.iv.diagnostics import Diagnostics
from openecon.econometrics.iv.ivregress import SOLVERS, describe, model_test, resolve_covariance
from openecon.engines.absorb import absorbed_degrees_of_freedom, demean, singleton_mask
from openecon.models import ModelSpec, ResultBundle

_WMATRIX = {"nonrobust": "unadjusted", "robust": "robust", "cluster": "cluster"}
# A within residual below this multiple of the demeaning tolerance (relative to the within
# variation of y) is demeaning error, not data (the same rule as reghdfe).
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
    for several observations and is never a singleton (as in the duplicated-row data)."""
    dimensions = [frame.codes(name) for name in absorb]
    if frame.spec.weight_type == "fweight":
        repeated = frame.weights() >= 2
        if bool(repeated.any()):
            # One extra copy of the codes of every repeated row keeps its levels at a count
            # of at least two, exactly as the replicated observations would.
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
    """reghdfe's (levels, redundant, nested, total) of the absorbed dimensions.

    A dimension counts as nested when ANY cluster column nests it (so the order of the
    cluster columns cannot change the result); the remaining dimensions are ranked among
    themselves as first, second, ... . The kernel takes one cluster column and already
    skips the dimensions nested in it, so it runs once per cluster column; if no single
    run carries the full set of nested flags, the non-nested dimensions are ranked once
    more on their own. This mirrors ``econometrics.linear.reghdfe``.
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


def fit_ivreghdfe(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ivreghdfe``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    method, small = frame.option("method"), bool(frame.option("small"))
    tolerance, max_iterations = _controls(frame)
    absorb = frame.role("absorb")
    dropped = _drop_singletons(frame, absorb) if frame.option("drop_singletons") else 0
    dimensions = [frame.codes(name) for name in absorb]
    weights, nobs = estimation_weights(frame)
    clustered = spec.covariance == "cluster"
    clusters = frame.cluster_dimensions() if clustered else None
    levels, redundant, nested, total = _absorbed_dof(dimensions, clusters or [])
    constant_added = all(nested)
    df_absorbed = total + int(constant_added)
    if nobs - df_absorbed < 1:
        raise AnalysisError(
            "insufficient_observations", f"The absorbed fixed effects use all {nobs} "
            f"observation(s) ({df_absorbed} absorbed degrees of freedom): nothing is left to "
            "estimate the slopes. Absorb fewer or coarser dimensions.")
    y = frame.numeric(spec.outcome)
    exog, endog, instr = role_designs(frame, intercept=False)
    k1, q = exog.x.shape[1], endog.x.shape[1]
    absorbed = kernel_call(demean, torch.cat([y[:, None], exog.x, endog.x, instr.x], dim=1),
                           dimensions, weights, tol=tolerance, max_iter=max_iterations)
    within = absorbed.values
    y_within = within[:, 0]
    blocks = screen(frame, exog, endog, instr, weights,
                    transformed=(within[:, 1:1 + k1], within[:, 1 + k1:1 + k1 + q],
                                 within[:, 1 + k1 + q:]))
    setting = Setting(nobs, weights, spec.covariance, small, group_factor=small,
                      absorbed=df_absorbed, clusters=clusters,
                      cluster_names=registry.cluster_columns(spec) or None)
    est = estimate(frame, setting, y_within, blocks, method=method,
                   wmatrix=_WMATRIX[spec.covariance])
    terms = blocks.terms
    k = len(terms)
    df_resid = nobs - k - df_absorbed
    tss = sum_of_squares(y, weights, centered=True)
    tss_within = sum_of_squares(y_within, weights, centered=False)
    # Iterative demeaning leaves an error of about ten tolerances (relative) in every
    # column, so an exact fit stops near (32 tolerance)^2 TSS_within instead of at rounding.
    resolution = 0.0 if absorbed.method == "within" \
        else (_DEMEANING_NOISE * tolerance) ** 2 * tss_within
    check_fit(est.rss, tss, df_resid, sum_of_squares(y, weights, centered=False),
              resolution=resolution)
    diagnostics = Diagnostics(frame, setting, y_within, blocks, est).run()
    r_squared, r_squared_within = 1 - est.rss / tss, 1 - est.rss / tss_within
    metrics: dict[str, Any] = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1 - (1 - r_squared) * (nobs - 1) / df_resid,
        "r_squared_within": r_squared_within,
        "adjusted_r_squared_within":
            1 - (est.rss / df_resid) / (tss_within / (nobs - df_absorbed)),
        "rmse": (est.rss / (df_resid if small else nobs)) ** 0.5,
    }
    if est.kappa is not None:
        metrics["kappa"] = est.kappa
    metrics.update({"df_model": k, "df_resid": df_resid, "df_absorbed": df_absorbed,
                    "n_instruments": len(blocks.instr.terms),
                    "n_endogenous": len(blocks.endog.terms), "n_singletons_dropped": dropped})
    if est.gmm is not None:
        metrics["j"] = est.gmm["j"]
    info = dict(est.info)
    info.update({
        "df_inference": info.get("df_inference") if small else None, "nobs": nobs,
        "method": method, "small": small, "k_total": k + df_absorbed,
        "absorbed_degrees_of_freedom": df_absorbed,
        "error_variance": "RSS/(N-K)" if small else "RSS/N",
        "degrees_of_freedom_convention": "reghdfe: observed levels minus redundant (connected "
        "components); dimensions nested within any cluster column not counted, one degree of "
        "freedom added for the constant when every dimension is nested",
    })
    extra = describe(blocks, est, diagnostics)
    extra.update({
        "absorbed": [{"column": name, "levels": count, "redundant": lost, "nested": flag}
                     for name, count, lost, flag
                     in zip(absorb, levels, redundant, nested, strict=True)],
        "constant_degree_of_freedom_added": constant_added,
        "iterations": absorbed.iterations, "converged": absorbed.converged,
        "demeaning_method": absorbed.method, "max_update": absorbed.max_update,
        "tolerance": tolerance, "singletons_dropped": dropped,
    })
    return build_result(
        frame, terms=terms, params=est.beta, covariance=est.covariance,
        title=f"IV ({method.upper()}) regression with absorbed fixed effects", use_t=small,
        df_inference=est.info.get("df_inference") if small else None, df_resid=df_resid,
        metrics=metrics, fitted=y - est.resid, nobs=nobs, inference=info,
        tests={"model": model_test(est, terms, small), **diagnostics.tests}, extra=extra,
        categories=blocks.exog.categories, solver="alternating_projections_" + SOLVERS[method],
        solver_diagnostics={
            "first_stage_condition_number": est.proj.first.condition_number,
            "second_stage_condition_number": est.condition_number,
            "demeaning_sweeps": absorbed.iterations, "demeaning_method": absorbed.method})


def ivreghdfe(*, data: Any, y: str, x: Sequence[str] | None = None, endog: Sequence[str],
              instruments: Sequence[str], absorb: Sequence[str], method: str = "2sls",
              covariance: str | None = None, cluster: str | Sequence[str] | None = None,
              small: bool = True, weights: str | None = None, weight_type: str | None = None,
              drop_singletons: bool = True, tolerance: float = 1e-8,
              max_iterations: int = 10000, categorical: Sequence[str] | None = None,
              missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """IV regression absorbing several fixed-effect dimensions (``ivreghdfe``).

    Model
        ``y = x'b1 + endog'b2 + alpha_1[d1] + alpha_2[d2] + ... + u``. The fixed effects
        of every dimension in ``absorb`` are projected out of the outcome, the
        regressors and the instruments (conjugate-gradient alternating projections;
        exact one-pass demeaning for one dimension); 2SLS (default), LIML or two-step
        GMM is then applied to the residualized data, which reproduces the estimates
        of the IV regression with explicit dummies. No ``Intercept`` is reported.
        Regressors or instruments without variation left are omitted and recorded.

    Degrees of freedom (reghdfe)
        ``K = k + df_absorbed``; ``df_absorbed`` sums the observed levels minus the
        redundant coefficients of each dimension (connected components for the second,
        pairwise for later ones). A dimension nested in any cluster column is not
        counted; when all are nested one degree of freedom is added for the constant.

    Parameters
        y, x, endog, instruments: as in :func:`ivregress` (``x`` may be omitted).
        absorb: list of columns to absorb (any scalar labels).
        method: ``'2sls'``, ``'liml'`` or ``'gmm'`` (two-step efficient GMM whose weight
            matrix has the covariance type; with ``'nonrobust'`` it equals 2SLS).
        covariance: ``'nonrobust'`` (default; ``'unadjusted'`` accepted), ``'robust'``,
            ``'cluster'`` (implied by ``cluster``; one or two columns).
        small: ``True`` (default, as ivreghdfe): t/F statistics, ``RSS/(N-K)``, robust
            ``N/(N-K)``, cluster ``G/(G-1) (N-1)/(N-K)``. ``False``: z/chi2 statistics,
            ``RSS/N`` and no finite-sample factor (ivreg2 without ``small``).
        weights, weight_type: ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights
            need robust or cluster; robust is their default).
        drop_singletons: iteratively drop observations alone in a level (default True).
        tolerance, max_iterations: demeaning convergence tolerance and sweep limit.
        categorical: exogenous regressors expanded as dummies. missing, alpha: as usual.

    Errors
        ``empty_sample`` (every row a singleton), ``insufficient_observations`` (the
        fixed effects use every degree of freedom), ``no_endogenous_regressors``,
        ``underidentified``, ``absorption_nonconvergence``, ``invalid_option``,
        ``perfect_fit``, ``constant_outcome``, ``singular_weight_matrix`` (GMM with
        fewer clusters than instruments), ``unsupported_covariance`` (pweights with the
        conventional covariance), ``insufficient_clusters``, ``missing_values``.

    Result
        ``metrics``: r_squared (with the fixed effects), adjusted_r_squared,
        r_squared_within, adjusted_r_squared_within, rmse, kappa (liml), df_model,
        df_resid, df_absorbed, n_instruments, n_endogenous, n_singletons_dropped, j
        (gmm). ``tests`` and ``extra['first_stage']``: as :func:`ivregress`, computed on
        the residualized data. ``extra['absorbed']``: per dimension column, levels,
        redundant, nested; plus the demeaning iterations and tolerance.

    Stata
        ``ivreghdfe y x1 (p = z1 z2), absorb(firm year) cluster(firm)`` is
        ``oe.ivreghdfe(data=df, y='y', x=['x1'], endog=['p'], instruments=['z1', 'z2'],
        absorb=['firm', 'year'], cluster='firm')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"firm": np.repeat(range(60), 8), "year": np.tile(range(8), 60)})
        >>> df["z"] = rng.normal(size=480); v = rng.normal(size=480)
        >>> df["p"] = df.z + 0.1 * df.firm + v
        >>> df["y"] = 1.5 * df.p + 0.3 * df.year + 0.7 * v + rng.normal(size=480)
        >>> print(oe.ivreghdfe(data=df, y="y", endog=["p"], instruments=["z"],
        ...                    absorb=["firm", "year"], cluster="firm").summary())
    """
    spec = make_spec(
        "ivreghdfe", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"endogenous": column_list(endog, "endog"),
                 "instruments": column_list(instruments, "instruments"),
                 "absorb": column_list(absorb, "absorb")},
        options={"method": method, "small": None if small else False,
                 "drop_singletons": drop_singletons, "tolerance": tolerance,
                 "max_iterations": max_iterations})
    return fit(spec, data=data)
