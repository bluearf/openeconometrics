"""Poisson pseudo-maximum likelihood with high-dimensional fixed effects: ``ppmlhdfe``.

Model (Correia, Guimaraes and Zylkin 2020)
------------------------------------------
``E[y_i | x_i, d_i] = mu_i = exp(x_i'b + sum_d alpha_d[codes_d(i)] + offset_i)``
for any nonnegative outcome (counts, trade flows, expenditures; zeros
included). The Poisson score equations only require the conditional mean to
be right, so the estimator is consistent under heteroskedasticity of unknown
form (Santos Silva and Tenreyro 2006) and, unlike most nonlinear models, free
of the incidental-parameter problem for the slopes in the classical one- and
two-way structures.

Algorithm
---------
IRLS on the Poisson likelihood, never forming a dummy variable. With working
weights ``W = w mu`` and working response ``z = eta - offset + (y - mu)/mu``:

1. partial the fixed effects out of ``[z X]`` in the ``W``-weighted inner
   product (``engines.absorb.demean``; exact for one dimension, conjugate-
   gradient alternating projections for more);
2. weighted QR least squares of ``z~`` on ``X~`` gives ``b``;
3. the fitted value of the full dummy regression is ``z`` minus the residual
   of step 2 (Frisch-Waugh-Lovell), so
   ``eta = offset + z - (z~ - X~ b)`` without ever estimating the effects;
4. halve the step while the deviance rises; stop when its relative change is
   at most ``tolerance`` and no linear predictor is still moving.

The start is ``mu_0 = (y + ybar)/2``.

Sample (as ppmlhdfe)
--------------------
* Observations in a fixed-effect level whose outcomes are ALL zero are
  dropped (iteratively): their effect diverges to minus infinity, they are
  fitted perfectly and carry no information ("separated by a fixed effect").
  ppmlhdfe's general separation search (ReLU / simplex: regressors and
  linear combinations with the fixed effects) is NOT implemented: such
  separation is DETECTED during the iteration (zero-outcome observations
  whose linear predictor keeps falling while the deviance is flat) and
  reported as ``separation_detected`` instead of dropping the observations.
* Singletons are dropped iteratively (``drop_singletons``).

Degrees of freedom and covariance (reghdfe's conventions)
----------------------------------------------------------
``K = k + df_absorbed`` where ``df_absorbed`` counts observed levels minus
redundant ones (connected components) and skips dimensions nested within the
cluster variable (one degree of freedom added back for the constant when all
are nested). With ``u = y - mu``, scores ``x~_i w_i u_i`` and bread
``B = (X~' W X~)^-1`` at the converged weights:

    robust (default)   N/(N-K)              B (sum s_i s_i') B
    cluster            G/(G-1) (N-1)/(N-K)  B (sum_g s_g s_g') B
    nonrobust          B                    (inverse information)

Coefficient tests are z tests; ``tests['model']`` is the Wald chi2 of the slopes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, wald_test,
)
from openecon.econometrics.glm.common import (
    build_spec, check_pweights, count_outcome, likelihood_weights, linear_offset,
)
from openecon.econometrics.glm.kernels import ppml_irls
from openecon.econometrics.linear.common import nested_in_clusters
from openecon.engines.absorb import AbsorbedDF, absorbed_degrees_of_freedom, demean, singleton_mask
from openecon.engines.covariance import group_sums
from openecon.models import ModelSpec, ResultBundle

_DEMEAN_SWEEPS = 10_000


def _singleton_keep(frame: ModelFrame, absorb: list[str]) -> Tensor:
    """Keep-mask of the non-singleton observations of the current sample.

    Frequency weights replicate observations, so a level is a singleton only
    when its total frequency is one (a single row with ``f >= 2`` is two
    identical observations in the duplicated-row data set). One round is
    enough here: the caller iterates until nothing changes.
    """
    dimensions = [frame.codes(name) for name in absorb]
    if frame.spec.weight_type != "fweight":
        return kernel_call(singleton_mask, dimensions)
    frequency = frame.weights()
    keep = torch.ones(frame.n, dtype=torch.bool)
    for codes, count in dimensions:
        keep &= kernel_call(group_sums, frequency, codes, count)[codes] > 1
    return keep


def _prune(frame: ModelFrame, absorb: list[str], drop_singletons: bool) -> tuple[int, int]:
    """Drop all-zero fixed-effect levels and singletons until neither remains."""
    outcome = frame.spec.outcome
    separated = singletons = 0
    while True:
        y = frame.numeric(outcome)
        keep = torch.ones(frame.n, dtype=torch.bool)
        for name in absorb:
            codes, count = frame.codes(name)
            keep &= (kernel_call(group_sums, y, codes, count) > 0)[codes]
        dropped = frame.n - int(keep.sum())
        if dropped == frame.n:
            raise AnalysisError("constant_outcome", f"'{outcome}' is zero in every observation; "
                                "nothing can be estimated.")
        changed = dropped > 0
        if changed:
            separated += dropped
            frame.restrict(keep)
        if drop_singletons:
            keep = _singleton_keep(frame, absorb)
            dropped = frame.n - int(keep.sum())
            if dropped == frame.n:
                raise AnalysisError("empty_sample", "Every observation is a singleton in some "
                                    "absorbed dimension. Absorb fewer dimensions or set "
                                    "drop_singletons=False.")
            if dropped:
                singletons += dropped
                frame.restrict(keep)
                changed = True
        if not changed:
            break
    if separated:
        frame.warn(f"Dropped {separated} observation(s) in fixed-effect levels whose outcomes "
                   "are all zero (separated by a fixed effect: they are fitted perfectly and "
                   "carry no information).")
    if singletons:
        frame.warn(f"Dropped {singletons} singleton observation(s) alone in a level of an "
                   "absorbed dimension (drop_singletons=False keeps them).")
    return separated, singletons


def _poisson_log_likelihood(y: Tensor, mu: Tensor, w: Tensor) -> float:
    from openecon.engines.count_numeric import poisson_logmass
    return float((w * poisson_logmass(y, mu)).sum())


def fit_ppmlhdfe(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``ppmlhdfe``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    tolerance, max_iterations = frame.option("tolerance"), frame.option("max_iterations")
    if not 0 < tolerance < 1:
        raise AnalysisError("invalid_option", "tolerance must be a number in (0, 1), e.g. 1e-8.")
    absorb = frame.role("absorb")
    count_outcome(frame, frame.numeric(spec.outcome), "ppmlhdfe", note=False)
    separated, singletons = _prune(frame, absorb, frame.option("drop_singletons"))
    y = frame.numeric(spec.outcome)
    weights = likelihood_weights(frame)
    w, nobs = weights.user, weights.nobs
    offset = linear_offset(frame)
    dimensions = [frame.codes(name) for name in absorb]
    clustered = spec.covariance == "cluster"
    clusters = frame.cluster_dimensions() if clustered else []
    design = frame.design(intercept=False)
    demean_tol = min(1e-10, tolerance / 10)
    screened = kernel_call(demean, design.x, dimensions, weights.for_screen(), tol=demean_tol,
                           max_iter=_DEMEAN_SWEEPS)
    design, _ = frame.drop_absorbed(design, screened.values, weights.for_screen())
    if not design.terms:
        raise AnalysisError("empty_design", "Every regressor is collinear with the absorbed fixed "
                            "effects; nothing is left to estimate.")
    k = len(design.terms)
    # A dimension nested in either clustering dimension is already counted by
    # the cluster adjustment. Count connected components only among the rest.
    nested = nested_in_clusters(dimensions, clusters) if clustered else [False] * len(dimensions)
    partial = kernel_call(absorbed_degrees_of_freedom,
                          [dim for dim, inside in zip(dimensions, nested, strict=True) if not inside])
    remaining = iter(partial.redundant)
    dof = AbsorbedDF([count for _, count in dimensions],
                    [count if inside else next(remaining)
                     for (_, count), inside in zip(dimensions, nested, strict=True)],
                    nested, partial.total)
    constant_added = all(dof.nested)
    df_absorbed = dof.total + int(constant_added)
    k_total = k + df_absorbed
    df_resid = nobs - k_total
    if df_resid <= 0:
        raise AnalysisError("insufficient_observations", "The model has no residual degrees of "
                            "freedom: it needs more observations than slopes plus absorbed "
                            "fixed effects.")
    result = kernel_call(ppml_irls, design.x, y, w, offset, dimensions, tol=tolerance,
                         max_iter=max_iterations, demean_tol=demean_tol,
                         demean_max_iter=_DEMEAN_SWEEPS)
    if not result.converged and result.separated:
        culprit = (f" The coefficient of '{design.terms[result.diverging]}' is diverging."
                   if result.diverging >= 0 else "")
        raise AnalysisError(
            "separation_detected",
            f"ppmlhdfe: {result.separated} observation(s) with a zero outcome have fitted "
            "means that keep shrinking toward zero while the deviance no longer changes: a "
            "regressor (or a combination of regressors and fixed effects) predicts zero "
            f"outcomes perfectly, so finite estimates do not exist.{culprit} Drop that "
            "regressor or the separated observations (only separation by a fixed effect is "
            "removed automatically).")
    if not result.converged:
        raise AnalysisError(
            "nonconvergence",
            f"ppmlhdfe did not converge: {result.message} Fitted means running to zero "
            "indicate separation by a regressor; drop the regressor that predicts zero "
            "outcomes perfectly, or raise max_iterations.")
    mu = result.mu
    if spec.covariance == "nonrobust":
        covariance = result.xtx_inv
        info: dict[str, Any] = {"covariance": "nonrobust",
                                "correction": "observed information of the Poisson likelihood "
                                              "on the partialled-out regressors"}
    else:
        covariance, info = linear_covariance(
            frame, x=result.x_within, resid=y - mu, bread=result.xtx_inv, n=nobs, k=k_total,
            df_resid=df_resid, weights=weights.for_screen(), clusters=clusters or None,
            cluster_names=registry.cluster_columns(spec) if clustered else None)
    info.update({"df_inference": None, "nobs": nobs, "k_total": k_total,
                 "absorbed_degrees_of_freedom": df_absorbed,
                 "degrees_of_freedom_convention": "reghdfe: observed levels minus redundant "
                 "(connected components); cluster-nested dimensions not counted, one degree of "
                 "freedom added for the constant when every dimension is nested"})
    log_likelihood = _poisson_log_likelihood(y, mu, w)
    exposure = torch.ones_like(y) if offset is None else torch.exp(offset)
    null_mu = exposure * ((w * y).sum() / (w * exposure).sum())
    null = _poisson_log_likelihood(y, null_mu, w)
    metrics = {
        "pseudo_r_squared": 1 - log_likelihood / null if null != 0 else None,
        "log_likelihood": log_likelihood, "deviance": result.deviance, "df_resid": df_resid,
        "df_absorbed": df_absorbed, "iterations": result.iterations,
        "n_separated_dropped": separated, "n_singletons_dropped": singletons,
    }
    tests = {"model": wald_test(result.beta, covariance, list(range(k)),
                                label="Wald chi2 test of the slopes")}
    extra = {
        "absorbed": [{"column": name, "levels": levels, "redundant": redundant, "nested": nested}
                     for name, levels, redundant, nested
                     in zip(absorb, dof.levels, dof.redundant, dof.nested, strict=True)],
        "constant_degree_of_freedom_added": constant_added,
        "null_log_likelihood": null, "null_model": "constant only (with the offset)",
        "log_likelihood_note": "Poisson log pseudolikelihood",
        "separation_check": "fixed-effect levels with all-zero outcomes only",
        "demeaning_sweeps": result.demeaning_sweeps, "tolerance": tolerance,
    }
    return build_result(
        frame, terms=design.terms, params=result.beta, covariance=covariance, use_t=False,
        df_resid=df_resid, metrics=metrics, fitted=mu, nobs=nobs, inference=info, tests=tests,
        extra=extra, categories=design.categories, solver="irls_partialled_out_qr",
        solver_diagnostics={"converged": True, "iterations": result.iterations,
                            "demeaning_sweeps": result.demeaning_sweeps},
        optimizer={"method": "irls_partialled_out_qr", "iterations": result.iterations,
                   "converged": True, "tolerance": tolerance, "maxiter": max_iterations,
                   "message": result.message})


def ppmlhdfe(*, data: Any, y: str, x: Sequence[str], absorb: Sequence[str],
             covariance: str | None = None, cluster: str | Sequence[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             offset: str | None = None, exposure: str | None = None,
             drop_singletons: bool = True, tolerance: float = 1e-8, max_iterations: int = 100,
             categorical: Sequence[str] | None = None, missing: str = "raise",
             alpha: float = 0.05) -> ResultBundle:
    """Poisson pseudo-likelihood regression with absorbed fixed effects (``ppmlhdfe``).

    Model
        ``E[y | x, d] = exp(x'b + alpha_1[d1] + alpha_2[d2] + ... + offset)`` for a
        nonnegative outcome (zeros welcome): the gravity-model workhorse. Only the
        conditional mean has to be correct, so ``y`` need not be a count and the
        default covariance is robust. No ``Intercept`` is reported: the constant is
        absorbed.

    Estimator
        IRLS whose weighted least-squares steps partial the fixed effects out of the
        working response and the regressors (weights ``w mu``); the slopes equal those
        of the Poisson regression on explicit dummies. Step halving on the deviance;
        convergence when the relative deviance change is below ``tolerance``.
        Observations in fixed-effect levels with all-zero outcomes ("separated") and
        singletons are dropped iteratively and counted. Regressors without within
        variation are omitted and recorded.

    Parameters
        data, y, x: data, nonnegative outcome and the list of regressors.
        absorb: list of columns whose levels are absorbed as fixed effects.
        covariance: ``'robust'`` (default; sandwich on the partialled-out regressors
            times ``N/(N-K)``, ``K = k + df_absorbed``), ``'cluster'`` (CR1
            ``G/(G-1) (N-1)/(N-K)``; two columns use inclusion-exclusion with the
            smallest ``G``), ``'nonrobust'`` (inverse information).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'``.
        offset / exposure: as in :func:`poisson`.
        drop_singletons: drop observations alone in a level (default True).
        tolerance, max_iterations: IRLS convergence tolerance (relative deviance
            change, default 1e-8) and iteration limit.
        categorical, missing, alpha: as in :func:`glm`.

    Result
        z statistics. ``metrics``: pseudo_r_squared (``1 - ll/ll_0``, ``ll_0`` the
        constant-only Poisson model on the estimation sample), log_likelihood (log
        pseudolikelihood), deviance, df_resid, df_absorbed, iterations,
        n_separated_dropped, n_singletons_dropped. ``tests['model']``: Wald chi2 of
        the slopes. ``extra['absorbed']``: per dimension column, levels, redundant,
        nested.

    Limitations
        Only observations separated by a fixed effect are dropped automatically.
        Separation caused by regressors (ppmlhdfe drops those observations after its
        ReLU / simplex search) is detected and raises ``separation_detected``.

    Stata
        ``ppmlhdfe trade fta ldist, absorb(exp#year imp#year) cluster(pair)`` is
        ``oe.ppmlhdfe(data=df, y='trade', x=['fta', 'ldist'], absorb=['exp_year',
        'imp_year'], cluster='pair')`` (build the interaction columns first).

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"firm": np.repeat(range(100), 8), "year": np.tile(range(8), 100)})
        >>> df["x"] = rng.normal(size=800)
        >>> df["y"] = rng.poisson(np.exp(0.02 * df.firm + 0.1 * df.year + 0.5 * df.x - 1))
        >>> print(oe.ppmlhdfe(data=df, y="y", x=["x"], absorb=["firm", "year"]).summary())
    """
    spec = build_spec(
        "ppmlhdfe", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=False,
        covariance=covariance, cluster=cluster, weights=weights, weight_type=weight_type,
        missing=missing, alpha=alpha,
        columns={"absorb": column_list(absorb, "absorb"), "offset": offset,
                 "exposure": exposure},
        options={"drop_singletons": drop_singletons, "tolerance": tolerance,
                 "max_iterations": max_iterations})
    return fit(spec, data=data)
