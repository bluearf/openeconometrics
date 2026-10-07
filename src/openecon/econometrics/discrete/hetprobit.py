"""Heteroskedastic probit by full maximum likelihood: Stata's ``hetprobit``.

Model
-----
A latent ``y* = x'b + e`` whose error standard deviation depends on covariates
through a multiplicative (Harvey) variance function,

    e ~ N(0, s^2),   s = exp(z'g),      Pr(y = 1 | x, z) = Phi(x'b / exp(z'g)).

The variance equation ``z`` has no constant: a constant would only rescale
``b``. With ``g = 0`` the model is the ordinary probit. The coefficients of
``z`` are reported in the equation ``lnsigma`` (terms ``lnsigma:<column>``);
Stata labels the same parameters ``lnsigma2`` in older releases and
``lnsigma`` since release 16.

Estimation
----------
Newton-Raphson on ``(b, g)`` with the analytic score and Hessian of
``kernels.HetprobitObjective``, started at the probit estimates of ``b`` and
``g = 0``. The likelihood is not globally concave; non-concave iterations use
the optimizer's Marquardt steps.

The iteration runs on mean-centred variance regressors. Because the variance
equation has no constant, ``z'g = (z - m)'g + m'g`` only rescales the mean
equation: ``Phi(x'b / exp(z'g)) = Phi(x'c / exp((z - m)'g))`` with
``b = c exp(m'g)``. The centred problem ``(c, g)`` is well conditioned even
when ``z`` has a large mean (age, a calendar year); estimates and covariance
are mapped back exactly, ``V(b, g) = J V(c, g) J'`` with
``J = [[exp(m'g) I, b m'], [0, I]]``, so the reported model is the one
specified (Stata's parameterization). With a constant in the mean equation its
regressors are centred as well (``common.centre_regressors``) and the constant
is mapped back in the same way.

Reported
--------
``tests['model']``: Wald chi2 of the slopes of the mean equation (what Stata's
header prints for hetprobit, under every covariance). ``tests['lnsigma']``:
test of homoskedasticity ``g = 0``: the likelihood-ratio chi2(q) against the
probit model under a likelihood-based covariance (``nonrobust``, ``opg``), the
Wald chi2(q) under ``robust`` / ``cluster``. ``metrics``: log_likelihood, aic,
bic (``k + q`` parameters).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test,
    wald_test,
)
from openecon.econometrics.discrete.common import (
    EXTREME, EXTREME_EARLY, Separated, binary_outcome, build_spec, centre_regressors,
    check_pweights, constant_shift, implicit_constant_design, likelihood_covariance,
    likelihood_weights, maximize, optimizer_record, perfectly_predicted, probit_fit,
    raise_not_converged, reparameterize, require_observations, resolve_covariance,
    separation_watch,
)
from openecon.econometrics.discrete.kernels import HetprobitObjective
from openecon.models import ModelSpec, ResultBundle


# exp(2 * 300) is the largest factor a variance can absorb without leaving float64.
_MAX_LOG_SCALE = 300.0


def _uncentre(centred: Tensor, covariance: Tensor, centre: Tensor, k: int,
              command: str) -> tuple[Tensor, Tensor]:
    """Map ``(c, g)`` estimated with centred variance regressors to ``(b, g)``.

    ``b = c exp(m'g)`` and ``V(b, g) = J V(c, g) J'`` with
    ``J = [[exp(m'g) I, b m'], [0, I]]`` (exact for every covariance estimator,
    since scores and Hessian transform linearly at the maximum).
    """
    shift = float(centre @ centred[k:])
    if abs(shift) > _MAX_LOG_SCALE:
        raise AnalysisError(
            "variance_scale_overflow",
            f"{command}: the variance equation scales the mean equation by exp({shift:.4g}); "
            "coefficients and variances on that scale are outside the float64 range, because "
            "the het columns have large means and the variance equation has no constant. "
            "Centre the het columns (subtract their means).")
    factor = math.exp(shift)
    theta = centred.clone()
    theta[:k] = centred[:k] * factor
    jacobian = torch.eye(len(centred), dtype=torch.float64)
    jacobian[:k, :k] *= factor
    jacobian[:k, k:] = torch.outer(theta[:k], centre)
    return theta, jacobian @ covariance @ jacobian.T


def fit_hetprobit(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``hetprobit``: ``(spec, data) -> ResultBundle``."""
    command = "hetprobit"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    het = frame.role("het")
    if spec.outcome in het:
        raise AnalysisError("invalid_spec", f"{command}: the outcome cannot be a variance "
                            "regressor.")
    y = binary_outcome(frame, spec.outcome, command)
    weights = likelihood_weights(frame)
    w = weights.user
    positives = float((w * y).sum())
    total = float(w.sum())
    if positives == 0 or positives == total:
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{spec.outcome}' does "
                            "not vary in the estimation sample.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    variance = implicit_constant_design(frame, het, weights.for_screen(), prefix="lnsigma:")
    mean_centre = centre_regressors(design, weights)
    k, q = len(design.terms), len(variance.terms)
    if k == 0:
        raise AnalysisError("no_parameters", f"{command}: the mean equation has no regressor "
                            "left; include a constant or a regressor that varies.")
    if q == 0:
        raise AnalysisError(
            "no_variance_regressors",
            f"{command}: no variance regressor varies (a constant only rescales the "
            "coefficients), so the model is the ordinary probit. Give het columns that vary.")
    require_observations(weights.nobs, k + q)
    _, probit = probit_fit(design.x, y, w, command=command, what="probit starting values",
                           scale=weights.scale)
    centre = (variance.x * w[:, None]).sum(dim=0) / total
    objective = kernel_call(HetprobitObjective, design.x, variance.x - centre, y, w)

    def separated(theta: Tensor, threshold: float) -> int:
        return perfectly_predicted(objective.outcome_probabilities(theta), threshold)

    start = torch.cat([probit.theta, torch.zeros(q, dtype=torch.float64)])
    advice = ("The heteroskedastic probit likelihood can be flat or unbounded when the variance "
              "regressors nearly separate the outcome; simplify the variance equation or "
              "rescale its regressors.")
    try:
        result = maximize(objective, start, what=command, callback=separation_watch(separated),
                          scale=weights.scale)
    except Separated as stop:
        raise_not_converged(command, separated(stop.theta, EXTREME_EARLY),
                            "the likelihood is monotone.")
    if not result.converged:
        raise_not_converged(command, separated(result.theta, EXTREME),
                            result.diagnostics.get("message"), advice=advice)
    centred, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    # Undo the centring of the mean regressors (constant a = a_c - m'b), then that of z.
    jacobian = torch.eye(k + q, dtype=torch.float64)
    jacobian[:k, :k] = constant_shift(mean_centre)
    shifted, covariance = reparameterize(centred, covariance, jacobian)
    theta, covariance = _uncentre(shifted, covariance, centre, k, command)
    if abs(float(centre @ theta[k:])) > 20:
        frame.warn("The variance regressors have large means: the coefficients of the mean "
                   "equation are scaled by exp(mean(z)'g) = "
                   f"{math.exp(float(centre @ theta[k:])):.3g}. Centre the het columns to "
                   "obtain coefficients on the usual probit scale.")
    slopes = [i for i, term in enumerate(design.terms) if term != "Intercept"]
    tests: dict[str, Any] = {
        "model": wald_test(theta, covariance, slopes,
                           label="Wald chi2 test of the slopes of the mean equation")}
    if spec.covariance in {"nonrobust", "opg"}:
        tests["lnsigma"] = lr_test(log_likelihood, probit.value, q,
                                   label="LR test of lnsigma = 0 (homoskedastic probit)")
    else:
        tests["lnsigma"] = wald_test(theta, covariance, range(k, k + q),
                                     label="Wald test of lnsigma = 0 (homoskedastic probit)")
    criteria = information_criteria(log_likelihood, k + q, weights.nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"]}
    extra = {
        "zero_outcomes": total - positives, "nonzero_outcomes": positives,
        "probit_log_likelihood": probit.value,
        "variance_function": "sigma = exp(z'g), no constant",
        "variance_terms": variance.terms,
        "starting_values": "probit estimates and g = 0",
        "variance_regressor_means": centre.tolist(),
    }
    return build_result(
        frame, terms=[*design.terms, *variance.terms], params=theta, covariance=covariance,
        equations=[spec.outcome] * k + ["lnsigma"] * q, use_t=False, metrics=metrics,
        fitted=objective.probabilities(centred), nobs=weights.nobs, inference=info, tests=tests,
        extra=extra, categories={**design.categories, **variance.categories},
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def hetprobit(*, data: Any, y: str, x: Sequence[str], het: Sequence[str],
              covariance: str | None = None, cluster: str | Sequence[str] | None = None,
              weights: str | None = None, weight_type: str | None = None,
              categorical: Sequence[str] | None = None, intercept: bool = True,
              missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Heteroskedastic probit regression, Stata's ``hetprobit``.

    Model
        ``Pr(y = 1 | x, z) = Phi(x'b / exp(z'g))``: a probit whose latent error has
        standard deviation ``sigma = exp(z'g)``. The variance equation has no constant
        (it would only rescale ``b``); ``g = 0`` is the ordinary probit. Because ``z``
        changes the scale, the sign of the effect of a variable that appears in both
        ``x`` and ``z`` is not given by its coefficient alone.

    Estimator
        Full maximum likelihood over ``(b, g)`` by Newton-Raphson with the analytic score
        ``(m x / s, -m t z)`` and Hessian, where ``t = x'b / s`` and ``m`` is the inverse
        Mills ratio of the observed outcome, computed stably in both tails. Starting
        values: the probit estimates and ``g = 0``. The iteration uses mean-centred
        variance regressors and maps the estimates back exactly, so a ``het`` column
        with a large mean does not hurt convergence.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome coded 0/1.
        x: regressors of the mean equation; collinear columns are omitted with a warning.
        het: regressors of the variance equation ``ln sigma = z'g`` (at least one; may
            overlap ``x``). A constant or collinear column is omitted with a warning.
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        categorical: columns of ``x`` or ``het`` to expand into treatment-coded indicators.
        intercept: include a constant in the mean equation (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms: the mean equation (equation = the outcome), then
        ``lnsigma:<column>`` (equation ``lnsigma``). ``metrics``: log_likelihood, aic, bic.
        ``tests['model']``: Wald chi2 of the mean-equation slopes. ``tests['lnsigma']``:
        LR chi2(q) of ``g = 0`` against the probit model (``nonrobust``, ``opg``) or the
        Wald chi2(q) (``robust``, ``cluster``). ``extra``: zero_outcomes,
        nonzero_outcomes, probit_log_likelihood, variance_terms,
        variance_regressor_means. The chart sample holds the fitted ``Pr(y = 1)``.
        ``variance_scale_overflow`` is raised when ``|mean(z)'g| > 300`` (the rescaled
        coefficients and variances would leave the float64 range).

    Stata
        ``hetprobit y x1 x2, het(z) vce(robust)`` is
        ``oe.hetprobit(data=df, y='y', x=['x1', 'x2'], het=['z'], covariance='robust')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=2000), "z": rng.normal(size=2000)})
        >>> latent = 0.3 + 1.0 * df.x + np.exp(0.6 * df.z) * rng.normal(size=2000)
        >>> df["y"] = (latent > 0) * 1.0
        >>> print(oe.hetprobit(data=df, y="y", x=["x"], het=["z"]).summary())
    """
    spec = build_spec(
        "hetprobit", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"het": column_list(het, "het")})
    return fit(spec, data=data)
