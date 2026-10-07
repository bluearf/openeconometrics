"""Truncated normal regression by full maximum likelihood: Stata's ``truncreg``.

Model
-----
``y = x'b + e`` with ``e ~ N(0, sigma^2)``, but the sample contains only
observations with ``ll < y < ul``: units outside the limits are not sampled at
all (in contrast to censoring, where they are sampled and their outcome is
recorded at the limit). The density of an observed outcome is the normal
density divided by the probability of being inside the limits,

    ln L = sum_i w_i { -ln sigma + ln phi((y_i - m_i) / sigma)
                       - ln[Phi((ul - m_i) / sigma) - Phi((ll - m_i) / sigma)] },

``m_i = x_i'b + offset_i``, with ``Phi((ul - m)/sigma) = 1`` without an upper
limit and ``Phi((ll - m)/sigma) = 0`` without a lower limit. The coefficients
describe the untruncated population.

Estimation
----------
Observations with ``y <= ll`` or ``y >= ul`` are excluded and counted
(Stata's "obs. truncated"). Newton-Raphson on ``(b, ln sigma)`` with the
analytic score and Hessian of ``kernels.TruncatedObjective``, started at the
least-squares fit of the retained sample. The likelihood is not globally
concave; non-concave iterations use the optimizer's Marquardt steps. With a
constant the regressors are centred at their means during the iteration and
the constant is mapped back exactly (``common`` module notes).

Reported (Stata's conventions)
------------------------------
z tests, the ancillary ``/sigma`` with its delta-method standard error, the
Wald chi2 test of the slopes under every covariance, and no pseudo R-squared.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, wald_test,
)
from openecon.econometrics.limited.common import (
    build_spec, centre, check_pweights, constant_shift, count, likelihood_covariance,
    likelihood_weights, limit_options, offset_column, ols_start, require_observations,
    resolve_covariance, resolve_limits, run_newton, scale_to_sigma, slope_indices,
    solver_fields, uncentre, weight_scale,
)
from openecon.econometrics.limited.kernels import TruncatedObjective
from openecon.models import ModelSpec, ResultBundle

_ADVICE = ("The truncated-normal likelihood can be flat or unbounded when the retained "
           "outcomes do not look like a truncated normal sample (for example when they pile up "
           "at a limit); check the limits ll / ul and the scale of the regressors.")


def fit_truncreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``truncreg``: ``(spec, data) -> ResultBundle``."""
    command = "truncreg"
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    y = frame.numeric(spec.outcome)
    ll, ul = resolve_limits(frame, y, command)
    outside = torch.zeros(len(y), dtype=torch.bool)
    if ll is not None:
        outside |= y <= ll
    if ul is not None:
        outside |= y >= ul
    n_truncated = count(outside, likelihood_weights(frame))
    if bool(outside.all()):
        raise AnalysisError("empty_sample", f"{command}: no outcome lies strictly between the "
                            "limits; check ll / ul against the data.")
    if n_truncated:
        frame.restrict(~outside, f"Excluded {int(outside.sum())} observation(s) outside the "
                                 "truncation limits (truncated).")
        y = frame.numeric(spec.outcome)
    weights = likelihood_weights(frame)
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    offset = offset_column(frame)
    k = len(design.terms)
    if k == 0:
        raise AnalysisError("no_parameters", f"{command}: no regressor is left; include a "
                            "constant or a regressor that varies.")
    require_observations(weights.nobs, k + 1)
    # The iteration runs on centred regressors; the constant is mapped back exactly.
    means = centre(design, weights.for_screen())
    objective = kernel_call(TruncatedObjective, design.x, y, weights.user, ll, ul, offset)
    beta, tau = ols_start(design.x, y, weights.user, offset, command=command)
    result = run_newton(objective, torch.cat([beta, torch.tensor([tau], dtype=torch.float64)]),
                        command=command, advice=_ADVICE, scale=weight_scale(weights))
    centred, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centred), weights)
    theta, covariance = uncentre(centred, covariance, constant_shift([means], extra=1))
    params, reported = scale_to_sigma(theta, covariance, k)
    info["correction"] = f"{info['correction']}; /sigma by the delta method"
    slopes = slope_indices(design)
    tests: dict[str, Any] = {}
    if slopes:
        tests["model"] = wald_test(params, reported, slopes,
                                   label="Wald chi2 test of the slopes")
    criteria = information_criteria(log_likelihood, k + 1, weights.nobs)
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "sigma": float(params[k]), "n_truncated": n_truncated, "df_model": len(slopes)}
    extra = {
        "limits": {"lower": ll, "upper": ul},
        "lnsigma": {"estimate": float(theta[k]),
                    "std_error": float(covariance[k, k].clamp_min(0).sqrt())},
        "parameterization": "estimated in (b, ln sigma); /sigma by the delta method",
        "starting_values": "least squares on the retained sample and its ML residual scale",
    }
    return build_result(
        frame, terms=[*design.terms, "/sigma"], params=params, covariance=reported,
        equations=[spec.outcome] * k + [None], use_t=False, metrics=metrics,
        fitted=objective.index(centred), nobs=weights.nobs, inference=info, tests=tests,
        extra=extra, categories=design.categories,
        provenance={"prediction_definition": "linear prediction x'b of the untruncated "
                                             "outcome"},
        **solver_fields(result))


def truncreg(*, data: Any, y: str, x: Sequence[str], ll: float | None = None,
             ul: float | None = None, covariance: str | None = None,
             cluster: str | Sequence[str] | None = None, weights: str | None = None,
             weight_type: str | None = None, offset: str | None = None,
             categorical: Sequence[str] | None = None, intercept: bool = True,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Truncated normal regression, Stata's ``truncreg``.

    Model
        ``y = x'b + e``, ``e ~ N(0, sigma^2)``, sampled only when ``ll < y < ul``. Least
        squares on such a sample is biased towards zero; the truncated likelihood
        recovers the coefficients of the untruncated population.

    Estimator
        Full maximum likelihood,

            ``ln L = sum w {-ln sigma + ln phi((y - x'b)/sigma)
                     - ln[Phi((ul - x'b)/sigma) - Phi((ll - x'b)/sigma)]}``,

        by Newton-Raphson on ``(b, ln sigma)`` with the analytic score and Hessian,
        started at least squares on the retained sample. ``/sigma`` and its standard
        error follow from the delta method. With a constant the regressors are centred
        during the iteration and the constant is mapped back exactly. An outcome that
        the regressors fit exactly raises ``perfect_fit``; a likelihood without an
        interior maximum raises ``nonconvergence``.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        y: outcome. Rows with ``y <= ll`` or ``y >= ul`` are excluded and counted in
            ``metrics['n_truncated']`` (with a warning), as Stata does.
        x: regressors; collinear columns are omitted with a warning.
        ll, ul: truncation limits (numbers) or None for no truncation on that side. With
            neither limit the fit is the normal regression by ML.
        covariance: ``'nonrobust'`` (default; inverse observed information), ``'opg'``,
            ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'`` (``G/(G-1)``; implied by
            ``cluster``; two columns give two-way clustering).
        cluster: cluster column, or a list of two columns.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        offset: column added to ``x'b`` with coefficient one.
        categorical: columns of ``x`` to expand into treatment-coded indicators.
        intercept: include a constant (default True).
        missing: ``'raise'`` (default) or ``'drop'`` incomplete rows.
        alpha: significance level of the confidence intervals.

    Result
        z statistics. Terms: the regressors (equation = the outcome) and the ancillary
        ``/sigma``. ``metrics``: log_likelihood, aic, bic (``K + 1`` parameters), sigma,
        n_truncated, df_model. ``tests['model']``: Wald chi2 of the slopes (what Stata
        prints for truncreg under every covariance). ``extra``: limits, lnsigma. The chart
        sample holds the linear prediction ``x'b``. ``result.nobs`` counts the retained
        observations.

    Stata / EViews
        ``truncreg whrs kl6 k618 wa we, ll(0)`` is
        ``oe.truncreg(data=df, y='whrs', x=['kl6', 'k618', 'wa', 'we'], ll=0)``. EViews:
        censored regression with the "truncated sample" option (``censored(d=n, t)``).

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=4000)})
        >>> df["y"] = 0.5 + df.x + rng.normal(size=4000)
        >>> sample = df[df.y > 0]                  # only positive outcomes are sampled
        >>> print(oe.truncreg(data=sample, y="y", x=["x"], ll=0).summary())
    """
    options = limit_options(ll, ul, "truncreg")
    if "ll_at_min" in options or "ul_at_max" in options:
        raise AnalysisError("invalid_spec", "truncreg: ll and ul must be numbers; a limit at "
                            "the smallest or largest outcome would exclude that observation.")
    spec = build_spec(
        "truncreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset}, options=options)
    return fit(spec, data=data)
