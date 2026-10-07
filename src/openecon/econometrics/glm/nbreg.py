"""Negative binomial regression by full maximum likelihood: Stata's ``nbreg``.

Model
-----
``E[y|x] = mu = exp(x'b + offset)`` with overdispersion relative to Poisson:

* ``dispersion='mean'`` (NB2, Stata's default): ``Var = mu (1 + alpha mu)``;
  ``y`` is Poisson with a gamma-distributed multiplicative effect of mean 1
  and variance ``alpha``.
* ``dispersion='constant'`` (NB1): ``Var = mu (1 + delta)``.

The parameters ``(b, ln alpha)`` (resp. ``(b, ln delta)``) are estimated
jointly by Newton-Raphson with the analytic score and Hessian of
``kernels.NegativeBinomialObjective``, starting from the Poisson estimates of
``b`` and a method-of-moments dispersion: NB2
``alpha_0 = sum w [(y - mu)^2 - mu] / sum w mu^2``, NB1
``delta_0 = sum w (y - mu)^2 / sum w mu - 1`` (kept inside [0.05, 50]).

Reported (as Stata)
-------------------
The ancillary parameter is the term ``/lnalpha`` (``/lndelta``); ``alpha``
itself is in ``metrics`` and ``extra['alpha']`` with its delta-method standard
error ``alpha se(ln alpha)`` and the confidence interval obtained by
exponentiating the interval of ``ln alpha``. ``tests['alpha']`` is the
likelihood-ratio test of ``alpha = 0`` against Poisson; the null is on the
boundary of the parameter space, so the reference distribution is
``chibar2(01)``: half the ``chi2(1)`` tail (reported only under a
likelihood-based covariance). ``tests['model']`` is the LR chi2 against the
constant-only negative binomial model (``nonrobust``) or the Wald chi2 of the
slopes.

Boundary
--------
For data that are not overdispersed the likelihood increases monotonically as
``alpha -> 0`` (the Poisson limit) and has no interior maximum; the fit is
rejected with ``boundary_solution`` and the advice to use ``oe.poisson``.
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
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call, lr_test,
    wald_test,
)
from openecon.econometrics.glm.common import (
    Weights, build_spec, center_design, check_pweights, count_outcome, likelihood_covariance,
    likelihood_weights, linear_offset, maximize, optimizer_record, require_observations,
    require_terms, resolve_covariance, safe_exp, slope_indices, to_centered, uncenter,
)
from openecon.econometrics.glm.families import Log, Poisson
from openecon.econometrics.glm.glm import estimate
from openecon.econometrics.glm.kernels import NegativeBinomialObjective
from openecon.engines import optimize
from openecon.engines.count_numeric import validate_nb_precision
from openecon.engines.distributions import chi2_sf
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle

# A small dispersion alone does not certify the Poisson boundary.
_BOUNDARY = 1e-4
_START_RANGE = (0.05, 50.0)


def _moment_start(y: Tensor, mu: Tensor, w: Tensor, form: str) -> float:
    """Method-of-moments dispersion at the Poisson fit, kept inside a safe range."""
    squares = (y - mu).square()
    if form == "mean":
        value = float((w * (squares - mu)).sum() / (w * mu.square()).sum())
    else:
        value = float((w * squares).sum() / (w * mu).sum()) - 1
    if not math.isfinite(value):
        value = 1.0
    return min(max(value, _START_RANGE[0]), _START_RANGE[1])


def _boundary_score(y: Tensor, mu: Tensor, weights: Tensor, form: str) -> bool:
    """Independent one-sided dispersion score at the fitted Poisson limit.

    NB2 has score ``[(y-mu)^2-y]/2`` in alpha at zero; NB1 divides it by mu.
    This certificate never evaluates a large-shape NB gamma difference.
    """
    residual = (y - mu).square() - y
    scale = (y - mu).square() + y
    if form == "constant":
        residual, scale = residual / mu, scale / mu
    score = float((weights * residual).sum())
    allowance = 16 * torch.finfo(torch.float64).eps * max(
        1.0, float((weights * scale).sum()))
    return math.isfinite(score) and score <= allowance


def _at_boundary(result: optimize.OptimResult, k: int, ll_poisson: float,
                 poisson_boundary_score: bool) -> bool:
    magnitude = max(1.0, abs(ll_poisson))
    return (poisson_boundary_score
            and float(result.theta[k]) < math.log(_BOUNDARY)
            and abs(float(result.hessian[k, k])) <= _BOUNDARY * magnitude
            and abs(result.value - ll_poisson) <= 1e-6 * magnitude)


def _precision_failure(result: optimize.OptimResult) -> None:
    if result.diagnostics.get("precision_rejected"):
        raise AnalysisError(
            "precision_unsupported",
            "The negative binomial iteration reached count/shape values outside the "
            "validated float64 likelihood domain; no unverified maximum is reported.")


def _null(frame: ModelFrame, y: Tensor, weights: Weights, offset: Tensor | None, form: str,
          log_dispersion: float, intercept: bool) -> tuple[float | None, str | None]:
    """Log likelihood of the constant-only negative binomial model (free dispersion)."""
    w = weights.user
    columns = 1 if intercept else 0
    design = Design(torch.ones((frame.n, columns), dtype=torch.float64),
                    ["Intercept"][:columns], {}, intercept)
    poisson = estimate(frame, design, y, Poisson(), Log(), weights=weights, offset=offset,
                       command="nbreg (constant-only model)")
    objective = kernel_call(NegativeBinomialObjective, design.x, y, w, offset, form)
    start = torch.cat([poisson.beta, torch.tensor([log_dispersion], dtype=torch.float64)])
    result = maximize(objective, start, what="constant-only negative binomial")
    if _at_boundary(result, columns, poisson.log_likelihood(),
                    _boundary_score(y, poisson.state.mu, w, form)):
        frame.warn("The constant-only negative binomial model is at the Poisson boundary "
                   "(alpha = 0); its Poisson log likelihood is used as the null model.")
        return poisson.log_likelihood(), "constant-only Poisson model (alpha at the boundary)"
    if result.converged:
        return result.value, ("constant-only negative binomial model" if intercept else
                              "negative binomial model with every coefficient zero")
    _precision_failure(result)
    raise AnalysisError("nonconvergence", "The constant-only negative binomial model did not "
                        f"converge: {result.diagnostics.get('message')}")


def fit_nbreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``nbreg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    form = frame.option("dispersion")
    name = "alpha" if form == "mean" else "delta"
    weights = likelihood_weights(frame)
    offset = linear_offset(frame)
    y = frame.numeric(spec.outcome)
    count_outcome(frame, y, "nbreg")
    kernel_call(validate_nb_precision, y)
    if not bool((y > 0).any()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' is zero in every observation; "
                            "the model is not identified.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    k, nobs = len(design.terms), weights.nobs
    require_observations(nobs, k + 1)
    poisson = estimate(frame, design, y, Poisson(), Log(), weights=weights, offset=offset,
                       command="nbreg (Poisson starting values)")
    ll_poisson = poisson.log_likelihood()
    start_dispersion = _moment_start(y, poisson.state.mu, weights.user, form)
    # The likelihood is maximized on the centered design (see common.center_design).
    x, means = center_design(design, weights.user)
    objective = kernel_call(NegativeBinomialObjective, x, y, weights.user, offset, form)
    start = torch.cat([to_centered(poisson.beta, means),
                       torch.tensor([math.log(start_dispersion)], dtype=torch.float64)])
    result = maximize(objective, start, what="negative binomial")
    if _at_boundary(result, k, ll_poisson,
                    _boundary_score(y, poisson.state.mu, weights.user, form)):
        raise AnalysisError(
            "boundary_solution",
            f"The maximum likelihood estimate of {name} lies at zero: the data are not "
            "overdispersed relative to Poisson, so the negative binomial likelihood has no "
            "interior maximum (it increases toward the Poisson limit). Use oe.poisson, "
            "with covariance='robust' if the variance may differ from the mean.")
    if not result.converged:
        _precision_failure(result)
        raise AnalysisError("nonconvergence", "The negative binomial likelihood did not "
                            f"converge: {result.diagnostics.get('message')}")
    centered, log_likelihood = result.theta, result.value
    covariance, info = likelihood_covariance(frame, result.hessian,
                                             lambda: objective.score_rows(centered), weights)
    theta, covariance = uncenter(centered, covariance, [(0, means)])
    info["df_inference"] = None
    log_dispersion = float(theta[k])
    value = math.exp(log_dispersion)
    se_log = math.sqrt(float(covariance[k, k]))
    critical = critical_value(spec.alpha)
    try:
        null, null_model = _null(frame, y, weights, offset, form, log_dispersion, spec.intercept)
    except AnalysisError as exc:
        if exc.code != "precision_unsupported":
            raise
        null, null_model = None, None
        frame.warn("The constant-only negative binomial comparison exceeds the validated "
                   "count/shape precision domain. The fitted model is retained; the null "
                   "likelihood and pseudo R-squared are unavailable and a Wald slope test "
                   "replaces the model LR test.")
    slopes = slope_indices(design.terms)
    likelihood_based = spec.covariance in {"nonrobust", "opg"}
    if spec.covariance == "nonrobust" and null is not None:
        against = "the constant-only model" if spec.intercept \
            else "the model with every coefficient zero"
        model = lr_test(log_likelihood, null, len(slopes), label=f"LR chi2 test against {against}")
    else:
        model = wald_test(theta, covariance, slopes, label="Wald chi2 test of the slopes")
    tests: dict[str, Any] = {"model": model}
    if likelihood_based:
        statistic = max(0.0, 2 * (log_likelihood - ll_poisson))
        tests["alpha"] = {
            "statistic": statistic, "df": 1,
            "p_value": 0.5 * kernel_call(chi2_sf, statistic, 1) if statistic > 0 else 1.0,
            "distribution": "chibar2",
            "label": f"LR test of {name} = 0 against Poisson (chibar2(01): chi2(1) tail halved)"}
    criteria = information_criteria(log_likelihood, k + 1, nobs)
    metrics = {"log_likelihood": log_likelihood,
               "pseudo_r_squared": 1 - log_likelihood / null if null is not None and null != 0 else None,
               "aic": criteria["aic"], "bic": criteria["bic"], name: value,
               "df_resid": nobs - k - 1}
    extra = {
        "dispersion": form,
        "variance_function": "mu (1 + alpha mu)" if form == "mean" else "mu (1 + delta)",
        "alpha": {"parameter": name, "estimate": value, "std_error": value * se_log,
                  "ci_low": safe_exp(log_dispersion - critical * se_log),
                  "ci_high": safe_exp(log_dispersion + critical * se_log),
                  "method": "delta method; interval exponentiated from ln " + name},
        "poisson_log_likelihood": ll_poisson, "null_log_likelihood": null,
        "null_model": null_model,
        "starting_values": {name: start_dispersion, "method": "poisson_and_moments"},
    }
    if not likelihood_based:
        extra["alpha_test_note"] = ("The LR test of the dispersion is not reported with a robust "
                                    "or cluster covariance (it is not a likelihood-based test).")
    return build_result(
        frame, terms=[*design.terms, f"/ln{name}"], params=theta, covariance=covariance,
        equations=[spec.outcome] * k + [None], use_t=False, df_resid=nobs - k - 1,
        metrics=metrics, fitted=objective.mean(centered), nobs=nobs, inference=info, tests=tests,
        extra=extra, categories=design.categories, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def nbreg(*, data: Any, y: str, x: Sequence[str], dispersion: str = "mean",
          covariance: str | None = None, cluster: str | Sequence[str] | None = None,
          weights: str | None = None, weight_type: str | None = None,
          offset: str | None = None, exposure: str | None = None,
          categorical: Sequence[str] | None = None, intercept: bool = True,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Negative binomial regression, Stata's ``nbreg``.

    Model
        ``E[y|x] = mu = exp(x'b + offset)`` with

        * ``dispersion='mean'`` (NB2, default): ``Var(y|x) = mu (1 + alpha mu)``;
        * ``dispersion='constant'`` (NB1): ``Var(y|x) = mu (1 + delta)``.

        NB2 log likelihood with ``m = 1/alpha``:
        ``lnG(y+m) - lnG(m) - lnG(y+1) - (y+m) ln(1 + alpha mu) + y ln(alpha mu)``;
        NB1 with ``m = mu/delta``:
        ``lnG(y+m) - lnG(m) - lnG(y+1) - m ln(1+delta) + y ln(delta/(1+delta))``.
        ``(b, ln alpha)`` are estimated jointly by Newton-Raphson with analytic
        derivatives, starting from Poisson and a moment estimate of the dispersion.

    Parameters
        data, y, x: data, nonnegative count outcome and the list of regressors (may be
            empty: the constant-only model).
        dispersion: ``'mean'`` (NB2) or ``'constant'`` (NB1).
        covariance: ``'nonrobust'`` (default, inverse observed information of all
            parameters), ``'opg'``, ``'robust'`` (``N/(N-1)`` sandwich), ``'cluster'``
            (``G/(G-1)``; one or two columns).
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'pweight'`` (robust by
            default), ``'iweight'``.
        offset / exposure: as in :func:`poisson`.
        categorical, intercept, cluster, missing, alpha: as in :func:`glm`.

    Result
        z statistics. Terms: the regressors (equation = the outcome) and ``/lnalpha``
        (``/lndelta`` for NB1). ``metrics``: log_likelihood, pseudo_r_squared
        (``1 - ll/ll_0``, ``ll_0`` the constant-only negative binomial model), aic, bic
        (K + 1 parameters), alpha (or delta), df_resid. ``extra['alpha']``: estimate,
        delta-method std_error and the confidence interval exponentiated from
        ``ln alpha``. ``tests['alpha']``: LR test of ``alpha = 0`` against Poisson with
        the boundary-corrected ``chibar2(01)`` p-value (likelihood-based covariances
        only). ``tests['model']``: LR chi2 (``nonrobust``) or Wald chi2.

        Data without overdispersion have no interior maximum: ``boundary_solution``
        is raised with the advice to use :func:`poisson`. To fix ``alpha`` instead of
        estimating it use ``oe.glm(family='nbinomial', dispersion=alpha)``.

    Stata
        ``nbreg deaths i.cohort, exposure(exposure) dispersion(constant)`` is
        ``oe.nbreg(data=df, y='deaths', x=['cohort'], categorical=['cohort'],
        exposure='exposure', dispersion='constant')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=1000)})
        >>> mu = np.exp(0.5 + 0.4 * df.x)
        >>> df["y"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))    # alpha = 0.5
        >>> print(oe.nbreg(data=df, y="y", x=["x"]).summary())
    """
    spec = build_spec(
        "nbreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"offset": offset, "exposure": exposure}, options={"dispersion": dispersion})
    return fit(spec, data=data)
