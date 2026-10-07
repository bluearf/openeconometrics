"""Left-truncated Poisson and negative binomial regression: Stata's ``tpoisson``, ``tnbreg``.

Model
-----
The sample contains only outcomes above a truncation point, ``y > ll`` (by
default ``ll = 0``: zero-truncated counts such as the length of a hospital
stay or the number of items bought by a customer). The likelihood is the count
density conditional on being observed,

    ln Pr(y | y > ll) = ln f(y) - ln{1 - F(ll)},     F(ll) = sum_{j <= ll} f(j),

with ``f`` Poisson (``tpoisson``) or negative binomial (``tnbreg``: NB2,
``Var = mu (1 + alpha mu)``, or NB1 with ``dispersion='constant'``,
``Var = mu (1 + delta)``) and ``mu = exp(x'b + offset)`` the mean of the
*untruncated* distribution. ``ll`` is one integer or a column of
observation-specific truncation points.

Estimation
----------
Newton-Raphson with the analytic score and Hessian of
``kernels.TruncatedPieces``. The Poisson tail is the regularized incomplete
gamma function ``1 - F(ll) = P(ll + 1, mu)`` (``-expm1(-mu)`` for ``ll = 0``),
accurate where ``mu`` is far below the truncation point; the negative binomial
tail is the complement of the lower sum, accumulated with its derivatives.
The truncated Poisson likelihood is globally concave in ``b`` (an exponential
family); ``tnbreg`` starts from the ``tpoisson`` estimates.

Reported (as Stata)
-------------------
z statistics; ``tests['model']``: LR chi2 against the constant-only model
(``nonrobust``) or the Wald chi2 of the slopes; McFadden's pseudo R-squared;
``tnbreg``: ``/lnalpha`` (``/lndelta``), ``alpha`` in ``metrics`` with a
log-transformed interval in ``extra['alpha']`` and ``tests['alpha']``, the LR
test of ``alpha = 0`` against the truncated Poisson model (``chibar2(01)``).
The chart sample compares the outcome with the conditional mean
``E[y | y > ll] = (mu - sum_{j<=ll} j f(j)) / {1 - F(ll)}``.

Outcomes at or below the truncation point contradict the model and are an
error (``outcome_not_truncated``); they are never dropped silently.

Likelihoods without a maximum
-----------------------------
``Pr(y = ll + 1 | y > ll)`` tends to one as the mean goes to zero. A regressor
that singles out observations whose outcomes all equal ``ll + 1`` therefore
has a coefficient that diverges to minus infinity (the truncated-count analogue
of separation). Gradient and curvature vanish together along that direction,
so the iteration would stop at an arbitrary large negative coefficient with an
enormous standard error; ``_collapsed`` certifies the situation (a fitted mean
below 1e-8 at an outcome of ``ll + 1`` *and* less than 0.01 observations' worth
of information along some direction, ``common.flattest_information``) and the
fit is refused with ``separation_detected``.

The truncated negative binomial has two more limits outside its parameter
space: the Poisson model (dispersion to zero) and the logarithmic-series
distribution (means to zero; in the NB2 form with alpha to infinity). Counts
that favour a limit leave a likelihood that increases towards it ever more
slowly; ``_flat`` applies the same certificate to converged and to stalled
runs and the fit is refused with ``boundary_solution``. A constant-only
comparison model in that state is not replaced by anything: the Wald test is
reported instead of the LR test. A negative binomial fit whose lower sum would
need more than 3e8 term evaluations per likelihood evaluation is refused with
``truncation_too_large``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call,
)
from openecon.econometrics.count import common
from openecon.econometrics.count.common import Block, Boundary, Model
from openecon.econometrics.count.kernels import NegBinDensity, PoissonDensity, TruncatedPieces
from openecon.econometrics.glm.common import (
    Weights, build_spec, optimizer_record, require_observations, require_terms,
    resolve_covariance, safe_exp, slope_indices,
)
from openecon.econometrics.glm.families import Log, Poisson
from openecon.econometrics.glm.glm import estimate
from openecon.engines import optimize
from openecon.models import ModelSpec, ResultBundle

# A fitted mean below this at an outcome of ll + 1 is a candidate for a monotone likelihood.
_TINY_MEAN = 1e-8
_WATCH_FROM, _WATCH_EVERY = 20, 5
# Integers are exact in float64 up to 2^53; a truncation point beyond it has no meaning.
_LARGEST_LIMIT = 2 ** 53
# Largest number of lower-tail terms (observations times ll + 1) per negative binomial
# likelihood evaluation; each term costs a few special functions per observation.
_MAX_TAIL_TERMS = 3e8


@dataclass
class TruncatedSample:
    """A truncated count sample with its design and Poisson starting values.

    ``y``, ``weights``, ``offset`` and the design rows may be a subsample of
    the frame (the positive counts of a hurdle model).
    """

    frame: ModelFrame
    command: str
    y: Tensor
    weights: Weights
    offset: Tensor | None
    limit: Tensor | int
    design: Design
    count: Block
    start: Tensor                  # Poisson estimates (centered design)
    moment: float                  # moment estimate of the NB2 dispersion at the Poisson fit


@dataclass
class TruncatedFit:
    model: Model
    result: optimize.OptimResult
    poisson: optimize.OptimResult      # the truncated Poisson fit (start and comparison model)
    null: float | None                 # log likelihood of the constant-only model
    null_model: str | None
    dispersion: str | None             # None (Poisson), "mean" or "constant"


def _limit(frame: ModelFrame, command: str) -> tuple[Tensor | int, dict[str, Any]]:
    """The truncation point: the option ``ll`` or the ``truncation`` column."""
    columns = frame.role("truncation")
    if not columns:
        value = int(frame.option("ll"))
        if value > _LARGEST_LIMIT:
            raise AnalysisError("invalid_spec", f"{command}: ll must be a nonnegative integer "
                                "no larger than 2^53 (counts beyond that are not exact).")
        return value, {"truncation_point": value}
    if "ll" in frame.spec.options:
        raise AnalysisError("invalid_spec", f"{command}: give the truncation point either as "
                            "the number ll or as a truncation column, not both.")
    limit = frame.numeric(columns[0])
    if bool((limit < 0).any()) or bool((limit != limit.round()).any()) \
            or bool((limit > _LARGEST_LIMIT).any()):
        raise AnalysisError("invalid_truncation", f"The truncation column '{columns[0]}' must "
                            "contain nonnegative integers (no larger than 2^53).")
    return limit, {"truncation_column": columns[0], "truncation_min": int(limit.min()),
                   "truncation_max": int(limit.max())}


def truncated_sample(frame: ModelFrame, command: str, y: Tensor, weights: Weights,
                     offset: Tensor | None, limit: Tensor | int, design: Design,
                     ) -> TruncatedSample:
    """Validate a truncated sample and compute its Poisson starting values."""
    outcome = frame.spec.outcome
    if bool((y == limit + 1).all()):
        raise AnalysisError(
            "constant_outcome", f"Every truncated value of '{outcome}' equals ll + 1, the "
            "smallest outcome the truncated model admits: the likelihood increases as the mean "
            "goes to zero and has no maximum.")
    poisson = estimate(frame, design, y, Poisson(), Log(), weights=weights, offset=offset,
                       command=f"{command} (Poisson starting values)")
    w, mu = weights.user, poisson.state.mu
    moment = float((w * ((y - mu).square() - mu)).sum() / (w * mu.square()).sum())
    moment = min(max(moment if math.isfinite(moment) else 0.5, 0.05), 20.0)
    return TruncatedSample(frame, command, y, weights, offset, limit, design,
                           common.design_block(design, w, outcome, offset),
                           poisson.state.beta, moment)


def _prepare(spec: ModelSpec, data: Any, command: str) -> tuple[TruncatedSample, dict[str, Any]]:
    frame = ModelFrame(spec, data)
    y, weights, offset = common.count_sample(frame, command, integer=True)
    limit, record = _limit(frame, command)
    below = int((y <= limit).sum())
    if below:
        raise AnalysisError(
            "outcome_not_truncated",
            f"{command}: {below} observation(s) have '{spec.outcome}' at or below the "
            "truncation point. The truncated model describes only outcomes greater than ll: "
            "drop those rows if they cannot occur in the population studied, or model them "
            "with oe.hurdle or oe.zip if they are part of the data.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    require_observations(weights.nobs, len(design.terms) + int(command == "tnbreg"))
    return truncated_sample(frame, command, y, weights, offset, limit, design), record


def _collapsed(s: TruncatedSample, model: Model, theta: Tensor) -> int:
    """Observations at ``ll + 1`` whose fitted mean has run to zero on a flat likelihood.

    ``Pr(y = ll + 1 | y > ll)`` tends to one as the mean goes to zero, so a
    regressor (or a combination of regressors) that singles out observations
    whose outcomes all equal ``ll + 1`` has a coefficient that diverges to
    minus infinity: the likelihood is monotone (the analogue of separation in
    a binary model). Its gradient and curvature vanish together along that
    direction, so Newton-Raphson can stop there as if it had converged. The
    certificate has two parts:

    1. some observation with ``y = ll + 1`` has a fitted mean below 1e-8, and
    2. the information has collapsed along a direction: the smallest
       generalized eigenvalue of ``-H = X' diag(w Var_i) X`` relative to
       ``X' diag(w) X`` (the average conditional variance along the least
       informative direction) times N is below 0.01. One observation above
       ``ll + 1`` in the support of a direction already makes it at least one,
       so an extreme regressor value alone (a tiny fitted mean for a few
       observations of an identified model) is never flagged.

    Returns the number of observations of part 1 when both hold, otherwise 0.
    """
    least = s.y == s.limit + 1
    if not bool(least.any()):
        return 0
    mean = torch.exp(model.objective.indices(theta)[0])
    count = int((least & (mean < _TINY_MEAN)).sum())
    if not count:
        return 0
    information = -model.objective(theta)[2]
    flat = common.flattest_information(information, s.count.x, s.weights)
    return count if flat < common.FLAT_INFORMATION else 0


def _fit_poisson(s: TruncatedSample) -> tuple[Model, optimize.OptimResult]:
    """The truncated Poisson fit (also the start and comparison model of the NB fit)."""
    model = Model(TruncatedPieces(PoissonDensity(), s.y, s.limit), [s.count], s.weights.user)

    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if iteration >= _WATCH_FROM and iteration % _WATCH_EVERY == 0 \
                and _collapsed(s, model, theta):
            raise Boundary(theta.clone(), "mean")

    result, stop = common.run(model, s.start, what="truncated Poisson", weights=s.weights,
                              callback=watch)
    count = _collapsed(s, model, stop.theta if stop is not None else result.theta)
    if count:
        least = "1" if s.command == "hurdle" else "ll + 1"
        raise AnalysisError(
            "separation_detected",
            f"{s.command}: the fitted mean of {count} observation(s) runs to zero and the "
            "likelihood keeps increasing, so finite estimates do not exist. A regressor (or a "
            "combination of regressors) of the count equation singles out observations whose "
            f"'{s.frame.spec.outcome}' always equals {least}, the smallest outcome the truncated "
            "count model admits, and a coefficient diverges. Remove that regressor or merge "
            "the category it identifies with another one.")
    if result is None or not result.converged:
        if result is None:
            raise common.not_converged(s.command, "the fitted means run to zero",
                                       "Check the scale of the regressors and the truncation "
                                       "point.")
        hint = ""
        mu = torch.exp(model.objective.indices(result.theta)[0])
        if bool((mu < 1e-8).any()):
            hint = ("Some fitted means are numerically zero: a regressor may identify a group "
                    "whose outcomes all equal the smallest admissible value ll + 1. ")
        raise common.not_converged(
            s.command, result.diagnostics.get("message"),
            hint + "Check the scale of the regressors and the truncation point.")
    return model, result


def _constant_only(s: TruncatedSample, pieces: Any, start: Tensor,
                   extra: Sequence[Block] = ()) -> tuple[optimize.OptimResult | None, bool]:
    """Fit of the constant-only model: ``(result, False)`` when it converges.

    Otherwise ``(None, at_zero)`` where ``at_zero`` says that the run failed
    because its dispersion ran to zero (the Poisson boundary), as opposed to a
    dispersion that grows without bound or any other failure.
    """
    constant = common.constant_block(len(s.y), s.frame.spec.outcome, s.offset)
    null = Model(pieces, [constant, *extra], s.weights.user)
    try:
        result, stop = common.run(null, start, what=f"{s.command} (constant-only model)",
                                  weights=s.weights,
                                  callback=common.alpha_watch(1) if extra else None)
    except AnalysisError as exc:
        if exc.code != "precision_unsupported":
            raise
        s.frame.warn("The optional constant-only comparison exceeded count/shape precision; the model Wald test is reported.")
        return None, False
    if result is not None and result.converged:
        return result, False
    theta = stop.theta if stop is not None else result.theta
    return None, bool(extra) and float(theta[1]) < math.log(common.ALPHA_BOUNDARY)


_NO_NULL = ("The constant-only comparison model did not converge; the Wald test of the slopes "
            "is reported instead of the LR test.")


def _null_poisson(s: TruncatedSample, model: Model, result: optimize.OptimResult,
                  ) -> tuple[float | None, str | None]:
    if s.count.size == 1:
        return result.value, "the fitted model (no slopes)"
    fitted, _ = _constant_only(s, model.objective.pieces, result.theta[:1])
    if fitted is None:
        s.frame.warn(_NO_NULL)
        return None, None
    return fitted.value, "constant only, same offset and weights"


def _null_negbin(s: TruncatedSample, pieces: Any, result: optimize.OptimResult, scalar: Block,
                 poisson_pieces: Any, poisson: optimize.OptimResult,
                 ) -> tuple[float | None, str | None]:
    """Constant-only truncated negative binomial model with a free dispersion."""
    k = s.count.size
    if k == 1:
        return result.value, "the fitted model (no slopes)"
    start = torch.cat([result.theta[:1], result.theta[k:]])
    fitted, at_zero = _constant_only(s, pieces, start, [scalar])
    if fitted is not None:
        return fitted.value, "constant only with a free dispersion, same offset and weights"
    if not at_zero:
        # Typically the dispersion of the constant-only model grows without bound (the
        # truncated counts look like a logarithmic-series distribution): its likelihood has
        # a supremum but no maximum, and the Poisson model is NOT its limit.
        s.frame.warn("The constant-only truncated negative binomial model has no maximum (its "
                     "dispersion does not settle at a finite value); the Wald test of the "
                     "slopes is reported instead of the LR test, and no pseudo R-squared.")
        return None, None
    fallback, _ = _constant_only(s, poisson_pieces, poisson.theta[:1])
    if fallback is None:
        s.frame.warn(_NO_NULL)
        return None, None
    s.frame.warn("The constant-only truncated negative binomial model is at the Poisson "
                 "boundary (alpha = 0); its truncated Poisson log likelihood is used as the "
                 "null model.")
    return fallback.value, "constant-only truncated Poisson model (alpha at the boundary)"


_FLAT_DISPERSION = "the likelihood is flat in the dispersion, which is running to zero"
_FLAT_MEAN = "the likelihood is flat in the direction in which fitted means run to zero"


def _flat(s: TruncatedSample, model: Model, theta: Tensor, hessian: Tensor) -> str | None:
    """Why a truncated negative binomial iterate lies on a flat likelihood, or None.

    Two limits of the model lie outside its parameter space and attract the
    iteration when the data favour them; gradient and curvature vanish
    together on the way, so the convergence rules can be met at an arbitrary
    point. Each is recognized by ``common.flattest_information``:

    * the Poisson limit (dispersion to zero): no information about the
      logarithm of the dispersion, which is negative;
    * the logarithmic-series limit (means to zero, with the dispersion of the
      mean form growing without bound): no information along some direction
      of the count equation while fitted means are below 1e-6, or along a
      joint direction of all parameters while the dispersion exceeds one.
    """
    k = s.count.size
    if not bool(torch.isfinite(hessian).all()):
        return None
    if float(theta[k]) < 0 and common.flattest_information(
            -hessian[k, k], None, s.weights) < common.FLAT_INFORMATION:
        return _FLAT_DISPERSION
    mean = torch.exp(model.objective.indices(theta)[0])
    if bool((mean < 1e-6).any()) and common.flattest_information(
            -hessian[:k, :k], s.count.x, s.weights) < common.FLAT_INFORMATION:
        return _FLAT_MEAN
    if float(theta[k]) > 0 and common.flattest_information(
            -hessian, [s.count.x, None], s.weights) < common.FLAT_INFORMATION:
        return _FLAT_MEAN
    return None


def fit_truncated(s: TruncatedSample, dispersion: str | None, *, intercept: bool,
                  poisson_command: str = "oe.tpoisson") -> TruncatedFit:
    """Fit the truncated Poisson (``dispersion=None``) or negative binomial model.

    ``intercept`` says whether a constant-only comparison model exists;
    ``poisson_command`` is the advice given when the dispersion is estimated
    at zero.
    """
    poisson_model, poisson = _fit_poisson(s)
    if dispersion is None:
        null, null_model = _null_poisson(s, poisson_model, poisson) if intercept \
            else (None, None)
        return TruncatedFit(poisson_model, poisson, poisson, null, null_model, None)
    name = "alpha" if dispersion == "mean" else "delta"
    top = int(s.limit.max()) if isinstance(s.limit, Tensor) else int(s.limit)
    if (top + 1) * max(len(s.y), 3000) > _MAX_TAIL_TERMS:
        raise AnalysisError(
            "truncation_too_large",
            f"{s.command}: the negative binomial probability Pr(y > ll) is the complement of a "
            f"sum of ll + 1 terms per observation; with ll up to {top} and {len(s.y)} "
            "observations one likelihood evaluation would take far too long. Use oe.tpoisson "
            "(whose tail is the incomplete gamma function and costs the same for every ll), or "
            "measure the outcome in excess of a lower truncation point.")
    pieces = TruncatedPieces(NegBinDensity(dispersion), s.y, s.limit)
    scalar = common.scalar_block(f"/ln{name}")
    model = Model(pieces, [s.count, scalar], s.weights.user)
    k = s.count.size
    result, theta, message = None, poisson.theta, ""
    for value in (s.moment, 0.1, 1.0):
        start = torch.cat([poisson.theta, torch.tensor([math.log(value)], dtype=torch.float64)])
        try:
            attempt, stop = common.run(model, start, what="truncated negative binomial",
                                       weights=s.weights, callback=common.alpha_watch(k))
        except AnalysisError as exc:
            if exc.code not in {"invalid_start", "precision_unsupported"}:
                raise
            if exc.code == "precision_unsupported":
                model.objective.precision_rejected = True
            message = str(exc)
            continue
        if attempt is not None and attempt.converged:
            flat = _flat(s, model, attempt.theta, attempt.hessian)
            if flat is not None:
                # "Converged" on the flat part of a likelihood without a maximum: the
                # iteration stopped because gradient and curvature vanished together.
                # The other starting values may still lead to a genuine maximum.
                theta, message = attempt.theta, flat
                continue
            result = attempt
            break
        theta = stop.theta if stop is not None else attempt.theta
        message = "the dispersion runs to zero" if stop is not None \
            else str(attempt.diagnostics.get("message"))
    if result is None:
        if model.objective.precision_rejected:
            # A true flat boundary remains distinct from an optimizer that
            # cannot certify an interior maximum within numerical precision.
            certified = _flat(s, model, theta, model.objective(theta)[2]) if len(theta) > k else None
            if certified not in {_FLAT_DISPERSION, _FLAT_MEAN}:
                raise AnalysisError("precision_unsupported", f"{s.command}: count/shape precision prevented a certified maximum; no rounded likelihood is reported.")
            message = certified
        if len(theta) > k and (float(theta[k]) < math.log(common.ALPHA_BOUNDARY)
                               or message == _FLAT_DISPERSION):
            raise AnalysisError(
                "boundary_solution",
                f"{s.command}: the maximum likelihood estimate of {name} lies at zero: the "
                "truncated counts are not overdispersed relative to Poisson, so the "
                f"likelihood has no interior maximum. Use {poisson_command}.")
        hint = ""
        if len(theta) > k and message != _FLAT_MEAN:
            # A run that stopped short of convergence on a ridge towards a limit.
            message = _flat(s, model, theta, model.objective(theta)[2]) or message
        if message == _FLAT_DISPERSION:
            raise AnalysisError(
                "boundary_solution",
                f"{s.command}: the maximum likelihood estimate of {name} lies at zero: the "
                "truncated counts are not overdispersed relative to Poisson, so the "
                f"likelihood has no interior maximum. Use {poisson_command}.")
        if message == _FLAT_MEAN:
            raise AnalysisError(
                "boundary_solution",
                f"{s.command}: the likelihood has no interior maximum: it keeps increasing, "
                "ever more slowly, as fitted means run to zero"
                + (f" and {name} grows without bound" if dispersion == "mean" else "")
                + ". In that limit the truncated negative binomial becomes a logarithmic-"
                "series distribution, which these counts follow more closely than any "
                "negative binomial with finite parameters. Use fewer regressors, the other "
                f"dispersion form or {poisson_command} with covariance='robust'.")
        if len(theta) > k and float(theta[k]) > math.log(1e4):
            hint = (f"The estimate of {name} diverges: the counts are more dispersed than a "
                    "truncated negative binomial with finite dispersion can describe. ")
        raise common.not_converged(
            s.command, message, hint + "If Pr(y > ll) is below 1e-12 for some observations "
            "the truncated likelihood cannot be evaluated; check the truncation point and "
            "the scale of the regressors.")
    null, null_model = (None, None)
    if intercept:
        null, null_model = _null_negbin(s, pieces, result, scalar,
                                        poisson_model.objective.pieces, poisson)
    return TruncatedFit(model, result, poisson, null, null_model, dispersion)


def _report(s: TruncatedSample, fitted: TruncatedFit, limit_record: dict[str, Any],
            ) -> ResultBundle:
    frame, spec = s.frame, s.frame.spec
    model, result, dispersion = fitted.model, fitted.result, fitted.dispersion
    nobs, k = s.weights.nobs, s.count.size
    log_likelihood, null = result.value, fitted.null
    theta, covariance, info = common.covariance_of(frame, model, result.theta, result.hessian,
                                                   s.weights)
    slopes = slope_indices(s.design.terms)
    tests: dict[str, Any] = {"model": common.model_test(frame, theta, covariance, slopes,
                                                        log_likelihood, null, "count")}
    criteria = information_criteria(log_likelihood, model.size, nobs)
    metrics: dict[str, Any] = {
        "log_likelihood": log_likelihood,
        "pseudo_r_squared": common.pseudo_r_squared(log_likelihood, null),
        "aic": criteria["aic"], "bic": criteria["bic"]}
    record = {**limit_record, "null_log_likelihood": null, "null_model": fitted.null_model,
              "fitted_values": "conditional mean E[y | y > ll]"}
    if dispersion is not None:
        name = "alpha" if dispersion == "mean" else "delta"
        log_value = float(theta[k])
        metrics[name] = safe_exp(log_value)
        record.update({
            "dispersion": dispersion,
            "variance_function": "mu (1 + alpha mu)" if dispersion == "mean"
            else "mu (1 + delta)",
            "alpha": common.exponentiated(log_value, math.sqrt(float(covariance[k, k])),
                                          spec.alpha, name),
            "tpoisson_log_likelihood": fitted.poisson.value})
        if spec.covariance in {"nonrobust", "opg"}:
            tests["alpha"] = common.boundary_test(log_likelihood, fitted.poisson.value, name,
                                                  "the truncated Poisson model")
        else:
            record["alpha_test_note"] = (
                "The LR test of the dispersion is not reported with a robust or cluster "
                "covariance (it is not a likelihood-based test).")
    mean = kernel_call(model.objective.pieces.conditional_mean,
                       model.objective.indices(result.theta))
    return build_result(
        frame, terms=model.terms, params=theta, covariance=covariance,
        equations=model.equations if dispersion is not None else None, use_t=False,
        metrics=metrics, fitted=mean, nobs=nobs, inference=info, tests=tests, extra=record,
        categories=s.design.categories, solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result), df_resid=nobs - model.size)


def fit_tpoisson(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``tpoisson``: ``(spec, data) -> ResultBundle``."""
    s, record = _prepare(spec, data, "tpoisson")
    return _report(s, fit_truncated(s, None, intercept=spec.intercept), record)


def fit_tnbreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``tnbreg``: ``(spec, data) -> ResultBundle``."""
    s, record = _prepare(spec, data, "tnbreg")
    fitted = fit_truncated(s, s.frame.option("dispersion"), intercept=spec.intercept)
    return _report(s, fitted, record)


def _truncation(ll: int | float | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``ll`` as a column role (a name) or an option (a nonnegative integer)."""
    if isinstance(ll, str):
        return {"truncation": ll}, {}
    if isinstance(ll, bool) or not isinstance(ll, (int, float)) or not math.isfinite(ll) \
            or ll != int(ll) or ll < 0:
        raise AnalysisError("invalid_spec", "ll must be a nonnegative integer or the name of a "
                            "column of truncation points.")
    return {}, {"ll": int(ll)}


def tpoisson(*, data: Any, y: str, x: Sequence[str], ll: int | str = 0,
             covariance: str | None = None, cluster: str | Sequence[str] | None = None,
             weights: str | None = None, weight_type: str | None = None,
             offset: str | None = None, exposure: str | None = None,
             categorical: Sequence[str] | None = None, intercept: bool = True,
             missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Truncated Poisson regression, Stata's ``tpoisson`` (``ztp`` for ``ll=0``).

    Model
        Counts are observed only when they exceed the truncation point ``ll``. With
        ``mu = exp(x'b + offset)`` the Poisson mean of the untruncated count,

            Pr(y | y > ll) = exp(-mu) mu^y / y! / Pr(Y > ll),
            Pr(Y > ll) = 1 - sum_{j <= ll} exp(-mu) mu^j / j! = P(ll + 1, mu),

        ``P`` the regularized incomplete gamma function. The log likelihood is
        globally concave in ``b``; it is maximized by Newton-Raphson with the analytic
        score ``X' w (y - E[y | y > ll])`` and Hessian ``-X' diag(w Var[y | y > ll]) X``
        from Poisson starting values.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: integer count outcome; every value must exceed the truncation point.
        x: list of regressors (``[]``: constant only).
        ll: truncation point: a nonnegative integer (default 0, zero truncation) or
            the name of a column with observation-specific truncation points.
        covariance: ``'nonrobust'`` (default: inverse observed information),
            ``'opg'``, ``'robust'`` (sandwich times ``N/(N-1)``), ``'cluster'``
            (``G/(G-1)``; one or two cluster columns).
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        offset / exposure: ``offset`` enters with coefficient 1; ``exposure`` enters
            as ``ln(exposure)``. Mutually exclusive.
        categorical: columns of ``x`` to expand into indicator terms.
        intercept: include the constant (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics; coefficients are those of the untruncated Poisson mean (as in
        Stata, ``exp(b)`` is an incidence-rate ratio of the latent count).
        ``metrics``: log_likelihood, pseudo_r_squared (``1 - ll/ll_0``), aic, bic.
        ``tests['model']``: LR chi2 against the constant-only model (``nonrobust``)
        or the Wald chi2 of the slopes. ``extra``: the truncation point and the
        constant-only log likelihood. The chart sample (``predictions``) uses the
        conditional mean ``E[y | y > ll]``.

        An outcome at or below ``ll`` raises ``outcome_not_truncated``; non-integer
        outcomes raise ``invalid_count_outcome``. A regressor that singles out
        observations whose outcomes all equal ``ll + 1`` has no finite coefficient
        (the likelihood keeps increasing as their mean goes to zero):
        ``separation_detected``.

    Stata
        ``tpoisson stay age i.hmo, ll(0) vce(robust)`` is
        ``oe.tpoisson(data=df, y='stay', x=['age', 'hmo'], categorical=['hmo'],
        covariance='robust')``; ``ll(minstay)`` with a variable is ``ll='minstay'``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=4000)})
        >>> df["y"] = rng.poisson(np.exp(0.3 + 0.5 * df.x))
        >>> df = df[df.y > 0]                        # zeros are never observed
        >>> print(oe.tpoisson(data=df, y="y", x=["x"]).summary())
    """
    columns, options = _truncation(ll)
    spec = build_spec(
        "tpoisson", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={**columns, "offset": offset, "exposure": exposure}, options=options)
    return fit(spec, data=data)


def tnbreg(*, data: Any, y: str, x: Sequence[str], ll: int | str = 0,
           dispersion: str = "mean", covariance: str | None = None,
           cluster: str | Sequence[str] | None = None, weights: str | None = None,
           weight_type: str | None = None, offset: str | None = None,
           exposure: str | None = None, categorical: Sequence[str] | None = None,
           intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Truncated negative binomial regression, Stata's ``tnbreg`` (``ztnb`` for ``ll=0``).

    Model
        As :func:`tpoisson` with the negative binomial distribution: mean
        ``mu = exp(x'b + offset)`` of the untruncated count and

        * ``dispersion='mean'`` (NB2, default): ``Var = mu (1 + alpha mu)``;
        * ``dispersion='constant'`` (NB1): ``Var = mu (1 + delta)``.

        ``ln Pr(y | y > ll) = ln f(y) - ln{1 - sum_{j <= ll} f(j)}``; for zero
        truncation and NB2 the normalizer is ``1 - (1 + alpha mu)^(-1/alpha)``.
        ``(b, ln alpha)`` are estimated jointly by Newton-Raphson with analytic
        derivatives, starting from the truncated Poisson estimates.

    Parameters
        data: the data set (pandas DataFrame, dict of columns or row records).
        y: integer count outcome; every value must exceed the truncation point.
        x: list of regressors (``[]``: constant only).
        ll: truncation point: a nonnegative integer (default 0) or the name of a
            column with observation-specific truncation points. The cost of one
            likelihood evaluation grows with the largest truncation point.
        dispersion: ``'mean'`` (NB2) or ``'constant'`` (NB1).
        covariance: ``'nonrobust'`` (default), ``'opg'``, ``'robust'``, ``'cluster'``.
        cluster: cluster column(s); implies ``covariance='cluster'``.
        weights, weight_type: ``'fweight'``, ``'aweight'``, ``'iweight'``,
            ``'pweight'`` (robust by default).
        offset / exposure: offset, or exposure entering as ``ln(exposure)``.
        categorical: columns of ``x`` to expand into indicator terms.
        intercept: include the constant (default True).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing inputs.
        alpha: significance level of the confidence intervals (default 0.05).

    Result
        z statistics. Terms: the regressors (equation = the outcome) and ``/lnalpha``
        (``/lndelta`` for NB1). ``metrics``: log_likelihood, pseudo_r_squared, aic,
        bic, alpha (or delta). ``extra['alpha']``: the dispersion with its delta-method
        standard error and the interval exponentiated from its logarithm.
        ``tests['model']``: LR chi2 against the constant-only model with a free
        dispersion (``nonrobust``) or Wald chi2 (also when that comparison model
        has no maximum, with a warning). ``tests['alpha']``: LR test of
        ``alpha = 0`` against :func:`tpoisson` (``chibar2(01)``; likelihood-based
        covariances only).

        Counts that are not overdispersed raise ``boundary_solution`` (use
        :func:`tpoisson`), as do counts at the model's other limit, the
        logarithmic-series distribution (fitted means running to zero); a regressor
        that singles out observations whose outcomes all equal ``ll + 1`` raises
        ``separation_detected``; a truncation point so
        large that the lower sum of ``ll + 1`` terms per observation cannot be
        evaluated in reasonable time (observations times ``ll + 1`` above 3e8)
        raises ``truncation_too_large``.

    Stata
        ``tnbreg stay age i.hmo, ll(0) dispersion(constant)`` is
        ``oe.tnbreg(data=df, y='stay', x=['age', 'hmo'], categorical=['hmo'],
        dispersion='constant')``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=6000)})
        >>> mu = np.exp(0.5 + 0.5 * df.x)
        >>> df["y"] = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))      # alpha = 0.5
        >>> df = df[df.y > 0]
        >>> print(oe.tnbreg(data=df, y="y", x=["x"]).summary())
    """
    columns, options = _truncation(ll)
    spec = build_spec(
        "tnbreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={**columns, "offset": offset, "exposure": exposure},
        options={**options, "dispersion": dispersion})
    return fit(spec, data=data)
