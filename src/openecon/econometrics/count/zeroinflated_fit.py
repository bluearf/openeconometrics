"""Zero-inflated Poisson and negative binomial regression: Stata's ``zip`` and ``zinb``.

Model
-----
With probability ``F_i = F(z_i'g)`` (``F`` logistic, or standard normal with
``inflate_link='probit'``) observation i is a structural zero; otherwise it is
a draw from the count distribution with mean ``mu_i = exp(x_i'b + offset_i)``:

    Pr(y = 0) = F + (1 - F) f(0),        Pr(y = k) = (1 - F) f(k),  k >= 1,

``f`` Poisson (``zip``) or negative binomial with ``Var = mu (1 + alpha mu)``
(``zinb``, ancillary parameter ``/lnalpha``). ``E[y|x, z] = (1 - F) mu``.

Estimation
----------
All parameters ``(b, g[, ln alpha])`` are estimated jointly by Newton-Raphson
with the analytic score and Hessian of ``kernels.ZeroInflatedPieces`` (the zero
mixture is a log-sum-exp). ``zip`` starts from the Poisson estimates of ``b``
and a binary fit of ``1[y = 0]`` on the inflation regressors (falling back to a
constant matching the share of excess zeros); ``zinb`` starts from the ``zip``
estimates and a dispersion taken from the negative binomial fit.

Reported (as Stata)
-------------------
* ``tests['model']``: LR chi2 that the slopes of the count equation are zero -
  the comparison model has a constant-only count equation and the full
  inflation equation - under the conventional covariance; the Wald chi2 of the
  same slopes otherwise.
* ``tests['vuong']``: Vuong (1989) test against the model without inflation
  (Poisson for ``zip``, negative binomial for ``zinb``): with
  ``m_i = ln f1(y_i) - ln f2(y_i)``, ``V = sqrt(N) mean(m) / sd(m)`` (``sd``
  with divisor ``N - 1``), standard normal; large positive values favour the
  zero-inflated model and the p-value is the upper tail ``Pr(Z > V)`` that
  Stata's ``vuong`` option printed. ``extra['vuong']`` adds the AIC- and
  BIC-corrected statistics (Desmarais and Harden 2013), which subtract
  ``(k1 - k2)/N`` and ``(k1 - k2) ln(N)/(2N)`` from ``mean(m)``.
* ``tests['alpha']`` (``zinb``): LR test of ``alpha = 0`` against the
  zero-inflated Poisson model with the boundary-corrected ``chibar2(01)``
  p-value.

Boundaries
----------
If the data hold no excess zeros the inflation probability goes to zero and
the likelihood has no interior maximum; the same happens to ``alpha`` in
``zinb`` when the counts are not overdispersed. Both are reported as
``boundary_solution`` with the model to use instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, information_criteria, kernel_call,
)
from openecon.econometrics.count import common
from openecon.econometrics.count.common import Block, Boundary, Model
from openecon.econometrics.count.kernels import (
    BinaryPieces, IndexObjective, NegBinDensity, PoissonDensity, ZeroInflatedPieces,
    link_pieces,
)
from openecon.econometrics.glm.common import (
    Weights, optimizer_record, require_observations, require_terms, safe_exp, slope_indices,
)
from openecon.econometrics.glm.glm import GlmFit
from openecon.econometrics.glm.kernels import NegativeBinomialObjective
from openecon.engines import optimize
from openecon.engines.count_numeric import poisson_logmass
from openecon.engines.distributions import normal_sf
from openecon.models import ModelSpec, ResultBundle

# Fitted inflation probabilities below / above these are numerically degenerate.
_EXTREME = 1e-6
_WATCH_FROM, _WATCH_EVERY = 15, 5


@dataclass
class _Sample:
    frame: ModelFrame
    command: str
    y: Tensor
    zero: Tensor
    weights: Weights
    offset: Tensor | None
    link: str
    design: Design
    inflate: Design
    count: Block
    infl: Block
    poisson: GlmFit


def _prepare(spec: ModelSpec, data: Any, command: str) -> _Sample:
    frame = ModelFrame(spec, data)
    y, weights, offset = common.count_sample(frame, command)
    zero = y == 0
    if not bool(zero.any()):
        raise AnalysisError(
            "no_zero_outcomes", f"'{spec.outcome}' has no zero observations, so there is "
            "nothing to inflate and the inflation equation is not identified. Use "
            + ("oe.poisson." if command == "zip" else "oe.nbreg."))
    if bool(zero.all()):
        raise AnalysisError("constant_outcome", f"'{spec.outcome}' is zero in every observation; "
                            "the model is not identified.")
    design = frame.drop_collinear(frame.design(), weights.for_screen())
    require_terms(design)
    inflate = frame.drop_collinear(
        frame.design(frame.role("inflate"), intercept=True, prefix="inflate:"),
        weights.for_screen())
    require_observations(weights.nobs, len(design.terms) + len(inflate.terms)
                         + int(command == "zinb"))
    poisson = common.poisson_start(frame, design, y, weights, offset, command)
    w = weights.user
    return _Sample(frame, command, y, zero, weights, offset, frame.option("inflate_link"),
                   design, inflate, common.design_block(design, w, spec.outcome, offset),
                   common.design_block(inflate, w, "inflate"), poisson)


def _probability(model: Model, theta: Tensor, link: str) -> Tensor:
    """Fitted inflation probabilities ``F(z'g)`` at (centered) ``theta``."""
    return torch.exp(link_pieces(link, model.objective.indices(theta)[1], False)[0])


def _inflation_starts(s: _Sample) -> list[Tensor]:
    """Starting values of the inflation equation (centered design).

    First the binary fit of ``1[y = 0]`` on the inflation regressors, then a
    constant placed at the share of excess zeros left by the Poisson fit.
    """
    w, q = s.weights.user, s.infl.size
    total = float(w.sum())
    share = float((w * s.zero).sum()) / total
    starts = []
    binary = kernel_call(IndexObjective, BinaryPieces(s.zero, s.link), [s.infl.x], w)
    start = torch.zeros(q, dtype=torch.float64)
    start[0] = common.link_inverse(s.link, share)
    try:
        result = common.maximize(binary, start, what=f"{s.command} (inflation starting values)",
                                 scale=common.weight_scale(s.weights), max_iter=25)
        if result.converged:
            starts.append(result.theta)
    except AnalysisError:
        pass
    predicted = float((w * torch.exp(-s.poisson.state.mu)).sum()) / total
    excess = (share - predicted) / max(1 - predicted, 1e-12)
    constant = torch.zeros(q, dtype=torch.float64)
    constant[0] = common.link_inverse(s.link, min(max(excess, 0.02), 0.95))
    starts.append(constant)
    return starts


def _watch(model: Model, link: str, alpha_position: int | None = None):
    """Stop a run whose dispersion or inflation probability has collapsed to zero."""
    position = model.starts[1]
    threshold = common.link_inverse(link, _EXTREME)
    alpha_stop = common.alpha_watch(alpha_position) if alpha_position is not None else None

    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if alpha_stop is not None:
            alpha_stop(iteration, theta, value)
        if iteration >= _WATCH_FROM and iteration % _WATCH_EVERY == 0 \
                and float(theta[position]) < threshold \
                and float(_probability(model, theta, link).max()) < _EXTREME ** 2:
            raise Boundary(theta.clone(), "inflation")

    return watch


_FLAT = "the likelihood is flat in a direction that is running to a boundary: "


def _solve(model: Model, starts: Sequence[Tensor], what: str, weights: Weights, callback,
           degenerate=None) -> tuple[optimize.OptimResult | None, Tensor, str]:
    """Try the starting values in turn; return the converged run or the last failure.

    ``degenerate(result)`` names the limit on whose flat approach a run has
    "converged" (or returns None); such a run counts as a failure, with the
    message ``_FLAT + limit``, and the next starting values are tried.
    """
    theta, message = starts[0], "no starting values"
    precision_error = None
    attempted = False
    for start in starts:
        try:
            result, stop = common.run(model, start, what=what, weights=weights, callback=callback)
        except AnalysisError as exc:
            if exc.code != "precision_unsupported":
                raise
            precision_error = exc
            continue
        attempted = True
        if result is not None and result.converged:
            limit = degenerate(result) if degenerate is not None else None
            if limit is None:
                return result, result.theta, ""
            theta, message = result.theta, _FLAT + limit
        elif stop is not None:
            theta, message = stop.theta, f"the {stop.kind} parameter runs to its boundary"
        else:
            theta, message = result.theta, str(result.diagnostics.get("message"))
    if precision_error is not None and not attempted:
        raise precision_error
    if precision_error is not None:
        model.objective.precision_rejected = True
    return None, theta, message


def _poisson_dispersion_boundary(s: _Sample, model: Model, theta: Tensor,
                                 alpha_position: int) -> bool:
    """Certify a small, flat NB dispersion with a nonpositive ZIP-limit score.

    A rejected numerical trial alone is not evidence for a Poisson boundary.
    The score below is evaluated with the stable Poisson likelihood, including
    posterior count membership of zero observations in the mixture.
    """
    if float(theta[alpha_position]) >= math.log(common.ALPHA_BOUNDARY):
        return False
    indices = model.objective.indices(theta)
    mu = torch.exp(indices[0])
    limit_rows = ZeroInflatedPieces(PoissonDensity(), s.y, s.link)(indices[:2], False)[0]
    log_count_membership = link_pieces(s.link, indices[1], False)[1] - mu - limit_rows
    membership = torch.where(s.zero, torch.exp(log_count_membership), 1.)
    direction = (s.weights.user * membership * ((s.y - mu).square() - s.y) / 2).sum()
    value, _, hessian = model.objective(theta)
    weak = common.flattest_information(-hessian[alpha_position, alpha_position], None,
                                       s.weights)
    limit = (s.weights.user * limit_rows).sum()
    return (float(direction) <= 0 and abs(weak) < common.FLAT_INFORMATION
            and abs(float(value - limit)) <= 1e-6 * max(1., abs(float(limit))))


def _fail(s: _Sample, model: Model, theta: Tensor, message: str,
          alpha_position: int | None = None, plain: float | None = None, *,
          flat_alpha: bool = False) -> None:
    """The most informative error for a zero-inflated likelihood without a maximum.

    ``plain`` is the log likelihood of the model without inflation: a failed
    run that has not improved on it while its inflation probabilities shrink
    is at the no-inflation boundary. ``flat_alpha`` says that the run stopped
    where the likelihood is flat in a dispersion that was heading for zero.
    """
    command = s.command
    alpha_boundary = (alpha_position is not None and
        (_poisson_dispersion_boundary(s, model, theta, alpha_position)
         if model.objective.precision_rejected else
         flat_alpha or float(theta[alpha_position]) < math.log(common.ALPHA_BOUNDARY)))
    if alpha_boundary:
        raise AnalysisError(
            "boundary_solution",
            "zinb: the maximum likelihood estimate of alpha lies at zero: beyond the excess "
            "zeros the counts are not overdispersed relative to Poisson, so the likelihood "
            "has no interior maximum. Use oe.zip.")
    low = high = 0
    collapsed = False
    try:
        probability = _probability(model, theta, s.link)
        low = int((probability < _EXTREME).sum())
        high = int((probability > 1 - _EXTREME).sum())
        collapsed = low == len(probability)
        if not collapsed and plain is not None and float(probability.max()) < 1e-2:
            value = float(model.objective.value(theta))
            collapsed = value <= plain + 1e-6 * max(1.0, abs(plain))
    except (RuntimeError, ValueError):
        pass
    if collapsed:
        without = "oe.poisson" if command == "zip" else "oe.nbreg"
        raise AnalysisError(
            "boundary_solution",
            f"{command}: the estimated inflation probability lies at zero: the data hold no "
            "more zeros than the count model predicts, so the inflation equation is not "
            f"identified and the likelihood has no interior maximum. Use {without}.")
    if low or high:
        raise AnalysisError(
            "separation_detected",
            f"{command}: the inflation equation predicts {low + high} observation(s) "
            "perfectly (fitted inflation probabilities numerically 0 or 1) and the likelihood "
            "keeps increasing, so finite estimates do not exist. A regressor of the inflation "
            "equation may identify a group without zeros (or with zeros only): remove or "
            "combine it.")
    if model.objective.precision_rejected:
        raise AnalysisError("precision_unsupported", f"{command}: numerical count/shape precision prevented a certified maximum; no rounded likelihood is reported.")
    raise common.not_converged(command, message)


def _degenerate(s: _Sample, model: Model, result: optimize.OptimResult,
                alpha_position: int | None = None) -> str | None:
    """Why a "converged" run sits on the flat part of a likelihood without a maximum.

    Next to ``alpha = 0`` and where fitted inflation probabilities reach 0 or
    1 the gradient and the curvature vanish together, so the iteration can
    meet its convergence rules at an arbitrary point. Such a point is
    recognized by a parameter block that receives less than
    ``common.FLAT_INFORMATION`` observations' worth of information: the
    logarithm of a dispersion below one (returns ``'alpha'``), or the
    inflation equation when fitted inflation probabilities are numerically 0
    or 1 (returns ``'inflation'``). Otherwise None.
    """
    hessian = result.hessian
    if alpha_position is not None and float(result.theta[alpha_position]) < 0 \
            and common.flattest_information(-hessian[alpha_position, alpha_position], None,
                                            s.weights) < common.FLAT_INFORMATION:
        return "alpha"
    probability = _probability(model, result.theta, s.link)
    if not bool(((probability < _EXTREME) | (probability > 1 - _EXTREME)).any()):
        return None
    block = slice(model.starts[1], model.starts[1] + s.infl.size)
    flat = common.flattest_information(-hessian[block, block], s.infl.x, s.weights)
    return "inflation" if flat < common.FLAT_INFORMATION else None


def _fit_zip(s: _Sample) -> tuple[Model, optimize.OptimResult | None, Tensor, str]:
    model = Model(ZeroInflatedPieces(PoissonDensity(), s.y, s.link), [s.count, s.infl],
                  s.weights.user)
    starts = [torch.cat([s.poisson.state.beta, gamma]) for gamma in _inflation_starts(s)]
    result, theta, message = _solve(model, starts, "zero-inflated Poisson", s.weights,
                                    _watch(model, s.link),
                                    lambda run: _degenerate(s, model, run))
    return model, result, theta, message


def _null(s: _Sample, pieces: Any, result: optimize.OptimResult, model: Model,
          ) -> float | None:
    """Log likelihood with a constant-only count equation and the full other equations."""
    if not s.frame.spec.intercept:
        return None
    if s.count.size == 1:
        return result.value
    constant = common.constant_block(s.frame.n, s.frame.spec.outcome, s.offset)
    null = Model(pieces, [constant, *model.blocks[1:]], s.weights.user)
    k = s.count.size
    start = torch.cat([result.theta[:1], result.theta[k:]])
    try:
        fitted, _ = common.run(null, start, what=f"{s.command} (constant-only model)",
                               weights=s.weights)
    except AnalysisError as exc:
        if exc.code != "precision_unsupported":
            raise
        s.frame.warn("The optional constant-only comparison exceeded count/shape precision; "
                     "the model Wald test is reported instead of the LR test.")
        return None
    if fitted is None or not fitted.converged:
        s.frame.warn("The constant-only comparison model did not converge; the Wald test of "
                     "the count equation's slopes is reported instead of the LR test.")
        return None
    return fitted.value


def _vuong(difference: Tensor, weights: Weights, extra_parameters: int, label: str,
           ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Vuong's statistic from the per-observation log likelihood differences."""
    if weights.frequency is not None:
        f, n = weights.frequency, weights.nobs
        mean = float((f * difference).sum()) / n
        variance = float((f * (difference - mean).square()).sum()) / (n - 1)
    else:
        n = len(difference)
        mean, variance = float(difference.mean()), float(difference.var())
    if not variance > 0:
        return None
    sd, root = math.sqrt(variance), math.sqrt(n)
    z = root * mean / sd
    record = {"z": z, "z_aic": root * (mean - extra_parameters / n) / sd,
              "z_bic": root * (mean - extra_parameters * math.log(n) / (2 * n)) / sd,
              "mean_difference": mean, "sd_difference": sd, "observations": n,
              "extra_parameters": extra_parameters,
              "p_value_two_sided": 2 * kernel_call(normal_sf, abs(z))}
    for name in ("z_aic", "z_bic"):
        record[f"p_value_{name[2:]}"] = kernel_call(normal_sf, record[name])
    test = {"statistic": z, "p_value": kernel_call(normal_sf, z), "distribution": "normal",
            "label": label}
    return test, record


def _negative_binomial(s: _Sample) -> tuple[float, Tensor, float] | None:
    """The plain NB2 fit: ``(alpha, per-observation log likelihood, log likelihood)``."""
    w, y, mu = s.weights.user, s.y, s.poisson.state.mu
    moment = float((w * ((y - mu).square() - mu)).sum() / (w * mu.square()).sum())
    moment = min(max(moment if math.isfinite(moment) else 1.0, 0.05), 50.0)
    try:
        objective = kernel_call(NegativeBinomialObjective, s.count.x, y, w, s.offset, "mean")
        start = torch.cat([s.poisson.state.beta,
                           torch.tensor([math.log(moment)], dtype=torch.float64)])
        result = common.maximize(objective, start, what="negative binomial",
                                 scale=common.weight_scale(s.weights))
    except AnalysisError:
        return None
    if not result.converged:
        return None
    k = s.count.size
    eta = objective.predictor(result.theta)
    rows = NegBinDensity().terms(y, torch.lgamma(y + 1), [eta, result.theta[k]], False)[0]
    return safe_exp(float(result.theta[k])), rows, result.value


def _report(s: _Sample, model: Model, pieces: Any, result: optimize.OptimResult,
            tests: dict[str, Any], extra: dict[str, Any], alpha: bool) -> ResultBundle:
    frame, spec = s.frame, s.frame.spec
    k, q = s.count.size, s.infl.size
    nobs = s.weights.nobs
    log_likelihood = result.value
    theta, covariance, info = common.covariance_of(frame, model, result.theta, result.hessian,
                                                   s.weights)
    null = _null(s, pieces, result, model)
    slopes = slope_indices(s.design.terms)
    tests = {"model": common.model_test(frame, theta, covariance, slopes, log_likelihood, null,
                                        "count"), **tests}
    criteria = information_criteria(log_likelihood, model.size, nobs)
    metrics: dict[str, Any] = {
        "log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
        "n_zero_observations": common.count_total(s.zero, s.weights)}
    probability = _probability(model, result.theta, s.link)
    indices = model.objective.indices(result.theta)
    w = s.weights.user
    record = {
        "inflate_link": s.link, "null_log_likelihood": null,
        "null_model": "constant-only count equation, full inflation equation",
        "poisson_log_likelihood": s.poisson.log_likelihood(),
        "mean_inflation_probability": float((w * probability).sum() / w.sum()),
        "fitted_values": "E[y|x, z] = (1 - F(z'g)) exp(x'b + offset)", **extra}
    if alpha:
        position = model.size - 1
        log_alpha = float(theta[position])
        metrics["alpha"] = safe_exp(log_alpha)
        record["alpha"] = common.exponentiated(
            log_alpha, math.sqrt(float(covariance[position, position])), spec.alpha, "alpha")
    return build_result(
        frame, terms=model.terms, params=theta, covariance=covariance,
        equations=model.equations, use_t=False, metrics=metrics,
        fitted=(1 - probability) * torch.exp(indices[0]), nobs=nobs, inference=info,
        tests=tests, extra=record,
        categories={**s.design.categories, **s.inflate.categories},
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result), df_resid=nobs - k - q - int(alpha))


def _likelihood_based(s: _Sample) -> bool:
    return s.frame.spec.covariance in {"nonrobust", "opg"}


def _vuong_allowed(s: _Sample, extra: dict[str, Any]) -> bool:
    if s.weights.weighted and s.weights.frequency is None:
        extra["vuong_note"] = ("The Vuong test is defined for unweighted or frequency-weighted "
                               "likelihoods and is not reported with other weights.")
        return False
    if not _likelihood_based(s):
        extra["vuong_note"] = ("The Vuong test compares model-based likelihoods and is not "
                               "reported with a robust or cluster covariance (as in Stata).")
        return False
    return True


def fit_zip(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``zip``: ``(spec, data) -> ResultBundle``."""
    s = _prepare(spec, data, "zip")
    model, result, theta, message = _fit_zip(s)
    if result is None:
        _fail(s, model, theta, message, plain=s.poisson.log_likelihood())
    tests: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    if _vuong_allowed(s, extra):
        state = s.poisson.state
        plain = poisson_logmass(s.y, state.mu, eta=state.eta)
        outcome = _vuong(model.objective.observations(result.theta) - plain, s.weights,
                         s.infl.size, "Vuong test of zip vs. standard Poisson (z > 0 favours "
                         "zip; p-value Pr(Z > z))")
        if outcome is not None:
            tests["vuong"], extra["vuong"] = outcome
    return _report(s, model, model.objective.pieces, result, tests, extra, alpha=False)


def fit_zinb(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``zinb``: ``(spec, data) -> ResultBundle``."""
    s = _prepare(spec, data, "zinb")
    w = s.weights.user
    zip_result = _fit_zip(s)[1]
    plain = _negative_binomial(s)
    pieces = ZeroInflatedPieces(NegBinDensity(), s.y, s.link)
    model = Model(pieces, [s.count, s.infl, common.scalar_block("/lnalpha")], w)
    position = model.size - 1
    if zip_result is not None:
        base = [zip_result.theta]
    else:
        base = [torch.cat([s.poisson.state.beta, gamma]) for gamma in _inflation_starts(s)]
    dispersions = [0.1, 1.0]
    if plain is not None:
        dispersions.insert(0, min(max(0.5 * plain[0], 0.02), 10.0))
    starts = [torch.cat([start, torch.tensor([math.log(value)], dtype=torch.float64)])
              for value in dispersions for start in base]
    result, theta, message = _solve(model, starts, "zero-inflated negative binomial",
                                    s.weights, _watch(model, s.link, position),
                                    lambda run: _degenerate(s, model, run, position))
    if result is None:
        _fail(s, model, theta, message, position, None if plain is None else plain[2],
              flat_alpha=message == _FLAT + "alpha")
    tests: dict[str, Any] = {}
    extra: dict[str, Any] = {"starting_values": "zip estimates and the dispersion of nbreg"}
    if zip_result is not None:
        extra["zip_log_likelihood"] = zip_result.value
        if _likelihood_based(s):
            tests["alpha"] = common.boundary_test(result.value, zip_result.value, "alpha",
                                                  "the zero-inflated Poisson model")
    if zip_result is None or not _likelihood_based(s):
        extra["alpha_test_note"] = (
            "The LR test of alpha = 0 is not reported: "
            + ("it is not a likelihood-based test under a robust or cluster covariance."
               if zip_result is not None else
               "the zero-inflated Poisson comparison model did not converge."))
    if _vuong_allowed(s, extra):
        if plain is None:
            extra["vuong_note"] = ("The negative binomial comparison model did not converge "
                                   "(no overdispersion without inflation); no Vuong test.")
        else:
            extra["nbreg_log_likelihood"] = plain[2]
            outcome = _vuong(model.objective.observations(result.theta) - plain[1], s.weights,
                             s.infl.size, "Vuong test of zinb vs. standard negative binomial "
                             "(z > 0 favours zinb; p-value Pr(Z > z))")
            if outcome is not None:
                tests["vuong"], extra["vuong"] = outcome
    return _report(s, model, pieces, result, tests, extra, alpha=True)
