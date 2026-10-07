"""Conventions shared by the count family's analysis layer.

The family follows the glm family (``openecon.econometrics.glm.common``), whose
helpers it imports rather than copies:

* **Weights** (Stata's semantics for likelihood estimators): ``fweight``
  replicates rows (``N = sum w``), ``aweight`` is rescaled to sum to the number
  of rows, ``iweight`` and ``pweight`` enter as given; ``pweight`` needs a
  robust or cluster covariance (the convenience default).
* **Covariance**: ``core.ml_covariance`` - ``nonrobust`` (inverse observed
  information), ``opg``, ``robust`` (``N/(N-1)`` sandwich) and ``cluster``
  (``G/(G-1)``; two cluster columns by inclusion-exclusion). z statistics.
* **Offset / exposure** enter the count equation with coefficient one.
* **Centering**: every equation with a constant is estimated on regressors
  centered at their weighted means and mapped back exactly
  (``b0 = b0_c - m'b``, ``V = T V_c T'``).
* **Model test** (Stata's convention for multi-equation ``ml`` models): under
  the conventional covariance ``tests['model']`` is the LR chi2 that the slopes
  of the first (count / outcome) equation are zero; the comparison model keeps
  the other equations (inflation, selection, ln alpha) as specified and only
  reduces the first equation to its constant. With any other covariance, or
  without a constant, it is the Wald chi2 of the same slopes.

* **Flat likelihoods**: every model of the family has limits outside its
  parameter space (alpha -> 0, a probability at 0 or 1, a mean at zero). Data
  that favour a limit leave a likelihood that increases towards it while
  gradient and curvature vanish together, so an iteration can meet its
  convergence rules at an arbitrary point. :func:`flattest_information`
  measures the information of the least informative direction of a parameter
  block in observations' worth; below ``FLAT_INFORMATION`` together with the
  symptom of the limit the fit is refused (``boundary_solution`` /
  ``separation_detected``).

A model is a list of :class:`Block` (one per linear index of
``kernels.IndexObjective``) plus the per-observation pieces.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, kernel_call, lr_test, wald_test,
)
from openecon.econometrics.count.kernels import IndexObjective
from openecon.econometrics.glm.common import (
    Weights, center_design, check_pweights, count_outcome, likelihood_covariance,
    likelihood_weights, linear_offset, safe_exp, uncenter,
)
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf
from openecon.engines.inference import critical_value
from openecon.engines.linalg import weighted_crossprod

# A dispersion below this at a non-converged iterate is the Poisson boundary.
ALPHA_BOUNDARY = 1e-4
# Early exit of a run that is visibly heading for a boundary.
_LOG_ALPHA_STOP = math.log(1e-9)


# Information along the flattest direction of a parameter block, in observations' worth,
# below which the likelihood is flat there (see ``flattest_information``): the standard
# error of the index along that direction would exceed ten units of the linear predictor.
FLAT_INFORMATION = 1e-2


class Boundary(Exception):
    """Raised from a Newton callback to stop a run that is heading for a boundary."""

    def __init__(self, theta: Tensor, kind: str):
        super().__init__(kind)
        self.theta, self.kind = theta, kind


@dataclass
class Block:
    """One linear index: reporting terms and the (centered) estimation design."""

    terms: list[str]
    equation: str | None
    x: Tensor | None               # None: a scalar ancillary parameter
    means: Tensor | None = None    # centering of the non-constant columns
    offset: Tensor | None = None
    design: Design | None = None

    @property
    def size(self) -> int:
        return len(self.terms)


def design_block(design: Design, weights: Tensor, equation: str | None,
                 offset: Tensor | None = None) -> Block:
    x, means = center_design(design, weights)
    return Block(list(design.terms), equation, x, means, offset, design)


def scalar_block(term: str) -> Block:
    return Block([term], None, None)


def constant_block(n: int, equation: str | None, offset: Tensor | None = None) -> Block:
    """The constant-only version of an equation (comparison models)."""
    return Block(["Intercept"], equation, torch.ones((n, 1), dtype=torch.float64), None, offset)


class Model:
    """Blocks plus per-observation pieces: the objective and its coordinate maps."""

    def __init__(self, pieces: Callable[..., Any], blocks: Sequence[Block], weights: Tensor):
        self.blocks = list(blocks)
        self.objective = kernel_call(IndexObjective, pieces, [block.x for block in blocks],
                                     weights, [block.offset for block in blocks])
        self.starts = []
        position = 0
        for block in self.blocks:
            self.starts.append(position)
            position += block.size
        self.size = position

    @property
    def terms(self) -> list[str]:
        return [term for block in self.blocks for term in block.terms]

    @property
    def equations(self) -> list[str | None]:
        return [block.equation for block in self.blocks for _ in block.terms]

    def centers(self) -> list[tuple[int, Tensor | None]]:
        return [(start, block.means) for start, block in zip(self.starts, self.blocks,
                                                             strict=True)]

    def to_centered(self, theta: Tensor) -> Tensor:
        """Parameters of the centered designs from parameters of the original columns."""
        centered = theta.clone()
        for start, means in self.centers():
            if means is not None:
                centered[start] = theta[start] + means @ theta[start + 1:start + 1 + len(means)]
        return centered

    def uncenter(self, theta: Tensor, covariance: Tensor | None = None):
        return uncenter(theta, covariance, self.centers())

    def part(self, theta: Tensor, block: int) -> Tensor:
        start = self.starts[block]
        return theta[start:start + self.blocks[block].size]


def flattest_information(information: Tensor, design: Tensor | None | Sequence[Tensor | None],
                         weights: Weights) -> float:
    """Information along the least informative direction of one or several indices.

    ``information`` is the block ``X' diag(w c_i) X`` of minus the Hessian that
    belongs to the index ``X theta`` (``c_i`` the curvature of observation i
    in that index). Its smallest generalized eigenvalue relative to
    ``X' diag(w) X`` is the weighted average curvature along the least
    informative direction, in the units of the index and whatever the scale of
    the regressors; times N it is the information that direction receives from
    the whole sample, "in observations' worth". For a scalar parameter
    (``design`` None) the reference is ``sum(w)``. A list of designs measures
    several blocks jointly against their block-diagonal reference, which also
    sees a direction that is flat only in a combination of blocks (a ridge).

    A likelihood that keeps increasing as some observations' index runs to
    plus or minus infinity (separation, a dispersion that collapses to zero, a
    ridge towards a limiting distribution) drives this number to zero -
    gradient and curvature vanish together - while one informative observation
    in the support of a direction keeps it of order one. Returns ``inf`` when
    it cannot be computed.
    """
    total = float(weights.user.sum())
    blocks = list(design) if isinstance(design, (list, tuple)) else [design]
    if len(blocks) == 1 and blocks[0] is None:
        return float(information.reshape(-1)[0]) * weights.nobs / total
    size = sum(1 if block is None else block.shape[1] for block in blocks)
    reference = torch.zeros((size, size), dtype=torch.float64)
    position = 0
    for block in blocks:
        if block is None:
            reference[position, position] = total
            position += 1
        else:
            width = block.shape[1]
            reference[position:position + width, position:position + width] = \
                weighted_crossprod(block, weights.user)
            position += width
    factor, failed = torch.linalg.cholesky_ex(reference)
    if int(failed) or information.shape != reference.shape \
            or not bool(torch.isfinite(information).all()):
        return math.inf
    half = torch.linalg.solve_triangular(factor, information, upper=False)
    scaled = torch.linalg.solve_triangular(factor, half.T, upper=False)
    return float(torch.linalg.eigvalsh((scaled + scaled.T) / 2)[0]) * weights.nobs


def weight_scale(weights: Weights) -> float:
    """Factor that puts an iweighted / pweighted likelihood on the unweighted scale.

    The optimizer's convergence rules are stated for a log likelihood of
    ordinary size; importance and sampling weights of arbitrary magnitude
    multiply it by their mean. The iteration therefore runs on
    ``ll / mean(w)`` and the result is scaled back, so estimates do not depend
    on the unit of the weights.
    """
    if not weights.weighted or weights.frequency is not None:
        return 1.0
    mean = float(weights.user.mean())
    return 1.0 if 0.5 <= mean <= 2.0 else 1.0 / mean


def maximize(objective: Any, start: Tensor, *, what: str, scale: float = 1.0,
             max_iter: int = 200,
             callback: Callable[[int, Tensor, float], Any] | None = None,
             ) -> optimize.OptimResult:
    """Newton-Raphson that returns a non-converged run instead of raising."""
    function, value_function = objective, objective.value
    if scale != 1.0:
        def function(theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            value, gradient, hessian = objective(theta)
            return value * scale, gradient * scale, hessian * scale

        def value_function(theta: Tensor) -> Tensor:
            return objective.value(theta) * scale
    try:
        objective.check_precision(start)
        previous = objective.reject_precision_trials
        objective.reject_precision_trials = True
        try:
            result = optimize.maximize_newton(function, start, value_fn=value_function,
                                              max_iter=max_iter, callback=callback,
                                              raise_on_failure=False)
        finally:
            objective.reject_precision_trials = previous
        objective.check_precision(result.theta)
        if objective.precision_rejected:
            result.diagnostics["precision_rejected"] = True
    except KernelError as exc:
        if exc.code == "invalid_start":
            raise AnalysisError("invalid_start", f"The {what} likelihood is not finite at the "
                                "starting values; check the outcome's range and the scale of "
                                "the regressors.") from exc
        raise AnalysisError(exc.code,
                            f"The {what} likelihood could not be maximized: {exc}") from exc
    except (FloatingPointError, OverflowError, RuntimeError) as exc:
        raise AnalysisError("numerical_failure", f"The {what} likelihood could not be "
                            f"evaluated: {exc}") from exc
    if scale != 1.0:
        result.value = result.value / scale
        result.gradient = result.gradient / scale
        result.hessian = result.hessian / scale
    return result


def alpha_watch(position: int) -> Callable[[int, Tensor, float], None]:
    """Newton callback: stop once ``ln(alpha)`` (at ``position``) has collapsed."""
    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if float(theta[position]) < _LOG_ALPHA_STOP:
            raise Boundary(theta.clone(), "alpha")

    return watch


def run(model: Model, start: Tensor, *, what: str, weights: Weights,
        callback: Callable[[int, Tensor, float], Any] | None = None,
        ) -> tuple[optimize.OptimResult | None, Boundary | None]:
    """One Newton run: ``(result, None)`` or ``(None, boundary)`` when a watch stopped it."""
    try:
        return maximize(model.objective, start, what=what, scale=weight_scale(weights),
                        callback=callback), None
    except Boundary as stop:
        return None, stop


def count_sample(frame: ModelFrame, command: str, *, integer: bool = False,
                 ) -> tuple[Tensor, Weights, Tensor | None]:
    """Outcome, likelihood weights and offset of a count command.

    Negative outcomes are an error. Non-integer outcomes are a recorded note
    (Stata's rule for ``poisson``) unless ``integer`` is set: truncated and
    hurdle likelihoods condition on integer events and have no meaning for
    fractional outcomes.
    """
    check_pweights(frame)
    weights = likelihood_weights(frame)
    offset = linear_offset(frame) if frame.info.role("offset") is not None else None
    y = frame.numeric(frame.spec.outcome)
    count_outcome(frame, y, command, note=not integer)
    if integer and bool((y != y.round()).any()):
        raise AnalysisError("invalid_count_outcome", f"{command} needs integer counts; "
                            f"'{frame.spec.outcome}' contains non-integer values.")
    return y, weights, offset


def poisson_start(frame: ModelFrame, design: Design, y: Tensor, weights: Weights,
                  offset: Tensor | None, command: str) -> Any:
    """The Poisson fit that starts a count equation with zeros in its outcome.

    ``Pr(y = 0)`` tends to one as the mean goes to zero, so a regressor (or a
    combination of regressors) that singles out observations whose outcomes
    are all zero has a coefficient that diverges to minus infinity - in the
    Poisson model and equally in the zero-inflated and negative binomial
    models built on it. The Poisson iteration then fails; when the design is
    rank deficient among the observations with a positive outcome, that is
    the reason, and the failure is reported as ``separation_detected`` with
    the terms concerned.
    """
    from openecon.econometrics.glm.families import Log, Poisson
    from openecon.econometrics.glm.glm import estimate
    from openecon.engines.linalg import collinear_columns

    try:
        return estimate(frame, design, y, Poisson(), Log(), weights=weights, offset=offset,
                        command=f"{command} (Poisson starting values)")
    except AnalysisError as exc:
        if exc.code != "nonconvergence":
            raise
        x = design.x[y > 0]
        if design.intercept and x.shape[1] > 1:
            x = x.clone()
            x[:, 1:] -= x[:, 1:].mean(dim=0)
        omitted = kernel_call(collinear_columns, x, None)[1] if x.shape[0] else []
        if not omitted:
            raise
        names = ", ".join(design.terms[i] for i in omitted)
        raise AnalysisError(
            "separation_detected",
            f"{command}: the count equation has no finite estimates: {names} does not vary "
            "(given the other regressors) among the observations with a positive outcome, so "
            "the observations it singles out are all zeros, the likelihood keeps increasing as "
            "their fitted mean goes to zero and a coefficient diverges. Remove that regressor "
            "or merge the category it identifies with another one.") from exc


def count_total(mask: Tensor, weights: Weights) -> int:
    """Number of observations with ``mask`` as Stata counts them (frequencies add up)."""
    if weights.frequency is not None:
        return int(round(float(weights.frequency[mask].sum())))
    return int(mask.sum())


def covariance_of(frame: ModelFrame, model: Model, theta: Tensor, hessian: Tensor,
                  weights: Weights, rows: Callable[[], Tensor] | None = None,
                  ) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Covariance by ``spec.covariance`` and the estimates in reporting coordinates."""
    scores = rows if rows is not None else (lambda: model.objective.score_rows(theta))
    covariance, info = likelihood_covariance(frame, hessian, scores, weights)
    estimates, covariance = model.uncenter(theta, covariance)
    info["df_inference"] = None
    return estimates, covariance, info


def model_test(frame: ModelFrame, theta: Tensor, covariance: Tensor, slopes: Sequence[int],
               log_likelihood: float, null: float | None, equation: str) -> dict[str, Any]:
    """LR chi2 against the constant-only first equation, or the Wald chi2 of its slopes."""
    if frame.spec.covariance == "nonrobust" and null is not None:
        return lr_test(log_likelihood, null, len(slopes),
                       label=f"LR chi2 test that the slopes of the {equation} equation are zero")
    return wald_test(theta, covariance, slopes,
                     label=f"Wald chi2 test that the slopes of the {equation} equation are zero")


def boundary_test(log_likelihood: float, restricted: float, name: str,
                  against: str) -> dict[str, Any]:
    """LR test of a dispersion at its boundary value 0: ``chibar2(01)``, half the chi2(1) tail."""
    statistic = max(0.0, 2 * (log_likelihood - restricted))
    return {"statistic": statistic, "df": 1,
            "p_value": 0.5 * kernel_call(chi2_sf, statistic, 1) if statistic > 0 else 1.0,
            "distribution": "chibar2",
            "label": f"LR test of {name} = 0 against {against} "
                     "(chibar2(01): chi2(1) tail halved)"}


def exponentiated(log_value: float, std_error: float, alpha: float, name: str) -> dict[str, Any]:
    """A positive parameter estimated in logs: estimate, delta-method SE and exponentiated CI."""
    critical = critical_value(alpha)
    value = safe_exp(log_value)
    return {"parameter": name, "estimate": value, "std_error": value * std_error,
            "ci_low": safe_exp(log_value - critical * std_error),
            "ci_high": safe_exp(log_value + critical * std_error),
            "method": f"delta method; interval exponentiated from ln {name}"}


def pseudo_r_squared(log_likelihood: float, null: float | None) -> float | None:
    """McFadden's ``1 - ll / ll_0`` (None without a comparison model)."""
    if null is None or null == 0:
        return None
    return 1 - log_likelihood / null


def link_inverse(link: str, probability: float) -> float:
    """The index at which ``F(index) = probability`` (starting values)."""
    from openecon.engines.distributions import normal_ppf

    probability = min(max(probability, 1e-6), 1 - 1e-6)
    if link == "logit":
        return math.log(probability / (1 - probability))
    if link == "probit":
        return kernel_call(normal_ppf, probability)
    return math.log(-math.log1p(-probability))


def not_converged(command: str, message: Any, advice: str = "") -> AnalysisError:
    return AnalysisError("nonconvergence", f"{command} did not converge: {message} "
                         + (advice or "Check the scale of the regressors and whether every "
                                      "parameter is identified by the data."))
