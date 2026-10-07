"""Conventions shared by the discrete-choice family's analysis layer.

* **Weights** (Stata's semantics for likelihood estimators). Every likelihood,
  score and Hessian sum is ``sum_i w_i (.)`` with

  - ``fweight``: ``w_i`` the integer frequency, ``N = sum w_i``; results equal
    those of the duplicated-row data set (scores are passed unweighted with
    ``frequency=`` to ``core.ml_covariance``);
  - ``aweight``: ``w_i`` rescaled to sum to the number of rows, ``N`` = rows;
  - ``iweight``: ``w_i`` used as given (not rescaled), ``N`` = rows;
  - ``pweight``: as iweight for the point estimates and the log
    pseudolikelihood; they need a robust or cluster covariance (``robust`` is
    the convenience default), which is invariant to the scale of the weights.

* **Covariance.** ``core.ml_covariance``: ``nonrobust`` = inverse observed
  information, ``opg``, ``robust`` = ``N/(N-1)`` sandwich, ``cluster`` =
  ``G/(G-1)`` sandwich (two cluster columns: inclusion-exclusion with the
  smallest G). Coefficient tests are z tests.
* **Newton-Raphson.** ``engines.optimize.maximize_newton`` never declares
  convergence in a non-concave region or on a monotone likelihood; a failed
  run is returned (not raised) so that each estimator can say what happened
  (separation, a correlation at the boundary) in its own words.
* **Model test.** ``tests['model']`` is the likelihood-ratio chi2 against the
  estimator's null model under the conventional (``nonrobust``) covariance and
  the Wald chi2 of the same coefficients otherwise.
* **Implicit constants.** Ordered models (cutpoints), the conditional logit
  (group effects) and the variance equation of ``hetprobit`` have no constant
  of their own, yet a constant regressor is not identified in them. Their
  designs are therefore screened for collinearity together with a constant
  column, which is then removed again.
* **Centring.** In an equation with a constant (explicit, or the cutpoints of
  an ordered model) the index is ``a + x'b = (a + m'b) + (x - m)'b``: moving
  the regressors only moves the constant. The likelihood is therefore
  maximized on regressors centred at their (weighted) means ``m`` and the
  estimates are mapped back exactly, ``a = a_c - m'b`` with covariance
  ``J V J'`` for the Jacobian ``J`` of that linear map (every covariance
  estimator transforms this way, because scores and Hessian do). The reported
  model is the one specified; what changes is the arithmetic: the Hessian no
  longer loses ``(mean / spread)^2`` digits when a regressor has a large level
  (a year, a date, an income), and Newton-Raphson is as well conditioned as
  the data allow. Equations without a constant are not centred.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, kernel_call, lr_test, make_spec, ml_covariance, wald_test,
)
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec

# Fitted probabilities closer than this to 0 or 1 count as numerically degenerate.
EXTREME = 1e-6
# Early exit of a separated Newton run: checked every WATCH_EVERY iterations from
# iteration WATCH_FROM on, with a much stricter degeneracy threshold.
EXTREME_EARLY = 1e-10
WATCH_FROM, WATCH_EVERY = 20, 5


@dataclass
class Weights:
    user: Tensor               # weights of every likelihood sum (ones when unweighted)
    frequency: Tensor | None   # the integer frequencies under fweights
    nobs: int                  # N as Stata counts it
    weighted: bool

    def for_screen(self) -> Tensor | None:
        return self.user if self.weighted else None

    @property
    def scale(self) -> float:
        """Factor that puts an iweighted / pweighted likelihood on the unweighted scale.

        The convergence rules of the optimizer (scaled gradient, objective resolution)
        are stated for a log likelihood of ordinary size. Importance and sampling weights
        of arbitrary magnitude (1e-12 or 1e12 per row) multiply the objective by their
        mean, so the iteration runs on ``scale * ll`` with ``scale = 1 / mean(w)`` and
        the result is scaled back: estimates do not depend on the unit of the weights.
        Unweighted, frequency-weighted and analytic-weighted fits use 1.
        """
        if not self.weighted or self.frequency is not None:
            return 1.0
        mean = float(self.user.mean())
        return 1.0 if 0.5 <= mean <= 2.0 else 1.0 / mean


class Separated(Exception):
    """Raised from a Newton callback to stop a separated fit early."""

    def __init__(self, theta: Tensor):
        super().__init__("separated")
        self.theta = theta


def likelihood_weights(frame: ModelFrame) -> Weights:
    """Weights as the likelihood sums receive them and N as Stata counts it."""
    raw = frame.weights()
    if raw is None:
        return Weights(torch.ones(frame.n, dtype=torch.float64), None, frame.n, False)
    kind = frame.spec.weight_type
    if kind == "fweight":
        return Weights(raw, raw, int(round(float(raw.sum()))), True)
    if kind == "aweight":
        return Weights(raw * (frame.n / float(raw.sum())), None, frame.n, True)
    if bool((raw < 0).any()):
        raise AnalysisError("negative_weights", "Negative iweights are not supported by this "
                            "estimator: every weight must be positive.")
    return Weights(raw, None, frame.n, True)


def build_spec(estimator: str, **fields: Any) -> ModelSpec:
    """``core.make_spec`` for convenience functions: every rejection is an AnalysisError.

    The registry reports an invalid option, role, weight type or covariance as
    a validation ``ValueError``; a convenience function turns it into
    ``AnalysisError('invalid_spec', ...)`` with the registry's own message.
    """
    try:
        return make_spec(estimator, **fields)
    except AnalysisError:
        raise
    except ValueError as exc:
        errors = exc.errors() if hasattr(exc, "errors") else []
        message = "; ".join(str(error.get("msg", "")).removeprefix("Value error, ")
                            for error in errors) or str(exc)
        raise AnalysisError("invalid_spec", message) from exc


def resolve_covariance(covariance: str | None, cluster: Any, weight_type: str | None) -> str | None:
    """Convenience-function default: pweights imply the robust covariance."""
    if covariance is None and weight_type == "pweight" and not cluster:
        return "robust"
    return covariance


def check_pweights(frame: ModelFrame) -> None:
    if frame.spec.weight_type == "pweight" and frame.spec.covariance in {"nonrobust", "opg"}:
        raise AnalysisError(
            "unsupported_covariance",
            "pweights are sampling weights and need a robust covariance: choose "
            "covariance='robust' (Stata's vce(robust)) or give a cluster column.")


def require_observations(nobs: int, parameters: int) -> None:
    if nobs <= parameters:
        raise AnalysisError("insufficient_observations", f"Estimation needs more observations "
                            f"({nobs}) than parameters ({parameters}).")


def offset_column(frame: ModelFrame) -> Tensor | None:
    """The ``offset`` role of the current sample (coefficient fixed at one), or None."""
    names = frame.role("offset")
    return frame.numeric(names[0]) if names else None


def binary_outcome(frame: ModelFrame, name: str, command: str) -> Tensor:
    """A 0/1 outcome column of the current sample.

    Stata treats every nonzero value as a positive outcome; OpenEconometrics asks for
    an explicit 0/1 coding instead of recoding silently.
    """
    y = frame.numeric(name)
    if bool(((y != 0) & (y != 1)).any()):
        raise AnalysisError("invalid_binary_outcome", f"{command} needs '{name}' coded 0/1; "
                            "recode the positive outcome as 1 and the negative outcome as 0.")
    return y


def label_text(value: Any) -> str:
    """Category label as it appears in term and equation names (2.0 -> '2')."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.15g}" if isinstance(value, float) else str(value)


def implicit_constant_design(frame: ModelFrame, predictors: Sequence[str] | None,
                             weights: Tensor | None, *, prefix: str = "") -> Design:
    """Design of an equation without its own constant, screened as if it had one.

    The cutpoints of an ordered model and the scale normalization of a variance
    equation play the role of the constant: a constant regressor, or a set of
    regressors spanning it, is not identified. The columns are screened left
    to right together with a leading constant (Stata's rule), which is dropped
    again afterwards.
    """
    design = frame.design(predictors, intercept=True, prefix=prefix)
    design = frame.drop_collinear(design, weights)
    design = design.select(range(1, len(design.terms)))
    return Design(design.x, design.terms, design.categories, False)


def centre_regressors(design: Design, weights: Weights, *,
                      implicit_constant: bool = False) -> Tensor:
    """Centre the regressors of an equation that has a constant; return the centres.

    ``design.x`` is replaced by ``x - m`` with ``m`` the (weighted) column means; the
    constant column itself, and every column of an equation without an explicit or
    implicit constant, keeps ``m = 0``. See "Centring" in the module notes.
    """
    columns = design.x.shape[1]
    means = torch.zeros(columns, dtype=torch.float64)
    first = 1 if design.intercept else 0
    if (design.intercept or implicit_constant) and columns > first:
        block = design.x[:, first:]
        if weights.weighted:
            means[first:] = (block * weights.user[:, None]).sum(dim=0) / weights.user.sum()
        else:
            means[first:] = block.mean(dim=0)
        design.x = design.x - means
    return means


def constant_shift(means: Tensor) -> Tensor:
    """Jacobian of ``(a_c, b) -> (a, b)`` for one equation whose constant is column 0.

    ``a = a_c - m'b`` undoes ``centre_regressors``; the identity when nothing was centred.
    """
    block = torch.eye(len(means), dtype=torch.float64)
    if len(means) > 1:
        block[0, 1:] = -means[1:]
    return block


def reparameterize(theta: Tensor, covariance: Tensor, jacobian: Tensor) -> tuple[Tensor, Tensor]:
    """A linear change of parameters ``J theta`` and its covariance ``J V J'``."""
    return jacobian @ theta, jacobian @ covariance @ jacobian.T


def maximize(objective: Callable[[Tensor], Any], start: Tensor, *, what: str,
             max_iter: int = 200,
             callback: Callable[[int, Tensor, float], Any] | None = None,
             scale: float = 1.0) -> optimize.OptimResult:
    """Newton-Raphson that returns a non-converged run instead of raising.

    ``scale`` (see ``Weights.scale``) multiplies the objective during the
    iteration only; value, gradient and Hessian of the result, and the value
    passed to ``callback``, are those of the objective itself.
    """
    function, value_function, watch = objective, getattr(objective, "value", None), callback
    if scale != 1.0:
        raw_value = value_function

        def function(theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            value, gradient, hessian = objective(theta)
            return value * scale, gradient * scale, hessian * scale

        if raw_value is not None:
            def value_function(theta: Tensor) -> Tensor:
                return raw_value(theta) * scale

        if callback is not None:
            def watch(iteration: int, theta: Tensor, value: float) -> None:
                callback(iteration, theta, value / scale)
    try:
        result = optimize.maximize_newton(
            function, start, value_fn=value_function, max_iter=max_iter, callback=watch,
            raise_on_failure=False)
    except KernelError as exc:
        if exc.code == "invalid_start":
            raise AnalysisError("invalid_start", f"The {what} likelihood is not finite at the "
                                "starting values; check the outcome coding and the scale of "
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


def separation_watch(count: Callable[[Tensor, float], int]
                     ) -> Callable[[int, Tensor, float], None]:
    """Newton callback that stops a run whose likelihood is visibly monotone.

    ``count(theta, threshold)`` returns the number of perfectly predicted
    observations with numerically degenerate probabilities (zero unless every
    degenerate one is predicted correctly). A well-posed fit converges long
    before the first check.
    """
    def watch(iteration: int, theta: Tensor, value: float) -> None:
        if iteration >= WATCH_FROM and iteration % WATCH_EVERY == 0 \
                and count(theta, EXTREME_EARLY):
            raise Separated(theta.clone())

    return watch


def perfectly_predicted(probability: Tensor, threshold: float) -> int:
    """Observations whose observed outcome has fitted probability above 1 - threshold.

    Returns zero when some observation is instead predicted with probability
    below the threshold: a degenerate but wrong prediction is the signature of
    a diverging iteration, not of separation.
    """
    high = probability > 1 - threshold
    if not bool(high.any()) or bool((probability < threshold).any()):
        return 0
    return int(high.sum())


def raise_not_converged(command: str, count: int, message: Any, *, unit: str = "observation",
                        advice: str = "") -> None:
    """The most informative error for a likelihood iteration that did not converge."""
    if count:
        raise AnalysisError(
            "separation_detected",
            f"{command}: {count} {unit}(s) are predicted perfectly (fitted probabilities "
            "numerically 0 or 1) and the likelihood keeps increasing: complete or "
            "quasi-complete separation, so finite maximum likelihood estimates do not exist. "
            "Remove or combine the regressor (or outcome category) that separates the outcome.")
    raise AnalysisError("nonconvergence", f"{command} did not converge: {message} "
                        + (advice or "Check the scale of the regressors and whether every "
                                     "parameter is identified by the data."))


def likelihood_covariance(frame: ModelFrame, hessian: Tensor, rows: Callable[[], Tensor],
                          weights: Weights, **overrides: Any) -> tuple[Tensor, dict[str, Any]]:
    """``core.ml_covariance`` with the family's weight semantics.

    ``rows()`` returns the per-observation score rows before user weights; it
    is only called when the covariance needs them, so the conventional
    covariance never allocates an n-by-k score matrix. A singular information
    matrix is reported with advice instead of the kernel's code.
    """
    kind = overrides.get("kind") or frame.spec.covariance
    try:
        if kind == "nonrobust":
            covariance, info = ml_covariance(frame, hessian=hessian, nobs=weights.nobs, **overrides)
        else:
            scores = kernel_call(rows)
            if weights.frequency is not None:
                covariance, info = ml_covariance(
                    frame, hessian=hessian, scores=scores, nobs=weights.nobs,
                    frequency=weights.frequency, **overrides)
            elif weights.weighted and kind == "opg":
                # The OPG estimates the information of sum_i w_i l_i, which is linear
                # in w_i: (sum w s s')^-1, the same convention as the glm family.
                covariance, info = ml_covariance(
                    frame, hessian=hessian, scores=scores, nobs=weights.nobs,
                    frequency=weights.user, **overrides)
            else:
                if weights.weighted:
                    scores = scores * weights.user[:, None]
                covariance, info = ml_covariance(frame, hessian=hessian, scores=scores,
                                                 nobs=weights.nobs, **overrides)
    except AnalysisError as exc:
        if exc.code != "singular_information":
            raise
        raise AnalysisError(
            "singular_information",
            "The information matrix is not positive definite at the estimates, so standard "
            "errors are undefined. A parameter is not identified: check for regressors that "
            "predict the outcome perfectly or that vary in very few observations.") from exc
    info["df_inference"] = None
    return covariance, info


def model_test(frame: ModelFrame, theta: Tensor, covariance: Tensor, indices: Sequence[int],
               log_likelihood: float, null_log_likelihood: float | None, *,
               null_model: str, what: str = "the slopes") -> dict[str, Any]:
    """LR chi2 against the null model (conventional covariance) or Wald chi2."""
    if frame.spec.covariance == "nonrobust" and null_log_likelihood is not None:
        return lr_test(log_likelihood, null_log_likelihood, len(indices),
                       label=f"LR chi2 test against {null_model}")
    return wald_test(theta, covariance, indices, label=f"Wald chi2 test of {what}")


def pseudo_r_squared(log_likelihood: float, null_log_likelihood: float | None) -> float | None:
    """McFadden's ``1 - ll / ll_0`` (None when the null likelihood is zero or absent)."""
    if null_log_likelihood is None or null_log_likelihood == 0:
        return None
    return 1 - log_likelihood / null_log_likelihood


def optimizer_record(result: optimize.OptimResult) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {"method": result.method, "iterations": result.iterations,
            "converged": result.converged, "gradient_max": diagnostics.get("gradient_max"),
            "backtracks": diagnostics.get("backtracks"),
            "nonconcave_iterations": diagnostics.get("nonconcave_iterations"),
            "message": diagnostics.get("message")}


def json_label(value: Any) -> str | int | float | bool:
    """A category label as a JSON scalar (NumPy scalars unwrapped, 2.0 kept as 2)."""
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and value == value and abs(value) != float("inf"):
        return int(value) if value == int(value) and abs(value) < 2 ** 53 else value
    raise AnalysisError("unsupported_category", "Outcome categories must be finite numbers, "
                        "strings or booleans.")


def probit_fit(x: Tensor, y: Tensor, weights: Tensor, *, command: str, what: str,
               scale: float = 1.0):
    """Ordinary probit by Newton-Raphson: starting values and comparison likelihoods.

    Returns ``(objective, result)`` of ``kernels.HetprobitObjective`` without
    variance regressors. Separation is reported in the calling command's name.
    """
    from openecon.econometrics.discrete.kernels import HetprobitObjective

    empty = torch.empty((x.shape[0], 0), dtype=torch.float64)
    objective = HetprobitObjective(x, empty, y, weights)

    def separated(theta: Tensor, threshold: float) -> int:
        return perfectly_predicted(objective.outcome_probabilities(theta), threshold)

    start = torch.zeros(x.shape[1], dtype=torch.float64)
    try:
        result = maximize(objective, start, what=f"{command} ({what})",
                          callback=separation_watch(separated), scale=scale)
    except Separated as stop:
        raise_not_converged(f"{command} ({what})", separated(stop.theta, EXTREME_EARLY),
                            "the likelihood is monotone.")
    if not result.converged:
        raise_not_converged(f"{command} ({what})", separated(result.theta, EXTREME),
                            result.diagnostics.get("message"))
    return objective, result
