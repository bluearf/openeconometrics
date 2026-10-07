"""Maximum-likelihood optimizers and MLE covariance helpers on float64 tensors.

Every likelihood estimator maximizes its objective here. Derivatives are analytic
and supplied by the caller; no autograd graph is ever built.

Newton-Raphson (``maximize_newton``) mirrors Stata's -ml- stepping with stricter
tolerances. With gradient g and Hessian H of the objective, each iteration solves

    (-H) d = g

by Cholesky on the diagonally equilibrated matrix D(-H)D, D = diag(|H_jj|^-1/2).
Equilibration makes the positive-definiteness test, the solve and the Marquardt
ridge invariant to the units of the parameters. Where -H is not positive definite
(a non-concave region) the step solves D(-H)D + r I instead, i.e. it adds the
multiple r of diag(|H|) in the original units, with r increased until the
factorization succeeds and the step is within a relative trust radius; a diagonal
entry too small to carry curvature information (a saturated likelihood) is floored
by the gradient so that the ridge step stays bounded. Steps are halved until the
objective rises sufficiently (Armijo), so a trial point with a non-finite
likelihood is simply a rejected step.

BFGS (``maximize_bfgs``) updates an inverse-Hessian approximation from gradient
differences and uses a Wolfe line search (bracketing by extrapolation, then
safeguarded cubic interpolation), which keeps every update positive definite. The
reported Hessian is never the BFGS approximation, and convergence is judged by the
same scaled gradient g'(-H)^-1 g as Newton-Raphson, computed from that Hessian;
when the BFGS iterate falls short of it, a few Newton steps finish the job.

Both line searches compare objective values only down to the resolution of the
objective. A log likelihood is a sum of n terms that often cancel, so its rounding
noise can exceed the gain of a late step; within a relative band of 1e-9 the slope
along the step, which is computed from the analytic gradient, decides instead.

Numerical derivatives use Ridders' method: central differences at successively
halved steps are extrapolated to step zero in a Neville tableau, and each
coordinate keeps the entry with the smallest error estimate.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from .contracts import KernelError

_EPS = torch.finfo(torch.float64).eps
_ARMIJO = 1e-4           # sufficient-increase fraction of the predicted linear gain
_CURVATURE = 0.9         # strong-Wolfe bound on the slope left after overshooting
_SHORTFALL = 0.25        # largest fraction of the initial slope a step may leave uphill
_NOISE = 1e-9            # relative objective change below which slopes decide, not values
_MAX_HALVINGS = 50
_MAX_EXPANSIONS = 20
_MAX_ZOOM = 40
_RIDGE_FLOOR = 1e-3      # smallest curvature left in a non-concave direction (scaled units)
_TRUST_RADIUS = 1e2      # largest relative Marquardt step, max_j |d_j| / max(1, |theta_j|)
_MAX_RIDGE_GROWTH = 120  # quadruplings of the ridge: 4^120 covers any float64 step length
_POLISH_ITERATIONS = 10  # Newton steps allowed after BFGS stops short of the scaled tolerance
_HISTORY = 256
_DIFFERENCE_LEVELS = 8
_DIFFERENCE_SHRINKS = 12
_DIFFERENCE_RTOL = 1e-10


@dataclass
class OptimResult:
    theta: Tensor
    value: float
    gradient: Tensor
    hessian: Tensor
    iterations: int
    converged: bool
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _NewtonPoint:
    theta: Tensor
    value: float
    gradient: Tensor
    hessian: Tensor
    step: Tensor
    finite: bool
    concave: bool
    gradient_max: float
    scaled_gradient: float
    relative_step: float
    step_max: float


@dataclass(slots=True)
class _LinePoint:
    alpha: float
    theta: Tensor
    value: float
    gradient: Tensor
    finite: bool
    gradient_max: float
    gradient_square: float
    slope: float
    relative_step: float
    step_max: float
    extra: float


def _vector(theta: Any, code: str) -> Tensor:
    # A Python list goes straight to float64: torch.as_tensor(list) would round it
    # through the float32 default dtype first.
    try:
        vector = (theta.detach() if isinstance(theta, Tensor)
                  else torch.as_tensor(theta, dtype=torch.float64))
    except (TypeError, ValueError, RuntimeError) as error:
        raise KernelError(code, "The parameter vector must be a non-empty 1-D tensor.") from error
    if vector.ndim != 1 or vector.numel() == 0 or vector.is_complex():
        raise KernelError(code, "The parameter vector must be a non-empty 1-D tensor.")
    vector = vector.to(torch.float64).clone()
    if not bool(torch.isfinite(vector).all()):
        raise KernelError(code, "The parameter vector must be finite.")
    return vector


def _scalar(value: Any, like: Tensor) -> Tensor:
    try:
        scalar = torch.as_tensor(value, dtype=torch.float64, device=like.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise KernelError("invalid_objective", "The objective value must be a scalar.") from error
    if scalar.numel() != 1:
        raise KernelError("invalid_objective", "The objective value must be a scalar.")
    return scalar.reshape(())


def _derivative(value: Any, like: Tensor, shape: tuple[int, ...], what: str) -> Tensor:
    """A derivative returned by the objective as a fresh float64 tensor of the given shape.

    Copying costs k or k^2 elements and protects the iteration from an objective
    that fills the same buffer on every call.
    """
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device=like.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise KernelError("invalid_objective", f"Expected a {what} tensor.") from error
    if tensor.shape != shape:
        raise KernelError(
            "invalid_objective", f"Expected a {what} tensor, got shape {tuple(tensor.shape)}.")
    return tensor.clone()


def _outputs(output: Any, count: int, what: str) -> tuple:
    if not isinstance(output, (tuple, list)) or len(output) < count:
        raise KernelError("invalid_objective", f"The objective must return {what}.")
    return tuple(output[:count])


def _check_options(max_iter: int, **tolerances: float) -> None:
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise KernelError("invalid_solver_options", "max_iter must be a positive integer.")
    for name, tolerance in tolerances.items():
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
                or not math.isfinite(tolerance) or tolerance <= 0:
            raise KernelError("invalid_solver_options", f"{name} must be positive and finite.")


def _stack(*scalars: Tensor) -> list[float]:
    """Synchronize several 0-dim diagnostics with the host in one transfer."""
    return torch.stack([scalar.to(torch.float64) for scalar in scalars]).tolist()


def _equilibrated_cholesky(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Cholesky factor of D A D with D = diag(|a_jj|^-1/2), and a 0-dim definiteness flag.

    A squared pivot of the unit-diagonal matrix is one minus the R-squared of that
    coordinate on the preceding ones, so pivots at or below k * eps mean the matrix
    is singular to working precision whatever the units of its rows and columns.
    """
    diagonal = matrix.diagonal().abs()
    scale = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal)).rsqrt()
    chol, info = torch.linalg.cholesky_ex(matrix * scale[:, None] * scale, check_errors=False)
    definite = (info == 0) & (chol.diagonal().min().square() > matrix.shape[0] * _EPS)
    return scale, chol, definite


def _equilibrated_solve(scale: Tensor, chol: Tensor, rhs: Tensor) -> Tensor:
    return scale * torch.cholesky_solve((scale * rhs)[:, None], chol)[:, 0]


def _newton_point(fn: Callable[[Tensor], Any], theta: Tensor) -> _NewtonPoint:
    """Evaluate the objective and its Newton step with a single scalar synchronization."""
    value, gradient, hessian = _outputs(fn(theta), 3, "(value, gradient, hessian)")
    k = theta.numel()
    value = _scalar(value, theta)
    gradient = _derivative(gradient, theta, (k,), "[k] gradient")
    hessian = _derivative(hessian, theta, (k, k), "[k, k] Hessian")
    hessian = (hessian + hessian.T) / 2
    scale, chol, definite = _equilibrated_cholesky(-hessian)
    step = _equilibrated_solve(scale, chol, gradient)
    finite = torch.isfinite(value) & torch.isfinite(gradient).all() & torch.isfinite(hessian).all()
    magnitude = step.abs()
    stats = _stack(
        value, finite, definite & torch.isfinite(step).all(), gradient.abs().max(),
        torch.dot(gradient, step), (magnitude / theta.abs().clamp_min(1.0)).max(), magnitude.max(),
    )
    return _NewtonPoint(
        theta=theta, value=stats[0], gradient=gradient, hessian=hessian, step=step,
        finite=bool(stats[1]), concave=bool(stats[1]) and bool(stats[2]), gradient_max=stats[3],
        scaled_gradient=stats[4], relative_step=stats[5], step_max=stats[6],
    )


def _marquardt_step(
    negative_hessian: Tensor, gradient: Tensor, theta: Tensor,
) -> tuple[Tensor, float, float, float]:
    """Modified Marquardt step d = D (D(-H)D + r I)^-1 D g for a -H that is not positive definite.

    D = diag(E_jj^-1/2) with E_jj = max(|H_jj|, |g_j| / (1e2 max(1, |theta_j|))), so r I
    in the equilibrated units is r * E in the original ones: r * diag(|H|) wherever
    the curvature is informative, and otherwise a ridge under which the pure ridge
    step r^-1 E^-1 g moves coordinate j by at most 1e2 max(1, |theta_j|) / r. Without
    the floor a saturated likelihood, whose H_jj can be 1e-300, would produce a step
    1e300 times too long for any halving to recover. The ridge starts at
    r = -m + max(-m, 1e-3), where m is the lowest eigenvalue of D(-H)D: the most
    negative curvature |m| is reflected to +|m| (or raised to the floor when -H is
    merely singular) instead of being left barely positive, which would produce an
    arbitrarily long step. r is then quadrupled until the Cholesky factorization
    succeeds and the step is within the trust radius
    max_j |d_j| / max(1, |theta_j|) <= 1e2. Returns d, g'd,
    max_j |d_j| / max(1, |theta_j|) and max_j |d_j|.
    """
    diagonal = torch.maximum(negative_hessian.diagonal().abs(),
                             gradient.abs() / (_TRUST_RADIUS * theta.abs().clamp_min(1.0)))
    scale = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal)).rsqrt()
    scaled = negative_hessian * scale[:, None] * scale
    try:
        lowest = float(torch.linalg.eigvalsh(scaled)[0])
    except RuntimeError:
        lowest = -1.0
    if not math.isfinite(lowest):
        lowest = -1.0
    ridge = max(-lowest, 0.0) + max(-lowest, _RIDGE_FLOOR)
    identity = torch.eye(gradient.numel(), dtype=gradient.dtype, device=gradient.device)
    for _ in range(_MAX_RIDGE_GROWTH):
        chol, info = torch.linalg.cholesky_ex(scaled + ridge * identity, check_errors=False)
        step = _equilibrated_solve(scale, chol, gradient)
        magnitude = step.abs()
        failed, finite, slope, relative, largest = _stack(
            info, torch.isfinite(step).all(), torch.dot(gradient, step),
            (magnitude / theta.abs().clamp_min(1.0)).max(), magnitude.max())
        if failed == 0 and finite and slope > 0 and relative <= _TRUST_RADIUS:
            return step, slope, relative, largest
        ridge *= 4.0
    raise KernelError("numerical_failure", "No finite ascent step exists in a non-concave region.")


def _history(values: list[float], value: float) -> None:
    values.append(value)
    if len(values) > _HISTORY:
        del values[0]


def maximize_newton(
    fn: Callable[[Tensor], Any],
    theta0: Tensor,
    *,
    max_iter: int = 200,
    gradient_tol: float = 1e-8,
    step_tol: float = 1e-10,
    scaled_gradient_tol: float = 1e-10,
    value_fn: Callable[[Tensor], Any] | None = None,
    callback: Callable[[int, Tensor, float], Any] | None = None,
    raise_on_failure: bool = True,
) -> OptimResult:
    """Maximize an objective by Newton-Raphson with analytic derivatives.

    ``fn(theta) -> (value, gradient, hessian)`` returns the objective (0-dim tensor or
    float), its [k] gradient g and its [k, k] Hessian H. ``value_fn(theta) -> value`` is
    an optional cheaper evaluation used only for rejected line-search trials.
    ``callback(iteration, theta, value)`` runs after every accepted step.

    Iteration. The step d solves (-H) d = g by Cholesky on D(-H)D, D = diag(|H_jj|^-1/2).
    If -H is not positive definite to working precision the iteration is non-concave
    (Stata's "not concave") and d solves (D(-H)D + r I) D^-1 d = D g, a modified
    Marquardt step that adds r * diag(|H|) (diagonal entries floored by
    |g_j| / (1e2 max(1, |theta_j|)) where the curvature has vanished) with r
    increased until Cholesky succeeds and max_j |d_j| / max(1, |theta_j|) <= 1e2.
    The step length t starts at 1 and is halved until

        f(theta + t d) >= f(theta) + 1e-4 * t * g'd - 8 eps max(1, |f(theta)|),

    so a non-finite value, gradient or Hessian at a trial point is a rejected trial.
    A likelihood that is a sum of large cancelling terms carries rounding noise above
    that allowance; a trial within 1e-9 max(1, |f|) of f(theta) is therefore also
    accepted when the slope along d has fallen to |g(theta + t d)'d| <= 0.9 g'd, the
    test that is equivalent to an increase for a quadratic and does not need f.
    The full step is evaluated with ``fn`` because its derivatives are needed as soon
    as it is accepted; ``value_fn`` takes over once a step has been halved or while
    the region is non-concave. A concave problem therefore costs one ``fn`` call per
    iteration and nothing else. No step doubling is attempted: the unit Newton step
    is already optimal near the maximum and the Marquardt ridge sizes the step
    elsewhere.

    Convergence is declared at theta only when ALL of the following hold:

      1. -H(theta) is positive definite (never in a non-concave region, as in Stata);
      2. g'(-H)^-1 g <= scaled_gradient_tol. This is Stata's nrtolerance() criterion
         (default 1e-5 there). It is the squared length of the remaining Newton step
         in standard-error units, hence invariant to reparameterization and scaling;
      3. max_j |g_j| <= gradient_tol * max(1, |f(theta)|);
      4. max_j |d_j| / max(1, |theta_j|) <= step_tol for the Newton step d at theta.

    Rules 3 and 4 depend on the units of the parameters and can sit below float64
    rounding noise (regressors scaled by 1e9, a standard error far above the
    coefficient). They are therefore replaced by a resolution rule when the last
    accepted step changed the objective by no more than 8 eps max(1, |f|) or was
    accepted on its slope alone: rules 1-2 plus
    max_j |d_j| / max(1, |theta_j|) <= sqrt(step_tol). A monotone likelihood
    (separation, a variance parameter running to a boundary) keeps a large relative
    step and is reported as non-convergence rather than as a flat "maximum".
    The iteration also stops as non-converged as soon as halving has reduced a step
    below step_tol without meeting these rules, since no further progress is possible,
    and as soon as a Marquardt step of relative length at most sqrt(step_tol) leaves
    the objective unchanged to rounding: the next step from that point would be the
    same, so an unidentified parameter (a zero row of the Hessian) or a flat
    non-concave region is reported without grinding through max_iter iterations.

    A non-finite start raises KernelError("invalid_start"). Failure to converge raises
    KernelError("nonconvergence") unless ``raise_on_failure`` is false, in which case
    the last iterate is returned with ``converged=False``. ``hessian`` in the result
    is the symmetrized Hessian of the objective at ``theta``.
    """
    _check_options(max_iter, gradient_tol=gradient_tol, step_tol=step_tol,
                   scaled_gradient_tol=scaled_gradient_tol)
    # no_grad rather than inference_mode: results stay ordinary tensors that callers
    # may modify in place outside this function.
    with torch.no_grad():
        point = _newton_point(fn, _vector(theta0, "invalid_start"))
        if not point.finite:
            raise KernelError(
                "invalid_start",
                "The objective, gradient or Hessian is not finite at the starting values.")
        evaluations = derivative_evaluations = 1
        iterations = backtracks = nonconcave = 0
        last_step = 0.0
        stagnated = stalled = flat = halved = converged = False
        history = [point.value]
        while True:
            magnitude = max(1.0, abs(point.value))
            if point.concave and point.scaled_gradient <= scaled_gradient_tol:
                if point.gradient_max <= gradient_tol * magnitude \
                        and point.relative_step <= step_tol:
                    converged, message = True, "converged: gradient, scaled gradient and step"
                    break
                if stagnated and point.relative_step <= math.sqrt(step_tol):
                    converged, message = True, "converged: objective at float64 resolution"
                    break
            if point.concave:
                status = (f"max |gradient| = {point.gradient_max:.3e}, relative Newton step = "
                          f"{point.relative_step:.3e}, concave")
            else:
                status = f"max |gradient| = {point.gradient_max:.3e}, not concave"
            if flat:
                message = ("The objective does not change along the Marquardt direction where "
                           "the Hessian is not negative definite: a parameter may be "
                           f"unidentified, or the likelihood has no maximum here ({status}).")
                break
            if stalled:
                message = ("The objective does not increase along the Newton direction: the "
                           "likelihood is flat or noisy here, or its derivatives are inconsistent "
                           f"({status}).")
                break
            if iterations >= max_iter:
                message = f"Newton-Raphson did not converge in {max_iter} iterations ({status})."
                break
            if point.concave:
                direction, slope = point.step, point.scaled_gradient
                relative, step_max = point.relative_step, point.step_max
            else:
                if point.gradient_max == 0:
                    message = ("The gradient is zero where the Hessian is not negative definite "
                               "(a saddle point or a flat region); no ascent step exists.")
                    break
                nonconcave += 1
                direction, slope, relative, step_max = _marquardt_step(
                    -point.hessian, point.gradient, point.theta)
            allowance = 8 * _EPS * magnitude
            floor = point.value - _NOISE * magnitude
            probe = value_fn is not None and (halved or not point.concave)
            fraction = 1.0
            accepted = None
            for trial in range(_MAX_HALVINGS):
                candidate = point.theta + fraction * direction
                target = point.value + _ARMIJO * fraction * slope - allowance
                worthwhile = resolved = True
                if probe or (trial and value_fn is not None):
                    evaluations += 1
                    worthwhile = float(value_fn(candidate)) >= floor
                if worthwhile:
                    evaluations += 1
                    derivative_evaluations += 1
                    new = _newton_point(fn, candidate)
                    resolved = new.finite and new.value >= target
                    # Below the resolution of the objective the slope along d decides.
                    if resolved or (new.finite and new.value >= floor and abs(float(
                            torch.dot(new.gradient, direction))) <= _CURVATURE * slope):
                        accepted = new
                        break
                fraction *= 0.5
                backtracks += 1
            if accepted is None:
                message = ("The line search found no finite improving step; check the analytic "
                           f"derivatives ({status}).")
                break
            halved = fraction < 1.0
            unchanged = abs(accepted.value - point.value) <= allowance
            stagnated = not resolved or unchanged
            # Only the rounding allowance accepted this step: no further progress is possible.
            stalled = halved and fraction * relative <= step_tol
            # A tiny Marquardt step that left the objective unchanged would be recomputed
            # identically at the new point.
            flat = not point.concave and unchanged and fraction * relative <= math.sqrt(step_tol)
            last_step = fraction * step_max
            point = accepted
            iterations += 1
            _history(history, point.value)
            if callback is not None:
                callback(iterations, point.theta, point.value)
        if not converged and raise_on_failure:
            raise KernelError("nonconvergence", message)
        return OptimResult(
            theta=point.theta, value=point.value, gradient=point.gradient, hessian=point.hessian,
            iterations=iterations, converged=converged,
            method="newton_cholesky_marquardt_backtracking",
            diagnostics={
                "gradient_max": point.gradient_max,
                "scaled_gradient": point.scaled_gradient if point.concave else None,
                "relative_step": point.relative_step if point.concave else None,
                "last_step_max": last_step,
                "backtracks": backtracks,
                "nonconcave_iterations": nonconcave,
                "function_evaluations": evaluations,
                "derivative_evaluations": derivative_evaluations,
                "concave": point.concave,
                "message": message,
                "value_history": history,
            },
        )


def _line_point(
    fn: Callable[[Tensor], Any], origin: Tensor, direction: Tensor | None, alpha: float,
    extra: Tensor | None = None,
) -> _LinePoint:
    """Evaluate value and gradient at origin + alpha * direction with one synchronization."""
    theta = origin if direction is None else origin + alpha * direction
    value, gradient = _outputs(fn(theta), 2, "(value, gradient)")
    value = _scalar(value, theta)
    gradient = _derivative(gradient, theta, theta.shape, "[k] gradient")
    zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
    move = (theta - origin).abs()
    stats = _stack(
        value, torch.isfinite(value) & torch.isfinite(gradient).all(), gradient.abs().max(),
        torch.dot(gradient, gradient),
        zero if direction is None else torch.dot(gradient, direction),
        (move / origin.abs().clamp_min(1.0)).max(), move.max(), zero if extra is None else extra,
    )
    return _LinePoint(
        alpha=alpha, theta=theta, value=stats[0], gradient=gradient, finite=bool(stats[1]),
        gradient_max=stats[2], gradient_square=stats[3], slope=stats[4], relative_step=stats[5],
        step_max=stats[6], extra=stats[7],
    )


def _cubic_peak(low: _LinePoint, high: _LinePoint) -> float | None:
    """Stationary point of the cubic matching value and slope at both ends (maximization)."""
    width = high.alpha - low.alpha
    d1 = -low.slope - high.slope + 3 * (low.value - high.value) / (low.alpha - high.alpha)
    discriminant = d1 * d1 - low.slope * high.slope
    if not discriminant >= 0:
        return None
    d2 = math.copysign(math.sqrt(discriminant), width)
    denominator = low.slope - high.slope + 2 * d2
    if denominator == 0:
        return None
    return high.alpha - width * (d2 - d1 - high.slope) / denominator


def _wolfe_search(
    fn: Callable[[Tensor], Any], origin: _LinePoint, direction: Tensor, alpha: float,
) -> tuple[_LinePoint | None, int, float]:
    """Wolfe line search for h(a) = f(theta + a d), maximized over a > 0.

    Accepts a with h(a) >= h(0) + 1e-4 a h'(0) - 8 eps max(1, |h(0)|) and
    -0.9 h'(0) <= h'(a) <= 0.25 h'(0), which implies the strong Wolfe conditions with
    c2 = 0.9. The upper bound is deliberately tighter: a unit step that still leaves
    more than a quarter of the slope stopped short because the BFGS matrix
    underestimates the inverse curvature along d, and accepting such steps corrects
    it only slowly when curvatures differ by orders of magnitude.

    Within 1e-9 max(1, |h(0)|) of h(0), where rounding noise in a likelihood can
    exceed the change in h, the increase condition is replaced by its slope form
    h'(a) >= -0.9 h'(0) (the approximate Wolfe conditions of Hager and Zhang, 2005),
    and two values that close are ordered by their slopes rather than by h.

    The step is extrapolated (cubic, between 1.5 and 8 times) until the maximum is
    bracketed, then the bracket is refined by safeguarded cubic interpolation
    (bisection next to non-finite trials). Returns the accepted point (None when no
    improving step exists), the number of evaluations, and h'(0).
    """
    used = 1
    current = _line_point(fn, origin.theta, direction, alpha, torch.dot(origin.gradient, direction))
    slope0 = current.extra
    if not slope0 > 0:
        return None, used, slope0
    allowance = 8 * _EPS * max(1.0, abs(origin.value))
    noise = _NOISE * max(1.0, abs(origin.value))

    def sufficient(point: _LinePoint) -> bool:
        if not point.finite:
            return False
        if point.value >= origin.value + _ARMIJO * point.alpha * slope0 - allowance:
            return True
        return point.value >= origin.value - noise and point.slope >= -_CURVATURE * slope0

    def wolfe(point: _LinePoint) -> bool:
        return -_CURVATURE * slope0 <= point.slope <= _SHORTFALL * slope0

    low = high = None
    previous = dataclasses.replace(origin, alpha=0.0, slope=slope0)
    for expansion in range(_MAX_EXPANSIONS):
        if expansion:
            used += 1
            current = _line_point(fn, origin.theta, direction, alpha)
        if not sufficient(current) or (expansion and current.value < previous.value - noise):
            low, high = previous, current
            break
        if wolfe(current):
            return current, used, slope0
        if current.slope <= 0:
            low, high = current, previous
            break
        peak = _cubic_peak(previous, current)
        previous = current
        alpha = 2.0 * alpha if peak is None else min(max(peak, 1.5 * alpha), 8.0 * alpha)
    if low is None:
        return previous, used, slope0
    for _ in range(_MAX_ZOOM):
        width = high.alpha - low.alpha
        if abs(width) <= 4 * _EPS * max(low.alpha, high.alpha):
            break
        alpha = _cubic_peak(low, high) if high.finite else None
        inner, outer = low.alpha + 0.1 * width, low.alpha + 0.9 * width
        if alpha is None or not min(inner, outer) <= alpha <= max(inner, outer):
            alpha = low.alpha + 0.5 * width
        used += 1
        current = _line_point(fn, origin.theta, direction, alpha)
        if not sufficient(current) or current.value < low.value - noise:
            high = current
            continue
        if wolfe(current):
            return current, used, slope0
        if current.slope * width <= 0:
            high = low
        low = current
    return (low if low.alpha > 0 else None), used, slope0


def maximize_bfgs(
    fn: Callable[[Tensor], Any],
    theta0: Tensor,
    *,
    max_iter: int = 1000,
    gradient_tol: float = 1e-8,
    step_tol: float = 1e-12,
    hessian_fn: Callable[[Tensor], Tensor] | None = None,
    raise_on_failure: bool = True,
    scaled_gradient_tol: float = 1e-10,
    initial_inverse_hessian: Tensor | None = None,
    callback: Callable[[int, Tensor, float], Any] | None = None,
) -> OptimResult:
    """Maximize an objective by BFGS with an analytic gradient.

    ``fn(theta) -> (value, gradient)`` (extra outputs are ignored). With B the
    approximation to (-H)^-1, each iteration moves along d = B g by a line search
    satisfying the strong Wolfe conditions (sufficient increase with c1 = 1e-4, slope
    reduced to within [-0.9, 0.25] of its initial value) and, for s = theta_new - theta
    and y = g - g_new, updates

        B <- (I - s y'/y's) B (I - y s'/y's) + s s'/y's.

    B starts as the identity with first trial step min(1, 1/||g||) and is rescaled
    to (y's / y'y) I before its first update; ``initial_inverse_hessian`` supplies a
    positive definite start instead (for example the inverse information of a
    simpler model). The update is skipped when y's <= sqrt(eps) * s'g (curvature not
    positive along the step) and B restarts at the identity (steepest ascent) when
    B g is not an ascent direction or the line search fails. BFGS is not invariant
    to the units of the parameters the way Newton-Raphson is: when they differ by
    many orders of magnitude, standardize or pass ``initial_inverse_hessian``.

    The iteration stops when max_j |g_j| <= gradient_tol * max(1, |f|) and
    g'B g <= scaled_gradient_tol, or when it stalls (relative step
    max_j |s_j| / max(1, |theta_j|) <= step_tol, or no improving step exists). The
    gradient rule alone depends on the units of the parameters: with a log likelihood
    of 1e6 it tolerates a gradient of 1e-2, which on a flat likelihood is a sizeable
    fraction of a standard error. Convergence is therefore judged, as for
    Newton-Raphson, by the Hessian at the stopping point: it must be negative definite
    and satisfy g'(-H)^-1 g <= scaled_gradient_tol (the remaining Newton step in
    standard-error units, squared). When the BFGS iterate misses that bound, up to 10
    Newton-Raphson steps with the true Hessian polish it (``maximize_newton`` with the
    same tolerances; each step costs one Hessian, ``hessian_fn`` or a numerical one),
    counted in ``iterations`` and ``diagnostics["polish_iterations"]``. A saddle point
    or a maximum with a singular Hessian is never reported as a maximum.

    The returned ``hessian`` is ``hessian_fn(theta)`` when given, otherwise
    ``numerical_hessian`` of the analytic gradient; never the BFGS approximation.
    ``callback(iteration, theta, value)`` runs after every accepted step. Failure
    raises KernelError("nonconvergence") unless ``raise_on_failure`` is false.
    """
    _check_options(max_iter, gradient_tol=gradient_tol, step_tol=step_tol,
                   scaled_gradient_tol=scaled_gradient_tol)
    with torch.no_grad():
        theta = _vector(theta0, "invalid_start")
        k = theta.numel()
        point = _line_point(fn, theta, None, 0.0)
        if not point.finite:
            raise KernelError(
                "invalid_start", "The objective or gradient is not finite at the starting values.")
        identity = torch.eye(k, dtype=theta.dtype, device=theta.device)
        inverse, fresh = identity, True
        if initial_inverse_hessian is not None:
            inverse = torch.as_tensor(initial_inverse_hessian, dtype=theta.dtype,
                                      device=theta.device)
            if inverse.shape != (k, k) or not bool(torch.isfinite(inverse).all()):
                raise KernelError("invalid_solver_options",
                                  "initial_inverse_hessian must be a finite [k, k] matrix.")
            inverse, fresh = (inverse + inverse.T) / 2, False
        evaluations = 1
        iterations = backtracks = restarts = skipped = polish = 0
        last_step = 0.0
        stalled = False
        history = [point.value]
        while True:
            if fresh:
                direction, approximate = point.gradient, point.gradient_square
            else:
                direction = inverse @ point.gradient
                approximate = float(torch.dot(point.gradient, direction))     # g'B g
            if point.gradient_max <= gradient_tol * max(1.0, abs(point.value)) \
                    and approximate <= scaled_gradient_tol:
                reason = "gradient"
                break
            if stalled:
                reason = "step"
                break
            if iterations >= max_iter:
                reason = "max_iter"
                break
            alpha = min(1.0, 1.0 / math.sqrt(point.gradient_square)) if fresh else 1.0
            found, used, slope = _wolfe_search(fn, point, direction, alpha)
            evaluations += used
            backtracks += used - 1
            if found is None:
                if fresh:
                    reason = "line_search"
                    break
                inverse, fresh = identity, True
                restarts += 1
                continue
            step = found.theta - point.theta
            change = point.gradient - found.gradient
            curvature = found.alpha * (slope - found.slope)        # y's from synchronized slopes
            if curvature > math.sqrt(_EPS) * found.alpha * slope:
                if fresh:
                    inverse, fresh = identity * (curvature / torch.dot(change, change)), False
                product = inverse @ change
                cross = torch.outer(product, step)
                weight = (1 + torch.dot(change, product) / curvature) / curvature
                inverse = inverse + weight * torch.outer(step, step) - (cross + cross.T) / curvature
            else:
                skipped += 1
            last_step = found.step_max
            point = found
            iterations += 1
            _history(history, point.value)
            if callback is not None:
                callback(iterations, point.theta, point.value)
            stalled = found.relative_step <= step_tol
        if reason == "max_iter" and raise_on_failure:
            raise KernelError(
                "nonconvergence", f"BFGS did not converge in {max_iter} iterations "
                f"(max |gradient| = {point.gradient_max:.3e}).")

        def true_hessian(at: Tensor) -> Tensor:
            if hessian_fn is not None:
                return _derivative(hessian_fn(at), at, (k, k), "[k, k] Hessian")
            return numerical_hessian(
                lambda trial: _outputs(fn(trial), 2, "(value, gradient)")[1], at)

        hessian_failed = False
        try:
            hessian = true_hessian(point.theta)
            hessian = (hessian + hessian.T) / 2
        except KernelError as error:
            if error.code != "numerical_failure":
                raise
            hessian_failed = True
            hessian = torch.full((k, k), math.nan, dtype=theta.dtype, device=theta.device)
        concave, scaled_gradient = _concavity(hessian, point.gradient)
        if reason != "max_iter" and concave and scaled_gradient > scaled_gradient_tol:
            # The BFGS matrix misjudged the curvature somewhere: finish with Newton-Raphson
            # on the true Hessian, reusing the one just computed at the starting point.
            origin, known = point.theta, hessian

            def composite(trial: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                value, gradient = _outputs(fn(trial), 2, "(value, gradient)")
                if trial is origin or torch.equal(trial, origin):
                    return value, gradient, known
                return value, gradient, true_hessian(trial)

            def report(iteration: int, trial: Tensor, value: float) -> None:
                if callback is not None:
                    callback(iterations + iteration, trial, value)

            try:
                polished = maximize_newton(
                    composite, origin, max_iter=_POLISH_ITERATIONS, gradient_tol=gradient_tol,
                    scaled_gradient_tol=scaled_gradient_tol,
                    value_fn=lambda trial: _outputs(fn(trial), 2, "(value, gradient)")[0],
                    callback=report, raise_on_failure=False)
            except KernelError as error:
                if error.code != "numerical_failure":
                    raise
                polished = None
            if polished is not None and polished.iterations > 0:
                polish = polished.iterations
                iterations += polish
                evaluations += polished.diagnostics["function_evaluations"] - 1
                backtracks += polished.diagnostics["backtracks"]
                last_step = polished.diagnostics["last_step_max"]
                for value in polished.diagnostics["value_history"][1:]:
                    _history(history, value)
                point = dataclasses.replace(
                    point, theta=polished.theta, value=polished.value,
                    gradient=polished.gradient, gradient_max=polished.diagnostics["gradient_max"])
                hessian = polished.hessian
                concave, scaled_gradient = _concavity(hessian, point.gradient)
        status = f"max |gradient| = {point.gradient_max:.3e}"
        if concave:
            status += f", scaled gradient = {scaled_gradient:.3e}"
        if reason == "max_iter":
            converged, message = False, f"BFGS did not converge in {max_iter} iterations."
        elif hessian_failed:
            converged = False
            message = ("The Hessian at the BFGS solution could not be computed: the gradient "
                       "is not finite around it.")
        elif not concave:
            converged = False
            message = ("BFGS stopped where the Hessian is not negative definite (not concave); "
                       "the point is not a maximum.")
        elif scaled_gradient <= scaled_gradient_tol:
            converged = True
            if polish:
                message = f"converged: scaled gradient after {polish} Newton polishing step(s)"
            elif reason == "gradient":
                message = "converged: gradient and scaled gradient"
            else:
                message = "converged: no further step at float64 resolution"
        else:
            converged = False
            message = "BFGS stopped before reaching a stationary point: " + (
                "the relative step fell below step_tol" if reason == "step" else
                "the line search found no improving step" if reason == "line_search" else
                "the BFGS matrix misjudged the curvature") + (
                f", and {polish} Newton polishing step(s) did not finish the job."
                if polish else ".")
        if not converged and raise_on_failure:
            raise KernelError("nonconvergence", f"{message} ({status})")
        return OptimResult(
            theta=point.theta, value=point.value, gradient=point.gradient, hessian=hessian,
            iterations=iterations, converged=converged, method="bfgs_strong_wolfe",
            diagnostics={
                "gradient_max": point.gradient_max,
                "scaled_gradient": scaled_gradient if concave else None,
                "last_step_max": last_step,
                "backtracks": backtracks,
                "nonconcave_iterations": skipped,
                "restarts": restarts,
                "polish_iterations": polish,
                "function_evaluations": evaluations,
                "concave": concave,
                "hessian_source": "hessian_fn" if hessian_fn is not None else "numerical_hessian",
                "message": message,
                "value_history": history,
            },
        )


def _concavity(hessian: Tensor, gradient: Tensor) -> tuple[bool, float]:
    """Whether -H is positive definite to working precision, and g'(-H)^-1 g."""
    scale, chol, definite = _equilibrated_cholesky(-hessian)
    newton = _equilibrated_solve(scale, chol, gradient)
    concave, scaled_gradient = _stack(
        definite & torch.isfinite(newton).all() & torch.isfinite(hessian).all(),
        torch.dot(gradient, newton))
    return bool(concave), scaled_gradient


def _ridders_jacobian(
    fn: Callable[[Tensor], Any], theta: Tensor, relative_step: float | None, levels: int,
    hessian: bool,
) -> tuple[Tensor, Tensor]:
    """[m, k] Jacobian of fn: R^k -> R^m by extrapolated central differences, and its
    [k] per-column error estimates (inf when ``levels`` is 1).

    Column j at level i is D_i = (F(theta + h_i e_j) - F(theta - h_i e_j)) / (2 h_i) with
    h_i = h_0 / 2^i. Because the error of D is a series in h^2, the Neville recursion
    A[i][q] = A[i][q-1] + (A[i][q-1] - A[i-1][q-1]) / (4^q - 1) removes one order per
    column q. Following Ridders (1982), each coordinate keeps the entry whose
    difference from its two parents is smallest and stops once that estimate meets
    the tolerance or the next order is worse (rounding has taken over). All
    coordinates advance level by level, so the host is synchronized once per level.
    """
    if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
        raise KernelError("invalid_solver_options", "levels must be a positive integer.")
    if relative_step is None:
        relative_step = _EPS ** (1 / 3) if levels == 1 else _EPS ** 0.2
    if not math.isfinite(relative_step) or relative_step <= 0:
        raise KernelError("invalid_solver_options", "relative_step must be positive and finite.")
    k = theta.numel()
    step = relative_step * theta.abs().clamp_min(1.0)

    def evaluate(point: Tensor) -> Tensor:
        try:
            output = torch.as_tensor(fn(point), dtype=theta.dtype, device=theta.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise KernelError(
                "invalid_objective", "The differentiated function must return a tensor.") from error
        # A copy: the two sides of a difference must not share an output buffer.
        return output.reshape(-1).clone()

    def difference(j: int, h: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        upper, lower = theta.clone(), theta.clone()
        upper[j] += h
        lower[j] -= h
        above, below = evaluate(upper), evaluate(lower)
        size = torch.maximum(above.abs(), below.abs()).max()
        # Divide by the representable distance between the two evaluation points.
        return (above - below) / (upper[j] - lower[j]), size, above + below

    # Find a usable first step for every coordinate; a rejected step is divided by 8
    # and retried. A non-finite difference means the step left the domain. Otherwise
    # the step must lie inside the scale on which F is smooth, which max(|theta_j|, 1)
    # overstates for, say, a coefficient on a regressor measured in millions:
    #   Hessian columns (which cannot vanish): D(h) and D(h/2) agree within 10%;
    #   a scalar F: the second differences S(h) = F(+h) + F(-h) - 2 F(0) scale as h^2,
    #   |S(h) - 4 S(h/2)| <= 0.2 |S(h)|. Slopes cannot be compared here because the
    #   derivative may be zero and a saturated likelihood is linear on both sides.
    # Both tests are skipped once the quantities compared are rounding noise.
    center = None if hessian or levels == 1 else evaluate(theta)
    coarse: list[Tensor | None] = [None] * k
    fine: list[Tensor | None] = [None] * k
    sizes: list[Tensor | None] = [None] * k
    pending = list(range(k))
    for attempt in range(_DIFFERENCE_SHRINKS):
        checks = []
        for j in pending:
            coarse[j], sizes[j], total = difference(j, step[j])
            usable = torch.isfinite(coarse[j]).all()
            if levels > 1:
                fine[j], _, inner = difference(j, step[j] / 2)
                usable = usable & torch.isfinite(fine[j]).all()
                noise = 1e4 * _EPS * sizes[j]
                if hessian:
                    gap = (coarse[j] - fine[j]).abs().max()
                    scale = torch.maximum(coarse[j].abs().max(), fine[j].abs().max())
                    usable = usable & ~(gap > torch.maximum(0.1 * scale, noise / step[j]))
                else:
                    second = (total - 2 * center).abs().max()
                    mismatch = (total - 4 * inner + 6 * center).abs().max()
                    usable = usable & ~((mismatch > 0.2 * second) & (second > noise))
            checks.append(usable)
        pending = [j for j, valid in zip(pending, torch.stack(checks).tolist()) if not valid]
        if not pending or attempt == _DIFFERENCE_SHRINKS - 1:
            # A coordinate still pending keeps its last differences, so its step must
            # stay the one they were computed with.
            break
        step[pending] = step[pending] / 8
    if len({entry.numel() for entry in coarse}) != 1:
        raise KernelError("invalid_objective", "The differentiated function changed its shape.")
    best = torch.stack(coarse, dim=1)
    rounding = 16 * _EPS * torch.stack(sizes)
    best_error = torch.full((k,), math.inf, dtype=theta.dtype, device=theta.device)
    active = torch.ones(k, dtype=torch.bool, device=theta.device)
    remaining = list(range(k))
    previous = [best]
    for level in range(1, levels):
        if not remaining:
            break
        step = step / 2
        if level == 1:
            current = torch.stack(fine, dim=1)
        else:
            current = previous[0].clone()
            for j in remaining:
                current[:, j] = difference(j, step[j])[0]
        row = [current]
        weight = 4.0
        for entry in previous:
            row.append(row[-1] + (row[-1] - entry) / (weight - 1.0))
            weight *= 4.0
        for order in range(1, len(row)):
            error = torch.maximum((row[order] - row[order - 1]).abs(),
                                  (row[order] - previous[order - 1]).abs()).amax(dim=0)
            better = active & (error < best_error)
            best = torch.where(better, row[order], best)
            best_error = torch.where(better, error, best_error)
        # Stop a coordinate when the next order is worse, the tolerance is met, or the
        # estimate has reached the rounding noise eps |F| / h, which doubles per level.
        growth = (row[-1] - previous[-1]).abs().amax(dim=0)
        floor = torch.maximum(_DIFFERENCE_RTOL * best.abs().amax(dim=0), rounding / step)
        done = ~(growth < 2 * best_error) | (best_error <= floor)
        active = active & ~done
        remaining = [j for j, flag in enumerate(active.tolist()) if flag]
        previous = row
    return best, best_error


def numerical_gradient(
    value_fn: Callable[[Tensor], Any], theta: Tensor, *, relative_step: float | None = None,
    levels: int = _DIFFERENCE_LEVELS,
) -> Tensor:
    """Gradient of a scalar function by extrapolated central differences.

    Coordinate j starts from (f(theta + h_j e_j) - f(theta - h_j e_j)) / (2 h_j) with
    h_j = relative_step * max(|theta_j|, 1), default relative_step = eps^(1/5), and
    extrapolates the sequence h_j, h_j/2, h_j/4, ... to zero (Ridders' method) with
    at most ``levels`` step sizes. A smooth function uses three: 6 k + 1 evaluations
    and an error near 1e-12 |f|. The first step is divided by 8 until f is finite at
    both points and its second difference scales as h^2, so a parameter whose natural
    scale is far below max(|theta_j|, 1) is still differentiated correctly.

    ``levels=1`` is the plain central difference with h_j = eps^(1/3) max(|theta_j|, 1):
    2 k evaluations, error of order eps^(2/3) |f|, no step adaptation. For tests,
    diagnostics and likelihood pieces without closed-form derivatives.
    """
    with torch.no_grad():
        point = _vector(theta, "invalid_parameters")
        jacobian, _ = _ridders_jacobian(value_fn, point, relative_step, levels, False)
        if jacobian.shape[0] != 1:
            raise KernelError("invalid_objective", "value_fn must return a scalar.")
        if not bool(torch.isfinite(jacobian).all()):
            raise KernelError("numerical_failure",
                              "The function is not finite around the evaluation point.")
        return jacobian[0]


def numerical_hessian(
    gradient_fn: Callable[[Tensor], Tensor], theta: Tensor, *, relative_step: float | None = None,
    levels: int = _DIFFERENCE_LEVELS,
) -> Tensor:
    """Hessian from an ANALYTIC gradient by symmetrized extrapolated central differences.

    Column j starts from (g(theta + h_j e_j) - g(theta - h_j e_j)) / (2 h_j) with
    h_j = relative_step * max(|theta_j|, 1), default relative_step = eps^(1/5), and is
    extrapolated over halved steps exactly as in ``numerical_gradient``. Entry (i, j)
    is estimated twice, by column j (dJ_i/dtheta_j) and by column i, and the result
    averages the two with weights inverse to the squared error estimate of their
    columns: a column whose gradient components are large carries a rounding floor
    eps |g| / h that would otherwise contaminate the small entries it shares with a
    well-resolved column. The matrix is exactly symmetric. A smooth problem uses
    three step sizes (3 k pairs of gradient calls) and is accurate to about 1e-10
    relative to the largest entry of the better column. The first step is divided
    by 8 until the gradient is finite at both points and two successive step sizes
    agree within 10%, which keeps the accuracy when parameters are scaled very
    differently. ``levels=1`` is the plain central difference with
    h_j = eps^(1/3) max(|theta_j|, 1) (k pairs, no adaptation, plain average).
    """
    with torch.no_grad():
        point = _vector(theta, "invalid_parameters")
        jacobian, error = _ridders_jacobian(gradient_fn, point, relative_step, levels, True)
        if jacobian.shape != (point.numel(), point.numel()):
            raise KernelError("invalid_objective", "gradient_fn must return a [k] tensor.")
        if not bool(torch.isfinite(jacobian).all()):
            raise KernelError("numerical_failure",
                              "The gradient is not finite around the evaluation point.")
        if levels == 1 or not bool(torch.isfinite(error).all()):
            return (jacobian + jacobian.T) / 2
        # Weights 1 / error^2, normalized so that an exact column weighs 1 / eps^2.
        relative = error / error.max().clamp_min(torch.finfo(torch.float64).tiny)
        weight = relative.clamp_min(_EPS).square().reciprocal()
        return ((jacobian * weight[None, :] + jacobian.T * weight[:, None])
                / (weight[None, :] + weight[:, None]))


def information_inverse(negative_hessian: Tensor) -> Tensor:
    """Inverse of the observed information I = -H, the MLE covariance (Stata's vce(oim)).

    Computed as D (D I D)^-1 D with D = diag(I_jj^-1/2) from the Cholesky factor of the
    unit-diagonal matrix D I D, then symmetrized. Raises
    KernelError("singular_information") unless I is finite and positive definite to
    working precision (same test as the Newton convergence rule).
    """
    with torch.no_grad():
        information = torch.as_tensor(negative_hessian, dtype=torch.float64)
        if information.ndim != 2 or information.shape[0] != information.shape[1] \
                or information.shape[0] == 0:
            raise KernelError("singular_information", "The information matrix must be square.")
        information = (information + information.T) / 2
        scale, chol, definite = _equilibrated_cholesky(information)
        if not bool(definite & torch.isfinite(information).all()):
            raise KernelError(
                "singular_information",
                "The information matrix is not positive definite; the covariance is undefined.")
        inverse = torch.cholesky_inverse(chol) * scale[:, None] * scale
        if not bool(torch.isfinite(inverse).all()):
            raise KernelError("singular_information",
                              "The inverse information exceeds the float64 range.")
        return (inverse + inverse.T) / 2


def opg(scores: Tensor) -> Tensor:
    """Outer product of gradients S'S for an [n, k] matrix of per-observation scores.

    Its inverse is the BHHH covariance (Stata's vce(opg)) and it is the "meat" of
    the robust sandwich (-H)^-1 S'S (-H)^-1 before any finite-sample correction.
    """
    with torch.no_grad():
        if not isinstance(scores, Tensor) or scores.ndim != 2:
            raise KernelError("invalid_scores", "Scores must be an [n, k] matrix.")
        scores = scores.to(torch.float64)
        return scores.T @ scores


def check_derivatives(fn: Callable[[Tensor], Any], theta: Tensor) -> dict[str, float | None]:
    """Compare analytic derivatives returned by ``fn`` with numerical ones.

    ``fn(theta) -> (value, gradient)`` or ``(value, gradient, hessian)``. The gradient
    is checked against ``numerical_gradient`` of the value and the Hessian against
    ``numerical_hessian`` of the analytic gradient. Reported for each:

        *_max_abs_error = max |analytic - numerical|
        *_max_rel_error = max |analytic - numerical| / max(max |analytic|, max |numerical|)

    plus ``hessian_asymmetry = max |H - H'|``. Hessian entries are None when ``fn``
    returns no Hessian. Evaluate away from a stationary point: where the gradient is
    numerically zero its relative error is only rounding noise. A developer and test
    utility; correct derivatives give relative errors near 1e-9 or below.
    """
    with torch.no_grad():
        point = _vector(theta, "invalid_parameters")
        output = fn(point)
        k = point.numel()
        gradient = _derivative(_outputs(output, 2, "(value, gradient)")[1], point, (k,),
                               "[k] gradient")

        def discrepancy(analytic: Tensor, numerical: Tensor) -> tuple[float, float]:
            error, size = _stack((analytic - numerical).abs().max(),
                                 torch.maximum(analytic.abs().max(), numerical.abs().max()))
            return error, error / size if size > 0 else 0.0

        gradient_abs, gradient_rel = discrepancy(
            gradient, numerical_gradient(lambda trial: fn(trial)[0], point))
        report: dict[str, float | None] = {
            "gradient_max_abs_error": gradient_abs, "gradient_max_rel_error": gradient_rel,
            "hessian_max_abs_error": None, "hessian_max_rel_error": None,
            "hessian_asymmetry": None,
        }
        if len(output) > 2:
            hessian = _derivative(output[2], point, (k, k), "[k, k] Hessian")
            hessian_abs, hessian_rel = discrepancy(
                (hessian + hessian.T) / 2, numerical_hessian(lambda trial: fn(trial)[1], point))
            report.update({
                "hessian_max_abs_error": hessian_abs, "hessian_max_rel_error": hessian_rel,
                "hessian_asymmetry": float((hessian - hessian.T).abs().max()),
            })
        return report
