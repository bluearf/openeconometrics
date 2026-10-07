"""Native float64 binary-separation certificates with bounded constraint storage.

For signed rows A and their mean c, solve max c'd subject to Ad >= 0 and
|d| <= 1. A strictly positive feasible value witnesses complete or quasi
separation. For ANY nonnegative multipliers l on any subset of rows,
||c + A'l||_1 is an upper bound on the full problem. Solver convergence alone
is never a certificate: upper bounds are recomputed, witnesses replay every row.

The small box LP uses an infeasible-start primal-dual predictor/corrector Newton
method (Boyd/Vandenberghe, Convex Optimization, ch. 11). Constraint generation
retains at most max(8, 4k) rows; neither a full observation matrix nor n-by-n work
is allocated here. Ambiguous precision or exhausted budgets refuse the fit.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from .contracts import KernelError

OBJECTIVE_THRESHOLD = 1e-8
_EPS = torch.finfo(torch.float64).eps
_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass
class BoxLPResult:
    direction: Tensor
    multipliers: Tensor
    upper_bound: float
    iterations: int


def _failure(message: str) -> KernelError:
    return KernelError("separation_check_failed", message)


def _workspace(width: int, constraints: int) -> int:
    # Conservative live tensor allowance: rows, inequality matrix, weighted
    # matrix, Newton normalizations/factors and SVD face-candidate work. This is
    # an algorithm workspace estimate, not a process-wide allocator/RSS cap.
    return 8 * (3 * (constraints + 2 * width) * width + 12 * width * width
                + 24 * (constraints + 2 * width))


def _upper_bound(objective: Tensor, rows: Tensor, multipliers: Tensor) -> float:
    """Weak-duality bound, including conservative float64 reduction error."""
    product = rows.T @ multipliers
    residual = objective + product
    # All multiplier signs are checked before use. Positive normalization of
    # rows preserves their halfspaces. The bound remains valid off the LP path.
    absolute = rows.abs().T @ multipliers
    error = 64 * _EPS * (len(rows) + len(objective) + 1) * (objective.abs() + absolute).sum()
    value = residual.abs().sum() + error
    return float(value) if bool(torch.isfinite(value)) else math.inf


def _step(positive: Tensor, direction: Tensor) -> float:
    negative = direction < 0
    if not bool(negative.any()):
        return 1.0
    return min(1.0, float((-positive[negative] / direction[negative]).min()))


def solve_box_lp(objective: Tensor, rows: Tensor, *, max_iter: int = 100) -> BoxLPResult:
    """Solve a small relaxation; its returned upper bound is independently valid.

Inputs are CPU float64, rows have maximum absolute component at most one.
Singular signed designs are permitted: box faces make Newton normal equations
positive definite until the limiting boundary. A loss of usable precision fails.
"""
    if (not isinstance(objective, Tensor) or not isinstance(rows, Tensor)
            or objective.device.type != "cpu" or rows.device.type != "cpu"
            or objective.dtype != torch.float64 or rows.dtype != torch.float64
            or objective.ndim != 1 or rows.ndim != 2 or rows.shape[1] != len(objective)
            or not len(objective) or not bool(torch.isfinite(objective).all())
            or not bool(torch.isfinite(rows).all())
            or not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1):
        raise _failure("Invalid native separation-program input.")
    with torch.no_grad(), torch.device("cpu"):
        k, m = len(objective), len(rows)
        if _workspace(k, m) > _WORKSPACE_BYTES:
            raise _failure("The native separation LP exceeds its 128 MiB working-memory budget.")
        if not m:
            direction = objective.sign()
            return BoxLPResult(direction, torch.empty(0, dtype=torch.float64),
                               float(objective.abs().sum()) * (1 + 64 * _EPS * k), 0)
        identity = torch.eye(k, dtype=torch.float64)
        g = torch.cat((-rows, identity, -identity))
        h = torch.cat((torch.zeros(m, dtype=torch.float64), torch.ones(2*k, dtype=torch.float64)))
        q = -objective
        d = torch.zeros(k, dtype=torch.float64)
        s = torch.ones(len(g), dtype=torch.float64)
        z = torch.ones_like(s)
        best_bound = math.inf
        best_dual = z[:m].clone()
        for iteration in range(1, max_iter + 1):
            bound = _upper_bound(objective, rows, z[:m])
            if bound < best_bound:
                best_bound, best_dual = bound, z[:m].clone()
            rp, rd = g @ d + s - h, g.T @ z + q
            mu = float(torch.dot(s, z)) / len(s)
            if bound <= OBJECTIVE_THRESHOLD or (
                    float(rp.abs().max()) <= 2e-13
                    and float(rd.abs().max()) <= 2e-13 * max(1.0, float(objective.abs().max()))
                    and mu <= 2e-13):
                return BoxLPResult(d, best_dual, best_bound, iteration)
            ratio = z / s
            normal = g.T @ (ratio[:, None] * g)
            scale = normal.diagonal().sqrt()
            if not bool(torch.isfinite(normal).all()) or bool((scale <= 0).any()):
                break
            normalized = normal / scale[:, None] / scale[None, :]
            factor, info = torch.linalg.cholesky_ex((normalized + normalized.T) / 2)
            if int(info) != 0:
                break

            def newton(complement: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                rhs = -rd + g.T @ ((complement - z * rp) / s)
                dx = torch.cholesky_solve((rhs / scale)[:, None], factor).flatten() / scale
                ds = -rp - g @ dx
                dz = (-complement - z * ds) / s
                return dx, ds, dz

            _, affine_s, affine_z = newton(s * z)
            primal_step, dual_step = _step(s, affine_s), _step(z, affine_z)
            affine_mu = float(torch.dot(s + primal_step * affine_s,
                                        z + dual_step * affine_z)) / len(s)
            sigma = min(1.0, max(0.0, affine_mu / mu)) ** 3
            dx, ds, dz = newton(s * z + affine_s * affine_z - sigma * mu)
            if not all(bool(torch.isfinite(value).all()) for value in (dx, ds, dz)):
                break
            alpha_p, alpha_d = .995 * _step(s, ds), .995 * _step(z, dz)
            d = d + alpha_p * dx
            s, z = s + alpha_p * ds, z + alpha_d * dz
            if bool((s <= 0).any()) or bool((z <= 0).any()):
                break
        # A nonconverged relaxation can still provide a certified upper bound;
        # a witness is accepted downstream only after replaying the full source.
        if math.isfinite(best_bound) and bool(torch.isfinite(d).all()):
            return BoxLPResult(d, best_dual, best_bound, iteration)
        raise _failure("The native separation program exceeded float64 precision.")


def _face_direction(direction: Tensor, rows: Tensor) -> Tensor:
    """Remove tiny residuals on limiting LP faces, then reverify globally.

    This projection is only a candidate generator: it cannot certify a fit or
    separation by itself. Positive near-binding rows may also be selected.
    """
    if len(rows):
        margins = rows @ direction
        active = rows[margins.abs() <= 1e-8]
        if len(active):
            _, singular, right = torch.linalg.svd(active, full_matrices=False)
            limit = 64 * _EPS * max(active.shape) * float(singular[0])
            basis = right[singular > limit]
            direction = direction - basis.T @ (basis @ direction)
    magnitude = float(direction.abs().max())
    return direction / max(1.0, magnitude)


def certify_separation(replay: Callable[[], Iterable[Tensor]], objective: Tensor,
                       width: int, *, max_passes: int = 512) -> dict[str, Any]:
    """Return a no-separation dual certificate or raise on a verified witness.

    ``replay`` supplies all signed normalized design rows in bounded batches.
    The caller is responsible for source identity, count and design-rank checks.
    The retained objective threshold is 1e-8 in the standardized box coordinates.
    """
    if (not callable(replay) or not isinstance(width, int) or isinstance(width, bool)
            or width < 1 or not isinstance(objective, Tensor) or objective.shape != (width,)
            or objective.dtype != torch.float64 or objective.device.type != "cpu"
            or not bool(torch.isfinite(objective).all())
            or not isinstance(max_passes, int) or isinstance(max_passes, bool) or max_passes < 1):
        raise _failure("Invalid replayable separation-program input.")
    with torch.no_grad(), torch.device("cpu"):
        cuts: list[Tensor] = []
        capacity, peak, total_steps = max(8, 4 * width), 0, 0
        if _workspace(width, capacity) > _WORKSPACE_BYTES:
            raise _failure("The native separation LP exceeds its 128 MiB working-memory budget.")
        for iteration in range(1, max_passes + 1):
            matrix = torch.stack(cuts) if cuts else torch.empty((0, width), dtype=torch.float64)
            result = solve_box_lp(objective, matrix)
            total_steps += result.iterations
            # Do not trust an optimizer status or cached bound. Weak duality
            # gives a full-program certificate even when cuts were discarded.
            if (not isinstance(result.multipliers, Tensor)
                    or result.multipliers.device.type != "cpu"
                    or result.multipliers.dtype != torch.float64
                    or result.multipliers.shape != (len(matrix),)
                    or not bool(torch.isfinite(result.multipliers).all())
                    or bool((result.multipliers < 0).any())):
                raise _failure("Invalid native separation dual certificate.")
            upper = _upper_bound(objective, matrix, result.multipliers)
            if upper <= OBJECTIVE_THRESHOLD:
                return {
                    "separation_diagnostic": "native Torch primal-dual constraint generation with global replay",
                    "separation_lp_iterations": iteration,
                    "separation_newton_iterations": total_steps,
                    "separation_peak_constraints": peak,
                    "separation_constraint_capacity": capacity,
                    "separation_objective_threshold": OBJECTIVE_THRESHOLD,
                    "separation_dual_upper_bound": upper,
                    "separation_lp_workspace_budget_bytes": _WORKSPACE_BYTES,
                    "separation_lp_workspace_estimate_bytes": _workspace(width, peak),
                    "separation_feasibility_roundoff": "64 * eps * columns * max(1, sum(abs(row * direction)))",
                }
            if (not isinstance(result.direction, Tensor)
                    or result.direction.device.type != "cpu"
                    or result.direction.dtype != torch.float64
                    or result.direction.shape != (width,)
                    or not bool(torch.isfinite(result.direction).all())):
                raise _failure("Invalid native separation witness candidate.")
            direction = _face_direction(result.direction, matrix)
            value = float(torch.dot(objective, direction))
            objective_error = 64 * _EPS * width * float((objective.abs() * direction.abs()).sum())
            if not math.isfinite(value) or value - objective_error <= OBJECTIVE_THRESHOLD:
                raise _failure("The separation program could not certify either a bound or a witness.")
            worst_margin, worst_row, count = math.inf, None, 0
            for signed in replay():
                if (not isinstance(signed, Tensor) or signed.device.type != "cpu"
                        or signed.dtype != torch.float64 or signed.ndim != 2
                        or signed.shape[1] != width or not bool(torch.isfinite(signed).all())):
                    raise _failure("Invalid signed rows in the separation replay.")
                if not len(signed):
                    continue
                count += len(signed)
                margins = signed @ direction
                roundoff = 64 * _EPS * width * (signed.abs() @ direction.abs()).clamp_min(1)
                normalized_margin = (margins + roundoff) / signed.abs().amax(dim=1).clamp_min(1)
                index = int(normalized_margin.argmin())
                if float(normalized_margin[index]) < worst_margin:
                    worst_margin = float(normalized_margin[index])
                    worst_row = signed[index].clone()
            if not count:
                raise _failure("The separation replay yielded no observations.")
            if worst_margin >= 0:
                raise KernelError("separation_detected", "Complete or quasi-complete separation was detected; no finite unpenalized estimate exists.")
            assert worst_row is not None
            worst_row /= worst_row.abs().max().clamp_min(torch.finfo(torch.float64).tiny)
            if any(torch.equal(worst_row, cut) for cut in cuts):
                raise _failure("Separation constraints could not be resolved at float64 precision.")
            if len(cuts) == capacity:
                slack = matrix @ direction
                cuts = [cut for cut, margin in zip(cuts, slack, strict=True)
                        if abs(float(margin)) <= 1e-8]
                if len(cuts) == capacity:
                    raise _failure("The separation diagnostic exceeded its bounded active-constraint capacity.")
            cuts.append(worst_row)
            peak = max(peak, len(cuts))
        raise _failure(f"The native separation diagnostic did not converge in {max_passes} passes.")
