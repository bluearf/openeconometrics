"""Kinks of the EGARCH likelihood: smooth-piece Hessians and maxima that lie on a kink.

Nelson's news term ``g (|z_t| - E|z|)`` makes the EGARCH log likelihood L(theta) only
piecewise smooth: it has a kink on every surface ``z_t(theta) = 0`` (one per
observation), across which the gradient jumps. Two things follow.

1. A numerical Hessian whose difference stencil straddles a kink is meaningless (the
   jump is divided by the step). ``piece_hessian`` therefore differentiates the gradient
   of the smooth PIECE through the point: with the signs s_t = sign(z_t) frozen,
   ``L_s(theta) = F(theta, {s_t z_t})`` is smooth across the kinks and coincides with L
   wherever the signs are those of the point. The jumps have conditional mean zero
   (they multiply sum_k b^k (z_(t+k)^2 - 1) and its analogues, the scores of later
   variances), so the expectation of the piece Hessian is the information matrix and it
   plays the role of the observed information.

2. The maximum itself may lie ON a kink: both one-sided derivatives point towards
   ``z_t = 0`` for some observation, and no smooth optimizer can converge there (in
   simulated samples this happens in 10% to 40% of the EGARCH fits). ``polish``
   solves that case exactly with an active-set Newton method. For a set A of active
   kinks let the sign entries s_j, j in A, be continuous unknowns; the first-order
   conditions of a maximum on the kinks are

       grad L_s(theta) = 0,     z_j(theta) = 0  (j in A),     |s_j| <= 1,

   i.e. zero lies in the convex hull of the one-sided gradients, with s_j the weights.
   Each round solves the square system in (theta, s_A) by Newton's method (the Jacobian
   is [[H_s, C], [Z', 0]] with C = d grad / d s and Z = d z_A / d theta; it is reused
   while the iteration contracts), which gives a target point, and then takes the
   classical active-set step:

   - if the way to the target leaves the current piece (residuals of inactive
     observations change sign), the iterate stops at the FIRST kink it crosses and that
     observation becomes active; kinks are entered one at a time because the surfaces
     z_j = 0 are hyperplanes in the mean parameters, so that more of them than those
     parameters (or nearly parallel ones) cannot be active together;
   - when more than two residuals change sign at once the point is still far from the
     solution and the search simply continues on the piece of the target (unless the
     same observations cross back and forth between two adjacent pieces);
   - otherwise the target is accepted, and an active kink with |s_j| > 1 is released to
     the side of that sign. A kink is a ridge only if the likelihood falls as |z_j|
     grows (C_j'Z_j < 0); an active kink of the other type (a valley) is always
     released, to the opposite side.

   The loop ends when no residual crosses and nothing is released, which gives either
   an ordinary smooth maximum (A empty) or a maximum on the kinks in A.

Everything here works in the optimizer's scaled coordinates t, theta = a0 + A t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.engines import optimize
from openecon.engines.contracts import KernelError

_MAX_ROUNDS = 12           # changes of the active set
_MAX_NEWTON = 40           # Newton / chord iterations per active set
_MAX_JACOBIANS = 6         # Jacobian evaluations per active set
_MAX_NEW_KINKS = 2         # more sign changes than this: re-freeze the piece instead
_ZERO = 1e-10              # |z_j| of an active kink at convergence
_SCALED_GRADIENT = 1e-10   # g'(-H)^-1 g at convergence (the Newton and BFGS rule)
# Relative first step of every numerical Hessian of the family. The gradients of the
# GJR, power-ARCH and GED likelihoods are not smooth at a zero residual (1(e < 0) e,
# |e|^(phi-1), |e|^(s-1)): a difference stencil that straddles the zero of one residual
# gives a wrong second derivative for that observation, and with the engine's default
# first step (eps^0.2, about 7e-4 in the scaled parameters) this cost up to 6% of a
# standard error in samples of a few hundred observations. With 1e-6 the chance that any
# residual lies inside a stencil is negligible, and the analytic gradient is accurate
# enough for the extrapolated differences to keep about eight digits.
HESSIAN_STEP = 1e-6


@dataclass
class KinkSolution:
    t: Tensor                # scaled parameters
    signs: Tensor            # [T] sign pattern; the multipliers s_j at the active periods
    active: list[int]        # positions (time order) of the observations with z = 0
    hessian: Tensor          # Hessian of the piece L_s at t (scaled coordinates)
    iterations: int


def piece(residual: Tensor) -> Tensor:
    """The sign pattern of the smooth piece through a point: +1 where e_t >= 0, else -1."""
    return torch.where(residual < 0, -1.0, 1.0).to(torch.float64)


def flat_mask(like: ArchLikelihood, active: list[int] | None) -> Tensor | None:
    """GED only: the mask of the active kinks, whose density kernel is taken as zero.

    On the surface z_j = 0 the GED kernel |z_j / lam|^s of observation j is identically
    zero and (for s > 1) so is its gradient, so leaving it out changes neither the
    constrained maximum nor its first-order conditions. Its second derivative, however,
    is unbounded at z_j = 0 when s < 2: a Hessian that included it would be dominated by
    that one observation (and a finite-difference Hessian would not even be symmetric).
    """
    replay = getattr(like, "replay_flat_mask", None)
    if replay is not None:
        return replay(active)
    if like.layout.dist != "ged" or not active:
        return None
    mask = torch.zeros(like.n, dtype=torch.bool)
    mask[active] = True
    return mask


def piece_hessian(like: ArchLikelihood, offset: Tensor, transform: Tensor, t: Tensor,
                  signs: Tensor | None = None, active: list[int] | None = None) -> Tensor:
    """Hessian at ``t`` of the smooth piece of the likelihood (signs frozen at ``t``).

    Numerical derivative of the analytic gradient of ``L_s``; raises
    ``KernelError('numerical_failure')`` when the likelihood is undefined around ``t``.
    ``active`` lists the kinks held at z = 0 (see ``flat_mask``).
    """
    replay = getattr(like, "replay_piece_hessian", None)
    if replay is not None:
        return replay(offset, transform, t, signs, active)
    flat = flat_mask(like, active)
    if signs is None:
        base = like.evaluate(offset + transform @ t, derivatives=False)
        if base is None:
            raise KernelError("numerical_failure", "The likelihood is not defined at the point.")
        signs = piece(base.residual)
    undefined = torch.full_like(t, math.nan)

    def gradient(point: Tensor) -> Tensor:
        out = like.evaluate(offset + transform @ point, signs=signs, flat=flat)
        return undefined if out is None else transform.T @ out.gradient

    return optimize.numerical_hessian(gradient, t, relative_step=HESSIAN_STEP)


def _solve(like: ArchLikelihood, offset: Tensor, transform: Tensor, t: Tensor, signs: Tensor,
           active: list[int], known: Tensor | None = None,
           ) -> tuple[Tensor, Tensor, Tensor, list[float], int] | None:
    """Newton's method on ``grad L_s = 0, z_A = 0`` in (t, s_A); ``None`` on failure.

    Returns ``(t, signs, hessian, ridge, iterations)`` with the multipliers written into
    ``signs`` at the active positions, the piece Hessian at the solution and, per active
    kink, ``C_j'Z_j`` (negative when the kink is a ridge of the likelihood). ``known`` is
    the piece Hessian at the starting point when the caller already has it.
    """
    k, m = t.numel(), len(active)
    t, signs = t.clone(), signs.clone()
    start = t.clone()
    flat = flat_mask(like, active)

    def state(point: Tensor, pattern: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
        out = like.evaluate(offset + transform @ point, signs=pattern, active=active or None,
                            flat=flat)
        if out is None:
            return None
        gradient = transform.T @ out.gradient
        if not m:
            empty = torch.empty(0, dtype=torch.float64)
            return gradient, empty, torch.empty((0, k), dtype=torch.float64)
        z = out.residual[active] / out.variance[active].sqrt()
        return gradient, z, out.d_standardized @ transform

    def jacobian(point: Tensor, normals: Tensor) -> tuple[Tensor, Tensor] | None:
        """The KKT matrix and (-H)^-1 at ``point``; ``None`` when H is not negative definite."""
        try:
            hessian = known if known is not None and torch.equal(point, start) \
                else piece_hessian(like, offset, transform, point, signs, active)
            inverse = optimize.information_inverse(-hessian)
        except KernelError:
            return None
        matrix = torch.zeros((k + m, k + m), dtype=torch.float64)
        matrix[:k, :k] = hessian
        for column, position in enumerate(active):
            sides = []
            for value in (1.0, -1.0):
                pattern = signs.clone()
                pattern[position] = value
                side = state(point, pattern)
                if side is None:
                    return None
                sides.append(side[0])
            matrix[:k, k + column] = 0.5 * (sides[0] - sides[1])     # d grad / d s_j
        matrix[k:, :k] = normals
        return matrix, inverse

    system, previous, evaluations = None, math.inf, 0
    here = False                 # whether the Jacobian in ``system`` was taken at the current t
    for iteration in range(_MAX_NEWTON):
        current = state(t, signs)
        if current is None:
            return None
        gradient, z, normals = current
        if system is None:
            if evaluations >= _MAX_JACOBIANS:
                return None
            system = jacobian(t, normals)
            evaluations += 1
            here = True
            if system is None:
                return None
        matrix, inverse = system
        scaled = float(gradient @ inverse @ gradient)
        distance = float(z.abs().max()) if m else 0.0
        if not (math.isfinite(scaled) and math.isfinite(distance)):
            return None
        if scaled <= _SCALED_GRADIENT and distance <= _ZERO:
            hessian = matrix[:k, :k].clone()
            if not here:
                try:
                    hessian = piece_hessian(like, offset, transform, t, signs, active)
                    optimize.information_inverse(-hessian)
                except KernelError:
                    return None
            ridge = [float(matrix[:k, k + j] @ matrix[k + j, :k]) for j in range(m)]
            return t, signs, hessian, ridge, iteration
        measure = scaled + distance ** 2
        if measure > 0.25 * previous and iteration:
            # The reused Jacobian no longer contracts: take a fresh one at this point.
            system, previous = None, math.inf
            continue
        previous = measure
        try:
            step = torch.linalg.solve(matrix, -torch.cat([gradient, z]))
        except RuntimeError:
            return None
        if not bool(torch.isfinite(step).all()):
            return None
        t = t + step[:k]
        if m:
            signs[active] = signs[active] + step[k:]
        here = False
    return None


@torch.no_grad()
def polish(like: ArchLikelihood, offset: Tensor, transform: Tensor, start: Tensor,
           hessian: Tensor | None = None) -> KinkSolution | None:
    """Finish an EGARCH maximization near ``start`` with the active-set method (see module).

    Returns the solution (``active`` empty for an ordinary smooth maximum) or ``None``
    when no maximum is found: the Newton iteration fails, a piece Hessian is not negative
    definite, or the active set does not settle. ``hessian`` is the piece Hessian at
    ``start`` when it is already known.
    """
    replay = getattr(like, "replay_polish", None)
    if replay is not None:
        return replay(offset, transform, start, hessian)
    base = like.evaluate(offset + transform @ start, derivatives=False)
    if base is None:
        return None
    signs, t = piece(base.residual), start.clone()
    active: list[int] = []
    if like.layout.dist == "ged":
        # A line search of BFGS often stops exactly on a kink. With GED errors the piece
        # Hessian does not exist there (flat_mask): start with those kinks active.
        on_kink = base.residual.abs() <= _ZERO * base.variance.sqrt()
        active = on_kink.nonzero().flatten().tolist()
        if len(active) >= t.numel():
            return None
    refrozen: set[int] = set()         # observations that crossed at the last re-freezing
    iterations = 0
    for number in range(_MAX_ROUNDS):
        known = hessian if number == 0 and hessian is not None and not active \
            and bool(torch.isfinite(hessian).all()) else None
        solved = _solve(like, offset, transform, t, signs, active, known)
        if solved is None:
            return None
        target, target_signs, hessian, ridge, steps = solved
        iterations += steps
        out = like.evaluate(offset + transform @ target, signs=target_signs, derivatives=False)
        if out is None:
            return None
        crossed = out.residual * target_signs < 0.0
        if active:
            crossed[active] = False
        flipped = crossed.nonzero().flatten().tolist()
        if len(flipped) > _MAX_NEW_KINKS and not set(flipped) <= refrozen:
            # Far from the solution: continue on the piece of the new point. (When the same
            # observations cross again, the smooth maxima of two adjacent pieces lie in each
            # other's piece and re-freezing would cycle between them for ever; the kinks
            # are then entered one at a time like any others.)
            refrozen = set(flipped)
            t, fresh = target, piece(out.residual)
            if active:
                fresh[active] = target_signs[active]
            signs = fresh
            continue
        if flipped:
            # The step to the target leaves the current piece. Stop at the first kink it
            # crosses and make that kink active; the other observations are then still
            # on the side of their frozen signs.
            before = like.evaluate(offset + transform @ t, signs=signs, derivatives=False)
            if before is None:
                return None
            gap = before.residual[flipped] - out.residual[flipped]
            fraction = torch.where(gap != 0.0, before.residual[flipped] / gap,
                                   torch.ones_like(gap)).clamp(0.0, 1.0)
            index = int(fraction.argmin())
            share = float(fraction[index])
            t = t + share * (target - t)
            if active:
                signs = signs.clone()
                signs[active] += share * (target_signs[active] - signs[active])
            active = sorted({*active, flipped[index]})   # its multiplier starts at the sign
            if len(active) >= t.numel():
                return None
            continue
        t, signs = target, target_signs
        released = []
        for position, curvature in zip(active, ridge, strict=True):
            multiplier = float(signs[position])
            if curvature < 0.0 and abs(multiplier) <= 1.0:
                continue                             # a maximum on this kink
            side = math.copysign(1.0, multiplier)
            signs[position] = side if curvature < 0.0 else -side
            released.append(position)
        if not released:
            return KinkSolution(t, signs, active, hessian, iterations)
        active = [position for position in active if position not in released]
    return None
