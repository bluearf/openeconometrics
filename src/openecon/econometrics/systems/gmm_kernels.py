"""Tensor kernels of the general GMM estimator.

Moments
-------
Equation ``j = 1..M`` has a residual expression ``u_j(x_i; b)`` (parsed by the
``nl`` formula grammar, ``quantile.formula``) and instruments ``z_ij`` (``L_j``
columns, the constant first unless excluded). With weights ``w_i`` the sample
moments and their Jacobian are

    g(b) = sum_i w_i m_i(b),   m_i = (z_i1 u_i1, ..., z_iM u_iM)       [L = sum_j L_j]
    G(b) = dg/db' : block j = sum_i w_i z_ij (du_ij/db)'               [L, P]

``du/db`` is the analytic forward-mode Jacobian of the parsed expression (no
numerical differencing, no autograd); parameters shared between equations
are one global parameter.

Minimization
------------
For a positive definite weight matrix ``W = S^-1`` with ``S = C C'`` (Cholesky;
``C = I`` for the identity weight) the criterion is ``Q(b) = ||C^-1 g||^2``, a
nonlinear least-squares problem in the whitened moments ``g~ = C^-1 g`` with
Jacobian ``G~ = C^-1 G``. It is minimized by Gauss-Newton with
Levenberg-Marquardt damping: the step solves

    min_d ||g~ + G~ d||^2 + mu ||D d||^2,     D = diag(||G~_j||)

by Householder QR of the column-scaled ``[G~ D^-1; sqrt(mu) I]`` (Marquardt's
scaling, invariant to the units of the parameters). QR works with the
condition number of ``G~`` itself, not its square as the normal equations
``G'WG`` would, so regressors with large offsets (``x + 1e6``) still give
accurate steps. ``mu`` starts at 0 (pure Gauss-Newton, which solves a linear
model in one step), a step that does not lower ``Q`` is halved up to four
times and ``mu`` is then raised tenfold until ``Q`` falls. The iteration stops
when the Gauss-Newton predicted decrease ``||P g~||^2`` is at most
``tolerance^2 Q`` plus the rounding level of ``Q``, or when an accepted step
is small (``|d_j| <= tolerance (|b_j| + 1e-3)``) and ``Q`` has settled
(``Q_old - Q_new <= tolerance Q_new`` plus the rounding level).

The rounding level of ``Q``: a residual is computed with an error of about
``eps a_i``, ``a_i = |u_i| + |du_i/db'| |b|`` being the size of the terms that
cancel inside it (``y - b0 - b1 x`` with ``b0 = -4e5`` and ``b1 x = 4e5`` carries
rounding of order ``4e5 eps``, whatever the size of the residual). Treated as
independent errors, they perturb ``g~`` by ``nu`` with
``nu^2 = eps^2 tr(W S_a)``, ``S_a = sum_i w_i^2 a_i^2 z_i z_i'``, and ``Q`` by about
``2 sqrt(Q) nu + nu^2``; ten times that is the level below which differences of
``Q`` between two points are noise (step acceptance, "settled"). The predicted
decrease ``||P g~||^2`` is computed analytically, so its own noise is only of
order ``nu^2``: the Gauss-Newton test uses ``tolerance^2 Q + 10 nu^2``, which keeps
taking the (accurate) tiny steps that the iterated estimator needs. No inverse
of ``S`` or of ``G'WG`` is formed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from openecon.econometrics.quantile import formula as formulas
from openecon.engines.contracts import KernelError
from openecon.resources import plan_workspace, tensor_bytes

_EPS = torch.finfo(torch.float64).eps
_HALVINGS = 4
_MAX_DAMPING = 1e12
_PARAMETER_FLOOR = 1e-3
_NOISE = 10.0        # safety factor on the rounding level of Q


def gmm_workspace_plan(n, parameters, widths, equations, *, resident_bytes=0, budget_bytes=None,
                       formula_tape_elements=0):
    """Full joint instrument, residual-Jacobian and moment covariance footprint."""
    moments = sum(widths)
    return plan_workspace("combined nonlinear GMM", {
        "model_input_and_sample_indices": resident_bytes,
        "instruments_bases_and_qr_copies": tensor_bytes((n, moments), itemsize=40),
        "all_equation_parameter_jacobians": tensor_bytes((n, parameters, equations), itemsize=24),
        "moment_rows_and_weighted_copies": tensor_bytes((n, moments), itemsize=24),
        "residuals_and_formula_scratch": tensor_bytes((n, equations + 4), itemsize=32),
        # The formula evaluator retains values/partials for its whole tape.
        # Bound every node by a row vector plus each local parameter partial;
        # scalar/broadcast entries make this conservative, never a lower bound.
        "formula_tape_live_upper_bound": tensor_bytes((formula_tape_elements,)),
        "moment_covariances_weighting_and_factors": tensor_bytes((moments, moments), itemsize=80),
        "moment_parameter_jacobians_and_qr": tensor_bytes((moments + parameters, parameters), itemsize=48),
    }, budget_bytes=budget_bytes)


@dataclass
class MomentSystem:
    formulas: list[formulas.Formula]
    index: list[list[int]]           # global position of every formula parameter
    columns: dict[str, Tensor]       # data columns read by the formulas, [n] each
    instruments: list[Tensor]        # Z_j, [n, L_j]
    weights: Tensor | None
    n: int
    parameters: int
    resident_bytes: int = 0
    budget_bytes: int | None = None

    def __post_init__(self):
        tape_elements = max((self.n * len(formula.tape) * (1 + len(formula.parameters))
                             for formula in self.formulas if formula is not None), default=0)
        self.resource_plan = gmm_workspace_plan(self.n, self.parameters, self.widths, len(self.formulas),
                                                resident_bytes=self.resident_bytes, budget_bytes=self.budget_bytes,
                                                formula_tape_elements=tape_elements)

    @property
    def widths(self) -> list[int]:
        return [z.shape[1] for z in self.instruments]

    def residuals(self, theta: Tensor, jacobian: bool) -> tuple[Tensor, list[Tensor] | None]:
        values, slopes = [], []
        for formula, index in zip(self.formulas, self.index, strict=True):
            local = theta[index]
            value, local_jacobian = formulas.evaluate(formula, self.columns, local, self.n,
                                                      jacobian=jacobian)
            values.append(value)
            if jacobian:
                full = torch.zeros((self.n, self.parameters), dtype=torch.float64)
                full[:, index] = local_jacobian
                slopes.append(full)
        return torch.stack(values, dim=1), (slopes if jacobian else None)

    def rows(self, resid: Tensor) -> Tensor:
        """Moment rows ``m_i`` ``[n, L]`` (unweighted)."""
        return torch.cat([z * resid[:, j:j + 1] for j, z in enumerate(self.instruments)], dim=1)

    def moments(self, theta: Tensor, jacobian: bool = True
                ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Residuals ``[n, M]``, ``g`` ``[L]`` and ``G`` ``[L, P]`` (None without Jacobian).

        With ``jacobian`` the residual magnitudes ``|u_ij| + |du_ij/db'| |b|`` of the
        last evaluation are kept in ``self.magnitude`` (the rounding yardstick).
        """
        resid, slopes = self.residuals(theta, jacobian)
        w = self.weights
        parts, blocks = [], []
        for j, z in enumerate(self.instruments):
            weighted = z if w is None else z * w[:, None]
            parts.append(weighted.T @ resid[:, j])
            if jacobian:
                blocks.append(weighted.T @ slopes[j])
        g = torch.cat(parts)
        if jacobian:
            size = theta.abs()
            self.magnitude = resid.abs() + torch.stack([slope.abs() @ size for slope in slopes],
                                                       dim=1)
        return resid, g, (torch.cat(blocks, dim=0) if jacobian else None)

    def magnitude_meat(self) -> Tensor:
        """``S_a = sum_i w_i^2 a_i^2 z_i z_i'`` from the residual magnitudes ``a`` of the last
        Jacobian evaluation (the covariance of the rounding errors of ``g``, over eps^2)."""
        rows = self.rows(self.magnitude)
        if self.weights is not None:
            rows = rows * self.weights[:, None]
        return rows.T @ rows


class Weighting:
    """A positive definite weight matrix ``W``: ``S^-1`` through the Cholesky factor of ``S``,
    the identity (``s=None``), or ``W = L'L`` from an explicit whitener ``L``."""

    def __init__(self, s: Tensor | None, size: int, whitener: Tensor | None = None):
        self.size = size
        self.s = s
        self.factor = None
        self.whitener = whitener
        if s is not None:
            factor, info = torch.linalg.cholesky_ex(s)
            diagonal = s.diagonal()
            if int(info) != 0 or not bool(torch.isfinite(factor).all()) or bool(
                    (factor.diagonal().square() <= 8 * size * _EPS * diagonal).any()):
                raise KernelError("singular_weight_matrix", "The moment covariance S is "
                                  "singular, so the weight matrix S^-1 does not exist.")
            self.factor = factor

    def whiten(self, values: Tensor) -> Tensor:
        """``C^-1 values`` with ``S = C C'`` (``values`` [L] or [L, c]); ``W = C^-T C^-1``."""
        if self.whitener is not None:
            return self.whitener @ values
        if self.factor is None:
            return values
        column = values.ndim == 1
        solved = torch.linalg.solve_triangular(self.factor, values[:, None] if column else values,
                                               upper=False)
        return solved[:, 0] if column else solved

    def apply(self, values: Tensor) -> Tensor:
        if self.whitener is not None:
            return self.whitener.T @ (self.whitener @ values)
        if self.factor is None:
            return values
        column = values.ndim == 1
        solved = torch.cholesky_solve(values[:, None] if column else values, self.factor)
        return solved[:, 0] if column else solved

    def matrix(self) -> Tensor:
        return self.apply(torch.eye(self.size, dtype=torch.float64))


@dataclass
class StepResult:
    theta: Tensor
    criterion: float          # Q = g'Wg
    g: Tensor
    jacobian: Tensor          # G
    resid: Tensor             # [n, M]
    iterations: int
    evaluations: int
    history: list[float] = field(default_factory=list)


_RANK_TOL = 1e-12    # |R_jj| of the unit-column QR below which the GN problem is singular


def _direction(white_jac: Tensor, white_g: Tensor, damping: float) -> Tensor | None:
    """Damped Gauss-Newton step by QR; None when the (undamped) problem is singular."""
    scale = torch.linalg.vector_norm(white_jac, dim=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    matrix, target = white_jac / scale, -white_g
    if damping:
        size = matrix.shape[1]
        matrix = torch.cat([matrix, damping ** 0.5 * torch.eye(size, dtype=matrix.dtype)])
        target = torch.cat([target, torch.zeros(size, dtype=target.dtype)])
    if matrix.shape[0] < matrix.shape[1]:
        return None
    q, r = torch.linalg.qr(matrix, mode="reduced")
    if not bool(torch.isfinite(r).all()) or bool((r.diagonal().abs() <= _RANK_TOL).any()):
        return None
    step = torch.linalg.solve_triangular(r, (q.T @ target)[:, None], upper=True)[:, 0] / scale
    return step if bool(torch.isfinite(step).all()) else None


def _criterion(weighting: Weighting, g: Tensor) -> tuple[Tensor, float]:
    white = weighting.whiten(g)
    return white, float(white @ white)


def _floor(system: MomentSystem, weighting: Weighting, q: float) -> tuple[float, float]:
    """Rounding levels of ``Q`` and of the predicted decrease (see the module docstring)."""
    nu2 = _EPS ** 2 * max(float(weighting.apply(system.magnitude_meat()).diagonal().sum()), 0.0)
    return _NOISE * (2 * math.sqrt(max(q, 0.0) * nu2) + nu2), _NOISE * nu2


@torch.no_grad()
def minimize(system: MomentSystem, weighting: Weighting, start: Tensor, *,
             tolerance: float = 1e-8, max_iterations: int = 500) -> StepResult:
    """Minimize ``g(b)'W g(b)`` from ``start`` (see the module docstring)."""
    theta = start.clone()
    resid, g, jac = system.moments(theta)
    if not (bool(torch.isfinite(g).all()) and bool(torch.isfinite(jac).all())):
        raise KernelError("invalid_start", "The moment residuals or their derivatives are not "
                          "finite at the starting values; give starting values with "
                          "start={...} or {name=value} in the expressions.")
    white_g, q = _criterion(weighting, g)
    floor, resolution = _floor(system, weighting, q)
    evaluations, damping, history = 1, 0.0, [q]
    for iteration in range(1, max_iterations + 1):
        white_jac = weighting.whiten(jac)
        gradient = white_jac.T @ white_g
        accepted = None
        while accepted is None:
            step = _direction(white_jac, white_g, damping)
            if step is None:
                damping = max(10 * damping, 1e-6)
            else:
                if damping <= 1e-6:
                    predicted = float(-gradient @ step)
                    if predicted <= tolerance ** 2 * q + resolution:
                        return StepResult(theta, q, g, jac, resid, iteration - 1, evaluations,
                                          history)
                for halving in range(_HALVINGS + 1 if damping == 0.0 else 1):
                    trial = theta + step * 0.5 ** halving
                    _, trial_g, _ = system.moments(trial, jacobian=False)
                    evaluations += 1
                    if bool(torch.isfinite(trial_g).all()):
                        trial_q = _criterion(weighting, trial_g)[1]
                        if trial_q <= q + floor:
                            accepted = (trial, step * 0.5 ** halving, trial_q)
                            break
                if accepted is None:
                    damping = max(10 * damping, 1e-4)
            if accepted is None and damping > _MAX_DAMPING:
                raise KernelError("nonconvergence", "No step lowers the GMM criterion: the "
                                  "parameters are not identified near the current values or "
                                  "the starting values are poor. Give better starting values.")
        trial, step, trial_q = accepted
        small = bool((step.abs() <= tolerance * (trial.abs() + _PARAMETER_FLOOR)).all())
        settled = q - trial_q <= tolerance * trial_q + floor
        theta = trial
        resid, g, jac = system.moments(theta)
        evaluations += 1
        if not bool(torch.isfinite(jac).all()):
            raise KernelError("numerical_failure", "The derivatives of the moment residuals are "
                              "not finite at the current parameter values; give other "
                              "starting values.")
        white_g, q = _criterion(weighting, g)
        floor, resolution = _floor(system, weighting, q)
        history.append(q)
        del history[:-50]
        if small and settled and damping <= 1.0:
            return StepResult(theta, q, g, jac, resid, iteration, evaluations, history)
        damping = damping / 10 if damping > 1e-10 else 0.0
    raise KernelError("nonconvergence", f"The GMM criterion was not minimized in "
                      f"{max_iterations} iterations; give better starting values or raise "
                      "max_iterations.")


def unadjusted_s(system: MomentSystem, resid: Tensor, nobs: int,
                 sigma: Tensor | None = None) -> Tensor:
    """``S_jk = s_jk Z_j'diag(w)Z_k`` with ``s_jk = u_j'diag(w)u_k / N`` (or a given ``sigma``)."""
    w = system.weights
    if sigma is None:
        weighted = resid if w is None else resid * w[:, None]
        sigma = weighted.T @ resid / nobs
    rows = []
    for j, zj in enumerate(system.instruments):
        left = zj if w is None else zj * w[:, None]
        rows.append(torch.cat([float(sigma[j, k]) * (left.T @ zk)
                               for k, zk in enumerate(system.instruments)], dim=1))
    s = torch.cat(rows, dim=0)
    return (s + s.T) / 2


def orthonormal_instruments(blocks: Sequence[Tensor], weights: Tensor | None
                            ) -> tuple[list[Tensor], Tensor]:
    """Instrument bases ``Q_j = Z_j R_j^-1`` with ``sqrt(w) Q_j`` orthonormal, and the
    whitener ``blockdiag(R_j')`` of the identity weight on the original instruments.

    The two-step, iterated and every sandwich formula are invariant to a nonsingular
    change of basis ``Z_j -> Z_j A_j`` (``g -> A'g``, ``S -> A'SA``, ``W -> A^-1 W A^-T``), so
    the estimator runs on ``Q_j``, whose moment covariance is well conditioned even when an
    instrument has a large offset. Only the identity weight refers to the original
    columns: ``g_Z'g_Z = ||R' g_Q||^2``.
    """
    bases, factors = [], []
    for z in blocks:
        scaled = z if weights is None else z * weights.sqrt()[:, None]
        r = torch.linalg.qr(scaled, mode="r").R
        diagonal = r.diagonal().abs()
        if not bool(torch.isfinite(r).all()) or bool((diagonal <= _EPS * diagonal.max()).any()):
            raise KernelError("singular_instruments", "The instruments of a moment equation are "
                              "linearly dependent; remove the redundant instrument.")
        bases.append(torch.linalg.solve_triangular(r, z, upper=True, left=False))
        factors.append(r.T)
    return bases, torch.block_diag(*factors)


def identity_blocks(system: MomentSystem) -> Tensor:
    """``blockdiag (Z_j'diag(w)Z_j)``: winitial='unadjusted' with unit residual covariance."""
    m = len(system.instruments)
    return unadjusted_s(system, torch.zeros((system.n, m), dtype=torch.float64), 1,
                        sigma=torch.eye(m, dtype=torch.float64))


def relative_change(new: Tensor, old: Tensor) -> float:
    return float(((new - old).abs() / (old.abs() + 1)).max())


def finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)
