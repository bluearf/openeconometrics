"""Tensor kernels of the instrumental-variables family.

Notation. ``y`` [n] is the outcome, ``X1`` [n, K1] the included exogenous
regressors (constant first when present), ``X2`` [n, q] the endogenous
regressors, ``Z2`` [n, L2] the excluded instruments and ``w`` optional weights
(the inner product is ``<a, b> = sum_i w_i a_i b_i`` throughout). ``X = [X1 X2]``
has K = K1 + q columns and ``Z = [X1 Z2]`` has L = K1 + L2. ``P_A`` is the
projection on the columns of A and ``M_A = I - P_A``.

Two QR factorizations carry every estimator and diagnostic (``project``):

    first    [X2 y] on Z     ->  Xhat2 = P_Z X2,  E = M_Z X2,  e_y = M_Z y
    partial  [X2 y Z2] on X1 ->  X2p = M_1 X2,  y_p = M_1 y,  Z2p = M_1 Z2

No projection matrix is formed. With ``G = X2p - E = (P_Z - P_1) X2`` (the part
of the first-stage fit that the excluded instruments add), Frisch-Waugh-Lovell
gives the k-class estimator ``b = {X'(I - k M_Z)X}^-1 X'(I - k M_Z) y`` as

    S_k  = G'WG - (k - 1) E'WE                       [q, q]
    b2   = S_k^-1 {G'W y_p - (k - 1) E'W e_y}        endogenous coefficients
    b1   = c_y - C2 b2                               exogenous coefficients

where ``C2`` and ``c_y`` are the coefficients of X2 and y on X1, and

    {X'(I - k M_Z)X}^-1 = [ (X1'WX1)^-1 + C2 S_k^-1 C2'   -C2 S_k^-1 ]
                          [ -S_k^-1 C2'                    S_k^-1    ].

2SLS is k = 1, solved by a QR of G (``S_1 = G'WG`` is never inverted); LIML
uses ``k = kappa``, the smallest eigenvalue of ``(Y'M_Z Y)^-1 Y'M_1 Y`` with
``Y = [y X2]``, from a Cholesky factor of ``Y'M_Z Y`` and a symmetric
eigenproblem. The estimating equations are ``Xk'W(y - Xb) = 0`` with
``Xk = [X1, Xhat2 - (k - 1) E] = (I - k M_Z) X``; sandwich covariances use the
rows of ``Xk`` as score regressors.

Linear GMM minimizes ``g(b)'S^-1 g(b)``, ``g(b) = Z'W(y - Xb)``:

    b = (A'S^-1 A)^-1 A'S^-1 Z'Wy,   A = Z'WX,   J = g(b)'S^-1 g(b),

with score regressors ``Xg = Z S^-1 A`` (so that ``b = (Xg'WX)^-1 Xg'Wy``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.linalg import (
    LeastSquares, cholesky_inverse, cholesky_solve, least_squares, quadratic_form, symmetrize,
    weighted_crossprod,
)


def solve(x: Tensor, y: Tensor, weights: Tensor | None, what: str) -> LeastSquares:
    """QR least squares on a block that already passed the collinearity screen."""
    fit = least_squares(x, y, weights, drop_collinear=True, tol=0.0)
    if fit.omitted:
        raise KernelError("singular_design", f"The {what} are exactly rank deficient after the "
                          "collinearity screen; rescale or drop the affected columns.")
    return fit


@dataclass
class Projections:
    """The first-stage and partialling-out fits (see the module docstring)."""

    first: LeastSquares        # [X2 y] on Z = [X1 Z2]
    partial: LeastSquares      # [X2 y Z2] on X1
    q: int                     # endogenous regressors
    k1: int                    # included exogenous regressors
    l2: int                    # excluded instruments

    @property
    def e(self) -> Tensor:                 # M_Z X2
        return self.first.resid[:, :self.q]

    @property
    def e_y(self) -> Tensor:               # M_Z y
        return self.first.resid[:, self.q]

    @property
    def xhat2(self) -> Tensor:             # P_Z X2
        return self.first.fitted[:, :self.q]

    @property
    def x2p(self) -> Tensor:               # M_1 X2
        return self.partial.resid[:, :self.q]

    @property
    def y_p(self) -> Tensor:               # M_1 y
        return self.partial.resid[:, self.q]

    @property
    def z2p(self) -> Tensor:               # M_1 Z2
        return self.partial.resid[:, self.q + 1:]


@torch.no_grad()
def project(y: Tensor, x1: Tensor, x2: Tensor, z2: Tensor,
            weights: Tensor | None = None) -> Projections:
    """Regress [X2 y] on Z = [X1 Z2] and [X2 y Z2] on X1 (two Householder QRs)."""
    n, q = x2.shape
    if q < 1 or z2.shape[1] < q:
        raise KernelError("underidentified", "The order condition fails: at least as many "
                          "excluded instruments as endogenous regressors are needed.")
    if n <= x1.shape[1] + z2.shape[1]:
        raise KernelError("insufficient_observations", "The first stage needs more observations "
                          "than instruments (exogenous regressors plus excluded instruments).")
    targets = torch.cat([x2, y[:, None]], dim=1)
    first = solve(torch.cat([x1, z2], dim=1), targets, weights, "instruments [X1 Z2]")
    partial = solve(x1, torch.cat([targets, z2], dim=1), weights, "exogenous regressors")
    return Projections(first, partial, q, x1.shape[1], z2.shape[1])


@dataclass
class KClass:
    beta: Tensor               # [K] in the order [X1, X2]
    bread: Tensor              # {X'(I - k M_Z)X}^-1, [K, K]
    resid: Tensor              # y - X beta
    rss: float                 # weighted residual sum of squares
    kappa: float
    g: Tensor                  # (P_Z - P_1) X2, [n, q]
    inner: Tensor              # S_k^-1, [q, q]
    condition_number: float    # of the unit-norm scaled G (2SLS) or S_k (k-class)


@torch.no_grad()
def k_class(p: Projections, weights: Tensor | None = None, kappa: float = 1.0) -> KClass:
    """k-class estimator from the two projections; ``kappa = 1`` is 2SLS (QR of G)."""
    q = p.q
    g = p.x2p - p.e
    failure = KernelError(
        "underidentified", "The rank condition fails: after partialling out the exogenous "
        "regressors the excluded instruments do not move every endogenous regressor "
        "independently. Add relevant instruments or drop an endogenous regressor.")
    # Instruments that explain a share below 1e-20 of an endogenous regressor's variation
    # (given X1) are irrelevant to rounding: G is noise and must not be inverted.
    moved = weighted_crossprod(g, weights).diagonal()
    if bool((moved <= 1e-20 * weighted_crossprod(p.x2p, weights).diagonal()).any()):
        raise failure
    try:
        if kappa == 1.0:
            fit = least_squares(g, p.y_p, weights, drop_collinear=True, tol=0.0)
            if fit.omitted:
                raise failure
            b2, inner, condition = fit.beta, fit.xtx_inv, fit.condition_number
        else:
            shrink = kappa - 1.0
            gram = weighted_crossprod(g, weights) - shrink * weighted_crossprod(p.e, weights)
            rhs = weighted_crossprod(g, weights, p.y_p) \
                - shrink * weighted_crossprod(p.e, weights, p.e_y)
            inner = cholesky_inverse(gram, code="singular_design", what="k-class moment matrix")
            b2 = inner @ rhs
            spread = gram.diagonal().sqrt()
            condition = float(torch.linalg.cond(gram / spread[:, None] / spread))
    except KernelError as exc:
        if exc.code != "singular_design":
            raise
        raise failure from exc
    c2, c_y = p.partial.beta[:, :q], p.partial.beta[:, q]
    b1 = c_y - c2 @ b2
    k1 = p.k1
    bread = torch.empty((k1 + q, k1 + q), dtype=torch.float64)
    cross = c2 @ inner
    bread[:k1, :k1] = p.partial.xtx_inv + cross @ c2.T
    bread[:k1, k1:] = -cross
    bread[k1:, :k1] = -cross.T
    bread[k1:, k1:] = inner
    resid = p.y_p - p.x2p @ b2
    rss = float(resid.square().sum() if weights is None else (weights * resid.square()).sum())
    return KClass(torch.cat([b1, b2]), symmetrize(bread), resid, rss, float(kappa), g, inner,
                  condition)


@torch.no_grad()
def score_regressors(x1: Tensor, p: Projections, kappa: float = 1.0) -> Tensor:
    """Xk = (I - kappa M_Z) X = [X1, P_Z X2 - (kappa - 1) M_Z X2], [n, K]."""
    second = p.xhat2 if kappa == 1.0 else p.xhat2 - (kappa - 1.0) * p.e
    return torch.cat([x1, second], dim=1)


@torch.no_grad()
def liml_kappa(p: Projections, weights: Tensor | None = None) -> float:
    """Smallest eigenvalue of ``(Y'M_Z Y)^-1 Y'M_1 Y`` for ``Y = [X2 y]``.

    With ``Y'M_Z Y = L L'`` the eigenvalues are those of the symmetric matrix
    ``L^-1 (Y'M_1 Y) L^-T``. kappa >= 1, with equality when the model is
    exactly identified.
    """
    q = p.q
    inner = weighted_crossprod(p.first.resid, weights)
    outer = weighted_crossprod(p.partial.resid[:, :q + 1], weights)
    factor, info = torch.linalg.cholesky_ex(inner)
    # Residual variation below 1e-20 of the variation given X1 is rounding noise.
    lost = (factor.diagonal().square() <= 1e-13 * inner.diagonal()) \
        | (inner.diagonal() <= 1e-20 * outer.diagonal())
    if int(info) != 0 or bool(lost.any()) or not bool(torch.isfinite(factor).all()):
        raise KernelError("perfect_first_stage", "An endogenous regressor (or the outcome) is "
                          "fitted exactly by the instruments, so the LIML eigenvalue is "
                          "undefined. Use method='2sls', or treat that regressor as exogenous.")
    half = torch.linalg.solve_triangular(factor, outer, upper=False)
    reduced = torch.linalg.solve_triangular(factor, half.T, upper=False)
    return float(torch.linalg.eigvalsh(symmetrize(reduced))[0])


@torch.no_grad()
def cragg_donald(p: Projections, g: Tensor, weights: Tensor | None = None) -> float:
    """Minimum eigenvalue of ``(X2'M_Z X2)^-1 X2'(P_Z - P_1) X2``.

    Times ``(N - L) / L2`` this is the Cragg-Donald Wald F statistic (Stata's
    "minimum eigenvalue statistic"); it is the squared smallest canonical
    correlation c between the partialled-out endogenous regressors and excluded
    instruments expressed as c^2 / (1 - c^2).
    """
    noise = weighted_crossprod(p.e, weights)
    factor, info = torch.linalg.cholesky_ex(noise)
    lost = (factor.diagonal().square() <= 1e-13 * noise.diagonal()) \
        | (noise.diagonal() <= 1e-20 * weighted_crossprod(p.x2p, weights).diagonal())
    if int(info) != 0 or bool(lost.any()):
        raise KernelError("perfect_first_stage", "An endogenous regressor is fitted exactly by "
                          "the instruments; the weak-identification statistic is undefined.")
    half = torch.linalg.solve_triangular(factor, weighted_crossprod(g, weights), upper=False)
    reduced = torch.linalg.solve_triangular(factor, half.T, upper=False)
    return float(torch.linalg.eigvalsh(symmetrize(reduced))[0])


@torch.no_grad()
def rank_test_rows(p: Projections, weights: Tensor | None = None) -> tuple[float, Tensor]:
    """Ingredients of the Kleibergen-Paap rk statistic for H0: rank(Pi) = q - 1.

    With ``F_y'(X2p'W X2p)F_y = I`` and ``F_z'(Z2p'W Z2p)F_z = I`` (inverse
    Cholesky factors) the matrix ``theta = F_y' X2p'W Z2p F_z`` [q, L2] has the
    canonical correlations as singular values, ``theta = U S V'``. The rk
    statistic tests that the smallest one, s_q, is zero: with u the last left
    singular vector and V2 the last L2 - q + 1 right singular vectors,

        lambda = V2' theta' u = (s_q, 0, ..., 0)',
        rk     = lambda' Omega^-1 lambda,   Omega = Var(lambda),

    where Omega is the (robust, cluster, ...) meat of the moment rows
    ``(Z2p F_z V2)_i * (E F_y u)_i``. (Kleibergen and Paap's A_q and B_q differ
    from u and V2 only by orthogonal rotations, which leave rk unchanged.)
    Returns ``(s_q, rows)`` with rows [n, L2 - q + 1]; the first column pairs
    with s_q.
    """
    q = p.q
    try:
        lower_y = torch.linalg.cholesky(weighted_crossprod(p.x2p, weights))
        lower_z = torch.linalg.cholesky(weighted_crossprod(p.z2p, weights))
    except RuntimeError as exc:
        raise KernelError("singular_design", "The partialled-out endogenous regressors or "
                          "instruments are rank deficient.") from exc
    cross = weighted_crossprod(p.x2p, weights, p.z2p)
    theta = torch.linalg.solve_triangular(
        lower_z, torch.linalg.solve_triangular(lower_y, cross, upper=False).T, upper=False).T
    u, s, vh = torch.linalg.svd(theta, full_matrices=True)
    direction = torch.linalg.solve_triangular(lower_y.T, u[:, q - 1:q], upper=True)
    basis = torch.linalg.solve_triangular(lower_z.T, vh[q - 1:].T, upper=True)
    return float(s[q - 1]), (p.z2p @ basis) * (p.e @ direction)


@dataclass
class GMMStep:
    beta: Tensor               # [K]
    bread: Tensor              # (A'S^-1 A)^-1, [K, K]
    loading: Tensor            # S^-1 A, [L, K]: score regressors are Z @ loading
    j: float                   # g(b)'S^-1 g(b)


@torch.no_grad()
def gmm_step(zx: Tensor, zy: Tensor, s: Tensor) -> GMMStep:
    """One linear GMM step for moment covariance ``S``: see the module docstring.

    ``zx = Z'WX`` [L, K], ``zy = Z'Wy`` [L]. ``S`` must be positive definite
    (it is not when there are fewer clusters than instruments).
    """
    solved = cholesky_solve(s, torch.cat([zx, zy[:, None]], dim=1),
                            code="singular_weight_matrix", what="GMM moment covariance")
    loading, s_inv_zy = solved[:, :-1], solved[:, -1]
    bread = cholesky_inverse(zx.T @ loading, code="underidentified",
                             what="GMM moment Jacobian A'S^-1 A")
    beta = bread @ (zx.T @ s_inv_zy)
    j = quadratic_form(zy - zx @ beta, s, inverse=True)
    return GMMStep(beta, bread, loading, j)
