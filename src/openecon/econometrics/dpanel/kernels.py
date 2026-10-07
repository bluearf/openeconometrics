"""Linear GMM kernels of dynamic panel estimation (float64 tensors only).

Notation. The stacked system has R rows (transformed rows, then level rows in
system GMM) in G panels, regressors X [R, k], outcome y [R] and instruments
Z [R, L]. ``A = Z'X``, ``b = Z'y`` and the per-panel moment rows
``g_i = Z_i'u_i`` ([G, L], one ``index_add_`` pass).

GMM with weight W minimizes ``(b - A beta)' W (b - A beta)``. With a root R_W
(``R_W'R_W = W``, see ``weight_root``) this is the least-squares problem of
``R_W b`` on ``R_W A`` with r rows and k columns, solved by Householder QR:

    beta = (A'WA)^-1 A'W b,   bread = (A'WA)^-1,   J = ||R_W (b - A beta)||^2.

Weight matrices are generalized inverses: W = S (S Omega S)^+ S with
S = diag(Omega)^-1/2, i.e. the Moore-Penrose inverse of the instrument
correlation-scale matrix (invariant to the units of each instrument). A full
rank Omega gives the ordinary inverse. Omega is never formed: every weight
matrix is the inverse of a Gram matrix ``B'B`` (``B = N'Z`` for the one-step
``Z'HZ``, the [G, L] panel moments for ``sum_i g_i g_i'``), and its root comes
from a QR factor of B and an SVD of that small factor.

Covariances (Arellano and Bond 1991; Windmeijer 2005; Roodman 2009):

    one-step nonrobust   sigma2 (A'W1A)^-1
    one-step robust      P1 Omega(e1) P1',   P = (A'WA)^-1 A'W,  Omega(e) = sum_i g_i g_i'
    two-step nonrobust   (A'W2A)^-1,  W2 = Omega(e1)^+
    two-step robust      Windmeijer: V2 + D V2 + V2 D' + D V1r D'

where column j of D is ``V2 A'W2 [sum_i Z_i'(x_ij e1_i' + e1_i x_ij')Z_i] W2 Z'e2``
(the derivative of the two-step estimator with respect to the one-step
estimate through the weight matrix).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_sums
from openecon.engines.linalg import collinear_columns, least_squares, symmetrize

_EPS = torch.finfo(torch.float64).eps
# Elements of one temporary block (rows x columns) in the blocked products.
_BLOCK_ELEMENTS = 1 << 22
# Relative singular-value cutoff of the column-equilibrated moment factor (an
# eigenvalue cutoff of 1e-22 on the correlation-scale moment matrix).
RANK_TOL = 1e-11
# Relative sum-of-squares threshold of the instrument collinearity screen.
INSTRUMENT_TOL = 1e-18


def triangular_factor(blocks: Sequence[Tensor], scale: Tensor | None = None) -> Tensor:
    """R with ``R'R = sum_b B_b' B_b`` (columns multiplied by ``scale``), by blocked QR.

    Tall-skinny QR over row blocks of every matrix in ``blocks``: no Gram
    matrix is formed (its condition number would be squared) and no second
    copy of a tall matrix is made.
    """
    factor = None
    for matrix in blocks:
        step = max(matrix.shape[1], _BLOCK_ELEMENTS // max(matrix.shape[1], 1))
        for start in range(0, len(matrix), step):
            block = matrix[start:start + step]
            if scale is not None:
                block = block * scale
            stacked = block if factor is None else torch.cat([factor, block])
            factor = torch.linalg.qr(stacked, mode="r").R
    if factor is None:
        raise KernelError("insufficient_observations", "There are no rows to factor.")
    return factor


def weight_root(blocks: Sequence[Tensor], *, tol: float = RANK_TOL) -> tuple[Tensor, int]:
    """A root ``R_W`` [r, L] of the generalized inverse of ``Omega = sum_b B_b'B_b`` and r.

    With ``S = diag(Omega)^-1/2`` (scale 1 for an all-zero column) the blocks
    are column-equilibrated, factored by QR into ``T`` and ``T = U s V'`` by an
    SVD of the small factor, so ``S Omega S = V s^2 V'`` and

        R_W = s^-1 V' S,    R_W'R_W = S (S Omega S)^+ S,

    the ordinary inverse when Omega has full rank. Singular values at or below
    ``tol`` times the largest are treated as zero (r is the numerical rank);
    working with the factor instead of Omega keeps the precision of the
    moment directions whose eigenvalues are far below the largest one.
    """
    for matrix in blocks:
        if not bool(torch.isfinite(matrix).all()):
            raise KernelError("non_finite_values", "The GMM moment matrix is not finite.")
    norms = sum(torch.linalg.vector_norm(matrix, dim=0).square() for matrix in blocks)
    scale = torch.where(norms > 0, norms.clamp(min=1e-300).rsqrt(), torch.ones_like(norms))
    factor = triangular_factor(blocks, scale)
    try:
        _, singular, vh = torch.linalg.svd(factor, full_matrices=False)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The weight-matrix factorization failed: "
                          f"{exc}") from exc
    top = float(singular[0]) if len(singular) else 0.0
    if not top > 0:
        raise KernelError("singular_weight_matrix", "The GMM weight matrix is zero: the "
                          "instruments carry no information.")
    keep = singular > tol * top
    root = (vh[keep] / singular[keep, None]) * scale[None, :]
    return root, int(keep.sum())


@dataclass
class Step:
    beta: Tensor        # [k]
    bread: Tensor       # (A'WA)^-1, [k, k]
    criterion: float    # (b - A beta)' W (b - A beta)
    root: Tensor        # R_W, [r, L]
    rank: int           # rank of W


def gmm_step(a: Tensor, b: Tensor, root: Tensor, rank: int) -> Step:
    """Minimize the GMM criterion with weight ``root' root`` by QR least squares."""
    k = a.shape[1]
    if rank < k:
        raise KernelError("underidentified", f"The weight matrix has rank {rank} but {k} "
                          "coefficients are estimated: add instruments or remove regressors.")
    at, bt = root @ a, root @ b
    try:
        fit = least_squares(at, bt, drop_collinear=False)
    except KernelError as exc:
        if exc.code != "singular_design":
            raise
        raise KernelError("underidentified", "The instruments do not identify every "
                          "coefficient (A'WA is singular): some regressor is unrelated to all "
                          "instruments. Add instruments or remove that regressor.") from exc
    return Step(fit.beta, fit.xtx_inv, float(fit.ssr), root, rank)


def panel_moments(z: Tensor, resid: Tensor, codes: Tensor, groups: int) -> Tensor:
    """Per-panel moments ``g_i = Z_i'u_i`` as a [G, L] matrix (columns in blocks)."""
    out = torch.zeros((groups, z.shape[1]), dtype=torch.float64)
    step = max(1, _BLOCK_ELEMENTS // max(len(resid), 1))
    for start in range(0, z.shape[1], step):
        block = z[:, start:start + step] * resid[:, None]
        out[:, start:start + step].index_add_(0, codes, block)
    return out


def instrument_rank(z: Tensor) -> tuple[list[int], list[int]]:
    """Stata-style left-to-right collinearity screen of the instrument columns.

    The triangular factor R of Z (Z'Z = R'R) is accumulated over row blocks
    (``triangular_factor``), so no second copy of the stacked instrument matrix
    is made; the screen of ``linalg.collinear_columns`` then runs on R, whose
    columns have exactly the geometry of the columns of Z. Instruments cannot
    be centred (their zeros are structural), so the relative tolerance is
    ``INSTRUMENT_TOL`` (a relative length of 1e-9) rather than the regressor
    screen's 1e-13: columns of a variable with a large common level (y around
    1e8) differ only in a small part, which must not be mistaken for an exact
    dependency, while exact dependencies (relative length ~1e-15) are removed.
    """
    return collinear_columns(triangular_factor([z]), tol=INSTRUMENT_TOL)


def projector(step: Step, a: Tensor) -> Tensor:
    """``P = (A'WA)^-1 A'W`` [k, L], the influence of the moments on the estimate."""
    return step.bread @ (step.root @ a).T @ step.root


def sandwich(p: Tensor, moments: Tensor) -> Tensor:
    """``P (sum_i g_i g_i') P'`` without forming the L-by-L moment covariance."""
    scores = moments @ p.T
    return symmetrize(scores.T @ scores)


def windmeijer(*, z: Tensor, x: Tensor, codes: Tensor, groups: int, a: Tensor, b: Tensor,
               moments_one: Tensor, two: Step, v_one: Tensor) -> tuple[Tensor, Tensor]:
    """Windmeijer (2005) finite-sample corrected two-step covariance and the matrix D.

    With q = W2 Z'e2, s_i = g_i'q (one-step moments g_i) and a_i = Z_i'X_i:
    ``sum_i Z_i'(x_ij e1_i' + e1_i x_ij') Z_i q`` stacked over j is
    ``M = sum_i a_i s_i + g_i (a_i'q)'``; then ``D = V2 A'W2 M`` and
    ``V_c = V2 + D V2 + V2 D' + D V1 D'`` with V1 the one-step robust covariance.
    """
    root = two.root
    q = root.T @ (root @ (b - a @ two.beta))
    s = moments_one @ q
    term = z.T @ (x * s[codes][:, None])
    per_panel = group_sums(x * (z @ q)[:, None], codes, groups)
    m = term + moments_one.T @ per_panel
    d = two.bread @ (root @ a).T @ (root @ m)
    v2 = two.bread
    corrected = v2 + d @ v2 + v2 @ d.T + d @ v_one @ d.T
    return symmetrize(corrected), d


@dataclass
class ArInputs:
    """What the Arellano-Bond statistic needs about the fitted model."""

    resid_diff: Tensor          # [ND] differenced level residuals
    x_diff: Tensor              # [ND, k] differenced level regressors
    codes_diff: Tensor          # [ND] panel of each differenced row
    period_diff: Tensor         # [ND] period of each differenced row
    groups: int
    span: int
    covariance: Tensor          # covariance of beta (before any small-sample factor)
    p: Tensor                   # (A'WA)^-1 A'W of the reported step
    moments: Tensor | None      # [G, L] per-panel moments of the step's residuals (robust)
    sigma2: float | None = None             # homoskedastic form: error variance
    cross: Tensor | None = None             # [NL, L] M'Z_T (h = 2, 3) or [NT, L] Z_T (h = 1)
    adjoint: object | None = None           # differences operator (levels <- diffs)
    align: Tensor | None = None             # h = 1: differenced row dated like each row of Z_T


def arellano_bond(inputs: ArInputs, order: int) -> dict:
    """Arellano-Bond (1991) z test for zero order-``order`` autocorrelation of differenced errors.

    ``w`` is the differenced residual lagged ``order`` periods (zero when
    absent), a_i = sum_t w_it e_it per panel and the statistic is
    ``sum_i a_i / sqrt(v)`` with (xtabond2's ``_ARTests``)

        v = sum_i a_i^2 - 2 w'X* P (sum_i g_i a_i) + w'X* V X*'w

    after robust or two-step estimation (g_i the step's per-panel moments, V
    its covariance). After one-step nonrobust estimation the homoskedastic form
    replaces e_i e_i' by sigma2 H: ``sigma2 w'D D'w`` and the cross covariance
    ``sigma2 (M'Z_T)'(D'w)`` of the differenced errors with the transformed
    equation (M the transformation; xtabond2 sets the cross covariance with the
    level equation of system GMM to zero), or, for h = 1, ``sigma2 w'w`` and
    ``sigma2 Z_T'w`` with w matched to the transformed rows by date.
    """
    from openecon.engines.distributions import normal_sf

    e, codes, period = inputs.resid_diff, inputs.codes_diff, inputs.period_diff
    span = inputs.span
    lookup = torch.full((inputs.groups * span,), -1, dtype=torch.int64)
    lookup[codes * span + period] = torch.arange(len(e), dtype=torch.int64)
    earlier = period - order
    inside = earlier >= 0
    partner = torch.where(inside, lookup[codes * span + earlier.clamp(min=0)],
                          torch.full_like(period, -1))
    available = partner >= 0
    pairs = int(available.sum())
    label = f"Arellano-Bond test for AR({order}) in first differences"
    if pairs == 0:
        return {"statistic": None, "p_value": None, "distribution": "normal", "label": label,
                "note": f"no panel has differenced residuals {order} periods apart"}
    w = torch.where(available, e[partner.clamp(min=0)], torch.zeros_like(e))
    a = group_sums(w * e, codes, inputs.groups)
    numerator = float(a.sum())
    xw = inputs.x_diff.T @ w
    if inputs.moments is not None:
        first = float(a @ a)
        middle = float(xw @ (inputs.p @ (inputs.moments.T @ a)))
    elif inputs.align is not None:
        first = inputs.sigma2 * float(w @ w)
        dated = torch.where(inputs.align >= 0, w[inputs.align.clamp(min=0)],
                            torch.zeros(len(inputs.align), dtype=w.dtype))
        middle = inputs.sigma2 * float(xw @ (inputs.p @ (inputs.cross.T @ dated)))
    else:
        dw = inputs.adjoint.adjoint(w)
        first = inputs.sigma2 * float(dw @ dw)
        middle = inputs.sigma2 * float(xw @ (inputs.p @ (inputs.cross.T @ dw)))
    last = float(xw @ inputs.covariance @ xw)
    variance = first - 2 * middle + last
    if not math.isfinite(variance) or variance <= 1e-14 * max(abs(first), abs(last), _EPS):
        return {"statistic": None, "p_value": None, "distribution": "normal", "label": label,
                "note": "the variance of the test statistic is not positive", "pairs": pairs}
    statistic = numerator / math.sqrt(variance)
    return {"statistic": statistic, "p_value": 2 * normal_sf(abs(statistic)),
            "distribution": "normal", "label": label, "pairs": pairs}
