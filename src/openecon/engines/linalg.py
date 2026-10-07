"""Shared float64 least-squares and symmetric-matrix kernels.

Least squares.  min_b sum_i w_i (y_i - x_i'b)^2 is solved from ONE Householder QR
(LAPACK geqrf) of A = sqrt(W) X.  Q'sqrt(W)y is obtained from the stored reflectors
(ormqr), so X'X is never formed or inverted and an explicit Q is built only when
leverages are requested.  With A = Q R:

    b = R^{-1} Q'sqrt(W)y,   (X'WX)^{-1} = R^{-1} R^{-T},   h_i = ||row_i(Q)||^2.

Column scaling.  Householder QR commutes with column scaling: every reflector and
every transformed column scales exactly, so for power-of-two scales d_j the factor
of A diag(1/d) is bit-for-bit R diag(1/d).  The scaling is therefore applied to the
k-by-k factor instead of to a second n-by-k copy of the design, and b and
(X'WX)^{-1} are unscaled exactly (multiplication by powers of two).  All tolerance
tests, triangular solves and the condition number use the scaled factor, whose
columns have Euclidean norm in [1/2, 1).

Collinearity.  Columns are screened left to right, Stata style: column j is omitted
when its (weighted) sum of squares after projecting out the previously KEPT columns
is <= tol times its original (weighted) sum of squares.  Because R'R = A'A, the
columns of the small factor R have exactly the geometry of the columns of A, so the
sequential orthogonalization runs on R (O(k^3)) and never touches the n rows again.
The diagonal of an unpivoted R is the wanted residual norm only up to the first
dependent column; from there on the columns of R are orthogonalized explicitly
(classical Gram-Schmidt applied twice) against the kept ones, skipping dependent
columns.  Precision: R is the exact factor of A + E with ||E_j|| ~ eps ||A_j||, so
an exactly dependent column a_j = A_kept c leaves a relative residual of order
eps (1 + ||c||_1), i.e. a sum-of-squares ratio near 1e-30 for ordinary dependencies
(dummy traps, duplicates, sums).  This is far below what a sweep of the Gram matrix
resolves (about 1e-15), which is why the default tolerance can sit at 1e-13: exact
dependencies are found unless their coefficients are magnified beyond ~1e9, and a
column with independent variation of relative length above ~3e-7 is kept.

Symmetric helpers (Cholesky inverse/solve, weighted cross products, quadratic forms
and the generalized Wald statistic) are small k-by-k routines used by every
estimator's covariance and test code.  The Cholesky helpers treat a pivot at the
rounding level (L_jj^2 <= 8 k eps A_jj) as zero: LAPACK alone accepts about half of
all exactly singular Gram matrices, whose rounded pivot happens to be positive, and
then returns an "inverse" of order 1/eps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import KernelError

_EPS = torch.finfo(torch.float64).eps
# Rows are processed in blocks of about this many elements when a weighted cross
# product would otherwise need an n-by-k temporary.
_BLOCK_ELEMENTS = 1 << 22


@dataclass
class LeastSquares:
    beta: Tensor               # [k_kept] (or [k_kept, m] when y is [n, m])
    kept: list[int]            # input-column indices of retained columns, ascending
    omitted: list[int]         # input-column indices dropped for collinearity, ascending
    xtx_inv: Tensor            # (X'WX)^{-1} over kept columns, [k_kept, k_kept], symmetric
    fitted: Tensor             # X beta, same shape as y
    resid: Tensor              # y - X beta (NOT weighted)
    ssr: Tensor                # 0-dim (or [m]) weighted residual sum of squares
    rank: int
    condition_number: float    # of the unit-norm column-scaled kept design sqrt(W) X
    leverage: Tensor | None    # h_i = w_i x_i'(X'WX)^{-1} x_i when need_leverage=True


def _matrix(a: Tensor, what: str, code: str) -> int:
    if not isinstance(a, Tensor) or a.dtype != torch.float64 or a.ndim != 2 \
            or a.shape[0] != a.shape[1]:
        raise KernelError(code, f"The {what} must be a square float64 matrix.")
    return a.shape[0]


def _design(x: Tensor, weights: Tensor | None, tol: float) -> tuple[int, int]:
    if not isinstance(x, Tensor) or x.dtype != torch.float64 or x.ndim != 2:
        raise KernelError("invalid_design", "The design must be an n-by-k float64 tensor.")
    n, k = x.shape
    if n < 1:
        raise KernelError("invalid_design", "The design has no observations.")
    if weights is not None:
        if not isinstance(weights, Tensor) or weights.dtype != torch.float64 \
                or weights.shape != (n,):
            raise KernelError("invalid_weights", "Weights must be a float64 vector, one per row.")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise KernelError("invalid_weights", "Weights must be finite and nonnegative.")
    if not 0 <= tol < 1:
        raise KernelError("invalid_solver_options",
                          "The collinearity tolerance must lie in [0, 1).")
    return n, k


def _column_scale(r: Tensor) -> Tensor:
    """Power-of-two d_j with ||r_j / d_j|| in [1/2, 1); d_j = 1 for a zero column.

    The peak is removed before the norm is taken so that neither step can overflow
    or underflow, and both divisions are exact.
    """
    def power(exponents: Tensor) -> Tensor:
        return torch.tensor([math.ldexp(1.0, max(-1022, min(1023, e)))
                             for e in exponents.tolist()], dtype=r.dtype, device=r.device)

    first = torch.frexp(r.abs().amax(dim=0)).exponent
    second = torch.frexp(torch.linalg.vector_norm(r / power(first), dim=0)).exponent
    return power(first + second)


def _factor(x: Tensor, weights: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """geqrf of sqrt(W) X; returns (reflectors, tau, scaled R [min(n,k), k], scales)."""
    n, k = x.shape
    design = x if weights is None else x * weights.sqrt()[:, None]
    reflectors, tau = torch.geqrf(design)
    r = reflectors[: min(n, k)].triu()
    # A NaN or infinity anywhere in a column reaches that column of R.
    if not bool(torch.isfinite(r).all()):
        raise KernelError("non_finite_values", "The design must be finite within float64 range.")
    scale = _column_scale(r)
    return reflectors, tau, r / scale, scale


def _screen(r: Tensor, tol: float) -> tuple[list[int], list[int]]:
    """Left-to-right dependence screen on the columns of the (scaled) QR factor."""
    p, k = r.shape
    total = r.square().sum(dim=0)
    lead = min(p, k)
    # |R_jj|^2 is the residual sum of squares of column j given ALL earlier columns,
    # which is the sequential quantity as long as no earlier column was dependent.
    failed = (r.diagonal().square() <= tol * total[:lead]).nonzero()
    start = int(failed[0]) if failed.numel() else lead
    if start == k:
        return list(range(k)), []
    kept, omitted, rank = list(range(start)), [], start
    # The first `start` columns span e_1..e_start; later kept columns extend the basis.
    basis = torch.zeros((p, lead), dtype=r.dtype, device=r.device)
    basis.diagonal()[:start] = 1
    limits = (tol * total).tolist()
    for j in range(start, k):
        rows = min(j + 1, p)
        v = r[:rows, j].clone()
        if rank:
            q = basis[:rows, :rank]
            v -= q @ (q.T @ v)
            v -= q @ (q.T @ v)      # second pass restores orthogonality to rounding
        left = float(v @ v)
        if rank < p and left > limits[j]:
            basis[:rows, rank] = v / math.sqrt(left)
            kept.append(j)
            rank += 1
        else:
            omitted.append(j)
    return kept, omitted


@torch.no_grad()
def collinear_columns(
    x: Tensor, weights: Tensor | None = None, *, tol: float = 1e-13
) -> tuple[list[int], list[int]]:
    """Stata-style left-to-right collinearity screen. Returns (kept, omitted).

    With a_j = sqrt(w) * x_j and P_j the projector on the span of the columns kept
    before j, column j is omitted when

        ||a_j - P_j a_j||^2  <=  tol * ||a_j||^2 .

    Earlier columns win: with an intercept in column 0 a constant predictor is
    omitted (not the intercept), a duplicated column loses its LATER copy and an
    all-zero column is always omitted, as in Stata's "omitted because of
    collinearity".  The ordering is exact when several columns are dependent: the
    test is a sequential orthogonalization of the QR factor's columns that skips
    dependent ones (see the module docstring for the method and its precision).
    The sums of squares are uncentered.  The default 1e-13 is a relative length of
    about 3e-7 (the order of R's lm tolerance, 1e-7 on the length), so x and
    x + 1e-6 * noise are both kept while exact dependencies are removed.
    """
    n, k = _design(x, weights, tol)
    if k == 0:
        return [], []
    try:
        _, _, r, _ = _factor(x, weights)
        return _screen(r, tol)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The QR factorization failed: {exc}") from exc


@torch.no_grad()
def least_squares(
    x: Tensor,
    y: Tensor,
    weights: Tensor | None = None,
    *,
    drop_collinear: bool = True,
    need_leverage: bool = False,
    tol: float = 1e-13,
) -> LeastSquares:
    """Weighted least squares  min_b sum_i w_i (y_i - x_i'b)^2  by Householder QR.

    With A = sqrt(W) X = Q R over the kept columns (column scaled, see module
    docstring; X'X is never formed or inverted):

        beta    = R^{-1} Q' sqrt(W) y
        xtx_inv = R^{-1} R^{-T}                       = (X'WX)^{-1}
        resid   = y - X beta   (unweighted),   ssr = sum_i w_i resid_i^2
        h_i     = ||row_i(Q)||^2                      = w_i x_i'(X'WX)^{-1} x_i

    y may be [n] or [n, m] (all right-hand sides share one factorization).  weights
    are nonnegative; analytic, sampling and frequency weights all enter through
    this single w, and the caller applies the matching degrees of freedom.  Rows
    with zero weight do not affect the fit but still receive fitted values.

    Collinear columns are screened by collinear_columns' rule with the same tol:
    drop_collinear=True reports them in `omitted` and solves on `kept` (Stata's
    behaviour); drop_collinear=False raises KernelError("singular_design").  If
    every column is omitted the fit is empty (rank 0, resid = y,
    condition_number 1.0).  Columns of very different magnitude (1e-8 .. 1e8 and
    beyond) are handled by exact power-of-two column scaling.
    """
    n, k = _design(x, weights, tol)
    if not isinstance(y, Tensor) or y.dtype != torch.float64 or y.ndim not in (1, 2) \
            or y.shape[0] != n:
        raise KernelError("invalid_design", "The outcome must be float64 with one row per "
                          "design row ([n] or [n, m]).")
    if not bool(torch.isfinite(y).all()):
        raise KernelError("non_finite_values", "The outcome must be finite.")
    y2 = y if y.ndim == 2 else y[:, None]
    m = y2.shape[1]
    try:
        kept: list[int] = []
        omitted: list[int] = []
        if k:
            reflectors, tau, r, scale = _factor(x, weights)
            kept, omitted = _screen(r, tol)
        if omitted and not drop_collinear:
            raise KernelError(
                "singular_design",
                f"The design is rank deficient: column(s) {omitted} are collinear with "
                "earlier columns.",
            )
        rank = len(kept)
        leverage = None
        if rank == 0:
            beta = torch.zeros((0, m), dtype=x.dtype, device=x.device)
            xtx_inv = torch.zeros((0, 0), dtype=x.dtype, device=x.device)
            fitted = torch.zeros_like(y2)
            condition = 1.0
            if need_leverage:
                leverage = torch.zeros(n, dtype=x.dtype, device=x.device)
        else:
            p = min(n, k)
            target = y2 if weights is None else y2 * weights.sqrt()[:, None]
            qty = torch.ormqr(reflectors[:, :p], tau, target, left=True, transpose=True)[:p]
            if omitted:
                # A_kept = Q (R[:, kept]) = (Q U) R_kept: re-triangularize the small factor.
                rotation, r_kept = torch.linalg.qr(r[:, kept])
                qty = rotation.T @ qty
            else:
                rotation, r_kept = None, r
            singular = torch.linalg.svdvals(r_kept / torch.linalg.vector_norm(r_kept, dim=0))
            condition = float(singular[0] / singular[-1])
            if not math.isfinite(condition) or condition * _EPS >= 1:
                raise KernelError(
                    "singular_design",
                    "The kept columns are numerically rank deficient; raise the collinearity "
                    "tolerance or rescale the design.",
                )
            unit = scale[kept][:, None]
            beta = torch.linalg.solve_triangular(r_kept, qty, upper=True) / unit
            eye = torch.eye(rank, dtype=x.dtype, device=x.device)
            half = torch.linalg.solve_triangular(r_kept, eye, upper=True) / unit
            xtx_inv = symmetrize(half @ half.T)
            if not bool(torch.isfinite(beta).all()) or not bool(torch.isfinite(xtx_inv).all()):
                raise KernelError("numerical_failure", "Column units exceed float64 coefficient "
                                  "precision; rescale the design.")
            if omitted:
                # Zero coefficients on omitted columns avoid copying X[:, kept].
                full = torch.zeros((k, m), dtype=x.dtype, device=x.device)
                full[kept] = beta
                fitted = x @ full
            else:
                fitted = x @ beta
            if need_leverage:
                q = torch.linalg.householder_product(reflectors[:, :p], tau)
                if rotation is not None:
                    q = q @ rotation
                leverage = torch.einsum("ij,ij->i", q, q)
        resid = y2 - fitted
        squares = resid.square()
        ssr = (squares if weights is None else squares * weights[:, None]).sum(dim=0)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The QR least-squares solve failed: {exc}") from exc
    if y.ndim == 1:
        beta, fitted, resid, ssr = beta[:, 0], fitted[:, 0], resid[:, 0], ssr[0]
    return LeastSquares(
        beta=beta, kept=kept, omitted=omitted, xtx_inv=xtx_inv, fitted=fitted, resid=resid,
        ssr=ssr, rank=rank, condition_number=condition, leverage=leverage,
    )


@torch.no_grad()
def symmetrize(a: Tensor) -> Tensor:
    """(A + A') / 2: removes the rounding asymmetry of a mathematically symmetric matrix."""
    return (a + a.transpose(-1, -2)) / 2


def _cholesky(a: Tensor, code: str, what: str) -> Tensor:
    k = _matrix(a, what, code)
    matrix = symmetrize(a)
    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    # The pivot L_jj^2 = A_jj - sum_{i<j} L_ji^2 is the part of variable j not explained by
    # the earlier ones and carries a rounding error of order k eps A_jj.  For a singular
    # matrix (a Gram matrix with a collinear column, a meat with fewer clusters than
    # parameters) it is rounding noise of either sign: LAPACK reports the negative half and
    # "succeeds" on the positive half, returning an inverse of order 1/eps.  A pivot at the
    # noise level is therefore treated as zero.  The ratio is invariant to the units of A.
    lost = factor.diagonal().square() <= 8 * k * _EPS * matrix.diagonal()
    if int(info.item()) != 0 or not bool(torch.isfinite(factor).all()) or bool(lost.any()):
        raise KernelError(code, f"The {what} is not positive definite to working precision.")
    return factor


def _finite(result: Tensor, code: str, what: str) -> Tensor:
    if not bool(torch.isfinite(result).all()):
        raise KernelError(code, f"The inverse of the {what} is not representable in float64.")
    return result


@torch.no_grad()
def cholesky_inverse(
    a: Tensor, *, code: str = "singular_matrix", what: str = "matrix"
) -> Tensor:
    """A^{-1} = L^{-T} L^{-1} from the Cholesky factor A = L L' of a symmetric positive
    definite matrix; KernelError(code, ...) if A is not SPD. The result is symmetrized.

    "Not SPD" includes numerically singular matrices: a Cholesky pivot
    L_jj^2 <= 8 k eps A_jj is rounding noise and raises instead of producing an
    inverse of order 1/eps (matrices with condition numbers up to about 1e13 pass)."""
    return _finite(symmetrize(torch.cholesky_inverse(_cholesky(a, code, what))), code, what)


@torch.no_grad()
def cholesky_solve(
    a: Tensor, b: Tensor, *, code: str = "singular_matrix", what: str = "matrix"
) -> Tensor:
    """Solve A z = b (b is [k] or [k, m]) through A = L L': z = L^{-T} (L^{-1} b).

    KernelError(code, ...) if A is not symmetric positive definite to working
    precision (same pivot rule as cholesky_inverse)."""
    factor = _cholesky(a, code, what)
    if not isinstance(b, Tensor) or b.dtype != a.dtype or b.ndim not in (1, 2) \
            or b.shape[0] != a.shape[0]:
        raise KernelError("invalid_input", f"The right-hand side does not conform to the {what}.")
    if b.ndim == 1:
        return _finite(torch.cholesky_solve(b[:, None], factor)[:, 0], code, what)
    return _finite(torch.cholesky_solve(b, factor), code, what)


@torch.no_grad()
def weighted_crossprod(x: Tensor, w: Tensor | None, z: Tensor | None = None) -> Tensor:
    """X' diag(w) Z = sum_i w_i x_i z_i' without forming diag(w); z=None gives X'WX.

    x is [n, k]; z is [n, q] (result [k, q]) or [n] (result [k], e.g. X'Wy); w=None
    means unit weights. Weights may be negative (Hessians of non-canonical links).
    X'WX is symmetrized. Large inputs are accumulated over row blocks so the
    weighted copy of X never exceeds a few megabytes.
    """
    if not isinstance(x, Tensor) or x.ndim != 2 or x.dtype != torch.float64:
        raise KernelError("invalid_input", "weighted_crossprod expects an n-by-k float64 matrix.")
    n, k = x.shape
    right = x if z is None else z
    if not isinstance(right, Tensor) or right.dtype != x.dtype or right.ndim not in (1, 2) \
            or right.shape[0] != n or (w is not None and (
                not isinstance(w, Tensor) or w.dtype != x.dtype or w.shape != (n,))):
        raise KernelError("invalid_input", "weighted_crossprod operands must be float64 with "
                          "one row (weight) per row of x.")
    rows = max(1, _BLOCK_ELEMENTS // max(k, 1))
    if w is None:
        out = x.T @ right
    elif n <= rows:
        out = (x * w[:, None]).T @ right
    else:
        out = torch.zeros((k,) + right.shape[1:], dtype=x.dtype, device=x.device)
        for start in range(0, n, rows):
            stop = min(n, start + rows)
            out += (x[start:stop] * w[start:stop, None]).T @ right[start:stop]
    return symmetrize(out) if z is None else out


@torch.no_grad()
def quadratic_form(v: Tensor, a_inv_or_a: Tensor, *, inverse: bool) -> float:
    """Quadratic form in v.

    inverse=True:  the matrix is A (symmetric positive definite) and the result is
                   v' A^{-1} v = ||L^{-1} v||^2 with A = L L' (no explicit inverse);
                   KernelError("singular_matrix") if A is not SPD. Use wald_statistic
                   when A may be singular.
    inverse=False: the matrix is used as given (typically an already inverted A):
                   v' M v.
    """
    k = _matrix(a_inv_or_a, "matrix", "invalid_input")
    if not isinstance(v, Tensor) or v.dtype != a_inv_or_a.dtype or v.shape != (k,):
        raise KernelError("invalid_input", "The vector does not conform to the matrix.")
    if not inverse:
        return float(v @ (a_inv_or_a @ v))
    factor = _cholesky(a_inv_or_a, "singular_matrix", "matrix")
    half = torch.linalg.solve_triangular(factor, v[:, None], upper=False)
    return float(half.square().sum())


@torch.no_grad()
def wald_statistic(
    estimate: Tensor, covariance: Tensor, *, tol: float = 1e-12
) -> tuple[float, int]:
    """Return (b' V^- b, rank(V)): the Wald statistic with a generalized inverse.

    V is scaled to a correlation matrix C = D V D, D = diag(V)^{-1/2} (the statistic
    is invariant to this, and the eigenvalue cutoff becomes unit free), and
    decomposed as C = U diag(l) U'. With c = U'(D b),

        W = sum_{j: l_j > tol * l_max} c_j^2 / l_j ,   rank = #{j: l_j > tol * l_max}.

    For a nonsingular V this is b'V^{-1}b with rank = len(b). A singular V (redundant
    constraints) gives the generalized Wald statistic with reduced degrees of
    freedom, as Stata's `test` does when it drops constraints. Eigenvalues at or
    below the cutoff, including negative ones of a non-PSD V, are excluded.
    KernelError("singular_covariance") if no eigenvalue survives.
    """
    k = _matrix(covariance, "covariance matrix", "invalid_covariance")
    if not isinstance(estimate, Tensor) or estimate.dtype != covariance.dtype \
            or estimate.shape != (k,):
        raise KernelError("invalid_covariance", "The estimate does not conform to its covariance.")
    variance = covariance.diagonal()
    if not bool(torch.isfinite(covariance).all()) or not bool(torch.isfinite(estimate).all()) \
            or bool((variance < 0).any()):
        raise KernelError("invalid_covariance", "The covariance matrix must be finite with a "
                          "nonnegative diagonal.")
    spread = torch.where(variance > 0, variance.sqrt(), torch.ones_like(variance))
    try:
        values, vectors = torch.linalg.eigh(symmetrize(covariance / spread[:, None] / spread))
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The eigen-decomposition failed: {exc}") from exc
    keep = (values > 0) & (values > tol * values[-1]) if k else values > 0
    rank = int(keep.sum())
    if rank == 0:
        raise KernelError("singular_covariance", "The covariance matrix has no positive "
                          "eigenvalue; the Wald statistic is undefined.")
    rotated = vectors[:, keep].T @ (estimate / spread)
    return float((rotated.square() / values[keep]).sum()), rank
