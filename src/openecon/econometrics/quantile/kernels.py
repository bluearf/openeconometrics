"""Tensor kernels of quantile regression: the check-loss linear program and its pieces.

Problem.  For 0 < tau < 1 the tau-th regression quantile minimizes

    f(b) = sum_i w_i rho_tau(y_i - x_i'b),      rho_tau(u) = u (tau - 1[u < 0]),

a linear program.  Because w_i > 0 the weights are folded into the rows
(w_i rho(y_i - x_i'b) = rho(w_i y_i - (w_i x_i)'b)), so only the unweighted problem
is solved.  Its dual in bounded variables a_i in [0, 1] is

    max y'a   subject to   X'a = (1 - tau) X'1,   0 <= a <= 1,

with a_i = 1 above the fitted hyperplane, 0 below it and strictly inside (0, 1) only
for the k observations the hyperplane interpolates (zero residuals): the solution is a
vertex, "basis" h of k rows with X_h b = y_h.

Solver.  ``solve`` runs the Frisch-Newton interior-point method of Portnoy and
Koenker (1997) on that dual with Mehrotra predictor-corrector steps.  With slacks
s = 1 - a and multipliers z, w >= 0 of the two bounds (z - w = X b - y is kept exact),
every iteration needs the k-by-k matrix X' D X, D = diag(1 / (z/a + w/s)), its Cholesky
factor and a handful of O(nk) products; no n-by-n object exists.  The design is first
replaced by the orthonormal factor Q of its (column-scaled) QR decomposition, which
makes the first Newton system the identity and the iteration invariant to the units
and offsets of the regressors; coefficients are mapped back through R.

Exact vertex.  An interior-point iterate only approaches the vertex.  Once the
relative duality gap is below 1e-6 the k rows with the smallest residuals (linearly
independent ones) are taken as a trial basis, the vertex X_h b = y_h is computed
exactly and its optimality certificate is evaluated: with psi_i = tau - 1[r_i < 0],

    X_h' lambda = - sum_{i not in h} psi_i x_i,      tau - 1 <= lambda_j <= tau  for all j.

If the certificate holds the vertex IS the linear-programming solution and the
iteration stops (typically at a gap between 1e-7 and 1e-10).  If the interior-point
method ends (gap below 1e-10, iteration limit or a failed factorization) without a
certified vertex, Barrodale-Roberts-style simplex steps finish the job: the basis row
with the largest violation leaves along the edge direction X_h^-1 e_j, the
weighted-median line search over the residual breakpoints (one sort) finds the entering
row, and the certificate is re-evaluated.  Each such pivot costs O(nk + n log n); a
converged interior point on continuous data needs none.

Ties.  Discrete outcomes, dummy-only designs and duplicated rows give vertices with more
than k zero residuals (degenerate) and often several minimizers.  A residual below 1e-10
of the typical least-squares residual counts as zero.  Two devices deal with ties:

* The interior dual point.  At a degenerate vertex the basic certificate above (every
  tied row on a bound) may fail although the vertex is optimal.  The interior-point dual
  iterate a lies in the interior of the optimal dual face, so d = a - (1 - tau) on the
  zero-residual rows Z, shifted by the least-norm correction that restores
  sum_Z d_i x_i = - sum_{not Z} psi_i x_i, is a non-basic certificate when it respects
  tau - 1 <= d_i <= tau (``_tied_certificate``).  This certifies heavily tied problems
  (thousands of ties, hundreds of indicator columns) without any simplex step.
* The perturbation rule.  When pivots are needed, tied rows are treated as lying an
  infinitesimal distance off the hyperplane (see ``_vertex``); pivots that change the
  basis without moving the coefficients keep the vertex, its residuals and its set of
  tied rows exactly, and the certificate reached is valid for the original problem.

``unique`` is True only when a certificate proves that no other coefficient vector
attains the minimum (it is never True for a non-unique solution; in rare tied cases a
unique solution is reported as possibly non-unique).

Also here: the weighted check loss, empirical quantiles in Stata's definition, the
kernels of Stata's ``kdensity`` and the three sparsity bandwidths of ``qreg``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_pdf, normal_ppf

_EPS = torch.finfo(torch.float64).eps
_STEP = 0.9995              # fraction of the distance to the boundary taken by a step
_VERTEX_GAP = 1e-6          # relative gap below which exact vertices are tried
_GAP = 1e-10                # relative gap at which the interior-point iteration stops
_MAX_ITERATIONS = 200
_DUAL_TOL = 1e-9            # tolerance of the optimality certificate tau-1 <= lambda <= tau
_INDEPENDENT = 1e-8         # relative length a row must keep to enter a trial basis
_EXACT_FIT = 1e-13          # sum |OLS residual| / sum |y| below which the fit is exact
_FLAT = 1e-11               # relative size below which an edge direction leaves a row fixed
_INTERIOR = 1e-7            # distance from its bounds at which a dual value is interior
_TIE = 1e-10                # residual / typical residual below which a row is on the hyperplane

KERNELS = ("epanechnikov", "epan2", "biweight", "cosine", "gaussian", "parzen", "rectangle",
           "triangle")
BANDWIDTHS = ("hsheather", "bofinger", "chamberlain")


@dataclass
class QuantileProblem:
    """A design prepared for repeated quantile fits (several tau share one QR)."""

    q: Tensor            # [n, k] orthonormal factor of the weighted, column-scaled design
    r: Tensor            # [k, k] upper triangular: (w x) / scale = q r
    scale: Tensor        # [k] column norms of the weighted design
    ys: Tensor           # [n] weighted outcome with its least-squares fit removed
    weights: Tensor | None
    row_norm: Tensor     # [n] Euclidean norms of the rows of q
    ols: Tensor          # [k] least-squares coefficients of the weighted outcome (q basis)
    level: float         # sum of the absolute weighted outcomes
    spread: float        # weighted mean absolute least-squares residual (unweighted units)

    def coefficients(self, gamma: Tensor) -> Tensor:
        """Map coefficients of the orthonormal basis back to the original regressors."""
        return torch.linalg.solve_triangular(self.r, gamma[:, None], upper=True)[:, 0] / self.scale


@dataclass
class QuantileFit:
    beta: Tensor         # [k]
    resid: Tensor        # [n] y - x beta (unweighted units); exact zeros on the basis and ties
    basis: Tensor        # int64 [k] rows interpolated by the fitted hyperplane
    objective: float     # sum_i w_i rho_tau(resid_i)
    iterations: int      # interior-point iterations
    pivots: int          # simplex steps needed after the interior point
    gap: float           # relative duality gap when the interior point stopped
    unique: bool         # the certificate proves that no other minimizer exists
    exact_fit: bool      # the outcome is an exact linear function of the design


def check_loss(resid: Tensor, tau: float, weights: Tensor | None = None) -> Tensor:
    """sum_i w_i rho_tau(r_i) with rho_tau(u) = u (tau - 1[u < 0]) (0-dim tensor)."""
    loss = torch.where(resid < 0, (tau - 1.0) * resid, tau * resid)
    return loss.sum() if weights is None else (loss * weights).sum()


def _check_tau(tau: float) -> float:
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 < tau < 1.0:
        raise KernelError("invalid_quantile", "The quantile must lie strictly between 0 and 1.")
    return float(tau)


@torch.no_grad()
def prepare(x: Tensor, y: Tensor, weights: Tensor | None = None) -> QuantileProblem:
    """Validate the data and factor the weighted design once for any number of quantiles."""
    if not isinstance(x, Tensor) or x.dtype != torch.float64 or x.ndim != 2:
        raise KernelError("invalid_design", "The design must be an n-by-k float64 matrix.")
    n, k = x.shape
    if not isinstance(y, Tensor) or y.dtype != torch.float64 or y.shape != (n,):
        raise KernelError("invalid_design", "The outcome must be float64 with one value per row.")
    if weights is not None and (not isinstance(weights, Tensor) or weights.dtype != torch.float64
                                or weights.shape != (n,)):
        raise KernelError("invalid_weights", "Weights must be float64 with one value per row.")
    if k == 0:
        raise KernelError("invalid_design", "Quantile regression needs at least one column.")
    if n <= k:
        raise KernelError("insufficient_observations", "Quantile regression needs more "
                          "observations than parameters.")
    if not bool(torch.isfinite(y).all()) or not bool(torch.isfinite(x).all()):
        raise KernelError("non_finite_values", "The design and the outcome must be finite.")
    if weights is not None and not bool(((weights > 0) & torch.isfinite(weights)).all()):
        raise KernelError("invalid_weights", "Quantile-regression weights must be positive.")
    ys = y if weights is None else y * weights
    scale = torch.linalg.vector_norm(x if weights is None else x * weights[:, None], dim=0)
    if not bool((scale > 0).all()):
        raise KernelError("singular_design", "The design contains an all-zero column.")
    xs = x / scale if weights is None else x * (weights[:, None] / scale)
    try:
        q, r = torch.linalg.qr(xs, mode="reduced")
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The QR factorization failed: {exc}") from exc
    del xs
    diagonal = r.diagonal().abs()
    # Columns have unit length, so |r_jj| is the length of column j orthogonal to the
    # earlier ones; the design passed the Stata-style collinearity screen before.
    if not bool(torch.isfinite(r).all()) or float(diagonal.min()) <= 1e-11:
        raise KernelError("singular_design", "The design is rank deficient; drop collinear "
                          "regressors.")
    # The linear program is solved for the least-squares residuals e = y - Q Q'y, an exact
    # reparametrization (gamma = Q'y + gamma_e) that removes the level of the outcome: every
    # rounding-level decision below is then relative to the residual scale, not to |y|.
    ols = q.T @ ys
    residual = ys - q @ ols
    return QuantileProblem(q=q, r=r, scale=scale, ys=residual, weights=weights,
                           row_norm=torch.linalg.vector_norm(q, dim=1), ols=ols,
                           level=float(ys.abs().sum()),
                           spread=float(residual.abs().sum())
                           / (n if weights is None else float(weights.sum())))


# ---- exact vertices ---------------------------------------------------------------------


@dataclass
class _Vertex:
    basis: Tensor            # int64 [k]
    gamma: Tensor            # [k] coefficients in the q basis
    resid: Tensor            # [n] weighted residuals, exact zeros on the basis and on ties
    zero: Tensor             # bool [n] zero residuals (basis rows and ties)
    psi: Tensor              # [n] tau - 1[resid < 0] (ties by their virtual sign), 0 on the basis
    lam: Tensor              # [k] multipliers of the basis rows
    factor: tuple[Tensor, Tensor]    # LU factorization of the basis rows of q
    violation: float         # max distance of lam outside [tau - 1, tau]
    ties: int                # zero residuals outside the basis (a degenerate vertex)
    virtual: Tensor | None   # [n] first-order perturbation of the tied residuals


def _independent_rows(rows: Tensor, k: int) -> list[int]:
    """Greedy choice of k linearly independent rows, earlier rows first.

    Each of the (at most k) steps takes the first row that keeps more than a relative
    length of 1e-8 after projecting out the rows already chosen, then removes its
    direction from all rows at once (modified Gram-Schmidt, O(mk) per step).
    """
    length = torch.linalg.vector_norm(rows, dim=1)
    work = rows.clone()
    chosen: list[int] = []
    for _ in range(k):
        rest = torch.linalg.vector_norm(work, dim=1)
        free = ((rest > _INDEPENDENT * length) & (length > 0)).nonzero().flatten()
        if not free.numel():
            break
        first = int(free[0])
        chosen.append(first)
        direction = work[first] / rest[first]
        work = work - (work @ direction)[:, None] * direction
        work = work - (work @ direction)[:, None] * direction
    return chosen


def _basis_near(problem: QuantileProblem, gamma: Tensor) -> Tensor:
    """The k independent rows with the smallest absolute residuals at ``gamma``."""
    q = problem.q
    n, k = q.shape
    magnitude = (problem.ys - q @ gamma).abs()
    size = min(n, 2 * k + 8)
    while True:
        order = torch.topk(magnitude, size, largest=False, sorted=True).indices
        chosen = _independent_rows(q[order], k)
        if len(chosen) == k:
            return order[chosen]
        if size == n:
            raise KernelError("singular_design", "The design is rank deficient; drop collinear "
                              "regressors.")
        size = min(n, 4 * size)


def _basis_factor(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """LU factorization of a basis matrix; one factorization serves every solve at a vertex."""
    lu, pivots, info = torch.linalg.lu_factor_ex(matrix)
    if int(info) != 0 or not bool(torch.isfinite(lu).all()):
        raise KernelError("numerical_failure", "A quantile-regression basis is singular.")
    return lu, pivots


def _basis_solve(factor: tuple[Tensor, Tensor], rhs: Tensor, *, transpose: bool = False) -> Tensor:
    """Solve X_h v = rhs (or X_h' v = rhs) with the stored factorization."""
    return torch.linalg.lu_solve(factor[0], factor[1], rhs[:, None], adjoint=transpose)[:, 0]


def _vertex(problem: QuantileProblem, tau: float, basis: Tensor,
            virtual: Tensor | None = None, previous: _Vertex | None = None) -> _Vertex:
    """Solve X_h b = y_h exactly and evaluate the optimality certificate of that vertex.

    Residuals at the rounding level outside the basis are ties: the vertex is degenerate.
    They are handled by the classical perturbation device: tied row i is treated as if
    its residual were eps * virtual_i for an infinitesimal eps > 0, i.e. as an ordinary
    row on the side given by the sign of virtual_i (all positive, with distinct values,
    when the vertex is first met).  A certificate found this way has every multiplier of
    a tied row on a bound of [tau - 1, tau], so it is a valid certificate of the
    unperturbed problem.

    ``previous`` is the vertex a degenerate pivot started from: the basis changed but the
    coefficients did not, so its coefficients, residuals and set of zero residuals are
    kept exactly.  Recomputing them from the new basis would move them by rounding errors
    (amplified by the conditioning of the basis), reclassify tied rows and make the
    perturbation inconsistent, which lets the pivoting cycle on heavily tied problems.
    """
    q, ys = problem.q, problem.ys
    try:
        factor = _basis_factor(q[basis])
        if previous is None:
            target = ys[basis]
            gamma = _basis_solve(factor, target)
            # One step of iterative refinement keeps the tied residuals at the rounding level.
            gamma = gamma + _basis_solve(factor, target - q[basis] @ gamma)
            resid = ys - q @ gamma
            resid[basis] = 0.0
            # Rounding errors of the basis solve grow with its conditioning, so a tie is any
            # (unweighted) residual below 1e-10 of the typical least-squares residual, or
            # below the rounding bound of its own row, whichever is larger.
            bound = 64 * _EPS * (ys.abs() + problem.row_norm * torch.linalg.vector_norm(gamma))
            floor = _TIE * problem.spread
            zero = resid.abs() <= (bound.clamp_min(floor) if problem.weights is None
                                   else torch.maximum(bound, floor * problem.weights))
            # Residuals at the rounding level are zeros (duplicates of a basis row, ties):
            # their sign is noise, so they are reported as exact zeros like the basis itself.
            resid[zero] = 0.0
        else:
            gamma, resid, zero = previous.gamma, previous.resid, previous.zero
        ties = int(zero.sum()) - basis.numel()
        psi = torch.full_like(resid, tau)
        psi[resid < 0] = tau - 1.0
        if ties:
            if virtual is None:
                # Distinct positive perturbations (golden-ratio sequence) break further ties.
                index = torch.arange(resid.numel(), dtype=torch.float64)
                virtual = 1.0 + torch.remainder(index * 0.6180339887498949, 1.0)
            virtual = virtual.clone()
            virtual[basis] = 0.0
            psi[zero & (virtual < 0)] = tau - 1.0
        else:
            virtual = None
        psi[basis] = 0.0
        lam = -_basis_solve(factor, q.T @ psi, transpose=True)
    except RuntimeError as exc:
        raise KernelError("numerical_failure",
                          f"A quantile-regression basis could not be solved: {exc}") from exc
    if not bool(torch.isfinite(lam).all()) or not bool(torch.isfinite(gamma).all()):
        raise KernelError("numerical_failure", "A quantile-regression basis is singular.")
    violation = float(torch.maximum(lam - tau, (tau - 1.0) - lam).max().clamp_min(0.0))
    return _Vertex(basis=basis, gamma=gamma, resid=resid, zero=zero, psi=psi, lam=lam,
                   factor=factor, violation=violation, ties=ties, virtual=virtual)


def _pivot(problem: QuantileProblem, tau: float, vertex: _Vertex) -> tuple[Tensor, Tensor | None]:
    """One simplex step from a vertex without a certificate: (new basis, new virtual).

    The basis row j with the largest violation leaves along delta = sigma X_h^-1 e_j
    (sigma = +1 when lambda_j < tau - 1, -1 when lambda_j > tau).  Along b + t delta the
    objective is piecewise linear and convex; with g = X delta its slope starts at

        sigma lambda_j + (1 - tau if sigma > 0 else tau)  <  0

    and rises by |g_i| each time a residual crosses zero, at t_i = r_i / g_i > 0.  The
    row at which the slope turns nonnegative enters the basis.  Tied rows cross first, at
    the infinitesimal steps virtual_i / g_i > 0: if the slope turns there the pivot is
    degenerate (the basis changes, the coefficients do not) and the perturbations are
    updated; otherwise the step is real and the next vertex starts a fresh perturbation.
    """
    q = problem.q
    k = q.shape[1]
    over = vertex.lam - tau
    under = (tau - 1.0) - vertex.lam
    j = int(torch.argmax(torch.maximum(over, under)))
    sigma = 1.0 if float(under[j]) > 0 else -1.0
    unit = torch.zeros(k, dtype=q.dtype)
    unit[j] = sigma
    g = q @ _basis_solve(vertex.factor, unit)
    # Rows that do not move along the edge have g = 0 up to rounding; a noise-level g must
    # not bring a row into the basis (the basis would be singular).
    g[g.abs() <= _FLAT * g.abs().max()] = 0.0
    g[vertex.basis] = 0.0
    g[vertex.basis[j]] = sigma
    slope = float(-(g @ vertex.psi)) + ((1.0 - tau) if sigma > 0 else tau)
    basis = vertex.basis.clone()
    if vertex.ties:
        virtual = vertex.virtual
        rows = (vertex.zero & (virtual * g > 0)).nonzero().flatten()
        if rows.numel():
            steps, order = torch.sort(virtual[rows] / g[rows])
            climb = slope + torch.cumsum(g[rows][order].abs(), dim=0)
            passed = (climb >= 0).nonzero().flatten()
            if passed.numel():
                first = int(passed[0])
                basis[j] = rows[order[first]]
                return basis, virtual - steps[first] * g
            slope = float(climb[-1])
    rows = ((~vertex.zero) & (vertex.resid * g > 0)).nonzero().flatten()
    if rows.numel():
        steps, order = torch.sort(vertex.resid[rows] / g[rows])
        passed = (slope + torch.cumsum(g[rows][order].abs(), dim=0) >= 0).nonzero().flatten()
        if passed.numel():
            basis[j] = rows[order[int(passed[0])]]
            return basis, None
    raise KernelError("numerical_failure", "The quantile-regression line search found no "
                      "breakpoint; the check loss is not bounded along an edge.")


def _tied_certificate(problem: QuantileProblem, tau: float, vertex: _Vertex,
                      dual: Tensor) -> bool | None:
    """Certify a degenerate vertex with a non-basic dual point; None if that fails.

    With Z the rows on the hyperplane (the basis and its ties), the vertex is optimal iff
    some d with tau - 1 <= d_i <= tau on Z satisfies

        sum_{i in Z} d_i x_i = - sum_{i not in Z} psi_i x_i,      psi_i = tau - 1[r_i < 0].

    The basic certificate looks for such a d that is on a bound for every tied row, which
    on heavily tied data (discrete outcomes, many indicator regressors) takes one simplex
    step per tie to find.  The interior-point dual iterate a already is (nearly) such a
    point, in the interior of the optimal dual face: d = a - (1 - tau) on Z, corrected by
    the least-norm shift that makes the equation exact once the other rows are put on
    their bounds.  If the corrected d respects the bounds (to the tolerance of the basic
    certificate) the vertex is optimal.  Returns whether the minimizer is unique: it is
    when the rows whose d_i lies strictly inside the bounds have full rank, because those
    rows have a zero residual in every minimizer (complementary slackness).
    """
    if not vertex.ties:
        return None
    q = problem.q
    k = q.shape[1]
    zero = vertex.zero
    psi = torch.full_like(vertex.resid, tau)
    psi[vertex.resid < 0] = tau - 1.0
    psi[zero] = 0.0
    rows = q[zero]
    d = dual[zero] - (1.0 - tau)
    defect = -(q.T @ psi) - rows.T @ d
    factor, info = torch.linalg.cholesky_ex(rows.T @ rows)
    if int(info) != 0:
        return None
    d = d + rows @ torch.cholesky_solve(defect[:, None], factor)[:, 0]
    if not bool(torch.isfinite(d).all()):
        return None
    if float(torch.maximum(d - tau, (tau - 1.0) - d).max()) > _DUAL_TOL:
        return None
    inside = rows[torch.minimum(tau - d, d - (tau - 1.0)) > _INTERIOR]
    if inside.shape[0] < k:
        return False
    spectrum = torch.linalg.eigvalsh(inside.T @ inside)
    return bool(spectrum[0] > _INDEPENDENT * spectrum[-1])


# ---- interior point --------------------------------------------------------------------


def _ratio(values: Tensor, direction: Tensor) -> float:
    """Largest step t with values + t direction >= 0 (inf when the direction is inward)."""
    ratios = torch.where(direction < 0, -values / direction, torch.full_like(values, math.inf))
    return float(ratios.min())


@torch.no_grad()
def solve(problem: QuantileProblem, tau: float, *, max_iterations: int = _MAX_ITERATIONS,
          max_pivots: int | None = None) -> QuantileFit:
    """The tau-th regression quantile of a prepared problem (see the module docstring)."""
    tau = _check_tau(tau)
    q, ys = problem.q, problem.ys
    n, k = q.shape
    max_pivots = 200 + 20 * k if max_pivots is None else max_pivots
    spread = float(ys.abs().sum())
    if spread <= _EXACT_FIT * problem.level or spread == 0.0:
        origin = torch.zeros(k, dtype=torch.float64)
        return _finish(problem, tau, origin, torch.zeros_like(ys), _basis_near(problem, origin),
                       iterations=0, pivots=0, gap=0.0, unique=True, exact_fit=True)
    # Dual variables a in (0, 1) with slacks s = 1 - a, multipliers z, w of the two bounds
    # and the free vector v = -gamma; z - w = Q gamma - y holds throughout. The start is
    # the least-squares fit (gamma = 0 for the residualized outcome).
    a = torch.full((n,), 1.0 - tau, dtype=torch.float64)
    s = torch.full((n,), tau, dtype=torch.float64)
    v = torch.zeros(k, dtype=torch.float64)
    start = -ys
    z = start.clamp_min(0.0)
    w = (-start).clamp_min(0.0)
    floor = 1e-3 * spread / n
    flat = start.abs() < floor
    z[flat] += floor
    w[flat] += floor
    vertex: _Vertex | None = None
    certified: bool | None = None       # a tied vertex certified by the interior dual point
    iterations, gap = 0, math.inf
    for iterations in range(1, max_iterations + 1):
        d = 1.0 / (z / a + w / s)
        rr = z - w
        gram = q.T @ (q * d[:, None])
        factor, info = torch.linalg.cholesky_ex((gram + gram.T) / 2)
        if int(info) != 0 or not bool(torch.isfinite(factor).all()):
            iterations -= 1
            break
        dv = torch.cholesky_solve((q.T @ (d * rr))[:, None], factor)[:, 0]
        da = d * (q @ dv - rr)
        dz = -z * (1.0 + da / a)
        dw = -w * (1.0 - da / s)
        primal = min(1.0, _STEP * min(_ratio(a, da), _ratio(s, -da)))
        dual = min(1.0, _STEP * min(_ratio(z, dz), _ratio(w, dw)))
        if min(primal, dual) < 1.0:
            # Mehrotra corrector: centring parameter from the predicted gap, second-order
            # terms da dz and ds dw of the affine step.
            mu = float(z @ a + w @ s)
            ahead = float((z + dual * dz) @ (a + primal * da) + (w + dual * dw) @ (s - primal * da))
            mu = mu * (ahead / mu) ** 3 / (2 * n)
            lower = (mu - da * dz) / a          # (mu - da dz) / a
            upper = (mu + da * dw) / s          # (mu - ds dw) / s with ds = -da
            xi = lower - upper
            dv = torch.cholesky_solve((q.T @ (d * (rr - xi)))[:, None], factor)[:, 0]
            da = d * (q @ dv + xi - rr)
            dz = lower - z - z * da / a
            dw = upper - w + w * da / s
            primal = min(1.0, _STEP * min(_ratio(a, da), _ratio(s, -da)))
            dual = min(1.0, _STEP * min(_ratio(z, dz), _ratio(w, dw)))
        a += primal * da
        s -= primal * da
        z += dual * dz
        w += dual * dw
        objective = float(check_loss(w - z, tau))
        ratio = float(z @ a + w @ s) / max(objective, 1e-300)
        if not math.isfinite(ratio) or not bool(torch.isfinite(dv).all()):
            iterations -= 1
            break                       # keep the last finite iterate v
        v += dual * dv
        gap = ratio
        if gap < _VERTEX_GAP:
            vertex = _vertex(problem, tau, _basis_near(problem, -v))
            if vertex.violation <= _DUAL_TOL:
                break
            certified = _tied_certificate(problem, tau, vertex, a)
            if certified is not None:
                break
        if gap < _GAP:
            break
    if vertex is None:
        vertex = _vertex(problem, tau, _basis_near(problem, -v))
    pivots = 0
    try:
        while certified is None and vertex.violation > _DUAL_TOL:
            if pivots == max_pivots:
                raise KernelError(
                    "nonconvergence",
                    f"Quantile regression did not reach an optimal vertex in {max_pivots} "
                    "simplex steps. Either the problem is extremely tied (a discrete outcome "
                    "with very many indicator regressors) or the variation of the outcome is "
                    "at the limit of double precision relative to its level; rescale or shift "
                    "the outcome, or reduce the number of indicator variables.")
            pivots += 1
            basis, virtual = _pivot(problem, tau, vertex)
            # virtual is None after a real step (a new vertex); otherwise the pivot was
            # degenerate and the vertex, with its set of tied rows, is unchanged.
            vertex = _vertex(problem, tau, basis, virtual,
                             previous=None if virtual is None else vertex)
            if virtual is None and vertex.violation > _DUAL_TOL:
                certified = _tied_certificate(problem, tau, vertex, a)
        unique = _unique(problem, tau, vertex) if certified is None else certified
    except RuntimeError as exc:
        raise KernelError("numerical_failure",
                          f"A quantile-regression basis could not be solved: {exc}") from exc
    return _finish(problem, tau, vertex.gamma, vertex.resid, vertex.basis, iterations=iterations,
                   pivots=pivots, gap=gap, unique=unique, exact_fit=False)


def _unique(problem: QuantileProblem, tau: float, vertex: _Vertex) -> bool:
    """Whether the certified vertex is the only minimizer (never True for a non-unique one).

    The directional derivative of the objective is a sum of nonnegative terms, one per
    zero residual; the term of basis row j is strictly positive in every direction that
    moves it when lambda_j lies strictly inside (tau - 1, tau).  All multipliers strictly
    inside: unique.  One multiplier on a bound: the objective can only be flat along that
    row's edge, and it is not if a tied row crosses zero there (its term is |g_i| > 0).
    Several multipliers on bounds: reported as not unique (it may still be).
    """
    lam = vertex.lam
    bound = torch.minimum(tau - lam, lam - (tau - 1.0)) <= _DUAL_TOL
    count = int(bound.sum())
    if count == 0:
        return True
    if count > 1 or not vertex.ties:
        return False
    j = int(bound.nonzero()[0])
    q = problem.q
    unit = torch.zeros(q.shape[1], dtype=q.dtype)
    unit[j] = 1.0 if float(lam[j]) < tau - 0.5 else -1.0
    g = q @ _basis_solve(vertex.factor, unit)
    g[vertex.basis] = 0.0
    crossing = vertex.zero & (vertex.virtual * g > 0)
    return bool((g[crossing].abs() > 1e-9 * g.abs().max()).any())


def _finish(problem: QuantileProblem, tau: float, gamma: Tensor, resid: Tensor, basis: Tensor, *,
            iterations: int, pivots: int, gap: float, unique: bool, exact_fit: bool) -> QuantileFit:
    objective = float(check_loss(resid, tau))
    if problem.weights is not None:
        resid = resid / problem.weights
    return QuantileFit(beta=problem.coefficients(problem.ols + gamma), resid=resid, basis=basis,
                       objective=objective, iterations=iterations, pivots=pivots, gap=gap,
                       unique=unique, exact_fit=exact_fit)


def quantile_regression(x: Tensor, y: Tensor, tau: float,
                        weights: Tensor | None = None) -> QuantileFit:
    """Minimize sum_i w_i rho_tau(y_i - x_i'b): ``solve(prepare(x, y, weights), tau)``."""
    return solve(prepare(x, y, weights), tau)


# ---- empirical quantiles ------------------------------------------------------------------


def _sorted_cumulative(values: Tensor, weights: Tensor | None) -> tuple[Tensor, Tensor]:
    ordered, order = torch.sort(values)
    if weights is None:
        cumulative = torch.arange(1, values.numel() + 1, dtype=torch.float64)
    else:
        cumulative = torch.cumsum(weights[order], dim=0)
    return ordered, cumulative


@torch.no_grad()
def raw_quantile(values: Tensor, tau: float, weights: Tensor | None = None) -> float:
    """The "raw" tau-th quantile qreg prints as "(about ...)": order statistic int(tau (N + 1)).

    Stata reports the check loss of the outcome about the order statistic number
    floor(tau (N + 1)) (at least the first), which is a minimizer of the check loss when
    tau N is an integer and can be the neighbour of the minimizer otherwise (auto.dta,
    tau = 0.25, N = 74: the 18th price 4187, although the 19th minimizes the loss).
    With weights (cumulative weights W_(i), total N) this is the first sorted value with
    W_(i) > tau (N + 1) - 1, which equals the rule above on frequency-expanded data.
    """
    ordered, cumulative = _sorted_cumulative(values, weights)
    total = float(cumulative[-1])
    target = tau * (total + 1.0) - 1.0 + 8 * _EPS * (total + 1.0)
    index = int(torch.searchsorted(cumulative, torch.tensor(target, dtype=torch.float64),
                                   right=True))
    return float(ordered[min(index, ordered.numel() - 1)])


@torch.no_grad()
def stata_quantile(values: Tensor, p: float, weights: Tensor | None = None) -> float:
    """The p-th quantile in the definition of Stata's ``summarize, detail`` / ``_pctile``.

    With sorted values x_(i), cumulative weights W_(i) and P = p W_(n): the first i with
    W_(i) > P gives x_(i), unless W_(i-1) = P, which gives (x_(i-1) + x_(i)) / 2.
    """
    ordered, cumulative = _sorted_cumulative(values, weights)
    total = float(cumulative[-1])
    target = p * total
    slack = 8 * _EPS * total
    index = int(torch.searchsorted(cumulative, torch.tensor(target + slack, dtype=torch.float64),
                                   right=True))
    index = min(index, ordered.numel() - 1)
    if index > 0 and abs(float(cumulative[index - 1]) - target) <= slack:
        return float(ordered[index - 1] + ordered[index]) / 2
    return float(ordered[index])


# ---- kernels and bandwidths of Stata's qreg --------------------------------------------------


def kernel_values(u: Tensor, kernel: str) -> Tensor:
    """K(u) for the kernels of Stata's ``kdensity`` (each integrates to one).

    epanechnikov  3/4 (1 - u^2/5) / sqrt(5)             |u| < sqrt(5)
    epan2         3/4 (1 - u^2)                         |u| < 1
    biweight      15/16 (1 - u^2)^2                     |u| < 1
    cosine        1 + cos(2 pi u)                       |u| < 1/2
    gaussian      exp(-u^2 / 2) / sqrt(2 pi)
    parzen        4/3 - 8 u^2 + 8 |u|^3                 |u| <= 1/2
                  8 (1 - |u|)^3 / 3                     1/2 < |u| <= 1
    rectangle     1/2                                   |u| < 1
    triangle      1 - |u|                               |u| < 1
    """
    size = u.abs()
    zero = torch.zeros_like(u)
    if kernel == "epanechnikov":
        root = math.sqrt(5.0)
        return torch.where(size < root, 0.75 * (1 - u.square() / 5) / root, zero)
    if kernel == "epan2":
        return torch.where(size < 1, 0.75 * (1 - u.square()), zero)
    if kernel == "biweight":
        return torch.where(size < 1, 0.9375 * (1 - u.square()).square(), zero)
    if kernel == "cosine":
        return torch.where(size < 0.5, 1 + torch.cos(2 * math.pi * u), zero)
    if kernel == "gaussian":
        return torch.exp(-0.5 * u.square()) / math.sqrt(2 * math.pi)
    if kernel == "parzen":
        inner = 4.0 / 3.0 - 8 * u.square() + 8 * size ** 3
        outer = 8 * (1 - size) ** 3 / 3
        return torch.where(size <= 0.5, inner, torch.where(size <= 1, outer, zero))
    if kernel == "rectangle":
        return torch.where(size < 1, torch.full_like(u, 0.5), zero)
    if kernel == "triangle":
        return torch.where(size < 1, 1 - size, zero)
    raise KernelError("unsupported_kernel", f"Unknown kernel '{kernel}'. Choose one of: "
                      f"{', '.join(KERNELS)}.")


def bandwidth(tau: float, n: int, method: str = "hsheather", alpha: float = 0.05) -> float:
    """Bandwidth h_n (on the probability scale) of the sparsity estimators of qreg.

    With z = Phi^-1(tau), phi the normal density and z_a = Phi^-1(1 - alpha/2):

    hsheather    n^(-1/3) z_a^(2/3) [1.5 phi(z)^2 / (2 z^2 + 1)]^(1/3)    (Hall and Sheather 1988)
    bofinger     n^(-1/5) [4.5 phi(z)^4 / (2 z^2 + 1)^2]^(1/5)            (Bofinger 1975)
    chamberlain  z_a sqrt(tau (1 - tau) / n)                              (Chamberlain 1994)
    """
    tau = _check_tau(tau)
    if n < 1:
        raise KernelError("insufficient_observations", "A bandwidth needs observations.")
    z = normal_ppf(tau)
    density = normal_pdf(z)
    z_alpha = normal_ppf(1 - alpha / 2)
    if method == "hsheather":
        shape = (1.5 * density ** 2 / (2 * z * z + 1)) ** (1 / 3)
        return n ** (-1 / 3) * z_alpha ** (2 / 3) * shape
    if method == "bofinger":
        return n ** (-1 / 5) * (4.5 * density ** 4 / (2 * z * z + 1) ** 2) ** (1 / 5)
    if method == "chamberlain":
        return z_alpha * math.sqrt(tau * (1 - tau) / n)
    raise KernelError("unsupported_bandwidth", f"Unknown bandwidth '{method}'. Choose one of: "
                      f"{', '.join(BANDWIDTHS)}.")
