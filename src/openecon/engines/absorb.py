"""High-dimensional fixed-effect absorption without dummy variables.

Every fixed-effect dimension d has a dummy matrix D_d that is never formed. In the
weighted inner product <a, b> = sum_i w_i a_i b_i the projection on its column space is
the level mean, P_d x = mean_d(x)[codes_d], so M_d = I - P_d is the within transform and
costs one index_add_ (scatter) plus one index_select (gather): O(n) per column block.

One dimension is solved exactly by M_1. For two or more, the residual maker M of the
joint dummy space [D_1 ... D_D] is the limit of alternating projections (Halperin). The
symmetric Kaczmarz sweep

    T = M_1 M_2 ... M_D ... M_2 M_1

is self-adjoint with spectrum in [0, 1]; its fixed points are exactly range(M). Hence
A = I - T is positive semidefinite, A (M x) = 0, and the absorbed part z = x - M x is the
only solution of the consistent system A z = A x inside range(A). Conjugate gradients
(Hestenes-Stiefel) started at zero stay in that range, so they converge to it at the
sqrt(condition) rate instead of the plain sweep's linear rate; this is the acceleration
of Correia's reghdfe (2017, "Linear Models with High-Dimensional Fixed Effects").

Two details make a sweep cheaper and more accurate than the textbook form:

* The block is projected by M_1 once. Every later iterate, residual and search direction
  stays in range(M_1), so the leading M_1 of each sweep is the identity and is skipped
  (2D - 2 projections per sweep rather than 2D - 1; a third less work for two-way models).
  Rounding lets the residual drift out of range(M_1) by an amount that does not shrink
  with it, so it and the search direction are projected again every time the residual
  has dropped an order of magnitude (at most every eighth sweep). Without this, long
  runs on ill-conditioned designs lose digits and finally the symmetry of A.
* A u is accumulated as the sum of the level means that the sweep removes instead of
  u - T u, which would cancel catastrophically in poorly connected designs where T u ~ u.
  The same level means are the fixed-effect coefficients, so the estimates of the level
  effects come from the same recursion at O(levels) extra cost.

All columns advance together as one [n, m] block with per-column step lengths. A column
stops when its update is at most tol times the CURRENT norm of that column (the demeaned
variable, not the raw one, so a large mean cannot hide a poorly resolved within part), or
when its residual reaches machine precision relative to M_1 x. Finished columns are
removed from the working block so they cost nothing and cannot be polluted by noise steps.
An update is a lower bound of the remaining error: with a per-sweep contraction rate q the
error is about update / (1 - q). On sparse worker-firm style designs the accelerated
iterate ends within roughly 10 tol of the exact projection, while the plain one can be
hundreds of tol away after thousands of sweeps.

Degrees of freedom follow reghdfe: redundant coefficients are counted from the connected
components ("mobility groups", Abowd-Creecy-Kramarz 2002) of the bipartite level graph,
found by min-label hooking with pointer jumping in O(log) vectorized rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import operator

import torch
from torch import Tensor

from .contracts import KernelError


_EPS = torch.finfo(torch.float64).eps
_TINY = torch.finfo(torch.float64).tiny
# A residual below this multiple of the working norm cannot be told apart from rounding.
_FLOOR = 8 * _EPS
# The conjugate-gradient vectors are projected on range(M_1) again once the squared
# residual has dropped by this factor, but not more often than every _SPACING sweeps.
_REPROJECT = 1e-2
_SPACING = 8


@dataclass
class AbsorbResult:
    values: Tensor
    iterations: int
    converged: bool
    max_update: float
    method: str


@dataclass
class AbsorbedDF:
    levels: list[int]
    redundant: list[int]
    nested: list[bool]
    total: int


def _block(values: Tensor) -> Tensor:
    if not isinstance(values, Tensor) or values.dtype != torch.float64 or values.ndim not in (1, 2):
        raise KernelError("invalid_values", "Expected a float64 tensor of shape [n] or [n, m].")
    return values[:, None] if values.ndim == 1 else values


def _dimensions(dimensions: list[tuple[Tensor, int]], n: int | None = None,
                device: torch.device | None = None) -> tuple[list[tuple[Tensor, int]], int]:
    checked = []
    for codes, levels in dimensions:
        try:
            if isinstance(levels, bool):
                raise TypeError
            levels = operator.index(levels)
        except TypeError as exc:
            raise KernelError("invalid_codes", "The number of levels must be an integer.") from exc
        if not isinstance(codes, Tensor) or codes.dtype != torch.int64 or codes.ndim != 1:
            raise KernelError("invalid_codes", "Fixed-effect codes must be an int64 tensor [n].")
        n = len(codes) if n is None else n
        device = codes.device if device is None else device
        if len(codes) != n or codes.device != device:
            raise KernelError("invalid_codes", "Each dimension needs one code per observation "
                              "on the device of the data.")
        if n:
            low, high = torch.aminmax(codes)
            if int(low) < 0 or int(high) >= levels:
                raise KernelError("invalid_codes", "Fixed-effect codes must lie in [0, n_levels).")
        checked.append((codes, levels))
    return checked, 0 if n is None else n


def _weights(weights: Tensor | None, block: Tensor) -> Tensor | None:
    if weights is None:
        return None
    if (not isinstance(weights, Tensor) or weights.dtype != torch.float64
            or weights.shape != block.shape[:1] or weights.device != block.device):
        raise KernelError("invalid_weights",
                          "Weights must be a float64 tensor with one entry per row.")
    if len(weights) and not (bool(torch.isfinite(weights).all()) and float(weights.min()) >= 0
                             and float(weights.sum()) > 0):
        raise KernelError("invalid_weights",
                          "Weights must be finite, non-negative and not all zero.")
    return weights


def _controls(tol: float, max_iter: int) -> tuple[float, int]:
    """Validated (tol, max_iter) as Python float and int (numpy scalars are accepted)."""
    try:
        if isinstance(max_iter, bool):
            raise TypeError
        max_iter = operator.index(max_iter)
    except TypeError as exc:
        raise KernelError("invalid_solver_options", "max_iter must be a positive integer.") from exc
    if max_iter < 1:
        raise KernelError("invalid_solver_options", "max_iter must be a positive integer.")
    if (isinstance(tol, bool) or not isinstance(tol, numbers.Real) or not math.isfinite(tol)
            or tol <= 0):
        raise KernelError("invalid_solver_options", "tol must be positive and finite.")
    return float(tol), max_iter


def _column_scaling(block: Tensor) -> tuple[Tensor, Tensor]:
    """Powers of two (scale, inverse) as [m] that bring the largest entry of each column to
    [0.5, 1).

    Scaling by a power of two is exact, so in the ordinary range the recursion is bit for
    bit the same. It keeps the sums of squares of the recursion inside the float64 range:
    a column around 1e-160 has a squared norm that underflows to 0, which would satisfy
    the machine-precision exit after the first sweep and return the within transform of
    the first dimension alone, and one around 1e+160 overflows. The exponent is clamped so
    that the scale itself stays normal; a column of zeros, or with non-finite entries
    (rejected later), is left alone.
    """
    low, high = torch.aminmax(block, dim=0)
    magnitude = torch.maximum(low.abs(), high.abs())
    exponent = torch.frexp(magnitude)[1].to(torch.int64)
    exponent = torch.where(torch.isfinite(magnitude) & (magnitude > 0), exponent, 0)
    exponent.clamp_(-1000, 1000)
    # Build 2 ** k from its bit pattern (biased exponent field) rather than through pow.
    scale = ((1023 - exponent) << 52).view(torch.float64)
    inverse = ((1023 + exponent) << 52).view(torch.float64)
    return scale, inverse


def _inverse_totals(codes: Tensor, levels: int, weights: Tensor | None) -> Tensor:
    """1 / (sum of weights in the level) as [levels, 1]; zero for empty levels."""
    totals = torch.bincount(codes, weights=weights, minlength=levels).to(torch.float64)
    return torch.where(totals > 0, 1 / totals, 0.0)[:, None]


def _means(block: Tensor, codes: Tensor, inverse: Tensor, weights: Tensor | None,
           scratch: Tensor | None) -> Tensor:
    source = block if weights is None else torch.mul(block, weights[:, None], out=scratch)
    sums = torch.zeros((len(inverse), block.shape[1]), dtype=block.dtype, device=block.device)
    return sums.index_add_(0, codes, source).mul_(inverse)


def _dots(a: Tensor, b: Tensor, weights: Tensor | None, scratch: Tensor) -> Tensor:
    """Column-wise weighted inner products sum_i w_i a_ij b_ij as [m]."""
    torch.mul(a, b, out=scratch)
    if weights is not None:
        scratch.mul_(weights[:, None])
    return scratch.sum(dim=0)


def _sweep(u: Tensor, v: Tensor, x: Tensor | None, scratch: Tensor,
           dims: list[tuple[Tensor, int]], inverses: list[Tensor], weights: Tensor | None,
           track: bool) -> list[Tensor] | None:
    """v <- (I - M_1 M_2 ... M_D ... M_2) u for u in range(M_1); u is left untouched.

    v is built as the sum of the removed level means g_j = P_{d_j} x_{j-1},
    x_j = x_{j-1} - g_j, x_0 = u. Because P_1 u = 0 the closing projection is
    P_1 x = -P_1 (g_1 + ... ), which needs v only. With track=True the level
    coefficients c_d with v = sum_d c_d[codes_d] are returned.
    """
    count = len(dims)
    order = list(range(1, count)) + list(range(count - 2, 0, -1))
    coefficients: list = [None] * count
    work = u
    for position, d in enumerate(order):
        codes = dims[d][0]
        means = _means(work, codes, inverses[d], weights, scratch)
        if track:
            coefficients[d] = means if coefficients[d] is None else coefficients[d] + means
        if position == 0:
            torch.index_select(means, 0, codes, out=v)
            if len(order) > 1:
                work = torch.sub(u, v, out=x)
        else:
            v.add_(torch.index_select(means, 0, codes, out=scratch))
            if position < len(order) - 1:
                work.sub_(scratch)
    codes = dims[0][0]
    means = _means(v, codes, inverses[0], weights, scratch)
    v.sub_(torch.index_select(means, 0, codes, out=scratch))
    if not track:
        return None
    coefficients[0] = means.neg_()
    return coefficients


def _absorb(block: Tensor, dims: list[tuple[Tensor, int]], weights: Tensor | None, tol: float,
            max_iter: int, accelerate: bool, track: bool,
            ) -> tuple[Tensor, list[Tensor] | None, int, float, str]:
    """Demeaned copy of block [n, m], level effects (if tracked), sweeps, last update, method."""
    n, m = block.shape
    # Row-major working memory whatever the strides of the input (clone alone keeps them).
    out = block.clone(memory_format=torch.contiguous_format)
    effects = ([torch.zeros((levels, m), dtype=block.dtype, device=block.device)
                for _, levels in dims] if track else None)
    if not dims or n == 0 or m == 0:
        return out, effects, 0, 0.0, "none"
    count = len(dims)
    inverses = [_inverse_totals(codes, levels, weights) for codes, levels in dims]
    scratch = torch.empty_like(out)
    if count > 1:
        # The recursion below works on sums of squares: run it on exactly rescaled columns
        # (powers of two) and undo the scaling on the way out. The one-pass within
        # transform needs none.
        scale, unscale = _column_scaling(out)
        out.mul_(scale)
    codes = dims[0][0]
    means = _means(out, codes, inverses[0], weights, scratch)
    # Any non-finite entry poisons the mean of its level.
    if not bool(torch.isfinite(means).all()):
        raise KernelError("non_finite_values", "Values to absorb must be finite.")
    out.sub_(torch.index_select(means, 0, codes, out=scratch))
    if track:
        effects[0] += means
    if count == 1:
        return out, effects, 1, 0.0, "within"

    y, active = out, torch.arange(m, device=block.device)
    v = torch.empty_like(out)
    x = torch.empty_like(out) if count > 2 else None
    r = u = cr = cu = ssr = peak = None
    # ||y|| only decreases along the iteration, so the stored norm is an upper bound that
    # is refreshed just when some column passes the test against it.
    norm = _dots(y, y, weights, scratch).sqrt_()
    if not bool(torch.isfinite(norm).all()):
        raise KernelError("non_finite_values", "Squares of the values to absorb must stay "
                          "inside the float64 range; rescale the input.")
    # The recursion works at the scale of M_1 x: a residual below a few eps of that
    # norm is rounding, whatever the tolerance.
    start = norm.clamp_min(_TINY)
    floor = _FLOOR * norm
    sweeps, cleaned, worst, pending = 0, 1, 0.0, math.inf
    while True:
        if sweeps >= max_iter:
            raise KernelError(
                "absorption_nonconvergence",
                f"Fixed-effect absorption did not converge in {max_iter} sweeps "
                f"(largest relative update {pending:.3e}, tolerance {tol:.3e}).")
        cv = _sweep(y if u is None else u, v, x, scratch, dims, inverses, weights, track)
        sweeps += 1
        stepped = True
        if not accelerate:
            # Plain alternating projections: y <- T y, and the update is A y itself.
            y.sub_(v)
            residual = change = _dots(v, v, weights, scratch).sqrt_()
            if track:
                for d in range(count):
                    effects[d].index_add_(1, active, cv[d])
        elif u is None:
            # First sweep: residual of A z = A y at z = 0. No step has been taken yet, so
            # only the machine-precision exit may fire.
            r, v = v, torch.empty_like(v)
            u, cr = r.clone(), cv
            cu = [c.clone() for c in cv] if track else None
            peak = ssr = _dots(r, r, weights, scratch)
            residual = change = ssr.sqrt()
            stepped = False
        else:
            # <u, A u> > 0 on range(A). A non-positive value means the direction has lost
            # that property to rounding: take no step and restart from the residual.
            curvature = _dots(u, v, weights, scratch)
            alpha = torch.where(curvature > 0, ssr / curvature, 0.0)
            y.addcmul_(u, alpha, value=-1)
            r.addcmul_(v, alpha, value=-1)
            previous, ssr = ssr, _dots(r, r, weights, scratch)
            residual = change = ssr.sqrt()
            # The update is the step just taken or, if larger, what one more sweep would
            # move (the residual). The step norm costs a pass over the block, so it is
            # only evaluated once some residual is small enough for it to decide.
            if bool((residual <= tol * norm).any()):
                step = _dots(u, u, weights, scratch).sqrt_().mul_(alpha)
                change = torch.maximum(step, residual)
            beta = torch.where(curvature > 0, ssr / previous, 0.0)
            torch.addcmul(r, u, beta, out=u)
            if track:
                for d in range(count):
                    effects[d].index_add_(1, active, cu[d] * alpha)
                    cr[d].addcmul_(cv[d], alpha, value=-1)
                    cu[d] = torch.addcmul(cr[d], cu[d], beta)
            # Each update leaves a component outside range(M_1) of rounding size relative
            # to the residual of THAT step. The shortened sweep never removes it, so as
            # the residual falls it stops being negligible: steps of length alpha carry
            # it into y (the accuracy loss grows with 1 / smallest eigenvalue) and it
            # finally breaks the symmetry of A. Project r and u again whenever the
            # residual has dropped an order of magnitude. Fast runs, where the drift is
            # harmless, are spared by the minimum spacing.
            peak = torch.maximum(peak, ssr)
            if sweeps - cleaned >= _SPACING and bool((ssr < _REPROJECT * peak).any()):
                codes = dims[0][0]
                for index, vector in enumerate((r, u)):
                    means = _means(vector, codes, inverses[0], weights, scratch)
                    vector.sub_(torch.index_select(means, 0, codes, out=scratch))
                    if track:
                        (cr, cu)[index][0] -= means
                cleaned, peak = sweeps, ssr
        exact = residual <= floor
        finished = exact.clone()
        if stepped and bool((change <= tol * norm).any()):
            norm = _dots(y, y, weights, scratch).sqrt_()
            finished |= change <= tol * norm
        regular = change / norm.clamp_min(_TINY)
        relative = torch.where(exact & (regular > tol), residual / start, regular)
        if not bool(finished.any()):
            pending = float(relative.max())
            continue
        worst = max(worst, float(relative[finished].max()))
        if y is not out:
            out[:, active[finished]] = y[:, finished]
        keep = ~finished
        if not bool(keep.any()):
            break
        pending = float(relative[keep].max())
        # Freeze the finished columns: continue on a contiguous block of the rest.
        active, norm, start, floor = active[keep], norm[keep], start[keep], floor[keep]
        y = y[:, keep]
        v, scratch = torch.empty_like(y), torch.empty_like(y)
        x = None if x is None else torch.empty_like(y)
        if u is not None:
            r, u, ssr, peak = r[:, keep], u[:, keep], ssr[keep], peak[keep]
            if track:
                cr, cu = [c[:, keep] for c in cr], [c[:, keep] for c in cu]
    out.mul_(unscale)
    if track:
        for effect in effects:
            effect.mul_(unscale)
    method = "symmetric_kaczmarz_cg" if accelerate else "symmetric_kaczmarz"
    return out, effects, sweeps, worst, method


@torch.no_grad()
def group_means(values: Tensor, codes: Tensor, n_levels: int,
                weights: Tensor | None = None) -> Tensor:
    """Weighted level means of each column, [n_levels, m] (or [n_levels] for a vector).

        mean[g, j] = sum_{i: codes_i = g} w_i x_ij / sum_{i: codes_i = g} w_i

    with w = 1 when weights is None. Levels without observations (or with zero total
    weight) return 0. One index_add_ pass, O(n m); Stata: `egen ... = mean(x), by(g)`
    with analytic/frequency weights.
    """
    block = _block(values)
    (dimension,), _ = _dimensions([(codes, n_levels)], len(block), block.device)
    weights = _weights(weights, block)
    inverse = _inverse_totals(dimension[0], dimension[1], weights)
    means = _means(block, dimension[0], inverse, weights, None)
    return means[:, 0] if values.ndim == 1 else means


@torch.no_grad()
def demean(values: Tensor, dimensions: list[tuple[Tensor, int]], weights: Tensor | None = None,
           *, tol: float = 1e-10, max_iter: int = 10_000, accelerate: bool = True) -> AbsorbResult:
    """Residuals of every column from a weighted regression on all fixed-effect dummies.

    Returns M x with M = I - D (D' W D)^- D' W, D = [D_1 ... D_D] the (never formed) dummy
    matrices of the dimensions [(int64 codes [n], n_levels), ...] and W = diag(weights):
    the projection of each column off the dummy span in <a, b> = sum_i w_i a_i b_i. This
    is the partialling-out step of Stata's xtreg, fe / areg (one dimension) and reghdfe
    (any number); by Frisch-Waugh-Lovell, OLS on the demeaned columns reproduces the
    dummy-variable coefficients.

    * 0 dimensions: a copy (iterations 0, method "none").
    * 1 dimension: the exact within transform x - mean_g(x) (iterations 1, "within").
    * 2+ dimensions: symmetric Kaczmarz sweeps T = M_1 M_2 ... M_D ... M_2 M_1 of the
      method of alternating projections, accelerated by conjugate gradients on
      (I - T) z = (I - T) x as in reghdfe ("symmetric_kaczmarz_cg"), or the plain
      fixed-point iteration x <- T x with accelerate=False ("symmetric_kaczmarz").
      `iterations` counts sweeps (applications of T).

    Convergence is tested per column: a column is finished when
    ||update|| <= tol * ||column||, both in the weighted norm, where ||column|| is the
    current (demeaned) column - never larger than the input norm, so the test is at
    least as strict as one relative to the raw column - and the update is the conjugate
    gradient step (or, if larger, the change one more sweep would make); without
    acceleration it is the change made by the sweep. A column whose residual x - T x is
    below 8 eps times the norm of its within transform M_1 x is finished at machine
    precision (this is how columns that lie in the dummy span, whose demeaned norm is
    pure rounding, stop; `max_update` then reports that ratio).
    Finished columns are frozen. `max_update` is the largest final relative update.
    The update bounds the remaining error from below: in poorly connected designs the
    error is about update / (1 - rate), i.e. up to ~10 tol with acceleration and
    hundreds of tol for the slowly contracting plain iteration. Not converging in
    max_iter sweeps raises KernelError("absorption_nonconvergence").

    The input is not modified; the result has the shape of `values` ([n] or [n, m]).
    Columns are rescaled by exact powers of two internally, so finite values of any
    magnitude are absorbed without the squared norms under- or overflowing.
    """
    block = _block(values)
    dims, _ = _dimensions(dimensions, len(block), block.device)
    weights = _weights(weights, block)
    tol, max_iter = _controls(tol, max_iter)
    out, _, sweeps, update, method = _absorb(block, dims, weights, tol, max_iter, accelerate, False)
    return AbsorbResult(out.reshape(values.shape), sweeps, True, update, method)


@torch.no_grad()
def fixed_effect_estimates(residual_with_fe: Tensor, dimensions: list[tuple[Tensor, int]],
                           weights: Tensor | None = None, *, tol: float = 1e-10,
                           max_iter: int = 10_000) -> list[Tensor]:
    """Level effects alpha_d with sum_d alpha_d[codes_d] = P x, the fitted fixed effects.

    x is usually y - X b (the residual that still contains the fixed effects) and P = I - M
    the weighted projection on the dummy span, so x - sum_d alpha_d[codes_d] is the
    regression residual. This is Stata's `predict, u` after xtreg, fe / `predict, d`
    after areg / the saved fixed effects of reghdfe. One tensor [n_levels_d] per
    dimension ([n_levels_d, m] for a block of m vectors); empty levels get 0.

    For one dimension alpha is the level mean. For more, the decomposition is not
    unique; the one returned is the limit of backfitting (Gauss-Seidel on the normal
    equations D' W D alpha = D' W x) started at alpha = 0 with the symmetric sweep
    1, 2, ..., D, ..., 2, 1. It is reached by the same conjugate-gradient recursion as
    demean(): the level means removed by each sweep are accumulated with the step
    lengths, and both iterations live in the same preconditioned Krylov subspace, where
    the solution is unique. For two dimensions this means the effects of the second
    dimension have weighted mean zero over the observations of every connected
    component, and the first dimension carries the level (constant included).
    """
    block = _block(residual_with_fe)
    dims, _ = _dimensions(dimensions, len(block), block.device)
    weights = _weights(weights, block)
    tol, max_iter = _controls(tol, max_iter)
    _, effects, _, _, _ = _absorb(block, dims, weights, tol, max_iter, True, True)
    return [effect[:, 0] if residual_with_fe.ndim == 1 else effect for effect in effects]


@torch.no_grad()
def singleton_mask(dimensions: list[tuple[Tensor, int]]) -> Tensor:
    """Boolean keep-mask [n] after iteratively dropping singleton observations.

    An observation is a singleton when it is the only one left in its level of some
    dimension: its dummy fits it perfectly, it carries no information about the slopes
    and it biases cluster-robust standard errors downward (Correia 2015). Dropping one
    can leave a level of ANOTHER dimension with a single observation, so the rule is
    applied until nothing changes - reghdfe's default.

    The first round is one vectorized pass, O(n D). Later rounds do not rescan the data:
    level counts are decremented for the dropped rows, and only levels whose count just
    reached one are opened through a sorted level -> observation index. Every level is
    opened at most once, so the total cost is O(n D) plus one sort per dimension however
    many rounds the cascade takes.
    """
    dims, n = _dimensions(dimensions)
    if not dims:
        raise KernelError("invalid_dimensions", "singleton_mask needs at least one dimension.")
    device = dims[0][0].device
    keep = torch.ones(n, dtype=torch.bool, device=device)
    counts = [torch.bincount(codes, minlength=levels) for codes, levels in dims]
    single = torch.zeros(n, dtype=torch.bool, device=device)
    for (codes, _), count in zip(dims, counts):
        single |= count[codes] == 1
    drop = single.nonzero()[:, 0]
    if not len(drop):
        return keep
    sizes = [count.clone() for count in counts]
    starts = [torch.cumsum(size, 0) - size for size in sizes]
    orders = [torch.argsort(codes) for codes, _ in dims]
    while len(drop):
        keep[drop] = False
        survivors = []
        for (codes, _), count, size, start, order in zip(dims, counts, sizes, starts, orders):
            touched = codes[drop]
            count.index_add_(0, touched, torch.full_like(touched, -1))
            opened = torch.unique(touched[count[touched] == 1])
            if not len(opened):
                continue
            # Ragged gather of all original members of the opened levels.
            length = size[opened]
            ends = torch.cumsum(length, 0)
            offset = torch.arange(int(ends[-1]), device=device)
            offset -= torch.repeat_interleave(ends - length, length)
            members = order[torch.repeat_interleave(start[opened], length) + offset]
            survivors.append(members[keep[members]])
        drop = torch.unique(torch.cat(survivors)) if survivors else drop[:0]
    return keep


@torch.no_grad()
def relabel(codes: Tensor) -> tuple[Tensor, int]:
    """Dense relabeling of arbitrary int64 codes: (new_codes in 0..L-1, L).

    New labels follow the increasing order of the original values (new_i < new_j iff
    code_i < code_j), so the result is deterministic and independent of row order -
    the convention of Stata's `egen group()`. Use it after masking observations
    (e.g. singletons) so that every level is non-empty. When the code range is at most
    a few times n a presence table gives O(n + range) without sorting; otherwise the
    codes are sorted (torch.unique).
    """
    if not isinstance(codes, Tensor) or codes.dtype != torch.int64 or codes.ndim != 1:
        raise KernelError("invalid_codes", "Fixed-effect codes must be an int64 tensor [n].")
    n = len(codes)
    if n == 0:
        return codes.clone(), 0
    low, high = (int(bound) for bound in torch.aminmax(codes))
    span = high - low + 1
    if span <= 4 * n + 1024:
        shifted = codes - low
        present = torch.zeros(span, dtype=torch.bool, device=codes.device)
        present[shifted] = True
        lookup = torch.cumsum(present, 0) - 1
        return lookup[shifted], int(lookup[-1]) + 1
    levels, inverse = torch.unique(codes, return_inverse=True)
    return inverse, len(levels)


def _components(codes_a: Tensor, n_a: int, codes_b: Tensor, n_b: int) -> int:
    if not len(codes_a):
        return 0
    if n_a < n_b:
        codes_a, n_a, codes_b, n_b = codes_b, n_b, codes_a, n_a
    # Contract the larger side: every a-level links all its b-levels to its smallest one,
    # leaving a graph on the b-levels with the same components.
    anchor = torch.full((n_a,), n_b, dtype=torch.int64, device=codes_a.device)
    anchor.scatter_reduce_(0, codes_a, codes_b, "amin")
    left, right = codes_b, anchor[codes_a]
    identity = torch.arange(n_b, device=codes_a.device)
    parent = identity.clone()
    while True:
        cross = left != right
        if not bool(cross.any()):
            break
        left, right = left[cross], right[cross]
        # Both ends are roots. Hook each root under its smallest neighbouring root
        # (labels only decrease, so no cycles), then jump pointers to full compression.
        parent.scatter_reduce_(0, torch.maximum(left, right), torch.minimum(left, right), "amin")
        while True:
            jumped = parent[parent]
            if torch.equal(jumped, parent):
                break
            parent = jumped
        left, right = parent[left], parent[right]
    observed = torch.bincount(codes_b, minlength=n_b) > 0
    return int((observed & (parent == identity)).sum())


@torch.no_grad()
def connected_components(codes_a: Tensor, n_a: int, codes_b: Tensor, n_b: int) -> int:
    """Number of connected components of the bipartite graph between two level sets.

    Nodes are the OBSERVED levels of a and of b; a level of a is linked to a level of b
    when some observation has both. The components are the "mobility groups" of Abowd,
    Creecy and Kramarz (2002): within a component the two sets of effects are identified
    up to one constant, so

        rank([D_a D_b]) = levels_a + levels_b - components

    counting observed levels (levels without observations are not counted as components).

    Vectorized union-find: the side with more levels is contracted first (each of its
    levels ties all its neighbours to one anchor), then roots are hooked under their
    smallest neighbouring root and pointers are jumped to full compression. A root that
    does not hook is a local minimum whose neighbours all hook, so the number of live
    components at least halves every two rounds: O(log levels) rounds of O(n) work,
    with resolved edges discarded after each round.
    """
    (first, second), _ = _dimensions([(codes_a, n_a), (codes_b, n_b)])
    return _components(first[0], first[1], second[0], second[1])


@torch.no_grad()
def absorbed_degrees_of_freedom(dimensions: list[tuple[Tensor, int]],
                                cluster: tuple[Tensor, int] | None = None, *,
                                pairwise: bool = True) -> AbsorbedDF:
    """Degrees of freedom lost to the fixed effects, with reghdfe's accounting.

    total = sum_d (levels_d - redundant_d), where levels_d is the number of OBSERVED
    levels and redundant_d the dummy coefficients of dimension d that are not
    identified given the earlier dimensions:

    * first dimension: 0 (it also absorbs the constant);
    * second dimension: the number of connected components of the bipartite graph
      between dimensions one and two. This is exact:
      rank([D_1 D_2]) = levels_1 + levels_2 - components (Abowd-Creecy-Kramarz 2002);
    * third and later dimensions: no exact graph formula exists. With pairwise=True
      (reghdfe's default `dofadjustments(pairwise)`) redundant_d is the largest number
      of connected components between d and any earlier dimension; with pairwise=False
      it is 1 (reghdfe's `firstpair`). Both are LOWER bounds on the true redundancy, so
      `total` can overstate the lost degrees of freedom and inference is conservative.
      In a connected design both give 1.

    A dimension whose every level lies inside a single cluster of `cluster` is flagged
    nested and counted as fully redundant (redundant_d = levels_d): with cluster-robust
    inference those effects cost no degrees of freedom, the xtreg, fe / reghdfe
    convention. Nested dimensions are skipped when ranking the others as "first",
    "second", ... . If every dimension is nested, total is 0 and the constant they
    absorb is NOT counted: the caller adds it to the regressor count, as xtreg, fe does.
    """
    dims, n = _dimensions(dimensions)
    levels = [int((torch.bincount(codes, minlength=count) > 0).sum()) for codes, count in dims]
    nested = [False] * len(dims)
    if cluster is not None and dims:
        ((groups, _),), _ = _dimensions([cluster], n, dims[0][0].device)
        for index, (codes, count) in enumerate(dims):
            lowest = torch.full((count,), torch.iinfo(torch.int64).max, device=codes.device)
            highest = torch.full((count,), -1, device=codes.device)
            lowest.scatter_reduce_(0, codes, groups, "amin")
            highest.scatter_reduce_(0, codes, groups, "amax")
            nested[index] = bool((highest <= lowest).all())
    redundant, earlier = [], []
    for index, (codes, count) in enumerate(dims):
        if nested[index]:
            redundant.append(levels[index])
            continue
        if not earlier:
            redundant.append(0)
        elif len(earlier) > 1 and not pairwise:
            redundant.append(min(1, levels[index]))
        else:
            redundant.append(max(_components(*dims[other], codes, count) for other in earlier))
        earlier.append(index)
    return AbsorbedDF(levels, redundant, nested, sum(levels) - sum(redundant))
