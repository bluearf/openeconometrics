"""Sandwich "meat" builders shared by every robust covariance in OpenEconometrics.

A sandwich covariance is  V = B M B'  with bread B (for least squares (X'WX)^{-1},
for likelihood models the inverse information) and meat M, the estimated variance
of the summed score.  All builders take the n-by-k matrix S whose rows are the
per-observation scores s_i (for linear models x_i * u_i, already multiplied by any
weight) and return a symmetric k-by-k meat WITHOUT small-sample factors unless the
function says otherwise; callers apply factors (cluster_factor, n/(n-k), ...).

    White / HC        M = sum_i s_i s_i'
    one-way cluster   M = sum_g t_g t_g',  t_g = sum_{i in g} s_i
    multiway cluster  inclusion-exclusion over intersections of the cluster
                      dimensions (Cameron, Gelbach and Miller 2011)
    HAC               M = G_0 + sum_l w_l (G_l + G_l'),  G_l = sum_t s_t s_{t-l}'
    Driscoll-Kraay    HAC applied to the cross-sectional sums h_t = sum_i s_it

Cost.  Group sums use index_add_ (O(n k)).  The HAC double sum is never formed lag
by lag as k-by-k products: with the lag-weighted partner sum p_t = sum_l w_l s_{t-l}
(built in O(n L k)) the meat is S'S + S'P + P'S, one extra matrix product instead of
L of them.  With gaps or panels the rows are sorted once by the integer key
unit * stride + period.  Periods are distinct within a unit, so the row j positions
earlier lies at least j periods earlier: scanning offsets j = 1..L and weighting
each pair by the kernel at its actual period distance (zero across units or beyond
the bandwidth) reaches every contributing pair with L contiguous O(n k) updates,
whatever the input row order.  When more than _FFT_OFFSETS offsets would be needed
(the quadratic spectral kernel, whose support is unbounded, or very long bandwidths)
p is instead the convolution of S, laid out on the dense period grid, with the kernel
sequence, computed by one real FFT per column: O(k m log m), m ~ periods + reach,
instead of O(n k reach).  No n-by-n object is ever built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import KernelError
from .linalg import symmetrize

_EPS = torch.finfo(torch.float64).eps
_KERNELS = ("bartlett", "truncated", "parzen", "quadratic_spectral")
# Quadratic spectral kernel 3/a^2 (sin(a)/a - cos(a)) = sum_m (-1)^(m+1) c_m a^(2m-2) with
# c_m = 6m/(2m+1)!; nine terms are exact to double precision for a < 1.
_QS_SERIES = [6 * m / math.factorial(2 * m + 1) for m in range(1, 10)]
# Lag sums reaching further than this many periods use the FFT convolution, provided
# the dense period grid is at most _FFT_SPARSITY times the number of rows.
_FFT_OFFSETS = 64
_FFT_SPARSITY = 4
# Columns are transformed in blocks of about this many grid elements.
_FFT_ELEMENTS = 1 << 22


def _scores(scores: Tensor) -> tuple[int, int]:
    if not isinstance(scores, Tensor) or scores.dtype != torch.float64 or scores.ndim != 2:
        raise KernelError("invalid_scores", "Scores must be an n-by-k float64 tensor.")
    return scores.shape


def _codes(codes: Tensor, n: int, n_groups: int, what: str = "group codes") -> None:
    if not isinstance(codes, Tensor) or codes.dtype != torch.int64 or codes.shape != (n,):
        raise KernelError("invalid_clusters", f"The {what} must be an int64 vector, one per row.")
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
        raise KernelError("invalid_clusters", "The number of groups must be a positive integer.")
    if n and (int(codes.min()) < 0 or int(codes.max()) >= n_groups):
        raise KernelError("invalid_clusters", f"The {what} must lie in 0..n_groups-1.")


def _compact(codes: Tensor, n_groups: int) -> tuple[Tensor, int, int]:
    """(codes, range, number of nonempty groups) for group sums over validated codes.

    A declared range far above the row count (sparse identifiers passed as codes) is
    recoded first, so no temporary is ever larger than O(n).
    """
    if n_groups > 2 * codes.numel() + 64:
        cells, codes = torch.unique(codes, return_inverse=True)
        return codes, max(cells.numel(), 1), cells.numel()
    return codes, n_groups, int((torch.bincount(codes, minlength=n_groups) > 0).sum())


def _lags(lags: int, kernel: str) -> None:
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 0:
        raise KernelError("invalid_lags", "The number of lags must be a nonnegative integer.")
    if kernel not in _KERNELS:
        raise KernelError("unsupported_kernel", f"Unsupported HAC kernel: {kernel}. "
                          f"Choose one of {', '.join(_KERNELS)}.")


@torch.no_grad()
def group_sums(values: Tensor, codes: Tensor, n_groups: int) -> Tensor:
    """Sums by integer code: out[g] = sum_{i: codes_i = g} values_i.

    values is [n] or [n, k]; the result is [G] or [G, k]. One index_add_ pass, O(n k).
    """
    if not isinstance(values, Tensor) or values.ndim not in (1, 2):
        raise KernelError("invalid_scores", "group_sums expects an [n] or [n, k] tensor.")
    _codes(codes, values.shape[0], n_groups)
    out = torch.zeros((n_groups,) + values.shape[1:], dtype=values.dtype, device=values.device)
    return out.index_add_(0, codes, values)


@torch.no_grad()
def group_counts(codes: Tensor, n_groups: int) -> Tensor:
    """Number of rows per integer code, int64 [G] (zero for empty groups)."""
    _codes(codes, codes.numel() if isinstance(codes, Tensor) else 0, n_groups)
    return torch.bincount(codes, minlength=n_groups)


@torch.no_grad()
def hc_residuals(resid: Tensor, leverage: Tensor | None, kind: str) -> Tensor:
    """Residuals entering the heteroskedasticity-consistent meat sum_i x_i x_i' e_i^2.

    HC0, HC1:  e_i = u_i               (HC1's n/(n-k) factor is applied by the caller;
                                        Stata's vce(robust))
    HC2:       e_i = u_i / sqrt(1-h_i)  (Stata's vce(hc2))
    HC3:       e_i = u_i / (1-h_i)      (Stata's vce(hc3))

    resid is [n] or [n, m]; leverage h is [n]. HC0/HC1 return resid itself (no
    copy). HC2/HC3 raise KernelError("undefined_leverage_correction") when any
    1 - h_i <= 100 * eps (an observation fitted exactly by its own dummy).
    """
    if kind in ("HC0", "HC1"):
        return resid
    if kind not in ("HC2", "HC3"):
        raise KernelError("unsupported_covariance", f"Unknown HC residual kind: {kind}.")
    if not isinstance(resid, Tensor) or not isinstance(leverage, Tensor) \
            or resid.ndim not in (1, 2) or leverage.shape != resid.shape[:1]:
        raise KernelError("missing_leverage", f"{kind} needs one leverage value per residual.")
    gap = 1 - leverage
    if not bool((gap > 100 * _EPS).all()):      # also catches a NaN leverage
        raise KernelError("undefined_leverage_correction",
                          f"{kind} is undefined for observations with unit leverage.")
    if resid.ndim == 2:
        gap = gap[:, None]
    return resid / (gap.sqrt() if kind == "HC2" else gap)


@torch.no_grad()
def meat_white(scores: Tensor) -> Tensor:
    """Heteroskedasticity-robust (White) meat  M = S'S = sum_i s_i s_i'."""
    _scores(scores)
    return symmetrize(scores.T @ scores)


@torch.no_grad()
def meat_cluster(scores: Tensor, codes: Tensor, n_groups: int) -> Tensor:
    """One-way cluster meat  M = sum_g t_g t_g'  with  t_g = sum_{i in g} s_i.

    No small-sample factor; Stata's regress multiplies by cluster_factor(n, k, G).
    """
    rows, _ = _scores(scores)
    _codes(codes, rows, n_groups)
    codes, groups, _ = _compact(codes, n_groups)
    totals = group_sums(scores, codes, groups)
    return symmetrize(totals.T @ totals)


@dataclass
class MultiwayMeat:
    meat: Tensor               # [k, k], factors of `adjust` (and n, k) already applied
    group_counts: list[int]    # nonempty groups per cluster dimension, in input order
    min_groups: int            # G_min = min(group_counts)
    psd_adjusted: bool         # True when negative eigenvalues were replaced by zero
    terms: list[dict]          # one per subset: dimensions, sign, groups, factor


def _intersection(dimensions: list[tuple[Tensor, int]], members: list[int]) -> tuple[Tensor, int]:
    """Compact codes and count of the nonempty cells of the crossed dimensions."""
    key, bound = dimensions[members[0]]
    for index in members[1:]:
        codes, groups = dimensions[index]
        if bound * groups >= 1 << 62:      # recode before the mixed-radix key can overflow
            cells, key = torch.unique(key, return_inverse=True)
            bound = cells.numel()
        key = key * groups + codes
        bound *= groups
    cells, key = torch.unique(key, return_inverse=True)
    return key, cells.numel()


@torch.no_grad()
def meat_multiway(
    scores: Tensor,
    dimensions: list[tuple[Tensor, int]],
    *,
    adjust: str = "min",
    n: int | None = None,
    k: int | None = None,
    force_psd: bool = False,
) -> MultiwayMeat:
    """Cameron-Gelbach-Miller multiway cluster meat by inclusion-exclusion.

    dimensions is a list of (int64 codes, n_groups), one per clustering variable.
    For every nonempty subset S of the dimensions, rows are clustered on the
    intersection of S (the crossed codes) and

        M = sum_S (-1)^{|S|+1} c_S M_S ,   M_S = sum_{g in cells(S)} t_g t_g' ,

    e.g. two-way: M_1 + M_2 - M_12. G_S is the number of nonempty cells of S.

    adjust="none":  c_S = 1.
    adjust="min":   the whole meat is multiplied by G_min/(G_min-1), G_min the
                    smallest one-way group count (reghdfe / ivreg2 convention).
    adjust="each":  c_S = G_S/(G_S-1) for every term (CGM / cgmreg convention).
    If n and k are both given the meat is also multiplied by (n-1)/(n-k).

    force_psd=True: the inclusion-exclusion meat need not be positive
    semidefinite; negative eigenvalues are then replaced by zero (CGM 2011,
    eq. 2.13) and psd_adjusted is set.
    """
    rows, width = _scores(scores)
    if adjust not in ("none", "min", "each"):
        raise KernelError("invalid_cluster_adjustment", "adjust must be none, min or each.")
    if (n is None) != (k is None) or (n is not None and n <= k):
        raise KernelError("invalid_cluster_adjustment",
                          "Give both n and k with n > k for the (n-1)/(n-k) factor, or neither.")
    if not dimensions:
        raise KernelError("invalid_clusters", "Multiway clustering needs at least one dimension.")
    if not isinstance(dimensions, (list, tuple)) or not all(
            isinstance(entry, (list, tuple)) and len(entry) == 2 for entry in dimensions):
        raise KernelError("invalid_clusters", "Each cluster dimension is a (codes, n_groups) pair.")
    counts = []
    compact = []
    for codes, groups in dimensions:
        _codes(codes, rows, groups, "cluster codes")
        codes, groups, cells = _compact(codes, groups)
        compact.append((codes, groups))
        counts.append(cells)
    dimensions = compact
    smallest = min(counts)
    if smallest < 2 and adjust != "none":
        raise KernelError("insufficient_clusters",
                          "Cluster covariance requires at least two groups per dimension.")
    overall = smallest / (smallest - 1) if adjust == "min" else 1.0
    if n is not None:
        overall *= (n - 1) / (n - k)
    meat = torch.zeros((width, width), dtype=scores.dtype, device=scores.device)
    white = None
    terms = []
    for mask in range(1, 1 << len(dimensions)):
        members = [d for d in range(len(dimensions)) if mask >> d & 1]
        if len(members) == 1:
            codes, groups, cells = *dimensions[members[0]], counts[members[0]]
        else:
            codes, cells = _intersection(dimensions, members)
            groups = cells
        if cells == rows:
            # Every row is its own cell: the term is the White meat (computed once).
            white = scores.T @ scores if white is None else white
            term = white
        else:
            totals = group_sums(scores, codes, groups)
            term = totals.T @ totals
        sign = 1 if len(members) % 2 else -1
        factor = cells / (cells - 1) if adjust == "each" else 1.0
        meat.add_(term, alpha=sign * factor)
        terms.append({"dimensions": members, "sign": sign, "groups": cells,
                      "factor": factor * overall})
    meat = symmetrize(meat * overall)
    adjusted = False
    if force_psd:
        meat, adjusted = nearest_psd(meat)
    return MultiwayMeat(meat=meat, group_counts=counts, min_groups=smallest,
                        psd_adjusted=adjusted, terms=terms)


def _kernel(z: Tensor, kernel: str) -> Tensor:
    """Kernel k(z) at z = lag / bandwidth > 0."""
    if kernel == "bartlett":
        return (1 - z).clamp_min(0)
    if kernel == "truncated":
        return (z < 1).to(z.dtype)
    if kernel == "parzen":
        return torch.where(z <= 0.5, 1 - 6 * z.square() + 6 * z.pow(3),
                           2 * (1 - z).clamp_min(0).pow(3))
    angle = 6 * math.pi * z / 5
    square = angle.square()
    direct = 3 / square * (torch.sin(angle) / angle - torch.cos(angle))
    # Below a = 1 the closed form cancels two terms of order one down to a^2/3 and loses
    # up to 2 eps / a^2 relative accuracy (1e-5 at lag 1 of a bandwidth of 1e6); the
    # alternating power series has no cancellation there.
    series = torch.full_like(angle, _QS_SERIES[-1])
    for coefficient in reversed(_QS_SERIES[:-1]):
        series = coefficient - square * series
    return torch.where(angle < 1, series, direct)


@torch.no_grad()
def kernel_weights(lags: int, kernel: str = "bartlett", *, count: int | None = None) -> Tensor:
    """HAC kernel weights w_l for lags l = 1..L (float64 [L]), bandwidth L + 1, z = l/(L+1).

    bartlett            w = 1 - z                       (Newey-West; Stata's newey,
                                                         ivreg2 bw(L+1))
    truncated           w = 1                           (l <= L)
    parzen              w = 1 - 6 z^2 + 6 z^3           (z <= 1/2)
                        w = 2 (1 - z)^3                 (1/2 < z < 1)
    quadratic_spectral  w = 3/a^2 (sin(a)/a - cos(a)),  a = 6 pi z / 5

    The first three vanish for l > L. The quadratic spectral kernel has unbounded
    support: there `lags` only sets the bandwidth L + 1, and meat_hac uses every
    available lag. count (default L) asks for the weights of lags 1..count.
    """
    _lags(lags, kernel)
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise KernelError("invalid_lags", "count must be a nonnegative integer.")
    size = lags if count is None else count
    return _kernel(torch.arange(1, size + 1, dtype=torch.float64) / (lags + 1), kernel)


def _series_keys(
    time: Tensor, panel: Tensor | None, span: int, reach: int
) -> tuple[Tensor, Tensor | None]:
    """Sorted integer keys unit * stride + period (stride = span + reach) and the sort.

    span is the period range and reach the largest period distance that carries
    weight: two rows of one unit are at most span - 1 apart and rows of different
    units at least reach + 1 apart, so a key distance of at most reach identifies a
    pair within one unit.  Units are renumbered 0..U-1 in key order, so the dense
    period grid has U * stride cells whatever the unit codes.  The returned order
    is None when the rows were already sorted.
    """
    period = time - int(time.min())
    stride = span + reach
    if panel is None:
        keys = period
    else:
        first = int(panel.min())
        if (int(panel.max()) - first + 1) * stride < 1 << 62:
            keys = (panel - first) * stride + period
        else:       # raw identifiers too wide to combine directly: renumber them first
            keys = torch.unique(panel, return_inverse=True)[1] * stride + period
    order = None
    if not bool((keys[1:] > keys[:-1]).all()):
        keys, order = torch.sort(keys)
        if bool((keys[1:] == keys[:-1]).any()):
            raise KernelError("repeated_time_values", "Each period may appear only once "
                              "per series (repeated time values).")
    if panel is not None:
        unit = keys // stride
        fresh = torch.zeros_like(keys)
        fresh[1:] = (unit[1:] != unit[:-1]).to(torch.int64).cumsum(0)
        keys = fresh * stride + keys - unit * stride
    return keys, order


def _lagged_sums_fft(
    ordered: Tensor, keys: Tensor | None, extent: int, reach: int, lags: int, kernel: str
) -> Tensor:
    """p_t = sum_{d=1..reach} k(d / (L+1)) s_{t-d} for every row, by FFT convolution.

    Rows sit on the dense period grid at position keys (their own index when keys is
    None); empty positions hold zeros.  The grid is zero-padded to m >= extent + reach
    so the circular convolution has no wrap-around.  Rounding is of order
    eps * ||s||_2 * ||k||_2 in absolute terms, i.e. relative to the meat's own scale.
    """
    n, width = ordered.shape
    size = 1 << (extent + reach - 1).bit_length()
    taps = torch.zeros(size, dtype=ordered.dtype, device=ordered.device)
    distance = torch.arange(1, reach + 1, dtype=ordered.dtype, device=ordered.device)
    taps[1: reach + 1] = _kernel(distance / (lags + 1), kernel)
    spectrum = torch.fft.rfft(taps)[:, None]
    partner = torch.empty_like(ordered)
    block = max(1, _FFT_ELEMENTS // size)
    for start in range(0, width, block):
        columns = ordered[:, start: start + block]
        if keys is not None:
            grid = torch.zeros((extent, columns.shape[1]), dtype=ordered.dtype,
                               device=ordered.device)
            columns = grid.index_copy_(0, keys, columns)
        lagged = torch.fft.irfft(torch.fft.rfft(columns, n=size, dim=0) * spectrum, n=size, dim=0)
        partner[:, start: start + block] = lagged[:n] if keys is None else lagged[keys]
    return partner


@torch.no_grad()
def meat_hac(
    scores: Tensor,
    lags: int,
    kernel: str = "bartlett",
    *,
    time: Tensor | None = None,
    panel: Tensor | None = None,
) -> Tensor:
    """Newey-West style HAC meat  M = G_0 + sum_{l=1..L} w_l (G_l + G_l'),
    G_l = sum_t s_t s_{t-l}',  w_l = kernel_weights(L, kernel).

    time=None:   rows are consecutive periods of ONE series, already in time order.
    time given:  int64 period index, possibly with gaps and in any row order; only
                 pairs exactly l periods apart contribute (Stata's newey on tsset
                 data with gaps). Periods must be distinct within a series.
    panel given: int64 unit codes (requires time); autocovariances are formed within
                 units only (Stata's newey on xtset data, ivreg2 bw() robust). Any
                 int64 identifiers will do: units are renumbered internally.

    lags=0 reduces to meat_white. For the quadratic spectral kernel lags is the
    bandwidth minus one and every available lag is used. No n/(n-k) factor.

    Computed as S'S + S'P + P'S with the lag-weighted partner sum
    p_t = sum_l w_l s_{t-l}. With time given, rows are sorted by (unit, period); the
    row j positions earlier is then at least j periods earlier, so offsets
    j = 1..L cover every pair within L periods, each offset being one contiguous
    O(n k) update weighted by the kernel at the actual period distance. Lag sums
    reaching beyond 64 periods (always the case for the quadratic spectral kernel
    on a long series) are formed by an FFT convolution on the period grid instead,
    O(k T log T), unless the periods are so sparse that the grid would exceed four
    times the number of rows (then the offset scan is used: O(n k * offsets)).
    """
    n, _ = _scores(scores)
    _lags(lags, kernel)
    if panel is not None and time is None:
        raise KernelError("invalid_time", "A panel HAC meat needs the period of every row.")
    for name, index in (("time", time), ("panel", panel)):
        if index is not None and (not isinstance(index, Tensor) or index.dtype != torch.int64
                                  or index.shape != (n,)):
            raise KernelError("invalid_time", f"{name} must be an int64 vector, one per row.")
    meat = scores.T @ scores
    if lags == 0 or n < 2:
        return symmetrize(meat)
    unbounded = kernel == "quadratic_spectral"
    ordered, keys = scores, None
    # reach is the largest period distance that can carry weight.
    if time is None:
        extent, reach = n, n - 1 if unbounded else min(lags, n - 1)
    else:
        span = int(time.max()) - int(time.min()) + 1
        if span >= 1 << 62:
            raise KernelError("invalid_time", "The period index is too wide for int64 "
                              "arithmetic; recode periods as compact integers.")
        # Distances of span or more can only be pairs from different units.
        reach = (span if unbounded else min(span, lags + 1)) - 1
        keys, order = _series_keys(time, panel, span, reach)
        if order is not None:
            ordered = scores.index_select(0, order)
        extent = int(keys[-1]) + 1
    if reach > _FFT_OFFSETS and extent <= _FFT_SPARSITY * n:
        partner = _lagged_sums_fft(ordered, keys, extent, reach, lags, kernel)
    elif keys is None:
        partner = torch.zeros_like(scores)
        weights = kernel_weights(lags, kernel, count=reach)
        for offset, weight in enumerate(weights.tolist(), start=1):
            partner[offset:].add_(scores[: n - offset], alpha=weight)
    else:
        partner = torch.zeros_like(scores)
        for offset in range(1, n):
            distance = keys[offset:] - keys[: n - offset]
            if bool((distance > reach).all()):
                break       # distances grow with the offset: nothing further contributes
            weights = _kernel(distance.to(scores.dtype) / (lags + 1), kernel)
            weights = torch.where(distance <= reach, weights, torch.zeros_like(weights))
            partner[offset:].addcmul_(ordered[: n - offset], weights[:, None])
    cross = ordered.T @ partner
    return symmetrize(meat + cross + cross.T)


@torch.no_grad()
def meat_driscoll_kraay(
    scores: Tensor, time: Tensor, lags: int, kernel: str = "bartlett"
) -> Tensor:
    """Driscoll-Kraay meat: HAC of the cross-sectional score sums.

        h_t = sum_{i: time_i = t} s_i ,
        M = H_0 + sum_{l=1..L} w_l (H_l + H_l') ,   H_l = sum_t h_t h_{t-l}' ,

    where h_t and h_{t-l} are exactly l periods apart (period gaps are respected).
    Robust to heteroskedasticity, serial correlation up to L lags and arbitrary
    cross-sectional dependence (Stata's xtscc / ivreg2 cluster(time) bw()).
    """
    n, _ = _scores(scores)
    if not isinstance(time, Tensor) or time.dtype != torch.int64 or time.shape != (n,):
        raise KernelError("invalid_time", "time must be an int64 vector, one period per row.")
    _lags(lags, kernel)
    if n == 0:
        return symmetrize(scores.T @ scores)
    periods, index = torch.unique(time, return_inverse=True)
    return meat_hac(group_sums(scores, index, periods.numel()), lags, kernel, time=periods)


def newey_west_lags(n: int) -> int:
    """Newey-West rule-of-thumb lag length  L = floor(4 * (n / 100)^(2/9))  for n periods."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise KernelError("invalid_lags", "The number of periods must be a positive integer.")
    return math.floor(4 * (n / 100) ** (2 / 9))


@torch.no_grad()
def sandwich(bread: Tensor, meat: Tensor) -> Tensor:
    """Sandwich covariance  V = B M B'  (symmetrized); B is [p, k], M is [k, k]."""
    for matrix in (bread, meat):
        if not isinstance(matrix, Tensor) or matrix.dtype != torch.float64 or matrix.ndim != 2:
            raise KernelError("invalid_covariance", "Bread and meat must be float64 matrices.")
    if meat.shape != (bread.shape[1], bread.shape[1]):
        raise KernelError("invalid_covariance", "The meat does not conform to the bread.")
    return symmetrize(bread @ meat @ bread.T)


def cluster_factor(n: int, k: int, groups: int) -> float:
    """Stata regress cluster (CR1) small-sample factor  G/(G-1) * (n-1)/(n-k)."""
    if groups < 2:
        raise KernelError("insufficient_clusters", "Cluster covariance requires at least two "
                          "groups.")
    if n <= k:
        raise KernelError("insufficient_observations", "The cluster factor needs more "
                          "observations than parameters.")
    return groups / (groups - 1) * (n - 1) / (n - k)


@torch.no_grad()
def nearest_psd(a: Tensor) -> tuple[Tensor, bool]:
    """Replace negative eigenvalues of a symmetric matrix by zero.

    With A = U diag(l) U' the result is U diag(max(l, 0)) U' (the Frobenius-nearest
    positive semidefinite matrix; CGM 2011 eq. 2.13). Returns (matrix, adjusted);
    eigenvalues above -64 k eps max|l| count as rounding noise, in which case the
    symmetrized input is returned unchanged with adjusted=False.
    """
    if not isinstance(a, Tensor) or a.dtype != torch.float64 or a.ndim != 2 \
            or a.shape[0] != a.shape[1]:
        raise KernelError("invalid_covariance", "nearest_psd expects a square float64 matrix.")
    a = symmetrize(a)
    if a.shape[0] == 0:
        return a, False
    if not bool(torch.isfinite(a).all()):
        raise KernelError("invalid_covariance", "nearest_psd expects a finite matrix.")
    try:
        values, vectors = torch.linalg.eigh(a)
    except RuntimeError as exc:
        raise KernelError("numerical_failure", f"The eigen-decomposition failed: {exc}") from exc
    if float(values[0]) >= -64 * a.shape[0] * _EPS * float(values.abs().max()):
        return a, False
    return symmetrize((vectors * values.clamp_min(0)) @ vectors.T), True
