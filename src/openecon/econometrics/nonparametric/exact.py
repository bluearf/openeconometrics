"""Exact null distributions of rank and goodness-of-fit statistics (float64 tensors).

Every function works on the support of the statistic, never on the n! or 2^n
arrangements themselves:

* ``rank_sum_distribution``: the permutation distribution of the sum of m of N
  integer scores (Mann-Whitney / Wilcoxon rank sum, with or without ties) by
  the subset-sum recursion  f_k(s) += f_{k-1}(s - w)  over the scores w.
* ``signed_sum_distribution``: the distribution of the sum of the scores with
  positive sign under independent fair signs (Wilcoxon signed rank), the
  coefficients of  prod_j (1 + q^{w_j}) / 2^n.
* ``kolmogorov_sf_exact``: P(D_n >= d) for the one-sample two-sided Kolmogorov
  statistic. Marsaglia, Tsang and Wang (2003) evaluate P(D_n < d) as the
  (k, k) element of the n-th power of Durbin's (2k-1)-square matrix; in the far
  right tail, where that power loses relative accuracy, the function returns
  twice the exact one-sided probability of Birnbaum and Tingey (1951) (the two
  one-sided events overlap with probability below 1e-13 there).
* ``kolmogorov_sf``: the limiting Kolmogorov distribution, from whichever of
  its two series converges fastest.
* ``smirnov_exact``: P(D_{m,n} >= d) for two samples without ties by counting
  the lattice paths that leave the band |i/m - j/n| < d.
* ``runs_distribution``: the distribution of the number of runs given n1, n2.
* ``hypergeometric``: the pmf of the (1, 1) cell of a 2x2 table (Fisher), from the
  ratios of successive probabilities.

Recursions add nonnegative terms only, so their relative error is a few ulps
per step; counts are rescaled before they can overflow.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

FLOAT = torch.float64
# Largest dynamic-programming table (elements) an exact rank distribution may use.
MAX_TABLE = 6_000_000
# Largest (support size) x (number of scores) an exact signed-rank distribution may use.
MAX_SIGNED_WORK = 300_000_000
# Largest number of support points of a hypergeometric distribution tabulated at once.
MAX_SUPPORT = 4_000_000


def _unavailable(what: str) -> KernelError:
    return KernelError("exact_unavailable", f"The exact distribution of {what} is too large to "
                       "enumerate for this sample. Use exact=False for the asymptotic p-value.")


def rank_sum_distribution(scores: Tensor, m: int) -> Tensor:
    """P(sum of the scores of a random m-subset = s), s = 0..S, for integer scores.

    ``scores`` are nonnegative int64 (ranks, or doubled midranks when there are
    ties). S is the sum of the m largest scores.
    """
    total = scores.numel()
    if not 0 < m <= total:
        raise KernelError("invalid_input", "The subset size must lie between 1 and the number "
                          "of scores.")
    top = int(torch.sort(scores, descending=True).values[:m].sum())
    if (m + 1) * (top + 1) > MAX_TABLE:
        raise _unavailable("the rank sum")
    table = torch.zeros((m + 1, top + 1), dtype=FLOAT)
    table[0, 0] = 1.0
    for index, weight in enumerate(scores.tolist()):
        rows = min(m, index + 1)
        if weight == 0:
            table[1:rows + 1] = table[1:rows + 1] + table[:rows]
        else:
            shifted = table[:rows, :top + 1 - weight]
            table[1:rows + 1, weight:] = table[1:rows + 1, weight:] + shifted
        if index % 64 == 63:
            peak = float(table.max())
            if peak > 1e250:
                table /= peak
    counts = table[m]
    return counts / counts.sum()


def signed_sum_distribution(scores: Tensor) -> Tensor:
    """P(sum of the positively signed scores = s), s = 0..sum(scores), under fair signs."""
    top = int(scores.sum())
    if (top + 1) * max(scores.numel(), 1) > MAX_SIGNED_WORK:
        raise _unavailable("the signed-rank sum")
    counts = torch.zeros(top + 1, dtype=FLOAT)
    counts[0] = 1.0
    for index, weight in enumerate(scores.tolist()):
        if weight == 0:
            counts = counts * 2.0
        else:
            counts[weight:] = counts[weight:] + counts[:top + 1 - weight]
        if index % 64 == 63:
            counts /= counts.sum()
    return counts / counts.sum()


def tail_probabilities(pmf: Tensor, observed: int, centre: float) -> tuple[float, float, float]:
    """(P(S <= s), P(S >= s), P(|S - centre| >= |s - centre|)) on an integer support."""
    support = torch.arange(pmf.numel(), dtype=FLOAT)
    lower = float(pmf[:observed + 1].sum())
    upper = float(pmf[observed:].sum())
    distance = abs(observed - centre)
    far = (support - centre).abs() >= distance - 1e-9 * max(1.0, distance)
    return min(1.0, lower), min(1.0, upper), min(1.0, float(pmf[far].sum()))


# ---- Kolmogorov ------------------------------------------------------------------------


def kolmogorov_sf(z: float) -> float:
    """Q(z) = P(K > z) for the limiting Kolmogorov distribution (z = sqrt(n) D)."""
    if z <= 0.0:
        return 1.0
    if z < 1.18:
        # theta-function form, fast for small z: K(z) = sqrt(2 pi)/z sum exp(-(2j-1)^2 pi^2/(8 z^2))
        base = -math.pi ** 2 / (8.0 * z * z)
        cdf = math.sqrt(2.0 * math.pi) / z * sum(math.exp((2 * j - 1) ** 2 * base)
                                                 for j in range(1, 12))
        return min(1.0, max(0.0, 1.0 - cdf))
    total = 0.0
    for j in range(1, 60):
        term = math.exp(-2.0 * j * j * z * z)
        total += term if j % 2 else -term
        if term < 1e-17 * abs(total):
            break
    return min(1.0, max(0.0, 2.0 * total))


def kolmogorov_one_sided_sf(n: int, d: float) -> float:
    """Exact P(D_n^+ >= d) (Birnbaum and Tingey 1951, Smirnov):

        d * sum_{j=0}^{floor(n (1 - d))} C(n, j) (d + j/n)^(j-1) (1 - d - j/n)^(n-j).
    """
    if d <= 0.0:
        return 1.0
    if d >= 1.0:
        return 0.0
    j = torch.arange(int(math.floor(n * (1.0 - d))) + 1, dtype=FLOAT)
    rest = (1.0 - d - j / n).clamp_min(0.0)
    log_terms = (math.lgamma(n + 1) - torch.lgamma(j + 1) - torch.lgamma(n - j + 1)
                 + (j - 1.0) * torch.log(d + j / n) + torch.xlogy(n - j, rest))
    return min(1.0, d * float(torch.exp(log_terms).sum()))


def _matrix_power(matrix: Tensor, power: int) -> tuple[Tensor, float]:
    """(M^power / 2^shift, shift * ln 2) by repeated squaring with rescaling."""
    result, log_scale = torch.eye(matrix.shape[0], dtype=FLOAT), 0.0
    base, base_scale = matrix.clone(), 0.0
    while power:
        if power & 1:
            result = result @ base
            log_scale += base_scale
            peak = float(result.abs().max())
            if peak > 1e140 or 0.0 < peak < 1e-140:
                result, log_scale = result / peak, log_scale + math.log(peak)
        power >>= 1
        if power:
            base = base @ base
            base_scale *= 2.0
            peak = float(base.abs().max())
            if peak > 1e140 or 0.0 < peak < 1e-140:
                base, base_scale = base / peak, base_scale + math.log(peak)
    return result, log_scale


def kolmogorov_cdf_matrix(n: int, d: float) -> float:
    """P(D_n < d) by the Durbin matrix algorithm of Marsaglia, Tsang and Wang (2003)."""
    if d >= 1.0:
        return 1.0
    if n * d <= 0.5:
        return 0.0
    k = int(n * d) + 1
    m = 2 * k - 1
    h = k - n * d
    if m > 1500:                                     # about n = 150,000 at ordinary D
        raise _unavailable("the Kolmogorov statistic")
    index = torch.arange(m)
    gap = index[:, None] - index[None, :] + 1                     # i - j + 1
    matrix = (gap >= 0).to(FLOAT)
    powers = torch.pow(torch.tensor(h, dtype=FLOAT), torch.arange(1, m + 1, dtype=FLOAT))
    matrix[:, 0] -= powers                                        # h^(i+1)
    matrix[m - 1, :] -= powers.flip(0)                            # h^(m-j)
    if 2.0 * h - 1.0 > 0.0:
        matrix[m - 1, 0] += (2.0 * h - 1.0) ** m
    matrix = matrix / torch.exp(torch.lgamma(gap.clamp_min(0).to(FLOAT) + 1.0))
    matrix = torch.where(gap >= 0, matrix, torch.zeros_like(matrix))
    power, log_scale = _matrix_power(matrix, n)
    value = float(power[k - 1, k - 1])
    if value <= 0.0:
        return 0.0
    log_cdf = math.log(value) + log_scale + math.lgamma(n + 1) - n * math.log(n)
    return min(1.0, math.exp(log_cdf))


def kolmogorov_sf_exact(n: int, d: float) -> float:
    """Exact P(D_n >= d) for the two-sided one-sample Kolmogorov statistic."""
    if d <= 0.0:
        return 1.0
    if d >= 1.0:
        return 0.0
    s = n * d * d
    if s > 7.24 or (s > 3.76 and n > 99) or d >= 0.5:
        # P(D+ >= d, D- >= d) < 2 exp(-8 n d^2): negligible here, and zero for d >= 1/2.
        return min(1.0, 2.0 * kolmogorov_one_sided_sf(n, d))
    return min(1.0, max(0.0, 1.0 - kolmogorov_cdf_matrix(n, d)))


def smirnov_exact(n1: int, n2: int, h: int) -> float:
    """P(max_t |i(t) n2 - j(t) n1| >= h) for two samples of sizes n1, n2 without ties.

    ``h`` is the observed statistic on the integer lattice, D = h / (n1 n2).
    Lattice paths from (0, 0) to (n1, n2) are counted row by row in two
    classes: A, paths that have stayed strictly inside the band
    |i n2 - j n1| < h, and B, paths that have left it at least once. Inside the
    band A is a running sum of the previous row; B collects the paths that
    step out. Only nonnegative terms are added, so small p-values keep their
    relative accuracy (no 1 - P(inside)).
    """
    if h <= 0:
        return 1.0
    if n1 > n2:
        n1, n2 = n2, n1
    if n1 * n2 > 400_000_000:
        raise _unavailable("the two-sample Kolmogorov-Smirnov statistic")
    j = torch.arange(n2 + 1, dtype=torch.int64)
    stayed = ((j * n1) < h).to(FLOAT)                   # row i = 0: a single path per point
    left = 1.0 - stayed
    zero = torch.zeros(1, dtype=FLOAT)
    log_scale = 0.0
    for i in range(1, n1 + 1):
        inside = ((i * n2 - j * n1).abs() < h).to(FLOAT)
        arriving = torch.cumsum(stayed * inside, dim=0) * inside
        exits = (stayed + torch.cat([zero, arriving[:-1]])) * (1.0 - inside)
        left = torch.cumsum(left + exits, dim=0)
        stayed = arriving
        peak = max(float(stayed.max()), float(left.max()))
        if peak > 1e200:
            stayed, left, log_scale = stayed / peak, left / peak, log_scale + math.log(peak)
    outside = float(left[n2])
    if outside <= 0.0:
        return 0.0
    log_total = math.lgamma(n1 + n2 + 1) - math.lgamma(n1 + 1) - math.lgamma(n2 + 1)
    return min(1.0, math.exp(math.log(outside) + log_scale - log_total))


# ---- runs and hypergeometric -----------------------------------------------------------


def _log_choose(n: Tensor | float, k: Tensor) -> Tensor:
    n = torch.as_tensor(n, dtype=FLOAT)
    valid = (k >= 0) & (k <= n)
    safe = torch.where(valid, k, torch.zeros_like(k))
    value = torch.lgamma(n + 1) - torch.lgamma(safe + 1) - torch.lgamma(n - safe + 1)
    return torch.where(valid, value, torch.full_like(value, -math.inf))


def runs_distribution(n1: int, n2: int) -> tuple[Tensor, Tensor]:
    """(support r, P(R = r)) of the number of runs in a random arrangement of n1 + n2 items.

    P(R = 2k) = 2 C(n1-1, k-1) C(n2-1, k-1) / C(N, n1);
    P(R = 2k+1) = [C(n1-1, k-1) C(n2-1, k) + C(n1-1, k) C(n2-1, k-1)] / C(N, n1).
    """
    top = 2 * min(n1, n2) + (1 if n1 != n2 else 0)
    support = torch.arange(2, top + 1, dtype=FLOAT)
    k = torch.floor(support / 2.0)
    log_total = (math.lgamma(n1 + n2 + 1) - math.lgamma(n1 + 1) - math.lgamma(n2 + 1))
    a, b = float(n1 - 1), float(n2 - 1)
    even = math.log(2.0) + _log_choose(a, k - 1) + _log_choose(b, k - 1)
    odd = torch.logaddexp(_log_choose(a, k - 1) + _log_choose(b, k),
                          _log_choose(a, k) + _log_choose(b, k - 1))
    pmf = torch.exp(torch.where(support % 2 == 0, even, odd) - log_total)
    return support, pmf / pmf.sum()


def hypergeometric(row1: int, col1: int, total: int) -> tuple[int, Tensor]:
    """(smallest tabulated a, P(A = a)) for the (1, 1) cell of a 2x2 table with fixed margins.

    The pmf is built from the ratios

        P(a + 1) / P(a) = (row1 - a)(col1 - a) / ((a + 1)(total - row1 - col1 + a + 1))

    by a cumulative sum of their logarithms and then normalized, so no
    factorial of the table total is evaluated and the relative error does not
    grow with the counts. When the support has more than ``MAX_SUPPORT`` points
    (frequency-weighted tables with very large totals) only the window of
    50 standard deviations + 300 on each side of the mean is tabulated; the
    probability outside it is below 1e-300. ``exact_unavailable`` when even
    that window is too large.
    """
    low, high = max(0, row1 + col1 - total), min(row1, col1)
    if high - low + 1 > MAX_SUPPORT:
        mean = row1 * (col1 / total)
        deviation = math.sqrt(max(0.0, mean * (1.0 - row1 / total) * (total - col1)
                                  / max(total - 1.0, 1.0)))
        reach = 50.0 * deviation + 300.0
        low, high = max(low, int(math.floor(mean - reach))), min(high, int(math.ceil(mean + reach)))
        if high - low + 1 > MAX_SUPPORT:
            raise _unavailable("the 2x2 table (Fisher's exact test)")
    a = torch.arange(low, high, dtype=FLOAT)
    steps = torch.log((row1 - a) * (col1 - a)) - torch.log((a + 1.0) * (total - row1 - col1
                                                                       + a + 1.0))
    log_pmf = torch.cat([torch.zeros(1, dtype=FLOAT), torch.cumsum(steps, 0)])
    pmf = torch.exp(log_pmf - log_pmf.max())
    return low, pmf / pmf.sum()
