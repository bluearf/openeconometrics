"""Nearest-neighbour search for matching estimators without n-by-n matrices.

Matching with replacement finds, for every query unit, the ``m`` nearest units
of a pool (the other treatment group, or the unit's own group without itself)
and, as Stata's ``teffects nnmatch``/``psmatch`` do, keeps every further pool
unit whose distance ties with the m-th one: the matched set ``J(i)`` holds
``#J(i) >= m`` units. Ties are exact equalities of the computed distances.

One dimension (the propensity score) is solved by sorting: the m nearest
pool units of a query form a contiguous window of the sorted pool. The window
grows from the insertion point by ``m`` comparisons of the next left and right
candidate (a loop over neighbours, vectorized over all queries); ties with
the m-th distance are then added by a vectorized binary search on each side,
which is exact because floating-point subtraction is monotone. Cost
O(n log n). Window sums of pool columns come from prefix sums and the usage
weights ``K_j = sum_i 1{j in J(i)} / #J(i)`` (and ``K'_j`` with ``#J(i)^2``)
from difference arrays: O(n) after the sort.

Several dimensions use blockwise exact distances: queries are processed in
blocks whose distance matrix to the pool holds at most ``_BLOCK_ELEMENTS``
entries, computed from exact coordinate differences (``torch.cdist`` without
the ``|a|^2 + |b|^2 - 2ab`` expansion, so duplicated covariate rows are at
distance exactly zero); the m-th distance comes from ``topk`` and the matched
pairs are accumulated sparsely with ``index_add_``. Time is
O(n_query n_pool k) with bounded memory, about 3e8 pairs per second on one
CPU; ``MAX_PAIRS`` guards against searches that would take minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

_BLOCK_ELEMENTS = 1 << 22
# Largest number of query-pool distance evaluations of one blockwise search (~15 s).
MAX_PAIRS = 4_000_000_000
# Up to this many neighbours the m-th distance comes from topk (much faster than kthvalue).
_TOPK_LIMIT = 32


@dataclass
class Matches:
    count: Tensor          # [nq] size of each matched set #J(i) (float64)
    sums: Tensor           # [nq, c] sums of the pool columns over J(i)
    usage: Tensor | None   # [np] K_j = sum_i 1{j in J(i)} / #J(i)
    usage2: Tensor | None  # [np] K'_j = sum_i 1{j in J(i)} / #J(i)^2
    distance: Tensor       # [nq] distance of the m-th match (Euclidean in the coordinates)


def _extend_left(query: Tensor, pool: Tensor, lo: Tensor, reach: Tensor) -> Tensor:
    """Smallest j <= lo with query - pool[j] <= reach (the predicate is monotone in j)."""
    low, high = torch.zeros_like(lo), lo.clone()
    active = low < high
    while bool(active.any()):
        mid = (low + high) // 2
        inside = (query - pool[mid]) <= reach
        high = torch.where(active & inside, mid, high)
        low = torch.where(active & ~inside, mid + 1, low)
        active = low < high
    return low


def _extend_right(query: Tensor, pool: Tensor, hi: Tensor, reach: Tensor) -> Tensor:
    """One past the largest j >= hi - 1 with pool[j] - query <= reach."""
    size = pool.numel()
    low, high = hi.clone(), torch.full_like(hi, size)
    active = low < high
    while bool(active.any()):
        mid = (low + high) // 2
        inside = (pool[mid.clamp(max=size - 1)] - query) <= reach
        low = torch.where(active & inside, mid + 1, low)
        high = torch.where(active & ~inside, mid, high)
        active = low < high
    return low


def windows_sorted(query: Tensor, pool: Tensor, m: int, own: Tensor | None = None
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Windows ``[lo, hi)`` of the sorted ``pool`` holding the tie-extended m nearest.

    ``own`` gives, for queries that belong to the pool, their position in it:
    the window then contains the query itself, which does not count towards
    ``m`` (the caller removes it from sums and counts). Returns lo, hi and the
    m-th distance.
    """
    size = pool.numel()
    available = size - (0 if own is None else 1)
    if m > available:
        raise KernelError("insufficient_observations",
                          f"Matching needs at least {m} candidate(s) but only {available} exist.")
    if own is None:
        lo = torch.searchsorted(pool, query)
        hi = lo.clone()
    else:
        lo, hi = own.clone(), own + 1
    reach = torch.zeros_like(query)
    infinity = torch.tensor(float("inf"), dtype=query.dtype)
    for _ in range(m):
        left, right = lo - 1, hi
        dl = torch.where(left >= 0, query - pool[left.clamp(min=0)], infinity)
        dr = torch.where(right < size, pool[right.clamp(max=size - 1)] - query, infinity)
        take_left = dl <= dr
        reach = torch.where(take_left, dl, dr)
        lo = torch.where(take_left, left, lo)
        hi = torch.where(take_left, hi, right + 1)
    return _extend_left(query, pool, lo, reach), _extend_right(query, pool, hi, reach), reach


def _prefix(values: Tensor) -> Tensor:
    return torch.cat([torch.zeros((1, values.shape[1]), dtype=values.dtype),
                      values.cumsum(dim=0)], dim=0)


def _usage(lo: Tensor, hi: Tensor, share: Tensor, size: int) -> Tensor:
    """sum_i share_i 1{lo_i <= j < hi_i} by a difference array; unused units are exactly 0."""
    marks = torch.zeros(size + 1, dtype=torch.float64)
    marks.index_add_(0, lo, share)
    marks.index_add_(0, hi, -share)
    cover = torch.zeros(size + 1, dtype=torch.int64)
    cover.index_add_(0, lo, torch.ones_like(lo))
    cover.index_add_(0, hi, -torch.ones_like(hi))
    used = cover.cumsum(dim=0)[:size] > 0
    return torch.where(used, marks.cumsum(dim=0)[:size], 0.0)


def match_sorted(query: Tensor, pool: Tensor, values: Tensor, m: int, *,
                 own: Tensor | None = None, usage: bool = True) -> Matches:
    """One-dimensional matching of ``query`` [nq] to the sorted ``pool`` [np].

    ``values`` [np, c] are pool columns in sorted order (centre them before the
    call: window sums are differences of prefix sums). With ``own`` the query
    unit is excluded from its own matched set.
    """
    lo, hi, reach = windows_sorted(query, pool, m, own)
    prefix = _prefix(values)
    sums = prefix[hi] - prefix[lo]
    count = (hi - lo).to(torch.float64)
    if own is not None:
        sums = sums - values[own]
        count = count - 1
    share, share2 = None, None
    if usage and own is None:
        share = _usage(lo, hi, count.reciprocal(), pool.numel())
        share2 = _usage(lo, hi, count.reciprocal().square(), pool.numel())
    return Matches(count, sums, share, share2, reach)


def match_blocks(query: Tensor, pool: Tensor, values: Tensor, m: int, *,
                 own: Tensor | None = None, usage: bool = True) -> Matches:
    """Matching in several dimensions: ``query`` [nq, k] to ``pool`` [np, k].

    Distances are Euclidean in the given coordinates (whiten the covariates
    first for the Mahalanobis or inverse-variance metric). ``own`` [nq] are
    the pool positions of queries that belong to the pool (excluded from their
    own matched sets).
    """
    nq, size = query.shape[0], pool.shape[0]
    available = size - (0 if own is None else 1)
    if m > available:
        raise KernelError("insufficient_observations",
                          f"Matching needs at least {m} candidate(s) but only {available} exist.")
    if nq * size > MAX_PAIRS:
        raise KernelError("matching_too_large",
                          f"Covariate matching would evaluate {nq * size:.3g} distances (limit "
                          f"{MAX_PAIRS:.1g}). Use method='psmatch' (sorted one-dimensional "
                          "search) or a smaller sample.")
    block = max(1, _BLOCK_ELEMENTS // max(size, 1))
    count = torch.empty(nq, dtype=torch.float64)
    sums = torch.empty((nq, values.shape[1]), dtype=torch.float64)
    reach = torch.empty(nq, dtype=torch.float64)
    share = torch.zeros(size, dtype=torch.float64) if usage and own is None else None
    share2 = torch.zeros(size, dtype=torch.float64) if share is not None else None
    for start in range(0, nq, block):
        stop = min(nq, start + block)
        # Exact differences (no |a|^2 + |b|^2 - 2ab expansion), so duplicated rows are at
        # distance exactly zero and equal coordinate differences give equal distances.
        distance = torch.cdist(query[start:stop], pool,
                               compute_mode="donot_use_mm_for_euclid_dist")
        if own is not None:
            distance[torch.arange(stop - start), own[start:stop]] = float("inf")
        if m <= _TOPK_LIMIT:
            kth = torch.topk(distance, m, dim=1, largest=False).values[:, -1]
        else:
            kth = torch.kthvalue(distance, m, dim=1).values
        # matched pairs (ties with the m-th distance included), accumulated sparsely
        rows, columns = torch.nonzero(distance <= kth[:, None], as_tuple=True)
        size_j = torch.bincount(rows, minlength=stop - start).to(torch.float64)
        count[start:stop] = size_j
        sums[start:stop] = torch.zeros((stop - start, values.shape[1]), dtype=torch.float64
                                       ).index_add_(0, rows, values[columns])
        reach[start:stop] = kth
        if share is not None:
            inverse = size_j.reciprocal()[rows]
            share.index_add_(0, columns, inverse)
            share2.index_add_(0, columns, inverse.square())
    return Matches(count, sums, share, share2, reach)
