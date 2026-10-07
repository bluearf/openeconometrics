"""Local-polynomial kernels of rdrobust (Calonico, Cattaneo and Titiunik 2014).

Each side of the cutoff is handled separately on its observations sorted by
the running variable, ``dx = x - c``. Kernel weights carry the 1/h factor of
``rdrobust_kweight``: ``K(u)/h`` with u = dx/h and K the triangular
``(1 - |u|)``, Epanechnikov ``0.75 (1 - u^2)`` or uniform ``0.5`` kernel on
|u| <= 1. Observations with positive weight form a contiguous slice of the
sorted side (closest to the cutoff), so every kernel works on a slice.

Residuals for the variance (``rdrobust_res``):

* ``nn`` (default): nearest-neighbour residuals
  ``sqrt(J/(J+1)) (y_i - mean of the J matched y)``; the matched set of i holds
  all other observations with the same running value and is enlarged by whole
  blocks of tied values, nearest first (both sides when equidistant), until it
  has at least ``min(nnmatch, n - 1)`` members. The search runs on the
  distinct values of the slice, one vectorized step per added block.
* ``hc0`` .. ``hc3``: fitted residuals scaled by 1, sqrt(n/(n-d)),
  1/sqrt(1-h_ii) or 1/(1-h_ii), h_ii the weighted leverage of the order-p fit.

The meat of the sandwich is ``sum_i (r_i s' e_i)^2 x_i x_i'`` over rows
``x_i`` (kernel-weighted design rows) and residual vectors ``e_i`` (outcome,
and treatment for fuzzy designs) combined by ``s``; with clusters it is
``sum_g t_g t_g'`` times ((n-1)/(n-k)) (G/(G-1)).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

KERNELS = ("triangular", "epanechnikov", "uniform")


def covariate_adjustment(gram: Tensor, responses: int) -> Tensor:
    """Common-slope adjustment from polynomial-residualized weighted crossproducts."""
    result = torch.eye(gram.shape[0], responses, dtype=torch.float64)
    if gram.shape[0] > responses:
        zz = gram[responses:, responses:]
        scale = zz.diagonal().clamp_min(0).sqrt()
        if bool((scale <= 0).any()):
            raise KernelError("singular_covariates", "RD covariates must vary after local polynomial projection.")
        normalized = zz / scale[:, None] / scale[None, :]
        if float(torch.linalg.eigvalsh(normalized).min()) < 1e-10:
            raise KernelError("singular_covariates", "RD covariates are collinear after local polynomial projection.")
        result[responses:] = -torch.linalg.solve(zz, gram[responses:, :responses])
    return result


def kernel_weights(dx: Tensor, h: float, kernel: str) -> Tensor:
    """``rdrobust_kweight``: K(dx/h)/h (zero outside |u| <= 1)."""
    u = dx / h
    inside = u.abs() <= 1
    if kernel == "triangular":
        k = (1 - u.abs()).clamp_min(0)
    elif kernel == "epanechnikov":
        k = 0.75 * (1 - u.square()).clamp_min(0)
    elif kernel == "uniform":
        k = torch.full_like(u, 0.5)
    else:
        raise KernelError("invalid_kernel", f"Unknown kernel '{kernel}'.")
    return torch.where(inside, k, torch.zeros_like(k)) / h


def powers(dx: Tensor, order: int) -> Tensor:
    """[1, dx, dx^2, ..., dx^order]."""
    return dx[:, None] ** torch.arange(order + 1, dtype=torch.float64)


@dataclass
class Side:
    """One side of the cutoff, sorted by the running variable."""

    dx: Tensor                 # x - c, ascending
    y: Tensor                  # [n, m]: outcome (and treatment for fuzzy designs)
    cluster: Tensor | None     # int64 codes
    values: Tensor             # distinct dx values (ascending)
    starts: Tensor             # first sorted position of each distinct value
    counts: Tensor             # observations per distinct value
    block: Tensor              # distinct-value index of every observation
    weights: Tensor | None = None
    responses: int = 1

    @property
    def n(self) -> int:
        return self.dx.numel()


def make_side(dx: Tensor, y: Tensor, cluster: Tensor | None, weights=None, responses=1) -> Side:
    order = torch.argsort(dx, stable=True)
    dx, y = dx[order], y[order]
    cluster = None if cluster is None else cluster[order]
    values, block, counts = torch.unique_consecutive(dx, return_inverse=True,
                                                     return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    return Side(dx, y, cluster, values, starts, counts, block,
                None if weights is None else weights[order], responses)


def nn_residuals(side: Side, part: slice, matches: int) -> Tensor:
    """Nearest-neighbour residuals of the observations ``part`` (contiguous) of a side."""
    first, last = int(side.block[part.start]), int(side.block[part.stop - 1]) + 1
    values = side.values[first:last]
    counts = side.counts[first:last]
    blocks = values.numel()
    n = part.stop - part.start
    target = min(matches, n - 1)
    if target < 1:
        raise KernelError("insufficient_observations", "Nearest-neighbour residuals need at "
                          "least two observations within the bandwidth on each side.")
    cum = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    left = torch.arange(blocks)             # window [left, right] of distinct values
    right = torch.arange(blocks)
    for _ in range(target):
        others = cum[right + 1] - cum[left] - 1
        active = others < target
        if not bool(active.any()):
            break
        has_left, has_right = left > 0, right < blocks - 1
        dl = torch.where(has_left, values - values[(left - 1).clamp_min(0)],
                         torch.full_like(values, float("inf")))
        dr = torch.where(has_right, values[(right + 1).clamp(max=blocks - 1)] - values,
                         torch.full_like(values, float("inf")))
        tied = has_left & has_right & ((dl-dr).abs() <= torch.maximum(dl,dr)*math.sqrt(torch.finfo(torch.float64).eps))
        grow_left = active & has_left & ((dl < dr) | tied)
        grow_right = active & has_right & ((dr < dl) | tied)
        left = left - grow_left.to(torch.int64)
        right = right + grow_right.to(torch.int64)
    local_block = side.block[part] - first
    lo, hi = cum[left[local_block]], cum[right[local_block] + 1]
    y = side.y[part]
    centred = y - y.mean(dim=0)
    prefix = torch.cat([torch.zeros((1, y.shape[1]), dtype=torch.float64), centred.cumsum(0)])
    size = (hi - lo - 1).to(torch.float64)[:, None]
    mean_others = (prefix[hi] - prefix[lo] - centred) / size
    return (size / (size + 1)).sqrt() * (centred - mean_others)


@dataclass
class LocalFit:
    part: slice                # observations with positive weight
    design: Tensor             # [n_part, order + 1]
    weights: Tensor            # kernel weights K(u)/h of the slice
    inverse: Tensor            # (R'WR)^-1
    beta: Tensor               # [order + 1, m]


def weighted_part(side: Side, h: float, kernel: str) -> tuple[slice, Tensor]:
    w = kernel_weights(side.dx, h, kernel)
    if side.weights is not None:
        w = w * side.weights
    positive = torch.nonzero(w > 0).flatten()
    if positive.numel() == 0:
        return slice(0, 0), w[:0]
    part = slice(int(positive[0]), int(positive[-1]) + 1)
    return part, w[part]


def local_fit(side: Side, part: slice, weights: Tensor, order: int, what: str) -> LocalFit:
    """Weighted least squares of the side's columns on [1, dx, ..., dx^order] (QR)."""
    from openecon.engines.linalg import least_squares

    distinct = int(side.block[part.stop - 1] - side.block[part.start]) + 1 if part.stop > \
        part.start else 0
    if distinct <= order:
        raise KernelError("insufficient_observations",
                          f"The {what} needs more than {order} distinct running-variable values "
                          f"with positive kernel weight on each side; found {distinct}. Choose "
                          "a larger bandwidth or a lower polynomial order.")
    design = powers(side.dx[part], order)
    fit = least_squares(design, side.y[part], weights, drop_collinear=False, tol=0.0)
    return LocalFit(part, design, weights, fit.xtx_inv, fit.beta)


def residuals(side: Side, fit: LocalFit, vce: str, matches: int, leverage_fit: LocalFit | None,
              columns: int) -> Tensor:
    """``rdrobust_res`` on the slice of ``fit``; ``columns`` is d of the hc1 factor."""
    if vce == "nn":
        return nn_residuals(side, fit.part, matches)
    resid = side.y[fit.part] - fit.design @ fit.beta
    n = resid.shape[0]
    if vce == "hc0":
        return resid
    if vce == "hc1":
        if n <= columns:
            raise KernelError("insufficient_observations", "HC1 needs more observations than "
                              "polynomial coefficients within the bandwidth.")
        return resid * (n / (n - columns)) ** 0.5
    source = leverage_fit or fit
    rows = source.design * source.weights[:, None]
    leverage = torch.einsum("ij,jk,ik->i", source.design, source.inverse, rows)
    if source.part != fit.part:
        full = torch.zeros(n, dtype=torch.float64)
        offset = source.part.start - fit.part.start
        full[offset:offset + leverage.numel()] = leverage
        leverage = full
    if bool((leverage >= 1).any()):
        raise KernelError("numerical_failure", "A leverage of one makes HC2/HC3 undefined; "
                          "use vce='nn' or a larger bandwidth.")
    factor = (1 - leverage).rsqrt() if vce == "hc2" else (1 - leverage).reciprocal()
    return resid * factor[:, None]


def meat(rows: Tensor, resid: Tensor, combine: Tensor, cluster: Tensor | None) -> Tensor:
    """``rdrobust_vce``: sum (s'e_i)^2 x_i x_i', or the cluster version with its factor."""
    scalar = resid @ combine
    scores = rows * scalar[:, None]
    if cluster is None:
        return scores.T @ scores
    codes, groups = torch.unique(cluster, return_inverse=True)
    count = codes.numel()
    if count < 2:
        raise KernelError("insufficient_clusters", "Cluster-robust variance needs at least two "
                          "clusters within the bandwidth on each side.")
    totals = torch.zeros((count, rows.shape[1]), dtype=torch.float64)
    totals.index_add_(0, groups, scores)
    n, k = rows.shape
    factor = (n - 1) / (n - k) * count / (count - 1)
    return factor * (totals.T @ totals)
