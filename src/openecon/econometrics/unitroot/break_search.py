"""Batched least squares for unit-root tests that search over break dates.

Lee--Strazicich and Clemente--Montanes--Reyes statistics run one augmented
regression per candidate break configuration and lag order. Those regressions
share their size, so they are solved together: a batch of designs [B, n, p] is
factored by one batched QR (``torch.linalg.qr``), never through X'X.

An impulse dummy that equals one in a single row of a regression fits that row
exactly; every other coefficient, the residual sum of squares and the residual
degrees of freedom equal those of the regression without the row. Impulse
dummies are therefore applied as row masks: masked rows are zeroed in the
design and the target and subtracted from the observation count. Duplicate
impulses (two dummies on the same row) collapse to one mask, which spans the
same column space as the duplicated dummy columns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# Rows of a batch processed together are limited to about this many elements.
BLOCK_ELEMENTS = 1 << 22
# A pivot below this fraction of its column norm makes a regression rank deficient.
RANK_TOLERANCE = 1e-10


@dataclass
class BatchFit:
    """Selected-coefficient t ratios of a batch of regressions."""

    t: Tensor          # [B, c] t ratios of the requested columns
    beta: Tensor       # [B, c] their coefficients
    ssr: Tensor        # [B]
    df: Tensor         # [B] residual degrees of freedom
    valid: Tensor      # [B] full rank, positive df and residual above rounding
    exact: Tensor      # [B] residual at the rounding level of the target


@torch.no_grad()
def batch_ols(design: Tensor, target: Tensor, mask: Tensor | None,
              columns: tuple[int, ...]) -> BatchFit:
    """OLS of ``target`` on ``design`` for every batch member; t ratios of ``columns``.

    design [B, n, p]; target [B, n] (or [n], shared); mask [B, n] with 1 for
    used rows and 0 for rows absorbed by impulse dummies (or None). Classical
    standard errors s^2 (X'X)^{-1} with s^2 = SSR / (rows - p).
    """
    batch, rows, width = design.shape
    if target.dim() == 1:
        target = target.expand(batch, rows)
    if mask is not None:
        design = design * mask[:, :, None]
        target = target * mask
        used = mask.sum(dim=1)
    else:
        used = torch.full((batch,), float(rows), dtype=torch.float64)
    q, r = torch.linalg.qr(design)
    theta = (q.transpose(1, 2) @ target[:, :, None])[:, :, 0]
    resid = target - (q @ theta[:, :, None])[:, :, 0]
    ssr = resid.square().sum(dim=1)
    pivots = r.diagonal(dim1=1, dim2=2).abs()
    norms = design.square().sum(dim=1).sqrt()
    full_rank = (pivots > RANK_TOLERANCE * norms.clamp_min(1e-300)).all(dim=1)
    eye = torch.eye(width, dtype=torch.float64).expand(batch, width, width)
    safe = torch.where(full_rank[:, None, None], r, eye)
    inverse = torch.linalg.solve_triangular(safe, eye, upper=True)
    beta = (inverse @ theta[:, :, None])[:, :, 0]
    df = used - width
    s2 = ssr / df.clamp_min(1.0)
    picked = list(columns)
    spread = inverse[:, picked, :].square().sum(dim=2)          # diag of (X'X)^{-1}
    se = (s2[:, None] * spread).sqrt()
    scale = target.square().sum(dim=1)
    exact = ~(ssr > 1e-22 * scale)
    valid = full_rank & (df > 0) & ~exact & torch.isfinite(se).all(dim=1) & (se > 0).all(dim=1)
    t = torch.where(valid[:, None], beta[:, picked] / torch.where(valid[:, None], se, 1.0),
                    torch.full_like(se, float("nan")))
    return BatchFit(t=t, beta=beta[:, picked], ssr=ssr, df=df, valid=valid, exact=exact)


def chunk_size(rows: int, width: int) -> int:
    """Batch members per block so that one design block holds about BLOCK_ELEMENTS."""
    return max(1, BLOCK_ELEMENTS // max(1, rows * width))


def gauss_round(value: float) -> int:
    """GAUSS/RATS ``round``: halves away from zero (Python rounds halves to even)."""
    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


def general_to_specific(last_t: Tensor, threshold: float, *, inclusive: bool) -> Tensor:
    """Largest lag k >= 1 whose last augmentation t ratio is significant, else 0.

    ``last_t`` [B, K + 1] holds the t ratio of lag k in the regression with k
    lags (column 0 unused). Starting from K, a lag is kept when |t| > threshold
    (``inclusive=False``, Lee and Strazicich's GAUSS code) or |t| >= threshold
    (``inclusive=True``, Perron and Ng's RATS procedures).
    """
    batch, width = last_t.shape
    order = torch.arange(width, dtype=torch.int64).expand(batch, width)
    magnitude = last_t.abs()
    hit = magnitude >= threshold if inclusive else magnitude > threshold
    hit[:, 0] = False
    hit = hit & torch.isfinite(last_t)
    chosen = torch.where(hit, order, torch.zeros_like(order))
    return chosen.amax(dim=1)
