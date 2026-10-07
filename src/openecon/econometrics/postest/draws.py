"""Tensor kernels of the resampling procedures: index draws and replicate summaries.

All draws are O(number drawn) and come from one ``torch.Generator``, so a seed
reproduces every replicate exactly. Units (rows or clusters) are grouped by
stratum once; a replicate then needs one uniform vector:

- equal-probability draws (observations or clusters): slot ``j`` of stratum
  ``h`` takes unit ``order[start_h + floor(u_j n_h)]``;
- frequency-weighted observations: ``N_h`` units are drawn with probability
  proportional to the weights by inverting the cumulative weights of the
  stratum (``searchsorted``), and the draw counts become the new frequency
  weights. This is exactly the bootstrap of the expanded data set.

Replicate summaries implement Stata's bootstrap and jackknife formulas (see
``bootstrap_summary`` and ``jackknife_covariance``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_cdf, normal_ppf


@dataclass
class Strata:
    """Units grouped by stratum: ``order`` lists units stratum by stratum."""

    order: Tensor        # int64 [U] unit indices sorted by stratum (stable)
    start: Tensor        # int64 [H] first position of each stratum in ``order``
    count: Tensor        # int64 [H] units per stratum

    @classmethod
    def build(cls, codes: Tensor, n_strata: int) -> Strata:
        order = torch.sort(codes, stable=True).indices
        count = torch.bincount(codes, minlength=n_strata)
        start = torch.cumsum(count, 0) - count
        return cls(order, start, count)


def draw_units(strata: Strata, sizes: Tensor, generator: torch.Generator) -> Tensor:
    """Units drawn with replacement and equal probability, ``sizes[h]`` from stratum h."""
    slots = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    u = torch.rand(len(slots), generator=generator, dtype=torch.float64)
    offset = torch.minimum((u * strata.count[slots]).to(torch.int64), strata.count[slots] - 1)
    return strata.order[strata.start[slots] + offset]


@dataclass
class WeightedStrata:
    """Cumulative frequency weights of the rows, stratum by stratum."""

    order: Tensor        # int64 [n] rows sorted by stratum
    cumulative: Tensor   # float64 [n] running total of the sorted weights
    before: Tensor       # float64 [H] total weight of the earlier strata
    total: Tensor        # float64 [H] total weight of each stratum
    end: Tensor          # int64 [H] one past the last sorted position of each stratum

    @classmethod
    def build(cls, codes: Tensor, n_strata: int, weights: Tensor) -> WeightedStrata:
        order = torch.sort(codes, stable=True).indices
        cumulative = torch.cumsum(weights[order], 0)
        total = torch.zeros(n_strata, dtype=torch.float64).index_add_(0, codes, weights)
        before = torch.cumsum(total, 0) - total
        end = torch.cumsum(torch.bincount(codes, minlength=n_strata), 0)
        return cls(order, cumulative, before, total, end)


def draw_weighted(strata: WeightedStrata, sizes: Tensor, n_rows: int,
                  generator: torch.Generator) -> Tensor:
    """Frequency-weighted draws: the number of times each row is drawn (int64 [n])."""
    slots = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    u = torch.rand(len(slots), generator=generator, dtype=torch.float64)
    target = strata.before[slots] + u * strata.total[slots]
    position = torch.searchsorted(strata.cumulative, target, right=True)
    position = torch.minimum(position, strata.end[slots] - 1)
    return torch.bincount(strata.order[position], minlength=n_rows)


@dataclass
class Clusters:
    """Rows grouped by cluster for whole-cluster draws."""

    rows: Tensor         # int64 [n] rows sorted by cluster
    start: Tensor        # int64 [G] first sorted position of each cluster
    size: Tensor         # int64 [G] rows per cluster

    @classmethod
    def build(cls, codes: Tensor, n_clusters: int) -> Clusters:
        rows = torch.sort(codes, stable=True).indices
        size = torch.bincount(codes, minlength=n_clusters)
        return cls(rows, torch.cumsum(size, 0) - size, size)

    def expand(self, drawn: Tensor) -> tuple[Tensor, Tensor]:
        """Rows of the drawn clusters (in draw order) and the draw slot of every row."""
        lengths = self.size[drawn]
        slot = torch.repeat_interleave(torch.arange(len(drawn)), lengths)
        first = torch.cumsum(lengths, 0) - lengths
        within = torch.arange(int(lengths.sum())) - first[slot]
        return self.rows[self.start[drawn][slot] + within], slot


# ---- replicate summaries ----------------------------------------------------------


def stata_percentile(sorted_values: Tensor, p: float) -> float:
    """Stata's ``_pctile`` definition (empirical distribution with averaging).

    With R sorted values and P = R p: the mean of the P-th and (P+1)-th values
    when P is an integer, otherwise the ceil(P)-th value (1-based).
    """
    r = len(sorted_values)
    position = r * p
    nearest = round(position)
    if abs(position - nearest) < 1e-9 * max(1.0, position):
        lower = min(max(nearest, 1), r)
        upper = min(max(nearest + 1, 1), r)
        return float((sorted_values[lower - 1] + sorted_values[upper - 1]) / 2)
    return float(sorted_values[min(max(math.ceil(position), 1), r) - 1])


def bootstrap_covariance(replicates: Tensor) -> Tensor:
    """Sample covariance of the replicates around their mean, divisor R - 1 (Stata)."""
    r = replicates.shape[0]
    if r < 2:
        raise KernelError("bootstrap_failed", "At least two successful replicates are needed.")
    centered = replicates - replicates.mean(dim=0)
    return (centered.T @ centered) / (r - 1)


def bootstrap_intervals(replicates: Tensor, observed: Tensor, alpha: float,
                        kind: str) -> list[tuple[float | None, float | None]]:
    """Percentile or bias-corrected percentile intervals per parameter.

    percentile: the alpha/2 and 1 - alpha/2 percentiles of the replicates.
    bc (Efron's bias-corrected): z0 = Phi^-1(#{b_r <= b} / R),
    p1 = Phi(2 z0 - z), p2 = Phi(2 z0 + z) with z = Phi^-1(1 - alpha/2); the
    interval is the (p1, p2) percentiles. When every (or no) replicate lies at
    or below the estimate z0 is infinite and the interval is undefined (None).
    """
    ordered = torch.sort(replicates, dim=0).values
    z = normal_ppf(1 - alpha / 2)
    intervals: list[tuple[float | None, float | None]] = []
    for j in range(replicates.shape[1]):
        column = ordered[:, j]
        if kind == "percentile":
            intervals.append((stata_percentile(column, alpha / 2),
                              stata_percentile(column, 1 - alpha / 2)))
            continue
        share = float((replicates[:, j] <= observed[j]).to(torch.float64).mean())
        if share <= 0.0 or share >= 1.0:
            intervals.append((None, None))
            continue
        z0 = normal_ppf(share)
        p1, p2 = normal_cdf(2 * z0 - z), normal_cdf(2 * z0 + z)
        intervals.append((stata_percentile(column, p1), stata_percentile(column, p2)))
    return intervals


def jackknife_covariance(replicates: Tensor, multiplicity: Tensor) -> tuple[Tensor, Tensor, float]:
    """Jackknife covariance ``(N-1)/N sum_i f_i (b_(i) - b_bar)(b_(i) - b_bar)'``.

    ``multiplicity`` is 1 per deleted unit, or the frequency weight of a row
    when one replicated observation is deleted at a time (all its copies give
    the same replicate). ``b_bar = sum f_i b_(i) / N`` with ``N = sum f_i``.
    Returns the covariance, ``b_bar`` and ``N``.
    """
    total = float(multiplicity.sum())
    if total < 2:
        raise KernelError("jackknife_failed", "At least two successful replicates are needed.")
    mean = (multiplicity[:, None] * replicates).sum(dim=0) / total
    centered = replicates - mean
    covariance = (centered.T @ (centered * multiplicity[:, None])) * ((total - 1) / total)
    return covariance, mean, total
