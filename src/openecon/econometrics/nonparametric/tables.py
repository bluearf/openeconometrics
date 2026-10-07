"""Chi-square and exact tests on a two-way table of counts (float64 tensors).

A table is an [R, C] tensor of nonnegative counts (frequency weights may make
them non-integer). Rows and columns with a zero total carry no information
and are removed by ``trim`` before any statistic is computed, as SPSS
CROSSTABS and Stata ``tabulate`` do.

* Pearson chi2 = sum (O - E)^2 / E and the likelihood ratio
  G2 = 2 sum O ln(O / E), both on (R-1)(C-1) degrees of freedom.
* Yates' continuity-corrected chi2 for 2x2 tables:
  N (max(0, |ad - bc| - N/2))^2 / (r1 r2 c1 c2).
* Linear-by-linear association (Mantel-Haenszel): (N - 1) r^2 on 1 df, r the
  Pearson correlation of the row and column scores.
* Fisher's exact test. 2x2: the hypergeometric distribution of the (1, 1)
  cell; the two-sided p sums the tables that are no more probable than the
  observed one (the rule of SPSS, Stata, R and SciPy). r x c (Fisher-Freeman-
  Halton): the same rule over every table with the observed margins, by
  breadth-first enumeration when there are few enough tables and otherwise by
  Monte Carlo sampling of tables with fixed margins (random permutations).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf, normal_isf
from openecon.econometrics.nonparametric.exact import hypergeometric

FLOAT = torch.float64
# Tables that are no more probable than the observed one, up to this relative slack.
_RELATIVE = 1e-7
MAX_ENUMERATION = 3_000_000      # partial tables held at once during exact enumeration
MAX_MONTE_CARLO = 50_000_000     # N * reps a Monte Carlo exact test may use


def trim(counts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(table without empty rows/columns, kept row mask, kept column mask)."""
    rows, columns = counts.sum(1) > 0, counts.sum(0) > 0
    return counts[rows][:, columns], rows, columns


def expected_counts(counts: Tensor) -> Tensor:
    return torch.outer(counts.sum(1), counts.sum(0)) / counts.sum()


def pearson_chi2(counts: Tensor) -> tuple[float, int]:
    """(Pearson chi2, df) of a table without empty margins."""
    expected = expected_counts(counts)
    statistic = float(((counts - expected) ** 2 / expected).sum())
    return statistic, (counts.shape[0] - 1) * (counts.shape[1] - 1)


def likelihood_ratio(counts: Tensor) -> float:
    expected = expected_counts(counts)
    return max(0.0, 2.0 * float(torch.xlogy(counts, counts / expected).sum()))


def yates_chi2(counts: Tensor) -> float:
    """Continuity-corrected chi2 of a 2x2 table."""
    total = float(counts.sum())
    determinant = abs(float(counts[0, 0] * counts[1, 1] - counts[0, 1] * counts[1, 0]))
    margins = float(counts.sum(1).prod() * counts.sum(0).prod())
    return total * max(0.0, determinant - total / 2.0) ** 2 / margins


def linear_by_linear(counts: Tensor, row_scores: Tensor, column_scores: Tensor) -> float | None:
    """(N - 1) r^2 for the given scores; None when a score has no variation."""
    total = counts.sum()
    rows, columns = counts.sum(1), counts.sum(0)
    x = row_scores - (rows * row_scores).sum() / total
    y = column_scores - (columns * column_scores).sum() / total
    sxx, syy = float((rows * x * x).sum()), float((columns * y * y).sum())
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = float((x[:, None] * counts * y[None, :]).sum())
    return (float(total) - 1.0) * sxy * sxy / (sxx * syy)


def fisher_2x2(counts: Tensor) -> tuple[float, float, float]:
    """(two-sided, P(A <= a), P(A >= a)) for the (1, 1) cell a of an integer 2x2 table."""
    a, row1 = int(round(float(counts[0, 0]))), int(round(float(counts[0].sum())))
    col1, total = int(round(float(counts[:, 0].sum()))), int(round(float(counts.sum())))
    low, pmf = hypergeometric(row1, col1, total)
    observed = a - low
    if observed < 0:                 # beyond the tabulated window: probability below 1e-300
        return 0.0, 0.0, 1.0
    if observed >= pmf.numel():
        return 0.0, 1.0, 0.0
    two_sided = float(pmf[pmf <= pmf[observed] * (1.0 + _RELATIVE)].sum())
    return min(1.0, two_sided), min(1.0, float(pmf[:observed + 1].sum())), \
        min(1.0, float(pmf[observed:].sum()))


def is_integer_table(counts: Tensor) -> bool:
    return bool((counts == counts.round()).all())


def _log_probability_constant(rows: Tensor, columns: Tensor) -> float:
    total = rows.sum()
    return float(torch.lgamma(rows + 1).sum() + torch.lgamma(columns + 1).sum()
                 - torch.lgamma(total + 1))


def table_log_probability(counts: Tensor) -> float:
    """log P(table | margins) under independence (multiple hypergeometric)."""
    return _log_probability_constant(counts.sum(1), counts.sum(0)) \
        - float(torch.lgamma(counts + 1).sum())


def fisher_enumerate(counts: Tensor) -> float | None:
    """Exact Fisher-Freeman-Halton p of an integer r x c table, or None when too large.

    Tables are built column by column. A state is the vector of row totals
    still to be allocated together with the accumulated -sum ln(f_ij!); each
    step expands every state by the admissible values of one cell (vectorized
    with repeat_interleave), so the work is proportional to the number of
    partial tables and no Python loop runs over tables.
    """
    if counts.shape[0] > counts.shape[1]:
        counts = counts.T
    rows = counts.sum(1).round().to(torch.int64)
    columns = counts.sum(0).round().to(torch.int64)
    r, c = counts.shape
    observed = table_log_probability(counts)
    constant = _log_probability_constant(rows.to(FLOAT), columns.to(FLOAT))
    remaining = rows[None, :].clone()                    # [states, r]
    log_p = torch.zeros(1, dtype=FLOAT)
    for j in range(c - 1):
        left = torch.full((remaining.shape[0],), int(columns[j]), dtype=torch.int64)
        later = columns[j + 1:].sum()                    # still to place after this column
        for i in range(r - 1):
            # Cell (i, j) = x: x <= min(row remainder, column remainder), and the rows
            # below must be able to absorb the rest of the column.
            below = remaining[:, i + 1:].sum(1)
            lowest = (left - below).clamp_min(0)
            # the row must keep no more than what later columns can still take
            lowest = torch.maximum(lowest, remaining[:, i] - later)
            highest = torch.minimum(remaining[:, i], left)
            width = (highest - lowest + 1).clamp_min(0)
            states = int(width.sum())
            if states > MAX_ENUMERATION:
                return None
            if states == 0:
                return None
            parent = torch.repeat_interleave(torch.arange(width.numel()), width)
            first = torch.cumsum(width, 0) - width
            value = lowest[parent] + (torch.arange(states) - first[parent])
            remaining = remaining[parent]
            remaining[:, i] -= value
            left = left[parent] - value
            log_p = log_p[parent] - torch.lgamma(value.to(FLOAT) + 1)
        # last row of the column takes what is left
        valid = left <= remaining[:, r - 1]
        remaining, left, log_p = remaining[valid], left[valid], log_p[valid]
        remaining[:, r - 1] -= left
        log_p = log_p - torch.lgamma(left.to(FLOAT) + 1)
    # the last column is determined by the remaining row totals
    log_p = log_p - torch.lgamma(remaining.to(FLOAT) + 1).sum(1) + constant
    slack = math.log1p(_RELATIVE)
    return min(1.0, float(torch.exp(log_p[log_p <= observed + slack]).sum()))


def fisher_monte_carlo(counts: Tensor, reps: int, seed: int,
                       level: float = 0.99) -> tuple[float, float, float]:
    """(p, ci_low, ci_high) of the exact test by Monte Carlo sampling of tables.

    Each replication permutes the column labels of the N observations, which
    samples a table with the observed margins under independence; the p-value
    is the share of sampled tables that are no more probable than the observed
    one, with a normal confidence interval at ``level`` (SPSS reports 99%; when
    the share is 0 or 1 the open end is the exact binomial bound).
    """
    total = int(round(float(counts.sum())))
    if total * reps > MAX_MONTE_CARLO:
        raise KernelError("exact_unavailable", f"A Monte Carlo exact test with {reps} "
                          f"replications of {total} observations is too large. Lower "
                          "exact_reps or use the asymptotic tests (exact=False).")
    r, c = counts.shape
    integer = counts.round().to(torch.int64)
    row_of = torch.repeat_interleave(torch.arange(r), integer.sum(1))
    column_of = torch.repeat_interleave(torch.arange(c), integer.sum(0))
    observed = float(torch.lgamma(counts + 1).sum())
    generator = torch.Generator().manual_seed(seed)
    batch = max(1, min(reps, 2_000_000 // max(total, 1)))
    extreme, done = 0, 0
    while done < reps:
        size = min(batch, reps - done)
        order = torch.argsort(torch.rand((size, total), generator=generator, dtype=FLOAT), dim=1)
        cell = row_of[None, :] * c + column_of[order] + torch.arange(size)[:, None] * (r * c)
        tables = torch.bincount(cell.reshape(-1), minlength=size * r * c).reshape(size, r * c)
        statistic = torch.lgamma(tables.to(FLOAT) + 1).sum(1)
        # less probable  <=>  larger sum of ln(f_ij!)
        extreme += int((statistic >= observed - 1e-7).sum())
        done += size
    p = extreme / reps
    half = normal_isf((1.0 - level) / 2.0) * math.sqrt(p * (1.0 - p) / reps)
    low, high = max(0.0, p - half), min(1.0, p + half)
    if extreme == 0:                       # exact one-sided binomial bound when no table counted
        high = 1.0 - (1.0 - level) ** (1.0 / reps)
    elif extreme == reps:
        low = (1.0 - level) ** (1.0 / reps)
    return p, low, high


def chi2_p(statistic: float, df: int) -> float:
    return chi2_sf(statistic, df) if df > 0 else 1.0
