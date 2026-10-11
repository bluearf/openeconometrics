"""Randomization and large-sample kernels of rdlocrand (tensors only).

Assignment laws act on the units of one window, held in a canonical order
(running variable, then outcome and the other used columns) so that results do
not depend on the input row order:

* fixed margins (complete randomization): every subset of n1 of the n units is
  treated with equal probability. ``exact`` enumerates all C(n, n1) subsets in
  colex order by unranking the combinatorial number system in bounded chunks;
  ``monte_carlo`` treats the n1 units with the smallest of n uniform keys (a
  uniformly random permutation) drawn from a seeded CPU ``torch.Generator``.
* Bernoulli trials: unit i is treated with probability pi_i, independently.
  ``exact`` enumerates the 2^f assignments of the f units with 0 < pi_i < 1,
  each weighted by its probability; ``monte_carlo`` treats U_i < pi_i with
  U_i ~ U[0, 1). Assignments without a treated or without a control unit are
  inadmissible (the package records them as NA and drops them).

A p-value is the probability-weighted share of admissible assignments whose
statistic is at least as large in absolute value as the observed one, the
package's ``mean(abs(T*) >= abs(T))`` without a plus-one correction.

Statistics (vectorized over a chunk A of assignments, one row per assignment):

* ``diffmeans``: mean(Y | treated) - mean(Y | control), computed on outcomes
  centred at their mean; with kernel weights w the weighted means
  sum(A w Y)/sum(A w) - sum((1-A) w Y)/sum((1-A) w).
* ``ranksum``: (T0 - n0 (n+1)/2) / sqrt(n0 n1 s_r^2 / n) with T0 the control
  rank sum, mid-ranks for ties and s_r^2 the variance of the ranks (the
  tie-corrected permutation variance). Under fixed margins it is monotone in
  2 T0 - n0 (n+1), an integer, so comparisons are exact.
* ``ksmirnov``: D = max_v |F1(v) - F0(v)| over the distinct values, evaluated as
  the integer max |c1 n0 - c0 n1| (c1, c0: treated and control counts up to v)
  over n0 n1; comparisons use integer cross-products, so ties are exact.
* ``hotelling``: T^2 = (n0 n1/n) delta' S_pool^-1 delta. With TSS the total
  cross-product of the centred covariates, S_pool = (TSS - c delta delta')/(n-2)
  and c = n0 n1/n, so by Sherman-Morrison T^2 = c (n-2) b/(1 - c b) with
  b = delta' TSS^-1 delta: a single quadratic form per assignment.

Floating comparisons (``diffmeans``, Bernoulli ``ranksum``, ``hotelling``)
count a draw as extreme when |T*| >= |T| - tol, tol a bound on the float64
rounding of the statistic (16 n eps times its natural scale), so assignments
whose statistic equals the observed one in exact arithmetic are counted.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import f_sf, normal_cdf, normal_sf
from openecon.engines.linalg import least_squares
from openecon.econometrics.nonparametric.common import binomial_lower, row_midranks
from openecon.resources import plan_workspace

from .common import EPS, FLOAT

# Elements of one [assignments, units] chunk; every chunk buffer is linear in it.
CHUNK_ELEMENTS = 1 << 20
# Largest number of assignments enumerated exactly.
EXACT_LIMIT = 10_000_000
# Largest number of Monte Carlo draws.
MAX_REPS = 10_000_000
# Largest work (assignments x units x tested columns) of one randomization run and of
# all windows of one rdwinselect/rdsensitivity call.
MAX_WORK = 4_000_000_000
# Exact Smirnov law below this n0 n1 (R's ks.test rule), Kolmogorov limit above.
SMIRNOV_EXACT = 10_000


# ---- assignment laws -------------------------------------------------------------------------


@dataclass(frozen=True)
class Law:
    """Assignment law of one window: fixed margins (``prob`` None) or Bernoulli trials."""

    observed: Tensor               # bool [n], the realized assignment
    prob: Tensor | None = None     # float [n] treatment probabilities

    @property
    def n(self) -> int:
        return self.observed.numel()

    @property
    def fixed(self) -> bool:
        return self.prob is None

    def support(self, limit: int) -> int:
        """Number of assignments, or ``limit + 1`` when there are more."""
        if self.fixed:
            return fixed_support(self.n, int(self.observed.sum()), limit)
        free = int(((self.prob > 0) & (self.prob < 1)).sum())
        return 2 ** free if free < limit.bit_length() else limit + 1

    def plan(self, reps: int, method: str) -> tuple[str, int]:
        """('exact' | 'monte_carlo', number of assignments evaluated)."""
        return plan(self.support(max(reps, EXACT_LIMIT)), reps, method)


def fixed_support(n: int, n1: int, limit: int) -> int:
    """C(n, n1), or ``limit + 1`` when it is larger.

    Capped so that large windows never build C(n, n1) as a huge integer
    (C(10^6, 5 10^5) alone takes seconds and has 300,000 digits).
    """
    m = min(n1, n - n1)
    count = 1
    for i in range(1, m + 1):  # count = C(n - m + i, i), an integer at every step
        count = count * (n - m + i) // i
        if count > limit:
            return limit + 1
    return count


def plan(support: int, reps: int, method: str) -> tuple[str, int]:
    """('exact' | 'monte_carlo', assignments) for a law with ``support`` assignments
    (capped above max(reps, EXACT_LIMIT))."""
    if method == "exact" or (method == "auto" and support <= reps):
        if support > EXACT_LIMIT:
            raise AnalysisError("work_budget_exceeded", f"Exact enumeration needs more than "
                                f"{EXACT_LIMIT:,} assignments; use method='monte_carlo' or a "
                                "smaller window.")
        return "exact", support
    return "monte_carlo", reps


def check_work(work: int, what: str = "Randomization inference") -> None:
    """Refuse a call whose assignment-by-unit work exceeds MAX_WORK (before any draw)."""
    if work > MAX_WORK:
        raise AnalysisError("work_budget_exceeded", f"{what} needs about {work:,} unit evaluations, "
                            f"above the budget of {MAX_WORK:,}; reduce reps, the windows, the grid "
                            "or the number of tested columns.")


def _colex_tables(n: int, m: int, limit: int) -> Tensor:
    """[m, n] tables C(c, i), i = 1..m, c = 0..n-1, clamped at ``limit`` (monotone in c)."""
    tables = torch.empty((m, n), dtype=torch.int64)
    row = torch.ones(n, dtype=torch.int64)
    for i in range(m):
        shifted = torch.zeros(n, dtype=torch.int64)
        shifted[1:] = torch.cumsum(row, 0)[:-1]
        row = shifted.clamp(max=limit)
        tables[i] = row
    return tables


def _fixed_exact(n: int, n1: int, total: int, rows: int) -> Iterator[tuple[Tensor, None]]:
    m = min(n1, n - n1)
    tables = _colex_tables(n, m, total + 1)
    for start in range(0, total, rows):
        rank = torch.arange(start, min(start + rows, total), dtype=torch.int64)
        chosen = torch.empty((rank.numel(), m), dtype=torch.int64)
        for i in range(m, 0, -1):
            table = tables[i - 1]
            position = torch.searchsorted(table, rank, right=True) - 1
            chosen[:, i - 1] = position
            rank = rank - table[position]
        block = torch.zeros((chosen.shape[0], n), dtype=torch.bool)
        block.scatter_(1, chosen, True)
        yield (block if m == n1 else ~block), None


def _fixed_draws(n: int, n1: int, reps: int, rows: int, generator: torch.Generator
                 ) -> Iterator[tuple[Tensor, None]]:
    for start in range(0, reps, rows):
        size = min(rows, reps - start)
        keys = torch.rand((size, n), generator=generator, dtype=FLOAT)
        # The n1 smallest keys by selection (linear time; a full argsort was ~7x slower).
        chosen = torch.topk(keys, n1, dim=1, largest=False, sorted=False).indices
        block = torch.zeros((size, n), dtype=torch.bool)
        block.scatter_(1, chosen, True)
        yield block, None


def _bernoulli_exact(prob: Tensor, rows: int) -> Iterator[tuple[Tensor, Tensor]]:
    free = torch.nonzero((prob > 0) & (prob < 1)).flatten()
    base = prob >= 1
    p = prob[free]
    logit = torch.log(p) - torch.log1p(-p)
    offset = float(torch.log1p(-p).sum())
    bits = 2 ** torch.arange(free.numel(), dtype=torch.int64)
    total = 2 ** free.numel()
    for start in range(0, total, rows):
        rank = torch.arange(start, min(start + rows, total), dtype=torch.int64)
        chosen = (rank[:, None] & bits) != 0
        block = base.expand(rank.numel(), -1).clone()
        block[:, free] = chosen
        yield block, torch.exp(chosen.to(FLOAT) @ logit + offset)


def _bernoulli_draws(prob: Tensor, reps: int, rows: int, generator: torch.Generator
                     ) -> Iterator[tuple[Tensor, None]]:
    for start in range(0, reps, rows):
        size = min(rows, reps - start)
        yield torch.rand((size, prob.numel()), generator=generator, dtype=FLOAT) < prob, None


def assignments(law: Law, kind: str, total: int, seed: int, width: int
                ) -> Iterator[tuple[Tensor, Tensor | None]]:
    """Chunks (A [c, n] bool, probability weights [c] or None) of the planned assignments."""
    n = law.n
    rows = max(1, CHUNK_ELEMENTS // max(n, width, 1))
    n1 = int(law.observed.sum())
    if kind == "exact":
        return _fixed_exact(n, n1, total, rows) if law.fixed else _bernoulli_exact(law.prob, rows)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if law.fixed:
        return _fixed_draws(n, n1, total, rows, generator)
    return _bernoulli_draws(law.prob, total, rows, generator)


# ---- statistics --------------------------------------------------------------------------------


def _centre(y: Tensor, weights: Tensor | None) -> Tensor:
    """Two-pass (weighted) centring: the centred columns sum to zero up to rounding."""
    for _ in range(2):
        if weights is None:
            y = y - y.mean(0)
        else:
            y = y - (weights[:, None] * y).sum(0) / weights.sum()
    return y


class DiffMeans:
    """Difference in (kernel-weighted) means, one column per outcome."""

    name = "diffmeans"

    def __init__(self, y: Tensor, observed: Tensor, weights: Tensor | None = None):
        self.n = y.shape[0]
        self.weights = weights
        self.yc = _centre(y, weights)
        self.total = self.yc.sum(0)
        if weights is not None:
            self.wy = weights[:, None] * self.yc
            self.w_total, self.wy_total = weights.sum(), self.wy.sum(0)
        self.tol = 16 * self.n * EPS * self.yc.abs().amax(0)
        self.observed = self.values(observed[None])[0]
        if not bool(torch.isfinite(self.observed).all()):
            raise AnalysisError("numerical_failure", "The observed difference in weighted means is "
                                "undefined: a side of the window has zero total kernel weight.")

    def values(self, a: Tensor) -> Tensor:
        af = a.to(FLOAT)
        if self.weights is None:
            n1 = af.sum(1, keepdim=True)
            s1 = af @ self.yc
            return s1 / n1 - (self.total - s1) / (self.n - n1)
        w1 = af @ self.weights
        s1 = af @ self.wy
        return s1 / w1[:, None] - (self.wy_total - s1) / (self.w_total - w1)[:, None]

    def compare(self, a: Tensor) -> tuple[Tensor, Tensor]:
        values = self.values(a)
        valid = torch.isfinite(values)
        return valid & (values.abs() >= self.observed.abs() - self.tol), valid


class RankSum:
    """Standardized Wilcoxon rank sum of the control units (mid-ranks, tie-corrected variance)."""

    name = "ranksum"

    def __init__(self, y: Tensor, observed: Tensor, fixed: bool):
        self.n = y.shape[0]
        self.fixed = fixed
        ranks, _ = row_midranks(y.T.contiguous())
        ranks = ranks.T.contiguous()
        self.r2 = 2 * ranks
        self.total2 = self.r2.sum(0)
        self.variance = ranks.var(0, unbiased=True)
        self.observed_dev = self._dev(observed[None])[0]
        self.observed = self.values(observed[None])[0]

    def _dev(self, a: Tensor) -> Tensor:
        """2 (T0 - n0 (n + 1)/2): an integer held exactly in float64."""
        af = a.to(FLOAT)
        n0 = self.n - af.sum(1, keepdim=True)
        return self.total2 - af @ self.r2 - n0 * (self.n + 1)

    def values(self, a: Tensor) -> Tensor:
        af = a.to(FLOAT)
        n1 = af.sum(1, keepdim=True)
        scale = (n1 * (self.n - n1) * self.variance / self.n).sqrt()
        # All-tied outcomes (zero rank variance) carry no evidence: z = 0. Assignments
        # without treated or control units also get scale 0 and are inadmissible anyway.
        return torch.where(scale > 0, self._dev(a) / 2 / scale.clamp_min(1e-300), torch.zeros_like(scale))

    def compare(self, a: Tensor) -> tuple[Tensor, Tensor]:
        if self.fixed:
            dev = self._dev(a)
            return dev.abs() >= self.observed_dev.abs(), torch.ones_like(dev, dtype=torch.bool)
        values = self.values(a)
        valid = torch.isfinite(values)
        observed = self.observed.abs()
        tol = 64 * EPS * observed.clamp_min(1.0)
        return valid & (values.abs() >= observed - tol), valid


class KolmogorovSmirnov:
    """Two-sample Kolmogorov-Smirnov D with exact integer numerators max |c1 n0 - c0 n1|."""

    name = "ksmirnov"

    def __init__(self, y: Tensor, observed: Tensor, fixed: bool):
        self.n, self.m = y.shape
        self.fixed = fixed
        self.order = torch.argsort(y, dim=0, stable=True)
        ordered = torch.gather(y, 0, self.order)
        self.ends = torch.ones_like(ordered, dtype=torch.bool)
        if self.n > 1:
            self.ends[:-1] = ordered[1:] != ordered[:-1]
        self.positions = torch.arange(1, self.n + 1, dtype=torch.int64)
        self.n1 = int(observed.sum())
        self.n0 = self.n - self.n1
        if not fixed and self.n ** 4 // 16 >= 2 ** 62:
            raise AnalysisError("work_budget_exceeded", "Bernoulli Kolmogorov-Smirnov comparisons "
                                "use exact int64 cross-products and accept windows of at most "
                                "100,000 observations.")
        self.observed_num = torch.stack([self._numerator(observed[None], j)[0][0]
                                         for j in range(self.m)])
        self.observed = self.observed_num.to(FLOAT) / (self.n0 * self.n1)

    def _numerator(self, a: Tensor, j: int) -> tuple[Tensor, Tensor]:
        treated = a[:, self.order[:, j]].cumsum(1)
        n1 = treated[:, -1:]
        gap = (treated * (self.n - n1) - (self.positions - treated) * n1).abs()
        return gap[:, self.ends[:, j]].amax(1), n1[:, 0]

    def compare(self, a: Tensor) -> tuple[Tensor, Tensor]:
        extreme = torch.empty((a.shape[0], self.m), dtype=torch.bool)
        valid = torch.empty_like(extreme)
        for j in range(self.m):
            numerator, n1 = self._numerator(a, j)
            ok = (n1 > 0) & (n1 < self.n)
            if self.fixed:
                extreme[:, j] = numerator >= self.observed_num[j]
            else:
                extreme[:, j] = ok & (numerator * (self.n0 * self.n1)
                                      >= self.observed_num[j] * (n1 * (self.n - n1)))
            valid[:, j] = ok
        return extreme, valid

    def values(self, a: Tensor) -> Tensor:
        columns = []
        for j in range(self.m):
            numerator, n1 = self._numerator(a, j)
            columns.append(numerator.to(FLOAT) / (n1 * (self.n - n1)).to(FLOAT))
        return torch.stack(columns, 1)


class Hotelling:
    """Hotelling's T^2 of the covariate mean differences (fixed margins only)."""

    name = "hotelling"

    def __init__(self, x: Tensor, observed: Tensor):
        self.n, self.k = x.shape
        xc = _centre(x, None)
        tss = xc.T @ xc
        scale = tss.diagonal().sqrt()
        if bool((scale <= 0).any()):
            raise AnalysisError("singular_covariance", "A covariate is constant in a window; the "
                                "Hotelling statistic needs varying, non-collinear covariates.")
        normalized = tss / scale[:, None] / scale[None, :]
        factor, info = torch.linalg.cholesky_ex(normalized)
        if int(info) != 0 or float(factor.diagonal().min()) ** 2 <= 64 * self.k * EPS:
            raise AnalysisError("singular_covariance", "The covariates are collinear in a window; "
                                "the Hotelling statistic is undefined.")
        # M = xc TSS^{-1/2} so that a = ||A' M||^2 = (A' xc) TSS^{-1} (xc' A).
        self.half = torch.linalg.solve_triangular(factor, (xc / scale).T, upper=False).T
        self.n1 = int(observed.sum())
        self.observed_a = self._a(observed[None])[0]
        self.tol = 16 * self.n * EPS * max(float(self.observed_a), float(self.half.square().sum()) / self.n)
        self.observed = self.statistic(float(self.observed_a), self.n1)

    def _a(self, a: Tensor) -> Tensor:
        return (a.to(FLOAT) @ self.half).square().sum(1)

    def statistic(self, a: float, n1: int) -> float:
        n0 = self.n - n1
        c = n0 * n1 / self.n
        b = (self.n / (n0 * n1)) ** 2 * a
        denominator = 1 - c * b
        if not denominator > 0:
            raise AnalysisError("singular_covariance", "The pooled covariance of the covariates is "
                                "singular in a window; the Hotelling statistic is undefined.")
        return c * (self.n - 2) * b / denominator

    def compare(self, a: Tensor) -> tuple[Tensor, Tensor]:
        values = self._a(a)[:, None]
        return values >= float(self.observed_a) - self.tol, torch.ones_like(values, dtype=torch.bool)


@dataclass
class Outcome:
    """Result of one randomization run."""

    kind: str                      # 'exact' or 'monte_carlo'
    draws: int                     # assignments evaluated
    p_values: list[Tensor]         # per statistic, [m]
    valid: list[Tensor]            # per statistic, [m] admissible (weighted) count
    distribution: Tensor | None    # statistic values of the kept column, one per assignment


def randomize(statistics: Sequence, law: Law, *, reps: int, seed: int, method: str,
              keep: tuple[int, int] | None = None) -> Outcome:
    """Randomization p-values of ``statistics`` under ``law`` (shared assignments).

    ``keep = (statistic index, column)`` also returns that statistic's value for
    every evaluated assignment (the randomization distribution used by the
    interference confidence interval).
    """
    kind, total = law.plan(reps, method)
    columns = sum(_width(statistic) for statistic in statistics)
    check_work(total * law.n * max(columns, 1))
    width = max(_width(statistic) for statistic in statistics)
    rows = max(1, CHUNK_ELEMENTS // max(law.n, width, 1))
    plan_workspace("rdlocrand assignment chunk", {
        "assignment_chunk": 33 * rows * law.n,
        "statistic_chunk": 24 * rows * width,
        "kept_distribution": 8 * total if keep is not None else 0,
    })
    extreme = [torch.zeros(_width(s), dtype=FLOAT) for s in statistics]
    valid = [torch.zeros(_width(s), dtype=FLOAT) for s in statistics]
    kept: list[Tensor] = []
    n = law.n
    for block, weight in assignments(law, kind, total, seed, width):
        n1 = block.sum(1)
        admissible = (n1 > 0) & (n1 < n)
        for index, statistic in enumerate(statistics):
            hit, ok = statistic.compare(block)
            ok = ok & admissible[:, None]
            hit = hit & ok
            if weight is None:
                extreme[index] += hit.sum(0).to(FLOAT)
                valid[index] += ok.sum(0).to(FLOAT)
            else:
                extreme[index] += (weight[:, None] * hit).sum(0)
                valid[index] += (weight[:, None] * ok).sum(0)
            if keep is not None and keep[0] == index:
                values = statistic.values(block)[:, keep[1]]
                kept.append(values[admissible & torch.isfinite(values)])
    p_values = []
    for hit, ok in zip(extreme, valid, strict=True):
        if bool((ok <= 0).any()):
            raise AnalysisError("no_valid_assignments", "No admissible assignment (with treated and "
                                "control units and defined statistics) was generated; increase reps "
                                "or check the assignment probabilities.")
        p_values.append((hit / ok).clamp(0, 1))
    distribution = torch.cat(kept) if keep is not None else None
    return Outcome(kind, total, p_values, valid, distribution)


def _width(statistic) -> int:
    if isinstance(statistic, DiffMeans):
        return statistic.yc.shape[1]
    if isinstance(statistic, RankSum):
        return statistic.r2.shape[1]
    if isinstance(statistic, KolmogorovSmirnov):
        return statistic.m
    return 1


# ---- large-sample tests ---------------------------------------------------------------------


def two_sided(z: float) -> float:
    return min(1.0, 2.0 * normal_sf(abs(z)))


def z_test(estimate: float, se: float) -> tuple[float | None, float | None]:
    """(z, two-sided normal p); a zero standard error gives p = 1 for a zero contrast, 0 otherwise.

    An undefined (NaN) standard error, e.g. a Neyman variance with one unit on a
    side, gives (None, None) rather than a spurious p-value.
    """
    if math.isnan(se):
        return None, None
    if se > 0:
        z = estimate / se
        return z, two_sided(z)
    return None, (1.0 if estimate == 0 else 0.0)


def neyman(y: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    """Difference in means and Neyman standard error sqrt(s1^2/n1 + s0^2/n0) per column."""
    y1, y0 = y[d], y[~d]
    variance = y1.var(0, unbiased=True) / y1.shape[0] + y0.var(0, unbiased=True) / y0.shape[0]
    return y1.mean(0) - y0.mean(0), variance.clamp_min(0).sqrt()


def power_mean(delta: float, se: float) -> float | None:
    """The package's power against delta: 1 - Phi(1.96 - delta/se) + Phi(-1.96 - delta/se)."""
    if not se > 0:
        return None
    return normal_sf(1.96 - delta / se) + normal_cdf(-1.96 - delta / se)


def power_rank(delta: float, sigma: float, n0: int, n1: int) -> float | None:
    """The package's rank-sum power: Phi(sqrt(3 n0 n1/((n0+n1+1) pi)) delta/sigma - 1.96)."""
    if not sigma > 0:
        return None
    return normal_cdf(math.sqrt(3 * n0 * n1 / ((n0 + n1 + 1) * math.pi)) * delta / sigma - 1.96)


def kolmogorov_sf(x: float) -> float:
    """P(K > x) of the Kolmogorov distribution (two series, each to double precision)."""
    if x <= 0:
        return 1.0
    if x < 1:
        total = sum(math.exp(-(2 * k - 1) ** 2 * math.pi ** 2 / (8 * x * x)) for k in range(1, 12))
        return min(1.0, max(0.0, 1 - math.sqrt(2 * math.pi) / x * total))
    total = sum((-1) ** (k - 1) * math.exp(-2 * k * k * x * x) for k in range(1, 12))
    return min(1.0, max(0.0, 2 * total))


def smirnov_sf(numerator: int, n0: int, n1: int, ends: Tensor) -> float:
    """Exact P(D >= numerator/(n0 n1)) over all C(n0 + n1, n0) splits of the pooled sample.

    Lattice paths (Smirnov) along anti-diagonals t = i + j, with i the controls
    and j the treated units among the t smallest values. The DP runs in
    probability space (a uniformly random split draws the next unit as a
    control with probability (n0 - i)/(N - t)); wherever ``ends`` marks the last
    position of a tied value, the mass outside the band |i n1 - j n0| <
    numerator is removed and accumulated, so the upper tail is a sum of
    positive terms without cancellation (the tie-aware law of R's psmirnov).
    """
    if numerator <= 0:
        return 1.0
    total = n0 + n1
    mass = torch.zeros(n0 + 1, dtype=FLOAT)
    mass[0] = 1.0
    i = torch.arange(n0 + 1, dtype=torch.int64)
    controls = (n0 - i).to(FLOAT)
    hit = 0.0
    for t in range(1, total + 1):
        j_before = (t - 1) - i
        stay = (n1 - j_before).clamp_min(0).to(FLOAT) / (total - t + 1)
        move = controls / (total - t + 1)
        nxt = mass * stay
        nxt[1:] += mass[:-1] * move[:-1]
        j = t - i
        nxt = torch.where((j >= 0) & (j <= n1), nxt, torch.zeros_like(nxt))
        if bool(ends[t - 1]):
            outside = (i * n1 - j * n0).abs() >= numerator
            hit += float(nxt[outside].sum())
            nxt = torch.where(outside, torch.zeros_like(nxt), nxt)
        mass = nxt
    return min(1.0, max(0.0, hit))


def ks_large_sample(statistic: KolmogorovSmirnov, column: int) -> float:
    """R's ks.test p-value: exact (tie-aware) Smirnov law when n0 n1 < 10,000, else the limit."""
    numerator = int(statistic.observed_num[column])
    n0, n1 = statistic.n0, statistic.n1
    if n0 * n1 < SMIRNOV_EXACT:
        return smirnov_sf(numerator, n0, n1, statistic.ends[:, column])
    return kolmogorov_sf(math.sqrt(n0 * n1 / (n0 + n1)) * numerator / (n0 * n1))


def binomial_half(treated: int, total: int) -> float:
    """Two-sided exact binomial test of P(treated) = 1/2 (R's binom.test)."""
    if 2 * treated == total:
        return 1.0
    return min(1.0, 2.0 * binomial_lower(min(treated, total - treated), total, 0.5))


def hotelling_f(t2: float, n: int, k: int) -> float:
    """F approximation of Hotelling's T^2: F = (n-k-1)/((n-2) k) T^2 ~ F(k, n-1-k)."""
    if n - 1 - k <= 0:
        raise AnalysisError("insufficient_observations", "The Hotelling F approximation needs more "
                            "observations than covariates plus one in every window.")
    return f_sf((n - k - 1) / ((n - 2) * k) * t2, k, n - 1 - k)


@dataclass
class HCFit:
    estimate: Tensor     # [m] coefficient on the treatment indicator (or treatment received)
    se: Tensor           # [m]
    n: int               # observations with positive weight


def hc_fit(y: Tensor, d: Tensor, terms: Tensor | None, weights: Tensor | None, vce: str,
           treatment: Tensor | None = None) -> HCFit:
    """Weighted polynomial (or TSLS) regression with an HC1/HC2/HC3 sandwich.

    Regressors z = [1, D, P, D P] (P the powers of the adjusted running
    variable); with ``treatment`` the second column of x is the treatment
    received, instrumented by D through z (projected leverage, structural
    residuals). Only units with positive weight enter. The standard error of
    coefficient 2 is ||(x_hat B e_2) w e s||, B = (x_hat'W x_hat)^{-1} and
    s = sqrt(n/(n-k)), 1/sqrt(1-h) or 1/(1-h) for HC1, HC2, HC3 (the package's
    ``rdlocrand_hc_fit``).
    """
    w = torch.ones_like(d, dtype=FLOAT) if weights is None else weights
    use = w > 0
    y2 = (y if y.ndim == 2 else y[:, None])[use]
    dd, w = d[use].to(FLOAT), w[use]
    pieces = [torch.ones_like(dd)[:, None], dd[:, None]]
    if terms is not None and terms.shape[1]:
        pieces += [terms[use], dd[:, None] * terms[use]]
    z = torch.cat(pieces, 1)
    n, k = z.shape
    message = ("Polynomial regression needs more observations and distinct running-variable "
               "values than fitted coefficients in the window.")
    if n <= k:
        raise AnalysisError("insufficient_observations", message)
    x = z.clone()
    try:
        if treatment is not None:
            x[:, 1] = treatment[use]
            first = least_squares(z, x[:, 1], w, drop_collinear=False)
            projected = z.clone()
            projected[:, 1] = first.fitted
        else:
            projected = x
        fit = least_squares(projected, y2, w, drop_collinear=False, need_leverage=True)
    except KernelError as exc:
        if exc.code == "singular_design":
            raise AnalysisError("insufficient_observations", message if treatment is None else
                                "The TSLS polynomial regression is not identified (no first-stage "
                                "variation in the window).") from exc
        raise AnalysisError(exc.code, str(exc)) from exc
    leverage = fit.leverage
    if vce != "HC1" and bool((1 - leverage <= 1e-10).any()):
        raise AnalysisError("unit_leverage", "HC2/HC3 is undefined for a fit with unit leverage; "
                            "use vce='HC1' or a larger window.")
    residual = y2 - x @ fit.beta
    if vce == "HC1":
        scale = torch.full_like(leverage, math.sqrt(n / (n - k)))
    elif vce == "HC2":
        scale = 1 / (1 - leverage).sqrt()
    else:
        scale = 1 / (1 - leverage)
    loading = projected @ fit.xtx_inv[:, 1]
    influence = (loading * w * scale)[:, None] * residual
    return HCFit(fit.beta[1], influence.square().sum(0).sqrt(), n)
