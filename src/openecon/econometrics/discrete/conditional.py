"""Conditional (fixed-effects) logit likelihood on float64 tensors.

For a group g with observations ``i = 1..T``, ``m`` positive outcomes and
``u_i = exp(e_i)``, ``e_i = x_i'b + offset_i``, the conditional likelihood of
the observed outcomes given their number is

    L_g = exp(sum_i y_i e_i) / f(T, m),    f(T, m) = sum_{|S| = m} prod_{i in S} u_i,

the elementary symmetric function of degree m, which obeys the recursion
(Howard 1972; Gail, Lubin and Rubinstein 1981; Stata's clogit)

    f(t, j) = f(t-1, j) + u_t f(t-1, j-1),     f(t, 0) = 1,  f(0, j > 0) = 0.

Normalized recursion
--------------------
``f`` and its derivatives overflow and underflow easily, so the recursion is
carried in a normalized form that cannot. Write ``S_t = sum_{s <= t} y_s`` for
the running count under the conditional model and

    b(t, j) = u_t f(t-1, j-1) / f(t, j) = Pr(y_t = 1 | S_t = j),

computed as ``sigmoid(e_t + ln f(t-1, j-1) - ln f(t-1, j))`` with
``ln f(t, j) = logaddexp(ln f(t-1, j), e_t + ln f(t-1, j-1))``. The conditional
mean and covariance of the partial sufficient statistic
``s_t = sum_{s <= t} y_s x_s`` given ``S_t = j`` are then a two-component
mixture of the previous step's moments:

    d      = D(t-1, j-1) + x_t - D(t-1, j)
    D(t, j) = D(t-1, j) + b d
    C(t, j) = (1 - b) C(t-1, j) + b C(t-1, j-1) + b (1 - b) d d'.

At ``t = T``, ``j = m`` they are the derivatives of ``ln f``:

    d ln f / db = D(T, m),      d2 ln f / db db' = C(T, m),

so the score of the group is ``sum_i y_i x_i - D`` and its Hessian ``-C``, a
sum of positive semidefinite terms with no cancellation. The recursion runs
over the position t within the group for all groups of a block at once (groups
sorted by size, so the active ones are a prefix); a Python loop touches only
the T positions, never observations.

Shortcuts
---------
* A group with more positives than negatives is reflected (``y -> 1 - y``,
  ``e -> -e``), which leaves its likelihood unchanged and bounds the recursion
  width by ``min(m, T - m)``.
* Groups with ``min(m, T - m) = 1`` need no recursion: their likelihood is a
  multinomial logit over the rows of the group (log-sum-exp by ``index_add_``).

Per-observation scores ``(y_i - Pr(y_i = 1 | m)) x_i`` (cluster covariances)
come from a backward pass over the stored ``b``: with
``r_t(j) = Pr(S_t = j | S_T = m)``, ``Pr(y_t = 1 | m) = sum_j r_t(j) b(t, j)``
and ``r_{t-1}(j) = r_t(j) (1 - b(t, j)) + r_t(j+1) b(t, j+1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError
from openecon.engines.covariance import group_counts, group_sums

# Elements of the widest recursion temporary of one block (16 MiB of float64).
_BLOCK_ELEMENTS = 2_000_000


@dataclass
class _Block:
    """Groups with the same recursion width, laid out position-major."""

    width: int          # min(m, T - m) + 1 states
    groups: Tensor      # group codes in rank order (sizes descending)
    rows: Tensor        # observation rows ordered by (position, rank)
    counts: list[int]   # active groups at each position
    x: Tensor           # design rows in that order


class ConditionalLogitObjective:
    """Conditional log likelihood ``sum_g w_g ln L_g`` and its analytic derivatives.

    ``codes`` are dense group codes ``0..G-1`` and ``group_weights`` one weight
    per group. Every group must contain both outcomes.
    """

    def __init__(self, x: Tensor, y: Tensor, codes: Tensor, n_groups: int,
                 group_weights: Tensor, offset: Tensor | None = None):
        self.x, self.codes, self.offset = x, codes, offset
        self.n, self.k = x.shape
        self.groups = n_groups
        self.group_w = group_weights
        sizes = group_counts(codes, n_groups)
        positives = group_sums(y, codes, n_groups).round().to(torch.int64)
        if bool(((positives <= 0) | (positives >= sizes)).any()):
            raise KernelError("no_outcome_variation", "Every group of a conditional logit must "
                              "contain both positive and negative outcomes.")
        reflect = 2 * positives > sizes
        self.width = torch.where(reflect, sizes - positives, positives)      # min(m, T - m)
        self.sign = torch.where(reflect, -1.0, 1.0).to(torch.float64)
        self.sign_obs = self.sign[codes]
        self.chosen = torch.where(reflect[codes], 1.0 - y, y)
        self.w_obs = group_weights[codes]
        self.chosen_x = group_sums(x * self.chosen[:, None], codes, n_groups)
        self.gradient_constant = (self.chosen_x * (group_weights * self.sign)[:, None]).sum(dim=0)
        self.sizes, self.positives = sizes, positives
        self._layout(sizes)

    # ---- layout -----------------------------------------------------------------

    def _layout(self, sizes: Tensor) -> None:
        codes, width = self.codes, self.width
        single = width == 1
        self.single_groups = single.nonzero().flatten()
        compact = torch.full((self.groups,), -1, dtype=torch.int64)
        compact[self.single_groups] = torch.arange(len(self.single_groups))
        single_rows = single[codes].nonzero().flatten()
        self.single_rows = single_rows
        self.single_codes = compact[codes[single_rows]]
        self.single_x = self.x if len(single_rows) == self.n else self.x[single_rows]
        self.blocks: list[_Block] = []
        multi = (~single).nonzero().flatten()
        if not len(multi):
            return
        # Position of every observation inside its group, in input order.
        by_group = torch.argsort(codes, stable=True)
        starts = torch.cumsum(sizes, dim=0) - sizes
        position = torch.empty(self.n, dtype=torch.int64)
        position[by_group] = torch.arange(self.n) - starts[codes[by_group]]
        # Groups ordered by (width, size descending) and cut into blocks.
        order = multi[torch.argsort(sizes[multi], descending=True, stable=True)]
        order = order[torch.argsort(width[order], stable=True)]
        block_of = torch.full((self.groups,), -1, dtype=torch.int64)
        rank_of = torch.zeros(self.groups, dtype=torch.int64)
        members: list[Tensor] = []
        widths, lengths = torch.unique_consecutive(width[order], return_counts=True)
        begin = 0
        for value, length in zip(widths.tolist(), lengths.tolist(), strict=True):
            per_block = max(16, _BLOCK_ELEMENTS // ((value + 1) * self.k * self.k))
            for first in range(begin, begin + length, per_block):
                chunk = order[first:min(first + per_block, begin + length)]
                block_of[chunk] = len(members)
                rank_of[chunk] = torch.arange(len(chunk))
                members.append(chunk)
            begin += length
        rows = (block_of[codes] >= 0).nonzero().flatten()
        # Lexicographic (block, position, rank) by three stable sorts.
        rows = rows[torch.argsort(rank_of[codes[rows]], stable=True)]
        rows = rows[torch.argsort(position[rows], stable=True)]
        rows = rows[torch.argsort(block_of[codes[rows]], stable=True)]
        totals = group_sums(sizes[order].to(torch.float64), block_of[order], len(members))
        begin = 0
        for index, chunk in enumerate(members):
            count = int(totals[index])
            block_rows = rows[begin:begin + count]
            counts = torch.bincount(position[block_rows]).tolist()
            self.blocks.append(_Block(int(width[chunk[0]]) + 1, chunk, block_rows, counts,
                                      self.x[block_rows]))
            begin += count

    # ---- recursion ----------------------------------------------------------------

    def _recursion(self, block: _Block, index: Tensor, derivatives: bool, keep: bool):
        """Run the normalized recursion of one block.

        Returns ``ln f`` and, when requested, the conditional mean [G_b, k], the
        conditional covariance [G_b, k, k] and the step probabilities ``b`` (one
        [active, reachable states] tensor per position). At position t only the
        states ``1..min(t, width - 1)`` can change, so only those are touched.
        """
        groups, width, k = len(block.groups), block.width, self.k
        log_f = torch.full((groups, width), -math.inf, dtype=torch.float64)
        log_f[:, 0] = 0.0
        mean = cov = None
        if derivatives:
            mean = torch.zeros((groups, width, k), dtype=torch.float64)
            cov = torch.zeros((groups, width, k, k), dtype=torch.float64)
        steps: list[Tensor] = []
        begin = 0
        for position, active in enumerate(block.counts):
            top = min(position + 1, width - 1)
            stay = log_f[:active, 1:top + 1]                              # ln f(t-1, j)
            take = log_f[:active, :top] + index[begin:begin + active, None]
            if derivatives or keep:
                step = torch.sigmoid(take - stay)                          # b(t, j)
                if keep:
                    steps.append(step)
            if derivatives:
                # In-place updates on views of the state.
                own_mean, own_cov = mean[:active, 1:top + 1], cov[:active, 1:top + 1]
                delta = block.x[begin:begin + active, None, :] + mean[:active, :top] - own_mean
                change = cov[:active, :top] - own_cov
                change *= step[:, :, None, None]
                scaled = delta * (step * (1 - step))[:, :, None]
                change.addcmul_(scaled[:, :, :, None], delta[:, :, None, :])
                own_cov += change
                delta *= step[:, :, None]
                own_mean += delta
            log_f[:active, 1:top + 1] = torch.logaddexp(stay, take)
            begin += active
        last = width - 1
        if derivatives:
            return log_f[:, last], mean[:, last], cov[:, last], steps
        return log_f[:, last], None, None, steps

    def _index(self, theta: Tensor) -> Tensor:
        eta = self.x @ theta
        if self.offset is not None:
            eta = eta + self.offset
        return self.sign_obs * eta

    def _single(self, index: Tensor) -> tuple[Tensor, Tensor]:
        """ln sum_i exp(e_i) per single-width group and the within-group softmax."""
        values = index if len(self.single_rows) == self.n else index[self.single_rows]
        count = len(self.single_groups)
        top = torch.full((count,), -math.inf, dtype=torch.float64).scatter_reduce(
            0, self.single_codes, values, "amax")
        scaled = torch.exp(values - top[self.single_codes])
        total = group_sums(scaled, self.single_codes, count)
        return top + torch.log(total), scaled / total[self.single_codes]

    def _evaluate(self, theta: Tensor, derivatives: bool, probabilities: bool = False):
        index = self._index(theta)
        log_f = torch.zeros(self.groups, dtype=torch.float64)
        mean = torch.zeros((self.groups, self.k), dtype=torch.float64) if derivatives else None
        spread = torch.zeros((self.k, self.k), dtype=torch.float64)
        chance = torch.zeros(self.n, dtype=torch.float64) if probabilities else None
        if len(self.single_groups):
            log_f[self.single_groups], share = self._single(index)
            if probabilities:
                chance[self.single_rows] = share
            if derivatives:
                x = self.single_x
                centre = group_sums(x * share[:, None], self.single_codes,
                                    len(self.single_groups))
                mean[self.single_groups] = centre
                weights = self.group_w[self.single_groups]
                row_weights = self.w_obs if x is self.x else self.w_obs[self.single_rows]
                spread += x.T @ (x * (row_weights * share)[:, None])
                spread -= (centre * weights[:, None]).T @ centre
        for block in self.blocks:
            values, centre, covariance, steps = self._recursion(
                block, index[block.rows], derivatives, probabilities)
            log_f[block.groups] = values
            if derivatives:
                mean[block.groups] = centre
                spread += (covariance * self.group_w[block.groups][:, None, None]).sum(dim=0)
            if probabilities:
                chance[block.rows] = self._backward(block, steps)
        value = (self.w_obs * self.chosen * index).sum() - (self.group_w * log_f).sum()
        return value, index, log_f, mean, (spread + spread.T) / 2, chance

    def _backward(self, block: _Block, steps: list[Tensor]) -> Tensor:
        """``Pr(chosen_t = 1 | group total)`` for the rows of a block, in block order."""
        occupancy = torch.zeros((len(block.groups), block.width), dtype=torch.float64)
        occupancy[:, -1] = 1.0
        chance = torch.empty(len(block.rows), dtype=torch.float64)
        end = len(block.rows)
        for active, step in zip(reversed(block.counts), reversed(steps), strict=True):
            top = step.shape[1]
            taken = occupancy[:active, 1:top + 1] * step
            chance[end - active:end] = taken.sum(dim=1)
            occupancy[:active, 1:top + 1] -= taken
            occupancy[:active, :top] += taken
            end -= active
        return chance

    # ---- objective ----------------------------------------------------------------

    def value(self, theta: Tensor) -> Tensor:
        return self._evaluate(theta, False)[0]

    def __call__(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value, _, _, mean, covariance, _ = self._evaluate(theta, True)
        gradient = self.gradient_constant \
            - (mean * (self.group_w * self.sign)[:, None]).sum(dim=0)
        return value, gradient, -covariance

    def group_log_likelihood(self, theta: Tensor) -> Tensor:
        """``ln L_g`` of every group, before weights."""
        _, index, log_f, _, _, _ = self._evaluate(theta, False)
        return group_sums(self.chosen * index, self.codes, self.groups) - log_f

    def group_scores(self, theta: Tensor) -> Tensor:
        """Score of each group's conditional likelihood [G, k], before weights."""
        mean = self._evaluate(theta, True)[3]
        return self.sign[:, None] * (self.chosen_x - mean)

    def score_rows(self, theta: Tensor) -> Tensor:
        """Per-observation scores ``(y_i - Pr(y_i = 1 | m_g)) x_i`` [n, k], before weights."""
        chance = self._evaluate(theta, False, probabilities=True)[5]
        return self.x * (self.sign_obs * (self.chosen - chance))[:, None]
