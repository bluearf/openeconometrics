"""Tensor kernels of stepwise regression: cross products, sweeps and the search.

Cross products.  The data are read ONCE, in row blocks: with a fixed shift ``s``
close to the column means, the block sums ``sum w``, ``sum w (z - s)`` and
``sum w (z - s)(z - s)'`` give the weighted means and the mean-centred
cross-product matrix

    C = sum_i w_i (z_i - zbar)(z_i - zbar)'
      = sum_i w_i (z_i - s)(z_i - s)' - W (zbar - s)(zbar - s)',     W = sum_i w_i,

without a second pass and without the cancellation of the raw-moment formula
(the correction term is tiny when ``s`` is near the mean).  Centring is the
sweep of the constant, so every model below contains an intercept.

Sweep operator.  ``C`` is scaled to the correlation matrix ``R`` of
``[x_1 .. x_p, y]`` (SPSS REGRESSION works on the same matrix).  Sweeping the
pivots of a set ``S`` of columns turns ``R`` into

    A_SS = -R_SS^{-1},     A_Sj = R_SS^{-1} R_Sj,     A_jk = R_jk - R_jS R_SS^{-1} R_Sk ,

so that for the current model: ``A_yy = 1 - R^2``; ``A_ky`` (k in S) is the
standardized coefficient; ``A_kk`` (k outside S) is the tolerance of x_k, the
share of its variance not explained by the model; ``A_ky^2 / A_kk`` is the
R-squared gained by entering x_k and ``A_ky^2 / (-A_kk)`` (k in S) the R-squared
lost by removing it.  A step costs one O(p^2) rank-one update; the n rows are
never touched again.

Search.  Terms are blocks of columns (one column, or the dummies of a
categorical predictor, which enter and leave together and are tested with a
q-degree-of-freedom F test).  The F statistics are

    enter :  F = (gain / q) / ((A_yy - gain) / (n - 1 - k - q))
    remove:  F = (loss / q) / ( A_yy         / (n - 1 - k))

with ``k`` the number of columns in the model.  Everything here is float64
tensors and plain Python control flow over terms and steps, never over rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from openecon.engines import distributions as dist
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import cholesky_inverse, symmetrize

# Residual share of the outcome's variance below which a fit is exact to rounding.
_PERFECT = 1e-13
_TINY = 1e-300


class CrossProducts:
    """One-pass weighted means and centred cross products of the columns of z."""

    def __init__(self, shift: Tensor):
        k = shift.numel()
        self.shift = shift
        self.weight = 0.0
        self.rows = 0
        self.first = torch.zeros(k, dtype=torch.float64)
        self.second = torch.zeros((k, k), dtype=torch.float64)

    @torch.no_grad()
    def add(self, z: Tensor, weights: Tensor | None = None) -> None:
        """Accumulate a block of rows ``z`` [m, k] with optional weights [m]."""
        centred = z - self.shift
        if weights is None:
            self.first += centred.sum(dim=0)
            self.second += centred.T @ centred
            self.weight += float(z.shape[0])
        else:
            weighted = centred * weights[:, None]
            self.first += weighted.sum(dim=0)
            self.second += weighted.T @ centred
            self.weight += float(weights.sum())
        self.rows += z.shape[0]

    @torch.no_grad()
    def finish(self) -> tuple[Tensor, Tensor]:
        """(weighted means, centred cross-product matrix)."""
        if not self.weight > 0:
            raise KernelError("empty_sample", "No observations with positive weight remain.")
        offset = self.first / self.weight
        centred = symmetrize(self.second - self.weight * torch.outer(offset, offset))
        if not bool(torch.isfinite(centred).all()):
            raise KernelError("non_finite_values", "The cross products are not finite; rescale "
                              "the columns.")
        return self.shift + offset, centred


@torch.no_grad()
def correlation(cross: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(R, root sums of squares d, valid) with R = D^-1 C D^-1 and a unit diagonal.

    A column without variation (``d_j = 0`` up to rounding of its own mean) is
    marked invalid; its row and column of R are zero apart from the diagonal.
    """
    diagonal = cross.diagonal().clamp_min(0.0)
    scale = diagonal.sqrt()
    valid = scale > 0
    safe = torch.where(valid, scale, torch.ones_like(scale))
    corr = cross / safe[:, None] / safe
    corr[~valid] = 0.0
    corr[:, ~valid] = 0.0
    corr.fill_diagonal_(1.0)
    return symmetrize(corr), scale, valid


@torch.no_grad()
def sweep(a: Tensor, k: int) -> None:
    """Sweep the symmetric matrix ``a`` on pivot ``k`` in place (column k enters)."""
    pivot = float(a[k, k])
    column = a[:, k].clone()
    row = column / pivot
    a.addr_(column, row, alpha=-1.0)
    a[k, :] = row
    a[:, k] = row
    a[k, k] = -1.0 / pivot


@torch.no_grad()
def unsweep(a: Tensor, k: int) -> None:
    """Reverse sweep on pivot ``k`` in place (column k leaves)."""
    pivot = float(a[k, k])
    column = a[:, k].clone()
    row = column / pivot
    a.addr_(column, row, alpha=-1.0)
    a[k, :] = -row
    a[:, k] = -row
    a[k, k] = -1.0 / pivot


def _f_sf(statistic: float, df1: float, df2: float) -> float:
    if math.isinf(statistic):
        return 0.0
    return dist.f_sf(max(statistic, 0.0), df1, df2)


@dataclass
class Move:
    """One candidate or executed step."""

    action: str                 # "entered", "removed" or "start"
    term: int                   # term index (-1 for the starting model)
    statistic: float | None     # F of the term given the model (overall F for "start")
    df1: int
    df2: float
    p_value: float | None
    delta: float = 0.0          # change of the information criterion
    r_squared: float = 0.0      # of the model AFTER the step
    columns: int = 0            # regressor columns in the model AFTER the step


@dataclass
class SearchResult:
    steps: list[Move]
    order: list[int]                         # terms in the final model, in order of entry
    start: list[int] = field(default_factory=list)          # terms of the starting model
    skipped: dict[int, str] = field(default_factory=dict)   # term -> why it could not start
    stopped: str = ""                        # set when the step limit ended the search
    exhausted: bool = False                  # 1 - R^2 reached the rounding level of the sweeps


class Search:
    """Stepwise search state on the swept correlation matrix (y is the last column)."""

    def __init__(self, corr: Tensor, blocks: list[list[int]], valid: Tensor, *, n: float,
                 tolerance: float):
        self.corr = corr
        self.a = corr.clone()
        self.y = corr.shape[0] - 1
        self.blocks = blocks
        self.n = float(n)
        self.tolerance = tolerance
        terms = len(blocks)
        self.size = torch.tensor([len(block) for block in blocks], dtype=torch.float64)
        self.usable = torch.tensor([bool(block) and bool(valid[block].all())
                                    for block in blocks], dtype=torch.bool)
        self.single = torch.tensor([len(block) == 1 for block in blocks], dtype=torch.bool)
        self.first = torch.tensor([block[0] if block else 0 for block in blocks],
                                  dtype=torch.int64)
        self.term_in = torch.zeros(terms, dtype=torch.bool)
        self.column_in = torch.zeros(self.y, dtype=torch.bool)
        self.order: list[int] = []

    # ---- state -------------------------------------------------------------------

    @property
    def k(self) -> int:
        return int(self.column_in.sum())

    @property
    def residual(self) -> float:
        """1 - R^2 of the current model."""
        return min(1.0, max(0.0, float(self.a[self.y, self.y])))

    def enter(self, term: int) -> None:
        for column in self.blocks[term]:
            sweep(self.a, column)
            self.column_in[column] = True
        self.term_in[term] = True
        self.order.append(term)

    def remove(self, term: int) -> None:
        for column in self.blocks[term]:
            unsweep(self.a, column)
            self.column_in[column] = False
        self.term_in[term] = False
        self.order.remove(term)

    def refresh(self) -> None:
        """Recompute the swept matrix from R, discarding accumulated rounding."""
        self.a = self.corr.clone()
        for term in self.order:
            for column in self.blocks[term]:
                sweep(self.a, column)

    # ---- candidates --------------------------------------------------------------

    def _block_entry(self, columns: list[int], inside: Tensor) -> tuple[float, float, float]:
        """(R-squared gain, tolerance, minimum tolerance) of entering a block of columns."""
        a, y = self.a, self.y
        index = torch.tensor(columns, dtype=torch.int64)
        block = a[index][:, index]
        if bool((block.diagonal() <= 0).any()):
            return 0.0, 0.0, 0.0
        try:
            inverse = cholesky_inverse(block)
        except KernelError:
            return 0.0, 0.0, 0.0
        cross = a[y, index]
        gain = float(cross @ inverse @ cross)
        tolerance = float((1.0 / inverse.diagonal()).min())
        low = tolerance
        if inside.numel():
            link = a[inside][:, index]
            grown = -a[inside, inside] + ((link @ inverse) * link).sum(dim=1)
            low = min(low, float(1.0 / grown.max()))
        return gain, tolerance, low

    def entry(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per term: (R-squared gain, tolerance, minimum tolerance, signed A_ky).

        Terms already in the model get zeros. The tolerance of a block is the
        smallest tolerance of its columns once the whole block is in; the
        minimum tolerance also covers the columns already in the model.
        """
        a, y = self.a, self.y
        terms = len(self.blocks)
        gain = torch.zeros(terms, dtype=torch.float64)
        tolerance = torch.zeros(terms, dtype=torch.float64)
        low = torch.zeros(terms, dtype=torch.float64)
        signed = torch.zeros(terms, dtype=torch.float64)
        inside = self.column_in.nonzero().flatten()
        outside = self.usable & ~self.term_in
        singles = (outside & self.single).nonzero().flatten()
        if singles.numel():
            columns = self.first[singles]
            pivot = a[columns, columns]
            safe = pivot.clamp_min(_TINY)
            cross = a[y, columns]
            gain[singles] = torch.where(pivot > 0, cross.square() / safe, torch.zeros_like(pivot))
            tolerance[singles] = pivot.clamp(0.0, 1.0)
            signed[singles] = cross
            least = pivot.clamp(0.0, 1.0)
            if inside.numel():
                grown = -a[inside, inside][:, None] + a[inside][:, columns].square() / safe
                least = torch.minimum(least, 1.0 / grown.max(dim=0).values)
            low[singles] = torch.where(pivot > 0, least, torch.zeros_like(least))
        for term in (outside & ~self.single).nonzero().flatten().tolist():
            gain[term], tolerance[term], low[term] = self._block_entry(self.blocks[term], inside)
        return gain.clamp(max=self.residual), tolerance, low, signed

    def removal(self, removable: Tensor) -> Tensor:
        """Per term: the R-squared lost by removing it (zeros for terms not removable)."""
        a, y = self.a, self.y
        loss = torch.zeros(len(self.blocks), dtype=torch.float64)
        inside = self.term_in & removable
        singles = (inside & self.single).nonzero().flatten()
        if singles.numel():
            columns = self.first[singles]
            loss[singles] = a[y, columns].square() / (-a[columns, columns]).clamp_min(_TINY)
        for term in (inside & ~self.single).nonzero().flatten().tolist():
            index = torch.tensor(self.blocks[term], dtype=torch.int64)
            cross = a[y, index]
            try:
                inverse = cholesky_inverse(-a[index][:, index])
            except KernelError:
                continue
            loss[term] = float(cross @ inverse @ cross)
        return loss.clamp_min(0.0)

    # ---- tests -------------------------------------------------------------------

    def enter_test(self, term: int, gain: float) -> Move:
        q = len(self.blocks[term])
        df2 = self.n - 1.0 - self.k - q
        left = self.residual - gain
        statistic = math.inf if left <= _PERFECT * 1e-2 else (gain / q) / (left / df2)
        return Move("entered", term, statistic, q, df2, _f_sf(statistic, q, df2))

    def remove_test(self, term: int, loss: float) -> Move:
        q = len(self.blocks[term])
        df2 = self.n - 1.0 - self.k
        statistic = (loss / q) / (max(self.residual, _TINY) / df2)
        return Move("removed", term, statistic, q, df2, _f_sf(statistic, q, df2))

    def eligible(self, tolerance: Tensor, low: Tensor) -> Tensor:
        """Terms that may enter: usable, out, passing both tolerances, leaving residual df."""
        room = (self.n - 1.0 - self.k - self.size) > 0
        return (self.usable & ~self.term_in & room & (tolerance >= self.tolerance)
                & (low >= self.tolerance))

    def best_entry(self) -> Move | None:
        """The eligible term with the smallest p-value of its F-to-enter."""
        gain, tolerance, low, _ = self.entry()
        allowed = self.eligible(tolerance, low)
        candidates: list[Move] = []
        singles = (allowed & self.single).nonzero().flatten()
        if singles.numel():
            # With one degree of freedom the smallest p-value is the largest gain.
            term = int(singles[torch.argmax(gain[singles])])
            candidates.append(self.enter_test(term, float(gain[term])))
        for term in (allowed & ~self.single).nonzero().flatten().tolist():
            candidates.append(self.enter_test(term, float(gain[term])))
        if not candidates:
            return None
        return min(candidates, key=lambda move: (move.p_value, -move.statistic, move.term))

    def worst_removal(self, removable: Tensor) -> Move | None:
        """The removable term with the largest p-value of its F-to-remove."""
        inside = self.term_in & removable
        if not bool(inside.any()):
            return None
        loss = self.removal(removable)
        candidates: list[Move] = []
        singles = (inside & self.single).nonzero().flatten()
        if singles.numel():
            term = int(singles[torch.argmin(loss[singles])])
            candidates.append(self.remove_test(term, float(loss[term])))
        for term in (inside & ~self.single).nonzero().flatten().tolist():
            candidates.append(self.remove_test(term, float(loss[term])))
        return max(candidates, key=lambda move: (move.p_value, -move.statistic, -move.term))

    def best_by_criterion(self, penalty: float, removable: Tensor, *, enter: bool,
                          remove: bool) -> Move | None:
        """The move that lowers  n ln(SSE / n) + penalty * k  the most (None if none exists)."""
        best: tuple[float, str, int, float] | None = None
        residual = max(self.residual, _TINY)
        if remove and bool((self.term_in & removable).any()):
            inside = self.term_in & removable
            loss = self.removal(removable)
            delta = self.n * torch.log1p(loss / residual) - penalty * self.size
            delta = torch.where(inside, delta, torch.full_like(delta, math.inf))
            term = int(torch.argmin(delta))
            best = (float(delta[term]), "removed", term, float(loss[term]))
        if enter:
            gain, tolerance, low, _ = self.entry()
            allowed = self.eligible(tolerance, low)
            if bool(allowed.any()):
                share = (1.0 - gain / residual).clamp_min(_TINY)
                delta = self.n * torch.log(share) + penalty * self.size
                delta = torch.where(allowed, delta, torch.full_like(delta, math.inf))
                term = int(torch.argmin(delta))
                if best is None or float(delta[term]) < best[0]:
                    best = (float(delta[term]), "entered", term, float(gain[term]))
        if best is None:
            return None
        delta_value, action, term, change = best
        move = self.enter_test(term, change) if action == "entered" \
            else self.remove_test(term, change)
        move.delta = delta_value
        return move

    # ---- driver ------------------------------------------------------------------

    def _apply(self, move: Move, steps: list[Move]) -> bool:
        """Execute a move; False when 1 - R^2 has reached the rounding level of the sweeps.

        Below that level (``_PERFECT``) the residual share is a difference of O(1)
        numbers that carries no correct digits, so neither the statistic of this move
        nor any further test is meaningful: the move's F and p are withdrawn and the
        search stops. Whether the fit is exact is decided afterwards from the residuals
        of the data, not from the swept matrix.
        """
        if move.action == "entered":
            self.enter(move.term)
        else:
            self.remove(move.term)
        if len(steps) % 32 == 31:
            self.refresh()                     # bound the rounding accumulated by sweeps
        move.r_squared = 1.0 - self.residual
        move.columns = self.k
        steps.append(move)
        if self.residual <= _PERFECT:
            move.statistic = move.p_value = None
            return False
        return True

    def _start(self, terms: list[int], skipped: dict[int, str]) -> None:
        """Enter the starting terms in order, leaving out those that fail the tolerance."""
        for term in terms:
            if not bool(self.usable[term]):
                skipped[term] = "no variation"
                continue
            if self.n - 1.0 - self.k - len(self.blocks[term]) <= 0:
                raise KernelError("insufficient_observations", "The starting model has as many "
                                  "parameters as observations; reduce the number of terms.")
            inside = self.column_in.nonzero().flatten()
            _, tolerance, low = self._block_entry(self.blocks[term], inside)
            if tolerance < self.tolerance or low < self.tolerance:
                skipped[term] = "tolerance"
                continue
            self.enter(term)


@torch.no_grad()
def stepwise_search(corr: Tensor, blocks: list[list[int]], valid: Tensor, forced: list[int], *,
                    n: float, method: str, criterion: str, p_enter: float, p_remove: float,
                    tolerance: float) -> tuple[Search, SearchResult]:
    """Run forward, backward or stepwise selection on the correlation matrix ``corr``.

    ``blocks[t]`` lists the columns of term ``t``; ``forced`` terms start in the
    model and are never removed. ``criterion`` is ``"pvalue"`` (enter the most
    significant term while p < ``p_enter``; remove the least significant while
    p > ``p_remove``; the stepwise method tries a removal before every entry,
    as SPSS does) or ``"aic"`` / ``"bic"`` (take the single entry or removal
    that lowers the criterion most, until none does).
    """
    search = Search(corr, blocks, valid, n=n, tolerance=tolerance)
    result = SearchResult(steps=[], order=search.order)
    terms = len(blocks)
    removable = torch.ones(terms, dtype=torch.bool)
    if forced:
        removable[forced] = False
    start = list(forced)
    if method == "backward":
        start += [term for term in range(terms) if term not in set(forced)]
    search._start(start, result.skipped)
    result.start = list(search.order)
    if search.k:
        k, residual = search.k, search.residual
        df2 = search.n - 1.0 - k
        statistic = p_value = None
        if residual > _PERFECT:
            statistic = ((1.0 - residual) / k) / (residual / df2)
            p_value = _f_sf(statistic, k, df2)
        result.steps.append(Move("start", -1, statistic, k, df2, p_value,
                                 r_squared=1.0 - residual, columns=k))
        if residual <= _PERFECT:
            result.exhausted = True
            return search, result
    penalty = 2.0 if criterion == "aic" else math.log(max(n, 2.0))
    can_enter, can_remove = method != "backward", method != "forward"
    limit = 2 * terms + 2
    for _ in range(limit):
        if criterion == "pvalue":
            move = search.worst_removal(removable) if can_remove else None
            if move is not None and move.p_value > p_remove:
                if not search._apply(move, result.steps):
                    result.exhausted = True
                    break
                continue
            move = search.best_entry() if can_enter else None
            if move is not None and move.p_value < p_enter:
                if not search._apply(move, result.steps):
                    result.exhausted = True
                    break
                continue
            break
        move = search.best_by_criterion(penalty, removable, enter=can_enter, remove=can_remove)
        if move is None or not move.delta < -1e-9:
            break
        if not search._apply(move, result.steps):
            result.exhausted = True
            break
    else:
        result.stopped = f"the search was stopped after {limit} steps"
    return search, result


@torch.no_grad()
def final_model(corr: Tensor, columns: list[int]) -> tuple[Tensor, Tensor, float]:
    """Fresh solve of the selected model: (standardized betas, R_SS^{-1}, 1 - R^2).

    The correlation block of the selected columns is inverted by Cholesky
    (not taken from the accumulated sweeps), so the reported coefficients do
    not inherit rounding from the search path. The coefficients are accurate to
    about cond(R_SS) * eps; the returned 1 - R^2 = 1 - r' R_SS^{-1} r is a
    difference and loses digits as R^2 approaches 1, so the caller recomputes
    the residual sum of squares from the data (``stepwise.residual_sum``).
    """
    y = corr.shape[0] - 1
    if not columns:
        empty = torch.zeros(0, dtype=torch.float64)
        return empty, torch.zeros((0, 0), dtype=torch.float64), 1.0
    index = torch.tensor(columns, dtype=torch.int64)
    inverse = cholesky_inverse(corr[index][:, index], code="singular_design",
                               what="correlation matrix of the selected regressors")
    cross = corr[index, y]
    beta = inverse @ cross
    return beta, inverse, min(1.0, max(0.0, 1.0 - float(cross @ beta)))
