"""Panel layout and the transformations of dynamic panel GMM (tensor kernels).

Notation. The sorted estimation frame has ``N0`` rows in ``G`` panels; row r
belongs to panel ``c_r`` and period ``p_r = t_r - t_min`` (0 .. S-1). The
``(panel, period)`` grid of size ``G * S`` maps every cell to its frame row
(``-1`` when the period is absent), so lags, leads and instrument values are
one gather each; gaps in the time variable are visible as absent cells.

A *level observation* is a frame row whose ``p`` lagged dependent variables
exist (rows of periods ``p_r - 1 .. p_r - lags`` are present). Level
observations are the space in which every transformation is defined:

- first differences ``D``: row (r, r') for consecutive level observations
  ``p_r' = p_r - 1``; ``(D v)_r = v_r - v_r'``. The transformed row is dated
  ``p_r``.
- forward orthogonal deviations ``F`` (Arellano and Bover 1995): for a level
  observation with ``T_r >= 1`` later level observations in its panel,
  ``(F v)_r = c_r (v_r - mean of the later ones)``, ``c_r = sqrt(T_r / (T_r + 1))``.
  As in xtabond2 the deviation of period t is dated ``t + 1``, so instrument
  lags mean the same thing under both transformations.

Both maps are linear in the level vector; ``adjoint`` applies the transpose,
which turns instrument columns of the transformed rows into vectors on the
level observations. With ``M`` the transformation, ``M M'`` is xtabond2's H
of the transformed equation (2 on the diagonal and -1 between adjacent
differences; the identity for orthogonal deviations) and ``M`` itself the
covariance between transformed and level errors under i.i.d. errors.

Nothing here loops over observations: within-panel cumulative sums are
computed on a dense ``(G, width)`` layout of the level observations in blocks
of columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from openecon.engines.contracts import KernelError

# Cells of the (panel, period) grid; the int64 lookup then needs at most 128 MiB.
MAX_GRID_CELLS = 1 << 24
# Elements of one dense block of a within-panel cumulative sum.
_BLOCK_ELEMENTS = 1 << 22


@dataclass
class Layout:
    """The (panel, period) grid of the sorted frame."""

    codes: Tensor          # [N0] panel codes, nondecreasing
    period: Tensor         # [N0] period index 0 .. span-1
    groups: int
    span: int
    lookup: Tensor         # [groups * span] frame row of each cell or -1

    def rows_at(self, codes: Tensor, period: Tensor) -> Tensor:
        """Frame rows at ``(codes, period)``; -1 where the period is absent or out of range."""
        inside = (period >= 0) & (period < self.span)
        cells = codes * self.span + period.clamp(0, self.span - 1)
        return torch.where(inside, self.lookup[cells], torch.full_like(period, -1))

    def values_at(self, values: Tensor, codes: Tensor, period: Tensor) -> tuple[Tensor, Tensor]:
        """``values`` of the frame rows at ``(codes, period)`` (0 where absent) and presence."""
        rows = self.rows_at(codes, period)
        present = rows >= 0
        out = torch.where(present, values[rows.clamp(min=0)], torch.zeros_like(rows, dtype=values.dtype))
        return out, present


def layout(codes: Tensor, period: Tensor, groups: int) -> Layout:
    """Build the grid; ``period`` must already start at 0 and be unique within panel."""
    span = int(period.max()) + 1 if period.numel() else 1
    if groups * span > MAX_GRID_CELLS:
        raise KernelError("time_span_too_large",
                          f"The panel-by-period grid has {groups * span:,} cells (limit "
                          f"{MAX_GRID_CELLS:,}). Recode the time variable as consecutive integer "
                          "periods (for example 1, 2, 3 instead of dates such as 20240101).")
    lookup = torch.full((groups * span,), -1, dtype=torch.int64)
    lookup[codes * span + period] = torch.arange(len(codes), dtype=torch.int64)
    return Layout(codes, period, groups, span, lookup)


def within_cumsum(values: Tensor, codes: Tensor, rank: Tensor, groups: int, width: int, *,
                  reverse: bool = False) -> Tensor:
    """Inclusive cumulative sums of ``values`` ([n] or [n, m]) within panels.

    Rows of one panel are ordered by ``rank`` (0 .. size-1); ``reverse`` sums
    from the end of the panel. The rows are laid out on a dense ``(groups,
    width)`` block, a few columns at a time, so the sums never mix panels and
    keep the precision of a per-panel loop.
    """
    flat = values.reshape(len(values), -1)
    out = torch.empty_like(flat)
    cells = codes * width + rank
    chunk = max(1, _BLOCK_ELEMENTS // max(groups * width, 1))
    for start in range(0, flat.shape[1], chunk):
        block = flat[:, start:start + chunk]
        dense = torch.zeros((groups * width, block.shape[1]), dtype=flat.dtype)
        dense[cells] = block
        dense = dense.view(groups, width, -1)
        summed = dense.flip(1).cumsum(1).flip(1) if reverse else dense.cumsum(1)
        out[:, start:start + chunk] = summed.reshape(groups * width, -1)[cells]
    return out.reshape(values.shape)


@dataclass
class Levels:
    """Level observations: frame rows whose lagged dependent variables all exist."""

    rows: Tensor           # [NL] frame rows (sorted by panel and period)
    codes: Tensor          # [NL] panel codes
    period: Tensor         # [NL] period index
    rank: Tensor           # [NL] position among the panel's level observations
    later: Tensor          # [NL] number of later level observations in the panel
    width: int             # largest number of level observations in one panel
    groups: int

    @property
    def n(self) -> int:
        return len(self.rows)


def levels(grid: Layout, lags: int) -> Levels:
    """Select the level observations of a model with ``lags`` lagged dependent variables."""
    complete = torch.ones(len(grid.codes), dtype=torch.bool)
    for lag in range(1, lags + 1):
        complete &= grid.rows_at(grid.codes, grid.period - lag) >= 0
    rows = complete.nonzero().flatten()
    codes = grid.codes[rows]
    counts = torch.bincount(codes, minlength=grid.groups)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(len(rows), dtype=torch.int64) - starts[codes]
    later = counts[codes] - 1 - rank
    width = int(counts.max()) if len(rows) else 1
    return Levels(rows, codes, grid.period[rows], rank, later, width, grid.groups)


class Differences:
    """First differences of consecutive level observations."""

    name = "first differences"

    def __init__(self, grid: Layout, obs: Levels):
        index = torch.full((len(grid.codes),), -1, dtype=torch.int64)
        index[obs.rows] = torch.arange(obs.n, dtype=torch.int64)
        previous = grid.rows_at(obs.codes, obs.period - 1)
        previous = torch.where(previous >= 0, index[previous.clamp(min=0)], previous)
        valid = previous >= 0
        self.cur = valid.nonzero().flatten()           # level index of period t
        self.prev = previous[valid]                     # level index of period t - 1
        self.source = self.cur
        self.codes = obs.codes[self.cur]
        self.period = obs.period[self.cur]              # dating of the transformed row
        self.n_levels = obs.n

    @property
    def n(self) -> int:
        return len(self.cur)

    def apply(self, values: Tensor) -> Tensor:
        return values[self.cur] - values[self.prev]

    def adjoint(self, values: Tensor) -> Tensor:
        out = torch.zeros((self.n_levels,) + values.shape[1:], dtype=values.dtype)
        out.index_add_(0, self.cur, values)
        out.index_add_(0, self.prev, -values)
        return out


class OrthogonalDeviations:
    """Forward orthogonal deviations over the later level observations of each panel."""

    name = "forward orthogonal deviations"

    def __init__(self, obs: Levels):
        self.obs = obs
        has_future = obs.later >= 1
        self.source = has_future.nonzero().flatten()
        later = obs.later[self.source].to(torch.float64)
        self.inverse_count = 1.0 / later
        self.scale = torch.sqrt(later / (later + 1.0))
        self.codes = obs.codes[self.source]
        self.period = obs.period[self.source] + 1       # xtabond2 dates F_t at t + 1
        self.cur = self.source
        self.n_levels = obs.n

    @property
    def n(self) -> int:
        return len(self.source)

    def _shape(self, vector: Tensor, values: Tensor) -> Tensor:
        return vector if values.ndim == 1 else vector[:, None]

    def apply(self, values: Tensor) -> Tensor:
        obs = self.obs
        future = within_cumsum(values, obs.codes, obs.rank, obs.groups, obs.width,
                               reverse=True) - values
        src = self.source
        return self._shape(self.scale, values) * (
            values[src] - future[src] * self._shape(self.inverse_count, values))

    def adjoint(self, values: Tensor) -> Tensor:
        obs = self.obs
        direct = torch.zeros((self.n_levels,) + values.shape[1:], dtype=values.dtype)
        direct.index_add_(0, self.source, values * self._shape(self.scale, values))
        spread = torch.zeros_like(direct)
        spread.index_add_(0, self.source,
                          values * self._shape(self.scale * self.inverse_count, values))
        # Each later observation s receives -sum over earlier rows r of c_r / T_r * v_r.
        earlier = within_cumsum(spread, obs.codes, obs.rank, obs.groups, obs.width) - spread
        return direct - earlier

    def grid_adjoint(self, values: Tensor, span: int) -> tuple[Tensor, Tensor]:
        """xtabond2's ``F_g' v``: the adjoint with the weights of a panel spanning all periods.

        xtabond2 builds the transformed-level blocks of H (h = 3) and of the
        homoskedastic AR test from one deviation matrix ``F_g`` of a complete
        panel of ``span`` periods: the row of period t is ``c (e_t - mean of
        e_{t+1} .. e_{span-1})`` with ``c = sqrt(n / (n + 1))``, ``n = span - 1 -
        t``, whatever the panel's own later observations are. It coincides with
        ``adjoint`` for panels observed up to the last period without gaps.
        Returns the values at the level observations [NL, m] and, for the grid
        cells without a level observation that ``F_g'`` reaches (gaps after a
        deviation's source and the periods after the panel's last observation),
        one row per run of such cells: the cells of a run carry equal values, so
        the run is one row times sqrt(run length), which keeps ``B'B`` exact.
        """
        obs = self.obs
        flat = values.reshape(len(values), -1)
        later = (span - 1 - obs.period[self.source]).to(torch.float64)
        scale = torch.sqrt(later / (later + 1.0))
        direct = torch.zeros((self.n_levels, flat.shape[1]), dtype=flat.dtype)
        direct.index_add_(0, self.source, flat * scale[:, None])
        spread = torch.zeros_like(direct)
        spread.index_add_(0, self.source, flat * (scale / later)[:, None])
        inclusive = within_cumsum(spread, obs.codes, obs.rank, obs.groups, obs.width)
        at_levels = direct - (inclusive - spread)
        same = obs.codes[1:] == obs.codes[:-1]
        gap = torch.where(same, obs.period[1:] - obs.period[:-1] - 1, torch.zeros_like(same,
                                                                                    dtype=torch.int64))
        tail = torch.where(obs.later == 0, span - 1 - obs.period, torch.zeros_like(obs.period))
        runs = torch.cat([gap, tail])
        rows = torch.cat([torch.arange(len(gap), dtype=torch.int64),
                          torch.arange(obs.n, dtype=torch.int64)])
        keep = runs > 0
        lengths = runs[keep].to(torch.float64)
        absent = -inclusive[rows[keep]] * torch.sqrt(lengths)[:, None]
        shape = (values.shape[1:] if values.ndim > 1 else ())
        return at_levels.reshape((self.n_levels,) + shape), absent.reshape((-1,) + shape)
