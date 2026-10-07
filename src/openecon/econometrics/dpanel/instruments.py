"""xtabond2-style instrument specifications and the stacked instrument matrix.

Two kinds of instrument groups (Roodman 2009, "How to do xtabond2"):

``gmm`` (gmmstyle) ``{"columns": [...], "lags": [lo, hi], "equation": e, "collapse": c}``
    Transformed equation (first differences or orthogonal deviations, dated t):
    the levels ``v_{i,t-l}`` for ``l = lo .. hi`` (``hi = None``: every
    available lag). Uncollapsed, each (period t, lag l) pair is its own column,
    zero outside period t; collapsed, one column per lag holds ``v_{i,t-l}`` in
    every period. Missing values (absent periods) are zeros.
    ``equation="both"`` (default) in system GMM adds, for the level equation,
    the single lagged difference ``v_{i,t-m} - v_{i,t-m-1}`` with ``m = lo - 1``
    (``m = 0`` when ``lo = 0``), one column per period or one collapsed column;
    the deeper lagged differences are redundant given the transformed-equation
    instruments (xtabond2's "extra" Blundell-Bond instruments).
    ``equation="level"`` (system GMM) instruments only the level equation, with
    the differences ``v_{i,t-l} - v_{i,t-l-1}`` for ``l = lo .. hi``: here the
    lags count the lags of the difference itself, exactly as xtabond2's
    ``gmm(v, lag(lo hi) eq(level))``.
    ``equation="diff"`` instruments only the transformed equation. In
    difference GMM ``"both"`` means ``"diff"``. Default lags ``[1, None]``.

``iv`` (ivstyle) ``{"columns": [...], "equation": e, "passthru": p}``
    One column per variable. In the transformed equation the instrument is
    transformed like the regressors; ``passthru=True`` uses its untransformed
    level at the date of the transformed row (period t for differences, t + 1
    for orthogonal deviations, which xtabond2 stores one period late; zero when
    that period is absent). In the level equation it enters as is. With
    ``equation="both"`` (default) the transformed and level values share ONE
    column, as in xtabond2 (which rejects ``passthru`` with ``"both"`` in system
    GMM, as OpenEconometrics does).

The stacked instrument matrix has the transformed rows first and the level
rows after them. Columns that are zero in every row are dropped (xtabond2
does the same), so ``n_instruments`` counts the columns that carry data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES

# Upper bound on the number of instrument columns (before zero columns are dropped).
MAX_INSTRUMENTS = 2000
_EQUATIONS = ("diff", "level", "both")


@dataclass(frozen=True)
class GmmGroup:
    columns: tuple[str, ...]
    lo: int
    hi: int | None
    equation: str
    collapse: bool

    def label(self) -> str:
        hi = "." if self.hi is None else str(self.hi)
        parts = [", ".join(self.columns), f"lags {self.lo}-{hi}", self.equation]
        if self.collapse:
            parts.append("collapsed")
        return f"gmm({'; '.join(parts)})"


@dataclass(frozen=True)
class IvGroup:
    columns: tuple[str, ...]
    equation: str
    passthru: bool = False

    def label(self) -> str:
        parts = [", ".join(self.columns), self.equation] + (["passthru"] if self.passthru else [])
        return f"iv({'; '.join(parts)})"


def _invalid(message: str) -> AnalysisError:
    return AnalysisError("invalid_spec", message)


def _columns(entry: dict, where: str) -> tuple[str, ...]:
    columns = entry.get("columns")
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, list) or not columns or not all(
            isinstance(name, str) and name.strip() for name in columns):
        raise _invalid(f"{where} needs 'columns': a non-empty list of column names, for example "
                       "{'columns': ['y']}.")
    if len(set(columns)) != len(columns):
        raise _invalid(f"{where} lists a column twice.")
    return tuple(columns)


def _equation(entry: dict, where: str, system: bool) -> str:
    equation = entry.get("equation", "both")
    if equation not in _EQUATIONS:
        raise _invalid(f"{where}: 'equation' must be 'diff', 'level' or 'both'.")
    if not system:
        if equation == "level":
            raise _invalid(f"{where} asks for level-equation instruments, which exist only in "
                           "system GMM; pass system=True or use equation='diff'.")
        equation = "diff"
    return equation


def _entries(value: Any, kind: str) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(entry, dict) for entry in value):
        raise _invalid(f"{kind} must be a list of instrument groups (dictionaries).")
    return value


def parse_gmm(value: Any, *, system: bool, collapse: bool) -> list[GmmGroup]:
    groups = []
    for number, entry in enumerate(_entries(value, "gmm"), start=1):
        where = f"gmm group {number}"
        unknown = set(entry) - {"columns", "lags", "equation", "collapse"}
        if unknown:
            raise _invalid(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))} "
                           "(allowed: columns, lags, equation, collapse).")
        lags = entry.get("lags", [1, None])
        if (not isinstance(lags, (list, tuple)) or len(lags) != 2
                or not isinstance(lags[0], int) or isinstance(lags[0], bool)
                or not (lags[1] is None or (isinstance(lags[1], int)
                                            and not isinstance(lags[1], bool)))):
            raise _invalid(f"{where}: 'lags' must be [lo, hi] with integers (hi may be None "
                           "for all available lags).")
        lo, hi = int(lags[0]), lags[1]
        if lo < 0 or (hi is not None and hi < lo):
            raise _invalid(f"{where}: 'lags' needs 0 <= lo <= hi (forward lags are not "
                           "supported).")
        equation = _equation(entry, where, system)
        if equation == "both" and lo == 0 and hi == 0:
            raise _invalid(f"{where}: with lags [0, 0] the level-equation instrument would be a "
                           "forward difference, which is not supported; use equation='diff' or "
                           "a separate group with equation='level'.")
        group_collapse = entry.get("collapse", collapse)
        if not isinstance(group_collapse, bool):
            raise _invalid(f"{where}: 'collapse' must be true or false.")
        groups.append(GmmGroup(_columns(entry, where), lo, hi, equation, group_collapse))
    return groups


def parse_iv(value: Any, *, system: bool) -> list[IvGroup]:
    groups = []
    for number, entry in enumerate(_entries(value, "iv"), start=1):
        where = f"iv group {number}"
        unknown = set(entry) - {"columns", "equation", "passthru"}
        if unknown:
            raise _invalid(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))} "
                           "(allowed: columns, equation, passthru).")
        passthru = entry.get("passthru", False)
        if not isinstance(passthru, bool):
            raise _invalid(f"{where}: 'passthru' must be true or false.")
        equation = _equation(entry, where, system)
        if passthru and system and equation == "both":
            raise _invalid(f"{where}: passthru is not valid with equation='both' in system GMM "
                           "(as in xtabond2); give separate groups with equation='diff' and "
                           "equation='level'.")
        groups.append(IvGroup(_columns(entry, where), equation, passthru))
    return groups


@dataclass
class Rows:
    """Row structure the instruments are built on."""

    codes_t: Tensor            # [NT] panel of each transformed row
    period_t: Tensor           # [NT] dating of each transformed row
    source_t: Tensor           # [NT] level observation each transformed row comes from
    codes_l: Tensor            # [NL] panel of each level row
    period_l: Tensor           # [NL] period of each level row
    system: bool

    @property
    def n_t(self) -> int:
        return len(self.codes_t)

    @property
    def n_l(self) -> int:
        return len(self.codes_l) if self.system else 0


@dataclass
class Instruments:
    z: Tensor                                  # [NT + NL, L] stacked instruments
    groups: list[str]                          # group key of each column
    equations: list[str]                       # 'diff', 'level' or 'both' per column
    labels: dict[str, str] = field(default_factory=dict)
    planned: int = 0                           # columns before zero columns were dropped
    zero_columns: int = 0


@dataclass
class _Plan:
    """One block of columns: a key, its width and a filler writing into Z."""

    key: str
    equation: str
    width: int
    fill: Callable[[Tensor, int], None]


def _periods(period: Tensor) -> list[int]:
    return torch.unique(period).tolist()


Lookup = Callable[[str, Tensor, Tensor], tuple[Tensor, Tensor]]


def _lag_plan(key: str, equation: str, name: str, codes: Tensor, period: Tensor,
              row_offset: int, lags: list[int], collapse: bool, gap: int,
              values: Callable[[str, Tensor, Tensor, int], tuple[Tensor, Tensor]]) -> _Plan | None:
    """One GMM-style block over ``rows[row_offset:]``: a column per (period, lag) or per lag.

    ``values(name, codes, period, lag)`` gives the instrument of every row at
    that lag and whether it exists; ``gap`` is how many periods before
    ``t - lag`` the value also needs (0 for levels, 1 for differences), so
    columns that can never be filled are not planned.
    """
    if not lags or not len(period):
        return None
    periods = _periods(period)
    if collapse:
        table = {(None, lag): j for j, lag in enumerate(lags)}
    else:
        pairs = [(t, lag) for t in periods for lag in lags if t - lag - gap >= 0]
        table = {pair: j for j, pair in enumerate(pairs)}
    if not table:
        return None
    index = torch.full((max(periods) + 1, len(lags)), -1, dtype=torch.int64)
    first = lags[0]
    for (t, lag), j in table.items():
        if t is None:
            index[:, lag - first] = j
        else:
            index[t, lag - first] = j

    def fill(z: Tensor, offset: int) -> None:
        for lag in lags:
            found, present = values(name, codes, period, lag)
            column = index[period, lag - first]
            where = (present & (column >= 0)).nonzero().flatten()
            z[row_offset + where, offset + column[where]] = found[where]

    return _Plan(key, equation, len(table), fill)


def _gmm_plans(group: GmmGroup, key: str, rows: Rows, lookup: Lookup) -> list[_Plan]:
    """Columns of one gmmstyle group (see the module docstring for the semantics)."""

    def level(name: str, codes: Tensor, period: Tensor, lag: int) -> tuple[Tensor, Tensor]:
        return lookup(name, codes, period - lag)

    def difference(name: str, codes: Tensor, period: Tensor, lag: int) -> tuple[Tensor, Tensor]:
        now, present = lookup(name, codes, period - lag)
        before, earlier = lookup(name, codes, period - lag - 1)
        return now - before, present & earlier

    plans: list[_Plan | None] = []
    if group.equation in ("diff", "both") and rows.n_t:
        top = int(rows.period_t.max())
        hi = top if group.hi is None else min(group.hi, top)
        lags = list(range(group.lo, hi + 1))
        plans += [_lag_plan(key, "diff", name, rows.codes_t, rows.period_t, 0, lags,
                            group.collapse, 0, level) for name in group.columns]
    if rows.system and rows.n_l and group.equation in ("level", "both"):
        if group.equation == "both":
            lags = [group.lo - 1 if group.lo >= 1 else 0]
        else:
            top = int(rows.period_l.max()) - 1
            hi = top if group.hi is None else min(group.hi, top)
            lags = list(range(group.lo, hi + 1))
        plans += [_lag_plan(key, "level", name, rows.codes_l, rows.period_l, rows.n_t, lags,
                            group.collapse, 1, difference) for name in group.columns]
    return [plan for plan in plans if plan is not None]


def _iv_plans(group: IvGroup, key: str, rows: Rows, level_values: Callable[[str], Tensor],
              transform: Callable[[Tensor], Tensor], lookup: Lookup) -> list[_Plan]:
    plans = []
    for name in group.columns:
        def fill(z: Tensor, offset: int, name: str = name) -> None:
            if group.equation in ("diff", "both") and rows.n_t:
                if group.passthru:          # the level at the transformed row's date
                    found, _ = lookup(name, rows.codes_t, rows.period_t)
                    z[:rows.n_t, offset] = found
                else:
                    z[:rows.n_t, offset] = transform(level_values(name))
            if group.equation in ("level", "both") and rows.system:
                z[rows.n_t:, offset] = level_values(name)
        plans.append(_Plan(key, group.equation, 1, fill))
    return plans


def build(*, rows: Rows, gmm: Sequence[GmmGroup], iv: Sequence[IvGroup], lookup: Lookup,
          level_values: Callable[[str], Tensor], transform: Callable[[Tensor], Tensor],
          extra_iv: Sequence[tuple[str, str, str, Tensor]] = ()) -> Instruments:
    """Assemble the stacked instrument matrix.

    ``lookup(name, codes, period)`` returns a column's values at grid cells and
    their presence; ``level_values(name)`` the column on the level observations;
    ``transform`` maps a level vector to the transformed rows. ``extra_iv``
    adds IV-style blocks given directly as ``(key, label, equation, values)``
    with ``values`` [NL, m] on the level observations (time dummies, constant).
    """
    plans: list[_Plan] = []
    labels: dict[str, str] = {}
    for number, group in enumerate(gmm, start=1):
        key = f"gmm{number}"
        labels[key] = group.label()
        plans.extend(_gmm_plans(group, key, rows, lookup))
    for number, group in enumerate(iv, start=1):
        key = f"iv{number}"
        labels[key] = group.label()
        plans.extend(_iv_plans(group, key, rows, level_values, transform, lookup))
    for key, label, equation, values in extra_iv:
        labels[key] = label
        for j in range(values.shape[1]):
            def fill(z: Tensor, offset: int, column: Tensor = values[:, j],
                     equation: str = equation) -> None:
                if equation in ("diff", "both") and rows.n_t:
                    z[:rows.n_t, offset] = transform(column)
                if equation in ("level", "both") and rows.system:
                    z[rows.n_t:, offset] = column
            plans.append(_Plan(key, equation, 1, fill))
    planned = sum(plan.width for plan in plans)
    total_rows = rows.n_t + rows.n_l
    if planned > MAX_INSTRUMENTS or total_rows * planned * 8 > _MAX_DESIGN_BYTES:
        raise AnalysisError(
            "instrument_count_too_large",
            f"The specification creates {planned:,} instrument columns over {total_rows:,} "
            f"stacked rows (limits: {MAX_INSTRUMENTS:,} columns and a 256 MiB instrument "
            "matrix). Use collapse=True, cap the GMM lags (for example 'lags': [2, 4]) or "
            "instrument fewer variables GMM-style.")
    if planned == 0:
        raise AnalysisError("underidentified", "The specification has no instruments. Give GMM- "
                            "or IV-style instrument groups (gmm=[...], iv=[...]).")
    z = torch.zeros((total_rows, planned), dtype=torch.float64)
    keys, equations, offset = [], [], 0
    for plan in plans:
        plan.fill(z, offset)
        keys.extend([plan.key] * plan.width)
        equations.extend([plan.equation] * plan.width)
        offset += plan.width
    nonzero = (z != 0).any(dim=0)
    zero = int((~nonzero).sum())
    if zero:
        kept = nonzero.nonzero().flatten()
        z = z[:, kept]
        keys = [keys[j] for j in kept.tolist()]
        equations = [equations[j] for j in kept.tolist()]
    return Instruments(z, keys, equations, labels, planned, zero)
