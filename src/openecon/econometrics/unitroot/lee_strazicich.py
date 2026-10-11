"""Lee--Strazicich minimum LM unit-root test with one or two endogenous breaks.

Lee, J. and Strazicich, M. C. (2003). Minimum Lagrange multiplier unit root
test with two structural breaks. Review of Economics and Statistics 85(4),
1082-1089. Lee, J. and Strazicich, M. C. (2013). Minimum LM unit root test
with one structural break. Economics Bulletin 33(4), 2483-2492
(https://www.accessecon.com/Pubs/EB/2013/Volume33/EB-13-V33-I4-P234.pdf).

The procedure follows the authors' GAUSS programs LS_One.g / LS_two.g (Junsoo
Lee, 2005/2008, http://www.cba.ua.edu/~jlee/gauss/, archived at
web.archive.org): break search between round(0.1 T) and round(0.9 T), at
least maxk + 3; second break at least 2 (crash) or 3 (break) periods after
the first; general-to-specific lag choice at every configuration (largest k
<= maxk whose last augmentation term has |t| > 1.645, each k fitted on its own
sample t = k+2..T); minimum of the t ratio of S~_{t-1}. Critical values: see
``break_tables``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot import break_tables
from openecon.econometrics.unitroot.break_search import (
    batch_ols, chunk_size, gauss_round, general_to_specific,
)
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, check_fraction, period_label, regression_table,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.series import one_series
from openecon.engines.distributions import normal_isf

_MODELS = ("crash", "break")
LEVELS = ("1%", "5%", "10%")
SOURCES = [
    "https://doi.org/10.1162/003465303772815961",
    "https://www.accessecon.com/Pubs/EB/2013/Volume33/EB-13-V33-I4-P234.pdf",
    "https://github.com/aptech/tspdlib/blob/master/src/lm_2br.src",
]


def candidate_breaks(total: int, breaks: int, model: str, maxk: int, trim: float) -> Tensor:
    """Break configurations [C, breaks] (1-based TB = last observation of the first regime).

    Scan order is the authors': the first break ascending, then the second.
    """
    first = gauss_round(trim * total)
    last = gauss_round((1.0 - trim) * total)
    if first < maxk + 2:
        first = maxk + 3
    last = min(last, total - 2)
    if last < first:
        return torch.empty((0, breaks), dtype=torch.int64)
    if breaks == 1:
        dates = torch.arange(first, last + 1, dtype=torch.int64)
        return dates[:, None]
    gap = 2 if model == "crash" else 3
    one = torch.arange(first, last + 1, dtype=torch.int64)
    pairs = torch.cartesian_prod(one, one) if one.numel() else torch.empty((0, 2), dtype=torch.int64)
    if pairs.dim() == 1:
        pairs = pairs[None, :]
    return pairs[pairs[:, 1] >= pairs[:, 0] + gap]


def _detrend(y: Tensor, dates: Tensor, model: str) -> tuple[Tensor, Tensor, Tensor]:
    """S~ [C, T], residuals e~ [C, T-1] of Delta y on Delta Z, and Delta Z columns [C, T-1, m].

    Delta Z_t = [1, B_jt, (D_jt)] for t = 2..T with B_jt = 1(t = TB_j + 1),
    D_jt = 1(t >= TB_j + 1); the impulses B are applied as row masks.
    """
    total = y.shape[0]
    count = dates.shape[0]
    dy = y[1:] - y[:-1]
    t = torch.arange(2, total + 1, dtype=torch.int64)
    columns = [torch.ones((count, total - 1), dtype=torch.float64)]
    mask = torch.ones((count, total - 1), dtype=torch.float64)
    for j in range(dates.shape[1]):
        start = dates[:, j:j + 1] + 1
        mask = mask * (t[None, :] != start).to(torch.float64)
        if model == "break":
            columns.append((t[None, :] >= start).to(torch.float64))
    dz = torch.stack(columns, dim=2)
    design = dz * mask[:, :, None]
    q, r = torch.linalg.qr(design)
    theta = (q.transpose(1, 2) @ (dy[None, :] * mask)[:, :, None])
    fitted = (dz @ torch.linalg.solve_triangular(r, theta, upper=True))[:, :, 0]
    resid = (dy[None, :] - fitted) * mask          # the impulse rows fit exactly
    s = torch.cat([torch.zeros((count, 1), dtype=torch.float64), resid.cumsum(dim=1)], dim=1)
    return s, resid, dz


@torch.no_grad()
def ls_statistics(y: Tensor, dates: Tensor, model: str, maxk: int,
                  fixed: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """(min-LM t ratio, selected lag, exact-fit flag) for every configuration in ``dates``.

    With ``fixed`` the lag order is fixed; otherwise the general-to-specific rule.
    """
    total = y.shape[0]
    count = dates.shape[0]
    tau = torch.full((count,), float("nan"), dtype=torch.float64)
    chosen = torch.zeros(count, dtype=torch.int64)
    exact = torch.zeros(count, dtype=torch.bool)
    dy = y[1:] - y[:-1]
    t_index = torch.arange(2, total + 1, dtype=torch.int64)
    orders = [fixed] if fixed is not None else list(range(maxk + 1))
    width = 2 + max(orders) + 2 * dates.shape[1]
    step = chunk_size(total, width + 2)
    threshold = normal_isf(0.05)
    for low in range(0, count, step):
        block = dates[low:low + step]
        size = block.shape[0]
        s, resid, dz = _detrend(y, block, model)
        taus = torch.full((size, maxk + 1), float("nan"), dtype=torch.float64)
        last = torch.full((size, maxk + 1), float("nan"), dtype=torch.float64)
        exacts = torch.zeros((size, maxk + 1), dtype=torch.bool)
        for k in orders:
            rows = slice(k, total - 1)                  # t = k+2..T
            parts = [s[:, k:total - 1, None]]
            parts.extend(resid[:, k - j:total - 1 - j, None] for j in range(1, k + 1))
            parts.append(dz[:, rows, :])
            design = torch.cat(parts, dim=2)
            mask = torch.ones((size, total - 1 - k), dtype=torch.float64)
            for j in range(block.shape[1]):
                mask = mask * (t_index[None, rows] != block[:, j:j + 1] + 1).to(torch.float64)
            fit = batch_ols(design, dy[rows], mask, (0, k) if k else (0,))
            taus[:, k] = fit.t[:, 0]
            if k:
                last[:, k] = fit.t[:, 1]
            exacts[:, k] = fit.exact
        if fixed is not None:
            pick = torch.full((size,), fixed, dtype=torch.int64)
        else:
            pick = general_to_specific(last, threshold, inclusive=False)
        rows_ = torch.arange(size)
        tau[low:low + size] = taus[rows_, pick]
        chosen[low:low + size] = pick
        exact[low:low + size] = exacts[rows_, pick]
    return tau, chosen, exact


def _critical(breaks: int, model: str, fractions: list[float]) -> tuple[dict[str, float], str, str]:
    """(critical values, inference status, rule) from the authors' tables."""
    if model == "crash":
        values = break_tables.LS1_CRASH if breaks == 1 else break_tables.LS2_CRASH
        return (dict(zip(LEVELS, values, strict=True)), "published_table",
                "Model A critical values do not depend on the break fractions.")
    if breaks == 1:
        nodes = sorted(break_tables.LS1_BREAK)
        folded = min(fractions[0], 1.0 - fractions[0])
        clamp = min(max(folded, nodes[0]), nodes[-1])
        for left, right in zip(nodes[:-1], nodes[1:]):
            if clamp <= right:
                w = (clamp - left) / (right - left)
                values = tuple(a + w * (b - a) for a, b in zip(break_tables.LS1_BREAK[left],
                                                              break_tables.LS1_BREAK[right]))
                break
        rule = ("Linear interpolation in lambda = TB/T folded to min(lambda, 1 - lambda), as "
                "Lee and Strazicich (2013, Table 1 note) prescribe.")
        if clamp != folded:
            rule += f" The folded fraction {folded:.3f} lies outside 0.1..0.5; the nearest row is used."
        return dict(zip(LEVELS, values, strict=True)), "published_table_interpolated", rule
    cells = break_tables.LS2_BREAK
    point = (fractions[0], fractions[1])
    nearest = min(cells, key=lambda cell: (cell[0] - point[0]) ** 2 + (cell[1] - point[1]) ** 2)
    rule = (f"Nearest published (lambda1, lambda2) cell {nearest} of Lee and Strazicich (2003, "
            "Table 2) to the estimated fractions; the paper's interpolation rule could not be "
            "confirmed from the journal text, so this is an OpenEconometrics convention.")
    return dict(zip(LEVELS, cells[nearest], strict=True)), "published_table_nearest_cell", rule


def lee_strazicich(data: Any, y: str, *, time: str | None = None, breaks: int = 1,
                   model: str = "crash", maxlag: int = 8, lags: int | None = None,
                   trim: float = 0.10, max_work: int = 20_000_000_000) -> TableSet:
    """Lee--Strazicich minimum LM unit-root test with one or two endogenous breaks.

    Model and statistic
    -------------------
    y_t = delta' Z_t + X_t, X_t = beta X_{t-1} + e_t. ``model="crash"`` (model A)
    has Z_t = [1, t, D_jt], ``"break"`` (model C) adds DT_jt, with
    D_jt = 1(t >= TB_j + 1) and DT_jt = (t - TB_j) D_jt. With delta~ from the
    regression of Delta y_t on Delta Z_t (t = 2..T), S~_t = y_t - psi~ - Z_t delta~
    and psi~ = y_1 - Z_1 delta~, the LM regression is

        Delta y_t = delta' Delta Z_t + phi S~_{t-1} + sum_{j=1..k} g_j Delta S~_{t-j} + u_t,

    t = k+2..T. The statistic is the minimum over break configurations of the
    t ratio of phi. H0: unit root with breaks (beta = 1); H1: trend-break
    stationarity. Small values reject; breaks are allowed under both.

    Parameters
    ----------
    breaks : 1 or 2.
    model : ``"crash"`` (level shifts) or ``"break"`` (level and trend shifts).
    maxlag : largest augmentation order k for the general-to-specific rule
        (default 8, as the authors): at each configuration the largest k whose
        last term has |t| > 1.645, each k on its own sample; 0 if none.
    lags : fixed k instead of the general-to-specific rule (then ``maxlag``
        only sets the first admissible break date, as in the authors' code).
    trim : fraction of each end excluded from the break search (default 0.10).
    max_work : refusal bound on planned row-by-column-squared QR work.

    Critical values
    ---------------
    Model A: Lee and Strazicich (2013, Table 1) one break -4.239/-3.566/-3.211;
    (2003, Table 2) two breaks -4.545/-3.842/-3.504 (1/5/10%). Model C, one
    break: 2013 Table 1 by lambda = TB/T, symmetric in lambda and 1 - lambda and
    linearly interpolated (the authors' note). Model C, two breaks: 2003 Table 2
    has six (lambda1, lambda2) cells; the nearest cell is used and flagged.
    All tables were simulated with T = 100; no p-value is tabulated.

    Returns
    -------
    TableSet with ``test`` (statistic, critical values, break periods, lag),
    ``regression`` (final LM regression at the selected dates and lag, with
    descriptive OLS t ratios; impulse dummies B are explicit there) and, for
    one break, ``search`` (statistic and lag at every candidate date).
    ``attrs`` include ``statistic``, ``p_value`` (None), ``critical_values``,
    ``inference_status``, ``critical_value_rule``, ``break_indices``
    (0-based positions of TB), ``break_periods``, ``break_fractions``, ``lags``.

    Example
    -------
    >>> oe.lee_strazicich(df, "gdp", breaks=2, model="break")
    """
    if breaks not in (1, 2) or isinstance(breaks, bool):
        raise AnalysisError("invalid_option", "breaks must be 1 or 2.")
    check_choice(model, "model", _MODELS)
    trim = check_fraction(trim, "trim")
    maxk = check_count(maxlag, "maxlag")
    fixed = None if lags is None else check_count(lags, "lags")
    if fixed is not None:
        maxk = fixed
    work_limit = check_count(max_work, "max_work", minimum=1)
    x, labels = one_series(data, y, time, "The Lee-Strazicich test", minimum=12)
    total = x.shape[0]
    level = x - x[0]
    # The statistic is invariant to the origin and the unit of the series.
    x = level / float(level.abs().max())
    dates = candidate_breaks(total, breaks, model, maxk, trim)
    count = dates.shape[0]
    columns = 1 + breaks * (2 if model == "break" else 1)
    if count == 0 or total - 1 - maxk - breaks <= 1 + maxk + columns:
        raise AnalysisError("insufficient_observations", "No admissible break configuration "
                            "leaves enough observations; use fewer lags, a smaller trim or a "
                            "longer series.")
    orders = [fixed] if fixed is not None else range(maxk + 1)
    work = count * sum((total - 1 - k) * (1 + k + columns) ** 2 for k in orders)
    if work > work_limit:
        raise AnalysisError("work_limit", f"The break search needs about {work:,} planned "
                            f"row-column products (limit {work_limit:,}); shorten the series, "
                            "use fewer lags or raise max_work.")
    tau, chosen, exact = ls_statistics(x, dates, model, maxk, fixed)
    if bool(exact.any()):
        raise AnalysisError("perfect_fit", "At a candidate break configuration the LM "
                            "regression fits exactly; the series is deterministic.")
    finite = torch.isfinite(tau)
    if not bool(finite.any()):
        raise AnalysisError("collinear_regressors", "The LM regression is rank deficient at "
                            "every candidate break configuration.")
    best = int(torch.argmin(torch.where(finite, tau, torch.full_like(tau, math.inf))))
    statistic = float(tau[best])
    lag = int(chosen[best])
    tb = [int(v) for v in dates[best]]
    fractions = [v / total for v in tb]
    values, status, rule = _critical(breaks, model, fractions)
    positions = [v - 1 for v in tb]
    periods = [period_label(labels, p) for p in positions]

    # Final regression with explicit impulse dummies, for the coefficient table.
    s, resid, _ = _detrend(level, dates[best:best + 1], model)
    rows = slice(lag, total - 1)
    t_index = torch.arange(2, total + 1, dtype=torch.int64)[rows]
    regressors = [s[0, lag:total - 1]]
    names = ["S(t-1)"]
    for j in range(1, lag + 1):
        regressors.append(resid[0, lag - j:total - 1 - j])
        names.append("LD.S" if j == 1 else f"L{j}D.S")
    regressors.append(torch.ones(total - 1 - lag, dtype=torch.float64))
    names.append("Intercept")
    for j, date in enumerate(tb, start=1):
        regressors.append((t_index == date + 1).to(torch.float64))
        names.append(f"B{j}")
    if model == "break":
        for j, date in enumerate(tb, start=1):
            regressors.append((t_index >= date + 1).to(torch.float64))
            names.append(f"D{j}")
    dy = (level[1:] - level[:-1])[rows]
    fit = ols(torch.stack(regressors, dim=1), dy, "The Lee-Strazicich LM regression")
    regression = regression_table(names, fit, note="The t ratio of S(t-1) has the nonstandard "
                                  "minimum-LM distribution: its Student-t p-value and interval "
                                  "are withheld.")
    regression.loc["S(t-1)", ["p_value", "ci_low", "ci_high"]] = None
    period_columns = [f"break{j}_period" for j in range(1, breaks + 1)]
    test = table([[statistic, None, *(values[level] for level in LEVELS), *periods, lag]],
                 columns=["statistic", "p_value", "critical_1pct", "critical_5pct",
                          "critical_10pct", *period_columns, "lags"], index=["min LM t"])
    tables = {"test": test, "regression": regression}
    if breaks == 1:
        order = torch.arange(count)
        tables["search"] = table(
            [[period_label(labels, int(dates[i, 0]) - 1), int(dates[i, 0]) - 1,
              None if not bool(finite[i]) else float(tau[i]), int(chosen[i])] for i in order],
            columns=["break_period", "break_index", "statistic", "lags"])
    notes = [
        "Critical values were simulated by the authors for T = 100; they are finite-sample "
        "values at that size, not exact for other T.",
        rule,
        "Lag choice: largest k <= maxlag with |t| > 1.645 on its last term, re-made at every "
        "configuration on the sample t = k+2..T (Lee and Strazicich's GAUSS code); this rule "
        "over-rejects in finite samples (about 10% for model A, 15% for model C at a nominal 5% "
        "with T = 100 iid random walks)."
        if fixed is None else f"Fixed augmentation k = {fixed}.",
    ]
    return TableSet(
        tables, title=f"Lee-Strazicich minimum LM unit-root test for {y}",
        test="lee_strazicich", series=y, statistic=statistic, p_value=None,
        critical_values=values, inference_status=status, critical_value_rule=rule,
        critical_value_table=("2013 Table 1" if breaks == 1 else "2003 Table 2"),
        distribution="Lee-Strazicich minimum LM (nonstandard)", breaks=breaks, model=model,
        model_label="A (crash)" if model == "crash" else "C (level and trend break)",
        break_indices=positions, break_periods=periods, break_fractions=fractions,
        lags=lag, lag_selection="fixed" if fixed is not None else "general-to-specific t (10%)",
        maxlag=maxk, trim=trim, nobs_series=total, nobs=int(fit.nobs), candidates=count,
        first_candidate=int(dates[0, 0]), last_candidate=int(dates[-1, -1]),
        null="Unit root with break(s) at the estimated dates",
        alternative="Stationary around a broken level/trend",
        label="Minimum LM t over break dates (H0: unit root with breaks)",
        reject={level: statistic < value for level, value in values.items()},
        sources=SOURCES, dtype="float64", device="cpu", estimated_work=work, notes=notes)
