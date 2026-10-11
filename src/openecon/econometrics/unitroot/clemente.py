"""Clemente--Montanes--Reyes / Perron--Vogelsang unit-root tests with mean shifts.

Perron, P. and Vogelsang, T. J. (1992). Nonstationarity and level shifts with
an application to purchasing power parity. JBES 10, 301-320 (one shift).
Clemente, J., Montanes, A. and Reyes, M. (1998). Testing for a unit root in
variables with a double change in the mean. Economics Letters 59, 175-182
(two shifts).

The regressions follow Perron and Ng's RATS procedures for model 0 (iom0alph /
aom0alph, https://blogs.bu.edu/perron/files/2020/05/unit-root.zip): the
innovational-outlier (IO) regression in levels with DU and DTB, the additive-
outlier (AO) two-step detrending with DTB_{t-i}, i = 0..k, and the t-sig lag
rule at every candidate date. The trimming and minimum separation of the two
dates follow Baum's Stata commands clemao1/clemao2/clemio1/clemio2.
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
    batch_ols, chunk_size, general_to_specific,
)
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, check_fraction, period_label, regression_table,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.series import one_series
from openecon.engines.distributions import normal_isf

LEVELS = ("1%", "5%", "10%")
_BATCH = 16
SOURCES = ["https://doi.org/10.1080/07350015.1992.10509907",
           "https://doi.org/10.1016/S0165-1765(98)00052-4",
           "https://blogs.bu.edu/perron/files/2020/05/unit-root.zip"]


def candidate_breaks(total: int, breaks: int, model: str, maxk: int, trim: float,
                     separation: int) -> Tensor:
    """Configurations [C, breaks] of TB (1-based last observation before each shift)."""
    edge = int(trim * total + 0.49)                    # Baum's ntrim
    first = max(edge + 1, maxk + 2)
    last = min(total - edge, total - 2 if model == "io" else total - maxk - 2)
    if last < first:
        return torch.empty((0, breaks), dtype=torch.int64)
    one = torch.arange(first, last + 1, dtype=torch.int64)
    if breaks == 1:
        return one[:, None]
    if one.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    pairs = torch.cartesian_prod(one, one).reshape(-1, 2)
    return pairs[pairs[:, 1] >= pairs[:, 0] + separation]


def _step_columns(t: Tensor, dates: Tensor) -> list[Tensor]:
    return [(t[None, :] > dates[:, j:j + 1]).to(torch.float64) for j in range(dates.shape[1])]


@torch.no_grad()
def cmr_statistics(y: Tensor, dates: Tensor, model: str, maxk: int,
                   fixed: int | None) -> tuple[Tensor, Tensor, Tensor]:
    """(t ratio of alpha - 1, selected lag, exact-fit flag) for every configuration."""
    total = y.shape[0]
    count = dates.shape[0]
    tau = torch.full((count,), float("nan"), dtype=torch.float64)
    chosen = torch.zeros(count, dtype=torch.int64)
    exact = torch.zeros(count, dtype=torch.bool)
    orders = [fixed] if fixed is not None else list(range(maxk + 1))
    width = 2 + max(orders) + dates.shape[1]
    step = chunk_size(total, width + 2)
    threshold = normal_isf(0.05)
    times = torch.arange(1, total + 1, dtype=torch.int64)     # t = 1..T
    for low in range(0, count, step):
        block = dates[low:low + step]
        size = block.shape[0]
        if model == "ao":
            # Step 1: y_t = mu + sum theta_j DU_jt + y~_t on the whole sample.
            steps = _step_columns(times, block)
            design = torch.stack([torch.ones((size, total), dtype=torch.float64), *steps], dim=2)
            q, _ = torch.linalg.qr(design)
            series = y[None, :] - (q @ (q.transpose(1, 2) @ y[None, :, None].expand(size, total, 1)))[:, :, 0]
        else:
            series = y[None, :].expand(size, total)
        diff = series[:, 1:] - series[:, :-1]                  # Delta at t = 2..T
        taus = torch.full((size, maxk + 1), float("nan"), dtype=torch.float64)
        last = torch.full((size, maxk + 1), float("nan"), dtype=torch.float64)
        exacts = torch.zeros((size, maxk + 1), dtype=torch.bool)
        for k in orders:
            t = times[k + 1:]                                  # t = k+2..T
            parts = [series[:, k:total - 1, None]]             # level at t-1
            parts.extend(diff[:, k - j:total - 1 - j, None] for j in range(1, k + 1))
            mask = torch.ones((size, t.shape[0]), dtype=torch.float64)
            if model == "io":
                parts.append(torch.ones((size, t.shape[0], 1), dtype=torch.float64))
                parts.extend(column[:, k + 1:, None] for column in _step_columns(times, block))
                for j in range(block.shape[1]):
                    mask = mask * (t[None, :] != block[:, j:j + 1] + 1).to(torch.float64)
            else:
                for j in range(block.shape[1]):
                    offset = t[None, :] - block[:, j:j + 1] - 1
                    mask = mask * ~((offset >= 0) & (offset <= k))
            fit = batch_ols(torch.cat(parts, dim=2), diff[:, k:], mask.to(torch.float64),
                            (0, k) if k else (0,))
            taus[:, k] = fit.t[:, 0]
            if k:
                last[:, k] = fit.t[:, 1]
            exacts[:, k] = fit.exact
        if fixed is not None:
            pick = torch.full((size,), fixed, dtype=torch.int64)
        else:
            pick = general_to_specific(last, threshold, inclusive=True)
        rows = torch.arange(size)
        tau[low:low + size] = taus[rows, pick]
        chosen[low:low + size] = pick
        exact[low:low + size] = exacts[rows, pick]
    return tau, chosen, exact


def asymptotic_p(statistic: float) -> tuple[float, str | None]:
    """p-value of Vogelsang's PVAL program: linear interpolation between percentiles.

    Outside the 1st..99th percentiles the bound is returned with a note.
    """
    y = break_tables.PV_MODEL0_PERCENTILES
    if statistic < y[0]:
        return 0.01, "The asymptotic p-value is below 0.01."
    if statistic >= y[-1]:
        return 0.99, "The asymptotic p-value is above 0.99."
    for i in range(98):
        if y[i] <= statistic < y[i + 1]:
            return 0.01 * (i + 1) + (statistic - y[i]) * 0.01 / (y[i + 1] - y[i]), None
    return 0.99, None


def _minimum(tau: Tensor, exact: Tensor) -> int:
    if bool(exact.any()):
        raise AnalysisError("perfect_fit", "At a candidate break date the unit-root regression "
                            "fits exactly; the series is deterministic.")
    finite = torch.isfinite(tau)
    if not bool(finite.any()):
        raise AnalysisError("collinear_regressors", "The unit-root regression is rank "
                            "deficient at every candidate break date.")
    return int(torch.argmin(torch.where(finite, tau, torch.full_like(tau, math.inf))))


def clemente(data: Any, y: str, *, time: str | None = None, model: str = "ao",
             breaks: int = 1, maxlag: int = 8, lags: int | None = None, trim: float = 0.05,
             separation: int = 2, calibration: str = "asymptotic", replications: int = 499,
             seed: int = 0, max_work: int = 20_000_000_000) -> TableSet:
    """Unit-root test allowing one or two shifts in the mean at unknown dates.

    Models
    ------
    ``model="io"`` (innovational outlier; Stata clemio1/clemio2):

        y_t = mu + sum_j (d_j DU_jt + delta_j DTB_jt) + alpha y_{t-1}
              + sum_{i=1..k} c_i Delta y_{t-i} + e_t,          t = k+2..T,

    ``model="ao"`` (additive outlier; Stata clemao1/clemao2): first
    y_t = mu + sum_j d_j DU_jt + y~_t on t = 1..T, then

        y~_t = sum_j sum_{i=0..k} w_ji DTB_j,t-i + alpha y~_{t-1}
               + sum_{i=1..k} c_i Delta y~_{t-i} + e_t,          t = k+2..T,

    with DU_jt = 1(t > TB_j) and DTB_jt = 1(t = TB_j + 1). The statistic is
    the minimum over candidate dates of t = (alpha - 1)/se. H0: unit root (a
    random walk with mean shifts entering as impulses); H1: stationarity
    around one or two shifted means. Small values reject.

    Parameters
    ----------
    breaks : 1 (Perron--Vogelsang) or 2 (Clemente--Montanes--Reyes).
    maxlag : largest augmentation k (default 8). At every candidate date k is
        the largest order whose last lag has |t| >= 1.645 (the t-sig rule of
        Perron and Vogelsang and of Perron and Ng's RATS code), else 0.
    lags : fixed k instead.
    trim : fraction of each end excluded, as Baum's Stata commands:
        ntrim = int(trim T + 0.49) and TB in [ntrim + 1, T - ntrim]; also
        TB >= maxlag + 2 and TB <= T - 2 (IO) or T - maxlag - 2 (AO).
    separation : minimum TB_2 - TB_1 (default 2, as clemao2/clemio2).
    calibration : ``"asymptotic"``, ``"monte_carlo"`` or ``"none"``.
    replications, seed : Monte Carlo draws (19..9999) and local generator seed.

    Inference
    ---------
    ``asymptotic`` (one break only): the asymptotic distribution of min
    t_alpha for Perron and Vogelsang's model 0, common to IO and AO,
    tabulated at percentiles 1..99 in Vogelsang's PVAL program; 1/5/10%
    critical values -4.949/-4.444/-4.194 and the program's interpolated
    p-value (bounded to [0.01, 0.99]). That program accompanies procedures
    that search all dates, so a trimmed search is conservative under it.
    With two breaks the Clemente--Montanes--Reyes (1998) table could not be
    retrieved from a primary source: critical values and p-value are None
    and ``inference_status`` says so. ``monte_carlo`` repeats the entire
    procedure (date search and lag rule) on independent Gaussian random
    walks of the same length; p = (1 + #{simulated <= observed})/(B + 1)
    and empirical quantiles. Failures abort; no draw is discarded.

    Returns
    -------
    TableSet with ``test``, the final regression(s) at the selected dates
    (``regression`` for IO; ``level_shifts`` and ``regression`` for AO) and,
    with one break, ``search``. ``attrs``: ``statistic``, ``p_value``,
    ``critical_values``, ``inference_status``, ``break_indices`` (0-based
    position of TB), ``break_periods``, ``lags`` and the calibration record.

    Example
    -------
    >>> oe.clemente(df, "rate", model="io", breaks=2)
    """
    check_choice(model, "model", ("ao", "io"))
    check_choice(calibration, "calibration", ("asymptotic", "monte_carlo", "none"))
    if breaks not in (1, 2) or isinstance(breaks, bool):
        raise AnalysisError("invalid_option", "breaks must be 1 or 2.")
    trim = check_fraction(trim, "trim")
    maxk = check_count(maxlag, "maxlag")
    fixed = None if lags is None else check_count(lags, "lags")
    if fixed is not None:
        maxk = fixed
    separation = check_count(separation, "separation", minimum=2)
    draws = check_count(replications, "replications", minimum=19)
    seed = check_count(seed, "seed")
    work_limit = check_count(max_work, "max_work", minimum=1)
    if draws > 9999 or seed >= 2**63:
        raise AnalysisError("invalid_option", "replications <= 9999 and seed < 2**63 are required.")
    x, labels = one_series(data, y, time, "The Clemente-Montanes-Reyes test", minimum=12)
    total = x.shape[0]
    level = x - x[0]
    x = level / float(level.abs().max())
    dates = candidate_breaks(total, breaks, model, maxk, trim, separation)
    count = dates.shape[0]
    width = 2 + maxk + (breaks if model == "io" else 0)
    if count == 0 or total - 2 - maxk - breaks * (maxk + 1) <= width:
        raise AnalysisError("insufficient_observations", "No admissible break date leaves "
                            "enough observations; use fewer lags, a smaller trim or a longer "
                            "series.")
    orders = [fixed] if fixed is not None else range(maxk + 1)
    work = count * sum((total - 1 - k) * (2 + k + breaks) ** 2 for k in orders)
    total_work = work * (1 + (draws if calibration == "monte_carlo" else 0))
    if total_work > work_limit:
        raise AnalysisError("work_limit", f"The break search needs about {total_work:,} planned "
                            f"row-column products (limit {work_limit:,}); shorten the series, "
                            "use fewer lags or replications, or raise max_work.")
    tau, chosen, exact = cmr_statistics(x, dates, model, maxk, fixed)
    best = _minimum(tau, exact)
    statistic = float(tau[best])
    lag = int(chosen[best])
    tb = [int(v) for v in dates[best]]
    positions = [v - 1 for v in tb]
    periods = [period_label(labels, p) for p in positions]
    notes = []
    record: dict[str, Any] = {}
    p_value = None
    values = None
    if calibration == "asymptotic" and breaks == 1:
        y_ = break_tables.PV_MODEL0_PERCENTILES
        values = {"1%": y_[0], "5%": y_[4], "10%": y_[9]}
        p_value, note = asymptotic_p(statistic)
        status = "asymptotic_published_distribution"
        notes.append("Asymptotic distribution of Perron and Vogelsang's (1992) model-0 min "
                     "t_alpha (IO and AO), Vogelsang's PVAL percentiles; the procedures it "
                     "accompanies search all dates, so a trimmed search is conservative.")
        if note:
            notes.append(note)
    elif calibration == "asymptotic":
        status = ("unavailable: Clemente, Montanes and Reyes (1998) critical values could not "
                  "be obtained from a primary source; use calibration='monte_carlo'")
        notes.append("No critical values: the two-break table of Clemente et al. (1998) was "
                     "not retrievable from the paper or author code.")
    elif calibration == "none":
        status = "statistic only (calibration='none')"
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        simulated = []
        for offset in range(0, draws, _BATCH):
            size = min(_BATCH, draws - offset)
            # Draw-major order and a fixed batch size are part of seeded replay.
            noise = torch.randn((size, total), dtype=torch.float64, generator=generator)
            for path in noise.cumsum(dim=1):
                path = path - path[0]
                path = path / float(path.abs().max())
                sim_tau, _, sim_exact = cmr_statistics(path, dates, model, maxk, fixed)
                if bool(sim_exact.any()) or not bool(torch.isfinite(sim_tau).any()):
                    raise AnalysisError("numerical_failure", "A Monte Carlo draw failed; no "
                                        "draw is discarded, so the calibration is aborted.")
                simulated.append(float(sim_tau[torch.isfinite(sim_tau)].min()))
        null = torch.tensor(simulated, dtype=torch.float64)
        exceed = int((null <= statistic).sum())
        p_value = (exceed + 1) / (draws + 1)
        quantiles = torch.quantile(null, torch.tensor([0.01, 0.05, 0.10], dtype=torch.float64))
        values = dict(zip(LEVELS, quantiles.tolist(), strict=True))
        status = "finite_sample_monte_carlo"
        record = {"replications": draws, "seed": seed, "successful_replications": draws,
                  "failed_replications": 0, "monte_carlo_denominator": draws + 1,
                  "exceedances": exceed, "null_process": "Gaussian random walk, T levels",
                  "failure_policy": "abort; never remove failed draws from the denominator"}
        notes.append("Finite-sample critical values and p-value from repeating the complete "
                     "date search and lag rule on independent Gaussian random walks.")

    tables = {}
    test_columns = ["statistic", "p_value", "critical_1pct", "critical_5pct", "critical_10pct",
                    *[f"break{j}_period" for j in range(1, breaks + 1)], "lags"]
    cvs = [None, None, None] if values is None else [values[level_] for level_ in LEVELS]
    tables["test"] = table([[statistic, p_value, *cvs, *periods, lag]], columns=test_columns,
                           index=["min t"])
    tables.update(_final_regressions(level, tb, lag, model))
    if breaks == 1:
        tables["search"] = table(
            [[period_label(labels, int(dates[i, 0]) - 1), int(dates[i, 0]) - 1,
              None if not math.isfinite(float(tau[i])) else float(tau[i]), int(chosen[i])]
             for i in range(count)],
            columns=["break_period", "break_index", "statistic", "lags"])
    notes.append("Lag choice: largest k <= maxlag with |t| >= 1.645 on its last lag, re-made at "
                 "every candidate date (Perron and Vogelsang's t-sig rule)." if fixed is None
                 else f"Fixed augmentation k = {fixed}.")
    name = {"ao": "additive outlier", "io": "innovational outlier"}[model]
    return TableSet(
        tables, title=f"Clemente-Montanes-Reyes unit-root test ({name}, {breaks} mean "
        f"shift{'s' if breaks == 2 else ''}) for {y}", test="clemente", series=y,
        statistic=statistic, p_value=p_value, critical_values=values,
        reject=None if values is None else {k: statistic < v for k, v in values.items()},
        inference_status=status, calibration=calibration, model=model, breaks=breaks,
        stata=f"clem{model}{breaks}", break_indices=positions, break_periods=periods,
        lags=lag, lag_selection="fixed" if fixed is not None else "t-sig (|t| >= 1.645)",
        maxlag=maxk, trim=trim, separation=separation, nobs_series=total,
        candidates=count, null="Unit root (shifts as impulses under the null)",
        alternative="Stationarity around shifted means",
        label="Minimum t of (alpha - 1) over shift dates (H0: unit root)",
        sources=SOURCES, dtype="float64", device="cpu", estimated_work=total_work,
        **record, notes=notes)


def _final_regressions(level: Tensor, tb: list[int], lag: int, model: str) -> dict[str, Any]:
    """Coefficient tables at the selected dates, with explicit impulse dummies."""
    total = level.shape[0]
    times = torch.arange(1, total + 1, dtype=torch.float64)
    t = times[lag + 1:]
    tables = {}
    if model == "io":
        series = level
    else:
        steps = [(times > date).to(torch.float64) for date in tb]
        design = torch.stack([torch.ones(total, dtype=torch.float64), *steps], dim=1)
        first = ols(design, level, "The level-shift regression")
        if first.exact:
            raise AnalysisError("perfect_fit", "The series is an exact step function.")
        tables["level_shifts"] = regression_table(
            ["Intercept", *[f"DU{j}" for j in range(1, len(tb) + 1)]], first,
            note="Step 1 of the additive-outlier model; level = y - y_1.")
        series = first.resid
    diff = series[1:] - series[:-1]
    regressors, names = [series[lag:total - 1]], ["L1.y"]
    for j in range(1, lag + 1):
        regressors.append(diff[lag - j:total - 1 - j])
        names.append("LD.y" if j == 1 else f"L{j}D.y")
    if model == "io":
        regressors.append(torch.ones(t.shape[0], dtype=torch.float64))
        names.append("Intercept")
        for j, date in enumerate(tb, start=1):
            regressors.append((t > date).to(torch.float64))
            names.append(f"DU{j}")
    impulses = sorted({date + 1 + i for date in tb for i in range(1 if model == "io" else lag + 1)})
    for position in impulses:
        regressors.append((t == position).to(torch.float64))
        names.append(f"DTB(t={position})")
    fit = ols(torch.stack(regressors, dim=1), diff[lag:], "The unit-root regression")
    # The first coefficient is alpha - 1 (the target is the difference).
    regression = regression_table(names, fit, note="L1.y is alpha - 1; its t ratio has the "
                                  "nonstandard break-search distribution.")
    regression.loc["L1.y", ["p_value", "ci_low", "ci_high"]] = None
    tables["regression"] = regression
    return tables
