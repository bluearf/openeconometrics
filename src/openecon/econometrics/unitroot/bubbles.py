"""Phillips--Shi--Yu SADF/GSADF explosive-root tests and BSADF date stamping.

Phillips, P. C. B., Shi, S. and Yu, J. (2015). Testing for multiple bubbles:
historical episodes of exuberance and collapse in the S&P 500. International
Economic Review 56(4), 1043-1078 (Cowles Foundation Discussion Paper 1843,
https://cowles.yale.edu/sites/default/files/2022-08/d1843.pdf).
Phillips, P. C. B. and Shi, S. (2020). Real time monitoring of asset markets:
bubbles and crises. Handbook of Statistics 42, 61-80 (CFDP 2152,
https://cowles.yale.edu/sites/default/files/2022-08/d2152.pdf), section 5:
composite wild bootstrap with family-wise size control.

Window statistics. For a window of levels y_s..y_e the ADF regression

    Delta y_t = a + b y_{t-1} + sum_{j=1..k} c_j Delta y_{t-j} + u_t,  t = s+1+k..e,

gives ADF(s, e) = b / se(b). Every window shares the cross products of its
rows, so they are accumulated once per window start by cumulative sums over t,
on levels measured from y_s (the regression has an intercept, so the shift
leaves the statistic unchanged and keeps the sums well scaled), and each
window's small normal-equation system is solved by a Cholesky factor of its
diagonally scaled Gram matrix. tests/test_econ_bubble_tests.py checks every
window statistic against separate NumPy least-squares fits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, period_label,
)
from openecon.econometrics.unitroot.series import one_series

_BATCH = 64
# Elements of one block of cumulative cross products [B, starts, T, (p+1)^2].
_BLOCK = 1 << 22
SOURCES = ["https://doi.org/10.1111/iere.12132",
           "https://cowles.yale.edu/sites/default/files/2022-08/d1843.pdf",
           "https://doi.org/10.1016/bs.host.2018.12.002",
           "https://cowles.yale.edu/sites/default/files/2022-08/d2152.pdf"]
QUANTILES = (0.90, 0.95, 0.99)


def default_window(total: int) -> int:
    """floor(T (0.01 + 1.8 / sqrt(T))), the PSY minimum window (psymonitor's swindow0)."""
    return int(math.floor(total * (0.01 + 1.8 / math.sqrt(total))))


def null_paths(total: int, replications: int, generator: torch.Generator) -> Tensor:
    """PSY null: y_t = d T^-eta + y_{t-1} + e_t with d = eta = 1, e ~ N(0, 1), y_0 = 0.

    Draws are made in blocks of ``_BATCH`` paths in draw-major order (seeded replay).
    """
    blocks = []
    for offset in range(0, replications, _BATCH):
        size = min(_BATCH, replications - offset)
        noise = torch.randn((size, total), dtype=torch.float64, generator=generator)
        blocks.append((noise + 1.0 / total).cumsum(dim=1))
    return torch.cat(blocks)


@torch.no_grad()
def recursive_statistics(paths: Tensor, window: int, lags: int) -> dict[str, Tensor]:
    """ADF statistics of every window of at least ``window`` levels, reduced to PSY sequences.

    paths [B, T]. Returns ``bsadf`` [B, T - window + 1] (window ends e = window-1..T-1,
    0-based), ``start`` (0-based start of the maximizing window), ``sadf``,
    ``gsadf``, ``adf`` (whole sample) [B] and ``undefined`` (count of windows
    whose regression is singular or fits exactly). Undefined windows are
    excluded from the maxima.
    """
    batch, total = paths.shape
    width = lags + 2
    diff = torch.cat([torch.zeros((batch, 1), dtype=torch.float64),
                      paths[:, 1:] - paths[:, :-1]], dim=1)          # Delta y at t (t >= 1)
    ends = total - window + 1
    bsadf = torch.full((batch, ends), -math.inf, dtype=torch.float64)
    start = torch.zeros((batch, ends), dtype=torch.int64)
    sadf_row = None
    undefined = torch.zeros(batch, dtype=torch.int64)
    t = torch.arange(total)
    # Lagged differences at row t: Delta y_{t-j}; rows t < j + 1 are never used.
    lagged = [torch.cat([torch.zeros((batch, j), dtype=torch.float64), diff[:, :total - j]], dim=1)
              for j in range(1, lags + 1)]
    step = max(1, _BLOCK // (batch * total * (width + 1) ** 2))
    for low in range(0, ends, step):
        starts = torch.arange(low, min(low + step, ends))
        count = starts.shape[0]
        shift = paths[:, starts]                                        # y_s  [B, S]
        level = paths[:, None, :-1] - shift[:, :, None]                 # y_{t-1} - y_s, t = 1..T-1
        level = torch.cat([torch.zeros((batch, count, 1), dtype=torch.float64), level], dim=2)
        used = (t[None, :] >= starts[:, None] + 1 + lags).to(torch.float64)   # [S, T]
        columns = [level, torch.ones((batch, count, total), dtype=torch.float64)]
        columns.extend(item[:, None, :].expand(batch, count, total) for item in lagged)
        columns.append(diff[:, None, :].expand(batch, count, total))
        z = torch.stack(columns, dim=3) * used[None, :, :, None]        # [B, S, T, p+1]
        gram = (z[..., :, None] * z[..., None, :]).cumsum(dim=2)        # [B, S, T, p+1, p+1]
        rows = used.cumsum(dim=1)                                       # [S, T] rows so far
        e = t[None, :]
        valid_window = (e - starts[:, None] + 1 >= window)              # [S, T]
        xx = gram[..., :width, :width]
        xy = gram[..., :width, width]
        yy = gram[..., width, width]
        statistic = _t_ratio(xx, xy, yy, rows[None, :, :].expand(batch, count, total), width)
        ok = valid_window[None, :, :] & torch.isfinite(statistic)
        undefined += (valid_window[None, :, :] & ~torch.isfinite(statistic)).sum(dim=(1, 2))
        statistic = torch.where(ok, statistic, torch.full_like(statistic, -math.inf))
        # Columns e = window-1..T-1 of the BSADF sequence.
        tail = statistic[:, :, window - 1:]                             # [B, S, ends]
        best, where = tail.max(dim=1)
        better = best > bsadf
        bsadf = torch.where(better, best, bsadf)
        start = torch.where(better, starts[where], start)
        if low == 0:
            sadf_row = tail[:, 0, :]
    adf = sadf_row[:, -1]
    sadf = sadf_row.max(dim=1).values
    gsadf = bsadf.max(dim=1).values
    return {"bsadf": bsadf, "start": start, "sadf": sadf, "gsadf": gsadf, "adf": adf,
            "sadf_sequence": sadf_row, "undefined": undefined}


def _t_ratio(xx: Tensor, xy: Tensor, yy: Tensor, rows: Tensor, width: int) -> Tensor:
    """t ratio of the first coefficient from cross products; NaN when undefined."""
    if width == 2:
        n = rows
        su, s1 = xx[..., 0, 0], xx[..., 0, 1]
        sd, sud, sdd = xy[..., 1], xy[..., 0], yy
        safe = torch.where(n > 0, n, torch.ones_like(n))
        sxx = su - s1.square() / safe
        sxy = sud - s1 * sd / safe
        syy = sdd - sd.square() / safe
        beta = sxy / sxx
        ssr = syy - beta * sxy
        df = n - 2
        resolved = (df > 0) & (sxx > 1e-12 * su.clamp_min(1e-300)) & (ssr > 1e-12 * sdd)
        s2 = ssr / torch.where(df > 0, df, torch.ones_like(df))
        out = beta / (s2 / sxx).sqrt()
        return torch.where(resolved, out, torch.full_like(out, float("nan")))
    diagonal = xx.diagonal(dim1=-2, dim2=-1).clamp_min(1e-300)
    scale = diagonal.rsqrt()
    scaled = xx * scale[..., :, None] * scale[..., None, :]
    factor, info = torch.linalg.cholesky_ex(scaled)
    good = (info == 0) & (factor.diagonal(dim1=-2, dim2=-1) > 1e-7).all(dim=-1)
    eye = torch.eye(width, dtype=torch.float64)
    factor = torch.where(good[..., None, None], factor, eye.expand_as(factor))
    rhs = torch.stack([xy * scale, eye[0].expand_as(xy)], dim=-1)
    solved = torch.cholesky_solve(rhs, factor)
    beta = solved[..., :, 0] * scale
    inverse00 = solved[..., 0, 1] * scale[..., 0] * scale[..., 0]
    ssr = yy - (beta * xy).sum(dim=-1)
    df = rows - width
    resolved = good & (df > 0) & (ssr > 1e-12 * yy)
    s2 = ssr / torch.where(df > 0, df, torch.ones_like(df))
    out = beta[..., 0] / (s2 * inverse00).sqrt()
    return torch.where(resolved, out, torch.full_like(out, float("nan")))


def bootstrap_paths(y: Tensor, lags: int, replications: int,
                    generator: torch.Generator) -> tuple[Tensor, dict[str, Any]]:
    """Phillips--Shi (2020) composite wild bootstrap samples under the unit-root null.

    Step 1: Delta y_t = a + sum_{j<=k} phi_j Delta y_{t-j} + e_t on t = k+2..T.
    Step 2: Delta y*_t = sum phi_j Delta y*_{t-j} + w_t e_l(t), w_t ~ N(0, 1) and
    l(t) drawn uniformly with replacement, initial values y*_i = y_i for
    i = 1..k+1. The AR recursion is solved as one banded lower-triangular system.
    """
    total = y.shape[0]
    dy = y[1:] - y[:-1]
    rows = total - 1 - lags
    design = [torch.ones(rows, dtype=torch.float64)]
    design.extend(dy[lags - j:total - 1 - j] for j in range(1, lags + 1))
    x = torch.stack(design, dim=1)
    q, r = torch.linalg.qr(x)
    coefficient = torch.linalg.solve_triangular(r, q.T @ dy[lags:, None], upper=True)[:, 0]
    resid = dy[lags:] - x @ coefficient
    phi = coefficient[1:]
    size = total - 1
    system = torch.eye(size, dtype=torch.float64)
    index = torch.arange(lags, size)
    for j in range(1, lags + 1):
        system[index, index - j] = -phi[j - 1]
    blocks = []
    for offset in range(0, replications, _BATCH):
        count = min(_BATCH, replications - offset)
        picks = torch.randint(0, rows, (count, size - lags), generator=generator)
        weights = torch.randn((count, size - lags), dtype=torch.float64, generator=generator)
        shocks = torch.cat([dy[:lags].expand(count, lags), weights * resid[picks]], dim=1)
        increments = torch.linalg.solve_triangular(system, shocks.T, upper=False).T
        blocks.append(torch.cat([y[:1].expand(count, 1), y[0] + increments.cumsum(dim=1)], dim=1))
    record = {"null_regression": "Delta y on constant and lags (Phillips and Shi 2020, step 1)",
              "ar_coefficients": phi.tolist(), "residuals": rows}
    return torch.cat(blocks), record


def episodes(bsadf: Tensor, cv: Tensor, minimum: float) -> list[tuple[int, int | None]]:
    """PSY first-crossing date stamps as (origination, termination) positions in the sequence.

    Origination: first point with BSADF > cv. Termination: first point at or after
    origination + ceil(minimum) with BSADF < cv (None if the episode is ongoing).
    The search for the next origination restarts after the termination point.
    """
    above = bsadf > cv
    below = bsadf < cv
    count = bsadf.shape[0]
    index = torch.arange(count)
    found = []
    position = 0
    skip = int(math.ceil(minimum))
    while position < count:
        candidates = above & (index >= position)
        if not bool(candidates.any()):
            break
        origin = int(torch.argmax(candidates.to(torch.int8)))
        ends = below & (index >= origin + skip)
        if not bool(ends.any()):
            found.append((origin, None))
            break
        end = int(torch.argmax(ends.to(torch.int8)))
        found.append((origin, end))
        position = end + 1
    return found


def psy(data: Any, y: str, *, time: str | None = None, lags: int = 0,
        min_window: int | None = None, calibration: str = "monte_carlo",
        replications: int = 999, seed: int = 0, level: float = 0.95,
        min_duration: float | None = None, horizon: int | None = None,
        max_work: int = 20_000_000_000) -> TableSet:
    """Phillips--Shi--Yu SADF and GSADF tests for explosive behaviour with BSADF date stamps.

    Statistics
    ----------
    ADF(s, e) is the t ratio of b in Delta y_t = a + b y_{t-1} +
    sum_{j=1..lags} c_j Delta y_{t-j} + u_t fitted to the levels y_s..y_e.
    With w the minimum window (levels), BSADF(e) = max_{s <= e-w+1} ADF(s, e),
    SADF = max_e ADF(0, e) (Phillips, Wu and Yu's forward recursion) and GSADF
    = max_e BSADF(e). H0: unit root (random walk with asymptotically negligible
    drift); H1: mildly explosive episodes. Large values reject (right tail).

    Parameters
    ----------
    lags : fixed augmentation order (default 0, as PSY's simulations and
        psymonitor/exuber).
    min_window : minimum window w in levels; default floor(T (0.01 + 1.8/sqrt(T))).
    calibration : ``"monte_carlo"`` (default): finite-sample null draws
        y_t = 1/T + y_{t-1} + e_t, e ~ N(0, 1) (PSY's d = eta = 1), the same T,
        w and lags; critical values are empirical quantiles and the BSADF
        sequence uses the quantile at each end date. ``"wild_bootstrap"``:
        Phillips and Shi (2020) composite bootstrap from the data's null
        regression; the date-stamping critical value is the ``level`` quantile
        of max BSADF over the first ``horizon`` monitoring points (family-wise
        size over that window), also used as GSADF critical value when
        ``horizon`` spans the sample. ``"none"``: statistics only.
    replications, seed : draws (19..9999; default 999) and local CPU seed.
    level : date-stamping quantile, 0.90, 0.95 (default) or 0.99.
    min_duration : minimum episode length delta log(T) in observations
        before a termination is accepted; default log(T) (delta = 1).
    horizon : bootstrap family-wise window tau_b (monitoring points); default
        all T - w + 1 points.

    Returns
    -------
    TableSet with ``tests`` (ADF, SADF, GSADF: statistic, p-value, 90/95/99%
    critical values), ``sequence`` (BSADF, its critical value and the start of
    the maximizing window at every end date) and ``episodes`` (origination,
    termination, duration, peak). p-values are (1 + #{null >= observed})/(B + 1).
    Failures abort; no draw is discarded.

    Example
    -------
    >>> result = oe.psy(df, "price_dividend", time="month", seed=7)
    >>> result["episodes"]
    """
    check_choice(calibration, "calibration", ("monte_carlo", "wild_bootstrap", "none"))
    lags = check_count(lags, "lags")
    draws = check_count(replications, "replications", minimum=19)
    seed = check_count(seed, "seed")
    work_limit = check_count(max_work, "max_work", minimum=1)
    if draws > 9999 or seed >= 2**63:
        raise AnalysisError("invalid_option", "replications <= 9999 and seed < 2**63 are required.")
    if not isinstance(level, float) or level not in QUANTILES:
        raise AnalysisError("invalid_option", "level must be 0.90, 0.95 or 0.99.")
    x, labels = one_series(data, y, time, "The PSY bubble test", minimum=12)
    total = x.shape[0]
    window = default_window(total) if min_window is None else check_count(min_window, "min_window")
    if window < max(6, 2 * lags + 4) or window > total:
        raise AnalysisError("invalid_option", f"min_window must lie in [{max(6, 2 * lags + 4)}, "
                            f"{total}] for {lags} lag(s) and {total} observations.")
    monitoring = total - window + 1
    if horizon is None:
        horizon = monitoring
    horizon = check_count(horizon, "horizon", minimum=1)
    if horizon > monitoring:
        raise AnalysisError("invalid_option", f"horizon must be at most {monitoring} monitoring "
                            "points.")
    if min_duration is None:
        min_duration = math.log(total)
    if (not isinstance(min_duration, (int, float)) or isinstance(min_duration, bool)
            or not 0 <= min_duration < total):
        raise AnalysisError("invalid_option", "min_duration must be a number of observations "
                            "in [0, T).")
    windows = monitoring * (monitoring + 1) // 2
    simulated = 0 if calibration == "none" else draws
    work = (1 + simulated) * (windows * (lags + 3) ** 2 + total * monitoring * (lags + 3) ** 2)
    if work > work_limit:
        raise AnalysisError("work_limit", f"The recursive regressions need about {work:,} planned "
                            f"operations (limit {work_limit:,}); shorten the series, raise "
                            "min_window, use fewer replications or raise max_work.")
    level_ = x - x[0]
    scale = float(level_.abs().max())
    observed = recursive_statistics((level_ / scale)[None, :], window, lags)
    if int(observed["undefined"][0]) == windows:
        raise AnalysisError("perfect_fit", "Every window regression is singular or fits exactly; "
                            "the series is deterministic.")
    names = ["ADF", "SADF", "GSADF"]
    values = [float(observed["adf"][0]), float(observed["sadf"][0]), float(observed["gsadf"][0])]
    bsadf = observed["bsadf"][0]
    q = torch.tensor(QUANTILES, dtype=torch.float64)
    record: dict[str, Any] = {}
    notes = []
    p_values = [None, None, None]
    critical = [[None] * 3 for _ in names]
    cv_sequence = None
    family_cv = None
    if calibration != "none":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if calibration == "monte_carlo":
            paths = null_paths(total, draws, generator)
            record.update(null_process="y_t = d T^-eta + y_{t-1} + e_t, d = eta = 1, "
                          "e ~ iid N(0, 1), y_0 = 0 (PSY 2015, section 3)")
        else:
            paths, details = bootstrap_paths(level_ / scale, lags, draws, generator)
            record.update(null_process="Phillips-Shi (2020) composite wild bootstrap", **details)
        null = {key: [] for key in ("adf", "sadf", "gsadf", "bsadf")}
        for offset in range(0, draws, _BATCH):
            stats = recursive_statistics(paths[offset:offset + _BATCH], window, lags)
            if bool((stats["undefined"] > 0).any()):
                raise AnalysisError("numerical_failure", "A null draw produced an undefined window "
                                    "regression; no draw is discarded, so the calibration is "
                                    "aborted.")
            for key in null:
                null[key].append(stats[key])
        null = {key: torch.cat(value) for key, value in null.items()}
        for i, key in enumerate(("adf", "sadf", "gsadf")):
            exceed = int((null[key] >= values[i]).sum())
            p_values[i] = (exceed + 1) / (draws + 1)
            critical[i] = torch.quantile(null[key], q).tolist()
        cv_sequence = torch.quantile(null["bsadf"], q, dim=0)                # [3, monitoring]
        if calibration == "wild_bootstrap":
            family = null["bsadf"][:, :horizon].max(dim=1).values
            family_cv = torch.quantile(family, q).tolist()
            record["family_wise_critical_values"] = dict(zip(("90%", "95%", "99%"), family_cv))
        record.update(replications=draws, seed=seed, successful_replications=draws,
                      failed_replications=0, monte_carlo_denominator=draws + 1,
                      failure_policy="abort; never remove failed draws from the denominator")
    # Date stamping.
    pick = QUANTILES.index(level)
    if calibration == "monte_carlo":
        stamp_cv = cv_sequence[pick]
        notes.append(f"Date stamps compare BSADF(e) with the {level:.0%} quantile of its "
                     "simulated null distribution at the same end date (PSY 2015).")
    elif calibration == "wild_bootstrap":
        stamp_cv = torch.full((monitoring,), family_cv[pick], dtype=torch.float64)
        notes.append(f"Date stamps compare BSADF with the {level:.0%} bootstrap quantile of its "
                     f"maximum over the first {horizon} monitoring points (Phillips and Shi "
                     "2020, family-wise size control).")
        notes.append("The composite bootstrap multiplies resampled residuals by N(0, 1) weights; "
                     "its heavy-tailed innovations make it conservative under iid errors in small "
                     "samples, and resampling does not reproduce a time pattern of the variance.")
    else:
        stamp_cv = None
    episode_rows = []
    if stamp_cv is not None:
        for number, (origin, end) in enumerate(episodes(bsadf, stamp_cv, min_duration), start=1):
            first = origin + window - 1
            stop = None if end is None else end + window - 1
            above = (bsadf > stamp_cv)[origin:(monitoring if end is None else end)]
            last_above = origin + int(torch.nonzero(above).max()) + window - 1
            segment = bsadf[origin:(monitoring if end is None else end + 1)]
            episode_rows.append([number, period_label(labels, first),
                                 None if stop is None else period_label(labels, stop),
                                 period_label(labels, last_above), first,
                                 stop, (total - 1 if stop is None else stop) - first,
                                 float(segment.max()), stop is None])
    tests = table([[name, value, p_values[i], *critical[i]] for i, (name, value)
                   in enumerate(zip(names, values))],
                  columns=["test", "statistic", "p_value", "critical_90pct", "critical_95pct",
                           "critical_99pct"])
    ends = torch.arange(window - 1, total)
    sequence_rows = []
    for j in range(monitoring):
        cvs = ([None, None, None] if cv_sequence is None
               else [float(cv_sequence[i, j]) for i in range(3)])
        statistic = float(bsadf[j])
        sequence_rows.append([period_label(labels, int(ends[j])), int(ends[j]),
                              statistic if math.isfinite(statistic) else None,
                              period_label(labels, int(observed["start"][0, j])), *cvs,
                              None if stamp_cv is None else statistic > float(stamp_cv[j])])
    tables = {
        "tests": tests,
        "sequence": table(sequence_rows, columns=["period", "index", "bsadf", "window_start",
                                                  "critical_90pct", "critical_95pct",
                                                  "critical_99pct", "exceeds"]),
        "episodes": table(episode_rows, columns=["episode", "start_period", "end_period",
                                                 "last_exceedance_period", "start_index",
                                                 "end_index", "duration", "peak_bsadf",
                                                 "ongoing"]),
    }
    notes.append("Episode end_period is PSY's termination date: the first observation, at least "
                 "min_duration after origination, whose BSADF falls below the critical value.")
    if int(observed["undefined"][0]):
        notes.append(f"{int(observed['undefined'][0])} window regression(s) were singular or exact "
                     "and are excluded from the maxima.")
    return TableSet(
        tables, title=f"Phillips-Shi-Yu explosive-root tests for {y}", test="psy", series=y,
        statistic=values[2], p_value=p_values[2], sadf=values[1], gsadf=values[2], adf=values[0],
        p_value_sadf=p_values[1],
        critical_values={"90%": critical[2][0], "95%": critical[2][1], "99%": critical[2][2]},
        critical_values_sadf={"90%": critical[1][0], "95%": critical[1][1], "99%": critical[1][2]},
        calibration=calibration, inference_status=(
            "statistics only" if calibration == "none" else
            "finite-sample Monte Carlo under the PSY null" if calibration == "monte_carlo" else
            "Phillips-Shi composite wild bootstrap"),
        lags=lags, min_window=window, r0=window / total, nobs=total, level=level,
        min_duration=float(min_duration), horizon=horizon if calibration == "wild_bootstrap" else None,
        episodes=len(episode_rows), undefined_windows=int(observed["undefined"][0]),
        null="Unit root (random walk with asymptotically negligible drift)",
        alternative="Mildly explosive episodes", tail="right",
        label="GSADF (H0: unit root; large values indicate explosive episodes)",
        sources=SOURCES, dtype="float64", device="cpu", estimated_work=work, **record,
        notes=notes)
