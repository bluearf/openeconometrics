"""Pure multiple-change regression: Gaussian calibration, robust tests and date intervals.

Bai & Perron (2003), sections 3.2/3.3 and 5.1/5.2, doi:10.1002/jae.659.
All coefficients may change; partial change is absent. With iid Gaussian
errors the Monte Carlo repeats the entire search. Heteroskedasticity- and
HAC-robust supF/UDmax/WDmax, the sequential supF(l+1|l) tests and the break-
date intervals live in ``bai_perron_robust`` and use Bai and Perron's
asymptotic tables (``break_tables``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot import bai_perron_robust as robust
from openecon.econometrics.unitroot.break_tables import BP_LEVELS, DMAX_BREAKS
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, check_flag, check_fraction, column_names, json_label, load_series,
)

_BATCH = 16
_MAX_BYTES = 256 * 1024**2
_MAX_WORK = 250_000_000
_MAX_ROWS = 1000


def _index_label(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_index_label(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return json_label(value)


@dataclass
class _Segments:
    q: Tensor
    starts: Tensor
    ends: Tensor
    inverse: Tensor
    h: int
    breaks: int
    excluded: int

    @property
    def n(self) -> int:
        return self.q.shape[0]


def _segments(q: Tensor, h: int, breaks: int) -> _Segments:
    n, p = q.shape
    pairs = torch.triu_indices(n + 1, n + 1, offset=h)
    starts, ends = pairs[0], pairs[1]
    cumulative = torch.cat((torch.zeros((1, p, p), dtype=torch.float64),
                            (q[:, :, None] * q[:, None, :]).cumsum(0)))
    gram = cumulative[ends] - cumulative[starts]
    # A rank decision depends only on the fixed design, never on the outcome.
    eigenvalues = torch.linalg.eigvalsh(gram)
    valid = eigenvalues[:, 0] > 1e-11 * eigenvalues[:, -1]
    excluded = int((~valid).sum())
    starts, ends, gram = starts[valid], ends[valid], gram[valid]
    return _Segments(q, starts, ends, torch.linalg.inv(gram), h, breaks, excluded)


def _search(design: _Segments, outcomes: Tensor, *, paths: bool = False):
    """SSR(s,t) from cumulative products; exact optimal partition by recurrence (4)."""
    n, p = design.q.shape
    b = outcomes.shape[1]
    xy = torch.cat((torch.zeros((1, p, b), dtype=torch.float64),
                    (design.q[:, :, None] * outcomes[:, None, :]).cumsum(0)))
    yy = torch.cat((torch.zeros((1, b), dtype=torch.float64), outcomes.square().cumsum(0)))
    cross = xy[design.ends] - xy[design.starts]
    ssr = yy[design.ends] - yy[design.starts] - torch.einsum(
        "spi,spq,sqi->si", cross, design.inverse, cross)
    tolerance = 1e-10 * yy[-1].clamp_min(1e-300)
    if bool((ssr < -tolerance).any()) or not bool(torch.isfinite(ssr).all()):
        raise AnalysisError("numerical_failure", "Segment SSR lost precision; rescale the inputs.")
    costs = torch.full((n + 1, n + 1, b), torch.inf, dtype=torch.float64)
    costs[design.starts, design.ends] = ssr.clamp_min(0)
    dp = costs[0].clone()
    objectives = [dp[n]]
    predecessors = []
    for segments in range(2, design.breaks + 2):
        current = torch.full_like(dp, torch.inf)
        previous = torch.full((n + 1, b), -1, dtype=torch.int64) if paths else None
        for end in range(segments * design.h, n + 1):
            begin = (segments - 1) * design.h
            candidates = dp[begin:end - design.h + 1] + costs[begin:end - design.h + 1, end]
            best, arg = candidates.min(dim=0)
            current[end] = best
            if paths:
                previous[end] = arg + begin
        dp = current
        objectives.append(dp[n])
        if paths:
            predecessors.append(previous)
    objectives = torch.stack(objectives)
    if not bool(torch.isfinite(objectives).all()):
        raise AnalysisError("singular_subsample", "No full-rank trimmed partition exists for every requested break count.")
    partitions = [[]]
    if paths:
        # The observed search also returns its segment costs for the sequential tests.
        for k in range(1, design.breaks + 1):
            end, points = n, []
            for j in range(k - 1, -1, -1):
                end = int(predecessors[j][end, 0])
                points.append(end)
            partitions.append(list(reversed(points)))
    return objectives, partitions, (costs[:, :, 0] if paths else None)


def _statistics(ssr: Tensor, n: int, p: int) -> Tensor:
    k = torch.arange(1, ssr.shape[0], dtype=torch.float64)[:, None]
    if bool((ssr[1:] <= 1e-20 * ssr[0]).any()):
        raise AnalysisError("perfect_fit", "An optimal partition fits exactly; supF is undefined.")
    return ((ssr[0] - ssr[1:]).clamp_min(0) / (k * p)
            / (ssr[1:] / (n - (k + 1) * p)))


def _asymptotic_tests(fstats: list[float], q: int, fraction: float, m: int) -> tuple[list, str]:
    """Long table rows (test, level, statistic, critical value) from Bai and Perron's tables."""
    rows = []
    dmax_ok = DMAX_BREAKS.get(next((e for e in DMAX_BREAKS if abs(e - fraction) < 1e-12), -1)) == m
    for level in BP_LEVELS:
        cvs = [robust.critical_value("supf", fraction, q, k, level) for k in range(m)]
        for k, (value, cv) in enumerate(zip(fstats, cvs), start=1):
            rows.append([f"supF({k})", level, value, cv])
        udmax = max(fstats)
        rows.append(["UDmax", level, udmax,
                     robust.critical_value("dmax", fraction, q, 0, level) if dmax_ok else None])
        if all(cv is not None for cv in cvs):
            wdmax = max(cvs[0] / cv * value for value, cv in zip(fstats, cvs))
        else:
            wdmax = None
        rows.append(["WDmax", level, wdmax,
                     robust.critical_value("dmax", fraction, q, 1, level)
                     if dmax_ok and wdmax is not None else None])
    if robust.critical_value("supf", fraction, q, 0, "5%") is None:
        status = ("unavailable: Bai-Perron tables cover trim in {0.05, 0.10, 0.15, 0.20, 0.25} "
                  "and at most 10 changing coefficients")
    elif not dmax_ok:
        status = ("asymptotic Bai-Perron tables; UDmax/WDmax critical values need max_breaks = "
                  f"{DMAX_BREAKS.get(next(e for e in DMAX_BREAKS if abs(e - fraction) < 1e-12))}")
    else:
        status = "asymptotic Bai-Perron (1998, 2003) tables"
    return rows, status


def bai_perron(
    data: Any, y: str, x: list[str], *, time: str | None = None,
    intercept: bool = True, max_breaks: int = 3, trim: float = 0.15,
    min_segment: int | None = None, replications: int = 499, seed: int = 0,
    errors: str = "iid_gaussian", prewhiten: bool = True, selection: str | None = None,
    level: float = 0.05,
) -> TableSet:
    """Globally optimal pure-change OLS with supF, UDmax/WDmax, sequential tests and date intervals.

    Parameters
    ----------
    data : DataFrame or column mapping
        Complete in-memory table; Dataset collection is unsupported.
    y : str
        Outcome column name.
    x : list[str]
        Fixed exogenous numeric regressors; x=[] tests mean shifts.
    time : str or None
        Consecutive integer periods or regular datetimes; rows are sorted.
    intercept : bool
        Allow a constant to change with the other coefficients.
    max_breaks : int
        Upper bound 1..5, specified before observing the outcome.
    trim : float
        Minimum segment fraction; h=ceil(trim*T), at least parameters+2.
    min_segment : int or None
        Optional larger integer minimum segment length.
    replications : int
        Gaussian null draw count, at least 19 (iid_gaussian only).
    seed : int
        Local CPU generator seed; global random state is preserved.
    errors : str
        ``"iid_gaussian"`` (default): common variance, Monte Carlo calibration
        conditional on X. ``"heteroskedastic"``: segment-wise heteroskedasticity-
        robust (n/(n-q) sum z z'u^2) covariance. ``"hac"``: Bai and Perron's
        (2003) segment-wise quadratic-spectral HAC with Andrews' AR(1)
        bandwidth. The robust F statistics are evaluated at the SSR-minimizing
        dates (Bai and Perron's GAUSS pftest); their inference uses the
        asymptotic tables of Bai and Perron (1998, 2003), which require trim in
        {0.05, 0.10, 0.15, 0.20, 0.25} and at most 10 changing coefficients.
    prewhiten : bool
        AR(1)/VAR(1) prewhitening of the HAC scores (hac only; default True).
    selection : ``"bic"``, ``"sequential"`` or None
        Number of breaks used for the coefficient, covariance and interval
        tables. None means BIC with iid_gaussian errors and the sequential
        supF(l+1|l) rule otherwise (BIC when its critical values are
        unavailable, with a note).
    level : float
        Significance level of the sequential rule and of the ``reject``
        columns: 0.10, 0.05 (default), 0.025 or 0.01.

    Returns
    -------
    TableSet : ``models`` (optimal SSR/BIC/first-row-after-break positions),
        ``tests`` (supF(k) and UDmax; Monte Carlo p-values and quantiles with
        iid_gaussian errors, Bai--Perron asymptotic critical values otherwise),
        ``asymptotic_tests`` (supF(k), UDmax and WDmax against the Bai--Perron
        tables at 10/5/2.5/1%), ``sequential`` (supF(l+1|l) for l = 0..m-1 with
        the l breaks of the global optimum), ``date_intervals`` (Bai 1997 /
        Bai--Perron 1998 90% and 95% intervals for the selected dates, allowing
        regime-specific regressor moments and error variances), ``segments``,
        ``coefficients``, ``covariance`` and ``sample``. Segment covariance and
        date intervals are conditional on the estimated dates; there is no
        selective inference or coefficient p-value.

    All admissible partitions are searched; singular candidates are excluded
    using the fixed design and recorded. The iid calibration repeats the same
    full search on independent standard normals: invariance to X*beta and
    sigma removes null nuisance parameters. p=(1+exceedances)/(B+1). Failures
    abort rather than dropping draws. Input, quadratic memory and draw work
    budgets are checked before constructing the segment arrays. Runtime is CPU
    float64.

    Example
    -------
    >>> result = oe.bai_perron(df, "consumption", ["income"], time="year", errors="hac")
    >>> result["sequential"], result["date_intervals"]
    """
    check_flag(intercept, "intercept")
    check_flag(prewhiten, "prewhiten")
    check_choice(errors, "errors", robust.ERRORS)
    if selection is not None:
        check_choice(selection, "selection", ("bic", "sequential"))
    if isinstance(level, bool) or level not in (0.10, 0.05, 0.025, 0.01):
        raise AnalysisError("invalid_option", "level must be 0.10, 0.05, 0.025 or 0.01.")
    level_name = {0.10: "10%", 0.05: "5%", 0.025: "2.5%", 0.01: "1%"}[level]
    names = column_names(x, "x")
    if not isinstance(y, str) or (not names and not intercept):
        raise AnalysisError("invalid_spec", "Specify one outcome and at least one regressor or intercept.")
    m = check_count(max_breaks, "max_breaks", minimum=1)
    b = check_count(replications, "replications", minimum=19)
    seed = check_count(seed, "seed")
    fraction = check_fraction(trim, "trim")
    if m > 5 or b > 9999 or seed >= 2**63:
        raise AnalysisError("invalid_option", "max_breaks<=5, replications<=9999 and seed<2**63 are required.")
    if min_segment is not None:
        min_segment = check_count(min_segment, "min_segment", minimum=1)
    gaussian = errors == "iid_gaussian"
    draws_used = b if gaussian else 0
    frame = _coerce_frame(data)
    n, p = len(frame), len(names) + int(intercept)
    if n > _MAX_ROWS or p > 8:
        raise AnalysisError("resource_limit", "Multiple-break search supports at most 1000 rows and 8 parameters; restrict the sample.")
    h = max(math.ceil(fraction * n), p + 2, min_segment or 0)
    if (m + 1) * h > n or n <= (m + 1) * p:
        raise AnalysisError("insufficient_observations", "Requested breaks and minimum segment length cannot fit this sample.")
    cells = (n + 1) * (n + 2) // 2
    estimated_bytes = 8 * (cells * (4 * p * p + p * _BATCH + 4)
                           + 2 * (n + 1)**2 * _BATCH)
    estimated_work = cells * (p**3 + draws_used * (p * p + m + 1))
    if estimated_bytes > _MAX_BYTES or estimated_work > _MAX_WORK:
        raise AnalysisError("resource_limit", "Multiple-break memory/draw work budget exceeded; reduce rows, regressors or replications.")
    values, labels = load_series(frame, [y, *names], time)
    # Positions, not index labels, identify rows; duplicate original indices are retained.
    positions = (frame[time].reset_index(drop=True).sort_values(kind="stable").index.tolist()
                 if time is not None else list(range(n)))
    if time is not None and pd.api.types.is_datetime64_any_dtype(labels.dtype):
        if n >= 3 and pd.infer_freq(pd.DatetimeIndex(labels)) is None:
            raise AnalysisError("time_gaps", "Time dates must have a regular frequency; restrict or fill gaps explicitly.")
    predictors = values[:, 1:]
    centers = predictors.mean(0) if intercept else torch.zeros(len(names), dtype=torch.float64)
    scales = (predictors - centers).abs().amax(0).clamp_min(1e-300)
    z = (predictors - centers) / scales
    if intercept:
        z = torch.cat((torch.ones((n, 1), dtype=torch.float64), z), dim=1)
    singular = torch.linalg.svdvals(z)
    if float(singular[-1]) <= 1e-11 * float(singular[0]):
        raise AnalysisError("collinear_regressors", "All changing regressors must be full rank; remove duplicates or constants.")
    q, r = torch.linalg.qr(z, mode="reduced")
    ycenter = values[:, 0].mean() if intercept else torch.tensor(0.0, dtype=torch.float64)
    target = values[:, 0] - ycenter
    yscale = target.abs().max()
    if float(yscale) == 0:
        raise AnalysisError("perfect_fit", "The outcome has no unexplained variation.")
    target = target / yscale
    design = _segments(q, h, m)
    ssr, partitions, costs = _search(design, target[:, None], paths=True)
    if float(ssr[0, 0]) <= 1e-20 * float(target.square().sum()):
        raise AnalysisError("perfect_fit", "The null regression fits exactly; supF is undefined.")
    # Original-unit design for the robust statistics (the HAC bandwidth is not
    # invariant to rescaling individual regressors, so Bai and Perron's raw units are kept).
    raw = torch.cat(([torch.ones((n, 1), dtype=torch.float64)] if intercept else []) + [predictors], dim=1)
    outcome = values[:, 0]
    if gaussian:
        observed = _statistics(ssr, n, p)[:, 0]
    else:
        observed = torch.tensor([robust.wald_f(raw, outcome, [0, *partitions[k], n], errors, prewhiten)
                                 for k in range(1, m + 1)], dtype=torch.float64)
    observed = torch.cat((observed, observed.max()[None]))
    fstats = observed[:-1].tolist()
    test_names = [f"supF({k})" for k in range(1, m + 1)] + ["UDmax"]
    asymptotic_rows, asymptotic_status = _asymptotic_tests(fstats, p, fraction, m)
    if gaussian:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        simulated = []
        for offset in range(0, b, _BATCH):
            count = min(_BATCH, b - offset)
            # Draw-major order and a fixed batch size are part of seeded replay.
            draws = torch.randn((count, n), dtype=torch.float64, generator=generator).T.contiguous()
            null_ssr, _, _ = _search(design, draws)
            stats = _statistics(null_ssr, n, p)
            simulated.append(torch.cat((stats, stats.max(0).values[None, :]), dim=0))
        draws = torch.cat(simulated, dim=1)
        exceedances = (draws >= observed[:, None]).sum(1)
        pvalues = (exceedances + 1).to(torch.float64) / (b + 1)
        critical = torch.quantile(draws, torch.tensor([.90, .95, .99], dtype=torch.float64), dim=1)
        tests = [[name, float(observed[i]), float(pvalues[i]), int(exceedances[i]),
                  *[float(critical[j, i]) for j in range(3)]] for i, name in enumerate(test_names)]
    else:
        lookup = {(row[0], row[1]): row[3] for row in asymptotic_rows}
        tests = [[name, float(observed[i]), None, None,
                  *[lookup[(name, lev)] for lev in ("10%", "5%", "1%")]]
                 for i, name in enumerate(test_names)]
    # Sequential supF(l+1|l) with the l breaks of the global optimum.
    sequence = robust.sequential_tests(raw, outcome, costs, partitions, h, errors, prewhiten,
                                       fstats[0])
    sequential_rows, sequential_breaks = [], None
    cv_available = robust.critical_value("sequential", fraction, p, 0, level_name) is not None
    for row in sequence:
        cvs = [robust.critical_value("sequential", fraction, p, row["l"], lev) for lev in BP_LEVELS]
        chosen = cvs[BP_LEVELS.index(level_name)]
        reject = None if chosen is None else row["statistic"] > chosen
        if sequential_breaks is None and reject is False:
            sequential_breaks = row["l"]
        new = row["new_break"]
        sequential_rows.append([f"supF({row['l'] + 1}|{row['l']})", row["l"], row["statistic"],
                                new, None if new is None else json_label(labels.iloc[new]),
                                *cvs, reject])
    if cv_available and sequential_breaks is None:
        sequential_breaks = m
    bic = [n * (math.log(float(ssr[k, 0]) / n) + 2 * math.log(float(yscale)))
           + ((k + 1) * p + k) * math.log(n)
           for k in range(m + 1)]
    bic_breaks = min(range(m + 1), key=lambda k: (bic[k], k))
    notes = []
    method = selection or ("bic" if gaussian else "sequential")
    if method == "sequential" and not cv_available:
        if selection == "sequential":
            raise AnalysisError("critical_values_unavailable", "The sequential rule needs Bai-Perron "
                                "critical values: use trim in {0.05, 0.10, 0.15, 0.20, 0.25} and at "
                                "most 10 changing coefficients, or selection='bic'.")
        method = "bic"
        notes.append("Sequential critical values are unavailable for this trim/q; BIC selects the break count.")
    selected = bic_breaks if method == "bic" else sequential_breaks
    terms = (["Intercept"] if intercept else []) + names
    bounds = [0, *partitions[selected], n]
    segment_rows, coefficient_rows = [], []
    covariances = []
    if gaussian:
        rinv = torch.linalg.solve_triangular(r, torch.eye(p, dtype=torch.float64), upper=True)
        transform = torch.diag(torch.cat((torch.ones(1, dtype=torch.float64) if intercept
                                         else torch.empty(0, dtype=torch.float64), 1 / scales)))
        if intercept and names:
            transform[0, 1:] = -centers / scales
        sigma2 = float(ssr[selected, 0]) / (n - (selected + 1) * p)
    else:
        betas, resid = robust.segment_fits(raw, outcome, bounds)
        robust_covariances = robust.segment_covariances(raw, resid, bounds, errors, prewhiten)
    for regime, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        if gaussian:
            gram = q[start:end].T @ q[start:end]
            beta_q = torch.linalg.solve(gram, q[start:end].T @ target[start:end])
            beta = yscale * transform @ rinv @ beta_q
            if intercept:
                beta[0] += ycenter
            cov = float(yscale)**2 * sigma2 * transform @ rinv @ torch.linalg.inv(gram) @ rinv.T @ transform.T
        else:
            beta, cov = betas[regime - 1], robust_covariances[regime - 1]
        covariances.append(cov)
        segment_rows.append([regime, start, end, end - start, json_label(labels.iloc[start]),
                             json_label(labels.iloc[end - 1])])
        coefficient_rows.extend([regime, term, float(value)] for term, value in zip(terms, beta))
    cov_labels = [f"regime{j}:{term}" for j in range(1, selected + 2) for term in terms]
    covariance = torch.block_diag(*covariances)
    interval_rows = []
    if selected:
        def period(index: int | None) -> Any:
            # Bounds beyond the sample are labelled by the first or last period.
            return None if index is None else json_label(labels.iloc[min(max(index, 0), n - 1)])
        for row in robust.date_intervals(raw, outcome, bounds, errors, prewhiten):
            interval_rows.append([row["break"], row["index"], period(row["index"]),
                                  row["lower_95"], row["upper_95"], row["lower_90"], row["upper_90"],
                                  period(row["lower_95"]), period(row["upper_95"]),
                                  period(row["lower_90"]), period(row["upper_90"]),
                                  row["lower_95_raw"], row["upper_95_raw"], row["lower_90_raw"],
                                  row["upper_90_raw"], row["status"]])
    if gaussian:
        calibration_attrs = dict(
            seed=seed, replications=b, successful_replications=b, failed_replications=0,
            failure_policy="abort; never remove failed draws from the denominator",
            monte_carlo_denominator=b + 1, p_value=float(pvalues[-1]),
            calibration="finite-sample Gaussian iid fixed-design Monte Carlo; full search on every draw",
            critical_method="empirical quantiles of the recorded null simulation")
        notes = ["Requires fixed exogenous regressors, independent Gaussian errors and common error variance.",
                 *notes]
    else:
        calibration_attrs = dict(
            seed=None, replications=0, p_value=None,
            calibration=asymptotic_status,
            critical_method="Bai-Perron asymptotic tables divided by q (F units)")
        notes = [f"{errors} covariance; F statistics at the SSR-minimizing dates (Bai-Perron pftest); "
                 "no p-value is tabulated.", *notes]
    if h > math.ceil(fraction * n):
        notes.append(f"The minimum segment h = {h} exceeds trim*T; the tabulated critical values for "
                     "trim are then conservative.")
    notes.append("The calibrated tests address no break; the selected break count and dates are "
                 "estimates, and date intervals and covariance are conditional on them.")
    notes.append("Partial-change models are unsupported.")
    result = TableSet({
        "models": table([[k, float(ssr[k, 0]) * float(yscale)**2, bic[k], partitions[k],
                          [json_label(labels.iloc[j]) for j in partitions[k]], k == selected]
                         for k in range(m + 1)],
                        columns=["breaks", "ssr", "bic", "break_indices", "break_periods", "selected"]),
        "tests": table(tests, columns=["test", "statistic", "p_value", "exceedances",
                                        "critical_10pct", "critical_5pct", "critical_1pct"]),
        "asymptotic_tests": table(
            [[*row, None if row[2] is None or row[3] is None else row[2] > row[3]]
             for row in asymptotic_rows],
            columns=["test", "level", "statistic", "critical_value", "reject"],
            status=asymptotic_status),
        "sequential": table(sequential_rows,
                            columns=["test", "l", "statistic", "new_break_index", "new_break_period",
                                     "critical_10pct", "critical_5pct", "critical_2_5pct",
                                     "critical_1pct", "reject"]),
        "date_intervals": table(interval_rows, columns=[
            "break", "break_index", "break_period", "lower_95", "upper_95", "lower_90", "upper_90",
            "lower_95_period", "upper_95_period", "lower_90_period", "upper_90_period",
            "lower_95_unrounded", "upper_95_unrounded", "lower_90_unrounded", "upper_90_unrounded",
            "status"]),
        "segments": table(segment_rows, columns=["regime", "start", "end_exclusive", "nobs", "first_period", "last_period"]),
        "coefficients": table(coefficient_rows, columns=["regime", "term", "estimate"]),
        "covariance": table(covariance.numpy(), columns=cov_labels, index=cov_labels,
                            inference_scope="conditional on estimated dates; not selective inference"),
        "sample": table([[pos, _index_label(frame.index[pos]), json_label(labels.iloc[i])]
                         for i, pos in enumerate(positions)], columns=["original_position", "original_index", "period"]),
    }, title="Bai–Perron pure multiple-change OLS", procedure="bai_perron", nobs=n,
        terms=terms, selected_breaks=selected, max_breaks=m, min_segment=h, trim=fraction,
        break_index_convention="0-based first row of the following regime", errors=errors,
        prewhiten=prewhiten if errors == "hac" else None,
        null="All regression coefficients are constant", alternative="1..max_breaks common coefficient break dates",
        statistic=float(observed[-1]), **calibration_attrs,
        selection_method=method, bic_breaks=bic_breaks, sequential_breaks=sequential_breaks,
        level=level, asymptotic_inference=asymptotic_status,
        selection=("BIC: T*log(SSR/T)+((k+1)*q+k)*log(T); includes estimated dates" if method == "bic"
                   else f"sequential supF(l+1|l) at {level_name} with the global-optimum dates"),
        date_interval_method=("Bai (1997) / Bai-Perron (1998) shrinking-shift intervals, regime-specific "
                              "regressor moments and error (long-run) variances; GAUSS int() rounding"),
        coefficient_inference="descriptive estimates and covariance conditional on estimated dates; no selective CI/p-values",
        excluded_rank_deficient_segments=design.excluded, dtype="float64", device="cpu",
        estimated_workspace_bytes=estimated_bytes, estimated_work=estimated_work,
        data_hash=_frame_hasher(frame.iloc[positions][[y, *names, *([time] if time is not None else [])]]).hexdigest(),
        sources=["https://doi.org/10.1002/jae.659", "https://doi.org/10.2307/2998540",
                 "https://doi.org/10.1111/1368-423x.00102", "https://doi.org/10.1162/003465397557132",
                 "https://blogs.bu.edu/perron/files/2020/05/m-Break.zip"],
        notes=notes)
    # The console persists table cells, while pandas attrs are a source-side
    # contract. Keep the complete scientific settings in an ordinary table too.
    result["settings"] = table([[key, json.dumps(value, ensure_ascii=False, sort_keys=True)]
                                for key, value in result.attrs.items()], columns=["setting", "value_json"])
    return result
