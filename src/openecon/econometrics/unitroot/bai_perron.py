"""Pure multiple-change regression with conditional-design Gaussian calibration.

Bai & Perron (2003), sections 3.2/3.3 and 5.1/5.2, doi:10.1002/jae.659.
All coefficients may change. Partial-change, HAC and break-date confidence
intervals are deliberately absent. Monte Carlo repeats the entire search.
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
        for k in range(1, design.breaks + 1):
            end, points = n, []
            for j in range(k - 1, -1, -1):
                end = int(predecessors[j][end, 0])
                points.append(end)
            partitions.append(list(reversed(points)))
    return objectives, partitions


def _statistics(ssr: Tensor, n: int, p: int) -> Tensor:
    k = torch.arange(1, ssr.shape[0], dtype=torch.float64)[:, None]
    if bool((ssr[1:] <= 1e-20 * ssr[0]).any()):
        raise AnalysisError("perfect_fit", "An optimal partition fits exactly; supF is undefined.")
    return ((ssr[0] - ssr[1:]).clamp_min(0) / (k * p)
            / (ssr[1:] / (n - (k + 1) * p)))


def bai_perron(
    data: Any, y: str, x: list[str], *, time: str | None = None,
    intercept: bool = True, max_breaks: int = 3, trim: float = 0.15,
    min_segment: int | None = None, replications: int = 499, seed: int = 0,
    errors: str = "iid_gaussian",
) -> TableSet:
    """Globally optimal pure-change OLS, supF(k), UDmax and BIC break selection.

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
        Gaussian null draw count, at least 19.
    seed : int
        Local CPU generator seed; global random state is preserved.
    errors : str
        Only iid_gaussian, common error variance independent of fixed X.

    Returns
    -------
    TableSet : models (optimal SSR/BIC/first-row-after-break positions), tests
        (supF for each k and UDmax with Monte Carlo p-values), segments,
        coefficients, covariance and sample. Segment covariance is descriptive
        conditional on estimated dates, not selective inference. No date CI,
        coefficient p-value, HAC, partial-change or sequential-test claim.

    All admissible partitions are searched; singular candidates are excluded
    using the fixed design and recorded. Calibration repeats the same full
    search on independent standard normals: invariance to X*beta and sigma
    removes null nuisance parameters. p=(1+exceedances)/(B+1). Failures abort
    rather than dropping draws. Input, quadratic memory and draw work budgets
    are checked before constructing the segment arrays. Runtime is CPU float64.

    Example
    -------
    >>> result = oe.bai_perron(df, "consumption", ["income"], time="year")
    >>> result["tests"], result.attrs["selected_breaks"]
    """
    check_flag(intercept, "intercept")
    check_choice(errors, "errors", ("iid_gaussian",))
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
    estimated_work = cells * (p**3 + b * (p * p + m + 1))
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
    ssr, partitions = _search(design, target[:, None], paths=True)
    if float(ssr[0, 0]) <= 1e-20 * float(target.square().sum()):
        raise AnalysisError("perfect_fit", "The null regression fits exactly; supF is undefined.")
    observed = _statistics(ssr, n, p)[:, 0]
    observed = torch.cat((observed, observed.max()[None]))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    simulated = []
    for offset in range(0, b, _BATCH):
        count = min(_BATCH, b - offset)
        # Draw-major order and a fixed batch size are part of seeded replay.
        draws = torch.randn((count, n), dtype=torch.float64, generator=generator).T.contiguous()
        null_ssr, _ = _search(design, draws)
        stats = _statistics(null_ssr, n, p)
        simulated.append(torch.cat((stats, stats.max(0).values[None, :]), dim=0))
    draws = torch.cat(simulated, dim=1)
    exceedances = (draws >= observed[:, None]).sum(1)
    pvalues = (exceedances + 1).to(torch.float64) / (b + 1)
    critical = torch.quantile(draws, torch.tensor([.90, .95, .99], dtype=torch.float64), dim=1)
    bic = [n * (math.log(float(ssr[k, 0]) / n) + 2 * math.log(float(yscale)))
           + ((k + 1) * p + k) * math.log(n)
           for k in range(m + 1)]
    selected = min(range(m + 1), key=lambda k: (bic[k], k))
    terms = (["Intercept"] if intercept else []) + names
    bounds = [0, *partitions[selected], n]
    segment_rows, coefficient_rows = [], []
    covariances = []
    rinv = torch.linalg.solve_triangular(r, torch.eye(p, dtype=torch.float64), upper=True)
    transform = torch.diag(torch.cat((torch.ones(1, dtype=torch.float64) if intercept
                                     else torch.empty(0, dtype=torch.float64), 1 / scales)))
    if intercept and names:
        transform[0, 1:] = -centers / scales
    sigma2 = float(ssr[selected, 0]) / (n - (selected + 1) * p)
    for regime, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        gram = q[start:end].T @ q[start:end]
        beta_q = torch.linalg.solve(gram, q[start:end].T @ target[start:end])
        beta = yscale * transform @ rinv @ beta_q
        if intercept:
            beta[0] += ycenter
        cov = float(yscale)**2 * sigma2 * transform @ rinv @ torch.linalg.inv(gram) @ rinv.T @ transform.T
        covariances.append(cov)
        segment_rows.append([regime, start, end, end - start, json_label(labels.iloc[start]),
                             json_label(labels.iloc[end - 1])])
        coefficient_rows.extend([regime, term, float(value)] for term, value in zip(terms, beta))
    cov_labels = [f"regime{j}:{term}" for j in range(1, selected + 2) for term in terms]
    covariance = torch.block_diag(*covariances)
    test_names = [f"supF({k})" for k in range(1, m + 1)] + ["UDmax"]
    tests = [[name, float(observed[i]), float(pvalues[i]), int(exceedances[i]),
              *[float(critical[j, i]) for j in range(3)]] for i, name in enumerate(test_names)]
    result = TableSet({
        "models": table([[k, float(ssr[k, 0]) * float(yscale)**2, bic[k], partitions[k],
                          [json_label(labels.iloc[j]) for j in partitions[k]], k == selected]
                         for k in range(m + 1)],
                        columns=["breaks", "ssr", "bic", "break_indices", "break_periods", "selected"]),
        "tests": table(tests, columns=["test", "statistic", "p_value", "exceedances",
                                        "critical_10pct", "critical_5pct", "critical_1pct"]),
        "segments": table(segment_rows, columns=["regime", "start", "end_exclusive", "nobs", "first_period", "last_period"]),
        "coefficients": table(coefficient_rows, columns=["regime", "term", "estimate"]),
        "covariance": table(covariance.numpy(), columns=cov_labels, index=cov_labels,
                            inference_scope="conditional on estimated dates; not selective inference"),
        "sample": table([[pos, _index_label(frame.index[pos]), json_label(labels.iloc[i])]
                         for i, pos in enumerate(positions)], columns=["original_position", "original_index", "period"]),
    }, title="Bai–Perron pure multiple-change OLS", procedure="bai_perron", nobs=n,
        terms=terms, selected_breaks=selected, max_breaks=m, min_segment=h, trim=fraction,
        break_index_convention="0-based first row of the following regime", errors=errors,
        null="All regression coefficients are constant", alternative="1..max_breaks common coefficient break dates",
        seed=seed, replications=b, successful_replications=b, failed_replications=0,
        failure_policy="abort; never remove failed draws from the denominator",
        monte_carlo_denominator=b + 1, p_value=float(pvalues[-1]), statistic=float(observed[-1]),
        calibration="finite-sample Gaussian iid fixed-design Monte Carlo; full search on every draw",
        critical_method="empirical quantiles of the recorded null simulation",
        selection="BIC: T*log(SSR/T)+((k+1)*q+k)*log(T); includes estimated dates",
        coefficient_inference="descriptive estimates and covariance conditional on estimated dates; no selective CI/p-values",
        excluded_rank_deficient_segments=design.excluded, dtype="float64", device="cpu",
        estimated_workspace_bytes=estimated_bytes, estimated_work=estimated_work,
        data_hash=_frame_hasher(frame.iloc[positions][[y, *names, *([time] if time is not None else [])]]).hexdigest(),
        sources=["https://doi.org/10.1002/jae.659"],
        notes=["Requires fixed exogenous regressors, independent Gaussian errors and common error variance.",
               "BIC selects the number of breaks; calibrated tests address no break, not the selected break count.",
               "HAC, partial-change, sequential tests and break-date confidence intervals are unsupported."])
    # The console persists table cells, while pandas attrs are a source-side
    # contract. Keep the complete scientific settings in an ordinary table too.
    result["settings"] = table([[key, json.dumps(value, ensure_ascii=False, sort_keys=True)]
                                for key, value in result.attrs.items()], columns=["setting", "value_json"])
    return result
