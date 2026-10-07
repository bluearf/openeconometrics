"""Threshold regression (Stata's ``threshold``; EViews: threshold regression, TAR; Hansen 2000).

    y_t = w_t'a + sum_(r=1..m+1) 1{gamma_(r-1) < q_t <= gamma_r} x_t'b_r + e_t,
    gamma_0 = -inf < gamma_1 < ... < gamma_m < gamma_(m+1) = +inf,

with x the region-specific terms (``regions``; by default every term including the
constant) and w the terms common to all regions. For given thresholds the model is
OLS; the thresholds minimize the sum of squared residuals over the observed values of
the threshold variable q, keeping at least ``trim`` * N observations in every region.

Grid search without refitting: the data are sorted by q once; for a candidate split at
sorted position i the regression's cross-product matrix consists of prefix sums of
x x', w x' and x y (formed chunk by chunk, so memory stays bounded) and the SSR is
y'y - c'G^-1 c, solved by batched Cholesky for all candidates of a chunk. The few best
candidates are then refitted by QR (``engines.linalg.least_squares``) and the smallest
exact SSR wins, so the normal equations only screen. Several thresholds are found
sequentially (each new threshold given the previous ones), followed by Bai's (1997)
refinement: every threshold is re-estimated given the others until none changes.

Inference treats the thresholds as known (their estimation error does not affect the
slope estimates asymptotically, Hansen 2000): OLS covariance with N - K degrees of
freedom (``nonrobust``), or White HC1/HC2/HC3. For a single threshold the Hansen (2000)
likelihood-ratio confidence set {gamma: LR(gamma) <= c}, LR = N (S(gamma) - S(gamma^)) /
S(gamma^), c = -2 ln(1 - sqrt(1 - level)) (homoskedastic asymptotic distribution) is
reported in ``extra``. With ``bootstrap`` > 0 the test of no threshold effect uses
Hansen's (1996, 2000) fixed-regressor bootstrap of F = N (S_0 - S_1) / S_1, S_0 the
linear model's SSR and S_1 that of the best one-threshold model (also when
``nthresholds`` > 1): y*_t = e^_t u_t with u_t ~ N(0, 1) and e^ the residuals of that
one-threshold model, the same candidate grid in every replication.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, wald_test,
)
from openecon.econometrics.tsmodels.common import (
    as_int, build_spec, ordered_frame, perfect_fit, require_variation,
)
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_BUDGET = 4_000_000          # floats per chunk of candidate cross-product matrices
_REFIT = 5                   # best screened candidates refitted by QR
_PATH = 200                  # points of the SSR path kept in extra


def _check_lr_precision(best_ssr: float, total_squared_outcome: float) -> None:
    # The screening SSR is y'y - y'X(X'X)^-1X'y. A tiny cancellation
    # remainder cannot certify Hansen's LR ratio even when a final QR fit has
    # positive residual variance; do not fabricate a confidence hull from it.
    if (
        not math.isfinite(best_ssr)
        or best_ssr <= 32.0 * torch.finfo(torch.float64).eps * total_squared_outcome
    ):
        raise AnalysisError(
            "threshold_precision",
            "The candidate SSR is too small relative to the outcome for a reliable threshold confidence set in float64. Coefficient residuals may be positive, but the screened LR ratio is not representable accurately.",
        )


class _Grid:
    """Sorted data and the SSR of every admissible split given fixed splits."""

    def __init__(self, y: Tensor, xv: Tensor, xw: Tensor, q: Tensor, minimum: int):
        order = torch.argsort(q, stable=True)
        self.order = order
        self.q = q[order]
        self.scale_y = float(y.square().mean().sqrt()) or 1.0
        self.y = y[order] / self.scale_y
        scale_v = xv.square().mean(0).sqrt().clamp_min(1e-300)
        scale_w = xw.square().mean(0).sqrt().clamp_min(1e-300)
        self.xv, self.xw = xv[order] / scale_v, xw[order] / scale_w
        self.n, self.kv, self.kw = y.shape[0], xv.shape[1], xw.shape[1]
        self.minimum = minimum
        self.yy = float(self.y.square().sum())
        self.wy = self.xw.T @ self.y
        self.ww = self.xw.T @ self.xw
        distinct = torch.zeros(self.n + 1, dtype=torch.bool)
        distinct[1:self.n] = self.q[1:] > self.q[:-1]
        self.distinct = distinct                      # split position i separates i-1 and i

    def _block(self, a: int, b: int) -> tuple[Tensor, Tensor, Tensor]:
        xv, xw, y = self.xv[a:b], self.xw[a:b], self.y[a:b]
        return xv.T @ xv, xw.T @ xv, xv.T @ y

    def ssr(self, splits: list[int], start: int, stop: int) -> tuple[Tensor, Tensor]:
        """Screened SSR (scaled units) for every new split position in [start, stop).

        ``splits`` are the fixed split positions; candidates closer than ``minimum`` to any
        split or to the ends are inadmissible (SSR = inf). Returns (positions, ssr).
        """
        bounds = sorted({0, self.n, *splits})
        positions = torch.arange(start, stop)
        admissible = self.distinct[positions].clone()
        for edge in bounds:
            admissible &= (positions - edge).abs() >= self.minimum
        result = torch.full((positions.shape[0],), math.inf, dtype=torch.float64)
        blocks = [self._block(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        k = self.kw + len(bounds) * self.kv              # one more region than now
        chunk = max(1, _BUDGET // max(k * k, 1))
        for r, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            inside = (positions > a) & (positions < b) & admissible
            if not bool(inside.any()):
                continue
            cand = positions[inside]
            values = torch.full((cand.shape[0],), math.inf, dtype=torch.float64)
            cursor = int(cand[0])
            running_vv, running_wv, running_vy = self._block(a, cursor)
            total_vv, total_wv, total_vy = blocks[r]
            for c0 in range(0, cand.shape[0], chunk):
                piece = cand[c0:c0 + chunk]
                top = int(piece[-1])
                rows = slice(cursor, top)
                xv, xw, y = self.xv[rows], self.xw[rows], self.y[rows]
                cum_vv = torch.cumsum(xv[:, :, None] * xv[:, None, :], 0)
                cum_wv = torch.cumsum(xw[:, :, None] * xv[:, None, :], 0)
                cum_vy = torch.cumsum(xv * y[:, None], 0)
                zero_vv = torch.zeros((1, self.kv, self.kv), dtype=torch.float64)
                zero_wv = torch.zeros((1, self.kw, self.kv), dtype=torch.float64)
                zero_vy = torch.zeros((1, self.kv), dtype=torch.float64)
                cum_vv = torch.cat([zero_vv, cum_vv]) + running_vv
                cum_wv = torch.cat([zero_wv, cum_wv]) + running_wv
                cum_vy = torch.cat([zero_vy, cum_vy]) + running_vy
                index = piece - cursor                        # prefix sums up to each split
                left_vv, left_wv, left_vy = cum_vv[index], cum_wv[index], cum_vy[index]
                running_vv, running_wv, running_vy = cum_vv[-1], cum_wv[-1], cum_vy[-1]
                cursor = top
                values[c0:c0 + piece.shape[0]] = self._solve(
                    blocks, r, left_vv, left_wv, left_vy, total_vv, total_wv, total_vy)
            result[inside] = values
        return positions, result

    def _solve(self, blocks, r, left_vv, left_wv, left_vy, total_vv, total_wv, total_vy) -> Tensor:
        batch = left_vv.shape[0]
        parts_vv, parts_wv, parts_vy = [], [], []
        for index, (vv, wv, vy) in enumerate(blocks):
            if index == r:
                parts_vv += [left_vv, total_vv - left_vv]
                parts_wv += [left_wv, total_wv - left_wv]
                parts_vy += [left_vy, total_vy - left_vy]
            else:
                parts_vv.append(vv.expand(batch, -1, -1))
                parts_wv.append(wv.expand(batch, -1, -1))
                parts_vy.append(vy.expand(batch, -1))
        kv, kw = self.kv, self.kw
        regions = len(parts_vv)
        k = kw + regions * kv
        gram = torch.zeros((batch, k, k), dtype=torch.float64)
        rhs = torch.zeros((batch, k), dtype=torch.float64)
        gram[:, :kw, :kw] = self.ww
        rhs[:, :kw] = self.wy
        for j in range(regions):
            s = slice(kw + j * kv, kw + (j + 1) * kv)
            gram[:, s, s] = parts_vv[j]
            gram[:, :kw, s] = parts_wv[j]
            gram[:, s, :kw] = parts_wv[j].transpose(1, 2)
            rhs[:, s] = parts_vy[j]
        chol, info = torch.linalg.cholesky_ex(gram)
        fitted = (rhs * torch.cholesky_solve(rhs[..., None], chol).squeeze(-1)).sum(1)
        ssr = self.yy - fitted
        ssr[info != 0] = math.inf
        return ssr.clamp_min(0.0)

    def design(self, splits: list[int]) -> Tensor:
        """Explicit sorted design for the given splits (common block, then regions)."""
        bounds = sorted({0, self.n, *splits})
        columns = [self.xw]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            block = torch.zeros_like(self.xv)
            block[a:b] = self.xv[a:b]
            columns.append(block)
        return torch.cat(columns, dim=1)

    def exact(self, splits: list[int]) -> float:
        fit = kernel_call(least_squares, self.design(splits), self.y)
        return float(fit.ssr)

    def best(self, fixed: list[int]) -> tuple[int, float, Tensor, Tensor]:
        positions, ssr = self.ssr(fixed, 1, self.n)
        finite = torch.isfinite(ssr)
        if not bool(finite.any()):
            raise AnalysisError("no_admissible_threshold", "No admissible threshold remains: every "
                                "candidate leaves a region with fewer than the minimum number of "
                                "observations. Lower trim or nthresholds, or use more data.")
        top = torch.argsort(torch.where(finite, ssr, torch.full_like(ssr, math.inf)))[:_REFIT]
        top = [int(i) for i in top if bool(finite[i])]
        exact = [(self.exact(fixed + [int(positions[i])]), int(positions[i])) for i in top]
        value, position = min(exact)
        return position, value, positions[finite], ssr[finite]


def fit_threshold(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``threshold`` estimator (see the module docstring and ``threshold``)."""
    q_name = spec.columns["threshold_var"]
    q_name = q_name if isinstance(q_name, str) else q_name[0]
    if spec.time is not None:
        frame, _ = ordered_frame(spec, data, what="Threshold regression", require_regular=False)
    else:
        frame = ModelFrame(spec, data)
    design = frame.drop_collinear(frame.design())
    regions_option = frame.option("regions")
    varying_terms = design.terms if regions_option is None else list(regions_option)
    unknown = [term for term in varying_terms if term not in design.terms]
    if unknown:
        raise AnalysisError("invalid_option", f"regions names terms that are not in the model: "
                            f"{', '.join(unknown)} (terms: {', '.join(design.terms)}).")
    if not varying_terms:
        raise AnalysisError("invalid_option", "regions must name at least one term.")
    varying = [i for i, term in enumerate(design.terms) if term in varying_terms]
    common = [i for i, term in enumerate(design.terms) if term not in varying_terms]
    y, q = frame.numeric(spec.outcome), frame.numeric(q_name)
    require_variation(y, spec.outcome)
    if q.numel() > 1 and bool((q == q[0]).all()):
        raise AnalysisError("constant_threshold_variable", f"The threshold variable '{q_name}' is "
                            "constant, so it cannot split the sample into regions.")
    n = frame.n
    trim = float(frame.option("trim"))
    if not 0.0 < trim < 0.5:
        raise AnalysisError("invalid_option", "trim must lie strictly between 0 and 0.5.")
    m = int(frame.option("nthresholds"))
    kv, kw = len(varying), len(common)
    minimum = max(math.ceil(trim * n), kv + 1)
    if (m + 1) * minimum > n:
        raise AnalysisError("insufficient_observations", f"{n} observations cannot hold {m + 1} "
                            f"regions of at least {minimum} observations each; lower trim or "
                            "nthresholds.")
    grid = _Grid(y, design.x[:, varying], design.x[:, common], q, minimum)
    splits: list[int] = []
    sequence = []
    null_ssr = grid.exact([])
    path_positions = path_ssr = None
    for step in range(m):
        position, value, positions, ssr = grid.best(splits)
        if step == 0:
            path_positions, path_ssr = positions, ssr
            single = [position]                 # the one-threshold model of Hansen's test
        splits = sorted(splits + [position])
        for _ in range(10 if step else 0):                     # Bai (1997) refinement
            changed = False
            for index in range(len(splits)):
                others = splits[:index] + splits[index + 1:]
                new, value, _, _ = grid.best(others)
                if new != splits[index]:
                    splits = sorted(others + [new])
                    changed = True
            if not changed:
                break
        sequence.append({"thresholds": [float(grid.q[i - 1]) for i in splits],
                         "ssr": grid.exact(splits) * grid.scale_y ** 2})
    thresholds = [float(grid.q[i - 1]) for i in splits]
    region = torch.zeros(n, dtype=torch.int64)
    for gamma in thresholds:
        region += (q > gamma).to(torch.int64)
    columns, terms, equations = [], [], []
    for i in common:
        columns.append(design.x[:, i])
        terms.append(design.terms[i])
        equations.append(None)
    for r in range(m + 1):
        indicator = (region == r).to(torch.float64)
        for i in varying:
            columns.append(design.x[:, i] * indicator)
            terms.append(f"region{r + 1}:{design.terms[i]}")
            equations.append(f"region{r + 1}")
    x = torch.stack(columns, dim=1)
    fit = kernel_call(least_squares, x, y, drop_collinear=False)
    k = x.shape[1]
    df_resid = n - k
    if df_resid < 1:
        raise AnalysisError("insufficient_observations", "Too few observations for the regions.")
    ssr = float(fit.ssr)
    if perfect_fit(ssr, y) or perfect_fit(null_ssr * grid.scale_y ** 2, y):
        raise AnalysisError("perfect_fit", "The regression fits the outcome exactly; the sum of "
                            "squared residuals, information criteria and standard errors are "
                            "undefined.")
    covariance, info = linear_covariance(frame, x=x, resid=fit.resid, bread=fit.xtx_inv, n=n, k=k,
                                         df_resid=df_resid, ssr=ssr)
    has_constant = "Intercept" in design.terms
    total = float((y - y.mean()).square().sum()) if has_constant else float(y.square().sum())
    r_squared = 1.0 - ssr / total if total > 0 else math.nan

    def criteria(value: float, parameters: int) -> dict[str, float]:
        base = n * math.log(value / n)
        return {"aic": base + 2 * parameters, "bic": base + parameters * math.log(n),
                "hqic": base + 2 * parameters * math.log(math.log(n))}

    metrics = {"r_squared": r_squared,
               "adjusted_r_squared": 1.0 - (1.0 - r_squared) * (n - int(has_constant)) / df_resid,
               "rmse": math.sqrt(ssr / df_resid), "ssr": ssr, **criteria(ssr, k),
               "df_model": k - int(has_constant), "df_resid": df_resid}
    for index, gamma in enumerate(thresholds, start=1):
        metrics[f"threshold{index}"] = gamma
    null_ssr_units = null_ssr * grid.scale_y ** 2
    by_count = [{"thresholds": 0, "ssr": null_ssr_units, **criteria(null_ssr_units, kw + kv)}]
    for count, record in enumerate(sequence, start=1):
        by_count.append({"thresholds": count, "ssr": record["ssr"],
                         **criteria(record["ssr"], kw + (count + 1) * kv)})
    keep = torch.linspace(0, path_positions.shape[0] - 1,
                          steps=min(_PATH, path_positions.shape[0])).round().to(torch.int64)
    extra: dict[str, Any] = {
        "thresholds": thresholds, "threshold_variable": q_name, "trim": trim,
        "minimum_region_size": minimum,
        "region_sizes": [int((region == r).sum()) for r in range(m + 1)],
        "regions": varying_terms, "common": [design.terms[i] for i in common],
        "sequence": sequence, "by_number_of_thresholds": by_count,
        "candidates": int(path_positions.shape[0]),
        "ssr_path": {"threshold": grid.q[path_positions[keep] - 1].tolist(),
                     "ssr": (path_ssr[keep] * grid.scale_y ** 2).tolist(),
                     "note": "screened SSR of the first threshold search (normal equations); "
                             "at most 200 evenly spaced candidates"},
        "region_definition": "region 1: q <= threshold1; region r: threshold(r-1) < q <= threshold r",
    }
    if m == 1:
        level = 1.0 - spec.alpha
        critical = -2.0 * math.log(1.0 - math.sqrt(level))
        best_ssr = float(path_ssr.min())
        _check_lr_precision(best_ssr, grid.yy)
        lr = n * (path_ssr - best_ssr) / best_ssr
        inside = grid.q[path_positions[lr <= critical] - 1]
        extra["threshold_confidence_set"] = {
            "low": float(inside.min()), "high": float(inside.max()), "level": level,
            "critical_value": critical,
            "method": "Hansen (2000) LR inversion, homoskedastic asymptotic distribution, "
                      "screened SSR; reported as the hull of the accepted candidates"}
    tests: dict[str, Any] = {}
    slopes = [i for i, term in enumerate(terms) if not term.endswith("Intercept")]
    if slopes and has_constant:
        tests["model"] = wald_test(fit.beta, covariance, slopes, df_resid=df_resid,
                                   label="F test that all coefficients except the constants are zero")
    reps = int(frame.option("bootstrap"))
    if reps:
        # Linear model against the best ONE-threshold model (also when nthresholds > 1),
        # bootstrapped with that model's residuals (sorted, scaled units).
        single_fit = kernel_call(least_squares, grid.design(single), grid.y)
        tests["threshold_effect"] = _bootstrap(grid, single, single_fit.resid, null_ssr, reps,
                                               int(frame.option("seed")))
    info["correction"] = f"{info.get('correction')}; thresholds treated as known"
    return build_result(
        frame, terms=terms, params=fit.beta, covariance=covariance,
        title=f"Threshold regression ({m} threshold{'s' if m > 1 else ''})", equations=equations,
        df_inference=df_resid, df_resid=df_resid, metrics=metrics, fitted=fit.fitted,
        solver="grid_search_qr", solver_diagnostics={"candidates": int(path_positions.shape[0]),
                                                     "condition_number": fit.condition_number},
        inference=info, tests=tests, extra=extra, categories=design.categories,
        provenance={"threshold_search": "exhaustive over observed threshold values (screened by "
                    "normal equations, best candidates refitted by QR); sequential with Bai (1997) "
                    "refinement for several thresholds"},
    )


def _bootstrap(grid: _Grid, split: list[int], resid: Tensor, null_ssr: float, reps: int,
               seed: int) -> dict[str, Any]:
    """Hansen's fixed-regressor bootstrap of F = N (S0 - S1) / S1 for one threshold."""
    n = grid.n
    s1 = grid.exact(split)
    statistic = n * (null_ssr - s1) / s1
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randn((n, reps), generator=generator, dtype=torch.float64) * resid[:, None]
    exceed = 0
    original = grid.y
    try:
        for r in range(reps):
            grid.y = draws[:, r].contiguous()
            grid.yy = float(grid.y.square().sum())
            grid.wy = grid.xw.T @ grid.y
            null = grid.exact([])
            _, ssr = grid.ssr([], 1, n)
            alternative = float(ssr[torch.isfinite(ssr)].min())
            exceed += n * (null - alternative) / alternative >= statistic
    finally:
        grid.y = original
        grid.yy = float(original.square().sum())
        grid.wy = grid.xw.T @ original
    return {"statistic": statistic, "df": None, "p_value": (exceed + 0.0) / reps,
            "distribution": "bootstrap", "label": "Hansen (1996) fixed-regressor bootstrap test of "
            "no threshold effect (F = N (S0 - S1) / S1)", "reps": reps, "seed": seed}


def threshold(*, data: Any, y: str, x: Any, threshold_var: str, time: str | None = None,
              nthresholds: int = 1, trim: float = 0.1, regions: Any = None,
              covariance: str | None = None, categorical: Any = None, intercept: bool = True,
              bootstrap: int = 0, seed: int = 12345, missing: str = "raise",
              alpha: float = 0.05) -> ResultBundle:
    """Threshold regression with grid-searched thresholds (Stata's ``threshold``; Hansen 2000).

    Model
    -----
    ``y_t = w_t'a + x_t'b_r + e_t`` when ``gamma_(r-1) < q_t <= gamma_r`` (region r), with
    q = ``threshold_var``. ``regions`` lists the terms whose coefficients differ across
    regions (default: every term, including ``"Intercept"``); the other terms are common.

    Estimator
    ---------
    The thresholds minimize the SSR over the observed values of q with at least
    ``trim`` * N observations in every region (exhaustive search; prefix-sum normal
    equations screen every candidate, the best are refitted by QR). Several thresholds
    are estimated sequentially with Bai's (1997) refinement. Given the thresholds the
    coefficients are OLS.

    Parameters
    ----------
    data, y, x : table, outcome and regressors. threshold_var : the threshold variable
    (may be one of x, e.g. a lagged outcome for a TAR model). time : optional time column
    (used only to order the rows). nthresholds : number of thresholds (1..5). trim :
    minimum region share (0 < trim < 0.5, default 0.1 as Stata's trim(10)). regions : terms
    with region-specific coefficients. covariance : ``"nonrobust"`` (default),
    ``"robust"``/``"HC1"``, ``"HC2"``, ``"HC3"``. categorical, intercept : as usual. bootstrap :
    replications of Hansen's fixed-regressor bootstrap test of no threshold (0 = off).
    seed : random seed of the bootstrap. missing, alpha : as usual.

    Result
    ------
    Coefficients ``w`` terms, then ``region1:<term>``, ``region2:<term>``, ...; t tests on
    N - K degrees of freedom (thresholds treated as known). ``metrics``: ``r_squared``,
    ``adjusted_r_squared``, ``rmse``, ``ssr``, ``aic``/``bic``/``hqic`` (N ln(SSR/N) + penalty *
    K), ``threshold1``..., ``df_model``, ``df_resid``. ``tests``: ``model`` (F) and, with
    ``bootstrap``, ``threshold_effect`` (linear model against the best one-threshold model). ``extra``: ``thresholds``, ``region_sizes``,
    ``sequence``, ``by_number_of_thresholds`` (SSR and criteria for 0..m thresholds),
    ``ssr_path`` (at most 200 points), ``threshold_confidence_set`` (one threshold).

    Stata: ``threshold y x1, threshvar(q) regionvars(x2) nthresholds(2) trim(10)``.
    EViews: Threshold regression (``threshold`` estimation method).

    Example
    -------
    >>> fit = oe.threshold(data=df, y="growth", x=["inflation"], threshold_var="debt")
    >>> fit.extra["thresholds"], fit.extra["threshold_confidence_set"]
    """
    spec = build_spec(
        "threshold", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept, time=time,
        covariance=covariance, missing=missing, alpha=alpha,
        columns={"threshold_var": threshold_var},
        options={"nthresholds": as_int(nthresholds), "trim": trim,
                 "regions": None if regions is None else column_list(regions, "regions"),
                 "bootstrap": as_int(bootstrap), "seed": as_int(seed)},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
