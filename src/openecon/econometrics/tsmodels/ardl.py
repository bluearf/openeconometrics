"""Autoregressive distributed lag models (Stata's community ``ardl``; EViews: ARDL).

The ARDL(p, q_1, ..., q_k) model

    y_t = c_0 + c_1 t + sum_(i=1..p) phi_i y_(t-i) + sum_j sum_(l=0..q_j) b_jl x_(j,t-l)
          + w_t'g + e_t

is estimated by OLS (``engines.linalg.least_squares``, Householder QR) on the
observations after the largest lag. ``w`` are exogenous regressors entering only
contemporaneously (Stata's ``exog()``).

Lag selection. Without ``lags`` every combination p = 1..m_y, q_j = 0..m_j is
fitted on the common sample that holds back max(m) observations and the
combination minimizing AIC = -2 ll + 2 K or BIC = -2 ll + K ln N (ll the
Gaussian log likelihood, K the number of coefficients) is kept and reported on
that same sample. The grid is evaluated without refitting the data: the
deterministic and exogenous columns are partialled out once, the candidate lag
columns are written in nested difference form (y_(t-1), Dy_(t-1), ...,
x_t, Dx_t, ...: each model uses a prefix of every block and spans the same
space as its levels lags) and reduced by one QR factorization Z = Q R; the SSR
of any subset S is then ||y_perp||^2 + the residual of the small least-squares
problem (R_S, Q'y), solved by batched QR. Nothing is squared, so the selection
is as accurate as fitting every model by QR.

Error-correction form (Stata's ``ardl, ec``):

    D.y_t = c - a (y_(t-1) - theta'x_t) + sum_(i=1..p-1) psi_i D.y_(t-i)
            + sum_j sum_(l=0..q_j-1) omega_jl D.x_(j,t-l) + w_t'g + e_t

with a = 1 - sum phi_i (reported as ADJ:L.y = -a), theta_j = sum_l b_jl / a,
psi_i = -sum_(m>i) phi_m and omega_jl = -sum_(m>l) b_jm. The EC coefficients
are exact functions of the levels estimates; their covariance is the delta
method (exact for the linear short-run terms, first order for theta).

Bounds test (Pesaran, Shin and Smith 2001). In the conditional EC regression
the F statistic tests that the level coefficients (a and a*theta, plus the
restricted constant or trend in cases II and IV) are zero, and the t statistic
tests a = 0 (cases I, III, V). Both are computed from the levels estimates with
the classical OLS covariance (the PSS derivation assumes iid errors) and
compared with the asymptotic I(0)/I(1) bounds of ``bounds.py``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, build_result, column_list, kernel_call, linear_covariance, wald_test,
)
from openecon.econometrics.tsmodels import bounds
from openecon.econometrics.tsmodels.common import (
    as_int_list, build_spec, durbin_watson, lag, linear_restriction_test, ordered_frame,
    perfect_fit, require_variation,
)
from openecon.engines.distributions import t_sf
from openecon.engines.linalg import LeastSquares, collinear_columns, least_squares
from openecon.models import ModelSpec, ResultBundle

GRID_WARNING = 50_000
GRID_LIMIT = 2_000_000
_CHUNK = 20_000
_TOP = 10


def _case(trend: str, restricted: bool) -> int:
    if trend == "none":
        if restricted:
            raise AnalysisError("invalid_option", "restricted=True needs a deterministic term to "
                                "restrict: use trend='constant' (case II) or trend='trend' (case IV).")
        return 1
    if trend == "constant":
        return 2 if restricted else 3
    return 4 if restricted else 5


def _lag_name(name: str, lag_: int, difference: bool = False) -> str:
    prefix = ("L" if lag_ == 1 else f"L{lag_}" if lag_ > 1 else "") + ("D" if difference else "")
    return f"{prefix}.{name}" if prefix else name


def _read_orders(frame, k: int) -> tuple[list[int] | None, list[int]]:
    lags, maxlags = frame.option("lags"), frame.option("maxlags")
    if lags is not None:
        if len(lags) != k + 1 or any(value < 0 for value in lags):
            raise AnalysisError("invalid_lags", f"lags must list {k + 1} non-negative integers: the "
                                f"lag order of {frame.spec.outcome} followed by one order per "
                                "regressor in x, for example lags=[2, 1, 0].")
        return list(lags), list(lags)
    if maxlags is None:
        maxlags = [4]
    if len(maxlags) == 1:
        maxlags = maxlags * (k + 1)
    if len(maxlags) != k + 1 or maxlags[0] < 1 or any(value < 0 for value in maxlags):
        raise AnalysisError("invalid_lags", f"maxlags must be one integer (>= 1) or {k + 1} "
                            "integers: the largest lag of the outcome (>= 1) and of each regressor.")
    return None, list(maxlags)


def _select(y: Tensor, always: Tensor, blocks: list[Tensor], ranges: list[range], ic: str,
            frame, *, nobs=None) -> tuple[list[int], dict[str, Any]]:
    """Exhaustive prefix search with bounded candidate batches and ten records."""
    import heapq
    from openecon.resources import plan_workspace, workspace_budget_bytes

    n = y.shape[0] if nobs is None else nobs
    size = math.prod(len(values) for values in ranges)
    if size > GRID_LIMIT:
        raise AnalysisError("lag_grid_too_large", f"The lag grid has {size} combinations "
                            f"(limit {GRID_LIMIT}); lower maxlags or give lags explicitly.")
    if size > GRID_WARNING:
        frame.warn(f"The lag search fits {size} models; consider smaller maxlags.")
    if always.shape[1]:
        base = kernel_call(least_squares, always, torch.cat([y[:, None], *blocks], dim=1))
        residual = base.resid
        a = len(base.kept)
    else:
        residual, a = torch.cat([y[:, None], *blocks], dim=1), 0
    target, candidates = residual[:, 0], residual[:, 1:]
    scale = candidates.square().sum(dim=0).sqrt().clamp_min(1e-300)
    q, r = torch.linalg.qr(candidates / scale, mode="reduced")
    c = q.T @ target
    perpendicular = float((target-q@c).square().sum())
    offsets = list(itertools.accumulate([0, *(block.shape[1] for block in blocks)]))
    width = max(1, candidates.shape[1])
    budget = min(128*1024**2, workspace_budget_bytes())
    rows = min(_CHUNK, max(1, (budget-1024*width**2)//(128*width**2+512*width)))
    plan_workspace("bounded exhaustive ARDL lag search", {
        "master_and_small_factors": 1024*width**2,
        "candidate_QR_and_metadata": rows*(128*width**2+512*width),
    }, budget_bytes=budget)
    combos = iter(itertools.product(*ranges))
    best, processed = [], 0
    penalty = 2. if ic == "aic" else math.log(n)
    while chunk := list(itertools.islice(combos, rows)):
        by_size, columns = {}, []
        for index, combo in enumerate(chunk):
            chosen = [column for block, order in enumerate(combo)
                      for column in range(offsets[block], offsets[block]+(order if block == 0 else order+1))]
            columns.append(chosen)
            by_size.setdefault(len(chosen), []).append(index)
        ssr = torch.full((len(chunk),), math.inf, dtype=torch.float64)
        for count, members in by_size.items():
            if count == 0:
                ssr[members] = perpendicular+float(c.square().sum())
                continue
            idx = torch.tensor([columns[i] for i in members], dtype=torch.int64)
            sub = r[:, idx].permute(1, 0, 2)
            qs, rs = torch.linalg.qr(sub, mode="reduced")
            fitted = qs@(qs.transpose(1, 2)@c[None, :, None])
            rest = (c[None, :, None]-fitted).square().sum(dim=(1, 2))
            norms = sub.square().sum(dim=1).sqrt()
            deficient = (rs.diagonal(dim1=1, dim2=2).abs() <= 1e-9*norms.clamp_min(1e-300)).any(1)
            value = perpendicular+rest
            value[deficient] = math.inf
            ssr[torch.tensor(members)] = value
        count = torch.tensor([a+len(cols) for cols in columns], dtype=torch.float64)
        ll = -.5*n*(1.+math.log(2.*math.pi)+torch.log(ssr/n))
        criteria = -2.*ll+penalty*count
        records = [(float(value), processed+i, combo) for i, (value, combo) in enumerate(zip(criteria, chunk, strict=True))
                   if math.isfinite(float(value))]
        best = heapq.nsmallest(_TOP, itertools.chain(best, records))
        processed += len(chunk)
    if not best:
        raise AnalysisError("collinear_lags", "Every candidate lag combination is rank deficient; check for constant or duplicated regressors.")
    chosen = list(best[0][2])
    return chosen, {"criterion": ic, "models": size, "selected": chosen,
                    "best": [{"lags": list(combo), ic: value} for value, _, combo in best],
                    "sample": "common sample holding back the largest maxlag"}


def _delta(jacobian: Tensor, covariance: Tensor) -> Tensor:
    return jacobian @ covariance @ jacobian.T


def _constraint_basis(matrix: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Homogeneous constraints R b=0, reduced to independent rows by QR."""
    if matrix.ndim != 2 or matrix.shape[1] != k or not bool(torch.isfinite(matrix).all()):
        raise AnalysisError("invalid_constraint", "ARDL restrictions must be a finite matrix "
                            "with one column per levels coefficient.")
    kept, _ = kernel_call(collinear_columns, matrix.T)
    matrix = matrix[kept]
    rank = len(kept)
    if rank >= k:
        raise AnalysisError("invalid_constraint", "Restrictions leave no free coefficients.")
    if not rank:
        return torch.eye(k, dtype=torch.float64), matrix
    orthogonal, _ = torch.linalg.qr(matrix.T, mode="complete")
    return orthogonal[:, rank:], matrix


def _select_constrained(y, always, blocks, ranges, ic, frame, builder, names, always_terms, *, nobs=None):
    """Constrained lag search after one observation-level QR, in levels space.

    Each candidate solves only (R_columns Q2, Q'y). Information criteria count
    its free parameters; every candidate retains the same hold-back sample.
    """
    n = len(y) if nobs is None else nobs
    size = math.prod(len(values) for values in ranges)
    if size > GRID_LIMIT:
        raise AnalysisError("lag_grid_too_large", f"The lag grid has {size} combinations "
                            f"(limit {GRID_LIMIT}); lower maxlags or give lags explicitly.")
    if size > GRID_WARNING:
        frame.warn(f"The constrained lag search fits {size} models; consider smaller maxlags.")
    kept = kernel_call(collinear_columns, always)[0]
    always, always_terms = always[:, kept], [always_terms[i] for i in kept]
    master = torch.cat([always, *blocks], dim=1)
    scales = master.abs().amax(0).clamp_min(1e-300)
    orthogonal, factor = torch.linalg.qr(master / scales, mode="reduced")
    target = orthogonal.T @ y
    perpendicular = float((y - orthogonal @ target).square().sum())
    offsets = list(itertools.accumulate([always.shape[1], *(b.shape[1] for b in blocks)]))
    records = []
    import heapq
    for order_index, combo in enumerate(itertools.product(*ranges)):
        indices, terms = list(range(always.shape[1])), list(always_terms)
        for j, order in enumerate(combo):
            lags = range(1, order + 1) if j == 0 else range(order + 1)
            indices.extend(range(offsets[j], offsets[j] + len(lags)))
            terms.extend(_lag_name(frame.spec.outcome if j == 0 else names[j - 1], i)
                         for i in lags)
        basis, matrix = _constraint_basis(builder(terms, list(combo)), len(terms))
        free = basis.shape[1]
        try:
            candidate = kernel_call(least_squares, (factor[:, indices] * scales[indices]) @ basis,
                                    target, drop_collinear=False)
        except AnalysisError as exc:
            if exc.code != "singular_design":
                raise
            continue
        ssr = perpendicular + float(candidate.ssr)
        if not ssr > 0 or not math.isfinite(ssr):
            continue
        ll = -0.5 * n * (1 + math.log(2 * math.pi) + math.log(ssr / n))
        criterion = -2 * ll + (2 if ic == "aic" else math.log(n)) * free
        records.append((criterion, order_index, {"lags": list(combo), ic: criterion,
                                                "free_parameters": free, "constraints": matrix.shape[0]}))
        records = heapq.nsmallest(_TOP, records)
    if not records:
        raise AnalysisError("collinear_lags", "Every constrained lag combination is rank "
                            "deficient; check predictors or give lags explicitly.")
    records = [record for _, _, record in sorted(records)]
    return records[0]["lags"], {
        "criterion": ic, "models": size, "selected": records[0]["lags"],
        "best": records[:_TOP], "constraints_in_selection": True,
        "sample": "common sample holding back the largest maxlag",
    }


def fit_ardl(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``ardl`` estimator (see the module docstring and ``ardl``)."""
    frame, _ = ordered_frame(spec, data, what="ARDL estimation")
    return _fit_ardl_frame(frame)


def _fit_ardl_frame(frame, *, names: list[str] | None = None,
                    x_all: Tensor | None = None, retain_levels: bool = False,
                    constraint_builder=None, _replay=None) -> ResultBundle:
    """Fit an ordered frame, optionally using derived predictor tensors.

    NARDL supplies partial sums here while retaining the original input frame,
    row positions and fingerprints. The ARDL estimation and lag search stay shared.
    """
    spec = frame.spec
    names = list(spec.predictors) if names is None else names
    exog = frame.role("exog")
    clash = set(exog) & (set(names) | {spec.outcome})
    if clash:
        raise AnalysisError("invalid_spec", f"exog must not repeat the outcome or x: {', '.join(clash)}.")
    k = len(names)
    trend, restricted = frame.option("trend"), bool(frame.option("restricted"))
    ec, ic = bool(frame.option("ec")), frame.option("ic")
    case = _case(trend, restricted)
    lags, maxlags = _read_orders(frame, k)
    largest = max(maxlags)
    if _replay is None:
        total = frame.n
        y_all, w_all = frame.numeric(spec.outcome), frame.matrix(exog)
        x_all = frame.matrix(names) if x_all is None else x_all
        if x_all.shape != (frame.n, k):
            raise AnalysisError("invalid_design", "Derived predictors must align with the ordered sample.")
        require_variation(y_all, spec.outcome)
        rows = slice(largest, total)
        n = total - largest
        deterministic, det_terms = [], []
        if trend != "none":
            deterministic.append(torch.ones(total, dtype=torch.float64))
            det_terms.append("Intercept")
        if trend == "trend":
            deterministic.append(torch.arange(1, total + 1, dtype=torch.float64))
            det_terms.append("trend")
        det = torch.stack(deterministic, dim=1) if deterministic else torch.empty((total, 0),
                                                                                    dtype=torch.float64)
        always = torch.cat([det, w_all], dim=1)[rows]
        if n <= len(det_terms) + len(exog) + sum(maxlags) + k + 2:
            raise AnalysisError("insufficient_observations", f"{total} observations are too few for "
                                f"lags up to {largest} and the requested regressors.")
        selection = None
        if lags is None:
            blocks = [torch.stack([lag(y_all, 1)] + [lag(y_all, i) - lag(y_all, i + 1)
                                                     for i in range(1, maxlags[0])], dim=1)[rows]]
            for j in range(k):
                xj = x_all[:, j]
                blocks.append(torch.stack([xj] + [lag(xj, i) - lag(xj, i + 1)
                                                  for i in range(maxlags[j + 1])], dim=1)[rows])
            ranges = [range(1, maxlags[0] + 1)] + [range(0, m + 1) for m in maxlags[1:]]
            if constraint_builder is None:
                lags, selection = _select(y_all[rows], always, blocks, ranges, ic, frame)
            else:
                level_blocks = [torch.stack([lag(y_all, i) for i in range(1, maxlags[0] + 1)],
                                            dim=1)[rows]]
                level_blocks += [torch.stack([lag(x_all[:, j], i) for i in range(maxlags[j + 1] + 1)],
                                             dim=1)[rows] for j in range(k)]
                lags, selection = _select_constrained(y_all[rows], always, level_blocks, ranges, ic,
                                                      frame, constraint_builder, names, det_terms + exog)
        p, q = lags[0], lags[1:]
        if ec and p < 1:
            raise AnalysisError("invalid_option", "The error-correction form needs at least one lag of "
                                "the outcome (p >= 1).")
        columns, terms, roles = [], [], []
        for name, column in zip(det_terms, det.T, strict=True):
            columns.append(column)
            terms.append(name)
            roles.append(("det", name, 0))
        for i in range(1, p + 1):
            columns.append(lag(y_all, i))
            terms.append(_lag_name(spec.outcome, i))
            roles.append(("y", None, i))
        for j, name in enumerate(names):
            for lag_ in range(q[j] + 1):
                columns.append(lag(x_all[:, j], lag_))
                terms.append(_lag_name(name, lag_))
                roles.append(("x", j, lag_))
        for j, name in enumerate(exog):
            columns.append(w_all[:, j])
            terms.append(name)
            roles.append(("exog", name, 0))
        if len(set(terms)) != len(terms):
            raise AnalysisError("duplicate_terms", "Generated lag terms collide with column names; "
                                "rename the affected columns.")
        x_design = torch.stack(columns, dim=1)[rows]
        frame.restrict(torch.arange(total) >= largest,
                       f"The first {largest} observation(s) provide the initial lags.")
        y = y_all[rows]
    else:
        prepared = _replay.prepare(names, exog, lags, maxlags, trend, ec, ic, constraint_builder)
        total, n, largest = prepared['total'], prepared['n'], prepared['largest']
        lags, selection, terms, roles = (prepared[key] for key in ('lags', 'selection', 'terms', 'roles'))
        p, q = lags[0], lags[1:]
        y, x_design, det_terms = prepared['y'], prepared['x'], prepared['det_terms']
        y_all = None
        frame.warn(f"The first {largest} observation(s) provide the initial lags.")
    if constraint_builder is None:
        design = frame.drop_collinear(Design(x_design, terms, {}, "Intercept" in det_terms))
    else:
        # Restrictions can identify coefficients even when the unrestricted
        # lag design is deficient. Do not omit dynamic columns before R b=0.
        # Redundant unrestricted deterministic/exogenous columns remain safe
        # to screen on their own, as they never enter NARDL symmetry rows.
        static = [i for i, role in enumerate(roles) if role[0] in {"det", "exog"}]
        static_design = frame.drop_collinear(Design(
            x_design[:, static], [terms[i] for i in static], {}, "Intercept" in det_terms))
        keep = [i for i, role in enumerate(roles)
                if role[0] not in {"det", "exog"} or terms[i] in static_design.terms]
        design = Design(x_design[:, keep], [terms[i] for i in keep], {},
                        "Intercept" in static_design.terms)
    lost = [term for term, role in zip(terms, roles, strict=True)
            if role[0] in {"y", "x"} and term not in design.terms]
    if lost:
        raise AnalysisError("collinear_lags", f"Lag terms are collinear: {', '.join(lost)}. The ARDL "
                            "structure needs every lag; remove constant or duplicated regressors.")
    roles = [role for term, role in zip(terms, roles, strict=True) if term in design.terms]
    terms = design.terms
    if _replay is not None:
        _replay.set_design(design)
    covariance_fn = linear_covariance if _replay is None else _replay.covariance
    big_k = len(terms)
    df_resid = n - big_k
    if constraint_builder is None:
        if df_resid < 1:
            raise AnalysisError("insufficient_observations", "Too few observations for the ARDL model.")
        fit = kernel_call(least_squares, design.x, y, drop_collinear=False)
    constraint_record, constraint_test, parameter_basis = None, None, None
    if constraint_builder is not None:
        parameter_basis, matrix = _constraint_basis(constraint_builder(terms, lags), big_k)
        rank, free = matrix.shape[0], parameter_basis.shape[1]
        fixed_levels = (torch.linalg.vector_norm(parameter_basis, dim=1) <= 1e-12).nonzero().flatten()
        parameter_basis[fixed_levels] = 0.0
        if n - free < 1:
            raise AnalysisError("insufficient_observations", "Too few observations for the "
                                "constrained ARDL model.")
        if rank:
            reference = None
            try:
                if n - big_k > 0:
                    reference = kernel_call(least_squares, design.x, y, drop_collinear=False)
            except AnalysisError as exc:
                if exc.code != "singular_design":
                    raise
            if reference is not None and not perfect_fit(float(reference.ssr), y):
                unrestricted_cov, _ = covariance_fn(
                    frame, x=design.x, resid=reference.resid, bread=reference.xtx_inv, n=n,
                    k=big_k, df_resid=n - big_k, ssr=float(reference.ssr))
                constraint_test = linear_restriction_test(
                    reference.beta, unrestricted_cov, matrix, torch.zeros(rank, dtype=torch.float64),
                    df_resid=n - big_k, label="Wald test of imposed symmetry on the "
                    "same-sample unrestricted reference model")
                constraint_test.update(reference="unrestricted, same selected orders and sample",
                                       covariance=spec.covariance)
            elif reference is None:
                constraint_test = {
                    "statistic": None, "p_value": None, "df": rank, "df2": None,
                    "distribution": "F", "reference": "unavailable",
                    "covariance": spec.covariance,
                    "label": "Wald test of imposed symmetry on the same-sample "
                             "unrestricted reference model",
                    "note": "The unrestricted reference has a rank-deficient design or "
                            "no residual degrees of freedom; only the constrained model "
                            "is identified, so no unrestricted Wald statistic is reported.",
                }
        free_fit = kernel_call(least_squares, design.x @ parameter_basis, y, drop_collinear=False)
        fit = LeastSquares(
            beta=parameter_basis @ free_fit.beta, kept=list(range(big_k)), omitted=[],
            xtx_inv=parameter_basis @ free_fit.xtx_inv @ parameter_basis.T,
            fitted=free_fit.fitted, resid=free_fit.resid, ssr=free_fit.ssr, rank=free,
            condition_number=free_fit.condition_number, leverage=free_fit.leverage)
        df_resid = n - free
        if _replay is not None:
            _replay.parameter_basis = parameter_basis
        covariance_free, info = covariance_fn(
            frame, x=design.x @ parameter_basis, resid=fit.resid, bread=free_fit.xtx_inv,
            n=n, k=free, df_resid=df_resid, ssr=float(fit.ssr))
        covariance = parameter_basis @ covariance_free @ parameter_basis.T
        constraint_record = {"terms": terms, "matrix": matrix.tolist(), "rank": rank,
                             "free_parameters": free, "parameter_basis": parameter_basis.tolist(),
                             "free_covariance": covariance_free.tolist(),
                             "free_coefficients": free_fit.beta.tolist(),
                             "values": [0.0] * rank, "unrestricted_df_resid": n - big_k}
    beta, resid, ssr = fit.beta, fit.resid, float(fit.ssr)
    if perfect_fit(ssr, y):
        raise AnalysisError("perfect_fit", "The ARDL model fits the outcome exactly; standard errors "
                            "and the bounds test are undefined.")
    if constraint_builder is None:
        covariance, info = covariance_fn(frame, x=design.x, resid=resid, bread=fit.xtx_inv, n=n,
                                             k=big_k, df_resid=df_resid, ssr=ssr)
    classical = fit.xtx_inv * (ssr / df_resid)
    y_rows = [i for i, role in enumerate(roles) if role[0] == "y"]
    x_rows = [[i for i, role in enumerate(roles) if role[0] == "x" and role[1] == j]
              for j in range(k)]
    alpha_ = 1.0 - float(beta[y_rows].sum()) if y_rows else 1.0

    # Long-run coefficients theta_j = sum_l b_jl / (1 - sum phi) and their delta-method covariance.
    long_terms, long_jac = [], []
    lr_targets = [(name, x_rows[j]) for j, name in enumerate(names)]
    if case == 2:
        lr_targets.insert(0, ("Intercept", [terms.index("Intercept")]))
    if case == 4:
        lr_targets.insert(0, ("trend", [terms.index("trend")]))
    for name, members in lr_targets:
        row = torch.zeros(big_k, dtype=torch.float64)
        total_b = float(beta[members].sum())
        row[members] = 1.0 / alpha_
        row[y_rows] = total_b / alpha_ ** 2
        long_terms.append((name, total_b / alpha_))
        long_jac.append(row)
    jac_lr = torch.stack(long_jac) if long_jac else torch.empty((0, big_k), dtype=torch.float64)
    lr_cov = _delta(jac_lr, covariance)
    long_run = []
    for i, (name, value) in enumerate(long_terms):
        se = math.sqrt(max(float(lr_cov[i, i]), 0.0))
        stat = value / se if se > 0 else math.nan
        long_run.append({"term": name, "estimate": value, "std_error": se, "statistic": stat,
                         "p_value": 2.0 * t_sf(abs(stat), df_resid) if math.isfinite(stat) else None})

    tests: dict[str, Any] = {}
    if constraint_test is not None:
        tests["imposed_symmetry"] = constraint_test
    slopes = [i for i, term in enumerate(terms) if term != "Intercept" and
              (parameter_basis is None or float(parameter_basis[i].norm()) > 1e-12)]
    if slopes and ec:
        # The EC regressors span the levels regressors, so "every EC coefficient except the
        # constant is zero" (D.y_t = c + e_t) is the linear restriction phi_1 = 1 and every
        # other non-constant levels coefficient = 0; the Wald F is invariant to the mapping.
        matrix = torch.eye(big_k, dtype=torch.float64)[slopes]
        value = torch.zeros(len(slopes), dtype=torch.float64)
        value[slopes.index(y_rows[0])] = 1.0
        tests["model"] = linear_restriction_test(
            beta, covariance, matrix, value, df_resid=df_resid,
            label="F test that all error-correction coefficients except the constant are zero")
    elif slopes:
        tests["model"] = wald_test(beta, covariance, slopes, df_resid=df_resid,
                                   label="F test that all coefficients except the constant are zero"
                                   if "Intercept" in terms else "F test that all coefficients are zero")
    if p >= 1 and abs(alpha_) > 0:
        restrictions, values = [], []
        row = torch.zeros(big_k, dtype=torch.float64)
        row[y_rows] = 1.0
        restrictions.append(row)
        values.append(1.0)
        for members in x_rows:
            row = torch.zeros(big_k, dtype=torch.float64)
            row[members] = 1.0
            restrictions.append(row)
            values.append(0.0)
        if case in (2, 4):
            row = torch.zeros(big_k, dtype=torch.float64)
            row[terms.index("Intercept" if case == 2 else "trend")] = 1.0
            restrictions.append(row)
            values.append(0.0)
        f_test = linear_restriction_test(beta, classical, torch.stack(restrictions),
                                         torch.tensor(values, dtype=torch.float64),
                                         df_resid=df_resid, label="")
        crit = bounds.critical_values("F", case, k)
        tests["bounds_f"] = {
            "statistic": f_test["statistic"], "df": f_test["df"], "df2": df_resid, "p_value": None,
            "distribution": "PSS bounds F",
            "label": f"Pesaran-Shin-Smith bounds F test of no level relationship ({bounds.CASES[case]})",
            "case": case, "k": k, "critical_values": crit,
            "decision": bounds.decide("F", f_test["statistic"], crit) if crit else None,
            "note": "asymptotic critical values of Pesaran, Shin and Smith (2001) Table CI; the "
                    "tables give no p-values" if crit else "no tabulated bounds for this k",
        }
        if case in (1, 3, 5):
            se = math.sqrt(float(classical[y_rows][:, y_rows].sum()))
            t_value = -alpha_ / se
            crit_t = bounds.critical_values("t", case, k)
            tests["bounds_t"] = {
                "statistic": t_value, "df": None, "p_value": None, "distribution": "PSS bounds t",
                "label": f"Pesaran-Shin-Smith bounds t test of no level relationship ({bounds.CASES[case]})",
                "case": case, "k": k, "critical_values": crit_t,
                "decision": bounds.decide("t", t_value, crit_t) if crit_t else None,
                "note": "asymptotic critical values of Pesaran, Shin and Smith (2001) Table CII; "
                        "one-sided (large negative values reject)" if crit_t else
                        "no tabulated bounds for this k",
            }

    fitted = y - resid
    observed = y
    report_terms, report_params, report_cov, equations = terms, beta, covariance, None
    report_jacobian = torch.eye(big_k, dtype=torch.float64) if parameter_basis is not None else None
    if ec:
        report_terms, jac, equations = [], [], []
        values: list[float] = []

        def add(term: str, equation: str, gradient: Tensor, value: float) -> None:
            report_terms.append(f"{equation}:{term}")
            equations.append(equation)
            jac.append(gradient)
            values.append(value)

        row = torch.zeros(big_k, dtype=torch.float64)
        row[y_rows] = 1.0
        add(_lag_name(spec.outcome, 1), "ADJ", row, -alpha_)
        for i, (name, value) in enumerate(long_terms):
            add(name, "LR", jac_lr[i], value)
        for i in range(1, p):
            row = torch.zeros(big_k, dtype=torch.float64)
            members = [y_rows[m - 1] for m in range(i + 1, p + 1)]
            row[members] = -1.0
            add(_lag_name(spec.outcome, i, difference=True), "SR", row, -float(beta[members].sum()))
        for j, name in enumerate(names):
            for lag_ in range(q[j]):
                row = torch.zeros(big_k, dtype=torch.float64)
                members = x_rows[j][lag_ + 1:]
                row[members] = -1.0
                add(_lag_name(name, lag_, difference=True), "SR", row, -float(beta[members].sum()))
        for i, role in enumerate(roles):
            if role[0] == "exog" or (role[0] == "det" and not (
                    (case == 2 and role[1] == "Intercept") or (case == 4 and role[1] == "trend"))):
                row = torch.zeros(big_k, dtype=torch.float64)
                row[i] = 1.0
                add(terms[i], "SR", row, float(beta[i]))
        jacobian = torch.stack(jac)
        report_jacobian = jacobian
        report_params = torch.tensor(values, dtype=torch.float64)
        report_cov = _delta(jacobian, covariance) if parameter_basis is None else (
            _delta(jacobian @ parameter_basis, covariance_free))
        lagged = lag(y_all, 1)[largest:] if _replay is None else _replay.lagged_outcome_factor
        observed, fitted = y - lagged, fitted - lagged
    fixed_terms = {}
    if parameter_basis is not None:
        projected = report_jacobian @ parameter_basis
        fixed = (torch.linalg.vector_norm(projected, dim=1) <= 1e-12 *
                 torch.linalg.vector_norm(report_jacobian, dim=1).clamp_min(1)).nonzero().flatten().tolist()
        fixed_terms = {report_terms[i]: float(report_params[i]) for i in fixed}
        keep = [i for i in range(len(report_terms)) if i not in set(fixed)]
        report_terms, report_params = [report_terms[i] for i in keep], report_params[keep]
        report_cov = report_cov[keep][:, keep]
        equations = [equations[i] for i in keep] if equations is not None else None
    response = observed
    centered = float((response - response.mean()).square().sum()) if "Intercept" in terms \
        else float(response.square().sum())
    if _replay is not None:
        centered = _replay.response_tss(ec, "Intercept" in terms)
    r_squared = 1.0 - ssr / centered if centered > 0 else math.nan
    has_constant = int("Intercept" in terms)
    ll = -0.5 * n * (1.0 + math.log(2.0 * math.pi) + math.log(ssr / n))
    order_label = "ARDL(" + ",".join(str(value) for value in lags) + ")"
    metrics = {
        "r_squared": r_squared,
        "adjusted_r_squared": 1.0 - (1.0 - r_squared) * (n - has_constant) / df_resid,
        "rmse": math.sqrt(ssr / df_resid), "log_likelihood": ll, "aic": -2 * ll + 2 * big_k,
        "bic": -2 * ll + math.log(n) * big_k, "hqic": -2 * ll + 2 * big_k * math.log(math.log(n)),
        "durbin_watson": durbin_watson(resid) if _replay is None else _replay.durbin_watson(beta), "df_model": big_k - has_constant,
        "df_resid": df_resid, "ssr": ssr,
    }
    if constraint_record is not None:
        free = constraint_record["free_parameters"]
        metrics.update(aic=-2 * ll + 2 * free, bic=-2 * ll + math.log(n) * free,
                       hqic=-2 * ll + 2 * free * math.log(math.log(n)),
                       df_model=free - has_constant, df_constraints=constraint_record["rank"])
        info.update(constraints=constraint_record["rank"], free_parameters=free, n_parameters=free)
    extra = {
        "ardl_order": order_label, "lags": {spec.outcome: p, **dict(zip(names, q, strict=True))},
        "lag_selection": selection, "case": case, "case_label": bounds.CASES[case],
        "trend": trend, "restricted": restricted, "ec": ec, "speed_of_adjustment": -alpha_,
        "long_run": long_run, "trend_definition": "t = 1, 2, ... over the ordered series "
                                                  "(the first row is 1)" if trend == "trend" else None,
        "levels_coefficients": {term: float(value) for term, value in zip(terms, beta, strict=True)},
        "levels_covariance": covariance.tolist(),
    }
    if retain_levels:
        extra["levels_coefficients"] = dict(zip(terms, beta.tolist(), strict=True))
        extra["levels_covariance"] = covariance.tolist()
    if constraint_record is not None:
        extra["constraints"] = constraint_record
        extra["constrained_terms"] = fixed_terms
    info["correction"] = f"{info.get('correction')}" + (
        "; EC coefficients by the delta method from the levels estimates" if ec else "")
    if _replay is not None:
        fitted, observed = _replay.preview(beta, ec)
    bundle = build_result(
        frame, terms=report_terms, params=report_params, covariance=report_cov,
        title=f"{order_label} regression" + (", error-correction form" if ec else ""),
        nobs=n, equations=equations, df_inference=df_resid, df_resid=df_resid, metrics=metrics,
        fitted=fitted, observed=observed, solver="qr_least_squares" if constraint_builder is None
        else "constraint_reparameterization_householder_qr",
        solver_diagnostics={"condition_number": fit.condition_number, "rank": fit.rank},
        inference=info, tests=tests, extra=extra, categories={},
        provenance={"lag_selection": "exhaustive grid by " + ic.upper() if selection else
                    "lags given by the user",
                    "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
                    "fitted_values": "one-step fitted values of D.y" if ec else
                    "one-step fitted values of y",
                    "bounds_covariance": "classical OLS covariance (iid errors, as in PSS 2001)"},
    )

    return bundle if _replay is None else _replay.finalize(bundle)


def ardl(*, data: Any, y: str, x: Any, time: str | None = None, lags: Any = None,
         maxlags: Any = 4, ic: str = "aic", trend: str = "constant", restricted: bool = False,
         exog: Any = None, ec: bool = False, covariance: str | None = None, missing: str = "raise",
         alpha: float = 0.05) -> ResultBundle:
    """Autoregressive distributed lag (ARDL) model with lag selection and the bounds test.

    Model
    -----
    ``y_t = c_0 [+ c_1 t] + sum_(i=1..p) phi_i y_(t-i) + sum_j sum_(l=0..q_j) b_jl x_(j,t-l)
    + w_t'g + e_t``, estimated by OLS (QR) on the observations after the largest lag.
    With ``p = 0`` (``lags=[0, ...]``) this is a finite distributed-lag model.

    Lag selection
    -------------
    ``lags=[p, q_1, ..., q_k]`` fixes the orders. Otherwise every combination
    ``p = 1..maxlags_y`` and ``q_j = 0..maxlags_j`` is fitted on the common sample
    that holds back ``max(maxlags)`` observations, and the one minimizing ``ic``
    (``"aic"``: -2 ll + 2K, default as in EViews; ``"bic"``: -2 ll + K ln N, the
    default of Kripfganz and Schneider's Stata ``ardl``) is reported on that same
    sample. ``maxlags`` is one integer for all variables (default 4) or k + 1 integers.

    Error-correction form and bounds test
    -------------------------------------
    ``ec=True`` reports Stata's ``ardl, ec`` parameterization: ``ADJ:L.y`` = -(1 - sum
    phi) (speed of adjustment), ``LR:x_j`` = theta_j = sum_l b_jl / (1 - sum phi) (long-run
    coefficients) and ``SR:`` short-run terms on ``LD.y``, ``D.x``, ``LD.x``, ...;
    covariance by the delta method. The Pesaran-Shin-Smith (2001) bounds tests are
    always computed: ``tests["bounds_f"]`` (F test that the level terms are zero) and,
    for cases I, III and V, ``tests["bounds_t"]`` (t test of the adjustment
    coefficient), with the asymptotic I(0)/I(1) critical-value bounds at 10%, 5% and
    1% and a decision per level (reject above the I(1) bound, do not reject below the
    I(0) bound, inconclusive in between). The PSS case follows from ``trend`` and
    ``restricted``: ``"none"`` -> I; ``"constant"`` -> III (``restricted=True``: II, the
    constant enters the long-run relation); ``"trend"`` -> V (``restricted=True``: IV).

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of row records.
    y : outcome column. x : list of regressors entering with lags (k >= 1).
    time : optional time column (consecutive integer periods or datetimes).
    lags, maxlags, ic : see Lag selection.
    trend : ``"constant"`` (default), ``"trend"`` (constant and linear trend t = 1, 2, ...)
        or ``"none"``. restricted : restrict the constant (trend) to the long run.
    exog : list of exogenous regressors entering only contemporaneously (not in the
        bounds test or long run; Stata's ``exog()``).
    ec : report the error-correction form.
    covariance : ``"nonrobust"`` (default), ``"robust"``/``"HC1"``, ``"HC2"``, ``"HC3"`` (used for
        the coefficient table and the long-run coefficients; the bounds tests use the
        classical covariance, as PSS).
    missing : ``"raise"`` or ``"drop"`` (rows only at the start or end).
    alpha : significance level of the confidence intervals.

    Result
    ------
    Coefficients in levels (``Intercept``, ``trend``, ``L.y``, ``L2.y``, ``x``, ``L.x``, ...,
    exog) or in EC form, with t tests on N - K degrees of freedom. ``metrics``:
    ``r_squared`` (of y, or of D.y in EC form), ``adjusted_r_squared``, ``rmse``,
    ``log_likelihood``, ``aic``, ``bic``, ``hqic``, ``durbin_watson``, ``df_model``,
    ``df_resid``, ``ssr``. ``tests``: ``model`` (F that every reported coefficient except the
    constant is zero: the levels slopes, or in EC form the EC coefficients, i.e. D.y_t = c +
    e_t; all coefficients without a constant), ``bounds_f``, ``bounds_t``.
    ``extra``: ``ardl_order``, ``lags``, ``lag_selection`` (criterion, number of models,
    the ten best), ``case``, ``speed_of_adjustment``, ``long_run`` (theta with delta-method
    standard errors, t tests), ``levels_coefficients`` (in EC form).

    Stata: ``ardl y x1 x2, lags(2 1 0)``, ``ardl y x1 x2, maxlags(4) aic ec``, then
    ``estat ectest``. EViews: ARDL estimation with automatic selection and the Bounds
    Test view.

    Example
    -------
    >>> fit = oe.ardl(data=df, y="ln_consumption", x=["ln_income"], time="quarter", ec=True)
    >>> fit.extra["ardl_order"], fit.tests["bounds_f"]["decision"]
    """
    spec = build_spec(
        "ardl", outcome=y, predictors=column_list(x, "x"), time=time, covariance=covariance,
        missing=missing, alpha=alpha, columns={"exog": column_list(exog, "exog")},
        options={"lags": as_int_list(lags, "lags"),
                 "maxlags": as_int_list(maxlags, "maxlags") if lags is None else None, "ic": ic, "trend": trend, "restricted": restricted,
                 "ec": ec},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
