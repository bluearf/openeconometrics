"""Common-factor panel regressions using native float64 QR.

Static models allow observed-row unbalanced panels; dynamic models require
balanced consecutive panels. Rank-deficient units are never silently dropped.
Mean-group inference uses dispersion across the unit coefficients.
"""
from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec
from openecon.engines.linalg import collinear_columns, least_squares


def _ols(x, y, *, strict=True):
    return kernel_call(least_squares, x.contiguous(), y.contiguous(), drop_collinear=not strict)


def _residualize(z, value):
    return _ols(z, value, strict=False).resid


def _slopes(z, x):
    # Screen the original joint design: projecting a constant out first leaves
    # roundoff that a scale-relative standalone screen would mistake for signal.
    _, omitted = kernel_call(collinear_columns, torch.cat((z, x), dim=1))
    if omitted:
        raise AnalysisError("singular_design", "A unit slope is unidentified after its common-factor controls.")
    return _residualize(z, x)


def _long_run(beta, covariance, p, q, k):
    companion = torch.zeros((p, p), dtype=torch.float64)
    companion[0] = beta[:p]
    if p > 1:
        companion[1:, :-1] = torch.eye(p - 1, dtype=torch.float64)
    if bool((torch.linalg.eigvals(companion).abs() >= 1).any()):
        raise AnalysisError("unstable_long_run", "A unit's autoregressive roots do not admit a stable long-run multiplier.")
    denominator = 1 - beta[:p].sum()
    if abs(float(denominator)) < 1e-8:
        raise AnalysisError("undefined_long_run", "A unit's autoregressive sum is too close to one.")
    result, gradients = [], []
    for j in range(k):
        positions = torch.arange(p + j * (q + 1), p + (j + 1) * (q + 1))
        numerator = beta[positions].sum()
        gradient = torch.zeros_like(beta)
        gradient[:p] = numerator / denominator.square()
        gradient[positions] = 1 / denominator
        result.append(numerator / denominator)
        gradients.append(gradient)
    jacobian = torch.stack(gradients)
    return torch.stack(result), jacobian @ covariance @ jacobian.T


def fit_xtcce(spec, data):
    with torch.device("cpu"), torch.no_grad():
        return _fit(spec, data)


def _fit(spec, data):
    frame = ModelFrame(spec, data)
    frame.sort_panel()
    model = frame.option("model")
    p, q, cl = frame.option("y_lags"), frame.option("x_lags"), frame.option("cs_lags")
    dynamic = model in {"dcce", "csardl"}
    p = (1 if dynamic else 0) if p is None else p
    q = (1 if model in {"csardl", "csdl"} else 0) if q is None else q
    if (model in {"ccemg", "ccep", "amg"} and (p or q)) or (model == "csdl" and p):
        raise AnalysisError("unsupported_domain", "Static CCE/AMG require zero lags; CS-DL has no lagged outcome.")
    if dynamic and p < 1:
        raise AnalysisError("invalid_lags", "Dynamic CCE/CS-ARDL require at least one outcome lag.")
    codes, count = frame.codes(spec.panel)
    sizes = torch.bincount(codes, minlength=count)
    if count < 3:
        raise AnalysisError("insufficient_panels", "Common-factor inference needs at least three units.")
    clock_values = frame.time_index()
    if not bool((sizes == sizes[0]).all()):
        if model in {"ccemg", "ccep", "amg"} and cl in {None, 0}:
            return _unbalanced_static(frame, codes, count, model, clock_values)
        raise AnalysisError("unbalanced_panel", "Dynamic/lag-augmented CCE requires a complete balanced panel.")
    periods, k = int(sizes[0]), len(spec.predictors)
    clock = clock_values.reshape(count, periods)
    if not bool((clock == clock[0]).all()) or not bool((clock[:, 1:] - clock[:, :-1] == 1).all()):
        if model in {"ccemg", "ccep", "amg"} and cl in {None, 0}:
            return _unbalanced_static(frame, codes, count, model, clock_values)
        raise AnalysisError("time_gaps", "Every unit must have the same consecutive integer periods.")
    cl = (int(periods ** (1 / 3)) if model in {"dcce", "csardl", "csdl"} else 0) if cl is None else cl
    if model == "amg" and cl:
        raise AnalysisError("invalid_lags", "AMG extracts a common process rather than lagged cross-sectional means.")
    start = max(p, q, cl)
    width = p + k * (q + 1)
    nuisance_width = 1 + int(frame.option("trend")) + (1 if model == "amg" else (k + 1) * (cl + 1))
    used = periods - start
    if used <= width + nuisance_width:
        raise AnalysisError("insufficient_observations", "Too few periods for the specified unit slopes and common-factor controls.")
    planned_work = 4 * count * used * (width + nuisance_width) ** 2
    if model == "amg":
        planned_work += 4 * count * (periods - 1) * (k + periods - 1) ** 2
    if planned_work > frame.option("max_work"):
        raise AnalysisError("work_budget", "The admitted unit QR work exceeds max_work.")
    frame.workspace_plan("common-factor panel regression", {
        "panels_and_fitted": 64 * frame.n * (k + 2),
        "unit_and_pooled_design": 64 * count * used * (width + nuisance_width),
        "unit_covariances_and_result": 64 * count * (width + nuisance_width) ** 2,
        "amg_first_stage": 64 * frame.n * (k + periods) if model == "amg" else 0,
    })
    values = frame.matrix([spec.outcome, *spec.predictors]).reshape(count, periods, k + 1)
    averages = values.mean(dim=0)
    process = None
    if model == "amg":
        differences = values[:, 1:] - values[:, :-1]
        # Differenced level-time dummies; omitted first level fixes location.
        time_design = torch.eye(periods, dtype=torch.float64)[1:, 1:] - torch.eye(periods, dtype=torch.float64)[:-1, 1:]
        stage_x = torch.cat((differences[:, :, 1:].reshape(-1, k), time_design.repeat(count, 1)), dim=1)
        stage = _ols(stage_x, differences[:, :, 0].reshape(-1))
        process = torch.cat((torch.zeros(1, dtype=torch.float64), stage.beta[k:]))
    z_columns = [torch.ones(used, dtype=torch.float64)]
    if frame.option("trend"):
        z_columns.append(torch.arange(start, periods, dtype=torch.float64))
    nuisance_terms = ["Intercept"] + (["Trend"] if frame.option("trend") else [])
    if model == "amg":
        z_columns.append(process[start:])
        nuisance_terms.append("AMG common process")
    else:
        for j, name in enumerate([spec.outcome, *spec.predictors]):
            limit = 0 if model == "csdl" and j == 0 else cl
            for lag in range(limit + 1):
                z_columns.append(averages[start - lag:periods - lag, j])
                nuisance_terms.append(f"CSA.{name}.L{lag}")
    z = torch.stack(z_columns, dim=1)
    z_fit = _ols(z, values[0, start:, 0], strict=False)
    nuisance_rank = len(z_fit.kept)
    z = z[:, z_fit.kept]
    terms = [f"L{lag}.{spec.outcome}" for lag in range(1, p + 1)]
    for name in spec.predictors:
        terms.append(name)
        terms.extend([f"D.L{lag}.{name}" for lag in range(q)] if model == "csdl"
                     else [f"L{lag}.{name}" for lag in range(1, q + 1)])
    x_units, y_units, estimates, variances = [], [], [], []
    for unit in range(count):
        v = values[unit]
        columns = [v[start - lag:periods - lag, 0] for lag in range(1, p + 1)]
        for j in range(1, k + 1):
            columns.append(v[start:, j])
            for lag in range(q):
                columns.append(v[start - lag:periods - lag, j] - v[start - lag - 1:periods - lag - 1, j] if model == "csdl" else v[start - lag - 1:periods - lag - 1, j])
        x = _slopes(z, torch.stack(columns, dim=1))
        y = _residualize(z, v[start:, 0])
        fit = _ols(x, y)
        df = used - width - nuisance_rank
        variance = fit.xtx_inv * float(fit.ssr) / df
        if not bool(torch.isfinite(variance).all()) or bool((variance.diagonal() <= 0).any()):
            raise AnalysisError("invalid_covariance", "A unit regression has no positive residual uncertainty.")
        x_units.append(x)
        y_units.append(y)
        estimates.append(fit.beta)
        variances.append(variance)
    unit_beta, unit_v = torch.stack(estimates), torch.stack(variances)
    if model == "ccep":
        xx, yy = torch.cat(x_units), torch.cat(y_units)
        fit = _ols(xx, yy)
        beta = fit.beta
        scores = torch.stack([x.T @ (y - x @ beta) for x, y in zip(x_units, y_units)])
        variance = fit.xtx_inv @ (scores.T @ scores) @ fit.xtx_inv * count / (count - 1)
        correction = "unit-cluster sandwich G/(G-1); common slopes, unit-specific CSA loadings"
    else:
        beta = unit_beta.mean(dim=0)
        deviations = unit_beta - beta
        variance = deviations.T @ deviations / (count * (count - 1))
        correction = "mean-group between-unit dispersion / G"
    predictions = []
    for unit, (x, y) in enumerate(zip(x_units, y_units)):
        b = beta if model == "ccep" else unit_beta[unit]
        predictions.append(values[unit, start:, 0] - (y - x @ b))
    labels = frame.sample.groupby(spec.panel, sort=False, observed=True)[spec.panel].first().tolist()
    extra = {"model": model, "unit_labels": labels, "unit_terms": terms,
        "unit_coefficients": unit_beta.tolist(), "unit_covariance": unit_v.tolist(),
        "common_factor_controls": [nuisance_terms[i] for i in z_fit.kept],
        "omitted_common_factor_controls": [nuisance_terms[i] for i in z_fit.omitted],
        "common_process": None if process is None else process.tolist(),
        "cross_sectional_mean_domain": "all complete input units before lag restriction",
        "lags": {"y": p, "x": q, "cross_sectional": cl}, "unit_nobs": used,
        "planned_qr_work": planned_work, "max_work": frame.option("max_work"),
        "assumptions": "factor rank covered by CSA; heterogeneous loadings; weakly dependent idiosyncratic errors; strict exogeneity static / weak exogeneity dynamic"}
    if model == "csardl":
        lr = [_long_run(b, v, p, q, k) for b, v in zip(unit_beta, unit_v)]
        lr_b = torch.stack([b for b, _ in lr])
        lr_mean = lr_b.mean(dim=0)
        centered = lr_b - lr_mean
        extra["long_run"] = {"terms": list(spec.predictors), "estimate": lr_mean.tolist(),
            "covariance": (centered.T @ centered / (count * (count - 1))).tolist(),
            "unit_coefficients": lr_b.tolist(), "unit_delta_covariance": [v.tolist() for _, v in lr],
            "target": "mean of unit-specific long-run ratios"}
    if model == "csdl":
        positions = [j * (q + 1) for j in range(k)]
        extra["long_run"] = {"terms": list(spec.predictors), "estimate": beta[positions].tolist(),
            "covariance": variance[positions][:, positions].tolist(), "target": "direct long-run level slopes"}
    keep = torch.arange(frame.n).reshape(count, periods)[:, start:].flatten()
    mask = torch.zeros(frame.n, dtype=torch.bool)
    mask[keep] = True
    frame.restrict(mask, f"Excluded the first {start} periods per unit for declared lags/common process.")
    return build_result(frame, terms=terms, params=beta, covariance=variance,
        fitted=torch.cat(predictions), solver="native float64 QR with unit-specific common-factor residualization",
        use_t=False, metrics={"n_groups": count, "unit_periods": used},
        inference={"correction": correction, "target": "pooled slopes" if model == "ccep" else "unit-mean slopes"}, extra=extra)


def _unbalanced_static(frame, codes, count, model, clock):
    """Static unit OLS uses period means of all observed complete rows.

    No missing unit is invented and no unit regression is silently dropped.
    AMG differences use only consecutive observed pairs; its second stage
    includes every complete level observation.
    """
    spec = frame.spec
    k = len(spec.predictors)
    periods, times = torch.unique(clock, sorted=True, return_inverse=True)
    if len(periods) < 2 or not bool((periods[1:] - periods[:-1] == 1).all()):
        raise AnalysisError("time_gaps", "The global common-factor calendar must be consecutive.")
    t = len(periods)
    nuisance_width = 1 + int(frame.option("trend")) + (1 if model == "amg" else k + 1)
    work = 4 * frame.n * (k + nuisance_width) ** 2
    if model == "amg":
        work += 4 * frame.n * (k + t - 1) ** 2
    if work > frame.option("max_work"):
        raise AnalysisError("work_budget", "Admitted unit/common-process QR work exceeds max_work.")
    frame.workspace_plan("unbalanced static common-factor panel", {
        "observed_and_residualized_rows": 96 * frame.n * (k + nuisance_width + 2),
        "unit_coefficients_and_results": 96 * count * (k + nuisance_width) ** 2,
        "amg_common_stage": 96 * frame.n * (k + t) if model == "amg" else 0,
    })
    values = frame.matrix([spec.outcome, *spec.predictors])
    average = torch.zeros((t, k + 1), dtype=torch.float64)
    average.index_add_(0, times, values)
    average /= torch.bincount(times, minlength=t)[:, None]
    process = None
    if model == "amg":
        adjacent = (codes[1:] == codes[:-1]) & (clock[1:] - clock[:-1] == 1)
        current = adjacent.nonzero().flatten() + 1
        if not len(current):
            raise AnalysisError("insufficient_observations", "AMG first stage needs consecutive within-unit pairs.")
        changes = values[current] - values[current - 1]
        basis = torch.eye(t, dtype=torch.float64)[:, 1:]
        stage_x = torch.cat((changes[:, 1:], basis[times[current]] - basis[times[current - 1]]), dim=1)
        stage = _ols(stage_x, changes[:, 0])
        process = torch.cat((torch.zeros(1, dtype=torch.float64), stage.beta[k:]))
    control_names = ["Intercept"] + (["Trend"] if frame.option("trend") else [])
    control_names += ["AMG common process"] if model == "amg" else [f"CSA.{name}.L0" for name in [spec.outcome, *spec.predictors]]
    unit_b, unit_v, xs, ys, zs, used, ranks, controls = [], [], [], [], [], [], [], []
    for unit in range(count):
        rows = (codes == unit).nonzero().flatten()
        tt = times[rows]
        z = [torch.ones(len(rows), dtype=torch.float64)]
        if frame.option("trend"):
            z.append((clock[rows] - clock.min()).to(torch.float64))
        z.extend([process[tt]] if model == "amg" else [average[tt, j] for j in range(k + 1)])
        z = torch.stack(z, dim=1)
        auxiliary = _ols(z, values[rows, 0], strict=False)
        z = z[:, auxiliary.kept]
        rank = len(auxiliary.kept)
        if len(rows) <= k + rank:
            raise AnalysisError("insufficient_observations", "A unit has too few rows for its common-factor regression.")
        x = _slopes(z, values[rows, 1:])
        y = _residualize(z, values[rows, 0])
        fit = _ols(x, y)
        unit_b.append(fit.beta)
        variance = fit.xtx_inv * float(fit.ssr) / (len(rows) - k - rank)
        if not bool(torch.isfinite(variance).all()) or bool((variance.diagonal() <= 0).any()):
            raise AnalysisError("invalid_covariance", "A unit regression has no positive residual uncertainty.")
        unit_v.append(variance)
        xs.append(x)
        ys.append(y)
        zs.append((rows, z))
        used.append(len(rows))
        ranks.append(rank)
        controls.append({"retained": [control_names[i] for i in auxiliary.kept],
                         "omitted": [control_names[i] for i in auxiliary.omitted]})
    unit_b, unit_v = torch.stack(unit_b), torch.stack(unit_v)
    if model == "ccep":
        fit = _ols(torch.cat(xs), torch.cat(ys))
        beta = fit.beta
        scores = torch.stack([x.T @ (y - x @ beta) for x, y in zip(xs, ys)])
        v = fit.xtx_inv @ (scores.T @ scores) @ fit.xtx_inv * count / (count - 1)
    else:
        beta = unit_b.mean(dim=0)
        deviations = unit_b - beta
        v = deviations.T @ deviations / (count * (count - 1))
    fitted = torch.empty(frame.n, dtype=torch.float64)
    loadings = []
    for unit, (x, y, (rows, z)) in enumerate(zip(xs, ys, zs)):
        b = beta if model == "ccep" else unit_b[unit]
        fitted[rows] = values[rows, 0] - (y - x @ b)
        loadings.append(_ols(z, values[rows, 0] - values[rows, 1:] @ b).beta.tolist())
    labels = frame.sample.groupby(spec.panel, sort=False, observed=True)[spec.panel].first().tolist()
    return build_result(frame, terms=spec.predictors, params=beta, covariance=v, fitted=fitted,
        use_t=False, solver="native float64 unit QR; observed-row period means",
        metrics={"n_groups": count, "unit_periods_min": min(used), "unit_periods_max": max(used)},
        inference={"target": "pooled slopes" if model == "ccep" else "unit-mean slopes",
                   "correction": "unit-cluster G/(G-1)" if model == "ccep" else "between-unit dispersion / G"},
        extra={"model": model, "balanced": False, "unit_labels": labels, "unit_terms": list(spec.predictors),
            "unit_coefficients": unit_b.tolist(), "unit_covariance": unit_v.tolist(),
            "unit_nobs": used, "unit_control_rank": ranks, "unit_controls": controls,
            "unit_control_coefficients": loadings, "lags": {"y": 0, "x": 0, "cross_sectional": 0},
            "common_process": None if process is None else process.tolist(),
            "cross_sectional_mean_domain": "all observed complete model-input rows in each period",
            "planned_qr_work": work, "max_work": frame.option("max_work"),
            "assumptions": "factor span/rank and heterogeneous loading conditions; static strict exogeneity"})


def xtcce(*, data: Any, y: str, x: list[str], panel: str, time: str,
          model: str = "ccemg", y_lags: int | None = None, x_lags: int | None = None,
          cs_lags: int | None = None, trend: bool = False, missing: str = "raise",
          max_work: int = 100_000_000, alpha: float = .05):
    """CCEMG/CCEP, AMG, dynamic CCE, CS-ARDL or CS-DL on declared panel domains.

    Slopes have unit-dispersion inference (CCEP: unit-cluster sandwich). CSA
    includes y/x, with declared lags. Static models allow unbalanced observed
    rows; dynamic/lag-augmented models require balanced consecutive clocks.
    No categorical terms, weights or implicit unit omission.
    See docs/econometrics/common-factors.md.
    """
    from openecon.analysis import fit
    return fit(make_spec("xtcce", outcome=y, predictors=column_list(x, "x"), panel=panel,
        time=time, missing=missing, alpha=alpha, options={"model": model, "y_lags": y_lags,
        "x_lags": x_lags, "cs_lags": cs_lags, "trend": trend, "max_work": max_work}), data=data)
