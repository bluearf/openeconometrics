"""Horizon regressions with full HAC or unit-cluster covariance of responses."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import ls, regular_frame
from openecon.econometrics.core import build_result, column_list, kernel_call, make_spec
from openecon.engines import covariance as cov
from openecon.engines.linalg import cholesky_inverse, least_squares


def fit_projection(spec, data):
    frame, units = regular_frame(spec, data)
    if len(spec.predictors) != 1:
        raise AnalysisError(
            "invalid_spec",
            "Specify exactly one shock column in x, and other exogenous columns in controls.",
        )
    p, horizon = frame.option("lags"), frame.option("horizons")
    cumulative = frame.option("cumulative")
    start = max(p, int(cumulative))
    bandwidth = frame.option("bandwidth")
    bandwidth = horizon if bandwidth is None else bandwidth
    if spec.panel is None and bandwidth < horizon:
        raise AnalysisError(
            "invalid_bandwidth", "Overlap-aware LP inference requires bandwidth >= horizons."
        )
    if spec.panel is not None and frame.option("bandwidth") is not None:
        raise AnalysisError(
            "invalid_option", "Panel LP uses unit clusters; bandwidth does not apply."
        )
    y = frame.numeric(spec.outcome)
    shock = frame.numeric(spec.predictors[0])
    controls = frame.matrix(frame.role("controls"))
    instruments = frame.matrix(frame.role("instruments")) if spec.estimator == "lpiv" else None
    width = (
        2 + controls.shape[1] + p * (2 + controls.shape[1]) + (len(units) - 1 if spec.panel else 0)
    )
    frame.workspace_plan(
        "local projection joint horizons",
        {"design_and_influence": 48 * frame.n * (width + horizon + 1) + 64 * width * width},
    )
    influence = torch.zeros((frame.n, horizon + 1), dtype=torch.float64)
    estimates, equations, all_rows = [], [], []
    counts = []
    for h in range(horizon + 1):
        pieces, outcomes, rows_used, zp = [], [], [], []
        for unit, rows in enumerate(units):
            t = torch.arange(start, len(rows) - h)
            if not len(t):
                raise AnalysisError(
                    "insufficient_observations",
                    "Every unit must support every requested horizon and lag.",
                )
            current = rows[t]
            pieces_u = [
                torch.ones((len(t), 1), dtype=torch.float64),
                shock[current, None],
                controls[current],
            ]
            for lag in range(1, p + 1):
                previous = rows[t - lag]
                pieces_u += [y[previous, None], shock[previous, None], controls[previous]]
            if spec.panel:
                dummy = torch.zeros((len(t), len(units) - 1), dtype=torch.float64)
                if unit:
                    dummy[:, unit - 1] = 1
                pieces_u.append(dummy)
            xu = torch.cat(pieces_u, dim=1)
            pieces.append(xu)
            outcomes.append(y[rows[t + h]] - y[rows[t - 1]] if cumulative else y[rows[t + h]])
            rows_used.append(current)
            if instruments is not None:
                zp.append(torch.cat((xu[:, :1], instruments[current], xu[:, 2:]), dim=1))
        design, outcome, rows = torch.cat(pieces), torch.cat(outcomes), torch.cat(rows_used)
        if instruments is None:
            fit = ls(design, outcome)
            beta, bread, residual, score_design = fit.beta, fit.xtx_inv, fit.resid, design
        else:
            z = torch.cat(zp)
            # QR checks rank of instruments before projection; no pseudo-IV fallback.
            kernel_call(least_squares, z, shock[rows], drop_collinear=False)
            q, _ = torch.linalg.qr(z, mode="reduced")
            projected = q @ (q.T @ design)
            ls(projected, outcome)
            bread = kernel_call(
                cholesky_inverse,
                projected.T @ projected,
                what="LP-IV projected regressor identification",
            )
            beta = bread @ projected.T @ outcome
            residual, score_design = outcome - design @ beta, projected
        estimates.append(beta[1])
        influence[rows, h] = (score_design @ bread[:, 1]) * residual
        count = len(rows)
        counts.append(count)
        if spec.panel:
            groups = len(units)
            if groups < 2:
                raise AnalysisError(
                    "insufficient_clusters", "Panel LP needs at least two unit clusters."
                )
            influence[:, h] *= (
                groups / (groups - 1) * (count - 1) / (count - design.shape[1])
            ) ** 0.5
        else:
            influence[:, h] *= (count / (count - design.shape[1])) ** 0.5
        equations.append(
            {
                "horizon": h,
                "nobs": count,
                "df_resid": count - design.shape[1],
                "coefficients": beta.tolist(),
                "sample_positions": [frame.positions[int(i)] for i in rows],
                "design_width": design.shape[1],
                "instrument_columns": frame.role("instruments") if instruments is not None else [],
                "instrument_rank": z.shape[1] if instruments is not None else None,
                "projected_regressor_rank": design.shape[1] if instruments is not None else None,
            }
        )
        all_rows.append(rows)
    if spec.panel:
        group, count = frame.codes(spec.panel)
        v = kernel_call(cov.meat_cluster, influence, group, count)
        use_t, df = True, count - 1
    else:
        v = kernel_call(cov.meat_hac, influence, bandwidth, "bartlett", time=frame.time_index())
        use_t, df = False, None
    keep = torch.zeros(frame.n, dtype=torch.bool)
    keep[torch.cat(all_rows)] = True
    frame.restrict(
        keep,
        "Excluded lag boundaries within units; each horizon records its further future-outcome boundary.",
    )
    terms = [f"h={h}:{spec.predictors[0]}" for h in range(horizon + 1)]
    return build_result(
        frame,
        terms=terms,
        params=torch.stack(estimates),
        covariance=v,
        use_t=use_t,
        df_inference=df,
        solver="float64_horizon_qr_iv_projection",
        metrics={"horizons": horizon, "n_groups": len(units)},
        inference={
            "correction": "CR1 unit-cluster, full horizon covariance"
            if spec.panel
            else "Bartlett HAC with HC1; full horizon covariance",
            "lags": bandwidth if not spec.panel else None,
        },
        extra={
            "equations": equations,
            "lags": p,
            "cumulative": cumulative,
            "shock_scale": "one original unit; no implicit standardization",
            "response": "y(t+h)-y(t-1)" if cumulative else "y(t+h)",
            "identification": "instrument relevance and exclusion"
            if instruments is not None
            else "conditional shock exogeneity required for causal interpretation",
            "unit_effects": "fixed intercepts" if spec.panel else None,
            "nobs_by_horizon": counts,
            "calendar": "integer periods, no gaps compressed",
        },
    )


def _fit(
    name,
    *,
    data,
    y,
    x,
    time,
    controls=None,
    instruments=None,
    panel=None,
    horizons=8,
    lags=1,
    bandwidth=None,
    cumulative=False,
    missing="raise",
    alpha=0.05,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=y,
            predictors=column_list(x, "x"),
            panel=panel,
            time=time,
            covariance="cluster" if panel else "hac",
            cluster=panel,
            columns={
                "controls": column_list(controls, "controls"),
                "instruments": column_list(instruments, "instruments"),
            },
            options={
                "horizons": horizons,
                "lags": lags,
                "bandwidth": bandwidth,
                "cumulative": cumulative,
            },
            missing=missing,
            alpha=alpha,
        ),
        data=data,
    )


def lp(
    *,
    data,
    y,
    x,
    time,
    instruments=None,
    controls=None,
    horizons=8,
    lags=1,
    bandwidth=None,
    cumulative=False,
    missing="raise",
    alpha=0.05,
):
    """Local projections with overlap-aware HAC."""
    return _fit(
        "lp",
        data=data,
        y=y,
        x=x,
        time=time,
        controls=controls,
        instruments=instruments,
        horizons=horizons,
        lags=lags,
        bandwidth=bandwidth,
        cumulative=cumulative,
        missing=missing,
        alpha=alpha,
    )


def lpiv(
    *,
    data,
    y,
    x,
    time,
    instruments,
    controls=None,
    horizons=8,
    lags=1,
    bandwidth=None,
    cumulative=False,
    missing="raise",
    alpha=0.05,
):
    """2SLS local projections with explicit instruments and HAC."""
    return _fit(
        "lpiv",
        data=data,
        y=y,
        x=x,
        time=time,
        controls=controls,
        instruments=instruments,
        horizons=horizons,
        lags=lags,
        bandwidth=bandwidth,
        cumulative=cumulative,
        missing=missing,
        alpha=alpha,
    )


def panel_lp(
    *,
    data,
    y,
    x,
    time,
    panel,
    instruments=None,
    controls=None,
    horizons=8,
    lags=1,
    cumulative=False,
    missing="raise",
    alpha=0.05,
):
    """Unit fixed-effect projections with full unit-cluster horizon covariance."""
    return _fit(
        "panel_lp",
        data=data,
        y=y,
        x=x,
        time=time,
        controls=controls,
        instruments=instruments,
        panel=panel,
        horizons=horizons,
        lags=lags,
        cumulative=cumulative,
        missing=missing,
        alpha=alpha,
    )
