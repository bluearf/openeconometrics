"""Float64 Torch cointegrating regressions (Phillips-Hansen, Stock-Watson, Park).

The methods assume an I(1) regressor system with an I(0) cointegrating error;
fitting them is not a cointegration test. Every reported covariance is retained.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import deterministic, ls, regular_frame
from openecon.econometrics.core import build_result, column_list, kernel_call, make_spec
from openecon.engines import covariance as cov
from openecon.engines.linalg import cholesky_inverse, least_squares


def _lrv(values, bandwidth, kernel):
    """Uncentered Gamma_0, Delta=sum_{j>=0} w_j Gamma_j and Omega."""
    n = len(values)
    if bandwidth >= n:
        raise AnalysisError(
            "invalid_bandwidth", "bandwidth must be smaller than the effective sample."
        )
    short = values.T @ values / n
    one = short.clone()
    # Quadratic spectral has infinite support; finite data have n-1 lag products.
    reach = (n - 1 if bandwidth else 0) if kernel == "quadratic_spectral" else bandwidth
    for lag in range(1, reach + 1):
        if kernel == "bartlett":
            weight = 1 - lag / (bandwidth + 1)
        elif kernel == "parzen":
            a = lag / (bandwidth + 1)
            weight = 1 - 6 * a * a + 6 * a**3 if a <= 0.5 else 2 * (1 - a) ** 3
        else:
            a = 6 * math.pi * lag / (5 * bandwidth)
            weight = 3 * (math.sin(a) / a - math.cos(a)) / (a * a)
        # Gamma_j = E[eta_t eta_{t-j}']; direction matters for the bias.
        one += weight * (values[lag:].T @ values[:-lag]) / n
    return short, one, one + one.T - short


def _meat(scores, bandwidth, kernel):
    if bandwidth == 0:
        return scores.T @ scores
    # OpenEcon HAC takes the lag-index parameter L, with QS scale L+1.
    return kernel_call(
        cov.meat_hac, scores, bandwidth - 1 if kernel == "quadratic_spectral" else bandwidth, kernel
    )


def _augmented(y, x, trend, leads, lags, *, trim=None):
    n = len(y)
    start, stop = trim or (lags + 1, n - leads)
    rows = torch.arange(start, stop, dtype=torch.int64)
    d, names = deterministic(n, trend)
    dx = x[1:] - x[:-1]
    pieces = [x[rows], d[rows]]
    for shift in range(-lags, leads + 1):
        pieces.append(dx[rows + shift - 1])
    return rows, torch.cat(pieces, dim=1), names


def _series(y, x, options, method, covariance):
    n, k = x.shape
    trend, bw, kernel = (options[a] for a in ("trend", "bandwidth", "kernel"))
    d, dt = deterministic(n, trend)
    full = torch.cat((x, d), dim=1)
    if method == "dols":
        leads, lags = options["leads"], options["lags"]
        candidates = []
        if options["selection"] != "fixed":
            a, b = options["max_leads"], options["max_lags"]
            for p in range(b + 1):
                for q in range(a + 1):
                    rows, z, _ = _augmented(y, x, trend, q, p, trim=(b + 1, n - a))
                    fit = ls(z, y[rows])
                    factor = {
                        "aic": 2.0,
                        "bic": math.log(len(rows)),
                        "hqic": 2 * math.log(math.log(len(rows))),
                    }[options["selection"]]
                    candidates.append(
                        (
                            math.log(float(fit.ssr) / len(rows)) + factor * z.shape[1] / len(rows),
                            q,
                            p,
                        )
                    )
            _, leads, lags = min(candidates)
        rows, z, _ = _augmented(y, x, trend, leads, lags)
        fit = ls(z, y[rows])
        bread, resid = fit.xtx_inv, fit.resid
        factor = len(rows) / (len(rows) - z.shape[1]) if options["df_adjust"] else 1.0
        if covariance == "hac":
            v = bread @ _meat(z * resid[:, None], bw, kernel) @ bread * factor
            variance = None
        else:
            variance = float(_lrv(resid[:, None], bw, kernel)[2][0, 0]) * factor
            if variance <= 0:
                raise AnalysisError(
                    "invalid_lrv", "The residual long-run variance must be positive."
                )
            v = bread * variance
        names = dt + [f"D({j},shift={s})" for s in range(-lags, leads + 1) for j in range(k)]
        return {
            "rows": rows,
            "z": z,
            "target": y[rows],
            "bias": torch.zeros(z.shape[1], dtype=torch.float64),
            "beta": fit.beta,
            "v": v,
            "variance": variance,
            "names": names,
            "fitted": fit.fitted,
            "leads": leads,
            "lags": lags,
            "selection_candidates": len(candidates),
        }
    if options["selection"] != "fixed" or options["leads"] or options["lags"]:
        raise AnalysisError(
            "invalid_option", "Lead/lag augmentation and selection apply only to DOLS."
        )
    initial = ls(full, y)
    # Remove the declared deterministic components of x before differencing.
    xd = x
    if d.shape[1]:
        xd = x - d @ kernel_call(least_squares, d, x, drop_collinear=False).beta
    eta = torch.cat((initial.resid[1:, None], xd[1:] - xd[:-1]), dim=1)
    short, delta, omega = _lrv(eta, bw, kernel)
    xx_inv = kernel_call(cholesky_inverse, omega[1:, 1:], what="regressor long-run covariance")
    correction = omega[0, 1:] @ xx_inv
    variance = float(omega[0, 0] - correction @ omega[1:, 0])
    if variance <= 0:
        raise AnalysisError(
            "invalid_lrv", "The conditional long-run error variance must be positive."
        )
    rows = torch.arange(1, n)
    bias = torch.zeros(full.shape[1], dtype=torch.float64)
    if method == "fmols":
        z = full[rows]
        target = y[rows] - eta[:, 1:] @ correction
        bias[:k] = delta[0, 1:] - correction @ delta[1:, 1:]
        fit = ls(z, target)
        beta = fit.beta - len(rows) * (fit.xtx_inv @ bias)
    else:
        short_inv = kernel_call(cholesky_inverse, short, what="short-run innovation covariance")
        transform = short_inv @ delta[:, 1:]
        z = torch.cat((x[rows] - eta @ transform, d[rows]), dim=1)
        shift = transform @ initial.beta[:k]
        shift[1:] += correction
        target = y[rows] - eta @ shift
        fit = ls(z, target)
        beta = fit.beta
    factor = len(rows) / (len(rows) - z.shape[1]) if options["df_adjust"] else 1.0
    return {
        "rows": rows,
        "z": z,
        "target": target,
        "bias": bias,
        "beta": beta,
        "v": fit.xtx_inv * variance * factor,
        "variance": variance * factor,
        "names": dt,
        "fitted": full[rows] @ beta,
        "omega": omega.tolist(),
        "one_sided": delta.tolist(),
    }


def fit_longrun(spec, data):
    frame, units = regular_frame(spec, data)
    y, x = frame.numeric(spec.outcome), frame.matrix(spec.predictors)
    options = {o.name: frame.option(o.name) for o in frame.info.options}
    method = spec.estimator.removeprefix("panel_")
    pieces = [_series(y[r], x[r], options, method, spec.covariance) for r in units]
    all_rows = torch.cat([rows[part["rows"]] for rows, part in zip(units, pieces, strict=True)])
    k = x.shape[1]
    panel = spec.panel is not None
    extra = {
        "cointegration_assumed": True,
        "deterministic": options["trend"],
        "kernel": options["kernel"],
        "bandwidth": options["bandwidth"],
        "df_adjust": options["df_adjust"],
        "units": [],
        "calendar": "integer unit periods; no compression",
        "selection": options["selection"],
    }
    labels = frame.levels(spec.panel) if panel else ["series"]
    for label, r, p in zip(labels, units, pieces, strict=True):
        extra["units"].append(
            {
                "unit": label,
                "effective_rows": [frame.positions[int(j)] for j in r[p["rows"]]],
                "coefficients": p["beta"].tolist(),
                "covariance": p["v"].tolist(),
                "leads": p.get("leads"),
                "lags": p.get("lags"),
                "selection_candidates": p.get("selection_candidates"),
                "long_run_covariance": p.get("omega"),
                "one_sided_covariance": p.get("one_sided"),
            }
        )
    use_t, df = False, None
    fitted = None
    if not panel:
        beta, v, fitted = pieces[0]["beta"], pieces[0]["v"], pieces[0]["fitted"]
        terms = [*spec.predictors, *pieces[0]["names"]]
    elif options["pooling"] == "mean_group":
        if len(units) < 2:
            raise AnalysisError(
                "insufficient_groups", "Mean-group inference needs at least two units."
            )
        estimates = torch.stack([p["beta"][:k] for p in pieces])
        beta = estimates.mean(0)
        deviations = estimates - beta
        v = deviations.T @ deviations / (len(units) * (len(units) - 1))
        terms = list(spec.predictors)
        use_t, df = True, len(units) - 1
    else:
        widths = [p["z"].shape[1] - k for p in pieces]
        width = k + sum(widths)
        frame.workspace_plan(
            "panel long-run pooled design",
            {"design_and_factors": 40 * len(all_rows) * width + 64 * width * width},
        )
        z = torch.zeros((len(all_rows), width), dtype=torch.float64)
        target, bias = (
            torch.cat([p["target"] for p in pieces]),
            torch.zeros(width, dtype=torch.float64),
        )
        terms = list(spec.predictors)
        offset, col = 0, k
        for label, p, size in zip(labels, pieces, widths, strict=True):
            stop = offset + len(p["rows"])
            z[offset:stop, :k] = p["z"][:, :k]
            z[offset:stop, col : col + size] = p["z"][:, k:]
            bias[:k] += len(p["rows"]) * p["bias"][:k]
            terms += [f"unit[{label}]:{name}" for name in p["names"]]
            offset, col = stop, col + size
        fit = ls(z, target)
        beta = fit.beta - fit.xtx_inv @ bias
        meat = torch.zeros((width, width), dtype=torch.float64)
        offset = 0
        for p in pieces:
            stop = offset + len(p["rows"])
            zi = z[offset:stop]
            if spec.covariance == "hac":
                ri = target[offset:stop] - zi @ beta
                meat += _meat(zi * ri[:, None], options["bandwidth"], options["kernel"])
            else:
                meat += p["variance"] * (zi.T @ zi)
            offset = stop
        v = fit.xtx_inv @ meat @ fit.xtx_inv
        extra["pooled_contract"] = (
            "common level slopes; separate unit deterministics and DOLS short-run augmentation; cross-unit independence"
        )
    keep = torch.zeros(frame.n, dtype=torch.bool)
    keep[all_rows] = True
    frame.restrict(keep, "Excluded lag/lead boundaries separately within each series.")
    extra["pooling"] = options.get("pooling", "single_series")
    return build_result(
        frame,
        terms=terms,
        params=beta,
        covariance=v,
        fitted=fitted,
        use_t=use_t,
        df_inference=df,
        df_resid=frame.n - len(beta),
        solver="float64_qr_and_long_run_covariance",
        extra=extra,
        inference={
            "correction": "mean-group between-unit dispersion"
            if use_t
            else "cointegrating asymptotic normal; long-run error correction"
        },
    )


def _fit(
    name,
    *,
    data,
    y,
    x,
    time=None,
    panel=None,
    covariance=None,
    missing="raise",
    alpha=0.05,
    **options,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=y,
            predictors=column_list(x, "x"),
            time=time,
            panel=panel,
            intercept=False,
            covariance=covariance,
            missing=missing,
            alpha=alpha,
            options=options,
        ),
        data=data,
    )


def fmols(
    *,
    data,
    y,
    x,
    time=None,
    trend="c",
    kernel="bartlett",
    bandwidth=4,
    df_adjust=False,
    missing="raise",
    alpha=0.05,
):
    """Fully modified OLS with explicit fixed-bandwidth long-run correction."""
    return _fit(
        "fmols",
        data=data,
        y=y,
        x=x,
        time=time,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        df_adjust=df_adjust,
        missing=missing,
        alpha=alpha,
    )


def dols(
    *,
    data,
    y,
    x,
    time=None,
    trend="c",
    kernel="bartlett",
    bandwidth=4,
    df_adjust=False,
    leads=0,
    lags=0,
    selection="fixed",
    max_leads=4,
    max_lags=4,
    covariance="nonrobust",
    missing="raise",
    alpha=0.05,
):
    """DOLS with explicit lead/lag augmentation and fixed or common-sample IC selection."""
    return _fit(
        "dols",
        data=data,
        y=y,
        x=x,
        time=time,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        df_adjust=df_adjust,
        leads=leads,
        lags=lags,
        selection=selection,
        max_leads=max_leads,
        max_lags=max_lags,
        covariance=covariance,
        missing=missing,
        alpha=alpha,
    )


def ccr(
    *,
    data,
    y,
    x,
    time=None,
    trend="c",
    kernel="bartlett",
    bandwidth=4,
    df_adjust=False,
    missing="raise",
    alpha=0.05,
):
    """Canonical cointegrating regression with retained transformation and LRV."""
    return _fit(
        "ccr",
        data=data,
        y=y,
        x=x,
        time=time,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        df_adjust=df_adjust,
        missing=missing,
        alpha=alpha,
    )


def panel_fmols(
    *,
    data,
    y,
    x,
    panel,
    time,
    pooling="pooled",
    trend="c",
    kernel="bartlett",
    bandwidth=4,
    df_adjust=False,
    missing="raise",
    alpha=0.05,
):
    """Common-slope or equal-unit mean-group panel FMOLS."""
    return _fit(
        "panel_fmols",
        data=data,
        y=y,
        x=x,
        time=time,
        panel=panel,
        pooling=pooling,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        df_adjust=df_adjust,
        missing=missing,
        alpha=alpha,
    )


def panel_dols(
    *,
    data,
    y,
    x,
    panel,
    time,
    pooling="pooled",
    trend="c",
    kernel="bartlett",
    bandwidth=4,
    df_adjust=False,
    leads=0,
    lags=0,
    selection="fixed",
    max_leads=4,
    max_lags=4,
    covariance="nonrobust",
    missing="raise",
    alpha=0.05,
):
    """Common-slope or equal-unit mean-group panel DOLS."""
    return _fit(
        "panel_dols",
        data=data,
        y=y,
        x=x,
        time=time,
        panel=panel,
        pooling=pooling,
        trend=trend,
        kernel=kernel,
        bandwidth=bandwidth,
        df_adjust=df_adjust,
        leads=leads,
        lags=lags,
        selection=selection,
        max_leads=max_leads,
        max_lags=max_lags,
        covariance=covariance,
        missing=missing,
        alpha=alpha,
    )
