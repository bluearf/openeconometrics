"""Fama-MacBeth regression (community xtfmb).

For every period t a cross-sectional OLS of y on X (constant included) gives
b_t; the reported coefficient is the mean over the T periods,

    b = (1/T) sum_t b_t,   V = (1/T) * S / (T - 1),   S = sum_t (b_t - b)(b_t - b)',

i.e. standard errors sd(b_t)/sqrt(T) with t inference on T - 1 degrees of
freedom (the regression of the period estimates on a constant). With
``covariance='hac'`` the period estimates are treated as a time series:
V = T/(T-1) * HAC(b_t - b; Bartlett, lags)/T^2, Newey-West with ``lags``
(default floor(4 (T/100)^(2/9))), which equals the plain estimator at lags = 0.
Periods must be integers or datetimes; gaps between periods are respected by
the HAC kernel. Every period needs more observations than parameters and a
full-rank design. The period regressions run on grand-mean-centered slopes
(an exact reparameterization, mapped back per period), so regressors with a
large offset are not mistaken for the constant. An outcome without
cross-sectional variation, or period regressions that fit exactly, are refused
(``constant_outcome`` / ``perfect_fit``) rather than reported with
rounding-noise standard errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec, wald_test,
)
from openecon.econometrics.panel.common import center_slopes, require_residual_variation
from openecon.engines import covariance as cov
from openecon.engines import linalg
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle


def fit_xtfmb(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``xtfmb``: period-by-period OLS, then the mean and spread over periods.

    The only Python loop is over the T periods (never over observations); each
    period is one QR least-squares solve on its rows of the sorted sample.
    """
    frame = ModelFrame(spec, data)
    if spec.covariance != "hac" and "lags" in spec.options:
        raise AnalysisError("invalid_spec", "lags applies to covariance='hac' only.")
    frame.sort_panel()
    design = frame.drop_collinear(frame.design())
    y = frame.numeric(spec.outcome)
    time = frame.time_index()
    periods, index = torch.unique(time, return_inverse=True)
    count = periods.numel()
    kk = design.x.shape[1]
    if count < 2:
        raise AnalysisError("insufficient_periods", "Fama-MacBeth needs at least two periods.")
    means = center_slopes(design.x)
    order = torch.argsort(index, stable=True)
    sizes = torch.bincount(index, minlength=count)
    starts = torch.cumsum(sizes, 0) - sizes
    estimates = torch.empty((count, kk), dtype=torch.float64)
    r_squared = torch.empty(count, dtype=torch.float64)
    total_ssr = total_tss = 0.0
    for t in range(count):
        rows = order[int(starts[t]): int(starts[t] + sizes[t])]
        if len(rows) <= kk:
            raise AnalysisError("insufficient_observations",
                                f"Period {periods[t].item()} has {len(rows)} observation(s) for "
                                f"{kk} parameters; every period needs more observations.")
        try:
            fit = linalg.least_squares(design.x[rows], y[rows], drop_collinear=False)
        except KernelError as exc:
            raise AnalysisError(exc.code, f"Period {periods[t].item()}: {exc}") from exc
        estimates[t] = fit.beta
        centered = y[rows] - y[rows].mean()
        tss = float(centered @ centered)
        r_squared[t] = 1 - float(fit.ssr) / tss if tss > 0 else float("nan")
        total_ssr, total_tss = total_ssr + float(fit.ssr), total_tss + tss
    require_residual_variation(total_ssr, total_tss, float(y.square().sum()), spec.outcome,
                               "the period cross-sections")
    fitted = design.x @ estimates.mean(dim=0)
    estimates[:, 0] -= estimates[:, 1:] @ means       # period intercepts in the original units
    beta = estimates.mean(dim=0)
    deviations = estimates - beta
    if spec.covariance == "nonrobust":
        v = deviations.T @ deviations / (count * (count - 1))
        info = {"correction": "Fama-MacBeth: sd of the period estimates / sqrt(T)"}
    else:
        lags = frame.option("lags")
        lags = cov.newey_west_lags(count) if lags is None else int(lags)
        meat = kernel_call(cov.meat_hac, deviations, lags, "bartlett", time=periods)
        v = meat * (count / (count - 1)) / count ** 2
        info = {"correction": f"Newey-West over periods (bartlett, {lags} lags): T/(T-1)",
                "lags": lags, "kernel": "bartlett", "small_sample_correction": count / (count - 1)}
    info.update({"covariance": spec.covariance, "df_inference": count - 1, "periods": count})
    valid = r_squared[torch.isfinite(r_squared)]
    metrics = {"r_squared": float(valid.mean()) if valid.numel() else None, "n_periods": count,
               "n_groups": frame.codes(spec.panel)[1], "obs_per_period_min": int(sizes.min()),
               "obs_per_period_avg": frame.n / count, "obs_per_period_max": int(sizes.max())}
    tests = {"model": wald_test(beta, v, range(1, kk), df_resid=count - 1,
                                label="F test of the slopes")}
    return build_result(
        frame, terms=design.terms, params=beta, covariance=v, title="Fama-MacBeth regression",
        use_t=True, df_inference=count - 1, df_resid=count - 1, metrics=metrics,
        fitted=fitted, solver="qr_by_period", inference=info, tests=tests,
        extra={"period_estimates_sd": deviations.std(dim=0).tolist()},
        categories=design.categories,
    )


def xtfmb(*, data: Any, y: str, x: Sequence[str], panel: str, time: str,
          covariance: str | None = None, lags: int | None = None,
          categorical: Sequence[str] | None = None, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Fama-MacBeth two-step regression (community ``xtfmb``).

    Model and estimator. For every period t the cross-section is regressed
    on ``x`` and a constant by OLS, giving b_t; the reported coefficients are
    b = (1/T) sum_t b_t.

    Covariance. ``'nonrobust'`` (default) is V = S / (T (T - 1)) with
    S = sum_t (b_t - b)(b_t - b)', i.e. standard errors sd(b_t)/sqrt(T).
    ``'hac'`` treats the b_t as a time series: Newey-West with the Bartlett
    kernel and ``lags`` (default floor(4 (T/100)^(2/9))), times T/(T - 1); it
    equals the default at ``lags=0`` and respects gaps between periods.
    Inference is Student t with T - 1 degrees of freedom; ``tests['model']``
    is the F(k, T - 1) test of the slopes.

    Parameters: ``data`` a DataFrame (or columns/records); ``y`` the outcome;
    ``x`` the list of regressors; ``panel`` the panel identifier and ``time``
    the integer (or datetime) period, both required; ``categorical``
    regressors to treatment-code; ``missing`` ``'raise'`` or ``'drop'``;
    ``alpha`` the test size. Weights and cluster columns are not accepted.

    Result: ``metrics`` hold ``r_squared`` (the average period R-squared),
    ``n_periods``, ``n_groups`` and the observations per period;
    ``extra['period_estimates_sd']`` the standard deviations of the b_t.

    Errors: every period needs more observations than parameters
    (``insufficient_observations``) and a full-rank design
    (``singular_design``, naming the period); at least two periods
    (``insufficient_periods``); an outcome without cross-sectional variation
    or period regressions that fit exactly raise ``constant_outcome`` /
    ``perfect_fit``; ``lags`` without ``covariance='hac'`` is ``invalid_spec``.

    Example::

        oe.xtfmb(data=df, y="ret", x=["beta", "size"], panel="firm", time="month",
                 covariance="hac", lags=3).summary()
    """
    from openecon.analysis import fit

    spec = make_spec("xtfmb", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
                     covariance=covariance, categorical=column_list(categorical, "categorical"),
                     missing=missing, alpha=alpha, options={"lags": lags})
    return fit(spec, data=data)
