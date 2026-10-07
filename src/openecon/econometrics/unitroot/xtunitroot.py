"""Panel unit-root tests: LLC, IPS, Fisher, Hadri, Breitung and Harris-Tzavalis."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.unitroot import critical
from openecon.econometrics.unitroot.common import (
    KERNELS, Panel, check_choice, check_count, check_flag, load_panel,
)
from openecon.econometrics.unitroot.panel_kernels import (
    adf_batch, adf_blocks, batch_ols, choose_lags, demean_cross_section, detrend, groups,
    long_run_variances, mackinnon_p_batch,
)
from openecon.engines.distributions import chi2_sf, normal_cdf, normal_sf, t_cdf

_TESTS = ("llc", "ips", "fisher", "hadri", "breitung", "ht", "cips", "cadf")
_CASE = {"none": "n", "constant": "c", "trend": "ct"}


def _ips_lag_cap(trend: str, total: int, limit: int) -> int:
    """Largest lag order <= limit whose moments Im, Pesaran and Shin tabulate for T = total."""
    for order in range(limit, 0, -1):
        try:
            critical.ips_moments(trend, order, total - order - 1)
        except AnalysisError:
            continue
        return order
    return 0


def _lag_batches(panel: Panel, lags: int | str, trend: str, maxlag: int | None,
                 tabulated: bool = False):
    """Batches (members, series [N_g, T_g], lag order) and the per-panel lag orders.

    ``tabulated`` limits the default maxlag of the automatic choice to the lag
    orders for which the Im-Pesaran-Shin moments exist at the panel's length.
    """
    chosen = torch.zeros(panel.n_panels, dtype=torch.int64)
    batches = []
    for members, y in groups(panel):
        total = y.shape[1]
        if isinstance(lags, str):
            limit = maxlag
            if limit is None:
                limit = max(0, min(8, int(12 * (total / 100) ** 0.25), (total - 5) // 2))
                if tabulated:
                    limit = _ips_lag_cap(trend, total, limit)
            if total - limit - 1 <= limit + 3:
                raise AnalysisError("invalid_lags", f"maxlag={limit} is too large for panels "
                                    f"with {total} observations.")
            orders = choose_lags(y, trend, lags, limit)
        else:
            orders = torch.full((y.shape[0],), lags, dtype=torch.int64)
        chosen[members] = orders
        for order in torch.unique(orders).tolist():
            pick = orders == order
            batches.append((members[pick], y[pick], int(order)))
    return batches, chosen


def _t_statistics(fit) -> Tensor:
    if bool((fit.se[:, 0] <= 0).any()) or not bool(torch.isfinite(fit.se[:, 0]).all()):
        raise AnalysisError("perfect_fit", "A panel's Dickey-Fuller regression fits exactly; "
                            "its t statistic is undefined.")
    return fit.beta[:, 0] / fit.se[:, 0]


def _llc(panel: Panel, lags, trend: str, maxlag, kernel: str, kernel_lags: int | None,
         lrv: str = "paper") -> dict:
    y = panel.cube()[:, :, 0]
    count, total = y.shape
    batches, chosen = _lag_batches(panel, lags, trend, maxlag)
    sigma_eps = torch.empty(count, dtype=torch.float64)
    sum_ev = sum_vv = sum_ee = 0.0
    observations = 0
    for members, series, order in batches:
        target, level, others = adf_blocks(series, order, trend)
        n = target.shape[1]
        if others.shape[2]:
            e = batch_ols(others, target, "The Levin-Lin-Chu auxiliary regression").resid
            v = batch_ols(others, level, "The Levin-Lin-Chu auxiliary regression").resid
        else:
            e, v = target, level
        vv = v.square().sum(dim=1)
        if bool((vv <= 0).any()):
            raise AnalysisError("collinear_regressors", "A panel's lagged level is explained "
                                "exactly by its deterministic terms and lags.")
        rho = (e * v).sum(dim=1) / vv
        variance = (e - rho[:, None] * v).square().sum(dim=1) / n
        if not bool((variance * n > 1e-22 * target.square().sum(dim=1)).all()):
            raise AnalysisError("perfect_fit", "A panel's Dickey-Fuller regression fits "
                                "exactly (the panel is constant or deterministic); the "
                                "Levin-Lin-Chu test is undefined.")
        scale = variance.sqrt()
        sigma_eps[members] = scale
        e, v = e / scale[:, None], v / scale[:, None]
        sum_ev += float((e * v).sum())
        sum_vv += float(v.square().sum())
        sum_ee += float(e.square().sum())
        observations += e.numel()
    delta = sum_ev / sum_vv
    sigma2 = (sum_ee - sum_ev * sum_ev / sum_vv) / observations
    std = math.sqrt(sigma2 / sum_vv)
    unadjusted = delta / std
    if kernel_lags is None:
        kernel_lags = min(total - 2, int(3.21 * total ** (1 / 3)))
    # Step 2 of LLC: Delta y as it is without deterministic terms, minus its panel mean
    # when panel means or trends are in the model (a linear trend in y is a mean of
    # Delta y). lrv="null" keeps Delta y as it is with panel means: under H0 there is
    # no drift then.
    demeaned = trend == "trend" or (trend == "constant" and lrv == "paper")
    differences = detrend(y[:, 1:] - y[:, :-1], "constant" if demeaned else "none")
    ratio = long_run_variances(differences, kernel_lags, kernel).clamp_min(0).sqrt() / sigma_eps
    s_n = float(ratio.mean())
    t_tilde = observations / count
    mean, deviation, note = critical.llc_adjustment(trend, t_tilde)
    adjusted = (unadjusted - observations * s_n * std * mean / sigma2) / deviation
    p_value = normal_cdf(adjusted)
    notes = ["Adjustment terms: Levin, Lin and Chu (2002, Table 2), interpolated at the "
             f"average number of observations per panel ({t_tilde:g})."]
    if note:
        notes.append(note)
    if trend == "constant" and lrv == "paper" and kernel_lags and t_tilde < 250:
        notes.append("With panel means the long-run variance of the demeaned differences is "
                     f"biased downwards by about (kernel_lags + 1)/T = {kernel_lags + 1}/"
                     f"{total - 1}, which shifts t* to the left under the null in short "
                     "panels; lrv='null' (no demeaning) or fewer kernel_lags avoid it.")
    return {
        "rows": [["unadjusted_t", unadjusted, None, None], ["adjusted_t", adjusted, None,
                                                            p_value]],
        "statistic": adjusted, "p_value": p_value, "distribution": "normal",
        "lags_by_panel": chosen, "extra": {
            "unadjusted_t": unadjusted, "coefficient": delta, "mu_star": mean,
            "sigma_star": deviation, "s_n": s_n, "kernel": kernel, "kernel_lags": kernel_lags,
            "t_tilde": t_tilde, "lrv": lrv},
        "label": "Levin-Lin-Chu adjusted t* (H0: all panels contain a unit root)",
        "h0": "Panels contain unit roots", "ha": "Panels are stationary",
        "notes": notes,
    }


def _ips(panel: Panel, lags, trend: str, maxlag) -> dict:
    batches, chosen = _lag_batches(panel, lags, trend, maxlag, tabulated=True)
    count = panel.n_panels
    statistics = torch.empty(count, dtype=torch.float64)
    means = torch.empty(count, dtype=torch.float64)
    variances = torch.empty(count, dtype=torch.float64)
    for members, series, order in batches:
        fit = adf_batch(series, order, trend)
        statistics[members] = _t_statistics(fit)
        mean, variance = critical.ips_moments(trend, order, fit.nobs)
        means[members], variances[members] = mean, variance
    t_bar = float(statistics.mean())
    w = math.sqrt(count) * (t_bar - float(means.mean())) / math.sqrt(float(variances.mean()))
    p_value = normal_cdf(w)
    return {
        "rows": [["t_bar", t_bar, None, None], ["w_t_bar", w, None, p_value]],
        "statistic": w, "p_value": p_value, "distribution": "normal", "lags_by_panel": chosen,
        "extra": {"t_bar": t_bar, "mean_adjustment": float(means.mean()),
                  "variance_adjustment": float(variances.mean())},
        "label": "Im-Pesaran-Shin W-t-bar (H0: all panels contain a unit root)",
        "h0": "All panels contain unit roots", "ha": "Some panels are stationary",
        "notes": ["E[t] and Var[t]: Im, Pesaran and Shin (2003, Table 3), interpolated in the "
                  "number of observations of each panel's regression."],
    }


def _pp_statistics(series: Tensor, lags: int, trend: str) -> Tensor:
    """Phillips-Perron Z(t) of every row of series [N, T] (Bartlett, ``lags`` truncation)."""
    count, total = series.shape
    n = total - 1
    columns = [series[:, :-1]]
    if trend in {"constant", "trend"}:
        if trend == "trend":
            columns.append(torch.arange(1, n + 1, dtype=torch.float64).expand(count, n))
        columns.append(torch.ones((count, n), dtype=torch.float64))
    fit = batch_ols(torch.stack(columns, dim=2), series[:, 1:], "The Phillips-Perron regression")
    rho, sigma = fit.beta[:, 0], fit.se[:, 0]
    gamma0 = fit.ssr / n
    lambda2 = long_run_variances(fit.resid, lags, "bartlett")
    s2 = fit.ssr / (n - fit.k)
    if bool((lambda2 <= 0).any()) or bool((sigma <= 0).any()):
        raise AnalysisError("perfect_fit", "A panel's Phillips-Perron regression is degenerate; "
                            "its statistic is undefined.")
    # Ratios of variances only (no product of two variances: see series.pperron).
    return (gamma0 / lambda2).sqrt() * (rho - 1) / sigma \
        - 0.5 * ((lambda2 - gamma0) / s2) * n * sigma * (s2 / lambda2).sqrt()


def _fisher(panel: Panel, lags, trend: str, maxlag, method: str) -> dict:
    count = panel.n_panels
    statistics = torch.empty(count, dtype=torch.float64)
    if method == "dfuller":
        batches, chosen = _lag_batches(panel, lags, trend, maxlag)
        for members, series, order in batches:
            statistics[members] = _t_statistics(adf_batch(series, order, trend))
    else:
        if isinstance(lags, str):
            raise AnalysisError("invalid_lags", "With method='pperron' lags is the Newey-West "
                                "truncation and must be an integer.")
        chosen = torch.full((count,), lags, dtype=torch.int64)
        for members, series in groups(panel):
            if lags >= series.shape[1] - 1:
                raise AnalysisError("invalid_lags", "lags must be smaller than the number of "
                                    "observations of every panel.")
            statistics[members] = _pp_statistics(series, lags, trend)
    p = mackinnon_p_batch(statistics, _CASE[trend])
    notes = ["Panel p-values: MacKinnon (1994) approximation for each panel's "
             f"{'Dickey-Fuller' if method == 'dfuller' else 'Phillips-Perron'} statistic."]
    tiny = 1e-300
    clipped = p.clamp(tiny, 1 - 1e-16)
    if bool((clipped != p).any()):
        notes.append("Panel p-values of exactly 0 or 1 were moved inside (0, 1) before "
                     "combining.")
    log_p = torch.log(clipped)
    chi2 = float(-2 * log_p.sum())
    z = float(torch.special.ndtri(clipped).sum()) / math.sqrt(count)
    logit = float((log_p - torch.log1p(-clipped)).sum())
    k = 3 * (5 * count + 4) / (math.pi ** 2 * count * (5 * count + 2))
    l_star = math.sqrt(k) * logit
    modified = float(-(log_p + 1).sum()) / math.sqrt(count)
    df_chi2, df_t = 2 * count, 5 * count + 4
    rows = [["P", chi2, df_chi2, chi2_sf(chi2, df_chi2)],
            ["Z", z, None, normal_cdf(z)],
            ["L*", l_star, df_t, t_cdf(l_star, df_t)],
            ["Pm", modified, None, normal_sf(modified)]]
    return {
        "rows": rows, "statistic": chi2, "p_value": rows[0][3], "distribution": "chi2",
        "lags_by_panel": chosen,
        "extra": {"df": df_chi2, "inverse_normal_z": z, "inverse_logit_l": l_star,
                  "modified_inverse_chi2": modified, "method": method},
        "label": "Fisher-type inverse chi-squared P (H0: all panels contain a unit root)",
        "h0": "All panels contain unit roots", "ha": "At least one panel is stationary",
        "notes": notes,
    }


def _hadri(panel: Panel, trend: str, kernel: str, kernel_lags: int | None, robust: bool) -> dict:
    y = panel.cube()[:, :, 0]
    count, total = y.shape
    resid = detrend(y, trend)
    partial = resid.cumsum(dim=1).square().sum(dim=1) / total ** 2
    terms = 1 if trend == "constant" else 2
    if kernel_lags is None:
        variances = resid.square().sum(dim=1) / (total - terms)
    else:
        variances = long_run_variances(resid, kernel_lags, kernel)       # divisor T
    if bool((resid.square().sum(dim=1) <= 1e-22 * y.square().sum(dim=1)).any()) \
            or bool((variances <= 0).any()):
        raise AnalysisError("constant_series", "A panel has no variation around its "
                            f"{'trend' if trend == 'trend' else 'mean'}; the Hadri test is "
                            "undefined.")
    if robust:
        lm = float((partial / variances).mean())
    else:
        lm = float(partial.mean()) / float(variances.mean())
    mean, variance = (1 / 6, 1 / 45) if trend == "constant" else (1 / 15, 11 / 6300)
    z = math.sqrt(count) * (lm - mean) / math.sqrt(variance)
    p_value = normal_sf(z)
    return {
        "rows": [["lm", lm, None, None], ["z", z, None, p_value]],
        "statistic": z, "p_value": p_value, "distribution": "normal", "lags_by_panel": None,
        "extra": {"lm": lm, "robust": robust, "kernel": None if kernel_lags is None else kernel,
                  "kernel_lags": kernel_lags},
        "label": "Hadri LM z (H0: all panels are stationary)",
        "h0": "All panels are stationary", "ha": "Some panels contain unit roots",
        "notes": ["Moments of the LM statistic: Hadri (2000): 1/6 and 1/45 with panel means, "
                  "1/15 and 11/6300 with panel trends."],
    }


def _breitung(panel: Panel, lags, trend: str, maxlag) -> dict:
    """Breitung (2000) lambda as in Stata's Methods and formulas (nonrobust version)."""
    batches, chosen = _lag_batches(panel, lags, trend, maxlag)
    numerator = denominator = 0.0
    for _, series, order in batches:
        count, total = series.shape
        dy = series[:, 1:] - series[:, :-1]
        n = total - 1 - order                              # usable differences per panel
        if n < 3:
            raise AnalysisError("insufficient_observations", "The Breitung test needs at "
                                "least 3 usable differences per panel.")
        d = dy[:, order:]                                  # Delta y_t, t = p+2..T
        lagged = [dy[:, order - j:total - 1 - j] for j in range(1, order + 1)]
        if trend == "trend":
            # One prewhitening regression (with a constant); its lag coefficients filter
            # both the differences and the levels.
            level = series[:, order:]                      # y_{t-1}, plus the last level
            if order:
                design = torch.stack([*lagged, torch.ones((count, n), dtype=torch.float64)],
                                     dim=2)
                gamma = batch_ols(design, d, "The Breitung prewhitening regression").beta
                d, level = d.clone(), level.clone()
                for j in range(1, order + 1):
                    d -= gamma[:, j - 1:j] * lagged[j - 1]
                    level -= gamma[:, j - 1:j] * series[:, order - j:total - j]
            variance = (d - d.mean(dim=1, keepdim=True)).square().sum(dim=1) / (n - 1)
            index = torch.arange(n - 1, dtype=torch.float64)            # s - 1, s = 1..n-1
            ahead = n - 1 - index                                       # n - s
            tail = d.flip(1).cumsum(dim=1).flip(1)[:, 1:]               # sum of d[s+1..n]
            y_star = (ahead / (ahead + 1)).sqrt() * (d[:, :n - 1] - tail / ahead)
            x_star = level[:, :n - 1] - level[:, :1] - index * d.mean(dim=1, keepdim=True)
        else:
            # y_{t-1}, measured from the first lagged level of the sample with panel means.
            level = series[:, order:total - 1]
            if trend == "constant":
                level = level - level[:, :1]
            if order:
                design = torch.stack(lagged, dim=2)
                d = batch_ols(design, d, "The Breitung prewhitening regression").resid
                level = batch_ols(design, level, "The Breitung prewhitening regression").resid
            variance = d.square().sum(dim=1) / (n - 1)
            y_star, x_star = d, level
        if bool((variance <= 0).any()):
            raise AnalysisError("constant_series", "A panel's differences do not vary; the "
                                "Breitung test is undefined.")
        weight = (1.0 / variance)[:, None]
        numerator += float((weight * x_star * y_star).sum())
        denominator += float((weight * x_star.square()).sum())
    if not denominator > 0.0:
        raise AnalysisError("constant_series", "The transformed levels do not vary; the "
                            "Breitung test is undefined.")
    statistic = numerator / math.sqrt(denominator)
    p_value = normal_cdf(statistic)
    return {
        "rows": [["lambda", statistic, None, p_value]],
        "statistic": statistic, "p_value": p_value, "distribution": "normal",
        "lags_by_panel": chosen, "extra": {"coefficient": numerator / denominator},
        "label": "Breitung lambda (H0: all panels contain a unit root)",
        "h0": "Panels contain unit roots", "ha": "Panels are stationary",
        "notes": ["lambda = sum(x* y* / s_i^2) / sqrt(sum(x*^2 / s_i^2)) on the transformed "
                  "differences y* and levels x* (forward orthogonal deviations with trends); "
                  "s_i^2 is each panel's variance of the (prewhitened) differences."],
    }


def _ht(panel: Panel, trend: str, altt: bool) -> dict:
    y = panel.cube()[:, :, 0]
    count, total = y.shape
    # T of the moment formulas: the number of periods (Stata's default) or, with altt,
    # the number of observations per panel in the regression (Harris and Tzavalis).
    periods = total - 1 if altt else total
    current, lagged = detrend(y[:, 1:], trend), detrend(y[:, :-1], trend)
    denominator = float(lagged.square().sum())
    varies = lagged.square().sum(dim=1) > 1e-22 * y.square().sum(dim=1)
    if not bool(varies.all()) or bool((y[:, 1:] == y[:, :-1]).all(dim=1).any()):
        raise AnalysisError("constant_series", "A panel does not vary around its "
                            "deterministic terms (it is constant or an exact trend); the "
                            "Harris-Tzavalis test is undefined.")
    rho = float((lagged * current).sum()) / denominator
    t = float(periods)
    if trend == "none":
        mean, variance = 1.0, 2 / (t * (t - 1))
    elif trend == "constant":
        mean = 1 - 3 / (t + 1)
        variance = 3 * (17 * t * t - 20 * t + 17) / (5 * (t - 1) * (t + 1) ** 3)
    else:
        if total < 4:
            raise AnalysisError("insufficient_observations", "The Harris-Tzavalis test with "
                                "trends needs at least 4 periods per panel.")
        mean = 1 - 15 / (2 * (t + 2))
        variance = 15 * (193 * t * t - 728 * t + 1147) / (112 * (t + 2) ** 3 * (t - 2))
    z = math.sqrt(count) * (rho - mean) / math.sqrt(variance)
    p_value = normal_cdf(z)
    notes = ["Mean and variance of the pooled autoregressive coefficient under the null: "
             f"Harris and Tzavalis (1999), evaluated at T = {periods}."]
    if not altt and trend != "none":
        notes.append("T is the number of periods, as in Stata's default. Harris and Tzavalis "
                     "define T as the observations per panel in the regression (periods - 1); "
                     "altt=True uses it and is correctly sized in short panels, where the "
                     "default over-rejects.")
    return {
        "rows": [["rho", rho, None, None], ["z", z, None, p_value]],
        "statistic": z, "p_value": p_value, "distribution": "normal", "lags_by_panel": None,
        "extra": {"rho": rho, "rho_mean": mean, "rho_variance": variance, "altt": altt,
                  "moment_periods": periods},
        "label": "Harris-Tzavalis z (H0: all panels contain a unit root)",
        "h0": "Panels contain unit roots", "ha": "Panels are stationary",
        "notes": notes,
    }


def xtunitroot(data: Any, y: str, panel: str, time: str, *, test: str = "llc",
               lags: int | str = 0, trend: str = "constant", demean: bool = False,
               kernel: str = "bartlett", kernel_lags: int | None = None,
               maxlag: int | None = None, method: str = "dfuller", robust: bool = False,
               lrv: str = "paper", altt: bool = False):
    """Panel unit-root tests (Stata's ``xtunitroot``; EViews' panel unit root tests).

    ``test='cips'`` dispatches to ``oe.xtcips`` for Pesaran's cross-sectionally
    augmented average/truncated-average tests; ``test='cadf'`` returns unit
    statistics. These require complete common-date panels and use nonstandard
    published quantiles. Use ``oe.xtcips`` directly for inference/truncation
    and fitting-workspace controls; first-generation-specific options are rejected.
    For these two tests, automatic lags compare 0..maxlag and the default
    maxlag is min(8, floor((T_levels-d-5)/3)), with d deterministic columns.

    The data are a long panel: one row per unit and period, no gaps inside a
    unit. ``trend`` sets the deterministic terms of every panel's regression:
    ``"none"`` (Stata's ``noconstant``), ``"constant"`` (panel means, the
    default) or ``"trend"`` (panel means and linear trends). ``lags`` is the
    number of lagged differences of the panels' Dickey-Fuller regressions: an
    integer for all panels, or ``"aic"`` / ``"bic"`` / ``"hqic"`` to choose it
    panel by panel among 1..maxlag for the first-generation tests below (as
    Stata's ``lags(aic #)``; default maxlag ``min(8, int(12 (T/100)^(1/4)))``).
    The default is ``lags=0`` for every test (Stata's ``llc`` defaults to one
    lag). ``demean=True`` first subtracts the cross-sectional mean of each
    period (Stata's ``demean``), which mitigates cross-sectional dependence.

    Tests (``test=``)
    -----------------
    ``"llc"`` -- Levin, Lin and Chu (2002); balanced panels. H0: every panel has
        a unit root; Ha: every panel is stationary with a common autoregressive
        parameter. Step 1: for each panel the residuals ``e`` (of Delta y_t on
        the lagged differences and deterministic terms) and ``v`` (of y_{t-1}
        on the same) are divided by ``s_i``, the standard deviation of
        ``e - d_i v`` with divisor T - p_i - 1. Step 2: ``S_N`` is the average
        over panels of (long-run standard deviation of Delta y) / ``s_i``; the
        long-run variance uses the ``kernel`` with ``kernel_lags`` lags
        (default ``int(3.21 T^(1/3))``, T periods) and divisor T - 1. Step 3:
        the pooled regression of ``e`` on ``v`` gives ``delta``, its standard
        error (error variance with divisor N T~) and ``t_delta``. Reported:

            t* = (t_delta - N T~ S_N se(delta) mu* / sigma_e^2) / sigma*,

        with ``mu*``, ``sigma*`` interpolated linearly in LLC's Table 2 at the
        average number of observations per panel T~ = T - mean(p_i) - 1.
        ``t*`` ~ N(0, 1), lower tail. ``lrv`` says what is removed from Delta y
        before the long-run variance is estimated. ``"paper"`` (default) is the
        procedure printed in LLC (2002) and used by Stata: nothing without
        deterministic terms, the panel mean of Delta y with panel means and
        with panel trends. ``"null"`` removes only what the null hypothesis
        implies: with panel means Delta y is used as it is (no drift under
        H0). The adjustment terms of Table 2 are centred for the ``"null"``
        estimator; with panel means and short panels ``"paper"`` shifts ``t*``
        to the left under H0 (see the docs page for the simulation).
    ``"ips"`` -- Im, Pesaran and Shin (2003); unbalanced panels allowed;
        ``trend`` must be ``"constant"`` or ``"trend"``. H0: every panel has a
        unit root; Ha: some panels are stationary. ``t-bar`` is the average of
        the panels' Dickey-Fuller t statistics and

            W = sqrt(N) (t-bar - mean_i E[t_i]) / sqrt(mean_i Var[t_i])  ~  N(0, 1),

        lower tail, with the moments of IPS's Table 3 for each panel's lag
        order and number of observations (lags <= 8), interpolated linearly.
        With ``lags=0`` this is IPS's Z-t-bar; Stata's default output without
        lags (t-tilde-bar and its Z) is not reproduced.
    ``"fisher"`` -- Maddala and Wu (1999), Choi (2001); unbalanced panels
        allowed. Each panel's Dickey-Fuller (``method="dfuller"``) or
        Phillips-Perron (``method="pperron"``; ``lags`` is then the Newey-West
        truncation) statistic is converted to a MacKinnon (1994) p-value p_i,
        and the p-values are combined: inverse chi-squared
        ``P = -2 sum ln p_i`` ~ chi2(2N); inverse normal
        ``Z = sum Phi^{-1}(p_i) / sqrt(N)`` ~ N(0,1) (lower tail); inverse
        logit ``L* = sqrt(k) sum ln(p_i / (1 - p_i))`` ~ t(5N + 4) (lower tail)
        with ``k = 3(5N+4) / (pi^2 N (5N+2))``; modified inverse chi-squared
        ``Pm = -sum (ln p_i + 1) / sqrt(N)`` ~ N(0,1) (upper tail).
    ``"hadri"`` -- Hadri (2000) LM test; balanced panels; ``trend`` must be
        ``"constant"`` or ``"trend"``. H0: every panel is stationary. With
        ``e_it`` the residuals of each panel on its deterministic terms and
        ``S_it`` their partial sums, ``LM = mean_i (sum_t S_it^2 / T^2) /
        sigma^2`` with ``sigma^2 = sum e_it^2 / (N (T - d))`` (d = 1 or 2
        deterministic terms per panel); ``robust=True`` divides each panel by
        its own variance instead; ``kernel_lags`` replaces the variance by the
        average of the panels' long-run variances (kernel ``kernel``, divisor
        T). ``Z = sqrt(N) (LM - mu) / sigma`` ~ N(0,1), upper tail, with
        (mu, sigma^2) = (1/6, 1/45) or, with trends, (1/15, 11/6300).
    ``"breitung"`` -- Breitung (2000); balanced panels. H0: every panel has a
        unit root. With n = T - p - 1 usable differences per panel:

        - ``"none"`` / ``"constant"``: ``y*_t = Delta y_t`` and
          ``x*_t = y_{t-1}`` (minus the first lagged level of the sample with
          panel means); with ``lags = p > 0`` both are replaced by their
          residuals on Delta y_{t-1..t-p}. ``s_i^2 = sum y*^2 / (n - 1)``.
        - ``"trend"``: the regression of Delta y_t on a constant and its p lags
          gives the lag coefficients a_j; ``Du_s = Dy_s - sum a_j Dy_{s-j}``,
          ``u_s = y_{s-1} - sum a_j y_{s-j-1}``, ``s_i^2`` the variance of Du
          (divisor n - 1), and

              y*_s = sqrt((n-s)/(n-s+1)) (Du_s - mean(Du_{s+1..n})),
              x*_s = u_s - u_1 - (s-1) mean(Du),              s = 1..n-1

          (Breitung 2000; Stata's manual prints (T-p-1) mean(Du) in x*, which
          would not be invariant to the panels' trends).

        ``lambda = sum_i sum_t x* y* / s_i^2 / sqrt(sum_i sum_t x*^2 / s_i^2)``
        ~ N(0,1), lower tail (Stata's nonrobust statistic).
    ``"ht"`` -- Harris and Tzavalis (1999); balanced panels; no lagged
        differences (``lags`` must be 0). H0: every panel has a unit root.
        ``rho`` is the pooled least-squares coefficient of y_{t-1} after
        removing the panels' deterministic terms from y_t and y_{t-1};
        ``Z = sqrt(N) (rho - mu) / sigma`` ~ N(0,1), lower tail, where
        (mu, sigma^2) is ``(1, 2/(T(T-1)))`` without deterministic terms,
        ``(1 - 3/(T+1), 3(17T^2 - 20T + 17) / (5(T-1)(T+1)^3))`` with panel
        means and ``(1 - 15/(2(T+2)), 15(193T^2 - 728T + 1147) /
        (112(T+2)^3 (T-2)))`` with panel trends. ``T`` is the number of
        periods, as in Stata's default; ``altt=True`` uses T - 1, the number
        of observations per panel in the regression, which is how Harris and
        Tzavalis define T (Stata's ``altt``). With panel means or trends the
        default over-rejects in short panels; ``altt=True`` is correctly sized.

    Other parameters
    ----------------
    kernel : ``"bartlett"`` (default), ``"parzen"`` or ``"quadratic_spectral"``
        (LLC and Hadri long-run variances).
    kernel_lags : truncation of that kernel (see the tests).
    maxlag : largest lag order of the automatic choice.
    lrv : ``"paper"`` or ``"null"`` (LLC only; see above).
    method : ``"dfuller"`` or ``"pperron"`` (Fisher only).
    robust : heteroskedasticity across panels (Hadri only).
    altt : use T - 1 in the Harris-Tzavalis moments (HT only).

    Returns
    -------
    A table indexed by statistic name with ``statistic``, ``df`` and
    ``p_value``. ``attrs``: ``statistic`` and ``p_value`` of the headline
    statistic (``adjusted_t``, ``w_t_bar``, ``P``, ``z``, ``lambda``, ``z``),
    ``distribution``, ``test``, ``n_panels``, ``periods`` (average per panel),
    ``nobs``, ``balanced``, ``lags`` (average), ``lags_min``, ``lags_max``,
    ``trend``, ``demean``, ``h0``, ``ha``, ``label``, ``notes`` and the
    test-specific scalars (LLC: ``unadjusted_t``, ``coefficient``,
    ``mu_star``, ``sigma_star``, ``s_n``, ``kernel_lags``, ``t_tilde``; IPS:
    ``t_bar``; Fisher: ``inverse_normal_z``, ``inverse_logit_l``,
    ``modified_inverse_chi2``; Hadri: ``lm``; HT: ``rho``, ``rho_mean``,
    ``rho_variance``).

    Stata: ``xtunitroot llc y, lags(1) trend`` / ``xtunitroot ips y, lags(aic 4)``
    / ``xtunitroot fisher y, dfuller lags(2)`` / ``xtunitroot hadri y, robust``
    / ``xtunitroot breitung y, lags(1)`` / ``xtunitroot ht y, altt``. EViews:
    Unit Root Test on a pool or panel series (Levin-Lin-Chu, Im-Pesaran-Shin,
    Fisher ADF / PP, Hadri, Breitung).

    Example
    -------
    >>> result = oe.xtunitroot(df, "lnrxrate", "country", "month", test="ips", lags=2)
    >>> result.attrs["statistic"], result.attrs["p_value"]
    """
    check_choice(test, "test", _TESTS)
    check_choice(trend, "trend", tuple(_CASE))
    check_choice(kernel, "kernel", KERNELS)
    check_choice(method, "method", ("dfuller", "pperron"))
    check_choice(lrv, "lrv", ("paper", "null"))
    check_flag(demean, "demean")
    check_flag(robust, "robust")
    check_flag(altt, "altt")
    if test in {"cips", "cadf"}:
        unused = [name for name, given in (
            ("demean", demean), ("kernel", kernel != "bartlett"),
            ("kernel_lags", kernel_lags is not None), ("method", method != "dfuller"),
            ("robust", robust), ("lrv", lrv != "paper"), ("altt", altt),
        ) if given]
        if unused:
            raise AnalysisError("invalid_option", f"test='{test}' does not use: {', '.join(unused)}.")
        from openecon.econometrics.unitroot.cips import xtcips

        return xtcips(data, y, panel, time, lags=lags, maxlag=maxlag, trend=trend,
                      individual=test == "cadf")
    if isinstance(lags, str):
        check_choice(lags, "lags", ("aic", "bic", "hqic"))
        if maxlag is not None:
            maxlag = check_count(maxlag, "maxlag")
    else:
        lags = check_count(lags, "lags")
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag applies only with lags='aic', "
                                "'bic' or 'hqic'.")
    if kernel_lags is not None:
        kernel_lags = check_count(kernel_lags, "kernel_lags")
    unused = [name for name, given, tests in (
        ("kernel", kernel != "bartlett", {"llc", "hadri"}),
        ("kernel_lags", kernel_lags is not None, {"llc", "hadri"}),
        ("method", method != "dfuller", {"fisher"}),
        ("robust", robust, {"hadri"}),
        ("lrv", lrv != "paper", {"llc"}),
        ("altt", altt, {"ht"}),
    ) if given and test not in tests]
    if unused:
        raise AnalysisError("invalid_option", f"test='{test}' does not use the option(s): "
                            f"{', '.join(unused)}.")
    if test in {"hadri", "ht"} and lags != 0:
        raise AnalysisError("invalid_lags", f"test='{test}' has no lagged differences; leave "
                            "lags at 0" + (" and use kernel_lags for serial correlation."
                                           if test == "hadri" else "."))
    if test in {"ips", "hadri"} and trend == "none":
        raise AnalysisError("invalid_option", f"test='{test}' needs trend='constant' or "
                            "trend='trend'.")
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    sample = load_panel(data, [y], panel, time)
    count = sample.n_panels
    if count < 2:
        raise AnalysisError("insufficient_panels", "Panel unit-root tests need at least two "
                            "panels.")
    shortest = int(sample.counts.min())
    needed = 4 if test == "ht" else 6
    if shortest < needed:
        raise AnalysisError("insufficient_observations", f"Every panel needs at least {needed} "
                            f"observations; the shortest has {shortest}.")
    if test in {"llc", "hadri", "breitung", "ht"} and not sample.balanced:
        raise AnalysisError("unbalanced_panel", f"test='{test}' needs a balanced panel (the "
                            "same periods for every unit). Use test='ips' or 'fisher', or "
                            "restrict the sample.")
    if kernel_lags is not None and kernel_lags >= shortest - (1 if test == "llc" else 0):
        raise AnalysisError("invalid_lags", "kernel_lags must be smaller than the number of "
                            f"{'differences' if test == 'llc' else 'periods'} per panel "
                            f"({shortest - (1 if test == 'llc' else 0)}).")
    if demean:
        demean_cross_section(sample)
    if trend != "none":
        # Every statistic with panel means is invariant to the level of a panel: measure
        # each panel from its first observation so that a large level cannot hide its
        # variation in floating point.
        starts = sample.counts.cumsum(dim=0) - sample.counts
        sample.values[:, 0] -= sample.values[starts, 0][sample.codes]
    if test == "llc":
        result = _llc(sample, lags, trend, maxlag, kernel, kernel_lags, lrv)
    elif test == "ips":
        result = _ips(sample, lags, trend, maxlag)
    elif test == "fisher":
        result = _fisher(sample, lags, trend, maxlag, method)
    elif test == "hadri":
        result = _hadri(sample, trend, kernel, kernel_lags, robust)
    elif test == "ht":
        result = _ht(sample, trend, altt)
    else:
        result = _breitung(sample, lags, trend, maxlag)
    orders = result["lags_by_panel"]
    lag_info = {} if orders is None else {
        "lags": float(orders.to(torch.float64).mean()), "lags_min": int(orders.min()),
        "lags_max": int(orders.max()),
        "lag_method": lags if isinstance(lags, str) else "fixed"}
    rows = result["rows"]
    return table(
        [row[1:] for row in rows], columns=["statistic", "df", "p_value"],
        index=[row[0] for row in rows],
        title=f"Panel unit-root test ({test}) for {y}", test=test, series=y,
        statistic=result["statistic"], p_value=result["p_value"],
        distribution=result["distribution"], n_panels=count,
        periods=float(sample.counts.to(torch.float64).mean()), nobs=int(sample.counts.sum()),
        balanced=sample.balanced, trend=trend, demean=demean, h0=result["h0"],
        ha=result["ha"], label=result["label"], notes=result["notes"], **lag_info,
        **result["extra"])
