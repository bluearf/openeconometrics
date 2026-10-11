"""Fourier cointegration tests: Tsong et al. (2016) and Banerjee, Arcabic and Lee (2017).

Smooth breaks in the deterministic component are approximated by one Fourier
frequency, d_t = d0 (+ d1 t) + a sin(2 pi k t / T) + b cos(2 pi k t / T),
t = 1..T, with k chosen by the smallest sum of squared residuals.

``tsong_fourier`` tests the NULL OF COINTEGRATION (KPSS type, large values
reject); ``fourier_adl`` tests the null of NO cointegration (ADL t ratio,
small values reject). See ``docs/econometrics/cointegration-breaks.md`` for
the provenance of the critical values (secondary transcriptions of the
published tables, corroborated by a committed seeded simulation).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, column_names, load_series,
)
from openecon.engines.linalg import least_squares

LEVELS = ("1%", "5%", "10%")

# Tsong, Lee, Tsai and Hu (2016), Table 1 (CI_f, 10/5/1% upper quantiles): rows k = 1..3,
# within them m = 1..4 regressors. Cells on which tspdlib (cv_coint_tsongetal, commit
# a3660eaa) and the SSC module fcoint (_fcoint_cv_tsong.ado, 2026-03-07) agree; the
# constant-model cells k = 3, m = 4 (tspdlib repeats the k = 2 row) and the 10% value of
# k = 3, m = 1 (0.224 vs 0.225) are withheld.
TSONG = {
    "constant": {
        1: ((0.095, 0.124, 0.198), (0.070, 0.092, 0.155), (0.059, 0.076, 0.130),
            (0.050, 0.061, 0.096)),
        2: ((0.200, 0.276, 0.473), (0.132, 0.182, 0.328), (0.098, 0.132, 0.215),
            (0.072, 0.097, 0.171)),
        3: ((None, 0.304, 0.507), (0.148, 0.202, 0.383), (0.112, 0.146, 0.250),
            (None, None, None)),
    },
    "trend": {
        1: ((0.042, 0.048, 0.063), (0.038, 0.045, 0.059), (0.036, 0.042, 0.055),
            (0.034, 0.038, 0.050)),
        2: ((0.078, 0.099, 0.163), (0.063, 0.081, 0.127), (0.051, 0.066, 0.103),
            (0.044, 0.055, 0.086)),
        3: ((0.090, 0.114, 0.170), (0.075, 0.094, 0.143), (0.061, 0.075, 0.116),
            (0.053, 0.065, 0.099)),
    },
}

# Banerjee, Arcabic and Lee (2017), Tables 1a (constant) and 1b (constant and trend), as
# transcribed in the SSC module fcoint (_fcoint_cv_fadl_single.ado, 2026-03-07): rows
# m = 1..3 regressors, within them k = 1..5; each row 1/5/10% at T = 100, 500 and a larger
# tabulated sample (labelled 2500 in CRAN cointests).
FADL = {
    "constant": {
        1: ((-4.73, -4.09, -3.76, -4.61, -4.03, -3.72, -4.60, -4.02, -3.72),
            (-4.44, -3.75, -3.37, -4.33, -3.70, -3.36, -4.33, -3.70, -3.35),
            (-4.21, -3.51, -3.14, -4.13, -3.48, -3.14, -4.15, -3.48, -3.14),
            (-4.07, -3.38, -3.03, -4.01, -3.37, -3.04, -4.04, -3.38, -3.04),
            (-4.00, -3.32, -2.97, -3.94, -3.32, -2.99, -3.96, -3.32, -3.00)),
        2: ((-4.96, -4.32, -3.98, -4.85, -4.26, -3.95, -4.81, -4.25, -3.95),
            (-4.79, -4.10, -3.73, -4.66, -4.03, -3.70, -4.64, -4.03, -3.70),
            (-4.56, -3.87, -3.49, -4.48, -3.83, -3.49, -4.48, -3.83, -3.49),
            (-4.43, -3.73, -3.36, -4.34, -3.70, -3.36, -4.36, -3.71, -3.38),
            (-4.35, -3.65, -3.28, -4.27, -3.64, -3.30, -4.28, -3.65, -3.31)),
        3: ((-5.17, -4.51, -4.17, -5.06, -4.46, -4.15, -5.02, -4.45, -4.14),
            (-5.04, -4.36, -3.99, -4.94, -4.30, -3.96, -4.90, -4.29, -3.96),
            (-4.90, -4.16, -3.79, -4.76, -4.13, -3.78, -4.75, -4.13, -3.78),
            (-4.75, -4.03, -3.65, -4.64, -4.00, -3.65, -4.65, -4.01, -3.67),
            (-4.66, -3.94, -3.57, -4.58, -3.94, -3.58, -4.56, -3.93, -3.60)),
    },
    "trend": {
        1: ((-5.17, -4.55, -4.24, -5.04, -4.47, -4.19, -5.00, -4.46, -4.18),
            (-5.01, -4.34, -4.00, -4.86, -4.27, -3.96, -4.86, -4.27, -3.95),
            (-4.79, -4.11, -3.76, -4.70, -4.07, -3.75, -4.68, -4.07, -3.74),
            (-4.64, -3.96, -3.61, -4.58, -3.94, -3.61, -4.55, -3.95, -3.61),
            (-4.53, -3.87, -3.52, -4.48, -3.86, -3.54, -4.46, -3.86, -3.54)),
        2: ((-5.36, -4.72, -4.40, -5.23, -4.66, -4.37, -5.20, -4.65, -4.37),
            (-5.22, -4.57, -4.23, -5.10, -4.51, -4.20, -5.07, -4.49, -4.19),
            (-5.07, -4.39, -4.02, -4.95, -4.34, -4.01, -4.91, -4.32, -4.00),
            (-4.94, -4.23, -3.87, -4.84, -4.22, -3.88, -4.81, -4.20, -3.87),
            (-4.86, -4.15, -3.79, -4.76, -4.13, -3.80, -4.74, -4.13, -3.80)),
        3: ((-5.54, -4.89, -4.55, -5.39, -4.83, -4.53, -5.39, -4.83, -4.53),
            (-5.47, -4.77, -4.42, -5.30, -4.73, -4.40, -5.28, -4.70, -4.39),
            (-5.33, -4.63, -4.27, -5.18, -4.57, -4.25, -5.18, -4.55, -4.23),
            (-5.19, -4.49, -4.11, -5.08, -4.46, -4.12, -5.06, -4.44, -4.11),
            (-5.08, -4.38, -4.01, -5.01, -4.36, -4.03, -4.99, -4.37, -4.03)),
    },
}


def tsong_critical(model: str, frequency: int, regressors: int) -> dict[str, float | None] | None:
    rows = TSONG[model].get(frequency)
    if rows is None or not 1 <= regressors <= 4:
        return None
    ten, five, one = rows[regressors - 1]
    if ten is None and five is None and one is None:
        return None
    return {"1%": one, "5%": five, "10%": ten}


def fadl_critical(model: str, frequency: int, regressors: int,
                  nobs: int) -> tuple[dict[str, float] | None, str]:
    """Conservative bracket of the tabulated sample sizes; never interpolated or extrapolated.

    For 100 <= T < 500 the elementwise minimum of the T = 100 and T = 500 rows, for
    T >= 500 the minimum of the T = 500 row and the larger-sample row; below 100 none.
    """
    rows = FADL[model].get(regressors)
    if rows is None or not 1 <= frequency <= 5:
        return None, "no critical values beyond m = 3 regressors or k = 5"
    row = rows[frequency - 1]
    if nobs < 100:
        return None, "no critical values below T = 100 (smallest tabulated sample size)"
    low, high = (row[0:3], row[3:6]) if nobs < 500 else (row[3:6], row[6:9])
    bracket = "T = 100 and 500" if nobs < 500 else "T = 500 and the largest tabulated T"
    values = {level: min(a, b) for level, a, b in zip(LEVELS, low, high, strict=True)}
    return values, f"conservative bracket (more negative of the rows for {bracket})"


def _ols(design: Tensor, target: Tensor, what: str):
    if design.shape[0] <= design.shape[1] + 1:
        raise AnalysisError("insufficient_observations", f"{what} has {design.shape[0]} "
                            f"observations for {design.shape[1]} parameters.")
    fit = kernel_call(least_squares, design, target)
    if fit.omitted:
        raise AnalysisError("collinear_regressors", f"{what} is rank deficient.")
    ssr = float(fit.ssr)
    df = design.shape[0] - design.shape[1]
    se = (ssr / df * fit.xtx_inv.diagonal()).sqrt()
    return fit, ssr, se


def _fourier(total: int, frequency: int) -> list[Tensor]:
    t = torch.arange(1, total + 1, dtype=torch.float64)
    angle = 2.0 * math.pi * frequency * t / total
    return [torch.sin(angle), torch.cos(angle)]


def _deterministic(total: int, model: str) -> list[Tensor]:
    columns = [torch.ones(total, dtype=torch.float64)]
    if model == "trend":
        t = torch.arange(1, total + 1, dtype=torch.float64)
        columns.append((t - t.mean()) / total)
    return columns


def _load(data: Any, y: str, x: Any, time: str | None, what: str) -> tuple[Tensor, list[str]]:
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", f"{what} needs at least one regressor in x.")
    values, _ = load_series(data, [y, *names], time)
    if values.shape[0] > 200_000:
        raise AnalysisError("workspace_limit", f"{what} admits at most 200,000 observations.")
    for index, name in enumerate([y, *names]):
        if not bool((values[:, index] != values[0, index]).any()):
            raise AnalysisError("constant_series", f"Column '{name}' is constant.")
    return values - values.mean(dim=0), names


def _long_run(e: Tensor, kernel: str, bandwidth: int) -> float:
    n = e.shape[0]
    value = float(e.square().sum()) / n
    if kernel == "bartlett":
        for lag in range(1, min(bandwidth, n - 1) + 1):
            value += 2.0 * (1.0 - lag / (bandwidth + 1)) * float(e[lag:] @ e[:-lag]) / n
    return value


def tsong_fourier(data: Any, y: str, x: list[str], *, time: str | None = None,
                  model: str = "constant", max_frequency: int = 3, estimator: str = "dols",
                  leads_lags: int | None = None, kernel: str = "bartlett",
                  bandwidth: int | None = None) -> TableSet:
    """Tsong, Lee, Tsai and Hu (2016) Fourier test of the null of cointegration.

    Model and estimator
    -------------------
    y_t = d0 (+ d1 t) + a sin(2 pi k t/T) + b cos(2 pi k t/T) + beta'x_t + eta_t,
    estimated by dynamic OLS (``estimator="dols"``, default: q leads and lags
    and the current value of Dx_t added; observations t = q+2..T-q) or OLS
    (``"ols"``, valid only for strictly exogenous regressors). The frequency
    k = 1..``max_frequency`` (at most 3, the range of the table) minimizes
    the SSR of that regression. With S_t the partial sums of its residuals,

        CI_f = n^-2 sum_t S_t^2 / omega^2,

    n the number of regression observations and omega^2 the long-run
    variance of the residuals: ``kernel="bartlett"`` (default; weights
    1 - j/(B+1), bandwidth B default round(4 (n/100)^(2/9))) or ``"iid"``
    (the residual variance, the tspdlib default). H0: cointegration; large
    values reject. ``leads_lags`` q defaults to int(4 (T/100)^(2/9)).

    The F statistic of the Fourier terms (sin, cos jointly zero) is reported
    without a p-value: it is not calibrated here.

    Critical values
    ---------------
    Tsong et al. (2016, Table 1) for k = 1..3 and m = 1..4 regressors (upper
    1/5/10% points), as agreed by two secondary transcriptions (the article
    could not be retrieved); a seeded simulation of the asymptotic null
    (``scripts/generate_cointegration_breaks_oracles.py --fourier-check``)
    reproduces them within 13% (the tabulated values are the larger ones).
    Disagreeing cells are withheld (None). No p-value is tabulated.
    """
    check_choice(model, "model", ("constant", "trend"))
    check_choice(estimator, "estimator", ("dols", "ols"))
    check_choice(kernel, "kernel", ("bartlett", "iid"))
    max_frequency = check_count(max_frequency, "max_frequency", minimum=1)
    if max_frequency > 5:
        raise AnalysisError("invalid_option", "max_frequency must be at most 5.")
    what = "The Tsong et al. Fourier cointegration test"
    values, names = _load(data, y, x, time, what)
    total, m = values.shape[0], len(names)
    q = int(4 * (total / 100) ** (2 / 9)) if leads_lags is None else check_count(
        leads_lags, "leads_lags")
    if estimator == "ols" and leads_lags is not None:
        raise AnalysisError("invalid_option", "leads_lags applies only to estimator='dols'.")
    xs = values[:, 1:]
    if estimator == "dols":
        diff = torch.cat([torch.zeros((1, m), dtype=torch.float64), xs[1:] - xs[:-1]])
        rows = torch.arange(q + 1, total - q)                 # 0-based t = q+2..T-q
        extra = [diff[rows + j] for j in range(-q, q + 1)]
        extra = torch.cat(extra, dim=1)
    else:
        rows = torch.arange(total)
        extra = torch.zeros((total, 0), dtype=torch.float64)
    n = rows.shape[0]
    if n <= 3 * (m + 4) + extra.shape[1]:
        raise AnalysisError("insufficient_observations", f"{what} has {n} regression "
                            "observations; use fewer leads/lags or a longer series.")
    width = round(4 * (n / 100) ** (2 / 9)) if bandwidth is None else check_count(
        bandwidth, "bandwidth")
    base = torch.cat([torch.stack(_deterministic(total, model), dim=1)[rows], xs[rows], extra],
                     dim=1)
    target = values[rows, 0]
    restricted, ssr_r, _ = _ols(base, target, "The cointegrating regression without Fourier terms")
    best = None
    for k in range(1, max_frequency + 1):
        fourier = torch.stack(_fourier(total, k), dim=1)[rows]
        fit, ssr, _ = _ols(torch.cat([base, fourier], dim=1), target,
                           "The Fourier cointegrating regression")
        if best is None or ssr < best[1]:
            best = (k, ssr, fit)
    k, ssr, fit = best
    resid = fit.resid
    omega = _long_run(resid, kernel, width)
    if not omega > 0:
        raise AnalysisError("invalid_covariance", "The long-run variance estimate is not "
                            "positive; use kernel='iid' or a smaller bandwidth.")
    partial = resid.cumsum(dim=0)
    statistic = float(partial.square().sum()) / (n * n * omega)
    params = base.shape[1] + 2
    f_stat = ((ssr_r - ssr) / 2.0) / (ssr / (n - params))
    critical = tsong_critical(model, k, m) if k <= 3 else None
    status = ("Tsong et al. (2016, Table 1) as agreed by two transcriptions"
              if critical is not None else
              "no critical value for this frequency/number of regressors")
    if critical is not None and estimator == "ols":
        status += "; the OLS statistic shares the null distribution only with exogenous regressors"
    return TableSet(
        {"test": table([[statistic, None, *(None if critical is None else critical[level]
                                            for level in LEVELS), k, f_stat]],
                       columns=["statistic", "p_value", "critical_1pct", "critical_5pct",
                                "critical_10pct", "frequency", "fourier_F"], index=["CI_f"])},
        title=f"Tsong et al. Fourier test of the null of cointegration: {y} on {', '.join(names)}",
        test="tsong_fourier", statistic=statistic, p_value=None, critical_values=critical,
        frequency=k, fourier_F=f_stat, model=model, estimator=estimator, leads_lags=(
            q if estimator == "dols" else None), kernel=kernel,
        bandwidth=width if kernel == "bartlett" else None, long_run_variance=omega, nobs=n,
        sample_size=total, regressors=m, inference_status=status, null="Cointegration",
        dtype="float64", device="cpu",
        source="Tsong, Lee, Tsai and Hu (2016) Empirical Economics 51",
        notes=["H0: cointegration; reject for statistics above the critical value.",
               "The Fourier F statistic is not calibrated (no p-value)."])


def fourier_adl(data: Any, y: str, x: list[str], *, time: str | None = None,
                model: str = "constant", max_frequency: int = 3, lags: int | str = "aic",
                max_lag: int = 6, dx_lags: int | None = None) -> TableSet:
    """Banerjee, Arcabic and Lee (2017) Fourier ADL test of the null of no cointegration.

    Model and estimator
    -------------------
    Dy_t = d0 (+ d1 t) + a sin(2 pi k t/T) + b cos(2 pi k t/T) + delta y_(t-1)
           + gamma'x_(t-1) + sum_(j=0..q_i) phi_ij' Dx_i,(t-j) + sum_(i=1..p) alpha_i Dy_(t-i) + e_t

    by OLS. The statistic is the t ratio of delta (H0: delta = 0, no
    cointegration; small values reject). Following the authors' RATS code (as
    relayed by the SSC module fcoint): every regression uses the common sample
    t = max_lag+2..T; for each frequency k = 1..``max_frequency`` the lag
    orders p, q_i = 1..``max_lag`` minimize ``lags="aic"`` (default) or
    ``"bic"`` (ln(SSR/n) + c K/n with K all coefficients; one q per regressor
    for at most two regressors, a common q otherwise); the frequency with the
    smallest SSR of its selected model is kept. An integer ``lags`` fixes p
    (and ``dx_lags`` q, default p) with sample t = max(p, q)+2..T.

    Critical values
    ---------------
    Banerjee, Arcabic and Lee (2017, Tables 1a/1b) for m = 1..3 regressors and
    k = 1..5, as transcribed in the SSC module fcoint (the article could not be
    retrieved), corroborated by a seeded simulation. The table is not
    interpolated: for T between tabulated sample sizes the more negative
    (conservative) of the two bracketing values is reported, below T = 100
    none. No p-value is tabulated.
    """
    check_choice(model, "model", ("constant", "trend"))
    max_frequency = check_count(max_frequency, "max_frequency", minimum=1)
    if max_frequency > 5:
        raise AnalysisError("invalid_option", "max_frequency must be at most 5.")
    what = "The Fourier ADL cointegration test"
    values, names = _load(data, y, x, time, what)
    total, m = values.shape[0], len(names)
    if isinstance(lags, str):
        criterion = check_choice(lags, "lags", ("aic", "bic"))
        top = check_count(max_lag, "max_lag", minimum=1)
        if dx_lags is not None:
            raise AnalysisError("invalid_option", "dx_lags applies only with an integer lags.")
        grid_p = list(range(1, top + 1))
        orders = range(1, top + 1)
        if m == 1:
            q_grid = [(a,) for a in orders]
        elif m == 2:
            q_grid = [(a, b) for a in orders for b in orders]
        else:
            q_grid = [(a,) * m for a in orders]
    else:
        criterion = None
        p = check_count(lags, "lags")
        q = p if dx_lags is None else check_count(dx_lags, "dx_lags")
        top = max(p, q)
        grid_p, q_grid = [p], [(q,) * m]
    start = top + 1                                         # 0-based first observation
    n = total - start
    if n <= 2 * (m + 2) * (top + 2) + 6:
        raise AnalysisError("insufficient_observations", f"{what} has {n} observations for "
                            f"lag orders up to {top}.")
    diff = torch.cat([torch.zeros((1, m + 1), dtype=torch.float64),
                      values[1:] - values[:-1]])           # row t = D at t (row 0 unused)
    rows = torch.arange(start, total)
    target = diff[rows, 0]
    det = torch.stack(_deterministic(total, model), dim=1)[rows]
    levels = values[rows - 1]                               # y_(t-1), x_(t-1)
    best = None
    for k in range(1, max_frequency + 1):
        fourier = torch.stack(_fourier(total, k), dim=1)[rows]
        head = torch.cat([det, fourier, levels], dim=1)
        position = det.shape[1] + 2                         # column of y_(t-1)
        k_best = None
        for p in grid_p:
            for qs in q_grid:
                pieces = [head]
                for i, qi in enumerate(qs):
                    pieces.append(torch.stack([diff[rows - j, 1 + i] for j in range(qi + 1)],
                                              dim=1))
                if p:
                    pieces.append(torch.stack([diff[rows - j, 0] for j in range(1, p + 1)],
                                              dim=1))
                design = torch.cat(pieces, dim=1)
                fit, ssr, se = _ols(design, target, "The Fourier ADL regression")
                count = design.shape[1]
                if criterion is None:
                    ic = 0.0
                else:
                    penalty = 2.0 if criterion == "aic" else math.log(n)
                    ic = math.log(ssr / n) + penalty * count / n
                if k_best is None or ic < k_best["ic"]:
                    k_best = {"ic": ic, "ssr": ssr, "p": p, "q": list(qs),
                              "delta": float(fit.beta[position]),
                              "se": float(se[position])}
        if best is None or k_best["ssr"] < best["ssr"]:
            best = {**k_best, "k": k}
    statistic = best["delta"] / best["se"]
    critical, bracket = fadl_critical(model, best["k"], m, total)
    status = ("Banerjee, Arcabic and Lee (2017, Tables 1a/1b) via the fcoint transcription; "
              + bracket) if critical is not None else bracket
    return TableSet(
        {"test": table([[statistic, None, *(None if critical is None else critical[level]
                                            for level in LEVELS), best["k"], best["p"],
                         ",".join(map(str, best["q"]))]],
                       columns=["statistic", "p_value", "critical_1pct", "critical_5pct",
                                "critical_10pct", "frequency", "lags", "dx_lags"],
                       index=["t_FADL"])},
        title=f"Fourier ADL test for cointegration: {y} on {', '.join(names)}",
        test="fourier_adl", statistic=statistic, p_value=None, critical_values=critical,
        frequency=best["k"], lags=best["p"], dx_lags=best["q"], delta=best["delta"],
        se_delta=best["se"], criterion=criterion, max_lag=top, model=model, nobs=n,
        sample_size=total, regressors=m, inference_status=status, null="No cointegration",
        dtype="float64", device="cpu",
        source="Banerjee, Arcabic and Lee (2017) Economic Modelling 67",
        notes=["H0: no cointegration; reject for statistics below the critical value.",
               "No p-value is tabulated."])
