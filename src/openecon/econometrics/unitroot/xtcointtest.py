"""Panel cointegration test of Kao (1999)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.unitroot.common import (
    KERNELS, check_choice, check_count, check_flag, column_names, load_panel,
)
from openecon.engines import covariance as cov
from openecon.engines.distributions import normal_cdf
from openecon.engines.linalg import cholesky_solve, least_squares


def _conditional_variance(matrix: Tensor) -> float:
    """m_uu - m_ue m_ee^{-1} m_eu for a covariance of (u, e')'."""
    if matrix.shape[0] == 1:
        return float(matrix[0, 0])
    solved = kernel_call(cholesky_solve, matrix[1:, 1:].contiguous(),
                         matrix[1:, :1].contiguous())
    return float(matrix[0, 0] - (matrix[:1, 1:] @ solved)[0, 0])


def xtcointtest(data: Any, y: str, x: list[str], panel: str, time: str, *, test: str = "kao",
                lags: int = 1, kernel: str = "bartlett", kernel_lags: int | None = None,
                demean: bool = False, deterministic: str = "constant", ar: str | None = None,
                all_panels: bool = False, bandwidth: str | None = None, bootstrap: int = 0,
                seed: int = 0, block_length: int | None = None, max_work: int = 100_000_000):
    """Kao (1999) residual-based tests for cointegration in a panel (Stata's ``xtcointtest kao``).

    Model and estimator
    -------------------
    The cointegrating regression has panel-specific intercepts and a common
    slope vector, estimated by the within (LSDV) estimator on a balanced panel:

        y_it = a_i + x_it'b + e_it.

    H0: no cointegration (``e_it`` has a unit root in every panel); Ha: all
    panels are cointegrated with the same autoregressive parameter. With the
    residuals ``e`` the pooled Dickey-Fuller regression ``e_it = rho e_i,t-1 +
    v_it`` gives ``rho`` and ``t_rho = (rho - 1) sqrt(sum e_i,t-1^2) / s_v``
    (``s_v^2`` = SSR over the number of observations used); the augmented
    regression adds ``lags`` lagged differences of ``e`` and gives ``t_ADF``.
    With ``w_it = (Delta y_it, Delta x_it')'``, its covariance ``Sigma`` and
    its long-run covariance ``Omega`` (kernel ``kernel`` with ``kernel_lags``
    lags, autocovariances within panels), define the conditional variances

        s2_v  = Sigma_uu - Sigma_ue Sigma_ee^{-1} Sigma_eu,
        s2_0v = Omega_uu - Omega_ue Omega_ee^{-1} Omega_eu.

    The five statistics (N panels, T periods), each N(0, 1) under H0 with
    rejection in the lower tail, are

        DF_rho  = (sqrt(N) T (rho - 1) + 3 sqrt(N)) / sqrt(10.2)
        DF_t    = sqrt(1.25) t_rho + sqrt(1.875 N)
        DF*_rho = (sqrt(N) T (rho - 1) + 3 sqrt(N) s2_v / s2_0v)
                  / sqrt(3 + 36 s2_v^2 / (5 s2_0v^2))
        DF*_t   = (t_rho + sqrt(6 N) s_v / (2 s_0v))
                  / sqrt(s2_0v / (2 s2_v) + 3 s2_v / (10 s2_0v))
        ADF     = (t_ADF + sqrt(6 N) s_v / (2 s_0v))
                  / sqrt(s2_0v / (2 s2_v) + 3 s2_v / (10 s2_0v))

    DF_rho and DF_t assume strictly exogenous regressors and no serial
    correlation; the starred statistics and ADF correct for both. Stata labels
    them "Unadjusted modified Dickey-Fuller t" (DF_rho), "Unadjusted
    Dickey-Fuller t" (DF_t), "Modified Dickey-Fuller t" (DF*_rho),
    "Dickey-Fuller t" (DF*_t) and "Augmented Dickey-Fuller t" (ADF).

    Parameters
    ----------
    data : long panel (one row per unit and period; balanced, no gaps).
    y, x : outcome column and list of regressor columns.
    panel, time : unit and period columns.
    test : ``"kao"``, ``"pedroni"`` or Westerlund (2005) ``"westerlund"``.
        Heterogeneous methods have separate moment and bootstrap contracts;
        see docs/econometrics/panel-cointegration.md.
    lags : lagged differences in the augmented regression (default 1, Stata's
        default).
    kernel : ``"bartlett"`` (default), ``"parzen"`` or ``"quadratic_spectral"``.
    kernel_lags : truncation of the long-run covariance; default
        ``int(4 (T/100)^(2/9))``.
    demean : subtract the cross-sectional means of y and x per period first.

    Collinear regressors (after removing panel means) are omitted and listed
    in ``attrs["omitted"]``.

    Returns
    -------
    A table indexed ``modified_df``, ``df``, ``adf``, ``unadjusted_modified_df``,
    ``unadjusted_df`` with ``statistic`` and ``p_value``. ``attrs``:
    ``statistic`` and ``p_value`` (the ADF statistic), ``distribution``,
    ``rho``, ``t_rho``, ``t_adf``, ``sigma2_v``, ``sigma2_0v``,
    ``cointegrating_vector``, ``n_panels``, ``periods``, ``lags``, ``kernel``,
    ``kernel_lags``, ``omitted``, ``label``, ``notes``.

    Stata: ``xtcointtest kao y x1 x2, lags(1) kernel(bartlett 2)``. EViews:
    Panel Cointegration Test / Kao (reports the ADF statistic).

    Example
    -------
    >>> result = oe.xtcointtest(df, "productivity", ["rd_domestic", "rd_foreign"],
    ...                         "country", "year")
    >>> result.loc["adf", ["statistic", "p_value"]]
    """
    check_choice(test, "test", ("kao", "pedroni", "westerlund"))
    check_flag(all_panels, "all_panels")
    if all_panels and test != "westerlund":
        raise AnalysisError("unsupported_option", "all_panels is a Westerlund alternative hypothesis.")
    if all_panels and ar not in {None, "same"}:
        raise AnalysisError("invalid_option", "all_panels conflicts with ar='panel'.")
    if test != "kao":
        from openecon.econometrics.unitroot.panel_cointegration import run
        return run(data, y, x, panel, time, test=test, deterministic=deterministic,
                   ar=("same" if all_panels else "panel") if ar is None else ar,
                   lags=lags, kernel=kernel, kernel_lags=kernel_lags,
                   bandwidth="auto" if bandwidth is None else bandwidth,
                   demean=demean, bootstrap=bootstrap, seed=seed,
                   block_length=block_length, max_work=max_work)
    if deterministic != "constant" or ar not in {None, "same"} or bootstrap or block_length is not None:
        raise AnalysisError("unsupported_option", "Kao preserves constant/common-AR inference; heterogeneous options require Pedroni or Westerlund.")
    bandwidth = "legacy" if bandwidth is None else bandwidth
    check_choice(bandwidth, "bandwidth", ("legacy", "fixed", "auto"))
    if bandwidth == "auto" and kernel != "bartlett":
        raise AnalysisError("unsupported_bandwidth", "Automatic Newey-West bandwidth is implemented for Bartlett only.")
    if bandwidth == "fixed" and kernel_lags is None:
        raise AnalysisError("invalid_lags", "Fixed bandwidth requires kernel_lags.")
    check_choice(kernel, "kernel", KERNELS)
    check_flag(demean, "demean")
    lags = check_count(lags, "lags")
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", "The cointegration test needs at least one "
                            "regressor in x.")
    sample = load_panel(data, [y, *names], panel, time)
    count = sample.n_panels
    if count < 2:
        raise AnalysisError("insufficient_panels", "The test needs at least two panels.")
    if not sample.balanced:
        raise AnalysisError("unbalanced_panel", "The Kao test needs a balanced panel (the same "
                            "periods for every unit); restrict the sample.")
    cube = sample.cube()                                    # [N, T, 1 + k]
    periods = cube.shape[1]
    if periods < lags + 6:
        raise AnalysisError("insufficient_observations", f"Each panel needs at least "
                            f"{lags + 6} periods; {periods} given.")
    if demean:
        cube = cube - cube.mean(dim=0, keepdim=True)
    if kernel_lags is not None:
        bandwidth = "fixed"
    if kernel_lags is None:
        kernel_lags = int(4 * (periods / 100) ** (2 / 9))
    kernel_lags = check_count(kernel_lags, "kernel_lags")
    if kernel_lags >= periods - 1:
        raise AnalysisError("invalid_lags", "kernel_lags must be smaller than the number of "
                            "periods minus one.")
    within = (cube - cube.mean(dim=1, keepdim=True)).reshape(count * periods, -1)
    fit = kernel_call(least_squares, within[:, 1:].contiguous(), within[:, 0].contiguous())
    omitted = [names[i] for i in fit.omitted]
    kept_names = [names[i] for i in fit.kept]
    if not kept_names:
        raise AnalysisError("collinear_regressors", "No regressor varies within panels.")
    resid = fit.resid.reshape(count, periods)
    lagged, current = resid[:, :-1], resid[:, 1:]
    sum_ll = float(lagged.square().sum())
    if not float(fit.ssr) > 1e-22 * float(within[:, 0].square().sum()) or not sum_ll > 0.0:
        raise AnalysisError("perfect_fit", "The cointegrating regression fits exactly (y is a "
                            "linear combination of the regressors and the panel means); the "
                            "test is undefined.")
    rho = float((lagged * current).sum()) / sum_ll
    used = current.numel()
    s2 = float((current - rho * lagged).square().sum()) / used
    if not s2 > 0.0:
        raise AnalysisError("perfect_fit", "The residual Dickey-Fuller regression fits exactly.")
    t_rho = (rho - 1) * math.sqrt(sum_ll) / math.sqrt(s2)
    # Augmented regression, pooled: Delta e_t on e_{t-1} and lagged differences.
    de = current - lagged                                  # [N, T-1]
    columns = [resid[:, lags:periods - 1]] + [de[:, lags - j:periods - 1 - j]
                                               for j in range(1, lags + 1)]
    design = torch.stack([c.reshape(-1) for c in columns], dim=1)
    augmented = kernel_call(least_squares, design, de[:, lags:].reshape(-1))
    if augmented.omitted:
        raise AnalysisError("collinear_regressors", "The augmented Dickey-Fuller regression of "
                            "the residuals is rank deficient; use fewer lags.")
    df = design.shape[0] - design.shape[1]
    variance = float(augmented.ssr) / df * float(augmented.xtx_inv[0, 0])
    if not variance > 0.0:
        raise AnalysisError("perfect_fit", "The augmented Dickey-Fuller regression fits "
                            "exactly.")
    t_adf = float(augmented.beta[0]) / math.sqrt(variance)
    # Covariance and long-run covariance of w = (Delta y, Delta x).
    keep = [0, *(1 + i for i in fit.kept)]
    w = (cube[:, 1:, :] - cube[:, :-1, :])[:, :, keep].reshape(count * (periods - 1), -1)
    w = w.contiguous()
    codes = torch.arange(count).repeat_interleave(periods - 1)
    clock = torch.arange(periods - 1).repeat(count)
    total = w.shape[0]
    sigma = (w.T @ w) / total
    selected_lags = [kernel_lags] * count
    if bandwidth == "auto":
        from openecon.econometrics.unitroot.series import kpss_bandwidth
        unit_w = w.reshape(count, periods - 1, -1)
        selected_lags = []
        omega = torch.zeros_like(sigma)
        for unit, unit_resid in zip(unit_w, resid):
            innovations = unit_resid[1:] - rho * unit_resid[:-1]
            selected = kpss_bandwidth(innovations)
            selected_lags.append(selected)
            omega += kernel_call(cov.meat_hac, unit, selected, kernel) / total
    else:
        omega = kernel_call(cov.meat_hac, w, kernel_lags, kernel, time=clock, panel=codes) / total
    s2_v, s2_0v = _conditional_variance(sigma), _conditional_variance(omega)
    if not (s2_v > 0.0 and s2_0v > 0.0):
        raise AnalysisError("numerical_failure", "The conditional (long-run) variance of the "
                            "differences is not positive; use fewer kernel lags or drop a "
                            "regressor.")
    root_n = math.sqrt(count)
    bias = root_n * periods * (rho - 1)
    ratio = s2_v / s2_0v
    scale = math.sqrt(s2_0v / (2 * s2_v) + 3 * s2_v / (10 * s2_0v))
    shift = math.sqrt(6 * count) * math.sqrt(ratio) / 2
    statistics = {
        "modified_df": (bias + 3 * root_n * ratio) / math.sqrt(3 + 36 * ratio * ratio / 5),
        "df": (t_rho + shift) / scale,
        "adf": (t_adf + shift) / scale,
        "unadjusted_modified_df": (bias + 3 * root_n) / math.sqrt(10.2),
        "unadjusted_df": math.sqrt(1.25) * t_rho + math.sqrt(1.875 * count),
    }
    rows = [[value, normal_cdf(value)] for value in statistics.values()]
    notes = ["All statistics are N(0, 1) under H0 of no cointegration; p-values are lower-tail.",
             "Long-run covariance of (Delta y, Delta x): "
             f"{kernel} kernel, {bandwidth} bandwidth, within panels."]
    if omitted:
        notes.append(f"Omitted because of collinearity: {', '.join(omitted)}.")
    return table(
        rows, columns=["statistic", "p_value"], index=list(statistics),
        title=f"Kao test for cointegration of {y} with {', '.join(kept_names)}", test="kao",
        statistic=statistics["adf"], p_value=normal_cdf(statistics["adf"]),
        distribution="normal", rho=rho, t_rho=t_rho, t_adf=t_adf, sigma2_v=s2_v,
        sigma2_0v=s2_0v,
        cointegrating_vector=dict(zip(kept_names, fit.beta.tolist(), strict=True)),
        n_panels=count, periods=periods, nobs=count * periods, lags=lags, kernel=kernel,
        kernel_lags=kernel_lags if bandwidth != "auto" else None,
        kernel_lags_by_panel=selected_lags, bandwidth=bandwidth,
        demean=demean, omitted=omitted,
        h0="No cointegration", ha="All panels are cointegrated",
        label="Kao augmented Dickey-Fuller t (H0: no cointegration)", notes=notes)
