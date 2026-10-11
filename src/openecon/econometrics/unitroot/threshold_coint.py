"""Enders-Siklos (2001) threshold cointegration tests (TAR and momentum TAR adjustment).

The Engle-Granger residuals mu_t of y on (1, x) follow

    D mu_t = I_t rho1 mu_(t-1) + (1 - I_t) rho2 mu_(t-1) + sum_(i=1..p) g_i D mu_(t-i) + e_t,

with I_t = 1(mu_(t-1) >= tau) (TAR) or I_t = 1(D mu_(t-1) >= tau) (M-TAR), and
tau = 0 or Chan's (1993) consistent estimate. The statistics are Phi (the F
statistic of rho1 = rho2 = 0; Phi* with the estimated threshold), t-Max (the
larger of the two t ratios) and the F test of symmetric adjustment rho1 = rho2.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot.common import (
    check_choice, check_count, column_names, load_series, regress,
)
from openecon.engines.distributions import f_sf

_MAX_OBS = 100_000


def _zero_threshold_ssr(dmu: Tensor, lagged: Tensor, signal: Tensor, history: Tensor) -> float:
    """SSR of the TAR regression with tau = 0 (lag-order comparison)."""
    above = (signal >= 0.0).to(torch.float64)
    design = torch.cat([torch.stack([above * lagged, (1.0 - above) * lagged], dim=1), history],
                       dim=1)
    return regress(design, dmu, "The threshold adjustment regression").ssr


@torch.no_grad()
def threshold_ssr(dmu: Tensor, lagged: Tensor, signal: Tensor, history: Tensor,
                  grid: Tensor) -> Tensor:
    """SSR of the TAR regression at every threshold in ``grid``, in O(n p) work.

    The design (I z, (1 - I) z, H) spans the same space as (I z, z, H): with
    the base regression on B = (z, H) = Q R and its residual r0, adding
    a = I z gives SSR = r0'r0 - (a'r0)^2 / (a'a - |Q'a|^2). For I_t = 1(s_t >= tau)
    the three cross products are suffix sums over the rows sorted by s.
    Thresholds leaving a regime without information are +inf.
    """
    base = torch.cat([lagged[:, None], history], dim=1)
    fit = regress(base, dmu, "The threshold adjustment regression without regimes")
    q, _ = torch.linalg.qr(base)
    order = torch.argsort(signal, descending=True, stable=True)
    z, resid, basis = lagged[order], fit.resid[order], q[order]
    cross = torch.cumsum(z * resid, dim=0)
    square = torch.cumsum(z * z, dim=0)
    projected = torch.cumsum(z[:, None] * basis, dim=0)
    ascending = torch.sort(signal).values
    count = signal.shape[0] - torch.searchsorted(ascending, grid, side="left")
    index = (count - 1).clamp(min=0)
    denominator = square[index] - projected[index].square().sum(dim=1)
    usable = (count > 0) & (denominator > 1e-10 * square[index].clamp_min(1e-300))
    ssr = fit.ssr - cross[index].square() / torch.where(usable, denominator,
                                                         torch.ones_like(denominator))
    return torch.where(usable, ssr, torch.full_like(ssr, float("inf")))


def _statistics(design: Tensor, target: Tensor) -> dict[str, float]:
    fit = regress(design, target, "The threshold adjustment regression")
    if fit.exact:
        raise AnalysisError("perfect_fit", "The threshold regression fits exactly.")
    b, v = fit.beta[:2], fit.s2 * fit.xtx_inv[:2, :2]
    phi = float(b @ torch.linalg.solve(v, b)) / 2.0
    t1, t2 = float(fit.beta[0] / fit.se[0]), float(fit.beta[1] / fit.se[1])
    contrast = torch.tensor([1.0, -1.0], dtype=torch.float64)
    symmetry = float((contrast @ b) ** 2 / (contrast @ v @ contrast))
    return {"phi": phi, "t_max": max(t1, t2), "t_above": t1, "t_below": t2,
            "rho_above": float(fit.beta[0]), "rho_below": float(fit.beta[1]),
            "symmetry_F": symmetry, "df_resid": fit.df, "ssr": fit.ssr, "nobs": fit.nobs}


def enders_siklos(data: Any, y: str, x: list[str], *, time: str | None = None,
                  model: str = "tar", threshold: str = "zero", lags: int | str = 1,
                  max_lag: int | None = None, trim: float = 0.15) -> TableSet:
    """Enders and Siklos (2001) tests for cointegration with threshold adjustment.

    Model and estimator
    -------------------
    Step 1: OLS of y on a constant and x gives mu_t. Step 2: OLS without
    constant of D mu_t on I_t mu_(t-1), (1 - I_t) mu_(t-1) and p lagged
    differences on t = p+2..T (``model="tar"``, I_t = 1(mu_(t-1) >= tau)) or
    t = max(p, 1)+2..T (``"mtar"``, I_t = 1(D mu_(t-1) >= tau)).

    ``threshold="zero"`` sets tau = 0; ``"consistent"`` is Chan's (1993)
    estimate: the value of the threshold variable minimizing the SSR among its
    sorted distinct sample values at positions fix(trim N)..N - fix(trim N) + 1
    (Doan's ENDERSSIKLOS procedure on Enders's web site; trim 0.15 as in the
    paper). ``lags`` fixes p or chooses it by ``"aic"``/``"bic"``
    (ln(SSR/n) + c (p+2)/n over p = 0..max_lag on the common sample of the
    largest order).

    Statistics
    ----------
    Phi (Phi* with the estimated threshold): F statistic of rho1 = rho2 = 0;
    t-Max: the larger t ratio of rho1 and rho2; both test H0: no cointegration
    with nonstandard distributions. F(rho1 = rho2): symmetric adjustment,
    with the F(1, n-k) p-value that Enders and Siklos use when the series are
    cointegrated.

    Critical values
    ---------------
    Enders and Siklos (2001, Tables 1 and 5) could not be obtained from an
    authoritative source: Phi and t-Max are reported WITHOUT critical values
    or p-values (``inference_status`` says so).

    Returns
    -------
    TableSet with ``test`` (Phi, t-Max, symmetry F rows) and ``adjustment``
    (rho1, rho2 with t ratios); ``attrs`` hold the threshold, lags and the
    statistics.
    """
    check_choice(model, "model", ("tar", "mtar"))
    check_choice(threshold, "threshold", ("zero", "consistent"))
    if isinstance(trim, bool) or not isinstance(trim, (int, float)) or not 0.0 < trim < 0.5:
        raise AnalysisError("invalid_option", "trim must be a fraction in (0, 0.5).")
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", "The Enders-Siklos test needs at least one regressor.")
    if isinstance(lags, str):
        criterion = check_choice(lags, "lags", ("aic", "bic"))
        top = 4 if max_lag is None else check_count(max_lag, "max_lag")
    else:
        if max_lag is not None:
            raise AnalysisError("invalid_option", "max_lag applies only with lags='aic'/'bic'.")
        criterion, top = None, check_count(lags, "lags")
    values, _ = load_series(data, [y, *names], time)
    total = values.shape[0]
    if total > _MAX_OBS:
        raise AnalysisError("workspace_limit", f"The Enders-Siklos test admits at most "
                            f"{_MAX_OBS:,} observations.")
    means = values.mean(dim=0)
    values = values - means
    design = torch.cat([torch.ones((total, 1), dtype=torch.float64), values[:, 1:]], dim=1)
    first = regress(design, values[:, 0], "The Engle-Granger cointegrating regression")
    if first.exact:
        raise AnalysisError("perfect_fit", "y is an exact linear combination of the regressors.")
    mu = first.resid
    dmu = torch.cat([torch.zeros(1, dtype=torch.float64), mu[1:] - mu[:-1]])   # row t = D mu_t

    def pieces(p: int, start: int):
        rows = torch.arange(start, total)
        history = torch.stack([dmu[rows - i] for i in range(1, p + 1)], dim=1) if p else \
            torch.zeros((rows.shape[0], 0), dtype=torch.float64)
        signal = mu[rows - 1] if model == "tar" else dmu[rows - 1]
        return rows, dmu[rows], mu[rows - 1], signal, history

    def first_row(p: int) -> int:                       # 0-based first observation
        return p + 1 if model == "tar" else max(p, 1) + 1

    if criterion is not None:
        start = first_row(top)
        best = None
        for p in range(top + 1):
            rows, target, lagged, signal, history = pieces(p, start)
            ssr = _zero_threshold_ssr(target, lagged, signal, history)
            count = rows.shape[0]
            penalty = 2.0 if criterion == "aic" else math.log(count)
            ic = math.log(ssr / count) + penalty * (p + 2) / count
            if best is None or ic < best[0]:
                best = (ic, p)
        p = best[1]
    else:
        p = top
    rows, target, lagged, signal, history = pieces(p, first_row(p))
    if rows.shape[0] < 3 * (p + 2) + 6:
        raise AnalysisError("insufficient_observations", "The threshold regression has too few "
                            "observations for the requested lags.")
    if threshold == "zero":
        tau = 0.0
        candidates = 1
    else:
        distinct = torch.unique(signal)                  # sorted
        count = distinct.shape[0]
        first_pos = int(trim * count)
        last_pos = count - first_pos + 1
        first_pos, last_pos = max(first_pos, 1), min(last_pos, count)
        grid = distinct[first_pos - 1:last_pos]
        if grid.shape[0] < 1:
            raise AnalysisError("insufficient_observations", "No threshold candidate remains "
                                "after trimming.")
        ssr = threshold_ssr(target, lagged, signal, history, grid)
        index = int(torch.argmin(ssr))
        if not math.isfinite(float(ssr[index])):
            raise AnalysisError("collinear_regressors", "No threshold candidate gives a full-rank "
                                "regression.")
        tau = float(grid[index])
        candidates = int(grid.shape[0])
    above = (signal >= tau).to(torch.float64)
    regime_counts = (int(above.sum()), int((1 - above).sum()))
    if min(regime_counts) < 2:
        raise AnalysisError("collinear_regressors", "One threshold regime has fewer than two "
                            "observations.")
    final_design = torch.cat([torch.stack([above * lagged, (1 - above) * lagged], dim=1),
                              history], dim=1)
    stats = _statistics(final_design, target)
    symmetry_p = f_sf(stats["symmetry_F"], 1, stats["df_resid"])
    phi_name = "Phi" if threshold == "zero" else "Phi*"
    status = ("Phi and t-Max: no critical values (Enders-Siklos 2001 tables not obtained from an "
              "authoritative source); symmetry F: F(1, n-k) p-value, valid under cointegration")
    test_rows = [[stats["phi"], None, "rho1 = rho2 = 0"], [stats["t_max"], None, "max t"],
                 [stats["symmetry_F"], symmetry_p, "rho1 = rho2"]]
    return TableSet(
        {"test": table(test_rows, columns=["statistic", "p_value", "hypothesis"],
                       index=[phi_name, "t-Max", "F_symmetry"]),
         "adjustment": table([[stats["rho_above"], stats["t_above"], regime_counts[0]],
                              [stats["rho_below"], stats["t_below"], regime_counts[1]]],
                             columns=["coefficient", "t_ratio", "nobs_regime"],
                             index=["rho1 (above threshold)", "rho2 (below threshold)"])},
        title=f"Enders-Siklos {model.upper()} threshold cointegration test: {y} on "
        f"{', '.join(names)}", test="enders_siklos", model=model, threshold=threshold, tau=tau,
        threshold_candidates=candidates, lags=p, lag_method=criterion or "fixed", trim=float(trim),
        phi=stats["phi"], t_max=stats["t_max"], symmetry_F=stats["symmetry_F"],
        symmetry_p_value=symmetry_p, critical_values=None, p_value=None,
        nobs=stats["nobs"], df_resid=stats["df_resid"], regime_counts=list(regime_counts),
        cointegrating_vector={"Intercept": float(first.beta[0] + means[0] - first.beta[1:] @ means[1:]),
                              **dict(zip(names, first.beta[1:].tolist(), strict=True))},
        inference_status=status, null="No cointegration (Phi, t-Max); symmetric adjustment (F)",
        dtype="float64", device="cpu",
        source="Enders and Siklos (2001) J. Business & Economic Statistics 19",
        notes=["Phi and t-Max have nonstandard null distributions; no p-value is reported."])
