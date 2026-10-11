"""Bayer-Hanck combined non-cointegration test and its ECM components.

``bayer_hanck`` follows the authors' Stata implementation ``bayerhanck.ado``
(version 0.9, 5 June 2009; SSC). Its four components, each a test of the null
of no cointegration for y on x with ``lags`` augmentation lags L and
deterministic case none / constant / trend, are

* Engle-Granger: OLS of y on x (and the deterministic terms) on all T
  observations; the t ratio of rho in De_t = rho e_(t-1) + sum_(i<=L) c_i
  De_(t-i) + v_t (no constant; Stata ``dfuller, lags(L) nocon``);
* Johansen: the maximum-eigenvalue statistic for rank 0 of the VECM with L
  lagged differences (Stata ``vecrank, lags(L+1) trend(none|constant|trend)``:
  no deterministic terms, unrestricted constant, unrestricted trend);
* Banerjee, Dolado and Mestre (1998) and Boswijk (1994): with W_t = (Dx_t,
  Dy_(t-i), Dx_(t-i), i = 1..L) and the deterministic terms d_t, the
  residuals of Dy_t and of every lagged level z_(t-1) = (y, x)_(t-1) on
  (W_t, d_t) give the partialled ECM regression

      r_Dy,t = a_y r_y,t + a_x' r_x,t (+ d_t) + u_t,      t = L+2..T;

  Banerjee's statistic is the t ratio of a_y, Boswijk's the Wald statistic
  b'V^-1 b of all coefficients with V = s^2 (X'X)^-1 (the deterministic
  coefficients are zero by construction).

Component p-values are the authors' empirical ones: the share of their 10,000
simulated null draws beyond the statistic (lower tail for EG and Banerjee,
upper tail for Johansen and Boswijk) plus 1e-12. The Fisher statistics
EG-J = -2 (ln p_EG + ln p_J) and EG-J-Ba-Bo = -2 sum ln p are compared with
the authors' critical values; no p-value of the combination is published.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot import bayer_hanck_null as null
from openecon.econometrics.unitroot import coint_breaks_tables as tables
from openecon.econometrics.unitroot.common import (
    adf_design, check_choice, check_count, column_names, load_series, regress,
)
from openecon.econometrics.var.johansen import johansen

TRENDS = ("none", "constant", "trend")
_MAX_OBS = 1_000_000


def _deterministic(rows: int, trend: str, start: int) -> list[Tensor]:
    """Constant and trend columns for observations start..start+rows-1 (1-based t)."""
    if trend == "none":
        return []
    columns = [torch.ones(rows, dtype=torch.float64)]
    if trend == "trend":
        columns.append(torch.arange(start, start + rows, dtype=torch.float64))
    return columns


def _prepare(data: Any, y: str, x: Any, time: str | None, lags: Any, trend: str,
             what: str) -> tuple[Tensor, list[str], int]:
    check_choice(trend, "trend", TRENDS)
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", f"{what} needs at least one regressor in x.")
    if len(names) > 11:
        raise AnalysisError("critical_values_unavailable", f"{what} is calibrated for 1 to 11 "
                            "regressors (the authors' null distributions).")
    lags = check_count(lags, "lags")
    values, _ = load_series(data, [y, *names], time)
    if values.shape[0] > _MAX_OBS:
        raise AnalysisError("workspace_limit", f"{what} admits at most {_MAX_OBS:,} "
                            "observations.")
    for index, name in enumerate([y, *names]):
        if not bool((values[:, index] != values[0, index]).any()):
            raise AnalysisError("constant_series", f"Column '{name}' is constant.")
    # Deviations from the first observation keep the regressions well conditioned;
    # the statistics with deterministic terms are invariant to them, and without
    # deterministic terms the raw levels are used.
    if trend != "none":
        values = values - values[0]
    needed = 2 * (len(names) + 1) * (lags + 1) + 6
    if values.shape[0] - lags - 1 < needed:
        raise AnalysisError("insufficient_observations", f"{what} with {lags} lag(s) and "
                            f"{len(names)} regressor(s) needs more than {needed + lags + 1} "
                            "observations.")
    return values, names, lags


def _engle_granger(values: Tensor, lags: int, trend: str) -> float:
    total = values.shape[0]
    design = torch.stack([*values[:, 1:].T, *_deterministic(total, trend, 1)], dim=1)
    first = regress(design, values[:, 0], "The Engle-Granger cointegrating regression")
    if first.exact:
        raise AnalysisError("perfect_fit", "y is an exact linear combination of the regressors.")
    adf_x, adf_y, _ = adf_design(first.resid.contiguous(), lags, "n", name="e")
    second = regress(adf_x, adf_y, "The residual Dickey-Fuller regression")
    return float(second.beta[0] / second.se[0])


def _johansen_max(values: Tensor, names: list[str], lags: int, trend: str) -> float:
    result = kernel_call(johansen, values, names, lags + 1, trend)
    return result.maximum(0)


def ecm_statistics(values: Tensor, lags: int, trend: str) -> tuple[float, float, int]:
    """(Banerjee t, Boswijk Wald, observations) of the partialled ECM regression."""
    total, columns = values.shape
    diff = values[1:] - values[:-1]                       # row s = D at observation s+2
    rows = total - lags - 1
    start = lags + 1                                      # 0-based first observation used
    pieces = [diff[start - 1:, 1:]]                       # contemporaneous Dx_t
    for i in range(1, lags + 1):
        pieces.append(diff[start - 1 - i:total - 1 - i])  # D(y, x)_(t-i)
    det = _deterministic(rows, trend, start + 1)
    w = torch.cat([*pieces, *(column[:, None] for column in det)], dim=1)
    levels = values[start - 1:total - 1]                  # z_(t-1)
    what = "The partialled error-correction regressions"
    resid_levels = torch.stack([regress(w, levels[:, j].contiguous(), what).resid
                                for j in range(columns)], dim=1)
    resid_dy = regress(w, diff[start - 1:, 0].contiguous(), what).resid
    final_design = torch.cat([resid_levels, *(column[:, None] for column in det)], dim=1)
    final = regress(final_design, resid_dy, "The error-correction test regression")
    if final.exact or not final.s2 > 0:
        raise AnalysisError("perfect_fit", "The error-correction regression fits exactly.")
    banerjee = float(final.beta[0] / final.se[0])
    fitted = final_design @ final.beta
    boswijk = float(fitted.square().sum()) / final.s2
    return banerjee, boswijk, rows


def bayer_hanck(data: Any, y: str, x: list[str], *, time: str | None = None, lags: int = 1,
                trend: str = "constant") -> TableSet:
    """Bayer-Hanck Fisher combinations of four non-cointegration tests.

    Parameters
    ----------
    data, y, x, time : series as in ``oe.egranger`` (complete, regularly spaced).
    lags : augmentation lags L >= 0 used in every component (default 1, as the
        authors' ado; the Johansen VECM has L lagged differences).
    trend : ``"none"``, ``"constant"`` (default) or ``"trend"`` (see the
        module docstring for each component).

    Returns
    -------
    TableSet with ``components`` (statistic, p-value and tail of EG, Johansen
    lambda-max, Banerjee t and Boswijk Wald) and ``test`` (EG-J and
    EG-J-Ba-Bo Fisher statistics with the authors' 1/5/10% critical values;
    ``p_value`` is None). H0: no cointegration; large Fisher statistics
    reject. K = 1..11 regressors.

    Stata: ``bayerhanck y, rhs(x1 x2) lags(1) trend(constant) crit(5)``.

    Example
    -------
    >>> oe.bayer_hanck(df, "consumption", ["income", "wealth"], lags=2).attrs["statistics"]
    """
    what = "The Bayer-Hanck test"
    values, names, lags = _prepare(data, y, x, time, lags, trend, what)
    n_variables = len(names) + 1
    eg = _engle_granger(values, lags, trend)
    jo = _johansen_max(values, [y, *names], lags, trend)
    banerjee, boswijk, rows = ecm_statistics(values, lags, trend)
    statistics = [eg, jo, banerjee, boswijk]
    p_values = [null.component_p_value(value, j, n_variables, trend)
                for j, value in enumerate(statistics)]
    fisher_ej = -2.0 * (math.log(p_values[0]) + math.log(p_values[1]))
    fisher_all = -2.0 * sum(math.log(p) for p in p_values)
    critical = tables.bayer_hanck_critical(trend, len(names))
    component_rows = [[stat, p, "lower" if lower else "upper"]
                      for stat, p, lower in zip(statistics, p_values, null.LOWER_TAIL,
                                                strict=True)]
    test_rows = [[value, None, *(critical[name][level] for level in tables.LEVELS)]
                 for name, value in (("EG-J", fisher_ej), ("EG-J-Ba-Bo", fisher_all))]
    return TableSet(
        {"components": table(component_rows, columns=["statistic", "p_value", "rejection_tail"],
                             index=list(null.COMPONENTS)),
         "test": table(test_rows, columns=["statistic", "p_value", "critical_1pct",
                                           "critical_5pct", "critical_10pct"],
                       index=["EG-J", "EG-J-Ba-Bo"])},
        title=f"Bayer-Hanck test for cointegration of {y} with {', '.join(names)}",
        test="bayer_hanck", statistics={"EG-J": fisher_ej, "EG-J-Ba-Bo": fisher_all},
        components=dict(zip(null.COMPONENTS, statistics, strict=True)),
        component_p_values=dict(zip(null.COMPONENTS, p_values, strict=True)),
        p_value=None, critical_values=critical, lags=lags, trend=trend, nobs=values.shape[0],
        ecm_nobs=rows, regressors=len(names),
        inference_status="component p-values from the authors' 10,000 null draws; Fisher "
        "critical values from bayerhanck.ado; the combination p-value is not published",
        null="No cointegration", dtype="float64", device="cpu",
        source="Bayer and Hanck (2013) J. Time Series Analysis 34; bayerhanck.ado 0.9 and "
        "NullDistr.dta (SSC)",
        notes=["Johansen component: maximum-eigenvalue statistic for rank 0 (as the authors' "
               "code).",
               "Large Fisher statistics reject the null of no cointegration."])


def ecm_cointegration(data: Any, y: str, x: list[str], *, time: str | None = None,
                      test: str = "banerjee", lags: int = 1,
                      trend: str = "constant") -> TableSet:
    """Single-equation error-correction tests of no cointegration (Banerjee et al.; Boswijk).

    The statistics are the Bayer-Hanck components (see ``bayer_hanck``):
    ``test="banerjee"`` is the t ratio of the lagged-level coefficient of y in
    the partialled ECM regression (Banerjee, Dolado and Mestre 1998; reject for
    small values); ``test="boswijk"`` is Boswijk's (1994) Wald statistic on all
    lagged-level coefficients (reject for large values). The p-value is the
    tail share of the Bayer-Hanck null draws for 1..11 regressors and the
    chosen deterministic case (+1e-12, as the authors).
    """
    check_choice(test, "test", ("banerjee", "boswijk"))
    what = "The error-correction cointegration test"
    values, names, lags = _prepare(data, y, x, time, lags, trend, what)
    banerjee, boswijk, rows = ecm_statistics(values, lags, trend)
    component = 2 if test == "banerjee" else 3
    statistic = banerjee if test == "banerjee" else boswijk
    p_value = null.component_p_value(statistic, component, len(names) + 1, trend)
    label = "Banerjee-Dolado-Mestre t" if test == "banerjee" else "Boswijk Wald"
    return TableSet(
        {"test": table([[statistic, p_value, rows]], columns=["statistic", "p_value", "nobs"],
                       index=[label])},
        title=f"{label} test for cointegration of {y} with {', '.join(names)}",
        test=f"ecm_{test}", statistic=statistic, p_value=p_value, lags=lags, trend=trend,
        nobs=rows, regressors=len(names), rejection_tail="lower" if component == 2 else "upper",
        inference_status="p-value from the Bayer-Hanck authors' 10,000 simulated null draws",
        null="No cointegration", dtype="float64", device="cpu")
