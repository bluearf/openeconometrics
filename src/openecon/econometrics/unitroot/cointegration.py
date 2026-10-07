"""Engle-Granger residual-based cointegration test for single series."""

from __future__ import annotations

from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot import critical
from openecon.econometrics.unitroot.common import (
    CRITICAL_NAMES, adf_design, check_choice, check_count, check_flag, column_names,
    critical_columns, deterministic_columns, load_series, regression_table, uncentre,
)
from openecon.econometrics.unitroot.common import regress as ols
from openecon.econometrics.unitroot.series import adf_lag_choice, schwert_maxlag
from openecon.engines.linalg import collinear_columns

_CASES = {"none": "n", "constant": "c", "trend": "ct", "quadratic": "ctt"}


def egranger(data: Any, y: str, x: list[str], *, time: str | None = None, lags: int | str = 0,
             trend: str = "constant", maxlag: int | None = None, regress: bool = False,
             ecm: bool = False):
    """Engle-Granger two-step test for cointegration.

    Model and estimator
    -------------------
    Step 1, the cointegrating regression by OLS on all T observations:

        y_t = d_t + x_t'b + e_t,

    with ``d_t`` empty (``trend="none"``), a constant (default), a constant and
    a linear trend (``"trend"``) or additionally a squared trend
    (``"quadratic"``). Step 2, the Dickey-Fuller regression of the residuals
    WITHOUT deterministic terms on t = k+2..T:

        Delta e_t = rho e_{t-1} + sum_{j=1..k} c_j Delta e_{t-j} + v_t.

    The statistic is the t ratio of ``rho``. H0: no cointegration (the
    residuals have a unit root); small (very negative) values reject.

    Parameters
    ----------
    data : DataFrame, mapping of columns, or list of records.
    y : outcome series; x : list of the other I(1) series (at most 5, the range
        of MacKinnon's tables).
    time : optional time column (rows are sorted by it; no gaps).
    lags : lagged residual differences k (default 0), or ``"aic"`` / ``"bic"`` /
        ``"t"`` to choose k from 0..maxlag on a common sample.
    trend : deterministic terms of the cointegrating regression (see above).
    maxlag : largest order of the automatic choice; default Schwert's
        ``int(12 (T/100)^(1/4))``.
    regress : add the residual Dickey-Fuller regression table.
    ecm : add the second-step error-correction model

            Delta y_t = c + a e_{t-1}
                        + sum_{j=1..k} (g_j Delta y_{t-j} + h_j' Delta x_{t-j}) + u_t,

        estimated by OLS with classical t inference (valid when the series are
        cointegrated). The constant is omitted with ``trend="none"``.

    Reference distribution
    ----------------------
    The statistic does not have the Dickey-Fuller distribution because the
    residuals are estimated; it depends on N = 1 + (number of regressors) and
    on the deterministic terms. p-value: MacKinnon (1994, Tables 3-4)
    approximate asymptotic p-value. Critical values: MacKinnon (2010, Tables
    2-4) response surfaces evaluated at the number of observations of the
    Dickey-Fuller regression; for ``trend="none"`` with regressors, where the
    2010 tables have no entry, the asymptotic critical values implied by the
    1994 distribution function.

    Collinear regressors are omitted left to right (Stata style), listed in
    ``attrs["omitted"]``, and do not count towards N. The sample must hold at
    least six observations more than the parameters of the cointegrating
    regression.

    Returns
    -------
    A ``TableSet`` with the tables ``test`` (one row: ``statistic``,
    ``p_value``, three critical values), ``cointegrating_regression``
    (coefficients; their OLS standard errors are shown but do NOT have a
    standard distribution under cointegration), and optionally
    ``adf_regression`` and ``ecm``. ``attrs``: ``statistic``, ``p_value``,
    ``critical_values``, ``lags``, ``trend``, ``nobs`` (Dickey-Fuller
    regression), ``n_series``, ``cointegrating_vector`` (term -> coefficient),
    ``adjustment`` (the ECM coefficient of e_{t-1}, with ``ecm=True``),
    ``omitted``, ``label``, ``notes``.

    Stata: ``egranger y x1 x2, lags(2) trend regress ecm`` (Schaffer's
    ``egranger``). EViews: Cointegration Test / Engle-Granger (tau statistic).

    Example
    -------
    >>> result = oe.egranger(df, "consumption", ["income"], lags=1)
    >>> result.attrs["statistic"], result.attrs["p_value"]
    >>> result["cointegrating_regression"]
    """
    check_choice(trend, "trend", tuple(_CASES))
    check_flag(regress, "regress")
    check_flag(ecm, "ecm")
    if not isinstance(y, str):
        raise AnalysisError("invalid_spec", "y must be the name of one numeric column.")
    names = column_names(x, "x")
    if not names:
        raise AnalysisError("invalid_spec", "The cointegration test needs at least one "
                            "regressor in x.")
    if not isinstance(lags, str):
        lags = check_count(lags, "lags")
    values, _ = load_series(data, [y, *names], time)
    total = values.shape[0]
    case = _CASES[trend]
    smallest = 7 + {"n": 0, "c": 1, "ct": 2, "ctt": 3}[case]     # one regressor kept
    if total < smallest:
        raise AnalysisError("insufficient_observations", "The cointegration test needs at "
                            f"least {smallest} observations (6 more than the parameters of "
                            f"the cointegrating regression); {total} given.")
    # With a constant the regression is invariant to the level of every series: measure
    # them from their first observation so that a large level cannot look collinear with
    # the constant. The intercept is mapped back below.
    origins = values[0].clone() if case != "n" else torch.zeros(values.shape[1],
                                                                dtype=torch.float64)
    values = values - origins
    target = values[:, 0].contiguous()
    extra, extra_names = deterministic_columns(total, case)
    block = values[:, 1:]
    # Stata-style omission of collinear regressors, screened after the deterministic terms.
    screen = torch.cat([torch.stack(extra, dim=1), block], dim=1) if extra else block
    kept, dropped = kernel_call(collinear_columns, screen)
    offset = len(extra)
    if any(index < offset for index in dropped):
        raise AnalysisError("insufficient_observations", "The sample is too short for the "
                            "deterministic terms of the cointegrating regression.")
    omitted = [names[index - offset] for index in dropped]
    names = [names[index - offset] for index in kept if index >= offset]
    if not names:
        raise AnalysisError("collinear_regressors", "Every regressor is constant or collinear "
                            "with the deterministic terms.")
    n_series = 1 + len(names)
    if n_series > critical.MAX_SERIES:
        raise AnalysisError("too_many_regressors", "MacKinnon's tables cover at most "
                            f"{critical.MAX_SERIES - 1} regressors; {len(names)} given.")
    columns = [values[:, 1 + column_index] for column_index in
               (index - offset for index in kept if index >= offset)]
    design = torch.stack([*columns, *extra], dim=1)
    terms = [*names, *extra_names]
    if total < len(terms) + 6:
        raise AnalysisError("insufficient_observations", "The cointegration test needs at "
                            f"least {len(terms) + 6} observations for {len(terms)} parameter(s) "
                            f"in the cointegrating regression; {total} given.")
    first = ols(design, target, "The cointegrating regression")
    resid = first.resid.contiguous()
    if case != "n":
        shifts = {position: float(origins[1 + column_index]) for position, column_index in
                  enumerate(index - offset for index in kept if index >= offset)}
        first = uncentre(first, len(terms) - 1, shifts, constant=float(origins[0]))
    centered = target - target.mean()
    scale = float(centered.square().sum()) if case != "n" else float(target.square().sum())
    if not first.ssr > 1e-22 * max(scale, 1e-300):
        raise AnalysisError("perfect_fit", "y is an exact linear combination of the "
                            "regressors; the cointegration test is undefined.")
    if isinstance(lags, str):
        check_choice(lags, "lags", ("aic", "bic", "t"))
        limit = schwert_maxlag(total, 2) if maxlag is None else check_count(maxlag, "maxlag")
        if total - limit - 1 <= limit + 2:
            raise AnalysisError("invalid_lags", f"maxlag={limit} is too large for {total} "
                                "observations.")
        method, lags = lags, adf_lag_choice(resid, "n", lags, limit)
    else:
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag applies only with lags='aic', 'bic' "
                                "or 't'.")
        method = "fixed"
    adf_x, adf_y, adf_terms = adf_design(resid, lags, "n", name="e")
    second = ols(adf_x, adf_y, "The residual Dickey-Fuller regression")
    if second.exact or not second.s2 > 0.0:
        raise AnalysisError("perfect_fit", "The residual Dickey-Fuller regression fits exactly; "
                            "the test is undefined.")
    statistic = float(second.beta[0] / second.se[0])
    p_value = critical.mackinnon_p(statistic, case, n_series)
    notes = ["MacKinnon (1994) approximate p-value for N = "
             f"{n_series} series ({trend} case)."]
    crit = critical.mackinnon_critical(case, n_series, second.nobs)
    if crit is None:
        crit = critical.mackinnon_asymptotic_critical(case, n_series)
        notes.append("Critical values are asymptotic (implied by MacKinnon 1994): the 2010 "
                     "response surfaces do not cover the no-constant case with regressors.")
    else:
        notes.append("MacKinnon (2010) finite-sample critical values.")
    if omitted:
        notes.append(f"Omitted because of collinearity: {', '.join(omitted)}.")
    notes.append("The standard errors of the cointegrating regression are not valid for "
                 "inference; they are shown for reference only.")
    attrs = {
        "title": f"Engle-Granger test for cointegration of {y} with {', '.join(names)}",
        "test": "egranger", "statistic": statistic, "p_value": p_value,
        "distribution": "Engle-Granger (MacKinnon)", "critical_values": crit, "lags": lags,
        "lag_method": method, "trend": trend, "nobs": second.nobs, "nobs_first_step": total,
        "n_series": n_series,
        "cointegrating_vector": dict(zip(terms, first.beta.tolist(), strict=True)),
        "omitted": omitted, "label": "Engle-Granger tau (H0: no cointegration)", "notes": notes,
    }
    result = {
        "test": table([[statistic, p_value, *critical_columns(crit)]],
                      columns=["statistic", "p_value", *CRITICAL_NAMES], index=["Z(t)"]),
        "cointegrating_regression": regression_table(terms, first),
    }
    if regress:
        result["adf_regression"] = regression_table(adf_terms, second)
    if ecm:
        start = lags + 1                               # first usable difference index
        dy = values[1:] - values[:-1]                  # [T-1, 1 + all x], row s = Delta at s+1
        kept_columns = [0, *(1 + index - offset for index in kept if index >= offset)]
        dy = dy[:, kept_columns]
        rows = total - 1 - lags
        pieces = [resid[lags:total - 1]]
        ecm_terms = ["L1.e"]
        for j in range(1, lags + 1):
            for column, name in enumerate([y, *names]):
                pieces.append(dy[lags - j:total - 1 - j, column])
                ecm_terms.append(f"LD.{name}" if j == 1 else f"L{j}D.{name}")
        if case != "n":
            pieces.append(torch.ones(rows, dtype=torch.float64))
            ecm_terms.append("Intercept")
        third = ols(torch.stack(pieces, dim=1), dy[start - 1:, 0].contiguous(),
                    "The error-correction regression")
        result["ecm"] = regression_table(ecm_terms, third)
        attrs["adjustment"] = float(third.beta[0])
    return TableSet(result, **attrs)
