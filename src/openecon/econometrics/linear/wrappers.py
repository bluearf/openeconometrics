"""Stata-named entry points ``regress`` and ``newey`` that delegate to ``oe.ols``.

Plain OLS/WLS lives in ``openecon.linear_ols`` (reached as ``oe.ols``, estimator
``'ols'``), which owns the HC/cluster/HAC/bootstrap covariances and the
post-estimation commands (``oe.test``, ``oe.lincom``, ``oe.nlcom``,
``oe.predict``, ``oe.margins``). The two functions here only translate Stata's
vocabulary into ``oe.ols`` keyword arguments and return its result unchanged:

* ``regress``: ``covariance='robust'`` (Stata's ``vce(robust)``) becomes ``HC1``;
  everything else is passed through as given.
* ``newey``: ``lag`` becomes ``covariance='hac'`` with ``lags=lag`` and the
  Bartlett kernel by default; ``time`` names Stata's ``tsset`` variable.

What ``oe.ols`` cannot express raises ``AnalysisError('unsupported_option')``:

* ``panel`` for ``newey``: autocovariances within units need ``xtset`` support
  that ``oe.ols`` does not have;
* a HAC covariance without ``time``: ``oe.ols`` forms autocovariances on an
  explicit integer period index only and never assumes that rows are
  consecutive periods (Stata's ``newey`` likewise requires ``tsset`` data).

Stata's ``newey`` allows analytic weights only; frequency, sampling and
importance weights with the HAC covariance raise
``AnalysisError('unsupported_weights')`` here too, because replicated
observations have no time-series meaning.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openecon.analysis import ols
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import column_list
from openecon.models import ResultBundle


def _translate_covariance(covariance: str | None) -> str | None:
    """Stata's ``vce(robust)`` is HC1 for linear least squares."""
    return "HC1" if covariance == "robust" else covariance


def _check_hac(covariance: str | None, weight_type: str | None, time: str | None) -> None:
    """Reject the HAC requests that Stata's newey disallows or ``oe.ols`` cannot express."""
    if covariance != "hac":
        return
    if weight_type not in (None, "aweight"):
        raise AnalysisError("unsupported_weights", "Newey-West (HAC) standard errors allow "
                            "analytic weights only, as Stata's newey does: replicated or "
                            "sampling-weighted observations have no time-series ordering. Use "
                            "weight_type='aweight' or another covariance.")
    if time is None:
        raise AnalysisError("unsupported_option", "oe.ols forms HAC autocovariances on an explicit "
                            "period index and does not assume that rows are consecutive periods: "
                            "pass time='<column of integer periods>' (Stata's newey likewise "
                            "requires tsset data).")


def regress(*, data: Any, y: str, x: Sequence[str], covariance: str | None = None,
            cluster: str | Sequence[str] | None = None, weights: str | None = None,
            weight_type: str | None = None, lags: int | None = None, kernel: str | None = None,
            time: str | None = None, categorical: Sequence[str] | None = None,
            intercept: bool = True, missing: str = "drop", alpha: float = 0.05) -> ResultBundle:
    """Linear regression, Stata's ``regress``: a thin wrapper that delegates to ``oe.ols``.

    This function performs no estimation of its own. It translates Stata's
    argument names into those of :func:`openecon.analysis.ols` and returns that
    result (an OLS result with ``oe.test``, ``oe.lincom``, ``oe.predict`` and
    ``oe.margins`` post-estimation), so ``oe.regress(...)`` and the equivalent
    ``oe.ols(...)`` call give identical coefficients, covariance and metrics.

    Translation
        ``covariance='robust'`` (Stata's ``vce(robust)``) -> ``'HC1'``; ``'nonrobust'``,
        ``'HC0'``..``'HC3'``, ``'cluster'`` (with ``cluster``), ``'hac'`` (with ``lags``,
        ``kernel`` and ``time``, see :func:`newey`), ``'bootstrap'`` and ``'jackknife'``
        are passed through. ``covariance=None`` lets ``oe.ols`` choose: conventional,
        ``cluster`` when a cluster column is given, HC1 for pweights.

    Parameters
        data: DataFrame, mapping of columns or list of row records.
        y: outcome column. x: list of regressor columns (a bare string is an error).
        covariance, cluster: as above; up to four cluster columns.
        weights, weight_type: ``'aweight'``, ``'fweight'``, ``'pweight'`` or ``'iweight'``
            with ``oe.ols``'s conventions (pweights need a robust or cluster covariance;
            the HAC covariance takes aweights only, raising ``unsupported_weights``).
        lags, kernel, time: HAC lag length, kernel (``'bartlett'``, ``'parzen'``,
            ``'quadratic_spectral'``, ``'truncated'``) and integer period column. ``lags``
            and ``kernel`` are valid only with ``covariance='hac'`` (``oe.ols`` raises
            ``invalid_spec`` otherwise); ``covariance='hac'`` needs ``time`` (unique integer
            periods, gaps respected) and raises ``unsupported_option`` without it.
        categorical: regressors expanded as ``name[level]`` dummies.
        intercept: include the constant (``False`` = Stata's ``noconstant``).
        missing: ``'drop'`` (default, as ``oe.ols``) or ``'raise'``.
        alpha: significance level of the confidence intervals.

    Stata
        ``regress y x1 x2 [aw=w], vce(robust)`` is
        ``oe.regress(data=df, y='y', x=['x1', 'x2'], weights='w', weight_type='aweight',
        covariance='robust')``.

    Example
        >>> import pandas as pd, numpy as np, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=200), "g": np.repeat(range(20), 10)})
        >>> df["y"] = 1 + 2 * df.x + rng.normal(size=200)
        >>> print(oe.regress(data=df, y="y", x=["x"], cluster="g").summary())
    """
    _check_hac(covariance, weight_type, time)
    return ols(data=data, y=y, x=column_list(x, "x"),
               covariance=_translate_covariance(covariance), cluster=cluster, weights=weights,
               weight_type=weight_type, lags=lags, kernel=kernel, time=time,
               categorical=column_list(categorical, "categorical") or None, intercept=intercept,
               missing=missing, alpha=alpha)


def newey(*, data: Any, y: str, x: Sequence[str], lag: int, time: str | None = None,
          panel: str | None = None, kernel: str = "bartlett", weights: str | None = None,
          weight_type: str | None = None, categorical: Sequence[str] | None = None,
          intercept: bool = True, missing: str = "drop", alpha: float = 0.05) -> ResultBundle:
    """Regression with Newey-West standard errors, Stata's ``newey``: delegates to ``oe.ols``.

    This function performs no estimation of its own: it calls
    :func:`openecon.analysis.ols` with ``covariance='hac'``, ``lags=lag`` and the
    chosen ``kernel`` and returns that result unchanged. ``lag=0`` is White's
    estimator. There is no automatic bandwidth: ``lag`` must be given, as Stata
    requires ``lag()``; a common rule of thumb is ``floor(4 (T/100)^(2/9))``
    (``openecon.engines.covariance.newey_west_lags``).

    Parameters
        lag: lag length ``L`` (bandwidth ``L + 1``), a nonnegative integer.
        time: integer period column (Stata's ``tsset``), unique values, gaps respected.
            Required in effect: ``oe.ols`` does not assume that rows are consecutive
            periods, so omitting it raises ``AnalysisError('unsupported_option')``.
        panel: not supported by ``oe.ols`` (Stata's ``newey ..., force`` on ``xtset``
            data); giving it raises ``AnalysisError('unsupported_option')``.
        kernel: ``'bartlett'`` (default, Newey-West), ``'parzen'``,
            ``'quadratic_spectral'`` or ``'truncated'``.
        weights, weight_type: analytic weights only (Stata's newey); other weight types
            raise ``AnalysisError('unsupported_weights')``.
        data, y, x, categorical, intercept, missing, alpha: as in :func:`regress`.

    Stata
        ``tsset t`` then ``newey y x1 x2, lag(3)`` is
        ``oe.newey(data=df, y='y', x=['x1', 'x2'], lag=3, time='t')``.

    Example
        >>> import pandas as pd, numpy as np, openecon as oe
        >>> rng = np.random.default_rng(1)
        >>> df = pd.DataFrame({"t": range(120), "x": rng.normal(size=120)})
        >>> df["y"] = 0.5 * df.x + np.cumsum(rng.normal(size=120)) * 0.3
        >>> print(oe.newey(data=df, y="y", x=["x"], lag=4, time="t").summary())
    """
    if panel is not None:
        raise AnalysisError("unsupported_option", "oe.ols has no panel option, so newey cannot "
                            "form autocovariances within units (Stata's newey ..., force on "
                            "xtset data). Drop panel, or cluster on the unit instead.")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise AnalysisError("invalid_spec", "lag must be a nonnegative integer (the Newey-West "
                            "lag length; 0 gives White's estimator).")
    _check_hac("hac", weight_type, time)
    return ols(data=data, y=y, x=column_list(x, "x"), covariance="hac", lags=lag, kernel=kernel,
               time=time, weights=weights, weight_type=weight_type,
               categorical=column_list(categorical, "categorical") or None, intercept=intercept,
               missing=missing, alpha=alpha)
