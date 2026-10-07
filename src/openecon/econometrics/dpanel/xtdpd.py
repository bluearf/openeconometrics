"""Dynamic panel-data GMM: Stata's xtabond2 / xtdpd, with xtabond and xtdpdsys wrappers.

``fit_xtdpd`` is the registry entry point; ``xtdpd``, ``xtabond`` and
``xtdpdsys`` are the keyword-only public functions (``oe.xtdpd`` ...). The
estimator itself lives in ``model.py`` (assembly, GMM steps, tests) on top of
the tensor kernels in ``structure.py``, ``instruments.py`` and ``kernels.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.econometrics.dpanel.model import assemble, estimate, read_options
from openecon.models import ModelSpec, ResultBundle


def fit_xtdpd(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point: Arellano-Bond / Blundell-Bond GMM for one validated spec."""
    opts = read_options(spec)
    frame = ModelFrame(spec, data, extra_columns=opts.columns)
    frame.sort_panel()
    model = assemble(frame, opts)
    result = estimate(model, opts)
    keep = torch.zeros(frame.n, dtype=torch.bool)
    keep[result.sample_rows] = True
    frame.restrict(keep)
    steps = "Two-step" if opts.twostep else "One-step"
    kind = "system" if opts.system else "difference"
    nobs = int(keep.sum())
    inference = {**result.info,
                 "residual_definition": ("level equation: y minus x'b (includes the panel "
                                         "effect)" if opts.system else
                                         f"transformed equation ({model.transform.name})")}
    return build_result(
        frame, terms=model.terms, params=result.beta, covariance=result.covariance,
        title=f"{steps} {kind} GMM (dynamic panel data)", use_t=opts.small,
        df_inference=result.df_inference, df_resid=result.df_inference,
        metrics=result.metrics, fitted=result.fitted, observed=result.observed,
        solver="gmm_qr", inference=inference, tests=result.tests, extra=result.extra,
        nobs=nobs,
        provenance={"transformation": model.transform.name, "steps": 2 if opts.twostep else 1,
                    "equations": kind, "h": opts.h,
                    "weight_matrix_inverse": "Moore-Penrose inverse of the correlation-scale "
                                             "moment matrix (ordinary inverse when full rank)",
                    "derivatives": "closed form (linear GMM)"},
    )


def _covariance(robust: bool | None, covariance: str | None) -> str | None:
    if robust is None:
        return covariance
    if not isinstance(robust, bool):
        raise AnalysisError("invalid_spec", "robust must be True, False or None.")
    wanted = "robust" if robust else "nonrobust"
    if covariance is not None and covariance != wanted:
        raise AnalysisError("invalid_spec", f"robust={robust} contradicts "
                            f"covariance='{covariance}'; give only one of them.")
    return wanted


def _groups(value: Any, kind: str) -> list[dict] | None:
    """Copy instrument groups into plain JSON values (lists, not tuples)."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnalysisError("invalid_spec", f"{kind} must be a list of instrument groups, for "
                            f"example {kind}=[{{'columns': ['y'], 'lags': [2, None]}}].")
    groups = []
    for entry in value:
        if not isinstance(entry, dict):
            raise AnalysisError("invalid_spec", f"Every {kind} group must be a dictionary.")
        copied = {}
        for key, item in entry.items():
            if isinstance(item, (tuple, list)):
                item = list(item)
            elif isinstance(item, str) and key == "columns":
                item = [item]
            copied[key] = item
        groups.append(copied)
    return groups


def xtdpd(*, data: Any, y: str, x: Sequence[str] | None = None, panel: str, time: str,
          lags: int = 1, gmm: Sequence[dict] | None = None, iv: Sequence[dict] | None = None,
          system: bool = False, twostep: bool = False, robust: bool | None = None,
          orthogonal: bool = False, collapse: bool = False, small: bool = False,
          constant: bool = True, time_dummies: bool = False, h: int = 3, artests: int = 2,
          covariance: str | None = None, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Arellano-Bond difference GMM and Blundell-Bond system GMM (Stata's ``xtabond2``).

    Model
        ``y_it = a_1 y_i,t-1 + ... + a_p y_i,t-p + x_it'b + u_i + e_it`` for panels
        ``i`` and integer periods ``t``. The panel effect ``u_i`` is removed by
        first differences (default) or forward orthogonal deviations
        (``orthogonal=True``); lagged levels instrument the transformed equation.
        System GMM (``system=True``) adds the equation in levels, instrumented by
        lagged differences (Arellano-Bover 1995; Blundell-Bond 1998).

    Estimator
        Stacking the transformed rows (and, in system GMM, the level rows) of panel
        ``i`` into ``X_i``, ``y_i`` with instruments ``Z_i``:
        one-step ``b1 = (X'Z W1 Z'X)^-1 X'Z W1 Z'y`` with ``W1 = (sum_i Z_i'H Z_i)^-1``;
        two-step (``twostep=True``) replaces W1 by ``W2 = (sum_i Z_i'e1_i e1_i'Z_i)^-1``
        built from the one-step residuals. ``H`` (option ``h``) is xtabond2's
        ``h(3)`` by default: the covariance of the stacked errors under i.i.d.
        errors -- 2 on the diagonal and -1 between adjacent first differences, the
        identity for orthogonal deviations and for levels, and the
        transformed-level cross covariances; ``h=2`` sets the cross blocks to zero
        and ``h=1`` uses ``H = I`` (one-step difference GMM is then 2SLS on the
        transformed data). Singular weight matrices (more instruments than
        panels) use a generalized inverse, with a warning. In system GMM the
        regressors and the outcome are centred at their level-equation means
        internally and the constant is mapped back exactly.

    Parameters
        data: table with the variables. y: dependent variable. x: regressors
        (contemporaneous; include lagged regressors as their own columns).
        panel, time: panel identifier and integer period (gaps allowed;
        differences need consecutive periods, orthogonal deviations use every
        later available observation). lags: number of lagged dependent variables
        (terms ``L1.y`` ... ``Lp.y``). gmm: GMM-style groups (xtabond2
        ``gmmstyle``): ``{"columns": [...], "lags": [lo, hi], "equation":
        "diff"|"level"|"both", "collapse": bool}``; the transformed equation gets
        the levels dated ``t-lo .. t-hi`` (``hi=None``: all), one column per period
        and lag (or per lag when collapsed); with ``"both"`` (default) the level
        equation of system GMM also gets the difference at lag ``lo-1`` (lag 0 when
        ``lo=0``); ``"level"`` gives only the level equation the differences at lags
        ``lo .. hi`` (xtabond2's ``eq(level)``: lags of the difference itself).
        Default: ``[{"columns": [y], "lags": [2, None]}]``. iv: IV-style groups
        (``ivstyle``): ``{"columns": [...], "equation": ..., "passthru": bool}``,
        transformed like the regressors in the transformed equation (one shared
        column with ``"both"``; ``passthru`` keeps the level dated like the
        transformed row and is not allowed with ``"both"`` in system GMM).
        Default: ``[{"columns": x, "equation": "diff"}]`` over the regressors that
        no gmm group instruments (strictly exogenous regressors).
        system: add the level equation. twostep: two-step GMM. robust /
        covariance: ``"robust"`` gives the sandwich clustered on panels (one-step,
        no finite-sample factor) or the Windmeijer (2005) corrected two-step
        covariance; ``"nonrobust"`` (default) gives ``sigma2 (X'Z W1 Z'X)^-1``
        (one-step) or ``(X'Z W2 Z'X)^-1`` (two-step). orthogonal: forward orthogonal
        deviations (dated ``t+1`` as in xtabond2). collapse: collapse every GMM
        group that does not say otherwise. small: t and F statistics (xtabond2):
        one-step nonrobust ``V N/(N-k)`` with ``N-k`` degrees of freedom; robust or
        two-step ``V G/(G-1) (N-1)/(N-k)`` with ``G`` (difference GMM) or ``G-1``
        (system GMM with a constant) degrees of freedom. constant: in system GMM an
        ``Intercept`` in the level equation (also an instrument there); it
        differences out of difference GMM. time_dummies: period dummies as
        regressors and IV-style instruments (transformed equation in difference
        GMM, level equation in system GMM). h: 1, 2 or 3 (see above). artests:
        highest Arellano-Bond test order. missing: ``"raise"`` or ``"drop"`` (rows
        with missing inputs are removed before lags and instruments are formed).
        alpha: significance level of the confidence intervals.

    Error variance (xtabond2)
        ``sigma2 = e1'e1 / (c wttot)`` from the one-step residuals of the
        transformed equation (of the level equation when ``h=1`` in system GMM),
        ``c = 2`` for first differences unless ``h=1``, ``c = 1`` otherwise;
        ``wttot`` is the number of transformed observations in system GMM with
        ``h > 1`` and the reported number of observations otherwise.

    Returns
        A ``ResultBundle``: coefficients (``Intercept``, ``L1.y``, x, time dummies);
        metrics ``n_groups``, ``n_instruments`` (columns less the rank deficiency
        of ``sum Z_i'HZ_i``, xtabond2's count), ``obs_per_group_min/avg/max``,
        ``df_model``, ``sigma_e`` (``sqrt(sigma2)`` of the reported step, times
        ``N/(N-k)`` with small), ``df_resid`` with small; tests ``ar1`` .. ``arK``
        (Arellano-Bond z on differenced residuals; homoskedastic form after
        one-step nonrobust estimation), ``sargan`` (one-step criterion / sigma2:
        not robust, not weakened by many instruments), ``hansen`` (two-step J:
        robust, weakened by many instruments; with robust or twostep), the
        difference tests ``hansen_excl_<g>`` / ``diff_hansen_<g>`` (robust or
        twostep) or ``sargan_excl_<g>`` / ``diff_sargan_<g>`` (one-step nonrobust)
        for the level instruments of system GMM (``level``) and every gmm/iv group
        (``gmm1``, ``iv1``, ``time``), and ``model`` (Wald chi2 of the slopes, or F
        with small); extra: transformation, instrument groups and counts,
        weight-matrix ranks, one-step sigma2. ``nobs`` counts transformed
        observations (difference GMM) or level observations (system).

    Stata
        ``xtabond2 y L.y x, gmm(L.y) iv(x) noleveleq robust twostep [orthogonal
        collapse small]`` is ``oe.xtdpd(data=df, y="y", x=["x"], panel="id", time="t",
        twostep=True, robust=True)`` (``gmm`` default: y lags 2 and deeper; note
        that xtabond2 defaults to system GMM, OpenEconometrics to difference GMM: drop
        ``noleveleq`` and pass ``system=True``). Stata's ``xtdpd`` with
        ``dgmmiv()/lgmmiv()/div()/liv()`` maps onto ``gmm``/``iv`` groups with
        ``equation="diff"`` or ``"level"``. EViews' panel GMM/DPD estimator
        (Arellano-Bond) corresponds to difference GMM here.

    Example
        >>> fit = oe.xtdpd(data=df, y="n", x=["w", "k"], panel="id", time="year",
        ...                gmm=[{"columns": ["n"], "lags": [2, None]},
        ...                     {"columns": ["w", "k"], "lags": [2, None]}],
        ...                iv=[], system=True, twostep=True, robust=True, collapse=True)
        >>> fit.tests["ar2"]["p_value"], fit.tests["hansen"]["p_value"]
    """
    from openecon.analysis import fit

    spec = make_spec(
        "xtdpd", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
        intercept=constant, covariance=_covariance(robust, covariance), missing=missing,
        alpha=alpha,
        options={"lags": lags, "gmm": _groups(gmm, "gmm"), "iv": _groups(iv, "iv"),
                 "system": system, "twostep": twostep, "orthogonal": orthogonal,
                 "collapse": collapse, "small": small, "time_dummies": time_dummies,
                 "h": h, "artests": artests},
    )
    return fit(spec, data=data)


def _wrapper_groups(y: str, lags: int, predetermined: list[str], endogenous: list[str],
                    maxldep: int | None, maxlags: int | None, equation: str) -> list[dict]:
    for name, value in (("maxldep", maxldep), ("maxlags", maxlags)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)
                                  or value < 1):
            raise AnalysisError("invalid_spec", f"{name} must be a positive integer.")
    if not isinstance(lags, int) or isinstance(lags, bool) or lags < 1:
        raise AnalysisError("invalid_spec", "lags must be at least 1 (the model is dynamic).")
    groups = [{"columns": [y], "lags": [2, None if maxldep is None else 1 + maxldep],
               "equation": equation}]
    if predetermined:
        groups.append({"columns": predetermined, "lags": [1, maxlags], "equation": equation})
    if endogenous:
        groups.append({"columns": endogenous,
                       "lags": [2, None if maxlags is None else 1 + maxlags],
                       "equation": equation})
    return groups


def _wrapper_robust(robust: Any) -> bool | None:
    """robust=True requests the robust covariance; False leaves the choice to ``covariance``."""
    if not isinstance(robust, bool):
        raise AnalysisError("invalid_spec", "robust must be True or False.")
    return True if robust else None


def _wrapper_columns(x: Any, predetermined: Any, endogenous: Any,
                     instruments: Any) -> tuple[list[str], list[str], list[str], list[str]]:
    x, pre = column_list(x, "x"), column_list(predetermined, "predetermined")
    endo, extra = column_list(endogenous, "endogenous"), column_list(instruments, "instruments")
    regressors = [*x, *pre, *endo]
    if len(set(regressors)) != len(regressors):
        raise AnalysisError("invalid_spec", "A variable may appear only once among x, "
                            "predetermined and endogenous.")
    return regressors, pre, endo, list(dict.fromkeys([*x, *extra]))


def xtabond(*, data: Any, y: str, x: Sequence[str] | None = None, panel: str, time: str,
            lags: int = 1, endogenous: Sequence[str] | None = None,
            predetermined: Sequence[str] | None = None,
            instruments: Sequence[str] | None = None, maxldep: int | None = None,
            maxlags: int | None = None, twostep: bool = False, robust: bool = False,
            covariance: str | None = None, time_dummies: bool = False, artests: int = 2,
            missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Arellano-Bond (1991) difference GMM, Stata's ``xtabond``.

    Model ``y_it = sum_l a_l y_i,t-l + x_it'b + w_it'c + u_i + e_it`` estimated in first
    differences. Instruments (Stata's defaults): the levels ``y_i,t-2, y_i,t-3, ...``
    of the dependent variable (GMM style, one column per period and lag; ``maxldep``
    limits them to ``y_t-2 .. y_t-1-maxldep``), the strictly exogenous ``x`` (and
    ``instruments``) as differenced IV-style instruments, predetermined variables
    ``w`` with levels dated ``t-1`` and earlier and endogenous ones with ``t-2`` and
    earlier (``maxlags`` lags each). One-step GMM with xtabond2's ``H`` (2 on the
    diagonal, -1 off it); ``twostep=True`` for two-step; ``robust=True`` gives the
    one-step sandwich clustered on panels or the Windmeijer-corrected two-step
    covariance. z statistics; the result reports the Arellano-Bond AR(1)..AR(artests)
    tests, the Sargan test (and Hansen with robust or twostep) and the
    difference-in-Sargan/Hansen tests of each instrument group.

    This builds the instrument groups and calls ``oe.xtdpd`` (estimator ``xtdpd``);
    see its documentation for the formulas and every returned statistic. As in
    xtabond2, no constant is estimated in difference GMM (Stata's ``xtabond``
    reports one); ``time_dummies=True`` adds period dummies (differenced IV-style
    instruments).

    Stata: ``xtabond y x, lags(1) pre(w) endogenous(v) maxldep(3) twostep
    vce(robust)`` is ``oe.xtabond(data=df, y="y", x=["x"], panel="id", time="t",
    predetermined=["w"], endogenous=["v"], maxldep=3, twostep=True, robust=True)``.

    Example
        >>> fit = oe.xtabond(data=df, y="n", x=["w", "k"], panel="id", time="year",
        ...                  lags=2, twostep=True, robust=True)
        >>> print(fit.summary())
    """
    regressors, pre, endo, exogenous = _wrapper_columns(x, predetermined, endogenous,
                                                        instruments)
    gmm = _wrapper_groups(y, lags, pre, endo, maxldep, maxlags, "diff")
    iv = [{"columns": exogenous, "equation": "diff"}] if exogenous else []
    return xtdpd(data=data, y=y, x=regressors, panel=panel, time=time, lags=lags, gmm=gmm,
                 iv=iv, system=False, twostep=twostep, robust=_wrapper_robust(robust),
                 covariance=covariance,
                 constant=False, time_dummies=time_dummies, artests=artests, missing=missing,
                 alpha=alpha)


def xtdpdsys(*, data: Any, y: str, x: Sequence[str] | None = None, panel: str, time: str,
             lags: int = 1, endogenous: Sequence[str] | None = None,
             predetermined: Sequence[str] | None = None,
             instruments: Sequence[str] | None = None, maxldep: int | None = None,
             maxlags: int | None = None, twostep: bool = False, robust: bool = False,
             covariance: str | None = None, constant: bool = True, time_dummies: bool = False,
             artests: int = 2, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Blundell-Bond (1998) system GMM, Stata's ``xtdpdsys``.

    The ``xtabond`` model and instruments for the differenced equation, plus the
    equation in levels instrumented by lagged differences: ``D.y_i,t-1`` for the
    dependent variable, ``D.w_it`` for predetermined and ``D.v_i,t-1`` for endogenous
    variables (xtabond2's lag ``lo-1`` rule), and the constant (an ``Intercept`` in
    the level equation, unless ``constant=False``). Strictly exogenous ``x`` and
    ``instruments`` instrument the differenced equation. One-step GMM uses
    xtabond2's default ``h(3)`` weight matrix; ``twostep`` and ``robust`` as in
    ``oe.xtabond`` (Windmeijer correction for two-step robust).

    This builds the instrument groups and calls ``oe.xtdpd(system=True)``; see its
    documentation for the formulas and the returned statistics (Arellano-Bond AR
    tests, Sargan, Hansen and the difference-in-Hansen test of the level
    instruments).

    Stata: ``xtdpdsys y x, lags(1) pre(w) twostep vce(robust)`` is
    ``oe.xtdpdsys(data=df, y="y", x=["x"], panel="id", time="t", predetermined=["w"],
    twostep=True, robust=True)``.

    Example
        >>> fit = oe.xtdpdsys(data=df, y="n", x=["w", "k"], panel="id", time="year",
        ...                   twostep=True, robust=True)
        >>> fit.tests["diff_hansen_level"]
    """
    regressors, pre, endo, exogenous = _wrapper_columns(x, predetermined, endogenous,
                                                        instruments)
    gmm = _wrapper_groups(y, lags, pre, endo, maxldep, maxlags, "both")
    iv = [{"columns": exogenous, "equation": "diff"}] if exogenous else []
    return xtdpd(data=data, y=y, x=regressors, panel=panel, time=time, lags=lags, gmm=gmm,
                 iv=iv, system=True, twostep=twostep, robust=_wrapper_robust(robust),
                 covariance=covariance,
                 constant=constant, time_dummies=time_dummies, artests=artests,
                 missing=missing, alpha=alpha)
