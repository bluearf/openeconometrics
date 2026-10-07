"""Three-stage least squares and its variants: Stata's ``reg3``.

Model
-----
A system of structural equations ``y_i = X_i b_i + e_i`` (``i = 1..M``) whose
right-hand sides may contain endogenous variables (other equations' outcomes
or variables declared endogenous), with ``Var(e) = Sigma (x) I_N`` and a set of
exogenous variables ``Z`` (the instruments, constant included) uncorrelated
with every error.

Which variables are endogenous (Stata's rules)
    By default every equation's outcome is endogenous, ``endogenous`` adds
    right-hand-side variables that are no equation's outcome, every other
    right-hand-side variable is exogenous and ``exogenous`` adds instruments
    that appear in no equation. ``instruments`` instead gives the complete
    exogenous list: every right-hand-side variable not in it is endogenous.
    The constant is an instrument whenever an equation has a constant.

Estimators (Methods and formulas of [R] reg3)
    3sls  2SLS of every equation (``b_i = (X_i'P_Z X_i)^-1 X_i'P_Z y_i``), the
          residual covariance ``s_ij = e_i'e_j / N`` (``dfk``: ``/ sqrt((N -
          k_i)(N - k_j))``) of the structural residuals ``e_i = y_i - X_i b_i``,
          then ``b = [X'(S^-1 (x) P_Z)X]^-1 X'(S^-1 (x) P_Z) y`` with
          ``V = [X'(S^-1 (x) P_Z)X]^-1``. ``ireg3`` recomputes ``S`` from the 3SLS
          residuals until the coefficients converge.
    2sls  equation-by-equation 2SLS; implies ``dfk``, ``small`` and independent
          equations: ``V`` is block diagonal with ``s_ii (X_i'P_Z X_i)^-1``,
          ``s_ii = e_i'e_i / (N - k_i)``.
    ols   equation-by-equation OLS (every variable exogenous); implies ``dfk``,
          ``small`` and independent equations.
    sure  seemingly unrelated regression (every variable exogenous), as
          ``sureg``; ``ireg3`` iterates it.

The projections never form ``P_Z``: with ``Z`` leading the distinct columns of
the data, the projection of any column on ``Z`` is the leading block of rows
of the QR factor (``common``). Every equation must satisfy the rank condition
(``P_Z X_i`` of full column rank); otherwise ``underidentified`` is raised.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, column_list, kernel_call
from openecon.econometrics.systems import kernels
from openecon.econometrics.systems.common import (
    SystemData, build_system, check_system_role, compressed_columns, drop_columns,
    parse_equations, record_omitted,
)
from openecon.econometrics.systems.linear import (
    Prepared, assemble, dfk_divisors, prepare, restriction, shift,
)
from openecon.econometrics.systems.sureg import (
    constraints_option, equations_option, system_spec,
)
from openecon.engines.linalg import collinear_columns
from openecon.models import ModelSpec, ResultBundle

TITLES = {"3sls": "Three-stage least-squares regression", "2sls": "Two-stage least-squares "
          "regression (system)", "ols": "Equation-by-equation OLS (system)",
          "sure": "Seemingly unrelated regression (reg3)"}


def _classify(frame: ModelFrame, outcomes: list[str], rhs: list[str]) -> tuple[list[str],
                                                                                list[str]]:
    """Exogenous (instrument) and endogenous variable lists by Stata's rules."""
    endog_extra, exog_extra = frame.role("endogenous"), frame.role("exogenous")
    listed = frame.role("instruments")
    if listed and (endog_extra or exog_extra):
        raise AnalysisError("invalid_spec", "instruments gives the complete exogenous list; do "
                            "not combine it with endogenous or exogenous.")
    unused = [name for name in endog_extra if name not in rhs]
    if unused:
        raise AnalysisError("invalid_spec", f"Variables declared endogenous appear in no "
                            f"equation: {', '.join(unused)}.")
    for name in [*exog_extra, *listed]:
        if name in outcomes:
            raise AnalysisError("invalid_spec", f"'{name}' is an equation's outcome and "
                                "therefore endogenous; it cannot be an instrument.")
        if name in endog_extra:
            raise AnalysisError("invalid_spec", f"'{name}' is declared both endogenous and "
                                "exogenous.")
    if listed:
        exogenous = list(listed)
        endogenous = [name for name in rhs if name not in set(listed)]
    else:
        endogenous = [name for name in rhs if name in set(outcomes) | set(endog_extra)]
        exogenous = [name for name in rhs if name not in set(endogenous)] + \
            [name for name in exog_extra if name not in rhs]
    return list(dict.fromkeys(exogenous)), list(dict.fromkeys(endogenous))


def _screen_instruments(frame: ModelFrame, system: SystemData) -> SystemData:
    count = system.instruments
    if count == 0:
        raise AnalysisError("underidentified", "The system has no exogenous variables (no "
                            "constant and no instruments); 2SLS/3SLS are not defined.")
    columns = compressed_columns(system, range(count), system.constant)
    kept, omitted = kernel_call(collinear_columns, columns)
    if omitted:
        record_omitted(frame, [system.labels[i] for i in omitted],
                       "collinearity among the instruments")
        system = drop_columns(system, [*kept, *range(count, len(system.labels))])
    if system.rows <= system.instruments:
        raise AnalysisError("insufficient_observations", f"The system has {system.instruments} "
                            f"instruments but only {system.rows} observation(s).")
    return system


def _check_identified(prepared: Sequence[Prepared], rows: int, endogenous: set[str],
                      system: SystemData) -> None:
    for item in prepared:
        projected = item.data.design[:rows]
        _, omitted = kernel_call(collinear_columns, projected)
        if omitted:
            endog = [name for name in item.equation.regressors if name in endogenous]
            raise AnalysisError(
                "underidentified", f"Equation '{item.equation.name}' is not identified: its "
                f"{len(endog)} endogenous regressor(s) ({', '.join(endog) or 'none'}) need at "
                f"least as many excluded instruments, and the projections of its regressors "
                f"on the {rows} instrument column(s) are collinear. Add instruments with "
                "exogenous=[...] or drop endogenous regressors.")


def fit_reg3(spec: ModelSpec, data: Any, *, _frame=None, _system_factory=None) -> ResultBundle:
    """Entry point of ``reg3``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data) if _frame is None else _frame
    system_factory = _system_factory or build_system
    equations = parse_equations(frame.option("equations"), constant=spec.intercept)
    method, ireg3 = frame.option("method"), bool(frame.option("ireg3"))
    if ireg3 and method not in {"3sls", "sure"}:
        raise AnalysisError("invalid_spec", "ireg3 applies to method='3sls' or 'sure' only.")
    outcomes = list(dict.fromkeys(equation.outcome for equation in equations))
    rhs = list(dict.fromkeys(name for equation in equations for name in equation.regressors))
    constant = any(equation.constant for equation in equations)
    allexog = method in {"sure", "ols"}
    if allexog:
        if any(frame.role(name) for name in ("endogenous", "exogenous", "instruments")):
            frame.warn(f"method='{method}' treats every right-hand-side variable as exogenous; "
                       "the endogenous/exogenous/instruments lists are ignored.")
        exogenous, endogenous = [], []
        check_system_role(frame, [*outcomes, *rhs])
        system = system_factory(frame, first=[], rest=[*rhs, *outcomes], constant=constant,
                              outcomes=outcomes)
        rows = None
    else:
        exogenous, endogenous = _classify(frame, outcomes, rhs)
        check_system_role(frame, [*outcomes, *rhs, *exogenous])
        system = system_factory(frame, first=exogenous,
                              rest=[*[name for name in rhs if name not in set(exogenous)],
                                    *outcomes], constant=constant, outcomes=outcomes)
        system = _screen_instruments(frame, system)
        rows = system.instruments
    prepared = prepare(frame, system, equations)
    if rows is not None:
        _check_identified(prepared, rows, set(endogenous) | set(outcomes), system)
    implied = method in {"2sls", "ols"}
    small = implied or bool(frame.option("small"))
    dfk = implied or bool(frame.option("dfk"))
    nobs = system.nobs
    divisor = dfk_divisors(prepared, nobs, dfk)
    data_blocks = [item.data for item in prepared]
    first = kernel_call(kernels.equation_fits, data_blocks, rows if method != "ols" else None)
    start = torch.cat([fit.beta for fit in first])
    restricted, record = restriction(frame, prepared, shift(prepared))
    if implied:
        resid = kernels.residuals(data_blocks, start)
        sigma = torch.diag(resid.square().sum(dim=0) / divisor.diagonal())
        if restricted is None:
            beta = start
            covariance = kernels.block_diagonal(first, sigma.diagonal().tolist())
        else:
            beta, covariance = kernel_call(kernels.stacked_gls, data_blocks, sigma,
                                           None if method == "ols" else rows, restricted)
            resid = kernels.residuals(data_blocks, beta)
        fit = kernels.SystemFit(beta, covariance, sigma, resid, 1, True)
    else:
        fit = kernel_call(kernels.iterate_gls, data_blocks, start,
                          lambda resid: resid.T @ resid / divisor, rows, restricted,
                          iterate=ireg3, tolerance=float(frame.option("tolerance")),
                          max_iterations=int(frame.option("max_iterations")))
        if not fit.converged:
            raise AnalysisError("nonconvergence", f"Iterated {method} did not converge in "
                                f"{fit.iterations} iterations; raise max_iterations or set "
                                "ireg3=False.")
    covariance_text = {
        "3sls": "[X'(S^-1 kron P_Z)X]^-1", "sure": "[X'(S^-1 kron I)X]^-1",
        "2sls": "block diagonal s_ii (X_i'P_Z X_i)^-1", "ols": "block diagonal s_ii (X_i'X_i)^-1",
    }[method]
    divisor_text = "sqrt((N-k_i)(N-k_j))" if dfk else "N"
    extra: dict[str, Any] = {"method": method, "ireg3": ireg3, "dfk": dfk,
                             "endogenous": endogenous if not allexog else [],
                             "exogenous": exogenous if not allexog else rhs}
    if rows is not None:
        extra["instruments"] = system.labels[:rows]
    return assemble(
        frame, system, prepared, fit,
        title=TITLES[method] + (" (iterated)" if ireg3 else ""), small=small,
        divisors=divisor.diagonal().tolist(), restricted=restricted, constraint_record=record,
        solver="compressed_qr_three_stage" if rows is not None else "compressed_qr_feasible_gls",
        inference={"correction": f"conventional covariance {covariance_text} with S = E'E/"
                                 f"{divisor_text}", "sigma_divisor": divisor_text,
                   "method": method, "implied_options": "dfk, small, independent equations"
                   if implied else None},
        extra=extra)


def reg3(*, data: Any, equations: Sequence[dict[str, Any]],
         endogenous: Sequence[str] | None = None, exogenous: Sequence[str] | None = None,
         instruments: Sequence[str] | None = None, method: str = "3sls", ireg3: bool = False,
         small: bool = False, dfk: bool = False,
         constraints: Sequence[dict[str, Any]] | None = None, covariance: str | None = None,
         categorical: Sequence[str] | None = None, weights: str | None = None,
         weight_type: str | None = None, tolerance: float = 1e-9, max_iterations: int = 1000,
         missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Three-stage least squares for systems of simultaneous equations (Stata ``reg3``;
    EViews system 3SLS).

    Model: structural equations ``y_i = X_i b_i + e_i`` whose regressors may be
    endogenous (other equations' outcomes or declared variables), errors
    correlated across equations (``Var(e) = Sigma kron I_N``) and exogenous
    instruments ``Z`` (constant included) uncorrelated with every error.

    Endogenous variables: every outcome, plus ``endogenous``; all other
    right-hand-side variables are exogenous and ``exogenous`` adds
    instruments outside the equations. ``instruments`` gives the complete
    exogenous list instead (Stata's ``inst()``).

    Estimators (``method``):

    - ``'3sls'`` (default): 2SLS of each equation, ``S`` from the structural
      2SLS residuals (``s_ij = e_i'e_j / N``, ``dfk``: ``/ sqrt((N-k_i)(N-k_j))``),
      then ``b = [X'(S^-1 kron P_Z)X]^-1 X'(S^-1 kron P_Z)y`` with
      ``V = [X'(S^-1 kron P_Z)X]^-1``; ``ireg3=True`` iterates ``S``.
    - ``'2sls'``: equation-by-equation 2SLS; implies ``dfk``, ``small`` and
      independent equations (block-diagonal ``V``), as in Stata.
    - ``'ols'``: equation-by-equation OLS; implies ``dfk``, ``small`` and
      independent equations. ``'sure'``: SUR with every variable exogenous.

    Parameters
    ----------
    data, equations : as for ``oe.sureg`` (``{"y", "x", "name"?, "constant"?}``).
    endogenous, exogenous, instruments : see above.
    method, ireg3 : see above. small : t statistics with the first equation's
        ``N - k_1`` degrees of freedom and F tests. dfk : small-sample divisor.
    constraints : ``[{"terms": {"label:term": coef}, "value": v}]``; the first
        step is unconstrained, the system step constrained.
    covariance : ``'nonrobust'`` only (Stata: conventional, bootstrap,
        jackknife). categorical, weights (``'aweight'``, ``'fweight'``),
        tolerance, max_iterations, missing, alpha : as usual.

    Returns
    -------
    ``metrics``: ``<label>:rmse`` (``sqrt(e_i'e_i/d_ii)``), ``<label>:r_squared``
    (``1 - RSS/TSS`` with the structural residuals; may be negative);
    ``tests``: ``model`` and ``<label>:model`` Wald tests of the slopes (chi2,
    or F with ``small``); ``extra``: the residual covariance ``sigma`` used by
    the final step, ``correlation``, the endogenous and exogenous lists, the
    instrument columns and per-equation records. Errors: ``underidentified``
    (rank condition fails), ``singular_sigma``, ``nonconvergence``.

    Stata: ``reg3 (consump wagepriv wagegovt) (wagepriv consump govt capital1),
    endog(...) exog(...) ireg3``.

    Example::

        import openecon as oe
        fit = oe.reg3(data=df, equations=[
            {"y": "consump", "x": ["wagepriv", "wagegovt"]},
            {"y": "wagepriv", "x": ["consump", "govt", "capital1"]}])
        print(fit.summary())
    """
    from openecon.analysis import fit

    roles = {"endogenous": column_list(endogenous, "endogenous") or None,
             "exogenous": column_list(exogenous, "exogenous") or None,
             "instruments": column_list(instruments, "instruments") or None}
    spec = system_spec(
        "reg3", equations, roles={name: value for name, value in roles.items() if value},
        covariance=covariance, categorical=column_list(categorical, "categorical"),
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        options={"equations": equations_option(equations), "method": method,
                 "ireg3": ireg3 or None, "small": small or None, "dfk": dfk or None,
                 "constraints": constraints_option(constraints),
                 "tolerance": None if tolerance == 1e-9 else tolerance,
                 "max_iterations": None if max_iterations == 1000 else max_iterations})
    return fit(spec, data=data)
