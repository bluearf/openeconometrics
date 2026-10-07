"""Seemingly unrelated regression (Stata's ``sureg``) and multivariate regression (``mvreg``).

Model
-----
``y_i = X_i b_i + e_i`` for equations ``i = 1..M`` observed on the same ``N``
units, with ``E[e_i e_j'] = s_ij I_N``: the errors of one unit are correlated
across equations, those of different units are not. Stacked,
``y = X b + e`` with ``X = blockdiag(X_1 .. X_M)`` and ``Var(e) = Sigma (x) I_N``.

sureg (Methods and formulas of [R] reg3, of which sureg is the SUR case)
------------------------------------------------------------------------
1. OLS equation by equation; residuals ``e_i``.
2. ``s_ij = e_i'e_j / N`` (``dfk``: ``/ sqrt((N - k_i)(N - k_j))``).
3. Feasible GLS ``b = (X'(S^-1 (x) I)X)^-1 X'(S^-1 (x) I)y`` with the
   conventional covariance ``V = (X'(S^-1 (x) I)X)^-1``.
4. ``iterate=True`` (``isure``) repeats 2-3 with the GLS residuals until the
   coefficients change by less than ``tolerance`` (relative); the limit is the
   Gaussian maximum-likelihood estimator.

The cross products ``X_i'X_j`` and ``X_i'y_j`` are never expanded into the
``NM``-row Kronecker system: one QR of the distinct columns reduces the data
to a small triangular factor on which steps 1-4 run (``common``, ``kernels``).
Inference is z (``small``: t with the first equation's ``N - k_1`` degrees of
freedom and F tests), the per-equation tests are Wald tests of the slopes,
``tests['breusch_pagan']`` is ``N sum_{i>j} r_ij^2 ~ chi2(M(M-1)/2)`` with the
correlations of the residual covariance used in the final GLS step, and
``metrics['log_likelihood']`` is ``-N/2 (M (1 + ln 2 pi) + ln |E'E/N|)`` at the
final residuals.

mvreg (Methods and formulas of [MV] mvreg)
------------------------------------------
All equations share the regressors, so GLS equals OLS equation by equation.
``s_ij = e_i'e_j / (N - k)`` and ``Cov(b_i, b_j) = s_ij (X'X)^-1`` (the general
form ``s_ij (X_i'X_i)^-1 X_i'X_j (X_j'X_j)^-1`` is used); Student t with
``N - k`` degrees of freedom, per-equation F tests, and with ``corr=True`` the
residual correlation matrix and the Breusch-Pagan test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, column_list, kernel_call, make_spec
from openecon.econometrics.systems import kernels
from openecon.econometrics.systems.common import (
    Equation, breusch_pagan, build_system, check_system_role, log_likelihood, parse_equations,
    system_columns,
)
from openecon.econometrics.systems.linear import (
    assemble, dfk_divisors, prepare, restriction, shift,
)
from openecon.models import ModelSpec, ResultBundle


def _regressors(equations: Sequence[Equation]) -> list[str]:
    return list(dict.fromkeys(name for equation in equations for name in equation.regressors))


def fit_sureg(spec: ModelSpec, data: Any, *, _frame=None, _system_factory=None) -> ResultBundle:
    """Entry point of ``sureg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data) if _frame is None else _frame
    equations = parse_equations(frame.option("equations"), constant=spec.intercept)
    regressors = _regressors(equations)
    outcomes = list(dict.fromkeys(equation.outcome for equation in equations))
    check_system_role(frame, [*outcomes, *regressors])
    system = (_system_factory or build_system)(frame, first=[], rest=[*regressors, *outcomes],
                          constant=any(equation.constant for equation in equations),
                          outcomes=outcomes)
    prepared = prepare(frame, system, equations)
    small, dfk, iterate = (bool(frame.option(name)) for name in ("small", "dfk", "iterate"))
    divisor = dfk_divisors(prepared, system.nobs, dfk)
    data_blocks = [item.data for item in prepared]
    first = kernel_call(kernels.equation_fits, data_blocks, None)
    start = torch.cat([fit.beta for fit in first])
    restricted, record = restriction(frame, prepared, shift(prepared))
    fit = kernel_call(kernels.iterate_gls, data_blocks, start, lambda resid: resid.T @ resid
                      / divisor, None, restricted, iterate=iterate,
                      tolerance=float(frame.option("tolerance")),
                      max_iterations=int(frame.option("max_iterations")))
    if not fit.converged:
        raise AnalysisError("nonconvergence", f"Iterated SUR did not converge in "
                            f"{fit.iterations} iterations; raise max_iterations or use the "
                            "two-step estimator (iterate=False).")
    nobs = system.nobs
    ll = log_likelihood(fit.resid.T @ fit.resid / nobs, nobs)
    return assemble(
        frame, system, prepared, fit,
        title="Seemingly unrelated regression" + (" (iterated)" if iterate else ""),
        small=small, divisors=divisor.diagonal().tolist(), restricted=restricted,
        constraint_record=record, solver="compressed_qr_feasible_gls",
        inference={"correction": "conventional FGLS covariance (X'(S^-1 kron I)X)^-1 with "
                                 + ("S = E'E/sqrt((N-k_i)(N-k_j))" if dfk else "S = E'E/N"),
                   "sigma_divisor": "sqrt((N-k_i)(N-k_j))" if dfk else "N",
                   "estimator": "iterated FGLS (ML)" if iterate else "two-step FGLS"},
        metrics={"log_likelihood": ll}, breusch_pagan=breusch_pagan(fit.sigma, nobs),
        extra={"method": "isure" if iterate else "sure", "dfk": dfk})


def fit_mvreg(spec: ModelSpec, data: Any, *, _frame=None, _system_factory=None) -> ResultBundle:
    """Entry point of ``mvreg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data) if _frame is None else _frame
    outcomes = frame.role("outcomes")
    if not outcomes or outcomes[0] != spec.outcome:
        raise AnalysisError("invalid_spec", "The role 'outcomes' must list every dependent "
                            "variable, starting with ModelSpec.outcome.")
    if len(outcomes) < 2:
        raise AnalysisError("invalid_spec", "mvreg needs at least two dependent variables.")
    overlap = [name for name in outcomes if name in spec.predictors]
    if overlap:
        raise AnalysisError("invalid_spec", f"Dependent variables cannot also be regressors: "
                            f"{', '.join(overlap)}.")
    equations = [Equation(name, name, list(spec.predictors), spec.intercept) for name in outcomes]
    system = (_system_factory or build_system)(frame, first=[], rest=[*spec.predictors, *outcomes],
                          constant=spec.intercept, outcomes=outcomes)
    prepared = prepare(frame, system, equations)
    data_blocks = [item.data for item in prepared]
    fits = kernel_call(kernels.equation_fits, data_blocks, None)
    beta = torch.cat([fit.beta for fit in fits])
    resid = kernels.residuals(data_blocks, beta)
    nobs, k = system.nobs, prepared[0].k
    sigma = resid.T @ resid / (nobs - k)
    covariance = kernels.cross_covariance(data_blocks, fits, sigma)
    fit = kernels.SystemFit(beta, covariance, sigma, resid, 1, True)
    corr = bool(frame.option("corr"))
    return assemble(
        frame, system, prepared, fit, title="Multivariate regression", small=True,
        divisors=[nobs - k] * len(prepared), restricted=None, constraint_record=None,
        solver="compressed_qr_ols",
        inference={"correction": "Cov(b_i, b_j) = s_ij (X'X)^-1 with S = E'E/(N-k)",
                   "sigma_divisor": "N-k"},
        breusch_pagan=breusch_pagan(sigma, nobs) if corr else None,
        extra={"method": "mvreg", "corr": corr})


# ---- convenience functions ------------------------------------------------------------------


def equations_option(equations: Any) -> list[dict[str, Any]]:
    """Validated copy of a convenience function's ``equations`` argument."""
    parse_equations(equations)
    return [dict(entry) for entry in equations]


def constraints_option(constraints: Any) -> list[dict[str, Any]] | None:
    """Validated copy of a convenience function's ``constraints`` argument."""
    if constraints is None:
        return None
    if isinstance(constraints, (str, bytes, dict)) or not isinstance(constraints, Sequence) \
            or not all(isinstance(entry, dict) for entry in constraints):
        raise AnalysisError("invalid_constraint", "constraints must be a list of mappings such "
                            "as [{'terms': {'y1:x': 1, 'y2:x': -1}, 'value': 0}].")
    return [dict(entry) for entry in constraints]


def system_spec(estimator: str, equations: Any, *, roles: dict[str, Any] | None = None,
                **fields: Any) -> ModelSpec:
    """ModelSpec of a multi-equation estimator: outcome = first equation's dependent variable."""
    parse_equations(equations)
    roles = roles or {}
    columns = system_columns(equations, *roles.values())
    return make_spec(estimator, outcome=equations[0]["y"], columns={"system": columns, **roles},
                     **fields)


def sureg(*, data: Any, equations: Sequence[dict[str, Any]], iterate: bool = False,
          small: bool = False, dfk: bool = False,
          constraints: Sequence[dict[str, Any]] | None = None, covariance: str | None = None,
          categorical: Sequence[str] | None = None, weights: str | None = None,
          weight_type: str | None = None, tolerance: float = 1e-9, max_iterations: int = 1000,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Zellner's seemingly unrelated regressions (Stata ``sureg``; EViews system SUR).

    Model: ``y_i = X_i b_i + e_i`` for ``M`` equations on the same ``N`` units,
    with errors correlated across equations (``Var(e) = Sigma kron I_N``).

    Estimator (Stata's Methods and formulas of reg3/sureg):

    1. OLS equation by equation, residuals ``e_i``;
    2. ``S`` with ``s_ij = e_i'e_j / N`` (``dfk=True``:
       ``/ sqrt((N - k_i)(N - k_j))``);
    3. FGLS ``b = (X'(S^-1 kron I)X)^-1 X'(S^-1 kron I) y`` with covariance
       ``(X'(S^-1 kron I)X)^-1``;
    4. ``iterate=True`` repeats 2-3 on the GLS residuals until convergence
       (Stata's ``isure``): the Gaussian maximum-likelihood estimator. If an
       outcome is also a regressor of another equation that likelihood is
       unbounded; the iteration then raises ``singular_sigma``.

    The stacked Kronecker system is never formed: one Householder QR of the
    distinct data columns gives a small triangular factor on which every step
    runs, so a system on a million rows costs one pass over the data.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    equations : list of ``{"y": outcome, "x": [regressors]}``; optional
        ``"name"`` (equation label, default the outcome) and ``"constant"``
        (default True). Terms are reported as ``label:term``.
    iterate : iterate FGLS to convergence (ML). small : t (with the first
        equation's ``N - k_1`` degrees of freedom) and F instead of z and chi2.
    dfk : small-sample divisor of the residual covariance.
    constraints : linear constraints, e.g. ``[{"terms": {"y1:x": 1, "y2:x": -1},
        "value": 0}]`` for equal coefficients of ``x`` (Stata ``constraint``).
        The first step is unconstrained OLS; the GLS step is constrained.
    covariance : ``'nonrobust'`` only (Stata's sureg offers conventional,
        bootstrap and jackknife).
    categorical : regressors to expand as treatment-coded dummies.
    weights, weight_type : ``'aweight'`` (rescaled to sum to N) or
        ``'fweight'`` (replicate rows).
    tolerance, max_iterations : convergence of ``iterate``.
    missing : ``'raise'`` (default) or ``'drop'`` (casewise over all
        equations, as Stata does). alpha : test size.

    Returns
    -------
    ResultBundle with ``metrics``: ``n_equations``, ``<label>:rmse``
    (``sqrt(e_i'e_i / d_ii)``, divisor ``N`` or ``N - k_i``),
    ``<label>:r_squared``, ``log_likelihood``, ``df_model``, ``df_resid``;
    ``tests``: ``model`` (all slopes), ``<label>:model`` (each equation's
    slopes, chi2 or F) and ``breusch_pagan`` (independence of the equations,
    ``N sum r_ij^2``); ``extra``: per-equation records, the residual covariance
    ``sigma`` used by the final GLS step and its ``correlation``.

    Stata: ``sureg (y1 x1 x2) (y2 x1 x3), isure dfk small corr``.

    Example::

        import openecon as oe
        fit = oe.sureg(data=df, equations=[{"y": "price", "x": ["foreign", "weight"]},
                                           {"y": "mpg", "x": ["foreign", "weight"]}])
        print(fit.summary())
        print(fit.tests["breusch_pagan"])
    """
    from openecon.analysis import fit

    spec = system_spec(
        "sureg", equations, covariance=covariance,
        categorical=column_list(categorical, "categorical"), weights=weights,
        weight_type=weight_type, missing=missing, alpha=alpha,
        options={"equations": equations_option(equations),
                 "iterate": iterate or None, "small": small or None, "dfk": dfk or None,
                 "constraints": constraints_option(constraints),
                 "tolerance": None if tolerance == 1e-9 else tolerance,
                 "max_iterations": None if max_iterations == 1000 else max_iterations})
    return fit(spec, data=data)


def mvreg(*, data: Any, y: Sequence[str], x: Sequence[str], intercept: bool = True,
          corr: bool = False, categorical: Sequence[str] | None = None,
          weights: str | None = None, weight_type: str | None = None, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Multivariate regression (Stata ``mvreg``; SPSS ``GLM`` with several dependents).

    Model: ``y_j = X b_j + e_j`` for every outcome ``j`` with the SAME
    regressors ``X`` (``k`` columns with the constant); errors correlated
    across equations. GLS equals OLS equation by equation here, so the
    coefficients are the OLS ones; what mvreg adds is their joint covariance

        Cov(b_i, b_j) = s_ij (X'X)^-1,   s_ij = e_i'e_j / (N - k),

    which allows tests across equations.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    y : list of two or more dependent variables. x : regressors.
    intercept : include the constant (Stata's ``noconstant`` is False).
    corr : report the residual correlation matrix (``extra['correlation']``
        always holds it) and the Breusch-Pagan test of independence
        ``N sum_{i>j} r_ij^2 ~ chi2(M(M-1)/2)`` in ``tests['breusch_pagan']``.
    categorical, weights / weight_type (``'aweight'`` or ``'fweight'``),
    missing, alpha : as for the other estimators.

    Returns
    -------
    Student t tests with ``N - k`` degrees of freedom; ``metrics``:
    ``<y>:rmse`` (``sqrt(RSS/(N-k))``), ``<y>:r_squared``; ``tests``:
    ``<y>:model`` (each equation's F test of its slopes, the classical
    regression F), ``model`` (all slopes of all equations); terms are
    ``<y>:<term>``.

    Stata: ``mvreg y1 y2 y3 = x1 x2, corr``.

    Example::

        import openecon as oe
        fit = oe.mvreg(data=df, y=["headroom", "trunk", "turn"],
                       x=["price", "mpg", "displacement"], corr=True)
        print(fit.summary())
    """
    from openecon.analysis import fit

    outcomes = column_list(y, "y")
    if not outcomes:
        raise AnalysisError("invalid_spec", "y must list at least two dependent variables.")
    spec = make_spec(
        "mvreg", outcome=outcomes[0], predictors=column_list(x, "x"), intercept=intercept,
        categorical=column_list(categorical, "categorical"), weights=weights,
        weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"outcomes": outcomes}, options={"corr": corr or None})
    return fit(spec, data=data)
