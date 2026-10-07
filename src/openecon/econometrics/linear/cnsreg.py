"""Least squares under linear equality constraints: Stata's ``cnsreg``.

Model and estimator
-------------------
``y = x'b + u`` subject to ``R b = r`` with ``R`` a ``q x K`` matrix of full
row rank over the design terms (constant included). The constraints are
removed by an exact reparameterization instead of a Lagrangian: with the
complete QR decomposition of ``R'``,

    ``R' = [Q1 Q2] [T; 0]``,   ``b = b_p + Q2 g``,   ``b_p = Q1 T'^-1 r``,

every ``b`` of this form satisfies ``R b = r`` and ``g`` (``K - q`` free
parameters) is unrestricted. ``g`` solves the ordinary (weighted) least-squares
problem ``(y - X b_p) on X Q2`` by Householder QR, and

    ``V_b = Q2 V_g Q2'``   (rank ``K - q``).

The solution equals the constrained minimiser of ``sum w (y - x'b)^2`` (the
Lagrangian solution ``b_ols - (X'WX)^-1 R' [R (X'WX)^-1 R']^-1 (R b_ols - r)``)
without ever forming ``(X'WX)^-1``.

Constraints are given as a list of ``{"terms": {term: coefficient}, "value":
number}`` over the design term names (``Intercept``, ``x``, ``sector[b]``);
every number must be finite. Linearly dependent constraints are screened left
to right: a dependent constraint consistent with the earlier ones is dropped
with a warning (Stata drops redundant constraints the same way); an
inconsistent one is an error.

Conventions (Stata's cnsreg)
----------------------------
``df_resid = N - K + q`` (``K - q`` free parameters); ``sigma^2 = RSS/df_resid``;
HC1 (``vce(robust)``) uses ``N/(N - K + q)`` and cluster CR1
``G/(G-1) (N-1)/(N-K+q)``, i.e. the regress formulas with ``K - q`` parameters.
Root MSE is reported; cnsreg reports no R-squared. ``tests['model']`` is the
Wald F test that the free, reported non-constant coefficients are zero; its
degrees of freedom are the rank of their covariance block, which equals the
rank of the corresponding rows of ``Q2`` unless a cluster covariance has
fewer clusters than that (then the test is marked ``rank_deficient``).

Coefficients fixed by the constraints
-------------------------------------
Coefficient ``j`` is determined completely by ``R b = r`` exactly when ``e_j``
lies in the row space of ``R``, i.e. when row ``j`` of the orthonormal ``Q2``
is zero. That criterion is structural (independent of the data and of the
units of the regressors), so a genuinely estimated coefficient with a tiny
variance is never mistaken for a fixed one. Fixed coefficients have zero
variance; ``build_result`` only accepts estimated quantities, so they are
reported in ``extra['constrained_terms']`` (term -> value), omitted from the
coefficient table and announced in ``warnings``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec,
)
from openecon.econometrics.linear.common import (
    check_fit, check_observations, check_pweights, covariance_kind, estimation_weights,
    resolve_covariance, robust_wald_test, solve_least_squares, solver_diagnostics,
)
from openecon.engines.linalg import collinear_columns, least_squares
from openecon.models import ModelSpec, ResultBundle

_CONSISTENT = 1e-8        # relative tolerance for a redundant constraint's value
# A row of the orthonormal Q2 whose norm is at most this many eps (times K) is zero up to
# the rounding of the QR factorization: the coefficient is fixed by the constraints.
_FIXED_ROW = 64 * torch.finfo(torch.float64).eps


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def constraint_matrix(constraints: Any, terms: list[str],
                      omitted: Sequence[str]) -> tuple[Tensor, Tensor]:
    """Validate the ``constraints`` option and build ``R`` [q, K] and ``r`` [q]."""
    if not isinstance(constraints, list) or not constraints:
        raise AnalysisError("invalid_constraint", "constraints must be a non-empty list of "
                            "{'terms': {term: coefficient, ...}, 'value': number} entries.")
    rows, values = [], []
    for position, entry in enumerate(constraints, start=1):
        if not isinstance(entry, dict) or set(entry) != {"terms", "value"} \
                or not isinstance(entry["terms"], dict):
            raise AnalysisError("invalid_constraint", f"Constraint {position} must be of the form "
                                "{'terms': {term: coefficient, ...}, 'value': number}.")
        if not _number(entry["value"]):
            raise AnalysisError("invalid_constraint", f"Constraint {position}: the value must be "
                                "a finite number.")
        row = torch.zeros(len(terms), dtype=torch.float64)
        for term, coefficient in entry["terms"].items():
            if not _number(coefficient):
                raise AnalysisError("invalid_constraint", f"Constraint {position}: the coefficient "
                                    f"of '{term}' must be a finite number.")
            if term in omitted:
                raise AnalysisError("invalid_constraint", f"Constraint {position} refers to "
                                    f"'{term}', which was omitted because of collinearity.")
            if term not in terms:
                raise AnalysisError("invalid_constraint", f"Constraint {position} refers to "
                                    f"unknown term '{term}'. Design terms: {', '.join(terms)}.")
            row[terms.index(term)] += float(coefficient)
        rows.append(row)
        values.append(float(entry["value"]))
    return torch.stack(rows), torch.tensor(values, dtype=torch.float64)


def reduce_constraints(frame: ModelFrame, r_matrix: Tensor,
                       r_value: Tensor) -> tuple[Tensor, Tensor]:
    """Drop redundant (dependent, consistent) constraints; reject inconsistent ones."""
    kept, dependent = kernel_call(collinear_columns, r_matrix.T)
    if not dependent:
        return r_matrix, r_value
    if not kept:
        # Every constraint row is zero: 0 = value.
        if bool((r_value.abs() > _CONSISTENT).any()):
            raise AnalysisError("inconsistent_constraints", "A constraint with no coefficients "
                                "requires a nonzero value; it cannot hold.")
        raise AnalysisError("invalid_constraint", "Every constraint has zero coefficients; give at "
                            "least one term per constraint.")
    # Express each dependent row as a combination of the kept rows; its value must follow.
    solved = kernel_call(least_squares, r_matrix[kept].T, r_matrix[dependent].T,
                         drop_collinear=False)
    implied = solved.beta.T @ r_value[kept]
    gap = (implied - r_value[dependent]).abs()
    if bool((gap > _CONSISTENT * (1 + r_value[dependent].abs())).any()):
        bad = [str(i + 1) for i, off in zip(dependent, gap.tolist()) if off > _CONSISTENT]
        raise AnalysisError("inconsistent_constraints", f"Constraint(s) {', '.join(bad)} "
                            "contradict earlier constraints (linearly dependent with a different "
                            "value).")
    frame.warn(f"Dropped redundant constraint(s) {', '.join(str(i + 1) for i in dependent)}: "
               "implied by the earlier constraints.")
    return r_matrix[kept], r_value[kept]


def _particular_solution(q1: Tensor, triangle: Tensor, r_value: Tensor, x: Tensor,
                         y: Tensor) -> tuple[Tensor, Tensor]:
    """``b_p = Q1 T'^-1 r`` and the adjusted outcome ``y - X b_p``, both finite."""
    particular = q1 @ torch.linalg.solve_triangular(triangle.T, r_value[:, None], upper=False)[:, 0]
    adjusted = y - x @ particular
    if not (bool(torch.isfinite(particular).all()) and bool(torch.isfinite(adjusted).all())
            and math.isfinite(float(adjusted.square().sum()))):
        raise AnalysisError("invalid_constraint", "The constraint values are too large for "
                            "float64 arithmetic once applied to the design; rescale the "
                            "constrained regressors or the constraint values.")
    return particular, adjusted


def fit_cnsreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``cnsreg``: ``(spec, data) -> ResultBundle``."""
    frame = ModelFrame(spec, data)
    check_pweights(frame)
    weights, nobs = estimation_weights(frame)
    kind = covariance_kind(frame)
    design = frame.design()
    check_observations(frame, design.x.shape[1])
    design = frame.drop_collinear(design, weights)
    terms = design.terms
    k = len(terms)
    if k == 0:
        raise AnalysisError("empty_design", "Every design column was omitted; nothing to estimate.")
    r_matrix, r_value = constraint_matrix(frame.option("constraints"), terms,
                                         frame.notes.get("omitted_terms", []))
    r_matrix, r_value = reduce_constraints(frame, r_matrix, r_value)
    q = r_matrix.shape[0]
    if q >= k:
        raise AnalysisError("invalid_constraint", f"{q} independent constraint(s) on {k} "
                            "coefficients determine every coefficient; nothing is left to "
                            "estimate.")
    y = frame.numeric(spec.outcome)
    # Exact reparameterization b = b_p + Q2 g through the complete QR of R'.
    basis, factor = torch.linalg.qr(r_matrix.T, mode="complete")
    q1, q2, triangle = basis[:, :q], basis[:, q:], factor[:q]
    particular, adjusted = _particular_solution(q1, triangle, r_value, design.x, y)
    x_free = design.x @ q2
    fitted = solve_least_squares(x_free, adjusted, weights)
    beta = particular + q2 @ fitted.beta
    df_resid = nobs - k + q
    rss = float(fitted.ssr)
    check_fit(rss, None, df_resid, float((y.square() * weights).sum() if weights is not None
                                         else y.square().sum()))
    covariance_free, info = linear_covariance(
        frame, x=x_free, resid=fitted.resid, bread=fitted.xtx_inv, n=nobs, k=k - q,
        df_resid=df_resid, weights=weights, ssr=rss, kind=kind)
    covariance = q2 @ covariance_free @ q2.T
    # Structural criterion: coefficient j is fixed by R b = r iff row j of Q2 vanishes.
    fixed = (torch.linalg.vector_norm(q2, dim=1) <= _FIXED_ROW * k).nonzero().flatten().tolist()
    free = [i for i in range(k) if i not in set(fixed)]
    if not free:
        raise AnalysisError("invalid_constraint", "The constraints fix every coefficient; nothing "
                            "is left to estimate.")
    if fixed:
        beta[fixed] = particular[fixed]
        covariance[fixed, :] = 0.0
        covariance[:, fixed] = 0.0
        listed = ", ".join(f"{terms[i]} = {float(beta[i]):.6g}" for i in fixed)
        frame.warn(f"Fixed by the constraints and not estimated: {listed}. These terms are "
                   "reported in extra['constrained_terms'] instead of the coefficient table.")
    reported = [terms[i] for i in free]
    params, reported_covariance = beta[free], covariance[free][:, free]
    slopes = [i for i, term in enumerate(reported) if term != "Intercept"]
    structural_rank = int(torch.linalg.matrix_rank(q2[[free[i] for i in slopes]])) if slopes else 0
    test = robust_wald_test(frame, params, reported_covariance, slopes,
                            df_inference=info["df_inference"],
                            label="Model F test (free parameters)", expected_rank=structural_rank)
    metrics = {"rmse": (rss / df_resid) ** 0.5, "df_model": test["df"], "df_resid": df_resid,
               "df_constraints": q}
    info.update({"nobs": nobs, "constraints": q, "free_parameters": k - q,
                 "degrees_of_freedom_convention": "cnsreg: N - K + q residual degrees of freedom"})
    extra = {"constrained_terms": {terms[i]: float(beta[i]) for i in fixed},
             "constraint_matrix": {"terms": terms, "R": r_matrix.tolist(), "r": r_value.tolist()}}
    return build_result(
        frame, terms=reported, params=params, covariance=reported_covariance, nobs=nobs,
        df_inference=info["df_inference"], df_resid=df_resid, metrics=metrics,
        fitted=y - fitted.resid, inference=info, tests={"model": test},
        categories=design.categories, solver="constraint_reparameterization_householder_qr",
        solver_diagnostics=solver_diagnostics(fitted), extra=extra)


def cnsreg(*, data: Any, y: str, x: Sequence[str], constraints: Sequence[dict[str, Any]],
           covariance: str | None = None, cluster: str | Sequence[str] | None = None,
           weights: str | None = None, weight_type: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Constrained linear regression, Stata's ``cnsreg``.

    Model
        ``y = x'b + u`` subject to linear equality constraints ``R b = r``. The
        constraints are eliminated exactly: with ``R' = [Q1 Q2][T; 0]`` every
        ``b = Q1 T'^-1 r + Q2 g`` satisfies them, ``g`` is estimated by ordinary
        (weighted) least squares of ``y - X b_p`` on ``X Q2`` and ``V_b = Q2 V_g Q2'``.
        This is the constrained least-squares (Lagrangian) solution.

    Parameters
        data: DataFrame, mapping of columns or list of row records.
        y: outcome column. x: list of regressor columns (a bare string is an error).
        constraints: list of ``{"terms": {term: coefficient, ...}, "value": number}``
            using the design term names (``'Intercept'``, a regressor name, or
            ``'sector[b]'`` for a categorical level), all numbers finite.
            ``{"terms": {"x1": 1, "x2": 1}, "value": 1}`` is Stata's
            ``constraint 1 x1 + x2 = 1``. Unknown terms, non-finite numbers or values too
            large for float64 raise ``invalid_constraint``; a dependent constraint
            consistent with earlier ones is dropped with a warning, an inconsistent one
            raises ``inconsistent_constraints``.
        covariance: ``'nonrobust'`` (default; ``RSS/(N-K+q)``), ``'HC1'``/``'robust'``
            (``N/(N-K+q)``), ``'cluster'`` (CR1 with ``K - q`` parameters; two cluster
            columns use the Cameron-Gelbach-Miller meat with ``G_min``).
        cluster: one cluster column, or two for two-way clustering.
        weights, weight_type: ``'aweight'``, ``'fweight'`` or ``'pweight'`` (pweights
            need HC1 or cluster; HC1 is their default).
        categorical: regressors expanded as ``name[level]`` dummies (first level omitted).
        intercept: include the constant ``Intercept`` (``False`` = Stata's noconstant).
        missing: ``'raise'`` (default) or ``'drop'`` rows with missing model inputs.
        alpha: significance level of the confidence intervals.

    Result
        ``coefficients``: the estimated coefficients with t tests on ``N - K + q``
        degrees of freedom (``G - 1`` with clusters). A coefficient fixed completely by
        the constraints (its row of ``Q2`` is zero: the term lies in the row space of
        ``R``) is not in the table but in ``extra['constrained_terms']`` as
        term -> value, with a warning naming it. ``metrics``: rmse, df_model (rank of
        the model test), df_resid, df_constraints; no R-squared (cnsreg reports none).
        ``tests['model']``: Wald F test that the free non-constant coefficients are
        zero (``rank_deficient`` when a cluster covariance has fewer clusters than
        restrictions). ``extra['constraint_matrix']`` records ``R`` and ``r`` after
        reduction.

    Errors
        ``invalid_constraint``, ``inconsistent_constraints``, ``insufficient_observations``
        (no more distinct observations than design columns, or no residual degrees of
        freedom), ``perfect_fit``, ``empty_design``, ``unsupported_covariance`` (pweights
        with the conventional covariance), ``missing_values``, ``invalid_spec``.

    Stata
        ``constraint 1 x1 = x2`` then ``cnsreg y x1 x2, constraints(1)`` is
        ``oe.cnsreg(data=df, y='y', x=['x1', 'x2'],
        constraints=[{"terms": {"x1": 1, "x2": -1}, "value": 0}])``.

    Example
        >>> import pandas as pd, numpy as np, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
        >>> df["y"] = 0.5 * df.x1 + 0.5 * df.x2 + rng.normal(size=100)
        >>> result = oe.cnsreg(data=df, y="y", x=["x1", "x2"],
        ...                    constraints=[{"terms": {"x1": 1, "x2": 1}, "value": 1}])
        >>> print(result.summary())
    """
    spec = make_spec(
        "cnsreg", outcome=y, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"), intercept=intercept,
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        options={"constraints": list(constraints) if constraints is not None else None})
    return fit(spec, data=data)
