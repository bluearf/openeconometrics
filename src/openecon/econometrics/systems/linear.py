"""Analysis layer shared by sureg, mvreg and reg3: equations, constraints, results.

* **Collinearity.** Each equation's regressors are screened left to right on
  the compressed (and, with a constant, centred) columns, as Stata's
  ``_rmcoll`` does; omitted terms are recorded as ``label:term``.
* **Constraints.** ``[{"terms": {"label:term": coefficient}, "value": v}]``
  over the reported terms, validated and reduced by ``cnsreg``'s helpers and
  imposed as an exact reparameterization of the stacked GLS problem.
  Coefficients that the constraints fix completely are reported in
  ``extra['constrained_terms']`` instead of the coefficient table.
* **Per-equation statistics** (Stata's sureg/reg3 header): RMSE
  ``sqrt(e_i'e_i / d_ii)`` with the divisor of the residual covariance
  (``N``, or ``N - k_i`` with ``dfk``), ``R^2 = 1 - RSS/TSS`` (centred TSS
  with a constant, uncentred without; it can be negative for 2SLS/3SLS), and
  the Wald test of the equation's slopes (chi2, or F with ``small``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, kernel_call, wald_test
from openecon.econometrics.linear.cnsreg import constraint_matrix, reduce_constraints
from openecon.econometrics.systems.common import (
    Equation, SystemData, check_equation_fit, column_means, compressed_columns, correlation,
    is_flat, record_omitted,
)
from openecon.econometrics.systems.kernels import EquationData, Restriction, SystemFit
from openecon.engines.linalg import collinear_columns
from openecon.models import ResultBundle

_FIXED_ROW = 64 * torch.finfo(torch.float64).eps


@dataclass
class Prepared:
    equation: Equation
    index: list[int]             # columns of W used as regressors (after the screen)
    terms: list[str]             # their term names
    data: EquationData           # compressed columns (centred with a constant)
    means: Tensor                # centring means (0 for the constant)
    outcome: int                 # column of W holding the outcome

    @property
    def k(self) -> int:
        return len(self.terms)

    @property
    def reported(self) -> list[str]:
        return [f"{self.equation.name}:{term}" for term in self.terms]


def prepare(frame: ModelFrame, system: SystemData, equations: Sequence[Equation]) -> list[Prepared]:
    """Screen every equation and build its compressed, centred columns."""
    prepared = []
    for equation in equations:
        outcome = system.blocks[equation.outcome][0]
        if equation.constant and is_flat(system, outcome):
            raise AnalysisError("constant_outcome", f"The outcome of equation '{equation.name}' "
                                "has no variation in the estimation sample.")
        index, terms = system.columns(equation.regressors, equation.constant)
        centred = equation.constant
        design = compressed_columns(system, index, centred)
        kept, omitted = kernel_call(collinear_columns, design) if index else ([], [])
        record_omitted(frame, [f"{equation.name}:{terms[i]}" for i in omitted], "collinearity")
        if not kept:
            raise AnalysisError("empty_equation", f"Every regressor of equation "
                                f"'{equation.name}' was omitted; nothing is left to estimate.")
        index, terms = [index[i] for i in kept], [terms[i] for i in kept]
        if system.nobs <= len(index) or system.rows <= len(index):
            raise AnalysisError("insufficient_observations", f"Equation '{equation.name}' has "
                                f"{len(index)} parameters but only {system.rows} "
                                "observation(s).")
        means = column_means(system, index) if centred else torch.zeros(len(index),
                                                                         dtype=torch.float64)
        data = EquationData(compressed_columns(system, index, centred), system.r[:, outcome])
        prepared.append(Prepared(equation, index, terms, data, means, outcome))
    return prepared


def outcome_scales(system: SystemData, prepared: Sequence[Prepared]) -> list[tuple[float, float]]:
    """``(TSS, sum w y^2)`` of every outcome from the compressed columns."""
    scales = []
    for item in prepared:
        column = system.r[:, item.outcome]
        total = float(column.square().sum())
        if item.equation.constant:
            tss = 0.0 if is_flat(system, item.outcome) else float(column[1:].square().sum())
        else:
            tss = total
        scales.append((tss, total))
    return scales


def shift(prepared: Sequence[Prepared]) -> Tensor:
    """Jacobian of the map from centred to reported parameters (``b_0 = b_0c - m'b``)."""
    blocks = []
    for item in prepared:
        block = torch.eye(item.k, dtype=torch.float64)
        if item.equation.constant and item.k > 1:
            block[0, 1:] = -item.means[1:]
        blocks.append(block)
    return torch.block_diag(*blocks)


def restriction(frame: ModelFrame, prepared: Sequence[Prepared], jacobian: Tensor
                ) -> tuple[Restriction | None, dict[str, Any] | None]:
    """The constraints option as a reparameterization of the centred parameters."""
    constraints = frame.option("constraints")
    if constraints is None:
        return None, None
    terms = [term for item in prepared for term in item.reported]
    matrix, value = constraint_matrix(constraints, terms, frame.notes.get("omitted_terms", []))
    matrix, value = reduce_constraints(frame, matrix, value)
    q, k = matrix.shape
    if q >= k:
        raise AnalysisError("invalid_constraint", f"{q} independent constraint(s) on {k} "
                            "coefficients leave nothing to estimate.")
    centred = matrix @ jacobian
    basis, factor = torch.linalg.qr(centred.T, mode="complete")
    particular = basis[:, :q] @ torch.linalg.solve_triangular(
        factor[:q].T, value[:, None], upper=False)[:, 0]
    if not bool(torch.isfinite(particular).all()):
        raise AnalysisError("invalid_constraint", "The constraint values are too large for "
                            "float64 arithmetic; rescale them.")
    record = {"terms": terms, "R": matrix.tolist(), "r": value.tolist(), "count": q}
    return Restriction(basis[:, q:], particular), record


@dataclass
class EquationStats:
    rss: list[float]
    tss: list[float]
    rmse: list[float]
    r_squared: list[float]


def equation_stats(system: SystemData, prepared: Sequence[Prepared], fit: SystemFit,
                   divisors: Sequence[float]) -> EquationStats:
    scales = outcome_scales(system, prepared)
    rss = [float(value) for value in fit.resid.square().sum(dim=0)]
    for item, value, (tss, total) in zip(prepared, rss, scales, strict=True):
        check_equation_fit(value, total, tss, item.equation.name)
    tss = [tss for tss, _ in scales]
    return EquationStats(
        rss, tss, [math.sqrt(value / d) for value, d in zip(rss, divisors, strict=True)],
        [1 - value / total for value, total in zip(rss, tss, strict=True)])


def _fixed(jacobian_basis: Tensor | None, k: int) -> list[int]:
    if jacobian_basis is None:
        return []
    norms = torch.linalg.vector_norm(jacobian_basis, dim=1)
    scale = float(norms.max()) if norms.numel() else 1.0
    return (norms <= _FIXED_ROW * k * max(scale, 1.0)).nonzero().flatten().tolist()


def assemble(frame: ModelFrame, system: SystemData, prepared: Sequence[Prepared], fit: SystemFit,
             *, title: str, small: bool, divisors: Sequence[float], restricted: Restriction | None,
             constraint_record: dict[str, Any] | None, solver: str, inference: dict[str, Any],
             metrics: dict[str, Any] | None = None, tests: dict[str, Any] | None = None,
             extra: dict[str, Any] | None = None,
             breusch_pagan: dict[str, Any] | None = None) -> ResultBundle:
    """Map the centred estimates back and build the ResultBundle of a linear system."""
    jacobian = shift(prepared)
    beta = jacobian @ fit.beta
    covariance = jacobian @ fit.covariance @ jacobian.T
    terms = [term for item in prepared for term in item.reported]
    labels = [item.equation.name for item in prepared for _ in item.terms]
    k_total = len(terms)
    fixed = _fixed(None if restricted is None else jacobian @ restricted.basis, k_total)
    free = [i for i in range(k_total) if i not in set(fixed)]
    if fixed:
        listed = ", ".join(f"{terms[i]} = {float(beta[i]):.6g}" for i in fixed)
        frame.warn(f"Fixed by the constraints and not estimated: {listed}. These terms are "
                   "reported in extra['constrained_terms'] instead of the coefficient table.")
    stats = equation_stats(system, prepared, fit, divisors)
    nobs = system.nobs
    small_df = [nobs - item.k for item in prepared]
    position = {old: new for new, old in enumerate(free)}
    params, reported_cov = beta[free], covariance[free][:, free]
    equation_tests: dict[str, Any] = {}
    records = []
    start = 0
    for i, item in enumerate(prepared):
        index = [position[j] for j in range(start, start + item.k)
                 if j in position and not terms[j].endswith(":Intercept")]
        start += item.k
        test = None
        if index:
            label = f"{'F' if small else 'Wald chi2'} test of the slopes of equation " \
                    f"{item.equation.name}"
            test = wald_test(params, reported_cov, index, df_resid=small_df[i] if small else None,
                             label=label)
            equation_tests[f"{item.equation.name}:model"] = test
        records.append({
            "name": item.equation.name, "outcome": item.equation.outcome, "terms": item.terms,
            "constant": item.equation.constant, "parameters": item.k,
            "slopes": item.k - int(item.equation.constant), "rmse": stats.rmse[i],
            "r_squared": stats.r_squared[i], "rss": stats.rss[i],
            "statistic": None if test is None else test["statistic"],
            "p_value": None if test is None else test["p_value"],
            "distribution": None if test is None else test["distribution"]})
    all_slopes = [position[j] for j in range(k_total)
                  if j in position and not terms[j].endswith(":Intercept")]
    model = wald_test(params, reported_cov, all_slopes,
                      df_resid=small_df[0] if small else None,
                      label=f"{'F' if small else 'Wald chi2'} test of all slopes of the system")
    sigma = fit.sigma
    result_metrics: dict[str, Any] = {"n_equations": len(prepared)}
    for item, rmse, r2 in zip(prepared, stats.rmse, stats.r_squared, strict=True):
        result_metrics[f"{item.equation.name}:rmse"] = rmse
        result_metrics[f"{item.equation.name}:r_squared"] = r2
    result_metrics.update(metrics or {})
    result_metrics.update({"df_model": len(all_slopes), "df_resid": small_df[0]})
    result_tests = {"model": model, **equation_tests, **(tests or {})}
    if breusch_pagan is not None:
        result_tests["breusch_pagan"] = breusch_pagan
    result_extra: dict[str, Any] = {
        "equations": records, "sigma": sigma.tolist(), "correlation": correlation(sigma).tolist(),
        "equation_names": [item.equation.name for item in prepared],
        "iterations": fit.iterations, "converged": fit.converged,
    }
    if constraint_record is not None:
        result_extra["constraints"] = constraint_record
        result_extra["constrained_terms"] = {terms[i]: float(beta[i]) for i in fixed}
    result_extra.update(extra or {})
    first = prepared[0]
    fitted = system.raw[:, first.index] @ beta[:first.k]
    df_first = small_df[0]
    record = {"nobs": nobs, "small": small, "df_inference": df_first if small else None,
              "residual_definition": "outcome minus X b with the observed regressors"}
    record.update(inference)
    return build_result(
        frame, terms=[terms[i] for i in free], params=params, covariance=reported_cov,
        equations=[labels[i] for i in free], title=title,
        use_t=small, df_inference=df_first if small else None,
        df_resid=small_df[0], metrics=result_metrics, fitted=fitted,
        observed=frame.numeric(first.equation.outcome), nobs=nobs, inference=record,
        tests=result_tests, extra=result_extra, categories=system.categories, solver=solver,
        solver_diagnostics={"iterations": fit.iterations, "converged": fit.converged,
                            "compressed_columns": int(system.r.shape[1]),
                            "method": "one Householder QR of the distinct columns; GLS on "
                                      "the R factor"})


def dfk_divisors(prepared: Sequence[Prepared], nobs: int, dfk: bool) -> Tensor:
    """Residual covariance divisors: N, or sqrt((N-k_i)(N-k_j)) with dfk."""
    if not dfk:
        m = len(prepared)
        return torch.full((m, m), float(nobs), dtype=torch.float64)
    df = torch.tensor([nobs - item.k for item in prepared], dtype=torch.float64)
    return (df[:, None] * df[None, :]).sqrt()
