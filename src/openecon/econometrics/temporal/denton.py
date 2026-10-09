"""Equality-constrained original Denton and first-difference Denton-Cholette.

The original methods include the initial adjustment in the difference penalty.
Cholette leaves the first adjustment unanchored; its singular penalty is solved
with the exact benchmark constraints, without a ridge or an invented prior.
The implementation follows the quadratic definitions, not reference code.
"""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.nonparametric.common import procedure
from .common import check_constraints, checked_solve, output, prepare


def _denton(method, low, indicator, *, order, criterion, cholette=False, **options):
    p = prepare(low, indicator, **options)
    if p.x.shape[1] != 1:
        raise AnalysisError("invalid_spec", "Denton requires exactly one high-frequency indicator.")
    if criterion not in ("additive", "proportional"):
        raise AnalysisError("invalid_option", "criterion must be 'additive' or 'proportional'.")
    x = p.x[:, 0]
    if criterion == "proportional" and not bool((x > 0).all()):
        raise AnalysisError("invalid_data", "Proportional Denton requires strictly positive indicators.")
    n, m = len(x), len(p.y)
    # e is the additive discrepancy in common units, or the relative discrepancy.
    # Neither scaling changes the minimizer. Scaling the benchmark rows also
    # leaves the equality constraints unchanged and avoids arbitrary data units.
    if criterion == "proportional":
        units = x
    else:
        scale = max(float(x.abs().max()), float(p.y.abs().max())) or 1.0
        units = torch.full_like(x, scale)
    b = p.y - p.C @ x
    B = p.C * units[None, :]
    row_scale = torch.linalg.vector_norm(B, dim=1)
    if not bool(torch.isfinite(row_scale).all()) or bool((row_scale <= 0).any()):
        raise AnalysisError("numerical_failure", "Denton benchmark scaling is non-finite or zero.")
    B, b = B / row_scale[:, None], b / row_scale
    difference = torch.eye(n, dtype=torch.float64, device="cpu")
    difference[torch.arange(1, n, device="cpu"), torch.arange(n - 1, device="cpu")] = -1.0
    if cholette:
        operator = difference[1:]
        penalty = operator.T @ operator
        # The constant adjustment is a genuine null direction of the Cholette
        # penalty. The benchmark rows identify it; checked_solve refuses a
        # singular or numerically unidentified KKT system rather than adding a
        # ridge. Positive x guarantees that proportional B does not lose it.
        kkt = torch.cat((torch.cat((penalty, B.T), dim=1),
                         torch.cat((B, torch.zeros((m, m), dtype=torch.float64, device="cpu")), dim=1)), dim=0)
        zero_target = bool((p.y == 0).all())
        analytic_zero = zero_target and (criterion == "proportional" or bool((x == x[0]).all()))
        if analytic_zero:
            # Ratio zero has constant proportional adjustment -1. With a
            # constant additive indicator, zero values likewise have constant
            # discrepancy. Both attain the global minimum zero penalty; the
            # benchmark rows identify the penalty's one constant null direction.
            e, multiplier = -x / units, torch.zeros_like(p.y)
        else:
            solution = checked_solve(kkt, torch.cat((torch.zeros(n, dtype=torch.float64, device="cpu"), b)),
                                     "Denton-Cholette constrained quadratic")
            e, multiplier = solution[:n], solution[n:]
        innovations = operator @ e
        gradient = operator.T @ innovations
        force = -(B.T @ multiplier)
        solver = ("analytic zero-series solution of semidefinite equality quadratic" if analytic_zero
                  else "unregularized semidefinite equality KKT")
    else:
        operator = difference if order == 1 else difference @ difference
        # Work in innovations w = Delta^order e. The inverse difference is a
        # cumulative sum, so the minimum-norm benchmark problem only needs an
        # m x m solve and never forms the ill-conditioned fourth-order penalty.
        integration = torch.tril(torch.ones((n, n), dtype=torch.float64, device="cpu"))
        if order == 2:
            integration = integration @ integration
        G = B @ integration
        innovation_scale = torch.linalg.vector_norm(G, dim=1)
        F = G / innovation_scale[:, None]
        multiplier = checked_solve(F @ F.T, b / innovation_scale,
                                   "Original Denton benchmark innovations")
        w = F.T @ multiplier
        e = integration @ w
        innovations = operator @ e
        gradient = operator.T @ innovations
        force = B.T @ (multiplier / innovation_scale)
        solver = "anchored innovation minimum norm with equilibrated benchmark Gram solve"
    if not bool(torch.isfinite(e).all()):
        raise AnalysisError("numerical_failure", "Denton reconstructed adjustments are non-finite.")
    # Check the primal gradient in independently reconstructed difference
    # coordinates, rather than only checking the solver's own linear residual.
    stationarity_error = float((gradient - force).abs().max())
    # Operator backward-error scaling remains defined for the valid Cholette
    # zero-penalty solution, whose true gradient and multiplier are both zero.
    stationarity_scale = max(float(operator.abs().sum(dim=0).max())
                             * float(operator.abs().sum(dim=1).max()) * float(e.abs().max())
                             + float(force.abs().max()), 1e-300)
    relative_stationarity = stationarity_error / stationarity_scale
    if not bool(torch.isfinite(innovations).all()) or relative_stationarity > 2e-10:
        raise AnalysisError("numerical_failure", "Denton quadratic stationarity gate failed.")
    values = x + units * e
    if cholette and analytic_zero:
        values = torch.zeros_like(x)
    if p.settings["aggregation"] in ("first", "last"):
        offset = p.settings["ratio"] - 1 if p.settings["aggregation"] == "last" else 0
        positions = torch.arange(m, device="cpu") * p.settings["ratio"] + offset
        # These cells are supplied observations, rather than estimated cells;
        # preserve their exact input representation (including known zeros).
        values[positions] = p.y
    check_constraints(p, values)
    penalty_value = float(innovations @ innovations)
    return output(method, p, values, settings={
        "criterion": criterion,
        "difference_order": order,
        "initial_condition": "unanchored constant adjustment" if cholette else "zero presample adjustment",
        "solver": solver,
        "stationarity_relative_error": relative_stationarity,
        "stationarity_scale": "||operator.T||inf*||operator||inf*||adjustment||inf+||constraint force||inf",
        "stationarity_tolerance": 2e-10,
        "scaled_penalty": penalty_value,
        "objective": penalty_value if criterion == "proportional" else penalty_value * scale**2,
        "penalty_units": "relative discrepancy" if criterion == "proportional" else "common magnitude scaled discrepancy",
        "additive_magnitude_scale": None if criterion == "proportional" else scale,
        "coefficient_inference": "not applicable; deterministic constrained movement preservation",
        "known_endpoint_cells_preserved": p.settings["aggregation"] in ("first", "last"),
        "references": ["https://journal.r-project.org/articles/RJ-2013-028/"],
    })


@procedure
def denton_additive(low, indicator, *, low_periods, high_periods,
                    low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
                    low_releases=None, high_releases=None, device="cpu", weights=None):
    """Original anchored first-difference additive Denton reconstruction."""
    return _denton("denton_additive", low, indicator, order=1, criterion="additive",
                   low_periods=low_periods, high_periods=high_periods,
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                   high_releases=high_releases, device=device, weights=weights)


@procedure
def denton_additive_second(low, indicator, *, low_periods, high_periods,
                           low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
                           low_releases=None, high_releases=None, device="cpu", weights=None):
    """Original anchored second-difference additive Denton reconstruction."""
    return _denton("denton_additive_second", low, indicator, order=2, criterion="additive",
                   low_periods=low_periods, high_periods=high_periods,
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                   high_releases=high_releases, device=device, weights=weights)


@procedure
def denton_proportional(low, indicator, *, low_periods, high_periods,
                        low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
                        low_releases=None, high_releases=None, device="cpu", weights=None):
    """Original anchored first-difference proportional Denton; positive indicator."""
    return _denton("denton_proportional", low, indicator, order=1, criterion="proportional",
                   low_periods=low_periods, high_periods=high_periods,
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                   high_releases=high_releases, device=device, weights=weights)


@procedure
def denton_proportional_second(low, indicator, *, low_periods, high_periods,
                               low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
                               low_releases=None, high_releases=None, device="cpu", weights=None):
    """Original anchored second-difference proportional Denton; positive indicator."""
    return _denton("denton_proportional_second", low, indicator, order=2, criterion="proportional",
                   low_periods=low_periods, high_periods=high_periods,
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                   high_releases=high_releases, device=device, weights=weights)


@procedure
def denton_cholette(low, indicator, *, low_periods, high_periods, criterion="proportional",
                    low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
                    low_releases=None, high_releases=None, device="cpu", weights=None):
    """Unanchored first-difference Denton-Cholette; additive or proportional penalty."""
    return _denton("denton_cholette", low, indicator, order=1, criterion=criterion, cholette=True,
                   low_periods=low_periods, high_periods=high_periods,
                   low_frequency=low_frequency, high_frequency=high_frequency,
                   aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                   high_releases=high_releases, device=device, weights=weights)
