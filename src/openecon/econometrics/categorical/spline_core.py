"""Declared-knot degree-one spline algebra; resident native CPU float64."""
from __future__ import annotations

import math
from numbers import Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from .frequency import _admit

DT = torch.float64
LIMIT_BYTES = 128 * 1024**2
WORK = 300_000_000


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _close(actual, expected, what, *, atol=2e-8):
    if (actual.shape != expected.shape or not bool(torch.isfinite(actual).all())
            or not torch.allclose(actual, expected, atol=atol, rtol=2e-8)):
        _error(f"Saved spline {what} is inconsistent.", "invalid_state")


def _rank(matrix):
    singular = torch.linalg.svdvals(matrix)
    norms = torch.linalg.vector_norm(matrix, dim=0)
    if (not bool(torch.isfinite(singular).all()) or not bool(torch.isfinite(norms).all())
            or bool((norms <= 0).any()) or not len(singular) or len(singular) < matrix.shape[1]
            or float(singular[-1]) <= 1e-10 * float(singular[0])):
        _error("Declared spline basis is rank deficient or ill conditioned.", "rank_deficient")
    return float(singular[0] / singular[-1])


def _knots(knots, variables):
    if not isinstance(knots, dict) or set(knots) != set(variables):
        _error("knots must give the complete ordered support knots for every variable.")
    result = {}
    for name in variables:
        values = knots[name]
        if not isinstance(values, (list, tuple)) or not 2 <= len(values) <= 8:
            _error("Each variable requires 2–8 knots including both support endpoints.")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               or abs(v) > 1e100 for v in values):
            _error("Knots must be finite real numbers of magnitude at most 1e100.")
        values = [float(v) for v in values]
        span = values[-1] - values[0]
        if span <= 0 or any(b-a <= 1e-10 * span for a, b in zip(values, values[1:])):
            _error("Knots must be strictly increasing and separated relative to their span.")
        result[name] = values
    if sum(len(v)-1 for v in result.values()) > 32:
        _error("The complete spline design admits at most 32 segment columns.", "resource_limit")
    return result


def _basis(values, knots):
    if not bool(torch.isfinite(values).all()):
        _error("Spline values must be finite.", "invalid_data")
    if bool(((values < knots[0]) | (values > knots[-1])).any()):
        _error("Spline values lie outside the caller-declared closed knot span.", "outside_support")
    left = torch.tensor(knots[:-1], dtype=DT)
    widths = torch.tensor([b-a for a, b in zip(knots, knots[1:])], dtype=DT)
    return ((values[:, None]-left)/widths).clamp(0, 1)


def _prepare(data, variables, knots, missing, max_bytes, max_work, operation):
    if not isinstance(data, pd.DataFrame):
        _error("Spline scaling requires a pandas DataFrame.")
    if (not isinstance(variables, list) or not 1 <= len(variables) <= 6
            or any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in variables)
            or len(set(variables)) != len(variables)):
        _error("Supply 1–6 distinct bounded spline variable names.")
    if missing not in ("drop", "raise"):
        _error("missing must be drop or raise.")
    n = len(data)
    if not 4 <= n <= 3000:
        _error("Spline input admits 4–3000 physical rows.", "resource_limit")
    for name in variables:
        if list(data.columns).count(name) != 1:
            _error(f"Column {name!r} must occur exactly once.")
    admitted_knots = _knots(knots, variables)
    total = sum(len(v)-1 for v in admitted_knots.values())
    plan = _admit(operation + " input", 512*n*(len(variables)+total+2),
                  64*n*total*total, max_bytes, max_work)
    selected = data.loc[:, variables]
    absent = selected.isna().any(axis=1).tolist()
    if missing == "raise" and any(absent):
        _error("Spline variables contain missing observations.", "missing_data")
    positions = [i for i, missing_row in enumerate(absent) if not missing_row]
    if len(positions) <= total:
        _error("Complete rows must exceed the declared spline design dimension.", "insufficient_sample")
    frame = selected.iloc[positions].reset_index(drop=True)
    for name in variables:
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            _error("Spline variables must be real numeric columns.", "invalid_data")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               or abs(v) > 1e100 for v in frame[name].tolist()):
            _error("Spline variables must contain bounded finite real values.", "invalid_data")
    raw = torch.tensor(frame.to_numpy(dtype=float).tolist(), dtype=DT)
    raw_bases = [_basis(raw[:, j], admitted_knots[name]) for j, name in enumerate(variables)]
    means = [b.mean(0) for b in raw_bases]
    bases = [b-m for b, m in zip(raw_bases, means)]
    for b in bases:
        _rank(b)
    return dict(frame=frame, raw=raw, positions=positions, input_nobs=n,
                knots=admitted_knots, bases=bases, basis_means=means,
                raw_bases=raw_bases, plan=plan)


def _stationary(matrix, target, coefficient, *, nonnegative=False):
    """Check residuals in normalized feasible directions, including near ties.

    Raw normal-equation gradients can hide large residual errors in nearly
    collinear directions. QR coordinates test active-subspace stationarity;
    inactive cone dual violations are scaled by the smallest singular value,
    so positive combinations of nearly cancelling columns cannot hide them.
    This validates saved coefficients without solving for replacements.
    """
    if not bool(torch.isfinite(coefficient).all()):
        return False
    norms = torch.linalg.vector_norm(matrix, dim=0)
    target_norm = target.norm()
    if (not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any())
            or not bool(torch.isfinite(target_norm)) or float(target_norm) <= 0):
        return False
    if nonnegative and bool((coefficient < 0).any()):
        return False
    normalized = matrix/norms
    residual = (matrix@coefficient-target)/target_norm
    if not bool(torch.isfinite(residual).all()):
        return False
    tolerance = 2e-7
    if nonnegative:
        singular = torch.linalg.svdvals(normalized)
        if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0:
            return False
        # Bound the complete projected KKT error relative to curvature. Active
        # residual tolerance must also shrink: otherwise its allowed error can
        # mask an inactive derivative along a nearly cancelling direction.
        tolerance = 2e-9*float(singular[-1])/max(1., float(normalized.norm()))
    active = coefficient > 0 if nonnegative else torch.ones(len(coefficient), dtype=torch.bool)
    if bool(active.any()):
        directions = torch.linalg.qr(normalized[:, active], mode="reduced").Q
        if float((directions.T@residual).norm()) > tolerance:
            return False
    if nonnegative and bool((~active).any()):
        gradient = normalized[:, ~active].T@residual
        negative = torch.minimum(gradient, torch.zeros_like(gradient))
        if not bool(torch.isfinite(gradient).all()) or float(negative.norm()) > tolerance:
            return False
    return True


def _nnls(matrix, target):
    """Bounded Lawson–Hanson active set on a full-rank small design.

    Centered ramp coefficients are nonnegative increments. No SciPy/runtime
    delegation, clipping of an unconstrained fit, or unverified convergence.
    """
    _rank(matrix)
    original_matrix, original_target = matrix, target
    width = matrix.shape[1]
    if target.ndim != 1 or len(target) != len(matrix) or not bool(torch.isfinite(target).all()):
        _error("Constrained spline target is invalid.", "invalid_data")
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    target_norm = float(target.norm())
    if not math.isfinite(target_norm):
        _error("Constrained spline target norm is nonfinite.", "degenerate_transform")
    if target_norm == 0:
        return torch.zeros(width, dtype=DT)
    # Normalize each column and the target so active-set/KKT tolerances do not
    # depend on the declared knot span or response units.
    matrix = matrix / column_norms
    target = target / target_norm
    scale = float(matrix.norm()*target.norm())
    # Pay for full-row QR once. Active-set least squares then uses only the
    # D-by-D triangular design, within the declared n*D**2 + 40*D**4
    # bound for at most 40*D active-set solves of at most D**3 work.
    orthogonal, matrix = torch.linalg.qr(matrix, mode="reduced")
    target = orthogonal.T @ target
    coefficient = torch.zeros(width, dtype=DT)
    passive = torch.zeros(width, dtype=torch.bool)
    tolerance = 2e-12 * scale
    limit = 40 * width + 1
    iterations = 0
    while iterations < limit:
        gradient = matrix.T @ (target-matrix@coefficient)
        available = gradient.clone()
        available[passive] = -torch.inf
        if not bool((available > tolerance).any()):
            break
        passive[int(torch.argmax(available))] = True
        while True:
            iterations += 1
            if iterations >= limit:
                _error("Constrained spline active set exhausted its finite work bound.", "nonconvergence")
            candidate = torch.zeros_like(coefficient)
            candidate[passive] = torch.linalg.lstsq(matrix[:, passive], target).solution
            if bool((candidate[passive] > 0).all()):
                coefficient = candidate
                break
            leaving = passive & (candidate <= 0)
            denominator = coefficient[leaving]-candidate[leaving]
            valid = denominator > 0
            alpha = (float((coefficient[leaving][valid]/denominator[valid]).min())
                     if bool(valid.any()) else 0.)
            coefficient += alpha*(candidate-coefficient)
            near_zero = passive & (coefficient <= 1e-14 * max(1., float(coefficient.abs().max())))
            coefficient[near_zero] = 0
            passive[near_zero] = False
    else:
        _error("Constrained spline active set did not converge.", "nonconvergence")
    residual_gradient = matrix.T@(matrix@coefficient-target)
    active = coefficient > 0
    if (not bool(torch.isfinite(coefficient).all()) or not bool(torch.isfinite(residual_gradient).all())
            or bool((coefficient < 0).any()) or bool((residual_gradient[~active] < -2e-9*scale).any())
            or bool((residual_gradient[active].abs() > 2e-9*scale).any())):
        _error("Constrained spline solution failed KKT acceptance.", "nonconvergence")
    scaled = coefficient * target_norm / column_norms
    if not bool(torch.isfinite(scaled).all()):
        _error("Constrained spline coefficient scale is nonfinite.", "degenerate_transform")
    if not _stationary(original_matrix, original_target, scaled, nonnegative=True):
        _error("Constrained spline solution failed normalized feasible-direction acceptance.", "nonconvergence")
    return scaled


def _metadata_bound(value):
    """Bound persisted primitive metadata before serialization or tensors."""
    pending, size, nodes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        size += 64
        if nodes > 120000 or depth > 16 or size > 8*1024**2:
            _error("Saved spline metadata exceeds its bounded domain.", "resource_limit")
        if isinstance(item, dict):
            if len(item) > 12000:
                _error("Saved spline metadata has excessive width.", "resource_limit")
            pending.extend((v, depth+1) for pair in item.items() for v in pair)
        elif isinstance(item, (list, tuple)):
            if len(item) > 12000:
                _error("Saved spline metadata has excessive width.", "resource_limit")
            pending.extend((v, depth+1) for v in item)
        elif isinstance(item, str):
            if len(item) > 4096:
                _error("Saved spline text exceeds 4096 characters.", "resource_limit")
            size += 4*len(item)
        elif item is not None and type(item) not in (int, float, bool):
            _error("Saved spline metadata must contain finite JSON primitives.", "invalid_state")
        elif type(item) is float and not math.isfinite(item) or type(item) is int and abs(item) > 2**53-1:
            _error("Saved spline numbers exceed the finite exact JSON domain.", "invalid_state")
    return size
