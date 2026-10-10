"""Bounded canonical Gaussian efficacy stopping laws, native CPU float64.

At information fraction t, Z(t) has mean drift*sqrt(t) and Brownian
correlation sqrt(min(t,s)/max(t,s)). All exits are *first* crossings.
Two-sided lower exits reject in the opposite direction; they are not futility.
Gauss--Legendre order means nodes per fixed panel of width at most six.
The refinement diagnostic is an empirical numerical admission check, not a
rigorous quadrature error theorem. One-sided truncation has a separate bound.
"""

from __future__ import annotations

from functools import lru_cache
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import normal_isf
from openecon.resources import plan_workspace

DTYPE = torch.float64
MAX_WORK = 2_000_000_000
MAX_BOUND = 12.0
TAIL_DISTANCE = 12.0
PANEL_WIDTH = 6.0
BISECTION_ITERATIONS = 60
CONSERVATION_TOLERANCE = 2e-10
CALIBRATION_TOLERANCE = 2e-11
DEFAULT_TOLERANCE = 1e-8
_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_TAIL = 0.5 * math.erfc(TAIL_DISTANCE / _SQRT_TWO)


def _fail(code, message):
    raise AnalysisError(code, message)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail("invalid_sequential_input", f"{name} must be a finite real number.")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_sequential_input", f"{name} must be a finite real number.") from exc
    if not math.isfinite(value):
        _fail("invalid_sequential_input", f"{name} must be a finite real number.")
    return value


def _vector(value, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype != DTYPE or value.ndim != 1:
            _fail("invalid_sequential_input", f"{name} tensors must be one-dimensional CPU float64.")
        value = value.tolist()
    if not isinstance(value, (tuple, list)):
        _fail("invalid_sequential_input", f"{name} must be a real vector.")
    return tuple(_real(item, name) for item in value)


def _sides(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (1, 2):
        _fail("invalid_sequential_input", "sides must be the integer 1 or 2.")
    return int(value)


def _fractions(value):
    t = _vector(value, "fractions")
    if not 2 <= len(t) <= 6 or t[0] < 0.2 or t[-1] != 1.0:
        _fail("unsupported_information_grid", "Require 2..6 looks, first fraction >=0.2 and final fraction exactly 1.")
    # Eight ulps only accommodate arithmetic representation of a .05 gap.
    if any(b - a < 0.05 - 8e-16 for a, b in zip(t, t[1:])):
        _fail("unsupported_information_grid", "Information fractions must increase with gaps >=0.05.")
    return t


def _order(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (64, 128, 256):
        _fail("invalid_quadrature_order", "Quadrature order must be 64, 128 or 256 nodes per panel.")
    return int(value)


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= MAX_WORK:
        _fail("invalid_resource_budget", "max_work must be an integer in [1,2000000000].")
    return int(value)


def _settings(fractions, bounds, drift, sides, order):
    t, b = _fractions(fractions), _vector(bounds, "bounds")
    if len(t) != len(b) or any(not 0.0 < item <= MAX_BOUND for item in b):
        _fail("invalid_sequential_input", "One positive boundary <=12 is required per look.")
    d = _real(drift, "drift")
    if abs(d) > 8:
        _fail("unsupported_drift", "Canonical standardized drift is bounded by |drift|<=8.")
    return t, b, d, _sides(sides), _order(order)


def _law(law, param):
    if not isinstance(law, str) or law.lower() not in ("ldof", "ldpocock", "hsd", "power"):
        _fail("invalid_spending_law", "Spending law must be LDOF, LDPocock, HSD or power.")
    law = law.lower()
    if law in ("ldof", "ldpocock"):
        if param is not None:
            _fail("unsupported_spending_parameter", "LDOF (rho=1) and LDPocock have no free parameter here.")
        return law, None
    param = _real((-4.0 if law == "hsd" else 2.0) if param is None else param, "spending parameter")
    if (law == "hsd" and not -8 <= param <= 8) or (law == "power" and not 0.25 <= param <= 4):
        _fail("unsupported_spending_parameter", "HSD gamma must be in [-8,8]; power rho in [.25,4].")
    return law, param


def spend(fractions, total_alpha, law, param=None, sides=1):
    """Cumulative TOTAL alpha: sides * official law(alpha/sides, t)."""
    t, s = _fractions(fractions), _sides(sides)
    alpha = _real(total_alpha, "total_alpha")
    if not 0.01 <= alpha <= 0.1:
        _fail("unsupported_alpha", "Total alpha must be in [.01,.1].")
    law, param = _law(law, param)
    a = alpha / s
    with torch.device("cpu"), torch.no_grad():
        values = torch.tensor(t, dtype=DTYPE, device="cpu")
        if law == "ldof":
            # R sfLDOF(a,t,rho=1): 2 * sf(qnorm(1-a/2)/sqrt(t)).
            cumulative = s * torch.erfc(normal_isf(a / 2) / torch.sqrt(2 * values))
        elif law == "ldpocock":
            cumulative = alpha * torch.log1p((math.e - 1) * values)
        elif law == "hsd":
            cumulative = alpha * values if param == 0 else alpha * torch.expm1(-param * values) / math.expm1(-param)
        else:
            cumulative = alpha * values.pow(param)
        # The supplied alpha is the exact terminal target, not a rounded tail.
        cumulative[-1] = alpha
        if not bool(torch.isfinite(cumulative).all()) or not bool((cumulative[1:] > cumulative[:-1]).all()):
            _fail("numerical_failure", "Spending targets are not finite and strictly increasing.")
        return cumulative


@lru_cache(maxsize=3)
def _nodes(order):
    """Golub--Welsch Jacobi eigensystem; cache immutable CPU node/weight arrays."""
    with torch.device("cpu"), torch.no_grad():
        index = torch.arange(1, order, dtype=DTYPE, device="cpu")
        off = index / torch.sqrt(4 * index.square() - 1)
        jacobi = torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        nodes, vectors = torch.linalg.eigh(jacobi)
        weights = 2 * vectors[0].square()
        return nodes, weights


def _domain(t, b, drift, sides):
    return (-b, b) if sides == 2 else (drift * math.sqrt(t) - TAIL_DISTANCE, b)


def _count_nodes(t, b, drift, sides, order):
    lower, upper = _domain(t, b, drift, sides)
    return max(1, math.ceil((upper - lower) / PANEL_WIDTH)) * order


def _work_components(t, b, drift, sides, order):
    counts = [_count_nodes(ti, bi, drift, sides, order) for ti, bi in zip(t, b)]
    # Preparation is reported deterministically even with a warm cache. An
    # aggregate operation can count it once per distinct order, then count
    # integration for every call. An isolated primitive budgets both pieces.
    preparation = 12 * order**3 + 32 * order**2
    integration = 300 * sum(counts)
    integration += 48 * sum(left * right for left, right in zip(counts, counts[1:]))
    return int(preparation), int(integration), max(counts)


def probability_work_bound(fractions, bounds, drift=0.0, sides=1, order=256):
    """Allocation-free cost components for a caller's cumulative work admission."""
    t, b, drift, sides, order = _settings(fractions, bounds, drift, sides, order)
    preparation, integration, count = _work_components(t, b, drift, sides, order)
    return {"work_bound": preparation + integration,
            "preparation_work_bound": preparation, "integration_work_bound": integration,
            "maximum_nodes_per_look": count}


def _admit(operation, work, count, order, max_work):
    limit = _budget(max_work)
    if work > limit:
        _fail("resource_limit", f"{operation} work bound {work} exceeds max_work={limit}.")
    return plan_workspace(operation, {
        "transition_and_density_temporaries": 8 * 5 * count**2,
        "nodes_weights_masses_and_tail_vectors": 8 * 20 * count,
        "cold_jacobi_eigensystem": 8 * 6 * order**2,
    }).record()


def _grid(t, b, drift, sides, order):
    lower, upper = _domain(t, b, drift, sides)
    panels = max(1, math.ceil((upper - lower) / PANEL_WIDTH))
    nodes, weights = _nodes(order)
    edges = torch.linspace(lower, upper, panels + 1, dtype=DTYPE, device="cpu")
    half = (edges[1:] - edges[:-1]) / 2
    centers = (edges[1:] + edges[:-1]) / 2
    return (centers[:, None] + half[:, None] * nodes).flatten(), (half[:, None] * weights).flatten()


def _transition(t, previous_t, drift, previous_nodes):
    if previous_nodes is None:
        return drift * math.sqrt(t), 1.0
    delta = t - previous_t
    return math.sqrt(previous_t / t) * previous_nodes + drift * delta / math.sqrt(t), math.sqrt(delta / t)


def _crossing(bound, means, sd, masses, sides):
    upper = 0.5 * torch.erfc((bound - means) / (sd * _SQRT_TWO))
    lower = 0.5 * torch.erfc((bound + means) / (sd * _SQRT_TWO)) if sides == 2 else None
    if masses is None:
        return float(upper), float(lower) if lower is not None else 0.0
    return float(masses @ upper), float(masses @ lower) if lower is not None else 0.0


def _density(t, bound, drift, sides, order, means, sd, previous_masses):
    nodes, weights = _grid(t, bound, drift, sides, order)
    if previous_masses is None:
        density = torch.exp(-0.5 * ((nodes - means) / sd).square()) / (sd * _SQRT_TWO_PI)
    else:
        standardized = (nodes[:, None] - means[None, :]) / sd
        transition = torch.exp(-0.5 * standardized.square()) / (sd * _SQRT_TWO_PI)
        density = transition @ previous_masses
    return nodes, density * weights


def _walk(t, bounds, drift, sides, order):
    upper, lower, continuation, reach = [], [], [], []
    previous_nodes = previous_masses = None
    previous_t, errors = 0.0, []
    for ti, bound in zip(t, bounds):
        mass_before = 1.0 if previous_masses is None else float(previous_masses.sum())
        reach.append(mass_before)
        means, sd = _transition(ti, previous_t, drift, previous_nodes)
        means = torch.as_tensor(means, dtype=DTYPE, device="cpu")
        up, lo = _crossing(bound, means, sd, previous_masses, sides)
        nodes, masses = _density(ti, bound, drift, sides, order, means, sd, previous_masses)
        remaining = float(masses.sum())
        upper.append(up)
        lower.append(lo)
        continuation.append(remaining)
        errors.append(abs(mass_before - up - lo - remaining))
        previous_nodes, previous_masses, previous_t = nodes, masses, ti
    power = math.fsum(upper + lower)
    error = max(*errors, abs(power + continuation[-1] - 1.0))
    values = upper + lower + continuation + reach + [power]
    if any(not math.isfinite(value) or not -CONSERVATION_TOLERANCE <= value <= 1 + CONSERVATION_TOLERANCE for value in values):
        _fail("numerical_failure", "Sequential probabilities are outside [0,1].")
    if error > CONSERVATION_TOLERANCE:
        _fail("numerical_failure", "Sequential quadrature failed the probability-conservation gate.")
    expected = t[0] + math.fsum((t[i] - t[i-1]) * reach[i] for i in range(1, len(t)))
    return {"upper": upper, "lower": lower, "continuation": continuation, "reach": reach,
            "power": power, "expectedfraction": expected, "error": error,
            "tail_bound": len(t) * _TAIL if sides == 1 else 0.0}


def probabilities(fractions, bounds, drift=0.0, sides=1, order=256, *, max_work=MAX_WORK):
    """First exits/reach and expected information for externally fixed boundaries.

    This primitive performs no boundary search. Call at both saved orders for
    the refinement admission. error records mass-conservation discrepancy,
    whereas tail_bound separately bounds omitted one-sided unbounded tails.
    """
    t, b, drift, sides, order = _settings(fractions, bounds, drift, sides, order)
    preparation, integration, count = _work_components(t, b, drift, sides, order)
    work = preparation + integration
    workspace = _admit("sequential probabilities", work, count, order, max_work)
    with torch.device("cpu"), torch.no_grad():
        result = _walk(t, b, drift, sides, order)
    return {**result, "order": order, "work_bound": work,
            "preparation_work_bound": preparation, "integration_work_bound": integration,
            "workspace": workspace}


def _calibrate_at_order(t, spending, sides, order):
    bounds, iterations = [], []
    previous_nodes = previous_masses = None
    previous_t, spent = 0.0, 0.0
    for ti, target in zip(t, spending):
        means, sd = _transition(ti, previous_t, 0.0, previous_nodes)
        means = torch.as_tensor(means, dtype=DTYPE, device="cpu")
        increment = target - spent
        low, high = 0.0, MAX_BOUND
        max_cross = math.fsum(_crossing(low, means, sd, previous_masses, sides))
        min_cross = math.fsum(_crossing(high, means, sd, previous_masses, sides))
        if not min_cross < increment < max_cross:
            _fail("numerical_failure", "Spending increment cannot be calibrated inside boundary [0,12].")
        for _ in range(BISECTION_ITERATIONS):
            middle = (low + high) / 2
            crossing = math.fsum(_crossing(middle, means, sd, previous_masses, sides))
            if crossing > increment:
                low = middle
            else:
                high = middle
        bound = (low + high) / 2
        spent += math.fsum(_crossing(bound, means, sd, previous_masses, sides))
        if abs(spent - target) > CALIBRATION_TOLERANCE:
            _fail("numerical_failure", "Boundary calibration failed the cumulative-spending gate.")
        bounds.append(bound)
        iterations.append(BISECTION_ITERATIONS)
        previous_nodes, previous_masses = _density(ti, bound, 0.0, sides, order, means, sd, previous_masses)
        previous_t = ti
    return bounds, iterations


def probability_refinement_error(coarse, fine):
    """Maximum absolute difference over all scientific probability outputs."""
    differences = [abs(coarse[key] - fine[key]) for key in ("power", "expectedfraction")]
    for key in ("upper", "lower", "continuation", "reach"):
        if len(coarse[key]) != len(fine[key]):
            _fail("invalid_sequential_input", "Probability refinement vectors must have identical lengths.")
        differences.extend(abs(a - b) for a, b in zip(coarse[key], fine[key]))
    return max(differences)


def calibration_work_bound(fractions, sides=1, order=128, refine_order=256, max_work=MAX_WORK):
    """Reproduce the complete deterministic plan without allocation or search."""
    t, sides, order, refined = _fractions(fractions), _sides(sides), _order(order), _order(refine_order)
    if refined != 2 * order:
        _fail("invalid_quadrature_order", "Use 64/128 or 128/256 refinement.")
    coarse_preparation, coarse_work, coarse_count = _work_components(t, (MAX_BOUND,) * len(t), 0.0, sides, order)
    fine_preparation, fine_work, fine_count = _work_components(t, (MAX_BOUND,) * len(t), 0.0, sides, refined)
    # Two calibrations, their full independent walks, and a third walk at fine
    # boundaries/coarse order. Each calibration's 60 tail-CDF evaluations and
    # its two bracket checks are counted in addition to density transitions.
    bisection_work = 48 * (BISECTION_ITERATIONS + 2) * len(t) * (coarse_count + fine_count)
    integration_work = 3 * coarse_work + 2 * fine_work + bisection_work
    preparation_work = coarse_preparation + fine_preparation
    work = preparation_work + integration_work
    workspace = _admit("sequential calibration", work, max(coarse_count, fine_count), refined, max_work)
    return {"work_bound": work, "max_work": int(max_work),
            "preparation_work_bound": preparation_work,
            "integration_work_bound": integration_work,
            "bisection_work_bound": bisection_work,
            "workspace": workspace, "panel_width": PANEL_WIDTH,
            "nodes_per_panel": [order, refined],
            "bisection_iterations": {"coarse": [BISECTION_ITERATIONS] * len(t),
                                      "fine": [BISECTION_ITERATIONS] * len(t)},
            "tail_distance": TAIL_DISTANCE, "tail_bound": len(t) * _TAIL if sides == 1 else 0.0,
            "conservation_tolerance": CONSERVATION_TOLERANCE,
            "calibration_tolerance": CALIBRATION_TOLERANCE,
            "quadrature_error_bound_is_rigorous": False}


def calibrate(fractions, total_alpha, law, param=None, sides=1, *, order=128,
              refine_order=256, tolerance=DEFAULT_TOLERANCE, max_work=MAX_WORK):
    """Calibrate both orders, retaining every boundary needed for search-free replay."""
    t, sides, order, refined = _fractions(fractions), _sides(sides), _order(order), _order(refine_order)
    tolerance = _real(tolerance, "tolerance")
    if refined != 2 * order or not 1e-12 <= tolerance <= DEFAULT_TOLERANCE:
        _fail("invalid_quadrature_order", "Use 64/128 or 128/256 refinement and tolerance in [1e-12,1e-8].")
    targets = spend(t, total_alpha, law, param, sides).tolist()
    numerical_receipt = calibration_work_bound(t, sides, order, refined, max_work)
    with torch.device("cpu"), torch.no_grad():
        bounds_coarse, coarse_iterations = _calibrate_at_order(t, targets, sides, order)
        bounds, fine_iterations = _calibrate_at_order(t, targets, sides, refined)
    if numerical_receipt["bisection_iterations"] != {"coarse": coarse_iterations, "fine": fine_iterations}:
        _fail("numerical_failure", "Calibration did not follow the prespecified iteration plan.")
    coarse = probabilities(t, bounds_coarse, 0.0, sides, order, max_work=max_work)
    fine = probabilities(t, bounds, 0.0, sides, refined, max_work=max_work)
    fine_at_coarse = probabilities(t, bounds, 0.0, sides, order, max_work=max_work)
    boundary_error = max(abs(a-b) for a, b in zip(bounds_coarse, bounds))
    probability_error = max(probability_refinement_error(coarse, fine),
                            probability_refinement_error(fine_at_coarse, fine))
    calibration_errors = []
    for result in (coarse, fine):
        cumulative = 0.0
        for up, lo, target in zip(result["upper"], result["lower"], targets):
            cumulative += up + lo
            calibration_errors.append(abs(cumulative - target))
    calibration_error = max(calibration_errors)
    if max(boundary_error, probability_error) > tolerance or calibration_error > CALIBRATION_TOLERANCE:
        _fail("numerical_failure", "Sequential boundary/probability refinement did not meet the declared tolerance.")
    return {"bounds": bounds, "bounds_coarse": bounds_coarse, "spending": targets,
            "probabilities": fine, "probabilities_coarse": coarse,
            "probabilities_fine_bounds_coarse": fine_at_coarse,
            "order": order, "refine_order": refined, "tolerance": tolerance,
            "max_refinement_error": max(boundary_error, probability_error),
            "boundary_refinement_error": boundary_error,
            "probability_refinement_error": probability_error,
            "calibration_error": calibration_error, "numerical_receipt": numerical_receipt}
