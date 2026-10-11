"""Calibration weighting: Deville–Särndal distance functions solved by Newton's method.

Calibrated weights ``w_k = d_k g_k`` minimise ``sum_k d_k G(g_k)`` subject to the
calibration equations ``sum_k w_k x_k = t_x`` (Deville and Särndal 1992, JASA 87;
Deville, Särndal and Sautory 1993, JASA 88). With ``F`` the inverse of ``G'`` the
solution is ``g_k = F(x_k' lambda)`` and Newton's method on the multipliers solves

    phi(lambda) = sum_k d_k F(x_k' lambda) x_k - t_x = 0,
    J(lambda)   = sum_k d_k F'(x_k' lambda) x_k x_k'.

Distances (``F``):

    linear     1 + u                               GREG / chi-square; exact after one step
    raking     exp(u)                              multiplicative; equals iterative proportional
                                                   fitting for categorical margins
    logit      L + (U - L) / (1 + exp(-(A u + c)))  A = (U-L)/((1-L)(U-1)), c = log((1-L)/(U-1));
                                                   L < g < U and F(0) = F'(0) = 1
    truncated  min(max(1 + u, L), U)               bounded linear (semismooth Newton)

These are R survey's ``cal.linear``, ``cal.raking`` and ``cal.logit`` (``cal.linear``
with finite bounds is the truncated method) and sampling::calib's methods. Each step
is damped by halving until the scaled misfit decreases (Armijo); a singular Jacobian
(units at a truncation bound) falls back to the pseudo-inverse step, as R survey's
``grake`` does. If no halving lowers the misfit (a unit sitting at a truncation kink),
the step is searched on the convex dual objective ``sum_k d_k Psi(x_k'lambda) - lambda't``
with ``Psi' = F`` instead, whose gradient is phi. Convergence requires
``max_j |phi_j| / s_j <= tol`` with the unit-free scale ``s_j = max(|t_j|, sum_k d_k |x_kj|)``.

Constraints are screened for redundancy before solving (Stata-style left-to-right
QR screen on ``sqrt(d) X``). A redundant constraint is admitted only when its target
equals the combination of the kept targets implied by the sample dependency; otherwise
the equations have no solution and the request fails. Bounded methods (and raking,
whose g-weights are positive) are also checked against the totals attainable with
every g-weight at its bounds before iterating.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.engines import linalg
from openecon.engines.contracts import KernelError
from openecon.resources import plan_workspace, tensor_bytes
from openecon.survey import SurveyDesign, _scalar

FLOAT = torch.float64
METHODS = ("linear", "raking", "logit", "truncated")
SCHEMA = "survey-calibration-v1"
MAX_ROWS = 1_000_000
MAX_GROUPS = 1_000
MAX_LEVELS = 1_000
MAX_CONSTRAINTS = 200
REFERENCE = ("Deville and Särndal (1992); Deville, Särndal and Sautory (1993); "
             "R survey calibrate()/grake() and sampling::calib distance definitions")


def _fail(message: str, code: str = "invalid_calibration_spec") -> AnalysisError:
    return AnalysisError(code, message)


def _key(value, what: str):
    """Matching identity of a level or group label: 1 and 1.0 match, True and 1 do not."""
    value = _scalar(value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Real):
        try:
            if math.isfinite(value):
                return ("number", value)
        except (OverflowError, ValueError):
            pass
    elif isinstance(value, str) and value and len(value) <= 256:
        return ("str", value)
    raise _fail(f"{what} must be finite numbers, booleans or nonempty strings of at most 256 characters.")


def _real(value, what: str, *, low: float | None = None) -> float:
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise _fail(f"{what} must be a finite real number.")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise _fail(f"{what} must be a finite real number.") from exc
    if not math.isfinite(value) or low is not None and value < low:
        raise _fail(f"{what} must be a finite real number" + ("" if low is None else f" of at least {low:g}") + ".")
    return value


def _column(name, what: str) -> str:
    if not isinstance(name, str) or not name.strip() or len(name) > 200:
        raise _fail(f"{what} must be nonempty column names of at most 200 characters.")
    return name


def _pairs(levels, column: str) -> list:
    """Category totals as [[level, count], ...]: a mapping or a sequence of pairs."""
    if isinstance(levels, Mapping):
        items = list(levels.items())
    elif isinstance(levels, Sequence) and not isinstance(levels, (str, bytes)) and all(
            isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2
            for pair in levels):
        items = [tuple(pair) for pair in levels]
    else:
        raise _fail(f"Category totals of '{column}' must map each level to its population count.")
    if not 1 <= len(items) <= MAX_LEVELS:
        raise _fail(f"'{column}' needs 1..{MAX_LEVELS} declared levels.")
    seen, out = set(), []
    for level, count in items:
        key = _key(level, f"Levels of '{column}'")
        if key in seen:
            raise _fail(f"Level {level!r} of '{column}' is declared twice.")
        seen.add(key)
        out.append([key[1], _real(count, f"The population count of {column}[{level}]", low=0.0)])
    return out


def _entry(group, population, categorical, totals, where: str) -> dict:
    entry = {"group": group, "population": None, "categorical": {}, "totals": {}}
    if population is not None:
        entry["population"] = _real(population, f"population{where}", low=0.0)
    if categorical is not None:
        if not isinstance(categorical, Mapping) or not categorical:
            raise _fail(f"categorical{where} must map columns to their level totals.")
        entry["categorical"] = {_column(c, "Categorical margins"): _pairs(levels, c) for c, levels in categorical.items()}
    if totals is not None:
        if not isinstance(totals, Mapping) or not totals:
            raise _fail(f"totals{where} must map numeric columns to population totals.")
        entry["totals"] = {_column(c, "Calibration totals"): _real(v, f"The population total of '{c}'")
                           for c, v in totals.items()}
    size = (entry["population"] is not None) + sum(len(v) for v in entry["categorical"].values()) + len(entry["totals"])
    if not size:
        raise _fail(f"Calibration{where} needs at least one constraint (population, categorical or totals).")
    if size > MAX_CONSTRAINTS:
        raise _fail(f"Calibration{where} declares {size} constraints; at most {MAX_CONSTRAINTS} are supported.")
    return entry


def _bounds(method: str, bounds) -> list | None:
    if method in ("linear", "raking"):
        if bounds is not None:
            raise _fail(f"method='{method}' is unbounded; use 'logit' or 'truncated' for bounded g-weights.")
        return None
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise _fail(f"method='{method}' needs bounds=(lower, upper) on the g-weights with lower < 1 < upper.")

    def side(value, sign):
        # None or an infinity pointing away from 1 leaves that side unbounded (stored as None).
        value = _scalar(value)
        if value is None or (isinstance(value, Real) and not isinstance(value, bool)
                             and math.isinf(value) and value * sign > 0):
            return None
        return _real(value, "Calibration bounds")

    low, high = side(bounds[0], -1), side(bounds[1], 1)
    if method == "logit" and (low is None or high is None):
        raise _fail("Logit calibration needs finite bounds (lower, upper).")
    if low is None and high is None:
        raise _fail("Truncated calibration needs at least one finite bound; use method='linear' without bounds.")
    if not ((low is None or low < 1.0) and (high is None or high > 1.0)):
        raise _fail("Calibration bounds must satisfy lower < 1 < upper.")
    return [low, high]


def _options(max_iter, tol) -> tuple[int, float]:
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 1 <= max_iter <= 1000:
        raise _fail("max_iter must be an integer in 1..1000.")
    tol = _real(tol, "tol")
    if not 1e-15 <= tol <= 1e-3:
        raise _fail("tol must lie in [1e-15, 1e-3].")
    return max_iter, tol


def calibration_spec(*, method="linear", totals=None, categorical=None, population=None, bounds=None,
                     by=None, max_iter=100, tol=1e-10) -> dict:
    """Validate svycal-style arguments into the canonical JSON calibration specification."""
    if not isinstance(method, str) or method not in METHODS:
        raise _fail(f"method must be one of {', '.join(METHODS)}.")
    limits = _bounds(method, bounds)
    max_iter, tol = _options(max_iter, tol)
    if by is None:
        groups = [_entry(None, population, categorical, totals, "")]
    else:
        _column(by, "by")
        given = {name: value for name, value in (("population", population), ("categorical", categorical),
                                                  ("totals", totals)) if value is not None}
        if not given or any(not isinstance(v, Mapping) or not v for v in given.values()):
            raise _fail("With by=, population/categorical/totals map each group label to that group's constraints.")
        labels = {}
        for value in given.values():
            labels.update({_key(g, "Group labels"): g for g in value})
        for name, value in given.items():
            if {_key(g, "Group labels") for g in value} != set(labels):
                raise _fail(f"{name} must declare constraints for exactly the same groups as the other arguments.")
        if len(labels) > MAX_GROUPS:
            raise _fail(f"At most {MAX_GROUPS} calibration groups are supported.")

        def pick(value, key):
            return None if value is None else next(v for g, v in value.items() if _key(g, "Group labels") == key)

        groups = [_entry(key[1], pick(population, key), pick(categorical, key), pick(totals, key),
                         f" for {by}={label!r}") for key, label in labels.items()]
    return {"schema": SCHEMA, "method": method, "bounds": limits, "by": by, "max_iter": max_iter,
            "tol": tol, "groups": groups}


def canonical_spec(spec) -> dict:
    """Re-validate a saved specification (``svycal(...).attrs['calibration']``, possibly via JSON)."""
    if isinstance(spec, TableSet):
        spec = spec.attrs.get("calibration")
    if not isinstance(spec, Mapping):
        raise _fail("calibration must be an svycal result, its attrs['calibration'] or a mapping of svycal arguments.")
    if "groups" not in spec:
        allowed = {"method", "totals", "categorical", "population", "bounds", "by", "max_iter", "tol"}
        if set(spec) - allowed:
            raise _fail(f"Unknown calibration arguments: {', '.join(sorted(map(str, set(spec) - allowed)))}.")
        return calibration_spec(**spec)
    if set(spec) != {"schema", "method", "bounds", "by", "max_iter", "tol", "groups"} or spec["schema"] != SCHEMA:
        raise _fail("The saved calibration specification is incomplete or has an unknown schema.")
    method = spec["method"]
    if not isinstance(method, str) or method not in METHODS:
        raise _fail(f"method must be one of {', '.join(METHODS)}.")
    groups = spec["groups"]
    if not isinstance(groups, Sequence) or not groups or len(groups) > MAX_GROUPS or (spec["by"] is None) != (
            len(groups) == 1 and isinstance(groups[0], Mapping) and groups[0].get("group") is None):
        raise _fail("The saved calibration groups are inconsistent with its by= column.")
    entries = []
    for item in groups:
        if not isinstance(item, Mapping) or set(item) != {"group", "population", "categorical", "totals"}:
            raise _fail("Each saved calibration group needs group/population/categorical/totals entries.")
        if spec["by"] is not None:
            _key(item["group"], "Group labels")
        entries.append(_entry(item["group"], item["population"], item["categorical"] or None,
                              item["totals"] or None, ""))
    if spec["by"] is not None:
        _column(spec["by"], "by")
        if len({_key(e["group"], "Group labels") for e in entries}) != len(entries):
            raise _fail("Saved calibration group labels must be unique.")
    bounds = spec["bounds"]
    if bounds is not None and isinstance(bounds, Sequence) and len(bounds) == 2:
        bounds = [-math.inf if bounds[0] is None else bounds[0], math.inf if bounds[1] is None else bounds[1]]
    max_iter, tol = _options(spec["max_iter"], spec["tol"])
    return {"schema": SCHEMA, "method": method, "bounds": _bounds(method, bounds), "by": spec["by"],
            "max_iter": max_iter, "tol": tol, "groups": entries}


# ---- tensor kernels -------------------------------------------------------------------


def _limits(spec: dict) -> tuple[float, float, bool]:
    """Range of the g-weights: (lower, upper, closed)."""
    method, bounds = spec["method"], spec["bounds"]
    if method == "linear":
        return -math.inf, math.inf, False
    if method == "raking":
        return 0.0, math.inf, False
    low, high = (-math.inf if bounds[0] is None else bounds[0], math.inf if bounds[1] is None else bounds[1])
    return low, high, method == "truncated"


def distance(method: str, low: float, high: float):
    """g = F(u) and F'(u) for one Deville–Särndal distance."""
    if method == "linear":
        return lambda u: (1.0 + u, torch.ones_like(u))
    if method == "raking":
        def raking(u):
            g = torch.exp(u)
            return g, g
        return raking
    if method == "logit":
        slope = (high - low) / ((1.0 - low) * (high - 1.0))
        shift = math.log((1.0 - low) / (high - 1.0))

        def logit(u):
            z = slope * u + shift
            return low + (high - low) * torch.sigmoid(z), (high - low) * slope * torch.sigmoid(z) * torch.sigmoid(-z)
        return logit

    def truncated(u):
        raw = 1.0 + u
        return raw.clamp(low, high), ((raw > low) & (raw < high)).to(FLOAT)
    return truncated


def primitive(method: str, low: float, high: float):
    """Psi(u) = integral of F from 0 to u: the convex dual objective is sum_k d_k Psi(x_k'lambda) - lambda't."""
    if method == "linear":
        return lambda u: u + u * u / 2.0
    if method == "raking":
        return lambda u: torch.expm1(u)
    if method == "logit":
        slope = (high - low) / ((1.0 - low) * (high - 1.0))
        shift = math.log((1.0 - low) / (high - 1.0))
        zero = math.log1p(math.exp(shift)) if shift < 30 else shift + math.log1p(math.exp(-shift))
        return lambda u: low * u + (high - low) / slope * (torch.logaddexp(torch.zeros_like(u), slope * u + shift)
                                                           - zero)

    def truncated(u):
        raw = 1.0 + u
        inner = raw.clamp(low, high)
        below = torch.where(raw < low, low * (raw - low), torch.zeros_like(raw))
        above = torch.where(raw > high, high * (raw - high), torch.zeros_like(raw))
        return (inner * inner - 1.0) / 2.0 + below + above
    return truncated


def _step(jacobian: Tensor, rhs: Tensor) -> Tensor:
    try:
        return linalg.cholesky_solve(jacobian, rhs, what="calibration Jacobian")
    except KernelError:
        # Units at a truncation bound (or underflowing logit slopes) leave J singular:
        # take the minimum-norm Newton step, as R survey's grake() does with ginv().
        return torch.linalg.pinv(jacobian, rtol=256 * torch.finfo(FLOAT).eps, hermitian=True) @ rhs


def newton(x: Tensor, d: Tensor, target: Tensor, scale: Tensor, link, *, max_iter: int, tol: float, psi=None):
    """Damped Newton iteration on the multipliers; returns (lambda, g, iterations, misfit, converged).

    Steps are halved until the squared scaled misfit decreases (Armijo). When no length
    passes -- a unit sitting at a truncation kink lets the misfit fall only for vanishing
    steps -- the same Newton direction is searched on the convex dual objective
    ``sum_k d_k Psi(x_k'lambda) - lambda't`` (``psi``; gradient phi, Hessian J), which
    decreases along it whenever phi is not orthogonal to the range of J.
    """
    lam = torch.zeros(x.shape[1], dtype=FLOAT)
    g, slope = link(x @ lam)
    if not x.shape[1]:
        return lam, g, 0, 0.0, True
    resid = x.T @ (d * g) - target
    misfit = float((resid.abs() / scale).max())
    iterations = 0

    def search(step, accept):
        length = 1.0
        while length >= 2.0 ** -40:
            trial = lam + length * step
            g_t, slope_t = link(x @ trial)
            resid_t = x.T @ (d * g_t) - target
            if bool(torch.isfinite(g_t).all()) and bool(torch.isfinite(resid_t).all()) and (
                    float((resid_t.abs() / scale).max()) <= tol or accept(trial, resid_t, length)):
                return trial, g_t, slope_t, resid_t
            length /= 2.0
        return None

    def dual(value: Tensor) -> float:
        return float(d @ psi(x @ value) - value @ target)

    while misfit > tol and iterations < max_iter:
        iterations += 1
        step = _step(x.T @ (x * (d * slope)[:, None]), -resid)
        merit = float(((resid / scale) ** 2).sum())
        found = search(step, lambda trial, r, length: float(((r / scale) ** 2).sum()) <= (1.0 - 1e-4 * length) * merit)
        descent = float(resid @ step)
        if found is None and psi is not None and descent < 0:
            current = dual(lam)
            found = search(step, lambda trial, r, length: dual(trial) <= current + 1e-4 * length * descent)
        if found is None:
            break
        lam, g, slope, resid = found
        misfit = float((resid.abs() / scale).max())
    return lam, g, iterations, misfit, misfit <= tol


# ---- constraint matrices -----------------------------------------------------------------


def _codes(series: pd.Series, column: str, pairs: list) -> Tensor:
    """Row -> declared-level index; every sampled value needs a declared level."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    codes, uniques = pd.factorize(series, sort=False)
    index = {_key(level, f"Levels of '{column}'"): i for i, (level, _) in enumerate(pairs)}
    mapping = []
    # Distinct sampled values are matched one by one; more of them than declared
    # levels always contains an undeclared value, so the loop stays bounded.
    for value in list(uniques[: len(pairs) + 1]):
        position = index.get(_key(value, f"Values of '{column}'"))
        if position is None:
            raise AnalysisError("calibration_level_missing",
                                f"Sampled value {value!r} of '{column}' has no population total; declare every sampled level.")
        mapping.append(position)
    if len(uniques) > len(pairs) or len(set(mapping)) != len(mapping):
        raise AnalysisError("calibration_level_missing",
                            f"Sampled values of '{column}' do not map one-to-one onto the declared levels.")
    return torch.tensor(mapping, dtype=torch.int64)[torch.as_tensor(codes, dtype=torch.int64)]


def constraints(frame: pd.DataFrame, entry: dict) -> tuple[Tensor, list[str], Tensor]:
    """Calibration matrix X [n, p], constraint labels and population targets."""
    blocks, labels, targets = [], [], []
    n = len(frame)
    if entry["population"] is not None:
        blocks.append(torch.ones((n, 1), dtype=FLOAT))
        labels.append("population")
        targets.append(entry["population"])
    for column, pairs in entry["categorical"].items():
        codes = _codes(frame[column], column, pairs)
        blocks.append((codes[:, None] == torch.arange(len(pairs))[None, :]).to(FLOAT))
        labels.extend(f"{column}[{level}]" for level, _ in pairs)
        targets.extend(count for _, count in pairs)
    for column, total in entry["totals"].items():
        series = frame[column]
        if not (is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype)):
            raise AnalysisError("non_numeric_column", f"Calibration variable '{column}' must be numeric; "
                                "declare categorical margins with categorical=.")
        blocks.append(_numeric(series, column)[:, None])
        labels.append(column)
        targets.append(total)
    if len(set(labels)) != len(labels):
        raise _fail("Calibration constraint labels must be unique; rename the colliding columns.")
    return torch.cat(blocks, 1), labels, torch.tensor(targets, dtype=FLOAT)


def _attainable(x: Tensor, d: Tensor, low: float, high: float) -> tuple[Tensor, Tensor]:
    """Totals reachable with every g-weight at its lower/upper bound (box relaxation)."""
    positive = d @ x.clamp(min=0.0)
    negative = d @ x.clamp(max=0.0)

    def side(first: float, second: float) -> Tensor:
        a = torch.where(positive > 0, first * positive, torch.zeros_like(positive))
        b = torch.where(negative < 0, second * negative, torch.zeros_like(negative))
        return a + b

    return side(low, high), side(high, low)


@dataclass
class GroupFit:
    rows: Tensor | None        # row positions of the group (None: every row)
    x: Tensor                  # kept constraint columns [n_g, p_kept]
    report: list[list]         # one row per declared constraint
    iterations: int
    misfit: float


@dataclass
class Calibrated:
    """Calibration of one sample: g-weights plus the kept calibration blocks for residuals."""

    spec: dict
    d: Tensor
    g: Tensor
    groups: list[GroupFit]

    @property
    def weights(self) -> Tensor:
        return self.d * self.g

    def residuals(self, u: Tensor) -> Tensor:
        """u minus its design-weighted least-squares projection on the calibration variables."""
        e = torch.empty_like(u)
        for fit in self.groups:
            rows = slice(None) if fit.rows is None else fit.rows
            solved = kernel_call(linalg.least_squares, fit.x, u[rows], self.d[rows])
            e[rows] = solved.resid
        return e

    def summary(self) -> dict:
        g = self.g
        low, high, _ = _limits(self.spec)
        return {
            "method": self.spec["method"], "bounds": self.spec["bounds"], "by": self.spec["by"],
            "n_groups": len(self.groups), "n_constraints": sum(len(f.report) for f in self.groups),
            "n_redundant": sum(r[-1] != "active" for f in self.groups for r in f.report),
            "iterations": max(f.iterations for f in self.groups), "converged": True,
            "max_relative_misfit": max(f.misfit for f in self.groups),
            "g_min": float(g.min()), "g_max": float(g.max()),
            "n_negative_weights": int((self.weights < 0).sum()),
            "n_at_bounds": int(((g <= low) | (g >= high)).sum()) if self.spec["method"] == "truncated" else 0,
            "residual_regression": "design-weighted (d_k) least squares on the kept calibration variables, within by-groups",
        }


def _fit_group(x: Tensor, labels: list[str], target: Tensor, d: Tensor, spec: dict, where: str):
    low, high, closed = _limits(spec)
    present = (x != 0).any(0)
    unreachable = torch.nonzero(~present & (target != 0)).flatten().tolist()
    if unreachable:
        j = unreachable[0]
        raise AnalysisError("calibration_infeasible",
                            f"Constraint '{labels[j]}'{where} has no sampled units but a nonzero target {float(target[j]):g}.")
    kept, omitted = kernel_call(linalg.collinear_columns, x, d)
    omitted = [j for j in omitted if bool(present[j])]
    if omitted:
        # A sample dependency X_j = X_kept b admits only the target t_j = t_kept'b.
        solved = kernel_call(linalg.least_squares, x[:, kept], x[:, omitted], d)
        coefficients = solved.beta.reshape(len(kept), len(omitted))
        implied = target[kept] @ coefficients
        size = target[omitted].abs() + target[kept].abs() @ coefficients.abs()
        for i, j in enumerate(omitted):
            if abs(float(target[j] - implied[i])) > 1e-8 * max(float(size[i]), 1e-300):
                raise AnalysisError(
                    "calibration_infeasible",
                    f"Constraint '{labels[j]}'{where} is a linear combination of other constraints in the sample, "
                    f"so its target must equal {float(implied[i]):.10g}, not {float(target[j]):.10g}; "
                    "make the population totals consistent (for example equal margin sums) or drop the constraint.")
    scale = torch.maximum(target.abs(), d @ x.abs()).clamp(min=1e-300)
    lower, upper = _attainable(x[:, kept], d, low, high)
    for i, j in enumerate(kept):
        t, s = float(target[j]), 1e-12 * float(scale[j])
        a, b = float(lower[i]), float(upper[i])
        if (t < a - s or t > b + s) if closed else (t <= a + s and a > -math.inf or t >= b - s and b < math.inf):
            raise AnalysisError(
                "calibration_infeasible",
                f"Constraint '{labels[j]}'{where} has target {t:.10g}, outside the totals {a:.10g}..{b:.10g} that "
                f"g-weights in the {'closed' if closed else 'open'} range {low:g}..{high:g} can reach; "
                "widen the bounds or check the totals.")
    link = distance(spec["method"], low, high)
    lam, g, iterations, misfit, converged = newton(x[:, kept], d, target[kept], scale[kept], link,
                                                   max_iter=spec["max_iter"], tol=spec["tol"],
                                                   psi=primitive(spec["method"], low, high))
    if not converged:
        raise AnalysisError(
            "calibration_not_converged",
            f"Calibration{where} stopped after {iterations} Newton iterations with relative misfit {misfit:.3g} "
            f"(tolerance {spec['tol']:g}); the totals may be infeasible for the '{spec['method']}' distance "
            "(widen the bounds, check the targets or raise max_iter).")
    w = d * g
    achieved = w @ x
    design = d @ x
    multipliers = dict(zip(kept, lam.tolist()))
    report = [[labels[j], float(target[j]), float(design[j]), float(achieved[j]), float(achieved[j] - target[j]),
               multipliers.get(j, math.nan), "active" if j in multipliers else
               ("redundant" if bool(present[j]) else "empty")] for j in range(len(labels))]
    full = (achieved - target).abs() / torch.maximum(target.abs(), d @ x.abs()).clamp(min=1e-300)
    return g, x[:, kept], report, iterations, float(full.max())


def calibrate(frame: pd.DataFrame, d: Tensor, spec: dict) -> Calibrated:
    """Solve the calibration equations of a canonical spec for design weights d."""
    n = len(frame)
    columns = {c for e in spec["groups"] for part in ("categorical", "totals") for c in e[part]}
    if spec["by"] is not None:
        columns.add(spec["by"])
    absent = sorted(c for c in columns if c not in frame.columns)
    if absent:
        raise AnalysisError("missing_columns", f"Calibration columns are absent: {', '.join(absent)}.")
    missing = [c for c in sorted(columns) if bool(frame[c].isna().any())]
    if missing:
        raise AnalysisError("calibration_missing", f"Calibration variables have missing values ({', '.join(missing)}); "
                            "calibration needs complete auxiliary data for every sampled unit.")
    # Groups partition the rows, so the live calibration blocks hold at most n x (widest group) cells.
    width = max((e["population"] is not None) + sum(len(v) for v in e["categorical"].values()) + len(e["totals"])
                for e in spec["groups"])
    plan_workspace("survey calibration", {
        "constraint matrices, Newton and residual temporaries": tensor_bytes((n, max(width, 1)), itemsize=48),
        "weights and indices": tensor_bytes((n,), itemsize=64),
    })
    g = torch.ones(n, dtype=FLOAT)
    fits = []
    if spec["by"] is None:
        parts = [(spec["groups"][0], None, frame, "")]
    else:
        series = frame[spec["by"]]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        codes, uniques = pd.factorize(series, sort=False)
        if len(uniques) > MAX_GROUPS:
            raise _fail(f"by='{spec['by']}' has more than {MAX_GROUPS} groups.")
        entries = {_key(e["group"], "Group labels"): e for e in spec["groups"]}
        codes = torch.as_tensor(codes, dtype=torch.int64)
        parts, seen = [], set()
        for i, label in enumerate(uniques):
            key = _key(label, f"Values of '{spec['by']}'")
            if key not in entries:
                raise AnalysisError("calibration_level_missing",
                                    f"Sampled group {spec['by']}={label!r} has no calibration constraints.")
            seen.add(key)
            rows = torch.nonzero(codes == i).flatten()
            parts.append((entries[key], rows, frame.iloc[rows.numpy()], f" in {spec['by']}={label!r}"))
        unused = [e["group"] for k, e in entries.items() if k not in seen]
        if unused:
            raise AnalysisError("calibration_infeasible",
                                f"Calibration groups {unused!r} of '{spec['by']}' have no sampled units.")
    for entry, rows, sub, where in parts:
        x, labels, target = constraints(sub, entry)
        dg = d if rows is None else d[rows]
        gg, kept, report, iterations, misfit = _fit_group(x, labels, target, dg, spec, where)
        if rows is None:
            g = gg
        else:
            g[rows] = gg
        for row in report:
            row.insert(0, entry["group"])
        fits.append(GroupFit(rows, kept, report, iterations, misfit))
    if not bool(torch.isfinite(d * g).all()):
        raise AnalysisError("calibration_not_converged", "Calibrated weights overflowed float64; rescale the totals.")
    return Calibrated(spec, d, g, fits)


def design_weights(frame: pd.DataFrame, design, data) -> tuple[Tensor, dict]:
    """Design weights from a revalidated SurveyDesign or a positive weight column."""
    if isinstance(design, SurveyDesign):
        validation = design.revalidate(data)
        return _numeric(frame[design.weights], design.weights), {
            "design": "SurveyDesign", "weights": design.weights,
            "design_input_sha256": validation.design_input_sha256}
    if not isinstance(design, str) or design not in frame.columns:
        raise AnalysisError("invalid_survey_design",
                            "design must be a validated single-stage SurveyDesign (oe.survey_design) or the name of "
                            "a design-weight column; multistage declarations are not supported.")
    if bool(frame[design].isna().any()):
        raise AnalysisError("invalid_survey_weights", "Design weights must be complete.")
    d = _numeric(frame[design], design)
    if bool((d <= 0).any()):
        raise AnalysisError("invalid_survey_weights", "Design weights must be strictly positive and finite.")
    return d, {"design": "weight column", "weights": design}


def svycal(data, design, *, method="linear", totals=None, categorical=None, population=None, bounds=None,
           by=None, max_iter=100, tol=1e-10) -> TableSet:
    """Calibrate survey design weights to known population totals (Stata ``svycal``, R ``calibrate``).

    The calibrated weights ``w_k = d_k g_k`` satisfy ``sum_k w_k x_k = t_x`` and minimise
    the Deville–Särndal distance of ``method`` from the design weights ``d_k``; Newton's
    method on the Lagrange multipliers solves the calibration equations.

    Parameters
    ----------
    data : DataFrame, column mapping or list of row records with every sampled unit.
    design : SurveyDesign (its weights are the design weights; the declaration is
        revalidated) or the name of a strictly positive design-weight column.
    method : 'linear' (GREG, chi-square distance; may produce negative weights),
        'raking' (multiplicative; equals raking ratio / iterative proportional fitting
        for categorical margins), 'logit' (bounded, lower < g < upper) or 'truncated'
        (bounded linear, lower <= g <= upper).
    totals : mapping numeric column -> population total.
    categorical : mapping column -> {level: population count} (or [[level, count], ...]);
        one indicator constraint per declared level. Every sampled value needs a level;
        a declared level absent from the sample needs a zero count.
    population : population size N (constraint on the sum of the weights).
    bounds : (lower, upper) on the g-weights, required by 'logit' and 'truncated'
        (lower < 1 < upper; 'truncated' accepts one infinite side).
    by : column defining groups calibrated separately (stratum-wise calibration). Then
        population/categorical/totals map each group label to that group's constraints.
    max_iter, tol : Newton iteration limit and convergence tolerance on
        ``max_j |sum_k w_k x_kj - t_j| / max(|t_j|, sum_k d_k |x_kj|)``.

    Returns
    -------
    TableSet with ``weights`` (one row per input row, input index: design_weight,
    g_weight, calibrated_weight) and ``totals`` (one row per constraint: target,
    design_total, calibrated_total, difference, multiplier, status). ``attrs`` holds
    convergence/g-weight diagnostics and ``attrs['calibration']``, the canonical JSON
    specification accepted by ``survey_calibrated_*`` and ``survey_pps_*``.
    """
    spec = calibration_spec(method=method, totals=totals, categorical=categorical, population=population,
                            bounds=bounds, by=by, max_iter=max_iter, tol=tol)
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates or not len(frame) or len(frame) > MAX_ROWS:
        raise AnalysisError("invalid_data", f"Calibration needs 1..{MAX_ROWS:,} rows with unique column names.")
    d, source = design_weights(frame, design, data)
    fit = calibrate(frame, d, spec)
    weights = table({"design_weight": fit.d.numpy(), "g_weight": fit.g.numpy(),
                     "calibrated_weight": fit.weights.numpy()},
                    columns=["design_weight", "g_weight", "calibrated_weight"], index=list(frame.index))
    rows = [row for group in fit.groups for row in group.report]
    labels = [label if by is None else f"{by}={group}: {label}" for group, label, *_ in rows]
    totals_table = table([row[2:] for row in rows], index=labels,
                         columns=["target", "design_total", "calibrated_total", "difference", "multiplier", "status"])
    summary = fit.summary()
    notes = []
    if summary["n_negative_weights"]:
        notes.append(f"{summary['n_negative_weights']} calibrated weight(s) are negative; "
                     "bounded methods ('logit', 'truncated') keep g-weights within limits.")
    return TableSet({"weights": weights, "totals": totals_table}, title=f"Calibration weights ({method})",
                    **summary, n=len(frame), sum_design_weights=float(fit.d.sum()),
                    sum_calibrated_weights=float(fit.weights.sum()), source=source, calibration=spec,
                    reference=REFERENCE, stata_parity_validated=False, notes=notes)
