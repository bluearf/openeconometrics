"""Exponential smoothing (Stata's ``tssmooth``; SPSS Exponential Smoothing; EViews smoothing).

Four smoothers, all written in error-correction form with the one-step error
e_t = x_t - f_t, where f_t is the forecast of x_t made at t - 1:

``exponential``   S_t = S_(t-1) + alpha e_t,                               f_t = S_(t-1)
``dexponential``  Brown's double smoothing S1_t = alpha x_t + (1-alpha) S1_(t-1),
                  S2_t = alpha S1_t + (1-alpha) S2_(t-1); equivalently, with
                  f_t = a_(t-1) + b_(t-1),
                  a_t = f_t + alpha (2 - alpha) e_t,   b_t = b_(t-1) + alpha^2 e_t
``hwinters``      Holt, f_t = a_(t-1) + b_(t-1):
                  a_t = alpha x_t + (1-alpha)(a_(t-1) + b_(t-1)),
                  b_t = beta (a_t - a_(t-1)) + (1-beta) b_(t-1)
``shwinters``     seasonal Holt-Winters with period m and the same trend recursion b_t;
                  additive, f_t = a_(t-1) + b_(t-1) + s_(t-m):
                      a_t = alpha (x_t - s_(t-m)) + (1-alpha)(a_(t-1) + b_(t-1)),
                      s_t = gamma (x_t - a_t) + (1-gamma) s_(t-m)
                  multiplicative, f_t = (a_(t-1) + b_(t-1)) s_(t-m):
                      a_t = alpha x_t / s_(t-m) + (1-alpha)(a_(t-1) + b_(t-1)),
                      s_t = gamma x_t / a_t + (1-gamma) s_(t-m)

Parameters not supplied are chosen to minimize the in-sample sum of squared
one-step errors (as Stata, SPSS and EViews do): BFGS (``optimize.maximize_bfgs``)
on -n/2 ln(SSE/n) with the exact analytic gradient, in the parameterization
parameter = sin(theta)^2, which keeps every parameter in [0, 1] and makes an
optimum on a bound an ordinary stationary point. The criterion can have several
local minima (often one in a corner of the unit cube), so the search descends from
the three best points of a coarse grid that includes the bounds and keeps the
lowest minimum; a parameter that ends on 0 or 1 is fixed there exactly and the
others are re-optimized.

No time loop for the linear smoothers. Each of them is a linear filter: the
errors solve theta(L) e_t = Delta(L) x_t with

    exponential     (1 - L) x_t        = e_t - (1 - alpha) e_(t-1)
    trend models    (1 - L)^2 x_t      = e_t - (2 - k1 - k2) e_(t-1) + (1 - k1) e_(t-2)
    additive HW     (1 - L)(1 - L^m) x_t = e_t + (k1 + k2 - 1) e_(t-1) + k2 sum_(j=2..m-1) e_(t-j)
                                           + (k2 + k3 - 1) e_(t-m) + (1 - k1 - k3) e_(t-m-1)

(k1 = alpha, k2 = alpha beta, k3 = gamma (1 - alpha); Brown: k1 = alpha (2 - alpha),
k2 = alpha^2), where the presample values of x continue the initial level, trend and
seasonal pattern backwards and presample errors are zero. ``filters.inverse_filter``
solves these recursions for all t at once, and d e / d k = -(d theta / d k)(L) e /
theta(L) gives the derivatives. Only the multiplicative seasonal smoother is
nonlinear; it runs a scalar recursion over time with forward derivative recursions.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import _require_variation, load_series
from openecon.econometrics.arima.filters import inverse_filter, shift
from openecon.econometrics.core import kernel_call, table
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares

METHODS = {"exponential": ("alpha",), "dexponential": ("alpha",), "hwinters": ("alpha", "beta"),
           "shwinters": ("alpha", "beta", "gamma")}
_GRID = (0.0, 0.1, 0.5, 0.9, 1.0)   # starting-value grid of every free parameter (with bounds)
_STARTS = 3                 # grid points from which the criterion is minimized
_BOUND = 1e-3               # distance from 0 or 1 that counts as the bound when BFGS stalls
_CONVERGED_BOUND = 1e-6     # the same for a converged iteration
_INTERIOR = 1e-3            # starting values are kept this far inside (0, 1)
_GRID_ROWS = 5_000          # observations used by the starting-value grid and its descents


def _line(x: Tensor) -> tuple[float, float]:
    """Intercept (value at t = 0) and slope of the OLS line through x_1..x_len."""
    count = x.shape[0]
    design = torch.stack([torch.ones(count, dtype=torch.float64),
                          torch.arange(1, count + 1, dtype=torch.float64)], dim=1)
    beta = kernel_call(least_squares, design, x.contiguous()).beta
    return float(beta[0]), float(beta[1])


class _Smoother:
    """One smoothing model on one series: errors, their derivatives and the fitted states."""

    def __init__(self, x: Tensor, method: str, period: int, additive: bool):
        self.x, self.n, self.method, self.m, self.additive = x, x.shape[0], method, period, additive
        n, m = self.n, period
        half = max(2, n // 2)
        self.seasonal0: Tensor | None = None
        if method == "exponential":
            self.level0, self.trend0 = float(x[:max(1, n // 2)].mean()), 0.0
        elif method in {"dexponential", "hwinters"}:
            self.level0, self.trend0 = _line(x[:half])
        else:
            seasons = max(2, min(n // m, (n // 2) // m))
            block = x[:seasons * m].reshape(seasons, m)
            means = block.mean(dim=1)
            self.trend0 = float(means[-1] - means[0]) / ((seasons - 1) * m)
            self.level0 = float(means[0]) - self.trend0 * (m + 1) / 2.0
            trend = self.level0 + self.trend0 * torch.arange(
                1, seasons * m + 1, dtype=torch.float64).reshape(seasons, m)
            if additive:
                seasonal = (block - trend).mean(dim=0)
                self.seasonal0 = seasonal - seasonal.mean()
            else:
                if bool((trend <= 0).any()) or bool((x <= 0).any()):
                    raise AnalysisError("nonpositive_series", "Multiplicative seasonal smoothing "
                                        "needs a strictly positive series; use additive=True.")
                seasonal = (block / trend).mean(dim=0)
                self.seasonal0 = seasonal / seasonal.mean()

    # ---- linear smoothers: gains, the MA polynomial and its derivatives ------------------

    def _gains(self, params: dict[str, float]) -> tuple[list[float], dict[str, list[float]]]:
        """(k1, k2, k3) and d k / d parameter."""
        a = params["alpha"]
        if self.method == "exponential":
            return [a, 0.0, 0.0], {"alpha": [1.0, 0.0, 0.0]}
        if self.method == "dexponential":
            return [a * (2.0 - a), a * a, 0.0], {"alpha": [2.0 - 2.0 * a, 2.0 * a, 0.0]}
        b = params["beta"]
        if self.method == "hwinters":
            return [a, a * b, 0.0], {"alpha": [1.0, b, 0.0], "beta": [0.0, a, 0.0]}
        g = params["gamma"]
        return [a, a * b, g * (1.0 - a)], {"alpha": [1.0, b, -g], "beta": [0.0, a, 0.0],
                                           "gamma": [0.0, 0.0, 1.0 - a]}

    def _polynomial(self, gains: list[float],
                    frozen: bool) -> tuple[list[float], list[list[tuple[int, float]]]]:
        """Coefficients theta_1.. of the error polynomial and d theta / d k as (lag, value)."""
        k1, k2, k3 = gains
        if self.method == "exponential":
            return [k1 - 1.0], [[(1, 1.0)], [], []]
        if self.method != "shwinters" or frozen:
            return [k1 + k2 - 2.0, 1.0 - k1], [[(1, 1.0), (2, -1.0)], [(1, 1.0)], []]
        m = self.m
        theta = [k2] * (m + 1)
        theta[0] += k1 - 1.0
        theta[m - 1] += k3 - 1.0
        theta[m] = 1.0 - k1 - k3
        return theta, [[(1, 1.0), (m + 1, -1.0)], [(j, 1.0) for j in range(1, m + 1)],
                       [(m, 1.0), (m + 1, -1.0)]]

    def _input(self, frozen: bool) -> Tensor:
        """Delta(L) x with the presample values implied by the initial state."""
        x, a0, b0 = self.x, self.level0, self.trend0
        if self.method == "exponential":
            return x - torch.cat([torch.tensor([a0], dtype=torch.float64), x[:-1]])
        if self.method != "shwinters" or frozen:
            if frozen:                      # the seasonal terms never change: remove them
                x = x - self.seasonal0[torch.arange(self.n) % self.m]
            ext = torch.cat([torch.tensor([a0 - b0, a0], dtype=torch.float64), x])
            return ext[2:] - 2.0 * ext[1:-1] + ext[:-2]
        m = self.m
        tau = torch.arange(-m, 1, dtype=torch.float64)
        pattern = self.seasonal0[torch.arange(-m, 1) % m - 1]     # s_tau, periodic; s_0 = s_m
        ext = torch.cat([a0 + b0 * tau + pattern, x])
        first = ext[1:] - ext[:-1]
        return first[m:] - first[:-m]

    def _errors(self, params: dict[str, float], derivatives: bool,
                names: list[str] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        """One-step errors and, with ``derivatives``, d e / d parameter for ``names``."""
        if self.method == "shwinters" and not self.additive:
            return self._multiplicative(params, derivatives)
        gains, chain = self._gains(params)
        names = list(chain) if names is None else names
        # With a zero seasonal gain (gamma = 0 or alpha = 1) the seasonal terms keep their
        # initial values and the smoother is the trend smoother of the deseasonalized
        # series: a second-order filter instead of one with m + 1 roots on the unit circle.
        frozen = self.method == "shwinters" and gains[2] == 0.0 and not (
            derivatives and any(chain[name][2] != 0.0 for name in names))
        theta, d_theta = self._polynomial(gains, frozen)
        errors = inverse_filter(self._input(frozen)[None], theta)
        slopes: dict[str, Tensor] = {}
        if derivatives:
            z = inverse_filter(errors, theta)
            by_gain = []
            for terms in d_theta:
                total = torch.zeros_like(z)
                for lag, weight in terms:
                    total -= weight * shift(z, lag)
                by_gain.append(total[0])
            for name in names:
                slopes[name] = torch.zeros(self.n, dtype=torch.float64)
                for i, weight in enumerate(chain[name]):
                    if weight:
                        slopes[name] += weight * by_gain[i]
        return errors[0], slopes

    def _multiplicative(self, params: dict[str, float], derivatives: bool,
                        states: bool = False) -> Any:
        """Scalar recursion of the multiplicative seasonal smoother (a loop over time).

        The recursion is nonlinear in the data, so this is the one smoother that runs a
        Python loop over the periods; the loop body is plain float arithmetic. With
        ``derivatives`` the forward derivative recursions of (a, b, s) by alpha, beta
        and gamma run alongside (unrolled, one block per parameter).
        """
        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
        oma, omb, omg = 1.0 - alpha, 1.0 - beta, 1.0 - gamma
        n, m, xs = self.n, self.m, self.x.tolist()
        a, b = self.level0, self.trend0
        s = self.seasonal0.tolist()                    # s[i] is the index used at t with t % m == i
        errors = [0.0] * n
        bad = torch.full((n,), math.nan, dtype=torch.float64)
        i = 0
        if not derivatives:
            level = [0.0] * n if states else errors
            for t in range(n):
                x, si = xs[t], s[i]
                base = a + b
                errors[t] = x - base * si
                grown = alpha * x / si + oma * base
                b = beta * (grown - a) + omb * b
                a = grown
                if not 0.0 < a < math.inf:
                    return (bad, bad, a, b, s) if states else (bad, {})
                si = gamma * x / a + omg * si
                if not si > 0.0:
                    return (bad, bad, a, b, s) if states else (bad, {})
                s[i] = si
                if states:
                    level[t] = a * si
                i += 1
                if i == m:
                    i = 0
            tensor = torch.tensor(errors, dtype=torch.float64)
            if states:
                return tensor, torch.tensor(level, dtype=torch.float64), a, b, s
            return tensor, {}
        e_alpha, e_beta, e_gamma = [0.0] * n, [0.0] * n, [0.0] * n
        a_alpha = a_beta = a_gamma = 0.0               # d a / d parameter
        b_alpha = b_beta = b_gamma = 0.0               # d b / d parameter
        s_alpha, s_beta, s_gamma = [0.0] * m, [0.0] * m, [0.0] * m
        for t in range(n):
            x, si = xs[t], s[i]
            base = a + b
            errors[t] = x - base * si
            ratio = x / si
            grown = alpha * ratio + oma * base
            if not 0.0 < grown < math.inf:
                return bad, {name: bad for name in ("alpha", "beta", "gamma")}
            pull = alpha * ratio / si                  # -d grown / d s
            feed = gamma * x / (grown * grown)         # -d s_new / d grown
            # alpha
            d_si, d_base = s_alpha[i], a_alpha + b_alpha
            e_alpha[t] = -(d_base * si + base * d_si)
            d_grown = oma * d_base - pull * d_si + ratio - base
            b_alpha = beta * (d_grown - a_alpha) + omb * b_alpha
            a_alpha = d_grown
            s_alpha[i] = omg * d_si - feed * d_grown
            # beta
            d_si, d_base = s_beta[i], a_beta + b_beta
            e_beta[t] = -(d_base * si + base * d_si)
            d_grown = oma * d_base - pull * d_si
            b_beta = beta * (d_grown - a_beta) + omb * b_beta + grown - base
            a_beta = d_grown
            s_beta[i] = omg * d_si - feed * d_grown
            # gamma
            d_si, d_base = s_gamma[i], a_gamma + b_gamma
            e_gamma[t] = -(d_base * si + base * d_si)
            d_grown = oma * d_base - pull * d_si
            b_gamma = beta * (d_grown - a_gamma) + omb * b_gamma
            a_gamma = d_grown
            s_gamma[i] = omg * d_si - feed * d_grown + x / grown - si
            # states
            b = beta * (grown - a) + omb * b
            a = grown
            si = gamma * x / a + omg * si
            if not si > 0.0:
                return bad, {name: bad for name in ("alpha", "beta", "gamma")}
            s[i] = si
            i += 1
            if i == m:
                i = 0
        return torch.tensor(errors, dtype=torch.float64), {
            "alpha": torch.tensor(e_alpha, dtype=torch.float64),
            "beta": torch.tensor(e_beta, dtype=torch.float64),
            "gamma": torch.tensor(e_gamma, dtype=torch.float64)}

    # ---- criterion ------------------------------------------------------------------------------

    def sse(self, params: dict[str, float]) -> float:
        errors, _ = self._errors(params, False)
        value = float(torch.dot(errors, errors))
        return value if math.isfinite(value) else math.inf

    def objective(self, free: list[str], fixed: dict[str, float]):
        """Criterion in theta with parameter = sin(theta)^2, for ``optimize.maximize_bfgs``.

        The map keeps every parameter in [0, 1] and reaches the bounds at finite theta,
        where the criterion is stationary: an optimum on a bound is an ordinary maximum
        in theta. Returns ``function(theta) -> (f, gradient)`` with f = -n/2 ln(SSE/n)
        and its exact gradient, and ``curvature(theta)``, the Gauss-Newton Hessian
        -(n/SSE) J'J of the errors (J = d e / d parameter) carried through the map plus
        the exact curvature of the map, diag(df/dp * 2 cos(2 theta)).
        """
        n = self.n
        cache: dict[str, Any] = {}

        def pieces(theta: Tensor) -> tuple[float, Tensor, Tensor] | None:
            if cache and bool((cache["theta"] == theta).all()):
                return cache["pieces"]
            values = torch.sin(theta).square()
            params = {**fixed, **dict(zip(free, values.tolist(), strict=True))}
            errors, slopes = self._errors(params, True, free)
            total = float(torch.dot(errors, errors))
            result = None
            if math.isfinite(total) and total > 0.0:
                jacobian = torch.stack([slopes[name] for name in free])
                slope, bend = torch.sin(2.0 * theta), 2.0 * torch.cos(2.0 * theta)
                by_parameter = -(n / total) * (jacobian @ errors)
                hessian = -(n / total) * (jacobian @ jacobian.T) * slope[:, None] * slope[None, :]
                hessian = hessian + torch.diag(by_parameter * bend)
                result = (-0.5 * n * math.log(total / n), by_parameter * slope, hessian)
            cache.update(theta=theta.clone(), pieces=result)
            return result

        def function(theta: Tensor) -> tuple[float, Tensor]:
            result = pieces(theta)
            if result is None:
                return -math.inf, torch.full_like(theta, math.nan)
            return result[0], result[1]

        def curvature(theta: Tensor) -> Tensor:
            result = pieces(theta)
            if result is None:
                return torch.full((len(free), len(free)), math.nan, dtype=torch.float64)
            return result[2]

        return function, curvature

    # ---- fitted states and forecasts --------------------------------------------------------

    def fitted(self, params: dict[str, float], steps: int) -> tuple[Tensor, Tensor, Tensor]:
        """(one-step forecasts f_t, smoothed values, out-of-sample forecasts)."""
        n, m = self.n, self.m
        horizon = torch.arange(1, steps + 1, dtype=torch.float64)
        if self.method == "shwinters" and not self.additive:
            errors, smoothed, a, b, s = self._multiplicative(params, False, states=True)
            ahead = torch.tensor([(a + b * h) * s[(n + h - 1) % m] for h in range(1, steps + 1)],
                                 dtype=torch.float64)
            return self.x - errors, smoothed, ahead
        errors, _ = self._errors(params, False)
        k1, k2, k3 = self._gains(params)[0]
        one_step = self.x - errors
        level = one_step + k1 * errors
        if self.method == "exponential":
            return one_step, level, level[-1].expand(steps).clone()
        trend = self.trend0 + k2 * float(errors.sum())
        ahead = level[-1] + trend * horizon
        if self.method != "shwinters":
            return one_step, level, ahead
        # s_t = s_(t-m) + k3 e_t: a cumulative sum within each season.
        padded = torch.zeros(-(-n // m) * m, dtype=torch.float64)
        padded[:n] = k3 * errors
        seasonal = (self.seasonal0[None, :] + padded.reshape(-1, m).cumsum(dim=0)).reshape(-1)[:n]
        smoothed = level + k3 * errors                  # a_t + s_t = f_t + (k1 + k3) e_t
        ahead = (smoothed[-1] - seasonal[-1]) + trend * horizon \
            + seasonal[n - m:][torch.arange(steps) % m]
        return one_step, smoothed, ahead


def _check_parameter(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise AnalysisError("invalid_parameter", f"{name} must be a number between 0 and 1.")
    return float(value)


def _descend(model: _Smoother, free: list[str], fixed: dict[str, float],
             start: dict[str, float], record: dict[str, Any]) -> dict[str, float]:
    """Minimize the SSE over the free parameters from one starting point.

    BFGS in the sin^2 parameterization; a parameter that ends on 0 or 1 is fixed there
    exactly and the others are re-optimized. ``record`` collects the iteration count,
    the parameters that ran to a bound and those that dropped out of the model.
    """
    fixed, start, free = dict(fixed), dict(start), list(free)
    floor = 1e-28 * float(torch.dot(model.x, model.x))
    while free:
        # alpha = 0 freezes level and trend (beta drops out); alpha = 1 leaves the seasonal
        # recursion unchanged (gamma drops out). Such a parameter is set to 0 and reported.
        idle = [name for name, at in (("beta", 0.0), ("gamma", 1.0))
                if name in free and fixed.get("alpha") == at]
        for name in idle:
            fixed[name] = 0.0
            record["not_identified"].append(name)
        free = [name for name in free if name not in fixed]
        if not free:
            break
        smallest = model.sse({**fixed, **{name: start[name] for name in free}})
        if not math.isfinite(smallest):
            raise AnalysisError("numerical_failure", "The smoothing recursion is not finite at "
                                "the trial parameters; check the series for extreme values.")
        if smallest <= floor:
            raise AnalysisError("perfect_fit", "The initial level, trend and seasonal terms "
                                "already reproduce the series exactly, so the smoothing "
                                "parameters are not identified; supply them explicitly.")
        angles = torch.tensor([math.asin(math.sqrt(min(max(start[name], _INTERIOR),
                                                         1.0 - _INTERIOR))) for name in free],
                              dtype=torch.float64)
        function, curvature = model.objective(free, fixed)
        try:
            result = optimize.maximize_bfgs(function, angles, hessian_fn=curvature,
                                            raise_on_failure=False, max_iter=500)
        except KernelError as exc:
            raise AnalysisError(
                exc.code, f"The smoothing parameters could not be optimized: {exc}") from exc
        record["iterations"] += result.iterations
        values = torch.sin(result.theta).square().tolist()
        # A parameter that has reached 0 or 1 is fixed there exactly.
        margin = _CONVERGED_BOUND if result.converged else _BOUND
        pinned = [(name, 0.0 if value < 0.5 else 1.0)
                  for name, value in zip(free, values, strict=True)
                  if value < margin or value > 1.0 - margin]
        if result.converged and not pinned:
            return {**fixed, **dict(zip(free, values, strict=True))}
        if not pinned:
            raise AnalysisError("nonconvergence", "The smoothing parameters did not converge: "
                                f"{result.diagnostics.get('message')} Supply some of them "
                                "explicitly.")
        if any(name == "alpha" for name, _ in pinned):
            # alpha on a bound removes another parameter from the model: settle it first.
            pinned = [(name, bound) for name, bound in pinned if name == "alpha"]
        for name, bound in pinned:
            fixed[name] = bound
            record["at_bounds"].append(name)
        start = {name: value for name, value in zip(free, values, strict=True)
                 if name not in fixed}
        free = [name for name in free if name not in fixed]
    return fixed


def _optimize(model: _Smoother, free: list[str], fixed: dict[str, float]) -> tuple[dict, dict]:
    """Minimize the SSE over the free parameters; returns the parameters and a record.

    The criterion can have several local minima, often with one of them in a corner
    of the unit cube (alpha = 1, beta = 0 is the random walk with drift). The search
    therefore evaluates a grid that includes the bounds, descends from its best
    _STARTS points and keeps the lowest minimum. A long series is represented by its
    first _GRID_ROWS observations in this search and then optimized once on the full
    sample from the point found.
    """
    record: dict[str, Any] = {"iterations": 0, "at_bounds": [], "not_identified": []}
    if not free:
        return dict(fixed), record
    coarse = model
    if model.n > _GRID_ROWS:
        try:
            coarse = _Smoother(model.x[:_GRID_ROWS], model.method, model.m, model.additive)
        except AnalysisError:
            coarse = model          # the short sample has no valid initial values: use it all
    grid: list[list[float]] = [[]]
    for _ in free:
        grid = [point + [value] for point in grid for value in _GRID]
    ranked = sorted((coarse.sse({**fixed, **dict(zip(free, point, strict=True))}), point)
                    for point in grid)
    if not math.isfinite(ranked[0][0]):
        raise AnalysisError("numerical_failure", "The smoothing recursion is not finite for "
                            "any trial parameters; check the series for extreme values.")
    best: tuple[float, dict[str, float], dict[str, Any]] | None = None
    failure: AnalysisError | None = None
    for value, point in ranked[:_STARTS]:
        if not math.isfinite(value):
            break
        trial: dict[str, Any] = {"iterations": 0, "at_bounds": [], "not_identified": []}
        try:
            params = _descend(coarse, free, fixed, dict(zip(free, point, strict=True)), trial)
        except AnalysisError as exc:
            if exc.code != "nonconvergence":
                raise
            failure = failure or exc
            continue
        record["iterations"] += trial["iterations"]
        reached = coarse.sse(params)
        if best is None or reached < best[0] * (1.0 - 1e-12):
            best = (reached, params, trial)
    if best is None:
        raise failure
    _, params, trial = best
    if coarse is not model:
        trial = {"iterations": 0, "at_bounds": [], "not_identified": []}
        params = _descend(model, free, fixed, {name: params[name] for name in free}, trial)
        record["iterations"] += trial["iterations"]
    record["at_bounds"], record["not_identified"] = trial["at_bounds"], trial["not_identified"]
    return params, record


def _future_periods(periods: pd.Series, steps: int, timed: bool) -> list[Any]:
    n = len(periods)
    if not timed:
        return list(range(n, n + steps))
    if is_datetime64_any_dtype(periods.dtype):
        frequency = pd.infer_freq(periods) if n >= 3 else None
        if frequency is None:
            return [pd.NaT] * steps
        return list(pd.date_range(periods.iloc[-1], periods=steps + 1, freq=frequency)[1:])
    last = periods.iloc[-1]
    return [last + h for h in range(1, steps + 1)]


def tssmooth(*, data: Any, y: str, method: str = "hwinters", alpha: float | None = None,
             beta: float | None = None, gamma: float | None = None, period: int | None = None,
             additive: bool = True, time: str | None = None, forecast: int = 0) -> pd.DataFrame:
    """Exponential smoothing and Holt-Winters forecasts (Stata's ``tssmooth``).

    Methods
    -------
    ``"exponential"``  simple exponential smoothing, ``S_t = alpha x_t + (1-alpha) S_(t-1)``.
    ``"dexponential"`` Brown's double exponential smoothing (one parameter, linear trend).
    ``"hwinters"``     Holt's linear-trend smoothing, ``a_t = alpha x_t + (1-alpha)(a_(t-1) +
                       b_(t-1))``, ``b_t = beta (a_t - a_(t-1)) + (1-beta) b_(t-1)``.
    ``"shwinters"``    seasonal Holt-Winters with ``period`` seasons; ``additive=True`` adds the
                       seasonal component (``x^ = a + b + s``), ``additive=False`` multiplies
                       (``x^ = (a + b) s``, strictly positive series only).

    Parameters
    ----------
    data, y : the table and the series to smooth.
    alpha, beta, gamma : smoothing parameters in [0, 1] for the level, trend and
        seasonal recursions. Any that the method uses and that is left ``None`` is
        estimated by minimizing the in-sample sum of squared one-step forecast
        errors; a parameter that the method does not use must stay ``None``. The
        criterion can have several local minima, so the search descends from the
        three best points of a grid over {0, 0.1, 0.5, 0.9, 1} per parameter and
        keeps the lowest minimum (a series longer than 5,000 observations is
        searched on its first 5,000 and then optimized once on the full sample).
    period : seasons per cycle (``shwinters`` only), e.g. 12 for monthly data.
    additive : additive (default) or multiplicative seasonality. Stata's
        ``tssmooth shwinters`` is multiplicative unless its ``additive`` option is given.
    time : optional time column (sorted; integer periods must be consecutive).
    forecast : number of out-of-sample periods to forecast.

    Initial values
    --------------
    ``exponential``: the mean of the first half of the sample. ``dexponential`` and
    ``hwinters``: intercept and slope of an OLS line through the first half.
    ``shwinters``: from the complete seasons in the first half of the sample (at
    least two): the trend is the change in season means per period, the level the
    first season's mean moved back to time 0, the seasonal terms the average
    detrended deviations (ratios) by season, normalized to sum to zero (average one).

    Returns
    -------
    Table with ``n + forecast`` rows: ``period``, ``observed``, ``smoothed``
    (the smoothed value after x_t is seen: S_t, the level a_t, or level plus/times
    the seasonal term) and ``forecast`` (in sample the one-step-ahead prediction of
    x_t made at t-1, which is the series Stata's tssmooth generates; out of sample
    the h-step forecasts; ``observed`` and ``smoothed`` are missing there).
    ``attrs`` holds ``parameters``, ``estimated``, ``sse``, ``rmse``
    (``sqrt(SSE/n)``), ``nobs``, ``initial`` values, ``at_bounds`` (estimated
    parameters that ran to 0 or 1) and ``not_identified`` (``beta`` when alpha = 0,
    ``gamma`` when alpha = 1: they do not affect the fit and are reported as 0).
    No standard errors are reported for the smoothing parameters, which is why
    this is a table function and not a registered estimator.

    Stata: ``tssmooth exponential``, ``tssmooth dexponential``, ``tssmooth hwinters``,
    ``tssmooth shwinters``. SPSS: Exponential Smoothing (Simple, Brown, Holt,
    Winters). EViews: Proc / Exponential Smoothing.

    Example
    -------
    >>> out = oe.tssmooth(data=df, y="sales", method="shwinters", period=12, forecast=12)
    >>> out.attrs["parameters"], out.attrs["rmse"]
    """
    if method not in METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(METHODS)}.")
    if not isinstance(additive, bool):
        raise AnalysisError("invalid_option", "additive must be True or False.")
    if not isinstance(forecast, int) or isinstance(forecast, bool) or not 0 <= forecast <= 10_000:
        raise AnalysisError("invalid_steps", "forecast must be an integer between 0 and 10000.")
    given = {"alpha": _check_parameter("alpha", alpha), "beta": _check_parameter("beta", beta),
             "gamma": _check_parameter("gamma", gamma)}
    unused = [name for name, value in given.items()
              if value is not None and name not in METHODS[method]]
    if unused:
        raise AnalysisError("invalid_parameter", f"method='{method}' does not use "
                            f"{', '.join(unused)}.")
    x, periods = load_series(data, y, time)
    seasonal = method == "shwinters"
    if seasonal:
        if not isinstance(period, int) or isinstance(period, bool) or period < 2:
            raise AnalysisError("invalid_option", "shwinters needs period, the number of seasons "
                                "per cycle (an integer of at least 2).")
        if x.shape[0] < 2 * period:
            raise AnalysisError("insufficient_observations", "Seasonal smoothing needs at least "
                                f"two complete seasons ({2 * period} observations).")
    elif period is not None:
        raise AnalysisError("invalid_option", "period is only used by method='shwinters'.")
    _require_variation(x, 4, "Exponential smoothing")
    with torch.no_grad():
        model = _Smoother(x, method, period or 1, additive)
        fixed = {name: given[name] for name in METHODS[method] if given[name] is not None}
        free = [name for name in METHODS[method] if given[name] is None]
        params, record = _optimize(model, free, fixed)
        sse = model.sse(params)
        if not math.isfinite(sse):
            raise AnalysisError("numerical_failure", "The smoothing recursion is not finite at "
                                "these parameters.")
        one_step, smoothed, ahead = model.fitted(params, forecast)
    n = x.shape[0]
    missing = [math.nan] * forecast
    return table(
        {"period": [*periods.tolist(), *_future_periods(periods, forecast, time is not None)],
         "observed": [*x.tolist(), *missing],
         "smoothed": [*smoothed.tolist(), *missing],
         "forecast": [*one_step.tolist(), *ahead.tolist()]},
        title=f"Exponential smoothing of {y}: {method}",
        method=method, series=y,
        parameters={name: params[name] for name in METHODS[method]},
        estimated=free, at_bounds=record["at_bounds"],
        not_identified=record["not_identified"], iterations=record["iterations"],
        seasonal_period=period if seasonal else None,
        additive=additive if seasonal else None,
        sse=sse, rmse=math.sqrt(sse / n), nobs=n, forecast_steps=forecast,
        initial={"level": model.level0, "trend": model.trend0,
                 "seasonal": None if model.seasonal0 is None else model.seasonal0.tolist()},
        criterion="in-sample sum of squared one-step forecast errors")
