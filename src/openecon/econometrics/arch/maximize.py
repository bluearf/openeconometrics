"""Starting values and BFGS maximization of the ARCH-family likelihood.

The optimizer works on a rescaled, unrestricted vector t with

    theta = a0 + A t,        A = R diag(d),

where d holds the natural size of each parameter (a regression coefficient is of
order sd(residual) / rms(regressor), a GARCH variance constant of order
var(residual), lag coefficients of order one) and R imposes the IGARCH restriction
(the last GARCH coefficient is one minus the other ARCH and GARCH coefficients;
R is the identity for every other model). This is an exact linear
reparameterization: gradients, scores and the Hessian are mapped with A, so the
reported estimates and covariance are those of the raw parameters, as in Stata, while
BFGS and the numerical Hessian see a problem whose coordinates are all of order one
whatever the units of the data. (The caller, ``estimators.fit_arch``, additionally
centres the regressors of both equations at their means, another exact linear map.)

Hessians (the convergence test of BFGS and the observed information) are numerical
derivatives of the analytic gradient with a small first step (``kinks.HESSIAN_STEP``),
because the gradients of the GJR, power-ARCH and GED likelihoods have kinks at zero
residuals.

Starting values: OLS for the mean equation, a Hannan-Rissanen regression for the ARMA
terms, and a short list of variance-equation candidates scaled to the residual
variance. The candidates are ranked by their log likelihood and BFGS
(``engines.optimize.maximize_bfgs``, analytic gradient, inverse outer product of the
scores at the start as the initial curvature) runs from the best one; the next ones are
tried only if it fails to converge.

EGARCH. The likelihood has a kink wherever a standardized residual is zero, so the
Hessian used for the convergence test and the covariance is that of the smooth piece
through the point (``kinks.piece_hessian``). BFGS is asked only to get close (it stops on
its gradient rule or when its steps fall below 1e-8, without Newton polishing, which
cannot succeed on a kink); unless the scaled gradient g'(-H)^-1 g of the piece is
already below 1e-10 there, the active-set Newton method of ``kinks.polish`` finishes the
job, whether the maximum is a smooth one or lies on a kink.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch import densities, kinks
from openecon.econometrics.arch.kernels import ArchLikelihood
from openecon.econometrics.arch.layout import Layout
from openecon.econometrics.arch.recursions import dense
from openecon.econometrics.arima.filters import inverse_filter, spectral_radius
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import least_squares

_MAX_STARTS = 3            # candidates from which BFGS is run before giving up
_STATIONARY = 0.95         # largest inverse root allowed in ARMA starting values
_T_START = math.log(6.0)   # Student t: 8 degrees of freedom
_GED_START = math.log(1.5)
_SCALED_GRADIENT = 1e-10   # convergence: g'(-H)^-1 g, as in engines.optimize
_VARIANCE_RANGE = (1e-120, 1e120)     # residual variances whose squares stay well inside float64
# EGARCH: BFGS only approaches the maximum; kinks.polish decides convergence.
_KINKED_OPTIONS = {"step_tol": 1e-8, "scaled_gradient_tol": 1e300}


@dataclass
class Fit:
    theta: Tensor            # [K] reported parameters
    transform: Tensor        # A: [K, K_free], theta = a0 + A t
    hessian: Tensor          # Hessian of the log likelihood in t
    record: dict[str, Any]
    signs: Tensor | None = None      # EGARCH maximum on a kink: the sign pattern of its piece
    flat: Tensor | None = None       # ... with GED errors: the mask of the kink observations


def restriction(layout: Layout) -> tuple[Tensor, Tensor]:
    """``(R, a0)`` with theta = a0 + R theta_free; the identity unless the model is IGARCH."""
    ix = layout.index
    k = ix.k
    offset = torch.zeros(k, dtype=torch.float64)
    if not layout.restricted:
        return torch.eye(k, dtype=torch.float64), offset
    last = ix.b[-1]
    free = [i for i in range(k) if i != last]
    matrix = torch.zeros((k, k - 1), dtype=torch.float64)
    for column, row in enumerate(free):
        matrix[row, column] = 1.0
        if row in ix.a or row in ix.b:
            matrix[last, column] = -1.0
    offset[last] = 1.0
    return matrix, offset


def _arma_start(resid: Tensor, layout: Layout) -> tuple[list[float], list[float]]:
    """Hannan-Rissanen starting values of the AR and MA coefficients at the given lags."""
    ar_lags, ma_lags = list(layout.ar_lags), list(layout.ma_lags)
    n = resid.shape[0]
    rho, theta = [0.0] * len(ar_lags), [0.0] * len(ma_lags)
    longest = max([0, *ar_lags, *ma_lags])
    if not longest:
        return rho, theta
    try:
        innovations, first = None, max([0, *ar_lags])
        if ma_lags:
            order = min(max(longest + 4, round(math.log(n) ** 2)), n // 4)
            if order < 1:
                return rho, theta
            columns = [resid[order - j:n - j] for j in range(1, order + 1)]
            long_ar = least_squares(torch.stack(columns, dim=1), resid[order:].contiguous())
            innovations = torch.zeros(n, dtype=torch.float64)
            innovations[order:] = long_ar.resid
            first = order + max(ma_lags)
        if n - first < 2 * (len(ar_lags) + len(ma_lags)) + 2:
            return rho, theta
        columns = [resid[first - lag:n - lag] for lag in ar_lags]
        columns += [innovations[first - lag:n - lag] for lag in ma_lags]
        fit = least_squares(torch.stack(columns, dim=1), resid[first:].contiguous())
        beta = torch.zeros(len(columns), dtype=torch.float64)
        beta[fit.kept] = fit.beta
        rho, theta = beta[:len(ar_lags)].tolist(), beta[len(ar_lags):].tolist()
    except KernelError:
        return [0.0] * len(ar_lags), [0.0] * len(ma_lags)
    # Pull the polynomials inside the stationary and invertible region.
    for _ in range(60):
        if not rho or spectral_radius(dense(rho, ar_lags)) < _STATIONARY:
            break
        rho = [0.8 * value for value in rho]
    for _ in range(60):
        if not theta or spectral_radius(dense(theta, ma_lags, -1.0)) < _STATIONARY:
            break
        theta = [0.8 * value for value in theta]
    return rho, theta


def _variance_candidates(layout: Layout, s2: float) -> list[dict[str, Any]]:
    """Variance-equation starting values: totals spread evenly over the lags."""
    n_a, n_b, kind = len(layout.arch_lags), len(layout.garch_lags), layout.kind

    def spread(total: float, count: int) -> list[float]:
        return [total / count] * count if count else []

    candidates = []
    if layout.restricted:
        for alpha in (0.05, 0.15, 0.30):
            candidates.append({"a": spread(alpha, n_a), "b": spread(1.0 - alpha, n_b),
                               "omega": 0.02 * s2})
    elif kind == "garch":
        pairs = [(0.05, 0.90), (0.10, 0.80), (0.20, 0.60), (0.10, 0.40)] if n_b \
            else [(0.20, 0.0), (0.40, 0.0), (0.05, 0.0)]
        for alpha, beta in pairs:
            candidates.append({"a": spread(alpha, n_a), "b": spread(beta, n_b),
                               "omega": s2 * (1.0 - alpha - beta)})
    elif kind == "gjr":
        triples = [(0.03, 0.06, 0.88), (0.05, 0.10, 0.75), (0.10, 0.10, 0.50)] if n_b \
            else [(0.15, 0.10, 0.0), (0.30, 0.10, 0.0), (0.05, 0.05, 0.0)]
        for alpha, gamma, beta in triples:
            candidates.append({"a": spread(alpha, n_a), "g": spread(gamma, n_a),
                               "b": spread(beta, n_b),
                               "omega": s2 * (1.0 - alpha - 0.5 * gamma - beta)})
    elif kind == "egarch":
        pairs = [(0.10, 0.95), (0.20, 0.80), (0.30, 0.50)] if n_b \
            else [(0.20, 0.0), (0.40, 0.0)]
        for gamma, beta in pairs:
            candidates.append({"a": [0.0] * n_a, "g": spread(gamma, n_a), "b": spread(beta, n_b),
                               "omega": (1.0 - beta) * math.log(s2)})
    else:
        pairs = [(0.05, 0.90), (0.10, 0.70)] if n_b else [(0.20, 0.0), (0.40, 0.0)]
        for power in (2.0, 1.2):
            moment = densities.abs_moment("normal", power)
            for alpha, beta in pairs:
                candidates.append({
                    "a": spread(alpha / moment, n_a), "b": spread(beta, n_b), "power": power,
                    "omega": s2 ** (0.5 * power) * (1.0 - alpha - beta)})
    return candidates


def starting_values(like: ArchLikelihood) -> tuple[list[Tensor], Tensor]:
    """Candidate starting vectors (full layout) and the scale d of each parameter."""
    replay = getattr(like, "replay_starting_values", None)
    if replay is not None:
        return replay()
    layout, ix, n = like.layout, like.layout.index, like.n
    y, x = like.y, like.x
    if layout.k_x:
        try:
            ols = least_squares(x, y)
        except KernelError as exc:
            raise AnalysisError(exc.code, str(exc)) from exc
        beta = torch.zeros(layout.k_x, dtype=torch.float64)
        beta[ols.kept] = ols.beta
        resid = ols.resid
    else:
        beta, resid = torch.zeros(0, dtype=torch.float64), y
    total = float(y.square().mean())
    s2 = float(resid.square().mean())
    if not s2 > 1e-20 * max(total, 1e-300):
        raise AnalysisError("perfect_fit", "The regressors reproduce the outcome exactly; there "
                            "is no disturbance whose variance could be modelled. Remove "
                            "regressors that determine the outcome.")
    if not _VARIANCE_RANGE[0] < s2 < _VARIANCE_RANGE[1]:
        # The variance constant is of order s2 and its sampling variance of order s2^2.
        raise AnalysisError(
            "numerical_failure", f"The residual variance of the outcome ({s2:.3g}) is outside "
            "the range in which a variance equation can be estimated in float64 "
            f"({_VARIANCE_RANGE[0]:g} to {_VARIANCE_RANGE[1]:g}): rescale the outcome, for "
            "example by expressing it in other units.")
    rho, theta = _arma_start(resid, layout)
    if rho or theta:
        w = resid.clone()
        for value, lag in zip(rho, layout.ar_lags, strict=True):
            w[lag:] -= value * resid[:n - lag]
        ma = dense(theta, layout.ma_lags)
        filtered = inverse_filter(w, ma) if ma else w
        value = float(filtered.square().mean())
        if math.isfinite(value) and 0.0 < value <= s2:
            s2 = value
        else:
            rho, theta = [0.0] * len(rho), [0.0] * len(theta)
    sd = math.sqrt(s2)
    exponential = layout.kind != "egarch"          # exp(l0 + z'l) versus l0 + z'l

    scale = torch.ones(ix.k, dtype=torch.float64)
    if layout.k_x:
        rms = x.square().mean(dim=0).sqrt()
        scale[ix.x] = sd / torch.where(rms > 0, rms, torch.ones_like(rms))
    if ix.m:
        scale[ix.m[0]] = {"variance": 1.0 / sd, "sd": 1.0, "log": sd}[layout.archm]
    if layout.k_z:
        spread = like.z.std(dim=0, correction=0)
        scale[ix.z] = 1.0 / torch.where(spread > 0, spread, torch.ones_like(spread))
    elif exponential:
        scale[ix.c] = s2
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise AnalysisError("numerical_failure", "The outcome or a regressor is too large or too "
                            "small to be scaled in float64; rescale the data.")

    starts = []
    for candidate in _variance_candidates(layout, s2):
        start = torch.zeros(ix.k, dtype=torch.float64)
        start[ix.x] = beta
        start[ix.ar] = torch.tensor(rho, dtype=torch.float64)
        start[ix.ma] = torch.tensor(theta, dtype=torch.float64)
        start[ix.a] = torch.tensor(candidate["a"], dtype=torch.float64)
        if ix.g:
            start[ix.g] = torch.tensor(candidate["g"], dtype=torch.float64)
        start[ix.b] = torch.tensor(candidate["b"], dtype=torch.float64)
        omega = candidate["omega"]
        start[ix.c] = math.log(omega) if layout.k_z and exponential else omega
        if ix.p:
            start[ix.p[0]] = candidate["power"]
        if ix.d:
            start[ix.d[0]] = _T_START if layout.dist == "t" else _GED_START
        starts.append(start)
    return starts, scale


def maximize(like: ArchLikelihood, tolerance: float, max_iterations: int) -> Fit:
    """Maximize the likelihood; raises ``AnalysisError('nonconvergence')`` on failure."""
    layout = like.layout
    starts, scale = starting_values(like)
    matrix, offset = restriction(layout)
    free = [i for i in range(layout.k) if not (layout.restricted and i == layout.index.b[-1])]
    transform = matrix * scale[free][None, :]
    infeasible = torch.full((len(free),), math.nan, dtype=torch.float64)

    def objective(t: Tensor) -> tuple[float, Tensor]:
        out = like.evaluate(offset + transform @ t)
        if out is None:
            return -math.inf, infeasible
        return out.value, transform.T @ out.gradient

    def gradient(t: Tensor) -> Tensor:
        out = like.evaluate(offset + transform @ t)
        return infeasible if out is None else transform.T @ out.gradient

    def smooth_hessian(t: Tensor) -> Tensor:
        # Numerical derivative of the analytic gradient with a small first step: the
        # gradient has kinks at zero residuals for GJR, power ARCH and GED (see kinks).
        return optimize.numerical_hessian(gradient, t, relative_step=kinks.HESSIAN_STEP)

    kinked = layout.kind == "egarch"
    hessian_fn = (lambda t: kinks.piece_hessian(like, offset, transform, t)) if kinked \
        else smooth_hessian

    ranked = []
    for number, start in enumerate(starts):
        out = like.evaluate(start, derivatives=False)
        if out is not None:
            ranked.append((out.value, number, start))
    ranked.sort(key=lambda item: -item[0])
    if not ranked:
        raise AnalysisError("invalid_start", "The likelihood could not be evaluated at any "
                            "starting value; check the series for extreme values or rescale it.")
    tried: dict[str, float] = {f"candidate_{number}": value for value, number, _ in ranked}
    failure = "no starting value could be improved."
    stopped: list[tuple[float, Tensor]] = []          # (log likelihood, theta) of failed runs
    for value, number, start in ranked[:_MAX_STARTS]:
        t0 = start[free] / scale[free]
        inverse = None
        first = like.evaluate(start, scores=True)
        if first is not None:
            scores = first.scores @ transform
            try:
                inverse = optimize.information_inverse(scores.T @ scores)
            except KernelError:
                inverse = None
        try:
            result = optimize.maximize_bfgs(
                objective, t0, max_iter=max_iterations, gradient_tol=tolerance,
                hessian_fn=hessian_fn, initial_inverse_hessian=inverse, raise_on_failure=False,
                **(_KINKED_OPTIONS if kinked else {}))
        except KernelError as exc:
            failure = str(exc)
            continue
        converged, solution = result.converged, None
        if kinked:
            scaled = result.diagnostics.get("scaled_gradient")
            converged = converged and scaled is not None and scaled <= _SCALED_GRADIENT
            if not converged and result.iterations < max_iterations:
                solution = kinks.polish(like, offset, transform, result.theta, result.hessian)
        if converged or solution is not None:
            diagnostics = result.diagnostics
            record = {
                "method": result.method, "iterations": result.iterations, "converged": True,
                "gradient_max": diagnostics.get("gradient_max"),
                "scaled_gradient": diagnostics.get("scaled_gradient"),
                "function_evaluations": diagnostics.get("function_evaluations"),
                "polish_iterations": diagnostics.get("polish_iterations"),
                "message": diagnostics.get("message"),
                "starting_values": f"candidate_{number}", "starts": tried,
                "gradient": "analytic (recursive derivatives of the conditional variance" + (
                    "; reverse mode: anti-causal filters of the likelihood weights)"
                    if like.linear else "; forward mode: parallel prefix scan)"),
                "hessian": "numerical (Ridders-extrapolated central differences of the analytic "
                           f"gradient, relative first step {kinks.HESSIAN_STEP:g})" + (
                               " of the smooth piece of the likelihood through the estimates "
                               "(signs of the residuals held fixed)" if kinked else ""),
                "initial_curvature": "inverse outer product of the scores at the starting values"
                if inverse is not None else "identity",
                "parameterization": "raw parameters divided by their natural scale; the "
                                    "reported estimates and covariance are in the raw parameters",
                "engine": "filter" if like.linear else "scan",
            }
            if solution is None:
                return Fit(offset + transform @ result.theta, transform, result.hessian, record)
            record.update({
                "method": result.method + "+active_set_newton",
                "iterations": result.iterations + solution.iterations,
                "active_set_iterations": solution.iterations,
                "gradient_max": None, "scaled_gradient": None, "polish_iterations": None,
                "kink_observations": list(solution.active),
                "kink_weights": [float(solution.signs[position]) for position in solution.active],
                "message": "converged: active-set Newton steps on the kinks of the EGARCH "
                           "likelihood" + (
                               f"; the maximum lies on {len(solution.active)} kink(s), where a "
                               "standardized residual is exactly zero and zero is a convex "
                               "combination of the one-sided gradients" if solution.active
                               else "; the maximum is an ordinary smooth one"),
            })
            return Fit(offset + transform @ solution.t, transform, solution.hessian, record,
                       solution.signs if solution.active else None,
                       kinks.flat_mask(like, solution.active))
        failure = str(result.diagnostics.get("message"))
        if kinked and result.converged:
            failure = ("BFGS stopped near a point from which the active-set Newton steps on the "
                       "kinks of the EGARCH likelihood found no maximum.")
        stopped.append((result.value, offset + transform @ result.theta))
        if layout.dist == "t":
            tau = float(stopped[-1][1][layout.index.d[0]])
            if tau > densities.TAU_BOUNDS["t"][1] - 1.0:
                raise AnalysisError(
                    "nonconvergence", "The Student t degrees of freedom diverge (the likelihood "
                    "keeps rising beyond several hundred degrees of freedom): the standardized "
                    "innovations are indistinguishable from normal. Use dist='normal'.")
    reason = _diagnose(like, max(stopped, key=lambda item: item[0])[1]) if stopped else None
    if reason is not None:
        raise AnalysisError("nonconvergence", reason)
    raise AnalysisError(
        "nonconvergence",
        f"The ARCH likelihood maximization did not converge: {failure} Try a simpler variance "
        "equation (fewer ARCH/GARCH lags), check that the series shows conditional "
        "heteroskedasticity at all (oe.archlm), rescale the outcome (for example returns in "
        "percent), or raise max_iterations.")


def _diagnose(like: ArchLikelihood, theta: Tensor) -> str | None:
    """Explain a search that ended where the likelihood has no regular maximum, if it did.

    No positivity constraints are imposed (as in Stata), so pathologies of the raw
    likelihood can stop the optimizer; they are properties of the model on the data:

    - a negative ARCH or GARCH coefficient lets the variance of one observation approach
      zero, where the likelihood is unbounded (a spike, not a maximum);
    - the power-ARCH news |e|^phi has a cusp at e = 0 when phi <= 1, so the likelihood
      has a ridge of non-differentiable local maxima at every zero residual;
    - an EGARCH maximum on a kink (a zero standardized residual) with GED errors of
      shape below 2 lies where the curvature of the GED density is unbounded.
    """
    replay = getattr(like, "replay_diagnose", None)
    if replay is not None:
        return replay(theta)
    layout, ix = like.layout, like.layout.index
    out = like.evaluate(theta, derivatives=False)
    if out is None:
        return None
    variance = out.variance
    if layout.kind == "egarch" and layout.dist == "ged":
        shape = math.exp(float(theta[ix.d[0]]))
        smallest = float((out.residual / variance.sqrt()).abs().min())
        if shape < 2.0 and smallest < 1e-8:
            return (
                "The EGARCH likelihood with GED errors has no regular maximum on these data: "
                "the search ended on a kink of the likelihood (a standardized residual that is "
                f"exactly zero), and with a GED shape below 2 (here {shape:.3f}) the curvature "
                "of the density is unbounded there, so neither Newton steps nor Hessian-based "
                "standard errors exist at that point. Use dist='t' or dist='normal' with "
                "model='egarch', or keep dist='ged' with model='gjr'.")
    if layout.kind == "parch":
        power = float(theta[ix.p[0]])
        if power <= 1.0:
            return (
                f"The power-ARCH likelihood has no regular maximum on these data: the search "
                f"moved to a power of {power:.3f}, and for a power of 1 or less the news term "
                "|e|^power is not differentiable at a zero residual, so the likelihood has a "
                "cusp at every observation and the optimizer stops on one of them. The data do "
                "not identify the power; use model='garch' (power 2) or a longer sample.")
    if layout.kind != "egarch":
        smallest = int(variance.argmin())
        coefficients = torch.cat([theta[ix.a], theta[ix.g], theta[ix.b]])
        negative = bool((coefficients < 0.0).any()) or (not layout.k_z and float(theta[ix.c]) < 0)
        if float(variance[smallest]) < 1e-6 * float(variance.median()):
            return (
                "The likelihood is unbounded on these data: with the negative ARCH/GARCH "
                "coefficient the search reached, the conditional variance of observation "
                f"{smallest + 1} (in time order) approaches zero, where the likelihood has a spike "
                "rather than a maximum. A lag of the variance equation is not supported by the "
                "data: remove it (fewer ARCH/GARCH lags), or use model='egarch', whose variance "
                "is positive by construction.")
        if negative:
            return (
                "The ARCH likelihood maximization did not converge: the search left the region "
                "of non-negative variance coefficients, where the likelihood is irregular "
                "(no positivity constraints are imposed). The data probably do not support "
                "this variance equation: use fewer ARCH/GARCH lags, check that the series "
                "shows conditional heteroskedasticity at all (oe.archlm), or use "
                "model='egarch'.")
    return None
