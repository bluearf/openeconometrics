"""Unobserved-components models (Stata's ``ucm``; EViews ``sspace`` with UC specifications).

    y_t = mu_t + gamma_t + psi_t + x_t'b + e_t,          e_t ~ N(0, var(e))
    mu_(t+1)   = mu_t + beta_t + eta_t                   (level, var(level))
    beta_(t+1) = beta_t + zeta_t                         (slope, var(slope))
    gamma_(t+1) = -(gamma_t + ... + gamma_(t-s+2)) + omega_t      (dummy seasonal, var(seasonal))
    [psi; psi*]_(t+1) = rho [cos l, sin l; -sin l, cos l] [psi; psi*]_t + kappa_t   (cycle)

``model`` selects the trend as in Stata's ``ucm, model()`` (see ``statespace.MODELS``):
which of level and slope exist, whether their disturbances have a free variance or
zero variance (a deterministic component), and whether there is an irregular term.

Estimation: exact Gaussian maximum likelihood with the exact diffuse Kalman filter
(``statespace.kalman``). Variances are optimized as logs (scaled by the variance of
the differenced series), frequency and damping through logits; the regression
coefficients b are concentrated out by GLS at every evaluation (profile likelihood)
and re-enter as parameters in the covariance. The optimizer is BFGS
(``engines.optimize.maximize_bfgs``) with gradients and Hessian from central
differences of the exact log likelihood, all perturbed points evaluated in one
batched filter pass (no analytic Kalman-derivative recursions; recorded in
provenance). A variance that runs to its zero boundary (below 1e-9 of the scale,
or below 1e-6 when the search stopped without certifying a maximum because the
log-variance direction is flat there) is fixed at zero, reported in ``extra`` and
the model re-maximized, as Stata reports a constrained variance; the restricted
maximum must not be lower than the point where the search stopped.

Covariance: ``nonrobust`` (default, Stata's vce(oim)): inverse of the numerical
Hessian of the full log likelihood in (log variances, logits, b), mapped to the
reported parameters by the delta method; ``robust``/``opg`` use per-period scores
(numerical for the variance parameters, analytic for b).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import jarque_bera_test, ljung_box_test
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, kernel_call, ml_covariance, table,
)
from openecon.econometrics.tsmodels import statespace as ss
from openecon.econometrics.tsmodels.common import as_int, build_spec, ordered_frame
from openecon.econometrics.tsmodels.filters import hp_trend
from openecon.engines.distributions import normal_isf
from openecon.engines.optimize import OptimResult, maximize_bfgs
from openecon.models import ModelSpec, ResultBundle

_ZERO = 1e-9                 # variance / scale below which a variance is fixed at zero
_LOOSE_ZERO = 1e-6           # the same when the search stopped without certifying a maximum
_EPS = torch.finfo(torch.float64).eps


class _Problem:
    """Profile and full log likelihood of one UC specification, batched over parameter sets.

    Deterministic states (``statespace.deterministic_states``) are moved into the GLS
    part as diffuse regressors W; their cross-products are Schur-complemented out and
    ``log det S_WW`` is added (de Jong's diffuse likelihood, equal to the exact diffuse
    one), so the reduced filter reaches its steady state. ``reduce=False`` runs the full
    exact diffuse filter instead (per-period contributions for score covariances).
    """

    def __init__(self, structure: ss.Structure, y: Tensor, x: Tensor, scale: float):
        self.structure, self.y, self.x, self.scale = structure, y, x, scale
        self.n, self.kx = y.shape[0], x.shape[1]
        self.fixed = ss.deterministic_states(structure)
        self.kept = [i for i in range(structure.m) if i not in set(self.fixed)]
        w = ss.deterministic_regressors(structure, self.fixed, self.n)
        self.data = torch.cat([y[None, :], x.T], dim=0).contiguous()
        self.reduced_data = torch.cat([self.data, w.T], dim=0).contiguous()
        self.layout = ss.Layout(structure.z[self.kept], structure.diffuse[self.kept])

    def _run(self, theta: Tensor, reduce: bool):
        values = ss.natural(self.structure, theta, self.scale)
        t, q, h, p0 = ss.system(self.structure, values, theta.shape[0])
        if not reduce or not self.fixed:
            return kernel_call(ss.kalman, self.structure, t, q, h, p0, self.data), 0
        if not self.kept:                      # every state is deterministic: F_t = h
            batch = theta.shape[0]
            if not bool((h > 0).all()):
                raise AnalysisError("degenerate_model", "The model has no stochastic component.")
            run = ss.FilterResult(self.reduced_data[None].expand(batch, -1, -1).clone(),
                                  h[:, None].expand(batch, self.n).clone(),
                                  torch.zeros(self.n, dtype=torch.bool),
                                  torch.zeros(self.n, dtype=torch.float64), self.n)
            return run, len(self.fixed)
        index = torch.tensor(self.kept)
        run = kernel_call(ss.kalman, self.layout, t[:, index][:, :, index], q[:, index], h,
                          p0[:, index][:, :, index], self.reduced_data)
        return run, len(self.fixed)

    def pieces(self, theta: Tensor, *, reduce: bool = True) -> dict[str, Any]:
        """GLS cross-products of the innovations for each row of theta [B, p]."""
        run, d = self._run(theta, reduce)
        used = ~run.diffuse_inf
        weight = torch.where(used, 1.0 / run.f, torch.zeros_like(run.f))            # [B, n]
        v = run.v
        cross = torch.einsum("bcn,bdn->bcd", v * weight[:, None, :], v)              # [B, C, C]
        logdet = torch.where(used, torch.log(run.f), torch.zeros_like(run.f)).sum(1) \
            + run.log_f_inf.sum()
        k = 1 + self.kx
        if d:
            sww = cross[:, k:, k:]
            chol, info = torch.linalg.cholesky_ex(sww)
            if bool((info != 0).any()):
                raise AnalysisError("insufficient_observations", "The deterministic components "
                                    "are not identified from this sample; use a longer series.")
            solved = torch.cholesky_solve(cross[:, k:, :k], chol)                   # [B, d, k]
            cross = cross[:, :k, :k] - cross[:, :k, k:] @ solved
            logdet = logdet + 2.0 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum(1)
        return {"syy": cross[:, 0, 0], "sxy": cross[:, 1:, 0], "sxx": cross[:, 1:, 1:],
                "logdet": logdet, "run": run, "weight": weight}

    def beta(self, piece: dict[str, Tensor]) -> Tensor:
        if self.kx == 0:
            return torch.zeros((piece["syy"].shape[0], 0), dtype=torch.float64)
        raw = (piece["weight"][:, :, None] * self.x[None].square()).sum(1)        # [B, kx]
        root = raw.clamp_min(1e-300).sqrt()
        scaled = piece["sxx"] / (root[:, :, None] * root[:, None, :])
        chol, info = torch.linalg.cholesky_ex(piece["sxx"])
        if bool((info != 0).any()) or bool((torch.linalg.eigvalsh(scaled)[:, 0] <= 1e-10).any()):
            raise AnalysisError("collinear_regressors", "The regressors are collinear with the "
                                "unobserved components (for example a constant or a time trend "
                                "together with a stochastic level); remove them.")
        return torch.cholesky_solve(piece["sxy"][..., None], chol).squeeze(-1)

    def profile(self, theta: Tensor) -> Tensor:
        piece = self.pieces(theta)
        quad = piece["syy"]
        if self.kx:
            quad = quad - (piece["sxy"] * self.beta(piece)).sum(1)
        return -0.5 * (self.n * math.log(2.0 * math.pi) + piece["logdet"] + quad)

    def full(self, piece: dict[str, Tensor], beta: Tensor) -> Tensor:
        quad = piece["syy"] - 2.0 * (piece["sxy"] * beta).sum(1) \
            + torch.einsum("k,bkl,l->b", beta, piece["sxx"], beta)
        return -0.5 * (self.n * math.log(2.0 * math.pi) + piece["logdet"] + quad)


def _steps(theta: Tensor, power: float) -> Tensor:
    return _EPS ** power * torch.clamp(theta.abs(), min=1.0)


def _gradient(problem: _Problem, theta: Tensor) -> tuple[Tensor, Tensor]:
    """Profile log likelihood and its central-difference gradient (one batched pass)."""
    p = theta.shape[0]
    h = _steps(theta, 1.0 / 3.0)
    points = theta.repeat(2 * p + 1, 1)
    for i in range(p):
        points[1 + i, i] += h[i]
        points[1 + p + i, i] -= h[i]
    values = problem.profile(points)
    gradient = (values[1:p + 1] - values[p + 1:]) / (2.0 * h)
    return values[0], gradient


def _hessian_points(theta: Tensor) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    p = theta.shape[0]
    h = _steps(theta, 0.25)
    rows = [theta.clone()]
    for i in range(p):
        for sign in (1.0, -1.0):
            point = theta.clone()
            point[i] += sign * h[i]
            rows.append(point)
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    for i, j in pairs:
        for si, sj in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            point = theta.clone()
            point[i] += si * h[i]
            point[j] += sj * h[j]
            rows.append(point)
    return torch.stack(rows), h, pairs


def _second_differences(values: Tensor, h: Tensor, pairs: list[tuple[int, int]], p: int) -> Tensor:
    hessian = torch.zeros((p, p), dtype=torch.float64)
    center = values[0]
    for i in range(p):
        hessian[i, i] = (values[1 + 2 * i] - 2.0 * center + values[2 + 2 * i]) / (h[i] * h[i])
    base = 1 + 2 * p
    for index, (i, j) in enumerate(pairs):
        pp, pm, mp, mm = values[base + 4 * index:base + 4 * index + 4]
        hessian[i, j] = hessian[j, i] = (pp - pm - mp + mm) / (4.0 * h[i] * h[j])
    return hessian


def _profile_hessian(problem: _Problem, theta: Tensor) -> Tensor:
    points, h, pairs = _hessian_points(theta)
    return _second_differences(problem.profile(points), h, pairs, theta.shape[0])


def _maximize(problem: _Problem, theta0: Tensor, max_iterations: int):
    if problem.structure.m == 0:
        # No latent state: the profile likelihood is ordinary Gaussian ML.
        # One global GLS pass gives beta and RSS/h at any positive h; the
        # exact variance maximum is RSS/N, without iterative tolerances.
        piece = problem.pieces(theta0[None])
        beta = problem.beta(piece)[0]
        precision_ss = piece["syy"][0]-(piece["sxy"][0]*beta).sum()
        variance = precision_ss*ss.natural(problem.structure, theta0[None], problem.scale)["var(e)"][0]/problem.n
        if not bool(torch.isfinite(variance)) or not bool(variance>0):
            raise AnalysisError("perfect_fit", "The Gaussian UC regression has no residual variance.")
        theta = torch.log(variance/problem.scale).reshape(1)
        return OptimResult(theta, float(problem.profile(theta[None])[0]), torch.zeros_like(theta),
                           torch.tensor([[-problem.n/2]], dtype=torch.float64), 0, True,
                           "closed_form_gaussian", {"message": "global Gaussian variance ML"})
    def objective(theta: Tensor):
        value, gradient = _gradient(problem, theta)
        if not math.isfinite(float(value)) or not bool(torch.isfinite(gradient).all()):
            return torch.tensor(-math.inf, dtype=torch.float64), torch.zeros_like(theta)
        return value, gradient

    return maximize_bfgs(objective, theta0, max_iter=max_iterations, gradient_tol=1e-6,
                         scaled_gradient_tol=1e-8, step_tol=1e-10,
                         hessian_fn=lambda theta: _profile_hessian(problem, theta),
                         raise_on_failure=False)


def _starting_values(structure: ss.Structure, y: Tensor, scale: float,
                     frequency: float | None) -> dict[str, float]:
    start = {"var(level)": 0.2 * scale, "var(slope)": 0.01 * scale,
             "var(seasonal)": 0.05 * scale, "var(cycle)": 0.2 * scale, "var(e)": 0.5 * scale,
             "damping": 0.9}
    if structure.cycle:
        if frequency is None:
            detrended = y - hp_trend(y, 1600.0) if structure.flags[0] else y - y.mean()
            power = torch.fft.rfft(detrended).abs().square()
            n = y.shape[0]
            band = torch.arange(power.shape[0])
            power[(band < 2) | (band > n // 2)] = 0.0
            frequency = 2.0 * math.pi * int(power.argmax()) / n
            frequency = min(max(frequency, 2.0 * math.pi / n * 2), math.pi * 0.9)
        start["frequency"] = frequency
    return start


def _structure_from(spec: ModelSpec, fixed: frozenset[str] = frozenset()) -> ss.Structure:
    model = spec.options.get("model", "rwalk")
    model = "strend" if model == "smooth_trend" else model
    seasonal = spec.options.get("seasonal") or 0
    return ss.Structure(model, int(seasonal), bool(spec.options.get("cycle", False)), fixed)


def _prepare(spec: ModelSpec, data: Any) -> tuple[ModelFrame, Any, Tensor, Tensor, int | None]:
    frame, last = ordered_frame(spec, data, what="The unobserved-components model")
    design = frame.drop_collinear(frame.design(intercept=False))
    return frame, design, frame.numeric(spec.outcome), design.x, last


def _scale(structure: ss.Structure, y: Tensor) -> float:
    base = y[1:] - y[:-1] if structure.flags[0] else y - y.mean()
    value = float(base.square().mean())
    if not value > 0.0 or not math.isfinite(value):
        raise AnalysisError("constant_outcome", "The outcome (or its first difference) is constant; "
                            "there is nothing to decompose.")
    return value


def fit_ucm(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``ucm`` estimator (see the module docstring and ``ucm``)."""
    frame, design, y, x, last = _prepare(spec, data)
    structure = _structure_from(spec)
    if structure.seasonal == 1:
        raise AnalysisError("invalid_option", "seasonal must be a period of at least 2.")
    if not structure.parameters:
        raise AnalysisError("invalid_model", f"model='{structure.model}' has no stochastic component; "
                            "add a seasonal or cycle component or choose another model.")
    n = frame.n
    if n < structure.m + len(structure.parameters) + x.shape[1] + 3:
        raise AnalysisError("insufficient_observations", f"{n} observations are too few for this "
                            "unobserved-components model.")
    frequency = frame.option("cycle_frequency")
    if frequency is not None and not 0.0 < frequency < math.pi:
        raise AnalysisError("invalid_option", "cycle_frequency must lie strictly between 0 and pi.")
    scale = _scale(structure, y)
    max_iterations = int(frame.option("max_iterations"))
    start = _starting_values(structure, y, scale, frequency)
    fixed: set[str] = set()
    iterations = 0
    unconverged_value = None
    while True:
        structure = ss.Structure(structure.model, structure.seasonal, structure.cycle,
                                 frozenset(fixed))
        if not structure.parameters or not any(name.startswith("var") for name in structure.parameters):
            raise AnalysisError("degenerate_model", "Every variance converged to zero; the model "
                                "fits the data exactly. Choose a model with fewer components.")
        problem = _Problem(structure, y, x, scale)
        theta0 = ss.unconstrained(structure, start, scale)
        result = _maximize(problem, theta0, max_iterations)
        iterations += result.iterations
        if unconverged_value is not None and float(result.value) < unconverged_value - 1e-6 * max(
                1.0, abs(unconverged_value)):
            raise AnalysisError("nonconvergence", "The likelihood maximization did not converge; "
                                "raise max_iterations, give cycle_frequency, or simplify the model.")
        values = {name: float(value[0]) for name, value in
                  ss.natural(structure, result.theta[None, :], scale).items()}
        small = [name for name in structure.parameters
                 if name.startswith("var") and values[name] < _ZERO * scale]
        if not small and not result.converged:
            # A variance running to its zero boundary leaves the log-variance direction flat,
            # so BFGS cannot certify a maximum and may stop just above the 1e-9 threshold.
            # Fix the variances that are already negligible and re-maximize; the restricted
            # maximum must not be worse than the point where the search stopped.
            small = [name for name in structure.parameters
                     if name.startswith("var") and values[name] < _LOOSE_ZERO * scale]
            unconverged_value = float(result.value) if small else None
        if structure.cycle and ("var(cycle)" in small or values["damping"] > 1.0 - 1e-6):
            raise AnalysisError("degenerate_cycle", "The stochastic cycle converged to a "
                                "deterministic, undamped sinusoid (damping -> 1 or zero cycle "
                                f"variance) at frequency {values['frequency']:.4g}; this is usually "
                                "a seasonal pattern: add seasonal=<period>, start the cycle at "
                                "another cycle_frequency, or drop the cycle.")
        if not small:
            break
        fixed.update(small)
        start = {**start, **values}
    if not result.converged:
        raise AnalysisError("nonconvergence", "The likelihood maximization did not converge; raise "
                            "max_iterations, give cycle_frequency, or simplify the model.")
    for name in sorted(fixed):
        frame.warn(f"{name} converged to its zero boundary; it is fixed at 0 and omitted from the "
                   "coefficient table (see extra['variances']).")
    theta = result.theta
    p, kx = theta.shape[0], x.shape[1]
    points, h, pairs = _hessian_points(theta)
    piece = problem.pieces(points)
    beta = problem.beta({k: v[:1] for k, v in piece.items() if k != "run"})[0] if kx else \
        torch.zeros(0, dtype=torch.float64)
    values_full = problem.full(piece, beta)
    log_likelihood = float(values_full[0])
    hessian = torch.zeros((p + kx, p + kx), dtype=torch.float64)
    hessian[:p, :p] = _second_differences(values_full, h, pairs, p)
    if kx:
        score_beta = piece["sxy"] - torch.einsum("bkl,l->bk", piece["sxx"], beta)   # [B, kx]
        for i in range(p):
            hessian[i, p:] = hessian[p:, i] = (score_beta[1 + 2 * i] - score_beta[2 + 2 * i]) / (2 * h[i])
        hessian[p:, p:] = -piece["sxx"][0]
    scores = None
    if spec.covariance in ("robust", "opg"):
        scores = _scores(problem, theta, beta)
    covariance, info = ml_covariance(frame, hessian=hessian, scores=scores)
    jac = torch.cat([ss.jacobian_diagonal(structure, theta, scale), torch.ones(kx, dtype=torch.float64)])
    covariance = jac[:, None] * covariance * jac[None, :]
    natural = [values[name] for name in structure.parameters]
    order = list(range(p, p + kx)) + list(range(p))           # report b first
    params = torch.tensor(beta.tolist() + natural, dtype=torch.float64)
    covariance = covariance[order][:, order]
    terms = design.terms + [f"/{name}" for name in structure.parameters]

    # One-step residuals, smoothed fit and the end-of-sample state.
    adjusted = y - x @ beta if kx else y
    final_values = {name: torch.tensor([values[name]], dtype=torch.float64)
                    for name in structure.parameters}
    t_, q_, h_, p0_ = ss.system(structure, final_values, 1)
    run = kernel_call(ss.kalman, structure, t_, q_, h_, p0_, adjusted[None, :], store=True)
    used = ~run.diffuse_inf
    standardized = (run.v[0, 0] / run.f[0].sqrt())[used]
    smoothed = kernel_call(ss.smooth, structure, t_, q_, h_, p0_, adjusted)
    signal = smoothed @ structure.z + (x @ beta if kx else 0.0)
    tests: dict[str, Any] = {}
    if standardized.shape[0] > 10 and bool((standardized != standardized[0]).any()):
        lags = max(1, min(standardized.shape[0] // 2 - 2, 40))
        tests["ljung_box"] = ljung_box_test(standardized, lags, label=f"Ljung-Box Q({lags}) test of "
                                            "the standardized one-step residuals")
        tests["jarque_bera"] = jarque_bera_test(standardized, label="Jarque-Bera normality test of "
                                                "the standardized one-step residuals")
    k_total = p + kx
    metrics = {**information_criteria(log_likelihood, k_total, n),
               "diffuse_periods": int(run.diffuse_inf.sum()), "iterations": iterations}
    variances = {name: (0.0 if name in fixed else values.get(name)) for name in
                 structure.all_parameters() if name.startswith("var")}
    extra = {
        "model": structure.model, "seasonal": structure.seasonal, "cycle": structure.cycle,
        "variances": variances, "fixed_zero": sorted(fixed),
        "cycle_parameters": {"frequency": values["frequency"], "damping": values["damping"],
                             "period": 2.0 * math.pi / values["frequency"]} if structure.cycle else None,
        "last_state": {"mean": run.final_state.tolist(), "covariance": run.final_covariance.tolist(),
                       "last_period": last},
        "state_names": _state_names(structure), "scale": scale,
    }
    info["correction"] = f"{info['correction']}; variances, frequency and damping by the delta " \
                         "method from the log/logit parameterization"
    bundle = build_result(
        frame, terms=terms, params=params, covariance=covariance,
        title=f"Unobserved-components model ({structure.model})", use_t=False, metrics=metrics,
        fitted=signal, solver="closed_form_gaussian" if structure.m == 0 else "bfgs_exact_diffuse_kalman",
        solver_diagnostics={"converged": True, "iterations": iterations},
        optimizer={"method": result.method, "iterations": iterations,
                   "gradient": "central differences of the exact log likelihood (batched filter)",
                   "hessian": "central second differences of the exact log likelihood (batched)"},
        inference=info, tests=tests, extra=extra, categories=design.categories,
        provenance={"likelihood": "exact diffuse Kalman filter (Durbin-Koopman 2012, eq. 7.4)",
                    "fitted_values": "smoothed signal Z alpha^_t + x_t'b",
                    "time_order": f"sorted by {spec.time}" if spec.time else "input row order",
                    "derivatives": "numerical (no analytic Kalman-derivative recursions)"},
    )
    for coefficient in bundle.coefficients:
        if coefficient.term.startswith("/var("):
            coefficient.p_value = 0.5 * coefficient.p_value
            coefficient.ci_low = max(0.0, coefficient.ci_low)
    return bundle


def _scores(problem: _Problem, theta: Tensor, beta: Tensor) -> Tensor:
    """Per-period scores [n, p + kx]: central differences in theta, analytic in b."""
    p = theta.shape[0]
    h = _steps(theta, 1.0 / 3.0)
    points = theta.repeat(2 * p + 1, 1)
    for i in range(p):
        points[1 + i, i] += h[i]
        points[1 + p + i, i] -= h[i]
    piece = problem.pieces(points, reduce=False)
    run = piece["run"]
    v = run.v[:, 0, :] - (torch.einsum("bkn,k->bn", run.v[:, 1:, :], beta) if beta.numel() else 0.0)
    used = ~run.diffuse_inf
    contribution = torch.where(used, -0.5 * (torch.log(run.f) + v * v / run.f),
                               torch.zeros_like(run.f))
    theta_scores = ((contribution[1:p + 1] - contribution[p + 1:]) / (2.0 * h[:, None])).T
    if not beta.numel():
        return theta_scores
    beta_scores = (run.v[0, 1:, :] * (piece["weight"][0] * v[0])[None, :]).T
    return torch.cat([theta_scores, beta_scores], dim=1)


def _state_names(structure: ss.Structure) -> list[str]:
    names = []
    level, _, slope, _, _ = structure.flags
    names += ["level"] if level else []
    names += ["slope"] if slope else []
    names += [f"seasonal{i}" for i in range(1, structure.seasonal)]
    names += ["cycle", "cycle*"] if structure.cycle else []
    return names


def _fitted_structure(result: ResultBundle) -> tuple[ss.Structure, dict[str, Tensor], Tensor]:
    if not isinstance(result, ResultBundle) or result.spec.estimator != "ucm":
        raise AnalysisError("invalid_result", "This function needs a result returned by oe.ucm.")
    structure = _structure_from(result.spec, frozenset(result.extra.get("fixed_zero") or []))
    estimates = {c.term: c.estimate for c in result.coefficients}
    values = {name: torch.tensor([estimates[f"/{name}"]], dtype=torch.float64)
              for name in structure.parameters}
    beta = torch.tensor([estimates[term] for term in estimates if not term.startswith("/")],
                        dtype=torch.float64)
    return structure, values, beta


def ucm_components(result: ResultBundle, data: Any) -> pd.DataFrame:
    """Smoothed components of a fitted ``oe.ucm`` model (Stata: ``predict, smethod(smooth)``).

    Runs the Kalman filter and the exact diffuse fixed-interval smoother at the
    estimated parameters over ``data`` (the table the model was fitted on, or the
    same columns over another sample) and returns one row per period with
    ``period``, ``observed``, the smoothed ``level``, ``slope``, ``seasonal`` and
    ``cycle`` (only the components the model has), ``regression`` (x_t'b, when there
    are regressors), ``fitted`` (their sum) and ``irregular`` (observed - fitted).

    Example
    -------
    >>> fit = oe.ucm(data=df, y="y", model="lltrend", seasonal=12, time="month")
    >>> parts = oe.ucm_components(fit, df)
    """
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from openecon.econometrics.streaming_ucm import ucm_components_streaming
        return ucm_components_streaming(result, data)
    structure, values, beta = _fitted_structure(result)
    frame, design, y, x, _ = _prepare(result.spec, data)
    if design.terms != [c.term for c in result.coefficients if not c.term.startswith("/")]:
        raise AnalysisError("invalid_data", "The data do not reproduce the regressors of the fitted "
                            "model.")
    regression = x @ beta if beta.numel() else torch.zeros_like(y)
    t, q, h, p0 = ss.system(structure, values, 1)
    smoothed = kernel_call(ss.smooth, structure, t, q, h, p0, y - regression)
    columns: dict[str, Any] = {}
    if result.spec.time is not None:
        columns["period"] = frame.series(result.spec.time).tolist()
    else:
        columns["period"] = frame.positions
    columns["observed"] = y.tolist()
    names = _state_names(structure)
    for name in ("level", "slope"):
        if name in names:
            columns[name] = smoothed[:, names.index(name)].tolist()
    if structure.seasonal:
        columns["seasonal"] = smoothed[:, structure.season_index].tolist()
    if structure.cycle:
        columns["cycle"] = smoothed[:, structure.cycle_index].tolist()
    if beta.numel():
        columns["regression"] = regression.tolist()
    fitted = smoothed @ structure.z + regression
    columns["fitted"] = fitted.tolist()
    columns["irregular"] = (y - fitted).tolist()
    return table(columns, title=f"Smoothed components of {result.spec.outcome}",
                 model=structure.model, smoother="exact diffuse fixed-interval smoother")


def ucm_forecast(result: ResultBundle, steps: int, *, data: Any = None, exog: Any = None,
                 alpha: float = 0.05) -> pd.DataFrame:
    """Forecasts from a fitted ``oe.ucm`` model (``oe.forecast`` dispatches here).

    From the filtered state at the end of the sample (``result.extra['last_state']``,
    or the end of ``data`` when given) the state is propagated with the transition
    matrix: ``yhat_(n+h) = Z T^(h-1) a_(n+1) + x_(n+h)'b`` with mean squared error
    ``Z P_(n+h) Z' + var(e)``, ``P <- T P T' + Q``. ``exog`` gives the future values
    of the regressors (one row per step). Parameter uncertainty is ignored (as Stata's
    ``predict, dynamic()``). Returns ``period``, ``forecast``, ``std_error``, ``ci_low``,
    ``ci_high``.

    Example
    -------
    >>> oe.forecast(fit, steps=12)
    """
    from openecon.econometrics.arima.forecast import _future_regressors
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from openecon.econometrics.streaming_ucm import ucm_forecast_streaming
        return ucm_forecast_streaming(result, steps, data=data, exog=exog, alpha=alpha)

    structure, values, beta = _fitted_structure(result)
    steps = as_int(steps)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 10_000:
        raise AnalysisError("invalid_steps", "steps must be an integer between 1 and 10000.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < alpha < 1.0:
        raise AnalysisError("invalid_option", "alpha must be a number strictly between 0 and 1.")
    t, q, h, p0 = ss.system(structure, values, 1)
    if data is None:
        state = result.extra["last_state"]
        a = torch.tensor(state["mean"], dtype=torch.float64)
        p = torch.tensor(state["covariance"], dtype=torch.float64).reshape(structure.m, structure.m)
        last, origin = state.get("last_period"), "end of the estimation sample"
    else:
        frame, design, y, x, last = _prepare(result.spec, data)
        adjusted = y - x @ beta if beta.numel() else y
        run = kernel_call(ss.kalman, structure, t, q, h, p0, adjusted[None, :], store=True)
        a, p, origin = run.final_state, run.final_covariance, "end of the supplied data"
    terms = [c.term for c in result.coefficients if not c.term.startswith("/")]
    if terms:
        future = _future_regressors(result, terms, exog, steps) @ beta
    else:
        if exog is not None:
            raise AnalysisError("invalid_exog", "The model has no regressors; do not pass exog.")
        future = torch.zeros(steps, dtype=torch.float64)
    z, tm, qm, hv = structure.z, t[0], torch.diag(q[0]), float(h[0])
    point, variance = [], []
    for _ in range(steps):
        point.append(float(z @ a))
        variance.append(float(z @ p @ z) + hv)
        a = tm @ a
        p = tm @ p @ tm.T + qm
    point_t = torch.tensor(point, dtype=torch.float64) + future
    std = torch.tensor(variance, dtype=torch.float64).clamp_min(0.0).sqrt()
    if not bool(torch.isfinite(point_t).all()) or not bool(torch.isfinite(std).all()):
        raise AnalysisError("non_finite_result", "The forecasts are not finite.")
    critical = normal_isf(alpha / 2.0)
    periods = list(range(1, steps + 1)) if last is None else [last + i for i in range(1, steps + 1)]
    return table({"period": periods, "forecast": point_t.tolist(), "std_error": std.tolist(),
                  "ci_low": (point_t - critical * std).tolist(),
                  "ci_high": (point_t + critical * std).tolist()},
                 title=f"Forecasts of {result.spec.outcome}: UC model ({structure.model})",
                 steps=steps, alpha=alpha, origin=origin,
                 std_error_definition="state and irregular uncertainty (no parameter uncertainty)")


def ucm(*, data: Any, y: str, x: Any = None, time: str | None = None, model: str = "rwalk",
        seasonal: int | None = None, cycle: bool = False, cycle_frequency: float | None = None,
        covariance: str | None = None, categorical: Any = None, max_iterations: int = 200,
        missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Unobserved-components model by exact diffuse Kalman-filter ML (Stata's ``ucm``).

    Model
    -----
    ``y_t = mu_t + gamma_t + psi_t + x_t'b + e_t`` with a trend ``mu_t`` chosen by
    ``model`` (Stata's names): ``"rwalk"`` (default; random walk, no irregular),
    ``"llevel"`` (local level + irregular), ``"lltrend"`` (local linear trend: level and
    slope disturbed), ``"strend"`` / ``"smooth_trend"`` (only the slope disturbed),
    ``"rtrend"`` (smooth trend without irregular), ``"rwdrift"`` (random walk with
    drift), ``"lldtrend"`` (local level with drift), ``"dtrend"`` (deterministic
    trend), ``"dconstant"`` (constant + irregular), ``"ntrend"`` (irregular only),
    ``"none"``. ``seasonal=s`` adds a stochastic dummy seasonal of period s;
    ``cycle=True`` a stochastic damped cycle (frequency, damping, variance).

    Estimator
    ---------
    Exact Gaussian ML with the exact diffuse initialization of Durbin and Koopman
    (trend and seasonal states diffuse; the cycle stationary). Variances are optimized
    in logs, b is concentrated out by GLS; BFGS with numerical derivatives of the exact
    likelihood (batched filter evaluations). Variances that converge to zero are fixed
    at zero (warning; listed in ``extra["fixed_zero"]``).

    Parameters
    ----------
    data, y : table and outcome. x : optional regressors (no constant: the level plays
    that role). time : optional time column (consecutive periods). model, seasonal,
    cycle : see Model. cycle_frequency : starting frequency of the cycle in radians
    (default: the periodogram peak of the HP-detrended series). covariance :
    ``"nonrobust"`` (default; observed information, Stata's ``vce(oim)``),
    ``"robust"`` (Huber-White with N/(N-1), Stata's ``vce(robust)``) or ``"opg"``.
    categorical : regressors to expand into indicators. max_iterations : BFGS limit.
    missing : ``"raise"`` or ``"drop"`` (rows only at the start or end). alpha : level.

    Result
    ------
    Coefficients: the regressors, then ``/frequency``, ``/damping`` (cycle),
    ``/var(level)``, ``/var(slope)``, ``/var(seasonal)``, ``/var(cycle)``, ``/var(e)``
    (those the model estimates); z tests, one-sided for variances with intervals
    truncated at zero (as Stata). ``metrics``: ``log_likelihood`` (exact diffuse),
    ``aic``, ``bic`` (k = estimated parameters, N = observations), ``diffuse_periods``,
    ``iterations``. ``tests``: Ljung-Box and Jarque-Bera of the standardized one-step
    residuals. ``extra``: ``variances`` (zero-fixed included), ``cycle_parameters``
    (frequency, damping, period 2 pi / frequency), ``last_state`` (for forecasts).
    Predictions plot the smoothed signal. ``oe.ucm_components(result, data)`` gives
    the smoothed components, ``oe.forecast(result, steps)`` forecasts.

    Stata: ``ucm y, model(llevel)``, ``ucm y x, model(lltrend) seasonal(12) cycle(1)``.
    EViews: ``sspace`` with ``@signal`` / ``@state`` UC specifications.

    Example
    -------
    >>> fit = oe.ucm(data=df, y="lgdp", model="lltrend", cycle=True, time="quarter")
    >>> fit.extra["cycle_parameters"]["period"]
    """
    spec = build_spec(
        "ucm", outcome=y, predictors=column_list(x, "x"), intercept=False,
        categorical=column_list(categorical, "categorical"), time=time, covariance=covariance,
        missing=missing, alpha=alpha,
        options={"model": model, "seasonal": as_int(seasonal), "cycle": cycle,
                 "cycle_frequency": cycle_frequency, "max_iterations": as_int(max_iterations)},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
