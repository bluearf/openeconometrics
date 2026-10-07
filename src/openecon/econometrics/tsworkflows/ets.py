"""Additive-error innovations ETS with full joint Gaussian ML information.

ANN/AAN/AdN/ANA/AAA/AdA are supported; multiplicative components are rejected.
Smoothing coefficients obey the usual coupled bounds beta<=alpha,
gamma<=1-alpha. Seasonal initial states have a zero-sum identifying constraint.
This estimates a statistical model, distinct from the existing tssmooth helper.
"""

from __future__ import annotations
import math
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, make_spec, build_result, information_criteria
from openecon.econometrics.tsworkflows.common import (
    finite_tensor,
    ordered,
    ml_optimize,
    horizon,
    forecast_table,
)


def matrices(alpha, beta, gamma, phi, trend, seasonal, period):
    m = 1 + int(trend) + (period if seasonal else 0)
    F = torch.zeros((m, m), dtype=torch.float64)
    w, g = torch.zeros(m, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
    F[0, 0], w[0], g[0] = 1.0, 1.0, alpha
    offset = 1 + int(trend)
    if trend:
        F[0, 1], F[1, 1], w[1], g[1] = phi, phi, phi, beta
    if seasonal:
        F[offset:-1, offset + 1 :] = torch.eye(period - 1, dtype=torch.float64)
        F[-1, offset], w[offset], g[-1] = 1.0, 1.0, gamma
    return F, w, g


class ETS:
    def __init__(self, frame, y):
        self.model = frame.option("model")
        self.trend, self.damped, self.seasonal = (
            self.model not in {"ANN", "ANA"},
            "d" in self.model,
            self.model.endswith("A"),
        )
        self.period = frame.option("period") if self.seasonal else 0
        self.fixed = frame.option("fixed")
        relevant = (
            ["alpha"]
            + (["beta"] if self.trend else [])
            + (["gamma"] if self.seasonal else [])
            + (["phi"] if self.damped else [])
        )
        if not isinstance(self.fixed, dict) or set(self.fixed) - set(relevant):
            raise AnalysisError(
                "invalid_option",
                "fixed may specify only smoothing parameters used by this ETS model.",
            )
        for key, value in self.fixed.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not 0 < value < 1
            ):
                raise AnalysisError(
                    "invalid_option", "Fixed smoothing parameters must lie strictly in (0,1)."
                )
        self.free = [key for key in relevant if key not in self.fixed]
        self.alpha_bounds = (self.fixed.get("beta", 0.0), 1 - self.fixed.get("gamma", 0.0))
        if self.alpha_bounds[0] >= self.alpha_bounds[1]:
            raise AnalysisError(
                "invalid_option", "Fixed beta/gamma leave no feasible alpha interval."
            )
        if self.seasonal and frame.n < 3 * self.period:
            raise AnalysisError(
                "insufficient_observations", "Seasonal ETS needs at least three complete cycles."
            )
        self.initial = frame.option("initial")
        if self.initial is not None:
            expected = 1 + int(self.trend) + (self.period if self.seasonal else 0)
            self.initial = finite_tensor(self.initial, (expected,), "initial")
            if self.seasonal and abs(float(self.initial[-self.period :].sum())) > 1e-10:
                raise AnalysisError(
                    "unidentified_system", "Initial additive seasonal effects must sum to zero."
                )
        start_values = {"alpha": 0.25, "beta": 0.1, "gamma": 0.1, "phi": 0.9}
        if "alpha" in self.free:
            low, high = self.alpha_bounds
            start_values["alpha"] = (0.25 - low) / (high - low) if low < 0.25 < high else 0.5
        starts = [math.log(start_values[k] / (1 - start_values[k])) for k in self.free]
        self.names = self.free.copy()
        if self.initial is None:
            width = self.period if self.seasonal else min(10, len(y))
            level = y[:width].mean()
            starts.append(float(level))
            self.names.append("initial_level")
            if self.trend:
                slope = (
                    (y[width : 2 * width].mean() - level) / width
                    if len(y) >= 2 * width
                    else y[1] - y[0]
                )
                starts.append(float(slope))
                self.names.append("initial_trend")
            if self.seasonal:
                effects = y[: self.period] - level
                starts.extend(effects[:-1].tolist())
                self.names.extend(f"initial_season_{i + 1}" for i in range(self.period - 1))
        variance = float(torch.diff(y).square().mean() / 2)
        if not variance > 0 or not math.isfinite(variance):
            raise AnalysisError("constant_outcome", "ETS needs nonconstant finite observations.")
        starts.append(0.5 * math.log(variance))
        self.names.append("sigma")
        self.start = torch.tensor(starts, dtype=torch.float64)
        if frame.n <= len(starts) + 3:
            raise AnalysisError(
                "insufficient_observations", "ETS needs N>K+3 observations for joint information."
            )
        self.decode(self.start)

    def decode(self, z):
        params = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in self.fixed.items()}
        for i, key in enumerate(self.free):
            fraction = torch.sigmoid(z[i])
            params[key] = (
                self.alpha_bounds[0] + fraction * (self.alpha_bounds[1] - self.alpha_bounds[0])
                if key == "alpha"
                else fraction
                * (
                    params["alpha"]
                    if key == "beta"
                    else 1 - params["alpha"]
                    if key == "gamma"
                    else 1
                )
            )
        zero, one = torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
        alpha, beta, gamma, phi = (
            params.get(k, default)
            for k, default in (("alpha", zero), ("beta", zero), ("gamma", zero), ("phi", one))
        )
        if float(beta.detach()) > float(alpha.detach()) or float(gamma.detach()) > 1 - float(
            alpha.detach()
        ):
            raise AnalysisError("invalid_option", "ETS requires beta<=alpha and gamma<=1-alpha.")
        offset = len(self.free)
        if self.initial is not None:
            initial = self.initial
        else:
            leading = 1 + int(self.trend)
            initial = z[offset : offset + leading]
            if self.seasonal:
                effects = z[offset + leading : -1]
                initial = torch.cat([initial, effects, -effects.sum().reshape(1)])
        sigma = torch.exp(z[-1])
        physical = torch.stack([params[k] for k in self.free] + list(z[offset:-1]) + [sigma])
        F, w, g = matrices(alpha, beta, gamma, phi, self.trend, self.seasonal, self.period)
        # Stability of the error-correction dynamics is needed for innovations ETS.
        radius = float(torch.linalg.eigvals((F - g[:, None] @ w[None, :]).detach()).abs().max())
        if radius > 1 + 1e-9:
            raise AnalysisError(
                "unstable_system", "ETS error-correction transition is not admissible."
            )
        return F, w, g, initial, sigma, physical, params

    def evaluate(self, z, y, retain=False):
        F, w, g, state, sigma, _, _ = self.decode(z)
        residuals, predictions = [], []
        for obs in y:
            mean = w @ state
            error = obs - mean
            residuals.append(error)
            if retain:
                predictions.append(mean)
            state = F @ state + g * error
        errors = torch.stack(residuals)
        ll = (
            -0.5 * len(y) * math.log(2 * math.pi)
            - len(y) * sigma.log()
            - 0.5 * errors.square().sum() / sigma.square()
        )
        return ll, state, torch.stack(predictions) if retain else None


def fit_ets(spec, data):
    frame = ModelFrame(spec, data)
    ordered(frame)
    y = frame.numeric(spec.outcome)
    engine = ETS(frame, y)
    k, m = len(engine.start), 1 + int(engine.trend) + engine.period
    if frame.n > 20000 or frame.n * m * k * k > 40_000_000:
        raise AnalysisError(
            "system_budget",
            "ETS joint-information geometry exceeds its explicit computation budget.",
        )
    frame.workspace_plan(
        "ETS joint likelihood derivatives", {"autodiff_graph_estimate": frame.n * m * k * 256}
    )
    solution, cov = ml_optimize(
        lambda z: engine.evaluate(z, y)[0],
        engine.start,
        frame.option("max_iterations"),
        frame.option("tolerance"),
    )
    fractions = torch.sigmoid(solution.theta[: len(engine.free)])
    if bool(((fractions < 1e-6) | (fractions > 1 - 1e-6)).any()):
        raise AnalysisError(
            "boundary_solution",
            "A free ETS smoothing parameter is on a constraint boundary; interior normal inference is unavailable.",
        )
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(lambda z: engine.decode(z)[5], solution.theta)
    F, w, g, initial, sigma, physical, smoothing = engine.decode(solution.theta)
    ll, state, predictions = engine.evaluate(solution.theta, y, True)
    return build_result(
        frame,
        terms=engine.names,
        params=physical,
        covariance=J @ cov @ J.T,
        use_t=False,
        title=f"ETS({engine.model})",
        metrics=information_criteria(float(ll), k, frame.n),
        fitted=predictions.detach(),
        solver="joint Gaussian innovations ML, analytic Torch derivatives",
        optimizer={
            "converged": solution.converged,
            "iterations": solution.iterations,
            "diagnostics": solution.diagnostics,
        },
        extra={
            "model": engine.model,
            "period": engine.period,
            "transition": F,
            "observation": w,
            "gain": g,
            "initial_state": initial,
            "last_state": state,
            "sigma": sigma,
            "smoothing": smoothing,
            "initial_estimated": engine.initial is None,
        },
        inference={
            "forecast_parameter_uncertainty": False,
            "initial_state_identification": "seasonal effects constrained to sum zero",
        },
        provenance={
            "likelihood": "Gaussian additive innovations; estimated initial states included in joint covariance",
            "supported_domain": "ANN/AAN/AdN/ANA/AAA/AdA; no multiplicative errors/trend/seasonality",
        },
    )


def ets(
    *,
    data,
    y,
    model="ANN",
    period=2,
    time=None,
    fixed=None,
    initial=None,
    max_iterations=300,
    tolerance=1e-7,
    alpha=0.05,
    missing="raise",
):
    return fit_ets(
        make_spec(
            "ets",
            outcome=y,
            predictors=[],
            intercept=False,
            time=time,
            alpha=alpha,
            missing=missing,
            options={
                "model": model,
                "period": period,
                "fixed": {} if fixed is None else fixed,
                "initial": initial,
                "max_iterations": max_iterations,
                "tolerance": tolerance,
            },
        ),
        data,
    )


def forecast(result, steps, *, alpha=None):
    horizon(steps)
    if result.spec.estimator != "ets":
        raise AnalysisError("invalid_result", "ETS forecast needs an ets result.")
    e = result.extra
    F, w, g, state = (
        finite_tensor(e[k]) for k in ("transition", "observation", "gain", "last_state")
    )
    sigma2 = float(e["sigma"]) ** 2
    covariance = torch.zeros_like(F)
    means, variances = [], []
    for _ in range(steps):
        means.append(w @ state)
        variances.append(w @ covariance @ w + sigma2)
        state = F @ state
        covariance = F @ covariance @ F.T + sigma2 * g[:, None] @ g[None, :]
    return forecast_table(
        torch.stack(means),
        torch.stack(variances),
        result.spec.alpha if alpha is None else alpha,
        origin=result.nobs,
        model=e["model"],
        initial_state_uncertainty="included in fitted-parameter covariance; excluded from conditional forecast intervals",
    )
