"""Bounded approximate ARFIMA CSS with complete saved forecast state."""

from __future__ import annotations

import hashlib
import json
import math
import torch
from pandas.api.types import is_bool_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, information_criteria, kernel_call, make_spec, table
from openecon.engines.optimize import information_inverse, maximize_bfgs
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle
from . import kernels as k
from .procedures import allocation, integer, number, regular_time

MAX_FIT_ROWS = 32768
MAX_FIT_WORK = 60_000_000
MAX_FORECAST = 256


def _state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def arfima(data, outcome, *, ar=0, ma=0, d=None, terms=256, burn=0, constant=True,
           time=None, method="css", covariance="nonrobust", alpha=.05,
           max_iterations=200, tolerance=1e-8, missing="raise"):
    """Fit approximate stationary ARFIMA with a finite filter and zero prehistory.

    p/q<=3, |d|<=.49 (estimated d strictly interior); no regressors/weights.
    The finite filter is held fixed during optimization. ``burn`` likelihood
    rows are excluded while their data still initialize the causal recursion.
    See docs/econometrics/fractional-memory.md for all supported domains.
    """
    spec = make_spec("arfima", outcome=outcome, covariance=covariance, alpha=alpha, intercept=constant,
                     time=time, missing=missing, options={"ar": ar, "ma": ma, "d": d,
                     "terms": terms, "burn": burn, "constant": constant, "method": method,
                     "max_iterations": max_iterations, "tolerance": tolerance})
    return fit_arfima(spec, data)


def fit_arfima(spec, data):
    # Reject large resident/Dataset inputs before selection and numerical work.
    if isinstance(data, dict) and spec.outcome in data and hasattr(data[spec.outcome], "__len__") and len(data[spec.outcome]) > MAX_FIT_ROWS:
        raise AnalysisError("work_budget", f"ARFIMA fits at most {MAX_FIT_ROWS} resident rows.")
    if hasattr(data, "__len__") and not isinstance(data, dict) and len(data) > MAX_FIT_ROWS:
        raise AnalysisError("work_budget", f"ARFIMA fits at most {MAX_FIT_ROWS} resident rows.")
    frame = ModelFrame(spec, data)
    n = frame.n
    if n != len(frame.original):
        raise AnalysisError("missing_values", "ARFIMA requires every period; missing rows are never compressed.")
    if n > MAX_FIT_ROWS:
        raise AnalysisError("work_budget", f"ARFIMA fits at most {MAX_FIT_ROWS} resident rows.")
    if is_bool_dtype(frame.original[spec.outcome].dtype):
        raise AnalysisError("non_numeric_column", "ARFIMA requires real numeric observations; boolean outcomes are unsupported.")
    options = spec.options
    p, q = options.get("ar", 0), options.get("ma", 0)
    terms, burn = options.get("terms", 256), options.get("burn", 0)
    constant, fixed_d = options.get("constant", spec.intercept), options.get("d")
    iterations, tolerance = options.get("max_iterations", 200), options.get("tolerance", 1e-8)
    count = int(constant) + int(fixed_d is None) + p + q + 1
    if terms > n or n - burn < max(32, 4 * count):
        raise AnalysisError("insufficient_observations", "Require terms<=N and at least max(32,4K) likelihood rows after burn.")
    if n * iterations * count > MAX_FIT_WORK:
        raise AnalysisError("work_budget", "N * max_iterations * parameter_count exceeds 60 million.")
    workspace = allocation(n, fit=True).record()
    if spec.time is not None:
        if spec.time == spec.outcome:
            raise AnalysisError("invalid_time", "time must differ from the outcome.")
        regular_time(frame.original[spec.time])
    y = frame.numeric(spec.outcome)
    location = y.mean() if constant else torch.tensor(0., dtype=k.FLOAT)
    scale = torch.sqrt((y - y.mean()).square().mean())
    if not torch.isfinite(scale) or scale <= 1e-12 * max(1., float(y.abs().max())):
        raise AnalysisError("degenerate_series", "ARFIMA requires a nonconstant, numerically resolved series.")
    z = (y - location) / scale

    def unpack(raw):
        pos = 0
        mean = raw[pos] if constant else torch.tensor(0., dtype=k.FLOAT)
        pos += int(constant)
        dvalue = .49 * torch.tanh(raw[pos]) if fixed_d is None else torch.tensor(fixed_d, dtype=k.FLOAT)
        pos += int(fixed_d is None)
        ar = k.pacf_coefficients(raw[pos:pos + p])
        pos += p
        ma = -k.pacf_coefficients(raw[pos:pos + q])
        return mean, dvalue, ar, ma, torch.exp(raw[-1])

    def objective(raw):
        mean, dvalue, ar, ma, sigma = unpack(raw)
        return k.loglike(z, mean, dvalue, ar, ma, sigma, terms, burn)

    def value_gradient(point):
        with torch.enable_grad():
            raw = point.detach().requires_grad_()
            value = objective(raw)
            gradient = torch.autograd.grad(value, raw)[0]
        return float(value.detach()), gradient.detach()

    def hessian(point):
        with torch.enable_grad():
            return torch.autograd.functional.hessian(objective, point).detach()

    candidates = []
    starts = (0., .25) if fixed_d is None else (0.,)
    for start_d in starts:
        raw = torch.zeros(count, dtype=k.FLOAT)
        if fixed_d is None:
            raw[int(constant)] = math.atanh(start_d / .49)
        trial = kernel_call(maximize_bfgs, value_gradient, raw, max_iter=iterations,
                            gradient_tol=tolerance, hessian_fn=hessian, raise_on_failure=False)
        if trial.converged:
            candidates.append(trial)
    if not candidates:
        raise AnalysisError("nonconvergence", "ARFIMA conditional likelihood did not converge to an identified interior maximum.")
    best = max(candidates, key=lambda trial: trial.value)
    raw = best.theta.detach()
    mean, dvalue, ar, ma, sigma = (v.detach() for v in unpack(raw))
    if (fixed_d is None and abs(float(dvalue)) > .485) or (p + q and float(raw[-1 - p - q:-1].abs().max()) > 4):
        raise AnalysisError("boundary_solution", "The fit is too close to a stationarity/invertibility bound for regular inference.")

    def reporting(point):
        m, dv, av, mv, sv = unpack(point)
        pieces = ([((m * scale) + location).reshape(1)] if constant else [])
        if fixed_d is None:
            pieces.append(dv.reshape(1))
        return torch.cat([*pieces, av, mv, (sv * scale).reshape(1)])
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(reporting, raw)
    covariance = jacobian @ kernel_call(information_inverse, -hessian(raw)) @ jacobian.T
    params = reporting(raw).detach()
    terms_names = (["Mean"] if constant else []) + (["d"] if fixed_d is None else [])
    terms_names += [f"AR{i + 1}" for i in range(p)] + [f"MA{i + 1}" for i in range(q)] + ["sigma"]
    mean = float(mean * scale + location) if constant else 0.
    sigma = float(sigma * scale)
    errors, polynomial = k.innovations(y, torch.tensor(mean, dtype=k.FLOAT), dvalue, ar, ma, terms)
    length = len(polynomial) - 1
    state = {"schema": 1, "model": "finite-filter-css-arfima", "mean": mean,
             "d": float(dvalue), "ar": ar.tolist(), "ma": ma.tolist(), "sigma": sigma,
             "terms": terms, "burn": burn, "n_input": n,
             "polynomial": polynomial.tolist(), "history": (y - mean)[-length:].tolist(),
             "innovations": errors[-q:].tolist() if q else [],
             "last_time": int(frame.original[spec.time].iloc[-1]) if spec.time else None,
             "parameter_terms": terms_names, "parameters": params.tolist()}
    state_hash = _state_hash(state)
    if burn:
        frame.restrict(torch.arange(n) >= burn, f"Excluded {burn} initial innovations from the conditional likelihood.")
    likelihood = best.value - (n - burn) * math.log(float(scale))
    warning = "Approximate finite fractional filter and zero prehistory; exact stationary likelihood and vendor parity are unvalidated."
    return build_result(frame, terms=terms_names, params=params, covariance=covariance,
                        title=f"Conditional ARFIMA({p},{float(dvalue):.4g},{q})", use_t=False,
                        df_resid=n - burn - count, fitted=(y - errors)[burn:],
                        metrics=information_criteria(likelihood, count, n - burn),
                        solver="Torch autograd BFGS, finite-filter CSS",
                        optimizer={"converged": True, "iterations": best.iterations,
                                   "starts": list(starts), "gradient": best.gradient.tolist()},
                        inference={"correction": "inverse full observed information; Gaussian z",
                                   "residual_definition": "conditional innovation y - one-step fitted",
                                   "parameter_uncertainty_in_forecast": False},
                        extra={"fractional_state": state, "fractional_state_sha256": state_hash,
                               "residuals": errors[burn:].tolist(), "fixed_d": fixed_d,
                               "approximation": "finite fractional filter; zero presample centered observations/innovations",
                               "first_omitted_fractional_coefficient": float(k.weights(dvalue, terms + 1)[-1])},
                        provenance={"fractional_workspace": workspace, "dataset_support": False}, warnings=[warning])


@torch.no_grad()
def forecast(result, steps, *, alpha=None):
    """Conditional finite-filter forecasts with full cross-horizon innovation covariance."""
    steps = integer(steps, "steps", 1, MAX_FORECAST)
    if not isinstance(result, ResultBundle) or result.spec.estimator != "arfima":
        raise AnalysisError("invalid_result", "An ARFIMA ResultBundle is required.")
    alpha = number(result.spec.alpha if alpha is None else alpha, "alpha", 1e-8, 1 - 1e-8)
    state = result.extra.get("fractional_state")
    try:
        if not isinstance(state, dict) or _state_hash(state) != result.extra.get("fractional_state_sha256"):
            raise ValueError("missing or changed state")
        names = [coefficient.term for coefficient in result.coefficients]
        params = [coefficient.estimate for coefficient in result.coefficients]
        if state["schema"] != 1 or state["model"] != "finite-filter-css-arfima" or state["parameter_terms"] != names or state["parameters"] != params:
            raise ValueError("state differs from fitted coefficients")
        d = number(state["d"], "state d", -.49, .49)
        count = integer(state["terms"], "state terms", 2, 16384)
        ar, ma = torch.tensor(state["ar"], dtype=k.FLOAT), torch.tensor(state["ma"], dtype=k.FLOAT)
        polynomial = k.convolve(k.weights(d, count), torch.cat((torch.ones(1, dtype=k.FLOAT), -ar)))
        history = list(state["history"])
        errors = list(state["innovations"])
        sigma = number(state["sigma"], "state sigma", 1e-300)
        mean = number(state["mean"], "state mean")
        if len(ar) > 3 or len(ma) > 3 or len(history) != len(polynomial) - 1 or len(errors) != len(ma):
            raise ValueError("invalid state dimensions")
        if not torch.isfinite(torch.tensor([*history, *errors, *ar, *ma], dtype=k.FLOAT)).all():
            raise ValueError("nonfinite state")
        if not torch.allclose(polynomial, torch.tensor(state["polynomial"], dtype=k.FLOAT), rtol=1e-13, atol=1e-13):
            raise ValueError("filter mismatch")
        if count != result.spec.options.get("terms", 256) or len(ar) != result.spec.options.get("ar", 0) or len(ma) != result.spec.options.get("ma", 0):
            raise ValueError("spec mismatch")
        expected = dict(zip(names, params))
        expected_d = expected.get("d", result.spec.options.get("d"))
        if expected_d != d or expected.get("Mean", 0.) != mean or expected["sigma"] != sigma:
            raise ValueError("state scalar differs from model")
        if ar.tolist() != [expected[f"AR{i + 1}"] for i in range(len(ar))] or ma.tolist() != [expected[f"MA{i + 1}"] for i in range(len(ma))]:
            raise ValueError("state polynomial differs from model")
        if state["n_input"] != result.nobs_original or state["burn"] != result.dropped_rows:
            raise ValueError("state sample mismatch")
    except (ValueError, KeyError, TypeError, RuntimeError, AnalysisError) as exc:
        raise AnalysisError("invalid_forecast_state", "The saved ARFIMA state is missing, changed or incompatible; refit explicitly.") from exc
    values = []
    for _ in range(steps):
        recent = torch.tensor(history[-(len(polynomial) - 1):][::-1], dtype=k.FLOAT)
        value = -float(polynomial[1:] @ recent)
        if len(ma):
            value += float(ma @ torch.tensor(errors[-len(ma):][::-1], dtype=k.FLOAT))
        history.append(value)
        errors.append(0.)
        values.append(mean + value)
    psi = k.convolve(k.inverse(polynomial, steps), torch.cat((torch.ones(1, dtype=k.FLOAT), ma)), steps)
    mapping = torch.zeros((steps, steps), dtype=k.FLOAT)
    for row in range(steps):
        mapping[row, :row + 1] = psi[:row + 1].flip(0)
    covariance = sigma ** 2 * mapping @ mapping.T
    se = covariance.diagonal().sqrt()
    critical = critical_value(alpha)
    center = torch.tensor(values, dtype=k.FLOAT)
    output = {"step": list(range(1, steps + 1)), "forecast": values, "std_error": se.tolist(),
              "ci_low": (center - critical * se).tolist(), "ci_high": (center + critical * se).tolist()}
    if state["last_time"] is not None:
        output["time"] = [state["last_time"] + i for i in range(1, steps + 1)]
    return table(output, alpha=alpha, covariance_matrix=covariance.tolist(), impulse_response=psi.tolist(),
                 conditional_on="saved centered history, zero presample values and fitted innovations",
                 uncertainty="Gaussian future innovations only; parameter/initial-state uncertainty excluded",
                 model="finite-filter-css-arfima", state_sha256=result.extra["fractional_state_sha256"],
                 device="cpu", precision="float64", horizon_budget=MAX_FORECAST)
