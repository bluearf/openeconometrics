"""Linear Gaussian systems with proper priors, observation masks and known paths.

y_t = Z a_t + d + e_t, e_t~N(0,H);
a_{t+1} = T a_t + c + u_t, u_t~N(0,Q), independent of e_t.
The initial prior is for a_1, before observing y_1. No covariance projection,
diffuse approximation, or silently dropped observation period is used.
"""

from __future__ import annotations
from functools import wraps
import math
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    make_spec,
    build_result,
    information_criteria,
    table,
)
from openecon.econometrics.tsworkflows.common import (
    finite_tensor,
    pd_check,
    ordered,
    ml_optimize,
    horizon,
    forecast_table,
)


from openecon.econometrics.tsworkflows.ssmodel import System, exogenous_columns
from openecon.econometrics.tsworkflows.ssengine import kalman, smooth, at


def _cpu_call(function):
    @wraps(function)
    def run(*args, **kwargs):
        with torch.device("cpu"), torch.inference_mode(False):
            return function(*args, **kwargs)
    return run


def _prepare(spec, data):
    from openecon.econometrics.registry import role_columns

    responses = list(dict.fromkeys([spec.outcome, *role_columns(spec, "responses")]))
    masked = spec.options.get("observation_missing", "raise") == "mask"
    frame = ModelFrame(spec, data, allow_missing=responses if masked else (),
                       extra_columns=exogenous_columns(spec.options.get("system")))
    # Missing exogenous values and dates cannot remove observation periods.
    if frame.dropped_missing and masked:
        raise AnalysisError("missing_values", "Masked state-space models require complete dates and exogenous paths.")
    ordered(frame)
    system = System(frame.option("system"), len(responses), frame=frame)
    frame.spec.options["system"] = system.original_record
    if (
        frame.n > 20000
        or system.lyapunov_work + system.path_work
        + frame.n * (system.m**3 + system.p**3) * max(1, len(system.start)) ** 2
        > 50_000_000
    ):
        raise AnalysisError(
            "system_budget",
            "Exact differentiable state-space likelihood exceeds the supported operation geometry.",
        )
    frame.workspace_plan(
        "state-space likelihood, autodiff and retained moments",
        {
            "Lyapunov_operator_factors_and_derivatives": system.lyapunov_bytes,
            "autodiff_graph_estimate": frame.n
            * (system.m**2 + system.p**2)
            * max(1, len(system.start))
            * 256,
            "retained_moments": frame.n * (system.m**2 + system.m + system.p**2 + system.p) * 64,
            "equation_paths_and_json_copies": frame.n * (3*system.m**2 + system.p**2 + system.m*system.p + system.m + system.p) * 64,
        },
    )
    y = torch.stack([frame.numeric(name, allow_missing=masked) for name in responses], dim=1)
    if not bool(torch.isfinite(y).any()) and len(system.start):
        raise AnalysisError("unidentified_system", "No observed measurements identify free parameters.")
    return frame, system, y, responses


@_cpu_call
def fit_sspace(spec, data):
    frame, system, y, responses = _prepare(spec, data)
    optimizer = None
    if len(system.start):
        solution, cov = ml_optimize(
            lambda z: kalman(y, system.values(z))["log_likelihood"],
            system.start,
            frame.option("max_iterations"),
            frame.option("tolerance"),
        )
        theta = solution.theta
        if not solution.converged:
            raise AnalysisError("nonconvergence", "State-space likelihood did not converge; no inferential fit returned.")
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(system.physical, theta)
        covariance = J @ cov @ J.T
        params = system.physical(theta)
        optimizer = {
            "converged": solution.converged,
            "iterations": solution.iterations,
            "gradient_max": float(solution.gradient.abs().max()),
            "diagnostics": solution.diagnostics,
        }
    else:
        theta, params, covariance = (
            system.start,
            system.start,
            torch.empty((0, 0), dtype=torch.float64),
        )
    values = system.values(theta)
    with torch.no_grad():
        output = kalman(y, values, retain=True)
    frozen = {k: v.tolist() for k, v in values.items()}
    result = build_result(
        frame,
        terms=system.names,
        params=params,
        covariance=covariance,
        use_t=False,
        title="Linear Gaussian state space",
        metrics=information_criteria(float(output["log_likelihood"]), len(params), frame.n),
        fitted=output["predicted"][:, 0] if bool(torch.isfinite(y[:, 0]).all()) else None,
        solver="native Kalman prediction-error likelihood; Torch analytic derivatives",
        optimizer=optimizer,
        extra={
            "system": frozen,
            "system_base": {k: v.tolist() for k, v in system.fixed_values(theta).items()},
            "responses": responses,
            "initialization": system.initialization,
            "state_schema": "proper-gaussian-path-v1",
            "system_template": system.original_record,
            "observations": [[None if math.isnan(v) else v for v in row] for row in y.tolist()],
            "periods": _periods(frame),
            "parameter_values": params.tolist(),
            "parameter_covariance": covariance.tolist(),
            "missing_policy": frame.option("observation_missing"),
            **{k: v.tolist() if isinstance(v, torch.Tensor) else v
               for k, v in output.items() if k != "log_likelihood"},
        },
        inference={
            "available": bool(len(params)),
            "identification": "static state observability; scheduled proper priors may retain unobserved directions; positive definite observed information for free parameters",
            "covariance": "nonrobust",
            "forecast_parameter_uncertainty": False,
        },
        provenance={
            "likelihood": "exact Gaussian; known finite or exact stationary prior",
            "initial_prior": "a_1 before y_1",
            "supported_domain": "proper-prior masked observations; caller-declared schedules and known exogenous paths; independent process and measurement shocks; <=16 states, <=8 measurements",
        },
    )

    result.metrics["observed_measurements"] = float(torch.isfinite(y).sum())
    result.provenance["likelihood_normalization"] = "Gaussian density of observed cells only; all-missing periods contribute zero"
    result.provenance["transition_timing"] = "T[t],c[t],Q[t] propagate a[t] to a[t+1] after measurement y[t]"
    if bool(torch.isnan(y[:, 0]).any()):
        present = torch.isfinite(y[:, 0]).nonzero().flatten().tolist()
        result.predictions = [{"row": frame.positions[i], "observed": float(y[i, 0]),
                               "fitted": float(output["predicted"][i, 0]),
                               "residual": float(y[i, 0] - output["predicted"][i, 0])}
                              for i in present[:512]]
        result.provenance["prediction_sample"] = "first 512 observed first-response cells in calendar order"
    result.extra["state_sha256"] = _digest(result)
    return result


@_cpu_call
def sspace(
    *,
    data,
    y,
    system,
    responses=None,
    time=None,
    max_iterations=300,
    tolerance=1e-7,
    alpha=0.05,
    missing="raise",
):
    from openecon.econometrics.core import column_list

    spec = make_spec(
        "sspace",
        outcome=y,
        predictors=[],
        intercept=False,
        time=time,
        alpha=alpha,
        missing="raise" if missing == "mask" else missing,
        columns={"responses": column_list(responses, "responses")},
        options={"system": system, "max_iterations": max_iterations, "tolerance": tolerance,
                 "observation_missing": "mask" if missing == "mask" else "raise"},
    )
    return fit_sspace(spec, data)


@_cpu_call
def sspace_filter(*, data, y, system, responses=None, time=None, missing="raise"):
    """Filter a fully specified system without estimating its parameters."""
    if not isinstance(system, dict):
        raise AnalysisError("invalid_system", "sspace_filter requires a system dictionary.")
    if system.get("parameters"):
        raise AnalysisError(
            "invalid_system",
            "sspace_filter needs a fully specified system with no free parameters.",
        )
    result = sspace(data=data, y=y, system=system, responses=responses, time=time, missing=missing)
    states = torch.tensor(result.extra["filtered"], dtype=torch.float64)
    covs = torch.tensor(result.extra["filtered_covariance"], dtype=torch.float64)
    return table(
        {
            "row": result.sample_positions,
            **{f"state_{i + 1}": states[:, i].tolist() for i in range(states.shape[1])},
        },
        **_table_metadata(result),
        covariance=covs.tolist(),
        log_likelihood=result.metrics["log_likelihood"],
        next_mean=result.extra["next_mean"],
        next_covariance=result.extra["next_covariance"],
    )



def _periods(frame):
    if not frame.spec.time:
        return list(range(frame.n))
    return [value.isoformat() if hasattr(value, "isoformat") else value
            for value in frame.sample[frame.spec.time].tolist()]


def _digest(result):
    import hashlib
    import json
    payload = result.model_dump(mode="json")
    payload["extra"].pop("state_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _restore(result):
    from openecon.models import ResultBundle
    try:
        if isinstance(result, str):
            result = ResultBundle.model_validate_json(result)
        elif isinstance(result, dict):
            result = ResultBundle.model_validate(result)
        if not isinstance(result, ResultBundle) or result.spec.estimator != "sspace":
            raise ValueError("not a state-space result")
        if result.extra.get("state_schema") != "proper-gaussian-path-v1":
            raise ValueError("state schema unavailable; refit with the proper-prior path implementation")
        n, m, p = result.nobs, len(result.extra.get("next_mean", [])), len(result.extra.get("responses", []))
        if not 1 <= n <= 20000 or not 1 <= m <= 16 or not 1 <= p <= 8:
            raise ValueError("saved state geometry outside the supported domain")
        from openecon.econometrics.tsworkflows.ssmodel import _shape
        system = result.extra.get("system", {})
        shapes = {"a0": (m,), "P0": (m, m), "T": (m, m), "Z": (p, m),
                  "Q": (m, m), "H": (p, p), "c": (m,), "d": (p,)}
        if not isinstance(system, dict) or set(system) != set(shapes):
            raise ValueError("saved system fields do not match the state schema")
        for key, shape in shapes.items():
            try:
                _shape(system[key], shape, "saved " + key)
            except AnalysisError:
                if key in {"a0", "P0"}:
                    raise
                _shape(system[key], (n, *shape), "saved " + key)
        _shape(result.extra.get("observations"), (n, p), "saved observations")
        _shape(result.extra.get("observed_mask"), (n, p), "saved masks")
        if len(result.coefficients) > 20:
            raise ValueError("saved parameter geometry outside the supported domain")
        if result.extra.get("state_sha256") != _digest(result):
            raise ValueError("persisted state digest differs")
        if (len(result.sample_positions) != n or len(set(result.sample_positions)) != n
                or any(type(i) is not int or not 0 <= i < result.nobs_original for i in result.sample_positions)
                or len(result.extra.get("periods", [])) != n):
            raise ValueError("saved calendar and physical positions do not align")
        finite_tensor(result.extra["next_mean"], (m,), "next state mean")
        covariance = finite_tensor(result.extra["next_covariance"], (m, m), "next state covariance")
        pd_check(covariance, "next state covariance", semidefinite=True)
        observations = result.extra.get("observations", [])
        masks = result.extra.get("observed_mask", [])
        if (len(observations) != n or len(masks) != n
                or any(not isinstance(row, list) or len(row) != p for row in observations)
                or any(not isinstance(row, list) or len(row) != p or any(type(v) is not bool for v in row) for row in masks)
                or any(mask != (value is not None) for row, flags in zip(observations, masks, strict=True)
                       for value, mask in zip(row, flags, strict=True))):
            raise ValueError("saved observations and masks do not align")
    except (ValueError, TypeError) as exc:
        raise AnalysisError("invalid_result", "Invalid or changed saved state-space result: " + str(exc)) from exc
    return result


def _table_metadata(result):
    return {"periods": result.extra["periods"], "sample_positions": result.sample_positions,
            "responses": result.extra["responses"], "observed_mask": result.extra["observed_mask"],
            "missing_policy": result.extra["missing_policy"],
            "state_sha256": result.extra["state_sha256"],
            "uncertainty": "conditional on fixed system parameters and supplied exogenous paths; parameter uncertainty excluded",
            "stata_parity_validated": False}


def _smoothing(result):
    result = _restore(result)
    from openecon.resources import plan_workspace
    n, m, p = result.nobs, len(result.extra["next_mean"]), len(result.extra["responses"])
    if n > 20000 or n * (m**3 + p**3) > 50_000_000:
        raise AnalysisError("system_budget", "Smoothing state geometry exceeds the bounded work domain.")
    plan_workspace("full state-space smoothing and serialization", {
        "joint_moments_and_output_copies": n * (6*m*m + 4*p*p + 4*m*p + 8*m + 4*p) * 64})
    values = {k: finite_tensor(v, name=k) for k, v in result.extra["system"].items()}
    y = torch.tensor([[math.nan if v is None else v for v in row]
                      for row in result.extra["observations"]], dtype=torch.float64)
    with torch.no_grad():
        output = smooth(y, values)
    from openecon.econometrics.core import _json_safe
    return result, _json_safe({k: v.tolist() if isinstance(v, torch.Tensor) else v
                              for k, v in output.items() if k != "log_likelihood"})


@_cpu_call
def sspace_smooth(result):
    """Proper-prior fixed-interval conditional state means and full covariance."""
    result, output = _smoothing(result)
    states = output["smoothed"]
    return table({"row": result.sample_positions,
                  **{f"state_{i+1}": [row[i] for row in states] for i in range(len(states[0]))}},
                 **{**_table_metadata(result), **output}, covariance=output["smoothed_covariance"])


@_cpu_call
def sspace_autocov(result):
    """Lag-one smoothed covariance Cov(a[t+1],a[t]|observations), in calendar order."""
    result, output = _smoothing(result)
    lag = output["lag_one_covariance"]
    m = len(result.extra["next_mean"])
    return table({"row": result.sample_positions[:-1], "next_row": result.sample_positions[1:],
                  **{f"cov_{i+1}_{j+1}": [v[i][j] for v in lag]
                     for i in range(m) for j in range(m)}},
                 **{**_table_metadata(result), **output},
                 orientation="Cov(a[t+1],a[t]|all observed Y)", covariance=lag)


@_cpu_call
def sspace_disturbances(result):
    """Conditional process/measurement disturbances, including correlated missing noise."""
    result, output = _smoothing(result)
    u, e = output["process_disturbance"], output["measurement_disturbance"]
    return table({"row": result.sample_positions,
                  **{f"process_{i+1}": [v[i] for v in u] for i in range(len(u[0]))},
                  **{f"measurement_{i+1}": [v[i] for v in e] for i in range(len(e[0]))}},
                 **{**_table_metadata(result), **output},
                 covariance=output["joint_disturbance_covariance"],
                 disturbance_order="process followed by measurement",
                 final_process_disturbance="unobserved outgoing shock: prior zero mean and Q[last]")


@_cpu_call
def forecast(result, steps, *, alpha=None, future=None):
    """Conditional forecasts from saved state; explicit paths required for scheduled/exogenous systems."""
    horizon(steps)
    if result.spec.estimator != "sspace":
        raise AnalysisError("invalid_result", "State-space forecast needs an sspace result.")
    record = result.extra
    # Legacy static fits remain forecastable under their original complete-observation contract.
    if record.get("state_schema"):
        result = _restore(result)
    if future is not None and (not isinstance(future, dict) or set(future) - {"schedules", "exogenous"}):
        raise AnalysisError("invalid_future", "future accepts schedules and exogenous paths only.")
    from openecon.resources import plan_workspace
    from openecon.econometrics.tsworkflows.ssmodel import future_values
    m, p = len(record.get("next_mean", [])), len(record.get("responses", []))
    if steps * (m**3 + p**3 + m*m*p + m*p*p) * 2 > 50_000_000:
        raise AnalysisError("system_budget", "Full forecast matrix work exceeds 50 million units.")
    plan_workspace("state-space full forecast moments", {
        "moment_output_and_copies": steps * (m*m + m + p*p + p) * 64})
    if record.get("state_schema"):
        values = future_values(record["system_template"], record["system_base"], steps,
                               schedules=(future or {}).get("schedules"),
                               exog=(future or {}).get("exogenous"))
    else:
        if future:
            raise AnalysisError("invalid_future", "Refit legacy results before supplying future schedules.")
        values = {k: finite_tensor(v, name=k) for k, v in record["system"].items()}
    a, P = finite_tensor(record["next_mean"]), finite_tensor(record["next_covariance"])
    pd_check(P, "forecast-origin state covariance", semidefinite=True)
    means, vars, states, state_covs, measurement_covs = [], [], [], [], []
    for t in range(steps):
        Z, T, H, Q, c, d = (at(values, k, t) for k in ("Z", "T", "H", "Q", "c", "d"))
        mean, variance = Z @ a + d, Z @ P @ Z.T + H
        if not all(bool(torch.isfinite(v).all()) for v in (mean, variance, a, P)):
            raise AnalysisError("non_finite_forecast", "Full forecast moments overflowed; no partial result returned.")
        means.append(mean)
        vars.append(variance.diagonal())
        states.append(a)
        state_covs.append(P)
        measurement_covs.append(variance)
        a, P = T @ a + c, T @ P @ T.T + Q
        P = (P + P.T) / 2
    means, vars = torch.stack(means), torch.stack(vars)
    alpha = result.spec.alpha if alpha is None else alpha
    return forecast_table(means[:, 0], vars[:, 0], alpha, origin=result.nobs,
                          origin_period=record.get("periods", [result.nobs])[-1],
                          origin_physical_row=result.sample_positions[-1],
                          responses=record["responses"],
                          state_mean=torch.stack(states).tolist(),
                          state_covariance=torch.stack(state_covs).tolist(),
                          measurement_mean=means.tolist(),
                          measurement_covariance=torch.stack(measurement_covs).tolist(),
                          future_paths={k: v.tolist() for k, v in values.items()
                                        if k in {"Z", "T", "Q", "H", "c", "d"}},
                          uncertainty="conditional state and future process/measurement uncertainty; parameter uncertainty excluded")
