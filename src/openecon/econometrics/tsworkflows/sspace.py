"""Time-invariant linear Gaussian systems with proper/known initialization.

y_t = Z a_t + d + e_t, e_t~N(0,H);
a_{t+1} = T a_t + c + u_t, u_t~N(0,Q), independent of e_t.
The initial prior is for a_1, before observing y_1. No covariance projection,
diffuse approximation, or silently dropped observation period is used.
"""

from __future__ import annotations
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


class System:
    def __init__(self, record: dict, p: int, frame=None):
        if not isinstance(record, dict) or set(record) - {
            "Z",
            "T",
            "Q",
            "H",
            "a0",
            "P0",
            "c",
            "d",
            "initialization",
            "parameters",
        }:
            raise AnalysisError("invalid_system", "Unknown or missing state-space system fields.")
        raw = record.get("T")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 16 or not 1 <= p <= 8:
            raise AnalysisError(
                "system_budget", "Supported domain: 1..16 states and 1..8 measurements."
            )
        self.m, self.p = len(raw), p
        m = self.m
        self.initialization = record.get("initialization", "known")
        if self.initialization not in {"known", "stationary"}:
            raise AnalysisError(
                "invalid_initialization",
                "Use known finite prior or exact stationary initialization; diffuse priors are unsupported.",
            )
        self.base = {
            key: finite_tensor(record.get(key), shape, key)
            for key, shape in (("Z", (p, m)), ("T", (m, m)), ("Q", (m, m)), ("H", (p, p)))
        }
        self.base.update(
            a0=finite_tensor(record.get("a0", [0.0] * m), (m,), "a0"),
            P0=finite_tensor(record.get("P0", torch.eye(m).tolist()), (m, m), "P0"),
            c=finite_tensor(record.get("c", [0.0] * m), (m,), "c"),
            d=finite_tensor(record.get("d", [0.0] * p), (p,), "d"),
        )
        self.parameters = record.get("parameters", [])
        if not isinstance(self.parameters, list) or len(self.parameters) > 20:
            raise AnalysisError(
                "system_budget", "At most20 individually identified free parameters are supported."
            )
        used, names, start = set(), set(), []
        for item in self.parameters:
            if not isinstance(item, dict) or set(item) - {
                "name",
                "matrix",
                "row",
                "col",
                "transform",
            }:
                raise AnalysisError(
                    "invalid_parameter",
                    "Free parameter needs name/matrix/row/col/transform fields.",
                )
            key, row, col = item.get("matrix"), item.get("row"), item.get("col")
            name, transform = item.get("name"), item.get("transform", "identity")
            if key not in self.base or not isinstance(name, str) or not name or name in names:
                raise AnalysisError(
                    "invalid_parameter", "Parameter names must be unique and matrices recognized."
                )
            shape = self.base[key].shape
            if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < shape[0]:
                raise AnalysisError(
                    "invalid_parameter", "Parameter row is outside matrix dimensions."
                )
            if len(shape) == 2:
                if isinstance(col, bool) or not isinstance(col, int) or not 0 <= col < shape[1]:
                    raise AnalysisError("invalid_parameter", "Matrix parameter needs a valid col.")
            elif col is not None:
                raise AnalysisError("invalid_parameter", "Vector parameter must omit col.")
            target = (
                (key, min(row, col), max(row, col)) if key in {"Q", "H", "P0"} else (key, row, col)
            )
            if target in used or (self.initialization == "stationary" and key in {"a0", "P0"}):
                raise AnalysisError(
                    "invalid_parameter",
                    "Duplicate parameter cells or free stationary prior are not identified.",
                )
            value = float(self.base[key][row] if col is None else self.base[key][row, col])
            if transform == "positive" and value > 0:
                value = math.log(value)
            elif transform == "unit" and -1 < value < 1:
                value = math.atanh(value)
            elif transform != "identity":
                raise AnalysisError(
                    "invalid_parameter",
                    "Transform identity/positive/unit must match its starting value.",
                )
            start.append(value)
            names.add(name)
            used.add(target)
        self.start = torch.tensor(start, dtype=torch.float64)
        self.names = [v["name"] for v in self.parameters]
        self.lyapunov_bytes = (
            m**4 * 8 * (12 + 4 * len(start)) if self.initialization == "stationary" else 0
        )
        self.lyapunov_work = (
            m**6 * max(1, len(start)) ** 2 if self.initialization == "stationary" else 0
        )
        if self.lyapunov_work > 50_000_000:
            raise AnalysisError(
                "system_budget",
                "Stationary Lyapunov solve/information geometry exceeds the supported operation budget.",
            )
        if self.lyapunov_bytes:
            buffers = {"Lyapunov_operator_factors_and_derivatives": self.lyapunov_bytes}
            if frame is None:
                from openecon.resources import plan_workspace

                plan_workspace("stationary system initialization", buffers)
            else:
                frame.workspace_plan("stationary system initialization", buffers)
        self.values(self.start)

    def physical(self, theta):
        return (
            torch.stack(
                [
                    torch.exp(theta[i])
                    if v.get("transform") == "positive"
                    else torch.tanh(theta[i])
                    if v.get("transform") == "unit"
                    else theta[i]
                    for i, v in enumerate(self.parameters)
                ]
            )
            if len(theta)
            else theta
        )

    def values(self, theta):
        values = {k: v.clone() for k, v in self.base.items()}
        physical = self.physical(theta)
        for i, item in enumerate(self.parameters):
            key, row, col = item["matrix"], item["row"], item.get("col")
            if col is None:
                values[key][row] = physical[i]
            else:
                values[key][row, col] = physical[i]
                if key in {"Q", "H", "P0"}:
                    values[key][col, row] = physical[i]
        for key in ("Q", "H", "P0"):
            pd_check(values[key], key, semidefinite=key != "H")
        T = values["T"]
        radius = float(torch.linalg.eigvals(T.detach()).abs().max())
        if (
            not math.isfinite(radius)
            or radius > 1 + 1e-12
            or (self.initialization == "stationary" and radius >= 1 - 1e-10)
        ):
            raise AnalysisError(
                "unstable_system",
                "T must have spectral radius <=1 (strictly<1 for stationary initialization).",
            )
        observation = torch.cat(
            [values["Z"] @ torch.linalg.matrix_power(T, i) for i in range(self.m)]
        )
        if int(torch.linalg.matrix_rank(observation.detach())) < self.m:
            raise AnalysisError(
                "unidentified_system",
                "The state system is not observable; fix or remove unidentified states.",
            )
        if self.initialization == "stationary":
            identity = torch.eye(self.m, dtype=torch.float64)
            values["a0"] = torch.linalg.solve(identity - T, values["c"])
            lyap = torch.eye(self.m**2, dtype=torch.float64) - torch.kron(
                T.contiguous(), T.contiguous()
            )
            P = torch.linalg.solve(lyap, values["Q"].reshape(-1)).reshape(self.m, self.m)
            values["P0"] = (P + P.T) / 2
            pd_check(values["P0"], "stationary P0", semidefinite=True)
        return values


def kalman(y, values, *, retain=False):
    Z, T, Q, H, a, P, c, d = (values[k] for k in ("Z", "T", "Q", "H", "a0", "P0", "c", "d"))
    identity = torch.eye(len(a), dtype=torch.float64)
    contributions, predicted, filtered, covs, innovation_covs = [], [], [], [], []
    for observation in y:
        mean = Z @ a + d
        error = observation - mean
        F = Z @ P @ Z.T + H
        if not all(bool(torch.isfinite(v).all()) for v in (mean, error, F)):
            raise AnalysisError(
                "non_finite_likelihood",
                "Full measurement likelihood moments overflowed; rescale the system.",
            )
        chol, status = torch.linalg.cholesky_ex(F)
        if int(status):
            raise AnalysisError(
                "invalid_covariance",
                "Innovation covariance is not positive definite; no projection applied.",
            )
        solve = torch.cholesky_solve(error[:, None], chol).flatten()
        contributions.append(
            -0.5
            * (len(error) * math.log(2 * math.pi) + 2 * chol.diagonal().log().sum() + error @ solve)
        )
        K = torch.cholesky_solve(Z @ P, chol).T
        a = a + K @ error
        # Joseph form preserves the mathematical covariance without eigenvalue clipping.
        update = identity - K @ Z
        P = update @ P @ update.T + K @ H @ K.T
        P = (P + P.T) / 2
        if not all(bool(torch.isfinite(v).all()) for v in (a, P, contributions[-1])):
            raise AnalysisError(
                "non_finite_likelihood",
                "Filtered state or likelihood overflowed; no partial/sanitized model is returned.",
            )
        if retain:
            predicted.append(mean)
            filtered.append(a)
            covs.append(P)
            innovation_covs.append(F)
        a = T @ a + c
        P = T @ P @ T.T + Q
        P = (P + P.T) / 2
        if not all(bool(torch.isfinite(v).all()) for v in (a, P)):
            raise AnalysisError(
                "non_finite_likelihood", "Next-period state moments overflowed; rescale the system."
            )
    output = {
        "log_likelihood": torch.stack(contributions).sum(),
        "next_mean": a,
        "next_covariance": P,
    }
    if retain:
        output.update(
            predicted=torch.stack(predicted),
            filtered=torch.stack(filtered),
            filtered_covariance=torch.stack(covs),
            innovation_covariance=torch.stack(innovation_covs),
        )
    return output


def _prepare(spec, data):
    frame = ModelFrame(spec, data)
    ordered(frame)
    from openecon.econometrics.registry import role_columns

    responses = list(dict.fromkeys([spec.outcome, *role_columns(spec, "responses")]))
    system = System(frame.option("system"), len(responses), frame=frame)
    if (
        frame.n > 20000
        or system.lyapunov_work
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
            "retained_moments": frame.n * (system.m**2 + system.m + system.p**2 + system.p) * 8,
        },
    )
    y = torch.stack([frame.numeric(name) for name in responses], dim=1)
    return frame, system, y, responses


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
    return build_result(
        frame,
        terms=system.names,
        params=params,
        covariance=covariance,
        use_t=False,
        title="Linear Gaussian state space",
        metrics=information_criteria(float(output["log_likelihood"]), len(params), frame.n),
        fitted=output["predicted"][:, 0],
        solver="native Kalman prediction-error likelihood; Torch analytic derivatives",
        optimizer=optimizer,
        extra={
            "system": frozen,
            "responses": responses,
            "initialization": system.initialization,
            **{k: v.tolist() for k, v in output.items() if k != "log_likelihood"},
        },
        inference={
            "available": bool(len(params)),
            "identification": "observable states and positive definite observed information for free parameters",
            "covariance": "nonrobust",
            "forecast_parameter_uncertainty": False,
        },
        provenance={
            "likelihood": "exact Gaussian; known finite or exact stationary prior",
            "initial_prior": "a_1 before y_1",
            "supported_domain": "time-invariant complete observations; independent process and measurement shocks; <=16 states, <=8 measurements",
        },
    )


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
        missing=missing,
        columns={"responses": column_list(responses, "responses")},
        options={"system": system, "max_iterations": max_iterations, "tolerance": tolerance},
    )
    return fit_sspace(spec, data)


def sspace_filter(*, data, y, system, responses=None, time=None):
    """Filter a fully specified system without estimating its parameters."""
    if not isinstance(system, dict):
        raise AnalysisError("invalid_system", "sspace_filter requires a system dictionary.")
    if system.get("parameters"):
        raise AnalysisError(
            "invalid_system",
            "sspace_filter needs a fully specified system with no free parameters.",
        )
    result = sspace(data=data, y=y, system=system, responses=responses, time=time)
    states = torch.tensor(result.extra["filtered"], dtype=torch.float64)
    covs = torch.tensor(result.extra["filtered_covariance"], dtype=torch.float64)
    return table(
        {
            "row": result.sample_positions,
            **{f"state_{i + 1}": states[:, i].tolist() for i in range(states.shape[1])},
        },
        covariance=covs.tolist(),
        log_likelihood=result.metrics["log_likelihood"],
        next_mean=result.extra["next_mean"],
        next_covariance=result.extra["next_covariance"],
    )


def forecast(result, steps, *, alpha=None):
    horizon(steps)
    if result.spec.estimator != "sspace":
        raise AnalysisError("invalid_result", "State-space forecast needs an sspace result.")
    record = result.extra
    from openecon.resources import plan_workspace

    states = len(record.get("next_mean", []))
    measurements = len(record.get("responses", []))
    plan_workspace(
        "state-space full forecast moments",
        {
            "moment_output_and_copies": steps
            * (states**2 + states + measurements**2 + measurements)
            * 32
        },
    )
    values = {k: finite_tensor(v, name=k) for k, v in record["system"].items()}
    a, P = finite_tensor(record["next_mean"]), finite_tensor(record["next_covariance"])
    Z, T, H, Q, c, d = (values[k] for k in ("Z", "T", "H", "Q", "c", "d"))
    means, vars, states, state_covs, measurement_covs = [], [], [], [], []
    for _ in range(steps):
        mean, variance = Z @ a + d, Z @ P @ Z.T + H
        if not all(bool(torch.isfinite(v).all()) for v in (mean, variance, a, P)):
            raise AnalysisError(
                "non_finite_forecast",
                "A full state/measurement forecast moment overflowed; no partial first-response forecast is returned.",
            )
        means.append(mean)
        vars.append(variance.diagonal())
        states.append(a)
        state_covs.append(P)
        measurement_covs.append(variance)
        a, P = T @ a + c, T @ P @ T.T + Q
    means, vars = torch.stack(means), torch.stack(vars)
    alpha = result.spec.alpha if alpha is None else alpha
    first = forecast_table(
        means[:, 0],
        vars[:, 0],
        alpha,
        origin=result.nobs,
        responses=record["responses"],
        state_mean=torch.stack(states).tolist(),
        state_covariance=torch.stack(state_covs).tolist(),
        measurement_mean=means.tolist(),
        measurement_covariance=torch.stack(measurement_covs).tolist(),
    )
    return first
