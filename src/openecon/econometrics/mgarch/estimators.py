"""Joint Gaussian CCC, DCC and full BEKK APIs and persisted covariance forecasts."""

from __future__ import annotations

import math

import torch
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame,
    build_result,
    column_list,
    kernel_call,
    make_spec,
    table,
)
from openecon.models import ResultBundle
from openecon.resources import plan_workspace

from . import kernels


def _estimate(
    kind,
    data,
    series,
    *,
    time=None,
    intercept=True,
    missing="raise",
    alpha=0.05,
    max_n=512,
    max_iterations=300,
    tolerance=1e-7,
    initial_covariance=None,
):
    series = column_list(series, "series")
    if len(series) < 2 or len(series) != len(set(series)):
        raise AnalysisError(
            "invalid_spec", "MGARCH requires two or three distinct numeric outcome series."
        )
    spec = make_spec(
        "mgarch_" + kind,
        outcome=series[0],
        columns={"series": series[1:]},
        time=time,
        intercept=intercept,
        missing=missing,
        alpha=alpha,
        options={
            "max_n": max_n,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "initial_covariance": initial_covariance,
        },
    )
    return fit_mgarch(spec, data)


def mgarch_ccc(data, series, **options):
    """Joint Gaussian CCC-GARCH(1,1), dimension2..3; constant or zero means."""
    return _estimate("ccc", data, series, **options)


def mgarch_dcc(data, series, **options):
    """Joint Gaussian DCC-GARCH(1,1), dimension2..3; estimated SPD target."""
    return _estimate("dcc", data, series, **options)


def mgarch_bekk(data, series, **options):
    """Joint Gaussian full BEKK(1,1), dimension2; full unrestricted A/B entries."""
    return _estimate("bekk", data, series, **options)


def _terms(kind, series, mean):
    terms = ["mean:" + name for name in series] if mean else []
    d = len(series)
    if kind == "bekk":
        terms += [f"C:{series[i]}:{series[j]}" for i, j in kernels.lower_pairs(d)]
        terms += [
            f"{matrix}:{series[i]}:{series[j]}"
            for matrix in ("A", "B")
            for i in range(d)
            for j in range(d)
        ]
    else:
        terms += [
            f"{parameter}:{name}" for name in series for parameter in ("omega", "alpha", "beta")
        ]
        terms += [f"corr:{series[i]}:{series[j]}" for i, j in kernels.corr_pairs(d)]
        if kind == "dcc":
            terms += ["dcc_alpha", "dcc_beta"]
    return terms


def fit_mgarch(spec, data):
    frame = ModelFrame(spec, data)
    series = [spec.outcome, *frame.role("series")]
    kind = spec.estimator.removeprefix("mgarch_")
    d = len(series)
    if len(set(series)) != d or d < 2 or d > (2 if kind == "bekk" else 3):
        raise AnalysisError(
            "mgarch_dimension", "Supported MGARCH dimensions are CCC/DCC 2..3 and full BEKK 2."
        )
    if spec.time is not None:
        if is_datetime64_any_dtype(frame.series(spec.time).dtype) or is_bool_dtype(
            frame.series(spec.time).dtype
        ):
            raise AnalysisError(
                "invalid_time",
                "MGARCH time must contain integer periods; convert dates to an explicit period index.",
            )
        frame.sort_panel()
        times = frame.time_index()
        if bool((times[1:] - times[:-1] != 1).any()):
            raise AnalysisError(
                "mgarch_time_gaps",
                "MGARCH requires consecutive integer periods after missing filtering; no gap compression is performed.",
            )
    elif frame.dropped_missing and frame.positions != list(
        range(frame.positions[0], frame.positions[-1] + 1)
    ):
        raise AnalysisError(
            "mgarch_time_gaps",
            "MGARCH missing='drop' can trim a contiguous sample but cannot compress interior missing periods.",
        )
    if frame.n > frame.option("max_n"):
        raise AnalysisError(
            "mgarch_sample_limit",
            f"MGARCH max_n={frame.option('max_n')} guards the sequential joint information geometry; supplied {frame.n}.",
        )
    p = kernels.parameter_count(kind, d, spec.intercept)
    if frame.n < max(30, 3 * p):
        raise AnalysisError(
            "insufficient_observations",
            "MGARCH requires at least max(30,3*parameter_count) complete consecutive periods.",
        )
    frame.workspace_plan(
        "MGARCH joint likelihood autodiff and full information",
        {
            "covariance_paths_recursion_and_derivative_graph": 8
            * frame.n
            * d
            * d
            * (32 * p + 16 * p * p),
            "parameter_information_and_jacobian": 8 * 24 * p * p,
            "outcomes_and_residuals": 8 * 8 * frame.n * d,
        },
    )
    y = frame.matrix(series)
    scale = y.std(dim=0, correction=0)
    if (
        not bool(torch.isfinite(scale).all())
        or bool((scale <= 1e-10).any())
        or float(scale.max() / scale.min()) > 1e4
    ):
        raise AnalysisError(
            "mgarch_numeric_domain",
            "Series must vary on finite comparable scales; explicitly rescale near-constant or extremely unequal series.",
        )
    initial_payload = frame.option("initial_covariance")
    if initial_payload is None:
        initial_residuals = y - (y.mean(dim=0) if spec.intercept else 0)
        backcast = initial_residuals[: min(75, frame.n)]
        initial = backcast.T @ backcast / len(backcast)
    else:
        try:
            initial = torch.tensor(initial_payload, dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as error:
            raise AnalysisError(
                "mgarch_invalid_covariance", "initial_covariance must be a numeric square matrix."
            ) from error
        if initial.shape != (d, d):
            raise AnalysisError(
                "mgarch_invalid_covariance",
                "initial_covariance must have one row and column per outcome series.",
            )
    kernel_call(kernels.check_spd, initial, "presample backcast covariance")
    params, covariance, ll, path, correlation, state, diagnostics = kernel_call(
        kernels.fit_joint,
        y,
        initial,
        kind,
        spec.intercept,
        max_iterations=frame.option("max_iterations"),
        tolerance=frame.option("tolerance"),
    )
    matrices = kernels.physical_matrices(params, kind, d, spec.intercept)
    extra = {
        "model": {
            "kind": kind,
            "order": [1, 1],
            "dimension": d,
            "distribution": "Gaussian",
            "series": series,
            "constant_mean": spec.intercept,
            "joint_estimation": True,
            "target_definition": "estimated unit-diagonal SPD Q target; not fixed empirical standardized-residual correlation"
            if kind == "dcc"
            else None,
        },
        "initialization": {
            "method": "user fixed covariance"
            if initial_payload is not None
            else "fixed equal-weight covariance of first min(75,N) residuals using initial sample mean or zero",
            "initial_covariance": initial.tolist(),
            "initial_covariance_conditioned_on": True,
            "first_observation": "uses initial H directly (BEKK), initial variances and model correlation target (CCC/DCC)",
        },
        "conditional_covariance": path.tolist(),
        "conditional_correlation": correlation.tolist(),
        "state": {key: value.tolist() for key, value in state.items()},
        "parameter_matrices": {key: value.tolist() for key, value in matrices.items()},
        "forecast_contract": "one-step full covariance exact for all models; BEKK full multi-step covariance exact; CCC marginal variance expectations exact but multi-step off-diagonal covariance plug-in; DCC multi-step Q-forward and variance plug-in approximation",
        "constraints": diagnostics["constraints"],
    }
    terms = _terms(kind, series, spec.intercept)
    mean = matrices["mu"][0].expand(frame.n)
    return build_result(
        frame,
        terms=terms,
        params=params,
        covariance=covariance,
        fitted=mean,
        df_resid=frame.n - p,
        metrics={
            "log_likelihood": ll,
            "aic": 2 * p - 2 * ll,
            "bic": math.log(frame.n) * p - 2 * ll,
        },
        solver="joint Gaussian MGARCH BFGS with native differentiated matrix recursion",
        solver_diagnostics=diagnostics,
        optimizer={"converged": True, "iterations": diagnostics["iterations"]},
        inference={
            "correction": "full joint observed information including all nuisance blocks; Gaussian assumption",
            "residual_definition": "outcome minus constant/zero conditional mean; first series chart only",
        },
        provenance={
            "mgarch_execution_domain": "CPU float64, consecutive periods, CCC/DCC 2..3 or full BEKK 2, Gaussian order (1,1), constant/zero mean; no Dataset route",
            "stationarity": "positive omega and alpha+beta<1-1e-6; DCC a+b<1-1e-6; full BEKK Kronecker spectral radius<1-1e-6",
        },
        extra=extra,
    )


def mgarch_forecast(result, steps):
    if not isinstance(result, ResultBundle) or result.spec.estimator not in {
        "mgarch_ccc",
        "mgarch_dcc",
        "mgarch_bekk",
    }:
        raise AnalysisError(
            "invalid_result",
            "MGARCH forecast requires a fitted CCC, DCC or full BEKK ResultBundle.",
        )
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise AnalysisError("invalid_forecast", "MGARCH forecast steps must be an integer1..1000.")
    model = result.extra.get("model", {})
    kind = model.get("kind")
    series = model.get("series")
    if (
        kind not in {"ccc", "dcc", "bekk"}
        or "mgarch_" + kind != result.spec.estimator
        or not isinstance(series, list)
        or len(series) != model.get("dimension")
    ):
        raise AnalysisError("invalid_result", "The persisted MGARCH model identity is invalid.")
    d = len(series)
    if (
        d < 2
        or d > (2 if kind == "bekk" else 3)
        or model.get("constant_mean") != result.spec.intercept
    ):
        raise AnalysisError(
            "invalid_result",
            "Persisted MGARCH dimension/mean identity is outside the supported domain.",
        )
    plan = plan_workspace(
        "MGARCH full matrix covariance forecasts",
        {"paths_and_correlations": 8 * 12 * steps * d * d},
    )
    try:
        coefficients = torch.tensor(
            [coefficient.estimate for coefficient in result.coefficients], dtype=torch.float64
        )
        if len(coefficients) != kernels.parameter_count(kind, d, result.spec.intercept):
            raise ValueError("bad parameter count")
        m = kernels.physical_matrices(coefficients, kind, d, result.spec.intercept)
        stored = result.extra["parameter_matrices"]
        if set(stored) != set(m) or any(
            not torch.allclose(
                m[key], torch.tensor(stored[key], dtype=torch.float64), atol=1e-12, rtol=1e-12
            )
            for key in m
        ):
            raise ValueError("inconsistent persisted parameters")
        state = {
            key: torch.tensor(value, dtype=torch.float64)
            for key, value in result.extra["state"].items()
        }
        if state["H"].shape != (d, d) or state["last_residual"].shape != (d,):
            raise ValueError("bad state dimensions")
        if kind != "bekk" and (state["h"].shape != (d,) or state["Q"].shape != (d, d)):
            raise ValueError("bad correlation state dimensions")
        path = kernel_call(kernels.forecast_path, m, state, kind, steps)
    except AnalysisError:
        raise
    except (KeyError, ValueError, TypeError, RuntimeError) as error:
        raise AnalysisError(
            "invalid_result", "The persisted MGARCH forecast parameters/state are invalid."
        ) from error
    correlation = (
        path
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, :, None]
        / path.diagonal(dim1=1, dim2=2).sqrt()[:, None, :]
    )
    rows = []
    for step in range(steps):
        record = {"step": step + 1}
        for i in range(d):
            for j in range(i + 1):
                record[f"cov:{series[i]}:{series[j]}"] = float(path[step, i, j])
                record[f"corr:{series[i]}:{series[j]}"] = float(correlation[step, i, j])
        rows.append(record)
    method = {
        "bekk": "exact full conditional covariance expectation recursion",
        "ccc": "one-step full covariance exact; marginal variances exact; beyond one-step off-diagonal fixed-correlation plug-in approximation",
        "dcc": "one-step full covariance exact; beyond one-step Engle-Sheppard section 7 Q-forward and variance plug-in approximation",
    }[kind]
    return table(
        rows,
        model=kind,
        conditional_covariance=path.tolist(),
        conditional_correlation=correlation.tolist(),
        resource_plan=plan.record(),
        forecast_method=method,
        publication_notes=[
            method,
            "Forecasts condition on fitted parameters; parameter uncertainty is excluded.",
        ],
        exact_full_covariance_horizons=list(range(1, steps + 1)) if kind == "bekk" else [1],
        approximate_full_covariance_horizons=[] if kind == "bekk" else list(range(2, steps + 1)),
        exact_marginal_variance_horizons=list(range(1, steps + 1)),
        parameter_uncertainty_included=False,
    )
