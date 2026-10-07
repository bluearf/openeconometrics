"""Bounded float64 likelihood tools shared only by these workflows."""

from __future__ import annotations
import math
import torch
from openecon.analysis_contracts import AnalysisError
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_bfgs, information_inverse
from openecon.econometrics.core import table
from openecon.engines.distributions import normal_isf


def ordered(frame):
    import pandas as pd

    if frame.spec.time:
        raw = frame.original[frame.spec.time]
        if pd.api.types.is_bool_dtype(raw) or pd.api.types.is_complex_dtype(raw):
            raise AnalysisError("invalid_time", "Time must be real numeric periods or datetimes.")
        if not pd.api.types.is_numeric_dtype(raw):
            try:
                dates = pd.to_datetime(raw, errors="raise")
            except (ValueError, TypeError) as exc:
                raise AnalysisError(
                    "invalid_time", "Time must be valid, consistently zoned datetimes."
                ) from exc
            if dates.isna().any():
                raise AnalysisError("invalid_time", "Time timestamps must be complete.")
            frame.original[frame.spec.time] = dates
            frame._sample = None
    if frame.spec.time or frame.spec.panel:
        frame.sort_panel()
    if frame.spec.time:
        t = frame.sample[frame.spec.time]
        from openecon.econometrics.arima.estimators import _gap_error

        if pd.api.types.is_numeric_dtype(t):
            d = t.diff().iloc[1:]
            if len(d) and (not bool((d > 0).all()) or not bool((d == d.iloc[0]).all())):
                raise _gap_error("Workflow time must have constant positive spacing.")
        else:
            values = pd.to_datetime(t, errors="coerce")
            if values.isna().any():
                raise AnalysisError(
                    "invalid_time", "Time must be regular numeric periods or datetimes."
                )
            if len(values) >= 3 and pd.infer_freq(pd.DatetimeIndex(values)) is None:
                raise _gap_error(
                    "Workflow datetimes must have an inferable regular calendar frequency."
                )
        if frame.dropped_missing:
            order = frame.original[frame.spec.time].sort_values(kind="stable").index.tolist()
            kept = set(frame.positions)
            indices = [i for i, pos in enumerate(order) if pos in kept]
            if indices != list(range(indices[0], indices[-1] + 1)):
                raise AnalysisError(
                    "time_gaps", "Excluded observations create an interior calendar gap."
                )
    if frame.dropped_missing:
        positions = frame.positions
        if (
            positions != list(range(positions[0], positions[0] + len(positions)))
            and frame.spec.time is None
        ):
            raise AnalysisError("time_gaps", "Dropping interior periods is not permitted.")


def finite_tensor(value, shape=None, name="matrix"):
    try:
        t = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_system", f"{name} must contain finite numbers.") from exc
    if (shape is not None and t.shape != shape) or not bool(torch.isfinite(t).all()):
        raise AnalysisError("invalid_system", f"{name} has invalid dimensions or nonfinite values.")
    return t


def pd_check(matrix, name, semidefinite=False):
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not torch.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
    ):
        raise AnalysisError("invalid_system", f"{name} must be symmetric square.")
    eigen = torch.linalg.eigvalsh(matrix.detach())
    if float(eigen.min()) < -1e-12 if semidefinite else float(eigen.min()) <= 0:
        raise AnalysisError(
            "invalid_covariance",
            f"{name} must be {'positive semidefinite' if semidefinite else 'positive definite'}; no projection is applied.",
        )


def ml_optimize(loglike, start, max_iterations, tolerance):
    """Torch analytic autodiff score/information of the exact caller likelihood."""

    def vg(theta):
        with torch.enable_grad():
            z = theta.detach().clone().requires_grad_(True)
            try:
                value = loglike(z)
                if not bool(torch.isfinite(value)):
                    return torch.tensor(-math.inf, dtype=torch.float64), torch.full_like(
                        z, math.nan
                    )
                gradient = torch.autograd.grad(value, z)[0]
            except (AnalysisError, RuntimeError):
                return torch.tensor(-math.inf, dtype=torch.float64), torch.full_like(z, math.nan)
            return value.detach(), gradient.detach()

    def hessian(theta):
        with torch.enable_grad():
            return torch.autograd.functional.hessian(loglike, theta.detach()).detach()

    try:
        fit = maximize_bfgs(
            vg,
            start,
            max_iter=max_iterations,
            gradient_tol=tolerance,
            hessian_fn=hessian,
            scaled_gradient_tol=max(tolerance**2, 1e-9),
        )
        covariance = information_inverse(-fit.hessian)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    return fit, covariance


def horizon(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10000:
        raise AnalysisError("invalid_horizon", "steps must be an integer in 1..10000.")


def forecast_table(means, variances, alpha, **attrs):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_alpha", "alpha must lie strictly between zero and one.")
    critical = normal_isf(alpha / 2)
    if (
        not bool(torch.isfinite(means).all())
        or not bool(torch.isfinite(variances).all())
        or bool((variances < 0).any())
    ):
        raise AnalysisError("non_finite_forecast", "Forecast moments are invalid.")
    errors = variances.sqrt()
    attrs.setdefault(
        "uncertainty",
        "conditional state and future process/measurement uncertainty; parameter uncertainty excluded",
    )
    attrs["publication_notes"] = [
        attrs["uncertainty"],
        *([attrs["initial_state_uncertainty"]] if "initial_state_uncertainty" in attrs else []),
    ]
    return table(
        {
            "period": list(range(1, len(means) + 1)),
            "forecast": means.tolist(),
            "std_error": errors.tolist(),
            "ci_low": (means - critical * errors).tolist(),
            "ci_high": (means + critical * errors).tolist(),
        },
        alpha=alpha,
        **attrs,
    )
