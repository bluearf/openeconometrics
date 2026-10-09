"""One bounded CPU float64 Kalman/RTS engine for proper Gaussian priors.

The prior at date t is for a[t] before observing y[t]. T[t] maps a[t] to
a[t+1]; the final outgoing transition is retained. NaN measurements are
unobserved, including all-missing dates, which still perform their transition.
No diffuse initialization, jitter, covariance clipping or row deletion occurs.
"""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace

FLOAT = torch.float64
MAX_WORK = 50_000_000
_EPSILON = torch.finfo(FLOAT).eps


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError(
            "non_finite_likelihood", f"{name} overflowed; rescale the state-space system."
        )
    return value


def _symmetrize(value):
    return value * 0.5 + value.transpose(-1, -2) * 0.5


def _covariance(value, name):
    _finite(value, name)
    scale = value.detach().abs().amax(dim=(-1, -2)).clamp_min(torch.finfo(FLOAT).tiny)
    tolerance = 64 * _EPSILON * value.shape[-1] * scale
    difference = (value - value.transpose(-1, -2)).detach().abs().amax(dim=(-1, -2))
    if bool((difference > tolerance).any()):
        raise AnalysisError("invalid_covariance", f"{name} must be symmetric.")
    try:
        eigenvalues = torch.linalg.eigvalsh(_symmetrize(value).detach())
    except RuntimeError as error:
        raise AnalysisError(
            "invalid_covariance", f"{name} exceeded finite covariance factorization precision."
        ) from error
    _finite(eigenvalues, f"{name} eigenvalues")
    smallest = eigenvalues[..., 0]
    if bool((smallest < -tolerance).any()):
        raise AnalysisError(
            "invalid_covariance", f"{name} must be positive semidefinite; no projection applied."
        )
    return value


def at(values, key, date):
    """Resolve a fixed matrix/vector or a schedule with one entry per input date."""
    value = values[key]
    dimension = 1 if key in {"c", "d"} else 2
    return value[date] if value.ndim == dimension + 1 else value


def _admit(y, values, *, retain, smoothing=False):
    if (
        not isinstance(y, torch.Tensor)
        or y.device.type != "cpu"
        or y.dtype != FLOAT
        or y.ndim != 2
        or not 1 <= y.shape[0] <= 20000
        or not 1 <= y.shape[1] <= 8
        or bool(torch.isinf(y).any())
        or not isinstance(values, dict)
        or any(key not in values for key in ("Z", "T", "Q", "H", "a0", "P0", "c", "d"))
    ):
        raise AnalysisError(
            "invalid_system",
            "Use 1..20000 CPU float64 dates with 1..8 measurements and complete system values.",
        )
    prior = values["a0"]
    if not isinstance(prior, torch.Tensor) or prior.ndim != 1 or not 1 <= len(prior) <= 16:
        raise AnalysisError("invalid_system", "a0 must declare 1..16 proper-prior state means.")
    n, p, m = len(y), y.shape[1], len(prior)
    work = n * (m**3 + p**3 + m*m*p + m*p*p) * (6 if smoothing else 2)
    if work > MAX_WORK:
        raise AnalysisError("system_budget", "Kalman/RTS matrix work exceeds 50 million units.")
    buffers = {"likelihood and mask arrays": n*(p+4)*16,
               "current moments and matrix factors": (m*m+p*p+m*p)*128}
    if retain or smoothing:
        buffers["retained prior/filter/innovation moments"] = n*(4*m*m+3*m+3*p*p+3*p)*16
    if smoothing:
        buffers["RTS, lag-one and full disturbance moments"] = n*(
            7*m*m+5*p*p+6*m*p+3*m+3*p
        )*16
    if any(isinstance(value, torch.Tensor) and value.requires_grad for value in values.values()):
        buffers["differentiable Kalman graph"] = n*(m*m+p*p+m*p)*256
    workspace = plan_workspace("proper-prior Kalman and RTS", buffers)
    for key, shape in (
        ("a0", (m,)), ("P0", (m, m)), ("Z", (p, m)), ("T", (m, m)),
        ("Q", (m, m)), ("H", (p, p)), ("c", (m,)), ("d", (p,)),
    ):
        value = values[key]
        scheduled = key not in {"a0", "P0"} and tuple(value.shape) == (n, *shape) if isinstance(value, torch.Tensor) else False
        if (
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != FLOAT or tuple(value.shape) != shape and not scheduled
        ):
            raise AnalysisError(
                "invalid_system", f"{key} needs fixed shape {shape} or an exact {n}-date schedule."
            )
        _finite(value, key)
        if key in {"P0", "Q", "H"}:
            _covariance(value, key)
    return workspace.record()


def _conditional_solve(matrix, right, name):
    """Solve a PSD Gaussian conditional equation only on a compatible range.

    Positive definite covariances use Cholesky. A structural PSD covariance
    uses its positive eigenspace after verifying that the right-hand side has
    no component on the numerical nullspace. No ridge or covariance repair is
    introduced; the rank threshold is 64*eps*dimension*covariance scale.
    """
    matrix = _symmetrize(_covariance(matrix, name))
    factor, status = torch.linalg.cholesky_ex(matrix)
    if not int(status):
        return _finite(torch.cholesky_solve(right, factor), f"{name} conditional solve")
    try:
        eigen, vectors = torch.linalg.eigh(matrix)
    except RuntimeError as error:
        raise AnalysisError(
            "invalid_covariance", f"{name} exceeded rank-compatible factorization precision."
        ) from error
    _finite(eigen, f"{name} conditional eigenvalues")
    scale = max(float(eigen.detach().abs().max()), torch.finfo(FLOAT).tiny)
    threshold = 64 * _EPSILON * len(eigen) * scale
    positive = eigen.detach() > threshold
    null = vectors[:, ~positive]
    if null.shape[1]:
        residual = null.T @ right
        bound = 256 * _EPSILON * len(eigen) * max(
            float(right.detach().abs().max()), torch.finfo(FLOAT).tiny
        )
        if bool((residual.detach().abs() > bound).any()):
            raise AnalysisError(
                "invalid_covariance",
                f"{name} is singular and its conditional cross covariance is rank incompatible.",
            )
    basis = vectors[:, positive]
    if not basis.shape[1]:
        return torch.zeros_like(right)
    solution = basis @ ((basis.T @ right) / eigen[positive, None])
    return _finite(solution, f"{name} rank-compatible conditional solve")


def kalman(y, values, *, retain=False):
    """Filter per-date observed measurements without deleting any time period.

    Retained innovation covariance covers every measurement, even when a
    subset is unobserved; likelihood and updating use the observed principal
    submatrix. Missing innovation entries are NaN, with explicit masks/counts.
    """
    if type(retain) is not bool:
        raise AnalysisError("invalid_system", "retain must be Boolean.")
    workspace = _admit(y, values, retain=retain)
    n, p = y.shape
    mean, covariance = values["a0"], values["P0"]
    identity = torch.eye(len(mean), dtype=FLOAT, device="cpu")
    masks = ~torch.isnan(y)
    counts = masks.sum(1)
    contributions = []
    prior_means, prior_covariances, predicted, innovations, innovation_covariances = [], [], [], [], []
    filtered, filtered_covariances = [], []
    for date in range(n):
        Z, T, Q, H, c, d = (at(values, key, date) for key in ("Z", "T", "Q", "H", "c", "d"))
        predicted_mean = _finite(Z @ mean + d, "Predicted measurement")
        full_innovation_covariance = _finite(_symmetrize(Z @ covariance @ Z.T + H), "Innovation covariance")
        mask = masks[date]
        if retain:
            prior_means.append(mean)
            prior_covariances.append(covariance)
            predicted.append(predicted_mean)
            innovations.append(y[date] - predicted_mean)
            innovation_covariances.append(full_innovation_covariance)
        if bool(mask.any()):
            observed_Z = Z[mask]
            observed_H = H[mask][:, mask]
            error = _finite(y[date, mask] - predicted_mean[mask], "Observed innovation")
            observed_covariance = full_innovation_covariance[mask][:, mask]
            factor, status = torch.linalg.cholesky_ex(observed_covariance)
            if int(status):
                raise AnalysisError(
                    "invalid_covariance",
                    "Observed innovation covariance must be positive definite; no projection applied.",
                )
            solved = torch.cholesky_solve(error[:, None], factor).flatten()
            contribution = _finite(-0.5 * (
                int(counts[date])*math.log(2*math.pi)
                + 2*factor.diagonal().log().sum() + error @ solved
            ), "Per-date likelihood")
            gain = torch.cholesky_solve(observed_Z @ covariance, factor).T
            mean = _finite(mean + gain @ error, "Filtered mean")
            update = identity - gain @ observed_Z
            covariance = _finite(_symmetrize(
                update @ covariance @ update.T + gain @ observed_H @ gain.T
            ), "Filtered covariance")
        else:
            contribution = torch.zeros((), dtype=FLOAT, device="cpu")
        contributions.append(contribution)
        if retain:
            filtered.append(mean)
            filtered_covariances.append(covariance)
        mean = _finite(T @ mean + c, "Next-period mean")
        covariance = _finite(_symmetrize(T @ covariance @ T.T + Q), "Next-period covariance")
    contributions = torch.stack(contributions)
    output = dict(
        log_likelihood=_finite(contributions.sum(), "Complete likelihood"),
        log_likelihood_contributions=contributions, observed_mask=masks, observed_count=counts,
        next_mean=mean, next_covariance=covariance, workspace=workspace,
    )
    if retain:
        output.update(
            prior_mean=torch.stack(prior_means), prior_covariance=torch.stack(prior_covariances),
            predicted=torch.stack(predicted), innovation=torch.stack(innovations),
            innovation_covariance=torch.stack(innovation_covariances),
            filtered=torch.stack(filtered), filtered_covariance=torch.stack(filtered_covariances),
        )
    return output


def smooth(y, values, *, filtered=None):
    """RTS states, lag-one moments and full posterior disturbance moments.

    Lag covariance is Cov(a[t+1],a[t]|all observed Y), with n-1 entries.
    Measurement/state cross covariance is Cov(e[t],a[t]); process/state is
    Cov(u[t],a[t]); process/measurement is Cov(u[t],e[t]). The joint disturbance
    block orders (u,e). Final u is unobserved and retains zero mean/Q[-1].
    """
    workspace = _admit(y, values, retain=True, smoothing=True)
    output = kalman(y, values, retain=True) if filtered is None else filtered
    required = ("prior_mean", "prior_covariance", "filtered", "filtered_covariance",
                "observed_mask", "next_mean", "next_covariance")
    if not isinstance(output, dict) or any(key not in output for key in required):
        raise AnalysisError("invalid_system", "RTS smoothing needs complete retained Kalman moments.")
    n, p = y.shape
    m = len(values["a0"])
    state_identity = torch.eye(m, dtype=FLOAT, device="cpu")
    measurement_identity = torch.eye(p, dtype=FLOAT, device="cpu")
    means, covariances = [None]*n, [None]*n
    gains = [None]*max(n-1, 0)
    means[-1], covariances[-1] = output["filtered"][-1], output["filtered_covariance"][-1]
    for date in range(n-2, -1, -1):
        filtered_covariance = output["filtered_covariance"][date]
        transition = at(values, "T", date)
        predicted_covariance = output["prior_covariance"][date+1]
        gain = _conditional_solve(
            predicted_covariance, transition @ filtered_covariance, "RTS predicted covariance"
        ).T
        gains[date] = gain
        means[date] = _finite(output["filtered"][date] + gain @ (
            means[date+1] - output["prior_mean"][date+1]
        ), "Smoothed state mean")
        conditional_map = state_identity - gain @ transition
        conditional_covariance = (
            conditional_map @ filtered_covariance @ conditional_map.T
            + gain @ at(values, "Q", date) @ gain.T
        )
        covariances[date] = _symmetrize(
            conditional_covariance + gain @ covariances[date+1] @ gain.T
        )
        _covariance(covariances[date], "Smoothed state covariance")
    lag = [covariances[date+1] @ gains[date].T for date in range(n-1)]
    process_means, process_covariances, process_state = [], [], []
    measurement_means, measurement_covariances, measurement_state = [], [], []
    cross_disturbances, joint_disturbances = [], []
    for date in range(n):
        Z, T, Q, H, c, d = (at(values, key, date) for key in ("Z", "T", "Q", "H", "c", "d"))
        state_mean, state_covariance = means[date], covariances[date]
        observed = output["observed_mask"][date]
        indices = torch.where(observed)[0]
        missing = torch.where(~observed)[0]
        if len(indices):
            observed_Z = Z[observed]
            observed_H = H[observed][:, observed]
            loading = torch.zeros((p, len(indices)), dtype=FLOAT, device="cpu")
            loading[indices] = torch.eye(len(indices), dtype=FLOAT, device="cpu")
            if len(missing):
                cross = H[observed][:, ~observed]
                missing_loading = _conditional_solve(
                    observed_H, cross, "Observed measurement noise covariance"
                ).T
                loading[missing] = missing_loading
            residual_map = measurement_identity - loading @ measurement_identity[indices]
            residual_covariance = residual_map @ H @ residual_map.T
            mapped_Z = loading @ observed_Z
            measurement_mean = loading @ (y[date, observed] - observed_Z @ state_mean - d[observed])
            measurement_covariance = _symmetrize(
                residual_covariance + mapped_Z @ state_covariance @ mapped_Z.T
            )
            measurement_cross = -mapped_Z @ state_covariance
        else:
            mapped_Z = torch.zeros((p, m), dtype=FLOAT, device="cpu")
            measurement_mean = torch.zeros(p, dtype=FLOAT, device="cpu")
            measurement_covariance = H
            measurement_cross = torch.zeros((p, m), dtype=FLOAT, device="cpu")
        if date < n-1:
            predicted_covariance = output["prior_covariance"][date+1]
            process_gain = _conditional_solve(
                predicted_covariance, Q, "Process predicted covariance"
            ).T
            process_mean = process_gain @ (
                means[date+1] - output["prior_mean"][date+1]
            )
            process_map = state_identity - process_gain
            filtered_process_covariance = T @ output["filtered_covariance"][date] @ T.T
            process_covariance = _symmetrize(
                process_map @ Q @ process_map.T
                + process_gain @ filtered_process_covariance @ process_gain.T
                + process_gain @ covariances[date+1] @ process_gain.T
            )
            process_cross = process_gain @ (
                covariances[date+1] - predicted_covariance
            ) @ gains[date].T
            disturbance_cross = -process_cross @ mapped_Z.T
        else:
            process_mean = torch.zeros(m, dtype=FLOAT, device="cpu")
            process_covariance = Q
            process_cross = torch.zeros((m, m), dtype=FLOAT, device="cpu")
            disturbance_cross = torch.zeros((m, p), dtype=FLOAT, device="cpu")
        _finite(process_mean, "Smoothed process disturbance")
        _finite(measurement_mean, "Smoothed measurement disturbance")
        _covariance(process_covariance, "Smoothed process disturbance covariance")
        _covariance(measurement_covariance, "Smoothed measurement disturbance covariance")
        joint = torch.cat((
            torch.cat((process_covariance, disturbance_cross), dim=1),
            torch.cat((disturbance_cross.T, measurement_covariance), dim=1),
        ), dim=0)
        _covariance(joint, "Joint smoothed disturbance covariance")
        process_means.append(process_mean)
        process_covariances.append(process_covariance)
        process_state.append(process_cross)
        measurement_means.append(measurement_mean)
        measurement_covariances.append(measurement_covariance)
        measurement_state.append(measurement_cross)
        cross_disturbances.append(disturbance_cross)
        joint_disturbances.append(joint)
    final_T, final_Q, final_c = (at(values, key, n-1) for key in ("T", "Q", "c"))
    result = dict(output)
    result.update(
        smoothed=torch.stack(means), smoothed_covariance=torch.stack(covariances),
        lag_one_covariance=torch.stack(lag) if lag else torch.empty((0, m, m), dtype=FLOAT, device="cpu"),
        process_disturbance=torch.stack(process_means),
        process_disturbance_covariance=torch.stack(process_covariances),
        measurement_disturbance=torch.stack(measurement_means),
        measurement_disturbance_covariance=torch.stack(measurement_covariances),
        process_state_covariance=torch.stack(process_state),
        measurement_state_covariance=torch.stack(measurement_state),
        process_measurement_covariance=torch.stack(cross_disturbances),
        joint_disturbance_covariance=torch.stack(joint_disturbances),
        smoothed_next_mean=_finite(final_T @ means[-1] + final_c, "Smoothed next mean"),
        smoothed_next_covariance=_finite(_symmetrize(
            final_T @ covariances[-1] @ final_T.T + final_Q
        ), "Smoothed next covariance"),
        smoother_workspace=workspace,
    )
    return result
