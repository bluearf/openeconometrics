"""Exact diffuse Gaussian moments; no finite large-prior approximation.

At date t, a[t] precedes y[t] and T[t] is the outgoing transition.
P(kappa) = kappa*P_inf + P_star. Diffuse rank is consumed by observations,
never by silently discarding an unidentified initial condition.
The bounded smoother uses exact Gaussian saddle-point conditioning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from openecon.econometrics.core import TableSet
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.state_lifecycle import _encoded_json_admission, _json_export_admission, _state_copy_admission
from openecon.econometrics.tsworkflows.ssengine import (
    FLOAT, _covariance, _finite, _symmetrize, at,
)

MAX_WORK = 50_000_000
MAX_INDEX_BYTES = 2 * 1024**2
MAX_STATE_BYTES = 64 * 1024**2
MAX_JSON_ITEMS = 4_000_000
_EPS = torch.finfo(FLOAT).eps
_KEYS = {"T", "Z", "Q", "H", "c", "d", "a0", "P_star", "P_inf"}
SCHEMA = "openecon.sspace.exact-diffuse.v1"


def _maxabs(value):
    return float(value.detach().abs().max()) if value.numel() else 0.0


def _input_covariance(value, name):
    """Input covariance admission uses its own units; no repair is applied."""
    _covariance(value, name)
    for matrix in value.detach().reshape(-1, value.shape[-1], value.shape[-1]):
        diagonal = matrix.diagonal()
        if bool((diagonal < 0).any()) or bool((matrix[diagonal == 0] != 0).any()):
            raise AnalysisError("invalid_covariance", f"{name} has a negative diagonal or a nonzero covariance with a zero-variance component.")
        active = diagonal > 0
        if bool(active.any()):
            scale = diagonal[active].sqrt()
            normalized = _symmetrize(matrix[active][:, active])/scale[:, None]/scale[None, :]
            if not bool(torch.isfinite(normalized).all()):
                raise AnalysisError("invalid_covariance", f"{name} cannot be validated in its declared coordinate units.")
            if bool((torch.linalg.eigvalsh(normalized) < 0).any()):
                raise AnalysisError("invalid_covariance", f"{name} has a negative or unresolved eigenvalue in its declared coordinate units.")


def _rank(matrix, name, *, upper_rank=None):
    _covariance(matrix, name)
    raw = _symmetrize(matrix).detach()
    diagonal = raw.diagonal()
    active = diagonal > 0
    if not bool(active.any()):
        if bool((raw != 0).any()):
            raise AnalysisError("diffuse_rank", f"{name} has unresolved zero-diagonal covariance geometry.")
        return 0
    if bool((raw[~active] != 0).any()):
        raise AnalysisError("diffuse_rank", f"{name} has unresolved zero-diagonal covariance geometry.")
    # Covariance units cannot decide whether an initial condition is diffuse:
    # diag(1,1e-16) has TWO directions, however small the second variance.
    # Equilibrate only for this detached rank decision, never alter moments.
    scale = diagonal[active].sqrt()
    normalized = raw[active][:, active]/scale[:, None]/scale[None, :]
    eigen = torch.linalg.eigvalsh(normalized)
    threshold = 128 * _EPS * len(eigen) * max(float(eigen.abs().max()), 1.)
    resolved = int((eigen > threshold).sum())
    if resolved == upper_rank:
        # A known algebraic rank bound resolves the remaining eigenvalues of
        # a derived covariance. Unit normalization can magnify subtraction
        # roundoff, but cannot create additional diffuse directions here.
        return resolved
    if bool((eigen < -threshold).any()):
        raise AnalysisError("diffuse_rank", f"{name} is not positive semidefinite in its declared coordinate units.")
    ambiguous = (eigen != 0) & (eigen.abs() <= threshold)
    if bool(ambiguous.any()) and resolved != upper_rank:
        raise AnalysisError("diffuse_rank", f"{name} has a positive near-singular direction whose rank cannot be resolved in float64.")
    return resolved


def _pivots(matrix, rank):
    """Choose a nonsingular principal block; decisions alone are detached."""
    if rank == len(matrix):
        return list(range(rank))
    residue = matrix.detach().clone()
    indices = []
    for _ in range(rank):
        diagonal = residue.diagonal().clone()
        if indices:
            diagonal[indices] = -math.inf
        index = int(diagonal.argmax())
        if float(diagonal[index]) <= 0:
            raise AnalysisError("diffuse_rank", "Diffuse principal block is numerically singular.")
        indices.append(index)
        column = residue[:, index].clone()
        residue = _symmetrize(residue - torch.outer(column, column) / residue[index, index])
    return indices


def _factor(matrix, name):
    factor, status = torch.linalg.cholesky_ex(_symmetrize(matrix))
    if int(status):
        raise AnalysisError(
            "singular_innovation", f"{name} must have a nondegenerate Gaussian density; no jitter or pseudoinverse."
        )
    return factor


def _solve(factor, rhs):
    vector = rhs.ndim == 1
    answer = torch.cholesky_solve(rhs[:, None] if vector else rhs, factor)
    return answer[:, 0] if vector else answer


def _plan(n, p, m, max_work, *, retain=True, smoothing=False, observed=0, differentiable=False):
    if type(max_work) is not int or not 1 <= max_work <= MAX_WORK:
        raise AnalysisError("invalid_resource_budget", "max_work must be an integer in 1..50000000.")
    work = n * (m**3 + p**3 + m*m*p + m*p*p) * 12
    buffers = {"current diffuse/proper moments and rank factors": (m*m + p*p + m*p)*256,
               "likelihood, ranks and original observation masks": n*(p+5)*16,
               "complete equation paths and input conversion": n*(2*m*m+m*p+p*p+m+p)*24}
    if retain or smoothing:
        buffers["retained exact diffuse/filter moments"] = n*(8*m*m+4*p*p+4*m+4*p)*16
        buffers["portable numerical state and typed identity serialization"] = n*(8*m*m+4*p*p+4*m+4*p)*96 + MAX_INDEX_BYTES*3
    if differentiable:
        buffers["differentiable exact diffuse recursion"] = n*(m*m+p*p+m*p)*768
    if smoothing:
        states, equations = (n+1)*m, observed+m
        if n > 256 or states > 1024 or equations > 512:
            raise AnalysisError("system_budget", "Exact batch smoothing admits <=256 dates, <=1024 stacked states and <=512 equations.")
        work += n*n*m**3*4 + equations**3*3 + states*states*max(1, equations)*4 + n*p*equations**2*2
        buffers["joint proper and posterior state covariance"] = states*states*8*8
        buffers["diffuse saddle-point matrix and solve buffers"] = equations*equations*8*8
        buffers["state/observation and complete disturbance moments"] = (states*equations + n*(m+p)**2)*8*8
        buffers["portable joint posterior serialization"] = states*states*96
        if differentiable:
            buffers["differentiable joint smoothing graph"] = (states*states+equations*equations)*8*16
    if work > max_work:
        raise AnalysisError("system_budget", f"Exact diffuse work {work} exceeds max_work={max_work}.")
    workspace = plan_workspace("exact diffuse Gaussian filtering/smoothing", buffers).record()
    workspace["estimated_work"] = work
    workspace["max_work"] = max_work
    return workspace


def _admit(y, values, max_work, *, retain=True, smoothing=False):
    if (not isinstance(y, torch.Tensor) or y.dtype != FLOAT or y.device.type != "cpu"
            or y.ndim != 2 or not 1 <= len(y) <= 20000 or not 1 <= y.shape[1] <= 8
            or not isinstance(values, dict) or set(values) != _KEYS):
        raise AnalysisError("invalid_system", "Use CPU float64 dates/measurements and the complete exact-diffuse system.")
    a0 = values["a0"]
    if not isinstance(a0, torch.Tensor) or a0.ndim != 1 or not 1 <= len(a0) <= 16:
        raise AnalysisError("invalid_system", "a0 must declare 1..16 state means.")
    n, p, m = len(y), y.shape[1], len(a0)
    shapes = {"a0": (m,), "P_star": (m, m), "P_inf": (m, m),
              "T": (m, m), "Z": (p, m), "Q": (m, m), "H": (p, p), "c": (m,), "d": (p,)}
    for key, shape in shapes.items():
        value = values[key]
        schedule = key not in {"a0", "P_star", "P_inf"}
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != FLOAT
                or tuple(value.shape) not in ([shape, (n, *shape)] if schedule else [shape])):
            raise AnalysisError("invalid_system", f"{key} needs CPU float64 shape {shape} or an exact date schedule.")
    differentiable = any(value.requires_grad for value in values.values())
    # First admit all resident geometry and the bounded mask scan itself.
    # The observation count is not known until that admitted scan finishes;
    # no conditioning matrix is allocated before the complete second plan.
    workspace = _plan(n, p, m, max_work, retain=retain, smoothing=smoothing,
                      observed=0, differentiable=differentiable)
    if bool(torch.isinf(y).any()):
        raise AnalysisError("invalid_system", "Infinite measurements are not missing values.")
    if smoothing:
        workspace = _plan(n, p, m, max_work, retain=retain, smoothing=True,
                          observed=int((~torch.isnan(y)).sum()), differentiable=differentiable)
    for key, value in values.items():
        _finite(value, key)
    # Admission precedes covariance factorization and all retained buffers.
    for key in ("P_star", "P_inf", "Q", "H"):
        _input_covariance(values[key], key)
    return workspace


def _raw_array(value, shape, *, missing=False):
    if shape:
        return (isinstance(value, list) and len(value) == shape[0]
                and all(_raw_array(item, shape[1:], missing=missing) for item in value))
    if missing and value is None:
        return True
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value) and (type(value) is not int or int(float(value)) == value)
    except OverflowError:
        return False


def _comparable(output):
    """Ambient admission capacity is not a fitted moment or an equation."""
    if not isinstance(output, dict):
        raise AnalysisError("invalid_state", "Saved moments must be a complete object.")
    result = dict(output)
    for key in ("workspace", "smoother_workspace"):
        if key in result and isinstance(result[key], dict):
            result[key] = {name: value for name, value in result[key].items() if name != "budget_bytes"}
    return result


def _remaining_loading(loading, measured):
    """Exact differentiable nullspace factor; no covariance subtraction/repair.

    For D=Z A, Pi_new=A[I-D'(DD')^-1 D]A'. Express this same projector as
    B(B'B)^-1 B' by elimination, preserving its known reduced rank.
    """
    consumed, rank = measured.shape
    if consumed == rank:
        return torch.empty((len(loading), 0), dtype=FLOAT, device="cpu")
    selected = _pivots(measured.T@measured, consumed)
    remaining = [i for i in range(rank) if i not in selected]
    basis = torch.zeros((rank, len(remaining)), dtype=FLOAT, device="cpu")
    basis[remaining] = torch.eye(len(remaining), dtype=FLOAT, device="cpu")
    basis[selected] = -torch.linalg.solve(measured[:, selected], measured[:, remaining])
    factor = _factor(basis.T@basis, "Diffuse nullspace metric")
    return torch.linalg.solve_triangular(factor, (loading@basis).T, upper=False).T


def _measurement_update(mean, proper, diffuse, observed_y, Z, H, offset, loading):
    """Determinant-one innovation decomposition preserves the diffuse measure."""
    error = observed_y - Z @ mean - offset
    design = Z@loading
    infinity = _symmetrize(design@design.T)
    finite = _symmetrize(Z @ proper @ Z.T + H)
    rank = _rank(infinity, "Diffuse innovation", upper_rank=loading.shape[1])
    p = len(error)
    if not rank:
        factor = _factor(finite, "Proper observed innovation covariance")
        gain = _solve(factor, Z @ proper).T
        contribution = -0.5 * (p*math.log(2*math.pi) + 2*factor.diagonal().log().sum()
                                + error @ _solve(factor, error))
        return (mean + gain @ error, _symmetrize(proper - gain @ finite @ gain.T),
                diffuse, contribution, 0, p, loading)
    selected = _pivots(infinity, rank)
    remainder = [i for i in range(p) if i not in selected]
    principal = infinity[selected][:, selected]
    factor_inf = _factor(principal, "Active diffuse innovation covariance")
    transform = torch.zeros((p, p), dtype=FLOAT, device="cpu")
    transform[:rank, selected] = torch.eye(rank, dtype=FLOAT, device="cpu")
    if remainder:
        contrast_loading = _solve(factor_inf, infinity[selected][:, remainder]).T
        transform[rank:, remainder] = torch.eye(len(remainder), dtype=FLOAT, device="cpu")
        transform[rank:, selected] = -contrast_loading
    transformed_Z = transform @ Z
    transformed_error = transform @ error
    transformed_finite = _symmetrize(transform @ finite @ transform.T)
    cross = proper @ transformed_Z.T
    gain_inf = _solve(factor_inf, design[selected]@loading.T).T
    diff_error = transformed_error[:rank]
    diff_cross = cross[:, :rank]
    diff_finite = transformed_finite[:rank, :rank]
    contribution = -factor_inf.diagonal().log().sum()
    if remainder:
        factor_proper = _factor(transformed_finite[rank:, rank:], "Proper innovation after diffuse elimination")
        gain_proper = _solve(factor_proper, cross[:, rank:].T).T
        proper_error = transformed_error[rank:]
        between = transformed_finite[:rank, rank:]
        mean = mean + gain_proper @ proper_error
        proper = proper - gain_proper @ transformed_finite[rank:, rank:] @ gain_proper.T
        diff_cross = diff_cross - gain_proper @ between.T
        diff_error = diff_error - between @ _solve(factor_proper, proper_error)
        diff_finite = diff_finite - between @ _solve(factor_proper, between.T)
        contribution = contribution - 0.5*(len(remainder)*math.log(2*math.pi)
            + 2*factor_proper.diagonal().log().sum() + proper_error @ _solve(factor_proper, proper_error))
    mean = mean + gain_inf @ diff_error
    proper = _symmetrize(proper - gain_inf @ diff_cross.T - diff_cross @ gain_inf.T
                          + gain_inf @ diff_finite @ gain_inf.T)
    loading = _remaining_loading(loading, design[selected])
    diffuse = _symmetrize(loading@loading.T)
    return mean, proper, diffuse, contribution, rank, p-rank, loading


def filter_exact_diffuse(y, values, *, retain=True, require_identified=True, max_work=MAX_WORK):
    """Exact rank-reducing filtering; all original dates and masks survive.

    Likelihood omits the divergent initial diffuse normalizers, exactly as
    KFAS's Gaussian exact-diffuse convention. It is not a proper marginal
    density suitable for comparing different diffuse rank/normalization.
    """
    if type(retain) is not bool or type(require_identified) is not bool:
        raise AnalysisError("invalid_system", "retain and require_identified must be Boolean.")
    workspace = _admit(y, values, max_work, retain=retain)
    mean, proper, diffuse = values["a0"], values["P_star"], values["P_inf"]
    loading = _initial_loading(diffuse)
    initial_rank = rank = loading.shape[1]
    masks = ~torch.isnan(y)
    contributions, prior_ranks, filtered_ranks, diffuse_counts, proper_counts = [], [], [], [], []
    retained = {key: [] for key in ("prior_mean", "prior_covariance", "prior_diffuse_covariance",
                                  "filtered", "filtered_covariance", "filtered_diffuse_covariance",
                                  "predicted", "innovation", "innovation_covariance", "innovation_diffuse_covariance")}
    for date in range(len(y)):
        Z, T, Q, H, c, d = (at(values, key, date) for key in ("Z", "T", "Q", "H", "c", "d"))
        prediction = Z @ mean + d
        prior_ranks.append(rank)
        if retain:
            for key, value in (("prior_mean", mean), ("prior_covariance", proper),
                    ("prior_diffuse_covariance", diffuse), ("predicted", prediction),
                    ("innovation", y[date]-prediction), ("innovation_covariance", Z@proper@Z.T+H),
                    ("innovation_diffuse_covariance", Z@diffuse@Z.T)):
                retained[key].append(value)
        observed = masks[date]
        old_scale = max(float(diffuse.detach().abs().max()), torch.finfo(FLOAT).tiny)
        if bool(observed.any()):
            mean, proper, diffuse, contribution, consumed, ordinary, loading = _measurement_update(
                mean, proper, diffuse, y[date, observed], Z[observed], H[observed][:, observed], d[observed], loading
            )
        else:
            contribution = torch.zeros((), dtype=FLOAT, device="cpu")
            consumed = ordinary = 0
        rank -= consumed
        if rank < 0:
            raise AnalysisError("diffuse_rank", "Observation consumed more diffuse directions than the current rank.")
        if not rank:
            if float(diffuse.detach().abs().max()) > 512*_EPS*len(mean)*old_scale:
                raise AnalysisError("diffuse_rank", "Diffuse rank elimination exceeded numerical precision.")
            # Algebraically zero after the last diffuse direction is observed.
            diffuse = torch.zeros_like(diffuse)
            _covariance(proper, "Identified filtered covariance")
        elif loading.shape[1] != rank:
            raise AnalysisError("diffuse_rank", "Diffuse covariance rank disagrees with observation rank reduction.")
        contributions.append(_finite(contribution, "Exact diffuse likelihood"))
        diffuse_counts.append(consumed)
        proper_counts.append(ordinary)
        filtered_ranks.append(rank)
        if retain:
            retained["filtered"].append(mean)
            retained["filtered_covariance"].append(proper)
            retained["filtered_diffuse_covariance"].append(diffuse)
        mean = _finite(T @ mean+c, "Next exact-diffuse mean")
        proper = _finite(_symmetrize(T @ proper @ T.T+Q), "Next finite covariance coefficient")
        loading = _finite(T@loading, "Next diffuse loading")
        diffuse = _finite(_symmetrize(loading@loading.T), "Next diffuse covariance")
        if rank and _rank(loading.T@loading, "Next diffuse loading covariance", upper_rank=rank) != rank:
            raise AnalysisError("unidentified_diffuse", "A transition destroys an unobserved diffuse initial direction.")
        if not rank:
            _covariance(proper, "Identified next covariance")
    if require_identified and rank:
        raise AnalysisError("unidentified_diffuse", "Observed data do not identify every initial diffuse direction.")
    result = dict(log_likelihood=torch.stack(contributions).sum(),
        log_likelihood_contributions=torch.stack(contributions), observed_mask=masks,
        observed_count=masks.sum(1), initial_diffuse_rank=initial_rank,
        prior_diffuse_rank=prior_ranks, filtered_diffuse_rank=filtered_ranks,
        diffuse_observation_count=diffuse_counts, proper_observation_count=proper_counts,
        final_diffuse_rank=rank, next_mean=mean, next_covariance=proper,
        next_diffuse_covariance=diffuse, workspace=workspace,
        likelihood_normalization="KFAS-exact-diffuse: no quadratic or log(2*pi) for active diffuse observations")
    if retain:
        result.update({key: torch.stack(value) for key, value in retained.items()})
    return result


def _initial_loading(diffuse):
    rank = _rank(diffuse, "Initial diffuse covariance")
    if not rank:
        return torch.empty((len(diffuse), 0), dtype=FLOAT, device="cpu")
    selected = _pivots(diffuse, rank)
    factor = _factor(diffuse[selected][:, selected], "Initial diffuse principal block")
    return torch.linalg.solve_triangular(factor, diffuse[selected], upper=False).T


def _joint_prior(y, values):
    """Proper Gaussian joint moments and exact flat-initial-condition loadings."""
    n, p = y.shape
    m = len(values["a0"])
    means = [values["a0"]]
    loadings = [_initial_loading(values["P_inf"])]
    blocks = [[values["P_star"]]]
    for date in range(n):
        T, Q, c = (at(values, key, date) for key in ("T", "Q", "c"))
        previous = list(blocks[-1])
        new_row = [T @ block for block in previous]
        for i, row in enumerate(blocks):
            row.append(new_row[i].T)
        new_row.append(_symmetrize(T @ previous[-1] @ T.T+Q))
        blocks.append(new_row)
        means.append(T @ means[-1]+c)
        loadings.append(T @ loadings[-1])
    joint = torch.cat([torch.cat(row, dim=1) for row in blocks], dim=0)
    mean = torch.cat(means)
    loading = torch.cat(loadings)
    observed = torch.where((~torch.isnan(y)).flatten())[0]
    observation = torch.zeros((len(observed), (n+1)*m), dtype=FLOAT, device="cpu")
    noise = torch.zeros((len(observed), len(observed)), dtype=FLOAT, device="cpu")
    offsets = []
    observed_dates = observed // p
    for i, physical in enumerate(observed.tolist()):
        date, component = divmod(physical, p)
        observation[i, date*m:(date+1)*m] = at(values, "Z", date)[component]
        offsets.append(at(values, "d", date)[component])
        same = torch.where(observed_dates == date)[0]
        components = observed[same] % p
        noise[i, same] = at(values, "H", date)[component, components]
    offset = torch.stack(offsets) if offsets else torch.empty(0, dtype=FLOAT, device="cpu")
    return mean, joint, loading, observation, noise, offset, observed


def smooth_exact_diffuse(y, values, *, max_work=MAX_WORK):
    """Exact flat-prior conditioning, full cross-date state and disturbance moments.

    Solve [Omega,D; D',0] for the universal-Gaussian predictor. This is exact
    diffuse conditioning, not a large-P0 smoother or an estimation procedure.
    """
    workspace = _admit(y, values, max_work, smoothing=True)
    output = filter_exact_diffuse(y, values, max_work=max_work)
    mean, prior, diffuse, observation, noise, offset, observed = _joint_prior(y, values)
    n, p = y.shape
    m, r = len(values["a0"]), diffuse.shape[1]
    cross = prior @ observation.T
    omega = _symmetrize(observation @ cross + noise)
    design = observation @ diffuse
    if r and int(torch.linalg.matrix_rank(design.detach())) != r:
        raise AnalysisError("unidentified_diffuse", "Joint observations do not identify the diffuse initial conditions.")
    saddle = torch.cat((torch.cat((omega, design), dim=1),
        torch.cat((design.T, torch.zeros((r, r), dtype=FLOAT, device="cpu")), dim=1)), dim=0)
    bridge = torch.cat((cross, diffuse), dim=1)
    rhs = torch.cat((y.flatten()[observed]-observation@mean-offset,
                     torch.zeros(r, dtype=FLOAT, device="cpu")))
    try:
        lu, pivots = torch.linalg.lu_factor(saddle)
        solved = torch.linalg.lu_solve(lu, pivots, torch.cat((rhs[:, None], bridge.T), dim=1))
    except RuntimeError as error:
        raise AnalysisError("singular_innovation", "Exact diffuse conditional saddle-point equations are singular.") from error
    _finite(solved, "Joint exact-diffuse conditioning solve")
    residual = saddle @ solved - torch.cat((rhs[:, None], bridge.T), dim=1)
    scale = max(_maxabs(saddle)*_maxabs(solved), _maxabs(bridge), _maxabs(rhs), 1.0)
    if _maxabs(residual) > 4096*_EPS*len(saddle)*scale:
        raise AnalysisError("singular_innovation", "Exact diffuse conditional solve exceeded residual precision.")
    conditional_mean = mean + bridge @ solved[:, 0]
    conditional = _symmetrize(prior - bridge @ solved[:, 1:])
    _covariance(conditional, "Complete exact-diffuse smoothed covariance")
    states = conditional_mean.reshape(n+1, m)
    covariance = conditional.reshape(n+1, m, n+1, m)
    process, process_cov, process_state = [], [], []
    measurement, measurement_cov, measurement_state = [], [], []
    cross_noise, joint_noise = [], []
    for date in range(n):
        T, H, c = (at(values, key, date) for key in ("T", "H", "c"))
        state_block = covariance[date, :, date, :]
        lag = covariance[date+1, :, date, :]
        process_mean = states[date+1]-T@states[date]-c
        process_variance = _symmetrize(covariance[date+1, :, date+1, :]+T@state_block@T.T-lag@T.T-T@lag.T)
        process_cross = lag-T@state_block
        noise_cross = torch.zeros((p, len(observed)), dtype=FLOAT, device="cpu")
        same = torch.where(observed//p == date)[0]
        noise_cross[:, same] = H[:, observed[same] % p]
        noise_bridge = torch.cat((noise_cross, torch.zeros((p, r), dtype=FLOAT, device="cpu")), dim=1)
        noise_mean = noise_bridge @ solved[:, 0]
        noise_state = -noise_bridge @ solved[:, 1:]
        try:
            noise_variance = _symmetrize(H-noise_bridge @ torch.linalg.lu_solve(lu, pivots, noise_bridge.T))
        except RuntimeError as error:
            raise AnalysisError("singular_innovation", "Measurement-noise conditioning is singular.") from error
        disturbance_cross = (noise_state[:, (date+1)*m:(date+2)*m]
                              -noise_state[:, date*m:(date+1)*m]@T.T).T
        joint_variance = torch.cat((torch.cat((process_variance, disturbance_cross), dim=1),
            torch.cat((disturbance_cross.T, noise_variance), dim=1)), dim=0)
        _covariance(joint_variance, "Smoothed joint disturbances")
        process.append(process_mean)
        process_cov.append(process_variance)
        process_state.append(process_cross)
        measurement.append(noise_mean)
        measurement_cov.append(noise_variance)
        measurement_state.append(noise_state[:, date*m:(date+1)*m])
        cross_noise.append(disturbance_cross)
        joint_noise.append(joint_variance)
    output.update(smoothed=states[:-1], smoothed_covariance=torch.stack([covariance[t, :, t, :] for t in range(n)]),
        smoothed_joint_covariance=covariance, smoothed_next_mean=states[-1],
        smoothed_next_covariance=covariance[n, :, n, :],
        lag_one_covariance=torch.stack([covariance[t+1, :, t, :] for t in range(n-1)]) if n > 1
            else torch.empty((0, m, m), dtype=FLOAT, device="cpu"),
        process_disturbance=torch.stack(process), process_disturbance_covariance=torch.stack(process_cov),
        process_state_covariance=torch.stack(process_state), measurement_disturbance=torch.stack(measurement),
        measurement_disturbance_covariance=torch.stack(measurement_cov),
        measurement_state_covariance=torch.stack(measurement_state), process_measurement_covariance=torch.stack(cross_noise),
        joint_disturbance_covariance=torch.stack(joint_noise), smoother_workspace=workspace)
    return output


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise AnalysisError("invalid_state", "Nonfinite exact-diffuse state value.")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _json_admission(value, *, limit=MAX_STATE_BYTES):
    """Bound caller-owned JSON trees before serialization, copying or decoding."""
    size, items = 0, 0

    def visit(item, depth=0):
        nonlocal size, items
        if depth > 32:
            raise AnalysisError("metadata_limit", "Saved state nesting exceeds 32 levels.")
        items += 1
        if items > MAX_JSON_ITEMS:
            raise AnalysisError("metadata_limit", "Saved state exceeds the bounded JSON item envelope.")
        if type(item) is dict:
            size += 16
            for key, child in item.items():
                if type(key) is not str or len(key) > 4096:
                    raise AnalysisError("metadata_limit", "Saved state keys must be bounded strings.")
                size += 8 + 6*len(key)
                visit(child, depth+1)
        elif type(item) is list:
            if len(item) > MAX_JSON_ITEMS-items:
                raise AnalysisError("metadata_limit", "Saved arrays exceed the bounded JSON item envelope.")
            size += 2
            for child in item:
                visit(child, depth+1)
        elif type(item) is str:
            if len(item) > 4096:
                raise AnalysisError("metadata_limit", "Saved strings exceed 4096 characters.")
            size += 2 + 6*len(item)
        elif type(item) is int:
            if item.bit_length() > 256:
                raise AnalysisError("metadata_limit", "Saved integers exceed 256 bits.")
            size += 4 + max(1, item.bit_length())
        elif type(item) is float and math.isfinite(item):
            size += 32
        elif item is None or type(item) is bool:
            size += 6
        else:
            raise AnalysisError("invalid_state", "Saved JSON values must be finite scalar, list or plain object values.")
        if size > limit:
            raise AnalysisError("metadata_limit", "Saved metadata exceed the declared JSON byte envelope.")

    visit(value)
    return size


def _load_record(value):
    if isinstance(value, str):
        # Check length before the first parser allocation. UTF-8 may need four
        # bytes per character; use a bounded running count without encoding it.
        _encoded_json_admission(value, limit=MAX_STATE_BYTES, operation="exact diffuse saved JSON decoding")
        value = json.loads(value)
    admitted_bytes = _json_admission(value)
    plan_workspace("exact diffuse saved state validation", {"bounded JSON copy and digest buffers": admitted_bytes*4})
    if type(value) is not dict:
        raise AnalysisError("invalid_state", "Saved state must be a plain JSON object.")
    if type(value.get("source")) is dict:
        for key in ("index", "time"):
            if value["source"].get(key) is not None:
                _json_admission(value["source"][key], limit=MAX_INDEX_BYTES)
    return dict(value)


def save_diffuse_state(y, values, *, smoothing=True, max_work=MAX_WORK):
    """Complete portable fixed-system state, with no optimizer or fit on replay."""
    if type(smoothing) is not bool:
        raise AnalysisError("invalid_state", "smoothing must be Boolean.")
    output = (smooth_exact_diffuse if smoothing else filter_exact_diffuse)(y, values, max_work=max_work)
    record = dict(schema=SCHEMA, smoothing=smoothing, max_work=max_work,
                  y=_json_value(y), system=_json_value(values), output=_json_value(output))
    _json_admission(record)
    record["sha256"] = _digest(record)
    return record


def restore_diffuse_state(state):
    """Verify complete equations/masks/moments by deterministic fixed-system replay."""
    try:
        record = _load_record(state)
    except (ValueError, TypeError, RecursionError) as error:
        raise AnalysisError("invalid_state", "Exact-diffuse state must be strict JSON or a mapping.") from error
    expected = {"schema", "smoothing", "max_work", "y", "system", "output", "sha256"}
    if set(record) != expected or record["schema"] != SCHEMA:
        raise AnalysisError("invalid_state", "Unknown or incomplete exact-diffuse state schema.")
    checksum = record.pop("sha256")
    try:
        if not isinstance(checksum, str) or checksum != _digest(record):
            raise AnalysisError("invalid_state", "Exact-diffuse state checksum mismatch.")
        if not isinstance(record["system"], Mapping) or set(record["system"]) != _KEYS:
            raise AnalysisError("invalid_state", "Saved exact-diffuse system fields are invalid.")
        # Validate source array geometry before tensor materialization.
        rows = record["y"]
        if (not isinstance(rows, list) or not 1 <= len(rows) <= 20000
                or not isinstance(rows[0], list) or not 1 <= len(rows[0]) <= 8
                or any(not isinstance(row, list) or len(row) != len(rows[0]) for row in rows)):
            raise AnalysisError("invalid_state", "Saved measurement geometry is invalid.")
        n, p = len(rows), len(rows[0])
        initial = record["system"]["a0"]
        if not isinstance(initial, list) or not 1 <= len(initial) <= 16:
            raise AnalysisError("invalid_state", "Saved initial-state geometry is invalid.")
        m = len(initial)
        shapes = {"a0": (m,), "P_star": (m, m), "P_inf": (m, m), "T": (m, m),
                  "Z": (p, m), "Q": (m, m), "H": (p, p), "c": (m,), "d": (p,)}
        if type(record["smoothing"]) is not bool or not _raw_array(rows, (n, p), missing=True):
            raise AnalysisError("invalid_state", "Saved observations and smoothing mode are invalid.")
        for key, shape in shapes.items():
            source = record["system"][key]
            if not (_raw_array(source, shape) or key not in {"a0", "P_star", "P_inf"}
                    and _raw_array(source, (n, *shape))):
                raise AnalysisError("invalid_state", f"Saved {key} geometry or numerical values are invalid.")
        _plan(n, p, m, record["max_work"], smoothing=record["smoothing"],
              observed=sum(x is not None for row in rows for x in row))
        y = torch.tensor([[math.nan if x is None else x for x in row] for row in rows], dtype=FLOAT, device="cpu")
        values = {key: torch.tensor(value, dtype=FLOAT, device="cpu") for key, value in record["system"].items()}
        replay = save_diffuse_state(y, values, smoothing=record["smoothing"], max_work=record["max_work"])
    except (ValueError, TypeError, RuntimeError, OverflowError) as error:
        raise AnalysisError("invalid_state", "Saved exact-diffuse numerical state is invalid.") from error
    if _digest(_comparable(replay["output"])) != _digest(_comparable(record["output"])):
        raise AnalysisError("invalid_state", "Saved exact-diffuse moments disagree with the complete equations and data.")
    return replay


def forecast_exact_diffuse(state, future_values, *, max_work=MAX_WORK):
    """Conditional forecasts from a restored proper terminal state; no refitting.

    Every future matrix/vector has an explicit per-date path, including the
    outgoing transition of each forecast date. Parameter uncertainty is absent.
    """
    replay = restore_diffuse_state(state)
    required = {"T", "Z", "Q", "H", "c", "d"}
    if not isinstance(future_values, dict) or set(future_values) != required:
        raise AnalysisError("invalid_system", "Supply all six explicit future equation paths.")
    raw_Z = future_values["Z"]
    if not isinstance(raw_Z, torch.Tensor) or raw_Z.ndim != 3 or not 1 <= len(raw_Z) <= 10000:
        raise AnalysisError("invalid_system", "Future Z must declare 1..10000 complete forecast dates.")
    n, p, _ = raw_Z.shape
    m = len(replay["system"]["a0"])
    _plan(n, p, m, max_work, differentiable=any(
        isinstance(value, torch.Tensor) and value.requires_grad for value in future_values.values()
    ))
    values = dict(future_values, a0=torch.tensor(replay["output"]["next_mean"], dtype=FLOAT, device="cpu"),
                  P_star=torch.tensor(replay["output"]["next_covariance"], dtype=FLOAT, device="cpu"),
                  P_inf=torch.zeros_like(torch.tensor(replay["system"]["P_inf"], dtype=FLOAT, device="cpu")))
    for key, value in future_values.items():
        dimension = 1 if key in {"c", "d"} else 2
        if not isinstance(value, torch.Tensor) or value.ndim != dimension+1 or len(value) != n:
            raise AnalysisError("invalid_system", "Every future equation requires an exact per-date path.")
    y = torch.full((n, p), math.nan, dtype=FLOAT, device="cpu")
    output = filter_exact_diffuse(y, values, max_work=max_work)
    return dict(mean=output["predicted"], covariance=output["innovation_covariance"],
                state_mean=output["prior_mean"], state_covariance=output["prior_covariance"],
                next_mean=output["next_mean"], next_covariance=output["next_covariance"],
                parameter_uncertainty=False, conditioning="fixed equations and learned diffuse initial state",
                workspace=output["workspace"])


# Public fixed-system adapter. Registration is additive and owned by registry
# integration; the existing known/stationary estimator is not rerouted here.
RESULT_SCHEMA = "openecon.sspace.diffuse-result.v1"


def _index_admission(index):
    """Probe scalar sizes on the caller-owned index before creating label codes."""
    import pandas as pd
    from decimal import Decimal

    size = 0

    def add(amount):
        nonlocal size
        size += amount
        if size > MAX_INDEX_BYTES:
            raise AnalysisError("metadata_limit", "Typed row/calendar identities exceed their 2 MiB envelope.")

    def label(value, depth=0):
        if depth > 16:
            raise AnalysisError("metadata_limit", "Index tuple nesting exceeds 16 levels.")
        add(48)
        if isinstance(value, tuple):
            if len(value) > 16:
                raise AnalysisError("metadata_limit", "Index tuples admit at most 16 components.")
            for item in value:
                label(item, depth+1)
        elif isinstance(value, (str, bytes)):
            if len(value) > 1024:
                raise AnalysisError("metadata_limit", "Index strings/bytes admit at most 1024 characters.")
            add(6*len(value))
        elif isinstance(value, Integral) and int(value).bit_length() > 256:
            raise AnalysisError("metadata_limit", "Index integers exceed the 256-bit envelope.")
        elif isinstance(value, Decimal):
            parts = value.as_tuple()
            if len(parts.digits) > 256 or isinstance(parts.exponent, int) and abs(parts.exponent) > 10000:
                raise AnalysisError("metadata_limit", "Index decimals exceed the bounded scalar envelope.")
        elif isinstance(value, pd.Interval):
            label(value.left, depth+1)
            label(value.right, depth+1)

    def visit(value):
        if len(value) > 20000 or value.nlevels > 8:
            raise AnalysisError("metadata_limit", "Index levels exceed their declared resident bounds.")
        if isinstance(value, pd.MultiIndex):
            add(16*len(value)*value.nlevels)
            for name in value.names:
                label(name)
            for level in value.levels:
                visit(level)
        elif isinstance(value, pd.CategoricalIndex):
            add(16*len(value))
            label(value.name)
            visit(value.categories)
        else:
            label(value.name)
            for item in value:
                label(item)

    visit(index)


def _resident_probe(data, names, time):
    """Admit projected resident columns before DataFrame construction/copying."""
    import numpy as np
    import pandas as pd

    columns = names + ([time] if time is not None else [])
    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 1024:
            raise AnalysisError("invalid_data", "Resident DataFrames admit at most 1024 source columns; project declared responses/calendar first.")
        if not data.columns.is_unique or any(name not in data.columns for name in columns):
            raise AnalysisError("invalid_data", "Supply distinct existing response/calendar columns.")
        n = len(data)
        projected = data
    elif isinstance(data, Mapping):
        if any(name not in data for name in columns):
            raise AnalysisError("invalid_data", "Supply all declared response/calendar columns.")
        projected = {name: data[name] for name in columns}
        if any(not isinstance(value, (list, tuple, range, pd.Series, pd.Index, np.ndarray))
               or isinstance(value, np.ndarray) and value.ndim != 1 for value in projected.values()):
            raise AnalysisError("invalid_data", "Mapping columns must be bounded resident one-dimensional arrays; iterators are unsupported.")
        lengths = [len(value) for value in projected.values()]
        if len(set(lengths)) != 1:
            raise AnalysisError("invalid_data", "Resident response/calendar columns need identical lengths.")
        n = lengths[0]
        # A Series-backed mapping can align on a different, much larger union.
        series = [value for value in projected.values() if isinstance(value, pd.Series)]
        if series and any(not series[0].index.equals(value.index) for value in series[1:]):
            raise AnalysisError("invalid_data", "Resident Series columns must have identical row identities.")
        if series:
            _index_admission(series[0].index)
    else:
        raise AnalysisError("unsupported_input", "Use a resident DataFrame or mapping of bounded columns; Dataset collection and generators are unsupported.")
    if not 1 <= n <= 20000:
        raise AnalysisError("invalid_data", "Exact diffuse data admit 1..20000 original dates.")
    if isinstance(data, pd.DataFrame):
        _index_admission(data.index)
    observed = 0
    for name in names:
        column = projected[name]
        for value in column:
            if value is None or value is pd.NA or value is pd.NaT:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise AnalysisError("non_numeric_column", "Gaussian measurements require finite real numeric values or missing values.")
            try:
                converted = float(value)
            except (OverflowError, ValueError):
                raise AnalysisError("precision_loss", "Measurements exceed the finite float64 domain.") from None
            if math.isnan(converted):
                continue
            if not math.isfinite(converted):
                raise AnalysisError("non_numeric_column", "Infinite Gaussian measurements are not missing values.")
            if isinstance(value, Integral) and int(converted) != int(value):
                raise AnalysisError("precision_loss", "Integer measurements must be exactly representable in float64; source values cannot be rounded silently.")
            observed += 1
    if time is not None:
        # Index construction is downstream of work admission; probe source
        # scalar metadata here so huge strings cannot enter that constructor.
        for value in projected[time]:
            if isinstance(value, (str, bytes)) and len(value) > 1024:
                raise AnalysisError("metadata_limit", "Calendar source strings exceed 1024 characters.")
    return n, observed, projected


def _index_record(index):
    import pandas as pd
    from openecon.econometrics.postest.index_codec import encode

    _index_admission(index)
    if isinstance(index, pd.MultiIndex):
        result = dict(kind="multi", names=[encode(v) for v in index.names],
                    levels=[_index_record(level) for level in index.levels],
                    codes=[codes.tolist() for codes in index.codes])
        _json_admission(result, limit=MAX_INDEX_BYTES)
        return result
    labels = [encode(value) for value in index]
    if any(len(value) > 4096 for value in labels):
        raise AnalysisError("invalid_index", "An encoded row identity exceeds 4096 characters.")
    result = dict(kind="index", name=encode(index.name), dtype=str(index.dtype), labels=labels)
    if isinstance(index, pd.RangeIndex):
        result.update(kind="range", start=index.start, stop=index.stop, step=index.step)
    elif isinstance(index, pd.CategoricalIndex):
        result.update(kind="category", categories=[encode(value) for value in index.categories], ordered=index.ordered)
    _json_admission(result, limit=MAX_INDEX_BYTES)
    return result


def _restore_index(record, n, *, _level=False):
    import pandas as pd
    from openecon.econometrics.postest.index_codec import decode, encode

    try:
        _json_admission(record, limit=MAX_INDEX_BYTES)
        if not isinstance(record, dict):
            raise ValueError("Invalid index object")
        kind = record.get("kind")
        if kind == "multi":
            if (_level or set(record) != {"kind", "names", "levels", "codes"}
                    or not isinstance(record["levels"], list) or not isinstance(record["names"], list)
                    or not isinstance(record["codes"], list)
                    or not 1 <= len(record["levels"]) <= 8 or len(record["names"]) != len(record["levels"])
                    or len(record["codes"]) != len(record["levels"])):
                raise ValueError("Invalid multi-index geometry")
            levels = [_restore_index(level, len(level.get("labels", [])), _level=True) for level in record["levels"]]
            if any(len(code) != n or any(type(v) is not int or not -1 <= v < len(level) for v in code)
                   for code, level in zip(record["codes"], levels)):
                raise ValueError("Invalid multi-index codes")
            index = pd.MultiIndex(levels=levels, codes=record["codes"], names=[decode(value) for value in record["names"]])
        else:
            keys = {"kind", "name", "dtype", "labels"}
            keys |= {"start", "stop", "step"} if kind == "range" else {"categories", "ordered"} if kind == "category" else set()
            if (kind not in {"index", "range", "category"} or set(record) != keys
                    or not isinstance(record["labels"], list) or len(record["labels"]) != n
                    or any(not isinstance(v, str) or len(v) > 4096 for v in record["labels"])):
                raise ValueError("Invalid index geometry")
            labels = [decode(value) for value in record["labels"]]
            if any(encode(value) != encoded for value, encoded in zip(labels, record["labels"])):
                raise ValueError("Noncanonical index scalar")
            name = decode(record["name"])
            if kind == "range":
                if any(type(record[k]) is not int for k in ("start", "stop", "step")) or not record["step"]:
                    raise ValueError("Invalid range geometry")
                index = pd.RangeIndex(record["start"], record["stop"], record["step"], name=name)
            elif kind == "category":
                if (type(record["ordered"]) is not bool or not isinstance(record["categories"], list)
                        or len(record["categories"]) > 20000):
                    raise ValueError("Invalid categorical ordering")
                index = pd.CategoricalIndex(labels, categories=[decode(v) for v in record["categories"]],
                                             ordered=record["ordered"], name=name)
            else:
                index = pd.Index(labels, dtype=record["dtype"], name=name, tupleize_cols=False)
        if len(index) != n or _index_record(index) != record:
            raise ValueError("Index metadata do not reproduce their identities")
        return index
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError, RecursionError) as error:
        raise AnalysisError("invalid_state", "Saved index identities or schema are invalid.") from error


def _calendar(index):
    import pandas as pd

    if index.isna().any() or not index.is_unique or not index.is_monotonic_increasing:
        raise AnalysisError("time_gaps", "Calendar dates must be complete, distinct and in chronological input order.")
    if isinstance(index, pd.DatetimeIndex):
        frequency = pd.infer_freq(index) if len(index) >= 3 else None
        if frequency is None:
            raise AnalysisError("time_gaps", "Datetime periods need at least three dates with an inferable regular calendar frequency.")
        return dict(kind="datetime", frequency=frequency)
    if pd.api.types.is_numeric_dtype(index.dtype) and not pd.api.types.is_bool_dtype(index.dtype):
        if pd.api.types.is_complex_dtype(index.dtype) or not all(math.isfinite(value) for value in index):
            raise AnalysisError("invalid_time", "Calendar periods must be finite real numbers.")
        integer = pd.api.types.is_integer_dtype(index.dtype)
        differences = [(int(index[i])-int(index[i-1])) if integer else float(index[i]-index[i-1])
                       for i in range(1, len(index))]
        if not differences:
            raise AnalysisError("time_gaps", "Numeric calendars need at least two dates to declare their spacing.")
        if differences and (differences[0] <= 0 or any(value != differences[0] for value in differences)):
            raise AnalysisError("time_gaps", "Numeric periods must have constant positive spacing.")
        return dict(kind="numeric", step=differences[0] if differences else None)
    raise AnalysisError("invalid_time", "Use real numeric periods or a DatetimeIndex-compatible time column.")


def _response_dtype_admission(dtypes, rows):
    import pandas as pd

    for j, descriptor in enumerate(dtypes):
        if not isinstance(descriptor, str) or not descriptor or len(descriptor) > 80:
            raise AnalysisError("invalid_state", "Saved response dtype descriptors must be bounded real numeric dtypes.")
        try:
            dtype = pd.api.types.pandas_dtype(descriptor)
            if (str(dtype) != descriptor or not pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)):
                raise ValueError("Invalid real numeric dtype")
            values = [row[j] for row in rows]
            # Every saved finite value must survive the declared original
            # dtype exactly. This detects forged integer/floating precision,
            # nonnullable integer missingness, and incompatible extensions.
            recovered = pd.Series(values, dtype=dtype).astype("float64").tolist()
            if any(value is not None and (not math.isfinite(other) or value != other)
                   for value, other in zip(values, recovered)):
                raise ValueError("Values differ in the declared response dtype")
        except (ValueError, TypeError, OverflowError) as error:
            raise AnalysisError("invalid_state", "Saved measurements do not reproduce their declared real numeric source dtype.") from error


def _public_tables(record):
    from openecon.econometrics.core import table
    from openecon.engines.distributions import normal_isf

    core, source = record["state"], record["source"]
    output = core["output"]
    n, m = len(core["y"]), len(core["system"]["a0"])
    index = _restore_index(source["index"], n)
    critical = float(normal_isf(source["alpha"]/2))
    tables = {}
    for kind in ("filtered", "smoothed") if core["smoothing"] else ("filtered",):
        rows = []
        for t in range(n):
            rank = output["filtered_diffuse_rank"][t] if kind == "filtered" else 0
            for i, name in enumerate(source["state_names"]):
                variance = output[kind+"_covariance"][t][i][i]
                # An improper filtered state has no finite variance/interval.
                finite = not rank or output["filtered_diffuse_covariance"][t][i][i] == 0
                sd = math.sqrt(variance) if finite and variance >= 0 else None
                estimate = output[kind][t][i] if finite else None
                rows.append([t, name, estimate, sd, variance if finite else None,
                             estimate-critical*sd if sd is not None else None,
                             estimate+critical*sd if sd is not None else None, rank])
        repeated = index.repeat(m)
        tables[kind] = table(rows, columns=["position", "state", "mean", "conditional_sd", "conditional_variance", "conditional_lo", "conditional_hi", "diffuse_rank"], index=repeated)
        tables[kind].index = repeated
    tables["likelihood"] = table({"position": list(range(n)), "log_likelihood": output["log_likelihood_contributions"],
        "observed_cells": output["observed_count"], "diffuse_cells": output["diffuse_observation_count"],
        "proper_cells": output["proper_observation_count"], "remaining_diffuse_rank": output["filtered_diffuse_rank"]}, index=index)
    tables["likelihood"].index = index
    if source["time"] is not None:
        calendar = _restore_index(source["time"], n)
        for name, frame in tables.items():
            frame.insert(0, "time", calendar.repeat(m) if name != "likelihood" else calendar)
    notes = ["Caller-declared fixed Gaussian equations; no estimated coefficient SE, df, p-value or fit convergence claim.",
             "State/noise uncertainty conditions on the fixed equations and all retained observations; parameter uncertainty is absent.",
             output["likelihood_normalization"], "Full cross-state/cross-date and disturbance covariance is retained in the versioned state."]
    return DiffuseResult(tables, title="Exact diffuse Gaussian state space", diffuse_result=record,
                         log_likelihood=output["log_likelihood"], n_periods=n, observed_cells=sum(output["observed_count"]),
                         sample_positions=list(range(n)), alpha=source["alpha"], parameter_uncertainty=False, notes=notes)


class DiffuseResult(TableSet):
    """Real console/export tables plus the complete fixed-system posterior state."""

    @resident_cpu
    def to_json(self):
        record = self.attrs["diffuse_result"]
        _json_export_admission(record, limit=MAX_STATE_BYTES, operation="exact diffuse complete JSON export")
        diffuse_restore(self)
        return json.dumps(record, sort_keys=True, allow_nan=False)

    @resident_cpu
    def __deepcopy__(self, memo=None):
        cells = sum(len(frame) * (len(frame.columns) + 2) for frame in self.values())
        _state_copy_admission(
            {"attrs": self.attrs, "table_attrs": [frame.attrs for frame in self.values()]},
            operation="exact diffuse complete result deep copy",
            extra_bytes=256 * cells,
        )
        diffuse_restore(self)
        return DiffuseResult(
            {name: frame.copy(deep=True) for name, frame in self.items()},
            title=self.title,
            **copy.deepcopy(self.attrs, memo),
        )


def sspace_diffuse(data, y, *, system, time=None, state_names=None, smoothing=True,
                   missing="mask", alpha=.05, max_work=MAX_WORK) -> DiffuseResult:
    """Evaluate a caller-declared exact-diffuse Gaussian system on resident data.

    Dates are never sorted or deleted. Complete chronological calendars and
    typed row identities are saved. All equations are fixed, including any
    supplied schedules; weights, Dataset collection and GPU are unsupported.
    """
    import pandas as pd
    from openecon.analysis import _coerce_frame

    names = [y] if isinstance(y, str) else list(y) if isinstance(y, (list, tuple)) else []
    if (not 1 <= len(names) <= 8
            or any(not isinstance(name, str) or not name or len(name) > 200 for name in names)
            or len(set(names)) != len(names)
            or missing not in {"mask", "raise"}
            or type(smoothing) is not bool):
        raise AnalysisError("invalid_data", "Use distinct response columns on bounded resident data with missing='mask' or 'raise'.")
    if type(alpha) not in {float, int} or not 0 < alpha < 1:
        raise AnalysisError("invalid_alpha", "alpha must be a real number strictly between zero and one.")
    if not isinstance(system, Mapping) or set(system)-_KEYS or _KEYS-set(system)-{"c", "d"}:
        raise AnalysisError("invalid_system", "Declare a0/P_inf/P_star and fixed T/Z/Q/H; c/d default to zero.")
    initial = system["a0"]
    if (not isinstance(initial, (list, torch.Tensor)) or isinstance(initial, torch.Tensor) and initial.ndim != 1
            or not 1 <= len(initial) <= 16):
        raise AnalysisError("invalid_system", "a0 must declare 1..16 states.")
    if time is not None and (not isinstance(time, str) or not time or len(time) > 200 or time in names):
        raise AnalysisError("invalid_time", "time must name a separate bounded calendar column.")
    m, p = len(initial), len(names)
    n, observed, admitted_data = _resident_probe(data, names, time)
    if missing == "raise" and observed != n*p:
        raise AnalysisError("missing_values", "Missing measurements require missing='mask'.")
    _plan(n, p, m, max_work, smoothing=smoothing, observed=observed)
    frame = _coerce_frame(admitted_data)
    if len(frame) != n:
        raise AnalysisError("invalid_data", "Resident coercion changed the admitted date geometry.")
    labels = list(state_names) if isinstance(state_names, (list, tuple)) else [f"state{i+1}" for i in range(m)] if state_names is None else []
    if len(labels) != m or any(not isinstance(v, str) or not v or len(v) > 200 for v in labels) or len(set(labels)) != m:
        raise AnalysisError("invalid_system", "Supply exactly one distinct bounded label per state.")
    for name in names:
        dtype = frame[name].dtype
        if (not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_complex_dtype(dtype)):
            raise AnalysisError("non_numeric_column", "Gaussian measurements require real numeric response columns.")
    index_record = _index_record(frame.index)
    calendar_record = calendar_rule = None
    if time is not None:
        if not isinstance(time, str) or time not in frame.columns or time in names:
            raise AnalysisError("invalid_time", "time must name a separate complete calendar column.")
        calendar_index = pd.Index(frame[time])
        calendar_rule = _calendar(calendar_index)
        calendar_record = _index_record(calendar_index)
    shapes = {"a0": (m,), "P_star": (m, m), "P_inf": (m, m), "T": (m, m),
              "Z": (p, m), "Q": (m, m), "H": (p, p), "c": (m,), "d": (p,)}
    normalized = dict(system, c=system.get("c", [0.]*m), d=system.get("d", [0.]*p))
    values = {}
    for key, shape in shapes.items():
        raw = normalized[key]
        if isinstance(raw, torch.Tensor):
            if raw.device.type != "cpu" or raw.dtype != FLOAT:
                raise AnalysisError("invalid_system", "System tensors must explicitly use CPU float64.")
            values[key] = raw
        else:
            if not (_raw_array(raw, shape) or key not in {"a0", "P_star", "P_inf"} and _raw_array(raw, (n, *shape))):
                raise AnalysisError("invalid_system", f"{key} has an invalid exact-diffuse array geometry.")
            values[key] = torch.tensor(raw, dtype=FLOAT, device="cpu")
    measurements = torch.tensor(frame[names].to_numpy(dtype="float64", na_value=math.nan), dtype=FLOAT, device="cpu")
    state = save_diffuse_state(measurements, values, smoothing=smoothing, max_work=max_work)
    record = dict(schema=RESULT_SCHEMA, state=state, source=dict(responses=names, state_names=labels,
                  index=index_record, time=calendar_record, calendar=calendar_rule, missing=missing,
                  response_dtypes=[str(frame[name].dtype) for name in names], alpha=float(alpha)))
    _json_admission(record)
    record["sha256"] = _digest(record)
    return _public_tables(record)


def diffuse_restore(result) -> DiffuseResult:
    """Restore complete source identities, equations and conditional posterior without estimation."""
    try:
        value = result.attrs["diffuse_result"] if isinstance(result, TableSet) else result
        record = _load_record(value)
        if set(record) != {"schema", "state", "source", "sha256"} or record["schema"] != RESULT_SCHEMA:
            raise ValueError("Invalid result schema")
        payload = {key: value for key, value in record.items() if key != "sha256"}
        if record["sha256"] != _digest(payload):
            raise ValueError("Invalid result digest")
        source = record["source"]
        if set(source) != {"responses", "state_names", "index", "time", "calendar", "missing", "response_dtypes", "alpha"}:
            raise ValueError("Invalid source schema")
        replay = restore_diffuse_state(record["state"])
        n, p = len(replay["y"]), len(replay["y"][0])
        m = len(replay["system"]["a0"])
        for key, length in (("responses", p), ("state_names", m), ("response_dtypes", p)):
            if not isinstance(source[key], list) or len(source[key]) != length or any(not isinstance(v, str) for v in source[key]):
                raise ValueError("Invalid source labels")
        if (len(set(source["responses"])) != p or len(set(source["state_names"])) != m
                or any(not v or len(v) > 200 for key in ("responses", "state_names") for v in source[key])
                or source["missing"] not in {"mask", "raise"}
                or source["missing"] == "raise" and any(v is None for row in replay["y"] for v in row)
                or type(source["alpha"]) not in {float, int} or not 0 < source["alpha"] < 1):
            raise ValueError("Invalid source metadata")
        _response_dtype_admission(source["response_dtypes"], replay["y"])
        _restore_index(source["index"], n)
        if source["time"] is not None:
            if _digest(_calendar(_restore_index(source["time"], n))) != _digest(source["calendar"]):
                raise ValueError("Invalid calendar metadata")
        elif source["calendar"] is not None:
            raise ValueError("Unexpected calendar metadata")
        # Preserve the recorded admission plan; replay has independently checked
        # that the current ambient capacity suffices for every numerical buffer.
        return _public_tables(record)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, RecursionError) as error:
        raise AnalysisError("invalid_state", "Saved exact-diffuse result/source metadata are invalid.") from error


def _future_labels(values, steps):
    import numpy as np
    import pandas as pd

    if (not isinstance(values, (list, tuple, range, pd.Index, pd.Series, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1 or len(values) != steps):
        raise AnalysisError("invalid_time", "Supply exactly one bounded resident future identity per equation; iterators are unsupported.")
    if isinstance(values, pd.Index):
        _index_admission(values)
    else:
        size = 0
        def scalar(item, depth=0):
            nonlocal size
            if depth > 16:
                raise AnalysisError("metadata_limit", "Future identity tuple nesting exceeds 16 levels.")
            size += 48
            if isinstance(item, (str, bytes)):
                if len(item) > 1024:
                    raise AnalysisError("metadata_limit", "Future identity strings exceed 1024 characters.")
                size += 6*len(item)
            elif isinstance(item, Integral) and int(item).bit_length() > 256:
                raise AnalysisError("metadata_limit", "Future identity integers exceed 256 bits.")
            elif isinstance(item, tuple):
                if len(item) > 16:
                    raise AnalysisError("metadata_limit", "Future identity tuples exceed 16 components.")
                for child in item:
                    scalar(child, depth+1)
            if size > MAX_INDEX_BYTES:
                raise AnalysisError("metadata_limit", "Future typed identities exceed 2 MiB.")
        for item in values:
            scalar(item)
        if isinstance(values, pd.Series) and isinstance(values.dtype, pd.CategoricalDtype):
            _index_admission(values.cat.categories)
    dates = pd.Index(values)
    _index_admission(dates)
    return dates


def diffuse_forecast(result, *, future_system, future_time=None, max_work=MAX_WORK) -> TableSet:
    """Conditional future measurement moments with explicit equations and calendar."""
    import pandas as pd
    from openecon.econometrics.core import table
    from openecon.engines.distributions import normal_isf

    restored = diffuse_restore(result)
    record = restored.attrs["diffuse_result"]
    source, core = record["source"], record["state"]
    if not isinstance(future_system, dict) or set(future_system) != {"T", "Z", "Q", "H", "c", "d"}:
        raise AnalysisError("invalid_system", "Declare every future equation path T/Z/Q/H/c/d.")
    raw_Z = future_system["Z"]
    if not isinstance(raw_Z, (list, torch.Tensor)) or not 1 <= len(raw_Z) <= 10000:
        raise AnalysisError("invalid_system", "Future Z must declare 1..10000 forecast dates.")
    steps, p, m = len(raw_Z), len(source["responses"]), len(source["state_names"])
    _plan(steps, p, m, max_work)
    shapes = {"T": (m, m), "Z": (p, m), "Q": (m, m), "H": (p, p), "c": (m,), "d": (p,)}
    values = {}
    for key, shape in shapes.items():
        raw = future_system[key]
        if isinstance(raw, torch.Tensor):
            values[key] = raw
        elif _raw_array(raw, (steps, *shape)):
            values[key] = torch.tensor(raw, dtype=FLOAT, device="cpu")
        else:
            raise AnalysisError("invalid_system", f"Future {key} requires exactly {steps} correctly shaped arrays.")
    if source["time"] is not None:
        historical = _restore_index(source["time"], len(core["y"]))
        if future_time is None:
            raise AnalysisError("invalid_time", "Supply explicit future dates for a saved calendar model.")
        dates = _future_labels(future_time, steps)
        combined = historical.append(dates)
        if _calendar(combined) != source["calendar"]:
            raise AnalysisError("time_gaps", "Future dates must continue the original calendar without a gap.")
    else:
        dates = pd.RangeIndex(len(core["y"]), len(core["y"])+steps) if future_time is None else _future_labels(future_time, steps)
        if len(dates) != steps:
            raise AnalysisError("invalid_time", "Supply exactly one future label per forecast date.")
    moments = forecast_exact_diffuse(core, values, max_work=max_work)
    critical = float(normal_isf(source["alpha"]/2))
    rows = []
    for t in range(steps):
        for j, name in enumerate(source["responses"]):
            mean, variance = float(moments["mean"][t, j]), float(moments["covariance"][t, j, j])
            sd = math.sqrt(variance) if variance >= 0 else None
            rows.append([t+1, name, mean, variance, sd, mean-critical*sd if sd is not None else None,
                         mean+critical*sd if sd is not None else None])
    forecast = table(rows, columns=["horizon", "response", "mean", "conditional_variance", "conditional_sd", "conditional_lo", "conditional_hi"], index=dates.repeat(p))
    forecast.index = dates.repeat(p)
    return TableSet({"forecast": forecast}, title="Conditional exact-diffuse state-space forecast",
                    full_moments=_json_value(moments), future_equations=_json_value(values),
                    future_index=_index_record(dates), source_result=record, alpha=source["alpha"], parameter_uncertainty=False,
                    notes=["Conditional state/process/measurement uncertainty; fixed equations and learned diffuse initial conditions.",
                           "No estimated-parameter uncertainty or simultaneous cross-horizon band."])
