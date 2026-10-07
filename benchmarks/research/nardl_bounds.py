"""PRIVATE research-only joint-sign NARDL null bootstrap; not a testing API.

The three-null construction is an explicitly unvalidated nonlinear extension
inspired by Bertelli/Vacca/Zoia (2022), section 3. Their linear ARDL validity
does not establish validity for dependent, sign-generated partial sums. Public
OpenEcon bounds guards are never imported or modified here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math
from numbers import Complex, Real
import struct
from typing import Any

import torch
from torch import Tensor

NULLS = ("joint_levels", "adjustment", "explanatory_levels")
TAILS = ("upper", "lower", "upper")
_FILTER = None
_FILTER_SOURCE = '''
def outcome(prefix: Tensor, forcing: Tensor, phi: Tensor, total: int) -> Tensor:
    hold = prefix.size(1)
    values = torch.empty((forcing.size(0), total), dtype=torch.float64, device="cpu")
    values[:, :hold] = prefix
    for t in range(hold, total):
        value = forcing[:, t-hold]
        for j in range(phi.numel()):
            value = value + phi[j]*values[:, t-j-1]
        values[:, t] = value
    return values
'''


class ResearchFailure(ValueError):
    """A whole-call failure, never an instruction to replace bootstrap draws."""

    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = {"whole_call_invalid": True, "invalid_draws_replaced": 0, **details}


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or maximum is not None and value > maximum):
        raise ResearchFailure("invalid_input", f"Invalid {name}.", value=repr(value))
    return value


def _contains_complex(value):
    # Inspect dtype/scalars before conversion: casting a complex Tensor to
    # float64 silently loses its imaginary part, even when that part is zero.
    # Do not probe real sequences via default-dtype tensors: that would reject
    # large otherwise-valid integer inputs or round float64 values to float32.
    if isinstance(value, Tensor):
        return value.is_complex()
    if isinstance(value, Complex) and not isinstance(value, Real):
        return True
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    if kind == "c":
        return True
    if kind == "O" and callable(getattr(value, "tolist", None)):
        return _contains_complex(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_complex(item) for item in value)
    return False


def _vector(value, name):
    if _contains_complex(value):
        raise ResearchFailure("invalid_input", f"{name} must be real; complex inputs are not supported.")
    try:
        values = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ResearchFailure("invalid_input", f"{name} must be a numeric vector.") from error
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ResearchFailure("invalid_input", f"{name} must be a finite one-dimensional vector.")
    return values


def _finite(values, stage):
    good = torch.isfinite(values).reshape(values.shape[0], -1).all(1)
    if not bool(good.all()):
        raise ResearchFailure("nonfinite", f"Nonfinite {stage}.", local_indices=(~good).nonzero()[:, 0].tolist())


def partial_sums(raw: Tensor, anchors: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Split actual raw increments; preserve optional original-block anchors."""
    raw = raw if raw.ndim == 2 else raw[None, :]
    increments = torch.cat((torch.zeros_like(raw[:, :1]), torch.diff(raw, dim=1)), dim=1)
    positive, negative = increments.clamp_min(0).cumsum(1), increments.clamp_max(0).cumsum(1)
    if anchors is not None:
        positive, negative = positive + anchors[:, :1], negative + anchors[:, 1:]
    return positive, negative


def restrictions(p: int, qp: int, qn: int, null: str | None):
    """Levels-space affine restrictions; adjustment is sum(phi)=1, not 0."""
    parameters = 1 + p + qp + 1 + qn + 1
    all_rows = torch.zeros((3, parameters), dtype=torch.float64, device="cpu")
    all_rows[0, 1:1+p] = 1
    all_rows[1, 1+p:2+p+qp] = 1
    all_rows[2, 2+p+qp:] = 1
    target = torch.tensor([1., 0., 0.], dtype=torch.float64, device="cpu")
    indices = {None: [], "joint_levels": [0, 1, 2], "adjustment": [0],
               "explanatory_levels": [1, 2]}[null]
    return all_rows[indices], target[indices]


def design(y: Tensor, positive: Tensor, negative: Tensor, p: int, qp: int, qn: int):
    y = y if y.ndim == 2 else y[None, :]
    positive = positive if positive.ndim == 2 else positive[None, :]
    negative = negative if negative.ndim == 2 else negative[None, :]
    total, hold = y.shape[1], max(p, qp, qn)
    columns = [torch.ones_like(y[:, hold:])]
    columns += [y[:, hold-i:total-i] for i in range(1, p+1)]
    for values, q in ((positive, qp), (negative, qn)):
        columns += [values[:, hold-i:total-i] for i in range(q+1)]
    return torch.stack(columns, dim=2), y[:, hold:]


@dataclass
class Fit:
    coefficients: Tensor
    covariance: Tensor
    residuals: Tensor
    df: int
    restriction_matrix: Tensor
    restriction_target: Tensor
    condition: Tensor


def _fit_affine_unchecked(matrix: Tensor, target: Tensor, restriction: Tensor, imposed: Tensor) -> Fit:
    """Profile unrestricted case-III intercept; scaled QR in an affine space."""
    _finite(matrix, "lag design")
    _finite(target, "outcome")
    batches, rows, parameters = matrix.shape
    slopes = parameters - 1
    r = restriction[:, 1:]
    if restriction.shape[0]:
        _, singular, vh = torch.linalg.svd(r, full_matrices=True)
        tolerance = max(r.shape) * torch.finfo(torch.float64).eps * singular[0]
        if not bool((singular > tolerance).all()):
            raise ResearchFailure("invalid_restriction", "Affine restrictions must be independent.")
        anchor = torch.linalg.lstsq(r, imposed[:, None], driver="gelsd").solution[:, 0]
        basis = vh[r.shape[0]:].T
    else:
        anchor = torch.zeros(slopes, dtype=torch.float64, device="cpu")
        basis = torch.eye(slopes, dtype=torch.float64, device="cpu")
    free = basis.shape[1] + 1
    df = rows - free
    if df < 2:
        raise ResearchFailure("insufficient_rows", "At least two residual degrees of freedom required.")
    z = matrix[:, :, 1:]
    zmean, ymean = z.mean(1), target.mean(1)
    centered, centered_y = z - zmean[:, None, :], target - ymean[:, None]
    w, rhs = centered @ basis, centered_y - centered @ anchor
    scales = torch.linalg.vector_norm(w, dim=1) / math.sqrt(rows)
    good = (scales > 0).all(1)
    if not bool(good.all()):
        raise ResearchFailure("rank", "A free lag-design column is constant.", local_indices=(~good).nonzero()[:, 0].tolist())
    scaled = w / scales[:, None, :]
    singular = torch.linalg.svdvals(scaled)
    threshold = max(rows, basis.shape[1]) * torch.finfo(torch.float64).eps * singular[:, 0]
    good = singular[:, -1] > threshold
    if not bool(good.all()):
        raise ResearchFailure("rank", "Free lag design is numerically rank deficient.", local_indices=(~good).nonzero()[:, 0].tolist())
    q, upper = torch.linalg.qr(scaled, mode="reduced")
    coordinates = torch.linalg.solve_triangular(upper, (q.transpose(1, 2) @ rhs[:, :, None]), upper=True)[:, :, 0] / scales
    bslopes = anchor[None, :] + coordinates @ basis.T
    intercept = ymean - (zmean*bslopes).sum(1)
    coefficients = torch.cat((intercept[:, None], bslopes), dim=1)
    residuals = centered_y - (centered*bslopes[:, None, :]).sum(2)
    sigma = residuals.square().sum(1) / df
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        bad = ~(torch.isfinite(sigma) & (sigma > 0))
        raise ResearchFailure("variance", "Residual variances must be positive and finite.", local_indices=bad.nonzero()[:, 0].tolist())
    inverse = torch.linalg.solve_triangular(upper, torch.eye(basis.shape[1], dtype=torch.float64, device="cpu").expand(batches, -1, -1), upper=True)
    inverse = inverse / scales[:, :, None]
    maps = torch.cat((-(zmean @ basis)[:, None, :], basis.expand(batches, -1, -1)), dim=1)
    factor = maps @ inverse
    covariance = sigma[:, None, None] * (factor @ factor.transpose(1, 2))
    covariance[:, 0, 0] += sigma / rows
    _finite(coefficients, "coefficient refit")
    _finite(covariance, "covariance refit")
    return Fit(coefficients, covariance, residuals, df, restriction, imposed, singular[:, 0]/singular[:, -1])


def fit_affine(matrix: Tensor, target: Tensor, restriction: Tensor, imposed: Tensor) -> Fit:
    """Convert native factorization failures into honest whole-call failures."""
    try:
        return _fit_affine_unchecked(matrix, target, restriction, imposed)
    except RuntimeError as error:
        raise ResearchFailure("torch_numerical", "Native affine refit failed.",
                              stage="affine_refit", local_indices=list(range(matrix.shape[0])),
                              native_error=str(error)) from error


def statistics(fit: Fit, p: int, qp: int, qn: int):
    rows, targets = restrictions(p, qp, qn, "joint_levels")
    difference = fit.coefficients @ rows.T - targets[None, :]
    covariance = rows[None, :, :] @ fit.covariance @ rows.T[None, :, :]
    diag = covariance.diagonal(dim1=1, dim2=2)
    good = (torch.isfinite(diag) & (diag > 0)).all(1)
    if not bool(good.all()):
        raise ResearchFailure("statistic_variance", "Levels contrasts have invalid variances.", local_indices=(~good).nonzero()[:, 0].tolist())
    scale = diag.sqrt()

    def f(indices):
        cov = covariance[:, indices][:, :, indices]
        sd = scale[:, indices]
        corr = cov / sd[:, :, None] / sd[:, None, :]
        delta = difference[:, indices] / sd
        try:
            factor, information = torch.linalg.cholesky_ex(corr)
            if bool((information != 0).any()):
                raise ResearchFailure("statistic_rank", "Levels contrast covariance is singular.", local_indices=(information != 0).nonzero()[:, 0].tolist())
            solved = torch.cholesky_solve(delta[:, :, None], factor)[:, :, 0]
        except RuntimeError as error:
            raise ResearchFailure("statistic_rank", "Levels contrast factorization failed.") from error
        return (delta*solved).sum(1) / len(indices)

    output = torch.stack((f([0, 1, 2]), difference[:, 0]/scale[:, 0], f([1, 2])), dim=1)
    _finite(output, "test statistic")
    return output


def _filter(prefix, forcing, phi, total):
    global _FILTER
    try:
        if _FILTER is None:
            _FILTER = torch.jit.CompilationUnit(_FILTER_SOURCE)
        return _FILTER.outcome(prefix, forcing, phi, total)
    except RuntimeError as error:
        raise ResearchFailure("torch_numerical", "Native outcome recursion failed.",
                              stage="native_recursion", local_indices=list(range(prefix.shape[0])),
                              native_error=str(error)) from error


def generate_y(prefix, positive, negative, coefficients, p, qp, qn, errors):
    total, hold = positive.shape[1], max(p, qp, qn)
    forcing = coefficients[0].expand(len(prefix), total-hold).clone()
    position = 1+p
    for values, q in ((positive, qp), (negative, qn)):
        for lag in range(q+1):
            forcing += coefficients[position] * values[:, hold-lag:total-lag]
            position += 1
    forcing += errors
    result = _filter(prefix, forcing, coefficients[1:1+p], total)
    _finite(result, "recursive outcome")
    return result


def _fingerprint(y, x):
    values = y.tolist() + x.tolist()
    return hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest()


def _validate(y, x, p, qp, qn, replications, seed, memory_mb, batch_size, time):
    y, x = _vector(y, "y"), _vector(x, "x")
    if len(y) != len(x) or len(y) > 10000:
        raise ResearchFailure("invalid_input", "Equal-length vectors with at most 10,000 ordered rows required.")
    for value, name in ((p, "p"), (qp, "q_positive"), (qn, "q_negative")):
        _integer(value, name, 1, 4)
    _integer(replications, "replications", 3, 20000)
    _integer(seed, "seed", 0, 2**63-1)
    _integer(memory_mb, "memory_mb", 1, 512)
    _integer(batch_size, "batch_size", 1, 20000)
    hold, parameters = max(p, qp, qn), 1+p+qp+qn+2
    rows = len(y)-hold
    if rows <= parameters+2:
        raise ResearchFailure("insufficient_rows", "Too few effective observations for unrestricted levels refits.")
    if time is not None:
        try:
            axis = torch.as_tensor(time, device="cpu")
        except (TypeError, ValueError, RuntimeError) as error:
            raise ResearchFailure("invalid_time", "Time must be an aligned integer sequence.") from error
        if (axis.ndim != 1 or len(axis) != len(y) or axis.dtype == torch.bool
                or axis.is_floating_point() or axis.is_complex()):
            raise ResearchFailure("invalid_time", "Time must be an aligned integer sequence.")
        keys = axis.tolist()
        if (not all(isinstance(key, int) and not isinstance(key, bool) for key in keys)
                or any(right-left != 1 for left, right in zip(keys, keys[1:]))):
            raise ResearchFailure("invalid_time", "Rows must be ordered consecutive integer periods; no sorting or dropping occurs.")
    work = 3*replications*rows*parameters**2
    if work > 300_000_000:
        raise ResearchFailure("work_limit", "Research refit work exceeds the declared 300 million element-operation proxy.", estimated_work=work)
    # Input copies plus raw/sign/y paths, lag designs, centered/scaled QR and
    # covariance factors. This is a conservative tensor estimate, not RSS.
    per_draw = 8*(len(y)*(18+8*parameters)+parameters**2*16) + 8*rows
    base = 8*(len(y)*(30+4*parameters)+replications*12)
    available = memory_mb*1024**2-base
    capacity = min(batch_size, available//per_draw)
    if capacity < 1:
        raise ResearchFailure("memory_limit", "Even one research bootstrap draw exceeds the tensor workspace estimate.", base_bytes=base, per_draw_bytes=per_draw)
    return y, x, hold, rows, parameters, int(capacity), base+int(capacity)*per_draw


def _tail_report(observed, draws, tail, alpha):
    events = draws >= observed if tail == "upper" else draws <= observed
    hits, count = int(events.sum()), len(draws)
    # Plus one is our explicit finite-Monte-Carlo convention, not a theorem
    # asserting validity of the proposed nonlinear bootstrap.
    estimate = (hits+1)/(count+1)
    z, rate = 1.959963984540054, hits/count
    denominator = 1+z*z/count
    center = (rate+z*z/(2*count))/denominator
    half = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/denominator
    ordered = draws.sort().values
    rank = math.ceil((1-alpha)*count)-1 if tail == "upper" else max(0, math.ceil(alpha*count)-1)
    return {"tail": tail, "observed_statistic": float(observed), "extreme_draws": hits,
            "replications": count, "research_tail_probability_plus_one": estimate,
            "tail_event_wilson_95": [max(0., center-half), min(1., center+half)],
            "research_order_statistic": float(ordered[rank]), "zero_based_order_rank": rank,
            "research_reject_at_alpha": estimate <= alpha,
            "monte_carlo_standard_error": math.sqrt(estimate*(1-estimate)/(count+1))}


def research_bootstrap(y: Any, x: Any, *, p=1, q_positive=1, q_negative=1,
                       replications=199, seed=0, batch_size=32, memory_mb=64,
                       initial="fixed_prefix", recenter="pool", residual_scale="df",
                       alpha=.05, time=None, retain_paths=False) -> dict:
    """Unvalidated PRIVATE research distributions, never public inference.

    ``recenter='draw'`` follows the per-resampled-vector centering in BVZ(21–22),
    while ``pool`` uses centered fitted pools. Both are predeclared research
    choices; no linear-paper claim is transferred to this nonlinear experiment.
    Original contiguous-block initialization keeps ORIGINAL partial-sum anchors.
    """
    if (not isinstance(initial, str) or initial not in {"fixed_prefix", "original_block"}
            or not isinstance(recenter, str) or recenter not in {"pool", "draw"}
            or not isinstance(residual_scale, str) or residual_scale not in {"none", "df"}):
        raise ResearchFailure("invalid_input", "Unknown research construction option.")
    if not isinstance(retain_paths, bool) or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < .5:
        raise ResearchFailure("invalid_input", "Invalid research reporting option.")
    with torch.device("cpu"), torch.inference_mode():
        y, x, hold, rows, parameters, batch, estimated_bytes = _validate(y, x, p, q_positive, q_negative, replications, seed, memory_mb, batch_size, time)
        if retain_paths and replications*(4*len(y)+3*rows)*3*8 > memory_mb*1024**2/2:
            raise ResearchFailure("memory_limit", "Retained research paths exceed half the declared workspace budget.")
        positive, negative = partial_sums(x)
        matrix, target = design(y, positive, negative, p, q_positive, q_negative)
        unrestricted = fit_affine(matrix, target, *restrictions(p, q_positive, q_negative, None))
        observed = statistics(unrestricted, p, q_positive, q_negative)[0]
        dx = x[hold:]-x[hold-1:-1]
        drift = dx.mean()
        marginal = dx-drift
        if not bool((marginal.square().sum() > 0)):
            raise ResearchFailure("marginal_variance", "Declared iid raw increments require positive innovation variance.")
        fits, innovations, metadata = {}, {}, {}
        for null in NULLS:
            try:
                fitted = fit_affine(matrix, target, *restrictions(p, q_positive, q_negative, null))
            except ResearchFailure as error:
                error.details.update(null=null, stage="original_restricted_refit", whole_call_invalid=True,
                                     invalid_draws_replaced=0)
                raise
            residual = fitted.residuals[0]-fitted.residuals[0].mean()
            correlation = float((residual*marginal).sum() / (torch.linalg.vector_norm(residual)*torch.linalg.vector_norm(marginal)))
            if abs(correlation) > 1e-8:
                raise ResearchFailure("conditioning", "Current signed differences failed to orthogonalize the raw innovation pool.", null=null, correlation=correlation)
            yscale = math.sqrt(rows/fitted.df) if residual_scale == "df" else 1.
            xscale = math.sqrt(rows/(rows-1)) if residual_scale == "df" else 1.
            fits[null] = fitted
            innovations[null] = torch.stack((residual*yscale, marginal*xscale), dim=1)
            metadata[null] = {"restriction_matrix": fitted.restriction_matrix.tolist(),
                "restriction_target": fitted.restriction_target.tolist(), "restricted_coefficients": fitted.coefficients[0].tolist(),
                "free_parameters": rows-fitted.df, "residual_df": fitted.df,
                "restriction_residual": (fitted.restriction_matrix @ fitted.coefficients[0]-fitted.restriction_target).tolist(),
                "pool_innovation_correlation": correlation, "conditional_scale": yscale, "marginal_scale": xscale}
        generator = torch.Generator(device="cpu").manual_seed(seed)
        starts = (torch.randint(len(y)-hold+1, (replications,), generator=generator, device="cpu")
                  if initial == "original_block" else torch.zeros(replications, dtype=torch.long, device="cpu"))
        draws = {null: torch.empty((replications, 3), dtype=torch.float64, device="cpu") for null in NULLS}
        paths = {null: [] for null in NULLS}
        index_hash = hashlib.sha256()
        for first in range(0, replications, batch):
            last = min(first+batch, replications)
            indices = torch.randint(rows, (last-first, rows), generator=generator, device="cpu")
            index_hash.update(struct.pack(f"<{indices.numel()}q", *indices.flatten().tolist()))
            prefix_positions = starts[first:last, None]+torch.arange(hold, device="cpu")[None, :]
            raw_prefix, y_prefix = x[prefix_positions], y[prefix_positions]
            anchors = torch.stack((positive[0, starts[first:last]], negative[0, starts[first:last]]), dim=1)
            for null in NULLS:
                paired = innovations[null][indices]
                if recenter == "draw":
                    paired = paired-paired.mean(1, keepdim=True)
                try:
                    raw = torch.cat((raw_prefix, raw_prefix[:, -1:]+(drift+paired[:, :, 1]).cumsum(1)), dim=1)
                    xp, xn = partial_sums(raw, anchors)
                    simulated = generate_y(y_prefix, xp, xn, fits[null].coefficients[0], p, q_positive, q_negative, paired[:, :, 0])
                    xm, yy = design(simulated, xp, xn, p, q_positive, q_negative)
                    fitted = fit_affine(xm, yy, *restrictions(p, q_positive, q_negative, None))
                    draws[null][first:last] = statistics(fitted, p, q_positive, q_negative)
                except (ResearchFailure, RuntimeError) as error:
                    if isinstance(error, RuntimeError):
                        error = ResearchFailure("torch_numerical", "Native bootstrap batch failed.", native_error=str(error))
                    local = error.details.pop("local_indices", list(range(last-first)))
                    error.details.update(null=null, replicate_indices=[first+i for i in local], completed_batches=first//batch,
                                         completed_all_null_replications=first,
                                         invalid_draws_replaced=0, whole_call_invalid=True)
                    raise error
                if retain_paths:
                    paths[null].append({"raw": raw.tolist(), "positive": xp.tolist(), "negative": xn.tolist(), "y": simulated.tolist(), "paired_innovations": paired.tolist(), "indices": indices.tolist()})
        tests = {null: _tail_report(observed[i], draws[null][:, i], TAILS[i], alpha) for i, null in enumerate(NULLS)}
        profile = {null: tests[null]["research_reject_at_alpha"] for null in NULLS}
        return {"private_research_only": True, "inferential_validity_established": False,
            "public_bounds_inference_unchanged": True, "construction": "test-specific joint raw iid increment/sign-reconstruction extension",
            "source": "https://arxiv.org/abs/2204.04939", "linear_paper_does_not_validate_nonlinear_extension": True,
            "rows_original": len(y), "rows_fitted": rows, "holdback": hold, "lags": [p, q_positive, q_negative],
            "case": 3, "assumed_raw_order": "I(1), supplied assumption; no pretest or marginal cointegration fit",
            "marginal": {"model": "raw iid increments with drift, no outcome feedback", "drift": float(drift), "innovation_variance": float(marginal.square().sum()/(rows-1))},
            "innovation_pool_periods_zero_based": [hold, len(y)-1],
            "statistic_covariance": "classical unrestricted residual variance with actual residual df; full affine levels contrasts; no reference F/t distribution used",
            "initial_policy": initial, "initial_block_starts": starts.tolist(), "partial_sum_anchors": "original sample block-start signed levels; fixed prefix starts at zero",
            "initial_prefix_length": hold,
            "recenter": recenter, "finite_sample_scale": residual_scale, "paired_common_indices_for_all_nulls": True,
            "seed": seed, "index_stream_sha256": index_hash.hexdigest(), "input_sha256": _fingerprint(y, x),
            "rng_scope": "local CPU generator; caller RNG untouched", "alpha": alpha,
            "quantile_convention": "explicit zero-based ordered statistic; ties count as extreme for plus-one tail estimates",
            "workspace": {"estimated_tensor_bytes": estimated_bytes, "memory_mb": memory_mb, "batch_size_requested": batch_size, "batch_size_actual": batch, "excludes_private_blas_allocator_and_python_json": True},
            "nulls": metadata, "observed_statistics": observed.tolist(), "tests": tests,
            "research_rejection_profile": profile, "research_all_three_rejected": all(profile.values()),
            "draw_statistics": {null: value.tolist() for null, value in draws.items()}, "paths": paths if retain_paths else None,
            "failure_policy": "Any nonfinite/rank/variance failure invalidates whole call; no rejection/replacement of draws, no stationary-root filter",
            "native_recursion": "trusted fixed TorchScript loop, batched across replications; no companion power squaring",
            "limitations": ["experimental nonlinear joint DGP; size/power calibration not established", "fixed or sampled original prefix is explicitly recorded", "one raw predictor, fixed lags, case III, iid increments only", "no GPU, streaming, symmetry, controls, lag reselection, serial innovations or automatic cointegration decision"]}


def conditional_f_benchmark(y, x, *, p=1, q_positive=1, q_negative=1,
                            replications=199, seed=0, batch_size=32, memory_mb=64,
                            residual_scale="none", alpha=.05, retain_paths=False):
    """Private fixed-x F mechanics benchmark, distinct from three-null research.

    Follows the original NARDL experiment's restricted outcome equation plus
    UNRESTRICTED residual pool and known signed paths/initial outcome values.
    Fitted empirical data, centered residuals, a locally seeded RNG and the
    finite-Monte-Carlo tail convention here are explicit benchmark choices;
    this is not an exact reproduction of the original paper's simulation.
    """
    if (not isinstance(residual_scale, str) or residual_scale not in {"none", "df"}
            or not isinstance(retain_paths, bool) or isinstance(alpha, bool)
            or not isinstance(alpha, (int, float)) or not 0 < alpha < .5):
        raise ResearchFailure("invalid_input", "Invalid conditional benchmark option.")
    with torch.device("cpu"), torch.inference_mode():
        y, x, hold, rows, _, batch, estimated = _validate(y, x, p, q_positive, q_negative, replications, seed, memory_mb, batch_size, None)
        if retain_paths and replications*(len(y)+2*rows)*8 > memory_mb*1024**2/2:
            raise ResearchFailure("memory_limit", "Retained benchmark paths exceed half the declared tensor budget.")
        positive, negative = partial_sums(x)
        matrix, target = design(y, positive, negative, p, q_positive, q_negative)
        unrestricted = fit_affine(matrix, target, *restrictions(p, q_positive, q_negative, None))
        null = fit_affine(matrix, target, *restrictions(p, q_positive, q_negative, "joint_levels"))
        observed = statistics(unrestricted, p, q_positive, q_negative)[0, 0]
        residuals = unrestricted.residuals[0]-unrestricted.residuals[0].mean()
        scale = math.sqrt(rows/unrestricted.df) if residual_scale == "df" else 1.
        residuals = residuals*scale
        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = torch.empty(replications, dtype=torch.float64, device="cpu")
        paths, digest = [], hashlib.sha256()
        for first in range(0, replications, batch):
            last = min(first+batch, replications)
            indices = torch.randint(rows, (last-first, rows), generator=generator, device="cpu")
            digest.update(struct.pack(f"<{indices.numel()}q", *indices.flatten().tolist()))
            errors = residuals[indices]
            try:
                simulated = generate_y(y[:hold].expand(last-first, -1), positive.expand(last-first, -1),
                                       negative.expand(last-first, -1), null.coefficients[0],
                                       p, q_positive, q_negative, errors)
                design_x, target_y = design(simulated, positive.expand(last-first, -1),
                                             negative.expand(last-first, -1), p, q_positive, q_negative)
                fitted = fit_affine(design_x, target_y, *restrictions(p, q_positive, q_negative, None))
                output[first:last] = statistics(fitted, p, q_positive, q_negative)[:, 0]
            except (ResearchFailure, RuntimeError) as error:
                if isinstance(error, RuntimeError):
                    error = ResearchFailure("torch_numerical", "Native conditional benchmark batch failed.", native_error=str(error))
                local = error.details.pop("local_indices", list(range(last-first)))
                error.details.update(replicate_indices=[first+i for i in local], invalid_draws_replaced=0,
                                     whole_call_invalid=True, benchmark="fixed_x_f")
                raise error
            if retain_paths:
                paths.append({"y": simulated.tolist(), "errors": errors.tolist(), "indices": indices.tolist()})
        return {"private_research_only": True, "inferential_validity_established": False,
                "public_bounds_inference_unchanged": True,
                "construction": "original-paper-style fixed-known-x conditional F mechanics benchmark",
                "source": "https://doi.org/10.1007/978-1-4899-8008-3_9",
                "exact_paper_simulation_reproduction": False,
                "rows_original": len(y), "rows_fitted": rows, "holdback": hold,
                "case": 3, "lags": [p, q_positive, q_negative], "seed": seed,
                "initial_policy": "original fixed observed y prefix; original x and signed levels held fixed",
                "residual_pool": "UNRESTRICTED centered empirical residuals",
                "finite_sample_scale": residual_scale, "residual_scale_factor": scale,
                "restriction_matrix": null.restriction_matrix.tolist(),
                "restriction_target": null.restriction_target.tolist(),
                "restricted_coefficients": null.coefficients[0].tolist(),
                "input_sha256": _fingerprint(y, x), "index_stream_sha256": digest.hexdigest(),
                "rng_scope": "local CPU generator; caller RNG untouched",
                "workspace": {"estimated_tensor_bytes": estimated, "memory_mb": memory_mb,
                              "batch_size_actual": batch, "excludes_private_blas_allocator_and_python_json": True},
                "f_joint": _tail_report(observed, output, "upper", alpha),
                "draw_statistics": output.tolist(), "paths": paths if retain_paths else None,
                "failure_policy": "Whole call invalid on any failed draw; no draw replacement",
                "limitations": ["conditional F benchmark alone does not diagnose both degeneracies",
                                "no t-adjustment or explanatory-levels decision supplied by this benchmark",
                                "known explanatory path; not joint raw/sign bootstrap inference"]}
