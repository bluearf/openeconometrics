"""Bounded Gaussian fixed-design OLS spatial diagnostic algebra.

The public adapter owns fitted-state and keyed-graph admission. This module
independently checks its numerical geometry and never estimates coefficients.
All buffers and random draws are explicit CPU float64; no third-party numerical
kernel, ridge, residual permutation, or discarded simulation is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_sf, f_sf, normal_sf
from openecon.resources import plan_workspace

TESTS = (
    "moran_normal",
    "moran_gaussian_mc",
    "lm_error",
    "lm_lag",
    "robust_lm_error",
    "robust_lm_lag",
    "lm_joint",
    "wx_f",
)
_LM = frozenset(TESTS[2:7])
_MAX_N = 512
_MAX_P = 32
_MAX_DRAWS = 100_000
_EPS = torch.finfo(torch.float64).eps


def _error(target: str, code: str, message: str) -> None:
    raise AnalysisError(f"spatial_{target}_{code}", f"{target}: {message}")


def resource_estimates(n: int, p: int, *, draws: int, mc: bool, wx: bool) -> dict:
    """Conservative cumulative scalar work and live owned-buffer reservation.

    Counts include QR/rank checks, dense projected moments, score geometry,
    independent WX directions and every simulated dense graph multiplication.
    This is a declared arithmetic proxy, not a timing or process-RSS promise.
    """
    b = draws if mc else 0
    work = 4 * n**3 + 8 * n**2 * p + 8 * n * p**2 + 4 * p**3
    if wx:
        work += 4 * n * p**2 + 4 * p**3
    work += b * (n * n + 4 * n * p + 16 * n)
    return {
        "work": work,
        "buffers": {
            "dense_graph_projection_moments_and_temporaries": 8 * 16 * n * n,
            "design_qr_rank_wx_and_temporaries": 8 * 24 * n * p,
            "small_factors_and_rank_checks": 8 * 16 * p * p,
            "row_vectors_and_simulation_scratch": 8 * 24 * n,
            "complete_gaussian_null_statistics": 8 * b,
        },
    }


def _integer(value, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        _error("diagnostics", "invalid_option", f"{name} must be an integer {low}..{high}.")
    return value


def _tensor(value, shape: tuple[int, ...], name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or tuple(value.shape) != shape
        or value.layout != torch.strided
    ):
        _error(
            "diagnostics",
            "invalid_geometry",
            f"{name} must be a CPU float64 tensor of shape {shape}.",
        )


def _finite(value: torch.Tensor, target: str, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        _error(target, "numerical_range", f"{name} is nonfinite in float64; rescale the inputs.")


def _project(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector - q @ (q.T @ vector)


def _basis(x: torch.Tensor, *, target: str) -> tuple[torch.Tensor, float, list[int]]:
    """An equilibrated basis of the unchanged design column space.

    With an explicit constant, shifting/centering other columns preserves that
    space and avoids attributing rank to large offsets. No columns are dropped.
    """
    scales = x.abs().amax(0)
    if bool((scales == 0).any()):
        _error(target, "rank", "Every requested design column must have positive scale.")
    z = x / scales
    constants = torch.nonzero((z == z[0]).all(0), as_tuple=False).flatten().tolist()
    if len(constants) > 1:
        _error(target, "rank", "More than one constant column is not identified.")
    if constants:
        constant = constants[0]
        # Shift first: close values can be subtracted without loss from a large
        # offset, then the mean acts on the bounded range rather than its level.
        z = z - z.amin(0)
        z = z - z.mean(0)
        z[:, constant] = 1.0
    norms = torch.linalg.vector_norm(z, dim=0)
    if bool((norms == 0).any()):
        _error(target, "rank", "The complete design has a constant or redundant direction.")
    z = z / norms
    q, r = torch.linalg.qr(z, mode="reduced")
    singular = torch.linalg.svdvals(r)
    largest, smallest = float(singular[0]), float(singular[-1])
    threshold = max(x.shape) * _EPS * largest * 128
    if smallest <= threshold:
        _error(
            target,
            "rank",
            "The complete requested design is numerically rank deficient; no directions are dropped.",
        )
    condition = largest / smallest
    if not math.isfinite(condition) or condition > 1e10:
        _error(
            target,
            "rank",
            "The equilibrated design is too ill-conditioned for resolved projection.",
        )
    _finite(q, target, "QR basis")
    return q, condition, constants


def _row(
    name: str,
    statistic: float,
    distribution: str,
    p_value: float,
    df_num: int | None = None,
    df_denom: int | None = None,
    **details,
) -> dict:
    if not math.isfinite(statistic) or not math.isfinite(p_value) or not 0 <= p_value <= 1:
        _error(
            name, "numerical_range", "The requested statistic or probability is not representable."
        )
    return {
        "test": name,
        "statistic": statistic,
        "distribution": distribution,
        "df_num": df_num,
        "df_denom": df_denom,
        "p_value": p_value,
        **details,
    }


def _normal_probability(z: float, alternative: str) -> float:
    if alternative == "greater":
        return float(normal_sf(z))
    if alternative == "less":
        return float(normal_sf(-z))
    return min(1.0, 2 * float(normal_sf(abs(z))))


def diagnose(
    x: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    *,
    tests: Sequence[str] = TESTS,
    wx_indices: Sequence[int] = (),
    alternative: str = "two-sided",
    draws: int = 9999,
    seed: int = 1729,
    max_work: int = 250_000_000,
    max_n: int = 512,
) -> dict:
    """Evaluate requested targets at admitted saved OLS coefficients.

    Any requested unresolved target raises a target-specific AnalysisError.
    Valid other targets remain available through an explicit requested subset.
    The caller, not this kernel, asserts independent Gaussian homoskedastic
    errors, a fixed exogenous graph and an unchanged complete fitted sample.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        _error("diagnostics", "invalid_geometry", "x must be a two-dimensional tensor.")
    n, p = x.shape
    max_n = _integer(max_n, 4, _MAX_N, "max_n")
    if not 4 <= n <= max_n or not 1 <= p <= _MAX_P or p >= n:
        _error("diagnostics", "dimension", "Require 4 <= n <= max_n <= 512 and 1 <= p <= 32 < n.")
    for value, shape, name in (
        (x, (n, p), "x"),
        (y, (n,), "y"),
        (beta, (p,), "beta"),
        (w, (n, n), "w"),
    ):
        _tensor(value, shape, name)
    if (
        not isinstance(tests, (list, tuple))
        or not tests
        or any(not isinstance(name, str) or name not in TESTS for name in tests)
        or len(set(tests)) != len(tests)
    ):
        _error("diagnostics", "invalid_option", "tests must select distinct supported targets.")
    requested = tuple(tests)
    if alternative not in ("two-sided", "greater", "less"):
        _error("diagnostics", "invalid_option", "alternative must be two-sided, greater or less.")
    draws = _integer(draws, 1, _MAX_DRAWS, "draws")
    seed = _integer(seed, 0, 2**63 - 1, "seed")
    max_work = _integer(max_work, 1, 10**12, "max_work")
    if (
        not isinstance(wx_indices, (list, tuple))
        or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < p for i in wx_indices)
        or len(set(wx_indices)) != len(wx_indices)
    ):
        _error("wx_f", "invalid_option", "wx_indices must select distinct design positions.")
    mc = "moran_gaussian_mc" in requested
    wx = "wx_f" in requested
    estimates = resource_estimates(n, p, draws=draws, mc=mc, wx=wx)
    if estimates["work"] > max_work:
        _error(
            "diagnostics",
            "work_limit",
            f"Requested cumulative work {estimates['work']} exceeds max_work={max_work}.",
        )
    plan = plan_workspace("OLS spatial diagnostics numerical buffers", estimates["buffers"])
    for value, name in ((x, "design"), (y, "outcome"), (beta, "coefficients"), (w, "weights")):
        _finite(value, "diagnostics", name)
    if bool((w < 0).any()) or bool((w.diagonal() != 0).any()):
        _error(
            "diagnostics",
            "graph",
            "The fixed graph must have nonnegative weights and a zero diagonal.",
        )
    weight_scale = float(w.abs().max())
    if not 1e-150 <= weight_scale <= 1e150:
        _error(
            "diagnostics",
            "weight_scale",
            "Require a nonzero graph with max weight in [1e-150, 1e150]; rescale W explicitly.",
        )
    wn = w / weight_scale
    q, condition, constants = _basis(x, target="diagnostics")
    if wx and (not wx_indices or any(i in constants for i in wx_indices)):
        _error(
            "wx_f",
            "rank",
            "Select at least one numeric slope; the intercept must never be spatially lagged.",
        )
    fitted = x @ beta
    residual = y - fitted
    _finite(fitted, "diagnostics", "saved fitted mean")
    _finite(residual, "diagnostics", "saved residual")
    residual_scale = float(residual.abs().max())
    if residual_scale == 0:
        _error("diagnostics", "variance", "The saved model has no positive residual variation.")
    e = residual / residual_scale
    sse = float(e @ e)
    sigma2 = sse / n
    orthogonal_loss = float(torch.linalg.vector_norm(q.T @ e))
    if orthogonal_loss > 1e-7 * math.sqrt(sse):
        _error(
            "diagnostics",
            "saved_fit",
            "The saved coefficients do not produce OLS residuals in the complete design.",
        )
    geometry = {
        "n": n,
        "p": p,
        "df_resid": n - p,
        "design_condition_number": condition,
        "projection_orthogonality_loss": orthogonal_loss,
        "residual_scale": residual_scale,
        "scaled_rss": sse,
        "scaled_sigma2_ml": sigma2,
        "weight_scale": weight_scale,
        "work_estimate": estimates["work"],
        "projection": "column-equilibrated thin QR; centered nonconstant columns when a constant is present",
    }
    result = {
        "tests": [],
        "fitted": fitted,
        "residuals": residual,
        "q": q,
        "geometry": geometry,
        "scores": None,
        "information": None,
        "null_statistics": torch.empty(0, dtype=torch.float64, device="cpu"),
        "resource_plan": plan.record(),
        "workspace_observed": {},
    }
    rows = {}
    buffers = {
        "normalized_weights": wn,
        "q": q,
        "fitted": fitted,
        "residuals": residual,
        "scaled_residuals": e,
    }

    if any(name.startswith("moran_") for name in requested):
        target = next(name for name in requested if name.startswith("moran_"))
        identity = torch.eye(n, dtype=torch.float64, device="cpu")
        m = identity - q @ q.T
        symmetric = (wn + wn.T) / 2
        projected = m @ symmetric @ m
        projected = (projected + projected.T) / 2
        nu = n - p
        s0 = float(wn.sum())
        multiplier = n / s0
        trace = float(projected.diagonal().sum())
        centered = projected - (trace / nu) * m
        spread = float(centered.square().sum())
        total = float(projected.square().sum())
        if not math.isfinite(spread) or spread <= 128 * (n + p) * _EPS * max(total, 1e-300):
            _error(
                target,
                "variance",
                "The projected Moran null variance is zero or numerically unresolved.",
            )
        expected = multiplier * trace / nu
        variance = multiplier**2 * 2 * spread / (nu * (nu + 2))
        if not math.isfinite(variance) or variance <= 0:
            _error(target, "variance", "The projected Moran null variance is not representable.")

        def statistic(vector):
            scale = float(vector.abs().max())
            if not math.isfinite(scale) or scale <= 0:
                _error(
                    target,
                    "draw",
                    "A Gaussian projected vector has no resolved positive scale; no draws are discarded.",
                )
            v = vector / scale
            denominator = float(v @ v)
            value = multiplier * float(v @ (wn @ v)) / denominator
            if not math.isfinite(value):
                _error(
                    target, "numerical_range", "Moran's ratio is nonfinite; no draws are discarded."
                )
            return value

        observed = statistic(e)
        z = (observed - expected) / math.sqrt(variance)
        moran_details = {
            "expected": expected,
            "variance": variance,
            "z": z,
            "alternative": alternative,
        }
        geometry.update(
            moran_multiplier=multiplier,
            moran_expected=expected,
            moran_variance=variance,
            projected_trace=trace,
            projected_centered_frobenius_squared=spread,
            residual_dimension=nu,
            normalized_s0=s0,
        )
        if "moran_normal" in requested:
            rows["moran_normal"] = _row(
                "moran_normal",
                observed,
                "normal",
                _normal_probability(z, alternative),
                **moran_details,
            )
        if mc:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            null = torch.empty(draws, dtype=torch.float64, device="cpu")
            for index in range(draws):
                innovation = torch.randn(
                    (n,), dtype=torch.float64, device="cpu", generator=generator
                )
                null[index] = statistic(_project(q, innovation))
            # Enlarging the inclusive extreme set is conservative. It protects
            # mathematical ties from QR/dot roundoff without random tie breaking.
            tolerance = 128 * (n + p) * _EPS * max(1.0, abs(observed), abs(expected))
            if alternative == "greater":
                extreme = int((null >= observed - tolerance).sum())
            elif alternative == "less":
                extreme = int((null <= observed + tolerance).sum())
            else:
                extreme = int(
                    ((null - expected).abs() >= abs(observed - expected) - tolerance).sum()
                )
            rows["moran_gaussian_mc"] = _row(
                "moran_gaussian_mc",
                observed,
                "conditional_gaussian_mc",
                (extreme + 1) / (draws + 1),
                **moran_details,
                draws=draws,
                seed=seed,
                extreme_count=extreme,
                tie_tolerance=tolerance,
                rng_policy="one independent torch.randn((n,), CPU float64) call per draw, local torch.Generator; Q projection; no residual permutation",
                calibration="inclusive plus-one rank probability under independent Gaussian homoskedastic fixed-X null",
            )
            result["null_statistics"] = null
            buffers["null_statistics"] = null
        buffers.update(
            projector=m,
            symmetric_weights=symmetric,
            projected_moran=projected,
            centered_projected_moran=centered,
        )

    if any(name in _LM for name in requested):
        target = next(name for name in requested if name in _LM)
        mu = fitted / residual_scale
        yn = y / residual_scale
        _finite(mu, target, "mean divided by residual scale")
        _finite(yn, target, "outcome divided by residual scale")
        a = float(e @ (wn @ e)) / sigma2
        b = float(e @ (wn @ yn)) / sigma2
        t = float(wn.square().sum() + (wn * wn.T).sum())
        v = _project(q, wn @ mu)
        h = float(v @ v) / sigma2
        d = t + h
        if not all(math.isfinite(value) for value in (a, b, t, h, d)) or t <= 0:
            _error(
                target,
                "information",
                "The full spatial score or information geometry is not representable.",
            )
        adjusted = [
            name for name in requested if name in ("robust_lm_error", "robust_lm_lag", "lm_joint")
        ]
        if adjusted and h <= 128 * (n + p) * _EPS * max(t, h):
            _error(
                adjusted[0],
                "identification",
                "Lag and error score directions cannot be separately resolved: M W fitted has zero or insufficient variation.",
            )
        scores = torch.tensor([b, a], dtype=torch.float64, device="cpu") * weight_scale
        information = (
            torch.tensor([[d, t], [t, t]], dtype=torch.float64, device="cpu")
            * weight_scale
            * weight_scale
        )
        _finite(scores, target, "full score vector in declared W units")
        _finite(information, target, "full score information in declared W units")
        if bool((information.diagonal() == 0).any()):
            _error(
                target,
                "information",
                "Nonzero information underflows in the declared W units; rescale W explicitly.",
            )
        result.update(scores=scores, information=information)
        geometry.update(
            score_order=["lag", "error"],
            normalized_error_score=a,
            normalized_lag_score=b,
            normalized_T=t,
            normalized_h=h,
            normalized_D=d,
            score_covariance="expected efficient Gaussian information after nuisance beta and variance elimination",
            score_variance_divisor="SSE/n",
            adjusted_semantics="robust to local competing spatial lag/error misspecification; not heteroskedasticity robust",
        )
        statistics = {"lm_error": a * a / t, "lm_lag": b * b / d}
        if adjusted:
            statistics.update(
                robust_lm_error=(a - (t / d) * b) ** 2 / (t * (h / d)),
                robust_lm_lag=(b - a) ** 2 / h,
                lm_joint=a * a / t + (b - a) ** 2 / h,
            )
        for name in requested:
            if name in _LM:
                statistic_value = statistics[name]
                df = 2 if name == "lm_joint" else 1
                rows[name] = _row(
                    name,
                    statistic_value,
                    "chi2",
                    float(chi2_sf(statistic_value, df)),
                    df_num=df,
                    reference="asymptotic Gaussian score reference",
                    adjustment="local competing spatial alternative"
                    if name.startswith("robust_")
                    else "none",
                )
        buffers.update(
            normalized_fitted=mu,
            normalized_outcome=yn,
            projected_lag_mean=v,
            score_vector=scores,
            score_information=information,
        )

    if wx:
        # Equilibrate original slopes before W multiplication. Each requested
        # direction keeps its identity; no intercept lag or rank reduction.
        slopes = x[:, list(wx_indices)]
        scales = slopes.abs().amax(0)
        if bool((scales == 0).any()):
            _error("wx_f", "rank", "A requested slope has zero scale.")
        lagged = wn @ (slopes / scales)
        directions = _project(q, lagged)
        direction_norms = torch.linalg.vector_norm(directions, dim=0)
        lag_norms = torch.linalg.vector_norm(lagged, dim=0)
        if bool((direction_norms <= 128 * (n + p) * _EPS * lag_norms.clamp_min(1e-300)).any()):
            _error(
                "wx_f",
                "rank",
                "A requested WX direction is already in the original design; no directions are dropped.",
            )
        wx_q, wx_condition, _ = _basis(directions, target="wx_f")
        r = len(wx_indices)
        df2 = n - p - r
        if df2 <= 0:
            _error(
                "wx_f", "degrees", "The augmented model needs positive residual degrees of freedom."
            )
        augmented_residual = _project(wx_q, e)
        unrestricted = float(augmented_residual @ augmented_residual)
        improvement = float((wx_q.T @ e).square().sum())
        if unrestricted <= 128 * (n + p + r) * _EPS * sse:
            _error(
                "wx_f",
                "variance",
                "The augmented residual variance is zero or numerically unresolved.",
            )
        statistic_value = (improvement / r) / (unrestricted / df2)
        rows["wx_f"] = _row(
            "wx_f",
            statistic_value,
            "F",
            float(f_sf(statistic_value, r, df2)),
            df_num=r,
            df_denom=df2,
            restricted_rss_scaled=sse,
            unrestricted_rss_scaled=unrestricted,
            added_rank=r,
            reference="finite-sample exact F under independent Gaussian homoskedastic errors, conditional on fixed X and W",
            tested_columns=list(wx_indices),
        )
        geometry.update(
            wx_rank=r,
            wx_condition_number=wx_condition,
            wx_df_resid=df2,
            wx_indices=list(wx_indices),
            wx_null="all requested fixed-W numeric slope lags have zero coefficients",
        )
        buffers.update(
            wx_slopes=slopes,
            wx_lags=lagged,
            wx_directions=directions,
            wx_q=wx_q,
            wx_augmented_residual=augmented_residual,
        )

    result["tests"] = [rows[name] for name in requested]
    result["workspace_observed"] = {
        name: {
            "shape": list(value.shape),
            "bytes": value.numel() * value.element_size(),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
        for name, value in buffers.items()
    }
    return result
