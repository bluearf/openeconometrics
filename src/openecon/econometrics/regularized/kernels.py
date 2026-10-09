"""Float64 squared-error and local-regression kernels; no external solver."""

from __future__ import annotations

import math
from statistics import NormalDist

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError


def work_guard(work: int, maximum: int, operation: str) -> None:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise AnalysisError("invalid_work_limit", "max_work must be a positive integer.")
    if work > maximum:
        raise AnalysisError(
            "work_limit",
            f"{operation} plans {work:,} scalar operations, above max_work={maximum:,}; no rows or candidates were truncated.",
        )


def solver_work(n: int, p: int, iterations: int, ratio: float) -> int:
    """Conservative arithmetic estimate, distinguishing ridge from CD sweeps."""
    return n * p * p + p**3 if ratio == 0 else n * p * iterations


def folds(n: int, k: int, seed: int) -> Tensor:
    if k > n // 2:
        raise AnalysisError(
            "insufficient_sample", "Each validation fold needs at least two observations."
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    assignment = torch.empty(n, dtype=torch.int64)
    assignment[order] = torch.arange(n) % k
    return assignment


def transform(x: Tensor, y: Tensor, intercept: bool, standardize: bool):
    n, p = x.shape
    center = x.mean(0) if intercept else torch.zeros(p, dtype=torch.float64)
    y_center = y.mean() if intercept else torch.zeros((), dtype=torch.float64)
    z = x - center
    scale = z.square().mean(0).sqrt() if standardize else torch.ones(p, dtype=torch.float64)
    # A constant predictor is left as a zero column with an intercept. With no
    # intercept its RMS is nonzero and it remains a legitimate penalized term.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    scaled, centered_y = z / scale, y - y_center
    if not all(
        bool(torch.isfinite(v).all()) for v in (center, y_center, scale, scaled, centered_y)
    ):
        raise AnalysisError(
            "numerical_failure",
            "Centering/scaling exceeds finite float64 arithmetic; rescale the inputs explicitly.",
        )
    return scaled, centered_y, center, scale, y_center


def objective(z: Tensor, y: Tensor, b: Tensor, lam: float, ratio: float, load: Tensor,
              factors: Tensor | None = None) -> Tensor:
    factors = torch.ones_like(b) if factors is None else factors
    return 0.5 * (y - z @ b).square().mean() + lam * (
        ratio * (factors * load * b.abs()).sum()
        + 0.5 * (1 - ratio) * (factors * b.square()).sum()
    )


def kkt(z: Tensor, y: Tensor, b: Tensor, lam: float, ratio: float, load: Tensor,
        factors: Tensor | None = None) -> float:
    factors = torch.ones_like(b) if factors is None else factors
    g = z.T @ (z @ b - y) / len(y) + lam * (1 - ratio) * factors * b
    threshold = lam * ratio * factors * load
    residual = torch.where(
        b != 0, (g + threshold * b.sign()).abs(), (g.abs() - threshold).clamp_min(0)
    )
    return float(residual.max()) if len(b) else 0.0


def solve(
    z: Tensor,
    y: Tensor,
    lam: float,
    ratio: float,
    load: Tensor,
    max_iterations: int,
    tolerance: float,
    initial: Tensor | None = None,
    *,
    factors: Tensor | None = None,
):
    n, p = z.shape
    factors = torch.ones(p, dtype=torch.float64) if factors is None else factors
    if ratio == 0 or lam == 0 or not bool((factors > 0).any()):
        if lam == 0 or not bool((factors > 0).any()):
            b = torch.linalg.lstsq(z, y, driver="gelsd").solution
        else:
            gram = z.T @ z / n + lam * torch.diag(factors)
            if not bool(torch.isfinite(gram).all()):
                raise AnalysisError("numerical_failure", "Factor-weighted normal equations exceed finite float64 arithmetic.")
            b = torch.linalg.solve(gram, z.T @ y / n)
        iterations = 1
    else:
        b = torch.zeros(p, dtype=torch.float64) if initial is None else initial.clone()
        r = y - z @ b
        ridge_penalty = lam * (1 - ratio) * factors
        diagonal = z.square().mean(0) + ridge_penalty
        if (not bool(torch.isfinite(diagonal).all())
                or not bool(torch.isfinite(lam * ratio * factors * load).all())):
            raise AnalysisError("numerical_failure", "Factor-weighted penalties exceed finite float64 arithmetic.")
        limit = tolerance * max(1.0, float((z.T @ y / n).abs().max()))
        for iterations in range(1, max_iterations + 1):
            for j in range(p):
                old = b[j].clone()
                if float(diagonal[j]) == 0:
                    b[j] = 0
                else:
                    partial = torch.dot(z[:, j], r) / n + (diagonal[j] - ridge_penalty[j]) * old
                    b[j] = (
                        partial.sign()
                        * (partial.abs() - lam * ratio * factors[j] * load[j]).clamp_min(0)
                        / diagonal[j]
                    )
                r += z[:, j] * (old - b[j])
            if kkt(z, y, b, lam, ratio, load, factors) <= limit:
                break
        else:
            raise AnalysisError(
                "nonconvergence",
                f"Coordinate descent did not satisfy KKT conditions after {max_iterations} sweeps.",
            )
    violation = kkt(z, y, b, lam, ratio, load, factors)
    loss = float(objective(z, y, b, lam, ratio, load, factors))
    if not bool(torch.isfinite(b).all()) or not math.isfinite(violation) or not math.isfinite(loss):
        raise AnalysisError(
            "numerical_failure",
            "Regularized estimates or objective exceed finite float64 arithmetic.",
        )
    return b, {
        "iterations": iterations,
        "kkt_max": violation,
        "objective": loss,
    }


def penalty_factors(value, p: int) -> Tensor:
    """Validate explicit factors; booleans and implicit normalization are refused."""
    if value is None:
        return torch.ones(p, dtype=torch.float64)
    if (not isinstance(value, list) or len(value) != p
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise AnalysisError("invalid_penalty_factors", "penalty_factors needs one numeric factor per predictor.")
    factors = torch.tensor(value, dtype=torch.float64)
    if not bool(torch.isfinite(factors).all()) or bool((factors < 0).any()):
        raise AnalysisError("invalid_penalty_factors", "Penalty factors must be finite and nonnegative.")
    return factors


def forced_rank(z: Tensor, factors: Tensor) -> None:
    """A zero-factor block must have unique unpenalized training coefficients."""
    block = z[:, factors == 0]
    if block.shape[1] == 0:
        return
    singular = torch.linalg.svdvals(block)
    if (len(singular) < block.shape[1] or float(singular[0]) == 0
            or float(singular[-1]) <= torch.finfo(torch.float64).eps * 100 * max(block.shape) * float(singular[0])):
        raise AnalysisError("unidentified_forced_controls", "Zero-penalty controls must be jointly identified in every training sample, including CV folds.")


def path_values(z: Tensor, y: Tensor, ratio: float, options: dict,
                factors: Tensor | None = None) -> list[float]:
    given = options.get("lambda_path")
    if given is not None:
        if not given or len(given) > 200 or any(v < 0 or not math.isfinite(v) for v in given):
            raise AnalysisError(
                "invalid_penalty", "lambda_path needs 1..200 finite nonnegative values."
            )
        if len(set(given)) != len(given):
            raise AnalysisError("invalid_penalty", "lambda_path must not repeat values.")
        return sorted(map(float, given), reverse=True)
    factors = torch.ones(z.shape[1], dtype=torch.float64) if factors is None else factors
    penalized = factors > 0
    if not bool(penalized.any()):
        raise AnalysisError("invalid_penalty", "An automatic lambda path requires at least one positive penalty factor; use a fixed penalty or explicit path for an unpenalized model.")
    forced = z[:, ~penalized]
    residual = y if forced.shape[1] == 0 else y - forced @ torch.linalg.lstsq(forced, y, driver="gelsd").solution
    score = (z[:, penalized].T @ residual / len(y)).abs() / factors[penalized]
    top = float(score.max()) / max(ratio, 0.01)
    if not math.isfinite(top):
        raise AnalysisError(
            "numerical_failure", "Lambda-path score exceeds finite float64 arithmetic."
        )
    top = max(top, 1e-8)
    count, fraction = options.get("n_lambdas", 30), options.get("lambda_ratio", 0.001)
    return torch.logspace(
        math.log10(top), math.log10(top * fraction), count, dtype=torch.float64
    ).tolist()


def fit_penalized(
    x: Tensor, y: Tensor, *, ratio: float, intercept: bool, options: dict, seed: int | None = None
) -> dict:
    """Tuning, including scaling and loadings, uses only the supplied sample."""
    n, p = x.shape
    if n < 4 or p == 0:
        raise AnalysisError(
            "insufficient_sample",
            "Penalized regression needs at least four observations and one predictor.",
        )
    standardize = options.get("standardize", True)
    selection = options.get("selection", "cv")
    iterations, tolerance = options.get("max_iterations", 2000), options.get("tolerance", 1e-8)
    local_seed = options.get("seed", 1729) if seed is None else seed
    factors = penalty_factors(options.get("penalty_factors"), p)
    forced_count = int((factors == 0).sum())
    work_guard(n * p + n * forced_count**2 + forced_count**3,
               options.get("max_work", 2000000000), "penalty factors and forced-control identification")
    z, yc, center, scale, ycenter = transform(x, y, intercept, standardize)
    forced_rank(z, factors)
    svd_limit = (not bool((factors > 0).any())
                 or options.get("penalty") == 0
                 or (options.get("lambda_path") is not None and 0 in options["lambda_path"]))
    planned_solver = max(solver_work(n, p, iterations, ratio), n*p*p+p**3) if svd_limit else solver_work(n, p, iterations, ratio)
    load = torch.ones(p, dtype=torch.float64)
    record = {
        "selection": selection,
        "seed": local_seed,
        "intercept_unpenalized": intercept,
        "standardize": standardize,
        "scaling_source": "supplied training sample only",
        "penalty_factors": factors.tolist(),
        "unpenalized_indices": torch.where(factors == 0)[0].tolist(),
        "penalty_factor_convention": "literal factors multiply both L1 and L2 penalties; no factor normalization",
    }
    records, betas = [], []
    if selection == "fixed":
        if options.get("penalty") is None:
            raise AnalysisError("invalid_penalty", "selection='fixed' requires penalty.")
        values = [float(options["penalty"])]
        if options.get("lambda_path") is not None:
            values = path_values(z, yc, ratio, options, factors)
            if float(options["penalty"]) not in values:
                raise AnalysisError("invalid_penalty", "Fixed penalty must occur in lambda_path.")
        chosen = values.index(float(options["penalty"]))
    elif selection == "plugin":
        if bool((factors != 1).any()):
            raise AnalysisError("unsupported_selection", "Score plug-in selection requires unit penalty factors and no forced controls; custom factors support fixed or CV selection.")
        if options.get("penalty") is not None or options.get("lambda_path") is not None:
            raise AnalysisError(
                "invalid_penalty", "Plug-in selection determines penalty; omit penalty/lambda_path."
            )
        if ratio == 0:
            raise AnalysisError(
                "unsupported_selection",
                "The heteroskedastic score plug-in requires an L1 penalty; ridge supports fixed or CV selection.",
            )
        lam = (
            options.get("plugin_c", 1.1)
            * NormalDist().inv_cdf(1 - options.get("plugin_gamma", 0.1) / (2 * p))
            / (math.sqrt(n) * ratio)
        )
        work_guard(
            solver_work(n, p, iterations, ratio) * options.get("plugin_iterations", 15),
            options.get("max_work", 2000000000),
            "plug-in penalized regression",
        )
        residual = yc.clone()
        b = torch.zeros(p, dtype=torch.float64)
        history = []
        converged_loading = False
        for step in range(options.get("plugin_iterations", 15)):
            load = (z.square() * residual.square()[:, None]).mean(0).sqrt().clamp_min(1e-12)
            b, diag = solve(z, yc, lam, ratio, load, iterations, tolerance, b)
            new_residual = yc - z @ b
            history.append({"iteration": step + 1, "loadings": load.tolist(), **diag})
            newload = (z.square() * new_residual.square()[:, None]).mean(0).sqrt().clamp_min(1e-12)
            if float(((newload - load).abs() / load.clamp_min(1e-8)).max()) <= 1e-4:
                converged_loading = True
                residual = new_residual
                break
            residual = new_residual
        if not converged_loading:
            raise AnalysisError(
                "nonconvergence",
                "Heteroskedastic plug-in penalty loadings did not stabilize; increase plugin_iterations.",
            )
        values, chosen = [lam], 0
        records, betas = [{"penalty": lam, **diag}], [b]
        record.update(
            {
                "plugin_formula": "c*Phi^-1(1-gamma/(2p))/(sqrt(n)*l1_ratio)",
                "plugin_c": options.get("plugin_c", 1.1),
                "plugin_gamma": options.get("plugin_gamma", 0.1),
                "loading_history": history,
                "loadings_converged": converged_loading,
            }
        )
    elif selection == "cv":
        if options.get("penalty") is not None:
            raise AnalysisError(
                "invalid_penalty", "CV selects penalty from lambda_path; omit penalty."
            )
        automatic_grid = options.get("lambda_path") is None
        values = [] if automatic_grid else path_values(z, yc, ratio, options, factors)
        path_count = options.get("n_lambdas", 30) if automatic_grid else len(values)
        assignment = folds(n, options.get("folds", 5), local_seed)
        k = options.get("folds", 5)
        work_guard(
            planned_solver * path_count * (k + 1),
            options.get("max_work", 2000000000),
            "penalty-path CV",
        )
        error_sum = torch.zeros(path_count, dtype=torch.float64)
        fold_mse = []
        fold_paths = []
        fold_diagnostics = []
        for fold in range(k):
            train, test = assignment != fold, assignment == fold
            tz, ty, tc, ts, tmy = transform(x[train], y[train], intercept, standardize)
            forced_rank(tz, factors)
            b = None
            errors = []
            diagnostics = []
            # With no user grid, compare dimensionless geometric fractions.
            # Validation labels never determine a training fold's lambda_max.
            fold_values = path_values(tz, ty, ratio, options, factors) if automatic_grid else values
            fold_paths.append(fold_values)
            for index, lam in enumerate(fold_values):
                b, diag = solve(tz, ty, lam, ratio, load, iterations, tolerance, b, factors=factors)
                diagnostics.append(diag)
                squared = (y[test] - ((x[test] - tc) / ts @ b + tmy)).square()
                if not bool(torch.isfinite(squared).all()):
                    raise AnalysisError(
                        "numerical_failure",
                        "CV prediction error exceeds finite float64 arithmetic.",
                    )
                error_sum[index] += squared.sum()
                errors.append(float(squared.mean()))
            fold_mse.append(errors)
            fold_diagnostics.append(diagnostics)
        scores = error_sum / n
        chosen = int(scores.argmin())
        if automatic_grid:
            values = path_values(z, yc, ratio, options, factors)
        record.update(
            {
                "fold_assignments": assignment.tolist(),
                "fold_mse": fold_mse,
                "cv_mse": scores.tolist(),
                "cv_rule": "minimum pooled out-of-fold MSE; ties choose largest lambda",
                "folds": k,
                "fold_lambda_paths": fold_paths,
                "fold_path_diagnostics": fold_diagnostics,
                "cv_selector_units": "fraction of training-fold lambda_max"
                if automatic_grid
                else "absolute user-supplied lambda",
                "cv_lambda_fractions": [v / values[0] for v in values] if automatic_grid else None,
                "selected_lambda_fraction": values[chosen] / values[0] if automatic_grid else None,
                "final_refit_path_source": "full supplied training sample after fraction selection"
                if automatic_grid
                else "user-supplied fixed grid",
            }
        )
    else:
        raise AnalysisError("invalid_selection", "Unknown penalty selection.")
    if not records:
        work_guard(
            planned_solver * len(values),
            options.get("max_work", 2000000000),
            "penalty path",
        )
        b = None
        for lam in values:
            b, diag = solve(z, yc, lam, ratio, load, iterations, tolerance, b, factors=factors)
            betas.append(b.clone())
            records.append({"penalty": lam, **diag})
    b = betas[chosen]
    coefficient = b / scale
    constant = float(ycenter - center @ coefficient)
    fitted = x @ coefficient + constant
    if (
        not bool(torch.isfinite(coefficient).all())
        or not math.isfinite(constant)
        or not bool(torch.isfinite(fitted).all())
    ):
        raise AnalysisError(
            "numerical_failure", "Original-unit prediction exceeds finite float64 arithmetic."
        )
    record.update(
        {
            "l1_ratio": ratio,
            "selected_index": chosen,
            "selected_penalty": values[chosen],
            "lambda_path": values,
            "path_diagnostics": records,
            "loadings": load.tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "outcome_center": float(ycenter),
            "standardized_coefficients": b.tolist(),
            "coefficients": coefficient.tolist(),
            "constant": constant,
            "coefficient_path": [(bb / scale).tolist() for bb in betas],
            "constant_path": [float(ycenter - center @ (bb / scale)) for bb in betas],
            "objective_definition": "mean(residual^2)/2 + lambda*(l1_ratio*sum(penalty_factors*loadings*abs(beta_z)) + (1-l1_ratio)*sum(penalty_factors*beta_z^2)/2)",
        }
    )
    return {"fitted": fitted, "state": record, "coefficient": coefficient, "constant": constant}


def bandwidth_vector(value, p: int) -> Tensor:
    if isinstance(value, bool):
        raise AnalysisError("invalid_bandwidth", "Bandwidth must be finite and positive.")
    try:
        h = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        raise AnalysisError(
            "invalid_bandwidth", "Bandwidth must be a positive scalar or vector."
        ) from None
    if h.ndim == 0:
        h = h.repeat(p)
    if h.shape != (p,) or not bool(torch.isfinite(h).all()) or bool((h <= 0).any()):
        raise AnalysisError(
            "invalid_bandwidth", "Bandwidth needs one finite positive value per predictor."
        )
    return h


def local_predict(
    x: Tensor,
    y: Tensor,
    query: Tensor,
    h: Tensor,
    *,
    degree: int,
    kernel: str,
    support: str,
    min_effective: float,
    leave_out: Tensor | None = None,
):
    """One query at a time: O(np) storage, not a hidden dense n-by-n matrix."""
    low, high = x.min(0).values, x.max(0).values
    out = ((query < low) | (query > high)).any(1)
    if support == "raise" and bool(out.any()):
        raise AnalysisError(
            "outside_support",
            "Query lies outside the observed predictor bounding box; use support='extrapolate' explicitly.",
        )
    values, details = [], []
    for row, point in enumerate(query):
        u = (x - point) / h
        if not bool(torch.isfinite(u).all()):
            raise AnalysisError(
                "numerical_failure", "Bandwidth normalization exceeds finite float64 arithmetic."
            )
        if kernel == "gaussian":
            exponent = -0.5 * u.square().sum(1)
            if not bool(torch.isfinite(exponent).all()):
                raise AnalysisError(
                    "numerical_failure",
                    "Gaussian kernel distance exceeds finite float64 arithmetic.",
                )
            # Common rescaling preserves both estimates and effective counts,
            # avoiding underflow when extrapolation is explicitly requested.
            weights = torch.exp(exponent - exponent.max())
        else:
            if kernel == "epanechnikov":
                component = (1 - u.square()).clamp_min(0)
            elif kernel == "triangular":
                component = (1 - u.abs()).clamp_min(0)
            else:
                component = (u.abs() <= 1).to(torch.float64)
            weights = component.prod(1)
        if leave_out is not None:
            weights[int(leave_out[row])] = 0
        total = weights.sum()
        if float(total) <= 0:
            raise AnalysisError("empty_local_support", "No nonzero kernel support at a query.")
        effective = float(total.square() / weights.square().sum())
        if effective < min_effective - 1e-10:
            raise AnalysisError(
                "insufficient_local_support", "Effective local sample is below min_effective."
            )
        if degree == 0:
            value = weights @ y / total
            condition = None
        else:
            design = torch.cat((torch.ones(len(x), 1, dtype=torch.float64), u), dim=1)
            weighted = design * weights.sqrt()[:, None]
            singular = torch.linalg.svdvals(weighted)
            if len(singular) < design.shape[1] or float(singular[-1]) <= 1e-10 * float(singular[0]):
                raise AnalysisError(
                    "singular_local_design",
                    "Local-linear support does not identify every slope; widen bandwidth or reduce predictors.",
                )
            b = torch.linalg.lstsq(weighted, y * weights.sqrt(), driver="gelsd").solution
            value, condition = b[0], float(singular[0] / singular[-1])
        if not math.isfinite(float(value)):
            raise AnalysisError(
                "numerical_failure", "Local regression produced nonfinite prediction."
            )
        values.append(value)
        details.append(
            {
                "effective_n": effective,
                "nonzero_support": int((weights > 0).sum()),
                "boundary": bool(((point - low < h) | (high - point < h)).any()),
                "outside_bounding_box": bool(out[row]),
                "condition_number": condition,
            }
        )
    return torch.stack(values), details
