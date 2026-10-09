"""Gaussian multiple imputation: NIW data augmentation and monotone regression.

All samplers use a private CPU Torch generator and float64 native kernels. A
finite burn-in is a declared computation, never evidence of MCMC convergence.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import DEFAULT_MAX_WORK, admit, check_seed, integer, make_result


def _factor(matrix: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(matrix).all()):
        raise AnalysisError("mi_numerical_failure", f"{name} is non-finite; rescale the data or prior.")
    factor, status = torch.linalg.cholesky_ex(matrix)
    if int(status) != 0 or not bool(torch.isfinite(factor).all()):
        raise AnalysisError("mi_numerical_failure", f"{name} is not positive definite in double precision; rescale the data or prior.")
    return factor


def _prior(prior: Any, p: int) -> dict[str, Any]:
    if not isinstance(prior, Mapping) or set(prior) != {"mean", "kappa", "df", "scale"}:
        raise AnalysisError("invalid_prior", "mi_mvn requires an explicit proper NIW prior with mean, kappa, df, and scale in the original data units.")
    kappa, df = prior["kappa"], prior["df"]
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (kappa, df)) or kappa <= 0 or df <= p - 1:
        raise AnalysisError("invalid_prior", "A proper NIW prior requires finite kappa > 0 and df > p - 1.")
    for key in ("mean", "scale"):
        value = prior[key]
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise AnalysisError("unsupported_device", "MI prior tensors must be resident on CPU.")
        if isinstance(value, torch.Tensor) and (value.is_complex() or value.dtype == torch.bool):
            raise AnalysisError("invalid_prior", "NIW mean and scale tensors require real numeric values, excluding booleans.")
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        if shape is not None and shape != ((p,) if key == "mean" else (p, p)):
            raise AnalysisError("invalid_prior", f"NIW prior {key} has the wrong shape.")
        if shape is None:
            if not isinstance(value, (list, tuple)) or len(value) != p or (key == "scale" and any(not isinstance(row, (list, tuple)) or len(row) != p for row in value)):
                raise AnalysisError("invalid_prior", f"NIW prior {key} has the wrong shape.")
            cells = value if key == "mean" else [v for row in value for v in row]
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in cells):
                raise AnalysisError("invalid_prior", "NIW mean and scale require finite real numeric values.")
    mean = torch.as_tensor(prior["mean"], dtype=torch.float64, device="cpu").clone()
    scale = torch.as_tensor(prior["scale"], dtype=torch.float64, device="cpu").clone()
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()) or not torch.allclose(scale, scale.T, rtol=1e-13, atol=0):
        raise AnalysisError("invalid_prior", "NIW mean must be finite and scale must be finite and symmetric.")
    try:
        _factor(scale, "NIW prior scale")
    except AnalysisError as error:
        raise AnalysisError("invalid_prior", "NIW prior scale must be positive definite in double precision.") from error
    return {"mean": mean, "kappa": float(kappa), "df": float(df), "scale": scale}


def _niw_posterior(completed: torch.Tensor, prior: Mapping) -> dict[str, Any]:
    """Conjugate complete-data posterior: IW(df+n, scale+scatter+mean correction)."""
    n = completed.shape[0]
    average = completed.mean(0)
    centred = completed - average
    kappa = prior["kappa"] + n
    delta = average - prior["mean"]
    scale = prior["scale"] + centred.T @ centred + (prior["kappa"] * n / kappa) * torch.outer(delta, delta)
    return {"mean": (prior["kappa"] * prior["mean"] + n * average) / kappa,
            "kappa": kappa, "df": prior["df"] + n, "scale": (scale + scale.T) * 0.5}


def _chi_square(df: float, generator: torch.Generator) -> torch.Tensor:
    # Torch's gamma kernel accepts a private generator, unlike Distribution.sample.
    draw = 2 * torch._standard_gamma(torch.tensor(df / 2, dtype=torch.float64, device="cpu"), generator=generator)
    if not bool(torch.isfinite(draw)) or float(draw) <= 0:
        raise AnalysisError("mi_numerical_failure", "A posterior chi-square draw underflowed or overflowed; use a less extreme prior.")
    return draw


def _draw_niw(parameters: Mapping, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """NIW draw via Bartlett's identity-scale Wishart factor, then its inverse.

If A A' ~ W(df,I), B = chol(scale) A^{-T}, then B B' ~ IW(df,scale).
This parameterization has E[Sigma]=scale/(df-p-1), when that moment exists.
"""
    p = parameters["mean"].numel()
    bartlett = torch.tril(torch.randn((p, p), dtype=torch.float64, device="cpu", generator=generator), diagonal=-1)
    for j in range(p):
        bartlett[j, j] = _chi_square(parameters["df"] - j, generator).sqrt()
    factor = _factor(parameters["scale"], "NIW posterior scale")
    inverse_factor = torch.linalg.solve_triangular(bartlett, factor.T, upper=False).T
    covariance = inverse_factor @ inverse_factor.T
    mean = parameters["mean"] + inverse_factor @ torch.randn(p, dtype=torch.float64, device="cpu", generator=generator) / math.sqrt(parameters["kappa"])
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("mi_numerical_failure", "A NIW posterior draw is non-finite; rescale the data or prior.")
    return mean, (covariance + covariance.T) * 0.5


def _patterns(missing: torch.Tensor):
    patterns, inverse = torch.unique(missing, dim=0, return_inverse=True)
    return [(torch.nonzero(inverse == j).reshape(-1), torch.nonzero(~pattern).reshape(-1),
             torch.nonzero(pattern).reshape(-1))
            for j, pattern in enumerate(patterns) if bool(pattern.any())]


def _impute_normal(completed: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor,
                   patterns: Sequence, generator: torch.Generator) -> None:
    for rows, observed, absent in patterns:
        conditional_mean = mean[absent].expand(rows.numel(), -1)
        conditional_covariance = covariance[absent][:, absent]
        if observed.numel():
            oo = covariance[observed][:, observed]
            om = covariance[observed][:, absent]
            coefficient = torch.cholesky_solve(om, _factor(oo, "Observed conditional covariance"))
            conditional_mean = conditional_mean + (completed[rows][:, observed] - mean[observed]) @ coefficient
            conditional_covariance = conditional_covariance - om.T @ coefficient
        conditional_covariance = (conditional_covariance + conditional_covariance.T) * 0.5
        factor = _factor(conditional_covariance, "Missing conditional covariance")
        noise = torch.randn((rows.numel(), absent.numel()), dtype=torch.float64, device="cpu", generator=generator)
        draws = conditional_mean + noise @ factor.T
        if not bool(torch.isfinite(draws).all()):
            raise AnalysisError("mi_numerical_failure", "Conditional imputation produced non-finite values; rescale the data or prior.")
        completed[rows[:, None], absent[None, :]] = draws


def mi_mvn(data: Any, columns: Sequence[str], *, m: int = 5, seed: int = 0,
           burn: int = 100, thin: int = 20, prior: Mapping | None = None,
           max_work: int = DEFAULT_MAX_WORK):
    """Multiple imputation by MVN data augmentation with an explicit proper NIW prior.

    ``prior={'mean': [...], 'kappa': positive, 'df': >p-1, 'scale': [[...]]}``
    specifies ``Sigma ~ IW(df,scale), mu|Sigma ~ N(mean,Sigma/kappa)`` in the
    original units. Each step draws missing values conditional on the current
    parameters and then a NIW posterior parameter draw conditional on the full
    augmented sample. Retain steps ``burn + thin, ..., burn + m*thin``.
    """
    seed = check_seed(seed)
    burn = integer(burn, "burn", maximum=100_000)
    thin = integer(thin, "thin", minimum=1, maximum=100_000)
    m = integer(m, "m", minimum=1, maximum=100)
    if not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= 16:
        raise AnalysisError("invalid_spec", "MI needs a list of 1..16 column names.")
    # The explicit prior is validated before admission or any data workspace.
    parameters = _prior(prior, len(columns))
    requested = burn + m * thin
    frame, values, missing, metadata = admit(data, columns, m=m, iterations=max(1, requested), max_work=max_work)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    completed = torch.where(missing, parameters["mean"].expand_as(values), values).clone()
    kept, trace = [], []
    actual = requested if bool(missing.any()) else 0
    if actual:
        patterns = _patterns(missing)
        mean = parameters["mean"].clone()
        covariance = parameters["scale"] / (parameters["df"] + len(columns) + 1)
        for step in range(1, requested + 1):
            _impute_normal(completed, mean, covariance, patterns, generator)
            mean, covariance = _draw_niw(_niw_posterior(completed, parameters), generator)
            if step > burn and (step - burn) % thin == 0:
                kept.append(completed.clone())
                trace.append({"iteration": step, "mean": mean.tolist(), "covariance_diagonal": covariance.diag().tolist()})
    else:
        kept = [completed.clone() for _ in range(m)]
    metadata.update({"prior": {"family": "normal_inverse_wishart", "mean": parameters["mean"].tolist(),
                     "kappa": parameters["kappa"], "df": parameters["df"], "scale": parameters["scale"].tolist()},
                     "burn": burn, "thin": thin, "iterations_requested": requested,
                     "iterations": actual, "retained_iterations": [v["iteration"] for v in trace],
                     "sampler": "I-step conditional MVN; P-step NIW Bartlett draw", "trace": trace,
                     "convergence": {"assessed": False, "reason": "A finite burn-in and thinning schedule do not establish convergence."},
                     "assumptions": ["continuous multivariate normal working model", "ignorable missingness given selected variables", "explicit proper NIW prior in original units"]})
    return make_result("mi_mvn", frame, values, missing, kept, seed, metadata)


def _regression_posterior(design: torch.Tensor, outcome: torch.Tensor):
    n, q = design.shape
    df = n - q
    if df <= 0:
        raise AnalysisError("mi_residual_df", "Every monotone regression needs positive observed residual degrees of freedom.")
    singular = torch.linalg.svdvals(design)
    if float(singular[-1]) <= 1e-12 * float(singular[0]):
        raise AnalysisError("mi_rank_deficient", "A monotone regression observed design is rank deficient or ill conditioned; choose an identifiable column set.")
    Q, R = torch.linalg.qr(design, mode="reduced")
    offset = outcome.mean()
    centred = outcome - offset
    centred_beta = torch.linalg.solve_triangular(R, (Q.T @ centred)[:, None], upper=True).reshape(-1)
    residual = centred - design @ centred_beta
    beta = centred_beta.clone()
    beta[0] += offset
    sse = float(residual.square().sum())
    magnitude = max(float(centred.square().sum()), 1e-280)
    if not math.isfinite(sse) or sse <= 1e-24 * magnitude:
        raise AnalysisError("mi_zero_residual", "A monotone regression has no numerically positive residual variation; the standard improper-prior posterior is not admissible.")
    return beta, R, sse, df


def _regression_draw(posterior: tuple, generator: torch.Generator):
    beta, R, sse, df = posterior
    sigma = (sse / _chi_square(df, generator)).sqrt()
    noise = torch.randn(beta.numel(), dtype=torch.float64, device="cpu", generator=generator)
    drawn_beta = beta + sigma * torch.linalg.solve_triangular(R, noise[:, None], upper=True).reshape(-1)
    return drawn_beta, sigma


def mi_monotone(data: Any, columns: Sequence[str], *, m: int = 5, seed: int = 0,
                order: Sequence[str] | None = None, max_work: int = DEFAULT_MAX_WORK):
    """Gaussian posterior-predictive MI for a verified monotone missing pattern.

    The order must have nested observed sets: observing a later column implies
    observing every earlier one. By default sort columns by ascending missing
    count, preserving ties. Impute each column from all preceding columns with
    an intercept, drawing sigma²~SSE/chi²(nobs-q), beta|sigma² from its normal
    posterior, and independent predictive Gaussian noise for missing cells.
    The prior is the standard improper p(beta,sigma²) proportional to 1/sigma²;
    full rank, positive residual df and positive residual variation are required.
    """
    seed = check_seed(seed)
    m = integer(m, "m", minimum=1, maximum=100)
    frame, values, missing, metadata = admit(data, columns, m=m, iterations=m, max_work=max_work)
    names = list(frame.columns)
    if order is None:
        ordered = sorted(range(len(names)), key=lambda j: int(missing[:, j].sum()))
    else:
        if isinstance(order, (str, bytes)) or not isinstance(order, (list, tuple)) or len(order) != len(names) or set(order) != set(names) or len(set(order)) != len(names):
            raise AnalysisError("invalid_spec", "order must be a permutation of all selected MI columns.")
        ordered = [names.index(name) for name in order]
    for previous, current in zip(ordered, ordered[1:]):
        if bool((missing[:, previous] & ~missing[:, current]).any()):
            raise AnalysisError("non_monotone_missingness", "The declared variable order does not have nested observed sets; use mi_mvn or mi_chained for arbitrary missingness.")
    # Fits use only target-observed rows, whose earlier variables must all be
    # observed by the verified nesting. Cache exact posterior sufficient state.
    fits = []
    for k, j in enumerate(ordered):
        absent = missing[:, j]
        if not bool(absent.any()):
            continue
        observed = ~absent
        previous = ordered[:k]
        x = values[:, previous]
        if previous:
            centre = x[observed].mean(0)
            scale = (x[observed] - centre).square().mean(0).sqrt()
            if bool((scale == 0).any()) or not bool(torch.isfinite(scale).all()):
                raise AnalysisError("mi_rank_deficient", "A monotone regression has a constant or non-finite observed predictor.")
        else:
            centre, scale = torch.empty(0, dtype=torch.float64, device="cpu"), torch.empty(0, dtype=torch.float64, device="cpu")
        design = torch.cat((torch.ones((int(observed.sum()), 1), dtype=torch.float64, device="cpu"), (x[observed] - centre) / scale), dim=1)
        posterior = _regression_posterior(design, values[observed, j])
        fits.append((j, previous, centre, scale, posterior))
    completed_matrices, diagnostics = [], []
    for i in range(m):
        chain_seed = (seed + i) % 2**63
        generator = torch.Generator(device="cpu").manual_seed(chain_seed)
        completed = values.clone()
        rows = []
        for j, previous, centre, scale, posterior in fits:
            absent = missing[:, j]
            beta, sigma = _regression_draw(posterior, generator)
            design = torch.cat((torch.ones((int(absent.sum()), 1), dtype=torch.float64, device="cpu"), (completed[absent][:, previous] - centre) / scale), dim=1)
            completed[absent, j] = design @ beta + sigma * torch.randn(int(absent.sum()), dtype=torch.float64, device="cpu", generator=generator)
            rows.append({"column": names[j], "n_observed": int((~absent).sum()),
                         "residual_df": posterior[3], "sigma": float(sigma), "beta_standardized": beta.tolist()})
        if not bool(torch.isfinite(completed).all()):
            raise AnalysisError("mi_numerical_failure", "Monotone posterior predictive draws are non-finite; rescale columns.")
        completed_matrices.append(completed)
        diagnostics.append({"imputation": i + 1, "regressions": rows})
    metadata.update({"order": [names[j] for j in ordered], "monotone_verified": True,
                     "prior": "p(beta,sigma_squared) proportional to 1/sigma_squared",
                     "iterations": 1 if fits else 0, "regression_draws": m * len(fits),
                     "sampler": "exact Gaussian regression posterior and predictive draws",
                     "diagnostics": diagnostics,
                     "assumptions": ["continuous Gaussian regressions with constant conditional residual variance", "ignorable missingness given preceding variables", "full-rank observed design and proper observed-data posterior"]})
    return make_result("mi_monotone", frame, values, missing, completed_matrices, seed, metadata,
                       imputation_seeds=[(seed + i) % 2**63 for i in range(m)])
