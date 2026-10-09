"""Bounded native float64 fully conditional multiple imputation.

Gaussian and PMM updates draw the full conjugate regression posterior. Binary
updates use a symmetric Metropolis transition targeting a stated proper-prior
logistic posterior. A finite run is never labelled a converged posterior sample.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import admit, check_seed, make_result
from openecon.resources import plan_workspace


_DTYPE = torch.float64
_MAX_CYCLES = 1000
_MAX_TRACE_ENTRIES = 100_000


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise AnalysisError("invalid_spec", f"{name} must be an integer in [{lower}, {upper}].")
    return int(value)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("invalid_spec", f"{name} must be a finite positive real number.")
    number = float(value)
    if (not math.isfinite(number) or number <= 0 or not math.isfinite(number * number)
            or number * number <= 0):
        raise AnalysisError("invalid_spec", f"{name} must have a finite positive square.")
    return number


def _finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{context} produced non-finite values.")


def _checked_qr(x: torch.Tensor, target: str = "conditional model"):
    """Reject rank loss instead of silently dropping columns or adding a ridge."""
    _finite(x, target)
    singular = torch.linalg.svdvals(x)
    tolerance = torch.finfo(_DTYPE).eps * max(x.shape) * singular[0]
    if x.shape[0] < x.shape[1] or bool(singular[-1] <= tolerance):
        raise AnalysisError(
            "singular_design", f"The observed predictor design for {target!r} is rank deficient."
        )
    return torch.linalg.qr(x, mode="reduced")


@dataclass(frozen=True)
class _NormalPosterior:
    beta_hat: torch.Tensor
    r: torch.Tensor
    sse: torch.Tensor
    df: int


def _normal_parameters(x: torch.Tensor, y: torch.Tensor,
                       target: str = "conditional model") -> _NormalPosterior:
    """Posterior under p(beta, sigma²) proportional to 1 / sigma²."""
    df = x.shape[0] - x.shape[1]
    if df <= 0:
        raise AnalysisError(
            "insufficient_observations",
            f"{target!r} needs more observed outcomes than regression coefficients."
        )
    q, r = _checked_qr(x, target)
    beta_hat = torch.linalg.solve_triangular(r, (q.T @ y).unsqueeze(1), upper=True)[:, 0]
    _finite(beta_hat, f"Regression for {target!r}")
    residual = y - x @ beta_hat
    sse = residual @ residual
    centered = y - y.mean()
    total = centered @ centered
    _finite(torch.stack((sse, total)), f"Residual variance for {target!r}")
    if bool(total <= 0) or bool(sse <= 64 * torch.finfo(_DTYPE).eps * total):
        raise AnalysisError(
            "zero_residual_variance",
            f"{target!r} has zero or numerically unresolved residual variance."
        )
    return _NormalPosterior(beta_hat, r, sse, df)


def _gaussian_draw(fit: _NormalPosterior, generator: torch.Generator):
    # Integer df permits an exact chi-square draw as a sum of squared normals;
    # all draws use the caller's local CPU generator, never the global RNG.
    chi = torch.randn(fit.df, dtype=_DTYPE, device="cpu", generator=generator).square().sum()
    if not bool(torch.isfinite(chi)) or bool(chi <= 0):
        raise AnalysisError("numerical_failure", "The posterior chi-square draw is not positive.")
    sigma2 = fit.sse / chi
    z = torch.randn(fit.r.shape[0], dtype=_DTYPE, device="cpu", generator=generator)
    delta = torch.linalg.solve_triangular(fit.r, z.unsqueeze(1), upper=True)[:, 0]
    beta = fit.beta_hat + sigma2.sqrt() * delta
    _finite(beta, "The Gaussian coefficient draw")
    if not bool(torch.isfinite(sigma2)) or bool(sigma2 <= 0):
        raise AnalysisError("numerical_failure", "The posterior variance draw is not positive.")
    return beta, sigma2


def _pmm_draw(x_observed: torch.Tensor, x_missing: torch.Tensor, y_observed: torch.Tensor,
              fit: _NormalPosterior, generator: torch.Generator, *, donors: int,
              ties: str):
    """Type-1 matching: observed fitted means versus missing posterior means."""
    beta, sigma2 = _gaussian_draw(fit, generator)
    observed_prediction = x_observed @ fit.beta_hat
    missing_prediction = x_missing @ beta
    _finite(observed_prediction, "PMM observed predictions")
    _finite(missing_prediction, "PMM missing predictions")
    n_observed = y_observed.numel()
    if not 1 <= donors <= n_observed:
        raise AnalysisError("invalid_spec", "PMM donors must not exceed the observed outcome count.")
    choices = torch.randint(donors, (x_missing.shape[0],), generator=generator, device="cpu")
    rows = torch.empty(x_missing.shape[0], dtype=torch.int64, device="cpu")
    ordered = torch.arange(n_observed, device="cpu")
    # One distance vector at a time avoids a missing-by-observed allocation.
    for j, prediction in enumerate(missing_prediction):
        order = (torch.randperm(n_observed, generator=generator, device="cpu")
                 if ties == "random" else ordered)
        distance = (observed_prediction[order] - prediction).abs()
        _finite(distance, "PMM matching distances")
        nearest = order[torch.argsort(distance, stable=True)[:donors]]
        rows[j] = nearest[choices[j]]
    return y_observed[rows], beta, sigma2, rows


def _logit_log_posterior(x: torch.Tensor, y: torch.Tensor, beta: torch.Tensor,
                         prior_scale: float) -> torch.Tensor:
    eta = x @ beta
    return (y * eta - torch.nn.functional.softplus(eta)).sum() - beta.square().sum() / (
        2 * prior_scale ** 2
    )


def _logit_draw(x: torch.Tensor, y: torch.Tensor, generator: torch.Generator, *,
                 prior_scale: float, proposal_scale: float, burn: int, steps: int,
                 initial: torch.Tensor | None = None):
    """Exact-target MH kernel; finite transitions need not have reached stationarity.

    The proposal factor depends on X and the prior, not the current beta. It is
    therefore symmetric and requires only the posterior density ratio.
    """
    dimension = x.shape[1]
    precision = 0.25 * (x.T @ x) + torch.eye(dimension, dtype=_DTYPE, device="cpu") / (
        prior_scale ** 2
    )
    _finite(precision, "The logistic proposal precision")
    factor, info = torch.linalg.cholesky_ex(precision)
    if int(info) != 0:
        raise AnalysisError("numerical_failure", "The logistic proposal factor is not positive definite.")
    beta = (torch.zeros(dimension, dtype=_DTYPE, device="cpu")
            if initial is None else initial.clone())
    log_target = _logit_log_posterior(x, y, beta, prior_scale)
    _finite(log_target, "The logistic posterior density")
    accepted_burn = accepted_steps = 0
    for transition in range(burn + steps):
        z = torch.randn(dimension, dtype=_DTYPE, device="cpu", generator=generator)
        delta = torch.linalg.solve_triangular(factor.T, z.unsqueeze(1), upper=True)[:, 0]
        candidate = beta + proposal_scale * delta
        candidate_target = _logit_log_posterior(x, y, candidate, prior_scale)
        _finite(candidate_target, "The logistic proposal posterior density")
        log_uniform = torch.rand((), dtype=_DTYPE, device="cpu", generator=generator).log()
        if bool(log_uniform < candidate_target - log_target):
            beta, log_target = candidate, candidate_target
            if transition < burn:
                accepted_burn += 1
            else:
                accepted_steps += 1
    proposals = burn + steps
    return beta, {
        "burn_proposals": burn, "sampling_proposals": steps,
        "accepted_burn": accepted_burn, "accepted_sampling": accepted_steps,
        "acceptance_rate": (accepted_burn + accepted_steps) / proposals,
        "final_log_posterior": float(log_target),
    }


def _predictor_spec(names: tuple[str, ...], targets: tuple[str, ...], predictors):
    if predictors is None:
        return {name: tuple(other for other in names if other != name) for name in targets}
    if not isinstance(predictors, Mapping) or set(predictors) != set(targets):
        raise AnalysisError("invalid_spec", "predictors must map every incomplete column to its predictors.")
    result = {}
    for target in targets:
        requested = predictors[target]
        if (isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence)
                or any(not isinstance(name, str) for name in requested)):
            raise AnalysisError("invalid_spec", f"predictors[{target!r}] must be a sequence of column names.")
        requested = tuple(requested)
        if (len(set(requested)) != len(requested) or target in requested
                or any(name not in names for name in requested)):
            raise AnalysisError("invalid_spec", f"Predictors for {target!r} must be unique other selected columns.")
        result[target] = requested
    return result


def mi_chained(data, columns, *, methods, m: int = 5, seed: int = 0, burn: int = 20,
               iterations: int = 10, donors: int = 5, predictors=None,
               pmm_ties: str = "stable", logit_prior_scale: float = 2.5,
               logit_proposal_scale: float = 1.0, logit_burn: int = 100,
               logit_steps: int = 100, max_work: int = 100_000_000):
    """Impute a selected resident numeric panel with normal, PMM and binary FCS.

    ``methods`` maps each incomplete selected column to ``normal``, ``pmm`` or
    ``logit``. ``predictors`` optionally maps those same columns to explicit
    predictor-name sequences; otherwise all other selected columns are used.
    Independent seeded chains each run ``burn + iterations`` ordered sweeps,
    returning their last completed panel. See ``docs/econometrics/mi-chained.md``.
    """
    check_seed(seed)
    m = _integer(m, "m", 1, 100)
    burn = _integer(burn, "burn", 0, _MAX_CYCLES)
    iterations = _integer(iterations, "iterations", 1, _MAX_CYCLES)
    if burn + iterations > _MAX_CYCLES:
        raise AnalysisError("resource_limit", f"burn + iterations must not exceed {_MAX_CYCLES}.")
    donors = _integer(donors, "donors", 1, 10_000)
    logit_burn = _integer(logit_burn, "logit_burn", 0, 100_000)
    logit_steps = _integer(logit_steps, "logit_steps", 1, 100_000)
    logit_prior_scale = _positive(logit_prior_scale, "logit_prior_scale")
    logit_proposal_scale = _positive(logit_proposal_scale, "logit_proposal_scale")
    max_work = _integer(max_work, "max_work", 1, 10_000_000_000)
    if not isinstance(pmm_ties, str) or pmm_ties not in ("stable", "random"):
        raise AnalysisError("invalid_spec", "pmm_ties must be 'stable' or 'random'.")
    frame, values, missing, admission = admit(
        data, columns, m=m, iterations=burn + iterations, max_work=max_work
    )
    names = tuple(frame.columns)
    targets = tuple(name for j, name in enumerate(names) if bool(missing[:, j].any()))
    if not targets:
        raise AnalysisError("no_missing_values", "mi_chained needs at least one incomplete selected column.")
    if not isinstance(methods, Mapping) or set(methods) != set(targets):
        raise AnalysisError("invalid_spec", "methods must specify exactly the incomplete selected columns.")
    if any(not isinstance(methods[name], str) or methods[name] not in ("normal", "pmm", "logit")
           for name in targets):
        raise AnalysisError("invalid_spec", "Supported chained methods are 'normal', 'pmm' and 'logit'.")
    predictor_spec = _predictor_spec(names, targets, predictors)
    n, p = values.shape
    cycles = burn + iterations
    if m * cycles * len(targets) > _MAX_TRACE_ENTRIES:
        raise AnalysisError("resource_limit", "The requested chain traces exceed 100,000 variable updates.")
    specifications = {}
    work_per_cycle = 0
    trace_bytes_per_cycle = 512
    for target in targets:
        j = names.index(target)
        mask = missing[:, j]
        n_missing = int(mask.sum())
        n_observed = n - n_missing
        if n_observed == 0:
            raise AnalysisError("all_missing", f"{target!r} has no observed values.")
        design_p = len(predictor_spec[target]) + 1
        if methods[target] == "logit":
            observed = values[~mask, j]
            if not bool(((observed == 0) | (observed == 1)).all()):
                raise AnalysisError("invalid_binary_outcome", f"Observed {target!r} values must be exactly 0 or 1.")
            if bool(observed.min() == observed.max()):
                raise AnalysisError("single_class", f"{target!r} requires both observed binary classes.")
        elif n_observed <= design_p:
            raise AnalysisError("insufficient_observations", f"{target!r} needs more observed outcomes than coefficients.")
        if methods[target] == "pmm" and donors > n_observed:
            raise AnalysisError("invalid_spec", f"donors={donors} exceeds the {n_observed} observed values of {target!r}.")
        work = n_observed * design_p ** 2 + design_p ** 3 + n_missing * design_p
        if methods[target] == "pmm":
            work += n_missing * n_observed * max(1, n_observed.bit_length())
        if methods[target] == "logit":
            work += (logit_burn + logit_steps) * (n_observed * design_p + design_p ** 2)
        work_per_cycle += work
        trace_bytes_per_cycle += 4096 + 160 * design_p
        if methods[target] == "pmm":
            trace_bytes_per_cycle += 192 * n_missing
        specifications[target] = (j, mask, tuple(names.index(name) for name in predictor_spec[target]))
    projected_work = m * (cycles * work_per_cycle + n * p)
    if projected_work > max_work:
        raise AnalysisError(
            "resource_limit", f"The requested FCS work ({projected_work:,}) exceeds max_work={max_work:,}."
        )
    trace_plan = plan_workspace("chained multiple imputation", {
        "admitted panels and saved state": admission["resource_plan"]["estimated_workspace_bytes"],
        "conditional traces and serialized checksum state": m * cycles * trace_bytes_per_cycle,
    })
    completed = []
    chain_diagnostics = []
    chain_seeds = [(int(seed) + chain) % (2 ** 63) for chain in range(m)]
    try:
        for imputation, chain_seed in enumerate(chain_seeds, 1):
            generator = torch.Generator(device="cpu").manual_seed(chain_seed)
            filled = values.clone()
            for target in targets:
                j, mask, _ = specifications[target]
                observed = values[~mask, j]
                positions = torch.randint(observed.numel(), (int(mask.sum()),), generator=generator, device="cpu")
                filled[mask, j] = observed[positions]
            logit_states = {}
            trace = []
            logistic_totals = {}
            for cycle in range(1, cycles + 1):
                cycle_trace = {"cycle": cycle, "phase": "burn" if cycle <= burn else "sampling", "variables": {}}
                for target in targets:
                    j, mask, predictor_indices = specifications[target]
                    design = torch.cat((torch.ones((n, 1), dtype=_DTYPE, device="cpu"),
                                        filled[:, predictor_indices]), dim=1)
                    x_observed, x_missing = design[~mask], design[mask]
                    y_observed = values[~mask, j]
                    diagnostic = {"method": methods[target]}
                    if methods[target] == "logit":
                        _checked_qr(x_observed, target)
                        beta, mh = _logit_draw(
                            x_observed, y_observed, generator, prior_scale=logit_prior_scale,
                            proposal_scale=logit_proposal_scale, burn=logit_burn, steps=logit_steps,
                            initial=logit_states.get(target)
                        )
                        logit_states[target] = beta
                        probability = torch.sigmoid(x_missing @ beta)
                        _finite(probability, f"Logistic predictions for {target!r}")
                        draw = (torch.rand(probability.shape, dtype=_DTYPE, device="cpu", generator=generator)
                                < probability).to(_DTYPE)
                        diagnostic.update(mh)
                        diagnostic["mean_probability"] = float(probability.mean())
                        totals = logistic_totals.setdefault(target, {"accepted": 0, "proposals": 0})
                        totals["accepted"] += mh["accepted_burn"] + mh["accepted_sampling"]
                        totals["proposals"] += logit_burn + logit_steps
                    else:
                        fit = _normal_parameters(x_observed, y_observed, target)
                        if methods[target] == "pmm":
                            draw, beta, sigma2, rows = _pmm_draw(
                                x_observed, x_missing, y_observed, fit, generator,
                                donors=donors, ties=pmm_ties
                            )
                            observed_positions = torch.nonzero(~mask, as_tuple=False)[:, 0]
                            diagnostic["donor_row_positions"] = observed_positions[rows].tolist()
                        else:
                            beta, sigma2 = _gaussian_draw(fit, generator)
                            noise = torch.randn(x_missing.shape[0], dtype=_DTYPE, device="cpu", generator=generator)
                            draw = x_missing @ beta + sigma2.sqrt() * noise
                        diagnostic.update({"residual_df": fit.df, "sigma2_draw": float(sigma2)})
                    _finite(draw, f"Imputations for {target!r}")
                    filled[mask, j] = draw
                    summary = torch.stack((draw.mean(), draw.var(unbiased=False)))
                    _finite(summary, f"Imputation trace for {target!r}")
                    diagnostic.update({"coefficient_draw": beta.tolist(), "imputed_mean": float(summary[0]),
                                       "imputed_variance": float(summary[1])})
                    cycle_trace["variables"][target] = diagnostic
                trace.append(cycle_trace)
            for totals in logistic_totals.values():
                totals["acceptance_rate"] = totals["accepted"] / totals["proposals"]
            completed.append(filled)
            chain_diagnostics.append({"imputation": imputation, "seed": chain_seed, "trace": trace,
                                      "logistic_acceptance": logistic_totals})
    except torch.linalg.LinAlgError as exc:
        raise AnalysisError("numerical_failure", "A chained conditional factorization failed.") from exc
    metadata = {
        **admission, "methods": {name: methods[name] for name in targets},
        "predictors": {name: list(predictor_spec[name]) for name in targets},
        "visit_order": list(targets), "burn": burn, "iterations": iterations,
        "cycles_per_chain": cycles, "initialization": "independent random observed donor per missing cell",
        "donors": donors, "pmm_match_type": 1, "pmm_ties": pmm_ties,
        "normal_prior": "p(beta,sigma2) proportional to 1/sigma2; full-rank Gaussian regression",
        "logit_prior": {"family": "independent Gaussian", "mean": 0.0, "scale": logit_prior_scale,
                        "coefficient_units": "raw predictors, including intercept"},
        "logit_sampler": {"kernel": "symmetric random-walk Metropolis", "exact_target": True,
                          "stationarity_claim": False, "burn_per_update": logit_burn,
                          "steps_per_update": logit_steps, "proposal_scale": logit_proposal_scale,
                          "proposal_precision": "0.25 Xobs'Xobs + I/prior_scale^2; fixed within update"},
        "convergence_claim": False, "projected_work": projected_work,
        "chained_resource_plan": trace_plan.record(),
        "chain_diagnostics": chain_diagnostics,
    }
    return make_result("mi_chained", frame, values, missing, completed, seed, metadata,
                       imputation_seeds=chain_seeds)
