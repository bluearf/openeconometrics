"""Proper analytic regression, contrasts and saved joint posterior predictions."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.mi.common import _encode_index
from openecon.engines.distributions import t_isf

from .core import (DEFAULT_MAX_BYTES, DEFAULT_MAX_WORK, MAX_DRAWS, admit, digest,
                   integer, load_mapping, matrices, names, posterior_algebra, real,
                   resident, source_values, metadata_bytes)
from openecon.resources import plan_workspace, workspace_budget_bytes
from .posterior import (IndexState, NormalInverseGammaPrior, PosteriorBundle,
                        PosteriorContrast, PosteriorDraws, PosteriorState, _contrast_algebra)


def _bundle(value):
    """Revalidate even constructed/copied result objects, with no optimizer refit."""
    if isinstance(value, PosteriorBundle):
        admit(value.nobs_original, len(value.terms), max_work=value.max_work, max_bytes=value.max_bytes)
        plan_workspace("bayesian_typed_result_copy", {"typed_result_serialization": 3*metadata_bytes(value)},
                       budget_bytes=min(value.max_bytes, workspace_budget_bytes()))
        value = value.model_dump(mode="json")
    return PosteriorBundle.model_validate(dict(load_mapping(value)))


def _prior(value):
    if isinstance(value, NormalInverseGammaPrior):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise AnalysisError("invalid_prior", "Supply a complete proper NormalInverseGammaPrior or its mapping.")
    return NormalInverseGammaPrior.model_validate(dict(value))


def bayes_linear(*, data: Any, y: str, prior: NormalInverseGammaPrior | Mapping,
                 x: Sequence[str] | None = None, intercept: bool = True,
                 missing: str = "raise", alpha: float = 0.05,
                 max_work: int = DEFAULT_MAX_WORK, max_bytes: int = DEFAULT_MAX_BYTES,
                 device: str = "cpu") -> PosteriorBundle:
    """Exact iid Gaussian regression with an explicit proper normal–inverse-gamma prior.

    Prior coefficient order is Intercept (when present), followed by x. Credible
    intervals use the multivariate Student-t marginal scale, not its covariance.
    No weights, categorical encoding, streaming, GPU or MCMC support is implied.
    """
    if device != "cpu":
        raise AnalysisError("unsupported_device", "Analytic Bayesian regression supports CPU float64 only.")
    if type(intercept) is not bool or missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_spec", "intercept must be boolean and missing must be raise or drop.")
    if not isinstance(y, str) or not y.strip() or len(y) > 1000:
        raise AnalysisError("invalid_spec", "Declare a nonempty outcome column name.")
    alpha = real(alpha, "alpha", low=0, high=1)
    predictors = names(x)
    columns = [y, *predictors]
    if len(set(columns)) != len(columns) or "Intercept" in predictors:
        raise AnalysisError("invalid_spec", "Outcome/predictors must be distinct; Intercept is a reserved term.")
    n, k = resident(data, columns), len(predictors) + int(intercept)
    admit(n, k, max_work=max_work, max_bytes=max_bytes,
          index_bytes=int(data.index.memory_usage(deep=True)))
    # Prior conversion, source/index serialization and numerical copies follow admission.
    prior = _prior(prior)
    if len(prior.mean) != k:
        raise AnalysisError("invalid_prior", "Prior coefficient count disagrees with the declared design.")
    with torch.device("cpu"):
        values, dtypes = source_values(data, columns)
        positions = tuple(i for i in range(n) if all(column[i] is not None for column in values))
        if not positions:
            raise AnalysisError("empty_sample", "The Bayesian complete-case sample is empty.")
        if missing == "raise" and len(positions) != n:
            raise AnalysisError("missing_data", "Model columns contain missing values; declare missing='drop' explicitly.")
        index = IndexState.model_validate(_encode_index(data.index))
        source_hash = digest({"columns": columns, "dtypes": dtypes, "values": values,
                              "index": index.descriptor()})
        state = PosteriorState(outcome=y, predictors=tuple(predictors), intercept=intercept,
                               missing=missing, source_columns=tuple(columns), source_dtypes=dtypes,
                               source_values=values, source_index=index, sample_positions=positions,
                               source_sha256=source_hash)
        observed, design = matrices(values, positions, intercept=intercept)
        solved = posterior_algebra(observed, design, prior)
        terms = (("Intercept",) if intercept else ()) + tuple(predictors)
        critical = t_isf(alpha / 2, solved["degrees_of_freedom"])
        intervals = [[mean - critical * math.sqrt(row[i]), mean + critical * math.sqrt(row[i])]
                     for i, (mean, row) in enumerate(zip(solved["mean"], solved["coefficient_scale_matrix"], strict=True))]
        body = {"schema_version": "openecon.posterior_bundle.v1", "method": "proper_conjugate_gaussian",
                "terms": terms, "prior": prior.model_dump(mode="json"), "state": state.model_dump(mode="json"),
                **solved, "alpha": alpha, "credible_intervals": intervals,
                "max_work": integer(max_work, "max_work"), "max_bytes": integer(max_bytes, "max_bytes"),
                "source_rank": int(torch.linalg.matrix_rank(design))}
        return PosteriorBundle.model_validate(body | {"integrity_sha256": digest(body)})


def bayes_contrast(*, result: PosteriorBundle | Mapping | str, weights: Sequence[float] | Mapping[str, float],
                   threshold: float = 0, alpha: float | None = None) -> PosteriorContrast:
    """Exact marginal linear contrast and posterior probability above a threshold.

    This is a continuous posterior probability; it is not a point-null Bayes factor.
    """
    result = _bundle(result)
    alpha = result.alpha if alpha is None else real(alpha, "alpha", low=0, high=1)
    threshold = real(threshold, "threshold")
    if isinstance(weights, Mapping):
        if any(term not in result.terms for term in weights):
            raise AnalysisError("invalid_contrast", "Contrast contains an unknown coefficient term.")
        weights = [weights.get(term, 0) for term in result.terms]
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence) or len(weights) != len(result.terms):
        raise AnalysisError("invalid_contrast", "Contrast weights must match the complete coefficient order.")
    weights = [real(value, "contrast weight") for value in weights]
    body = {"schema_version": "openecon.posterior_contrast.v1",
            "posterior": result.model_dump(mode="json"),
            "posterior_sha256": result.integrity_sha256, "terms": result.terms,
            "weights": tuple(weights), "alpha": alpha, "threshold": threshold,
            **_contrast_algebra(result, weights, threshold, alpha)}
    return PosteriorContrast.model_validate(body | {"integrity_sha256": digest(body)})


def _query(result, data, missing):
    if missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_spec", "Prediction missing policy must be raise or drop.")
    predictors = list(result.state.predictors)
    n = resident(data, predictors)
    admit(result.nobs_original, len(result.terms), queries=n,
          max_work=result.max_work, max_bytes=result.max_bytes,
          index_bytes=int(data.index.memory_usage(deep=True)))
    values, _ = source_values(data, predictors)
    positions = tuple(i for i in range(n) if all(column[i] is not None for column in values))
    if not positions:
        raise AnalysisError("empty_sample", "The Bayesian query sample is empty.")
    if missing == "raise" and len(positions) != n:
        raise AnalysisError("missing_data", "Query columns contain missing values.")
    dummy = (tuple(0 for _ in range(n)), *values)
    _, design = matrices(dummy, positions, intercept=result.state.intercept)
    return design, data.index[list(positions)], positions


def bayes_predict(*, result: PosteriorBundle | Mapping | str, data: Any,
                  missing: str = "raise", alpha: float | None = None) -> pd.DataFrame:
    """Saved analytic mean credible and new-outcome predictive intervals, with full joint scale.

    attrs stores factorized cross-row mean scale and parameter covariance. Independent
    future observation variance is added only for the new-outcome predictive target.
    """
    result = _bundle(result)
    alpha = result.alpha if alpha is None else real(alpha, "alpha", low=0, high=1)
    with torch.device("cpu"):
        design, index, positions = _query(result, data, missing)
        mean = design @ torch.tensor(result.mean, dtype=torch.float64, device="cpu")
        vn = torch.tensor(result.conditional_scale_matrix, dtype=torch.float64, device="cpu")
        conditional_factor = design @ torch.linalg.cholesky(vn)
        factor = conditional_factor * math.sqrt(result.scale/result.shape)
        mean_scale_squared = factor.square().sum(1)
        outcome_scale_squared = mean_scale_squared + result.scale/result.shape
        mean_scale, outcome_scale = mean_scale_squared.sqrt(), outcome_scale_squared.sqrt()
        covariance_factor = variance = multiplier = None
        multiplier_status = "moment_unavailable"
        if result.coefficient_covariance is not None:
            covariance_factor = conditional_factor * math.sqrt(result.variance_mean)
            variance = covariance_factor.square().sum(1)
            if not bool(torch.isfinite(covariance_factor).all()) or not bool(torch.isfinite(variance).all()):
                raise AnalysisError("numerical_failure", "Posterior predictive covariance exceeds float64 range.")
            denominator = math.fsum((result.prior.shape, (result.nobs-2)/2))
            multiplier = result.shape / denominator
            multiplier_status = "finite" if math.isfinite(multiplier) else "unrepresentable_multiplier"
            if not math.isfinite(multiplier):
                multiplier = None
        critical = t_isf(alpha / 2, result.degrees_of_freedom)
        values = torch.stack((mean, mean_scale, outcome_scale, mean-critical*mean_scale,
                              mean+critical*mean_scale, mean-critical*outcome_scale,
                              mean+critical*outcome_scale), dim=1)
        if not bool(torch.isfinite(values).all()):
            raise AnalysisError("numerical_failure", "Posterior predictions exceed supported float64 range.")
        frame = pd.DataFrame(values.tolist(), index=index,
                             columns=["mean", "mean_student_t_scale", "outcome_student_t_scale",
                                      "mean_credible_lower", "mean_credible_upper",
                                      "outcome_predictive_lower", "outcome_predictive_upper"])
        frame["mean_posterior_variance"] = None if variance is None else variance.tolist()
        frame.attrs.update({"schema_version": "openecon.bayesian_prediction.v1",
                            "posterior_sha256": result.integrity_sha256,
                            "sample_positions": list(positions), "degrees_of_freedom": result.degrees_of_freedom,
                            "credible_probability": 1-alpha, "mean_scale_factor": factor.tolist(),
                            "outcome_independent_scale": result.scale/result.shape,
                            "scale_to_covariance_multiplier": multiplier,
                            "scale_to_covariance_status": multiplier_status,
                            "mean_covariance_factor": None if covariance_factor is None else covariance_factor.tolist(),
                            "outcome_independent_variance": result.variance_mean,
                            "parameter_covariance": result.coefficient_covariance,
                            "prediction_target": "iid Gaussian future observations under the declared proper prior"})
        return frame


def _draw_arrays(result, draws, seed, query_design):
    with torch.device("cpu"):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        gamma = torch._standard_gamma(torch.full((draws,), result.shape, dtype=torch.float64, device="cpu"),
                                      generator=generator)
        sigma = result.scale / gamma
        vn = torch.tensor(result.conditional_scale_matrix, dtype=torch.float64, device="cpu")
        normal = torch.randn((draws, len(result.terms)), dtype=torch.float64, device="cpu", generator=generator)
        beta = (torch.tensor(result.mean, dtype=torch.float64, device="cpu")
                + sigma.sqrt()[:, None] * (normal @ torch.linalg.cholesky(vn).T))
        means, outcomes = None, None
        blocks = [sigma, beta]
        if query_design is not None:
            query = torch.tensor(query_design, dtype=torch.float64, device="cpu")
            means = beta @ query.T
            noise = torch.randn(means.shape, dtype=torch.float64, device="cpu", generator=generator)
            outcomes = means + sigma.sqrt()[:, None] * noise
            blocks.extend((means, outcomes))
        if not all(bool(torch.isfinite(block).all()) for block in blocks) or bool((sigma <= 0).any()):
            raise AnalysisError("numerical_failure", "Posterior draws exceed supported float64 range.")
        return {"beta": beta.tolist(), "sigma_squared": sigma.tolist(),
                "mean_draws": None if means is None else means.tolist(),
                "outcome_draws": None if outcomes is None else outcomes.tolist()}


def bayes_draws(*, result: PosteriorBundle | Mapping | str, draws: int = 1000,
                seed: int = 0, data: Any = None, missing: str = "raise") -> PosteriorDraws:
    """Bounded, independent joint beta/sigma² draws and optional iid predictive draws.

    A local CPU generator preserves global RNG state. These are exact conjugate
    draws, so warmup, chain R-hat/ESS and sampler-convergence claims are absent.
    The saved draw result replays its target/seed on validation.
    """
    result = _bundle(result)
    draws = integer(draws, "draws", high=MAX_DRAWS)
    seed = integer(seed, "seed", low=0)
    queries = 0 if data is None else resident(data, list(result.state.predictors))
    admit(result.nobs_original, len(result.terms), operation="bayesian_joint_draws",
          draws=draws, queries=queries, max_work=result.max_work, max_bytes=result.max_bytes,
          index_bytes=0 if data is None else int(data.index.memory_usage(deep=True)))
    design, index = None, None
    if data is not None:
        with torch.device("cpu"):
            matrix, labels, _ = _query(result, data, missing)
            design, index = matrix.tolist(), IndexState.model_validate(_encode_index(labels))
    arrays = _draw_arrays(result, draws, seed, design)
    body = {"schema_version": "openecon.posterior_draws.v1", "posterior": result.model_dump(mode="json"),
            "seed": seed, **arrays, "query_design": design,
            "query_index": None if index is None else index.model_dump(mode="json")}
    return PosteriorDraws.model_validate(body | {"integrity_sha256": digest(body)})
