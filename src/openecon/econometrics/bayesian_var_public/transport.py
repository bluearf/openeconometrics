"""Source-only proposal: exact typed routes to original complete-state replay.

Cached Bartlett and normal primitives are replayed algebraically. These routes
never fit, sample, regenerate a seed stream, trim a draw or repair a covariance.
"""
from __future__ import annotations

from importlib import import_module

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.econometrics.bayesian_var.admission import (
    DEFAULT_MAX_BYTES, DEFAULT_MAX_WORK, integer, load_mapping, metadata_admit)

ROUTES = {
    "openecon.bayesian_var_posterior.v1": ("posterior", "BayesianVARPosterior"),
    "openecon.bayesian_var_fixed_prediction.v1": ("api", "BayesianVARPrediction"),
    "openecon.bayesian_var_rank_one_contrast.v1": ("api", "BayesianVARContrast"),
    "openecon.bayesian_var_joint_draws.v1": ("draws", "BayesianVARDraws"),
    "openecon.bayesian_var_forecast.v1": ("forecast", "BayesianVARForecast"),
    "openecon.bayesian_var_impulse.v1": ("impulse", "BayesianVARImpulse"),
}


def _primitive_tree(value):
    """Preserve every admitted field and force nested typed semantic replay.

    BaseModel.model_dump can omit illicit __dict__/extra fields; complete public
    transport refuses those fields explicitly before it creates semantic input.
    No checked/dump override, fit or random generator is invoked here.
    """
    from collections.abc import Mapping
    from pydantic import BaseModel
    if isinstance(value, BaseModel):
        fields = type(value).__pydantic_fields__
        body = value.__dict__
        if (set(body) != set(fields)
                or getattr(value, "__pydantic_extra__", None)
                or getattr(value, "__pydantic_private__", None)):
            raise AnalysisError("invalid_state", "Complete typed BVAR fields cannot omit or conceal metadata.")
        return {key: _primitive_tree(item) for key, item in body.items()}
    if isinstance(value, Mapping):
        return {key: _primitive_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_primitive_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_primitive_tree(item) for item in value)
    return value


def mapping(value):
    # Combined public admission already precedes these complete primitive copies;
    # retain the original metadata gate also for explicit internal mapping calls.
    metadata_admit(value)
    from pydantic import BaseModel
    raw = value if isinstance(value, BaseModel) else load_mapping(value)
    return _primitive_tree(raw)


def restore(value, *, expected=None, _admitted=False):
    if not _admitted:
        from .tables import admission
        value, _ = admission(value)
    raw = mapping(value)
    schema = raw.get("schema_version")
    if type(schema) is not str or schema not in ROUTES or expected is not None and schema != expected:
        raise AnalysisError("invalid_state", "Supply the complete exact BVAR state for this explicit restore route.")
    module, name = ROUTES[schema]
    cls = getattr(import_module("openecon.econometrics.bayesian_var."+module), name)
    try:
        return cls.model_validate(raw)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError("invalid_state", "Complete BVAR state fails its unchanged original typed replay.") from exc


def bayes_var_restore(value):
    """Restore any of the six declared complete typed BVAR states by exact schema."""
    return restore(value)


def bayes_var_predict_restore(value):
    """Restore full fixed-design matrix-t state including cross-row covariance."""
    return restore(value, expected="openecon.bayesian_var_fixed_prediction.v1")


def bayes_var_contrast_restore(value):
    """Restore full original rank-one contrast vectors and exact Student-t law."""
    return restore(value, expected="openecon.bayesian_var_rank_one_contrast.v1")


def bayes_var_draws_restore(value):
    """Restore every original joint draw and cached primitive without RNG."""
    return restore(value, expected="openecon.bayesian_var_joint_draws.v1")


def bayes_var_forecast_restore(value):
    """Restore both full recursive paths and moment/MC-error availability laws."""
    return restore(value, expected="openecon.bayesian_var_forecast.v1")


def bayes_var_irf_restore(value):
    """Restore full ordered simple/orthogonalized responses and MC labels."""
    return restore(value, expected="openecon.bayesian_var_impulse.v1")


def bayes_var_prior_restore(value, *, max_work=DEFAULT_MAX_WORK, max_bytes=DEFAULT_MAX_BYTES):
    """Validate a complete proper MNIW declaration using original strict SPD law."""
    max_work = integer(max_work, "max_work", low=1)
    max_bytes = integer(max_bytes, "max_bytes", low=1)
    size = metadata_admit(value, max_bytes=max_bytes)
    if hasattr(value, "__pydantic_fields__"):
        from pydantic import BaseModel
        value = BaseModel.model_dump(value, mode="python")
    elif isinstance(value, (str, bytes, bytearray)):
        value = load_mapping(value)
    if not isinstance(value, dict) or set(value) != {"mean", "row_scale", "innovation_scale", "degrees_of_freedom"}:
        raise AnalysisError("invalid_prior", "Supply every original proper MNIW prior field.")
    if not isinstance(value["mean"], (list, tuple)) or not isinstance(value["innovation_scale"], (list, tuple)):
        raise AnalysisError("invalid_prior", "Prior mean and innovation scale require complete matrices.")
    k, m = len(value["mean"]), len(value["innovation_scale"])
    if not 2 <= k <= 17 or not 2 <= m <= 4:
        raise AnalysisError("invalid_prior", "Original MNIW dimensions require k2..17 and m2..4.")
    if 128*(k**3+m**3)+50*size > max_work:
        raise AnalysisError("work_limit", "Full prior validation/copy/factor work exceeds max_work.")
    plan_workspace("complete public MNIW prior validation", {
        "complete primitive prior/state/copies/escaped output": 16*size,
        "original exact symmetric SPD factors": 128*(k*k+m*m)},
        budget_bytes=min(max_bytes, workspace_budget_bytes()))
    from openecon.econometrics.bayesian_var.posterior import MatrixNormalInverseWishartPrior
    from openecon.econometrics.bayesian_var.kernels import spd, tensor
    prior = MatrixNormalInverseWishartPrior.model_validate(value)
    spd(tensor(prior.row_scale), "prior row scale")
    spd(tensor(prior.innovation_scale), "prior innovation scale")
    return prior
