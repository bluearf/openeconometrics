"""Saved-parameter predictions and delta-method effects for scalar mean models.

Encoding is reconstructed from the *fitted* treatment levels and omitted terms,
never from the new sample. No estimator, optimizer or refit is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import wraps
from itertools import product
import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.glm.families import CANONICAL_LINK, FAMILY_LINKS, make_link
from openecon.econometrics.postest.inference import _Parameters, _alpha, _parameters, _scalar
from openecon.econometrics.postest.limited_prediction import ESTIMATORS, LimitedNormal, saved_normal
from openecon.econometrics.postest.linear_prediction import ESTIMATORS as LINEAR_ESTIMATORS, configure as configure_linear, validate_snapshot as validate_linear_snapshot
from openecon.econometrics.postest.mixture_prediction import ESTIMATORS as MIXTURE_ESTIMATORS, configure as configure_mixture
from openecon.econometrics.postest.group_response import ESTIMATORS as GROUP_ESTIMATORS, configure as configure_group, fixed_config
from openecon.econometrics.postest.advanced_prediction import ESTIMATORS as ADVANCED_ESTIMATORS, configure as configure_advanced
from openecon.econometrics.postest.control_prediction import ESTIMATORS as CONTROL_ESTIMATORS, configure as configure_control
from openecon.econometrics.postest.heteroskedastic_prediction import (
    HeteroskedasticProbit, check_workspace as check_heteroskedastic_workspace,
    parameter_uncertainty, predict_heteroskedastic, saved_heteroskedastic, variance_roles,
)
from openecon.econometrics.postest.ordinal_prediction import (
    ESTIMATORS as CATEGORY_ESTIMATORS, CategoryResponse, category_standard_error, predict_categories, saved_categories,
)
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.frame import as_frame
from openecon.dataset import Dataset
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, workspace_budget_bytes

# Explicit allow-list: absence of a response adapter is never treated as identity.
SUPPORTED = frozenset({"ols", "cnsreg", "ivregress", "logit", "probit", "glm", "poisson",
                       "nbreg", "cloglog", "fracreg", "betareg", "gnbreg", "cpoisson", "cnbreg",
                       "hetprobit", *ESTIMATORS, *CATEGORY_ESTIMATORS, *LINEAR_ESTIMATORS, *MIXTURE_ESTIMATORS, *GROUP_ESTIMATORS, *ADVANCED_ESTIMATORS, *CONTROL_ESTIMATORS})
_LINEAR = {"ols", "cnsreg", "ivregress"}
_PROVENANCE_FIELDS = {"estimator", "model", "postestimation", "categorical_encoding", "omitted_terms", "design_terms"}
_EXTRA_FIELDS = {
    "constrained_terms", "link", "family", "model", "quantiles", "equations", "limits",
    "inflate_link", "zero_link", "dist", "dispersion", "alpha", "truncation_column", "truncation_point",
    "variance_function", "variance_terms", "variance_regressor_means", "zero_outcomes", "nonzero_outcomes",
    "categories", "category_counts", "cutpoints", "category_order", "base",
    "group_state", "random_effects", "random_terms", "covstructure",
    "method", "select_link", "ll", "thresholds", "threshold_variable", "regions", "common",
    "distribution", "cost",
    "loss", "scale_equation_target", "scale_fixed_during_mm", "joint_estimating_equation_order",
    "exogenous", "endogenous", "instruments", "grouping", "levels", "jacobian_rank", "center",
}
_INFERENCE_FIELDS = {"use_t", "distribution", "df_inference", "df_resid", "alpha", "dfadjust", "hansen",
                     "covariance", "n_parameters", "scale_uncertainty", "cluster_count"}


def _error(code, message):
    raise AnalysisError(code, message)


def _heteroskedastic_cpu(function):
    @wraps(function)
    def scoped(result, *args, **kwargs):
        with torch.device("cpu"), torch.inference_mode(False):
            return function(result, *args, **kwargs)
    return scoped


def _role(result, name):
    value = result.spec.columns.get(name)
    if value is None or value == []:
        return None
    if not isinstance(value, str) or not value:
        _error("invalid_result", f"Saved {name} must identify one column.")
    return value


@dataclass
class _Model:
    result: ResultBundle
    state: _Parameters
    link: Any
    family: str
    predictors: list[str]
    categories: dict[str, list]
    features: dict[str, tuple[str, Any]]
    fixed: dict[str, float]
    mean_terms: set[str]
    offset: str | None
    exposure: str | None
    trials: str | None
    normal: LimitedNormal | None
    choice: CategoryResponse | None
    heteroskedastic: HeteroskedasticProbit | None
    response_adapter: Any = None

    def required(self, weights=False):
        extra = [] if self.response_adapter is None else self.response_adapter.required()
        return list(dict.fromkeys([*self.predictors, *extra, *[name for name in
            (self.offset, self.exposure, self.trials, self.result.spec.weights if weights else None)
            if name is not None]]))

    def intervention_variables(self):
        extra = [] if self.response_adapter is None else getattr(self.response_adapter,"intervention_variables",lambda:[])()
        return list(dict.fromkeys([*self.predictors,*extra]))


def _model(result, outcome=None, *, dataset=False):
    if not isinstance(result, ResultBundle):
        _error("invalid_result", "Provide a fitted OpenEconometrics ResultBundle.")
    estimator = result.spec.estimator
    postestimation = result.provenance.get("postestimation") or {}
    if not isinstance(postestimation, Mapping):
        _error("invalid_result", "Saved postestimation metadata are invalid.")
    if estimator not in SUPPORTED or postestimation.get("method") == "suest":
        _error("unsupported_prediction", f"Generic scalar predictions for '{estimator}' are not implemented; "
               "this model needs an explicit response adapter.")
    k = len(result.coefficients)
    plan_workspace("saved prediction parameter reconstruction", {
        "covariance_validation_and_snapshot": 96 * k * k,
        "coefficient_vector_and_reporting": 64 * k,
    }, budget_bytes=min(128 * 1024**2, workspace_budget_bytes()) if dataset else None)
    validate_linear_snapshot(result)
    if estimator in CONTROL_ESTIMATORS:
        state = _parameters(result)
        config = configure_control(result, state, outcome=outcome)
        # Full original state has been semantically replayed. Keep only compact
        # query roles and reporting fields; never revalidate/refit per batch.
        fields = {name: getattr(result, name) for name in ResultBundle.model_fields}
        fields.update(predictions=[], sample_positions=[], tests={}, warnings=[], extra={},
                      provenance={"estimator": estimator})
        result = ResultBundle.model_construct(**deepcopy(fields))
        adapter = config.pop("adapter")
        return _Model(result=result, state=state, normal=None, choice=None,
                      heteroskedastic=None, response_adapter=adapter, **config)
    # Freeze prediction roles/parameters, excluding row-sized fitted samples,
    # diagnostics, random effects and private caller frames. They are neither
    # needed nor valid substitutes for explicit new evaluation rows.
    fields = {name: getattr(result, name) for name in ResultBundle.model_fields}
    fields.update(predictions=[], sample_positions=[], tests={}, warnings=[])
    for name, needed in (("provenance", _PROVENANCE_FIELDS), ("extra", _EXTRA_FIELDS),
                         ("inference", _INFERENCE_FIELDS)):
        record = fields[name]
        if not isinstance(record, Mapping):
            _error("invalid_result", f"Saved {name} metadata are invalid.")
        fields[name] = {key: record[key] for key in needed if key in record}
    if estimator not in GROUP_ESTIMATORS:
        fields['extra'].pop('random_effects', None)
        if estimator != 'mixedflex':
            fields['extra'].pop('random_terms', None)
        fields['extra'].pop('covariance_structure', None)
    result = ResultBundle.model_construct(**deepcopy(fields))
    if (estimator == "frontier" and result.inference.get("use_t") is False
            and result.inference.get("distribution") in {"hnormal","tnormal","exponential"}
            and result.inference["distribution"] == result.extra.get("distribution")):
        # Historical frontier results used this field for the inefficiency
        # distribution. Their coefficient/contrast reference is standard normal.
        result.inference["distribution"] = "normal"
    if estimator in LINEAR_ESTIMATORS or estimator in MIXTURE_ESTIMATORS or estimator in GROUP_ESTIMATORS or estimator in ADVANCED_ESTIMATORS:
        state = _parameters(result)
        integrated = estimator in GROUP_ESTIMATORS or estimator in {'xtlogit', 'xtprobit', 'xtpoisson'} and '/lnsig2u' in state.terms
        config = configure_group(result, state, outcome=outcome) if integrated else configure_advanced(result, state, outcome=outcome) if estimator in ADVANCED_ESTIMATORS else configure_linear(result, state, outcome=outcome) if estimator in LINEAR_ESTIMATORS else configure_mixture(result, state)
        if estimator in LINEAR_ESTIMATORS and not integrated:
            state, config = fixed_config(result, state, config)
        adapter = config.pop("adapter")
        return _Model(result=result, state=state, normal=None, choice=None,
                      heteroskedastic=None, response_adapter=adapter, **config)
    variance_predictors = variance_roles(result)
    if result.spec.time or result.spec.panel or result.spec.options.get("formula"):
        _error("unsupported_prediction_design", "Panel/time/formula designs require their fitted prediction adapter.")
    state = _parameters(result)
    normal = saved_normal(result, state.terms)
    if normal is not None:
        normal.sigma(state.beta)
    predictors = [*result.spec.predictors, *variance_predictors]
    if estimator == "ivregress":
        endogenous = result.spec.columns.get("endogenous", [])
        predictors += [endogenous] if isinstance(endogenous, str) else list(endogenous)
    predictors = list(dict.fromkeys(predictors))
    categories, features = {}, {}
    coding = result.provenance.get("categorical_encoding", {})
    if not isinstance(coding, Mapping):
        _error("invalid_result", "Saved categorical encoding is invalid.")
    if result.spec.intercept:
        features["Intercept"] = ("constant", None)
    for name in predictors:
        if name not in result.spec.categorical:
            features[name] = ("numeric", name)
            continue
        record = coding.get(name)
        if (not isinstance(record, Mapping) or not isinstance(record.get("levels"), list)
                or len(record["levels"]) < 2 or record.get("reference") != record["levels"][0]):
            _error("invalid_result", f"The fitted category levels/reference for '{name}' are unavailable.")
        levels = list(record["levels"])
        try:
            pd.Categorical([], categories=levels)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_result", f"The fitted levels of '{name}' are invalid.") from exc
        categories[name] = levels
        for level in levels[1:]:
            term = f"{name}[{level}]"
            if term in features:
                _error("unsupported_prediction_design", "Saved feature names collide with generated category terms.")
            features[term] = ("category", (name, level))
    choice, features, choice_terms = saved_categories(result, state, features)
    heteroskedastic, features, heteroskedastic_terms = saved_heteroskedastic(result, state, features)
    mean_terms = heteroskedastic_terms if heteroskedastic is not None else set() if choice is None else choice_terms
    auxiliary = {"nbreg": {"/lnalpha", "/lndelta"}, "cnbreg": {"/lnalpha", "/lndelta"},
                 "betareg": {"scale"}, "gnbreg": {"lnalpha"}}
    for coefficient in result.coefficients:
        if choice is not None or heteroskedastic is not None:
            continue  # Dedicated adapter already validated every slope/equation/cutpoint.
        if coefficient.term in features and coefficient.equation in {None, result.spec.outcome}:
            mean_terms.add(coefficient.term)
        elif ((estimator in {"nbreg", "cnbreg"} and coefficient.term in auxiliary[estimator]
               and coefficient.equation is None)
              or (estimator in {"betareg", "gnbreg"}
                  and coefficient.equation in auxiliary[estimator]
                  and coefficient.term.startswith(coefficient.equation + ":"))
              or (normal is not None and coefficient.term == state.terms[normal.scale_index]
                  and coefficient.equation is None)):
            pass
        else:
            _error("unsupported_prediction_design", f"Saved term '{coefficient.term}' is not a supported mean/ancillary term.")
    fixed_record = result.extra.get("constrained_terms", {}) if estimator == "cnsreg" else {}
    if not isinstance(fixed_record, Mapping):
        _error("invalid_result", "Saved fixed coefficients are invalid.")
    fixed = {}
    for term, value in fixed_record.items():
        if term not in features or term in mean_terms:
            _error("invalid_result", "Saved fixed coefficients conflict with the reported design.")
        fixed[term] = _scalar(value, code="invalid_result")
    omitted = result.provenance.get("omitted_terms", [])
    if not isinstance(omitted, list) or any(not isinstance(name, str) for name in omitted):
        _error("invalid_result", "Saved omitted terms are invalid.")
    if set(features) - mean_terms - set(fixed) - set(omitted):
        _error("invalid_result", "Saved mean coefficients do not account for all fitted features and omissions.")
    if not mean_terms and not fixed and choice is None:
        _error("invalid_result", "The result has no supported mean coefficients.")
    if choice is not None:
        family, link_name = "categorical", "identity"
    elif estimator in _LINEAR or normal is not None:
        family, link_name = "gaussian", "identity"
    elif estimator in {"logit", "probit", "cloglog", "hetprobit"}:
        family, link_name = "binomial", {"logit": "logit", "probit": "probit", "cloglog": "cloglog", "hetprobit": "probit"}[estimator]
    elif estimator in {"fracreg", "betareg"}:
        family, link_name = "binomial", result.spec.options.get("link", "logit")
    elif estimator == "glm":
        family = result.spec.options.get("family", "gaussian")
        link_name = result.spec.options.get("link") or CANONICAL_LINK.get(family)
        if family not in FAMILY_LINKS or link_name not in FAMILY_LINKS[family]:
            _error("invalid_result", "The saved GLM family/link pair is invalid.")
    else:
        family, link_name = "poisson", "log"
    try:
        link = make_link(link_name, power=result.spec.options.get("power"),
                         k=result.spec.options.get("dispersion", 1.))
    except (KernelError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_result", "The saved mean link is invalid.") from exc
    if choice is None and (result.extra.get("link", link.name) != link.name
            or result.extra.get("family", family) != family):
        _error("invalid_result", "The fitted mean link/family disagree with the saved specification.")
    offset, exposure, trials = (_role(result, role) for role in ("offset", "exposure", "trials"))
    if ((offset and exposure) or (trials and (estimator != "glm" or family != "binomial"))
            or (normal is not None and exposure)
            or (choice is not None and (exposure or trials or estimator == "mlogit" and offset))):
        _error("invalid_result", "The saved offset/exposure/trials combination is invalid.")
    return _Model(result, state, link, family, predictors, categories, features, fixed,
                  mean_terms, offset, exposure, trials, normal, choice, heteroskedastic)


def _data(model, data, *, weights=False):
    if data is None:
        _error("prediction_data_required", "Pass evaluation data explicitly; a saved chart sample is not an estimation dataset.")
    frame = _coerce_frame(data)
    if model.result.spec.estimator in {"sreg", "mmreg", "ivcue", "mixedflex", *CONTROL_ESTIMATORS}:
        plan_workspace("saved linear target evaluation", {
            "encoded_design_features_and_delta_buffers": 48 * len(frame) * max(1, len(model.state.terms)),
            "row_values_inference_positions_and_scatter": 128 * len(frame),
        })
    if frame.columns.has_duplicates:
        _error("duplicate_columns", "Prediction data must have unique column names.")
    required = model.required(weights)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        _error("missing_columns", "Prediction data lack required fitted covariates: " + ", ".join(missing) + ".")
    keep = ~frame.loc[:, required].isna().any(axis=1)
    if not bool(keep.all()) and model.result.spec.missing == "raise":
        _error("missing_values", "Prediction inputs contain missing values; this fit records missing='raise'.")
    positions = torch.tensor([i for i, present in enumerate(keep.tolist()) if present], dtype=torch.int64)
    retained = frame.iloc[positions.tolist()].loc[:, required].copy(deep=False)
    if not weights:
        _check_categories(model, retained)
    return frame, retained, positions


def _check_categories(model, frame):
    for name, levels in model.categories.items():
        if len(frame) and bool((pd.Categorical(frame[name], categories=levels).codes < 0).any()):
            _error("unknown_category", f"Column '{name}' contains a category absent from the fitted levels.")


@dataclass
class _Design:
    x: torch.Tensor
    deterministic: torch.Tensor
    scale: torch.Tensor


def _encode(model, frame):
    n, k = len(frame), len(model.state.terms)
    if 8 * n * max(1, k) > _MAX_DESIGN_BYTES:
        _error("prediction_memory_limit", "Prediction design exceeds 256 MiB; predict in smaller batches.")
    x = torch.zeros((n, k), dtype=torch.float64)
    deterministic = torch.zeros(n, dtype=torch.float64)
    columns = {}
    used = model.mean_terms | set(model.fixed)
    for term, (kind, value) in model.features.items():
        if term not in used:
            continue
        if kind == "constant":
            column = torch.ones(n, dtype=torch.float64)
        elif kind == "numeric":
            column = _numeric(frame[value], value)
        else:
            name, level = value
            if len(frame) and bool((pd.Categorical(frame[name], categories=model.categories[name]).codes < 0).any()):
                _error("unknown_category", f"Column '{name}' has an unfitted category.")
            column = torch.tensor((frame[name] == level).tolist(), dtype=torch.float64)
        columns[term] = column
    for index, term in enumerate(model.state.terms):
        if term in model.mean_terms:
            x[:, index] = columns[term]
    for term, value in model.fixed.items():
        deterministic += columns[term] * value
    if model.offset:
        deterministic += _numeric(frame[model.offset], model.offset)
    if model.exposure:
        exposure = _numeric(frame[model.exposure], model.exposure)
        if bool((exposure <= 0).any()):
            _error("invalid_exposure", "Prediction exposure must be strictly positive.")
        deterministic += exposure.log()
    scale = torch.ones(n, dtype=torch.float64)
    if model.trials:
        scale = _numeric(frame[model.trials], model.trials)
        if bool(((scale < 1) | (scale != scale.round())).any()):
            _error("invalid_trials", "Prediction binomial trials must be positive integers.")
    design = _Design(x, deterministic, scale)
    return design if model.response_adapter is None else model.response_adapter.encode(frame, design)


def _link_eta(model, eta):
    # Beyond these bounds the inverse link and both float64 derivatives have
    # already rounded to their limiting values. Bound only the nested intermediate
    # exponent/square so backward passes never form 0*inf at exact saturation.
    bound = math.log(torch.finfo(torch.float64).max) - 1
    if model.link.name == "cloglog":
        return eta.clamp(max=bound)
    if model.link.name == "loglog":
        return eta.clamp(min=-bound)
    if model.link.name == "probit":
        square_bound = math.sqrt(torch.finfo(torch.float64).max) / 4
        return eta.clamp(min=-square_bound, max=square_bound)
    return eta


def _mean(model, design, beta, kind="response"):
    if model.response_adapter is not None:
        return model.response_adapter.values(design, beta, kind)
    if model.heteroskedastic is not None:
        return model.heteroskedastic.values(design, beta, kind)
    if model.choice is not None:
        return model.choice.values(design, beta, kind)
    eta = design.x @ beta + design.deterministic
    if not bool(torch.isfinite(eta).all()):
        _error("non_finite_prediction", "The saved coefficients produce nonfinite linear predictions.")
    if kind == "xb":
        return eta
    if model.normal is not None:
        return model.normal.evaluate(eta, beta, kind)[0]
    if not bool(model.link.valid(eta).all()):
        _error("prediction_domain", "Prediction inputs lie outside the fitted link's domain.")
    value = model.link.inverse(_link_eta(model, eta))
    if (not bool(torch.isfinite(value).all())
            or (model.family == "binomial" and bool(((value < 0) | (value > 1)).any()))
            or (model.family not in {"gaussian", "binomial"} and bool((value <= 0).any()))):
        _error("prediction_domain", "The fitted inverse link does not give valid finite response means.")
    return value * design.scale


def _slope(model, beta, variable):
    if variable not in model.predictors or variable in model.categories:
        _error("invalid_prediction_term", "Choose one continuous original predictor for derivatives.")
    if variable in {model.offset, model.exposure, model.trials}:
        _error("unsupported_margins_transform", "A predictor also used as offset/exposure/trials needs an explicit intervention adapter.")
    if variable in model.mean_terms:
        return beta[model.state.terms.index(variable)]
    return beta.sum() * 0 + model.fixed.get(variable, 0.)


def _effect(model, design, beta, variable, kind):
    if model.response_adapter is not None:
        if variable in {model.offset, model.exposure, model.trials}:
            _error("unsupported_margins_transform", "A predictor also used as offset/exposure/trials needs an explicit intervention adapter.")
        return model.response_adapter.effects(design, beta, variable, kind)
    slope = _slope(model, beta, variable)
    if model.choice is not None:
        return model.choice.effects(design, beta, variable, kind)
    if model.heteroskedastic is not None:
        return model.heteroskedastic.effects(design, beta, variable, kind)
    if kind == "xb":
        return torch.ones(len(design.x), dtype=torch.float64) * slope
    eta = design.x @ beta + design.deterministic
    if model.normal is not None:
        return model.normal.evaluate(eta, beta, kind)[1] * slope
    response = _mean(model, design, beta) / design.scale
    return design.scale * model.link.derivative(_link_eta(model, eta), response) * slope


def _kind(kind, model):
    aliases = {"linear": "xb", "mean": "response", "mu": "response", "fitted": "response",
               "dydx": "derivative"}
    if model.choice is not None or model.heteroskedastic is not None:
        aliases.update(pr="response", probability="response")
    kind = aliases.get(kind, kind) if isinstance(kind, str) else None
    if model.response_adapter is not None:
        check = "response" if kind == "derivative" else kind
        if check not in model.response_adapter.kinds:
            _error("unsupported_prediction_kind", "This saved model requires its fitted response state, including group effects where applicable, for that prediction kind; use an explicitly supported index or population response.")
        return kind
    if kind == "sigma" and model.heteroskedastic is not None:
        return kind
    if kind in {"latent", "conditional"}:
        if model.normal is None or (kind == "conditional" and model.normal.estimator == "intreg"):
            _error("unsupported_prediction_kind", "Latent/conditional normal means require their fitted limited-response adapter and limits.")
        return kind
    if kind not in {"xb", "response", "stdp", "derivative"}:
        _error("unsupported_prediction_kind", "Use response, xb, stdp, or derivative with a continuous term.")
    return kind


@_heteroskedastic_cpu
def predict(result, data=None, kind="response", alpha=None, *, interval=None, term=None, outcome=None, batch_rows=None, target=None, random_effects=None, max_group_rows=100000, max_disk_bytes=1073741824):
    """Predict scalar means/linear indexes, optionally with delta-method mean CIs.

    Required covariates depend on the saved target (CF includes instruments).
    Missing='drop' preserves the
    input index and fills excluded rows with NaN. Observation/prediction bands,
    residuals, latent classifications and random/fixed effects are separate.
    """
    if not isinstance(result, ResultBundle):
        _error('invalid_result', 'Provide a fitted OpenEconometrics ResultBundle.')
    grouped = result.spec.estimator == 'clogit' or result.spec.estimator == 'xtlogit' and result.spec.options.get('model', 're') == 'fe'
    if grouped or target == 'posterior':
        if random_effects is not None or term is not None or outcome is not None:
            _error('unsupported_prediction_target', 'Complete-group evaluation does not accept explicit effects, derivative terms or outcome selection.')
        from .grouped_prediction import group_predict
        return group_predict(result, data, kind=kind, target='conditional' if grouped and target is None else target,
                             interval=interval, alpha=alpha, batch_rows=batch_rows, max_group_rows=max_group_rows,
                             max_disk_bytes=max_disk_bytes)
    model = _model(result, outcome=outcome, dataset=isinstance(data, Dataset))
    kind = _kind(kind, model)
    if model.response_adapter is not None and (hasattr(model.response_adapter, 'set_target') or hasattr(model.response_adapter, 'record')):
        model.response_adapter.evaluation_kind = kind
    if target is not None or random_effects is not None:
        setter = None if model.response_adapter is None else getattr(model.response_adapter, 'set_target', None)
        if setter is None:
            if target != 'conditional' or random_effects is not None or not result.extra.get('group_state'):
                _error('unsupported_prediction_target', 'This result has no adapter for the requested group target.')
        else:
            setter(target or 'conditional', random_effects)
    if (not (interval is None or interval == "mean")
            or (kind == "stdp" and interval is not None)):
        _error("unsupported_prediction_interval", "Only mean delta-method confidence intervals are supported.")
    if (kind == "derivative") != (term is not None):
        _error("invalid_prediction_term", "Supply term only with kind='derivative'.")
    if kind == "derivative" and (term not in model.intervention_variables() or term in model.categories):
        _error("invalid_prediction_term", "Choose one continuous original predictor for derivatives.")
    significance = model.state.alpha if alpha is None else _alpha(alpha)
    if isinstance(data, Dataset):
        from .streaming_prediction import predict_dataset
        return predict_dataset(model, data, kind=kind, alpha=significance,
                               interval=interval, term=term, outcome=outcome, batch_rows=batch_rows)
    if batch_rows is not None:
        _error("invalid_batch_size", "batch_rows applies only to Dataset evaluation data.")
    return _prediction_frame(model, data, kind=kind, significance=significance,
                             interval=interval, term=term, outcome=outcome)


def _prediction_frame(model, data, *, kind, significance, interval, term, outcome):
    """Evaluate one block using an already validated parameter/design snapshot."""
    result = model.result
    original, frame, positions = _data(model, data)
    design = _encode(model, frame)
    if model.choice is not None:
        return predict_categories(model, design, original, positions, kind, outcome,
                                  significance, interval, term)
    if outcome is not None and not (model.response_adapter is not None and model.response_adapter.accepts_outcome):
        _error("unsupported_prediction_outcome", "outcome selection requires a fitted categorical-response adapter.")
    if model.heteroskedastic is not None:
        return predict_heteroskedastic(model, design, original, positions, kind,
                                      significance, interval, term)
    eta = _mean(model, design, model.state.beta, "xb")
    jac = design.x
    label = "dydx[" + str(term) + "]" if kind == "derivative" else kind
    if model.response_adapter is not None:
        value = model.response_adapter.effects(design, model.state.beta, term, "response") if kind == "derivative" else model.response_adapter.values(design, model.state.beta, kind)
        if not (getattr(model.response_adapter, 'point_only', False) and interval is None and kind == 'response'):
            jac = model.response_adapter.jacobian(design, model.state.beta, kind, variable=term)
    elif model.normal is not None and kind not in {"xb", "stdp"}:
        response_kind = "response" if kind == "derivative" else kind
        value, first, scale_first, second, cross = model.normal.evaluate(eta, model.state.beta, response_kind)
        jac = first[:, None] * design.x
        chain = model.normal.sigma(model.state.beta) if model.normal.log_scale else 1.
        jac[:, model.normal.scale_index] += scale_first * chain
        if kind == "derivative":
            slope = _slope(model, model.state.beta, term)
            value = first * slope
            jac = second[:, None] * slope * design.x
            jac[:, model.normal.scale_index] += cross * slope * chain
            if term in model.mean_terms:
                jac[:, model.state.terms.index(term)] += first
    elif kind == "derivative":
        value = _effect(model, design, model.state.beta, term, "response")
        response = _mean(model, design, model.state.beta) / design.scale
        stable_eta = _link_eta(model, eta)
        derivative = model.link.derivative(stable_eta, response)
        second = derivative * model.link.second_ratio(stable_eta, response)
        slope = _slope(model, model.state.beta, term)
        jac = design.scale[:, None] * second[:, None] * slope * design.x
        if term in model.mean_terms:
            jac[:, model.state.terms.index(term)] += design.scale * derivative
    elif kind == "response":
        value = _mean(model, design, model.state.beta)
        jac = design.scale[:, None] * model.link.derivative(_link_eta(model, eta), value / design.scale)[:, None] * design.x
    else:
        value = eta
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(jac).all()):
        _error("non_finite_prediction", "Predicted values or parameter derivatives are nonfinite.")
    columns = {}
    if kind == "stdp" or interval is not None:
        variance = torch.einsum("nk,kl,nl->n", jac, model.state.covariance, jac)
        tolerance = 64 * torch.finfo(torch.float64).eps * torch.einsum(
            "nk,kl,nl->n", jac.abs(), model.state.covariance.abs(), jac.abs()).clamp_min(1e-300)
        if not bool(torch.isfinite(variance).all()) or bool((variance < -tolerance).any()):
            _error("invalid_inference", "Prediction covariance did not produce finite nonnegative variances.")
        se = variance.clamp_min(0).sqrt()
        if kind == "stdp":
            value = se
        else:
            critical = critical_value(significance, model.state.df)
            columns.update(std_error=se, ci_low=value - critical * se, ci_high=value + critical * se)
    columns = {label: value, **columns}
    output = {}
    for name, values in columns.items():
        if not bool(torch.isfinite(values).all()):
            _error("non_finite_prediction", "Prediction intervals exceed finite float64 range.")
        scattered = torch.full((len(original),), float("nan"), dtype=torch.float64)
        scattered[positions] = values
        output[name] = scattered.tolist()
    output = as_frame(pd.DataFrame(output, index=original.index))
    present = set(positions.tolist())
    output.attrs.update(kind=kind, estimator=result.spec.estimator, precision="float64",
                        missing_row_positions=[i for i in range(len(original)) if i not in present],
                        interval_method="pointwise delta method" if interval else None,
                        response_definition=model.response_adapter.definition(kind) if model.response_adapter is not None else model.normal.definition(kind) if model.normal is not None else
                        "latent unconditional E[Y|X]" if result.spec.estimator in
                        {"cpoisson", "cnbreg"} else "binomial count mean" if model.trials else "conditional response mean")
    return output


def _effect_inference(state, estimate, gradient, *, categorical=False, std_error=None):
    if std_error is not None:
        se = float(std_error)
    elif categorical:
        se = float(category_standard_error(gradient, state.covariance))
    else:
        variance = float(gradient @ state.covariance @ gradient)
        absolute = float(gradient.abs() @ state.covariance.abs() @ gradient.abs())
        if not math.isfinite(variance) or variance < -64 * torch.finfo(torch.float64).eps * max(absolute, 1e-300):
            _error("invalid_inference", "Marginal-effect variance must be finite and nonnegative.")
        se = math.sqrt(max(0., variance))
    statistic = estimate / se if se else None
    critical = critical_value(state.alpha, state.df)
    values = [estimate, se, estimate - critical * se, estimate + critical * se]
    if not all(math.isfinite(value) for value in values) or (statistic is not None and not math.isfinite(statistic)):
        _error("invalid_inference", "Marginal-effect inference exceeds finite float64 range.")
    return {"estimate": estimate, "std_error": se, "statistic": statistic,
            "distribution": "t" if state.df is not None else "normal", "df": state.df,
            "p_value": None if statistic is None else student_t_two_sided(statistic, state.df)
            if state.df is not None else math.erfc(abs(statistic) / math.sqrt(2)),
            "ci_low": values[2], "ci_high": values[3], "alpha": state.alpha}


def _margins_setting(model, variables, method, kind, selected, setting, encoded, weights, *, inference=True):
    """One intervention grid, using a bounded row design or its global mean."""
    parameter_scale, correlation = parameter_uncertainty(model.state) if model.heteroskedastic is not None else (None, None)
    make_design = encoded
    def encoded(overrides):
        design = make_design(overrides)
        if design.x.is_inference() or design.deterministic.is_inference() or design.scale.is_inference():
            return _Design(design.x.detach().clone(), design.deterministic.detach().clone(), design.scale.detach().clone())
        return design
    def aggregate(values):
        return values[0] if method == "mem" else weights @ values
    rows, gradients, scaled_gradients = [], [], []
    for variable in variables:
        contrasts = [(variable, None)] if variable not in model.categories else [
            (f"{variable}[{level}]", level) for level in model.categories[variable][1:]]
        for label, level in contrasts:
            with torch.inference_mode(False), torch.enable_grad():
                weights = weights.detach().clone()
                beta = model.state.beta.detach().clone().requires_grad_(model.choice is None and model.heteroskedastic is None)
                if level is None:
                    design = encoded({})
                    effect = aggregate(_effect(model, design, beta, variable, kind))
                    category_gradient = model.choice.jacobian(design, beta, "derivative_xb" if kind == "xb" else "derivative", variable) if model.choice is not None else None
                    heteroskedastic_gradient = model.heteroskedastic.jacobian(design, beta, "derivative_" + kind if kind in {"xb", "sigma"} else "derivative", variable) if model.heteroskedastic is not None else None
                    heteroskedastic_scaled_gradient = model.heteroskedastic.jacobian(design, beta, "derivative_" + kind if kind in {"xb", "sigma"} else "derivative", variable, parameter_scale) if model.heteroskedastic is not None else None
                else:
                    alternate_design = encoded({variable: level})
                    baseline_design = encoded({variable: model.categories[variable][0]})
                    adapter = model.choice if model.choice is not None else model.heteroskedastic
                    effect = aggregate(adapter.difference(alternate_design, baseline_design, beta, kind) if adapter is not None else _mean(model, alternate_design, beta, kind) - _mean(model, baseline_design, beta, kind))
                    category_gradient = (model.choice.jacobian(alternate_design, beta, kind) - model.choice.jacobian(baseline_design, beta, kind)) if model.choice is not None else None
                    heteroskedastic_gradient = (model.heteroskedastic.jacobian(alternate_design, beta, kind) - model.heteroskedastic.jacobian(baseline_design, beta, kind)) if model.heteroskedastic is not None else None
                    heteroskedastic_scaled_gradient = (model.heteroskedastic.jacobian(alternate_design, beta, kind, parameter_scale=parameter_scale) - model.heteroskedastic.jacobian(baseline_design, beta, kind, parameter_scale=parameter_scale)) if model.heteroskedastic is not None else None
                if category_gradient is not None:
                    category_gradient = category_gradient[0] if method == "mem" else torch.einsum("n,njk->jk", weights, category_gradient)
                if heteroskedastic_gradient is not None:
                    heteroskedastic_gradient = heteroskedastic_gradient[0] if method == "mem" else weights @ heteroskedastic_gradient
                    heteroskedastic_scaled_gradient = heteroskedastic_scaled_gradient[0] if method == "mem" else weights @ heteroskedastic_scaled_gradient
                indices = selected if selected is not None else (None,)
                for number, index in enumerate(indices):
                    scalar = effect if index is None else effect[index]
                    if category_gradient is not None:
                        gradient = category_gradient[index]
                    elif heteroskedastic_gradient is not None:
                        gradient = heteroskedastic_gradient
                    else:
                        gradient, = torch.autograd.grad(scalar, beta, retain_graph=number < len(indices)-1)
                    estimate, gradient = float(scalar.detach()), gradient.detach()
                    if not bool(torch.isfinite(gradient).all()):
                        _error("invalid_margins", "Marginal-effect derivatives are not finite.")
                    outcome_record = {} if selected is None else {"outcome":
                        None if model.choice.estimator != "mlogit" and kind == "xb" else model.choice.labels[index]}
                    inference_record = _effect_inference(model.state, estimate, gradient,
                        categorical=model.choice is not None or model.heteroskedastic is not None,
                        std_error=category_standard_error(heteroskedastic_scaled_gradient, correlation) if model.heteroskedastic is not None else None) if inference else {"estimate": estimate}
                    rows.append({"variable": label, "method": method, **outcome_record,
                                 **{f"at[{name}]": value for name, value in setting.items()},
                                 **inference_record})
                    gradients.append(gradient.tolist())
                    scaled_gradients.append(heteroskedastic_scaled_gradient.tolist() if model.heteroskedastic is not None else None)
    return rows, gradients, scaled_gradients


@_heteroskedastic_cpu
def margins(result, variables=None, *, data=None, method="ame", at=None, kind="response", alpha=None, outcome=None, batch_rows=None):
    """Effects over explicit evaluation rows (AME) or their encoded means (MEM).

    Continuous predictors use analytic inverse-link derivatives. Categories use
    discrete response changes against their fitted reference. Covariance uses
    the gradient of each aggregate effect with respect to saved parameters.
    """
    model = _model(result, outcome=outcome, dataset=isinstance(data, Dataset))
    result = model.result
    if (not isinstance(method, str) or not isinstance(kind, str)
            or method not in {"ame", "mem"} or kind not in {"response", "xb", "latent", "conditional", *( ["sigma"] if model.heteroskedastic is not None else [])}):
        _error("invalid_margins", "Use method='ame'/'mem' and kind='response'/'xb'.")
    _kind(kind, model)
    if model.response_adapter is not None:
        if hasattr(model.response_adapter, 'set_target') or hasattr(model.response_adapter, 'record'):
            model.response_adapter.evaluation_kind = kind
        if method == 'mem' and kind != 'xb' and (hasattr(model.response_adapter, 'names') or hasattr(model.response_adapter, 'record')):
            _error('unsupported_margins_method', 'Group response MEM requires a declared representative random/group state; use AME over explicit rows.')
    if model.choice is None and outcome is not None and not (model.response_adapter is not None and model.response_adapter.accepts_outcome):
        _error("unsupported_prediction_outcome", "outcome selection requires a fitted categorical-response adapter.")
    selected = model.choice.selection(outcome, kind) if model.choice is not None else None
    model.state = replace(model.state, alpha=model.state.alpha if alpha is None else _alpha(alpha))
    variables = model.predictors if variables is None else [variables] if isinstance(variables, str) else variables
    if (not isinstance(variables, (list, tuple)) or not variables
            or any(not isinstance(name, str) or name not in model.intervention_variables() for name in variables)
            or len(set(variables)) != len(variables)):
        _error("invalid_margins", "Select distinct original mean predictor names.")
    if isinstance(data, Dataset):
        from .streaming_prediction import margins_dataset
        return margins_dataset(model, data, variables, method=method, at=at, kind=kind,
                               outcome=outcome, selected=selected, batch_rows=batch_rows)
    if batch_rows is not None:
        _error("invalid_batch_size", "batch_rows applies only to Dataset evaluation data.")
    original, base, _ = _data(model, data, weights=True)
    if not len(base):
        _error("empty_sample", "Marginal effects require at least one complete evaluation row.")
    weights = torch.ones(len(base), dtype=torch.float64)
    if result.spec.weights:
        weights = _numeric(base[result.spec.weights], result.spec.weights)
        if bool((weights < 0).any()) or not float(weights.sum()) > 0:
            _error("invalid_weights", "Marginal-effect weights must be nonnegative with a positive sum.")
        if result.spec.weight_type == "fweight" and bool((weights != weights.round()).any()):
            _error("invalid_weights", "Frequency weights must be integers.")
    complete_rows = len(base)
    positive = weights > 0
    base = base.iloc[positive.nonzero().flatten().tolist()].copy(deep=False)
    weights = weights[positive]
    _check_categories(model, base)
    weights = weights / weights.max()
    weights = weights / weights.sum()
    at = {} if at is None else at
    if not isinstance(at, Mapping) or any(name not in model.intervention_variables() for name in at):
        _error("invalid_margins", "at must map original predictors to scalar values or finite grids.")
    grids = []
    for name, value in at.items():
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not values:
            _error("invalid_margins", "Evaluation grids must not be empty.")
        for point in values:
            if name in model.categories:
                if point not in model.categories[name]:
                    _error("unknown_category", f"Column '{name}' has an unfitted at category.")
            else:
                _scalar(point, code="invalid_margins")
        grids.append((name, values))
    if math.prod(len(values) for _, values in grids) > 1000:
        _error("margins_grid_limit", "Use at most 1000 evaluation-grid combinations.")

    def encoded(frame):
        design = _encode(model, frame)
        if method == "mem":
            if model.heteroskedastic is not None:
                check_heteroskedastic_workspace(1, len(model.state.terms))
            return _Design((weights @ design.x)[None], (weights @ design.deterministic)[None],
                           (weights @ design.scale)[None])
        if model.choice is not None and 8 * len(design.x) * max(8, 4 * len(model.choice.labels)) * len(model.state.terms) > _MAX_DESIGN_BYTES:
            _error("prediction_memory_limit", "Category marginal-effect gradients exceed 256 MiB; use fewer evaluation rows.")
        if model.heteroskedastic is not None:
            check_heteroskedastic_workspace(len(design.x), len(model.state.terms))
        return design

    rows, gradients, scaled_gradients = [], [], []
    for values in product(*(values for _, values in grids)):
        setting = dict(zip((name for name, _ in grids), values, strict=True))
        def scenario_design(overrides):
            scenario = base.copy(deep=True)
            for name, value in {**setting, **overrides}.items():
                scenario[name] = value
            return encoded(scenario)
        new_rows, new_gradients, new_scaled = _margins_setting(
            model, variables, method, kind, selected, setting, scenario_design, weights)
        rows.extend(new_rows)
        gradients.extend(new_gradients)
        scaled_gradients.extend(new_scaled)
    output = as_frame(pd.DataFrame(rows))
    output.attrs.update(estimator=result.spec.estimator, evaluation_rows=len(base),
                        input_evaluation_rows=len(original), complete_evaluation_rows=complete_rows,
                        zero_weight_rows_excluded=complete_rows - len(base),
                        delta_gradients=gradients, delta_scaled_gradients=scaled_gradients,
                        parameter_terms=list(model.state.terms),
                        covariance_source="saved fitted parameter covariance", kind=kind,
                        mem_definition="at weighted means of encoded design, offset and trials",
                        evaluation_sample="explicit supplied rows; outcomes not required")
    if model.choice is not None:
        output.attrs.update(outcome_labels=list(model.choice.labels), outcome=model.choice.labels[selected[0]] if outcome is not None else None,
                            base_outcome=model.choice.labels[model.choice.base] if model.choice.base is not None else None)
    return output
