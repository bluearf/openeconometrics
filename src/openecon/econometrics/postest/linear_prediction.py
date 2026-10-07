"""Saved scalar prediction contracts for linear and population-mean models.

These adapters never estimate a model or infer unrecorded fixed effects.  A
panel/time field may describe the fit's covariance without transforming its
response design (Prais, PCSE, GLS and GEE).  First-difference fits are different:
their input transformation is essential and cannot be discarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import pandas as pd
import torch
from pydantic import ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.glm.families import FAMILY_LINKS, make_link
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec

ESTIMATORS = frozenset({
    "areg", "reghdfe", "ivreghdfe", "xtreg", "xtivreg", "xtgls", "xtpcse",
    "xtfmb", "prais", "rreg", "qreg", "bsqreg", "iqreg", "sqreg", "xtgee",
    "ppmlhdfe", "mixed", "xtlogit", "xtprobit", "xtpoisson",
})
_ABSORBED = {"areg", "reghdfe", "ivreghdfe", "ppmlhdfe"}
_IV = {"ivreghdfe", "xtivreg"}
_PANEL_LIKELIHOODS = {"xtlogit", "xtprobit", "xtpoisson"}


def _error(code: str, message: str):
    raise AnalysisError(code, message)


def _role(spec, name):
    value = spec.columns.get(name)
    if value is None or value == []:
        return None
    if not isinstance(value, str) or not value:
        _error("invalid_result", f"Saved {name} must identify one column.")
    return value


@dataclass(frozen=True)
class ScalarResponse:
    """Single saved equation; the numerical design retains all K parameters."""

    link: Any
    family: str
    terms: tuple[str, ...]
    continuous: dict[str, str | None]
    kinds: frozenset[str]
    response_definition: str
    xb_definition: str
    equation: str | None = None
    accepts_outcome: bool = False

    def required(self):
        return []

    def encode(self, frame, design):
        # Shared encoding has already reconstructed the saved features, offset
        # and exposure. No grouping/time transformation belongs to this design.
        return design

    def definition(self, kind):
        return self.xb_definition if kind in {"xb", "stdp"} else self.response_definition

    def _check_kind(self, kind):
        if kind not in self.kinds:
            _error("unsupported_prediction_kind", "This saved result has no absorbed-effect response state. "
                   "Use kind='xb' for the explicitly unabsorbed linear index; conditional fitted means "
                   "and nonlinear effects require saved group effects.")

    def _eta(self, design, beta):
        eta = design.x @ beta + design.deterministic
        if not bool(torch.isfinite(eta).all()):
            _error("non_finite_prediction", "Saved coefficients produce nonfinite linear predictions.")
        return eta

    def _stable_eta(self, eta):
        # Asymptotic limits only bound nested exponentials, not probabilities
        # or positive means. The same limits have zero float64 derivatives.
        bound = math.log(torch.finfo(torch.float64).max) - 4
        if self.link.name == "cloglog":
            return eta.clamp(max=bound)
        if self.link.name == "loglog":
            return eta.clamp(min=-bound)
        if self.link.name == "probit":
            square_bound = math.sqrt(torch.finfo(torch.float64).max) / 4
            return eta.clamp(min=-square_bound, max=square_bound)
        return eta

    def _response(self, eta):
        if not bool(self.link.valid(eta).all()):
            _error("prediction_domain", "Prediction inputs lie outside the saved link's domain.")
        stable = self._stable_eta(eta)
        if self.link.name == "logit":
            # torch.sigmoid's backward uses its rounded forward probability:
            # at eta=40 it becomes exactly zero. The complement representation
            # retains the correct, representable derivative in that upper tail.
            response = torch.where(stable >= 0, 1 - torch.sigmoid(-stable), torch.sigmoid(stable))
        else:
            response = self.link.inverse(stable)
        invalid = (not bool(torch.isfinite(response).all())
                   or (self.family == "binomial" and bool(((response < 0) | (response > 1)).any()))
                   or (self.family not in {"gaussian", "binomial"} and bool((response <= 0).any())))
        if invalid:
            _error("prediction_domain", "The saved inverse link does not give finite response means in its domain.")
        return stable, response

    def _slope(self, beta, variable):
        if variable not in self.continuous:
            _error("invalid_prediction_term", "Choose one continuous original predictor for derivatives.")
        term = self.continuous[variable]
        # An explicitly omitted predictor has a zero effect, with a zero delta
        # gradient. Linking this zero to beta retains an autograd graph.
        return beta.sum() * 0 if term is None else beta[self.terms.index(term)]

    def values(self, design, beta, kind):
        self._check_kind(kind)
        eta = self._eta(design, beta)
        if kind in {"xb", "stdp"}:
            return eta
        return self._response(eta)[1] * design.scale

    def effects(self, design, beta, variable, kind):
        self._check_kind(kind)
        slope = self._slope(beta, variable)
        if kind == "xb":
            return torch.ones_like(design.deterministic) * slope
        eta, response = self._response(self._eta(design, beta))
        return design.scale * self.link.derivative(eta, response) * slope

    def jacobian(self, design, beta, kind, variable=None):
        if variable is None and kind in {"xb", "stdp"}:
            self._check_kind(kind)
            return design.x
        response_kind = "response" if kind == "derivative" else kind
        self._check_kind(response_kind)
        if variable is not None and response_kind == "xb":
            self._slope(beta, variable)  # validate the requested original predictor
            output = torch.zeros_like(design.x)
            term = self.continuous[variable]
            if term is not None:
                output[:, self.terms.index(term)] = 1.
            return output
        eta, response = self._response(self._eta(design, beta))
        first = design.scale * self.link.derivative(eta, response)
        if variable is None:
            return first[:, None] * design.x
        slope = self._slope(beta, variable)
        second = first * self.link.second_ratio(eta, response)
        output = second[:, None] * slope * design.x
        term = self.continuous[variable]
        if term is not None:
            output[:, self.terms.index(term)] += first
        if not bool(torch.isfinite(output).all()):
            _error("non_finite_prediction", "Saved effects have nonfinite parameter derivatives.")
        return output


def _coding(result, predictors):
    record = result.provenance.get("categorical_encoding", {})
    if not isinstance(record, Mapping):
        _error("invalid_result", "Saved categorical encoding is invalid.")
    features, categories = {}, {}
    if result.spec.intercept:
        features["Intercept"] = ("constant", None)
    for name in predictors:
        if name not in result.spec.categorical:
            if name in features:
                _error("unsupported_prediction_design", "Saved numeric/category terms have colliding names.")
            features[name] = ("numeric", name)
            continue
        coding = record.get(name)
        if (not isinstance(coding, Mapping) or not isinstance(coding.get("levels"), list)
                or len(coding["levels"]) < 2 or coding.get("reference") != coding["levels"][0]):
            _error("invalid_result", f"The fitted levels/reference for '{name}' are unavailable.")
        levels = list(coding["levels"])
        try:
            pd.Categorical([], categories=levels)
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_result", f"The fitted levels of '{name}' are invalid.") from exc
        categories[name] = levels
        for level in levels[1:]:
            term = f"{name}[{level}]"
            if term in features:
                _error("unsupported_prediction_design", "Saved category terms have colliding names.")
            features[term] = ("category", (name, level))
    return features, categories


def _panel_model(result):
    value = result.spec.options.get("model", "fe" if result.spec.estimator in {"xtreg", "xtivreg"} else "re")
    recorded = result.extra.get("model", result.provenance.get("model", value))
    if recorded != value:
        _error("invalid_result", "The saved panel model and fitted model disagree.")
    if value not in {"fe", "re", "be", "fd", "pooled", "mle", "pa"}:
        _error("invalid_result", "The saved panel model is invalid.")
    return value


def configure(result, state, outcome=None):
    """Return a validated config for the common saved prediction engine.

    Fixed effects are deliberately not set to zero under a response-mean
    label.  ``xb`` for an absorbed fit is explicitly the unabsorbed portion.
    Selected ``sqreg`` equations retain the full stored parameter covariance.
    """
    spec, estimator = result.spec, result.spec.estimator
    if estimator not in ESTIMATORS:
        return None
    try:
        # ResultBundle.model_copy and assignment can bypass pydantic field
        # validation. Recheck the complete estimator contract, including its
        # mandatory absorb/instrument/group/panel roles and option ownership.
        ModelSpec.model_validate(spec.model_dump())
    except ValidationError as exc:
        raise AnalysisError("invalid_result", "The saved scalar estimator specification is invalid: "
                            "required model roles/options must match the fitted estimator.") from exc
    if result.provenance.get("estimator") != estimator:
        _error("invalid_result", "The saved fitted estimator identity does not match its specification.")
    if spec.options.get("formula"):
        _error("unsupported_prediction_design", "A formula fit requires its fitted formula adapter.")
    predictors = list(spec.predictors)
    if estimator in _IV:
        endogenous = spec.columns.get("endogenous", [])
        predictors += [endogenous] if isinstance(endogenous, str) else list(endogenous)
    predictors = list(dict.fromkeys(predictors))
    features, categories = _coding(result, predictors)
    omitted = result.provenance.get("omitted_terms", [])
    if not isinstance(omitted, list) or any(not isinstance(term, str) for term in omitted):
        _error("invalid_result", "Saved omitted terms are invalid.")
    family, link_name = "gaussian", "identity"
    kinds = frozenset({"xb", "stdp", "response", "derivative"})
    definition = "linear conditional response mean"
    xb_definition = "saved linear predictor X beta"
    auxiliary = set()
    panel_model = _panel_model(result) if estimator in {"xtreg", "xtivreg", *_PANEL_LIKELIHOODS} else None
    if panel_model == "fd":
        _error("unsupported_prediction_design", "First-difference predictions require consecutive within-panel "
               "differences; raw level covariates cannot be used as the fitted design.")
    if estimator in _ABSORBED or panel_model in {"fe", "be"}:
        kinds = frozenset({"xb", "stdp", "derivative"})
        xb_definition = ("saved between-equation index evaluated at supplied panel-mean covariates"
                         if panel_model == "be" else "unabsorbed linear predictor X beta, excluding fitted group effects")
        definition = "slope effect of the unabsorbed linear index, holding group effects fixed"
    if estimator in {"xtreg", "xtivreg"} and panel_model in {"re", "mle"}:
        definition = "population response mean X beta, integrating mean-zero Gaussian panel effects"
    if estimator == "xtreg" and panel_model == "mle":
        auxiliary = {"/sigma_u", "/sigma_e"}
        for term in auxiliary:
            if term not in state.terms or float(state.beta[state.terms.index(term)]) <= 0:
                _error("invalid_result", "Saved Gaussian panel standard deviations must be positive.")
    if estimator == "mixed":
        auxiliary = {coefficient.term for coefficient in result.coefficients if coefficient.equation != spec.outcome}
        if (not auxiliary or any(not (term.startswith("/var(") or term.startswith("/cov(")) for term in auxiliary)):
            _error("invalid_result", "Saved Gaussian mixed variance parameters are invalid.")
        for term in auxiliary:
            if term.startswith("/var(") and float(state.beta[state.terms.index(term)]) < 0:
                _error("invalid_result", "Saved Gaussian mixed variances must be nonnegative.")
        definition = "population response mean X beta, integrating mean-zero Gaussian random effects"
    if estimator in {"qreg", "bsqreg"}:
        quantile = spec.options.get("quantile", .5)
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)) or not 0 < quantile < 1:
            _error("invalid_result", "Saved quantile is invalid.")
        definition = f"conditional quantile Q({quantile:g}|X), not the conditional mean"
    if estimator == "iqreg":
        quantiles = result.extra.get("quantiles", spec.options.get("quantiles"))
        if (not isinstance(quantiles, list) or len(quantiles) != 2
                or any(isinstance(q, bool) or not isinstance(q, (int, float)) for q in quantiles)
                or not 0 < quantiles[0] < quantiles[1] < 1):
            _error("invalid_result", "Saved interquantile pair is invalid.")
        definition = f"conditional interquantile difference Q({quantiles[1]:g}|X)-Q({quantiles[0]:g}|X)"
    if estimator == "xtfmb":
        definition = "linear predictor using the mean of the fitted period coefficient vectors"
    if estimator == "prais":
        definition = "structural response X_t beta; no lagged residual or conditional AR(1) forecast"
    if estimator == "rreg":
        definition = "robust fitted linear location X beta"
    if estimator == "ppmlhdfe":
        family, link_name = "poisson", "log"
        # Even slope effects on the response scale require the unrecorded FE
        # sum inside exp(X beta + FE). The xb slope remains identified.
    if estimator in _PANEL_LIKELIHOODS and panel_model != "pa":
        _error("unsupported_prediction_design", "Panel FE/RE likelihood responses require their conditional "
               "or integrated random-effect adapter; the population GEE adapter applies only to model='pa'.")
    if estimator == "xtgee" or estimator in _PANEL_LIKELIHOODS:
        family_record = result.extra.get("family")
        family = {"igaussian": "inverse_gaussian"}.get(family_record, family_record)
        link_name = result.extra.get("link")
        if family not in FAMILY_LINKS or link_name not in FAMILY_LINKS[family]:
            _error("invalid_result", "The saved GEE family/link pair is invalid.")
        if estimator == "xtgee":
            requested_family = {"igaussian": "inverse_gaussian"}.get(spec.options.get("family", "gaussian"), spec.options.get("family", "gaussian"))
            if requested_family != family or spec.options.get("link", link_name) not in {None, link_name}:
                _error("invalid_result", "The saved GEE fitted family/link disagree with the specification.")
        else:
            required_family, required_link = {"xtlogit": ("binomial", "logit"), "xtprobit": ("binomial", "probit"), "xtpoisson": ("poisson", "log")}[estimator]
            if (family, link_name) != (required_family, required_link):
                _error("invalid_result", "The saved panel GEE family/link disagree with the command.")
        definition = "GEE population-averaged response mean, with the saved link and offset/exposure"
    try:
        link = make_link(link_name, power=spec.options.get("power"), k=spec.options.get("nbk", 1.))
    except (KernelError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_result", "The saved scalar mean link is invalid.") from exc
    if estimator != "xtgee" and estimator not in _PANEL_LIKELIHOODS:
        if result.extra.get("link", link_name) != link_name or result.extra.get("family", family) != family:
            _error("invalid_result", "The saved fitted mean family/link disagree with the scalar estimator.")
    equation = None
    if estimator == "sqreg":
        equations = result.extra.get("equations")
        quantiles = result.extra.get("quantiles")
        if (not isinstance(equations, list) or not equations or len(set(equations)) != len(equations)
                or any(not isinstance(label, str) or not label for label in equations)
                or not isinstance(quantiles, list) or len(quantiles) != len(equations)
                or any(isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 < q < 1 for q in quantiles)):
            _error("invalid_result", "Saved simultaneous quantile equations are invalid.")
        if outcome is None:
            if len(equations) != 1:
                _error("prediction_outcome_required", "Select one saved simultaneous quantile equation with outcome='q25', for example.")
            outcome = equations[0]
        if not isinstance(outcome, str) or outcome not in equations:
            _error("unknown_prediction_outcome", "Select one of the saved simultaneous quantile equations: " + ", ".join(equations) + ".")
        equation = outcome
        original_features = features
        features = {f"{equation}:{name}": value for name, value in original_features.items()}
        all_features = {f"{label}:{name}" for label in equations for name in original_features}
        auxiliary = set(state.terms) - set(features)
        if set(state.terms) - all_features or any(coefficient.equation not in equations or not coefficient.term.startswith(coefficient.equation + ":") for coefficient in result.coefficients):
            _error("unsupported_prediction_design", "Saved simultaneous quantile terms do not match their equations.")
        omitted = [f"{equation}:{term}" for term in omitted]
        definition = f"conditional quantile Q({quantiles[equations.index(equation)]:g}|X) from saved equation {equation}"
    elif outcome is not None:
        _error("unsupported_prediction_outcome", "A scalar fitted equation does not accept outcome selection.")
    mean_terms = set(state.terms) - auxiliary
    for coefficient in result.coefficients:
        if coefficient.term in auxiliary:
            continue
        if coefficient.term not in features or coefficient.equation not in {None, spec.outcome, equation}:
            _error("unsupported_prediction_design", f"Saved term '{coefficient.term}' is not a supported scalar response feature.")
    if set(features) - mean_terms - set(omitted):
        _error("invalid_result", "Saved scalar coefficients do not account for all fitted features and omissions.")
    if not mean_terms:
        _error("invalid_result", "The saved result has no scalar response coefficients.")
    offset, exposure, trials = (_role(spec, role) for role in ("offset", "exposure", "trials"))
    if offset and exposure or trials:
        _error("invalid_result", "The saved scalar offset/exposure/trials combination is invalid.")
    if exposure and link_name != "log":
        _error("invalid_result", "The saved exposure requires a log link.")
    continuous = {}
    for variable in predictors:
        if variable not in categories and variable not in {offset, exposure}:
            term = f"{equation}:{variable}" if equation else variable
            continuous[variable] = term if term in mean_terms else None
    adapter = ScalarResponse(link, family, state.terms, continuous, kinds, definition, xb_definition,
                             equation=equation, accepts_outcome=estimator == "sqreg")
    return {"link": link, "family": family, "predictors": predictors, "categories": categories,
            "features": features, "fixed": {}, "mean_terms": mean_terms, "offset": offset,
            "exposure": exposure, "trials": None, "adapter": adapter}
