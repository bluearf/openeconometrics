"""Explicit saved targets for formula, systems, selection and frontier models.

The design carries original covariates, so MEM evaluates nonlinear functions at
their raw means. Ancillary parameters remain in the saved full covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from pydantic import ValidationError

from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec
from .linear_prediction import _coding

ESTIMATORS = frozenset(
    {
        "nl",
        "sureg",
        "mvreg",
        "reg3",
        "threshold",
        "biprobit",
        "heckman",
        "heckprobit",
        "churdle",
        "ivprobit",
        "ivtobit",
        "frontier",
    }
)


def error(code, message):
    raise AnalysisError(code, message)


def phi(x):
    return torch.exp(-0.5 * x.square()) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * torch.special.erfc(-x / math.sqrt(2))


def mills(x):
    central = x.clamp(min=-12)
    ratio = torch.exp(
        -0.5 * central.square() - math.log(math.sqrt(2 * math.pi)) - torch.special.log_ndtr(central)
    )
    u = (-x).clamp(min=12)
    tail = torch.zeros_like(u)
    for j in range(60, 0, -1):
        tail = j / (u + tail)
    return torch.where(x < -12, u + tail, ratio)


class _Bivariate(torch.autograd.Function):
    """Native Genz values, closed-form probability derivatives including rho."""

    @staticmethod
    def forward(ctx, first, second, rho):
        from openecon.econometrics.discrete.bivariate import bvn_cdf

        ctx.save_for_backward(first, second, rho)
        return bvn_cdf(first, second, rho)

    @staticmethod
    def backward(ctx, gradient):
        first, second, rho = ctx.saved_tensors
        scale = torch.sqrt(1 - rho.square())
        a = phi(first) * Phi((second - rho * first) / scale)
        b = phi(second) * Phi((first - rho * second) / scale)
        c = torch.exp(
            -(first.square() - 2 * rho * first * second + second.square()) / (2 * scale.square())
        ) / (2 * math.pi * scale)
        return gradient * a, gradient * b, (gradient * c).sum_to_size(rho.shape)


class _LogBivariate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second, rho):
        from openecon.econometrics.discrete.bivariate import log_bvn_cdf

        value = log_bvn_cdf(first, second, rho)
        ctx.save_for_backward(first, second, rho, value)
        return value

    @staticmethod
    def backward(ctx, gradient):
        first, second, rho, value = ctx.saved_tensors
        variance = 1 - rho.square()
        scale = torch.sqrt(variance)
        a = torch.exp(
            -0.5 * first.square()
            - 0.5 * math.log(2 * math.pi)
            + torch.special.log_ndtr((second - rho * first) / scale)
            - value
        )
        b = torch.exp(
            -0.5 * second.square()
            - 0.5 * math.log(2 * math.pi)
            + torch.special.log_ndtr((first - rho * second) / scale)
            - value
        )
        c = torch.exp(
            -(first.square() - 2 * rho * first * second + second.square()) / (2 * variance)
            - math.log(2 * math.pi)
            - torch.log(scale)
            - value
        )
        return gradient * a, gradient * b, (gradient * c).sum_to_size(rho.shape)


@dataclass
class Response:
    estimator: str
    terms: tuple
    raw_features: dict
    equations: dict
    fixed: dict
    extra: dict
    target: str
    formula: object = None
    threshold_variable: str | None = None
    observed: str | None = None
    kinds: frozenset = frozenset({"xb", "stdp", "response", "derivative", "latent", "conditional"})
    accepts_outcome: bool = True

    @property
    def encoded_width(self):
        return len(self.raw_features)

    def workspace_row_bytes(self, *, kind):
        # Formula tape and differentiable value/effect graphs are batch bounded.
        tape = len(self.formula.tape) if self.formula is not None else 80
        return 64 * (tape + len(self.raw_features) + len(self.terms))

    def required(self):
        return [name for name in [self.threshold_variable, self.observed] if name is not None]

    def intervention_variables(self):
        return [] if self.threshold_variable is None else [self.threshold_variable]

    def encode(self, frame, design):
        from .prediction import _numeric
        from openecon.resources import plan_workspace

        plan_workspace(
            "saved explicit target buffers",
            {
                "raw_design_and_differentiable_tape": len(frame)
                * self.workspace_row_bytes(kind="response")
            },
        )

        values = []
        for feature, payload in self.raw_features.values():
            if feature == "numeric":
                values.append(_numeric(frame[payload], payload))
            else:
                name, level = payload
                values.append(torch.tensor((frame[name] == level).tolist(), dtype=torch.float64))
        x = (
            torch.stack(values, dim=1)
            if values
            else torch.zeros((len(frame), 0), dtype=torch.float64)
        )
        return replace(design, x=x)

    def column(self, design, name):
        if name == "Intercept":
            return torch.ones_like(design.deterministic)
        return design.x[:, list(self.raw_features).index(name)]

    def parameter(self, beta, name):
        if name not in self.terms:
            error(
                "prediction_state_missing",
                f"Saved {self.estimator} requires parameter '{name}' for this target.",
            )
        return beta[self.terms.index(name)]

    def index(self, design, beta, equation):
        answer = beta.sum() * 0 + torch.zeros_like(design.deterministic)
        for reported, feature in self.equations[equation].items():
            coefficient = (
                self.fixed[reported] if reported in self.fixed else self.parameter(beta, reported)
            )
            answer = answer + coefficient * self.column(design, feature)
        if not bool(torch.isfinite(answer).all()):
            error("prediction_domain", "Saved equation indexes exceed finite float64 range.")
        return answer

    def definition(self, kind):
        if kind in {"xb", "stdp", "latent"}:
            return f"saved {self.estimator} structural index; equation/target={self.target}"
        return f"saved {self.estimator} target={self.target}; kind={kind}; full parameter delta covariance"

    def values(self, design, beta, kind):
        if kind not in self.kinds:
            error("unsupported_prediction_kind", "This target is unavailable from the saved state.")
        value = self._values(design, beta, kind)
        if not bool(torch.isfinite(value).all()):
            error(
                "prediction_domain",
                "The explicit saved target is nonfinite at these evaluation inputs.",
            )
        return value

    def _values(self, design, beta, kind):
        estimator = self.estimator
        if estimator == "nl":
            from openecon.econometrics.quantile.formula import evaluate

            columns = {name: self.column(design, name) for name in self.formula.columns}
            parameters = torch.stack(
                [self.parameter(beta, name) for name in self.formula.parameters]
            )
            return evaluate.__wrapped__(
                self.formula, columns, parameters, len(design.x), jacobian=False
            )[0]
        if estimator in {"sureg", "mvreg", "reg3"}:
            return self.index(design, beta, self.target)
        if estimator == "threshold":
            q = self.column(design, self.threshold_variable)
            region = torch.zeros_like(q, dtype=torch.int64)
            for threshold in self.extra["thresholds"]:
                region = region + (q > threshold)
            value = self.index(design, beta, "common")
            for number in range(len(self.extra["thresholds"]) + 1):
                value = value + (region == number) * self.index(design, beta, f"region{number + 1}")
            return value
        a = self.index(design, beta, "outcome")
        if kind in {"xb", "stdp", "latent"}:
            return (
                self.index(design, beta, "select")
                if self.target in {"marginal2", "selection", "participation"}
                else a
            )
        if estimator in {"biprobit", "heckprobit"}:
            b = self.index(design, beta, "select")
            if self.target == "marginal1":
                return Phi(a)
            if self.target in {"marginal2", "selection"}:
                return Phi(b)
            rho = torch.tanh(self.parameter(beta, "/athrho"))
            if bool((rho.abs() >= 1).any()):
                error(
                    "prediction_domain", "The saved bivariate correlation is numerically singular."
                )
            if kind == "conditional" or self.target in {"conditional1", "conditional2"}:
                if bool(((a.abs() > 35) | (b.abs() > 35)).any()):
                    error(
                        "prediction_precision",
                        "Conditional bivariate probabilities beyond 35 standard deviations require a separately validated tail algorithm.",
                    )
                denominator = a if self.target == "conditional2" else b
                return torch.exp(
                    _LogBivariate.apply(a, b, rho) - torch.special.log_ndtr(denominator)
                )
            sign1, sign2 = (
                (-1 if self.target in {"joint00", "joint01"} else 1),
                (-1 if self.target in {"joint00", "joint10"} else 1),
            )
            return _Bivariate.apply(sign1 * a, sign2 * b, sign1 * sign2 * rho)
        if estimator == "heckman":
            b = self.index(design, beta, "select")
            if self.target == "selection":
                return Phi(b)
            if kind == "conditional" or self.target == "conditional":
                loading = (
                    self.parameter(beta, "mills:lambda")
                    if "mills:lambda" in self.terms
                    else torch.tanh(self.parameter(beta, "/athrho"))
                    * torch.exp(self.parameter(beta, "/lnsigma"))
                )
                return a + loading * mills(b)
            return a
        if estimator == "churdle":
            b = self.index(design, beta, "select")
            probability = Phi(b) if self.extra["select_link"] == "probit" else torch.sigmoid(b)
            if self.target == "participation":
                return probability
            sigma = torch.exp(self.parameter(beta, "/lnsigma"))
            lower = self.extra["ll"]
            if self.extra["model"] == "linear":
                from .limited_prediction import LimitedNormal

                normal = LimitedNormal("truncreg", self.terms.index("/lnsigma"), True, lower, None)
                conditional = normal.evaluate(a, beta, "conditional")[0]
            elif lower <= 0:
                conditional = torch.exp(a + 0.5 * sigma.square())
            else:
                shift = (a - math.log(lower)) / sigma
                conditional = torch.exp(
                    a
                    + 0.5 * sigma.square()
                    + torch.special.log_ndtr(shift + sigma)
                    - torch.special.log_ndtr(shift)
                )
            return (
                conditional
                if kind == "conditional" or self.target == "conditional"
                else lower * (1 - probability) + probability * conditional
            )
        if estimator == "ivprobit":
            return Phi(a)
        if estimator == "ivtobit":
            limits = self.extra["limits"]
            if set(limits) != {"lower", "upper"}:
                error(
                    "prediction_state_missing",
                    "Censored prediction requires the saved lower/upper limits.",
                )
            lower, upper = limits["lower"], limits["upper"]
            from .limited_prediction import LimitedNormal

            normal = LimitedNormal("tobit", self.terms.index("/lnsigma1"), True, lower, upper)
            return normal.evaluate(a, beta, kind)[0]
        if estimator == "frontier":
            if self.target == "frontier":
                return a
            distribution = self.extra["distribution"]
            if distribution == "tnormal":
                variance = torch.exp(self.parameter(beta, "/lnsigma2"))
                gamma = torch.sigmoid(self.parameter(beta, "/ilgtgamma"))
                u2, v2 = variance * gamma, variance * (1 - gamma)
                mu = self.parameter(beta, "/mu")
            else:
                u2 = torch.exp(self.parameter(beta, "/lnsig2u"))
                v2 = torch.exp(self.parameter(beta, "/lnsig2v"))
                mu = beta.sum() * 0
            sign = -1 if self.extra["cost"] else 1
            if self.target == "mean":
                inefficiency = (
                    torch.sqrt(u2)
                    if distribution == "exponential"
                    else mu + torch.sqrt(u2) * mills(mu / torch.sqrt(u2))
                )
                return a - sign * inefficiency
            residual = self.column(design, self.observed) - a
            if distribution == "exponential":
                center, spread = -sign * residual - v2 / torch.sqrt(u2), torch.sqrt(v2)
            else:
                center = (mu * v2 - sign * residual * u2) / (u2 + v2)
                spread = torch.sqrt(u2 * v2 / (u2 + v2))
            z = center / spread
            if self.target == "u":
                return center + spread * mills(z)
            te_sign = -sign
            return torch.exp(
                te_sign * center
                + 0.5 * spread.square()
                + torch.special.log_ndtr(z + te_sign * spread)
                - torch.special.log_ndtr(z)
            )
        error("unsupported_prediction", "No explicit saved response function is available.")

    def effects(self, design, beta, variable, kind):
        if variable not in self.raw_features or self.raw_features[variable][0] != "numeric":
            error("invalid_prediction_term", "Choose one continuous original predictor.")
        if self.estimator == "threshold" and variable == self.threshold_variable:
            q = self.column(design, variable)
            if any(bool((q == threshold).any()) for threshold in self.extra["thresholds"]):
                error(
                    "undefined_threshold_effect",
                    "The threshold-variable derivative is undefined at a fitted regime boundary.",
                )
        with torch.enable_grad():
            x = design.x.detach().clone().requires_grad_(True)
            value = self.values(replace(design, x=x), beta, kind)
            (gradient,) = torch.autograd.grad(value.sum(), x, create_graph=True, allow_unused=True)
            return (
                beta.sum() * 0 + torch.zeros(len(x), dtype=torch.float64)
                if gradient is None
                else gradient[:, list(self.raw_features).index(variable)]
            )

    def jacobian(self, design, beta, kind, variable=None):
        # Forward directional columns avoid an N-by-N reverse Jacobian. Each
        # direction is evaluated in the same bounded block; all K are retained.
        function = (
            (lambda b: self.values(design, b, kind))
            if variable is None
            else (
                lambda b: self.effects(
                    design, b, variable, "response" if kind == "derivative" else kind
                )
            )
        )
        columns = []
        with torch.enable_grad():
            for index in range(len(beta)):
                direction = torch.zeros_like(beta)
                direction[index] = 1
                _, column = torch.autograd.functional.jvp(function, beta, direction, strict=False)
                columns.append(column)
        matrix = torch.stack(columns, dim=1)
        if not bool(torch.isfinite(matrix).all()):
            error("prediction_domain", "Saved target derivatives are nonfinite at these inputs.")
        return matrix


def configure(result, state, outcome=None):
    try:
        return _configure(result, state, outcome)
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise AnalysisError(
            "invalid_result", "Saved target/equation metadata are missing or inconsistent."
        ) from exc


def _configure(result, state, outcome=None):
    spec, estimator = result.spec, result.spec.estimator
    try:
        ModelSpec.model_validate(spec.model_dump())
    except ValidationError as exc:
        raise AnalysisError(
            "invalid_result", "The saved advanced specification is invalid."
        ) from exc
    if result.provenance.get("estimator") != estimator:
        error("invalid_result", "Saved estimator identity disagrees with its specification.")
    extra = result.extra
    predictors = list(spec.predictors)
    equations, extra_required, formula = {}, [], None
    target = outcome
    ancillary = set()
    if estimator == "nl":
        from openecon.econometrics.quantile.formula import parse

        formula = parse(spec.options.get("formula"))
        if set(state.terms) != set(formula.parameters) or spec.categorical:
            error("invalid_result", "Saved formula parameters/coding do not match its safe tape.")
        predictors = list(formula.columns)
        target = target or "function"
        allowed = {"function"}
    elif estimator in {"sureg", "mvreg", "reg3"}:
        records = extra.get("equations")
        definitions = spec.options.get("equations")
        if estimator == "mvreg":
            definitions = [
                {"y": name, "name": name, "x": list(spec.predictors), "constant": spec.intercept}
                for name in spec.columns.get("outcomes", [])
            ]
        if (
            not isinstance(records, list)
            or not records
            or not isinstance(definitions, list)
            or len(records) != len(definitions)
        ):
            error("prediction_state_missing", "System prediction needs saved equation definitions.")
        allowed = {record["name"] for record in records}
        if len(allowed) != len(records):
            error("invalid_result", "Saved system equation names are not unique.")
        if target is None:
            if len(allowed) != 1:
                error(
                    "prediction_outcome_required",
                    "Select a saved system equation with outcome='equation_name'.",
                )
            target = records[0]["name"]
        predictors = list(
            dict.fromkeys(name for definition in definitions for name in definition["x"])
        )
    elif estimator == "biprobit":
        predictors += list(spec.columns.get("predictors2") or spec.predictors)
        target = target or "joint11"
        allowed = {
            "joint00",
            "joint01",
            "joint10",
            "joint11",
            "marginal1",
            "marginal2",
            "conditional1",
            "conditional2",
        }
        if spec.columns["outcome2"] in spec.predictors or spec.outcome in (
            spec.columns.get("predictors2") or spec.predictors
        ):
            error(
                "unsupported_prediction_design",
                "Recursive bivariate probabilities need an explicit endogenous binary intervention path; supplied outcome rows are not a joint probability target.",
            )
        ancillary = {"/athrho"}
    elif estimator in {"heckman", "heckprobit", "churdle"}:
        predictors += list(spec.columns["select_x"])
        target = target or ("joint11" if estimator == "heckprobit" else "mean")
        allowed = (
            {"joint11", "marginal1", "selection", "conditional1"}
            if estimator == "heckprobit"
            else {"mean", "conditional", "selection"}
            if estimator == "heckman"
            else {"mean", "conditional", "participation"}
        )
        ancillary = (
            {"/athrho"}
            if estimator == "heckprobit"
            else {"mills:lambda"}
            if estimator == "heckman" and extra.get("method") == "twostep"
            else {"/athrho", "/lnsigma"}
            if estimator == "heckman"
            else {"/lnsigma"}
        )
        if estimator == "churdle" and (
            extra.get("select_link") not in {"probit", "logit"}
            or extra.get("model") not in {"linear", "exponential"}
            or not isinstance(extra.get("ll"), (int, float))
        ):
            error(
                "prediction_state_missing",
                "Hurdle prediction needs its saved model, link and finite lower limit.",
            )
        if estimator == "churdle" and (
            not math.isfinite(extra["ll"])
            or type(extra["ll"]) is bool
            or extra["ll"] != spec.options.get("ll", 0.0)
            or extra["model"] != spec.options.get("model", "exponential")
            or extra["select_link"] != spec.options.get("select_link", "probit")
        ):
            error(
                "invalid_result",
                "Saved hurdle limit/model/link disagree with the fitted specification.",
            )
    elif estimator in {"ivprobit", "ivtobit"}:
        predictors += list(spec.columns["endogenous"])
        target = target or "structural"
        allowed = {"structural"}
        ancillary = {term for term in state.terms if term.startswith("/") or ":" in term}
        if extra.get("method") != spec.options.get("method", "ml"):
            error("prediction_state_missing", "IV prediction needs the fitted method identity.")
        if estimator == "ivtobit" and extra.get("method") == "ml":
            limits = extra.get("limits")
            if not isinstance(limits, dict) or set(limits) != {"lower", "upper"}:
                error(
                    "prediction_state_missing",
                    "IV censored targets need the resolved fitted limits.",
                )
            low, high = limits["lower"], limits["upper"]
            if any(
                value is not None and (type(value) not in {int, float} or not math.isfinite(value))
                for value in [low, high]
            ) or (low is not None and high is not None and low >= high):
                error(
                    "invalid_result",
                    "Saved censoring limits must be finite ordered numbers or None.",
                )
    elif estimator == "threshold":
        target = target or "regime_mean"
        allowed = {"regime_mean"}
        thresholds = extra.get("thresholds")
        if (
            not isinstance(thresholds, list)
            or not thresholds
            or any(type(x) not in {int, float} or not math.isfinite(x) for x in thresholds)
            or any(a >= b for a, b in zip(thresholds, thresholds[1:]))
            or not isinstance(extra.get("threshold_variable"), str)
        ):
            error(
                "prediction_state_missing",
                "Threshold prediction needs its saved ordered thresholds and threshold column.",
            )
        extra_required = [extra["threshold_variable"]]
    else:
        target = target or "frontier"
        allowed = {"frontier", "mean", "u", "te"}
        ancillary = {term for term in state.terms if term.startswith("/")}
        if (
            extra.get("distribution") not in {"hnormal", "tnormal", "exponential"}
            or type(extra.get("cost")) is not bool
        ):
            error(
                "prediction_state_missing",
                "Frontier prediction needs its saved distribution and cost/production convention.",
            )
        if target in {"u", "te"}:
            extra_required = [spec.outcome]
    if target not in allowed:
        error(
            "unknown_prediction_outcome",
            "Choose an explicit target: " + ", ".join(sorted(allowed)) + ".",
        )
    predictors = list(dict.fromkeys(predictors))
    features, categories = _coding(result, predictors)
    raw_features = {name: value for name, value in features.items() if value[0] != "constant"}
    for name in extra_required:
        raw_features.setdefault(name, ("numeric", name))

    def basis(names, prefix="", constant=None):
        use_constant = spec.intercept if constant is None else constant
        selected = {
            name: value
            for name, value in features.items()
            if (name == "Intercept" and use_constant)
            or (value[0] == "numeric" and value[1] in names)
            or (value[0] == "category" and value[1][0] in names)
        }
        if use_constant:
            selected["Intercept"] = ("constant", None)
        return {prefix + name: name for name in selected}

    if estimator in {"sureg", "mvreg", "reg3"}:
        for record, definition in zip(records, definitions, strict=True):
            name = record["name"]
            if definition.get("name", definition.get("y")) != name or not isinstance(
                record.get("terms"), list
            ):
                error(
                    "invalid_result", "Saved system equation reporting and specification disagree."
                )
            equations[name] = basis(definition["x"], name + ":", definition.get("constant", True))
    elif estimator == "biprobit":
        equations["outcome"] = basis(spec.predictors, spec.outcome + ":")
        equations["select"] = basis(
            spec.columns.get("predictors2") or spec.predictors, spec.columns["outcome2"] + ":"
        )
    elif estimator in {"heckman", "heckprobit", "churdle"}:
        equations["outcome"] = basis(spec.predictors)
        equations["select"] = basis(spec.columns["select_x"], "select:", True)
    elif estimator == "threshold":
        common, varying = extra.get("common"), extra.get("regions")
        if (
            not isinstance(common, list)
            or not isinstance(varying, list)
            or set(common) & set(varying)
            or set(common) | set(varying) != set(features)
        ):
            error("invalid_result", "Saved threshold feature partition is invalid.")
        equations["common"] = {name: name for name in common}
        for region in range(len(extra["thresholds"]) + 1):
            equations[f"region{region + 1}"] = {
                f"region{region + 1}:{name}": name for name in varying
            }
    elif estimator != "nl":
        equations["outcome"] = basis(predictors)
    fixed = extra.get("constrained_terms", {})
    if not isinstance(fixed, dict) or any(
        type(value) not in {float, int} or not math.isfinite(value) for value in fixed.values()
    ):
        error("invalid_result", "Saved fixed system constraints are invalid.")
    if estimator != "nl":
        expected = {term for equation in equations.values() for term in equation}
        omitted = result.provenance.get("omitted_terms", [])
        if (
            not isinstance(omitted, list)
            or set(state.terms) - expected - ancillary
            or set(fixed) - expected
        ):
            error(
                "invalid_result",
                "Saved equation coefficients do not match the explicit target design.",
            )
        missing = expected - set(state.terms) - set(fixed)
        if missing - set(omitted):
            error(
                "prediction_state_missing",
                "The saved equations omit required feature coefficients without omission metadata.",
            )
        equations = {
            name: {term: feature for term, feature in equation.items() if term not in missing}
            for name, equation in equations.items()
        }
    kinds = {"xb", "stdp", "response", "derivative", "latent", "conditional"}
    if target in {
        "marginal1",
        "marginal2",
        "selection",
        "participation",
        "joint00",
        "joint01",
        "joint10",
    }:
        kinds.discard("conditional")
    if estimator in {"nl", "sureg", "mvreg", "reg3", "threshold", "frontier"}:
        kinds -= {"conditional", "latent"}
    if estimator in {"ivprobit", "ivtobit"} and extra.get("method") != "ml":
        kinds = {"xb", "stdp", "latent"}
    elif estimator == "ivprobit":
        kinds.discard("conditional")
    adapter = Response(
        estimator,
        state.terms,
        raw_features,
        equations,
        fixed,
        extra,
        target,
        formula,
        extra.get("threshold_variable") if estimator == "threshold" else None,
        spec.outcome if estimator == "frontier" and target in {"u", "te"} else None,
        frozenset(kinds),
    )
    return {
        "link": None,
        "family": "explicit",
        "predictors": predictors,
        "categories": categories,
        "features": {},
        "fixed": {},
        "mean_terms": set(),
        "offset": None,
        "exposure": None,
        "trials": None,
        "adapter": adapter,
    }
