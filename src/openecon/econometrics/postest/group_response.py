"""Saved normal-population, explicit-effect and fitted fixed-effect responses."""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.analysis import _numeric
from openecon.econometrics.glm.families import make_link
from openecon.econometrics.postest.group_state import keys, lookup
from openecon.econometrics.postest.linear_prediction import _coding
from openecon.engines.distributions import gauss_hermite
from openecon.resources import plan_workspace

ESTIMATORS = frozenset({"melogit", "meprobit", "mepoisson", "menbreg"})


def _error(code, message):
    raise AnalysisError(code, message)


class NormalResponse:
    kinds = frozenset({"xb", "stdp", "response", "derivative"})
    accepts_outcome = False

    def __init__(self, result, state, names, family, continuous):
        self.result, self.state, self.names, self.family = result, state, names, family
        self.continuous = continuous
        self.target, self.explicit = "population", None
        group = result.spec.columns.get("group", result.spec.panel)
        self.log_variance = "/lnsig2u" in state.terms
        self.variances = (
            [state.terms.index("/lnsig2u")]
            if self.log_variance
            else [state.terms.index(f"/var({name}[{group}])") for name in names]
        )
        self.covariances = []
        for i in range(len(names)):
            for j in range(i):
                term = f"/cov({names[i]},{names[j]}[{group}])"
                if term in state.terms:
                    self.covariances.append((i, j, state.terms.index(term)))

    def required(self):
        if getattr(self, "evaluation_kind", None) in {"xb", "stdp"}:
            return []
        columns = [name for name in self.names if name != "_cons"]
        if self.explicit:
            columns += [value for value in self.explicit.values() if isinstance(value, str)]
        return list(dict.fromkeys(columns))

    def workspace_row_bytes(self, **_):
        return 32 * 256 * (len(self.state.terms) + 2)

    def set_target(self, target, explicit):
        if target not in {"population", "conditional"}:
            _error(
                "unsupported_prediction_target",
                "Use population, explicit conditional effects, or complete-group posterior evaluation.",
            )
        if target == "conditional":
            if not isinstance(explicit, dict) or set(explicit) != set(self.names):
                _error(
                    "random_effects_required",
                    "Conditional predictions require one explicit value or evaluation column for every saved random effect.",
                )
            for value in explicit.values():
                if not isinstance(value, str) and (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                ):
                    _error(
                        "invalid_random_effects",
                        "Explicit effects must be finite numbers or numeric column names.",
                    )
            self.explicit = explicit
        elif explicit is not None:
            _error(
                "invalid_random_effects", "Explicit effects belong only to the conditional target."
            )
        self.target = target

    def encode(self, frame, design):
        if getattr(self, "evaluation_kind", None) in {"xb", "stdp"}:
            return design
        plan_workspace(
            "saved normal response integration",
            {"row_derivatives": len(frame) * self.workspace_row_bytes()},
        )
        design.random_z = torch.stack(
            [
                torch.ones(len(frame), dtype=torch.float64)
                if name == "_cons"
                else _numeric(frame[name], name)
                for name in self.names
            ],
            1,
        )
        if self.explicit:
            design.explicit_u = torch.stack(
                [
                    _numeric(frame[value], value)
                    if isinstance(value, str)
                    else torch.full((len(frame),), float(value), dtype=torch.float64)
                    for value in [self.explicit[name] for name in self.names]
                ],
                1,
            )
        return design

    def matrix(self, beta):
        q = len(self.names)
        variance = beta[self.variances]
        matrix = torch.diag(variance.exp() if self.log_variance else variance)
        for i, j, index in self.covariances:
            unit = torch.zeros((q, q), dtype=torch.float64)
            unit[i, j] = unit[j, i] = 1.0
            matrix = matrix + beta[index] * unit
        if int(torch.linalg.cholesky_ex(matrix)[1]):
            _error(
                "invalid_random_covariance",
                "Saved random-effect covariance must be positive definite.",
            )
        return matrix

    def definition(self, kind):
        if kind in {"xb", "stdp"}:
            return "saved fixed linear predictor, excluding random effects"
        return (
            "normal-population mean integrated over all saved variance/covariance parameters"
            if self.target == "population"
            else "conditional mean at explicitly supplied random effects; effects held fixed"
        )

    def _parts(self, design, beta, variable=None):
        eta = design.x @ beta + design.deterministic
        slope = beta.sum() * 0
        if variable is not None:
            if variable not in self.continuous:
                _error(
                    "invalid_prediction_term",
                    "Choose an original continuous predictor or random slope.",
                )
            term = self.continuous[variable]
            slope = beta.sum() * 0 if term is None else beta[self.state.terms.index(term)]
        z = design.random_z
        if self.target == "conditional":
            eta = eta + (z * design.explicit_u).sum(1)
            if variable in self.names:
                slope = slope + design.explicit_u[:, self.names.index(variable)]
            return eta, None, slope, None
        matrix = self.matrix(beta)
        variance = ((z @ matrix) * z).sum(1)
        derivative = torch.zeros_like(variance)
        if variable in self.names:
            derivative = 2 * (z @ matrix)[:, self.names.index(variable)]
        return eta, variance, slope, derivative

    def _evaluate(self, design, beta, variable=None):
        eta, variance, slope, derivative = self._parts(design, beta, variable)
        if self.target == "conditional":
            if self.family == "logit":
                mean = torch.sigmoid(eta)
                return mean if variable is None else mean * torch.sigmoid(-eta) * slope
            if self.family == "probit":
                return (
                    torch.special.ndtr(eta)
                    if variable is None
                    else torch.exp(-eta.square() / 2) / math.sqrt(2 * math.pi) * slope
                )
            mean = eta.exp()
            return mean if variable is None else mean * slope
        if self.family == "probit":
            root = (1 + variance).sqrt()
            normalized = eta / root
            return (
                torch.special.ndtr(normalized)
                if variable is None
                else torch.exp(-normalized.square() / 2)
                / math.sqrt(2 * math.pi)
                * (slope / root - eta * derivative / (2 * root**3))
            )
        if self.family == "log":
            mean = (eta + variance / 2).exp()
            return mean if variable is None else mean * (slope + derivative / 2)
        # The scalar Gaussian index is exactly N(eta, z'Gz), even with
        # correlated random slopes. Never integrate a separate row logit.
        root = variance.sqrt()
        previous = None
        for order in (64, 128, 256):
            nodes, weights = gauss_hermite(order)
            nodes, weights = math.sqrt(2) * nodes, weights / math.sqrt(math.pi)
            index = eta[:, None] + root[:, None] * nodes
            probability = torch.sigmoid(index)
            value = (
                (probability * weights).sum(1)
                if variable is None
                else (
                    probability
                    * torch.sigmoid(-index)
                    * (
                        slope
                        if isinstance(slope, float)
                        else slope[:, None]
                        if slope.ndim
                        else slope
                    )
                    + probability
                    * torch.sigmoid(-index)
                    * derivative[:, None]
                    / (2 * root[:, None])
                    * nodes
                )
                .mul(weights)
                .sum(1)
            )
            if previous is not None and (
                not len(value) or float((value - previous).detach().abs().max()) < 1e-8
            ):
                return value
            previous = value
        _error(
            "integration_not_converged",
            "Population logistic integration did not settle at 256 nodes; this index/variance domain needs a larger validated rule.",
        )

    def values(self, design, beta, kind):
        if kind in {"xb", "stdp"}:
            return design.x @ beta + design.deterministic
        value = self._evaluate(design, beta)
        if not bool(torch.isfinite(value).all()):
            _error("non_finite_prediction", "Integrated response is nonfinite.")
        return value

    def effects(self, design, beta, variable, kind):
        if kind == "xb":
            term = self.continuous.get(variable)
            return torch.ones(len(design.x), dtype=torch.float64) * (
                0 if term is None else beta[self.state.terms.index(term)]
            )
        return self._evaluate(design, beta, variable)

    def jacobian(self, design, beta, kind, variable=None):
        if variable is None and kind in {"xb", "stdp"}:
            return design.x
        with torch.enable_grad():
            return torch.autograd.functional.jacobian(
                lambda point: (
                    self.values(design, point, kind)
                    if variable is None
                    else self.effects(
                        design, point, variable, "response" if kind == "derivative" else kind
                    )
                ),
                beta,
            )


def configure(result, state, outcome=None):
    if outcome is not None:
        _error(
            "unsupported_prediction_outcome",
            "Normal group responses do not select an outcome equation.",
        )
    spec = result.spec
    if spec.estimator in {"xtlogit", "xtprobit", "xtpoisson"}:
        if spec.options.get("model", "re") != "re" or result.extra.get("model") != "re":
            _error("invalid_result", "Saved panel likelihood model and fitted RE model disagree.")
    family = {
        "melogit": "logit",
        "meprobit": "probit",
        "mepoisson": "log",
        "menbreg": "log",
        "xtlogit": "logit",
        "xtprobit": "probit",
        "xtpoisson": "log",
    }[spec.estimator]
    saved = result.extra.get("random_effects", {})
    names = (saved if isinstance(saved, list) else saved.get("effects")) or result.extra.get(
        "random_terms"
    )
    if "/lnsig2u" in state.terms:
        names = ["_cons"]
    if not isinstance(names, list) or not 1 <= len(names) <= 2:
        _error("invalid_result", "Saved random-effect terms are absent or invalid.")
    features, categories = _coding(result, spec.predictors)
    mean_terms = {term for term in state.terms if term in features}
    auxiliary = set(state.terms) - mean_terms
    if any(
        not (
            term.startswith("/var(") or term.startswith("/cov(") or term in {"/lnalpha", "/lnsig2u"}
        )
        for term in auxiliary
    ):
        _error(
            "invalid_result",
            "Saved normal group parameters have unsupported reporting coordinates.",
        )
    omitted = result.provenance.get("omitted_terms", [])
    if set(features) - mean_terms - set(omitted):
        _error("invalid_result", "Saved group mean design is incomplete.")
    predictors = list(
        dict.fromkeys([*spec.predictors, *[name for name in names if name != "_cons"]])
    )
    continuous = {
        name: name if name in mean_terms else None for name in predictors if name not in categories
    }
    adapter = NormalResponse(result, state, names, family, continuous)
    adapter.matrix(state.beta)
    return {
        "link": make_link(
            "logit" if family == "logit" else "probit" if family == "probit" else "log"
        ),
        "family": "binomial" if family in {"logit", "probit"} else "poisson",
        "predictors": predictors,
        "categories": categories,
        "features": features,
        "fixed": {},
        "mean_terms": mean_terms,
        "offset": spec.columns.get("offset"),
        "exposure": spec.columns.get("exposure"),
        "trials": None,
        "adapter": adapter,
    }


class FixedResponse:
    kinds = frozenset({"xb", "stdp", "response", "derivative"})
    accepts_outcome = False

    def __init__(self, scalar, state, record, result):
        self.scalar, self.state, self.record, self.result = scalar, state, record, result
        self.k = len(result.coefficients)
        self.initial = state.beta[: self.k].clone()
        self.joint = record.get("full_inference")
        self.point_only = self.joint is None

    def required(self):
        if getattr(self, "evaluation_kind", None) in {"xb", "stdp"}:
            return []
        return self.record["group_columns"]

    def workspace_row_bytes(self, **_):
        return 32 * len(self.state.terms)

    def definition(self, kind):
        if kind in {"xb", "stdp"}:
            return self.scalar.definition(kind)
        return (
            "conditional response using saved fitted fixed-effect sum; full nuisance information"
            if self.joint
            else "conditional point response using saved fitted fixed-effect sum; full nuisance uncertainty unavailable"
        )

    def encode(self, frame, design):
        if getattr(self, "evaluation_kind", None) in {"xb", "stdp"}:
            return design
        records = []
        with lookup(self.record) as get:
            for key in keys(frame, self.required()):
                item = get(key)
                if item is None:
                    _error(
                        "unknown_group",
                        "An evaluation group combination is absent from saved fitted effects; zero effects cannot be assumed.",
                    )
                records.append(item)
        design.saved_effect = torch.tensor(
            [item["effect"] for item in records], dtype=torch.float64
        )
        design.projection = torch.zeros((len(frame), self.k), dtype=torch.float64)
        design.nuisance = torch.zeros(
            (len(frame), len(self.state.terms) - self.k), dtype=torch.float64
        )
        if self.joint:
            for i, item in enumerate(records):
                for term, value in zip(self.joint["mean_terms"], item["projection"], strict=True):
                    design.projection[i, self.state.terms.index(term)] = value
                design.nuisance[i, item["nuisance"]] = 1.0
        return design

    def _eta(self, design, beta):
        delta = beta[: self.k] - self.initial
        return (
            design.x[:, : self.k] @ beta[: self.k]
            + design.deterministic
            + design.saved_effect
            - design.projection @ delta
            + design.nuisance @ beta[self.k :]
        )

    def values(self, design, beta, kind):
        if kind in {"xb", "stdp"}:
            return design.x @ beta + design.deterministic
        return self.scalar.link.inverse(self._eta(design, beta)) * design.scale

    def effects(self, design, beta, variable, kind):
        slope = self.scalar._slope(beta[: self.k], variable)
        if kind == "xb":
            return torch.ones(len(design.x), dtype=torch.float64) * slope
        eta = self._eta(design, beta)
        return (
            design.scale * self.scalar.link.derivative(eta, self.scalar.link.inverse(eta)) * slope
        )

    def jacobian(self, design, beta, kind, variable=None):
        if variable is None and kind in {"xb", "stdp"}:
            return design.x
        if self.point_only:
            _error(
                "missing_group_inference",
                "This fitted state supports point means; full nuisance covariance is absent. Use xb inference or fit a nonrobust model within the recorded nuisance geometry domain.",
            )
        with torch.enable_grad():
            return torch.autograd.functional.jacobian(
                lambda point: (
                    self.values(design, point, kind)
                    if variable is None
                    else self.effects(
                        design, point, variable, "response" if kind == "derivative" else kind
                    )
                ),
                beta,
            )


def fixed_config(result, state, config):
    record = result.extra.get("group_state")
    if not record:
        return state, config
    if record.get("kind") != "fixed_effect_sum":
        return state, config
    joint = record.get("full_inference")
    if joint:
        width = joint["nuisance_width"]
        plan_workspace(
            "saved fixed-effect nuisance covariance snapshot",
            {"full_covariance": 96 * (len(state.terms) + width) ** 2},
        )
        vc = torch.tensor(joint["covariance"], dtype=torch.float64)
        if (
            vc.shape != (width, width)
            or not bool(torch.isfinite(vc).all())
            or not torch.allclose(vc, vc.T, atol=1e-10, rtol=1e-10)
            or float(torch.linalg.eigvalsh(vc).min()) < -1e-8 * max(1, float(vc.diagonal().max()))
        ):
            _error(
                "invalid_group_inference",
                "Saved nuisance covariance is not finite symmetric positive semidefinite.",
            )
        state = replace(
            state,
            beta=torch.cat((state.beta, torch.zeros(width, dtype=torch.float64))),
            covariance=torch.block_diag(state.covariance, vc),
            terms=(*state.terms, *[f"__saved_fe_{i}" for i in range(width)]),
        )
    config["adapter"] = FixedResponse(config["adapter"], state, record, result)
    return state, config
