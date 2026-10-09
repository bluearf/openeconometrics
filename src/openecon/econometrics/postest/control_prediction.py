"""Common conditional CF targets from one semantically validated joint state.

The encoded design holds the original Z/X columns, never a frozen control
residual. Every parameter evaluation reconstructs d-Z gamma, including when
the global mean design is used for MEM. No sample-sized training state survives
in the adapter; validation runs once before any query or Dataset replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.control_function.kernels import _family_link
from openecon.econometrics.control_function.state import validate_state

ESTIMATORS = frozenset(
    {
        "cfregress",
        "cflogit",
        "cfprobit",
        "cfcloglog",
        "cfpoisson",
        "cfgamma",
        "cfinvgauss",
        "cffraclogit",
    }
)


@dataclass
class ConditionalControl:
    kz: int
    x_terms: tuple[str, ...]
    z_terms: tuple[str, ...]
    endogenous: str
    predictors: tuple[str, ...]
    instruments: tuple[str, ...]
    family: object
    link: object

    kinds = frozenset({"xb", "stdp", "response"})
    accepts_outcome = False

    def required(self):
        return [self.endogenous, *self.instruments]

    def intervention_variables(self):
        return [self.endogenous, *self.instruments]

    def workspace_row_bytes(self, **_):
        # Extra raw design, full parameter Jacobians and cross-derivative buffers.
        return 128 * (self.kz + len(self.x_terms) + 1)

    def encode(self, frame, design):
        return design

    def _parts(self, design, parameters):
        z, x = design.x[:, : self.kz], design.x[:, self.kz : -1]
        gamma, beta = parameters[: self.kz], parameters[self.kz :]
        residual = x[:, self.x_terms.index(self.endogenous)] - z @ gamma
        q = torch.cat((x, residual[:, None]), dim=1)
        eta = q @ beta
        if not bool(torch.isfinite(eta).all()) or not bool(torch.isfinite(residual).all()):
            raise AnalysisError(
                "prediction_domain",
                "The saved conditional control/index exceeds finite float64 support.",
            )
        return z, q, eta

    def _link(self, eta):
        mu = self.link.inverse(eta)
        first = self.link.derivative(eta, mu)
        if (
            not bool(self.family.valid_mu(mu).all())
            or not bool(torch.isfinite(first).all())
            or not bool((first > 0).all())
            or (self.family.binomial and not bool((self.link.complement(eta, mu) > 0).all()))
        ):
            raise AnalysisError(
                "prediction_domain",
                "The CF conditional mean/derivative reaches an unresolved link boundary.",
            )
        return mu, first

    def _slope(self, parameters, variable):
        if variable not in {*self.predictors, self.endogenous, *self.instruments}:
            raise AnalysisError(
                "invalid_prediction_term",
                "Choose a declared exogenous, endogenous or excluded-instrument covariate.",
            )
        gamma, beta = parameters[: self.kz], parameters[self.kz :]
        slope = parameters.sum() * 0
        if variable in self.x_terms:
            slope = slope + beta[self.x_terms.index(variable)]
        residual_slope = float(variable == self.endogenous)
        if variable in self.z_terms:
            residual_slope = residual_slope - gamma[self.z_terms.index(variable)]
        return slope + beta[-1] * residual_slope

    def values(self, design, parameters, kind):
        _, _, eta = self._parts(design, parameters)
        return eta if kind in {"xb", "stdp"} else self._link(eta)[0]

    def effects(self, design, parameters, variable, kind):
        _, _, eta = self._parts(design, parameters)
        slope = self._slope(parameters, variable)
        return torch.ones_like(eta) * slope if kind == "xb" else self._link(eta)[1] * slope

    def jacobian(self, design, parameters, kind, variable=None):
        z, q, eta = self._parts(design, parameters)
        jac = torch.cat((-parameters[-1] * z, q), dim=1)
        if kind in {"xb", "stdp"}:
            return jac
        mu, first = self._link(eta)
        if kind != "derivative":
            return first[:, None] * jac
        slope = self._slope(parameters, variable)
        slope_jac = torch.zeros_like(parameters)
        if variable in self.x_terms:
            slope_jac[self.kz + self.x_terms.index(variable)] = 1
        residual_slope = float(variable == self.endogenous)
        if variable in self.z_terms:
            j = self.z_terms.index(variable)
            slope_jac[j] = -parameters[-1]
            residual_slope -= parameters[j]
        slope_jac[-1] = residual_slope
        second = first * self.link.second_ratio(eta, mu)
        return second[:, None] * slope * jac + first[:, None] * slope_jac

    def definition(self, kind):
        target = (
            "conditional response-mean covariate derivative"
            if kind == "derivative"
            else "conditional link index"
            if kind in {"xb", "stdp"}
            else "conditional response mean"
        )
        return (
            target
            + "; observed endogenous and instruments; full Gamma/Beta covariance; structural intervention effects unidentified"
        )


def configure(result, state, outcome=None):
    if outcome is not None:
        raise AnalysisError(
            "unsupported_prediction_outcome",
            "CF targets are conditional means; equation/structural outcome selection is unsupported.",
        )
    saved = validate_state(result)
    if tuple(item["term"] for item in saved["parameter_order"]) != state.terms:
        raise AnalysisError(
            "invalid_result", "The complete CF parameter order disagrees with common inference."
        )
    spec = result.spec
    z_terms, x_terms = tuple(saved["z_terms"]), tuple(saved["x_terms"])
    family_type, link_type = _family_link(saved["kind"])
    adapter = ConditionalControl(
        len(z_terms),
        x_terms,
        z_terms,
        spec.columns["endogenous"],
        tuple(spec.predictors),
        tuple(spec.columns["instruments"]),
        family_type(),
        link_type(),
    )
    features = {}
    for prefix, terms in (("first_stage", z_terms), ("outcome", x_terms)):
        for term in terms:
            features[f"{prefix}:{term}"] = (
                ("constant", None) if term == "Intercept" else ("numeric", term)
            )
    return dict(
        link=adapter.link,
        family="control_function",
        predictors=list(spec.predictors),
        categories={},
        features=features,
        fixed={},
        mean_terms=set(features),
        offset=None,
        exposure=None,
        trials=None,
        adapter=adapter,
    )
