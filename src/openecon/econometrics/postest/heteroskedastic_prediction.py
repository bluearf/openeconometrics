"""Native saved heteroskedastic-probit predictions in reported coordinates.

The fitted latent error standard deviation is exp(z gamma), not exp(z gamma/2).
Internal fitting centres have already been mapped out of beta and covariance.
Only explicit saved equation roles and the full reported covariance are used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError, _MAX_DESIGN_BYTES
from openecon.econometrics.postest.inference import _numeric
from openecon.econometrics.postest.ordinal_prediction import (
    _exp_times,
    _same,
    category_standard_error,
)
from openecon.engines.inference import critical_value
from openecon.frame import as_frame


def _error(code, message):
    raise AnalysisError(code, message)


def variance_roles(result):
    """Return explicit original variance regressors, never infer from prefixes."""
    if result.spec.estimator != "hetprobit":
        return []
    if not isinstance(result.spec.columns, Mapping) or set(result.spec.columns) != {"het"}:
        _error(
            "invalid_result", "Saved hetprobit must contain only the supported het equation role."
        )
    roles = result.spec.columns["het"]
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(name, str) or not name for name in roles)
        or len(set(roles)) != len(roles)
        or result.spec.outcome in roles
    ):
        _error(
            "invalid_result", "Saved variance regressors must be distinct original column names."
        )
    if not isinstance(result.spec.options, Mapping) or result.spec.options:
        _error(
            "invalid_result",
            "Saved hetprobit options are not supported by the native fitted contract.",
        )
    if (
        not isinstance(result.spec.predictors, list)
        or any(not isinstance(name, str) or not name for name in result.spec.predictors)
        or len(set(result.spec.predictors)) != len(result.spec.predictors)
        or result.spec.outcome in result.spec.predictors
        or not isinstance(result.spec.intercept, bool)
        or not isinstance(result.spec.categorical, list)
        or any(not isinstance(name, str) or not name for name in result.spec.categorical)
        or len(set(result.spec.categorical)) != len(result.spec.categorical)
        or set(result.spec.categorical) - (set(result.spec.predictors) | set(roles))
    ):
        _error("invalid_result", "Saved mean-regressor/intercept specification is invalid.")
    return list(roles)


@dataclass(frozen=True)
class HeteroskedasticProbit:
    mean_indices: tuple[int, ...]
    variance_indices: tuple[int, ...]
    mean_slopes: dict[str, int]
    variance_slopes: dict[str, int]

    @staticmethod
    def index(design, beta, indices):
        value = design.x[:, list(indices)] @ beta[list(indices)]
        if not bool(torch.isfinite(value).all()):
            _error(
                "non_finite_prediction",
                "The requested equation index exceeds finite float64 range.",
            )
        return value

    def indexes(self, design, beta):
        return self.index(design, beta, self.mean_indices), self.index(
            design, beta, self.variance_indices
        )

    def standardized(self, design, beta):
        mean, log_scale = self.indexes(design, beta)
        index = _exp_times(-log_scale[:, None], mean[:, None])[:, 0]
        if not bool(torch.isfinite(index).all()):
            _error("prediction_precision", "The standardized probit index exceeds float64 range.")
        if bool(((mean != 0) & (index == 0)).any()):
            _error(
                "prediction_precision",
                "A nonzero standardized index underflows; its later derivative products cannot be certified.",
            )
        return mean, log_scale, index

    @staticmethod
    def log_density(index):
        # A finite standardized index ties log-scale to log(abs(mean)); for
        # nonzero mean its inverse-scale log cannot exceed about 752 here.
        # At |a|=1500, -a^2/2 is below -1,125,000: even all finite design,
        # coefficient and scale factors in these gradients cannot recover it.
        safe = index.clamp(-1500, 1500)
        return -safe.square() / 2 - math.log(2 * math.pi) / 2

    def values(self, design, beta, kind):
        if kind == "xb":
            return self.index(design, beta, self.mean_indices)
        if kind == "sigma":
            log_scale = self.index(design, beta, self.variance_indices)
            scale = log_scale.exp()
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
                _error(
                    "prediction_precision",
                    "Latent-error sigma must be positive and finite in float64.",
                )
            return scale
        _, _, index = self.standardized(design, beta)
        return 0.5 * torch.erfc(-index / math.sqrt(2))

    def slopes(self, beta, variable):
        zero = beta[0] * 0
        b = beta[self.mean_slopes[variable]] if variable in self.mean_slopes else zero
        g = beta[self.variance_slopes[variable]] if variable in self.variance_slopes else zero
        return b, g

    @staticmethod
    def product(left, right):
        value = left * right
        if not bool(torch.isfinite(value).all()) or bool(
            ((left != 0) & (right != 0) & (value == 0)).any()
        ):
            _error(
                "prediction_precision",
                "A factored effect product is outside representable float64 range.",
            )
        return value

    def effect_factor(self, design, beta, variable, mean):
        b, g = self.slopes(beta, variable)
        if variable in self.mean_slopes:
            position = self.mean_slopes[variable]
            other = [i for i in self.mean_indices if i != position]
            rest = design.x[:, other] @ beta[other]
            if not bool(torch.isfinite(rest).all()):
                _error(
                    "prediction_precision", "The factored remainder index exceeds float64 range."
                )
            contrast = 1 - self.product(design.x[:, position], g)
            factor = self.product(b, contrast) - self.product(rest, g)
        else:
            factor = b - self.product(mean, g)
        if not bool(torch.isfinite(factor).all()):
            _error("prediction_precision", "The factored response effect exceeds float64 range.")
        return factor

    def effects(self, design, beta, variable, kind):
        b, g = self.slopes(beta, variable)
        n = len(design.x)
        if kind == "xb":
            return torch.ones(n, dtype=torch.float64) * b
        if kind == "sigma":
            self.values(design, beta, "sigma")
            log_scale = self.index(design, beta, self.variance_indices)
            return _exp_times(log_scale[:, None], torch.ones((n, 1), dtype=torch.float64) * g)[:, 0]
        mean, log_scale, index = self.standardized(design, beta)
        density = self.log_density(index)
        factor = self.effect_factor(design, beta, variable, mean)
        return _exp_times((density - log_scale)[:, None], factor[:, None])[:, 0]

    def matrices(self, design):
        mean, variance = torch.zeros_like(design.x), torch.zeros_like(design.x)
        mean[:, list(self.mean_indices)] = design.x[:, list(self.mean_indices)]
        variance[:, list(self.variance_indices)] = design.x[:, list(self.variance_indices)]
        return mean, variance

    def jacobian(self, design, beta, kind, variable=None, parameter_scale=None):
        mean_x, variance_z = self.matrices(design)
        scale = (
            torch.ones((1, len(beta)), dtype=torch.float64)
            if parameter_scale is None
            else parameter_scale[None, :]
        )

        def multiply(log_weight, *factors):
            value = _exp_times(log_weight, *factors, scale)
            if parameter_scale is not None:
                nonzero = torch.ones_like(value, dtype=torch.bool)
                for factor in (*factors, scale):
                    nonzero = nonzero & (factor != 0)
                if bool((nonzero & (value == 0)).any()):
                    _error(
                        "prediction_precision",
                        "A nonzero covariance-scaled gradient product underflows; zero uncertainty cannot be certified.",
                    )
            return value

        if kind == "xb":
            return self.product(mean_x, scale)

        def unit(indices):
            result = torch.zeros_like(design.x)
            if variable in indices:
                result[:, indices[variable]] = 1
            return result

        if kind == "derivative_xb":
            return unit(self.mean_slopes) * scale
        if kind in {"sigma", "derivative_sigma"}:
            self.values(design, beta, "sigma")
            log_scale = self.index(design, beta, self.variance_indices)
            if kind == "sigma":
                return multiply(log_scale[:, None], variance_z)
            g = self.slopes(beta, variable)[1]
            factor = torch.ones((len(design.x), 1), dtype=torch.float64) * g
            return multiply(log_scale[:, None], unit(self.variance_slopes)) + multiply(
                log_scale[:, None], factor, variance_z
            )
        mean, log_scale, index = self.standardized(design, beta)
        density = self.log_density(index)
        if kind == "response":
            return multiply((density - log_scale)[:, None], mean_x) - multiply(
                density[:, None], index[:, None], variance_z
            )
        g = self.slopes(beta, variable)[1]
        h = self.effect_factor(design, beta, variable, mean)[:, None]
        dh = unit(self.mean_slopes) - self.product(g, mean_x)
        a = index[:, None]
        square_minus_one = index.clamp(-1500, 1500).square()[:, None] - 1
        # Factor h=b-m*g and dh/d(beta)=e_b-g*X before the density; exact
        # mean/variance overlap cancellations must remain zero even under huge
        # covariance. Unrepresentable factored products are explicitly refused.
        gradient = multiply((density - log_scale)[:, None], dh)
        gradient = gradient - multiply((density - 2 * log_scale)[:, None], a, h, mean_x)
        gradient = gradient + multiply(
            (density - log_scale)[:, None], square_minus_one, h, variance_z
        )
        gradient = gradient - multiply(
            (density - log_scale)[:, None], mean[:, None], unit(self.variance_slopes)
        )
        return gradient

    def difference(self, alternative, baseline, beta, kind):
        if kind == "xb":
            return self.values(alternative, beta, kind) - self.values(baseline, beta, kind)

        def logarithms(design):
            if kind == "sigma":
                self.values(design, beta, "sigma")
                return self.index(design, beta, self.variance_indices)
            index = self.standardized(design, beta)[2]
            return torch.where(
                index <= 0,
                torch.special.log_ndtr(index),
                torch.log1p(-0.5 * torch.erfc(index / math.sqrt(2))),
            )

        a, b = logarithms(alternative), logarithms(baseline)
        both_zero = torch.isneginf(a) & torch.isneginf(b)
        a, b = torch.where(both_zero, 0.0, a), torch.where(both_zero, 0.0, b)
        result = torch.where(
            a >= b,
            a.exp() * -torch.expm1((b - a).clamp_max(0)),
            b.exp() * torch.expm1((a - b).clamp_max(0)),
        )
        return torch.where(both_zero, 0.0, result)


def saved_heteroskedastic(result, state, features):
    if result.spec.estimator != "hetprobit":
        return None, features, None
    variance = variance_roles(result)
    if not isinstance(result.extra, Mapping):
        _error("invalid_result", "Saved heteroskedastic-probit metadata are invalid.")
    if result.extra.get("variance_function") != "sigma = exp(z'g), no constant":
        _error("invalid_result", "Saved latent standard-deviation function is missing or stale.")
    if state.df is not None:
        _error("invalid_result", "Native hetprobit records normal, rather than t, inference.")
    if result.extra.get("constrained_terms"):
        _error(
            "invalid_result",
            "The native hetprobit fitted contract has no constrained coefficients.",
        )
    mean_features, variance_features = {}, {}
    for term, feature in features.items():
        name = (
            feature[1]
            if feature[0] == "numeric"
            else feature[1][0]
            if feature[0] == "category"
            else None
        )
        if feature[0] == "constant" or name in result.spec.predictors:
            mean_features[term] = feature
        if name in variance:
            variance_features[f"lnsigma:{term}"] = feature
    if set(mean_features) & set(variance_features):
        _error("invalid_result", "Mean and variance term names collide in the saved design.")
    mapped = {**mean_features, **variance_features}
    omitted = result.provenance.get("omitted_terms", [])
    if (
        not isinstance(omitted, list)
        or any(not isinstance(term, str) for term in omitted)
        or len(set(omitted)) != len(omitted)
        or set(omitted) - set(mapped)
    ):
        _error("invalid_result", "Saved mean/variance omissions are invalid.")
    if set(state.terms) != set(mapped) - set(omitted):
        _error(
            "invalid_result",
            "Saved coefficients do not account for both fitted equations and omissions.",
        )
    mean_terms = set(mean_features) & set(state.terms)
    variance_terms = set(variance_features) & set(state.terms)
    recorded = result.extra.get("variance_terms")
    if (
        not mean_terms
        or not variance_terms
        or not isinstance(recorded, list)
        or any(not isinstance(term, str) for term in recorded)
        or len(set(recorded)) != len(recorded)
        or set(recorded) != variance_terms
    ):
        _error("invalid_result", "Saved variance terms or fitted equation blocks are incomplete.")
    raw_centres = result.extra.get("variance_regressor_means")
    if not isinstance(raw_centres, list) or any(isinstance(value, bool) for value in raw_centres):
        _error("invalid_result", "Saved fitting centres must be a list of finite real values.")
    centres = _numeric(raw_centres, code="invalid_result", ndim=1)
    if len(centres) != len(recorded):
        _error(
            "invalid_result", "Saved fitting-centre metadata do not match the variance equation."
        )
    for coefficient in result.coefficients:
        if coefficient.equation != (
            result.spec.outcome if coefficient.term in mean_terms else "lnsigma"
        ):
            _error(
                "invalid_result",
                "Saved coefficient equations disagree with fitted mean/variance roles.",
            )
    coding = result.provenance.get("categorical_encoding", {})
    used_categories = set(result.spec.categorical) & (set(result.spec.predictors) | set(variance))
    if set(coding) != used_categories:
        _error(
            "invalid_result", "Saved categorical encodings do not match the fitted equation roles."
        )
    for name in used_categories:
        record = coding[name]
        if record.get("coding") != "treatment_drop_first" or not _same(
            record.get("reference"), record["levels"][0]
        ):
            _error("invalid_result", "Saved categorical reference/coding is stale.")
    for name in ["zero_outcomes", "nonzero_outcomes"]:
        value = _numeric(result.extra.get(name), code="invalid_result", ndim=0)
        if float(value) <= 0:
            _error("invalid_result", "Saved binary-outcome counts must be positive.")
    mean_slopes = {
        name: state.terms.index(name)
        for name in result.spec.predictors
        if name in mean_terms and name not in result.spec.categorical
    }
    variance_slopes = {
        name: state.terms.index(f"lnsigma:{name}")
        for name in variance
        if f"lnsigma:{name}" in variance_terms and name not in result.spec.categorical
    }
    adapter = HeteroskedasticProbit(
        tuple(i for i, term in enumerate(state.terms) if term in mean_terms),
        tuple(i for i, term in enumerate(state.terms) if term in variance_terms),
        mean_slopes,
        variance_slopes,
    )
    return adapter, mapped, set(state.terms)


def check_workspace(rows, parameters):
    # Two changed categorical designs/Jacobians can coexist; each component-
    # wise log product also retains scale/sign/exponent buffers. Count thirty-
    # two dense float64 matrices, rather than only the returned Jacobian.
    if 32 * rows * max(1, parameters) * 8 > _MAX_DESIGN_BYTES:
        _error(
            "prediction_memory_limit",
            "Heteroskedastic-probit gradients exceed 256 MiB; use smaller evaluation batches.",
        )


def parameter_uncertainty(state):
    """Keep covariance SD inside density products before any underflow."""
    scale = state.covariance.diagonal().sqrt()
    divisor = torch.where(scale > 0, scale, 1.0)
    correlation = state.covariance / divisor[:, None] / divisor[None, :]
    return scale, correlation


def predict_heteroskedastic(model, design, original, positions, kind, significance, interval, term):
    adapter = model.heteroskedastic
    if kind == "derivative" and (term not in model.predictors or term in model.categories):
        _error(
            "invalid_prediction_term", "Choose one continuous original mean or variance regressor."
        )
    jac_kind = "xb" if kind == "stdp" else kind
    need_jac = kind == "stdp" or interval is not None
    if need_jac:
        check_workspace(len(design.x), len(model.state.terms))
    beta = model.state.beta
    value = (
        adapter.effects(design, beta, term, "response")
        if kind == "derivative"
        else adapter.values(design, beta, jac_kind)
    )
    columns = {}
    if need_jac:
        parameter_scale, correlation = parameter_uncertainty(model.state)
        jac = adapter.jacobian(design, beta, jac_kind, term, parameter_scale)
        se = category_standard_error(jac, correlation)
        if kind == "stdp":
            value = se
        else:
            critical = critical_value(significance, model.state.df)
            columns.update(
                std_error=se, ci_low=value - critical * se, ci_high=value + critical * se
            )
    columns = {f"dydx[{term}]" if kind == "derivative" else kind: value, **columns}
    output = {}
    for name, values in columns.items():
        if not bool(torch.isfinite(values).all()):
            _error(
                "non_finite_prediction",
                "Heteroskedastic-probit predictions or intervals exceed float64 range.",
            )
        scattered = torch.full((len(original),), float("nan"), dtype=torch.float64)
        scattered[positions] = values
        output[name] = scattered.tolist()
    result = as_frame(pd.DataFrame(output, index=original.index))
    present = set(positions.tolist())
    result.attrs.update(
        kind=kind,
        estimator="hetprobit",
        precision="float64",
        missing_row_positions=[i for i in range(len(original)) if i not in present],
        interval_method="pointwise delta method" if interval else None,
        response_definition="latent error standard deviation exp(z gamma)"
        if kind == "sigma"
        else "mean-equation linear index x beta"
        if kind in {"xb", "stdp"}
        else "probability of a positive outcome Phi(x beta exp(-z gamma))",
        covariance_source="full saved mean/variance parameter covariance",
    )
    return result
