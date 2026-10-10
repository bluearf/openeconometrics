"""Exhaustive proper Gaussian BMA with complete original-unit mixture uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from openecon.econometrics.state_lifecycle import _state_copy_admission
from openecon.resources import workspace_budget_bytes
from .posterior import NormalInverseGammaPrior
from . import bma_common as c, bma_kernels as k

SCHEMA = "openecon.bayesian_bma.v1"
SPEC_KEYS = ("y", "forced", "optional", "missing", "alpha", "max_work", "max_bytes")
POSTERIOR_KEYS = (
    "mean",
    "conditional_scale_matrix",
    "shape",
    "scale",
    "degrees_of_freedom",
    "coefficient_scale_matrix",
    "coefficient_covariance",
    "variance_mean",
    "log_marginal_likelihood",
)
COMPONENT_KEYS = (
    "indices",
    "posterior",
    "location",
    "scale_matrix",
    "covariance",
    "mean_exists",
    "variance_exists",
    "sigma_variance_exists",
    "sigma_variance",
)
RESULT_KEYS = (
    "log_model_evidence",
    "log_model_prior",
    "log_model_weights",
    "model_weights",
    "log_mixture_evidence",
    "prior_inclusion",
    "posterior_inclusion",
    "inclusion_covariance",
    "coefficient_mean",
    "coefficient_variance",
    "mean_exists",
    "variance_exists",
    "within_coefficient_covariance",
    "between_coefficient_covariance",
    "coefficient_covariance",
    "joint_mean",
    "joint_covariance",
    "within_joint_covariance",
    "between_joint_covariance",
    "variance_mean",
    "variance_mean_exists",
    "coefficient_quantiles",
    "coefficient_atom_zero",
    "sigma2_quantiles",
)


def spec_admission(raw):
    c.keys(raw, SPEC_KEYS, "BMA specification")
    spec = c.specification(**raw)
    c.same(raw, spec, "BMA specification")
    return spec


def prior_admission(value, dimension):
    if isinstance(value, NormalInverseGammaPrior):
        value = value.__dict__
    c.keys(
        value,
        ("schema_version", "mean", "scale_matrix", "shape", "scale"),
        "per-model proper NIG prior",
    )
    if value["schema_version"] != "openecon.normal_inverse_gamma_prior.v1":
        c.fail("Unsupported model prior schema.", "invalid_prior")
    c.array(value["mean"], (dimension,), "prior mean")
    c.array(value["scale_matrix"], (dimension, dimension), "prior scale")
    c.real(value["shape"], "prior shape", positive=True)
    c.real(value["scale"], "prior scale", positive=True)
    if value["shape"] > 1e12:
        c.fail("Proper prior shape exceeds1e12.", "invalid_prior")
    return NormalInverseGammaPrior.model_validate(dict(value))


def models_admission(priors, odds, spec):
    dimensions = c.indices(spec)
    m = len(dimensions)
    if not isinstance(priors, (tuple, list)) or len(priors) != m:
        c.fail("Supply exactly one proper NIG prior for every model mask.", "invalid_prior")
    if not isinstance(odds, (list, tuple)) or len(odds) != m:
        c.fail(
            "Supply exactly one positive prior model odds value for every mask.", "invalid_prior"
        )
    c.array(odds, (m,), "prior model odds")
    odds = [c.real(v, "prior model odds", positive=True) for v in odds]
    prior = [prior_admission(v, len(index)) for v, index in zip(priors, dimensions, strict=True)]
    return prior, odds


def component_admission(value, index, total):
    c.keys(value, COMPONENT_KEYS, "saved model component")
    c.array(value["indices"], (len(index),), "component indices", integers=True)
    if list(value["indices"]) != index:
        c.fail("Component subset mask/order disagrees.")
    c.array(value["location"], (total,), "embedded component location")
    c.array(value["scale_matrix"], (total, total), "embedded component t scale")
    c.array(value["covariance"], (total, total), "embedded component covariance", nullable=True)
    for key in ("mean_exists", "variance_exists", "sigma_variance_exists"):
        c.array(value[key], (), key, booleans=True)
    c.array(value["sigma_variance"], (), "sigma variance", nullable=True)
    p = value["posterior"]
    c.keys(p, POSTERIOR_KEYS, "complete per-model posterior")
    d = len(index)
    c.array(p["mean"], (d,), "component posterior location")
    for key in ("conditional_scale_matrix", "coefficient_scale_matrix", "coefficient_covariance"):
        c.array(p[key], (d, d), key, nullable=key == "coefficient_covariance")
    for key in ("shape", "scale", "degrees_of_freedom", "log_marginal_likelihood"):
        c.array(p[key], (), key)
    c.array(p["variance_mean"], (), "component variance mean", nullable=True)


def results_admission(value, spec):
    c.keys(value, RESULT_KEYS, "complete mixture results")
    p = len(spec["optional"])
    m = 1 << p
    total = 1 + len(spec["forced"]) + p
    for key in ("log_model_evidence", "log_model_prior", "log_model_weights", "model_weights"):
        c.array(value[key], (m,), key)
    for key in ("prior_inclusion", "posterior_inclusion"):
        c.array(value[key], (p,), key)
    c.array(value["inclusion_covariance"], (p, p), "inclusion covariance")
    for key in ("coefficient_mean", "coefficient_variance"):
        c.array(value[key], (total,), key, items_nullable=True)
    for key in ("mean_exists", "variance_exists"):
        c.array(value[key], (total,), key, booleans=True)
    for key in (
        "within_coefficient_covariance",
        "between_coefficient_covariance",
        "coefficient_covariance",
    ):
        c.array(value[key], (total, total), key, nullable=True)
    for key in ("within_joint_covariance", "between_joint_covariance", "joint_covariance"):
        c.array(value[key], (total + 1, total + 1), key, nullable=True)
    c.array(value["joint_mean"], (total + 1,), "joint parameter mean", items_nullable=True)
    c.array(value["variance_mean"], (), "variance mean", nullable=True)
    c.array(value["variance_mean_exists"], (), "variance mean availability", booleans=True)
    c.array(value["log_mixture_evidence"], (), "complete marginal log evidence")
    c.array(value["coefficient_quantiles"], (total, 2), "mixture coefficient quantiles")
    c.array(value["coefficient_atom_zero"], (total,), "coefficient zero atom")
    c.array(value["sigma2_quantiles"], (2,), "mixture variance quantiles")


@c.cpu_call
def derive(source, spec, priors, odds):
    y, x = c.source_matrices(source)
    total = x.shape[1]
    components = [
        k.component(y, x, prior, index, total)
        for prior, index in zip(priors, c.indices(spec), strict=True)
    ]
    result = k.aggregate(components, odds, spec["optional"], spec["forced"], spec["alpha"])
    return components, result


@c.cpu_call
def replay(saved):
    raw = c.load(saved, SCHEMA)
    c.keys(
        raw,
        (
            "schema",
            "spec",
            "source",
            "priors",
            "prior_model_odds",
            "components",
            "results",
            "digest",
        ),
        "BMA complete state",
    )
    spec = spec_admission(raw["spec"])
    source = raw["source"]
    if not isinstance(source, Mapping):
        c.fail("Saved BMA source must be a mapping.")
    n = c.integer(source.get("n"), "saved source rows", 1, c.MAX_ROWS)
    c.plan(n, spec)
    _state_copy_admission(
        raw,
        operation="BMA full replay metadata",
        budget_bytes=min(spec["max_bytes"], workspace_budget_bytes()),
    )
    models = c.indices(spec)
    total = 1 + len(spec["forced"]) + len(spec["optional"])
    if not isinstance(raw["components"], (list, tuple)) or len(raw["components"]) != len(models):
        c.fail("Saved component count differs from exhaustive model space.")
    # Complete cached shapes/types precede source reconstruction and every factorization.
    for value, index in zip(raw["components"], models, strict=True):
        component_admission(value, index, total)
    results_admission(raw["results"], spec)
    priors, odds = models_admission(raw["priors"], raw["prior_model_odds"], spec)
    c.source_admission(source, [spec["y"], *spec["forced"], *spec["optional"]], spec["missing"])
    components, results = derive(source, spec, priors, odds)
    c.same(raw["components"], components, "per-model full posterior")
    c.same(raw["results"], results, "full model-mixture posterior")
    if raw["digest"] != c.digest({key: value for key, value in raw.items() if key != "digest"}):
        c.fail("Complete BMA digest disagrees.")
    return raw


class BMAResult(c.TypedState):
    """Versioned exhaustive proper-NIG model posterior with genuine mixture uncertainty."""

    schema_version: Literal["openecon.bayesian_bma.v1"] = SCHEMA

    @classmethod
    def _replay(cls, value):
        return replay(value)

    @c.cpu_call
    def summary(self):
        """Return embedded coefficient mixture means, inclusion, atoms and exact equal-tail limits."""
        state = replay(self)
        r = state["results"]
        s = state["spec"]
        terms = ["Intercept", *s["forced"], *s["optional"]]
        inclusion = [1.0] * (1 + len(s["forced"])) + list(r["posterior_inclusion"])
        frame = pd.DataFrame(
            dict(
                posterior_mean=r["coefficient_mean"],
                posterior_variance=r["coefficient_variance"],
                inclusion_probability=inclusion,
                atom_at_zero=r["coefficient_atom_zero"],
                credible_lower=[v[0] for v in r["coefficient_quantiles"]],
                credible_upper=[v[1] for v in r["coefficient_quantiles"]],
                mean_exists=r["mean_exists"],
                variance_exists=r["variance_exists"],
            ),
            index=pd.Index(terms, name="term"),
        )
        frame.attrs.update(
            coefficient_covariance=r["coefficient_covariance"],
            within_covariance=r["within_coefficient_covariance"],
            between_covariance=r["between_coefficient_covariance"],
            joint_parameter_covariance=r["joint_covariance"],
            uncertainty="finite posterior t/point-mass mixture",
            credible_probability=1 - s["alpha"],
        )
        return frame

    @c.cpu_call
    def model_summary(self):
        """Return every declared model mask, exact evidence and normalized prior/posterior probabilities."""
        state = replay(self)
        r = state["results"]
        return pd.DataFrame(
            dict(
                mask=range(len(r["model_weights"])),
                terms=[
                    [
                        "Intercept",
                        *[
                            state["spec"]["forced"][j - 1]
                            if j <= len(state["spec"]["forced"])
                            else state["spec"]["optional"][j - 1 - len(state["spec"]["forced"])]
                            for j in v["indices"][1:]
                        ],
                    ]
                    for v in state["components"]
                ],
                log_marginal_likelihood=r["log_model_evidence"],
                log_prior_probability=r["log_model_prior"],
                log_posterior_probability=r["log_model_weights"],
                posterior_probability=r["model_weights"],
            )
        )


@c.cpu_call
def bayes_bma(
    *,
    data,
    y,
    optional,
    priors,
    model_prior_odds,
    forced=(),
    missing="raise",
    alpha=0.05,
    max_work=c.DEFAULT_WORK,
    max_bytes=c.DEFAULT_BYTES,
) -> BMAResult:
    """Enumerate every optional subset on one common sample using explicit proper per-model NIG priors.

    Intercept and forced predictors are always included. Priors/model odds are
    ordered by integer bit mask of optional names. No BIC/AIC weights, model
    search, improper priors or implicit complete-case changes are applied.
    """
    spec = c.specification(y, forced, optional, missing, alpha, max_work, max_bytes)
    n = c.resident(data, [y, *spec["forced"], *spec["optional"]])
    c.plan(n, spec, index_bytes=c.index_buffer_bytes(data.index, budget_bytes=spec["max_bytes"]))
    _state_copy_admission(
        (priors, model_prior_odds),
        operation="BMA all prior/model input",
        budget_bytes=min(spec["max_bytes"], workspace_budget_bytes()),
    )
    priors, odds = models_admission(priors, model_prior_odds, spec)
    source = c.capture(data, [y, *spec["forced"], *spec["optional"]], missing, spec)
    components, results = derive(source, spec, priors, odds)
    body = dict(
        schema=SCHEMA,
        spec=spec,
        source=source,
        priors=[dict(v.__dict__) for v in priors],
        prior_model_odds=odds,
        components=components,
        results=results,
    )
    body["digest"] = c.digest(body)
    return BMAResult(payload=body)


def bayes_bma_restore(saved) -> BMAResult:
    """Restore and replay all finite model priors/evidence/uncertainty without selecting or fitting a new model."""
    return BMAResult(payload=replay(saved))
