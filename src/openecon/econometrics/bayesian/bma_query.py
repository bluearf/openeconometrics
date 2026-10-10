"""Indexed Gaussian BMA prediction with full within/between uncertainty."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import pandas as pd
import torch

from openecon.econometrics.state_lifecycle import _state_copy_admission
from openecon.resources import workspace_budget_bytes
from . import bma as b, bma_common as c, bma_kernels as k

SCHEMA = "openecon.bayesian_bma_prediction.v1"
RESULT_KEYS = (
    "positions",
    "model_mean",
    "model_mean_scale",
    "model_outcome_scale",
    "mean",
    "mean_exists",
    "variance_exists",
    "mean_quantiles",
    "outcome_quantiles",
    "within_mean_covariance",
    "between_mean_covariance",
    "mean_covariance",
    "outcome_covariance",
    "expected_conditional_variance",
    "parameter_query_covariance",
    "within_parameter_query_covariance",
    "between_parameter_query_covariance",
)


def header(saved):
    raw = c.load(saved, b.SCHEMA)
    if not isinstance(raw.get("source"), Mapping):
        c.fail("Prediction requires a complete posterior source header.")
    spec = b.spec_admission(raw.get("spec"))
    n = c.integer(raw["source"].get("n"), "posterior source rows", 1, c.MAX_ROWS)
    return raw, spec, n


def query_admission(source, parent, spec, n):
    if not isinstance(source, Mapping):
        c.fail("Query source must be a mapping.")
    nq = c.integer(source.get("n"), "query source rows", 1, c.MAX_ROWS)
    c.plan(n, spec, queries=nq)
    _state_copy_admission(
        (parent, source),
        operation="BMA complete query target/source",
        budget_bytes=min(spec["max_bytes"], workspace_budget_bytes()),
    )
    return c.source_admission(source, [*spec["forced"], *spec["optional"]], source.get("missing"))


def result_admission(result, m, q, total):
    c.keys(result, RESULT_KEYS, "complete BMA prediction cache")
    c.array(result["positions"], (q,), "query complete positions", integers=True)
    for key in ("model_mean", "model_mean_scale", "model_outcome_scale"):
        c.array(result[key], (m, q), key)
    c.array(result["mean"], (q,), "mixture mean", items_nullable=True)
    for key in ("mean_exists", "variance_exists"):
        c.array(result[key], (), key, booleans=True)
    for key in ("mean_quantiles", "outcome_quantiles"):
        c.array(result[key], (q, 2), key)
    for key in (
        "within_mean_covariance",
        "between_mean_covariance",
        "mean_covariance",
        "outcome_covariance",
    ):
        c.array(result[key], (q, q), key, nullable=True)
    c.array(
        result["expected_conditional_variance"], (), "expected conditional variance", nullable=True
    )
    for key in (
        "parameter_query_covariance",
        "within_parameter_query_covariance",
        "between_parameter_query_covariance",
    ):
        c.array(result[key], (total + 1, q), key, nullable=True)


@c.cpu_call
def derive(parent, source, alpha):
    _, design = c.source_matrices(source, outcome=False)
    components = parent["components"]
    weights = parent["results"]["model_weights"]
    nobs = len(parent["source"]["positions"])
    q, total = len(design), design.shape[1]
    means = []
    mean_scales = []
    outcome_scales = []
    covariances = []
    for v in components:
        locations = k.tensor(v["location"])
        means.append((design @ locations).tolist())
        subset = design[:, v["indices"]]
        posterior = v["posterior"]
        vn = k.tensor(posterior["conditional_scale_matrix"])
        factor = subset @ torch.linalg.cholesky(vn)
        variance = posterior["scale"] / posterior["shape"]
        scale = factor * math.sqrt(variance)
        scalar = scale.square().sum(1)
        if bool((scalar <= 0).any()):
            c.fail(
                "A proper-model query uncertainty is not float64 representable.",
                "numerical_failure",
            )
        mean_scales.append(scalar.sqrt().tolist())
        outcome_scales.append((scalar + variance).sqrt().tolist())
        if v["variance_exists"]:
            factor = factor * math.sqrt(posterior["variance_mean"])
            covariances.append((factor @ factor.T).tolist())
    mean_exists = all(v["mean_exists"] for v in components)
    variance_exists = all(v["variance_exists"] for v in components)
    estimated = (
        [
            k.finite(
                math.fsum(w * row[j] for w, row in zip(weights, means, strict=True)),
                "Mixture query mean",
            )
            for j in range(q)
        ]
        if mean_exists
        else [None] * q
    )
    within = between = cm = cy = noise = None
    if variance_exists:
        estimated, within, between, cm = k.mixture_moments(means, covariances, weights)
        noise = k.finite(
            math.fsum(
                w * v["posterior"]["variance_mean"]
                for w, v in zip(weights, components, strict=True)
            ),
            "Predictive independent variance",
        )
        cy = (k.tensor(cm) + noise * torch.eye(q, dtype=k.DT, device="cpu")).tolist()
    mean_limits = []
    outcome_limits = []
    for j in range(q):
        laws = [
            (means[i][j], mean_scales[i][j], v["posterior"]["degrees_of_freedom"])
            for i, v in enumerate(components)
        ]
        ylaws = [
            (means[i][j], outcome_scales[i][j], v["posterior"]["degrees_of_freedom"])
            for i, v in enumerate(components)
        ]
        mean_limits.append(
            [k.quantile(alpha / 2, laws, weights), k.quantile(1 - alpha / 2, laws, weights)]
        )
        outcome_limits.append(
            [k.quantile(alpha / 2, ylaws, weights), k.quantile(1 - alpha / 2, ylaws, weights)]
        )
    pc = wp = bp = None
    if all(math.fsum((p["shape"], (nobs - 3) / 2)) > 0 for p in parent["priors"]):
        parameter_means = k.tensor(
            [[*v["location"], v["posterior"]["variance_mean"]] for v in components]
        )
        mean_parameters = k.tensor(parent["results"]["joint_mean"])
        query_means = k.tensor(means)
        query_center = k.tensor(estimated)
        wp = torch.zeros((total + 1, q), dtype=k.DT, device="cpu")
        for weight, v in zip(weights, components, strict=True):
            wp[:total] += weight * (k.tensor(v["covariance"]) @ design.T)
        bp = torch.einsum(
            "m,mi,mj->ij",
            k.tensor(weights),
            parameter_means - mean_parameters,
            query_means - query_center,
        )
        pc = (wp + bp).tolist()
        wp = wp.tolist()
        bp = bp.tolist()
    result = dict(
        positions=list(source["positions"]),
        model_mean=means,
        model_mean_scale=mean_scales,
        model_outcome_scale=outcome_scales,
        mean=estimated,
        mean_exists=mean_exists,
        variance_exists=variance_exists,
        mean_quantiles=mean_limits,
        outcome_quantiles=outcome_limits,
        within_mean_covariance=within,
        between_mean_covariance=between,
        mean_covariance=cm,
        outcome_covariance=cy,
        expected_conditional_variance=noise,
        parameter_query_covariance=pc,
        within_parameter_query_covariance=wp,
        between_parameter_query_covariance=bp,
    )
    c._state_copy_admission(result, operation="BMA derived complete query")
    return result


@c.cpu_call
def replay(saved):
    raw = c.load(saved, SCHEMA)
    c.keys(
        raw,
        ("schema", "posterior", "posterior_digest", "source", "alpha", "results", "digest"),
        "BMA query state",
    )
    parent, spec, n = header(raw["posterior"])
    alpha = c.real(raw["alpha"], "query alpha")
    if not 0 < alpha / 2 < 1 - alpha / 2 < 1:
        c.fail("Query alpha does not resolve two strict probabilities.")
    source = raw["source"]
    if not isinstance(source, Mapping):
        c.fail("Query source must be a mapping.")
    nq = c.integer(source.get("n"), "query rows", 1, c.MAX_ROWS)
    positions = source.get("positions")
    if not isinstance(positions, (list, tuple)) or not 1 <= len(positions) <= nq:
        c.fail("Invalid query positions header.")
    result_admission(
        raw["results"],
        1 << len(spec["optional"]),
        len(positions),
        1 + len(spec["forced"]) + len(spec["optional"]),
    )
    query_admission(source, parent, spec, n)
    parent = b.replay(parent)
    if raw["posterior_digest"] != parent["digest"]:
        c.fail("Prediction target lineage disagrees.")
    expected = derive(parent, source, alpha)
    c.same(raw["results"], expected, "full joint mixture prediction")
    if raw["digest"] != c.digest({key: value for key, value in raw.items() if key != "digest"}):
        c.fail("BMA query digest disagrees.")
    return raw


class BMAPrediction(c.TypedState):
    """Indexed finite Student-t mixture predictions with full shared-model cross-row uncertainty."""

    schema_version: Literal["openecon.bayesian_bma_prediction.v1"] = SCHEMA

    @classmethod
    def _replay(cls, value):
        return replay(value)

    @c.cpu_call
    def summary(self):
        """Return original query labels, true mean/outcome mixture intervals and full covariance attrs."""
        raw = replay(self)
        r = raw["results"]
        index = c.decode_index(raw["source"]["index"], raw["source"]["n"])
        columns = {
            "posterior_mean": r["mean"],
            "mean_credible_lower": [v[0] for v in r["mean_quantiles"]],
            "mean_credible_upper": [v[1] for v in r["mean_quantiles"]],
            "outcome_predictive_lower": [v[0] for v in r["outcome_quantiles"]],
            "outcome_predictive_upper": [v[1] for v in r["outcome_quantiles"]],
        }
        full = {key: [None] * len(index) for key in columns}
        for key, values in columns.items():
            for position, value in zip(r["positions"], values, strict=True):
                full[key][position] = value
        frame = pd.DataFrame(full, index=index)
        frame.attrs.update(
            mean_covariance=r["mean_covariance"],
            outcome_covariance=r["outcome_covariance"],
            within_mean_covariance=r["within_mean_covariance"],
            between_mean_covariance=r["between_mean_covariance"],
            parameter_query_covariance=r["parameter_query_covariance"],
            sample_positions=r["positions"],
            mean_exists=r["mean_exists"],
            variance_exists=r["variance_exists"],
            credible_probability=1 - raw["alpha"],
        )
        return frame


@c.cpu_call
def bayes_bma_predict(saved, *, data, missing="raise", alpha=None) -> BMAPrediction:
    """Predict joint indexed conditional means/new outcomes using the entire model posterior.

    Equal-tail limits invert the finite Student-t mixture CDF. Full covariance
    includes within-model uncertainty, between-model means and independent new
    noise on the outcome diagonal. Query source is admitted before target replay.
    """
    parent, spec, n = header(saved)
    if missing not in ("raise", "drop"):
        c.fail("Query missing must be raise or drop.", "invalid_spec")
    alpha = spec["alpha"] if alpha is None else c.real(alpha, "query alpha")
    if not 0 < alpha / 2 < 1 - alpha / 2 < 1:
        c.fail("Query alpha must resolve strict probabilities.", "invalid_option")
    columns = [*spec["forced"], *spec["optional"]]
    nq = c.resident(data, columns)
    c.plan(
        n,
        spec,
        queries=nq,
        index_bytes=c.index_buffer_bytes(data.index, budget_bytes=spec["max_bytes"]),
    )
    source = c.capture(data, columns, missing, {**spec, "source_n": n}, query=True)
    query_admission(source, parent, spec, n)
    parent = b.replay(parent)
    result = derive(parent, source, alpha)
    body = dict(
        schema=SCHEMA,
        posterior=parent,
        posterior_digest=parent["digest"],
        source=source,
        alpha=alpha,
        results=result,
    )
    body["digest"] = c.digest(body)
    return BMAPrediction(payload=body)


def bayes_bma_prediction_restore(saved) -> BMAPrediction:
    """Replay complete saved indexed mixture predictions without a new posterior fit or draw."""
    return BMAPrediction(payload=replay(saved))
