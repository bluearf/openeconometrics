"""Saved empirical survey targets replayed from complete coefficient replicas.

The primitive record retains the actual weight matrix, including complete
zero-domain and census PSUs. A digest detects mutation; it does not authenticate
the source data or a supplied bootstrap's sampling justification.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.resources import plan_workspace, workspace_budget_bytes
from .common import FLOAT, digest, finite, number
from .regression_common import MAX_PARAMETERS, MAX_WORK, SurveyRegressionResult, cpu_call

_PRIMITIVE_KEYS = {
    "X", "y", "sample_positions", "base_weights", "groups", "psu_factors",
    "replicate_weights", "sample_input_sha256", "replicate_weight_sha256",
}
_UNCERTAINTY = (
    "complete coefficient refits and replicate-weighted empirical covariate distribution; "
    "no future-outcome interval"
)
_VARIANCE = "sum_r multiplier[r] * (target[r]-center[r]) outer product"


def target_replay_work(n_design, n_used, k, m, R):
    """Conservative operation bound, evaluated before tensor or replica allocation."""
    if any(type(v) is not int or v < 1 for v in (n_design, n_used, k, m, R)):
        raise AnalysisError("invalid_resource_budget", "Target replay dimensions must be positive integers.")
    return ((R + 1) * n_used * k * m * 8 + R * m * m * 4 + n_design * R * 8
            + (R + 1) * n_used * k * k * 8 + n_design * k * 8)


def _target_labels(target, profiles, variables, regressors):
    if target == "mean":
        if variables or not isinstance(profiles, dict) or not 1 <= len(profiles) <= MAX_PARAMETERS:
            raise ValueError("Predictive means require 1..32 ordered partial covariate profiles only.")
        for label, at in profiles.items():
            if (not isinstance(label, str) or not label.strip() or len(label) > 200
                    or not isinstance(at, dict) or not 0 < len(at) < len(regressors)
                    or any(key not in regressors for key in at)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in at.values())):
                raise ValueError("Each mean profile fixes a nonempty proper subset of numeric regressors.")
        return tuple(profiles)
    if target != "ame" or profiles or not isinstance(variables, (list, tuple)) or not variables:
        raise ValueError("AME requires a nonempty ordered variable list and no mean profiles.")
    if (len(variables) > MAX_PARAMETERS
            or any(not isinstance(v, str) or v not in regressors for v in variables)
            or len(set(variables)) != len(variables)):
        raise ValueError("AME variables must be distinct fitted numeric regressors.")
    return tuple("AME:" + v for v in variables)


@cpu_call
def evaluate_targets(X, coefficients, weights, sample_positions, *, family, labels,
                     target, profiles, variables, regressors, intercept):
    """Evaluate targets on selected X rows using each replica's actual row weights.

    X contains the original admitted rows in physical order; weights retains all
    design rows. Zero replica weights are omitted from the target denominator.
    """
    if family not in {"logit", "probit", "poisson"}:
        raise ValueError("Empirical replicated targets support logit, probit and Poisson.")
    expected = (("_cons",) if intercept else ()) + tuple(regressors)
    if tuple(labels) != expected:
        raise ValueError("Target coefficient order differs from the fitted regression.")
    output_labels = _target_labels(target, profiles, variables, regressors)
    if (X.ndim != 2 or X.shape != (len(sample_positions), len(expected))
            or coefficients.shape != (len(expected),) or weights.ndim != 1):
        raise ValueError("Target primitive dimensions differ from the original admitted sample.")
    finite(X, "Target covariates")
    finite(coefficients, "Target coefficients")
    finite(weights, "Target weights")
    if bool((weights < 0).any()):
        raise ValueError("Target weights must be nonnegative.")
    used = weights[sample_positions]
    selected = used > 0
    if not bool(selected.any()):
        raise AnalysisError("empty_survey_domain", "A target replica has no positive-weight admitted rows.")
    used = used[selected]
    scaled = used / used.max()
    if bool((scaled <= 0).any()):
        raise AnalysisError("survey_numerical_failure", "Positive target weight underflowed during normalization.")
    normalized = scaled / finite(scaled.sum(), "Target weight denominator")
    if bool((normalized <= 0).any()):
        raise AnalysisError("survey_numerical_failure", "Positive normalized target weights underflowed.")
    values = X[selected]

    def response(index, *, derivative=False):
        finite(index, "Target linear predictor")
        if family == "logit":
            log_mean = -torch.nn.functional.softplus(-index)
            return (torch.exp(log_mean - torch.nn.functional.softplus(index))
                    if derivative else torch.exp(log_mean))
        if family == "probit":
            return (torch.exp(-0.5 * index.square()) / math.sqrt(2 * math.pi)
                    if derivative else torch.exp(torch.special.log_ndtr(index)))
        mean = finite(torch.exp(index), "Poisson target mean")
        if bool((mean <= 0).any()):
            raise AnalysisError("survey_numerical_failure", "Positive Poisson target means underflowed.")
        return mean

    if target == "mean":
        estimates = []
        for at in profiles.values():
            fixed = values.clone()
            for variable, value in at.items():
                fixed[:, expected.index(variable)] = value
            estimates.append((normalized * response(fixed @ coefficients)).sum())
        output = torch.stack(estimates)
    else:
        average_derivative = (normalized * response(values @ coefficients, derivative=True)).sum()
        output = coefficients[[expected.index(v) for v in variables]] * average_derivative
    if output.shape != (len(output_labels),):
        raise ValueError("Empirical target order changed during evaluation.")
    return finite(output, "Empirical replicated targets")


@cpu_call
def target_covariance(original, replicas, record):
    """Use the already admitted coefficient plan's centering and multipliers."""
    original = torch.as_tensor(original, dtype=FLOAT, device="cpu")
    replicas = torch.as_tensor(replicas, dtype=FLOAT, device="cpu")
    if record["centering"] == "original":
        difference = replicas - original
    elif record["centering"] == "replicate_mean":
        difference = replicas - replicas.mean(0)
    else:
        difference = torch.empty_like(replicas)
        strata = record["replicate_stratum_indices"]
        for h in sorted(set(strata)):
            indices = [r for r, value in enumerate(strata) if value == h]
            difference[indices] = replicas[indices] - replicas[indices].mean(0)
    multipliers = torch.tensor(record["variance_multipliers"], dtype=FLOAT, device="cpu")
    covariance = finite((difference.T * multipliers) @ difference, "Empirical target covariance")
    return (covariance + covariance.T) * 0.5


def _sequence(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"Saved {name} dimensions are inconsistent.")


def _matrix(value, rows, columns, name):
    _sequence(value, rows, name)
    if any(not isinstance(row, (list, tuple)) or len(row) != columns for row in value):
        raise ValueError(f"Saved {name} dimensions are inconsistent.")


def _real_values(values, name):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f"Saved {name} must contain finite real scalars without coercion.")


def _workspace(n, used, k, m, count, groups, memory_mb):
    return plan_workspace(
        "saved empirical survey target replay",
        {"actual weights, factors, hashes and serialized replay": n * count * 384 + count * groups * 192,
         "selected covariates, responses and source hash replay": used * (k + 1) * 192 + n * (k + 1) * 96,
         "targets and full covariance replay": (count + 1) * m * 192 + m * m * 512},
        budget_bytes=min(workspace_budget_bytes(), memory_mb * 1024**2),
    )


class SurveyReplicateMarginsResult(BaseModel):
    """Complete empirical targets; source and target digests are not authentication."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["survey-replicate-margins-result-v1"] = "survey-replicate-margins-result-v1"
    source_result: SurveyRegressionResult
    target: Literal["mean", "ame"]
    profiles: dict[str, dict[str, float]]
    variables: tuple[str, ...]
    labels: tuple[str, ...] = Field(min_length=1, max_length=MAX_PARAMETERS)
    estimates: tuple[float, ...]
    replicate_estimates: tuple[tuple[float, ...], ...]
    covariance: tuple[tuple[float, ...], ...]
    alpha: float = Field(gt=0, lt=1)
    null: tuple[float, ...]
    metadata: dict[str, Any]
    integrity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def admit_raw_state(cls, data):
        """Shape/workspace admission precedes Pydantic numeric or tensor copies."""
        if not isinstance(data, dict):
            return data
        source = data.get("source_result")
        if isinstance(source, SurveyRegressionResult):
            source = source.model_dump(mode="json")
        if not isinstance(source, dict) or source.get("family") not in {"logit", "probit", "poisson"}:
            raise ValueError("Empirical target state needs a supported nonlinear coefficient result.")
        try:
            n = source["design"]["validation"]["nobs"]
            groups = source["design"]["validation"]["n_psu"]
            memory_mb = source["design"]["max_memory_mb"]
            count = source["metadata"]["replication"]["replicate_count"]
            k = len(source["labels"])
            positions = source["metadata"]["sample_positions"]
            m = len(data["labels"])
            primitive = data["metadata"]["primitive"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Saved target requires complete coefficient replication and primitives.") from exc
        if (any(type(v) is not int or v < 1 for v in (n, groups, count, k, m))
                or not 2 <= count <= 4096 or not k <= MAX_PARAMETERS or not m <= MAX_PARAMETERS
                or type(memory_mb) is not int or not 1 <= memory_mb <= 512
                or not isinstance(positions, (list, tuple)) or not 0 < len(positions) <= n
                or not isinstance(primitive, dict) or set(primitive) != _PRIMITIVE_KEYS):
            raise ValueError("Saved empirical target dimensions or primitive schema are invalid.")
        work = target_replay_work(n, len(positions), k, m, count)
        if n * count > 8_000_000 or work > MAX_WORK:
            raise AnalysisError("survey_regression_budget", "Saved empirical target replay exceeds its bounded work plan.")
        _workspace(n, len(positions), k, m, count, groups, memory_mb)
        _matrix(primitive["X"], len(positions), k, "selected X")
        _sequence(primitive["y"], len(positions), "selected y")
        _sequence(primitive["sample_positions"], len(positions), "sample positions")
        _sequence(primitive["base_weights"], n, "base weights")
        _sequence(primitive["groups"], n, "complete PSU groups")
        _matrix(primitive["psu_factors"], count, groups, "PSU factors")
        _matrix(primitive["replicate_weights"], count, n, "actual replica weights")
        _sequence(data.get("estimates"), m, "target estimates")
        _sequence(data.get("null"), m, "target null")
        _matrix(data.get("replicate_estimates"), count, m, "target replicas")
        _matrix(data.get("covariance"), m, m, "target covariance")
        for key in ("X", "psu_factors", "replicate_weights"):
            _real_values((v for row in primitive[key] for v in row), key)
        for key in ("base_weights", "y"):
            _real_values(primitive[key], key)
        for key in ("estimates", "null"):
            _real_values(data[key], key)
        for key in ("replicate_estimates", "covariance"):
            _real_values((v for row in data[key] for v in row), key)
        _real_values([data.get("alpha")], "alpha")
        _target_labels(data.get("target"), data.get("profiles"), data.get("variables"),
                       source.get("regressors", ()))
        return data

    @model_validator(mode="after")
    @cpu_call
    def validate_state(self):
        source = self.source_result
        record = source.metadata["replication"]
        primitive = self.metadata["primitive"]
        n, g, k, count = (source.design.validation.nobs, source.design.validation.n_psu,
                          len(source.labels), record["replicate_count"])
        positions = source.metadata["sample_positions"]
        m = len(self.labels)
        specified = _target_labels(self.target, self.profiles, self.variables, source.regressors)
        order_matches = (set(self.labels) == set(specified) if self.target == "mean" else self.labels == specified)
        if (not order_matches
                or len(set(self.labels)) != m or primitive["sample_positions"] != positions
                or any(type(v) is not int for v in primitive["sample_positions"])
                or self.metadata.get("precision") != "float64" or self.metadata.get("device") != "cpu"
                or self.metadata.get("uncertainty") != _UNCERTAINTY
                or self.metadata.get("variance_formula") != _VARIANCE
                or self.metadata.get("empirical_covariate_uncertainty") is not True
                or self.metadata.get("coefficient_uncertainty") is not True
                or self.metadata.get("dataset_support") is not False
                or self.metadata.get("stata_parity_validated") is not False):
            raise ValueError("Saved empirical target order, sample or uncertainty contract changed.")
        group_ids = primitive["groups"]
        if any(type(v) is not int or not 0 <= v < g for v in group_ids):
            raise ValueError("Complete physical PSU groups must be exact integer indices.")
        first, counts = {}, [0] * g
        for row, group in enumerate(group_ids):
            first.setdefault(group, row)
            counts[group] += 1
        if list(first) != list(range(g)):
            raise ValueError("Saved complete PSU order differs from physical first appearance.")
        strata = source.metadata["stratum_psu_indices"]
        if any(sum(counts[group] for group in members) != stratum.nobs
               for members, stratum in zip(strata, source.design.validation.strata)):
            raise ValueError("Saved complete PSU rows differ from the declared stratum geometry.")
        base = torch.tensor(primitive["base_weights"], dtype=FLOAT)
        weights = torch.tensor(primitive["replicate_weights"], dtype=FLOAT)
        factors = torch.tensor(primitive["psu_factors"], dtype=FLOAT)
        ids = torch.tensor(group_ids, dtype=torch.int64)
        if (bool((base <= 0).any()) or bool((weights < 0).any()) or bool((factors < 0).any())
                or not math.isclose(float(base.sum()), source.design.validation.sum_weights,
                                    rel_tol=1e-12, abs_tol=0.)
                or not math.isclose(float(base[positions].sum()), source.metadata["sum_target_weights"],
                                    rel_tol=1e-12, abs_tol=0.)):
            raise ValueError("Saved complete sampling weights or target denominator changed.")
        ratios = finite(weights / base, "Saved PSU factors")
        if (not torch.equal(factors, ratios[:, list(first.values())])
                or not torch.allclose(ratios, factors[:, ids], atol=1e-12, rtol=1e-12)):
            raise ValueError("Saved weights differ from their complete whole-PSU factor record.")
        certainty = [group for h, members in enumerate(strata)
                     if not source.metadata["fpc_multipliers"][h] for group in members]
        if certainty and not torch.allclose(factors[:, certainty], torch.ones_like(factors[:, certainty]),
                                            atol=1e-12, rtol=1e-12):
            raise ValueError("Every census PSU must retain its original weight in every target replica.")
        if (primitive["replicate_weight_sha256"] != record["replicate_weight_sha256"]
                or digest(primitive["replicate_weights"]) != record["replicate_weight_sha256"]):
            raise ValueError("Saved actual target weights differ from the coefficient weight-plan hash.")
        self._validate_weight_plan(base, weights, factors, ids)
        for r, evidence in enumerate(record["replicate_convergence"]):
            support = [position for position in positions if primitive["replicate_weights"][r][position] > 0]
            if support != evidence["sample_positions"]:
                raise ValueError("Saved target positive-weight support differs from its actual coefficient refit.")
        X = torch.tensor(primitive["X"], dtype=FLOAT)
        if source.intercept and not torch.equal(X[:, 0], torch.ones(len(positions), dtype=FLOAT)):
            raise ValueError("Saved target intercept column differs from the fitted design.")
        y = primitive["y"]
        if (source.family in {"logit", "probit"} and any(v not in (0., 1.) for v in y)
                or source.family == "poisson" and any(v < 0 or v > 2**53 or int(v) != v for v in y)):
            raise ValueError("Saved target outcome is outside the fitted family's response support.")
        values = [[0.] * (1 + len(source.regressors)) for _ in range(n)]
        selected = [False] * n
        for index, position in enumerate(positions):
            values[position] = [y[index], *primitive["X"][index][int(source.intercept):]]
            selected[position] = True
        sample_hash = digest([source.design.validation.design_input_sha256, selected, values,
                              [[1.] * (1 + len(source.regressors)) for _ in range(n)]])
        if primitive["sample_input_sha256"] != source.metadata["sample_input_sha256"] or sample_hash != primitive["sample_input_sha256"]:
            raise ValueError("Saved selected target covariates/responses differ from the admitted sample hash.")
        self._validate_coefficients(X, base, weights, ids)
        ordered_profiles = {label: self.profiles[label] for label in self.labels} if self.target == "mean" else {}
        expected = evaluate_targets(X, torch.tensor(source.coefficients, dtype=FLOAT), base, positions,
                                    family=source.family, labels=source.labels, target=self.target,
                                    profiles=ordered_profiles, variables=self.variables,
                                    regressors=source.regressors, intercept=source.intercept)
        replicas = torch.stack([
            evaluate_targets(X, torch.tensor(beta, dtype=FLOAT), weights[r], positions,
                             family=source.family, labels=source.labels, target=self.target,
                             profiles=ordered_profiles, variables=self.variables,
                             regressors=source.regressors, intercept=source.intercept)
            for r, beta in enumerate(record["replicate_estimates"])
        ])
        actual = torch.tensor(self.estimates, dtype=FLOAT)
        actual_replicas = torch.tensor(self.replicate_estimates, dtype=FLOAT)
        # Fit and restore use the same CPU float64 expression; this tolerance
        # permits harmless BLAS rounding, never replacement or omitted targets.
        if (not torch.allclose(actual, expected, atol=0., rtol=1e-11)
                or not torch.allclose(actual_replicas, replicas, atol=0., rtol=1e-11)):
            raise ValueError("Saved empirical targets differ from complete coefficient/weight replay.")
        covariance = torch.tensor(self.covariance, dtype=FLOAT)
        replay = target_covariance(expected, replicas, record)
        scale = float(replay.abs().max())
        normalized = covariance / scale if scale else covariance
        if (not bool(torch.isfinite(normalized).all())
                or not torch.allclose(normalized, normalized.T, atol=1e-12 if scale else 0., rtol=1e-12)
                or float(torch.linalg.eigvalsh(normalized).min()) < -1e-10
                or (not torch.allclose(normalized, replay / scale, atol=1e-10, rtol=1e-10)
                    if scale else bool((covariance != 0).any()))):
            raise ValueError("Saved empirical target covariance differs from complete method replay.")
        required = target_replay_work(n, len(positions), k, m, count)
        work = self.metadata.get("work")
        workspace = self.metadata.get("workspace")
        if (not isinstance(work, dict) or work.get("max_work_units") != MAX_WORK
                or any(type(work.get(key)) is not int for key in ("planned_work_units", "actual_work_units"))
                or not required <= work["planned_work_units"] <= MAX_WORK
                or not record["work"]["actual_work_units"] + required <= work["actual_work_units"] <= MAX_WORK
                or work.get("target_replay_work_units") != required):
            raise ValueError("Saved empirical target cumulative work evidence is incomplete.")
        if (not isinstance(workspace, dict) or not isinstance(workspace.get("buffers"), dict)
                or any(type(value) is not int or value < 0 for value in workspace["buffers"].values())
                or type(workspace.get("estimated_workspace_bytes")) is not int
                or type(workspace.get("budget_bytes")) is not int
                or not 0 < workspace["estimated_workspace_bytes"] <= workspace["budget_bytes"]
                or workspace["budget_bytes"] > source.design.max_memory_mb * 1024**2
                or workspace["estimated_workspace_bytes"] != sum(workspace["buffers"].values())
                or workspace["estimated_workspace_bytes"] < _workspace(
                    n, len(positions), k, m, count, g, source.design.max_memory_mb).estimated_bytes):
            raise ValueError("Saved empirical target workspace evidence is incomplete.")
        if digest(self.model_dump(mode="json", exclude={"integrity_sha256"})) != self.integrity_sha256:
            raise ValueError("Survey empirical target state integrity changed.")
        return self

    def _validate_coefficients(self, X, base, replica_weights, group_ids):
        """Replay likelihood equations and original PSU scores without solving a fit."""
        from .regression import _components, _solve

        source = self.source_result
        positions = source.metadata["sample_positions"]
        response = torch.tensor(self.metadata["primitive"]["y"], dtype=FLOAT)
        record = source.metadata["replication"]
        original_tolerance = source.metadata["convergence"].get("tolerance")
        if (type(original_tolerance) not in (int, float) or not math.isfinite(original_tolerance)
                or not 1e-14 <= original_tolerance <= 1e-3):
            raise ValueError("Saved target source needs a bounded actual fit tolerance.")
        coefficients = [source.coefficients, *record["replicate_estimates"]]
        evidence = [source.metadata["convergence"], *record["replicate_convergence"]]
        for r, (beta, fit) in enumerate(zip(coefficients, evidence)):
            weight = (base if r == 0 else replica_weights[r - 1])[positions]
            active = weight > 0
            matrix, y, weight = X[active], response[active], weight[active]
            relative = weight / weight.max()
            normalized = finite(relative * (len(weight) / relative.sum()), "Saved normalized fit weights")
            scales = matrix.abs().amax(0)
            if bool((normalized <= 0).any()) or bool((scales <= 0).any()):
                raise ValueError("Saved target source has degenerate positive-weight fit support.")
            scaled = matrix / scales
            singular = torch.linalg.svdvals(scaled * normalized.sqrt()[:, None])
            threshold = 10 * torch.finfo(FLOAT).eps * max(scaled.shape) * float(singular[0])
            if len(singular) != len(source.labels) or float(singular[-1]) <= threshold:
                raise ValueError("Saved target coefficients have lost full-rank weighted support.")
            coefficient = torch.tensor(beta, dtype=FLOAT) * scales
            objective, score, sensitivity, factor = _components(scaled, y, normalized, coefficient, source.family)
            direction = _solve(sensitivity, score)
            gradient = float(score.abs().max()) / len(y)
            gradient_scale = max(1., float((normalized * y.abs()).sum()) / len(y))
            tolerance = fit.get("tolerance")
            rounding = 64 * torch.finfo(FLOAT).eps * max(
                1., float((scaled.abs().T @ (normalized * factor.abs())).max()) / len(y))
            if (type(tolerance) not in (int, float) or tolerance != original_tolerance
                    or gradient > tolerance * gradient_scale + rounding
                    or float(direction.abs().max()) > math.sqrt(tolerance) * (1 + float(coefficient.abs().max()))
                    + 64 * torch.finfo(FLOAT).eps * (1 + float(coefficient.abs().max()))):
                raise ValueError("Saved coefficients do not solve the actual primitive weighted score equations.")
            saved_gradient, saved_objective = fit.get("score_max_abs"), fit.get("objective")
            # Poisson's y-mu can lose low bits when both terms are large.
            # Restoring raw beta and multiplying by the column scales can
            # change mu by an ulp even when the accepted fit is unchanged.
            # This bound only compares retained diagnostics; it does not
            # loosen the weighted-score or Newton-step acceptance above.
            replay_rounding = rounding
            cancellation = None
            if source.family == "poisson":
                cancellation = 64 * torch.finfo(FLOAT).eps * normalized * (2 * y.abs() + factor.abs())
                replay_rounding = max(rounding, float((scaled.abs().T @ cancellation).max()) / len(y))
            if (type(saved_gradient) not in (int, float) or type(saved_objective) not in (int, float)
                    or not math.isfinite(saved_gradient) or not math.isfinite(saved_objective)
                    or not math.isclose(gradient, saved_gradient, rel_tol=1e-9, abs_tol=replay_rounding)
                    or not math.isclose(objective, saved_objective, rel_tol=1e-11, abs_tol=1e-12)):
                raise ValueError("Saved target source convergence differs from its actual weighted likelihood replay.")
            if r == 0:
                bread = scales[:, None] * sensitivity * scales[None, :]
                scores = torch.zeros((source.design.validation.n_psu, len(source.labels)), dtype=FLOAT)
                scores.index_add_(0, group_ids[positions][active], matrix * (normalized * factor)[:, None])
                expected_bread = torch.tensor(source.metadata["sensitivity"], dtype=FLOAT)
                expected_scores = torch.tensor(source.metadata["psu_score_sums"], dtype=FLOAT)
                bread_scale = max(float(expected_bread.abs().max()), 1e-300)
                score_scale = max(float(expected_scores.abs().max()), 1e-300)
                scores_match = torch.allclose(scores, expected_scores, atol=score_scale * 1e-10 + rounding,
                                              rtol=1e-10)
                if cancellation is not None:
                    score_rounding = torch.zeros_like(scores)
                    score_rounding.index_add_(0, group_ids[positions][active], matrix.abs() * cancellation[:, None])
                    scores_match = bool(((scores - expected_scores).abs()
                                         <= expected_scores.abs() * 1e-10 + score_rounding + rounding).all())
                if (not torch.allclose(bread, expected_bread, atol=bread_scale * 1e-11, rtol=1e-11)
                        or not scores_match):
                    raise ValueError("Saved target primitives differ from the original sensitivity or complete PSU scores.")

    def _validate_weight_plan(self, base, weights, factors, groups):
        source = self.source_result
        record = source.metadata["replication"]
        strata = source.metadata["stratum_psu_indices"]
        stochastic = [h for h, correction in enumerate(source.metadata["fpc_multipliers"]) if correction]
        expected = torch.ones_like(factors)
        if record["method"] in {"brr", "fay"}:
            rho = record["fay_rho"]
            signs = record["balanced_signs"]
            for column, h in enumerate(stochastic):
                first, second = strata[h]
                for r, row in enumerate(signs):
                    expected[r, first] = 1 + (1 - rho) * row[column]
                    expected[r, second] = 1 - (1 - rho) * row[column]
            if not torch.allclose(factors, expected, atol=1e-12, rtol=1e-12):
                raise ValueError("Saved BRR/Fay target weights differ from paired-PSU selection signs.")
            generated = "Sylvester" in record["generation"]
        elif record["method"] == "jackknife":
            ids, r = [], 0
            for h in stochastic:
                members = strata[h]
                for deleted in members:
                    expected[r, members] = len(members) / (len(members) - 1)
                    expected[r, deleted] = 0.
                    ids.append(f"stratum-{h}-delete-psu-{deleted}")
                    r += 1
            if record["replicate_ids"] != ids or not torch.allclose(factors, expected, atol=1e-12, rtol=1e-12):
                raise ValueError("Saved jackknife target weights omit or replace a complete delete-one PSU plan.")
            generated = True
        else:
            return
        if generated and not torch.equal(weights, base * expected[:, groups]):
            raise ValueError("Saved generated target weights differ from their actual declared plan.")

    @property
    def df(self):
        return self.source_result.df

    @property
    def design(self):
        return self.source_result.design

    @property
    def family(self):
        return self.source_result.family

    @property
    def method(self):
        return self.source_result.metadata["replication"]["method"]

    def to_state(self):
        """Return a fresh, validated JSON-compatible state without shared mutable data."""
        return self.from_state(self.model_dump_json()).model_dump(mode="json")

    @classmethod
    def from_state(cls, state):
        """Restore from a dictionary or JSON without the original data or refitting."""
        if isinstance(state, cls):
            state = state.model_dump_json()
        if isinstance(state, str):
            return cls.model_validate_json(state)
        if not isinstance(state, dict):
            raise AnalysisError("invalid_survey_result", "Supply a saved empirical target dictionary or JSON.")
        cls.admit_raw_state(state)
        return cls.model_validate(deepcopy(state))

    @cpu_call
    def to_frame(self):
        """Replay target-width null tests and design-t intervals with complete metadata."""
        state = self.from_state(self.model_dump_json())
        critical = critical_value(state.alpha, state.df) if state.df else None
        rows = []
        for estimate, null, row in zip(state.estimates, state.null, state.covariance):
            index = len(rows)
            se = math.sqrt(max(row[index], 0.))
            statistic = (estimate - null) / se if se > 0 and state.df else None
            p = student_t_two_sided(statistic, state.df) if statistic is not None else None
            width = critical * se if critical is not None else 0.
            rows.append([estimate, se, statistic, p, estimate - width, estimate + width])
        frame = pd.DataFrame(rows, index=list(state.labels),
                             columns=["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"])
        frame.attrs.update(
            survey_replicate_margins_state=state.model_dump(mode="json"),
            covariance_matrix=[list(row) for row in state.covariance], target=state.target,
            method=f"single-stage {state.method} full empirical-target replication", design_df=state.df,
            alpha=state.alpha, uncertainty=_UNCERTAINTY,
            empirical_covariate_uncertainty=True, coefficient_uncertainty=True,
        )
        return frame


@cpu_call
def build_target_result(source_result, *, target, profiles, variables, labels, estimates,
                        replicate_estimates, primitive, alpha, null, workspace, work):
    """Construct the independent target envelope without modifying coefficient state."""
    source = SurveyRegressionResult.model_validate_json(source_result.model_dump_json())
    alpha = number(alpha, "alpha", 1e-8, 1 - 1e-8)
    labels = list(labels)
    null = ([number(value, "target null") for value in null]
            if isinstance(null, (list, tuple)) else [number(null, "target null")] * len(labels))
    if len(null) != len(labels):
        raise AnalysisError("invalid_survey_option", "One null per empirical target is required.")
    if isinstance(estimates, torch.Tensor):
        estimates = estimates.tolist()
    if isinstance(replicate_estimates, torch.Tensor):
        replicate_estimates = replicate_estimates.tolist()
    covariance = target_covariance(estimates, replicate_estimates, source.metadata["replication"])
    payload = dict(
        schema_version="survey-replicate-margins-result-v1", source_result=source.model_dump(mode="json"),
        target=target, profiles={} if profiles is None else deepcopy(profiles),
        variables=[] if variables is None else list(variables), labels=labels, estimates=estimates,
        replicate_estimates=replicate_estimates, covariance=covariance.tolist(), alpha=alpha, null=null,
        metadata={"primitive": deepcopy(primitive), "workspace": deepcopy(workspace), "work": deepcopy(work),
                  "uncertainty": _UNCERTAINTY, "variance_formula": _VARIANCE,
                  "empirical_covariate_uncertainty": True, "coefficient_uncertainty": True,
                  "precision": "float64", "device": "cpu", "dataset_support": False,
                  "stata_parity_validated": False},
    )
    return SurveyReplicateMarginsResult(**payload, integrity_sha256=digest(payload))
