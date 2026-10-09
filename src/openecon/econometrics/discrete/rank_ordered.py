"""Strict-prefix rank ordered logit with native analytic likelihood moments.

Positive ranks are consecutive, with one the most preferred alternative. Zero
means unranked below the observed prefix, rather than a positive-rank tie. The
likelihood removes each selected alternative from the subsequent denominator.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.resources import plan_workspace

DT = torch.float64
SCHEMA = "rank_ordered_v1"
NOTES = [
    "Rank 1 is best. Zero denotes an unranked bottom alternative; positive-rank ties are unsupported.",
    "Each successive denominator contains all available alternatives not selected earlier in the same case.",
    "HC0 aggregates all stage scores within each ranking case; CR0 additionally aggregates whole cases within the declared cluster. Neither has a degrees-of-freedom multiplier.",
    "Inference is asymptotic normal. No finite degrees of freedom or global full-rank robust Wald test is supplied.",
    "Availability is conditioned on as supplied; its selection mechanism and ranking stopping rule are not modeled.",
]


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < level < 1:
        _error("level must be a finite probability strictly between zero and one.", "invalid_option")
    z = float(torch.special.ndtri(torch.tensor((1 + float(level))/2, dtype=DT, device="cpu")))
    if not math.isfinite(z):
        _error("level is too close to one for float64 normal inference.", "invalid_option")
    return float(level), z


def _options(vce="oim", max_iterations=200, tolerance=1e-9, max_work=2_000_000_000):
    if not isinstance(vce, str) or vce not in {"oim", "hc0", "cr0"}:
        _error("vce must be exactly oim, hc0 or cr0.", "invalid_option")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or not 1 <= max_iterations <= 1000:
        _error("max_iterations must be an integer in [1,1000].", "invalid_option")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 1e-12 <= tolerance <= 1e-5:
        _error("tolerance must be finite and in [1e-12,1e-5].", "invalid_option")
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= 2_000_000_000:
        _error("max_work must be an integer in [1,2e9].", "invalid_resource_budget")
    return dict(vce=vce, max_iterations=int(max_iterations), tolerance=float(tolerance), max_work=int(max_work))


def _columns(rank, x, case, alternative, available=None, cluster=None):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 8:
        _error("x must list 1..8 distinct numeric attribute column names.", "invalid_spec")
    names = [rank, case, alternative, *x] + [v for v in (available, cluster) if v is not None]
    if any(not isinstance(v, str) or not v for v in names) or len(set(names)) != len(names):
        _error("Selected column names must be distinct nonempty strings.", "invalid_spec")
    return dict(rank=rank, x=list(x), case=case, alternative=alternative, available=available, cluster=cluster)


def _list(value, n, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad or value.ndim != 1 or value.is_complex():
            _error(f"{name} requires a real one-dimensional CPU vector without gradients.", "unsupported_input")
        values = value.tolist()
    elif isinstance(value, pd.Series) or type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", None) != 1:
            _error(f"{name} requires a one-dimensional vector.", "unsupported_input")
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        _error(f"{name} requires a resident positional vector.", "unsupported_input")
    if len(values) != n:
        _error("Selected input vectors must have equal lengths.", "shape_mismatch")
    return values


def _numeric(values, name):
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        _error(f"{name} requires real numeric cells without booleans or missing values.")
    if any(isinstance(v, Integral) and abs(v) > 1e12 for v in values):
        _error(f"{name} magnitudes exceed the supported 1e12 bound.")
    if any(not math.isfinite(float(v)) for v in values):
        _error(f"{name} contains missing or nonfinite cells; rows cannot be silently dropped.", "missing_values")
    if any(abs(v) > 1e12 for v in values):
        _error(f"{name} magnitudes exceed the supported 1e12 bound.")
    return [float(v) for v in values]


def _identity(value, name):
    if isinstance(value, str) and 1 <= len(value) <= 128:
        return value
    if isinstance(value, Integral) and not isinstance(value, bool) and abs(value) <= 2**53:
        return int(value)
    _error(f"{name} identifiers require nonempty strings of at most 128 characters or exact integers with magnitude at most 2^53; no coercion.", "invalid_identifier")


def _key(value):
    return (type(value).__name__, value)


def _plan(n, q, options, *, risk_cells=None, stages=None, budget_bytes=None):
    if type(n) is not int or not 2 <= n <= 4096 or not 1 <= q <= 8:
        _error("Rank ordered logit supports 2..4096 alternative rows and 1..8 attributes.", "resource_limit")
    # Before any model tensor is built, bound all possible successive risksets.
    e = 20*n if risk_cells is None else risk_cells
    s = n if stages is None else stages
    work = e*q*q*options["max_iterations"]
    if work > options["max_work"]:
        _error("The risk-set/attribute/iteration work exceeds max_work.", "resource_limit")
    return plan_workspace("rank_ordered_logit", {
        "inputs_and_centered_design": 8*n*(3*q + 8),
        "riskset_moments": 8*(e*(3*q + 5) + s*(3*q + 6) + 24*q*q),
        "complete_saved_scores": 8*(n*q + 512*q + 8*q*q),
    }, budget_bytes=budget_bytes).record() | {"estimated_iteration_cells": work, "max_iteration_cells": options["max_work"]}


@dataclass
class Prepared:
    X: torch.Tensor
    Z: torch.Tensor
    scale: torch.Tensor
    rank: list
    available: list
    case_labels: list
    alternative_labels: list
    case_rows: list
    row_cases: list
    stages: list
    cluster_labels: list
    case_cluster_codes: list
    columns: dict
    options: dict
    inputs: dict
    resources: dict
    risk_rows: torch.Tensor
    risk_stages: torch.Tensor
    selected_rows: torch.Tensor
    stage_cases: torch.Tensor

    @property
    def x(self):
        return self.X

    @property
    def n(self):
        return self.X.shape[0]

    @property
    def q(self):
        return self.X.shape[1]

    @property
    def case_weights(self):
        return [1.0]*len(self.case_labels)


def _prepare(data, columns, options, *, require_rank=True, for_fit=True, budget_bytes=None):
    names = [columns["case"], columns["alternative"], *columns["x"]]
    if require_rank:
        names.append(columns["rank"])
    names += [columns[k] for k in ("available", "cluster") if columns[k] is not None]
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            _error("Input column labels must be unique.", "invalid_spec")
        if any(name not in data.columns for name in names):
            _error("Every selected column must be present in data.", "invalid_spec")
        n = len(data)
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Every selected column must be present in data.", "invalid_spec")
        try:
            n = len(data[columns["case"]])
        except TypeError as error:
            raise AnalysisError("unsupported_input", "Data columns require resident positional vectors.") from error
    else:
        _error("Use a resident DataFrame or column mapping; Dataset and implicit collection are unsupported.", "unsupported_input")
    q = len(columns["x"])
    _plan(n, q, options, budget_bytes=budget_bytes)
    values = {name: _list(data[name], n, name) for name in names}
    cases = [_identity(v, "case") for v in values[columns["case"]]]
    alternatives = [_identity(v, "alternative") for v in values[columns["alternative"]]]
    rank = _numeric(values[columns["rank"]], "rank") if require_rank else [0.0]*n
    if any(v < 0 or v != int(v) for v in rank):
        _error("Ranks require nonnegative integers, with rank 1 best and zero unranked.", "invalid_rank")
    rank = [int(v) for v in rank]
    availability_values = values[columns["available"]] if columns["available"] else [1]*n
    # Availability is an explicit indicator, so booleans carry its exact
    # meaning. Numeric attributes and ranks deliberately remain nonboolean.
    available = _numeric([int(v) if isinstance(v, bool) else v for v in availability_values], "available")
    if any(v not in (0, 1) for v in available):
        _error("Availability requires exact numeric 0/1; no recoding.", "invalid_availability")
    available = [int(v) for v in available]
    xv = [_numeric(values[name], name) for name in columns["x"]]
    case_labels, case_rows, row_cases, case_map = [], [], [], {}
    seen = set()
    for i, (case, alt) in enumerate(zip(cases, alternatives)):
        pair = (_key(case), _key(alt))
        if pair in seen:
            _error("Each typed case/alternative pair must occur exactly once.", "duplicate_alternative")
        seen.add(pair)
        if _key(case) not in case_map:
            case_map[_key(case)] = len(case_labels)
            case_labels.append(case)
            case_rows.append([])
        g = case_map[_key(case)]
        case_rows[g].append(i)
        row_cases.append(g)
    if not (2 if for_fit else 1) <= len(case_labels) <= 512:
        _error("Rank ordered logit supports at most 512 cases; fitting requires at least two.", "resource_limit")
    stages = []
    for g, rows in enumerate(case_rows):
        remaining = [i for i in rows if available[i]]
        if not 2 <= len(remaining) <= 20:
            _error("Each case requires 2..20 available alternatives.", "invalid_availability")
        if any(rank[i] for i in rows if not available[i]):
            _error("An unavailable alternative cannot have a positive rank.", "invalid_rank")
        selected = sorted((i for i in remaining if rank[i]), key=lambda i: rank[i])
        if require_rank and (not selected or [rank[i] for i in selected] != list(range(1, len(selected)+1))):
            _error("Each case requires one strict contiguous ranked prefix 1..K, without ties or gaps.", "invalid_rank")
        for s, i in enumerate(selected, 1):
            stages.append(dict(case_index=g, stage=s, selected=i, riskset=list(remaining)))
            remaining.remove(i)
    cluster_labels, case_cluster_codes, clusters = [], [], None
    if columns["cluster"] is not None:
        clusters = [_identity(v, "cluster") for v in values[columns["cluster"]]]
        cmap = {}
        for rows in case_rows:
            key = _key(clusters[rows[0]])
            if any(_key(clusters[i]) != key for i in rows):
                _error("The declared cluster must be constant within every complete ranking case.", "invalid_cluster")
            if key not in cmap:
                cmap[key] = len(cluster_labels)
                cluster_labels.append(clusters[rows[0]])
            case_cluster_codes.append(cmap[key])
        if len(cluster_labels) < 2:
            _error("CR0 requires at least two declared clusters.", "insufficient_clusters")
    e = sum(len(s["riskset"]) for s in stages)
    resources = _plan(n, q, options, risk_cells=e, stages=len(stages), budget_bytes=budget_bytes)
    X = torch.tensor(list(zip(*xv)), dtype=DT, device="cpu")
    centered = torch.zeros_like(X, device="cpu")
    for rows in case_rows:
        indices = [i for i in rows if available[i]]
        centered[indices] = X[indices] - X[indices].mean(0)
    largest = centered.abs().amax(0)
    if for_fit and bool((largest == 0).any()):
        _error("Every attribute must vary within an available choice set; no common intercept or automatic omission.", "rank_deficient")
    safe_largest = torch.where(largest > 0, largest, torch.ones_like(largest, device="cpu"))
    scale = safe_largest*(centered/safe_largest).square().mean(0).sqrt()
    if not for_fit:
        scale = torch.where(scale > 0, scale, torch.ones_like(scale, device="cpu"))
    Z = centered/scale
    if not bool(torch.isfinite(Z).all()) or not bool((scale > 0).all()):
        _error("Attribute units cannot be represented safely after within-case scaling.", "numerical_failure")
    if for_fit:
        singular = torch.linalg.svdvals(Z)
        if float(singular[-1]) <= float(singular[0])*1e-10:
            _error("The within-case attribute design is rank deficient or ill-conditioned.", "rank_deficient")
    inputs = dict(rank=rank, x=X.tolist(), case=cases, alternative=alternatives, available=available, cluster=clusters)
    return Prepared(X, Z, scale, rank, available, case_labels, alternatives, case_rows, row_cases, stages,
                    cluster_labels, case_cluster_codes, columns, options, inputs, resources,
                    torch.tensor([i for s in stages for i in s["riskset"]], dtype=torch.int64, device="cpu"),
                    torch.tensor([j for j, s in enumerate(stages) for _ in s["riskset"]], dtype=torch.int64, device="cpu"),
                    torch.tensor([s["selected"] for s in stages], dtype=torch.int64, device="cpu"),
                    torch.tensor([s["case_index"] for s in stages], dtype=torch.int64, device="cpu"))


def _moments(theta, p):
    z = p.Z[p.risk_rows]
    eta = z@theta
    s = len(p.stages)
    maxima = torch.full((s,), -torch.inf, dtype=DT, device="cpu")
    maxima.scatter_reduce_(0, p.risk_stages, eta, reduce="amax", include_self=True)
    exponential = (eta-maxima[p.risk_stages]).exp()
    sums = torch.zeros(s, dtype=DT, device="cpu").index_add_(0, p.risk_stages, exponential)
    probability = exponential/sums[p.risk_stages]
    mean = torch.zeros((s, p.q), dtype=DT, device="cpu").index_add_(0, p.risk_stages, probability[:, None]*z)
    residual = z-mean[p.risk_stages]
    information = residual.T@(probability[:, None]*residual)
    # Direct pairwise utility-attribute differences preserve the score when
    # the selected probability rounds to one. selected-E[X] would falsely
    # become zero under separation while the losing probability remains >0.
    differences = p.Z[p.selected_rows][p.risk_stages]-z
    scores = torch.zeros((s, p.q), dtype=DT, device="cpu").index_add_(0, p.risk_stages, probability[:, None]*differences)
    log_probability = eta-maxima[p.risk_stages]-sums[p.risk_stages].log()
    selected_cells = p.risk_rows == p.selected_rows[p.risk_stages]
    stage_ll = log_probability[selected_cells]
    if any(not bool(torch.isfinite(v).all()) for v in (information, scores, log_probability, stage_ll, probability)):
        _error("The rank likelihood moments are not finite.", "numerical_failure")
    return dict(information=(information+information.T)/2, stage_scores=scores,
                stage_loglikelihood=stage_ll, log_probabilities=log_probability,
                probabilities=probability, gradient=scores.sum(0), log_likelihood=float(stage_ll.sum()))


def _inverse(information):
    diagonal = information.diag()
    if not bool((diagonal > 0).all()):
        _error("The unregularized likelihood information is not positive definite.", "no_finite_mle")
    sd = diagonal.sqrt()
    normalized = information/sd[:, None]/sd[None, :]
    eigen = torch.linalg.eigvalsh(normalized)
    if float(eigen[0]) <= 1e-10*float(eigen[-1]):
        _error("The unregularized likelihood information is unidentified or ill-conditioned.", "rank_deficient")
    cholesky, status = torch.linalg.cholesky_ex(normalized)
    if int(status) != 0:
        _error("The unregularized observed information cannot be inverted.", "no_finite_mle")
    inverse = torch.cholesky_inverse(cholesky)/sd[:, None]/sd[None, :]
    return (inverse+inverse.T)/2


def _stationarity(theta, moments, tolerance, *, code="no_finite_mle"):
    bread = _inverse(moments["information"])
    step = bread@moments["gradient"]
    decrement = float(moments["gradient"]@step)
    # theta already uses fixed within-case attribute scales. Dividing again
    # by a diverging theta could turn a boundary sequence into convergence.
    normalized_step = float(step.abs().max())
    step_limit = max(20*tolerance, 1e-10)
    decrement_limit = max(100*tolerance*tolerance, 1e-16)*max(1, len(moments["stage_loglikelihood"]))
    if not math.isfinite(decrement) or not math.isfinite(normalized_step):
        _error("Likelihood stationarity diagnostics are not finite.", code)
    diagnostic = dict(score_decrement=max(0.0, decrement), normalized_parameter_step=normalized_step,
                      parameter_step=float(step.abs().max()), step_limit=step_limit,
                      decrement_limit=decrement_limit)
    return diagnostic, step, bread, normalized_step <= step_limit and decrement <= decrement_limit


def _fit(p):
    theta = torch.zeros(p.q, dtype=DT, device="cpu")
    backtracks = 0
    for iteration in range(1, p.options["max_iterations"]+1):
        moments = _moments(theta, p)
        diagnostic, step, _, stationary = _stationarity(theta, moments, p.options["tolerance"])
        if stationary:
            return theta, dict(iterations=iteration, backtracks=backtracks, converged=True, **diagnostic)
        factor = 1.0
        directional = float(moments["gradient"]@step)
        for _ in range(50):
            trial = theta + factor*step
            candidate = _moments(trial, p)
            roundoff = 8*torch.finfo(DT).eps*max(1, abs(moments["log_likelihood"]))
            if candidate["log_likelihood"] >= moments["log_likelihood"] + 1e-4*factor*directional-roundoff:
                theta = trial
                break
            factor /= 2
            backtracks += 1
        else:
            _error("The rank likelihood could not take a finite improving Newton step.", "nonconvergence")
    _error("Rank ordered ML did not reach the finite interior score and Newton-step gates within max_iterations.", "nonconvergence")


def _evaluate(beta, p):
    theta = beta*p.scale
    moments = _moments(theta, p)
    # Reconstruct raw utilities only as a representation check. Common case
    # levels cancel mathematically, but their float64 cancellation can be unsafe.
    for rows in p.case_rows:
        indices = [i for i in rows if p.available[i]]
        raw = (p.X[indices]-p.X[indices[0]])@beta
        expected = p.Z[indices]@theta
        difference = raw
        target = expected-expected[0]
        bound = 2e-9*max(1.0, float(target.abs().max()))
        if not bool(torch.isfinite(raw).all()) or float((difference-target).abs().max()) > bound:
            _error("Original-unit coefficients cannot reproduce the fitted within-case utility differences safely.", "numerical_failure")
    diagnostic, _, inverse, stationary = _stationarity(theta, moments, p.options["tolerance"])
    if not stationary:
        _error("Saved/final coefficients fail the actual score-decrement and normalized Newton-step gates.", "no_finite_mle")
    stage_scores = moments["stage_scores"]*p.scale
    case_scores = torch.zeros((len(p.case_labels), p.q), dtype=DT, device="cpu").index_add_(0, p.stage_cases, stage_scores)
    case_ll = torch.zeros(len(p.case_labels), dtype=DT, device="cpu").index_add_(0, p.stage_cases, moments["stage_loglikelihood"])
    case_code = torch.tensor(p.case_cluster_codes, dtype=torch.int64, device="cpu")
    cluster_scores = (torch.zeros((len(p.cluster_labels), p.q), dtype=DT, device="cpu").index_add_(0, case_code, case_scores)
                      if p.cluster_labels else torch.empty((0, p.q), dtype=DT, device="cpu"))
    information = moments["information"]*p.scale[:, None]*p.scale[None, :]
    bread = inverse/p.scale[:, None]/p.scale[None, :]
    scores = cluster_scores if p.options["vce"] == "cr0" else case_scores
    meat = scores.T@scores
    influence = scores@bread
    covariance = bread if p.options["vce"] == "oim" else influence.T@influence
    covariance = (covariance+covariance.T)/2
    probability_jacobian = []
    offset = 0
    for stage in p.stages:
        indices = stage["riskset"]
        probability = moments["probabilities"][offset:offset+len(indices)]
        # Raw centered X has stable moments and the exact same beta derivative.
        design = p.Z[indices]*p.scale
        difference = design[:, None, :]-design[None, :, :]
        centered_derivative = (difference*probability[None, :, None]).sum(1)
        probability_jacobian.extend((probability[:, None]*centered_derivative).tolist())
        offset += len(indices)
    result = dict(beta=beta.tolist(), information=information.tolist(), bread=bread.tolist(), meat=meat.tolist(), covariance=covariance.tolist(),
                  case_scores=case_scores.tolist(), cluster_scores=cluster_scores.tolist(), stage_scores=stage_scores.tolist(),
                  stage_loglikelihood=moments["stage_loglikelihood"].tolist(), case_loglikelihood=case_ll.tolist(),
                  probabilities=moments["probabilities"].tolist(), log_probabilities=moments["log_probabilities"].tolist(),
                  probability_jacobian=probability_jacobian, log_likelihood=moments["log_likelihood"],
                  parameter_names=list(p.columns["x"]), scale=p.scale.tolist(), stationarity=diagnostic,
                  covariance_type=p.options["vce"], n=p.n, n_cases=len(p.case_labels), n_clusters=len(p.cluster_labels),
                  n_stages=len(p.stages), n_risk_cells=len(p.risk_rows))
    if not all(_finite_matrix(result[key], p.q, p.q) for key in ("information", "bread", "meat", "covariance")):
        _error("Original-unit full inference matrices exceed finite float64 support.", "numerical_failure")
    _psd(covariance, "covariance", code="numerical_failure")
    return result


def _checksum(state):
    return hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "checksum"}, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _marker(base, names):
    while base in names:
        base = "_"+base
    return base


def _settings(p, level):
    return dict(columns=p.columns, **p.options, level=level, device="cpu", precision="float64", missing="raise", weights=None,
                n_input=p.n, n_retained=p.n, n_dropped=0, n_cases=len(p.case_labels), n_clusters=len(p.cluster_labels),
                rank_direction="1 is most preferred", rank_zero="unranked below observed strict prefix",
                stopping_rule="condition on the observed ranked prefix", availability="condition on declared available choice sets",
                covariance="full OIM, case HC0 or cluster CR0, without df multiplier", inference_df=None,
                parameter_null=0, parameter_test="asymptotic normal two-sided zero-null coefficient test",
                global_wald=None, robust_covariance_may_be_singular=True,
                max_rows=4096, max_cases=512, max_available_alternatives=20, max_attributes=8,
                numeric_magnitude_bound=1e12, complete_inputs_saved=True, complete_covariance_saved=True,
                restoration="validate complete portable state and actual stationarity; no optimizer or refit",
                resources=p.resources, excluded_scope=["weights", "positive-rank ties", "implicit category encoding", "Dataset", "missing-row deletion", "automatic attribute omission", "random coefficients", "endogenous availability", "modeled stopping rule"])


def _matrix(value, names):
    return table([[name, *row] for name, row in zip(names, value)], columns=[_marker("parameter", names), *names])


def _parameter_table(fit, level):
    _, zcrit = _confidence(level)
    rows = []
    for j, (name, beta) in enumerate(zip(fit["parameter_names"], fit["beta"])):
        variance = fit["covariance"][j][j]
        if variance < 0:
            _error("A coefficient variance is negative.", "invalid_covariance")
        se = math.sqrt(variance)
        z = beta/se if se else None
        limits = [beta-zcrit*se, beta+zcrit*se] if se else [None, None]
        if any(v is not None and not math.isfinite(v) for v in limits):
            _error("Coefficient confidence limits exceed finite float64 support.", "numerical_failure")
        rows.append([name, beta, se, z, math.erfc(abs(z)/math.sqrt(2)) if z is not None else None, *limits,
                     "available" if se else "unavailable: zero first-order variance"])
    return table(rows, columns=["parameter", "estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper", "inference_status"])


def _assemble(state, p=None):
    fit, columns, inputs = state["fit"], state["columns"], state["inputs"]
    names = fit["parameter_names"]
    if p is None:
        p = _prepared_from_state(state)
    input_names = [columns["case"], columns["alternative"], columns["rank"], *names]
    if columns["available"]:
        input_names.append(columns["available"])
    if columns["cluster"]:
        input_names.append(columns["cluster"])
    rows = []
    for i in range(p.n):
        row = [i, inputs["case"][i], inputs["alternative"][i], inputs["rank"][i], *inputs["x"][i]]
        if columns["available"]:
            row.append(inputs["available"][i])
        if columns["cluster"]:
            row.append(inputs["cluster"][i])
        rows.append(row)
    stage_rows, probability_rows = [], []
    offset = 0
    for j, stage in enumerate(p.stages):
        g, selected = stage["case_index"], stage["selected"]
        stage_rows.append([p.case_labels[g], stage["stage"], selected, p.alternative_labels[selected], len(stage["riskset"]), fit["stage_loglikelihood"][j]])
        for i in stage["riskset"]:
            probability_rows.append([p.case_labels[g], stage["stage"], i, p.alternative_labels[i], int(i == selected), fit["probabilities"][offset], fit["log_probabilities"][offset]])
            offset += 1
    matrices = {key: _matrix(fit[key], names) for key in ("information", "bread", "meat", "covariance")}
    metadata = {key: value for key, value in fit.items() if key not in {
        "beta", "information", "bread", "meat", "covariance", "case_scores", "cluster_scores", "stage_scores",
        "stage_loglikelihood", "case_loglikelihood", "probabilities", "log_probabilities", "probability_jacobian"}}
    frames = dict(inputs=table(rows, columns=[_marker("row", input_names), *input_names]), parameters=_parameter_table(fit, state["level"]),
                  **matrices,
                  case_scores=table([[label, *score] for label, score in zip(p.case_labels, fit["case_scores"])], columns=[_marker("case", names), *names]),
                  cluster_scores=table([[label, *score] for label, score in zip(p.cluster_labels, fit["cluster_scores"])], columns=[_marker("cluster", names), *names]),
                  stage_scores=table([[p.case_labels[s["case_index"]], s["stage"], *score] for s, score in zip(p.stages, fit["stage_scores"])], columns=[_marker("case", names), _marker("stage", names), *names]),
                  stages=table(stage_rows, columns=["case", "stage", "selected_row", "selected_alternative", "riskset_size", "log_likelihood"]),
                  case_likelihood=table([[label, ll, math.exp(ll)] for label, ll in zip(p.case_labels, fit["case_loglikelihood"])], columns=["case", "log_likelihood", "ranking_probability"]),
                  probabilities=table(probability_rows, columns=["case", "stage", "row", "alternative", "selected", "probability", "log_probability"]),
                  probability_jacobian=table([[j, *row] for j, row in enumerate(fit["probability_jacobian"])], columns=[_marker("risk_cell", names), *names]),
                  fit_summary=table([[key, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)] for key, value in sorted(metadata.items())], columns=["setting", "json"]))
    return TableSet(frames, title="Strict rank ordered logit", method="rologit", contract=SCHEMA,
                    settings=state["settings"], rank_ordered_state=state, notes=NOTES)


@procedure
def rologit(data, rank, x, *, case, alternative, available=None, vce="oim", cluster=None,
            level=.95, device="cpu", weights=None, max_iterations=200, tolerance=1e-9, max_work=2_000_000_000):
    """Fit strict full or partial ranked-prefix logit on explicit available sets."""
    if device != "cpu":
        _error("Rank ordered logit supports explicit native CPU float64 only.", "unsupported_device")
    if weights is not None:
        _error("Weights are outside the strict ranking-case covariance contract.", "unsupported_weights")
    options = _options(vce, max_iterations, tolerance, max_work)
    if (vce == "cr0") != (cluster is not None):
        _error("Declare cluster exactly when vce='cr0'; clusters cannot be silently ignored.", "invalid_cluster")
    level, _ = _confidence(level)
    columns = _columns(rank, x, case, alternative, available, cluster)
    p = _prepare(data, columns, options)
    theta, convergence = _fit(p)
    fit = _evaluate(theta/p.scale, p)
    fit["convergence"] = convergence
    fit["solver"] = "native analytic centered/scaled Newton ML; unregularized information; backtracking"
    state = dict(schema=SCHEMA, level=level, columns=columns, options=options, inputs=p.inputs,
                 fit=fit, settings=_settings(p, level))
    state["checksum"] = _checksum(state)
    return _assemble(state, p)


def _finite_vector(value, count):
    return isinstance(value, list) and len(value) == count and all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v)) for v in value)


def _finite_matrix(value, rows, cols):
    return isinstance(value, list) and len(value) == rows and all(_finite_vector(row, cols) for row in value)


def _psd(value, name, *, code="invalid_covariance"):
    """Check covariance in its dimensionless correlation coordinates."""
    diagonal = value.diag()
    if not bool(torch.isfinite(value).all()) or bool((diagonal < 0).any()):
        _error(f"The full {name} has nonfinite cells or negative variances.", code)
    positive = diagonal > 0
    if bool((value[~positive] != 0).any()) or bool((value[:, ~positive] != 0).any()):
        _error(f"The full {name} has nonzero covariance for a zero-variance coordinate.", code)
    if bool(positive.any()):
        selected = value[positive][:, positive]
        sd = diagonal[positive].sqrt()
        correlation = selected/sd[:, None]/sd[None, :]
        if float((correlation-correlation.T).abs().max()) > 1e-11:
            _error(f"The full {name} is not symmetric in standardized coordinates.", code)
        eigen = torch.linalg.eigvalsh((correlation+correlation.T)/2)
        if float(eigen[0]) < -1e-10*max(1.0, float(eigen[-1])):
            _error(f"The full {name} is indefinite in standardized coordinates.", code)


def _matches(saved, expected, name):
    actual = torch.tensor(expected, dtype=DT, device="cpu")
    value = torch.tensor(saved, dtype=DT, device="cpu")
    if actual.shape != value.shape or not bool(torch.isfinite(value).all()):
        _error(f"Saved {name} has incomplete or nonfinite dimensions.", "invalid_result")
    if name in {"information", "bread", "meat", "covariance"}:
        scale = actual.diag().abs().sqrt()
        allowed = 2e-9*scale[:, None]*scale[None, :]
        valid = bool(((value-actual).abs() <= allowed).all()) and bool((value[actual == 0] == 0).all())
    else:
        valid = torch.allclose(value, actual, rtol=2e-9, atol=0)
    if not valid:
        _error(f"Saved {name} disagrees with complete scientific input/parameter replay.", "invalid_result")


def _prepared_from_state(state):
    columns, inputs = state["columns"], state["inputs"]
    data = {columns["rank"]: inputs["rank"], columns["case"]: inputs["case"], columns["alternative"]: inputs["alternative"],
            **{name: [row[j] for row in inputs["x"]] for j, name in enumerate(columns["x"])}}
    if columns["available"]:
        data[columns["available"]] = inputs["available"]
    if columns["cluster"]:
        data[columns["cluster"]] = inputs["cluster"]
    return _prepare(data, columns, state["options"])


def _validate_state(result):
    """Validate complete saved scientific state, without fitting or optimization."""
    attrs = result.attrs if isinstance(result, TableSet) else result
    if not isinstance(attrs, Mapping) or set(attrs) != {"method", "contract", "settings", "rank_ordered_state", "notes"}:
        _error("Supply the complete rank ordered fit TableSet or portable fit attrs.", "invalid_result")
    state = attrs["rank_ordered_state"]
    if not isinstance(state, dict) or set(state) != {"schema", "level", "columns", "options", "inputs", "fit", "settings", "checksum"} or state["schema"] != SCHEMA:
        _error("Unsupported rank ordered saved schema or fields.", "invalid_result")
    if not isinstance(state["columns"], dict) or set(state["columns"]) != {"rank", "x", "case", "alternative", "available", "cluster"}:
        _error("Saved column schema is incomplete.", "invalid_result")
    columns = _columns(**state["columns"])
    if not isinstance(state["options"], dict) or set(state["options"]) != {"vce", "max_iterations", "tolerance", "max_work"}:
        _error("Saved options schema is incomplete.", "invalid_result")
    options = _options(**state["options"])
    if (options["vce"] == "cr0") != (columns["cluster"] is not None):
        _error("Saved cluster and covariance declarations disagree.", "invalid_result")
    inputs = state["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"rank", "x", "case", "alternative", "available", "cluster"} or not isinstance(inputs["rank"], list):
        _error("Saved complete input schema is unsupported.", "invalid_result")
    n, q = len(inputs["rank"]), len(columns["x"])
    _plan(n, q, options)
    if not _finite_matrix(inputs["x"], n, q) or any(not isinstance(inputs[key], list) or len(inputs[key]) != n for key in ("case", "alternative", "available")):
        _error("Saved inputs have incomplete finite dimensions.", "invalid_result")
    if columns["cluster"] is None and inputs["cluster"] is not None:
        _error("Saved unclustered inputs contain undeclared cluster metadata.", "invalid_result")
    p = _prepared_from_state(state)
    if json.dumps(inputs, sort_keys=True, ensure_ascii=False, allow_nan=False) != json.dumps(p.inputs, sort_keys=True, ensure_ascii=False, allow_nan=False):
        _error("Saved inputs disagree with the complete canonical retained sample; undeclared availability cannot change its meaning.", "invalid_result")
    fit = state["fit"]
    fields = {"beta", "information", "bread", "meat", "covariance", "case_scores", "cluster_scores", "stage_scores", "stage_loglikelihood", "case_loglikelihood", "probabilities", "log_probabilities", "probability_jacobian", "log_likelihood", "parameter_names", "scale", "stationarity", "covariance_type", "n", "n_cases", "n_clusters", "n_stages", "n_risk_cells", "convergence", "solver"}
    if not isinstance(fit, dict) or set(fit) != fields or not _finite_vector(fit["beta"], q):
        _error("Saved complete fit schema is unsupported.", "invalid_result")
    shapes = {"information": (q, q), "bread": (q, q), "meat": (q, q), "covariance": (q, q),
              "case_scores": (len(p.case_labels), q), "cluster_scores": (len(p.cluster_labels), q), "stage_scores": (len(p.stages), q), "probability_jacobian": (len(p.risk_rows), q)}
    if any(not _finite_matrix(fit[name], *shape) for name, shape in shapes.items()):
        _error("Saved fit matrices require complete finite dimensions.", "invalid_result")
    for name in ("information", "bread", "meat", "covariance"):
        _psd(torch.tensor(fit[name], dtype=DT, device="cpu"), name)
    for name, count in (("stage_loglikelihood", len(p.stages)), ("case_loglikelihood", len(p.case_labels)), ("probabilities", len(p.risk_rows)), ("log_probabilities", len(p.risk_rows)), ("scale", q)):
        if not _finite_vector(fit[name], count):
            _error("Saved fit vectors require complete finite dimensions.", "invalid_result")
    try:
        if not isinstance(state["checksum"], str) or len(state["checksum"]) != 64 or not hmac.compare_digest(state["checksum"], _checksum(state)):
            _error("Saved rank ordered checksum does not match.", "invalid_result")
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved rank ordered state is not finite canonical JSON.") from error
    replay = _evaluate(torch.tensor(fit["beta"], dtype=DT, device="cpu"), p)
    for name, expected in replay.items():
        if name in shapes or name in {"stage_loglikelihood", "case_loglikelihood", "probabilities", "log_probabilities", "scale", "log_likelihood"}:
            _matches(fit[name], expected, name)
        elif fit[name] != expected or type(fit[name]) is not type(expected):
            _error(f"Saved {name} metadata disagrees with complete replay.", "invalid_result")
    diagnostic = fit["convergence"]
    keys = {"iterations", "backtracks", "converged", "score_decrement", "normalized_parameter_step", "parameter_step", "step_limit", "decrement_limit"}
    if not isinstance(diagnostic, dict) or set(diagnostic) != keys or diagnostic["converged"] is not True or type(diagnostic["iterations"]) is not int or not 1 <= diagnostic["iterations"] <= options["max_iterations"] or type(diagnostic["backtracks"]) is not int or not 0 <= diagnostic["backtracks"] <= 50*options["max_iterations"]:
        _error("Saved optimizer convergence metadata is invalid.", "invalid_result")
    for key, actual in fit["stationarity"].items():
        value = diagnostic[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0:
            _error("Saved convergence diagnostics require finite nonnegative values.", "invalid_result")
        # Optimizer and final-unit replay can differ by coefficient rounding.
        if key.endswith("limit") and value != actual:
            _error("Saved convergence thresholds disagree with declared tolerance.", "invalid_result")
    if diagnostic["normalized_parameter_step"] > diagnostic["step_limit"] or diagnostic["score_decrement"] > diagnostic["decrement_limit"] or fit["solver"] != "native analytic centered/scaled Newton ML; unregularized information; backtracking":
        _error("Saved optimizer diagnostics do not certify finite interior ML.", "invalid_result")
    level, _ = _confidence(state["level"])
    resources = state.get("settings", {}).get("resources") if isinstance(state["settings"], dict) else None
    if not isinstance(resources, dict) or type(resources.get("budget_bytes")) is not int:
        _error("Saved resource metadata is incomplete.", "invalid_result")
    plan = _plan(n, q, options, risk_cells=len(p.risk_rows), stages=len(p.stages), budget_bytes=resources["budget_bytes"])
    p.resources = plan
    if state["settings"] != _settings(p, level):
        _error("Saved settings disagree with the complete bounded contract.", "invalid_result")
    expected = _assemble(state, p)
    if dict(attrs) != expected.attrs:
        _error("Saved fit attrs disagree with the complete sealed state.", "invalid_result")
    if isinstance(result, TableSet) and (set(result) != set(expected) or any(not isinstance(result[key], pd.DataFrame) or not result[key].equals(expected[key]) for key in expected)):
        _error("Saved fit tables disagree with the complete sealed state.", "invalid_result")
    return state, p


@procedure
def rologit_restore(result, *, level=None):
    """Restore all fitted tables and full covariance without a refit."""
    state, p = _validate_state(result)
    if level is None or level == state["level"]:
        return _assemble(state, p)
    level, _ = _confidence(level)
    changed = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    changed["level"] = level
    changed["settings"]["level"] = level
    changed["checksum"] = _checksum(changed)
    return _assemble(changed, p)
