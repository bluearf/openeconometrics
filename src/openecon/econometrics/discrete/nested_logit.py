"""Bounded two-level disjoint-nest RUM logit with joint native ML.

The root utility scale is one. Within-nest utility is divided by its nest's
dissimilarity; all coefficients and free dissimilarities are fitted jointly.
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
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.engines.optimize import maximize_bfgs
from openecon.resources import plan_workspace

DT = torch.float64
SCHEMA = "nested_logit_v1"
MIN_LAMBDA = 1e-3
STARTS = (0.25, 0.5, 0.75)
SOLVER = "native Torch exact-chain-rule joint multistart BFGS; physical stationary interior ML; unregularized joint information"
NOTES = [
    "Two-level disjoint-nest random-utility logit; normalized root scale one and shared numeric utility coefficients.",
    "Free dissimilarities must lie strictly inside [0.001,0.999]; computational boundary solutions are rejected. Fixed dissimilarities may be in [0.001,1]. Global singleton nests have fixed dissimilarity one.",
    "HC0 uses each complete choice-case joint score. CR0 sums complete choice cases within the declared respondent cluster. Neither has a degrees-of-freedom multiplier.",
    "Inference is asymptotic normal. No boundary LR test, finite degrees of freedom, or global full-rank robust Wald test is supplied.",
    "Three common-dissimilarity starts are compared. This certifies a finite stationary local maximum, not a proof of the global maximum.",
    "Availability is conditioned on as supplied; its selection mechanism is not modeled. Sealing detects corruption, not an authenticated signature.",
]


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not math.isfinite(float(level)) or not 0 < level < 1:
        _error("level must be a finite probability strictly between zero and one.", "invalid_option")
    z = float(torch.special.ndtri(torch.tensor((1 + float(level))/2, dtype=DT, device="cpu")))
    if not math.isfinite(z):
        _error("level is too close to one for float64 normal inference.", "invalid_option")
    return float(level), z


def _options(vce="oim", max_iterations=300, tolerance=1e-9, max_work=2_000_000_000):
    if not isinstance(vce, str) or vce not in {"oim", "hc0", "cr0"}:
        _error("vce must be exactly oim, hc0 or cr0.", "invalid_option")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or not 1 <= max_iterations <= 1000:
        _error("max_iterations must be an integer in [1,1000].", "invalid_option")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 1e-12 <= tolerance <= 1e-5:
        _error("tolerance must be finite and in [1e-12,1e-5].", "invalid_option")
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or not 1 <= max_work <= 2_000_000_000:
        _error("max_work must be an integer in [1,2e9].", "invalid_resource_budget")
    return dict(vce=vce, max_iterations=int(max_iterations), tolerance=float(tolerance), max_work=int(max_work))


def _columns(chosen, x, case, alternative, nest, available=None, cluster=None):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 8:
        _error("x must list 1..8 distinct numeric attribute column names.", "invalid_spec")
    names = [chosen, case, alternative, nest, *x] + [v for v in (available, cluster) if v is not None]
    if any(not isinstance(v, str) or not v for v in names) or len(set(names)) != len(names):
        _error("Selected column names must be distinct nonempty strings.", "invalid_spec")
    return dict(chosen=chosen, x=list(x), case=case, alternative=alternative, nest=nest, available=available, cluster=cluster)


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
    if any(not math.isfinite(float(v)) for v in values):
        _error(f"{name} contains missing or nonfinite cells; rows cannot be silently dropped.", "missing_values")
    if any(abs(v) > 1e12 for v in values):
        _error(f"{name} magnitudes exceed the supported 1e12 bound.")
    return [float(v) for v in values]


def _indicator(values, name):
    result = _numeric([int(v) if isinstance(v, bool) else v for v in values], name)
    if any(v not in (0.0, 1.0) for v in result):
        _error(f"{name} requires booleans or exact zero/one indicators.")
    return [int(v) for v in result]


def _identity(value, name):
    if isinstance(value, str) and 1 <= len(value) <= 128:
        return value
    if isinstance(value, Integral) and not isinstance(value, bool) and abs(value) <= 2**53:
        return int(value)
    _error(f"{name} requires nonempty strings of at most 128 characters or exact integers with magnitude at most 2^53; no coercion.", "invalid_identifier")


def _key(value):
    return (type(value).__name__, value)


def _fixed(values):
    if values is None:
        return []
    pairs = list(values.items()) if isinstance(values, Mapping) else values
    if not isinstance(pairs, (list, tuple)):
        _error("fixed_dissimilarity requires a mapping or typed (nest,value) pairs.", "invalid_option")
    result, seen = [], set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            _error("Every fixed dissimilarity declaration must contain a nest and value.", "invalid_option")
        label = _identity(pair[0], "fixed nest")
        value = pair[1]
        if _key(label) in seen or isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or not MIN_LAMBDA <= value <= 1:
            _error("Fixed dissimilarities require unique typed nests and finite values in [0.001,1].", "invalid_option")
        result.append([label, float(value)])
        seen.add(_key(label))
    return result


def _plan(n, q, options, *, free=0, nests=1, cases=1, active_groups=None, budget_bytes=None, for_fit=True):
    if type(n) is not int or not 2 <= n <= 4096 or not 1 <= q <= 8:
        _error("Nested logit supports 2..4096 rows and 1..8 numeric attributes.", "resource_limit")
    if not 1 <= nests <= 8 or not 1 <= cases <= 512:
        _error("Nested logit supports at most eight nests and 512 choice cases.", "resource_limit")
    k = q + free
    starts = len(STARTS) if free else 1
    work = n*k*k*options["max_iterations"]*starts*8 if for_fit else n*k*k*8
    if work > options["max_work"]:
        _error("Joint derivative/iteration work exceeds max_work before any derivative tape is allocated.", "resource_limit")
    groups = cases*nests if active_groups is None else active_groups
    plan = plan_workspace("two_level_nested_logit", {
        "complete_inputs_and_centered_design": 8*n*(4*q + 16),
        "packed_choice_geometry": 8*(groups*20 + cases*8 + 8*n),
        "exact_derivative_tapes": 8*(n*k*k*16 + groups*20*k*8),
        "complete_joint_scores_and_information": 8*(cases*k*4 + 24*k*k),
    }, budget_bytes=budget_bytes).record()
    return plan | {"estimated_iteration_cells": work, "max_iteration_cells": options["max_work"]}


@dataclass
class Prepared:
    X: torch.Tensor
    Z: torch.Tensor
    scale: torch.Tensor
    chosen: list
    available: list
    case_labels: list
    case_rows: list
    alternative_labels: list
    nest_labels: list
    nest_codes: list
    free_nests: list
    fixed_lambdas: dict
    parameters: list
    cluster_labels: list
    case_cluster_codes: list
    columns: dict
    options: dict
    catalogue: dict
    inputs: dict
    resources: dict
    available_rows: list
    group_rows: list
    group_nests: list
    group_cases: list
    case_groups: list
    selected_rows: list
    selected_groups: list
    group_indices: torch.Tensor
    group_mask: torch.Tensor
    group_nest_tensor: torch.Tensor
    case_group_indices: torch.Tensor
    case_group_mask: torch.Tensor

    @property
    def n(self):
        return len(self.available)

    @property
    def q(self):
        return len(self.columns["x"])


def _prepare(data, columns, options, *, fixed_dissimilarity=None, catalogue=None,
             require_chosen=True, for_fit=True, budget_bytes=None):
    names = [columns["case"], columns["alternative"], columns["nest"], *columns["x"]]
    if require_chosen:
        names.append(columns["chosen"])
    names += [columns[k] for k in ("available", "cluster") if columns[k] is not None and (k != "cluster" or for_fit)]
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique or any(name not in data.columns for name in names):
            _error("Input columns must be unique and every selected column present.", "invalid_spec")
        n = len(data)
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Every selected column must be present in data.", "invalid_spec")
        try:
            n = len(data[columns["case"]])
        except TypeError as error:
            raise AnalysisError("unsupported_input", "Columns require resident positional vectors.") from error
    else:
        _error("Use a resident DataFrame or column mapping; Dataset is unsupported.", "unsupported_input")
    q = len(columns["x"])
    _plan(n, q, options, budget_bytes=budget_bytes, for_fit=False)
    values = {name: _list(data[name], n, name) for name in names}
    cases = [_identity(v, "case") for v in values[columns["case"]]]
    alternatives = [_identity(v, "alternative") for v in values[columns["alternative"]]]
    nests = [_identity(v, "nest") for v in values[columns["nest"]]]
    chosen = _indicator(values[columns["chosen"]], "chosen") if require_chosen else [0]*n
    available = _indicator(values[columns["available"]], "available") if columns["available"] else [1]*n
    raw_x = [_numeric(values[name], name) for name in columns["x"]]
    clusters = ([_identity(v, "cluster") for v in values[columns["cluster"]]]
                if columns["cluster"] is not None and for_fit else None)
    declarations = _fixed(fixed_dissimilarity)
    case_labels, case_rows, case_map = [], [], {}
    for i, label in enumerate(cases):
        key = _key(label)
        if key not in case_map:
            case_map[key] = len(case_labels)
            case_labels.append(label)
            case_rows.append([])
        case_rows[case_map[key]].append(i)
    if len(case_labels) > 512 or (for_fit and len(case_labels) < 2):
        _error("Fit requires 2..512 complete choice cases; queries require 1..512.", "resource_limit")
    observed_nests, observed_alternatives, alternative_map = [], [], {}
    nest_map = {}
    for alternative, nest in zip(alternatives, nests):
        key, nk = _key(alternative), _key(nest)
        if nk not in nest_map:
            nest_map[nk] = len(observed_nests)
            observed_nests.append(nest)
        if key in alternative_map and _key(alternative_map[key]) != nk:
            _error("A globally known alternative cannot change its disjoint nest.", "invalid_nesting")
        if key not in alternative_map:
            alternative_map[key] = nest
            observed_alternatives.append([alternative, nest])
    if len(observed_nests) > 8:
        _error("At most eight globally declared nests are supported.", "resource_limit")
    if catalogue is None:
        nest_labels = observed_nests
        fixed_map = {_key(label): value for label, value in declarations}
        if any(key not in nest_map for key in fixed_map):
            _error("A fixed dissimilarity refers to an unknown nest.", "invalid_option")
        sizes = {key: sum(_key(nest) == key for _, nest in observed_alternatives) for key in nest_map}
        for label in nest_labels:
            key = _key(label)
            if sizes[key] == 1:
                if key in fixed_map and fixed_map[key] != 1:
                    _error("A globally singleton nest requires fixed dissimilarity one.", "invalid_option")
                fixed_map[key] = 1.0
        canonical_fixed = [[label, fixed_map[_key(label)]] for label in nest_labels if _key(label) in fixed_map]
        free_nests = [j for j, label in enumerate(nest_labels) if _key(label) not in fixed_map]
        catalogue = dict(nests=nest_labels, alternatives=observed_alternatives,
                         fixed_dissimilarity=canonical_fixed, free_nests=free_nests)
    else:
        if not isinstance(catalogue, dict) or set(catalogue) != {"nests", "alternatives", "fixed_dissimilarity", "free_nests"}:
            _error("The query nest catalogue requires the complete fitted geometry.", "invalid_result")
        if not isinstance(catalogue["nests"], list) or not isinstance(catalogue["alternatives"], list):
            _error("The typed nest/alternative catalogue requires canonical lists.", "invalid_result")
        nest_labels = [_identity(v, "declared nest") for v in catalogue["nests"]]
        if not 1 <= len(nest_labels) <= 8 or len({_key(v) for v in nest_labels}) != len(nest_labels):
            _error("The declared typed nest catalogue is invalid.", "invalid_result")
        nest_map = {_key(v): j for j, v in enumerate(nest_labels)}
        fixed_map = {_key(label): value for label, value in _fixed(catalogue["fixed_dissimilarity"])}
        if declarations != catalogue["fixed_dissimilarity"] or any(key not in nest_map for key in fixed_map):
            _error("Fixed declarations disagree with the fitted typed nest catalogue.", "invalid_result")
        free_nests = catalogue["free_nests"]
        if not isinstance(free_nests, list) or any(type(v) is not int for v in free_nests) or free_nests != [j for j, label in enumerate(nest_labels) if _key(label) not in fixed_map]:
            _error("Free dissimilarity order disagrees with the declared geometry.", "invalid_result")
        declared = {}
        for pair in catalogue["alternatives"]:
            if not isinstance(pair, list) or len(pair) != 2:
                _error("The alternative catalogue contains an invalid typed pair.", "invalid_result")
            alternative, nest = (_identity(pair[0], "declared alternative"), _identity(pair[1], "declared nest"))
            if _key(alternative) in declared or _key(nest) not in nest_map:
                _error("The declared disjoint alternative catalogue is invalid.", "invalid_result")
            declared[_key(alternative)] = nest
        for alternative, nest in zip(alternatives, nests):
            if _key(nest) not in nest_map or (_key(alternative) in declared and _key(declared[_key(alternative)]) != _key(nest)):
                _error("New choices require fitted nests and the same nest for known alternatives.", "invalid_nesting")
    fixed_lambdas = {nest_map[key]: value for key, value in fixed_map.items()}
    nest_codes = [nest_map[_key(v)] for v in nests]
    cluster_labels, cluster_map, case_cluster_codes = [], {}, []
    available_rows, group_rows, group_nests, group_cases, case_groups = [], [], [], [], []
    selected_rows, selected_groups = [], []
    joint_variation = set()
    for ci, rows in enumerate(case_rows):
        if len({_key(alternatives[i]) for i in rows}) != len(rows):
            _error("Alternatives must be unique inside each typed choice case.", "duplicate_alternative")
        active = [i for i in rows if available[i]]
        if not 2 <= len(active) <= 20:
            _error("Each case requires 2..20 available alternatives.", "invalid_choice")
        if any(chosen[i] and not available[i] for i in rows) or (require_chosen and sum(chosen[i] for i in rows) != 1):
            _error("Each case requires exactly one chosen available alternative.", "invalid_choice")
        available_rows.extend(active)
        groups = []
        for ni in range(len(nest_labels)):
            members = [i for i in active if nest_codes[i] == ni]
            if members:
                gi = len(group_rows)
                groups.append(gi)
                group_rows.append(members)
                group_nests.append(ni)
                group_cases.append(ci)
                if len(members) > 1:
                    joint_variation.add(ni)
        case_groups.append(groups)
        selected = next((i for i in active if chosen[i]), active[0])
        selected_rows.append(selected)
        selected_groups.append(next(gi for gi in groups if selected in group_rows[gi]))
        if clusters is not None:
            labels = {_key(clusters[i]) for i in rows}
            if len(labels) != 1:
                _error("Respondent cluster must be constant within the complete case.", "split_case_cluster")
            label = clusters[rows[0]]
            if _key(label) not in cluster_map:
                cluster_map[_key(label)] = len(cluster_labels)
                cluster_labels.append(label)
            case_cluster_codes.append(cluster_map[_key(label)])
    if for_fit and any(ni not in joint_variation for ni in free_nests):
        _error("Every free dissimilarity needs simultaneously available alternatives within its nest.", "rank_deficient")
    if for_fit and options["vce"] == "cr0" and len(cluster_labels) < 2:
        _error("CR0 requires at least two declared respondent clusters.", "invalid_cluster")
    resources = _plan(n, q, options, free=len(free_nests), nests=len(nest_labels), cases=len(case_labels),
                      active_groups=len(group_rows), budget_bytes=budget_bytes, for_fit=for_fit)
    X = torch.tensor(list(zip(*raw_x)), dtype=DT, device="cpu")
    centered = torch.empty_like(X, device="cpu")
    for rows in case_rows:
        active = [i for i in rows if available[i]]
        centered[rows] = X[rows] - X[active[0]]
    scale = centered[available_rows].square().mean(0).sqrt()
    if not for_fit:
        scale = torch.where(scale > 0, scale, torch.ones_like(scale, device="cpu"))
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        _error("Within-case utility design contains an unidentified constant attribute.", "rank_deficient")
    Z = centered/scale
    if for_fit:
        singular = torch.linalg.svdvals(Z[available_rows])
        if singular.numel() < q or float(singular[-1]) <= float(singular[0])*1e-10:
            _error("Within-case numeric attribute design is rank deficient or ill-conditioned.", "rank_deficient")
    parameters = list(columns["x"])
    for ni in free_nests:
        name = f"lambda[n{ni+1}]"
        while name in parameters:
            name = "_" + name
        parameters.append(name)
    gi = [rows + [rows[0]]*(20-len(rows)) for rows in group_rows]
    gm = [[j < len(rows) for j in range(20)] for rows in group_rows]
    cg = [groups + [groups[0]]*(8-len(groups)) for groups in case_groups]
    cm = [[j < len(groups) for j in range(8)] for groups in case_groups]
    inputs = dict(chosen=chosen, x=X.tolist(), case=cases, alternative=alternatives, nest=nests,
                  available=available, cluster=clusters)
    return Prepared(X, Z, scale, chosen, available, case_labels, case_rows, alternatives, nest_labels, nest_codes,
                    free_nests, fixed_lambdas, parameters, cluster_labels, case_cluster_codes, columns, options,
                    catalogue, inputs, resources, available_rows, group_rows, group_nests, group_cases, case_groups,
                    selected_rows, selected_groups, torch.tensor(gi, dtype=torch.int64, device="cpu"), torch.tensor(gm, dtype=torch.bool, device="cpu"),
                    torch.tensor(group_nests, dtype=torch.int64, device="cpu"), torch.tensor(cg, dtype=torch.int64, device="cpu"), torch.tensor(cm, dtype=torch.bool, device="cpu"))


def _lambdas(params, p):
    free = {ni: params[p.q+j] for j, ni in enumerate(p.free_nests)}
    return torch.stack([free[ni] if ni in free else params.new_tensor(p.fixed_lambdas[ni]) for ni in range(len(p.nest_labels))])


def _case_logs(params, p):
    """Chosen log probabilities retain nonzero losing mass in saturated tails."""
    utility = (p.Z*p.scale) @ params[:p.q]
    lambdas = _lambdas(params, p)
    group_utility = utility[p.group_indices]
    pivot, pivot_position = group_utility.masked_fill(~p.group_mask, -torch.inf).max(dim=1)
    within = ((group_utility-pivot[:, None])/lambdas[p.group_nest_tensor, None]).masked_fill(~p.group_mask, -torch.inf)
    positions = torch.arange(20, dtype=torch.int64, device="cpu")
    losing = within.exp().masked_fill(~p.group_mask | (positions[None, :] == pivot_position[:, None]), 0)
    # lambda*logsumexp(V/lambda) loses its lambda derivative by cancellation
    # when one utility dominates. Separate the raw maximum and use log1p for
    # the remaining mass, retaining entropy derivatives far below epsilon.
    log_residual = torch.log1p(losing.sum(1))
    upper = pivot + lambdas[p.group_nest_tensor]*log_residual
    case_upper = upper[p.case_group_indices].masked_fill(~p.case_group_mask, -torch.inf)
    result = params.new_zeros(len(p.case_labels))
    lower_cases = [ci for ci, gi in enumerate(p.selected_groups) if len(p.group_rows[gi]) > 1]
    if lower_cases:
        selected_groups = torch.tensor([p.selected_groups[ci] for ci in lower_cases], dtype=torch.int64, device="cpu")
        selected_rows = torch.tensor([p.selected_rows[ci] for ci in lower_cases], dtype=torch.int64, device="cpu")
        others = within[selected_groups].masked_fill(p.group_indices[selected_groups] == selected_rows[:, None], -torch.inf)
        selected = (utility[selected_rows]-pivot[selected_groups])/lambdas[p.group_nest_tensor[selected_groups]]
        values = -torch.nn.functional.softplus(torch.logsumexp(others, dim=1)-selected)
        result = result.index_add(0, torch.tensor(lower_cases, dtype=torch.int64, device="cpu"), values)
    upper_cases = [ci for ci, groups in enumerate(p.case_groups) if len(groups) > 1]
    if upper_cases:
        indices = torch.tensor(upper_cases, dtype=torch.int64, device="cpu")
        selected_groups = torch.tensor([p.selected_groups[ci] for ci in upper_cases], dtype=torch.int64, device="cpu")
        others = case_upper[indices].masked_fill(p.case_group_indices[indices] == selected_groups[:, None], -torch.inf)
        values = -torch.nn.functional.softplus(torch.logsumexp(others, dim=1)-upper[selected_groups])
        result = result.index_add(0, indices, values)
    return result


def _components(params, p, ci):
    """Differentiable physical-parameter distributions, aligned available rows."""
    rows = [i for i in p.case_rows[ci] if p.available[i]]
    utility = (p.X[rows]-p.X[rows[0]]) @ params[:p.q]
    lambdas = _lambdas(params, p)
    group_nests = [ni for ni in range(len(p.nest_labels)) if any(p.nest_codes[i] == ni for i in rows)]
    groups = [[j for j, i in enumerate(rows) if p.nest_codes[i] == ni] for ni in group_nests]
    scaled, inclusive, upper = [], [], []
    for group, ni in zip(groups, group_nests):
        raw = utility[group]
        pivot_position = int(raw.detach().argmax())
        pivot = raw[pivot_position]
        relative = (raw-pivot)/lambdas[ni]
        other = [j for j in range(len(group)) if j != pivot_position]
        residual = torch.log1p(relative[other].exp().sum()) if other else params.sum()*0
        scaled.append(relative)
        inclusive.append(pivot/lambdas[ni]+residual)
        upper.append(pivot+lambdas[ni]*residual)
    inclusive, upper = torch.stack(inclusive), torch.stack(upper)
    log_group = []
    for j in range(len(groups)):
        others = [k for k in range(len(groups)) if k != j]
        log_group.append(-torch.nn.functional.softplus(torch.logsumexp(upper[others], dim=0)-upper[j])
                         if others else params.sum()*0)
    group_logs = torch.stack(log_group)
    log_conditional, log_nest = [None]*len(rows), [None]*len(rows)
    for gi, members in enumerate(groups):
        for j, position in enumerate(members):
            others = [k for k in range(len(members)) if k != j]
            log_conditional[position] = (-torch.nn.functional.softplus(torch.logsumexp(scaled[gi][others], dim=0)-scaled[gi][j])
                                         if others else params.sum()*0)
            log_nest[position] = group_logs[gi]
    lc, ln = torch.stack(log_conditional), torch.stack(log_nest)
    return dict(rows=rows, probability=(lc+ln).exp(), log_probability=lc+ln,
                conditional=lc.exp(), log_conditional=lc, nest_probability=ln.exp(), log_nest_probability=ln,
                group_nests=group_nests, group_probability=group_logs.exp(), group_log_probability=group_logs,
                inclusive_value=inclusive, upper_utility=upper)


class _Objective:
    def __init__(self, p):
        self.p, self.used = p, 0

    def charge(self, multiplier):
        k = len(self.p.parameters)
        self.used += self.p.n*k*multiplier
        if self.used > self.p.options["max_work"]:
            _error("Actual joint derivative work exceeded max_work.", "work_budget")

    def physical(self, point):
        return torch.cat((point[:self.p.q]/self.p.scale, torch.sigmoid(point[self.p.q:])))

    def value(self, point):
        params = self.physical(point)
        # Inadmissible trial points are rejected by the line search without
        # constructing divisions by a numerically zero dissimilarity.
        if self.p.free_nests and not bool(((params[self.p.q:] > 1e-10) & (params[self.p.q:] < 1-1e-10)).all()):
            return point.sum()*0 + point.new_tensor(-torch.inf)
        return _case_logs(params, self.p).sum()

    def __call__(self, point):
        self.charge(6)
        with torch.enable_grad():
            x = point.detach().requires_grad_()
            value = self.value(x)
            gradient = torch.autograd.grad(value, x)[0]
        return value.detach(), gradient.detach()

    def hessian(self, point):
        self.charge(8*(len(point)+1))
        with torch.enable_grad():
            return torch.autograd.functional.hessian(self.value, point, vectorize=True).detach()


def _inverse(information):
    diagonal = information.diag()
    if not bool(torch.isfinite(information).all()) or not bool((diagonal > 0).all()):
        _error("Full joint observed information is nonfinite or nonpositive.", "no_finite_mle")
    sd = diagonal.sqrt()
    normalized = information/sd[:, None]/sd[None, :]
    eigen = torch.linalg.eigvalsh((normalized+normalized.T)/2)
    if float(eigen[0]) <= 1e-10*float(eigen[-1]):
        _error("Full coefficient/dissimilarity information is unidentified or ill-conditioned.", "rank_deficient")
    chol, status = torch.linalg.cholesky_ex(normalized)
    if int(status):
        _error("Full joint observed information is not positive definite.", "no_finite_mle")
    inverse = torch.cholesky_inverse(chol)/sd[:, None]/sd[None, :]
    return (inverse+inverse.T)/2


def _moments(params, p):
    with torch.enable_grad():
        point = params.detach().requires_grad_()
        values = _case_logs(point, p)
        total = values.sum()
        gradient = torch.autograd.grad(total, point)[0]
        hessian = torch.autograd.functional.hessian(lambda x: _case_logs(x, p).sum(), params, vectorize=True)
        scores = torch.autograd.functional.jacobian(lambda x: _case_logs(x, p), params, vectorize=True, strategy="forward-mode")
    information = -(hessian+hessian.T)/2
    if any(not bool(torch.isfinite(v).all()) for v in (values, gradient, information, scores)):
        _error("Full physical likelihood moments are nonfinite.", "numerical_failure")
    return dict(information=information.detach(), gradient=gradient.detach(), case_scores=scores.detach(),
                case_loglikelihood=values.detach(), log_likelihood=float(total.detach()))


def _stationarity(params, p, moments):
    if p.free_nests:
        lambdas = params[p.q:]
        if not bool(((lambdas > MIN_LAMBDA) & (lambdas < 1-MIN_LAMBDA)).all()):
            _error("Free dissimilarity is at or beyond the strict [0.001,0.999] computational interior.", "boundary_solution")
    bread = _inverse(moments["information"])
    step = bread@moments["gradient"]
    units = torch.cat((p.scale, torch.ones(len(p.free_nests), dtype=DT, device="cpu")))
    norm_step = float((step*units).abs().max())
    decrement = float(moments["gradient"]@step)
    step_limit = max(20*p.options["tolerance"], 1e-8)
    decrement_limit = max(100*p.options["tolerance"]**2, 1e-16)*len(p.case_labels)
    diag = dict(normalized_parameter_step=norm_step, score_decrement=max(0.0, decrement),
                physical_score_max=float(moments["gradient"].abs().max()), step_limit=step_limit, decrement_limit=decrement_limit)
    if not math.isfinite(norm_step) or not math.isfinite(decrement) or decrement < -1e-12:
        _error("Physical stationarity diagnostics are nonfinite.", "no_finite_mle")
    return diag, bread, norm_step <= step_limit and decrement <= decrement_limit


def _fit(p):
    objective = _Objective(p)
    candidates, attempts = [], []
    starts = STARTS if p.free_nests else (None,)
    for start in starts:
        initial = torch.zeros(len(p.parameters), dtype=DT, device="cpu")
        if start is not None:
            initial[p.q:] = math.log(start/(1-start))
        try:
            fitted = kernel_call(maximize_bfgs, objective, initial, hessian_fn=objective.hessian,
                                   max_iter=p.options["max_iterations"], scaled_gradient_tol=min(1e-12, p.options["tolerance"]**2),
                                   raise_on_failure=False)
            params = objective.physical(fitted.theta).detach()
            # BFGS transformed convergence is advisory; physical replay decides.
            moments = _moments(params, p)
            diagnostic, _, stationary = _stationarity(params, p, moments)
            if not stationary:
                _error("Joint ML did not pass physical score/Newton-step stationarity.", "nonconvergence")
            record = dict(start_dissimilarity=start, status="stationary_interior", iterations=fitted.iterations,
                          params=params.tolist(), log_likelihood=moments["log_likelihood"], stationarity=diagnostic)
            candidates.append((moments["log_likelihood"], params, len(attempts)))
            attempts.append(record)
        except AnalysisError as error:
            if error.code in {"work_budget", "resource_limit", "workspace_limit"}:
                raise
            attempts.append(dict(start_dissimilarity=start, status="rejected", code=error.code))
    if not candidates:
        code = "boundary_solution" if any(r.get("code") == "boundary_solution" for r in attempts) else "no_finite_mle"
        _error("No multistart candidate passed finite interior joint ML, identification and physical stationarity.", code)
    _, params, selected = max(candidates, key=lambda item: item[0])
    convergence = dict(starts=attempts, selected_start=selected, work_used=objective.used,
                       max_work=p.options["max_work"], globally_certified=False, solver=SOLVER)
    return params, convergence


def _psd(matrix, name, code="invalid_covariance"):
    diagonal = matrix.diag()
    if not bool(torch.isfinite(matrix).all()) or bool((diagonal < 0).any()):
        _error(f"Full {name} has nonfinite cells or negative diagonal.", code)
    positive = diagonal > 0
    if bool((matrix[~positive] != 0).any()) or bool((matrix[:, ~positive] != 0).any()):
        _error(f"Full {name} has nonzero covariance for a zero variance.", code)
    if bool(positive.any()):
        sd = diagonal[positive].sqrt()
        normalized = matrix[positive][:, positive]/sd[:, None]/sd[None, :]
        if float((normalized-normalized.T).abs().max()) > 1e-10 or float(torch.linalg.eigvalsh((normalized+normalized.T)/2).min()) < -1e-10*max(1.0, float(normalized.abs().max())):
            _error(f"Full {name} is not positive semidefinite in standardized units.", code)


def _evaluate(params, p):
    moments = _moments(params, p)
    diagnostic, bread, stationary = _stationarity(params, p, moments)
    if not stationary:
        _error("Saved/final parameters fail actual physical stationarity.", "no_finite_mle")
    cluster_scores = torch.zeros((len(p.cluster_labels), len(params)), dtype=DT, device="cpu")
    if p.cluster_labels:
        cluster_scores.index_add_(0, torch.tensor(p.case_cluster_codes, dtype=torch.int64, device="cpu"), moments["case_scores"])
    score = cluster_scores if p.options["vce"] == "cr0" else moments["case_scores"]
    meat = score.T@score
    influence = score@bread
    covariance = bread if p.options["vce"] == "oim" else influence.T@influence
    covariance = (covariance+covariance.T)/2
    _psd(covariance, "covariance", "numerical_failure")
    probabilities, nest_probabilities = [], []
    for ci in range(len(p.case_labels)):
        components = _components(params, p, ci)
        for position, row in enumerate(components["rows"]):
            probabilities.append([row, *[float(components[key][position]) for key in (
                "probability", "log_probability", "conditional", "log_conditional", "nest_probability", "log_nest_probability")]])
        for j, ni in enumerate(components["group_nests"]):
            nest_probabilities.append([ci, ni, float(components["group_probability"][j]), float(components["group_log_probability"][j]),
                                       float(components["inclusive_value"][j]), float(components["upper_utility"][j])])
    # Independent raw centered utility representation must agree with the
    # standardized optimizer design, including every available choice row.
    for rows in p.case_rows:
        active = [i for i in rows if p.available[i]]
        raw = (p.X[active]-p.X[active[0]])@params[:p.q]
        standardized = p.Z[active]@(params[:p.q]*p.scale)
        if not bool(torch.isfinite(raw).all()) or float((raw-standardized).abs().max()) > 2e-9*max(1.0, float(raw.abs().max())):
            _error("Original-unit coefficients cannot reproduce fitted case utility differences.", "numerical_failure")
    result = dict(params=params.tolist(), beta=params[:p.q].tolist(), lambdas=_lambdas(params, p).tolist(),
                  parameter_names=p.parameters, information=moments["information"].tolist(), bread=bread.tolist(),
                  meat=meat.tolist(), covariance=covariance.tolist(), case_scores=moments["case_scores"].tolist(),
                  cluster_scores=cluster_scores.tolist(), case_loglikelihood=moments["case_loglikelihood"].tolist(),
                  log_likelihood=moments["log_likelihood"], probability_components=probabilities,
                  probability_rows=[row[0] for row in probabilities], probabilities=[row[1] for row in probabilities],
                  log_probabilities=[row[2] for row in probabilities], nest_probabilities=nest_probabilities,
                  scale=p.scale.tolist(), stationarity=diagnostic, covariance_type=p.options["vce"],
                  n=p.n, n_cases=len(p.case_labels), n_clusters=len(p.cluster_labels), n_nests=len(p.nest_labels))
    if any(not math.isfinite(v) for row in probabilities for v in row) or any(not math.isfinite(v) for row in nest_probabilities for v in row):
        _error("Saved full probability geometry exceeds finite float64 support.", "numerical_failure")
    return result


def _checksum(state):
    return hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "checksum"}, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _settings(p, level):
    return dict(columns=p.columns, **p.options, level=level, device="cpu", precision="float64", missing="raise", weights=None,
                n_input=p.n, n_retained=p.n, n_dropped=0, n_cases=len(p.case_labels), n_clusters=len(p.cluster_labels),
                root_scale=1.0, lower_utility="V divided by nest dissimilarity", nests="two-level disjoint global alternative partition",
                covariance="full joint beta/dissimilarity OIM, choice-case HC0 or respondent CR0; no multiplier", inference_df=None,
                global_wald=None, parameter_test="beta zero-null normal; dissimilarities reported without boundary hypothesis tests",
                free_dissimilarity_interior=[MIN_LAMBDA, 1-MIN_LAMBDA], fixed_dissimilarity_range=[MIN_LAMBDA, 1.0],
                availability="condition on declared choice sets; empty nests structural zero", max_rows=4096, max_cases=512,
                max_available_alternatives=20, max_attributes=8, max_nests=8, numeric_magnitude_bound=1e12,
                complete_inputs_saved=True, complete_covariance_saved=True, robust_covariance_may_be_singular=True,
                restoration="complete portable numerical and physical-stationarity replay; no optimizer or refit",
                resources=p.resources, excluded_scope=["fit weights", "implicit category encoding", "common intercept", "missing-row deletion",
                    "Dataset", "CUDA/MPS", "cross nesting", "deeper nests", "random coefficients", "endogenous availability", "boundary inference", "vendor parity"])


def _marker(base, names):
    while base in names:
        base = "_" + base
    return base


def _matrix(value, names):
    return table([[name, *row] for name, row in zip(names, value)], columns=[_marker("parameter", names), *names])


def _assemble(state, p):
    fit, level = state["fit"], state["level"]
    names, _, zcrit = fit["parameter_names"], *_confidence(level)
    parameters = []
    for j, (name, value) in enumerate(zip(names, fit["params"])):
        variance = fit["covariance"][j][j]
        se = math.sqrt(variance)
        z = value/se if se and j < p.q else None
        lower, upper = (value-zcrit*se, value+zcrit*se) if se else (None, None)
        if j >= p.q and se:
            location = math.log(value/(1-value))
            logit_se = se/(value*(1-value))
            lower = float(torch.sigmoid(torch.tensor(location-zcrit*logit_se, dtype=DT, device="cpu")))
            upper = float(torch.sigmoid(torch.tensor(location+zcrit*logit_se, dtype=DT, device="cpu")))
        if any(v is not None and not math.isfinite(v) for v in (z, lower, upper)):
            _error("Parameter inference exceeds finite float64 support.", "numerical_failure")
        parameters.append([name, "coefficient" if j < p.q else "dissimilarity", value, se, z,
                           math.erfc(abs(z)/math.sqrt(2)) if z is not None else None, lower, upper,
                           "asymptotic normal" if se and j < p.q else "transformed asymptotic normal; no boundary test" if se else "unavailable: zero first-order variance"])
    dissimilarities = []
    for ni, label in enumerate(p.nest_labels):
        free = ni in p.free_nests
        j = p.q+p.free_nests.index(ni) if free else None
        estimate = fit["lambdas"][ni]
        se = math.sqrt(fit["covariance"][j][j]) if free else None
        logit_se = se/(estimate*(1-estimate)) if se else None
        location = math.log(estimate/(1-estimate)) if free else None
        lower = float(torch.sigmoid(torch.tensor(location-zcrit*logit_se, dtype=DT, device="cpu"))) if logit_se else None
        upper = float(torch.sigmoid(torch.tensor(location+zcrit*logit_se, dtype=DT, device="cpu"))) if logit_se else None
        dissimilarities.append([label, names[j] if free else None, estimate, se, lower, upper,
                               "free: transformed normal interval" if se else "free: zero first-order variance" if free else "fixed; no fitted uncertainty"])
    columns, inputs = state["columns"], state["inputs"]
    input_names = [columns["case"], columns["alternative"], columns["nest"], columns["chosen"], *columns["x"]]
    if columns["available"]:
        input_names.append(columns["available"])
    if columns["cluster"]:
        input_names.append(columns["cluster"])
    input_rows = []
    for i in range(p.n):
        row = [i, inputs["case"][i], inputs["alternative"][i], inputs["nest"][i], inputs["chosen"][i], *inputs["x"][i]]
        if columns["available"]:
            row.append(inputs["available"][i])
        if columns["cluster"]:
            row.append(inputs["cluster"][i])
        input_rows.append(row)
    probability_rows = [[inputs["case"][row[0]], row[0], inputs["alternative"][row[0]], inputs["nest"][row[0]], inputs["chosen"][row[0]], *row[1:]] for row in fit["probability_components"]]
    nests = {(int(row[0]), int(row[1])): row[2:] for row in fit["nest_probabilities"]}
    nest_rows = []
    for ci, label in enumerate(p.case_labels):
        for ni, nest in enumerate(p.nest_labels):
            values = nests.get((ci, ni))
            nest_rows.append([label, nest, *(values if values is not None else [0.0, None, None, None]), values is not None])
    skip = {"params", "beta", "lambdas", "information", "bread", "meat", "covariance", "case_scores", "cluster_scores", "case_loglikelihood", "probabilities", "log_probabilities", "probability_rows", "probability_components", "nest_probabilities"}
    frames = dict(inputs=table(input_rows, columns=[_marker("row", input_names), *input_names]),
                  parameters=table(parameters, columns=["parameter", "kind", "estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper", "inference_status"]),
                  dissimilarities=table(dissimilarities, columns=["nest", "parameter", "estimate", "std_error", "ci_lower", "ci_upper", "inference_status"]),
                  **{key: _matrix(fit[key], names) for key in ("information", "bread", "meat", "covariance")},
                  case_scores=table([[label, *score] for label, score in zip(p.case_labels, fit["case_scores"])], columns=[_marker("case", names), *names]),
                  cluster_scores=table([[label, *score] for label, score in zip(p.cluster_labels, fit["cluster_scores"])], columns=[_marker("cluster", names), *names]),
                  case_likelihood=table([[label, value, math.exp(value)] for label, value in zip(p.case_labels, fit["case_loglikelihood"])], columns=["case", "log_likelihood", "chosen_probability"]),
                  probabilities=table(probability_rows, columns=["case", "row", "alternative", "nest", "chosen", "probability", "log_probability", "conditional", "log_conditional", "nest_probability", "log_nest_probability"]),
                  nest_probabilities=table(nest_rows, columns=["case", "nest", "probability", "log_probability", "inclusive_value", "upper_utility", "nonempty"]),
                  fit_summary=table([[key, json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)] for key, value in sorted(fit.items()) if key not in skip], columns=["setting", "json"]))
    return TableSet(frames, title="Two-level nested logit", method="nlogit", contract=SCHEMA,
                    settings=state["settings"], nested_logit_state=state, notes=NOTES)


@procedure
def nlogit(data, chosen, x, *, case, alternative, nest, available=None, fixed_dissimilarity=None,
           vce="oim", cluster=None, level=.95, device="cpu", weights=None,
           max_iterations=300, tolerance=1e-9, max_work=2_000_000_000):
    """Fit a bounded, two-level disjoint-nest random-utility choice model."""
    if device != "cpu":
        _error("Nested logit supports explicit native CPU float64 only.", "unsupported_device")
    if weights is not None:
        _error("Fit weights are outside this complete choice-case covariance contract.", "unsupported_weights")
    if (vce == "cr0") != (cluster is not None):
        _error("Declare a respondent cluster exactly when vce is cr0.", "invalid_option")
    options, columns = _options(vce, max_iterations, tolerance, max_work), _columns(chosen, x, case, alternative, nest, available, cluster)
    level, _ = _confidence(level)
    p = _prepare(data, columns, options, fixed_dissimilarity=fixed_dissimilarity)
    params, convergence = _fit(p)
    fit = _evaluate(params, p) | {"convergence": convergence, "solver": SOLVER}
    state = dict(schema=SCHEMA, level=level, columns=columns, options=options, inputs=p.inputs,
                 fixed_dissimilarity=p.catalogue["fixed_dissimilarity"], catalogue=p.catalogue,
                 fit=fit, settings=_settings(p, level))
    state["checksum"] = _checksum(state)
    return _assemble(state, p)


def _finite_vector(value, count):
    return isinstance(value, list) and len(value) == count and all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v)) for v in value)


def _finite_matrix(value, rows, cols):
    return isinstance(value, list) and len(value) == rows and all(_finite_vector(row, cols) for row in value)


def _matches(saved, expected, name):
    actual, value = torch.tensor(expected, dtype=DT, device="cpu"), torch.tensor(saved, dtype=DT, device="cpu")
    if actual.shape != value.shape or not bool(torch.isfinite(value).all()):
        _error(f"Saved {name} has incomplete or nonfinite dimensions.", "invalid_result")
    if name in {"information", "bread", "meat", "covariance"}:
        scale = actual.diag().abs().sqrt()
        allowed = 2e-9*scale[:, None]*scale[None, :]
        valid = bool(((actual-value).abs() <= allowed).all()) and bool((value[actual == 0] == 0).all())
    else:
        valid = bool(torch.allclose(value, actual, rtol=2e-9, atol=2e-12))
    if not valid:
        _error(f"Saved {name} disagrees with complete physical input/parameter replay.", "invalid_result")


def _prepared_from_state(state):
    columns, inputs = state["columns"], state["inputs"]
    data = {columns[key]: inputs[key] for key in ("chosen", "case", "alternative", "nest")}
    data.update({name: [row[j] for row in inputs["x"]] for j, name in enumerate(columns["x"])})
    if columns["available"]:
        data[columns["available"]] = inputs["available"]
    if columns["cluster"]:
        data[columns["cluster"]] = inputs["cluster"]
    return _prepare(data, columns, state["options"], fixed_dissimilarity=state["fixed_dissimilarity"])


def _validate_state(result):
    """Validate complete sealed state and physical moments; never optimize."""
    attrs = result.attrs if isinstance(result, TableSet) else result
    if not isinstance(attrs, Mapping) or set(attrs) != {"method", "contract", "settings", "nested_logit_state", "notes"}:
        _error("Supply the complete nested fit TableSet or its portable attrs.", "invalid_result")
    state = attrs["nested_logit_state"]
    fields = {"schema", "level", "columns", "options", "inputs", "fixed_dissimilarity", "catalogue", "fit", "settings", "checksum"}
    if not isinstance(state, dict) or set(state) != fields or state["schema"] != SCHEMA:
        _error("Unsupported nested logit saved schema or fields.", "invalid_result")
    try:
        if not isinstance(state["checksum"], str) or len(state["checksum"]) != 64 or not hmac.compare_digest(state["checksum"], _checksum(state)):
            _error("Saved nested logit checksum does not match.", "invalid_result")
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError("invalid_result", "Saved nested state must be finite canonical JSON.") from error
    if not isinstance(state["columns"], dict) or set(state["columns"]) != {"chosen", "x", "case", "alternative", "nest", "available", "cluster"}:
        _error("Saved column schema is incomplete.", "invalid_result")
    columns = _columns(**state["columns"])
    if not isinstance(state["options"], dict) or set(state["options"]) != {"vce", "max_iterations", "tolerance", "max_work"}:
        _error("Saved options schema is incomplete.", "invalid_result")
    options = _options(**state["options"])
    if (options["vce"] == "cr0") != (columns["cluster"] is not None):
        _error("Saved covariance and cluster declarations disagree.", "invalid_result")
    inputs = state["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"chosen", "x", "case", "alternative", "nest", "available", "cluster"} or not isinstance(inputs["chosen"], list):
        _error("Saved complete input schema is unsupported.", "invalid_result")
    n, q = len(inputs["chosen"]), len(columns["x"])
    _plan(n, q, options, for_fit=False)
    if not _finite_matrix(inputs["x"], n, q) or any(not isinstance(inputs[key], list) or len(inputs[key]) != n for key in ("case", "alternative", "nest", "available")):
        _error("Saved inputs have incomplete dimensions.", "invalid_result")
    if columns["cluster"] is None and inputs["cluster"] is not None:
        _error("Saved inputs contain an undeclared cluster column.", "invalid_result")
    p = _prepared_from_state(state)
    if (json.dumps(inputs, ensure_ascii=False, sort_keys=True) != json.dumps(p.inputs, ensure_ascii=False, sort_keys=True)
            or json.dumps(state["catalogue"], ensure_ascii=False, sort_keys=True) != json.dumps(p.catalogue, ensure_ascii=False, sort_keys=True)
            or state["fixed_dissimilarity"] != p.catalogue["fixed_dissimilarity"]):
        _error("Saved complete input/catalogue differs from the canonical retained geometry.", "invalid_result")
    fit = state["fit"]
    fit_keys = {"params", "beta", "lambdas", "parameter_names", "information", "bread", "meat", "covariance", "case_scores", "cluster_scores",
                "case_loglikelihood", "log_likelihood", "probabilities", "log_probabilities", "probability_rows", "probability_components", "nest_probabilities", "scale", "stationarity", "covariance_type",
                "n", "n_cases", "n_clusters", "n_nests", "convergence", "solver"}
    k = len(p.parameters)
    if not isinstance(fit, dict) or set(fit) != fit_keys or not _finite_vector(fit["params"], k):
        _error("Saved complete fit schema is unsupported.", "invalid_result")
    shapes = {name: (k, k) for name in ("information", "bread", "meat", "covariance")}
    shapes |= dict(case_scores=(len(p.case_labels), k), cluster_scores=(len(p.cluster_labels), k),
                   probability_components=(len(p.available_rows), 7), nest_probabilities=(len(p.group_rows), 6))
    if any(not _finite_matrix(fit[name], *shape) for name, shape in shapes.items()):
        _error("Saved full joint matrices require complete finite dimensions.", "invalid_result")
    vectors = dict(beta=q, lambdas=len(p.nest_labels), probabilities=len(p.available_rows),
                   log_probabilities=len(p.available_rows), case_loglikelihood=len(p.case_labels), scale=q)
    if any(not _finite_vector(fit[name], count) for name, count in vectors.items()) or isinstance(fit["log_likelihood"], bool) or not isinstance(fit["log_likelihood"], Real) or not math.isfinite(float(fit["log_likelihood"])):
        _error("Saved full likelihood vectors require finite numeric cells.", "invalid_result")
    if not isinstance(fit["probability_rows"], list) or any(type(v) is not int for v in fit["probability_rows"]):
        _error("Saved available row identities require canonical integer positions.", "invalid_result")
    for name in ("information", "bread", "meat", "covariance"):
        _psd(torch.tensor(fit[name], dtype=DT, device="cpu"), name)
    replay = _evaluate(torch.tensor(fit["params"], dtype=DT, device="cpu"), p)
    for name, expected in replay.items():
        if name in shapes or name in {*vectors, "log_likelihood"}:
            _matches(fit[name], expected, name)
        elif name == "stationarity":
            if not isinstance(fit[name], dict) or set(fit[name]) != set(expected):
                _error("Saved physical stationarity schema is incomplete.", "invalid_result")
            for key, actual in expected.items():
                value = fit[name][key]
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or abs(value-actual) > 2e-10*max(1.0, abs(actual)):
                    _error("Saved physical stationarity differs from actual replay.", "invalid_result")
        elif fit[name] != expected or type(fit[name]) is not type(expected):
            _error(f"Saved {name} differs from full scientific replay.", "invalid_result")
    convergence = fit["convergence"]
    keys = {"starts", "selected_start", "work_used", "max_work", "globally_certified", "solver"}
    starts = STARTS if p.free_nests else (None,)
    if not isinstance(convergence, dict) or set(convergence) != keys or convergence["globally_certified"] is not False or convergence["solver"] != SOLVER or fit["solver"] != SOLVER:
        _error("Saved optimizer protocol is unsupported.", "invalid_result")
    if type(convergence["work_used"]) is not int or not 0 < convergence["work_used"] <= options["max_work"] or convergence["max_work"] != options["max_work"]:
        _error("Saved derivative work accounting is invalid.", "invalid_result")
    attempts = convergence["starts"]
    if not isinstance(attempts, list) or len(attempts) != len(starts):
        _error("Saved multistart declarations are incomplete.", "invalid_result")
    accepted = []
    for i, (record, start) in enumerate(zip(attempts, starts)):
        if not isinstance(record, dict) or record.get("start_dissimilarity") != start:
            _error("Saved multistart sequence is invalid.", "invalid_result")
        if record.get("status") == "rejected":
            if set(record) != {"start_dissimilarity", "status", "code"} or record["code"] not in {"rank_deficient", "no_finite_mle", "boundary_solution", "nonconvergence", "numerical_failure"}:
                _error("Saved rejected-start diagnostics are invalid.", "invalid_result")
            continue
        if (set(record) != {"start_dissimilarity", "status", "iterations", "params", "log_likelihood", "stationarity"}
                or record["status"] != "stationary_interior" or type(record["iterations"]) is not int
                or not 1 <= record["iterations"] <= options["max_iterations"]+10 or not _finite_vector(record["params"], k)
                or isinstance(record["log_likelihood"], bool) or not isinstance(record["log_likelihood"], Real)
                or not math.isfinite(float(record["log_likelihood"]))):
            _error("Saved successful-start diagnostics are incomplete.", "invalid_result")
        candidate = _moments(torch.tensor(record["params"], dtype=DT, device="cpu"), p)
        diagnostic, _, stationary = _stationarity(torch.tensor(record["params"], dtype=DT, device="cpu"), p, candidate)
        if not stationary or abs(record["log_likelihood"]-candidate["log_likelihood"]) > 2e-9*max(1.0, abs(candidate["log_likelihood"])) or record["stationarity"] != diagnostic:
            _error("Saved successful multistart candidate fails actual physical replay.", "invalid_result")
        accepted.append((record["log_likelihood"], i))
    selected = convergence["selected_start"]
    if type(selected) is not int or not accepted or selected != max(accepted, key=lambda item: item[0])[1] or attempts[selected]["params"] != fit["params"]:
        _error("Saved selection is not the declared best certified multistart candidate.", "invalid_result")
    level, _ = _confidence(state["level"])
    resources = state["settings"].get("resources") if isinstance(state["settings"], dict) else None
    if not isinstance(resources, dict) or type(resources.get("budget_bytes")) is not int:
        _error("Saved resource metadata is incomplete.", "invalid_result")
    p.resources = _plan(n, q, options, free=len(p.free_nests), nests=len(p.nest_labels), cases=len(p.case_labels),
                        active_groups=len(p.group_rows), budget_bytes=resources["budget_bytes"])
    if state["settings"] != _settings(p, level):
        _error("Saved settings differ from the complete bounded contract.", "invalid_result")
    expected = _assemble(state, p)
    if dict(attrs) != expected.attrs:
        _error("Saved attrs differ from the complete sealed state.", "invalid_result")
    if isinstance(result, TableSet) and (set(result) != set(expected) or any(not isinstance(result[key], pd.DataFrame) or not result[key].equals(expected[key]) for key in expected)):
        _error("Saved fit tables disagree with complete portable state.", "invalid_result")
    return state, p


@procedure
def nlogit_restore(result, *, level=None):
    """Restore complete joint fit tables after numerical replay, without refit."""
    state, p = _validate_state(result)
    if level is None or level == state["level"]:
        return _assemble(state, p)
    level, _ = _confidence(level)
    changed = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
    changed["level"] = level
    changed["settings"]["level"] = level
    changed["checksum"] = _checksum(changed)
    return _assemble(changed, p)
