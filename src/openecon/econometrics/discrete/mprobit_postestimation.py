"""Saved multinomial-probit queries and complete physical-parameter deltas.

Choice sets, evaluation attributes and positive case weights are held fixed.
The raw-cell derivative changes exactly one alternative's attribute. Its
parameter derivative includes the estimated covariance coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.resources import plan_workspace, tensor_bytes, workspace_budget_bytes

from .rank_ordered_postestimation import (
    _confidence, _delta, _error, _factor, _jacobian, _label, _matrix, _normal,
)

DT = torch.float64
MAX_TARGETS = 256
MAX_CASE_ROWS = 8192
KINDS = ("probability", "log_probability")


def _limits(max_work, max_bytes):
    if type(max_work) is not int or max_work < 1:
        _error("max_work must be a positive integer.", "resource_budget")
    if type(max_bytes) is not int or max_bytes < 1:
        _error("max_bytes must be a positive integer.", "resource_budget")
    return max_work, min(max_bytes, workspace_budget_bytes())


def _query(result, data, budget_bytes, max_work):
    from .mprobit import _prepare, _validated

    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        state, p = _validated(result, budget_bytes=budget_bytes, max_work=max_work)
        if data is not None:
            p = _prepare(
                data, {**state["columns"], "cluster": None},
                {**state["options"], "vce": "oim", "max_work": min(max_work, state["options"]["max_work"])},
                catalogue=state["catalogue"], fixed_covariance=state["fixed_covariance"],
                require_chosen=False, for_fit=False, budget_bytes=budget_bytes,
            )
        theta = torch.tensor(state["fit"]["params"], dtype=DT, device="cpu")
        covariance = torch.tensor(state["fit"]["covariance"], dtype=DT, device="cpu")
    width = len(theta)
    replay_work = state["fit"]["n_cases"]*(96+6*width)*8*(width+1)
    replay_work *= 4 if state["fixed_covariance"] is None else 2
    prepare_work = len(p.case_labels)*(96+6*width)*8*(width+1) if data is not None else 0
    return state, p, theta, covariance, replay_work+prepare_work


def _logs(theta, p, ci, X=None):
    from .mprobit import _log_probabilities

    distribution = _log_probabilities(theta, p, ci, X=X)
    rows, logs = distribution["rows"], distribution["log_probability"]
    if (not bool(torch.isfinite(logs).all()) or bool((logs > 1e-12).any())
            or abs(float(logs.detach().exp().sum())-1.) > 2e-9):
        _error("Complete query probabilities fail finite Gaussian normalization without renormalization.",
               "numerical_failure")
    return rows, logs


def _log_target(theta, p, ci, row, X=None):
    rows, logs = _logs(theta, p, ci, X)
    return logs[rows.index(row)]


def _distribution_size(p, ci):
    return sum(bool(p.available[row]) for row in p.case_rows[ci])


def _resources(p, q, rows, *, transformed=0, max_work, max_bytes, mixed=False, admission_work=0):
    nodes = int(p.options.get("quadrature_order", 96))
    work = (rows + transformed) * max(q, 1) ** (2 if mixed else 1) * nodes * 128
    work += len(p.alternative_labels) * (q + 8)
    if mixed:
        work += len(p.alternative_labels)*len(p.columns["x"])*rows*8
    work += admission_work
    if work > max_work:
        _error("The complete query and joint derivatives exceed max_work before evaluation.",
               "resource_budget")
    size = rows + transformed
    plan = plan_workspace(
        "multinomial probit saved query",
        {
            "query choices and differentiable quadrature":
                tensor_bytes((len(p.alternative_labels), q + nodes)) * 16,
            "physical parameter covariance": tensor_bytes((q, q)) * 8,
            "joint Jacobian and delta factors": tensor_bytes((size, q)) * 8,
            "complete joint target covariance and tables": tensor_bytes((size, size)) * 16,
            "complete support diagnostics": tensor_bytes((MAX_CASE_ROWS if mixed else size, 10)) * 2,
        }, budget_bytes=max_bytes,
    ).record()
    return {**plan, "planned_work_units": work, "max_work_units": max_work,
            "saved_fit_replay_and_query_admission_work_units": admission_work,
            "derivatives": "exact native autograd mixed derivatives" if mixed else "native autograd"}


def _output(method, frames, state, p, data, level, **extra):
    settings = dict(
        method=method, level=level, precision="float64", device="cpu",
        source_fit_checksum=state["checksum"], query_inputs=p.inputs,
        columns=p.columns, vce=state["options"]["vce"],
        data_source="saved fit" if data is None else "new resident data",
        no_optimizer=True, refit=False, missing="raise", vendor_parity=False,
        inference_df=None,
        uncertainty="full joint physical coefficient and free covariance parameter delta; query choice sets, attributes and standardization weights fixed",
        denominator="all available known alternatives, including unmaterialized targets",
        empirical_distribution_sampling_uncertainty=False,
        max_declared_targets=MAX_TARGETS, max_per_case_support_rows=MAX_CASE_ROWS,
        **extra,
    )
    return TableSet(
        frames, title="Multinomial probit saved query", method=method,
        contract="mprobit_postestimation_v1", settings=settings,
        source_fit_state=state, notes=[settings["uncertainty"], settings["denominator"]],
    )


def _prediction_targets(targets, p, kind):
    if kind not in KINDS:
        _error("kind must be probability or log_probability.", "invalid_option")
    if targets is None:
        count = sum(_distribution_size(p, ci) for ci in range(len(p.case_labels)))
        if count > MAX_TARGETS:
            _error("Default predictions exceed 256 targets; declare an explicit subset.",
                   "resource_budget")
        targets = [dict(case=p.case_labels[ci], alternative=p.alternative_labels[row])
                   for ci, rows in enumerate(p.case_rows) for row in rows if p.available[row]]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error("Prediction requires one through 256 targets.", "resource_budget")
    cases = {_label(case): ci for ci, case in enumerate(p.case_labels)}
    parsed, seen = [], set()
    for target in targets:
        if (not isinstance(target, Mapping) or not {"case", "alternative"} <= set(target)
                or set(target) - {"case", "alternative", "kind"}):
            _error("Prediction targets require case, alternative and optional kind.")
        selected_kind = target.get("kind", kind)
        if selected_kind not in KINDS:
            _error("Target kind must be probability or log_probability.", "invalid_option")
        ci = cases.get(_label(target["case"]))
        if ci is None:
            _error("A requested case is absent from the query.", "no_support")
        row = next((r for r in p.case_rows[ci]
                    if _label(p.alternative_labels[r]) == _label(target["alternative"])), None)
        if row is None:
            _error("A requested alternative is absent from that query case.", "no_support")
        if not p.available[row] and selected_kind == "log_probability":
            _error("Unavailable alternatives have zero probability and undefined log probability.",
                   "no_support")
        key = (selected_kind, ci, row)
        if key in seen:
            _error("Prediction targets must be unique.")
        seen.add(key)
        parsed.append(key)
    return parsed


def _value_gradient(function, theta):
    with torch.inference_mode(False), torch.enable_grad(), torch.device("cpu"):
        value = function(theta)
        gradient = torch.autograd.functional.jacobian(function, theta)
    if not bool(torch.isfinite(value).all() & torch.isfinite(gradient).all()):
        _error("Query values or physical derivatives exceed finite float64 support.",
               "numerical_failure")
    return float(value.detach()), gradient


@procedure
def mprobit_predict(result, data=None, *, targets=None, kind="probability", level=0.95,
                    max_work=100_000_000, max_bytes=256_000_000):
    """Joint alternative probabilities or log probabilities from a sealed saved fit."""
    level, z = _confidence(level)
    max_work, max_bytes = _limits(max_work, max_bytes)
    state, p, theta, covariance, admission_work = _query(result, data, max_bytes, max_work)
    parsed = _prediction_targets(targets, p, kind)
    resources = _resources(p, len(theta), len(parsed), transformed=2*len(parsed),
                           max_work=max_work, max_bytes=max_bytes, admission_work=admission_work)
    names = [f"p{i+1}" for i in range(len(parsed))]
    values, gradients, log_values, log_gradients, odds_values, odds_gradients = [], [], [], [], [], []
    structural = []
    for target_kind, ci, row in parsed:
        reason = ("unavailable alternative" if not p.available[row] else
                  "singleton available choice set" if _distribution_size(p, ci) == 1 else None)
        if reason:
            probability = 0. if not p.available[row] else 1.
            values.append(0. if target_kind == "log_probability" else probability)
            gradients.append(torch.zeros_like(theta))
            log_values.append(0. if probability else None)
            log_gradients.append(None)
            odds_values.append(None)
            odds_gradients.append(None)
        else:
            def function(t, ci=ci, row=row):
                return _log_target(t, p, ci, row)
            lp, lg = _value_gradient(function, theta)

            def log_odds(t, ci=ci, row=row):
                rows, logs = _logs(t, p, ci)
                index = rows.index(row)
                return logs[index]-torch.logsumexp(
                    torch.stack([value for i, value in enumerate(logs) if i != index]), 0)

            odds, og = _value_gradient(log_odds, theta)
            probability = math.exp(lp)
            if probability > 1. + 1e-12:
                _error("The numerical choice probability exceeds one.", "numerical_failure")
            values.append(lp if target_kind == "log_probability" else probability)
            gradients.append(lg if target_kind == "log_probability" else probability*lg)
            log_values.append(lp)
            log_gradients.append(lg)
            odds_values.append(odds)
            odds_gradients.append(og)
        structural.append(reason)
    J = torch.stack(gradients)
    factor = _factor(covariance)
    V = _delta(J, factor)
    log_indices = [i for i, value in enumerate(log_gradients) if value is not None]
    joint_names = names+["log_"+names[i] for i in log_indices]+["log_odds_"+names[i] for i in log_indices]
    joint_J = torch.stack(gradients+[log_gradients[i] for i in log_indices]
                          +[odds_gradients[i] for i in log_indices])
    joint_V = _delta(joint_J, factor)
    records = []
    for i, (target_kind, ci, row) in enumerate(parsed):
        se = math.sqrt(float(V[i, i]))
        lower, upper, statistic, pv, status = _normal(values[i], se, z)
        log_se = odds_se = log_lower = log_upper = None
        if i in log_indices:
            position = log_indices.index(i)
            log_se = math.sqrt(float(joint_V[len(names)+position, len(names)+position]))
            offset = len(names)+len(log_indices)+position
            odds_se = math.sqrt(float(joint_V[offset, offset]))
            if target_kind == "probability" and odds_se > 0:
                bounds = [odds_values[i]-z*odds_se, odds_values[i]+z*odds_se]
                if not all(math.isfinite(v) for v in bounds):
                    _error("Transformed probability intervals exceed finite float64 support.",
                           "numerical_failure")
                lower, upper = [float(torch.sigmoid(torch.tensor(v, dtype=DT, device="cpu")))
                                for v in bounds]
                log_lower, log_upper = [-float(torch.nn.functional.softplus(
                    torch.tensor(-v, dtype=DT, device="cpu"))) for v in bounds]
                status = "asymptotic normal log-odds transformed interval"
                if lower == upper:
                    lower = upper = None
                    status += "; probability bounds round together; log bounds retained"
            elif target_kind == "log_probability" and upper is not None:
                upper = min(0., upper)
                status = "asymptotic normal log probability; upper interval projected to zero"
        if structural[i]:
            lower = upper = values[i]
            statistic = pv = None
            status = "known structural probability: "+structural[i]
        records.append([names[i], target_kind, p.case_labels[ci], p.alternative_labels[row],
                        _distribution_size(p, ci), values[i], se, lower, upper, statistic, pv,
                        log_values[i], log_se, log_lower, log_upper, odds_values[i], odds_se, status])
    frames = {
        "predictions": table(records, columns=["target", "kind", "case", "alternative",
            "riskset_size", "estimate", "standard_error", "ci_lower", "ci_upper", "z", "p_value",
            "log_probability", "log_standard_error", "log_ci_lower", "log_ci_upper", "log_odds",
            "log_odds_standard_error", "inference_status"]),
        "jacobian": _jacobian(J, names, p.parameters),
        "covariance": _matrix(V, names),
        "joint_jacobian": _jacobian(joint_J, joint_names, p.parameters),
        "joint_covariance": _matrix(joint_V, joint_names),
        "coefficient_covariance": _matrix(covariance, p.parameters, "parameter"),
    }
    return _output("mprobit_predict", frames, state, p, data, level, resources=resources,
                   kind=kind, joint_estimand_order=joint_names,
                   requested_estimand_order=names,
                   targets=[dict(case=p.case_labels[ci], alternative=p.alternative_labels[row], kind=k)
                            for k, ci, row in parsed])


def _margin_targets(targets, p, attribute):
    if attribute is not None and (not isinstance(attribute, str) or attribute not in p.columns["x"]):
        _error("attribute must name one fitted raw attribute.", "invalid_option")
    if targets is not None and attribute is not None:
        _error("Declare explicit targets or the attribute shorthand, not both.", "invalid_option")
    if targets is None:
        selected = set()
        for rows in p.case_rows:
            available = [row for row in rows if p.available[row]]
            for j in available:
                for k in available:
                    for name in ([attribute] if attribute is not None else p.columns["x"]):
                        selected.add((_label(p.alternative_labels[j]), _label(p.alternative_labels[k]), name))
                        if len(selected) > MAX_TARGETS:
                            _error("Default margins exceed 256 targets; declare an explicit subset.",
                                   "resource_budget")
        targets = [dict(outcome=j[1], changed=k[1], attribute=name) for j, k, name in sorted(selected)]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error("Margins require one through 256 targets.", "resource_budget")
    parsed, seen = [], set()
    for item in targets:
        if not isinstance(item, Mapping) or set(item) != {"outcome", "changed", "attribute"}:
            _error("Margin targets require exactly outcome, changed and attribute.")
        outcome, changed, name = (item[key] for key in ("outcome", "changed", "attribute"))
        if not isinstance(name, str) or name not in p.columns["x"]:
            _error("Margin attribute must be a fitted raw attribute.")
        key = (_label(outcome), _label(changed), name)
        if key in seen:
            _error("Margin targets must be unique.")
        seen.add(key)
        parsed.append((outcome, changed, name))
    return parsed


def _weights(p, weights):
    if weights is None:
        return [1.] * len(p.case_labels)
    if isinstance(weights, Mapping):
        pairs = list(weights.items())
    elif isinstance(weights, (list, tuple)) and all(isinstance(v, Real) and not isinstance(v, bool)
                                                 for v in weights):
        if len(weights) != len(p.case_labels):
            _error("Positional weights must cover every query case exactly.")
        pairs = list(zip(p.case_labels, weights))
    elif isinstance(weights, (list, tuple)) and all(isinstance(v, Mapping)
                                                  and set(v) == {"case", "weight"} for v in weights):
        pairs = [(v["case"], v["weight"]) for v in weights]
    else:
        _error("weights must be positional values, a complete case mapping, or typed case/weight records.")
    result = {}
    for case, value in pairs:
        key = _label(case)
        if (key in result or isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or float(value) <= 0):
            _error("Fixed case weights must be finite, strictly positive and cover cases exactly once.")
        result[key] = float(value)
    if set(result) != {_label(case) for case in p.case_labels}:
        _error("Fixed weights must cover every query case exactly.")
    return [result[_label(case)] for case in p.case_labels]


def _support(p, target):
    outcome, changed, _ = target
    for ci, rows in enumerate(p.case_rows):
        j = next((r for r in rows if _label(p.alternative_labels[r]) == _label(outcome)), None)
        k = next((r for r in rows if _label(p.alternative_labels[r]) == _label(changed)), None)
        if j is None or k is None:
            reason = "excluded: alternative absent from this case"
        elif not p.available[j] or not p.available[k]:
            reason = "excluded: unavailable alternative; raw-cell probability effect structurally zero"
        else:
            reason = "included: simultaneous available-alternative support"
        yield ci, j, k, reason.startswith("included"), reason


def _margin_value(theta, p, ci, j, k, column, elasticity):
    # A one-cell additive perturbation leaves every other raw cell fixed and
    # retains the exact mixed data/parameter derivative through the integral.
    if _distribution_size(p, ci) == 1:
        return theta.sum()*0
    raw = p.X.clone()
    perturbation = torch.zeros((), dtype=DT, device="cpu", requires_grad=True)
    mask = torch.zeros_like(raw)
    mask[k, column] = 1.
    lp = _log_target(theta, p, ci, j, raw+mask*perturbation)
    derivative = torch.autograd.grad(lp, perturbation, create_graph=True)[0]
    return p.X[k, column]*derivative if elasticity else lp.exp()*derivative


@procedure
def mprobit_margins(result, data=None, *, targets=None, attribute=None, elasticity=False,
                    average=False, weights=None, level=0.95,
                    max_work=100_000_000, max_bytes=256_000_000):
    """Own/cross raw-cell effects and optional fixed positive case-weighted averages."""
    if type(elasticity) is not bool or type(average) is not bool:
        _error("elasticity and average must be Boolean flags.", "invalid_option")
    if weights is not None and not average:
        _error("weights requires average=True.", "invalid_option")
    level, z = _confidence(level)
    max_work, max_bytes = _limits(max_work, max_bytes)
    state, p, theta, covariance, admission_work = _query(result, data, max_bytes, max_work)
    parsed = _margin_targets(targets, p, attribute)
    fixed_weights = _weights(p, weights)
    if len(parsed)*len(p.case_labels) > MAX_CASE_ROWS:
        _error("Complete support diagnostics exceed 8192 rows; request fewer targets.",
               "resource_budget")
    counts = []
    for target in parsed:
        column = p.columns["x"].index(target[2])
        count = 0
        for _, _, k, included, _ in _support(p, target):
            if included:
                count += 1
                if elasticity and float(p.X[k, column]) <= 0:
                    _error("Elasticities require a strictly positive changed attribute in every eligible case.",
                           "no_support")
        counts.append(count)
    if any(count == 0 for count in counts):
        _error("A requested alternative pair has no simultaneous available support.", "no_support")
    count = sum(counts)
    resources = _resources(p, len(theta), count+len(parsed)*int(average),
                           max_work=max_work, max_bytes=max_bytes, mixed=True,
                           admission_work=admission_work)
    values, gradients, names, case_meta, support_rows = [], [], [], [], []
    average_values, average_gradients, average_meta = [], [], []
    kind = "elasticity" if elasticity else "effect"
    for index, target in enumerate(parsed):
        outcome, changed, name = target
        column = p.columns["x"].index(name)
        members = list(_support(p, target))
        included_weights = [fixed_weights[ci] for ci, _, _, included, _ in members if included]
        maximum = max(included_weights)
        scaled_sum = math.fsum(w/maximum for w in included_weights)
        denominator = maximum*scaled_sum
        if not math.isfinite(denominator):
            denominator = None
        start = len(values)
        normalized = []
        for ci, j, k, included, reason in members:
            support_rows.append([f"m{index+1}", p.case_labels[ci], included, fixed_weights[ci],
                                 reason, 0. if not included and j is not None and k is not None else None])
            if not included:
                continue
            if elasticity and float(p.X[k, column]) <= 0:
                _error("Elasticities require a strictly positive changed attribute in every eligible case.",
                       "no_support")
            def function(t, ci=ci, j=j, k=k, column=column):
                return _margin_value(t, p, ci, j, k, column, elasticity)
            value, gradient = _value_gradient(function, theta)
            weight = (fixed_weights[ci]/maximum)/scaled_sum
            if weight == 0:
                _error("Positive normalized case weights underflowed.", "numerical_failure")
            normalized.append(weight)
            names.append(f"c{len(values)+1}")
            values.append(value)
            gradients.append(gradient)
            case_meta.append([names[-1], f"m{index+1}", p.case_labels[ci], outcome, changed, name,
                              kind, fixed_weights[ci], weight, _distribution_size(p, ci)])
        if average:
            average_values.append(math.fsum(w*v for w, v in zip(normalized, values[start:])))
            average_gradients.append(sum((w*g for w, g in zip(normalized, gradients[start:])),
                                         torch.zeros_like(theta)))
            average_meta.append([f"m{index+1}", outcome, changed, name, kind, counts[index],
                                 len(p.case_labels), denominator, maximum, scaled_sum])
    names += [f"m{i+1}" for i in range(len(average_values))]
    joint_gradients = gradients+average_gradients
    J = torch.stack(joint_gradients)
    V = _delta(J, _factor(covariance))
    records = [[*meta, values[i], math.sqrt(float(V[i, i])),
                *_normal(values[i], math.sqrt(float(V[i, i])), z)] for i, meta in enumerate(case_meta)]
    inference = ["estimate", "standard_error", "ci_lower", "ci_upper", "z", "p_value", "inference_status"]
    frames = {
        "elasticities" if elasticity else "effects": table(records, columns=[
            "target", "margin", "case", "outcome", "changed", "attribute", "kind", "case_weight",
            "normalized_weight", "riskset_size", *inference]),
    }
    if average:
        rows = [[*meta, average_values[i], math.sqrt(float(V[count+i, count+i])),
                 *_normal(average_values[i], math.sqrt(float(V[count+i, count+i])), z)]
                for i, meta in enumerate(average_meta)]
        frames["averages"] = table(rows, columns=["target", "outcome", "changed", "attribute",
            "kind", "eligible_cases", "query_cases", "weight_denominator", "weight_scale",
            "scaled_weight_denominator", *inference])
    frames.update(
        support=table(support_rows, columns=["margin", "case", "included", "case_weight", "reason",
                                             "structural_effect"]),
        jacobian=_jacobian(J, names, p.parameters), covariance=_matrix(V, names),
        coefficient_covariance=_matrix(covariance, p.parameters, "parameter"),
    )
    return _output("mprobit_margins", frames, state, p, data, level, resources=resources,
                   kind=kind, average=average, joint_estimand_order=names,
                   targets=[dict(outcome=o, changed=c, attribute=a) for o, c, a in parsed],
                   support_policy="each pair averages only simultaneous available support; all query-case reasons and exact scaled weight denominator retained",
                   weights=[dict(case=c, weight=w) for c, w in zip(p.case_labels, fixed_weights)],
                   raw_cell_derivative="one changed alternative attribute cell, with every other cell fixed")
