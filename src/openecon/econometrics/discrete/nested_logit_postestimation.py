"""Sealed two-level nested-logit queries and joint physical-parameter deltas.

Query choice sets and positive case-standardization weights are held fixed.
Alternative effects vary one raw attribute cell, including across nests;
elasticities differentiate log probability without dividing by its rounded
float64 value. Every available alternative contributes to each denominator.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.resources import plan_workspace, tensor_bytes

from .rank_ordered_postestimation import (
    _confidence,
    _delta,
    _error,
    _factor,
    _jacobian,
    _label,
    _matrix,
    _normal,
)

DT = torch.float64
MAX_TARGETS = 256
MAX_CASE_ROWS = 8192
ALT_KINDS = ("probability", "log_probability", "conditional", "log_conditional")
NEST_KINDS = ("nest_probability", "log_nest_probability")


def _query(result, data):
    from .nested_logit import _prepare, _validate_state

    state, p = _validate_state(result)
    if data is not None:
        p = _prepare(
            data,
            {**state["columns"], "cluster": None},
            {**state["options"], "vce": "oim"},
            fixed_dissimilarity=state["fixed_dissimilarity"],
            catalogue=state["catalogue"],
            require_chosen=False,
            for_fit=False,
        )
    theta = torch.tensor(state["fit"]["params"], dtype=DT, device="cpu")
    covariance = torch.tensor(state["fit"]["covariance"], dtype=DT, device="cpu")
    return state, p, theta, covariance


def _log_softmax_pair(values, index):
    """Probability and complementary logs retain tails when softmax rounds one."""
    if len(values) == 1:
        return values[index] * 0, torch.full((), -torch.inf, dtype=DT, device="cpu")
    others = torch.stack([value for i, value in enumerate(values) if i != index])
    odds = values[index] - torch.logsumexp(others, 0)
    return -torch.nn.functional.softplus(-odds), -torch.nn.functional.softplus(odds)


def _distribution(theta, p, ci):
    q = len(p.columns["x"])
    lambdas = {**p.fixed_lambdas, **dict(zip(p.free_nests, theta[q:]))}
    rows = [row for row in p.case_rows[ci] if p.available[row]]
    eta = (p.X[rows] - p.X[rows[0]]) @ theta[:q]
    if not bool(torch.isfinite(eta).all()):
        _error("Query utility differences exceed finite float64 support.", "numerical_failure")
    positions = {row: i for i, row in enumerate(rows)}
    groups = {}
    for row in rows:
        groups.setdefault(p.nest_codes[row], []).append(row)
    inclusive, conditional = [], {}
    group_nests = list(groups)
    for nest, members in groups.items():
        lam = lambdas[nest]
        raw = torch.stack([eta[positions[row]] for row in members])
        pivot_index = int(raw.argmax())
        pivot = raw[pivot_index]
        values = [(value - pivot) / lam for value in raw]
        scaled = torch.stack(values)
        if not bool(torch.isfinite(scaled).all()):
            _error("Scaled query utilities exceed finite float64 support.", "numerical_failure")
        # Keep lower losing mass and its lambda derivative even when the
        # dominant scaled sum would round to one. The utility pivot is not
        # multiplied and divided by lambda, so its derivative cannot cancel.
        tail = sum(
            (value.exp() for i, value in enumerate(values) if i != pivot_index),
            theta.sum() * 0,
        )
        inclusive.append(pivot + lam * torch.log1p(tail))
        for i, row in enumerate(members):
            conditional[row] = _log_softmax_pair(values, i)
    nest_logs = {nest: _log_softmax_pair(inclusive, i) for i, nest in enumerate(group_nests)}
    alternative_logs = {}
    for row in rows:
        lc, lcc = conditional[row]
        ln, lnc = nest_logs[p.nest_codes[row]]
        # 1-P(j)=1-q(j|m)+q(j|m)*(1-P(m)); no rounded subtraction.
        alternative_logs[row] = (lc + ln, torch.logaddexp(lcc, lc + lnc))
    return dict(
        rows=rows,
        groups=groups,
        lambdas=lambdas,
        conditional=conditional,
        nests=nest_logs,
        alternative=alternative_logs,
    )


def _prediction_targets(targets, p):
    if targets is None:
        if sum(sum(bool(p.available[row]) for row in rows) for rows in p.case_rows) > MAX_TARGETS:
            _error(
                "Default predictions exceed 256 targets; declare an explicit subset.",
                "resource_budget",
            )
        targets = [
            dict(case=p.case_labels[ci], alternative=p.alternative_labels[row])
            for ci, rows in enumerate(p.case_rows)
            for row in rows
            if p.available[row]
        ]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error("Predictions require one through 256 explicit targets.", "resource_budget")
    cases = {_label(label): i for i, label in enumerate(p.case_labels)}
    nests = {_label(label): i for i, label in enumerate(p.nest_labels)}
    parsed, seen = [], set()
    for item in targets:
        if not isinstance(item, Mapping) or "case" not in item:
            _error("Prediction targets must declare case, alternative or nest, and optional kind.")
        ci = cases.get(_label(item["case"]))
        if ci is None:
            _error("A requested case is absent from query data.", "no_support")
        kind = item.get("kind", "probability")
        if kind in ALT_KINDS:
            if set(item) - {"case", "alternative", "kind"} or "alternative" not in item:
                _error("Alternative targets accept only case, alternative and optional kind.")
            row = next(
                (
                    r
                    for r in p.case_rows[ci]
                    if _label(p.alternative_labels[r]) == _label(item["alternative"])
                ),
                None,
            )
            if row is None:
                _error("A requested alternative is absent from that query case.", "no_support")
            nest = p.nest_codes[row]
            if not p.available[row] and kind != "probability":
                _error(
                    "Unavailable alternatives have zero probability; log or conditional targets are undefined.",
                    "no_support",
                )
            key = (kind, ci, row)
        elif kind in NEST_KINDS:
            if set(item) != {"case", "nest", "kind"}:
                _error("Nest targets require exactly case, nest and kind.")
            nest = nests.get(_label(item["nest"]))
            if nest is None:
                _error("A requested nest is absent from the fitted nest catalogue.", "no_support")
            row = None
            active = any(p.available[r] and p.nest_codes[r] == nest for r in p.case_rows[ci])
            if not active and kind == "log_nest_probability":
                _error(
                    "Empty nests have zero probability and undefined log probability.", "no_support"
                )
            key = (kind, ci, nest)
        else:
            _error("Unknown prediction kind.", "invalid_option")
        if key in seen:
            _error("Prediction targets must be unique.")
        seen.add(key)
        parsed.append((kind, ci, row, nest))
    return parsed


def _target_logs(theta, p, target):
    kind, ci, row, nest = target
    distribution = _distribution(theta, p, ci)
    if row is not None and not p.available[row]:
        return None, None, "unavailable alternative", 0.0
    if kind in NEST_KINDS:
        if nest not in distribution["nests"]:
            return None, None, "empty nest", 0.0
        lp, lc = distribution["nests"][nest]
        structural = "only active nest" if len(distribution["groups"]) == 1 else None
    elif kind in ("conditional", "log_conditional"):
        lp, lc = distribution["conditional"][row]
        structural = "singleton available nest" if len(distribution["groups"][nest]) == 1 else None
    else:
        lp, lc = distribution["alternative"][row]
        structural = "singleton available choice set" if len(distribution["rows"]) == 1 else None
    return lp, lc, structural, 1.0 if structural else None


def _derivative(function, theta):
    value = function(theta)
    gradient = torch.autograd.functional.jacobian(function, theta, create_graph=False)
    if not bool(torch.isfinite(value).all() & torch.isfinite(gradient).all()):
        _error("Query value or derivative exceeds finite float64 support.", "numerical_failure")
    return float(value), gradient


def _output(method, frames, state, p, level, data, **extra):
    settings = dict(
        method=method,
        level=level,
        device="cpu",
        precision="float64",
        source_fit_checksum=state["checksum"],
        query_inputs=p.inputs,
        columns=p.columns,
        data_source="saved fit" if data is None else "new resident data",
        vce=state["options"]["vce"],
        inference_df=None,
        uncertainty="full joint physical beta and free-dissimilarity covariance delta; query choice sets and standardization weights fixed",
        max_materialized_targets=MAX_TARGETS,
        max_per_case_margin_rows=MAX_CASE_ROWS,
        denominator="all available alternatives and all active fitted nests, including unmaterialized alternatives",
        missing="raise",
        refit=False,
        vendor_parity=False,
        **extra,
    )
    return TableSet(
        frames,
        title="Nested logit saved query",
        method=method,
        contract="nested_logit_postestimation_v1",
        settings=settings,
        source_fit_state=state,
        notes=[settings["uncertainty"], settings["denominator"]],
    )


@procedure
def nlogit_predict(result, *, data=None, targets=None, level=0.95):
    """Alternative, conditional and nest probabilities from a sealed saved fit."""
    level, z = _confidence(level)
    state, p, theta, covariance = _query(result, data)
    parsed = _prediction_targets(targets, p)
    nt, q = len(parsed), len(theta)
    resources = plan_workspace(
        "nested_logit_predict",
        {
            "query_choice_work": tensor_bytes((len(p.alternative_labels), q)) * 8,
            "parameter_covariance_factor": tensor_bytes((q, q)) * 6,
            "joint_jacobians_and_factors": tensor_bytes((nt * 3, q)) * 4,
            "joint_target_covariance": tensor_bytes((nt * 3, nt * 3)) * 4,
        },
    ).record()
    names = [f"p{i + 1}" for i in range(nt)]
    values, gradients, metadata, log_grads, odds_grads = [], [], [], {}, {}
    for i, target in enumerate(parsed):
        kind, ci, row, nest = target
        lp, lc, structural, structural_value = _target_logs(theta, p, target)
        is_log = kind.startswith("log_")
        if structural_value is not None:
            value = 0.0 if is_log else structural_value
            gradient = torch.zeros_like(theta)
            log_value = 0.0 if structural_value == 1 else None
            odds = None
        else:

            def function(t, target=target):
                return _target_logs(t, p, target)[0]

            log_value, log_gradient = _derivative(function, theta)
            if is_log:
                value, gradient = log_value, log_gradient
            else:
                value = math.exp(log_value)
                gradient = value * log_gradient
                log_grads[i] = log_gradient
                log_grads[("value", i)] = log_value
            odds, odds_gradient = _derivative(
                lambda t, target=target: (
                    _target_logs(t, p, target)[0] - _target_logs(t, p, target)[1]
                ),
                theta,
            )
            if not is_log:
                odds_grads[i] = odds_gradient
                odds_grads[("value", i)] = odds
        values.append(value)
        gradients.append(gradient)
        metadata.append((kind, ci, row, nest, structural, log_value, odds))
    J = torch.stack(gradients)
    factor = _factor(covariance)
    V = _delta(J, factor)
    log_indices = [i for i in range(nt) if i in log_grads]
    odds_indices = [i for i in range(nt) if i in odds_grads]
    joint_names = (
        names
        + ["log_" + names[i] for i in log_indices]
        + ["log_odds_" + names[i] for i in odds_indices]
    )
    joint_J = torch.stack(
        gradients + [log_grads[i] for i in log_indices] + [odds_grads[i] for i in odds_indices]
    )
    joint_V = _delta(joint_J, factor)
    rows = []
    for i, (kind, ci, row, nest, structural, log_value, odds) in enumerate(metadata):
        se = math.sqrt(float(V[i, i]))
        lower, upper, statistic, pv, status = _normal(values[i], se, z)
        log_se = odds_se = log_lower = log_upper = None
        if i in log_grads:
            li = nt + log_indices.index(i)
            log_se = math.sqrt(float(joint_V[li, li]))
        if i in odds_grads:
            oi = nt + len(log_indices) + odds_indices.index(i)
            odds_se = math.sqrt(float(joint_V[oi, oi]))
            if odds_se > 0:
                bounds = [odds - z * odds_se, odds + z * odds_se]
                if not all(math.isfinite(v) for v in bounds):
                    _error(
                        "Transformed probability interval exceeds finite float64 support.",
                        "numerical_failure",
                    )
                lower, upper = [
                    float(torch.sigmoid(torch.tensor(v, dtype=DT, device="cpu"))) for v in bounds
                ]
                log_lower, log_upper = [
                    -float(torch.nn.functional.softplus(torch.tensor(-v, dtype=DT, device="cpu")))
                    for v in bounds
                ]
                status = "asymptotic normal log-odds transformed interval"
                if lower == upper:
                    lower = upper = None
                    status += "; probability bounds round together; log bounds retained"
        elif kind.startswith("log_") and lower is not None:
            upper = min(0.0, upper)
            status = "asymptotic normal log probability; upper interval projected to zero"
        if structural:
            lower = upper = values[i]
            statistic = pv = None
            status = "known structural probability: " + structural
        rows.append(
            [
                names[i],
                kind,
                p.case_labels[ci],
                p.alternative_labels[row] if row is not None else None,
                p.nest_labels[nest],
                sum(bool(p.available[r]) for r in p.case_rows[ci]),
                values[i],
                se,
                lower,
                upper,
                statistic,
                pv,
                log_value,
                log_se,
                log_lower,
                log_upper,
                odds,
                odds_se,
                status,
            ]
        )
    frames = {
        "predictions": table(
            rows,
            columns=[
                "target",
                "kind",
                "case",
                "alternative",
                "nest",
                "riskset_size",
                "estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "z",
                "p_value",
                "log_probability",
                "log_standard_error",
                "log_ci_lower",
                "log_ci_upper",
                "log_odds",
                "log_odds_standard_error",
                "inference_status",
            ],
        ),
        "covariance": _matrix(V, names),
        "jacobian": _jacobian(J, names, p.parameters),
        "joint_covariance": _matrix(joint_V, joint_names),
        "joint_jacobian": _jacobian(joint_J, joint_names, p.parameters),
        "coefficient_covariance": _matrix(covariance, p.parameters, "parameter"),
    }
    return _output(
        "nlogit_predict",
        frames,
        state,
        p,
        level,
        data,
        targets=[dict(item) for item in targets]
        if targets is not None
        else "all available alternatives",
        resources=resources,
        joint_estimand_order=joint_names,
        structural_log_probability="undefined at probability zero; log targets rejected",
    )


def _margin_targets(targets, p):
    if targets is None:
        choices = set()
        for rows in p.case_rows:
            risk = [r for r in rows if p.available[r]]
            for j in risk:
                for k in risk:
                    for attribute in p.columns["x"]:
                        choices.add(
                            (
                                _label(p.alternative_labels[j]),
                                _label(p.alternative_labels[k]),
                                attribute,
                            )
                        )
                        if len(choices) > MAX_TARGETS:
                            _error(
                                "Default margins exceed 256 targets; declare an explicit subset.",
                                "resource_budget",
                            )
        targets = [dict(outcome=j[1], changed=k[1], attribute=v) for j, k, v in sorted(choices)]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error("Margins require one through 256 explicit targets.", "resource_budget")
    parsed, seen = [], set()
    for item in targets:
        if not isinstance(item, Mapping) or set(item) != {"outcome", "changed", "attribute"}:
            _error("Margin target fields must be exactly outcome, changed and attribute.")
        outcome, changed, attribute = (item[k] for k in ("outcome", "changed", "attribute"))
        if not isinstance(attribute, str) or attribute not in p.columns["x"]:
            _error("Margin attribute must be one of the fitted raw attributes.")
        key = (_label(outcome), _label(changed), attribute)
        if key in seen:
            _error("Margin targets must be unique.")
        seen.add(key)
        parsed.append((outcome, changed, attribute))
    return parsed


def _weights(p, case_weights):
    if case_weights is None:
        return [1.0] * len(p.case_labels)
    if isinstance(case_weights, Mapping):
        pairs = list(case_weights.items())
    elif isinstance(case_weights, (list, tuple)) and all(
        isinstance(v, Real) and not isinstance(v, bool) for v in case_weights
    ):
        if len(case_weights) != len(p.case_labels):
            _error("Positional case_weights must cover every query case exactly.")
        pairs = list(zip(p.case_labels, case_weights))
    elif isinstance(case_weights, (list, tuple)) and all(
        isinstance(v, Mapping) and set(v) == {"case", "weight"} for v in case_weights
    ):
        pairs = [(v["case"], v["weight"]) for v in case_weights]
    else:
        _error(
            "case_weights must be positional weights, a complete case mapping, or typed case/weight records."
        )
    weights = {}
    for case, value in pairs:
        key = _label(case)
        if key in weights:
            _error("case_weights must cover every query case exactly once.")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 1e-12 <= value <= 1e12
        ):
            _error("Fixed standardization case weights must be finite between 1e-12 and 1e12.")
        weights[key] = float(value)
    if set(weights) != {_label(c) for c in p.case_labels}:
        _error("case_weights must cover every query case exactly.")
    return [weights[_label(c)] for c in p.case_labels]


def _margin_value(theta, p, ci, j, k, r, kind):
    d = _distribution(theta, p, ci)
    lc, lcc = d["conditional"][k]
    ln, lnc = d["nests"][p.nest_codes[k]]
    pk = (lc + ln).exp()
    if j == k:
        # Own derivative stays nonzero when q or P(m) rounds one.
        bracket = lcc.exp() / d["lambdas"][p.nest_codes[k]] + lc.exp() * lnc.exp()
    elif p.nest_codes[j] == p.nest_codes[k]:
        bracket = -(1 / d["lambdas"][p.nest_codes[k]] - 1) * lc.exp() - pk
    else:
        bracket = -pk
    if kind == "elasticity":
        return theta[r] * p.X[k, r] * bracket
    return theta[r] * d["alternative"][j][0].exp() * bracket


@procedure
def nlogit_margins(
    result, *, data=None, targets=None, kind="effect", case_weights=None, level=0.95
):
    """Analytical own/cross effects, elasticities and fixed-weight average margins."""
    if kind not in ("effect", "elasticity"):
        _error("kind must be effect or elasticity.", "invalid_option")
    level, z = _confidence(level)
    state, p, theta, covariance = _query(result, data)
    parsed, weights = _margin_targets(targets, p), _weights(p, case_weights)
    if len(parsed) * len(p.case_labels) > MAX_CASE_ROWS:
        _error(
            "Complete margin support diagnostics exceed 8192 rows; request fewer targets.",
            "resource_budget",
        )
    support, support_rows = [], []
    for index, (outcome, changed, attribute) in enumerate(parsed):
        members = []
        for ci, rows in enumerate(p.case_rows):
            j = next((r for r in rows if _label(p.alternative_labels[r]) == _label(outcome)), None)
            k = next((r for r in rows if _label(p.alternative_labels[r]) == _label(changed)), None)
            included = j is not None and k is not None and p.available[j] and p.available[k]
            structural_effect = structural_elasticity = None
            if j is None or k is None:
                reason = "excluded: alternative absent from this case"
            elif not included:
                structural_effect = 0.0
                structural_elasticity = (
                    0.0
                    if p.available[j] and float(p.X[k, p.columns["x"].index(attribute)]) > 0
                    else None
                )
                reason = "excluded: unavailable alternative; structural effect zero; elasticity requires positive outcome probability and changed attribute"
            else:
                if kind == "elasticity" and float(p.X[k, p.columns["x"].index(attribute)]) <= 0:
                    _error(
                        "Elasticities require a strictly positive changed attribute in every eligible case.",
                        "no_support",
                    )
                reason = "included: simultaneous available-alternative support"
                members.append((ci, j, k))
            support_rows.append(
                [
                    f"m{index + 1}",
                    p.case_labels[ci],
                    bool(included),
                    weights[ci],
                    reason,
                    structural_effect,
                    structural_elasticity,
                ]
            )
        if not members:
            _error(
                "A requested alternative pair has no simultaneous available support.", "no_support"
            )
        support.append(members)
    nt, nr, q = len(parsed), sum(len(m) for m in support), len(theta)
    resources = plan_workspace(
        "nested_logit_margins",
        {
            "query_choice_work": tensor_bytes((len(p.alternative_labels), q)) * 8,
            "parameter_covariance_factor": tensor_bytes((q, q)) * 6,
            "per_case_jacobians_and_factors": tensor_bytes((nr, q)) * 4,
            "average_jacobians_and_factors": tensor_bytes((nt, q)) * 4,
            "joint_average_covariance": tensor_bytes((nt, nt)) * 3,
            "complete_support_diagnostics": tensor_bytes((len(support_rows), 7)) * 2,
        },
    ).record()
    averages, average_gradients, case_values, case_gradients, case_meta, summary = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, ((outcome, changed, attribute), members) in enumerate(zip(parsed, support)):
        r = p.columns["x"].index(attribute)
        denominator = math.fsum(weights[ci] for ci, _, _ in members)
        average, avg_gradient = 0.0, torch.zeros_like(theta)
        for ci, j, k in members:
            value, gradient = _derivative(
                lambda t, ci=ci, j=j, k=k, r=r: _margin_value(t, p, ci, j, k, r, kind), theta
            )
            normalized = weights[ci] / denominator
            average += normalized * value
            avg_gradient += normalized * gradient
            d = _distribution(theta, p, ci)
            case_meta.append(
                [
                    f"c{len(case_meta) + 1}",
                    f"m{i + 1}",
                    p.case_labels[ci],
                    outcome,
                    changed,
                    attribute,
                    weights[ci],
                    normalized,
                    len(d["rows"]),
                    float(d["alternative"][j][0].exp()),
                    float(d["alternative"][k][0].exp()),
                ]
            )
            case_values.append(value)
            case_gradients.append(gradient)
        averages.append(average)
        average_gradients.append(avg_gradient)
        summary.append(
            [
                f"m{i + 1}",
                outcome,
                changed,
                attribute,
                kind,
                len(members),
                len(p.case_labels),
                denominator,
            ]
        )
    J, CJ = torch.stack(average_gradients), torch.stack(case_gradients)
    factor = _factor(covariance)
    V = _delta(J, factor)
    case_variances = (CJ @ factor).square().sum(1)
    if not bool(torch.isfinite(case_variances).all()) or not all(
        math.isfinite(v) for v in averages
    ):
        _error(
            "A margin or its joint delta covariance exceeds finite float64 support.",
            "numerical_failure",
        )
    average_rows, case_rows = [], []
    for i, meta in enumerate(summary):
        se = math.sqrt(float(V[i, i]))
        average_rows.append([*meta, averages[i], se, *_normal(averages[i], se, z)])
    for i, meta in enumerate(case_meta):
        se = math.sqrt(float(case_variances[i]))
        case_rows.append([*meta, case_values[i], se, *_normal(case_values[i], se, z)])
    inference = [
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "z",
        "p_value",
        "inference_status",
    ]
    names = [meta[0] for meta in summary]
    frames = {
        "margins": table(
            average_rows,
            columns=[
                "target",
                "outcome",
                "changed",
                "attribute",
                "kind",
                "n_support_cases",
                "n_query_cases",
                "support_weight",
                *inference,
            ],
        ),
        "per_case": table(
            case_rows,
            columns=[
                "case_target",
                "target",
                "case",
                "outcome",
                "changed",
                "attribute",
                "case_weight",
                "normalized_support_weight",
                "riskset_size",
                "outcome_probability",
                "changed_probability",
                *inference,
            ],
        ),
        "support": table(
            support_rows,
            columns=[
                "target",
                "case",
                "included_in_average",
                "case_weight",
                "support_status",
                "structural_effect",
                "structural_elasticity",
            ],
        ),
        "covariance": _matrix(V, names),
        "jacobian": _jacobian(J, names, p.parameters),
        "per_case_jacobian": _jacobian(CJ, [meta[0] for meta in case_meta], p.parameters),
        "coefficient_covariance": _matrix(covariance, p.parameters, "parameter"),
    }
    return _output(
        "nlogit_margins",
        frames,
        state,
        p,
        level,
        data,
        kind=kind,
        targets=[dict(outcome=o, changed=c, attribute=a) for o, c, a in parsed],
        case_weights=[dict(case=c, weight=w) for c, w in zip(p.case_labels, weights)],
        standardization="fixed positive case weights over explicitly reported pair support; not fit weights",
        averaging_population="pair-conditional eligible cases; excluded structural zeros do not enter the average denominator",
        support_policy="every target and query case audited; unavailable attributes have structural effect zero; absent alternatives have no defined derivative",
        per_case_joint_covariance="lossless full cross-case/cross-target factorization: per_case_jacobian @ coefficient_covariance @ per_case_jacobian.T",
        resources=resources,
    )
