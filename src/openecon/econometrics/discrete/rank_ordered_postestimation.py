"""Saved rank-ordered logit queries with complete joint delta inference.

Choice probabilities condition on the alternatives remaining at a declared
stage. Ranking probabilities refer to the observed strict ranked prefix, not
to a completion of its unranked tail. Attribute effects and elasticities hold
the query choice sets, observed prefixes and standardization weights fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.common import procedure
from openecon.resources import plan_workspace, tensor_bytes

DT = torch.float64
MAX_TARGETS = 256
MAX_CASE_ROWS = 8192


def _error(message, code="invalid_input"):
    raise AnalysisError(code, message)


def _confidence(level):
    if isinstance(level, bool) or not isinstance(level, Real) or not 0 < level < 1:
        _error(
            "level must be a finite probability strictly between zero and one.", "invalid_option"
        )
    level = float(level)
    z = float(torch.special.ndtri(torch.tensor((1 + level) / 2, dtype=DT, device="cpu")))
    if not math.isfinite(z):
        _error("level is too close to one for float64 normal inference.", "invalid_option")
    return level, z


def _label(value):
    if isinstance(value, bool):
        _error("Target case/alternative labels must be strings or exact integers.")
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, Integral) and abs(int(value)) <= 2**53:
        return ("int", int(value))
    _error("Target case/alternative labels must be strings or exact integers.")


def _stage(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 20:
        _error("Target stage must be an integer from one through twenty.")
    return int(value)


def _query(result, data, mode):
    if mode not in ("first", "stages"):
        _error("mode must be first or stages.", "invalid_option")
    # Validation replays the sealed fit and its actual stationarity; no optimizer.
    from .rank_ordered import _prepare, _validate_state

    state, prepared = _validate_state(result)
    if data is not None:
        columns = {**state["columns"], "cluster": None}
        options = {**state["options"], "vce": "oim"}
        prepared = _prepare(data, columns, options, require_rank=(mode == "stages"), for_fit=False)
    beta = torch.tensor(state["fit"]["beta"], dtype=DT, device="cpu")
    covariance = torch.tensor(state["fit"]["covariance"], dtype=DT, device="cpu")
    return state, prepared, beta, covariance


def _risksets(p, mode):
    sets, selected = {}, {}
    for ci, rows in enumerate(p.case_rows):
        remaining = [i for i in rows if p.available[i]]
        if mode == "first":
            sets[(ci, 1)] = remaining
            continue
        ordered = sorted((i for i in remaining if p.rank[i] > 0), key=lambda i: p.rank[i])
        if not ordered:
            _error("Stages queries require a nonempty strict ranked prefix in every case.")
        selected[ci] = []
        for step, row in enumerate(ordered, 1):
            sets[(ci, step)] = list(remaining)
            selected[ci].append((step, row))
            remaining.remove(row)
    return sets, selected


def _choice(beta, X, rows, outcome):
    """Log odds avoid subtracting a rounded probability from one."""
    if len(rows) == 1:
        zero = torch.zeros_like(beta)
        return dict(
            value=1.0,
            log_probability=0.0,
            log_odds=None,
            jacobian=zero,
            log_jacobian=zero,
            odds_jacobian=None,
            singleton=True,
        )
    x = X[rows] - X[rows[0]]
    eta = x @ beta
    if not bool(torch.isfinite(eta).all()):
        _error("Query utility differences exceed finite float64 support.", "numerical_failure")
    j = rows.index(outcome)
    other = [i for i in range(len(rows)) if i != j]
    log_odds = eta[j] - torch.logsumexp(eta[other], dim=0)
    log_probability = -torch.nn.functional.softplus(-log_odds)
    probability = torch.sigmoid(log_odds)
    complement = torch.sigmoid(-log_odds)
    other_mean = torch.softmax(eta[other], dim=0) @ x[other]
    odds_jacobian = x[j] - other_mean
    log_jacobian = complement * odds_jacobian
    jacobian = probability * log_jacobian
    if not bool(torch.isfinite(log_odds) & torch.isfinite(log_probability)):
        _error(
            "Query log probability or log odds exceeds finite float64 support.", "numerical_failure"
        )
    return dict(
        value=float(probability),
        log_probability=float(log_probability),
        log_odds=float(log_odds),
        jacobian=jacobian,
        log_jacobian=log_jacobian,
        odds_jacobian=odds_jacobian,
        singleton=False,
    )


def _factor(covariance):
    covariance = (covariance + covariance.T) / 2
    diagonal = covariance.diag()
    if not bool(torch.isfinite(covariance).all()) or bool((diagonal < 0).any()):
        _error(
            "The saved coefficient covariance is not finite positive semidefinite.",
            "invalid_covariance",
        )
    positive = diagonal > 0
    if bool((~positive).any()) and bool((covariance[~positive] != 0).any()):
        _error("A zero-variance coefficient has nonzero covariance.", "invalid_covariance")
    scale = torch.where(positive, diagonal.sqrt(), torch.ones_like(diagonal))
    correlation = covariance / scale[:, None] / scale[None, :]
    eigenvalues, vectors = torch.linalg.eigh(correlation)
    if float(eigenvalues.min()) < -1e-10 * max(1.0, float(eigenvalues.abs().max())):
        _error(
            "The saved coefficient covariance is not positive semidefinite.", "invalid_covariance"
        )
    return scale[:, None] * vectors * eigenvalues.clamp_min(0).sqrt()[None, :]


def _delta(jacobian, factor):
    transformed = jacobian @ factor
    covariance = transformed @ transformed.T
    if not bool(torch.isfinite(jacobian).all() & torch.isfinite(covariance).all()):
        _error(
            "Query derivative or covariance exceeds finite float64 support.", "numerical_failure"
        )
    return covariance


def _matrix(value, names, marker="target"):
    while marker in names:
        marker = "_" + marker
    return table(
        [[name, *row] for name, row in zip(names, value.tolist())], columns=[marker, *names]
    )


def _jacobian(value, names, parameters):
    marker = "target"
    while marker in parameters:
        marker = "_" + marker
    return table(
        [[name, *row] for name, row in zip(names, value.tolist())], columns=[marker, *parameters]
    )


def _normal(value, se, z):
    if se == 0:
        return None, None, None, None, "unavailable: zero first-order delta variance"
    statistic = value / se
    if not math.isfinite(statistic):
        _error("A normal statistic exceeds finite float64 support.", "numerical_failure")
    probability = math.erfc(abs(statistic) / math.sqrt(2))
    lower, upper = value - z * se, value + z * se
    if not all(math.isfinite(v) for v in (lower, upper)):
        _error("Normal inference exceeds finite float64 support.", "numerical_failure")
    return lower, upper, statistic, probability, "asymptotic normal"


def _output(method, frames, state, p, mode, level, data, **extra):
    settings = dict(
        method=method,
        mode=mode,
        level=level,
        device="cpu",
        precision="float64",
        source_fit_checksum=state["checksum"],
        query_inputs=p.inputs,
        columns=p.columns,
        data_source="saved fit" if data is None else "new resident data",
        vce=state["options"]["vce"],
        inference_df=None,
        uncertainty="full joint coefficient-covariance delta; fixed query sets, ranked prefixes and standardization weights",
        max_materialized_targets=MAX_TARGETS,
        max_per_case_margin_rows=MAX_CASE_ROWS,
        denominator="all available remaining alternatives, including unmaterialized alternatives",
        missing="raise",
        refit=False,
        vendor_parity=False,
        **extra,
    )
    return TableSet(
        frames,
        title="Rank-ordered logit saved query",
        method=method,
        contract="rank_ordered_postestimation_v1",
        settings=settings,
        source_fit_state=state,
        notes=[settings["uncertainty"], settings["denominator"]],
    )


def _prediction_targets(targets, p, mode, sets, selected):
    if targets is None:
        if (
            sum(len(rows) for rows in sets.values()) + (len(selected) if mode == "stages" else 0)
            > MAX_TARGETS
        ):
            _error(
                "Default predictions exceed256 targets; declare an explicit subset.",
                "resource_budget",
            )
        targets = [
            dict(case=p.case_labels[ci], stage=stage, alternative=p.alternative_labels[row])
            for (ci, stage), rows in sets.items()
            for row in rows
        ]
        if mode == "stages":
            targets += [dict(case=p.case_labels[ci], kind="ranking") for ci in selected]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error(
            "Prediction requires one through256 explicit targets; select fewer rows without changing risksets.",
            "resource_budget",
        )
    cases = {_label(value): i for i, value in enumerate(p.case_labels)}
    result, seen = [], set()
    for item in targets:
        if isinstance(item, (tuple, list)):
            if mode == "first" and len(item) == 2:
                item = dict(case=item[0], stage=1, alternative=item[1])
            elif mode == "stages" and len(item) == 3:
                item = dict(case=item[0], stage=item[1], alternative=item[2])
            else:
                _error(
                    "Choice tuples must be(case,alternative) for first or(case,stage,alternative) for stages."
                )
        if not isinstance(item, Mapping) or "case" not in item:
            _error(
                "Each prediction target must declare its case and choice alternative or ranking kind."
            )
        ci = cases.get(_label(item["case"]))
        if ci is None:
            _error("A requested prediction case is absent from the query data.", "no_support")
        if item.get("kind") == "ranking":
            if set(item) != {"case", "kind"} or mode != "stages":
                _error("Ranking targets require stages mode and only case/kind fields.")
            key = ("ranking", ci)
            row = None
            stage = None
        else:
            if (
                set(item) - {"case", "stage", "alternative", "kind"}
                or "alternative" not in item
                or item.get("kind", "choice") != "choice"
            ):
                _error(
                    "Choice targets accept only case/stage/alternative and optional kind=choice."
                )
            stage = _stage(item.get("stage", 1))
            if mode == "first" and stage != 1:
                _error("First mode permits only stage1.")
            risk = sets.get((ci, stage))
            if risk is None:
                _error("A requested stage has no observed prefix support.", "no_support")
            row = next(
                (
                    i
                    for i in p.case_rows[ci]
                    if _label(p.alternative_labels[i]) == _label(item["alternative"])
                ),
                None,
            )
            if row is None:
                _error("A requested alternative is absent from that query case.", "no_support")
            key = ("choice", ci, stage, row)
        if key in seen:
            _error("Prediction targets must be unique.")
        seen.add(key)
        result.append((key[0], ci, stage, row))
    return result


@procedure
def rologit_predict(result, *, data=None, mode="first", targets=None, level=0.95):
    """Conditional choice/observed-prefix probabilities from a sealed saved fit."""
    level, z = _confidence(level)
    state, p, beta, covariance = _query(result, data, mode)
    sets, selected = _risksets(p, mode)
    targets = _prediction_targets(targets, p, mode, sets, selected)
    nt, q = len(targets), len(beta)
    plan = plan_workspace(
        "rank_ordered_predict",
        {
            "query_inputs_and_choice_work": tensor_bytes((len(p.alternative_labels), q)) * 3,
            "coefficient_covariance_factor": tensor_bytes((q, q)) * 6,
            "all_scale_jacobians_and_factors": tensor_bytes((nt * 3, q)) * 4,
            "individual_and_joint_covariances": tensor_bytes((nt * 3, nt * 3)) * 4,
        },
    ).record()
    values, gradients, logs, log_gradients, odds, odds_gradients, names = [], [], [], [], [], [], []
    metadata = []
    for index, (kind, ci, stage, row) in enumerate(targets):
        name = f"p{index + 1}"
        if kind == "choice":
            risk = sets[(ci, stage)]
            if row not in risk:
                zero = torch.zeros_like(beta)
                item = dict(
                    value=0.0,
                    log_probability=None,
                    log_odds=None,
                    jacobian=zero,
                    log_jacobian=None,
                    odds_jacobian=None,
                    singleton=False,
                )
                structural = (
                    "unavailable alternative"
                    if not p.available[row]
                    else "already removed by ranked prefix"
                )
            else:
                item = _choice(beta, p.X, risk, row)
                structural = "singleton remaining riskset" if item["singleton"] else None
            odds.append(item["log_odds"])
            odds_gradients.append(item["odds_jacobian"])
        else:
            items = [_choice(beta, p.X, sets[(ci, s)], r) for s, r in selected[ci]]
            log_value = sum(item["log_probability"] for item in items)
            log_gradient = sum((item["log_jacobian"] for item in items), torch.zeros_like(beta))
            if not math.isfinite(log_value) or not bool(torch.isfinite(log_gradient).all()):
                _error(
                    "The ranked-prefix log probability or derivative exceeds finite float64 support.",
                    "numerical_failure",
                )
            value = math.exp(log_value)
            item = dict(
                value=value,
                log_probability=log_value,
                jacobian=value * log_gradient,
                log_jacobian=log_gradient,
                singleton=False,
            )
            odds.append(None)
            odds_gradients.append(None)
            risk = [i for i in p.case_rows[ci] if p.available[i]]
            structural = None
        values.append(item["value"])
        gradients.append(item["jacobian"])
        logs.append(item["log_probability"])
        log_gradients.append(item["log_jacobian"])
        names.append(name)
        metadata.append(
            [
                name,
                kind,
                p.case_labels[ci],
                stage,
                p.alternative_labels[row] if row is not None else None,
                len(risk),
                [p.alternative_labels[i] for i in risk],
                structural,
            ]
        )
    li = [i for i, value in enumerate(log_gradients) if value is not None]
    J = torch.stack(gradients)
    LJ = (
        torch.stack([log_gradients[i] for i in li])
        if li
        else torch.empty((0, q), dtype=DT, device="cpu")
    )
    factor = _factor(covariance)
    V = _delta(J, factor)
    LV = _delta(LJ, factor)
    oi = [i for i, value in enumerate(odds_gradients) if value is not None]
    OJ = (
        torch.stack([odds_gradients[i] for i in oi])
        if oi
        else torch.empty((0, len(beta)), dtype=DT, device="cpu")
    )
    OV = _delta(OJ, factor)
    joint_names = names + ["log_" + names[i] for i in li] + ["log_odds_" + names[i] for i in oi]
    joint_jacobian = torch.cat([J, LJ, OJ], dim=0)
    joint_covariance = _delta(joint_jacobian, factor)
    log_se_by_target = {i: math.sqrt(float(LV[j, j])) for j, i in enumerate(li)}
    odds_se = {i: math.sqrt(float(OV[j, j])) for j, i in enumerate(oi)}
    rows = []
    for i, meta in enumerate(metadata):
        se, log_se = math.sqrt(float(V[i, i])), log_se_by_target.get(i)
        lo = hi = llo = lhi = olo = ohi = None
        inference = "unavailable: zero first-order delta variance"
        if meta[-1]:
            lo = hi = values[i]
            if values[i] == 1:
                llo = lhi = 0.0
            inference = "known structural probability: " + meta[-1]
        elif odds[i] is not None and odds_se[i] > 0:
            olo, ohi = odds[i] - z * odds_se[i], odds[i] + z * odds_se[i]
            lo, hi = (
                float(torch.sigmoid(torch.tensor(v, dtype=DT, device="cpu"))) for v in (olo, ohi)
            )
            llo, lhi = (
                -float(torch.nn.functional.softplus(torch.tensor(-v, dtype=DT, device="cpu")))
                for v in (olo, ohi)
            )
            inference = "asymptotic normal log-odds transformed interval"
        elif log_se is not None and log_se > 0:
            llo, lhi = logs[i] - z * log_se, min(0.0, logs[i] + z * log_se)
            lo, hi = math.exp(llo), math.exp(lhi)
            inference = "asymptotic normal log-probability projected interval"
        if lo == hi and not meta[-1]:
            lo = hi = None
            if llo is not None:
                inference += "; probability bounds round together; log bounds retained"
        if any(
            value is not None and not math.isfinite(value) for value in (lo, hi, llo, lhi, olo, ohi)
        ):
            _error(
                "A transformed probability interval exceeds finite float64 support.",
                "numerical_failure",
            )
        rows.append(
            [
                *meta[:-1],
                values[i],
                se,
                lo,
                hi,
                logs[i],
                log_se,
                llo,
                lhi,
                odds[i],
                odds_se.get(i),
                olo,
                ohi,
                inference,
            ]
        )
    parameters = list(p.columns["x"])
    frames = {
        "predictions": table(
            rows,
            columns=[
                "target",
                "kind",
                "case",
                "stage",
                "alternative",
                "riskset_size",
                "riskset",
                "probability",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "log_probability",
                "log_standard_error",
                "log_ci_lower",
                "log_ci_upper",
                "log_odds",
                "log_odds_standard_error",
                "log_odds_ci_lower",
                "log_odds_ci_upper",
                "inference_status",
            ],
        ),
        "covariance": _matrix(V, names),
        "log_probability_covariance": _matrix(LV, [names[i] for i in li]),
        "joint_covariance": _matrix(joint_covariance, joint_names),
        "joint_jacobian": _jacobian(joint_jacobian, joint_names, parameters),
        "log_odds_covariance": _matrix(OV, [names[i] for i in oi]),
        "jacobian": _jacobian(J, names, parameters),
        "log_probability_jacobian": _jacobian(LJ, [names[i] for i in li], parameters),
        "log_odds_jacobian": _jacobian(OJ, [names[i] for i in oi], parameters),
        "coefficient_covariance": _matrix(covariance, parameters, "parameter"),
    }
    return _output(
        "rologit_predict",
        frames,
        state,
        p,
        mode,
        level,
        data,
        targets=[
            dict(zip(("kind", "case", "stage", "alternative"), (row[1], row[2], row[3], row[4])))
            for row in metadata
        ],
        ranking_target="probability of the observed strict prefix; no unranked-tail completion",
        resources=plan,
        joint_estimand_order=joint_names,
        structural_log_probability="undefined at known probability zero; excluded from log-scale covariance",
    )


def _margin_targets(targets, p, mode, sets):
    if targets is None:
        choices = set()
        for (_, stage), risk in sets.items():
            for j in risk:
                for k in risk:
                    for variable in p.columns["x"]:
                        choices.add(
                            (
                                stage,
                                _label(p.alternative_labels[j]),
                                _label(p.alternative_labels[k]),
                                variable,
                            )
                        )
                        if len(choices) > MAX_TARGETS:
                            _error(
                                "Default margins exceed256 targets; declare an explicit subset.",
                                "resource_budget",
                            )
        targets = [(s, j[1], k[1], v) for s, j, k, v in sorted(choices)]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= MAX_TARGETS:
        _error(
            "Margins require one through256 explicit targets; select fewer alternative/variable pairs.",
            "resource_budget",
        )
    result, seen = [], set()
    for item in targets:
        if isinstance(item, Mapping):
            if set(item) != {"stage", "alternative", "changed_alternative", "variable"}:
                _error(
                    "Margin target fields must be stage/alternative/changed_alternative/variable."
                )
            item = tuple(
                item[k] for k in ("stage", "alternative", "changed_alternative", "variable")
            )
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            _error(
                "Margin targets must be(stage,outcome alternative,changed alternative,variable)."
            )
        stage, outcome, changed, variable = item
        stage = _stage(stage)
        if mode == "first" and stage != 1:
            _error("First-mode margins permit only stage1.")
        if not isinstance(variable, str) or variable not in p.columns["x"]:
            _error("The margin variable must be a fitted alternative-varying attribute.")
        key = (stage, _label(outcome), _label(changed), variable)
        if key in seen:
            _error("Margin targets must be unique.")
        seen.add(key)
        result.append((stage, outcome, changed, variable))
    return result


def _weights(p, case_weights):
    if case_weights is None:
        return [1.0] * len(p.case_labels)
    if not isinstance(case_weights, Mapping):
        _error("case_weights must map every query case to a fixed positive standardization weight.")
    weights = {_label(k): v for k, v in case_weights.items()}
    if set(weights) != {_label(v) for v in p.case_labels}:
        _error("case_weights must cover every query case exactly.")
    result = []
    for case in p.case_labels:
        value = weights[_label(case)]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 1e-12 <= value <= 1e12
        ):
            _error("Fixed standardization case weights must be finite between1e-12 and1e12.")
        result.append(float(value))
    return result


def _margin(beta, X, risk, j, k, r, kind):
    cj, ck = _choice(beta, X, risk, j), _choice(beta, X, risk, k)
    pj, pk = cj["value"], ck["value"]
    own = j == k
    contrast = (
        float(torch.sigmoid(torch.tensor(-ck["log_odds"], dtype=DT, device="cpu")))
        if own and not ck["singleton"]
        else float(own) - pk
    )
    unit = torch.zeros_like(beta)
    unit[r] = 1
    if kind == "effect":
        base = pj * contrast
        value = float(beta[r]) * base
        gradient = unit * base + beta[r] * (cj["jacobian"] * contrast - pj * ck["jacobian"])
    else:
        attribute = float(X[k, r])
        if attribute <= 0:
            _error(
                "Elasticities require a strictly positive changed alternative attribute in every support case.",
                "no_support",
            )
        value = float(beta[r]) * attribute * contrast
        gradient = attribute * (unit * contrast - beta[r] * ck["jacobian"])
    return value, gradient, pj, pk


@procedure
def rologit_margins(
    result, *, data=None, mode="first", targets=None, kind="effect", case_weights=None, level=0.95
):
    """Own/cross attribute derivatives and fixed-case weighted average margins."""
    if kind not in ("effect", "elasticity"):
        _error("kind must be effect or elasticity.", "invalid_option")
    level, z = _confidence(level)
    state, p, beta, covariance = _query(result, data, mode)
    sets, _ = _risksets(p, mode)
    targets = _margin_targets(targets, p, mode, sets)
    weights = _weights(p, case_weights)
    support = []
    # Materialize a complete inclusion audit without assigning nonexistent
    # alternatives attributes or averaging structural zeros into eligible AMEs.
    if len(targets) * len(p.case_labels) > MAX_CASE_ROWS:
        _error(
            "Complete margin support diagnostics exceed8192 rows; request fewer targets.",
            "resource_budget",
        )
    support_rows = []
    for index, (stage, outcome, changed, variable) in enumerate(targets):
        members = []
        for ci, rows in enumerate(p.case_rows):
            risk = sets.get((ci, stage))
            j = next((i for i in rows if _label(p.alternative_labels[i]) == _label(outcome)), None)
            k = next((i for i in rows if _label(p.alternative_labels[i]) == _label(changed)), None)
            included = risk is not None and j in risk and k in risk
            effect, elasticity = None, None
            if risk is None:
                reason = "excluded: no observed stage support"
            elif j is None or k is None:
                reason = "excluded: alternative absent from this case"
            elif included:
                reason = "included: simultaneous remaining-riskset support"
                members.append((ci, risk, j, k))
            else:
                effect = 0.0
                reason = "excluded: known unavailable or already removed alternative; structural attribute effect zero"
                if j in risk and float(p.X[k, p.columns["x"].index(variable)]) > 0:
                    elasticity = 0.0
                else:
                    reason += "; elasticity undefined for zero outcome probability or nonpositive changed attribute"
            support_rows.append(
                [
                    f"m{index + 1}",
                    p.case_labels[ci],
                    included,
                    weights[ci],
                    reason,
                    effect,
                    elasticity,
                ]
            )
        if not members:
            _error(
                "A requested alternative pair has no simultaneous remaining-riskset support.",
                "no_support",
            )
        support.append(members)
    if sum(len(rows) for rows in support) > MAX_CASE_ROWS:
        _error(
            "Complete per-case margins exceed8192 rows; request fewer targets.", "resource_budget"
        )
    nt, nr, q = len(targets), sum(len(rows) for rows in support), len(beta)
    plan = plan_workspace(
        "rank_ordered_margins",
        {
            "query_inputs_and_choice_work": tensor_bytes((len(p.alternative_labels), q)) * 3,
            "coefficient_covariance_factor": tensor_bytes((q, q)) * 6,
            "per_case_gradients_factors_and_values": tensor_bytes((nr, q)) * 4
            + tensor_bytes((nr,)) * 3,
            "average_gradients_and_factors": tensor_bytes((nt, q)) * 4,
            "joint_average_covariance": tensor_bytes((nt, nt)) * 3,
            "complete_support_diagnostics": tensor_bytes((len(support_rows), 7)) * 2,
        },
    ).record()
    factor = _factor(covariance)
    averages, average_jacobians, case_values, case_jacobians, case_metadata, summaries = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, ((stage, outcome, changed, variable), members) in enumerate(zip(targets, support)):
        name = f"m{index + 1}"
        denominator = math.fsum(weights[ci] for ci, _, _, _ in members)
        average = 0.0
        average_gradient = torch.zeros_like(beta)
        r = p.columns["x"].index(variable)
        for ci, risk, j, k in members:
            value, gradient, pj, pk = _margin(beta, p.X, risk, j, k, r, kind)
            weight = weights[ci] / denominator
            average += weight * value
            average_gradient += weight * gradient
            case_metadata.append(
                [
                    f"c{len(case_metadata) + 1}",
                    name,
                    p.case_labels[ci],
                    stage,
                    outcome,
                    changed,
                    variable,
                    weights[ci],
                    weight,
                    len(risk),
                    [p.alternative_labels[i] for i in risk],
                    pj,
                    pk,
                ]
            )
            case_values.append(value)
            case_jacobians.append(gradient)
        averages.append(average)
        average_jacobians.append(average_gradient)
        summaries.append(
            [
                name,
                stage,
                outcome,
                changed,
                variable,
                kind,
                len(members),
                len(p.case_labels),
                denominator,
            ]
        )
    J = torch.stack(average_jacobians)
    V = _delta(J, factor)
    CJ = torch.stack(case_jacobians)
    transformed = CJ @ factor
    case_variances = transformed.square().sum(dim=1)
    if not bool(torch.isfinite(CJ).all() & torch.isfinite(case_variances).all()) or not all(
        math.isfinite(v) for v in case_values + averages
    ):
        _error(
            "A margin or its full delta derivative exceeds finite float64 support.",
            "numerical_failure",
        )
    average_rows = []
    for i, meta in enumerate(summaries):
        se = math.sqrt(float(V[i, i]))
        lo, hi, statistic, probability, status = _normal(averages[i], se, z)
        average_rows.append([*meta, averages[i], se, lo, hi, statistic, probability, status])
    per_case = []
    for i, meta in enumerate(case_metadata):
        se = math.sqrt(float(case_variances[i]))
        lo, hi, statistic, probability, status = _normal(case_values[i], se, z)
        per_case.append([*meta, case_values[i], se, lo, hi, statistic, probability, status])
    parameters = list(p.columns["x"])
    names = [row[0] for row in summaries]
    frames = {
        "margins": table(
            average_rows,
            columns=[
                "target",
                "stage",
                "alternative",
                "changed_alternative",
                "variable",
                "kind",
                "n_support_cases",
                "n_query_cases",
                "support_weight",
                "estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "z",
                "p_value",
                "inference_status",
            ],
        ),
        "per_case": table(
            per_case,
            columns=[
                "case_target",
                "target",
                "case",
                "stage",
                "alternative",
                "changed_alternative",
                "variable",
                "case_weight",
                "normalized_support_weight",
                "riskset_size",
                "riskset",
                "outcome_probability",
                "changed_probability",
                "estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "z",
                "p_value",
                "inference_status",
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
        "jacobian": _jacobian(J, names, parameters),
        "per_case_jacobian": _jacobian(CJ, [row[0] for row in case_metadata], parameters),
        "coefficient_covariance": _matrix(covariance, parameters, "parameter"),
    }
    return _output(
        "rologit_margins",
        frames,
        state,
        p,
        mode,
        level,
        data,
        kind=kind,
        targets=[
            dict(stage=s, alternative=j, changed_alternative=k, variable=v)
            for s, j, k, v in targets
        ],
        case_weights=[dict(case=c, weight=w) for c, w in zip(p.case_labels, weights)],
        standardization="fixed positive case weights over explicitly reported pair support; not fit weights",
        averaging_population="pair-conditional eligible cases only; excluded structural zeros are not included in the average denominator",
        support_policy="support table includes every target and query case; absent alternatives or unsupported stages have no defined derivative; known unavailable/removed alternatives have structural attribute effect zero",
        max_margin_support_rows=MAX_CASE_ROWS,
        per_case_joint_covariance="lossless full cross-case/cross-target factorization: per_case_jacobian @ coefficient_covariance @ per_case_jacobian.T",
        resources=plan,
    )
