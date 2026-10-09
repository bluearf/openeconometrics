"""Orthogonal scores, honest heterogeneous effects and policy evaluation.

Only declared causal targets are inferential. Forest intervals describe local
conditional averages; neither individual counterfactual effects nor arbitrary
pointwise approximation-bias coverage is asserted.
"""

from __future__ import annotations

import hashlib
import json
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import ResultBundle
from openecon.resources import plan_workspace

from .learners import Work, fit_model, grow, predict_model, route

ASSUMPTIONS = (
    "Consistency/SUTVA, pre-treatment controls/importance weights, conditional unconfoundedness, "
    "positive overlap, independent rows or independent declared clusters, finite moments and "
    "nuisance consistency/product rates sufficient for orthogonal inference. Cross-fitting "
    "does not verify these identification or rate assumptions."
)


def _call(
    name,
    data,
    y,
    treatment,
    x,
    *,
    roles=None,
    covariance=None,
    cluster=None,
    weights=None,
    weight_type="iweight",
    missing="raise",
    alpha=0.05,
    **options,
):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=y,
            predictors=column_list(x, "controls"),
            columns={"treatment": treatment, **(roles or {})},
            intercept=True,
            covariance=covariance or ("cluster" if cluster is not None else "HC0"),
            cluster=cluster,
            weights=weights,
            weight_type=weight_type,
            missing=missing,
            alpha=alpha,
            options=options,
        ),
        data=data,
    )


def dmlirm(*, data, y, treatment, x=None, estimand="ate", **options):
    """Cross-fitted binary IRM/AIPW ATE or ATET; no propensity clipping."""
    return _call("dmlirm", data, y, treatment, x, estimand=estimand, **options)


def dmliivm(*, data, y, treatment, instrument, x=None, **options):
    """Cross-fitted binary interactive-IV LATE with explicit relevance admission."""
    return _call("dmliivm", data, y, treatment, x, roles={"instrument": instrument}, **options)


def dmlcate(*, data, y, treatment, basis, x=None, **options):
    """Best linear projection of CATE on a fixed numeric basis and intercept."""
    return _call(
        "dmlcate", data, y, treatment, x, roles={"basis": column_list(basis, "basis")}, **options
    )


def dmlgate(*, data, y, treatment, group=None, x=None, **options):
    """Declared-group GATE, or independent learned-ranking GATES if group is omitted."""
    return _call(
        "dmlgate", data, y, treatment, x, roles={} if group is None else {"group": group}, **options
    )


def causalforest(*, data, y, treatment, x, **options):
    """Three-way honest DR CATE forest with complete saved prediction state."""
    return _call("causalforest", data, y, treatment, x, **options)


def policyvalue(*, data, y, treatment, x=None, policy=None, learned=False, cost=0.0, **options):
    """Evaluate a predeclared policy or a policy learned on independent observations."""
    return _call(
        "policyvalue",
        data,
        y,
        treatment,
        x,
        roles={} if policy is None else {"policy": policy},
        learned=learned,
        cost=cost,
        **options,
    )


def _binary(frame, name):
    values = frame.numeric(name)
    if not bool(((values == 0) | (values == 1)).all()) or values.min() == values.max():
        raise AnalysisError(
            "invalid_binary_role", f"{name} must contain both 0 and 1, with no other values."
        )
    return values


def _prepare(spec, data):
    names = [spec.outcome, spec.columns["treatment"], *spec.predictors]
    if "instrument" in spec.columns:
        names.append(spec.columns["instrument"])
    if len(names) != len(set(names)):
        raise AnalysisError(
            "invalid_roles", "Outcome, treatment, instrument and controls must be distinct."
        )
    frame = ModelFrame(spec, data)
    if spec.weights is not None:
        from openecon.analysis import _numeric

        raw = _numeric(frame.original[spec.weights].dropna(), spec.weights)
        if bool((raw <= 0).any()):
            raise AnalysisError(
                "invalid_importance_weights",
                "Causal importance weights must be strictly positive; zero weights are not silently excluded.",
            )
    n, p = frame.n, len(spec.predictors)
    if n < 30 or p > 128:
        raise AnalysisError(
            "sample_domain",
            "Causal learning requires at least 30 complete rows and at most 128 controls.",
        )
    o = {item.name: frame.option(item.name) for item in frame.info.options}
    honest = (
        spec.estimator == "causalforest"
        or (spec.estimator == "dmlgate" and not frame.role("group"))
        or (spec.estimator == "policyvalue" and o["learned"])
    )
    forest_fields = {"trees", "max_depth", "min_leaf", "mtry", "split_candidates"}
    if o["learner"] != "forest" and not honest and forest_fields & set(spec.options):
        raise AnalysisError(
            "unused_options", "Forest settings require a forest learner or honest forest target."
        )
    if o["learner"] != "forest" and "leaf_prior" in spec.options:
        raise AnalysisError("unused_options", "leaf_prior requires a forest probability nuisance.")
    if o["learner"] not in ("lasso", "ridge") and {"penalty", "max_iterations", "tolerance"} & set(
        spec.options
    ):
        raise AnalysisError(
            "unused_options", "Penalty/KKT settings require a lasso or ridge nuisance."
        )
    if honest and "folds" in spec.options:
        raise AnalysisError(
            "unused_options",
            "Honest forest targets use three independent partitions, not K-fold cross-fitting.",
        )
    group_column = frame.role("group") if spec.estimator == "dmlgate" else []
    if group_column and "groups" in spec.options:
        raise AnalysisError(
            "unused_options",
            "groups selects learned GATES; declared GATE uses the group column's levels.",
        )
    conditioning_names = [name for name in spec.predictors if name not in group_column]
    if spec.estimator == "dmlcate":
        for name in frame.role("basis"):
            if name in (spec.outcome, spec.columns["treatment"]):
                raise AnalysisError(
                    "invalid_basis",
                    "CATE basis must be pre-treatment and distinct from outcome/treatment.",
                )
            if name not in conditioning_names:
                conditioning_names.append(name)
    if spec.estimator == "policyvalue" and not o["learned"] and frame.role("policy"):
        name = frame.role("policy")[0]
        if name in (spec.outcome, spec.columns["treatment"]):
            raise AnalysisError(
                "invalid_policy", "Policy must be outcome-independent and pre-treatment."
            )
        policy = frame.numeric(name)
        if policy.min() != policy.max() and name not in conditioning_names:
            conditioning_names.append(name)
    x = frame.matrix(conditioning_names)
    nuisance_design = [{"column": name, "encoding": "numeric"} for name in conditioning_names]
    if group_column:
        name = group_column[0]
        if name in (spec.outcome, spec.columns["treatment"]):
            raise AnalysisError(
                "invalid_groups", "Groups must be pre-treatment, not outcome/treatment labels."
            )
        group_codes, group_count = frame.codes(name)
        if group_count < 2 or group_count > 20:
            raise AnalysisError("group_domain", "Declared GATE requires 2..20 predeclared groups.")
        levels = frame.levels(name)
        indicators = torch.nn.functional.one_hot(group_codes, group_count)[:, 1:].to(torch.float64)
        x = torch.cat((x, indicators), 1)
        nuisance_design += [
            {"column": name, "encoding": "indicator", "level": level} for level in levels[1:]
        ]
    p = x.shape[1]
    if p > 128:
        raise AnalysisError(
            "sample_domain",
            "At most 128 controls including declared basis/group/policy conditioning variables are supported.",
        )
    frame.notes["nuisance_design"] = nuisance_design
    if o["mtry"] > p:
        raise AnalysisError("invalid_mtry", "mtry must not exceed the number of controls.")
    frame.workspace_plan(
        "causal scores, nuisance copies and complete saved state",
        {
            "design_scores_and_json": n * (p + 32) * 256,
            "forest_nodes_and_json": o["trees"] * (2 ** (o["max_depth"] + 1)) * o["folds"] * 6 * 512
            if o["learner"] == "forest" or honest
            else 0,
            "nuisance_factorizations": (p + 1) ** 2 * 128,
        },
    )
    y, d = frame.numeric(spec.outcome), _binary(frame, spec.columns["treatment"])
    w = frame.weights()
    w = torch.ones(n, dtype=torch.float64) if w is None else w
    if not bool((w > 0).all()):
        raise AnalysisError(
            "invalid_importance_weights",
            "Causal importance weights must be strictly positive pre-treatment quantities.",
        )
    w = w / w.max()
    w = w / w.mean()
    codes = None
    if spec.cluster is not None:
        codes, count = frame.cluster_dimensions()[0]
        if count < max(6, 2 * o["folds"]):
            raise AnalysisError(
                "insufficient_clusters",
                "Causal cross-fitting needs at least two clusters per fold and at least six overall.",
            )
    return frame, o, x, y, d, w, codes, Work(o["max_work"])


def _partitions(n, k, seed, codes):
    generator = torch.Generator().manual_seed(seed)
    count = n if codes is None else int(codes.max()) + 1
    if count < 2 * k:
        raise AnalysisError(
            "insufficient_folds", "Each split needs at least two independent rows or clusters."
        )
    order = torch.randperm(count, generator=generator)
    assignment = torch.empty(count, dtype=torch.int64)
    assignment[order] = torch.arange(count) % k
    return assignment if codes is None else assignment[codes]


def _record_positions(frame, rows):
    return [frame.positions[i] for i in rows.tolist()]


def _train_nuisance(x, y, assignment_variable, d, train, test, o, seed, work, iv=False):
    """All fitted values are evaluated outside the supplied training positions."""
    if len(train) == 0 or len(test) == 0:
        raise AnalysisError(
            "empty_fold",
            "Every causal nuisance split requires training and evaluation observations.",
        )
    models, predictions = {}, {}
    models["propensity"] = fit_model(x[train], assignment_variable[train], True, o, seed, work)
    predictions["propensity"] = predict_model(models["propensity"], x[test], work)
    for level in (0, 1):
        rows = train[assignment_variable[train] == level]
        models[f"outcome{level}"] = fit_model(x[rows], y[rows], False, o, seed + level + 1, work)
        predictions[f"outcome{level}"] = predict_model(models[f"outcome{level}"], x[test], work)
        if iv:
            # Constant first-stage conditional strata are meaningful (e.g. one-sided compliance).
            if d[rows].min() == d[rows].max():
                models[f"treatment{level}"] = {
                    "kind": "constant",
                    "value": float(d[rows][0]),
                    "training_n": len(rows),
                }
                predictions[f"treatment{level}"] = torch.full(
                    (len(test),), float(d[rows][0]), dtype=torch.float64
                )
            else:
                models[f"treatment{level}"] = fit_model(
                    x[rows], d[rows], True, o, seed + level + 3, work
                )
                predictions[f"treatment{level}"] = predict_model(
                    models[f"treatment{level}"], x[test], work
                )
    return models, predictions


def _overlap(e, o):
    if not bool(torch.isfinite(e).all()) or bool(
        ((e < o["overlap"]) | (e > 1 - o["overlap"])).any()
    ):
        raise AnalysisError(
            "overlap_violation",
            "Held-out propensity violates declared overlap; no clipping or automatic row trimming is performed.",
        )


def _crossfit(frame, o, x, y, d, codes, work, z=None):
    a = d if z is None else z
    assignment = _partitions(frame.n, o["folds"], o["seed"], codes)
    keys = ["propensity", "outcome0", "outcome1"] + (
        [] if z is None else ["treatment0", "treatment1"]
    )
    predicted = {name: torch.empty(frame.n, dtype=torch.float64) for name in keys}
    records = []
    for fold in range(o["folds"]):
        train = (assignment != fold).nonzero().flatten()
        test = (assignment == fold).nonzero().flatten()
        models, fitted = _train_nuisance(
            x, y, a, d, train, test, o, o["seed"] + 11 * fold, work, z is not None
        )
        for name in keys:
            predicted[name][test] = fitted[name]
        records.append(
            {
                "fold": fold,
                "train_positions": _record_positions(frame, train),
                "test_positions": _record_positions(frame, test),
                "models": models,
            }
        )
    _overlap(predicted["propensity"], o)
    return predicted, {
        "cross_fitted": True,
        "fold_assignment": assignment.tolist(),
        "fold_records": records,
        "nuisance_predictions": {k: v.tolist() for k, v in predicted.items()},
        "fold_unit": "whole cluster" if codes is not None else "row",
    }


def _potential_scores(y, d, predicted):
    e, m0, m1 = (predicted[key] for key in ("propensity", "outcome0", "outcome1"))
    return m0 + (1 - d) * (y - m0) / (1 - e), m1 + d * (y - m1) / e


def _covariance(frame, contribution, codes, *, sample_n=None):
    """Contribution rows already include normalized weight and Jacobian."""
    n = len(contribution) if sample_n is None else sample_n
    df, count = None, None
    if codes is not None:
        _, dense = torch.unique(codes, sorted=True, return_inverse=True)
        count = int(dense.max()) + 1
        if count < 3:
            raise AnalysisError(
                "insufficient_clusters",
                "Inference needs at least three independent evaluation clusters.",
            )
        if count < 30:
            frame.warn(
                f"Only {count} independent evaluation clusters; asymptotic cluster inference may be unreliable."
            )
        totals = torch.zeros((count, contribution.shape[1]), dtype=torch.float64).index_add_(
            0, dense, contribution
        )
        covariance = totals.T @ totals * count / (count - 1)
        df = count - 1
    else:
        covariance = contribution.T @ contribution
        if frame.spec.covariance == "HC1":
            covariance *= n / (n - 1)
    return covariance, df, count


def _ratio(frame, b, a, w, codes):
    mass = w / w.sum()
    denominator = mass @ a
    if not torch.isfinite(denominator) or denominator <= 0:
        raise AnalysisError(
            "unidentified_target",
            "The causal ratio denominator must be finite and strictly positive.",
        )
    theta = (mass @ b) / denominator
    influence = mass * (b - theta * a) / denominator
    covariance, df, count = _covariance(frame, influence[:, None], codes)
    return theta.reshape(1), covariance, df, count, influence[:, None]


def _diagnostics(x, d, predicted, w):
    e = predicted["propensity"]
    controls = []
    for j in range(x.shape[1]):
        v = x[:, j]
        sd = torch.sqrt(((v[d == 0].var(correction=0) + v[d == 1].var(correction=0)) / 2))
        raw = float((v[d == 1].mean() - v[d == 0].mean()) / sd) if sd > 0 else None
        treated, control = w * d / e, w * (1 - d) / (1 - e)
        balanced = (
            float(((treated @ v) / treated.sum() - (control @ v) / control.sum()) / sd)
            if sd > 0
            else None
        )
        controls.append(
            {
                "feature_index": j,
                "raw_standardized_difference": raw,
                "weighted_standardized_difference": balanced,
            }
        )
    return {
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
        "brier_score": float((d - e).square().mean()),
        "inverse_weight_effective_n": [
            float((w * d / e).sum().square() / (w * d / e).square().sum()),
            float((w * (1 - d) / (1 - e)).sum().square() / (w * (1 - d) / (1 - e)).square().sum()),
        ],
        "balance": controls,
        "robustness": "Compare declared learners, folds, seeds and overlap thresholds explicitly; no automatic favorable-result selection.",
    }


def _finish(frame, terms, values, covariance, df, count, extra, work, **kwargs):
    return build_result(
        frame,
        terms=terms,
        params=values,
        covariance=covariance,
        use_t=df is not None,
        df_inference=df,
        solver="native float64 orthogonal causal score",
        solver_diagnostics={"nuisance_models_converged": True, "work_used": work.used},
        inference={
            "target": extra["target"],
            "correction": "sum of normalized influence outer products; HC1 N/(N-1), or cluster G/(G-1)",
            "cluster_count": count,
            "cluster_df": df,
            "nuisance_uncertainty": "Orthogonality controls first-order nuisance error under declared rates; rates are not empirically certified.",
        },
        extra={
            "assumptions": ASSUMPTIONS,
            "nuisance_design": frame.notes["nuisance_design"],
            **extra,
        },
        warnings=[ASSUMPTIONS],
        **kwargs,
    )


def fit_dmlirm(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    predicted, records = _crossfit(frame, o, x, y, d, codes, work)
    phi0, phi1 = _potential_scores(y, d, predicted)
    if o["estimand"] == "ate":
        b, a = phi1 - phi0, torch.ones_like(y)
    else:
        e, m0 = predicted["propensity"], predicted["outcome0"]
        b, a = d * (y - m0) - (1 - d) * e * (y - m0) / (1 - e), d
    values, covariance, df, count, influence = _ratio(frame, b, a, w, codes)
    e = predicted["propensity"]
    wt, wc = w * d / e, w * (1 - d) / (1 - e)
    diagnostic_targets = {
        "regression_adjustment_ATE": float(
            (w * (predicted["outcome1"] - predicted["outcome0"])).sum() / w.sum()
        ),
        "normalized_IPW_ATE": float(wt @ y / wt.sum() - wc @ y / wc.sum()),
        "AIPW_ATE": float((w * (phi1 - phi0)).sum() / w.sum()),
        "interpretation": "Predeclared nuisance/specification diagnostics; no automatic estimator or favorable-result selection.",
    }
    return _finish(
        frame,
        [o["estimand"].upper()],
        values,
        covariance,
        df,
        count,
        {
            "target": o["estimand"],
            "nuisance_predictions": {k: v.tolist() for k, v in predicted.items()},
            "score_b": b.tolist(),
            "score_a": a.tolist(),
            "influence_contributions": influence.tolist(),
            "importance_weights": w.tolist(),
            "diagnostics": _diagnostics(x, d, predicted, w),
            "robustness_estimates": diagnostic_targets,
            **records,
        },
        work,
    )


def fit_dmliivm(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    z = _binary(frame, spec.columns["instrument"])
    predicted, records = _crossfit(frame, o, x, y, d, codes, work, z=z)
    gy0, gy1 = _potential_scores(y, z, predicted)
    r0, r1, e = predicted["treatment0"], predicted["treatment1"], predicted["propensity"]
    gd0, gd1 = r0 + (1 - z) * (d - r0) / (1 - e), r1 + z * (d - r1) / e
    first = gd1 - gd0
    mass = w / w.sum()
    first_stage = mass @ first
    first_v, first_df, _ = _covariance(frame, (mass * (first - first_stage))[:, None], codes)
    first_se = float(first_v[0, 0].sqrt())
    critical = float(critical_value(spec.alpha, first_df))
    if float(first_stage) - critical * first_se < o["min_first_stage"]:
        raise AnalysisError(
            "weak_first_stage",
            "Positive first-stage relevance is insufficient; ordinary LATE Wald inference is refused. Reorient a decreasing instrument explicitly.",
        )
    values, covariance, df, count, influence = _ratio(frame, gy1 - gy0, first, w, codes)
    return _finish(
        frame,
        ["LATE"],
        values,
        covariance,
        df,
        count,
        {
            "target": "late",
            "iv_assumptions": "Conditional instrument independence, exclusion, a first stage bounded away from zero and monotone increasing compliance. Relevance diagnostics do not establish strong-IV asymptotic validity or identification assumptions.",
            "first_stage": float(first_stage),
            "first_stage_se": first_se,
            "first_stage_df": first_df,
            "first_stage_ci_low": float(first_stage) - critical * first_se,
            "reduced_form": float(mass @ (gy1 - gy0)),
            "score_b": (gy1 - gy0).tolist(),
            "score_a": first.tolist(),
            "influence_contributions": influence.tolist(),
            "nuisance_predictions": {k: v.tolist() for k, v in predicted.items()},
            "diagnostics": _diagnostics(x, z, predicted, w),
            **records,
        },
        work,
    )


def _wald(values, covariance, contrast):
    difference, v = contrast @ values, contrast @ covariance @ contrast.T
    rank = int(torch.linalg.matrix_rank(v))
    if rank != len(difference):
        return {"status": "unavailable", "reason": "contrast covariance is singular", "rank": rank}
    statistic = difference @ torch.linalg.solve(v, difference)
    return {
        "status": "ok",
        "statistic": float(statistic),
        "distribution": "chi2",
        "df": rank,
        "p_value": float(
            torch.special.gammaincc(torch.tensor(rank / 2, dtype=torch.float64), statistic / 2)
        ),
    }


def fit_dmlcate(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    names = frame.role("basis")
    if any(name in [spec.outcome, spec.columns["treatment"]] for name in names):
        raise AnalysisError(
            "invalid_basis", "CATE basis must be pre-treatment and distinct from outcome/treatment."
        )
    if len(names) > 32:
        raise AnalysisError("basis_domain", "At most 32 predeclared basis columns are supported.")
    b = torch.cat((torch.ones((frame.n, 1), dtype=torch.float64), frame.matrix(names)), 1)
    mass = w / w.sum()
    work.add(frame.n * b.shape[1] ** 2 + b.shape[1] ** 3)
    gram = b.T @ (mass[:, None] * b)
    if int(torch.linalg.matrix_rank(gram)) != len(gram):
        raise AnalysisError(
            "singular_basis", "Declared CATE basis is not full rank; no columns are dropped."
        )
    predicted, records = _crossfit(frame, o, x, y, d, codes, work)
    phi0, phi1 = _potential_scores(y, d, predicted)
    inverse = torch.linalg.inv(gram)
    values = inverse @ (b.T @ (mass * (phi1 - phi0)))
    influence = (b @ inverse) * (mass * (phi1 - phi0 - b @ values))[:, None]
    covariance, df, count = _covariance(frame, influence, codes)
    state = _seal(
        {
            "version": 1,
            "kind": "basis",
            "features": names,
            "coefficient": values.tolist(),
            "covariance": covariance.tolist(),
            "df": df,
            "alpha": spec.alpha,
            "support_min": b[:, 1:].min(0).values.tolist(),
            "support_max": b[:, 1:].max(0).values.tolist(),
        }
    )
    contrast = torch.eye(len(values), dtype=torch.float64)[1:]
    return _finish(
        frame,
        ["_cons", *names],
        values,
        covariance,
        df,
        count,
        {
            "target": "predeclared_basis_CATE_projection",
            "cate_state": state,
            "basis_approximation": "Inference concerns the declared best linear projection; equality with pointwise CATE needs a correct/sufficient basis.",
            "pseudo_outcome": (phi1 - phi0).tolist(),
            "influence_contributions": influence.tolist(),
            "nuisance_predictions": {k: v.tolist() for k, v in predicted.items()},
            **records,
        },
        work,
        tests={"heterogeneity": _wald(values, covariance, contrast)},
    )


def _seal(state):
    state["sha256"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return state


def _check_state(state):
    if not isinstance(state, dict) or state.get("version") != 1:
        raise AnalysisError("invalid_causal_state", "Complete versioned CATE state is required.")
    clean = {k: v for k, v in state.items() if k != "sha256"}
    try:
        checksum = hashlib.sha256(
            json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (ValueError, TypeError):
        raise AnalysisError(
            "invalid_causal_state", "Saved CATE state is not finite JSON."
        ) from None
    if checksum != state.get("sha256"):
        raise AnalysisError(
            "causal_state_integrity",
            "Saved CATE state checksum differs; refitting is not substituted.",
        )
    return clean


def _honest(frame, o, x, y, d, codes, work):
    if x.shape[1] == 0:
        raise AnalysisError(
            "missing_controls", "Honest heterogeneous forest learning requires numeric controls."
        )
    assignment = _partitions(frame.n, 3, o["seed"], codes)
    nuisance, structure, estimation = [(assignment == j).nonzero().flatten() for j in range(3)]
    if min(len(nuisance), len(structure), len(estimation)) < 2 * o["min_leaf"]:
        raise AnalysisError(
            "honest_sample_domain", "Each honest sample needs at least twice min_leaf rows."
        )
    evaluate = torch.cat((structure, estimation))
    models, predicted = _train_nuisance(x, y, d, d, nuisance, evaluate, o, o["seed"], work)
    _overlap(predicted["propensity"], o)
    phi0, phi1 = _potential_scores(y[evaluate], d[evaluate], predicted)
    scores = phi1 - phi0
    split_score, leaf_score = scores[: len(structure)], scores[len(structure) :]
    generator = torch.Generator().manual_seed(o["seed"] + 104729)
    trees = []
    for _ in range(o["trees"]):
        selected = torch.randperm(len(structure), generator=generator)[
            : max(2 * o["min_leaf"], math.ceil(0.8 * len(structure)))
        ]
        trees.append(
            grow(
                x[structure][selected],
                split_score[selected],
                o,
                generator,
                work,
                check_x=x[estimation],
            )
        )
    records = {
        "cross_fitted": False,
        "honest_three_way": True,
        "partition_assignment": assignment.tolist(),
        "nuisance_positions": _record_positions(frame, nuisance),
        "structure_positions": _record_positions(frame, structure),
        "estimation_positions": _record_positions(frame, estimation),
        "nuisance_models": models,
        "nuisance_predictions": {k: v.tolist() for k, v in predicted.items()},
        "prediction_positions": _record_positions(frame, evaluate),
        "partition_unit": "whole cluster" if codes is not None else "row",
        "honesty": "Nuisance fits use sample A only; splits use sample B outcomes only; sample C outcomes enter leaf estimation and held-out targets only.",
    }
    return (
        structure,
        estimation,
        trees,
        split_score,
        leaf_score,
        predicted,
        phi0[len(structure) :],
        phi1[len(structure) :],
        records,
    )


def _forest_weights(trees, estimation_x, query, w, work):
    mass = torch.zeros((len(query), len(estimation_x)), dtype=torch.float64)
    for tree in trees:
        work.add((len(query) + len(estimation_x)) * len(tree) + len(query) * len(estimation_x))
        estimation_leaf, query_leaf = route(tree, estimation_x), route(tree, query)
        same = query_leaf[:, None] == estimation_leaf[None, :]
        weights = same * w[None, :]
        total = weights.sum(1)
        if bool((total <= 0).any()):
            raise AnalysisError(
                "empty_honest_leaf", "A query has no honest estimation support; no tree is skipped."
            )
        mass += weights / total[:, None]
    return mass / len(trees)


def _forest_prediction(state, query, work):
    estimation_x = torch.tensor(state["estimation_x"], dtype=torch.float64)
    scores = torch.tensor(state["scores"], dtype=torch.float64)
    weights = _forest_weights(
        state["trees"],
        estimation_x,
        query,
        torch.tensor(state["importance_weights"], dtype=torch.float64),
        work,
    )
    values = weights @ scores
    work.add(len(query) ** 2 * len(scores) + len(query) * len(scores) * 4)
    residual = scores[None, :] - values[:, None]
    contribution = (weights * residual).T
    if state["cluster_codes"] is not None:
        codes = torch.tensor(state["cluster_codes"], dtype=torch.int64)
        _, dense = torch.unique(codes, sorted=True, return_inverse=True)
        count = int(dense.max()) + 1
        if count < 3:
            raise AnalysisError(
                "insufficient_clusters",
                "Honest forest intervals need three independent estimation clusters.",
            )
        totals = torch.zeros((count, len(query)), dtype=torch.float64).index_add_(
            0, dense, contribution
        )
        v = totals.T @ totals * count / (count - 1)
        df = count - 1
    else:
        v = contribution.T @ contribution
        if state["covariance_kind"] == "HC1":
            v *= len(scores) / (len(scores) - 1)
        df = None
    effective = 1 / weights.square().sum(1)
    return values, v, effective, df


def _forest_state(frame, x, estimation, trees, scores, w, codes):
    return _seal(
        {
            "version": 1,
            "kind": "honest_dr_forest",
            "features": list(frame.spec.predictors),
            "trees": trees,
            "estimation_x": x[estimation].tolist(),
            "scores": scores.tolist(),
            "importance_weights": w[estimation].tolist(),
            "cluster_codes": None if codes is None else codes[estimation].tolist(),
            "covariance_kind": frame.spec.covariance,
            "alpha": frame.spec.alpha,
            "support_min": x.min(0).values.tolist(),
            "support_max": x.max(0).values.tolist(),
            "interval_target": "Conditional local forest average of the DR score; excludes forest approximation bias and universal pointwise CATE guarantees.",
        }
    )


def _prediction_table(values, covariance, effective, df, alpha):
    variance = covariance.diagonal()
    if (
        not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(covariance).all())
        or bool((variance <= 0).any())
    ):
        raise AnalysisError(
            "invalid_covariance", "CATE queries do not provide finite positive sampling variances."
        )
    se = variance.sqrt()
    p = two_sided_p_values(values / se, df)
    critical = critical_value(alpha, df)
    return [
        {
            "row": i,
            "cate": float(values[i]),
            "std_error": float(se[i]),
            "statistic": float(values[i] / se[i]),
            "df_inference": df,
            "p_value": float(p[i]),
            "ci_low": float(values[i] - critical * se[i]),
            "ci_high": float(values[i] + critical * se[i]),
            "effective_n": None if effective is None else float(effective[i]),
        }
        for i in range(len(values))
    ]


def fit_causalforest(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    _, estimation, trees, _, scores, predicted, _, _, records = _honest(
        frame, o, x, y, d, codes, work
    )
    state = _forest_state(frame, x, estimation, trees, scores, w, codes)
    query = x if o["query"] is None else _query(o["query"], x.shape[1])
    _query_plan(len(query), len(estimation), x.shape[1])
    _support(query, state)
    values, v, effective, df = _forest_prediction(state, query, work)
    ate, ate_v, ate_df, count, influence = _ratio(
        frame,
        scores,
        torch.ones_like(scores),
        w[estimation],
        None if codes is None else codes[estimation],
    )
    return _finish(
        frame,
        ["ATE:honest_estimation_sample"],
        ate,
        ate_v,
        ate_df,
        count,
        {
            "target": "honest_DR_forest_CATE_and_ATE",
            "cate_state": state,
            "cate": _prediction_table(values, v, effective, df, spec.alpha),
            "cate_covariance": v.tolist(),
            "query": query.tolist(),
            "inference_rows": len(estimation),
            "influence_contributions": influence.tolist(),
            **records,
        },
        work,
    )


def _groups(frame, values, group_codes, labels, w, codes, work, records):
    work.add(len(values) * len(labels) ** 2 + len(labels) ** 3)
    g = len(labels)
    if g > 20:
        raise AnalysisError("group_domain", "At most 20 predeclared groups are supported.")
    mass = w / w.sum()
    columns, coefficients = [], []
    for level in range(g):
        selected = group_codes == level
        if int(selected.sum()) < 5:
            raise AnalysisError(
                "insufficient_group_sample", "Every group requires at least five evaluation rows."
            )
        total = mass[selected].sum()
        effect = (mass[selected] @ values[selected]) / total
        coefficients.append(effect)
        columns.append(mass * selected * (values - effect) / total)
    beta = torch.stack(coefficients)
    influence = torch.stack(columns, 1)
    covariance, df, count = _covariance(frame, influence, codes)
    contrast = torch.zeros((g - 1, g), dtype=torch.float64)
    contrast[:, 0] = -1
    contrast[:, 1:] = torch.eye(g - 1, dtype=torch.float64)
    return _finish(
        frame,
        [f"GATE:{j}:{label}" for j, label in enumerate(labels)],
        beta,
        covariance,
        df,
        count,
        {
            "target": "group_average_treatment_effects",
            "group_labels": labels,
            "group_assignment": group_codes.tolist(),
            "pseudo_outcome": values.tolist(),
            "importance_weights": w.tolist(),
            "influence_contributions": influence.tolist(),
            **records,
        },
        work,
        tests={"heterogeneity": _wald(beta, covariance, contrast)},
    )


def fit_dmlgate(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    if frame.role("group"):
        predicted, records = _crossfit(frame, o, x, y, d, codes, work)
        phi0, phi1 = _potential_scores(y, d, predicted)
        column = frame.role("group")[0]
        if column in [spec.outcome, spec.columns["treatment"]]:
            raise AnalysisError(
                "invalid_groups", "Groups must be pre-treatment, not outcome/treatment labels."
            )
        group, _ = frame.codes(column)
        return _groups(frame, phi1 - phi0, group, frame.levels(column), w, codes, work, records)
    structure, estimation, trees, split_scores, scores, _, _, _, records = _honest(
        frame, o, x, y, d, codes, work
    )
    # Ranking uses structure-sample leaf outcomes only, never estimation outcomes.
    ranks = torch.zeros(len(estimation), dtype=torch.float64)
    reference = torch.zeros(len(structure), dtype=torch.float64)
    for tree in trees:
        work.add((len(structure) + len(estimation)) * len(tree))
        ranks += torch.tensor(
            [tree[j]["value"] for j in route(tree, x[estimation]).tolist()], dtype=torch.float64
        )
        reference += torch.tensor(
            [tree[j]["value"] for j in route(tree, x[structure]).tolist()], dtype=torch.float64
        )
    ranks, reference = ranks / len(trees), reference / len(trees)
    boundaries = torch.quantile(
        reference, torch.arange(1, o["groups"], dtype=torch.float64) / o["groups"]
    )
    if len(boundaries.unique()) != len(boundaries):
        raise AnalysisError(
            "indistinct_gates",
            "Training-only ranking cannot form distinct requested GATES; reduce groups explicitly.",
        )
    group = torch.bucketize(ranks, boundaries, right=True)
    records.update(
        {
            "ranking_source": "structure-sample outcomes only",
            "ranking_boundaries": boundaries.tolist(),
            "evaluation_positions": _record_positions(frame, estimation),
        }
    )
    return _groups(
        frame,
        scores,
        group,
        list(range(o["groups"])),
        w[estimation],
        None if codes is None else codes[estimation],
        work,
        records,
    )


def fit_policyvalue(spec, data):
    frame, o, x, y, d, w, codes, work = _prepare(spec, data)
    policy_columns = frame.role("policy")
    if o["learned"]:
        if policy_columns:
            raise AnalysisError(
                "invalid_policy",
                "A learned policy and a predeclared policy column cannot both be supplied.",
            )
        structure, estimation, trees, _, _, predicted, phi0, phi1, records = _honest(
            frame, o, x, y, d, codes, work
        )
        rank = torch.zeros(len(estimation), dtype=torch.float64)
        for tree in trees:
            work.add(len(estimation) * len(tree))
            rank += torch.tensor(
                [tree[j]["value"] for j in route(tree, x[estimation]).tolist()], dtype=torch.float64
            )
        policy = (rank / len(trees) > o["cost"]).to(torch.float64)
        weights, clusters = w[estimation], None if codes is None else codes[estimation]
        records["policy_source"] = (
            "independent nuisance/structure samples; estimation outcomes never train the policy"
        )
        records["evaluation_positions"] = _record_positions(frame, estimation)
        diagnostic_predicted = {k: v[len(structure) :] for k, v in predicted.items()}
        diagnostics = _diagnostics(x[estimation], d[estimation], diagnostic_predicted, weights)
    else:
        if len(policy_columns) != 1 or policy_columns[0] in [
            spec.outcome,
            spec.columns["treatment"],
        ]:
            raise AnalysisError(
                "invalid_policy",
                "Provide an outcome-independent pre-treatment policy probability column, or learned=True.",
            )
        policy = frame.numeric(policy_columns[0])
        if bool(((policy < 0) | (policy > 1)).any()):
            raise AnalysisError("invalid_policy", "Policy probabilities must lie in [0,1].")
        predicted, records = _crossfit(frame, o, x, y, d, codes, work)
        phi0, phi1 = _potential_scores(y, d, predicted)
        weights, clusters = w, codes
        diagnostics = _diagnostics(x, d, predicted, weights)
        records["policy_source"] = (
            "user-declared outcome-independent policy; this independence is an assumption"
        )
    scores = torch.stack(
        (policy * (phi1 - o["cost"]) + (1 - policy) * phi0, phi1 - o["cost"], phi0), 1
    )
    names = ["policy_value", "treat_all_value", "treat_none_value"]
    gain = scores[:, 0] - phi0
    deterministic = None
    if float(gain.var(correction=0)) > 0:
        scores = torch.cat((scores, gain[:, None]), 1)
        names.append("gain_vs_none")
    else:
        deterministic = float(gain[0])
    mass = weights / weights.sum()
    values = mass @ scores
    influence = mass[:, None] * (scores - values)
    covariance, df, count = _covariance(frame, influence, clusters)
    return _finish(
        frame,
        names,
        values,
        covariance,
        df,
        count,
        {
            "target": "policy_value_and_paired_contrasts",
            "policy_probabilities": policy.tolist(),
            "treatment_cost": o["cost"],
            "importance_weights": weights.tolist(),
            "deterministic_gain_vs_none": deterministic,
            "value_scores": scores.tolist(),
            "influence_contributions": influence.tolist(),
            "evaluation_rows": len(scores),
            "diagnostics": diagnostics,
            **records,
        },
        work,
    )


def _query(value, p):
    try:
        query = torch.tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        raise AnalysisError(
            "invalid_query", "Query must be a finite nonempty m by p numeric matrix."
        ) from None
    if (
        query.ndim != 2
        or query.shape[1] != p
        or len(query) == 0
        or not bool(torch.isfinite(query).all())
    ):
        raise AnalysisError(
            "invalid_query", "Query must be a finite nonempty m by p numeric matrix."
        )
    return query


def _query_plan(m, n, p):
    if m > 1000:
        raise AnalysisError(
            "query_domain",
            "At most 1000 simultaneous CATE queries are supported; complete query covariance is retained.",
        )
    plan_workspace(
        "CATE joint query covariance and saved state",
        {
            "queries": m * (p + 4) * 64,
            "weights_scores": m * n * 48,
            "saved_estimation_copy": n * (p + 4) * 256,
            "covariance": m * m * 64,
        },
    )


def _support(query, state):
    lo, hi = (
        torch.tensor(state["support_min"], dtype=torch.float64),
        torch.tensor(state["support_max"], dtype=torch.float64),
    )
    if bool(((query < lo) | (query > hi)).any()):
        raise AnalysisError(
            "query_outside_support",
            "CATE query lies outside declared training marginal support; extrapolation is refused.",
        )


def causal_predict(result: ResultBundle, data, *, max_work=200000000):
    """Predict CATE from complete saved forest/basis state; preserves all input rows."""
    from openecon.analysis import _coerce_frame, _numeric
    from openecon.dataset import Dataset

    if not isinstance(result, ResultBundle) or result.spec.estimator not in (
        "causalforest",
        "dmlcate",
    ):
        raise AnalysisError(
            "invalid_result", "causal_predict requires a causalforest or dmlcate result."
        )
    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported", "Saved CATE prediction does not collect Dataset inputs."
        )
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1:
        raise AnalysisError("invalid_work_limit", "max_work must be a positive integer.")
    state = _check_state(result.extra.get("cate_state"))
    table = _coerce_frame(data)
    names = state["features"]
    if not len(table) or table.columns.has_duplicates or any(name not in table for name in names):
        raise AnalysisError(
            "invalid_predictors",
            "Finite complete CATE prediction columns must be present and unique.",
        )
    _query_plan(len(table), len(state.get("scores", [])), len(names))
    query = torch.stack([_numeric(table[name], name) for name in names], 1)
    _support(query, state)
    work = Work(max_work)
    if state["kind"] == "basis":
        work.add(len(query) ** 2 * (len(names) + 1) ** 2)
        design = torch.cat((torch.ones((len(query), 1), dtype=torch.float64), query), 1)
        values = design @ torch.tensor(state["coefficient"], dtype=torch.float64)
        covariance = design @ torch.tensor(state["covariance"], dtype=torch.float64) @ design.T
        effective, df = None, state["df"]
    elif state["kind"] == "honest_dr_forest":
        values, covariance, effective, df = _forest_prediction(state, query, work)
    else:
        raise AnalysisError("invalid_causal_state", "Unknown CATE state kind.")
    import pandas as pd

    output = pd.DataFrame(
        _prediction_table(values, covariance, effective, df, state["alpha"]), index=table.index
    )
    output.attrs.update(
        covariance_matrix=covariance.tolist(),
        target=state.get("interval_target", "declared CATE basis projection"),
        source_result_id=result.id,
        precision="float64",
    )
    return output
