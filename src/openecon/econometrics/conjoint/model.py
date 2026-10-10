"""Full-profile scored conjoint with full individual OLS and fitted-mean covariance."""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines import distributions as dist
from . import common as c


def confidence(level):
    value = c.c.check_number(level, "level", minimum=0, maximum=1, exclusive=True)
    if value == 1:
        raise AnalysisError("invalid_option", "level must be strictly between zero and one.")
    return value


def scored(data, plan, subject, profile, score, missing):
    """Join by typed explicit IDs, preserving original row positions and labels."""
    c.c.check_choice(missing, "missing", ("raise", "drop_subjects"))
    names = [c.name(subject), c.name(profile), c.name(score)]
    if len(set(names)) != 3:
        raise AnalysisError("invalid_spec", "Subject, profile and score must be different columns.")
    data = c.source(data)
    if set(names) - set(data.columns):
        raise AnalysisError("missing_columns", "Subject/profile/score columns must all be present.")
    if data[[subject, profile]].isna().any().any():
        raise AnalysisError("missing_ids", "Missing subject/profile IDs cannot be silently assigned or dropped.")
    c.c.require_numeric(data, [score])
    if pd.api.types.is_bool_dtype(data[score].dtype):
        raise AnalysisError("invalid_response", "Scored conjoint needs numeric preference scores, not booleans.")
    labels = [c.label(v) for v in data.index]
    ids = [c.label(v) for v in plan[profile]]
    profile_lookup = {c.key(v): i for i, v in enumerate(ids)}
    groups = {}
    for position, (who, card, value) in enumerate(data[names].itertuples(index=False, name=None)):
        who, card = c.label(who), c.label(card)
        if c.key(card) not in profile_lookup:
            raise AnalysisError("unknown_profile", "Responses contain a profile outside the declared training/validation plan.")
        group = groups.setdefault(c.key(who), {"subject": who, "rows": {}})
        index = profile_lookup[c.key(card)]
        if index in group["rows"]:
            raise AnalysisError("duplicate_response", "A subject/profile pair has more than one response.")
        missing_score = pd.isna(value)
        if not missing_score and (not math.isfinite(float(value)) or abs(float(value)) > 1e12):
            raise AnalysisError("invalid_response", "Scores must be finite and within +/-1e12.")
        group["rows"][index] = {"score": None if missing_score else float(value),
                               "source_position": position, "source_label": labels[position]}
    if len(groups) > 128:
        raise AnalysisError("resource_limit", "At most 128 respondents are supported.")
    used, dropped = [], []
    for group in groups.values():
        complete = len(group["rows"]) == len(ids) and all(v["score"] is not None for v in group["rows"].values())
        if not complete:
            if missing == "raise":
                raise AnalysisError("incomplete_subject", "Every subject must score every declared profile exactly once.")
            dropped.append({"subject": group["subject"], "reason": "incomplete/missing score"})
        else:
            used.append({"subject": group["subject"],
                         "scores": [group["rows"][i]["score"] for i in range(len(ids))],
                         "positions": [group["rows"][i]["source_position"] for i in range(len(ids))],
                         "labels": [group["rows"][i]["source_label"] for i in range(len(ids))]})
    if not used:
        raise AnalysisError("empty_sample", "No complete respondents remain.")
    return used, dropped, len(data)


def inference(estimate, variance, df, level):
    if not math.isfinite(estimate) or not math.isfinite(variance) or variance < -1e-12:
        raise AnalysisError("numerical_failure", "A utility estimate/covariance is nonfinite or materially negative.")
    se = math.sqrt(max(0, variance))
    critical = dist.t_isf((1-level)/2, df)
    statistic = estimate/se if se > 0 else None
    p_value = min(1.0, 2*dist.t_sf(abs(statistic), df)) if statistic is not None else None
    return [estimate, se, df, statistic, p_value, estimate-critical*se, estimate+critical*se]


def individual_covariance(x, q, bread, residual, method, cluster_ids):
    """Full profile-score sandwich, without ridge, profile deletion or clipping."""
    n, p = x.shape
    multiplier, df, cluster_count = 1.0, n-p, None
    if method == "nonrobust":
        return bread*(float(residual@residual)/df), df, multiplier, cluster_count
    scores = x*residual[:, None]
    if method in ("hc2", "hc3"):
        complement = 1-torch.sum(q*q, dim=1)
        if bool((complement <= 1e-12).any()):
            raise AnalysisError("unit_leverage", "HC2/HC3 require every profile leverage to be below one.")
        scores = scores/complement[:, None]**(0.5 if method == "hc2" else 1.0)
    if method.startswith("cr"):
        lookup = {}
        codes = []
        for value in cluster_ids:
            codes.append(lookup.setdefault(c.key(value), len(lookup)))
        cluster_count = len(lookup)
        if cluster_count < 2:
            raise AnalysisError("insufficient_clusters", "Each retained respondent needs at least two declared clusters.")
        grouped = torch.zeros((cluster_count, p), dtype=torch.float64)
        grouped.index_add_(0, torch.tensor(codes, dtype=torch.int64), scores)
        scores, df = grouped, cluster_count-1
        if method == "cr1":
            multiplier = cluster_count/(cluster_count-1)*(n-1)/(n-p)
    elif method == "hc1":
        multiplier = n/(n-p)
    covariance = multiplier* bread@(scores.T@scores)@bread
    return (covariance+covariance.T)/2, df, multiplier, cluster_count


@resident_cpu
def conjoint_fit(plan, responses, attributes=None, *, factors=None, subject="subject", profile="profile_id",
                 score="score", missing="raise", covariance="nonrobust", cluster=None, level=0.95,
                 max_work=100_000_000, device="cpu", weights=None):
    """Fit scored conjoint OLS with full iid, HC0..HC3 or within-subject CR0/CR1 covariance.

    Discrete factors use sum-zero effects coding. Linear and ideal/anti-ideal
    quadratic factors fit a centred/scaled design then export raw-scale
    coefficients and full transformed covariance. No concavity is forced.
    Missing='drop_subjects' excludes entire incomplete respondents explicitly.
    The default iid Gaussian t is exact; robust t is an approximate convention.
    Cluster names a response column, joined by the original subject/profile IDs.
    Equal-subject group utilities here remain descriptive; see conjoint_group_mean.
    """
    level = confidence(level)
    if covariance not in ("nonrobust", "hc0", "hc1", "hc2", "hc3", "cr0", "cr1"):
        raise AnalysisError("unsupported_covariance", "Choose nonrobust, hc0, hc1, hc2, hc3, cr0 or cr1.")
    clustered = covariance in ("cr0", "cr1")
    if clustered != (cluster is not None):
        raise AnalysisError("inapplicable_option", "A response cluster column is required only for cr0/cr1.")
    data, attributes = c.profiles(plan, attributes, profile=profile)
    attributes, modes, p = c.definition(attributes, factors)
    groups, dropped, original_n = scored(responses, data, subject, profile, score, missing)
    if clustered:
        c.name(cluster)
        response_frame = c.source(responses)
        if cluster in (subject, profile, score) or cluster not in response_frame.columns:
            raise AnalysisError("invalid_cluster", "Cluster must be a separate existing response column.")
        for group in groups:
            group["cluster_ids"] = [c.label(v) for v in response_frame.iloc[group["positions"]][cluster]]
    n, subjects = len(data), len(groups)
    extra = {} if covariance == "nonrobust" else {
        "robust_profile_scores_and_cluster_sums": 16*n*p+8*n,
    }
    work = None if covariance == "nonrobust" else subjects*(3*n*p*p+3*p**3)
    settings = c.guard("conjoint_fit", n, p, subjects, device=device, weights=weights,
                       max_work=max_work, work=work, extra_buffers=extra)
    if n <= p:
        raise AnalysisError("insufficient_df", "The training design needs more profiles than free coefficients.")
    x, terms, mapping, anchors, transform = c.coding(data, attributes, modes)
    singular = torch.linalg.svdvals(x)
    threshold = max(x.shape)*torch.finfo(torch.float64).eps*float(singular[0])
    if float(singular[-1]) <= threshold or float(singular[0]/singular[-1]) > 1e10:
        raise AnalysisError("unidentified_design", "Conjoint design is rank deficient or too ill-conditioned; revise the plan.")
    q, r = torch.linalg.qr(x, mode="reduced")
    inverse_r = torch.linalg.solve_triangular(r, torch.eye(p, dtype=torch.float64), upper=True)
    unit_covariance = inverse_r @ inverse_r.T
    df = n-p
    coefficients, covariances, utility_rows, diagnostics, fitted_rows, records = [], [], [], [], [], []
    raw_parameters = []
    for group in groups:
        who = group["subject"]
        y = torch.tensor(group["scores"], dtype=torch.float64)
        # Remove an arbitrary large score origin before QR; restore only intercept.
        origin = float(y.mean())
        beta = torch.linalg.solve_triangular(r, (q.T@(y-origin))[:, None], upper=True).flatten()
        fitted_centered = x@beta
        residual = (y-origin) - fitted_centered
        beta[0] += origin
        sse = float(residual@residual)
        covariance_scaled, inference_df, multiplier, cluster_count = individual_covariance(
            x, q, unit_covariance, residual, covariance, group.get("cluster_ids"))
        raw_beta = transform@beta
        raw_cov = transform@covariance_scaled@transform.T
        if not torch.isfinite(raw_beta).all() or not torch.isfinite(raw_cov).all():
            raise AnalysisError("numerical_failure", "Conjoint coefficient/covariance transformation overflowed.")
        raw_parameters.append(raw_beta)
        for j, term in enumerate(terms):
            coefficients.append([who, term, *inference(float(raw_beta[j]), float(raw_cov[j, j]), inference_df, level)])
            covariances.append([who, term, *raw_cov[j].tolist()])
        for item in mapping:
            attr = item["attribute"]
            for value in attributes[attr]:
                vector = c.level_vector(item, value, attributes[attr], p)@transform
                estimate = float(vector@beta)
                variance = float(vector@covariance_scaled@vector)
                utility_rows.append([who, attr, value, *inference(estimate, variance, inference_df, level)])
        fitted = origin+fitted_centered
        total = float(((y-y.mean())**2).sum())
        diagnostics.append([who, n, p, df, sse, sse/df, None if total == 0 else 1-sse/total,
                            float(singular[0]/singular[-1])])
        fitted_rows.extend([who, c.label(card), float(yy), float(pred), float(res)]
                           for card, yy, pred, res in zip(data[profile], y, fitted, residual))
        records.append({"subject": who, "parameters_scaled": beta.tolist(), "covariance_scaled": covariance_scaled.tolist(),
                        "df": inference_df, "sample_positions": group["positions"], "sample_labels": group["labels"]})
        if covariance != "nonrobust":
            records[-1].update(residual_df=df, covariance_multiplier=multiplier,
                               n_clusters=cluster_count, cluster_ids=group.get("cluster_ids"))
    group_beta = torch.stack(raw_parameters).mean(0)
    group_utilities = [[item["attribute"], value, float(c.level_vector(item, value, attributes[item["attribute"]], p)@group_beta)]
                       for item in mapping for value in attributes[item["attribute"]]]
    state = {"attributes": attributes, "attribute_order": list(attributes), "modes": modes, "terms": terms,
             "anchors": anchors, "mapping": mapping, "transform": transform.tolist(), "subjects": records,
             "training_profile_ids": [c.label(v) for v in data[profile]], "profile_column": profile,
             "training_profiles": [[c.label(v) for v in row] for row in data[list(attributes)].itertuples(index=False, name=None)],
             "response_columns": {"subject": subject, "profile": profile, "score": score},
             "original_response_rows": original_n, "dropped_subjects": dropped, "missing": missing,
             "covariance": covariance, "level": level, "settings": settings,
             "solver": "direct normalized full-rank float64 QR", "converged": True}
    if clustered:
        state["cluster_column"] = cluster
    infer_note = ("conditional iid individual OLS t; group descriptive" if covariance == "nonrobust" else
                  "approximate individual robust t; within-subject clusters only; group descriptive")
    inference_columns = ["estimate", "std_error", "df", "t", "p_value", "ci_low", "ci_high"]
    return c.seal("conjoint_fit", {
        "coefficients": table(coefficients, columns=["subject", "term", *inference_columns]),
        "covariance": table(covariances, columns=["subject", "term", *terms]),
        "utilities": table(utility_rows, columns=["subject", "attribute", "level", *inference_columns]),
        "group_utilities": table(group_utilities, columns=["attribute", "level", "mean_individual_utility"]),
        "group_coefficients": table([[t, float(b)] for t, b in zip(terms, group_beta)], columns=["term", "mean_individual_coefficient"]),
        "diagnostics": table(diagnostics, columns=["subject", "profiles", "coefficients", "df", "sse", "sigma2", "r_squared", "condition"]),
        "fitted": table(fitted_rows, columns=["subject", "profile_id", "observed", "fitted", "residual"]),
        "training_plan": table(data.to_numpy().tolist(), columns=data.columns),
    }, state, n_subjects=subjects, n_profiles=n, inference=infer_note,
       covariance=covariance, group_weighting="equal complete subjects", **settings)


def evaluated(result, plan, profile, device, weights, max_work):
    state = c.intact(result, "conjoint_fit")
    attributes = {a: state["attributes"][a] for a in state["attribute_order"]}
    data, _ = c.profiles(plan, attributes, profile=profile)
    modes, p = state["modes"], len(state["terms"])
    settings = c.guard("conjoint_predict", len(data), p, len(state["subjects"]),
                       device=device, weights=weights, max_work=max_work)
    x, terms, _, _, _ = c.coding(data, attributes, modes, anchors=state["anchors"])
    if terms != state["terms"]:
        raise AnalysisError("invalid_saved_result", "Saved term order does not match attribute coding.")
    scores = torch.stack([x@torch.tensor(r["parameters_scaled"], dtype=torch.float64) for r in state["subjects"]])
    if not torch.isfinite(scores).all():
        raise AnalysisError("numerical_failure", "Predicted utilities are nonfinite.")
    return state, data, x, scores, settings


@resident_cpu
def conjoint_predict(result, plan, *, profile="profile_id", level=0.95,
                     max_work=100_000_000, device="cpu", weights=None):
    """Predict saved subject/profile means with full iid or approximate robust t covariance.

    Only declared levels are accepted. No response-noise interval, refit,
    extrapolation or cross-subject covariance is inferred.
    """
    level = confidence(level)
    state, data, x, scores, settings = evaluated(result, plan, profile, device, weights, max_work)
    rows = []
    for i, record in enumerate(state["subjects"]):
        covariance = torch.tensor(record["covariance_scaled"], dtype=torch.float64)
        variances = torch.sum((x@covariance)*x, dim=1)
        for card, estimate, variance in zip(data[profile], scores[i], variances):
            infer = inference(float(estimate), float(variance), record["df"], level)
            rows.append([record["subject"], c.label(card), infer[0], infer[1], infer[2], infer[5], infer[6]])
    return c.seal("conjoint_predict", {
        "predictions": table(rows, columns=["subject", "profile_id", "predicted", "std_error", "df", "ci_low", "ci_high"]),
    }, {"fit_integrity_sha256": result.attrs["integrity_sha256"], "profile_ids": [c.label(v) for v in data[profile]],
        "level": level, "settings": settings},
        inference=("conditional fitted mean iid t; not future response" if state["covariance"] == "nonrobust" else
                   "conditional fitted mean approximate robust t; not future response"), **settings)
