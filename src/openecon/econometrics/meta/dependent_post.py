"""Strict saved-fit inference and whole-study sensitivity for dependent effects."""

from __future__ import annotations

import json
import math
from numbers import Real

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _json_scalar, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.engines.distributions import chi2_sf, f_sf
from openecon.resources import plan_workspace
from .common import critical, level_check, probability
from .dependent import _labels, cluster_covariance, load_state, make_result, meta_dependent, seal_state

ROBUST_SOURCE = "https://wviechtb.github.io/metafor/reference/robust.html"
PREDICTION_SOURCE = "https://wviechtb.github.io/metafor/reference/predict.rma.html"
MAX_POST_ROWS = 256
MAX_DIAGNOSTIC_WORK = 100_000_000


def _identity(value):
    value = _json_scalar(value)
    return json.dumps([type(value).__name__, value], allow_nan=False, separators=(",", ":"))


def _level(state, requested):
    return level_check(state["level"] if requested is None else requested)


def _settings(state, procedure, level, **extra):
    return {
        "procedure": procedure,
        "model": state["model"],
        "method": state["method"],
        "inference": state["inference"],
        "residual_df": state["residual_df"],
        "level": level,
        "n_effects": state["n_effects"],
        "n_studies": state["n_studies"],
        "prediction_state": seal_state(state),
        "refitted": False,
        "variance_component_uncertainty_integrated": False,
        "device": "cpu",
        "dtype": "float64",
        **extra,
    }


def _positive_matrix(value, message):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", message)
    eigenvalues = torch.linalg.eigvalsh((value + value.T) * .5)
    if float(eigenvalues[-1]) <= 0 or float(eigenvalues[0]) <= float(eigenvalues[-1]) * 1e-12:
        raise AnalysisError("singular_covariance", message)


def _nullable(records, columns=None):
    result = table(records, columns=columns)
    return result.astype(object).where(result.notna(), None)


@resident_cpu
@torch.no_grad()
def meta_dependent_robust(result, *, correction="CR1", level=None):
    """Replace saved GLS uncertainty with whole-study CR0/CR1 sandwich inference.

    The coefficients and variance component remain fitted values. Scores are
    X_g' V_g^-1 r_g; CR1 multiplies their sandwich by G/(G-p). Both choices use
    t(G-p) individual and F(m,G-p) joint tests. Requires G>p and full-rank
    sandwich uncertainty. No CR2/Satterthwaite, refit, or tau² integration.
    """
    state, values = load_state(result)
    if correction not in ("CR0", "CR1"):
        raise AnalysisError("invalid_correction", "correction must be CR0 or CR1.")
    level = _level(state, level)
    x, y, s, beta = (values[name] for name in ("x", "y", "s", "beta"))
    clusters = values["cluster_index"]
    g, p = state["n_studies"], len(state["terms"])
    if g <= p:
        raise AnalysisError("insufficient_clusters", "Study-robust inference requires G>p.")
    covariance = cluster_covariance(state, values, correction)
    _positive_matrix(covariance, "Study scores do not identify full-rank coefficient uncertainty.")
    state.update(covariance=covariance.tolist(), inference="cluster", residual_df=g-p,
                 cluster_adjustment=correction, level=level)
    output = make_result(seal_state(state), procedure="meta_dependent_robust",
                         title="Dependent-effect meta-analysis: study-robust inference")
    residual = y-x@beta
    used_columns = set(state["terms"])
    metadata_columns = {}
    for role in ("study", "n_effects"):
        name = role
        if name in used_columns:
            name = "__"+role+"__"
            while name in used_columns:
                name += "_"
        used_columns.add(name)
        metadata_columns[role] = name
    scores = []
    for group in range(g):
        keep = (clusters == group).nonzero().flatten()
        block = s[keep][:, keep].clone()
        if state["model"] == "effect":
            block += state["tau2"]*torch.eye(len(keep), dtype=torch.float64)
        elif state["model"] == "study":
            block += state["tau2"]*torch.ones_like(block)
        score = x[keep].T@torch.linalg.solve(block, residual[keep])
        scores.append({metadata_columns["study"]: state["studies"][group],
                       metadata_columns["n_effects"]: len(keep),
                       **{name: float(value) for name, value in zip(state["terms"], score)}})
    output["cluster_scores"] = table(scores)
    indices = list(range(int(state["intercept"]), p))
    if indices:
        coefficient = beta[indices]
        moderator_covariance = covariance[indices][:, indices]
        m = len(indices)
        statistic = float(coefficient@torch.linalg.solve(moderator_covariance, coefficient))/m
        output["moderator_test"] = table([{"test": "joint moderators", "statistic": statistic,
                                          "distribution": "F", "df1": m, "df2": g-p,
                                          "p_value": f_sf(statistic, m, g-p)}])
    output.attrs.update(correction=correction, cluster_multiplier=g/(g-p) if correction == "CR1" else 1.,
                        refitted=False, source=ROBUST_SOURCE)
    output.attrs["cluster_score_metadata_columns"] = metadata_columns
    return output


def _contrasts(contrasts, terms):
    if not isinstance(contrasts, pd.DataFrame):
        raise AnalysisError("invalid_contrasts", "contrasts must be a DataFrame with named coefficient columns.")
    if not 1 <= len(contrasts) <= len(terms) or contrasts.columns.has_duplicates:
        raise AnalysisError("invalid_contrasts", "Supply 1..p independent contrasts with unique coefficient columns.")
    if set(contrasts.columns) != set(terms):
        raise AnalysisError("invalid_contrasts", "Contrast columns must name every fitted coefficient exactly once.")
    labels = _labels(contrasts.index, unique=True, name="contrast")
    if len({_identity(value) for value in labels}) != len(labels):
        raise AnalysisError("duplicate_contrast", "Contrast row labels must be distinct.")
    columns = []
    for name in terms:
        if pd.api.types.is_bool_dtype(contrasts[name].dtype):
            raise AnalysisError("invalid_contrasts", "Contrast weights must be real numeric values, not booleans.")
        columns.append(_numeric(contrasts[name], name))
    matrix = torch.stack(columns, 1)
    singular = torch.linalg.svdvals(matrix)
    if float(singular[0]) <= 0 or float(singular[-1]) <= float(singular[0])*1e-10:
        raise AnalysisError("singular_contrasts", "The requested joint contrasts must be independent and nonzero.")
    return labels, matrix


def _null(null, labels):
    if null is None:
        return torch.zeros(len(labels), dtype=torch.float64)
    if isinstance(null, Real) and not isinstance(null, bool):
        if not math.isfinite(null):
            raise AnalysisError("invalid_null", "The null must be finite.")
        return torch.full((len(labels),), float(null), dtype=torch.float64)
    if isinstance(null, pd.Series):
        if [_identity(value) for value in null.index] != [_identity(value) for value in labels]:
            raise AnalysisError("invalid_null", "A labelled null Series must match contrast row labels in order.")
        raw = null.tolist()
    elif isinstance(null, (list, tuple)):
        raw = list(null)
    else:
        raise AnalysisError("invalid_null", "Use a finite scalar, ordered list, or matching labelled Series null.")
    if len(raw) != len(labels) or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in raw):
        raise AnalysisError("invalid_null", "Provide exactly one finite numeric null for each contrast.")
    return torch.tensor(raw, dtype=torch.float64)


@resident_cpu
@torch.no_grad()
def meta_dependent_contrast(result, *, contrasts, null=None, level=None):
    """Test named saved-coefficient contrasts against finite scalar/vector nulls.

    Preserves full L C L' covariance and reports individual z/t intervals and
    joint chi²/F tests according to saved model/cluster inference. Contrast
    weights must be a labelled DataFrame with every coefficient column; 1..p
    independent rows. No fitted parameter, tau², or observation is recomputed.
    """
    state, values = load_state(result)
    level = _level(state, level)
    labels, matrix = _contrasts(contrasts, state["terms"])
    nulls = _null(null, labels)
    estimate = matrix@values["beta"]
    covariance = matrix@values["covariance"]@matrix.T
    covariance = (covariance+covariance.T)*.5
    _positive_matrix(covariance, "The requested contrast uncertainty is singular or non-finite.")
    se = covariance.diagonal().sqrt()
    difference = estimate-nulls
    statistic = difference/se
    inference, df = state["inference"], state["residual_df"]
    multiplier = critical(level, inference, df)
    wald = float(difference@torch.linalg.solve(covariance, difference))
    if not bool(torch.isfinite(estimate).all()) or not bool(torch.isfinite(statistic).all()) or not math.isfinite(wald):
        raise AnalysisError("numerical_failure", "The requested contrast estimate or test is non-finite; rescale contrast weights.")
    m = len(labels)
    joint = {"null": nulls.tolist(), "statistic": wald if inference == "z" else wald/m,
             "distribution": "chi2" if inference == "z" else "F", "df1": m,
             "df2": None if inference == "z" else df,
             "p_value": chi2_sf(wald, m) if inference == "z" else f_sf(wald/m, m, df)}
    return TableSet({
        "contrasts": table({"contrast": labels, "estimate": estimate.tolist(), "null": nulls.tolist(),
                            "std_error": se.tolist(), "statistic": statistic.tolist(),
                            "p_value": [probability(float(v), inference, df) for v in statistic],
                            "ci_low": (estimate-multiplier*se).tolist(),
                            "ci_high": (estimate+multiplier*se).tolist()}),
        "covariance": table(covariance.tolist(), columns=labels, index=labels),
        "contrast_weights": table(matrix.tolist(), columns=state["terms"], index=labels),
        "joint_test": table([joint]),
    }, title="Dependent-effect meta-analysis: saved coefficient contrasts",
        **_settings(state, "meta_dependent_contrast", level, joint_null=nulls.tolist()))


def _new_design(data, state, *, latent):
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "Dependent-effect predictions require an in-memory table.")
    frame = _coerce_frame(data)
    if not 1 <= len(frame) <= MAX_POST_ROWS:
        raise AnalysisError("resource_limit", "Prediction requires 1..256 explicit rows.")
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Prediction columns must be unique.")
    names = state["moderators"]
    if any(name not in frame.columns for name in names):
        raise AnalysisError("missing_columns", "Supply every fitted moderator in the prediction table.")
    n, p, m = state["n_effects"], len(state["terms"]), len(frame)
    resource = plan_workspace("dependent meta joint prediction", {
        "retained_sampling_design_and_state": 24*n*n*8+32*(n*p+p*p+n)*8+512*(n*n+n*p+1024),
        "new_design_mean_interval_and_labels": 32*(m*p+m+p*p)*8,
        "joint_prediction_matrices_and_copies": (8 if latent else 4)*m*m*8,
    }).record()
    parts = [torch.ones(len(frame), dtype=torch.float64)] if state["intercept"] else []
    for name in names:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            raise AnalysisError("invalid_numeric", "Prediction moderators cannot be boolean.")
        values = _numeric(frame[name], name)
        if float(values.abs().max()) > 1e8:
            raise AnalysisError("numeric_domain", "Prediction moderators must lie in +/-1e8; rescale explicitly.")
        parts.append(values)
    return frame, torch.stack(parts, 1), resource


def _predict(result, data, level, study):
    state, values = load_state(result)
    level = _level(state, level)
    frame, x, resource = _new_design(data, state, latent=study is not None)
    mean = x@values["beta"]
    mean_covariance = x@values["covariance"]@x.T
    mean_covariance = (mean_covariance+mean_covariance.T)*.5
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(mean_covariance).all()):
        raise AnalysisError("numerical_failure", "Prediction mean/covariance is non-finite; rescale moderators.")
    variance = mean_covariance.diagonal()
    tolerance = max(1., float(mean_covariance.abs().max()))*1e-12
    if bool((variance < -tolerance).any()):
        raise AnalysisError("numerical_failure", "Prediction mean covariance is not positive semidefinite.")
    rows = list(range(len(frame)))
    labels = [_json_scalar(value) for value in frame.index]
    se = variance.clamp_min(0).sqrt()
    multiplier = critical(level, state["inference"], state["residual_df"])
    prediction = {"row": rows, "original_label": labels, "estimate": mean.tolist(),
                  "std_error": se.tolist(), "ci_low": (mean-multiplier*se).tolist(),
                  "ci_high": (mean+multiplier*se).tolist()}
    tables = {"prediction": table(prediction),
              "mean_covariance": table(mean_covariance.tolist(), columns=rows, index=rows),
              "design": table(x.tolist(), columns=state["terms"], index=rows)}
    metadata = {"source": PREDICTION_SOURCE, "resource_plan": resource, "prediction_rows": rows,
                "prediction_labels": labels, "future_sampling_error_included": False,
                "conditional_on_estimated_variance_component": True}
    if study is not None:
        if not isinstance(study, str) or not study or study not in frame.columns or study in state["moderators"]:
            raise AnalysisError("invalid_study", "study must name a distinct complete future-study column.")
        if bool(frame[study].isna().any()):
            raise AnalysisError("missing_values", "Future-study labels must be complete.")
        study_labels = _labels(frame[study], name="future study")
        typed = [_identity(value) for value in study_labels]
        if state["model"] == "study":
            _labels([*state["studies"], *study_labels], name="training/future study")
            if set(typed) & {_identity(value) for value in state["studies"]}:
                raise AnalysisError("existing_study", "Study-model latent prediction requires new study labels; existing-study BLUP is unsupported.")
        latent = torch.zeros_like(mean_covariance)
        if state["model"] == "effect":
            latent = state["tau2"]*torch.eye(len(frame), dtype=torch.float64)
        elif state["model"] == "study":
            latent = state["tau2"]*torch.tensor([[left == right for right in typed] for left in typed], dtype=torch.float64)
        predictive = mean_covariance+latent
        sd = predictive.diagonal().clamp_min(0).sqrt()
        prediction.update(study=study_labels, latent_std_dev=latent.diagonal().sqrt().tolist(),
                          prediction_std_dev=sd.tolist(), prediction_low=(mean-multiplier*sd).tolist(),
                          prediction_high=(mean+multiplier*sd).tolist())
        tables["prediction"] = table(prediction)
        tables["latent_covariance"] = table(latent.tolist(), columns=rows, index=rows)
        tables["predictive_covariance"] = table(predictive.tolist(), columns=rows, index=rows)
        metadata.update(future_study_labels=study_labels, prediction_interval="plug-in latent true-effect interval")
    procedure = "meta_dependent_predict" if study is None else "meta_dependent_predict_effect"
    return TableSet(tables, title="Dependent-effect meta-analysis: joint future means" if study is None
                    else "Dependent-effect meta-analysis: joint future latent effects",
                    **_settings(state, procedure, level, **metadata))


@resident_cpu
@torch.no_grad()
def meta_dependent_predict(result, *, data, level=None):
    """Predict joint new-row conditional means using full saved coefficient covariance.

    Complete numeric moderators, 1..256 rows, no refit. Mean CI uses saved
    normal or study-robust t(G-p) inference. No future sampling variance,
    random-effect BLUP, variance-component uncertainty, or implicit categories.
    """
    return _predict(result, data, level, None)


@resident_cpu
@torch.no_grad()
def meta_dependent_predict_effect(result, *, data, study, level=None):
    """Predict joint future latent effects under the retained covariance model.

    Adds zero heterogeneity for common, tau² I for effect, or tau² shared-label
    study blocks for study. Study-model labels must name new studies. Reports
    full mean/latent/predictive covariance and plug-in z/t marginal intervals;
    excludes future sampling error, tau² uncertainty, existing-study BLUPs.
    """
    if not isinstance(study, str) or not study:
        raise AnalysisError("invalid_study", "study must name an explicit future-study column.")
    return _predict(result, data, level, study)


@resident_cpu
@torch.no_grad()
def meta_dependent_diagnostics(result):
    """Refit every whole-study deletion, retaining every failed attempted fit.

    Requires a saved complete dependent fit. Deletes the full study and its
    covariance block; re-estimates the same model/method and, where requested,
    study-robust uncertainty. Full retained effect/study IDs, coefficient and
    covariance tables, fit state, failure codes and attempted denominator are
    preserved. A conservative cumulative 100M work gate precedes all refits.
    No effect-row deletions, selection model or publication-bias decision.
    """
    state, values = load_state(result)
    clusters, x, y, s = (values[name] for name in ("cluster_index", "x", "y", "s"))
    n, p, g = state["n_effects"], len(state["terms"]), state["n_studies"]
    if g < 2:
        raise AnalysisError("insufficient_studies", "Whole-study deletion requires at least two studies.")
    # Matches the core's 1,024-evaluation static likelihood allowance. It is
    # a structural admission bound, not measured FLOPs or wall time.
    sizes = [int((clusters == group).sum()) for group in range(g)]
    estimated = sum(12*(n-size)**3 + 12*1024*((n-size)*p*p+p**3) for size in sizes)
    if estimated > MAX_DIAGNOSTIC_WORK:
        raise AnalysisError("resource_limit", "Complete whole-study deletion exceeds the cumulative 100M work gate.")
    resource = plan_workspace("dependent meta complete whole-study deletion", {
        "retained_original_covariance_design_and_state": 24*n*n*8+32*(n*p+p*p+n)*8+512*(n*n+n*p+1024),
        "working_deleted_covariance_design_and_fit": 24*n*n*8+32*(n*p+p*p+n)*8+512*(n*n+n*p+1024),
        "complete_deletion_states_and_uncertainty": 512*g*(n*n+n*p+p*p+1024),
    }).record()
    moderators = state["moderators"]
    reserved = set(moderators)
    roles = []
    for base in ("__meta_effect__", "__meta_yi__", "__meta_study__"):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        roles.append(name)
    effect_role, y_role, study_role = roles
    data = {effect_role: state["effect_ids"], y_role: y.tolist(), study_role: state["study_ids"]}
    data.update({name: x[:, i+int(state["intercept"])].tolist() for i, name in enumerate(moderators)})
    frame = pd.DataFrame(data, index=state["sample_labels"])
    deletion, coefficients, covariances, states = [], [], [], []
    for group, label in enumerate(state["studies"]):
        keep = (clusters != group).nonzero().flatten()
        retained = frame.iloc[keep.tolist()].copy()
        effect_ids = [state["effect_ids"][i] for i in keep.tolist()]
        block = pd.DataFrame(s[keep][:, keep].tolist(), index=effect_ids, columns=effect_ids)
        row = {"study": label, "removed_effects": sizes[group], "n_effects": len(keep),
               "n_studies": g-1, "status": "failed", "error_code": "", "message": "",
               "tau2": None, "max_abs_coefficient_change": None}
        try:
            fitted = meta_dependent(data=retained, yi=y_role, effect=effect_role, study=study_role,
                                    covariance=block, moderators=moderators.copy(), intercept=state["intercept"],
                                    model=state["model"], method=state["method"], level=state["level"])
            if state["inference"] == "cluster":
                fitted = meta_dependent_robust(fitted, correction=state["cluster_adjustment"])
            deleted_state, deleted_values = load_state(fitted)
            change = deleted_values["beta"]-values["beta"]
            row.update(status="ok", tau2=deleted_state["tau2"], max_abs_coefficient_change=float(change.abs().max()))
            for term, record in zip(state["terms"], fitted["coefficients"].to_dict("records")):
                coefficients.append({"study": label, "status": "ok", "error_code": "", **record})
            for i, left in enumerate(state["terms"]):
                for j, right in enumerate(state["terms"]):
                    covariances.append({"study": label, "status": "ok", "error_code": "", "term_row": left, "term_column": right,
                                        "covariance": float(deleted_values["covariance"][i, j])})
            states.append({"study": label, "retained_effect_ids": effect_ids,
                           "retained_positions": [state["sample_positions"][i] for i in keep.tolist()],
                           "prediction_state": deleted_state})
        except AnalysisError as error:
            row.update(error_code=error.code, message=str(error))
            for term in state["terms"]:
                coefficients.append({"study": label, "term": term, "status": "failed", "error_code": error.code,
                                     "estimate": None, "std_error": None, "statistic": None,
                                     "p_value": None, "df": None, "ci_low": None, "ci_high": None})
            for left in state["terms"]:
                for right in state["terms"]:
                    covariances.append({"study": label, "status": "failed", "error_code": error.code,
                                        "term_row": left, "term_column": right, "covariance": None})
            states.append({"study": label, "retained_effect_ids": effect_ids,
                           "retained_positions": [state["sample_positions"][i] for i in keep.tolist()],
                           "prediction_state": None, "error_code": error.code})
        deletion.append(row)
    successful = sum(row["status"] == "ok" for row in deletion)
    summary = {"attempted": g, "succeeded": successful, "failed": g-successful,
               "cumulative_work_estimate": estimated, "cumulative_work_limit": MAX_DIAGNOSTIC_WORK}
    return TableSet({"leave_one_study_out": _nullable(deletion),
                     "deletion_coefficients": _nullable(coefficients, ["study", "term", "status", "error_code", "estimate", "std_error", "statistic", "p_value", "df", "ci_low", "ci_high"]),
                     "deletion_covariance": _nullable(covariances, ["study", "term_row", "term_column", "status", "error_code", "covariance"]),
                     "summary": table([summary])},
                    title="Dependent-effect meta-analysis: whole-study deletion sensitivity",
                    **_settings(state, "meta_dependent_diagnostics", state["level"], refitted=True,
                                **summary, refit_states=states, resource_plan=resource,
                                notes=["Every deletion is attempted after cumulative admission; failed fits remain in the denominator.",
                                       "Deletion sensitivity does not identify publication bias or justify removing a study."]))
