"""Persistable penalized and general local conditional-mean prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes
from .kernels import bandwidth_vector, fit_penalized, local_predict, penalty_factors, work_guard


def _convenience(name, data, outcome, predictors, *, categorical=None, weights=None,
                 weight_type="aweight", intercept=True, missing="raise", **options):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=outcome,
            predictors=column_list(predictors, "predictors"),
            categorical=column_list(categorical, "categorical"),
            weights=weights,
            weight_type=weight_type,
            intercept=intercept,
            missing=missing,
            options=options,
        ),
        data=data,
    )


def ridge(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
          intercept=True, missing="raise", **options):
    """Ridge prediction; selection='fixed' needs penalty, default CV."""
    return _convenience("ridge", data, y, x, categorical=categorical, weights=weights,
                        weight_type=weight_type, intercept=intercept, missing=missing, **options)


def lasso(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
          intercept=True, missing="raise", **options):
    """Lasso prediction, with fixed/CV/heteroskedastic plug-in selection."""
    return _convenience("lasso", data, y, x, categorical=categorical, weights=weights,
                        weight_type=weight_type, intercept=intercept, missing=missing, **options)


def elasticnet(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
               intercept=True, missing="raise", **options):
    """Elastic-net prediction: l1_ratio=1 is lasso, 0 is ridge."""
    return _convenience("elasticnet", data, y, x, categorical=categorical, weights=weights,
                        weight_type=weight_type, intercept=intercept, missing=missing, **options)


def kernelreg(*, data, y, x, missing="raise", **options):
    """General Nadaraya-Watson regression, not regression discontinuity."""
    return _convenience("kernelreg", data, y, x, missing=missing, **options)


def localreg(*, data, y, x, missing="raise", **options):
    """General multivariate local-linear conditional-mean regression."""
    return _convenience("localreg", data, y, x, missing=missing, **options)


def _options(frame: ModelFrame):
    return {option.name: frame.option(option.name) for option in frame.info.options}


def _prediction_result(frame, fitted, extra, diagnostics):
    y = frame.numeric(frame.spec.outcome)
    metrics = {}
    if fitted is not None:
        mse = (y - fitted).square().mean()
        if not bool(torch.isfinite(mse)):
            raise AnalysisError(
                "numerical_failure", "Prediction error exceeds finite float64 arithmetic."
            )
        metrics["training_mse"] = float(mse)
    note = "Prediction target only; coefficient and pointwise confidence intervals are unavailable."
    return build_result(
        frame,
        terms=[],
        params=torch.empty(0, dtype=torch.float64),
        covariance=torch.empty((0, 0), dtype=torch.float64),
        use_t=False,
        fitted=fitted,
        metrics=metrics,
        solver=diagnostics,
        inference={
            "target": "prediction",
            "available": False,
            "distribution": "none",
            "use_t": None,
            "correction": "no coefficient inference",
        },
        extra={"target": "prediction", "notes": [note], **extra},
        warnings=[note],
    )


@resident_cpu
def _penalized(spec, data, ratio):
    if spec.weights or spec.categorical or isinstance(spec.options.get("penalty_factors"), dict):
        from .extended import fit_extended
        return fit_extended(spec, data, ratio=ratio)
    frame = ModelFrame(spec, data)
    n, p = frame.n, len(spec.predictors)
    frame.workspace_plan(
        "penalized regression and tuning",
        {
            "design_scaling_cv_copies": tensor_bytes((n, p), itemsize=72),
            "ridge_gram_and_factors": tensor_bytes((p, p), itemsize=48) if ratio == 0 else 0,
            "path_and_result_state": tensor_bytes((max(n, 200), p + 8), itemsize=64),
        },
    )
    x, y = frame.matrix(spec.predictors), frame.numeric(spec.outcome)
    options = _options(frame)
    factors = penalty_factors(options.get("penalty_factors"), p)
    forced = options.get("forced_controls") or []
    if len(forced) != len(set(forced)) or any(name not in spec.predictors for name in forced):
        raise AnalysisError("invalid_forced_controls", "forced_controls must be distinct names already present in x.")
    for name in forced:
        factors[spec.predictors.index(name)] = 0
    options["penalty_factors"] = factors.tolist()
    result = fit_penalized(x, y, ratio=ratio, intercept=spec.intercept, options=options)
    state = result["state"]
    state["forced_controls"] = [name for name, factor in zip(spec.predictors, state["penalty_factors"], strict=True) if factor == 0]
    return _prediction_result(
        frame,
        result["fitted"],
        {
            "terms": spec.predictors,
            "predictive_coefficients": dict(
                zip(spec.predictors, state["coefficients"], strict=True)
            ),
            "constant": state["constant"],
            "penalized_state": state,
        },
        "analytic ridge solve" if ratio == 0 else "cyclic coordinate descent, KKT verified",
    )


def fit_ridge(spec, data):
    return _penalized(spec, data, 0.0)


def fit_lasso(spec, data):
    return _penalized(spec, data, 1.0)


def fit_elasticnet(spec, data):
    return _penalized(spec, data, spec.options.get("l1_ratio", 0.5))


def _query(value: Any, p: int):
    try:
        q = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        raise AnalysisError(
            "invalid_query", "query must be a finite numeric m by p matrix."
        ) from None
    if q.ndim != 2 or q.shape[1] != p or len(q) == 0 or not bool(torch.isfinite(q).all()):
        raise AnalysisError(
            "invalid_query", "query must be a nonempty finite numeric m by p matrix."
        )
    return q


def _kernel(spec, data, degree):
    frame = ModelFrame(spec, data)
    n, p = frame.n, len(spec.predictors)
    if n < max(4, p + 2):
        raise AnalysisError(
            "insufficient_sample",
            "Local regression needs at least max(4, p+2) complete observations.",
        )
    o = _options(frame)
    query_raw = o["query"]
    if query_raw is not None and (not isinstance(query_raw, list) or len(query_raw) == 0):
        raise AnalysisError("invalid_query", "query must be a nonempty list of numeric rows.")
    m = n if query_raw is None else len(query_raw)
    candidates = o["bandwidth_path"]
    candidate_count = len(candidates) if isinstance(candidates, list) else 1
    if candidate_count > 100:
        raise AnalysisError("invalid_bandwidth", "At most 100 bandwidth candidates are supported.")
    work_guard(
        (m + (n * candidate_count if o["selection"] == "cv" else 0)) * n * (p + 1) ** 2,
        o["max_work"],
        "local regression and bandwidth selection",
    )
    frame.workspace_plan(
        "local regression, queries and saved training state",
        {
            "training_local_design_and_json_state": tensor_bytes((n, p + 2), itemsize=144),
            "query_and_diagnostics_state": tensor_bytes((m, p + 8), itemsize=64),
            "local_linear_factors": tensor_bytes((p + 1, p + 1), itemsize=64),
        },
    )
    x, y = frame.matrix(spec.predictors), frame.numeric(spec.outcome)
    query = x if query_raw is None else _query(query_raw, p)
    if o["selection"] == "fixed":
        if o["bandwidth"] is None or candidates is not None:
            raise AnalysisError(
                "invalid_bandwidth", "Fixed selection needs bandwidth and no bandwidth_path."
            )
        h = bandwidth_vector(o["bandwidth"], p)
        selection = {"method": "fixed"}
    elif o["selection"] == "plugin":
        if o["bandwidth"] is not None or candidates is not None:
            raise AnalysisError(
                "invalid_bandwidth",
                "Plug-in bandwidth is determined by the training sample; use selection='fixed' for a supplied bandwidth.",
            )
        sd = x.std(0, correction=1)
        iqr = torch.quantile(x, 0.75, dim=0) - torch.quantile(x, 0.25, dim=0)
        robust = torch.where(iqr > 0, torch.minimum(sd, iqr / 1.349), sd)
        h = bandwidth_vector((1.06 * robust * n ** (-1 / (p + 4))).tolist(), p)
        selection = {
            "method": "plugin",
            "formula": "1.06*min(sd,IQR/1.349)*n^(-1/(p+4))",
            "interpretation": "normal-reference pilot; not an optimal regression bandwidth claim",
        }
    else:
        if o["bandwidth"] is not None or not isinstance(candidates, list) or not candidates:
            raise AnalysisError(
                "invalid_bandwidth", "CV needs a nonempty bandwidth_path and no bandwidth."
            )
        bandwidths = [bandwidth_vector(candidate, p) for candidate in candidates]
        scores, rejected = [], []
        for h in bandwidths:
            try:
                cv, _ = local_predict(
                    x,
                    y,
                    x,
                    h,
                    degree=degree,
                    kernel=o["kernel"],
                    support=o["support"],
                    min_effective=o["min_effective"],
                    leave_out=torch.arange(n),
                )
                error = (y - cv).square().mean()
                if not bool(torch.isfinite(error)):
                    raise AnalysisError(
                        "numerical_failure", "Bandwidth CV error exceeds finite float64 arithmetic."
                    )
                scores.append(float(error))
                rejected.append(None)
            except AnalysisError as exc:
                if exc.code not in {
                    "empty_local_support",
                    "insufficient_local_support",
                    "singular_local_design",
                }:
                    raise
                scores.append(None)
                rejected.append(exc.code)
        valid = [i for i, score in enumerate(scores) if score is not None]
        if not valid:
            raise AnalysisError(
                "insufficient_local_support",
                "No bandwidth candidate has valid support for every leave-one-out fit.",
            )
        index = min(valid, key=lambda i: scores[i])
        h = bandwidths[index]
        selection = {
            "method": "leave-one-out CV",
            "candidate_bandwidths": [h.tolist() for h in bandwidths],
            "cv_mse": scores,
            "invalid_candidate_reasons": rejected,
            "selected_index": index,
        }
    estimates, details = local_predict(
        x,
        y,
        query,
        h,
        degree=degree,
        kernel=o["kernel"],
        support=o["support"],
        min_effective=o["min_effective"],
    )
    fitted = estimates if query_raw is None else None
    state = {
        "training_x": x.tolist(),
        "training_y": y.tolist(),
        "bandwidth": h.tolist(),
        "kernel": o["kernel"],
        "degree": degree,
        "support": o["support"],
        "min_effective": o["min_effective"],
        "selection": selection,
        "observed_bounds": [x.min(0).values.tolist(), x.max(0).values.tolist()],
        "boundary_definition": "within one bandwidth of an observed bounding-box face; local-linear intercept corrects first-order boundary bias",
        "support_definition": "observed bounding box plus nonzero kernel weights and effective sample; box inclusion alone is not support",
    }
    return _prediction_result(
        frame,
        fitted,
        {
            "terms": spec.predictors,
            "smoother_state": state,
            "query": query.tolist(),
            "query_estimates": estimates.tolist(),
            "query_diagnostics": details,
        },
        "querywise float64 weighted local least squares",
    )


def fit_kernelreg(spec, data):
    return _kernel(spec, data, 0)


def fit_localreg(spec, data):
    return _kernel(spec, data, 1)


@resident_cpu
def regularized_predict(result: ResultBundle, data, *, support=None, max_work=200000000):
    """Predict from persisted state; return row-preserving conditional means.

    Incomplete prediction inputs are rejected, rather than silently changing
    row alignment. Dataset inputs are explicitly unsupported.
    """
    from openecon.dataset import Dataset
    from openecon.analysis import _coerce_frame, _numeric

    if not isinstance(result, ResultBundle) or result.extra.get("target") != "prediction":
        raise AnalysisError(
            "invalid_result", "regularized_predict needs a predictive regularized-family result."
        )
    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "Saved regularized prediction requires an in-memory table; Dataset is not collected.",
        )
    if "regularized_extended_state" in result.extra:
        from .extended import predict_extended
        return predict_extended(result, data, max_work=max_work)
    data = _coerce_frame(data)
    names = result.extra["terms"]
    if data.columns.has_duplicates or any(name not in data for name in names):
        raise AnalysisError("invalid_predictors", "Prediction columns must be present and unique.")
    plan_workspace(
        "saved regularized prediction",
        {"query": tensor_bytes((len(data), len(names) + 2), itemsize=48)},
    )
    x = torch.stack([_numeric(data[name], name) for name in names], dim=1)
    if len(x) == 0:
        raise AnalysisError("empty_data", "Prediction needs at least one row.")
    if "penalized_state" in result.extra:
        state = result.extra["penalized_state"]
        work_guard(len(x) * len(names), max_work, "saved penalized prediction")
        predicted = x @ torch.tensor(state["coefficients"], dtype=torch.float64) + state["constant"]
    elif "smoother_state" in result.extra:
        state = result.extra["smoother_state"]
        if support not in (None, "raise", "extrapolate"):
            raise AnalysisError("invalid_support", "support must be raise or extrapolate.")
        nx, p = len(state["training_y"]), len(names)
        work_guard(len(x) * nx * (p + 1) ** 2, max_work, "saved local regression prediction")
        plan_workspace(
            "saved local regression",
            {
                "saved_training_local_copies": tensor_bytes((nx, p + 2), itemsize=96),
                "query_output": tensor_bytes((len(x), p + 8), itemsize=64),
            },
        )
        predicted, _ = local_predict(
            torch.tensor(state["training_x"], dtype=torch.float64),
            torch.tensor(state["training_y"], dtype=torch.float64),
            x,
            torch.tensor(state["bandwidth"], dtype=torch.float64),
            degree=state["degree"],
            kernel=state["kernel"],
            support=support or state["support"],
            min_effective=state["min_effective"],
        )
    else:
        raise AnalysisError("invalid_result", "The saved result has no predictive state.")
    if not bool(torch.isfinite(predicted).all()):
        raise AnalysisError(
            "numerical_failure", "Saved prediction exceeds finite float64 arithmetic."
        )
    return pd.Series(predicted.tolist(), index=data.index, name="predicted")


def regularized_table(result: ResultBundle):
    """Export actual predictive estimates with no coefficient-SE fiction."""
    if not isinstance(result, ResultBundle) or result.extra.get("target") != "prediction":
        raise AnalysisError("invalid_result", "regularized_table requires a predictive result.")
    if "regularized_extended_state" in result.extra:
        from .extended import _validate_result
        _validate_result(result)
    if "penalized_state" in result.extra:
        state = result.extra["penalized_state"]
        rows = [("Intercept", state["constant"])] if result.spec.intercept else []
        rows.extend(zip(result.extra["terms"], state["coefficients"], strict=True))
        return table(
            rows,
            columns=["Predictive term", "Estimate"],
            target="prediction",
            inference="unavailable",
            penalty=state.get("selected_penalty"),
            components=state.get("components"),
            notes=result.extra["notes"],
        )
    rows = [
        [*query, estimate]
        for query, estimate in zip(
            result.extra["query"], result.extra["query_estimates"], strict=True
        )
    ]
    return table(
        rows,
        columns=[*result.extra["terms"], "Conditional mean"],
        target="prediction",
        inference="unavailable",
        bandwidth=result.extra["smoother_state"]["bandwidth"],
        notes=result.extra["notes"],
    )
