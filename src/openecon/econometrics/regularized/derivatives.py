"""Analytic derivatives of saved Gaussian local estimators; no tuning or inference."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral

import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes


def _fail(code, message):
    raise AnalysisError(code, message)


def _prepare(result, data, max_work):
    from openecon.dataset import Dataset

    if not isinstance(result, ResultBundle) or result.spec.estimator not in ("kernelreg", "localreg"):
        _fail("invalid_result", "Supply a saved kernelreg or localreg ResultBundle.")
    if isinstance(data, Dataset):
        _fail("streaming_unsupported", "Saved local derivatives require a resident query table.")
    if isinstance(max_work, bool) or not isinstance(max_work, Integral) or max_work < 1:
        _fail("invalid_option", "max_work must be a positive integer.")
    state = result.extra.get("smoother_state")
    cols = result.extra.get("terms")
    if not isinstance(state, dict) or not isinstance(cols, list) or not cols or cols != result.spec.predictors:
        _fail("invalid_state", "Saved local state and predictor order must match the model specification.")
    if state.get("kernel") != "gaussian":
        _fail("unsupported_derivative", "This analytic derivative route supports the smooth Gaussian kernel only.")
    degree = 0 if result.spec.estimator == "kernelreg" else 1
    if state.get("degree") != degree or result.extra.get("target") != "prediction":
        _fail("invalid_state", "Saved local degree/target does not match the estimator.")
    p = len(cols)
    train_x, train_y = state.get("training_x"), state.get("training_y")
    bandwidth = state.get("bandwidth")
    if not isinstance(train_x, list) or not isinstance(train_y, list) or not isinstance(bandwidth, list):
        _fail("invalid_state", "Saved training observations and bandwidth must be explicit lists.")
    n = len(train_y)
    if not 1 <= p <= 16 or not max(4, p + 2) <= n <= 100000 or len(train_x) != n or len(bandwidth) != p:
        _fail("invalid_state", "Saved derivative geometry needs 1–16 predictors and bounded matching training rows.")
    if any(not isinstance(row, list) or len(row) != p for row in train_x):
        _fail("invalid_state", "Saved training matrix rows must match the predictor count.")
    raw = _coerce_frame(data)
    q = len(raw)
    if not 1 <= q <= 8192 or raw.columns.has_duplicates or any(c not in raw for c in cols):
        _fail("invalid_predictors", "Supply 1–8192 query rows and every unique saved predictor column.")
    work = q * n * (p + 1) ** 2 * 8
    if work > max_work:
        _fail("work_limit", f"Saved local derivatives plan {work} scalar work units.")
    plan = plan_workspace(
        "saved Gaussian local derivatives",
        {
            "training_local_svd_and_derivative_buffers": tensor_bytes((n, p + 2), itemsize=256),
            "query_derivative_outputs": tensor_bytes((q, p + 4), itemsize=64),
            "local_factors": tensor_bytes((p + 1, p + 1), itemsize=64),
        },
    ).record()
    try:
        x = torch.tensor(train_x, dtype=torch.float64)
        y = torch.tensor(train_y, dtype=torch.float64)
        h = torch.tensor(bandwidth, dtype=torch.float64)
        minimum = float(state["min_effective"])
        encoded = json.dumps({"state": state, "spec": result.spec.model_dump(mode="json")},
                             sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, KeyError, RuntimeError):
        _fail("invalid_state", "Saved local state must contain finite real observations and settings.")
    if y.shape != (n,) or not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()) \
            or not bool(torch.isfinite(h).all()) or not bool((h > 0).all()) \
            or not 1 <= minimum <= n or state.get("support") not in ("raise", "extrapolate"):
        _fail("invalid_state", "Saved local observations, bandwidth or effective-support contract is invalid.")
    query = torch.stack([_numeric(raw[c], c) for c in cols], 1)
    low, high = x.min(0).values, x.max(0).values
    # Derivatives are deliberately restricted even if the original mean predictor
    # was explicitly allowed to extrapolate.
    if bool(((query < low) | (query > high)).any()):
        _fail("outside_support", "Local derivatives require queries inside the saved training bounding box.")
    return raw, query, x, y, h, cols, degree, minimum, plan, work, hashlib.sha256(encoded.encode()).hexdigest()


@resident_cpu
def local_derivatives(result: ResultBundle, *, data, max_work=200000000):
    """Differentiate a saved Gaussian kernelreg/localreg mean with bandwidth held fixed.

    Returns the estimator-mean derivative in each original predictor unit, with
    query indices preserved. The local-linear derivative includes changing kernel
    weights and local geometry; it is not just the fitted local slope. Bandwidth
    selection is never repeated. No standard error, confidence interval, causal
    interpretation, categorical derivative or extrapolation is supplied.
    """
    raw, query, x, y, h, cols, degree, minimum, plan, work, digest = _prepare(result, data, max_work)
    means, derivatives, counts = [], [], []
    for point in query:
        u = (x - point) / h
        exponent = -.5 * u.square().sum(1)
        if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(exponent).all()):
            _fail("numerical_failure", "Saved Gaussian local distances exceed finite float64 arithmetic.")
        weight = torch.softmax(exponent, 0)
        effective = float(1 / weight.square().sum())
        if effective < minimum - 1e-10:
            _fail("insufficient_local_support", "Effective local sample is below the saved minimum.")
        score = (u - weight @ u) / h
        if degree == 0:
            value = weight @ y
            gradient = weight @ ((y - value)[:, None] * score)
        else:
            design = torch.cat((torch.ones((len(x), 1), dtype=torch.float64), u), 1)
            root = weight.sqrt()
            weighted = design * root[:, None]
            singular = torch.linalg.svdvals(weighted)
            if float(singular[-1]) <= 1e-10 * float(singular[0]):
                _fail("singular_local_design", "Saved Gaussian support does not identify every local slope.")
            beta = torch.linalg.lstsq(weighted, y * root, driver="gelsd").solution
            e0 = torch.zeros(len(cols) + 1, dtype=torch.float64)
            e0[0] = 1
            influence = torch.linalg.lstsq(weighted.T, e0, driver="gelsd").solution * root
            residual = y - design @ beta
            value = beta[0]
            gradient = beta[1:] / h + influence @ (residual[:, None] * score)
        if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
            _fail("numerical_failure", "Saved local derivative exceeds finite float64 arithmetic.")
        means.append(float(value))
        derivatives.append(gradient.tolist())
        counts.append(effective)
    rows = [[mean, count, *gradient] for mean, count, gradient in zip(means, counts, derivatives, strict=True)]
    return table(
        rows, columns=["mean", "effective_n", *[f"d_mean_d_{c}" for c in cols]], index=list(raw.index),
        title="Saved Gaussian local derivatives", source_result_id=result.id,
        state_sha256=digest, procedure="local_derivatives", estimator=result.spec.estimator,
        predictors=cols, bandwidth=h.tolist(), query_rows=len(raw),
        target="derivative of saved conditional-mean estimator; bandwidth fixed",
        inference="unavailable; smoothing bias and bandwidth-selection uncertainty excluded",
        precision="float64", device="cpu", resource_plan=plan, planned_work=work,
    )


@resident_cpu
def local_average_derivatives(result: ResultBundle, *, data, max_work=200000000):
    """Average saved Gaussian estimator derivatives equally over supplied query rows.

    This descriptive empirical average has no population-weight, causal or
    inferential claim. All query observations must be finite and inside support.
    """
    derivatives = local_derivatives(result, data=data, max_work=max_work)
    names = derivatives.attrs["predictors"]
    values = torch.tensor(derivatives[[f"d_mean_d_{c}" for c in names]].to_numpy(), dtype=torch.float64)
    average = (values / len(values)).sum(0)
    if not bool(torch.isfinite(average).all()):
        _fail("numerical_failure", "Average local derivative exceeds finite float64 arithmetic.")
    return table(
        [[name, float(value)] for name, value in zip(names, average, strict=True)],
        columns=["predictor", "average_derivative"],
        **{**derivatives.attrs, "title": "Average saved Gaussian local derivatives",
           "procedure": "local_average_derivatives", "query_average": "equal weight per supplied query row"},
    )
