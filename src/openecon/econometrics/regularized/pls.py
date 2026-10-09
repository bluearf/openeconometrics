"""Scalar-response PLS1; all component tuning and scaling is training-only."""

from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.resources import tensor_bytes

from .kernels import folds, work_guard
from .prediction import _convenience, _prediction_result


def pls(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
        intercept=True, missing="raise", **options):
    """PLS1 prediction with fixed or cross-validated component count; no coefficient CI."""
    return _convenience("pls", data, y, x, categorical=categorical, weights=weights,
                        weight_type=weight_type, intercept=intercept, missing=missing, **options)


def _fit(x, y, components, *, intercept, standardize):
    n, p = x.shape
    if components > min(p, n - int(intercept)):
        raise AnalysisError("invalid_components", "Components exceed training design dimensions.")
    center = x.mean(0) if intercept else torch.zeros(p, dtype=torch.float64)
    yc = y.mean() if intercept else torch.tensor(0.0, dtype=torch.float64)
    z = x - center
    scale = (
        z.square().sum(0).div(n - int(intercept)).sqrt()
        if standardize
        else torch.ones(p, dtype=torch.float64)
    )
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    z = z / scale
    residual = y - yc
    weights, loadings, response = [], [], []
    for _ in range(components):
        direction = z.T @ residual
        norm = torch.linalg.vector_norm(direction)
        floor = (
            torch.finfo(torch.float64).eps
            * 100
            * torch.linalg.vector_norm(z)
            * torch.linalg.vector_norm(residual)
        )
        if not bool(torch.isfinite(norm)) or float(norm) <= float(floor) or float(norm) == 0:
            raise AnalysisError(
                "rank_deficient", "Requested PLS component has no identifiable response direction."
            )
        w = direction / norm
        score = z @ w
        ss = score @ score
        if float(ss) <= torch.finfo(torch.float64).eps * 100 * float(z.square().sum()):
            raise AnalysisError(
                "rank_deficient", "Requested PLS component has no identifiable predictor score."
            )
        loading, q = z.T @ score / ss, residual @ score / ss
        weights.append(w)
        loadings.append(loading)
        response.append(q)
        z = z - score[:, None] * loading
        residual = residual - score * q
    ws, ps, qs = torch.stack(weights, 1), torch.stack(loadings, 1), torch.stack(response)
    coefficients = (ws @ torch.linalg.solve(ps.T @ ws, qs)) / scale
    constant = yc - center @ coefficients
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(constant)):
        raise AnalysisError("numerical_failure", "PLS prediction state is not finite.")
    return {
        "components": components,
        "coefficients": coefficients.tolist(),
        "constant": float(constant),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "response_center": float(yc),
        "x_weights": ws.tolist(),
        "x_loadings": ps.tolist(),
        "y_loadings": qs.tolist(),
        "intercept": intercept,
        "standardize": standardize,
        "algorithm": "PLS1 deflation",
        "scaling": "training sample standard deviation (n-intercept divisor); response unscaled",
    }


def fit_pls(spec, data):
    if spec.weights or spec.categorical or spec.options.get("forced_controls"):
        from .extended import fit_extended
        return fit_extended(spec, data)
    frame = ModelFrame(spec, data)
    n, p = frame.n, len(spec.predictors)
    selection, requested = frame.option("selection"), frame.option("components")
    candidates = frame.option("component_path")
    if selection == "fixed":
        if requested is None or candidates is not None:
            raise AnalysisError(
                "invalid_option", "Fixed PLS needs components and no component_path."
            )
        candidates = [requested]
    elif requested is not None:
        raise AnalysisError("invalid_option", "CV PLS uses component_path, not fixed components.")
    elif candidates is None:
        candidates = list(range(1, min(p, frame.option("max_components"), n - 1) + 1))
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > 100
        or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > min(p, n - 1)
            for v in candidates
        )
        or len(set(candidates)) != len(candidates)
    ):
        raise AnalysisError(
            "invalid_components", "component_path needs 1–100 unique feasible positive integers."
        )
    k = frame.option("folds") if selection == "cv" else 1
    work_guard(
        n * p * sum(candidates) * (k + 1) * 8, frame.option("max_work"), "PLS components and tuning"
    )
    frame.workspace_plan(
        "PLS1 and component CV",
        {
            "design_deflation_and_cv": tensor_bytes((n, p), itemsize=64),
            "component_state": tensor_bytes((p, max(candidates)), itemsize=40),
        },
    )
    x, y = frame.matrix(spec.predictors), frame.numeric(spec.outcome)
    options = {"intercept": spec.intercept, "standardize": frame.option("standardize")}
    assignment = folds(n, k, frame.option("seed")) if selection == "cv" else None
    path = []
    for count in sorted(candidates):
        errors, failures = [], []
        if assignment is not None:
            for fold in range(k):
                train, test = assignment != fold, assignment == fold
                try:
                    state = _fit(x[train], y[train], count, **options)
                    prediction = (
                        x[test] @ torch.tensor(state["coefficients"], dtype=torch.float64)
                        + state["constant"]
                    )
                    fold_errors = (y[test] - prediction).square()
                    if not bool(torch.isfinite(fold_errors).all()):
                        raise AnalysisError(
                            "numerical_failure",
                            "PLS held-out errors exceed finite float64 arithmetic.",
                        )
                    errors.extend(fold_errors.tolist())
                except AnalysisError as exc:
                    failures.append({"fold": fold, "code": exc.code, "message": str(exc)})
        mse = sum(errors) / len(errors) if errors and not failures else None
        path.append({"components": count, "cv_mse": mse, "failures": failures})
    if selection == "cv":
        valid = [row for row in path if row["cv_mse"] is not None]
        if not valid:
            raise AnalysisError(
                "component_selection_failed", "No component candidate fits every training fold."
            )
        requested = min(valid, key=lambda row: (row["cv_mse"], row["components"]))["components"]
    state = _fit(x, y, requested, **options)
    state.update(
        selection=selection,
        component_path=path,
        fold_assignments=None if assignment is None else assignment.tolist(),
        seed=frame.option("seed"),
        preprocessing_scope="each training fold; final state refit on retained training sample",
    )
    fitted = x @ torch.tensor(state["coefficients"], dtype=torch.float64) + state["constant"]
    return _prediction_result(
        frame,
        fitted,
        {
            "terms": list(spec.predictors),
            "predictive_coefficients": state["coefficients"],
            "constant": state["constant"],
            "penalized_state": state,
            "pls_state": state,
        },
        "native float64 PLS1",
    )
