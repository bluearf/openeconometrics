"""Weighted/category Gaussian paths and scalar partial PLS; no selected inference.

The existing numeric IID kernels remain unchanged. Positive normalized weights
define an empirical prediction loss, never a survey/design covariance target.
"""

from __future__ import annotations

import hashlib
import json
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes, workspace_budget_bytes

from .glm_design import _factors as design_factors
from .glm_design import _normalize_weights, _validate_state, fit_design, transform_design
from .kernels import (
    folds,
    forced_rank,
    path_values,
    penalty_factors,
    solve,
    solver_work,
    work_guard,
)

VERSION = "openecon.regularized-extended.v1"
MAX_STATE_BYTES = 32 * 1024**2
STATE_KEYS = {
    "version",
    "family",
    "spec",
    "design",
    "paths",
    "grid",
    "fractions",
    "selected_index",
    "selection",
    "l1_ratio",
    "cv_scores",
    "fold_assignments",
    "cv",
    "raw_weights",
    "normalized_weights",
    "physical_positions",
    "physical_nobs",
    "weight_type",
    "frequency_total",
    "planned_work",
    "seed",
    "weight_normalization",
    "cv_rule",
    "preprocessing_scope",
    "digest",
}
DESIGN_KEYS = {
    "version",
    "predictors",
    "categorical",
    "levels",
    "categorical_encoding",
    "terms",
    "term_predictors",
    "term_levels",
    "centers",
    "scales",
    "intercept",
    "standardize",
    "penalty_factors",
    "forced_controls",
    "effective_factors",
    "level_order",
    "weight_moments",
}
GAUSSIAN_POINT_KEYS = {
    "penalty",
    "standardized_coefficients",
    "coefficients",
    "constant",
    "outcome_center",
    "kkt_threshold",
    "iterations",
    "kkt_max",
    "objective",
}
PLS_POINT_KEYS = {
    "components",
    "standardized_coefficients",
    "coefficients",
    "constant",
    "outcome_center",
    "x_weights",
    "x_loadings",
    "y_loadings",
    "forced_indices",
    "control_y",
    "control_x",
    "algorithm",
}
CV_KEYS = {
    "fold",
    "training_positions",
    "validation_positions",
    "design",
    "paths",
    "grid",
    "validation_mse",
    "failures",
}


def _shape_preflight(state, *, digest=True):
    """Reject unknown/ragged metadata without serialization or tensor construction."""

    def exact(value, keys):
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("unknown or missing state fields")

    def vector(value, length):
        if not isinstance(value, list) or len(value) != length:
            raise ValueError("state vector shape")

    def matrix(value, rows, columns):
        vector(value, rows)
        for row in value:
            vector(row, columns)

    def design(value):
        exact(value, DESIGN_KEYS)
        p = len(value["terms"])
        if not isinstance(value["terms"], list) or not 1 <= p <= 64:
            raise ValueError("design width")
        if not isinstance(value["predictors"], list) or not 1 <= len(value["predictors"]) <= 16:
            raise ValueError("raw design width")
        if not isinstance(value["levels"], dict) or len(value["levels"]) > 16:
            raise ValueError("category maps")
        for entries in value["levels"].values():
            if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
                raise ValueError("category levels")
            for entry in entries:
                exact(entry, {"type", "value"})
        for key in ("term_predictors", "term_levels", "centers", "scales", "effective_factors"):
            vector(value[key], p)
        for entry in value["term_levels"]:
            if entry is not None:
                exact(entry, {"type", "value"})
        return p

    def point(value, d):
        if value is None:
            return
        exact(value, PLS_POINT_KEYS if state["family"] == "pls1" else GAUSSIAN_POINT_KEYS)
        p = len(d["terms"])
        for key in ("standardized_coefficients", "coefficients"):
            vector(value[key], p)
        if state["family"] == "pls1":
            k = value["components"]
            if type(k) is not int or not 1 <= k <= 64:
                raise ValueError("component count")
            controls = sum(factor == 0 for factor in d["effective_factors"])
            for key in ("x_weights", "x_loadings"):
                matrix(value[key], p - controls, k)
            vector(value["y_loadings"], k)
            vector(value["forced_indices"], controls)
            vector(value["control_y"], controls)
            matrix(value["control_x"], controls, p - controls)

    exact(state, STATE_KEYS if digest else STATE_KEYS - {"digest"})
    if state["family"] not in {"gaussian", "pls1"}:
        raise ValueError("family")
    n = state["physical_nobs"]
    if type(n) is not int or not 4 <= n <= 5000:
        raise ValueError("sample size")
    for key in ("raw_weights", "normalized_weights", "physical_positions"):
        vector(state[key], n)
    count = len(state["grid"])
    if not isinstance(state["grid"], list) or not 1 <= count <= 200:
        raise ValueError("candidate count")
    vector(state["paths"], count)
    design(state["design"])
    for value in state["paths"]:
        point(value, state["design"])
    for key in ("fractions", "cv_scores"):
        if state[key] is not None:
            vector(state[key], count)
    if state["fold_assignments"] is not None:
        vector(state["fold_assignments"], n)
    if not isinstance(state["cv"], list) or len(state["cv"]) > 20:
        raise ValueError("fold count")
    for record in state["cv"]:
        exact(record, CV_KEYS)
        for key in ("training_positions", "validation_positions"):
            if not isinstance(record[key], list) or len(record[key]) > n:
                raise ValueError("fold positions")
        design(record["design"])
        for key in ("paths", "grid", "validation_mse", "failures"):
            vector(record[key], count)
        for value in record["paths"]:
            point(value, record["design"])


def _json_preflight(value, *, max_work):
    """Count exact canonical UTF-8 bytes with bounded scalar work, no JSON buffer."""
    budget = workspace_budget_bytes()
    size = visits = 0
    active = set()

    def add(count):
        nonlocal size, visits
        size += count
        visits += count
        work_guard(visits, max_work, "canonical saved-state byte preflight")
        if size > MAX_STATE_BYTES:
            raise ValueError("saved state exceeds 32 MiB JSON limit")
        if size * 8 > budget:
            plan_workspace(
                "canonical saved-state byte preflight",
                {"json_string_utf8_and_digest_buffers": size * 8},
            )

    def string(text):
        if type(text) is not str:
            raise ValueError("JSON keys must be strings")
        add(2)
        encoded = 0
        for character in text:
            code = ord(character)
            if 0xD800 <= code <= 0xDFFF:
                raise ValueError("unpaired Unicode surrogate")
            amount = (
                2
                if character in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}
                else 6
                if code < 32
                else 1
                if code < 128
                else 2
                if code < 2048
                else 3
                if code < 65536
                else 4
            )
            encoded += amount
            if encoded > 65536:
                raise ValueError("metadata string exceeds 64 KiB JSON bound")
            add(amount)

    def visit(item, depth=0):
        if depth > 16:
            raise ValueError("metadata nesting exceeds 16 levels")
        if isinstance(item, (dict, list)):
            identity = id(item)
            if identity in active:
                raise ValueError("cyclic metadata")
            active.add(identity)
            add(2)
            if isinstance(item, dict):
                for index, (key, child) in enumerate(item.items()):
                    if index:
                        add(1)
                    string(key)
                    add(1)
                    visit(child, depth + 1)
            else:
                for index, child in enumerate(item):
                    if index:
                        add(1)
                    visit(child, depth + 1)
            active.remove(identity)
        elif type(item) is str:
            string(item)
        elif item is None or type(item) is bool:
            add(4 if item is None or item is True else 5)
        elif type(item) is int:
            if item.bit_length() > 16384:
                raise ValueError("metadata integer exceeds bounded JSON encoding")
            add(len(str(item)))
        elif type(item) is float and math.isfinite(item):
            add(len(json.dumps(item, allow_nan=False)))
        else:
            raise ValueError("unsupported or nonfinite JSON metadata")

    visit(value)
    return size


def _digest(state):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "digest"},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _factors(frame):
    value = frame.option("penalty_factors") if frame.spec.estimator != "pls" else None
    if isinstance(value, list):
        factors = penalty_factors(value, len(frame.spec.predictors)).tolist()
        return dict(zip(frame.spec.predictors, factors, strict=True))
    return value


def _design(frame, sample, raw):
    z, design, factors = fit_design(
        sample,
        frame.spec.predictors,
        frame.spec.categorical,
        raw,
        intercept=frame.spec.intercept,
        standardize=frame.option("standardize"),
        penalty_factors=_factors(frame),
        forced_controls=frame.option("forced_controls"),
    )
    replay = transform_design(sample, design)
    if not torch.allclose(z, replay, rtol=1e-10, atol=1e-12):
        raise AnalysisError(
            "numerical_failure",
            "Training preprocessing cannot be replayed at float64 precision; rescale predictors.",
        )
    return z, design, factors


def _center_y(y, normalized, intercept):
    # Rescale before averaging to avoid overflow in a finite weighted mean.
    magnitude = y.abs().max().clamp_min(1)
    center = (normalized @ (y / magnitude)) * magnitude if intercept else y.new_zeros(())
    if intercept and bool((y == y[0]).all()):
        center = y[0]
    residual = y - center
    if not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(center)):
        raise AnalysisError("numerical_failure", "Weighted response centering exceeds float64.")
    return residual, float(center)


def _whiten(z, y, raw, intercept):
    normalized = _normalize_weights(raw, len(y))
    centered, mean = _center_y(y, normalized, intercept)
    root = (len(y) * normalized).sqrt()
    wz, wy = z * root[:, None], centered * root
    if not bool(torch.isfinite(wz).all()) or not bool(torch.isfinite(wy).all()):
        raise AnalysisError("numerical_failure", "Weighted design exceeds float64.")
    return wz, wy, mean, normalized


def _grid(value):
    if (
        not isinstance(value, list)
        or not value
        or len(value) > 200
        or any(type(v) not in {int, float} or not math.isfinite(v) or v < 0 for v in value)
        or len(set(value)) != len(value)
    ):
        raise AnalysisError(
            "invalid_penalty", "lambda_path needs 1..200 distinct finite nonnegative values."
        )
    return sorted(map(float, value), reverse=True)


def _gaussian_path(z, y, raw, design, factors, grid, frame, ratio):
    wz, wy, ycenter, _ = _whiten(z, y, raw, frame.spec.intercept)
    forced_rank(wz, factors)
    beta, paths = None, []
    for penalty in grid:
        beta, diagnostics = solve(
            wz,
            wy,
            penalty,
            ratio,
            torch.ones(z.shape[1], dtype=torch.float64),
            frame.option("max_iterations"),
            frame.option("tolerance"),
            beta,
            factors=factors,
        )
        threshold = frame.option("tolerance") * max(1.0, float((wz.T @ wy / len(y)).abs().max()))
        if diagnostics["kkt_max"] > threshold:
            raise AnalysisError(
                "nonconvergence", "Weighted Gaussian path failed its KKT threshold."
            )
        coefficients = beta / torch.tensor(design["scales"], dtype=torch.float64)
        constant = ycenter - torch.tensor(design["centers"], dtype=torch.float64) @ coefficients
        if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(constant)):
            raise AnalysisError(
                "numerical_failure", "Original-unit weighted coefficients exceed float64."
            )
        paths.append(
            {
                "penalty": penalty,
                "standardized_coefficients": beta.tolist(),
                "coefficients": coefficients.tolist(),
                "constant": float(constant),
                "outcome_center": ycenter,
                "kkt_threshold": threshold,
                **diagnostics,
            }
        )
    return paths


def _reference(z, y, raw, factors, frame, ratio):
    wz, wy, _, _ = _whiten(z, y, raw, frame.spec.intercept)
    forced_rank(wz, factors)
    return path_values(wz, wy, ratio, {"n_lambdas": 2, "lambda_ratio": 0.5}, factors)[0]


def _pls_point(z, y, raw, design, components, intercept):
    """Weighted NIPALS in whitened coordinates, after identified control projection."""
    wz, wy, ycenter, _ = _whiten(z, y, raw, intercept)
    forced = torch.tensor(design["effective_factors"], dtype=torch.float64) == 0
    factors = torch.where(forced, 0.0, 1.0)
    forced_rank(wz, factors)
    controls, candidate = wz[:, forced], wz[:, ~forced]
    if components > min(candidate.shape[1], len(y) - int(intercept) - controls.shape[1]):
        raise AnalysisError("invalid_components", "Components exceed residual training dimensions.")
    control_y = (
        torch.linalg.lstsq(controls, wy, driver="gelsd").solution
        if controls.shape[1]
        else wy.new_empty(0)
    )
    control_x = (
        torch.linalg.lstsq(controls, candidate, driver="gelsd").solution
        if controls.shape[1]
        else candidate.new_empty((0, candidate.shape[1]))
    )
    residual = wy - controls @ control_y
    working = candidate - controls @ control_x
    weights, loadings, responses = [], [], []
    for _ in range(components):
        direction = working.T @ residual
        norm = torch.linalg.vector_norm(direction)
        floor = (
            100
            * torch.finfo(torch.float64).eps
            * torch.linalg.vector_norm(working)
            * torch.linalg.vector_norm(residual)
        )
        if not bool(torch.isfinite(norm)) or float(norm) <= float(floor) or float(norm) == 0:
            raise AnalysisError(
                "rank_deficient", "Requested weighted PLS component has no response direction."
            )
        weight = direction / norm
        score = working @ weight
        ss = score @ score
        if not bool(torch.isfinite(ss)) or float(ss) <= 100 * torch.finfo(
            torch.float64
        ).eps * float(working.square().sum()):
            raise AnalysisError("rank_deficient", "Requested weighted PLS score is unidentified.")
        loading, response = working.T @ score / ss, residual @ score / ss
        weights.append(weight)
        loadings.append(loading)
        responses.append(response)
        working = working - score[:, None] * loading
        residual = residual - score * response
    ws, ps, qs = torch.stack(weights, 1), torch.stack(loadings, 1), torch.stack(responses)
    rotation = ps.T @ ws
    singular = torch.linalg.svdvals(rotation)
    if float(singular[-1]) <= 100 * torch.finfo(torch.float64).eps * float(singular[0]):
        raise AnalysisError("rank_deficient", "PLS component rotation is numerically singular.")
    latent = ws @ torch.linalg.solve(rotation, qs)
    beta = torch.zeros(z.shape[1], dtype=torch.float64)
    beta[~forced] = latent
    beta[forced] = control_y - control_x @ latent
    coefficients = beta / torch.tensor(design["scales"], dtype=torch.float64)
    constant = ycenter - torch.tensor(design["centers"], dtype=torch.float64) @ coefficients
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(constant)):
        raise AnalysisError("numerical_failure", "Weighted PLS coefficients exceed float64.")
    return {
        "components": components,
        "coefficients": coefficients.tolist(),
        "constant": float(constant),
        "standardized_coefficients": beta.tolist(),
        "outcome_center": ycenter,
        "x_weights": ws.tolist(),
        "x_loadings": ps.tolist(),
        "y_loadings": qs.tolist(),
        "forced_indices": torch.where(forced)[0].tolist(),
        "control_y": control_y.tolist(),
        "control_x": control_x.tolist(),
        "algorithm": "weighted PLS1 with identified control partialling",
    }


def _predict_point(sample, design, point):
    z = transform_design(sample, design)
    # Saved coefficients are original units; use standardized coefficients to
    # replay training centers without reconstructing raw extreme coordinates.
    prediction = (
        z @ torch.tensor(point["standardized_coefficients"], dtype=torch.float64)
        + point["outcome_center"]
    )
    if not bool(torch.isfinite(prediction).all()):
        raise AnalysisError("numerical_failure", "Weighted prediction exceeds float64.")
    return prediction


def _weighted_loss(observed, predicted, weights):
    # Subtract first when finite to preserve nearly equal operands. Weight
    # before squaring; use scaled subtraction only for overflowing differences.
    scale = weights.sqrt()
    difference = observed - predicted
    residual = scale * difference
    overflow = ~torch.isfinite(difference)
    if bool(overflow.any()):
        residual = torch.where(overflow, scale * observed - scale * predicted, residual)
    loss = residual @ residual
    if not bool(torch.isfinite(loss)):
        raise AnalysisError("numerical_failure", "Weighted squared loss exceeds float64.")
    return float(loss)


@resident_cpu
def fit_extended(spec, data, *, ratio=None):
    frame = ModelFrame(spec, data)
    if frame.n < 4 or frame.n > 5000 or not 1 <= len(spec.predictors) <= 16:
        raise AnalysisError(
            "dimension_limit",
            "Extended regularized prediction needs 4..5000 retained rows and 1..16 predictors.",
        )
    selection = frame.option("selection")
    if selection not in {"fixed", "cv"}:
        raise AnalysisError(
            "unsupported_selection",
            "Weighted/categorical prediction supports fixed/CV; IID score plug-in assumptions are not extended.",
        )
    pls = spec.estimator == "pls"
    upper = len(spec.predictors) - len(spec.categorical) + 16 * len(spec.categorical)
    upper_controls = sum(
        16 if name in spec.categorical else 1 for name in (frame.option("forced_controls") or [])
    )
    design_work = frame.n * upper * 8 + frame.n * upper_controls**2 + upper_controls**3
    work_guard(design_work, frame.option("max_work"), "weighted design and control preflight")
    frame.workspace_plan(
        "weighted/category regularized design and full fold state",
        {
            "design_and_scaling_copies": tensor_bytes((frame.n, upper + 2), itemsize=128),
            "factors_and_partialling": tensor_bytes((upper, upper), itemsize=128),
            "complete_paths_and_folds": tensor_bytes((200, upper + 24), itemsize=128)
            * (frame.option("folds") + 1 if selection == "cv" else 1),
        },
    )
    if spec.weight_type == "fweight" and any(
        type(v) is bool or v > 2**53 for v in frame.sample[spec.weights].tolist()
    ):
        raise AnalysisError(
            "invalid_weights",
            "Original frequency counts must be exactly represented integers <=2^53.",
        )
    y = frame.numeric(spec.outcome)
    raw = frame.weights()
    raw = torch.ones(frame.n, dtype=torch.float64) if raw is None else raw
    normalized = _normalize_weights(raw, frame.n)
    z, design, factors = _design(frame, frame.sample, raw)
    if z.shape[1] == 0:
        raise AnalysisError(
            "insufficient_predictors", "At least one encoded predictor is required."
        )
    assignment = (
        folds(frame.n, frame.option("folds"), frame.option("seed")) if selection == "cv" else None
    )
    fractions = None
    if pls:
        requested, given = frame.option("components"), frame.option("component_path")
        if selection == "fixed":
            if requested is None or given is not None:
                raise AnalysisError(
                    "invalid_components", "Fixed PLS requires components without component_path."
                )
            grid = [requested]
        else:
            if requested is not None:
                raise AnalysisError(
                    "invalid_components", "CV PLS selects component_path, not fixed components."
                )
            free_width = int((factors != 0).sum())
            grid = (
                given
                if given is not None
                else list(
                    range(1, min(free_width, frame.option("max_components"), frame.n - 1) + 1)
                )
            )
        if (
            not isinstance(grid, list)
            or not grid
            or len(grid) > 100
            or any(type(v) is not int or v < 1 or v > min(z.shape[1], frame.n - 1) for v in grid)
            or len(set(grid)) != len(grid)
        ):
            raise AnalysisError(
                "invalid_components",
                "component_path requires 1..100 unique feasible positive integers.",
            )
        grid = sorted(grid)
        planned = (
            frame.n
            * z.shape[1]
            * (sum(grid) + z.shape[1] ** 2)
            * (frame.option("folds") + 1 if assignment is not None else 1)
            * 16
        )
    else:
        given, requested = frame.option("lambda_path"), frame.option("penalty")
        if selection == "fixed":
            if requested is None:
                raise AnalysisError("invalid_penalty", "Fixed selection requires penalty.")
            grid = [float(requested)] if given is None else _grid(given)
            if requested not in grid:
                raise AnalysisError("invalid_penalty", "Fixed penalty must occur in lambda_path.")
        else:
            if requested is not None:
                raise AnalysisError("invalid_penalty", "CV selects a penalty; omit penalty.")
            if given is None:
                grid = None
            else:
                grid = _grid(given)
        per_path = max(
            solver_work(frame.n, z.shape[1], frame.option("max_iterations"), ratio),
            frame.n * z.shape[1] ** 2 + z.shape[1] ** 3,
        )
        partitions = frame.option("folds") + 1 if assignment is not None else 1
        count = len(grid) if grid is not None else frame.option("n_lambdas")
        reference_work = (
            (frame.n * z.shape[1] ** 2 + z.shape[1] ** 3) * partitions if grid is None else 0
        )
        planned = per_path * count * partitions + reference_work
    planned += design_work
    work_guard(
        planned, frame.option("max_work"), "complete weighted/category paths, controls and folds"
    )
    if grid is None:
        fractions = torch.logspace(
            0,
            math.log10(frame.option("lambda_ratio")),
            frame.option("n_lambdas"),
            dtype=torch.float64,
        ).tolist()
        reference = _reference(z, y, raw, factors, frame, ratio)
        grid = [reference * value for value in fractions]
    scores = [0.0] * len(grid)
    cv = []
    for fold in range(frame.option("folds") if assignment is not None else 0):
        train, test = assignment != fold, assignment == fold
        ti, vi = torch.where(train)[0].tolist(), torch.where(test)[0].tolist()
        tz, td, tf = _design(frame, frame.sample.iloc[ti], raw[train])
        # Encoding validation must happen even when every PLS component fails;
        # unseen validation categories never silently remove scored rows.
        transform_design(frame.sample.iloc[vi], td)
        if fractions is None:
            train_grid = grid
        else:
            train_reference = _reference(tz, y[train], raw[train], tf, frame, ratio)
            train_grid = [train_reference * value for value in fractions]
        points, losses, failures = [], [], []
        for index, value in enumerate(train_grid):
            try:
                point = (
                    _pls_point(tz, y[train], raw[train], td, value, spec.intercept)
                    if pls
                    else _gaussian_path(tz, y[train], raw[train], td, tf, [value], frame, ratio)[0]
                )
                prediction = _predict_point(frame.sample.iloc[vi], td, point)
                pooled = _weighted_loss(y[test], prediction, normalized[test])
                # A prior failed PLS fold permanently makes this candidate
                # ineligible; preserve later fold records without treating that
                # deliberate sentinel as new arithmetic overflow.
                if math.isfinite(scores[index]):
                    scores[index] += pooled
                    if not math.isfinite(scores[index]):
                        raise AnalysisError("numerical_failure", "Pooled weighted loss exceeds float64.")
                points.append(point)
                losses.append(_weighted_loss(y[test], prediction, _normalize_weights(raw[test], len(vi))))
                failures.append(None)
            except AnalysisError as exc:
                if not pls or exc.code not in {"rank_deficient", "invalid_components"}:
                    raise
                scores[index] = math.inf
                points.append(None)
                losses.append(None)
                failures.append(exc.code)
        cv.append(
            {
                "fold": fold,
                "training_positions": [frame.positions[i] for i in ti],
                "validation_positions": [frame.positions[i] for i in vi],
                "design": td,
                "paths": points,
                "grid": train_grid,
                "validation_mse": losses,
                "failures": failures,
            }
        )
    if assignment is not None:
        valid = [i for i, value in enumerate(scores) if math.isfinite(value)]
        if not valid:
            raise AnalysisError(
                "component_selection_failed", "No component candidate fits every training fold."
            )
        selected = min(valid, key=lambda i: (scores[i], grid[i] if pls else -grid[i]))
    else:
        selected = grid.index(requested)
    if pls:
        paths = []
        for index, value in enumerate(grid):
            try:
                paths.append(_pls_point(z, y, raw, design, value, spec.intercept))
            except AnalysisError as exc:
                if (
                    assignment is None
                    or index == selected
                    or exc.code not in {"rank_deficient", "invalid_components"}
                ):
                    raise
                paths.append(None)
    else:
        paths = _gaussian_path(z, y, raw, design, factors, grid, frame, ratio)
    point = paths[selected]
    fitted = _predict_point(frame.sample, design, point)
    state = {
        "version": VERSION,
        "family": "pls1" if pls else "gaussian",
        "spec": spec.model_dump(mode="json"),
        "design": design,
        "paths": paths,
        "grid": grid,
        "fractions": fractions,
        "selected_index": selected,
        "selection": selection,
        "l1_ratio": ratio,
        "cv_scores": [value if math.isfinite(value) else None for value in scores]
        if assignment is not None
        else None,
        "fold_assignments": assignment.tolist() if assignment is not None else None,
        "cv": cv,
        "raw_weights": raw.tolist(),
        "normalized_weights": normalized.tolist(),
        "physical_positions": frame.positions,
        "physical_nobs": frame.n,
        "weight_type": spec.weight_type,
        "frequency_total": sum(int(v) for v in raw.tolist())
        if spec.weight_type == "fweight"
        else None,
        "planned_work": planned,
        "seed": frame.option("seed"),
        "weight_normalization": "max-rescale then sum to one independently in each training partition",
        "cv_rule": "pooled normalized empirical loss; ties choose smallest components / largest lambda",
        "preprocessing_scope": "each training fold; selected refit on complete retained sample",
    }
    _shape_preflight(state, digest=False)
    encoded_bytes = _json_preflight(state, max_work=frame.option("max_work"))
    # The decimal work count and digest are appended after measurement. Reserve
    # their bounded growth and count traversal/serialization/UTF-8/hash passes.
    serialization_bound = encoded_bytes + 256
    work_guard(
        planned + 4 * serialization_bound,
        frame.option("max_work"),
        "complete fit and saved-state byte work",
    )
    plan_workspace(
        "canonical fitted state and digest",
        {"json_string_utf8_and_digest_buffers": serialization_bound * 8},
    )
    state["planned_work"] = planned + 4 * serialization_bound
    state["digest"] = _digest(state)
    note = "Prediction only; penalized/component-selected coefficients have no classical SE/df/p/CI. Analytic/sampling weights target empirical prediction loss, not survey inference."
    alias = {**point, "selected_penalty": None if pls else grid[selected]}
    return build_result(
        frame,
        terms=[],
        params=torch.empty(0, dtype=torch.float64),
        covariance=torch.empty((0, 0), dtype=torch.float64),
        use_t=False,
        fitted=fitted,
        metrics={"training_mse": _weighted_loss(y, fitted, normalized)},
        solver="weighted PLS1 deflation" if pls else "weighted native Gaussian path with KKT",
        inference={
            "target": "prediction",
            "available": False,
            "distribution": "none",
            "use_t": None,
            "correction": "no selected coefficient inference",
        },
        extra={
            "target": "prediction",
            "notes": [note],
            "terms": design["terms"],
            "constant": point["constant"],
            "predictive_coefficients": point["coefficients"],
            "regularized_extended_state": state,
            "penalized_state": alias,
        },
        warnings=[note],
    )


def _validate_point(saved, design, family, value, spec):
    _, _, _, _, centers, scales = _validate_state(design)
    beta = torch.tensor(saved["standardized_coefficients"], dtype=torch.float64)
    coefficient = torch.tensor(saved["coefficients"], dtype=torch.float64)
    if (
        beta.shape != scales.shape
        or coefficient.shape != scales.shape
        or not bool(torch.isfinite(beta).all())
        or not bool(torch.isfinite(coefficient).all())
    ):
        raise ValueError("coefficient shape")
    if not spec.intercept and (saved["outcome_center"] != 0 or saved["constant"] != 0):
        raise ValueError("no-intercept outcome centering")
    if not torch.allclose(beta / scales, coefficient, rtol=1e-13, atol=1e-13) or not math.isclose(
        saved["constant"],
        saved["outcome_center"] - float(centers @ coefficient),
        rel_tol=1e-13,
        abs_tol=1e-13,
    ):
        raise ValueError("coefficient scaling")
    if family == "gaussian":
        if (
            saved["penalty"] != value
            or not 0 <= saved["kkt_max"] <= saved["kkt_threshold"]
            or saved["objective"] < 0
            or type(saved["iterations"]) is not int
            or not 1 <= saved["iterations"] <= spec.options.get("max_iterations", 2000)
        ):
            raise ValueError("Gaussian diagnostics")
        return
    if saved["components"] != value:
        raise ValueError("PLS component binding")
    forced = torch.tensor(design["effective_factors"], dtype=torch.float64) == 0
    if saved["forced_indices"] != torch.where(forced)[0].tolist():
        raise ValueError("PLS forced block binding")
    width, controls = int((~forced).sum()), int(forced.sum())
    ws = torch.tensor(saved["x_weights"], dtype=torch.float64)
    ps = torch.tensor(saved["x_loadings"], dtype=torch.float64)
    qs = torch.tensor(saved["y_loadings"], dtype=torch.float64)
    cy = torch.tensor(saved["control_y"], dtype=torch.float64)
    cx = torch.tensor(saved["control_x"], dtype=torch.float64).reshape(controls, width)
    if (
        ws.shape != (width, value)
        or ps.shape != ws.shape
        or qs.shape != (value,)
        or cy.shape != (controls,)
        or any(not bool(torch.isfinite(v).all()) for v in (ws, ps, qs, cy, cx))
    ):
        raise ValueError("PLS loadings shape")
    if not torch.allclose(
        ws.square().sum(0), torch.ones(value, dtype=torch.float64), rtol=1e-12, atol=1e-12
    ):
        raise ValueError("PLS direction normalization")
    latent = ws @ torch.linalg.solve(ps.T @ ws, qs)
    reconstructed = torch.zeros_like(beta)
    reconstructed[~forced] = latent
    reconstructed[forced] = cy - cx @ latent
    if not torch.allclose(reconstructed, beta, rtol=1e-12, atol=1e-12):
        raise ValueError("PLS loadings coefficient binding")


@resident_cpu
def _validate_result(result, *, query_rows=0, max_work=200000000):
    state = result.extra.get("regularized_extended_state")
    try:
        _shape_preflight(state)
        encoded_bytes = _json_preflight(state, max_work=max_work)
        if (
            not isinstance(state, dict)
            or state.get("version") != VERSION
            or not isinstance(state.get("paths"), list)
            or not 1 <= len(state["paths"]) <= 200
            or not isinstance(state.get("raw_weights"), list)
            or not 4 <= len(state["raw_weights"]) <= 5000
            or not isinstance(state.get("cv"), list)
            or len(state["cv"]) > 20
        ):
            raise ValueError("state integrity")
        width = len(state["design"]["terms"])
        if not 1 <= width <= 64:
            raise ValueError("state design width")
        rows, count, partitions = (
            len(state["raw_weights"]),
            len(state["paths"]),
            len(state["cv"]) + 1,
        )
        metadata_cells = count * (width + 24) * partitions
        if state["family"] == "pls1":
            if any(
                point is not None
                and (type(point.get("components")) is not int or not 1 <= point["components"] <= 64)
                for point in state["paths"]
            ):
                raise ValueError("PLS state component bound")
            metadata_cells += (
                width
                * sum(point["components"] for point in state["paths"] if point is not None)
                * partitions
                * 3
            )
        work_guard(
            4 * encoded_bytes
            + rows * (width + partitions + 2)
            + metadata_cells * 32
            + query_rows * (width + 1),
            max_work,
            "saved complete regularized state and query",
        )
        plan_workspace(
            "saved complete weighted/category regularized state",
            {
                "bounded_state_validation_and_digest": tensor_bytes(
                    (metadata_cells + rows * 5,), itemsize=128
                )
                + encoded_bytes * 8,
                "query_design_and_output": tensor_bytes((query_rows, width + 2), itemsize=64),
            },
        )
        if state.get("digest") != _digest(state):
            raise ValueError("state integrity")
        if (
            state["spec"] != result.spec.model_dump(mode="json")
            or state["physical_positions"] != result.sample_positions
        ):
            raise ValueError("sample/spec binding")
        predictors, _, _, intercept, centers, scales = _validate_state(state["design"])
        if (
            predictors != result.spec.predictors
            or intercept != result.spec.intercept
            or state["design"]["categorical"] != result.spec.categorical
        ):
            raise ValueError("design binding")
        requested_factors = result.spec.options.get("penalty_factors")
        if isinstance(requested_factors, list):
            requested_factors = dict(
                zip(
                    predictors,
                    penalty_factors(requested_factors, len(predictors)).tolist(),
                    strict=True,
                )
            )
        expected_factors, expected_forced = design_factors(
            predictors, requested_factors, result.spec.options.get("forced_controls")
        )
        if (
            state["design"]["penalty_factors"] != expected_factors
            or state["design"]["forced_controls"] != expected_forced
            or state["design"]["standardize"] != result.spec.options.get("standardize", True)
        ):
            raise ValueError("preprocessing option binding")
        n = state["physical_nobs"]
        if type(n) is not int or n != len(state["physical_positions"]) or not 4 <= n <= 5000:
            raise ValueError("sample dimensions")
        if any(
            type(value) not in {int, float} or not math.isfinite(value) or value <= 0
            for value in state["raw_weights"]
        ):
            raise ValueError("original weight scalar domain")
        if result.spec.weight_type == "fweight" and any(
            value > 2**53 or value != int(value) for value in state["raw_weights"]
        ):
            raise ValueError("original frequency domain")
        raw = torch.tensor(state["raw_weights"], dtype=torch.float64)
        normalized = _normalize_weights(raw, n)
        if (
            state["normalized_weights"] != normalized.tolist()
            or state["weight_type"] != result.spec.weight_type
        ):
            raise ValueError("weight binding")
        if result.spec.weight_type == "fweight":
            if (
                bool((raw != raw.round()).any())
                or float(raw.max()) > 2**53
                or state["frequency_total"] != sum(int(v) for v in raw.tolist())
            ):
                raise ValueError("frequency binding")
        if state["family"] != ("pls1" if result.spec.estimator == "pls" else "gaussian"):
            raise ValueError("family binding")
        expected_ratio = (
            None
            if state["family"] == "pls1"
            else {
                "ridge": 0.0,
                "lasso": 1.0,
                "elasticnet": result.spec.options.get("l1_ratio", 0.5),
            }[result.spec.estimator]
        )
        if state["l1_ratio"] != expected_ratio or state["seed"] != result.spec.options.get(
            "seed", 1729
        ):
            raise ValueError("solver option binding")
        grid, paths, chosen = state["grid"], state["paths"], state["selected_index"]
        if (
            not isinstance(grid, list)
            or not grid
            or len(paths) != len(grid)
            or type(chosen) is not int
            or not 0 <= chosen < len(grid)
        ):
            raise ValueError("path shape")
        if state["selection"] != result.spec.options.get("selection", "cv"):
            raise ValueError("selection binding")
        if state["family"] == "gaussian":
            if _grid(grid) != grid:
                raise ValueError("Gaussian grid")
            given = result.spec.options.get("lambda_path")
            if given is not None and grid != _grid(given):
                raise ValueError("supplied lambda binding")
            if (
                state["selection"] == "fixed"
                and given is None
                and grid != [float(result.spec.options["penalty"])]
            ):
                raise ValueError("fixed lambda grid binding")
            if (state["fractions"] is not None) != (state["selection"] == "cv" and given is None):
                raise ValueError("automatic grid selection binding")
            if state["fractions"] is not None:
                expected_fractions = torch.logspace(
                    0,
                    math.log10(result.spec.options.get("lambda_ratio", 0.001)),
                    result.spec.options.get("n_lambdas", 30),
                    dtype=torch.float64,
                ).tolist()
                if (
                    given is not None
                    or state["selection"] != "cv"
                    or state["fractions"] != expected_fractions
                    or len(grid) != len(expected_fractions)
                ):
                    raise ValueError("automatic fraction binding")
                if any(
                    not math.isclose(value, grid[0] * fraction, rel_tol=1e-13)
                    for value, fraction in zip(grid, expected_fractions, strict=True)
                ):
                    raise ValueError("automatic grid scaling")
        else:
            if (
                any(type(value) is not int or value < 1 or value > 64 for value in grid)
                or grid != sorted(set(grid))
                or state["fractions"] is not None
            ):
                raise ValueError("PLS grid")
            if state["selection"] == "fixed":
                expected_grid = [result.spec.options["components"]]
            elif result.spec.options.get("component_path") is not None:
                expected_grid = sorted(result.spec.options["component_path"])
            else:
                free_width = sum(factor != 0 for factor in state["design"]["effective_factors"])
                expected_grid = list(
                    range(
                        1, min(free_width, result.spec.options.get("max_components", 10), n - 1) + 1
                    )
                )
            if grid != expected_grid:
                raise ValueError("PLS requested component grid binding")
        point = paths[chosen]
        for index, saved in enumerate(paths):
            if saved is None:
                if state["family"] != "pls1" or state["selection"] != "cv":
                    raise ValueError("invalid failed path")
                continue
            _validate_point(saved, state["design"], state["family"], grid[index], result.spec)
        if (
            point is None
            or result.extra["terms"] != state["design"]["terms"]
            or result.extra["constant"] != point["constant"]
            or result.extra["predictive_coefficients"] != point["coefficients"]
            or result.extra["penalized_state"]
            != {**point, "selected_penalty": None if state["family"] == "pls1" else grid[chosen]}
        ):
            raise ValueError("visible state binding")
        if state["selection"] == "fixed":
            option = "components" if state["family"] == "pls1" else "penalty"
            if (
                grid[chosen] != result.spec.options[option]
                or state["cv"]
                or state["fold_assignments"] is not None
            ):
                raise ValueError("fixed tuning binding")
        else:
            assignment = state["fold_assignments"]
            k = result.spec.options.get("folds", 5)
            if assignment != folds(n, k, state["seed"]).tolist() or len(state["cv"]) != k:
                raise ValueError("fold binding")
            reconstructed = [0.0] * len(grid)
            for fold, record in enumerate(state["cv"]):
                _validate_state(record["design"])
                if (
                    record["design"]["predictors"] != predictors
                    or record["design"]["categorical"] != result.spec.categorical
                    or record["design"]["penalty_factors"] != expected_factors
                    or record["design"]["forced_controls"] != expected_forced
                    or record["design"]["standardize"] != state["design"]["standardize"]
                    or record["design"]["intercept"] != intercept
                ):
                    raise ValueError("fold preprocessing binding")
                if (
                    record["fold"] != fold
                    or record["training_positions"]
                    != [
                        pos
                        for pos, group in zip(state["physical_positions"], assignment, strict=True)
                        if group != fold
                    ]
                    or record["validation_positions"]
                    != [
                        pos
                        for pos, group in zip(state["physical_positions"], assignment, strict=True)
                        if group == fold
                    ]
                ):
                    raise ValueError("fold physical binding")
                if any(
                    len(record[key]) != len(grid)
                    for key in ("paths", "grid", "validation_mse", "failures")
                ):
                    raise ValueError("fold path shape")
                if state["fractions"] is None and record["grid"] != grid:
                    raise ValueError("fold supplied grid binding")
                if state["fractions"] is not None:
                    if (
                        _grid(record["grid"]) != record["grid"]
                        or record["grid"][0] <= 0
                        or any(
                            not math.isclose(value, record["grid"][0] * fraction, rel_tol=1e-13)
                            for value, fraction in zip(
                                record["grid"], state["fractions"], strict=True
                            )
                        )
                    ):
                        raise ValueError("automatic fold grid fraction binding")
                mass = float(normalized[torch.tensor(assignment) == fold].sum())
                for index, loss in enumerate(record["validation_mse"]):
                    if loss is None:
                        if (
                            state["family"] != "pls1"
                            or record["paths"][index] is not None
                            or record["failures"][index]
                            not in {"rank_deficient", "invalid_components"}
                        ):
                            raise ValueError("invalid failed candidate")
                        reconstructed[index] = math.inf
                    elif (
                        type(loss) not in {int, float}
                        or not math.isfinite(loss)
                        or loss < 0
                        or record["failures"][index] is not None
                        or record["paths"][index] is None
                    ):
                        raise ValueError("invalid fold loss")
                    else:
                        _validate_point(
                            record["paths"][index],
                            record["design"],
                            state["family"],
                            record["grid"][index],
                            result.spec,
                        )
                        reconstructed[index] += mass * loss
            if (
                not isinstance(state["cv_scores"], list)
                or len(state["cv_scores"]) != len(grid)
                or any(
                    (
                        saved is not None
                        if not math.isfinite(score)
                        else type(saved) not in {int, float}
                        or not math.isclose(saved, score, rel_tol=1e-12, abs_tol=1e-12)
                    )
                    for score, saved in zip(reconstructed, state["cv_scores"], strict=True)
                )
            ):
                raise ValueError("pooled CV loss binding")
            valid = [index for index, score in enumerate(reconstructed) if math.isfinite(score)]
            selected = min(
                valid,
                key=lambda index: (
                    state["cv_scores"][index],
                    grid[index] if state["family"] == "pls1" else -grid[index],
                ),
            )
            if selected != chosen:
                raise ValueError("selected CV candidate binding")
        return state, point
    except AnalysisError as exc:
        if exc.code in {
            "work_limit",
            "invalid_work_limit",
            "workspace_limit",
            "invalid_resource_budget",
        }:
            raise
        raise AnalysisError(
            "invalid_prediction_state", "Saved weighted/category prediction state is inconsistent."
        ) from exc
    except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as exc:
        raise AnalysisError(
            "invalid_prediction_state", "Saved weighted/category prediction state is inconsistent."
        ) from exc


@resident_cpu
def predict_extended(result: ResultBundle, data, *, max_work=200000000):
    from openecon.analysis import _coerce_frame
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported", "Weighted/category saved prediction needs a resident table."
        )
    data = _coerce_frame(data)
    if not 1 <= len(data) <= 100000:
        raise AnalysisError("dimension_limit", "Saved prediction admits 1..100000 query rows.")
    # Validate the bounded metadata before allocating the query design.
    state, point = _validate_result(result, query_rows=len(data), max_work=max_work)
    prediction = _predict_point(data, state["design"], point)
    return pd.Series(prediction.tolist(), index=data.index, name="predicted")
