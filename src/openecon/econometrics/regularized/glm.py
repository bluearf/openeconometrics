"""Saved penalized Bernoulli/Poisson prediction with honest training-fold tuning."""

from __future__ import annotations

import hashlib
import json
import math

import openecon
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec, table
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes

from .glm_design import _factors, _validate_state, fit_design, transform_design
from .glm_kernels import solve_path, solver_work
from .kernels import folds, work_guard


def _convenience(name, data, y, x, categorical, weights, weight_type, intercept, missing, options):
    from openecon.analysis import fit
    return fit(make_spec(name, outcome=y, predictors=column_list(x, "x"),
                         categorical=column_list(categorical, "categorical"),
                         weights=weights, weight_type=weight_type, intercept=intercept,
                         missing=missing, options=options), data=data)


def elasticnet_logit(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
                    intercept=True, missing="raise", l1_ratio=0.5, selection="cv", penalty=None,
                    lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729,
                    standardize=True, penalty_factors=None, forced_controls=None,
                    max_iterations=200, tolerance=1e-8, max_work=2000000000, device="cpu") -> ResultBundle:
    """Bernoulli elastic-net prediction: weighted mean loss, train-only CV; no coefficient CI."""
    options = {key: value for key, value in locals().items() if key not in
               {"data", "y", "x", "categorical", "weights", "weight_type", "intercept", "missing"}}
    return _convenience("elasticnet_logit", data, y, x, categorical, weights,
                        weight_type, intercept, missing, options)


def elasticnet_poisson(*, data, y, x, categorical=None, weights=None, weight_type="aweight",
                      intercept=True, missing="raise", l1_ratio=0.5, selection="cv", penalty=None,
                      lambda_path=None, n_lambdas=20, lambda_ratio=0.001, folds=5, seed=1729,
                      standardize=True, penalty_factors=None, forced_controls=None,
                      max_iterations=200, tolerance=1e-8, max_work=2000000000, device="cpu") -> ResultBundle:
    """Poisson log-link elastic-net prediction with saved loss/KKT/CV; no coefficient CI."""
    options = {key: value for key, value in locals().items() if key not in
               {"data", "y", "x", "categorical", "weights", "weight_type", "intercept", "missing"}}
    return _convenience("elasticnet_poisson", data, y, x, categorical, weights,
                        weight_type, intercept, missing, options)


def _weights(raw):
    scaled = raw / raw.max()
    normalized = scaled / scaled.sum()
    if not bool(torch.isfinite(normalized).all()) or bool((normalized <= 0).any()):
        raise AnalysisError("invalid_weights", "Weight dynamic range exceeds positive float64 normalization.")
    return normalized


def _response(y, family):
    if family == "binomial":
        if bool(((y != 0) & (y != 1)).any()) or len(torch.unique(y)) != 2:
            raise AnalysisError("invalid_outcome", "Bernoulli fitting needs both observed 0 and 1 classes.")
    elif bool((y < 0).any()) or bool((y != y.round()).any()) or float(y.max()) > 1e6:
        raise AnalysisError("invalid_outcome", "Poisson fitting needs integer counts between 0 and 1,000,000.")
    elif not bool((y > 0).any()):
        raise AnalysisError("invalid_outcome", "Poisson fitting needs a positive observed count.")


def _grid(value):
    if not isinstance(value, list) or not value or len(value) > 100 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in value
    ) or len(set(value)) != len(value):
        raise AnalysisError("invalid_penalty", "lambda_path needs 1..100 distinct finite nonnegative numbers.")
    return sorted(map(float, value), reverse=True)


def _reference(z, y, weights, factors, family, ratio, intercept):
    w = _weights(weights)
    mean = float(w @ y) if intercept else (0.5 if family == "binomial" else 1.0)
    score = (z.T @ (w * (y - mean))).abs()
    penalized = factors > 0
    reference = float((score[penalized] / factors[penalized]).max()) if bool(penalized.any()) else 0.0
    return max(reference / max(ratio, 0.01), 1e-8)


def _loss(eta, y, family):
    if not bool(torch.isfinite(eta).all()) or (family == "poisson" and bool((eta.abs() > 700).any())):
        raise AnalysisError("numerical_failure", "Likelihood evaluation requires finite links; Poisson additionally requires |link|<=700.")
    if family == "binomial":
        loss = torch.nn.functional.softplus(torch.where(y == 1, -eta, eta))
    else:
        loss = eta.exp() - y * eta
    if not bool(torch.isfinite(loss).all()):
        raise AnalysisError("numerical_failure", "Held-out likelihood exceeds finite float64 arithmetic.")
    return loss


def _design(frame, sample, weights):
    return fit_design(sample, frame.spec.predictors, frame.spec.categorical, weights,
                      intercept=frame.spec.intercept, standardize=frame.option("standardize"),
                      penalty_factors=frame.option("penalty_factors"),
                      forced_controls=frame.option("forced_controls"))


def _digest(state):
    return hashlib.sha256(json.dumps({k: v for k, v in state.items() if k != "digest"},
                                   sort_keys=True, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":")).encode()).hexdigest()


def _fit(spec, data, family):
    if spec.options.get("device", "cpu") != "cpu":
        raise AnalysisError("unsupported_device", "Penalized GLM supports explicit CPU float64 only.")
    frame = ModelFrame(spec, data)
    if frame.n > 5000 or len(spec.predictors) > 16:
        raise AnalysisError("dimension_limit", "Penalized GLM needs at most 5,000 retained rows and 16 raw predictors.")
    if frame.n < 4:
        raise AnalysisError("insufficient_sample", "Penalized GLM needs at least four retained observations.")
    given = frame.option("lambda_path")
    selection, penalty = frame.option("selection"), frame.option("penalty")
    if selection == "fixed":
        if penalty is None:
            raise AnalysisError("invalid_penalty", "Fixed selection requires an explicit penalty.")
        grid = [float(penalty)] if given is None else _grid(given)
        if penalty not in grid:
            raise AnalysisError("invalid_penalty", "Fixed penalty must occur in lambda_path.")
    else:
        if penalty is not None:
            raise AnalysisError("invalid_penalty", "CV selects from lambda_path/fractions; fixed penalty is not accepted.")
        grid = None if given is None else _grid(given)
    path_count = len(grid) if grid is not None else frame.option("n_lambdas")
    # Conservative raw-category maximum before any expanded tensor allocation.
    upper_p = len(spec.predictors) - len(spec.categorical) + 16 * len(spec.categorical)
    if upper_p > 256:
        raise AnalysisError("dimension_limit", "Raw categorical design exceeds the admitted metadata domain.")
    frame.workspace_plan("penalized GLM preliminary design and saved paths", {
        "design_training_scaling_cv_and_saved_preprocessing": tensor_bytes((frame.n, upper_p + 2), itemsize=112),
        "newton_gram_and_coordinate_factors": tensor_bytes((upper_p + 1, upper_p + 1), itemsize=96),
        "complete_paths_and_fold_state": tensor_bytes((path_count, upper_p + 16), itemsize=128)
                                             * (frame.option("folds") + 1 if selection == "cv" else 1),
    })
    y = frame.numeric(spec.outcome)
    _response(y, family)
    if spec.weight_type == "fweight" and any(value > 2**53 for value in frame.sample[spec.weights].tolist()):
        raise AnalysisError("invalid_weights", "Original frequency counts must be integers <=2^53 before float64 conversion.")
    raw = frame.weights()
    weights = torch.ones(frame.n, dtype=torch.float64) if raw is None else raw
    normalized = _weights(weights)
    if spec.weight_type == "fweight" and float(weights.max()) > 2**53:
        raise AnalysisError("invalid_weights", "Frequency counts must be exactly represented integers <=2^53.")
    z, design, factors = _design(frame, frame.sample, weights)
    p = z.shape[1]
    k = frame.option("folds") if selection == "cv" else 0
    assignment = folds(frame.n, k, frame.option("seed")) if k else None
    planned = solver_work(frame.n, p, frame.option("max_iterations"), path_count)
    if assignment is not None:
        planned += sum(solver_work(int((assignment != fold).sum()), p,
                                   frame.option("max_iterations"), path_count) for fold in range(k))
    work_guard(planned, frame.option("max_work"), "complete penalized GLM path and fold fitting")
    solver_options = dict(family=family, l1_ratio=frame.option("l1_ratio"),
                          intercept=spec.intercept, max_iterations=frame.option("max_iterations"),
                          tolerance=frame.option("tolerance"))
    fractions = None
    if grid is None:
        fractions = torch.logspace(0, math.log10(frame.option("lambda_ratio")),
                                   path_count, dtype=torch.float64).tolist()
        reference = _reference(z, y, weights, factors, family, frame.option("l1_ratio"), spec.intercept)
        grid = [reference * fraction for fraction in fractions]
    cv, scores = [], [0.0] * path_count
    if assignment is not None:
        for fold in range(k):
            train, test = assignment != fold, assignment == fold
            train_idx, test_idx = torch.where(train)[0].tolist(), torch.where(test)[0].tolist()
            _response(y[train], family)
            tz, td, tf = _design(frame, frame.sample.iloc[train_idx], weights[train])
            vz = transform_design(frame.sample.iloc[test_idx], td)
            # Different fold widths are possible when an entire category is absent.
            # transform_design refuses such unseen held-out categories, never changes the scored sample.
            train_grid = list(grid) if fractions is None else [
                _reference(tz, y[train], weights[train], tf, family, frame.option("l1_ratio"), spec.intercept) * f
                for f in fractions]
            paths = solve_path(tz, y[train], weights[train], train_grid,
                               penalty_factors=tf, **solver_options)
            fold_losses = []
            for index, point in enumerate(paths):
                eta = vz @ torch.tensor(point["coefficients"], dtype=torch.float64) + point["constant"]
                loss = _loss(eta, y[test], family)
                scores[index] += float(normalized[test] @ loss)
                fold_losses.append(float(_weights(weights[test]) @ loss))
            cv.append({"fold": fold, "training_positions": [frame.positions[i] for i in train_idx],
                       "validation_positions": [frame.positions[i] for i in test_idx], "design": td,
                       "paths": paths, "lambda_path": train_grid, "validation_loss": fold_losses})
    paths = solve_path(z, y, weights, grid, penalty_factors=factors, **solver_options)
    selected = min(range(path_count), key=lambda i: (scores[i], -grid[i])) if k else grid.index(float(penalty))
    point = paths[selected]
    eta = z @ torch.tensor(point["coefficients"], dtype=torch.float64) + point["constant"]
    fitted = eta.sigmoid() if family == "binomial" else eta.exp()
    if not bool(torch.isfinite(fitted).all()):
        raise AnalysisError("numerical_failure", "Fitted response exceeds finite float64 arithmetic.")
    state = {"version": "openecon.regularized-glm.v1", "family": family,
             "spec": spec.model_dump(mode="json"), "design": design, "paths": paths,
             "selected_index": selected, "selection": selection, "lambda_path": grid,
             "fractions": fractions, "grid_units": "absolute supplied grid" if fractions is None else "training-null-score reference fractions",
             "fold_assignments": None if assignment is None else assignment.tolist(), "cv": cv,
             "cv_scores": scores if k else None, "seed": frame.option("seed"),
             "physical_positions": frame.positions, "raw_weights": weights.tolist(),
             "normalized_weights": normalized.tolist(), "weight_type": spec.weight_type,
             "physical_nobs": frame.n,
             "frequency_total": sum(int(v) for v in weights.tolist()) if spec.weight_type == "fweight" else None,
             "weight_normalization": "max-rescale then sum to one separately for each fit/fold",
             "preprocessing_scope": "each training fold; selected refit on complete retained training sample",
             "planned_work": planned, "penalty_factor_normalization": "none; literal factors on both L1 and L2",
             "loss": "normalized weighted negative log likelihood; Poisson log-factorial constants omitted"}
    state["digest"] = _digest(state)
    note = "Prediction only: penalized/CV-selected parameters have no classical coefficient SE/df/p/CI; a/p weights are empirical loss weights, not survey inference."
    return build_result(frame, terms=[], params=torch.empty(0, dtype=torch.float64),
                        covariance=torch.empty((0, 0), dtype=torch.float64), use_t=False,
                        fitted=fitted, metrics={"training_loss": float(normalized @ _loss(eta, y, family)),
                                                "selected_penalty": grid[selected]},
                        inference={"target": "prediction", "available": False, "distribution": "none",
                                   "use_t": None, "correction": "no penalized coefficient inference"},
                        extra={"target": "prediction", "regularized_glm_state": state, "notes": [note]},
                        solver="native float64 proximal Newton, verified KKT and finite free-direction geometry",
                        solver_diagnostics={"kkt_max": point["kkt_max"], "iterations": point["iterations"], "converged": True},
                        categories=design.get("levels", {}), warnings=[note])


def fit_elasticnet_logit(spec, data):
    return _fit(spec, data, "binomial")


def fit_elasticnet_poisson(spec, data):
    return _fit(spec, data, "poisson")


def _validated(result):
    if not isinstance(result, ResultBundle) or result.spec.estimator not in {"elasticnet_logit", "elasticnet_poisson"}:
        raise AnalysisError("invalid_result", "Saved penalized GLM helper needs an elasticnet_logit/elasticnet_poisson ResultBundle.")
    state = result.extra.get("regularized_glm_state")
    try:
        if not isinstance(state, dict) or state.get("version") != "openecon.regularized-glm.v1" or state.get("digest") != _digest(state):
            raise ValueError("digest")
        if state["spec"] != result.spec.model_dump(mode="json") or state["physical_positions"] != result.sample_positions:
            raise ValueError("spec/sample")
        family = "binomial" if result.spec.estimator == "elasticnet_logit" else "poisson"
        if state["family"] != family or result.coefficients or result.covariance_matrix or result.inference.get("available") is not False:
            raise ValueError("family/inference")
        design, paths = state["design"], state["paths"]
        _validate_state(design)
        if not isinstance(paths, list) or not 1 <= len(paths) <= 100 or type(state["selected_index"]) is not int or not 0 <= state["selected_index"] < len(paths):
            raise ValueError("dimensions")
        if design["predictors"] != result.spec.predictors or design["categorical"] != result.spec.categorical or design["intercept"] != result.spec.intercept:
            raise ValueError("design")
        expected_factors, expected_forced = _factors(result.spec.predictors, result.spec.options.get("penalty_factors"), result.spec.options.get("forced_controls"))
        def design_bound(value):
            if value["standardize"] != result.spec.options.get("standardize", True) or value["intercept"] != result.spec.intercept or value["predictors"] != result.spec.predictors or value["categorical"] != result.spec.categorical or value["penalty_factors"] != expected_factors or value["forced_controls"] != expected_forced:
                raise ValueError("preprocessing/factors specification")
        design_bound(design)
        grid = _grid(state["lambda_path"])
        if grid != state["lambda_path"] or len(grid) != len(paths):
            raise ValueError("grid")
        tolerance = result.spec.options.get("tolerance", 1e-8)
        def points_valid(values, penalties, width):
            if not isinstance(values, list) or len(values) != len(penalties):
                raise ValueError("path count")
            for point, penalty in zip(values, penalties, strict=True):
                if len(point["coefficients"]) != width or point["converged"] is not True or point["penalty"] != penalty:
                    raise ValueError("path")
                if not all(type(v) in {int, float} and math.isfinite(v) for v in [*point["coefficients"], point["constant"], point["objective"], point["kkt_max"]]):
                    raise ValueError("finite")
                if not 0 <= point["kkt_max"] <= tolerance or point["kkt_limit"] != tolerance:
                    raise ValueError("KKT")
                correction = point["free_newton_correction"]
                if type(correction) not in {int, float} or not math.isfinite(correction) or not 0 <= correction <= math.sqrt(tolerance):
                    raise ValueError("free geometry convergence")
                if type(point["iterations"]) is not int or not 0 <= point["iterations"] <= result.spec.options.get("max_iterations", 200):
                    raise ValueError("iterations")
                if not result.spec.intercept and point["constant"] != 0:
                    raise ValueError("unfitted intercept")
        points_valid(paths, grid, len(design["terms"]))
        positions = state["physical_positions"]
        if not 4 <= len(positions) <= 5000 or state["physical_nobs"] != len(positions) or any(type(v) is not int or v < 0 for v in positions) or len(set(positions)) != len(positions):
            raise ValueError("sample")
        if result.nobs != len(positions) or result.dropped_rows != result.nobs_original-len(positions) or any(v >= result.nobs_original for v in positions):
            raise ValueError("physical sample counts")
        raw, normalized = state["raw_weights"], state["normalized_weights"]
        if len(raw) != len(positions) or len(normalized) != len(raw) or state["weight_type"] != result.spec.weight_type:
            raise ValueError("weights")
        if not all(type(v) in {int, float} and math.isfinite(v) and v > 0 for v in [*raw, *normalized]):
            raise ValueError("weights")
        expected = _weights(torch.tensor(raw, dtype=torch.float64))
        if not torch.allclose(expected, torch.tensor(normalized, dtype=torch.float64), rtol=1e-12, atol=0):
            raise ValueError("weight normalization")
        if result.spec.weight_type == "fweight" and (any(v != int(v) or v > 2**53 for v in raw) or state["frequency_total"] != sum(int(v) for v in raw)):
            raise ValueError("frequency")
        selection = result.spec.options.get("selection", "cv")
        if state["selection"] != selection:
            raise ValueError("selection")
        requested_grid = result.spec.options.get("lambda_path")
        if requested_grid is not None and grid != _grid(requested_grid):
            raise ValueError("requested grid")
        if selection == "fixed":
            if state["fold_assignments"] is not None or state["cv"] or state["cv_scores"] is not None or grid[state["selected_index"]] != result.spec.options.get("penalty"):
                raise ValueError("fixed selection")
            if state["fractions"] is not None or (requested_grid is None and len(grid) != 1):
                raise ValueError("fixed grid")
        else:
            k = result.spec.options.get("folds", 5)
            assignment = state["fold_assignments"]
            if not isinstance(assignment, list) or len(assignment) != len(positions) or any(type(v) is not int or not 0 <= v < k for v in assignment) or len(state["cv"]) != k or len(state["cv_scores"]) != len(grid):
                raise ValueError("CV dimensions")
            if not all(type(v) in {int, float} and math.isfinite(v) for v in state["cv_scores"]):
                raise ValueError("CV scores")
            if state["selected_index"] != min(range(len(grid)), key=lambda i: (state["cv_scores"][i], -grid[i])):
                raise ValueError("CV selection")
            fractions = state["fractions"]
            if requested_grid is None:
                wanted = torch.logspace(0, math.log10(result.spec.options.get("lambda_ratio", 0.001)), result.spec.options.get("n_lambdas", 20), dtype=torch.float64).tolist()
                if not isinstance(fractions, list) or len(fractions) != len(grid) or len(fractions) != len(wanted) or any(type(f) not in {int, float} or not math.isfinite(f) or not math.isclose(f, w, rel_tol=1e-12, abs_tol=1e-15) for f, w in zip(fractions, wanted, strict=True)) or any(not math.isclose(v, grid[0]*f, rel_tol=1e-12, abs_tol=0) for v, f in zip(grid, fractions, strict=True)):
                    raise ValueError("reference fractions")
            elif fractions is not None:
                raise ValueError("absolute grid units")
            aggregate = [0.0] * len(grid)
            for fold, record in enumerate(state["cv"]):
                _validate_state(record["design"])
                design_bound(record["design"])
                if record["fold"] != fold or record["training_positions"] != [v for i, v in enumerate(positions) if assignment[i] != fold] or record["validation_positions"] != [v for i, v in enumerate(positions) if assignment[i] == fold]:
                    raise ValueError("CV row identity")
                fold_grid = _grid(record["lambda_path"])
                if len(fold_grid) != len(grid) or fold_grid != record["lambda_path"]:
                    raise ValueError("CV grid")
                if (requested_grid is not None and fold_grid != grid) or (fractions is not None and any(not math.isclose(v, fold_grid[0]*f, rel_tol=1e-12, abs_tol=0) for v, f in zip(fold_grid, fractions, strict=True))):
                    raise ValueError("fold grid units")
                points_valid(record["paths"], fold_grid, len(record["design"]["terms"]))
                if record["design"]["predictors"] != result.spec.predictors or record["design"]["categorical"] != result.spec.categorical:
                    raise ValueError("CV schema")
                losses = record["validation_loss"]
                if not isinstance(losses, list) or len(losses) != len(grid) or any(type(v) not in {int, float} or not math.isfinite(v) for v in losses):
                    raise ValueError("CV loss dimensions")
                mass = sum(normalized[i] for i, value in enumerate(assignment) if value == fold)
                aggregate = [total+mass*loss for total, loss in zip(aggregate, losses, strict=True)]
            if any(not math.isclose(total, score, rel_tol=1e-10, abs_tol=1e-12) for total, score in zip(aggregate, state["cv_scores"], strict=True)):
                raise ValueError("pooled CV loss")
    except (AnalysisError, ValueError, TypeError, KeyError, OverflowError, RuntimeError):
        raise AnalysisError("invalid_result", "Saved penalized GLM state is corrupted or disagrees with its fitted specification/sample.") from None
    return state


def regularized_glm_predict(result, data, *, kind="response", missing="raise", device="cpu", max_work=200000000) -> openecon.DataFrame:
    """Predict saved penalized GLM response/link; preserve retained physical query rows; no intervals."""
    from openecon.analysis import _coerce_frame
    from openecon.dataset import Dataset
    if device != "cpu":
        raise AnalysisError("unsupported_device", "Saved penalized GLM prediction supports CPU float64 only.")
    if kind not in {"response", "link"} or missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_option", "kind is response/link and missing is raise/drop.")
    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "Saved penalized GLM does not collect Dataset inputs.")
    state = _validated(result)
    data = _coerce_frame(data)
    if data.columns.has_duplicates or not 1 <= len(data) <= 100000:
        raise AnalysisError("invalid_query", "Query requires unique columns and 1..100,000 physical rows.")
    absent = [name for name in result.spec.predictors if name not in data]
    if absent:
        raise AnalysisError("missing_columns", "Query is missing fitted predictor columns.")
    n, p = len(data), len(state["design"]["terms"])
    work_guard(n * (p + 1) * 8, max_work, "saved penalized GLM prediction")
    plan_workspace("saved penalized GLM query encoding and results", {
        "query_input_and_indices": sum(int(data[name].memory_usage(index=False, deep=True)) for name in result.spec.predictors) * 2 + n * 32,
        "encoding_standardization_and_output": tensor_bytes((n, p + 2), itemsize=48)})
    mask = data[result.spec.predictors].isna().any(axis=1)
    if bool(mask.any()) and missing == "raise":
        raise AnalysisError("missing_values", "Query contains missing predictors; choose missing='drop' explicitly.")
    positions = (~mask).to_numpy().nonzero()[0].tolist()
    if not positions:
        raise AnalysisError("empty_sample", "No complete query rows remain.")
    sample = data.iloc[positions]
    z = transform_design(sample, state["design"])
    point = state["paths"][state["selected_index"]]
    eta = z @ torch.tensor(point["coefficients"], dtype=torch.float64) + point["constant"]
    if not bool(torch.isfinite(eta).all()):
        raise AnalysisError("numerical_failure", "Saved query link exceeds finite float64 arithmetic; saturated transforms cannot hide overflow.")
    if kind == "response" and state["family"] == "poisson" and bool((eta.abs() > 700).any()):
        raise AnalysisError("numerical_failure", "Poisson response prediction requires |link|<=700; means are never clipped or underflowed to zero.")
    predicted = eta if kind == "link" else eta.sigmoid() if state["family"] == "binomial" else eta.exp()
    if not bool(torch.isfinite(predicted).all()):
        raise AnalysisError("numerical_failure", "Saved query response exceeds finite float64 arithmetic.")
    return table([[v] for v in predicted.tolist()], columns=["predicted"], index=sample.index,
                 target="penalized GLM prediction", kind=kind, inference="unavailable",
                 source_result_id=result.id, state_digest=state["digest"], row_positions=positions,
                 dropped_rows=n-len(positions), selected_penalty=point["penalty"])


def regularized_glm_path(result) -> openecon.DataFrame:
    """Export full original-unit predictive paths and exact loss/KKT; no coefficient inference."""
    state = _validated(result)
    design, rows = state["design"], []
    scale, center = design["scales"], design["centers"]
    for i, point in enumerate(state["paths"]):
        coefficients = [b / s for b, s in zip(point["coefficients"], scale, strict=True)]
        constant = point["constant"] - sum(c*b for c, b in zip(center, coefficients, strict=True))
        if not all(math.isfinite(v) for v in [constant, *coefficients]):
            raise AnalysisError("numerical_failure", "Original-unit predictive coefficients exceed finite float64 arithmetic.")
        terms = list(zip(design["terms"], coefficients, strict=True))
        if design["intercept"]:
            terms = [("Intercept", constant), *terms]
        for term, coefficient in terms:
            rows.append([point["penalty"], term, coefficient, point["objective"], point["kkt_max"],
                         point["iterations"], i == state["selected_index"]])
    return table(rows, columns=["Penalty", "Term", "Estimate", "Objective", "KKT", "Iterations", "Selected"],
                 target="prediction", inference="unavailable", state_digest=state["digest"],
                 family=state["family"], penalty_factor_normalization="none", cv_scores=state["cv_scores"])
