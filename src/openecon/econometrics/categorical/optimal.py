"""Bounded single-vector optimal scaling; descriptive local solutions only."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace, workspace_budget_bytes

DTYPE = torch.float64
DEFAULT_WORK = 300_000_000
DEFAULT_BYTES = 128 * 1024**2
SOURCES = [
    "https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catreg.pdf",
    "https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf",
    "https://arxiv.org/abs/1611.05433",
]


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _error(f"{name} must be an integer in [{low}, {high}].", "invalid_option")
    return int(value)


def _real(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
        _error(f"{name} must be finite in [{low}, {high}].", "invalid_option")
    return float(value)


def _names(names, low=1):
    if not isinstance(names, (list, tuple)) or not low <= len(names) <= 12 or any(not isinstance(x, str) or not x for x in names) or len(set(names)) != len(names):
        _error(f"Use {low}–12 distinct variable names.")
    return list(names)


def _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights):
    if device != "cpu" or weights is not None:
        _error("Optimal scaling supports unweighted resident CPU float64 only.", "unsupported_option")
    return {
        "n_starts": _integer(n_starts, "n_starts", 1, 12),
        "seed": _integer(seed, "seed", 0, 2**31-1),
        "maxiter": _integer(maxiter, "maxiter", 1, 1000),
        "tol": _real(tol, "tol", 1e-12, 1e-3),
        "max_work": _integer(max_work, "max_work", 1, 2**63-1),
        "max_bytes": _integer(max_bytes, "max_bytes", 1, 2**63-1),
    }


def _frame(data, names, missing, max_bytes):
    if missing not in ("drop", "raise"):
        _error("missing must be 'drop' or 'raise'.", "invalid_option")
    if isinstance(data, pd.DataFrame):
        size = len(data)
    elif isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Required columns must occur exactly once.")
        try:
            size = max(len(data[name]) for name in names)
        except (TypeError, ValueError):
            _error("Input mapping must contain finite column sequences.")
    elif isinstance(data, list):
        size = len(data)
    else:
        _error("Supply a resident DataFrame, column mapping or list of records; Dataset is unsupported.", "unsupported_data")
    if not 1 <= size <= 3000:
        _error("Optimal scaling admits 1–3000 input rows before coercion.", "resource_limit")
    input_plan = plan_workspace("optimal-scaling selected input and row masks", {
        "bounded_named_input_copies": 32*size*len(names),
        "missing_mask_positions_and_indices": 32*size,
    }, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    if isinstance(data, Mapping):
        if any(name not in data for name in names):
            _error("Required columns must occur exactly once.")
        data = {name: data[name] for name in names}
    elif isinstance(data, list):
        if any(not isinstance(row, Mapping) for row in data) or any(not any(name in row for row in data) for name in names):
            _error("Supply records with every required column represented.")
        data = [{name: row.get(name) for name in names} for row in data]
    try:
        frame = _coerce_frame(data)
    except (ValueError, TypeError) as exc:
        raise AnalysisError("invalid_spec", "Selected input columns cannot form a valid resident table.") from exc
    if frame.columns.has_duplicates or any(name not in frame for name in names):
        _error("Required columns must occur exactly once.")
    # The row count was checked before this bounded selection and mask.
    selected = frame.loc[:, names]
    absent = selected.isna().any(axis=1)
    if bool(absent.any()) and missing == "raise":
        _error("Missing values occur in analysis columns.", "missing_values")
    positions = [i for i, keep in enumerate((~absent).tolist()) if keep]
    return selected.loc[~absent].reset_index(drop=True), positions, size, input_plan.record()


def _label(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, str):
        kind = "str"
    elif isinstance(value, Integral):
        kind, value = "int", int(value)
    elif isinstance(value, Real) and math.isfinite(value):
        kind, value = "float", float(value)
    else:
        _error("Categorical values must be finite strings, booleans, integers or floats.")
    return [kind, value]


def _key(label):
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _numeric(series):
    if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
        _error(f"Explicit numeric column {series.name!r} must be numeric.")
    values = series.to_numpy(dtype=float).tolist()
    if any(not math.isfinite(x) or abs(x) > 1e100 for x in values):
        _error("Numeric columns must be finite with magnitude at most 1e100.")
    return torch.tensor(values, dtype=DTYPE)


def _standardize(vector):
    mean = vector.mean()
    centered = vector - mean
    scale = torch.sqrt(centered.square().mean())
    if not float(scale) > 1e-12 * max(float(vector.abs().max()), 1.0):
        _error("A numeric column or transformed category score is constant/degenerate.", "degenerate_transform")
    return centered / scale, float(mean), float(scale)


def _prepare(frame, variables, scales, orders, default, controls, components=0):
    if scales is None:
        scales = {name: default for name in variables}
    if not isinstance(scales, dict) or set(scales) != set(variables) or any(value not in ("nominal", "ordinal", "numeric") for value in scales.values()):
        _error("scales must assign nominal, ordinal or numeric to every variable.")
    orders = {} if orders is None else orders
    if not isinstance(orders, dict) or set(orders) != {name for name in variables if scales[name] == "ordinal"}:
        _error("orders must explicitly name exactly the ordinal variables.")
    descriptors, tensors = [], []
    # Small category maps are built before numerical buffers; each map is bounded.
    for name in variables:
        if scales[name] == "numeric":
            descriptors.append({"name": name, "scale": "numeric"})
            tensors.append(None)
            continue
        observed, labels = [], []
        seen = set()
        for value in frame[name].tolist():
            label = _label(value)
            key = _key(label)
            observed.append(key)
            if key not in seen:
                labels.append(label)
                seen.add(key)
                if len(labels) > 32:
                    _error("At most 32 levels per categorical variable.", "resource_limit")
        if not 2 <= len(labels) <= 32:
            _error("Categorical variables need 2–32 observed levels.", "degenerate_transform")
        if scales[name] == "ordinal":
            order = orders[name]
            if not isinstance(order, (list, tuple)):
                _error("Each ordinal order must be a list of all observed levels.")
            labels = [_label(value) for value in order]
            keys = [_key(value) for value in labels]
            if len(keys) != len(set(keys)) or set(keys) != seen:
                _error("Each ordinal order must contain every complete-sample level exactly once.")
        mapping = {_key(label): i for i, label in enumerate(labels)}
        codes = [mapping[value] for value in observed]
        counts = [codes.count(i) for i in range(len(labels))]
        descriptors.append({"name": name, "scale": scales[name], "levels": labels, "counts": counts})
        tensors.append(codes)
    n, p = len(frame), len(variables)
    # Worst-case SVD/least-squares and coordinate/group operations for every iteration.
    category_work = sum(len(descriptor.get("levels", []))**2 for descriptor in descriptors)
    ordinal_inner = sum(100*(4*n*components+16*len(descriptor["levels"])*components)
                        for descriptor in descriptors if components and descriptor["scale"] == "ordinal")
    work = controls["n_starts"] * (controls["maxiter"]+2) * (8*n*p*p + 64*n*p + 64*category_work + ordinal_inner)
    if work > controls["max_work"]:
        _error(f"Declared optimal-scaling work {work} exceeds max_work.", "resource_limit")
    plan = plan_workspace("categorical optimal scaling", {
        "input_codes_scores_and_residuals": 8*n*(12*p+16),
        "svd_and_design_buffers": 8*(8*n*p + 12*p*p),
        "category_and_inner_buffers": 8*32*p*(32+2*p),
        "complete_objective_traces": 8*controls["n_starts"]*(controls["maxiter"]+1)*6,
    }, budget_bytes=min(controls["max_bytes"], workspace_budget_bytes()))
    prepared = []
    for descriptor, codes in zip(descriptors, tensors):
        if descriptor["scale"] == "numeric":
            values, mean, sd = _standardize(_numeric(frame[descriptor["name"]]))
            descriptor.update(mean=mean, sd=sd)
            prepared.append(values)
        else:
            prepared.append(torch.tensor(codes, dtype=torch.int64))
    return descriptors, prepared, plan.record(), work


def _pava(values, counts):
    """Weighted Euclidean projection onto the nondecreasing cone."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values.tolist(), counts.tolist())):
        blocks.append([i, i+1, weight, weight*value])
        while len(blocks) > 1 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
            last = blocks.pop()
            previous = blocks.pop()
            blocks.append([previous[0], last[1], previous[2]+last[2], previous[3]+last[3]])
    output = torch.empty_like(values)
    for start, stop, weight, total in blocks:
        output[start:stop] = total/weight
    return output


def _means(target, codes, levels):
    count = torch.bincount(codes, minlength=levels).to(DTYPE)
    if target.ndim == 1:
        total = torch.zeros(levels, dtype=DTYPE).index_add_(0, codes, target)
        return total/count, count
    total = torch.zeros((levels, target.shape[1]), dtype=DTYPE).index_add_(0, codes, target)
    return total/count[:, None], count


def _normalize_categories(values, counts, *, nominal=False):
    values = values - (values*counts).sum()/counts.sum()
    norm = torch.sqrt((values.square()*counts).sum()/counts.sum())
    if float(norm) <= 1e-12:
        _error("Optimal quantification collapses to a constant.", "degenerate_transform")
    values = values/norm
    if nominal:
        index = int(torch.argmax(values.abs()))
        if float(values[index]) < 0:
            values = -values
    return values


def _initial(descriptors, prepared, start, generator):
    columns = []
    for descriptor, values in zip(descriptors, prepared):
        if descriptor["scale"] == "numeric":
            columns.append(values.clone())
        else:
            count = torch.tensor(descriptor["counts"], dtype=DTYPE)
            score = torch.arange(len(count), dtype=DTYPE) if start == 0 else torch.randn(len(count), generator=generator, dtype=DTYPE)
            if descriptor["scale"] == "ordinal":
                score = torch.sort(score).values
            score = _normalize_categories(score, count, nominal=descriptor["scale"] == "nominal")
            columns.append(score[values])
    return torch.stack(columns, dim=1)


def _rank(matrix):
    singular = torch.linalg.svdvals(matrix)
    if len(singular) != matrix.shape[1] or float(singular[-1]) <= 1e-10*float(singular[0]):
        _error("Transformed regression design is rank-deficient or ill-conditioned.", "rank_deficient")
    return float(singular[0]/singular[-1])


def _seal(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _saved(output):
    saved_summary(output)
    for name, frame in list(output.items()):
        # summary_state persists the common-dtype array payload. Normalize new
        # numeric tables to that representation before both display and save.
        values = frame.to_numpy()
        if values.dtype.kind == "f":
            output[name] = table(values.tolist(), columns=list(frame.columns), index=list(frame.index))
    ordered = sorted(output.items())
    output.clear()
    output.update(ordered)
    return output


def _state(result, kinds):
    if not isinstance(result, TableSet) or result.attrs.get("method") not in kinds:
        _error("Supply a fitted or restored optimal-scaling result.", "invalid_result")
    state = result.attrs.get("optimal_state")
    try:
        if not isinstance(state, dict) or state["version"] != 1 or result.attrs["state_sha256"] != _seal(state):
            raise ValueError("Integrity mismatch")
        descriptors = state["descriptors"]
        names = _names([descriptor["name"] for descriptor in descriptors])
        if state["variables"] != names:
            raise ValueError("Invalid variable order")
        expected_kind = "catpca" if result.attrs["method"] == "catpca" else "catreg"
        if state["kind"] != expected_kind:
            raise ValueError("Wrong saved method")
        for descriptor in descriptors:
            if descriptor["scale"] == "numeric":
                if not math.isfinite(descriptor["mean"]) or not math.isfinite(descriptor["sd"]) or descriptor["sd"] <= 0:
                    raise ValueError("Invalid standardization")
            else:
                keys = [_key(label) for label in descriptor["levels"]]
                scores = descriptor["quantifications"]
                if descriptor["scale"] not in ("nominal", "ordinal") or not 2 <= len(keys) <= 32 or len(set(keys)) != len(keys) or len(scores) != len(keys) or any(not math.isfinite(value) for value in scores):
                    raise ValueError("Invalid category mapping")
                if any(not isinstance(label, list) or len(label) != 2 or _label(label[1]) != label for label in descriptor["levels"]):
                    raise ValueError("Invalid typed category identity")
                counts = descriptor["counts"]
                if len(counts) != len(scores) or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts):
                    raise ValueError("Invalid category frequencies")
                mean = sum(count*score for count, score in zip(counts, scores))/sum(counts)
                variance = sum(count*score*score for count, score in zip(counts, scores))/sum(counts)
                if abs(mean) > 1e-8 or abs(variance-1) > 1e-8:
                    raise ValueError("Invalid quantification normalization")
                if descriptor["scale"] == "ordinal" and any(b < a-1e-10 for a, b in zip(scores, scores[1:])):
                    raise ValueError("Nonmonotone saved mapping")
        if expected_kind == "catreg":
            if not isinstance(state["beta"], list) or len(state["beta"]) != len(names) or any(not math.isfinite(value) for value in state["beta"]) or not math.isfinite(state["intercept"]):
                raise ValueError("Invalid coefficient order")
        else:
            dimensions = state["components"]
            if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions < len(names):
                raise ValueError("Invalid component count")
            axes, eigenvalues = state["axes"], state["eigenvalues"]
            if len(axes) != len(names) or any(len(row) != dimensions or any(not math.isfinite(value) for value in row) for row in axes) or len(eigenvalues) != dimensions or any(not math.isfinite(value) or value <= 0 for value in eigenvalues):
                raise ValueError("Invalid component matrix dimensions")
        return state
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved optimal-scaling state has invalid structure or integrity.") from exc


def _mapped(frame, descriptors):
    output = []
    for descriptor in descriptors:
        if descriptor["scale"] == "numeric":
            output.append((_numeric(frame[descriptor["name"]])-descriptor["mean"])/descriptor["sd"])
        else:
            mapping = {_key(level): score for level, score in zip(descriptor["levels"], descriptor["quantifications"])}
            try:
                values = [mapping[_key(_label(value))] for value in frame[descriptor["name"]].tolist()]
            except KeyError as exc:
                raise AnalysisError("unknown_category", f"Unknown category in {descriptor['name']!r}; saved mappings are not refitted.") from exc
            output.append(torch.tensor(values, dtype=DTYPE))
    return torch.stack(output, dim=1) if len(frame) else torch.empty((0, len(descriptors)), dtype=DTYPE)


def _mapping_tables(descriptors):
    rows, numeric = [], []
    for descriptor in descriptors:
        if descriptor["scale"] == "numeric":
            numeric.append([descriptor["name"], descriptor["mean"], descriptor["sd"]])
        else:
            for level, count, score in zip(descriptor["levels"], descriptor["counts"], descriptor["quantifications"]):
                rows.append([descriptor["name"], descriptor["scale"], level[0], level[1], count, score])
    return {
        "quantifications": table(rows, columns=["variable", "scale", "category_type", "category", "count", "quantification"]),
        "numeric_scaling": table(numeric, columns=["variable", "mean", "population_sd"]),
    }


def _finish_maps(descriptors, prepared, transformed):
    for j, (descriptor, codes) in enumerate(zip(descriptors, prepared)):
        if descriptor["scale"] != "numeric":
            scores, _ = _means(transformed[:, j], codes, len(descriptor["levels"]))
            descriptor["quantifications"] = scores.tolist()


def _regression(data, outcome, predictors, scales, orders, default, controls, missing):
    variables = _names(predictors)
    if not isinstance(outcome, str) or outcome in variables:
        _error("A distinct numeric outcome is required.")
    if default == "nominal" and isinstance(scales, dict) and any(value == "ordinal" for value in scales.values()):
        _error("Ordinal predictors require catreg_ordinal and explicit orders.")
    frame, positions, n_input, input_plan = _frame(data, [outcome]+variables, missing, controls["max_bytes"])
    if len(frame) <= len(variables)+2:
        _error("Regression needs more complete rows than predictors + 2.", "insufficient_sample")
    descriptors, prepared, plan, work = _prepare(frame, variables, scales, orders, default, controls)
    y_original = _numeric(frame[outcome])
    y, y_mean, y_sd = _standardize(y_original)
    generator = torch.Generator(device="cpu").manual_seed(controls["seed"])
    histories, starts, candidates = [], [], []
    for start in range(controls["n_starts"]):
        iteration, objective = 0, None
        try:
            quantified = _initial(descriptors, prepared, start, generator)
            _rank(quantified)
            beta = torch.linalg.lstsq(quantified, y).solution
            objective = float((y-quantified@beta).square().mean())
            histories.append([start, 0, objective, 0.0])
            converged = False
            for iteration in range(1, controls["maxiter"]+1):
                old = objective
                fitted = quantified@beta
                for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
                    residual = y-fitted+beta[j]*quantified[:, j]
                    if descriptor["scale"] == "numeric":
                        contribution = torch.dot(values, residual)/len(y)*values
                        quantified[:, j] = values
                        beta[j] = torch.dot(values, residual)/len(y)
                    else:
                        means, counts = _means(residual, values, len(descriptor["levels"]))
                        if descriptor["scale"] == "ordinal":
                            increasing = _pava(means, counts)
                            decreasing = -_pava(-means, counts)
                            score_inc = float(((means-increasing).square()*counts).sum())
                            score_dec = float(((means-decreasing).square()*counts).sum())
                            contribution_scores = increasing if score_inc <= score_dec else decreasing
                        else:
                            contribution_scores = means
                        contribution_scores -= (contribution_scores*counts).sum()/counts.sum()
                        contribution = contribution_scores[values]
                        scores = _normalize_categories(contribution_scores, counts)
                        if descriptor["scale"] == "ordinal" and float(scores[-1]) < float(scores[0]):
                            scores = -scores
                        elif descriptor["scale"] == "nominal" and float(scores[int(torch.argmax(scores.abs()))]) < 0:
                            scores = -scores
                        quantified[:, j] = scores[values]
                        beta[j] = torch.dot(quantified[:, j], contribution)/len(y)
                    fitted = y-residual+contribution
                _rank(quantified)
                beta = torch.linalg.lstsq(quantified, y).solution
                objective = float((y-quantified@beta).square().mean())
                improvement = old-objective
                histories.append([start, iteration, objective, improvement])
                if improvement < -1e-9:
                    _error("ALS objective increased beyond roundoff.", "numerical_failure")
                if abs(improvement) <= controls["tol"]*max(1.0, old):
                    converged = True
                    break
            if not converged:
                starts.append([start, False, iteration, objective, "nonconvergence"])
                continue
            starts.append([start, True, iteration, objective, "accepted"])
            candidates.append((objective, start, quantified.clone(), beta.clone(), iteration))
        except AnalysisError as exc:
            starts.append([start, False, iteration, objective, getattr(exc, "code", "invalid_start")])
    if not candidates:
        _error("No start reached a nondegenerate, full-rank converged solution.", "nonconvergence")
    objective, chosen, quantified, beta, iterations = min(candidates, key=lambda candidate: candidate[:2])
    condition = _rank(quantified)
    _finish_maps(descriptors, prepared, quantified)
    beta_original = beta*y_sd
    fitted = y_mean+quantified@beta_original
    residual = y_original-fitted
    state = {"version": 1, "kind": "catreg", "variables": variables, "descriptors": descriptors,
             "outcome": outcome, "intercept": y_mean, "beta": beta_original.tolist()}
    result = TableSet({
        "coefficients": table([["_cons", y_mean]]+[[name, value] for name, value in zip(variables, beta_original.tolist())], columns=["term", "beta_descriptive"]),
        "fit": table([[len(y), float(residual.square().sum()), 1-objective, objective, condition]], columns=["n", "sse", "r_squared", "standardized_objective", "design_condition"]),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
        "fitted": table([[position, observed, fit, error] for position, observed, fit, error in zip(positions, y_original.tolist(), fitted.tolist(), residual.tolist())], columns=["source_position", "observed", "fitted", "residual"]),
        "iterations": table(histories, columns=["start", "iteration", "objective", "improvement"]),
        "starts": table(starts, columns=["start", "converged", "iterations", "objective", "status"]),
        **_mapping_tables(descriptors),
    }, title="Categorical regression with optimal scaling", method=f"catreg_{default}",
       variables=variables, outcome=outcome, n_input=n_input, n=len(y), n_missing=n_input-len(y),
       sample_positions=positions, missing=missing, settings=controls, resources=plan, input_preparation=input_plan, declared_work=work,
       chosen_start=chosen, converged=True, iterations=iterations, optimal_state=state, state_sha256=_seal(state),
       sources=SOURCES, inference="descriptive only; no covariance, SE, p-value or CI after adaptive quantification",
       solution="best converged declared deterministic start; local-solution guarantee only",
       dtype="float64", device="cpu", weight_type="unweighted")
    return _saved(result)


@resident_cpu
def catreg_nominal(data: Any, outcome: str, predictors: list[str], *, scales: dict | None = None,
                   n_starts: int = 3, seed: int = 0, maxiter: int = 500, tol: float = 1e-8,
                   missing: str = "drop", max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                   device: str = "cpu", weights: str | None = None) -> TableSet:
    """Numeric-outcome CATREG with nominal or explicitly numeric predictors.

    Nominal category contributions are updated from partial-residual means;
    weighted centering/unit population variance identifies each scalar map.
    Coefficients and R² are descriptive: adaptive maps do not receive naive
    OLS standard errors. Complete state uses oe.summary_state/restore_summary.
    """
    controls = _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    return _regression(data, outcome, predictors, scales, None, "nominal", controls, missing)


@resident_cpu
def catreg_ordinal(data: Any, outcome: str, predictors: list[str], *, orders: dict,
                   scales: dict | None = None, n_starts: int = 3, seed: int = 0,
                   maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
                   max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                   device: str = "cpu", weights: str | None = None) -> TableSet:
    """CATREG with declared ordinal orders and weighted isotonic updates.

    Every ordinal quantification is nondecreasing; negative beta expresses a
    decreasing response. Mixed nominal/numeric predictors require scales.
    Multiple deterministic starts assess local solutions, not global parity.
    """
    controls = _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    return _regression(data, outcome, predictors, scales, orders, "ordinal", controls, missing)


def _prediction(result, data, kinds, missing, max_work, max_bytes, device):
    if device != "cpu":
        _error("Saved mapping prediction supports resident CPU only.", "unsupported_option")
    _integer(max_work, "max_work", 1, 2**63-1)
    _integer(max_bytes, "max_bytes", 1, 2**63-1)
    state = _state(result, kinds)
    frame, positions, n_input, input_plan = _frame(data, state["variables"], missing, max_bytes)
    n, p = len(frame), len(state["variables"])
    if 8*n*p*p > max_work:
        _error("Saved-map prediction exceeds max_work.", "resource_limit")
    plan = plan_workspace("saved optimal-scaling prediction", {"mapped_scores_and_prediction": 8*n*(4*p+12)}, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    quantified = _mapped(frame, state["descriptors"])
    record = plan.record()
    record["input_preparation"] = input_plan
    return state, quantified, positions, n_input, record


@resident_cpu
def catreg_predict(result: TableSet, data: Any, *, missing: str = "raise", max_work: int = DEFAULT_WORK,
                   max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Apply complete saved CATREG maps and coefficients without refitting.

    Unknown levels fail; prediction is descriptive and supplies no adaptive
    transform confidence interval. Missing drop preserves source positions.
    """
    state, quantified, positions, n_input, plan = _prediction(result, data, ("catreg_nominal", "catreg_ordinal"), missing, max_work, max_bytes, device)
    beta = torch.tensor(state["beta"], dtype=DTYPE)
    if beta.shape != (len(state["variables"]),) or not bool(torch.isfinite(beta).all()) or not math.isfinite(state["intercept"]):
        _error("Saved regression coefficients are invalid.", "invalid_state")
    fitted = state["intercept"]+quantified@beta
    return _saved(TableSet({
        "predictions": table([[position, value] for position, value in zip(positions, fitted.tolist())], columns=["source_position", "fitted"]),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+state["variables"]),
    }, title="Saved CATREG predictions", method="catreg_predict", n_input=n_input, n=len(positions), n_missing=n_input-len(positions), sample_positions=positions, state_sha256=result.attrs["state_sha256"], resources=plan, inference="descriptive only", device="cpu", dtype="float64"))


def _components(quantified, dimensions):
    u, singular, vh = torch.linalg.svd(quantified, full_matrices=False)
    if float(singular[dimensions-1]) <= 1e-10*float(singular[0]):
        _error("Requested CATPCA component space is rank-deficient.", "rank_deficient")
    n = len(quantified)
    axes = vh[:dimensions].T
    # Canonical component signs; ties remain rotationally unidentified.
    for k in range(dimensions):
        if float(axes[int(torch.argmax(axes[:, k].abs())), k]) < 0:
            axes[:, k] *= -1
            u[:, k] *= -1
    scores = u[:, :dimensions]*math.sqrt(n)
    loadings = axes*(singular[:dimensions]/math.sqrt(n))[None, :]
    objective = float((quantified-scores@loadings.T).square().sum()/n)
    return scores, loadings, axes, singular.square()/n, objective


def _pca_update(scores, codes, descriptor, previous):
    centroids, counts = _means(scores, codes, len(descriptor["levels"]))
    if descriptor["scale"] == "nominal":
        weighted = centroids*torch.sqrt(counts)[:, None]
        left, singular, _ = torch.linalg.svd(weighted, full_matrices=False)
        if float(singular[0]) <= 1e-12:
            _error("CATPCA category centroids provide no nondegenerate direction.", "degenerate_transform")
        quantification = _normalize_categories(left[:, 0]/torch.sqrt(counts), counts, nominal=True)
        return quantification[codes]
    # Fix object scores, alternate rank-one loading and weighted isotonic map.
    quantification, _ = _means(previous, codes, len(counts))
    for _ in range(100):
        loading = scores.T@quantification[codes]/len(codes)
        target = centroids@loading
        next_q = _normalize_categories(_pava(target, counts), counts)
        difference = float((quantification-next_q).square().max())
        quantification = next_q
        if difference <= 1e-16:
            break
    else:
        _error("Ordinal CATPCA inner quantification did not converge.", "nonconvergence")
    return quantification[codes]


@resident_cpu
def catpca(data: Any, variables: list[str], *, components: int = 2, scales: dict | None = None,
           orders: dict | None = None, n_starts: int = 3, seed: int = 0,
           maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
           max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
           device: str = "cpu", weights: str | None = None) -> TableSet:
    """Single-vector nominal/ordinal/numeric categorical PCA by constrained ALS.

    Columns are centered with population variance one; object scores have
    X'X/n=I. Numeric-only scaling reduces to correlation PCA. Multiple-nominal
    maps, rotation, inference and automatic discretization are outside scope.
    """
    variables = _names(variables, 2)
    dimensions = _integer(components, "components", 1, len(variables)-1)
    controls = _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device, weights)
    frame, positions, n_input, input_plan = _frame(data, variables, missing, controls["max_bytes"])
    if len(frame) <= dimensions+2:
        _error("CATPCA needs more complete rows than components + 2.", "insufficient_sample")
    descriptors, prepared, plan, work = _prepare(frame, variables, scales, orders, "nominal", controls, components=dimensions)
    generator = torch.Generator(device="cpu").manual_seed(controls["seed"])
    histories, starts, candidates = [], [], []
    for start in range(controls["n_starts"]):
        iteration, objective = 0, None
        try:
            quantified = _initial(descriptors, prepared, start, generator)
            scores, loadings, axes, eigenvalues, objective = _components(quantified, dimensions)
            histories.append([start, 0, objective, 0.0])
            converged = False
            for iteration in range(1, controls["maxiter"]+1):
                old = objective
                for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
                    if descriptor["scale"] != "numeric":
                        quantified[:, j] = _pca_update(scores, values, descriptor, quantified[:, j])
                scores, loadings, axes, eigenvalues, objective = _components(quantified, dimensions)
                improvement = old-objective
                histories.append([start, iteration, objective, improvement])
                if improvement < -1e-9:
                    _error("CATPCA objective increased beyond roundoff.", "numerical_failure")
                if abs(improvement) <= controls["tol"]*max(1.0, old):
                    converged = True
                    break
            if not converged:
                starts.append([start, False, iteration, objective, "nonconvergence"])
                continue
            starts.append([start, True, iteration, objective, "accepted"])
            candidates.append((objective, start, quantified.clone(), scores.clone(), loadings.clone(), axes.clone(), eigenvalues.clone(), iteration))
        except AnalysisError as exc:
            starts.append([start, False, iteration, objective, getattr(exc, "code", "invalid_start")])
    if not candidates:
        _error("No CATPCA start reached a converged nondegenerate solution.", "nonconvergence")
    objective, chosen, quantified, scores, loadings, axes, eigenvalues, iterations = min(candidates, key=lambda candidate: candidate[:2])
    _finish_maps(descriptors, prepared, quantified)
    state = {"version": 1, "kind": "catpca", "variables": variables, "descriptors": descriptors,
             "components": dimensions, "axes": axes.tolist(), "eigenvalues": eigenvalues[:dimensions].tolist()}
    names = [f"component_{j+1}" for j in range(dimensions)]
    result = TableSet({
        "fit": table([[len(frame), dimensions, objective, 1-objective/len(variables)]], columns=["n", "components", "reconstruction_loss", "variance_proportion"]),
        "loadings": table(loadings.tolist(), columns=names, index=variables),
        "eigenvectors": table(axes.tolist(), columns=names, index=variables),
        "eigenvalues": table([[j+1, value, value/len(variables)] for j, value in enumerate(eigenvalues.tolist())], columns=["component", "eigenvalue", "proportion"]),
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
        "iterations": table(histories, columns=["start", "iteration", "objective", "improvement"]),
        "starts": table(starts, columns=["start", "converged", "iterations", "objective", "status"]),
        **_mapping_tables(descriptors),
    }, title="Categorical principal components with single-vector optimal scaling", method="catpca",
       variables=variables, n_input=n_input, n=len(frame), n_missing=n_input-len(frame), sample_positions=positions,
       components=dimensions, settings=controls, resources=plan, input_preparation=input_plan, declared_work=work, missing=missing,
       chosen_start=chosen, converged=True, iterations=iterations, optimal_state=state, state_sha256=_seal(state),
       sources=SOURCES, inference="descriptive reconstruction only; no adaptive uncertainty",
       solution="best converged declared deterministic start; local-solution guarantee only",
       normalization="population-standardized columns; object scores X'X/n=I; no multiple-nominal scaling",
       dtype="float64", device="cpu", weight_type="unweighted")
    return _saved(result)


@resident_cpu
def catpca_predict(result: TableSet, data: Any, *, missing: str = "raise", max_work: int = DEFAULT_WORK,
                   max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Saved single-vector CATPCA maps and normalized object-score projection."""
    state, quantified, positions, n_input, plan = _prediction(result, data, ("catpca",), missing, max_work, max_bytes, device)
    axes = torch.tensor(state["axes"], dtype=DTYPE)
    eigenvalues = torch.tensor(state["eigenvalues"], dtype=DTYPE)
    dimensions = state["components"]
    if axes.shape != (len(state["variables"]), dimensions) or eigenvalues.shape != (dimensions,) or not bool(torch.isfinite(axes).all()) or not bool((eigenvalues > 0).all()) or not torch.allclose(axes.T@axes, torch.eye(dimensions, dtype=DTYPE), atol=1e-8, rtol=1e-8):
        _error("Saved component axes/eigenvalues are invalid.", "invalid_state")
    scores = quantified@axes/torch.sqrt(eigenvalues)[None, :]
    reconstructed = scores@(axes*torch.sqrt(eigenvalues)[None, :]).T
    columns = [f"component_{j+1}" for j in range(dimensions)]
    return _saved(TableSet({
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+columns),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+state["variables"]),
        "reconstruction": table([[position]+row for position, row in zip(positions, reconstructed.tolist())], columns=["source_position"]+state["variables"]),
    }, title="Saved CATPCA projection", method="catpca_predict", n_input=n_input, n=len(positions), n_missing=n_input-len(positions), sample_positions=positions, state_sha256=result.attrs["state_sha256"], resources=plan, inference="descriptive only", device="cpu", dtype="float64"))
