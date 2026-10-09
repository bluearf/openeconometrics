"""CCA option adapters: frequency counts, declared joint moments and saved scores.

Summary and frequency routes whiten conditioned within-set correlation blocks;
the pre-existing resident unweighted route keeps its raw-data QR computation.
No observations, training means, sampling degrees of freedom or unique axes
are fabricated. Inference retains the existing Gaussian test assumptions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import plan_workspace
from . import common as c

MAX_VARIABLES = 64
MAX_PHYSICAL_ROWS = 1_000_000
MAX_WORK = 500_000_000
MAX_BLOCK_CONDITION = 1e10


def _names(x, y):
    xs, ys = c.name_list(x, "x"), c.name_list(y, "y")
    if set(xs) & set(ys):
        raise AnalysisError("invalid_spec", "Canonical x and y sets must contain distinct variables.")
    return xs, ys


def geometry(width: int, *, rows: int = 0, selected_rows: int = 0) -> dict:
    if width > MAX_VARIABLES:
        raise AnalysisError("workspace_limit", "CCA options support at most 64 joint variables.")
    if rows > MAX_PHYSICAL_ROWS:
        raise AnalysisError("work_limit", "CCA frequency/score options support at most 1000000 physical rows per source.")
    work = rows * width * width + 64 * width**3
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "CCA joint-moment/score work exceeds its 500000000 operation budget.")
    record = plan_workspace("canonical correlation options", {
        "joint_matrices_and_factorizations": 64 * width * width * 8,
        "selected_moment_or_score_buffers": selected_rows * (width + 1) * 96,
    }).record()
    return {**record, "estimated_work": work}


def _row_count(data, used):
    if isinstance(data, Dataset):
        absent = [name for name in used if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        return data.row_count
    if isinstance(data, pd.DataFrame):
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "Data must have unique column names.")
        return len(data)
    if isinstance(data, Mapping):
        absent = [name for name in used if name not in data]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        try:
            sizes = [len(data[name]) for name in used]
        except TypeError as exc:
            raise AnalysisError("invalid_data", "Supply resident columns of scalar observations.") from exc
        if len(set(sizes)) != 1:
            raise AnalysisError("invalid_data", "Input columns must have consistent lengths.")
        for name in used:
            if isinstance(data[name], Tensor) and data[name].device.type != "cpu":
                raise AnalysisError("unsupported_device", "CCA options require CPU inputs.")
        return sizes[0]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return len(data)
    raise AnalysisError("invalid_data", "Supply resident rows or a replayable Dataset.")


def _selected(data, used):
    if isinstance(data, pd.DataFrame):
        absent = [name for name in used if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        return data.loc[:, used]
    if isinstance(data, Mapping):
        return {name: data[name] for name in used}
    if not all(isinstance(row, Mapping) for row in data):
        raise AnalysisError("invalid_data", "Every resident row must be a mapping of scalar observations.")
    return [{name: row.get(name) for name in used} for row in data]


def _bounded_source(data: Dataset, used: list[str], width: int) -> Dataset:
    """Limit unknown row-count work before a source block enters the moment fit."""
    def batches():
        count = 0
        for raw in data.iter_batches(used):
            count += len(raw)
            if count > MAX_PHYSICAL_ROWS or count * width**2 + 64 * width**3 > MAX_WORK:
                raise AnalysisError("work_limit", "CCA source exceeds its physical-row or moment-work budget.")
            yield raw
    return Dataset.from_batches(batches, columns=used, row_count=data.row_count)


def weighted_canon(data, xs, ys, *, weights, missing):
    from .weighted import frequency_moments
    weights = c.check_name(weights, "weights")
    names = [*xs, *ys]
    if weights in names:
        raise AnalysisError("invalid_spec", "Weight and canonical variable roles must be distinct.")
    c.check_choice(missing, "missing", ("drop", "raise"))
    used = [*names, weights]
    count = _row_count(data, used)
    selected_rows = min(65536, count or 65536) if isinstance(data, Dataset) else count
    plan = geometry(len(names), rows=count or 0, selected_rows=selected_rows)
    source = _bounded_source(data, used, len(names)) if isinstance(data, Dataset) else _selected(data, used)
    mean, sscp, n, dropped, extra = frequency_moments(source, names, weights, missing)
    physical_total = extra["physical_rows"] + extra["n_zero_weight"] + dropped
    geometry(len(names), rows=physical_total)
    extra.update({"option_resource_plan": plan, "input_kind": "frequency_rows",
                  "training_means_supplied": True, "training_sds_supplied": True,
                  "physical_rows_total": physical_total,
                  "inference": "Gaussian canonical tests for literal original independent observation counts; no pseudo-replication, survey or cluster inference",
                  "frequency_contract": "nonnegative exact integer counts; covariance divisor sum(w)-1"})
    return _from_covariance(sscp / (n - 1), mean, n, dropped, xs, ys, extra)


def matrix_canon(values, *, n, x, y, matrix, means, sds):
    from .summary import prepare
    xs, ys = _names(x, y)
    names = [*xs, *ys]
    plan = geometry(len(names))
    state = prepare(values, n=n, columns=names, matrix=matrix, means=means, sds=sds)
    extra = {**state.attrs, "option_resource_plan": plan,
             "matrix_sample_contract": "one common centred complete sample; covariance divisor n-1; declared n",
             "inference": "Gaussian canonical tests conditional on the declared common sample n and valid sample covariance; no survey or cluster inference"}
    return _from_covariance(state.covariance, state.mean, state.n, 0, xs, ys, extra)


def _block(correlation: Tensor, what: str) -> tuple[Tensor, float]:
    eigenvalues = torch.linalg.eigvalsh(correlation)
    largest, smallest = float(eigenvalues[-1]), float(eigenvalues[0])
    if smallest <= 0 or largest / smallest > MAX_BLOCK_CONDITION:
        raise AnalysisError("singular_matrix", f"The canonical {what} correlation block is singular or ill-conditioned.")
    return torch.linalg.cholesky(correlation), largest / smallest


def _from_covariance(covariance, mean, n, dropped, xs, ys, extra):
    from .canon import _PERFECT, _result
    p, q, s = len(xs), len(ys), min(len(xs), len(ys))
    if n < p + q + 2:
        raise AnalysisError("insufficient_observations", f"Canonical correlation of {p} and {q} variables needs at least {p + q + 2} observations.")
    std = covariance.diagonal().sqrt()
    correlation = covariance / torch.outer(std, std)
    if not bool(torch.isfinite(correlation).all()):
        raise AnalysisError("numerical_failure", "CCA correlation matrix is non-finite; rescale the inputs.")
    lx, x_condition = _block(correlation[:p, :p], "x")
    ly, y_condition = _block(correlation[p:, p:], "y")
    left = torch.linalg.solve_triangular(lx, correlation[:p, p:], upper=False)
    whitened = torch.linalg.solve_triangular(ly, left.T, upper=False).T
    u, rho, vh = torch.linalg.svd(whitened)
    rho = rho[:s]
    if bool((rho > 1 + 1e-10).any()):
        raise AnalysisError("invalid_matrix", "Joint moments imply a canonical correlation above one; they are not repaired.")
    rho = rho.clamp(0, 1)
    rho = torch.where(rho.square() >= _PERFECT, torch.ones_like(rho), rho)
    a_std = torch.linalg.solve_triangular(lx.T, u[:, :s], upper=True)
    b_std = torch.linalg.solve_triangular(ly.T, vh.T[:, :s], upper=True)
    pivot = a_std.abs().argmax(0)
    sign = torch.sign(a_std[pivot, torch.arange(s)])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    a_std, b_std = a_std * sign, b_std * sign
    x_load, y_load = correlation[:p, :p] @ a_std, correlation[p:, p:] @ b_std
    extra.update({"algorithm": "joint-moment Cholesky whitening/SVD",
                  "x_block_condition": x_condition, "y_block_condition": y_condition,
                  "maximum_block_condition": MAX_BLOCK_CONDITION})
    result = _result(rho, a_std, b_std, x_load, y_load, std[:p], std[p:], mean,
                     n, dropped, xs, ys, extra)
    result["joint_covariance"] = c.frame(covariance, columns=[*xs, *ys], index=[*xs, *ys])
    if extra.get("training_sds_supplied") is False:
        # Coefficients in correlation units are already available separately;
        # unavailable raw units must not masquerade as original measurement units.
        result["raw_coefficients"].iloc[:, 1:] = float("nan")
        result["descriptives"]["std_dev"] = float("nan")
        result.attrs["notes"].append("Raw coefficients and training scales are unavailable for a correlation input without actual sds; standardized coefficients remain available.")
    return result


def _state(result):
    c.check_result(result, "canon", "canon_scores")
    xs, ys = _names(result.attrs.get("x"), result.attrs.get("y"))
    names = [*xs, *ys]
    geometry(len(names))
    if result.attrs.get("training_means_supplied") is False:
        raise AnalysisError("missing_training_moments", "Supply actual training means to project canonical scores.")
    if result.attrs.get("training_sds_supplied") is False:
        raise AnalysisError("missing_training_moments", "Supply actual training standard deviations for correlation-matrix raw score projection.")
    if "descriptives" not in result or "raw_coefficients" not in result:
        raise AnalysisError("invalid_result", "Canonical score state lacks saved training moments or coefficients.")
    table, stats = result["raw_coefficients"], result["descriptives"]
    labels = c.numbered("Canon", min(len(xs), len(ys)))
    if list(table.index) != names or list(stats.index) != names \
            or list(table.columns) != ["set", *labels] or list(stats.columns) != ["set", "mean", "std_dev"] \
            or list(table["set"]) != ["x"] * len(xs) + ["y"] * len(ys) \
            or list(stats["set"]) != ["x"] * len(xs) + ["y"] * len(ys):
        raise AnalysisError("invalid_result", "Canonical score state labels or set roles are invalid.")
    try:
        mean = torch.as_tensor(stats["mean"].to_numpy(dtype="float64").copy())
        std = torch.as_tensor(stats["std_dev"].to_numpy(dtype="float64").copy())
        coefficients = torch.as_tensor(table[labels].to_numpy(dtype="float64").copy())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result", "Canonical score state must contain finite numeric moments and coefficients.") from exc
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and torch.isfinite(coefficients).all()) \
            or not bool((std > 0).all()):
        raise AnalysisError("invalid_result", "Canonical score state must have finite moments/coefficients and positive scales.")
    return xs, ys, mean, coefficients, [*c.numbered("XCanon", len(labels)), *c.numbered("YCanon", len(labels))]


def project(result, data):
    xs, ys, mean, coefficients, columns = _state(result)
    names = [*xs, *ys]
    count = _row_count(data, names)
    selected_rows = min(65536, count or 65536) if isinstance(data, Dataset) else count
    geometry(len(names), rows=count or 0, selected_rows=selected_rows)
    if isinstance(data, Dataset):
        from .replay import score_source
        source = _bounded_source(data, names, len(names))
        return score_source(source, names, columns, lambda raw: project(result, raw),
                            procedure="canon_scores", options={"canonical_saved_basis": result.attrs.get("canonical_saved_basis")})
    frame = c.source(_selected(data, names))
    c.require_numeric(frame, names)
    keep = ~frame.isna().any(axis=1)
    centred = c.matrix(frame.loc[keep], names) - mean
    p = len(xs)
    scores = torch.cat((centred[:, :p] @ coefficients[:p], centred[:, p:] @ coefficients[p:]), dim=1)
    if not bool(torch.isfinite(scores).all()):
        raise AnalysisError("numerical_failure", "Canonical score projection exceeds float64 precision.")
    return c.aligned(scores, keep, columns)
