"""Validated summary matrices; never fabricate observations or training moments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from . import common as c

MAX_VARIABLES = 256


def geometry(p: int, *, rows: int = 0, iterations: int = 1, parameters: int = 0) -> dict:
    if p > MAX_VARIABLES:
        raise AnalysisError("workspace_limit", f"Multivariate options support at most {MAX_VARIABLES} variables.")
    if iterations * p ** 3 > 2_000_000_000:
        raise AnalysisError("work_limit", "Requested matrix iterations exceed the multivariate work budget.")
    if parameters > 256 or (iterations + parameters) * parameters * p * p > 2_000_000_000:
        raise AnalysisError("work_limit", "Minimum-residual optimizer parameter geometry exceeds its work budget.")
    return plan_workspace("multivariate options", {"matrix_workspace": 64 * p * p * 8,
                          "selected_blocks": rows * (p + 1) * 64,
                          "optimizer_workspace": 64 * parameters * parameters * 8}).record()


@dataclass
class Summary:
    covariance: Tensor
    mean: Tensor
    std: Tensor
    names: list[str]
    n: int
    attrs: dict


def _vector(value: Any, names: list[str], what: str, *, positive: bool = False) -> Tensor:
    if (isinstance(value, Tensor) and value.is_complex()) or getattr(getattr(value, "dtype", None), "kind", None) == "c":
        raise AnalysisError("invalid_spec", f"{what} must be real-valued.")
    if isinstance(value, pd.Series):
        if value.index.has_duplicates or set(value.index) != set(names):
            raise AnalysisError("invalid_spec", f"{what} labels must match the matrix variables.")
        value = value.loc[names].tolist()
    try:
        result = torch.as_tensor(value, dtype=c.FLOAT)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_spec", f"{what} must be a finite vector.") from exc
    if result.device.type != "cpu":
        raise AnalysisError("unsupported_device", "Multivariate options require CPU float64 inputs.")
    if result.shape != (len(names),) or not bool(torch.isfinite(result).all()) \
            or (positive and not bool((result > 0).all())):
        raise AnalysisError("invalid_spec", f"{what} must have one finite {'positive ' if positive else ''}value per variable.")
    return result


def prepare(values: Any, *, n: int, columns: list[str] | None, matrix: str,
            means: Any = None, sds: Any = None) -> Summary:
    n = c.check_count(n, "n", minimum=2, maximum=2**53)
    c.check_choice(matrix, "matrix", ("correlation", "covariance"))
    if isinstance(values, pd.DataFrame):
        if values.index.has_duplicates or values.columns.has_duplicates \
                or list(values.index) != list(values.columns):
            raise AnalysisError("invalid_spec", "Matrix row and column labels must agree in order.")
        inferred = c.name_list(list(values.columns), "matrix labels", minimum=2)
        if columns is not None and list(columns) != inferred:
            raise AnalysisError("invalid_spec", "columns must agree with the labelled matrix order.")
        columns = inferred
        geometry(len(columns))
        c.require_numeric(values, columns)
        values = values.to_numpy(dtype="float64")
    names = c.name_list(columns, "columns", minimum=2)
    plan = geometry(len(names))
    # Check shape before converting nested/array input into a resident tensor.
    try:
        shape = tuple(values.shape) if hasattr(values, "shape") else (len(values), len(values[0]))
    except (TypeError, IndexError) as exc:
        raise AnalysisError("invalid_spec", "Supply a square summary matrix.") from exc
    if shape != (len(names), len(names)):
        raise AnalysisError("invalid_spec", "Matrix dimensions must match columns.")
    if isinstance(values, Tensor) and values.device.type != "cpu":
        raise AnalysisError("unsupported_device", "Summary matrices require CPU float64 inputs.")
    if (isinstance(values, Tensor) and values.is_complex()) or getattr(getattr(values, "dtype", None), "kind", None) == "c":
        raise AnalysisError("invalid_matrix", "Summary matrices must be real-valued.")
    try:
        target = torch.as_tensor(values, dtype=c.FLOAT).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_spec", "Supply a finite numeric square matrix.") from exc
    if not bool(torch.isfinite(target).all()):
        raise AnalysisError("non_finite_values", "Summary matrix values must be finite.")
    scale = max(float(target.abs().max()), 1e-300)
    if scale > 1e300 / (n - 1):
        raise AnalysisError("non_finite_values", "Summary matrix is too large for float64 moments; rescale inputs.")
    if float((target - target.T).abs().max()) > 1e-12 * scale:
        raise AnalysisError("invalid_matrix", "Summary matrix must be symmetric; it is not repaired.")
    target = (target + target.T) / 2
    if not bool((target.diagonal() > 0).all()):
        raise AnalysisError("constant_column", "Summary matrix diagonal must be positive.")
    if float(torch.linalg.eigvalsh(target).min()) < -1e-12 * scale * len(names):
        raise AnalysisError("invalid_matrix", "Summary matrix must be positive semidefinite; it is not repaired.")
    if matrix == "correlation":
        if not torch.allclose(target.diagonal(), torch.ones(len(names), dtype=c.FLOAT), atol=1e-12, rtol=0):
            raise AnalysisError("invalid_matrix", "Correlation matrix diagonal must be one.")
        std = torch.ones(len(names), dtype=c.FLOAT) if sds is None else _vector(sds, names, "sds", positive=True)
        covariance = target * torch.outer(std, std)
    else:
        std = target.diagonal().sqrt()
        if sds is not None and not torch.allclose(_vector(sds, names, "sds", positive=True), std, rtol=1e-10, atol=1e-12):
            raise AnalysisError("invalid_spec", "sds disagree with the covariance matrix diagonal.")
        covariance = target
    if not bool(torch.isfinite(covariance).all()) or float(covariance.abs().max()) > 1e300 / (n - 1):
        raise AnalysisError("non_finite_values", "Summary covariance is too large for float64 moments; rescale inputs.")
    mean = torch.full((len(names),), float("nan"), dtype=c.FLOAT) if means is None else _vector(means, names, "means")
    attrs = {"input_kind": "summary_matrix", "input_matrix": matrix, "resource_plan": plan,
             "training_means_supplied": means is not None,
             "training_sds_supplied": matrix == "covariance" or sds is not None,
             "precision": "float64", "device": "cpu", "inference": "descriptive; loading SE/CI not provided"}
    return Summary(covariance, mean, std, names, n, attrs)


@c.procedure
def pca_matrix(values: Any, *, n: int, columns: list[str] | None = None,
               matrix: str = "correlation", means: Any = None, sds: Any = None,
               components: int | None = None, mineigen: float | None = None):
    """PCA of a declared covariance/correlation matrix, with optional training moments.

    Example
    -------
    >>> pca_matrix([[1, .4], [.4, 1]], n=100, columns=['a', 'b']).attrs['components']
    2
    """
    from .pca import _MINEIGEN, _from_moments
    state = prepare(values, n=n, columns=columns, matrix=matrix, means=means, sds=sds)
    if components is not None:
        components = c.check_count(components, "components")
    cutoff = _MINEIGEN if mineigen is None else c.check_number(mineigen, "mineigen", minimum=0)
    return _from_moments(state.mean, state.covariance * (n - 1), n, state.names,
                         matrix, components, cutoff, 0, state.attrs)


@c.procedure
def factor_matrix(values: Any, *, n: int, columns: list[str] | None = None,
                  matrix: str = "correlation", means: Any = None, sds: Any = None, **options):
    """EFA of a summary matrix using the same extraction/rotation kernels as raw data.

    Example
    -------
    >>> factor_matrix([[1, .6, .5], [.6, 1, .4], [.5, .4, 1]],
    ...               n=100, columns=['a', 'b', 'c'], factors=1).attrs['factors']
    1
    """
    from .factor import factor
    state = prepare(values, n=n, columns=columns, matrix=matrix, means=means, sds=sds)
    iterations = options.get("max_iterations")
    if iterations is not None:
        iterations = c.check_count(iterations, "max_iterations")
    geometry(len(state.names), iterations=iterations or 1000)
    return factor(state, state.names, **options)
