"""Saved rank-one repeated-measures contrasts under independent Gaussian subjects."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from numbers import Integral

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import encode
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.stats import common as c
from openecon.engines.inference import critical_value
from openecon.resources import plan_workspace, tensor_bytes

MAX_CELLS = MAX_BETWEEN_PARAMETERS = 128


def _digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def save_geometry(model, transform, within, within_levels):
    """Small original-cell sufficient state; no subject rows or Kronecker covariance."""
    if model.k > MAX_BETWEEN_PARAMETERS or model.m > MAX_CELLS:
        return None
    plan = plan_workspace(
        "saved repeated-measures contrast geometry",
        {"original_cell_coefficients_covariance_bread":
            tensor_bytes((model.k, model.m), itemsize=32)
            + tensor_bytes((model.k, model.k), itemsize=32)
            + tensor_bytes((model.m, model.m), itemsize=48)},
    ).record()
    beta = model.beta.clone()
    beta[model.columns["Intercept"][0]] += model.shift
    cell_beta = beta @ transform.T
    cell_covariance = transform @ model.error @ transform.T / model.df_error
    order = []
    for term in model.terms:
        indices = model.columns[term.name]
        order.extend([term.name] if len(indices) == 1
                     else [f"{term.name}[{j + 1}]" for j in range(len(indices))])
    state = {
        "schema": "openecon.rm.contrast.v1", "df_resid": model.df_error,
        "n_subjects": model.n, "within": list(within),
        "cell_order": [list(cell) for cell in itertools.product(*(within_levels[name] for name in within))],
        "typed_cell_order": [encode(tuple(cell)) for cell in itertools.product(*(within_levels[name] for name in within))],
        "between_design_columns": order,
        "between_levels": model.levels,
        "between_codings": {name: value.tolist() for name, value in model.codings.items()},
        "coefficients": cell_beta.tolist(), "bread": model.full.xtx_inv.tolist(),
        "residual_cell_covariance": ((cell_covariance + cell_covariance.T) / 2).tolist(),
        "target": "rank-one between-design and original-cell contrast",
        "assumptions": "independent Gaussian subjects; common unrestricted within-subject covariance; fixed complete design",
        "resource_plan": plan,
    }
    try:
        state["sha256"] = _digest(state)
    except (ValueError, TypeError):
        raise AnalysisError("numerical_failure", "Saved RM contrast geometry must be finite JSON.") from None
    return state


def _vector(value, size, name):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise AnalysisError("unsupported_device", "RM contrasts require CPU inputs.")
    if (isinstance(value, torch.Tensor) and (value.is_complex() or value.dtype == torch.bool)) \
            or getattr(getattr(value, "dtype", None), "kind", None) in ("b", "c"):
        raise AnalysisError("invalid_contrast", f"{name} must contain finite real weights, excluding booleans.")
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != size:
        raise AnalysisError("invalid_contrast", f"{name} needs {size} ordered real weights.")
    try:
        if any(isinstance(item, (bool, complex)) for item in value):
            raise ValueError("non-real weights")
        vector = torch.as_tensor(value, dtype=torch.float64)
    except (ValueError, TypeError, RuntimeError):
        raise AnalysisError("invalid_contrast", f"{name} must contain finite real weights.") from None
    if vector.shape != (size,) or not bool(torch.isfinite(vector).all()) or float(vector.abs().max()) == 0:
        raise AnalysisError("invalid_contrast", f"{name} must be a nonzero finite real vector.")
    return vector


@resident_cpu
def rm_contrast(result: TableSet, contrast, *, between_contrast=None, null=0.0, alpha=0.05):
    """Test one saved original-cell × between-design repeated-measures contrast.

    Cell weights follow result.attrs['rm_contrast_state']['cell_order']; optional
    between weights follow the recorded sum-to-zero coded design-column order.
    Default between weights select the intercept, an equally weighted marginal
    mean across a full factorial between design, including unequal group sizes.
    A rank-one contrast has exact t inference under independent Gaussian subjects
    with one common unrestricted within-subject covariance. Sphericity is not
    required for this scalar target. No refit, weights, joint L/M test, population
    sampling interpretation or incomplete-design substitution is performed.
    """
    alpha, null = c.check_alpha(alpha), c.check_number(null, "null")
    if not isinstance(result, TableSet) or not result.attrs.get("within"):
        raise AnalysisError("invalid_result", "Supply a saved rm_anova TableSet with contrast geometry.")
    state = result.attrs.get("rm_contrast_state")
    if not isinstance(state, dict):
        raise AnalysisError("unsupported_saved_geometry", "This RM result has no supported saved contrast state; refit within the documented geometry.")
    try:
        unsigned = {key: value for key, value in state.items() if key != "sha256"}
        if state["schema"] != "openecon.rm.contrast.v1" or _digest(unsigned) != state["sha256"]:
            raise ValueError("state digest/schema mismatch")
        k, cells = len(state["between_design_columns"]), len(state["cell_order"])
        df, subjects = state["df_resid"], state["n_subjects"]
        if not 1 <= k <= MAX_BETWEEN_PARAMETERS or not 2 <= cells <= MAX_CELLS \
                or isinstance(df, bool) or not isinstance(df, Integral) or df <= 0 \
                or isinstance(subjects, bool) or not isinstance(subjects, Integral) or subjects - k != df \
                or state["within"] != result.attrs["within"] \
                or len(state["typed_cell_order"]) != cells \
                or any(not isinstance(row, list) or len(row) != len(state["within"]) for row in state["cell_order"]) \
                or any(not isinstance(name, str) or not name for name in state["between_design_columns"]) \
                or len(set(state["between_design_columns"])) != k:
            raise ValueError("state geometry mismatch")
        for name, rows, width in (("coefficients", k, cells), ("bread", k, k),
                                   ("residual_cell_covariance", cells, cells)):
            if not isinstance(state[name], list) or len(state[name]) != rows \
                    or any(not isinstance(row, list) or len(row) != width for row in state[name]):
                raise ValueError("state matrix dimensions")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise AnalysisError("invalid_state", "Saved RM contrast schema, digest or geometry is invalid.") from None
    plan = plan_workspace(
        "saved repeated-measures scalar contrast",
        {"coefficient_bread_covariance": tensor_bytes((k, cells), itemsize=24)
            + tensor_bytes((k, k), itemsize=32) + tensor_bytes((cells, cells), itemsize=48)},
    ).record()
    try:
        beta = torch.tensor(state["coefficients"], dtype=torch.float64)
        bread = torch.tensor(state["bread"], dtype=torch.float64)
        covariance = torch.tensor(state["residual_cell_covariance"], dtype=torch.float64)
    except (ValueError, TypeError, RuntimeError):
        raise AnalysisError("invalid_state", "Saved RM matrices must be real-valued.") from None
    if not all(bool(torch.isfinite(value).all()) for value in (beta, bread, covariance)):
        raise AnalysisError("invalid_state", "Saved RM matrices must be finite.")
    for matrix in (bread, covariance):
        scale = max(float(matrix.abs().max()), 1e-300)
        if float((matrix - matrix.T).abs().max()) > 1e-12 * scale \
                or float(torch.linalg.eigvalsh(matrix).min()) < -1e-12 * scale * len(matrix):
            raise AnalysisError("invalid_covariance", "Saved RM matrices must be symmetric positive semidefinite.")
    if float(torch.linalg.eigvalsh(bread).min()) <= 0:
        raise AnalysisError("invalid_covariance", "Saved between-design bread must have full rank.")
    cell = _vector(contrast, cells, "contrast")
    if between_contrast is None:
        between_contrast = [1.0] + [0.0] * (k - 1)
    group = _vector(between_contrast, k, "between_contrast")
    estimate = float(group @ beta @ cell)
    variance = float(group @ bread @ group) * float(cell @ covariance @ cell)
    if not 0 < variance < float("inf") or not math.isfinite(estimate):
        raise AnalysisError("zero_contrast_variance", "The saved scalar contrast needs finite positive variance.")
    standard_error = variance ** .5
    statistic = (estimate - null) / standard_error
    critical = critical_value(alpha, df)
    low, high = estimate - critical * standard_error, estimate + critical * standard_error
    if not all(math.isfinite(value) for value in (statistic, low, high)):
        raise AnalysisError("numerical_failure", "Scalar RM inference exceeds finite float64 arithmetic.")
    output = TableSet(
        {"contrast": table([[estimate, standard_error, null, statistic, df,
                             c.t_two_sided(statistic, df), low, high]],
                            columns=["estimate", "std_error", "null", "statistic", "df",
                                     "p_value", "ci_low", "ci_high"]),
         "cell_weights": table([[*labels, float(weight)] for labels, weight in zip(state["cell_order"], cell, strict=True)],
                               columns=[*state["within"], "contrast_weight"]),
         "between_weights": table(zip(state["between_design_columns"], group.tolist(), strict=True),
                                  columns=["design_column", "contrast_weight"])},
        title="Saved repeated-measures scalar contrast", procedure="rm_contrast",
        source_state_sha256=state["sha256"], alpha=alpha, precision="float64", device="cpu",
        inference="exact scalar t under independent Gaussian subjects with common unrestricted cell covariance",
        sphericity_required=False, resource_plan=plan, n_subjects=subjects,
    )
    from openecon.econometrics.summary_state import saved_summary

    return saved_summary(output)
