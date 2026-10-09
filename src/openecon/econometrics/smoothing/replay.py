"""Digest-checked saved smoothing transformations; no new-data fitting."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import openecon
import pandas as pd
import torch

from openecon.analysis import _coerce_frame
from openecon.econometrics.core import table
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes
from .common import DT, fail, finite

if TYPE_CHECKING:
    from openecon.dataset import Dataset


def design(data, state):
    from .fractional import fp_basis
    from .splines import bs_basis, rcs_basis

    columns = [torch.ones((len(data), 1), dtype=DT)]
    from openecon.analysis import _numeric

    for transform in state["transforms"]:
        x = _numeric(data[transform["column"]], transform["column"])
        kind = transform["kind"]
        if kind == "bs":
            block = bs_basis(x, transform["knots"], transform["boundary"], transform["degree"])[
                :, 1:
            ]
            if "center" in transform:
                block -= torch.tensor(transform["center"], dtype=DT)
        elif kind == "rcs":
            block = rcs_basis(x, transform["knots"])
        elif kind == "fp":
            block = fp_basis(x, transform["powers"], transform["scale"])
        elif kind == "linear":
            block = x[:, None]
        else:
            fail("invalid_state", "Unknown saved transformation.")
        columns.append(block)
    return finite(torch.cat(columns, 1))


def validated_state(result):
    """Verify the saved smoothing digest and bind its transformations to the model spec."""
    state = result.extra.get("smoothing_state")
    try:
        unsigned = {k: v for k, v in state.items() if k != "digest"}
        digest = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()
        matches = (
            digest == state["digest"]
            and state["schema"] == "smoothing-v1"
            and state["method"] == result.spec.estimator
            and state["columns"] == result.spec.predictors
        )
    except (TypeError, ValueError, KeyError, AttributeError):
        fail("invalid_state", "Saved smoothing state is malformed.")
    if not matches:
        fail("invalid_state", "Saved smoothing state digest/spec does not match.")
    return state


def smoothing_predict(
    result: ResultBundle, *, data, interval=False, alpha=0.05, missing="raise", max_work=2000000000
) -> openecon.DataFrame | Dataset:
    """Replay saved smoothing state; optional conditional OLS/penalized-estimator mean intervals.

    New-data knots, centering, powers, spans, penalties and categories are never fitted.
    LOESS, MARS and mixed kernels reject interval=True. Gaussian GAM intervals exclude
    smoothing bias and selection uncertainty; they target the penalized estimator expectation.
    B/RCS spline and FP/MFP Dataset queries return owned indexed Parquet output;
    other smoothing methods retain the explicit resident-only replay domain.
    """
    from openecon.dataset import Dataset

    if not isinstance(result, ResultBundle) or result.spec.estimator not in {
        "bspline_regress",
        "rcs_regress",
        "gam_gaussian",
        "fp_regress",
        "mfp_regress",
        "mars",
        "loess",
        "npreg_mixed",
    }:
        fail("invalid_result", "Supply a saved smoothing ResultBundle.")
    if (
        type(interval) is not bool
        or missing not in ("raise", "drop")
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < alpha < 1
    ):
        fail(
            "invalid_option", "Specify interval as bool, alpha in (0,1) and missing='raise'/'drop'."
        )
    if isinstance(max_work, bool) or not isinstance(max_work, int) or max_work < 1:
        fail("invalid_option", "max_work must be a positive integer.")
    state = validated_state(result)
    method = result.spec.estimator
    if interval and method in ("loess", "mars", "npreg_mixed"):
        fail(
            "unsupported_inference",
            "This adaptive/local prediction method does not provide pointwise inference.",
        )
    if isinstance(data, Dataset):
        from .streaming import predict_dataset

        return predict_dataset(
            result, data, interval=interval, alpha=alpha, missing=missing, max_work=max_work
        )
    raw = _coerce_frame(data)
    cols = state["columns"]
    if raw.columns.has_duplicates or any(c not in raw for c in cols):
        fail("missing_columns", "Replay needs every unique saved predictor column.")
    n = len(raw)
    k = len(state.get("coefficients", [])) or len(cols) + 3
    training = len(state.get("train_y", []))
    work = n * (k * k * 8 + training * (len(cols) + 3) * 16)
    if work > max_work:
        fail("work_limit", f"Smoothing replay plans {work} scalar work units.")
    plan_workspace(
        "smoothing replay selection, basis and training state",
        {
            "input_selection_and_basis": sum(
                int(raw[c].memory_usage(index=False, deep=True)) for c in cols
            )
            * 2
            + tensor_bytes((n, k), itemsize=96),
            "covariance_and_training": tensor_bytes((k, k), itemsize=32)
            + tensor_bytes((training, len(cols) + 5), itemsize=96),
        },
    )
    selected = raw.loc[:, cols].reset_index(drop=True)
    keep = ~selected.isna().any(axis=1)
    if not keep.all() and missing == "raise":
        fail(
            "missing_values", "New-data predictors contain missing values; specify missing='drop'."
        )
    positions = [int(i) for i in keep.to_numpy().nonzero()[0]]
    selected = selected.loc[keep]
    if not len(selected):
        fail("empty_sample", "No complete query rows remain.")
    from openecon.analysis import _numeric

    if method == "loess":
        from .local import loess_values

        prediction = loess_values(
            torch.tensor(state["train_x"], dtype=DT),
            torch.tensor(state["train_y"], dtype=DT),
            _numeric(selected[cols[0]], cols[0]),
            state["span"],
            state["degree"],
        )
    elif method == "npreg_mixed":
        from .local import encode, mixed_values

        q = encode(selected, cols, state["variable_types"], state["categories"])
        prediction = mixed_values(
            torch.tensor(state["train_x"], dtype=DT),
            torch.tensor(state["train_y"], dtype=DT),
            q,
            state["bandwidth"],
            cols,
            state["variable_types"],
            state["categories"],
            state["min_effective"],
        )
    else:
        if method == "mars":
            from .adaptive import hinge_basis

            matrix = hinge_basis(
                torch.stack([_numeric(selected[c], c) for c in cols], 1), state["hinges"]
            )
        else:
            matrix = design(selected, state)
        prediction = finite(matrix @ torch.tensor(state["coefficients"], dtype=DT))
    rows = {"row": positions, "mean": prediction.tolist()}
    if interval:
        cov = torch.tensor(state["covariance"], dtype=DT)
        var = finite((matrix @ cov * matrix).sum(1))
        if bool((var < -1e-10).any()):
            fail("invalid_covariance", "Saved prediction variance is negative.")
        se = var.clamp_min(0).sqrt()
        critical = critical_value(alpha, None if method == "gam_gaussian" else state["df_resid"])
        rows.update(
            {
                "std_error": se.tolist(),
                "ci_low": (prediction - critical * se).tolist(),
                "ci_high": (prediction + critical * se).tolist(),
            }
        )
    return table(
        pd.DataFrame(rows),
        title="Saved smoothing prediction",
        source_result_id=result.id,
        state_digest=state["digest"],
        inference="conditional approximate penalized-estimator mean; smoothing bias/selection excluded"
        if method == "gam_gaussian" and interval
        else "conditional selected-model mean"
        if interval
        else "prediction only",
        alpha=alpha if interval else None,
        original_rows=n,
        dropped_rows=n - len(positions),
    )
