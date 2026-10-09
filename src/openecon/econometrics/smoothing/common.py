"""Shared exact Gaussian linear algebra and bounded replayable state."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, build_result, column_list, make_spec
from openecon.resources import tensor_bytes

DT = torch.float64


def fail(code, message):
    raise AnalysisError(code, message)


def call(name, data, y, x, missing="raise", alpha=0.05, **options):
    from openecon.analysis import fit

    return fit(
        make_spec(
            name,
            outcome=y,
            predictors=column_list(x, "x"),
            missing=missing,
            alpha=alpha,
            options=options,
        ),
        data=data,
    )


def prepare(spec, data, k, trials=1, quadratic=False):
    frame = ModelFrame(spec, data)
    if not 8 <= frame.n <= 100000 or not 1 <= len(spec.predictors) <= 20:
        fail("unsupported_dimensions", "Smoothing requires 8..100000 rows and 1..20 predictors.")
    work = trials * frame.n * k * k * 8 + (frame.n * frame.n * k * 8 if quadratic else 0)
    if work > frame.option("max_work"):
        fail(
            "work_limit", f"Smoothing plans {work} scalar work units; increase max_work explicitly."
        )
    frame.workspace_plan(
        "smoothing design, solves and persisted state",
        {
            "design_and_svd_copies": tensor_bytes((frame.n, k), itemsize=128),
            "gram_inverse_covariance_state": tensor_bytes((k, k), itemsize=128),
            "saved_training_local_buffers": tensor_bytes(
                (frame.n, len(spec.predictors) + 4), itemsize=128
            ),
        },
    )
    return frame


def finite(value):
    if not bool(torch.isfinite(value).all()):
        fail(
            "numerical_failure",
            "Transformation or fit exceeds finite float64 arithmetic; rescale explicitly.",
        )
    return value


def solve(x, y):
    finite(x)
    u, s, vt = torch.linalg.svd(x, full_matrices=False)
    if (
        len(y) <= x.shape[1]
        or len(s) < x.shape[1]
        or float(s[-1]) <= float(s[0]) * max(x.shape) * torch.finfo(DT).eps * 10
    ):
        fail(
            "rank_deficient",
            "Every declared transformed column must be identified with positive residual df.",
        )
    b = vt.T @ ((u.T @ y) / s)
    gram_inv = (vt.T / s.square()) @ vt
    rss = float((y - x @ b).square().sum())
    finite(b)
    if not math.isfinite(rss):
        fail("numerical_failure", "Residual sum of squares is not finite.")
    return b, rss, gram_inv


def seal(state):
    state = {**state, "schema": "smoothing-v1", "device": "cpu", "precision": "float64"}
    state["digest"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    return state


def result(frame, design, state, *, predictive=False, covariance=None, diagnostics=None):
    y = frame.numeric(frame.spec.outcome)
    b, rss, inv = solve(design, y)
    df = frame.n - design.shape[1]
    covariance = rss / df * inv if covariance is None else covariance
    state = seal(
        {
            **state,
            "columns": frame.spec.predictors,
            "coefficients": b.tolist(),
            "covariance": covariance.tolist(),
            "df_resid": df,
            "sample_positions": frame.positions,
            "method": frame.spec.estimator,
        }
    )
    note = (
        "Conditional on fixed transformation/model; selection uncertainty is excluded."
        if not predictive
        else "Adaptive prediction only; coefficient p values and confidence intervals are unavailable."
    )
    return build_result(
        frame,
        terms=[] if predictive else state["terms"],
        params=torch.empty(0, dtype=DT) if predictive else b,
        covariance=torch.empty((0, 0), dtype=DT) if predictive else covariance,
        use_t=not predictive,
        df_inference=None if predictive else df,
        df_resid=df,
        fitted=design @ b,
        metrics={"rss": rss, "training_mse": rss / frame.n},
        solver="float64 SVD with explicit rank gate",
        tests=diagnostics or {},
        inference={
            "distribution": "none" if predictive else "t",
            "available": not predictive,
            "conditioning": note,
            "correction": "iid Gaussian OLS",
        },
        warnings=[note],
        extra={"smoothing_state": state, "notes": [note]},
    )


def prediction_result(frame, fitted, state, diagnostics):
    note = "Conditional mean prediction only; no coefficient or pointwise inference is available."
    state = seal(
        {
            **state,
            "columns": frame.spec.predictors,
            "method": frame.spec.estimator,
            "sample_positions": frame.positions,
        }
    )
    return build_result(
        frame,
        terms=[],
        params=torch.empty(0, dtype=DT),
        covariance=torch.empty((0, 0), dtype=DT),
        fitted=finite(fitted),
        use_t=False,
        solver="bounded direct float64 smoother",
        tests=diagnostics,
        metrics={
            "training_mse": float((frame.numeric(frame.spec.outcome) - fitted).square().mean())
        },
        inference={"available": False, "distribution": "none", "correction": note},
        extra={"smoothing_state": state, "notes": [note]},
        warnings=[note],
    )
