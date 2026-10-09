"""Analytic saved-smoother derivatives and fixed-covariate linear functionals.

No query-data tuning or fitting. Local Taylor slopes are distinct from
differentiating the moving-neighborhood LOESS prediction operator.
"""

from __future__ import annotations

import math

import openecon
import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.econometrics.core import table
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, tensor_bytes
from .common import DT, fail, finite
from .replay import design, validated_state

LINEAR = {"bspline_regress", "rcs_regress", "fp_regress", "mfp_regress", "gam_gaussian"}
LOCAL = {"loess", "npreg_mixed", "mars"}


def _prepare(result, data, missing, max_work, interval, alpha, reference=None):
    from openecon.dataset import Dataset

    if not isinstance(result, ResultBundle) or result.spec.estimator not in LINEAR | LOCAL:
        fail("invalid_result", "Supply a saved smoothing ResultBundle.")
    if isinstance(data, Dataset) or isinstance(reference, Dataset):
        fail("streaming_unsupported", "Smoothing postestimation requires resident query tables.")
    if (
        type(interval) is not bool
        or missing not in ("raise", "drop")
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < alpha < 1
        or isinstance(max_work, bool)
        or not isinstance(max_work, int)
        or max_work < 1
    ):
        fail(
            "invalid_option",
            "Specify bool interval, alpha in (0,1), raise/drop missing and positive integer max_work.",
        )
    if interval and result.spec.estimator in LOCAL:
        fail(
            "unsupported_inference",
            "Adaptive/local methods provide descriptive functionals without intervals.",
        )
    state = validated_state(result)
    cols = state["columns"]
    frames = [_coerce_frame(data)]
    if reference is not None:
        frames.append(_coerce_frame(reference))
    n = len(frames[0])
    if not 1 <= n <= 100000 or any(len(f) != n for f in frames):
        fail(
            "invalid_query",
            "Query tables need 1..100000 rows and paired tables must have equal lengths.",
        )
    if any(f.columns.has_duplicates or any(c not in f for c in cols) for f in frames):
        fail("missing_columns", "Query tables need every unique saved predictor column.")
    k = len(state.get("coefficients", [])) or len(cols) + 3
    training = len(state.get("train_y", []))
    if not 1 <= k <= 1000 or training > 100000:
        fail("invalid_state", "Saved smoother dimensions exceed its supported bounds.")
    work = len(frames) * n * (k * k * 16 + len(cols) * 64 + training * (len(cols) + 5) * 32) + (
        16 * k**3 if interval else 0
    )
    if work > max_work:
        fail("work_limit", f"Smoothing functionals plan {work} scalar work units.")
    plan_workspace(
        "smoothing derivatives and paired query selection",
        {
            "selection_basis_jacobian": sum(
                int(f[c].memory_usage(index=False, deep=True)) for f in frames for c in cols
            )
            * 3
            + tensor_bytes((len(frames) * n, k + len(cols)), itemsize=160),
            "covariance_training_buffers": tensor_bytes((k, k), itemsize=64)
            + tensor_bytes((training, len(cols) + 5), itemsize=128),
        },
    )
    selected = [f.loc[:, cols].reset_index(drop=True) for f in frames]
    keep = torch.ones(n, dtype=torch.bool)
    for f in selected:
        keep &= torch.tensor((~f.isna().any(axis=1)).to_numpy(), dtype=torch.bool)
    if not bool(keep.all()) and missing == "raise":
        fail("missing_values", "Query predictors contain missing values; specify missing='drop'.")
    positions = keep.nonzero().flatten().tolist()
    if not positions:
        fail("empty_sample", "No complete paired query rows remain.")
    return state, [f.iloc[positions] for f in selected], positions, n


def _order(state, variable, order):
    if not isinstance(variable, str) or variable not in state["columns"]:
        fail("invalid_variable", "variable must name a saved predictor.")
    if type(order) is not int or order not in (1, 2):
        fail("invalid_order", "Only first and second pure partial derivatives are supported.")


def _bs_derivative(x, record, order):
    degree = record["degree"]
    if order > degree:
        fail("invalid_order", "Derivative order must not exceed the saved spline degree.")
    lo, hi = record["boundary"]
    if bool(((x < lo) | (x > hi)).any()):
        fail("outside_support", "Spline derivatives require the saved boundary.")
    # Simple interior knots have continuity degree-1. Do not choose a side
    # silently when the requested derivative is discontinuous.
    if order == degree and any(bool((x == knot).any()) for knot in record["knots"]):
        fail("nondifferentiable", "The requested derivative is discontinuous at an interior knot.")
    t = [lo] * (degree + 1) + record["knots"] + [hi] * (degree + 1)
    b = torch.stack([((x >= t[j]) & (x < t[j + 1])).to(DT) for j in range(len(t) - 1)], 1)
    # Interior one-sided limit at the upper boundary, for every recursion level.
    b[x == hi, len(t) - degree - 2] = 1.0
    derivatives = [b, torch.zeros_like(b), torch.zeros_like(b)]
    for d in range(1, degree + 1):
        previous = derivatives
        derivatives = []
        for r in range(3):
            pieces = []
            for j in range(len(t) - d - 1):
                left, right = t[j + d] - t[j], t[j + d + 1] - t[j + 1]
                if r == 0:
                    a = (x - t[j]) / left * previous[0][:, j] if left else torch.zeros_like(x)
                    c = (
                        (t[j + d + 1] - x) / right * previous[0][:, j + 1]
                        if right
                        else torch.zeros_like(x)
                    )
                    pieces.append(a + c)
                else:
                    a = d / left * previous[r - 1][:, j] if left else torch.zeros_like(x)
                    c = d / right * previous[r - 1][:, j + 1] if right else torch.zeros_like(x)
                    pieces.append(a - c)
            derivatives.append(torch.stack(pieces, 1))
    return finite(derivatives[order][:, 1:])


def _fp_derivative(x, record, order):
    z = x / record["scale"]
    if bool((z <= 0).any()):
        fail("invalid_fp_domain", "Fractional-polynomial queries must be strictly positive.")
    powers = record["powers"]
    parts = []
    for j, p in enumerate(powers):
        repeated = j > 0 and p == powers[j - 1]
        if p == 0:
            value = (
                (2 * z.log() / z if repeated else 1 / z)
                if order == 1
                else (2 * (1 - z.log()) / z.square() if repeated else -1 / z.square())
            )
        elif repeated:
            value = (
                z.pow(p - order) * (p * z.log() + 1)
                if order == 1
                else (z.pow(p - 2) * (p * (p - 1) * z.log() + 2 * p - 1))
            )
        else:
            value = p * z.pow(p - 1) if order == 1 else p * (p - 1) * z.pow(p - 2)
        parts.append(value / record["scale"] ** order)
    return finite(torch.stack(parts, 1)) if parts else torch.empty((len(x), 0), dtype=DT)


def _linear_derivative(data, state, variable, order):
    from .splines import bs_basis, rcs_basis
    from .fractional import fp_basis

    blocks = [torch.zeros((len(data), 1), dtype=DT)]
    for record in state["transforms"]:
        x = _numeric(data[record["column"]], record["column"])
        kind = record["kind"]
        if kind == "bs":
            block = bs_basis(x, record["knots"], record["boundary"], record["degree"])[:, 1:]
        elif kind == "rcs":
            block = rcs_basis(x, record["knots"])
        elif kind == "fp":
            block = fp_basis(x, record["powers"], record["scale"])
        elif kind == "linear":
            block = x[:, None]
        else:
            fail("invalid_state", "Unknown saved transformation.")
        if record["column"] != variable:
            block = torch.zeros_like(block)
        elif kind == "bs":
            block = _bs_derivative(x, record, order)
        elif kind == "fp":
            block = _fp_derivative(x, record, order)
        elif kind == "linear":
            block = torch.ones_like(block) if order == 1 else torch.zeros_like(block)
        else:
            knots = record["knots"]
            width = knots[-1] - knots[0]
            z = (x - knots[0]) / width
            t = [(v - knots[0]) / width for v in knots]
            last, penult = t[-1], t[-2]
            parts = [torch.ones_like(x) / width if order == 1 else torch.zeros_like(x)]
            factor = 3 if order == 1 else 6
            for a in t[:-2]:
                parts.append(
                    factor
                    / width**order
                    * (
                        (z - a).clamp_min(0).pow(3 - order)
                        - (z - penult).clamp_min(0).pow(3 - order) * (last - a) / (last - penult)
                        + (z - last).clamp_min(0).pow(3 - order) * (penult - a) / (last - penult)
                    )
                )
            block = torch.stack(parts, 1)
        blocks.append(block)
    return finite(torch.cat(blocks, 1))


def _mars_basis(data, state, variable):
    x = torch.stack([_numeric(data[c], c) for c in state["columns"]], 1)
    col = state["columns"].index(variable)
    blocks = []
    for factors in state["hinges"]:
        derivative = torch.zeros(len(x), dtype=DT)
        for j, factor in enumerate(factors):
            if factor["column_index"] != col:
                continue
            # Check the product of all remaining factors: a zero multiplier
            # makes this hinge derivative identically zero in the varied coordinate.
            multiplier = torch.ones(len(x), dtype=DT)
            for k, other in enumerate(factors):
                if k != j:
                    multiplier *= (
                        (x[:, other["column_index"]] - other["knot"]) * other["sign"]
                    ).clamp_min(0)
            distance = (x[:, col] - factor["knot"]) * factor["sign"]
            if bool(((distance == 0) & (multiplier != 0)).any()):
                fail(
                    "nondifferentiable",
                    "An active retained MARS hinge has no unique derivative at its knot.",
                )
            derivative += factor["sign"] * (distance > 0).to(DT) * multiplier
        blocks.append(derivative)
    return finite(torch.stack(blocks, 1))


def _loess_derivative(data, state, variable, order):
    from .local import loess_coefficients

    if order > state["degree"]:
        fail("invalid_order", "Derivative order exceeds the saved local-polynomial degree.")
    x, y = torch.tensor(state["train_x"], dtype=DT), torch.tensor(state["train_y"], dtype=DT)
    queries = _numeric(data[variable], variable)
    if bool(((queries < x.min()) | (queries > x.max())).any()):
        fail("outside_support", "Local derivatives require the saved training range.")
    parts = []
    for q in queries:
        b, radius = loess_coefficients(x, y, q, state["span"], state["degree"])
        parts.append(math.factorial(order) * b[order] / radius**order)
    return finite(torch.stack(parts))


def _kernel_derivative(data, state, variable, order):
    from .local import encode

    columns, types, categories = state["columns"], state["variable_types"], state["categories"]
    if types[variable] != "c":
        fail(
            "unsupported_derivative",
            "Category predictors have discrete contrasts, not numeric derivatives.",
        )
    q = encode(data, columns, types, categories)
    x, y = torch.tensor(state["train_x"], dtype=DT), torch.tensor(state["train_y"], dtype=DT)
    bw = state["bandwidth"]
    target = columns.index(variable)
    values = []
    for query in q:
        logw = torch.zeros(len(x), dtype=DT)
        for j, col in enumerate(columns):
            h = bw[j]
            if types[col] == "c":
                if float(query[j]) < float(x[:, j].min()) or float(query[j]) > float(x[:, j].max()):
                    fail(
                        "outside_support",
                        "Numeric kernel queries require the saved training ranges.",
                    )
                logw -= 0.5 * ((x[:, j] - query[j]) / h).square()
            elif types[col] == "u":
                logw += torch.where(
                    x[:, j] == query[j],
                    torch.tensor(1 - h, dtype=DT),
                    torch.tensor(h / (len(categories[col]) - 1), dtype=DT),
                ).log()
            else:
                distance = (x[:, j] - query[j]).abs()
                logw += torch.where(
                    distance == 0, torch.tensor(1 - h, dtype=DT), 0.5 * (1 - h) * h**distance
                ).log()
        if not bool(torch.isfinite(logw).any()):
            fail("no_local_support", "No training rows have positive product-kernel weight.")
        w = torch.exp(logw - logw.max())
        w /= w.sum()
        if float(1 / w.square().sum()) + 1e-12 < state["min_effective"]:
            fail("no_local_support", "Effective kernel sample is below the saved threshold.")
        score = (x[:, target] - query[target]) / bw[target] ** 2
        center = score - w @ score
        derivative = w @ (y * center)
        if order == 2:
            # Derivative of normalized Gaussian weights: the common -1/h^2
            # term cancels, leaving the centered-score square minus its mean.
            derivative = w @ (y * (center.square() - w @ center.square()))
        values.append(derivative)
    return finite(torch.stack(values))


def _functional(data, state, variable=None, order=1):
    method = state["method"]
    if variable is not None:
        _order(state, variable, order)
        if method in LINEAR:
            matrix = _linear_derivative(data, state, variable, order)
        elif method == "mars":
            if order != 1:
                fail(
                    "unsupported_derivative", "MARS supports first partials off active knots only."
                )
            matrix = _mars_basis(data, state, variable)
        elif method == "loess":
            return _loess_derivative(data, state, variable, order), None
        else:
            return _kernel_derivative(data, state, variable, order), None
    elif method in LINEAR:
        matrix = design(data, state)
    elif method == "mars":
        from .adaptive import hinge_basis

        matrix = hinge_basis(
            torch.stack([_numeric(data[c], c) for c in state["columns"]], 1), state["hinges"]
        )
    elif method == "loess":
        from .local import loess_values

        return loess_values(
            torch.tensor(state["train_x"], dtype=DT),
            torch.tensor(state["train_y"], dtype=DT),
            _numeric(data[state["columns"][0]], state["columns"][0]),
            state["span"],
            state["degree"],
        ), None
    else:
        from .local import encode, mixed_values

        q = encode(data, state["columns"], state["variable_types"], state["categories"])
        return mixed_values(
            torch.tensor(state["train_x"], dtype=DT),
            torch.tensor(state["train_y"], dtype=DT),
            q,
            state["bandwidth"],
            state["columns"],
            state["variable_types"],
            state["categories"],
            state["min_effective"],
        ), None
    b = torch.tensor(state["coefficients"], dtype=DT)
    if matrix.shape[1] != len(b):
        fail("invalid_state", "Saved basis dimensions do not match coefficients.")
    return finite(matrix @ b), matrix


def _finish(result, state, estimate, matrix, rows, n, interval, alpha, **attrs):
    values = {"row": rows, "estimate": finite(estimate).tolist()}
    inference = "descriptive fixed-training functional; inference unavailable"
    covariance = None
    if interval:
        k = matrix.shape[1]
        saved = state.get("covariance")
        if (
            not isinstance(saved, list)
            or len(saved) != k
            or any(not isinstance(r, list) or len(r) != k for r in saved)
        ):
            fail(
                "invalid_covariance",
                "Saved full covariance dimensions do not match the functional basis.",
            )
        covariance = finite(torch.tensor(saved, dtype=DT))
        if not torch.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-12):
            fail("invalid_covariance", "Saved covariance is not symmetric.")
        # A covariance PSD audit is cubic and is explicitly included in the work budget below.
        eig = torch.linalg.eigvalsh(covariance)
        if float(eig.min()) < -1e-10 * max(torch.finfo(DT).tiny, float(eig.abs().max())):
            fail("invalid_covariance", "Saved covariance is not positive semidefinite.")
        variance = finite((matrix @ covariance * matrix).sum(1))
        if bool((variance < -1e-10).any()):
            fail("invalid_covariance", "Functional variance is negative.")
        se = variance.clamp_min(0).sqrt()
        gam = state["method"] == "gam_gaussian"
        df = None if gam else state["df_resid"]
        if df is not None and (isinstance(df, bool) or not isinstance(df, int) or df <= 0):
            fail(
                "invalid_state", "Conditional OLS inference requires positive integer residual df."
            )
        critical = critical_value(alpha, df)
        values.update(
            std_error=se.tolist(),
            ci_low=(estimate - critical * se).tolist(),
            ci_high=(estimate + critical * se).tolist(),
        )
        if not gam:
            statistic = estimate / torch.where(se > 0, se, torch.ones_like(se))
            p = two_sided_p_values(statistic, df)
            values.update(
                df=[df] * len(rows),
                statistic=[float(t) if s > 0 else None for t, s in zip(statistic, se, strict=True)],
                p_value=[float(v) if s > 0 else None for v, s in zip(p, se, strict=True)],
            )
        inference = (
            "approximate normal around penalized-estimator expectation; smoothing bias and tuning uncertainty excluded"
            if gam
            else "Gaussian iid conditional selected-model t; fixed query covariates; selection uncertainty excluded"
        )
    return table(
        pd.DataFrame(values),
        title="Saved smoothing functional",
        source_result_id=result.id,
        source_method=state["method"],
        state_digest=state["digest"],
        original_rows=n,
        retained_positions=attrs.pop("retained_positions", rows),
        dropped_rows=n - len(attrs.get("sample_positions", rows)),
        alpha=alpha if interval else None,
        inference=inference,
        linear_functionals=matrix.tolist() if interval else None,
        coefficient_covariance=covariance.tolist() if interval else None,
        joint_covariance="L V L' in displayed row order" if interval else None,
        **attrs,
    )


def _derivative(result, data, variable, order, interval, alpha, missing, max_work, methods):
    state, frames, rows, n = _prepare(result, data, missing, max_work, interval, alpha)
    if state["method"] not in methods:
        fail("invalid_result", "This derivative helper does not accept the supplied method.")
    value, basis = _functional(frames[0], state, variable, order)
    return _finish(
        result,
        state,
        value,
        basis,
        rows,
        n,
        interval,
        alpha,
        variable=variable,
        order=order,
        target="local Taylor polynomial coefficient derivative"
        if state["method"] == "loess"
        else "saved fitted function partial derivative",
        boundary="interior one-sided at saved endpoints; no extrapolation"
        if state["method"] in {"bspline_regress", "gam_gaussian"}
        else None,
    )


def spline_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=True,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """First/second saved B-spline or restricted cubic partials with conditional full-covariance t inference."""
    return _derivative(
        result,
        data,
        variable,
        order,
        interval,
        alpha,
        missing,
        max_work,
        {"bspline_regress", "rcs_regress"},
    )


def fp_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=True,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """Analytic scaled FP/repeated-log or linear-adjustment partials; MFP inference conditional on selection."""
    return _derivative(
        result,
        data,
        variable,
        order,
        interval,
        alpha,
        missing,
        max_work,
        {"fp_regress", "mfp_regress"},
    )


def gam_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=True,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """Gaussian additive spline partials with approximate frequentist estimator-expectation intervals, excluding smoothing bias."""
    return _derivative(
        result, data, variable, order, interval, alpha, missing, max_work, {"gam_gaussian"}
    )


def mars_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=False,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """First piecewise MARS partials with interaction product rule; reject active knot ties and inference."""
    return _derivative(result, data, variable, order, interval, alpha, missing, max_work, {"mars"})


def loess_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=False,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """Saved-span local Taylor derivatives r! beta_r/radius^r; these do not differentiate moving query weights/neighbors."""
    return _derivative(result, data, variable, order, interval, alpha, missing, max_work, {"loess"})


def kernel_derivative(
    result: ResultBundle,
    *,
    data,
    variable: str,
    order=1,
    interval=False,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """First/second continuous partials of saved mixed product-kernel means; fixed bandwidth, no categorical derivatives or CI."""
    return _derivative(
        result, data, variable, order, interval, alpha, missing, max_work, {"npreg_mixed"}
    )


def _average_weights(weights, n, rows):
    if weights is None:
        return torch.full((len(rows),), 1 / len(rows), dtype=DT)
    if (
        not isinstance(weights, (list, tuple))
        or len(weights) != n
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in weights
        )
    ):
        fail(
            "invalid_weights",
            "Averaging weights must be an original-row-length list of finite nonnegative numbers.",
        )
    # Normalize by the maximum first, avoiding overflow of a finite supplied sum.
    w = torch.tensor([weights[i] for i in rows], dtype=DT)
    if float(w.max()) <= 0:
        fail("invalid_weights", "Retained query averaging weights need positive mass.")
    w /= w.max()
    return w / w.sum()


def smoothing_margins(
    result: ResultBundle,
    *,
    data,
    variable: str | None = None,
    averaging_weights=None,
    interval=False,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """Average response or first partial over fixed covariates; aggregate full basis before covariance.

    B/RCS spline and FP/MFP Dataset queries use bounded global reduction.
    Dataset averaging_weights is a column name; resident averaging weights
    remain an original-row-length list. Query rows never tune the saved model.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        from .streaming import margins_dataset

        return margins_dataset(
            result,
            data,
            variable=variable,
            averaging_weights=averaging_weights,
            interval=interval,
            alpha=alpha,
            missing=missing,
            max_work=max_work,
        )
    state, frames, rows, n = _prepare(result, data, missing, max_work, interval, alpha)
    value, basis = _functional(frames[0], state, variable)
    w = _average_weights(averaging_weights, n, rows)
    aggregate = (w @ value).reshape(1)
    contrast = (w @ basis).reshape(1, -1) if basis is not None else None
    return _finish(
        result,
        state,
        aggregate,
        contrast,
        [0],
        n,
        interval,
        alpha,
        variable=variable,
        target="fixed-covariate average local Taylor derivative"
        if variable and state["method"] == "loess"
        else "fixed-covariate average partial"
        if variable
        else "fixed-covariate average response",
        retained_positions=rows,
        sample_positions=rows,
        normalized_averaging_weights=w.tolist(),
    )


def smoothing_contrast(
    result: ResultBundle,
    *,
    data,
    reference,
    average=False,
    averaging_weights=None,
    interval=False,
    alpha=0.05,
    missing="raise",
    max_work=2000000000,
) -> openecon.DataFrame:
    """Paired positional counterfactual response changes; synchronized missing drop and full cross-profile covariance."""
    if (
        reference is None
        or type(average) is not bool
        or (averaging_weights is not None and not average)
    ):
        fail(
            "invalid_option",
            "Supply reference, bool average, and averaging weights only for average=True.",
        )
    state, frames, rows, n = _prepare(result, data, missing, max_work, interval, alpha, reference)
    value, basis = _functional(frames[0], state)
    baseline, base_basis = _functional(frames[1], state)
    value = finite(value - baseline)
    basis = finite(basis - base_basis) if basis is not None else None
    weights = None
    if average:
        weights = _average_weights(averaging_weights, n, rows)
        value = (weights @ value).reshape(1)
        basis = (weights @ basis).reshape(1, -1) if basis is not None else None
    return _finish(
        result,
        state,
        value,
        basis,
        [0] if average else rows,
        n,
        interval,
        alpha,
        target="paired fixed-covariate average response change"
        if average
        else "paired fixed-covariate response change",
        pairing="original positional rows; data minus reference",
        retained_positions=rows,
        sample_positions=rows,
        normalized_averaging_weights=weights.tolist() if weights is not None else None,
    )
