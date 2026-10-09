"""Declared-cell independent-Poisson loglinear fits, not row-level GLM aliases.

Hierarchical IPF fits generating margins on explicit structural-zero support.
General ML accepts a caller-declared full-rank cell design, with fixed offsets.
Both retain the entire cell support and observed information for restoration.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
import hashlib
import json
import math
from numbers import Integral, Real

import torch

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import chi2_sf, normal_isf, normal_sf
from openecon.resources import plan_workspace, workspace_budget_bytes

_F = torch.float64
_SCHEMA = "openecon.loglinear.v1"


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        _error(f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def _number(value, name, low, high):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not low <= value <= high
    ):
        _error(f"{name} must be finite in [{low}, {high}].")
    return float(value)


def _names(value, name, low=1, high=128):
    if (
        not isinstance(value, (list, tuple))
        or not low <= len(value) <= high
        or any(not isinstance(v, str) or not v or len(v) > 256 for v in value)
        or len(set(value)) != len(value)
    ):
        _error(f"{name} must contain {low}..{high} unique nonempty names.")
    return list(value)


def _key(value):
    if hasattr(value, "item"):
        value = value.item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (str, int, float))
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        _error("Cell categories must be finite non-boolean strings or numbers.")
    if isinstance(value, str) and len(value) > 256:
        _error("Cell categories must be <=256 characters.")
    return (type(value).__name__, value)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _options(max_iter, tol, level, max_work, max_bytes):
    return (
        _integer(max_iter, "max_iter", 1, 2000),
        _number(tol, "tol", 1e-12, 1e-4),
        _number(level, "level", 0.5, 0.999),
        _integer(max_work, "max_work", 1, 2_000_000_000),
        min(_integer(max_bytes, "max_bytes", 1, 512 * 1024**2), workspace_budget_bytes()),
    )


def _grid(data, dimensions, count, levels, structural, offset, budget):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        _error(
            "Loglinear cell input must be resident; Dataset collection is unsupported.",
            "streaming_unsupported",
        )
    dimensions = _names(dimensions, "dimensions", 2, 6)
    if not isinstance(levels, dict) or set(levels) != set(dimensions):
        _error("Declare every dimension's complete levels explicitly.")
    keys = {}
    values = {}
    for dim in dimensions:
        seq = levels[dim]
        if not isinstance(seq, (list, tuple)) or not 2 <= len(seq) <= 16:
            _error("Each dimension needs 2..16 declared categories.")
        keys[dim] = [_key(v) for v in seq]
        if len(set(keys[dim])) != len(seq):
            _error("Declared category levels must be unique by type and value.")
        values[dim] = [k[1] for k in keys[dim]]
    shape = [len(values[d]) for d in dimensions]
    cells = math.prod(shape)
    if cells > 4096:
        _error("Complete grid exceeds 4096 cells.", "resource_limit")
    # Admit selection/indexing before making our resident selection copies.
    plan_workspace(
        "loglinear grid admission",
        {"cell indexing and input": cells * (512 + 32 * len(dimensions))},
        budget_bytes=budget,
    )
    required = [
        *dimensions,
        count,
        *([structural] if structural else []),
        *([offset] if offset else []),
    ]
    if (
        not isinstance(count, str)
        or structural is not None
        and not isinstance(structural, str)
        or offset is not None
        and not isinstance(offset, str)
    ):
        _error("count, structural and offset must name columns.")
    if isinstance(data, Mapping):
        if any(c not in data for c in required):
            _error("Required cell columns are missing.")
        data = {c: data[c] for c in required}
    elif isinstance(data, list):
        if len(data) != cells or any(not isinstance(row, Mapping) for row in data):
            _error("Cell records must have exactly one mapping per declared cell.")
        data = [{c: row.get(c) for c in required} for row in data]
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates or len(frame) != cells:
        _error("Supply unique columns and exactly one row for every declared cell.")
    if any(c not in frame for c in required) or len(set(required)) != len(required):
        _error("Required cell columns must exist and have distinct roles.")
    if frame[required].isna().any().any():
        _error(
            "Missing cells/categories/counts/offsets are unsupported; no implicit zero or row drop.",
            "missing_values",
        )
    rows = {}
    for i, row in enumerate(frame[dimensions].itertuples(index=False, name=None)):
        key = tuple(_key(v) for v in row)
        if key in rows or any(key[j] not in keys[d] for j, d in enumerate(dimensions)):
            _error("Cells must be unique and use exactly the declared typed categories.")
        rows[key] = i
    ordered = list(product(*(keys[d] for d in dimensions)))
    if set(rows) != set(ordered):
        _error("Incomplete grid; include sampling-zero and structural-zero rows explicitly.")
    positions = [rows[k] for k in ordered]
    y_values = frame[count].iloc[positions].tolist()
    if any(
        isinstance(v, bool)
        or not isinstance(v, Real)
        or not math.isfinite(v)
        or v < 0
        or v > 10_000_000
        or int(v) != v
        for v in y_values
    ):
        _error("Counts must be finite nonnegative integers <=10,000,000.")
    if not 0 < sum(y_values) <= 1_000_000_000:
        _error("Total count must lie in (0, 1,000,000,000].")
    s = frame[structural].iloc[positions].tolist() if structural else [False] * cells
    if any(not isinstance(v, bool) for v in s):
        _error("Structural-zero column must contain explicit booleans.")
    if any(y and z for y, z in zip(y_values, s)) or all(s):
        _error("Structural-zero cells must have count zero and leave active support.")
    off = frame[offset].iloc[positions].tolist() if offset else [0.0] * cells
    off = [_number(v, "log offset", -20, 20) for v in off]
    grid = {
        "dimensions": dimensions,
        "levels": values,
        "shape": shape,
        "cells": [[k[1] for k in row] for row in ordered],
        "counts": y_values,
        "structural": s,
        "offset": off,
        "positions": positions,
        "input_rows": len(frame),
        "missing": "raise",
        "sampling": "independent_poisson_cells",
    }
    signature = _digest(
        {
            k: grid[k]
            for k in ("dimensions", "levels", "cells", "counts", "structural", "offset", "sampling")
        }
    )
    return grid, signature


def _admit(n, p, iterations, work, budget, extra=0, full_rows=None):
    if p < 1 or p > 128:
        _error("Cell design must have 1..128 parameters.", "resource_limit")
    projected = n * p * p * (iterations + 8) + (iterations + 4) * p**3 + extra
    if projected > work:
        _error(
            f"Declared fit requires {projected} structural work units; max_work={work}.",
            "resource_limit",
        )
    plan = plan_workspace(
        "loglinear likelihood and information",
        {
            "cell input/design/result copies": 8 * n * (8 * p + 24),
            "full declared design conversion": 8 * (n if full_rows is None else full_rows) * p,
            "joint information and linear solves": 8 * 12 * p * p,
            "convergence trace": 8 * 6 * iterations,
        },
        budget_bytes=budget,
    )
    return {**plan.record(), "estimated_work": projected, "max_work": work}


def _likelihood(y, eta):
    return float((y * eta - eta.exp() - torch.lgamma(y + 1)).sum())


def _information(x, mu):
    info = x.T @ (mu[:, None] * x)
    eigen = torch.linalg.eigvalsh(info)
    if not torch.isfinite(eigen).all() or float(eigen[0]) <= max(1e-10, float(eigen[-1]) * 1e-12):
        _error(
            "Observed information is singular or ill-conditioned; no fabricated covariance.",
            "unidentified_model",
        )
    return info, torch.linalg.inv(info)


def _fit_newton(x, y, offset, max_iter, tol):
    beta = torch.zeros(x.shape[1], dtype=_F)
    ones = torch.nonzero((x - 1).abs().max(0).values < 1e-12).flatten()
    if len(ones):
        beta[int(ones[0])] = math.log(float(y.sum() / offset.exp().sum()))
    trace = []
    norm = 1 + float(y.sum())
    converged = False
    for iteration in range(max_iter):
        eta = offset + x @ beta
        if float(eta.abs().max()) > 40:
            _error("Likelihood left the declared finite/interior mean domain.", "boundary_solution")
        mu = eta.exp()
        ll = _likelihood(y, eta)
        grad = x.T @ (y - mu)
        score = float(grad.abs().max()) / norm
        trace.append([float(iteration), ll, score])
        if float(mu.min()) <= 1e-8:
            _error("Active fitted cell left the finite interior mean domain.", "boundary_solution")
        info, _ = _information(x, mu)
        direction = torch.linalg.solve(info, grad)
        # A zero-cell boundary can have a tiny score but a constant Newton step.
        stationary = float((x @ direction).abs().max()) <= max(math.sqrt(tol), 1e-7)
        if score <= tol and stationary:
            converged = True
            break
        rate = 1.0
        accepted = False
        for _ in range(40):
            candidate = beta + rate * direction
            next_eta = offset + x @ candidate
            if (
                float(next_eta.abs().max()) <= 40
                and _likelihood(y, next_eta) >= ll + 1e-4 * rate * float(grad @ direction) - 1e-10
            ):
                beta = candidate
                accepted = True
                break
            rate *= 0.5
        if not accepted:
            _error("Poisson Newton line search failed; no partial fit returned.", "nonconvergence")
    eta = offset + x @ beta
    if not converged:
        _error("Poisson ML did not attain the declared score tolerance.", "nonconvergence")
    if float(eta.exp().min()) <= 1e-8:
        _error(
            "An active fitted cell approaches zero; finite interior MLE is outside this domain.",
            "boundary_solution",
        )
    return beta, eta.exp(), trace


def _state_result(grid, signature, x, names, beta, mu, trace, method, options, resource, extra):
    active = [i for i, z in enumerate(grid["structural"]) if not z]
    y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
    offset = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
    info, covariance = _information(x, mu)
    se = covariance.diag().sqrt()
    z = beta / se
    cut = normal_isf((1 - options["level"]) / 2)
    eta = offset + x @ beta
    deviance = float(
        2
        * (
            torch.where(
                y > 0,
                y * torch.log(torch.where(y > 0, y / mu, torch.ones_like(y))),
                torch.zeros_like(y),
            )
            - (y - mu)
        ).sum()
    )
    deviance = max(deviance, 0.0)
    pearson = float(((y - mu).square() / mu).sum())
    ll = _likelihood(y, eta)
    df = len(active) - len(names)
    log_variance = (x @ covariance * x).sum(1)
    if (log_variance < -1e-10).any():
        _error("Invalid mean variance.", "numerical_failure")
    log_se = log_variance.clamp_min(0).sqrt()
    if float((eta + cut * log_se).max()) > 700:
        _error(
            "Natural-scale mean confidence limits exceed the finite export domain.",
            "numerical_failure",
        )
    means = [0.0] * len(grid["cells"])
    mean_se = [None] * len(means)
    lower = [None] * len(means)
    upper = [None] * len(means)
    residual = [None] * len(means)
    for j, i in enumerate(active):
        means[i] = float(mu[j])
        mean_se[i] = float(mu[j] * log_se[j])
        lower[i] = float(torch.exp(eta[j] - cut * log_se[j]))
        upper[i] = float(torch.exp(eta[j] + cut * log_se[j]))
        residual[i] = float((y[j] - mu[j]) / mu[j].sqrt())
    state = {
        "schema": _SCHEMA,
        "grid": grid,
        "grid_signature": signature,
        "method": method,
        "design": x.tolist(),
        "terms": names,
        "coefficients": beta.tolist(),
        "fitted_active": mu.tolist(),
        "covariance": covariance.tolist(),
        "information": info.tolist(),
        "active_cells": active,
        "log_likelihood": ll,
        "deviance": deviance,
        "df": df,
        "options": options,
        "extra": extra,
    }
    state["sha256"] = _digest(state)
    output = TableSet(
        {
            "parameters": table(
                [
                    [
                        float(b),
                        float(s),
                        float(t),
                        2 * normal_sf(abs(float(t))),
                        float(b - cut * s),
                        float(b + cut * s),
                    ]
                    for b, s, t in zip(beta, se, z)
                ],
                columns=["estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper"],
                index=names,
            ),
            "covariance": table(covariance.tolist(), columns=names, index=names),
            "information": table(info.tolist(), columns=names, index=names),
            "cells": table(
                [
                    [
                        *cell,
                        grid["counts"][i],
                        grid["structural"][i],
                        means[i],
                        mean_se[i],
                        lower[i],
                        upper[i],
                        residual[i],
                    ]
                    for i, cell in enumerate(grid["cells"])
                ],
                columns=[
                    *grid["dimensions"],
                    "observed",
                    "structural_zero",
                    "fitted",
                    "fitted_se",
                    "fitted_ci_lower",
                    "fitted_ci_upper",
                    "pearson_residual",
                ],
            ),
            "goodness_of_fit": table(
                [
                    [deviance, float(df), chi2_sf(deviance, df) if df > 0 else None],
                    [pearson, float(df), chi2_sf(pearson, df) if df > 0 else None],
                ],
                columns=["statistic", "df", "p_value"],
                index=["deviance", "pearson"],
            ),
            "convergence": table(
                trace, columns=["iteration", "log_likelihood", "scaled_score_or_margin_error"]
            ),
        },
        title="Independent-Poisson loglinear " + method,
        method=method,
        nobs=len(active),
        total_count=sum(grid["counts"]),
        input_rows=grid["input_rows"],
        structural_zero_cells=sum(grid["structural"]),
        sample_positions=[grid["positions"][i] for i in active],
        df_resid=df,
        log_likelihood=ll,
        state=state,
        resource=resource,
        precision="float64",
        device="cpu",
        converged=True,
        inference="Normal Wald and asymptotic chi-square under independent Poisson cell sampling; no fixed-margin/cluster/weight correction",
        notes=[
            "Small fitted counts can make chi-square and Wald approximations inaccurate; no finite-sample calibration is asserted."
        ],
        **extra,
    )
    # summary_state stores sorted keys; use that same stable display order.
    ordered = TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)
    return saved_summary(ordered)


def _hierarchy(dimensions, levels, margins):
    if not isinstance(margins, (list, tuple)) or not 1 <= len(margins) <= 32:
        _error("margins must declare 1..32 generating dimension sets.")
    generators = []
    for margin in margins:
        group = _names(margin, "generating margin", 1, len(dimensions))
        if any(d not in dimensions for d in group):
            _error("Margins must use declared dimensions.")
        group = tuple(d for d in dimensions if d in group)
        if group in generators:
            _error("Repeated generating margin.")
        generators.append(group)
    if set().union(*map(set, generators)) != set(dimensions):
        _error("Every cell dimension must be included in generating margins.")
    generators = [g for g in generators if not any(set(g) < set(h) for h in generators)]
    subsets = {s for g in generators for k in range(1, len(g) + 1) for s in combinations(g, k)}
    contrasts = [
        (s, cat)
        for s in sorted(subsets, key=lambda s: (len(s), tuple(dimensions.index(d) for d in s)))
        for cat in product(*(range(1, len(levels[d])) for d in s))
    ]
    return generators, contrasts


@resident_cpu
def loglinear_ipf(
    data,
    dimensions,
    count,
    *,
    levels,
    margins,
    structural=None,
    offset=None,
    max_iter=1000,
    tol=1e-9,
    level=0.95,
    max_work=300_000_000,
    max_bytes=128 * 1024**2,
):
    """Fit hierarchical independent-Poisson cell counts by IPF on declared support.

    Include every cell explicitly. ``structural`` names a boolean zero-support
    column, distinct from sampling zeros. ``offset`` is a known log exposure.
    Treatment contrasts use the first declared category as reference; columns
    aliased by structural support are recorded and removed deterministically.
    No pseudocount/boundary fit, automatic term selection or weights are used.
    """
    max_iter, tol, level, max_work, budget = _options(max_iter, tol, level, max_work, max_bytes)
    grid, signature = _grid(data, dimensions, count, levels, structural, offset, budget)
    dimensions = grid["dimensions"]
    generators, contrasts = _hierarchy(dimensions, grid["levels"], margins)
    active = [i for i, z in enumerate(grid["structural"]) if not z]
    resource = _admit(
        len(active),
        1 + len(contrasts),
        max_iter,
        max_work,
        budget,
        len(active) * len(generators) * max_iter,
    )
    coords = list(product(*(range(k) for k in grid["shape"])))
    x_columns = [torch.ones(len(active), dtype=_F)]
    names = ["intercept"]
    aliases = []
    q = [x_columns[0] / math.sqrt(len(active))]
    for subset, categories in contrasts:
        name = ":".join(f"{d}[{c}]" for d, c in zip(subset, categories))
        column = torch.tensor(
            [
                float(all(coords[i][dimensions.index(d)] == c for d, c in zip(subset, categories)))
                for i in active
            ],
            dtype=_F,
        )
        residual = column.clone()
        for _ in range(2):
            for unit in q:
                residual -= unit * (unit @ residual)
        if float(residual.norm()) <= 1e-9 * max(1.0, float(column.norm())):
            aliases.append(name)
            continue
        q.append(residual / residual.norm())
        x_columns.append(column)
        names.append(name)
    x = torch.stack(x_columns, dim=1)
    y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
    off = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
    grouped = []
    for generator in generators:
        group_coords = [tuple(coords[i][dimensions.index(d)] for d in generator) for i in active]
        unique = sorted(set(group_coords))
        mapping = {v: j for j, v in enumerate(unique)}
        ids = torch.tensor([mapping[v] for v in group_coords], dtype=torch.int64)
        observed = torch.zeros(len(unique), dtype=_F).scatter_add_(0, ids, y)
        if (observed <= 0).any():
            _error(
                "A supported generating-margin cell has no counts; boundary MLE is unsupported.",
                "boundary_solution",
            )
        grouped.append((ids, observed))
    mu = off.exp()
    mu *= y.sum() / mu.sum()
    trace = []
    converged = False
    for iteration in range(max_iter):
        for ids, target in grouped:
            current = torch.zeros(len(target), dtype=_F).scatter_add_(0, ids, mu)
            mu *= (target / current)[ids]
        error = max(
            float(
                (
                    (torch.zeros(len(target), dtype=_F).scatter_add_(0, ids, mu) - target).abs()
                    / (1 + target)
                ).max()
            )
            for ids, target in grouped
        )
        trace.append([float(iteration), _likelihood(y, mu.log()), error])
        if error <= tol:
            converged = True
            break
    if not converged:
        _error("IPF did not attain the declared generating-margin tolerance.", "nonconvergence")
    if not torch.isfinite(mu).all() or float(mu.min()) <= 1e-8:
        _error("Active cell MLE is outside the finite interior mean domain.", "boundary_solution")
    beta = torch.linalg.lstsq(x, mu.log() - off, driver="gels").solution
    if float((x @ beta + off - mu.log()).abs().max()) > 1e-7 or float(
        (x.T @ (y - mu)).abs().max()
    ) / (1 + float(y.sum())) > max(10 * tol, 1e-8):
        _error("IPF fit did not certify its hierarchical likelihood score.", "nonconvergence")
    # Positive margins do not imply finite MLE existence: cell separation
    # can send means to zero while every fitted margin appears converged.
    information, _ = _information(x, mu)
    step = torch.linalg.solve(information, x.T @ (y - mu))
    stationary = float((x @ step).abs().max()) <= max(math.sqrt(tol), 1e-7)
    if not stationary:
        _error(
            "IPF did not certify a finite stationary interior MLE; no Wald inference returned.",
            "nonconvergence",
        )
    return _state_result(
        grid,
        signature,
        x,
        names,
        beta,
        mu,
        trace,
        "ipf",
        dict(max_iter=max_iter, tol=tol, level=level),
        resource,
        dict(
            generating_margins=[list(g) for g in generators],
            aliased_terms=aliases,
            contrast_coding="first-category treatment contrasts",
        ),
    )


@resident_cpu
def loglinear_ml(
    data,
    dimensions,
    count,
    *,
    levels,
    design,
    terms=None,
    structural=None,
    offset=None,
    max_iter=200,
    tol=1e-9,
    level=0.95,
    max_work=300_000_000,
    max_bytes=128 * 1024**2,
):
    """Fit general Poisson cell-design ML with joint observed information.

    ``design`` rows align exactly to supplied cell rows, including structural
    rows; the complete declared grid determines canonical saved ordering.
    Supply identifiable contrasts: aliased columns are rejected. Independent
    cells, known log offsets, Normal Wald and asymptotic GOF are the sole
    inference domain. This is not multinomial/fixed-margin inference.
    """
    max_iter, tol, level, max_work, budget = _options(max_iter, tol, level, max_work, max_bytes)
    grid, signature = _grid(data, dimensions, count, levels, structural, offset, budget)
    if not hasattr(design, "__len__") or len(design) != len(grid["cells"]):
        _error("Design rows must align to all supplied cell rows.")
    if not hasattr(design[0], "__len__"):
        _error("Design must be a two-dimensional numeric matrix.")
    p = len(design[0])
    active = [i for i, z in enumerate(grid["structural"]) if not z]
    resource = _admit(len(active), p, max_iter, max_work, budget, full_rows=len(grid["cells"]))
    if any(not hasattr(row, "__len__") or len(row) != p for row in design):
        _error("Every raw design row must have the declared column width.")
    if isinstance(design, torch.Tensor) and design.device.type != "cpu":
        _error(
            "Design must be resident CPU input; implicit device transfer is unsupported.",
            "unsupported_option",
        )
    names = _names(terms, "terms") if terms is not None else [f"beta{i}" for i in range(p)]
    if len(names) != p:
        _error("terms must align to design columns.")
    try:
        matrix = torch.as_tensor(design, dtype=_F, device="cpu")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise AnalysisError("invalid_spec", "Design must be a rectangular numeric matrix.") from exc
    if (
        tuple(matrix.shape) != (len(grid["cells"]), p)
        or not torch.isfinite(matrix).all()
        or float(matrix.abs().max()) > 100
    ):
        _error("Design must be finite with |entry|<=100 and align exactly to declared rows/terms.")
    x = matrix[[grid["positions"][i] for i in active]]
    singular = torch.linalg.svdvals(x)
    if len(active) < p or float(singular[-1]) <= 1e-8 * float(singular[0]):
        _error(
            "Caller cell design is rank-deficient or ill-conditioned on active support.",
            "unidentified_model",
        )
    y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
    off = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
    beta, mu, trace = _fit_newton(x, y, off, max_iter, tol)
    return _state_result(
        grid,
        signature,
        x,
        names,
        beta,
        mu,
        trace,
        "ml",
        dict(max_iter=max_iter, tol=tol, level=level),
        resource,
        dict(contrast_coding="caller-declared cell design"),
    )


def _checked(result, budget):
    if not isinstance(result, TableSet):
        _error("Use a loglinear TableSet or complete restore_summary result.", "invalid_result")
    state = result.attrs.get("state")
    if not isinstance(state, dict) or state.get("schema") != _SCHEMA:
        _error("No complete loglinear state.", "invalid_state")
    copy = {k: v for k, v in state.items() if k != "sha256"}
    try:
        if _digest(copy) != state.get("sha256"):
            _error("Loglinear state checksum changed.", "invalid_state")
        n, p = len(state["active_cells"]), len(state["terms"])
        if n > 4096 or not 1 <= p <= 128 or state["method"] not in ("ipf", "ml"):
            _error("Loglinear state domain changed.", "invalid_state")
        _admit(n, p, 1, 2_000_000_000, budget)
        matrix_fields = [("design", n, p), ("covariance", p, p), ("information", p, p)]
        for key, rows, columns in matrix_fields:
            matrix = state[key]
            if (
                not isinstance(matrix, list)
                or len(matrix) != rows
                or any(not isinstance(row, list) or len(row) != columns for row in matrix)
            ):
                _error("Saved matrix shapes changed before conversion.", "invalid_state")
        if len(state["coefficients"]) != p or len(state["fitted_active"]) != n:
            _error("Saved vector shapes changed before conversion.", "invalid_state")
        x = torch.tensor(state["design"], dtype=_F)
        beta = torch.tensor(state["coefficients"], dtype=_F)
        mu = torch.tensor(state["fitted_active"], dtype=_F)
        cov = torch.tensor(state["covariance"], dtype=_F)
        info = torch.tensor(state["information"], dtype=_F)
        grid = state["grid"]
        active = state["active_cells"]
        if (
            active != [i for i, z in enumerate(grid["structural"]) if not z]
            or (x.shape != (n, p))
            or (beta.shape != (p,))
            or (cov.shape != (p, p))
            or (info.shape != (p, p))
            or mu.shape != (n,)
        ):
            _error("Loglinear state dimensions/support changed.", "invalid_state")
        if (
            any(not torch.isfinite(v).all() for v in (x, beta, mu, cov, info))
            or float(mu.min()) <= 1e-8
        ):
            _error("Nonfinite/interior loglinear state.", "invalid_state")
        off = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
        y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
        if (
            not torch.allclose((off + x @ beta).exp(), mu, rtol=1e-7, atol=1e-9)
            or not torch.allclose(info, x.T @ (mu[:, None] * x), rtol=1e-7, atol=1e-8)
            or not torch.allclose(info @ cov, torch.eye(p, dtype=_F), rtol=1e-6, atol=1e-6)
        ):
            _error("Saved coefficients/means/information/covariance disagree.", "invalid_state")
        if (
            not torch.isfinite(y).all()
            or (y < 0).any()
            or (y.round() != y).any()
            or not 0 < float(y.sum()) <= 1_000_000_000
        ):
            _error("Saved Poisson counts changed.", "invalid_state")
        likelihood = _likelihood(y, off + x @ beta)
        if (
            not math.isfinite(likelihood)
            or abs(likelihood - state["log_likelihood"]) > 1e-6
            or state["df"] != n - p
        ):
            _error("Saved likelihood/rank changed.", "invalid_state")
        tolerance = state["options"]["tol"]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not 1e-12 <= tolerance <= 1e-4
        ):
            _error("Saved convergence tolerance changed.", "invalid_state")
        score = x.T @ (y - mu)
        direction = torch.linalg.solve(info, score)
        score_limit = max(10 * tolerance, 1e-8) if state["method"] == "ipf" else tolerance
        if float(score.abs().max()) / (1 + float(y.sum())) > score_limit or float(
            (x @ direction).abs().max()
        ) > max(math.sqrt(tolerance), 1e-7):
            _error("Saved model does not certify a stationary Poisson MLE.", "invalid_state")
        signature = _digest(
            {
                k: grid[k]
                for k in (
                    "dimensions",
                    "levels",
                    "cells",
                    "counts",
                    "structural",
                    "offset",
                    "sampling",
                )
            }
        )
        if signature != state["grid_signature"] or grid["sampling"] != "independent_poisson_cells":
            _error("Saved sampling/grid signature changed.", "invalid_state")
        return state, x
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError("invalid_state", "Invalid complete loglinear state.") from exc


@resident_cpu
def loglinear_compare(restricted, full, *, max_bytes=128 * 1024**2):
    """Asymptotic LR comparison of nested models on identical Poisson cells.

    Verifies entire counts/support/offset and design-space inclusion; matching
    dimensions or parameter counts alone do not establish nestedness.
    """
    budget = min(_integer(max_bytes, "max_bytes", 1, 512 * 1024**2), workspace_budget_bytes())
    small, x0 = _checked(restricted, budget)
    large, x1 = _checked(full, budget)
    if small["grid_signature"] != large["grid_signature"]:
        _error(
            "Compare exactly the same cell counts, levels, support, sampling and offsets.",
            "incompatible_models",
        )
    df = x1.shape[1] - x0.shape[1]
    if df <= 0:
        _error("Full model must add identified parameters.", "incompatible_models")
    projection = x1 @ torch.linalg.lstsq(x1, x0, driver="gels").solution
    if float((projection - x0).abs().max()) > 1e-8 * max(1.0, float(x0.abs().max())):
        _error("Restricted cell design is not nested in the full model.", "incompatible_models")
    statistic = 2 * (large["log_likelihood"] - small["log_likelihood"])
    if statistic < -1e-6:
        _error("Nested likelihoods violate MLE ordering.", "invalid_state")
    statistic = max(0.0, statistic)
    return saved_summary(
        TableSet(
            {
                "comparison": table(
                    [[statistic, float(df), chi2_sf(statistic, df)]],
                    columns=["lr_statistic", "df", "p_value"],
                ),
                "models": table(
                    [
                        [small["method"], len(small["terms"]), small["log_likelihood"]],
                        [large["method"], len(large["terms"]), large["log_likelihood"]],
                    ],
                    columns=["method", "rank", "log_likelihood"],
                    index=["restricted", "full"],
                ),
            },
            title="Nested independent-Poisson loglinear LR",
            sampling="independent_poisson_cells",
            statistic=statistic,
            df=df,
            p_value=chi2_sf(statistic, df),
            grid_signature=small["grid_signature"],
            restricted_state=small,
            full_state=large,
            nested_design_verified=True,
            device="cpu",
            precision="float64",
        )
    )
