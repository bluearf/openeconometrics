"""Conditional multinomial cell likelihoods with explicit sampling strata."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import chi2_sf, normal_isf, normal_sf
from openecon.resources import workspace_budget_bytes
from .loglinear import _admit, _digest, _error, _grid, _integer, _key, _names, _options

_F = torch.float64
_SCHEMA = "openecon.conditional-loglinear.v1"


def _saved(output):
    output = saved_summary(output)
    # The complete persistence format sorts table keys and uses each table's
    # common-dtype row array. Match that exact representation before display.
    for name, frame in list(output.items()):
        output[name] = table(
            frame.to_numpy().tolist(), columns=list(frame.columns), index=list(frame.index)
        )
    return TableSet(dict(sorted(output.items())), title=output.title, **output.attrs)


def _signature(grid, conditioning):
    return _digest(
        {
            "grid": {
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
            },
            "conditioning": conditioning,
        }
    )


def _groups(grid, conditioning, active):
    keys = list(product(*(grid["levels"][d] for d in conditioning))) if conditioning else [()]
    typed = [tuple(_key(v) for v in row) for row in keys]
    lookup = {row: i for i, row in enumerate(typed)}
    columns = [grid["dimensions"].index(d) for d in conditioning]
    ids = [lookup[tuple(_key(grid["cells"][i][j]) for j in columns)] for i in active]
    buckets = [[] for _ in keys]
    for j, g in enumerate(ids):
        buckets[g].append(j)
    groups = [torch.tensor(bucket, dtype=torch.int64) for bucket in buckets]
    totals = [sum(grid["counts"][active[j]] for j in idx.tolist()) for idx in groups]
    if any(not len(idx) or total <= 0 for idx, total in zip(groups, totals)):
        _error(
            "Every declared fixed-total stratum must have active support and positive count.",
            "empty_stratum",
        )
    return keys, ids, groups, torch.tensor(totals, dtype=_F)


def _evaluate(x, y, offset, beta, groups, totals):
    eta = offset + x @ beta
    logp = torch.empty_like(eta)
    mu = torch.empty_like(eta)
    info = torch.zeros((x.shape[1], x.shape[1]), dtype=_F)
    centered = torch.empty_like(x)
    for idx, total in zip(groups, totals):
        if float((eta[idx].max() - eta[idx].min())) > 80:
            _error(
                "Conditional contrast exceeded the interior probability domain.",
                "boundary_solution",
            )
        shifted = eta[idx] - eta[idx].max()
        lp = shifted - torch.logsumexp(shifted, 0)
        p = lp.exp()
        logp[idx], mu[idx] = lp, total * p
        z = x[idx] - (p[:, None] * x[idx]).sum(0)
        centered[idx] = z
        info += z.T @ ((total * p)[:, None] * z)
    ll = float((y * logp - torch.lgamma(y + 1)).sum() + torch.lgamma(totals + 1).sum())
    return ll, logp, mu, info, centered


def _invert(info):
    eigen = torch.linalg.eigvalsh(info)
    if not torch.isfinite(eigen).all() or float(eigen[0]) <= max(1e-10, float(eigen[-1]) * 1e-12):
        _error(
            "Conditional information is singular/ill-conditioned; no fabricated covariance.",
            "unidentified_model",
        )
    return torch.linalg.inv(info)


def _newton(x, y, offset, groups, totals, max_iter, tol):
    beta = torch.zeros(x.shape[1], dtype=_F)
    trace = []
    for iteration in range(max_iter):
        ll, logp, mu, info, centered = _evaluate(x, y, offset, beta, groups, totals)
        covariance = _invert(info)
        grad = x.T @ (y - mu)
        step = covariance @ grad
        score = float(grad.abs().max()) / (1 + float(y.sum()))
        predictor_step = float((centered @ step).abs().max())
        trace.append([float(iteration), ll, score, predictor_step])
        if score <= tol and predictor_step <= max(math.sqrt(tol), 1e-7):
            if float(mu.min()) <= 1e-8:
                _error("Active fitted counts approach a boundary MLE.", "boundary_solution")
            return beta, logp, mu, info, covariance, centered, trace
        direction = float(grad @ step)
        accepted = False
        for k in range(50):
            alpha = 0.5**k
            trial = beta + alpha * step
            try:
                trial_ll = _evaluate(x, y, offset, trial, groups, totals)[0]
            except AnalysisError as exc:
                if exc.code != "boundary_solution":
                    raise
                continue
            if trial_ll >= ll + 1e-4 * alpha * direction - 1e-10:
                beta = trial
                accepted = True
                break
        if not accepted:
            _error("Conditional likelihood line search failed.", "nonconvergence")
    _error("Conditional multinomial ML did not certify convergence.", "nonconvergence")


def _fit(
    data,
    dimensions,
    count,
    levels,
    design,
    terms,
    conditioning,
    structural,
    offset,
    max_iter,
    tol,
    level,
    max_work,
    max_bytes,
):
    max_iter, tol, level, max_work, budget = _options(max_iter, tol, level, max_work, max_bytes)
    dimensions = _names(dimensions, "dimensions", 2, 6)
    if not isinstance(levels, dict) or set(levels) != set(dimensions):
        _error("Declare all dimension levels.")
    # Avoid materializing oversized named column sequences before grid admission.
    if isinstance(data, Mapping):
        if any(d not in data for d in [*dimensions, count]):
            _error("Required cell columns are missing.")
        try:
            sizes = [len(data[d]) for d in [*dimensions, count]]
        except TypeError:
            _error("Cell columns must be finite resident sequences.")
        if any(n > 4096 for n in sizes):
            _error("Conditional cell input exceeds 4096 rows before coercion.", "resource_limit")
    elif hasattr(data, "__len__") and len(data) > 4096:
        _error("Conditional cell input exceeds 4096 rows before coercion.", "resource_limit")
    grid, _ = _grid(data, dimensions, count, levels, structural, offset, budget)
    if conditioning is None:
        conditioning = []
    else:
        conditioning = _names(conditioning, "conditioning", 1, len(dimensions) - 1)
        if any(d not in dimensions for d in conditioning):
            _error("Conditioning must be a proper subset of declared dimensions.")
        conditioning = [d for d in dimensions if d in conditioning]
    grid["sampling"] = (
        "product_multinomial_fixed_strata" if conditioning else "multinomial_fixed_total"
    )
    active = [i for i, structural_zero in enumerate(grid["structural"]) if not structural_zero]
    if (
        not hasattr(design, "__len__")
        or len(design) != len(grid["cells"])
        or not hasattr(design[0], "__len__")
    ):
        _error("Design must have a row for each supplied cell, including structural cells.")
    p = len(design[0])
    resource = _admit(
        len(active),
        p,
        max_iter,
        max_work,
        budget,
        # Each declared Newton iteration can evaluate all 50 Armijo trials.
        extra=50 * len(active) * p * p * max_iter + len(active) * p * (max_iter + 8),
        full_rows=len(grid["cells"]),
    )
    keys, ids, groups, totals = _groups(grid, conditioning, active)
    if any(not hasattr(row, "__len__") or len(row) != p for row in design):
        _error("Design rows must have a consistent declared width.")
    if isinstance(design, torch.Tensor) and design.device.type != "cpu":
        _error(
            "Supply resident CPU design; implicit transfer is unsupported.", "unsupported_option"
        )
    names = _names(terms, "terms") if terms is not None else [f"beta{i}" for i in range(p)]
    if len(names) != p:
        _error("terms must align with the cell design.")
    try:
        matrix = torch.as_tensor(design, dtype=_F, device="cpu")
    except (ValueError, TypeError, RuntimeError) as exc:
        raise AnalysisError(
            "invalid_spec", "Design must be a resident rectangular numeric matrix."
        ) from exc
    if (
        matrix.shape != (len(grid["cells"]), p)
        or not torch.isfinite(matrix).all()
        or float(matrix.abs().max()) > 100
    ):
        _error("Design entries must be finite, bounded by 100 and aligned to cells.")
    x = matrix[[grid["positions"][i] for i in active]]
    residual = x.clone()
    for idx in groups:
        residual[idx] -= x[idx].mean(0)
    singular = torch.linalg.svdvals(residual)
    if (
        len(active) - len(groups) < p
        or len(singular) < p
        or float(singular[-1]) <= 1e-8 * max(float(singular[0]), 1e-12)
    ):
        _error(
            "Design is aliased with fixed stratum constants; omit intercept/nuisance columns.",
            "unidentified_model",
        )
    y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
    off = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
    beta, logp, mu, info, covariance, centered, trace = _newton(
        x, y, off, groups, totals, max_iter, tol
    )
    ll = _evaluate(x, y, off, beta, groups, totals)[0]
    se = covariance.diag().sqrt()
    cut = normal_isf((1 - level) / 2)
    deviance = max(
        0.0, float(2 * torch.where(y > 0, y * (torch.where(y > 0, y, 1).log() - mu.log()), 0).sum())
    )
    pearson = float(((y - mu).square() / mu).sum())
    df = len(active) - len(groups) - p
    delta = mu[:, None] * centered
    mean_se = ((delta @ covariance) * delta).sum(1).clamp_min(0).sqrt()
    rows = []
    j = 0
    for i, cell in enumerate(grid["cells"]):
        if grid["structural"][i]:
            rows.append([*cell, grid["counts"][i], True, 0.0, 0.0, None, None])
        else:
            rows.append(
                [
                    *cell,
                    grid["counts"][i],
                    False,
                    float(logp[j].exp()),
                    float(mu[j]),
                    float(mean_se[j]),
                    float((y[j] - mu[j]) / mu[j].sqrt()),
                ]
            )
            j += 1
    state = dict(
        schema=_SCHEMA,
        grid=grid,
        conditioning=conditioning,
        signature=_signature(grid, conditioning),
        active=active,
        groups=ids,
        totals=totals.tolist(),
        design=x.tolist(),
        terms=names,
        coefficients=beta.tolist(),
        probabilities=logp.exp().tolist(),
        fitted=mu.tolist(),
        covariance=covariance.tolist(),
        information=info.tolist(),
        df=df,
        log_likelihood=ll,
        options=dict(tol=tol, max_iter=max_iter, level=level),
    )
    state["sha256"] = _digest(state)
    tables = {
        "parameters": table(
            [
                [
                    float(b),
                    float(s),
                    float(b / s),
                    2 * normal_sf(abs(float(b / s))),
                    float(b - cut * s),
                    float(b + cut * s),
                ]
                for b, s in zip(beta, se)
            ],
            columns=["estimate", "std_error", "z", "p_value", "ci_lower", "ci_upper"],
            index=names,
        ),
        "covariance": table(covariance.tolist(), columns=names, index=names),
        "information": table(info.tolist(), columns=names, index=names),
        "cells": table(
            rows,
            columns=[
                *dimensions,
                "observed",
                "structural_zero",
                "probability",
                "fitted",
                "fitted_se",
                "pearson_residual",
            ],
        ),
        "strata": table(
            [
                [*key, float(total), float(mu[idx].sum())]
                for key, total, idx in zip(keys, totals, groups)
            ],
            columns=[*conditioning, "observed_total", "fitted_total"],
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
            trace,
            columns=["iteration", "log_likelihood", "scaled_score", "centered_predictor_step"],
        ),
    }
    return _saved(
        TableSet(
            dict(sorted(tables.items())),
            title="Fixed-total conditional loglinear ML",
            method="conditional_ml",
            sampling=grid["sampling"],
            conditioning=conditioning,
            nobs=len(active),
            total_count=float(y.sum()),
            input_rows=len(grid["cells"]),
            sample_positions=[grid["positions"][i] for i in active],
            df_resid=df,
            log_likelihood=ll,
            state=state,
            resource=resource,
            converged=True,
            precision="float64",
            device="cpu",
            inference="Conditional multinomial Normal Wald and asymptotic chi-square; no finite-sample, survey, cluster or post-selection calibration",
            notes=[
                "Cell mean SE uses the stratum-normalized probability delta derivative; structural cells have no estimable probability uncertainty."
            ],
        )
    )


@resident_cpu
def loglinear_multinomial(
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
    """Direct fixed-grand-total multinomial loglinear ML on complete cell support.

    Supply contrasts without an intercept: common cell shifts are unidentified.
    Full likelihood includes the multinomial factorial constant. Independent
    Poisson nuisance parameters and covariance are not reused.
    """
    return _fit(
        data,
        dimensions,
        count,
        levels,
        design,
        terms,
        None,
        structural,
        offset,
        max_iter,
        tol,
        level,
        max_work,
        max_bytes,
    )


@resident_cpu
def loglinear_product_multinomial(
    data,
    dimensions,
    count,
    *,
    levels,
    conditioning,
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
    """Direct conditional ML with fixed totals for explicit conditioning dimensions.

    Every declared stratum needs positive observed count and active support.
    Contrasts constant within strata (including intercepts) are refused. Design
    rows align to supplied complete cells before canonical ordering.
    """
    return _fit(
        data,
        dimensions,
        count,
        levels,
        design,
        terms,
        conditioning,
        structural,
        offset,
        max_iter,
        tol,
        level,
        max_work,
        max_bytes,
    )


def _checked(result, budget):
    if not isinstance(result, TableSet):
        _error("Use a complete conditional-loglinear TableSet.", "invalid_state")
    state = result.attrs.get("state")
    try:
        if not isinstance(state, dict) or state.get("schema") != _SCHEMA:
            _error("Conditional state integrity/domain changed.", "invalid_state")
        grid, active = state["grid"], state["active"]
        expected_fields = {
            "schema",
            "grid",
            "conditioning",
            "signature",
            "active",
            "groups",
            "totals",
            "design",
            "terms",
            "coefficients",
            "probabilities",
            "fitted",
            "covariance",
            "information",
            "df",
            "log_likelihood",
            "options",
            "sha256",
        }
        expected_grid = {
            "dimensions",
            "levels",
            "shape",
            "cells",
            "counts",
            "structural",
            "offset",
            "positions",
            "input_rows",
            "missing",
            "sampling",
        }
        if (
            set(state) != expected_fields
            or not isinstance(grid, dict)
            or set(grid) != expected_grid
            or not isinstance(active, list)
        ):
            _error(
                "Conditional saved field structure changed before serialization.", "invalid_state"
            )
        n, p = len(active), len(state["terms"])
        if not 1 <= n <= 4096 or not 1 <= p <= 128 or len(grid["cells"]) > 4096:
            _error("Conditional saved dimensions exceeded bounds.", "invalid_state")
        _admit(n, p, 1, 2_000_000_000, budget)
        for field, r, c in [("design", n, p), ("information", p, p), ("covariance", p, p)]:
            v = state[field]
            if (
                not isinstance(v, list)
                or len(v) != r
                or any(not isinstance(row, list) or len(row) != c for row in v)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for row in v
                    for value in row
                )
            ):
                _error(
                    "Conditional saved matrix shapes changed before conversion.", "invalid_state"
                )
        for field, size in [
            ("coefficients", p),
            ("probabilities", n),
            ("fitted", n),
            ("groups", n),
        ]:
            if not isinstance(state[field], list) or len(state[field]) != size:
                _error(
                    "Conditional saved vector shapes changed before conversion.", "invalid_state"
                )
        for field in ("coefficients", "probabilities", "fitted"):
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in state[field]
            ):
                _error(
                    "Conditional saved vectors must contain finite numeric scalars.",
                    "invalid_state",
                )
        for field in ("signature", "sha256"):
            if not isinstance(state[field], str) or len(state[field]) != 64:
                _error("Conditional saved digests changed.", "invalid_state")
        if (
            isinstance(state["df"], bool)
            or not isinstance(state["df"], int)
            or not 0 <= state["df"] <= 4096
            or isinstance(state["log_likelihood"], bool)
            or not isinstance(state["log_likelihood"], (int, float))
            or not math.isfinite(state["log_likelihood"])
        ):
            _error("Conditional saved df/likelihood changed.", "invalid_state")
        dims = _names(grid["dimensions"], "saved dimensions", 2, 6)
        _names(state["terms"], "saved terms", 1, 128)
        ncells = len(grid["cells"])
        for field in ("counts", "structural", "offset", "positions"):
            if not isinstance(grid[field], list) or len(grid[field]) != ncells:
                _error("Conditional saved cell vector dimensions changed.", "invalid_state")
        if not isinstance(grid["cells"], list) or any(
            not isinstance(cell, list) or len(cell) != len(dims) for cell in grid["cells"]
        ):
            _error("Conditional saved category dimensions changed.", "invalid_state")
        if not isinstance(state["totals"], list) or not 1 <= len(state["totals"]) <= ncells:
            _error("Conditional saved stratum-total dimensions changed.", "invalid_state")
        if not isinstance(state["options"], dict) or set(state["options"]) != {
            "tol",
            "max_iter",
            "level",
        }:
            _error("Conditional saved options changed.", "invalid_state")
        _options(
            state["options"]["max_iter"],
            state["options"]["tol"],
            state["options"]["level"],
            1,
            budget,
        )
        conditioning = state["conditioning"]
        if (
            not isinstance(conditioning, list)
            or len(set(conditioning)) != len(conditioning)
            or any(d not in dims for d in conditioning)
            or len(conditioning) >= len(dims)
        ):
            _error("Conditional saved strata changed.", "invalid_state")
        # Re-admit the full canonical grid, including typed levels and structural counts.
        temporary = ["__oe_count", "__oe_structural", "__oe_offset"]
        while any(name in dims for name in temporary):
            temporary = ["_" + name for name in temporary]
        count, structural, offset = temporary
        rows = [
            dict(zip(dims, cell), **{count: y, structural: s, offset: o})
            for cell, y, s, o in zip(
                grid["cells"], grid["counts"], grid["structural"], grid["offset"]
            )
        ]
        checked_grid, _ = _grid(rows, dims, count, grid["levels"], structural, offset, budget)
        expected_sampling = (
            "product_multinomial_fixed_strata" if conditioning else "multinomial_fixed_total"
        )
        checked_grid["sampling"] = expected_sampling
        if (
            _signature(checked_grid, conditioning) != state["signature"]
            or _signature(grid, conditioning) != state["signature"]
            or grid["sampling"] != expected_sampling
            or grid["shape"] != checked_grid["shape"]
            or grid["input_rows"] != len(grid["cells"])
            or grid["missing"] != "raise"
            or not isinstance(grid["positions"], list)
            or any(isinstance(i, bool) or not isinstance(i, int) for i in grid["positions"])
            or sorted(grid["positions"]) != list(range(len(grid["cells"])))
            or active != [i for i, s in enumerate(grid["structural"]) if not s]
        ):
            _error("Conditional saved cells/support/signature disagree.", "invalid_state")
        _, ids, groups, totals = _groups(grid, conditioning, active)
        if state["groups"] != ids or state["totals"] != totals.tolist():
            _error("Conditional saved totals disagree with cells.", "invalid_state")
        x = torch.tensor(state["design"], dtype=_F)
        beta = torch.tensor(state["coefficients"], dtype=_F)
        cov = torch.tensor(state["covariance"], dtype=_F)
        saved_info = torch.tensor(state["information"], dtype=_F)
        if (
            any(not torch.isfinite(v).all() for v in [x, beta, cov, saved_info])
            or float(x.abs().max()) > 100
        ):
            _error("Conditional saved numerical domain changed.", "invalid_state")
        # Serialize only after bounded complete shapes, typed cell grid, terms,
        # options and numerical matrices have all been admitted.
        if _digest({k: v for k, v in state.items() if k != "sha256"}) != state.get("sha256"):
            _error("Conditional state checksum changed.", "invalid_state")
        y = torch.tensor([grid["counts"][i] for i in active], dtype=_F)
        off = torch.tensor([grid["offset"][i] for i in active], dtype=_F)
        ll, logp, mu, info, centered = _evaluate(x, y, off, beta, groups, totals)
        _invert(info)
        if (
            float(mu.min()) <= 1e-8
            or not torch.allclose(info, saved_info, rtol=1e-7, atol=1e-8)
            or not torch.allclose(info @ cov, torch.eye(p, dtype=_F), rtol=1e-6, atol=1e-6)
            or not torch.allclose(mu, torch.tensor(state["fitted"], dtype=_F), rtol=1e-7, atol=1e-9)
            or not torch.allclose(
                logp.exp(), torch.tensor(state["probabilities"], dtype=_F), rtol=1e-7, atol=1e-10
            )
            or abs(ll - state["log_likelihood"]) > 1e-6
            or state["df"] != n - len(groups) - p
        ):
            _error("Conditional saved means/likelihood/information disagree.", "invalid_state")
        tol = state["options"]["tol"]
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 1e-12 <= tol <= 1e-4:
            _error("Conditional convergence tolerance changed.", "invalid_state")
        score = x.T @ (y - mu)
        if float(score.abs().max()) / (1 + float(y.sum())) > tol or float(
            (centered @ torch.linalg.solve(info, score)).abs().max()
        ) > max(math.sqrt(tol), 1e-7):
            _error("Conditional saved fit is not a stationary interior MLE.", "invalid_state")
        contrast = x.clone()
        for idx in groups:
            contrast[idx] -= x[idx].mean(0)
        return state, contrast
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
        if isinstance(exc, AnalysisError):
            raise
        raise AnalysisError(
            "invalid_state", "Invalid complete conditional-loglinear state."
        ) from exc


@resident_cpu
def loglinear_sampling_compare(restricted, full, *, max_bytes=128 * 1024**2):
    """Asymptotic nested LR on identical fixed-total cells and conditioning strata.

    Checks full restored numerical states and nested conditional design spaces,
    allowing nuisance constant shifts within strata. No post-selection p-values.
    """
    budget = min(_integer(max_bytes, "max_bytes", 1, 512 * 1024**2), workspace_budget_bytes())
    small, x0 = _checked(restricted, budget)
    large, x1 = _checked(full, budget)
    if small["signature"] != large["signature"]:
        _error(
            "Conditional LR requires identical counts/support/offsets/sampling strata.",
            "incompatible_models",
        )
    df = x1.shape[1] - x0.shape[1]
    if df <= 0:
        _error("Full model must add identified conditional parameters.", "incompatible_models")
    projection = x1 @ torch.linalg.lstsq(x1, x0, driver="gels").solution
    if float((projection - x0).abs().max()) > 1e-8 * max(1.0, float(x0.abs().max())):
        _error("Conditional design spaces are not nested.", "incompatible_models")
    statistic = 2 * (large["log_likelihood"] - small["log_likelihood"])
    if statistic < -1e-6:
        _error("Conditional nested MLE likelihood ordering failed.", "invalid_state")
    statistic = max(0.0, statistic)
    return _saved(
        TableSet(
            {
                "comparison": table(
                    [[statistic, float(df), chi2_sf(statistic, df)]],
                    columns=["lr_statistic", "df", "p_value"],
                )
            },
            title="Nested conditional-loglinear LR",
            sampling=small["grid"]["sampling"],
            conditioning=small["conditioning"],
            statistic=statistic,
            df=df,
            p_value=chi2_sf(statistic, df),
            restricted_state=small,
            full_state=large,
            inference="Asymptotic conditional multinomial LR; no finite-sample or post-selection calibration",
        )
    )
