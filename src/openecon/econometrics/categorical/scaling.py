"""Bounded descriptive categorical geometries and ordinal stress majorization."""

from __future__ import annotations

import hashlib
import json
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace, workspace_budget_bytes
from ..core import TableSet, table
from ..resident_cpu import resident_cpu
from ..summary_state import saved_summary

DTYPE = torch.float64
BYTES = 128 * 1024**2
WORK = 300_000_000


def _error(message, code="invalid_options"):
    raise AnalysisError(code, message)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        _error(f"{name} must be an integer from {low} to {high}.")
    return value


def _admit(operation, size, work, max_bytes, max_work):
    _integer(max_bytes, "max_bytes", 1, BYTES)
    _integer(max_work, "max_work", 1, WORK)
    if work > max_work:
        _error(f"{operation} planned work {work} exceeds max_work.", "resource_limit")
    plan_workspace(
        operation,
        {"resident_tensors_and_saved_state": int(size)},
        budget_bytes=min(max_bytes, workspace_budget_bytes()),
    )
    return {
        "estimated_workspace_bytes": int(size),
        "planned_work": int(work),
        "max_bytes": max_bytes,
        "max_work": max_work,
    }


def _atom(value):
    if hasattr(value, "item"):
        value = value.item()
    if type(value) not in (str, int, float, bool) or (
        isinstance(value, float) and not math.isfinite(value)
    ):
        _error("Categories and row labels must be finite JSON scalars.", "invalid_data")
    if isinstance(value, str) and len(value) > 256:
        _error("Category/row label strings are bounded at 256 characters.", "invalid_data")
    if type(value) is int and abs(value) > 2**53 - 1:
        _error(
            "Integer category/row labels must be exactly representable in float64.", "invalid_data"
        )
    return value


def _key(value):
    value = _atom(value)
    return type(value).__name__, value


def _sample(data, variables, missing, max_bytes, max_work, *, operation, min_rows=4):
    if not isinstance(data, pd.DataFrame) or not min_rows <= len(data) <= 3000:
        _error(f"Supply a DataFrame with {min_rows} to 3000 physical rows.", "invalid_data")
    if not isinstance(variables, (list, tuple)) or not 2 <= len(variables) <= 12:
        _error("Select 2 to 12 distinct named variables.")
    if any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in variables) or len(
        set(variables)
    ) != len(variables):
        _error("Variable names must be distinct strings.")
    if any(list(data.columns).count(v) != 1 for v in variables):
        _error("Each selected variable must occur exactly once.", "invalid_data")
    if missing not in ("drop", "raise"):
        _error("missing must be 'drop' or 'raise'.")
    # Admit row/column geometry before allocating a selected DataFrame or mask.
    _admit(
        operation,
        256 * len(data) * (len(variables) + 1),
        len(data) * len(variables),
        max_bytes,
        max_work,
    )
    selected = data.loc[:, list(variables)]
    mask = selected.notna().all(axis=1).to_numpy()
    if missing == "raise" and not bool(mask.all()):
        _error("Missing selected values are not admitted with missing='raise'.", "invalid_data")
    positions = [i for i, keep in enumerate(mask.tolist()) if keep]
    if len(positions) < min_rows:
        _error(f"At least {min_rows} complete observations are required.", "insufficient_sample")
    values = [
        [_atom(v) for v in row]
        for row in selected.iloc[positions].itertuples(index=False, name=None)
    ]
    labels = [_atom(data.index[i]) for i in positions]
    return values, labels, positions


def _levels(values, variables, orders=None, *, numeric=()):
    columns, codes, levels = list(zip(*values)), [], []
    for name, column in zip(variables, columns):
        observed = {}
        for value in column:
            observed.setdefault(_key(value), value)
        if len(observed) < 2 or (name not in numeric and len(observed) > 32):
            _error("Each categorical variable needs 2 to 32 observed levels.", "invalid_data")
        if orders and name in orders:
            order = [_atom(v) for v in orders[name]]
            if (
                len(order) != len(observed)
                or len({_key(v) for v in order}) != len(order)
                or set(map(_key, order)) != set(observed)
            ):
                _error("Every ordinal order must list each observed category exactly once.")
        else:
            order = list(observed.values())
        mapping = {_key(v): i for i, v in enumerate(order)}
        codes.append([mapping[_key(v)] for v in column])
        levels.append(order)
    return codes, levels


def _seal(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _orient(coordinates, *others):
    sign = torch.where(
        coordinates[coordinates.abs().argmax(0), torch.arange(coordinates.shape[1])] < 0, -1.0, 1.0
    )
    return (coordinates * sign, *(other * sign for other in others))


def _axes(count):
    return [f"dimension_{i + 1}" for i in range(count)]


def _save(output):
    """Canonical table order survives the JSON envelope's sorted object keys."""
    saved_summary(output)
    # The generic envelope exports to_numpy(), which promotes mixed numeric
    # integer/float columns. Canonicalize the public tables to that same finite
    # row representation so restoration retains exact rendering and dtypes.
    canonical = {
        name: table(frame.to_numpy().tolist(), columns=list(frame.columns), index=list(frame.index))
        for name, frame in sorted(output.items())
    }
    return TableSet(canonical, title=output.title, **output.attrs)


@resident_cpu
def mca(data, variables, n_components=2, *, missing="drop", max_bytes=BYTES, max_work=WORK):
    """Unweighted multiple correspondence analysis of complete disjunctive coding.

    Returns raw inertia, principal coordinates, contributions and complete
    supplementary-row calibration; no adjusted-inertia or inference claims.
    """
    dimensions = _integer(n_components, "n_components", 1, 12)
    values, labels, positions = _sample(
        data, variables, missing, max_bytes, max_work, operation="MCA"
    )
    codes, levels = _levels(values, variables)
    n, q, k = len(values), len(variables), sum(map(len, levels))
    resource = _admit(
        "MCA",
        8 * (12 * n * k + 12 * k * k + 8 * n * dimensions) + 256 * n * q,
        n * k * min(n, k) + k * k * dimensions,
        max_bytes,
        max_work,
    )
    if dimensions > min(n - 1, k - q):
        _error("n_components exceeds the centered complete-disjunctive rank.")
    z = torch.zeros((n, k), dtype=DTYPE)
    offset = 0
    for column, categories in zip(codes, levels):
        z[torch.arange(n), torch.tensor(column) + offset] = 1.0
        offset += len(categories)
    mass = z.mean(0) / q
    profiles = z / q
    standardized = (profiles - mass) / torch.sqrt(mass) / math.sqrt(n)
    _, singular, vt = torch.linalg.svd(standardized, full_matrices=False)
    rank = int((singular > singular[0] * 1e-12).sum())
    if dimensions > rank:
        _error("Selected MCA dimensions include zero-inertia axes.", "singular_design")
    eigenvalues = singular[:rank].square()
    standard = vt[:dimensions].T / torch.sqrt(mass)[:, None]
    row = (profiles - mass) @ standard
    category = standard * singular[:dimensions]
    row, standard, category = _orient(row, standard, category)
    calibration = {
        "variables": list(variables),
        "levels": levels,
        "category_mass": mass.tolist(),
        "category_standard": standard.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "n_components": dimensions,
        "complete_disjunctive_rank": rank,
    }
    rows = []
    offset = 0
    for name, categories in zip(variables, levels):
        for value in categories:
            rows.append([name, value, float(mass[offset]), *category[offset].tolist()])
            offset += 1
    summary = TableSet(
        {
            "row_coordinates": table(row.tolist(), columns=_axes(dimensions), index=labels),
            "category_coordinates": table(
                rows, columns=["variable", "category", "mass", *_axes(dimensions)]
            ),
            "category_standard": table(standard.tolist(), columns=_axes(dimensions)),
            "inertia": table(
                [
                    [i + 1, float(v), float(v / eigenvalues.sum())]
                    for i, v in enumerate(eigenvalues)
                ],
                columns=["dimension", "raw_inertia", "raw_share"],
            ),
            "row_contributions": table(
                (row.square() / n / eigenvalues[:dimensions]).tolist(),
                columns=_axes(dimensions),
                index=labels,
            ),
            "category_contributions": table(
                (category.square() * mass[:, None] / eigenvalues[:dimensions]).tolist(),
                columns=_axes(dimensions),
            ),
        },
        title="Multiple correspondence analysis",
        method="mca",
        calibration=calibration,
        calibration_sha256=_seal(calibration),
        nobs=n,
        input_rows=len(data),
        variables=list(variables),
        sample_positions=positions,
        row_labels=labels,
        input_categories=values,
        missing=missing,
        raw_total_inertia=float(eigenvalues.sum()),
        adjusted_inertia=False,
        weights="equal persons",
        device="cpu",
        dtype="float64",
        converged=True,
        **resource,
    )
    return _save(summary)


def _checked_mca(result):
    try:
        if not isinstance(result, TableSet) or result.attrs.get("method") != "mca":
            raise ValueError()
        calibration = result.attrs["calibration"]
        if not isinstance(calibration, dict) or set(calibration) != {
            "variables",
            "levels",
            "category_mass",
            "category_standard",
            "eigenvalues",
            "n_components",
            "complete_disjunctive_rank",
        }:
            raise ValueError()
        variables, levels = calibration["variables"], calibration["levels"]
        d = _integer(calibration["n_components"], "n_components", 1, 12)
        if (
            not 2 <= len(variables) <= 12
            or len(levels) != len(variables)
            or any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in variables)
            or len(set(variables)) != len(variables)
        ):
            raise ValueError()
        if any(
            not 2 <= len(items) <= 32 or len(set(map(_key, items))) != len(items)
            for items in levels
        ):
            raise ValueError()
        k = sum(map(len, levels))
        raw_standard, raw_mass = calibration["category_standard"], calibration["category_mass"]
        if (
            not isinstance(raw_standard, list)
            or len(raw_standard) != k
            or any(not isinstance(row, list) or len(row) != d for row in raw_standard)
            or not isinstance(raw_mass, list)
            or len(raw_mass) != k
        ):
            raise ValueError()
        rank = _integer(
            calibration["complete_disjunctive_rank"],
            "complete_disjunctive_rank",
            d,
            k - len(variables),
        )
        eigen = calibration["eigenvalues"]
        if (
            not isinstance(eigen, list)
            or len(eigen) != rank
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in eigen)
        ):
            raise ValueError()
        if _seal(calibration) != result.attrs["calibration_sha256"]:
            raise ValueError()
        standard = torch.tensor(raw_standard, dtype=DTYPE)
        mass = torch.tensor(raw_mass, dtype=DTYPE)
        if (
            standard.shape != (k, d)
            or mass.shape != (k,)
            or not bool(torch.isfinite(standard).all())
            or not bool(torch.isfinite(mass).all())
        ):
            raise ValueError()
        if bool((mass <= 0).any()) or abs(float(mass.sum()) - 1) > 1e-10:
            raise ValueError()
        if not torch.allclose(
            standard.T @ (mass[:, None] * standard), torch.eye(d, dtype=DTYPE), atol=1e-9, rtol=1e-9
        ):
            raise ValueError()
        if not torch.allclose(mass @ standard, torch.zeros(d, dtype=DTYPE), atol=1e-9, rtol=0):
            raise ValueError()
        if result["category_standard"].shape != (k, d) or not torch.equal(
            torch.tensor(result["category_standard"].to_numpy(), dtype=DTYPE), standard
        ):
            raise ValueError()
        offset = 0
        for categories in levels:
            if (
                abs(float(mass[offset : offset + len(categories)].sum()) - 1 / len(variables))
                > 1e-10
            ):
                raise ValueError()
            offset += len(categories)
        return calibration, standard, mass
    except (KeyError, TypeError, ValueError, RuntimeError, AnalysisError) as exc:
        raise AnalysisError(
            "invalid_state", "MCA calibration checksum, shapes or orthogonality are invalid."
        ) from exc


@resident_cpu
def mca_project(result, data, *, missing="raise", max_bytes=BYTES, max_work=WORK):
    """Project supplementary complete rows with saved MCA category masses/axes.

    Calibration works after oe.restore_summary; unknown categories are refused.
    """
    # Bound restored calibration before constructing any tensor from its lists.
    if not isinstance(result, TableSet) or result.attrs.get("method") != "mca":
        _error("Supply an MCA result or its complete summary restoration.", "invalid_state")
    _admit("MCA projection calibration", 2 * 1024**2, 10_000, max_bytes, max_work)
    calibration, standard, mass = _checked_mca(result)
    variables = calibration["variables"]
    values, labels, positions = _sample(
        data, variables, missing, max_bytes, max_work, operation="MCA projection", min_rows=1
    )
    n, k, d = len(values), len(mass), standard.shape[1]
    resource = _admit(
        "MCA projection",
        256 * n * len(variables) + 8 * (4 * n * k + 8 * k * d + 4 * n * d),
        n * k * d,
        max_bytes,
        max_work,
    )
    z = torch.zeros((n, k), dtype=DTYPE)
    offset = 0
    for column, categories in enumerate(calibration["levels"]):
        mapping = {_key(v): j for j, v in enumerate(categories)}
        try:
            codes = [mapping[_key(row[column])] for row in values]
        except KeyError as exc:
            raise AnalysisError(
                "unknown_category", "Supplementary rows contain an unfitted category."
            ) from exc
        z[torch.arange(n), torch.tensor(codes) + offset] = 1.0
        offset += len(categories)
    row = (z / len(variables) - mass) @ standard
    return _save(
        TableSet(
            {"row_coordinates": table(row.tolist(), columns=_axes(d), index=labels)},
            title="Supplementary MCA rows",
            method="mca_project",
            source_calibration_sha256=result.attrs["calibration_sha256"],
            nobs=n,
            input_rows=len(data),
            sample_positions=positions,
            row_labels=labels,
            input_categories=values,
            missing=missing,
            device="cpu",
            dtype="float64",
            **resource,
        )
    )


def _pava(y, weights):
    """Weighted least-squares nondecreasing cone projection, including ties."""
    values, weight = y.tolist(), weights.tolist()
    blocks = []
    for i, (value, count) in enumerate(zip(values, weight)):
        blocks.append([i, i + 1, value * count, count])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    output = torch.empty(len(values), dtype=DTYPE)
    for start, stop, total, count in blocks:
        output[start:stop] = total / count
    return output


def _mean_by_category(residual, codes, counts):
    totals = torch.zeros((len(counts), residual.shape[1]), dtype=DTYPE)
    totals.index_add_(0, codes, residual)
    return totals / counts[:, None]


def _unit_scores(target):
    centered = target - target.mean(0)
    u, singular, vt = torch.linalg.svd(centered, full_matrices=False)
    if float(singular[-1]) <= 1e-12:
        _error("The compromise score update lacks the requested rank.", "singular_design")
    return (u @ vt) * math.sqrt(len(target))


def _normalize_scalar(category, codes):
    magnitude = float(category.abs().max())
    category = category / max(1.0, magnitude)
    category = category - category[codes].mean()
    norm = float(category[codes].square().mean().sqrt())
    if norm < 1e-12:
        return None
    return category / norm


def _overals_start(codes, levels, variables, sets, scales, dimensions, seed, max_iter, tol):
    n = len(codes[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = _unit_scores(torch.randn((n, dimensions), generator=generator, dtype=DTYPE))
    counts = [
        torch.bincount(c, minlength=len(categories)).to(DTYPE)
        for c, categories in zip(codes, levels)
    ]
    quantifications, loadings, contribution = [], [], []
    for name, c, categories in zip(variables, codes, levels):
        if scales[name] == "multiple_nominal":
            quantifications.append(torch.zeros((len(categories), dimensions), dtype=DTYPE))
            loadings.append(torch.eye(dimensions, dtype=DTYPE))
        else:
            base = torch.tensor(
                categories if scales[name] == "numeric" else list(range(len(categories))),
                dtype=DTYPE,
            )
            quantifications.append(_normalize_scalar(base, c)[:, None])
            loadings.append(torch.zeros((1, dimensions), dtype=DTYPE))
        contribution.append(torch.zeros_like(x))
    group_indices = [[variables.index(name) for name in group] for group in sets]
    trace = []
    previous = float(dimensions)
    converged = False
    for iteration in range(1, max_iter + 1):
        for group in group_indices:
            fitted = sum((contribution[i] for i in group), torch.zeros_like(x))
            for i in group:
                residual = x - fitted + contribution[i]
                scale = scales[variables[i]]
                if scale == "multiple_nominal":
                    q = _mean_by_category(residual, codes[i], counts[i])
                    a = torch.eye(dimensions, dtype=DTYPE)
                elif scale == "numeric":
                    q = quantifications[i]
                    a = q[codes[i]].T @ residual / n
                elif scale == "nominal":
                    centroid = _mean_by_category(residual, codes[i], counts[i])
                    u, singular, vt = torch.linalg.svd(
                        torch.sqrt(counts[i])[:, None] * centroid, full_matrices=False
                    )
                    scalar = _normalize_scalar(u[:, 0] / torch.sqrt(counts[i]), codes[i])
                    if scalar is None:
                        q, a = quantifications[i], torch.zeros((1, dimensions), dtype=DTYPE)
                    else:
                        q = scalar[:, None]
                        a = q[codes[i]].T @ residual / n
                else:
                    # Exact coordinate steps for a monotone scalar and its loading.
                    q = quantifications[i]
                    a = q[codes[i]].T @ residual / n
                    if float(a.square().sum()) > 1e-18:
                        target = residual @ a.T / a.square().sum()
                        raw = _mean_by_category(target, codes[i], counts[i]).flatten()
                        scalar = _normalize_scalar(_pava(raw, counts[i]), codes[i])
                        if scalar is not None:
                            q = scalar[:, None]
                            a = q[codes[i]].T @ residual / n
                new = q[codes[i]] @ a
                fitted += new - contribution[i]
                contribution[i], quantifications[i], loadings[i] = new, q, a
        fits = [
            sum((contribution[i] for i in group), torch.zeros_like(x)) for group in group_indices
        ]
        x = _unit_scores(sum(fits, torch.zeros_like(x)) / len(sets))
        loss = float(
            sum(((x - f).square().sum() for f in fits), torch.zeros((), dtype=DTYPE))
            / (n * len(sets))
        )
        if loss > previous + 1e-9 * max(1.0, previous):
            _error("OVERALS objective increased; numerical convergence refused.", "nonconvergence")
        change = previous - loss
        trace.append([iteration, loss, change])
        if change <= tol * max(1.0, previous):
            converged = True
            break
        previous = loss
    return {
        "x": x,
        "quantifications": quantifications,
        "loadings": loadings,
        "fits": fits,
        "loss": loss,
        "trace": trace,
        "converged": converged,
    }


@resident_cpu
def overals(
    data,
    sets,
    n_components=2,
    *,
    scales=None,
    orders=None,
    missing="drop",
    n_starts=4,
    seed=0,
    max_iter=500,
    tol=1e-7,
    max_bytes=BYTES,
    max_work=WORK,
):
    """Multiset nonlinear canonical/homogeneity analysis by constrained block ALS.

    Explicit scales map each variable to multiple_nominal, nominal, ordinal or
    numeric; ordinal variables require explicit category orders. Descriptive
    local solutions use equal-person/equal-set loss and multiple starts.
    """
    dimensions = _integer(n_components, "n_components", 1, 6)
    starts = _integer(n_starts, "n_starts", 1, 12)
    _integer(seed, "seed", 0, 2**31 - 1)
    _integer(max_iter, "max_iter", 1, 1000)
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 1e-12 <= tol <= 1e-3:
        _error("tol must be finite between 1e-12 and 1e-3.")
    if (
        not isinstance(sets, (list, tuple))
        or not 2 <= len(sets) <= 12
        or any(not isinstance(group, (list, tuple)) or not group for group in sets)
    ):
        _error("Supply at least two nonempty disjoint variable sets.")
    variables = [name for group in sets for name in group]
    if any(not isinstance(name, str) for name in variables):
        _error("Set variables must be named strings.")
    if len(set(variables)) != len(variables):
        _error("A variable may belong to only one set.")
    if (
        not isinstance(scales, dict)
        or set(scales) != set(variables)
        or any(
            s not in ("multiple_nominal", "nominal", "ordinal", "numeric") for s in scales.values()
        )
    ):
        _error("scales must explicitly map every selected variable to an admitted scaling level.")
    ordinal = {name for name in variables if scales[name] == "ordinal"}
    if orders is None:
        orders = {}
    if (
        not isinstance(orders, dict)
        or set(orders) != ordinal
        or any(not isinstance(order, (list, tuple)) for order in orders.values())
    ):
        _error("orders must explicitly list each ordinal variable's observed levels.")
    values, labels, positions = _sample(
        data, variables, missing, max_bytes, max_work, operation="OVERALS"
    )
    numeric = {name for name in variables if scales[name] == "numeric"}
    if any(
        type(row[variables.index(name)]) not in (int, float) for name in numeric for row in values
    ):
        _error("Numeric scaling requires finite numeric values.", "invalid_data")
    code_lists, levels = _levels(values, variables, orders, numeric=numeric)
    n, q, k = len(values), len(variables), sum(map(len, levels))
    resource = _admit(
        "OVERALS",
        256 * n * q + 8 * (40 * n * q * dimensions + 24 * k * dimensions + 12 * starts * max_iter),
        starts * max_iter * n * q * dimensions * dimensions * 20,
        max_bytes,
        max_work,
    )
    ranks = [
        sum(
            len(levels[variables.index(name)]) - 1 if scales[name] == "multiple_nominal" else 1
            for name in group
        )
        for group in sets
    ]
    allowed_rank = (
        min(ranks) if len(sets) == 2 and "multiple_nominal" not in scales.values() else sum(ranks)
    )
    if dimensions >= n or dimensions > allowed_rank:
        _error("Requested dimensions exceed the declared set/scaling rank domain.")
    codes = [torch.tensor(column, dtype=torch.int64) for column in code_lists]
    solutions, trace = [], []
    for start in range(starts):
        solution = _overals_start(
            codes, levels, variables, sets, scales, dimensions, seed + start, max_iter, float(tol)
        )
        solutions.append(solution)
        trace.extend([[start, *row, solution["converged"]] for row in solution["trace"]])
    accepted = [(i, s) for i, s in enumerate(solutions) if s["converged"]]
    if not accepted:
        _error("No OVERALS start converged within max_iter.", "nonconvergence")
    best_index, best = min(accepted, key=lambda pair: pair[1]["loss"])
    x, fits = best["x"], best["fits"]
    compromise = sum(fits, torch.zeros_like(x)) / len(sets)
    explained = x.T @ compromise / n
    eigen, rotation = torch.linalg.eigh((explained + explained.T) / 2)
    rotation = rotation[:, torch.argsort(eigen, descending=True)]
    x, rotated = _orient(x @ rotation, rotation)
    fits = [fit @ rotated for fit in fits]
    quant_rows, loading_rows = [], []
    quantifications, loadings = [], []
    for i, (name, categories) in enumerate(zip(variables, levels)):
        quant, loading = best["quantifications"][i], best["loadings"][i] @ rotated
        if scales[name] == "multiple_nominal":
            quant = quant @ rotated
            loading = torch.eye(dimensions, dtype=DTYPE)
        elif scales[name] == "nominal" and float(quant[quant.abs().argmax(), 0]) < 0:
            quant, loading = -quant, -loading
        quantifications.append(quant.tolist())
        loadings.append(loading.tolist())
        for category, row in zip(categories, quant.tolist()):
            for copy, value in enumerate(row):
                quant_rows.append([name, category, copy + 1, value])
        for copy, row in enumerate(loading.tolist()):
            loading_rows.append([name, copy + 1, *row])
    fit_rows = [
        [j, float((x - fit).square().sum() / n), *torch.diag(x.T @ fit / n).tolist()]
        for j, fit in enumerate(fits)
    ]
    set_scores = [
        [j, positions[i], labels[i], *fit[i].tolist()]
        for j, fit in enumerate(fits)
        for i in range(n)
    ]
    return _save(
        TableSet(
            {
                "object_scores": table(x.tolist(), columns=_axes(dimensions), index=labels),
                "category_quantifications": table(
                    quant_rows, columns=["variable", "category", "copy", "quantification"]
                ),
                "variable_loadings": table(
                    loading_rows, columns=["variable", "copy", *_axes(dimensions)]
                ),
                "set_scores": table(
                    set_scores, columns=["set", "source_position", "row_label", *_axes(dimensions)]
                ),
                "set_fit": table(fit_rows, columns=["set", "loss_per_person", *_axes(dimensions)]),
                "iterations": table(
                    trace, columns=["start", "iteration", "loss", "decrease", "start_converged"]
                ),
                "starts": table(
                    [
                        [i, s["loss"], len(s["trace"]), s["converged"]]
                        for i, s in enumerate(solutions)
                    ],
                    columns=["start", "loss", "iterations", "converged"],
                ),
            },
            title="Multiset nonlinear canonical analysis",
            method="overals",
            nobs=n,
            input_rows=len(data),
            variables=variables,
            sets=[list(group) for group in sets],
            scales=scales,
            orders=orders,
            levels=levels,
            codes=code_lists,
            input_categories=values,
            sample_positions=positions,
            row_labels=labels,
            quantifications=quantifications,
            loadings=loadings,
            n_components=dimensions,
            selected_start=best_index,
            objective=best["loss"],
            converged=True,
            n_starts=starts,
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            missing=missing,
            score_normalization="X centered; X'X/n = I",
            loss_definition="sum_set ||X - sum_variable Q_variable[codes] A_variable||^2 / (n * sets)",
            global_optimum=False,
            inference=False,
            weights="equal persons and equal sets",
            device="cpu",
            dtype="float64",
            **resource,
        )
    )


def _disparities(observed, distances, ties):
    if ties == "primary":
        # Within a tie block any order is feasible; distance order minimizes LS.
        distance_order = torch.argsort(distances, stable=True)
        order = distance_order[torch.argsort(observed[distance_order], stable=True)]
        fitted = _pava(distances[order], torch.ones(len(order), dtype=DTYPE))
        result = torch.empty_like(fitted)
        result[order] = fitted
    else:
        order = torch.argsort(observed, stable=True)
        unique, inverse, counts = torch.unique_consecutive(
            observed[order], return_inverse=True, return_counts=True
        )
        totals = torch.zeros(len(unique), dtype=DTYPE)
        totals.index_add_(0, inverse, distances[order])
        fitted = _pava(totals / counts, counts.to(DTYPE))
        result = torch.empty_like(distances)
        result[order] = fitted[inverse]
    norm = float(result.square().sum())
    if norm <= 1e-20:
        _error("The fitted disparity norm collapsed.", "nonconvergence")
    return result * math.sqrt(len(result) / norm)


def _nmds_start(delta, ii, jj, observed, weight, inverse, dimensions, seed, max_iter, tol, initial):
    n = len(delta)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (
        initial.clone()
        if initial is not None
        else torch.randn((n, dimensions), generator=generator, dtype=DTYPE)
    )
    x /= max(1.0, float(x.abs().max()))
    x -= x.mean(0)
    distance = torch.linalg.vector_norm(x[ii] - x[jj], dim=1)
    if float(distance.square().sum()) < 1e-20:
        _error("Initial coordinates must have positive pairwise distance.")
    x *= math.sqrt(len(observed) / float(distance.square().sum()))
    trace, previous, converged = [], math.inf, False
    for iteration in range(max_iter + 1):
        distance = torch.linalg.vector_norm(x[ii] - x[jj], dim=1)
        disparity = _disparities(observed, distance, weight["ties"])
        stress = float((distance - disparity).square().sum() / len(observed))
        if stress > previous + 1e-10:
            _error("NMDS stress increased; numerical convergence refused.", "nonconvergence")
        decrease = None if iteration == 0 else previous - stress
        trace.append([iteration, stress, decrease])
        if iteration and decrease <= tol * max(previous, 1e-3):
            converged = True
            break
        if iteration == max_iter:
            break
        ratio = torch.where(distance > 1e-12, disparity / distance.clamp_min(1e-12), 0.0)
        b = torch.zeros((n, n), dtype=DTYPE)
        b[ii, jj], b[jj, ii] = -ratio, -ratio
        b.diagonal().copy_(-b.sum(1))
        x = inverse @ b @ x
        x -= x.mean(0)
        previous = stress
    return {
        "x": x,
        "distance": distance,
        "disparity": disparity,
        "stress": stress,
        "converged": converged,
        "trace": trace,
    }


@resident_cpu
def mds_nonmetric(
    dissimilarities,
    n_components=2,
    *,
    ties="secondary",
    zero="include",
    n_starts=4,
    seed=0,
    max_iter=500,
    tol=1e-7,
    init=None,
    max_bytes=BYTES,
    max_work=WORK,
):
    """Ordinal multidimensional scaling by monotone-disparity SMACOF.

    Secondary ties remain equal; primary ties may split. Zero off-diagonals are
    included as the lowest level or explicitly excluded with a connected graph.
    Returns complete pair disparities, stress traces and identified coordinates.
    """
    dimensions = _integer(n_components, "n_components", 1, 6)
    starts = _integer(n_starts, "n_starts", 1, 12)
    _integer(seed, "seed", 0, 2**31 - 1)
    _integer(max_iter, "max_iter", 1, 1000)
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not 1e-12 <= tol <= 1e-3:
        _error("tol must be finite between 1e-12 and 1e-3.")
    if ties not in ("primary", "secondary") or zero not in ("include", "exclude"):
        _error("Use primary/secondary ties and include/exclude zero pairs.")
    try:
        n = len(dissimilarities)
    except TypeError as exc:
        raise AnalysisError(
            "invalid_data", "Supply a finite symmetric square dissimilarity matrix."
        ) from exc
    if not 4 <= n <= 150 or dimensions >= n:
        _error("NMDS admits 4 to 150 objects and fewer dimensions than objects.", "invalid_data")
    resource = _admit(
        "Nonmetric MDS",
        8 * (32 * n * n + 12 * starts * max_iter + 24 * n * dimensions),
        starts * max_iter * n * n * dimensions * 8 + n**3,
        max_bytes,
        max_work,
    )
    if isinstance(dissimilarities, pd.DataFrame):
        if dissimilarities.shape != (n, n):
            _error("Dissimilarities must be square before conversion.", "invalid_data")
        if list(dissimilarities.columns) != list(dissimilarities.index):
            _error(
                "A labeled dissimilarity matrix must have identical row/column labels.",
                "invalid_data",
            )
        labels = [_atom(label) for label in dissimilarities.index]
        raw = dissimilarities.to_numpy()
    else:
        labels, raw = list(range(n)), dissimilarities
    try:
        if len(raw) != n or any(len(row) != n for row in raw):
            _error("Dissimilarities must be square before conversion.", "invalid_data")
    except TypeError as exc:
        raise AnalysisError(
            "invalid_data", "Dissimilarities require n equal rows of n values."
        ) from exc
    try:
        delta = torch.as_tensor(raw, dtype=DTYPE).clone()
    except (ValueError, TypeError, RuntimeError) as exc:
        raise AnalysisError(
            "invalid_data", "Dissimilarities must form a finite numeric square matrix."
        ) from exc
    if (
        delta.shape != (n, n)
        or not bool(torch.isfinite(delta).all())
        or bool((delta < 0).any())
        or not torch.allclose(delta, delta.T, atol=1e-12, rtol=1e-12)
        or bool((delta.diagonal().abs() > 1e-12).any())
    ):
        _error(
            "Dissimilarities require symmetry, finite nonnegativity and a zero diagonal.",
            "invalid_data",
        )
    ii, jj = torch.triu_indices(n, n, offset=1)
    if zero == "exclude":
        keep = delta[ii, jj] > 0
        ii, jj = ii[keep], jj[keep]
    observed = delta[ii, jj]
    if len(observed) < n - 1 or float(observed.max()) <= 0 or len(torch.unique(observed)) < 2:
        _error(
            "NMDS needs at least two observed dissimilarity levels and connected pairs.",
            "invalid_data",
        )
    w = torch.zeros((n, n), dtype=DTYPE)
    w[ii, jj], w[jj, ii] = 1.0, 1.0
    laplacian = torch.diag(w.sum(1)) - w
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    if float(eigenvalues[1]) < 1e-10:
        _error("Excluded zero pairs leave a disconnected graph.", "invalid_data")
    inverse = (eigenvectors[:, 1:] / eigenvalues[1:]) @ eigenvectors[:, 1:].T
    initial = None
    if init is not None:
        try:
            if len(init) != n or any(len(row) != dimensions for row in init):
                _error("init must have n rows of n_components before conversion.")
            initial = torch.tensor(init, dtype=DTYPE)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise AnalysisError(
                "invalid_options", "init must be finite n by n_components coordinates."
            ) from exc
        if initial.shape != (n, dimensions) or not bool(torch.isfinite(initial).all()):
            _error("init must be finite n by n_components coordinates.")
    solutions, trace = [], []
    for start in range(starts):
        solution = _nmds_start(
            delta,
            ii,
            jj,
            observed,
            {"ties": ties},
            inverse,
            dimensions,
            seed + start,
            max_iter,
            tol,
            initial if start == 0 else None,
        )
        solutions.append(solution)
        trace.extend([[start, *row, solution["converged"]] for row in solution["trace"]])
    accepted = [(i, s) for i, s in enumerate(solutions) if s["converged"]]
    if not accepted:
        _error("No nonmetric MDS start converged within max_iter.", "nonconvergence")
    best_index, best = min(accepted, key=lambda pair: pair[1]["stress"])
    x = best["x"]
    _, singular, rotation = torch.linalg.svd(x, full_matrices=False)
    x = _orient(x @ rotation.T)[0]
    distances = torch.linalg.vector_norm(x[ii] - x[jj], dim=1)
    disparity = best["disparity"]
    raw_stress = float((distances - disparity).square().sum())
    stress1 = math.sqrt(raw_stress / float(distances.square().sum()))
    pairs = [
        [
            int(i),
            int(j),
            labels[int(i)],
            labels[int(j)],
            float(o),
            float(d),
            float(h),
            float((d - h) ** 2),
        ]
        for i, j, o, d, h in zip(ii, jj, observed, distances, disparity)
    ]
    return _save(
        TableSet(
            {
                "coordinates": table(x.tolist(), columns=_axes(dimensions), index=labels),
                "pairs": table(
                    pairs,
                    columns=[
                        "row_i",
                        "row_j",
                        "label_i",
                        "label_j",
                        "dissimilarity",
                        "distance",
                        "disparity",
                        "squared_residual",
                    ],
                ),
                "iterations": table(
                    trace,
                    columns=[
                        "start",
                        "iteration",
                        "normalized_stress",
                        "decrease",
                        "start_converged",
                    ],
                ),
                "starts": table(
                    [
                        [i, s["stress"], len(s["trace"]) - 1, s["converged"]]
                        for i, s in enumerate(solutions)
                    ],
                    columns=["start", "normalized_stress", "iterations", "converged"],
                ),
                "stress": table(
                    [[raw_stress, raw_stress / len(observed), stress1]],
                    columns=["raw_stress", "normalized_stress", "stress_1"],
                ),
            },
            title="Nonmetric multidimensional scaling",
            method="mds_nonmetric",
            nobs=n,
            n_components=dimensions,
            dissimilarities=delta.tolist(),
            row_labels=labels,
            observed_pair_positions=torch.stack((ii, jj), dim=1).tolist(),
            ties=ties,
            zero=zero,
            disparity_normalization="sum observed disparities squared = number of observed pairs",
            selected_start=best_index,
            n_starts=starts,
            seed=seed,
            max_iter=max_iter,
            tol=tol,
            initial_coordinates=None if initial is None else initial.tolist(),
            raw_stress=raw_stress,
            normalized_stress=raw_stress / len(observed),
            stress_1=stress1,
            converged=True,
            effective_rank=int((singular > singular[0] * 1e-10).sum()),
            global_optimum=False,
            inference=False,
            device="cpu",
            dtype="float64",
            **resource,
        )
    )
