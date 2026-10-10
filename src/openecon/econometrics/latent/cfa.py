"""Continuous CFA covariance ML, with exact observed joint information.

Native CPU float64 Torch; pandas is input/output only. Profiled unrestricted
observed means and normal-ML covariance divisor n are explicit. Boundary,
unidentified and nonconverged fits do not receive ordinary Wald inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import normal_isf, normal_sf
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.mi.common import (
    _decode_index,
    _encode_index,
    _freeze,
    _index_envelope,
    _thaw,
)
from openecon.econometrics.state_lifecycle import (
    _encoded_json_admission,
    _json_export_admission,
    _state_copy_admission,
)
from openecon.resources import plan_workspace

DTYPE = torch.float64
DEVICE = "cpu"
MAX_ROWS, MAX_COLUMNS, MAX_FACTORS, MAX_PARAMETERS = 10000, 16, 4, 80
MAX_WORK = 5_000_000_000
MAX_INDEX_BYTES = 2 * 1024 * 1024
SCHEMA = "openecon.cfa.v1"


def _error(code: str, text: str):
    raise AnalysisError(code, text)


def _integer(value, name, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not minimum <= value <= maximum
    ):
        _error("invalid_option", f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def _finite_real(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _real(value, name, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not _finite_real(value)
        or not minimum <= value <= maximum
    ):
        _error("invalid_option", f"{name} must be finite in [{minimum}, {maximum}].")
    return float(value)


def _tensor(value):
    return torch.tensor(value, dtype=DTYPE, device=DEVICE)


def _digest(value):
    return hashlib.sha256(
        json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _spec(factors, identification, markers):
    if identification not in ("marker", "unit_variance"):
        _error("invalid_specification", "identification must be marker or unit_variance.")
    if not isinstance(factors, Mapping) or not 1 <= len(factors) <= MAX_FACTORS:
        _error("invalid_specification", "Declare one to four named factors and their indicators.")
    names, indicators = [], []
    for name, cols in factors.items():
        if not isinstance(name, str) or not name or len(name) > 128 or name in names:
            _error("invalid_specification", "Factor names must be distinct nonempty strings.")
        if (
            isinstance(cols, (str, bytes))
            or not isinstance(cols, Sequence)
            or not 2 <= len(cols) <= MAX_COLUMNS
        ):
            _error(
                "invalid_specification",
                "Each factor needs an explicit sequence of 2–16 indicators.",
            )
        if any(not isinstance(c, str) or not c or len(c) > 256 for c in cols) or len(
            set(cols)
        ) != len(cols):
            _error(
                "invalid_specification",
                "Indicator names must be distinct nonempty strings within a factor.",
            )
        names.append(name)
        indicators.append(list(cols))
    columns = list(dict.fromkeys(c for cols in indicators for c in cols))
    if not 3 <= len(columns) <= MAX_COLUMNS:
        _error("invalid_specification", "CFA admits 3–16 distinct observed indicators.")
    if markers is None:
        anchors = [cols[0] for cols in indicators]
    else:
        if (
            identification != "marker"
            or not isinstance(markers, Mapping)
            or set(markers) != set(names)
        ):
            _error(
                "invalid_specification",
                "markers must declare one indicator for every marker-identified factor.",
            )
        anchors = [markers[name] for name in names]
    if any(anchor not in cols for anchor, cols in zip(anchors, indicators)) or len(
        set(anchors)
    ) != len(anchors):
        _error("invalid_specification", "Factor anchors must be distinct declared indicators.")
    free = [
        (columns.index(col), f)
        for f, cols in enumerate(indicators)
        for col in cols
        if identification == "unit_variance" or col != anchors[f]
    ]
    pairs = [
        (i, j)
        for i in range(len(names))
        for j in range(i + 1)
        if identification == "marker" or i != j
    ]
    q = len(free) + len(pairs) + len(columns)
    if q > MAX_PARAMETERS or q > len(columns) * (len(columns) + 1) // 2:
        _error(
            "not_identified",
            "More free parameters than covariance moments; ordinary CFA inference is unavailable.",
        )
    return {
        "factors": dict(zip(names, indicators)),
        "factor_names": names,
        "columns": columns,
        "identification": identification,
        "anchors": anchors,
        "free_loadings": free,
        "factor_pairs": pairs,
        "parameter_count": q,
    }


def _options(max_iterations, tolerance, max_work, level):
    return {
        "max_iterations": _integer(max_iterations, "max_iterations", 1, 500),
        "tolerance": _real(tolerance, "tolerance", 1e-10, 1e-5),
        "max_work": _integer(max_work, "max_work", 1, MAX_WORK),
        "level": _real(level, "level", 0.5, 0.9999),
    }


def _admit(n, spec, options, *, raw=False):
    n = _integer(n, "n", len(spec["columns"]) + 1, MAX_ROWS)
    p, q = len(spec["columns"]), spec["parameter_count"]
    # Two LBFGS starts, line searches, Newton polish, and exact Hessian/Jacobian.
    estimated_work = n * p * p + (options["max_iterations"] * 12 + q * q * 80 + 120) * p * p * p
    if estimated_work > options["max_work"]:
        _error(
            "work_limit",
            f"CFA declares {estimated_work:,} scalar-work units; max_work={options['max_work']:,}.",
        )
    plan = plan_workspace(
        "continuous_cfa",
        {
            "source_and_centered": 8 * n * p * 5,
            "moments_and_linear_algebra": 8 * p * p * 20,
            "autograd_information": 8 * q * q * p * p * 8,
            "optimizer_history": 8 * q * 100,
            "joint_covariance_and_jacobian": 8 * (q * q * 6 + p * p * q * 4),
            "encoded_index_and_digest": MAX_INDEX_BYTES * 3 if raw else 0,
        },
    )
    return {**plan.record(), "estimated_work": estimated_work, "max_work": options["max_work"]}


def _resident(data, columns):
    if isinstance(data, pd.DataFrame):
        n = len(data)
        if not data.columns.is_unique or not set(columns) <= set(data.columns):
            _error("invalid_input", "Input needs unique columns including all declared indicators.")
    elif isinstance(data, Mapping):
        if not set(columns) <= set(data):
            _error("invalid_input", "Missing declared indicator columns.")
        try:
            sizes = [len(data[c]) for c in columns]
        except TypeError:
            _error(
                "invalid_input", "Provide bounded resident columns, not scalar values or iterators."
            )
        if len(set(sizes)) != 1:
            _error("invalid_input", "Indicator columns must have equal lengths.")
        if any(
            isinstance(data[c], (str, bytes)) or getattr(data[c], "ndim", 1) != 1 for c in columns
        ):
            _error("invalid_input", "Declare bounded one-dimensional resident indicator columns.")
        indexed = [data[c].index for c in columns if isinstance(data[c], pd.Series)]
        if indexed and any(not value.identical(indexed[0]) for value in indexed[1:]):
            _error(
                "invalid_input",
                "Series mappings must share identical typed row indexes; no implicit outer alignment.",
            )
        n = sizes[0]
    else:
        _error(
            "unsupported_input", "CFA accepts a resident DataFrame or mapping of bounded columns."
        )
    return n


def _index_admission(index):
    """Probe caller-owned index labels without building a serialized index."""
    size = 2048

    def add(amount):
        nonlocal size
        size += amount
        if size > MAX_INDEX_BYTES:
            _error(
                "metadata_limit", "CFA typed index exceeds its 2 MiB metadata admission envelope."
            )

    def label(value, depth=0):
        if depth > 16:
            _error("metadata_limit", "CFA index tuple nesting exceeds 16 levels.")
        if isinstance(value, tuple):
            if len(value) > 16:
                _error("metadata_limit", "CFA index tuple labels admit at most 16 components.")
            add(512)
            for item in value:
                label(item, depth + 1)
        elif isinstance(value, str):
            if len(value) > 1024:
                _error("metadata_limit", "CFA index strings admit at most 1024 characters.")
            add(512 + 4 * len(value))
        else:
            if isinstance(value, Integral) and int(value).bit_length() > 256:
                _error("metadata_limit", "CFA index integers exceed the 256-bit label envelope.")
            add(512)

    def visit(value, depth=0):
        if depth > 16 or len(value) > MAX_ROWS:
            _error("metadata_limit", "CFA index levels exceed declared resident bounds.")
        if isinstance(value, pd.MultiIndex):
            if value.nlevels > 16:
                _error("metadata_limit", "CFA admits at most 16 index levels.")
            add(16 * len(value) * value.nlevels)
            for name in value.names:
                label(name)
            for level in value.levels:
                visit(level, depth + 1)
        elif isinstance(value, pd.CategoricalIndex):
            add(16 * len(value))
            label(value.name)
            visit(value.categories, depth + 1)
        elif isinstance(value, pd.RangeIndex):
            label(value.name)
        else:
            label(value.name)
            for item in value:
                label(item)

    visit(index)


def _saved_index_admission(index, n):
    _index_envelope(index, n)
    size = 0

    def visit(value, depth=0):
        nonlocal size
        if depth > 64:
            _error("metadata_limit", "CFA saved index nesting exceeds the metadata envelope.")
        if isinstance(value, Mapping):
            size += 96
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            size += 16 * len(value)
            for item in value:
                visit(item, depth + 1)
        elif isinstance(value, str):
            if len(value) > 1024:
                _error("metadata_limit", "CFA saved index strings exceed 1024 characters.")
            size += 128 + 4 * len(value)
        elif isinstance(value, Integral):
            if int(value).bit_length() > 256:
                _error("metadata_limit", "CFA saved index integer exceeds 256 bits.")
            size += 16
        else:
            size += 16
        if size > MAX_INDEX_BYTES:
            _error("metadata_limit", "CFA saved typed index exceeds its 2 MiB envelope.")

    visit(index)


def _moments(frame, missing):
    if missing not in ("raise", "drop"):
        _error("invalid_option", "missing must be raise or drop.")
    if any(
        not is_numeric_dtype(frame[c].dtype)
        or is_bool_dtype(frame[c].dtype)
        or is_complex_dtype(frame[c].dtype)
        for c in frame
    ):
        _error(
            "invalid_input",
            "Continuous CFA indicators must be real numeric, excluding booleans and complex values.",
        )
    raw = frame.astype("float64")
    for c in frame:
        if is_integer_dtype(frame[c].dtype) and any(
            not pd.isna(v) and int(float(v)) != int(v) for v in frame[c]
        ):
            _error(
                "precision_loss",
                "CFA integer indicators must be exactly representable in float64; source values are not rounded silently.",
            )
    for c in raw:
        if any(math.isinf(v) for v in raw[c]):
            _error("invalid_input", "Infinite indicators are not missing values.")
    present = ~raw.isna().any(axis=1)
    positions = [i for i, keep in enumerate(present) if keep]
    if len(positions) != len(raw) and missing == "raise":
        _error(
            "missing_data",
            "Indicators contain missing values; declare missing='drop' for complete-case CFA.",
        )
    if len(positions) <= len(frame.columns):
        _error("insufficient_sample", "Retained n must exceed the number of observed indicators.")
    y = _tensor(raw.iloc[positions].to_numpy())
    means = y.mean(0)
    centered = y - means
    cov = centered.T @ centered / len(positions)
    rows = [
        [None if pd.isna(v) else float(v) for v in row]
        for row in raw.itertuples(index=False, name=None)
    ]
    return (
        means,
        cov,
        {
            "kind": "raw",
            "rows": rows,
            "original_n": len(raw),
            "index": _encode_index(frame.index),
            "dtypes": [str(frame[c].dtype) for c in frame],
            "positions": positions,
            "missing": missing,
        },
    )


def _spd(cov):
    if cov.ndim != 2 or not torch.isfinite(cov).all():
        _error("invalid_input", "Covariance must be finite and symmetric.")
    if bool((cov.diag() <= 0).any()):
        _error("singular_sample", "Every observed indicator needs positive variance.")
    scales = cov.diag().sqrt()
    correlation = cov / scales[:, None] / scales[None, :]
    if not torch.allclose(correlation, correlation.T, atol=1e-12, rtol=1e-10):
        _error("invalid_input", "Covariance must be symmetric in normalized indicator units.")
    _, info = torch.linalg.cholesky_ex(cov)
    if int(info) != 0:
        _error(
            "singular_sample",
            "CFA needs a strictly positive definite observed covariance; no PSD repair is applied.",
        )


def _decode_parameters(x, spec, scales):
    p, k = len(spec["columns"]), len(spec["factor_names"])
    load = torch.zeros((p, k), dtype=DTYPE, device=DEVICE)
    if spec["identification"] == "marker":
        for f, anchor in enumerate(spec["anchors"]):
            load[spec["columns"].index(anchor), f] = 1.0
    for value, (j, f) in zip(x, spec["free_loadings"]):
        load[j, f] = value
    offset = len(spec["free_loadings"])
    chol = torch.eye(k, dtype=DTYPE, device=DEVICE)
    for value, (i, j) in zip(x[offset:], spec["factor_pairs"]):
        chol[i, j] = value.exp() if i == j else value
    phi = chol @ chol.T
    if spec["identification"] == "unit_variance":
        norm = phi.diag().sqrt()
        phi = phi / norm[:, None] / norm[None, :]
    theta = x[offset + len(spec["factor_pairs"]) :].exp()
    sigma = load @ phi @ load.T + torch.diag(theta)
    if spec["identification"] == "marker":
        units = scales[[spec["columns"].index(c) for c in spec["anchors"]]]
    else:
        units = torch.ones(k, dtype=DTYPE, device=DEVICE)
    raw_load = load * scales[:, None] / units[None, :]
    raw_phi = phi * units[:, None] * units[None, :]
    raw_theta = theta * scales.square()
    natural = torch.stack(
        [raw_load[j, f] for j, f in spec["free_loadings"]]
        + [raw_phi[i, j] for i, j in spec["factor_pairs"]]
        + list(raw_theta)
    )
    return sigma, natural, raw_load, raw_phi, raw_theta, theta


def _objective(x, spec, scales, standardized):
    sigma = _decode_parameters(x, spec, scales)[0]
    chol = torch.linalg.cholesky(sigma)
    return 0.5 * (
        2 * torch.log(chol.diag()).sum() + torch.trace(torch.cholesky_solve(standardized, chol))
    )


def _start(spec, amount):
    k = len(spec["factor_names"])
    values = [amount] * len(spec["free_loadings"])
    values += [math.log(0.7) if i == j else 0.12 for i, j in spec["factor_pairs"]]
    values += [math.log(0.5)] * len(spec["columns"])
    if k == 1 and spec["identification"] == "unit_variance":
        values[: len(spec["free_loadings"])] = [amount] * len(spec["free_loadings"])
    return _tensor(values)


def _solve(spec, scales, standardized, options):
    accepted, records = [], []
    for start_id, amount in enumerate((0.7, 0.4)):
        x = _start(spec, amount).requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [x],
            max_iter=options["max_iterations"],
            max_eval=options["max_iterations"] * 5,
            tolerance_grad=options["tolerance"] * 0.1,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0

        def closure():
            nonlocal evaluations
            evaluations += 1
            if evaluations > options["max_iterations"] * 6:
                _error("nonconvergence", "CFA line search exceeded its declared evaluation budget.")
            optimizer.zero_grad()
            loss = _objective(x, spec, scales, standardized)
            if not torch.isfinite(loss):
                _error("nonconvergence", "CFA optimizer reached a nonfinite objective.")
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            # A full observed-Hessian Newton polish avoids accepting LBFGS's
            # small-function-change stop as score convergence.
            lbfgs_iterations = int(optimizer.state[x].get("n_iter", 0))
            polish_iterations = 0
            for _ in range(min(20, options["max_iterations"] - lbfgs_iterations)):
                current = x.detach().requires_grad_(True)
                objective = _objective(current, spec, scales, standardized)
                grad = torch.autograd.grad(objective, current)[0]
                if float(grad.abs().max()) <= options["tolerance"]:
                    break
                hess = torch.autograd.functional.hessian(
                    lambda v: _objective(v, spec, scales, standardized), current
                )
                if float(torch.linalg.eigvalsh(hess).min()) <= 0:
                    break
                step = torch.linalg.solve(hess, grad)
                moved = False
                for power in range(20):
                    proposal = current.detach() - step / 2**power
                    new = _objective(proposal, spec, scales, standardized)
                    if torch.isfinite(new) and float(new.detach()) < float(objective.detach()):
                        with torch.no_grad():
                            x.copy_(proposal)
                        moved = True
                        break
                if not moved:
                    break
                polish_iterations += 1
            fitted = x.detach().requires_grad_(True)
            objective = _objective(fitted, spec, scales, standardized)
            grad = torch.autograd.grad(objective, fitted)[0]
            score_max = float(grad.abs().max())
            success = math.isfinite(float(objective.detach())) and score_max <= options["tolerance"]
            records.append(
                {
                    "start": start_id,
                    "initial_loading": amount,
                    "evaluations": evaluations,
                    "lbfgs_iterations": lbfgs_iterations,
                    "polish_iterations": polish_iterations,
                    "objective_per_observation": float(objective.detach()),
                    "score_max": score_max,
                    "converged": success,
                    "failure": None if success else "score_tolerance",
                }
            )
            if success:
                accepted.append((float(objective.detach()), fitted.detach()))
        except (RuntimeError, AnalysisError) as exc:
            records.append(
                {
                    "start": start_id,
                    "initial_loading": amount,
                    "evaluations": evaluations,
                    "converged": False,
                    "failure": str(exc)[:600],
                }
            )
    if not accepted:
        _error(
            "nonconvergence", f"Neither declared CFA start passed the score tolerance: {records}."
        )
    accepted.sort(key=lambda item: item[0])
    return accepted[0][1], records


def _names(spec):
    return (
        [
            f"loading:{spec['columns'][j]}<-{spec['factor_names'][f]}"
            for j, f in spec["free_loadings"]
        ]
        + [
            f"factor_cov:{spec['factor_names'][i]},{spec['factor_names'][j]}"
            for i, j in spec["factor_pairs"]
        ]
        + [f"residual_var:{c}" for c in spec["columns"]]
    )


def _evaluate(x, spec, cov, n, means, options):
    _spd(cov)
    scales = cov.diag().sqrt()
    standardized = cov / scales[:, None] / scales[None, :]
    x = x.detach().requires_grad_(True)
    sigma, natural, loading, phi, theta, std_theta = _decode_parameters(x, spec, scales)
    loss = _objective(x, spec, scales, standardized)
    gradient = torch.autograd.grad(loss, x)[0]
    if float(gradient.abs().max()) > options["tolerance"] * 1.01:
        _error("nonconvergence", "CFA per-observation score does not pass the declared tolerance.")
    latent_scales = phi.diag().sqrt()
    latent_correlation = phi / latent_scales[:, None] / latent_scales[None, :]
    latent_eigenvalues = torch.linalg.eigvalsh(latent_correlation)
    latent_variances = phi.diag()
    if spec["identification"] == "marker":
        anchor_variances = cov.diag()[[spec["columns"].index(c) for c in spec["anchors"]]]
        latent_variances = latent_variances / anchor_variances
    if (
        float(std_theta.min().detach()) <= 1e-8
        or float(latent_variances.min().detach()) <= 1e-8
        or float((latent_eigenvalues.min() / latent_eigenvalues.max()).detach()) <= 1e-10
    ):
        _error(
            "boundary_fit",
            "CFA reached a Heywood/latent-covariance boundary; ordinary OIM Wald inference is refused.",
        )
    if spec["identification"] == "unit_variance" and any(
        float(loading[spec["columns"].index(c), f].detach()) <= 0
        for f, c in enumerate(spec["anchors"])
    ):
        _error(
            "orientation_failure",
            "Unit-variance factors require positive first-indicator orientation for this fit.",
        )
    lower = torch.tril_indices(len(scales), len(scales), device=DEVICE)
    moment_jac = torch.autograd.functional.jacobian(
        lambda v: _decode_parameters(v, spec, scales)[0][lower[0], lower[1]], x
    )
    singular = torch.linalg.svdvals(moment_jac)
    if len(singular) < len(x) or float(singular.min() / singular.max()) <= 1e-9:
        _error(
            "not_identified",
            "The declared CFA covariance Jacobian is rank deficient or numerically weak; inference is refused.",
        )
    information = n * torch.autograd.functional.hessian(
        lambda v: _objective(v, spec, scales, standardized), x
    )
    information = (information + information.T) / 2
    eig = torch.linalg.eigvalsh(information)
    if not torch.isfinite(information).all() or float(eig.min() / eig.max()) <= 1e-10:
        _error(
            "singular_information",
            "The full observed information is not stably positive definite; no expected-information substitution.",
        )
    jac = torch.autograd.functional.jacobian(lambda v: _decode_parameters(v, spec, scales)[1], x)
    joint_cov = jac @ torch.linalg.solve(information, jac.T)
    joint_cov = (joint_cov + joint_cov.T) / 2
    if not torch.isfinite(joint_cov).all() or bool((joint_cov.diag() <= 0).any()):
        _error(
            "numerical_failure",
            "Full original-unit CFA uncertainty is not representable in float64; choose suitable indicator units.",
        )
    raw_sigma = sigma * scales[:, None] * scales[None, :]
    log_likelihood = -n * (
        float(loss.detach())
        + 0.5 * (len(scales) * math.log(2 * math.pi) + float(torch.log(scales.square()).sum()))
    )
    return {
        "parameter_names": _names(spec),
        "parameters": natural.detach().tolist(),
        "loadings": loading.detach().tolist(),
        "factor_covariance": phi.detach().tolist(),
        "residual_variances": theta.detach().tolist(),
        "implied_covariance": raw_sigma.detach().tolist(),
        "covariance": joint_cov.detach().tolist(),
        "observed_information_solver": information.detach().tolist(),
        "natural_jacobian": jac.detach().tolist(),
        "score_solver": (-n * gradient).detach().tolist(),
        "score_max_per_observation": float(gradient.abs().max()),
        "moment_jacobian_singular_values": singular.tolist(),
        "information_eigenvalues": eig.tolist(),
        "log_likelihood": log_likelihood,
        "means": None if means is None else means.tolist(),
        "sample_covariance_ml": cov.tolist(),
        "scales": scales.tolist(),
        "n": n,
        "df_covariance": len(scales) * (len(scales) + 1) // 2 - len(x),
    }


def _close(actual, expected, label, *, absolute=0.0):
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            _error("invalid_state", f"CFA {label} fields differ from semantic replay.")
        for key in expected:
            _close(actual[key], expected[key], label + "." + key, absolute=absolute)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            _error("invalid_state", f"CFA {label} shape differs from semantic replay.")
        for i, (a, e) in enumerate(zip(actual, expected)):
            _close(a, e, label + f"[{i}]", absolute=absolute)
    elif isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, Real)
            or not _finite_real(actual)
            or not math.isclose(actual, expected, rel_tol=2e-8, abs_tol=absolute)
        ):
            _error("invalid_state", f"CFA {label} disagrees with semantic replay.")
    elif actual != expected or type(actual) is not type(expected):
        _error("invalid_state", f"CFA {label} disagrees with semantic replay.")


def _cached_shapes(cached, p, f, q):
    """Admit cached dimensions before source scans and derivative replay."""
    matrices = {
        "loadings": (p, f), "factor_covariance": (f, f),
        "implied_covariance": (p, p), "covariance": (q, q),
        "observed_information_solver": (q, q), "natural_jacobian": (q, q),
        "sample_covariance_ml": (p, p),
    }
    vectors = {
        "parameter_names": q, "parameters": q, "residual_variances": p,
        "score_solver": q, "moment_jacobian_singular_values": q,
        "information_eigenvalues": q, "scales": p,
    }
    required = set(matrices) | set(vectors) | {
        "means", "score_max_per_observation", "log_likelihood", "n", "df_covariance",
    }
    if not isinstance(cached, Mapping) or set(cached) != required:
        _error("invalid_state", "CFA results fields differ from the declared contract.")
    for name, (height, width) in matrices.items():
        value = cached[name]
        if (
            not isinstance(value, (list, tuple)) or len(value) != height
            or any(not isinstance(row, (list, tuple)) or len(row) != width for row in value)
        ):
            _error("invalid_state", "CFA results." + name + " shape differs from the declared contract.")
        if any(not _finite_real(v) for row in value for v in row):
            _error("invalid_state", "CFA results." + name + " must contain finite real numeric values.")
    for name, size in vectors.items():
        value = cached[name]
        if not isinstance(value, (list, tuple)) or len(value) != size:
            _error("invalid_state", "CFA results." + name + " shape differs from the declared contract.")
        if name != "parameter_names" and any(not _finite_real(v) for v in value):
            _error("invalid_state", "CFA results." + name + " must contain finite real numeric values.")
    means = cached["means"]
    if means is not None and (not isinstance(means, (list, tuple)) or len(means) != p):
        _error("invalid_state", "CFA results.means shape differs from the declared contract.")
    if means is not None and any(not _finite_real(v) for v in means):
        _error("invalid_state", "CFA results.means must contain finite real numeric values.")
    for name in ("log_likelihood", "score_max_per_observation"):
        if not _finite_real(cached[name]):
            _error("invalid_state", "CFA results." + name + " must be finite real numeric.")


def _replay(payload):
    required = {
        "schema",
        "spec",
        "source",
        "options",
        "solver_parameters",
        "starts",
        "results",
        "resource_plan",
        "digest",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != SCHEMA
    ):
        _error("invalid_state", "Unknown or incomplete CFA saved-state contract.")
    state = payload
    declared = state["spec"]
    if not isinstance(declared, Mapping):
        _error("invalid_state", "Invalid CFA specification.")
    ordered_names = declared.get("factor_names")
    declared_factors = declared.get("factors")
    if (
        not isinstance(ordered_names, (list, tuple))
        or not 1 <= len(ordered_names) <= MAX_FACTORS
        or not isinstance(declared_factors, Mapping)
        or set(ordered_names) != set(declared_factors)
    ):
        _error("invalid_state", "Invalid CFA ordered factor names.")
    markers = (
        dict(zip(ordered_names, declared.get("anchors", ())))
        if declared.get("identification") == "marker"
        else None
    )
    spec = _spec(
        {name: declared_factors[name] for name in ordered_names},
        declared.get("identification"),
        markers,
    )
    _close(declared, _thaw(_freeze(spec)), "spec")
    opts = state["options"]
    if not isinstance(opts, Mapping) or set(opts) != {
        "max_iterations",
        "tolerance",
        "max_work",
        "level",
    }:
        _error("invalid_state", "Invalid CFA options.")
    options = _options(**opts)
    source = state["source"]
    if not isinstance(source, Mapping):
        _error("invalid_state", "Invalid CFA source.")
    p = len(spec["columns"])
    n_original = _integer(source.get("original_n"), "original_n", p + 1, MAX_ROWS)
    resource = _admit(n_original, spec, options, raw=source.get("kind") == "raw")
    _cached_shapes(state["results"], p, len(spec["factor_names"]), spec["parameter_count"])
    saved_resource = state["resource_plan"]
    if not isinstance(saved_resource, Mapping) or set(saved_resource) != set(resource):
        _error("invalid_state", "Invalid CFA resource plan.")
    _integer(
        saved_resource["budget_bytes"],
        "saved budget_bytes",
        resource["estimated_workspace_bytes"],
        2**63 - 1,
    )
    # The current caller budget is independently enforced; its byte limit may
    # legitimately differ from the fit-time limit.
    _close(
        {k: v for k, v in saved_resource.items() if k != "budget_bytes"},
        {k: v for k, v in resource.items() if k != "budget_bytes"},
        "resource_plan",
    )
    if source.get("kind") == "raw":
        keys = {"kind", "rows", "original_n", "index", "dtypes", "positions", "missing"}
        if (
            set(source) != keys
            or not isinstance(source["rows"], (list, tuple))
            or len(source["rows"]) != n_original
        ):
            _error("invalid_state", "Invalid CFA raw-source dimensions.")
        if any(not isinstance(row, (list, tuple)) or len(row) != p for row in source["rows"]):
            _error("invalid_state", "Invalid CFA raw-source column dimensions.")
        if any(
            v is not None
            and (isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v))
            for row in source["rows"]
            for v in row
        ):
            _error(
                "invalid_state",
                "CFA raw source must contain only finite real values or null missing values.",
            )
        if not isinstance(source["dtypes"], (list, tuple)) or len(source["dtypes"]) != p:
            _error("invalid_state", "Invalid CFA source dtypes.")
        for dtype in source["dtypes"]:
            if (
                not isinstance(dtype, str)
                or len(dtype) > 80
                or not is_numeric_dtype(pd.api.types.pandas_dtype(dtype))
                or is_bool_dtype(dtype)
                or is_complex_dtype(dtype)
            ):
                _error("invalid_state", "Invalid CFA numeric dtype descriptor.")
        for col, dtype in enumerate(source["dtypes"]):
            values = [row[col] for row in source["rows"]]
            try:
                recovered = pd.Series(values, dtype=dtype).astype("float64").tolist()
            except (ValueError, TypeError, OverflowError):
                _error(
                    "invalid_state",
                    "CFA values cannot be represented in the declared original dtype.",
                )
            if any(
                v is not None and (not math.isfinite(r) or v != r)
                for v, r in zip(values, recovered)
            ):
                _error("invalid_state", "CFA values disagree with the declared original dtype.")
        _saved_index_admission(source["index"], n_original)
        index = _decode_index(source["index"])
        frame = pd.DataFrame(source["rows"], columns=spec["columns"], index=index)
        means, cov, rebuilt = _moments(frame, source["missing"])
        _close(source["positions"], rebuilt["positions"], "source.positions")
        n = len(rebuilt["positions"])
    elif source.get("kind") == "summary":
        if set(source) != {"kind", "original_n", "covariance", "divisor", "means"}:
            _error("invalid_state", "Invalid CFA summary source.")
        values = source["covariance"]
        if (
            not isinstance(values, (list, tuple))
            or len(values) != p
            or any(not isinstance(row, (list, tuple)) or len(row) != p for row in values)
        ):
            _error("invalid_state", "Invalid CFA summary covariance dimensions.")
        if source["divisor"] not in ("n", "n-1"):
            _error("invalid_state", "Unknown CFA summary covariance divisor.")
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v)
            for row in values
            for v in row
        ):
            _error("invalid_state", "CFA summary covariance needs finite real numeric values.")
        cov = _tensor(values)
        if source["divisor"] == "n-1":
            cov = cov * (n_original - 1) / n_original
        means = None
        if source["means"] is not None:
            if not isinstance(source["means"], (list, tuple)) or len(source["means"]) != p:
                _error("invalid_state", "Invalid CFA summary means dimensions.")
            if any(
                isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v)
                for v in source["means"]
            ):
                _error("invalid_state", "CFA summary means need finite real numeric values.")
            means = _tensor(source["means"])
            if not torch.isfinite(means).all():
                _error("invalid_state", "Invalid CFA summary means.")
        n = n_original
    else:
        _error("invalid_state", "Unknown CFA source type.")
    solver = state["solver_parameters"]
    if (
        not isinstance(solver, (list, tuple))
        or len(solver) != spec["parameter_count"]
        or any(
            isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v) or abs(v) > 1e8
            for v in solver
        )
    ):
        _error("invalid_state", "Invalid CFA solver parameters.")
    starts = state["starts"]
    if not isinstance(starts, (list, tuple)) or len(starts) != 2:
        _error(
            "invalid_state", "CFA must retain both declared optimizer starts, including failures."
        )
    success = False
    for i, record in enumerate(starts):
        minimal_start = {"start", "initial_loading", "evaluations", "converged", "failure"}
        evaluated_start = minimal_start | {
            "lbfgs_iterations",
            "polish_iterations",
            "objective_per_observation",
            "score_max",
        }
        if (
            not isinstance(record, Mapping)
            or record.get("start") != i
            or record.get("initial_loading") != (0.7, 0.4)[i]
            or not isinstance(record.get("converged"), bool)
            or set(record) not in (minimal_start, evaluated_start)
        ):
            _error("invalid_state", "Invalid CFA start record.")
        _integer(record.get("evaluations"), "evaluations", 1, options["max_iterations"] * 6 + 1)
        if "lbfgs_iterations" in record:
            count = _integer(
                record["lbfgs_iterations"], "lbfgs_iterations", 0, options["max_iterations"]
            )
            _integer(
                record.get("polish_iterations"),
                "polish_iterations",
                0,
                min(20, options["max_iterations"] - count),
            )
        if record["converged"]:
            if set(record) != evaluated_start:
                _error(
                    "invalid_state",
                    "Converged CFA start is missing its complete iteration and score record.",
                )
            success = True
            _real(record.get("score_max"), "start score_max", 0.0, options["tolerance"])
            _real(record.get("objective_per_observation"), "start objective", -1e100, 1e100)
            if record.get("failure") is not None:
                _error("invalid_state", "Converged CFA start has a failure marker.")
        elif not isinstance(record.get("failure"), str) or not 1 <= len(record["failure"]) <= 600:
            _error("invalid_state", "Unsuccessful CFA start needs an explicit failure.")
    if not success:
        _error("invalid_state", "Saved CFA has no converged declared start.")
    expected = _evaluate(_tensor(solver), spec, cov, n, means, options)
    objective = -expected["log_likelihood"] / n - 0.5 * (
        p * math.log(2 * math.pi) + sum(math.log(s * s) for s in expected["scales"])
    )
    best = min(record["objective_per_observation"] for record in starts if record["converged"])
    if not math.isclose(objective, best, rel_tol=2e-8, abs_tol=2e-9):
        _error(
            "invalid_state",
            "CFA solver parameters do not reproduce the best converged start objective.",
        )
    cached = state["results"]
    if not isinstance(cached, Mapping) or set(cached) != set(expected):
        _error("invalid_state", "CFA results fields differ from semantic replay.")
    for key, values in expected.items():
        if key == "covariance":
            if not isinstance(cached[key], (list, tuple)) or len(cached[key]) != len(values):
                _error(
                    "invalid_state", "CFA results.covariance shape differs from semantic replay."
                )
            for i, row in enumerate(values):
                if not isinstance(cached[key][i], (list, tuple)) or len(cached[key][i]) != len(row):
                    _error(
                        "invalid_state",
                        "CFA results.covariance shape differs from semantic replay.",
                    )
                for j, value in enumerate(row):
                    unit = math.sqrt(values[i][i]) * math.sqrt(values[j][j])
                    _close(
                        cached[key][i][j],
                        value,
                        f"results.covariance[{i}][{j}]",
                        absolute=unit * 2e-10,
                    )
        else:
            # Solver scores differentiate the dimensionless standardized
            # objective. Their absolute floor never applies to raw-unit results.
            floor = (
                n * 1e-10
                if key == "score_solver"
                else 1e-10
                if key == "score_max_per_observation"
                else 0.0
            )
            _close(cached[key], values, "results." + key, absolute=floor)
    if state["digest"] != _digest({k: v for k, v in state.items() if k != "digest"}):
        _error("invalid_state", "CFA state digest does not match its source and result payload.")
    return state


class CFAState(BaseModel):
    """Immutable, versioned multi-outcome CFA state; validation replays numerics."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    schema_version: Literal["openecon.cfa.v1"] = SCHEMA
    payload: Any

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        _encoded_json_admission(json_data, limit=32 * 1024 * 1024, operation="CFA typed JSON decoding")
        return super().model_validate_json(json_data, **kwargs)

    def __deepcopy__(self, memo=None):
        _state_copy_admission(self, operation="CFA typed state deep copy")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(self.payload)
        return super().__deepcopy__(memo)

    @wraps(BaseModel.model_dump)
    def model_dump(self, **kwargs):
        _state_copy_admission(self, operation="CFA portable state copy")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(self.payload)
        return super().model_dump(**kwargs)

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, **kwargs):
        _json_export_admission(self, indent=kwargs.get("indent"), limit=32 * 1024 * 1024,
                               operation="CFA complete JSON serialization")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(self.payload)
        return super().model_dump_json(**kwargs)

    @model_validator(mode="after")
    def validate_semantics(self):
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(self.payload)
        object.__setattr__(self, "payload", _freeze(self.payload))
        return self

    @field_serializer("payload")
    def serialize_payload(self, value):
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(value)
        return _thaw(value)

    def to_tables(self):
        """Return the validated CFA estimates, full covariance, loadings and implied moments."""
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported CFA schema.")
        _replay(self.payload)
        return _tables(self.payload)

    def __str__(self):
        return str(self.to_tables())


def _tables(state):
    result, spec = state["results"], state["spec"]
    estimates, names = result["parameters"], result["parameter_names"]
    level = state["options"]["level"]
    critical = normal_isf((1 - level) / 2)
    se = [math.sqrt(result["covariance"][i][i]) for i in range(len(names))]
    z = [estimate / s for estimate, s in zip(estimates, se)]
    coefficients = table(
        {
            "estimate": estimates,
            "std_error": se,
            "z": z,
            "p_value": [2 * normal_sf(abs(value)) for value in z],
            "ci_lower": [e - critical * s for e, s in zip(estimates, se)],
            "ci_upper": [e + critical * s for e, s in zip(estimates, se)],
        },
        index=names,
    )
    return TableSet(
        {
            "parameters": coefficients,
            "covariance": table(result["covariance"], columns=names, index=names),
            "loadings": table(
                result["loadings"], columns=spec["factor_names"], index=spec["columns"]
            ),
            "factor_covariance": table(
                result["factor_covariance"],
                columns=spec["factor_names"],
                index=spec["factor_names"],
            ),
            "implied_covariance": table(
                result["implied_covariance"], columns=spec["columns"], index=spec["columns"]
            ),
        },
        title="Continuous confirmatory factor analysis",
        procedure="cfa",
        n=result["n"],
        log_likelihood=result["log_likelihood"],
        df_covariance=result["df_covariance"],
        identification=spec["identification"],
        covariance_method="full observed information; normal ML divisor n",
        inference="iid Gaussian regular interior model; asymptotic normal Wald",
        mean_structure="unrestricted observed means profiled; no latent mean/path structure",
        notes=[
            "Markers are fixed at one and have no fabricated uncertainty."
            if spec["identification"] == "marker"
            else "Latent variances fixed at one; positive first-indicator orientation.",
            "No MAR FIML, robust/survey uncertainty or generalized outcomes in this stage.",
        ],
        cfa_state=_thaw(state),
        resource_plan=state["resource_plan"],
    )


def _fit(spec, means, cov, source, options, resource):
    _spd(cov)
    scales = cov.diag().sqrt()
    standardized = cov / scales[:, None] / scales[None, :]
    x, starts = _solve(spec, scales, standardized, options)
    n = len(source["positions"]) if source["kind"] == "raw" else source["original_n"]
    result = _evaluate(x, spec, cov, n, means, options)
    payload = {
        "schema": SCHEMA,
        "spec": _thaw(_freeze(spec)),
        "source": source,
        "options": options,
        "solver_parameters": x.tolist(),
        "starts": starts,
        "results": result,
        "resource_plan": resource,
    }
    payload["digest"] = _digest(payload)
    bundle = CFAState(payload=payload)
    return bundle.to_tables()


def cfa(
    data,
    *,
    factors,
    identification="marker",
    markers=None,
    missing="raise",
    max_iterations=300,
    tolerance=1e-7,
    max_work=MAX_WORK,
    level=0.95,
):
    """Fit identified continuous CFA on bounded resident real indicators.

    Factor declarations determine indicator/parameter order. Marker loadings are
    fixed at one in original observed units. Unit-variance identification fixes
    latent variances at one; the first declared indicator sets positive sign.
    Complete-case removal is explicit, with original row/index identity saved.
    """
    spec = _spec(factors, identification, markers)
    options = _options(max_iterations, tolerance, max_work, level)
    n = _resident(data, spec["columns"])
    resource = _admit(n, spec, options, raw=True)
    if isinstance(data, pd.DataFrame):
        _index_admission(data.index)
    elif isinstance(data, Mapping):
        indexed = [data[c].index for c in spec["columns"] if isinstance(data[c], pd.Series)]
        if indexed:
            _index_admission(indexed[0])
    if isinstance(data, pd.DataFrame):
        frame = data.loc[:, spec["columns"]].copy()
    else:
        frame = pd.DataFrame({c: data[c] for c in spec["columns"]})
    means, cov, source = _moments(frame, missing)
    return _fit(spec, means, cov, source, options, resource)


def cfa_covariance(
    covariance,
    *,
    factors,
    n,
    divisor,
    means=None,
    identification="marker",
    markers=None,
    max_iterations=300,
    tolerance=1e-7,
    max_work=MAX_WORK,
    level=0.95,
):
    """Fit covariance-summary CFA with a caller-declared n or n-1 divisor.

    Matrix order is the first occurrence of indicators in factor declarations.
    Profiled observed-mean likelihood is normalized identically to raw-data ML;
    omitted means do not create individual scores or latent-mean evidence.
    """
    spec = _spec(factors, identification, markers)
    options = _options(max_iterations, tolerance, max_work, level)
    resource = _admit(n, spec, options)
    p = len(spec["columns"])
    if divisor not in ("n", "n-1"):
        _error("invalid_option", "Declare covariance divisor='n' or 'n-1'.")
    if isinstance(covariance, pd.DataFrame):
        if list(covariance.columns) != spec["columns"] or list(covariance.index) != spec["columns"]:
            _error(
                "invalid_input", "Summary covariance labels/order must match declared indicators."
            )
        covariance = covariance.to_numpy().tolist()
    if (
        not isinstance(covariance, Sequence)
        or isinstance(covariance, (str, bytes))
        or len(covariance) != p
        or any(not isinstance(row, Sequence) or len(row) != p for row in covariance)
    ):
        _error(
            "invalid_input", "Provide a square bounded covariance sequence or labeled DataFrame."
        )
    if any(
        isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v)
        for row in covariance
        for v in row
    ):
        _error("invalid_input", "Covariance entries must be finite real values.")
    mean_values = None
    if means is not None:
        if (
            not isinstance(means, Sequence)
            or isinstance(means, (str, bytes))
            or len(means) != p
            or any(
                isinstance(v, bool) or not isinstance(v, Real) or not _finite_real(v)
                for v in means
            )
        ):
            _error("invalid_input", "Declare one finite observed mean per indicator.")
        mean_values = [float(v) for v in means]
    raw_cov = [[float(v) for v in row] for row in covariance]
    cov = _tensor(raw_cov)
    if divisor == "n-1":
        cov = cov * (n - 1) / n
    source = {
        "kind": "summary",
        "original_n": int(n),
        "covariance": raw_cov,
        "divisor": divisor,
        "means": mean_values,
    }
    return _fit(
        spec, None if mean_values is None else _tensor(mean_values), cov, source, options, resource
    )


def cfa_restore(saved):
    """Validate complete source and joint state without rerunning the optimizer.

    Accept a CFA TableSet, CFAState, state mapping, or CFAState JSON. Checksums
    supplement numeric replay; recomputing a checksum cannot legitimize a forged
    covariance, source sample or likelihood.
    """
    if isinstance(saved, CFAState):
        return CFAState.model_validate(saved.model_dump()).to_tables()
    if isinstance(saved, TableSet):
        if saved.attrs.get("procedure") != "cfa":
            _error("invalid_state", "Pass a CFA result.")
        saved = saved.attrs.get("cfa_state")
    if isinstance(saved, str):
        _encoded_json_admission(saved, limit=32 * 1024 * 1024, operation="CFA public restore decoding")
        saved = json.loads(saved)
    if isinstance(saved, Mapping) and set(saved) == {"schema_version", "payload"}:
        return CFAState.model_validate(saved).to_tables()
    return CFAState(payload=saved).to_tables()
