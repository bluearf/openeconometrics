"""Acyclic continuous Gaussian SEM with joint means, observed information and replay.

Every observed and latent node obeys z = intercept + B z + disturbance. All
observed variables, including exogenous predictors, have an estimated joint
distribution. CPU float64 Torch is the numerical engine; pandas supplies typed
input/output. Missing-case removal is explicit and is not MAR FIML.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from functools import wraps
from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.latent.cfa import (
    DEVICE,
    DTYPE,
    MAX_INDEX_BYTES,
    MAX_ROWS,
    _close,
    _decode_index,
    _digest,
    _error,
    _freeze,
    _index_admission,
    _integer,
    _moments,
    _real as _cfa_real,
    _resident,
    _saved_index_admission,
    _spd,
    _tensor,
    _thaw,
)
from openecon.engines.distributions import chi2_sf, normal_isf, normal_sf
from openecon.econometrics.state_lifecycle import (
    _encoded_json_admission,
    _json_export_admission,
    _state_copy_admission,
)
from openecon.resources import plan_workspace

SCHEMA = "openecon.latent_sem.v1"
MAX_COLUMNS, MAX_FACTORS, MAX_PARAMETERS = 16, 4, 120
MAX_WORK = 10_000_000_000


def _finite_real(value, *, exact_integer=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value) and (
            not exact_integer or not isinstance(value, Integral) or int(float(value)) == value
        )
    except (OverflowError, ValueError):
        return False


def _real(value, name, minimum, maximum):
    if not _finite_real(value, exact_integer=True):
        _error("invalid_option", f"{name} must be finite and representable in float64.")
    return _cfa_real(value, name, minimum, maximum)


def _names(values, label, maximum):
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not 1 <= len(values) <= maximum
        or any(
            not isinstance(v, str)
            or not v
            or len(v) > 128
            or any(token in v for token in (":", ",", "<-"))
            for v in values
        )
        or len(set(values)) != len(values)
    ):
        _error(
            "invalid_specification", f"{label} must be a bounded ordered sequence of unique names."
        )
    return list(values)


def _coefficient(value, label):
    if value is None:
        return None
    return _real(value, label, -1e100, 1e100)


def _spec(
    columns,
    factors,
    paths,
    identification,
    markers,
    residual_covariances,
    intercepts,
    meanstructure,
    fixed,
    equalities,
):
    columns = _names(columns, "columns", MAX_COLUMNS)
    if not isinstance(factors, Mapping) or len(factors) > MAX_FACTORS:
        _error("invalid_specification", "factors must map at most four latent names to indicators.")
    latent = list(factors)
    if latent:
        _names(latent, "factors", MAX_FACTORS)
    if set(columns) & set(latent):
        _error("invalid_specification", "Observed and latent names must be distinct.")
    factors = {f: _names(v, "indicators", MAX_COLUMNS) for f, v in factors.items()}
    if any(len(v) < 2 or not set(v) <= set(columns) for v in factors.values()):
        _error(
            "invalid_specification",
            "Each latent factor needs at least two declared observed indicators.",
        )
    if identification not in ("marker", "unit_disturbance"):
        _error("invalid_specification", "identification must be marker or unit_disturbance.")
    if markers is not None and (
        identification != "marker"
        or not isinstance(markers, Mapping)
        or set(markers) != set(latent)
    ):
        _error(
            "invalid_specification", "markers must specify every marker-identified latent factor."
        )
    anchors = {f: factors[f][0] if markers is None else markers[f] for f in latent}
    if any(anchors[f] not in factors[f] for f in latent) or len(set(anchors.values())) != len(
        latent
    ):
        _error("invalid_specification", "Latent anchors must be distinct declared indicators.")
    if not isinstance(meanstructure, bool):
        _error("invalid_option", "meanstructure must be Boolean.")
    nodes = columns + latent
    records = []

    def add(kind, i, j=None, value=None, *, locked=False):
        prefix = "loading" if kind == "loading" else "path" if kind == "path" else kind
        name = (
            f"{prefix}:{nodes[i]}<-{nodes[j]}"
            if j is not None and kind in ("loading", "path")
            else (f"{prefix}:{nodes[i]},{nodes[j]}" if j is not None else f"{prefix}:{nodes[i]}")
        )
        records.append(
            {"name": name, "kind": kind, "i": i, "j": j, "fixed": value, "locked": locked}
        )

    edges = set()
    for f in latent:
        for col in factors[f]:
            i, j = nodes.index(col), nodes.index(f)
            add(
                "loading",
                i,
                j,
                1.0 if identification == "marker" and col == anchors[f] else None,
                locked=identification == "marker" and col == anchors[f],
            )
            edges.add((i, j))
    paths = {} if paths is None else paths
    if not isinstance(paths, Mapping):
        _error("invalid_specification", "paths must map target nodes to source/value mappings.")
    for target, sources in paths.items():
        if target not in nodes or not isinstance(sources, Mapping):
            _error(
                "invalid_specification", "Every path target and source must name a declared node."
            )
        for source, value in sources.items():
            if source not in nodes or source == target:
                _error("invalid_specification", "Paths need distinct declared target/source nodes.")
            i, j = nodes.index(target), nodes.index(source)
            if (i, j) in edges:
                _error("invalid_specification", "A path duplicates a declared measurement loading.")
            add("path", i, j, _coefficient(value, "fixed path"))
            edges.add((i, j))
    for i in range(len(nodes)):
        lock = identification == "unit_disturbance" and i >= len(columns)
        add("variance", i, value=1.0 if lock else None, locked=lock)
    residual_covariances = {} if residual_covariances is None else residual_covariances
    if not isinstance(residual_covariances, Mapping):
        _error(
            "invalid_specification",
            "residual_covariances must map node-name pairs to free/fixed values.",
        )
    seen = set()
    for pair, value in residual_covariances.items():
        if not isinstance(pair, tuple) or len(pair) != 2 or any(v not in nodes for v in pair):
            _error(
                "invalid_specification",
                "Residual covariance keys must be pairs of declared node names.",
            )
        i, j = sorted([nodes.index(v) for v in pair])
        if i == j or (i, j) in seen:
            _error(
                "invalid_specification",
                "Residual covariance pairs must be distinct, without diagonals.",
            )
        seen.add((i, j))
        add("covariance", i, j, _coefficient(value, "fixed covariance"))
    intercepts = {} if intercepts is None else intercepts
    if not isinstance(intercepts, Mapping) or not set(intercepts) <= set(nodes):
        _error("invalid_specification", "intercepts must map declared nodes to free/fixed values.")
    if not meanstructure and intercepts:
        _error("invalid_specification", "Explicit intercepts require meanstructure=True.")
    if meanstructure:
        for i, node in enumerate(nodes):
            value = intercepts.get(node, None if i < len(columns) else 0.0)
            add("intercept", i, value=_coefficient(value, "fixed intercept"))
    fixed = {} if fixed is None else fixed
    equalities = {} if equalities is None else equalities
    by_name = {r["name"]: r for r in records}
    if (
        not isinstance(fixed, Mapping)
        or not isinstance(equalities, Mapping)
        or (not set(fixed) <= set(by_name) or not set(equalities) <= set(by_name))
    ):
        _error(
            "invalid_specification", "fixed/equalities must refer to declared full parameter names."
        )
    for name, value in fixed.items():
        if by_name[name]["locked"]:
            _error(
                "invalid_specification",
                "Identification anchors cannot be overridden by fixed parameters.",
            )
        by_name[name]["fixed"] = _real(value, "fixed parameter", -1e100, 1e100)
    groups, labels = [], {}
    for record_index, r in enumerate(records):
        if r["kind"] == "variance" and r["fixed"] is not None and r["fixed"] <= 0:
            _error("boundary_fit", "Fixed disturbance variances must be strictly positive.")
        label = equalities.get(r["name"])
        if label is not None and (
            r["fixed"] is not None or not isinstance(label, str) or not label or len(label) > 128
        ):
            _error(
                "invalid_specification",
                "Equality labels need free parameters and bounded nonempty names.",
            )
        if r["fixed"] is not None:
            r["group"] = None
        else:
            family = "path" if r["kind"] in ("loading", "path") else r["kind"]
            key = r["name"] if label is None else "equality:" + label
            if key not in labels:
                labels[key] = len(groups)
                groups.append({"name": key, "kind": family, "representative": record_index})
            elif groups[labels[key]]["kind"] != family:
                _error(
                    "invalid_specification",
                    "Equality labels cannot mix paths, variances, covariances and intercepts.",
                )
            r["group"] = labels[key]
        r.pop("locked")
    # Every potentially nonzero edge participates in the acyclic model contract.
    remaining = set(range(len(nodes)))
    ordered = []
    while remaining:
        roots = [
            i
            for i in sorted(remaining)
            if not any(
                r["kind"] in ("loading", "path")
                and r["i"] == i
                and r["j"] in remaining
                and r["fixed"] != 0.0
                for r in records
            )
        ]
        if not roots:
            _error(
                "cyclic_model", "Continuous SEM currently requires an acyclic declared path graph."
            )
        ordered.extend(roots)
        remaining.difference_update(roots)
    q = len(groups)
    moments = len(columns) * (len(columns) + 1) // 2 + (len(columns) if meanstructure else 0)
    if not 1 <= q <= min(MAX_PARAMETERS, moments):
        _error(
            "not_identified",
            "Free parameters exceed available mean/covariance moments or resident bounds.",
        )
    return {
        "columns": columns,
        "latent": latent,
        "nodes": nodes,
        "factors": factors,
        "anchors": anchors,
        "identification": identification,
        "meanstructure": meanstructure,
        "records": records,
        "groups": groups,
        "topological_order": ordered,
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
    p, m, q = len(spec["columns"]), len(spec["nodes"]), spec["parameter_count"]
    records = len(spec["records"])
    work = n * p * p + (options["max_iterations"] * 60 + q * q * 80 + 200) * m**3
    if work > options["max_work"]:
        _error(
            "work_limit",
            f"SEM declares {work:,} scalar-work units; increase max_work within its bound.",
        )
    plan = plan_workspace(
        "continuous_latent_sem",
        {
            "source_centered_and_scores": 8 * n * (p * 5 + m * 2),
            "matrices_and_joint_state": 8 * (m * m * 30 + q * q * 12 + records * records * 4),
            "autograd_information": 8 * q * q * m * m * 4,
            "jacobians_and_optimizer": 8 * (q * m * m * 12 + q * q * 10),
            "typed_index": MAX_INDEX_BYTES * 3 if raw else 0,
        },
    )
    return {**plan.record(), "estimated_work": work, "max_work": options["max_work"]}


def _units(spec, scales):
    latent = [
        scales[spec["columns"].index(spec["anchors"][f])]
        if spec["identification"] == "marker"
        else _tensor(1.0)
        for f in spec["latent"]
    ]
    return torch.cat([scales, torch.stack(latent)]) if latent else scales


def _parameter_unit(record, units):
    kind, i, j = record["kind"], record["i"], record["j"]
    if kind in ("loading", "path"):
        return units[i] / units[j]
    if kind == "variance":
        return units[i].square()
    if kind == "covariance":
        return units[i] * units[j]
    return units[i]


def _decode(x, spec, scales):
    units = _units(spec, scales)
    m, p = len(units), len(scales)
    group_values = torch.stack(
        [
            (x[g].exp() if group["kind"] == "variance" else x[g])
            * _parameter_unit(spec["records"][group["representative"]], units)
            for g, group in enumerate(spec["groups"])
        ]
    )
    b = torch.zeros((m, m), dtype=DTYPE, device=DEVICE)
    psi = torch.zeros_like(b)
    intercept = torch.zeros(m, dtype=DTYPE, device=DEVICE)
    for r in spec["records"]:
        raw = _tensor(r["fixed"]) if r["group"] is None else group_values[r["group"]]
        value = raw / _parameter_unit(r, units)
        i, j, kind = r["i"], r["j"], r["kind"]
        if kind in ("loading", "path"):
            b[i, j] = value
        elif kind == "intercept":
            intercept[i] = value
        else:
            psi[i, i if j is None else j] = value
            if j is not None:
                psi[j, i] = value
    inverse = torch.linalg.solve(
        torch.eye(m, dtype=DTYPE, device=DEVICE) - b, torch.eye(m, dtype=DTYPE, device=DEVICE)
    )
    omega = inverse @ psi @ inverse.T
    mean = inverse @ intercept
    raw_b = b * units[:, None] / units[None, :]
    raw_psi = psi * units[:, None] * units[None, :]
    raw_omega = omega * units[:, None] * units[None, :]
    raw_mean = mean * units
    raw_intercept = intercept * units
    total_scales = omega.diag().sqrt()
    standardized = []
    for r in spec["records"]:
        i, j, kind = r["i"], r["j"], r["kind"]
        if kind in ("loading", "path"):
            standardized.append(b[i, j] * total_scales[j] / total_scales[i])
        elif kind == "intercept":
            standardized.append(intercept[i] / total_scales[i])
        elif kind == "variance" and not any(
            edge["kind"] in ("loading", "path") and edge["i"] == i and edge["fixed"] != 0.0
            for edge in spec["records"]
        ):
            # A root node's standardized variance is identically one. Preserve
            # its zero derivative instead of quotient-roundoff Wald inference.
            standardized.append(_tensor(1.0))
        else:
            standardized.append(
                psi[i, i if j is None else j]
                / total_scales[i]
                / total_scales[i if j is None else j]
            )
    return {
        "sigma": omega[:p, :p],
        "mean": mean[:p],
        "psi": psi,
        "omega": omega,
        "b": b,
        "natural": group_values,
        "standardized": torch.stack(standardized),
        "raw_b": raw_b,
        "raw_psi": raw_psi,
        "raw_omega": raw_omega,
        "raw_mean": raw_mean,
        "raw_intercept": raw_intercept,
    }


def _objective(x, spec, scales, cov, means):
    decoded = _decode(x, spec, scales)
    _, info = torch.linalg.cholesky_ex(decoded["psi"])
    if int(info) or not torch.isfinite(decoded["psi"]).all():
        _error(
            "inadmissible_covariance",
            "Declared disturbance covariance is not strictly positive definite.",
        )
    chol = torch.linalg.cholesky(decoded["sigma"])
    discrepancy = torch.trace(torch.cholesky_solve(cov, chol))
    if spec["meanstructure"]:
        diff = means - decoded["mean"]
        discrepancy = discrepancy + diff @ torch.cholesky_solve(diff[:, None], chol)[:, 0]
    return 0.5 * (2 * torch.log(chol.diag()).sum() + discrepancy)


def _start(spec, scales, means, amount):
    units = _units(spec, scales)
    values = []
    for group in spec["groups"]:
        r = spec["records"][group["representative"]]
        kind = group["kind"]
        if kind == "variance":
            values.append(math.log(0.5 if r["i"] < len(scales) else 0.7))
        elif kind == "path":
            values.append(amount if r["kind"] == "loading" else 0.1)
        elif kind == "intercept":
            values.append(
                0.0
                if r["i"] >= len(scales) or means is None
                else float(means[r["i"]] * scales[r["i"]] / units[r["i"]])
            )
        else:
            values.append(0.0)
    return _tensor(values)


def _solve(spec, scales, cov, means, options):
    accepted, records = [], []
    for start_id, amount in enumerate((0.7, 0.4)):
        x = _start(spec, scales, means, amount)
        h = torch.eye(len(x), dtype=DTYPE, device=DEVICE)
        evaluations, iterations, newton_iterations = 0, 0, 0
        record = {"start": start_id, "initial_loading": amount, "variance_rescalings": 0}
        try:
            # Fixed selected covariances can exceed the generic starting
            # variances. Inflate only free variances until the declared sparse
            # disturbance matrix is admissible; fixed constraints stay exact.
            variance_groups = [i for i, g in enumerate(spec["groups"]) if g["kind"] == "variance"]
            for rescalings in range(31):
                evaluations += 1
                _, admissible = torch.linalg.cholesky_ex(_decode(x, spec, scales)["psi"])
                record["variance_rescalings"] = rescalings
                if int(admissible) == 0:
                    break
                if rescalings == 30 or not variance_groups:
                    _error(
                        "inadmissible_covariance",
                        "Fixed disturbance covariances have no admitted deterministic start.",
                    )
                x[variance_groups] += math.log(2)
            for iteration in range(options["max_iterations"] + 1):
                current = x.detach().requires_grad_(True)
                objective = _objective(current, spec, scales, cov, means)
                evaluations += 1
                gradient = torch.autograd.grad(objective, current)[0].detach()
                score = float(gradient.abs().max())
                if score <= options["tolerance"]:
                    break
                if iteration == options["max_iterations"]:
                    _error(
                        "nonconvergence",
                        "SEM exhausted its declared iteration budget before score convergence.",
                    )
                # Exact Newton near the optimum; BFGS otherwise. All line-search
                # candidates must remain in the declared covariance domain.
                direction = -h @ gradient
                if score < 0.02 and newton_iterations < 20:
                    information = torch.autograd.functional.hessian(
                        lambda v: _objective(v, spec, scales, cov, means), current
                    )
                    newton_iterations += 1
                    if float(torch.linalg.eigvalsh(information).min()) > 1e-10:
                        direction = -torch.linalg.solve(information, gradient)
                if not torch.isfinite(direction).all() or float(direction @ gradient) >= 0:
                    h = torch.eye(len(x), dtype=DTYPE, device=DEVICE)
                    direction = -gradient
                moved = False
                for power in range(24):
                    candidate = x + direction / 2**power
                    evaluations += 1
                    try:
                        trial = _objective(candidate, spec, scales, cov, means)
                    except (AnalysisError, RuntimeError):
                        continue
                    if (
                        torch.isfinite(trial)
                        and float(trial)
                        <= float(objective.detach()) + 1e-4 * float(gradient @ direction) / 2**power
                    ):
                        moved = True
                        break
                if not moved:
                    _error(
                        "nonconvergence",
                        "SEM admissible line search failed before score convergence.",
                    )
                updated = candidate.detach().requires_grad_(True)
                new_gradient = torch.autograd.grad(
                    _objective(updated, spec, scales, cov, means), updated
                )[0].detach()
                evaluations += 1
                step, change = candidate - x, new_gradient - gradient
                curvature = float(change @ step)
                if curvature > 1e-14 * float(
                    torch.linalg.vector_norm(change) * torch.linalg.vector_norm(step)
                ):
                    rho = 1.0 / curvature
                    eye = torch.eye(len(x), dtype=DTYPE, device=DEVICE)
                    left = eye - rho * step[:, None] @ change[None, :]
                    h = left @ h @ left.T + rho * step[:, None] @ step[None, :]
                x = candidate.detach()
                iterations += 1
            record.update(
                evaluations=evaluations,
                iterations=iterations,
                objective_per_observation=float(objective.detach()),
                score_max=score,
                converged=True,
                failure=None,
            )
            accepted.append((float(objective.detach()), x))
        except (AnalysisError, RuntimeError) as exc:
            record.update(
                evaluations=evaluations,
                iterations=iterations,
                converged=False,
                failure=str(exc)[:600],
            )
        records.append(record)
    if not accepted:
        _error(
            "nonconvergence",
            f"Neither deterministic SEM start passed admission and score convergence: {records}.",
        )
    accepted.sort(key=lambda item: item[0])
    return accepted[0][1], records


def _diagnostics(decoded, sample, means, spec, n, loss, scales):
    p, q = len(scales), spec["parameter_count"]
    chol = torch.linalg.cholesky(sample)
    saturated = 0.5 * (2 * torch.log(chol.diag()).sum() + p)
    discrepancy = float(2 * (loss.detach() - saturated))
    if discrepancy < -1e-9:
        _error("numerical_failure", "SEM fitted likelihood exceeds its saturated reference.")
    discrepancy = max(discrepancy, 0.0)
    df = p * (p + 1) // 2 + (p if spec["meanstructure"] else 0) - q
    statistic = n * discrepancy
    baseline_statistic = max(0.0, -n * float(2 * torch.log(chol.diag()).sum()))
    baseline_df = p * (p - 1) // 2
    excess, baseline_excess = max(statistic - df, 0.0), baseline_statistic - baseline_df
    denominator = max(excess, baseline_excess, 0.0)
    cfi = 1 - excess / denominator if denominator > 0 else 1.0
    tli_numerator = (statistic - df) * baseline_df
    tli_denominator = (baseline_statistic - baseline_df) * df
    tli = 1 - tli_numerator / tli_denominator if df > 0 and tli_denominator != 0 else 1.0
    rmsea = math.sqrt(max(statistic - df, 0.0) / (n * df)) if df > 0 else 0.0
    lower = torch.tril_indices(p, p, device=DEVICE)
    residual = sample - decoded["sigma"]
    covariance_squares = residual[lower[0], lower[1]].square()
    covariance_srmr = float(torch.sqrt(covariance_squares.mean()).detach())
    mean_squares = (means - decoded["mean"]).square() if spec["meanstructure"] else None
    srmr = (
        float(
            torch.sqrt(
                (covariance_squares.sum() + mean_squares.sum()) / (covariance_squares.numel() + p)
            ).detach()
        )
        if mean_squares is not None
        else covariance_srmr
    )
    mean_rms = float(torch.sqrt(mean_squares.mean()).detach()) if mean_squares is not None else None
    return {
        "statistic": statistic,
        "df": df,
        "p_value": chi2_sf(statistic, df) if df > 0 else None,
        "discrepancy": discrepancy,
        "baseline_statistic": baseline_statistic,
        "baseline_df": baseline_df,
        "baseline_covariance": torch.diag(scales.square()).tolist(),
        "baseline_means": None if means is None else (means * scales).tolist(),
        "baseline_parameters": 2 * p if spec["meanstructure"] else p,
        "CFI": cfi,
        "TLI": tli,
        "RMSEA": rmsea,
        "SRMR": srmr,
        "SRMR_covariance": covariance_srmr,
        "standardized_mean_residual_rms": mean_rms,
        "SRMR_convention": "sample-SD standardized lower covariance triangle including diagonal, plus modeled mean residuals",
        "test_convention": "normal ML: n times mean/covariance discrepancy; independence baseline with free observed means",
    }


def _evaluate(x, spec, cov, means, n, options):
    _spd(cov)
    scales = cov.diag().sqrt()
    sample = cov / scales[:, None] / scales[None, :]
    normalized_means = None if means is None else means / scales
    x = x.detach().requires_grad_(True)
    decoded = _decode(x, spec, scales)
    loss = _objective(x, spec, scales, sample, normalized_means)
    gradient = torch.autograd.grad(loss, x)[0]
    score = float(gradient.abs().max())
    if score > options["tolerance"] * 1.01:
        _error(
            "nonconvergence",
            "SEM saved solution does not pass its declared per-observation score tolerance.",
        )
    psi, omega = decoded["psi"], decoded["omega"]
    correlation = psi / psi.diag().sqrt()[:, None] / psi.diag().sqrt()[None, :]
    eigen = torch.linalg.eigvalsh(correlation)
    if (
        float((psi.diag() / omega.diag()).min().detach()) <= 1e-8
        or float((eigen.min() / eigen.max()).detach()) <= 1e-10
    ):
        _error(
            "boundary_fit",
            "SEM reached a disturbance covariance/Heywood boundary; ordinary OIM inference is refused.",
        )
    if spec["identification"] == "unit_disturbance" and any(
        float(
            decoded["b"][spec["nodes"].index(spec["anchors"][f]), spec["nodes"].index(f)].detach()
        )
        <= 0
        for f in spec["latent"]
    ):
        _error(
            "orientation_failure", "Unit-disturbance factors require positive anchor orientation."
        )
    p = len(scales)
    lower = torch.tril_indices(p, p, device=DEVICE)

    def moments(v):
        d = _decode(v, spec, scales)
        values = d["sigma"][lower[0], lower[1]]
        return torch.cat([values, d["mean"]]) if spec["meanstructure"] else values

    moment_jacobian = torch.autograd.functional.jacobian(moments, x)
    singular = torch.linalg.svdvals(moment_jacobian)
    if len(singular) < len(x) or float(singular.min() / singular.max()) <= 1e-9:
        _error(
            "not_identified",
            "SEM mean/covariance moment Jacobian is rank deficient or numerically weak.",
        )
    information = n * torch.autograd.functional.hessian(
        lambda v: _objective(v, spec, scales, sample, normalized_means), x
    )
    information = (information + information.T) / 2
    eig = torch.linalg.eigvalsh(information)
    if not torch.isfinite(information).all() or float(eig.min() / eig.max()) <= 1e-10:
        _error(
            "singular_information", "SEM full observed information is not stably positive definite."
        )
    jac = torch.autograd.functional.jacobian(lambda v: _decode(v, spec, scales)["natural"], x)
    standardized_jac = torch.autograd.functional.jacobian(
        lambda v: _decode(v, spec, scales)["standardized"], x
    )
    covariance = jac @ torch.linalg.solve(information, jac.T)
    covariance = (covariance + covariance.T) / 2
    standard_cov = standardized_jac @ torch.linalg.solve(information, standardized_jac.T)
    standard_cov = (standard_cov + standard_cov.T) / 2
    if (
        not torch.isfinite(covariance).all()
        or bool((covariance.diag() <= 0).any())
        or not torch.isfinite(standard_cov).all()
    ):
        _error(
            "numerical_failure",
            "SEM original-unit joint uncertainty is not representable; choose suitable units.",
        )
    log_likelihood = -n * (
        float(loss.detach())
        + 0.5 * (p * math.log(2 * math.pi) + float(torch.log(scales.square()).sum()))
    )
    raw_sigma = decoded["sigma"] * scales[:, None] * scales[None, :]
    total_scales = decoded["omega"].diag().sqrt()
    standardized_paths = decoded["b"] * total_scales[None, :] / total_scales[:, None]
    return {
        "parameter_names": [g["name"] for g in spec["groups"]],
        "standardized_parameter_names": [r["name"] for r in spec["records"]],
        "parameters": decoded["natural"].detach().tolist(),
        "covariance": covariance.detach().tolist(),
        "standardized_parameters": decoded["standardized"].detach().tolist(),
        "standardized_covariance": standard_cov.detach().tolist(),
        "paths": decoded["raw_b"].detach().tolist(),
        "standardized_paths": standardized_paths.detach().tolist(),
        "disturbance_covariance": decoded["raw_psi"].detach().tolist(),
        "node_covariance": decoded["raw_omega"].detach().tolist(),
        "intercepts": decoded["raw_intercept"].detach().tolist() if spec["meanstructure"] else None,
        "node_means": decoded["raw_mean"].detach().tolist() if spec["meanstructure"] else None,
        "implied_covariance": raw_sigma.detach().tolist(),
        "sample_covariance_ml": cov.tolist(),
        "sample_means": None if means is None else means.tolist(),
        "natural_jacobian": jac.detach().tolist(),
        "standardized_jacobian": standardized_jac.detach().tolist(),
        "observed_information_solver": information.detach().tolist(),
        "moment_jacobian_singular_values": singular.detach().tolist(),
        "information_eigenvalues": eig.detach().tolist(),
        "score_solver": (-n * gradient).detach().tolist(),
        "score_max_per_observation": score,
        "scales": scales.tolist(),
        "n": n,
        "log_likelihood": log_likelihood,
        "diagnostics": _diagnostics(decoded, sample, normalized_means, spec, n, loss, scales),
    }


def _rebuild_spec(saved):
    fields = {
        "columns",
        "latent",
        "nodes",
        "factors",
        "anchors",
        "identification",
        "meanstructure",
        "records",
        "groups",
        "topological_order",
        "parameter_count",
    }
    if (
        not isinstance(saved, Mapping)
        or set(saved) != fields
        or not isinstance(saved.get("records"), (list, tuple))
    ):
        _error("invalid_state", "SEM saved specification is incomplete.")
    columns = _names(saved["columns"], "saved columns", MAX_COLUMNS)
    nodes = _names(saved["nodes"], "saved nodes", MAX_COLUMNS + MAX_FACTORS)
    if not isinstance(nodes, (list, tuple)) or len(nodes) > MAX_COLUMNS + MAX_FACTORS:
        _error("invalid_state", "SEM saved nodes exceed its declared resident bounds.")
    if not isinstance(saved["records"], (list, tuple)) or len(saved["records"]) > 500:
        _error("invalid_state", "SEM saved parameter map exceeds its resident bounds.")
    if not isinstance(saved["anchors"], Mapping):
        _error("invalid_state", "SEM saved anchor mapping is invalid.")
    paths, residuals, intercepts, fixed, equalities = {}, {}, {}, {}, {}
    groups = saved.get("groups")
    if not isinstance(groups, (list, tuple)) or not 1 <= len(groups) <= MAX_PARAMETERS:
        _error("invalid_state", "SEM saved free-parameter map is invalid.")
    for r in saved["records"]:
        if not isinstance(r, Mapping) or set(r) != {"name", "kind", "i", "j", "fixed", "group"}:
            _error("invalid_state", "SEM saved parameter record is invalid.")
        i = _integer(r["i"], "saved target", 0, len(nodes) - 1)
        j = None if r["j"] is None else _integer(r["j"], "saved source", 0, len(nodes) - 1)
        if r["kind"] == "path":
            if j is None:
                _error("invalid_state", "SEM path lacks a source node.")
            paths.setdefault(nodes[i], {})[nodes[j]] = r["fixed"]
        elif r["kind"] == "covariance":
            if j is None:
                _error("invalid_state", "SEM covariance lacks its second node.")
            residuals[(nodes[i], nodes[j])] = r["fixed"]
        elif r["kind"] == "intercept":
            intercepts[nodes[i]] = r["fixed"]
        elif r["kind"] not in ("loading", "variance"):
            _error("invalid_state", "Unknown SEM parameter family.")
        if r["group"] is not None:
            g = _integer(r["group"], "saved free group", 0, len(groups) - 1)
            group = groups[g]
            if not isinstance(group, Mapping) or not isinstance(group.get("name"), str):
                _error("invalid_state", "SEM saved equality group is invalid.")
            if group["name"].startswith("equality:"):
                equalities[r["name"]] = group["name"][9:]
        elif r["kind"] == "loading":
            anchor = saved.get("anchors", {}).get(nodes[j]) if j is not None else None
            if saved.get("identification") != "marker" or nodes[i] != anchor:
                fixed[r["name"]] = r["fixed"]
        elif r["kind"] == "variance":
            if saved.get("identification") != "unit_disturbance" or i < len(columns):
                fixed[r["name"]] = r["fixed"]
    latent = saved.get("latent")
    factors = saved.get("factors")
    if (
        not isinstance(latent, (list, tuple))
        or not isinstance(factors, Mapping)
        or set(latent) != set(factors)
    ):
        _error("invalid_state", "SEM ordered factor declarations are invalid.")
    spec = _spec(
        columns,
        {name: factors[name] for name in latent},
        paths,
        saved.get("identification"),
        saved.get("anchors") if saved.get("identification") == "marker" else None,
        residuals,
        intercepts,
        saved.get("meanstructure"),
        fixed,
        equalities,
    )
    _close(saved, _thaw(_freeze(spec)), "spec")
    return spec


def _source_moments(source, spec):
    p = len(spec["columns"])
    n = _integer(source.get("original_n"), "original_n", p + 1, MAX_ROWS)
    if source.get("kind") == "raw":
        if set(source) != {"kind", "rows", "original_n", "index", "dtypes", "positions", "missing"}:
            _error("invalid_state", "SEM raw source has unknown or missing fields.")
        rows = source["rows"]
        if (
            not isinstance(rows, (list, tuple))
            or len(rows) != n
            or any(not isinstance(row, (list, tuple)) or len(row) != p for row in rows)
        ):
            _error("invalid_state", "SEM raw-source dimensions differ from its specification.")
        if any(
            v is not None and not _finite_real(v, exact_integer=True) for row in rows for v in row
        ):
            _error(
                "invalid_state",
                "SEM raw source requires finite numeric values or null missing values.",
            )
        if not isinstance(source["dtypes"], (list, tuple)) or len(source["dtypes"]) != p:
            _error("invalid_state", "SEM raw-source dtypes are incomplete.")
        _saved_index_admission(source["index"], n)
        index = _decode_index(source["index"])
        series = {}
        for i, (name, dtype) in enumerate(zip(spec["columns"], source["dtypes"])):
            if not isinstance(dtype, str) or len(dtype) > 80:
                _error("invalid_state", "SEM numeric dtype descriptor is invalid.")
            try:
                series[name] = pd.Series([row[i] for row in rows], index=index, dtype=dtype)
                recovered = series[name].astype("float64").tolist()
            except (ValueError, TypeError, OverflowError):
                _error(
                    "invalid_state",
                    "SEM source cannot be represented in its declared original dtype.",
                )
            if any(
                row[i] is not None and (not math.isfinite(r) or row[i] != r)
                for row, r in zip(rows, recovered)
            ):
                _error("invalid_state", "SEM source values disagree with their original dtype.")
        means, cov, rebuilt = _moments(pd.DataFrame(series, index=index), source["missing"])
        _close(source["positions"], rebuilt["positions"], "source.positions")
        return means, cov, len(rebuilt["positions"])
    if source.get("kind") != "summary" or set(source) != {
        "kind",
        "original_n",
        "covariance",
        "divisor",
        "means",
    }:
        _error("invalid_state", "SEM summary source has unknown or missing fields.")
    cov, means, _ = _summary_values(
        source["covariance"], source["means"], spec, n, source["divisor"]
    )
    return means, cov, n


def _cached_admission(cached, spec):
    """Bound all cached geometry and primitives before tensor/AD reconstruction."""
    p, m, q, s = (
        len(spec["columns"]),
        len(spec["nodes"]),
        spec["parameter_count"],
        len(spec["records"]),
    )
    matrices = {
        "covariance": (q, q),
        "standardized_covariance": (s, s),
        "paths": (m, m),
        "standardized_paths": (m, m),
        "disturbance_covariance": (m, m),
        "node_covariance": (m, m),
        "implied_covariance": (p, p),
        "sample_covariance_ml": (p, p),
        "natural_jacobian": (q, q),
        "standardized_jacobian": (s, q),
        "observed_information_solver": (q, q),
    }
    vectors = {
        "parameters": q,
        "standardized_parameters": s,
        "moment_jacobian_singular_values": q,
        "information_eigenvalues": q,
        "score_solver": q,
        "scales": p,
    }
    fields = (
        set(matrices)
        | set(vectors)
        | {
            "parameter_names",
            "standardized_parameter_names",
            "intercepts",
            "node_means",
            "sample_means",
            "n",
            "score_max_per_observation",
            "log_likelihood",
            "diagnostics",
        }
    )
    if not isinstance(cached, Mapping) or set(cached) != fields:
        _error("invalid_state", "SEM cached numerical result fields are incomplete.")

    def vector(value, length, name):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != length
            or any(not _finite_real(v, exact_integer=True) for v in value)
        ):
            _error("invalid_state", f"SEM cached {name} needs a bounded finite numeric vector.")

    def matrix(value, shape, name):
        if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
            _error("invalid_state", f"SEM cached {name} shape differs from its declaration.")
        for row in value:
            vector(row, shape[1], name)

    for name, shape in matrices.items():
        matrix(cached[name], shape, name)
    for name, length in vectors.items():
        vector(cached[name], length, name)
    for field, expected in (
        ("parameter_names", [g["name"] for g in spec["groups"]]),
        ("standardized_parameter_names", [r["name"] for r in spec["records"]]),
    ):
        value = cached[field]
        if (
            not isinstance(value, (list, tuple))
            or len(value) != len(expected)
            or any(type(a) is not str or a != b for a, b in zip(value, expected))
        ):
            _error("invalid_state", "SEM cached parameter names differ from the declared groups.")
    for field in ("intercepts", "node_means"):
        if spec["meanstructure"]:
            vector(cached[field], m, field)
        elif cached[field] is not None:
            _error("invalid_state", "SEM covariance-only cache cannot assert latent means.")
    if cached["sample_means"] is not None:
        vector(cached["sample_means"], p, "sample_means")
    _integer(cached["n"], "cached n", 1, MAX_ROWS)
    vector([cached["score_max_per_observation"], cached["log_likelihood"]], 2, "likelihood/score")
    diagnostics = cached["diagnostics"]
    numerical = {
        "statistic",
        "p_value",
        "discrepancy",
        "baseline_statistic",
        "CFI",
        "TLI",
        "RMSEA",
        "SRMR",
        "SRMR_covariance",
        "standardized_mean_residual_rms",
    }
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != numerical | {
        "df",
        "baseline_df",
        "baseline_covariance",
        "baseline_means",
        "baseline_parameters",
        "SRMR_convention",
        "test_convention",
    }:
        _error("invalid_state", "SEM cached diagnostics schema is invalid.")
    for field in numerical:
        nullable = field == "p_value" or (
            field == "standardized_mean_residual_rms" and not spec["meanstructure"]
        )
        if diagnostics[field] is None and not nullable:
            _error("invalid_state", "SEM cached ordinary fit indices must be finite values.")
        if diagnostics[field] is not None:
            vector([diagnostics[field]], 1, "diagnostics." + field)
    for field in ("df", "baseline_df", "baseline_parameters"):
        _integer(diagnostics[field], "diagnostics." + field, 0, p * (p + 3) // 2)
    matrix(diagnostics["baseline_covariance"], (p, p), "baseline_covariance")
    if diagnostics["baseline_means"] is not None:
        vector(diagnostics["baseline_means"], p, "baseline_means")
    for field in ("SRMR_convention", "test_convention"):
        if type(diagnostics[field]) is not str or len(diagnostics[field]) > 256:
            _error("invalid_state", "SEM cached diagnostic conventions need bounded text.")


def _replay(payload):
    keys = {
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
    if not isinstance(payload, Mapping) or set(payload) != keys or payload.get("schema") != SCHEMA:
        _error("invalid_state", "Unknown or incomplete SEM saved-state contract.")
    spec = _rebuild_spec(payload["spec"])
    options = payload["options"]
    if not isinstance(options, Mapping) or set(options) != {
        "max_iterations",
        "tolerance",
        "max_work",
        "level",
    }:
        _error("invalid_state", "SEM saved options are invalid.")
    options = _options(**options)
    source = payload["source"]
    if not isinstance(source, Mapping):
        _error("invalid_state", "SEM source is invalid.")
    # Work/workspace and source dimensions are admitted before tensor allocation,
    # pandas reconstruction, autodiff or cached-matrix conversion.
    resource = _admit(source.get("original_n"), spec, options, raw=source.get("kind") == "raw")
    cached_resource = payload["resource_plan"]
    if not isinstance(cached_resource, Mapping) or set(cached_resource) != set(resource):
        _error("invalid_state", "SEM resource plan is invalid.")
    _integer(
        cached_resource["budget_bytes"],
        "saved budget_bytes",
        resource["estimated_workspace_bytes"],
        2**63 - 1,
    )
    _close(
        {k: v for k, v in cached_resource.items() if k != "budget_bytes"},
        {k: v for k, v in resource.items() if k != "budget_bytes"},
        "resource_plan",
    )
    _cached_admission(payload["results"], spec)
    means, cov, n = _source_moments(source, spec)
    if spec["meanstructure"] and means is None:
        _error("invalid_state", "SEM mean structure needs saved observed means.")
    solver = payload["solver_parameters"]
    if (
        not isinstance(solver, (list, tuple))
        or len(solver) != spec["parameter_count"]
        or any(not _finite_real(v, exact_integer=True) or abs(v) > 1e100 for v in solver)
    ):
        _error("invalid_state", "SEM solver parameters are invalid.")
    starts = payload["starts"]
    if not isinstance(starts, (list, tuple)) or len(starts) != 2:
        _error("invalid_state", "SEM retains both declared deterministic optimizer starts.")
    objectives = []
    for i, r in enumerate(starts):
        minimal = {
            "start",
            "initial_loading",
            "variance_rescalings",
            "evaluations",
            "iterations",
            "converged",
            "failure",
        }
        complete = minimal | {"objective_per_observation", "score_max"}
        if (
            not isinstance(r, Mapping)
            or set(r) not in (minimal, complete)
            or type(r.get("start")) is not int
            or r["start"] != i
            or type(r.get("initial_loading")) is not float
            or r["initial_loading"] != (0.7, 0.4)[i]
            or not isinstance(r.get("converged"), bool)
        ):
            _error("invalid_state", "SEM optimizer start record is invalid.")
        _integer(r["iterations"], "iterations", 0, options["max_iterations"])
        _integer(r["variance_rescalings"], "variance_rescalings", 0, 30)
        _integer(r["evaluations"], "evaluations", 1, options["max_iterations"] * 27 + 32)
        if r["converged"]:
            if set(r) != complete or r["failure"] is not None:
                _error(
                    "invalid_state",
                    "Converged SEM start needs objective, score and no failure marker.",
                )
            _real(r["score_max"], "score_max", 0, options["tolerance"])
            objectives.append(_real(r["objective_per_observation"], "objective", -1e100, 1e100))
        elif not isinstance(r["failure"], str) or not 1 <= len(r["failure"]) <= 600:
            _error("invalid_state", "Unsuccessful SEM start must preserve its explicit failure.")
    if not objectives:
        _error("invalid_state", "SEM saved state has no converged start.")
    expected = _evaluate(_tensor(solver), spec, cov, means, n, options)
    loss = -expected["log_likelihood"] / n - 0.5 * (
        len(spec["columns"]) * math.log(2 * math.pi)
        + sum(math.log(s * s) for s in expected["scales"])
    )
    _close(loss, min(objectives), "best objective", absolute=1e-10)
    cached = payload["results"]
    if not isinstance(cached, Mapping) or set(cached) != set(expected):
        _error("invalid_state", "SEM cached numerical result fields are incomplete.")
    for key, value in expected.items():
        if key in ("covariance", "standardized_covariance"):
            actual = cached[key]
            if (
                not isinstance(actual, (list, tuple))
                or len(actual) != len(value)
                or any(
                    not isinstance(row, (list, tuple)) or len(row) != len(value) for row in actual
                )
            ):
                _error("invalid_state", "SEM joint covariance shape differs from replay.")
            for i, row in enumerate(value):
                for j, v in enumerate(row):
                    unit = math.sqrt(abs(value[i][i])) * math.sqrt(abs(value[j][j]))
                    _close(actual[i][j], v, f"results.{key}[{i},{j}]", absolute=unit * 2e-10)
        else:
            floor = (
                n * 1e-10
                if key == "score_solver"
                else 1e-10
                if key == "score_max_per_observation"
                else 0.0
            )
            _close(cached[key], value, "results." + key, absolute=floor)
    if payload["digest"] != _digest({k: v for k, v in payload.items() if k != "digest"}):
        _error(
            "invalid_state", "SEM digest does not match its complete source and numerical state."
        )
    return payload


class SEMState(BaseModel):
    """Immutable, versioned continuous SEM state with complete semantic replay."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    schema_version: Literal["openecon.latent_sem.v1"] = SCHEMA
    payload: Any

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        """Admit bounded encoded input before JSON decoding and numerical replay."""
        _encoded_json_admission(json_data, limit=32 * 1024 * 1024,
                                operation="SEM typed JSON decoding")
        return super().model_validate_json(json_data, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def admit_metadata(cls, value):
        """Admit complete metadata before model conversion and payload freezing."""
        _state_copy_admission(value, operation="SEM typed metadata admission")
        return value

    @wraps(BaseModel.model_copy)
    def model_copy(self, *, update=None, deep=False):
        if deep:
            _state_copy_admission((self, update), operation="SEM typed deep-copy update")
        return super().model_copy(update=update, deep=deep)

    def __deepcopy__(self, memo=None):
        _state_copy_admission(self, operation="SEM typed state deep copy")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported SEM schema.")
        _replay(self.payload)
        return super().__deepcopy__(memo)

    @wraps(BaseModel.model_dump)
    def model_dump(self, **kwargs):
        _state_copy_admission(self, operation="SEM complete portable state copy")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported SEM schema.")
        _replay(self.payload)
        return super().model_dump(**kwargs)

    @wraps(BaseModel.model_dump_json)
    def model_dump_json(self, **kwargs):
        _json_export_admission(self, indent=kwargs.get("indent"), limit=32 * 1024 * 1024,
                              operation="SEM complete JSON serialization")
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unsupported SEM schema.")
        _replay(self.payload)
        return super().model_dump_json(**kwargs)

    @model_validator(mode="after")
    def validate_semantics(self):
        """Validate statistical equations before freezing the complete state payload."""
        object.__setattr__(self, "payload", _freeze(_replay(self.payload)))
        return self

    @field_serializer("payload")
    def serialize_payload(self, value):
        """Return portable JSON-compatible source, constraints and numerical state."""
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unknown SEMState schema version.")
        return _thaw(_replay(value))

    def to_tables(self):
        """Render original-unit and standardized joint inference and saved fit diagnostics."""
        if self.schema_version != SCHEMA:
            _error("invalid_state", "Unknown SEMState schema version.")
        return _tables(_replay(self.payload))

    def __str__(self):
        return str(self.to_tables())


def _coefficient_table(values, covariance, names, level):
    critical = normal_isf((1 - level) / 2)
    se = [math.sqrt(max(covariance[i][i], 0.0)) for i in range(len(values))]
    z = [e / s if s > 0 else None for e, s in zip(values, se)]
    return table(
        {
            "estimate": values,
            "std_error": se,
            "z": z,
            "p_value": [None if v is None else 2 * normal_sf(abs(v)) for v in z],
            "ci_lower": [e - critical * s for e, s in zip(values, se)],
            "ci_upper": [e + critical * s for e, s in zip(values, se)],
        },
        index=names,
    )


def _tables(state):
    result, spec = state["results"], state["spec"]
    names, nodes = result["parameter_names"], spec["nodes"]
    standardized_names = result["standardized_parameter_names"]
    return TableSet(
        {
            "parameters": _coefficient_table(
                result["parameters"], result["covariance"], names, state["options"]["level"]
            ),
            "covariance": table(result["covariance"], index=names, columns=names),
            "standardized_parameters": _coefficient_table(
                result["standardized_parameters"],
                result["standardized_covariance"],
                standardized_names,
                state["options"]["level"],
            ),
            "standardized_covariance": table(
                result["standardized_covariance"],
                index=standardized_names,
                columns=standardized_names,
            ),
            "paths": table(result["paths"], index=nodes, columns=nodes),
            "standardized_paths": table(result["standardized_paths"], index=nodes, columns=nodes),
            "disturbance_covariance": table(
                result["disturbance_covariance"], index=nodes, columns=nodes
            ),
            "implied_covariance": table(
                result["implied_covariance"], index=spec["columns"], columns=spec["columns"]
            ),
            "fit": table(
                {
                    k: [v]
                    for k, v in result["diagnostics"].items()
                    if isinstance(v, (Real, type(None)))
                }
            ),
        },
        title="Continuous latent structural equation model",
        procedure="latent_sem",
        n=result["n"],
        log_likelihood=result["log_likelihood"],
        df=result["diagnostics"]["df"],
        covariance_method="full joint observed information; normal ML divisor n",
        inference="iid Gaussian regular interior SEM; asymptotic normal Wald",
        mean_structure="joint constrained means and covariance"
        if spec["meanstructure"]
        else "unrestricted observed means profiled",
        sem_state=_thaw(state),
        resource_plan=state["resource_plan"],
        notes=[
            "All observed predictors are modeled jointly as random variables.",
            "Unit-disturbance identification fixes latent innovation variance; it does not fix endogenous total variance.",
            "Multigroup, MAR FIML, robust/scaled and generalized/multilevel inference remain separate domains.",
        ],
    )


def _fit(spec, means, cov, source, options, resource):
    _spd(cov)
    if spec["meanstructure"] and means is None:
        _error("invalid_input", "Mean-structure SEM requires an observed mean vector.")
    scales = cov.diag().sqrt()
    x, starts = _solve(
        spec,
        scales,
        cov / scales[:, None] / scales[None, :],
        None if means is None else means / scales,
        options,
    )
    n = len(source["positions"]) if source["kind"] == "raw" else source["original_n"]
    result = _evaluate(x, spec, cov, means, n, options)
    state = {
        "schema": SCHEMA,
        "spec": _thaw(_freeze(spec)),
        "source": source,
        "options": options,
        "solver_parameters": x.tolist(),
        "starts": starts,
        "results": result,
        "resource_plan": resource,
    }
    state["digest"] = _digest(state)
    return _tables(SEMState(payload=state).payload)


def latent_sem(
    data,
    *,
    columns,
    factors=None,
    paths=None,
    identification="marker",
    markers=None,
    residual_covariances=None,
    intercepts=None,
    meanstructure=True,
    fixed=None,
    equalities=None,
    missing="raise",
    max_iterations=300,
    tolerance=1e-7,
    max_work=MAX_WORK,
    level=0.95,
):
    """Fit an explicit acyclic continuous SEM to bounded resident raw data.

    Paths map each target to source/free(None)/fixed coefficients. Factor indicators
    define measurement edges; markers fix one original-unit loading per factor.
    Selected disturbance covariances use pairs of node names. Observed intercepts
    are free and latent intercepts zero by default; explicit restrictions and
    same-family equality labels use the named parameter map. All observed variables
    have estimated joint means/covariance; complete-case dropping is explicit.
    """
    spec = _spec(
        columns,
        {} if factors is None else factors,
        paths,
        identification,
        markers,
        residual_covariances,
        intercepts,
        meanstructure,
        fixed,
        equalities,
    )
    options = _options(max_iterations, tolerance, max_work, level)
    n = _resident(data, spec["columns"])
    resource = _admit(n, spec, options, raw=True)
    if isinstance(data, pd.DataFrame):
        _index_admission(data.index)
        frame = data.loc[:, spec["columns"]].copy()
    else:
        indexes = [data[c].index for c in spec["columns"] if isinstance(data[c], pd.Series)]
        if indexes:
            _index_admission(indexes[0])
        frame = pd.DataFrame({c: data[c] for c in spec["columns"]})
    means, cov, source = _moments(frame, missing)
    return _fit(spec, means, cov, source, options, resource)


def _summary_values(covariance, means, spec, n, divisor):
    columns = spec["columns"]
    p = len(columns)
    if divisor not in ("n", "n-1"):
        _error("invalid_option", "Declare summary covariance divisor='n' or 'n-1'.")
    if isinstance(covariance, pd.DataFrame):
        if list(covariance.columns) != list(columns) or list(covariance.index) != list(columns):
            _error(
                "invalid_input", "Summary covariance labels and order must match declared columns."
            )
        covariance = covariance.to_numpy().tolist()
    if (
        not isinstance(covariance, Sequence)
        or isinstance(covariance, (str, bytes))
        or len(covariance) != p
        or any(
            not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != p
            for row in covariance
        )
    ):
        _error(
            "invalid_input", "Provide a bounded square covariance sequence or labelled DataFrame."
        )
    if any(not _finite_real(v) for row in covariance for v in row):
        _error("invalid_input", "Summary covariance entries must be finite real values.")
    if isinstance(means, pd.Series):
        if list(means.index) != list(columns):
            _error("invalid_input", "Summary mean labels/order must match declared columns.")
        means = means.tolist()
    if means is not None and (
        not isinstance(means, Sequence)
        or isinstance(means, (str, bytes))
        or len(means) != p
        or any(not _finite_real(v) for v in means)
    ):
        _error("invalid_input", "Summary input needs one finite observed mean per column.")
    values = [v for row in covariance for v in row] + ([] if means is None else list(means))
    if any(isinstance(v, Integral) and int(float(v)) != int(v) for v in values):
        _error(
            "precision_loss", "Summary integer moments must be exactly representable in float64."
        )
    raw_covariance = [[float(v) for v in row] for row in covariance]
    cov = _tensor(raw_covariance)
    if divisor == "n-1":
        cov = cov * (n - 1) / n
    return cov, None if means is None else _tensor(means), raw_covariance


def latent_sem_covariance(
    covariance,
    *,
    columns,
    n,
    divisor,
    means=None,
    factors=None,
    paths=None,
    identification="marker",
    markers=None,
    residual_covariances=None,
    intercepts=None,
    meanstructure=True,
    fixed=None,
    equalities=None,
    max_iterations=300,
    tolerance=1e-7,
    max_work=MAX_WORK,
    level=0.95,
):
    """Fit continuous SEM from labelled moments with an explicit covariance divisor.

    Summary means are required for constrained mean-structure likelihood. Setting
    meanstructure=False profiles observed means; omitted means cannot reconstruct
    the training rows or support default individual factor-score predictions.
    """
    spec = _spec(
        columns,
        {} if factors is None else factors,
        paths,
        identification,
        markers,
        residual_covariances,
        intercepts,
        meanstructure,
        fixed,
        equalities,
    )
    options = _options(max_iterations, tolerance, max_work, level)
    resource = _admit(n, spec, options)
    cov, mean, raw_cov = _summary_values(covariance, means, spec, n, divisor)
    source = {
        "kind": "summary",
        "original_n": int(n),
        "covariance": raw_cov,
        "divisor": divisor,
        "means": None if mean is None else mean.tolist(),
    }
    return _fit(spec, mean, cov, source, options, resource)


def latent_sem_restore(saved):
    """Replay source, constraints, moments, full OIM and diagnostics without refitting.

    Accept an SEM TableSet, immutable SEMState, state mapping or SEMState JSON.
    Rehashed forged caches cannot override the saved statistical equations.
    """
    if isinstance(saved, SEMState):
        return saved.to_tables()
    if isinstance(saved, TableSet):
        if saved.attrs.get("procedure") != "latent_sem":
            _error("invalid_state", "Pass a continuous latent SEM result.")
        saved = saved.attrs.get("sem_state")
    if isinstance(saved, str):
        _encoded_json_admission(saved, limit=32 * 1024 * 1024,
                                operation="SEM public JSON restoration")
        saved = json.loads(saved)
    if isinstance(saved, Mapping) and set(saved) == {"schema_version", "payload"}:
        return _tables(SEMState.model_validate(saved).payload)
    return _tables(SEMState(payload=saved).payload)


def _state(result):
    restored = latent_sem_restore(result)
    return restored.attrs["sem_state"]


def sem_effects(result, *, source, target, standardized=False, level=None):
    """Compute one direct/indirect/total path effect and its full joint delta covariance.

    The acyclic model sums all directed paths, including observed and latent nodes.
    Uncertainty differentiates both the paths and total-variance standardization;
    zero or fixed effects have zero sampling variance and no fabricated Wald p.
    """
    state = _state(result)
    spec, fit = state["spec"], state["results"]
    if source not in spec["nodes"] or target not in spec["nodes"] or source == target:
        _error("invalid_option", "Effects require distinct declared source and target nodes.")
    if not isinstance(standardized, bool):
        _error("invalid_option", "standardized must be Boolean.")
    level = state["options"]["level"] if level is None else _real(level, "level", 0.5, 0.9999)
    i, j = spec["nodes"].index(target), spec["nodes"].index(source)
    x, scales = _tensor(state["solver_parameters"]), _tensor(fit["scales"])

    def values(v):
        d = _decode(v, spec, scales)
        b = d["raw_b"]
        inverse = torch.linalg.solve(
            torch.eye(len(b), dtype=DTYPE, device=DEVICE) - b,
            torch.eye(len(b), dtype=DTYPE, device=DEVICE),
        )
        direct, total = b[i, j], inverse[i, j]
        factor = (d["raw_omega"][j, j] / d["raw_omega"][i, i]).sqrt() if standardized else 1.0
        return torch.stack([direct, total - direct, total]) * factor

    estimates = values(x)
    jac = torch.autograd.functional.jacobian(values, x)
    cov = jac @ torch.linalg.solve(_tensor(fit["observed_information_solver"]), jac.T)
    cov = (cov + cov.T) / 2
    names = ["direct", "indirect", "total"]
    return TableSet(
        {
            "effects": _coefficient_table(estimates.tolist(), cov.tolist(), names, level),
            "covariance": table(cov.tolist(), index=names, columns=names),
        },
        title=f"SEM effects: {source} to {target}",
        procedure="sem_effects",
        source=source,
        target=target,
        standardized=standardized,
        parent_digest=state["digest"],
        inference="full joint observed-information delta method",
    )


def latent_scores(result, data=None, *, missing="raise"):
    """Predict regression factor scores from saved joint Gaussian moments.

    Return aligned conditional latent means; attrs retain their joint conditional
    covariance with model parameters treated as estimated constants. Default rows
    come only from saved raw data, including explicit complete-case omissions.
    """
    state = _state(result)
    spec, fit = state["spec"], state["results"]
    if not spec["latent"]:
        _error("unsupported_prediction", "The path model has no latent nodes to score.")
    if missing not in ("raise", "drop"):
        _error("invalid_option", "Score missing must be raise or drop.")
    if data is None:
        source = state["source"]
        if source["kind"] != "raw":
            _error(
                "unsupported_prediction",
                "Summary statistics cannot reconstruct individual training scores; supply new rows.",
            )
        data = pd.DataFrame(
            source["rows"], columns=spec["columns"], index=_decode_index(source["index"])
        )
        missing = source["missing"]
    n = _resident(data, spec["columns"])
    _integer(n, "prediction rows", 1, MAX_ROWS)
    p, k = len(spec["columns"]), len(spec["latent"])
    plan_workspace(
        "latent_sem_scores",
        {
            "resident_data_and_scores": 8 * n * (p * 4 + k * 3),
            "conditional_moments": 8 * (p * p * 10 + k * p * 5 + k * k * 5),
            "typed_index": MAX_INDEX_BYTES * 2,
        },
    )
    if isinstance(data, pd.DataFrame):
        _index_admission(data.index)
        frame = data.loc[:, spec["columns"]].copy()
    else:
        indexes = [data[c].index for c in spec["columns"] if isinstance(data[c], pd.Series)]
        if indexes:
            _index_admission(indexes[0])
        frame = pd.DataFrame({c: data[c] for c in spec["columns"]})
    # Reuse strict real/integer admission without requiring prediction n>p.
    from pandas.api.types import is_bool_dtype, is_complex_dtype, is_integer_dtype, is_numeric_dtype

    if any(
        not is_numeric_dtype(frame[c].dtype)
        or is_bool_dtype(frame[c].dtype)
        or is_complex_dtype(frame[c].dtype)
        for c in frame
    ):
        _error("invalid_input", "Scores need real numeric continuous indicators.")
    if any(
        is_integer_dtype(frame[c].dtype)
        and any(not pd.isna(v) and int(float(v)) != int(v) for v in frame[c])
        for c in frame
    ):
        _error(
            "precision_loss", "Score integer indicators must be exactly representable in float64."
        )
    raw = frame.astype("float64")
    if any(math.isinf(v) for c in raw for v in raw[c]):
        _error("invalid_input", "Infinite score inputs are not missing values.")
    keep = ~raw.isna().any(axis=1)
    if not keep.all() and missing == "raise":
        _error("missing_data", "Score inputs contain missing values; declare missing='drop'.")
    omega = _tensor(fit["node_covariance"])
    chol = torch.linalg.cholesky(omega[:p, :p])
    weight = torch.cholesky_solve(omega[:p, p:], chol).T
    if spec["meanstructure"]:
        means = _tensor(fit["node_means"])
        observed_mean, latent_mean = means[:p], means[p:]
    else:
        if fit["sample_means"] is None:
            _error(
                "unsupported_prediction",
                "Covariance-only summary scores require saved observed centering means.",
            )
        observed_mean, latent_mean = (
            _tensor(fit["sample_means"]),
            torch.zeros(k, dtype=DTYPE, device=DEVICE),
        )
    scores = torch.full((n, k), float("nan"), dtype=DTYPE, device=DEVICE)
    positions = [i for i, v in enumerate(keep) if v]
    if positions:
        values = _tensor(raw.iloc[positions].to_numpy())
        scores[positions] = latent_mean + (values - observed_mean) @ weight.T
    conditional = omega[p:, p:] - weight @ omega[:p, p:]
    conditional = (conditional + conditional.T) / 2
    output = pd.DataFrame(scores.tolist(), index=frame.index.copy(), columns=spec["latent"])
    output.attrs.update(
        procedure="latent_scores",
        method="Gaussian regression conditional mean",
        conditional_covariance=conditional.tolist(),
        parameter_uncertainty="plug-in; not included",
        parent_digest=state["digest"],
        positions=positions,
    )
    return output
