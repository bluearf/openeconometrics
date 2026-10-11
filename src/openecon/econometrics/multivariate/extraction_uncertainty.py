"""Strict fixed one-factor extraction-functional IID uncertainty.

Each refit uses the declared method and its documented deterministic start.
Finite-sample gates detect singular local charts; they cannot certify population
regularity, bootstrap consistency or exact coverage. No failed draw is dropped.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.resources import plan_workspace
from openecon.econometrics.resident_cpu import resident_cpu
from . import common as c
from . import extraction as ex
from .factor import factor
from .factor_uncertainty import _identities, _export_admission, _state_digest
from .uncertainty import MAX_ROWS, MAX_VARIABLES, MAX_REPLICATIONS, MAX_WORK, _input_rows, _selected_input

METHODS = ("ml", "minres", "alpha", "image_covariance", "generalized_image")
MAX_EXPORT_BYTES = 32 * 1024 * 1024
ITERATIONS = 1000
TOLERANCE = 1e-8
DIAGNOSTICS = ("correlation_condition", "minimum_uniqueness", "anchor_loading",
               "spectral_gap", "local_chart_margin", "stationarity_residual")


def _positive_curvature(hessian: Tensor) -> float:
    hessian = (hessian + hessian.T) / 2
    if not bool(torch.isfinite(hessian).all()):
        raise AnalysisError("unidentified_factor", "The extraction Hessian is non-finite.")
    roots = torch.linalg.eigvalsh(hessian)
    scale = float(roots.abs().max())
    margin = float(roots.min()) / max(scale, 1e-300)
    if scale <= 64 * len(roots) * torch.finfo(c.FLOAT).eps or margin <= TOLERANCE:
        raise AnalysisError("unidentified_factor", "The extraction is not a strict, numerically identified interior local minimum.")
    return margin


def _regularity(fitted: TableSet, r: Tensor, loading: Tensor, uniqueness: Tensor):
    """Method-specific observed local chart, independent of sign anchoring."""
    method, p = fitted.attrs["method"], len(loading)
    condition = float(torch.linalg.cond(r))
    if condition > 1e8:
        raise AnalysisError("unidentified_factor", "The sample correlation is too ill-conditioned for this uncertainty chart.")
    gap = float("nan")
    residual = 0.
    if method == "ml":
        if bool((uniqueness <= ex.PSI_FLOOR + 1e-6).any()):
            raise AnalysisError("heywood_case", "ML bootstrap requires uniqueness strictly above its declared numerical bound.")
        theta = (uniqueness - ex.PSI_FLOOR).log()
        # A caller's inference mode disables autograd even inside the Hessian
        # helper. Clone all derivative inputs while inference is disabled so
        # ordinary tensors, rather than inference tensors, enter its graph.
        with torch.inference_mode(False):
            derivative_r = r.detach().clone()
            parameters = torch.cat((loading, uniqueness)).detach().clone()

            def full_discrepancy(value):
                a, psi = value[:p], value[p:]
                sigma = a[:, None] @ a[None, :] + torch.diag(psi)
                return torch.linalg.slogdet(sigma)[1] + torch.trace(torch.linalg.solve(sigma, derivative_r)) - torch.linalg.slogdet(derivative_r)[1] - p

            hessian = torch.autograd.functional.hessian(full_discrepancy, parameters)
        margin = _positive_curvature(hessian)
        residual = float(ex.ml_discrepancy(theta, r, 1)[1].abs().max())
        if residual > 1e-6:
            raise AnalysisError("no_convergence", "The ML refit is not stationary in the declared uniqueness chart.")
        roots = torch.linalg.eigvalsh(r * torch.outer(uniqueness.rsqrt(), uniqueness.rsqrt()))
        gap = float(roots[-1] - roots[-2])
        if float(roots[-1]) <= 1 + TOLERANCE:
            raise AnalysisError("unidentified_factor", "The retained ML factor root is not above one.")
    elif method == "minres":
        with torch.inference_mode(False):
            derivative_r, derivative_loading = r.detach().clone(), loading.detach().clone()
            hessian = torch.autograd.functional.hessian(
                lambda a: ex.minres_objective(a[:, None], derivative_r)[0], derivative_loading)
        margin = _positive_curvature(hessian)
        residual = float(ex.minres_objective(loading[:, None], r)[1].abs().max())
        if residual > 1e-7:
            raise AnalysisError("no_convergence", "The minimum-residual refit is not stationary.")
    elif method == "alpha":
        h = torch.as_tensor(fitted.attrs["last_iteration_communalities"], dtype=c.FLOAT)

        def update(value):
            scale = value.rsqrt()
            matrix = (r - torch.eye(p, dtype=c.FLOAT)) * torch.outer(scale, scale) + torch.eye(p, dtype=c.FLOAT)
            roots, vectors = torch.linalg.eigh(matrix)
            return value * vectors[:, -1].square() * roots[-1]

        # Only top-vs-discarded gaps occur in this dominant eigenvector
        # derivative. Repeated discarded roots are allowed; differentiating a
        # complete eigenbasis would spuriously divide their zero gaps.
        scale = h.rsqrt()
        operator = (r - torch.eye(p, dtype=c.FLOAT)) * torch.outer(scale, scale)
        roots, vectors = torch.linalg.eigh(operator + torch.eye(p, dtype=c.FLOAT))
        top, v = roots[-1], vectors[:, -1]
        if float(top-roots[-2]) <= TOLERANCE * max(1., float(top.abs())):
            raise AnalysisError("unidentified_factor", "The leading alpha iteration root is not separated.")
        columns = []
        for i in range(p):
            derivative = torch.zeros_like(operator)
            derivative[i, :] -= .5 * operator[i, :] / h[i]
            derivative[:, i] -= .5 * operator[:, i] / h[i]
            droot = v @ derivative @ v
            dvector = vectors[:, :-1] @ ((vectors[:, :-1].T @ derivative @ v) / (top-roots[:-1]))
            column = h * (2*v*dvector*top + v.square()*droot)
            column[i] += v[i].square()*top
            columns.append(column)
        jacobian = torch.stack(columns, dim=1)
        singular = torch.linalg.svdvals(torch.eye(p, dtype=c.FLOAT) - jacobian)
        margin = float(singular.min() / singular.max().clamp_min(1.))
        if margin <= TOLERANCE:
            raise AnalysisError("unidentified_factor", "The alpha fixed-point chart is locally singular.")
        residual = float((update(h) - h).abs().max())
        roots = torch.as_tensor(fitted["eigenvalues"].eigenvalue.to_numpy().copy(), dtype=c.FLOAT)
        gap = float(roots[0] - roots[1])
        if float(roots[0]) <= 1 + TOLERANCE:
            raise AnalysisError("unidentified_factor", "The alpha root must exceed one for this regular chart.")
    else:
        roots = torch.as_tensor(fitted["eigenvalues"].eigenvalue.to_numpy().copy(), dtype=c.FLOAT)
        gap = float(roots[0] - roots[1])
        margin = gap / max(1., float(roots.abs().max()))
    if not torch.isfinite(torch.tensor(margin)) or margin <= TOLERANCE:
        raise AnalysisError("unidentified_factor", "The extraction's local chart is not numerically identified.")
    if method != "minres" and gap <= TOLERANCE * max(1., abs(float(roots[-1] if method == "ml" else roots[0]))):
        raise AnalysisError("unidentified_factor", "The retained extraction root is not separated.")
    return condition, gap, margin, residual


@c.procedure
@resident_cpu
def _parameters(sample: pd.DataFrame, names: list[str], anchor: str | None, method: str):
    options = {"max_iterations": ITERATIONS} if method in ("ml", "minres", "alpha") else {}
    if method == "alpha":
        options["tolerance"] = 1e-10
    fitted = factor(sample, names, method=method, factors=1, **options)
    loading = torch.as_tensor(fitted["loadings"].iloc[:, 0].to_numpy(dtype="float64").copy())
    uniqueness = torch.as_tensor(fitted["uniqueness"].iloc[:, 0].to_numpy(dtype="float64").copy())
    if fitted.attrs["factors"] != 1 or not bool(torch.isfinite(loading).all() and torch.isfinite(uniqueness).all()):
        raise AnalysisError("unidentified_factor", "One finite identified factor is required.")
    if fitted.attrs["heywood"] or bool((uniqueness <= TOLERANCE).any()):
        raise AnalysisError("heywood_case", "Bootstrap refuses nonpositive or boundary uniqueness.")
    fixed_anchor = names[int(loading.abs().argmax())] if anchor is None else anchor
    value = float(loading[names.index(fixed_anchor)])
    if abs(value) <= TOLERANCE * max(1., float(loading.abs().max())):
        raise AnalysisError("unidentified_factor", "The fixed loading sign anchor is numerically zero.")
    sign = -1 if value < 0 else 1
    if sign < 0:
        loading = -loading
        # These unrotated one-factor tables share the factor's orientation.
        # Keep saved scoring and the complete point snapshot in the same chart
        # as the reported point vector and every bootstrap refit.
        for name in ("loadings", "score_coefficients"):
            if name in fitted:
                fitted[name] = -fitted[name]
    fitted.attrs.update(sign_anchor=fixed_anchor, sign_convention="fixed anchor loading positive",
                        native_to_anchor_sign=sign)
    x = c.matrix(sample, names)
    # Match factor's raw-input correlation exactly. Dividing the SSCP by n-1
    # first can unnecessarily underflow an otherwise representable matrix.
    _, sscp, _ = c.moments(x, names)
    r = c.correlation(sscp)
    condition, gap, margin, residual = _regularity(fitted, r, loading, uniqueness)
    diag = torch.tensor([condition, float(uniqueness.min()), abs(value), gap, margin, residual], dtype=c.FLOAT)
    return torch.cat((loading, uniqueness)), fixed_anchor, fitted, diag


@resident_cpu
def extraction_bootstrap(data: Any, columns: list[str], *, method: str, replications: int,
                         confidence: float, seed: int, anchor: str | None, missing: str) -> TableSet:
    c.check_choice(method, "method", METHODS)
    names = c.name_list(columns, "columns", minimum=4 if method in ("ml", "minres") else 3)
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1 or (replications + 1) * (1-confidence) / 2 < 1-1e-12:
        raise AnalysisError("insufficient_replications", "Each percentile tail needs an expected bootstrap order statistic.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63-1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    if anchor is not None:
        c.check_name(anchor, "anchor")
        if anchor not in names:
            raise AnalysisError("invalid_spec", "anchor must be an analysed variable.")
    rows, p = _input_rows(data, names), len(names)
    if rows > MAX_ROWS or p > MAX_VARIABLES:
        raise AnalysisError("workspace_limit", "Extraction bootstrap exceeds its 10000-row/16-variable domain.")
    iterations = ITERATIONS if method in ("ml", "minres", "alpha") else 1
    work = (replications+1) * (rows*p*p + 4*iterations*p**3)
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Requested iterative extraction refits exceed the work budget.")
    identities = list(_identities(data, rows))
    export_bytes = _export_admission(names, rows, 1, replications, identities[4], identities[1]+identities[2], False)
    export_bytes += 96 * (replications*rows + 16*p*p + replications*len(DIAGNOSTICS))
    if export_bytes > MAX_EXPORT_BYTES:
        raise AnalysisError("export_limit", "The complete sample/index/draw result exceeds the 32 MiB portable budget.")
    plan = plan_workspace("strict one-factor extraction bootstrap", {
        "selected_sample_and_refit": rows*p*64,
        "full_draw_indices": rows*replications*32,
        "parameters_diagnostics_covariance": (replications*(2*p+len(DIAGNOSTICS))+16*p*p)*32,
        "extraction_derivative_workspace": 128*p**3*8,
        "portable_result_upper_bound": export_bytes,
    }).record()
    sample, keep, dropped = c.select(_selected_input(data, names), names, missing=missing)
    n = len(sample)
    if n < max(20, 2*p+1):
        raise AnalysisError("insufficient_observations", "At least max(20,2*p+1) complete IID rows are required.")
    x = c.matrix(sample, names)
    point, fixed_anchor, point_fit, point_diagnostics = _parameters(sample, names, anchor, method)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, 2*p), dtype=c.FLOAT)
    diagnostics = torch.empty((replications, len(DIAGNOSTICS)), dtype=c.FLOAT)
    indices = torch.empty((replications, n), dtype=torch.int64)
    failures = []
    for b in range(replications):
        indices[b] = torch.randint(n, (n,), generator=generator, device="cpu")
        replicate = pd.DataFrame(x[indices[b]].numpy(), columns=names)
        try:
            draws[b], _, _, diagnostics[b] = _parameters(replicate, names, fixed_anchor, method)
        except AnalysisError as exc:
            failures.append({"replication": b+1, "code": exc.code, "message": str(exc)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} extraction refits failed; no draw is dropped and no uncertainty is returned.")
        error.failures, error.replications_attempted = failures, replications
        error.successful_replications = replications-len(failures)
        error.replicate_indices = indices.tolist()
        failed = {entry["replication"] for entry in failures}
        error.successful_positions = [b+1 for b in range(replications) if b+1 not in failed]
        error.successful_vectors = [draws[b-1].tolist() for b in error.successful_positions]
        raise error
    centred = draws-draws.mean(0)
    covariance = centred.T @ centred/(replications-1)
    covariance = (covariance+covariance.T)/2
    intervals = torch.quantile(draws, torch.tensor([(1-confidence)/2, (1+confidence)/2], dtype=c.FLOAT), dim=0)
    undefined = torch.full_like(point, float("nan"))
    labels = [f"loading:{name}" for name in names]+[f"uniqueness:{name}" for name in names]
    positions = [i for i, kept in enumerate(keep.tolist()) if kept]
    digest = hashlib.sha256(json.dumps({"method": method, "variables": names, "positions": positions,
                                       "identities": identities}, separators=(",", ":")).encode())
    digest.update(x.numpy().tobytes())
    tables = {
        "estimates": c.frame(torch.stack((point, covariance.diagonal().clamp_min(0).sqrt(), intervals[0], intervals[1], draws.mean(0)-point, undefined, undefined), dim=1), columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=list(range(1, replications+1))),
        "replicate_indices": c.frame(indices, columns=[f"draw{i+1}" for i in range(n)], index=list(range(1, replications+1))),
        "replicate_diagnostics": c.frame(diagnostics, columns=list(DIAGNOSTICS), index=list(range(1, replications+1))),
        "point_diagnostics": c.frame(point_diagnostics[None, :], columns=list(DIAGNOSTICS), index=[method]),
        "sample": c.frame(x, columns=names, index=positions),
        **{"point_"+key: frame.copy() for key, frame in point_fit.items()},
    }
    result = TableSet(tables, title=f"Fixed one-factor {method} IID functional bootstrap", procedure="factor_bootstrap",
                    method=method, factors=1, rotate=None, variables=names, n=n, n_missing=dropped, physical_rows=rows,
                    sample_positions=positions, source_identities=identities, source_content_sha256=digest.hexdigest(),
                    replications=replications, successful_replications=replications, failed_replications=[], seed=seed,
                    confidence=confidence, sign_anchor=fixed_anchor, covariance_divisor=replications-1,
                    quantile_interpolation="linear", uncertainty="IID nonparametric marginal percentile bootstrap",
                    inference_target=f"fixed one-factor {method} estimator functional", missing="listwise",
                    state_schema="openecon.extraction_bootstrap.v1", point_fit_attrs=point_fit.attrs,
                    diagnostics_contract="correlation conditioning; interior uniqueness; fixed sign; separated root (non-minres); strict local curvature (ML/minres); regular fixed point (alpha)",
                    inferential_assumptions="IID complete-case rows; fixed method/count/dimension; finite fourth moments; interior nonsingular population correlation; regular local extraction/sign chart. Population assumptions are not established by observed gates.",
                    p_values_available=False, inference_df_available=False, precision="float64", device="cpu",
                    rng="torch.Generator CPU randint", rng_version=torch.__version__, estimated_work=work,
                    resource_plan=plan, portable_bytes_upper_bound=export_bytes,
                    notes=["Marginal first-order functional intervals; no simultaneous or exact finite-sample coverage, p/df, latent posterior, selected count/rotation or global optimizer guarantee.",
                           "All planned refits must pass; source sample, complete indices/parameters, full covariance and method-specific diagnostics persist."])

    result.attrs["state_content_sha256"] = _state_digest(result)
    return result
