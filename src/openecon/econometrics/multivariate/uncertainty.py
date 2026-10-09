"""Bounded iid bootstrap uncertainty for a fixed principal-factor functional.

The target is the unrotated, one-factor principal-factor estimator, including
its re-estimated sample correlation and SMC diagonal. It is not an ML loading
standard error or inference for a data-selected factor count/rotation.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.resources import plan_workspace

from . import common as c
from .factor import factor
from .summary import Summary

MAX_ROWS = 10_000
MAX_VARIABLES = 16
MAX_REPLICATIONS = 1_999
MAX_WORK = 250_000_000
_GEOMETRY_TOLERANCE = 1e-8


def _index_label(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Integral):
        return c.label(value)
    if isinstance(value, Real) and not math.isfinite(float(value)):
        return None
    return c.label(value)


def _input_rows(data: Any, names: list[str]) -> int:
    """Inspect declared geometry without converting the caller's data."""
    if isinstance(data, (Dataset, Summary)):
        raise AnalysisError("unsupported_input", "Factor bootstrap requires resident raw observations; Dataset and summary matrices are unsupported.")
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, Mapping):
        absent = [name for name in names if name not in data]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        try:
            lengths = [len(data[name]) for name in names]
        except TypeError as exc:
            raise AnalysisError("invalid_data", "Supply resident columns of scalar observations.") from exc
        if len(set(lengths)) != 1:
            raise AnalysisError("invalid_data", "Data columns must have consistent lengths.")
        for name in names:
            if isinstance(data[name], torch.Tensor) and data[name].device.type != "cpu":
                raise AnalysisError("unsupported_device", "Factor bootstrap requires CPU inputs.")
        return lengths[0]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return len(data)
    raise AnalysisError("invalid_data", "Supply a resident DataFrame, mapping of columns or row records.")


def _selected_input(data: Any, names: list[str]) -> Any:
    if isinstance(data, pd.DataFrame):
        # Keep duplicate-column checks on the complete declared frame.
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "Data must have unique column names.")
        absent = [name for name in names if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        return data.loc[:, names]
    if isinstance(data, Mapping):
        return {name: data[name] for name in names}
    if not all(isinstance(row, Mapping) for row in data):
        raise AnalysisError("invalid_data", "Each resident row must be a mapping of scalar observations.")
    # Missing keys remain missing observations, consistently with DataFrame input.
    return [{name: row.get(name) for name in names} for row in data]


def _parameters(sample: pd.DataFrame, names: list[str], anchor: str | None):
    fitted = factor(sample, names, method="pf", factors=1, rotate=None)
    loading = torch.as_tensor(fitted["loadings"].iloc[:, 0].to_numpy(dtype="float64").copy())
    uniqueness = torch.as_tensor(fitted["uniqueness"].iloc[:, 0].to_numpy(dtype="float64").copy())
    eigenvalues = torch.as_tensor(fitted["eigenvalues"]["eigenvalue"].to_numpy(dtype="float64").copy())
    gap = float(eigenvalues[0] - eigenvalues[1])
    if not bool(torch.isfinite(loading).all() and torch.isfinite(uniqueness).all()) \
            or fitted.attrs["factors"] != 1:
        raise AnalysisError("unidentified_factor", "The fixed one-factor fit must have finite, identified loadings.")
    if gap <= _GEOMETRY_TOLERANCE * max(1.0, abs(float(eigenvalues[0]))):
        raise AnalysisError("unidentified_factor", "The leading reduced-matrix eigenvalue is not separated; loading orientation is unidentified.")
    if fitted.attrs["heywood"] or bool((uniqueness <= _GEOMETRY_TOLERANCE).any()):
        raise AnalysisError("heywood_case", "Factor bootstrap refuses boundary or nonpositive uniquenesses.")
    if anchor is None:
        anchor = names[int(loading.abs().argmax())]
    anchor_value = float(loading[names.index(anchor)])
    if abs(anchor_value) <= _GEOMETRY_TOLERANCE * max(1.0, float(loading.abs().max())):
        raise AnalysisError("unidentified_factor", "The recorded sign anchor must have a nonzero loading away from numerical zero.")
    if anchor_value < 0:
        loading = -loading
    return torch.cat((loading, uniqueness)), anchor, gap, fitted


@c.procedure
def factor_bootstrap(data: Any, columns: list[str], *, method: str = "pf",
                     replications: int = 199, confidence: float = .95,
                     seed: int = 0, anchor: str | None = None,
                     missing: str = "drop") -> TableSet:
    """Marginal iid bootstrap intervals for fixed one-factor principal factoring.

    Every replicate resamples complete raw rows, recalculates their moments and
    refits ``factor(..., method='pf', factors=1, rotate=None)``. A single recorded
    loading anchor fixes the otherwise arbitrary sign. Complete replicate
    loading/uniqueness vectors give joint sample covariance (divisor B-1), SEs
    and marginal percentile intervals with linear quantile interpolation.

    The inferential target is the principal-factor estimator functional under
    iid complete-case sampling, fixed p, finite fourth moments, nonsingular
    population correlation away from its boundary, separated leading
    eigenvalues, an anchor away from zero and interior positive uniqueness.
    These population assumptions are not verified by a finite observed sample.
    These are first-order bootstrap intervals; neither a familywise region nor
    finite-sample coverage is claimed. P-values and inference df are unavailable.
    Factor selection, rotations, ML, weights, clusters and summary/Dataset input
    are unsupported. Any failed replicate refuses the whole result; none are
    silently removed. Raw complete sample and every replicate parameter persist.

    Example
    -------
    >>> import random
    >>> rng = random.Random(42)
    >>> f = [rng.gauss(0, 1) for _ in range(120)]
    >>> data = {f'x{j}': [.7*v + .6*rng.gauss(0, 1) for v in f]
    ...         for j in range(4)}
    >>> result = factor_bootstrap(data, list(data), replications=39, seed=7)
    >>> result.attrs['successful_replications']
    39
    """
    names = c.name_list(columns, "columns", minimum=3)
    c.check_choice(method, "method", ("pf",))
    replications = c.check_count(replications, "replications", minimum=19, maximum=MAX_REPLICATIONS)
    confidence = c.check_number(confidence, "confidence", minimum=0, maximum=1, exclusive=True)
    if confidence == 1:
        raise AnalysisError("invalid_option", "confidence must be strictly below one.")
    if (replications + 1) * (1 - confidence) / 2 < 1 - 1e-12:
        raise AnalysisError("insufficient_replications", "Each requested percentile tail needs at least one expected bootstrap order statistic; raise replications or lower confidence.")
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63 - 1)
    c.check_choice(missing, "missing", ("drop", "raise"))
    if anchor is not None:
        c.check_name(anchor, "anchor")
        if anchor not in names:
            raise AnalysisError("invalid_spec", "anchor must name one of the analysed variables.")
    p, rows = len(names), _input_rows(data, names)
    if p > MAX_VARIABLES or rows > MAX_ROWS:
        raise AnalysisError("workspace_limit", f"Factor bootstrap supports at most {MAX_VARIABLES} variables and {MAX_ROWS} resident physical rows.")
    work = (replications + 1) * (rows * p * p + 64 * p**3)
    if work > MAX_WORK:
        raise AnalysisError("work_limit", "Requested bootstrap moment/fit/score work exceeds the factor-bootstrap budget.")
    plan = plan_workspace("one-factor iid bootstrap", {
        "selected_and_refit_blocks": rows * p * 64,
        "row_indices_and_masks": rows * 64,
        "factor_matrix_and_score_workspace": 64 * p * p * 8,
        "full_replicate_vectors": replications * (2 * p) * 32,
        "joint_covariance_and_quantiles": 64 * (2 * p)**2 * 8,
    }).record()
    sample, keep, dropped = c.select(_selected_input(data, names), names, missing=missing)
    n = len(sample)
    if n < max(20, 2 * p + 1):
        raise AnalysisError("insufficient_observations", "Factor bootstrap needs at least max(20, 2*p+1) complete iid rows.")
    x = c.matrix(sample, names)
    point, fixed_anchor, point_gap, point_fit = _parameters(sample, names, anchor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.empty((replications, 2 * p), dtype=c.FLOAT)
    diagnostics = torch.empty((replications, 2), dtype=c.FLOAT)
    failures = []
    for replication in range(replications):
        indices = torch.randint(n, (n,), generator=generator, device="cpu")
        replicate = pd.DataFrame(x[indices].numpy(), columns=names)
        try:
            parameters, _, gap, _ = _parameters(replicate, names, fixed_anchor)
            draws[replication] = parameters
            diagnostics[replication] = torch.tensor([gap, float(parameters[p:].min())], dtype=c.FLOAT)
        except AnalysisError as exc:
            failures.append({"replication": replication + 1, "code": exc.code, "message": str(exc)})
    if failures:
        error = AnalysisError("bootstrap_failure", f"{len(failures)} of {replications} iid bootstrap refits failed. No uncertainty result is returned and no failed replicate is dropped.")
        error.failures = failures
        error.replications_attempted = replications
        error.successful_replications = replications - len(failures)
        raise error
    centred = draws - draws.mean(0)
    covariance = centred.T @ centred / (replications - 1)
    covariance = (covariance + covariance.T) / 2
    se = covariance.diagonal().clamp_min(0).sqrt()
    tails = torch.tensor([(1 - confidence) / 2, (1 + confidence) / 2], dtype=c.FLOAT)
    intervals = torch.quantile(draws, tails, dim=0, interpolation="linear")
    undefined = torch.full_like(point, float("nan"))
    labels = [f"loading:{name}" for name in names] + [f"uniqueness:{name}" for name in names]
    positions = [i for i, kept in enumerate(keep.tolist()) if kept]
    digest = hashlib.sha256(json.dumps({"variables": names, "positions": positions,
                                       "shape": list(x.shape)}, separators=(",", ":")).encode())
    digest.update(x.numpy().tobytes())
    notes = ["Marginal percentile bootstrap intervals for the fixed principal-factor estimator functional; no familywise or exact finite-sample coverage.",
             "IID complete-case row sampling, fixed p, finite fourth moments and interior nonsingular population correlation are assumed, not empirically verified; selected factor counts, rotations, weights, clusters and latent-parameter ML inference are unsupported.",
             "P-values and inference degrees of freedom are unavailable; every requested replicate must pass the same geometry checks."]
    tables = {
        "estimates": c.frame(torch.stack((point, se, intervals[0], intervals[1],
                                           draws.mean(0) - point, undefined, undefined), dim=1),
                             columns=["estimate", "std_error", "ci_lower", "ci_upper", "bootstrap_bias", "p_value", "df"], index=labels),
        "covariance": c.frame(covariance, columns=labels, index=labels),
        "replicates": c.frame(draws, columns=labels, index=list(range(1, replications + 1))),
        "replicate_diagnostics": c.frame(diagnostics, columns=["leading_eigenvalue_gap", "minimum_uniqueness"], index=list(range(1, replications + 1))),
        "point_loadings": c.frame(point[:p, None], columns=["Factor1"], index=names),
        "point_uniqueness": c.frame(point[p:, None], columns=["uniqueness"], index=names),
        "descriptives": point_fit["descriptives"].copy(),
        "sample": c.frame(x, columns=names, index=positions),
    }
    return TableSet(tables, title="One-factor principal-factor iid bootstrap uncertainty",
                    procedure="factor_bootstrap", method=method, factors=1, rotate=None,
                    n=n, n_missing=dropped, physical_rows=rows, variables=names,
                    replications=replications, successful_replications=replications,
                    failed_replications=[], confidence=confidence, seed=seed,
                    sign_anchor=fixed_anchor, anchor_selection="strongest point loading" if anchor is None else "caller declared",
                    point_eigenvalue_gap=point_gap, missing="listwise", sample_positions=positions,
                    original_index=[_index_label(value) for value, kept in zip(keep.index, keep.tolist(), strict=True) if kept],
                    source_content_sha256=digest.hexdigest(), precision="float64", device="cpu",
                    rng="torch.Generator CPU randint", rng_version=torch.__version__,
                    covariance_divisor=replications - 1, quantile_interpolation="linear",
                    uncertainty="iid nonparametric marginal percentile bootstrap",
                    inference_target="fixed one-factor principal-factor estimator functional",
                    state_schema="openecon.factor_bootstrap.v1",
                    inferential_assumptions="iid complete-case rows; fixed p; finite fourth moments; population correlation positive definite away from singularity; separated leading reduced-matrix eigenvalue; fixed sign anchor away from zero; interior positive uniqueness; population conditions assumed, not empirically verified",
                    p_values_available=False, inference_df_available=False,
                    resource_plan=plan, estimated_work=work, notes=notes)
