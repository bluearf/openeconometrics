"""Bounded public fitting for replayable local datasets.

OLS, logit and probit replay bounded batches, including categorical predictors
and disk-backed one-way cluster inference. Each pass
hashes projected input values and retained positions incrementally. Neither the
design nor sample-position provenance grows with the total observation count.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import Coefficient, ModelSpec, ResultBundle

WORKING_BYTES = 128 * 1024 * 1024
MAX_BATCH_ROWS = 65_536
MAX_PARAMETERS = 384
SAMPLE_LIMIT = 400
CLUSTER_WORKING_BYTES = 12 * 1024 * 1024


def batch_rows_for(parameters: int, *, cluster: bool = False) -> int:
    """Budget factor and batch scratch storage, independently of total rows."""
    factor_budget = 80 * (parameters + 1) ** 2 * 8 + (CLUSTER_WORKING_BYTES if cluster else 0)
    if parameters > MAX_PARAMETERS or factor_budget >= WORKING_BYTES:
        raise AnalysisError("model_too_wide", "The model's factor matrices exceed the bounded working-memory budget; reduce the number of predictors.")
    return max(1, min(MAX_BATCH_ROWS, (WORKING_BYTES - factor_budget) // (8 * max(1, parameters) * 32)))


class _Batches:
    def __init__(self, source: Dataset, spec: ModelSpec):
        from openecon.streaming_design import StreamingDesign
        self.source, self.spec = source, spec
        clustered = spec.covariance == "cluster"
        self.reader_rows = batch_rows_for(len(spec.predictors) + int(spec.intercept), cluster=clustered)
        self.design = StreamingDesign(source, spec, self.reader_rows).prepare()
        self.columns = self.design.columns
        self.rows_per_batch = batch_rows_for(len(self.design.terms), cluster=clustered)
        self.baseline = self.design.baseline
        self.sample_rows = self.design.sample_rows
        self.passes = int(self.baseline is not None)
        self.max_encoded_rows = 0

    def __call__(self, include_clusters: bool = True):
        from openecon.streaming_design import encode_cluster_labels, row_hash_bytes
        digest = hashlib.sha256()
        digest.update(json.dumps({"columns": self.columns, "missing": self.spec.missing},
                                 sort_keys=True, separators=(",", ":")).encode())
        sample_digest = hashlib.sha256()
        original_count = used_count = 0
        sample_rows = []
        self.source.assert_unchanged()
        iterator = self.source.iter_batches(self.columns, batch_rows=self.reader_rows)
        try:
            for batch in iterator:
                if batch.empty:
                    continue
                if len(batch) > self.reader_rows:
                    raise AnalysisError("batch_too_large", "The dataset returned a batch larger than its working-memory budget.")
                projected = batch.loc[:, self.columns]
                # Validate category declarations before pandas can hash an
                # unbounded unused dictionary supplied by a changed factory.
                valid = self.design.validate_batch(projected)
                digest.update(row_hash_bytes(projected))
                keep = ~projected.isna().any(axis=1)
                offsets = torch.from_numpy(keep.to_numpy()).nonzero().flatten().to(torch.int64)
                positions = offsets + original_count
                sample_digest.update(positions.numpy().astype("<i8", copy=False).tobytes())
                if len(sample_rows) < SAMPLE_LIMIT:
                    sample_rows.extend(positions[:SAMPLE_LIMIT - len(sample_rows)].tolist())
                original_count += len(projected)
                used_count += len(valid)
                start = 0
                block_rows = self.rows_per_batch
                while start < len(valid):
                    block = valid.iloc[start:start + block_rows]
                    keys = None
                    if self.spec.covariance == "cluster" and include_clusters:
                        try:
                            keys = encode_cluster_labels(block[self.spec.cluster])
                        except AnalysisError as exc:
                            if exc.code == "cluster_key_budget" and block_rows > 1:
                                block_rows = max(1, block_rows // 2)
                                continue
                            raise
                    design, outcome = self.design.encode(block)
                    self.max_encoded_rows = max(self.max_encoded_rows, len(block))
                    yield (design, outcome, keys) if keys is not None else (design, outcome)
                    start += len(block)
        finally:
            iterator.close()
        self.source.assert_unchanged()
        current = {"original": original_count, "used": used_count,
                   "data_hash": digest.hexdigest(), "positions_hash": sample_digest.hexdigest()}
        if self.baseline is not None and current != self.baseline:
            raise AnalysisError("source_changed", "Dataset values or selected observations changed between regression passes; no result was produced.")
        self.baseline = current
        self.sample_rows = sample_rows
        self.passes += 1


def fit_streaming(spec: ModelSpec, source: Dataset) -> ResultBundle:
    from openecon import __version__
    from openecon.engines.streaming_ols import solve_ols
    if spec.estimator not in {"ols", "logit", "probit"}:
        raise AnalysisError("unsupported_streaming_model", "Streaming estimation supports OLS, logit and probit.")
    from openecon.analysis import _MAX_BINARY_ITERATIONS
    batches = _Batches(source, spec)
    terms = batches.design.terms
    scratch = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
    options = {"covariance": spec.covariance, "intercept": spec.intercept,
               "sample_limit": SAMPLE_LIMIT, "scratch_directory": Path(scratch) if scratch else None}
    if spec.covariance == "cluster":
        # Every replay still hashes the cluster column and drops missing labels.
        # Canonical group keys are needed only for the covariance accumulation.
        options["numeric_batch_factory"] = lambda: batches(include_clusters=False)
    try:
        if spec.estimator == "ols":
            solution = solve_ols(batches, **options)
        else:
            from openecon.engines.streaming_binary import solve_binary
            solution = solve_binary(spec.estimator, batches, max_iter=_MAX_BINARY_ITERATIONS,
                                    tolerance=1e-10, **options)
    except KernelError as exc:
        if batches.baseline is not None and batches.baseline["used"] == 0:
            code = "empty_sample" if batches.baseline["original"] else "empty_data"
            raise AnalysisError(code, "The dataset has no complete observations for estimation.") from exc
        raise AnalysisError(exc.code, str(exc)) from exc
    n, k = solution.nobs, solution.nparams
    parameters, covariance = solution.parameters, solution.covariance
    if not all(bool(torch.isfinite(value).all()) for value in (parameters, covariance)):
        raise AnalysisError("non_finite_result", "Streaming estimation did not produce finite coefficients and covariance.")
    if bool((covariance.diag() <= 0).any()):
        raise AnalysisError("invalid_covariance", "The fitted covariance does not provide strictly positive standard errors.")
    standard_errors = covariance.diag().sqrt()
    statistics = parameters / standard_errors
    group_count = solution.group_count
    use_t = spec.estimator == "ols"
    inference_df = (group_count - 1 if group_count is not None else n - k) if use_t else None
    p_values = two_sided_p_values(statistics, inference_df)
    critical = critical_value(spec.alpha, inference_df)
    intervals = torch.stack((parameters - critical * standard_errors,
                             parameters + critical * standard_errors), dim=1)
    if not all(bool(torch.isfinite(value).all()) for value in (standard_errors, statistics, p_values, intervals)):
        raise AnalysisError("non_finite_result", "Streaming estimation did not provide finite statistical inference.")
    coefficients = [Coefficient(term=term, estimate=float(parameters[i]),
                                std_error=float(standard_errors[i]), statistic=float(statistics[i]),
                                p_value=float(p_values[i]), ci_low=float(intervals[i, 0]),
                                ci_high=float(intervals[i, 1])) for i, term in enumerate(terms)]
    r_squared = (1 - solution.residual_ss / solution.total_ss
                 if use_t and solution.total_ss > 0 else None)
    adjusted = (1 - (n - int(spec.intercept)) / (n - k) * (1 - r_squared)
                if r_squared is not None else None)
    pseudo = None if use_t else 1 - solution.log_likelihood / solution.null_log_likelihood
    correction = (group_count / (group_count - 1) * (n - 1) / (n - k)
                  if group_count is not None else (n / (n - k) if spec.covariance == "HC1" else None))
    metrics = {"r_squared": r_squared, "adjusted_r_squared": adjusted, "pseudo_r_squared": pseudo,
               "aic": -2 * solution.log_likelihood + 2 * k,
               "bic": -2 * solution.log_likelihood + math.log(n) * k,
               "log_likelihood": solution.log_likelihood,
               "rmse": math.sqrt(solution.residual_ss / n), "df_resid": n - k,
               "condition_number_scaled": solution.condition_number}
    warnings = []
    if solution.condition_number > 1e8:
        warnings.append("The centered/scaled design is ill-conditioned; coefficient and inference sensitivity should be reviewed.")
    if group_count is not None and group_count < 30:
        warnings.append(f"Only {group_count} clusters are available; cluster-robust inference may be unreliable.")
    baseline = batches.baseline
    if baseline["original"] != n:
        warnings.append(f"Excluded {baseline['original'] - n} observation(s) with missing model inputs.")
    sample_hash = hashlib.sha256((baseline["data_hash"] + baseline["positions_hash"]).encode()).hexdigest()
    predictions = [{"row": row, "observed": float(solution.observed[i]),
                    "fitted": float(solution.fitted[i]), "residual": float(solution.residuals[i])}
                   for i, row in enumerate(batches.sample_rows)]
    return ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=spec,
        nobs=n, nobs_original=baseline["original"], dropped_rows=baseline["original"] - n,
        coefficients=coefficients, covariance_matrix=covariance.tolist(),
        metrics={name: value if value is None or math.isfinite(value) else None
                 for name, value in metrics.items()}, warnings=warnings, predictions=predictions,
        sample_positions=[], provenance={
            "schema_version": "4", "backend": "openecon.torch", "engine": "openecon",
            "device": "cpu", "estimator": spec.estimator, "precision": "float64",
            "versions": {"openecon": __version__, "python": platform.python_version(),
                         "pandas": pd.__version__, "torch": torch.__version__},
            "data_hash": baseline["data_hash"], "sample_hash": sample_hash,
            "sample_positions_hash": baseline["positions_hash"],
            "hash_algorithm": "SHA-256 over projected pandas row hashes plus missing policy; selected position stream hashed separately; pandas version recorded",
            "hash_scope": "selected model-input columns in physical row order; index labels excluded",
            "input_columns": batches.columns, "design_terms": terms,
            "categorical_encoding": batches.design.categorical_encoding,
            "sample_positions_omitted": True, "sample_position_count": n, "sample_position_base": 0,
            "prediction_sample": "first retained positional observations", "prediction_limit": SAMPLE_LIMIT,
            "stata_parity_validated": False, "solver": solution.solver,
            "solver_diagnostics": solution.diagnostics, "optimizer": None if use_t else {
                "method": solution.solver, "maxiter": _MAX_BINARY_ITERATIONS, "tolerance": 1e-10,
                "iterations": solution.iterations, "converged": True,
            },
            "separation_diagnostic": None if use_t else "native Torch primal-dual constraint generation with global replay",
            "condition_number_basis": "centered_and_scaled_design" if spec.intercept else "scaled_design",
            "streaming": {"passes": batches.passes, "batch_rows": batches.rows_per_batch,
                          "reader_batch_rows": batches.reader_rows,
                          "maximum_encoded_batch_rows": batches.max_encoded_rows,
                          "working_memory_budget_bytes": WORKING_BYTES,
                          "source": {name: value for name, value in source.provenance.items()
                                     if name != "path"}, "row_limit": None},
        }, inference={
            "covariance": spec.covariance, "use_t": use_t, "distribution": "t" if use_t else "normal",
            "alpha": spec.alpha, "confidence_level": 1 - spec.alpha, "df_resid": n - k,
            "df_inference": inference_df, "cluster_count": group_count,
            "cluster_df": group_count - 1 if group_count is not None else None,
            "cluster_column": spec.cluster, "small_sample_correction": correction,
            "correction": ("CR1: G/(G-1) * (N-1)/(N-K)" if group_count else
                           ("HC1: N/(N-K)" if spec.covariance == "HC1" else spec.covariance)),
            "intercept": spec.intercept, "n_parameters": k,
            "residual_definition": "observed minus fitted response",
        },
    )
