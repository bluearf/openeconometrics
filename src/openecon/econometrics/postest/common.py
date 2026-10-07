"""Shared pieces of the post-estimation procedures.

Every procedure in this package takes finished ``ResultBundle`` objects. The
helpers here read what a bundle records (coefficients, log likelihood, sample
positions, data hash), check that a dataset is the one a result was fitted on,
refit a specification through the registry and rebuild a bundle around a new
covariance matrix so that standard errors, test statistics, p-values,
confidence intervals, the inference record and the provenance always agree.

pandas is used only to select rows and columns of the user's data; all
numerical work is float64 Torch.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame, _frame_hasher
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import Coefficient, ModelSpec, ResultBundle

# Provenance keys that describe inference of the ORIGINAL covariance (OLS stores
# standard errors and the model F test there); they are removed when a
# post-estimation procedure replaces the covariance.
_STALE_PROVENANCE = ("inference_details",)
# Metric names whose value depends on the covariance matrix (model Wald/F tests).
_STALE_METRIC_MARKERS = ("wald", "f_stat", "fstat", "chi2", "p_value", "f_test")
# Covariances under which the log likelihood is a genuine likelihood (Stata's
# lrtest refuses vce(robust), vce(cluster) and pweights unless force).
LIKELIHOOD_COVARIANCES = frozenset({"nonrobust", "opg", "bootstrap", "jackknife"})
# The flag a combined suest bundle carries; such a bundle is not a single fit.
SUEST_FLAG = "suest"


def require_result(value: Any, role: str) -> ResultBundle:
    """The argument as a ResultBundle, or ``invalid_result``."""
    if not isinstance(value, ResultBundle):
        raise AnalysisError("invalid_result", f"The {role} must be a fitted OpenEconometrics result "
                            "(the ResultBundle returned by an estimator such as oe.logit).")
    return value


def is_suest(result: ResultBundle) -> bool:
    return (result.provenance.get("postestimation") or {}).get("method") == SUEST_FLAG


def log_likelihood(result: ResultBundle) -> float | None:
    value = result.metrics.get("log_likelihood")
    return None if value is None or not math.isfinite(value) else float(value)


def null_log_likelihood(result: ResultBundle) -> float | None:
    """The constant-only log likelihood a result reports (Stata's ``e(ll_0)``), if any.

    Read from ``metrics['log_likelihood_null']`` / ``metrics['ll_null']``,
    ``extra['null_log_likelihood']``, or for the core logit/probit fits from the
    McFadden pseudo R-squared they store (``ll_0 = ll / (1 - R2_p)``, the exact
    inverse of how ``openecon.analysis`` computes it). For an unweighted or
    frequency-weighted OLS fit with a constant, ``ll_0 = ll + (N/2) ln(1 - R^2)``.
    """
    for key in ("log_likelihood_null", "ll_null"):
        value = result.metrics.get(key)
        if value is not None and math.isfinite(value):
            return float(value)
    value = result.extra.get("null_log_likelihood")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    ll, pseudo = log_likelihood(result), result.metrics.get("pseudo_r_squared")
    if (result.spec.estimator in {"logit", "probit"} and ll is not None and pseudo is not None
            and pseudo < 1):
        return ll / (1 - pseudo)
    spec, r_squared = result.spec, result.metrics.get("r_squared")
    if (spec.estimator == "ols" and spec.intercept and spec.weight_type in {None, "fweight"}
            and ll is not None and r_squared is not None and 0 <= r_squared < 1):
        # Gaussian log likelihoods of the model and of the constant-only model differ by
        # (N/2) ln(RSS/TSS) = (N/2) ln(1 - R^2) (Stata's e(ll_0) after regress).
        return ll + 0.5 * result.nobs * math.log1p(-r_squared)
    return None


def parameter_count(result: ResultBundle) -> int:
    """Number of estimated parameters (Stata's ``e(rank)``): the reported coefficients.

    Terms omitted for collinearity are not reported and therefore not counted.
    """
    return len(result.coefficients)


def coefficient_vector(result: ResultBundle) -> Tensor:
    return torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)


def covariance_matrix(result: ResultBundle) -> Tensor:
    return torch.tensor(result.covariance_matrix, dtype=torch.float64)


def result_covariance(result: ResultBundle) -> str:
    """The covariance a result actually carries (a post-estimation bundle records it)."""
    return str(result.inference.get("covariance") or result.spec.covariance)


def sample_key(result: ResultBundle) -> tuple[Any, ...]:
    """An identifier of the estimation rows (streamed OLS fits record only their hash)."""
    if result.provenance.get("sample_positions_omitted"):
        return ("hash", result.provenance.get("sample_positions_hash"))
    return ("rows", *sorted(result.sample_positions))


def label(result: ResultBundle) -> str:
    return f"{result.spec.estimator} of {result.spec.outcome}"


# ---- data checks ----------------------------------------------------------------


def input_columns(result: ResultBundle) -> list[str]:
    columns = result.provenance.get("input_columns")
    if isinstance(columns, list) and columns and all(isinstance(c, str) for c in columns):
        return list(columns)
    return registry.spec_columns(result.spec)


def _streamed_hash(frame: pd.DataFrame) -> str:
    """The data hash of a streamed (out-of-core) OLS fit: SHA-256 over the row hashes."""
    import hashlib

    from openecon.streaming_design import row_hash_bytes

    digest = hashlib.sha256()
    digest.update(row_hash_bytes(frame))
    return digest.hexdigest()


def _streamed_positions(result: ResultBundle, frame: pd.DataFrame, role: str) -> list[int]:
    """Estimation rows of a streamed fit, recovered and verified against its position hash.

    A streamed OLS fit records only the count and a SHA-256 of its estimation
    rows. The candidate rows are those with complete model inputs (and a
    positive weight); they are accepted only if count and hash agree.
    """
    import hashlib

    columns = input_columns(result)
    keep = frame.loc[:, columns].notna().all(axis=1)
    weights = result.spec.weights
    if weights is not None and weights in frame.columns:
        keep &= pd.to_numeric(frame[weights], errors="coerce").fillna(0) > 0
    candidate = torch.as_tensor(keep.to_numpy().nonzero()[0], dtype=torch.int64)
    digest = hashlib.sha256()
    digest.update(candidate.numpy().tobytes())
    if (len(candidate) != result.provenance.get("sample_position_count")
            or digest.hexdigest() != result.provenance.get("sample_positions_hash")):
        raise AnalysisError("sample_positions_unavailable", f"The {role} was fitted out of core "
                            "and its estimation rows could not be recovered from the data; refit "
                            "it on the complete-case table (drop rows with missing values first).")
    return candidate.tolist()


def matched_frame(result: ResultBundle, data: Any,
                  role: str = "result") -> tuple[pd.DataFrame, list[int]]:
    """The user's data as a DataFrame and the estimation rows of ``result`` in it.

    The data are checked to be the dataset ``result`` was fitted on: the hash
    of the model-input columns in positional row order (index labels excluded)
    is recomputed exactly as the estimator recorded it in
    ``provenance['data_hash']``. A different, filtered, re-sorted or edited
    dataset is refused with ``data_mismatch``. Streamed OLS fits, which do not
    store their rows, are supported when the rows can be recovered and match
    the recorded position hash.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "Post-estimation procedures need the "
                            "in-memory table the model was fitted on, not an out-of-core dataset.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    columns = input_columns(result)
    absent = [name for name in columns if name not in frame.columns]
    if absent:
        raise AnalysisError("data_mismatch", f"The data lack columns the {role} was fitted on: "
                            f"{', '.join(absent)}. Pass the dataset used for estimation.")
    recorded = result.provenance.get("data_hash")
    if not isinstance(recorded, str):
        raise AnalysisError("data_mismatch", f"The {role} does not record a data hash, so the "
                            "dataset cannot be verified; refit it with a current OpenEconometrics.")
    streamed = bool(result.provenance.get("sample_positions_omitted"))
    projected = frame.loc[:, columns]
    observed = _streamed_hash(projected) if streamed else _frame_hasher(projected).hexdigest()
    if len(frame) != result.nobs_original or observed != recorded:
        raise AnalysisError("data_mismatch", f"The data differ from the dataset the {role} was "
                            "fitted on (rows, order or values of the model columns changed). Pass "
                            "exactly the table used for estimation.")
    positions = (_streamed_positions(result, frame, role) if streamed
                 else list(result.sample_positions))
    if not positions or max(positions) >= len(frame):
        raise AnalysisError("data_mismatch", f"The {role}'s estimation rows do not exist in the "
                            "data.")
    return frame, positions


# ---- refitting ---------------------------------------------------------------------


def refit_spec(result: ResultBundle) -> ModelSpec:
    """The specification each replicate is refitted with: ``result.spec`` with data-driven
    choices of the original fit pinned, so that every replicate estimates the same parameters.

    ``mlogit`` without ``base=`` takes the most frequent outcome as its base; a
    resample can make another category the most frequent and so change every
    coefficient's meaning. The refits use the base of the original fit
    (``extra['base']``), as Stata recommends ``baseoutcome()`` with resampling.
    """
    spec = result.spec
    if spec.estimator == "mlogit" and spec.options.get("base") is None \
            and result.extra.get("base") is not None:
        return spec.model_copy(update={"options": {**spec.options,
                                                   "base": result.extra["base"]}})
    return spec


def refitter(spec: ModelSpec):
    """A callable ``frame -> ResultBundle`` refitting ``spec`` without re-validation.

    Registry estimators are called through their entry function directly; the
    core estimators (ols, logit, probit) go through ``openecon.analysis.fit``,
    which owns their validation.
    """
    info = registry.get(spec.estimator)
    if info.legacy:
        from openecon.analysis import fit

        return lambda frame: fit(spec, data=frame)
    entry = registry.load_entry(info)
    return lambda frame: entry(spec, frame)


def refit_parameters(refit, frame: pd.DataFrame, terms: list[str]) -> tuple[Tensor | None, str]:
    """Parameters of one refit aligned with ``terms``, or ``(None, reason)`` on failure.

    A refit fails when the estimator raises an ``AnalysisError`` (or a kernel
    error) or when it reports different terms, which happens when a resample
    makes a regressor collinear and the term is omitted. A numerical exception
    that escapes an estimator on a degenerate resample (a singular solve, an
    overflow) is counted as a failed replicate rather than aborting the whole
    procedure; its type is recorded as the reason.
    """
    try:
        fitted = refit(frame)
    except (AnalysisError, KernelError) as exc:
        return None, getattr(exc, "code", "analysis_error")
    except (ArithmeticError, RuntimeError, ValueError, IndexError) as exc:
        return None, f"numerical_failure ({type(exc).__name__})"
    if [c.term for c in fitted.coefficients] != terms:
        return None, "different_terms"
    values = torch.tensor([c.estimate for c in fitted.coefficients], dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        return None, "non_finite_estimates"
    return values, ""


# ---- rebuilding a bundle around a new covariance ----------------------------------


def coefficients(result: ResultBundle, params: Tensor, covariance: Tensor, *, alpha: float,
                 df: float | None, terms: list[str] | None = None,
                 equations: list[str | None] | None = None) -> list[Coefficient]:
    """Coefficient rows (estimate, SE, z or t, p-value, CI) from ``params`` and ``covariance``."""
    covariance = (covariance + covariance.T) / 2
    variances = covariance.diagonal()
    if not bool(torch.isfinite(params).all()) or not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("non_finite_result", "The procedure produced non-finite estimates "
                            "or covariance.")
    if bool((variances <= 0).any()):
        bad = [(terms or [c.term for c in result.coefficients])[i]
               for i in (variances <= 0).nonzero().flatten().tolist()]
        raise AnalysisError("invalid_covariance", "The covariance gives no positive variance for: "
                            f"{', '.join(bad)} (the estimate did not vary across replicates).")
    errors = variances.sqrt()
    statistics = params / errors
    try:
        p_values = two_sided_p_values(statistics, df)
        critical = critical_value(alpha, df)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    low, high = params - critical * errors, params + critical * errors
    terms = terms or [c.term for c in result.coefficients]
    equations = equations or [c.equation for c in result.coefficients]
    return [Coefficient(term=term, estimate=float(params[i]), std_error=float(errors[i]),
                        statistic=float(statistics[i]), p_value=float(p_values[i]),
                        ci_low=float(low[i]), ci_high=float(high[i]), equation=equations[i])
            for i, term in enumerate(terms)]


def json_safe(value: Any) -> Any:
    from openecon.econometrics.core import _json_safe

    return _json_safe(value)


def rebuild(result: ResultBundle, covariance: Tensor, *, method: str, use_t: bool,
            df_inference: float | None, correction: str, inference: dict[str, Any],
            provenance: dict[str, Any], extra: dict[str, Any], tests: dict[str, Any],
            warnings: list[str], title: str, spec: ModelSpec | None = None) -> ResultBundle:
    """A new plain ResultBundle: the point estimates of ``result`` with ``covariance``.

    Coefficient standard errors, statistics, p-values and intervals are
    recomputed; covariance-dependent metrics, the original specification tests
    and the original ``extra`` output are not carried over (they would describe
    the original covariance); the inference record is rebuilt and the
    provenance drops the original inference details. ``result.spec`` is kept
    (only ``alpha`` may be replaced through ``spec``): it describes the point
    estimates, while ``inference['covariance']`` names the new covariance.
    """
    spec = result.spec if spec is None else spec
    params = coefficient_vector(result)
    alpha = spec.alpha
    rows = coefficients(result, params, covariance, alpha=alpha,
                        df=df_inference if use_t else None)
    original = result_covariance(result)
    record: dict[str, Any] = {
        "covariance": method, "use_t": use_t, "distribution": "t" if use_t else "normal",
        "alpha": alpha, "confidence_level": 1 - alpha,
        "df_resid": result.inference.get("df_resid"),
        "df_inference": df_inference if use_t else None, "cluster_count": None,
        "cluster_df": None, "cluster_column": None, "small_sample_correction": None,
        "correction": correction, "intercept": spec.intercept,
        "n_parameters": len(rows), "residual_definition": "observed minus fitted response",
        "original_covariance": original,
    }
    record.update(inference)
    record_provenance = {key: value for key, value in result.provenance.items()
                         if key not in _STALE_PROVENANCE}
    record_provenance.update(provenance)
    record_provenance["stata_parity_validated"] = False
    metrics = {name: value for name, value in result.metrics.items()
               if not any(marker in name for marker in _STALE_METRIC_MARKERS)}
    notes = list(result.warnings)
    if result.tests:
        notes.append("Specification tests of the original fit were not carried over (they may "
                     "depend on the original covariance); see the original result.")
    notes.extend(note for note in warnings if note not in notes)
    return ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=spec,
        nobs=result.nobs, nobs_original=result.nobs_original, dropped_rows=result.dropped_rows,
        coefficients=rows, covariance_matrix=((covariance + covariance.T) / 2).tolist(),
        metrics=metrics, warnings=notes, predictions=list(result.predictions),
        sample_positions=list(result.sample_positions), provenance=json_safe(record_provenance),
        inference=json_safe(record), title=title, tests=json_safe(tests),
        extra=json_safe(extra),
    )


def model_wald(result: ResultBundle, covariance: Tensor, *, df_resid: float | None,
               label_text: str) -> dict[str, Any]:
    """Joint Wald test of all coefficients except constants and ancillary parameters."""
    from openecon.econometrics.core import wald_test

    names = [c.term.split(":")[-1] for c in result.coefficients]
    indices = [i for i, name in enumerate(names)
               if name != "Intercept" and not name.startswith("/")]
    if not indices:
        return {}
    return {"model": wald_test(coefficient_vector(result), covariance, indices,
                               df_resid=df_resid, label=label_text)}
