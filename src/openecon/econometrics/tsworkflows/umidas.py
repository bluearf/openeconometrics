"""Unrestricted MIDAS: native OLS on explicitly released high-frequency lags."""

from __future__ import annotations

from copy import deepcopy
import math

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, column_list, make_spec, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.common import finite_tensor, pd_check
from openecon.econometrics.tsworkflows.midas import _alignment
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle
from openecon.resources import plan_workspace


def bound_alignment(record, data, names):
    """Validate both the existing hash binding and the exact dated lag geometry."""
    try:
        _alignment(record, data, names)
        offset = pd.tseries.frequencies.to_offset(record["frequency"])
        if offset.n <= 0:
            raise ValueError("nonpositive calendar")
        for row in record["rows"]:
            origin = pd.Timestamp(row["origin"])
            dates = [pd.Timestamp(v) for v in row["observation_times"]]
            releases = [pd.Timestamp(v) for v in row["release_times"]]
            if not all(offset.is_on_offset(date) for date in dates) or dates != [
                dates[0] - j * offset for j in range(len(names))
            ]:
                raise ValueError("lag calendar does not match the declared frequency")
            if any(release < date or release > origin for date, release in zip(dates, releases, strict=True)):
                raise ValueError("invalid observation/publication timing")
    except AnalysisError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise AnalysisError("invalid_alignment", "Preserve a complete, regular, release-bound MIDAS alignment.") from exc


def _resident(data):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "U-MIDAS requires a resident release-aligned table; no Dataset collection is performed.")
    return _coerce_frame(data)


def _saved_alignment(result, state, names):
    """Validate retained calendar structure independently of its duplicate copy.

    A new evaluation table cannot recompute the original training-data hash.
    Its own complete alignment is bound separately below; this gate validates
    every dated training row and the original/fitted physical-position maps.
    """
    record = state.get("alignment")
    try:
        if not isinstance(record, dict) or record != result.spec.options.get("alignment"):
            raise ValueError("alignment copies differ")
        if type(record.get("schema")) is not int or record["schema"] != 1 or record.get("lags") != names:
            raise ValueError("alignment schema or lag names differ")
        rows = record.get("rows")
        if not isinstance(rows, list) or len(rows) != result.nobs_original or not rows:
            raise ValueError("complete original alignment rows are missing")
        positions = record.get("original_positions")
        dropped = record.get("dropped_positions")
        for values, count in ((positions, len(rows)), (dropped, None)):
            if (not isinstance(values, list) or count is not None and len(values) != count
                    or any(type(value) is not int or value < 0 for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid original alignment physical positions")
        if set(positions) & set(dropped):
            raise ValueError("retained and dropped alignment positions overlap")
        fitted = state.get("sample_alignment_positions")
        if (not isinstance(fitted, list) or fitted != result.sample_positions
                or len(fitted) != result.nobs or len(set(fitted)) != len(fitted)
                or any(type(value) is not int or not 0 <= value < len(rows) for value in fitted)):
            raise ValueError("fitted physical positions do not select original alignment rows")
        low_time, origin_column = record.get("low_time"), record.get("origin_column")
        if any(not isinstance(value, str) or not value for value in (low_time, origin_column, record.get("high_value"))):
            raise ValueError("source calendar column names are incomplete")
        if record.get("binding_columns") != list(dict.fromkeys([low_time, origin_column, *names])):
            raise ValueError("source binding columns do not match the dated lag design")
        digest = record.get("data_hash")
        if not isinstance(digest, str) or len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError("source alignment digest is malformed")
        if record.get("lookahead") != "observation and release timestamps <= explicit row origin":
            raise ValueError("source publication contract is missing")
        if not isinstance(record.get("frequency"), str):
            raise ValueError("source frequency is missing")
        offset = pd.tseries.frequencies.to_offset(record["frequency"])
        if offset.n <= 0:
            raise ValueError("source frequency is nonpositive")
        plan_workspace("saved U-MIDAS dated history validation", {
            "retained_dates_and_positions": len(rows) * (len(names)+4) * 32,
        })
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("source calendar row is not a record")
            target, origin = pd.Timestamp(row["target"]), pd.Timestamp(row["origin"])
            if pd.isna(target) or pd.isna(origin):
                raise ValueError("source target or origin is missing")
            if (not isinstance(row.get("observation_times"), list)
                    or not isinstance(row.get("release_times"), list)
                    or len(row["observation_times"]) != len(names)
                    or len(row["release_times"]) != len(names)):
                raise ValueError("source lag/publication dates are incomplete")
            dates = [pd.Timestamp(value) for value in row["observation_times"]]
            releases = [pd.Timestamp(value) for value in row["release_times"]]
            if (any(pd.isna(value) for value in [*dates, *releases])
                    or not all(offset.is_on_offset(date) for date in dates)
                    or dates != [dates[0]-j*offset for j in range(len(names))]
                    or any(release < date or release > origin for date, release in zip(dates, releases, strict=True))):
                raise ValueError("source dated lags violate the frequency or publication contract")
        return record
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved U-MIDAS alignment or physical-position history is incomplete or changed: " + str(exc)) from exc


@resident_cpu
def fit_umidas(spec, data):
    from openecon.econometrics.registry import role_columns
    from openecon.linear_ols import fit_ols

    data = _resident(data)
    names = role_columns(spec, "lags")
    if not 3 <= len(names) <= 256 or len(set(names)) != len(names) or set(names) & set(spec.predictors):
        raise AnalysisError("invalid_lags", "U-MIDAS requires 3..256 distinct released lags, separate from controls.")
    if len(spec.predictors) > 64:
        raise AnalysisError("model_budget", "U-MIDAS admits at most 64 numeric controls.")
    if spec.intercept and "Intercept" in [*spec.predictors, *names]:
        raise AnalysisError("duplicate_terms", "With an intercept, rename any literal control/lag column named Intercept.")
    alignment = spec.options["alignment"]
    bound_alignment(alignment, data, names)
    frame = ModelFrame(spec, data)
    n, k = frame.n, frame.design_width()+len(names)
    if n > 20000 or n * k * k + k**3 > 100000000:
        raise AnalysisError("model_budget", "U-MIDAS resident design/QR geometry exceeds its supported work budget.")
    plan = frame.workspace_plan("U-MIDAS full-rank QR and inference", {
        "design_qr_and_covariance": n * k * 64 + k * k * 128,
        "alignment_snapshot": len(data) * (len(names) + 4) * 160,
    })
    if n <= k:
        raise AnalysisError("insufficient_observations", "U-MIDAS requires positive residual degrees of freedom.")
    controls = frame.design()
    lag = torch.stack([frame.numeric(name) for name in names], 1)
    design = torch.cat([controls.x, lag], 1)
    if int(torch.linalg.matrix_rank(design)) != k:
        raise AnalysisError("rank_deficient", "Every unrestricted lag/control coefficient must be identified; no columns are silently omitted.")
    # Native OLS owns the sample, covariance corrections and t reference. Time
    # and release labels describe alignment, not a lag-transformed OLS design.
    ols_spec = make_spec("ols", outcome=spec.outcome,
                         predictors=[*spec.predictors, *names], intercept=spec.intercept,
                         covariance=spec.covariance, missing=spec.missing, alpha=spec.alpha)
    raw = fit_ols(ols_spec, data=data, device="cpu")
    terms = [*controls.terms, *names]
    if [c.term for c in raw.coefficients] != terms or raw.sample_positions != frame.positions:
        raise AnalysisError("incomparable_sample", "The native U-MIDAS QR result must retain every declared term and exact response row.")
    payload = raw.model_dump(mode="json")
    payload.update(spec=spec.model_dump(mode="json"), title="Unrestricted MIDAS")
    payload["extra"] = {"umidas_state": {
        "schema": 1, "alignment": deepcopy(alignment), "lag_columns": names,
        "terms": terms, "sample_alignment_positions": raw.sample_positions,
        "lag_coefficients": [c.estimate for c in raw.coefficients[-len(names):]],
        "coefficient_constraint": "each released lag has a separate unrestricted coefficient",
    }}
    payload["provenance"].update(
        estimator="umidas", family="tsworkflows", device="cpu", precision="float64",
        forecast_origin_contract=alignment["lookahead"],
        parameterization="unrestricted numeric high-frequency lag coefficients; native OLS",
        resource_plans=[plan.record()],
    )
    # Return the public saved contract rather than private OLS prediction methods
    # that would bypass the explicit release checks of umidas_predict.
    return ResultBundle.model_validate(payload)


def umidas(*, data, y, x=None, lags=None, alignment=None, time=None,
           intercept=True, covariance="nonrobust", missing="raise", alpha=0.05):
    """Fit U-MIDAS with numeric controls and an explicit midas_align calendar."""
    alignment = alignment if alignment is not None else getattr(data, "attrs", {}).get("midas_alignment")
    if lags is None and isinstance(alignment, dict):
        lags = alignment.get("lags")
    spec = make_spec("umidas", outcome=y, predictors=column_list(x, "x"),
                     columns={"lags": column_list(lags, "lags")}, time=time,
                     intercept=intercept, covariance=covariance, missing=missing,
                     alpha=alpha, options={"alignment": alignment})
    return fit_umidas(spec, data)


@resident_cpu
def umidas_predict(result, *, data, alignment=None, alpha=None, kind="mean"):
    """Saved U-MIDAS means/t intervals, with separately validated new release lags.

    Mean intervals use the saved full OLS covariance and residual df. Outcome
    intervals additionally require the conventional independent Gaussian noise
    law; that route is refused for heteroskedastic covariance.
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "umidas":
        raise AnalysisError("invalid_result", "Supply a saved U-MIDAS ResultBundle.")
    if kind not in {"mean", "outcome"}:
        raise AnalysisError("invalid_prediction", "Use U-MIDAS mean or outcome prediction.")
    significance = result.spec.alpha if alpha is None else alpha
    if isinstance(significance, bool) or not isinstance(significance, (int, float)) or not 0 < significance < 1:
        raise AnalysisError("invalid_alpha", "alpha must be strictly between zero and one.")
    state = result.extra.get("umidas_state")
    names = result.spec.columns.get("lags")
    terms = (["Intercept"] if result.spec.intercept else []) + result.spec.predictors + (names if isinstance(names, list) else [])
    if not isinstance(names, list) or not 3 <= len(names) <= 256 or len(set(names)) != len(names) or not isinstance(state, dict) or state.get("schema") != 1 or state.get("lag_columns") != names or state.get("terms") != terms or [c.term for c in result.coefficients] != terms:
        raise AnalysisError("invalid_state", "Saved U-MIDAS lag/term order must match its specification and coefficients.")
    saved_alignment = _saved_alignment(result, state, names)
    raw = _resident(data)
    if not 1 <= len(raw) <= 20000 or raw.columns.has_duplicates or any(name not in raw for name in [*result.spec.predictors, *names]):
        raise AnalysisError("invalid_predictors", "Supply 1..20000 rows with all distinct saved control/lag columns.")
    alignment = alignment if alignment is not None else getattr(data, "attrs", {}).get("midas_alignment")
    bound_alignment(alignment, raw, names)
    if alignment["frequency"] != saved_alignment["frequency"]:
        raise AnalysisError("invalid_alignment", "New U-MIDAS lags must use the saved high-frequency calendar.")
    k = len(terms)
    if len(raw) * k * k > 100000000:
        raise AnalysisError("model_budget", "U-MIDAS prediction covariance geometry exceeds its work budget.")
    plan_workspace("saved U-MIDAS means and covariance", {"design_and_variance": len(raw) * k * 32 + k * k * 64})
    design = torch.stack([torch.ones(len(raw), dtype=torch.float64) if term == "Intercept" and result.spec.intercept
                          else _numeric(raw[term], term) for term in terms], 1)
    beta = finite_tensor([c.estimate for c in result.coefficients], (k,), "saved coefficients")
    covariance = finite_tensor(result.covariance_matrix, (k, k), "saved covariance")
    pd_check(covariance, "saved covariance", semidefinite=True)
    mean = design @ beta
    variance = (design @ covariance * design).sum(1)
    df = result.inference.get("df_inference")
    if isinstance(df, bool) or not isinstance(df, (int, float)) or not math.isfinite(df) or df <= 0 or result.inference.get("use_t") is not True:
        raise AnalysisError("invalid_state", "Saved U-MIDAS predictions require positive native OLS residual df.")
    if kind == "outcome":
        if result.spec.covariance != "nonrobust":
            raise AnalysisError("unsupported_uncertainty", "Independent Gaussian outcome intervals require nonrobust U-MIDAS covariance.")
        sigma = result.metrics.get("rmse")
        if not isinstance(sigma, (int, float)) or not math.isfinite(sigma) or sigma <= 0:
            raise AnalysisError("invalid_state", "Saved U-MIDAS Gaussian residual scale must be positive and finite.")
        variance += sigma**2
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise AnalysisError("non_finite_prediction", "Saved U-MIDAS mean/variance is invalid; rescale inputs.")
    se = variance.sqrt()
    critical = critical_value(significance, df)
    return table({"row": list(range(len(raw))), "mean": mean.tolist(),
                  "std_error": se.tolist(), "ci_low": (mean-critical*se).tolist(),
                  "ci_high": (mean+critical*se).tolist()}, index=raw.index,
                 alpha=significance, distribution="t", df=df, kind=kind,
                 target="conditional released-lag mean" if kind == "mean" else "independent Gaussian new outcome",
                 uncertainty="full saved coefficient covariance" + (" plus independent Gaussian residual noise" if kind == "outcome" else "; outcome noise excluded"),
                 alignment=deepcopy(alignment), source_result_id=result.id,
                 precision="float64", device="cpu", refitted=False)
