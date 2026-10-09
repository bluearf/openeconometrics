"""Structural weak-IV inference from restored specs and verified original data."""

import math

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.econometrics.iv.weak_inference import iv_weak_test, iv_ar_confidence_set


def _saved(result, data):
    result = require_result(result, "saved IV result")
    s = result.spec
    if s.estimator != "ivregress" or s.weights or s.categorical or s.panel:
        raise AnalysisError(
            "unsupported_spec",
            "Saved structural tests support numeric unweighted ivregress specifications.",
        )
    frame, positions = matched_frame(result, data)
    options = dict(
        data=frame.iloc[positions].copy(),
        y=s.outcome,
        x=s.predictors,
        endog=s.columns["endogenous"],
        instruments=s.columns["instruments"],
        intercept=s.intercept,
        missing="raise",
    )
    return result, positions, options


def iv_saved_weak_test(result, *, data, null, method="ar", covariance=None, alpha=None):
    """Test a complete structural null using a restored IV spec and matched data.

    No IV coefficient fit is rerun. AR permits iid/HC0/one-way cluster/HAC;
    scalar CLR requires iid homoskedasticity. Original rows/order/model columns
    must match the saved hash. Matrices and full null covariance are preserved.
    """
    result, positions, options = _saved(result, data)
    s = result.spec
    kind = s.covariance if covariance is None else covariance
    level = s.alpha if alpha is None else alpha
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(level)
        or not 0 < level < 1
    ):
        raise AnalysisError("invalid_option", "alpha must lie strictly between zero and one.")
    output = iv_weak_test(
        **options,
        null=null,
        method=method,
        covariance=kind,
        cluster=s.cluster if kind == "cluster" else None,
        time=s.time if kind == "hac" else None,
        lags=s.options.get("lags") if kind == "hac" else None,
        kernel=s.options.get("kernel") or "bartlett",
    )
    output.attrs.update(
        source_result_id=result.id,
        original_sample_positions=positions,
        source_data_hash=result.provenance["data_hash"],
        iv_refitted=False,
        alpha=level,
    )
    return output


def iv_saved_ar_confidence_set(result, *, data, alpha=None):
    """Analytic scalar iid AR inversion using a restored IV spec, without refitting.

    Empty, disjoint and unbounded sets are preserved. Robust/HAC/cluster or
    multivariate confidence-set inversion is explicitly outside this domain.
    """
    result, positions, options = _saved(result, data)
    if result.spec.covariance != "nonrobust":
        raise AnalysisError(
            "unsupported_covariance",
            "Analytic saved AR inversion requires iid homoskedastic covariance.",
        )
    endogenous = options["endog"]
    if len(endogenous) != 1:
        raise AnalysisError(
            "unsupported_spec", "Analytic AR inversion requires one endogenous regressor."
        )
    options["endog"] = endogenous[0]
    output = iv_ar_confidence_set(**options, alpha=result.spec.alpha if alpha is None else alpha)
    output.attrs.update(
        source_result_id=result.id,
        original_sample_positions=positions,
        source_data_hash=result.provenance["data_hash"],
        iv_refitted=False,
    )
    return output
