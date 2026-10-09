"""Cross-sectional SAR 2SLS with explicit excluded instruments and fixed keyed W."""

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.core import (
    ModelFrame,
    TableSet,
    build_result,
    column_list,
    kernel_call,
    make_spec,
)
from openecon.engines.linalg import least_squares
from openecon.engines.distributions import normal_isf
from openecon.models import ResultBundle
from openecon.econometrics.resident_cpu import resident_cpu

from .weights import SpatialWeights, _keys
from .kernels import multiplier_impacts, stable_bound


def sar_iv(
    data,
    outcome,
    predictors,
    *,
    key,
    instruments,
    spatial_weights,
    covariance="nonrobust",
    intercept=True,
    missing="raise",
    alpha=0.05,
    max_n=512,
):
    """SAR 2SLS: instrument Wy with supplied excluded columns; iid or HC0 V.

    W and instruments are assumed fixed and exogenous. No automatic lagged-X
    instruments, spatial-error covariance, weak-ID guarantee or panel route.
    Reduced-form means and impacts condition on the effective keyed network.
    """
    spec = make_spec(
        "sar_iv",
        outcome=outcome,
        predictors=column_list(predictors, "predictors"),
        columns={"key": key, "instruments": column_list(instruments, "instruments")},
        options={
            "spatial_weights": SpatialWeights.from_payload(spatial_weights).to_payload(),
            "max_n": max_n,
        },
        covariance=covariance,
        intercept=intercept,
        missing=missing,
        alpha=alpha,
    )
    return fit_sar_iv(spec, data)


@resident_cpu
def fit_sar_iv(spec, data):
    frame = ModelFrame(spec, data)
    key = frame.role("key")[0]
    excluded = frame.role("instruments")
    if key in {spec.outcome, *spec.predictors, *excluded} or set(excluded) & {
        spec.outcome,
        *spec.predictors,
    }:
        raise AnalysisError(
            "invalid_spatial_keys",
            "Key, outcome, exogenous controls and excluded instruments must be distinct.",
        )
    weights = SpatialWeights.from_payload(frame.option("spatial_weights"))
    weights.align(_keys(frame.original[key].tolist()))
    weights = weights.align(_keys(frame.series(key).tolist()), subset=True)
    n, k, instrument_count = frame.n, frame.design_width() + 1, frame.design_width() + len(excluded)
    if (
        n > frame.option("max_n")
        or k > 32
        or instrument_count > 64
        or n <= instrument_count
        or instrument_count < k
    ):
        raise AnalysisError(
            "spatial_dense_limit", "SAR IV needs N<=max_n, K<=32, instruments<=64, N>L>=K."
        )
    frame.workspace_plan(
        "SAR IV projection, full covariance and dense multiplier",
        {
            "dense_weight_and_multiplier_derivatives": 8 * 32 * n**2,
            "design_projection_QR": 8 * 24 * n * (k + instrument_count),
            "full_covariance": 8 * 12 * k**2,
        },
    )
    w = weights.dense(max_n=frame.option("max_n"))
    design = frame.design()
    if any(t in {"rho", "lambda", "ln_sigma2"} or t.startswith("W:") for t in design.terms):
        raise AnalysisError(
            "duplicate_terms", "Spatial slope terms cannot use reserved spatial names."
        )
    y = frame.numeric(spec.outcome)
    x = torch.cat((design.x, (w @ y)[:, None]), 1)
    z = torch.cat((design.x, frame.matrix(excluded)), 1)
    # Rank-check through native QR before constructing a projection; never N by N Pz.
    zfit = kernel_call(least_squares, z, y, drop_collinear=False)
    del zfit
    q, _ = torch.linalg.qr(z, mode="reduced")
    projected = q @ (q.T @ x)
    fit = kernel_call(least_squares, q.T @ x, q.T @ y, drop_collinear=False)
    params, bread = fit.beta, fit.xtx_inv
    if abs(float(params[-1])) >= stable_bound(w):
        raise AnalysisError(
            "unstable_spatial_parameter",
            "Unconstrained SAR 2SLS rho lies outside the nonsingular stable domain; it is not clipped.",
        )
    residual = y - x @ params
    sigma2 = float(residual.square().sum()) / (n - k)
    if spec.covariance == "nonrobust":
        covariance = bread * sigma2
    else:
        scores = projected * residual[:, None]
        covariance = bread @ (scores.T @ scores) @ bread
    terms = [*design.terms, "rho"]
    mean = torch.linalg.solve(
        torch.eye(n, dtype=torch.float64) - params[-1] * w, design.x @ params[:-1]
    )
    impacts = kernel_call(
        multiplier_impacts,
        params,
        covariance,
        w,
        terms=terms,
        predictors=spec.predictors,
        model="sar",
        alpha=spec.alpha,
    )
    impacts["method"] = "full 2SLS covariance delta method"
    if frame.dropped_missing:
        frame.warn("Missing filtering induced a keyed subgraph and re-normalized row-normalized W.")
    return build_result(
        frame,
        terms=terms,
        params=params,
        covariance=covariance,
        fitted=mean,
        df_resid=n - k,
        use_t=spec.covariance == "nonrobust",
        df_inference=n - k,
        metrics={"sigma2": sigma2},
        solver="full-rank native QR 2SLS; explicit excluded instruments",
        inference={
            "correction": "SSE/(N-K) iid"
            if spec.covariance == "nonrobust"
            else "HC0 projected-design sandwich; normal reference",
            "fitted_definition": "unconditional reduced-form network mean",
            "weak_identification_validated": False,
        },
        provenance={
            "spatial_weight_hash": weights.summary()["sha256"],
            "spatial_execution_domain": "numeric unweighted cross-sectional CPU; fixed exogenous W and IV",
        },
        extra={
            "spatial_weights": weights.to_payload(),
            "spatial_summary": weights.summary(),
            "structural_residuals": residual.tolist(),
            "impacts": impacts,
            "model_equation": "y=rho Wy+X beta+epsilon",
            "instrument_terms": [*design.terms, *excluded],
        },
    )


@resident_cpu
def sar_iv_predict(result, *, data):
    """Restored closed-network conditional mean and full-parameter delta SE.

    Provide all original effective keys once, in any order, and numeric X.
    Returns a mean confidence interval, not a future-observation interval.
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "sar_iv":
        raise AnalysisError("invalid_result", "Pass a restored SAR IV ResultBundle.")
    from openecon.analysis import _coerce_frame
    from openecon.resources import plan_workspace
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported", "Saved network prediction requires a resident keyed design."
        )
    source = _coerce_frame(data)
    key = result.spec.columns["key"]
    required = [key, *result.spec.predictors]
    if (
        source.columns.has_duplicates
        or any(c not in source for c in required)
        or source[required].isna().any().any()
    ):
        raise AnalysisError(
            "prediction_domain",
            "Prediction requires all effective network keys and complete numeric predictors.",
        )
    if len(source) > result.spec.options["max_n"]:
        raise AnalysisError("spatial_dense_limit", "Saved network prediction exceeds max_n.")
    plan_workspace(
        "SAR IV prediction input admission",
        {
            "resident_network_buffers": 8 * 16 * len(source) ** 2,
            "projected_numeric_inputs": int(
                source[required].memory_usage(index=True, deep=True).sum()
            )
            * 2,
        },
    )
    inputs = source[required].copy()
    dummy = "__openecon_prediction_dummy__"
    while dummy in inputs:
        dummy += "_"
    inputs[dummy] = 0.0
    frame = ModelFrame(
        make_spec(
            "ols", outcome=dummy, predictors=result.spec.predictors, intercept=result.spec.intercept
        ),
        inputs,
        extra_columns=[key],
    )
    weights = SpatialWeights.from_payload(result.extra["spatial_weights"])
    if weights.summary()["sha256"] != result.provenance.get("spatial_weight_hash"):
        raise AnalysisError(
            "invalid_result", "Saved effective spatial weights differ from their recorded hash."
        )
    aligned = weights.align(_keys(frame.series(key).tolist()))
    if frame.n != len(weights.keys):
        raise AnalysisError(
            "prediction_domain", "Prediction requires the complete effective closed network."
        )
    frame.workspace_plan(
        "SAR IV saved network delta prediction", {"dense_multiplier": 8 * 12 * frame.n**2}
    )
    w = aligned.dense(max_n=result.spec.options["max_n"])
    beta = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
    v = torch.tensor(result.covariance_matrix, dtype=torch.float64)
    x = frame.design().x
    if [c.term for c in result.coefficients] != [*frame.design().terms, "rho"] or abs(
        float(beta[-1])
    ) >= stable_bound(w):
        raise AnalysisError(
            "invalid_result", "Saved SAR IV term order or stable parameter domain is invalid."
        )
    s = torch.linalg.solve(
        torch.eye(frame.n, dtype=torch.float64) - beta[-1] * w,
        torch.eye(frame.n, dtype=torch.float64),
    )
    mean = s @ x @ beta[:-1]
    gradient = torch.cat((s @ x, (s @ w @ mean)[:, None]), 1)
    variance = torch.sum((gradient @ v) * gradient, 1)
    if bool((variance < -1e-10).any()) or not bool(torch.isfinite(variance).all()):
        raise AnalysisError("invalid_covariance", "Saved prediction covariance is invalid.")
    se = variance.clamp_min(0).sqrt()
    critical = normal_isf(result.spec.alpha / 2)
    # Same normal delta reference as impacts, including when coefficient tests use t.
    return saved_summary(
        TableSet(
            {
                "network mean": pd.DataFrame(
                    {
                        "key": aligned.keys,
                        "mean": mean.tolist(),
                        "std_error": se.tolist(),
                        "ci_low": (mean - critical * se).tolist(),
                        "ci_high": (mean + critical * se).tolist(),
                    }
                )
            },
            source_result_id=result.id,
            refitted=False,
            reference="normal full covariance delta method",
            conditioning="fixed closed network and X; no innovation/prediction error",
        )
    )
