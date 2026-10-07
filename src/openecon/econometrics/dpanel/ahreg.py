"""Anderson–Hsiao AR(1) panel IV, using the shared float64/QR 2SLS kernels.

The fixed effect is removed by first differences. The endogenous regressor
is D.L.y; its one excluded instrument is L2.y or D.L2.y. No dense panel/time
grid, observation projection matrix or differenced-error covariance matrix
is built. See ``docs/econometrics/ahreg.md`` for assumptions and conventions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, linear_covariance, make_spec, wald_test,
)
from openecon.econometrics.iv import kernels
from openecon.econometrics.iv.common import check_fit
from openecon.engines.covariance import sandwich
from openecon.engines.linalg import collinear_columns
from openecon.models import ModelSpec, ResultBundle


def _periods(frame: ModelFrame) -> Tensor:
    """Require exact integer periods: ranking dates would silently close gaps."""
    series = frame.series(frame.spec.time)
    if is_datetime64_any_dtype(series.dtype) or is_bool_dtype(series.dtype):
        raise AnalysisError("invalid_time", "ahreg needs integer period codes with a step of 1; "
                            "convert dates to explicit regular period codes first.")
    if is_integer_dtype(series.dtype):
        if int(series.min()) < -(2**63) or int(series.max()) >= 2**63:
            raise AnalysisError("invalid_time", "Period codes must fit in signed int64.")
    else:
        values = frame.numeric(frame.spec.time)
        if bool(((values < -(2**63)) | (values >= 2**63)).any()):
            raise AnalysisError("invalid_time", "Period codes must fit in signed int64.")
    return frame.time_index()


def _equation(frame: ModelFrame, instrument: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return source rows, D.y, D.x and [D.L.y, excluded instrument]."""
    period = _periods(frame)
    codes, _ = frame.codes(frame.spec.panel)
    lag = 2 if instrument == "levels" else 3
    if frame.n <= lag:
        raise AnalysisError("insufficient_observations", f"instrument='{instrument}' needs at "
                            f"least {lag + 1} consecutive periods within usable panels.")
    cur = torch.arange(lag, frame.n, dtype=torch.int64)
    keep = torch.ones(len(cur), dtype=torch.bool)
    for offset in range(1, lag + 1):
        prev = cur - offset
        keep &= (codes[cur] == codes[prev]) & (period[cur] - period[prev] == offset)
    cur = cur[keep]
    if not len(cur):
        raise AnalysisError("insufficient_observations", f"No complete consecutive {lag + 1}-period "
                            f"windows remain for instrument='{instrument}'.")
    y, x = frame.numeric(frame.spec.outcome), frame.matrix(frame.spec.predictors)
    z = y[cur - 2] if instrument == "levels" else y[cur - 2] - y[cur - 3]
    dynamic = torch.stack((y[cur - 1] - y[cur - 2], z), dim=1)
    return cur, y[cur] - y[cur - 1], x[cur] - x[cur - 1], dynamic


def _ma1_covariance(score_x: Tensor, bread: Tensor, rss: float,
                    codes: Tensor, period: Tensor, df: int) -> tuple[Tensor, dict[str, Any]]:
    """B Xhat' H Xhat B * RSS/[2(N-K)], with sparse within-panel H."""
    adjacent = (codes[1:] == codes[:-1]) & (period[1:] - period[:-1] == 1)
    cross = score_x[1:][adjacent].T @ score_x[:-1][adjacent]
    meat = 2 * (score_x.T @ score_x) - cross - cross.T
    variance = rss / (2 * df)
    covariance = kernel_call(sandwich, bread, meat * variance)
    return covariance, {
        "covariance": "nonrobust", "df_inference": df,
        "correction": "homoskedastic level errors: RSS/[2(N-K)] * B Xhat' H Xhat B",
        "small_sample_correction": None, "level_error_variance": variance,
        "error_covariance": "H: diagonal 2; -1 for consecutive equations in the same panel",
    }


def _first_stage(frame: ModelFrame, projection: kernels.Projections, z: Tensor,
                 term: str, instrument: str, groups: tuple[Tensor, int]) -> dict[str, Any]:
    """Excluded-instrument relevance, always with panel-clustered uncertainty."""
    rss, partial_rss = float(projection.first.ssr[0]), float(projection.partial.ssr[0])
    covariance, info = linear_covariance(
        frame, x=z, resid=projection.e[:, 0], bread=projection.first.xtx_inv,
        n=frame.n, k=z.shape[1], df_resid=frame.n - z.shape[1], kind="cluster",
        clusters=[groups], cluster_names=[frame.spec.panel],
    )
    test = wald_test(projection.first.beta[:, 0], covariance, [z.shape[1] - 1],
                     df_resid=info["df_inference"], label="Panel-clustered excluded-instrument F")
    return {
        "term": term, "excluded_instrument": instrument,
        "partial_r_squared": 1 - rss / partial_rss if partial_rss > 0 else None,
        "excluded_coefficient": float(projection.first.beta[-1, 0]),
        "f_statistic": test["statistic"], "df": test["df"], "df2": test.get("df2"),
        "p_value": test["p_value"], "covariance": "cluster", "cluster_column": frame.spec.panel,
        "note": "Relevance diagnostic; not an AR/CLR weak-instrument-robust inference test.",
    }


def fit_ahreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Registry entry point: one lagged outcome, one AH instrument, exogenous x."""
    frame = ModelFrame(spec, data)
    declared = registry.cluster_columns(spec)
    if declared and declared != [spec.panel]:
        raise AnalysisError("invalid_spec", "ahreg clusters only on its model panel column.")
    frame.sort_panel()
    instrument = frame.option("instrument")
    cur, dy, dx, dynamic = _equation(frame, instrument)
    dlag, excluded = dynamic[:, :1], dynamic[:, 1:]
    z = torch.cat((dx, excluded), dim=1)
    n, k = len(cur), z.shape[1]
    if n <= k:
        raise AnalysisError("insufficient_observations", "The differenced equation needs more "
                            "observations than regressors and instruments.")
    _, omitted = kernel_call(collinear_columns, z)
    if omitted:
        exog = [spec.predictors[i] for i in omitted if i < dx.shape[1]]
        if exog:
            raise AnalysisError("singular_design", "Differenced predictors are time-invariant or "
                                f"collinear: {', '.join(exog)}. Remove these predictors.")
        raise AnalysisError("underidentified", "The AH instrument has no variation independent "
                            "of the differenced exogenous predictors.")
    projection = kernel_call(kernels.project, dy, dx, dlag, excluded)
    estimate = kernel_call(kernels.k_class, projection)
    check_fit(estimate.rss, float(dy.square().sum()), n - k, float(dy.square().sum()))
    score_x = kernel_call(kernels.score_regressors, dx, projection)
    mask = torch.zeros(frame.n, dtype=torch.bool)
    mask[cur] = True
    excluded_rows = frame.n - n
    frame.restrict(mask, f"Excluded {excluded_rows} observation(s) without the required "
                   f"consecutive lag window for instrument='{instrument}'.")
    groups = frame.codes(spec.panel)
    if groups[1] < 2:
        raise AnalysisError("insufficient_clusters", "ahreg needs at least two panels with "
                            "usable differenced equations.")
    if spec.covariance == "nonrobust":
        covariance, info = _ma1_covariance(score_x, estimate.bread, estimate.rss,
                                           groups[0], _periods(frame), n - k)
    else:
        covariance, info = linear_covariance(
            frame, x=torch.cat((dx, dlag), dim=1), score_x=score_x, resid=estimate.resid,
            bread=estimate.bread, n=n, k=k, df_resid=n - k, kind="cluster",
            clusters=[groups], cluster_names=[spec.panel],
        )
    term = f"L1.{spec.outcome}"
    instrument_term = f"L2.{spec.outcome}" if instrument == "levels" else f"D.L2.{spec.outcome}"
    first_stage = _first_stage(frame, projection, z, term, instrument_term, groups)
    # Shared IV kernels report included exogenous coefficients first; dynamic
    # panel tables conventionally report the lagged dependent variable first.
    order = torch.tensor([k - 1, *range(k - 1)], dtype=torch.int64)
    beta, covariance = estimate.beta[order], covariance[order][:, order]
    terms = [term, *spec.predictors]
    tss = float(dy.square().sum())
    counts = torch.bincount(groups[0], minlength=groups[1])
    info.update({
        "covariance": spec.covariance, "effective_covariance": "panel_cluster" if
        spec.covariance != "nonrobust" else "homoskedastic_level_ma1",
        "residual_definition": "D.y minus rho * D.L.y minus D.x'b",
        "r_squared_definition": "uncentered, first-difference equation; can be negative",
        "error_variance": "RSS/[2(N-K)]" if spec.covariance == "nonrobust" else None,
    })
    return build_result(
        frame, terms=terms, params=beta, covariance=covariance, use_t=True,
        df_inference=info["df_inference"], df_resid=n - k, observed=dy,
        fitted=dy - estimate.resid, title="Anderson–Hsiao dynamic panel IV",
        solver="householder_qr_two_stage",
        solver_diagnostics={"condition_number": estimate.condition_number},
        inference=info,
        metrics={"r_squared": 1 - estimate.rss / tss, "rmse": (estimate.rss / (n - k)) ** 0.5,
                 "df_model": k, "df_resid": n - k, "n_groups": groups[1],
                 "n_instruments": k, "n_endogenous": 1, "n_obs_transformed": n,
                 "group_min": int(counts.min()), "group_max": int(counts.max()),
                 "group_avg": n / groups[1]},
        tests={"model": wald_test(beta, covariance, range(k), df_resid=info["df_inference"],
                                  label="Joint significance of the differenced equation"),
               "overidentification": {"statistic": None, "df": 0, "p_value": None,
                                      "note": "Exactly identified; instrument validity is not tested."}},
        extra={"instrument": instrument, "endogenous": [term],
               "instruments": [instrument_term, *(f"D.{name}" for name in spec.predictors)],
               "first_stage": [first_stage], "equations": "difference",
               "excluded_lag_window_rows": excluded_rows, "missing_rows": frame.dropped_missing},
        provenance={"transformation": "first differences", "equations": "difference",
                    "dynamic_order": 1, "instrument": instrument_term,
                    "predictor_exogeneity": "strictly exogenous with respect to all level errors",
                    "instrument_validity_assumption": "level errors serially uncorrelated; "
                    "lagged y uncorrelated with future errors; independent panels",
                    "lag_semantics": "integer periods, step 1; no cross-panel or across-gap lags",
                    "derivatives": "closed form (linear IV)", "dense_observation_matrices": False},
    )


def ahreg(*, data: Any, y: str, panel: str, time: str, x: Sequence[str] | None = None,
          instrument: str = "levels", covariance: str = "cluster", cluster: str | None = None,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Fit AR(1) Anderson–Hsiao IV; ``x`` must be strictly exogenous.

    ``instrument='levels'`` uses y(t−2); ``'differences'`` uses y(t−2)−y(t−3).
    ``cluster`` defaults to the model panel and cannot select another column.
    ``robust`` means the same panel-clustered CR1 covariance; ``nonrobust``
    assumes homoskedastic, serially uncorrelated level errors and includes the
    MA(1) covariance induced by differencing. Time uses integer period codes.
    """
    from openecon.analysis import fit

    effective_cluster = panel if covariance == "cluster" and cluster is None else cluster
    spec = make_spec("ahreg", outcome=y, predictors=column_list(x, "x"), panel=panel, time=time,
                     intercept=False, covariance=covariance, cluster=effective_cluster,
                     missing=missing, alpha=alpha, options={"instrument": instrument})
    return fit(spec, data=data)
