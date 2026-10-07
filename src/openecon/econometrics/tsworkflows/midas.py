"""Calendar-bound, release-aware exponential Almon MIDAS regression.

The normalized weights are softmax(theta_1*j + theta_2*j**2), with j scaled
to [0,1] for conditioning; lag zero is the latest available high-frequency
observation at the row's explicit forecast origin. No future release is used.
"""

from __future__ import annotations
import hashlib
from bisect import bisect_right
import pandas as pd
import torch
from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, make_spec, build_result, column_list, table
from openecon.econometrics.tsworkflows.common import finite_tensor, ml_optimize
from openecon.engines.optimize import information_inverse
from openecon.engines.contracts import KernelError
from openecon.resources import plan_workspace


def _hash(data, columns):
    values = pd.util.hash_pandas_object(data.loc[:, columns], index=False).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def midas_align(
    *,
    low,
    high,
    low_time,
    high_time,
    value,
    lags,
    frequency,
    origin=None,
    release_time=None,
    prefix="midas",
    missing="raise",
):
    """Return low-frequency rows with strictly dated high-frequency lag columns.

    ``origin`` optionally names a low-table timestamp column distinct from the
    target period. ``release_time`` prevents use before publication. Rows lacking
    an exact regular-calendar lag are raised or explicitly dropped, never shifted
    over missing periods. Frequency uses pandas calendar offsets (e.g. MS, D).
    """
    if isinstance(lags, bool) or not isinstance(lags, int) or not 3 <= lags <= 256:
        raise AnalysisError("invalid_option", "Exponential Almon identification needs 3..256 lags.")
    if missing not in {"raise", "drop"} or not isinstance(prefix, str) or not prefix:
        raise AnalysisError("invalid_option", "Use missing raise/drop and a nonempty prefix.")
    low, high = _coerce_frame(low), _coerce_frame(high)
    if low.columns.has_duplicates or high.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "MIDAS inputs need unique columns.")
    origin = origin or low_time
    if (
        any(c not in low for c in (low_time, origin))
        or any(c not in high for c in (high_time, value))
        or (release_time and release_time not in high)
    ):
        raise AnalysisError(
            "missing_columns", "Time, origin, release and value columns must exist."
        )
    plan_workspace(
        "MIDAS calendar alignment", {"lags": len(low) * lags * 32, "high_calendar": len(high) * 96}
    )
    names = [f"{prefix}_L{i}" for i in range(lags)]
    if set(names) & set(low.columns):
        raise AnalysisError(
            "duplicate_columns", "Generated lag columns already exist; choose another prefix."
        )
    try:
        dates = pd.to_datetime(high[high_time], errors="raise")
        releases = pd.to_datetime(high[release_time], errors="raise") if release_time else dates
        origins = pd.to_datetime(low[origin], errors="raise")
        targets = pd.to_datetime(low[low_time], errors="raise")
        offset = pd.tseries.frequencies.to_offset(frequency)
    except (ValueError, TypeError) as exc:
        raise AnalysisError(
            "invalid_calendar",
            "Frequency and datetime columns must be valid and share a time zone.",
        ) from exc
    if offset.n <= 0:
        raise AnalysisError("invalid_calendar", "High-frequency calendar offset must be positive.")
    try:
        zones = {str(pd.DatetimeIndex(series).tz) for series in (dates, releases, origins)}
    except (ValueError, TypeError) as exc:
        raise AnalysisError(
            "invalid_calendar", "Calendar columns must have consistent time zones."
        ) from exc
    if len(zones) > 1:
        raise AnalysisError(
            "invalid_calendar",
            "Observation, release and origin dates must share the same time zone.",
        )
    if (
        dates.isna().any()
        or releases.isna().any()
        or origins.isna().any()
        or targets.isna().any()
        or dates.duplicated().any()
        or targets.duplicated().any()
    ):
        raise AnalysisError(
            "invalid_calendar",
            "Calendar timestamps must be complete and unique per observation period.",
        )
    numeric = pd.to_numeric(high[value], errors="coerce")
    if numeric.isna().any() or not bool(
        torch.isfinite(torch.as_tensor(numeric.to_numpy(dtype="float64"))).all()
    ):
        raise AnalysisError("non_finite_data", "High-frequency values must be complete and finite.")
    index = pd.DatetimeIndex(dates)
    if not all(offset.is_on_offset(t) for t in index):
        raise AnalysisError(
            "invalid_calendar", "High-frequency timestamps must lie on the declared calendar grid."
        )
    mapping = {t: (float(v), r) for t, v, r in zip(dates, numeric, releases, strict=True)}
    ordered_dates = sorted(mapping)
    ordered_releases = [mapping[t][1] for t in ordered_dates]
    if any(r < t for t, r in zip(ordered_dates, ordered_releases, strict=True)) or any(
        later < earlier
        for earlier, later in zip(ordered_releases[:-1], ordered_releases[1:], strict=True)
    ):
        raise AnalysisError(
            "invalid_calendar",
            "Release dates must be at/after observations and monotone in observation time; align vintages separately.",
        )
    records, positions, dropped, lag_values = [], [], [], []
    for position, (target, asof) in enumerate(zip(targets, origins, strict=True)):
        latest = min(bisect_right(ordered_dates, asof), bisect_right(ordered_releases, asof)) - 1
        calendar = [] if latest < 0 else [ordered_dates[latest] - j * offset for j in range(lags)]
        if not calendar or any(
            t > asof or t not in mapping or mapping[t][1] > asof for t in calendar
        ):
            if missing == "raise":
                raise AnalysisError(
                    "unavailable_lags",
                    f"Low row{position} lacks a complete released lag calendar at {asof}.",
                )
            dropped.append(position)
            continue
        lag_values.append([mapping[t][0] for t in calendar])
        positions.append(position)
        records.append(
            {
                "target": target.isoformat(),
                "origin": asof.isoformat(),
                "observation_times": [t.isoformat() for t in calendar],
                "release_times": [mapping[t][1].isoformat() for t in calendar],
            }
        )
    if not positions:
        raise AnalysisError("empty_sample", "No complete MIDAS calendar remains.")
    out = table(low.iloc[positions].reset_index(drop=True))
    for j, name in enumerate(names):
        out[name] = [row[j] for row in lag_values]
    binding_columns = list(dict.fromkeys([low_time, origin, *names]))
    out.attrs["midas_alignment"] = {
        "schema": 1,
        "frequency": offset.freqstr,
        "low_time": low_time,
        "origin_column": origin,
        "lags": names,
        "high_value": value,
        "rows": records,
        "original_positions": positions,
        "dropped_positions": dropped,
        "binding_columns": binding_columns,
        "data_hash": _hash(out, binding_columns),
        "lookahead": "observation and release timestamps <= explicit row origin",
    }
    return out


def _alignment(record, data, lag_names):
    plan_workspace(
        "MIDAS bound alignment validation",
        {"hash_and_date_checks": len(data) * (len(lag_names) + 4) * 32},
    )
    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or record.get("lags") != lag_names
        or len(record.get("rows", [])) != len(data)
    ):
        raise AnalysisError(
            "invalid_alignment", "Use midas_align and preserve its matching alignment metadata."
        )
    columns = record.get("binding_columns", [])
    if (
        not columns
        or any(c not in data for c in columns)
        or record.get("data_hash") != _hash(data, columns)
    ):
        raise AnalysisError(
            "stale_alignment", "MIDAS lag/calendar data changed after alignment; rerun midas_align."
        )
    low_time, origin_column = record.get("low_time"), record.get("origin_column")
    if low_time not in data or origin_column not in data:
        raise AnalysisError(
            "invalid_alignment", "MIDAS target/origin columns must match the bound table."
        )
    for position, row in enumerate(record["rows"]):
        origin = pd.Timestamp(row["origin"])
        if origin != pd.Timestamp(data[origin_column].iloc[position]) or pd.Timestamp(
            row["target"]
        ) != pd.Timestamp(data[low_time].iloc[position]):
            raise AnalysisError(
                "invalid_alignment",
                "MIDAS metadata target/origin differs from its bound table row.",
            )
        if (
            len(row["observation_times"]) != len(lag_names)
            or len(row["release_times"]) != len(lag_names)
            or any(
                pd.Timestamp(t) > origin for t in [*row["observation_times"], *row["release_times"]]
            )
        ):
            raise AnalysisError(
                "lookahead", "MIDAS alignment contains unreleased/future observations."
            )


def _weights(z, count):
    j = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    return torch.softmax(z[0] * j + z[1] * j.square(), dim=0)


def fit_midas(spec, data):
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError(
            "streaming_unsupported",
            "MIDAS requires an explicit resident aligned low-frequency table; no Dataset collection is performed.",
        )
    data = _coerce_frame(data)
    from openecon.econometrics.registry import role_columns

    names = role_columns(spec, "lags")
    if (
        not 3 <= len(names) <= 256
        or len(names) != len(set(names))
        or set(names) & set(spec.predictors)
    ):
        raise AnalysisError(
            "invalid_lags", "MIDAS needs 3..256 unique lag columns distinct from predictors."
        )
    alignment = spec.options["alignment"]
    _alignment(alignment, data, names)
    frame = ModelFrame(spec, data)
    frame.workspace_plan(
        "MIDAS nonlinear least squares derivatives",
        {
            "lag_design": frame.n * len(names) * 8,
            "jacobian": frame.n * (len(spec.predictors) + 5) * 64,
        },
    )
    if frame.n * len(names) > 20_000_000:
        raise AnalysisError(
            "model_budget", "MIDAS lag-design operation geometry exceeds its explicit budget."
        )
    y, X, lag = (
        frame.numeric(spec.outcome),
        frame.design(),
        torch.stack([frame.numeric(name) for name in names], dim=1),
    )
    if int(torch.linalg.matrix_rank(X.x)) != X.x.shape[1]:
        raise AnalysisError("rank_deficient", "MIDAS linear controls must have full column rank.")
    k = X.x.shape[1] + 3
    if frame.n <= k + 2:
        raise AnalysisError("insufficient_observations", "MIDAS needs N>K+2 observations.")
    startdesign = torch.cat([X.x, lag.mean(dim=1, keepdim=True)], dim=1)
    if int(torch.linalg.matrix_rank(startdesign)) != startdesign.shape[1]:
        raise AnalysisError("rank_deficient", "MIDAS aggregate and controls are not identified.")
    beta = torch.linalg.lstsq(startdesign, y).solution
    start = torch.cat([beta, torch.zeros(2, dtype=torch.float64)])

    def fitted(z):
        return X.x @ z[: X.x.shape[1]] + z[X.x.shape[1]] * (lag @ _weights(z[-2:], len(names)))

    def objective(z):
        return -0.5 * (y - fitted(z)).square().sum()

    solution, _ = ml_optimize(
        objective, start, frame.option("max_iterations"), frame.option("tolerance")
    )
    theta = solution.theta
    predictions = fitted(theta)
    residuals = y - predictions
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(fitted, theta)
    # Observed Hessian bread includes nonlinear residual curvature. Gaussian
    # NLS sandwich uses per-row mean scores; sigma is concentrated out.
    try:
        bread = information_inverse(-solution.hessian)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    sigma2 = residuals.square().sum() / (frame.n - k)
    if not float(sigma2) > 0:
        raise AnalysisError("perfect_fit", "MIDAS residual variance is zero.")
    if spec.covariance == "nonrobust":
        cov = sigma2 * torch.linalg.inv(J.T @ J)
        correction = "N-K residual variance; expected Gaussian nonlinear least-squares information"
    else:
        scores = J * residuals[:, None]
        cov = bread @ (scores.T @ scores) @ bread
        correction = "HC0" if spec.covariance == "HC0" else "N/(N-K)"
        if spec.covariance == "HC1":
            cov *= frame.n / (frame.n - k)
    terms = [*X.terms, "midas_amplitude", "almon_linear", "almon_quadratic"]
    return build_result(
        frame,
        terms=terms,
        params=theta,
        covariance=cov,
        use_t=True,
        df_inference=frame.n - k,
        df_resid=frame.n - k,
        fitted=predictions,
        title="Exponential-Almon MIDAS",
        metrics={"sigma": sigma2.sqrt(), "ssr": residuals.square().sum()},
        solver="native nonlinear least squares, analytic Torch derivatives",
        optimizer={"converged": solution.converged, "iterations": solution.iterations},
        inference={"correction": correction, "parameter_order": terms},
        extra={
            "weights": _weights(theta[-2:], len(names)),
            "alignment": alignment,
            "linear_terms": X.terms,
            "lag_columns": names,
            "params": theta,
            "sample_alignment_positions": frame.positions,
        },
        provenance={
            "weight_function": "normalized exponential Almon, j in[0,1]",
            "forecast_origin_contract": alignment["lookahead"],
        },
    )


def midas(
    *,
    data,
    y,
    x=None,
    lags=None,
    alignment=None,
    time=None,
    intercept=True,
    covariance="nonrobust",
    missing="raise",
    alpha=0.05,
    max_iterations=300,
    tolerance=1e-7,
):
    alignment = alignment or getattr(data, "attrs", {}).get("midas_alignment")
    if lags is None and isinstance(alignment, dict):
        lags = alignment.get("lags")
    spec = make_spec(
        "midas",
        outcome=y,
        predictors=column_list(x, "x"),
        intercept=intercept,
        time=time,
        covariance=covariance,
        alpha=alpha,
        missing=missing,
        columns={"lags": column_list(lags, "lags")},
        options={"alignment": alignment, "max_iterations": max_iterations, "tolerance": tolerance},
    )
    return fit_midas(spec, data)


def midas_predict(result, *, data, alignment=None, alpha=None, kind="mean"):
    """Predict from saved MIDAS parameters with release-date validation and delta intervals.

    ``kind='mean'`` includes parameter uncertainty. ``kind='outcome'`` adds
    independent Gaussian outcome noise; intervals use the normal approximation.
    """
    if result.spec.estimator != "midas":
        raise AnalysisError("invalid_result", "midas_predict needs a MIDAS result.")
    if kind not in {"mean", "outcome"}:
        raise AnalysisError("invalid_prediction", "MIDAS prediction kind must be mean or outcome.")
    alignment = alignment or getattr(data, "attrs", {}).get("midas_alignment")
    frame = _coerce_frame(data)
    names = result.extra["lag_columns"]
    _alignment(alignment, frame, names)
    plan_workspace(
        "MIDAS predictions",
        {
            "lag_design": len(frame) * len(names) * 8,
            "Jacobian": len(frame) * len(result.coefficients) * 64,
        },
    )
    terms = result.extra["linear_terms"]
    X = (
        torch.stack(
            [
                torch.ones(len(frame), dtype=torch.float64)
                if name == "Intercept"
                else finite_tensor(frame[name].to_numpy(dtype="float64"))
                for name in terms
            ],
            dim=1,
        )
        if terms
        else torch.empty((len(frame), 0), dtype=torch.float64)
    )
    L = torch.stack([finite_tensor(frame[name].to_numpy(dtype="float64")) for name in names], dim=1)
    params = finite_tensor(result.extra["params"])

    def mean(z):
        return X @ z[: len(terms)] + z[len(terms)] * (L @ _weights(z[-2:], len(names)))

    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(mean, params)
    cov = finite_tensor(result.covariance_matrix)
    variance = (J @ cov * J).sum(dim=1)
    if kind == "outcome":
        variance += float(result.metrics["sigma"]) ** 2
    from openecon.econometrics.tsworkflows.common import forecast_table

    return forecast_table(
        mean(params),
        variance,
        result.spec.alpha if alpha is None else alpha,
        target="conditional mean" if kind == "mean" else "new outcome",
        uncertainty="delta-method parameter uncertainty only; new-outcome noise excluded; asymptotic normal intervals"
        if kind == "mean"
        else "delta-method parameter uncertainty plus independent Gaussian new-outcome noise; asymptotic normal intervals",
        alignment=alignment,
    )
