"""Diebold-Yilmaz generalized forecast-error variance connectedness."""

from __future__ import annotations

import torch
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, column_list, table
from openecon.econometrics.var.postestimation import _system
from openecon.models import ResultBundle


def _ma(a, horizon):
    p, k, _ = a.shape
    sequence = [torch.eye(k, dtype=torch.float64)]
    for h in range(1, horizon):
        sequence.append(
            sum(
                (sequence[h - lag] @ a[lag - 1] for lag in range(1, min(p, h) + 1)),
                torch.zeros((k, k), dtype=torch.float64),
            )
        )
    return torch.stack(sequence)


def connectedness(result, horizon=10):
    """Generalized FEVD with rows normalized to one; directional shares in percent."""
    if not isinstance(result, ResultBundle) or result.spec.estimator not in ("var", "svar"):
        raise AnalysisError("invalid_result", "connectedness needs a fitted VAR or SVAR result.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 500:
        raise AnalysisError("invalid_horizon", "horizon must be an integer from 1 to 500.")
    system = _system(result)
    sigma, names = system["sigma"], system["names"]
    phi = _ma(system["a"], horizon)
    numerator = (phi @ sigma).square().sum(0) / sigma.diagonal()[None, :]
    denominator = torch.einsum("hij,jk,hik->i", phi, sigma, phi)
    raw = numerator / denominator[:, None]
    normalized = raw / raw.sum(1, keepdim=True)
    k = len(names)
    own = normalized.diagonal()
    from_ = (normalized.sum(1) - own) * 100
    to = (normalized.sum(0) - own) * 100
    total = float(from_.mean())
    shares = table(
        [
            [names[i], names[j], float(raw[i, j]), float(normalized[i, j] * 100)]
            for i in range(k)
            for j in range(k)
        ],
        columns=["response", "shock", "raw_share", "normalized_percent"],
    )
    directional = table(
        [
            [name, float(own[i] * 100), float(from_[i]), float(to[i]), float(to[i] - from_[i])]
            for i, name in enumerate(names)
        ],
        columns=["variable", "own_percent", "from_percent", "to_percent", "net_percent"],
    )
    return TableSet(
        {"fevd": shares, "directional": directional},
        title="Diebold-Yilmaz connectedness",
        total_percent=total,
        horizon=horizon,
        normalization="each response row sums to 100 percent; own effects excluded from connectedness",
        window_nobs=result.nobs,
        variables=names,
        innovation_covariance="generalized; order invariant",
        result_id=result.id,
    )


def rolling_connectedness(*, data, y, time, window, horizon=10, lags=1):
    """Fixed-size consecutive integer windows, preserving their start/end periods."""
    from openecon.econometrics.var.estimators import var

    y = column_list(y, "y")
    if not y or isinstance(lags, bool) or not isinstance(lags, int) or not 1 <= lags <= 12:
        raise AnalysisError(
            "invalid_spec", "Use named endogenous variables and integer lags 1..12."
        )
    df = _coerce_frame(data)
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window <= lags + len(y) * lags + 1
        or window > len(df)
    ):
        raise AnalysisError(
            "invalid_window", "window must exceed the VAR design and lie within the input sample."
        )
    if time not in df or df[time].isna().any() or df[time].duplicated().any():
        raise AnalysisError("invalid_time", "Use a complete, unique integer time calendar.")
    if (
        is_datetime64_any_dtype(df[time].dtype)
        or not is_numeric_dtype(df[time].dtype)
        or not bool((df[time] % 1 == 0).all())
    ):
        raise AnalysisError(
            "invalid_time",
            "Use an explicit integer calendar; dates and fractional periods are not ranked.",
        )
    df = df.sort_values(time, kind="stable").reset_index(drop=True)
    if not bool((df[time].diff().dropna() == 1).all()):
        raise AnalysisError("time_gaps", "Rolling windows cannot compress missing periods.")
    rows = []
    for end in range(window, len(df) + 1):
        piece = df.iloc[end - window : end]
        result = var(
            data=piece,
            y=y,
            time=time,
            lags=lags,
            maxlag=lags,
            lm_lags=0,
            irf_steps=1,
            irf_kinds=["simple"],
        )
        value = connectedness(result, horizon)
        rows.append(
            [int(piece[time].iloc[0]), int(piece[time].iloc[-1]), value.attrs["total_percent"]]
        )
    return table(
        rows, columns=["start", "end", "total_percent"], window=window, horizon=horizon, lags=lags
    )
