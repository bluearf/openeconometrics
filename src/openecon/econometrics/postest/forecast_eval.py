"""Forecast accuracy: EViews' forecast evaluation table and the Diebold-Mariano test.

Both procedures take series of actual values and forecasts over an evaluation
sample of h periods (columns of a DataFrame or plain sequences/tensors) and
work on float64 tensors. The forecast error is ``e_t = y_t - f_t`` (actual
minus forecast), so a positive mean error means the forecast is too low.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis import _coerce_frame
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.engines.distributions import normal_sf, t_sf

_METRICS = ("n", "mean_error", "mae", "mse", "rmse", "mape", "smape", "theil_u1", "theil_u2",
            "bias_proportion", "variance_proportion", "covariance_proportion", "mase")


def _series(value: Any, data: Any, role: str) -> Tensor:
    """A float64 vector from a column name (with ``data``), a Series, a tensor or a sequence."""
    if isinstance(value, str):
        if data is None:
            raise AnalysisError("invalid_data", f"{role} is a column name ({value!r}); pass "
                                "data= as well, or pass the values themselves.")
        frame = _coerce_frame(data)
        if value not in frame.columns:
            raise AnalysisError("missing_columns", f"Column {value!r} ({role}) is not in the "
                                "data.")
        value = frame[value]
    try:
        if isinstance(value, Tensor):
            vector = value.detach().to(torch.float64)
        elif isinstance(value, (pd.Series, pd.Index)):
            vector = torch.as_tensor(pd.to_numeric(pd.Series(value), errors="raise")
                                     .to_numpy(dtype="float64", na_value=math.nan).copy())
        elif hasattr(value, "__len__") and not isinstance(value, (bytes, dict, set, frozenset)):
            try:      # numeric arrays and lists convert in one step
                vector = torch.as_tensor(value, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError):
                vector = torch.tensor([math.nan if item is None else float(item)
                                       for item in value], dtype=torch.float64)
        else:
            raise TypeError
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_data", f"{role} must be a numeric column name, Series, "
                            "tensor or sequence of numbers.") from exc
    if sum(size > 1 for size in vector.shape) > 1:
        raise AnalysisError("invalid_data", f"{role} must be one-dimensional (one value per "
                            f"period); got values of shape {tuple(vector.shape)}.")
    return vector.reshape(-1).clone()


def _aligned(named: dict[str, Any], data: Any, missing: str) -> tuple[dict[str, Tensor], list[str]]:
    if missing not in {"raise", "drop"}:
        raise AnalysisError("invalid_spec", "missing must be 'raise' or 'drop'.")
    vectors = {role: _series(value, data, role) for role, value in named.items()}
    lengths = {len(v) for v in vectors.values()}
    if len(lengths) != 1:
        raise AnalysisError("length_mismatch", "The actual values and the forecasts must have the "
                            "same length (one value per evaluation period).")
    finite = torch.stack([torch.isfinite(v) for v in vectors.values()]).all(dim=0)
    notes = []
    if not bool(finite.all()):
        count = int((~finite).sum())
        if missing == "raise":
            raise AnalysisError("missing_values", f"{count} period(s) have a missing or "
                                "non-finite actual value or forecast. Restrict the series to the "
                                "evaluation sample, or choose missing='drop' explicitly.")
        vectors = {role: v[finite] for role, v in vectors.items()}
        notes.append(f"Excluded {count} period(s) with missing values; changes and lags are "
                     "taken between the remaining periods.")
    return vectors, notes


def _autocovariances(centered: Tensor, lags: int) -> Tensor:
    """``sum_(t>k) c_t c_(t-k)`` for k = 0..lags, in O(T lags) or, for many lags, by FFT."""
    n = len(centered)
    if lags <= 64:
        return torch.stack([centered.square().sum()]
                           + [(centered[k:] * centered[:-k]).sum() for k in range(1, lags + 1)])
    size = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=size)
    return torch.fft.irfft(spectrum * spectrum.conj(), n=size)[:lags + 1]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def fcast_eval(actual: Any, forecast: Any, *, data: Any = None, naive: Any = None,
               seasonal_period: int | None = None, missing: str = "raise"):
    """Forecast evaluation statistics (EViews' forecast evaluation output).

    With actual values ``y_t``, forecasts ``f_t`` and errors ``e_t = y_t - f_t``
    over the h evaluation periods:

    - ``mean_error`` = (1/h) sum e_t; ``mae`` = (1/h) sum |e_t|;
      ``mse`` = (1/h) sum e_t^2; ``rmse`` = sqrt(mse);
    - ``mape`` = 100 (1/h) sum |e_t / y_t| (``None`` with a note when an actual
      value is zero);
    - ``smape`` (EViews' symmetric MAPE) = 100 (1/h) sum |f_t - y_t| /
      ((|f_t| + |y_t|) / 2) (``None`` when both are zero in some period);
    - ``theil_u1`` = rmse / (sqrt((1/h) sum f_t^2) + sqrt((1/h) sum y_t^2)),
      between 0 (perfect) and 1;
    - ``theil_u2`` (EViews) = sqrt(sum_t ((f_(t+1) - y_(t+1)) / y_t)^2 /
      sum_t ((y_(t+1) - y_t) / y_t)^2), t = 1..h-1: below 1 the forecast beats
      the no-change forecast;
    - the MSE decomposition (EViews; standard deviations with divisor h, so
      the three proportions sum to one): ``bias_proportion`` =
      (mean f - mean y)^2 / mse, ``variance_proportion`` = (s_f - s_y)^2 / mse,
      ``covariance_proportion`` = 2 (1 - r) s_f s_y / mse;
    - ``mase`` = mae / scale (Hyndman and Koehler 2006). The scale is the MAE of
      the benchmark forecast ``naive`` when given, otherwise the mean absolute
      m-period change of the actual series, (1/(h-m)) sum_(t>m) |y_t - y_(t-m)|
      with m = ``seasonal_period`` (default 1: the random-walk forecast).
      Values below 1 beat the benchmark.

    Parameters
    ----------
    actual, forecast : column names (with ``data``), Series, tensors or sequences
        The evaluation-sample values, aligned period by period.
    data : DataFrame, optional
        The table holding the named columns.
    naive : column name, Series or sequence, optional
        A benchmark forecast for the MASE scale.
    seasonal_period : int, optional
        m of the seasonal naive scale (ignored when ``naive`` is given).
    missing : {'raise', 'drop'}
        Missing values raise ``missing_values`` unless dropped explicitly.

    Returns
    -------
    A table indexed by statistic with one column ``value``; undefined
    statistics are empty and explained in ``attrs['notes']``. ``attrs`` also
    hold ``n``, ``error_definition``, ``seasonal_period`` and ``mase_scale``.

    EViews: the forecast evaluation table of ``Forecast`` (``fcast`` with
    evaluation statistics). Stata has no built-in equivalent.

    Example::

        oe.fcast_eval("gdp", "gdp_f", data=holdout)
        oe.fcast_eval(y_test, model_forecast, naive=last_value_forecast)
    """
    named = {"actual": actual, "forecast": forecast}
    if naive is not None:
        named["naive"] = naive
    vectors, notes = _aligned(named, data, missing)
    y, f = vectors["actual"], vectors["forecast"]
    h = len(y)
    if h < 2:
        raise AnalysisError("insufficient_observations", "Forecast evaluation needs at least two "
                            "periods.")
    if seasonal_period is not None and (isinstance(seasonal_period, bool)
                                        or not isinstance(seasonal_period, int)
                                        or seasonal_period < 1):
        raise AnalysisError("invalid_spec", "seasonal_period must be a positive integer.")
    m = seasonal_period or 1
    error = y - f
    mse = float(error.square().mean())
    mae = float(error.abs().mean())
    values: dict[str, float | None] = {"n": float(h), "mean_error": float(error.mean()),
                                       "mae": mae, "mse": mse, "rmse": math.sqrt(mse)}
    if bool((y == 0).any()):
        values["mape"] = None
        notes.append("MAPE is undefined: some actual values are zero.")
    else:
        values["mape"] = 100 * float((error / y).abs().mean())
    scale = (f.abs() + y.abs()) / 2
    if bool((scale == 0).any()):
        values["smape"] = None
        notes.append("Symmetric MAPE is undefined: actual and forecast are both zero in some "
                     "period.")
    else:
        values["smape"] = 100 * float(((f - y).abs() / scale).mean())
    values["theil_u1"] = _ratio(math.sqrt(mse), math.sqrt(float(f.square().mean()))
                                + math.sqrt(float(y.square().mean())))
    base = y[:-1]
    if bool((base == 0).any()):
        values["theil_u2"] = None
        notes.append("Theil's U2 is undefined: an actual value used as a base is zero.")
    else:
        u2_num = float((((f[1:] - y[1:]) / base).square()).sum())
        u2_den = float((((y[1:] - base) / base).square()).sum())
        values["theil_u2"] = math.sqrt(u2_num / u2_den) if u2_den > 0 else None
        if u2_den <= 0:
            notes.append("Theil's U2 is undefined: the actual series does not change.")
    mean_f, mean_y = float(f.mean()), float(y.mean())
    s_f = math.sqrt(float((f - mean_f).square().mean()))
    s_y = math.sqrt(float((y - mean_y).square().mean()))
    cross = float(((f - mean_f) * (y - mean_y)).mean())
    if mse > 0:
        values["bias_proportion"] = (mean_f - mean_y) ** 2 / mse
        values["variance_proportion"] = (s_f - s_y) ** 2 / mse
        values["covariance_proportion"] = 2 * (s_f * s_y - cross) / mse
    else:
        values.update(bias_proportion=None, variance_proportion=None, covariance_proportion=None)
        notes.append("The forecast is exact; the MSE decomposition is undefined.")
    if "naive" in vectors:
        mase_scale: float | None = float((y - vectors["naive"]).abs().mean())
        scale_source = "MAE of the benchmark forecast"
    elif h > m:
        mase_scale = float((y[m:] - y[:-m]).abs().mean())
        scale_source = f"mean absolute {m}-period change of the actual series"
    else:
        mase_scale, scale_source = None, "unavailable"
        notes.append(f"MASE needs more than seasonal_period={m} periods.")
    values["mase"] = _ratio(mae, mase_scale) if mase_scale is not None else None
    if mase_scale is not None and mase_scale <= 0:
        notes.append("MASE is undefined: the scale (benchmark error) is zero.")
    return table({"value": [values[name] for name in _METRICS]}, index=list(_METRICS),
                 title="Forecast evaluation", n=h, error_definition="actual - forecast",
                 seasonal_period=m, mase_scale=mase_scale, mase_scale_source=scale_source,
                 notes=notes)


def dm_test(actual: Any, forecast1: Any, forecast2: Any, *, data: Any = None, horizon: int = 1,
            loss: str = "squared", harvey: bool = True, kernel: str = "bartlett",
            missing: str = "raise"):
    """Diebold-Mariano test of equal forecast accuracy, with the HLN correction.

    With errors ``e_it = y_t - f_it`` and loss ``L`` (``squared``: e^2,
    ``absolute``: |e|), the loss differential is ``d_t = L(e_1t) - L(e_2t)``.
    Under H0: E[d_t] = 0 (equal expected loss),

        DM = d_bar / sqrt(LRV / T),
        LRV = gamma_0 + 2 sum_(k=1..h-1) w_k gamma_k,
        gamma_k = (1/T) sum_(t>k) (d_t - d_bar)(d_(t-k) - d_bar),

    where h is the forecast ``horizon`` (h-step forecast errors are MA(h-1),
    so h - 1 autocovariances enter). ``kernel='bartlett'`` uses the
    Newey-West weights w_k = 1 - k/h (always nonnegative); ``'uniform'`` the
    truncated (rectangular) weights w_k = 1 of Diebold and Mariano (1995),
    which can give a negative variance (then ``nonpositive_variance`` is
    raised). With ``harvey=True`` (default) the Harvey-Leybourne-Newbold
    (1997) small-sample statistic

        DM* = DM sqrt((T + 1 - 2h + h(h-1)/T) / T)

    is compared with Student t(T-1); otherwise DM with the standard normal.
    The p-value is two-sided. A positive statistic means forecast 1 has the
    larger loss (forecast 2 is more accurate).

    Parameters
    ----------
    actual, forecast1, forecast2 : column names (with ``data``), Series, tensors
        or sequences, aligned over the T evaluation periods.
    horizon : int
        Forecast horizon h >= 1 (T must exceed h).
    loss : {'squared', 'absolute'}
    harvey : bool
        Apply the HLN correction and the t(T-1) reference.
    kernel : {'bartlett', 'uniform'}
        Weights of the long-run variance.
    missing : {'raise', 'drop'}

    Returns
    -------
    A one-row table with ``statistic``, ``p_value``, ``df`` (T-1, or empty for
    the normal reference), ``mean_loss_differential``, ``long_run_variance``
    and ``n``; ``attrs`` hold ``distribution``, ``loss``, ``horizon``,
    ``harvey``, ``kernel``, ``dm_uncorrected`` and ``notes``.

    Stata: ``dmariano y f1 f2, maxlag(h-1) kernel(bartlett)`` (SSC); R:
    ``forecast::dm.test``. EViews has no built-in DM test.

    Example::

        oe.dm_test("y", "f_ar", "f_rw", data=holdout, horizon=4)
    """
    if loss not in {"squared", "absolute"}:
        raise AnalysisError("invalid_spec", "loss must be 'squared' or 'absolute'.")
    if kernel not in {"bartlett", "uniform"}:
        raise AnalysisError("invalid_spec", "kernel must be 'bartlett' or 'uniform'.")
    if not isinstance(harvey, bool):
        raise AnalysisError("invalid_spec", "harvey must be True or False.")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise AnalysisError("invalid_spec", "horizon must be a positive integer.")
    vectors, notes = _aligned({"actual": actual, "forecast1": forecast1,
                               "forecast2": forecast2}, data, missing)
    y = vectors["actual"]
    e1, e2 = y - vectors["forecast1"], y - vectors["forecast2"]
    d = e1.square() - e2.square() if loss == "squared" else e1.abs() - e2.abs()
    n, h = len(d), horizon
    if n <= h or n < 3:
        raise AnalysisError("insufficient_observations", f"The DM test needs more evaluation "
                            f"periods ({n}) than the horizon ({h}) and at least three.")
    mean = float(d.mean())
    centered = d - mean
    # A differential that varies only by rounding (relative to the losses) is constant.
    losses = e1.square() + e2.square() if loss == "squared" else e1.abs() + e2.abs()
    if float(centered.abs().max()) <= 64 * torch.finfo(torch.float64).eps * float(losses.max()):
        raise AnalysisError("constant_loss_differential", "The loss differential is constant, so "
                            "its variance is zero and the test is undefined.")
    gamma = _autocovariances(centered, h - 1) / n
    lag = torch.arange(1, h, dtype=torch.float64)
    weights = 1 - lag / h if kernel == "bartlett" else torch.ones(h - 1, dtype=torch.float64)
    variance = float(gamma[0] + 2 * (weights * gamma[1:]).sum())
    if not variance > 0:
        raise AnalysisError("nonpositive_variance", "The truncated (uniform) long-run variance is "
                            "not positive; use kernel='bartlett'.")
    statistic = mean / math.sqrt(variance / n)
    uncorrected = statistic
    if harvey:
        statistic *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p_value, df, distribution = 2 * t_sf(abs(statistic), n - 1), n - 1, "t"
    else:
        p_value, df, distribution = 2 * normal_sf(abs(statistic)), None, "normal"
    notes.append("H0: equal expected loss. A positive statistic means forecast 1 has the larger "
                 "loss.")
    return table([[statistic, min(1.0, p_value), df, mean, variance, n]],
                 columns=["statistic", "p_value", "df", "mean_loss_differential",
                          "long_run_variance", "n"],
                 index=["Diebold-Mariano"], distribution=distribution, loss=loss, horizon=h,
                 harvey=harvey, kernel=kernel, dm_uncorrected=uncorrected,
                 label="Diebold-Mariano test" + (" (HLN corrected)" if harvey else ""),
                 notes=notes)
