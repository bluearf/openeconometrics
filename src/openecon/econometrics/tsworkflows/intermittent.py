"""Intermittent-demand forecasting: Croston, SBA, SBJ and TSB.

Methods (point forecasts only):

- Croston (1972): exponential smoothing of the nonzero demand sizes (``alpha``) and of
  the intervals between demands (``beta``), both updated only in demand periods; the
  demand-rate forecast is size / interval.
- SBA (Syntetos and Boylan 2005): Croston times (1 - beta / 2).
- SBJ (Shale, Boylan and Johnston 2006): Croston times (1 - beta / (2 - beta)).
- TSB (Teunter, Syntetos and Babai 2011): sizes smoothed in demand periods
  (``alpha``), the demand probability smoothed every period (``beta``) towards the
  occurrence indicator; forecast = probability x size.

All methods start at the first demand period f: size = y_f, interval = f (periods
counted from the start of the sample, as R ``forecast::croston_model``) and, for TSB,
probability = 1 / f. One-step-ahead fitted values exist for t > f. The h-step
forecasts are flat at the forecast after the last observation (using every demand,
including one in the last period). The recursions are exponentially weighted sums,
evaluated as batched FFT convolutions (no loop over observations).

A smoothing constant given as ``"estimate"`` is chosen by minimizing the in-sample
one-step loss (``"mse"`` mean squared or ``"mae"`` mean absolute error over t > f) on
a 0.02 grid in [0.02, 1] followed by coordinatewise golden-section refinement in the
neighbouring grid cells (three sweeps, tolerance 1e-7), within [0.01, 1]; an estimate
at a bound is flagged in ``attrs["at_bound"]``. The search is deterministic
but not a global-optimum guarantee; the grid minimum and the loss are reported.
``beta=None`` ties the interval / probability constant to ``alpha`` (R's
single-parameter convention), also when alpha is estimated.

No prediction intervals are produced: Croston-type methods have no consistent
stochastic model (Shenstone and Hyndman 2005), so none is implied.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arima.diagnostics import load_series
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.seasonal_common import (
    MAX_PERIODS, _integer, future_labels, sample_contract,
)
from openecon.econometrics.tsworkflows.ssdiffuse import _calendar
from openecon.resources import plan_workspace

METHODS = ("croston", "sba", "sbj", "tsb")
LOSSES = ("mse", "mae")
GRID = torch.arange(1, 51, dtype=torch.float64) * 0.02
LOWER = 0.01
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


def _smooth(values: Tensor, weights: Tensor) -> Tensor:
    """Exponential smoothing levels L_i = (1-a) L_(i-1) + a v_i with L_1 = v_1, rows of weights.

    values: [m]; weights: [G] smoothing constants. Returns [G, m].
    """
    m = values.shape[0]
    decay = (1.0 - weights)[:, None] ** torch.arange(m, dtype=torch.float64)[None, :]
    scaled = weights[:, None] * values[None, :]
    scaled[:, 0] = values[0]
    size = 1 << max(1, (2 * m - 2).bit_length())
    out = torch.fft.irfft(torch.fft.rfft(scaled, size) * torch.fft.rfft(decay, size), size)[:, :m]
    return out


def _probability(occurs: Tensor, start: float, weights: Tensor) -> Tensor:
    """p_0 = start, p_t = (1-b) p_(t-1) + b d_t for t = 1..m; returns [G, m + 1]."""
    values = torch.cat([torch.tensor([start], dtype=torch.float64), occurs])
    m = values.shape[0]
    decay = (1.0 - weights)[:, None] ** torch.arange(m, dtype=torch.float64)[None, :]
    scaled = weights[:, None] * values[None, :]
    scaled[:, 0] = start
    size = 1 << max(1, (2 * m - 2).bit_length())
    return torch.fft.irfft(torch.fft.rfft(scaled, size) * torch.fft.rfft(decay, size), size)[:, :m]


class _Model:
    def __init__(self, y: Tensor, method: str):
        self.y, self.method = y, method
        positions = torch.nonzero(y > 0).flatten() + 1                      # 1-based
        self.positions = positions
        self.first = int(positions[0])
        self.sizes = y[positions - 1]
        self.intervals = torch.diff(positions, prepend=torch.tensor([0])).to(torch.float64)
        n = y.shape[0]
        t = torch.arange(1, n + 1)
        # Number of demand events in periods 1..t-1 (information for the forecast of t).
        self.events_before = torch.searchsorted(positions, t, right=False)
        self.scored = t > self.first

    def paths(self, alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(fitted [G, n] with NaN for t <= f, size levels [G, k], second levels)."""
        size = _smooth(self.sizes, alpha)
        n = self.y.shape[0]
        index = (self.events_before - 1).clamp(min=0)
        if self.method == "tsb":
            occurs = (self.y[self.first:] > 0).to(torch.float64)
            prob = _probability(occurs, 1.0 / self.first, beta)            # p at t = f, f+1, .., n
            previous = torch.full((beta.shape[0], n), math.nan, dtype=torch.float64)
            previous[:, self.first:] = prob[:, :-1]
            fitted = previous * size[:, index]
            second = prob
        else:
            interval = _smooth(self.intervals, beta)
            factor = {"croston": torch.ones_like(beta), "sba": 1.0 - beta / 2.0,
                      "sbj": 1.0 - beta / (2.0 - beta)}[self.method]
            fitted = factor[:, None] * size[:, index] / interval[:, index]
            second = interval
        fitted = torch.where(self.scored[None, :], fitted, torch.full_like(fitted, math.nan))
        return fitted, size, second

    def loss(self, alpha: Tensor, beta: Tensor, kind: str) -> Tensor:
        fitted, _, _ = self.paths(alpha, beta)
        error = (self.y[None, :] - fitted)[:, self.scored]
        return error.square().mean(1) if kind == "mse" else error.abs().mean(1)


def _golden(function, low: float, high: float) -> float:
    a, b = low, high
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = function(c), function(d)
    while b - a > 1e-7:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = function(d)
    return (a + b) / 2.0


def _optimize(model: _Model, alpha: float | None, beta: float | None, tied: bool,
              kind: str) -> tuple[float, float, dict[str, Any]]:
    """Grid then golden-section minimization over the free constants (None = free)."""
    def one(a: float, b: float) -> float:
        return float(model.loss(torch.tensor([a], dtype=torch.float64),
                                torch.tensor([b], dtype=torch.float64), kind)[0])

    if tied:
        losses = model.loss(GRID, GRID, kind)
        best = int(torch.argmin(losses))
        a = GRID[best].item()
        a = _golden(lambda v: one(v, v), max(LOWER, a - 0.02), min(1.0, a + 0.02))
        return a, a, {"grid_minimum": float(losses[best]), "grid_step": 0.02}
    free_alpha, free_beta = alpha is None, beta is None
    a0 = GRID if free_alpha else torch.tensor([alpha], dtype=torch.float64)
    b0 = GRID if free_beta else torch.tensor([beta], dtype=torch.float64)
    surface = torch.empty((a0.shape[0], b0.shape[0]), dtype=torch.float64)
    for i in range(a0.shape[0]):                       # one grid row of size constants at a time
        surface[i] = model.loss(a0[i].repeat(b0.shape[0]), b0, kind)
    flat = int(torch.argmin(surface))
    a, b = a0[flat // b0.shape[0]].item(), b0[flat % b0.shape[0]].item()
    for _ in range(3):
        if free_alpha:
            a = _golden(lambda v: one(v, b), max(LOWER, a - 0.02), min(1.0, a + 0.02))
        if free_beta:
            b = _golden(lambda v: one(a, v), max(LOWER, b - 0.02), min(1.0, b + 0.02))
    return a, b, {"grid_minimum": float(surface.min()), "grid_step": 0.02}


def _constant(value: Any, name: str) -> float | None:
    if value == "estimate":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise AnalysisError("invalid_option", f"{name} must be a number in (0, 1] or 'estimate'.")
    return float(value)


@resident_cpu
def croston(data: Any, y: str, *, steps: int = 12, method: str = "croston", time: str | None = None,
            alpha: float | str = 0.1, beta: float | str | None = None,
            loss: str = "mse") -> TableSet:
    """Intermittent-demand forecasts: Croston, SBA, SBJ or TSB (point forecasts only).

    Parameters
    ----------
    data, y : table and a nonnegative series with at least two nonzero demands.
    steps : forecast horizon (flat forecasts), 1..10000.
    method : ``"croston"``, ``"sba"`` (Syntetos-Boylan approximation), ``"sbj"``
        (Shale-Boylan-Johnston) or ``"tsb"`` (Teunter-Syntetos-Babai).
    time : optional time column (consecutive integers or a regular datetime calendar).
    alpha : smoothing constant of the demand sizes in (0, 1], or ``"estimate"``.
    beta : smoothing constant of the intervals (Croston/SBA/SBJ; the SBA/SBJ bias
        factors use it) or of the demand probability (TSB), ``"estimate"``, or None to
        tie it to alpha (R ``forecast::croston`` uses one constant).
    loss : ``"mse"`` or ``"mae"``: the one-step in-sample loss over the periods after
        the first demand, minimized for estimated constants and always reported.

    Returns
    -------
    ``TableSet`` with ``fitted`` (period, observed, size_level, interval_level or
    probability_level after each period's update where defined, fitted one-step
    forecast; NaN before the first forecast) and ``forecast`` (period, forecast);
    attrs: method, alpha, beta, loss, the in-sample loss, ADI (mean inter-demand
    interval) and CV^2 of the sizes, initialization and notes.

    R: ``forecast::croston(y, alpha = 0.1, type = "sba")``.
    """
    if method not in METHODS:
        raise AnalysisError("invalid_option", f"method must be one of: {', '.join(METHODS)}.")
    if loss not in LOSSES:
        raise AnalysisError("invalid_option", "loss must be 'mse' or 'mae'.")
    steps = _integer(steps, "steps", 1, 10_000)
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "croston needs a resident table; no Dataset "
                            "collection is performed.")
    series, labels = load_series(data, y, time)
    n = int(series.shape[0])
    if n > MAX_PERIODS:
        raise AnalysisError("system_budget", f"croston admits at most {MAX_PERIODS} periods.")
    calendar = None
    index = pd.Index(labels.to_numpy() if time is not None else range(n))
    if time is not None:
        calendar = _calendar(pd.DatetimeIndex(index) if pd.api.types.is_datetime64_any_dtype(index.dtype)
                             else index)
        if calendar["kind"] == "datetime":
            index = pd.DatetimeIndex(index)
    if bool((series < 0).any()):
        raise AnalysisError("negative_values", "Intermittent-demand methods need nonnegative demand.")
    if int((series > 0).sum()) < 2:
        raise AnalysisError("insufficient_demands", "At least two nonzero demands are required.")
    tied = beta is None
    alpha = _constant(alpha, "alpha")
    beta = alpha if tied else _constant(beta, "beta")
    estimated = [name for name, value in (("alpha", alpha), ("beta", beta)) if value is None]
    if tied and alpha is None:
        estimated = ["alpha"]
    model = _Model(series, method)
    if not bool(model.scored.any()):
        raise AnalysisError("insufficient_observations", "No period follows the first demand.")
    plan_workspace("intermittent smoothing paths", {"grid_paths": n * GRID.shape[0] * 8 * 8})
    info: dict[str, Any] = {}
    if estimated:
        alpha, beta, info = _optimize(model, alpha, beta, tied, loss)
    a = torch.tensor([alpha], dtype=torch.float64)
    b = torch.tensor([beta], dtype=torch.float64)
    fitted, size, second = model.paths(a, b)
    fitted, size, second = fitted[0], size[0], second[0]
    k = size.shape[0]
    if method == "tsb":
        point = float(second[-1] * size[-1])
        level_name = "probability_level"
        level = torch.full((n,), math.nan, dtype=torch.float64)
        level[model.first - 1:] = second
    else:
        factor = {"croston": 1.0, "sba": 1.0 - beta / 2.0, "sbj": 1.0 - beta / (2.0 - beta)}[method]
        point = float(factor * size[-1] / second[-1])
        level_name = "interval_level"
        level = torch.full((n,), math.nan, dtype=torch.float64)
    sizes = torch.full((n,), math.nan, dtype=torch.float64)
    events_through = torch.searchsorted(model.positions, torch.arange(1, n + 1), right=True)
    defined = events_through > 0
    sizes[defined] = size[events_through[defined] - 1]
    if method != "tsb":
        level[defined] = second[events_through[defined] - 1]
    error = (series - fitted)[model.scored]
    in_sample = float(error.square().mean() if loss == "mse" else error.abs().mean())
    positive = series[series > 0]
    adi = float(model.intervals.mean())
    cv2 = float(positive.var(unbiased=False) / positive.mean().square()) if k > 1 else 0.0
    labels_out = list(index)
    future = future_labels(index, calendar, steps)
    return TableSet(
        {"fitted": table({"period": labels_out, "observed": series.tolist(),
                          "size_level": sizes.tolist(), level_name: level.tolist(),
                          "fitted": fitted.tolist()}),
         "forecast": table({"period": future, "forecast": [point] * steps})},
        title=f"{method.upper()} intermittent-demand forecasts of {y}",
        method=method, alpha=alpha, beta=beta, tied_smoothing=tied, estimated=estimated,
        at_bound=[name for name in estimated
                  if min(abs((alpha if name == "alpha" else beta) - edge) for edge in (LOWER, 1.0)) < 1e-6],
        loss=loss, in_sample_loss=in_sample,
        optimization=info, demands=k, nobs=n, first_demand_position=model.first,
        sample=sample_contract(n),
        inference="point forecasts only: no standard errors for the constants and no intervals",
        adi=adi, cv2=cv2, forecast=point, steps=steps,
        initialization="size = first demand; interval = its position (TSB probability = 1/position)",
        notes=["Point forecasts only; no prediction intervals are implied by Croston-type methods.",
               "The forecast uses every demand, including one in the last period (R forecast 9.x "
               "returns the previous ratio in that case)."],
    )
