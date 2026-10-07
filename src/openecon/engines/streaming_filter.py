"""Native polynomial filters whose exact state survives bounded input blocks.

The inner filter uses cyclic reduction. Python loops run over polynomial terms,
never over observations; no block is treated as a new time series.
"""
from __future__ import annotations

import torch

from openecon.econometrics.arima.filters import apply_polynomial, inverse_filter
from openecon.engines.contracts import KernelError


class PolynomialState:
    def __init__(self, coefficients, *, inverse=False, unit=1):
        self.coefficients = list(coefficients)
        self.inverse, self.unit = bool(inverse), int(unit)
        if self.unit < 1:
            raise KernelError("invalid_spec", "Filter lag unit must be positive.")
        self.width = (len(self.coefficients) if inverse else max(0, len(self.coefficients)-1))*self.unit
        self.history = None

    def __call__(self, values):
        if values.ndim != 2 or values.dtype != torch.float64:
            raise KernelError("invalid_design", "Stateful filters need float64 rows by periods.")
        rows, count = values.shape
        if not count or not self.width:
            return values.clone()
        if self.history is None:
            self.history = torch.zeros((rows, self.width), dtype=torch.float64)
        if self.history.shape[0] != rows:
            raise KernelError("invalid_design", "A filter's row dimension cannot change between blocks.")
        if self.inverse:
            adjusted = values.clone()
            for lag, coefficient in enumerate(self.coefficients, 1):
                distance = lag*self.unit
                used = min(distance, count)
                adjusted[:, :used] -= coefficient*self.history[:, self.width-distance:self.width-distance+used]
            output = inverse_filter(adjusted, self.coefficients, self.unit)
            joined = torch.cat((self.history, output), dim=1)
        else:
            joined = torch.cat((self.history, values), dim=1)
            output = apply_polynomial(joined, self.coefficients)[:, self.width:]
        self.history = joined[:, -self.width:].clone()
        if not bool(torch.isfinite(output).all()):
            raise KernelError("numerical_failure", "Polynomial filter overflowed; rescale data or change unstable orders.")
        return output


class LagState:
    def __init__(self, maximum):
        self.maximum = int(maximum)
        self.history = None

    def __call__(self, values, lags):
        if self.history is None:
            self.history = torch.zeros(self.maximum, dtype=torch.float64)
        joined = torch.cat((self.history, values))
        output = [joined[self.maximum-lag:self.maximum-lag+len(values)] for lag in lags]
        self.history = joined[-self.maximum:].clone() if self.maximum else joined[:0]
        return output
