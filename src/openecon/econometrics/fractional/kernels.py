"""Differentiable linear-memory finite filters and conditional likelihood."""

from __future__ import annotations

import math
import torch

FLOAT = torch.float64


def weights(d, terms):
    d = torch.as_tensor(d, dtype=FLOAT)
    k = torch.arange(1, terms, dtype=FLOAT)
    return torch.cat((torch.ones(1, dtype=FLOAT), torch.cumprod((k - 1 - d) / k, 0)))


def convolve(a, b, length=None):
    """Causal linear convolution, with zero padding, never circular filtering."""
    size = len(a) + len(b) - 1
    fft = 1 << (size - 1).bit_length()
    value = torch.fft.irfft(torch.fft.rfft(a, fft) * torch.fft.rfft(b, fft), fft)
    return value[:size if length is None else min(size, length)]


def inverse(b, length):
    """Formal power-series reciprocal by Newton doubling, O(n log n) storage/work.

    Each step doubles the number of correct causal coefficients. This is a
    finite polynomial identity, not an FFT division on a circular sample.
    """
    value = torch.ones(1, dtype=FLOAT) / b[0]
    while len(value) < length:
        size = min(2 * len(value), length)
        product = convolve(b[:size], value, size)
        product = torch.nn.functional.pad(product, (0, size - len(product)))
        unit = torch.cat((torch.ones(1, dtype=FLOAT), torch.zeros(size - 1, dtype=FLOAT)))
        value = convolve(value, 2 * unit - product, size)
    return value


def pacf_coefficients(raw):
    """Levinson reflection-coefficient map to a stationary AR polynomial."""
    coefficients = raw[:0]
    for reflection in .98 * torch.tanh(raw):
        coefficients = torch.cat((coefficients - reflection * coefficients.flip(0),
                                  reflection.reshape(1)))
    return coefficients


def innovations(y, mean, d, ar, ma, terms):
    polynomial = convolve(weights(d, terms), torch.cat((torch.ones(1, dtype=FLOAT), -ar)))
    transformed = convolve(y - mean, polynomial, len(y))
    if len(ma):
        transformed = convolve(transformed, inverse(
            torch.cat((torch.ones(1, dtype=FLOAT), ma)), len(y)), len(y))
    return transformed, polynomial


def loglike(y, mean, d, ar, ma, sigma, terms, burn):
    error, _ = innovations(y, mean, d, ar, ma, terms)
    error = error[burn:]
    return -.5 * (len(error) * (math.log(2 * math.pi) + 2 * torch.log(sigma))
                  + error.square().sum() / sigma.square())
