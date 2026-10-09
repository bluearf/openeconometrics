"""Finite fractional differencing and Geweke--Porter-Hudak diagnostics."""

from __future__ import annotations

import math
from numbers import Real
import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import kernels as k

MAX_ROWS = 1_000_000
MAX_TERMS = 16384


def number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise AnalysisError("invalid_option", f"{name} must be a finite real number.")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise AnalysisError("invalid_option", f"{name} is outside its supported domain.")
    return float(value)


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise AnalysisError("invalid_option", f"{name} must be an integer in [{lower}, {upper}].")
    return value


def allocation(n, *, fit=False):
    # Conservative live FFT/autograd buffers; caller-owned input/RSS are separate.
    fft = 1 << (2 * n - 1).bit_length()
    return plan_workspace("fractional-memory FFT", {"FFT/autograd buffers": fft * (768 if fit else 96)},
                          budget_bytes=workspace_budget_bytes())


def regular_time(values):
    series = pd.Series(values)
    if not pd.api.types.is_integer_dtype(series.dtype) or series.isna().any():
        raise AnalysisError("invalid_time", "time must be a complete integer unit-spaced calendar.")
    if len(series) > 1 and not (series.diff().iloc[1:] == 1).all():
        raise AnalysisError("time_gaps", "Rows must be in calendar order, with unique consecutive periods.")


def series(data, column=None, time=None, maximum=MAX_ROWS):
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "Fractional-memory procedures require resident input.")
    if isinstance(data, (pd.DataFrame, dict)):
        frame = pd.DataFrame(data)
        if column is None or column not in frame or frame.columns.has_duplicates:
            raise AnalysisError("invalid_columns", "Specify one unique numeric outcome column.")
        if time is not None:
            if time == column or time not in frame:
                raise AnalysisError("invalid_time", "time must name a distinct calendar column.")
            regular_time(frame[time])
        data = frame[column]
    elif column is not None or time is not None:
        raise AnalysisError("invalid_columns", "column/time are only accepted with a resident table.")
    if isinstance(data, (str, bytes)) or not hasattr(data, "__len__"):
        raise AnalysisError("invalid_values", "Supply a bounded one-dimensional real series.")
    n = len(data)
    if not 2 <= n <= maximum:
        raise AnalysisError("work_budget", f"The series must have 2..{maximum} rows.")
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) != 1:
        raise AnalysisError("invalid_values", "Supply a one-dimensional real series.")
    allocation(n)
    if isinstance(data, torch.Tensor):
        if data.device.type != "cpu" or data.dtype == torch.bool or data.is_complex():
            raise AnalysisError("unsupported_input", "Supply real CPU tensors; no implicit device fallback.")
        value = data.to(dtype=k.FLOAT)
    else:
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in data):
            raise AnalysisError("invalid_values", "Missing, boolean and nonnumeric observations are unsupported.")
        value = torch.tensor(list(data), dtype=k.FLOAT)
    if not torch.isfinite(value).all():
        raise AnalysisError("missing_values", "Fractional memory requires a complete finite series; no rows are dropped.")
    return value


@torch.no_grad()
def fracdiff(data, d, *, column=None, time=None, terms=256, initial="zero", tolerance=None):
    """Apply finite (1-L)^d with zero prehistory or drop initial partial windows.

    ``tolerance`` bounds the *first omitted coefficient*, not the infinite tail
    or transformation error. If the fixed terms budget cannot meet it, raise.
    """
    d = number(d, "d", -1, 1)
    terms = integer(terms, "terms", 2, MAX_TERMS)
    if initial not in ("zero", "drop"):
        raise AnalysisError("invalid_option", "initial must be zero or drop.")
    y = series(data, column, time)
    weights = k.weights(d, terms + 1)
    used = terms
    if tolerance is not None:
        tolerance = number(tolerance, "tolerance", 1e-15, .1)
        candidates = (weights[2:].abs() <= tolerance).nonzero().flatten()
        if not len(candidates):
            raise AnalysisError("truncation_budget", "terms cannot satisfy the first-omitted-coefficient tolerance.")
        used = int(candidates[0]) + 2
    if used > len(y):
        raise AnalysisError("insufficient_observations", "terms must not exceed the series length.")
    start = used - 1 if initial == "drop" else 0
    output = k.convolve(y, weights[:used], len(y))[start:]
    if not torch.isfinite(output).all():
        raise AnalysisError("numerical_failure", "Fractional filtering overflowed; rescale the input explicitly.")
    return table({"position": list(range(start, len(y))), "difference": output.tolist()},
                 d=d, terms=used, requested_terms=terms, initial=initial,
                 filter_weights=weights[:used].tolist(), first_omitted_coefficient=float(weights[used]),
                 tolerance=tolerance, tolerance_scope="first omitted coefficient, not tail error",
                 n_input=len(y), n_dropped=start, missing="raise", weights="unsupported",
                 device="cpu", precision="float64", dataset_support=False,
                 initial_value="zero prehistory", workspace=allocation(len(y)).record())


@torch.no_grad()
def gph(data, *, column=None, time=None, bandwidth=None, alpha=.05):
    """GPH low-frequency log-periodogram estimate; explicit asymptotic inference."""
    y = series(data, column, time)
    n = len(y)
    if n < 64:
        raise AnalysisError("insufficient_observations", "GPH needs at least 64 complete observations.")
    m = integer(int(math.sqrt(n)) if bandwidth is None else bandwidth, "bandwidth", 8, min(n // 4, 100000))
    alpha = number(alpha, "alpha", 1e-8, 1 - 1e-8)
    frequency = 2 * math.pi * torch.arange(1, m + 1, dtype=k.FLOAT) / n
    spectrum = torch.fft.rfft(y - y.mean())[1:m + 1].abs().square() / (2 * math.pi * n)
    if not (spectrum > 0).all() or not torch.isfinite(spectrum).all():
        raise AnalysisError("degenerate_spectrum", "Low-frequency periodogram values must be strictly positive; no epsilon repair.")
    x = -torch.log(4 * torch.sin(frequency / 2).square())
    design = torch.stack((torch.ones_like(x), x), 1)
    bread = torch.linalg.inv(design.T @ design)
    beta = bread @ design.T @ torch.log(spectrum)
    residual = torch.log(spectrum) - design @ beta
    theoretical = bread * (math.pi ** 2 / 6)
    regression = bread * (float(residual.square().sum()) / (m - 2))
    def coefficients(covariance, df):
        se = covariance.diagonal().sqrt()
        critical = critical_value(alpha, df)
        statistic = beta / se
        p = [math.erfc(abs(float(z)) / math.sqrt(2)) if df is None
             else student_t_two_sided(float(z), df) for z in statistic]
        return table({"estimate": beta.tolist(), "std_error": se.tolist(),
                      "statistic": statistic.tolist(), "p_value": p,
                      "ci_low": (beta - critical * se).tolist(), "ci_high": (beta + critical * se).tolist()},
                     index=["intercept", "d"], covariance_matrix=covariance.tolist(), df=df,
                     reference="asymptotic normal" if df is None else "OLS Student t (diagnostic only)")
    return TableSet({"estimate": coefficients(theoretical, None),
                     "regression": coefficients(regression, m - 2),
                     "periodogram": table({"frequency": frequency.tolist(), "periodogram": spectrum.tolist()})},
                    n=n, bandwidth=m, alpha=alpha, d=float(beta[1]),
                    covariance="GPH asymptotic log-periodogram variance pi^2/6",
                    assumptions="stationary fractional process; m/n -> 0 and m -> infinity; short-memory spectral bias remains",
                    stationary_estimate=abs(float(beta[1])) < .5, convergence="finite OLS",
                    device="cpu", precision="float64", missing="raise", weights="unsupported",
                    dataset_support=False, workspace=allocation(n).record())
