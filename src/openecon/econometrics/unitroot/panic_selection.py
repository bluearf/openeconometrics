"""Bai--Ng ICp factor selection and Brownian-bridge limit calibration."""

from __future__ import annotations

from functools import lru_cache
import math
import torch

from openecon.analysis_contracts import AnalysisError


def select_factors(values, trend, criterion, maximum):
    """Bai--Ng (2002), equation (9), on the PANIC differenced common sample."""
    differences = (values[:, 1:] - values[:, :-1]).T
    scale = float(differences.abs().amax())
    if not math.isfinite(scale) or scale <= 0:
        raise AnalysisError(
            "constant_series", "Factor selection requires finite, nonzero differences."
        )
    differences = differences / scale
    if trend == "trend":
        differences -= differences.mean(0)
    t, n = differences.shape
    if maximum >= min(n, t):
        raise AnalysisError("invalid_factor_count", "max_factors must be smaller than N and T-1.")
    spectrum = torch.linalg.svdvals(differences)
    penalty = {
        "icp1": (n + t) / (n * t) * math.log(n * t / (n + t)),
        "icp2": (n + t) / (n * t) * math.log(min(n, t)),
        "icp3": math.log(min(n, t)) / min(n, t),
    }[criterion]
    rows = []
    for k in range(maximum + 1):
        variance = float(spectrum[k:].square().sum()) / (n * t)
        if variance <= 1e-24:
            raise AnalysisError(
                "degenerate_idiosyncratic",
                "Factor selection leaves no identifiable idiosyncratic variance.",
            )
        rows.append(
            {
                "factors": k,
                "residual_variance_scaled": variance,
                "criterion": math.log(variance) + k * penalty,
            }
        )
    selected = min(rows, key=lambda row: row["criterion"])["factors"]
    return selected, {
        "method": criterion,
        "candidates": rows,
        "chosen": selected,
        "common_sample": [2, t + 1],
        "n_panels": n,
        "source": "Bai and Ng (2002), equation (9)",
        "standardized_by_unit": False,
    }


@lru_cache(maxsize=1)
def bridge_calibration():
    """Bai--Ng (2004), Theorem 3.1(1): -1/(2 sqrt(integral B(s)^2 ds)).

    Karhunen--Loeve integral is sum_j Z_j^2/(pi*j)^2. The first 512
    independent normals are simulated with a local seed. Omitted positive
    tail is replaced by its exact expectation; its SD is bounded by
    sqrt(2/(3*pi^4*512^3)). This is a recorded Monte Carlo calibration,
    not a published critical-value table or finite-sample guarantee.
    """
    draws, modes, seed = 65536, 512, 271828
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = (math.pi * torch.arange(1, modes + 1, dtype=torch.float64, device="cpu")) ** -2
    tail_mean = 1 / 6 - float(weights.sum())
    statistics = torch.empty(draws, dtype=torch.float64, device="cpu")
    # Small blocks avoid spawning a large OpenMP team for each short draw.
    for first in range(0, draws, 32):
        normals = torch.randn((32, modes), dtype=torch.float64, generator=generator, device="cpu")
        integral = normals.square() @ weights + tail_mean
        statistics[first : first + 32] = -0.5 / integral.sqrt()
    statistics = statistics.sort().values
    return statistics, {
        "distribution": "Bai--Ng trend idiosyncratic Brownian bridge",
        "method": "Karhunen-Loeve Monte Carlo",
        "seed": seed,
        "draws": draws,
        "modes": modes,
        "cdf_max_standard_error": 0.5 / math.sqrt(draws),
        "omitted_integral_tail_mean": tail_mean,
        "omitted_integral_tail_sd_bound": math.sqrt(2 / (3 * math.pi**4 * modes**3)),
        "finite_sample_calibration": False,
    }


def bridge_probabilities(statistics):
    simulated, metadata = bridge_calibration()
    counts = torch.searchsorted(simulated, statistics)
    probabilities = (counts.to(torch.float64) + 1) / (len(simulated) + 1)
    critical = {
        label: float(torch.quantile(simulated, q))
        for label, q in (("1%", 0.01), ("5%", 0.05), ("10%", 0.10))
    }
    return probabilities, probabilities.log(), critical, metadata
