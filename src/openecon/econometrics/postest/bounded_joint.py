"""Fixed-sample simultaneous bounds for prespecified bounded population means.

The guarantees use iid rows, known support and a union bound. Cross-coordinate
dependence is unrestricted. They are conservative probability inequalities,
not normal/Student-t inference, exact nominal coverage or optional-stopping
confidence sequences. Empirical Bernstein uses Maurer/Pontil (2009), theorem 4.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.index_codec import encode as encode_label
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.resources import plan_workspace

MAX_ROWS = 100_000
MAX_COLUMNS = 32
MAX_WORK = 100_000_000
MAX_IDENTITY_BYTES = 4 * 1024**2
MAX_LABEL_BYTES = 4096
SOURCE = "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AnalysisError("invalid_option", f"{name} must be a finite real number.")
    original = value
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise AnalysisError("invalid_option", f"{name} must be a finite real number.") from exc
    if not math.isfinite(value):
        raise AnalysisError("invalid_option", f"{name} must be a finite real number.")
    if isinstance(original, Integral) and value != int(original):
        raise AnalysisError("numerical_range", f"{name} is not exactly representable in float64.")
    return value


def _bounds(bounds, names):
    if isinstance(bounds, Mapping):
        if set(bounds) != set(names):
            raise AnalysisError("invalid_bounds", "Bounds must name exactly the requested columns.")
        bounds = [bounds[name] for name in names]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != len(names):
        raise AnalysisError("invalid_bounds", "Supply one prespecified (lower, upper) pair per column.")
    pairs = []
    for pair in bounds:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AnalysisError("invalid_bounds", "Each support bound must be a (lower, upper) pair.")
        low, high = _number(pair[0], "lower support"), _number(pair[1], "upper support")
        width = high - low
        if low >= high or not math.isfinite(width) or width <= 0:
            raise AnalysisError("invalid_bounds", "Support needs a finite, resolved positive width.")
        pairs.append((low, high))
    return pairs


def _identity_scan(index):
    """Check/hash typed row identities without retaining their full list yet."""
    total = 0
    digest = hashlib.sha256()
    for value in index:
        coded = encode_label(value).encode("ascii")
        total += len(coded)
        if len(coded) > MAX_LABEL_BYTES or total > MAX_IDENTITY_BYTES:
            raise AnalysisError("work_budget", "Encoded row labels exceed the bounded identity domain.")
        digest.update(len(coded).to_bytes(4, "little"))
        digest.update(coded)
    return total, digest.hexdigest()


def _sample(data, columns, bounds, sampling_model, alpha, missing, method):
    alpha = _number(alpha, "alpha")
    if not 1e-12 <= alpha < 1:
        raise AnalysisError("invalid_option", "alpha must be in [1e-12, 1).")
    if not isinstance(sampling_model, str) or sampling_model != "iid_bounded":
        raise AnalysisError("unsupported_sampling_model", "Declare sampling_model='iid_bounded' explicitly.")
    if not isinstance(data, pd.DataFrame):
        raise AnalysisError("unsupported_domain", "Bounded mean regions require a resident pandas DataFrame.")
    if not isinstance(columns, (list, tuple)) or not columns or any(
        not isinstance(c, str) or not c for c in columns
    ) or len(set(columns)) != len(columns):
        raise AnalysisError("invalid_columns", "Supply a nonempty list of unique numeric column names.")
    names = list(columns)
    raw_n, m = len(data), len(names)
    # Admission precedes label/numeric scans, projected copies and tensor creation.
    work = raw_n * m * 24 + raw_n * 16 + m * 64
    if raw_n > MAX_ROWS or m > MAX_COLUMNS or work > MAX_WORK:
        raise AnalysisError("work_budget", "Bounded mean regions allow at most 100000 rows and 32 columns.")
    if raw_n == 0:
        raise AnalysisError("insufficient_observations", "Supply at least one observation.")
    if data.columns.has_duplicates or any(name not in data for name in names):
        raise AnalysisError("invalid_columns", "Required columns must each exist exactly once.")
    if not isinstance(missing, str) or missing not in ("raise", "drop"):
        raise AnalysisError("invalid_missing", "Choose missing='raise' or missing='drop'.")
    if any(not pd.api.types.is_numeric_dtype(data[name].dtype)
           or pd.api.types.is_bool_dtype(data[name].dtype)
           or pd.api.types.is_complex_dtype(data[name].dtype) for name in names):
        raise AnalysisError("invalid_values", "Selected columns must contain real numeric values, excluding booleans.")
    pairs = _bounds(bounds, names)
    identity_bytes, raw_index_hash = _identity_scan(data.index)
    plan = plan_workspace(method, {
        "projected pandas sample and raw numerical conversion": 8 * raw_n * m * 3,
        "retained, normalized, shifted, centered and squared buffers": 8 * raw_n * m * 5,
        "missing, finite, support masks and sample-selection buffers": 4 * raw_n * m + 8 * raw_n * 3,
        "column reductions, supports and endpoints": 8 * m * 24,
        "position, typed-index and hash working buffers": 8 * raw_n * 6 + identity_bytes * 2,
    }).record()
    selected = data.loc[:, names]
    for name in names:
        series = selected[name]
        if pd.api.types.is_integer_dtype(series.dtype):
            minimum, maximum = series.min(skipna=True), series.max(skipna=True)
            if not pd.isna(minimum) and (int(minimum) < -(2**53) or int(maximum) > 2**53):
                raise AnalysisError("numerical_range", "Integer observations must lie within the exact float64 domain [-2**53,2**53].")
        elif getattr(series.dtype, "itemsize", 8) > 8:
            raise AnalysisError("numerical_range", "Use float64-or-narrower numeric columns; wider values may lose support identity.")
    incomplete = selected.isna().any(axis=1)
    if bool(incomplete.any()) and missing == "raise":
        raise AnalysisError("missing_values", "All columns require one complete sample; select missing='drop' explicitly.")
    try:
        raw_array = selected.to_numpy(dtype="float64", na_value=math.nan, copy=True)
        raw = torch.as_tensor(raw_array, dtype=torch.float64, device="cpu")
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_values", "Selected values must be representable as finite float64 numbers.") from exc
    low = torch.tensor([a for a, _ in pairs], dtype=torch.float64, device="cpu")
    high = torch.tensor([b for _, b in pairs], dtype=torch.float64, device="cpu")
    present = ~torch.isnan(raw)
    # Validate even available values in rows excluded for another column's NaN.
    if bool((present & ~torch.isfinite(raw)).any()):
        raise AnalysisError("nonfinite_values", "Infinite values are not missing observations.")
    if bool((present & ((raw < low) | (raw > high))).any()):
        raise AnalysisError("outside_support", "An observed value lies outside its prespecified support.")
    complete = ~incomplete.to_numpy()
    positions = complete.nonzero()[0].tolist()
    n = len(positions)
    minimum = 2 if method == "empirical_bernstein_mean_ci" else 1
    if n < minimum:
        raise AnalysisError("insufficient_observations", f"{method} requires at least {minimum} complete rows.")
    mask = torch.as_tensor(complete, dtype=torch.bool, device="cpu")
    x = raw[mask]
    width = high - low
    normalized = (x - low) / width
    if not bool(torch.isfinite(normalized).all()):
        raise AnalysisError("numerical_range", "Support normalization is unresolved in float64.")
    # Only roundoff at verified support endpoints may be clipped.
    normalized = normalized.clamp(0, 1)
    # Origin shifting gives an exactly zero variance for constant stored samples.
    origin = normalized[0]
    shifted = normalized - origin
    mean_shift = shifted.mean(0)
    mean = origin + mean_shift
    centered = shifted - mean_shift
    variance = centered.square().sum(0) / (n - 1) if n > 1 else torch.zeros_like(mean)
    if bool(((centered != 0).any(0) & (variance == 0)).any()):
        raise AnalysisError("numerical_range", "Nonzero sample variance underflows float64 normalization.")
    labels = [encode_label(data.index[position]) for position in positions]
    sample_index_hash = _identity_scan(data.index[complete])[1]
    from openecon.analysis import _frame_hasher

    sample_hash = _frame_hasher(selected.iloc[positions].reset_index(drop=True)).hexdigest()
    actual = {
        "raw_float64_array_bytes": raw_array.nbytes,
        "retained_tensor_bytes": x.numel() * x.element_size(),
        "normalized_tensor_bytes": normalized.numel() * normalized.element_size(),
        "shifted_tensor_bytes": shifted.numel() * shifted.element_size(),
        "centered_tensor_bytes": centered.numel() * centered.element_size(),
        "complete_mask_bytes": mask.numel() * mask.element_size(),
        "encoded_retained_label_bytes": sum(len(label.encode("ascii")) for label in labels),
        "scope": "observed sizes of named buffers; not peak RSS or allocator/private workspace",
    }
    return dict(alpha=alpha, names=names, pairs=pairs, n=n, raw_n=raw_n, low=low,
                high=high, width=width, mean=mean, variance=variance, positions=positions,
                labels=labels, raw_index_hash=raw_index_hash, sample_index_hash=sample_index_hash,
                sample_hash=sample_hash, plan=plan, work=work, actual=actual,
                raw_identity_bytes=identity_bytes)


def _region(sample, method, missing):
    s = sample
    m, n, alpha = len(s["names"]), s["n"], s["alpha"]
    delta = alpha / (2 * m)
    log_factor = math.log(2 * m) - math.log(alpha)
    if method == "hoeffding_mean_ci":
        normalized_radius = torch.full((m,), math.sqrt(log_factor / (2 * n)),
                                       dtype=torch.float64, device="cpu")
        formula = "width*sqrt(log(2*m/alpha)/(2*n))"
        theorem = "Hoeffding (1963); Maurer/Pontil (2009), theorem 1; union over 2*m tails"
    else:
        log_factor = math.log(4 * m) - math.log(alpha)
        normalized_radius = (2 * s["variance"] * log_factor / n).sqrt() \
            + 7 * log_factor / (3 * (n - 1))
        formula = "sqrt(2*s_squared*log(4*m/alpha)/n)+7*width*log(4*m/alpha)/(3*(n-1))"
        theorem = "Maurer/Pontil (2009), theorem 4 and mirrored bound; union over 2*m tails"
    estimate = s["low"] + s["width"] * s["mean"]
    radius = s["width"] * normalized_radius
    std_dev = s["width"] * s["variance"].sqrt()
    variance = std_dev.square()
    low = s["low"] + s["width"] * (s["mean"] - normalized_radius).clamp(0, 1)
    high = s["low"] + s["width"] * (s["mean"] + normalized_radius).clamp(0, 1)
    if any(not bool(torch.isfinite(v).all()) for v in (estimate, radius, variance, low, high)) \
            or bool(((s["variance"] > 0) & (variance == 0)).any()) \
            or bool(((normalized_radius > 0) & (radius == 0)).any()):
        raise AnalysisError("numerical_range", "Mean, variance, radius or endpoints exceed resolved float64 reporting units.")
    # Outward rounding protects endpoint inclusion; still intersect known support.
    lows = [max(a, math.nextafter(float(v), -math.inf))
            for (a, _), v in zip(s["pairs"], low, strict=True)]
    highs = [min(b, math.nextafter(float(v), math.inf))
             for (_, b), v in zip(s["pairs"], high, strict=True)]
    interval = table({"variable": s["names"], "estimate": estimate.tolist(),
                      "ci_low": lows, "ci_high": highs, "radius_unclipped": radius.tolist(),
                      "support_low": s["low"].tolist(), "support_high": s["high"].tolist()})
    diagnostics = table({"variable": s["names"], "nobs": [n] * m,
                         "sample_variance": variance.tolist() if n > 1 else [None] * m,
                         "normalized_sample_variance": s["variance"].tolist() if n > 1 else [None] * m,
                         "one_sided_failure_budget": [delta] * m,
                         "log_factor": [log_factor] * m})
    attrs = dict(
        method=method, sampling_model="iid_bounded", alpha=alpha,
        family_size=m, family_members=s["names"], bounds=[list(pair) for pair in s["pairs"]],
        bounds_origin="Caller-prespecified population support; never inferred from sample extrema",
        nobs=n, nobs_original=s["raw_n"], dropped_rows=s["raw_n"] - n,
        sample_positions=s["positions"], sample_position_base=0,
        sample_labels_encoded=s["labels"], sample_label_codec="postest.index_codec.encode",
        raw_index_hash=s["raw_index_hash"], sample_index_hash=s["sample_index_hash"],
        sample_hash=s["sample_hash"], missing_policy=missing,
        missing_sample_contract="Complete-case aligned family; guarantee needs retained rows iid for the declared target, including after deletion",
        assumptions="Fixed n; iid rows within each marginal; arbitrary cross-coordinate dependence; support and family fixed before outcomes; no informative selection or optional stopping",
        guarantee="Joint probability at least 1-alpha under declared assumptions; conservative finite-sample inequality, not exact nominal coverage",
        assumptions_verified=False, formula=formula, theorem=theorem, source=SOURCE,
        one_sided_failure_budget=delta, tail_count=2 * m, log_factor=log_factor,
        covariance="No joint covariance law assumed; marginal variance enters the Bernstein inequality"
        if method == "empirical_bernstein_mean_ci" else "No covariance model used; marginal sample variance is descriptive only",
        variance_convention="Bessel/unbiased U-stat variance with denominator n-1" if n > 1
        else "Undefined with one row; null variance diagnostics are unused by Hoeffding",
        inference="No normal SE, p-value, fitted-model degrees of freedom or likelihood",
        endpoint_policy="Support intersection and one float64 ULP outward rounding",
        weights_support=False, dataset_support=False, precision="float64", device="cpu",
        input_conversion="Float64-or-narrower numeric columns; integer observations restricted to exact [-2**53,2**53] domain",
        stochastic_draws=0, max_rows=MAX_ROWS, max_columns=MAX_COLUMNS,
        estimated_work=s["work"], resource_plan=s["plan"], workspace_observed=s["actual"],
        raw_encoded_identity_bytes=s["raw_identity_bytes"], stata_parity_validated=False,
        persistence="Full attrs and all table rows via oe.summary_state; settings preview pointers for large state",
    )
    sample_table = table([{
        "nobs_original": s["raw_n"], "nobs": n, "dropped_rows": s["raw_n"] - n,
        "missing_policy": missing, "sample_hash": s["sample_hash"],
        "sample_index_hash": s["sample_index_hash"], "position_base": 0,
    }])
    return saved_summary(TableSet({"intervals": interval, "diagnostics": diagnostics,
                                  "sample": sample_table},
                                 title=method + " — simultaneous bounded mean region", **attrs))


@resident_cpu
@torch.no_grad()
def hoeffding_mean_ci(data, columns, *, bounds, sampling_model, alpha=.05, missing="raise") -> TableSet:
    """Conservative fixed-n simultaneous mean intervals with prespecified support.

    Explicit ``sampling_model='iid_bounded'`` declares iid rows. Arbitrary
    cross-coordinate dependence is allowed. Bounds are pairs in column order
    or a mapping naming exactly those columns. One shared complete sample is
    used; explicit deletion requires the retained sample still iid for target.
    Radius is width*sqrt(log(2*m/alpha)/(2*n)), intersected with known support.
    Resident CPU float64 only; n<=100000, m<=32; no weights or Dataset route.
    Integer observations must be in [-2**53,2**53]; wider floats are refused.
    """
    return _region(_sample(data, columns, bounds, sampling_model, alpha, missing,
                           "hoeffding_mean_ci"), "hoeffding_mean_ci", missing)


@resident_cpu
@torch.no_grad()
def empirical_bernstein_mean_ci(data, columns, *, bounds, sampling_model, alpha=.05,
                                missing="raise") -> TableSet:
    """Maurer/Pontil theorem-4 simultaneous bounded mean intervals, n>=2.

    Declare ``sampling_model='iid_bounded'`` and caller-prespecified finite
    support pairs or an exact column mapping. Use unbiased sample variance,
    delta=alpha/(2*m), and mirror the one-sided theorem. Radius is
    sqrt(2*s_squared*log(4*m/alpha)/n)+7*width*log(4*m/alpha)/(3*(n-1));
    support intersection keeps conservative joint coverage. Constant samples
    retain the finite-n width term. Fixed n, no outcome-based family/support
    selection or optional stopping; deletion must preserve the target's iid law.
    """
    return _region(_sample(data, columns, bounds, sampling_model, alpha, missing,
                           "empirical_bernstein_mean_ci"), "empirical_bernstein_mean_ci", missing)
