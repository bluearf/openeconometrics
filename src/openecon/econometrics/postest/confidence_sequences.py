"""Conservative, time-uniform intervals from summable fixed-time error budgets.

For a fixed family of m marginal targets, look n spends alpha/(m*n*(n+1)).
Summing over all n and all targets is alpha, so the union bound controls any
data-dependent reporting/stopping time. This is an error-spending construction,
not a mixture-martingale or LIL-width construction. Rows remain in input order.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.bounded_joint import _bounds, _identity_scan, _number
from openecon.econometrics.postest.index_codec import encode as encode_label
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import chi2_isf, chi2_ppf, f_isf, normal_isf, t_isf
from openecon.resources import plan_workspace

MAX_ROWS = 512
MAX_COLUMNS = 8
MAX_TOTAL_COUNT = 1_000_000
MAX_MAGNITUDE = 1e100
MIN_POSITIVE = 1e-100
RESERVED = {"position", "index_identity"}
MODELS = {
    "bernoulli": "iid_bernoulli", "poisson": "iid_poisson",
    "normal_mean": "iid_normal", "student_mean": "iid_normal",
    "normal_variance": "iid_normal", "exponential_mean": "iid_exponential",
    "uniform_endpoint": "iid_uniform_zero", "hoeffding": "iid_bounded",
}


def _error(code, message):
    raise AnalysisError(code, message)


def _sample(data, columns, sampling_model, alpha, kind):
    alpha = _number(alpha, "alpha")
    if not .001 <= alpha < .5:
        _error("invalid_alpha", "alpha must be in [0.001, 0.5).")
    if not isinstance(sampling_model, str) or sampling_model != MODELS[kind]:
        _error("unsupported_sampling_model", f"Declare sampling_model='{MODELS[kind]}' explicitly.")
    if not isinstance(data, pd.DataFrame):
        _error("unsupported_data", "Supply a resident DataFrame in observation order.")
    if not isinstance(columns, (list, tuple)) or not 1 <= len(columns) <= MAX_COLUMNS:
        _error("invalid_columns", "Supply one to eight prespecified column names.")
    names = list(columns)
    if any(not isinstance(c, str) or not c or len(c) > 256 for c in names) or len(set(names)) != len(names):
        _error("invalid_columns", "Column names must be distinct nonempty bounded strings.")
    n, m = len(data), len(names)
    if not 1 <= n <= MAX_ROWS:
        _error("work_budget", "A resident prefix must contain 1..512 rows; no rows are truncated.")
    if data.columns.has_duplicates or any(c not in data or c in RESERVED for c in names):
        _error("invalid_columns", "Selected columns must exist once and avoid position/index_identity.")
    for name in names:
        dtype = data[name].dtype
        if (not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_complex_dtype(dtype)
                or (getattr(dtype, "kind", None) == "f" and dtype.itemsize > 8)):
            _error("invalid_values", "Observations must be real numeric values representable in float64.")
    identity_bytes, identity_hash = _identity_scan(data.index)
    resource = plan_workspace("time-uniform confidence sequence", {
        "projected input, tensors and prefix moments": 8*n*m*12,
        "complete interval output, Python rows and saved-state buffers": 64*n*m*16,
        "typed ordered identities and row positions": 2*identity_bytes + 64*n,
    }).record()
    for name in names:
        for value in data[name]:
            if isinstance(value, Integral) and abs(value) > 2**53:
                _error("invalid_values", "Integers must be exactly representable in float64.")
    try:
        x = torch.tensor(data.loc[:, names].to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise AnalysisError("invalid_values", "Cannot represent observations in float64.") from exc
    if not bool(torch.isfinite(x).all()):
        _error("missing_values", "Every ordered observation must be finite; missing deletion is unsupported.")
    if bool((x.abs() > MAX_MAGNITUDE).any()):
        _error("numerical_range", "Absolute observations must not exceed 1e100.")
    if kind == "bernoulli" and not bool(((x == 0) | (x == 1)).all()):
        _error("invalid_binary", "Bernoulli observations must be numeric 0 or 1.")
    if kind == "poisson":
        if not bool(((x >= 0) & (x == x.floor())).all()):
            _error("invalid_counts", "Poisson observations must be nonnegative integer counts.")
        if bool((x.sum(0) > MAX_TOTAL_COUNT).any()):
            _error("work_budget", "Cumulative counts may not exceed 1000000 per marginal.")
    if kind in ("exponential_mean", "uniform_endpoint") and bool((x < MIN_POSITIVE).any()):
        _error("invalid_support", "Continuous positive observations must be at least 1e-100; zero is unsupported.")
    sample = table([[i, encode_label(label), *row] for i, (label, row) in
                    enumerate(zip(data.index, x.tolist(), strict=True))],
                   columns=["position", "index_identity", *names])
    metadata = dict(n=n, family_size=m, columns=names, sampling_model=sampling_model,
                    sampling_order="Input row positions; never sorted by index or outcome",
                    missing_policy="raise", sample_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                    ordered_index_sha256=identity_hash, resource_plan=resource)
    return x, sample, metadata, alpha


def _sds(sd, names):
    if isinstance(sd, Mapping):
        if set(sd) != set(names):
            _error("invalid_sd", "Known SD must name exactly the fixed family.")
        sd = [sd[name] for name in names]
    if not isinstance(sd, (list, tuple)) or len(sd) != len(names):
        _error("invalid_sd", "Supply one known population SD per marginal.")
    values = [_number(v, "known SD") for v in sd]
    if any(not MIN_POSITIVE <= v <= MAX_MAGNITUDE for v in values):
        _error("invalid_sd", "Known SD must be in [1e-100, 1e100].")
    return values


def _binomial_limits(k, n, tail):
    # Reciprocal F identity avoids forming 1-tail at extreme later looks.
    low = 0. if k == 0 else k/(k+(n-k+1)*f_isf(tail, 2*(n-k+1), 2*k))
    if k == n:
        return low, 1.
    f = f_isf(tail, 2*(k+1), 2*(n-k))
    return low, (k+1)*f/(n-k+(k+1)*f)


def _outward(low, high, support):
    """Keep null for genuinely unbounded endpoints; widen finite float outputs."""
    if low is not None and not math.isfinite(low) or high is not None and not math.isfinite(high):
        _error("numerical_range", "Confidence endpoints exceed finite float64 reporting units.")
    if low is not None and high is not None and low > high:
        _error("numerical_range", "Confidence endpoints are numerically unresolved.")
    # Native scalar quantiles are independently checked; this additionally widens
    # their final arithmetic by 128 relative ulps, then intersects known support.
    if low is not None:
        low = math.nextafter(low-128*math.ulp(low), -math.inf)
        if support[0] is not None:
            low = max(support[0], low)
    if high is not None:
        high = math.nextafter(high+128*math.ulp(high), math.inf)
        if support[1] is not None:
            high = min(support[1], high)
    return low, high


def _sequence(data, columns, sampling_model, alpha, kind, *, sd=None, bounds=None):
    x, sample, meta, alpha = _sample(data, columns, sampling_model, alpha, kind)
    nmax, m = x.shape
    sds = _sds(sd, meta["columns"]) if kind == "normal_mean" else None
    pairs = _bounds(bounds, meta["columns"]) if kind == "hoeffding" else None
    if pairs is not None:
        for j, (a, b) in enumerate(pairs):
            if abs(a) > MAX_MAGNITUDE or abs(b) > MAX_MAGNITUDE:
                _error("numerical_range", "Support endpoints must not exceed 1e100 in absolute value.")
            if bool(((x[:, j] < a) | (x[:, j] > b)).any()):
                _error("invalid_support", "An observation lies outside its prespecified support.")
    rows = []
    for n in range(1, nmax+1):
        local_alpha = alpha/(m*n*(n+1))
        tail = local_alpha/2
        prefix = x[:n]
        # An origin shift avoids catastrophic cancellation of raw squared sums.
        delta = prefix-prefix[0]
        mean_delta = delta.mean(0)
        means = prefix[0]+mean_delta
        ss = ((delta-mean_delta).square()).sum(0)
        totals = prefix.sum(0) if kind in ("bernoulli", "poisson", "exponential_mean") else None
        for j, name in enumerate(meta["columns"]):
            estimate = float(means[j])
            support = (None, None)
            status = "finite"
            if kind == "bernoulli":
                low, high = _binomial_limits(int(totals[j]), n, tail)
                support = (0., 1.)
            elif kind == "poisson":
                k = int(totals[j])
                low = 0. if k == 0 else chi2_ppf(tail, 2*k)/(2*n)
                high = chi2_isf(tail, 2*(k+1))/(2*n)
                support = (0., None)
            elif kind == "normal_mean":
                radius = normal_isf(tail)*sds[j]/math.sqrt(n)
                low, high = estimate-radius, estimate+radius
            elif kind in ("student_mean", "normal_variance"):
                variance = float(ss[j])/(n-1) if n > 1 else None
                if n > 1 and float(ss[j]) > 0 and variance == 0:
                    _error("numerical_range", "The observed variance underflows float64 reporting units.")
                if kind == "normal_variance":
                    estimate, support = variance, (0., None)
                if n == 1 or float(ss[j]) == 0:
                    # The positive-variance continuous Gaussian model assigns
                    # exactly constant prefixes probability zero. Do not turn
                    # rounded/degenerate observations into spurious certainty.
                    low, high = support[0], None
                    status = "insufficient_prefix" if n == 1 else "zero_observed_variation"
                elif kind == "student_mean":
                    radius = t_isf(tail, n-1)*math.sqrt(variance/n)
                    low, high = estimate-radius, estimate+radius
                else:
                    low = float(ss[j])/chi2_isf(tail, n-1)
                    high = float(ss[j])/chi2_ppf(tail, n-1)
            elif kind == "exponential_mean":
                total = float(totals[j])
                low, high = 2*total/chi2_isf(tail, 2*n), 2*total/chi2_ppf(tail, 2*n)
                support = (0., None)
            elif kind == "uniform_endpoint":
                estimate = float(prefix[:, j].max())
                low = estimate/math.exp(math.log1p(-tail)/n)
                high = estimate/math.exp(math.log(tail)/n)
                support = (0., None)
            else:
                a, b = pairs[j]
                radius = (b-a)*math.sqrt(math.log(2/local_alpha)/(2*n))
                low, high, support = max(a, estimate-radius), min(b, estimate+radius), (a, b)
            if estimate is not None and not math.isfinite(estimate):
                _error("numerical_range", "The estimate exceeds float64 reporting units.")
            if kind in ("normal_variance", "exponential_mean", "uniform_endpoint") and status == "finite" and (low <= 0 or high <= 0):
                _error("numerical_range", "A positive confidence endpoint underflows float64 reporting units.")
            low, high = _outward(low, high, support)
            rows.append([n, name, estimate, low, high, low is None, high is None,
                         local_alpha, status])
    first = 2 if kind in ("student_mean", "normal_variance") else 1
    meta.update(alpha=alpha, minimum_family_coverage=1-alpha, simultaneous=True,
                time_uniform=True, optional_stopping_valid=True, precision="float64", device="cpu",
                method=kind, spending="alpha / (family_size * n * (n + 1))",
                infinite_horizon_alpha_budget=alpha,
                prefix_alpha_spent=alpha*(1/first-1/(nmax+1)) if nmax >= first else 0.,
                remaining_alpha_budget=alpha/(nmax+1), first_informative_look=first,
                unused_initial_alpha=alpha*(1-1/first),
                fixed_family_required=True, nested_intervals=False,
                known_sd=sds, known_bounds=pairs, failures=0, dataset_support=False,
                source_reference="https://arxiv.org/abs/1810.08240",
                guarantee="Countable union bound over prespecified marginals and all integer prefixes; conservative error spending",
                assumptions=f"Each column follows {sampling_model}; iid observations within each marginal; arbitrary within-row dependence; fixed targets and scale/support declarations",
                unsupported="Adaptive family selection, dependent rows, weights, missing deletion, censored/shifted exponential, unknown uniform origin, Dataset/online state, tighter mixture/LIL widths",
                numeric_endpoint_guard="128 relative ulps plus outward nextafter; intersect declared parameter support",
                stata_parity_validated=False)
    frames = {"intervals": table(rows, columns=["n", "variable", "estimate", "ci_low", "ci_high",
              "lower_unbounded", "upper_unbounded", "local_alpha", "status"]), "sample": sample}
    return saved_summary(TableSet(frames, title=f"Time-uniform {kind.replace('_', ' ')} confidence sequence", **meta))


@resident_cpu
@torch.no_grad()
def bernoulli_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform Bernoulli proportions via exact binomial error spending.

    Declare iid_bernoulli and a fixed numeric 0/1 family in observation order.
    All prefixes and complete typed row identities are saved. Coverage permits
    arbitrary stopping; intervals are conservative and need not be nested.
    """
    return _sequence(data, columns, sampling_model, alpha, "bernoulli")


@resident_cpu
@torch.no_grad()
def poisson_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform iid Poisson mean limits via Garwood error spending.

    Each row is one equal-exposure count; cumulative counts are at most one
    million per fixed marginal. Zero-count prefixes retain a positive upper
    limit. Unequal exposures, overdispersion and dependent counts are unsupported.
    """
    return _sequence(data, columns, sampling_model, alpha, "poisson")


@resident_cpu
@torch.no_grad()
def normal_mean_confidence_sequence(data, columns, *, sd, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform iid Gaussian means with prespecified known population SD.

    Supply sd as a list aligned to columns or an exact-name mapping. Error
    spending uses exact normal pivots; estimating SD from these data is invalid.
    """
    return _sequence(data, columns, sampling_model, alpha, "normal_mean", sd=sd)


@resident_cpu
@torch.no_grad()
def student_mean_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform iid Gaussian means with unknown positive population SD.

    Student-t error spending starts at n=2. The first and exactly constant
    prefixes report the whole real line with explicit unbounded flags.
    """
    return _sequence(data, columns, sampling_model, alpha, "student_mean")


@resident_cpu
@torch.no_grad()
def normal_variance_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform unknown-mean iid Gaussian population variance limits.

    Exact chi-square pivots start at n=2. The first and exactly constant prefixes
    report the whole nonnegative line; this is variance, not standard deviation.
    """
    return _sequence(data, columns, sampling_model, alpha, "normal_variance")


@resident_cpu
@torch.no_grad()
def exponential_mean_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform means of uncensored iid zero-origin exponential observations.

    Strictly positive samples use the exact cumulative-sum gamma pivot. The
    target is the mean, not the rate; censoring and estimated origins are refused.
    """
    return _sequence(data, columns, sampling_model, alpha, "exponential_mean")


@resident_cpu
@torch.no_grad()
def uniform_endpoint_confidence_sequence(data, columns, *, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform upper endpoints of iid Uniform(0, theta) marginals.

    The lower support must be known zero. Positive prefix maxima use the exact
    maximum pivot with equal-tail spending, evaluated without tiny powers.
    """
    return _sequence(data, columns, sampling_model, alpha, "uniform_endpoint")


@resident_cpu
@torch.no_grad()
def hoeffding_confidence_sequence(data, columns, *, bounds, sampling_model, alpha=.05) -> TableSet:
    """Time-uniform bounded iid means via Hoeffding and summable error spending.

    Supply prespecified finite (lower, upper) supports aligned to columns or in
    an exact-name mapping. Observed extrema cannot replace known population
    bounds. Intervals intersect those supports; no distribution is estimated.
    """
    return _sequence(data, columns, sampling_model, alpha, "hoeffding", bounds=bounds)
