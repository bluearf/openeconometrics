"""Finite-sample conservative confidence families with declared iid marginals.

No distribution or support is inferred from observations. Arbitrary dependence
within a sampled row or among binomial counts is allowed by a union bound.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral, Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.nonparametric.distribution import clopper_pearson
from openecon.econometrics.postest.multiple import _labels, _numeric, _shape
from openecon.econometrics.stats.planning import _binomial_pmf
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.engines.distributions import f_isf
from openecon.resources import plan_workspace

MAX_ROWS = 10_000
MAX_COLUMNS = 32
MAX_CELLS = 100_000
MAX_FAMILY = 128
MAX_TRIALS = 1_000_000


def _error(code, message):
    raise AnalysisError(code, message)


def _alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 1e-8 <= alpha < .5:
        _error("invalid_alpha", "alpha must be finite, at least 1e-8 and below .5.")
    return float(alpha)


def _model(value, expected):
    if not isinstance(value, str) or value != expected:
        _error("unsupported_sampling_model", f"Declare sampling_model='{expected}' explicitly.")


def _sample(data, columns, missing, *, output_multiplier=1):
    if not isinstance(data, pd.DataFrame):
        _error("unsupported_data", "Provide a resident DataFrame; Dataset/streaming inputs are unsupported.")
    if (isinstance(columns, (str, bytes, dict)) or not hasattr(columns, "__len__")
            or not 1 <= len(columns) <= MAX_COLUMNS):
        _error("invalid_columns", "Select one to 32 distinct numeric column names.")
    columns = list(columns)
    if any(not isinstance(c, str) or not c or len(c) > 256 for c in columns) or len(set(columns)) != len(columns):
        _error("invalid_columns", "Columns must have distinct nonempty names up to 256 characters.")
    if set(columns) & {"original_position", "original_label"}:
        _error("invalid_columns", "Rename outcome columns reserved for sample provenance: original_position/original_label.")
    if any(list(data.columns).count(c) != 1 for c in columns):
        _error("invalid_columns", "Each requested column must exist exactly once.")
    n, m = len(data), len(columns)
    if not 1 <= n <= MAX_ROWS or n * m > MAX_CELLS:
        _error("work_budget", "Raw input exceeds 10000 rows or 100000 selected cells.")
    if not isinstance(missing, str) or missing not in ("raise", "drop"):
        _error("invalid_missing", "missing must be raise or explicit complete-row drop.")
    plan = plan_workspace("distribution-free family", {
        "sample copies, masks, sorted values and rank buffers": 8 * n * m * 8 + 64 * n,
        "complete numeric output buffers": 8 * n * m * output_multiplier * 8,
    }).record()
    selected = data.loc[:, columns]
    for c in columns:
        _numeric(selected[c], c, missing=True)
        if pd.api.types.is_float_dtype(selected[c].dtype) and selected[c].dtype.itemsize > 8:
            _error("invalid_values", "Wider-than-float64 observations require explicit user conversion.")
        for value in selected[c]:
            if isinstance(value, Integral) and not -2**53 <= value <= 2**53:
                _error("invalid_values", "Integer observations must lie within the exact float64 safe range +/-2**53.")
            scalar_dtype = getattr(value, "dtype", None)
            if getattr(scalar_dtype, "kind", None) == "f" and scalar_dtype.itemsize > 8:
                _error("invalid_values", "Wider-than-float64 scalar observations require explicit user conversion.")
    try:
        values = torch.as_tensor(selected.to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64).clone()
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_values", "Selected sample must contain real numeric values.") from exc
    if torch.isinf(values).any():
        _error("invalid_values", "Infinite observations are unsupported.")
    keep = torch.isfinite(values).all(1)
    if not keep.all() and missing == "raise":
        _error("missing_values", "The entire declared family needs complete aligned observations.")
    positions = torch.where(keep)[0].tolist()
    if not positions:
        _error("insufficient_sample", "No complete sample rows remain.")
    labels = []
    for i in positions:
        label = data.index[i]
        if isinstance(label, bool) or not isinstance(label, (str, Integral, Real)):
            _error("invalid_index", "Sample labels must be finite strings or real scalars.")
        if isinstance(label, Real):
            try:
                finite = math.isfinite(float(label))
            except (OverflowError, ValueError):
                finite = False
            if not finite:
                _error("invalid_index", "Sample labels must be finite representable scalars.")
        if isinstance(label, str) and len(label) > 256:
            _error("invalid_index", "Sample labels are limited to 256 characters.")
        labels.append(label.item() if hasattr(label, "item") else label)
    values = values[keep]
    sample = table([[i, label, *row] for i, label, row in zip(positions, labels, values.tolist())],
                   columns=["original_position", "original_label", *columns])
    meta = dict(raw_n=n, n=len(values), columns=columns, missing_policy=missing,
                included_positions=positions, excluded_positions=torch.where(~keep)[0].tolist(),
                sample_sha256=hashlib.sha256(values.numpy().tobytes()).hexdigest(), resource_plan=plan,
                selection_assumption="Prespecified complete-case population; outcome-dependent deletion requires separate justification")
    return values, sample, meta


def _output(name, frames, alpha, **meta):
    return saved_summary(TableSet(frames, title=name, alpha=alpha,
        minimum_family_coverage=1-alpha, simultaneous=True,
        precision="float64", device="cpu", dataset_support=False,
        failures=0, stata_parity_validated=False, **meta))


@resident_cpu
@torch.no_grad()
def simultaneous_dkw_band(data, columns, *, sampling_model, alpha=.05, missing="raise"):
    """Entire-real-line simultaneous empirical CDF bands (DKW-Massart union bound).

    Each marginal is an iid sample; coordinates in a row may be dependent.
    Every unique knot is retained. At x below the first knot use empirical CDF
    zero, between knots use the previous right-CDF, and above the last use one.
    At a knot right-CDF includes all ties; its left limit excludes those ties.
    """
    alpha = _alpha(alpha)
    _model(sampling_model, "iid_marginals")
    x, sample, meta = _sample(data, columns, missing, output_multiplier=2)
    n, m = x.shape
    epsilon = min(1., math.sqrt(math.log(2*m/alpha)/(2*n)))
    rows = []
    for j, c in enumerate(meta["columns"]):
        knots, counts = torch.unique(x[:, j], sorted=True, return_counts=True)
        right = counts.cumsum(0).to(torch.float64)/n
        left = right-counts.to(torch.float64)/n
        for v, left_cdf, right_cdf, count in zip(knots.tolist(), left.tolist(), right.tolist(), counts.tolist()):
            rows.append([c, v, count, left_cdf, right_cdf, max(0., left_cdf-epsilon), min(1., left_cdf+epsilon),
                         max(0., right_cdf-epsilon), min(1., right_cdf+epsilon)])
    return _output("Simultaneous distribution-free CDF bands", {
        "bands": table(rows, columns=["variable", "knot", "ties", "empirical_left", "empirical_right",
              "left_ci_low", "left_ci_high", "right_ci_low", "right_ci_high"]), "sample": sample,
    }, alpha, **meta, family_size=m, sampling_model=sampling_model, epsilon=epsilon,
       method="DKW-Massart with Bonferroni over entire marginal CDFs",
       below_first={"empirical": 0., "ci_low": 0., "ci_high": epsilon},
       above_last={"empirical": 1., "ci_low": 1-epsilon, "ci_high": 1.},
       between_knots="Use right-CDF and its bounds at the immediately preceding knot",
       assumptions="Prespecified iid marginal distributions; arbitrary within-row dependence; discrete/tied samples allowed")


def _probabilities(probabilities):
    (k,) = _shape(probabilities, "probabilities", 1)
    if not 1 <= k <= MAX_FAMILY:
        _error("work_budget", "Provide one to 128 fixed quantile probabilities.")
    _numeric(probabilities, "probabilities")
    if any(not 1e-6 <= p <= 1-1e-6 for p in probabilities):
        _error("invalid_probability", "Use fixed probabilities in [1e-6, 1-1e-6].")
    result = [float(p) for p in probabilities]
    if any(not 1e-6 <= p <= 1-1e-6 for p in result) or len(set(result)) != k:
        _error("invalid_probability", "Use distinct fixed probabilities in [1e-6, 1-1e-6].")
    return result


@resident_cpu
@torch.no_grad()
def simultaneous_quantile_ci(data, columns, probabilities, *, sampling_model, alpha=.05, missing="raise"):
    """Bonferroni exact-binomial order-statistic intervals for lower quantiles.

    The target is inf{x:F(x)>=p}. Discrete observations/ties conserve coverage.
    Missing endpoints use None plus unbounded flags, never a finite surrogate.
    """
    alpha = _alpha(alpha)
    _model(sampling_model, "iid_marginals")
    probabilities = _probabilities(probabilities)
    if not hasattr(columns, "__len__") or len(columns)*len(probabilities) > MAX_FAMILY:
        _error("work_budget", "The fixed column-by-probability family is limited to 128 members.")
    x, sample, meta = _sample(data, columns, missing)
    n, m = x.shape
    size = m*len(probabilities)
    tail = alpha/(2*size)
    # A conservative 64*n*eps allowance only widens a borderline rank.
    guard = 64*(n+1)*torch.finfo(torch.float64).eps
    rows = []
    sorted_x = x.sort(dim=0).values
    for p in probabilities:
        pmf = _binomial_pmf(n, p)
        lower = torch.cat((torch.zeros(1, dtype=torch.float64), pmf.cumsum(0)))
        upper = torch.cat((pmf.flip(0).cumsum(0).flip(0), torch.zeros(1, dtype=torch.float64)))
        r = max(i for i in range(n+1) if i == 0 or float(lower[i])+guard <= tail)
        s = min(i for i in range(1, n+2) if i == n+1 or float(upper[i])+guard <= tail)
        for j, c in enumerate(meta["columns"]):
            rows.append([c, p, float(sorted_x[math.ceil(n*p)-1, j]),
                         None if r == 0 else float(sorted_x[r-1, j]),
                         None if s == n+1 else float(sorted_x[s-1, j]),
                         r == 0, s == n+1, r, s, float(lower[r]), float(upper[s]),
                         1-float(lower[r])-float(upper[s])])
    return _output("Simultaneous lower-quantile confidence intervals", {
        "intervals": table(rows, columns=["variable", "probability", "estimate", "ci_low", "ci_high",
            "lower_unbounded", "upper_unbounded", "lower_rank", "upper_rank", "lower_tail_probability",
            "upper_tail_probability", "binomial_marginal_coverage"]), "sample": sample,
    }, alpha, **meta, family_size=size, probabilities=probabilities, sampling_model=sampling_model,
       method="Bonferroni equal-tail binomial order statistics", per_tail_alpha=tail,
       rank_cdf_roundoff_guard=guard, quantile_definition="inf{x:F(x)>=p}",
       unbounded_representation="JSON null endpoints with explicit flags; ranks 0 and n+1",
       assumptions="Prespecified iid marginal lower-quantile targets; arbitrary within-row and cross-quantile dependence; ties conservative")


def _counts(values, name, *, minimum=0):
    (m,) = _shape(values, name, 1)
    if not 1 <= m <= MAX_FAMILY:
        _error("work_budget", "Count families are limited to 128 members.")
    if isinstance(values, torch.Tensor):
        if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
            _error("invalid_counts", "Counts must have integer types, excluding booleans.")
        values = values.tolist()
    result = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, Integral) or not minimum <= v <= MAX_TRIALS:
            _error("invalid_counts", "Counts must be integers within the declared 1000000-trial budget.")
        result.append(int(v))
    return result


def _family_labels(labels, n):
    if labels is not None:
        if isinstance(labels, (str, bytes, dict)) or not hasattr(labels, "__len__") or len(labels) != n:
            _error("invalid_labels", "Provide one fixed bounded scalar identity per count family member.")
        for label in labels:
            if isinstance(label, bool) or not isinstance(label, (str, Integral, Real)):
                _error("invalid_labels", "Labels must be nonempty strings or finite real scalars.")
            if isinstance(label, str) and (not label or len(label) > 256):
                _error("invalid_labels", "String identities must contain 1..256 characters.")
            if isinstance(label, Real):
                try:
                    valid = math.isfinite(float(label))
                except (ValueError, OverflowError):
                    valid = False
                if not valid:
                    _error("invalid_labels", "Numeric family identities must be finite.")
    return _labels(labels,n)


def _cp(k, n, alpha):
    low, high = clopper_pearson(k, n, alpha)
    if k:
        # F(a,b) lower quantile = reciprocal upper quantile F(b,a).
        # Passing 1-alpha/2 into an upper-tail routine loses tiny-tail bits.
        reciprocal = f_isf(alpha/2, 2*(n-k+1), 2*k)
        low = k/((n-k+1)*reciprocal+k)
    low = max(0., math.nextafter(low, -math.inf))
    high = min(1., math.nextafter(high, math.inf))
    if not 0 <= low <= high <= 1 or not low <= k/n <= high:
        _error("nonfinite_result", "The native exact-binomial confidence limits are unresolved.")
    return low, high


@resident_cpu
@torch.no_grad()
def simultaneous_proportion_ci(successes, trials, *, sampling_model, labels=None, alpha=.05):
    """Fixed binomial marginals, arbitrary dependence, Bonferroni CP intervals."""
    alpha = _alpha(alpha)
    _model(sampling_model, "binomial_marginals")
    successes = _counts(successes, "successes")
    trials = _counts(trials, "trials", minimum=1)
    if len(successes) != len(trials) or any(k > n for k, n in zip(successes, trials)):
        _error("invalid_counts", "Supply one valid successes/trials pair per fixed family member.")
    m = len(successes)
    names = _family_labels(labels, m)
    plan = plan_workspace("simultaneous binomial limits", {"counts, limits and native scalar inversion": 8192+1024*m}).record()
    rows = [[label, k, n, k/n, *_cp(k, n, alpha/m)] for label, k, n in zip(names, successes, trials)]
    return _output("Simultaneous exact-binomial proportion intervals", {
        "intervals": table(rows, columns=["hypothesis", "successes", "trials", "estimate", "ci_low", "ci_high"]),
    }, alpha, family_size=m, family_members=names, sampling_model=sampling_model, resource_plan=plan,
       method="Bonferroni Clopper-Pearson", per_member_alpha=alpha/m, per_tail_alpha=alpha/(2*m),
       successes=successes, trials=trials,
       assumptions="Fixed binomial marginal counts; arbitrary dependence across family members; no outcome-selected family")


@resident_cpu
@torch.no_grad()
def multinomial_region(counts, *, sampling_model, labels=None, alpha=.05):
    """Bonferroni CP box intersected with the complete probability simplex.

    Prespecified exhaustive categories include every zero-count category.
    Coordinate projections are exact for this box/simplex geometry, not a
    declaration that every combination of projected coordinates is feasible.
    """
    alpha = _alpha(alpha)
    _model(sampling_model, "iid_multinomial")
    counts = _counts(counts, "counts")
    n, m = sum(counts), len(counts)
    if not 1 <= n <= MAX_TRIALS:
        _error("invalid_counts", "The complete category total must be positive and at most 1000000.")
    names = _family_labels(labels, m)
    plan = plan_workspace("multinomial simplex region", {"counts, raw limits and projected limits": 8192+2048*m}).record()
    raw = [_cp(k, n, alpha/m) for k in counts]
    lows, highs = zip(*raw)
    if math.fsum(lows) > 1 or math.fsum(highs) < 1:
        _error("nonfinite_result", "The binomial box has no probability-simplex intersection.")
    rows = []
    for j, (name, k, low, high) in enumerate(zip(names, counts, lows, highs)):
        projected_low = max(low, 1-math.fsum(highs[i] for i in range(m) if i != j))
        projected_high = min(high, 1-math.fsum(lows[i] for i in range(m) if i != j))
        if not 0 <= projected_low <= projected_high <= 1:
            _error("nonfinite_result", "The simplex coordinate projection is unresolved.")
        rows.append([name, k, k/n, low, high, projected_low, projected_high])
    return _output("Simultaneous multinomial probability-simplex region", {
        "region": table(rows, columns=["category", "count", "estimate", "raw_ci_low", "raw_ci_high", "ci_low", "ci_high"]),
    }, alpha, family_size=m, family_members=names, n=n, counts=counts,
       sampling_model=sampling_model, resource_plan=plan, per_member_alpha=alpha/m, per_tail_alpha=alpha/(2*m),
       method="Bonferroni Clopper-Pearson box intersect probability simplex",
       constraint="sum(category_probabilities)=1; every raw lower <= probability <= raw upper",
       projection_interpretation="Exact separate coordinate projections; their Cartesian box includes infeasible combinations",
       assumptions="Independent identically distributed categorical rows; fixed exhaustive category universe includes zero counts")
