"""Native multiple-testing and explicit joint-null simultaneous inference.

No fitting, resampling design or independence assumption is inferred from a
table of p-values. Null draws must be generated under a documented null/design.
"""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table

METHODS = ("bonferroni", "sidak", "holm", "holm_sidak", "hochberg", "hommel", "bh", "by")
MAX_FAMILY = 100000
MAX_HOMMEL = 5000
MAX_CELLS = 8000000
MAX_COMPARISONS = 100000000
_ASSUMPTIONS = {
    "bonferroni": "FWER under arbitrary dependence",
    "holm": "FWER under arbitrary dependence",
    "sidak": "FWER under independence (or an applicable Sidak inequality)",
    "holm_sidak": "FWER under independence (or an applicable Sidak inequality)",
    "hochberg": "FWER under independence or appropriate positive dependence",
    "hommel": "FWER under the Simes inequality; independence or appropriate positive dependence",
    "bh": "FDR under independence or PRDS dependence",
    "by": "FDR under arbitrary dependence",
}


def _level(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_alpha", "alpha must be between zero and one.")


def _shape(values, name, dimensions):
    """Inspect bounded resident inputs before any list/pandas/tensor copy."""
    if isinstance(values, (str, bytes, dict)) or not hasattr(values, "__len__"):
        raise AnalysisError("invalid_values", f"{name} must be a bounded numeric sequence.")
    if isinstance(values, torch.Tensor) and values.device.type != "cpu":
        raise AnalysisError(
            "unsupported_device", "These resident procedures accept CPU inputs only."
        )
    shape = getattr(values, "shape", None)
    if shape is not None:
        if len(shape) != dimensions:
            raise AnalysisError("invalid_values", f"{name} must have {dimensions} dimensions.")
        return tuple(shape)
    try:
        n = len(values)
        if dimensions == 1:
            return (n,)
        if not n or not hasattr(values[0], "__len__") or isinstance(values[0], (str, bytes)):
            raise ValueError()
        return n, len(values[0])
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        raise AnalysisError(
            "invalid_values", f"{name} must be a rectangular numeric matrix."
        ) from exc


def _numeric(values, name, *, missing=False):
    if isinstance(values, torch.Tensor):
        if values.dtype == torch.bool or values.is_complex():
            raise AnalysisError(
                "invalid_values", f"{name} must contain real numbers, not booleans."
            )
        return
    for value in values:
        if value is None or value is pd.NA:
            if missing:
                continue
            raise AnalysisError("missing_values", f"{name} must have no missing values.")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise AnalysisError(
                "invalid_values", f"{name} must contain real numbers, not strings or booleans."
            )


def _vector(values, name, *, missing=False, maximum=MAX_FAMILY):
    (n,) = _shape(values, name, 1)
    if n > maximum:
        raise AnalysisError("work_budget", f"{name} exceeds its declared resident input budget.")
    if not n:
        raise AnalysisError("invalid_values", f"{name} must be nonempty.")
    _numeric(values, name, missing=missing)
    try:
        if isinstance(values, torch.Tensor):
            value = values.to(dtype=torch.float64)
        else:
            series = pd.Series(values)
            value = torch.as_tensor(
                series.to_numpy(dtype="float64", na_value=math.nan), dtype=torch.float64
            )
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError(
            "invalid_values", f"{name} must be a one-dimensional numeric sequence."
        ) from exc
    if value.ndim != 1 or value.numel() == 0 or torch.isinf(value).any():
        raise AnalysisError("invalid_values", f"{name} must be nonempty with no infinite values.")
    if not missing and not torch.isfinite(value).all():
        raise AnalysisError("missing_values", f"{name} must have no missing values.")
    return value


def _matrix(values, name, *, columns, rows=None, step_work=False):
    r, c = _shape(values, name, 2)
    if c != columns or (rows is not None and r != rows) or (rows is None and r < 2):
        raise AnalysisError("invalid_values", f"{name} has the wrong dimensions.")
    if r * c > MAX_CELLS or (step_work and r * c * c > MAX_COMPARISONS):
        raise AnalysisError("work_budget", f"{name} exceeds the declared resident work budget.")
    if isinstance(values, torch.Tensor):
        _numeric(values, name)
    else:
        # pandas iteration yields column labels, not matrix entries.
        content = (
            values.itertuples(index=False, name=None)
            if isinstance(values, pd.DataFrame)
            else values
        )
        for row in content:
            if isinstance(row, (str, bytes)) or not hasattr(row, "__len__") or len(row) != c:
                raise AnalysisError("invalid_values", f"{name} must be rectangular.")
            _numeric(row, name)
    try:
        supplied = values.to_numpy() if isinstance(values, pd.DataFrame) else values
        matrix = torch.as_tensor(supplied, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_values", f"{name} must be a real numeric matrix.") from exc
    if not torch.isfinite(matrix).all():
        raise AnalysisError("invalid_values", f"{name} must contain finite values.")
    return matrix


def _labels(labels, n):
    if labels is not None and (
        isinstance(labels, (str, bytes, dict)) or not hasattr(labels, "__len__") or len(labels) != n
    ):
        raise AnalysisError(
            "invalid_labels", "Supply one unique, nonmissing scalar label per hypothesis."
        )
    try:
        names = list(range(n)) if labels is None else list(labels)
        invalid = any(not pd.api.types.is_scalar(x) or pd.isna(x) for x in names)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            "invalid_labels", "Supply one unique, nonmissing scalar label per hypothesis."
        ) from exc
    if len(names) != n or len({str(x) for x in names}) != n or invalid:
        raise AnalysisError("invalid_labels", "Supply one unique, nonmissing label per hypothesis.")
    return names


def _family_metadata(names, order, groups, *, tested=None, rule="adjusted_p_value <= alpha"):
    return {
        "family_members": names,
        "tested_positions": list(range(len(names))) if tested is None else tested,
        "ordering_positions": order,
        "tie_groups": groups,
        "rejection_rule": rule,
        "device": "cpu",
        "resident": True,
        "sample_domain": "Caller-supplied compatible summaries; no observations are fitted",
        "weights": "Not accepted; weighting belongs to the supplied test/covariance design",
        "degrees_of_freedom": None,
        "convergence": "Not applicable to these finite procedures",
        "dataset_support": False,
    }


def _ties(sorted_values, positions):
    groups = []
    for value, position in zip(sorted_values, positions):
        if groups and value == groups[-1][0]:
            groups[-1][1].append(position)
        else:
            groups.append((value, [position]))
    return [indices for _, indices in groups]


def _reverse_min(value):
    return torch.cummin(value.flip(0), 0).values.flip(0)


@torch.no_grad()
def multipletests(
    p_values, *, method="holm", alpha=0.05, labels=None, missing="raise"
) -> pd.DataFrame:
    """Adjust a declared p-value family, preserving the supplied hypothesis order.

    Methods: Bonferroni, Sidak, Holm, Holm-Sidak, Hochberg, Hommel, BH and BY.
    ``missing='drop'`` explicitly excludes missing tests from the tested family;
    their original rows remain present with missing adjusted values/rejections.
    No estimators are fitted. See docs/econometrics/multiple-testing.md.
    """
    _level(alpha)
    if method not in METHODS or missing not in ("raise", "drop"):
        raise AnalysisError(
            "invalid_method", f"Choose method from {METHODS} and missing='raise' or 'drop'."
        )
    raw = _vector(
        p_values,
        "p_values",
        missing=missing == "drop",
        maximum=MAX_HOMMEL if method == "hommel" else MAX_FAMILY,
    )
    valid = torch.isfinite(raw)
    if not valid.any() or ((raw[valid] < 0) | (raw[valid] > 1)).any():
        raise AnalysisError(
            "invalid_p_values", "The tested family needs p-values between zero and one."
        )
    n = int(valid.sum())
    names = _labels(labels, len(raw))
    p, order = torch.sort(raw[valid], stable=True)
    rank = torch.arange(1, n + 1, dtype=torch.float64)
    remaining = n + 1 - rank
    if method == "bonferroni":
        adjusted = n * p
    elif method == "sidak":
        adjusted = -torch.expm1(n * torch.log1p(-p))
    elif method == "holm":
        adjusted = torch.cummax(remaining * p, 0).values
    elif method == "holm_sidak":
        adjusted = torch.cummax(-torch.expm1(remaining * torch.log1p(-p)), 0).values
    elif method == "hochberg":
        adjusted = _reverse_min(remaining * p)
    elif method == "hommel":
        adjusted = p.clone()
        for size in range(n, 1, -1):
            pivot = torch.min(size * p[-size:] / torch.arange(1, size + 1, dtype=torch.float64))
            adjusted[-size:] = torch.maximum(adjusted[-size:], pivot)
            adjusted[:-size] = torch.maximum(
                adjusted[:-size], torch.minimum(size * p[:-size], pivot)
            )
    else:
        multiplier = float((1 / rank).sum()) if method == "by" else 1.0
        adjusted = _reverse_min(multiplier * n * p / rank)
    restored = torch.empty_like(adjusted)
    restored[order] = adjusted.clamp(0, 1)
    full = torch.full_like(raw, math.nan)
    full[valid] = restored
    rejects = [bool(v <= alpha) if math.isfinite(v) else None for v in full.tolist()]
    tested = torch.nonzero(valid).flatten().tolist()
    ordered_positions = [tested[i] for i in order.tolist()]
    return table(
        {
            "hypothesis": names,
            "p_value": raw.tolist(),
            "adjusted_p_value": full.tolist(),
            "reject": rejects,
        },
        title="Multiple-testing adjustment",
        method=method,
        alpha=alpha,
        family_size=len(raw),
        tested_family_size=n,
        excluded_missing=len(raw) - n,
        assumption=_ASSUMPTIONS[method],
        missing_policy=missing,
        excluded_positions=torch.nonzero(~valid).flatten().tolist(),
        **_family_metadata(
            names, ordered_positions, _ties(p.tolist(), ordered_positions), tested=tested
        ),
        precision="float64",
        stata_parity_validated=False,
    )


@torch.no_grad()
def stepdown(
    observed,
    null_draws,
    *,
    method="romano_wolf",
    tail="two-sided",
    calibration="monte_carlo",
    alpha=0.05,
    labels=None,
    null_description=None,
) -> pd.DataFrame:
    """Dependence-aware stepdown from supplied joint null draws.

    Romano-Wolf uses maxT of comparable studentized statistics. Westfall-Young
    uses supplied marginal p-values and their joint-null p-value draws (minP),
    requiring subset pivotality. Enumerated calibration requires all equally
    likely null assignments; Monte Carlo uses the finite-draw plus-one rule.
    Tied observed tests are removed together to preserve label invariance.
    """
    _level(alpha)
    if method not in ("romano_wolf", "westfall_young") or tail not in (
        "two-sided",
        "greater",
        "less",
    ):
        raise AnalysisError(
            "invalid_method", "Choose romano_wolf/maxT or westfall_young/minP and a valid tail."
        )
    if (
        calibration not in ("monte_carlo", "enumerated")
        or not isinstance(null_description, str)
        or not null_description.strip()
    ):
        raise AnalysisError(
            "missing_null_design", "Describe the null-generating design and choose its calibration."
        )
    observed = _vector(observed, "observed")
    supplied = observed.clone()
    draws = _matrix(null_draws, "null_draws", columns=len(observed), step_work=True)
    names = _labels(labels, len(observed))
    min_p = method == "westfall_young"
    if min_p:
        if (
            tail != "two-sided"
            or ((observed < 0) | (observed > 1)).any()
            or ((draws < 0) | (draws > 1)).any()
        ):
            raise AnalysisError(
                "invalid_p_values",
                "Westfall-Young takes marginal p-values in [0,1]; tail must be two-sided.",
            )
    elif tail == "two-sided":
        observed, draws = observed.abs(), draws.abs()
    elif tail == "less":
        observed, draws = -observed, -draws
    order = torch.argsort(observed, descending=not min_p, stable=True)
    adjusted = torch.empty_like(observed)
    b = len(draws)
    correction = int(calibration == "monte_carlo")
    previous, start = 0.0, 0
    while start < len(observed):
        end = start + 1
        while end < len(observed) and observed[order[end]] == observed[order[start]]:
            end += 1
        remaining = draws[:, order[start:]]
        maximum = remaining.amin(1) if min_p else remaining.amax(1)
        cutoff = observed[order[start]]
        count = (maximum <= cutoff).sum() if min_p else (maximum >= cutoff).sum()
        value = max(previous, (float(count) + correction) / (b + correction))
        adjusted[order[start:end]] = value
        previous, start = value, end
    raw_count = (draws <= observed).sum(0) if min_p else (draws >= observed).sum(0)
    calibrated = (raw_count.to(torch.float64) + correction) / (b + correction)
    raw = supplied if min_p else calibrated
    ordered_positions = order.tolist()
    return table(
        {
            "hypothesis": names,
            "statistic": supplied.tolist(),
            "ordering_statistic": observed.tolist(),
            "p_value": raw.tolist(),
            "calibrated_marginal_p_value": calibrated.tolist(),
            "adjusted_p_value": adjusted.tolist(),
            "reject": (adjusted <= alpha).tolist(),
        },
        title="Joint-null stepdown inference",
        method=method,
        alpha=alpha,
        tail=tail,
        family_size=len(observed),
        draws=b,
        calibration=calibration,
        null_description=null_description.strip(),
        precision="float64",
        assumptions="Valid joint studentized subset-null approximation with monotone critical values"
        if not min_p
        else "Subset pivotality and valid joint-null marginal p-values",
        statistic_kind="supplied marginal p-value" if min_p else "comparable studentized statistic",
        null_design_verified=False,
        missing_policy="raise",
        p_value_source="Supplied marginal p-values"
        if min_p
        else "Supplied joint-null marginal tail counts",
        **_family_metadata(
            names, ordered_positions, _ties(observed[order].tolist(), ordered_positions)
        ),
        stata_parity_validated=False,
    )


@torch.no_grad()
def simultaneous_ci(
    estimates,
    covariance,
    *,
    labels=None,
    alpha=0.05,
    draws=50000,
    seed=1729,
    family_description=None,
) -> pd.DataFrame:
    """Asymptotic MVN max-|z| simultaneous intervals with a local RNG.

    Accept the full joint covariance of the compatible estimate vector. These
    are normal-limit intervals, not small-sample t or arbitrary-model bootstrap
    intervals. Monte Carlo critical-value precision is recorded explicitly.
    """
    _level(alpha)
    estimates = _vector(estimates, "estimates", maximum=384)
    n = len(estimates)
    if not isinstance(family_description, str) or not family_description.strip():
        raise AnalysisError(
            "missing_test_family", "Describe the common family/estimand and its covariance source."
        )
    if isinstance(draws, bool) or not isinstance(draws, int) or not 1000 <= draws <= 200000:
        raise AnalysisError("invalid_draws", "draws must be an integer between 1000 and 200000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise AnalysisError("invalid_seed", "seed must be a nonnegative signed 64-bit integer.")
    if n > 384 or n * draws > 8000000:
        raise AnalysisError(
            "work_budget", "The joint interval simulation exceeds the declared budget."
        )
    names = _labels(labels, n)
    cov = _matrix(covariance, "covariance", columns=n, rows=n)
    if (cov.diag() <= 0).any():
        raise AnalysisError("invalid_covariance", "Each estimate needs strictly positive variance.")
    se = cov.diag().sqrt()
    # Dividing twice avoids overflow/underflow in products of marginal SEs.
    corr = cov / se[:, None] / se[None, :]
    if not torch.isfinite(corr).all() or not torch.allclose(corr, corr.T, atol=1e-12, rtol=1e-10):
        raise AnalysisError(
            "invalid_covariance", "The joint covariance must be symmetric in marginal-SE units."
        )
    corr = corr * 0.5 + corr.T * 0.5
    simulation_order = (
        sorted(range(n), key=lambda i: str(names[i])) if labels is not None else list(range(n))
    )
    canonical_corr = corr[simulation_order][:, simulation_order]
    eig, vectors = torch.linalg.eigh(canonical_corr)
    if eig.min() < -1e-12:
        raise AnalysisError(
            "invalid_covariance", "The joint covariance must be positive semidefinite."
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(draws, n, dtype=torch.float64, generator=generator)
    z = noise @ (vectors * eig.clamp_min(0).sqrt()).T
    # Empirical inverse CDF; higher interpolation keeps the finite-grid target.
    critical = float(torch.quantile(z.abs().amax(1), 1 - alpha, interpolation="higher"))
    low, high = estimates - critical * se, estimates + critical * se
    if (
        not math.isfinite(critical)
        or not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
    ):
        raise AnalysisError(
            "nonfinite_result", "The joint normal simulation produced nonfinite intervals."
        )
    return table(
        {
            "hypothesis": names,
            "estimate": estimates.tolist(),
            "std_error": se.tolist(),
            "ci_low": (estimates - critical * se).tolist(),
            "ci_high": high.tolist(),
        },
        title="Normal-limit simultaneous confidence intervals",
        alpha=alpha,
        family_size=n,
        critical_value=critical,
        distribution="Joint asymptotic normal max-|z|",
        draws=draws,
        seed=seed,
        family_description=family_description.strip(),
        joint_covariance=cov.tolist(),
        joint_correlation=corr.tolist(),
        simulation_order_positions=simulation_order,
        simulation_hypotheses=[names[i] for i in simulation_order],
        seed_order_policy="Canonical unique explicit labels"
        if labels is not None
        else "Caller order; unlabeled permutation equivalence is Monte Carlo approximate",
        smallest_correlation_eigenvalue=float(eig.min()),
        psd_tolerance=1e-12,
        clipped_eigenvalues=int((eig < 0).sum()),
        inference_target="Joint asymptotic normal compatible estimate family",
        missing_policy="raise",
        **_family_metadata(
            names,
            list(range(n)),
            [],
            rule="Compatible true estimates lie within all simultaneous intervals",
        ),
        cdf_monte_carlo_std_error=math.sqrt(alpha * (1 - alpha) / draws),
        precision="float64",
        stata_parity_validated=False,
    )
