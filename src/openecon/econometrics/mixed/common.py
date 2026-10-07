"""Bookkeeping shared by the mixed-effects and GEE estimators.

* Group structure: dense codes of one or two nested grouping columns, the map
  from lowest-level groups to their top-level group, Stata's group-size
  summary (min / average / max observations per group, frequency weighted).
* Variance parameters are estimated on a log (standard deviation) scale and
  reported on the variance scale, as Stata's ``mixed`` and ``me`` commands do:
  the standard error of a variance v is the delta-method ``se(ln v) * v`` and
  its confidence interval is ``exp(ln v -/+ z se(ln v))`` (asymmetric, never
  below zero); covariances get the symmetric Wald interval.
* Likelihood-ratio tests of variance components lie on the boundary of the
  parameter space: one component gives ``chibar2(01)`` (the chi2(1) tail
  halved); several give the conservative chi2 with that many degrees of
  freedom, as Stata notes ("LR test is conservative").
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.engines.covariance import group_sums
from openecon.engines.distributions import chi2_sf
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle


def parent_codes(child: Tensor, parent: Tensor, n_child: int) -> Tensor:
    """Parent code of every child group (the child groups must nest in the parents)."""
    out = torch.zeros(n_child, dtype=torch.int64)
    return out.scatter_(0, child, parent)


def nested(inner: Tensor, outer: Tensor, n_inner: int) -> bool:
    """Whether every group of ``inner`` lies inside one group of ``outer``."""
    values = outer.to(torch.float64)
    high = torch.full((n_inner,), -math.inf, dtype=torch.float64).scatter_reduce(
        0, inner, values, "amax")
    low = torch.full((n_inner,), math.inf, dtype=torch.float64).scatter_reduce(
        0, inner, values, "amin")
    return bool((high == low).all())


def group_sizes(codes: Tensor, count: int, frequency: Tensor | None = None) -> dict[str, float]:
    """Observations per group (frequency weighted): min, average and max."""
    weights = torch.ones(len(codes), dtype=torch.float64) if frequency is None else frequency
    sizes = group_sums(weights, codes, count)
    return {"n_groups": count, "size_min": float(sizes.min()),
            "size_avg": float(sizes.sum()) / count, "size_max": float(sizes.max())}


def grouping_codes(frame: ModelFrame, groups: Sequence[str], *,
                   minimum: int = 2) -> list[tuple[Tensor, int]]:
    """Dense codes per level, top first; lower levels are nested in the upper ones.

    A lower-level label is interpreted within its upper group (class 1 of
    school A and class 1 of school B are different classes), as Stata's
    ``|| school: || class:`` does. Estimation needs ``minimum`` = 2 groups per
    level; predictions accept a single group.
    """
    levels = []
    for depth in range(1, len(groups) + 1):
        codes, count = frame.codes(list(groups[:depth]))
        if count < minimum:
            raise AnalysisError("insufficient_groups", f"The grouping '{groups[depth - 1]}' has "
                                "fewer than two groups; a random effect needs several groups.")
        levels.append((codes, count))
    return levels


def level_labels(frame: ModelFrame, groups: Sequence[str], codes: Tensor,
                 count: int) -> list[Any]:
    """Label of every group of a level in code order (the lowest column's value)."""
    first = torch.full((count,), len(codes), dtype=torch.int64).scatter_reduce(
        0, codes, torch.arange(len(codes)), "amin")
    column = frame.series(groups[-1])
    from openecon.analysis import _json_scalar

    return [_json_scalar(value) for value in column.iloc[first.numpy()].tolist()]


def check_cluster_nesting(frame: ModelFrame, top: Tensor, n_top: int, group: str) -> list[
        tuple[Tensor, int]]:
    """Cluster codes per top-level group; the clusters must contain whole groups."""
    from openecon.econometrics import registry

    clusters = []
    for name, (codes, _) in zip(registry.cluster_columns(frame.spec),
                                frame.cluster_dimensions(), strict=True):
        if not nested(top, codes, n_top):
            raise AnalysisError(
                "cluster_not_nested",
                f"The groups of '{group}' must be nested within the clusters of '{name}' "
                "(Stata's requirement for the highest level).")
        group_cluster = parent_codes(top, codes, n_top)
        unique, inverse = torch.unique(group_cluster, return_inverse=True)
        clusters.append((inverse, len(unique)))
    return clusters


def boundary_lr(statistic: float, components: int, against: str) -> dict[str, Any]:
    """LR test of the variance components (chibar2(01) for one, conservative chi2 otherwise)."""
    statistic = max(0.0, statistic)
    if components == 1:
        return {"statistic": statistic, "df": 1, "p_value": 0.5 * chi2_sf(statistic, 1),
                "distribution": "chibar2",
                "label": f"LR test vs. {against} (chibar2(01): chi2(1) tail halved)"}
    return {"statistic": statistic, "df": components, "p_value": chi2_sf(statistic, components),
            "distribution": "chi2",
            "label": f"LR test vs. {against} (conservative: variance parameters on the boundary)",
            "note": "LR test is conservative and provided only for reference."}


# exp() of a log-scale half-width above this overflows float64: the variance is not
# estimated with any precision (a flat likelihood).
_MAX_LOG_SPREAD = 700.0


def _log_interval(term: Any, spread: float) -> None:
    if not math.isfinite(spread) or spread > _MAX_LOG_SPREAD:
        raise AnalysisError(
            "boundary_solution", f"The standard error of {term.term} is so large that its "
                                 "confidence interval is not finite: the likelihood is flat in "
                                 "that variance. Simplify the random-effects specification.")
    term.ci_low = term.estimate * math.exp(-spread)
    term.ci_high = term.estimate * math.exp(spread)


def variance_intervals(bundle: ResultBundle, positions: Sequence[int], alpha: float) -> None:
    """Replace the Wald interval of variance terms by exp(ln v -/+ z se/v) (in place)."""
    critical = critical_value(alpha, None)
    for position in positions:
        term = bundle.coefficients[position]
        _log_interval(term, critical * term.std_error / term.estimate)


def log_intervals(bundle: ResultBundle, positions: Sequence[int], log_se: Sequence[float],
                  alpha: float) -> None:
    """Interval exp(ln v -/+ z se_ln) for positive terms whose ln-scale se is known."""
    critical = critical_value(alpha, None)
    for position, se in zip(positions, log_se, strict=True):
        _log_interval(bundle.coefficients[position], critical * se)


def require_independent_effects(z: Tensor, names: Sequence[str], weights: Tensor | None,
                                command: str) -> None:
    """Refuse random-effects columns that are collinear (a constant random slope duplicates
    the random intercept): their variances are not separately identified."""
    from openecon.econometrics.core import kernel_call
    from openecon.engines.linalg import collinear_columns

    names = list(names)
    if "_cons" in names and len(names) > 1:
        # With the random intercept, screen the mean-deviated slopes (a slope with a large
        # level, such as calendar time, is not mistaken for the intercept).
        slopes = [i for i, name in enumerate(names) if name != "_cons"]
        block = z[:, slopes]
        w = torch.ones(len(z), dtype=torch.float64) if weights is None else weights
        block = block - (w[:, None] * block).sum(dim=0) / w.sum()
        _, dropped = kernel_call(collinear_columns, block, weights)
        omitted = [slopes[i] for i in dropped]
    else:
        _, omitted = kernel_call(collinear_columns, z, weights)
    if omitted:
        listed = ", ".join(names[i] for i in omitted)
        raise AnalysisError("collinear_random_effects", f"{command}: the random-effects column(s) "
                            f"{listed} are collinear with the other random effects (a constant "
                            "random slope duplicates the random intercept); remove them.")


def require_variation(y: Tensor, command: str, name: str) -> None:
    """Refuse a constant outcome (an all-one or all-zero binary outcome, all-zero counts)."""
    if bool((y == y[0]).all()):
        raise AnalysisError("constant_outcome", f"{command}: the outcome '{name}' does not vary "
                            f"(every observation equals {float(y[0]):g}), so the model is not "
                            "identified; check the estimation sample.")


def optimizer_summary(result: Any) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {"method": result.method, "iterations": result.iterations,
            "converged": result.converged, "gradient_max": diagnostics.get("gradient_max"),
            "message": diagnostics.get("message")}
