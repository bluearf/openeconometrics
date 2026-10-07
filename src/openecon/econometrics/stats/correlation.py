"""Correlation matrices and partial correlations: ``oe.correlate`` and ``oe.pcorr``.

Pearson's r, Spearman's rho (Pearson's r of midranks) and Kendall's tau-b with
pairwise or listwise deletion (SPSS CORRELATIONS / NONPAR CORR, Stata correlate,
pwcorr, spearman, ktau), and partial / semipartial correlations from one
regression (Stata pcorr, SPSS PARTIAL CORR).

Kendall's tau-b is computed in O(n log n): the pairs are sorted by x (ties by y)
and the discordant pairs are the inversions of the y ranks, counted by a bottom-up
merge sort in which every level is one vectorized sort (Knight 1966).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.econometrics.stats import common as c
from openecon.engines import distributions as dist
from openecon.engines import linalg

# Variance of Fisher's z: 1/(n-3) for Pearson; Fieller, Hartley and Pearson (1957)
# for the rank correlations: 1.06/(n-3) (Spearman) and 0.437/(n-4) (Kendall).
_Z_VARIANCE = {"pearson": (1.0, 3), "spearman": (1.06, 3), "kendall": (0.437, 4)}


def midranks(x: Tensor) -> Tensor:
    """Ranks 1..n with ties replaced by the mean of the ranks they span."""
    order = torch.argsort(x, stable=True)
    _, inverse, counts = torch.unique_consecutive(x[order], return_inverse=True,
                                                  return_counts=True)
    counts = counts.to(torch.float64)
    average = torch.cumsum(counts, 0) - 0.5 * (counts - 1.0)
    ranks = torch.empty_like(x)
    ranks[order] = average[inverse]
    return ranks


def inversions(values: Tensor) -> int:
    """Number of pairs i < j with values[i] > values[j] (int64 input), in O(n log^2 n)."""
    n = values.numel()
    if n < 2:
        return 0
    span = 2 * (int(values.max()) + 1)
    index = torch.arange(n)
    total, width = 0, 1
    while width < n:
        pair = index // (2 * width)
        right = (index // width) % 2
        # Left elements sort before equal right elements, so ties are not inversions.
        order = torch.argsort(pair * span + 2 * values + right)
        moved = right[order]
        rights_before = torch.cumsum(moved, 0) - moved - pair * width
        left_not_greater = index - 2 * width * pair - rights_before
        left_size = torch.clamp(n - 2 * width * pair, max=width)
        total += int(((left_size - left_not_greater) * moved).sum())
        values = values[order]
        width *= 2
    return total


def _tie_terms(counts: Tensor) -> tuple[float, float, float]:
    t = counts.to(torch.float64)
    return (float((t * (t - 1.0)).sum()), float((t * (t - 1.0) * (t - 2.0)).sum()),
            float((t * (t - 1.0) * (2.0 * t + 5.0)).sum()))


def kendall_tau_b(x: Tensor, y: Tensor) -> tuple[float | None, float | None]:
    """(tau-b, z) with the tie-corrected variance of S = concordant - discordant.

    tau_b = S / sqrt((n0 - n1)(n0 - n2)), n0 = n(n-1)/2, n1 and n2 the numbers of
    pairs tied on x and on y. Under independence
    var(S) = [n(n-1)(2n+5) - sum t(t-1)(2t+5) - sum u(u-1)(2u+5)] / 18
           + sum t(t-1) sum u(u-1) / (2n(n-1))
           + sum t(t-1)(t-2) sum u(u-1)(u-2) / (9n(n-1)(n-2)),
    and z = S / sqrt(var(S)) (no continuity correction).
    """
    n = x.shape[0]
    if n < 2:
        return None, None
    order = torch.argsort(y, stable=True)
    order = order[torch.argsort(x[order], stable=True)]
    xs, ys = x[order], y[order]
    _, x_counts = torch.unique_consecutive(xs, return_counts=True)
    _, y_rank, y_counts = torch.unique(ys, return_inverse=True, return_counts=True)
    new_pair = torch.ones(n, dtype=torch.bool)
    new_pair[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
    starts = torch.nonzero(new_pair).reshape(-1)
    joint_counts = torch.diff(starts, append=torch.tensor([n]))
    t1, t2, t3 = _tie_terms(x_counts)
    u1, u2, u3 = _tie_terms(y_counts)
    pairs = 0.5 * n * (n - 1.0)
    tied_x, tied_y, tied_both = 0.5 * t1, 0.5 * u1, 0.5 * _tie_terms(joint_counts)[0]
    score = pairs - tied_x - tied_y + tied_both - 2.0 * inversions(y_rank)
    denominator = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    if denominator == 0.0:
        return None, None
    variance = (n * (n - 1.0) * (2.0 * n + 5.0) - t3 - u3) / 18.0 \
        + t1 * u1 / (2.0 * n * (n - 1.0))
    if n > 2:
        variance += t2 * u2 / (9.0 * n * (n - 1.0) * (n - 2.0))
    tau = max(-1.0, min(1.0, score / denominator))
    return tau, (score / math.sqrt(variance) if variance > 0 else None)


def _pearson(a: Tensor, b: Tensor) -> float | None:
    a, b = a - a.mean(), b - b.mean()
    denominator = c.root_product(float(a.square().sum()), float(b.square().sum()))
    if denominator == 0.0:
        return None
    return max(-1.0, min(1.0, float((a * b).sum()) / denominator))


def _pearson_matrix(x: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise-complete Pearson correlations and pair counts from masked cross products."""
    mask = present.to(torch.float64)
    counts = mask.sum(0).clamp_min(1.0)
    centred = torch.where(present, x, torch.zeros_like(x))
    centred = torch.where(present, centred - centred.sum(0) / counts, torch.zeros_like(x))
    n = mask.T @ mask
    sums = centred.T @ mask                      # sum of column i over rows where j is present
    squares = centred.square().T @ mask
    safe = n.clamp_min(1.0)
    cross = centred.T @ centred - sums * sums.T / safe
    spread = (squares - sums.square() / safe).clamp_min(0.0)
    # sqrt(a b) is exact for perfectly related integers; the product of two sums of
    # squares can overflow or underflow, and then sqrt(a) sqrt(b) is used instead.
    product = spread * spread.T
    denominator = torch.where((product > 1e-290) & torch.isfinite(product), product.sqrt(),
                              spread.sqrt() * spread.sqrt().T)
    r = torch.where(denominator > 0, cross / denominator.clamp_min(1e-300),
                    torch.full_like(cross, float("nan")))
    return r.clamp(-1.0, 1.0), n


def _t_p_value(r: float | None, n: int) -> float | None:
    if r is None or n < 3:
        return None
    if abs(r) >= 1.0:
        return 0.0
    return c.t_two_sided(r * math.sqrt((n - 2) / (1.0 - r * r)), n - 2)


def correlate(data: Any, columns: list[str], *, method: str = "pearson", pairwise: bool = True,
              ci: bool = False, alpha: float = 0.05) -> TableSet:
    """Correlation matrix with two-sided p-values and the number of observations.

    Methods

    * ``"pearson"``: r = S_xy / sqrt(S_xx S_yy); t = r sqrt((n-2)/(1-r^2)) with n - 2
      degrees of freedom.
    * ``"spearman"``: Pearson's r of the midranks (ties get the mean of their ranks),
      with the same t approximation (SPSS, Stata ``spearman``).
    * ``"kendall"``: tau-b with the tie-corrected normal approximation,
      z = S / sqrt(var(S)) without continuity correction (SPSS; Stata's ``ktau``
      p-value applies a continuity correction and will differ slightly).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : two or more numeric columns.
    method : ``"pearson"``, ``"spearman"`` or ``"kendall"``.
    pairwise : True uses, for each pair, the rows where both variables are observed
        (SPSS ``/MISSING=PAIRWISE``, Stata ``pwcorr``; ranks are recomputed for each
        pair); False uses only rows complete in all columns (listwise, Stata
        ``correlate``).
    ci : add the ``intervals`` table: Fisher-z confidence limits
        tanh(atanh(r) -+ z_(alpha/2) * se) with se^2 = 1/(n-3) (Pearson), 1.06/(n-3)
        (Spearman) and 0.437/(n-4) (Kendall), the Fieller-Hartley-Pearson variances.
    alpha : 1 - confidence level of the intervals.

    Returns
    -------
    TableSet with square tables ``coefficients``, ``p_values`` (missing on the
    diagonal) and ``n`` indexed by the column names, plus ``intervals`` (var_i, var_j,
    coefficient, ci_low, ci_high, n) when ``ci=True``. A coefficient is missing when
    a variable does not vary in the rows used. ``attrs``: ``method``, ``missing``
    (``"pairwise"`` or ``"listwise"``), ``n`` (rows supplied), ``n_complete``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.1, 3.9, 6.2, 8.1, 9.8],
    ...         "z": [5.0, 3.0, 4.0, 1.0, 2.0]}
    >>> result = oe.correlate(data, ["x", "y", "z"], method="spearman")
    >>> float(result["coefficients"].loc["x", "y"])
    1.0
    """
    names = c.name_list(columns, "columns", minimum=2)
    c.check_choice(method, "method", ("pearson", "spearman", "kendall"))
    c.check_flag(pairwise, "pairwise")
    c.check_flag(ci, "ci")
    alpha = c.check_alpha(alpha)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_correlation import correlate as replay_correlate
        return replay_correlate(data, names, method=method, pairwise=pairwise, ci=ci, alpha=alpha)
    frame, _ = c.select(data, names, numeric=names, listwise=False)
    x = torch.stack([c.values(frame, name, allow_missing=True) for name in names], dim=1)
    rows = x.shape[0]
    present = ~torch.isnan(x)
    complete = present.all(1)
    if not pairwise:
        x, present = x[complete], present[complete]
    if x.shape[0] < 2:
        raise AnalysisError("insufficient_observations", "Correlations need at least two "
                            "complete observations.")
    p = len(names)
    whole = bool(present.all())
    if method == "pearson" or (method == "spearman" and whole):
        base = x if method == "pearson" else torch.stack([midranks(x[:, j]) for j in range(p)], 1)
        matrix, counts = _pearson_matrix(base, present)
        coefficient = [[None if math.isnan(v) else v for v in row] for row in matrix.tolist()]
        n = [[int(v) for v in row] for row in counts.tolist()]
        p_values = [[_t_p_value(coefficient[i][j], n[i][j]) if i != j else None
                     for j in range(p)] for i in range(p)]
    else:
        coefficient = [[None] * p for _ in range(p)]
        p_values = [[None] * p for _ in range(p)]
        n = [[0] * p for _ in range(p)]
        for i in range(p):
            n[i][i] = int(present[:, i].sum())
            for j in range(i + 1, p):
                keep = present[:, i] & present[:, j]
                a, b = x[keep, i], x[keep, j]
                n[i][j] = n[j][i] = count = int(keep.sum())
                if method == "spearman":
                    value = _pearson(midranks(a), midranks(b)) if count > 1 else None
                    p_value = _t_p_value(value, count)
                else:
                    value, z = kendall_tau_b(a, b)
                    p_value = None if z is None else min(1.0, 2.0 * dist.normal_sf(abs(z)))
                coefficient[i][j] = coefficient[j][i] = value
                p_values[i][j] = p_values[j][i] = p_value
    for i in range(p):
        column = x[present[:, i], i]
        n[i][i] = int(column.numel())
        coefficient[i][i] = 1.0 if n[i][i] > 1 and bool((column != column[0]).any()) else None
    tables = {"coefficients": c.frame(coefficient, columns=names, index=names),
              "p_values": c.frame(p_values, columns=names, index=names),
              "n": c.frame(n, columns=names, index=names)}
    if ci:
        factor, offset = _Z_VARIANCE[method]
        bound = dist.normal_isf(0.5 * alpha)
        intervals = []
        for i in range(p):
            for j in range(i + 1, p):
                r, count = coefficient[i][j], n[i][j]
                low = high = None
                if r is not None and count > offset and abs(r) < 1.0:
                    half = bound * math.sqrt(factor / (count - offset))
                    low, high = math.tanh(math.atanh(r) - half), math.tanh(math.atanh(r) + half)
                elif r is not None and count > offset:
                    low = high = r
                intervals.append([names[i], names[j], r, low, high, count])
        tables["intervals"] = c.frame(intervals, columns=["var_i", "var_j", "coefficient",
                                                          "ci_low", "ci_high", "n"])
    return TableSet(tables, title=f"{method.capitalize()} correlations", method=method,
                    missing="pairwise" if pairwise else "listwise", n=rows,
                    n_complete=int(complete.sum()), alpha=alpha,
                    distribution="normal" if method == "kendall" else "t")


def pcorr(data: Any, y: str, x: list[str], *, controls: list[str] | None = None,
          missing: str = "drop") -> Any:
    """Partial and semipartial correlations of ``y`` with each variable in ``x``.

    From the regression of y on a constant, all of ``x`` and all of ``controls``,
    with t_j the t statistic of x_j, df = n - k (k coefficients) and R^2 the fit:

        partial r_j      = t_j / sqrt(t_j^2 + df)
        semipartial r_j  = sign(t_j) sqrt(t_j^2 (1 - R^2) / df)

    The partial correlation relates the parts of y and x_j that are orthogonal to
    the other regressors; the semipartial (part) correlation relates y itself to
    that part of x_j, so its square is the R^2 lost by dropping x_j. The p-value is
    that of t_j (two-sided, df degrees of freedom).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    y : numeric outcome.
    x : variables whose (semi)partial correlations with ``y`` are reported, each
        holding the other ``x`` and all ``controls`` constant (Stata ``pcorr y x1 x2``).
    controls : variables that are only held constant (SPSS ``PARTIAL CORR y x BY
        controls`` is ``pcorr(data, "y", ["x"], controls=[...])``).
    missing : ``"drop"`` (listwise) or ``"raise"``.

    Returns
    -------
    Table indexed by the names in ``x`` with partial_corr, semipartial_corr,
    partial_corr_sq, semipartial_corr_sq, statistic (t), df and p_value;
    ``attrs``: ``n``, ``n_missing``, ``r_squared``, ``controls``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.8], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ...         "w": [0.5, 0.1, 0.9, 0.3, 0.8, 0.2]}
    >>> table = oe.pcorr(data, "y", ["x"], controls=["w"])
    >>> list(table.columns)[:2]
    ['partial_corr', 'semipartial_corr']
    """
    c.check_name(y, "y")
    names = c.name_list(x, "x")
    controls = c.name_list(controls, "controls", minimum=0)
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from .streaming_correlation import pcorr as replay_pcorr
        return replay_pcorr(data, y, names, controls, missing)
    frame, dropped = c.select(data, [y, *names, *controls], numeric=[y, *names, *controls],
                              missing=missing)
    # Slopes, their t statistics and R^2 do not depend on the origin of any variable.
    outcome = c.centre(c.values(frame, y))[0]
    regressors = c.matrix(frame, [*names, *controls])
    design = torch.cat([torch.ones((len(frame), 1), dtype=torch.float64),
                        regressors - regressors.mean(0)], dim=1)
    n, k = design.shape
    df = n - k
    if df < 1:
        raise AnalysisError("insufficient_observations", f"Partial correlations need more "
                            f"observations ({n}) than regression coefficients ({k}).")
    try:
        fit = kernel_call(linalg.least_squares, design, outcome, drop_collinear=False)
    except AnalysisError as exc:
        raise AnalysisError("collinear_design", "A variable is constant or an exact linear "
                            "combination of the others, so its partial correlation is "
                            "undefined.") from exc
    total = float((outcome - outcome.mean()).square().sum())
    ssr = float(fit.ssr)
    if not total > 0.0 or not ssr > 1e-28 * total:
        raise AnalysisError("perfect_fit", "y is constant or fitted exactly by the regressors, "
                            "so the partial correlations are undefined.")
    r2 = 1.0 - ssr / total
    se = torch.sqrt(ssr / df * fit.xtx_inv.diagonal())
    rows = []
    for position in range(1, len(names) + 1):
        t = float(fit.beta[position] / se[position])
        partial = t / math.sqrt(t * t + df)
        semi = math.copysign(math.sqrt(t * t * (1.0 - r2) / df), t)
        rows.append([partial, semi, partial * partial, semi * semi, t, df,
                     c.t_two_sided(t, df)])
    return c.frame(rows, columns=["partial_corr", "semipartial_corr", "partial_corr_sq",
                                  "semipartial_corr_sq", "statistic", "df", "p_value"],
                   index=names, n=n, n_missing=dropped, r_squared=r2, controls=controls,
                   outcome=y, distribution="t", missing="listwise")
