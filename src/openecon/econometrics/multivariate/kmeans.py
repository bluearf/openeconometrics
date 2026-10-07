"""k-means cluster analysis: ``oe.cluster_kmeans`` and ``oe.cluster_assign``.

SPSS ``QUICK CLUSTER``; Stata ``cluster kmeans``. Lloyd's algorithm, fully
vectorized: squared distances to the k centres come from the expansion
||x||^2 - 2 x'c + ||c||^2 evaluated in row blocks (never an n-by-n matrix), and
centres are updated in O(n p) with ``index_add_``. The data are centred at the
variable means first, which keeps the expansion free of cancellation.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c
from openecon.engines.contracts import KernelError

INITS = ("first", "random", "kmeans++", "spss")
_BLOCK_ELEMENTS = 1 << 22
# SPSS initial centres: k * p up to which dense replacements are stepped on Python
# floats, and how many cases one such run covers.
_STEPWISE_WORK = 1024
_STEPWISE_RUN = 64


def nearest(x: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
    """(index of the nearest centre [n], squared distance to it [n]) in row blocks."""
    n, k = x.shape[0], centers.shape[0]
    norms = centers.square().sum(1)
    codes = torch.empty(n, dtype=torch.int64)
    distance = torch.empty(n, dtype=x.dtype)
    block = max(1, _BLOCK_ELEMENTS // max(k, 1))
    for start in range(0, n, block):
        rows = x[start:start + block]
        value, index = torch.addmm(norms.expand(rows.shape[0], k), rows, centers.T,
                                   alpha=-2.0).min(1)
        codes[start:start + block] = index
        distance[start:start + block] = value + rows.square().sum(1)
    return codes, distance.clamp_min(0.0)


def _pairwise(a: Tensor, b: Tensor) -> Tensor:
    return (a.square().sum(1)[:, None] - 2.0 * (a @ b.T) + b.square().sum(1)).clamp_min(0.0)


def _squared(a: Tensor, b: Tensor) -> Tensor:
    """Squared Euclidean distances between the rows of ``a`` and ``b``, from differences."""
    return (a[:, None, :] - b[None, :, :]).square().sum(2)


def _python_squared(u: list[float], v: list[float]) -> float:
    total = 0.0
    for a, b in zip(u, v, strict=True):
        total += (a - b) * (a - b)
    return total


def _stepwise(x: Tensor, centers: Tensor, start: int, count: int) -> tuple[int, int]:
    """Apply SPSS's two replacement tests to ``count`` cases one at a time.

    Used where replacements are dense (for example data sorted along a clustering
    variable, where almost every case replaces a centre): for small k * p the
    per-case work on Python floats costs microseconds, while every vectorized
    evaluation costs several tensor operations. Updates ``centers`` in place and
    returns (next position, number of replacements).
    """
    k = centers.shape[0]
    rows = x[start:start + count].tolist()
    current = centers.tolist()
    between = [[math.inf if i == j else _python_squared(current[i], current[j])
                for j in range(k)] for i in range(k)]

    def summary() -> tuple[float, int, int, list[float]]:
        value, first, second = math.inf, 0, 1
        for i in range(k - 1):
            row = between[i]
            for j in range(i + 1, k):
                if row[j] < value:
                    value, first, second = row[j], i, j
        return value, first, second, [min(row) for row in between]

    closest, pair_a, pair_b, nearest_other = summary()
    replaced = 0
    for row in rows:
        distance = [_python_squared(row, centre) for centre in current]
        nearest = min(range(k), key=distance.__getitem__)
        if distance[nearest] > closest:
            target = pair_a if distance[pair_a] < distance[pair_b] else pair_b
        elif min(value for j, value in enumerate(distance) if j != nearest) \
                > nearest_other[nearest]:
            target = nearest
        else:
            continue
        current[target] = row
        for j in range(k):
            if j != target:
                between[target][j] = between[j][target] = distance[j]
        closest, pair_a, pair_b, nearest_other = summary()
        replaced += 1
    if replaced:
        centers.copy_(torch.tensor(current, dtype=centers.dtype))
    return start + len(rows), replaced


def spss_initial(x: Tensor, k: int) -> Tensor:
    """SPSS QUICK CLUSTER's default initial centres: k well-separated cases.

    The first k cases start as centres. Each later case x replaces a centre when
    (a) its distance to the nearest centre exceeds the distance between the two
    closest centres (it replaces whichever of those two is nearer to x), or
    otherwise (b) its distance to the second-nearest centre exceeds the smallest
    distance from the nearest centre to any other centre (it replaces the nearest
    centre). Ties go to the centre listed first.

    The pass is sequential by definition. Runs of cases without a replacement are
    screened in vectorized blocks (doubling in size while nothing is replaced);
    after a replacement the following cases are stepped one at a time on Python
    floats when k * p is small, which keeps data sorted along a clustering
    variable (a replacement at almost every case) at a few microseconds per case.
    Both paths evaluate the same squared distances from coordinate differences.
    """
    centers = x[:k].clone()
    if k < 2:
        return centers
    n, p = x.shape
    stepwise = k * p <= _STEPWISE_WORK
    limit = max(1, _BLOCK_ELEMENTS // (k * p))
    position, block, dense = k, 256, False
    while position < n:
        if dense and stepwise:
            position, replaced = _stepwise(x, centers, position, _STEPWISE_RUN)
            dense = replaced > 0
            continue
        rows = x[position:position + min(block, limit)]
        distance = _squared(rows, centers)
        between = _squared(centers, centers)
        between.diagonal().fill_(math.inf)
        flat = int(between.argmin())
        pair_a, pair_b = flat // k, flat % k
        nearest = distance.argmin(1)
        own = distance.gather(1, nearest[:, None]).squeeze(1)
        second = distance.scatter(1, nearest[:, None], math.inf).min(1).values
        first_test = own > between[pair_a, pair_b]
        second_test = second > between.min(1).values[nearest]
        hits = torch.nonzero(first_test | second_test).flatten()
        if hits.numel() == 0:
            position += rows.shape[0]
            block = min(block * 2, 1 << 16)
            continue
        j = int(hits[0])
        if bool(first_test[j]):
            target = pair_a if float(distance[j, pair_a]) < float(distance[j, pair_b]) \
                else pair_b
        else:
            target = int(nearest[j])
        centers[target] = rows[j]
        position += j + 1
        block, dense = 16, True
    return centers


def plus_plus(x: Tensor, k: int, generator: torch.Generator) -> Tensor:
    """k-means++ seeding (Arthur and Vassilvitskii 2007): D^2-weighted sampling."""
    n = x.shape[0]
    chosen = [int(torch.randint(n, (1,), generator=generator))]
    distance = (x - x[chosen[0]]).square().sum(1)
    for _ in range(1, k):
        total = float(distance.sum())
        if not total > 0.0:
            raise KernelError("duplicate_centers", "Fewer than k distinct observations exist, "
                              "so k distinct initial centres cannot be chosen; lower k.")
        index = int(torch.multinomial(distance / total, 1, generator=generator))
        chosen.append(index)
        distance = torch.minimum(distance, (x - x[index]).square().sum(1))
    return x[chosen].clone()


def lloyd(x: Tensor, centers: Tensor, *, max_iter: int, tolerance: float
          ) -> tuple[Tensor, Tensor, list[list[float]], bool]:
    """Lloyd iterations: (centres, codes, history, converged).

    Each iteration assigns every observation to its nearest centre (the centre
    minimizing ||c||^2 - 2 x'c; ||x||^2 is added once, to the reported within sum
    of squares) and replaces the centres by the cluster means (an empty cluster
    keeps its centre). It stops when no assignment changes, or when the largest
    centre shift is at most ``tolerance`` (an absolute distance).
    """
    k, p = centers.shape
    # One transposed copy [p, n]: the products centres @ X' and the column-wise
    # index_add_ run several times faster on it than on the [n, p] layout.
    xt = x.T.contiguous()
    n = xt.shape[1]
    block = max(1, _BLOCK_ELEMENTS // k)

    def assign(current: Tensor) -> tuple[Tensor, float]:
        norms = current.square().sum(1)[:, None]
        out = torch.empty(n, dtype=torch.int64)
        partial = 0.0
        for start in range(0, n, block):
            columns = xt[:, start:start + block]
            value, index = torch.addmm(norms.expand(k, columns.shape[1]), current, columns,
                                       alpha=-2.0).min(0)
            out[start:start + block] = index
            partial += float(value.sum())
        return out, partial

    codes, _ = assign(centers)
    total = float(x.square().sum())
    history: list[list[float]] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        counts = torch.bincount(codes, minlength=k).to(x.dtype)
        sums = torch.zeros((p, k), dtype=x.dtype).index_add_(1, codes, xt).T
        updated = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1.0)[:, None], centers)
        shift = float((updated - centers).square().sum(1).max().sqrt())
        centers = updated
        new_codes, partial = assign(centers)
        history.append([iteration, shift, max(total + partial, 0.0)])
        unchanged = bool(torch.equal(new_codes, codes))
        codes = new_codes
        if unchanged or shift <= tolerance:
            converged = True
            break
    return centers, codes, history, converged


def _initial_centers(init: Any, x: Tensor, k: int, seed: int | None, names: list[str],
                     transform) -> tuple[Tensor, str, int | None]:
    if isinstance(init, str):
        c.check_choice(init, "init", INITS)
        if init == "first":
            return x[:k].clone(), init, None
        if init == "spss":
            return spss_initial(x, k), init, None
        if seed is None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)))
        generator = torch.Generator().manual_seed(seed)
        if init == "random":
            return x[torch.randperm(x.shape[0], generator=generator)[:k]].clone(), init, seed
        return plus_plus(x, k, generator), init, seed
    try:
        centers = torch.as_tensor(init, dtype=c.FLOAT)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_option", "init must be one of "
                            f"{', '.join(INITS)} or a k-by-p list of initial centres.") from exc
    if centers.shape != (k, len(names)) or not bool(torch.isfinite(centers).all()):
        raise AnalysisError("invalid_option", f"Initial centres must be a {k}-by-{len(names)} "
                            "table of finite numbers (one row per cluster, one column per "
                            "variable, in the order of columns).")
    return transform(centers), "matrix", None


@c.procedure
def cluster_kmeans(data: Any, columns: list[str], k: int, *, init: Any = "first",
                   seed: int | None = None, max_iterations: int = 10000,
                   tolerance: float = 0.0, standardize: bool = False,
                   missing: str = "drop") -> TableSet:
    """k-means cluster analysis (Lloyd's algorithm).

    The n observations are partitioned into k clusters that minimize the
    within-cluster sum of squared Euclidean distances to the cluster means,

        W = sum_i || x_i - m_{c(i)} ||^2.

    Starting from k initial centres, every observation is assigned to its nearest
    centre, the centres are replaced by the means of their clusters, and the two
    steps repeat until no observation changes cluster (or the largest centre
    shift is at most ``tolerance`` times the smallest distance between initial
    centres, SPSS's CONVERGE). Centres are updated after each full pass (SPSS's
    running-means option is off; Stata's behaviour).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : one or more numeric columns.
    k : number of clusters, between 1 and the number of observations.
    init : initial centres. ``"first"`` (default): the first k complete cases (Stata
        ``start(firstk)``, SPSS ``/INITIAL`` off with NOINITIAL); ``"spss"``: SPSS
        QUICK CLUSTER's default selection of k well-separated cases; ``"random"``: k
        distinct observations at random (Stata's default ``start(krandom)``);
        ``"kmeans++"``: D^2-weighted random seeding (Arthur and Vassilvitskii 2007);
        or a k-by-p list of centres in the units of the variables (with
        ``standardize=True`` they are standardized with the sample means and standard
        deviations, like the data; ``initial_centers`` then reports them as z-scores).
    seed : seed of the random initializations; drawn and recorded when omitted.
    max_iterations : iteration limit (default 10000 as Stata; SPSS's default is 10).
        Reaching it is reported in ``attrs["converged"]`` and a note, as SPSS does.
    tolerance : SPSS's convergence criterion as a share of the smallest distance
        between initial centres (default 0: iterate until nothing changes).
    standardize : analyse z-scores (mean 0, standard deviation 1 with divisor n - 1)
        instead of the raw variables; centres are then reported in z-score units.
    missing : ``"drop"`` (listwise deletion) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``initial_centers`` and ``centers``: one row per cluster, one column per variable
      (final centres are the cluster means);
    * ``sizes``: ``n``, ``percent`` and ``within_ss`` of each cluster;
    * ``anova``: for each variable the between-cluster and within-cluster mean
      squares, their degrees of freedom (k - 1, n - k), F and its p-value. As SPSS
      notes, the clusters were chosen to maximize these differences, so the F tests
      are descriptive only;
    * ``iterations``: ``iteration``, ``max_change`` (largest centre shift) and
      ``within_ss`` after each iteration;
    * ``descriptives``: mean and standard deviation of each variable.

    ``attrs``: ``n``, ``n_missing``, ``k``, ``within_ss``, ``between_ss``, ``total_ss``,
    ``calinski_harabasz`` = [B/(k-1)] / [W/(n-k)], ``iterations``, ``converged``,
    ``init``, ``seed``, ``standardize``, ``variables``, ``notes``.

    Cluster membership is per observation and is therefore not stored: use
    ``oe.cluster_assign(result, data)``.

    Equivalent commands: SPSS ``QUICK CLUSTER x1 x2 /CRITERIA=CLUSTER(3) MXITER(100)
    /PRINT ANOVA``; Stata ``cluster kmeans x1 x2, k(3) start(firstk)``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 1.2, 0.8, 5.0, 5.2, 4.9, 9.0, 9.1, 8.8],
    ...         "y": [1.0, 0.9, 1.1, 5.1, 4.8, 5.0, 1.0, 1.2, 0.9]}
    >>> result = oe.cluster_kmeans(data, ["x", "y"], 3, init="spss")
    >>> result["sizes"]["n"].tolist()
    [3, 3, 3]
    """
    names = c.name_list(columns, "columns", minimum=1)
    k = c.check_count(k, "k")
    max_iterations = c.check_count(max_iterations, "max_iterations")
    tolerance = c.check_number(tolerance, "tolerance", minimum=0.0, maximum=1.0)
    c.check_flag(standardize, "standardize")
    if seed is not None:
        seed = c.check_count(seed, "seed", minimum=0)
    sample, _, dropped = c.select(data, names, missing=missing)
    raw = c.matrix(sample, names)
    n, p = raw.shape
    if k > n:
        raise AnalysisError("too_many_clusters", f"k = {k} exceeds the {n} complete "
                            "observation(s); lower k.")
    mean, sscp, x = c.moments(raw, names, need_variation=standardize)
    std = (sscp.diagonal() / max(n - 1, 1)).sqrt()
    if standardize:
        x = x / std

    def transform(values: Tensor) -> Tensor:
        return (values - mean) / std if standardize else values - mean

    def report(values: Tensor) -> Tensor:
        return values if standardize else values + mean

    start, init_name, seed = _initial_centers(init, x, k, seed, names, transform)
    gaps = _pairwise(start, start)
    gaps.diagonal().fill_(float("inf"))
    smallest = math.sqrt(float(gaps.min())) if k > 1 else 0.0
    if k > 1 and smallest == 0.0:
        raise AnalysisError("duplicate_centers", "Two initial centres coincide, which leaves "
                            "a cluster empty. Choose another init (for example 'kmeans++' or "
                            "'spss') or lower k.")
    centers, codes, history, converged = lloyd(
        x, start, max_iter=max_iterations, tolerance=tolerance * smallest)
    counts = torch.bincount(codes, minlength=k).to(c.FLOAT)
    if bool((counts == 0).any()):
        raise AnalysisError("empty_cluster", f"{int((counts == 0).sum())} cluster(s) ended "
                            "without observations. Choose another init (for example "
                            "'kmeans++' or 'spss') or lower k.")
    group_mean = torch.zeros((k, p), dtype=c.FLOAT).index_add_(0, codes, x) / counts[:, None]
    deviation = x - group_mean[codes]
    within = deviation.square().sum(0)
    cluster_within = torch.zeros(k, dtype=c.FLOAT).index_add_(0, codes, deviation.square().sum(1))
    between = (counts[:, None] * group_mean.square()).sum(0)
    total_within, total_between = float(within.sum()), float(between.sum())
    notes = []
    if not converged:
        notes.append(f"Iterations stopped at max_iterations = {max_iterations} before the "
                     "clusters stabilized; the centres are those of the last update.")
    anova = []
    df1, df2 = k - 1, n - k
    for j in range(p):
        ms_between = float(between[j]) / df1 if df1 > 0 else None
        ms_within = float(within[j]) / df2 if df2 > 0 else None
        statistic = ms_between / ms_within if ms_between is not None and ms_within else None
        anova.append([ms_between, df1, ms_within, df2, statistic,
                      c.f_upper(statistic, df1, df2)])
    labels = list(range(1, k + 1))
    pseudo_f = None
    if df1 > 0 and df2 > 0 and total_within > 0:
        pseudo_f = (total_between / df1) / (total_within / df2)
    tables = {
        "initial_centers": c.frame(report(start), columns=names, index=labels),
        "centers": c.frame(report(centers), columns=names, index=labels),
        "sizes": c.frame(torch.stack([counts, 100.0 * counts / n, cluster_within], dim=1),
                         columns=["n", "percent", "within_ss"], index=labels),
        "anova": c.frame(anova, columns=["cluster_ms", "cluster_df", "error_ms", "error_df",
                                         "statistic", "p_value"], index=names),
        "iterations": c.frame(history, columns=["iteration", "max_change", "within_ss"],
                              index=list(range(1, len(history) + 1))),
        "descriptives": c.frame(torch.stack([mean, std], dim=1), columns=["mean", "std_dev"],
                                index=names),
    }
    tables["iterations"]["iteration"] = tables["iterations"]["iteration"].astype("int64")
    tables["sizes"]["n"] = tables["sizes"]["n"].astype("int64")
    for name in ("initial_centers", "centers", "sizes"):
        tables[name].index.name = "cluster"
    return TableSet(
        tables, title=f"k-means cluster analysis of {', '.join(names)} ({k} clusters)",
        procedure="cluster_kmeans", n=n, n_missing=dropped, k=k, within_ss=total_within,
        between_ss=total_between, total_ss=total_within + total_between,
        calinski_harabasz=pseudo_f, iterations=len(history), converged=converged,
        init=init_name, seed=seed, standardize=standardize, variables=names, notes=notes,
        missing="listwise")


@c.procedure
def cluster_assign(result: TableSet, data: Any) -> pd.DataFrame:
    """Cluster membership of the rows of ``data`` from a k-means result.

    Every complete row is assigned to the nearest final centre (Euclidean
    distance in the units of the analysis: z-scores based on the estimation-sample
    means and standard deviations when the clustering used ``standardize=True``).
    Applied to the estimation data of a converged analysis this reproduces the
    final clusters (SPSS ``/SAVE CLUSTER DISTANCE``, Stata's generated variable).

    Parameters
    ----------
    result : the TableSet returned by ``oe.cluster_kmeans``.
    data : rows to classify; must contain the analysed columns. Rows with a missing
        value get a missing cluster.

    Returns
    -------
    A table indexed like ``data`` with ``cluster`` (1..k) and ``distance`` (to the
    centre of that cluster).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 1.2, 0.8, 5.0, 5.2, 4.9], "y": [1.0, 0.9, 1.1, 5.1, 4.8, 5.0]}
    >>> result = oe.cluster_kmeans(data, ["x", "y"], 2)
    >>> oe.cluster_assign(result, data)["cluster"].tolist()
    [1, 1, 1, 2, 2, 2]
    """
    c.check_result(result, "cluster_kmeans", "cluster_assign")
    names = list(result.attrs["variables"])
    frame = c.source(data)
    c.require_numeric(frame, names)
    frame = frame.loc[:, names]
    keep = ~frame.isna().any(axis=1)
    x = c.matrix(frame.loc[keep], names)
    stats = result["descriptives"]
    mean = torch.as_tensor(stats.loc[names, "mean"].to_numpy(dtype="float64").copy())
    centers = torch.as_tensor(result["centers"].loc[:, names].to_numpy(dtype="float64").copy())
    if result.attrs["standardize"]:
        std = torch.as_tensor(stats.loc[names, "std_dev"].to_numpy(dtype="float64").copy())
        x = (x - mean) / std
    else:
        x, centers = x - mean, centers - mean
    codes, distance = nearest(x, centers)
    values = torch.stack([codes.to(c.FLOAT) + 1.0, distance.sqrt()], dim=1)
    return c.aligned(values, keep, ["cluster", "distance"], integer=["cluster"])
