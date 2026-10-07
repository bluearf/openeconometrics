"""Agglomerative hierarchical clustering: ``oe.cluster_hierarchical`` and ``oe.cluster_cut``.

SPSS ``CLUSTER``; Stata ``cluster singlelinkage`` ... ``wardslinkage``, ``cluster
generate`` and ``cluster stop``.

This is the one procedure of the family that needs the n-by-n dissimilarity
matrix, so the number of observations is guarded (``max_n``). Clusters are
merged with the Lance-Williams recurrence: when clusters a and b (sizes n_a,
n_b, dissimilarity d_ab) merge, the dissimilarity of the new cluster to any
other cluster k is

    single     min(d_ak, d_bk)                 complete  max(d_ak, d_bk)
    average    (n_a d_ak + n_b d_bk) / (n_a + n_b)        weighted  (d_ak + d_bk) / 2
    centroid   (n_a d_ak + n_b d_bk) / (n_a + n_b) - n_a n_b d_ab / (n_a + n_b)^2
    median     (d_ak + d_bk) / 2 - d_ab / 4
    ward       [(n_a + n_k) d_ak + (n_b + n_k) d_bk - n_k d_ab] / (n_a + n_b + n_k)

with centroid, median and Ward's linkage applied to SQUARED Euclidean distances.
Single, complete, average, weighted and Ward's linkage are reducible, so the
merges are found with the nearest-neighbour chain algorithm (O(n^2) work: follow
nearest neighbours until two clusters are each other's nearest neighbour, merge
them, continue from the rest of the chain) and then sorted by height (never
ahead of the merges they depend on, even when ties round a height down). Centroid
and median linkage can produce inversions; they use the generic algorithm that
merges the globally closest pair, tracked through per-row nearest neighbours.
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

LINKAGES = ("ward", "average", "complete", "single", "centroid", "median", "weighted")
METRICS = ("euclidean", "sqeuclidean", "manhattan", "correlation")
_SQUARED = ("ward", "centroid", "median")       # recurrences defined on squared distances
_INF = float("inf")


def distances(x: Tensor, metric: str) -> Tensor:
    """The [n, n] dissimilarity matrix between the rows of ``x``."""
    if metric in ("euclidean", "sqeuclidean"):
        d = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
        return d.square_() if metric == "sqeuclidean" else d
    if metric == "manhattan":
        return torch.cdist(x, x, p=1.0)
    centred = x - x.mean(1, keepdim=True)
    norm = centred.square().sum(1).sqrt()
    if x.shape[1] < 2 or bool((norm <= 1e-12 * norm.max().clamp_min(1e-300)).any()):
        raise AnalysisError("constant_row", "The correlation dissimilarity 1 - r between "
                            "observations needs at least two variables and no observation "
                            "whose values are all equal.")
    unit = centred / norm[:, None]
    d = (1.0 - unit @ unit.T).clamp_min(0.0)
    d.diagonal().fill_(0.0)
    return d


def _update(method: str, da: Tensor, db: Tensor, dab: float, na: float, nb: float,
            sizes: Tensor) -> Tensor:
    """Lance-Williams dissimilarities of the merged cluster to every other cluster."""
    if method == "single":
        return torch.minimum(da, db)
    if method == "complete":
        return torch.maximum(da, db)
    if method == "average":
        return (na * da + nb * db) / (na + nb)
    if method == "weighted":
        return 0.5 * (da + db)
    if method == "centroid":
        return (na * da + nb * db) / (na + nb) - na * nb * dab / (na + nb) ** 2
    if method == "median":
        return 0.5 * (da + db) - 0.25 * dab
    return ((na + sizes) * da + (nb + sizes) * db - sizes * dab) / (na + nb + sizes)


def _merge(d: Tensor, sizes: Tensor, a: int, b: int, method: str) -> None:
    """Merge slot a into slot b in place; slot a becomes inactive (infinite distances)."""
    na, nb = float(sizes[a]), float(sizes[b])
    new = _update(method, d[a], d[b], float(d[a, b]), na, nb, sizes)
    new[a] = new[b] = _INF
    d[b, :] = new
    d[:, b] = new
    d[a, :] = _INF
    d[:, a] = _INF
    sizes[b] = na + nb
    sizes[a] = 0.0


def nn_chain(d: Tensor, method: str) -> list[tuple[int, int, float]]:
    """Merges (slot a, slot b, height) of a reducible linkage, in chain order."""
    n = d.shape[0]
    d.diagonal().fill_(_INF)
    sizes = torch.ones(n, dtype=d.dtype)
    active = [True] * n
    merges: list[tuple[int, int, float]] = []
    chain: list[int] = []
    first = 0
    for _ in range(n - 1):
        if not chain:
            while not active[first]:
                first += 1
            chain.append(first)
        while True:
            a = chain[-1]
            row = d[a]
            b = int(row.argmin())
            if len(chain) > 1:
                # Prefer the predecessor on ties so that the chain terminates.
                previous = chain[-2]
                if float(row[previous]) <= float(row[b]):
                    b = previous
                if b == previous:
                    break
            chain.append(b)
        b, a = chain.pop(), chain.pop()
        merges.append((a, b, float(d[a, b])))
        _merge(d, sizes, a, b, method)
        active[a] = False
    return merges


def height_order(merges: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Chain-order merges sorted by height, never before the merges they depend on.

    For a reducible linkage a merge is never lower than the merges that formed its
    two clusters, but with tied dissimilarities a Lance-Williams average can round a
    dependent height one unit in the last place below its parent's. Sorting on the
    running maximum of the heights along each cluster's history (stable, so equal
    keys keep chain order, in which dependencies come first) keeps the tree valid;
    the reported heights are unchanged.
    """
    key_of_slot: dict[int, float] = {}
    keyed = []
    for position, (a, b, height) in enumerate(merges):
        key = max(height, key_of_slot.get(a, -math.inf), key_of_slot.get(b, -math.inf))
        key_of_slot[b] = key                     # slot b holds the merged cluster
        keyed.append((key, position))
    keyed.sort()
    return [merges[position] for _, position in keyed]


def generic_linkage(d: Tensor, method: str) -> list[tuple[int, int, float]]:
    """Merges of any linkage, always joining the globally closest pair of clusters."""
    n = d.shape[0]
    d.diagonal().fill_(_INF)
    sizes = torch.ones(n, dtype=d.dtype)
    nearest_value, nearest_index = d.min(1)
    merges: list[tuple[int, int, float]] = []
    for _ in range(n - 1):
        a = int(nearest_value.argmin())
        b = int(nearest_index[a])
        merges.append((a, b, float(nearest_value[a])))
        _merge(d, sizes, a, b, method)
        new = d[b]
        stale = (nearest_index == a) | (nearest_index == b)
        stale[b] = True
        stale &= sizes > 0
        nearest_value[a] = _INF
        rows = torch.nonzero(stale).flatten()
        if rows.numel():
            values, indices = d[rows].min(1)
            nearest_value[rows] = values
            nearest_index[rows] = indices
        better = (new < nearest_value) & ~stale & (sizes > 0)
        nearest_value = torch.where(better, new, nearest_value)
        nearest_index = torch.where(better, torch.full_like(nearest_index, b), nearest_index)
    return merges


def label_merges(merges: list[tuple[int, int, float]], n: int
                 ) -> tuple[list[list[float]], list[list[Any]]]:
    """Linkage matrix rows (left, right, height, size) and SPSS's agglomeration schedule.

    Cluster ids follow the usual convention: 0..n-1 are the observations and
    n + s is the cluster formed at stage s (0-based). In the schedule clusters
    are named by their lowest case number (1-based), ``first_stage`` is the stage
    at which each cluster was formed (0 for a single case) and ``next_stage`` the
    stage at which the new cluster is merged again (0 for the last stage).
    """
    parent = list(range(n))
    cluster_id = list(range(n))           # id of the cluster rooted at each observation
    lowest = list(range(n))               # lowest case index of that cluster
    size = [1] * n
    formed = [0] * (2 * n - 1)            # stage (1-based) that formed each cluster id

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    linkage: list[list[float]] = []
    schedule: list[list[Any]] = []
    for stage, (a, b, height) in enumerate(merges):
        ra, rb = find(a), find(b)
        ia, ib = cluster_id[ra], cluster_id[rb]
        if ia > ib:
            ra, rb, ia, ib = rb, ra, ib, ia
        total = size[ra] + size[rb]
        linkage.append([ia, ib, height, total])
        low_a, low_b = lowest[ra], lowest[rb]
        first, second = (ra, rb) if low_a < low_b else (rb, ra)
        schedule.append([stage + 1, lowest[first] + 1, lowest[second] + 1, height,
                         formed[cluster_id[first]], formed[cluster_id[second]], 0, total])
        for child in (ia, ib):
            if formed[child]:
                schedule[formed[child] - 1][6] = stage + 1
        parent[ra] = rb
        cluster_id[rb] = n + stage
        formed[n + stage] = stage + 1
        lowest[rb] = min(low_a, low_b)
        size[rb] = total
    return linkage, schedule


def stopping_rules(x: Tensor, linkage: list[list[float]], limit: int = 15
                   ) -> tuple[list[list[float | None]], list[float]]:
    """Calinski-Harabasz and Duda-Hart indices for 1..limit clusters; within SS per stage.

    The within-cluster sum of squares of a merged cluster follows from
    SS(a u b) = SS(a) + SS(b) + n_a n_b / (n_a + n_b) ||mean_a - mean_b||^2, so one
    pass over the merges gives every index. With W(g) the total within SS of the
    g-cluster solution and T the total SS:

        pseudo-F(g) = [(T - W(g)) / (g - 1)] / [W(g) / (n - g)],
        Je(2)/Je(1) and pseudo-T^2 = (Je(1) - Je(2)) / (Je(2) / (n_1 + n_2 - 2))

    for the cluster that is split in two when going from g to g + 1 clusters
    (Je(1) its within SS, Je(2) the sum of the within SS of its two parts).
    """
    n, p = x.shape
    total_nodes = 2 * n - 1
    means = torch.zeros((total_nodes, p), dtype=x.dtype)
    means[:n] = x
    ss = [0.0] * total_nodes
    increase = []
    for stage, (left, right, _, size) in enumerate(linkage):
        left, right = int(left), int(right)
        na = 1.0 if left < n else linkage[left - n][3]
        nb = size - na
        gap = float((means[left] - means[right]).square().sum())
        delta = na * nb / size * gap
        means[n + stage] = (na * means[left] + nb * means[right]) / size
        ss[n + stage] = ss[left] + ss[right] + delta
        increase.append(delta)
    within_after = torch.cumsum(torch.tensor(increase, dtype=x.dtype), 0).tolist()
    total = float(x.square().sum())            # x is centred at the grand mean
    rows: list[list[float | None]] = []
    for g in range(1, min(limit, n - 1) + 1):
        within = within_after[n - g - 1] if g < n else 0.0
        pseudo_f = None
        if g > 1 and within > 0:
            pseudo_f = ((total - within) / (g - 1)) / (within / (n - g))
        stage = n - g - 1                        # the merge undone when going to g + 1
        left, right, _, size = linkage[stage]
        je1 = ss[n + stage]
        je2 = ss[int(left)] + ss[int(right)]
        ratio = je2 / je1 if je1 > 0 else None
        t2 = (je1 - je2) / (je2 / (size - 2)) if je2 > 0 and size > 2 else None
        rows.append([g, pseudo_f, ratio, t2])
    return rows, within_after


@c.procedure
def cluster_hierarchical(data: Any, columns: list[str], *, linkage: str = "ward",
                         metric: str = "euclidean", standardize: bool = False,
                         max_n: int = 5000, missing: str = "drop") -> TableSet:
    """Agglomerative hierarchical cluster analysis.

    Every observation starts as its own cluster; at each of the n - 1 stages the
    two closest clusters are merged, with the between-cluster dissimilarity
    defined by ``linkage`` (Lance-Williams recurrences; see the module
    documentation for the formulas and algorithms).

    Parameters
    ----------
    data : DataFrame, mapping of columns or list of records.
    columns : one or more numeric columns.
    linkage : ``"ward"`` (default; minimum increase of the within-cluster sum of
        squares), ``"average"`` (UPGMA; SPSS between-groups linkage), ``"complete"``
        (furthest neighbour), ``"single"`` (nearest neighbour), ``"centroid"``,
        ``"median"`` or ``"weighted"`` (WPGMA; Stata waveragelinkage).
    metric : dissimilarity between observations: ``"euclidean"`` (default),
        ``"sqeuclidean"``, ``"manhattan"`` (city block) or ``"correlation"`` (1 minus
        the Pearson correlation between two observations' profiles). Ward's, centroid
        and median linkage are defined on squared Euclidean distances and accept
        only the first two: with ``"euclidean"`` their heights are reported as
        square roots (the convention of SciPy and R's ward.D2), with
        ``"sqeuclidean"`` on the squared scale (Stata's default L2squared for these
        linkages, and the measure SPSS recommends).
    standardize : cluster z-scores (divisor n - 1) instead of the raw variables.
    max_n : largest number of observations accepted. The method needs the n-by-n
        dissimilarity matrix (8 n^2 bytes: 200 MB at 5000); larger samples raise
        ``too_many_observations``. Use ``oe.cluster_kmeans`` or a sample instead, or
        raise ``max_n`` deliberately.
    missing : ``"drop"`` (listwise deletion) or ``"raise"``.

    Returns
    -------
    TableSet with

    * ``agglomeration``: SPSS's agglomeration schedule: ``stage``, ``cluster1``,
      ``cluster2`` (clusters named by their lowest case number, 1-based),
      ``coefficient`` (the merge height), ``first_stage1``, ``first_stage2`` (stage
      at which each cluster was formed; 0 = single case), ``next_stage``, ``size``;
      with Ward's linkage also ``within_ss``, the cumulative within-cluster sum of
      squares that SPSS prints as the coefficient;
    * ``dendrogram``: the linkage matrix ``left``, ``right``, ``height``, ``size`` with
      0-based ids: 0..n-1 are observations, n + s is the cluster formed at row s
      (the format of SciPy's and MATLAB's linkage functions);
    * ``stopping``: for 1 to 15 clusters the Calinski-Harabasz pseudo-F
      (``calinski_harabasz``) and the Duda-Hart ``je2_je1`` ratio with its
      ``pseudo_t2`` (Stata ``cluster stop``), computed from the analysed variables.
      Large pseudo-F, large Je(2)/Je(1) and small pseudo-T^2 indicate distinct
      clustering;
    * ``cases``: for each case number the 0-based row position in ``data``.

    ``attrs``: ``n``, ``n_missing``, ``linkage``, ``metric``, ``standardize``,
    ``variables``, ``monotone`` (False when centroid or median linkage produced an
    inversion).

    Cluster membership: ``oe.cluster_cut(result, k=3)`` or ``height=...``.

    Equivalent commands: SPSS ``CLUSTER x1 x2 /METHOD WARD /MEASURE=SEUCLID
    /PRINT SCHEDULE``; Stata ``cluster wardslinkage x1 x2``, ``cluster generate g =
    groups(3)``, ``cluster stop``.

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 1.2, 0.8, 5.0, 5.2, 4.9], "y": [1.0, 0.9, 1.1, 5.1, 4.8, 5.0]}
    >>> result = oe.cluster_hierarchical(data, ["x", "y"], linkage="average")
    >>> list(oe.cluster_cut(result, k=2)["cluster"])
    [1, 1, 1, 2, 2, 2]
    """
    names = c.name_list(columns, "columns", minimum=1)
    c.check_choice(linkage, "linkage", LINKAGES)
    c.check_choice(metric, "metric", METRICS)
    c.check_flag(standardize, "standardize")
    max_n = c.check_count(max_n, "max_n", minimum=2)
    if linkage in _SQUARED and metric not in ("euclidean", "sqeuclidean"):
        raise AnalysisError("invalid_option", f"{linkage} linkage is defined on squared "
                            "Euclidean distances; use metric='euclidean' or 'sqeuclidean'.")
    sample, keep, dropped = c.select(data, names, missing=missing)
    n = len(sample)
    if n < 2:
        raise AnalysisError("insufficient_observations", "Cluster analysis needs at least two "
                            "complete observations.")
    if n > max_n:
        raise AnalysisError("too_many_observations", f"Hierarchical clustering needs the "
                            f"{n}-by-{n} distance matrix, above the max_n = {max_n} guard. "
                            "Use oe.cluster_kmeans, cluster a sample, or raise max_n "
                            f"(memory: about {8 * n * n / 1e9:.1f} GB).")
    raw = c.matrix(sample, names)
    _, sscp, x = c.moments(raw, names, need_variation=standardize)
    if standardize:
        x = x / (sscp.diagonal() / (n - 1)).sqrt()
    squared = linkage in _SQUARED
    # Distances between points do not depend on the centring; the correlation between
    # two observations' profiles does, so it uses the values as analysed.
    profile = raw if metric == "correlation" and not standardize else x
    d = distances(profile, "sqeuclidean" if squared else metric)
    merges = generic_linkage(d, linkage) if linkage in ("centroid", "median") \
        else height_order(nn_chain(d, linkage))
    del d
    if squared and metric == "euclidean":
        merges = [(a, b, math.sqrt(max(height, 0.0))) for a, b, height in merges]
    rows, schedule = label_merges(merges, n)
    heights = [row[2] for row in rows]
    if not all(math.isfinite(height) for height in heights):
        raise AnalysisError("numerical_failure", "The dissimilarities are not finite; check "
                            "the data for extreme values.")
    stopping, within = stopping_rules(x, rows)
    columns_ = ["stage", "cluster1", "cluster2", "coefficient", "first_stage1", "first_stage2",
                "next_stage", "size"]
    agglomeration = c.frame(schedule, columns=columns_, index=list(range(1, n)))
    for name in columns_:
        if name != "coefficient":
            agglomeration[name] = agglomeration[name].astype("int64")
    if linkage == "ward":
        agglomeration["within_ss"] = within
    dendrogram = c.frame(rows, columns=["left", "right", "height", "size"],
                         index=list(range(n - 1)))
    for name in ("left", "right", "size"):
        dendrogram[name] = dendrogram[name].astype("int64")
    stop = c.frame(stopping, columns=["clusters", "calinski_harabasz", "je2_je1", "pseudo_t2"],
                   index=[int(row[0]) for row in stopping])
    stop["clusters"] = stop["clusters"].astype("int64")
    positions = torch.nonzero(torch.as_tensor(keep.to_numpy())).flatten().tolist()
    cases = c.frame([[position] for position in positions], columns=["row"],
                    index=list(range(1, n + 1)))
    cases["row"] = cases["row"].astype("int64")
    monotone = all(later >= earlier for earlier, later in zip(heights, heights[1:]))
    return TableSet(
        {"agglomeration": agglomeration, "dendrogram": dendrogram, "stopping": stop,
         "cases": cases},
        title=f"Hierarchical cluster analysis of {', '.join(names)} ({linkage} linkage, "
              f"{metric} distances)",
        procedure="cluster_hierarchical", n=n, n_missing=dropped, linkage=linkage,
        metric=metric, standardize=standardize, variables=names, monotone=monotone,
        missing="listwise")


def membership(left: list[int], right: list[int], n: int, merged: list[bool]) -> list[int]:
    """Cluster numbers 1.. (ordered by lowest case) after applying the flagged merges."""
    parent = list(range(2 * n - 1))
    for stage, apply in enumerate(merged):
        if apply:
            parent[left[stage]] = parent[right[stage]] = n + stage
    labels: dict[int, int] = {}
    out = []
    for case in range(n):
        root = case
        while parent[root] != root:
            root = parent[root]
        out.append(labels.setdefault(root, len(labels) + 1))
    return out


@c.procedure
def cluster_cut(result: TableSet, *, k: int | None = None, height: float | None = None,
                data: Any = None) -> pd.DataFrame:
    """Cluster membership from a hierarchical clustering (cut the dendrogram).

    Parameters
    ----------
    result : the TableSet returned by ``oe.cluster_hierarchical``.
    k : number of clusters: the last k - 1 merges are undone (Stata ``cluster
        generate, groups(k)``; SPSS ``/SAVE CLUSTER(k)``).
    height : alternatively, cut at this height: observations are in the same
        cluster when they are joined at a height not above it (Stata ``cut()``).
        With inversions (centroid / median linkage) a merge counts as below the cut
        only if every merge inside it is. Give exactly one of ``k`` and ``height``.
    data : optional; the data passed to ``oe.cluster_hierarchical``. The result is
        then indexed like ``data`` (rows that were not clustered hold a missing
        value) and can be assigned as a new column.

    Returns
    -------
    A table with the column ``cluster`` (1, 2, ... in order of the lowest case
    number of each cluster), indexed by the 0-based row position of each case, or
    like ``data`` when it is given. ``attrs``: ``k`` (number of clusters).

    Example
    -------
    >>> import openecon as oe
    >>> data = {"x": [1.0, 1.2, 0.8, 5.0, 5.2, 4.9], "y": [1.0, 0.9, 1.1, 5.1, 4.8, 5.0]}
    >>> result = oe.cluster_hierarchical(data, ["x", "y"])
    >>> oe.cluster_cut(result, k=2).attrs["k"]
    2
    """
    c.check_result(result, "cluster_hierarchical", "cluster_cut")
    if (k is None) == (height is None):
        raise AnalysisError("invalid_option", "Give exactly one of k (number of clusters) and "
                            "height (cut level).")
    n = int(result.attrs["n"])
    tree = result["dendrogram"]
    left, right = tree["left"].tolist(), tree["right"].tolist()
    heights = tree["height"].tolist()
    if k is not None:
        k = c.check_count(k, "k", maximum=n)
        merged = [stage < n - k for stage in range(n - 1)]
    else:
        level = c.check_number(height, "height")
        top = list(heights)
        for stage in range(n - 1):
            for child in (left[stage], right[stage]):
                if child >= n:
                    top[stage] = max(top[stage], top[child - n])
        merged = [value <= level for value in top]
    clusters = membership(left, right, n, merged)
    rows = result["cases"]["row"].tolist()
    count = max(clusters)
    if data is None:
        out = c.frame([[cluster] for cluster in clusters], columns=["cluster"], index=rows, k=count)
        out["cluster"] = out["cluster"].astype("int64")
        return out
    frame = c.source(data)
    if rows and rows[-1] >= len(frame):
        raise AnalysisError("invalid_data", "data has fewer rows than the data that were "
                            "clustered; pass the same data.")
    keep = pd.Series(False, index=frame.index)
    keep.iloc[rows] = True
    out = c.aligned(torch.tensor(clusters, dtype=c.FLOAT)[:, None], keep, ["cluster"],
                    integer=["cluster"])
    out.attrs["k"] = count
    return out
