"""k-means and hierarchical clustering against SciPy and explicit NumPy algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2

import openecon as oe
from openecon.analysis_contracts import AnalysisError

NAMES = ["a", "b", "c"]


def _blobs(seed: int = 0, spread: float = 1.0, size: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(loc=m, scale=spread, size=(size, 3))
                   for m in ([0, 0, 0], [4, 4, 0], [0, 5, 5])])
    rng.shuffle(x)
    return pd.DataFrame(x, columns=NAMES)


def _lloyd(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Plain Lloyd iterations written with explicit distance computations."""
    centers = centers.copy()
    labels = ((x[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
    iterations = 0
    while True:
        iterations += 1
        centers = np.vstack([x[labels == g].mean(0) for g in range(len(centers))])
        new = ((x[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
        if (new == labels).all():
            return centers, labels, iterations
        labels = new


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    table = pd.crosstab(np.asarray(a), np.asarray(b)).to_numpy()
    return ((table > 0).sum(0) == 1).all() and ((table > 0).sum(1) == 1).all()


# ---- k-means ----------------------------------------------------------------------


def test_kmeans_matches_scipy_and_numpy_lloyd():
    frame = _blobs(spread=2.0)                                 # overlapping clusters
    x = frame.to_numpy()
    result = oe.cluster_kmeans(frame, NAMES, 3)
    centers, labels, iterations = _lloyd(x, x[:3])
    np.testing.assert_allclose(result["centers"].to_numpy(), centers, atol=1e-10)
    np.testing.assert_allclose(result["initial_centers"].to_numpy(), x[:3], atol=1e-12)
    assert result.attrs["iterations"] == iterations and result.attrs["converged"]
    assigned = oe.cluster_assign(result, frame)
    assert (assigned["cluster"].to_numpy() - 1 == labels).all()
    np.testing.assert_allclose(assigned["distance"],
                               np.sqrt(((x - centers[labels]) ** 2).sum(1)), atol=1e-10)
    scipy_centers, scipy_labels = kmeans2(x, x[:3].copy(), minit="matrix", iter=200)
    np.testing.assert_allclose(result["centers"].to_numpy(), scipy_centers, atol=1e-8)
    assert (scipy_labels == labels).all()

    sizes = result["sizes"]
    np.testing.assert_allclose(sizes["n"], np.bincount(labels))
    within = np.array([((x[labels == g] - centers[g]) ** 2).sum() for g in range(3)])
    np.testing.assert_allclose(sizes["within_ss"], within, rtol=1e-10)
    total = ((x - x.mean(0)) ** 2).sum()
    assert result.attrs["within_ss"] == pytest.approx(within.sum())
    assert result.attrs["total_ss"] == pytest.approx(total)
    assert result.attrs["between_ss"] == pytest.approx(total - within.sum())
    n = len(x)
    assert result.attrs["calinski_harabasz"] == pytest.approx(
        ((total - within.sum()) / 2) / (within.sum() / (n - 3)))
    for j, name in enumerate(NAMES):
        f, p = stats.f_oneway(*[x[labels == g, j] for g in range(3)])
        assert result["anova"].loc[name, "statistic"] == pytest.approx(f, rel=1e-9)
        assert result["anova"].loc[name, "p_value"] == pytest.approx(p, rel=1e-7)
    assert list(result["anova"]["cluster_df"]) == [2, 2, 2]
    history = result["iterations"]
    assert history["max_change"].iloc[-1] == 0.0 or history["within_ss"].is_monotonic_decreasing
    assert history["within_ss"].iloc[-1] == pytest.approx(within.sum())


def test_kmeans_explicit_centers_and_standardize():
    frame = _blobs(seed=3, spread=1.5)
    frame["c"] = frame["c"] * 100.0
    frame.loc[4, "b"] = np.nan
    complete = frame.dropna()
    x = complete.to_numpy()
    start = [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 5.0, 500.0]]
    result = oe.cluster_kmeans(frame, NAMES, 3, init=start)
    centers, labels, _ = _lloyd(x, np.array(start))
    np.testing.assert_allclose(result["centers"].to_numpy(), centers, rtol=1e-9, atol=1e-9)
    assert result.attrs["init"] == "matrix" and result.attrs["n_missing"] == 1

    z = (x - x.mean(0)) / x.std(0, ddof=1)
    standardized = oe.cluster_kmeans(frame, NAMES, 3, standardize=True)
    z_centers, z_labels, _ = _lloyd(z, z[:3])
    np.testing.assert_allclose(standardized["centers"].to_numpy(), z_centers, atol=1e-9)
    assigned = oe.cluster_assign(standardized, frame)
    assert pd.isna(assigned.loc[4, "cluster"]) and len(assigned) == len(frame)
    assert (assigned["cluster"].dropna().to_numpy().astype(int) - 1 == z_labels).all()
    # New observations are classified with the estimation-sample moments.
    new = pd.DataFrame(z_centers * x.std(0, ddof=1) + x.mean(0), columns=NAMES)
    again = oe.cluster_assign(standardized, new)
    assert list(again["cluster"]) == [1, 2, 3]
    np.testing.assert_allclose(again["distance"], 0.0, atol=1e-9)


def _spss_initial(x: np.ndarray, k: int) -> np.ndarray:
    """SPSS QUICK CLUSTER's initial-centre selection, one case at a time."""
    centers = x[:k].copy()
    for row in x[k:]:
        d = np.sqrt(((centers - row) ** 2).sum(1))
        between = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(2))
        np.fill_diagonal(between, np.inf)
        m, n = np.unravel_index(between.argmin(), between.shape)
        if d.min() > between[m, n]:
            centers[m if d[m] < d[n] else n] = row
            continue
        order = np.argsort(d)
        q, second = order[0], order[1]
        if d[second] > between[q].min():
            centers[q] = row
    return centers


def test_kmeans_initializations():
    frame = _blobs(seed=5, spread=1.2)
    x = frame.to_numpy()
    spss = oe.cluster_kmeans(frame, NAMES, 3, init="spss")
    np.testing.assert_allclose(spss["initial_centers"].to_numpy(), _spss_initial(x, 3),
                               atol=1e-12)
    wide = pd.DataFrame(np.random.default_rng(8).normal(size=(3000, 2)), columns=["a", "b"])
    np.testing.assert_allclose(
        oe.cluster_kmeans(wide, ["a", "b"], 6, init="spss")["initial_centers"].to_numpy(),
        _spss_initial(wide.to_numpy(), 6), atol=1e-12)

    for init in ("random", "kmeans++"):
        one = oe.cluster_kmeans(frame, NAMES, 3, init=init, seed=11)
        two = oe.cluster_kmeans(frame, NAMES, 3, init=init, seed=11)
        pd.testing.assert_frame_equal(one["centers"], two["centers"])
        assert one.attrs["seed"] == 11 and one.attrs["init"] == init
        start = one["initial_centers"].to_numpy()
        # Initial centres are observations of the sample.
        assert all((np.abs(x - row).sum(1) < 1e-9).any() for row in start)
        centers, _, _ = _lloyd(x, start)
        np.testing.assert_allclose(one["centers"].to_numpy(), centers, atol=1e-9)
        assert isinstance(oe.cluster_kmeans(frame, NAMES, 3, init=init).attrs["seed"], int)

    stopped = oe.cluster_kmeans(_blobs(spread=2.5), NAMES, 3, max_iterations=1)
    assert not stopped.attrs["converged"] and "max_iterations" in stopped.attrs["notes"][0]
    one_cluster = oe.cluster_kmeans(frame, NAMES, 1)
    np.testing.assert_allclose(one_cluster["centers"].to_numpy()[0], x.mean(0), atol=1e-10)
    assert one_cluster.attrs["calinski_harabasz"] is None


def test_kmeans_error_codes():
    frame = _blobs()

    def code(*args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            oe.cluster_kmeans(*args, **kwargs)
        return error.value.code

    assert code(frame, NAMES, 0) == "invalid_option"
    assert code(frame, NAMES, 500) == "too_many_clusters"
    assert code(frame, NAMES, 3, init="pam") == "invalid_option"
    assert code(frame, NAMES, 3, init=[[0.0, 1.0]]) == "invalid_option"
    assert code(frame, NAMES, 3, max_iterations=0) == "invalid_option"
    assert code(frame, NAMES, 3, seed=-1) == "invalid_option"
    twins = pd.concat([frame.iloc[:1], frame], ignore_index=True)
    assert code(twins, NAMES, 2) == "duplicate_centers"
    assert code(frame.assign(k=1.0), [*NAMES, "k"], 2, standardize=True) == "constant_column"
    far = [[100.0, 100.0, 100.0], [0.0, 0.0, 0.0], [200.0, 200.0, 200.0]]
    assert code(frame, NAMES, 3, init=far) == "empty_cluster"
    with pytest.raises(AnalysisError) as error:
        oe.cluster_assign(oe.pca(frame, NAMES), frame)
    assert error.value.code == "invalid_result"


# ---- hierarchical clustering ------------------------------------------------------


_CASES = [(link, metric) for link in ("average", "complete", "single", "weighted")
          for metric in ("euclidean", "sqeuclidean", "manhattan", "correlation")] + [
    (link, metric) for link in ("ward", "centroid", "median")
    for metric in ("euclidean", "sqeuclidean")]


@pytest.mark.parametrize(("link", "metric"), _CASES)
def test_hierarchical_matches_scipy_linkage(link, metric):
    frame = _blobs(seed=2, spread=1.5, size=40)
    x = frame.to_numpy()
    result = oe.cluster_hierarchical(frame, NAMES, linkage=link, metric=metric)
    if link in ("ward", "centroid", "median"):
        reference = linkage(x, method=link, metric="euclidean")
        if metric == "sqeuclidean":
            reference[:, 2] = reference[:, 2] ** 2
    else:
        reference = linkage(x, method=link,
                            metric={"manhattan": "cityblock"}.get(metric, metric))
    ours = result["dendrogram"].to_numpy(dtype=float)
    np.testing.assert_allclose(ours[:, 2], reference[:, 2], rtol=1e-9, atol=1e-12)
    assert (ours[:, [0, 1, 3]] == reference[:, [0, 1, 3]]).all()
    assert result.attrs["monotone"] == bool((np.diff(reference[:, 2]) >= 0).all())
    for k in (2, 3, 7):
        cut = oe.cluster_cut(result, k=k)
        assert cut.attrs["k"] == k
        if result.attrs["monotone"]:
            assert _same_partition(cut["cluster"], fcluster(reference, k, "maxclust"))
    level = float(np.sort(reference[:, 2])[-4]) * 1.0000001
    by_height = oe.cluster_cut(result, height=level)
    assert _same_partition(by_height["cluster"], fcluster(reference, level, "distance"))


def test_agglomeration_schedule_and_ward_within_ss():
    frame = _blobs(seed=4, size=15)
    x = frame.to_numpy()
    n = len(x)
    result = oe.cluster_hierarchical(frame, NAMES, linkage="ward", metric="sqeuclidean")
    schedule = result["agglomeration"]
    assert list(schedule["stage"]) == list(range(1, n))
    assert (schedule["cluster1"] < schedule["cluster2"]).all()
    members = {case: {case} for case in range(1, n + 1)}       # replay the schedule
    formed: dict[int, int] = {}
    for row in schedule.itertuples():
        assert row.first_stage1 == formed.get(row.cluster1, 0)
        assert row.first_stage2 == formed.get(row.cluster2, 0)
        members[row.cluster1] |= members.pop(row.cluster2)
        assert len(members[row.cluster1]) == row.size
        assert min(members[row.cluster1]) == row.cluster1
        formed[row.cluster1] = row.stage
        formed.pop(row.cluster2, None)
        if row.next_stage:
            later = schedule.loc[row.next_stage]
            assert row.cluster1 in (later.cluster1, later.cluster2)
        partition = np.zeros(n, dtype=int)
        for label, cases in enumerate(members.values()):
            partition[[case - 1 for case in cases]] = label
        ess = sum(((x[partition == g] - x[partition == g].mean(0)) ** 2).sum()
                  for g in np.unique(partition))
        assert row.within_ss == pytest.approx(ess, rel=1e-9, abs=1e-9)
    assert schedule["next_stage"].iloc[-1] == 0
    # Ward heights on the squared scale are twice the increase of the within SS.
    np.testing.assert_allclose(np.cumsum(schedule["coefficient"]) / 2, schedule["within_ss"],
                               rtol=1e-9)


def test_stopping_rules_match_explicit_computation():
    frame = _blobs(seed=6, size=25)
    x = frame.to_numpy()
    n = len(x)
    result = oe.cluster_hierarchical(frame, NAMES, linkage="average")
    stopping = result["stopping"]
    assert list(stopping["clusters"]) == list(range(1, 16))

    def within(rows: np.ndarray) -> float:
        return ((x[rows] - x[rows].mean(0)) ** 2).sum()

    total = within(np.arange(n))
    for g in range(1, 16):
        now = oe.cluster_cut(result, k=g)["cluster"].to_numpy()
        after = oe.cluster_cut(result, k=g + 1)["cluster"].to_numpy()
        w = sum(within(np.where(now == label)[0]) for label in np.unique(now))
        if g > 1:
            assert stopping.loc[g, "calinski_harabasz"] == pytest.approx(
                ((total - w) / (g - 1)) / (w / (n - g)), rel=1e-9)
        else:
            assert np.isnan(stopping.loc[g, "calinski_harabasz"])
        split = next(label for label in np.unique(now)
                     if len(np.unique(after[now == label])) == 2)
        rows = np.where(now == split)[0]
        parts = [rows[after[rows] == label] for label in np.unique(after[rows])]
        je1, je2 = within(rows), sum(within(part) for part in parts)
        assert stopping.loc[g, "je2_je1"] == pytest.approx(je2 / je1, rel=1e-9)
        if len(rows) > 2:
            assert stopping.loc[g, "pseudo_t2"] == pytest.approx(
                (je1 - je2) / (je2 / (len(rows) - 2)), rel=1e-9)
        else:                                     # a pair splits into two single cases
            assert np.isnan(stopping.loc[g, "pseudo_t2"])


def test_hierarchical_missing_rows_standardize_and_guards():
    frame = _blobs(seed=7, size=12)
    frame.index = [f"id{i}" for i in range(len(frame))]
    frame.loc["id3", "a"] = np.nan
    frame["c"] = frame["c"] * 50
    result = oe.cluster_hierarchical(frame, NAMES, linkage="complete", standardize=True)
    complete = frame.dropna()
    z = ((complete - complete.mean()) / complete.std(ddof=1)).to_numpy()
    np.testing.assert_allclose(result["dendrogram"]["height"], linkage(z, "complete")[:, 2],
                               rtol=1e-9)
    assert result.attrs["n"] == 35 and result.attrs["n_missing"] == 1
    assert 3 not in list(result["cases"]["row"]) and len(result["cases"]) == 35
    cut = oe.cluster_cut(result, k=3)
    assert list(cut.index) == list(result["cases"]["row"])
    aligned = oe.cluster_cut(result, k=3, data=frame)
    assert list(aligned.index) == list(frame.index) and pd.isna(aligned.loc["id3", "cluster"])
    assert (aligned["cluster"].dropna().to_numpy() == cut["cluster"].to_numpy()).all()
    assert _same_partition(cut["cluster"], fcluster(linkage(z, "complete"), 3, "maxclust"))

    def code(function, *args, **kwargs) -> str:
        with pytest.raises(AnalysisError) as error:
            function(*args, **kwargs)
        return error.value.code

    assert code(oe.cluster_hierarchical, frame, NAMES, max_n=10) == "too_many_observations"
    assert code(oe.cluster_hierarchical, frame, NAMES, linkage="mcquitty") == "invalid_option"
    assert code(oe.cluster_hierarchical, frame, NAMES, linkage="ward",
                metric="manhattan") == "invalid_option"
    assert code(oe.cluster_hierarchical, frame, ["a"], linkage="single",
                metric="correlation") == "constant_row"
    assert code(oe.cluster_hierarchical, frame.iloc[:1], NAMES) == "insufficient_observations"
    assert code(oe.cluster_cut, result) == "invalid_option"
    assert code(oe.cluster_cut, result, k=2, height=1.0) == "invalid_option"
    assert code(oe.cluster_cut, result, k=99) == "invalid_option"
    assert code(oe.cluster_cut, oe.cluster_kmeans(frame, NAMES, 2), k=2) == "invalid_result"
    assert code(oe.cluster_cut, result, k=2, data=frame.iloc[:5]) == "invalid_data"
    text = str(result)
    assert "[agglomeration]" in text and "[stopping]" in text
    assert result.to_latex().count(r"\begin{tabular}") == 4


def test_kmeans_spss_convergence_criterion_and_large_offset():
    frame = _blobs(seed=9, spread=2.5, size=200)
    exact = oe.cluster_kmeans(frame, NAMES, 3)
    loose = oe.cluster_kmeans(frame, NAMES, 3, tolerance=0.02)
    assert loose.attrs["converged"] and loose.attrs["iterations"] < exact.attrs["iterations"]
    start = frame[NAMES].to_numpy()[:3]
    smallest = min(np.linalg.norm(start[i] - start[j]) for i in range(3) for j in range(i))
    assert loose["iterations"]["max_change"].iloc[-1] <= 0.02 * smallest
    assert (loose["iterations"]["max_change"].iloc[:-1] > 0.02 * smallest).all()
    # Distances are computed on centred data: a huge common level changes nothing.
    shifted = oe.cluster_kmeans(frame + 1e8, NAMES, 3)
    assert shifted.attrs["iterations"] == exact.attrs["iterations"]
    np.testing.assert_allclose(shifted["centers"].to_numpy() - 1e8, exact["centers"].to_numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(shifted.attrs["within_ss"], exact.attrs["within_ss"], rtol=1e-7)
