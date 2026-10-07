"""Independent oracles for clustering, discriminant, canonical, MDS and CA procedures.

Expected values come from explicit NumPy loops and algebra, SciPy's hierarchy /
vq / stats routines, statsmodels' MANOVA, brute-force leave-one-out refits and
invariances; nothing here reuses the implementation's formulas.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import linalg as sla
from scipy import stats
from scipy.cluster import hierarchy, vq
from scipy.spatial.distance import pdist, squareform
from statsmodels.multivariate.manova import MANOVA

import openecon as oe

# ---- k-means --------------------------------------------------------------------------


def _blobs(n: int = 240, seed: int = 0, p: int = 3) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(4, p))
    labels = rng.integers(0, 4, n)
    x = centres[labels] + rng.normal(size=(n, p)) * np.array([1.0, 2.0, 0.5])[:p]
    names = [f"v{j}" for j in range(1, p + 1)]
    return pd.DataFrame(x * np.array([1.0, 10.0, 0.1])[:p], columns=names), names


def _lloyd(x: np.ndarray, centres: np.ndarray, max_iter: int = 1000) -> tuple:
    c = centres.copy()
    labels = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
    for _ in range(max_iter):
        c = np.array([x[labels == g].mean(0) if (labels == g).any() else c[g]
                      for g in range(len(c))])
        new = ((x[:, None, :] - c[None]) ** 2).sum(2).argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
    return c, labels


def test_kmeans_against_numpy_lloyd_and_scipy_kmeans2():
    frame, names = _blobs()
    x = frame[names].to_numpy()
    n, p = x.shape
    k = 4
    result = oe.cluster_kmeans(frame, names, k)
    centres, labels = _lloyd(x, x[:k])
    np.testing.assert_allclose(result["centers"].to_numpy(), centres, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result["initial_centers"].to_numpy(), x[:k])
    np.testing.assert_array_equal(oe.cluster_assign(result, frame)["cluster"].to_numpy(),
                                  labels + 1)
    scipy_c, scipy_l = vq.kmeans2(x, x[:k].copy(), iter=100, minit="matrix")
    np.testing.assert_allclose(result["centers"].to_numpy(), scipy_c, rtol=1e-10, atol=1e-10)
    sizes = np.bincount(labels, minlength=k)
    np.testing.assert_array_equal(result["sizes"]["n"].to_numpy(), sizes)
    np.testing.assert_allclose(result["sizes"]["percent"], 100 * sizes / n)
    within_g = np.array([((x[labels == g] - centres[g]) ** 2).sum() for g in range(k)])
    np.testing.assert_allclose(result["sizes"]["within_ss"], within_g, rtol=1e-10)
    w = within_g.sum()
    t = ((x - x.mean(0)) ** 2).sum()
    assert result.attrs["within_ss"] == pytest.approx(w, rel=1e-10)
    assert result.attrs["between_ss"] == pytest.approx(t - w, rel=1e-10)
    assert result.attrs["total_ss"] == pytest.approx(t, rel=1e-10)
    assert result.attrs["calinski_harabasz"] == pytest.approx(
        ((t - w) / (k - 1)) / (w / (n - k)), rel=1e-10)
    anova = result["anova"]
    for j, name in enumerate(names):
        f, pval = stats.f_oneway(*[x[labels == g, j] for g in range(k)])
        assert anova.loc[name, "statistic"] == pytest.approx(f, rel=1e-9)
        assert anova.loc[name, "p_value"] == pytest.approx(pval, rel=1e-6, abs=1e-300)
        assert anova.loc[name, "cluster_df"] == k - 1 and anova.loc[name, "error_df"] == n - k
    assert result.attrs["converged"]
    assert result["iterations"]["within_ss"].iloc[-1] == pytest.approx(w, rel=1e-10)


def _spss_start(x: np.ndarray, k: int) -> np.ndarray:
    """SPSS QUICK CLUSTER initial centres, case by case (Algorithms manual, step 1)."""
    c = x[:k].copy()
    for i in range(k, len(x)):
        d = np.sqrt(((c - x[i]) ** 2).sum(1))
        cc = np.sqrt(((c[:, None] - c[None]) ** 2).sum(2))
        np.fill_diagonal(cc, np.inf)
        a, b = np.unravel_index(np.argmin(cc), cc.shape)
        order = np.argsort(d, kind="stable")
        if d[order[0]] > cc[a, b]:
            c[a if d[a] < d[b] else b] = x[i]
        elif d[order[1]] > cc[order[0]].min():
            c[order[0]] = x[i]
    return c


def test_kmeans_spss_start_standardize_and_tolerance():
    frame, names = _blobs(seed=5)
    x = frame[names].to_numpy()
    spss = oe.cluster_kmeans(frame, names, 4, init="spss")
    start = _spss_start(x, 4)
    np.testing.assert_allclose(spss["initial_centers"].to_numpy(), start, rtol=1e-12)
    np.testing.assert_allclose(spss["centers"].to_numpy(), _lloyd(x, start)[0], rtol=1e-10,
                               atol=1e-10)
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    std = oe.cluster_kmeans(frame, names, 3, standardize=True)
    np.testing.assert_allclose(std["centers"].to_numpy(), _lloyd(z, z[:3])[0], rtol=1e-9,
                               atol=1e-10)
    # SPSS CONVERGE: stop when the largest centre shift <= tol * min initial distance.
    tol = 0.02
    gap = min(np.linalg.norm(z[i] - z[j]) for i in range(3) for j in range(i + 1, 3))
    c = z[:3].copy()
    for _ in range(1000):
        lab = ((z[:, None] - c[None]) ** 2).sum(2).argmin(1)
        new = np.array([z[lab == g].mean(0) for g in range(3)])
        shift = np.sqrt(((new - c) ** 2).sum(1)).max()
        c = new
        if shift <= tol * gap:
            break
    early = oe.cluster_kmeans(frame, names, 3, standardize=True, tolerance=tol)
    np.testing.assert_allclose(early["centers"].to_numpy(), c, rtol=1e-9, atol=1e-10)


def test_kmeans_translation_invariance_seeded_inits_and_missing_rows():
    frame, names = _blobs(seed=8)
    shifted = frame + 1e6
    a = oe.cluster_kmeans(frame, names, 4, init="kmeans++", seed=3)
    b = oe.cluster_kmeans(shifted, names, 4, init="kmeans++", seed=3)
    np.testing.assert_allclose(b["centers"].to_numpy() - 1e6, a["centers"].to_numpy(),
                               atol=1e-7)
    assert a.attrs["seed"] == 3 and a.attrs["init"] == "kmeans++"
    # A converged solution is a Lloyd fixed point.
    x = frame[names].to_numpy()
    labels = oe.cluster_assign(a, frame)["cluster"].to_numpy() - 1
    means = np.array([x[labels == g].mean(0) for g in range(4)])
    np.testing.assert_allclose(a["centers"].to_numpy(), means, rtol=1e-10, atol=1e-10)
    r1 = oe.cluster_kmeans(frame, names, 4, init="random", seed=11)
    r2 = oe.cluster_kmeans(frame, names, 4, init="random", seed=11)
    np.testing.assert_array_equal(r1["centers"].to_numpy(), r2["centers"].to_numpy())
    holes = frame.copy()
    holes.iloc[[0, 3], 1] = np.nan
    c = oe.cluster_kmeans(holes, names, 3)
    d = oe.cluster_kmeans(holes.dropna(), names, 3)
    assert c.attrs["n_missing"] == 2
    np.testing.assert_allclose(c["centers"].to_numpy(), d["centers"].to_numpy(), atol=1e-12)
    assign = oe.cluster_assign(c, holes)
    assert assign["cluster"].isna().sum() == 2 and len(assign) == len(holes)


# ---- hierarchical clustering ---------------------------------------------------------


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "weighted", "ward",
                                     "centroid", "median"])
def test_hierarchical_heights_against_scipy(linkage):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 3)) * np.array([1.0, 3.0, 0.3])
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    squared = linkage in ("ward", "centroid", "median")
    z = hierarchy.linkage(x, method=linkage, metric="euclidean")
    ours = oe.cluster_hierarchical(frame, ["a", "b", "c"], linkage=linkage)
    tree = ours["dendrogram"]
    np.testing.assert_allclose(tree["height"], z[:, 2], rtol=1e-9)
    np.testing.assert_array_equal(tree["size"], z[:, 3].astype(int))
    for k in (2, 3, 5, 9):
        expected = hierarchy.fcluster(z, k, criterion="maxclust")
        got = oe.cluster_cut(ours, k=k)["cluster"].to_numpy()
        # Same partition (labels may be numbered differently).
        assert pd.crosstab(got, expected).gt(0).sum(1).eq(1).all()
        assert len(set(got)) == k
    if squared:
        sq = oe.cluster_hierarchical(frame, ["a", "b", "c"], linkage=linkage,
                                     metric="sqeuclidean")
        np.testing.assert_allclose(sq["dendrogram"]["height"], z[:, 2] ** 2, rtol=1e-9)
    if linkage == "ward":
        # SPSS's Ward coefficient: cumulative within-cluster sum of squares; each merge of
        # clusters a, b adds n_a n_b / (n_a + n_b) ||mean_a - mean_b||^2 = height^2 / 2.
        np.testing.assert_allclose(ours["agglomeration"]["within_ss"],
                                   np.cumsum(z[:, 2] ** 2 / 2), rtol=1e-9)


@pytest.mark.parametrize("metric,scipy_metric", [("manhattan", "cityblock"),
                                                 ("correlation", "correlation"),
                                                 ("sqeuclidean", "sqeuclidean")])
def test_hierarchical_metrics_against_scipy(metric, scipy_metric):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, 4))
    frame = pd.DataFrame(x, columns=list("abcd"))
    for linkage in ("average", "complete", "single"):
        z = hierarchy.linkage(pdist(x, scipy_metric), method=linkage)
        ours = oe.cluster_hierarchical(frame, list("abcd"), linkage=linkage, metric=metric)
        np.testing.assert_allclose(ours["dendrogram"]["height"], z[:, 2], rtol=1e-9, atol=1e-12)


def test_agglomeration_schedule_and_stopping_rules_from_cut_partitions():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(loc, 0.6, size=(15, 2)) for loc in ((0, 0), (5, 0), (0, 5))])
    frame = pd.DataFrame(x, columns=["a", "b"])
    result = oe.cluster_hierarchical(frame, ["a", "b"], linkage="average")
    z = hierarchy.linkage(x, "average")
    n = len(x)
    stopping = result["stopping"]
    total = ((x - x.mean(0)) ** 2).sum()

    def within(labels):
        return sum(((x[labels == g] - x[labels == g].mean(0)) ** 2).sum()
                   for g in np.unique(labels))

    for g in range(1, 11):
        labels = hierarchy.fcluster(z, g, criterion="maxclust")
        w = within(labels)
        if g > 1:
            assert stopping.loc[g, "calinski_harabasz"] == pytest.approx(
                ((total - w) / (g - 1)) / (w / (n - g)), rel=1e-9)
        # Duda-Hart: the cluster that splits when going from g to g + 1 clusters.
        finer = hierarchy.fcluster(z, g + 1, criterion="maxclust")
        table = pd.crosstab(labels, finer)
        parent = table.index[(table > 0).sum(1) == 2][0]
        members = labels == parent
        je1 = ((x[members] - x[members].mean(0)) ** 2).sum()
        parts = np.unique(finer[members])
        je2 = sum(((x[finer == q] - x[finer == q].mean(0)) ** 2).sum() for q in parts)
        size = members.sum()
        assert stopping.loc[g, "je2_je1"] == pytest.approx(je2 / je1, rel=1e-9)
        if size > 2:
            assert stopping.loc[g, "pseudo_t2"] == pytest.approx(
                (je1 - je2) / (je2 / (size - 2)), rel=1e-9)
    # The schedule replays the dendrogram: cluster sizes and next-stage pointers agree.
    sched = result["agglomeration"]
    np.testing.assert_allclose(sched["coefficient"], z[:, 2], rtol=1e-10)
    np.testing.assert_array_equal(sched["size"], z[:, 3].astype(int))
    for stage, row in sched.iterrows():
        if row["next_stage"]:
            later = sched.loc[row["next_stage"]]
            assert stage in (later["first_stage1"], later["first_stage2"])
    # Height cut equals the k cut at a height between merges.
    cut = oe.cluster_cut(result, height=float((z[-3, 2] + z[-2, 2]) / 2))["cluster"].to_numpy()
    assert len(set(cut)) == 3


@pytest.mark.parametrize("linkage", ["centroid", "median", "average"])
def test_height_cut_matches_scipy_cophenetic_rule_even_with_inversions(linkage):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 2))
    frame = pd.DataFrame(x, columns=["a", "b"])
    z = hierarchy.linkage(x, linkage)
    result = oe.cluster_hierarchical(frame, ["a", "b"], linkage=linkage)
    assert result.attrs["monotone"] == bool(hierarchy.is_monotonic(z))
    if linkage == "centroid":
        assert not result.attrs["monotone"]                 # the design has an inversion
    levels = np.unique(z[:, 2])
    # Cut midway between merge heights (a cut exactly at a height depends on rounding).
    for t in ((levels[1:] + levels[:-1]) / 2)[::3]:
        expected = hierarchy.fcluster(z, t, criterion="distance")
        got = oe.cluster_cut(result, height=float(t))["cluster"].to_numpy()
        assert len(set(got)) == len(set(expected))
        assert pd.crosstab(got, expected).gt(0).sum(1).eq(1).all()
    # A k cut undoes the last k - 1 merges, so it always has exactly k clusters.
    for k in range(2, 20):
        assert oe.cluster_cut(result, k=k).attrs["k"] == k


# ---- discriminant analysis -----------------------------------------------------------


def _groups(n_per=(25, 30, 20), seed: int = 0, p: int = 3) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for g, size in enumerate(n_per):
        mean = np.array([g * 1.2, (-1) ** g * 0.8, g * g * 0.3, 0.1 * g])[:p]
        a = rng.normal(size=(p, p)) * 0.3 + np.eye(p) * (1 + 0.2 * g)
        blocks.append(rng.normal(size=(size, p)) @ a + mean)
        labels += [f"g{g}"] * size
    names = [f"x{j}" for j in range(1, p + 1)]
    frame = pd.DataFrame(np.vstack(blocks) * np.array([1.0, 100.0, 0.01, 5.0])[:p], columns=names)
    frame.insert(0, "grp", labels)
    return frame, names


def _lda_oracle(x, codes, G):
    n, p = x.shape
    means = np.array([x[codes == g].mean(0) for g in range(G)])
    w = sum((x[codes == g] - means[g]).T @ (x[codes == g] - means[g]) for g in range(G))
    t = (x - x.mean(0)).T @ (x - x.mean(0))
    return means, w, t


def test_lda_canonical_functions_against_generalized_eigenproblem():
    frame, names = _groups()
    x = frame[names].to_numpy()
    codes = pd.factorize(frame["grp"], sort=True)[0]
    n, p = x.shape
    G = 3
    means, w, t = _lda_oracle(x, codes, G)
    b = t - w
    vals, vecs = sla.eigh(b, w)
    vals, vecs = vals[::-1][:2], vecs[:, ::-1][:, :2]
    sw = w / (n - G)
    vecs = vecs / np.sqrt(np.einsum("ij,jk,ki->i", vecs.T, sw, vecs))   # v' S_w v = 1
    result = oe.discrim(frame, "grp", names)
    canon = result["canonical_functions"]
    np.testing.assert_allclose(canon["eigenvalue"], vals, rtol=1e-9)
    np.testing.assert_allclose(canon["percent"], 100 * vals / vals.sum(), rtol=1e-9)
    np.testing.assert_allclose(canon["canonical_correlation"], np.sqrt(vals / (1 + vals)),
                               rtol=1e-9)
    for k in range(2):
        lam = np.prod(1 / (1 + vals[k:]))
        chi2 = -(n - 1 - (p + G) / 2) * math.log(lam)
        df = (p - k) * (G - k - 1)
        assert canon["wilks_lambda"].iloc[k] == pytest.approx(lam, rel=1e-9)
        assert canon["chi2"].iloc[k] == pytest.approx(chi2, rel=1e-9)
        assert canon["df"].iloc[k] == df
        assert canon["p_value"].iloc[k] == pytest.approx(stats.chi2.sf(chi2, df), rel=1e-7)
    raw = result["unstandardized_coefficients"]
    coef = raw.loc[names].to_numpy()
    for k in range(2):
        s = np.sign(coef[:, k] @ vecs[:, k])
        np.testing.assert_allclose(coef[:, k], s * vecs[:, k], rtol=1e-8)
    np.testing.assert_allclose(raw.loc["Intercept"].to_numpy(), -x.mean(0) @ coef, rtol=1e-8)
    scores = x @ coef + raw.loc["Intercept"].to_numpy()
    np.testing.assert_allclose(scores.mean(0), 0, atol=1e-9)
    pooled = sum(((scores[codes == g] - scores[codes == g].mean(0)) ** 2).sum(0)
                 for g in range(G)) / (n - G)
    np.testing.assert_allclose(pooled, 1.0, rtol=1e-9)
    np.testing.assert_allclose(result["standardized_coefficients"].to_numpy(),
                               coef * np.sqrt(np.diag(sw))[:, None], rtol=1e-8)
    within_dev = x - means[codes]
    sdev = scores - np.array([scores[codes == g].mean(0) for g in range(G)])[codes]
    structure = (within_dev.T @ sdev) / np.sqrt(
        np.outer((within_dev ** 2).sum(0), (sdev ** 2).sum(0)))
    np.testing.assert_allclose(result["structure_matrix"].to_numpy(), structure, rtol=1e-8,
                               atol=1e-12)
    np.testing.assert_allclose(result["centroids"].to_numpy(),
                               np.array([scores[codes == g].mean(0) for g in range(G)]),
                               rtol=1e-8, atol=1e-10)
    eq = result["tests_of_equality"]
    for j, name in enumerate(names):
        f, pval = stats.f_oneway(*[x[codes == g, j] for g in range(G)])
        assert eq.loc[name, "statistic"] == pytest.approx(f, rel=1e-9)
        assert eq.loc[name, "wilks_lambda"] == pytest.approx(w[j, j] / t[j, j], rel=1e-10)
        assert eq.loc[name, "p_value"] == pytest.approx(pval, rel=1e-7)


def _box_m(x, codes, G):
    n, p = x.shape
    ng = np.bincount(codes)
    covs = [np.cov(x[codes == g].T) for g in range(G)]
    pooled = sum((ng[g] - 1) * covs[g] for g in range(G)) / (n - G)
    m = (n - G) * np.linalg.slogdet(pooled)[1] - sum(
        (ng[g] - 1) * np.linalg.slogdet(covs[g])[1] for g in range(G))
    a1 = (np.sum(1 / (ng - 1)) - 1 / (n - G)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (G - 1))
    a2 = (np.sum(1 / (ng - 1) ** 2) - 1 / (n - G) ** 2) * (p - 1) * (p + 2) / (6 * (G - 1))
    df1 = (G - 1) * p * (p + 1) / 2
    if a2 > a1 ** 2:
        df2 = (df1 + 2) / (a2 - a1 ** 2)
        b_ = df1 / (1 - a1 - df1 / df2)
        f = m / b_
    else:
        df2 = (df1 + 2) / (a1 ** 2 - a2)
        b_ = df2 / (1 - a1 + 2 / df2)
        f = df2 * m / (df1 * (b_ - m))
    return m, f, df1, df2, m * (1 - a1)


@pytest.mark.parametrize("sizes", [(25, 30, 20), (12, 40, 15)])
def test_box_m_against_independent_formula(sizes):
    frame, names = _groups(n_per=sizes, seed=2)
    x = frame[names].to_numpy()
    codes = pd.factorize(frame["grp"], sort=True)[0]
    m, f, df1, df2, chi2 = _box_m(x, codes, 3)
    box = oe.discrim(frame, "grp", names)["box_m"].iloc[0]
    assert box["statistic"] == pytest.approx(m, rel=1e-9)
    assert box["f"] == pytest.approx(f, rel=1e-9)
    assert box["df1"] == df1 and box["df2"] == pytest.approx(df2, rel=1e-10)
    assert box["p_value"] == pytest.approx(stats.f.sf(f, df1, df2), rel=1e-7)
    assert box["chi2"] == pytest.approx(chi2, rel=1e-9)
    assert box["chi2_p_value"] == pytest.approx(stats.chi2.sf(chi2, df1), rel=1e-7)


def _posteriors(x, train_x, train_codes, G, prior, method):
    n_t, p = train_x.shape
    means = np.array([train_x[train_codes == g].mean(0) for g in range(G)])
    if method == "lda":
        w = sum((train_x[train_codes == g] - means[g]).T @ (train_x[train_codes == g] - means[g])
                for g in range(G))
        covs = [w / (n_t - G)] * G
    else:
        covs = [np.cov(train_x[train_codes == g].T) for g in range(G)]
    logs = np.column_stack([stats.multivariate_normal(means[g], covs[g]).logpdf(x)
                            + math.log(prior[g]) for g in range(G)])
    logs -= logs.max(1, keepdims=True)
    post = np.exp(logs)
    return post / post.sum(1, keepdims=True)


@pytest.mark.parametrize("method", ["lda", "qda"])
@pytest.mark.parametrize("priors", ["equal", "proportional"])
def test_classification_and_leave_one_out_against_brute_force(method, priors):
    frame, names = _groups(n_per=(18, 22, 15), seed=7)
    x = frame[names].to_numpy()
    codes = pd.factorize(frame["grp"], sort=True)[0]
    G = 3
    prior = np.full(G, 1 / G) if priors == "equal" else np.bincount(codes) / len(codes)
    result = oe.discrim(frame, "grp", names, method=method, priors=priors, loo=True)
    post = _posteriors(x, x, codes, G, prior, method)
    predicted = post.argmax(1)
    table = pd.crosstab(codes, predicted).reindex(index=range(G), columns=range(G), fill_value=0)
    ours = result["classification_table"]
    np.testing.assert_array_equal(ours[["g0", "g1", "g2"]].to_numpy(), table.to_numpy())
    assert result.attrs["percent_correct"] == pytest.approx(100 * (predicted == codes).mean())
    pred = oe.discrim_predict(result, frame)
    np.testing.assert_allclose(pred[["posterior_g0", "posterior_g1", "posterior_g2"]].to_numpy(),
                               post, atol=1e-9)
    loo = np.empty(len(x), dtype=int)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        loo[i] = _posteriors(x[i:i + 1], x[keep], codes[keep], G, prior, method).argmax(1)[0]
    loo_table = pd.crosstab(codes, loo).reindex(index=range(G), columns=range(G), fill_value=0)
    np.testing.assert_array_equal(
        result["classification_table_loo"][["g0", "g1", "g2"]].to_numpy(), loo_table.to_numpy())
    if method == "lda":
        means, w, _ = _lda_oracle(x, codes, G)
        sw = w / (len(x) - G)
        coef = np.linalg.solve(sw, means.T)
        const = -0.5 * np.einsum("gj,jg->g", means, coef) + np.log(prior)
        funcs = result["classification_functions"]
        np.testing.assert_allclose(funcs.loc[names].to_numpy(), coef, rtol=1e-8)
        np.testing.assert_allclose(funcs.loc["Intercept"].to_numpy(), const, rtol=1e-8)


def test_discrim_invariance_to_row_order_and_label_type():
    frame, names = _groups(seed=4)
    a = oe.discrim(frame, "grp", names)
    shuffled = frame.sample(frac=1.0, random_state=1)
    shuffled["grp"] = shuffled["grp"].map({"g0": 10, "g1": 20, "g2": 30})
    b = oe.discrim(shuffled, "grp", names)
    for table in ("canonical_functions", "unstandardized_coefficients", "structure_matrix"):
        np.testing.assert_allclose(a[table].to_numpy(dtype=float), b[table].to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(a["classification_table"].to_numpy(),
                                  b["classification_table"].to_numpy())


# ---- canonical correlation ------------------------------------------------------------


def _canon_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    x = np.column_stack([z[:, 0] + rng.normal(size=n), z[:, 1] + rng.normal(size=n),
                         rng.normal(size=n)]) * np.array([1.0, 50.0, 0.02]) + 3.0
    y = np.column_stack([0.7 * z[:, 0] + rng.normal(size=n), z[:, 1] * 0.4 + rng.normal(size=n)])
    frame = pd.DataFrame(np.column_stack([x, y]), columns=["x1", "x2", "x3", "y1", "y2"])
    return frame, ["x1", "x2", "x3"], ["y1", "y2"]


def test_canon_against_covariance_eigenproblem_and_direct_correlations():
    frame, xs, ys = _canon_data()
    x, y = frame[xs].to_numpy(), frame[ys].to_numpy()
    n, p, q = len(x), len(xs), len(ys)
    s = np.cov(np.column_stack([x, y]).T)
    sxx, syy, sxy = s[:p, :p], s[p:, p:], s[:p, p:]
    m = np.linalg.solve(syy, sxy.T) @ np.linalg.solve(sxx, sxy)
    rho2 = np.sort(np.linalg.eigvals(m).real)[::-1]
    result = oe.canon(frame, x=xs, y=ys)
    corr = result["correlations"]
    np.testing.assert_allclose(corr["correlation"], np.sqrt(rho2), rtol=1e-9)
    raw = result["raw_coefficients"]
    a, b = raw.loc[xs, ["Canon1", "Canon2"]].to_numpy(), raw.loc[ys, ["Canon1",
                                                                     "Canon2"]].to_numpy()
    u, v = x @ a, y @ b
    np.testing.assert_allclose(np.var(u, axis=0, ddof=1), 1.0, rtol=1e-9)
    np.testing.assert_allclose(np.var(v, axis=0, ddof=1), 1.0, rtol=1e-9)
    full = np.corrcoef(np.column_stack([u, v]).T)
    np.testing.assert_allclose(np.diag(full[:2, 2:]), np.sqrt(rho2), rtol=1e-9)
    assert abs(full[0, 1]) < 1e-9 and abs(full[0, 3]) < 1e-9 and abs(full[1, 2]) < 1e-9
    std = result["standardized_coefficients"]
    np.testing.assert_allclose(std.loc[xs, ["Canon1", "Canon2"]].to_numpy(),
                               a * x.std(0, ddof=1)[:, None], rtol=1e-9)
    load = result["loadings"]
    for k, name in enumerate(["Canon1", "Canon2"]):
        for j, var in enumerate(xs):
            assert load.loc[var, name] == pytest.approx(np.corrcoef(x[:, j], u[:, k])[0, 1],
                                                        abs=1e-10)
            assert load.loc[var, f"Cross{k + 1}"] == pytest.approx(
                np.corrcoef(x[:, j], v[:, k])[0, 1], abs=1e-10)
        for j, var in enumerate(ys):
            assert load.loc[var, name] == pytest.approx(np.corrcoef(y[:, j], v[:, k])[0, 1],
                                                        abs=1e-10)
    red = result["redundancy"]
    xl = load.loc[xs, ["Canon1", "Canon2"]].to_numpy()
    np.testing.assert_allclose(red["x_variance"], (xl ** 2).mean(0), rtol=1e-10)
    np.testing.assert_allclose(red["x_redundancy"], (xl ** 2).mean(0) * rho2, rtol=1e-10)
    # Sequential tests: Bartlett chi2 and Rao's F with dimensions p-k+1, q-k+1.
    w = n - 1 - (p + q + 1) / 2
    for k in range(2):
        lam = np.prod(1 - rho2[k:])
        pk, qk = p - k, q - k
        assert corr["wilks_lambda"].iloc[k] == pytest.approx(lam, rel=1e-9)
        chi2 = -w * math.log(lam)
        assert corr["chi2"].iloc[k] == pytest.approx(chi2, rel=1e-9)
        assert corr["chi2_df"].iloc[k] == pk * qk
        denom = pk * pk + qk * qk - 5
        tt = math.sqrt((pk * pk * qk * qk - 4) / denom) if denom > 0 else 1.0
        df2 = w * tt - (pk * qk - 2) / 2
        f = (1 - lam ** (1 / tt)) / lam ** (1 / tt) * df2 / (pk * qk)
        assert corr["f"].iloc[k] == pytest.approx(f, rel=1e-9)
        assert corr["df2"].iloc[k] == pytest.approx(df2, rel=1e-12)
        assert corr["p_value"].iloc[k] == pytest.approx(stats.f.sf(f, pk * qk, df2), rel=1e-7)


def test_canon_multivariate_tests_against_statsmodels_manova():
    frame, xs, ys = _canon_data(seed=3)
    result = oe.canon(frame, x=xs, y=ys)
    tests = result["tests"]
    mv = MANOVA.from_formula("y1 + y2 ~ x1 + x2 + x3", data=frame).mv_test(
        hypotheses=[("all", np.eye(4)[1:])])
    sm = mv.results["all"]["stat"]
    assert tests.loc["wilks", "value"] == pytest.approx(sm.loc["Wilks' lambda", "Value"], rel=1e-9)
    assert tests.loc["wilks", "statistic"] == pytest.approx(sm.loc["Wilks' lambda", "F Value"],
                                                            rel=1e-8)
    assert tests.loc["pillai", "value"] == pytest.approx(sm.loc["Pillai's trace", "Value"],
                                                         rel=1e-9)
    assert tests.loc["pillai", "statistic"] == pytest.approx(sm.loc["Pillai's trace", "F Value"],
                                                             rel=1e-8)
    assert tests.loc["hotelling", "value"] == pytest.approx(
        sm.loc["Hotelling-Lawley trace", "Value"], rel=1e-9)
    assert tests.loc["roy", "value"] == pytest.approx(sm.loc["Roy's greatest root", "Value"],
                                                      rel=1e-9)
    assert tests.loc["roy", "statistic"] == pytest.approx(sm.loc["Roy's greatest root", "F Value"],
                                                          rel=1e-8)
    # Lawley-Hotelling: the classical F approximation (Pillai 1954; SPSS / Stata manova).
    n, p, q = len(frame), 3, 2
    s, m_, nn = min(p, q), (abs(p - q) - 1) / 2, (n - p - q - 2) / 2
    u = tests.loc["hotelling", "value"]
    f = 2 * (s * nn + 1) * u / (s * s * (2 * m_ + s + 1))
    assert tests.loc["hotelling", "statistic"] == pytest.approx(f, rel=1e-10)
    assert tests.loc["hotelling", "df2"] == 2 * (s * nn + 1)


# ---- multidimensional scaling ----------------------------------------------------------


def test_classical_mds_against_double_centring():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4)) * np.array([3.0, 2.0, 1.0, 0.5])
    d = squareform(pdist(x))
    d_noisy = d + np.triu(rng.uniform(0, 0.3, size=d.shape), 1)
    d_noisy = d_noisy + np.triu(d_noisy, 1).T - np.triu(d, 1).T
    np.fill_diagonal(d_noisy, 0)
    n = len(d_noisy)
    j = np.eye(n) - 1 / n
    b = -0.5 * j @ (d_noisy ** 2) @ j
    w, v = np.linalg.eigh(b)
    w, v = w[::-1], v[:, ::-1]
    result = oe.mds(distances=d_noisy, dimensions=3)
    coords = _sign(v[:, :3] * np.sqrt(w[:3]))
    np.testing.assert_allclose(result["coordinates"].to_numpy(), coords, atol=1e-9)
    shown = result["eigenvalues"]["eigenvalue"].to_numpy()
    np.testing.assert_allclose(shown, w[:len(shown)], atol=1e-9)
    assert result.attrs["mardia1"] == pytest.approx(w[:3].sum() / np.abs(w).sum(), rel=1e-9)
    assert result.attrs["mardia2"] == pytest.approx((w[:3] ** 2).sum() / (w ** 2).sum(),
                                                    rel=1e-9)
    # From data: classical scaling of Euclidean distances = centred principal coordinates.
    frame = pd.DataFrame(x, columns=list("abcd"))
    from_data = oe.mds(frame, list("abcd"), dimensions=2)
    from_d = oe.mds(distances=d, dimensions=2)
    np.testing.assert_allclose(from_data["coordinates"].to_numpy(),
                               from_d["coordinates"].to_numpy(), atol=1e-9)
    assert from_data.attrs["mardia1"] == pytest.approx(from_d.attrs["mardia1"], rel=1e-9)


def _sign(v):
    idx = np.abs(v).argmax(0)
    s = np.sign(v[idx, np.arange(v.shape[1])])
    return v * np.where(s == 0, 1, s)


def test_smacof_against_independent_guttman_loop():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(25, 3))
    delta = squareform(pdist(x)) + squareform(rng.uniform(0, 0.4, size=25 * 24 // 2))
    result = oe.mds(distances=delta, method="smacof", dimensions=2, tolerance=1e-12,
                    max_iterations=20000)
    start = oe.mds(distances=delta, dimensions=2)["coordinates"].to_numpy()
    conf = start.copy()
    n = len(delta)

    def raw(c):
        return ((squareform(pdist(c)) - delta) ** 2).sum() / 2

    for _ in range(20000):
        dist = squareform(pdist(conf))
        ratio = np.divide(delta, dist, out=np.zeros_like(delta), where=dist > 0)
        bmat = -ratio
        np.fill_diagonal(bmat, ratio.sum(1))
        new = bmat @ conf / n
        if raw(conf) - raw(new) <= 1e-12 * raw(conf):
            conf = new
            break
        conf = new
    assert result.attrs["raw_stress"] == pytest.approx(raw(conf), rel=1e-6)
    ours = result["coordinates"].to_numpy()
    dist = squareform(pdist(ours))
    assert raw(ours) == pytest.approx(result.attrs["raw_stress"], rel=1e-9)
    stress1 = math.sqrt(((dist - delta) ** 2).sum() / (dist ** 2).sum())
    assert result.attrs["stress1"] == pytest.approx(stress1, rel=1e-9)
    assert raw(ours) <= raw(start) + 1e-12
    np.testing.assert_allclose(ours.mean(0), 0, atol=1e-10)                # principal axes
    cross = ours.T @ ours
    assert abs(cross[0, 1]) < 1e-8 * cross[0, 0]


# ---- correspondence analysis ----------------------------------------------------------


def test_ca_against_numpy_svd_and_chi2_contingency():
    table = np.array([[68, 119, 26, 7], [20, 84, 17, 94], [15, 54, 14, 10], [5, 29, 14, 16]],
                     dtype=float)
    rows, cols = ["black", "brown", "red", "blond"], ["brown", "hazel", "green", "blue"]
    records = pd.DataFrame([(r, c, table[i, j]) for i, r in enumerate(rows)
                            for j, c in enumerate(cols)], columns=["hair", "eye", "count"])
    result = oe.ca(records, "hair", "eye", weights="count", dimensions=2)
    order_r, order_c = sorted(rows), sorted(cols)
    t = table[[rows.index(r) for r in order_r]][:, [cols.index(c) for c in order_c]]
    n = t.sum()
    p = t / n
    r, c = p.sum(1), p.sum(0)
    s = (p - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    u, sv, vt = np.linalg.svd(s)
    u, sv, v = u[:, :3], sv[:3], vt.T[:, :3]
    chi2, pval, dof, _ = stats.chi2_contingency(t, correction=False)
    assert result.attrs["chi2"] == pytest.approx(chi2, rel=1e-10)
    assert result.attrs["p_value"] == pytest.approx(pval, rel=1e-8)
    assert result.attrs["df"] == dof
    assert result.attrs["total_inertia"] == pytest.approx(chi2 / n, rel=1e-10)
    inertia = result["inertia"]
    np.testing.assert_allclose(inertia["singular_value"], sv, rtol=1e-10)
    np.testing.assert_allclose(inertia["principal_inertia"], sv ** 2, rtol=1e-10)
    np.testing.assert_allclose(inertia["percent"], 100 * sv ** 2 / (sv ** 2).sum(), rtol=1e-10)
    row_std = u / np.sqrt(r)[:, None]
    row_pr = row_std * sv
    col_std = v / np.sqrt(c)[:, None]
    col_pr = col_std * sv
    ours_r, ours_c = result["rows"], result["columns"]
    for k in range(2):
        sgn = np.sign(ours_r[f"dim{k + 1}_standard"].to_numpy() @ row_std[:, k])
        np.testing.assert_allclose(ours_r[f"dim{k + 1}_standard"], sgn * row_std[:, k], rtol=1e-9)
        np.testing.assert_allclose(ours_r[f"dim{k + 1}"], sgn * row_pr[:, k], rtol=1e-9)
        np.testing.assert_allclose(ours_c[f"dim{k + 1}"], sgn * col_pr[:, k], rtol=1e-9)
        np.testing.assert_allclose(ours_r[f"dim{k + 1}_contribution"],
                                   r * row_pr[:, k] ** 2 / sv[k] ** 2, rtol=1e-9)
        d2 = (row_pr ** 2).sum(1)
        np.testing.assert_allclose(ours_r[f"dim{k + 1}_sqcorr"], row_pr[:, k] ** 2 / d2,
                                   rtol=1e-9)
    # Transition formula: row principal coordinates are averages of column standard ones.
    prof = p / r[:, None]
    np.testing.assert_allclose(ours_r[["dim1", "dim2"]].to_numpy(),
                               prof @ ours_c[["dim1_standard", "dim2_standard"]].to_numpy(),
                               atol=1e-10)
    np.testing.assert_allclose(ours_r["mass"], r, rtol=1e-12)
    np.testing.assert_allclose(ours_r["quality"], (row_pr[:, :2] ** 2).sum(1) / (row_pr ** 2).sum(
        1), rtol=1e-9)
    np.testing.assert_allclose(ours_r["inertia"], r * (row_pr ** 2).sum(1) / (sv ** 2).sum(),
                               rtol=1e-9)
    # Individual records give the same analysis as the weighted table.
    expanded = records.loc[records.index.repeat(records["count"].astype(int))]
    again = oe.ca(expanded, "hair", "eye")
    np.testing.assert_allclose(again["rows"].to_numpy(), ours_r.to_numpy(), atol=1e-12)
