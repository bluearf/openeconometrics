"""Independent probability integration, original-author fixtures and agreement oracles."""

from __future__ import annotations

import copy
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, special, stats
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


@pytest.fixture
def ratings():
    # Fixed complete unit/rater geometry, includes disagreement and unequal marginals.
    return pd.DataFrame(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 2, 1],
            [2, 2, 2],
            [2, 1, 2],
            [2, 0, 1],
            [0, 2, 0],
            [1, 0, 2],
        ],
        columns=list("abc"),
        index=[f"subject-{i}" for i in range(10)],
    )


def pair_agreement(x, w):
    return np.mean([np.mean([w[a, b] for a, b in itertools.permutations(row, 2)]) for row in x])


@pytest.mark.parametrize("kind", ["unweighted", "linear", "quadratic"])
def test_cohen_against_statsmodels_and_permutation(ratings, kind):
    count = (
        pd.crosstab(ratings.a, ratings.b)
        .reindex(index=range(3), columns=range(3), fill_value=0)
        .to_numpy()
    )
    weighting = None if kind == "unweighted" else kind
    expected = cohens_kappa(
        count, wt=weighting, weights=None if weighting is None else np.arange(3)
    ).kappa
    r = oe.cohen_kappa(ratings, ["a", "b"], categories=[0, 1, 2], agreement_weights=kind)
    assert r.attrs["estimate"] == pytest.approx(expected, abs=1e-12)
    reordered = oe.cohen_kappa(
        ratings.iloc[::-1], ["b", "a"], categories=[2, 1, 0], agreement_weights=kind
    )
    assert reordered.attrs["estimate"] == pytest.approx(expected, abs=1e-12)


def test_fleiss_against_statsmodels(ratings):
    count = np.array([[sum(row == j) for j in range(3)] for row in ratings.to_numpy()])
    r = oe.fleiss_kappa(ratings, list("abc"), categories=[0, 1, 2])
    assert r.attrs["estimate"] == pytest.approx(fleiss_kappa(count), abs=1e-12)
    assert oe.fleiss_kappa(ratings, list("cba"), categories=[0, 1, 2, 3]).attrs[
        "estimate"
    ] == pytest.approx(r.attrs["estimate"])


@pytest.mark.parametrize("weighting", ["unweighted", "linear", "quadratic", "custom"])
def test_gwet_from_distinct_rater_pairs_and_author_chance(ratings, weighting):
    q = 4  # include unused level: declared universe must enter AC chance denominator.
    d = abs(np.arange(q)[:, None] - np.arange(q)[None, :]) / (q - 1)
    w = np.eye(q) if weighting == "unweighted" else 1 - d ** (2 if weighting == "quadratic" else 1)
    if weighting == "custom":
        w = np.array(
            [[1, 0.1, 0.3, 0.2], [0.1, 1, 0.6, 0.3], [0.3, 0.6, 1, 0.7], [0.2, 0.3, 0.7, 1]]
        )
    x = ratings.to_numpy()
    p = np.array([(x == j).mean() for j in range(q)])
    pa = pair_agreement(x, w)
    pe = w.sum() * np.dot(p, 1 - p) / (q * (q - 1))  # kgwet/irrCAC original-author formula.
    result = oe.gwet_ac(
        ratings,
        list("abc"),
        categories=list(range(q)),
        agreement_weights=w if weighting == "custom" else weighting,
    )
    assert result.attrs["estimate"] == pytest.approx((pa - pe) / (1 - pe), abs=1e-12)
    assert result.attrs["state"]["diagnostics"]["chance_agreement"] == pytest.approx(pe)


def test_krippendorff_hayes_original_author_fixture():
    # Visually transcribed p1 of afhayes.com/public/kalpha.pdf. Printed alpha .7434,
    # 11 pairable units, 55 pairs and the observed/expected coincidence matrices.
    x = [
        [1, 1, None, 1],
        [2, 2, 3, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [4, 4, 4, 4],
        [1, 1, 2, 1],
        [2, 2, 2, 2],
        [None, 5, 5, 5],
        [None, None, 1, 1],
        [None, None, 3, None],
    ]
    frame = pd.DataFrame(x, columns=list("abcd"))
    r = oe.krippendorff_alpha(frame, list("abcd"), categories=[1, 2, 3, 4, 5])
    expected = np.array(
        [
            [7, 4 / 3, 1 / 3, 1 / 3, 0],
            [4 / 3, 10, 4 / 3, 1 / 3, 0],
            [1 / 3, 4 / 3, 8, 1 / 3, 0],
            [1 / 3, 1 / 3, 1 / 3, 4, 0],
            [0, 0, 0, 0, 3],
        ]
    )
    np.testing.assert_allclose(r["coincidences"], expected, atol=1e-12)
    assert r.attrs["estimate"] == pytest.approx(0.7434210526315789, abs=1e-12)
    assert r.attrs["n"] == 11 and r.attrs["n_missing"] == 1
    assert r.attrs["state"]["diagnostics"]["pairs"] == 55


@pytest.mark.parametrize("metric", ["nominal", "ordinal", "interval"])
def test_krippendorff_explicit_ordered_pair_oracle(ratings, metric):
    frame = ratings.copy().astype(float)
    frame.iloc[0, 1] = np.nan
    frame.iloc[1, 2] = np.nan
    q = 3
    coincidence = np.zeros((q, q))
    for row in frame.to_numpy():
        present = row[np.isfinite(row)].astype(int)
        for a, b in itertools.permutations(present, 2):
            coincidence[a, b] += 1 / (len(present) - 1)
    mass = coincidence.sum(1)
    expected = np.outer(mass, mass) - np.diag(mass)
    expected /= mass.sum() - 1
    centers = mass.cumsum() - mass / 2 if metric == "ordinal" else np.arange(q)
    delta = 1 - np.eye(q) if metric == "nominal" else (centers[:, None] - centers[None, :]) ** 2
    alpha = 1 - (coincidence * delta).sum() / (expected * delta).sum()
    r = oe.krippendorff_alpha(frame, list("abc"), categories=[0, 1, 2], metric=metric)
    assert r.attrs["estimate"] == pytest.approx(alpha, abs=1e-12)
    np.testing.assert_allclose(r["coincidences"], coincidence, atol=1e-12)


SF = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
    dtype=float,
)


def icc_oracle(x, model, average):
    # Independent projection-matrix ANOVA, distinct from implementation centering.
    n, k = x.shape
    js, jr = np.eye(n) - np.ones((n, n)) / n, np.eye(k) - np.ones((k, k)) / k
    between = np.linalg.norm(js @ x @ np.ones((k, 1)) / np.sqrt(k)) ** 2 / (n - 1)
    raters = np.linalg.norm(np.ones((1, n)) @ x @ jr / np.sqrt(n)) ** 2 / (k - 1)
    error = np.linalg.norm(js @ x @ jr) ** 2 / ((n - 1) * (k - 1))
    within = np.linalg.norm(x @ jr) ** 2 / (n * (k - 1))
    if model == "ICC1":
        return (between - within) / (between if average else between + (k - 1) * within)
    if model == "ICC2":
        return (between - error) / (
            between + (raters - error) / n
            if average
            else between + (k - 1) * error + k * (raters - error) / n
        )
    return (between - error) / (between if average else between + (k - 1) * error)


@pytest.mark.parametrize("model", ["ICC1", "ICC2", "ICC3"])
@pytest.mark.parametrize("average", [False, True])
def test_icc_schrout_fleiss_fixture_projection_and_jackknife(model, average):
    f = pd.DataFrame(SF, columns=list("abcd"))
    r = oe.icc(f, list("abcd"), model=model, average=average, inference="jackknife")
    assert r.attrs["estimate"] == pytest.approx(icc_oracle(SF, model, average), abs=1e-12)
    replicas = [icc_oracle(np.delete(SF, i, axis=0), model, average) for i in range(len(SF))]
    np.testing.assert_allclose(r["jackknife"].estimate, replicas, atol=1e-12)
    variance = (len(SF) - 1) * np.var(replicas, ddof=0)
    assert r["covariance"].iloc[0, 0] == pytest.approx(variance, abs=1e-12)
    row = r["estimate"].iloc[0]
    np.testing.assert_allclose(
        [row.ci_lower, row.ci_upper],
        row.estimate + np.array([-1, 1]) * stats.t.ppf(0.975, 5) * math.sqrt(variance),
        atol=1e-12,
    )
    assert row.p_value == pytest.approx(2 * stats.t.sf(abs(row.statistic), 5), abs=1e-12)
    assert oe.icc(f * 1e-14 + 1e-12, list("abcd"), model=model, average=average).attrs[
        "estimate"
    ] == pytest.approx(row.estimate, abs=2e-12)


@pytest.mark.parametrize("standardized", [False, True])
def test_omega_full_likelihood_oracle_and_unit_refits(standardized):
    rng = np.random.default_rng(85)
    x = (
        rng.normal(size=(36, 1)) * np.array([[0.8, 0.6, 0.75, 0.9]])
        + rng.normal(size=(36, 4)) * 0.9
    ) * [1, 2, 3, 4]

    def oracle(raw):
        cov = np.corrcoef(raw.T) if standardized else np.cov(raw.T)
        scale = np.diag(cov) ** 0.5
        target = cov / np.outer(scale, scale)

        # Unconcentrated Gaussian factor likelihood over loadings AND uniqueness.
        def objective(theta):
            loading, unique = theta[:4], np.exp(theta[4:])
            sigma = np.outer(loading, loading) + np.diag(unique)
            return np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, target))

        opt = optimize.minimize(
            objective,
            np.r_[np.repeat(0.6, 4), np.log(np.repeat(0.5, 4))],
            method="BFGS",
            options={"gtol": 1e-7, "maxiter": 1000},
        )
        loading, unique = opt.x[:4] * scale, np.exp(opt.x[4:]) * scale**2
        common = loading.sum() ** 2
        return common / (common + unique.sum()), np.outer(loading, loading) + np.diag(unique)

    f = pd.DataFrame(x, columns=list("abcd"))
    r = oe.omega_total(
        f, list("abcd"), standardized=standardized, inference="jackknife", max_work=1_000_000_000
    )
    value, sigma = oracle(x)
    assert r.attrs["estimate"] == pytest.approx(value, abs=1e-6)
    np.testing.assert_allclose(r["implied_covariance"], sigma, atol=2e-5)
    replicas = [oracle(np.delete(x, i, axis=0))[0] for i in range(len(x))]
    np.testing.assert_allclose(r["jackknife"].estimate, replicas, atol=2e-6)
    assert r["covariance"].iloc[0, 0] == pytest.approx((len(x) - 1) * np.var(replicas), rel=2e-5)


def ordinal_frame():
    rng = np.random.default_rng(128)
    z = rng.normal(size=(40, 2))
    z[:, 1] = 0.5 * z[:, 0] + math.sqrt(0.75) * z[:, 1]
    return pd.DataFrame(
        {
            "a": np.digitize(z[:, 0], [-0.6, 0.5]),
            "b": np.digitize(z[:, 1], [-0.6, 0.5]),
            "x": z[:, 0],
        }
    )


def ordinal_oracle(f, serial):
    q = 3
    counts = np.bincount(f.b, minlength=q)
    ty = np.r_[-np.inf, stats.norm.ppf(counts.cumsum()[:-1] / len(f)), np.inf]
    if serial:
        x = (f.x - f.x.mean()) / f.x.std(ddof=1)

        def loss(rho):
            a = (ty[f.b] - rho * x) / math.sqrt(1 - rho * rho)
            b = (ty[f.b + 1] - rho * x) / math.sqrt(1 - rho * rho)
            return -np.log(special.ndtr(b) - special.ndtr(a)).sum()
    else:
        count = (
            pd.crosstab(f.a, f.b).reindex(index=range(q), columns=range(q), fill_value=0).to_numpy()
        )
        tx = np.r_[-np.inf, stats.norm.ppf(count.sum(1).cumsum()[:-1] / len(f)), np.inf]

        def probabilities(rho):
            def cell(i, j):
                def integrand(x):
                    return stats.norm.pdf(x) * (
                        special.ndtr((ty[j + 1] - rho * x) / math.sqrt(1 - rho * rho))
                        - special.ndtr((ty[j] - rho * x) / math.sqrt(1 - rho * rho))
                    )

                return integrate.quad(integrand, tx[i], tx[i + 1], epsabs=1e-11, epsrel=1e-11)[0]

            return np.array([[cell(i, j) for j in range(q)] for i in range(q)])

        def loss(rho):
            return -np.sum(count * np.log(probabilities(rho)))

    opt = optimize.minimize_scalar(
        loss, bounds=(-0.9, 0.9), method="bounded", options={"xatol": 1e-10}
    )
    return opt.x


@pytest.mark.parametrize("serial", [False, True])
def test_latent_normal_against_independent_adaptive_integration(serial):
    f = ordinal_frame()
    call = (
        (lambda data, **kw: oe.polyserial(data, "x", "b", categories=[0, 1, 2], **kw))
        if serial
        else (
            lambda data, **kw: oe.polychoric(
                data, ["a", "b"], categories=[[0, 1, 2], [0, 1, 2]], **kw
            )
        )
    )
    r = call(f, inference="jackknife", max_work=20_000_000_000)
    assert r.attrs["estimate"] == pytest.approx(ordinal_oracle(f, serial), abs=1e-7)
    # Independent nuisance refits on selected leave-one units; all 40 persisted.
    for i in (0, 7, 18, 39):
        assert r["jackknife"].iloc[i, 0] == pytest.approx(
            ordinal_oracle(f.drop(index=i), serial), abs=2e-7
        )
    assert len(r["jackknife"]) == 40
    assert r.attrs["state"]["diagnostics"]["bracket_width"] < 1e-9


@pytest.mark.parametrize("name", ["cohen_kappa", "fleiss_kappa", "gwet_ac", "krippendorff_alpha"])
def test_agreement_uncertainty_persistence_missing_and_corruption(ratings, name, tmp_path):
    raters = ["a", "b"] if name == "cohen_kappa" else list("abc")

    def call(f, **kw):
        return getattr(oe, name)(f, raters, categories=[0, 1, 2], **kw)

    r = call(ratings, inference="jackknife")
    expected = [call(ratings.drop(index=i)).attrs["estimate"] for i in ratings.index]
    np.testing.assert_allclose(r["jackknife"].estimate, expected, atol=1e-12)
    assert r["covariance"].iloc[0, 0] == pytest.approx(
        (len(expected) - 1) * np.var(expected), abs=1e-12
    )
    path = tmp_path / f"{name}.json"
    artifact = oe.reliability_save(r, path)
    restored = oe.reliability_load(path)
    assert oe.reliability_save(restored) == artifact
    assert restored.attrs["unit_labels"] == list(ratings.index)
    assert "\\begin{tabular}" in restored.to_latex()
    changed = copy.deepcopy(artifact)
    changed["payload"]["tables"]["estimate"]["data"][0][0] += 0.1
    with pytest.raises(AnalysisError, match="checksum"):
        oe.reliability_load(changed)
    assert oe.reliability_save(r) == artifact  # artifact is detached, no mutable alias.
    hole = ratings.copy().astype(float)
    hole.iloc[0, 0] = np.nan
    if name != "krippendorff_alpha":
        assert call(hole).attrs["positions"] == list(range(1, len(hole)))
    with pytest.raises(AnalysisError) as err:
        call(hole, missing="raise")
    assert err.value.code == "missing_values"


@pytest.mark.parametrize(
    "name",
    [
        "icc",
        "omega_total",
        "cohen_kappa",
        "fleiss_kappa",
        "gwet_ac",
        "krippendorff_alpha",
        "polychoric",
        "polyserial",
    ],
)
def test_explicit_device_weights_budget_and_category_domain(name, ratings):
    def call(**kw):
        if name == "icc":
            return oe.icc(ratings, list("abc"), **kw)
        if name == "omega_total":
            return oe.omega_total(ratings, list("abc"), **kw)
        if name == "polychoric":
            return oe.polychoric(ratings, ["a", "b"], categories=[[0, 1, 2], [0, 1, 2]], **kw)
        if name == "polyserial":
            return oe.polyserial(ratings, "a", "b", categories=[0, 1, 2], **kw)
        return getattr(oe, name)(
            ratings,
            ["a", "b"] if name == "cohen_kappa" else list("abc"),
            categories=[0, 1, 2],
            **kw,
        )

    for option, code in [
        ({"device": "cuda"}, "unsupported_device"),
        ({"weights": "w"}, "unsupported_weights"),
        ({"max_work": 1}, "work_budget_exceeded"),
        ({"inference": "jackknife", "max_fits": 2}, "work_budget_exceeded"),
        ({"level": 1}, "invalid_option"),
    ]:
        with pytest.raises(AnalysisError) as err:
            call(**option)
        assert err.value.code == code
    with use_workspace_budget(1):
        large = pd.concat([ratings] * 1000, ignore_index=True)
        with pytest.raises(AnalysisError) as err:
            oe.icc(large, list("abc"))
        assert err.value.code == "workspace_limit"


def test_empty_unknown_degenerate_boundary_and_failed_replicate(ratings):
    with pytest.raises(AnalysisError) as err:
        oe.polychoric(ratings, ["a", "b"], categories=[[0, 1, 2, 3], [0, 1, 2]])
    assert err.value.code == "empty_category"
    with pytest.raises(AnalysisError) as err:
        oe.cohen_kappa(ratings, ["a", "b"], categories=[0, 1])
    assert err.value.code == "unknown_category"
    f = pd.DataFrame({"a": [0, 0, 1, 1] * 3, "b": [0, 0, 1, 1] * 3})
    with pytest.raises(AnalysisError) as err:
        oe.polychoric(f, ["a", "b"], categories=[[0, 1], [0, 1]])
    assert err.value.code == "correlation_boundary"
    assert oe.cohen_kappa(f, ["a", "b"], categories=[0, 1]).attrs["estimate"] == 1
    with pytest.raises(AnalysisError) as err:
        oe.cohen_kappa(f, ["a", "b"], categories=[0, 1], inference="jackknife")
    assert err.value.code == "degenerate_inference"
    f = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 3]})
    with pytest.raises(AnalysisError) as err:
        oe.icc(f, ["a", "b"], inference="jackknife")
    assert err.value.code == "replicate_failure"


def test_saved_state_guard_and_json_finite(ratings):
    r = oe.gwet_ac(ratings, list("abc"), categories=[0, 1, 2])
    json.dumps(oe.reliability_save(r), allow_nan=False)
    r.attrs["state"]["estimate"] += 0.1
    with pytest.raises(AnalysisError) as err:
        oe.reliability_save(r)
    assert err.value.code == "invalid_result"
